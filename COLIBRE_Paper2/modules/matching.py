#!/usr/bin/env python3
"""COLIBRE Paper 2: match compact/quenched high-z analogues to z=0 SRGs.

Run from COLIBRE_Paper2:
    python3 -m modules.matching --dry-run --limit 20
    python3 -m modules.matching --dry-run
    python3 -m modules.matching --write
    python3 -m modules.matching --write --overwrite

These are *candidate analogues*, NOT verified non-relic descendants.
Uses only the ASCII catalogue at each target's exact observed z_match.
Does not perform merger-tree traversal or infer descendant fates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import FORMATION, INPUTS, MATCHING, OUTPUT, REFERENCE_SAMPLE, UNITS
from modules.data_io import discover_ascii_catalogues, read_ascii_snapshot
from modules.formation import _cosmology, settings as formation_settings


def project_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(__file__).resolve().parents[1] / path).resolve()


def settings():
    cfg = dict(MATCHING)
    required = ('n_controls_per_srg', 'mass_tolerance_dex', 'radius_tolerance_dex',
                'with_replacement', 'require_compact', 'require_quenched',
                'apply_mass_cut', 'minimum_matching_redshift', 'exclude_reference_srgs')
    missing = set(required) - cfg.keys()
    if missing:
        raise ValueError(f'MATCHING missing keys: {sorted(missing)}')
    if int(cfg['n_controls_per_srg']) < 1:
        raise ValueError('n_controls_per_srg must be >= 1')
    for key in ('mass_tolerance_dex', 'radius_tolerance_dex'):
        if not np.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f'{key} must be finite and > 0')
    if float(cfg['minimum_matching_redshift']) < 0:
        raise ValueError('minimum_matching_redshift must be >= 0')
    if not cfg['require_compact'] or not cfg['require_quenched']:
        raise ValueError('This initial implementation requires both compact and quenched candidates')
    if bool(cfg['apply_mass_cut']) != bool(FORMATION['apply_mass_cut']):
        raise ValueError('MATCHING.apply_mass_cut must agree with FORMATION.apply_mass_cut')
    return cfg


def load_targets(path=None):
    path = project_path(path or Path(OUTPUT['root']) / 'formation' / 'red_nugget_matching_epochs.csv')
    if not path.is_file():
        raise FileNotFoundError(f'Missing matching epochs: {path}. Run modules.formation --write first.')
    df = pd.read_csv(path, dtype={'track_id': 'int64'})
    required = {'track_id', 'population', 'red_nugget_detected', 'z_match',
                'log10_stellar_mass_match', 'log10_rhalf_match', 'compactness_match', 'ssfr_match'}
    missing = required - set(df.columns)
    if missing or df['track_id'].duplicated().any():
        raise ValueError(f'Invalid matching-epoch catalogue; missing={sorted(missing)}, duplicate IDs={df.track_id.duplicated().sum()}')
    return df


def load_reference_srgs(path=None):
    path = project_path(path or Path(OUTPUT['reference_dir']) / 'reference_galaxies_z0_all.csv')
    if not path.is_file():
        raise FileNotFoundError(f'Missing reference catalogue: {path}')
    ref = pd.read_csv(path, usecols=['track_id', 'population'], dtype={'track_id': 'int64'})
    if ref['track_id'].duplicated().any():
        raise ValueError('Duplicate z=0 reference TrackIDs')
    return set(ref.loc[ref['population'] == 'SRG', 'track_id'].astype('int64'))


def load_epoch_pool(path, redshift, fcfg, cosmology):
    """Read a single ASCII epoch, applying only independently checked ASCII units."""
    for key in ('ascii_stellar_mass_to_msun', 'ascii_radius_to_physical_kpc',
                'ascii_sfr_to_msun_per_year'):
        if UNITS.get(key) is None or not np.isfinite(float(UNITS[key])) or float(UNITS[key]) <= 0:
            raise ValueError(f'Unverified/missing unit factor: UNITS[{key!r}]')
    raw = read_ascii_snapshot(path, columns=['track_id', 'stellar_mass', 'rhalf_stars', 'sfr_instant'])
    mass = pd.to_numeric(raw['stellar_mass'], errors='coerce').to_numpy(float) * float(UNITS['ascii_stellar_mass_to_msun'])
    radius = pd.to_numeric(raw['rhalf_stars'], errors='coerce').to_numpy(float) * float(UNITS['ascii_radius_to_physical_kpc'])
    sfr = pd.to_numeric(raw['sfr_instant'], errors='coerce').to_numpy(float) * float(UNITS['ascii_sfr_to_msun_per_year'])
    valid_mr = np.isfinite(mass) & (mass > 0) & np.isfinite(radius) & (radius > 0)
    valid_sfr = np.isfinite(sfr) & (sfr >= 0)
    valid = valid_mr & valid_sfr
    logm = np.full(len(raw), np.nan)
    logr = np.full(len(raw), np.nan)
    ssfr = np.full(len(raw), np.nan)
    logm[valid_mr] = np.log10(mass[valid_mr])
    logr[valid_mr] = np.log10(radius[valid_mr])
    ssfr[valid] = sfr[valid] / mass[valid]
    compact = logm - float(REFERENCE_SAMPLE['compactness_exponent']) * logr
    threshold = float(fcfg['quenching_factor']) / (float(cosmology.age(float(redshift)).value) * 1e9)
    keep = valid & (compact >= float(fcfg['compactness_threshold'])) & (ssfr < threshold)
    if fcfg['apply_mass_cut']:
        keep &= mass > float(fcfg['minimum_stellar_mass_msun'])
    pool = pd.DataFrame({'track_id': raw['track_id'].to_numpy(dtype=np.int64),
                         'log10_stellar_mass': logm, 'log10_rhalf': logr,
                         'compactness': compact, 'ssfr': ssfr,
                         'stellar_mass_msun': mass, 'rhalf_physical_kpc': radius})
    return pool.loc[keep].reset_index(drop=True), {'n_ascii_rows': len(raw), 'n_eligible': int(keep.sum()),
                                                     'quenching_threshold_per_year': threshold}


def match_at_epoch(targets, pool, cfg, excluded_ids, redshift):
    """Greedy nearest neighbours; replacement means reuse across *targets*."""
    selected = []
    diagnostics = []
    used = set()
    pool = pool.copy()
    if cfg['exclude_reference_srgs']:
        pool = pool.loc[~pool['track_id'].isin(excluded_ids)].copy()
    # Exclude ALL target IDs, even when known-SRG exclusion is disabled, per target below.
    for row in targets.sort_values('track_id').itertuples(index=False):
        tid = int(row.track_id)
        dm = pool['log10_stellar_mass'].to_numpy(float) - float(row.log10_stellar_mass_match)
        dr = pool['log10_rhalf'].to_numpy(float) - float(row.log10_rhalf_match)
        eligible = (np.abs(dm) <= float(cfg['mass_tolerance_dex'])) & (np.abs(dr) <= float(cfg['radius_tolerance_dex']))
        eligible &= pool['track_id'].to_numpy(np.int64) != tid
        if not cfg['with_replacement']:
            eligible &= ~pool['track_id'].isin(used).to_numpy()
        distances = np.hypot(dm / float(cfg['mass_tolerance_dex']), dr / float(cfg['radius_tolerance_dex']))
        candidate_idx = np.flatnonzero(eligible)
        # Stable deterministic ties: distance then TrackID.
        ranked = sorted(candidate_idx, key=lambda j: (float(distances[j]), int(pool.iloc[j]['track_id'])))
        chosen = ranked[:int(cfg['n_controls_per_srg'])]
        diagnostics.append({'reference_track_id': tid, 'population': str(row.population),
                            'z_match': float(redshift), 'n_available_candidates': len(ranked),
                            'n_selected': len(chosen), 'n_requested': int(cfg['n_controls_per_srg']),
                            'matching_complete': len(chosen) == int(cfg['n_controls_per_srg']),
                            'nearest_distance': float(distances[ranked[0]]) if ranked else np.nan,
                            'farthest_selected_distance': float(distances[chosen[-1]]) if chosen else np.nan})
        for rank, j in enumerate(chosen, 1):
            p = pool.iloc[j]
            control_id = int(p['track_id'])
            selected.append({'reference_track_id': tid, 'reference_population': str(row.population),
                             'z_match': float(redshift), 'control_track_id': control_id,
                             'rank': rank, 'distance': float(distances[j]),
                             'delta_log10_stellar_mass': float(dm[j]), 'delta_log10_rhalf': float(dr[j]),
                             'reference_log10_stellar_mass': float(row.log10_stellar_mass_match),
                             'reference_log10_rhalf': float(row.log10_rhalf_match),
                             'reference_compactness': float(row.compactness_match),
                             'reference_ssfr': float(row.ssfr_match),
                             'control_log10_stellar_mass': float(p['log10_stellar_mass']),
                             'control_log10_rhalf': float(p['log10_rhalf']),
                             'control_compactness': float(p['compactness']),
                             'control_ssfr': float(p['ssfr']),
                             'control_stellar_mass_msun': float(p['stellar_mass_msun']),
                             'control_rhalf_physical_kpc': float(p['rhalf_physical_kpc']),
                             'descendant_status': 'NOT_EVALUATED'})
            if not cfg['with_replacement']:
                used.add(control_id)
    return selected, diagnostics


def build_matches(targets, inventory, cfg=None, fcfg=None, excluded_ids=None, pool_loader=None):
    cfg = settings() if cfg is None else cfg
    fcfg = formation_settings() if fcfg is None else fcfg
    excluded_ids = set() if excluded_ids is None else set(excluded_ids)
    cosmology = _cosmology(fcfg['cosmology'])
    # Exact numeric redshift labels, not nearest-neighbour snapshot guesses.
    path_by_z = {round(float(r.redshift), 8): Path(r.path) for r in inventory.itertuples(index=False)}
    detected = targets['red_nugget_detected'].astype('boolean').fillna(False)
    eligible = targets.loc[detected & targets['z_match'].notna()].copy()
    skipped_low_z = eligible['z_match'] < float(cfg['minimum_matching_redshift'])
    eligible = eligible.loc[~skipped_low_z].copy()
    selected, diag, epochs = [], [], []
    for z, group in eligible.groupby('z_match', sort=True):
        key = round(float(z), 8)
        if key not in path_by_z:
            raise FileNotFoundError(f'No ASCII catalogue at exact z_match={z}; cannot substitute another epoch')
        if pool_loader is None:
            pool, info = load_epoch_pool(path_by_z[key], z, fcfg, cosmology)
        else:
            pool, info = pool_loader(path_by_z[key], z, fcfg, cosmology)
        rows, diagnostics = match_at_epoch(group, pool, cfg, excluded_ids, z)
        selected.extend(rows)
        diag.extend(diagnostics)
        epochs.append({'redshift': float(z), 'n_targets': len(group), **info})
        print(f'z={z:5.2f}: {len(group):3d} targets, {len(pool):6d} compact+quenched candidates, {len(rows):4d} matches', flush=True)
    matches = pd.DataFrame(selected)
    diagnostics = pd.DataFrame(diag)
    epochs = pd.DataFrame(epochs)
    n_reused = int(matches['control_track_id'].duplicated().sum()) if not matches.empty else 0
    report = {'status': 'candidate_analogues_only', 'n_reference_rows': len(targets),
              'n_detected': int(detected.sum()), 'n_skipped_no_epoch': int((~detected | targets['z_match'].isna()).sum()),
              'n_skipped_low_redshift': int(skipped_low_z.sum()), 'n_targets_attempted': len(eligible),
              'n_targets_with_five_or_requested': int(diagnostics['matching_complete'].sum()) if not diagnostics.empty else 0,
              'n_targets_unmatched': int((diagnostics['n_selected'] == 0).sum()) if not diagnostics.empty else 0,
              'n_selected_pairs': len(matches), 'n_unique_control_trackids': int(matches['control_track_id'].nunique()) if not matches.empty else 0,
              'n_repeated_assignments': n_reused, 'n_matching_epochs': len(epochs),
              'matching_configuration': cfg, 'formation_configuration': fcfg,
              'note': 'Controls have NOT been verified as non-relic descendants. HBT TrackID persistence and mergers require separate descendant analysis.'}
    return matches, diagnostics, epochs, report


def save_outputs(matches, diagnostics, epochs, report, output_dir=None, overwrite=False):
    out = project_path(output_dir or Path(OUTPUT['root']) / 'controls')
    paths = [out / 'matched_control_candidates.csv', out / 'matching_diagnostics.csv',
             out / 'matching_epoch_pools.csv', out / 'matching_report.json']
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Output(s) exist; use --write --overwrite: {existing}')
    out.mkdir(parents=True, exist_ok=True)
    matches.to_csv(paths[0], index=False)
    diagnostics.to_csv(paths[1], index=False)
    epochs.to_csv(paths[2], index=False)
    paths[3].write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--write', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='First N reference TrackIDs for testing')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.overwrite and not args.write:
        parser.error('--overwrite requires --write')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be >= 1')
    cfg = settings()
    fcfg = formation_settings()
    targets = load_targets()
    if args.limit is not None:
        targets = targets.sort_values('track_id').head(args.limit).copy()
    srgs = load_reference_srgs() if cfg['exclude_reference_srgs'] else set()
    inventory = discover_ascii_catalogues()
    if inventory.empty:
        raise FileNotFoundError('No ASCII catalogues found')
    matches, diagnostics, epochs, report = build_matches(targets, inventory, cfg, fcfg, srgs)
    print('\nMatching report:\n' + json.dumps(report, indent=2))
    if args.write:
        for path in save_outputs(matches, diagnostics, epochs, report, args.output_dir, args.overwrite):
            print('Saved:', path)
    else:
        print('Dry run: no files written.')


if __name__ == '__main__':
    main()
