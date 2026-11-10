#!/usr/bin/env python3
"""COLIBRE Paper 2: identify compact/quenched epochs of z=0-selected galaxies.

Run from COLIBRE_Paper2:
    python3 -m modules.formation --dry-run --limit 20
    python3 -m modules.formation --dry-run
    python3 -m modules.formation --write

This module only analyses already-calibrated evolutionary_properties.csv.
It does not select control galaxies, infer continuous onset times, or rewrite
z=0 SOAP-based classifications.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT, REFERENCE_SAMPLE

try:
    from config import FORMATION
except ImportError:
    FORMATION = None


DEFAULT_FORMATION = {
    'compactness_threshold': 9.75,
    'compactness_comparison': '>=',
    'quenching_method': 'age_of_universe',
    'quenching_factor': 0.2,
    'quenching_comparison': '<',
    'apply_mass_cut': False,
    'minimum_stellar_mass_msun': 1.0e10,
    'matching_epoch_definition': 'first_compact_and_quenched',
    'red_nugget_min_redshift': 2.0,
    'cosmology': 'Planck13',
}


def project_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(__file__).resolve().parents[1] / path).resolve()


def settings():
    if FORMATION is None:
        raise RuntimeError(
            'Add the FORMATION dictionary to config.py before running formation.py. '
            'See the accompanying instructions.'
        )
    cfg = {**DEFAULT_FORMATION, **FORMATION}
    if cfg['quenching_method'] != 'age_of_universe':
        raise ValueError('Only age_of_universe quenching is currently implemented')
    if cfg['matching_epoch_definition'] != 'first_compact_and_quenched':
        raise ValueError('Unsupported matching epoch definition')
    if cfg['compactness_comparison'] != '>=' or cfg['quenching_comparison'] != '<':
        raise ValueError('Expected compactness >= cut and sSFR < threshold')
    if not np.isfinite(float(cfg['quenching_factor'])) or float(cfg['quenching_factor']) <= 0:
        raise ValueError('quenching_factor must be positive and finite')
    if not np.isfinite(float(cfg['compactness_threshold'])):
        raise ValueError('compactness_threshold must be finite')
    if float(cfg['minimum_stellar_mass_msun']) <= 0:
        raise ValueError('minimum_stellar_mass_msun must be positive')
    if float(cfg['red_nugget_min_redshift']) < 0:
        raise ValueError('red_nugget_min_redshift must be nonnegative')
    return cfg


def load_history(path=None):
    path = project_path(path or Path(OUTPUT['tracks_dir']) / 'evolutionary_properties.csv')
    if not path.is_file():
        raise FileNotFoundError(f'Evolutionary properties missing: {path}')
    frame = pd.read_csv(path, dtype={'track_id': 'int64'})
    return frame


def _cosmology(name):
    try:
        from astropy import cosmology as cosmo
    except ImportError as exc:
        raise ImportError('formation.py requires astropy (pip/conda install astropy)') from exc
    obj = getattr(cosmo, str(name), None)
    if obj is None or not hasattr(obj, 'age'):
        raise ValueError(f'Unknown astropy cosmology: {name!r}')
    return obj


def _first_true(g, column):
    subset = g.loc[g[column].fillna(False)]
    return subset.iloc[0] if len(subset) else None


def classify_snapshots(history, cfg=None):
    """Return one row per observed galaxy/epoch with tri-state physical flags.

    Flags use pandas nullable booleans: <NA> means insufficient measurement,
    not False. Quenching is evaluated using physical (unfloored) sSFR.
    """
    cfg = settings() if cfg is None else cfg
    required = {'track_id', 'population', 'redshift', 'stellar_mass_msun',
                'rhalf_physical_kpc', 'sfr_msun_per_year'}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f'Missing required physical columns: {sorted(missing)}')
    if history.empty:
        raise ValueError('Empty evolutionary catalogue')
    h = history.copy()
    h['redshift'] = pd.to_numeric(h['redshift'], errors='coerce')
    if h['redshift'].isna().any() or (h['redshift'] < 0).any():
        raise ValueError('Invalid redshift in evolutionary catalogue')
    if h.duplicated(['track_id', 'redshift']).any():
        raise ValueError('Duplicate (track_id, redshift) entries')
    if h.groupby('track_id')['population'].nunique(dropna=False).gt(1).any():
        raise ValueError('Population changes within a TrackID history')
    h = h.sort_values(['track_id', 'redshift'], ascending=[True, False]).reset_index(drop=True)
    cosmo = _cosmology(cfg['cosmology'])
    z = h['redshift'].to_numpy(dtype=float)
    h['universe_age_gyr'] = cosmo.age(z).value
    h['ssfr_quenching_threshold'] = float(cfg['quenching_factor']) / (h['universe_age_gyr'] * 1e9)

    mass = pd.to_numeric(h['stellar_mass_msun'], errors='coerce').to_numpy(dtype=float)
    radius = pd.to_numeric(h['rhalf_physical_kpc'], errors='coerce').to_numpy(dtype=float)
    sfr = pd.to_numeric(h['sfr_msun_per_year'], errors='coerce').to_numpy(dtype=float)
    good_mr = np.isfinite(mass) & (mass > 0) & np.isfinite(radius) & (radius > 0)
    good_sfr = good_mr & np.isfinite(sfr) & (sfr >= 0)
    logm = np.full(len(h), np.nan)
    logr = np.full(len(h), np.nan)
    logm[good_mr] = np.log10(mass[good_mr])
    logr[good_mr] = np.log10(radius[good_mr])
    exponent = float(REFERENCE_SAMPLE['compactness_exponent'])
    compactness = logm - exponent * logr
    h['compactness_sigma_1p5'] = compactness
    h['log10_stellar_mass'] = logm
    h['log10_rhalf'] = logr
    ssfr = np.full(len(h), np.nan)
    ssfr[good_sfr] = sfr[good_sfr] / mass[good_sfr]
    h['ssfr_instant'] = ssfr

    def flag(known, yes):
        result = pd.Series(pd.NA, index=h.index, dtype='boolean')
        result.loc[known] = yes[known]
        return result

    h['is_compact'] = flag(good_mr, compactness >= float(cfg['compactness_threshold']))
    h['is_quenched'] = flag(good_sfr, ssfr < h['ssfr_quenching_threshold'].to_numpy(dtype=float))
    h['passes_mass_cut'] = flag(np.isfinite(mass) & (mass > 0), mass > float(cfg['minimum_stellar_mass_msun']))
    # Boolean Kleene logic retains unknown where the outcome cannot be established.
    h['is_compact_quenched'] = h['is_compact'] & h['is_quenched']
    h['is_red_nugget'] = h['is_compact_quenched'] & h['passes_mass_cut'] if cfg['apply_mass_cut'] else h['is_compact_quenched']
    h['is_highz_red_nugget'] = h['is_red_nugget'] & (h['redshift'] >= float(cfg['red_nugget_min_redshift']))
    return h


def summarise_matching_epochs(classified, cfg=None):
    """First observed compact+quenched epoch, ordered early to late (high z to low z)."""
    cfg = settings() if cfg is None else cfg
    rows = []
    for tid, g in classified.groupby('track_id', sort=True):
        g = g.sort_values('redshift', ascending=False).reset_index(drop=True)
        first = _first_true(g, 'is_red_nugget')
        first_compact = _first_true(g, 'is_compact')
        first_quenched = _first_true(g, 'is_quenched')
        detected = first is not None
        idx = int(first.name) if detected else None
        # At first positive observation, a preceding negative observation brackets
        # the onset. A preceding unknown cannot establish a transition.
        previous = g.iloc[idx - 1] if detected and idx > 0 else None
        prev_state = previous['is_red_nugget'] if previous is not None else pd.NA
        onset_bracketed = bool(previous is not None and pd.notna(prev_state) and not bool(prev_state))
        onset_uncertain = bool(detected and not onset_bracketed)
        n_positive = int(g['is_red_nugget'].fillna(False).sum())
        n_unknown = int(g['is_red_nugget'].isna().sum())
        rows.append({
            'track_id': int(tid),
            'population': str(g.iloc[0]['population']),
            'red_nugget_detected': detected,
            'z_match': float(first['redshift']) if detected else np.nan,
            'stellar_mass_match': float(first['stellar_mass_msun']) if detected else np.nan,
            'rhalf_match': float(first['rhalf_physical_kpc']) if detected else np.nan,
            'log10_stellar_mass_match': float(first['log10_stellar_mass']) if detected else np.nan,
            'log10_rhalf_match': float(first['log10_rhalf']) if detected else np.nan,
            'compactness_match': float(first['compactness_sigma_1p5']) if detected else np.nan,
            'ssfr_match': float(first['ssfr_instant']) if detected else np.nan,
            'ssfr_quenching_threshold_match': float(first['ssfr_quenching_threshold']) if detected else np.nan,
            'universe_age_gyr_match': float(first['universe_age_gyr']) if detected else np.nan,
            'z_first_compact': float(first_compact['redshift']) if first_compact is not None else np.nan,
            'z_first_quenched': float(first_quenched['redshift']) if first_quenched is not None else np.nan,
            'n_red_nugget_snapshots': n_positive,
            'n_unknown_snapshots': n_unknown,
            'n_observed_snapshots': len(g),
            'onset_bracketed': onset_bracketed,
            'onset_uncertain': onset_uncertain,
            'onset_prior_z': float(previous['redshift']) if onset_bracketed else np.nan,
            'highz_red_nugget': bool(g['is_highz_red_nugget'].fillna(False).any()),
            'earliest_observed_z': float(g.iloc[0]['redshift']),
            'latest_observed_z': float(g.iloc[-1]['redshift']),
        })
    return pd.DataFrame(rows)


def build_formation_catalogues(history, cfg=None):
    cfg = settings() if cfg is None else cfg
    classified = classify_snapshots(history, cfg)
    summary = summarise_matching_epochs(classified, cfg)
    report = {
        'n_reference_galaxies': int(summary['track_id'].nunique()),
        'n_observed_rows': int(len(classified)),
        'n_red_nugget_detected': int(summary['red_nugget_detected'].sum()),
        'n_never_detected': int((~summary['red_nugget_detected']).sum()),
        'n_highz_red_nugget': int(summary['highz_red_nugget'].sum()),
        'n_onset_uncertain': int(summary['onset_uncertain'].sum()),
        'n_unknown_classification_rows': int(classified['is_red_nugget'].isna().sum()),
        'n_match_epochs': int(summary['z_match'].nunique()),
        'configuration': cfg,
        'note': ('z_match is the earliest OBSERVED epoch satisfying the criteria, '
                 'not a continuous formation time. No control matching or descendant '
                 'analysis is performed. Missing measurements are not classified as star forming.'),
    }
    return classified, summary, report


def save_outputs(classified, summary, report, directory=None, overwrite=False):
    directory = project_path(directory or Path(OUTPUT['root']) / 'formation')
    files = (directory / 'red_nugget_snapshots.csv',
             directory / 'red_nugget_matching_epochs.csv',
             directory / 'formation_report.json')
    existing = [str(p) for p in files if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Output(s) already exist; use --write --overwrite: {existing}')
    directory.mkdir(parents=True, exist_ok=True)
    classified.to_csv(files[0], index=False)
    summary.to_csv(files[1], index=False)
    files[2].write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--write', action='store_true', help='Save catalogues and report')
    action.add_argument('--dry-run', action='store_true', help='Validate and report without writing (default)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs (requires --write)')
    parser.add_argument('--limit', type=int, help='First N TrackIDs for development only')
    parser.add_argument('--input-file', type=Path, help='Override evolutionary_properties.csv path')
    parser.add_argument('--output-dir', type=Path, help='Override formation output directory')
    args = parser.parse_args()
    if args.overwrite and not args.write:
        parser.error('--overwrite requires --write')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.limit is not None and args.write:
        parser.error('Refusing to write a limited development sample as production output')
    cfg = settings()
    history = load_history(args.input_file)
    if args.limit is not None:
        ids = np.sort(history['track_id'].unique())[:args.limit]
        history = history.loc[history['track_id'].isin(ids)].copy()
    classified, summary, report = build_formation_catalogues(history, cfg)
    print('Formation report:\n' + json.dumps(report, indent=2))
    if args.write:
        for path in save_outputs(classified, summary, report, args.output_dir, args.overwrite):
            print('Saved:', path)
    else:
        print('Dry run: no files written.')


if __name__ == '__main__':
    main()
