#!/usr/bin/env python3
"""COLIBRE Paper 2: matched-epoch initial-condition diagnostics.

Run from COLIBRE_Paper2:
  python3 -m modules.control_analysis --dry-run --limit 20
  python3 -m modules.control_analysis --dry-run
  python3 -m modules.control_analysis --write
  python3 -m modules.control_analysis --write --overwrite

Compares SRG progenitors with matched high-z analogues, NOT confirmed
non-relic descendants. Reads only required columns from each ASCII epoch.
All non-calibrated quantities are retained in native ASCII units; no claims
about their physical units are made. No merger-tree or causal inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import INPUTS, OUTPUT
from modules.data_io import discover_ascii_catalogues, read_ascii_snapshot

RAW_PROPERTIES = [
    'bh_mass', 'bh_accretion_rate', 'bh_thermal_energy_cum', 'n_agn_events',
    'is_central', 'halo_mass_m200crit', 'kappa_co_stars',
    'disc_to_total_stars', 'stellar_angular_momentum', 'stellar_age',
]
# Positive, continuous variables: use log ratios only if BOTH are positive.
LOG_RATIO_PROPERTIES = ['bh_mass', 'halo_mass_m200crit']
DIFFERENCE_PROPERTIES = [
    'bh_accretion_rate', 'bh_thermal_energy_cum', 'n_agn_events',
    'kappa_co_stars', 'disc_to_total_stars',
    'stellar_angular_momentum', 'stellar_age',
]


def project_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(__file__).resolve().parents[1] / path).resolve()


def load_pairs(path, limit=None):
    df = pd.read_csv(path, dtype={'reference_track_id': 'int64', 'control_track_id': 'int64'})
    needed = {'reference_track_id', 'control_track_id', 'z_match',
              'reference_log10_stellar_mass', 'control_log10_stellar_mass',
              'reference_log10_rhalf', 'control_log10_rhalf'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Matching catalogue missing: {sorted(missing)}')
    if df.empty:
        raise ValueError('Matching catalogue contains no pairs')
    if df[['reference_track_id', 'control_track_id', 'z_match']].isna().any().any():
        raise ValueError('Null IDs or matching redshifts')
    if limit is not None:
        ids = sorted(df.reference_track_id.unique())[:limit]
        df = df.loc[df.reference_track_id.isin(ids)].copy()
    if (df.reference_track_id == df.control_track_id).any():
        raise ValueError('A reference galaxy was matched to itself')
    return df.reset_index(drop=True)


def verify_exclusion(pairs, reference_path):
    ref = pd.read_csv(reference_path, usecols=['track_id', 'population'], dtype={'track_id': 'int64'})
    srg_ids = set(ref.loc[ref.population.eq('SRG'), 'track_id'])
    overlap = srg_ids.intersection(set(pairs.control_track_id))
    if overlap:
        raise ValueError(f'Known SRGs selected as controls: {sorted(overlap)[:12]}')
    return len(srg_ids)


def assemble_pairs(pairs, inventory):
    lookup = {round(float(row.redshift), 8): Path(row.path)
              for row in inventory.itertuples(index=False)}
    outputs = []
    epoch_stats = []
    for z, group in pairs.groupby('z_match', sort=True):
        key = round(float(z), 8)
        if key not in lookup:
            raise FileNotFoundError(f'No ASCII file at exact matching redshift {z}')
        ids = set(group.reference_track_id).union(set(group.control_track_id))
        raw = read_ascii_snapshot(lookup[key], columns=['track_id'] + RAW_PROPERTIES,
                                  track_ids=ids)
        if raw.track_id.duplicated().any():
            raise ValueError(f'Duplicate TrackIDs at z={z}')
        raw = raw.set_index('track_id')
        missing = ids - set(raw.index)
        if missing:
            raise ValueError(f'z={z}: {len(missing)} matched objects missing from ASCII; examples {sorted(missing)[:8]}')
        ref = raw.loc[group.reference_track_id.to_numpy()].reset_index(drop=True)
        ctrl = raw.loc[group.control_track_id.to_numpy()].reset_index(drop=True)
        chunk = group.reset_index(drop=True).copy()
        for prop in RAW_PROPERTIES:
            chunk['reference_' + prop + '_raw'] = pd.to_numeric(ref[prop], errors='coerce').to_numpy()
            chunk['control_' + prop + '_raw'] = pd.to_numeric(ctrl[prop], errors='coerce').to_numpy()
        outputs.append(chunk)
        epoch_stats.append({'redshift': float(z), 'n_pairs': len(group),
                            'n_reference': group.reference_track_id.nunique(),
                            'n_unique_controls': group.control_track_id.nunique(),
                            'n_ids_read': len(raw)})
        print(f'z={z:5.2f}: {len(group):4d} pairs, {len(raw):4d} distinct galaxies', flush=True)
    out = pd.concat(outputs, ignore_index=True)
    # Structural differences should reproduce the matching catalogue exactly.
    out['delta_log10_stellar_mass_check'] = (out.control_log10_stellar_mass - out.reference_log10_stellar_mass)
    out['delta_log10_rhalf_check'] = (out.control_log10_rhalf - out.reference_log10_rhalf)
    for col, check in [('delta_log10_stellar_mass', 'delta_log10_stellar_mass_check'),
                       ('delta_log10_rhalf', 'delta_log10_rhalf_check')]:
        if col in out and not np.allclose(out[col], out[check], atol=1e-8, rtol=0):
            raise ValueError(f'Matching differences inconsistent: {col}')
    for prop in LOG_RATIO_PROPERTIES:
        a = out['reference_' + prop + '_raw'].to_numpy(float)
        b = out['control_' + prop + '_raw'].to_numpy(float)
        delta = np.full(len(out), np.nan)
        valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        delta[valid] = np.log10(b[valid] / a[valid])
        out['delta_log10_' + prop + '_raw'] = delta
    for prop in DIFFERENCE_PROPERTIES:
        out['delta_' + prop + '_raw'] = (out['control_' + prop + '_raw'] - out['reference_' + prop + '_raw'])
    # Central flag is a category, not a continuous quantity.
    a = out['reference_is_central_raw'].to_numpy(float)
    b = out['control_is_central_raw'].to_numpy(float)
    valid = np.isin(a, [0, 1]) & np.isin(b, [0, 1])
    out['central_status_same'] = pd.array(np.where(valid, a == b, False), dtype='boolean')
    out.loc[~valid, 'central_status_same'] = pd.NA
    return out, pd.DataFrame(epoch_stats)


def summarize(pairs):
    # Per-reference median first: one SRG is one observational/statistical unit.
    cols = ['delta_log10_stellar_mass_check', 'delta_log10_rhalf_check']
    cols += ['delta_log10_' + p + '_raw' for p in LOG_RATIO_PROPERTIES]
    cols += ['delta_' + p + '_raw' for p in DIFFERENCE_PROPERTIES]
    per_reference = (pairs.groupby('reference_track_id', as_index=False)[cols]
                     .median(numeric_only=True))
    ncontrols = pairs.groupby('reference_track_id').size().rename('n_controls')
    z = pairs.groupby('reference_track_id').z_match.first()
    per_reference = per_reference.join(ncontrols, on='reference_track_id').join(z, on='reference_track_id')
    stats = []
    for col in cols:
        vals = per_reference[col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        stats.append({'quantity': col, 'n_reference_with_valid_pairs': len(vals),
                      'median_of_reference_medians': float(np.median(vals)) if len(vals) else np.nan,
                      'p16': float(np.percentile(vals, 16)) if len(vals) else np.nan,
                      'p84': float(np.percentile(vals, 84)) if len(vals) else np.nan})
    # Categorical fractions: compute one fraction per SRG, then summarize.
    per_reference['fraction_controls_same_central_status'] = (
        pairs.assign(same=pairs.central_status_same.astype('Float64'))
        .groupby('reference_track_id')['same'].mean().reindex(per_reference.reference_track_id).to_numpy(dtype=float, na_value=np.nan))
    return per_reference, pd.DataFrame(stats)


def plot_diagnostics(pairs, per_reference, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    quantities = [
        ('delta_log10_bh_mass_raw', r'$\Delta\log_{10} M_{\rm BH}$ (raw ratio)'),
        ('delta_log10_halo_mass_m200crit_raw', r'$\Delta\log_{10} M_{200\rm crit}$ (raw ratio)'),
        ('delta_kappa_co_stars_raw', r'$\Delta\kappa_{\rm co,\star}$ (raw)'),
        ('delta_stellar_age_raw', r'$\Delta\mathrm{stellar\ age}$ (raw units)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, label) in zip(axes.flat, quantities):
        vals = per_reference[col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            ax.hist(vals, bins=min(30, max(5, int(np.sqrt(len(vals)) * 2))), alpha=0.8)
            ax.axvline(np.median(vals), ls='-', lw=1.5, color='C3', label='Median difference')
        ax.axvline(0, color='black', ls='--', lw=1, label="No difference")
        ax.set_xlabel(label)
        ax.set_ylabel('Number of SRGs')
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=10, frameon=False)
    fig.suptitle('Control minus SRG: median difference within each matched set')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'initial_condition_paired_differences.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)

    # Central/satellite contingency, pair-level descriptive only.
    a = pairs.reference_is_central_raw
    b = pairs.control_is_central_raw
    valid = a.isin([0, 1]) & b.isin([0, 1])
    table = pd.crosstab(a[valid].astype(int), b[valid].astype(int))
    table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(table.to_numpy(), cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(table.iloc[i, j]), ha='center', va='center')
    ax.set_xticks([0, 1], ['Satellite', 'Central'])
    ax.set_yticks([0, 1], ['Satellite', 'Central'])
    ax.set_xlabel('Matched analogue')
    ax.set_ylabel('SRG progenitor')
    ax.set_title('Central/satellite status at matching epoch (pairs)')
    fig.colorbar(im, ax=ax, label='Number of pairs')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(outdir / f'initial_condition_central_status.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    table.to_csv(outdir / 'central_status_contingency.csv')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group()
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--write', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--pairs-file', type=Path)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if args.overwrite and not args.write:
        p.error('--overwrite requires --write')
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    root = Path(OUTPUT['root'])
    pairs_path = project_path(args.pairs_file or root / 'controls' / 'matched_control_candidates.csv')
    reference_path = project_path(Path(OUTPUT['reference_dir']) / 'reference_galaxies_z0_all.csv')
    outdir = project_path(args.output_dir or root / 'controls' / 'initial_conditions')
    pairs = load_pairs(pairs_path, args.limit)
    n_known_srgs = verify_exclusion(pairs, reference_path)
    inventory = discover_ascii_catalogues()
    full, epochs = assemble_pairs(pairs, inventory)
    per_reference, stats = summarize(full)
    report = {
        'status': 'candidate_analogues_only_not_confirmed_nonrelics',
        'n_pairs': len(full), 'n_reference_srgs': full.reference_track_id.nunique(),
        'n_unique_controls': full.control_track_id.nunique(),
        'n_known_reference_srgs_excluded': n_known_srgs,
        'n_reused_control_assignments': int(full.control_track_id.duplicated().sum()),
        'n_matching_epochs': len(epochs),
        'source': 'ASCII_at_exact_matching_redshift',
        'units': 'Extra properties are native ASCII values; BH/halo mass and stellar age units NOT verified',
        'inference': 'Descriptive matched-set comparisons only; no independent-pair significance tests',
        'caveats': ['Descendant outcomes unknown', 'Repeated controls create dependence',
                    'Zero BH masses excluded from log-ratio comparisons',
                    'Stellar age weighting and extra-property units require verification',
                    'No local-density or enclosed dark-matter fraction measurement'],
    }
    print('\nINITIAL-CONDITION REPORT')
    print(json.dumps(report, indent=2))
    print('\nPER-SRG MEDIAN DIFFERENCE SUMMARY')
    print(stats.to_string(index=False))
    if not args.write:
        print('\nDry run: no files written.')
        return
    paths = [outdir / name for name in [
        'initial_condition_pairs.csv', 'initial_condition_per_srg.csv',
        'initial_condition_summary.csv', 'initial_condition_epoch_counts.csv',
        'initial_condition_report.json', 'initial_condition_paired_differences.png',
        'initial_condition_paired_differences.pdf', 'initial_condition_central_status.png',
        'initial_condition_central_status.pdf', 'central_status_contingency.csv']]
    existing = [str(path) for path in paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f'Outputs exist; use --write --overwrite: {existing}')
    outdir.mkdir(parents=True, exist_ok=True)
    full.to_csv(paths[0], index=False)
    per_reference.to_csv(paths[1], index=False)
    stats.to_csv(paths[2], index=False)
    epochs.to_csv(paths[3], index=False)
    paths[4].write_text(json.dumps(report, indent=2) + '\n')
    plot_diagnostics(full, per_reference, outdir)
    print('\nSaved outputs in:', outdir)


if __name__ == '__main__':
    main()
