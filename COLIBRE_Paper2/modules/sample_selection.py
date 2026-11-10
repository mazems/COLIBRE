#!/usr/bin/env python3
"""Build the COLIBRE Paper 2 z=0 reference population from SOAP + SFH/DoR.

Run from project root:
    python -m modules.sample_selection --dry-run
    python -m modules.sample_selection --write

Scientific definitions follow the supplied DoRandtime_relations.py. The
conversion factors are *legacy Paper 1 conventions*, not independently
verified physical-unit metadata. No evolutionary ASCII measurements are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import INPUTS, OUTPUT, REFERENCE_SAMPLE, SIMULATION, validate_config
from .data_io import CatalogueError, read_dor_catalogue, read_soap_catalogue

# Exact arithmetic used in the supplied Paper 1 script (legacy conventions).
LEGACY_MASS_FACTOR = 1.988e43 / 1.989e33
LEGACY_RADIUS_FACTOR_Z0 = 1e3

SOAP_FIELDS = {
    'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/TrackId'),
    'ExclusiveSphere/50kpc': ('StellarMass', 'HalfMassRadiusStars'),
}

TIME_COLUMNS = (
    't_start', 't50', 't50_span', 't75', 't75_span', 't90', 't90_span',
    't95', 't95_span', 't998', 't998_span', 'tfin', 'tfin_span',
)


def _strict_integer_array(values, name: str) -> np.ndarray:
    """Require integer-valued SOAP identifiers without rounding."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise CatalogueError(f'{name} must be a one-dimensional array')
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64)
    if not np.issubdtype(arr.dtype, np.number):
        raise CatalogueError(f'{name}: unexpected dtype {arr.dtype}')
    if not np.all(np.isfinite(arr)) or not np.all(arr == np.floor(arr)):
        raise CatalogueError(f'{name}: missing or non-integral identifiers')
    if np.any(np.abs(arr) >= 2**53):
        raise CatalogueError(f'{name}: float identifiers exceed safe precision')
    return arr.astype(np.int64)


def build_reference_catalogue(
    *,
    mass_factor: float = LEGACY_MASS_FACTOR,
    radius_factor: float = LEGACY_RADIUS_FACTOR_Z0,
    dor_path: Path | str | None = None,
    soap_directory: Path | str | None = None,
    reference_snapshot: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return all SOAP-eligible galaxies with DoR matches and a diagnostic report.

    Class labels are mutually exclusive: SRG, SAG_noncompact,
    compact_nonrelic, noncompact_nonrelic, or unclassified (boundary cases).
    SAG_all is a *selection alias* for SRG union SAG_noncompact, not a label.
    """
    validate_config()
    if not np.isfinite(mass_factor) or mass_factor <= 0:
        raise ValueError('mass_factor must be finite and positive')
    if not np.isfinite(radius_factor) or radius_factor <= 0:
        raise ValueError('radius_factor must be finite and positive')
    if REFERENCE_SAMPLE['dor_comparison'] != '>':
        raise ValueError('This implementation reproduces Paper 1 strict DoR > threshold')
    if (REFERENCE_SAMPLE['srg_compactness_comparison'] != '>=' or
            REFERENCE_SAMPLE['noncompact_sag_compactness_comparison'] != '<'):
        raise ValueError('Unexpected compactness boundary convention')
    if REFERENCE_SAMPLE['compactness_exponent'] != 1.5:
        raise ValueError('Unexpected compactness exponent; review Paper 1 parity')

    snap = SIMULATION['reference_snapshot'] if reference_snapshot is None else int(reference_snapshot)
    soap = read_soap_catalogue(snap, SOAP_FIELDS, directory=soap_directory)
    idx = _strict_integer_array(soap['InputHalos/HaloCatalogueIndex'], 'SOAP HaloCatalogueIndex')
    tid = _strict_integer_array(soap['InputHalos/HBTplus/TrackId'], 'SOAP TrackId')
    central = np.asarray(soap['InputHalos/IsCentral'])
    mass_raw = np.asarray(soap['ExclusiveSphere/50kpc/StellarMass'], dtype=float)
    radius_raw = np.asarray(soap['ExclusiveSphere/50kpc/HalfMassRadiusStars'], dtype=float)
    n = len(idx)
    if any(len(x) != n for x in (tid, central, mass_raw, radius_raw)):
        raise CatalogueError('SOAP arrays are not row-aligned')
    if len(np.unique(idx)) != n or len(np.unique(tid)) != n:
        raise CatalogueError('SOAP HaloCatalogueIndex / TrackId not unique')

    mass = mass_raw * mass_factor
    radius = radius_raw * radius_factor
    eligible = (np.isfinite(mass) & (mass >= REFERENCE_SAMPLE['minimum_stellar_mass_msun'])
                & np.isfinite(radius))
    if REFERENCE_SAMPLE['require_positive_radius']:
        eligible &= radius > 0
    # This is the order and the mask used by the supplied Paper 1 script.
    soap_df = pd.DataFrame({
        'halo_catalogue_index': idx[eligible],
        'track_id': tid[eligible],
        'is_central_z0': central[eligible],
        'stellar_mass_z0': mass[eligible],
        'rhalf_z0': radius[eligible],
    })
    soap_df['log10_stellar_mass_z0'] = np.log10(soap_df['stellar_mass_z0'])
    soap_df['log10_rhalf_z0'] = np.log10(soap_df['rhalf_z0'])
    soap_df['compactness_z0'] = (soap_df['log10_stellar_mass_z0']
                                - REFERENCE_SAMPLE['compactness_exponent'] * soap_df['log10_rhalf_z0'])

    dor = read_dor_catalogue(path=dor_path, dor_column=REFERENCE_SAMPLE['dor_column'])
    id_column = dor.attrs['identifier_column']
    # The Paper 1 script matched its CSV subhalo_id to SOAP HaloCatalogueIndex.
    # A TrackId-labelled DoR file is not interchangeable and must be explicit.
    if id_column.lower() in ('track_id', 'trackid'):
        raise CatalogueError('DoR CSV uses TrackID, but Paper 1 matched HaloCatalogueIndex; '
                             'verify ID semantics before changing the join')
    dor = dor.rename(columns={id_column: 'halo_catalogue_index'})
    if 'track_id' in dor.columns:
        raise CatalogueError('DoR CSV contains an extra track_id column; resolve ID semantics first')
    dor['halo_catalogue_index'] = dor['halo_catalogue_index'].astype(np.int64)
    if dor['halo_catalogue_index'].duplicated().any():
        raise CatalogueError('Duplicate DoR HaloCatalogueIndex')
    dor_keep = ['halo_catalogue_index', REFERENCE_SAMPLE['dor_column']]
    dor_keep += [c for c in dor.columns if c.startswith('DoR_') or c in TIME_COLUMNS
                 if c not in dor_keep]
    matched = soap_df.merge(dor[dor_keep], on='halo_catalogue_index', how='left',
                            validate='one_to_one', indicator=True)
    dor_col = REFERENCE_SAMPLE['dor_column']
    has_dor = matched[dor_col].notna()
    ancient = matched[dor_col] > REFERENCE_SAMPLE['dor_threshold']
    compact = matched['compactness_z0'] >= REFERENCE_SAMPLE['compactness_threshold']
    matched['population'] = 'unclassified'
    matched.loc[has_dor & ancient & compact, 'population'] = 'SRG'
    matched.loc[has_dor & ancient & ~compact, 'population'] = 'SAG_noncompact'
    # Paper 1 uses strict '<' for non-relic DoR, and strict '>' for compact non-relics.
    nonrelic = matched[dor_col] < REFERENCE_SAMPLE['dor_threshold']
    strictly_compact = matched['compactness_z0'] > REFERENCE_SAMPLE['compactness_threshold']
    matched.loc[has_dor & nonrelic & strictly_compact, 'population'] = 'compact_nonrelic'
    matched.loc[has_dor & nonrelic & ~strictly_compact, 'population'] = 'noncompact_nonrelic'
    matched['reference_snapshot'] = snap
    matched['reference_redshift'] = SIMULATION['reference_redshift']
    matched['has_dor'] = has_dor
    matched['is_ancient'] = has_dor & ancient
    matched['is_compact'] = compact
    matched = matched.drop(columns='_merge')

    report = {
        'snapshot': snap,
        'soap_rows': int(n),
        'soap_eligible': int(len(soap_df)),
        'dor_rows': int(len(dor)),
        'eligible_with_dor': int(has_dor.sum()),
        'eligible_without_dor': int((~has_dor).sum()),
        'population_counts': {str(k): int(v) for k, v in matched['population'].value_counts().items()},
        'sag_all': int(matched['is_ancient'].sum()),
        'dor_column': dor_col,
        'mass_factor': float(mass_factor),
        'radius_factor': float(radius_factor),
        'conversion_status': 'legacy_paper1_unverified_units',
        'reference_soap_path': str((Path(soap_directory) if soap_directory is not None else INPUTS['soap_dir']) /
                                   INPUTS['soap_filename_pattern'].format(snapshot=snap)),
        'dor_path': str(dor_path if dor_path is not None else INPUTS['dor_sfh_csv']),
    }
    return matched, report


def select_populations(catalogue: pd.DataFrame, populations=None) -> pd.DataFrame:
    """Select populations from a full reference catalogue; SAG_all is a union."""
    populations = tuple(REFERENCE_SAMPLE['populations'] if populations is None else populations)
    allowed = set(REFERENCE_SAMPLE['allowed_populations'])
    if not populations or not set(populations) <= allowed:
        raise ValueError(f'Unknown requested populations: {set(populations) - allowed}')
    names = set(populations)
    if 'SAG_all' in names:
        names.remove('SAG_all')
        names.update(('SRG', 'SAG_noncompact'))
    selected = catalogue.loc[catalogue['population'].isin(names)].copy()
    if selected['track_id'].duplicated().any():
        raise CatalogueError('Selected reference sample has duplicate TrackIDs')
    return selected.reset_index(drop=True)


def save_reference_catalogue(catalogue: pd.DataFrame, report: dict, *,
                             directory: Path | str | None = None,
                             overwrite: bool | None = None) -> dict:
    """Save the complete reference sample and diagnostics; never overwrite silently."""
    directory = Path(directory if directory is not None else OUTPUT['reference_dir'])
    overwrite = OUTPUT['overwrite'] if overwrite is None else bool(overwrite)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / 'reference_galaxies_z0_all.csv'
    report_path = directory / 'reference_galaxies_z0_report.json'
    for path in (csv_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f'Output already exists: {path}. Use --overwrite intentionally.')
    catalogue.to_csv(csv_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return {'catalogue': csv_path, 'report': report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description='Build Paper 1-consistent z=0 COLIBRE reference sample')
    parser.add_argument('--dry-run', action='store_true', help='Inspect counts without writing files')
    parser.add_argument('--write', action='store_true', help='Write the full reference catalogue')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing outputs when writing')
    parser.add_argument('--dor-file', type=Path, default=None)
    parser.add_argument('--soap-dir', type=Path, default=None)
    parser.add_argument('--mass-factor', type=float, default=LEGACY_MASS_FACTOR)
    parser.add_argument('--radius-factor', type=float, default=LEGACY_RADIUS_FACTOR_Z0)
    args = parser.parse_args()
    if not args.dry_run and not args.write:
        parser.error('Specify --dry-run or --write')
    catalogue, report = build_reference_catalogue(
        mass_factor=args.mass_factor, radius_factor=args.radius_factor,
        dor_path=args.dor_file, soap_directory=args.soap_dir,
    )
    print('Reference sample report:')
    print(json.dumps(report, indent=2))
    selected = select_populations(catalogue)
    print(f"Selected populations {REFERENCE_SAMPLE['populations']}: {len(selected)} galaxies")
    if args.write:
        saved = save_reference_catalogue(catalogue, report, overwrite=args.overwrite)
        print('Saved:', {k: str(v) for k, v in saved.items()})
    else:
        print('Dry run: no files written.')
    print('NOTE: conversion factors reproduce supplied Paper 1 arithmetic; units remain to be verified.')


if __name__ == '__main__':
    main()
