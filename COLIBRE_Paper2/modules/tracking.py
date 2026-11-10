#!/usr/bin/env python3
"""COLIBRE Paper 2: TrackID-based identity histories (no physical properties).

From the project root:
    python3 -m modules.tracking --dry-run
    python3 -m modules.tracking --write
    python3 -m modules.tracking --write --overwrite
    python3 -m modules.tracking --dry-run --limit 20

This module matches persistent HBT+ TrackIDs across the processed ASCII
catalogues. It does NOT reconstruct secondary merger-tree branches, infer
formation epochs, convert units, or automatically substitute SOAP values.
The reference population label is always the z=0 classification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEVELOPMENT, INPUTS, OUTPUT, REFERENCE_SAMPLE, TRACKING, validate_config
from .data_io import CatalogueError, discover_ascii_catalogues, read_ascii_snapshot

IDENTITY_COLUMNS = (
    'track_id', 'population', 'redshift', 'snapshot', 'snapshot_file',
    'source', 'halo_catalogue_index', 'is_central', 'host_fof_halo_id',
    'descendant_id',
)
ASCII_IDENTITY_COLUMNS = (
    'track_id', 'halo_catalogue_index', 'is_central',
    'host_fof_halo_id', 'descendant_id',
)
REFERENCE_FILE = 'reference_galaxies_z0_all.csv'
REPORT_FILE = 'reference_galaxies_z0_report.json'


def _project_path(path: Path | str) -> Path:
    path = Path(path).expanduser()
    root = Path(__file__).resolve().parents[1]
    return (path if path.is_absolute() else root / path).resolve()


def load_reference(path: Path | str | None = None) -> pd.DataFrame:
    """Load and check the saved Paper 1 reference catalogue, without SOAP reads."""
    if path is None:
        path = _project_path(OUTPUT['reference_dir']) / REFERENCE_FILE
    path = _project_path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Reference catalogue missing: {path}')
    frame = pd.read_csv(path, dtype={'track_id': 'int64'})
    required = {'track_id', 'population', 'halo_catalogue_index'}
    missing = required - set(frame.columns)
    if missing:
        raise CatalogueError(f'Reference catalogue missing: {sorted(missing)}')
    if frame['track_id'].isna().any() or frame['track_id'].duplicated().any():
        raise CatalogueError('Reference TrackIDs must be unique and nonmissing')
    return frame


def select_reference(frame: pd.DataFrame, populations=None) -> pd.DataFrame:
    """Select reference classes; SAG_all is SRG union SAG_noncompact."""
    populations = tuple(REFERENCE_SAMPLE['populations'] if populations is None else populations)
    allowed = set(REFERENCE_SAMPLE['allowed_populations'])
    if not populations or not set(populations) <= allowed:
        raise ValueError(f'Invalid population selection: {populations}')
    labels = set(populations) - {'SAG_all'}
    if 'SAG_all' in populations:
        labels.update(('SRG', 'SAG_noncompact'))
    selected = frame.loc[frame['population'].isin(labels)].copy()
    if selected.empty:
        raise ValueError(f'No reference galaxies for {populations}')
    return selected


def _inventory(redshift_min=None, redshift_max=None, selected_redshifts=None,
               ascii_dir=None) -> pd.DataFrame:
    inventory = discover_ascii_catalogues(directory=ascii_dir)
    if inventory.empty:
        raise FileNotFoundError('No processed ASCII catalogues found')
    zmin = TRACKING['redshift_min'] if redshift_min is None else float(redshift_min)
    zmax = TRACKING['redshift_max'] if redshift_max is None else float(redshift_max)
    if zmin > zmax:
        raise ValueError('redshift_min > redshift_max')
    inventory = inventory.loc[inventory['redshift'].between(zmin, zmax)].copy()
    chosen = TRACKING['selected_redshifts'] if selected_redshifts is None else selected_redshifts
    if chosen is not None:
        chosen = np.asarray(chosen, dtype=float)
        inventory = inventory.loc[inventory['redshift'].apply(
            lambda z: bool(np.any(np.isclose(z, chosen, atol=1e-8, rtol=0))))].copy()
        missing = [float(z) for z in chosen if not np.any(np.isclose(
            inventory['redshift'].to_numpy(dtype=float), z, atol=1e-8, rtol=0))]
        if missing:
            raise ValueError(f'Requested ASCII redshifts unavailable: {missing}')
    if inventory.empty:
        raise ValueError('No ASCII catalogues within requested redshift selection')
    return inventory.sort_values('redshift').reset_index(drop=True)


def build_track_identities(reference: pd.DataFrame, *, ascii_dir=None,
                           redshift_min=None, redshift_max=None,
                           selected_redshifts=None, verbose=True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (observed identities, epoch completeness, summary report).

    Observed identities contain one row per detected (TrackID, redshift).
    Missing galaxies are counted in completeness, not fabricated as rows.
    No SOAP snapshot numbers are inferred from redshift or filenames.
    """
    validate_config()
    if TRACKING['method'] != 'HBT_TrackID':
        raise NotImplementedError('Only HBT_TrackID tracking is implemented')
    if TRACKING['snapshot_selection'] != 'available_ascii':
        raise NotImplementedError('Only available_ascii snapshot selection is implemented')
    if TRACKING['duplicate_trackid_policy'] != 'raise':
        raise ValueError('Tracking requires duplicate_trackid_policy=raise')
    if TRACKING['missing_snapshot_policy'] != 'report':
        raise ValueError('Tracking requires missing_snapshot_policy=report')
    if reference['track_id'].duplicated().any() or reference['track_id'].isna().any():
        raise CatalogueError('Reference TrackIDs must be unique and nonmissing')
    targets = reference[['track_id', 'population']].copy()
    targets['track_id'] = targets['track_id'].astype('int64')
    inventory = _inventory(redshift_min, redshift_max, selected_redshifts, ascii_dir)
    target_ids = set(targets['track_id'].tolist())
    frames, coverage = [], []
    for row in inventory.itertuples(index=False):
        # Low-level reader validates all TrackIDs in each source snapshot.
        data = read_ascii_snapshot(row.path, columns=ASCII_IDENTITY_COLUMNS,
                                   track_ids=target_ids, validate_track_ids=True)
        if data['track_id'].duplicated().any():
            raise CatalogueError(f'{row.path}: repeated matched TrackID')
        data = data.merge(targets, on='track_id', how='left', validate='one_to_one')
        data['redshift'] = float(row.redshift)
        data['snapshot'] = pd.NA  # No verified ASCII redshift-to-snapshot mapping yet.
        data['snapshot_file'] = Path(row.path).name
        data['source'] = 'ASCII'
        frames.append(data.loc[:, IDENTITY_COLUMNS])
        n_found = len(data)
        coverage.append({'redshift': float(row.redshift), 'snapshot_file': Path(row.path).name,
                         'n_targets': len(targets), 'n_found': n_found,
                         'n_missing': len(targets) - n_found,
                         'fraction_found': n_found / len(targets)})
        if verbose:
            print(f'z={row.redshift:5.2f}: {n_found:5d}/{len(targets)} TrackIDs found')
    tracks = pd.concat(frames, ignore_index=True)
    tracks['snapshot'] = pd.array(tracks['snapshot'], dtype='Int64')
    tracks = tracks.sort_values(['track_id', 'redshift'],
                                ascending=[True, False]).reset_index(drop=True)
    if tracks.duplicated(['track_id', 'redshift']).any():
        raise CatalogueError('Duplicate (track_id, redshift) pairs in assembled tracks')
    coverage_df = pd.DataFrame(coverage)
    n_unique = int(tracks['track_id'].nunique())
    report = {
        'tracking_method': 'HBT_TrackID',
        'source': 'ASCII',
        'n_reference_targets': int(len(targets)),
        'n_targets_seen_at_least_once': n_unique,
        'n_targets_never_seen': int(len(targets) - n_unique),
        'n_epochs': int(len(inventory)),
        'n_observed_track_rows': int(len(tracks)),
        'redshift_min': float(inventory['redshift'].min()),
        'redshift_max': float(inventory['redshift'].max()),
        'population_counts': {str(k): int(v) for k, v in targets['population'].value_counts().items()},
        'note': ('Only observed TrackID matches are included. Missing entries are not '
                 'interpreted as physical disappearance. Snapshot numbers and units are not inferred.'),
    }
    return tracks, coverage_df, report


def save_track_identities(tracks: pd.DataFrame, coverage: pd.DataFrame, report: dict,
                          *, directory=None, overwrite=False) -> tuple[Path, Path, Path]:
    directory = _project_path(directory if directory is not None else OUTPUT['tracks_dir'])
    files = (directory / 'track_identities.csv',
             directory / 'tracking_completeness.csv',
             directory / 'tracking_report.json')
    existing = [str(p) for p in files if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Output(s) exist; use --overwrite: {existing}')
    directory.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(files[0], index=False)
    coverage.to_csv(files[1], index=False)
    files[2].write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--write', action='store_true', help='Write tracking catalogues')
    group.add_argument('--dry-run', action='store_true', help='Run without writing (default)')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing tracking files with --write')
    parser.add_argument('--limit', type=int, help='First N selected reference galaxies (development only)')
    parser.add_argument('--redshift-min', type=float)
    parser.add_argument('--redshift-max', type=float)
    args = parser.parse_args()
    if args.overwrite and not args.write:
        parser.error('--overwrite requires --write')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    selected = select_reference(load_reference())
    if args.limit is not None:
        selected = selected.sort_values('track_id').head(args.limit).copy()
    tracks, coverage, report = build_track_identities(
        selected, redshift_min=args.redshift_min, redshift_max=args.redshift_max)
    print('\nTracking report:\n' + json.dumps(report, indent=2))
    if args.write:
        if args.limit is not None:
            parser.error('Refusing to write development subset as production track catalogue')
        paths = save_track_identities(tracks, coverage, report, overwrite=args.overwrite)
        for path in paths:
            print('Saved:', path)
    else:
        print('Dry run: no files written.')


if __name__ == '__main__':
    main()
