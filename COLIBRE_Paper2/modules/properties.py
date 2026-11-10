#!/usr/bin/env python3
"""COLIBRE Paper 2: attach ASCII physical measurements to saved TrackID identities.

Run from the project root:
    python3 -m modules.properties --dry-run --limit 20
    python3 -m modules.properties --dry-run
    python3 -m modules.properties --write

Raw values are ALWAYS preserved. Physical-unit conversions and dependent
quantities are only computed when all required factors are explicitly set in
config.UNITS. The default configuration therefore yields a useful raw-property
catalogue without pretending that units have been verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import INPUTS, OUTPUT, PROPERTIES, UNITS, validate_config
from .data_io import CatalogueError, discover_ascii_catalogues, read_ascii_snapshot

IDENTITY_KEYS = ('track_id', 'redshift', 'snapshot_file')
IDENTITY_FILE = 'track_identities.csv'
RAW_PROPERTY_NAMES = (
    'stellar_mass', 'rhalf_stars', 'sfr_instant', 'bh_mass',
    'halo_mass_m200crit', 'sfr_100myr', 'bh_accretion_rate',
    'bh_thermal_energy_cum', 'n_agn_events', 'stellar_age',
)
IDENTITY_ONLY_NAMES = {
    'track_id', 'halo_catalogue_index', 'host_fof_halo_id',
    'descendant_id', 'is_central',
}


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    root = Path(__file__).resolve().parents[1]
    return (path if path.is_absolute() else root / path).resolve()


def load_track_identities(path: str | Path | None = None) -> pd.DataFrame:
    path = project_path(path if path is not None else Path(OUTPUT['tracks_dir']) / IDENTITY_FILE)
    if not path.is_file():
        raise FileNotFoundError(f'Tracking catalogue not found: {path}. Run python3 -m modules.tracking --write first.')
    df = pd.read_csv(path, dtype={'track_id': 'int64', 'snapshot_file': 'string'})
    required = set(IDENTITY_KEYS) | {'population'}
    missing = required - set(df.columns)
    if missing:
        raise CatalogueError(f'Track catalogue missing columns: {sorted(missing)}')
    if df[list(IDENTITY_KEYS)].isna().any().any():
        raise CatalogueError('Missing TrackID/redshift/snapshot_file in identities')
    if df.duplicated(list(IDENTITY_KEYS)).any():
        raise CatalogueError('Duplicate identity key (track_id, redshift, snapshot_file)')
    return df


def _property_selection(requested=None) -> tuple[str, ...]:
    requested = tuple(PROPERTIES['requested'] if requested is None else requested)
    # Identity columns are already in the track catalogue; avoid rereading them.
    unknown = set(requested) - set(RAW_PROPERTY_NAMES) - IDENTITY_ONLY_NAMES
    if unknown:
        raise ValueError(f'Unknown property names: {sorted(unknown)}')
    return tuple(dict.fromkeys(p for p in requested if p not in IDENTITY_ONLY_NAMES))


def _positive_log(values: pd.Series) -> pd.Series:
    a = pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)
    out = np.full(a.shape, np.nan, dtype=float)
    ok = np.isfinite(a) & (a > 0)
    out[ok] = np.log10(a[ok])
    return pd.Series(out, index=values.index)


def _unit_factor(name: str) -> float | None:
    factor = UNITS[name]
    if factor is None:
        return None
    try:
        factor = float(factor)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'UNITS[{name!r}] must be a positive finite number or None') from exc
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f'UNITS[{name!r}] must be positive and finite')
    return factor


def add_physical_and_derived_properties(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert only explicitly calibrated raw quantities; keep zeros and NaNs distinct."""
    out = frame.copy()
    mapping = {
        'stellar_mass': ('ascii_stellar_mass_to_msun', 'stellar_mass_msun'),
        'rhalf_stars': ('ascii_radius_to_physical_kpc', 'rhalf_physical_kpc'),
        'sfr_instant': ('ascii_sfr_to_msun_per_year', 'sfr_msun_per_year'),
    }
    factors = {key: _unit_factor(key) for key, _ in mapping.values()}
    converted = []
    for raw, (factor_key, target) in mapping.items():
        if raw not in out.columns:
            continue
        factor = factors[factor_key]
        if factor is not None:
            out[target] = pd.to_numeric(out[raw], errors='coerce') * factor
            converted.append(target)

    if 'stellar_mass_msun' in out:
        out['log10_stellar_mass'] = _positive_log(out['stellar_mass_msun'])
    if 'rhalf_physical_kpc' in out:
        out['log10_rhalf'] = _positive_log(out['rhalf_physical_kpc'])
    if 'log10_stellar_mass' in out and 'log10_rhalf' in out:
        exponent = float(__import__('config').REFERENCE_SAMPLE['compactness_exponent'])
        out['compactness_sigma_1p5'] = out['log10_stellar_mass'] - exponent * out['log10_rhalf']
    if 'sfr_msun_per_year' in out and 'stellar_mass_msun' in out:
        sfr = out['sfr_msun_per_year'].to_numpy(dtype=float)
        mass = out['stellar_mass_msun'].to_numpy(dtype=float)
        ssfr = np.full(len(out), np.nan, dtype=float)
        good = np.isfinite(sfr) & (sfr >= 0) & np.isfinite(mass) & (mass > 0)
        ssfr[good] = sfr[good] / mass[good]
        out['ssfr_instant'] = ssfr  # genuine SFR=0 -> sSFR=0
        out['log10_ssfr_instant'] = _positive_log(out['ssfr_instant'])
        floor = float(PROPERTIES['ssfr_plot_floor_log10_per_year'])
        plot_log = out['log10_ssfr_instant'].copy()
        zero = good & (sfr == 0)
        plot_log.loc[zero] = floor
        plot_log = plot_log.clip(lower=floor)
        out['log10_ssfr_plot'] = plot_log  # missing remains NaN

    return out, {
        'unit_factors': factors,
        'converted_columns': converted,
        'units_validated_by_pipeline': False,
        'conversion_policy': 'Factors explicitly supplied in config.UNITS; physical interpretation must be independently verified.',
    }


def build_evolutionary_properties(identities: pd.DataFrame, *, ascii_dir=None,
                                  requested=None, verbose=True) -> tuple[pd.DataFrame, dict]:
    """Join raw properties by exact TrackID + source filename, without row-order assumptions."""
    validate_config()
    required = set(IDENTITY_KEYS)
    if not required <= set(identities.columns):
        raise CatalogueError(f'Identity input missing {sorted(required - set(identities.columns))}')
    if identities[list(IDENTITY_KEYS)].isna().any().any() or identities.duplicated(list(IDENTITY_KEYS)).any():
        raise CatalogueError('Invalid or duplicate identity keys')
    if identities.empty:
        raise ValueError('No identity rows supplied')

    requested_properties = _property_selection(requested)
    inventory = discover_ascii_catalogues(directory=ascii_dir)
    path_map = {p.name: (p, float(z)) for z, p in zip(inventory['redshift'], inventory['path'])}
    missing_files = sorted(set(identities['snapshot_file'].astype(str)) - set(path_map))
    if missing_files:
        raise FileNotFoundError(f'Identity snapshots not found in ASCII inventory: {missing_files}')

    pieces = []
    for filename, group in identities.groupby('snapshot_file', sort=False):
        path, z = path_map[str(filename)]
        if not np.all(np.isclose(group['redshift'].to_numpy(dtype=float), z, rtol=0, atol=1e-8)):
            raise CatalogueError(f'Redshift disagreement for {filename}')
        # Optional columns (e.g. sfr_100myr) are absent from some 32-column files.
        from .data_io import ascii_schema
        available = set(ascii_schema(path))
        cols = [p for p in requested_properties if p in available]
        absent = [p for p in requested_properties if p not in available]
        data = read_ascii_snapshot(path, columns=['track_id', *cols],
                                   track_ids=group['track_id'].astype('int64'),
                                   validate_track_ids=True)
        if len(data) != len(group):
            raise CatalogueError(f'{filename}: matched {len(data)} rows but expected {len(group)}; identities may be stale')
        # Rename source fields to clearly distinguish raw values from calibrated values.
        data = data.rename(columns={p: f'{p}_raw' for p in cols})
        # Join on IDs; population and all identity fields remain from tracking.
        joined = group.merge(data, on='track_id', how='left', validate='one_to_one', indicator=True)
        if (joined['_merge'] != 'both').any():
            raise CatalogueError(f'{filename}: some tracked IDs could not be joined')
        joined = joined.drop(columns='_merge')
        for name in absent:
            joined[f'{name}_raw'] = np.nan
        pieces.append(joined)
        if verbose:
            print(f'z={z:5.2f}: joined {len(joined):5d} tracked galaxies; unavailable fields: {absent or "none"}')

    result = pd.concat(pieces, ignore_index=True)
    if len(result) != len(identities):
        raise CatalogueError('Property join changed the number of track rows')
    if result.duplicated(list(IDENTITY_KEYS)).any():
        raise CatalogueError('Duplicate identity keys after property extraction')
    # Restore canonical raw names temporarily for conversion, then retain only suffixed names.
    for name in requested_properties:
        raw = f'{name}_raw'
        if raw in result:
            result[name] = result[raw]
    result, unit_report = add_physical_and_derived_properties(result)
    for name in requested_properties:
        if name in result:
            result = result.drop(columns=name)
    result = result.sort_values(['track_id', 'redshift'], ascending=[True, False]).reset_index(drop=True)
    report = {
        'n_input_identity_rows': int(len(identities)),
        'n_output_property_rows': int(len(result)),
        'n_unique_tracks': int(result['track_id'].nunique()),
        'n_epochs': int(result['redshift'].nunique()),
        'source': 'ASCII',
        'requested_raw_properties': list(requested_properties),
        'raw_column_suffix': '_raw',
        'note': 'No SOAP/ASCII mixing; missing snapshots are not invented; zero SFR is preserved.',
        **unit_report,
    }
    return result, report


def save_evolutionary_properties(frame: pd.DataFrame, report: dict, *, directory=None,
                                 overwrite=False) -> tuple[Path, Path]:
    directory = project_path(directory if directory is not None else OUTPUT['tracks_dir'])
    files = (directory / 'evolutionary_properties.csv', directory / 'evolutionary_properties_report.json')
    existing = [str(p) for p in files if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Output(s) exist; use --overwrite: {existing}')
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(files[0], index=False)
    files[1].write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--write', action='store_true', help='Save property catalogue and report')
    group.add_argument('--dry-run', action='store_true', help='Run without saving (default)')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing saved property outputs with --write')
    parser.add_argument('--limit', type=int, help='Development only: first N distinct tracked galaxies')
    args = parser.parse_args()
    if args.overwrite and not args.write:
        parser.error('--overwrite requires --write')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.limit is not None and args.write:
        parser.error('Refusing to write a development subset as the production property catalogue')
    identities = load_track_identities()
    if args.limit is not None:
        ids = sorted(identities['track_id'].unique())[:args.limit]
        identities = identities.loc[identities['track_id'].isin(ids)].copy()
    result, report = build_evolutionary_properties(identities)
    print('\nProperty extraction report:\n' + json.dumps(report, indent=2))
    if args.write:
        for path in save_evolutionary_properties(result, report, overwrite=args.overwrite):
            print('Saved:', path)
    else:
        print('Dry run: no files written.')


if __name__ == '__main__':
    main()
