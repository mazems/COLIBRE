#!/usr/bin/env python3
"""COLIBRE Paper 2: low-level, non-scientific catalogue input/output.

Readers preserve source values and identifiers. No implicit unit conversion,
classification, TrackID traversal, or automatic SOAP/ASCII fallback occurs.

Run from the project root, e.g. `python -m modules.data_io --inventory`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np
import pandas as pd

from decimal import Decimal, InvalidOperation

from config import INPUTS, SIMULATION


ASCII_COLUMNS_32 = (
    'halo_catalogue_index', 'is_central', 'x_galaxy', 'y_galaxy',
    'z_galaxy', 'stellar_mass', 'sfr_instant', 'rhalf_stars',
    'h1_mass', 'h2_mass', 'kappa_co_stars', 'kappa_co_gas',
    'disc_to_total_stars', 'stellar_angular_momentum', 'stellar_age',
    'gas_metallicity_lower', 'gas_metallicity_upper', 'dust_mass',
    'descendant_id', 'track_id', 'vel_x', 'vel_y', 'vel_z',
    'host_halo_x', 'host_halo_y', 'host_halo_z', 'halo_mass_m200crit',
    'bh_mass', 'bh_accretion_rate', 'bh_thermal_energy_cum',
    'n_agn_events', 'host_fof_halo_id',
)
ASCII_COLUMNS_33 = (
    ASCII_COLUMNS_32[:27] + ('sfr_100myr',) + ASCII_COLUMNS_32[27:]
)
ASCII_SCHEMAS = {32: ASCII_COLUMNS_32, 33: ASCII_COLUMNS_33}
ASCII_PATTERN = re.compile(r'^GalaxyProperties_SFR_GE_0_z(?P<z>\d+(?:\.\d+)?)\.txt$')
SOAP_PATTERN = re.compile(r'^halo_properties_(?P<snap>\d+)\.hdf5$')


class CatalogueError(RuntimeError):
    """Input catalogue does not satisfy the declared format or constraints."""


def _existing_file(path: Path | str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Catalogue not found: {path}')
    return path


def _validate_unique_ids(frame: pd.DataFrame, column: str, label: str) -> None:
    if frame[column].isna().any():
        raise CatalogueError(f'{label}: null identifiers in {column}')
    if frame[column].duplicated().any():
        sample = frame.loc[frame[column].duplicated(keep=False), column].head(5).tolist()
        raise CatalogueError(f'{label}: duplicate {column} identifiers, examples: {sample}')


def _integer_ids(series: pd.Series, label: str) -> pd.Series:
    """
    Convert catalogue identifiers to signed 64-bit integers.

    Supports:
        7758881
        "7758881"
        "7.758881000000000000e+06"

    Rejects:
        missing values
        non-finite values
        non-integral values
        identifiers outside the signed int64 range

    Scientific notation is interpreted using Decimal to avoid
    unnecessary floating-point precision loss.
    """

    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max

    converted = []

    for value in series:

        if pd.isna(value):
            raise CatalogueError(
                f"{label}: missing identifier"
            )

        try:
            decimal_value = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            raise CatalogueError(
                f"{label}: invalid identifier {value!r}"
            )

        if not decimal_value.is_finite():
            raise CatalogueError(
                f"{label}: non-finite identifier {value!r}"
            )

        if decimal_value != decimal_value.to_integral_value():
            raise CatalogueError(
                f"{label}: non-integer identifier {value!r}"
            )

        integer_value = int(decimal_value)

        if not (int64_min <= integer_value <= int64_max):
            raise CatalogueError(
                f"{label}: identifier outside int64 range: "
                f"{value!r}"
            )

        converted.append(integer_value)

    return pd.Series(
        converted,
        index=series.index,
        dtype=np.int64,
        name=series.name,
    )

def discover_ascii_catalogues(directory: Path | str | None = None,
                              pattern: str | None = None) -> pd.DataFrame:
    """Return numeric-redshift-sorted ASCII inventory (not a snapshot map)."""
    directory = Path(directory if directory is not None else INPUTS['ascii_dir'])
    pattern = pattern or INPUTS['ascii_glob']
    rows = []
    for path in directory.glob(pattern):
        match = ASCII_PATTERN.fullmatch(path.name)
        if path.is_file() and match:
            rows.append({'redshift': float(match.group('z')), 'path': path})
    result = pd.DataFrame(rows, columns=['redshift', 'path'])
    if not result.empty:
        if result['redshift'].duplicated().any():
            raise CatalogueError('Duplicate ASCII redshift entries')
        result = result.sort_values('redshift').reset_index(drop=True)
    return result


def discover_soap_catalogues(directory: Path | str | None = None) -> pd.DataFrame:
    """Return locally available SOAP-HBT snapshot numbers; redshifts unknown."""
    directory = Path(directory if directory is not None else INPUTS['soap_dir'])
    rows = []
    for path in directory.glob('halo_properties_*.hdf5'):
        match = SOAP_PATTERN.fullmatch(path.name)
        if path.is_file() and match:
            rows.append({'snapshot': int(match.group('snap')), 'path': path})
    result = pd.DataFrame(rows, columns=['snapshot', 'path'])
    if not result.empty:
        if result['snapshot'].duplicated().any():
            raise CatalogueError('Duplicate SOAP snapshot entries')
        result = result.sort_values('snapshot').reset_index(drop=True)
    return result


def soap_path(snapshot: int, directory: Path | str | None = None) -> Path:
    """Resolve an explicitly selected SOAP-HBT snapshot, without fallback."""
    directory = Path(directory if directory is not None else INPUTS['soap_dir'])
    return directory / INPUTS['soap_filename_pattern'].format(snapshot=int(snapshot))


def read_soap_catalogue(snapshot: int, fields: Mapping[str, Sequence[str]],
                        directory: Path | str | None = None,
                        include_attributes: bool = False):
    """Read named HDF5 datasets in one file open, without unit conversions.

    Parameters
    ----------
    fields : mapping of group path to dataset names, as in existing common.py.
        E.g. {'InputHalos': ('HaloCatalogueIndex', 'HBTplus/TrackId')}.
    include_attributes : bool
        If True return (data, attrs), with attributes per full dataset path.

    Returns
    -------
    dict[str, numpy.ndarray] keyed by complete HDF5 dataset paths; optionally
    paired with dict of raw dataset attributes. Does not assume equal shapes.
    """
    path = _existing_file(soap_path(snapshot, directory))
    if not fields:
        raise ValueError('SOAP fields cannot be empty')
    data, attributes = {}, {}
    with h5py.File(path, 'r') as handle:
        for group, names in fields.items():
            for name in names:
                key = f'{group.strip("/")}/{name.strip("/")}'
                if key in data:
                    raise ValueError(f'Repeated SOAP dataset: {key}')
                if key not in handle or not isinstance(handle[key], h5py.Dataset):
                    raise CatalogueError(f'{path}: missing dataset {key}')
                ds = handle[key]
                data[key] = ds[()]
                if include_attributes:
                    attributes[key] = dict(ds.attrs.items())
    return (data, attributes) if include_attributes else data


def ascii_schema(path: Path | str) -> tuple[str, ...]:
    """Identify 32/33-column format from the first nonblank, noncomment row."""
    path = _existing_file(path)
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith('#'):
                ncols = len(line.split())
                if ncols not in ASCII_SCHEMAS:
                    raise CatalogueError(f'{path}: expected 32 or 33 columns, found {ncols}')
                return ASCII_SCHEMAS[ncols]
    raise CatalogueError(f'{path}: no data rows')


def read_ascii_snapshot(path: Path | str, columns: Optional[Sequence[str]] = None,
                        track_ids: Optional[Iterable[int]] = None,
                        validate_track_ids: bool = True) -> pd.DataFrame:
    """Read selected ASCII columns, preserving unconverted values.

    If track_ids is provided, filtering occurs after parsing. All requested
    columns are loaded for the file; ASCII text cannot be indexed by TrackID.
    Redshift is read from the filename only when it follows the known format.
    """
    path = _existing_file(path)
    schema = ascii_schema(path)
    selected = list(schema if columns is None else columns)
    if track_ids is not None and 'track_id' not in selected:
        selected.append('track_id')
    if validate_track_ids and 'track_id' not in selected:
        selected.append('track_id')
    unknown = sorted(set(selected) - set(schema))
    if unknown:
        raise KeyError(f'{path}: unknown or unavailable ASCII columns: {unknown}')
    # pandas C parser supports whitespace separation and is faster than python engine.
    df = pd.read_csv(path, sep=r'\s+', header=None, names=schema,
                     usecols=selected, comment='#', engine='c',
                     dtype={'track_id': str, 'halo_catalogue_index': 'string',
                            'host_fof_halo_id': 'string', 'descendant_id': 'string'})
    if 'track_id' in df:
        df['track_id'] = _integer_ids(df['track_id'], f'{path}: track_id')
        if validate_track_ids:
            _validate_unique_ids(df, 'track_id', str(path))
    for id_column in ('halo_catalogue_index', 'host_fof_halo_id', 'descendant_id'):
        if id_column in df:
            # Other identifiers may have negative sentinel values, but not nulls.
            df[id_column] = _integer_ids(df[id_column], f'{path}: {id_column}')
    if track_ids is not None:
        targets = set(int(tid) for tid in track_ids)
        df = df.loc[df['track_id'].isin(targets)].copy()
    match = ASCII_PATTERN.fullmatch(path.name)
    df.attrs['source'] = 'ASCII'
    df.attrs['path'] = str(path)
    df.attrs['redshift_from_filename'] = float(match.group('z')) if match else None
    df.attrs['units_validated'] = False
    return df.reset_index(drop=True)


def read_dor_catalogue(path: Path | str | None = None,
                       id_column: str | None = None,
                       dor_column: str = 'DoR_t95') -> pd.DataFrame:
    """Load the SFH/DoR CSV without guessing identifier semantics.

    Default ID detection follows the original script's precedence, but raises
    instead of assuming the first column. Duplicates are rejected.
    """
    path = _existing_file(path if path is not None else INPUTS['dor_sfh_csv'])
    df = pd.read_csv(path, low_memory=False)
    if id_column is None:
        id_column = next((name for name in
                          ('subhalo_id', 'HaloCatalogueIndex', 'subhaloId',
                           'HaloIndex', 'track_id', 'TrackId') if name in df), None)
    if id_column is None or id_column not in df:
        raise CatalogueError(f'{path}: no recognised identifier column; specify id_column')
    if dor_column not in df:
        raise CatalogueError(f'{path}: required DoR column {dor_column!r} absent')
    df = df.copy()
    df[id_column] = _integer_ids(df[id_column], f'{path}: {id_column}')
    _validate_unique_ids(df, id_column, str(path))
    df[dor_column] = pd.to_numeric(df[dor_column], errors='coerce')
    df.attrs['identifier_column'] = id_column
    df.attrs['path'] = str(path)
    return df


def read_bh_correction_catalogue(path: Path | str | None = None) -> pd.DataFrame:
    """Load BH correction table, without applying any corrections."""
    path = _existing_file(path if path is not None else INPUTS['bh_correction_csv'])
    df = pd.read_csv(path)
    required = {'track_id', 'corrected_bh_mass'}
    if not required.issubset(df.columns):
        raise CatalogueError(f'{path}: missing columns {sorted(required - set(df.columns))}')
    df['track_id'] = _integer_ids(df['track_id'], f'{path}: track_id')
    _validate_unique_ids(df, 'track_id', str(path))
    return df


def inspect_exsitu_catalogue(path: Path | str | None = None) -> dict:
    """Inspect ex-situ HDF5 metadata; DO NOT infer identifier columns."""
    path = _existing_file(path if path is not None else INPUTS['exsitu_z0_hdf5'])
    result = {'path': str(path), 'datasets': {}}
    with h5py.File(path, 'r') as handle:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                result['datasets'][name] = {
                    'shape': tuple(obj.shape), 'dtype': str(obj.dtype),
                    'attributes': {str(k): str(v) for k, v in obj.attrs.items()},
                }
        handle.visititems(visit)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect available COLIBRE catalogues')
    parser.add_argument('--inventory', action='store_true', help='List available ASCII and SOAP files')
    parser.add_argument('--check-ascii', action='store_true', help='Read one ASCII file and report its schema')
    parser.add_argument('--ascii-file', type=Path, default=None)
    args = parser.parse_args()
    if not (args.inventory or args.check_ascii):
        parser.print_help()
        return
    if args.inventory:
        ascii_files = discover_ascii_catalogues()
        soap_files = discover_soap_catalogues()
        print(f'ASCII catalogues: {len(ascii_files)}')
        print(ascii_files.to_string(index=False))
        print(f'\nSOAP-HBT catalogues: {len(soap_files)}')
        print(soap_files.to_string(index=False))
        print('\nNote: SOAP snapshot numbers are not redshifts.')
    if args.check_ascii:
        path = args.ascii_file
        if path is None:
            inventory = discover_ascii_catalogues()
            if inventory.empty:
                raise FileNotFoundError('No ASCII catalogues discovered')
            path = inventory.iloc[0]['path']
        schema = ascii_schema(path)
        frame = read_ascii_snapshot(path, columns=['track_id', 'stellar_mass', 'rhalf_stars'])
        print(f'ASCII file: {path}')
        print(f'Schema columns: {len(schema)}; rows: {len(frame)}')
        print(f'Unique TrackIDs: {frame.track_id.nunique()}')
        print(f'Redshift filename label: {frame.attrs["redshift_from_filename"]}')
        print('Units: NOT VERIFIED; no conversions applied')


if __name__ == '__main__':
    main()
