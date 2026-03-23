#!/usr/bin/env python3
"""
comparison_soap_obs.py

Reads SOAP catalogues for L0200N3008/THERMAL_AGN (z=0 snapshot),
loads the extra hyades file in SOAP-HBT/extra/ (auto-detect),
merges velocity dispersions, computes simple JAM mass estimates,
and produces a 3-panel figure similar to the attached observational plots.

Usage:
    python3 comparison_soap_obs.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats import gaussian_kde
import h5py


# --- Adjust these to your environment if needed ---
MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN/'
SOAP_EXTRA_DIR = os.path.join(MODEL_DIR, 'SOAP-HBT', 'extra')
SNAP_FILE = '0127'   # z=0 in your snippet
ZTARGET = 0.0

OUTFIG = 'soap_vs_observations_comparison.png'

# import the project's helper 'common' module (assumes it's on PYTHONPATH)
import common

# --- utility: find the hyades file in the SOAP extra directory ---
def find_hyades_file(extra_dir):
    files = glob.glob(os.path.join(extra_dir, '*'))
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {extra_dir}")
    # If only one file exists, use it. Otherwise try to pick a likely hyades file.
    if len(files) == 1:
        return files[0]
    # heuristics: look for 'hyades' or 'sigma' in name
    for f in files:
        ln = os.path.basename(f).lower()
        if 'hyades' in ln or 'sigma' in ln or 'veldisp' in ln or 'sigmae' in ln:
            return f
    # fallback: return the first file (but warn)
    print("Warning: multiple files found in extra/; using the first:", files[0])
    return files[0]

def hdf5_list_datasets(hf, prefix=''):
    """Recursively list (path, shape, dtype) for datasets in an h5py.File or group"""
    out = []
    for key in hf:
        obj = hf[key]
        path = prefix + '/' + key if prefix else key
        if isinstance(obj, h5py.Dataset):
            out.append((path, obj.shape, obj.dtype))
        elif isinstance(obj, h5py.Group):
            out.extend(hdf5_list_datasets(obj, path))
    return out

def read_hyades_file(path):
    """Read hyades 'extra' file. If HDF5, try to extract common ID and sigma datasets.
       Returns a pandas.DataFrame with columns at least ['sgn_hy', 'sigma_kms'].
    """
    # First, quick binary check: many HDF5 files start with byte 0x89 'HDF'
    try:
        with open(path, 'rb') as f:
            head = f.read(8)
        # HDF5 signature is b'\x89HDF\r\n\x1a\n'
        if head.startswith(b'\x89HDF'):
            is_hdf5 = True
        else:
            is_hdf5 = False
    except Exception:
        is_hdf5 = False

    if is_hdf5:
        try:
            with h5py.File(path, 'r') as hf:
                datasets = hdf5_list_datasets(hf)
                print("HDF5 datasets (path, shape, dtype):")
                for p, sh, dt in datasets:
                    print("  ", p, sh, dt)

                # Heuristics: try to find an ID column and a sigma column
                possible_id_names = ['HaloCatalogueIndex', 'HaloIndex', 'SubhaloIndex', 'id', 'HaloId', 'Halo_ID', 'SubhaloID', 'SubhaloIndex']
                possible_sigma_names = [
                    'StellarProjectedVelocityDispersion', 'StellarProjectedVelocityDispersion23',
                    'StellarProjectedVelocityDispersion23', 'StellarProjectedVelocityDispersion',
                    'StellarVelocityDispersion', 'ProjectedStellarVelocityDispersion',
                    'StellarProjectedVelocityDispersion23', 'sigma', 'sigma_e'
                ]

                # flatten dataset names to a simple mapping: basename->fullpath
                base_to_paths = {}
                for p, sh, dt in datasets:
                    base = os.path.basename(p)
                    base_to_paths.setdefault(base, []).append(p)

                # find id path
                id_path = None
                for cand in possible_id_names:
                    if cand in base_to_paths:
                        id_path = base_to_paths[cand][0]
                        break
                # if still None, try exact matches by ignoring case
                if id_path is None:
                    for base, paths in base_to_paths.items():
                        if base.lower() in [c.lower() for c in possible_id_names]:
                            id_path = paths[0]
                            break

                # find sigma path
                sigma_path = None
                for cand in possible_sigma_names:
                    if cand in base_to_paths:
                        sigma_path = base_to_paths[cand][0]
                        break
                if sigma_path is None:
                    for base, paths in base_to_paths.items():
                        if any(cand.lower() == base.lower() for cand in possible_sigma_names):
                            sigma_path = paths[0]
                            break

                # if not found by basename heuristics, try substring matching on full path
                if id_path is None:
                    for p,_,_ in datasets:
                        if any(cand.lower() in p.lower() for cand in possible_id_names):
                            id_path = p
                            break
                if sigma_path is None:
                    for p,_,_ in datasets:
                        if any(cand.lower() in p.lower() for cand in possible_sigma_names):
                            sigma_path = p
                            break

                if id_path is None or sigma_path is None:
                    # list potential numeric arrays for manual inspection
                    numeric_ds = [p for p,sh,dt in datasets if (len(sh) > 0 and dt.kind in ('i', 'u', 'f'))]
                    raise RuntimeError(
                        "Could not auto-detect ID or sigma dataset. "
                        "Candidates (numeric datasets):\n" + "\n".join(numeric_ds)
                        + "\nPlease inspect the HDF5 file and provide the dataset names."
                    )

                # read arrays
                id_arr = hf[id_path][()]
                sigma_arr = hf[sigma_path][()]
                # try to coerce shapes: if id_arr is scalar per group, ensure 1D
                id_arr = np.asarray(id_arr).ravel()
                sigma_arr = np.asarray(sigma_arr).ravel()

                # if arrays lengths mismatch, try matching by subhalo index ordering; otherwise error
                if id_arr.shape[0] != sigma_arr.shape[0]:
                    print("Warning: id length", id_arr.shape, "sigma length", sigma_arr.shape)
                    # attempt to read both into dataframe by index if one is a table-like structured dataset
                    # simplest fallback: stack into DataFrame using min length
                    nmin = min(id_arr.shape[0], sigma_arr.shape[0])
                    id_arr = id_arr[:nmin]
                    sigma_arr = sigma_arr[:nmin]

                df = pd.DataFrame({'sgn_hy': id_arr, 'sigma_kms': sigma_arr})
                # try to ensure sigma is in km/s; if units are cm/s or m/s convert if necessary:
                # we cannot automatically know units; assume km/s as SOAP lists L/t fields in km/s
                return df

        except Exception as e:
            raise RuntimeError("HDF5 read error: " + str(e))

    # FALLBACK: not HDF5 — try as text like before
    # try pandas read_table and read_csv and numpy loadtxt
    try:
        df = pd.read_table(path, sep=r'\s+', comment='#', engine='python')
        print("Read hyades as whitespace-delimited table, columns:", df.columns.tolist())
        return df
    except Exception as e:
        print("read_table failed:", e)
    try:
        df = pd.read_csv(path, sep=None, engine='python', comment='#')
        print("Read hyades with read_csv, columns:", df.columns.tolist())
        return df
    except Exception as e:
        print("read_csv failed:", e)
    try:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        cols = [f'c{i}' for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
        print("Read hyades with np.loadtxt, created columns:", cols)
        return df
    except Exception as e:
        raise RuntimeError("Could not read hyades file (not HDF5 and text parse failed): " + str(e))

# --- find id and sigma columns in hyades DF ---
def detect_id_sigma_columns(df):
    # Common candidate names
    id_candidates = ['sgn', 'halo', 'haloindex', 'halo_id', 'haloid', 'id', 'HaloCatalogueIndex']
    sigma_candidates = ['sigma', 'sigma_e', 'sigmae', 'veldisp', 'vel_disp', 'sigma_kms', 'sigma_km_s']
    cols = [c.lower() for c in df.columns.astype(str)]
    # try to detect id col
    id_col = None
    for cand in id_candidates:
        for c in df.columns:
            if cand == str(c).lower():
                id_col = c
                break
        if id_col is not None:
            break
    # try to detect sigma col
    sigma_col = None
    for cand in sigma_candidates:
        for c in df.columns:
            if cand == str(c).lower():
                sigma_col = c
                break
        if sigma_col is not None:
            break
    # If not detected, fallback to first two numeric columns
    if id_col is None or sigma_col is None:
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if len(numeric_cols) >= 2:
            if id_col is None:
                id_col = numeric_cols[0]
            if sigma_col is None:
                sigma_col = numeric_cols[1]
    return id_col, sigma_col

# --- main processing ---
def main():
    print("Starting SOAP -> observational-style figure generation")

    # --- read SOAP HDF groups using your 'common' helper (as in snippet) ---
    fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')}
    # include metallicity candidates so we can plot [Z/H] if present
    fields = {
        'ExclusiveSphere/50kpc': (
            'StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'CentreOfMass',
            'MassWeightedMeanStellarAge', 'LuminosityWeightedMeanStellarAge',
            'StellarMassFractionInMetals'
        )
    }
    fields_proj = {'ProjectedAperture/50kpc/projz': ('StellarMass', 'HalfMassRadiusStars')}

    print("Reading SOAP fields from model_dir:", MODEL_DIR, " snap:", SNAP_FILE)
    h5data_groups = common.read_group_data_colibre(MODEL_DIR, SNAP_FILE, fields)
    h5data_idgroups = common.read_group_data_colibre(MODEL_DIR, SNAP_FILE, fields_sgn)
    h5data_groups_proj = common.read_group_data_colibre(MODEL_DIR, SNAP_FILE, fields_proj)

    (m30, sfr30, r50, cp, stellarage_mass, stellarage_lum,
     zmass) = h5data_groups
    (m30_proj, r50_proj) = h5data_groups_proj
    (sgn, is_central, desc_id, track_id) = h5data_idgroups

    # convert units (same conversions as your snippet)
    Lu = 3.086e+24 / (3.086e+24)  # cMpc scaling placeholder (kept for consistency)
    Mu = 1.988e+43 / (1.989e+33)
    tu = 3.086e+19 / (3.154e+7)
    comov_to_physical_length = 1.0 / (1.0 + ZTARGET)

    m30 = m30 * Mu
    m30_proj = m30_proj * Mu
    sfr30 = sfr30 * Mu / tu
    r50 = r50 * Lu * comov_to_physical_length * 1e3  # kpc
    r50_proj = r50_proj * Lu * comov_to_physical_length * 1e3  # kpc
    stellarage_mass = stellarage_mass * tu  # years
    stellarage_lum = stellarage_lum * tu   # years
    # metallicity fields (if they are present in file) are likely unitless [Z/H] or Zmass;
    # we will assume the SOAP-provided values are already [Z] or something similar.
    # If they are in fraction, further conversion may be needed.

    # Select galaxies: stellar mass >= 1e9 (same selection)
    select = np.where(m30 >= 1e9)[0]
    if len(select) == 0:
        raise RuntimeError("No galaxies found with stellar mass >= 1e9")

    print("Selected", len(select), "galaxies with M* >= 1e9")

    # Build DataFrame
    df = pd.DataFrame({
        'sgn': sgn[select].astype(int),
        'is_central': is_central[select],
        'x': cp[select, 0],
        'y': cp[select, 1],
        'z': cp[select, 2],
        'Mstar_50kpc': m30[select],
        'Mstar_ap_proj_50kpc': m30_proj[select],
        'SFR_50kpc': sfr30[select],
        'Re_kpc': r50[select],
        'Re_proj_kpc': r50_proj[select],
        'age_mass_yr': stellarage_mass[select],
        'age_lum_yr': stellarage_lum[select],
        'desc_id': desc_id[select],
        'track_id': track_id[select]
    })

    # add metallicity if present (columns may be None depending on SOAP)
    try:
        df['Z_mass'] = zmass[select]
        print("Added metallicity columns from SOAP.")
    except Exception:
        print("No metallicity fields available in SOAP read; [Z/H] column will be NaN.")

    # --- read hyades file and merge ---
    hyades_path = find_hyades_file(SOAP_EXTRA_DIR)
    hy_df = read_hyades_file(hyades_path)

    id_col, sigma_col = detect_id_sigma_columns(hy_df)
    if id_col is None or sigma_col is None:
        print("Hyades file columns:", hy_df.columns.tolist())
        raise RuntimeError("Could not detect ID and sigma columns automatically in the hyades file.")

    print("Using hyades id column:", id_col, " sigma column:", sigma_col)
    # standardize names
    hy_df = hy_df.rename(columns={id_col: 'sgn_hy', sigma_col: 'sigma_kms'})
    # ensure integer id type if possible
    try:
        hy_df['sgn_hy'] = hy_df['sgn_hy'].astype(int)
    except Exception:
        pass

    # merge
    df_merged = df.merge(hy_df[['sgn_hy', 'sigma_kms']], left_on='sgn', right_on='sgn_hy', how='left')
    n_missing_sigma = df_merged['sigma_kms'].isna().sum()
    print(f"Merged hyades: {n_missing_sigma} galaxies missing sigma (out of {len(df_merged)})")

    # Drop rows missing sigma because the observational comparison needs sigma
    df_merged = df_merged.dropna(subset=['sigma_kms']).copy()
    print("After dropping missing sigma rows:", len(df_merged), "galaxies remain")

    # Derived quantities
    df_merged['lgMstar'] = np.log10(df_merged['Mstar_50kpc'])
    df_merged['lgRe'] = np.log10(df_merged['Re_proj_kpc'])
    df_merged['lgSigma_e'] = np.log10(df_merged['sigma_kms'])
    df_merged['lgAge_yr'] = np.log10(df_merged['age_lum_yr'])
    # approximate [Z/H] from the luminosity-weighted metallicity if present
    if 'Z_mass_lw' in df_merged.columns:
        df_merged['ZH_lw'] = df_merged['Z_mass_lw']
    else:
        df_merged['ZH_lw'] = np.nan

    # Compute JAM-like mass estimate: M_JAM = k * Re * sigma^2 / G
    # Re must be in cm, sigma in cm/s -> result in grams -> convert to Msun
    G_cgs = 6.67430e-8  # cm^3 / g / s^2
    Msun_g = 1.989e33
    # convert Re from kpc -> cm: 1 kpc = 3.0857e21 cm
    kpc_to_cm = 3.0857e21
    k = 5.0
    df_merged['M_JAM_g'] = k * (df_merged['Re_proj_kpc'] * kpc_to_cm) * (df_merged['sigma_kms'] * 1e5)**2 / G_cgs
    df_merged['M_JAM_msun'] = df_merged['M_JAM_g'] / Msun_g
    df_merged['lgM_JAM'] = np.log10(df_merged['M_JAM_msun'])

    # --- plotting: create three-panel figure similar to the observational layout ---
    fig = plt.figure(figsize=(10, 10))
    # top-left: lgM_JAM vs lgRe colored by lgAge
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1, rowspan=2)
    x = df_merged['lgM_JAM'].values
    y = df_merged['lgRe'].values
    c = df_merged['lgAge_yr'].values

    # density background (kde)
    try:
        xy = np.vstack([x, y])
        z_kde = gaussian_kde(xy)(xy)
        # sort by density so dense points are plotted beneath
        idx = z_kde.argsort()
        x_s, y_s, c_s, z_s = x[idx], y[idx], c[idx], z_kde[idx]
        sc = ax1.scatter(x_s, y_s, c=c_s, s=20, cmap='rainbow', marker='o', edgecolor='none', alpha=0.9)
    except Exception:
        sc = ax1.scatter(x, y, c=c, s=20, cmap='rainbow', edgecolor='none')
    # overplot contours from KDE (in x,y space)
    try:
        # grid for contour
        xi = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 100)
        yi = np.linspace(np.nanpercentile(y, 2), np.nanpercentile(y, 98), 100)
        xi_mesh, yi_mesh = np.meshgrid(xi, yi)
        positions = np.vstack([xi_mesh.ravel(), yi_mesh.ravel()])
        kde = gaussian_kde(np.vstack([x, y]))
        zi = np.reshape(kde(positions).T, xi_mesh.shape)
        cs = ax1.contour(xi_mesh, yi_mesh, zi, colors='k', linewidths=0.8, alpha=0.6)
    except Exception:
        pass

    ax1.set_xlabel(r'$\log_{10} M_{\mathrm{JAM}}\ [M_\odot]$')
    ax1.set_ylabel(r'$\log_{10} R_e^{\mathrm{maj}}\ [\mathrm{kpc}]$')
    cbar = fig.colorbar(sc, ax=ax1, pad=0.02)
    cbar.set_label(r'$\log_{10}(\mathrm{Age\ [yr]})$')

    # top-right: lgAge vs lgSigma_e colored by lgM_JAM
    ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1)
    x2 = df_merged['lgSigma_e'].values
    y2 = df_merged['lgAge_yr'].values
    c2 = df_merged['lgM_JAM'].values
    sc2 = ax2.scatter(x2, y2, c=c2, s=20, cmap='viridis', edgecolor='none')
    ax2.set_xlabel(r'$\log_{10} \sigma_e\ [\mathrm{km\,s^{-1}}]$')
    ax2.set_ylabel(r'$\log_{10} \mathrm{Age\ [yr]}$')
    cbar2 = fig.colorbar(sc2, ax=ax2, pad=0.02)
    cbar2.set_label(r'$\log_{10} M_{\mathrm{JAM}}$')

    # bottom-right: [Z/H] vs lgSigma_e colored by lgAge
    ax3 = plt.subplot2grid((3, 2), (1, 1), colspan=1, rowspan=2)
    x3 = df_merged['lgSigma_e'].values
    y3 = df_merged['ZH_lw'].values
    c3 = df_merged['lgAge_yr'].values
    # if metallicity array is all NaN, mark and skip plotting colored scatter
    if np.all(np.isnan(y3)):
        ax3.text(0.5, 0.5, 'No [Z/H] available in SOAP fields', ha='center')
        ax3.set_xlabel(r'$\log_{10} \sigma_e$')
        ax3.set_ylabel('[Z/H]')
    else:
        sc3 = ax3.scatter(x3, y3, c=c3, s=18, cmap='plasma', edgecolor='none')
        cbar3 = fig.colorbar(sc3, ax=ax3, pad=0.02)
        cbar3.set_label(r'$\log_{10}(\mathrm{Age\ [yr]})$')
        ax3.set_xlabel(r'$\log_{10} \sigma_e\ [\mathrm{km\,s^{-1}}]$')
        ax3.set_ylabel('[Z/H]')

    # aesthetics: tighten layout and save
    plt.tight_layout()
    fig.savefig(OUTFIG, dpi=300)
    print("Saved comparison figure to", OUTFIG)
    plt.show()

if __name__ == '__main__':
    main()