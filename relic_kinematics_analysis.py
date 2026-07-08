#!/usr/bin/env python3
"""
relic_kinematics_analysis.py

Standalone kinematics analysis for COLIBRE relic-candidate work.

What this script does
---------------------
1) Reads your DoR / candidate CSV and matches it to SOAP via HaloCatalogueIndex.
2) Loads z=0 SOAP galaxy properties (mass, radius, central/satellite, track id).
3) Loads kinematic summary quantities from the SOAP-HBT extra HDF5 file at the
   HalfMassRadiusStars aperture.
4) Computes v/sigma and a proxy for lambda_R when only aperture-integrated
   quantities are available.
5) Builds a matched kinematic table and several diagnostic plots.
6) Tests the question:
      At fixed stellar mass and compactness, are relics systematically more or
      less rotation-supported than non-relic galaxies, and does this depend on
      being a central/satellite or on ex-situ fraction?

Important caveat
----------------
The extra file you listed contains aperture-integrated quantities, not a spatially
resolved stellar kinematic map. Therefore the script cannot compute the true
observational lambda_R definition exactly. It computes a conservative proxy from
aperture-integrated rotational velocity and velocity dispersion:

    lambda_R_proxy = |V_rot| / sqrt(V_rot^2 + sigma^2)

This is useful for internal comparisons, but it should be labelled as a proxy in
any paper or figure caption. If you later obtain projected particle/spaxel data,
replace the proxy with the full lambda_R calculation.

Outputs
-------
- out_kinematics/matched_kinematics_table.csv
- out_kinematics/kinematics_summary_by_mass_bin.csv
- out_kinematics/kinematics_2d_mass_compactness_comparison.csv
- out_kinematics/*.png diagnostic figures

Dependencies
------------
- numpy, pandas, matplotlib, h5py
- scipy (for KDTree)
- your local common.py helper with read_group_data_colibre
"""
from __future__ import annotations

import os
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree

import common

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 13,
})

# ----------------------------- CONFIG ---------------------------------
CSV_IN = "sfh_times_all_with_DoR_variants_corrected.csv.gz"
MODEL_NAME = "L0200N3008/THERMAL_AGN/"
MODEL_DIR = "/mnt/su3-pro/colibre/" + MODEL_NAME
SNAP_FILE = "0127"  # z=0
EXTRA_H5 = os.path.join(MODEL_DIR, "SOAP-HBT", "extra", f"halo_properties_{SNAP_FILE}.hdf5")
EXSITU_H5 = "/mnt/su3ctm/kproctor/ForMax/exsitu_summary_SnapNum_127.hdf5"  # optional
OUTDIR = "out_kinematics"

# Your existing compactness cut from the relic work
COMPACTNESS_CUT = 9.72
EXTREME_DOR = 0.60

# Mass threshold for the analysis sample
MIN_STELLAR_MASS = 1e9

# Binning choices for the fixed-mass / fixed-compactness comparison
MASS_BIN_WIDTH = 0.25
COMPACT_BIN_WIDTH = 0.20
MIN_PER_BIN = 5

# If you want luminosity-weighted kinematics instead of mass-weighted ones,
# set USE_LUM_WEIGHTED = True.
USE_LUM_WEIGHTED = False

# --------------------------- SMALL HELPERS -----------------------------
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, outpath: str | Path) -> None:
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath)


def finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(x, dtype=float))


def as_scalar_velocity(arr: np.ndarray) -> np.ndarray:
    """Return a scalar rotational velocity from a flexible HDF5 array."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return np.abs(a)
    if a.ndim == 2:
        # If a vector is stored, use its norm; if multiple components are stored,
        # this still returns a non-negative scalar support measure.
        return np.linalg.norm(a, axis=1)
    raise ValueError(f"Unsupported velocity array shape: {a.shape}")


def as_scalar_sigma(arr: np.ndarray) -> np.ndarray:
    """Return a scalar velocity dispersion from a flexible HDF5 array.

    Accepted patterns:
      - shape (N,): already scalar
      - shape (N, 3): treated as three cylindrical components
      - shape (N, 6): treated as flattened symmetric 3x3 tensor, diagonal at 0,3,5
      - shape (N, 9): treated as row-major 3x3 tensor, diagonal at 0,4,8
      - shape (N, k): fallback to RMS over components
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return np.abs(a)
    if a.ndim != 2:
        raise ValueError(f"Unsupported dispersion array shape: {a.shape}")

    n, k = a.shape
    if k == 3:
        return np.sqrt(np.nanmean(a**2, axis=1))
    if k == 6:
        # common symmetric packing: xx, xy, xz, yy, yz, zz
        diag = np.column_stack([a[:, 0], a[:, 3], a[:, 5]])
        return np.sqrt(np.nanmean(diag**2, axis=1))
    if k == 9:
        diag = np.column_stack([a[:, 0], a[:, 4], a[:, 8]])
        return np.sqrt(np.nanmean(diag**2, axis=1))
    return np.sqrt(np.nanmean(a**2, axis=1))


def estimate_lambda_r_proxy(vrot: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Proxy for lambda_R using aperture-integrated V and sigma only."""
    vrot = np.asarray(vrot, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    out = np.full_like(vrot, np.nan, dtype=float)
    ok = np.isfinite(vrot) & np.isfinite(sigma) & (sigma >= 0)
    out[ok] = np.abs(vrot[ok]) / np.sqrt(vrot[ok] ** 2 + sigma[ok] ** 2)
    return out


def load_h5_row_aligned_dataset(h5f: h5py.File, candidates: Iterable[str], row_idx: np.ndarray) -> np.ndarray:
    """Try candidate dataset paths and return the first available one, row-selected."""
    last_err = None
    for key in candidates:
        try:
            ds = h5f[key]
            arr = np.asarray(ds[row_idx])
            return arr
        except Exception as e:
            last_err = e
            continue
    raise KeyError(f"None of these datasets were found/read: {list(candidates)}; last error: {last_err}")


def load_h5_optional_row_aligned_dataset(h5f: h5py.File, candidates: Iterable[str], row_idx: np.ndarray, fill_value=np.nan) -> np.ndarray:
    try:
        return load_h5_row_aligned_dataset(h5f, candidates, row_idx)
    except Exception:
        return None


def loess_like_2d_binned_summary(x: np.ndarray, y: np.ndarray, z: np.ndarray, xbins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple percentile-bin median + 16/84 summary for 1D plots."""
    xc = 0.5 * (xbins[:-1] + xbins[1:])
    med = np.full_like(xc, np.nan, dtype=float)
    lo = np.full_like(xc, np.nan, dtype=float)
    hi = np.full_like(xc, np.nan, dtype=float)
    ok = finite(x) & finite(y) & finite(z)
    for i in range(len(xc)):
        sel = ok & (x >= xbins[i]) & (x < xbins[i + 1])
        if np.sum(sel) >= MIN_PER_BIN:
            med[i] = np.nanmedian(z[sel])
            lo[i] = np.nanpercentile(z[sel], 16)
            hi[i] = np.nanpercentile(z[sel], 84)
    return xc, med, lo, hi


def plot_binned_median(ax, x, y, nbins=14, label=None, color="black"):
    ok = finite(x) & finite(y)
    if np.sum(ok) < MIN_PER_BIN:
        return
    bins = np.unique(np.percentile(x[ok], np.linspace(0, 100, nbins)))
    if bins.size < 2:
        return
    xc = 0.5 * (bins[:-1] + bins[1:])
    med = np.full_like(xc, np.nan, dtype=float)
    lo = np.full_like(xc, np.nan, dtype=float)
    hi = np.full_like(xc, np.nan, dtype=float)
    for i in range(len(xc)):
        sel = ok & (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(sel) >= MIN_PER_BIN:
            med[i] = np.nanmedian(y[sel])
            lo[i] = np.nanpercentile(y[sel], 16)
            hi[i] = np.nanpercentile(y[sel], 84)
    fm = np.isfinite(med)
    if np.any(fm):
        ax.plot(xc[fm], med[fm], "o-", color=color, lw=1.8, ms=4, label=label)
        ax.fill_between(xc[fm], lo[fm], hi[fm], color=color, alpha=0.18)


def scatter_with_colour(ax, x, y, c, cmap="viridis", s=12, alpha=0.85, vmin=None, vmax=None):
    ok = finite(x) & finite(y) & finite(c)
    if np.sum(ok) == 0:
        return None
    sc = ax.scatter(x[ok], y[ok], c=c[ok], cmap=cmap, s=s, alpha=alpha, edgecolors="none",
                    vmin=vmin, vmax=vmax)
    return sc


def compute_2d_bin_table(df: pd.DataFrame, value_col: str, group_col: str, mass_col: str = "logM", compact_col: str = "compactness") -> pd.DataFrame:
    """Return a 2D (mass, compactness) bin table split by group_col values.

    The group_col is typically relic_flag or central_flag.
    """
    work = df[[mass_col, compact_col, value_col, group_col]].copy()
    ok = work[mass_col].notna() & work[compact_col].notna() & work[value_col].notna() & work[group_col].notna()
    work = work.loc[ok].copy()
    if work.empty:
        return pd.DataFrame()

    mmin = math.floor(work[mass_col].min() / MASS_BIN_WIDTH) * MASS_BIN_WIDTH
    mmax = math.ceil(work[mass_col].max() / MASS_BIN_WIDTH) * MASS_BIN_WIDTH
    cmin = math.floor(work[compact_col].min() / COMPACT_BIN_WIDTH) * COMPACT_BIN_WIDTH
    cmax = math.ceil(work[compact_col].max() / COMPACT_BIN_WIDTH) * COMPACT_BIN_WIDTH
    mbins = np.arange(mmin, mmax + 1e-9, MASS_BIN_WIDTH)
    cbins = np.arange(cmin, cmax + 1e-9, COMPACT_BIN_WIDTH)

    rows = []
    groups = sorted(work[group_col].dropna().unique().tolist())
    for i in range(len(mbins) - 1):
        for j in range(len(cbins) - 1):
            sel_bin = (
                (work[mass_col] >= mbins[i]) & (work[mass_col] < mbins[i + 1]) &
                (work[compact_col] >= cbins[j]) & (work[compact_col] < cbins[j + 1])
            )
            if np.sum(sel_bin) < MIN_PER_BIN:
                continue
            row = {
                "mass_lo": mbins[i], "mass_hi": mbins[i + 1], "mass_center": 0.5 * (mbins[i] + mbins[i + 1]),
                "compact_lo": cbins[j], "compact_hi": cbins[j + 1], "compact_center": 0.5 * (cbins[j] + cbins[j + 1]),
                "n_total": int(np.sum(sel_bin)),
            }
            for g in groups:
                s = sel_bin & (work[group_col] == g)
                row[f"n_{g}"] = int(np.sum(s))
                row[f"med_{g}"] = float(np.nanmedian(work.loc[s, value_col])) if np.sum(s) >= MIN_PER_BIN else np.nan
            if len(groups) == 2:
                g0, g1 = groups
                row["delta_med"] = row.get(f"med_{g0}", np.nan) - row.get(f"med_{g1}", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


# -------------------------- READ INPUTS --------------------------------
def main() -> None:
    ensure_dir(OUTDIR)
    ensure_dir(Path(OUTDIR) / "figs")

    if not os.path.exists(CSV_IN):
        raise SystemExit(f"CSV not found: {CSV_IN}")
    if not os.path.exists(EXTRA_H5):
        raise SystemExit(f"Extra HDF5 not found: {EXTRA_H5}")

    print("Reading CSV:", CSV_IN)
    df = pd.read_csv(CSV_IN, low_memory=False)

    # identify ID column
    id_col = None
    for c in ("subhalo_id", "HaloCatalogueIndex", "subhaloId", "HaloIndex", "track_id", "TrackId"):
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]
        print("Warning: no obvious ID column found; using", id_col)

    sid = pd.to_numeric(df[id_col].astype(str).str.replace("\r", "").str.strip(), errors="coerce")
    bad = int(np.isnan(sid).sum())
    if bad > 0:
        print(f"Warning: {bad} rows have non-numeric IDs and will be dropped.")
    df = df.loc[np.isfinite(sid)].copy()
    df["subhalo_id"] = sid[np.isfinite(sid)].astype(np.int64)
    df_ucmg = df.set_index("subhalo_id", drop=False)

    dor_cols = [c for c in ("DoR_t90", "DoR_t95", "DoR_t998", "DoR_tfin", "dor", "DoR", "DoR_choice", "DoR_csv") if c in df_ucmg.columns]
    if not dor_cols:
        for c in df_ucmg.columns:
            if c.lower().startswith("dor"):
                dor_cols.append(c)
    if not dor_cols:
        raise SystemExit("No DoR-like column found in CSV.")
    primary_dor_col = dor_cols[0]
    print("Primary DoR column:", primary_dor_col)

    # SOAP fields
    print("Reading SOAP group data...")
    fields = {
        "ExclusiveSphere/50kpc": (
            "StellarMass", "HalfMassRadiusStars", "StarFormationRate",
            "MassWeightedMeanStellarAge", "LuminosityWeightedMeanStellarAge",
            "LinearMassWeightedIronOverHydrogenOfStars",
            "LinearMassWeightedMagnesiumOverHydrogenOfStars",
            "MostMassiveBlackHoleMass", "StellarMassFractionInMetals",
        ),
        "InputHalos": ("HaloCatalogueIndex", "IsCentral", "HBTplus/DescendantTrackId", "HBTplus/TrackId"),
        "SOAP": ("HostHaloIndex",),
    }
    h5data = common.read_group_data_colibre(MODEL_DIR, SNAP_FILE, fields)
    (m30, r50, sfr30, age_mass, age_lum, fe_lin, mg_lin, bh_mass_raw, zstar_raw, halo_idx, is_central, desc_id, track_id, host_halo_index) = h5data

    # units and basic derived quantities
    Mu = 1.988e43 / 1.989e33
    tu = 3.086e19 / 3.154e7
    m30 = np.asarray(m30, dtype=float) * Mu
    r50 = np.asarray(r50, dtype=float) * 1e3  # z=0 so no comoving-to-physical factor
    sfr30 = np.asarray(sfr30, dtype=float) * Mu / tu
    bh_mass_raw = np.asarray(bh_mass_raw, dtype=float) * Mu
    halo_idx = np.asarray(halo_idx, dtype=np.int64)
    track_id = np.asarray(track_id, dtype=np.int64)
    is_central = np.asarray(is_central)

    # Select galaxies used in the relic work
    mask_sel = (m30 >= MIN_STELLAR_MASS) & (m30 > 0) & (r50 > 0) & np.isfinite(m30) & np.isfinite(r50)
    row_idx = np.flatnonzero(mask_sel)
    print(f"Selected SOAP galaxies for kinematics: {len(row_idx)}")

    # selected arrays aligned to the extra HDF5 rows we will read
    m_sel = m30[mask_sel]
    r_sel = r50[mask_sel]
    halo_sel = halo_idx[mask_sel]
    track_sel = track_id[mask_sel]
    cen_sel = is_central[mask_sel]

    logM = np.log10(m_sel)
    logR = np.log10(r_sel)
    compactness = logM - 1.5 * logR

    # DoR aligned to selected SOAP rows by HaloCatalogueIndex
    dor_series = pd.Series(df_ucmg[primary_dor_col].astype(float).to_numpy(), index=df_ucmg["subhalo_id"].to_numpy(dtype=np.int64))
    dor_selected = dor_series.reindex(halo_sel.astype(np.int64)).to_numpy(dtype=float)
    matched_positions = np.where(np.isfinite(dor_selected))[0]
    print(f"Matched UCMG CSV -> selected SOAP rows: {len(matched_positions)}")

    # ex-situ lookup (optional)
    exsitu_lookup: Dict[int, float] = {}
    if os.path.exists(EXSITU_H5):
        try:
            with h5py.File(EXSITU_H5, "r") as fh:
                if "stars" in fh:
                    data = np.asarray(fh["stars"])
                    if data.ndim == 2 and data.shape[1] >= 4:
                        # pick the ID column with the best overlap with halo indices
                        overlaps = []
                        for c in (0, 1, 2):
                            try:
                                ids = data[:, c].astype(np.int64)
                                overlaps.append(np.intersect1d(ids, halo_sel).size)
                            except Exception:
                                overlaps.append(-1)
                        keycol = int(np.argmax(overlaps))
                        ids = data[:, keycol].astype(np.int64)
                        exfrac = data[:, 3].astype(float)
                        exsitu_lookup = dict(zip(ids.tolist(), exfrac.tolist()))
                        print(f"Loaded ex-situ entries from {EXSITU_H5}; using column {keycol}.")
        except Exception as e:
            print("Warning: failed to read ex-situ file:", e)
    else:
        print("Ex-situ file not found; ex-situ diagnostics will be skipped.")

    exsitu_series = pd.Series(exsitu_lookup, dtype=float)
    exsitu_selected = exsitu_series.reindex(halo_sel.astype(np.int64)).to_numpy(dtype=float)

    # ------------------------- KINEMATICS READ -------------------------
    print("Reading kinematics from:", EXTRA_H5)
    with h5py.File(EXTRA_H5, "r") as fh:
        grp = fh["/ExclusiveSphere/HalfMassRadiusStars"]

        # Always read these two if available
        vrot_key = "StellarRotationalVelocityLuminosityWeighted" if USE_LUM_WEIGHTED else "StellarRotationalVelocity"
        sig_key = "StellarCylindricalVelocityDispersionLuminosityWeighted" if USE_LUM_WEIGHTED else "StellarCylindricalVelocityDispersion"
        kappa_key = "KappaCorotStarsLuminosityWeighted" if USE_LUM_WEIGHTED else "KappaCorotStars"
        d2t_key = "DiscToTotalMassRatioLuminosityWeighted" if USE_LUM_WEIGHTED else "DiscToTotalStellarMassFraction"
        if USE_LUM_WEIGHTED:
            d2t_key = "DiscToTotalLuminosityRatioLuminosityWeighted"

        vrot_raw = np.asarray(grp[vrot_key][row_idx])
        sig_raw = np.asarray(grp[sig_key][row_idx])
        kappa_raw = np.asarray(grp[kappa_key][row_idx]) if kappa_key in grp else None
        d2t_raw = np.asarray(grp[d2t_key][row_idx]) if d2t_key in grp else None

        # Optional extras
        angmom_raw = np.asarray(grp["AngularMomentumStarsLuminosityWeighted" if USE_LUM_WEIGHTED and "AngularMomentumStarsLuminosityWeighted" in grp else "AngularMomentumStars"][row_idx]) if ("AngularMomentumStars" in grp or "AngularMomentumStarsLuminosityWeighted" in grp) else None
        inertia_raw = None
        for k in ("StellarInertiaTensorReducedLuminosityWeighted" if USE_LUM_WEIGHTED else "StellarInertiaTensorReduced",
                  "StellarInertiaTensorLuminosityWeighted" if USE_LUM_WEIGHTED else "StellarInertiaTensor"):
            if k in grp:
                inertia_raw = np.asarray(grp[k][row_idx])
                inertia_key = k
                break
        else:
            inertia_key = None

        # scalar support measures
        vrot = as_scalar_velocity(vrot_raw)
        sigma = as_scalar_sigma(sig_raw)
        v_over_sigma = np.full_like(vrot, np.nan, dtype=float)
        ok_vs = finite(vrot) & finite(sigma) & (sigma > 0)
        v_over_sigma[ok_vs] = vrot[ok_vs] / sigma[ok_vs]
        lambda_proxy = estimate_lambda_r_proxy(vrot, sigma)

        # these may already be scalar arrays
        kappa = np.asarray(kappa_raw, dtype=float).reshape(-1) if kappa_raw is not None else np.full_like(vrot, np.nan, dtype=float)
        d2t = np.asarray(d2t_raw, dtype=float).reshape(-1) if d2t_raw is not None else np.full_like(vrot, np.nan, dtype=float)

        # A 3D flattening proxy from the inertia tensor if possible
        flattening_3d = np.full_like(vrot, np.nan, dtype=float)
        if inertia_raw is not None:
            arr = np.asarray(inertia_raw, dtype=float)
            try:
                if arr.ndim == 2 and arr.shape[1] in (3, 6, 9):
                    if arr.shape[1] == 3:
                        # treat as axis moments
                        vals = np.sort(np.abs(arr), axis=1)
                        c = np.maximum(vals[:, 0], 1e-12)
                        a = np.maximum(vals[:, 2], 1e-12)
                        flattening_3d = 1.0 - c / a
                    elif arr.shape[1] == 6:
                        mats = np.zeros((arr.shape[0], 3, 3), dtype=float)
                        mats[:, 0, 0] = arr[:, 0]
                        mats[:, 0, 1] = mats[:, 1, 0] = arr[:, 1]
                        mats[:, 0, 2] = mats[:, 2, 0] = arr[:, 2]
                        mats[:, 1, 1] = arr[:, 3]
                        mats[:, 1, 2] = mats[:, 2, 1] = arr[:, 4]
                        mats[:, 2, 2] = arr[:, 5]
                        eig = np.linalg.eigvalsh(mats)
                        eig = np.sort(np.abs(eig), axis=1)
                        flattening_3d = 1.0 - np.sqrt(np.maximum(eig[:, 0], 1e-12) / np.maximum(eig[:, 2], 1e-12))
                    else:
                        mats = arr.reshape(arr.shape[0], 3, 3)
                        eig = np.linalg.eigvalsh(mats)
                        eig = np.sort(np.abs(eig), axis=1)
                        flattening_3d = 1.0 - np.sqrt(np.maximum(eig[:, 0], 1e-12) / np.maximum(eig[:, 2], 1e-12))
            except Exception:
                pass

    # align kinematics to the matched subset of selected SOAP rows
    vrot_m = vrot[matched_positions]
    sigma_m = sigma[matched_positions]
    vos_m = v_over_sigma[matched_positions]
    lambda_proxy_m = lambda_proxy[matched_positions]
    kappa_m = kappa[matched_positions]
    d2t_m = d2t[matched_positions]
    flatten_m = flattening_3d[matched_positions]
    exsitu_m = exsitu_selected[matched_positions]
    dor_m = dor_selected[matched_positions]
    logM_m = logM[matched_positions]
    logR_m = logR[matched_positions]
    compact_m = compactness[matched_positions]
    track_m = track_sel[matched_positions]
    halo_m = halo_sel[matched_positions]
    cen_m = np.asarray(cen_sel[matched_positions]).astype(bool)

    # Build the matched kinematic table
    matched = pd.DataFrame({
        "subhalo_id": halo_m.astype(np.int64),
        "track_id": track_m.astype(np.int64),
        "logM": logM_m,
        "logR": logR_m,
        "compactness": compact_m,
        "DoR": dor_m,
        "is_central": cen_m,
        "is_relic": dor_m > EXTREME_DOR,
        "v_rot": vrot_m,
        "sigma": sigma_m,
        "v_over_sigma": vos_m,
        "lambda_R_proxy": lambda_proxy_m,
        "kappa_corot": kappa_m,
        "disc_to_total": d2t_m,
        "flattening_3d_proxy": flatten_m,
        "exsitu_frac": exsitu_m,
    })

    matched_csv = Path(OUTDIR) / "matched_kinematics_table.csv"
    matched.to_csv(matched_csv, index=False)
    print("Saved:", matched_csv)

    # Basic diagnostics
    n_all = len(matched)
    n_relic = int(matched["is_relic"].sum())
    print(f"Matched sample: {n_all} galaxies; relics: {n_relic}; non-relics: {n_all - n_relic}")
    print("Median v/sigma (all):", float(np.nanmedian(matched["v_over_sigma"])))
    print("Median lambda_R_proxy (all):", float(np.nanmedian(matched["lambda_R_proxy"])))
    print("Median kappa_corot (all):", float(np.nanmedian(matched["kappa_corot"])))

    # --------------------------- FIGURES ------------------------------
    figdir = Path(OUTDIR) / "figs"
    ensure_dir(figdir)

    # 1) Mass-size coloured by v/sigma
    fig, ax = plt.subplots(figsize=(8, 6))
    bg = ax.scatter(logM, logR, s=6, color="lightgrey", alpha=0.35, label="selected SOAP galaxies")
    sc = scatter_with_colour(ax, logM_m, logR_m, vos_m, cmap="viridis", s=18, alpha=0.9)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"$v/\sigma$")
    xm = np.linspace(np.nanmin(logM) - 0.1, np.nanmax(logM) + 0.1, 400)
    yr = (xm - COMPACTNESS_CUT) / 1.5
    ax.plot(xm, yr, "--", color="black", lw=1.5, label=fr"compactness cut $\lg\Sigma_{{1.5}}={COMPACTNESS_CUT}$")
    ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
    ax.set_ylabel(r"$\log(R_{1/2,\star}/\mathrm{kpc})$")
    ax.grid(True)
    ax.legend(fontsize=8)
    save_fig(fig, figdir / "mass_size_coloured_by_v_over_sigma.png")

    # 2) v/sigma vs stellar mass
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = scatter_with_colour(ax, logM_m, vos_m, dor_m, cmap="viridis", s=18, alpha=0.85)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("DoR")
    plot_binned_median(ax, logM_m, vos_m, label="all matched", color="black")
    ax.axhline(1.0, ls="--", lw=1.3, color="C1", label=r"$v/\sigma=1$")
    ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
    ax.set_ylabel(r"$v/\sigma$")
    ax.grid(True)
    ax.legend(fontsize=8)
    save_fig(fig, figdir / "v_over_sigma_vs_mass.png")

    # 3) v/sigma vs compactness
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = scatter_with_colour(ax, compact_m, vos_m, dor_m, cmap="viridis", s=18, alpha=0.85)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("DoR")
    plot_binned_median(ax, compact_m, vos_m, label="all matched", color="black")
    ax.axvline(COMPACTNESS_CUT, ls="--", lw=1.3, color="black", label="compactness cut")
    ax.axhline(1.0, ls="--", lw=1.0, color="C1")
    ax.set_xlabel(r"Compactness $\log(M_\star/R_{1/2,\star}^{1.5})$")
    ax.set_ylabel(r"$v/\sigma$")
    ax.grid(True)
    ax.legend(fontsize=8)
    save_fig(fig, figdir / "v_over_sigma_vs_compactness.png")

    # 4) proxy lambda_R vs mass
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = scatter_with_colour(ax, logM_m, lambda_proxy_m, dor_m, cmap="viridis", s=18, alpha=0.85)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("DoR")
    plot_binned_median(ax, logM_m, lambda_proxy_m, label="all matched", color="black")
    ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
    ax.set_ylabel(r"$\lambda_{R,\,proxy}$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(fontsize=8)
    save_fig(fig, figdir / "lambda_proxy_vs_mass.png")

    # 5) kappa_corot vs mass
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = scatter_with_colour(ax, logM_m, kappa_m, dor_m, cmap="viridis", s=18, alpha=0.85)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("DoR")
    plot_binned_median(ax, logM_m, kappa_m, label="all matched", color="black")
    ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
    ax.set_ylabel(r"$\kappa_{\rm corot}$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(fontsize=8)
    save_fig(fig, figdir / "kappa_corot_vs_mass.png")

    # 6) Compare relics vs non-relics at fixed mass + compactness (2D bin table)
    compare = matched.copy()
    compare["relic_flag"] = np.where(compare["is_relic"], "relic", "nonrelic")
    table_2d = compute_2d_bin_table(compare, value_col="v_over_sigma", group_col="relic_flag")
    if not table_2d.empty:
        table_2d_csv = Path(OUTDIR) / "kinematics_2d_mass_compactness_comparison.csv"
        table_2d.to_csv(table_2d_csv, index=False)
        print("Saved:", table_2d_csv)

        # heatmap of delta median v/sigma (relic - nonrelic)
        if "delta_med" in table_2d.columns and table_2d["delta_med"].notna().any():
            pivot = table_2d.pivot(index="compact_center", columns="mass_center", values="delta_med")
            fig, ax = plt.subplots(figsize=(9, 6))
            im = ax.imshow(
                pivot.values,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
                cmap="coolwarm",
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"Median $(v/\sigma)_{\rm relic} - (v/\sigma)_{\rm nonrelic}$")
            ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
            ax.set_ylabel(r"Compactness")
            ax.set_title("2D median difference in rotation support")
            save_fig(fig, figdir / "delta_v_over_sigma_relic_minus_nonrelic_2d.png")
    else:
        print("2D comparison table is empty; sample too sparse for the chosen binning.")

    # 7) Separate panels for central/satellite and ex-situ trends
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        (axes[0], "is_central", "Central", {True: "C0", False: "C3"}),
        (axes[1], "is_relic", "Relic", {True: "C1", False: "0.5"}),
        (axes[2], "relic_flag", "Ex-situ fraction", None),
    ]
    for ax, flagcol, title, colors in panels:
        if flagcol == "relic_flag":
            sc = scatter_with_colour(ax, compact_m, vos_m, exsitu_m, cmap="viridis", s=18, alpha=0.85)
            if sc is not None:
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label(r"$f_{\rm ex-situ}$")
            ax.set_title("Rotation support coloured by ex-situ fraction")
        else:
            for state, col in colors.items():
                sel = compare[flagcol].to_numpy() == state
                ax.scatter(compact_m[sel], vos_m[sel], s=16, alpha=0.75, color=col, label=f"{title.lower()}={state}")
            ax.legend(fontsize=8)
            ax.set_title(f"{title} split")
        ax.axvline(COMPACTNESS_CUT, ls="--", lw=1.2, color="black")
        ax.axhline(1.0, ls="--", lw=1.0, color="black")
        ax.set_xlabel("Compactness")
        ax.set_ylabel(r"$v/\sigma$")
        ax.grid(True)
    save_fig(fig, figdir / "v_over_sigma_central_relic_exsitu_panels.png")

    # 8) Distribution summary for the core comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    sel_relic = compare["is_relic"].to_numpy()
    sel_compact = compare["compactness"].to_numpy() >= COMPACTNESS_CUT
    groups = {
        "relic compact": sel_relic & sel_compact,
        "relic diffuse": sel_relic & (~sel_compact),
        "non-relic compact": (~sel_relic) & sel_compact,
        "non-relic diffuse": (~sel_relic) & (~sel_compact),
    }
    for lab, sel in groups.items():
        vals = compare.loc[sel, "v_over_sigma"].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 2:
            ax.hist(vals, bins=20, density=True, histtype="step", lw=1.8, label=f"{lab} (n={len(vals)})")
    ax.set_xlabel(r"$v/\sigma$")
    ax.set_ylabel("Probability density")
    ax.legend(fontsize=8)
    ax.grid(True)
    save_fig(fig, figdir / "v_over_sigma_group_distributions.png")

    # -------------------------- BIN SUMMARY ---------------------------
    # Mass-binned summary, split by relic / non-relic and central / satellite
    rows = []
    mass_bins = np.arange(
        math.floor(np.nanmin(logM_m) / MASS_BIN_WIDTH) * MASS_BIN_WIDTH,
        math.ceil(np.nanmax(logM_m) / MASS_BIN_WIDTH) * MASS_BIN_WIDTH + 1e-9,
        MASS_BIN_WIDTH,
    )
    if mass_bins.size >= 2:
        for i in range(len(mass_bins) - 1):
            sel_mass = (logM_m >= mass_bins[i]) & (logM_m < mass_bins[i + 1])
            if np.sum(sel_mass) < MIN_PER_BIN:
                continue
            row = {
                "mass_lo": mass_bins[i],
                "mass_hi": mass_bins[i + 1],
                "mass_center": 0.5 * (mass_bins[i] + mass_bins[i + 1]),
                "n_total": int(np.sum(sel_mass)),
                "n_relic": int(np.sum(sel_mass & sel_relic)),
                "n_nonrelic": int(np.sum(sel_mass & (~sel_relic))),
                "n_central": int(np.sum(sel_mass & cen_m)),
                "n_satellite": int(np.sum(sel_mass & (~cen_m))),
            }
            for name, arr in {
                "v_over_sigma": vos_m,
                "lambda_R_proxy": lambda_proxy_m,
                "kappa_corot": kappa_m,
                "exsitu_frac": exsitu_m,
                "compactness": compact_m,
            }.items():
                vals = arr[sel_mass]
                vals = vals[np.isfinite(vals)]
                if len(vals) >= MIN_PER_BIN:
                    row[f"{name}_median"] = float(np.nanmedian(vals))
                    row[f"{name}_p16"] = float(np.nanpercentile(vals, 16))
                    row[f"{name}_p84"] = float(np.nanpercentile(vals, 84))
            # split by relic / non-relic for v/sigma
            for grp_name, grp_sel in {"relic": sel_mass & sel_relic, "nonrelic": sel_mass & (~sel_relic)}.items():
                vals = vos_m[grp_sel]
                vals = vals[np.isfinite(vals)]
                row[f"{grp_name}_v_over_sigma_median"] = float(np.nanmedian(vals)) if len(vals) >= MIN_PER_BIN else np.nan
                row[f"{grp_name}_v_over_sigma_p16"] = float(np.nanpercentile(vals, 16)) if len(vals) >= MIN_PER_BIN else np.nan
                row[f"{grp_name}_v_over_sigma_p84"] = float(np.nanpercentile(vals, 84)) if len(vals) >= MIN_PER_BIN else np.nan
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_csv = Path(OUTDIR) / "kinematics_summary_by_mass_bin.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)

    if not summary_df.empty and {"relic_v_over_sigma_median", "nonrelic_v_over_sigma_median"}.issubset(summary_df.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            summary_df["mass_center"],
            summary_df["relic_v_over_sigma_median"],
            yerr=[summary_df["relic_v_over_sigma_median"] - summary_df["relic_v_over_sigma_p16"],
                  summary_df["relic_v_over_sigma_p84"] - summary_df["relic_v_over_sigma_median"]],
            fmt="o-", capsize=3, lw=1.5, label="relics",
        )
        ax.errorbar(
            summary_df["mass_center"],
            summary_df["nonrelic_v_over_sigma_median"],
            yerr=[summary_df["nonrelic_v_over_sigma_median"] - summary_df["nonrelic_v_over_sigma_p16"],
                  summary_df["nonrelic_v_over_sigma_p84"] - summary_df["nonrelic_v_over_sigma_median"]],
            fmt="s--", capsize=3, lw=1.5, label="non-relics",
        )
        ax.set_xlabel(r"$\log(M_\star/M_\odot)$")
        ax.set_ylabel(r"Median $v/\sigma$")
        ax.grid(True)
        ax.legend(fontsize=8)
        save_fig(fig, figdir / "mass_binned_relic_vs_nonrelic_v_over_sigma.png")

    # --------------------------- PRINT ANSWER -------------------------
    # These lines are the simple, interpretable summary you can quote in your notes.
    def summarize_mask(mask: np.ndarray, label: str):
        vals = matched.loc[mask, "v_over_sigma"].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return f"{label}: no data"
        return (
            f"{label}: n={len(vals)}, median(v/sigma)={np.nanmedian(vals):.3f}, "
            f"median(lambda_proxy)={np.nanmedian(matched.loc[mask, 'lambda_R_proxy']):.3f}, "
            f"median(kappa_corot)={np.nanmedian(matched.loc[mask, 'kappa_corot']):.3f}"
        )

    print("\nCore descriptive summary")
    print("------------------------")
    print(summarize_mask(matched["is_relic"].to_numpy() & (matched["compactness"].to_numpy() >= COMPACTNESS_CUT), "relic + compact"))
    print(summarize_mask(matched["is_relic"].to_numpy() & (matched["compactness"].to_numpy() < COMPACTNESS_CUT), "relic + diffuse"))
    print(summarize_mask((~matched["is_relic"].to_numpy()) & (matched["compactness"].to_numpy() >= COMPACTNESS_CUT), "non-relic + compact"))
    print(summarize_mask((~matched["is_relic"].to_numpy()) & (matched["compactness"].to_numpy() < COMPACTNESS_CUT), "non-relic + diffuse"))

    # central/satellite split among relics
    relic_mask = matched["is_relic"].to_numpy()
    print("\nCentral / satellite among relics")
    print("--------------------------------")
    print(summarize_mask(relic_mask & matched["is_central"].to_numpy(), "relic centrals"))
    print(summarize_mask(relic_mask & (~matched["is_central"].to_numpy()), "relic satellites"))

    # ex-situ split (median only)
    ex = matched["exsitu_frac"].to_numpy()
    finite_ex = finite(ex)
    if np.any(finite_ex):
        q1, q2 = np.nanpercentile(ex[finite_ex], [33, 67])
        print("\nEx-situ split (terciles)")
        print("------------------------")
        print(summarize_mask(finite_ex & (ex <= q1), f"low ex-situ <= {q1:.2f}"))
        print(summarize_mask(finite_ex & (ex > q1) & (ex <= q2), f"mid ex-situ {q1:.2f}-{q2:.2f}"))
        print(summarize_mask(finite_ex & (ex > q2), f"high ex-situ > {q2:.2f}"))

    print("\nDone. See:")
    print(" -", matched_csv)
    print(" -", summary_csv)
    print(" -", figdir)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
