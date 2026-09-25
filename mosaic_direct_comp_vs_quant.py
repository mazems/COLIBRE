#!/usr/bin/env python3
"""
mosaic_direct_comp_vs_quant.py

Create a 3x2 mass-size mosaic for:
  a) luminosity-weighted stellar age
  b) stellar metallicity [Z/H]
  c) [Mg/Fe]
  d) log10(sSFR / yr^-1)
  e) log10(sigma / km s^-1)
  f) ex-situ stellar mass fraction

Modes
-----
layout:
    Fast layout preview. Does not import `common`, read SOAP/HDF5 data,
    or run LOESS.

full:
    Loads all data, computes all six LOESS grids, writes a cache, and
    makes the final mosaic.

cached:
    Reads the previously generated cache and redraws the mosaic in seconds.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, BSpline
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree as KDTree
from typing import Any, Dict, List, Optional, Sequence


# =============================================================================
# USER SETTINGS
# =============================================================================

MODEL_NAME = "L0200N3008/THERMAL_AGN/"
MODEL_DIR = Path("/mnt/su3-pro/colibre") / MODEL_NAME
SNAP_FILE = "0127"
ZTARGET = 0.0

SIGMA_FILE = (
    Path("/mnt/su3-pro/colibre")
    / MODEL_NAME
    / "SOAP-HBT/extra/halo_properties_0127.hdf5"
)

SIGMA_DATASET = (
    "/ExclusiveSphere/HalfMassRadiusStars/"
    "StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"
)

EXSITU_DIR = Path("/mnt/su3ctm/kproctor/ForMax")
EXSITU_GLOB = "*exsitu*summary*.hdf5"

OUTDIR = Path("plots")

CACHE_FILE = OUTDIR / "compactness_quantity_cache.npz"

LAYOUT_OUTPUT = OUTDIR / "compactness_quantity_layout"
FINAL_OUTPUT = OUTDIR / "compactness_quantity_mosaic"

MASS_LIMIT = 1e9

ZSUN = 0.0134
LOG_MGFE_SUN = 0.10
SSFR_FLOOR = -12.0

###############################################################################
# Figure layout
###############################################################################

FIGSIZE = (13.5, 15)

LEFT = 0.08
RIGHT = 0.97
BOTTOM = 0.06
TOP = 0.985

HSPACE = 0.08
WSPACE = 0.08

SHOW_PANEL_LEGENDS = True

SHOW_XLABEL_ONLY_BOTTOM = True
SHOW_YLABEL_ONLY_LEFT = True

###############################################################################
# Compactness axis
###############################################################################

X_LIMITS = (8.8, 11.2)

###############################################################################
# Individual panel settings
###############################################################################

QUANTITY_KEYS = [
    "age",
    "metallicity",
    "mgfe",
    "ssfr",
    "sigma",
    "exsitu",
]

PANEL_LABELS = [
    "a)",
    "b)",
    "c)",
    "d)",
    "e)",
    "f)",
]

YLABELS = [
    r"$\text{Age} [\mathrm{Gyr}]$",
    r"$[\mathrm{Z}/\mathrm{H}]$",
    r"$[\mathrm{Mg}/\mathrm{Fe}]$",
    r"$\log_{10}(\mathrm{sSFR}\,/\,\mathrm{yr}^{-1})$",
    r"$\log_{10}(\sigma\,/\,\mathrm{km}\,\mathrm{s}^{-1})$",
    r"$f_{\mathrm{ex-situ}}$",
]

###############################################################################
# Spline configuration
###############################################################################

TARGET_PER_BIN = {
    "age":1000,
    "metallicity":1000,
    "mgfe":1000,
    "ssfr":1000,
    "sigma":1000,
    "exsitu":1000,
}

SPLINE_S = {
    "age":3.5,
    "metallicity":0.005,
    "mgfe":0.0007,
    "ssfr":0.10,
    "sigma":0.015,
    "exsitu":0.02,
}

SEARCH_LO = {
    "age":None,
    "metallicity":9.5,
    "mgfe":None,
    "ssfr":None,
    "sigma":None,
    "exsitu":None,
}

SEARCH_HI = {
    "age":None,
    "metallicity":10.2,
    "mgfe":None,
    "ssfr":None,
    "sigma":None,
    "exsitu":None,
}

SPLINE_K = 3
MIN_BIN_COUNT = 5

###############################################################################
# Dummy layout ranges
###############################################################################

LAYOUT_YRANGES = {
    "age":(0,12),
    "metallicity":(-0.5,0.2),
    "mgfe":(0.05,0.22),
    "ssfr":(-12,-9),
    "sigma":(1.3,2.0),
    "exsitu":(0,0.5),
}

# ------------------------------------------------------------------
# Weighted spline fit
# ------------------------------------------------------------------
def _fit_weighted_bspline(
    x: Sequence[float],
    y: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    spline_k: int = 3,
    spline_s_factor: float = 0.05,
) -> BSpline:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if weights is not None:
        weights = np.asarray(weights, dtype=float).ravel()

    finite = np.isfinite(x) & np.isfinite(y)
    if weights is not None:
        finite &= np.isfinite(weights)

    x = x[finite]
    y = y[finite]
    w = None if weights is None else weights[finite]

    if x.size < 4:
        raise ValueError("Need at least 4 finite points for spline fitting.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if w is not None:
        w = w[order]

    xu, inv = np.unique(x, return_inverse=True)
    if xu.size != x.size:
        y_num = np.zeros_like(xu, dtype=float)
        w_num = np.zeros_like(xu, dtype=float)

        if w is None:
            for i, j in enumerate(inv):
                y_num[j] += y[i]
                w_num[j] += 1.0
        else:
            for i, j in enumerate(inv):
                y_num[j] += y[i] * w[i]
                w_num[j] += w[i]

        x = xu
        y = y_num / np.maximum(w_num, 1e-12)
        w = w_num

    k = int(min(max(1, spline_k), x.size - 1))
    s = float(spline_s_factor) * x.size

    tck = splrep(x, y, w=w, k=k, s=s)
    return BSpline(*tck)

def _bootstrap_bin_medians(
    compactness_all: np.ndarray,
    quantity_all: np.ndarray,
    cbins_edges: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    compactness_all = np.asarray(compactness_all, dtype=float).ravel()
    quantity_all = np.asarray(quantity_all, dtype=float).ravel()
    cbins_edges = np.asarray(cbins_edges, dtype=float).ravel()
    n = cbins_edges.size - 1

    idxs = np.searchsorted(cbins_edges, compactness_all, side="right") - 1
    valid_mask = (idxs >= 0) & (idxs < n) & np.isfinite(quantity_all) & np.isfinite(compactness_all)
    if valid_mask.sum() == 0:
        return np.full(n, np.nan, dtype=float)

    bins_values: List[List[float]] = [[] for _ in range(n)]
    for ind, q in zip(idxs[valid_mask], quantity_all[valid_mask]):
        bins_values[int(ind)].append(float(q))

    med_bs = np.full(n, np.nan, dtype=float)
    for i in range(n):
        vals = bins_values[i]
        if len(vals) == 0:
            continue
        sample = rng.choice(vals, size=len(vals), replace=True)
        med_bs[i] = np.nanmedian(sample)

    return med_bs

def find_compactness_threshold_spline(
    cbin_centers: Sequence[float],
    cmed: Sequence[float],
    compactness_all: Sequence[float],
    quantity_all: Sequence[float],
    counts_per_bin: Optional[Sequence[int]] = None,
    edge_frac: float = 0.05,
    deriv_thresh_factor: float = 1.0, #default 3.0
    min_bin_count: int = 5,
    bootstrap_n: int = 0,
    random_seed: int = 12345,
    cbins_edges: Optional[Sequence[float]] = None,
    search_lo: Optional[float] = 9.5,
    search_hi: Optional[float] = 10.5,
    spline_s_factor: float = 0.05,
    spline_k: int = 3,
) -> Dict[str, Any]:
    cbin_centers = np.asarray(cbin_centers, dtype=float).ravel()
    cmed = np.asarray(cmed, dtype=float).ravel()
    n = cbin_centers.size

    if n < 4:
        raise ValueError("Need at least 4 compactness bins.")

    finite_mask = np.isfinite(cbin_centers) & np.isfinite(cmed)
    if counts_per_bin is not None:
        counts_per_bin = np.asarray(counts_per_bin, dtype=int).ravel()
        if counts_per_bin.size != n:
            raise ValueError("counts_per_bin must have same length as cbin_centers.")
        finite_mask &= np.isfinite(counts_per_bin)

    if finite_mask.sum() < 4:
        raise ValueError("Too few finite values for spline fitting.")

    x = cbin_centers[finite_mask]
    y = cmed[finite_mask]
    if counts_per_bin is not None:
        w = np.sqrt(np.maximum(counts_per_bin[finite_mask], 1))
    else:
        w = np.ones_like(x, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]

    xu, inv = np.unique(x, return_inverse=True)
    if xu.size != x.size:
        y_num = np.zeros_like(xu, dtype=float)
        w_num = np.zeros_like(xu, dtype=float)
        for i, j in enumerate(inv):
            y_num[j] += y[i] * w[i]
            w_num[j] += w[i]
        x = xu
        y = y_num / np.maximum(w_num, 1e-12)
        w = w_num

    if x.size < 4:
        raise ValueError("Not enough unique x values for spline fitting.")

    k = int(min(max(1, spline_k), x.size - 1))
    s = float(spline_s_factor) * x.size

    spl = _fit_weighted_bspline(x, y, weights=w, spline_k=spline_k, spline_s_factor=spline_s_factor)
    cmed_s = np.asarray(spl(cbin_centers), dtype=float)
    deriv = np.asarray(spl.derivative()(cbin_centers), dtype=float)
    abs_deriv = np.abs(deriv)

    search_mask = np.isfinite(cbin_centers)
    if search_lo is not None:
        search_mask &= cbin_centers >= search_lo
    if search_hi is not None:
        search_mask &= cbin_centers <= search_hi

    if np.count_nonzero(search_mask) >= 4:
        abs_stat = abs_deriv[search_mask]
        global_indices = np.flatnonzero(search_mask)
    else:
        abs_stat = abs_deriv
        global_indices = np.arange(n)

    baseline = float(np.nanmedian(abs_stat))
    mad = float(np.nanmedian(np.abs(abs_stat - baseline)))
    deriv_threshold = baseline + deriv_thresh_factor * (1.4826 * mad)

    idx_local_max = int(np.nanargmax(abs_stat))
    idx_max = int(global_indices[idx_local_max])

    left_edge_idx = int(np.floor(edge_frac * n))
    right_edge_idx = int(np.ceil((1.0 - edge_frac) * n)) - 1

    method: Optional[str] = None
    threshold_value: Optional[float] = None

    if (idx_max > left_edge_idx) and (idx_max < right_edge_idx) and (abs_deriv[idx_max] >= deriv_threshold):
        method = "turning_point"
        threshold_value = float(cbin_centers[idx_max])
    else:
        found = False
        for i in global_indices[::-1]:
            if abs_deriv[i] >= deriv_threshold:
                if (counts_per_bin is None) or (counts_per_bin[i] >= min_bin_count):
                    threshold_value = float(cbin_centers[i])
                    method = "start_exceed"
                    found = True
                    break
        if not found:
            method = "fallback_percentile"
            threshold_value = float(np.nanpercentile(np.asarray(compactness_all, dtype=float), 90))

    bootstrap_stats = None
    if bootstrap_n and bootstrap_n > 0:
        if cbins_edges is None:
            if n >= 2:
                edges = np.empty(n + 1, dtype=float)
                edges[1:-1] = 0.5 * (cbin_centers[:-1] + cbin_centers[1:])
                first_half = edges[1] - cbin_centers[0]
                last_half = cbin_centers[-1] - edges[-2]
                edges[0] = cbin_centers[0] - first_half
                edges[-1] = cbin_centers[-1] + last_half
                cbins_edges = edges
            else:
                cbins_edges = np.array([cbin_centers[0] - 0.5, cbin_centers[0] + 0.5], dtype=float)
        else:
            cbins_edges = np.asarray(cbins_edges, dtype=float).ravel()

        rng = np.random.default_rng(random_seed)
        thr_boot: List[float] = []
        thr_boot_method: List[str] = []

        for _ in range(int(bootstrap_n)):
            med_bs = _bootstrap_bin_medians(
                compactness_all=compactness_all,
                quantity_all=quantity_all,
                cbins_edges=cbins_edges,
                rng=rng,
            )

            finite_bs = np.isfinite(med_bs)
            if finite_bs.sum() < 4:
                thr_boot.append(float(np.nanpercentile(np.asarray(compactness_all, dtype=float), 90)))
                thr_boot_method.append("fallback_percentile")
                continue

            x_b = cbin_centers[finite_bs]
            y_b = med_bs[finite_bs]
            if counts_per_bin is not None:
                w_b = np.sqrt(np.maximum(counts_per_bin[finite_bs], 1))
            else:
                w_b = np.ones_like(x_b, dtype=float)

            order_b = np.argsort(x_b)
            x_b = x_b[order_b]
            y_b = y_b[order_b]
            w_b = w_b[order_b]

            xu_b, inv_b = np.unique(x_b, return_inverse=True)
            if xu_b.size != x_b.size:
                y_num_b = np.zeros_like(xu_b, dtype=float)
                w_num_b = np.zeros_like(xu_b, dtype=float)
                for i, j in enumerate(inv_b):
                    y_num_b[j] += y_b[i] * w_b[i]
                    w_num_b[j] += w_b[i]
                x_b = xu_b
                y_b = y_num_b / np.maximum(w_num_b, 1e-12)
                w_b = w_num_b

            if x_b.size < 4:
                thr_boot.append(float(np.nanpercentile(np.asarray(compactness_all, dtype=float), 90)))
                thr_boot_method.append("fallback_percentile")
                continue

            k_b = int(min(max(1, spline_k), x_b.size - 1))
            s_b = float(spline_s_factor) * x_b.size
            spl_b = _fit_weighted_bspline(x_b, y_b, weights=w_b, spline_k=spline_k, spline_s_factor=spline_s_factor)
            deriv_b = np.asarray(spl_b.derivative()(cbin_centers), dtype=float)
            abs_bs = np.abs(deriv_b)

            if np.count_nonzero(search_mask) >= 4:
                abs_stat_b = abs_bs[search_mask]
                global_indices_b = np.flatnonzero(search_mask)
            else:
                abs_stat_b = abs_bs
                global_indices_b = np.arange(n)

            baseline_b = float(np.nanmedian(abs_stat_b))
            mad_b = float(np.nanmedian(np.abs(abs_stat_b - baseline_b)))
            thr_b = baseline_b + deriv_thresh_factor * (1.4826 * mad_b)
            idx_local_max_b = int(np.nanargmax(abs_stat_b))
            idx_max_b = int(global_indices_b[idx_local_max_b])

            if (idx_max_b > left_edge_idx) and (idx_max_b < right_edge_idx) and (abs_bs[idx_max_b] >= thr_b):
                thr_boot.append(float(cbin_centers[idx_max_b]))
                thr_boot_method.append("turning_point")
            else:
                found_b = False
                for j in global_indices_b[::-1]:
                    if abs_bs[j] >= thr_b and (counts_per_bin is None or counts_per_bin[j] >= min_bin_count):
                        thr_boot.append(float(cbin_centers[j]))
                        thr_boot_method.append("start_exceed")
                        found_b = True
                        break
                if not found_b:
                    thr_boot.append(float(np.nanpercentile(np.asarray(compactness_all, dtype=float), 90)))
                    thr_boot_method.append("fallback_percentile")

        thr_arr = np.asarray(thr_boot, dtype=float)
        bootstrap_stats = {
            "median": float(np.nanmedian(thr_arr)),
            "p16": float(np.nanpercentile(thr_arr, 16)),
            "p84": float(np.nanpercentile(thr_arr, 84)),
            "raw": thr_arr,
            "methods": thr_boot_method,
        }

    return {
        "threshold": threshold_value,
        "method": method,
        "cmed_smooth": cmed_s,
        "derivative": deriv,
        "deriv_threshold": deriv_threshold,
        "bootstrap": bootstrap_stats,
    }

# ------------------------------------------------------------------
# KDE background
# ------------------------------------------------------------------
def _plot_density_contours(ax_main, x, y):
    fin = np.isfinite(x) & np.isfinite(y)
    if np.sum(fin) < 10:
        return None

    xs = x[fin]
    ys = y[fin]
    pts = np.vstack([xs, ys])
    kde = gaussian_kde(pts, bw_method="scott")

    nx_grid = 200
    ny_grid = 200
    x_min, x_max = np.nanpercentile(x, [1, 99])
    y_min, y_max = np.nanpercentile(y, [1, 99])
    xpad = 0.05 * (x_max - x_min + 1e-9)
    ypad = 0.05 * (y_max - y_min + 1e-9)
    xg = np.linspace(x_min - xpad, x_max + xpad, nx_grid)
    yg = np.linspace(y_min - ypad, y_max + ypad, ny_grid)
    Xgrid, Ygrid = np.meshgrid(xg, yg)
    grid_pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T

    tree = KDTree(np.column_stack((xs, ys)))
    d_grid, _ = tree.query(grid_pts, k=1)
    d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
    typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
    cut = max(typical_spacing * 1.3, 1e-6)
    mask_far = d_grid > cut

    Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
    Z_flat = Z.ravel()
    Z_flat[mask_far] = np.nan
    Z = Z_flat.reshape(Xgrid.shape)

    finite_vals = Z[np.isfinite(Z)]
    if finite_vals.size == 0:
        return None

    levs = np.percentile(finite_vals, [50, 75, 90, 97])
    cf = ax_main.contourf(Xgrid, Ygrid, Z, levels=50, cmap="viridis", antialiased=True)
    ax_main.contour(Xgrid, Ygrid, Z, levels=levs, colors="k", linewidths=0.6, alpha=0.5)
    return cf


def robust_limits(z, key):
    finite = np.asarray(z, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        idx = QUANTITY_KEYS.index(key)
        return float(LAYOUT_VMIN[idx]), float(LAYOUT_VMAX[idx])

    vmin = float(np.nanpercentile(finite, 5))
    vmax = float(np.nanpercentile(finite, 95))

    if key == "exsitu":
        vmin = max(0.0, vmin)
        vmax = min(1.0, vmax)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        med = float(np.nanmedian(finite))
        span = max(0.2, 0.5 * max(abs(med), 1e-6))
        vmin, vmax = med - span, med + span

    return vmin, vmax


# =============================================================================
# DATA LOADING
# =============================================================================

def find_exsitu_file():
    matches = sorted(EXSITU_DIR.glob(EXSITU_GLOB))
    if not matches:
        raise FileNotFoundError(f"No ex-situ HDF5 matching {EXSITU_GLOB!r} in {EXSITU_DIR}")
    return max(matches, key=lambda p: p.stat().st_mtime)

def load_full_data():
    try:
        import common
    except ImportError as exc:
        raise ImportError(
            "Could not import `common`. Run this script from the COLIBRE-analysis directory "
            "or add that directory to PYTHONPATH."
        ) from exc

    Lu = 1.0
    Mu = 1.988e43 / 1.989e33
    tu = 3.086e19 / 3.154e7
    comov_to_physical_length = 1.0 / (1.0 + ZTARGET)

    fields = {
        "ExclusiveSphere/50kpc": (
            "StellarMass",
            "StarFormationRate",
            "HalfMassRadiusStars",
            "CentreOfMass",
            "MassWeightedMeanStellarAge",
            "LuminosityWeightedMeanStellarAge",
            "LinearMassWeightedIronOverHydrogenOfStars",
            "LinearMassWeightedMagnesiumOverHydrogenOfStars",
            "StellarMassFractionInMetals",
        )
    }

    fields_ids = {
        "InputHalos": (
            "HaloCatalogueIndex",
            "IsCentral",
            "HBTplus/DescendantTrackId",
            "HBTplus/TrackId",
        )
    }

    print("Reading SOAP quantities...")
    (
        m30,
        sfr30,
        r50,
        cp,
        stellarage,
        stellarage_lum,
        fe_over_h,
        mg_over_h,
        zstar_raw,
    ) = common.read_group_data_colibre(
        str(MODEL_DIR) + os.sep,
        SNAP_FILE,
        fields,
    )

    (
        sgn,
        is_central,
        desc_id,
        track_id,
    ) = common.read_group_data_colibre(
        str(MODEL_DIR) + os.sep,
        SNAP_FILE,
        fields_ids,
    )

    m30 = np.asarray(m30, dtype=float) * Mu
    sfr30 = np.asarray(sfr30, dtype=float) * Mu / tu
    r50 = np.asarray(r50, dtype=float) * Lu * comov_to_physical_length * 1e3
    stellarage_lum = np.asarray(stellarage_lum, dtype=float) * tu / 1e9
    fe_over_h = np.asarray(fe_over_h, dtype=float)
    mg_over_h = np.asarray(mg_over_h, dtype=float)
    zstar = np.asarray(zstar_raw, dtype=float)
    sgn = np.asarray(sgn)

    base_mask = (
        (m30 >= MASS_LIMIT)
        & np.isfinite(m30)
        & np.isfinite(r50)
        & (m30 > 0)
        & (r50 > 0)
    )
    if not np.any(base_mask):
        raise RuntimeError("No galaxies survive the mass/radius selection.")

    log_m = np.log10(m30[base_mask])
    log_r = np.log10(r50[base_mask])

    compactness = log_m - 1.5 * log_r

    age = stellarage_lum[base_mask]

    with np.errstate(divide="ignore", invalid="ignore"):
        metallicity = np.where(
            (zstar[base_mask] > 0) & np.isfinite(zstar[base_mask]),
            np.log10(zstar[base_mask] / ZSUN),
            np.nan,
        )
        mgfe_number = mg_over_h[base_mask] / fe_over_h[base_mask]
        mgfe = np.where(
            mgfe_number > 0,
            np.log10(mgfe_number) - LOG_MGFE_SUN,
            np.nan,
        )
        ssfr = sfr30[base_mask] / m30[base_mask]
        log_ssfr = np.where(ssfr > 0, np.log10(ssfr), np.nan)

    log_ssfr[~np.isfinite(log_ssfr)] = SSFR_FLOOR

    print("Reading velocity dispersion...")
    row_idx = np.flatnonzero(base_mask)
    with h5py.File(SIGMA_FILE, "r") as handle:
        rows = np.asarray(handle[SIGMA_DATASET][row_idx, :], dtype=np.float32)

    if rows.ndim != 2 or rows.shape[1] < 9:
        raise RuntimeError(
            f"Unexpected sigma dataset shape {rows.shape}; expected at least 9 columns."
        )

    sigma_sel = np.sqrt((rows[:, 0] ** 2 + rows[:, 4] ** 2 + rows[:, 8] ** 2) / 3.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sigma = np.where(sigma_sel > 0, np.log10(sigma_sel), np.nan)

    exsitu_file = find_exsitu_file()
    print("Reading ex-situ fractions from:", exsitu_file)
    with h5py.File(exsitu_file, "r") as handle:
        if "stars" not in handle:
            raise KeyError(f"{exsitu_file} has no 'stars' dataset.")
        stars = np.asarray(handle["stars"])

    if stars.ndim != 2 or stars.shape[1] < 4:
        raise RuntimeError(
            f"Unexpected ex-situ 'stars' shape {stars.shape}; expected (N, >=4)."
        )

    exsitu_lookup = pd.Series(
        dict(zip(stars[:, 0].astype(np.int64), stars[:, 3].astype(float))),
        dtype=float,
    )
    halo_selected = sgn[base_mask].astype(np.int64)
    exsitu = exsitu_lookup.reindex(halo_selected).to_numpy(dtype=float)

    quantities = np.vstack([age, metallicity, mgfe, log_ssfr, log_sigma, exsitu])

    print(f"Selected galaxies: {log_m.size}")
    for key, values in zip(QUANTITY_KEYS, quantities):
        print(f"  finite {key:12s}: {np.isfinite(values).sum():7d}")

    return {
        "compactness": compactness,
        "age": stellarage_lum,
        "metallicity": metallicity,
        "mgfe": mgfe,
        "ssfr": log_ssfr,
        "sigma": log_sigma,
        "exsitu": exsitu,
    }

# =============================================================================
# CACHE
# =============================================================================

def build_full_cache():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    compactness, quantities = load_full_data()

    result = {
        "compactness": compactness,
        "quantities": quantities,
    }

    np.savez_compressed(
        CACHE_FILE,
        compactness=result["compactness"],
        quantities=result["quantities"],
    )

    print("Saved cache:", CACHE_FILE)

    return result


def load_cache():
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_FILE}\n"
            "Run once with --mode full before --mode cached."
        )

    with np.load(CACHE_FILE) as data:
        return {
            "compactness": data["compactness"],
            "quantities": data["quantities"],
        }


def make_layout_preview_data():
    xgrid = np.linspace(X_LIMITS[0], X_LIMITS[1], 180)
    ygrid = np.linspace(Y_LIMITS[0], Y_LIMITS[1], 130)
    Xg, Yg = np.meshgrid(xgrid, ygrid)

    base = 0.55 + 0.25 * np.sin(1.8 * (Xg - 9.0)) - 0.18 * np.cos(2.4 * (Yg + 0.5))
    envelope = (
        (Yg > -0.45 + 0.16 * (Xg - 9.0) ** 1.7)
        & (Yg < 1.48 - 0.03 * (Xg - 10.3) ** 2)
    )

    grids = []
    for idx in range(6):
        phase = base + 0.12 * idx * np.sin(Yg * (idx + 1))
        scaled = LAYOUT_VMIN[idx] + (
            (phase - np.nanmin(phase)) / (np.nanmax(phase) - np.nanmin(phase))
        ) * (LAYOUT_VMAX[idx] - LAYOUT_VMIN[idx])
        grids.append(np.where(envelope, scaled, np.nan))

    return {
        "Xg": Xg,
        "Yg": Yg,
        "grids": np.stack(grids),
        "vmins": LAYOUT_VMIN.copy(),
        "vmaxs": LAYOUT_VMAX.copy(),
        "log_m": np.array([]),
        "log_r": np.array([]),
        "missing_masks": np.zeros((6, 0), dtype=bool),
    }


# # =============================================================================
# # MOSAIC DRAWING
# # =============================================================================

# def style_axis(ax, index):
#     row, col = divmod(index, 2)
#     ax.set_xlim(*X_LIMITS)
#     ax.set_ylim(*Y_LIMITS)
#     ax.grid(True, alpha=0.55)
#     ax.tick_params(
#         axis="both",
#         labelsize=12,
#         direction="in",
#         length=5,
#         width=0.9,
#         top=True,
#         right=True,
#     )

#     if SHOW_XLABEL_ONLY_BOTTOM and row < 2:
#         ax.tick_params(labelbottom=False)
#     else:
#         ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$", fontsize=18)

#     if SHOW_YLABEL_ONLY_LEFT and col == 1:
#         ax.tick_params(labelleft=False)
#     else:
#         ax.set_ylabel(r"$\log_{10}(R_{1/2,\star} / \mathrm{kpc})$", fontsize=18)

#     ax.text(
#         0.025,
#         0.975,
#         PANEL_LABELS[index],
#         transform=ax.transAxes,
#         ha="left",
#         va="top",
#         fontsize=16,
#     )


# =============================================================================
# SPLINE COMPACTNESS MOSAIC DRAWING
# =============================================================================

def draw_spline_mosaic(data, output_stem, preview=False):

    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.size": 16,
        }
    )

    compactness = data["compactness"]
    quantities = data["quantities"]
    labels = data["quantity_labels"]
    spline_results = data["spline_results"]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(14, 16),
        sharex=True,
        constrained_layout=True,
    )

    axes = axes.flatten()


    for i, ax in enumerate(axes):

        x = np.asarray(compactness, dtype=float)
        y = np.asarray(quantities[i], dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)

        if np.sum(valid) == 0:
            continue


        xv = x[valid]
        yv = y[valid]


        # --------------------------------------------------------------
        # Scatter
        # --------------------------------------------------------------

        ax.scatter(
            xv,
            yv,
            s=5,
            color="lightgrey",
            alpha=0.5,
            rasterized=True,
            label="galaxies",
        )


        # --------------------------------------------------------------
        # KDE density
        # --------------------------------------------------------------

        cf = _plot_density_contours(
            ax,
            xv,
            yv,
        )


        # --------------------------------------------------------------
        # Reconstruct binned median curve
        # --------------------------------------------------------------

        nbins = max(
            3,
            int(len(xv) / 1000)
        )

        bins = np.nanpercentile(
            xv,
            np.linspace(0, 100, nbins + 1)
        )


        bins = np.unique(bins)

        if len(bins) > 3:

            centers = 0.5 * (
                bins[:-1]
                +
                bins[1:]
            )

            idx = np.digitize(
                xv,
                bins
            ) - 1


            med = np.full(
                len(centers),
                np.nan
            )

            for b in range(len(centers)):
                vals = yv[idx == b]

                if len(vals) > 5:
                    med[b] = np.nanmedian(vals)


            ok = np.isfinite(med)

            if np.any(ok):

                ax.plot(
                    centers[ok],
                    med[ok],
                    color="C0",
                    lw=2.5,
                    label="binned median",
                )


        # --------------------------------------------------------------
        # Spline result
        # --------------------------------------------------------------

        res = spline_results[i]


        if res is not None:

            spline_y = res["cmed_smooth"]

            deriv = res["derivative"]


            # x-grid used by spline
            spline_x = np.linspace(
                np.nanmin(x),
                np.nanmax(x),
                len(spline_y),
            )


            ax.plot(
                spline_x,
                spline_y,
                color="C2",
                linestyle="--",
                lw=2.5,
                label="B-spline",
            )


            ax.axvline(
                res["threshold"],
                color="C2",
                linestyle=":",
                lw=2,
                label=(
                    rf"$\log_{{10}}\Sigma_{{1.5}}="
                    f"{res['threshold']:.2f}$"
                ),
            )


            # ----------------------------------------------------------
            # derivative axis
            # ----------------------------------------------------------

            ax2 = ax.twinx()

            ax2.plot(
                spline_x,
                deriv,
                color="C3",
                lw=1.5,
                alpha=0.8,
                label="derivative",
            )


            ax2.axhline(
                0,
                color="C3",
                linestyle=":",
                alpha=0.5,
            )


            ax2.set_ylabel(
                "d(quantity)/d(log Σ1.5)",
                color="C3",
                fontsize=12,
            )


            ax2.tick_params(
                axis="y",
                colors="C3",
                labelsize=10,
            )


        # --------------------------------------------------------------
        # Formatting
        # --------------------------------------------------------------

        row, col = divmod(i, 2)

        ax.text(
            0.03,
            0.95,
            chr(65+i),
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
        )


        ax.set_ylabel(
            labels[i],
            fontsize=16,
        )


        ax.grid(
            True,
            alpha=0.35,
        )


        ax.legend(
            fontsize=9,
            loc="best",
        )


        if row == 2:

            ax.set_xlabel(
                r"$\log_{10}(\Sigma_{1.5}/M_\odot\,{\rm kpc}^{-1.5})$",
                fontsize=17,
            )


    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")


    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight",
    )


    plt.close(fig)


    print("Saved:", pdf_path)
    print("Saved:", png_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the six-panel COLIBRE mass-size LOESS mosaic."
    )
    parser.add_argument(
        "--mode",
        choices=("layout", "full", "cached"),
        default="layout",
        help=(
            "layout: instant dummy preview; "
            "full: load data and compute/cache LOESS; "
            "cached: redraw from existing cache"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "layout":
        print("LAYOUT MODE: no simulation data or spline calculation.")
        draw_spline_mosaic(
            make_layout_preview_data(),
            LAYOUT_OUTPUT,
            preview=True,
        )

    elif args.mode == "full":
        print("FULL MODE: loading data and computing spline thresholds.")
        draw_spline_mosaic(
            build_full_cache(),
            FINAL_OUTPUT,
            preview=False,
        )

    else:
        print("CACHED MODE: redrawing from", CACHE_FILE)
        draw_spline_mosaic(
            load_cache(),
            FINAL_OUTPUT,
            preview=False,
        )


if __name__ == "__main__":
    main()
