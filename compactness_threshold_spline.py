#!/usr/bin/env python3
"""
compactness_threshold_spline_standalone.py

Sigma-only compactness-threshold analysis using a weighted smoothing spline
fit to the binned median curve, and the spline derivative to locate the
threshold.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree as KDTree

import common  # your helper that provides read_group_data_colibre

plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "serif", "font.size": 13})

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
# Data loading
# ------------------------------------------------------------------
model_name = "L0200N3008/THERMAL_AGN/"
model_dir = "/mnt/su3-pro/colibre/" + model_name

base_dir = Path("/mnt/su3ctm/kproctor/ForMax")
matches = sorted(base_dir.glob("*exsitu*summary*.hdf5"))
if len(matches) == 0:
    raise FileNotFoundError(f"No ex-situ HDF5 file found in {base_dir}")
h5path = str(max(matches, key=lambda p: p.stat().st_mtime)) if len(matches) > 1 else str(matches[0])

print("Using ex-situ file:", h5path)

snap_file = "0127"
ztarget = 0.0
comov_to_physical_length = 1.0 / (1.0 + ztarget)

outdir = os.path.join(os.getcwd(), "plots")
os.makedirs(outdir, exist_ok=True)

fields_sgn = {
    "InputHalos": (
        "HaloCatalogueIndex",
        "IsCentral",
        "HBTplus/DescendantTrackId",
        "HBTplus/TrackId",
    )
}
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

h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)

m30, sfr30, r50, cp, stellarage, stellarage_lum, FeoverH, MgoverH, Zstar_raw = h5data_groups
sgn, is_central, desc_id, track_id = h5data_idgroups

Lu = 1.0
Mu = 1.988e+43 / (1.989e+33)
tu = 3.086e+19 / (3.154e+7)

m30 = m30 * Mu
sfr30 = sfr30 * Mu / tu
r50 = r50 * Lu * comov_to_physical_length * 1e3
stellarage = stellarage * tu / 1e9
stellarage_lum = stellarage_lum * tu / 1e9

Zsun = 0.0134
Zstar = np.asarray(Zstar_raw, dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    logZstar = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar), np.nan)
    logZstar_rel = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar / Zsun), np.nan)

select = np.where(m30 >= 1e9)
if len(select[0]) == 0:
    raise SystemExit("No galaxies selected (m30 >= 1e9)")

m_in = np.asarray(m30[select]).ravel()
r50_in = np.asarray(r50[select]).ravel()
sfr_in = np.asarray(sfr30[select]).ravel()
stellarage_lum_in = np.asarray(stellarage_lum[select]).ravel()
Fe_in = np.asarray(FeoverH[select]).ravel()
Mg_in = np.asarray(MgoverH[select]).ravel()
sgn_in = np.asarray(sgn[select]).ravel()
logZstar_rel_in = np.asarray(logZstar_rel[select]).ravel()

mask_positive = (m_in > 0) & (r50_in > 0)
if not np.any(mask_positive):
    raise RuntimeError("No positive mass/radius values to plot.")

log_m = np.log10(m_in[mask_positive])
log_r = np.log10(r50_in[mask_positive])
compactness = log_m - 1.5 * log_r

Mg = np.asarray(Mg_in[mask_positive], dtype=float)
Fe = np.asarray(Fe_in[mask_positive], dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    MgFe_number = Mg / Fe
    log10_number = np.where(MgFe_number > 0, np.log10(MgFe_number), np.nan)
    mgfe = log10_number - 0.10

sfr = np.asarray(sfr_in[mask_positive], dtype=float)
m = np.asarray(m_in[mask_positive], dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    ssfr = np.where(m > 0, sfr / m, np.nan)
log_ssfr = np.full_like(ssfr, np.nan, dtype=float)
mask_pos = (ssfr > 0) & np.isfinite(ssfr)
log_ssfr[mask_pos] = np.log10(ssfr[mask_pos])

# ex-situ fractions
halo_selected = np.asarray(sgn_in[mask_positive], dtype=np.int64)
exsitu_fracs = np.full(halo_selected.shape, np.nan, dtype=np.float32)

if os.path.exists(h5path):
    with h5py.File(h5path, "r") as fh:
        if "stars" in fh:
            dset = fh["stars"]
            nrows = dset.shape[0]
            ids_all = np.empty(nrows, dtype=np.int64)
            fracs_all = np.empty(nrows, dtype=np.float32)
            chunk = 500_000
            for i0 in range(0, nrows, chunk):
                i1 = min(nrows, i0 + chunk)
                block = dset[i0:i1, :]
                ids_all[i0:i1] = block[:, 0].astype(np.int64, copy=False)
                fracs_all[i0:i1] = block[:, 3].astype(np.float32, copy=False)

            order = np.argsort(ids_all)
            ids_s = ids_all[order]
            fracs_s = fracs_all[order]
            pos = np.searchsorted(ids_s, halo_selected)
            ok = (pos < ids_s.size) & (ids_s[pos] == halo_selected)
            exsitu_fracs[ok] = fracs_s[pos[ok]]

sigma_path = "/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN/SOAP-HBT/extra/halo_properties_0127.hdf5"
sigma_ds = "/ExclusiveSphere/HalfMassRadiusStars/StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"

mask_positive_full = (m30 >= 1e9) & (m30 > 0) & (r50 > 0)
row_idx = np.flatnonzero(mask_positive_full)
sigma_full = np.full(m30.shape, np.nan, dtype=np.float32)
log_sigma_full = np.full(m30.shape, np.nan, dtype=np.float32)

if os.path.exists(sigma_path):
    with h5py.File(sigma_path, "r") as f:
        ds = f[sigma_ds]
        rows = np.asarray(ds[row_idx, :], dtype=np.float32)

        sigma_rr = rows[:, 0]
        sigma_pphi = rows[:, 4]
        sigma_zz = rows[:, 8]

        sigma_sel = np.sqrt((sigma_rr ** 2 + sigma_pphi ** 2 + sigma_zz ** 2) / 3)
        sigma_full[row_idx] = sigma_sel
        log_sigma_full[row_idx] = np.where(sigma_sel > 0, np.log10(sigma_sel), np.nan)

sigma_vals = sigma_full[mask_positive_full]
log_sigma_vals = log_sigma_full[mask_positive_full]

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

# ------------------------------------------------------------------
# Generic plotting block
# ------------------------------------------------------------------
def run_quantity_block(
    compactness,
    q_arr,
    y_label,
    fig_name,
    search_lo=None,
    search_hi=None,
    target_per_bin=200,
    min_count_per_bin=5,
    spline_s_factor=0.05,
    spline_k=3,
    bootstrap_n=0,
):
    valid_pair = np.isfinite(compactness) & np.isfinite(q_arr)
    if np.sum(valid_pair) == 0:
        raise RuntimeError("No finite compactness/quantity pairs to bin.")

    nbins = max(3, int(np.floor(np.sum(valid_pair) / max(1, int(target_per_bin)))))
    cbins = np.nanpercentile(compactness[valid_pair], np.linspace(0, 100, nbins + 1))
    for i in range(1, len(cbins)):
        if cbins[i] <= cbins[i - 1]:
            cbins[i] = cbins[i - 1] + 1e-9

    cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])
    idxs = np.searchsorted(cbins, compactness, side="right") - 1
    valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(q_arr)

    group = defaultdict(list)
    for idx, q in zip(idxs[valid_mask], q_arr[valid_mask]):
        group[int(idx)].append(float(q))

    cmed = np.full(nbins, np.nan)
    clow = np.full(nbins, np.nan)
    chigh = np.full(nbins, np.nan)
    counts_per_bin = np.zeros(nbins, dtype=int)

    for b in range(nbins):
        vals = group.get(b, [])
        counts_per_bin[b] = len(vals)
        if len(vals) >= min_count_per_bin:
            arr = np.asarray(vals, dtype=float)
            cmed[b] = np.nanmedian(arr)
            clow[b] = np.nanpercentile(arr, 16)
            chigh[b] = np.nanpercentile(arr, 84)

    res = find_compactness_threshold_spline(
        cbin_centers=cbin_centers,
        cmed=cmed,
        compactness_all=compactness[valid_pair],
        quantity_all=q_arr[valid_pair],
        counts_per_bin=counts_per_bin,
        bootstrap_n=bootstrap_n,
        cbins_edges=cbins,
        search_lo=search_lo,
        search_hi=search_hi,
        spline_s_factor=spline_s_factor,
        spline_k=spline_k,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax_main, ax_proj = axes

    ax_main.scatter(compactness, q_arr, s=8, color="lightgrey", alpha=0.8, label="galaxies")
    cf = _plot_density_contours(ax_main, compactness, q_arr)
    if cf is not None:
        fig.colorbar(cf, ax=ax_main, label="Density (KDE)")

    ax_main.set_xlabel(r"$\log_{10}(\Sigma_{1.5}\,[M_\odot\,\mathrm{kpc}^{-1.5}])$")
    ax_main.set_ylabel(y_label)
    ax_main.grid(True)

    ok = np.isfinite(cmed)
    if np.any(ok):
        ax_main.plot(cbin_centers[ok], cmed[ok], color="black", lw=2, label="median (binned)")
        ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15)

    ax_proj.plot(cbin_centers, cmed, color="C0", lw=2)
    ax_proj.fill_between(cbin_centers, clow, chigh, color="C0", alpha=0.25)
    ax_proj.set_xlabel(r"$\log_{10}(\Sigma_{1.5}\,[M_\odot\,\mathrm{kpc}^{-1.5}])$")
    ax_proj.set_ylabel(y_label)
    ax_proj.grid(True)

    ax_proj.plot(cbin_centers, res["cmed_smooth"], linestyle="--", color="C2", lw=2, label="smoothed median")
    ax_der = ax_proj.twinx()
    ax_der.plot(cbin_centers, res["derivative"], color="C3", lw=1.4, alpha=0.9)
    ax_der.axhline(0.0, linestyle=":", color="C3", alpha=0.6)
    ax_der.set_ylabel("Derivative (arb. units)", color="C3")
    ax_der.tick_params(axis="y", labelcolor="C3")

    ax_proj.axvline(res["threshold"], color="C2", linestyle="--", lw=1.6, label=f"thr={res['threshold']:.2f}")

    lines1, labels1 = ax_proj.get_legend_handles_labels()
    lines2, labels2 = ax_der.get_legend_handles_labels()
    ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return res

# ------------------------------------------------------------------
# Run sigma block
# ------------------------------------------------------------------
# res_sigma = run_quantity_block(
#     compactness=compactness,
#     q_arr=log_sigma_vals,
#     y_label=r'$\log_{10}(\sigma \, / \, \mathrm{km}\ \mathrm{s}^{-1})$',
#     fig_name=os.path.join(outdir, "compactness_sigma_two_panel_spline.pdf"),
#     search_lo=None,
#     search_hi=None,
#     target_per_bin=1000,
#     spline_s_factor=0.015, # 0.01 or 0.02; 0.015 ideal
#     spline_k=3,
# )

# print("sigma threshold:", res_sigma["threshold"], res_sigma["method"])

res_mgfe = run_quantity_block(
    compactness=compactness,
    q_arr=mgfe,
    y_label="[Mg/Fe]",
    fig_name=os.path.join(outdir, f"compactness_mgfe_two_panel_spline.pdf"),
    search_lo=None,
    search_hi=None,
    target_per_bin=1000,
    spline_s_factor=0.0007,
    spline_k=3,
)
print("mgfe threshold:", res_mgfe["threshold"], res_mgfe["method"])

# res_age = run_quantity_block(
#     compactness=compactness,
#     q_arr=stellarage_lum_in,
#     y_label="Age [Gyr]",
#     fig_name=os.path.join(outdir, f"compactness_lumage_two_panel_spline.pdf"),
#     search_lo=None,
#     search_hi=None,
#     target_per_bin=1000,
#     spline_s_factor=3.5, #between 3.0 and 5.0 , 3.5 and 3.8 was good 
#     spline_k=3,
# )
# print("age threshold:", res_age["threshold"], res_age["method"])

# res_metallicity = run_quantity_block(
#     compactness=compactness,
#     q_arr=logZstar_rel_in,
#     y_label="[Z/H]",
#     fig_name=os.path.join(outdir, f"compactness_metallicity_two_panel_spline.pdf"),
#     search_lo=9.5,
#     search_hi=10.2,
#     target_per_bin=1000,
#     spline_s_factor=0.005, #0.005 or 0.006 (fallback percentile) but 0.004 reproduces the most similar to the original plot
#     spline_k=3,
# )
# print("metallicity threshold:", res_metallicity["threshold"], res_metallicity["method"])

# res_ssfr = run_quantity_block(
#     compactness=compactness,
#     q_arr=log_ssfr,
#     y_label=r"$\log_{10}(\mathrm{sSFR}\ /\ \mathrm{yr}^{-1})$",
#     fig_name=os.path.join(outdir, f"compactness_ssfr_two_panel_spline.pdf"),
#     search_lo=None,
#     search_hi=None,
#     target_per_bin=1000,
#     spline_s_factor=0.1,
#     spline_k=3,
# )
# print("ssfr threshold:", res_ssfr["threshold"], res_ssfr["method"])

# res_exsitu = run_quantity_block(
#     compactness=compactness,
#     q_arr=exsitu_fracs,
#     y_label=r"$f_\mathrm{ex-situ}$",
#     fig_name=os.path.join(outdir, f"compactness_exsitu_two_panel_spline.pdf"),
#     search_lo=None,
#     search_hi=None,
#     target_per_bin=1000,
#     spline_s_factor=0.02,
#     spline_k=3,
# )

# print("exsitu threshold:", res_exsitu["threshold"], res_exsitu["method"])

if __name__ == "__main__":
    pass