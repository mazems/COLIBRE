#!/usr/bin/env python3
"""
compactness_threshold_spline_standalone.py

Standalone sigma-only version of the compactness-threshold analysis.

This version implements the smoothing-spline workflow exactly in the way your
supervisor described it:
- fit a smoothing spline to the binned median curve
- use the returned B-spline object directly
- obtain the derivative from .derivative()

It keeps the same data loading, binning style, plot style, KDE background, and
threshold logic as your previous script.
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
from scipy.spatial import cKDTree

import common  # your helper that provides read_group_data_colibre

plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "serif", "font.size": 13})


# ------------------------------------------------------------------
# Spline-based threshold finder
# ------------------------------------------------------------------
def _fit_smoothing_spline(
    x: np.ndarray,
    y: np.ndarray,
    spline_s_factor: Optional[float] = 1.0,
    spline_k: int = 5,
):
    """Fit a smoothing spline and return a BSpline-like object."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 4:
        raise ValueError("Need at least 4 finite points to fit a spline.")

    x = x[finite]
    y = y[finite]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Collapse duplicate x values by averaging y values.
    xu, inv = np.unique(x, return_inverse=True)
    if xu.size != x.size:
        y_num = np.zeros_like(xu, dtype=float)
        y_den = np.zeros_like(xu, dtype=float)
        for i, j in enumerate(inv):
            y_num[j] += y[i]
            y_den[j] += 1.0
        x = xu
        y = y_num / np.maximum(y_den, 1e-12)

    if x.size < 4:
        raise ValueError("Not enough unique x values for spline fitting.")

    k = int(min(max(1, spline_k), x.size - 1))
    if spline_s_factor is None:
        spline_s_factor = 1.0

    # s is the smoothing target for splrep. Here we keep the intended workflow
    # simple: a smoothing spline fit, then derivative from the resulting B-spline.
    s = float(spline_s_factor) * x.size

    tck = splrep(x, y, k=k, s=s)
    return BSpline(*tck)


def _bootstrap_bin_medians(
    compactness_all: np.ndarray,
    quantity_all: np.ndarray,
    cbins_edges: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap median per compactness bin using raw galaxy arrays."""
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
    deriv_thresh_factor: float = 3.0,
    min_bin_count: int = 5,
    bootstrap_n: int = 0,
    random_seed: int = 12345,
    cbins_edges: Optional[Sequence[float]] = None,
    search_lo: Optional[float] = None,
    search_hi: Optional[float] = None,
    spline_s_factor: Optional[float] = 1.0,
    spline_k: int = 5,
) -> Dict[str, Any]:
    """Find the compactness threshold using a smoothing spline derivative."""
    cbin_centers = np.asarray(cbin_centers, dtype=float).ravel()
    cmed = np.asarray(cmed, dtype=float).ravel()
    n = cbin_centers.size

    if n < 4:
        raise ValueError("Need at least 4 compactness bins.")

    finite_mask = np.isfinite(cmed) & np.isfinite(cbin_centers)
    if finite_mask.sum() < 4:
        raise ValueError("cmed contains too few finite values for spline fitting.")

    if counts_per_bin is not None:
        counts_per_bin = np.asarray(counts_per_bin, dtype=int).ravel()
        if counts_per_bin.size != n:
            raise ValueError("counts_per_bin must have same length as cbin_centers.")

    spl = _fit_smoothing_spline(
        cbin_centers[finite_mask],
        cmed[finite_mask],
        spline_s_factor=spline_s_factor,
        spline_k=spline_k,
    )

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

            spl_b = _fit_smoothing_spline(
                cbin_centers[finite_bs],
                med_bs[finite_bs],
                spline_s_factor=spline_s_factor,
                spline_k=spline_k,
            )
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


def thresholds_for_targets_spline(
    compactness_all: Sequence[float],
    quantity_all: Sequence[float],
    targets_per_bin: Sequence[int] = (50, 100, 300, 1000),
    min_count: int = 5,
    bootstrap_n: int = 0,
    random_seed: int = 12345,
    do_bootstrap: bool = True,
    search_lo: Optional[float] = None,
    search_hi: Optional[float] = None,
    spline_s_factor: Optional[float] = 1.0,
    spline_k: int = 5,
) -> List[Dict[str, Any]]:
    """Quantile-binned threshold scan using the spline-based derivative."""
    res_list: List[Dict[str, Any]] = []
    compactness_all = np.asarray(compactness_all, dtype=float).ravel()
    quantity_all = np.asarray(quantity_all, dtype=float).ravel()

    N = compactness_all.size
    if N == 0:
        return []

    for target in np.atleast_1d(targets_per_bin):
        target = int(target)
        if target <= 0:
            continue

        nbins = max(3, int(np.floor(N / target)))
        quantiles = np.linspace(0.0, 1.0, nbins + 1)
        edges = np.nanpercentile(compactness_all, 100.0 * quantiles)

        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9

        cbin_centers = 0.5 * (edges[:-1] + edges[1:])
        idxs = np.searchsorted(edges, compactness_all, side="right") - 1
        valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(quantity_all) & np.isfinite(compactness_all)

        bins_values: List[List[float]] = [[] for _ in range(nbins)]
        for ii, q in zip(idxs[valid_mask], quantity_all[valid_mask]):
            bins_values[int(ii)].append(float(q))

        counts = np.zeros(nbins, dtype=int)
        med = np.full(nbins, np.nan, dtype=float)
        low = np.full(nbins, np.nan, dtype=float)
        high = np.full(nbins, np.nan, dtype=float)

        for b in range(nbins):
            vals = bins_values[b]
            counts[b] = len(vals)
            if counts[b] >= min_count:
                arr = np.asarray(vals, dtype=float)
                med[b] = np.nanmedian(arr)
                low[b] = np.nanpercentile(arr, 16)
                high[b] = np.nanpercentile(arr, 84)

        try:
            thr_res = find_compactness_threshold_spline(
                cbin_centers=cbin_centers,
                cmed=med,
                compactness_all=compactness_all,
                quantity_all=quantity_all,
                counts_per_bin=counts,
                bootstrap_n=(bootstrap_n if do_bootstrap else 0),
                random_seed=random_seed,
                cbins_edges=edges,
                search_lo=search_lo,
                search_hi=search_hi,
                spline_s_factor=spline_s_factor,
                spline_k=spline_k,
            )
        except Exception as e:
            thr_res = {
                "threshold": float(np.nanpercentile(compactness_all, 90)),
                "method": f"fallback_error: {e}",
                "bootstrap": None,
            }

        res_list.append(
            {
                "target_per_bin": target,
                "nbins": nbins,
                "edges": edges,
                "cbin_centers": cbin_centers,
                "counts": counts,
                "cmed": med,
                "clow": low,
                "chigh": high,
                "threshold_result": thr_res,
            }
        )

    return res_list


# ------------------------------------------------------------------
# Configuration / dataset selection
# ------------------------------------------------------------------
model_name = "L0200N3008/THERMAL_AGN/"
model_dir = "/mnt/su3-pro/colibre/" + model_name

# Automatically find the ex-situ file
base_dir = Path("/mnt/su3ctm/kproctor/ForMax")
matches = sorted(base_dir.glob("*exsitu*summary*.hdf5"))

if len(matches) == 0:
    raise FileNotFoundError(f"No ex-situ HDF5 file found in {base_dir}")
elif len(matches) == 1:
    h5path = str(matches[0])
else:
    h5path = str(max(matches, key=lambda p: p.stat().st_mtime))

print("Using ex-situ file:", h5path)

# snapshot selection (z=0 in your workflow)
snap_files = ["0127", "0119", "0114", "0102", "0092", "0076", "0064", "0056", "0048", "0040", "0026", "0018"]
zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
snap_file = snap_files[0]
ztarget = zstarget[0]
comov_to_physical_length = 1.0 / (1.0 + ztarget)

outdir = os.path.join(os.getcwd(), "plots")
os.makedirs(outdir, exist_ok=True)


# ------------------------------------------------------------------
# Fields to read
# ------------------------------------------------------------------
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
fields_proj = {"ProjectedAperture/50kpc/projz": ("StellarMass", "HalfMassRadiusStars")}

h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
h5data_groups_proj = common.read_group_data_colibre(model_dir, snap_file, fields_proj)

# unpack
(m30, sfr30, r50, cp, stellarage, stellarage_lum, FeoverH, MgoverH, Zstar_raw) = h5data_groups
(m30_proj, r50_proj) = h5data_groups_proj
(sgn, is_central, desc_id, track_id) = h5data_idgroups

soap_id = {"SOAP": ("HostHaloIndex",)}
h5data_soap = common.read_group_data_colibre(model_dir, snap_file, soap_id)
(host_halo_index) = h5data_soap


# ------------------------------------------------------------------
# Units conversion
# ------------------------------------------------------------------
Lu = 3.086e+24 / (3.086e+24)  # cMpc -> Mpc factor (kept 1)
Mu = 1.988e+43 / (1.989e+33)   # raw mass -> Msun

tu = 3.086e+19 / (3.154e+7)    # time unit -> yrs

m30 = m30 * Mu
m30_proj = m30_proj * Mu
sfr30 = sfr30 * Mu / tu
r50 = r50 * Lu * comov_to_physical_length * 1e3
r50_proj = r50_proj * Lu * comov_to_physical_length * 1e3
stellarage = stellarage * tu / 1e9
stellarage_lum = stellarage_lum * tu / 1e9
cp = cp * Lu * comov_to_physical_length

Zsun = 0.0134  # AGSS09 convention

Zstar = np.asarray(Zstar_raw, dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    logZstar = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar), np.nan)
    logZstar_rel = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar / Zsun), np.nan)


# ------------------------------------------------------------------
# select galaxies
# ------------------------------------------------------------------
select = np.where(m30 >= 1e9)
ngals = len(m30[select])
if ngals == 0:
    raise SystemExit("No galaxies selected (m30 >= 1e9)")

m_in = m30[select]
r50_in = r50[select]
sgn_in = sgn[select]
Fe_in = FeoverH[select]
Mg_in = MgoverH[select]
Zstar_in = Zstar[select]
logZstar_in = logZstar[select]
logZstar_rel_in = logZstar_rel[select]
stellarage_in = stellarage[select]
stellarage_lum_in = stellarage_lum[select]
sfr_in = sfr30[select]
desc_id_in = desc_id[select]
track_id_in = track_id[select]

m_in = np.asarray(m_in).ravel()
r50_in = np.asarray(r50_in).ravel()
Fe_in = np.asarray(Fe_in).ravel()
Mg_in = np.asarray(Mg_in).ravel()
sgn_in = np.asarray(sgn_in).ravel()
Zstar_in = np.asarray(Zstar_in).ravel()
logZstar_in = np.asarray(logZstar_in).ravel()
logZstar_rel_in = np.asarray(logZstar_rel_in).ravel()

mask_positive = (m_in > 0) & (r50_in > 0)
if not np.any(mask_positive):
    raise RuntimeError("No positive mtot/r50 values to plot after filtering selection.")

log_m = np.log10(m_in[mask_positive])
log_r = np.log10(r50_in[mask_positive])
compactness = log_m - 1.5 * log_r

log_MgFe_sun = +0.10
Mg = np.asarray(Mg_in[mask_positive], dtype=float)
Fe = np.asarray(Fe_in[mask_positive], dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    MgFe_number = (Mg / Fe)
    log10_number = np.where(MgFe_number > 0, np.log10(MgFe_number), np.nan)
    mgfe = log10_number - log_MgFe_sun

print(f"Selected galaxies: {len(compactness)} ; Mg/Fe finite: {np.isfinite(mgfe).sum()}")

# choose targets (galaxies per compactness bin) to test
targets = (100, 300, 500, 1000, 5000, 7000, 10000)

# compute sSFR
sfr = np.asarray(sfr_in)[mask_positive]    # Msun/yr
m   = np.asarray(m_in)[mask_positive]      # Msun
with np.errstate(divide='ignore', invalid='ignore'):
    ssfr = np.where(m > 0, sfr / m, np.nan)  # yr^-1

# Safe, warning-free log10 of sSFR:
log_ssfr = np.full_like(ssfr, np.nan, dtype=float)
mask_pos = (ssfr > 0) & np.isfinite(ssfr)
if np.any(mask_pos):
    log_ssfr[mask_pos] = np.log10(ssfr[mask_pos])


# ------------------------------------------------------------------
# Ex-situ fractions from HDF5
# ------------------------------------------------------------------
halo_selected = np.asarray(sgn_in[mask_positive], dtype=np.int64)
exsitu_fracs = np.full(halo_selected.shape, np.nan, dtype=np.float32)

if os.path.exists(h5path):
    with h5py.File(h5path, "r") as fh:
        if "stars" not in fh:
            print("HDF5 file missing dataset 'stars'; skipping ex-situ matching.")
        else:
            dset = fh["stars"]
            nrows = dset.shape[0]

            chunk = 500_000
            ids_all = np.empty(nrows, dtype=np.int64)
            fracs_all = np.empty(nrows, dtype=np.float32)

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

            print(f"Loaded+matched ex-situ in vectorised mode: {ok.sum()} / {halo_selected.size}")
else:
    print("Ex-situ HDF5 not found at:", h5path)

n_matched = np.isfinite(exsitu_fracs).sum()
print(f"Matched ex-situ fraction for {n_matched} / {len(exsitu_fracs)} selected galaxies")


# ------------------------------------------------------------------
# Host velocity dispersion
# ------------------------------------------------------------------
sigma_path = "/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN/SOAP-HBT/extra/halo_properties_0127.hdf5"
sigma_ds = "/ExclusiveSphere/HalfMassRadiusStars/StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"

mask_positive_full = (m30 >= 1e9) & (m30 > 0) & (r50 > 0)
row_idx = np.flatnonzero(mask_positive_full)
sigma_full = np.full(m30.shape, np.nan, dtype=np.float32)
log_sigma_full = np.full(m30.shape, np.nan, dtype=np.float32)

if os.path.exists(sigma_path):
    with h5py.File(sigma_path, "r") as f:
        ds = f[sigma_ds]
        print("sigma dataset shape:", ds.shape)

        rows = np.asarray(ds[row_idx, :], dtype=np.float32)

        sigma_rr = rows[:, 0]
        sigma_pphi = rows[:, 4]
        sigma_zz = rows[:, 8]

        sigma_sel = np.sqrt((sigma_rr ** 2 + sigma_pphi ** 2 + sigma_zz ** 2) / 3)
        sigma_full[row_idx] = sigma_sel
        log_sigma_full[row_idx] = np.where(sigma_sel > 0, np.log10(sigma_sel), np.nan)

    print("Loaded sigma values:", np.isfinite(sigma_sel).sum(), "/", sigma_sel.size)
    print("N(sigma == 0):", np.count_nonzero(np.isclose(sigma_sel[np.isfinite(sigma_sel)], 0.0)))
else:
    print("Sigma file not found.")

sigma_vals = sigma_full[mask_positive_full]
log_sigma_vals = log_sigma_full[mask_positive_full]


# ------------------------------------------------------------------
# Helper for plotting a compactness-vs-quantity panel
# ------------------------------------------------------------------
def _plot_density_contours(ax_main, x, y):
    fin = np.isfinite(x) & np.isfinite(y)
    if np.sum(fin) < 10:
        print("Skipping KDE contours: too few finite points.")
        return None

    xs = x[fin]
    ys = y[fin]
    pts = np.vstack([xs, ys])
    kde = gaussian_kde(pts, bw_method='scott')

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

    tree = cKDTree(np.column_stack((xs, ys)))
    d_grid, _ = tree.query(grid_pts, k=1)
    d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
    typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
    cut = max(typical_spacing * 1.3, 1e-6)
    mask_far = (d_grid > cut)

    Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
    Z_flat = Z.ravel()
    Z_flat[mask_far] = np.nan
    Z = Z_flat.reshape(Xgrid.shape)

    finite_vals = Z[np.isfinite(Z)]
    if finite_vals.size > 0:
        levs = np.percentile(finite_vals, [50, 75, 90, 97])
        cf = ax_main.contourf(Xgrid, Ygrid, Z, levels=50, cmap='viridis', antialiased=True)
        ax_main.contour(Xgrid, Ygrid, Z, levels=levs, colors='k', linewidths=0.6, alpha=0.5)
        return cf
    return None


# # ------------------------------------------------------------------
# # Compactness vs velocity dispersion (two-panel) -- smoothing spline derivative (individual quantity)
# # ------------------------------------------------------------------
# print("sigma finite:", np.isfinite(sigma_vals).sum(), "/", sigma_vals.size)
# print("sigma min/max:", np.nanmin(sigma_vals), np.nanmax(sigma_vals))

# summary_sigma = thresholds_for_targets_spline(
#     compactness,
#     log_sigma_vals,
#     targets_per_bin=targets,
#     min_count=5,
#     bootstrap_n=0,
#     do_bootstrap=False,
#     search_lo=None,
#     search_hi=None,
#     spline_s_factor=1.0,
#     spline_k=5,
# )

# for s in summary_sigma:
#     thr = s['threshold_result']
#     print(
#         f"sigma target={s['target_per_bin']} -> nbins={s['nbins']} "
#         f"threshold={thr['threshold']:.3f} method={thr['method']}"
#     )

# fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})
# ax_main, ax_proj = axes

# ax_main.scatter(compactness, log_sigma_vals, s=8, color="lightgrey", alpha=0.8, label="galaxies")
# cf = _plot_density_contours(ax_main, compactness, log_sigma_vals)
# if cf is not None:
#     fig.colorbar(cf, ax=ax_main, label='Density (KDE)')

# ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_main.set_ylabel(r'$\lg(\sigma \, / \, \mathrm{km}\ \mathrm{s}^{-1})$')
# ax_main.grid(True)

# # ------------------ Running-median / projection ------------------
# use_quantile_bins = True
# target_per_bin = 200
# min_count_per_bin = 5
# q_arr = log_sigma_vals

# valid_pair = np.isfinite(compactness) & np.isfinite(q_arr)
# Ntot = np.sum(valid_pair)
# if Ntot == 0:
#     raise RuntimeError("No finite compactness/sigma pairs to bin.")

# # Build bins on the full valid dataset.
# nbins = max(3, int(np.floor(np.sum(valid_pair) / max(1, int(target_per_bin)))))
# cbins = np.nanpercentile(compactness[valid_pair], np.linspace(0, 100, nbins + 1))
# for i in range(1, len(cbins)):
#     if cbins[i] <= cbins[i - 1]:
#         cbins[i] = cbins[i - 1] + 1e-9

# cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])
# idxs = np.searchsorted(cbins, compactness, side='right') - 1
# valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(q_arr)

# group = defaultdict(list)
# for idx, q in zip(idxs[valid_mask], q_arr[valid_mask]):
#     group[int(idx)].append(float(q))

# cmed = np.full(nbins, np.nan)
# clow = np.full(nbins, np.nan)
# chigh = np.full(nbins, np.nan)
# counts_per_bin = np.zeros(nbins, dtype=int)

# for b in range(nbins):
#     vals = group.get(b, [])
#     counts_per_bin[b] = len(vals)
#     if len(vals) >= min_count_per_bin:
#         arr = np.asarray(vals, dtype=float)
#         cmed[b] = np.nanmedian(arr)
#         clow[b] = np.nanpercentile(arr, 16)
#         chigh[b] = np.nanpercentile(arr, 84)

# ok = np.isfinite(cmed)
# if np.any(ok):
#     ax_main.plot(cbin_centers[ok], cmed[ok], color="black", lw=2, label="median (binned)")
#     ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15)

# ax_proj.plot(cbin_centers, cmed, color="C0", lw=2)
# ax_proj.fill_between(cbin_centers, clow, chigh, color="C0", alpha=0.25)
# ax_proj.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_proj.set_ylabel(r'$\lg(\sigma \, / \, \mathrm{km}\ \mathrm{s}^{-1})$')
# ax_proj.grid(True)

# res_sigma = find_compactness_threshold_spline(
#     cbin_centers=cbin_centers,
#     cmed=cmed,
#     compactness_all=compactness[valid_pair],
#     quantity_all=q_arr[valid_pair],
#     counts_per_bin=counts_per_bin,
#     bootstrap_n=0,
#     cbins_edges=cbins,
#     search_lo=None,
#     search_hi=None,
#     spline_s_factor=1.0,
#     spline_k=5,
# )

# print("sigma threshold:", res_sigma["threshold"], res_sigma["method"])

# ax_proj.plot(cbin_centers, res_sigma["cmed_smooth"], linestyle="--", color="C2", lw=2, label="smoothed median")

# ax_der = ax_proj.twinx()
# ax_der.plot(cbin_centers, res_sigma["derivative"], color="C3", lw=1.4, alpha=0.9)
# ax_der.axhline(0.0, linestyle=":", color="C3", alpha=0.6)
# ax_der.set_ylabel("Derivative (arb. units)", color="C3")
# ax_der.tick_params(axis="y", labelcolor="C3")

# ax_proj.axvline(res_sigma["threshold"], color="C2", linestyle="--", lw=1.6, label=f"thr={res_sigma['threshold']:.2f}")

# lines1, labels1 = ax_proj.get_legend_handles_labels()
# lines2, labels2 = ax_der.get_legend_handles_labels()
# ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# fig.tight_layout()
# fig_name = os.path.join(outdir, f"compactness_sigma_two_panel_z{ztarget:.1f}.png")
# fig.savefig(fig_name, dpi=300, bbox_inches='tight')
# plt.close(fig)

# print("Saved sigma plot to:", fig_name)

# function for automatic processing of any quantity
def run_quantity_block(
    compactness,
    q_arr,
    y_label,
    fig_name,
    search_lo=None,
    search_hi=None,
    target_per_bin=200,
    min_count_per_bin=5,
    spline_s_factor=1.0,
    spline_k=5,
    bootstrap_n=0,
):
    valid_pair = np.isfinite(compactness) & np.isfinite(q_arr)
    if np.sum(valid_pair) == 0:
        raise RuntimeError("No finite compactness/quantity pairs to bin.")

    # build bins
    nbins = max(3, int(np.floor(np.sum(valid_pair) / max(1, int(target_per_bin)))))
    cbins = np.nanpercentile(compactness[valid_pair], np.linspace(0, 100, nbins + 1))
    for i in range(1, len(cbins)):
        if cbins[i] <= cbins[i - 1]:
            cbins[i] = cbins[i - 1] + 1e-9

    cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])
    idxs = np.searchsorted(cbins, compactness, side='right') - 1
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

    # threshold finder
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

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax_main, ax_proj = axes

    ax_main.scatter(compactness, q_arr, s=8, color="lightgrey", alpha=0.8, label="galaxies")
    cf = _plot_density_contours(ax_main, compactness, q_arr)
    if cf is not None:
        fig.colorbar(cf, ax=ax_main, label='Density (KDE)')

    ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
    ax_main.set_ylabel(y_label)
    ax_main.grid(True)

    ok = np.isfinite(cmed)
    if np.any(ok):
        ax_main.plot(cbin_centers[ok], cmed[ok], color="black", lw=2, label="median (binned)")
        ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15)

    ax_proj.plot(cbin_centers, cmed, color="C0", lw=2)
    ax_proj.fill_between(cbin_centers, clow, chigh, color="C0", alpha=0.25)
    ax_proj.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
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

res_sigma = run_quantity_block(
    compactness=compactness,
    q_arr=log_sigma_vals,
    y_label=r'$\lg(\sigma \, / \, \mathrm{km}\ \mathrm{s}^{-1})$',
    fig_name=os.path.join(outdir, f"compactness_sigma_two_panel_spline.png"),
    search_lo=None,
    search_hi=None,
    target_per_bin=1000,
    spline_s_factor=0.1,
    spline_k=3,
)
print("sigma threshold:", res_sigma["threshold"], res_sigma["method"])

res_mgfe = run_quantity_block(
    compactness=compactness,
    q_arr=mgfe,
    y_label="[Mg/Fe]",
    fig_name=os.path.join(outdir, f"compactness_mgfe_two_panel_spline.png"),
    search_lo=None,
    search_hi=None,
    target_per_bin=1000,
    spline_s_factor=0.1,
    spline_k=3,
)
print("mgfe threshold:", res_mgfe["threshold"], res_mgfe["method"])

res_age = run_quantity_block(
    compactness=compactness,
    q_arr=stellarage_lum_in,
    y_label="Age [Gyr]",
    fig_name=os.path.join(outdir, f"compactness_lumage_two_panel_spline.png"),
    search_lo=8.0,
    search_hi=10.0,
    target_per_bin=200,
    spline_s_factor=0.1,
    spline_k=3,
)
print("age threshold:", res_age["threshold"], res_age["method"])

res_metallicity = run_quantity_block(
    compactness=compactness,
    q_arr=logZstar_rel_in,
    y_label="[Z/H]",
    fig_name=os.path.join(outdir, f"compactness_metallicity_two_panel_spline.png"),
    search_lo=None,
    search_hi=None,
    target_per_bin=1000,
    spline_s_factor=0.1,
    spline_k=3,
)
print("metallicity threshold:", res_metallicity["threshold"], res_metallicity["method"])

res_ssfr = run_quantity_block(
    compactness=compactness,
    q_arr=log_ssfr,
    y_label=fr'$\lg(sSFR / \mathrm{yr}$^{-1})$',
    fig_name=os.path.join(outdir, f"compactness_ssfr_two_panel_spline.png"),
    search_lo=8.0,
    search_hi=10.0,
    target_per_bin=1000,
    spline_s_factor=0.1,
    spline_k=3,
)
print("ssfr threshold:", res_ssfr["threshold"], res_ssfr["method"])

res_exsitu = run_quantity_block(
    compactness=compactness,
    q_arr=exsitu_fracs,
    y_label="[Z/H]",
    fig_name=os.path.join(outdir, f"compactness_exsitu_two_panel_spline.png"),
    search_lo=None,
    search_hi=None,
    target_per_bin=1000,
    spline_s_factor=0.1,
    spline_k=3,
)

print("exsitu threshold:", res_exsitu["threshold"], res_exsitu["method"])

if __name__ == "__main__":
    # Running the file directly executes the full analysis above.
    pass