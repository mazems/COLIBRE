#!/usr/bin/env python3
"""
DoRandtime_relations.py  -- self-consistent single-file version

- Matches UCMG CSV (by subhalo_id / HaloCatalogueIndex) to SOAP (HaloCatalogueIndex).
- Loads SOAP-derived mass, r50, sfr, age, Mg/Fe (linear proxies).
- Loads ex-situ summary HDF5 by track id (if available).
- Produces both scatter and LOESS mass-size plots for multiple DoR variants and time columns.
- Produces per-mass-bin statistics and saves a CSV with medians/percentiles for later plotting.
- Does NOT modify any input files.
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.colors import Normalize
import h5py
import common
from scipy.spatial import cKDTree as KDTree

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 13
})

# ------------------------ CONFIG ------------------------
csv_in = "sfh_times_all_with_DoR_variants_corrected.csv.gz"
exsitu_h5 = "/mnt/su3ctm/kproctor/ForMax/exsitu_summary_SnapNum_127.hdf5"  # optional
model_name = 'L0200N3008/THERMAL_AGN/'
model_dir  = '/mnt/su3-pro/colibre/' + model_name
snap_file  = '0127'   # z=0
ztarget    = 0.0
comov_to_physical_length = 1.0 / (1.0 + ztarget)

outdir = "plots_dor"
os.makedirs(outdir, exist_ok=True)

COMPACTNESS_CUT = 9.72
EXTREME_DOR = 0.6
dor_column_candidates = ["DoR_t95"] #, "DoR_t998", "DoR_t90", "DoR_tfin", "dor", "DoR", "DoR_choice", "DoR_csv"]

# LOESS evaluation budget (None => all points)
MAX_EVAL_PTS = 12000

# ------------------------ SMALL HELPERS (define early) ------------------------
def save_fig(fig, fname):
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close(fig)

# binned median + 16/84 percentiles plotting helper without loess (used in many places)
def plot_dor_vs_quantity(x, y, xlabel, fname, color_arr=None, xlim=None):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 6:
        print(f"Skipping {fname} — insufficient finite points ({ok.sum()}).")
        return

    fig, ax = plt.subplots(figsize=(7,5))

    if color_arr is not None:
        order = np.argsort(color_arr[ok])[::-1]
        sc = ax.scatter(x[ok][order], y[ok][order], c=color_arr[ok][order], s=12, alpha=0.5, vmin=-0.2, vmax=0.2)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(r"$\lg[Z_\star / Z_\odot]$")
        cbar.solids.set_alpha(1)
    else:
        ax.scatter(x[ok], y[ok], s=12, alpha=0.7)

    q = np.linspace(0, 100, 15)
    bins = np.unique(np.percentile(x[ok], q))
    if bins.size < 2:
        save_fig(fig, fname)
        return

    xc = 0.5 * (bins[:-1] + bins[1:])
    med = np.full_like(xc, np.nan, dtype=float)
    lo = np.full_like(xc, np.nan, dtype=float)
    hi = np.full_like(xc, np.nan, dtype=float)

    for i in range(len(xc)):
        sel = (x >= bins[i]) & (x < bins[i+1]) & ok
        if sel.sum() > 4:
            vals = y[sel]
            med[i] = np.nanmedian(vals)
            lo[i] = np.nanpercentile(vals, 16)
            hi[i] = np.nanpercentile(vals, 84)

    finite_med = np.isfinite(med)
    if finite_med.sum() > 0:
        ax.plot(xc[finite_med], med[finite_med], color="black", lw=2)
        ax.fill_between(xc[finite_med], lo[finite_med], hi[finite_med], color="black", alpha=0.2)

    ax.axhline(EXTREME_DOR, color='C1', linestyle='--', lw=1.5, label=f"relic threshold DoR={EXTREME_DOR}")
    if isinstance(xlabel, str) and "compact" in xlabel.lower():
        ax.axvline(COMPACTNESS_CUT, color='black', linestyle='--', lw=1.5,
                   label=fr"compactness threshold $\lg\Sigma_{{1.5}}={COMPACTNESS_CUT}$")

    ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("DoR")
    ax.set_ylim(0, 1)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(True)

    save_fig(fig, fname)

# def _plot_dor_vs_quantity_core(
#     ax,
#     x,
#     y,
#     xlabel,
#     color_arr=None,
#     cbar_label="Ex-situ mass fraction",
# ):
#     ok = np.isfinite(x) & np.isfinite(y)
#     if color_arr is not None:
#         ok &= np.isfinite(color_arr)

#     if ok.sum() < 6:
#         print(f"Skipping {xlabel} — insufficient finite points ({ok.sum()}).")
#         return None

#     sc = None

#     if color_arr is not None:
#         order = np.argsort(color_arr[ok])   # ascending, exactly as before
#         sc = ax.scatter(
#             x[ok][order],
#             y[ok][order],
#             c=color_arr[ok][order],
#             s=12,
#             alpha=0.5
#         )
#     else:
#         ax.scatter(x[ok], y[ok], s=12, alpha=0.7)

#     # binned median using percentile bins in X
#     q = np.linspace(0, 100, 15)
#     bins = np.percentile(x[ok], q)
#     bins = np.unique(bins)

#     if bins.size >= 2:
#         xc = 0.5 * (bins[:-1] + bins[1:])
#         med = np.full_like(xc, np.nan, dtype=float)
#         lo = np.full_like(xc, np.nan, dtype=float)
#         hi = np.full_like(xc, np.nan, dtype=float)

#         for i in range(len(xc)):
#             sel = (x >= bins[i]) & (x < bins[i + 1]) & ok
#             if sel.sum() > 4:
#                 vals = y[sel]
#                 med[i] = np.nanmedian(vals)
#                 lo[i] = np.nanpercentile(vals, 16)
#                 hi[i] = np.nanpercentile(vals, 84)

#         finite_med = np.isfinite(med)
#         if finite_med.sum() > 0:
#             ax.plot(xc[finite_med], med[finite_med], color="black", lw=2)
#             ax.fill_between(
#                 xc[finite_med],
#                 lo[finite_med],
#                 hi[finite_med],
#                 color="black",
#                 alpha=0.2
#             )

#     ax.axhline(
#         EXTREME_DOR,
#         color="C1",
#         linestyle="--",
#         lw=1.5,
#         label=f"relic threshold DoR={EXTREME_DOR}"
#     )

#     if isinstance(xlabel, str) and "compact" in xlabel.lower():
#         ax.axvline(
#             COMPACTNESS_CUT,
#             color="black",
#             linestyle="--",
#             lw=1.5,
#             label=fr"compactness threshold $\lg\Sigma_{{1.5}}={COMPACTNESS_CUT}$"
#         )

#     ax.legend(fontsize=8)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel("DoR")
#     ax.set_ylim(0, 1)
#     ax.grid(True)

#     return sc


# def plot_dor_vs_quantity(x, y, xlabel, fname, color_arr=None):
#     fig, ax = plt.subplots(figsize=(7, 5))

#     sc = _plot_dor_vs_quantity_core(
#         ax=ax,
#         x=x,
#         y=y,
#         xlabel=xlabel,
#         color_arr=color_arr
#     )

#     if sc is not None and color_arr is not None:
#         cbar = fig.colorbar(sc, ax=ax)
#         cbar.set_label("Ex-situ mass fraction")
#         cbar.solids.set_alpha(1)

#     save_fig(fig, fname)

# plotting helper with loess
def loess_coloured_dor_vs_quantity(xvals, yvals, zvals, xlabel, fname,
                                    cbar_label=None, nx=300, ny=220,
                                    pad_frac=0.05, max_eval_pts=None):
    """
    xvals = quantity on x-axis
    yvals = DoR on y-axis
    zvals = colour quantity (e.g. log_ssfr)
    """
    ok = np.isfinite(xvals) & np.isfinite(yvals) & np.isfinite(zvals)
    if ok.sum() < 6:
        print(f"LOESS: too few finite values for {fname}; saving scatter fallback.")
        return

    x_in = np.asarray(xvals[ok], dtype=float)
    y_in = np.asarray(yvals[ok], dtype=float)
    z_in = np.asarray(zvals[ok], dtype=float)

    N = x_in.size
    if (max_eval_pts is not None) and (N > int(max_eval_pts)):
        rng = np.random.default_rng(12345)
        sel = rng.choice(N, size=int(max_eval_pts), replace=False)
        x_loess = x_in[sel].copy()
        y_loess = y_in[sel].copy()
        z_loess = z_in[sel].copy()
    else:
        x_loess = x_in.copy()
        y_loess = y_in.copy()
        z_loess = z_in.copy()

    pad_x = pad_frac * (np.nanmax(x_loess) - np.nanmin(x_loess) + 1e-6)
    pad_y = pad_frac * (np.nanmax(y_loess) - np.nanmin(y_loess) + 1e-6)

    xg = np.linspace(np.nanmin(x_loess) - pad_x, np.nanmax(x_loess) + pad_x, nx)
    yg = np.linspace(np.nanmin(y_loess) - pad_y, np.nanmax(y_loess) + pad_y, ny)
    Xg, Yg = np.meshgrid(xg, yg)

    pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

    tree_data = KDTree(np.column_stack((x_loess, y_loess)))
    d_grid, _ = tree_data.query(pts_grid, k=1)
    d_data, _ = tree_data.query(np.column_stack((x_loess, y_loess)), k=2)

    if d_data.ndim == 2 and d_data.shape[1] >= 2:
        typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
    else:
        typical_spacing = float(np.nanmedian(d_grid))

    d_thresh = max(typical_spacing * 2, 1e-6)
    inside_mask = (d_grid <= d_thresh)
    idx_inside = np.nonzero(inside_mask)[0]
    # # Evaluate loess field on full grid instead of masking
    # idx_inside = np.arange(pts_grid.shape[0])

    Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
    if idx_inside.size > 0:
        xout = pts_grid[idx_inside, 0]
        yout = pts_grid[idx_inside, 1]
        frac_loess = 0.10
        degree = 1
        Zflat_inside, _ = loess_2d(
            x_loess, y_loess, z_loess,
            frac=frac_loess, degree=degree,
            xout=xout, yout=yout
        )
        Zflat[idx_inside] = Zflat_inside

    Zgrid = Zflat.reshape((ny, nx))
    Zmask = np.ma.masked_invalid(Zgrid)

    try:
        vmin = float(np.nanpercentile(z_in, 5)) #vmin = np.nanmin(z_in) 
        vmax = float(np.nanpercentile(z_in, 95)) #vmax = np.nanmax(z_in) 
        
    except Exception:
        vmin, vmax = float(np.nanmin(z_in)), float(np.nanmax(z_in))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        med = float(np.nanmedian(z_in))
        span = max(0.2, 0.5 * max(1e-6, abs(med)))
        vmin = med - span
        vmax = med + span

    fig, ax = plt.subplots(figsize=(7, 5))
    # background data
    ax.scatter(x_in, y_in, s=5, color="lightgrey", alpha=0.8, zorder=0)
    # loess field
    im = ax.pcolormesh(Xg, Yg, Zmask, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax, zorder=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label or r"$\lg(\mathrm{sSFR}\ /\ \mathrm{yr}^{-1})$")

    if idx_inside.size > 0:
        ax.scatter(pts_grid[idx_inside, 0], pts_grid[idx_inside, 1],
                   s=1, c="k", alpha=0.03, linewidths=0)

    ax.axhline(EXTREME_DOR, color="C1", linestyle="--", lw=1.5,
               label=f"relic threshold DoR={EXTREME_DOR}")
    if isinstance(xlabel, str) and "compact" in xlabel.lower():
        ax.axvline(COMPACTNESS_CUT, color="black", linestyle="--", lw=1.5,
                   label=fr"CMG threshold $\lg\Sigma_{{1.5}}={COMPACTNESS_CUT}$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("DoR")
    ax.legend(fontsize=8)
    ax.grid(True)

    # binned median using percentile bins in X (so each bin contains variable counts)
    q = np.linspace(0, 100, 15)
    bins = np.percentile(xvals[ok], q)
    # avoid repeated edges
    bins = np.unique(bins)
    if bins.size < 2:
        save_fig(fig, fname)
        return
    xc = 0.5 * (bins[:-1] + bins[1:])
    med = np.full_like(xc, np.nan, dtype=float)
    lo = np.full_like(xc, np.nan, dtype=float)
    hi = np.full_like(xc, np.nan, dtype=float)
    for i in range(len(xc)):
        sel = (xvals >= bins[i]) & (xvals < bins[i+1]) & ok
        if sel.sum() > 4:
            vals = yvals[sel]
            med[i] = np.nanmedian(vals)
            lo[i] = np.nanpercentile(vals, 16)
            hi[i] = np.nanpercentile(vals, 84)
    finite_med = np.isfinite(med)
    if finite_med.sum() > 0:
        ax.plot(xc[finite_med], med[finite_med], color="black", lw=2, zorder=5)
        ax.fill_between(xc[finite_med], lo[finite_med], hi[finite_med], color="black", alpha=0.2, zorder=4)

    save_fig(fig, fname)

# ---- LOESS internals ---------------------------------------------------
def polyfit_2d(x, y, z, degree=1, weights=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    W = np.ones_like(z, dtype=float) if weights is None else np.asarray(weights).ravel()
    xc = np.average(x, weights=W)
    yc = np.average(y, weights=W)
    dx = x - xc
    dy = y - yc
    if degree == 0:
        sw = W.sum()
        if sw == 0:
            return np.array([np.nan])
        a0 = (W @ z) / sw
        return np.array([a0])
    A = np.column_stack((np.ones_like(dx), dx, dy))
    ATW = (A.T * W)
    ATA = ATW @ A
    ATy = ATW @ z
    ridge = 1e-12 * np.trace(ATA) if np.isfinite(np.trace(ATA)) and np.trace(ATA) != 0 else 1e-12
    try:
        ATA[0, 0] += ridge
        beta = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(ATA) @ ATy
    return np.array(beta)

def _biweight_scale(resid):
    resid = np.abs(resid)
    if resid.size == 0:
        return 1.0
    mad = np.median(resid)
    if mad <= 0:
        return 1e-9
    return 1.4826 * mad

def loess_2d(x1, y1, z, frac=0.5, degree=1, npoints=None, xout=None, yout=None, sigz=None):
    x1 = np.asarray(x1).ravel()
    y1 = np.asarray(y1).ravel()
    z = np.asarray(z).ravel()
    if not (x1.size == y1.size == z.size):
        raise ValueError("X, Y, Z must be same length")
    n = x1.size
    if n == 0:
        return np.array([]), np.array([])
    if npoints is None:
        npoints = int(np.ceil(frac * n))
    npoints = max(2, min(npoints, n))
    if xout is None or yout is None:
        xout = x1.copy()
        yout = y1.copy()
    xout = np.asarray(xout).ravel()
    yout = np.asarray(yout).ravel()
    if xout.size != yout.size:
        raise ValueError("xout and yout same length required")
    m = xout.size
    zout = np.empty(m, dtype=float)
    wout = np.empty(m, dtype=float)
    tree = KDTree(np.column_stack((x1, y1)))
    for j, (xx, yy) in enumerate(zip(xout, yout)):
        dists, inds = tree.query([xx, yy], k=npoints)
        if np.isscalar(dists):
            dists = np.array([dists]); inds = np.array([inds])
        rmax = np.max(dists)
        if rmax == 0:
            zout[j] = z[inds[0]]
            wout[j] = 1.0
            continue
        u = dists / rmax
        distWeights = (1.0 - u**3)**3
        distWeights = np.where(u >= 1.0, 0.0, distWeights)
        xw = x1[inds]; yw = y1[inds]; zw = z[inds]
        w_init = distWeights.copy()
        coeffs = polyfit_2d(xw, yw, zw, degree=degree, weights=w_init)
        if degree == 0:
            zfit = np.full_like(zw, coeffs[0], dtype=float)
        else:
            xc = np.average(xw, weights=w_init)
            yc = np.average(yw, weights=w_init)
            dx = xw - xc; dy = yw - yc
            a0, ax, ay = coeffs
            zfit = a0 + ax * dx + ay * dy
        biWeights = np.ones_like(zw)
        for it in range(10):
            if sigz is None:
                resid = zfit - zw
                scale = _biweight_scale(resid)
                uu = (np.abs(resid) / (6.0 * scale)) ** 2.0
            else:
                uu = ((zfit - zw) / (4.0 * sigz[inds])) ** 2.0
            uu = np.clip(uu, 0.0, 1.0)
            biWeights_new = (1.0 - uu) ** 2.0
            totWeights = distWeights * biWeights_new
            coeffs = polyfit_2d(xw, yw, zw, degree=degree, weights=totWeights)
            if degree == 0:
                zfit = np.full_like(zw, coeffs[0], dtype=float)
            else:
                xc = np.average(xw, weights=totWeights) if np.sum(totWeights) > 0 else np.mean(xw)
                yc = np.average(yw, weights=totWeights) if np.sum(totWeights) > 0 else np.mean(yw)
                dx = xw - xc; dy = yw - yc
                a0, ax, ay = coeffs
                zfit = a0 + ax * dx + ay * dy
            if np.allclose(biWeights, biWeights_new, atol=1e-6):
                biWeights = biWeights_new
                break
            biWeights = biWeights_new
        zout[j] = coeffs[0]
        wout[j] = biWeights[0] if biWeights.size > 0 else 1.0
    return zout, wout

# scatter plot helper
def scatter_coloured_mass_size(xvals, yvals, zvals, fname, cbar_label=None):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(logM, logR, s=6, color="lightgrey", alpha=0.5, label=f"simulated galaxies at $z=0$")
    finite = np.isfinite(zvals)
    if np.any(finite):
        sc = ax.scatter(xvals[finite], yvals[finite], c=zvals[finite], cmap="viridis", s=18, edgecolors="none", alpha=0.9)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label if cbar_label is not None else fname)
    if np.any(~finite):
        ax.scatter(xvals[~finite], yvals[~finite], color="lightgrey", s=8, alpha=0.6, label="missing")
    xm = np.linspace(np.nanmin(logM)-0.1, np.nanmax(logM)+0.1, 400)
    yr = (xm - COMPACTNESS_CUT) / 1.5
    ax.plot(xm, yr, linestyle='--', color='black', label=f"compactness = {COMPACTNESS_CUT}")
    ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
    ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
    ax.legend(fontsize=8)
    ax.grid(True)
    save_fig(fig, fname)

# LOESS wrapper (subsampling allowed)
def loess_coloured_mass_size(xvals, yvals, zvals, fname, cbar_label=None, nx=300, ny=220, pad_frac=0.05, max_eval_pts=MAX_EVAL_PTS):
    if np.sum(np.isfinite(zvals)) < 2:
        print(f"LOESS: too few finite values for {fname}; saving scatter fallback.")
        scatter_coloured_mass_size(xvals, yvals, zvals, fname.replace(".png", "_fallback_scatter.png"), cbar_label=cbar_label)
        return

    finite_idx = np.where(np.isfinite(zvals))[0]
    x_in = xvals[finite_idx].astype(float); y_in = yvals[finite_idx].astype(float); z_in = zvals[finite_idx].astype(float)

    N = x_in.size
    if (max_eval_pts is not None) and (N > int(max_eval_pts)):
        rng = np.random.default_rng(seed=12345)
        sel = rng.choice(N, size=int(max_eval_pts), replace=False)
        x_loess = x_in[sel].copy(); y_loess = y_in[sel].copy(); z_loess = z_in[sel].copy()
    else:
        x_loess = x_in.copy(); y_loess = y_in.copy(); z_loess = z_in.copy()

    pad_x = pad_frac * (np.nanmax(x_loess) - np.nanmin(x_loess) + 1e-6)
    pad_y = pad_frac * (np.nanmax(y_loess) - np.nanmin(y_loess) + 1e-6)
    xg = np.linspace(np.nanmin(x_loess) - pad_x, np.nanmax(x_loess) + pad_x, nx)
    yg = np.linspace(np.nanmin(y_loess) - pad_y, np.nanmax(y_loess) + pad_y, ny)
    Xg, Yg = np.meshgrid(xg, yg)
    pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

    tree_data = KDTree(np.column_stack((x_loess, y_loess)))
    d_grid, _ = tree_data.query(pts_grid, k=1)
    d_data, _ = tree_data.query(np.column_stack((x_loess, y_loess)), k=2)
    if d_data.ndim == 2 and d_data.shape[1] >= 2:
        typical_spacing = float(np.nanpercentile(d_data[:,1], 95))
    else:
        typical_spacing = float(np.nanmedian(d_grid))
    d_thresh = max(typical_spacing * 1.3, 1e-6)
    inside_mask = (d_grid <= d_thresh)
    idx_inside = np.nonzero(inside_mask)[0]

    Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
    if idx_inside.size > 0:
        xout = pts_grid[idx_inside,0]; yout = pts_grid[idx_inside,1]
        frac_loess = 0.10; degree = 1
        Zflat_inside, _ = loess_2d(x_loess, y_loess, z_loess, frac=frac_loess, degree=degree, xout=xout, yout=yout)
        Zflat[idx_inside] = Zflat_inside

    Zgrid = Zflat.reshape((ny, nx))
    Zmask = np.ma.masked_invalid(Zgrid)

    try:
        vmin = float(np.nanpercentile(z_in, 5)); vmax = float(np.nanpercentile(z_in, 95))
    except Exception:
        vmin, vmax = float(np.nanmin(z_in)), float(np.nanmax(z_in))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        med = float(np.nanmedian(z_in)); span = max(0.2, 0.5 * max(1e-6, abs(med)))
        vmin = med - span; vmax = med + span

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(logM, logR, s=6, color="lightgrey", alpha=0.5, label=f"simulated galaxies at $z=0$")
    im = ax.pcolormesh(Xg, Yg, Zmask, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label if cbar_label is not None else fname)
    if idx_inside.size > 0:
        ax.scatter(pts_grid[idx_inside,0], pts_grid[idx_inside,1], s=1, c='k', alpha=0.03, linewidths=0)
    xm = np.linspace(np.nanmin(logM)-0.1, np.nanmax(logM)+0.1, 400)
    yr = (xm - COMPACTNESS_CUT) / 1.5
    ax.plot(xm, yr, linestyle='--', color='black', label=f"compactness = {COMPACTNESS_CUT}")
    ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
    ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
    ax.legend(fontsize=8)
    ax.grid(True)
    save_fig(fig, fname)

# ------------------------ READ CSV (DoR & time columns) ------------------------
if not os.path.exists(csv_in):
    raise SystemExit(f"UCMG CSV not found: {csv_in}")
print("Reading:", csv_in)
df_ucmg = pd.read_csv(csv_in, low_memory=False)

# canonical id column selection
id_col = None
for c in ("subhalo_id", "HaloCatalogueIndex", "subhaloId", "HaloIndex", "track_id", "TrackId"):
    if c in df_ucmg.columns:
        id_col = c; break
if id_col is None:
    id_col = df_ucmg.columns[0]; print("Warning: no id column found, using", id_col)

# normalize numeric ids -> 'subhalo_id' (int)
s = df_ucmg[id_col].astype(str).str.replace("\r", "").str.strip()
df_ucmg["_subhalo_id_numeric"] = pd.to_numeric(s, errors="coerce").astype("Int64")
n_bad = int(df_ucmg["_subhalo_id_numeric"].isna().sum())
if n_bad > 0:
    print(f"Warning: {n_bad} rows have non-numeric {id_col}; they will be ignored for matching.")
df_ucmg = df_ucmg[df_ucmg["_subhalo_id_numeric"].notna()].copy()
df_ucmg["subhalo_id"] = df_ucmg["_subhalo_id_numeric"].astype("int64")
df_ucmg.drop(columns=["_subhalo_id_numeric"], inplace=True)

# build index for fast reindexing by subhalo_id later
df_ucmg_indexed = df_ucmg.set_index("subhalo_id", drop=False)

# pick DoR column candidates present
dor_cols_found = [c for c in dor_column_candidates if c in df_ucmg_indexed.columns]
if len(dor_cols_found) == 0:
    # fallback: any col starting with 'dor' (case-insensitive)
    for c in df_ucmg_indexed.columns:
        if c.lower().startswith("dor"):
            dor_cols_found.append(c)
if len(dor_cols_found) == 0:
    raise SystemExit("No DoR column found in CSV.")
print("DoR columns present in CSV (candidates):", dor_cols_found)

# ------------------------ READ SOAP ------------------------
print("Reading SOAP groups (common.read_group_data_colibre)...")
fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')}
fields = {'ExclusiveSphere/50kpc': (
            'StellarMass', 'StarFormationRate', 'HalfMassRadiusStars',
            'MassWeightedMeanStellarAge', 'LuminosityWeightedMeanStellarAge',
            'LinearMassWeightedIronOverHydrogenOfStars',
            'LinearMassWeightedMagnesiumOverHydrogenOfStars', 'MostMassiveBlackHoleMass', 'StellarMassFractionInMetals'
         )}

h5data_groups   = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
(halo_index, is_central, desc_id, track_id) = h5data_idgroups
(m30, sfr30, r50, stellarage, stellarage_lum, Fe_lin, Mg_lin, bh_mass_raw, Zstar_raw) = h5data_groups

soap_id = {'SOAP': ('HostHaloIndex',)}
h5data_soap = common.read_group_data_colibre(model_dir, snap_file, soap_id)
(host_halo_index) = h5data_soap

# unit conversions
Mu = 1.988e43 / 1.989e33
tu = 3.086e19 / 3.154e7
m30 = m30 * Mu
sfr30 = sfr30 * Mu / tu
r50 = r50 * comov_to_physical_length * 1e3
stellarage_lum = stellarage_lum * tu / 1e9
bh_mass = bh_mass_raw * Mu  

Zsun = 0.0134   # AGSS09 convention
# Zsun = 0.0139 # Asplund et al. 2021 present-day photospheric value
    
Zstar = np.asarray(Zstar_raw, dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    logZstar = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar), np.nan)
    logZstar_rel = np.where((Zstar > 0) & np.isfinite(Zstar),
                            np.log10(Zstar / Zsun),
                            np.nan)

# selection and masking
select = np.where(m30 >= 1e9)
m = m30[select]; r = r50[select]; halo_idx = halo_index[select]; track = track_id[select]
sfr = sfr30[select]; Mg_lin = Mg_lin[select]; Fe_lin = Fe_lin[select]; age = stellarage_lum[select]
bh_mass = bh_mass[select] 
Zstar = Zstar[select]
logZstar = logZstar[select]
logZstar_rel = logZstar_rel[select]
is_central_selected = is_central[select] 
mask_pos = (m > 0) & (r > 0)
m = m[mask_pos]; r = r[mask_pos]; halo_idx = halo_idx[mask_pos]; track = track[mask_pos]
sfr = sfr[mask_pos]; Mg_lin = Mg_lin[mask_pos]; Fe_lin = Fe_lin[mask_pos]; age = age[mask_pos]
bh_mass = bh_mass[mask_pos] 
Zstar = Zstar[mask_pos]
logZstar = logZstar[mask_pos]
logZstar_rel = logZstar_rel[mask_pos]
is_central_selected = is_central_selected[mask_pos]  
is_central_selected = np.asarray(is_central_selected)
print(f"Selected SOAP galaxies after mass/radius filter: {len(m)}")

# derived
logM = np.log10(m); logR = np.log10(r)
compactness = logM - 1.5 * logR
with np.errstate(divide="ignore", invalid="ignore"):
    mgfe = np.where((Mg_lin > 0) & (Fe_lin > 0),
                    np.log10(Mg_lin / Fe_lin) - 0.10,
                    np.nan)

    ssfr = np.where((m > 0) & np.isfinite(sfr),
                    sfr / m,
                    np.nan)

    log_ssfr = np.log10(ssfr)
    # --- apply floor ---
    SSFR_FLOOR = -12.0
    log_ssfr[~np.isfinite(log_ssfr)] = SSFR_FLOOR
    log_ssfr = np.clip(log_ssfr, SSFR_FLOOR, None)
    bh_ratio = np.where((bh_mass > 0) & (m > 0) & np.isfinite(bh_mass) & np.isfinite(m),
                        bh_mass / m,
                        np.nan)
    log_bh_ratio = np.where(np.isfinite(bh_ratio) & (bh_ratio > 0), np.log10(bh_ratio), np.nan)

# # --------------------------------------------------------------
# # LOAD HOST VELOCITY DISPERSION (CORRECT + ALIGNED)
# # --------------------------------------------------------------
# sigma_path = "/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN/SOAP-HBT/extra/halo_properties_0127.hdf5"

# # mask of galaxies you actually want to plot
# mask_positive_full = (m30 >= 1e9) & (m30 > 0) & (r50 > 0)

# # row positions in the SOAP catalogue
# row_idx = np.flatnonzero(mask_positive_full)

# # allocate full-length array if you want to keep SOAP alignment
# sigma_full = np.full(m30.shape, np.nan, dtype=np.float32)

# sigma_path = "/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN/SOAP-HBT/extra/halo_properties_0127.hdf5"
# sigma_ds = "/ExclusiveSphere/HalfMassRadiusStars/StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"

# if os.path.exists(sigma_path):
#     with h5py.File(sigma_path, "r") as f:
#         ds = f[sigma_ds]
#         print("sigma dataset shape:", ds.shape)

#         # read only the selected rows
#         rows = np.asarray(ds[row_idx, :], dtype=np.float32)   # shape (N, 9)

#         # diagonal components of the 3x3 tensor
#         sigma_rr   = rows[:, 0]
#         sigma_pphi = rows[:, 4]
#         sigma_zz   = rows[:, 8]

#         # your requested scalar sigma
#         sigma_sel = np.sqrt(sigma_rr**2 + sigma_pphi**2 + sigma_zz**2)

#         # put back into full SOAP-aligned array
#         sigma_full[row_idx] = sigma_sel

#         # log sigma for plotting
#         log_sigma_full = np.full(m30.shape, np.nan, dtype=np.float32)
#         log_sigma_full[row_idx] = np.where(sigma_sel > 0, np.log10(sigma_sel), np.nan)

#     print("Loaded sigma values:", np.isfinite(sigma_sel).sum(), "/", sigma_sel.size)
#     print("N(sigma == 0):", np.count_nonzero(np.isclose(sigma_sel[np.isfinite(sigma_sel)], 0.0)))
# else:
#     print("Sigma file not found.")

# sigma_vals = sigma_full[mask_positive_full]
# log_sigma_vals = log_sigma_full[mask_positive_full]

# ------------------------ LOAD ex-situ summary (optional) ------------------------
exsitu_lookup = {}
if os.path.exists(exsitu_h5):
    try:
        with h5py.File(exsitu_h5, "r") as fh:
            if "stars" in fh:
                data = np.array(fh["stars"])
                if data.ndim == 2 and data.shape[1] >= 4:
                    # choose the ID column that overlaps best with SOAP HaloCatalogueIndex
                    candidate_cols = (0, 1, 2)
                    overlaps = []
                    for c in candidate_cols:
                        ids = data[:, c].astype(np.int64)
                        overlaps.append(np.intersect1d(ids, halo_idx).size)

                    keycol = candidate_cols[int(np.argmax(overlaps))]
                    print(f"Using ex-situ key column {keycol}; overlaps = {overlaps}")

                    ids = data[:, keycol].astype(np.int64)
                    exfrac = data[:, 3].astype(float)
                    exsitu_lookup = dict(zip(ids.tolist(), exfrac.tolist()))
                    print(f"Loaded {len(exsitu_lookup)} ex-situ entries from {exsitu_h5} (dataset 'stars').")
            else:
                for k in fh:
                    try:
                        arr = np.array(fh[k])
                        if arr.ndim == 2 and arr.shape[1] >= 4:
                            candidate_cols = (0, 1, 2)
                            overlaps = []
                            for c in candidate_cols:
                                ids = arr[:, c].astype(np.int64)
                                overlaps.append(np.intersect1d(ids, halo_idx).size)

                            keycol = candidate_cols[int(np.argmax(overlaps))]
                            print(f"Using ex-situ key column {keycol}; overlaps = {overlaps}")

                            ids = arr[:, keycol].astype(np.int64)
                            exfrac = arr[:, 3].astype(float)
                            exsitu_lookup = dict(zip(ids.tolist(), exfrac.tolist()))
                            print(f"Loaded {len(exsitu_lookup)} ex-situ entries from {exsitu_h5} (dataset '{k}').")
                            break
                    except Exception:
                        continue
    except Exception as e:
        print("Warning: failed to read ex-situ HDF5:", e)
else:
    print("Ex-situ summary HDF5 not found; skipping ex-situ matching.")

# ------------------------ MATCH CSV -> SOAP (by HaloCatalogueIndex) ------------------------
# create a DoR lookup dict from the CSV (use first available dor column as default)
dor_lookup = {}
primary_dor_col = dor_cols_found[0]
for _, row in df_ucmg.iterrows():
    try:
        sid = int(row["subhalo_id"])
        v = row.get(primary_dor_col, np.nan)
        if pd.isna(v): continue
        dor_lookup[sid] = float(v)
    except Exception:
        continue
ucmg_ids_set = set(df_ucmg["subhalo_id"].unique())
print(f"Unique UCMG ids in CSV: {len(ucmg_ids_set)} ; Loaded DoR entries: {len(dor_lookup)}")

# reindex DoR onto SOAP-selected halo_idx
dor_series = pd.Series(dor_lookup, dtype=float)
halo_idx_int = halo_idx.astype(np.int64)
dor_for_each_soap_row = dor_series.reindex(halo_idx_int).to_numpy(dtype=float)
matched_positions = np.where(np.isfinite(dor_for_each_soap_row))[0]
matched_subids = halo_idx_int[matched_positions]
matched_dor = dor_for_each_soap_row[matched_positions].astype(float)
print(f"Matched UCMG CSV -> SOAP: {len(matched_positions)} matched positions")

#------------------------------------------------------------
# quick diagnostics
full_dor = dor_for_each_soap_row        # DoR for every SOAP row (NaN where no DoR)
soap_logM = logM                        # SOAP logM aligned to same indices as full_dor
# extremes in full SOAP (any DoR present there)
sel_full_ext = np.isfinite(full_dor) & (full_dor > EXTREME_DOR)
print("Extremes in full SOAP:", int(sel_full_ext.sum()))
# extremes among matched UCMGs (the arrays you plot)
sel_matched_ext = (matched_dor > EXTREME_DOR)
print("Extremes in matched UCMGs:", int(sel_matched_ext.sum()))
# extremes present in full SOAP but NOT in matched subset
missing_from_matched = sel_full_ext.sum() - sel_matched_ext.sum()
print("Extremes in SOAP but not in matched subset:", missing_from_matched)
# optionally list a few subhalo ids for inspection
if missing_from_matched > 0:
    full_halo_idx = halo_idx.astype(int)
    missing_ids = full_halo_idx[np.where(sel_full_ext & ~np.isfinite(dor_for_each_soap_row.reindex(full_halo_idx).to_numpy())==False)[0][:20]]
    print("example SOAP halo idx of extremes (full but missing matched):", missing_ids[:10])
#------------------------------------------------------------

# aligned arrays for matched UCMGs
m_logM = logM[matched_positions]; m_logR = logR[matched_positions]
m_compactness = compactness[matched_positions]; m_mgfe = mgfe[matched_positions]
m_age = age[matched_positions]; m_log_ssfr = log_ssfr[matched_positions]
m_bh_ratio = log_bh_ratio[matched_positions] 
m_Zstar = Zstar[matched_positions] 
m_logZstar_rel = logZstar_rel[matched_positions] 

# # matched ex-situ with track_id
# matched_exsitu = np.full_like(matched_dor, np.nan, dtype=float)
# for i,pos in enumerate(matched_positions):
#     try:
#         tid = int(track[pos])
#     except Exception:
#         tid = None
#     matched_exsitu[i] = exsitu_lookup.get(tid, np.nan)

# matched ex-situ with HaloCatalogueIndex
exsitu_series = pd.Series(exsitu_lookup, dtype=float)
exsitu_for_each_soap_row = exsitu_series.reindex(halo_idx.astype(np.int64)).to_numpy(dtype=float)

matched_exsitu = exsitu_for_each_soap_row[matched_positions]
exsitu_fracs_matched = matched_exsitu.copy()

# <<< INSERT THESE MINIMAL LINES HERE >>>
# raw BH mass from SOAP aligned to matched subset
bh_mass_matched = bh_mass[matched_positions]
# sigma_matched = sigma_vals[matched_positions]
# log_sigma_matched = log_sigma_vals[matched_positions]

# is_central aligned to matched subset (bool)
is_central_matched = np.asarray(is_central_selected[matched_positions]).astype(bool)

# ex-situ already computed as 'matched_exsitu' — create a convenient alias
exsitu_fracs_matched = matched_exsitu.copy()
# <<< END INSERT >>>

# ----------------- collect matched DoR variants and time columns -----------------
# helper to get column aligned to matched_subids
def col_aligned(colname):
    if colname not in df_ucmg_indexed.columns:
        return np.full(len(matched_subids), np.nan, dtype=float)
    s = df_ucmg_indexed.reindex(matched_subids)[colname]
    return s.to_numpy(dtype=float)

possible_dor_cols = ["DoR_t90", "DoR_t95", "DoR_t998", "DoR_tfin", "DoR_tfin_existing", "dor", "DoR", "DoR_choice", "DoR_csv"]
dor_variants_matched = {}
for col in possible_dor_cols:
    if col in df_ucmg_indexed.columns:
        key = "DoR_tfin" if col == "DoR_tfin_existing" else col
        dor_variants_matched[key] = col_aligned(col)
# ensure at least one entry exists
if len(dor_variants_matched) == 0:
    dor_variants_matched["DoR_csv_dor"] = matched_dor.copy()
print("Found DoR variants (matched):", list(dor_variants_matched.keys()))

time_cols_want = ["t_start", "t50", "t50_span", "t75", "t75_span", "t90", "t90_span", "t95", "t95_span", "t998", "t998_span", "tfin", "tfin_span"]
time_matched = {}
for col in time_cols_want:
    if col in df_ucmg_indexed.columns:
        time_matched[col] = col_aligned(col)
print("Found time columns (matched):", list(time_matched.keys()))

# ------------------------ Produce both scatter and LOESS plots ------------------------
# 1) global mass-size colored by the default matched_dor (for reference)
fig = plt.figure(figsize=(8, 6))

# prepare colour map: use percentile range like in the metallicity plot
cmap = plt.get_cmap("viridis")

finite_mask = np.isfinite(matched_dor)
if finite_mask.sum() > 0:
    vmin = float(np.nanpercentile(matched_dor[finite_mask], 1))
    vmax = float(np.nanpercentile(matched_dor[finite_mask], 99))
    if vmin == vmax:
        vmin, vmax = 0.0, 1.0
else:
    vmin, vmax = 0.0, 1.0

# coloured matched galaxies
sc = plt.scatter(
    m_logM,
    m_logR,
    c=matched_dor,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    alpha=0.85,
    s=18,
    edgecolors="none"
)

# overlay grey markers for missing DoR values
if finite_mask.sum() < len(matched_dor):
    missing_idx = ~finite_mask
    plt.scatter(
        m_logM[missing_idx],
        m_logR[missing_idx],
        color=(0.6, 0.6, 0.6),
        alpha=0.5,
        s=10,
        label="no DoR data"
    )

# background population
plt.scatter(logM, logR, s=6, color="lightgrey", alpha=0.5)

# compactness threshold line, same style as the metallicity plot
xm = np.linspace(np.nanmin(logM) - 0.1, np.nanmax(logM) + 0.1, 400)
yr = (xm - COMPACTNESS_CUT) / 1.5
plt.plot(
    xm,
    yr,
    linestyle="--",
    color="black",
    label=fr"Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {COMPACTNESS_CUT}$)"
)

plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
plt.ylabel(r"lg(Half Mass Radius / kpc)")
plt.legend(fontsize=8)
plt.grid(True)

cbar = plt.colorbar(sc)
cbar.set_label("DoR")

outpath = os.path.join(outdir, "mass_size_DoR.png")
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.close()

print("Saved DoR-coloured mass-size plot:", outpath)

print("DoR-vs-exsitu block:")
print("finite ex-situ:", np.isfinite(exsitu_fracs_matched).sum(), "/", exsitu_fracs_matched.size)
print("min/max:", np.nanmin(exsitu_fracs_matched), np.nanmax(exsitu_fracs_matched))
# Slow rotator excluding mask
mask_sr = ~(m_logR > 12.4 - m_logM) #(m_logM > 11)
# print("Number of galaxies before", len(m_logR), "and after", len(m_logR[mask_sr]), "removing slow rotators")
# Summary (all matched UCMGs) DoR vs quantities
# loess_coloured_dor_vs_quantity(m_compactness, matched_dor, exsitu_fracs_matched, r"Compactness ($\lg[M_\odot \text{kpc}^{-1.5}]$)", "DoR_vs_compactness.png", cbar_label="Ex-situ mass fraction")
# plot_dor_vs_quantity(m_compactness,  matched_dor, "Compactness",           "DoR_vs_compactness.png", color_arr=exsitu_fracs_matched)
# plot_dor_vs_quantity(m_logZstar_rel,  matched_dor, r"$\lg[Z_\star / Z_\odot]$",           "DoR_vs_metallicity.png", color_arr=exsitu_fracs_matched)
# plot_dor_vs_quantity(m_mgfe,  matched_dor, "[Mg/Fe]",           "DoR_vs_MgFe.png", color_arr=exsitu_fracs_matched)
# plot_dor_vs_quantity(exsitu_fracs_matched,  matched_dor, "Ex-situ mass fraction",           "DoR_vs_exsitu.png", color_arr=m_compactness)
# loess_coloured_dor_vs_quantity(m_mgfe,      matched_dor, exsitu_fracs_matched, "[Mg/Fe]",    "DoR_vs_MgFe.png", cbar_label="Ex-situ mass fraction")
# loess_coloured_dor_vs_quantity(m_age,       matched_dor, exsitu_fracs_matched, "Lum-weighted age [Gyr]",     "DoR_vs_age.png", cbar_label="Ex-situ mass fraction")
# loess_coloured_dor_vs_quantity(m_log_ssfr,       matched_dor, exsitu_fracs_matched, "lg(sSFR / yr⁻¹)",     "DoR_vs_sSFR.png", cbar_label="Ex-situ mass fraction")
# plot_dor_vs_quantity(m_log_ssfr,  matched_dor, "lg(sSFR / yr⁻¹)",           "DoR_vs_sSFR.png")
# loess_coloured_dor_vs_quantity(exsitu_fracs_matched, matched_dor, exsitu_fracs_matched, "Ex-situ mass fraction",           "DoR_vs_exsitu.png", cbar_label=r"$\lg(\text{sSFR}\ /\ \text{yr}^{-1})$")
# loess_coloured_dor_vs_quantity(m_logZstar_rel, matched_dor, exsitu_fracs_matched, r"$\lg[Z_\star / Z_\odot]$",           "DoR_vs_metallicity.png", cbar_label="Ex-situ mass fraction")
# loess_coloured_dor_vs_quantity(log_sigma_matched,      matched_dor, exsitu_fracs_matched, r'$\lg(\sigma / \mathrm{km}\ \mathrm{s}^{-1})$',    "DoR_vs_sigma.png", cbar_label="Ex-situ mass fraction")

# Combined DoR vs quantity 3-figure
# fig = plt.figure(figsize=(18, 5))
# gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.10)
# ax1 = fig.add_subplot(gs[0, 0])
# ax2 = fig.add_subplot(gs[0, 1])
# ax3 = fig.add_subplot(gs[0, 2])
# cax = fig.add_subplot(gs[0, 3])

# sc1 = _plot_dor_vs_quantity_core(
#     ax1,
#     m_logZstar_rel,
#     matched_dor,
#     r"$\lg[Z_\star / Z_\odot]$",
#     color_arr=exsitu_fracs_matched
# )

# sc2 = _plot_dor_vs_quantity_core(
#     ax2,
#     m_mgfe,
#     matched_dor,
#     "[Mg/Fe]",
#     color_arr=exsitu_fracs_matched
# )

# sc3 = _plot_dor_vs_quantity_core(
#     ax3,
#     m_compactness,
#     matched_dor,
#     "Compactness",
#     color_arr=exsitu_fracs_matched
# )

# # keep y-axis only on left panel
# ax1.set_ylabel("DoR")
# ax2.set_ylabel("")
# ax3.set_ylabel("")
# ax2.tick_params(labelleft=False)
# ax3.tick_params(labelleft=False)

# # one shared colorbar on the right
# if sc3 is not None:
#     cbar = fig.colorbar(sc3, cax=cax)
#     cbar.set_label("Ex-situ mass fraction")
#     cbar.solids.set_alpha(1)
# plt.savefig(os.path.join(outdir, "combined_panels.png"), dpi=250, bbox_inches="tight")
# plt.close(fig)

# # 3) scatter + LOESS for time columns
# for col, arr in time_matched.items():
#     scatter_name = f"mass_size_time_{col}_scatter.png"
#     loess_name = f"mass_size_time_{col}_loess.png"
#     try:
#         scatter_coloured_mass_size(m_logM, m_logR, arr, scatter_name, cbar_label=col)
#         print("Saved scatter time plot:", scatter_name)
#     except Exception as e:
#         print("Scatter failed for", col, ":", e)
#     try:
#         loess_coloured_mass_size(m_logM, m_logR, arr, loess_name, cbar_label=col)
#         print("Saved LOESS time plot:", loess_name)
#     except Exception as e:
#         print("LOESS failed for", col, ":", e)

# ------------------------ BINNED STATISTICS (save CSV & plots) ------------------------
bin_outdir = os.path.join(outdir, "by_mass_bin")
os.makedirs(bin_outdir, exist_ok=True)

# mass bins (0.2 dex)
bin_width = 1.0
min_mass = np.nanmin(m_logM) if np.isfinite(np.nanmin(m_logM)) else 9.0
max_mass = np.nanmax(m_logM) if np.isfinite(np.nanmax(m_logM)) else 12.5
bin_start = math.floor(min_mass / bin_width) * bin_width
bin_end = math.ceil(max_mass / bin_width) * bin_width
bins = np.arange(bin_start, bin_end + 1e-9, bin_width)
nbins = len(bins) - 1
print(f"Creating {nbins} mass bins from {bin_start:.2f} to {bin_end:.2f} (width={bin_width} dex)")

# container for bin stats for all quantities
bin_summary_rows = []

# Prepare the quantities and labels we'll compute medians for (DoR + several summary quantities)
quantities_for_bins = {
    "DoR": matched_dor,
    "compactness": m_compactness,
    "MgFe": m_mgfe,
    "lum_age_gyr": m_age,
    "log_ssfr": m_log_ssfr,
    "exsitu_frac": matched_exsitu,
    "BH_log10_ratio": m_bh_ratio
}

for ib in range(nbins):
    lo = bins[ib]; hi = bins[ib+1]
    sel = (m_logM >= lo) & (m_logM < hi) & np.isfinite(matched_dor)
    count = int(np.sum(sel))
    if count == 0:
        print(f"Bin {ib:02d} [{lo:.2f},{hi:.2f}): empty -> skipping")
        continue

    # store per-quantity summary for this mass bin
    row = {"bin_lo": lo, "bin_hi": hi, "bin_center": 0.5*(lo+hi), "count": count}
    for qname, qarr in quantities_for_bins.items():
        vals = qarr[sel]
        row[f"{qname}_median"] = float(np.nanmedian(vals))
        row[f"{qname}_p16"] = float(np.nanpercentile(vals, 16))
        row[f"{qname}_p84"] = float(np.nanpercentile(vals, 84))
    # extreme relics count
    dor_sel = matched_dor[sel]
    central_sel = is_central_selected[matched_positions][sel].astype(bool)

    row["n_extreme"] = int(np.sum(dor_sel > EXTREME_DOR))
    row["n_extreme_central"] = int(np.sum((dor_sel > EXTREME_DOR) & central_sel))
    row["n_extreme_satellite"] = int(np.sum((dor_sel > EXTREME_DOR) & (~central_sel)))
    row["frac_extreme"] = float(row["n_extreme"] / count)
    bin_summary_rows.append(row)

    # --- create once, before bin loop, arrays aligned to the matched subset ----
    # matched_positions is indices into full SOAP arrays (halo rows) for the matched UCMGs
    is_central_matched = is_central[matched_positions]   # now aligned with matched_dor, matched_subids, etc.
    # (optional) coerce to boolean 0/1 -> bool
    is_central_matched = np.asarray(is_central_matched).astype(bool)

    # Now, inside each bin, when you build `sel` over the matched arrays:
    # sel is a boolean array selecting entries within matched arrays (same length as matched_dor)
    dor_sel = matched_dor[sel]                # matched_dor aligned with matched_positions
    central_sel = is_central_matched[sel]     # safe: same alignment

    row["n_extreme"] = int(np.sum(dor_sel > EXTREME_DOR))
    row["n_extreme_central"] = int(np.sum((dor_sel > EXTREME_DOR) & central_sel))
    row["n_extreme_satellite"] = int(np.sum((dor_sel > EXTREME_DOR) & (~central_sel)))
    row["frac_extreme"] = float(row["n_extreme"] / count)

    #-------- DIAGNOSTICS for satellite counts - comment out if no longer needed ---------------
    print("shapes: is_central (full)  :", getattr(is_central,'shape',None))
    print("shapes: halo_index (full)   :", getattr(halo_index,'shape',None))
    print("shapes: halo_idx (selected) :", getattr(halo_idx,'shape',None))
    print("shapes: matched_positions   :", matched_positions.shape if 'matched_positions' in globals() else None)
    print("shapes: matched_dor         :", matched_dor.shape if 'matched_dor' in globals() else None)
    print("sample matched_positions[:10]:", matched_positions[:10])
    print("sample matched_subids[:10]:", matched_subids[:10] if 'matched_subids' in globals() else None)
    # quick consistency:
    is_cen_matched = is_central_selected[matched_positions]    # <-- this is the thing we want to compare with
    print("is_cen_matched dtype/len:", type(is_cen_matched), len(is_cen_matched))
    print("n_matched extremes (DoR>0.7):", np.sum(matched_dor>EXTREME_DOR))
    print("n_matched extremes & is_cen True:", np.sum((matched_dor>EXTREME_DOR) & np.asarray(is_cen_matched).astype(bool)))

    # # also produce bin-specific mass-size plot coloured by DoR (scatter)
    suf = f"mass_{lo:.2f}_{hi:.2f}".replace(".", "p").replace("-", "m")
    # fig, ax = plt.subplots(figsize=(7,6))
    # ax.scatter(logM, logR, s=6, color="lightgrey", alpha=0.5, label=f"simulated galaxies at $z=0$")
    # ax.scatter(m_logM, m_logR, s=6, color="lightgrey", alpha=0.6)
    # sc = ax.scatter(m_logM[sel], m_logR[sel], c=matched_dor[sel], cmap="viridis", s=24, edgecolors="none")
    # cbar = fig.colorbar(sc, ax=ax); cbar.set_label("DoR")
    # xm = np.linspace(np.nanmin(logM)-0.1, np.nanmax(logM)+0.1, 400)
    # yr = (xm - COMPACTNESS_CUT) / 1.5
    # ax.plot(xm, yr, "--", color="black", lw=2, label=f"compactness = {COMPACTNESS_CUT}")
    # ax.set_xlabel("lg(Total Stellar Mass / M⊙)"); ax.set_ylabel("lg(Half Mass Radius / kpc)")
    # ax.set_title(f"Mass-size (DoR) — mass bin [{lo:.2f},{hi:.2f})"); ax.legend(fontsize=8); ax.grid(True)
    # fname = os.path.join(bin_outdir, f"mass_size_DoR_bin_{suf}.png")
    # fig.savefig(fname, dpi=200, bbox_inches="tight"); plt.close(fig)
    # print("  Saved:", fname)

    # Restrict x-range
    compact_xlim = (8.0, np.nanmax(m_compactness))
    exsitu_xlim = (np.nanmin(matched_exsitu), np.nanmax(matched_exsitu))

    # For each quantity, make a small DoR vs quantity plot restricted to this bin
    plot_dor_vs_quantity(m_compactness[sel], matched_dor[sel],
                          f"Compactness [{lo:.2f},{hi:.2f})",
                          f"DoR_vs_compactness_bin_{suf}.png", color_arr=m_logZstar_rel[sel], xlim=compact_xlim)
    # plot_dor_vs_quantity(m_mgfe[sel], matched_dor[sel],
    #                      f"[Mg/Fe]  [{lo:.2f},{hi:.2f})",
    #                      f"DoR_vs_MgFe_bin_{suf}.png")
    # plot_dor_vs_quantity(m_age[sel], matched_dor[sel],
    #                      f"Lum-weighted age [Gyr]  [{lo:.2f},{hi:.2f})",
    #                      f"DoR_vs_age_bin_{suf}.png")
    # plot_dor_vs_quantity(m_log_ssfr[sel], matched_dor[sel],
    #                      f"lg(sSFR / yr⁻¹)  [{lo:.2f},{hi:.2f})",
    #                      f"DoR_vs_sSFR_bin_{suf}.png")
    plot_dor_vs_quantity(matched_exsitu[sel], matched_dor[sel],
                          f"Ex-situ mass fraction  [{lo:.2f},{hi:.2f})",
                          f"DoR_vs_exsitu_bin_{suf}.png", color_arr=m_logZstar_rel[sel], xlim=exsitu_xlim)

# Save bin summary table to CSV so you can plot medians only later
bin_summary_df = pd.DataFrame(bin_summary_rows)
bin_summary_csv = os.path.join(bin_outdir, "DoR_mass_bin_summary.csv")
bin_summary_df.to_csv(bin_summary_csv, index=False)
print("Saved mass-bin summary CSV:", bin_summary_csv)

# # Also produce the aggregated median-vs-mass summary plot (medians + 16/84) for DoR
# if len(bin_summary_rows) > 0:
#     bdf = bin_summary_df
#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.errorbar(bdf["bin_center"], bdf["DoR_median"], yerr=[bdf["DoR_median"] - bdf["DoR_p16"], bdf["DoR_p84"] - bdf["DoR_median"]],
#                 fmt='o-', capsize=3, lw=1.5, label='median DoR (16/84)')
#     ax.axhline(EXTREME_DOR, color='C1', linestyle='--', lw=1.5, label=f"extreme threshold DoR={EXTREME_DOR}")
#     ax.set_xlabel("lg(Stellar Mass / M⊙)"); ax.set_ylabel("Median DoR"); ax.set_ylim(-0.05, 1.05); ax.grid(True)
#     for x,y,cnt in zip(bdf["bin_center"], bdf["DoR_median"], bdf["count"]):
#         ax.text(x, y + 0.04, f"{int(cnt)}", ha='center', fontsize=8, alpha=0.7)
#     save_fig(fig, os.path.join("by_mass_bin", "DoR_median_vs_mass_bin_with_extremes.png"))
#     # For convenience, also save medians-only CSV at top level
#     top_bin_csv = os.path.join(outdir, "DoR_median_vs_mass_bin_with_extremes.csv")
#     bdf.to_csv(top_bin_csv, index=False)
#     print("Saved aggregated median plot and CSV:", top_bin_csv)

# print("Done. Plots and diagnostics in:", outdir)

# bin_summary_df already saved to disk; use it to make aggregated median-vs-mass summary plots
if (bin_summary_df is not None) and (len(bin_summary_df) > 0):
    bdf = bin_summary_df.copy()
    # ensure numeric and sorted by bin_center
    bdf = bdf.sort_values("bin_center").reset_index(drop=True)

    # arrays for plotting (safe access)
    x = bdf["bin_center"].to_numpy()
    y = bdf["DoR_median"].to_numpy()
    y_lo = (bdf["DoR_median"] - bdf["DoR_p16"]).to_numpy()
    y_hi = (bdf["DoR_p84"] - bdf["DoR_median"]).to_numpy()
    counts = bdf["count"].to_numpy()
    n_extreme = bdf.get("n_extreme", np.zeros_like(x)).to_numpy()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.errorbar(x, y, yerr=[y_lo, y_hi], fmt='o-', capsize=3, lw=1.5, label='median DoR (16/84)')
    # ax.axhline(EXTREME_DOR, color='C1', linestyle='--', lw=1.5, label=f"extreme threshold DoR={EXTREME_DOR}")
    ax.set_xlabel("lg(Stellar Mass / M⊙)")
    ax.set_ylabel("Median DoR")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    # annotate counts above points
    for xi, yi, cnt in zip(x, y, counts):
        ax.text(xi, yi + 0.04, f"{int(cnt)}", ha='center', fontsize=8, alpha=0.7)

        # Right-axis: number of extreme relics (for several thresholds)
    ax2 = ax.twinx()
    thr_list = [0.6, 0.65, 0.7]          # thresholds you requested
    colors = ["C1", "C2", "C3"]

    # Compute counts per mass-bin for each threshold from matched arrays (m_logM, matched_dor)
    counts_by_thr = {}
    for thr in thr_list:
        arr = []
        for lo, hi in zip(bdf["bin_lo"], bdf["bin_hi"]):
            sel_thr = (m_logM >= lo) & (m_logM < hi) & (matched_dor > thr)
            arr.append(int(np.sum(sel_thr)))
        counts_by_thr[thr] = np.array(arr, dtype=int)

    # Plot each threshold on the right axis
    # maxcnt_all = 0
    # for thr, col in zip(thr_list, colors):
    #     arr = counts_by_thr[thr]
    #     ax2.plot(x, arr, marker='s', linestyle='--', ms=6, color=col, label=f"DoR > {thr}")
    #     maxcnt_all = max(maxcnt_all, int(np.nanmax(arr)) if arr.size > 0 else 0)
    #     maxcnt = maxcnt_all if maxcnt_all > 0 else 1
    #     for xi, cnt in zip(x, arr):
    #         ax2.text(xi, cnt + max(1, 0.03 * maxcnt), f"{int(cnt)}", ha='center', va='bottom', fontsize=7, color=col, alpha=0.8)
    maxcnt_all = 0
for thr, col in zip(thr_list, colors):
    arr = counts_by_thr[thr]
    ax2.plot(x, arr, marker='s', linestyle='--', ms=6, color=col, label=f"DoR > {thr}")
    maxcnt_all = max(maxcnt_all, int(np.nanmax(arr)) if arr.size > 0 else 0)

# final max for offset calculations
maxcnt = maxcnt_all if maxcnt_all > 0 else 1

# Annotate counts for each threshold, staggered vertically (offset points)
for idx, (thr, col) in enumerate(zip(thr_list, colors)):
    arr = counts_by_thr[thr]
    # base offset in points; increase per-threshold to avoid overlap
    base_offset = 6  # pixels/points above marker
    stagger = 6 * idx  # extra offset per threshold line
    for xi, cnt in zip(x, arr):
        # use annotate with offset in points so placement is stable visually
        ax2.annotate(
            f"{int(cnt)}",
            xy=(xi, cnt),
            xytext=(0, base_offset + stagger),           # offset in points: (x_offset, y_offset)
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=7,
            color=col,
            alpha=0.9,
            zorder=50,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
            clip_on=False
        )
        
    # Set right-axis limits using the global maximum over all thresholds
    ax2.set_ylabel('Number of SAGs')
    ax2.set_ylim(0, max(3, int(maxcnt_all) * 1.15 if maxcnt_all > 0 else 3))

    # # Annotate counts for the primary threshold (0.6) above points to keep plot readable
    # # (If you want annotations for all lines, you can loop similarly for each threshold.)
    # primary_arr = counts_by_thr[thr_list[0]]
    # maxcnt = maxcnt_all if maxcnt_all > 0 else 1
    # for xi, cnt in zip(x, primary_arr):
    #     ax2.text(xi, cnt + max(1, 0.03 * maxcnt), f"{int(cnt)}", ha='center', va='bottom', fontsize=7, color='C1', alpha=0.8)

    # # annotate central / satellite counts (from bdf) just BELOW the primary extreme-count markers
    # n_cen = bdf.get("n_extreme_central", np.zeros_like(x)).to_numpy()
    # n_sat = bdf.get("n_extreme_satellite", np.zeros_like(x)).to_numpy()
    # for xi, cnt, cen, sat in zip(x, primary_arr, n_cen, n_sat):
    #     if cnt <= 0:
    #         continue
    #     ax2.text(
    #         xi,
    #         cnt - max(0.8, 0.06 * maxcnt),
    #         f"C:{int(cen)} / S:{int(sat)}",
    #         ha='center',
    #         va='top',
    #         fontsize=7,
    #         color='black',
    #         alpha=0.85
    #     )

    # combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)

    out_sum = os.path.join(bin_outdir, "DoR_median_vs_mass_bin_with_extremes.png")
    fig.savefig(out_sum, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved summary median DoR vs mass bin (with extreme counts):", out_sum)

    # also write a top-level CSV copy for convenience (medians + counts)
    top_bin_csv = os.path.join(outdir, "DoR_median_vs_mass_bin_with_extremes.csv")
    bdf.to_csv(top_bin_csv, index=False)
    print("Saved aggregated medians CSV:", top_bin_csv)
else:
    print("No mass-bin statistics produced (no matched UCMGs?).")


print("Total extremes:", np.sum(matched_dor > EXTREME_DOR))
print("Total central extremes:", np.sum((matched_dor > EXTREME_DOR) & is_central_matched))
print("Total satellite extremes:", np.sum((matched_dor > EXTREME_DOR) & (~is_central_matched)))

# ==============================================================
# SPLIT: COMPACT vs NON-COMPACT SAGs
# ==============================================================

# fig, ax = plt.subplots(figsize=(8,5))

# # left axis: median DoR curve
# ax.errorbar(
#     x, y,
#     yerr=[y_lo, y_hi],
#     fmt='o-',
#     capsize=3,
#     lw=1.5,
#     label='median DoR (16/84)'
# )

# ax.set_xlabel("lg(Stellar Mass / M⊙)")
# ax.set_ylabel("Median DoR")
# ax.set_ylim(-0.05, 1.05)
# ax.grid(True)

# # annotate galaxy counts
# for xi, yi, cnt in zip(x, y, counts):
#     ax.text(
#         xi,
#         yi + 0.04,
#         f"{int(cnt)}",
#         ha='center',
#         fontsize=8,
#         alpha=0.7
#     )

# # --------------------------------------------------------------
# # right axis
# # --------------------------------------------------------------
# ax2 = ax.twinx()

# compact_counts = []
# diffuse_counts = []

# for lo, hi in zip(bdf["bin_lo"], bdf["bin_hi"]):

#     sel_bin = (m_logM >= lo) & (m_logM < hi)
#     sel_ext = sel_bin & (matched_dor > EXTREME_DOR)

#     sel_compact = sel_ext & (m_compactness >= COMPACTNESS_CUT)
#     sel_diffuse = sel_ext & (m_compactness < COMPACTNESS_CUT)

#     compact_counts.append(np.sum(sel_compact))
#     diffuse_counts.append(np.sum(sel_diffuse))

# compact_counts = np.array(compact_counts)
# diffuse_counts = np.array(diffuse_counts)

# # plot lines
# ax2.plot(
#     x,
#     compact_counts,
#     marker='s',
#     linestyle='--',
#     ms=6,
#     color='C1',
#     label=fr'compact ($\Sigma_{{1.5}} \geq {COMPACTNESS_CUT}$)'
# )

# ax2.plot(
#     x,
#     diffuse_counts,
#     marker='o',
#     linestyle=':',
#     ms=6,
#     color='C4',
#     label=fr'non-compact ($\Sigma_{{1.5}} < {COMPACTNESS_CUT}$)'
# )

# # --------------------------------------------------------------
# # annotations
# # --------------------------------------------------------------

# for idx, (arr, col) in enumerate([
#     (compact_counts, 'C1'),
#     (diffuse_counts, 'C4')
# ]):

#     base_offset = 6
#     stagger = 6 * idx

#     for xi, cnt in zip(x, arr):

#         ax2.annotate(
#             f"{int(cnt)}",
#             xy=(xi, cnt),
#             xytext=(0, base_offset + stagger),
#             textcoords='offset points',
#             ha='center',
#             va='bottom',
#             fontsize=7,
#             color=col,
#             alpha=0.9,
#             zorder=50,
#             bbox=dict(
#                 facecolor='white',
#                 alpha=0.7,
#                 edgecolor='none',
#                 pad=1
#             ),
#             clip_on=False
#         )

# # axis limits
# maxcnt_all = max(
#     np.nanmax(compact_counts),
#     np.nanmax(diffuse_counts)
# )

# ax2.set_ylabel('Number of SAGs')
# ax2.set_ylim(
#     0,
#     max(3, int(maxcnt_all) * 1.15 if maxcnt_all > 0 else 3)
# )

# # combined legend
# h1, l1 = ax.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()

# ax.legend(
#     h1 + h2,
#     l1 + l2,
#     loc='upper right',
#     fontsize=9
# )

# fig.tight_layout()

# out_compact = os.path.join(
#     bin_outdir,
#     "DoR_median_vs_mass_split_compactness.png"
# )

# fig.savefig(out_compact, dpi=200, bbox_inches="tight")
# plt.close(fig)

# print("Saved compact vs non-compact split plot:", out_compact)

# ==============================================================
# SPLIT: CENTRAL vs SATELLITE (FIXED)
# ==============================================================

# fig, ax = plt.subplots(figsize=(8,5))
# ax.errorbar(x, y, yerr=[y_lo, y_hi], fmt='o-', capsize=3, lw=1.5,
#             label='median DoR (16/84)')
# ax.set_xlabel("lg(Stellar Mass / M⊙)")
# ax.set_ylabel("Median DoR")
# ax.set_ylim(-0.05, 1.05)
# ax.grid(True)

# ax2 = ax.twinx()

# # --- CRITICAL: define properly aligned central array ---
# is_central_match = np.asarray(is_central_selected[matched_positions], dtype=bool)
# relic_mask = (matched_dor > EXTREME_DOR) & (m_compactness >= COMPACTNESS_CUT)

# counts_central = []
# counts_sat = []

# for lo, hi in zip(bdf["bin_lo"], bdf["bin_hi"]):
#     in_bin = (m_logM >= lo) & (m_logM < hi)
#     sel_ext = in_bin & relic_mask

#     counts_central.append(np.count_nonzero(sel_ext & is_central_match))
#     counts_sat.append(np.count_nonzero(sel_ext & (~is_central_match)))

# counts_central = np.array(counts_central)
# counts_sat = np.array(counts_sat)

# # --- sanity check (VERY IMPORTANT) ---
# print("CHECK:")
# print("central sum:", counts_central.sum())
# print("satellite sum:", counts_sat.sum())
# print("total:", counts_central.sum() + counts_sat.sum())

# # --- plotting ---
# ax2.plot(x, counts_central, 's--', color='C1', label='Central')
# ax2.plot(x, counts_sat, 'o--', color='C2', label='Satellite')

# ax2.set_ylabel("Number of relics")
# ax2.set_ylim(0, max(3, int(max(np.nanmax(counts_central),
#                               np.nanmax(counts_sat))) * 1.2))

# # combined legend
# h1, l1 = ax.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()
# ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)

# fig.tight_layout()
# fig.savefig(os.path.join(bin_outdir,
#             "DoR_median_vs_mass_split_central_satellite.png"), dpi=200)
# plt.close(fig)

# print("Saved central vs satellite split plot.")

# ==============================================================
# SPLIT: BH = 0 vs BH > 0
# ==============================================================

# fig, ax = plt.subplots(figsize=(8,5))
# ax.errorbar(x, y, yerr=[y_lo, y_hi], fmt='o-', capsize=3, lw=1.5, label='median DoR (16/84)')
# ax.set_xlabel("lg(Stellar Mass / M⊙)")
# ax.set_ylabel("Median DoR")
# ax.set_ylim(-0.05, 1.05)
# ax.grid(True)

# ax2 = ax.twinx()

# counts_bh0 = []
# counts_bhpos = []

# for lo, hi in zip(bdf["bin_lo"], bdf["bin_hi"]):
#     sel_bin = (m_logM >= lo) & (m_logM < hi)
#     sel_ext = sel_bin & (matched_dor > EXTREME_DOR) & (m_compactness >= COMPACTNESS_CUT)

#     counts_bh0.append(np.sum(sel_ext & (bh_mass_matched <= 0)))
#     counts_bhpos.append(np.sum(sel_ext & (bh_mass_matched > 0)))

# counts_bh0 = np.array(counts_bh0)
# counts_bhpos = np.array(counts_bhpos)

# ax2.plot(x, counts_bh0, 's--', color='C3', label='BH = 0')
# ax2.plot(x, counts_bhpos, 'o--', color='C4', label='BH > 0')

# ax2.set_ylabel("Number of relics")
# ax2.set_ylim(0, max(3, int(max(np.nanmax(counts_bh0), np.nanmax(counts_bhpos))) * 1.2))

# h1, l1 = ax.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()
# ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)

# fig.tight_layout()
# fig.savefig(os.path.join(bin_outdir, "DoR_median_vs_mass_split_BH.png"), dpi=200)
# plt.close(fig)

# print("Saved BH split plot.")

# # ==============================================================
# # SPLIT: Ex-situ ≤ 0.1 vs > 0.1
# # ==============================================================

# fig, ax = plt.subplots(figsize=(8,5))
# ax.errorbar(x, y, yerr=[y_lo, y_hi], fmt='o-', capsize=3, lw=1.5, label='median DoR (16/84)')
# ax.set_xlabel("lg(Stellar Mass / M⊙)")
# ax.set_ylabel("Median DoR")
# ax.set_ylim(-0.05, 1.05)
# ax.grid(True)

# ax2 = ax.twinx()

# counts_low_ex = []
# counts_high_ex = []

# for lo, hi in zip(bdf["bin_lo"], bdf["bin_hi"]):
#     sel_bin = (m_logM >= lo) & (m_logM < hi)
#     sel_ext = sel_bin & (matched_dor > EXTREME_DOR)

#     counts_low_ex.append(np.sum(sel_ext & (exsitu_fracs_matched <= 0.1)))
#     counts_high_ex.append(np.sum(sel_ext & (exsitu_fracs_matched > 0.1)))

# counts_low_ex = np.array(counts_low_ex)
# counts_high_ex = np.array(counts_high_ex)

# ax2.plot(x, counts_low_ex, 's--', color='C5', label='ex-situ ≤ 0.1')
# ax2.plot(x, counts_high_ex, 'o--', color='C6', label='ex-situ > 0.1')

# ax2.set_ylabel("Number of extreme relics")
# ax2.set_ylim(0, max(3, int(max(np.nanmax(counts_low_ex), np.nanmax(counts_high_ex))) * 1.2))

# h1, l1 = ax.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()
# ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)

# fig.tight_layout()
# fig.savefig(os.path.join(bin_outdir, "DoR_median_vs_mass_split_exsitu.png"), dpi=200)
# plt.close(fig)

# print("Saved ex-situ split plot.")

# ==============================================================
# SPLIT: Ex-situ < 0.1 ; 0.1–0.3 ; > 0.3
# ==============================================================

# fig, ax = plt.subplots(figsize=(8,5))
# ax.errorbar(x, y, yerr=[y_lo, y_hi], fmt='o-', capsize=3, lw=1.5,
#             label='median DoR (16/84)')
# ax.set_xlabel("lg(Stellar Mass / M⊙)")
# ax.set_ylabel("Median DoR")
# ax.set_ylim(-0.05, 1.05)
# ax.grid(True)

# ax2 = ax.twinx()

# counts_low = []
# counts_mid = []
# counts_high = []

# for lo, hi in zip(bdf["bin_lo"], bdf["bin_hi"]):
#     sel_bin = (m_logM >= lo) & (m_logM < hi)
#     sel_ext = sel_bin & (matched_dor > EXTREME_DOR) & (m_compactness >= COMPACTNESS_CUT)

#     counts_low.append(np.sum(sel_ext & (exsitu_fracs_matched < 0.1)))
#     counts_mid.append(np.sum(sel_ext & (exsitu_fracs_matched >= 0.1) & (exsitu_fracs_matched <= 0.3)))
#     counts_high.append(np.sum(sel_ext & (exsitu_fracs_matched > 0.3)))

# counts_low = np.array(counts_low)
# counts_mid = np.array(counts_mid)
# counts_high = np.array(counts_high)

# ax2.plot(x, counts_low,  's--', color='C5', label='ex-situ < 0.1')
# ax2.plot(x, counts_mid,  'o--', color='C6', label='0.1 ≤ ex-situ ≤ 0.3')
# ax2.plot(x, counts_high, '^--', color='C7', label='ex-situ > 0.3')

# ax2.set_ylabel("Number of relics")
# ax2.set_ylim(0, max(3, int(max(np.nanmax(counts_low),
#                                 np.nanmax(counts_mid),
#                                 np.nanmax(counts_high))) * 1.2))

# h1, l1 = ax.get_legend_handles_labels()
# h2, l2 = ax2.get_legend_handles_labels()
# ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)

# fig.tight_layout()
# fig.savefig(os.path.join(bin_outdir, "DoR_median_vs_mass_split_exsitu_3bins.png"), dpi=200)
# plt.close(fig)

# print("Saved ex-situ 3-bin split plot.")

# Plot BH mass ratio (medians) and overlay individual extreme relics
# ---------------------------------------------------------------------

# check how many extremes are dropped due to BH missingness
n_extreme_total = int(np.sum(matched_dor > EXTREME_DOR))
n_extreme_with_bh = int(np.sum((matched_dor > EXTREME_DOR) & np.isfinite(m_bh_ratio)))
n_extreme_no_bh = n_extreme_total - n_extreme_with_bh
print("extreme total (matched):", n_extreme_total)
print("extreme with finite BH ratio (matched):", n_extreme_with_bh)
print("extreme missing BH or BH==0:", n_extreme_no_bh)
# show masses for extremes missing BH
if n_extreme_no_bh > 0:
    idxs = np.where((matched_dor > EXTREME_DOR) & ~np.isfinite(m_bh_ratio))[0]
    # raw BH mass (linear, Msun) from SOAP
    bh_mass_matched = bh_mass[matched_positions]
    print("stellar masses for some extremes missing BH:", m_logM[idxs][:10])
    print("raw black hole masses for some extremes missing BH:", bh_mass_matched[idxs][:20])
    print("BH ratios for some extremes missing BH:", m_bh_ratio[idxs][:20])
#----------------------------------------------------------------

if ("BH_log10_ratio_median" in bdf.columns) or ("BH_ratio_median" in bdf.columns):

    # choose which median columns exist in bin summary
    if "BH_log10_ratio_median" in bdf.columns:
        med_col = "BH_log10_ratio_median"
        p16_col = "BH_log10_ratio_p16"
        p84_col = "BH_log10_ratio_p84"
    else:
        med_col = "BH_ratio_median"
        p16_col = "BH_ratio_p16"
        p84_col = "BH_ratio_p84"

    x = bdf["bin_center"].to_numpy()
    med = bdf[med_col].to_numpy()
    lo = med - bdf[p16_col].to_numpy()
    hi = bdf[p84_col].to_numpy() - med

    fig, ax = plt.subplots(figsize=(8,5))
    ax.errorbar(x, med, yerr=[lo, hi], fmt='o-', capsize=3, lw=1.5, zorder=100, label='median (16/84)')
    ax.set_xlabel(r"$\lg(M_\star / M_\odot)$")
    ax.set_ylabel(r"$\lg(M_{\mathrm{BH}} / M_\star)$")
    ax.grid(True)

    # --- Overlay INDIVIDUAL extreme relic points (use in-memory matched arrays) ---
    # m_logM, m_bh_ratio, matched_dor are aligned arrays for matched UCMGs
    # EXTREME_DOR is defined earlier (e.g. 0.7)

    # diagnostic counts
    n_matched_total = matched_dor.size
    n_extreme_all = int(np.sum(matched_dor > EXTREME_DOR))
    n_extreme_with_bh = int(np.sum((matched_dor > EXTREME_DOR) & np.isfinite(m_bh_ratio)))

    print(f"Matched UCMGs: {n_matched_total}; DoR>{EXTREME_DOR}: {n_extreme_all}; with finite BH ratio: {n_extreme_with_bh}")

    # build selection of extreme relics that have finite BH ratio
    sel_ext = (matched_dor > EXTREME_DOR) & np.isfinite(m_logM) & np.isfinite(m_bh_ratio) #all extremes
    # sel_ext1 = (matched_dor > 0.6) & (matched_dor <= 0.65) & np.isfinite(m_logM) & np.isfinite(m_bh_ratio)
    # sel_ext2 = (matched_dor > 0.65) & (matched_dor <= 0.7) & np.isfinite(m_logM) & np.isfinite(m_bh_ratio)
    # sel_ext3 = (matched_dor > 0.7) & np.isfinite(m_logM) & np.isfinite(m_bh_ratio)
    sel_rel = sel_ext & (m_compactness > COMPACTNESS_CUT)
    sel_sag = sel_ext & (m_compactness < COMPACTNESS_CUT)
    sel_compnonrel = (matched_dor < EXTREME_DOR) & np.isfinite(m_logM) & (m_compactness > COMPACTNESS_CUT)

    # overlay extremes from SOAP (uses halo_idx, logM, log_bh_ratio computed earlier)
    sel_soap_ext = np.isfinite(dor_for_each_soap_row) & (dor_for_each_soap_row > EXTREME_DOR)
    # get BH ratio computed for SOAP rows: log_bh_ratio (you computed earlier for full SOAP)
    sel_soap_ext_and_bh = sel_soap_ext & np.isfinite(log_bh_ratio)
    print("soap extremes:", sel_soap_ext.sum(), "soap extremes with BH:", sel_soap_ext_and_bh.sum())

    if np.sum(sel_ext) == 0:
        print("No matched extreme relics with finite BH ratio found in-memory (nothing to overlay).")
    else:
        # Plot them conspicuously
        ax.scatter(m_logM[sel_compnonrel], m_bh_ratio[sel_compnonrel],
                color='lightgrey', alpha=0.5, s=10, linewidth=0.7,
                zorder=10, label='compact non-relics')
        ax.scatter(m_logM[sel_sag], m_bh_ratio[sel_sag],
                facecolor='C1', edgecolor='C1', s=15, marker='d', linewidth=0.7,
                zorder=10, label='SAGs')
        ax.scatter(m_logM[sel_rel], m_bh_ratio[sel_rel],
                facecolor='C2', edgecolor='C2', s=30, marker='*', linewidth=0.7,
                zorder=20, label='SRGs')
        # # Plot 3 ranges of extreme relics        
        # ax.scatter(m_logM[sel_ext1], m_bh_ratio[sel_ext1],
        #         facecolor='C1', edgecolor='k', s=120, marker='*', linewidth=0.7,
        #         zorder=110, label='DoR = 0.6 - 0.65')
        # ax.scatter(m_logM[sel_ext2], m_bh_ratio[sel_ext2],
        #         facecolor='C2', edgecolor='k', s=120, marker='*', linewidth=0.7,
        #         zorder=110, label='DoR = 0.65 - 0.7')
        # ax.scatter(m_logM[sel_ext3], m_bh_ratio[sel_ext3],
        #         facecolor='None', edgecolor='k', s=120, marker='*', linewidth=0.7,
        #         zorder=110, label='DoR > 0.7')
        # ax.scatter(logM[sel_soap_ext_and_bh], log_bh_ratio[sel_soap_ext_and_bh],
        #     marker='*', s=100, facecolor='C1', edgecolor='k', zorder=110,
        #     label='individual extreme relics')
        # ensure plotted points are inside axes limits (prevent clipping)
        ax.relim(); ax.autoscale_view(True, True, True)
        # small y-padding so markers don't sit on axis border
        ymin, ymax = ax.get_ylim(); pad = 0.06 * (ymax - ymin); ax.set_ylim(ymin - pad, ymax + pad)
        print(f"Plotted {int(np.sum(sel_ext))} individual extreme relics (in-memory).")

    ax.legend(loc='best', fontsize=9)
    outbh = os.path.join(bin_outdir, "BHratio_log10_median_vs_mass_bin.png")
    fig.savefig(outbh, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved BH ratio summary with extreme markers:", outbh)

else:
    print("BH ratio not present in bin summary table; skip BH plot.")

#     # --- Overlay INDIVIDUAL extreme relic points split by central/satellite ---
#     is_central_matched_bool = np.asarray(is_central_matched).astype(bool)

#     # DoR bins and styling
#     extreme_bins = [
#         (0.6, 0.65, "C1"),
#         (0.65, 0.7, "C2"),
#         (0.7, 1.0, "C3"),
#     ]

#     n_extreme_all = int(np.sum(matched_dor > EXTREME_DOR))
#     n_extreme_with_bh = int(np.sum((matched_dor > EXTREME_DOR) & np.isfinite(m_bh_ratio)))
#     print(f"Matched UCMGs: {matched_dor.size}; DoR>{EXTREME_DOR}: {n_extreme_all}; with finite BH ratio: {n_extreme_with_bh}")

#     if n_extreme_with_bh == 0:
#         print("No matched extreme relics with finite BH ratio found in-memory (nothing to overlay).")
#     else:
#         for lo_dor, hi_dor, col in extreme_bins:
#             sel_dor = (matched_dor > lo_dor) & (matched_dor <= hi_dor) & np.isfinite(m_logM) & np.isfinite(m_bh_ratio)

#             sel_cen = sel_dor & is_central_matched_bool
#             sel_sat = sel_dor & (~is_central_matched_bool)

#             if np.any(sel_cen):
#                 ax.scatter(
#                     m_logM[sel_cen], m_bh_ratio[sel_cen],
#                     facecolor=col, edgecolor='k',
#                     s=120, marker='o', linewidth=0.7,
#                     zorder=110,
#                     label=f"central DoR {lo_dor:g} - {hi_dor if np.isfinite(hi_dor) else 'inf'}"
#                 )

#             if np.any(sel_sat):
#                 ax.scatter(
#                     m_logM[sel_sat], m_bh_ratio[sel_sat],
#                     facecolor=col, edgecolor='k',
#                     s=120, marker='^', linewidth=0.7,
#                     zorder=110,
#                     label=f"satellite DoR {lo_dor:g} - {hi_dor if np.isfinite(hi_dor) else 'inf'}"
#                 )

#         ax.relim()
#         ax.autoscale_view(True, True, True)
#         ymin, ymax = ax.get_ylim()
#         pad = 0.06 * (ymax - ymin)
#         ax.set_ylim(ymin - pad, ymax + pad)

#     ax.legend(loc='best', fontsize=9)