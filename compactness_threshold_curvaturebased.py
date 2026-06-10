#!/usr/bin/env python3
"""
compactness2panel_topview_projection_curvature.py

Standalone script that reproduces the same 2-panel compactness plots as the
current pipeline, but uses a curvature-based breakpoint criterion as the
primary threshold finder.

Main differences versus the old version:
- breakpoint is chosen from the strongest curvature peak in the smoothed median
  relation (second derivative of the median trend)
- same plotting flow and outputs
- cmasher colormap used for the density layers
"""
from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmasher as cmr

import common  # your helper that provides read_group_data_colibre

from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree

plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "serif", "font.size": 13})


# ------------------------------------------------------------------
# threshold finder: curvature-based breakpoint
# ------------------------------------------------------------------
def find_compactness_threshold(
    cbin_centers,
    cmed,
    compactness_all,
    quantity_all,
    counts_per_bin=None,
    sg_window=9,
    sg_poly=2,
    gauss_sigma=1.0,
    edge_frac=0.02,
    bootstrap_n=0,
    random_seed=12345,
    cbins_edges=None,
):
    """
    Curvature-based breakpoint finder.

    Method:
      - smooth median relation
      - compute 1st derivative
      - compute 2nd derivative (curvature)
      - pick maximum |curvature| away from edges

    Returns:
      threshold, smoothed median, derivative, curvature, bootstrap stats
    """

    # --- setup ---
    cbin_centers = np.asarray(cbin_centers, dtype=float)
    cmed = np.asarray(cmed, dtype=float)
    compactness_all = np.asarray(compactness_all, dtype=float)
    quantity_all = np.asarray(quantity_all, dtype=float)

    n = len(cbin_centers)
    if n < 3:
        raise ValueError("Need at least 3 bins")

    # --- smoothing ---
    sw = min(sg_window if sg_window % 2 == 1 else sg_window + 1,
             n if n % 2 == 1 else n - 1)
    if sw < 3:
        sw = 3

    finite_mask = np.isfinite(cmed)
    fill_value = float(np.nanmedian(cmed[finite_mask]))
    cmed_fill = np.nan_to_num(cmed, nan=fill_value)

    try:
        cmed_s = savgol_filter(cmed_fill, window_length=sw, polyorder=sg_poly)
    except Exception:
        cmed_s = gaussian_filter1d(cmed_fill, sigma=gauss_sigma)

    # --- derivatives ---
    deriv = np.gradient(cmed_s, cbin_centers)
    deriv_s = gaussian_filter1d(deriv, sigma=gauss_sigma)

    curvature = np.gradient(deriv_s, cbin_centers)
    curvature_s = gaussian_filter1d(curvature, sigma=gauss_sigma)

    # --- curvature-based threshold ---
    abs_curv = np.abs(curvature_s)

    valid = np.isfinite(abs_curv)

    # exclude edges (CRITICAL)
    n_edge = int(edge_frac * n)
    valid[:n_edge] = False
    valid[-n_edge:] = False

    if np.any(valid):
        curv_masked = np.full_like(abs_curv, np.nan)
        curv_masked[valid] = abs_curv[valid]

        idx_thr = int(np.nanargmax(curv_masked))
        threshold_value = float(cbin_centers[idx_thr])
        method = "max_curvature"
    else:
        threshold_value = float(np.nanpercentile(compactness_all, 90))
        method = "fallback_percentile"

    # --- bootstrap (UNCHANGED except curvature-based logic) ---
    rng = np.random.default_rng(seed=random_seed)
    bootstrap_stats = None

    if bootstrap_n > 0:

        if cbins_edges is None:
            edges = np.empty(n + 1)
            edges[1:-1] = 0.5 * (cbin_centers[:-1] + cbin_centers[1:])
            edges[0] = cbin_centers[0] - (edges[1] - cbin_centers[0])
            edges[-1] = cbin_centers[-1] + (cbin_centers[-1] - edges[-2])
            cbins_edges = edges

        idxs = np.searchsorted(cbins_edges, compactness_all) - 1
        valid_mask = (idxs >= 0) & (idxs < n) & np.isfinite(quantity_all)

        if valid_mask.sum() > 0:

            from collections import defaultdict
            bins_values = defaultdict(list)
            for i, q in zip(idxs[valid_mask], quantity_all[valid_mask]):
                bins_values[int(i)].append(float(q))

            thr_boot = []

            for _ in range(bootstrap_n):

                med_bs = np.full(n, np.nan)
                for i in range(n):
                    vals = bins_values.get(i, [])
                    if len(vals) > 0:
                        sample = rng.choice(vals, size=len(vals), replace=True)
                        med_bs[i] = np.nanmedian(sample)

                med_fill = float(np.nanmedian(med_bs[np.isfinite(med_bs)]))
                med_bs = np.nan_to_num(med_bs, nan=med_fill)

                try:
                    meds_s = savgol_filter(med_bs, window_length=sw, polyorder=sg_poly)
                except Exception:
                    meds_s = gaussian_filter1d(med_bs, sigma=gauss_sigma)

                deriv_b = np.gradient(meds_s, cbin_centers)
                deriv_b = gaussian_filter1d(deriv_b, sigma=gauss_sigma)

                curv_b = np.gradient(deriv_b, cbin_centers)
                curv_b = gaussian_filter1d(curv_b, sigma=gauss_sigma)

                abs_curv_b = np.abs(curv_b)

                valid_b = np.isfinite(abs_curv_b)
                valid_b[:n_edge] = False
                valid_b[-n_edge:] = False

                if np.any(valid_b):
                    curv_masked = np.full_like(abs_curv_b, np.nan)
                    curv_masked[valid_b] = abs_curv_b[valid_b]
                    idx = int(np.nanargmax(curv_masked))
                    thr_boot.append(float(cbin_centers[idx]))
                else:
                    thr_boot.append(float(np.nanpercentile(compactness_all, 90)))

            thr_arr = np.array(thr_boot)
            bootstrap_stats = {
                "median": float(np.nanmedian(thr_arr)),
                "p16": float(np.nanpercentile(thr_arr, 16)),
                "p84": float(np.nanpercentile(thr_arr, 84)),
                "raw": thr_arr,
            }

    return {
        "threshold": threshold_value,
        "method": method,
        "cmed_smooth": cmed_s,
        "derivative": deriv_s,
        "curvature": curvature_s,
        "bootstrap": bootstrap_stats,
    }


# --- helper block to test multiple bin sizes (equal-count / quantile bins) ---
def thresholds_for_targets(
    compactness_all,
    quantity_all,
    targets_per_bin=(50, 100, 300, 1000),
    min_count=5,
    bootstrap_n=0,
    random_seed=12345,
    do_bootstrap=True,
):
    res_list = []
    compactness_all = np.asarray(compactness_all, dtype=float)
    quantity_all = np.asarray(quantity_all, dtype=float)

    N = compactness_all.size
    if N == 0:
        return []

    for target in np.atleast_1d(targets_per_bin):
        if target <= 0:
            continue
        nbins = max(3, int(np.floor(N / target)))
        quantiles = np.linspace(0.0, 1.0, nbins + 1)
        edges = np.nanpercentile(compactness_all, 100.0 * quantiles)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9

        cbin_centers = 0.5 * (edges[:-1] + edges[1:])
        counts = np.zeros(nbins, dtype=int)
        med = np.full(nbins, np.nan)
        low = np.full(nbins, np.nan)
        high = np.full(nbins, np.nan)

        idxs = np.searchsorted(edges, compactness_all, side="right") - 1
        valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(quantity_all)
        group = defaultdict(list)
        for ii, q in zip(idxs[valid_mask], quantity_all[valid_mask]):
            group[int(ii)].append(float(q))

        for b in range(nbins):
            vals = group.get(b, [])
            counts[b] = len(vals)
            if counts[b] >= min_count:
                arr = np.asarray(vals, dtype=float)
                med[b] = np.nanmedian(arr)
                low[b] = np.nanpercentile(arr, 16)
                high[b] = np.nanpercentile(arr, 84)

        try:
            thr_res = find_compactness_threshold(
                cbin_centers=cbin_centers,
                cmed=med,
                compactness_all=compactness_all,
                quantity_all=quantity_all,
                counts_per_bin=counts,
                bootstrap_n=(bootstrap_n if do_bootstrap else 0),
                random_seed=random_seed,
                cbins_edges=edges,
            )
        except Exception:
            thr_res = {"threshold": float(np.nanpercentile(compactness_all, 90)), "method": "fallback_error", "bootstrap": None}

        res_list.append(
            {
                "target_per_bin": int(target),
                "nbins": int(nbins),
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
# data / plotting helpers
# ------------------------------------------------------------------

def _safe_minmax(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.nanmin(arr)), float(np.nanmax(arr))


def _make_edges(arrays, nbins=45, pad_frac=0.03):
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        raise RuntimeError("Not enough finite values to build bin edges.")
    lo, hi = np.nanpercentile(vals, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = np.nanmin(vals), np.nanmax(vals)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = pad_frac * (hi - lo + 1e-6)
    return np.linspace(lo - pad, hi + pad, nbins + 1)


def _slugify(text):
    return (
        str(text)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(">", "gt")
        .replace("<", "lt")
        .replace("=", "eq")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


def _fraction_panel(ax, x, y, select_mask, xedges, yedges, title, xlabel, ylabel, min_count=20, add_total_contours=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sel = np.asarray(select_mask, dtype=bool)

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return None

    x_ok = x[ok]
    y_ok = y[ok]
    sel_ok = sel[ok]

    H_tot, xe, ye = np.histogram2d(x_ok, y_ok, bins=[xedges, yedges])
    H_sel, _, _ = np.histogram2d(x_ok[sel_ok], y_ok[sel_ok], bins=[xedges, yedges])

    frac = np.full_like(H_tot, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(H_sel, H_tot, out=frac, where=(H_tot > 0))

    frac[H_tot < min_count] = np.nan
    frac = np.ma.masked_invalid(frac.T)

    cmap = cmr.iceburn
    finite_vals = frac.compressed()
    vmax = float(np.nanpercentile(finite_vals, 98)) if finite_vals.size > 0 else 1.0
    vmax = max(vmax, 0.05)

    im = ax.pcolormesh(
        xe,
        ye,
        frac,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        zorder=1,
    )

    if add_total_contours:
        good = H_tot >= min_count
        if np.any(good):
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            X, Y = np.meshgrid(xc, yc)
            levels = np.unique(np.nanpercentile(H_tot[good], [68, 90, 95]))
            levels = np.asarray(levels, dtype=float)
            levels = levels[np.isfinite(levels)]
            if levels.size >= 2:
                ax.contour(X, Y, H_tot.T, levels=np.sort(levels), colors="k", linewidths=0.8, alpha=0.55, zorder=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return im


def make_fraction_figure_mass_env(select_mask, selection_label, y_by_aperture, outname, y_label, logM_matched, apertures, min_count=20, nbins_x=45, nbins_y=45):
    xedges = _make_edges([logM_matched], nbins=nbins_x, pad_frac=0.03)
    yedges = _make_edges(y_by_aperture, nbins=nbins_y, pad_frac=0.03)

    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    im_for_cbar = None
    for j, ap in enumerate(apertures):
        title = f"{selection_label} | Aperture: {ap/1000:.1f} Mpc"
        im = _fraction_panel(
            axes[j],
            logM_matched,
            y_by_aperture[j],
            select_mask,
            xedges,
            yedges,
            title=title,
            xlabel=r"$\log_{10}(M_\star / M_\odot)$",
            ylabel=y_label,
            min_count=min_count,
            add_total_contours=True,
        )
        if im_for_cbar is None and im is not None:
            im_for_cbar = im
        if j > 0:
            axes[j].tick_params(axis="y", left=False, labelleft=False)
            axes[j].set_ylabel("")

    if im_for_cbar is not None:
        cbar = fig.colorbar(im_for_cbar, cax=cax)
        cbar.set_label(f"Fraction of {selection_label.lower()}")

    axes[0].legend(handles=[plt.Line2D([0], [0], color="k", lw=0.8, label="total-sample contours")], fontsize=8, loc="upper left", frameon=True)
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.10, top=0.92)
    fig.savefig(outname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outname)


def make_fraction_figure_mass_compactness(select_mask, selection_label, outname, logM_matched, compactness_matched, min_count=20, nbins_x=45, nbins_y=45):
    xedges = _make_edges([logM_matched], nbins=nbins_x, pad_frac=0.03)
    yedges = _make_edges([compactness_matched], nbins=nbins_y, pad_frac=0.03)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = _fraction_panel(
        ax,
        logM_matched,
        compactness_matched,
        select_mask,
        xedges,
        yedges,
        title=selection_label,
        xlabel=r"$\log_{10}(M_\star / M_\odot)$",
        ylabel=r"Compactness $\,\log_{10}(M_\star) - 1.5\log_{10}(R_{1/2})$",
        min_count=min_count,
        add_total_contours=True,
    )
    if im is not None:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Fraction of {selection_label.lower()}")
    ax.legend(handles=[plt.Line2D([0], [0], color="k", lw=0.8, label="total-sample contours")], fontsize=8, loc="upper left", frameon=True)
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.11, top=0.92)
    fig.savefig(outname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outname)


def make_two_panel_plot(
    quantity_name,
    y_all,
    y_label,
    compactness,
    logM,
    logR,
    outdir,
    targets,
    search_window=None,
    cmasher_map="iceburn",
    line_color_main="C0",
    line_color_smooth="C2",
    line_color_deriv="C3",
    threshold_label_override=None,
):
    """Generic two-panel plot with the same structure as the current pipeline."""
    y_all = np.asarray(y_all, dtype=float)
    compactness = np.asarray(compactness, dtype=float)
    logM = np.asarray(logM, dtype=float)
    logR = np.asarray(logR, dtype=float)

    if search_window is None:
        comp_use = compactness
        y_use = y_all
        search_lo = None
        search_hi = None
    else:
        search_lo, search_hi = search_window
        raw_search_mask = np.isfinite(compactness) & np.isfinite(y_all) & (compactness >= search_lo)
        if np.isfinite(search_hi):
            raw_search_mask &= (compactness <= search_hi)
        comp_use = compactness[raw_search_mask]
        y_use = y_all[raw_search_mask]

    summary = thresholds_for_targets(comp_use, y_use, targets_per_bin=targets, min_count=5, bootstrap_n=0, do_bootstrap=False)
    for s in summary:
        thr = s["threshold_result"]
        print(f"{quantity_name} target={s['target_per_bin']} -> nbins={s['nbins']} threshold={thr['threshold']:.3f} method={thr['method']}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax_main, ax_proj = axes

    # scatter
    ax_main.scatter(compactness, y_all, s=8, color="lightgrey", alpha=0.8, label="galaxies")

    # KDE contours
    x = compactness
    y = y_all
    fin = np.isfinite(x) & np.isfinite(y)
    if np.sum(fin) >= 10:
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

        tree = cKDTree(np.column_stack((xs, ys)))
        d_grid, _ = tree.query(grid_pts, k=1)
        d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
        if d_data.ndim == 2 and d_data.shape[1] >= 2:
            typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
        else:
            typical_spacing = float(np.median(d_grid))
        cut = max(typical_spacing * 1.3, 1e-6)
        mask_far = d_grid > cut

        Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
        Z_flat = Z.ravel()
        Z_flat[mask_far] = np.nan
        Z = Z_flat.reshape(Xgrid.shape)
        finite_vals = Z[np.isfinite(Z)]
        if finite_vals.size > 0:
            levs = np.percentile(finite_vals, [50, 75, 90, 97])
            cmap = getattr(cmr, cmasher_map, cmr.iceburn)
            cf = ax_main.contourf(Xgrid, Ygrid, Z, levels=50, cmap=cmap, antialiased=True)
            ax_main.contour(Xgrid, Ygrid, Z, levels=levs, colors="k", linewidths=0.6, alpha=0.5)
            fig.colorbar(cf, ax=ax_main, label="Density (KDE)")

    ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
    ax_main.set_ylabel(y_label)
    ax_main.grid(True)

    # compactness bins
    use_quantile_bins = True
    target_per_bin = 1000
    nbins_fixed = 18
    min_count_per_bin = 5

    q_arr = y_use
    valid_comp = np.isfinite(comp_use)
    Ntot = np.sum(valid_comp)
    if Ntot == 0:
        raise RuntimeError(f"No valid galaxies to bin in compactness ({quantity_name}).")

    if use_quantile_bins:
        nbins = max(3, int(np.floor(Ntot / max(1, int(target_per_bin)))))
        quantiles = np.linspace(0.0, 100.0, nbins + 1)
        cbins = np.nanpercentile(comp_use, quantiles)
        for i in range(1, cbins.size):
            if cbins[i] <= cbins[i - 1]:
                cbins[i] = cbins[i - 1] + 1e-9
    else:
        nbins = int(nbins_fixed)
        cbins = np.linspace(np.nanmin(comp_use), np.nanmax(comp_use), nbins + 1)

    cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])
    idxs = np.searchsorted(cbins, comp_use, side="right") - 1
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
        if counts_per_bin[b] >= min_count_per_bin:
            arr = np.asarray(vals, dtype=float)
            cmed[b] = np.nanmedian(arr)
            clow[b] = np.nanpercentile(arr, 16)
            chigh[b] = np.nanpercentile(arr, 84)

    ok = np.isfinite(cmed)
    if np.any(ok):
        ax_main.plot(cbin_centers[ok], cmed[ok], color="black", lw=2, label="median (binned)")
        ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15, label="16–84 pct")
        ax_main.legend(fontsize=8, loc="upper left")

    ax_proj.plot(cbin_centers, cmed, color=line_color_main, lw=2)
    ax_proj.fill_between(cbin_centers, clow, chigh, color=line_color_main, alpha=0.25)
    ax_proj.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
    ax_proj.set_ylabel(y_label)
    ax_proj.grid(True)

    res = find_compactness_threshold(
        cbin_centers=cbin_centers,
        cmed=cmed,
        compactness_all=comp_use,
        quantity_all=y_use,
        counts_per_bin=counts_per_bin,
        bootstrap_n=0,
        cbins_edges=cbins,
    )

    # if metallicity-like search window is requested, keep the threshold inside the chosen window
    if search_window is not None:
        cmed_smooth = res["cmed_smooth"]
        curvature = res["curvature"]
        finite_curv = np.isfinite(curvature)
        if np.any(finite_curv):
            if np.isfinite(search_hi):
                window_mask = np.isfinite(cbin_centers) & (cbin_centers >= search_lo) & (cbin_centers <= search_hi)
            else:
                window_mask = np.isfinite(cbin_centers) & (cbin_centers >= search_lo)
            if np.count_nonzero(window_mask) >= 3:
                idx_local = np.nanargmax(np.abs(np.where(window_mask, curvature, np.nan)))
                if np.isfinite(curvature[idx_local]):
                    res["threshold"] = float(cbin_centers[idx_local])
                    res["method"] = "curvature_peak_window"

    print(f"{quantity_name} threshold:", res["threshold"], res["method"])

    bs = res.get("bootstrap", None)
    if bs is not None:
        print(f"{quantity_name} bootstrap median,16,84:", bs["median"], bs["p16"], bs["p84"])

    ax_proj.plot(cbin_centers, res["cmed_smooth"], linestyle="--", color=line_color_smooth, lw=2, label="smoothed median")
    ax_der = ax_proj.twinx()

    # 1st derivative (slope) — faint
    ax_der.plot(
        cbin_centers,
        res["derivative"],
        color="C3",
        lw=1.2,
        alpha=0.4,
        label="1st derivative"
    )

    # 2nd derivative (curvature) — dominant (THIS defines threshold)
    ax_der.plot(
        cbin_centers,
        res["curvature"],
        color="C4",
        lw=2.2,
        label="2nd derivative (curvature)"
    )

    # zero line for reference
    ax_der.axhline(0.0, linestyle=":", color="k", alpha=0.4)

    # axis label
    ax_der.set_ylabel("Derivatives")
    ax_der.tick_params(axis="y")

    # threshold line (on main axis)
    thr = res["threshold"]
    ax_proj.axvline(
        thr,
        color="C2",
        linestyle="--",
        lw=1.6,
        label=f"threshold = {thr:.2f}"
    )

    # OPTIONAL: mark the actual peak (highly recommended)
    idx_peak = np.nanargmax(np.abs(res["curvature"]))
    ax_der.scatter(
        cbin_centers[idx_peak],
        res["curvature"][idx_peak],
        color="red",
        s=40,
        zorder=10,
        label="chosen breakpoint"
    )

    # combine legends
    lines1, labels1 = ax_proj.get_legend_handles_labels()
    lines2, labels2 = ax_der.get_legend_handles_labels()
    ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    ax_proj.axvline(res["threshold"], color=line_color_smooth, linestyle="--", lw=1.6, label=f"thr={res['threshold']:.2f}")

    handles1, labels1 = ax_proj.get_legend_handles_labels()
    handles2, labels2 = ax_der.get_legend_handles_labels()
    ax_proj.legend(handles1 + handles2, labels1 + labels2, fontsize=8)

    if not use_quantile_bins:
        counts_per_bin = np.array([np.sum((comp_use >= cbins[i]) & (comp_use < cbins[i + 1])) for i in range(nbins)])
        ax2 = ax_proj.twinx()
        ax2.bar(cbin_centers, counts_per_bin, width=(cbins[1] - cbins[0]) * 0.9, alpha=0.12, color="gray", edgecolor="none")
        ax2.set_ylabel("N (per compactness bin)", color="gray")
        ax2.tick_params(axis="y", labelsize=10)

    # save
    if threshold_label_override is None:
        qname = quantity_name
    else:
        qname = threshold_label_override

    fig.tight_layout()
    fig_name = os.path.join(outdir, f"compactness_{_slugify(qname)}_two_panel_curvature.png")
    fig.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved two-panel", quantity_name, "plot to:", fig_name)

    return res

# ------------------------------------------------------------------
#two-regime fitting
# ------------------------------------------------------------------

# threshold finder
def find_compactness_threshold_tworegime(
    cbin_centers,
    cmed,
    compactness_all,
    quantity_all,
    counts_per_bin=None,
    min_bin_count=5,
):
    """
    Two-regime breakpoint:
    - fits two linear relations
    - finds optimal split
    - returns intersection
    """

    cbin_centers = np.asarray(cbin_centers, dtype=float)
    cmed = np.asarray(cmed, dtype=float)

    valid = np.isfinite(cmed)
    if counts_per_bin is not None:
        valid &= (counts_per_bin >= min_bin_count)

    x = cbin_centers[valid]
    y = cmed[valid]

    if len(x) < 6:
        return {"threshold": np.nan, "method": "insufficient_data"}

    best_err = np.inf
    best = None
    best_split = None

    for i in range(2, len(x)-2):

        x1, y1 = x[:i], y[:i]
        x2, y2 = x[i:], y[i:]

        p1 = np.polyfit(x1, y1, 1)
        p2 = np.polyfit(x2, y2, 1)

        err = np.sum((y1 - np.polyval(p1, x1))**2) + \
              np.sum((y2 - np.polyval(p2, x2))**2)

        if err < best_err:
            best_err = err
            best = (p1, p2)
            best_split = i

    if best is None:
        return {"threshold": np.nan, "method": "fit_failed"}

    p1, p2 = best
    a1, b1 = p1
    a2, b2 = p2

    if np.isclose(a1, a2):
        thr = x[len(x)//2]
    else:
        thr = (b2 - b1) / (a1 - a2)

    fit1 = np.polyval(p1, cbin_centers)
    fit2 = np.polyval(p2, cbin_centers)

    return {
        "threshold": float(thr),
        "method": "two_regime_fit",
        "fit1": fit1,
        "fit2": fit2,
    }

# plotting wrapper
def make_two_panel_plot_tworegime(
    quantity_name,
    y_all,
    y_label,
    compactness,
    logM,
    logR,
    outdir,
    targets,
):
    """
    SAME plot as curvature version,
    BUT threshold from two-regime fit.
    """

    # ---- COPY of make_two_panel_plot UNTIL threshold ----
    # (this ensures IDENTICAL visuals)

    y_all = np.asarray(y_all, dtype=float)
    compactness = np.asarray(compactness, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax_main, ax_proj = axes

    # --- scatter ---
    ax_main.scatter(compactness, y_all, s=8, color="lightgrey", alpha=0.8)

    # --- KDE (same as your code) ---
    x = compactness
    y = y_all
    fin = np.isfinite(x) & np.isfinite(y)
    if np.sum(fin) >= 10:
        xs, ys = x[fin], y[fin]
        kde = gaussian_kde(np.vstack([xs, ys]))

        xg = np.linspace(*np.nanpercentile(xs, [1,99]), 200)
        yg = np.linspace(*np.nanpercentile(ys, [1,99]), 200)
        X, Y = np.meshgrid(xg, yg)

        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        cf = ax_main.contourf(X, Y, Z, levels=50, cmap=cmr.iceburn)
        ax_main.contour(X, Y, Z, levels=np.percentile(Z[np.isfinite(Z)], [50,75,90]), colors="k", lw=0.5)
        fig.colorbar(cf, ax=ax_main)

    ax_main.set_xlabel("Compactness")
    ax_main.set_ylabel(y_label)
    ax_main.grid(True)

    # ---- binning (same as your pipeline) ----
    target_per_bin = 1000
    nbins = max(3, int(len(compactness) / target_per_bin))

    edges = np.nanpercentile(compactness, np.linspace(0,100,nbins+1))
    for i in range(1,len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-9

    centers = 0.5*(edges[:-1] + edges[1:])
    idxs = np.searchsorted(edges, compactness, side='right') - 1

    group = defaultdict(list)
    valid = (idxs>=0)&(idxs<nbins)&np.isfinite(y_all)

    for i,v in zip(idxs[valid], y_all[valid]):
        group[int(i)].append(v)

    cmed = np.full(nbins, np.nan)
    clow = np.full(nbins, np.nan)
    chigh = np.full(nbins, np.nan)
    counts = np.zeros(nbins)

    for b in range(nbins):
        vals = group.get(b, [])
        counts[b] = len(vals)
        if len(vals) >= 5:
            arr = np.array(vals)
            cmed[b] = np.nanmedian(arr)
            clow[b] = np.nanpercentile(arr,16)
            chigh[b] = np.nanpercentile(arr,84)

    # ---- plot median ----
    ax_main.plot(centers, cmed, color="black", lw=2)
    ax_main.fill_between(centers, clow, chigh, color="black", alpha=0.15)

    ax_proj.plot(centers, cmed, color="C0", lw=2)
    ax_proj.fill_between(centers, clow, chigh, color="C0", alpha=0.25)

    # ============================
    # 🔥 TWO-REGIME THRESHOLD
    # ============================
    res = find_compactness_threshold_tworegime(
        centers, cmed, compactness, y_all, counts
    )

    thr = res["threshold"]

    if "fit1" in res:
        ax_proj.plot(centers, res["fit1"], "--", color="C4", label="regime 1")
        ax_proj.plot(centers, res["fit2"], "--", color="C5", label="regime 2")

    ax_proj.axvline(thr, color="red", linestyle="--", lw=2,
                    label=f"threshold = {thr:.2f}")

    ax_proj.set_xlabel("Compactness")
    ax_proj.set_ylabel(y_label)
    ax_proj.grid(True)
    ax_proj.legend(fontsize=8)

    fig.tight_layout()

    fname = os.path.join(
        outdir,
        f"compactness_{_slugify(quantity_name)}_two_panel_tworegime.png"
    )

    fig.savefig(fname, dpi=300)
    plt.close(fig)

    print(f"{quantity_name} (two-regime) threshold:", thr)
    print("Saved:", fname)

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    # ------------------ Config / dataset selection ------------------
    model_name = "L0200N3008/THERMAL_AGN/"
    model_dir = "/mnt/su3-pro/colibre/" + model_name

    base_dir = Path("/mnt/su3ctm/kproctor/ForMax")
    matches = sorted(base_dir.glob("*exsitu*summary*.hdf5"))
    if len(matches) == 0:
        raise FileNotFoundError(f"No ex-situ HDF5 file found in {base_dir}")
    elif len(matches) == 1:
        h5path = str(matches[0])
    else:
        h5path = str(max(matches, key=lambda p: p.stat().st_mtime))
    print("Using ex-situ file:", h5path)

    snap_files = ["0127", "0119", "0114", "0102", "0092", "0076", "0064", "0056", "0048", "0040", "0026", "0018"]
    zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    snap_file = snap_files[0]
    ztarget = zstarget[0]
    comov_to_physical_length = 1.0 / (1.0 + ztarget)

    outdir = os.path.join(os.getcwd(), "plots")
    os.makedirs(outdir, exist_ok=True)

    fields_sgn = {"InputHalos": ("HaloCatalogueIndex", "IsCentral", "HBTplus/DescendantTrackId", "HBTplus/TrackId")}
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

    (m30, sfr30, r50, cp, stellarage, stellarage_lum, FeoverH, MgoverH, Zstar_raw) = h5data_groups
    (m30_proj, r50_proj) = h5data_groups_proj
    (sgn, is_central, desc_id, track_id) = h5data_idgroups

    Lu = 3.086e+24 / (3.086e+24)
    Mu = 1.988e+43 / (1.989e+33)
    tu = 3.086e+19 / (3.154e+7)

    m30 = np.asarray(m30).ravel() * Mu
    m30_proj = np.asarray(m30_proj).ravel() * Mu
    sfr30 = np.asarray(sfr30).ravel() * Mu / tu
    r50 = np.asarray(r50).ravel() * Lu * comov_to_physical_length * 1e3
    r50_proj = np.asarray(r50_proj).ravel() * Lu * comov_to_physical_length * 1e3
    stellarage = np.asarray(stellarage).ravel() * tu / 1e9
    stellarage_lum = np.asarray(stellarage_lum).ravel() * tu / 1e9
    cp = np.asarray(cp) * Lu * comov_to_physical_length

    Zsun = 0.0134
    Zstar = np.asarray(Zstar_raw, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logZstar = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar), np.nan)
        logZstar_rel = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar / Zsun), np.nan)

    select = np.where(m30 >= 1e9)
    ngals = len(m30[select])
    if ngals == 0:
        raise SystemExit("No galaxies selected (m30 >= 1e9)")

    m_in = np.asarray(m30[select]).ravel()
    r50_in = np.asarray(r50[select]).ravel()
    sgn_in = np.asarray(sgn[select]).ravel()
    Fe_in = np.asarray(FeoverH[select]).ravel()
    Mg_in = np.asarray(MgoverH[select]).ravel()
    logZstar_rel_in = np.asarray(logZstar_rel[select]).ravel()
    stellarage_lum_in = np.asarray(stellarage_lum[select]).ravel()
    sfr_in = np.asarray(sfr30[select]).ravel()

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

    targets = (100, 300, 500, 1000, 5000, 7000, 10000)

    summary_mgfe = thresholds_for_targets(compactness, mgfe, targets_per_bin=targets, min_count=5, bootstrap_n=0, do_bootstrap=False)
    for s in summary_mgfe:
        thr = s["threshold_result"]
        print(f"Mg/Fe target={s['target_per_bin']} -> nbins={s['nbins']} threshold={thr['threshold']:.3f} method={thr['method']}")

    # ---------- load ex-situ fractions from HDF5 WITHOUT huge dict ----------
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
    print(f"Matched ex-situ fraction for {np.isfinite(exsitu_fracs).sum()} / {len(exsitu_fracs)} selected galaxies")

    # ------------------ Mg/Fe two-panel ------------------
    make_two_panel_plot(
        "Mg/Fe",
        mgfe,
        "[Mg/Fe]",
        compactness,
        log_m,
        log_r,
        outdir,
        targets,
        search_window=None,
        cmasher_map="iceburn",
        line_color_main="C0",
        line_color_smooth="C2",
        line_color_deriv="C3",
    )

    # ------------------ Ex-situ two-panel ------------------
    make_two_panel_plot(
        "Ex-situ mass fraction",
        exsitu_fracs,
        "Ex-situ mass fraction",
        compactness,
        log_m,
        log_r,
        outdir,
        targets,
        search_window=None,
        cmasher_map="iceburn",
        line_color_main="C0",
        line_color_smooth="C2",
        line_color_deriv="C3",
    )

    # ------------------ Age two-panel ------------------
    ages = np.asarray(stellarage_lum_in)[mask_positive]
    make_two_panel_plot(
        "Lum-weighted age",
        ages,
        "Lum-weighted age (Gyr)",
        compactness,
        log_m,
        log_r,
        outdir,
        targets,
        search_window=None,
        cmasher_map="iceburn",
        line_color_main="C0",
        line_color_smooth="C2",
        line_color_deriv="C3",
    )

    # ------------------ sSFR two-panel ------------------
    sfr = np.asarray(sfr_in)[mask_positive]
    m = np.asarray(m_in)[mask_positive]
    with np.errstate(divide="ignore", invalid="ignore"):
        ssfr = np.where(m > 0, sfr / m, np.nan)
    log_ssfr = np.full_like(ssfr, np.nan, dtype=float)
    mask_pos = (ssfr > 0) & np.isfinite(ssfr)
    if np.any(mask_pos):
        log_ssfr[mask_pos] = np.log10(ssfr[mask_pos])

    make_two_panel_plot(
        "sSFR",
        log_ssfr,
        r"lg(sSFR / yr$^{-1}$)",
        compactness,
        log_m,
        log_r,
        outdir,
        targets,
        search_window=None,
        cmasher_map="iceburn",
        line_color_main="C0",
        line_color_smooth="C2",
        line_color_deriv="C3",
    )

    # ------------------ metallicity two-panel ------------------
    metallicity = np.asarray(logZstar_rel_in)[mask_positive]
    compactness_sel = compactness[np.isfinite(compactness) & np.isfinite(metallicity)]
    metallicity_sel = metallicity[np.isfinite(compactness) & np.isfinite(metallicity)]
    print("number of remaining galaxies", len(metallicity_sel))

    make_two_panel_plot(
        "Metallicity",
        metallicity_sel,
        r"$\lg[Z_* / Z_\odot]$",
        compactness_sel,
        np.log10(m_in[mask_positive][np.isfinite(compactness) & np.isfinite(metallicity)]),
        np.log10(r50_in[mask_positive][np.isfinite(compactness) & np.isfinite(metallicity)]),
        outdir,
        targets,
        search_window=None, #(8.5, np.inf),
        cmasher_map="iceburn",
        line_color_main="C0",
        line_color_smooth="C2",
        line_color_deriv="C3",
    )

    # # ---------- 2D fraction maps (kept as in your pipeline) ----------
    # logM_matched = np.log10(m_in[mask_positive])
    # compactness_matched = logM_matched - 1.5 * np.log10(r50_in[mask_positive])

    # # aligned matched-sample environment arrays
    # Nmatch = len(logM_matched)
    # is_central_matched = np.asarray(is_central[select][mask_positive], dtype=bool)

    # exsitu_series = pd.Series(exsitu_fracs, dtype=float)
    # exsitu_aligned = exsitu_series.to_numpy(dtype=float)

    # # fraction maps use these masks
    # fraction_cases = {
    #     "Ancient galaxies": np.isfinite(mgfe) & np.isfinite(mgfe),
    # }
    # fraction_cases = {
    #     "Ancient galaxies": np.isfinite(mgfe) & np.isfinite(mgfe),
    # }
    # fraction_cases = {
    #     "Ancient galaxies": np.isfinite(exsitu_aligned) & (exsitu_aligned >= 0.0),
    #     "Old galaxies (age > 10 Gyr)": np.isfinite(ages) & (ages > 10.0),
    #     "Ex-situ > 0.3": np.isfinite(exsitu_aligned) & (exsitu_aligned > 0.3),
    # }

    # y_density_by_ap = [exsitu_aligned for _ in range(3)]
    # density_ylabel = r"Ex-situ mass fraction"

    # for label, sel_mask in fraction_cases.items():
    #     make_fraction_figure_mass_env(
    #         sel_mask,
    #         label,
    #         y_by_aperture=y_density_by_ap,
    #         outname=os.path.join(outdir, f"fraction_{_slugify(label)}_vs_mass_densitycontrast.png"),
    #         y_label=density_ylabel,
    #         logM_matched=logM_matched,
    #         apertures=np.array([1.0, 2.0, 3.0]),
    #         min_count=20,
    #         nbins_x=45,
    #         nbins_y=45,
    #     )

    # for label, sel_mask in fraction_cases.items():
    #     make_fraction_figure_mass_compactness(
    #         sel_mask,
    #         label,
    #         outname=os.path.join(outdir, f"fraction_{_slugify(label)}_vs_mass_compactness.png"),
    #         logM_matched=logM_matched,
    #         compactness_matched=compactness_matched,
    #         min_count=20,
    #         nbins_x=45,
    #         nbins_y=45,
    #     )

    # print("Finished fraction-map plots.")

    print("\n=== TWO-REGIME COMPARISON ===\n")

    make_two_panel_plot_tworegime("Mg/Fe", mgfe, "[Mg/Fe]", compactness, log_m, log_r, outdir, targets)
    make_two_panel_plot_tworegime("Ex-situ", exsitu_fracs, "Ex-situ mass fraction", compactness, log_m, log_r, outdir, targets)
    make_two_panel_plot_tworegime("Age", ages, "Lum-weighted age", compactness, log_m, log_r, outdir, targets)
    make_two_panel_plot_tworegime("sSFR", log_ssfr, "sSFR", compactness, log_m, log_r, outdir, targets)
    make_two_panel_plot_tworegime("Metallicity", metallicity_sel, "Metallicity", compactness_sel, log_m, log_r, outdir, targets)


if __name__ == "__main__":
    main()
