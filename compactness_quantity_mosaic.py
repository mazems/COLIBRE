#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline, splrep
from scipy.spatial import cKDTree as KDTree
from scipy.stats import gaussian_kde

# =========================
# USER SETTINGS
# =========================

MODEL_NAME = "L0200N3008/THERMAL_AGN/"
MODEL_DIR = Path("/mnt/su3-pro/colibre") / MODEL_NAME
SNAP_FILE = "0127"
ZTARGET = 0.0

SIGMA_FILE = MODEL_DIR / "SOAP-HBT/extra/halo_properties_0127.hdf5"
SIGMA_DATASET = (
    "/ExclusiveSphere/HalfMassRadiusStars/"
    "StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"
)

EXSITU_DIR = Path("/mnt/su3ctm/kproctor/ForMax")
EXSITU_GLOB = "*exsitu*summary*.hdf5"

OUTDIR = Path("plots")
CACHE_FILE = OUTDIR / "compactness_quantity_mosaic_cache.npz"
LAYOUT_OUTPUT = OUTDIR / "compactness_quantity_mosaic_layout"
FINAL_OUTPUT = OUTDIR / "compactness_quantity_mosaic_3x2"

MASS_LIMIT = 1e9
ZSUN = 0.0134
LOG_MGFE_SUN = 0.10

FIGSIZE = (15.5, 15.0)
WIDTH_RATIOS = [1.0, 0.045, 0.20, 1.0, 0.045]
LEFT, RIGHT, BOTTOM, TOP = 0.07, 0.945, 0.065, 0.985
HSPACE, WSPACE = 0.040, 0.025

GLOBAL_FONT = 18
AXIS_LABEL_FONT = 18
TICK_FONT = 15
CBAR_LABEL_FONT = 16
CBAR_TICK_FONT = 14
PANEL_LABEL_FONT = 18
LEGEND_FONT = 10

SHOW_XLABEL_ONLY_BOTTOM = True
SHOW_YLABEL_ONLY_LEFT = True
SHOW_PANEL_LEGENDS = True

PANEL_LABELS = ["a)", "b)", "c)", "d)", "e)", "f)"]
QUANTITY_KEYS = ["age", "metallicity", "mgfe", "ssfr", "sigma", "exsitu"]
Y_LABELS = [
    "Age [Gyr]",
    r"$[Z/H]$",
    r"$[\mathrm{Mg}/\mathrm{Fe}]$",
    r"$\log_{10}(\mathrm{sSFR}\,/\,\mathrm{yr}^{-1})$",
    r"$\log_{10}(\sigma\,/\,\mathrm{km}\,\mathrm{s}^{-1})$",
    r"$f_{\mathrm{ex\! -\! situ}}$",
]
X_LABEL = r"$\log_{10}(\Sigma_{1.5}\,[M_\odot\,\mathrm{kpc}^{-1.5}])$"

TARGET_PER_BIN = {k: 1000 for k in QUANTITY_KEYS}
SPLINE_S_FACTOR = {
    "age": 3.5,
    "metallicity": 0.005,
    "mgfe": 0.0007,
    "ssfr": 0.1,
    "sigma": 0.015,
    "exsitu": 0.02,
}
SEARCH_LIMITS = {
    "age": (None, None),
    "metallicity": (9.5, 10.2),
    "mgfe": (None, None),
    "ssfr": (None, None),
    "sigma": (None, None),
    "exsitu": (None, None),
}

SPLINE_K = 3
MIN_COUNT_PER_BIN = 5
DERIV_THRESH_FACTOR = 1.0
EDGE_FRAC = 0.05

KDE_NX = 200
KDE_NY = 200
KDE_DISTANCE_FACTOR = 1.3
KDE_LEVELS = 50
KDE_CONTOUR_PERCENTILES = [50, 75, 90, 97]

X_LIMITS = (8.8, 10.8)
LAYOUT_Y_LIMITS = [
    (1.0, 10.5),
    (-0.5, 0.2),
    (0.06, 0.20),
    (-12.2, -9.3),
    (1.2, 2.1),
    (0.0, 0.45),
]

# =========================
# SPLINE HELPERS
# =========================

def fit_weighted_bspline(x, y, weights=None, spline_k=3, spline_s_factor=0.05):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    w = None if weights is None else np.asarray(weights, float).ravel()

    finite = np.isfinite(x) & np.isfinite(y)
    if w is not None:
        finite &= np.isfinite(w)

    x, y = x[finite], y[finite]
    if w is not None:
        w = w[finite]

    if x.size < 4:
        raise ValueError("Need at least 4 finite points for spline fitting.")

    order = np.argsort(x)
    x, y = x[order], y[order]
    if w is not None:
        w = w[order]

    k = min(max(1, spline_k), x.size - 1)
    s = spline_s_factor * x.size
    return BSpline(*splrep(x, y, w=w, k=k, s=s))


def find_threshold(centers, medians, compactness_all, counts, key):
    finite = np.isfinite(centers) & np.isfinite(medians)
    x = centers[finite]
    y = medians[finite]
    w = np.sqrt(np.maximum(counts[finite], 1))

    spline = fit_weighted_bspline(
        x,
        y,
        weights=w,
        spline_k=SPLINE_K,
        spline_s_factor=SPLINE_S_FACTOR[key],
    )

    smooth = np.asarray(spline(centers), float)
    deriv = np.asarray(spline.derivative()(centers), float)
    abs_deriv = np.abs(deriv)

    search_lo, search_hi = SEARCH_LIMITS[key]
    search_mask = np.isfinite(centers)
    if search_lo is not None:
        search_mask &= centers >= search_lo
    if search_hi is not None:
        search_mask &= centers <= search_hi

    if np.count_nonzero(search_mask) >= 4:
        stat = abs_deriv[search_mask]
        global_idx = np.flatnonzero(search_mask)
    else:
        stat = abs_deriv
        global_idx = np.arange(centers.size)

    baseline = np.nanmedian(stat)
    mad = np.nanmedian(np.abs(stat - baseline))
    deriv_threshold = baseline + DERIV_THRESH_FACTOR * 1.4826 * mad

    idx_max = global_idx[int(np.nanargmax(stat))]
    left_edge = int(np.floor(EDGE_FRAC * centers.size))
    right_edge = int(np.ceil((1 - EDGE_FRAC) * centers.size)) - 1

    if left_edge < idx_max < right_edge and abs_deriv[idx_max] >= deriv_threshold:
        threshold = float(centers[idx_max])
        method = "turning_point"
    else:
        threshold = None
        method = None
        for idx in global_idx[::-1]:
            if abs_deriv[idx] >= deriv_threshold and counts[idx] >= MIN_COUNT_PER_BIN:
                threshold = float(centers[idx])
                method = "start_exceed"
                break
        if threshold is None:
            threshold = float(np.nanpercentile(compactness_all, 90))
            method = "fallback_percentile"

    return smooth, deriv, threshold, method


# =========================
# DATA LOADING
# =========================

def find_exsitu_file():
    matches = sorted(EXSITU_DIR.glob(EXSITU_GLOB))
    if not matches:
        raise FileNotFoundError(f"No ex-situ file in {EXSITU_DIR}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_full_data():
    import common

    Lu = 1.0
    Mu = 1.988e43 / 1.989e33
    tu = 3.086e19 / 3.154e7
    physical = 1.0 / (1.0 + ZTARGET)

    fields = {
        "ExclusiveSphere/50kpc": (
            "StellarMass",
            "StarFormationRate",
            "HalfMassRadiusStars",
            "LuminosityWeightedMeanStellarAge",
            "LinearMassWeightedIronOverHydrogenOfStars",
            "LinearMassWeightedMagnesiumOverHydrogenOfStars",
            "StellarMassFractionInMetals",
        )
    }
    fields_ids = {"InputHalos": ("HaloCatalogueIndex",)}

    print("Reading SOAP quantities...")
    m30, sfr30, r50, age_lum, fe, mg, zstar = common.read_group_data_colibre(
        str(MODEL_DIR) + os.sep, SNAP_FILE, fields
    )
    (sgn,) = common.read_group_data_colibre(
        str(MODEL_DIR) + os.sep, SNAP_FILE, fields_ids
    )

    m30 = np.asarray(m30, float) * Mu
    sfr30 = np.asarray(sfr30, float) * Mu / tu
    r50 = np.asarray(r50, float) * Lu * physical * 1e3
    age_lum = np.asarray(age_lum, float) * tu / 1e9
    fe = np.asarray(fe, float)
    mg = np.asarray(mg, float)
    zstar = np.asarray(zstar, float)
    sgn = np.asarray(sgn)

    mask = (
        (m30 >= MASS_LIMIT)
        & np.isfinite(m30)
        & np.isfinite(r50)
        & (m30 > 0)
        & (r50 > 0)
    )

    m = m30[mask]
    r = r50[mask]
    compactness = np.log10(m) - 1.5 * np.log10(r)
    age = age_lum[mask]

    with np.errstate(divide="ignore", invalid="ignore"):
        metallicity = np.where(
            zstar[mask] > 0, np.log10(zstar[mask] / ZSUN), np.nan
        )
        mgfe_ratio = mg[mask] / fe[mask]
        mgfe = np.where(
            mgfe_ratio > 0, np.log10(mgfe_ratio) - LOG_MGFE_SUN, np.nan
        )
        ssfr = sfr30[mask] / m
        log_ssfr = np.where(ssfr > 0, np.log10(ssfr), np.nan)

    print("Reading velocity dispersion...")
    rows_idx = np.flatnonzero(mask)
    with h5py.File(SIGMA_FILE, "r") as handle:
        rows = np.asarray(handle[SIGMA_DATASET][rows_idx, :], np.float32)

    sigma = np.sqrt((rows[:, 0] ** 2 + rows[:, 4] ** 2 + rows[:, 8] ** 2) / 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_sigma = np.where(sigma > 0, np.log10(sigma), np.nan)

    print("Reading ex-situ fractions...")
    halo_selected = sgn[mask].astype(np.int64)
    exsitu = np.full(halo_selected.shape, np.nan, np.float32)

    with h5py.File(find_exsitu_file(), "r") as handle:
        dset = handle["stars"]
        ids_all = np.asarray(dset[:, 0], np.int64)
        fracs_all = np.asarray(dset[:, 3], np.float32)

    order = np.argsort(ids_all)
    ids_sorted = ids_all[order]
    fracs_sorted = fracs_all[order]
    pos = np.searchsorted(ids_sorted, halo_selected)
    safe = np.minimum(pos, ids_sorted.size - 1)
    ok = (pos < ids_sorted.size) & (ids_sorted[safe] == halo_selected)
    exsitu[ok] = fracs_sorted[pos[ok]]

    quantities = np.vstack([age, metallicity, mgfe, log_ssfr, log_sigma, exsitu])
    return compactness, quantities


# =========================
# PANEL CALCULATIONS
# =========================

def kde_grid(x, y):
    finite = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[finite], y[finite]

    xlo, xhi = np.nanpercentile(xs, [1, 99])
    ylo, yhi = np.nanpercentile(ys, [1, 99])
    xpad = 0.05 * (xhi - xlo + 1e-9)
    ypad = 0.05 * (yhi - ylo + 1e-9)

    xg = np.linspace(xlo - xpad, xhi + xpad, KDE_NX)
    yg = np.linspace(ylo - ypad, yhi + ypad, KDE_NY)
    Xg, Yg = np.meshgrid(xg, yg)

    kde = gaussian_kde(np.vstack([xs, ys]), bw_method="scott")
    pts = np.column_stack((Xg.ravel(), Yg.ravel()))
    tree = KDTree(np.column_stack((xs, ys)))

    d_grid, _ = tree.query(pts, k=1)
    d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
    typical = float(np.nanpercentile(d_data[:, 1], 95))
    cut = max(typical * KDE_DISTANCE_FACTOR, 1e-6)

    Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    Z.ravel()[d_grid > cut] = np.nan

    vals = Z[np.isfinite(Z)]
    levels = np.percentile(vals, KDE_CONTOUR_PERCENTILES)
    return Xg, Yg, Z, levels


def quantity_relation(compactness, quantity, key):
    valid = np.isfinite(compactness) & np.isfinite(quantity)
    x = compactness[valid]
    y = quantity[valid]

    nbins = max(3, int(np.floor(x.size / TARGET_PER_BIN[key])))
    edges = np.nanpercentile(x, np.linspace(0, 100, nbins + 1))
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.searchsorted(edges, compactness, side="right") - 1

    groups = defaultdict(list)
    usable = (
        (idx >= 0)
        & (idx < nbins)
        & np.isfinite(compactness)
        & np.isfinite(quantity)
    )
    for i, q in zip(idx[usable], quantity[usable]):
        groups[int(i)].append(float(q))

    median = np.full(nbins, np.nan)
    low = np.full(nbins, np.nan)
    high = np.full(nbins, np.nan)
    counts = np.zeros(nbins, int)

    for b in range(nbins):
        vals = np.asarray(groups.get(b, []), float)
        counts[b] = vals.size
        if vals.size >= MIN_COUNT_PER_BIN:
            median[b] = np.nanmedian(vals)
            low[b] = np.nanpercentile(vals, 16)
            high[b] = np.nanpercentile(vals, 84)

    smooth, deriv, threshold, method = find_threshold(
        centers, median, x, counts, key
    )
    Xg, Yg, Z, levels = kde_grid(x, y)

    return {
        "Xg": Xg,
        "Yg": Yg,
        "Z": Z,
        "levels": levels,
        "centers": centers,
        "median": median,
        "low": low,
        "high": high,
        "smooth": smooth,
        "threshold": np.asarray(threshold),
        "method": np.asarray(method),
    }


# =========================
# CACHE
# =========================

def build_cache():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    compactness, quantities = load_full_data()

    cache = {}
    for i, (key, q) in enumerate(zip(QUANTITY_KEYS, quantities)):
        print("Processing", key)
        result = quantity_relation(compactness, q, key)
        for name, value in result.items():
            cache[f"p{i}_{name}"] = np.asarray(value)

    np.savez_compressed(CACHE_FILE, **cache)
    print("Saved cache:", CACHE_FILE)
    return cache


def load_cache():
    with np.load(CACHE_FILE, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def layout_cache():
    cache = {}
    rng = np.random.default_rng(1234)

    for i, yrange in enumerate(LAYOUT_Y_LIMITS):
        ymin, ymax = yrange
        x = rng.normal(9.75, 0.32, 10000)
        y0 = ymin + 0.55 * (ymax - ymin)

        xg = np.linspace(*X_LIMITS, 160)
        yg = np.linspace(ymin, ymax, 140)
        Xg, Yg = np.meshgrid(xg, yg)
        Z = np.exp(
            -0.5 * ((Xg - 9.75) / 0.32) ** 2
            -0.5 * ((Yg - y0) / (0.22 * (ymax - ymin))) ** 2
        )
        Z[Z < 0.01] = np.nan

        centers = np.linspace(9.0, 10.5, 35)
        median = y0 + 0.18 * (ymax - ymin) * np.tanh((centers - 9.75) * 1.7)

        result = {
            "Xg": Xg,
            "Yg": Yg,
            "Z": Z,
            "levels": np.array([0.08, 0.2, 0.4, 0.7]),
            "centers": centers,
            "median": median,
            "low": median - 0.12 * (ymax - ymin),
            "high": median + 0.12 * (ymax - ymin),
            "smooth": median,
            "threshold": np.asarray(9.75),
            "method": np.asarray("layout"),
        }
        for name, value in result.items():
            cache[f"p{i}_{name}"] = np.asarray(value)

    return cache


# =========================
# DRAWING
# =========================

def panel_data(cache, i):
    out = {}
    for name in [
        "Xg", "Yg", "Z", "levels", "centers", "median",
        "low", "high", "smooth", "threshold", "method"
    ]:
        out[name] = cache[f"p{i}_{name}"]
    out["threshold"] = float(out["threshold"])
    out["method"] = str(out["method"].item())
    return out


def style_axis(ax, index):
    row, col = divmod(index, 2)
    ax.set_xlim(*X_LIMITS)
    ax.grid(True, alpha=0.55)
    ax.tick_params(
        axis="both",
        labelsize=TICK_FONT,
        direction="in",
        length=6,
        width=1.0,
        top=True,
        right=True,
    )

    if SHOW_XLABEL_ONLY_BOTTOM and row < 2:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(X_LABEL, fontsize=AXIS_LABEL_FONT)

    if SHOW_YLABEL_ONLY_LEFT and col == 1:
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel(Y_LABELS[index], fontsize=AXIS_LABEL_FONT)

    ax.text(
        0.015, 0.975, PANEL_LABELS[index],
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PANEL_LABEL_FONT,
        zorder=20,
    )


def draw_mosaic(cache, output_stem):
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": GLOBAL_FONT,
    })

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        3, 5,
        width_ratios=WIDTH_RATIOS,
        left=LEFT,
        right=RIGHT,
        bottom=BOTTOM,
        top=TOP,
        hspace=HSPACE,
        wspace=WSPACE,
    )

    first_ax = None

    for i in range(6):
        row, col = divmod(i, 2)
        main_col, cbar_col = (0, 1) if col == 0 else (3, 4)

        if first_ax is None:
            ax = fig.add_subplot(gs[row, main_col])
            first_ax = ax
        else:
            ax = fig.add_subplot(gs[row, main_col], sharex=first_ax)

        cax = fig.add_subplot(gs[row, cbar_col])
        p = panel_data(cache, i)

        cf = ax.contourf(
            p["Xg"], p["Yg"], np.ma.masked_invalid(p["Z"]),
            levels=KDE_LEVELS, cmap="viridis", antialiased=True
        )
        ax.contour(
            p["Xg"], p["Yg"], np.ma.masked_invalid(p["Z"]),
            levels=p["levels"], colors="black",
            linewidths=0.6, alpha=0.5
        )

        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label("Density (KDE)", fontsize=CBAR_LABEL_FONT)
        cbar.ax.tick_params(labelsize=CBAR_TICK_FONT)

        ok = np.isfinite(p["median"])
        ax.plot(
            p["centers"][ok], p["median"][ok],
            color="black", lw=2, label="median (binned)", zorder=10
        )
        ax.fill_between(
            p["centers"], p["low"], p["high"],
            color="black", alpha=0.15, zorder=8
        )
        ax.plot(
            p["centers"], p["smooth"],
            ls="--", color="C2", lw=2,
            label="smoothed median", zorder=11
        )
        ax.axvline(
            p["threshold"], color="C3", ls="--", lw=1.7,
            label=rf"threshold $={p['threshold']:.2f}$", zorder=12
        )

        style_axis(ax, i)

        if SHOW_PANEL_LEGENDS:
            ax.legend(
                loc="best", fontsize=LEGEND_FONT,
                frameon=True, borderpad=0.35, handlelength=2.0
            )

    pdf = output_stem.with_suffix(".pdf")
    png = output_stem.with_suffix(".png")
    fig.savefig(pdf, dpi=300)
    fig.savefig(png, dpi=180)
    plt.close(fig)
    print("Saved:", pdf)
    print("Saved:", png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("layout", "full", "cached"), default="layout"
    )
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "layout":
        draw_mosaic(layout_cache(), LAYOUT_OUTPUT)
    elif args.mode == "full":
        draw_mosaic(build_cache(), FINAL_OUTPUT)
    else:
        draw_mosaic(load_cache(), FINAL_OUTPUT)


if __name__ == "__main__":
    main()
