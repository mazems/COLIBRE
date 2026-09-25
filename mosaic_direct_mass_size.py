#!/usr/bin/env python3
"""
mosaic_direct_mass_size.py

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
from scipy.spatial import cKDTree as KDTree


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
CACHE_FILE = OUTDIR / "mass_size_mosaic_loess_cache.npz"
LAYOUT_OUTPUT = OUTDIR / "mass_size_mosaic_layout"
FINAL_OUTPUT = OUTDIR / "mass_size_mosaic_3x2"

MASS_LIMIT = 1.0e9
ZSUN = 0.0134
LOG_MGFE_SUN = 0.10
SSFR_FLOOR = -12.0
LOGSIGMA_REF = 9.75

X_LIMITS = (8.95, 12.20)
Y_LIMITS = (-0.50, 1.55)

LOESS_FRAC = 0.01
LOESS_DEGREE = 1
GRID_NX = 300
GRID_NY = 220
DISTANCE_MASK_FACTOR = 1.3

FIGSIZE = (13.0, 15.0)
MAIN_TO_CBAR_WIDTH = [1.0, 0.045, 0.15, 1.0, 0.045]
LEFT = 0.075
RIGHT = 0.91
BOTTOM = 0.065
TOP = 0.985
HSPACE = 0.04
WSPACE = 0.025

SHOW_XLABEL_ONLY_BOTTOM = True
SHOW_YLABEL_ONLY_LEFT = True
SHOW_PANEL_LEGENDS = True
SHOW_MISSING_POINTS = True

PANEL_LABELS = ["a)", "b)", "c)", "d)", "e)", "f)"]
QUANTITY_KEYS = ["age", "metallicity", "mgfe", "ssfr", "sigma", "exsitu"]
COLORBAR_LABELS = [
    r"$\text{Age} [\mathrm{Gyr}]$",
    r"$[\mathrm{Z}/\mathrm{H}]$",
    r"$[\mathrm{Mg}/\mathrm{Fe}]$",
    r"$\log_{10}(\mathrm{sSFR}\,/\,\mathrm{yr}^{-1})$",
    r"$\log_{10}(\sigma\,/\,\mathrm{km}\,\mathrm{s}^{-1})$",
    r"$f_{\mathrm{ex\! -\! situ}}$",
]

LAYOUT_VMIN = np.array([2.0, -0.40, 0.08, -12.0, 1.30, 0.00])
LAYOUT_VMAX = np.array([10.0, 0.10, 0.18, -9.80, 1.95, 0.40])


# =============================================================================
# LOESS HELPERS
# =============================================================================

def polyfit_2d(x, y, z, degree=1, weights=None):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    w = np.ones_like(z) if weights is None else np.asarray(weights, dtype=float).ravel()
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(w)
    x, y, z, w = x[good], y[good], z[good], w[good]

    if z.size == 0 or np.sum(w) <= 0:
        return np.array([np.nan]) if degree == 0 else np.array([np.nan] * 3)

    xc = np.average(x, weights=w)
    yc = np.average(y, weights=w)
    dx = x - xc
    dy = y - yc

    if degree == 0:
        return np.array([(w @ z) / np.sum(w)])
    if degree != 1:
        raise NotImplementedError("Only degree 0 or 1 is supported.")

    design = np.column_stack((np.ones_like(dx), dx, dy))
    atw = design.T * w
    ata = atw @ design
    aty = atw @ z

    trace = np.trace(ata)
    ridge = 1e-12 * trace if np.isfinite(trace) and trace != 0 else 1e-12
    ata[0, 0] += ridge

    try:
        beta = np.linalg.solve(ata, aty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(ata) @ aty

    return np.asarray(beta)


def _biweight_scale(resid):
    resid = np.abs(np.asarray(resid, dtype=float))
    if resid.size == 0:
        return 1.0
    mad = np.nanmedian(resid)
    return 1.4826 * mad if np.isfinite(mad) and mad > 0 else 1e-9


def loess_2d(x1, y1, z, frac=0.01, degree=1, npoints=None, sigz=None, xout=None, yout=None):
    x1 = np.asarray(x1, dtype=float).ravel()
    y1 = np.asarray(y1, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    if not (x1.size == y1.size == z.size):
        raise ValueError("x1, y1 and z must have equal lengths.")

    n = x1.size
    if n == 0:
        return np.array([]), np.array([])

    if npoints is None:
        npoints = int(np.ceil(frac * n))
    npoints = max(2, min(int(npoints), n))

    if xout is None or yout is None:
        xout = x1.copy()
        yout = y1.copy()
    else:
        xout = np.asarray(xout, dtype=float).ravel()
        yout = np.asarray(yout, dtype=float).ravel()

    if xout.size != yout.size:
        raise ValueError("xout and yout must have equal lengths.")

    zout = np.empty(xout.size, dtype=float)
    wout = np.empty(xout.size, dtype=float)
    tree = KDTree(np.column_stack((x1, y1)))

    for j, (xx, yy) in enumerate(zip(xout, yout)):
        dists, inds = tree.query([xx, yy], k=npoints)
        dists = np.atleast_1d(dists)
        inds = np.atleast_1d(inds)

        rmax = np.max(dists)
        if rmax == 0:
            zout[j] = z[inds[0]]
            wout[j] = 1.0
            continue

        u = dists / rmax
        dist_weights = np.where(u >= 1.0, 0.0, (1.0 - u**3) ** 3)
        xw, yw, zw = x1[inds], y1[inds], z[inds]

        coeffs = polyfit_2d(xw, yw, zw, degree=degree, weights=dist_weights)

        if degree == 0:
            zfit = np.full_like(zw, coeffs[0], dtype=float)
        else:
            if np.sum(dist_weights) > 0:
                xc = np.average(xw, weights=dist_weights)
                yc = np.average(yw, weights=dist_weights)
            else:
                xc, yc = np.mean(xw), np.mean(yw)
            zfit = coeffs[0] + coeffs[1] * (xw - xc) + coeffs[2] * (yw - yc)

        biweights = np.ones_like(zw)

        for _ in range(10):
            if sigz is None:
                scale = _biweight_scale(zfit - zw)
                uu = (np.abs(zfit - zw) / (6.0 * scale)) ** 2
            else:
                uu = ((zfit - zw) / (4.0 * sigz[inds])) ** 2

            uu = np.clip(uu, 0.0, 1.0)
            new_biweights = (1.0 - uu) ** 2
            total_weights = dist_weights * new_biweights
            if np.sum(total_weights) <= 0:
                total_weights = dist_weights.copy()

            coeffs = polyfit_2d(xw, yw, zw, degree=degree, weights=total_weights)

            if degree == 0:
                zfit = np.full_like(zw, coeffs[0], dtype=float)
            else:
                if np.sum(total_weights) > 0:
                    xc = np.average(xw, weights=total_weights)
                    yc = np.average(yw, weights=total_weights)
                else:
                    xc, yc = np.mean(xw), np.mean(yw)
                zfit = coeffs[0] + coeffs[1] * (xw - xc) + coeffs[2] * (yw - yc)

            if np.allclose(biweights, new_biweights, atol=1e-6):
                biweights = new_biweights
                break
            biweights = new_biweights

        zout[j] = coeffs[0]
        wout[j] = biweights[0] if biweights.size else 1.0

    return zout, wout


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


def compute_loess_grid(x, y, z, Xg, Yg, key):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    have = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    missing = ~np.isfinite(z)
    xvals, yvals, zvals = x[have], y[have], z[have]

    output = np.full(Xg.shape, np.nan, dtype=float)
    vmin, vmax = robust_limits(zvals, key)
    if zvals.size < 2:
        return output, vmin, vmax, missing

    points_grid = np.column_stack((Xg.ravel(), Yg.ravel()))
    tree = KDTree(np.column_stack((xvals, yvals)))
    d_grid, _ = tree.query(points_grid, k=1)
    d_data, _ = tree.query(np.column_stack((xvals, yvals)), k=2)

    if d_data.ndim == 2 and d_data.shape[1] >= 2:
        typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
    else:
        typical_spacing = float(np.nanmedian(d_grid))

    threshold = max(typical_spacing * DISTANCE_MASK_FACTOR, 1e-6)
    inside_idx = np.flatnonzero(d_grid <= threshold)
    if inside_idx.size == 0:
        return output, vmin, vmax, missing

    xout = points_grid[inside_idx, 0]
    yout = points_grid[inside_idx, 1]

    print(f"  {key:12s}: {zvals.size:7d} values, {inside_idx.size:7d} grid evaluations")
    z_inside, _ = loess_2d(
        xvals, yvals, zvals,
        frac=LOESS_FRAC,
        degree=LOESS_DEGREE,
        xout=xout,
        yout=yout,
    )

    flat = output.ravel()
    flat[inside_idx] = z_inside
    return flat.reshape(Xg.shape), vmin, vmax, missing


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
            "LuminosityWeightedMeanStellarAge",
            "LinearMassWeightedIronOverHydrogenOfStars",
            "LinearMassWeightedMagnesiumOverHydrogenOfStars",
            "StellarMassFractionInMetals",
        )
    }
    fields_ids = {"InputHalos": ("HaloCatalogueIndex", "HBTplus/TrackId")}

    print("Reading SOAP quantities...")
    (
        m30,
        sfr30,
        r50,
        stellarage_lum,
        fe_over_h,
        mg_over_h,
        zstar_raw,
    ) = common.read_group_data_colibre(str(MODEL_DIR) + os.sep, SNAP_FILE, fields)
    sgn, track_id = common.read_group_data_colibre(
        str(MODEL_DIR) + os.sep, SNAP_FILE, fields_ids
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

    return log_m, log_r, quantities


# =============================================================================
# CACHE
# =============================================================================

def build_full_cache():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    log_m, log_r, quantities = load_full_data()

    xgrid = np.linspace(X_LIMITS[0], X_LIMITS[1], GRID_NX)
    ygrid = np.linspace(Y_LIMITS[0], Y_LIMITS[1], GRID_NY)
    Xg, Yg = np.meshgrid(xgrid, ygrid)

    grids, vmins, vmaxs, missing_masks = [], [], [], []
    print("Computing LOESS grids...")
    for key, z in zip(QUANTITY_KEYS, quantities):
        grid, vmin, vmax, missing = compute_loess_grid(log_m, log_r, z, Xg, Yg, key)
        grids.append(grid)
        vmins.append(vmin)
        vmaxs.append(vmax)
        missing_masks.append(missing)

    result = {
        "Xg": Xg,
        "Yg": Yg,
        "grids": np.stack(grids),
        "vmins": np.asarray(vmins),
        "vmaxs": np.asarray(vmaxs),
        "log_m": log_m,
        "log_r": log_r,
        "missing_masks": np.stack(missing_masks),
    }

    np.savez_compressed(CACHE_FILE, **result)
    print("Saved cache:", CACHE_FILE)
    return result


def load_cache():
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Cache not found: {CACHE_FILE}\nRun once with --mode full before --mode cached."
        )
    with np.load(CACHE_FILE, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


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


# =============================================================================
# MOSAIC DRAWING
# =============================================================================

def style_axis(ax, index):
    row, col = divmod(index, 2)
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.grid(True, alpha=0.55)
    ax.tick_params(
        axis="both",
        labelsize=12,
        direction="in",
        length=5,
        width=0.9,
        top=True,
        right=True,
    )

    if SHOW_XLABEL_ONLY_BOTTOM and row < 2:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r"$\log_{10}(M_\star / \mathrm{M}_\odot)$", fontsize=18)

    if SHOW_YLABEL_ONLY_LEFT and col == 1:
        ax.tick_params(labelleft=False)
    else:
        ax.set_ylabel(r"$\log_{10}(R_{1/2,\star} / \mathrm{kpc})$", fontsize=18)

    ax.text(
        0.025,
        0.975,
        PANEL_LABELS[index],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
    )


def draw_mosaic(data, output_stem, preview=False):
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.size": 18,
        }
    )

    Xg = data["Xg"]
    Yg = data["Yg"]
    grids = data["grids"]
    vmins = data["vmins"]
    vmaxs = data["vmaxs"]
    log_m = data["log_m"]
    log_r = data["log_r"]
    missing_masks = data["missing_masks"]

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=5,
        width_ratios=MAIN_TO_CBAR_WIDTH,
        left=LEFT,
        right=RIGHT,
        bottom=BOTTOM,
        top=TOP,
        hspace=HSPACE,
        wspace=WSPACE,
    )

    first_ax = None

    for index in range(6):
        row, col = divmod(index, 2)
        if col == 0:
            main_col = 0
            cbar_col = 1
        else:
            main_col = 3
            cbar_col = 4

        if first_ax is None:
            ax = fig.add_subplot(gs[row, main_col])
            first_ax = ax
        else:
            ax = fig.add_subplot(gs[row, main_col], sharex=first_ax, sharey=first_ax)

        cax = fig.add_subplot(gs[row, cbar_col])

        image = ax.pcolormesh(
            Xg,
            Yg,
            np.ma.masked_invalid(grids[index]),
            shading="auto",
            cmap="viridis",
            vmin=float(vmins[index]),
            vmax=float(vmaxs[index]),
            rasterized=True,
        )

        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label(COLORBAR_LABELS[index], fontsize=18)
        cbar.ax.tick_params(labelsize=12)

        if (
            not preview
            and SHOW_MISSING_POINTS
            and log_m.size
            and missing_masks.shape[1] == log_m.size
        ):
            missing = missing_masks[index]
            if np.any(missing):
                ax.scatter(
                    log_m[missing],
                    log_r[missing],
                    s=2,
                    color="lightgrey",
                    alpha=0.45,
                    linewidths=0,
                    rasterized=True,
                    label="missing",
                )

        threshold_mass = np.linspace(X_LIMITS[0], X_LIMITS[1], 300)
        threshold_radius = (2.0 / 3.0) * (threshold_mass - LOGSIGMA_REF)
        ax.plot(
            threshold_mass,
            threshold_radius,
            linestyle="--",
            color="black",
            linewidth=1.25,
            label=(
                f"compactness threshold\n"
                rf"$\log_{{10}}(\Sigma_{{1.5}}/(\mathrm{{M}}_\odot\,\mathrm{{kpc}}^{{-1.5}}))" rf" = {LOGSIGMA_REF}$" 
            ),
        )

        style_axis(ax, index)

        if SHOW_PANEL_LEGENDS:
            ax.legend(
                loc="lower right",
                fontsize=13,
                frameon=True,
                borderpad=0.35,
                handlelength=2.0,
            )

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=180)
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
        print("LAYOUT MODE: no simulation data or LOESS calculation.")
        draw_mosaic(make_layout_preview_data(), LAYOUT_OUTPUT, preview=True)
    elif args.mode == "full":
        print("FULL MODE: loading data and computing all LOESS surfaces.")
        draw_mosaic(build_full_cache(), FINAL_OUTPUT, preview=False)
    else:
        print("CACHED MODE: redrawing from", CACHE_FILE)
        draw_mosaic(load_cache(), FINAL_OUTPUT, preview=False)


if __name__ == "__main__":
    main()
