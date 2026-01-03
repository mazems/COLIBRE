#!/usr/bin/env python3
"""
relic_plots_interactive.py

Load merged relicness CSV and produce a suite of plots (histograms, scatter, hexbin, corr heatmap).
Saves figures to --outdir and shows them interactively.

Usage:
  python3 relic_plots_interactive.py --csv relicness_merged.csv
  python3 relic_plots_interactive.py --csv relicness_merged.csv --outdir figs --hexbin 60

Notes:
 - Script tries to compute 'logsigma' if possible from available mass & radius columns.
 - It will not overwrite your CSV. It prints exactly which columns it used.
"""
from __future__ import annotations
import os
import sys
import argparse
import math
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------- CLI ---------------------
parser = argparse.ArgumentParser(description="Make histograms/scatter/hexbin for relicness CSV")
parser.add_argument("--csv", "-c", required=True, help="Path to merged CSV (relicness_merged.csv)")
parser.add_argument("--outdir", "-o", default="relic_plots", help="Directory to save figures")
parser.add_argument("--hexbin", type=int, default=40, help="hexbin gridsize (default 40)")
parser.add_argument("--no-show", action="store_true", help="Do not call plt.show() (only save figs)")
parser.add_argument("--quiet", action="store_true", help="Reduce console output")
args = parser.parse_args()

OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)

def info(msg: str):
    if not args.quiet:
        print(msg, flush=True)

# ------------------ helpers --------------------
def try_find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first matching column name in df from candidates (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        lc = cand.lower()
        if lc in cols_lower:
            return cols_lower[lc]
    # also try partial substring matches
    for cand in candidates:
        lc = cand.lower()
        for col in df.columns:
            if lc in col.lower():
                return col
    return None

def safe_log10(x):
    return np.log10(x)

# ------------------ load data ------------------
info(f"Loading CSV: {args.csv}")
df = pd.read_csv(args.csv)
n_rows = df.shape[0]
info(f"Loaded {n_rows} rows, {df.shape[1]} columns.")

# --- CLEAN DoR (must be between 0 and 1) ---
if "DoR" in df.columns:
    DoR_raw = pd.to_numeric(df["DoR"], errors="coerce")
    bad = (~DoR_raw.between(0, 1)) | (~np.isfinite(DoR_raw))
    n_bad = bad.sum()
    if n_bad > 0:
        info(f"Cleaning DoR: removing {n_bad} invalid rows (DoR must be in [0,1]).")
    df.loc[bad, "DoR"] = np.nan

# ------------------ attempt to compute logsigma ------------------
# logsigma := log10(M_star_Msun) - 1.5 * log10(r_half_kpc)

# candidate column names for SOAP mass (preferred), then particle current mass as fallback.
mass_candidates_preferred = [
    "StellarMass", "stellar_mass_soap", "stellar_mass_total",
    "Mstar", "M_star", "stellar_mass", "Stellar_Mass", "stellarmass", "stellar_mass_SOAP"
]
# column names that might represent current particle-level mass (often in 1e10 Msun)
mass_candidates_fallback = [
    "stellar_mass_current", "stellar_mass_curr", "stellar_mass_particle", "stellar_mass_current_1e10Msun"
]

rhalf_candidates = [
    "stellar_halfmass_radius_kpc", "r_half_kpc", "half_mass_radius_kpc",
    "HalfMassRadiusStars", "HalfMassRadius", "half_mass_radius", "rhalf", "half_mass_radius_stars"
]

mass_col = try_find_column(df, mass_candidates_preferred)
used_mass_source = None
mass_in_msun = None  # numpy array in Msun when found

if mass_col is not None:
    info(f"Found preferred SOAP mass column: '{mass_col}' -> using this for logsigma if radius present.")
    used_mass_source = mass_col
    # assume values are in Msun (common for SOAP). If values are very small (<1), warn (maybe in 1e10 units)
    mass_vals = pd.to_numeric(df[mass_col], errors="coerce").to_numpy(dtype=float)
    # If most values are <1e-2, assume maybe values are in 1e10 Msun -> multiply by 1e10
    median = np.nanmedian(np.abs(mass_vals)) if mass_vals.size>0 else np.nan
    if np.isfinite(median) and median < 1e-2:
        info(f"  NOTE: median({mass_col}) ~ {median:.3e} -> appears small; assuming it's in 1e10 Msun and converting -> Msun")
        mass_in_msun = mass_vals * 1e10
    else:
        mass_in_msun = mass_vals.copy()
else:
    # fallback to particle-level current mass columns (which your merged file likely contains, in 1e10 Msun)
    mass_col = try_find_column(df, mass_candidates_fallback)
    if mass_col is not None:
        info(f"No SOAP mass found; using fallback mass column '{mass_col}' (particle-level current mass).")
        used_mass_source = mass_col
        mass_vals = pd.to_numeric(df[mass_col], errors="coerce").to_numpy(dtype=float)
        # assume this is in 1e10 Msun (common in your pipeline). Convert to Msun:
        info("  Assuming this column is in 1e10 Msun -> converting to Msun for logsigma.")
        mass_in_msun = mass_vals * 1e10
    else:
        info("No suitable mass column found for logsigma computation. Logsigma-related plots will be skipped.")

# find radius
r_col = try_find_column(df, rhalf_candidates)
r_vals = None
if r_col is not None:
    info(f"Found half-mass radius column: '{r_col}' (assumed in kpc).")
    r_vals = pd.to_numeric(df[r_col], errors="coerce").to_numpy(dtype=float)
else:
    info("No half-mass radius column found for logsigma computation. Logsigma-related plots will be skipped.")

# --- logsigma must already be present and correct ---
if "logsigma" not in df.columns:
    raise RuntimeError("CSV does not contain logsigma. Refusing to recompute it.")

# ------------------ select columns for plotting ------------------
def col_or_none(names):
    c = try_find_column(df, names)
    return c

cols_needed = {
    "t_start": col_or_none(["t_start", "tstart", "t_start_gyr"]),
    "t50_span": col_or_none(["t50_span", "t50span", "t50_span_gyr", "t50_span_gyr"]),
    "t75_span": col_or_none(["t75_span", "t75span"]),
    "t90_span": col_or_none(["t90_span", "t90span", "t90_span_gyr"]),
    "f_Mz2": col_or_none(["f_Mz2", "fMz2", "f_M_z2"]),
    "DoR": col_or_none(["DoR", "dor", "DoR_value"]),
    "stellar_mass_current_1e10": col_or_none(["stellar_mass_current", "stellar_mass_current_1e10", "stellar_mass_current_1e10Msun"])
}
info("Column detection summary:")
for k, v in cols_needed.items():
    info(f"  {k}: {'FOUND -> ' + v if v else 'MISSING'}")
info(f"  logsigma: {'FOUND' if 'logsigma' in df.columns else 'MISSING'}")
if used_mass_source:
    info(f"Used mass source for logsigma calculation: {used_mass_source}")
if r_col:
    info(f"Used radius source for logsigma calculation: {r_col}")

# ------------------ prepare cleaned df for plotting ------------------
# pick a working dataframe with numeric columns we may plot
plot_df = df.copy()

# If stellar_mass_current is in 1e10Msun, but the column name may be 'stellar_mass_current', keep as is for hist.
# Many of the histograms use raw units (Gyr or 1e10 Msun) — label axes accordingly.

# For plotting we will create a numeric-only view for the quantities we will use
# Create a small helper to safely get array or None
def get_arr(name):
    if name is None:
        return None
    return pd.to_numeric(plot_df[name], errors="coerce").to_numpy(dtype=float)

t_start_arr = get_arr(cols_needed["t_start"])
t50_span_arr = get_arr(cols_needed["t50_span"])
t90_span_arr = get_arr(cols_needed["t90_span"])
f_Mz2_arr = get_arr(cols_needed["f_Mz2"])
dor_arr = get_arr(cols_needed["DoR"])
mcur_1e10_arr = get_arr(cols_needed["stellar_mass_current_1e10"])  # in 1e10 Msun if present
logsigma_arr = plot_df["logsigma"].to_numpy(dtype=float) if "logsigma" in plot_df.columns else None

# ------------------ plotting ------------------
info("Creating plots... (will save PNGs to directory: {})".format(OUTDIR))

plt.rcParams.update({'figure.dpi': 150})

# simple function to save-and-show (or only save)
def save_fig(fig, fname):
    outpath = os.path.join(OUTDIR, fname)
    try:
        fig.savefig(outpath, bbox_inches='tight')
        info(f"  saved: {outpath}")
    except Exception as e:
        info(f"  failed to save {outpath}: {e}")

# 1) histograms: t_start, t50_span, t90_span, f_Mz2, DoR, stellar_mass_current
def make_hist(array, xlabel, title, fname, bins=40, log=False):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(1,1,1)
    arr = array[~np.isnan(array)] if array is not None else np.array([])
    if arr.size == 0:
        info(f"  SKIP histogram {title}: no valid data")
        plt.close(fig)
        return
    if log:
        ax.hist(arr[arr>0], bins=bins)
        ax.set_xscale('log')
    else:
        ax.hist(arr, bins=bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    save_fig(fig, fname)
    if not args.no_show:
        plt.show()
    plt.close(fig)

if t_start_arr is not None:
    make_hist(t_start_arr, "t_start (Gyr)", "Distribution of t_start", "hist_t_start.png", bins=50)
if t50_span_arr is not None:
    make_hist(t50_span_arr, "t50_span (Gyr)", "Distribution of t50_span (burstiness)", "hist_t50_span.png", bins=50)
if t90_span_arr is not None:
    make_hist(t90_span_arr, "t90_span (Gyr)", "Distribution of t90_span", "hist_t90_span.png", bins=50)
if f_Mz2_arr is not None:
    make_hist(f_Mz2_arr, "f_Mz2", "Fraction formed before z=2", "hist_f_Mz2.png", bins=40)
if dor_arr is not None:
    make_hist(dor_arr, "DoR", "Degree of Relicness (DoR)", "hist_DoR.png", bins=40)
if mcur_1e10_arr is not None:
    # plot stellar mass in 1e10 Msun
    arr = mcur_1e10_arr
    make_hist(arr, "stellar_mass_current (1e10 Msun)", "Current stellar mass (particle-level)", "hist_stellar_mass_current_1e10.png", bins=50, log=False)

# 2) scatter: t_start vs t50_span (with hexbin overlay)
def scatter_or_hex(x, y, xlabel, ylabel, title, fname, gridsize=40):
    if x is None or y is None:
        info(f"  SKIP scatter {title}: missing columns")
        return
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 10:
        info(f"  SKIP scatter {title}: too few valid points ({np.sum(mask)})")
        return
    # create hexbin plot
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(1,1,1)
    hb = ax.hexbin(x[mask], y[mask], gridsize=gridsize, mincnt=1, cmap='viridis')
    fig.colorbar(hb, ax=ax, label='counts')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_fig(fig, fname)
    if not args.no_show:
        plt.show()
    plt.close(fig)

scatter_or_hex(t_start_arr, t50_span_arr, "t_start (Gyr)", "t50_span (Gyr)", "Earliness vs Burstiness (t_start vs t50_span)", "hex_tstart_t50span.png", gridsize=args.hexbin)

# 3) logsigma vs DoR and logsigma vs f_Mz2
if ("logsigma" in plot_df.columns):
    logsigma_arr_local = plot_df["logsigma"].to_numpy(dtype=float)
    # scatter logsigma vs DoR
    scatter_or_hex(logsigma_arr_local, dor_arr, "logsigma", "DoR", "logsigma vs DoR", "hex_logsigma_DoR.png", gridsize=args.hexbin)
    scatter_or_hex(logsigma_arr_local, f_Mz2_arr, "logsigma", "f_Mz2", "logsigma vs f_Mz2", "hex_logsigma_fMz2.png", gridsize=args.hexbin)
else:
    info("Skipping logsigma vs DoR / f_Mz2 plots: logsigma not available.")

# --- New helper: scatter with optional color (mass) ----
def scatter_with_color(x, y, c=None, xlabel="", ylabel="", title="", fname="", gridsize=None, cmap=None, log_color=False):
    """
    If gridsize is provided -> draw hexbin; otherwise simple scatter colored by c.
    c may be None (no color) or an array of same length as x,y.
    """
    if x is None or y is None:
        info(f"  SKIP {title}: missing columns")
        return
    mask = np.isfinite(x) & np.isfinite(y)
    if c is not None:
        mask &= np.isfinite(c)
    if np.sum(mask) < 10:
        info(f"  SKIP {title}: too few valid points ({np.sum(mask)})")
        return

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(1,1,1)

    if gridsize is not None:
        # hexbin with optional colormap
        if c is None:
            hb = ax.hexbin(x[mask], y[mask], gridsize=gridsize, mincnt=1)
        else:
            # hexbin weighted by counts (default) — color by counts; to color by c you'd need bin-aggregation
            hb = ax.hexbin(x[mask], y[mask], C=c[mask], reduce_C_function=np.nanmedian,
                           gridsize=gridsize, mincnt=1, cmap=cmap)
        cb = fig.colorbar(hb, ax=ax, label='median(color)' if c is not None else 'counts')
    else:
        sc = ax.scatter(x[mask], y[mask], c=(c[mask] if c is not None else None),
                        s=8, alpha=0.6, cmap=cmap)
        if c is not None:
            if log_color:
                from matplotlib.colors import LogNorm
                sc.set_norm(LogNorm())
            fig.colorbar(sc, ax=ax, label='color')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_fig(fig, fname)
    if not args.no_show:
        plt.show()
    plt.close(fig)
    

# 4) t_start vs DoR scatter (simple scatter with alpha)
def scatter_simple(x, y, xlabel, ylabel, title, fname):
    if x is None or y is None:
        info(f"  SKIP scatter {title}: missing")
        return
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 10:
        info(f"  SKIP scatter {title}: too few valid points ({np.sum(mask)})")
        return
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x[mask], y[mask], s=8, alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_fig(fig, fname)
    if not args.no_show:
        plt.show()
    plt.close(fig)

scatter_simple(t_start_arr, dor_arr, "t_start (Gyr)", "DoR", "t_start vs DoR", "scatter_tstart_DoR.png")

# Compactness (logsigma) vs earliness (t_start)
if "logsigma" in plot_df.columns and t_start_arr is not None:
    scatter_with_color(
        plot_df["logsigma"].to_numpy(dtype=float), t_start_arr,
        c=mcur_1e10_arr,  # color by current stellar mass (optional)
        xlabel="logsigma", ylabel="t_start (Gyr)",
        title="Compactness (logsigma) vs Earliness (t_start)",
        fname="hex_logsigma_tstart.png",
        gridsize=args.hexbin,
        cmap='viridis', log_color=True
    )

# Compactness (logsigma) vs burstiness (t50_span)
if "logsigma" in plot_df.columns and t50_span_arr is not None:
    scatter_with_color(
        plot_df["logsigma"].to_numpy(dtype=float), t50_span_arr,
        c=mcur_1e10_arr,
        xlabel="logsigma", ylabel="t50_span (Gyr)",
        title="Compactness (logsigma) vs Burstiness (t50_span)",
        fname="hex_logsigma_t50span.png",
        gridsize=args.hexbin,
        cmap='viridis', log_color=True
    )

# If you want a simple scatter (unbinned) as well:
if "logsigma" in plot_df.columns and dor_arr is not None:
    scatter_simple(plot_df["logsigma"].to_numpy(dtype=float), dor_arr,
                   "logsigma", "DoR", "logsigma vs DoR", "scatter_logsigma_DoR.png")

# 5) correlation heatmap for numeric columns of interest
numeric_cols = []
for k in ("t_start", "t50_span", "t90_span", "f_Mz2", "DoR", "logsigma"):
    # map to actual col names
    if k == "logsigma":
        colname = "logsigma" if "logsigma" in plot_df.columns else None
    else:
        colname = cols_needed.get(k)
    if colname is not None and colname in plot_df.columns:
        numeric_cols.append(colname)
    elif k == "logsigma" and "logsigma" in plot_df.columns:
        numeric_cols.append("logsigma")

# also include stellar_mass_current if present
if cols_needed["stellar_mass_current_1e10"] is not None:
    numeric_cols.append(cols_needed["stellar_mass_current_1e10"])

numeric_cols = [c for c in numeric_cols if c is not None]
if len(numeric_cols) >= 2:
    info("Computing correlation matrix for columns: " + ", ".join(numeric_cols))
    corr_df = plot_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = corr_df.corr()
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    ax.set_title("Correlation matrix")
    save_fig(fig, "corr_matrix.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)
else:
    info("Not enough numeric columns to compute correlation matrix (need >=2).")

info("All plots created. Figures saved in directory: {}".format(OUTDIR))
info("If running in a terminal, plots were displayed unless you passed --no-show.")

# optional: write augmented CSV with logsigma if computed
if "logsigma" in plot_df.columns:
    out_csv_aug = os.path.join(OUTDIR, "augmented_with_logsigma.csv")
    try:
        plot_df.to_csv(out_csv_aug, index=False)
        info(f"Augmented CSV written: {out_csv_aug}")
    except Exception as e:
        info(f"Failed to write augmented CSV: {e}")

# keep the figures on screen in interactive mode (if requested)
if not args.no_show:
    # Matplotlib windows already shown via plt.show() within functions.
    pass

info("Done.")