#!/usr/bin/env python3
"""
DoR_relations.py  -- final, robust version

- Matches UCMG CSV (by subhalo_id / HaloCatalogueIndex) to SOAP (HaloCatalogueIndex).
- Loads SOAP-derived mass, r50, sfr, age, Mg/Fe (linear proxies) exactly like the working scripts.
- Loads ex-situ summary HDF5 by track id (if available) and matches to SOAP track ids.
- Produces plots in plots_dor/ and writes a small diagnostic CSV with matched entries.
- Does NOT modify any input files.
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import common
from scipy.spatial import cKDTree as KDTree, ConvexHull

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 13
})

# ------------------------ CONFIG ------------------------
csv_in = "sfh_times_all_with_DoR_variants.csv.gz"  # CSV with DoR information, for only UCMGs use "plots_dor/relicness_merged_allabove9p9_with_DoR_variants.csv" 
exsitu_h5 = "/mnt/su3ctm/kproctor/ForMax/L0200N3008_exsitu_summary.hdf5"  # ex-situ summary (optional)
model_name = 'L0200N3008/THERMAL_AGN/'
model_dir  = '/mnt/su3-pro/colibre/' + model_name
snap_file  = '0127'   # z=0
ztarget    = 0.0
comov_to_physical_length = 1.0 / (1.0 + ztarget)

outdir = "plots_dor"
os.makedirs(outdir, exist_ok=True)

COMPACTNESS_CUT = 9.8
EXTREME_DOR = 0.7
dor_column_candidates = ["DoR_t998"] #, "dor", "DoR_choice", "DoR_csv"]

# ------------------------ READ CSV (DoR lookup) ------------------------
if not os.path.exists(csv_in):
    raise SystemExit(f"UCMG CSV not found: {csv_in}")
print("Reading UCMG CSV:", csv_in)
df_ucmg = pd.read_csv(csv_in, low_memory=False)

# pick id column (prefer subhalo_id / HaloCatalogueIndex)
id_col = None
for c in ("subhalo_id", "HaloCatalogueIndex", "subhaloId", "HaloIndex", "track_id", "TrackId"):
    if c in df_ucmg.columns:
        id_col = c
        break
if id_col is None:
    id_col = df_ucmg.columns[0]
    print("Warning: no canonical id column found in CSV, falling back to", id_col)

# normalize numeric ids -> 'subhalo_id' (int)
s = df_ucmg[id_col].astype(str).str.replace("\r", "").str.strip()
df_ucmg["_subhalo_id_numeric"] = pd.to_numeric(s, errors="coerce").astype("Int64")
n_bad = int(df_ucmg["_subhalo_id_numeric"].isna().sum())
if n_bad > 0:
    print(f"Warning: {n_bad} rows in CSV have non-numeric {id_col} and will be ignored for matching.")
df_ucmg = df_ucmg[df_ucmg["_subhalo_id_numeric"].notna()].copy()
df_ucmg["subhalo_id"] = df_ucmg["_subhalo_id_numeric"].astype("int64")
df_ucmg.drop(columns=["_subhalo_id_numeric"], inplace=True)

# find DoR column
dor_col = None
for cand in dor_column_candidates:
    if cand in df_ucmg.columns:
        dor_col = cand
        break
if dor_col is None:
    for c in df_ucmg.columns:
        if c.lower().startswith("dor"):
            dor_col = c
            break
if dor_col is None:
    raise SystemExit("No DoR column found in CSV. Expected one of: " + ", ".join(dor_column_candidates))

print(f"Using CSV id column '{id_col}' normalized -> 'subhalo_id', DoR column: '{dor_col}'")

# build DoR lookup: subhalo_id -> DoR (float)
dor_lookup = {}
bad_dor = 0
for _, row in df_ucmg.iterrows():
    try:
        sid = int(row["subhalo_id"])
        v = row[dor_col]
        if pd.isna(v):
            continue
        dor_lookup[sid] = float(v)
    except Exception:
        bad_dor += 1
if bad_dor:
    print(f"Note: {bad_dor} CSV rows had unparsable DoR values and were skipped.")
ucmg_ids_set = set(df_ucmg["subhalo_id"].unique())
print(f"Unique UCMG subhalo ids in CSV: {len(ucmg_ids_set)} ; Loaded DoR entries: {len(dor_lookup)}")

# ------------------------ READ SOAP ------------------------
print("Reading SOAP groups (common.read_group_data_colibre)...")
fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')}
fields = {'ExclusiveSphere/50kpc': (
            'StellarMass', 'StarFormationRate', 'HalfMassRadiusStars',
            'MassWeightedMeanStellarAge', 'LuminosityWeightedMeanStellarAge',
            'LinearMassWeightedIronOverHydrogenOfStars',
            'LinearMassWeightedMagnesiumOverHydrogenOfStars'
         )}

h5data_groups   = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)

# Unpack — ensure HaloCatalogueIndex kept (sgn)
(halo_index, is_central, desc_id, track_id) = h5data_idgroups
(m30, sfr30, r50, stellarage, stellarage_lum, Fe_lin, Mg_lin) = h5data_groups

# conversions (same as working scripts)
Mu = 1.988e43 / 1.989e33
tu = 3.086e19 / 3.154e7
m30 = m30 * Mu
sfr30 = sfr30 * Mu / tu
r50 = r50 * comov_to_physical_length * 1e3
stellarage_lum = stellarage_lum * tu / 1e9

# selection: mass limit (same as before)
select = np.where(m30 >= 1e9)
m = m30[select]
r = r50[select]
halo_idx = halo_index[select]   # HaloCatalogueIndex values for selected entries
track = track_id[select]
sfr = sfr30[select]
Mg_lin = Mg_lin[select]
Fe_lin = Fe_lin[select]
age = stellarage_lum[select]

mask_pos = (m > 0) & (r > 0)
m = m[mask_pos]
r = r[mask_pos]
halo_idx = halo_idx[mask_pos]
track = track[mask_pos]
sfr = sfr[mask_pos]
Mg_lin = Mg_lin[mask_pos]
Fe_lin = Fe_lin[mask_pos]
age = age[mask_pos]

print(f"Selected SOAP galaxies after mass/radius filter: {len(m)}")

# derived quantities
logM = np.log10(m)
logR = np.log10(r)
compactness = logM - 1.5 * logR
with np.errstate(divide="ignore", invalid="ignore"):
    mgfe = np.where((Mg_lin > 0) & (Fe_lin > 0), np.log10(Mg_lin / Fe_lin) - 0.10, np.nan)
    ssfr = np.where((m > 0) & np.isfinite(sfr), sfr / m, np.nan)
    log_ssfr = np.where((ssfr > 0) & np.isfinite(ssfr), np.log10(ssfr), np.nan)

# ------------------------ LOAD ex-situ summary (optional) ------------------------
exsitu_lookup = {}
if os.path.exists(exsitu_h5):
    try:
        with h5py.File(exsitu_h5, "r") as fh:
            # try dataset 'stars' as used before
            if "stars" in fh:
                data = np.array(fh["stars"])
                if data.ndim == 2 and data.shape[1] >= 4:
                    track_ids = data[:,0].astype(int)
                    exfrac = data[:,3].astype(float)
                    exsitu_lookup = dict(zip(track_ids.tolist(), exfrac.tolist()))
                    print(f"Loaded {len(exsitu_lookup)} ex-situ entries from {exsitu_h5} (dataset 'stars').")
                else:
                    print("HDF5 'stars' dataset shape unexpected; skipping ex-situ matching.")
            else:
                # try common alternatives
                # scan top-level keys for a candidate dataset with shape (N,>=4)
                for k in fh:
                    ds = fh[k]
                    try:
                        arr = np.array(ds)
                        if arr.ndim == 2 and arr.shape[1] >= 4:
                            track_ids = arr[:,0].astype(int)
                            exfrac = arr[:,3].astype(float)
                            exsitu_lookup = dict(zip(track_ids.tolist(), exfrac.tolist()))
                            print(f"Loaded {len(exsitu_lookup)} ex-situ entries from {exsitu_h5} (dataset '{k}').")
                            break
                    except Exception:
                        continue
    except Exception as e:
        print("Warning: failed to read ex-situ HDF5:", e)
else:
    print("Ex-situ summary HDF5 not found at:", exsitu_h5, " -> skipping ex-situ matching.")

# ------------------------ MATCH UCMG CSV -> SOAP (HaloCatalogueIndex) ------------------------
# Build a pandas Series of DoR indexed by subhalo_id (this preserves all CSV entries)
dor_series = pd.Series(dor_lookup, dtype=float)
# Ensure halo_idx is integer array of the same shape as SOAP-selected halo_idx
halo_idx_int = halo_idx.astype(np.int64)

# Reindex the Series onto the SOAP-selected halo_idx values:
# dor_for_each_soap_row[i] = DoR of halo_idx[i] if present in CSV, else NaN
dor_for_each_soap_row = dor_series.reindex(halo_idx_int).to_numpy(dtype=float)

# Now find positions (indices in the SOAP-selected arrays) that have a DoR from the CSV
matched_positions = np.where(np.isfinite(dor_for_each_soap_row))[0]
matched_subids = halo_idx_int[matched_positions]
matched_dor = dor_for_each_soap_row[matched_positions].astype(float)

# ----------------- Build matched dictionaries for DoR variants and time columns ---------------
# arrays of SOAP mass/size for the matched UCMG positions (used by scatter/LOESS functions)
m_logM = logM[matched_positions]
m_logR = logR[matched_positions]
m_compactness = compactness[matched_positions]
m_mgfe = mgfe[matched_positions]
m_age = age[matched_positions]
m_log_ssfr = log_ssfr[matched_positions]

# index CSV by subhalo_id for quick reindexing (ensure int index)
if "subhalo_id" not in df_ucmg.columns:
    raise SystemExit("CSV missing 'subhalo_id' column — cannot align matched rows.")
df_ucmg_indexed = df_ucmg.set_index("subhalo_id", drop=False)

# Helper to safely extract a column aligned to matched_subids; returns numpy float array
def col_aligned(colname):
    if colname not in df_ucmg_indexed.columns:
        return np.full(len(matched_subids), np.nan, dtype=float)
    # reindex in the same order as matched_subids; missing -> NaN
    s = df_ucmg_indexed.reindex(matched_subids)[colname]
    return s.to_numpy(dtype=float)

# Collect DoR variant column names that might exist in the CSV
possible_dor_cols = [
    "DoR_t90", "DoR_t95", "DoR_t998", "DoR_tfin", "DoR_tfin_existing",
    "dor", "DoR", "DoR_choice", "DoR_csv"
]
dor_variants_matched = {}
for col in possible_dor_cols:
    if col in df_ucmg_indexed.columns:
        arr = col_aligned(col)
        # normalize names for dictionary keys (so filenames are friendly)
        key = col
        # optionally rename DoR_tfin_existing -> DoR_tfin if you prefer:
        if key == "DoR_tfin_existing":
            key = "DoR_tfin"
        dor_variants_matched[key] = arr

# If no explicit DoR variants were present, fall back to the single loaded dor_lookup / matched_dor
if len(dor_variants_matched) == 0:
    dor_variants_matched["DoR_csv_dor"] = matched_dor.copy()

print("Found DoR variant columns (matched):", list(dor_variants_matched.keys()))

# Time-related columns to make plots for (only include those present in CSV)
time_cols_want = [
    "t50", "t50_span",
    "t75", "t75_span",
    "t90", "t90_span",
    "t95", "t95_span",
    "t998", "t998_span",
    "tfin", "tfin_span"
]
time_matched = {}
for col in time_cols_want:
    if col in df_ucmg_indexed.columns:
        time_matched[col] = col_aligned(col)

print("Found time columns (matched):", list(time_matched.keys()))
# ----------------------------------------------------------------------------------------------

# matched_exsitu: lookup by track id for all matched positions
matched_exsitu = np.full_like(matched_dor, np.nan, dtype=float)
for i, pos in enumerate(matched_positions):
    tid = None
    try:
        tid = int(track[pos])
    except Exception:
        tid = None
    if (tid is not None) and (tid in exsitu_lookup):
        matched_exsitu[i] = exsitu_lookup[tid]
    else:
        matched_exsitu[i] = np.nan

print(f"Matched UCMG CSV -> SOAP: {len(matched_positions)} / {len(ucmg_ids_set)} (present in SOAP selection)")

# # ------------------------ Produce both LOESS and scatter plots for DoR variants & time columns ----

# # Optional runtime control: if dataset is huge, LOESS will be subsampled internally.
# MAX_EVAL_PTS = 12000   # None => use all points; set lower to speed up LOESS

# # simple scatter-colour helper (fast)
# def scatter_coloured_mass_size(xvals, yvals, zvals, fname, cbar_label=None):
#     fig, ax = plt.subplots(figsize=(8,6))
#     # background full SOAP-selected
#     ax.scatter(logM, logR, s=6, color="lightgrey", alpha=0.5, label="all SOAP-selected")
#     # points to plot (show NaNs as light grey)
#     finite = np.isfinite(zvals)
#     if np.any(finite):
#         sc = ax.scatter(xvals[finite], yvals[finite], c=zvals[finite], cmap="viridis",
#                         s=18, edgecolors="none", alpha=0.9)
#         cbar = fig.colorbar(sc, ax=ax)
#         cbar.set_label(cbar_label if cbar_label is not None else fname)
#     # missing points shown faintly
#     if np.any(~finite):
#         ax.scatter(xvals[~finite], yvals[~finite], color="lightgrey", s=8, alpha=0.6, label="missing")
#     xm = np.linspace(np.nanmin(logM)-0.1, np.nanmax(logM)+0.1, 400)
#     yr = (xm - COMPACTNESS_CUT) / 1.5
#     ax.plot(xm, yr, "--", color="black", lw=2, label=f"compactness = {COMPACTNESS_CUT}")
#     ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
#     ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
#     ax.legend(fontsize=8)
#     ax.grid(True)
#     save_fig(fig, fname)

# # LOESS wrapper (same as earlier but exposes MAX_EVAL_PTS subsampling)
# def loess_coloured_mass_size(xvals, yvals, zvals, fname, cbar_label=None, nx=300, ny=220, pad_frac=0.05, max_eval_pts=MAX_EVAL_PTS):
#     # quick check
#     if np.sum(np.isfinite(zvals)) < 2:
#         # fallback to scatter (use same naming but indicate fallback)
#         print(f"LOESS: too few finite values for {fname}; saving scatter fallback.")
#         scatter_coloured_mass_size(xvals, yvals, zvals, fname.replace(".png", "_fallback_scatter.png"), cbar_label=cbar_label)
#         return

#     # subsample LOESS *inputs* if requested to limit time
#     finite_idx = np.where(np.isfinite(zvals))[0]
#     x_in = xvals[finite_idx].astype(float)
#     y_in = yvals[finite_idx].astype(float)
#     z_in = zvals[finite_idx].astype(float)

#     N = x_in.size
#     if (max_eval_pts is not None) and (N > int(max_eval_pts)):
#         rng = np.random.default_rng(seed=12345)
#         sel = rng.choice(N, size=int(max_eval_pts), replace=False)
#         x_loess = x_in[sel].copy()
#         y_loess = y_in[sel].copy()
#         z_loess = z_in[sel].copy()
#     else:
#         x_loess = x_in.copy(); y_loess = y_in.copy(); z_loess = z_in.copy()

#     # build tight grid around LOESS input
#     pad_x = pad_frac * (np.nanmax(x_loess) - np.nanmin(x_loess) + 1e-6)
#     pad_y = pad_frac * (np.nanmax(y_loess) - np.nanmin(y_loess) + 1e-6)
#     xg = np.linspace(np.nanmin(x_loess) - pad_x, np.nanmax(x_loess) + pad_x, nx)
#     yg = np.linspace(np.nanmin(y_loess) - pad_y, np.nanmax(y_loess) + pad_y, ny)
#     Xg, Yg = np.meshgrid(xg, yg)
#     pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

#     # KDTree mask to avoid extrapolation
#     tree_data = KDTree(np.column_stack((x_loess, y_loess)))
#     d_grid, _ = tree_data.query(pts_grid, k=1)
#     d_data, _ = tree_data.query(np.column_stack((x_loess, y_loess)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.nanmedian(d_grid))
#     d_thresh = max(typical_spacing * 1.3, 1e-6)
#     inside_mask = (d_grid <= d_thresh)
#     idx_inside = np.nonzero(inside_mask)[0]

#     Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
#     if idx_inside.size > 0:
#         xout = pts_grid[idx_inside, 0]
#         yout = pts_grid[idx_inside, 1]
#         frac_loess = 0.10
#         degree = 1
#         Zflat_inside, _ = loess_2d(x_loess, y_loess, z_loess, frac=frac_loess, degree=degree,
#                                    xout=xout, yout=yout)
#         Zflat[idx_inside] = Zflat_inside

#     Zgrid = Zflat.reshape((ny, nx))
#     Zmask = np.ma.masked_invalid(Zgrid)

#     # color limits robust from z_in
#     try:
#         vmin = float(np.nanpercentile(z_in, 5))
#         vmax = float(np.nanpercentile(z_in, 95))
#     except Exception:
#         vmin, vmax = float(np.nanmin(z_in)), float(np.nanmax(z_in))
#     if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#         med = float(np.nanmedian(z_in))
#         span = max(0.2, 0.5 * max(1e-6, abs(med)))
#         vmin = med - span
#         vmax = med + span

#     fig, ax = plt.subplots(figsize=(8,6))
#     ax.scatter(logM, logR, s=6, color="lightgrey", alpha=0.5, label="all SOAP-selected galaxies")
#     im = ax.pcolormesh(Xg, Yg, Zmask, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label(cbar_label if cbar_label is not None else fname)
#     # faint markers of LOESS-eval points
#     if idx_inside.size > 0:
#         ax.scatter(pts_grid[idx_inside,0], pts_grid[idx_inside,1], s=1, c='k', alpha=0.03, linewidths=0)
#     xm = np.linspace(np.nanmin(logM)-0.1, np.nanmax(logM)+0.1, 400)
#     yr = (xm - COMPACTNESS_CUT) / 1.5
#     ax.plot(xm, yr, "--", color="black", lw=2, label=f"compactness = {COMPACTNESS_CUT}")
#     ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
#     ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
#     ax.legend(fontsize=8)
#     ax.grid(True)
#     save_fig(fig, fname)

# # Make both types of plots for each DoR variant found
# for key, arr in dor_variants_matched.items():
#     # scatter plot (fast)
#     scatter_name = f"mass_size_matched_{key}_scatter.png"
#     try:
#         scatter_coloured_mass_size(m_logM, m_logR, arr, scatter_name, cbar_label=f"DoR ({key})")
#         print("Saved scatter DoR plot:", scatter_name)
#     except Exception as e:
#         print("Scatter failed for", key, ":", e)
#     # LOESS plot (slower)
#     loess_name = f"mass_size_matched_{key}_loess.png"
#     try:
#         loess_coloured_mass_size(m_logM, m_logR, arr, loess_name, cbar_label=f"DoR ({key})")
#         print("Saved LOESS DoR plot:", loess_name)
#     except Exception as e:
#         print("LOESS failed for", key, ":", e)

# # Make both types of plots for each time column found
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

# # ------------------------------------------------------------------------------------------

if len(matched_positions) == 0:
    raise SystemExit("No UCMG subhalo_ids from CSV matched SOAP-selected arrays. Abort.")

# # aligned arrays for plotting
# m_logM = logM[matched_positions]
# m_logR = logR[matched_positions]
# m_compactness = compactness[matched_positions]
# m_mgfe = mgfe[matched_positions]
# m_age = age[matched_positions]
# m_log_ssfr = log_ssfr[matched_positions]

# DoR sanity check and clip to [0,1] but report raw range first
finite_dor = matched_dor[np.isfinite(matched_dor)]
if finite_dor.size > 0:
    raw_min, raw_max = float(np.nanmin(finite_dor)), float(np.nanmax(finite_dor))
    print(f"Raw matched DoR range: {raw_min:.6g} – {raw_max:.6g}")
    if raw_min < 0 or raw_max > 1:
        print("Warning: some DoR values outside [0,1]. Clipping to [0,1]. Investigate upstream.")
matched_dor = np.where(np.isfinite(matched_dor), np.clip(matched_dor, 0.0, 1.0), np.nan)

# ------------------------ SAVE matched diagnostic CSV (SAFE, new file) ------------------------
diag_fn = os.path.join(outdir, "soap_ucmg_matched_summary.csv")
df_diag = pd.DataFrame({
    "subhalo_id": matched_subids,
    "DoR": matched_dor,
    "logM": m_logM,
    "logR": m_logR,
    "compactness": m_compactness,
    "MgFe": m_mgfe,
    "lum_age_gyr": m_age,
    "log_ssfr": m_log_ssfr,
    "exsitu_frac": matched_exsitu
})
df_diag.to_csv(diag_fn, index=False)
print("Wrote diagnostic matched table:", diag_fn)

# ------------------------ PLOTTING HELPERS ------------------------
def save_fig(fig, fname):
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("Saved:", path)
    plt.close(fig)

def plot_dor_vs_quantity(x, y, xlabel, fname):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 6:
        print(f"Skipping {fname} — insufficient matched finite points ({ok.sum()}).")
        return
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(x[ok], y[ok], s=12, alpha=0.7)
    # binned median
    q = np.linspace(0, 100, 15)
    bins = np.percentile(x[ok], q)
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
        ax.fill_between(xc, lo, hi, color="black", alpha=0.2)
    ax.axhline(EXTREME_DOR, color='C1', linestyle='--', lw=1.5, label=f"extreme threshold DoR={EXTREME_DOR}")
    plt.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("DoR")
    ax.set_ylim(0, 1)
    ax.grid(True)
    save_fig(fig, fname)

# ------------------------ PLOTS ------------------------
# mass-size plane: all SOAP-selected (grey) and UCMGs (DoR)
fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(logM, logR, s=6, color="lightgrey", alpha=0.5, label=r"all galaxies at $z=0$")
sc = ax.scatter(m_logM, m_logR, c=matched_dor, cmap="viridis", s=18, edgecolors="none", label="UCMGs")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("DoR")
xm = np.linspace(np.nanmin(logM)-0.1, np.nanmax(logM)+0.1, 400)
yr = (xm - COMPACTNESS_CUT) / 1.5
ax.plot(xm, yr, "--", color="black", lw=2, label=f"$\lg \Sigma_{{1.5}} = {COMPACTNESS_CUT}$")
ax.set_xlabel("lg(Total Stellar Mass / M⊙)")
ax.set_ylabel("lg(Half Mass Radius / kpc)")
ax.legend()
ax.grid(True)
save_fig(fig, "mass_size_DoR.png")

# DoR vs compactness, Mg/Fe, age, log sSFR, ex-situ
plot_dor_vs_quantity(m_compactness, matched_dor, "Compactness (logM - 1.5 logR)", "DoR_vs_compactness.png")
plot_dor_vs_quantity(m_mgfe, matched_dor, "[Mg/Fe] (dex, SOAP-derived)", "DoR_vs_MgFe.png")
plot_dor_vs_quantity(m_age, matched_dor, "Lum-weighted age (Gyr, SOAP)", "DoR_vs_age.png")
plot_dor_vs_quantity(m_log_ssfr, matched_dor, "lg(sSFR / yr⁻¹) (SOAP)", "DoR_vs_sSFR.png")
# ex-situ vs DoR (only where ex-situ present)
plot_dor_vs_quantity(matched_exsitu, matched_dor, "Ex-situ mass fraction", "DoR_vs_exsitu.png")

print("Done. Plots and diagnostics in:", outdir)

# ---- LOESS helpers (copy from full MgFe working script) ----
def polyfit_2d(x, y, z, degree=1, weights=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    if weights is None:
        W = np.ones_like(z, dtype=float)
    else:
        W = np.asarray(weights).ravel()
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
    elif degree == 1:
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
    else:
        raise NotImplementedError("Only degree 0 or 1 supported")

def _biweight_scale(resid):
    resid = np.abs(resid)
    if resid.size == 0:
        return 1.0
    mad = np.median(resid)
    if mad <= 0:
        return 1e-9
    return 1.4826 * mad

def loess_2d(x1, y1, z, frac=0.5, degree=1, rescale=False, npoints=None, sigz=None,
             xout=None, yout=None):
    x1 = np.asarray(x1).ravel()
    y1 = np.asarray(y1).ravel()
    z = np.asarray(z).ravel()
    if not (x1.size == y1.size == z.size):
        raise ValueError("Input vectors (X, Y, Z) must have the same size")
    n = x1.size
    if n == 0:
        return np.array([]), np.array([])
    if npoints is None:
        npoints = int(np.ceil(frac * n))
    npoints = max(2, min(npoints, n))
    if xout is None or yout is None:
        xout = x1.copy()
        yout = y1.copy()
    else:
        xout = np.asarray(xout).ravel()
        yout = np.asarray(yout).ravel()
        if xout.size != yout.size:
            raise ValueError("xout and yout must have same length")
    m = xout.size
    zout = np.empty(m, dtype=float)
    wout = np.empty(m, dtype=float)
    tree = KDTree(np.column_stack((x1, y1)))
    for j, (xx, yy) in enumerate(zip(xout, yout)):
        dists, inds = tree.query([xx, yy], k=npoints)
        if np.isscalar(dists):
            dists = np.array([dists])
            inds = np.array([inds])
        rmax = np.max(dists)
        if rmax == 0:
            zout[j] = z[inds[0]]
            wout[j] = 1.0
            continue
        u = dists / rmax
        distWeights = (1.0 - u**3)**3
        distWeights = np.where(u >= 1.0, 0.0, distWeights)
        xw = x1[inds]
        yw = y1[inds]
        zw = z[inds]
        w_init = distWeights.copy()
        coeffs = polyfit_2d(xw, yw, zw, degree=degree, weights=w_init)
        if degree == 0:
            zfit = np.full_like(zw, coeffs[0], dtype=float)
        else:
            xc = np.average(xw, weights=w_init)
            yc = np.average(yw, weights=w_init)
            dx = xw - xc
            dy = yw - yc
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
                dx = xw - xc
                dy = yw - yc
                a0, ax, ay = coeffs
                zfit = a0 + ax * dx + ay * dy
            if np.allclose(biWeights, biWeights_new, atol=1e-6):
                biWeights = biWeights_new
                break
            biWeights = biWeights_new
        if degree == 0:
            zout[j] = coeffs[0]
        else:
            zout[j] = coeffs[0]
        wout[j] = biWeights[0] if biWeights.size > 0 else 1.0
    return zout, wout

# ---- LOESS plot for DoR on the mass-size plane (applies to matched UCMGs) ----
# use only matched UCMG points with finite DoR for smoothing
zvals = matched_dor.copy()
xvals = m_logM.copy()
yvals = m_logR.copy()

# mask finite
have_mask = np.isfinite(zvals)
missing_mask = ~have_mask
n_have = int(have_mask.sum())
n_missing = int(missing_mask.sum())
print(f"LOESS DoR: have={n_have}, missing={n_missing}, total={len(zvals)}")

fig, ax = plt.subplots(figsize=(8,6))
if n_have == 0:
    # nothing to LOESS -> fallback scatter
    ax.scatter(logM, logR, s=6, color="lightgrey", alpha=0.5, label="all SOAP-selected")
    if len(xvals)>0:
        ax.scatter(xvals, yvals, s=12, color="lightgrey", alpha=0.8, label="UCMGs (no DoR)")
else:
    # restrict LOESS inputs to finite subset
    x_lo = xvals[have_mask]
    y_lo = yvals[have_mask]
    z_lo = zvals[have_mask]

    # build a tight grid around LOESS input points (avoid extrapolation)
    pad_x = 0.05 * (np.nanmax(x_lo) - np.nanmin(x_lo) + 1e-6)
    pad_y = 0.05 * (np.nanmax(y_lo) - np.nanmin(y_lo) + 1e-6)
    nx, ny = 300, 220
    xg = np.linspace(np.nanmin(x_lo) - pad_x, np.nanmax(x_lo) + pad_x, nx)
    yg = np.linspace(np.nanmin(y_lo) - pad_y, np.nanmax(y_lo) + pad_y, ny)
    Xg, Yg = np.meshgrid(xg, yg)
    pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

    # KDTree on LOESS points to compute distance mask
    tree_data = KDTree(np.column_stack((x_lo, y_lo)))
    d_grid, _ = tree_data.query(pts_grid, k=1)
    d_data, _ = tree_data.query(np.column_stack((x_lo, y_lo)), k=2)
    if d_data.ndim == 2 and d_data.shape[1] >= 2:
        typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
    else:
        typical_spacing = float(np.nanmedian(d_grid))
    d_thresh = max(typical_spacing * 1.3, 1e-6)
    inside_mask = (d_grid <= d_thresh)
    idx_inside = np.nonzero(inside_mask)[0]

    # overlay all SOAP-selected lightgrey background for context
    ax.scatter(logM, logR, s=6, color="lightgrey", alpha=0.5, label="all SOAP-selected galaxies")

    if idx_inside.size > 0:
        xout = pts_grid[idx_inside, 0]
        yout = pts_grid[idx_inside, 1]

        # LOESS parameters (same as your script)
        frac_loess = 0.10
        degree = 1

        Zflat_inside, Wflat = loess_2d(x_lo, y_lo, z_lo, frac=frac_loess, degree=degree,
                                       xout=xout, yout=yout)

        # place predictions back into full grid and mask invalids
        Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
        Zflat[idx_inside] = Zflat_inside
        Zgrid = Zflat.reshape((ny, nx))
        Zmask = np.ma.masked_invalid(Zgrid)

        # color limits robustly from z_lo
        try:
            vmin = float(np.nanpercentile(z_lo, 5))
            vmax = float(np.nanpercentile(z_lo, 95))
        except Exception:
            vmin, vmax = float(np.nanmin(z_lo)), float(np.nanmax(z_lo))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            med = float(np.nanmedian(z_lo))
            span = max(0.2, 0.5 * max(1e-6, abs(med)))
            vmin = med - span
            vmax = med + span

        cmap = plt.get_cmap("viridis")
        im = ax.pcolormesh(Xg, Yg, Zmask, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("DoR (LOESS)")
        # faint markers showing evaluated LOESS points (optional)
        ax.scatter(xout, yout, s=1, c='k', alpha=0.03, linewidths=0)
    else:
        # fallback scatter of available points
        ax.scatter(x_lo, y_lo, c=z_lo, cmap='viridis', s=12, edgecolors='none')

    # optionally overlay missing matched UCMGs as light grey
    if n_missing > 0:
        ax.scatter(xvals[missing_mask], yvals[missing_mask], color="lightgrey", s=8, alpha=0.6, label="no DoR")


# compactness threshold line
xm = np.linspace(np.nanmin(logM)-0.1, np.nanmax(logM)+0.1, 400)
yr = (xm - COMPACTNESS_CUT) / 1.5
ax.plot(xm, yr, "--", color="black", lw=2, label=f"compactness = {COMPACTNESS_CUT}")

ax.set_xlabel("lg(Stellar Mass / M⊙)")
ax.set_ylabel("lg(Half Mass Radius / kpc)")
ax.set_title(f"Mass–size plane coloured by DoR (LOESS) z={ztarget}")
ax.legend(fontsize=8)
ax.grid(True)

outpath_loess = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_DoR_loess.png")
fig.savefig(outpath_loess, dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved LOESS DoR mass-size:", outpath_loess)


# ------------------------ BINNED BY STELLAR MASS (0.2 dex) ------------------------
# Create 0.2 dex bins in stellar mass (log10(M⋆))
bin_width = 0.2
min_mass = np.nanmin(m_logM) if np.isfinite(np.nanmin(m_logM)) else 9.0
max_mass = np.nanmax(m_logM) if np.isfinite(np.nanmax(m_logM)) else 12.5

# floor/ceil so bins align nicely
bin_start = math.floor(min_mass / bin_width) * bin_width
bin_end = math.ceil(max_mass / bin_width) * bin_width
bins = np.arange(bin_start, bin_end + 1e-9, bin_width)  # edges
nbins = len(bins) - 1

print(f"Creating {nbins} mass bins from {bin_start:.2f} to {bin_end:.2f} (width={bin_width} dex)")

# Prepare arrays aligned to matched UCMGs
# m_logM, m_logR, m_compactness, m_mgfe, m_age, m_log_ssfr, matched_exsitu, matched_dor already exist
# matched_subids gives subhalo ids for matched entries

# create a folder for bin-specific plots
bin_outdir = os.path.join(outdir, "by_mass_bin")
os.makedirs(bin_outdir, exist_ok=True)

# # threshold for 'extreme relics'
# EXTREME_DOR = 0.7

# we'll store summary statistics (median DoR per mass bin)
bin_centers = []
bin_med = []
bin_p16 = []
bin_p84 = []
bin_counts = []
bin_extreme_counts = []   # N(DoR > EXTREME_DOR) per mass bin
bin_extreme_frac = []     # fraction of bin that is extreme

for ib in range(nbins):
    lo = bins[ib]
    hi = bins[ib+1]
    sel = (m_logM >= lo) & (m_logM < hi) & np.isfinite(matched_dor)
    count = int(np.sum(sel))

    if count == 0:
        print(f"Bin {ib:02d} [{lo:.2f},{hi:.2f}): empty -> skipping")
        continue

    # record summary stats
    dor_sel = matched_dor[sel]
    med = float(np.nanmedian(dor_sel))
    p16 = float(np.nanpercentile(dor_sel, 16))
    p84 = float(np.nanpercentile(dor_sel, 84))
    center = 0.5 * (lo + hi)
    bin_centers.append(center)
    bin_med.append(med)
    bin_p16.append(p16)
    bin_p84.append(p84)
    bin_counts.append(count)

    # count extreme relics in this bin
    extreme_count = int(np.sum(dor_sel > EXTREME_DOR))
    extreme_frac = extreme_count / float(count) if count > 0 else 0.0
    bin_extreme_counts.append(extreme_count)
    bin_extreme_frac.append(extreme_frac)

    print(f"Bin {ib:02d} [{lo:.2f},{hi:.2f}) count={count} median_DoR={med:.3f}")

    # make a safe label suffix for filenames
    suf = f"mass_{lo:.2f}_{hi:.2f}".replace(".", "p").replace("-", "m")

    # 1) mass-size plane for this bin: UCMGs in the bin coloured by DoR, background all SOAP-selected
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(logM, logR, s=6, color="lightgrey", alpha=0.5, label="all SOAP-selected galaxies")
    # plot all matched UCMGs in grey for context
    ax.scatter(m_logM, m_logR, s=6, color="lightgrey", alpha=0.6)
    sc = ax.scatter(m_logM[sel], m_logR[sel], c=matched_dor[sel], cmap="viridis", s=24, edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("DoR")
    xm = np.linspace(np.nanmin(logM)-0.1, np.nanmax(logM)+0.1, 400)
    yr = (xm - COMPACTNESS_CUT) / 1.5
    ax.plot(xm, yr, "--", color="black", lw=2, label=f"compactness = {COMPACTNESS_CUT}")
    ax.set_xlabel("lg(Total Stellar Mass / M⊙)")
    ax.set_ylabel("lg(Half Mass Radius / kpc)")
    ax.set_title(f"Mass-size (DoR) — mass bin [{lo:.2f},{hi:.2f})")
    ax.legend(fontsize=8)
    ax.grid(True)
    fname = os.path.join(bin_outdir, f"mass_size_DoR_bin_{suf}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved:", fname)

    # 2) For each quantity, call your existing helper but restricted to this bin
    # We'll reuse plot_dor_vs_quantity but pass subset arrays. Filenames include bin suffix.
    # Compactness
    plot_dor_vs_quantity(m_compactness[sel], matched_dor[sel],
                         f"Compactness (logM - 1.5 logR)  [{lo:.2f},{hi:.2f})",
                         f"DoR_vs_compactness_bin_{suf}.png")
    # Mg/Fe
    plot_dor_vs_quantity(m_mgfe[sel], matched_dor[sel],
                         f"[Mg/Fe] (dex, SOAP-derived)  [{lo:.2f},{hi:.2f})",
                         f"DoR_vs_MgFe_bin_{suf}.png")
    # Lum-weighted age
    plot_dor_vs_quantity(m_age[sel], matched_dor[sel],
                         f"Lum-weighted age (Gyr, SOAP)  [{lo:.2f},{hi:.2f})",
                         f"DoR_vs_age_bin_{suf}.png")
    # log sSFR
    plot_dor_vs_quantity(m_log_ssfr[sel], matched_dor[sel],
                         f"lg(sSFR / yr⁻¹) (SOAP)  [{lo:.2f},{hi:.2f})",
                         f"DoR_vs_sSFR_bin_{suf}.png")
    # ex-situ
    plot_dor_vs_quantity(matched_exsitu[sel], matched_dor[sel],
                         f"Ex-situ mass fraction  [{lo:.2f},{hi:.2f})",
                         f"DoR_vs_exsitu_bin_{suf}.png")

# After looping, produce a summary plot: median DoR vs mass bin center (with 16/84)
if len(bin_centers) > 0:
    bin_centers = np.array(bin_centers)
    bin_med = np.array(bin_med)
    bin_p16 = np.array(bin_p16)
    bin_p84 = np.array(bin_p84)
    bin_counts = np.array(bin_counts)
    bin_extreme_counts = np.array(bin_extreme_counts)
    bin_extreme_frac = np.array(bin_extreme_frac)

    fig, ax = plt.subplots(figsize=(8,5))

    # Left axis: median DoR with 16/84
    ax.errorbar(bin_centers, bin_med, yerr=[bin_med - bin_p16, bin_p84 - bin_med],
                fmt='o-', capsize=3, lw=1.5, label='center of mass bin')
    ax.set_xlabel("lg(Stellar Mass / M⊙)")
    ax.set_ylabel("Median DoR")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    # annotate median counts above points (optional)
    for x, y, cnt in zip(bin_centers, bin_med, bin_counts):
        ax.text(x, y + 0.04, f"{int(cnt)}", ha='center', fontsize=8, alpha=0.7)

    # Right axis: extreme relics count (DoR > EXTREME_DOR)
    ax2 = ax.twinx()
    ax2.plot(bin_centers, bin_extreme_counts, 's--', ms=6, color='C1', label=f'DoR > {EXTREME_DOR}')
    ax2.set_ylabel(f'Number of extreme relics')
    # optionally set sensible ylim for counts
    maxcnt = int(np.nanmax(bin_extreme_counts)) if bin_extreme_counts.size > 0 else 1
    ax2.set_ylim(0, max(3, maxcnt * 1.15))

    # If you prefer fraction instead of raw counts, uncomment the following and comment the ax2.plot above:
    # ax2.plot(bin_centers, bin_extreme_frac, 's--', ms=6, color='C1', label=f'Fraction (DoR > {EXTREME_DOR})')
    # ax2.set_ylabel(f'Fraction with DoR > {EXTREME_DOR}')
    # ax2.set_ylim(0, 1.05)

    # legends: combine handles from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=9)

    # optional: annotate extreme counts above the right-axis line
    for x, cnt in zip(bin_centers, bin_extreme_counts):
        ax2.text(x, cnt + max(1, 0.03 * maxcnt), f"{int(cnt)}", ha='center', va='bottom', fontsize=7, color='C1', alpha=0.8)

    out_sum = os.path.join(bin_outdir, "DoR_median_vs_mass_bin_with_extremes.png")
    fig.savefig(out_sum, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved summary median DoR vs mass bin (with extreme counts):", out_sum)
else:
    print("No mass-bin statistics produced (no matched UCMGs?).")