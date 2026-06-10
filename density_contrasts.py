#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from scipy.spatial import cKDTree
import cmasher as cmr
import common

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12
})

# ---------------- CONFIG ----------------
csv_in = "sfh_times_all_with_DoR_variants_corrected.csv.gz"
exsitu_h5 = "/mnt/su3ctm/kproctor/ForMax/exsitu_summary_SnapNum_127.hdf5"  # optional
model_name = "L0200N3008/THERMAL_AGN/"
model_dir = "/mnt/su3-pro/colibre/" + model_name
snap_file = "0127"   # z = 0
outdir = "density_contrast_env"
os.makedirs(outdir, exist_ok=True)

EXTREME_DOR = 0.6
MIN_STELLAR_MASS = 1e9

# If the box size cannot be inferred automatically, set it here in kpc physical.
BOXSIZE_P_KPC = None

# ---------------- HELPERS ----------------
def save_fig(fig, fname):
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def compute_density_contrast(soap_hc_p, soap_hc_pop_p, mtot, boxsize_p, izarray):
    """
    Compute subhalo density contrast for given galaxy positions.
    soap_hc_p:     array of shape (N, 3) with physical positions of all galaxies in kpc
    soap_hc_pop_p: array of shape (M, 3) with physical positions of a population in kpc
    mtot:          array of shape (N,) with mass weights in Msun
    boxsize_p:     box size in physical kpc
    Returns: dcont_pop array of shape (M, 3)
    """

    # convert everything to comoving units
    boxsize_c = boxsize_p * (1 + izarray)
    soap_hc_c = np.asarray(soap_hc_p, dtype=float) * (1 + izarray)
    soap_hc_pop_c = np.asarray(soap_hc_pop_p, dtype=float) * (1 + izarray)
    mtot = np.asarray(mtot, dtype=float)

    # periodic wrapping
    xcentre = soap_hc_c[:, 0]
    xcentre[xcentre > boxsize_c] = xcentre[xcentre > boxsize_c] - boxsize_c
    ycentre = soap_hc_c[:, 1]
    ycentre[ycentre > boxsize_c] = ycentre[ycentre > boxsize_c] - boxsize_c
    zcentre = soap_hc_c[:, 2]
    zcentre[zcentre > boxsize_c] = zcentre[zcentre > boxsize_c] - boxsize_c
    positions_c = np.vstack([xcentre, ycentre, zcentre]).T

    xcentre = soap_hc_pop_c[:, 0]
    xcentre[xcentre > boxsize_c] = xcentre[xcentre > boxsize_c] - boxsize_c
    ycentre = soap_hc_pop_c[:, 1]
    ycentre[ycentre > boxsize_c] = ycentre[ycentre > boxsize_c] - boxsize_c
    zcentre = soap_hc_pop_c[:, 2]
    zcentre[zcentre > boxsize_c] = zcentre[zcentre > boxsize_c] - boxsize_c
    positions_pop_c = np.vstack([xcentre, ycentre, zcentre]).T

    tree = cKDTree(positions_c, boxsize=boxsize_c)

    # three apertures: 0.3, 1, 3 Mpc comoving
    apertures = 1e3 * np.array([0.3, 1.0, 3.0])  # kpc comoving
    dcont_pop = np.full((len(soap_hc_pop_c), len(apertures)), np.nan)

    # mean mass density of the full reference population
    volume_box = boxsize_c ** 3
    mdcont = np.sum(mtot) / volume_box

    for k, aperture in enumerate(apertures):
        volume_neigh = 4.0 / 3.0 * np.pi * (aperture ** 3)
        neighbors_pop_c = tree.query_ball_point(positions_pop_c, r=aperture)

        for l, inds in enumerate(neighbors_pop_c):
            mtot_neigh = mtot[inds]
            dcont_pop[l, k] = ((np.sum(mtot_neigh) / volume_neigh) - mdcont) / mdcont

    return dcont_pop, apertures

# ---------------- READ CSV ----------------
if not os.path.exists(csv_in):
    raise SystemExit(f"CSV not found: {csv_in}")

print("Reading CSV:", csv_in)
df_ucmg = pd.read_csv(csv_in, low_memory=False)

id_col = None
for c in ("subhalo_id", "HaloCatalogueIndex", "subhaloId", "HaloIndex", "track_id", "TrackId"):
    if c in df_ucmg.columns:
        id_col = c
        break
if id_col is None:
    id_col = df_ucmg.columns[0]
    print("Warning: no ID column found; using", id_col)

s = df_ucmg[id_col].astype(str).str.replace("\r", "").str.strip()
df_ucmg["_subhalo_id_numeric"] = pd.to_numeric(s, errors="coerce").astype("Int64")
df_ucmg = df_ucmg[df_ucmg["_subhalo_id_numeric"].notna()].copy()
df_ucmg["subhalo_id"] = df_ucmg["_subhalo_id_numeric"].astype("int64")
df_ucmg.drop(columns=["_subhalo_id_numeric"], inplace=True)

primary_dor_col = None
for c in ("DoR_t95", "DoR_t90", "DoR_t998", "DoR_tfin", "DoR", "dor"):
    if c in df_ucmg.columns:
        primary_dor_col = c
        break
if primary_dor_col is None:
    for c in df_ucmg.columns:
        if c.lower().startswith("dor"):
            primary_dor_col = c
            break
if primary_dor_col is None:
    raise SystemExit("No DoR-like column found in CSV.")

print("Using DoR column:", primary_dor_col)

dor_lookup = {}
for _, row in df_ucmg.iterrows():
    try:
        sid = int(row["subhalo_id"])
        v = row.get(primary_dor_col, np.nan)
        if pd.isna(v):
            continue
        dor_lookup[sid] = float(v)
    except Exception:
        continue
print("Loaded DoR entries:", len(dor_lookup))

# ---------------- READ SOAP ----------------
print("Reading SOAP arrays via common.read_group_data_colibre...")

fields_gal = {
    "ExclusiveSphere/50kpc": (
        "StellarMass",
        "TotalMass",
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
fields_id = {
    "InputHalos": ("HaloCatalogueIndex", "IsCentral", "HBTplus/DescendantTrackId", "HBTplus/TrackId")
}

h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields_gal)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_id)

(m30, mtot, sfr30, r50, centre_of_mass, stellarage, stellarage_lum, Fe_lin, Mg_lin, Zstar_raw) = h5data_groups
(halo_index, is_central, desc_id, track_id) = h5data_idgroups

ztarget = 0.0
comov_to_physical_length = 1.0 / (1.0 + ztarget)
Mu = 1.988e43 / 1.989e33
tu = 3.086e19 / 3.154e7

m30 = np.asarray(m30).ravel() * Mu
mtot = np.asarray(mtot).ravel() * Mu
sfr30 = np.asarray(sfr30).ravel() * Mu / tu
r50 = np.asarray(r50).ravel() * comov_to_physical_length * 1e3
centre_of_mass = np.asarray(centre_of_mass) * comov_to_physical_length * 1e3
stellarage_lum = stellarage_lum * tu / 1e9
halo_index = np.asarray(halo_index).ravel().astype(np.int64)
is_central = np.asarray(is_central).ravel().astype(bool)
track_id = np.asarray(track_id).ravel().astype(np.int64)

Zsun = 0.0134   # AGSS09 convention
# Zsun = 0.0139 # Asplund et al. 2021 present-day photospheric value
    
Zstar = np.asarray(Zstar_raw, dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    logZstar = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar), np.nan)
    logZstar_rel = np.where((Zstar > 0) & np.isfinite(Zstar),
                            np.log10(Zstar / Zsun),
                            np.nan)

# Full SOAP population reference
sel = np.where(m30 >= MIN_STELLAR_MASS)[0]
if sel.size == 0:
    raise SystemExit("No galaxies meet MIN_STELLAR_MASS selection.")

m = m30[sel]
mtot_all = mtot[sel]
r = r50[sel]
stellarage_lum = stellarage_lum[sel]
Zstar = Zstar[sel]
logZstar = logZstar[sel]
logZstar_rel = logZstar_rel[sel]
soap_hc_all = centre_of_mass[sel]
halo_idx = halo_index[sel]
is_central_sel = is_central[sel]
track_sel = track_id[sel]

# ------------------------ LOAD ex-situ summary (optional) ------------------------
exsitu_lookup = {}
if os.path.exists(exsitu_h5):
    try:
        with h5py.File(exsitu_h5, "r") as fh:
            if "stars" in fh:
                data = np.array(fh["stars"])
                print("exsitu raw shape:", data.shape)
                print("first 5 rows:\n", data[:5, :])
                print("finite in col 1:", np.isfinite(data[:, 1]).sum())
                print("finite in col 2:", np.isfinite(data[:, 2]).sum())
                print("col 0 min/max:", np.nanmin(data[:, 0]), np.nanmax(data[:, 0]))
                print("col 3 min/max:", np.nanmin(data[:, 3]), np.nanmax(data[:, 3]))
                for keycol in (0, 1, 2):
                    ids = data[:, keycol].astype(np.int64)
                    overlap_halo = np.intersect1d(ids, halo_idx).size
                    overlap_track = np.intersect1d(ids, track_id).size
                    print(
                        f"candidate key column {keycol}: "
                        f"overlap with HaloCatalogueIndex = {overlap_halo}, "
                        f"overlap with track_id = {overlap_track}"
                    )
                ids = data[:, 0].astype(np.int64)   # or whichever column has the best overlap with halo_idx
                exfrac = data[:, 3].astype(float)
                exsitu_lookup = dict(zip(ids.tolist(), exfrac.tolist()))
                print(f"Loaded {len(exsitu_lookup)} ex-situ entries from {exsitu_h5} (dataset 'stars').")
            else:
                for k in fh:
                    try:
                        arr = np.array(fh[k])
                        if arr.ndim == 2 and arr.shape[1] >= 4:
                            ids = arr[:,0].astype(int); exfrac = arr[:,3].astype(float)
                            exsitu_lookup = dict(zip(ids.tolist(), exfrac.tolist()))
                            print(f"Loaded {len(exsitu_lookup)} ex-situ entries from {exsitu_h5} (dataset '{k}').")
                            break
                    except Exception:
                        continue
    except Exception as e:
        print("Warning: failed to read ex-situ HDF5:", e)
else:
    print("Ex-situ summary HDF5 not found; skipping ex-situ matching.")

# ---------------- MATCH DoR TO SOAP ----------------
dor_series = pd.Series(dor_lookup, dtype=float)
dor_for_each = dor_series.reindex(halo_idx).to_numpy(dtype=float)

if not np.any(np.isfinite(dor_for_each)):
    dor_try = dor_series.reindex(halo_idx - 1).to_numpy(dtype=float)
    if np.any(np.isfinite(dor_try)):
        dor_for_each = dor_try
    else:
        dor_try = dor_series.reindex(halo_idx + 1).to_numpy(dtype=float)
        if np.any(np.isfinite(dor_try)):
            dor_for_each = dor_try

matched_positions = np.where(np.isfinite(dor_for_each))[0]
print(f"Matched DoR entries for selected SOAP rows: {matched_positions.size} / {len(halo_idx)}")

m_matched = m[matched_positions]
r_matched = r[matched_positions]
stellarage_lum_matched = stellarage_lum[matched_positions]
logZstar_rel_matched = logZstar_rel[matched_positions]
hc_matched = soap_hc_all[matched_positions]
dor_matched = dor_for_each[matched_positions]
is_central_matched = is_central_sel[matched_positions]
track_matched = track_sel[matched_positions]

# matched ex-situ
exsitu_series = pd.Series(exsitu_lookup, dtype=float)
matched_exsitu = exsitu_series.reindex(halo_idx.astype(np.int64)).to_numpy(dtype=float)

# # <<< INSERT THESE MINIMAL LINES HERE >>>
# # raw BH mass from SOAP aligned to matched subset
# bh_mass_matched = bh_mass[matched_positions]

# # is_central aligned to matched subset (bool)
# is_central_matched = np.asarray(is_central_selected[matched_positions]).astype(bool)

# ex-situ already computed as 'matched_exsitu' — create a convenient alias
exsitu_fracs_matched = matched_exsitu.copy()

mask_relic = np.isfinite(dor_matched) & (dor_matched > EXTREME_DOR)
mask_control = np.isfinite(dor_matched) & (dor_matched <= EXTREME_DOR)

print("Matched objects:", len(dor_matched))
print("Relics:", int(mask_relic.sum()))
print("Non-relic control:", int(mask_control.sum()))

# ---------------- BOX SIZE ----------------
BOXSIZE_P_KPC = 200_000.0 / (1 + ztarget)


# ---------------- COMPUTE DENSITY CONTRAST ----------------
# relics and non-relics
dcont_relic, apertures = compute_density_contrast(
    soap_hc_all, hc_matched[mask_relic], mtot_all, BOXSIZE_P_KPC, ztarget
)
dcont_control, _ = compute_density_contrast(
    soap_hc_all, hc_matched[mask_control], mtot_all, BOXSIZE_P_KPC, ztarget
)
# full population
# dcont_all = np.vstack((dcont_relic, dcont_control))
dcont_all, apertures = compute_density_contrast(
    soap_hc_all, soap_hc_all, mtot_all, BOXSIZE_P_KPC, ztarget
)

# Per-object table
df_env = pd.DataFrame({
    "track_id": track_matched,
    "halo_index": halo_idx[matched_positions],
    "DoR": dor_matched,
    "is_relic": mask_relic,
    "is_central": is_central_matched,
})

for j, ap in enumerate(apertures):
    df_env[f"density_contrast_{int(ap)}kpc_relic"] = np.nan
    df_env[f"density_contrast_{int(ap)}kpc_control"] = np.nan

df_env.loc[mask_relic, [f"density_contrast_{int(ap)}kpc_relic" for ap in apertures]] = dcont_relic
df_env.loc[mask_control, [f"density_contrast_{int(ap)}kpc_control" for ap in apertures]] = dcont_control

csv_out = os.path.join(outdir, "density_contrast_relic_vs_nonrelic.csv")
df_env.to_csv(csv_out, index=False)
print("Wrote:", csv_out)

# ---------------- PLOT 1: THREE APERTURE PANELS ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for j, ap in enumerate(apertures):
    ax = axes[j]
    rel = dcont_relic[:, j]
    ctl = dcont_control[:, j]

    rel = rel[np.isfinite(rel)]
    ctl = ctl[np.isfinite(ctl)]

    if rel.size == 0 and ctl.size == 0:
        ax.set_title(f"{ap/1000:.1f} Mpc")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        continue

    allv = np.concatenate([rel, ctl]) if (rel.size and ctl.size) else (rel if rel.size else ctl)
    lo, hi = np.nanpercentile(allv, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1, 1

    bins = np.linspace(lo, hi, 40)

    if ctl.size > 0:
        ax.hist(ctl, bins=bins, histtype="step", density=True, lw=2, label=f"non-relic (N={ctl.size})")
        ax.axvline(np.nanmedian(ctl), color="C0", ls="--", lw=1)

    if rel.size > 0:
        ax.hist(rel, bins=bins, histtype="step", density=True, lw=2, label=f"relic (N={rel.size})")
        ax.axvline(np.nanmedian(rel), color="C1", ls="--", lw=1)

    ax.axvline(0.0, color="k", lw=1, alpha=0.5)
    ax.set_title(f"Aperture: {ap/1000:.1f} Mpc")
    ax.set_xlabel(r"$\delta = (\rho - \bar{\rho}) / \bar{\rho}$")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Probability density")
axes[0].legend(fontsize=8)

fig.subplots_adjust(left=0.06, right=0.92, bottom=0.08, top=0.94, wspace=0.12, hspace=0.22)
save_fig(fig, "density_contrast_relic_vs_nonrelic_three_apertures.png")

# ---------------- PLOT 2: MEDIANS VS APERTURE ----------------
rel_med = np.nanmedian(dcont_relic, axis=0) if dcont_relic.size else np.full(3, np.nan)
ctl_med = np.nanmedian(dcont_control, axis=0) if dcont_control.size else np.full(3, np.nan)

rel_p16 = np.nanpercentile(dcont_relic, 16, axis=0) if dcont_relic.size else np.full(3, np.nan)
rel_p84 = np.nanpercentile(dcont_relic, 84, axis=0) if dcont_relic.size else np.full(3, np.nan)
ctl_p16 = np.nanpercentile(dcont_control, 16, axis=0) if dcont_control.size else np.full(3, np.nan)
ctl_p84 = np.nanpercentile(dcont_control, 84, axis=0) if dcont_control.size else np.full(3, np.nan)

fig, ax = plt.subplots(figsize=(7, 5))
x = apertures / 1000.0

ax.errorbar(x, rel_med, yerr=[rel_med - rel_p16, rel_p84 - rel_med], fmt="o-", capsize=3, lw=2, label="relics")
ax.errorbar(x, ctl_med, yerr=[ctl_med - ctl_p16, ctl_p84 - ctl_med], fmt="s--", capsize=3, lw=2, label="non-relics")

ax.set_xscale("log")
ax.set_xlabel("Aperture [Mpc comoving]")
ax.set_ylabel(r"Median density contrast $\delta$")
ax.axhline(0.0, color="k", lw=1, alpha=0.5)
ax.grid(True, alpha=0.3)
ax.legend()
save_fig(fig, "density_contrast_medians_vs_aperture.png")


# ==============================================================
# NORMALISED DENSITY CONTRAST
# ==============================================================

print("\nComputing normalised density contrast (eq. 13)...")

# --- compute global mean + std from FULL population ---
mean_all = np.nanmean(dcont_all, axis=0)
std_all  = np.nanstd(dcont_all, axis=0)

# avoid division by zero
std_all[std_all == 0] = np.nan

# --- compute normalised contrasts ---
dcont_relic_rel   = (dcont_relic   - mean_all) / std_all
dcont_control_rel = (dcont_control - mean_all) / std_all

# ---------------- PLOT 1: THREE APERTURE PANELS (NORMALISED) ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for j, ap in enumerate(apertures):
    ax = axes[j]
    rel = dcont_relic_rel[:, j]
    ctl = dcont_control_rel[:, j]

    rel = rel[np.isfinite(rel)]
    ctl = ctl[np.isfinite(ctl)]

    if rel.size == 0 and ctl.size == 0:
        ax.set_title(f"{ap/1000:.1f} Mpc")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        continue

    allv = np.concatenate([rel, ctl]) if (rel.size and ctl.size) else (rel if rel.size else ctl)
    lo, hi = np.nanpercentile(allv, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -3, 3

    bins = np.linspace(lo, hi, 40)

    if ctl.size > 0:
        ax.hist(ctl, bins=bins, histtype="step", density=True, lw=2, label=f"non-relic")
        ax.axvline(np.nanmedian(ctl), color="C0", ls="--", lw=1)

    if rel.size > 0:
        ax.hist(rel, bins=bins, histtype="step", density=True, lw=2, label=f"relic")
        ax.axvline(np.nanmedian(rel), color="C1", ls="--", lw=1)

    ax.axvline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_title(f"Aperture: {ap/1000:.1f} Mpc")
    ax.set_xlabel(r"$\delta_{\rm rel}$")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Probability density")
axes[0].legend(fontsize=8)

fig.tight_layout()
save_fig(fig, "density_contrast_normalised_three_apertures.png")


# ---------------- PLOT 2: MEDIANS VS APERTURE (NORMALISED) ----------------
rel_med = np.nanmedian(dcont_relic_rel, axis=0)
ctl_med = np.nanmedian(dcont_control_rel, axis=0)

rel_p16 = np.nanpercentile(dcont_relic_rel, 16, axis=0)
rel_p84 = np.nanpercentile(dcont_relic_rel, 84, axis=0)
ctl_p16 = np.nanpercentile(dcont_control_rel, 16, axis=0)
ctl_p84 = np.nanpercentile(dcont_control_rel, 84, axis=0)

fig, ax = plt.subplots(figsize=(7, 5))
x = apertures / 1000.0

ax.errorbar(x, rel_med, yerr=[rel_med - rel_p16, rel_p84 - rel_med],
            fmt="o-", capsize=3, lw=2, label="relics")
ax.errorbar(x, ctl_med, yerr=[ctl_med - ctl_p16, ctl_p84 - ctl_med],
            fmt="s--", capsize=3, lw=2, label="non-relics")

ax.set_xscale("log")
ax.set_xlabel("Aperture [Mpc comoving]")
ax.set_ylabel(r"Median $\delta_{\rm rel}$")
ax.axhline(0.0, color="k", lw=1, alpha=0.6)
ax.grid(True, alpha=0.3)
ax.legend()

save_fig(fig, "density_contrast_normalised_medians_vs_aperture.png")

print("Saved normalised density contrast plots.")


# ==============================================================
# CENTRAL vs SATELLITE ENVIRONMENT
# ==============================================================

# Requires:
#   dcont_all           : density contrast for the full matched sample, shape (N_all, 3)
#   is_central_selected : boolean array aligned with dcont_all
#   apertures           : 3 apertures

# is_central_selected = np.asarray(is_central_selected).astype(bool)

dcont_cen = dcont_relic[is_central_sel[mask_relic]]
dcont_sat = dcont_relic[~is_central_sel[mask_relic]]

print("Central galaxies:", dcont_cen.shape[0])
print("Satellite galaxies:", dcont_sat.shape[0])

# ---------------- PLOT 1: three aperture panels (centrals vs satellites) ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for j, ap in enumerate(apertures):
    ax = axes[j]

    cen = dcont_cen[:, j]
    sat = dcont_sat[:, j]

    cen = cen[np.isfinite(cen)]
    sat = sat[np.isfinite(sat)]

    if cen.size == 0 and sat.size == 0:
        ax.set_title(f"{ap/1000:.1f} Mpc")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        continue

    allv = np.concatenate([cen, sat]) if (cen.size and sat.size) else (cen if cen.size else sat)
    lo, hi = np.nanpercentile(allv, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1, 1

    bins = np.linspace(lo, hi, 40)

    if sat.size > 0:
        ax.hist(sat, bins=bins, histtype="step", density=True, lw=2,
                color="C2", label=f"satellites (N={sat.size})")
        ax.axvline(np.nanmedian(sat), color="C2", ls="--", lw=1)

    if cen.size > 0:
        ax.hist(cen, bins=bins, histtype="step", density=True, lw=2,
                color="C1", label=f"centrals (N={cen.size})")
        ax.axvline(np.nanmedian(cen), color="C1", ls="--", lw=1)

    ax.axvline(0.0, color="k", lw=1, alpha=0.5)
    ax.set_title(f"Aperture: {ap/1000:.1f} Mpc")
    ax.set_xlabel(r"$\delta = (\rho - \bar{\rho}) / \bar{\rho}$")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Probability density")
axes[0].legend(fontsize=8)

fig.tight_layout()
save_fig(fig, "density_contrast_central_vs_satellite_three_apertures.png")

# ---------------- PLOT 2: medians vs aperture ----------------
cen_med = np.nanmedian(dcont_cen, axis=0) if dcont_cen.size else np.full(len(apertures), np.nan)
sat_med = np.nanmedian(dcont_sat, axis=0) if dcont_sat.size else np.full(len(apertures), np.nan)

cen_p16 = np.nanpercentile(dcont_cen, 16, axis=0) if dcont_cen.size else np.full(len(apertures), np.nan)
cen_p84 = np.nanpercentile(dcont_cen, 84, axis=0) if dcont_cen.size else np.full(len(apertures), np.nan)
sat_p16 = np.nanpercentile(dcont_sat, 16, axis=0) if dcont_sat.size else np.full(len(apertures), np.nan)
sat_p84 = np.nanpercentile(dcont_sat, 84, axis=0) if dcont_sat.size else np.full(len(apertures), np.nan)

fig, ax = plt.subplots(figsize=(7, 5))
x = np.asarray(apertures, dtype=float) / 1000.0

ax.errorbar(x, cen_med, yerr=[cen_med - cen_p16, cen_p84 - cen_med],
            fmt="o-", capsize=3, lw=2, color="C1", label="centrals")
ax.errorbar(x, sat_med, yerr=[sat_med - sat_p16, sat_p84 - sat_med],
            fmt="s--", capsize=3, lw=2, color="C2", label="satellites")

ax.set_xscale("log")
ax.set_xlabel("Aperture [Mpc comoving]")
ax.set_ylabel(r"Median density contrast $\delta$")
ax.axhline(0.0, color="k", lw=1, alpha=0.5)
ax.grid(True, alpha=0.3)
ax.legend()

save_fig(fig, "density_contrast_central_vs_satellite_medians_vs_aperture.png")

print("Saved central vs satellite density-contrast plots.")

# ==============================================================
# NORMALISED DENSITY CONTRAST: CENTRAL vs SATELLITE
# ==============================================================

# relic-only mean/std at each aperture
mean_rel = np.nanmean(dcont_relic, axis=0)
std_rel  = np.nanstd(dcont_relic, axis=0)
std_rel[std_rel == 0] = np.nan

# normalised density contrast for relics only
dcont_relic_rel = (dcont_relic - mean_rel) / std_rel

# relic central/satellite split, aligned with dcont_relic
is_central_relic = np.asarray(is_central_sel[mask_relic]).astype(bool)
dcont_cen_rel = dcont_relic_rel[is_central_relic]
dcont_sat_rel = dcont_relic_rel[~is_central_relic]

# ---------------- PLOT 1: THREE APERTURE PANELS ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for j, ap in enumerate(apertures):
    ax = axes[j]
    cen = dcont_cen_rel[:, j]
    sat = dcont_sat_rel[:, j]

    cen = cen[np.isfinite(cen)]
    sat = sat[np.isfinite(sat)]

    if cen.size == 0 and sat.size == 0:
        ax.set_title(f"{ap/1000:.1f} Mpc")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        continue

    allv = np.concatenate([cen, sat]) if (cen.size and sat.size) else (cen if cen.size else sat)
    lo, hi = np.nanpercentile(allv, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -3, 3

    bins = np.linspace(lo, hi, 40)

    if sat.size > 0:
        ax.hist(sat, bins=bins, histtype="step", density=True, lw=2,
                label=f"satellite (N={sat.size})")
        ax.axvline(np.nanmedian(sat), color="C2", ls="--", lw=1)

    if cen.size > 0:
        ax.hist(cen, bins=bins, histtype="step", density=True, lw=2,
                label=f"central (N={cen.size})")
        ax.axvline(np.nanmedian(cen), color="C1", ls="--", lw=1)

    ax.axvline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_title(f"Aperture: {ap/1000:.1f} Mpc")
    ax.set_xlabel(r"$\delta_{\rm rel}$")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Probability density")
axes[0].legend(fontsize=8)

fig.tight_layout()
save_fig(fig, "density_contrast_normalised_central_vs_satellite_three_apertures.png")

# ---------------- PLOT 2: MEDIANS VS APERTURE ----------------
cen_med = np.nanmedian(dcont_cen_rel, axis=0) if dcont_cen_rel.size else np.full(3, np.nan)
sat_med = np.nanmedian(dcont_sat_rel, axis=0) if dcont_sat_rel.size else np.full(3, np.nan)

cen_p16 = np.nanpercentile(dcont_cen_rel, 16, axis=0) if dcont_cen_rel.size else np.full(3, np.nan)
cen_p84 = np.nanpercentile(dcont_cen_rel, 84, axis=0) if dcont_cen_rel.size else np.full(3, np.nan)
sat_p16 = np.nanpercentile(dcont_sat_rel, 16, axis=0) if dcont_sat_rel.size else np.full(3, np.nan)
sat_p84 = np.nanpercentile(dcont_sat_rel, 84, axis=0) if dcont_sat_rel.size else np.full(3, np.nan)

fig, ax = plt.subplots(figsize=(7, 5))
x = apertures / 1000.0

ax.errorbar(x, sat_med, yerr=[sat_med - sat_p16, sat_p84 - sat_med],
            fmt="o-", capsize=3, lw=2, label="satellites")
ax.errorbar(x, cen_med, yerr=[cen_med - cen_p16, cen_p84 - cen_med],
            fmt="s--", capsize=3, lw=2, label="centrals")

ax.set_xscale("log")
ax.set_xlabel("Aperture [Mpc comoving]")
ax.set_ylabel(r"Median $\delta_{\rm rel}$")
ax.axhline(0.0, color="k", lw=1, alpha=0.6)
ax.grid(True, alpha=0.3)
ax.legend()

save_fig(fig, "density_contrast_normalised_central_vs_satellite_medians_vs_aperture.png")

print("Saved normalised central/satellite density contrast plots.")


# ==============================================================
# CLEAN 2D HEATMAP + CONTOUR VERSIONS
#   - one figure vs stellar mass
#   - one figure vs compactness
#   - rows: relics / non-relics
#   - columns: apertures
#   - smooth heatmap, black contours, dedicated colourbar column
# ==============================================================

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import PowerNorm, LogNorm #from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

# ----------------------------
# aligned x-variables
# ----------------------------
logM_matched = np.log10(m_matched)
compactness_matched = logM_matched - 1.5 * np.log10(r_matched)

logM_relic = logM_matched[mask_relic]
logM_control = logM_matched[mask_control]
comp_relic = compactness_matched[mask_relic]
comp_control = compactness_matched[mask_control]

proxy_legend = [
    Patch(facecolor="lightgrey", edgecolor="none", alpha=0.35, label="background galaxies"),
    Line2D([0], [0], color="k", lw=1.0, label="contours from smoothed 2D histogram"),
]

def _smoothed_hist2d_panel(ax, x, y, xedges, yedges, title,
                           x_label, y_label,
                           add_points=True, point_color="lightgrey", contour_color="k",
                           point_alpha=0.18, point_size=4,
                           smooth_sigma=1.2, show_heatmap=True):
    """
    Smooth 2D histogram panel:
      - light-grey background points
      - smoothed histogram shown as a continuous image
      - black contour lines
    Returns the image artist (for a shared colorbar) or None.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axvline(0.0, color="k", lw=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        return None

    xx = np.asarray(x[ok], dtype=float)
    yy = np.asarray(y[ok], dtype=float)

    if add_points:
        ax.scatter(xx, yy, s=point_size, color=point_color, alpha=point_alpha,
                   linewidths=0, zorder=0)

    # Raw histogram (density normalised)
    H, xe, ye = np.histogram2d(xx, yy, bins=[xedges, yedges], density=True)

    # Smooth and renormalise so the colour scale is comparable between panels
    Hs = gaussian_filter(H.T.astype(float), sigma=smooth_sigma, mode="nearest")
    dx = float(np.mean(np.diff(xedges)))
    dy = float(np.mean(np.diff(yedges)))
    norm_int = np.nansum(Hs) * dx * dy
    if np.isfinite(norm_int) and norm_int > 0:
        Hs = Hs / norm_int

    Hm = np.ma.masked_where(Hs <= 0, Hs)

    positive = Hs[Hs > 0]
    if positive.size == 0:
        ax.set_title(title)
        ax.axvline(0.0, color="k", lw=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        return None

    vmin = float(np.nanpercentile(positive, 5))
    vmax = float(np.nanpercentile(positive, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmin == vmax:
        vmin = max(float(np.nanmin(positive)), 1e-6)
        vmax = max(float(np.nanmax(positive)), vmin * 10)

    cmap = cmr.iceburn

    if show_heatmap:
        im = ax.imshow(
            Hm,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            interpolation="bilinear",
            zorder=1,
        )
    else:
        im = None    
        

    # Contours from the same smoothed field
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xc, yc)

    levels = np.nanpercentile(positive, [68, 90, 95])
    levels = levels[np.isfinite(levels)]
    if levels.size >= 2:
        ax.contour(X, Y, Hs, levels=levels, colors=contour_color, linewidths=1.0, zorder=2)

    ax.axvline(0.0, color="k", lw=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)

    return im

# ----------------------------
# shared bin edges per figure
# ----------------------------
bins_x = 60
bins_y = 60

# --- Figure A: density contrast vs stellar mass ---
x_all_mass = dcont_all.ravel()
x_all_mass = x_all_mass[np.isfinite(x_all_mass)]
y_all_mass = np.concatenate([logM_relic, logM_control])
y_all_mass = y_all_mass[np.isfinite(y_all_mass)]

xlo_mass, xhi_mass = np.nanpercentile(x_all_mass, [1, 99])
ylo_mass, yhi_mass = np.nanpercentile(y_all_mass, [1, 99])
if not np.isfinite(xlo_mass) or not np.isfinite(xhi_mass) or xlo_mass == xhi_mass:
    xlo_mass, xhi_mass = np.nanmin(x_all_mass), np.nanmax(x_all_mass)
if not np.isfinite(ylo_mass) or not np.isfinite(yhi_mass) or ylo_mass == yhi_mass:
    ylo_mass, yhi_mass = np.nanmin(y_all_mass), np.nanmax(y_all_mass)

xpad_mass = 0.03 * (xhi_mass - xlo_mass + 1e-6)
ypad_mass = 0.03 * (yhi_mass - ylo_mass + 1e-6)

xedges_mass = np.linspace(xlo_mass - xpad_mass, xhi_mass + xpad_mass, bins_x)
yedges_mass = np.linspace(ylo_mass - ypad_mass, yhi_mass + ypad_mass, bins_y)

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12, hspace=0.28)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)])
cax = fig.add_subplot(gs[:, 3])

im_for_cbar = None
for j, ap in enumerate(apertures):
    im = _smoothed_hist2d_panel(
        axes[0, j],
        dcont_relic[:, j], logM_relic,
        xedges_mass, yedges_mass,
        title=f"Ancients | Aperture: {ap/1000:.1f} Mpc",
        x_label=r"$\delta = (\rho - \bar{\rho}) / \bar{\rho}$",
        y_label=r"$\log_{10}(M_\star / M_\odot)$",
        add_points=True,
        point_color="lightgrey",
        #contour_color="k",
        point_alpha=0.22,
        smooth_sigma=2.0,
    )
    if im_for_cbar is None and im is not None:
        im_for_cbar = im

    _smoothed_hist2d_panel(
        axes[1, j],
        dcont_control[:, j], logM_control,
        xedges_mass, yedges_mass,
        title=f"Non-ancients | Aperture: {ap/1000:.1f} Mpc",
        x_label=r"$\delta = (\rho - \bar{\rho}) / \bar{\rho}$",
        y_label=r"$\log_{10}(M_\star / M_\odot)$",
        add_points=True,
        point_color="lightgrey",
        contour_color="k",
        point_alpha=0.10,
        smooth_sigma=2.0,
        show_heatmap=False,
    )

    # overlay ancient contours
    _smoothed_hist2d_panel(
        axes[1,j],
        dcont_relic[:, j], logM_relic,
        xedges_mass, yedges_mass,
        title=None,
        x_label=None,
        y_label=None,
        add_points=False,
        smooth_sigma=2.0,
        contour_color="C1",
        show_heatmap=False,
    )

for row in range(2):
    for col in (1, 2):
        axes[row, col].tick_params(axis="y", left=False, labelleft=False)
        axes[row, col].set_ylabel("")

if im_for_cbar is not None:
    cbar = fig.colorbar(im_for_cbar, cax=cax)
    cbar.set_label(r"Smoothed probability density")

axes[0, 0].legend(handles=proxy_legend, fontsize=8, loc="upper left", frameon=True)
fig.subplots_adjust(left=0.06, right=0.93, bottom=0.08, top=0.93)
save_fig(fig, "density_contrast_vs_mass_2d_contours.png")

# --- Figure B: density contrast vs compactness ---
x_all_comp = dcont_all.ravel()
x_all_comp = x_all_comp[np.isfinite(x_all_comp)]
y_all_comp = np.concatenate([comp_relic, comp_control])
y_all_comp = y_all_comp[np.isfinite(y_all_comp)]

xlo_comp, xhi_comp = np.nanpercentile(x_all_comp, [1, 99])
ylo_comp, yhi_comp = np.nanpercentile(y_all_comp, [1, 99])
if not np.isfinite(xlo_comp) or not np.isfinite(xhi_comp) or xlo_comp == xhi_comp:
    xlo_comp, xhi_comp = np.nanmin(x_all_comp), np.nanmax(x_all_comp)
if not np.isfinite(ylo_comp) or not np.isfinite(yhi_comp) or ylo_comp == yhi_comp:
    ylo_comp, yhi_comp = np.nanmin(y_all_comp), np.nanmax(y_all_comp)

xpad_comp = 0.03 * (xhi_comp - xlo_comp + 1e-6)
ypad_comp = 0.03 * (yhi_comp - ylo_comp + 1e-6)

xedges_comp = np.linspace(xlo_comp - xpad_comp, xhi_comp + xpad_comp, bins_x)
yedges_comp = np.linspace(ylo_comp - ypad_comp, yhi_comp + ypad_comp, bins_y)

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12, hspace=0.28)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)])
cax = fig.add_subplot(gs[:, 3])

im_for_cbar = None
for j, ap in enumerate(apertures):
    im = _smoothed_hist2d_panel(
        axes[0, j],
        dcont_relic[:, j], comp_relic,
        xedges_comp, yedges_comp,
        title=f"Ancients | Aperture: {ap/1000:.1f} Mpc",
        x_label=r"$\delta = (\rho - \bar{\rho}) / \bar{\rho}$",
        y_label=r"Compactness [$\lg(M_\odot \text{kpc}^{-1.5})$]",
        add_points=True,
        point_color="lightgrey",
        point_alpha=0.22,
        smooth_sigma=2.0,
    )
    if im_for_cbar is None and im is not None:
        im_for_cbar = im

    _smoothed_hist2d_panel(
        axes[1, j],
        dcont_control[:, j], comp_control,
        xedges_comp, yedges_comp,
        title=f"Non-ancients | Aperture: {ap/1000:.1f} Mpc",
        x_label=r"$\delta = (\rho - \bar{\rho}) / \bar{\rho}$",
        y_label=r"Compactness [$\lg(M_\odot \text{kpc}^{-1.5})$]",
        add_points=True,
        point_color="lightgrey",
        contour_color="k",
        point_alpha=0.10,
        smooth_sigma=2.0,
        show_heatmap=False,
    )
    # overlay ancient contours (COMPACTNESS CASE)
    _smoothed_hist2d_panel(
        axes[1, j],                         
        dcont_relic[:, j], comp_relic,      
        xedges_comp, yedges_comp,           
        title=None,
        x_label=None,
        y_label=None,
        add_points=False,
        smooth_sigma=2.0,
        contour_color="C1",
        show_heatmap=False,
    )

for row in range(2):
    for col in (1, 2):
        axes[row, col].tick_params(axis="y", left=False, labelleft=False)
        axes[row, col].set_ylabel("")

if im_for_cbar is not None:
    cbar = fig.colorbar(im_for_cbar, cax=cax)
    cbar.set_label(r"Smoothed probability density")

axes[0, 0].legend(handles=proxy_legend, fontsize=8, loc="upper left", frameon=True)
fig.subplots_adjust(left=0.06, right=0.93, bottom=0.08, top=0.93)
save_fig(fig, "density_contrast_vs_compactness_2d_contours.png")

# ==============================================================
# CLEAN 2D HEATMAP + CONTOUR VERSIONS (NORMALISED δ_rel)
# ==============================================================

# --- compute normalised density contrast first ---
mean_all = np.nanmean(dcont_all, axis=0)
std_all  = np.nanstd(dcont_all, axis=0)
std_all[std_all == 0] = np.nan

dcont_relic_rel   = (dcont_relic   - mean_all) / std_all
dcont_control_rel = (dcont_control - mean_all) / std_all
dcont_all_rel     = (dcont_all     - mean_all) / std_all

# ----------------------------
# FIGURE A: δ_rel vs stellar mass
# ----------------------------
x_all_mass = dcont_all_rel.ravel()
x_all_mass = x_all_mass[np.isfinite(x_all_mass)]
y_all_mass = np.concatenate([logM_relic, logM_control])
y_all_mass = y_all_mass[np.isfinite(y_all_mass)]

xlo_mass, xhi_mass = np.nanpercentile(x_all_mass, [1, 99])
ylo_mass, yhi_mass = np.nanpercentile(y_all_mass, [1, 99])

xpad_mass = 0.03 * (xhi_mass - xlo_mass + 1e-6)
ypad_mass = 0.03 * (yhi_mass - ylo_mass + 1e-6)

# #fixed x range
# xlo_mass, xhi_mass = -3.0, 8.0
# xedges_mass = np.linspace(xlo_mass, xhi_mass, bins_x)

xedges_mass = np.linspace(xlo_mass - xpad_mass, xhi_mass + xpad_mass, bins_x)
yedges_mass = np.linspace(ylo_mass - ypad_mass, yhi_mass + ypad_mass, bins_y)

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12, hspace=0.28)
axes = np.array([[fig.add_subplot(gs[i, j], sharex=None) for j in range(3)] for i in range(2)])
cax = fig.add_subplot(gs[:,3])

im_for_cbar = None
for j, ap in enumerate(apertures):
    im = _smoothed_hist2d_panel(
        axes[0, j],
        dcont_relic_rel[:, j], logM_relic,
        xedges_mass, yedges_mass,
        title=f"Ancients | Aperture: {ap/1000:.1f} Mpc",
        x_label=r"$\delta_{\rm rel}$",
        y_label=r"$\log_{10}(M_\star / M_\odot)$",
        point_alpha=0.22,
        smooth_sigma=2.0,
    )
    if im_for_cbar is None and im is not None:
        im_for_cbar = im

    _smoothed_hist2d_panel(
        axes[1, j],
        dcont_control_rel[:, j], logM_control,
        xedges_mass, yedges_mass,
        title=f"Non-ancients | Aperture: {ap/1000:.1f} Mpc",
        x_label=r"$\delta_{\rm rel}$",
        y_label=r"$\log_{10}(M_\star / M_\odot)$",
        point_alpha=0.10,
        contour_color="k",
        smooth_sigma=2.0,
        show_heatmap=False,
    )

    # overlay ancient contours
    _smoothed_hist2d_panel(
        axes[1,j],
        dcont_relic_rel[:, j], logM_relic,
        xedges_mass, yedges_mass,
        title=None,
        x_label=None,
        y_label=None,
        add_points=False,
        smooth_sigma=2.0,
        contour_color="C1",
        show_heatmap=False,
    )

# remove duplicate y-axes
for row in range(2):
    for col in (1,2):
        axes[row, col].tick_params(axis="y", left=False, labelleft=False)
        axes[row, col].set_ylabel("")

if im_for_cbar is not None:
    cbar = fig.colorbar(im_for_cbar, cax=cax)
    cbar.set_label(r"Smoothed probability density")

axes[0,0].legend(handles=proxy_legend, fontsize=8, loc="upper left")
for ax in axes.ravel():
    ax.set_xlim(-1.0, 5.0)
fig.subplots_adjust(left=0.06, right=0.93, bottom=0.08, top=0.93)
save_fig(fig, "density_contrast_rel_vs_mass_2d_contours.png")


# ----------------------------
# FIGURE B: δ_rel vs compactness
# ----------------------------
x_all_comp = dcont_all_rel.ravel()
x_all_comp = x_all_comp[np.isfinite(x_all_comp)]
y_all_comp = np.concatenate([comp_relic, comp_control])
y_all_comp = y_all_comp[np.isfinite(y_all_comp)]

xlo_comp, xhi_comp = np.nanpercentile(x_all_comp, [1, 99])
ylo_comp, yhi_comp = np.nanpercentile(y_all_comp, [1, 99])

xpad_comp = 0.03 * (xhi_comp - xlo_comp + 1e-6)
ypad_comp = 0.03 * (yhi_comp - ylo_comp + 1e-6)

# #fixed x range
# xlo_mass, xhi_mass = -3.0, 8.0
# xedges_comp = np.linspace(xlo_mass, xhi_mass, bins_x)

xedges_comp = np.linspace(xlo_comp - xpad_comp, xhi_comp + xpad_comp, bins_x)
yedges_comp = np.linspace(ylo_comp - ypad_comp, yhi_comp + ypad_comp, bins_y)

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.12, hspace=0.28)
axes = np.array([[fig.add_subplot(gs[i, j], sharex=None) for j in range(3)] for i in range(2)])
cax = fig.add_subplot(gs[:,3])

im_for_cbar = None
for j, ap in enumerate(apertures):
    im = _smoothed_hist2d_panel(
        axes[0, j],
        dcont_relic_rel[:, j], comp_relic,
        xedges_comp, yedges_comp,
        title=f"Ancients | Aperture: {ap/1000:.1f} Mpc",
        x_label=r"$\delta_{\rm rel}$",
        y_label=r"Compactness [$\lg(M_\odot \mathrm{kpc}^{-1.5})$]",
        point_alpha=0.22,
        smooth_sigma=2.0,
    )
    if im_for_cbar is None and im is not None:
        im_for_cbar = im

    _smoothed_hist2d_panel(
        axes[1, j],
        dcont_control_rel[:, j], comp_control,
        xedges_comp, yedges_comp,
        title=f"Non-ancients | Aperture: {ap/1000:.1f} Mpc",
        x_label=r"$\delta_{\rm rel}$",
        y_label=r"Compactness [$\lg(M_\odot \mathrm{kpc}^{-1.5})$]",
        point_alpha=0.10,
        contour_color="k",
        smooth_sigma=2.0,
        show_heatmap=False,
    )

    # overlay ancient contours (COMPACTNESS CASE)
    _smoothed_hist2d_panel(
        axes[1, j],                         
        dcont_relic_rel[:, j], comp_relic,    
        xedges_comp, yedges_comp,           
        title=None,
        x_label=None,
        y_label=None,
        add_points=False,
        smooth_sigma=2.0,
        contour_color="C1",
        show_heatmap=False,
    )

for row in range(2):
    for col in (1,2):
        axes[row, col].tick_params(axis="y", left=False, labelleft=False)
        axes[row, col].set_ylabel("")

if im_for_cbar is not None:
    cbar = fig.colorbar(im_for_cbar, cax=cax)
    cbar.set_label(r"Smoothed probability density")

axes[0,0].legend(handles=proxy_legend, fontsize=8, loc="upper left")
for ax in axes.ravel():
    ax.set_xlim(-1.0, 5.0)
fig.subplots_adjust(left=0.06, right=0.93, bottom=0.08, top=0.93)
save_fig(fig, "density_contrast_rel_vs_compactness_2d_contours.png")

print("Done.")

# ==============================================================
# PENG-STYLE FRACTION MAPS
#   - colour = fraction of a selected subset in each bin
#   - denominator = all matched galaxies in that bin
#   - numerator   = any boolean selection you choose
#   - efficient: histogram2d only, no per-galaxy Python loops
# ==============================================================

from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

# --------------------------------------------------------------
# aligned arrays for the matched sample
# --------------------------------------------------------------
logM_matched = np.log10(m_matched)
compactness_matched = logM_matched - 1.5 * np.log10(r_matched)

# Rebuild a matched-aligned environment array so all masks line up
# with the same matched-sample ordering.
Nmatch = len(dor_matched)
dcont_matched_raw = np.full((Nmatch, len(apertures)), np.nan, dtype=float)

# fill correctly using indices, not masks
dcont_matched_raw[np.where(mask_relic)[0]] = dcont_relic
dcont_matched_raw[np.where(mask_control)[0]] = dcont_control

# Optional: normalised density contrast for the matched sample
mean_matched = np.nanmean(dcont_matched_raw, axis=0)
std_matched = np.nanstd(dcont_matched_raw, axis=0)
std_matched[std_matched == 0] = np.nan
dcont_matched_rel = (dcont_matched_raw - mean_matched) / std_matched

# Switch between raw and normalised density contrast here
USE_NORMALISED_DENSITY_CONTRAST = True
if USE_NORMALISED_DENSITY_CONTRAST:
    y_density_by_ap = [dcont_matched_rel[:, j] for j in range(len(apertures))]
    density_ylabel = r"$\delta_{\rm rel}$"
else:
    y_density_by_ap = [dcont_matched_raw[:, j] for j in range(len(apertures))]
    density_ylabel = r"$\delta$"

# --------------------------------------------------------------
# helpers
# --------------------------------------------------------------
def _slugify(text):
    return (
        str(text).lower()
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

def _make_edges(arrays, nbins=45, pad_frac=0.03):
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        raise RuntimeError("Not enough finite values to build bin edges.")
    lo, hi = np.nanpercentile(vals, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(vals), np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise RuntimeError("Could not determine bin edges.")
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    pad = pad_frac * (hi - lo + 1e-6)
    return np.linspace(lo - pad, hi + pad, nbins + 1)

def _make_quantile_edges(arr, target_per_bin=800):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    N = arr.size
    nbins = max(3, int(np.floor(N / target_per_bin)))
    quantiles = np.linspace(0, 100, nbins + 1)
    edges = np.nanpercentile(arr, quantiles)

    # enforce strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-9

    return edges

def _fraction_panel(ax, x, y, select_mask, xedges, yedges, title,
                    xlabel, ylabel, min_count=20, add_total_contours=True):
    """
    Fraction in each 2D bin: N(selected) / N(total).
    Bins with too few total galaxies are masked.
    """
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

    print("DEBUG panel:")
    print("  total finite points:", int(ok.sum()))
    print("  selected points:", int(sel_ok.sum()))
    print("  fraction selected:", float(sel_ok.mean()))
    print("  H_tot min/max:", float(np.nanmin(H_tot)), float(np.nanmax(H_tot)))
    print("  H_sel min/max:", float(np.nanmin(H_sel)), float(np.nanmax(H_sel)))

    cmap = cmr.iceburn
    finite_vals = frac.compressed()  # valid (unmasked) fractions only

    if finite_vals.size > 0:
        vmax = float(np.nanpercentile(finite_vals, 98))
        vmax = max(vmax, 0.05)   # keeps very small fractions visible
    else:
        vmax = 1.0

    im = ax.pcolormesh(
        xe, ye, frac,
        shading="auto",
        cmap=cmap,
        norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax), #norm=Normalize(vmin=0.0, vmax=vmax),
        zorder=1,
    )

    # Contours of the full sample, to show where the galaxies actually are
    if add_total_contours:
        good = H_tot >= min_count
        if np.any(good):
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            X, Y = np.meshgrid(xc, yc)

            levels = np.unique(np.nanpercentile(H_tot[good], [68, 90, 95])) #[50, 75, 90, 97]
            levels = np.asarray(levels, dtype=float)
            levels = levels[np.isfinite(levels)]
            if levels.size >= 2:
                levels = np.sort(levels)
                ax.contour(
                    X, Y, H_tot.T,
                    levels=levels,
                    colors="k",
                    linewidths=0.8,
                    alpha=0.55,
                    zorder=2,
                )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return im

def make_fraction_figure_mass_env(select_mask, selection_label, y_by_aperture, outname,
                                  y_label, min_count=200, nbins_x=45, nbins_y=45):
    """
    3-panel fraction map:
      x = logM
      y = density contrast (one panel per aperture)
      colour = fraction of the selected subset
    """
    xedges = _make_edges([logM_matched], nbins=nbins_x, pad_frac=0.03) #xedges = _make_quantile_edges(logM_matched, target_per_bin=800)
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

    axes[0].legend(
        handles=[Line2D([0], [0], color="k", lw=0.8, label="total-sample contours")],
        fontsize=8,
        loc="upper left",
        frameon=True,
    )

    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.10, top=0.92)
    save_fig(fig, outname)

def make_fraction_figure_mass_compactness(select_mask, selection_label, outname,
                                         min_count=20, nbins_x=45, nbins_y=45):
    """
    Fraction map:
      x = logM
      y = compactness
      colour = fraction of the selected subset
    """
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

    ax.legend(
        handles=[Line2D([0], [0], color="k", lw=0.8, label="total-sample contours")],
        fontsize=8,
        loc="upper left",
        frameon=True,
    )

    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.11, top=0.92)
    save_fig(fig, outname)

# --------------------------------------------------------------
# choose the numerator masks
# --------------------------------------------------------------
fraction_cases = {
    "Ancient galaxies": mask_relic,
    # "DoR > 0.6": np.isfinite(dor_matched) & (dor_matched > EXTREME_DOR),
}

# Optional extra cases: enable only if the aligned arrays exist in your script.
if "stellarage_lum_matched" in globals() and len(stellarage_lum_matched) == len(dor_matched):
    fraction_cases["Old galaxies (age > 10 Gyr)"] = np.isfinite(stellarage_lum_matched) & (stellarage_lum_matched > 10.0)
if "logZstar_rel_matched" in globals() and len(logZstar_rel_matched) == len(dor_matched):
    fraction_cases["Supersolar metallicity galaxies"] = np.isfinite(logZstar_rel_matched) & (logZstar_rel_matched > 0.0)
if "exsitu_fracs_matched" in globals() and len(exsitu_fracs_matched) == len(dor_matched):
     fraction_cases["Ex-situ > 0.3"] = np.isfinite(exsitu_fracs_matched) & (exsitu_fracs_matched > 0.3)

if "exsitu_fracs_matched" in globals() and len(exsitu_fracs_matched) == len(dor_matched):
    ex_mask = np.isfinite(exsitu_fracs_matched) & (exsitu_fracs_matched > 0.3)
    print("DEBUG ex-situ > 0.3:")
    print("  finite ex-situ entries:", int(np.isfinite(exsitu_fracs_matched).sum()), "/", len(exsitu_fracs_matched))
    print("  galaxies with ex-situ > 0.3:", int(ex_mask.sum()), "/", len(ex_mask))
    print("  fraction:", float(ex_mask.mean()))
    print("  min/max ex-situ:", float(np.nanmin(exsitu_fracs_matched)), float(np.nanmax(exsitu_fracs_matched)))
# --------------------------------------------------------------
# make the plots
# --------------------------------------------------------------
for label, sel_mask in fraction_cases.items():
    make_fraction_figure_mass_env(
        sel_mask,
        label,
        y_by_aperture=y_density_by_ap,
        outname=f"fraction_{_slugify(label)}_vs_mass_densitycontrast.png",
        y_label=density_ylabel,
        min_count=20,
        nbins_x=45,
        nbins_y=45,
    )

for label, sel_mask in fraction_cases.items():
    make_fraction_figure_mass_compactness(
        sel_mask,
        label,
        outname=f"fraction_{_slugify(label)}_vs_mass_compactness.png",
        min_count=20,
        nbins_x=45,
        nbins_y=45,
    )

print("Finished fraction-map plots.")