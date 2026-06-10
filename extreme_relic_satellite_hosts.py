#!/usr/bin/env python3
"""
extreme_relic_satellite_hosts.py

Corrected and streamlined version for the single-axis overlay host-mass histogram:
- Ancient galaxies: DoR > EXTREME_DOR
- Relic galaxies: DoR > EXTREME_DOR AND compactness > COMPACTNESS_CUT
- Split into centrals vs satellites
- Preserves the original histogram style:
  * ancient samples: filled histograms with alpha=0.6 and black edges
  * relic samples: dashed step histograms with thicker lines
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import common
from scipy.stats import gaussian_kde

plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "serif", "font.size": 12})

# ============================================================
# CONFIG
# ============================================================
csv_in = "sfh_times_all_with_DoR_variants_corrected.csv.gz"
model_name = "L0200N3008/THERMAL_AGN/"
model_dir = "/mnt/su3-pro/colibre/" + model_name
snap_file = "0127"
outdir = "extreme_relic_satellite_hosts"
os.makedirs(outdir, exist_ok=True)

EXTREME_DOR = 0.6
COMPACTNESS_CUT = 9.72
MIN_STELLAR_MASS = 1e9

# Candidate column names in the CSV
id_candidates = ("subhalo_id", "HaloCatalogueIndex", "subhaloId", "HaloIndex", "track_id", "TrackId")
dor_candidates = ("DoR_t95", "DoR_t998", "DoR_t90", "DoR_tfin", "dor", "DoR", "DoR_choice", "DoR_csv")

sigma_path = (
    "/mnt/su3-pro/colibre/"
    "L0200N3008/THERMAL_AGN/"
    "SOAP-HBT/extra/halo_properties_0127.hdf5"
)

# ============================================================
# FIND CSV COLUMNS
# ============================================================
if not os.path.exists(csv_in):
    raise SystemExit(f"CSV input not found: {csv_in}")

print("Reading CSV header:", csv_in)
header = pd.read_csv(csv_in, nrows=0, low_memory=False)
cols = set(header.columns)

id_col = next((c for c in id_candidates if c in cols), None)
if id_col is None:
    raise SystemExit("No usable ID column found in the CSV.")

dor_col = next((c for c in dor_candidates if c in cols), None)
if dor_col is None:
    raise SystemExit("No usable DoR column found in the CSV.")

print("Using ID column:", id_col)
print("Using DoR column:", dor_col)

# ============================================================
# READ ONLY THE REQUIRED CSV COLUMNS
# ============================================================
print("Reading CSV data...")
df_ucmg = pd.read_csv(csv_in, usecols=[id_col, dor_col], low_memory=False)

df_ucmg[id_col] = pd.to_numeric(df_ucmg[id_col], errors="coerce")
df_ucmg[dor_col] = pd.to_numeric(df_ucmg[dor_col], errors="coerce")
df_ucmg = df_ucmg.dropna(subset=[id_col]).copy()
df_ucmg[id_col] = df_ucmg[id_col].astype(np.int64)

# Make the ID mapping unique and stable
# If duplicates exist, keep the first non-NaN value.
df_ucmg = df_ucmg.dropna(subset=[dor_col]).drop_duplicates(subset=[id_col], keep="first")
dor_series = pd.Series(df_ucmg[dor_col].to_numpy(dtype=np.float32), index=df_ucmg[id_col].to_numpy(dtype=np.int64))
print(f"Loaded DoR entries: {len(dor_series)}")

# ============================================================
# READ SOAP ARRAYS
# ============================================================
print("Reading SOAP arrays...")

fields_id = {
    "InputHalos": (
        "HaloCatalogueIndex",
        "IsCentral",
    )
}
fields_gal = {
    "ExclusiveSphere/50kpc": (
        "StellarMass",
        "HalfMassRadiusStars",
        "CentreOfMass",
        "CentreOfMassVelocity",
    )
}
fields_fof = {
    "InputHalos": (
        "FOF/Masses",
        "FOF/Centres",
        "FOF/Radii",
    )
}
soap_id = {"SOAP": ("HostHaloIndex",)}

(halo_index, is_central) = common.read_group_data_colibre(model_dir, snap_file, fields_id)
(m30, r50, centre_of_mass, centre_of_mass_vel) = common.read_group_data_colibre(model_dir, snap_file, fields_gal)
(fof_masses, fof_centres, fof_radii) = common.read_group_data_colibre(model_dir, snap_file, fields_fof)
(host_halo_index,) = common.read_group_data_colibre(model_dir, snap_file, soap_id)
# ============================================================
# UNITS / SHAPES
# ============================================================
Mu = 1.988e43 / 1.989e33  # raw mass unit -> Msun

m30 = np.asarray(m30, dtype=np.float32).ravel() * Mu
r50 = np.asarray(r50, dtype=np.float32).ravel() * 1e3  # z=0, comoving==physical; kpc conversion only
centre_of_mass = np.asarray(centre_of_mass, dtype=np.float32) * 1e3 #converted to kpc
halo_index = np.asarray(halo_index, dtype=np.int64).ravel()
is_central = np.asarray(is_central).ravel().astype(bool)
fof_masses = np.asarray(fof_masses, dtype=np.float32).ravel() * Mu
fof_centres = np.asarray(fof_centres, dtype=np.float32) * 1e3 #converted to kpc
fof_radii   = np.asarray(fof_radii,   dtype=np.float32).ravel() * 1e3 #converted to kpc

if not (m30.size == r50.size == halo_index.size == is_central.size):
    raise RuntimeError("SOAP arrays do not have matching lengths.")

print("SOAP catalogue size:", m30.size)

# ============================================================
# INITIAL MASS / RADIUS SELECTION
# ============================================================
mask_sel = (
    np.isfinite(m30) & np.isfinite(r50) &
    (m30 >= MIN_STELLAR_MASS) &
    (r50 > 0)
)
sel = np.flatnonzero(mask_sel)
print("Selected galaxies after mass/radius cut:", sel.size)

if sel.size == 0:
    raise SystemExit("No galaxies pass the initial mass/radius selection.")

m = m30[sel]
r = r50[sel]
halo_idx = halo_index[sel]
is_central_sel = is_central[sel]

# Compactness on the selected sample
with np.errstate(divide="ignore", invalid="ignore"):
    compactness = np.where((m > 0) & (r > 0), np.log10(m) - 1.5 * np.log10(r), np.nan)

# ============================================================
# MATCH DoR TO SOAP ROWS (FAST, VECTORIZED)
# ============================================================
print("Matching DoR to SOAP HaloCatalogueIndex...")
dor_all = dor_series.reindex(halo_idx).to_numpy(dtype=np.float32)
valid_dor = np.isfinite(dor_all)
print(f"Matched DoR rows: {int(valid_dor.sum())} / {dor_all.size}")

# Keep only rows that actually have a DoR value
m = m[valid_dor]
r = r[valid_dor]
compactness = compactness[valid_dor]
halo_idx = halo_idx[valid_dor]
is_central_sel = is_central_sel[valid_dor]
dor = dor_all[valid_dor]
sel = sel[valid_dor]

if dor.size == 0:
    raise SystemExit("No matched DoR rows remain after reindexing.")

# ============================================================
# HOST MASS FROM FOF MASS ARRAY
# ============================================================
# HostHaloIndex is used to index host halo properties.
host_halo_index = np.asarray(common.read_group_data_colibre(model_dir, snap_file, {"SOAP": ("HostHaloIndex",)}), dtype=np.int64).ravel()
if host_halo_index.size != m30.size:
    raise RuntimeError("HostHaloIndex length does not match SOAP catalogue length.")
host_halo_index = host_halo_index[sel][valid_dor]

host_mass = np.full(host_halo_index.size, np.nan, dtype=np.float32)
valid_host = (host_halo_index >= 0) & (host_halo_index < fof_masses.size)
if np.any(valid_host):
    host_mass[valid_host] = fof_masses[host_halo_index[valid_host]]

# Optional lightweight fallback if the catalogue uses an off-by-one convention
if np.count_nonzero(np.isfinite(host_mass)) == 0:
    valid_host_m1 = (host_halo_index - 1 >= 0) & (host_halo_index - 1 < fof_masses.size)
    if np.any(valid_host_m1):
        host_mass[valid_host_m1] = fof_masses[host_halo_index[valid_host_m1] - 1]

print("Finite host masses:", int(np.isfinite(host_mass).sum()), "/", host_mass.size)

# ============================================================
# DATAFRAME
# ============================================================
df = pd.DataFrame(
    {
        "DoR": dor,
        "compactness": compactness,
        "host_mass": host_mass,
        "is_central": is_central_sel,
    }
)

# ============================================================
# SAMPLE DEFINITIONS
# ============================================================
ancient_mask = np.isfinite(df["DoR"]) & (df["DoR"] > EXTREME_DOR)
relic_mask = ancient_mask & np.isfinite(df["compactness"]) & (df["compactness"] > COMPACTNESS_CUT)

# Split ancient / relic and central / satellite
anc_sat = df.loc[ancient_mask & (~df["is_central"]), "host_mass"].to_numpy(dtype=np.float32)
anc_cen = df.loc[ancient_mask & (df["is_central"]), "host_mass"].to_numpy(dtype=np.float32)
rel_sat = df.loc[relic_mask & (~df["is_central"]), "host_mass"].to_numpy(dtype=np.float32)
rel_cen = df.loc[relic_mask & (df["is_central"]), "host_mass"].to_numpy(dtype=np.float32)

anc_sat_mask = np.isfinite(anc_sat) & (anc_sat > 0)
anc_cen_mask = np.isfinite(anc_cen) & (anc_cen > 0)
rel_sat_mask = np.isfinite(rel_sat) & (rel_sat > 0)
rel_cen_mask = np.isfinite(rel_cen) & (rel_cen > 0)

all_vals = np.concatenate(
    [
        anc_sat[anc_sat_mask],
        anc_cen[anc_cen_mask],
        rel_sat[rel_sat_mask],
        rel_cen[rel_cen_mask],
    ]
)

if all_vals.size == 0:
    raise SystemExit("No finite positive host masses available for plotting.")

bins = np.arange(11.0, 15.5, 0.25)

# ============================================================
# OVERLAYED HISTOGRAM (MATCHES THE ORIGINAL STYLE)
# ============================================================
plt.figure(figsize=(6, 4))

# Ancient: filled histograms, same as the original style
if anc_sat_mask.sum() > 0:
    plt.hist(
        np.log10(anc_sat[anc_sat_mask]),
        bins=16,
        alpha=0.6,
        edgecolor="k",
        label=f"Ancient satellites (N={anc_sat_mask.sum()})",
    )

if anc_cen_mask.sum() > 0:
    plt.hist(
        np.log10(anc_cen[anc_cen_mask]),
        bins=16,
        alpha=0.6,
        edgecolor="k",
        color="red",
        label=f"Ancient centrals (N={anc_cen_mask.sum()})",
    )

# Relic: dashed step histograms, same colors
if rel_sat_mask.sum() > 0:
    plt.hist(
        np.log10(rel_sat[rel_sat_mask]),
        bins=16,
        histtype="step",
        linewidth=1.5,
        linestyle="--",
        color="C0",
        label=f"Relic satellites (N={rel_sat_mask.sum()})",
    )

if rel_cen_mask.sum() > 0:
    plt.hist(
        np.log10(rel_cen[rel_cen_mask]),
        bins=16,
        histtype="step",
        linewidth=1.5,
        linestyle="--",
        color="red",
        label=f"Relic centrals (N={rel_cen_mask.sum()})",
    )

plt.xlabel(r"$\lg(M_{\mathrm{host}} / M_\odot)$")
plt.ylabel("N")
plt.grid(True, alpha=0.35)
plt.legend(fontsize=9)
plt.tight_layout()

p_overlay = os.path.join(outdir, "hist_host_mass_ancient_vs_relics_overlay.png")
plt.savefig(p_overlay, dpi=200)
plt.close()

print("Saved overlay histogram:", p_overlay)

# ============================================================
# SUMMARY
# ============================================================
print("\nSUMMARY")
print("-------")
print("Ancient galaxies:", int(ancient_mask.sum()))
print("Relic galaxies:", int(relic_mask.sum()))
print("Ancient centrals:", int(np.sum(ancient_mask & df["is_central"])))
print("Ancient satellites:", int(np.sum(ancient_mask & (~df["is_central"]))))
print("Relic centrals:", int(np.sum(relic_mask & df["is_central"])))
print("Relic satellites:", int(np.sum(relic_mask & (~df["is_central"]))))
print("Done.")


# ============================================================
# SIGMA FOR PHASE-SPACE DIAGRAM
# ============================================================

sigma_sel = np.full(sel.size, np.nan, dtype=np.float32)

if os.path.exists(sigma_path):
    with h5py.File(sigma_path, "r") as f:
        ds = f[
            "/ExclusiveSphere/HalfMassRadiusStars/"
            "StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"
        ]

        rows = np.asarray(ds[sel, :], dtype=np.float32)
        sigma_sel[:] = np.sqrt((rows[:, 0]**2 + rows[:, 4]**2 + rows[:, 8]**2)/3)
else:
    print("Sigma file not found; phase-space plot will skip sigma-normalised values.")

# ============================================================
# 3D PHASE-SPACE DIAGRAM FOR MASSIVE RELIC HOST CLUSTERS
# ============================================================

print("\nBuilding 3D phase-space diagram...")

# These are already aligned to the selected sample
gal_pos = np.asarray(centre_of_mass, dtype=np.float32)[sel][valid_dor]
gal_vel = np.asarray(centre_of_mass_vel, dtype=np.float32)[sel][valid_dor]

# host_halo_index was already reduced earlier to sel[valid_dor]
host_idx = np.asarray(host_halo_index, dtype=np.int64).ravel()
sigma_host = sigma_sel[valid_dor]

# keep only entries where host indices and sigma are valid
valid_phase = (
    (host_idx >= 0) &
    (host_idx < fof_masses.size) &
    (host_idx < fof_centres.shape[0]) &
    (host_idx < fof_radii.size) &
    np.isfinite(sigma_host)
)

gal_pos = gal_pos[valid_phase]
gal_vel = gal_vel[valid_phase] 
host_idx = host_idx[valid_phase]
sigma_host = sigma_host[valid_phase]
dor_phase = dor[valid_phase]
compactness_phase = compactness[valid_phase]

host_pos = np.asarray(fof_centres, dtype=np.float32)[host_idx]
host_r200 = np.asarray(fof_radii, dtype=np.float32)[host_idx] 
Mhalo = np.asarray(fof_masses, dtype=np.float64)[host_idx]
full_vel = np.asarray(centre_of_mass_vel, dtype=np.float32)
host_vel = full_vel[host_idx]
#Halo position as central subhalo position instead of fof_centre
# full_pos = np.asarray(centre_of_mass, dtype=np.float32)
# full_vel = np.asarray(centre_of_mass_vel, dtype=np.float32)

# host_pos = full_pos[host_idx]
# host_vel = full_vel[host_idx]

# host_r200 = np.asarray(fof_radii, dtype=np.float32)[host_idx]

print("host_idx min/max:", np.min(host_idx), np.max(host_idx))
print("full_vel shape:", full_vel.shape)

test = np.linalg.norm(gal_vel[:20] - full_vel[host_idx[:20]], axis=1)

print("example relative speeds:", test)
print("median relative speed:", np.nanmedian(test))

r_vec = gal_pos - host_pos
v_vec = gal_vel - host_vel
r_mag = np.linalg.norm(r_vec, axis=1)
G = 4.30091e-6 # kpc (km/s)^2 Msun^-1
V200 = np.sqrt(G * Mhalo / host_r200)

with np.errstate(invalid="ignore", divide="ignore"):
    vr = np.sum(v_vec * r_vec, axis=1) / r_mag
    r_over_r200 = r_mag / host_r200
    vr_over_sigma = vr / V200 #sigma_host

relics = (
    (dor_phase > EXTREME_DOR) &
    (compactness_phase > COMPACTNESS_CUT) &
    np.isfinite(vr_over_sigma) &
    np.isfinite(r_over_r200)
)

relic_host_ids = np.unique(host_idx[relics])

host_mass_unique = []
for hid in relic_host_ids:
    if 0 <= hid < fof_masses.size:
        host_mass_unique.append((hid, fof_masses[hid]))

host_mass_unique = sorted(host_mass_unique, key=lambda x: x[1], reverse=True)

MASS_THRESHOLD = 1e14  # Msun
top_hosts = [
    hid
    for hid, mass in host_mass_unique
    if mass > MASS_THRESHOLD
]
print("Hosts above 1e14 Msun:", len(top_hosts))

for hid in top_hosts:
    print(
        hid,
        np.log10(fof_masses[hid])
    )
print("Top host IDs:", top_hosts)

# # colouring individual clusters 
# # fig, ax = plt.subplots(figsize=(7, 6))
# # #colors = ['C0', 'C1', 'C2', 'C3', 'C4']

# # for i, hid in enumerate(top_hosts):
# #     cluster_mask = (
# #         (host_idx == hid) &
# #         np.isfinite(r_over_r200) &
# #         np.isfinite(vr_over_sigma)
# #     )

# #     relic_mask = cluster_mask & relics

# #     ax.scatter(
# #         r_over_r200[cluster_mask],
# #         vr_over_sigma[cluster_mask],
# #         s=8,
# #         alpha=0.25,
# #         #color = colors[i % len(colors)],
# #         rasterized=True
# #     )

# #     ax.scatter(
# #         r_over_r200[relic_mask],
# #         vr_over_sigma[relic_mask],
# #         s=120,
# #         marker='*',
# #         edgecolor='k',
# #         linewidth=0.8,
# #         #color = colors[i % len(colors)],
# #         label="SRG" #f"Cluster {i+1}"
# #     )

# # ax.axhline(0, linestyle='--', color='black', alpha=0.7)
# # ax.axvline(1.0, linestyle=':', color='black', alpha=0.7)

# # ax.set_xlabel(r"$r / R_{200}$")
# # ax.set_ylabel(r"$v_r / \sigma_{\rm host}$")
# # ax.set_xlim(0, 1.5)
# # ax.grid(True, alpha=0.3)
# # ax.legend(fontsize=8)

# # plt.tight_layout()

# # outname = os.path.join(outdir, "phase_space_relic_clusters.png")
# # plt.savefig(outname, dpi=250)
# # plt.close()

# # print("Saved:", outname)

# # same colour for all clusters
# fig, ax = plt.subplots(figsize=(7, 6))

# # fixed colours
# dot_color = "tab:blue"
# star_color = "tab:orange"   # or "tab:green" if you prefer

# # plot all cluster members in the same colour
# for hid in top_hosts:
#     cluster_mask = (
#         (host_idx == hid) &
#         np.isfinite(r_over_r200) &
#         np.isfinite(vr_over_sigma)
#     )

#     relic_mask = cluster_mask & relics

#     ax.scatter(
#         r_over_r200[cluster_mask],
#         vr_over_sigma[cluster_mask],
#         s=8,
#         alpha=0.25,
#         color=dot_color,
#         rasterized=True
#     )

#     # label only the first relic-star group so the legend has one entry
#     ax.scatter(
#         r_over_r200[relic_mask],
#         vr_over_sigma[relic_mask],
#         s=120,
#         marker='*',
#         facecolor=star_color,
#         edgecolor='k',
#         linewidth=0.8,
#         zorder=5,
#         label="SRG" if hid == top_hosts[0] else "_nolegend_"
#     )

# ax.axhline(0, linestyle='--', color='black', alpha=0.7)
# ax.axvline(1.0, linestyle=':', color='black', alpha=0.7)

# ax.set_xlabel(r"$r / R_{200}$")
# ax.set_ylabel(r"$v_r / \sigma_{\rm host}$")
# ax.set_xlim(0, 2)
# ax.grid(True, alpha=0.3)
# ax.legend(fontsize=8)

# plt.tight_layout()

# outname = os.path.join(outdir, "phase_space_relic_clusters.png")
# plt.savefig(outname, dpi=250)
# plt.close()

# print("Saved:", outname)

fig, ax = plt.subplots(figsize=(7, 6))

dot_color = "tab:blue"
star_color = "tab:orange"

# store all blue points for the contour calculation
blue_x_all = []
blue_y_all = []

for hid in top_hosts:
    cluster_mask = (
        (host_idx == hid) &
        np.isfinite(r_over_r200) &
        np.isfinite(vr_over_sigma)
    )

    relic_mask = cluster_mask & relics

    x = r_over_r200[cluster_mask]
    y = vr_over_sigma[cluster_mask]

    ax.scatter(
        x,
        y,
        s=8,
        alpha=0.25,
        color=dot_color,
        rasterized=True
    )

    blue_x_all.append(x)
    blue_y_all.append(y)

    ax.scatter(
        r_over_r200[relic_mask],
        vr_over_sigma[relic_mask],
        s=120,
        marker='*',
        facecolor=star_color,
        edgecolor='k',
        linewidth=0.8,
        zorder=5,
        label="SRG" if hid == top_hosts[0] else "_nolegend_"
    )

# --- add 50/68/95% contours for the blue points ---
blue_x = np.concatenate(blue_x_all)
blue_y = np.concatenate(blue_y_all)

good = np.isfinite(blue_x) & np.isfinite(blue_y)
blue_x = blue_x[good]
blue_y = blue_y[good]

print(np.max(blue_x))
print(np.percentile(blue_x, [95, 99, 99.9]))

if blue_x.size > 20:
    xgrid = np.linspace(0, 2.0, 250)
    ygrid = np.linspace(np.nanmin(blue_y), np.nanmax(blue_y), 250)
    X, Y = np.meshgrid(xgrid, ygrid)

    kde = gaussian_kde(np.vstack([blue_x, blue_y]))
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # convert density values to enclosed-mass contour levels
    def density_level_for_fraction(Z, frac):
        z = Z.ravel()
        z_sorted = np.sort(z)[::-1]
        cdf = np.cumsum(z_sorted)
        cdf /= cdf[-1]
        return z_sorted[np.searchsorted(cdf, frac)]

    lev50 = density_level_for_fraction(Z, 0.50)
    lev68 = density_level_for_fraction(Z, 0.68)
    lev95 = density_level_for_fraction(Z, 0.95)

    ax.contour(
        X, Y, Z,
        levels=sorted([lev95, lev68, lev50]),
        colors=["0.75", "0.55", "0.35"],
        linewidths=1.2
    )

ax.axhline(0, linestyle='--', color='black', alpha=0.7)
ax.axvline(1.0, linestyle=':', color='black', alpha=0.7)

ax.set_xlabel(r"$r / R_{200}$")
ax.set_ylabel(r"$v_r / \sigma_{\rm host}$")
ax.set_xlim(0, 2)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()

outname = os.path.join(outdir, "phase_space_relic_clusters.png")
plt.savefig(outname, dpi=250)
plt.close()

print("Saved:", outname)