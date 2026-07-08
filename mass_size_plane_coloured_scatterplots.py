#!/usr/bin/env python3

from __future__ import annotations
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Normalize
import numpy as np
import os
import common
import pandas as pd
import h5py

###### choose what kind of profiles you want (choose one only) ##################
family_method = 'radial_profiles'
method = 'circular_apertures_face_on_map'
#method = 'circular_apertures_random_map'
#method = 'spherical_apertures'

################## select the model and redshift you want #######################
model_name = 'L0200N3008/THERMAL_AGN/'
model_dir = '/mnt/su3-pro/colibre/' + model_name

#definitions correspond to z=0
snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0026', '0018']
zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

###################### units ##################################################
Lu = 3.086e+24/(3.086e+24) #cMpc -> Mpc factor (kept 1 for clarity)
Mu = 1.988e+43/(1.989e+33) # SOAP raw mass -> Msun
tu = 3.086e+19/(3.154e+7)  # time unit -> yr

# ---------- ex-situ HDF5 (change path if needed) -----------------------------
# The HDF5 is expected to have dataset 'stars' with shape (N,4)
# and columns: [track_id (int), mass_ex, mass_tot, fraction]
h5path = '/mnt/su3ctm/kproctor/ForMax/L0200N3008_exsitu_summary.hdf5'

# Select z=0 snapshot as before
snap_file = snap_files[0]
ztarget = zstarget[0]
comov_to_physical_length = 1.0 / (1.0 + ztarget)

# ---------- read SOAP/virtual snapshot properties (keeps your original code) -
fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')}
fields = {'ExclusiveSphere/50kpc': ('StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'CentreOfMass', 'MassWeightedMeanStellarAge', 'LuminosityWeightedMeanStellarAge', 'StellarMassFractionInMetals')}
fields_proj = {'ProjectedAperture/50kpc/projz': ('StellarMass', 'HalfMassRadiusStars')}

h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
h5data_groups_proj = common.read_group_data_colibre(model_dir, snap_file, fields_proj)

(m30, sfr30, r50, cp, stellarage, stellarage_lum, Zstar_raw) = h5data_groups
(m30_proj, r50_proj) = h5data_groups_proj
(sgn, is_central, desc_id, track_id) = h5data_idgroups

# unit conversions (unchanged)
m30 = m30 * Mu
m30_proj = m30_proj * Mu
sfr30 = sfr30 * Mu / tu
r50 = r50 * Lu * comov_to_physical_length * 1e3
r50_proj = r50_proj * Lu * comov_to_physical_length * 1e3
stellarage = stellarage * tu / 1e9
stellarage_lum = stellarage_lum * tu / 1e9
cp = cp * Lu * comov_to_physical_length

Zsun = 0.0134   # AGSS09 convention
# Zsun = 0.0139 # Asplund et al. 2021 present-day photospheric value
    
Zstar = np.asarray(Zstar_raw, dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    logZstar = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar), np.nan)
    logZstar_rel = np.where((Zstar > 0) & np.isfinite(Zstar),
                            np.log10(Zstar / Zsun),
                            np.nan)

# basic alignment check
n = len(m30)
assert len(sgn) == n and len(is_central) == n and len(track_id) == n, "Input arrays have mismatched lengths!"

# ---------- select galaxies (same selection as original) ---------------------
select = np.where(m30 >= 1e9)
ngals = len(m30[select])

outdir = os.path.join(os.getcwd(), "plots")
os.makedirs(outdir, exist_ok=True)

if ngals == 0:
    print("No galaxies selected; skipping plots.")
    raise SystemExit(0)

# selected arrays
m_in = m30[select]
m_in_proj = m30_proj[select]
sfr_in = sfr30[select]
sgn_in = sgn[select]            # subhalo id (HaloCatalogueIndex/Snap id)
is_central_in = is_central[select] #if int needed: is_central_in = (is_central[select].astype(int))
r50_in = r50[select]
r50_in_proj = r50_proj[select]
stellarage_in = stellarage[select]
stellarage_lum_in = stellarage_lum[select]
Zstar_in = Zstar[select]
logZstar_in = logZstar[select]
logZstar_rel_in = logZstar_rel[select]
x_in = cp[select][:,0]
y_in = cp[select][:,1]
z_in = cp[select][:,2]
desc_id_in = desc_id[select]
track_id_in = track_id[select]  # IMPORTANT: this is the 'track id' that maps to ex-situ HDF5

# ---------- compute mass / radius logs (same as original) -------------------
mask_positive = (m_in > 0) & (r50_in > 0)
if not np.any(mask_positive):
    raise RuntimeError("No positive mtot/r50 values to plot after filtering selection.")

log_m = np.log10(m_in[mask_positive])
log_r = np.log10(r50_in[mask_positive])
sfr_plot = sfr_in[mask_positive]
Zstar_in = Zstar_in[mask_positive]
logZstar_in = logZstar_in[mask_positive]
logZstar_rel_in = logZstar_rel_in[mask_positive]

# ---------- load ex-situ fractions from HDF5 and build lookup ---------------
exsitu_lookup = {}
if os.path.exists(h5path):
    with h5py.File(h5path, 'r') as fh:
        if 'stars' in fh:
            data = np.array(fh['stars'])
            if data.ndim == 2 and data.shape[1] >= 4:
                track_ids_in_file = data[:, 0].astype(int)
                fractions_in_file = data[:, 3].astype(float)
                exsitu_lookup = dict(zip(track_ids_in_file.tolist(), fractions_in_file.tolist()))
                print(f"Loaded {len(track_ids_in_file)} ex-situ entries from {h5path}")
            else:
                print("HDF5 'stars' dataset shape unexpected; will skip ex-situ matching.")
        else:
            print("HDF5 file missing dataset 'stars'; skipping ex-situ matching.")
else:
    print("Ex-situ HDF5 not found at:", h5path)
    exsitu_lookup = {}

# build ex-situ fraction array aligned with mask_positive
track_selected = track_id_in[mask_positive]
exsitu_fracs = np.full(track_selected.shape, np.nan, dtype=float)
for i, tid in enumerate(track_selected):
    if int(tid) in exsitu_lookup:
        exsitu_fracs[i] = float(exsitu_lookup[int(tid)])
# report match stats
n_matched = np.isfinite(exsitu_fracs).sum()
print(f"Matched ex-situ fraction for {n_matched} / {len(exsitu_fracs)} selected galaxies")

# ----- Compute Mg/Fe from full data table ----
csv_in = "relicness_merged_with_stellar_complete_updated.csv"
df_all = pd.read_csv(csv_in)

# require columns
required_cols = ("subhalo_id", "elem_Mg_mass", "elem_Fe_mass")
for c in required_cols:
    if c not in df_all.columns:
        raise RuntimeError(f"Missing column '{c}' in {csv_in}")

# numeric safe extraction
Mg = pd.to_numeric(df_all["elem_Mg_mass"], errors="coerce").to_numpy(dtype=float)
Fe = pd.to_numeric(df_all["elem_Fe_mass"], errors="coerce").to_numpy(dtype=float)
subids_all = df_all["subhalo_id"].astype(int).to_numpy()

# atomic weights (amu)
A_Mg = 24.305
A_Fe = 55.845
log_MgFe_sun = +0.10   # Asplund+2009

# mass -> number ratio (N_Mg/N_Fe) = (M_Mg / M_Fe) * (A_Fe / A_Mg)
with np.errstate(divide="ignore", invalid="ignore"):
    MgFe_number_all = (Mg / Fe) * (A_Fe / A_Mg)
    log10_number_all = np.where(MgFe_number_all > 0, np.log10(MgFe_number_all), np.nan)
    # [Mg/Fe] (dex) = log10(N_Mg/N_Fe) - log10(N_Mg/N_Fe)_sun
    mgfe_abund_all = log10_number_all - log_MgFe_sun

# make lookup subhalo_id -> abundance
mgfe_lookup = dict(zip(subids_all, mgfe_abund_all))

# align with plotted galaxies (same order as log_m/log_r)
subids_plot = sgn_in[mask_positive].astype(int)
mgfe_vals = np.array([mgfe_lookup.get(int(sid), np.nan) for sid in subids_plot], dtype=float)

# take array of finite values only for color scaling
finite_mask = np.isfinite(mgfe_vals)
n_have = finite_mask.sum()
print(f"Mg/Fe available for {n_have} / {len(mgfe_vals)} galaxies")

# --------------------------------------------------------------
# LOAD HOST VELOCITY DISPERSION (CORRECT + ALIGNED)
# --------------------------------------------------------------
sigma_path = "/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN/SOAP-HBT/extra/halo_properties_0127.hdf5"

# mask of galaxies you actually want to plot
mask_positive_full = (m30 >= 1e9) & (m30 > 0) & (r50 > 0)

# row positions in the SOAP catalogue
row_idx = np.flatnonzero(mask_positive_full)

# allocate full-length array if you want to keep SOAP alignment
sigma_full = np.full(m30.shape, np.nan, dtype=np.float32)

sigma_path = "/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN/SOAP-HBT/extra/halo_properties_0127.hdf5"
sigma_ds = "/ExclusiveSphere/3xHalfMassRadiusStars/StellarCylindricalVelocityDispersionVerticalLuminosityWeighted" #"/ExclusiveSphere/HalfMassRadiusStars/StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"

if os.path.exists(sigma_path):
    with h5py.File(sigma_path, "r") as f:
        ds = f[sigma_ds]
        print("sigma dataset shape:", ds.shape)

        # read only the selected rows
        rows = np.asarray(ds[row_idx, :], dtype=np.float32)   # shape (N, 9)

        # diagonal components of the 3x3 tensor
        sigma_rr   = rows[:, 0]
        sigma_pphi = rows[:, 4]
        sigma_zz   = rows[:, 8]

        # your requested scalar sigma
        sigma_sel = np.sqrt((sigma_rr**2 + sigma_pphi**2 + sigma_zz**2)/3)

        # put back into full SOAP-aligned array
        sigma_full[row_idx] = sigma_sel

        # log sigma for plotting
        log_sigma_full = np.full(m30.shape, np.nan, dtype=np.float32)
        log_sigma_full[row_idx] = np.where(sigma_sel > 0, np.log10(sigma_sel), np.nan)

    print("Loaded sigma values:", np.isfinite(sigma_sel).sum(), "/", sigma_sel.size)
    print("N(sigma == 0):", np.count_nonzero(np.isclose(sigma_sel[np.isfinite(sigma_sel)], 0.0)))
else:
    print("Sigma file not found.")

sigma_vals = sigma_full[mask_positive_full]
log_sigma_vals = log_sigma_full[mask_positive_full]

# ---------- main mass-size scatter (unchanged) ------------------------------
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 14
})
plt.figure(figsize=(8,6))
plt.scatter(log_m, log_r, alpha=0.7, s=10, label=f"Simulated galaxies at z={ztarget}")
# threshold line
stellar_masses = np.logspace(9, 12, 100)
logsigma_ref = 9.72
plt.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
         linestyle='--', color='black', label=fr'Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {logsigma_ref}$)')
plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
plt.ylabel(r"lg(Half Mass Radius / kpc)")
plt.title("Mass-size relation (COLIBRE 200m6)")
plt.legend(fontsize=8)
plt.grid(True)
plt.tick_params(axis='both', labelsize=12, direction='in', length=6, width=1)

# save original (unchanged) mass-size plot
outpath_base = os.path.join(outdir, f"mass_size_z{ztarget:.1f}.png")
plt.savefig(outpath_base, dpi=300, bbox_inches='tight')
print("Saved base mass-size plot:", outpath_base)
plt.close()

# ---------- mass-size coloured by ex-situ fraction --------------------------
plt.figure(figsize=(8,6))
# prepare colour map: use 0..1 range; missing values plotted in light grey
cmap = plt.get_cmap("viridis")
# compute reasonable vmin/vmax from finite values if any, else default 0..1
finite_mask = np.isfinite(exsitu_fracs)
if finite_mask.sum() > 0:
    vmin = float(np.nanpercentile(exsitu_fracs[finite_mask], 1))
    vmax = float(np.nanpercentile(exsitu_fracs[finite_mask], 99))
    if vmin == vmax:
        vmin, vmax = 0.0, 1.0
else:
    vmin, vmax = 0.0, 1.0

# scatter points with colour; plot missing as grey on top for visibility
sc = plt.scatter(log_m, log_r, c=exsitu_fracs, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.85, s=18, edgecolors='none')
# overlay grey markers for missing values so they are visible
if finite_mask.sum() < len(exsitu_fracs):
    missing_idx = ~finite_mask
    plt.scatter(log_m[missing_idx], log_r[missing_idx], color=(0.6,0.6,0.6), alpha=0.5, s=10, label='no ex-situ data')

plt.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
         linestyle='--', color='black', label=fr'Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {logsigma_ref}$)')
plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
plt.ylabel(r"lg(Half Mass Radius / kpc)")
plt.title("Mass-size relation coloured by ex-situ mass fraction")
plt.legend(fontsize=8)
plt.grid(True)
cbar = plt.colorbar(sc)
cbar.set_label("Ex-situ mass fraction")
outpath_exsitu = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_exsitu_scattered.png")
plt.savefig(outpath_exsitu, dpi=300, bbox_inches='tight')
plt.close()
print("Saved ex-situ coloured mass-size plot:", outpath_exsitu)

# ---------- histogram of matched ex-situ fractions (same as your previous script) ----------
if finite_mask.sum() > 0:
    plt.figure(figsize=(6,4))
    plt.hist(exsitu_fracs[finite_mask], bins=20, edgecolor='black')
    plt.xlabel("Ex-situ mass fraction")
    plt.ylabel("Number of galaxies")
    plt.title("Sample: ex-situ mass fractions (matched)")
    plt.tight_layout()
    hist_out = os.path.join(outdir, "ucmg_exsitu_hist.png")
    plt.savefig(hist_out, dpi=150)
    plt.close()
    print("Saved ex-situ histogram:", hist_out)
else:
    print("No ex-situ fractions available to plot histogram.")

# ------ Plot: Mass-size relation coloured by log(Mg/Fe) -----
SHOW_MISSING = True   # set False to hide missing Mg/Fe galaxies

# Ensure mgfe_vals is aligned with plotted arrays:
mgfe_series = pd.Series(mgfe_abund_all, index=subids_all)
# reindex to plotting order (this yields NaN where missing)
mgfe_aligned = mgfe_series.reindex(subids_plot).to_numpy(dtype=float)

have_mask = np.isfinite(mgfe_aligned)
missing_mask = ~have_mask
n_have = int(have_mask.sum())
n_missing = int(missing_mask.sum())
total_plot = len(subids_plot)

print(f"DEBUG Mg/Fe: have={n_have}, missing={n_missing}, total_plot={total_plot}")

# optionally show a few example missing/available ids for quick debugging
if n_missing > 0:
    example_missing = subids_plot[missing_mask][:10]
    print("Example missing subhalo_ids (up to 10):", example_missing.tolist())
if n_have > 0:
    example_have = subids_plot[have_mask][:10]
    print("Example have subhalo_ids (up to 10):", example_have.tolist())

fig, ax = plt.subplots(figsize=(8,6))

if n_have == 0:
    # No Mg/Fe values at all -> plot everything uncoloured (or grey if SHOW_MISSING)
    if SHOW_MISSING:
        ax.scatter(log_m, log_r, s=18, alpha=0.8, color="lightgrey", label="no Mg/Fe")
    else:
        ax.scatter(log_m, log_r, s=10, alpha=0.7, label="galaxies")
else:
    # robust color scaling from percentiles of the finite values
    vals = mgfe_aligned[have_mask]
    try:
        vmin = float(np.nanpercentile(vals, 5))
        vmax = float(np.nanpercentile(vals, 95))
    except Exception:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))

    # fallback if vmin==vmax (avoid zero-range)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        med = float(np.nanmedian(vals))
        span = max(0.3, 0.5 * max(1e-6, abs(med)))
        vmin = med - span
        vmax = med + span

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=vmin, vmax=vmax)

    # plot points WITH Mg/Fe
    sc = ax.scatter(
        log_m[have_mask],
        log_r[have_mask],
        c=vals,
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.9,
        edgecolors='none',
        label="[Mg/Fe] present"
    )

    # overlay missing points in grey on top if requested
    if SHOW_MISSING and n_missing > 0:
        ax.scatter(
            log_m[missing_mask],
            log_r[missing_mask],
            color="lightgrey",
            s=18,
            alpha=0.8,
            label="no Mg/Fe",
            edgecolors='none'
        )

    # colorbar (attach to the scatter)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("[Mg/Fe] (dex, Asplund+2009)")

# compactness threshold drawn last so it stays in the foreground
ax.plot(
    np.log10(stellar_masses),
    (2/3)*(np.log10(stellar_masses) - logsigma_ref),
    linestyle='--',
    color='black',
    label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)'
)

ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
ax.set_title("Mass–size plane coloured by [Mg/Fe]")
ax.legend(fontsize=8)
ax.grid(True)

outpath_MgFe = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_MgFe(UCMGs)_scattered.png")
fig.savefig(outpath_MgFe, dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved Mg/Fe-coloured mass-size plot:", outpath_MgFe)

# ---------- mass-size coloured by luminosity weighted age --------------------------
plt.figure(figsize=(8,6))
# prepare colour map: use 0..1 range; missing values plotted in light grey
cmap = plt.get_cmap("viridis")
# compute reasonable vmin/vmax from finite values if any, else default 0..1
finite_mask = np.isfinite(stellarage_lum_in)
if finite_mask.sum() > 0:
    vmin = float(np.nanpercentile(stellarage_lum_in[finite_mask], 1))
    vmax = float(np.nanpercentile(stellarage_lum_in[finite_mask], 99))
    if vmin == vmax:
        vmin, vmax = 0.0, 1.0
else:
    vmin, vmax = 0.0, 1.0

# scatter points with colour; plot missing as grey on top for visibility
sc = plt.scatter(log_m, log_r, c=stellarage_lum_in, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.85, s=18, edgecolors='none')
# overlay grey markers for missing values so they are visible
if finite_mask.sum() < len(stellarage_lum_in):
    missing_idx = ~finite_mask
    plt.scatter(log_m[missing_idx], log_r[missing_idx], color=(0.6,0.6,0.6), alpha=0.5, s=10, label='no age data')

plt.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
         linestyle='--', color='black', label=fr'Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {logsigma_ref}$)')
plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
plt.ylabel(r"lg(Half Mass Radius / kpc)")
# plt.title("Mass-size relation coloured by luminosity weighted mean stellar age")
plt.legend(fontsize=8)
plt.grid(True)
cbar = plt.colorbar(sc)
cbar.set_label("Luminosity Weighted Mean Stellar Age [Gyr]")
outpath_lumage = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_age(lum)_scattered.png")
plt.savefig(outpath_lumage, dpi=300, bbox_inches='tight')
plt.close()
print("Saved lum-age coloured mass-size plot:", outpath_lumage)

# #------------ mass-size plot lum weighted age (with satellite flag) --------------------------
# plt.figure(figsize=(8,6))

# # masks
# central_mask   = (is_central_in == 1)
# satellite_mask = (is_central_in == 0)

# # colour limits from finite ages
# finite_mask = np.isfinite(stellarage_lum_in)
# if finite_mask.sum() > 0:
#     vmin = float(np.nanpercentile(stellarage_lum_in[finite_mask], 1))
#     vmax = float(np.nanpercentile(stellarage_lum_in[finite_mask], 99))
#     if vmin == vmax:
#         vmin, vmax = 0.0, 1.0
# else:
#     vmin, vmax = 0.0, 1.0

# cmap = plt.get_cmap("viridis")

# # centrals: filled circles
# sc = plt.scatter(
#     log_m[central_mask],
#     log_r[central_mask],
#     c=stellarage_lum_in[central_mask],
#     cmap=cmap,
#     vmin=vmin, vmax=vmax,
#     s=20,
#     marker='o',
#     alpha=0.9,
#     edgecolors='none',
#     label='Centrals'
# )

# # satellites: triangles with black edge
# plt.scatter(
#     log_m[satellite_mask],
#     log_r[satellite_mask],
#     c=stellarage_lum_in[satellite_mask],
#     cmap=cmap,
#     vmin=vmin, vmax=vmax,
#     s=22,
#     marker='^',
#     alpha=0.9,
#     edgecolors='black',
#     linewidths=0.4,
#     label='Satellites'
# )

# # missing ages (optional overlay)
# missing_mask = ~np.isfinite(stellarage_lum_in)
# if missing_mask.sum() > 0:
#     plt.scatter(
#         log_m[missing_mask],
#         log_r[missing_mask],
#         color=(0.6, 0.6, 0.6),
#         s=12,
#         alpha=0.5,
#         marker='x',
#         label='No age'
#     )

# # compactness threshold
# plt.plot(
#     np.log10(stellar_masses),
#     (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#     linestyle='--',
#     color='black',
#     label=fr'Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {logsigma_ref}$)'
# )

# plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
# plt.ylabel(r"lg(Half Mass Radius / kpc)")
# plt.grid(True)

# plt.legend(fontsize=8)
# cbar = plt.colorbar(sc)
# cbar.set_label("Luminosity Weighted Mean Stellar Age [Gyr]")

# outpath_lumage = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_lumage_central_satellite.png")
# plt.savefig(outpath_lumage, dpi=300, bbox_inches='tight')
# plt.close()

# print("Saved mass-size plot with central/satellite flag:", outpath_lumage)

# Compute sSFR [yr^-1] and take log10
with np.errstate(divide="ignore", invalid="ignore"):
    ssfr_plot = np.where(m_in[mask_positive] > 0,
                          sfr_plot / m_in[mask_positive],
                          np.nan)
    log_ssfr_plot = np.where(ssfr_plot > 0,
                              np.log10(ssfr_plot),
                              np.nan)

log_ssfr_aligned = log_ssfr_plot.copy() 

# --- impose lower floor instead of masking ---
SSFR_FLOOR = -12.0

# replace -inf and other non-finite values with floor
log_ssfr_aligned[~np.isfinite(log_ssfr_aligned)] = SSFR_FLOOR

x_h = log_m[finite_mask]
y_h = log_r[finite_mask]
c_h = log_ssfr_aligned[finite_mask]

# ---------- mass-size coloured by ssfr --------------------------
plt.figure(figsize=(8,6))
# prepare colour map: use 0..1 range; missing values plotted in light grey
cmap = plt.get_cmap("viridis")
# compute reasonable vmin/vmax from finite values if any, else default 0..1
finite_mask = np.isfinite(log_ssfr_aligned)
if finite_mask.sum() > 0:
    vmin = float(np.nanpercentile(log_ssfr_aligned[finite_mask], 1))
    vmax = float(np.nanpercentile(log_ssfr_aligned[finite_mask], 99))
    if vmin == vmax:
        vmin, vmax = 0.0, 1.0
else:
    vmin, vmax = 0.0, 1.0

# scatter points with colour; plot missing as grey on top for visibility
sc = plt.scatter(log_m, log_r, c=log_ssfr_aligned, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.85, s=18, edgecolors='none')
# overlay grey markers for missing values so they are visible
if finite_mask.sum() < len(log_ssfr_aligned):
    missing_idx = ~finite_mask
    plt.scatter(log_m[missing_idx], log_r[missing_idx], color=(0.6,0.6,0.6), alpha=0.5, s=10, label='no ssfr data')

plt.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
         linestyle='--', color='black', label=fr'Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {logsigma_ref}$)')
plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
plt.ylabel(r"lg(Half Mass Radius / kpc)")
plt.legend(fontsize=8)
plt.grid(True)
cbar = plt.colorbar(sc)
cbar.set_label(r"$\lg(\mathrm{sSFR}\ /\ \mathrm{yr}^{-1})$")
outpath_ssfr = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_ssfr_scattered.png")
plt.savefig(outpath_ssfr, dpi=300, bbox_inches='tight')
plt.close()
print("Saved ssfr coloured mass-size plot:", outpath_ssfr)

# ---------- mass-size coloured by stellar metallicity --------------------------
plt.figure(figsize=(8,6))
# prepare colour map: use 0..1 range; missing values plotted in light grey
cmap = plt.get_cmap("viridis")
# compute reasonable vmin/vmax from finite values if any, else default 0..1
finite_mask = np.isfinite(logZstar_rel_in)
if finite_mask.sum() > 0:
    vmin = float(np.nanpercentile(logZstar_rel_in[finite_mask], 1))
    vmax = float(np.nanpercentile(logZstar_rel_in[finite_mask], 99))
    if vmin == vmax:
        vmin, vmax = 0.0, 1.0
else:
    vmin, vmax = 0.0, 1.0

# scatter points with colour; plot missing as grey on top for visibility
sc = plt.scatter(log_m, log_r, c=logZstar_rel_in, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.85, s=18, edgecolors='none')
# overlay grey markers for missing values so they are visible
if finite_mask.sum() < len(logZstar_rel_in):
    missing_idx = ~finite_mask
    plt.scatter(log_m[missing_idx], log_r[missing_idx], color=(0.6,0.6,0.6), alpha=0.5, s=10, label='no metallicity data')

plt.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
         linestyle='--', color='black', label=fr'Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {logsigma_ref}$)')
plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
plt.ylabel(r"lg(Half Mass Radius / kpc)")
# plt.title("Mass-size relation coloured by luminosity weighted mean stellar age")
plt.legend(fontsize=8)
plt.grid(True)
cbar = plt.colorbar(sc)
cbar.set_label(r"$\lg[Z / H]$")
outpath_zstar = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_logZrel_scattered.png")
plt.savefig(outpath_zstar, dpi=300, bbox_inches='tight')
plt.close()
print("Saved metallicity coloured mass-size plot:", outpath_zstar)

# ---------- mass-size coloured by velocity dispersion --------------------------
plt.figure(figsize=(8,6))
# prepare colour map: use 0..1 range; missing values plotted in light grey
cmap = plt.get_cmap("viridis")
# compute reasonable vmin/vmax from finite values if any, else default 0..1
finite_mask = np.isfinite(log_sigma_vals)
if finite_mask.sum() > 0:
    vmin = float(np.nanpercentile(log_sigma_vals[finite_mask], 1))
    vmax = float(np.nanpercentile(log_sigma_vals[finite_mask], 99))
    if vmin == vmax:
        vmin, vmax = 0.0, 1.0
else:
    vmin, vmax = 0.0, 1.0

# scatter points with colour; plot missing as grey on top for visibility
sc = plt.scatter(log_m, log_r, c=log_sigma_vals, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.85, s=18, edgecolors='none')
# overlay grey markers for missing values so they are visible
if finite_mask.sum() < len(log_sigma_vals):
    missing_idx = ~finite_mask
    plt.scatter(log_m[missing_idx], log_r[missing_idx], color=(0.6,0.6,0.6), alpha=0.5, s=10, label='no age data')

plt.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
         linestyle='--', color='black', label=fr'Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {logsigma_ref}$)')
plt.xlabel(r"$\lg(M_\star / M_{\odot})$")
plt.ylabel(r"$\lg(R_{1/2} / \mathrm{kpc})$")
# plt.title("Mass-size relation coloured by luminosity weighted mean stellar age")
plt.legend(fontsize=8)
plt.grid(True)
cbar = plt.colorbar(sc)
cbar.set_label(r'$\lg(\sigma / \mathrm{km}\ \mathrm{s}^{-1})$')
outpath_logsigma = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_sigma_scattered.png")
plt.savefig(outpath_logsigma, dpi=300, bbox_inches='tight')
plt.close()
print("Saved logsigma coloured mass-size plot:", outpath_logsigma)