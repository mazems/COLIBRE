#!/usr/bin/env python3
"""
mass_size_plane_with_exsitu.py

Same as your original mass_size_plane.py but colour-codes the mass-size scatter
by ex-situ mass fraction (loaded from an HDF5 table).

Produces:
 - plots/mass_size_z0.0_proj_exsitu.png   (mass-size scatter coloured by ex-situ)
 - plots/mass_size_z0.0_proj.png          (unchanged base plot)
 - plots/ucmg_exsitu_hist.png             (histogram of matched ex-situ fractions)
"""
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
import scipy.spatial as ssp
from scipy.spatial import cKDTree as KDTree, Delaunay
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

# Define Minimal pure-Python 2D LOESS function
# LOESS utilities (complete, self-contained)
import numpy as np
from scipy.spatial import cKDTree

def polyfit_2d(x, y, z, degree=1, weights=None):
    """
    Weighted 2D polynomial fit returning coefficients.
    For degree==0 returns [a0] (constant).
    For degree==1 returns [a0, ax, ay] corresponding to model:
        z ~ a0 + ax*(x - x0) + ay*(y - y0)
    Note: subtracting a local center improves conditioning outside caller.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    if weights is None:
        W = np.ones_like(z, dtype=float)
    else:
        W = np.asarray(weights).ravel()

    # centre coordinates to improve conditioning (choose mean of inputs)
    xc = np.average(x, weights=W)
    yc = np.average(y, weights=W)
    dx = x - xc
    dy = y - yc

    if degree == 0:
        # weighted constant fit
        sw = W.sum()
        if sw == 0:
            return np.array([np.nan])
        a0 = (W @ z) / sw
        return np.array([a0])
    elif degree == 1:
        A = np.column_stack((np.ones_like(dx), dx, dy))  # (N,3)
        # normal equations with weights: (A^T W A) beta = A^T W z
        # compute robustly:
        ATW = (A.T * W)
        ATA = ATW @ A
        ATy = ATW @ z
        # tiny ridge if ill-conditioned
        ridge = 1e-12 * np.trace(ATA) if np.isfinite(np.trace(ATA)) and np.trace(ATA) != 0 else 1e-12
        try:
            ATA[0, 0] += ridge
            beta = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            # fallback to pseudo-inverse
            beta = np.linalg.pinv(ATA) @ ATy
        # beta corresponds to offset around xc,yc; return coefficients such that
        # value at center (x0=0,y0=0 in dx,dy) is beta[0].
        return np.array(beta)  # [a0, ax, ay]
    else:
        raise NotImplementedError("Only degree 0 or 1 supported")

def _biweight_scale(resid):
    """
    Robust scale estimator: MAD -> approximate sigma using 1.4826 for Gaussian.
    If all zeros or insufficient points return small positive number.
    """
    resid = np.abs(resid)
    if resid.size == 0:
        return 1.0
    mad = np.median(resid)
    if mad <= 0:
        # fallback to small scale
        return 1e-9
    return 1.4826 * mad

def loess_2d(x1, y1, z, frac=0.5, degree=1, rescale=False, npoints=None, sigz=None,
             xout=None, yout=None):
    """
    LOESS 2D with robust biweight reweighting.
    - x1,y1,z: 1D data arrays
    - frac: fraction of dataset to use as local neighbourhood (if npoints None)
    - degree: 0 or 1 polynomial locally
    - sigz: optional measurement errors on z (for robust step)
    - xout,yout: optional target coords to predict (1D arrays or None).
         If None -> predictions at the original input coordinates.
    Returns:
      zout, wout
      zout: predicted z at target points (1D array)
      wout: the biweight value for the fitted centre (1D array; diagnostic)
    Notes:
      Use xout,yout as flattened meshgrid if you want a grid (reshape afterwards).
    """
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

    # choose prediction points
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

    # Build KD-tree on data coords
    tree = cKDTree(np.column_stack((x1, y1)))

    for j, (xx, yy) in enumerate(zip(xout, yout)):
        # nearest npoints neighbours
        dists, inds = tree.query([xx, yy], k=npoints)
        # ensure shapes: if k==1, make arrays
        if np.isscalar(dists):
            dists = np.array([dists])
            inds = np.array([inds])
        # use tricube distance weights
        rmax = np.max(dists)
        if rmax == 0:
            # prediction exactly at a data point: return its z
            zout[j] = z[inds[0]]
            wout[j] = 1.0
            continue

        u = dists / rmax
        # tricube weight: (1 - u^3)^3 but set weight zero where u>=1
        distWeights = (1.0 - u**3)**3
        distWeights = np.where(u >= 1.0, 0.0, distWeights)

        # initial fit using just distance weights
        xw = x1[inds]
        yw = y1[inds]
        zw = z[inds]
        w_init = distWeights.copy()

        # weighted least squares local fit
        coeffs = polyfit_2d(xw, yw, zw, degree=degree, weights=w_init)
        # compute fitted values at neighbours
        if degree == 0:
            zfit = np.full_like(zw, coeffs[0], dtype=float)
        else:
            # need to evaluate at neighbours using same centering as polyfit_2d
            # polyfit_2d used weighted mean-centre: recover center from xw,yw weighted by w_init
            xc = np.average(xw, weights=w_init)
            yc = np.average(yw, weights=w_init)
            dx = xw - xc
            dy = yw - yc
            a0, ax, ay = coeffs
            zfit = a0 + ax * dx + ay * dy

        # Iterative robust biweight scheme (Cleveland 1979)
        biWeights = np.ones_like(zw)
        for it in range(10):
            # residual-based scale
            if sigz is None:
                resid = zfit - zw
                scale = _biweight_scale(resid)
                # compute uu as in the snippet: (|resid|/(6*madt))^2
                uu = (np.abs(resid) / (6.0 * scale)) ** 2.0
            else:
                # if measurement errors known
                uu = ((zfit - zw) / (4.0 * sigz[inds])) ** 2.0
            uu = np.clip(uu, 0.0, 1.0)
            biWeights_new = (1.0 - uu) ** 2.0
            totWeights = distWeights * biWeights_new

            # re-fit with combined weights
            coeffs = polyfit_2d(xw, yw, zw, degree=degree, weights=totWeights)
            # recompute zfit
            if degree == 0:
                zfit = np.full_like(zw, coeffs[0], dtype=float)
            else:
                xc = np.average(xw, weights=totWeights) if np.sum(totWeights) > 0 else np.mean(xw)
                yc = np.average(yw, weights=totWeights) if np.sum(totWeights) > 0 else np.mean(yw)
                dx = xw - xc
                dy = yw - yc
                a0, ax, ay = coeffs
                zfit = a0 + ax * dx + ay * dy

            # check convergence of biweights (where they indicate outliers)
            # optionally detect outliers: bad = np.where(biWeights_new < 0.34)[0]
            if np.allclose(biWeights, biWeights_new, atol=1e-6):
                biWeights = biWeights_new
                break
            biWeights = biWeights_new

        # final predicted value at (xx,yy) is intercept term evaluated at that center:
        if degree == 0:
            zout[j] = coeffs[0]
        else:
            # value at the centre (dx=0,dy=0) is the 'intercept' returned by polyfit_2d
            # but polyfit_2d returns coefficients defined around the weighted mean center.
            zout[j] = coeffs[0]

        # record the biweight of the centre as diagnostic
        wout[j] = biWeights[0] if biWeights.size > 0 else 1.0

    return zout, wout

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
# h5path = '/mnt/su3ctm/kproctor/ForMax/L0200N3008_exsitu_summary.hdf5'

from pathlib import Path

# automatically find the ex-situ file
base_dir = Path("/mnt/su3ctm/kproctor/ForMax")
matches = sorted(base_dir.glob("*exsitu*summary*.hdf5"))

if len(matches) == 0:
    raise FileNotFoundError(f"No ex-situ HDF5 file found in {base_dir}")
elif len(matches) == 1:
    h5path = str(matches[0])
else:
    # if there are several, pick the newest one
    h5path = str(max(matches, key=lambda p: p.stat().st_mtime))

print("Using ex-situ file:", h5path)

# Select z=0 snapshot as before
snap_file = snap_files[0]
ztarget = zstarget[0]
comov_to_physical_length = 1.0 / (1.0 + ztarget)

# ---------- read SOAP/virtual snapshot properties (keeps your original code) -
fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')}
fields = {'ExclusiveSphere/50kpc': ('StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'CentreOfMass', 'MassWeightedMeanStellarAge', 'LuminosityWeightedMeanStellarAge')}
fields_proj = {'ProjectedAperture/50kpc/projz': ('StellarMass', 'HalfMassRadiusStars')}

h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
h5data_groups_proj = common.read_group_data_colibre(model_dir, snap_file, fields_proj)

(m30, sfr30, r50, cp, stellarage, stellarage_lum) = h5data_groups
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
is_central_in = is_central[select]
r50_in = r50[select]
r50_in_proj = r50_proj[select]
stellarage_in = stellarage[select]
stellarage_lum_in = stellarage_lum[select]
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

# ---------- load ex-situ fractions from HDF5 and build lookup ---------------
exsitu_lookup = {}
if os.path.exists(h5path):
    with h5py.File(h5path, 'r') as fh:
        if 'stars' in fh:
            data = np.array(fh['stars'])
            if data.ndim == 2 and data.shape[1] >= 4:
                halo_ids_in_file = data[:, 0].astype(int) #track_ids_in_file = data[:, 0].astype(int)
                fractions_in_file = data[:, 3].astype(float)
                exsitu_lookup = dict(zip(halo_ids_in_file.tolist(), fractions_in_file.tolist())) #exsitu_lookup = dict(zip(track_ids_in_file.tolist(), fractions_in_file.tolist()))
                print(f"Loaded {len(halo_ids_in_file)} ex-situ entries from {h5path}")
            else:
                print("HDF5 'stars' dataset shape unexpected; will skip ex-situ matching.")
        else:
            print("HDF5 file missing dataset 'stars'; skipping ex-situ matching.")
else:
    print("Ex-situ HDF5 not found at:", h5path)
    exsitu_lookup = {}

# build ex-situ fraction array aligned with mask_positive
halo_selected = sgn_in[mask_positive].astype(np.int64)   # HaloCatalogueIndex
exsitu_series = pd.Series(exsitu_lookup, dtype=float)
exsitu_fracs = exsitu_series.reindex(halo_selected).to_numpy(dtype=float)
# track_selected = track_id_in[mask_positive]
# exsitu_fracs = np.full(track_selected.shape, np.nan, dtype=float)
# for i, tid in enumerate(track_selected):
#     if int(tid) in exsitu_lookup:
#         exsitu_fracs[i] = float(exsitu_lookup[int(tid)])
# report match stats
n_matched = np.isfinite(exsitu_fracs).sum()
print(f"Matched ex-situ fraction for {n_matched} / {len(exsitu_fracs)} selected galaxies")
print("exsitu_fracs finite:", np.isfinite(exsitu_fracs).sum(), "out of", len(exsitu_fracs)) #check finiteness of ex-situ data
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

# ---------- main mass-size scatter (unchanged) ------------------------------
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 14
})
plt.figure(figsize=(8,6))
plt.scatter(log_m, log_r, alpha=0.7, s=10, label=f"Simulated galaxies at z={ztarget}")
# threshold line (Barro)
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
outpath_base = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_proj.png")
plt.savefig(outpath_base, dpi=300, bbox_inches='tight')
print("Saved base mass-size plot:", outpath_base)
plt.close()

# # ---------- mass-size coloured by ex-situ fraction (LOESS at points -> grid interpolation) ----------
plt.figure(figsize=(8,6))
cmap = plt.get_cmap("viridis")

# # prepare data: only finite ex-situ values
# finite_mask = np.isfinite(exsitu_fracs)
# xvals_all = log_m              # ALL plotting coords (for convex-hull mask)
# yvals_all = log_r
# xvals = log_m[finite_mask]     # coords with finite z
# yvals = log_r[finite_mask]
# zvals = exsitu_fracs[finite_mask]

# # fallback: nothing to smooth -> plain scatter
# if zvals.size == 0:
#     sc = plt.scatter(log_m, log_r, c=exsitu_fracs, cmap=cmap, vmin=0.0, vmax=1.0,
#                      alpha=0.85, s=18, edgecolors='none')
#     if np.any(~np.isfinite(exsitu_fracs)):
#         plt.scatter(log_m[~np.isfinite(exsitu_fracs)], log_r[~np.isfinite(exsitu_fracs)],
#                     color=(0.6,0.6,0.6), alpha=0.5, s=10, label='no ex-situ data')

# else:
#     # ---------------- Tunable parameters ----------------
#     frac_loess = 0.10        # LOESS neighbourhood fraction (0.15-0.30 typical)
#     max_eval_pts = 12000     # <=: how many LOESS fits to compute (None => use all)
#     degree = 1               # local plane (1) or constant (0)
#     nx, ny = 200, 140        # grid resolution (increase for finer pixels, slower)
#     hull_pad_factor = 1.2    # how permissive to be when masking by distance
#     # ----------------------------------------------------

#     N = xvals.size

#     # if dataset very large, subsample the data used for LOESS to limit runtime
#     if (max_eval_pts is not None) and (N > int(max_eval_pts)):
#         rng = np.random.default_rng(seed=12345)
#         sel_idx = rng.choice(N, size=int(max_eval_pts), replace=False)
#         x_loess = xvals[sel_idx]
#         y_loess = yvals[sel_idx]
#         z_loess = zvals[sel_idx]
#     else:
#         x_loess = xvals; y_loess = yvals; z_loess = zvals

#     # call your loess_2d implementation (returns zout, wout)
#     try:
#         zout_pts, wout = loess_2d(x_loess, y_loess, z_loess,
#                                   frac=frac_loess, degree=degree, npoints=None)
#     except TypeError:
#         # fallback if loess_2d has different arg order: try positional style
#         try:
#             zout_pts, wout = loess_2d(x_loess, y_loess, z_loess, frac_loess, degree, False, None, None)
#         except Exception as e:
#             raise RuntimeError("loess_2d call failed; check signature. Err: " + str(e))

#     # build plotting grid over the full plotting bbox (so mask uses full range)
#     xg = np.linspace(np.nanmin(xvals_all), np.nanmax(xvals_all), nx)
#     yg = np.linspace(np.nanmin(yvals_all), np.nanmax(yvals_all), ny)
#     Xg, Yg = np.meshgrid(xg, yg)

#     # interpolate LOESS estimates from the evaluated points onto the grid
#     pts = np.column_stack((x_loess, y_loess))
#     Zgrid = griddata(pts, zout_pts, (Xg, Yg), method='linear')  # linear keeps outside hull = NaN

#     # mask grid cells that are too far from real data (so outside stays white)
#     try:
#         tree_all = KDTree(np.column_stack((xvals_all, yvals_all)))
#         grid_pts = np.column_stack((Xg.ravel(), Yg.ravel()))
#         d_grid, _ = tree_all.query(grid_pts, k=1)
#         d_grid = d_grid.reshape(Xg.shape)

#         # typical spacing from data -> threshold
#         d_data, _ = tree_all.query(np.column_stack((xvals_all, yvals_all)), k=2)
#         if d_data.ndim == 2 and d_data.shape[1] >= 2:
#             typical_spacing = float(np.nanpercentile(d_data[:,1], 95))
#         else:
#             typical_spacing = float(np.nanmedian(d_grid))

#         d_thresh = max(typical_spacing * hull_pad_factor, 1e-6)
#         Zgrid_masked = np.array(Zgrid, copy=True)
#         Zgrid_masked[d_grid > d_thresh] = np.nan
#     except Exception:
#         # if KDTree/Delaunay fails for any reason, trust griddata NaNs
#         Zgrid_masked = Zgrid

#     # compute color limits robustly from the original zvals
#     try:
#         vmin = float(np.nanpercentile(zvals, 1))
#         vmax = float(np.nanpercentile(zvals, 99))
#         if vmin == vmax:
#             vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))
#     except Exception:
#         vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))

#     # plot LOESS surface (masked) with pcolormesh
#     im = plt.pcolormesh(Xg, Yg, Zgrid_masked, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

#     # optionally overlay small faint markers of the evaluated LOESS points
#     plt.scatter(x_loess, y_loess, s=3, c='k', alpha=0.06, linewidths=0)

#     # overlay missing points in grey (unchanged)
#     if finite_mask.sum() < len(exsitu_fracs):
#         missing_idx = ~finite_mask
#         plt.scatter(log_m[missing_idx], log_r[missing_idx],
#                     color=(0.6,0.6,0.6), alpha=0.5, s=10, label='no ex-situ data')

#     cbar = plt.colorbar(im)
#     cbar.set_label(fr'$f_\text{ex-situ}$')

# # draw compactness threshold (foreground)
# plt.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#          linestyle='--', color='black', label=fr'Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {logsigma_ref}$)')

# plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
# plt.ylabel(r"lg(Half Mass Radius / kpc)")
# plt.title("Mass-size relation coloured by ex-situ mass fraction (LOESS)")
# plt.legend(fontsize=8)
# plt.grid(True)

# outpath_exsitu = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_proj_exsitu_loess.png")
# plt.savefig(outpath_exsitu, dpi=300, bbox_inches='tight')
# plt.close()
# print("Saved ex-situ LOESS plot:", outpath_exsitu)

# --- LOESS-coloured ex-situ mass fraction mass–size plot (Mg/Fe-consistent version) ---

SHOW_MISSING = True

exsitu_aligned = exsitu_fracs.copy()

have_mask = np.isfinite(exsitu_aligned)
missing_mask = ~have_mask
n_have = int(have_mask.sum())
n_missing = int(missing_mask.sum())
total_plot = int(len(exsitu_aligned))

print(f"DEBUG ex-situ (LOESS block): have={n_have}, missing={n_missing}, total_plot={total_plot}")

fig, ax = plt.subplots(figsize=(8,6))

if n_have == 0:
    if SHOW_MISSING:
        ax.scatter(log_m, log_r, s=10, alpha=0.7,
                   color="lightgrey", label="no ex-situ data")
    else:
        ax.scatter(log_m, log_r, s=10, alpha=0.7, label="galaxies")

else:
    # points used for LOESS
    xvals = log_m[have_mask]
    yvals = log_r[have_mask]
    zvals = exsitu_aligned[have_mask]

    # grid tightly around the data used for LOESS (IDENTICAL to Mg/Fe)
    pad_x = 0.05 * (np.nanmax(xvals) - np.nanmin(xvals) + 1e-6)
    pad_y = 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-6)
    nx, ny = 300, 220

    xg = np.linspace(np.nanmin(xvals) - pad_x,
                     np.nanmax(xvals) + pad_x, nx)
    yg = np.linspace(np.nanmin(yvals) - pad_y,
                     np.nanmax(yvals) + pad_y, ny)
    Xg, Yg = np.meshgrid(xg, yg)
    pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

    # distance-based mask using SAME LOESS input points
    tree_data = KDTree(np.column_stack((xvals, yvals)))
    d_grid, _ = tree_data.query(pts_grid, k=1)

    d_data, _ = tree_data.query(np.column_stack((xvals, yvals)), k=2)
    if d_data.ndim == 2 and d_data.shape[1] >= 2:
        typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
    else:
        typical_spacing = float(np.nanmedian(d_grid))

    d_thresh = max(typical_spacing * 1.3, 1e-6)
    inside_mask = (d_grid <= d_thresh)
    idx_inside = np.nonzero(inside_mask)[0]

    if idx_inside.size > 0:
        xout = pts_grid[idx_inside, 0]
        yout = pts_grid[idx_inside, 1]

        # LOESS parameters — SAME as Mg/Fe
        frac_loess = 0.01
        degree = 1

        Zflat_inside, _ = loess_2d(
            xvals, yvals, zvals,
            frac=frac_loess, degree=degree,
            xout=xout, yout=yout
        )

        Zflat = np.full(pts_grid.shape[0], np.nan)
        Zflat[idx_inside] = Zflat_inside
        Zgrid = Zflat.reshape((ny, nx))
        Zmask = np.ma.masked_invalid(Zgrid)

        # color limits (robust, but clamp to [0,1])
        try:
            vmin = float(np.nanpercentile(zvals, 5))
            vmax = float(np.nanpercentile(zvals, 95))
        except Exception:
            vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))

        vmin = max(0.0, vmin)
        vmax = min(1.0, vmax)

        cmap = plt.get_cmap("viridis")
        im = ax.pcolormesh(Xg, Yg, Zmask,
                           shading='auto', cmap=cmap,
                           vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Ex-situ mass fraction")

        # faint markers showing evaluated LOESS points
        ax.scatter(xout, yout, s=1, c='k', alpha=0.05, linewidths=0)

    else:
        ax.scatter(xvals, yvals, c=zvals,
                   cmap='viridis', s=12, edgecolors='none')

    if SHOW_MISSING and n_missing > 0:
        ax.scatter(log_m[missing_mask], log_r[missing_mask],
                   color="lightgrey", s=8, alpha=0.6,
                   label="no ex-situ data")

# compactness threshold (foreground)
ax.plot(np.log10(stellar_masses),
        (2/3)*(np.log10(stellar_masses) - logsigma_ref),
        linestyle='--', color='black',
        label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
ax.legend(fontsize=8)
ax.grid(True)

outpath_exsitu = os.path.join(
    outdir, f"mass_size_z{ztarget:.1f}_exsitu_loess.png"
)
fig.savefig(outpath_exsitu, dpi=300, bbox_inches='tight')
plt.close(fig)

print("Saved ex-situ LOESS plot (Mg/Fe-consistent):", outpath_exsitu)
# ---------------------------------------------------------------------------------------------------------

# ----------------- Hexbin plot for ex-situ fraction (minimal paste) -----------------
finite_mask_exsitu = np.isfinite(exsitu_fracs)
x_h = log_m[finite_mask_exsitu]
y_h = log_r[finite_mask_exsitu]
c_h = exsitu_fracs[finite_mask_exsitu]

fig, ax = plt.subplots(figsize=(8,6))

if c_h.size == 0:
    # nothing to bin -> fallback to scatter (grey where missing)
    ax.scatter(log_m, log_r, color=(0.7,0.7,0.7), s=10, alpha=0.7, label='no ex-situ data')
else:
    # robust vmin/vmax (use percentiles to avoid outliers dominating)
    try:
        vmin_h = float(np.nanpercentile(c_h, 2))
        vmax_h = float(np.nanpercentile(c_h, 98))
        if vmin_h == vmax_h:
            vmin_h, vmax_h = 0.0, 1.0
    except Exception:
        vmin_h, vmax_h = 0.0, 1.0

    hb = ax.hexbin(
        x_h, y_h, C=c_h,
        gridsize=80,                    # adjust for coarser/finer bins
        reduce_C_function=np.nanmean,   # mean ex-situ fraction per hex
        mincnt=3,                       # hide bins with fewer points
        cmap='viridis', vmin=vmin_h, vmax=vmax_h
    )
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Ex-situ mass fraction")

    # optionally show points with missing ex-situ as light grey dots
    miss = ~finite_mask_exsitu
    if np.any(miss):
        ax.scatter(log_m[miss], log_r[miss], color='lightgrey', s=6, alpha=0.6, label='no ex-situ data')

# draw compactness threshold as in other plots
ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
        linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
ax.set_title("Mass–size relation coloured by ex-situ mass fraction (hexbin)")
ax.legend(fontsize=8)
ax.grid(True)

outpath_hex_exsitu = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_exsitu_hexbin.png")
fig.savefig(outpath_hex_exsitu, dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved ex-situ hexbin plot:", outpath_hex_exsitu)
# -------------------------------------------------------------------------------------

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

# ----- Mg/Fe coloured mass-size plane using loess_2d above -----
SHOW_MISSING = True   # set False to hide missing Mg/Fe galaxies

# Align mgfe to plotting order (unchanged)
mgfe_series = pd.Series(mgfe_abund_all, index=subids_all)
mgfe_aligned = mgfe_series.reindex(subids_plot).to_numpy(dtype=float)

have_mask = np.isfinite(mgfe_aligned)
missing_mask = ~have_mask
n_have = int(have_mask.sum())
n_missing = int(missing_mask.sum())

print(f"DEBUG Mg/Fe: have={n_have}, missing={n_missing}, total_plot={len(subids_plot)}")

fig, ax = plt.subplots(figsize=(8,6))

if n_have == 0:
    # nothing to LOESS: show grey points or plain scatter
    if SHOW_MISSING:
        ax.scatter(log_m, log_r, s=18, alpha=0.8, color="lightgrey", label="no Mg/Fe")
    else:
        ax.scatter(log_m, log_r, s=10, alpha=0.7, label="galaxies")
else:
    # Data to use for LOESS
    xvals = log_m[have_mask]
    yvals = log_r[have_mask]
    zvals = mgfe_aligned[have_mask]

    # Grid resolution (tune if needed)
    nx, ny = 300, 220
    xg = np.linspace(np.nanmin(log_m), np.nanmax(log_m), nx)
    yg = np.linspace(np.nanmin(log_r), np.nanmax(log_r), ny)
    Xg, Yg = np.meshgrid(xg, yg)           # shapes: (ny, nx)
    pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

    # Compute convex hull of the data points -> polygon
    # If there are <3 points, fall back to simple nearest-distance mask.
    if xvals.size >= 3:
        try:
            hull = ConvexHull(np.column_stack((xvals, yvals)))
            hull_verts = np.column_stack((xvals, yvals))[hull.vertices]
            hull_path = Path(hull_verts)
            inside_mask = hull_path.contains_points(pts_grid)   # boolean length ny*nx
        except Exception:
            # convex hull could fail on degenerate data; fallback to nearest-dist mask
            tree_tmp = cKDTree(np.column_stack((xvals, yvals)))
            dists_grid, _ = tree_tmp.query(pts_grid, k=1)
            # choose threshold e.g. max neighbour distance of the data (k-th neighbor for frac)
            # use median nearest-neighbor distance of data
            ddata, _ = tree_tmp.query(np.column_stack((xvals, yvals)), k=2)
            med_nn = np.median(ddata[:, 1])
            inside_mask = (dists_grid <= (2.0 * med_nn))
    else:
        # For 1-2 points use distance-based mask
        tree_tmp = cKDTree(np.column_stack((xvals, yvals)))
        dists_grid, _ = tree_tmp.query(pts_grid, k=1)
        ddata, _ = tree_tmp.query(np.column_stack((xvals, yvals)), k=2 if xvals.size>1 else 1)
        med_nn = np.median(ddata[:, -1]) if ddata.ndim>1 else np.median(ddata)
        inside_mask = (dists_grid <= (2.0 * med_nn if med_nn>0 else 1e-6))

    # Only predict at grid points that lie inside convex hull (avoid extrapolation)
    idx_inside = np.nonzero(inside_mask)[0]
    if idx_inside.size > 0:
        xout = pts_grid[idx_inside, 0]
        yout = pts_grid[idx_inside, 1]

        # LOESS parameters - tweak frac if smoothing too strong/weak
        frac_loess = 0.01
        # call loess_2d: must accept xout,yout and return (zout, wout)
        Zflat_inside, Wflat = loess_2d(xvals, yvals, zvals, frac=frac_loess, degree=1,
                                       xout=xout, yout=yout)

        # Build full grid and fill predicted values only at inside points; outside remain NaN
        Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
        Zflat[idx_inside] = Zflat_inside
        Zgrid = Zflat.reshape((ny, nx))
        Zmask = np.ma.masked_invalid(Zgrid)   # mask NaNs -> pcolormesh will leave them blank

        # color limits from data distribution
        try:
            vmin = float(np.nanpercentile(zvals, 5))
            vmax = float(np.nanpercentile(zvals, 95))
        except Exception:
            vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            med = float(np.nanmedian(zvals))
            span = max(0.3, 0.5 * max(1e-6, abs(med)))
            vmin = med - span
            vmax = med + span

        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Plot LOESS surface only inside hull (white elsewhere)
        im = ax.pcolormesh(Xg, Yg, Zmask, shading='auto', cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("[Mg/Fe] (dex)")

    else:
        # nothing inside hull (very unlikely) -> fallback to scatter
        ax.scatter(xvals, yvals, c=zvals, cmap='viridis', s=18, edgecolors='none')

    # overlay data points for reference (use same colormap / limits)
    # sc = ax.scatter(xvals, yvals, c=zvals, cmap='viridis', norm=norm, s=12, edgecolors='none', label="[Mg/Fe] present")

    # overlay missing points as grey markers on top if requested
    if SHOW_MISSING and n_missing > 0:
        ax.scatter(log_m[missing_mask], log_r[missing_mask], color="lightgrey", s=8, alpha=0.6, label="no Mg/Fe")

# Draw compactness threshold last (foreground)
ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
        linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
ax.set_title("Mass–size plane coloured by [Mg/Fe] (LOESS)")
ax.legend(fontsize=8)
ax.grid(True)

outpath_MgFe = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_MgFe.png")
fig.savefig(outpath_MgFe, dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved Mg/Fe-coloured mass-size plot (LOESS inside hull):", outpath_MgFe)