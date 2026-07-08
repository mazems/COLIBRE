
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import common
import h5py
import pandas as pd
from matplotlib.path import Path
from scipy.spatial import cKDTree as KDTree, ConvexHull
from scipy.interpolate import griddata
# import cmasher as cmr
#import utilities_statistics as us

# LOESS helpers
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

def loess_2d(x1, y1, z, frac=0.1, degree=1, rescale=False, npoints=None, sigz=None,
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
    tree = KDTree(np.column_stack((x1, y1)))

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

#family_method = 'grid'
#method = 'grid_face_on_map' #to be coded
#method = 'grid_random_map'
#method = 'voronoi_maps'  #to be coded

#################################################################################

################## select the model and redshift you want #######################
#model_name = 'L0100N0752/Thermal_non_equilibrium/'
#model_name = 'L0050N0752/Thermal_non_equilibrium/'
#model_name = 'L0025N0376/Thermal/'
#model_name = 'L0200N1504/THERMAL_AGN/'
model_name = 'L0200N3008/THERMAL_AGN/'
model_dir = '/mnt/su3-pro/colibre/' + model_name

#definitions below correspond to z=0
snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0026', '0018']
zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

#snap_files = ['0102', '0092', '0076', '0064', '0056', '0048', '0040', '0032', '0026', '0018']
#zstarget = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]

#snap_files = ['0056', '0048', '0040', '0026', '0018']
#zstarget = [4.0, 5.0, 6.0, 8.0, 10.0]

#snap_files = ['0123', '0088', '0072', '0060', '0048', '0040'] #, '0026', '0020']
#zstarget = [0.0, 1.0, 2.0, 3.5, 4.0, 5.0, 6.0] #, 8.0, 10.0]

#################################################################################
###################### simulation units #########################################
Lu = 3.086e+24/(3.086e+24) #cMpc
Mu = 1.988e+43/(1.989e+33) #Msun
tu = 3.086e+19/(3.154e+7) #yr
Tempu = 1 #K
density_cgs_conv = 6.767905773162602e-31 #conversion from simulation units to CGS for density
mH = 1.6735575e-24 #in gr
#################################################################################

#define radial bins of interest. This going from 0 to 50kpc, in bins of 1kpc
rmax = 50
rmin = 0
dr = 1.0
rbins = np.arange(rmin, rmax, dr)
xr = rbins + dr/2.0 
nr = len(xr) #number of radial bins

gmax = 50
gmin = -50
dg = 1.0
gbins = np.arange(gmin, gmax, dg)
gr = gbins + dg/2.0 
ng = len(gr) #number of radial bins

if(family_method == 'grid'):
    nr  = ng * ng
  

##### loop through redshifts ######
#for z in range(0,len(snap_files)):
#    snap_file =snap_files[z]
#    ztarget = zstarget[z]
#    comov_to_physical_length = 1.0 / (1.0 + ztarget)
# Select only the z = 0 snapshot
snap_file = snap_files[0]   # '0127' corresponds to z=0.0
ztarget = zstarget[0]       # ensure ztarget is a single value
comov_to_physical_length = 1.0 / (1.0 + ztarget)
    ################# read galaxy properties #########################################
    #fields_fof = /SOAP-HBT/HostHaloIndex, 
    #/InputHalos/HBTplus/HostFOFId
fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')} 
fields ={'ExclusiveSphere/50kpc': ('StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'CentreOfMass', 'MassWeightedMeanStellarAge', 'LuminosityWeightedMeanStellarAge', 'LinearMassWeightedIronOverHydrogenOfStars', 'LinearMassWeightedMagnesiumOverHydrogenOfStars', 'StellarMassFractionInMetals')}
fields_proj = {'ProjectedAperture/50kpc/projz': ('StellarMass', 'HalfMassRadiusStars')}
h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
h5data_groups_proj = common.read_group_data_colibre(model_dir, snap_file, fields_proj)
(m30, sfr30, r50, cp, stellarage, stellarage_lum, FeoverH, MgoverH, Zstar_raw) = h5data_groups
(m30_proj, r50_proj) = h5data_groups_proj

soap_id = {'SOAP': ('HostHaloIndex',)}
h5data_soap = common.read_group_data_colibre(model_dir, snap_file, soap_id)
(host_halo_index) = h5data_soap

#unit conversion
# mdust = (mdustl + mdusts) * Mu
m30 = m30 * Mu
m30_proj = m30_proj * Mu
# mHI = mHI * Mu
# mH2 = mH2 * Mu
sfr30 = sfr30 * Mu / tu 
r50 = r50 * Lu * comov_to_physical_length * 1e3 #convert in kpc
r50_proj = r50_proj * Lu * comov_to_physical_length * 1e3 #convert in kpc
stellarage = stellarage * tu / 1e9 #in Gyr
stellarage_lum = stellarage_lum * tu / 1e9 #in Gyr
cp = cp * Lu * comov_to_physical_length
# Jstars = Jstars * Mu / (Lu * comov_to_physical_length)**2 / tu

Zsun = 0.0134   # AGSS09 convention
# Zsun = 0.0139 # Asplund et al. 2021 present-day photospheric value
    
Zstar = np.asarray(Zstar_raw, dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    logZstar = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar), np.nan)
    logZstar_rel = np.where((Zstar > 0) & np.isfinite(Zstar),
                            np.log10(Zstar / Zsun),
                            np.nan)
  
(sgn, is_central, desc_id, track_id) = h5data_idgroups
xg = cp[:,0]
yg = cp[:,1]
zg = cp[:,2]
    ###################################################################################


    ######################### select galaxies of interest #############################
select = np.where(m30 >=1e9)
ngals = len(m30[select])
if(ngals > 0):
    print("Number of galaxies of interest", ngals, " at redshift", ztarget)
    m_in = m30[select]
    m_in_proj = m30_proj[select]
    sfr_in = sfr30[select]
    sgn_in = sgn[select]
    is_central_in = is_central[select]
    r50_in = r50[select]
    r50_in_proj = r50_proj[select]
    # mHI_in = mHI[select]
    # mH2_in = mH2[select]
    # kappacostar_in = kappacostar[select]
    # kappacogas_in = kappacogas[select]
    # disctotot_in = disctotot[select]
    stellarage_in = stellarage[select]
    stellarage_lum_in = stellarage_lum[select]
    FeoverH_in = FeoverH[select]
    MgoverH_in = MgoverH[select]
    Zstar_in = Zstar[select]
    logZstar_in = logZstar[select]
    logZstar_rel_in = logZstar_rel[select]
    # ZgasLow_in = ZgasLow[select]
    # ZgasHigh_in = ZgasHigh[select]
    x_in = xg[select]
    y_in = yg[select]
    z_in = zg[select]
    # Jstars_in = Jstars[select, :]
    # Jstars_in = Jstars_in[0]
    # Jstars_in_norm = np.sqrt(Jstars_in[:,0]**2 + Jstars_in[:,1]**2 + Jstars_in[:,2]**2)
    # mdust_in = mdust[select]
   
    #save galaxy properties of interest
    gal_props = np.zeros(shape = (ngals, 19))
    gal_props[:,0] = sgn_in
    gal_props[:,1] = is_central_in
    gal_props[:,2] = x_in
    gal_props[:,3] = y_in
    gal_props[:,4] = z_in
    gal_props[:,5] = m_in
    gal_props[:,6] = m_in_proj
    gal_props[:,7] = sfr_in
    gal_props[:,8] = r50_in
    gal_props[:,9] = r50_in_proj
    # gal_props[:,8] = mHI_in
    # gal_props[:,9] = mH2_in
    # gal_props[:,10] = kappacostar_in
    # gal_props[:,11] = kappacogas_in
    # gal_props[:,12] = disctotot_in
    # gal_props[:,13] = Jstars_in_norm
    gal_props[:,8] = stellarage_in
    gal_props[:,9] = stellarage_lum_in
    gal_props[:,10] = FeoverH_in
    gal_props[:,11] = MgoverH_in
    gal_props[:,12] = Zstar_in
    gal_props[:,13] = logZstar_in 
    gal_props[:,14] = logZstar_rel_in
    # gal_props[:,15] = ZgasLow_in
    # gal_props[:,16] = ZgasHigh_in
    # gal_props[:,17] = mdust_in
    gal_props[:,15] = desc_id[select]
    gal_props[:,16] = track_id[select]
    out_dir = "/home/mzemsch/COLIBRE-analysis/ProcessedData"
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(
        os.path.join(out_dir, f"GalaxyProperties_z{ztarget}.txt"),
        gal_props
    )
    np.savetxt('/home/mzemsch/COLIBRE-analysis/ProcessedData/GalaxyProperties_z' + str(ztarget) + '.txt', gal_props)

#Compact galaxy relation
stellar_masses = np.logspace(9, 12, 100)
a = 2/3
logsigma = 8.0 
logsigma_ref = 9.72 #10.0 cut by eye
effective_radii = (stellar_masses/(10**(logsigma)))**a

#Mg/Fe computation

A_Mg = 24.305
A_Fe = 55.845
log_MgFe_sun = +0.10
Mg = np.asarray(MgoverH_in, dtype=float)
Fe = np.asarray(FeoverH_in, dtype=float)

with np.errstate(divide="ignore", invalid="ignore"):
    MgFe_number = (Mg / Fe) # * (A_Fe / A_Mg)
    log10_number = np.where(MgFe_number > 0, np.log10(MgFe_number), np.nan)
    mgfe = log10_number - log_MgFe_sun

# ---- Use selected arrays produced above ----
# m_in and r50_in exist only if ngals > 0; otherwise handle gracefully
if ngals > 0:
    mt_plot = m_in
    r50_plot = r50_in
    mt_proj = m_in_proj
    r50_proj = r50_in_proj
    

    # filter non-positive values
    mask = (mt_plot > 0) & (r50_plot > 0)
    if not np.any(mask):
        raise RuntimeError("No positive mtot/r50 values to plot after filtering selection.")

    log_m = np.log10(mt_plot[mask])
    log_r = np.log10(r50_plot[mask])
    ucmgs = log_m - 3/2 * log_r - logsigma_ref
    print(f"Number of UCMGs: {np.sum(ucmgs > 0)}")
    ucmg_ids = sgn_in[ucmgs > 0]        # assuming sgn_in holds the subhalo IDs
    print("Saving", len(ucmg_ids), "UCMG IDs to CSV")
    # pd.DataFrame({"subhalo_id": ucmg_ids}).to_csv("ucmg_ids.csv", index=False) #writes ucmg_ids csv file

    mask_proj = (mt_proj > 0) & (r50_proj > 0)
    if not np.any(mask_proj):
        raise RuntimeError("No positive mtot/r50 values to plot after filtering selection.")

    log_m_proj = np.log10(mt_proj[mask_proj])
    log_r_proj = np.log10(r50_proj[mask_proj])
    mgfe_plot = mgfe[mask]
    stellar_lum_plot = stellarage_lum_in[mask]
    stellar_mw_plot = stellarage_in[mask]
    sfr_plot = sfr_in[mask]
    vmin = float(np.nanpercentile(mgfe_plot, 5))
    vmax = float(np.nanpercentile(mgfe_plot, 95))

    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": 14
    })
    plt.figure(figsize=(8,6))
    cmap = plt.get_cmap("viridis")
    sc = plt.scatter(log_m, log_r, alpha=0.7, s=10, c=mgfe, cmap=cmap, vmin=0.1, vmax=0.27, label=f"Simulated galaxies at z={ztarget}")
    # Threshold line Barro
    plt.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref), 
            linestyle='--', color='black', label=fr'Barro et al. (2013) ($\lg{{\Sigma_{{\mathrm{{1.5}}}}}} = {logsigma_ref}$)')
    plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
    plt.ylabel(r"lg(Half Mass Radius / kpc)")
    # plt.title("Mass-size relation (COLIBRE 200m6) with Mg/Fe")
    cbar = plt.colorbar(sc)
    cbar.set_label("Mg/Fe")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=12, direction='in', length=6, width=1)
    plt.show()


    # save plot
    outdir = os.path.join(os.getcwd(), "plots")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_fullMgFe.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print("Saved plot to:", outpath)
    plt.close()
else:
    print("No galaxies selected; skipping plot.")

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
sigma_ds = "/ExclusiveSphere/HalfMassRadiusStars/StellarCylindricalVelocityDispersionLuminosityWeighted" #"/ExclusiveSphere/HalfMassRadiusStars/StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"
sigma2_ds = "/ExclusiveSphere/50kpc/StellarCylindricalVelocityDispersionDiscPlane"

if os.path.exists(sigma_path):
    with h5py.File(sigma_path, "r") as f:
        ds_vert = f[sigma_ds]
        ds_disc = f[sigma2_ds]
        print(ds_vert.shape)
        print(ds_disc.shape)

        # read only the selected rows
        # rows = np.asarray(ds[row_idx, :], dtype=np.float32)   # shape (N, 9)
        rows_vert = np.asarray(ds_vert[row_idx, :], dtype=np.float32)
        rows_disc = np.asarray(ds_disc[row_idx], dtype=np.float32)

        # diagonal components of the 3x3 tensor
        sigma_rr_vert   = rows_vert[:, 0]
        sigma_pphi_vert = rows_vert[:, 4] 
        sigma_zz_vert   = rows_vert[:, 8]

        # sigma_rr_disc = rows_disc[:, 0]
        # sigma_pphi_disc = rows_disc[:, 4]
        # sigma_zz_disc   = rows_disc[:, 8]

        # your requested scalar sigma
        sigma_sel = np.sqrt((sigma_rr_vert**2 + sigma_pphi_vert**2 + sigma_zz_vert**2)/3)
        # sigma_sel_disc = np.sqrt((sigma_rr_disc**2 + sigma_pphi_disc**2 + sigma_zz_disc**2)/3)

        #sigma_sel = np.sqrt((2*rows_vert**2 + rows_disc**2)/3)

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


# # --- LOESS-coloured Mg/Fe mass-size plot (self-consistent replacement) ---

# SHOW_MISSING = True   # keep same behaviour as earlier

# # mgfe_plot is aligned with log_m/log_r (mgfe_plot = mgfe[mask])
# mgfe_aligned = mgfe_plot.copy()

# have_mask = np.isfinite(mgfe_aligned)
# missing_mask = ~have_mask
# n_have = int(have_mask.sum())
# n_missing = int(missing_mask.sum())
# total_plot = int(len(mgfe_aligned))

# print(f"DEBUG Mg/Fe (LOESS block): have={n_have}, missing={n_missing}, total_plot={total_plot}")

# fig, ax = plt.subplots(figsize=(8,6))

# if n_have == 0:
#     if SHOW_MISSING:
#         ax.scatter(log_m, log_r, s=10, alpha=0.7, color="lightgrey", label="no Mg/Fe")
#     else:
#         ax.scatter(log_m, log_r, s=10, alpha=0.7, label="galaxies")
# else:
#     # points used for LOESS
#     xvals = log_m[have_mask]
#     yvals = log_r[have_mask]
#     zvals = mgfe_aligned[have_mask]

#     # grid tightly around the data used for LOESS (avoid extrapolating to full plotting bbox)
#     pad_x = 0.05 * (np.nanmax(xvals) - np.nanmin(xvals) + 1e-6)   # small padding
#     pad_y = 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-6)
#     nx, ny = 300, 220
#     xg = np.linspace(np.nanmin(xvals) - pad_x, np.nanmax(xvals) + pad_x, nx)
#     yg = np.linspace(np.nanmin(yvals) - pad_y, np.nanmax(yvals) + pad_y, ny)
#     Xg, Yg = np.meshgrid(xg, yg)
#     pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

#     # Build KDTree on the *same* LOESS input points and compute distance mask
#     tree_data = KDTree(np.column_stack((xvals, yvals)))
#     d_grid, _ = tree_data.query(pts_grid, k=1)
#     # typical spacing: use 95th percentile of 2nd-neighbour distances of the data
#     d_data, _ = tree_data.query(np.column_stack((xvals, yvals)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.nanmedian(d_grid))
#     # threshold: multiplier controls how permissive mask is; adjust if necessary (1.0 - 2.0)
#     d_thresh = max(typical_spacing * 1.3, 1e-6)
#     inside_mask = (d_grid <= d_thresh)

#     idx_inside = np.nonzero(inside_mask)[0]

#     if idx_inside.size > 0:
#         # predict only at grid cells that are within d_thresh of a real data point
#         xout = pts_grid[idx_inside, 0]
#         yout = pts_grid[idx_inside, 1]

#         # LOESS parameters (tweak frac if smoothing too strong)
#         frac_loess = 0.01
#         degree = 1

#         Zflat_inside, Wflat = loess_2d(xvals, yvals, zvals, frac=frac_loess, degree=degree,
#                                        xout=xout, yout=yout)

#         # place predictions back into full grid
#         Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
#         Zflat[idx_inside] = Zflat_inside
#         Zgrid = Zflat.reshape((ny, nx))
#         Zmask = np.ma.masked_invalid(Zgrid)

#         # color limits from zvals (robust)
#         try:
#             vmin = float(np.nanpercentile(zvals, 5))
#             vmax = float(np.nanpercentile(zvals, 95))
#         except Exception:
#             vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))
#         if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#             med = float(np.nanmedian(zvals))
#             span = max(0.2, 0.5 * max(1e-6, abs(med)))
#             vmin = med - span
#             vmax = med + span

#         cmap = plt.get_cmap("viridis")
#         im = ax.pcolormesh(Xg, Yg, Zmask, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
#         cbar = fig.colorbar(im, ax=ax)
#         cbar.set_label("[Mg/Fe]")

#         # faint markers showing evaluated LOESS points (optional)
#         ax.scatter(xout, yout, s=1, c='k', alpha=0.05, linewidths=0)
#     else:
#         # no inside grid points (very unlikely) -> fallback scatter
#         ax.scatter(xvals, yvals, c=zvals, cmap='viridis', s=12, edgecolors='none')

#     # optionally overlay missing points in grey
#     if SHOW_MISSING and n_missing > 0:
#         ax.scatter(log_m[missing_mask], log_r[missing_mask], color="lightgrey", s=8, alpha=0.6, label="no Mg/Fe")

# # draw compactness threshold (foreground)
# ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#         linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

# ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
# ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
# # ax.set_title("Mass–size plane coloured by [Mg/Fe] (LOESS)")
# ax.legend(fontsize=8)
# ax.grid(True)

# outpath_MgFe = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_fullMgFe_loess.png")
# fig.savefig(outpath_MgFe, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved Mg/Fe-coloured mass-size plot (LOESS inside hull):", outpath_MgFe)

# # ----------------- Hexbin version (replace or add after LOESS block) -----------------
# # use only finite mgfe for binning
# finite_mask = np.isfinite(mgfe_plot)
# x_h = log_m[finite_mask]
# y_h = log_r[finite_mask]
# c_h = mgfe_plot[finite_mask]

# if x_h.size == 0:
#     print("No finite Mg/Fe values for hexbin — skipping.")
# else:
#     fig, ax = plt.subplots(figsize=(8,6))

#     # robust vmin/vmax based on your existing percentile logic
#     try:
#         vmin_h = float(np.nanpercentile(c_h, 5))
#         vmax_h = float(np.nanpercentile(c_h, 95))
#     except Exception:
#         vmin_h, vmax_h = float(np.nanmin(c_h)), float(np.nanmax(c_h))
#     if not np.isfinite(vmin_h) or not np.isfinite(vmax_h) or vmin_h == vmax_h:
#         med = float(np.nanmedian(c_h))
#         span = max(0.2, 0.5 * max(1e-6, abs(med)))
#         vmin_h = med - span
#         vmax_h = med + span

#     # hexbin parameters (tweak gridsize for resolution, mincnt to hide sparse bins)
#     gridsize = 80        # try 60-120 depending on number of points and desired resolution
#     mincnt = 3           # bins with fewer than mincnt points will be blank

#     # do the hexbin: reduce_C_function=np.nanmean computes mean [Mg/Fe] per bin
#     hb = ax.hexbin(
#         x_h, y_h, C=c_h,
#         gridsize=gridsize,
#         reduce_C_function=np.nanmean,
#         mincnt=mincnt,
#         cmap='viridis',
#         vmin=vmin_h, vmax=vmax_h,
#         linewidths=0.2,
#         edgecolors='none'
#     )

#     cbar = fig.colorbar(hb, ax=ax)
#     cbar.set_label("[Mg/Fe] (dex)")

#     # overlay missing points in grey if requested
#     if SHOW_MISSING:
#         miss_mask = ~np.isfinite(mgfe_plot)
#         if np.any(miss_mask):
#             ax.scatter(log_m[miss_mask], log_r[miss_mask],
#                        color='lightgrey', s=6, alpha=0.6, label='no Mg/Fe')

#     # draw compactness threshold and labels (same as before)
#     ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#             linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

#     ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
#     ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
#     # ax.set_title("Mass–size plane coloured by [Mg/Fe] (hexbin)")
#     ax.legend(fontsize=8)
#     ax.grid(True)

#     outpath_hex = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_fullMgFe_hexbin.png")
#     fig.savefig(outpath_hex, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print("Saved Mg/Fe-coloured mass-size hexbin:", outpath_hex)

#     # ----------------- Hexbin version (luminosity weighted mean stellar age) -----------------

# finite_mask = np.isfinite(stellar_lum_plot)
# x_h = log_m[finite_mask]
# y_h = log_r[finite_mask]
# c_h = stellar_lum_plot[finite_mask]

# if x_h.size == 0:
#     print("No finite age(lum) values for hexbin — skipping.")
# else:
#     fig, ax = plt.subplots(figsize=(8,6))

#     # robust vmin/vmax based on your existing percentile logic
#     try:
#         vmin_h = float(np.nanpercentile(c_h, 5))
#         vmax_h = float(np.nanpercentile(c_h, 95))
#     except Exception:
#         vmin_h, vmax_h = float(np.nanmin(c_h)), float(np.nanmax(c_h))
#     if not np.isfinite(vmin_h) or not np.isfinite(vmax_h) or vmin_h == vmax_h:
#         med = float(np.nanmedian(c_h))
#         span = max(0.2, 0.5 * max(1e-6, abs(med)))
#         vmin_h = med - span
#         vmax_h = med + span

#     # hexbin parameters (tweak gridsize for resolution, mincnt to hide sparse bins)
#     gridsize = 80        # try 60-120 depending on number of points and desired resolution
#     mincnt = 3           # bins with fewer than mincnt points will be blank

#     # do the hexbin: reduce_C_function=np.nanmean computes mean [Mg/Fe] per bin
#     hb = ax.hexbin(
#         x_h, y_h, C=c_h,
#         gridsize=gridsize,
#         reduce_C_function=np.nanmean,
#         mincnt=mincnt,
#         cmap='viridis',
#         vmin=vmin_h, vmax=vmax_h,
#         linewidths=0.2,
#         edgecolors='none'
#     )

#     cbar = fig.colorbar(hb, ax=ax)
#     cbar.set_label("Age [Gyr]")


#     # draw compactness threshold and labels (same as before)
#     ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#             linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

#     ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
#     ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
#    # ax.set_title("Mass–size plane coloured by mean stellar age (hexbin)")
#     ax.legend(fontsize=8)
#     ax.grid(True)

#     outpath_hex = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_age(lum)_hexbin.png")
#     fig.savefig(outpath_hex, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print("Saved Age(lum)-coloured mass-size hexbin:", outpath_hex)

# # ---------- lum-weighted age (LOESS version filling holes, hull_pad_factor) ----------
# plt.figure(figsize=(8,6))
# cmap = plt.get_cmap("viridis")

# # prepare data: only finite lum-age values
# finite_mask = np.isfinite(stellar_lum_plot)
# xvals_all = log_m              # ALL plotting coords (for convex-hull mask)
# yvals_all = log_r
# xvals = log_m[finite_mask]     # coords with finite z
# yvals = log_r[finite_mask]
# zvals = stellar_lum_plot[finite_mask]

# # fallback: nothing to smooth -> plain scatter
# if zvals.size == 0:
#     sc = plt.scatter(log_m, log_r, c=stellar_lum_plot, cmap=cmap, vmin=0.0, vmax=1.0,
#                      alpha=0.85, s=18, edgecolors='none')
#     if np.any(~np.isfinite(stellar_lum_plot)):
#         plt.scatter(log_m[~np.isfinite(stellar_lum_plot)], log_r[~np.isfinite(stellar_lum_plot)],
#                     color=(0.6,0.6,0.6), alpha=0.5, s=10, label='no lum age data')

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
#     if finite_mask.sum() < len(stellar_lum_plot):
#         missing_idx = ~finite_mask
#         plt.scatter(log_m[missing_idx], log_r[missing_idx],
#                     color=(0.6,0.6,0.6), alpha=0.5, s=10, label='no lum age data')

#     cbar = plt.colorbar(im)
#     cbar.set_label("Age [Gyr]")

# # draw compactness threshold (foreground)
# plt.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#          linestyle='--', color='black', label=fr'Compactness threshold ($\lg{{\Sigma_{{1.5}}}} = {logsigma_ref}$)')

# plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
# plt.ylabel(r"lg(Half Mass Radius / kpc)")
# # plt.title("Mass-size relation coloured by luminosity weighted mean stellar age (LOESS)")
# plt.legend(fontsize=8)
# plt.grid(True)

# outpath_loess = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_age(lum)_loess.png")
# plt.savefig(outpath_loess, dpi=300, bbox_inches='tight')
# plt.close()
# print("Saved lum age LOESS plot:", outpath_loess)

# # --- LOESS-coloured Luminosity-weighted mean stellar age (Mg/Fe-consistent) ---

# SHOW_MISSING = True

# lum_aligned = stellar_lum_plot.copy()

# have_mask = np.isfinite(lum_aligned)
# missing_mask = ~have_mask
# n_have = int(have_mask.sum())
# n_missing = int(missing_mask.sum())
# total_plot = int(len(lum_aligned))

# print(f"DEBUG LumAge (LOESS block): have={n_have}, missing={n_missing}, total_plot={total_plot}")

# fig, ax = plt.subplots(figsize=(8,6))

# if n_have == 0:
#     if SHOW_MISSING:
#         ax.scatter(log_m, log_r, s=10, alpha=0.7, color="lightgrey", label="no lum age")
#     else:
#         ax.scatter(log_m, log_r, s=10, alpha=0.7, label="galaxies")
# else:
#     xvals = log_m[have_mask]
#     yvals = log_r[have_mask]
#     zvals = lum_aligned[have_mask]

#     # grid tightly around the LOESS input points (same as Mg/Fe)
#     pad_x = 0.05 * (np.nanmax(xvals) - np.nanmin(xvals) + 1e-6)
#     pad_y = 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-6)
#     nx, ny = 300, 220
#     xg = np.linspace(np.nanmin(xvals) - pad_x, np.nanmax(xvals) + pad_x, nx)
#     yg = np.linspace(np.nanmin(yvals) - pad_y, np.nanmax(yvals) + pad_y, ny)
#     Xg, Yg = np.meshgrid(xg, yg)
#     pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

#     # KDTree on LOESS input points
#     tree_data = KDTree(np.column_stack((xvals, yvals)))
#     d_grid, _ = tree_data.query(pts_grid, k=1)

#     d_data, _ = tree_data.query(np.column_stack((xvals, yvals)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.nanmedian(d_grid))

#     d_thresh = max(typical_spacing * 1.3, 1e-6)
#     inside_mask = (d_grid <= d_thresh)
#     idx_inside = np.nonzero(inside_mask)[0]

#     if idx_inside.size > 0:
#         xout = pts_grid[idx_inside, 0]
#         yout = pts_grid[idx_inside, 1]

#         frac_loess = 0.01
#         degree = 1

#         Zflat_inside, _ = loess_2d(xvals, yvals, zvals, frac=frac_loess, degree=degree,
#                                    xout=xout, yout=yout)

#         Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
#         Zflat[idx_inside] = Zflat_inside
#         Zgrid = Zflat.reshape((ny, nx))
#         Zmask = np.ma.masked_invalid(Zgrid)

#         try:
#             vmin = float(np.nanpercentile(zvals, 5))
#             vmax = float(np.nanpercentile(zvals, 95))
#         except Exception:
#             vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))
#         if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#             med = float(np.nanmedian(zvals))
#             span = max(0.2, 0.5 * max(1e-6, abs(med)))
#             vmin = med - span
#             vmax = med + span

#         cmap = plt.get_cmap("viridis")
#         im = ax.pcolormesh(Xg, Yg, Zmask, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
#         cbar = fig.colorbar(im, ax=ax)
#         cbar.set_label("Age [Gyr]")

#         ax.scatter(xout, yout, s=1, c='k', alpha=0.05, linewidths=0)
#     else:
#         ax.scatter(xvals, yvals, c=zvals, cmap='viridis', s=12, edgecolors='none')

#     if SHOW_MISSING and n_missing > 0:
#         ax.scatter(log_m[missing_mask], log_r[missing_mask], color="lightgrey", s=8, alpha=0.6, label="no lum age")

# ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#         linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')
# ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
# ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
# ax.legend(fontsize=8)
# ax.grid(True)

# outpath_loess = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_age(lum)_loess.png")
# fig.savefig(outpath_loess, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved lum age LOESS plot (Mg/Fe-consistent):", outpath_loess)

# # ----------------- Hexbin version (total star formation rate) -----------------

# finite_mask = np.isfinite(sfr_plot)
# x_h = log_m[finite_mask]
# y_h = log_r[finite_mask]
# c_h = sfr_plot[finite_mask]

# if x_h.size == 0:
#     print("No finite sfr values for hexbin — skipping.")
# else:
#     fig, ax = plt.subplots(figsize=(8,6))

#     # robust vmin/vmax based on your existing percentile logic
#     try:
#         vmin_h = float(np.nanpercentile(c_h, 5))
#         vmax_h = float(np.nanpercentile(c_h, 95))
#     except Exception:
#         vmin_h, vmax_h = float(np.nanmin(c_h)), float(np.nanmax(c_h))
#     if not np.isfinite(vmin_h) or not np.isfinite(vmax_h) or vmin_h == vmax_h:
#         med = float(np.nanmedian(c_h))
#         span = max(0.2, 0.5 * max(1e-6, abs(med)))
#         vmin_h = med - span
#         vmax_h = med + span

#     # hexbin parameters (tweak gridsize for resolution, mincnt to hide sparse bins)
#     gridsize = 80        # try 60-120 depending on number of points and desired resolution
#     mincnt = 3           # bins with fewer than mincnt points will be blank

#     # do the hexbin: reduce_C_function=np.nanmean computes mean [Mg/Fe] per bin
#     hb = ax.hexbin(
#         x_h, y_h, C=c_h,
#         gridsize=gridsize,
#         reduce_C_function=np.nanmean,
#         mincnt=mincnt,
#         cmap='viridis',
#         vmin=vmin_h, vmax=vmax_h,
#         linewidths=0.2,
#         edgecolors='none'
#     )

#     cbar = fig.colorbar(hb, ax=ax)
#     cbar.set_label(r"Total Star Formation Rate [$M_{\odot}$ / yr]")


#     # draw compactness threshold and labels (same as before)
#     ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#             linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

#     ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
#     ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
#     ax.set_title("Mass–size plane coloured by star formation rate (hexbin)")
#     ax.legend(fontsize=8)
#     ax.grid(True)

#     outpath_hex = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_sfr_hexbin.png")
#     fig.savefig(outpath_hex, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print("Saved sfr-coloured mass-size hexbin:", outpath_hex)

# # ----------------- Hexbin version (specific star formation rate) -----------------

# # Compute sSFR [yr^-1] and take log10
# with np.errstate(divide="ignore", invalid="ignore"):
#     ssfr_plot = np.where(m_in[mask] > 0,
#                           sfr_plot / m_in[mask],
#                           np.nan)
#     log_ssfr_plot = np.where(ssfr_plot > 0,
#                               np.log10(ssfr_plot),
#                               np.nan)

# finite_mask = np.isfinite(log_ssfr_plot)
# x_h = log_m[finite_mask]
# y_h = log_r[finite_mask]
# c_h = log_ssfr_plot[finite_mask]

# if x_h.size == 0:
#     print("No finite sSFR values for hexbin — skipping.")
# else:
#     fig, ax = plt.subplots(figsize=(8,6))

#     # robust vmin/vmax (same logic as before)
#     try:
#         vmin_h = float(np.nanpercentile(c_h, 5))
#         vmax_h = float(np.nanpercentile(c_h, 95))
#     except Exception:
#         vmin_h, vmax_h = float(np.nanmin(c_h)), float(np.nanmax(c_h))
#     if not np.isfinite(vmin_h) or not np.isfinite(vmax_h) or vmin_h == vmax_h:
#         med = float(np.nanmedian(c_h))
#         span = max(0.3, 0.5 * max(1e-6, abs(med)))
#         vmin_h = med - span
#         vmax_h = med + span

#     gridsize = 80
#     mincnt = 3

#     hb = ax.hexbin(
#         x_h, y_h, C=c_h,
#         gridsize=gridsize,
#         reduce_C_function=np.nanmean,
#         mincnt=mincnt,
#         cmap='viridis',
#         vmin=vmin_h, vmax=vmax_h,
#         linewidths=0.2,
#         edgecolors='none'
#     )

#     cbar = fig.colorbar(hb, ax=ax)
#     cbar.set_label(r"$\lg(\mathrm{sSFR}\ /\ \mathrm{yr}^{-1})$")

#     # compactness threshold
#     ax.plot(np.log10(stellar_masses),
#             (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#             linestyle='--', color='black',
#             label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

#     ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
#     ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
#     # ax.set_title("Mass–size plane coloured by specific SFR (hexbin)")
#     ax.legend(fontsize=8)
#     ax.grid(True)

#     outpath_hex = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_ssfr_hexbin.png")
#     fig.savefig(outpath_hex, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print("Saved sSFR-coloured mass-size hexbin:", outpath_hex)

#  # ---------------- LOESS version specific SFR (hole filling, hull_pad_factor) -----------
# # Tunable parameters
# frac_loess = 0.10        # LOESS neighbourhood fraction
# max_eval_pts = 20000     # None to use all points (may be slow)
# degree = 1               # local plane (1) or constant (0)
# nx, ny = 200, 140        # grid resolution
# hull_pad_factor = 1.2    # looser mask
# min_fraction_coverage = 0.25  # fill with nearest if linear grid coverage < 25%

# # # Ensure log_ssfr_plot exists (compute safely if not)
# if 'log_ssfr_plot' not in globals():
#     # expects sfr_plot aligned with m_in[mask]
#     with np.errstate(divide="ignore", invalid="ignore"):
#         ssfr_tmp = np.where(m_in[mask] > 0, sfr_plot / m_in[mask], np.nan)
#         log_ssfr_plot = np.where(ssfr_tmp > 0, np.log10(ssfr_tmp), np.nan)
#     print("Computed log_ssfr_plot from sfr/mass; finite count:", np.count_nonzero(np.isfinite(log_ssfr_plot)))

# # Prepare arrays for LOESS: xvals_all, yvals_all must be full plotting coords
# xvals_all = log_m
# yvals_all = log_r

# # Only keep finite log_ssfr points for smoothing
# finite_mask = np.isfinite(log_ssfr_plot)
# xvals = log_m[finite_mask]
# yvals = log_r[finite_mask]
# zvals = log_ssfr_plot[finite_mask]

# # If no data -> fallback scatter
# if zvals.size == 0:
#     print("No finite log(sSFR) values for LOESS — skipping and falling back to scatter.")
#     fig, ax = plt.subplots(figsize=(8,6))
#     sc = ax.scatter(log_m, log_r, c=log_ssfr_plot, cmap='viridis', s=12, edgecolors='none')
#     cbar = fig.colorbar(sc, ax=ax)
#     cbar.set_label(r"$\lg(\mathrm{sSFR}\ /\ \mathrm{yr}^{-1})$")
#     ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#             linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')
#     ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)"); ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
#     ax.legend(fontsize=8); ax.grid(True)
#     outpath_loess = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_ssfr_loess.png")
#     fig.savefig(outpath_loess, dpi=300, bbox_inches='tight'); plt.close(fig)
#     print("Saved fallback ssfr scatter:", outpath_loess)

# else:
#     # Subsample evaluated LOESS points if requested to limit runtime
#     N = xvals.size
#     if (max_eval_pts is not None) and (N > int(max_eval_pts)):
#         rng = np.random.default_rng(seed=12345)
#         sel_idx = rng.choice(N, size=int(max_eval_pts), replace=False)
#         x_loess = xvals[sel_idx].copy(); y_loess = yvals[sel_idx].copy(); z_loess = zvals[sel_idx].copy()
#     else:
#         x_loess = xvals.copy(); y_loess = yvals.copy(); z_loess = zvals.copy()

#     print(f"LOESS: using {x_loess.size} evaluation points (from {N} finite points).")
#     print("LOESS input z summary (log10 sSFR):",
#           "min=", np.nanmin(z_loess), "med=", np.nanmedian(z_loess), "max=", np.nanmax(z_loess))

#     # Compute LOESS predictions at evaluated points
#     zout_pts, wout = loess_2d(x_loess, y_loess, z_loess,
#                               frac=frac_loess, degree=degree, npoints=None)

#     # Sanity check: zout_pts should be same-kind numbers as z_loess (not weights 0..1)
#     if not (np.isfinite(np.nanmedian(zout_pts)) and np.isfinite(np.nanmedian(z_loess))):
#         raise RuntimeError("LOESS returned non-finite predictions.")
#     med_in = float(np.nanmedian(z_loess))
#     med_out = float(np.nanmedian(zout_pts))
#     # if med_out very different from med_in (e.g. med_out >> med_in + 100 or med_out in 0..1),
#     # try swapping outputs in case function returned (wout,zout) in reverse order.
#     if (abs(med_out - med_in) > 2.0) and (0.0 <= med_out <= 1.0):
#         print("LOESS output median suspicious; attempting to swap zout_pts and wout (fallback).")
#         zout_pts, wout = wout, zout_pts
#         med_out = float(np.nanmedian(zout_pts))
#         print("After swap: LOESS output median:", med_out)

#     print("LOESS output zout_pts summary:", "min=", np.nanmin(zout_pts), "med=", med_out, "max=", np.nanmax(zout_pts))

#     # Build plotting grid
#     xg = np.linspace(np.nanmin(xvals_all), np.nanmax(xvals_all), nx)
#     yg = np.linspace(np.nanmin(yvals_all), np.nanmax(yvals_all), ny)
#     Xg, Yg = np.meshgrid(xg, yg)

#     # Interpolate LOESS estimates from evaluated points onto the grid
#     pts = np.column_stack((x_loess, y_loess))
#     try:
#         Zgrid = griddata(pts, zout_pts, (Xg, Yg), method='linear')
#     except Exception as e:
#         print("griddata(linear) failed; falling back to nearest. Err:", e)
#         Zgrid = griddata(pts, zout_pts, (Xg, Yg), method='nearest')

#     # If linear interpolation produced too many NaNs, fill those NaNs with nearest neighbour values
#     n_finite_linear = int(np.count_nonzero(np.isfinite(Zgrid)))
#     if n_finite_linear < min_fraction_coverage * Zgrid.size:
#         Zgrid_near = griddata(pts, zout_pts, (Xg, Yg), method='nearest')
#         mask_linear_nan = ~np.isfinite(Zgrid)
#         Zgrid[mask_linear_nan] = Zgrid_near[mask_linear_nan]
#         print(f"Linear grid coverage low ({n_finite_linear}/{Zgrid.size}); filled NaNs with nearest neighbour values.")

#     # Build distance mask to blank areas far from any real data (looser)
#     try:
#         tree_all = KDTree(np.column_stack((xvals_all, yvals_all)))
#         grid_pts = np.column_stack((Xg.ravel(), Yg.ravel()))
#         d_grid, _ = tree_all.query(grid_pts, k=1)
#         d_grid = d_grid.reshape(Xg.shape)

#         d_data, _ = tree_all.query(np.column_stack((xvals_all, yvals_all)), k=2)
#         if (d_data.ndim == 2) and (d_data.shape[1] >= 2):
#             typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#         else:
#             typical_spacing = float(np.nanmedian(d_grid))

#         d_thresh = max(typical_spacing * hull_pad_factor, 1e-6)
#         Zgrid_masked = np.array(Zgrid, copy=True)
#         Zgrid_masked[d_grid > d_thresh] = np.nan
#         print(f"LOESS mask applied: typical_spacing={typical_spacing:.4g}, d_thresh={d_thresh:.4g}")
#     except Exception as e:
#         print("KDTree masking failed; using Zgrid without extra mask. Err:", e)
#         Zgrid_masked = Zgrid

#         # right after Zgrid_masked is created
#         n_grid = Zgrid_masked.size
#         n_grid_finite = int(np.count_nonzero(np.isfinite(Zgrid_masked)))
#         pct_grid_filled = 100.0 * n_grid_finite / float(n_grid)
#         print(f"LOESS grid: finite cells {n_grid_finite}/{n_grid} ({pct_grid_filled:.1f}%)")

#         # how many original evaluation points lie outside the mask (i.e. in regions that got blanked)
#         # compute mask for grid points inside threshold and test nearest grid-cell for each eval point
#         grid_pts = np.column_stack((Xg.ravel(), Yg.ravel()))
#         tree_grid = KDTree(grid_pts)
#         d_to_grid, idx_grid = tree_grid.query(np.column_stack((x_loess, y_loess)))
#         inside_flags = np.isfinite(Zgrid_masked.ravel()[idx_grid])
#         print(f"LOESS eval points inside filled region: {inside_flags.sum()}/{len(inside_flags)} ({100*inside_flags.sum()/len(inside_flags):.1f}%)")

#     # Compute vmin/vmax robustly from original zvals (log10 sSFR)
#     try:
#         vmin = float(np.nanpercentile(zvals, 1))
#         vmax = float(np.nanpercentile(zvals, 99))
#         if vmin == vmax:
#             vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))
#     except Exception:
#         vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))

#     # Plot LOESS surface (masked) coloured by log10(sSFR)
#     fig, ax = plt.subplots(figsize=(8,6))
#     cmap_local = plt.get_cmap("viridis")
#     im = ax.pcolormesh(Xg, Yg, Zgrid_masked, shading='auto', cmap=cmap_local, vmin=vmin, vmax=vmax)
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label(r"$\lg(\mathrm{sSFR}\ /\ \mathrm{yr}^{-1})$")

#     # faint markers of the evaluated LOESS points (coverage)
#     ax.scatter(x_loess, y_loess, s=1.5, c='k', alpha=0.06, linewidths=0)

#     # # overlay points that had no ssfr (optional)
#     # if 'SHOW_MISSING' in globals() and SHOW_MISSING:
#     #     missing_idx_global = ~np.isfinite(log_ssfr_plot)
#     #     if np.any(missing_idx_global):
#     #         ax.scatter(log_m[missing_idx_global], log_r[missing_idx_global],
#     #                    color=(0.6,0.6,0.6), alpha=0.6, s=8, label='no ssfr data')

#     # compactness threshold line
#     ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#             linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

#     ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
#     ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
#     ax.legend(fontsize=8); ax.grid(True)

#     outpath_loess = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_ssfr_loess.png")
#     fig.savefig(outpath_loess, dpi=300, bbox_inches='tight'); plt.close(fig)
#     print("Saved ssfr LOESS plot (Fix A):", outpath_loess)
# # ------------------------------------------------------------------------------------

# # --- LOESS-coloured specific SFR (Mg/Fe-consistent) ---

# SHOW_MISSING = False

# ssfr_aligned = log_ssfr_plot.copy()   # already log10(sSFR) aligned to plotting arrays

# # --- impose lower floor instead of masking ---
# SSFR_FLOOR = -12.0

# # replace -inf and other non-finite values with floor
# ssfr_aligned[~np.isfinite(ssfr_aligned)] = SSFR_FLOOR

# # now everything is finite
# have_mask = np.isfinite(ssfr_aligned)
# missing_mask = np.zeros_like(have_mask, dtype=bool)

# n_have = int(have_mask.sum())
# n_missing = 0
# total_plot = int(len(ssfr_aligned))

# print(f"DEBUG sSFR (LOESS block): have={n_have}, missing={n_missing}, total_plot={total_plot}")

# fig, ax = plt.subplots(figsize=(8,6))

# if n_have == 0:
#     if SHOW_MISSING:
#         ax.scatter(log_m, log_r, s=10, alpha=0.7, color="lightgrey", label="no sSFR")
#     else:
#         ax.scatter(log_m, log_r, s=10, alpha=0.7, label="galaxies")
# else:
#     xvals = log_m[have_mask]
#     yvals = log_r[have_mask]
#     zvals = ssfr_aligned[have_mask]

#     pad_x = 0.05 * (np.nanmax(xvals) - np.nanmin(xvals) + 1e-6)
#     pad_y = 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-6)
#     nx, ny = 300, 220
#     xg = np.linspace(np.nanmin(xvals) - pad_x, np.nanmax(xvals) + pad_x, nx)
#     yg = np.linspace(np.nanmin(yvals) - pad_y, np.nanmax(yvals) + pad_y, ny)
#     Xg, Yg = np.meshgrid(xg, yg)
#     pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

#     tree_data = KDTree(np.column_stack((xvals, yvals)))
#     d_grid, _ = tree_data.query(pts_grid, k=1)

#     d_data, _ = tree_data.query(np.column_stack((xvals, yvals)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.nanmedian(d_grid))

#     d_thresh = max(typical_spacing * 1.3, 1e-6)
#     inside_mask = (d_grid <= d_thresh)
#     idx_inside = np.nonzero(inside_mask)[0]

#     if idx_inside.size > 0:
#         xout = pts_grid[idx_inside, 0]
#         yout = pts_grid[idx_inside, 1]

#         frac_loess = 0.01
#         degree = 1

#         Zflat_inside, _ = loess_2d(xvals, yvals, zvals, frac=frac_loess, degree=degree,
#                                    xout=xout, yout=yout)

#         Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
#         Zflat[idx_inside] = Zflat_inside
#         Zgrid = Zflat.reshape((ny, nx))
#         Zmask = np.ma.masked_invalid(Zgrid)

#         try:
#             vmin = float(np.nanpercentile(zvals, 5))
#             vmax = float(np.nanpercentile(zvals, 95))
#         except Exception:
#             vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))
#         if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#             med = float(np.nanmedian(zvals))
#             span = max(0.3, 0.5 * max(1e-6, abs(med)))
#             vmin = med - span
#             vmax = med + span

#         cmap = plt.get_cmap("viridis")
#         im = ax.pcolormesh(Xg, Yg, Zmask, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
#         cbar = fig.colorbar(im, ax=ax)
#         cbar.set_label(r"$\lg(\mathrm{sSFR}\ /\ \mathrm{yr}^{-1})$")

#         ax.scatter(xout, yout, s=1.5, c='k', alpha=0.06, linewidths=0)
#     else:
#         ax.scatter(xvals, yvals, c=zvals, cmap='viridis', s=12, edgecolors='none')

#     if SHOW_MISSING and n_missing > 0:
#         ax.scatter(log_m[missing_mask], log_r[missing_mask], color="lightgrey", s=8, alpha=0.6, label="no sSFR")

# ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#         linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')
# ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
# ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
# ax.legend(fontsize=8)
# ax.grid(True)

# outpath_ssfr = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_ssfr_loess.png")
# fig.savefig(outpath_ssfr, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved sSFR LOESS plot (Mg/Fe-consistent):", outpath_ssfr)

#    # ----------------- Hexbin version (stellar metallicity) -----------------

# finite_mask = np.isfinite(logZstar_rel_in)
# x_h = log_m[finite_mask]
# y_h = log_r[finite_mask]
# c_h = logZstar_rel_in[finite_mask]

# if x_h.size == 0:
#     print("No finite Zstar values for hexbin — skipping.")
# else:
#     fig, ax = plt.subplots(figsize=(8,6))

#     # robust vmin/vmax based on your existing percentile logic
#     try:
#         vmin_h = float(np.nanpercentile(c_h, 5))
#         vmax_h = float(np.nanpercentile(c_h, 95))
#     except Exception:
#         vmin_h, vmax_h = float(np.nanmin(c_h)), float(np.nanmax(c_h))
#     if not np.isfinite(vmin_h) or not np.isfinite(vmax_h) or vmin_h == vmax_h:
#         med = float(np.nanmedian(c_h))
#         span = max(0.2, 0.5 * max(1e-6, abs(med)))
#         vmin_h = med - span
#         vmax_h = med + span

#     # hexbin parameters (tweak gridsize for resolution, mincnt to hide sparse bins)
#     gridsize = 80        # try 60-120 depending on number of points and desired resolution
#     mincnt = 3           # bins with fewer than mincnt points will be blank

#     # do the hexbin: reduce_C_function=np.nanmean computes mean [Mg/Fe] per bin
#     hb = ax.hexbin(
#         x_h, y_h, C=c_h,
#         gridsize=gridsize,
#         reduce_C_function=np.nanmean,
#         mincnt=mincnt,
#         cmap='viridis',
#         vmin=vmin_h, vmax=vmax_h,
#         linewidths=0.2,
#         edgecolors='none'
#     )

#     cbar = fig.colorbar(hb, ax=ax)
#     cbar.set_label(r"$\lg[Z / H]$")

#      # draw compactness threshold and labels (same as before)
#     ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#             linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

#     ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
#     ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
#    # ax.set_title("Mass–size plane coloured by mean stellar age (hexbin)")
#     ax.legend(fontsize=8)
#     ax.grid(True)

#     outpath_hex = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_metallicity_hexbin.png")
#     fig.savefig(outpath_hex, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print("Saved Zstar-coloured mass-size hexbin:", outpath_hex)

#     # --- LOESS-coloured metallicity (Mg/Fe-consistent) ---

# SHOW_MISSING = True

# logZ_aligned = logZstar_rel_in.copy()

# have_mask = np.isfinite(logZ_aligned)
# missing_mask = ~have_mask
# n_have = int(have_mask.sum())
# n_missing = int(missing_mask.sum())
# total_plot = int(len(logZ_aligned))

# print(f"DEBUG Zstar (LOESS block): have={n_have}, missing={n_missing}, total_plot={total_plot}")

# fig, ax = plt.subplots(figsize=(8,6))

# if n_have == 0:
#     if SHOW_MISSING:
#         ax.scatter(log_m, log_r, s=10, alpha=0.7, color="lightgrey", label="no metallicity")
#     else:
#         ax.scatter(log_m, log_r, s=10, alpha=0.7, label="galaxies")
# else:
#     xvals = log_m[have_mask]
#     yvals = log_r[have_mask]
#     zvals = logZ_aligned[have_mask]

#     # grid tightly around the LOESS input points (same as Mg/Fe)
#     pad_x = 0.05 * (np.nanmax(xvals) - np.nanmin(xvals) + 1e-6)
#     pad_y = 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-6)
#     nx, ny = 300, 220
#     xg = np.linspace(np.nanmin(xvals) - pad_x, np.nanmax(xvals) + pad_x, nx)
#     yg = np.linspace(np.nanmin(yvals) - pad_y, np.nanmax(yvals) + pad_y, ny)
#     Xg, Yg = np.meshgrid(xg, yg)
#     pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

#     # KDTree on LOESS input points
#     tree_data = KDTree(np.column_stack((xvals, yvals)))
#     d_grid, _ = tree_data.query(pts_grid, k=1)

#     d_data, _ = tree_data.query(np.column_stack((xvals, yvals)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.nanmedian(d_grid))

#     d_thresh = max(typical_spacing * 1.3, 1e-6)
#     inside_mask = (d_grid <= d_thresh)
#     idx_inside = np.nonzero(inside_mask)[0]

#     if idx_inside.size > 0:
#         xout = pts_grid[idx_inside, 0]
#         yout = pts_grid[idx_inside, 1]

#         frac_loess = 0.01
#         degree = 1

#         Zflat_inside, _ = loess_2d(xvals, yvals, zvals, frac=frac_loess, degree=degree,
#                                    xout=xout, yout=yout)

#         Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
#         Zflat[idx_inside] = Zflat_inside
#         Zgrid = Zflat.reshape((ny, nx))
#         Zmask = np.ma.masked_invalid(Zgrid)

#         try:
#             vmin = float(np.nanpercentile(zvals, 5))
#             vmax = float(np.nanpercentile(zvals, 95))
#         except Exception:
#             vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))
#         if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#             med = float(np.nanmedian(zvals))
#             span = max(0.2, 0.5 * max(1e-6, abs(med)))
#             vmin = med - span
#             vmax = med + span

#         cmap = plt.get_cmap("viridis")
#         im = ax.pcolormesh(Xg, Yg, Zmask, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
#         cbar = fig.colorbar(im, ax=ax)
#         cbar.set_label(r"$\lg[Z_\star / Z_\odot]$")

#         ax.scatter(xout, yout, s=1, c='k', alpha=0.05, linewidths=0)
#     else:
#         ax.scatter(xvals, yvals, c=zvals, cmap='viridis', s=12, edgecolors='none')

#     if SHOW_MISSING and n_missing > 0:
#         ax.scatter(log_m[missing_mask], log_r[missing_mask], color="lightgrey", s=8, alpha=0.6, label="no metallicity")

# ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#         linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')
# ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
# ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
# ax.legend(fontsize=8)
# ax.grid(True)

# outpath_loess = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_metallicity_loess.png")
# fig.savefig(outpath_loess, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved metallicity LOESS plot (Mg/Fe-consistent):", outpath_loess)

#    # ----------------- Hexbin version (mass weighted mean stellar age) -----------------

# finite_mask = np.isfinite(stellar_mw_plot)
# x_h = log_m[finite_mask]
# y_h = log_r[finite_mask]
# c_h = stellar_mw_plot[finite_mask]

# if x_h.size == 0:
#     print("No finite age(mw) values for hexbin — skipping.")
# else:
#     fig, ax = plt.subplots(figsize=(8,6))

#     # robust vmin/vmax based on your existing percentile logic
#     try:
#         vmin_h = float(np.nanpercentile(stellar_lum_plot, 5))
#         vmax_h = float(np.nanpercentile(stellar_lum_plot, 95))
#     except Exception:
#         vmin_h, vmax_h = float(np.nanmin(c_h)), float(np.nanmax(c_h))
#     if not np.isfinite(vmin_h) or not np.isfinite(vmax_h) or vmin_h == vmax_h:
#         med = float(np.nanmedian(c_h))
#         span = max(0.2, 0.5 * max(1e-6, abs(med)))
#         vmin_h = med - span
#         vmax_h = med + span

#     # hexbin parameters (tweak gridsize for resolution, mincnt to hide sparse bins)
#     gridsize = 80        # try 60-120 depending on number of points and desired resolution
#     mincnt = 3           # bins with fewer than mincnt points will be blank

#     # do the hexbin: reduce_C_function=np.nanmean computes mean [Mg/Fe] per bin
#     hb = ax.hexbin(
#         x_h, y_h, C=c_h,
#         gridsize=gridsize,
#         reduce_C_function=np.nanmean,
#         mincnt=mincnt,
#         cmap='viridis',
#         vmin=vmin_h, vmax=vmax_h,
#         linewidths=0.2,
#         edgecolors='none'
#     )

#     cbar = fig.colorbar(hb, ax=ax)
#     cbar.set_label("Mass Weighted Mean Stellar Age [Gyr]")


#     # draw compactness threshold and labels (same as before)
#     ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#             linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

#     ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
#     ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
#    # ax.set_title("Mass–size plane coloured by mean stellar age (hexbin)")
#     ax.legend(fontsize=8)
#     ax.grid(True)

#     outpath_hex = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_age(mw)_hexbin.png")
#     fig.savefig(outpath_hex, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     print("Saved Age(mw)-coloured mass-size hexbin:", outpath_hex)

# # --- LOESS-coloured Mass-weighted mean stellar age (Mg/Fe-consistent) ---

# SHOW_MISSING = True

# mw_aligned = stellar_mw_plot.copy()

# have_mask = np.isfinite(mw_aligned)
# missing_mask = ~have_mask
# n_have = int(have_mask.sum())
# n_missing = int(missing_mask.sum())
# total_plot = int(len(mw_aligned))

# print(f"DEBUG MwAge (LOESS block): have={n_have}, missing={n_missing}, total_plot={total_plot}")

# fig, ax = plt.subplots(figsize=(8,6))

# if n_have == 0:
#     if SHOW_MISSING:
#         ax.scatter(log_m, log_r, s=10, alpha=0.7, color="lightgrey", label="no mw age")
#     else:
#         ax.scatter(log_m, log_r, s=10, alpha=0.7, label="galaxies")
# else:
#     xvals = log_m[have_mask]
#     yvals = log_r[have_mask]
#     zvals = mw_aligned[have_mask]

#     # grid tightly around the LOESS input points (same as Mg/Fe)
#     pad_x = 0.05 * (np.nanmax(xvals) - np.nanmin(xvals) + 1e-6)
#     pad_y = 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-6)
#     nx, ny = 300, 220
#     xg = np.linspace(np.nanmin(xvals) - pad_x, np.nanmax(xvals) + pad_x, nx)
#     yg = np.linspace(np.nanmin(yvals) - pad_y, np.nanmax(yvals) + pad_y, ny)
#     Xg, Yg = np.meshgrid(xg, yg)
#     pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

#     # KDTree on LOESS input points
#     tree_data = KDTree(np.column_stack((xvals, yvals)))
#     d_grid, _ = tree_data.query(pts_grid, k=1)

#     d_data, _ = tree_data.query(np.column_stack((xvals, yvals)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.nanmedian(d_grid))

#     d_thresh = max(typical_spacing * 1.3, 1e-6)
#     inside_mask = (d_grid <= d_thresh)
#     idx_inside = np.nonzero(inside_mask)[0]

#     if idx_inside.size > 0:
#         xout = pts_grid[idx_inside, 0]
#         yout = pts_grid[idx_inside, 1]

#         frac_loess = 0.01
#         degree = 1

#         Zflat_inside, _ = loess_2d(xvals, yvals, zvals, frac=frac_loess, degree=degree,
#                                    xout=xout, yout=yout)

#         Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
#         Zflat[idx_inside] = Zflat_inside
#         Zgrid = Zflat.reshape((ny, nx))
#         Zmask = np.ma.masked_invalid(Zgrid)

#         try:
#             vmin = float(np.nanpercentile(stellar_lum_plot, 5))
#             vmax = float(np.nanpercentile(stellar_lum_plot, 95))
#         except Exception:
#             vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))
#         if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
#             med = float(np.nanmedian(zvals))
#             span = max(0.2, 0.5 * max(1e-6, abs(med)))
#             vmin = med - span
#             vmax = med + span

#         cmap = plt.get_cmap("viridis")
#         im = ax.pcolormesh(Xg, Yg, Zmask, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
#         cbar = fig.colorbar(im, ax=ax)
#         cbar.set_label("Mass Weighted Mean Stellar Age [Gyr]")

#         ax.scatter(xout, yout, s=1, c='k', alpha=0.05, linewidths=0)
#     else:
#         ax.scatter(xvals, yvals, c=zvals, cmap='viridis', s=12, edgecolors='none')

#     if SHOW_MISSING and n_missing > 0:
#         ax.scatter(log_m[missing_mask], log_r[missing_mask], color="lightgrey", s=8, alpha=0.6, label="no lum age")

# ax.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref),
#         linestyle='--', color='black', label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')
# ax.set_xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
# ax.set_ylabel(r"lg(Half Mass Radius / kpc)")
# ax.legend(fontsize=8)
# ax.grid(True)

# outpath_loess = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_age(mw)_loess.png")
# fig.savefig(outpath_loess, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved mw age LOESS plot (Mg/Fe-consistent):", outpath_loess)

# --- LOESS-coloured velocity dispersion (Mg/Fe-consistent) ---

SHOW_MISSING = True

sigma_aligned = sigma_vals.copy()   # align to plotting sample
log_sigma_aligned = log_sigma_vals.copy()

have_mask = np.isfinite(log_sigma_aligned)

missing_mask = ~have_mask

n_have = int(have_mask.sum())

n_missing = int(missing_mask.sum())

total_plot = int(len(log_sigma_aligned))

print(f"DEBUG sigma (LOESS block): have={n_have}, missing={n_missing}, total_plot={total_plot}")

fig, ax = plt.subplots(figsize=(8,6))

if n_have == 0:

    if SHOW_MISSING:

        ax.scatter(log_m, log_r, s=10, alpha=0.7,

                   color="lightgrey", label="no sigma")

    else:

        ax.scatter(log_m, log_r, s=10, alpha=0.7, label="galaxies")

else:

    xvals = log_m[have_mask]

    yvals = log_r[have_mask]

    zvals = log_sigma_aligned[have_mask]

    # --- SAME GRID LOGIC ---

    pad_x = 0.05 * (np.nanmax(xvals) - np.nanmin(xvals) + 1e-6)

    pad_y = 0.05 * (np.nanmax(yvals) - np.nanmin(yvals) + 1e-6)

    nx, ny = 300, 220

    xg = np.linspace(np.nanmin(xvals) - pad_x,

                     np.nanmax(xvals) + pad_x, nx)

    yg = np.linspace(np.nanmin(yvals) - pad_y,

                     np.nanmax(yvals) + pad_y, ny)

    Xg, Yg = np.meshgrid(xg, yg)

    pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

    # --- KDTree masking (IDENTICAL) ---

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

        frac_loess = 0.01

        degree = 1

        Zflat_inside, _ = loess_2d(

            xvals, yvals, zvals,

            frac=frac_loess,

            degree=degree,

            xout=xout,

            yout=yout

        )

        Zflat = np.full(pts_grid.shape[0], np.nan)

        Zflat[idx_inside] = Zflat_inside

        Zgrid = Zflat.reshape((ny, nx))

        Zmask = np.ma.masked_invalid(Zgrid)

        # --- SAME COLOR SCALING ---

        try:

            vmin = float(np.nanpercentile(zvals, 5))

            vmax = float(np.nanpercentile(zvals, 95))

        except Exception:

            vmin, vmax = float(np.nanmin(zvals)), float(np.nanmax(zvals))

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:

            med = float(np.nanmedian(zvals))

            span = max(10.0, 0.5 * max(1e-6, abs(med)))

            vmin = med - span

            vmax = med + span

        cmap = plt.get_cmap("viridis")

        im = ax.pcolormesh(

            Xg, Yg, Zmask,

            shading='auto',

            cmap=cmap,

            vmin=vmin,

            vmax=vmax

        )

        cbar = fig.colorbar(im, ax=ax)

        cbar.set_label(r'$\lg(\sigma / \mathrm{km}\ \mathrm{s}^{-1})$')

        ax.scatter(xout, yout, s=1,

                   c='k', alpha=0.05, linewidths=0)

    else:

        ax.scatter(xvals, yvals,

                   c=zvals, cmap='viridis',

                   s=12, edgecolors='none')

    if SHOW_MISSING and n_missing > 0:

        ax.scatter(log_m[missing_mask],

                   log_r[missing_mask],

                   color="lightgrey", s=8, alpha=0.6,

                   label="no sigma")

# --- SAME DECORATION ---

ax.plot(np.log10(stellar_masses),

        (2/3)*(np.log10(stellar_masses) - logsigma_ref),

        linestyle='--', color='black',

        label=fr'Compactness threshold ($\lg\Sigma_{{1.5}} = {logsigma_ref}$)')

ax.set_xlabel(r"$\lg(M_\star / M_{\odot})$")
ax.set_ylabel(r"$\lg(R_{1/2, \star} / \mathrm{kpc})$")

ax.legend(fontsize=8)

ax.grid(True)

outpath_sigma = os.path.join(

    outdir,

    f"mass_size_z{ztarget:.1f}_sigma_loess.png"

)

fig.savefig(outpath_sigma, dpi=300, bbox_inches='tight')

plt.close(fig)

print("Saved sigma LOESS plot:", outpath_sigma)