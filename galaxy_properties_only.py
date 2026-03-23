
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import common
#import utilities_statistics as us


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
model_name = 'L0200N1504/THERMAL_AGN/'
#model_name = 'L0200N3008/THERMAL_AGN/'
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
fields ={'ExclusiveSphere/50kpc': ('StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'CentreOfMass', 'AtomicHydrogenMass', 'MolecularHydrogenMass', 'KappaCorotStars', 'KappaCorotGas', 'DiscToTotalStellarMassFraction', 'MassWeightedMeanStellarAge', 'LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasLowLimit' ,'LogarithmicMassWeightedDiffuseOxygenOverHydrogenOfGasHighLimit', 'AngularMomentumStars', 'DustLargeGrainMass', 'DustSmallGrainMass')}
h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
(m30, sfr30, r50, cp, mHI, mH2, kappacostar, kappacogas, disctotot, stellarage, ZgasLow, ZgasHigh, Jstars, mdustl, mdusts) = h5data_groups

#unit conversion
mdust = (mdustl + mdusts) * Mu
m30 = m30 * Mu
mHI = mHI * Mu
mH2 = mH2 * Mu
sfr30 = sfr30 * Mu / tu 
r50 = r50 * Lu * comov_to_physical_length * 1e3 #convert in kpc
stellarage = stellarage * tu / 1e9 #in Gyr
cp = cp * Lu * comov_to_physical_length
Jstars = Jstars * Mu / (Lu * comov_to_physical_length)**2 / tu
    
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
    sfr_in = sfr30[select]
    sgn_in = sgn[select]
    is_central_in = is_central[select]
    r50_in = r50[select]
    mHI_in = mHI[select]
    mH2_in = mH2[select]
    kappacostar_in = kappacostar[select]
    kappacogas_in = kappacogas[select]
    disctotot_in = disctotot[select]
    stellarage_in = stellarage[select]
    ZgasLow_in = ZgasLow[select]
    ZgasHigh_in = ZgasHigh[select]
    x_in = xg[select]
    y_in = yg[select]
    z_in = zg[select]
    Jstars_in = Jstars[select, :]
    Jstars_in = Jstars_in[0]
    Jstars_in_norm = np.sqrt(Jstars_in[:,0]**2 + Jstars_in[:,1]**2 + Jstars_in[:,2]**2)
    mdust_in = mdust[select]
   
    #save galaxy properties of interest
    gal_props = np.zeros(shape = (ngals, 20))
    gal_props[:,0] = sgn_in
    gal_props[:,1] = is_central_in
    gal_props[:,2] = x_in
    gal_props[:,3] = y_in
    gal_props[:,4] = z_in
    gal_props[:,5] = m_in
    gal_props[:,6] = sfr_in
    gal_props[:,7] = r50_in
    gal_props[:,8] = mHI_in
    gal_props[:,9] = mH2_in
    gal_props[:,10] = kappacostar_in
    gal_props[:,11] = kappacogas_in
    gal_props[:,12] = disctotot_in
    gal_props[:,13] = Jstars_in_norm
    gal_props[:,14] = stellarage_in
    gal_props[:,15] = ZgasLow_in
    gal_props[:,16] = ZgasHigh_in
    gal_props[:,17] = mdust_in
    gal_props[:,18] = desc_id[select]
    gal_props[:,19] = track_id[select]
    np.savetxt('/cosma8/data/do019/dc-zems1/COLIBRE/Runs/' + model_name + '/ProcessedData/GalaxyProperties_z' + str(ztarget) + '.txt', gal_props)

#Compact galaxy relation
stellar_masses = np.logspace(9, 12, 100)
a = 2/3
logsigma = 8.0 #10.3 in Barro 2013
logsigma_ref = 10.3
effective_radii = (stellar_masses/(10**(logsigma)))**a

# ---- Use selected arrays produced above ----
# m_in and r50_in exist only if ngals > 0; otherwise handle gracefully
if ngals > 0:
    mt_plot = m_in
    r50_plot = r50_in

    # filter non-positive values
    mask = (mt_plot > 0) & (r50_plot > 0)
    if not np.any(mask):
        raise RuntimeError("No positive mtot/r50 values to plot after filtering selection.")

    log_m = np.log10(mt_plot[mask])
    log_r = np.log10(r50_plot[mask])

    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": 14
    })
    plt.figure(figsize=(8,6))

    plt.scatter(log_m, log_r, alpha=0.7, s=10, label=f"Simulated galaxies at z={ztarget}")
    # Threshold line Barro
    plt.plot(np.log10(stellar_masses), (2/3)*(np.log10(stellar_masses) - logsigma_ref), 
            linestyle='--', color='black', label=fr'Barro et al. (2013) ($\lg{{\Sigma_{{\mathrm{{1.5}}}}}} = {logsigma_ref}$)')
    plt.xlabel(r"lg(Stellar Mass / $M_{\odot}$)")
    plt.ylabel(r"lg(Half Mass Radius / kpc)")
    plt.title("Mass-size relation (COLIBRE 200m6)")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=12, direction='in', length=6, width=1)
    plt.show()


    # save plot
    outdir = os.path.join(os.getcwd(), "plots")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"mass_size_z{ztarget:.1f}_proj.png")
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    print("Saved plot to:", outpath)
    plt.close()
else:
    print("No galaxies selected; skipping plot.")


