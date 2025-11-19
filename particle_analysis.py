import math
import matplotlib.pyplot as plt
import numpy as np
import os
import common
import swiftsimio as sw
from swiftgalaxy import SWIFTGalaxy, SOAP
# import cmasher as cmr
#import utilities_statistics as us


family_method = 'radial_profiles'
method = 'circular_apertures_face_on_map'

model_name = 'L0200N3008/THERMAL_AGN/'
model_dir = '/mnt/su3-pro/colibre/' + model_name

#definitions below correspond to z=0
snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0026', '0018']
zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

###################### simulation units #########################################
Lu = 3.086e+24/(3.086e+24) #cMpc
Mu = 1.988e+43/(1.989e+33) #Msun
tu = 3.086e+19/(3.154e+7) #yr
Tempu = 1 #K
density_cgs_conv = 6.767905773162602e-31 #conversion from simulation units to CGS for density
mH = 1.6735575e-24 #in gr

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

# Select only the z = 0 snapshot
snap_file = snap_files[0]   # '0127' corresponds to z=0.0
ztarget = zstarget[0]       # ensure ztarget is a single value
comov_to_physical_length = 1.0 / (1.0 + ztarget)





soap_catalogue_file = os.path.join(
    model_dir,
    "SOAP-HBT/halo_properties_" + snap_files[z] + ".hdf5",
)

virtual_snapshot_file = os.path.join(
    model_dir, "SOAP-HBT/colibre_with_SOAP_membership_" + snap_files[z] + ".hdf5"
)
            


for g in range(nstart,nend):
    print("processing galaxy:", g, "out of", ngals)
    sg = SWIFTGalaxy(
        virtual_snapshot_file,
        SOAP(
            soap_catalogue_file,
            soap_index=candidates[g],
        ),
    )

    coord_in_p4 = sg.stars.coordinates
    vT4_in = sg.stars.velocities
    m_part4 = sg.stars.masses
