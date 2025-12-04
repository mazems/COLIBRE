import math
import matplotlib.pyplot as plt
import numpy as np
import os
import common
import pandas as pd
import swiftsimio as sw
from swiftsimio import SWIFTDataset
from swiftgalaxy import SWIFTGalaxy, SOAP
import h5py 
import gc     # for explicit garbage collection to reduce memory footprint
from astropy.cosmology import Planck15
import astropy.units as au
import unyt as u
from swiftsimio.objects import cosmo_array, cosmo_factor, cosmo_quantity
from scipy.stats import binned_statistic_2d
import matplotlib as mpl
import utilities_statistics as us

# ---------- user configuration (keep as you had) ----------
family_method = 'radial_profiles'
method = 'circular_apertures_face_on_map'

model_name = 'L0200N3008/THERMAL_AGN/'
# CHANGED: use consistent model_dir base used earlier in your workflow
model_dir = '/mnt/su3-pro/colibre/' + model_name

# snapshot configuration (keep your existing setting)
snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0026', '0018']
zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# Select only the z = 0 snapshot (as before)
snap_file = snap_files[0]   # '0127' corresponds to z=0.0
ztarget = zstarget[0]
comov_to_physical_length = 1.0 / (1.0 + ztarget)

# ---------- file names used by common/SOAP ----------
soap_catalogue_file = os.path.join(model_dir, "SOAP-HBT", f"halo_properties_{snap_file}.hdf5")
virtual_snapshot_file = os.path.join(model_dir, "SOAP-HBT", f"colibre_with_SOAP_membership_{snap_file}.hdf5")

print("Using SOAP catalogue:", soap_catalogue_file)
print("Using virtual snapshot:", virtual_snapshot_file)
if not os.path.exists(soap_catalogue_file):
    raise SystemExit("SOAP catalogue not found at: " + soap_catalogue_file)
if not os.path.exists(virtual_snapshot_file):
    raise SystemExit("Virtual snapshot not found at: " + virtual_snapshot_file)

sd = SWIFTDataset(soap_catalogue_file)
idgal = sd.input_halos.halo_catalogue_index
ms50 = sd.exclusive_sphere_50kpc.stellar_mass
candidates = np.argwhere(
    np.logical_and(
        ms50
        > cosmo_quantity(
            1e9, u.Msun, comoving=True, scale_factor=sd.metadata.a, scale_exponent=0
        ),
        idgal == 3195990,
    )
).squeeze()
 
candidates = np.array([candidates])
 
print(candidates)
 
sg = SWIFTGalaxy(
    virtual_snapshot_file,
    SOAP(
        soap_catalogue_file,
        soap_index=candidates[0],
    ),
)
 
coord_in_p4 = sg.stars.coordinates
vT4_in = sg.stars.velocities
m_part4 = sg.stars.masses
birth_scales_T4 = sg.stars.birth_scale_factors
redshifts_T4 = 1/birth_scales_T4 - 1
lbt_T4 = us.look_back_time(redshifts_T4, h=0.6751, omegam=0.3121, omegal=0.6879)
t_start=float(np.min(lbt_T4))
print("t_start=", t_start)
print("N_stars:", m_part4.size)
print("mass sample:", m_part4)
print("Total stellar mass:", np.sum(m_part4))
