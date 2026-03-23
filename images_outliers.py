import matplotlib.pyplot as plt
import os
from astropy.io import fits
import argparse
from swiftsimio import SWIFTDataset
from swiftgalaxy import SWIFTGalaxy, SOAP
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PARTRIDGE"))

import partridge

#define outdir first
original_cwd = os.getcwd()
# dann partridge import + os.chdir(partridge_dir)
outdir = os.path.join(original_cwd, "images")
os.makedirs(outdir, exist_ok=True)

# ensure partridge's data files are found by making its folder the working dir
partridge_dir = os.path.dirname(partridge.__file__)
print("Using PARTRIDGE at:", partridge_dir)   # optional check
os.chdir(partridge_dir)

from partridge import make_galaxy_image

parser = argparse.ArgumentParser(description="Takes two arguments.")

parser.add_argument('id', type=int)
parser.add_argument('snap', type=int)
# Parse the arguments
args = parser.parse_args()
# model_name = 'L200_m6/Thermal/'
# out_dir = '/cosma8/data/dp004/ngdg66/Runs/' + model_name + '/'
# SIMULATION_PATH = '/cosma8/data/dp004/colibre/Runs/' + model_name
model_name = 'L0200N3008/THERMAL_AGN/'
model_dir = '/mnt/su3-pro/colibre/' + model_name
# model_dir = '/mnt/su3-pro/clagos/COLIBRE/Runs/' + model_name
# outdir = os.path.join(os.getcwd(), "images")
orientation = "edge_on"
if (orientation == "edge_on"):
    n_neigh = 64
else:
    n_neigh = 512
 
id_target = args.id
snap_num = args.snap
image_RGB = make_galaxy_image(
             simulation_path = model_dir, # this should be the directory path of the simulation
             snapshot_number = snap_num,
             halo_track_id = id_target, # the HaloCatalogueIndex ID (same as trackID) within SOAP of the object 
             image_size = 100, # full physical size (in kpc) of the image
             pixel_size = 500, # integer side-length of the image, i.e. the image will be pixel_size x pixel_size large. The calculations get slow and memory-intensive if of order 1000 pixels are used
             bound_only = False, # whether to include only particles bound to the object
             rotation_modes = [orientation], # face_on, edge_on, random_x, random_y, random_z. Note: 'random_xyz' means simply a projection along one of the box axes, which gives a random galaxy orientati    on
             image_modes = ["SDSS_original"], # which image mode to use, i.e. which bands and how to combine them, options: HST_gri, SDSS_original, Euclid
             redshift = 0.0, # shift the emission spectrum to be at a given redshift, i.e. create images as would be observed at z=0
             parallelize = True, # whether to do volume renders in parallel (fast but memory-heavy)
             N_ngb_target = n_neigh, # 512 is optimal in most cases, but for edge-on images, a much smaller neighbour number is better (e.g. try 512/8 = 64)
             min_brightness = 1, # minimum brightness of image in units of Lsun/pc^2, fainter pixels will be black
             dust_mode = 'full_scattering',  # options: None, absorption_only, isotropic_scattering, anisotropic_scattering, full_scattering
             h_min = 0.2, # the minimum smoothing length in physical kpc. Use 0.1/0.2/0.4 at m5/m6/m7

             output_no_dust = True,
             output_raw_brightness = False,
         )
 
# ------------------ Safe reading & saving of partridge output ------------------
from PIL import Image

result = image_RGB 

print("make_galaxy_image returned top-level keys:", list(result.keys()))
# select first image_mode
image_mode = list(result.keys())[0]
print("Using image_mode:", image_mode)

halo_keys = list(result[image_mode].keys())
if len(halo_keys) == 0:
    raise RuntimeError("make_galaxy_image returned no halo entries for this image mode.")
halo_id = halo_keys[0]
print("Found halo id:", halo_id)

# Extract RGB entry (either dict {'dusty', 'dust_free'} or Array)
rgb_entry = result[image_mode][halo_id].get("RGB", None)
if rgb_entry is None:
    raise RuntimeError("No 'RGB' entry found in returned data for this halo.")

# save either both variants ('dusty'/'dust_free') or the single Array
def _normalize_and_save(arr, fname):
    arr = np.asarray(arr)
    # formats: (H,W,3) or (3,H,W)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        # (3,H,W) -> (H,W,3)
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"Unexpected RGB array shape: {arr.shape}")
    
    if arr.dtype.kind == "f":
        cols = np.clip(arr, 0.0, 1.0)
    else:
        # integer types -> assume already 0..255
        arr = arr.astype(np.float32)
        cols = np.clip(arr / 255.0, 0.0, 1.0)
    img = Image.fromarray((cols * 255).astype(np.uint8))
    img.save(fname)
    print("Saved", fname)

# make sure outdir exists
os.makedirs(outdir, exist_ok=True)

if isinstance(rgb_entry, dict):
    # save every tag (e.g. 'dusty' and 'dust_free')
    for tag, arr in rgb_entry.items():
        if arr is None:
            print("No image for tag:", tag)
            continue
        fname = os.path.join(outdir, f"SDSS_id{int(id_target)}_snap{int(snap_num)}_{image_mode}_{tag}_{orientation}.png")
        _normalize_and_save(arr, fname)
else:
    # single Array
    fname = os.path.join(outdir, f"SDSS_id{int(id_target)}_snap{int(snap_num)}_{image_mode}_{orientation}.png")
    _normalize_and_save(rgb_entry, fname)
# -------------------------------------------------------------------------------------------------