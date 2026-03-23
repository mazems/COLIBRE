import numpy as np
from scipy.spatial import cKDTree
from __future__ import annotations
import os
import sys
import csv
import gc
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# ---- user/project imports ----
# make sure your PYTHONPATH includes the folder where `common` is located
import common

# ---------------- CONFIG / FLAGS ----------------
RUN_PLOT = True    # set True to create the z=0 mass-size plot
RUN_TRACE = False    # set True to run the trace of zero-BH extreme relics across snapshots

CORRECTED_DOR_CSV = "sfh_times_all_with_DoR_variants_corrected.csv.gz"
OUTDIR = "plots"
OUTNAME = "mass_size_extremes_compactness9p75.png"
OUT_CSV_TRACE = os.path.join(OUTDIR, "extreme_relics_zeroBH_central_status_by_snap.csv")

COMPACTNESS_CUT = 9.75
EXTREME_DOR = 0.6

def compute_density_contrast(soap_hc_p, soap_hc_pop_p, mtot, boxsize_p, izarray):
    """
    Compute subhalo density contrast for given galaxy positions.
    soap_hc_pop_: array of shape (N, 3) with physical positions of all galaxies in kpc (from soap_catalogue.input_halos.halo_centre, converted to physical kpc)
    soap_hc_pop_p: array of shape (M, 3) with physical positions of particular population of galaxies in kpc (from soap_catalogue.input_halos.halo_centre, converted to physical kpc)
    mtot: array of shape (N,) with total mass of all galaxies in Msun (from soap_catalogue.bound_subhalo.total_mass, converted to Msun)
    boxsize_p: size of the box in physical kpc at each redshift (from soap_meta.boxsize, converted to physical kpc)
    Returns: dcont_pop array of shape (len(soap_hc_pop_p), len(apertures))
    """
    
    # convert everything to comoving unitst
    boxsize_c = boxsize_p * (1 + izarray)
    soap_hc_c = soap_hc_p * (1 + izarray)
    soap_hc_pop_c = soap_hc_pop_p * (1 + izarray)
    
    # check that all positions are within the box and apply periodic boundary conditions if not
    xcentre = soap_hc_c[:,0]
    xcentre[xcentre>boxsize_c] = xcentre[xcentre>boxsize_c]-boxsize_c
    ycentre = soap_hc_c[:,1]
    ycentre[ycentre>boxsize_c] = ycentre[ycentre>boxsize_c]-boxsize_c
    zcentre = soap_hc_c[:,2]
    zcentre[zcentre>boxsize_c] = zcentre[zcentre>boxsize_c]-boxsize_c
    positions_c = np.vstack([xcentre, ycentre, zcentre]).T
    xcentre = soap_hc_pop_c[:,0]
    xcentre[xcentre>boxsize_c] = xcentre[xcentre>boxsize_c]-boxsize_c
    ycentre = soap_hc_pop_c[:,1]
    ycentre[ycentre>boxsize_c] = ycentre[ycentre>boxsize_c]-boxsize_c
    zcentre = soap_hc_pop_c[:,2]
    zcentre[zcentre>boxsize_c] = zcentre[zcentre>boxsize_c]-boxsize_c
    positions_pop_c = np.vstack([xcentre, ycentre, zcentre]).T
    
    # build a KDTree for the positions of all galaxies 
    tree = cKDTree(positions_c, boxsize=boxsize_c)
    # in this case we consider 3 apertures: 0.3, 1 and 3 Mpc (comoving), which are typical for the size of the CGM and the halo, but we can consider other apertures if needed
    apertures = 1e3 * np.array([0.3, 1, 3]) # in kpc (comoving)
    dcont_pop = np.full((len(soap_hc_pop_c), len(apertures)), np.nan)
    # mean mass density
    volume_box = boxsize_c**3
    mdcont = np.sum(mtot) / volume_box
    for k, aperture in enumerate(apertures):
        volume_neigh = 4/3 * np.pi * (aperture**3)
        # get neighbors within the aperture for each galaxy in the population using the KDTree
        neighbors_pop_c = tree.query_ball_point(positions_pop_c, r=aperture)
        # inds is a list of arrays, where each array contains the indices of the neighbors of each galaxy in the population within the aperture
        for l, inds in enumerate(neighbors_pop_c):
            mtot_neigh = mtot[inds]
            # subhalo density contrast for the galaxy l in the population and aperture k (equation 12 in paper)
            dcont_pop[l, k] = ((np.sum(mtot_neigh.value) / volume_neigh) - mdcont.value) / mdcont.value
    
    return dcont_pop

# measure environment through the density contrast
# loop over the particular population and compute the density contrast for each galaxy in the population using the positions of all galaxies in the catalogue
if np.shape(ind_p)[0]>0:
     dcont_p = compute_density_contrast(soap_hc, soap_hc_p, mtot, boxsize_p, izarray)
     # combine soap_hc_p (massive quenched) and soap_hc_sf (massive star-forming) to compute 
     # the mean and std of the density contrast for the whole population of massive galaxies (quenched and star-forming)
     soap_hc_all = np.vstack((soap_hc_p, soap_hc_sf))
     # compute density contrast for the whole population of massive galaxies (quenched and star-forming)
     dcont_all = compute_density_contrast(soap_hc, soap_hc_all, mtot, boxsize_p, izarray)
     # compute the deviation of the density contrast of the quenched population (equation 13 in the paper)
     # from the mean of the whole population in units of the std of the whole population
     devdcont_p = (dcont_p - np.mean(dcont_all,axis=0)) / np.std(dcont_all,axis=0)