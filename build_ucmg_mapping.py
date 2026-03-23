#!/usr/bin/env python3
import os, sys
import numpy as np
import pandas as pd
import h5py
from collections import defaultdict

MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN'
SNAP_FILE = 'colibre_with_SOAP_membership_0127.hdf5'
VIRTUAL_SNAPSHOT_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', SNAP_FILE)
UCMG_CSV = 'ucmg_ids.csv'
OUTNPZ = 'ucmg_particle_index_mapping.npz'
CHUNK = 2_000_000   # lower if memory spikes

if not os.path.exists(UCMG_CSV):
    raise SystemExit("ucmg_ids.csv missing")

ucmg = pd.read_csv(UCMG_CSV)
if 'subhalo_id' in ucmg.columns:
    ids = np.array(ucmg['subhalo_id'], dtype=np.int64)
else:
    ids = np.array(ucmg.iloc[:,0], dtype=np.int64)
req_set = set(int(x) for x in np.unique(ids))
print("Requested unique subhalo IDs:", len(req_set))

# build dict of lists
mapping = {sid: [] for sid in req_set}

print("Opening snapshot:", VIRTUAL_SNAPSHOT_FILE)
f = h5py.File(VIRTUAL_SNAPSHOT_FILE, 'r')
p4 = f['PartType4']
halo_ds = p4['HaloCatalogueIndex']
npart = int(halo_ds.shape[0])
print("HaloCatalogueIndex length:", npart)

for start in range(0, npart, CHUNK):
    stop = min(npart, start+CHUNK)
    chunk = np.array(halo_ds[start:stop], dtype=np.int64)
    mask = np.isin(chunk, list(req_set))
    if not mask.any():
        continue
    rel = np.nonzero(mask)[0]
    vals = chunk[rel]
    unique_vals, inv = np.unique(vals, return_inverse=True)
    for j,val in enumerate(unique_vals):
        pos = rel[inv==j]
        mapping[int(val)].append(pos + start)
    # allow GC
    del chunk

# Convert lists -> arrays and save
npzdict = {}
for sid, lst in mapping.items():
    if len(lst)==0:
        npzdict[str(sid)] = np.array([], dtype=np.int64)
    else:
        npzdict[str(sid)] = np.concatenate(lst).astype(np.int64)
print("Saving mapping to", OUTNPZ)
np.savez_compressed(OUTNPZ, **npzdict)
f.close()
print("Done.")
