#!/usr/bin/env python3
"""
Quick inspector: check whether given subhalo IDs have star particles in the virtual snapshot,
and print simple diagnostics (counts, sum of masses, sample indices + one particle's attributes).
"""

import os, sys
import numpy as np
import h5py
import pandas as pd

# ====== CONFIG - edit paths if needed ======
MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN'
SNAP_FILE = 'colibre_with_SOAP_membership_0127.hdf5'
VIRTUAL_SNAPSHOT_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', SNAP_FILE)
SOAP_CATALOGUE_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', 'halo_properties_0127.hdf5')
UCMG_CSV = 'ucmg_ids.csv'
# ===========================================

# Option A: read first N ids from ucmg csv
N_CHECK = 20

# Option B: or set explicit ids here:
# check_ids = [158660, 12345]
check_ids = None

# ---------- load ids ----------
if check_ids is None:
    if not os.path.exists(UCMG_CSV):
        raise SystemExit("ucmg csv not found: " + UCMG_CSV)
    df = pd.read_csv(UCMG_CSV)
    if 'subhalo_id' in df.columns:
        ids = df['subhalo_id'].astype(int).to_numpy()
    else:
        ids = df.iloc[:,0].astype(int).to_numpy()
    check_ids = list(ids[:N_CHECK])

print("Will inspect subhalo ids:", check_ids)

# ---------- open snapshot (lazy) ----------
if not os.path.exists(VIRTUAL_SNAPSHOT_FILE):
    raise SystemExit("Virtual snapshot not found: " + VIRTUAL_SNAPSHOT_FILE)
f = h5py.File(VIRTUAL_SNAPSHOT_FILE, 'r')
if 'PartType4' not in f:
    raise SystemExit("PartType4 not in snapshot")
p4 = f['PartType4']

# dataset handles
halo_ds = p4['HaloCatalogueIndex']    # particle -> subhalo id mapping (h5py dataset)
masses_ds = None
for n in ('InitialMasses','Masses','masses'):
    if n in p4:
        masses_ds = p4[n]
        break
birth_ds = p4['BirthScaleFactors'] if 'BirthScaleFactors' in p4 else None
ages_ds = p4['Ages'] if 'Ages' in p4 else None
elem_ds = p4['ElementMassFractions'] if 'ElementMassFractions' in p4 else None

print("Snapshot datasets found: HaloCatalogueIndex:", halo_ds.shape, "Masses:", getattr(masses_ds,'shape',None),
      "BirthScaleFactors:", getattr(birth_ds,'shape',None), "Ages:", getattr(ages_ds,'shape',None),
      "ElementMassFractions:", getattr(elem_ds,'shape',None))

# We'll scan HaloCatalogueIndex in chunks and collect indices for just the requested ids
req = set(int(x) for x in check_ids)
mapping_lists = {sid: [] for sid in req}
npart = int(halo_ds.shape[0])
chunk = 500000   # tune down if needed

import numpy as np
for start in range(0, npart, chunk):
    stop = min(npart, start+chunk)
    arr = np.array(halo_ds[start:stop], dtype=np.int64)
    # find only particles whose value is in req:
    mask = np.isin(arr, list(req))
    if not np.any(mask):
        continue
    rel_pos = np.nonzero(mask)[0]
    vals = arr[rel_pos]
    # group by value
    unique_vals, inv = np.unique(vals, return_inverse=True)
    for j,val in enumerate(unique_vals):
        pos = rel_pos[inv==j] + start
        mapping_lists[int(val)].append(pos)
    del arr

# Convert and inspect
for sid in check_ids:
    idxs_list = mapping_lists.get(int(sid), [])
    if len(idxs_list)==0:
        indices = np.array([], dtype=np.int64)
    else:
        indices = np.concatenate(idxs_list).astype(np.int64)
    print("\n=== subhalo", sid, "-> particle count:", indices.size, " ===")
    if indices.size == 0:
        print(" No star particles found for this subhalo in HaloCatalogueIndex.")
    else:
        # show first few indices
        print(" First indices (up to 10):", indices[:10])
        # sum masses (lazy read)
        try:
            masses_sel = np.array(masses_ds[indices], dtype=float)
            print("  sum(masses) = {:.6e} (units as in file)  n_particles = {}".format(np.sum(masses_sel), masses_sel.size))
        except Exception as e:
            print("  Could not read masses:", e)
        # if birth or ages present, show min/max birth times presence
        try:
            if birth_ds is not None:
                a_sel = np.array(birth_ds[indices], dtype=float)
                print("  BirthScaleFactors sample (first 10):", a_sel[:10])
            elif ages_ds is not None:
                ages_sel = np.array(ages_ds[indices], dtype=float)
                print("  Ages sample (first 10):", ages_sel[:10])
        except Exception as e:
            print("  Could not read birth/ages:", e)
        # elements info
        if elem_ds is not None:
            try:
                em = np.array(elem_ds[indices[:min(100, indices.size)]], dtype=float)
                print("  ElementMassFractions sample shape:", em.shape, " (showing first row):")
                print("   ", em[0] if em.ndim==2 and em.shape[0]>0 else em[:10])
            except Exception as e:
                print("  Could not read ElementMassFractions:", e)

# optional: check SOAP catalogue for stellar mass metadata for these halos
if os.path.exists(SOAP_CATALOGUE_FILE):
    try:
        sf = h5py.File(SOAP_CATALOGUE_FILE,'r')
        # try common dataset names, print a few fields for rows that match
        # find halo rows that match these subhalo ids
        # This depends strongly on SOAP file layout; we'll attempt a couple common keys:
        keys_to_try = ['InputHalos/HaloCatalogueIndex','HaloCatalogueIndex','HaloIndex','HaloID']
        dataset = None
        for k in keys_to_try:
            if k in sf:
                dataset = sf[k][()]
                print("Found SOAP dataset:", k, "shape:", dataset.shape)
                break
        if dataset is not None:
            # build reverse mapping value->row index (for quick lookup)
            val_to_row = {int(v):int(i) for i,v in enumerate(dataset)}
            for sid in check_ids:
                row = val_to_row.get(int(sid), None)
                print("SOAP mapping for", sid, "-> row", row)
                if row is not None:
                    # look for common stellar mass fields in SOAP file at this row
                    # try several names
                    for cand in ['StellarMass','StellarMassType','ApertureStellarMass','Mass_stars']:
                        if cand in sf:
                            try:
                                val = sf[cand][row]
                                print(" SOAP", cand, "=", val)
                            except Exception:
                                pass
        sf.close()
    except Exception as e:
        print("SOAP check failed:", e)
else:
    print("SOAP catalogue not present; skipped SOAP checks.")

f.close()
print("\nDone.")