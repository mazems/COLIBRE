#!/usr/bin/env python3
import os, numpy as np, h5py, pandas as pd

MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN'
SNAP_FILE = 'colibre_with_SOAP_membership_0127.hdf5'
VIRTUAL_SNAPSHOT_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', SNAP_FILE)
UCMG_CSV = 'ucmg_ids.csv'

df = pd.read_csv(UCMG_CSV)
if 'subhalo_id' in df.columns:
    all_ids = df['subhalo_id'].astype(int).to_numpy()
else:
    all_ids = df.iloc[:,0].astype(int).to_numpy()

check_ids = [158660, 2855260, 1623156, 1623099]   # problematic ones you saw
print("Checking:", check_ids)

with h5py.File(VIRTUAL_SNAPSHOT_FILE,'r') as f:
    p4 = f['PartType4']
    print("PartType4 keys:", list(p4.keys()))
    # show basic global stats for mass datasets
    for cand in ('InitialMasses','Masses','masses'):
        if cand in p4:
            arr = p4[cand]
            # sample first few, and compute fraction of zeros
            sample = np.array(arr[:1000], dtype=float)
            zeros_frac = np.sum(sample == 0.0) / sample.size
            print(f"Dataset {cand}: shape={arr.shape}, dtype={arr.dtype}, sample zeros frac ~ {zeros_frac:.4f}")
    # global zero-count for the mass dataset we used earlier (Masses/InitialMasses)
    mass_name = None
    for cand in ('InitialMasses','Masses','masses'):
        if cand in p4:
            mass_name = cand
            break
    if mass_name is None:
        raise SystemExit("No mass dataset found")
    mass_ds = p4[mass_name]
    # compute exact fraction of zero masses (scan in chunks to avoid full load)
    n = int(mass_ds.shape[0])
    chunk = 2_000_000
    zeros = 0
    total = 0
    for start in range(0, n, chunk):
        stop = min(n, start+chunk)
        a = np.array(mass_ds[start:stop], dtype=float)
        zeros += np.count_nonzero(a == 0.0)
        total += a.size
    print(f"Global: {zeros}/{total} zero-mass particles  -> fraction {zeros/total:.6f}")

    # now per problematic halo: count non-zero masses and fraction
    halo_ds = p4['HaloCatalogueIndex']
    for sid in check_ids:
        # find particle indices (scan in chunks; but we'll collect for these few ids)
        idxs = []
        for start in range(0, int(halo_ds.shape[0]), chunk):
            stop = min(int(halo_ds.shape[0]), start+chunk)
            chunkarr = np.array(halo_ds[start:stop], dtype=np.int64)
            rel = np.nonzero(chunkarr == sid)[0]
            if rel.size:
                idxs.append(rel + start)
        if len(idxs) == 0:
            print(f"SID {sid}: 0 particles")
            continue
        indices = np.concatenate(idxs).astype(np.int64)
        masses_sel = np.array(mass_ds[indices], dtype=float)
        print(f"SID {sid}: n_particles={indices.size}, n_zero_masses={np.count_nonzero(masses_sel==0.0)}, n_nonzero={np.count_nonzero(masses_sel!=0.0)}")
        # show small stats
        if np.count_nonzero(masses_sel!=0.0) > 0:
            nz = masses_sel[masses_sel!=0.0]
            print("  nonzero min/max/mean:", nz.min(), nz.max(), nz.mean())
        # show whether birthscale and elem are zero arrays for these indices (sample)
        if 'BirthScaleFactors' in p4:
            b = np.array(p4['BirthScaleFactors'][indices[:100]], dtype=float)
            print("  BirthScaleFactors sample unique (first 10):", np.unique(b)[:10])
        if 'ElementMassFractions' in p4:
            em = np.array(p4['ElementMassFractions'][indices[:100]], dtype=float)
            print("  ElementMassFractions sample row sum (first 5):", np.sum(em[:5], axis=1))