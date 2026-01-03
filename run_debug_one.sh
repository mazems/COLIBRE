#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 NSTART NEND" >&2
  exit 1
fi
NSTART="$1"
NEND="$2"

echo "=== debug run for chunk ${NSTART}_${NEND} ==="
date
echo "Activate conda..."
# adjust conda path if necessary
source /home/mzemsch/miniconda3/etc/profile.d/conda.sh
conda activate swift-conda

python3 - <<'PY'
import time, os, sys, traceback
import numpy as np, pandas as pd, h5py
from astropy.cosmology import Planck15

SNAP = "/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN/SOAP-HBT/colibre_with_SOAP_membership_0127.hdf5"
NPZ = "ucmg_particle_index_mapping.npz"
NSTART = int(os.environ.get("NSTART", sys.argv[1])) if False else int('${NSTART}')
NEND   = int(os.environ.get("NEND", sys.argv[2])) if False else int('${NEND}')

print("CWD:", os.getcwd())
t0_all = time.time()

# quick existence checks
print("Exists NPZ:", os.path.exists(NPZ))
if os.path.exists(NPZ):
    print(" NPZ size (MB):", os.path.getsize(NPZ)//(1024*1024))
else:
    print(" NPZ missing -> abort")
    sys.exit(2)

print("Exists SNAP:", os.path.exists(SNAP))
if os.path.exists(SNAP):
    print(" SNAP size (GB):", os.path.getsize(SNAP)//(1024*1024*1024))
else:
    print(" SNAP missing -> abort")
    sys.exit(2)

# load mapping with mmap to avoid memory copy
t0 = time.time()
print("Loading mapping with mmap_mode='r' (should be fast to open)...")
try:
    mp = np.load(NPZ, mmap_mode='r')
except Exception as e:
    print("np.load failed:", e)
    traceback.print_exc()
    sys.exit(3)
t1 = time.time()
print(f"np.load completed in {t1-t0:.3f} s; keys: {len(mp.files)}")
# show a few keys (first 10)
print("sample keys (first 10):", mp.files[:10])

# load ucmg_ids and pick slice
ucmg = pd.read_csv("ucmg_ids.csv", dtype=object)
subhalo_ids = ucmg.iloc[NSTART:NEND, 0].astype(int).tolist()
print("Subhalo ids for chunk:", subhalo_ids)

# open HDF5 and do small tests
with h5py.File(SNAP,'r') as f:
    print("Opened HDF5; PartType4 present?", 'PartType4' in f)
    p4 = f['PartType4']
    massds_name = None
    for name in ('InitialMasses','Masses','masses'):
        if name in p4:
            massds_name = name
            break
    if massds_name is None:
        print("No mass dataset found under PartType4; abort")
        sys.exit(4)
    masses_ds = p4[massds_name]
    print("Mass dataset:", massds_name, "shape:", masses_ds.shape)

    # iterate halos and do quick sample reads, then optional full read
    for i, sid in enumerate(subhalo_ids, start=1):
        print("\n--- HALO", i, "sid", sid, "START ---")
        t_s = time.time()
        key = str(sid)
        if key not in mp.files:
            print("  WARNING: sid not present in npz mapping keys")
            continue
        arr = np.load(NPZ, mmap_mode='r')[key]  # mmap safe
        arr = arr.astype('int64', copy=False)
        print("  mapping Nidx:", arr.size)
        if arr.size == 0:
            print("  zero particles -> skip")
            continue
        # quick coalesce summary
        idx = np.unique(arr)
        idx.sort()
        dif = np.diff(idx)
        n_runs = int((dif > 1).sum() + 1)
        print("  contiguous runs approx:", n_runs, "min,max:", int(idx[0]), int(idx[-1]))
        # small sample read (first up to 20 indices) - fast
        sample = idx[:20] if idx.size > 20 else idx
        t_r0 = time.time()
        try:
            sample_m = np.array(masses_ds[sample])
            t_r1 = time.time()
            print("  sample read OK (len):", sample_m.size, "time:", t_r1 - t_r0)
        except Exception as e:
            print("  sample read FAILED:", e)
            traceback.print_exc()
            print("--- halo", sid, "END (sample read failed) ---")
            continue

        # ask user: perform full read? we will attempt it but print time and allow you to kill
        print("  -> Now attempting full read of all unique indices for this halo (may be slow).")
        t_full0 = time.time()
        try:
            masses_all = np.array(masses_ds[idx], dtype=float)
            t_full1 = time.time()
            print("  full read OK count:", masses_all.size, "time:", t_full1 - t_full0)
        except Exception as e:
            print("  full read FAILED:", e)
            traceback.print_exc()
        print("--- halo", sid, "END. wall:", time.time() - t_s, "s ---")

print("All done, total wall:", time.time() - t0_all, "s")
PY

