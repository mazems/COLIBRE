#!/usr/bin/env python3
import h5py, numpy as np, pandas as pd, sys

h5path = '/mnt/su3ctm/kproctor/ForMax/L0200N3008_exsitu_summary.hdf5'
ucmg_csv = 'ucmg_ids.csv'
relic_csv = 'relicness_merged_with_stellar.csv'   # adjust if different

# load ucmg ids
df_ucmg = pd.read_csv(ucmg_csv)
for c in ['subhalo_id','TrackId','id','HaloCatalogueIndex','halo_id']:
    if c in df_ucmg.columns:
        ucmg_ids = df_ucmg[c].astype(int).values
        idcol_ucmg = c
        break
else:
    raise SystemExit("No ID column found in ucmg_ids.csv")

print("Loaded", len(ucmg_ids), "UCMG IDs from", idcol_ucmg)

# load relicness CSV (the catalogue you mentioned)
df_relic = pd.read_csv(relic_csv)
print("relicness CSV columns:", list(df_relic.columns)[:40])

# try to find an integer ID column in relic CSV to match sample (common names)
for c in ['subhalo_id','HaloCatalogueIndex','TrackId','halo_id','id','subhaloID']:
    if c in df_relic.columns:
        relic_idcol = c
        break
else:
    print("No obvious ID column found in relic CSV. Provide the column name.")
    sys.exit(1)

# try to find a stellar-mass column in relic CSV
mass_candidates = ['stellar_mass_current']
found_mass = None
for cand in mass_candidates:
    if cand in df_relic.columns:
        found_mass = cand
        break
if found_mass is None:
    print("No obvious stellar-mass column found in relic CSV. Columns:", df_relic.columns.tolist()[:50])
    sys.exit(1)

print("Using relic CSV ID column:", relic_idcol, "and mass column:", found_mass)

# build relic mapping: id -> mass_from_relic_csv
relic_ids = df_relic[relic_idcol].astype(int).values
relic_mass = df_relic[found_mass].astype(float).values
map_relic = dict(zip(relic_ids, relic_mass))

# read HDF5 stars: track, mass_ex, mass_tot, fraction
with h5py.File(h5path,'r') as f:
    data = f['stars'][:]
track = data[:,0].astype(int)
mass_ex = data[:,1].astype(float)
mass_tot = data[:,2].astype(float)   # this is the 'total stellar mass' used in fraction
# make map for exsitu file
map_h5_mass = dict(zip(track, mass_tot))

# Compare masses for UCMG IDs which exist in both
common = []
missing_in_h5 = []
missing_in_relic = []
rel_masses = []
h5_masses = []
for tid in set(ucmg_ids):
    if tid not in map_h5_mass:
        missing_in_h5.append(tid)
        continue
    if tid not in map_relic:
        missing_in_relic.append(tid)
        continue
    h = map_h5_mass[tid]
    r = map_relic[tid]
    h5_masses.append(h)
    rel_masses.append(r)
    common.append(tid)

n_common = len(common)
print(f"Common IDs with masses: {n_common}")
if len(missing_in_h5)>0:
    print("Example IDs in ucmg but missing in HDF5:", missing_in_h5[:10])
if len(missing_in_relic)>0:
    print("Example IDs missing in relic CSV:", missing_in_relic[:10])

if n_common == 0:
    print("No common IDs to compare. Abort.")
    sys.exit(0)

h5_masses = np.array(h5_masses, dtype=float)
rel_masses = np.array(rel_masses, dtype=float)

# compute ratio and differences (be mindful of units)
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(rel_masses>0, h5_masses / rel_masses, np.nan)
    diff_frac = np.where(rel_masses>0, (h5_masses - rel_masses) / rel_masses, np.nan)

print("Mass comparison stats (HDF5_mass / relic_csv_mass):")
print("  count:", np.sum(~np.isnan(ratio)))
print("  median ratio:", np.nanmedian(ratio))
print("  mean ratio:  ", np.nanmean(ratio))
print("  std ratio:   ", np.nanstd(ratio))
print("  min ratio:   ", np.nanmin(ratio))
print("  max ratio:   ", np.nanmax(ratio))

# flag large mismatches
mismatch_mask = (~np.isnan(ratio)) & ((ratio < 0.5) | (ratio > 2.0))
if mismatch_mask.any():
    print(f"Found {mismatch_mask.sum()} matches with mass differing by > factor 2. Example mismatches (first 10):")
    inds = np.where(mismatch_mask)[0][:10]
    for i in inds:
        print(" ID:", common[i], " HDF5_mass:", h5_masses[i], " relic_mass:", rel_masses[i], " ratio:", ratio[i])
else:
    print("No large mismatches found (all within factor 2).")

# If masses differ by a constant scale (e.g., HDF5 in Msun/h), you might see median ratio != 1.
# If so, note the scaling and convert units as needed.