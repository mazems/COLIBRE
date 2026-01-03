import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# paths
h5path = '/mnt/su3ctm/kproctor/ForMax/L0200N3008_exsitu_summary.hdf5'
ucmg_file = 'ucmg_ids.csv'

# load UCMG IDs (your output says this column exists)
df_ids = pd.read_csv(ucmg_file)
sample_ids = df_ids['subhalo_id'].astype(int).values
print(f"Loaded {len(sample_ids)} UCMG IDs")

# read HDF5 file
with h5py.File(h5path, 'r') as f:
    data = f['stars'][:]   # shape (N, 4)

# unpack columns (based on file structure + colleague's description)
track_id   = data[:, 0].astype(int)
mass_ex    = data[:, 1]
mass_tot   = data[:, 2]
fraction   = data[:, 3]

# build lookup table
lookup = dict(zip(track_id, fraction))

# extract fractions for UCMGs
fractions = []
missing = 0
for tid in sample_ids:
    if tid in lookup:
        fractions.append(lookup[tid])
    else:
        missing += 1

fractions = np.array(fractions)

print(f"Matched {len(fractions)} galaxies")
print(f"Missing {missing} IDs")

# simple stats
print("Median ex-situ fraction:", np.nanmedian(fractions))
print("Mean ex-situ fraction:  ", np.nanmean(fractions))

# plot histogram
plt.figure(figsize=(6,4))
plt.hist(fractions, bins=20, edgecolor='black')
plt.xlabel("Ex-situ mass fraction")
plt.ylabel("Number of UCMGs")
plt.title("UCMG sample: ex-situ mass fractions")
plt.tight_layout()
plt.savefig("ucmg_exsitu_hist.png", dpi=150)
plt.show()