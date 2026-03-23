"""
Replaces missing values in the final relicness data table complete_fn by recomputed ones in missing_fn and saves it as out_fn"
"""

import os, shutil, time
import pandas as pd
import numpy as np

# filenames (edit if needed)
complete_fn = "relicness_merged_with_stellar_complete.csv"
missing_fn  = "relicness_merged_missing.csv"
out_fn      = "relicness_merged_with_stellar_complete_updated.csv"

# safety: check files exist
for fn in (complete_fn, missing_fn):
    if not os.path.exists(fn):
        raise SystemExit(f"ERROR: required file not found: {fn}")

# make a safe backup of the complete file (timestamped)
bak = f"{complete_fn}.{int(time.time())}.bak"
shutil.copy2(complete_fn, bak)
print("Backup of complete file created:", bak)

# read CSVs (object dtype is robust; we'll convert indices explicitly)
df_complete = pd.read_csv(complete_fn, dtype=object)
df_missing  = pd.read_csv(missing_fn, dtype=object)

# ensure subhalo_id exists
if 'subhalo_id' not in df_complete.columns or 'subhalo_id' not in df_missing.columns:
    raise SystemExit("subhalo_id missing from one of the input files")

# list of columns to replace (your requested set)
cols_to_replace = [
    "soap_row_index","total_formed_mass","stellar_mass_current","t_start","t50","t50_span",
    "t75","t75_span","t90","t90_span","t95","t95_span","t998","t998_span","tfin","tfin_span",
    "f_Mz2","term1","term2","term3","DoR",
    "elem_H_mass","elem_He_mass","elem_C_mass","elem_N_mass","elem_O_mass","elem_Ne_mass",
    "elem_Mg_mass","elem_Si_mass","elem_Fe_mass","elem_Sr_mass","elem_Ba_mass","elem_Eu_mass"
]

# restrict to columns actually present in missing file
cols_present_in_missing = [c for c in cols_to_replace if c in df_missing.columns]
if not cols_present_in_missing:
    raise SystemExit("None of the requested replacement columns are present in the missing CSV")

print("Columns found in missing file and to be replaced:", cols_present_in_missing)

# set index by subhalo_id for both frames (preserve original dtype by converting to int64 index)
# first ensure subhalo_id parseable as ints
df_complete['subhalo_id'] = df_complete['subhalo_id'].astype(np.int64)
df_missing['subhalo_id']  = df_missing['subhalo_id'].astype(np.int64)

df_complete_idx = df_complete.set_index('subhalo_id', drop=False)
df_missing_idx  = df_missing.set_index('subhalo_id', drop=False)

# matching ids
matching_ids = df_complete_idx.index.intersection(df_missing_idx.index)
print("Number of matching subhalo_id between complete and missing:", len(matching_ids))
if len(matching_ids) == 0:
    print("No overlap between complete and missing files. Exiting without changes.")
    raise SystemExit(0)

# trimmed missing table with only the relevant columns (aligned by index)
missing_trim = df_missing_idx.loc[matching_ids, cols_present_in_missing]

# optional: attempt to coerce numeric types for numeric-looking columns in missing_trim
for col in cols_present_in_missing:
    # try convert missing values to numeric where sensible (coerce errors -> NaN)
    try:
        missing_trim[col] = pd.to_numeric(missing_trim[col], errors='coerce')
    except Exception:
        pass

# report how many cells will change (approx)
total_changes = 0
for col in cols_present_in_missing:
    old_vals = df_complete_idx.loc[matching_ids, col].astype(object)
    new_vals = missing_trim.loc[matching_ids, col].astype(object)
    neq = (old_vals.fillna("__NaN__") != new_vals.fillna("__NaN__")).sum()
    print(f"Column {col}: will change {int(neq)} / {len(matching_ids)} rows")
    total_changes += int(neq)
print("Total approx cell changes:", total_changes)

# perform the update on the indexed DataFrame
for col in cols_present_in_missing:
    df_complete_idx.loc[matching_ids, col] = missing_trim.loc[matching_ids, col]

# restore to a regular DataFrame and ensure subhalo_id is a column (NOT lost)
df_updated = df_complete_idx.reset_index(drop=True)   # temporary drop to reindex rows
# BUT we want to keep the original row order of df_complete. So do a safe re-merge to preserve order:
df_ordered = df_complete.copy()
df_ordered['subhalo_id'] = df_ordered['subhalo_id'].astype(np.int64)
df_ordered = df_ordered.set_index('subhalo_id', drop=False)
for col in cols_present_in_missing:
    df_ordered.loc[matching_ids, col] = df_complete_idx.loc[matching_ids, col]
# finally, reset index to a normal column ordering exactly as original (with 'subhalo_id' still present)
df_final = df_ordered.reset_index(drop=True)

# write output (original complete file untouched)
if os.path.exists(out_fn):
    # backup any existing output filename
    bak2 = f"{out_fn}.{int(time.time())}.bak"
    shutil.copy2(out_fn, bak2)
    print("Existing output backed up to:", bak2)

df_final.to_csv(out_fn, index=False)
print("Wrote updated file:", out_fn)
print("Original complete file retained at:", complete_fn)
print("Backup of complete file:", bak)