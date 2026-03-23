#!/usr/bin/env python3
import pandas as pd
import os

# === CONFIG ===
file_main      = "relicness_merged_with_stellar_complete_updated.csv"
file_extras    = "relicness_merged_extras.csv"
outfile        = "relicness_merged_allabove9p9_rebuild.csv"   # NEW file; does NOT overwrite old one

# columns you requested to keep (if present in sources)
cols_keep = [
 'subhalo_id','soap_row_index','total_formed_mass','stellar_mass_current',
 't_start','t50','t50_span','t75','t75_span','t90','t90_span','t95','t95_span',
 't998','t998_span','tfin','tfin_span','f_Mz2','term1','term2','term3','DoR',
 'elem_H_mass','elem_He_mass','elem_C_mass','elem_N_mass','elem_O_mass',
 'elem_Ne_mass','elem_Mg_mass','elem_Si_mass','elem_Fe_mass','elem_Sr_mass',
 'elem_Ba_mass','elem_Eu_mass'
]

# candidate id columns to try
id_candidates = ["subhalo_id"]
def find_idcol(df):
    for c in id_candidates:
        if c in df.columns:
            return c
    # fallback: first column name
    return df.columns[0]

def read_and_prepare(fn):
    print(f"Reading {fn} ...")
    df = pd.read_csv(fn, low_memory=False)
    idcol = find_idcol(df)
    if idcol != "subhalo_id":
        df = df.rename(columns={idcol: "subhalo_id"})
        print(f" - Renamed id col {idcol} -> subhalo_id")
    else:
        print(" - Using id col 'subhalo_id'")
    # # trim whitespace in id column and coerce to int where possible
    # df['subhalo_id'] = df['subhalo_id'].astype(str).str.strip().str.replace(r'\r','', regex=True)
    # # coerce numeric - invalid -> NaN
    # df['subhalo_id_numeric'] = pd.to_numeric(df['subhalo_id'], errors='coerce').astype('Int64')
    # drop rows with no valid numeric id
    n_before = len(df)
    df = df[df['subhalo_id'].notna()].copy()
    n_after = len(df)
    print(f" - {n_before - n_after} rows dropped due to non-numeric subhalo_id")
    # # replace 'subhalo_id' by numeric integer index
    # df['subhalo_id'] = df['subhalo_id_numeric'].astype('int64')
    # df.drop(columns=['subhalo_id_numeric'], inplace=True)
    # select only requested columns if present
    present = [c for c in cols_keep if c in df.columns]
    df_sel = df[present].copy()
    missing = [c for c in cols_keep if c not in df.columns]
    if missing:
        print(f" - Note: the following requested columns are missing in {fn} and will be absent: {missing}")
    return df_sel

# read both files
df_main = read_and_prepare(file_main)
df_extras = read_and_prepare(file_extras)

print(f"Rows: main={len(df_main)}  extras={len(df_extras)}")

# concat — main first so its values take precedence when dropping duplicates
df_concat = pd.concat([df_main, df_extras], ignore_index=True, sort=False)
print("Concatenated rows:", len(df_concat))

# drop duplicates by subhalo_id, keep first occurrence (main wins)
n_before = len(df_concat)
df_concat = df_concat.drop_duplicates(subset=['subhalo_id'], keep='first')
n_after = len(df_concat)
print(f"Dropped {n_before-n_after} duplicate subhalo_id rows (kept first occurrence)")

# sort by id
df_concat = df_concat.sort_values('subhalo_id').reset_index(drop=True)

# write output
if os.path.exists(outfile):
    print("Warning: outfile already exists; will overwrite", outfile)
df_concat.to_csv(outfile, index=False)
print("Wrote", outfile)
print("Final row count:", len(df_concat))

# extra diagnostics: how many of the original 5059 expected ids appear?
# If you have a file of expected ids (e.g. ucgm list) you could compare here.