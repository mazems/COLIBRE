#!/usr/bin/env python3
"""
Recompute DoR90 / DoR95 / DoR998 from merged SFH CSV.

- Expects the merged CSV (gzipped) to contain at least:
    - term1, term2, term3          (originally saved)
    - tfin_span, t90_span, t95_span, t998_span   (span columns)
    - optionally f_Mz2
- Produces merged_with_DoR_variants.csv.gz with new columns:
    DoR90, DoR95, DoR998, term3_t90, term3_t95, term3_t998
- Attempts to compute cosmic age at z=0 using astropy; falls back to 13.8 Gyr.
"""
import os
import sys
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as au


# cosmology params from COLIBRE 2025
COLIBRE_H = 0.681
COLIBRE_OMEGAM = 0.306
COLIBRE_OMEGAL = 0.693922

COSMO_PARAMS = dict(h=COLIBRE_H, omegam=COLIBRE_OMEGAM, omegal=COLIBRE_OMEGAL)
cosmo_colibre = FlatLambdaCDM(H0=100.0 * COLIBRE_H, Om0=COLIBRE_OMEGAM if 'COLIBRE_OMEGAM' in globals() else COLIBRE_OMEGAM)

# path to merged file (adjust if needed)
in_fn = "sfh_times_all.csv.gz"
out_fn = "sfh_times_all_with_DoR_variants.csv.gz"

if not os.path.exists(in_fn):
    raise SystemExit(f"Input file not found: {in_fn}")

print("Reading:", in_fn)
df = pd.read_csv(in_fn, low_memory=False)

# required span columns (some may be missing; script will handle gracefully)
span_cols = {
    "tfin": "tfin_span",
    "t90": "t90_span",
    "t95": "t95_span",
    "t998": "t998_span"
}

# try to get cosmic age at z=0 in Gyr using astropy; fallback to 13.8 Gyr
t_universe_gyr = None
try:
    # from astropy.cosmology import Planck15 as cosmo
    # from astropy import units as u
    # t_universe_gyr = float(cosmo.age(0).to(u.Gyr).value)
    # print("Using astropy.Planck15 age(0) =", t_universe_gyr, "Gyr")
    t_universe_gyr = float(cosmo_colibre.age(0).to(au.Gyr).value)
    print("Using colibre cosmo age(0) =", t_universe_gyr, "Gyr")
except Exception:
    t_universe_gyr = 13.8
    print("astropy not available or failed; using fallback t_universe_gyr =", t_universe_gyr, "Gyr")

# helper to compute term3 given a span column name
def compute_term3(span_series):
    # term3 = (0.7 + t_universe_gyr - span_val) / t_universe_gyr  if span finite else NaN
    span = span_series.astype(float)
    term3 = np.where(np.isfinite(span), (0.7 + t_universe_gyr - span) / t_universe_gyr, np.nan)
    return term3

# check we have term1 and term2 in the file; if not but f_Mz2 exists, fill term1 from it
if "f_Mz2" not in df.columns and "term1" in df.columns:
    print("f_Mz2 missing; copying term1 -> f_Mz2 (as in original code term1 == f_Mz2).")
    df["f_Mz2"] = df["term1"]

if "term2" not in df.columns:
    print("Warning: term2 column not found in CSV. Without term2 the DoR will be NaN. Aborting.")
    # we could attempt to recompute term2 from t75_span if present, but original code used 0.5/t75_span fallback 1.0
    # check if t75_span present to attempt recompute:
    if "t75_span" in df.columns:
        print("Attempting to recompute term2 from t75_span column (term2 = 0.5 / t75_span ; fallback 1.0).")
        t75 = df["t75_span"].astype(float)
        df["term2"] = np.where((np.isfinite(t75) & (t75 > 0.0)), 0.5 / t75, 1.0)
    else:
        raise SystemExit("term2 not found and t75_span not available; cannot reconstruct DoR.")

# compute term3 for each variant if the corresponding span column exists
for key, col in span_cols.items():
    if col in df.columns:
        df[f"term3_{key}"] = compute_term3(df[col])
        print(f"Computed term3_{key} from column '{col}'")
    else:
        df[f"term3_{key}"] = np.nan
        print(f"Column '{col}' not found -> term3_{key} set to NaN")

# Now compute DoR variants: mean(term1, term2, term3_variant) when all are finite
for key in ("t90", "t95", "t998", "tfin"):
    term3_col = f"term3_{key}"
    dor_col = f"DoR_{key}"  # tfin probably already under dor, but keep consistent
    # compute mean across term1, term2, term3_col
    t1 = df["f_Mz2"].astype(float) # use f_Mz2 here for consistency, term1 in table is computed from masses instead of tb and hence differs slightly
    t2 = df["term2"].astype(float)
    t3 = df[term3_col].astype(float) if term3_col in df.columns else np.full(len(df), np.nan)
    valid = np.isfinite(t1) & np.isfinite(t2) & np.isfinite(t3)
    dor = np.full(len(df), np.nan)
    dor[valid] = (t1[valid] + t2[valid] + t3[valid]) / 3.0
    df[dor_col] = dor
    print(f"Computed {dor_col}: {np.sum(np.isfinite(dor))} finite rows")

# # Provide a short diagnostic comparing f_Mz2 and term1 if both exist
# if ("f_Mz2" in df.columns) and ("term1" in df.columns):
#     diff = df["f_Mz2"].astype(float) - df["term1"].astype(float)
#     # show summary statistics
#     n_diff = np.sum(~np.isclose(df["f_Mz2"].astype(float), df["term1"].astype(float), atol=1e-12, rtol=0))
#     print(f"Rows where f_Mz2 != term1 (abs tol 1e-12): {n_diff} / {len(df)}")
#     if n_diff > 0:
#         print("Difference stats (non-NaN diffs):")
#         diffa = diff[np.isfinite(diff)]
#         print("  min, 5th, median, 95th, max:", np.nanmin(diffa), np.nanpercentile(diffa,5),
#               np.nanmedian(diffa), np.nanpercentile(diffa,95), np.nanmax(diffa))
#         # print up to 10 sample mismatches
#         idx_diff = np.where(~np.isclose(df["f_Mz2"].astype(float), df["term1"].astype(float), atol=1e-12, rtol=0))[0]
#         sample_idx = idx_diff[:10]
#         print("Showing up to 10 sample indices and values (index, f_Mz2, term1, diff):")
#         for ii in sample_idx:
#             print(ii, df.loc[ii, "f_Mz2"], df.loc[ii, "term1"], df.loc[ii, "f_Mz2"] - df.loc[ii, "term1"])
#     else:
#         print("f_Mz2 and term1 match to ~1e-12 precision across the file.")

# Save to new gzipped CSV (do not overwrite original)
print("Writing output:", out_fn)
df.to_csv(out_fn, index=False, compression="gzip")
print("Done.")