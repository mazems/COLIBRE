#!/usr/bin/env python3
"""
Recompute DoR95 / DoR998 (corrected: use actual tX times, not tX_span) from merged SFH CSV.

Input:  in_fn = "sfh_times_all.csv.gz"
Output: out_fn = "sfh_times_all_with_DoR_variants_corrected.csv.gz"

Behaviour:
 - Uses COLIBRE cosmology defined below to obtain cosmic ages when needed.
 - Reads f_Mz2 (term1) or term1 as fallback.
 - For term2 uses t75 (actual time) -> term2 = 0.5 / t75 (Gyr). If t75 missing, will fallback to t75_span and warn.
 - For term3(95/998) uses t95/t998 actual times. If missing, falls back to respective _span columns and warns.
 - Produces columns: term3_t95, term3_t998, DoR_t95, DoR_t998
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as au

# ------------------ CONFIG (COLIBRE cosmology) ------------------
COLIBRE_H = 0.681
COLIBRE_OMEGAM = 0.306
COLIBRE_OMEGAL = 0.693922

cosmo_colibre = FlatLambdaCDM(H0=100.0 * COLIBRE_H, Om0=COLIBRE_OMEGAM)

# filepaths
in_fn = "sfh_times_all.csv.gz"
out_fn = "sfh_times_all_with_DoR_variants_corrected.csv.gz"

if not os.path.exists(in_fn):
    raise SystemExit(f"Input file not found: {in_fn}")

print("Reading:", in_fn)
df = pd.read_csv(in_fn, low_memory=False)

# cosmic age at z=0 (Gyr) using COLIBRE cosmology above
t_universe_gyr = float(cosmo_colibre.age(0).to(au.Gyr).value)
t_z2_gyr = float(cosmo_colibre.age(2.0).to(au.Gyr).value)
print(f"COLIBRE age(0) = {t_universe_gyr:.4g} Gyr ; age(z=2) = {t_z2_gyr:.4g} Gyr")

# helper to pick a primary column or fallback to span
def pick_time(df, primary, fallback_span=None):
    """
    Return (arr, used_name, used_is_span_bool)
    arr is numpy float array in Gyr (NaN where missing)
    """
    if primary is not None and primary in df.columns:
        return pd.to_numeric(df[primary], errors="coerce").to_numpy(dtype=float), primary, False
    if fallback_span is not None and fallback_span in df.columns:
        print(f"Warning: primary '{primary}' not found; using fallback span '{fallback_span}'.")
        return pd.to_numeric(df[fallback_span], errors="coerce").to_numpy(dtype=float), fallback_span, True
    # not available
    print(f"Warning: neither '{primary}' nor fallback '{fallback_span}' found; resulting values will be NaN.")
    return np.full(len(df), np.nan, dtype=float), None, False

# --- retrieve term1 source (f_Mz2) ---
if "f_Mz2" in df.columns:
    term1_arr = pd.to_numeric(df["f_Mz2"], errors="coerce").to_numpy(dtype=float)
    print("Using f_Mz2 as term1 source.")
elif "term1" in df.columns:
    term1_arr = pd.to_numeric(df["term1"], errors="coerce").to_numpy(dtype=float)
    print("Using 'term1' column as term1 source.")
else:
    term1_arr = np.full(len(df), np.nan, dtype=float)
    print("Warning: no f_Mz2 or term1 in CSV -> term1 set to NaN.")

# --- times: prefer t75, t95, t998 (actual times). fallback to *_span if necessary
t75_arr, t75_used, t75_is_span = pick_time(df, "t75", "t75_span")
t95_arr, t95_used, t95_is_span = pick_time(df, "t95", "t95_span")
t998_arr, t998_used, t998_is_span = pick_time(df, "t998", "t998_span")
tfin_arr, tfin_used, tfin_is_span = pick_time(df, "tfin", "tfin_span")

print("Columns used: t75:", t75_used, "t95:", t95_used, "t998:", t998_used, "tfin:", tfin_used)

# --- compute term2 from t75: term2 = 0.5 / t75  (units: Gyr)
# If t75 is missing or <=0 we'll set NaN and handle fallback to 1.0 below.
with np.errstate(divide="ignore", invalid="ignore"):
    term2_arr = np.where(np.isfinite(t75_arr) & (t75_arr > 0.0), 0.5 / t75_arr, np.nan)

# If term2 NaN but t75_span exists and was the fallback, we already used it above (pick_time).
# As a final fallback, match your old behaviour: set term2 -> 1.0 where still NaN.
n_t2_nan = int(np.sum(~np.isfinite(term2_arr)))
if n_t2_nan > 0:
    print(f"term2: {n_t2_nan} entries missing or invalid -> setting fallback 1.0 (final fallback).")
    term2_arr[~np.isfinite(term2_arr)] = 1.0

# --- compute term3 from actual tX times: term3 = (0.7 + t_universe - tX) / t_universe   (if tX finite)
def term3_from_time(tX_arr):
    tX = np.asarray(tX_arr, dtype=float)
    return np.where(np.isfinite(tX), (0.7 + t_universe_gyr - tX) / t_universe_gyr, np.nan)

term3_t95 = term3_from_time(t95_arr)
term3_t998 = term3_from_time(t998_arr)
# also optionally term3_tfin if you want:
term3_tfin = term3_from_time(tfin_arr)

# --- compute DoR variants: mean(term1, term2, term3_variant) when all finite
def compute_dor_from_terms(t1, t2, t3):
    t1 = np.asarray(t1, dtype=float)
    t2 = np.asarray(t2, dtype=float)
    t3 = np.asarray(t3, dtype=float)
    dor = np.full(len(t1), np.nan, dtype=float)
    ok = np.isfinite(t1) & np.isfinite(t2) & np.isfinite(t3)
    dor[ok] = (t1[ok] + t2[ok] + t3[ok]) / 3.0
    return dor, ok

DoR_t95, ok95 = compute_dor_from_terms(term1_arr, term2_arr, term3_t95)
DoR_t998, ok998 = compute_dor_from_terms(term1_arr, term2_arr, term3_t998)
DoR_tfin, okfin = compute_dor_from_terms(term1_arr, term2_arr, term3_tfin)

# attach to dataframe
df["term2_from_t75"] = term2_arr
df["term3_t95"] = term3_t95
df["term3_t998"] = term3_t998
df["term3_tfin"] = term3_tfin
df["DoR_t95"] = DoR_t95
df["DoR_t998"] = DoR_t998
df["DoR_tfin"] = DoR_tfin

# diagnostics
print("Computed DoR counts (finite):")
print("  DoR_t95_corrected:", int(np.sum(np.isfinite(DoR_t95))), " / ", len(DoR_t95))
print("  DoR_t998_corrected:", int(np.sum(np.isfinite(DoR_t998))), " / ", len(DoR_t998))
print("  DoR_tfin_corrected:", int(np.sum(np.isfinite(DoR_tfin))), " / ", len(DoR_tfin))

# # optional: if original DoR columns exist, print a quick compare summary
# for col in ("DoR_t95", "DoR_t998", "DoR_tfin"):
#     if col in df.columns:
#         orig = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
#         new = df.get(col + "_corrected", None)
#         if new is None:
#             # we named the new columns DoR_t95_corrected etc.
#             new = df.get(col + "_corrected".replace("_corrected","_corrected"), None)
#         if new is not None:
#             # count differing finite elements
#             bothfinite = np.isfinite(orig) & np.isfinite(new)
#             if bothfinite.sum() > 0:
#                 diff = np.abs(orig[bothfinite] - new[bothfinite])
#                 print(f"  compare {col}: both finite {bothfinite.sum()} ; median abs diff = {np.nanmedian(diff):.4g}")

# ----------------------------------------------------------
# Create small human-readable test file (first 20 galaxies)
# ----------------------------------------------------------
n_test = 20  # change if you want more
test_cols = [
    "subhalo_id",
    "f_Mz2", "term2_from_t75",
    "t75", "t95", "t998", "tfin",
    "term3_t95", "term3_t998", "term3_tfin",
    "DoR_t95", "DoR_t998", "DoR_tfin"
]

# keep only columns that actually exist
test_cols_existing = [c for c in test_cols if c in df.columns]

df_test = df[test_cols_existing].head(n_test).copy()

test_fn = "sfh_times_DoR_corrected_TEST.csv"
df_test.to_csv(test_fn, index=False)
print(f"Wrote small test file: {test_fn}")

# write out corrected CSV (gzipped)
print("Writing output:", out_fn)
df.to_csv(out_fn, index=False, compression="gzip")
print("Done.")