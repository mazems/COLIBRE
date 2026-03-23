#!/usr/bin/env python3
import pandas as pd
fn = "plots_dor/relicness_merged_allabove9p9_with_DoR_variants.csv"
df = pd.read_csv(fn, low_memory=False)

# find DoR-like columns (case-insensitive)
dor_cols = [c for c in df.columns if c.lower().startswith("dor")]
print("Detected DoR-like columns:", dor_cols)

if len(dor_cols) == 0:
    raise SystemExit("No DoR-like columns found. Check header (see command A).")

# coerce to numeric and test > 1
dor_numeric = df[dor_cols].apply(pd.to_numeric, errors="coerce")
mask = (dor_numeric > 1).any(axis=1)

print(f"Rows with any DoR > 1: {mask.sum()}")

# show offending subhalo_id and the DoR columns and a few columns of interest
cols_to_print = ["subhalo_id"] + dor_cols
existing = [c for c in cols_to_print if c in df.columns]
offending = df.loc[mask, existing].copy()

# show first 50 on stdout
if not offending.empty:
    print(offending.head(50).to_string(index=False))
    # save full list
    offending.to_csv("dor_gt1_rows.csv", index=False)
    print("Saved all offending rows to dor_gt1_rows.csv")
else:
    print("No offending rows found.")