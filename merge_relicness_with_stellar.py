#!/usr/bin/env python3
import pandas as pd

PRIMARY = "relicness_merged.csv"
DONOR   = "relic_all_stellar.csv"
OUT     = "relicness_merged_with_stellar.csv"

print("Loading primary file:", PRIMARY)
df_main = pd.read_csv(PRIMARY)

print("Loading donor file:", DONOR)
df_stellar = pd.read_csv(DONOR)

# sanity checks
required_cols = {
    "subhalo_id",
    "stellar_mass_current",
    "stellar_halfmass_radius_kpc",
    "logsigma",
}
missing = required_cols - set(df_stellar.columns)
if missing:
    raise RuntimeError(f"Donor file missing columns: {missing}")

# keep only the columns we want from the donor
df_stellar_sel = df_stellar[
    [
        "subhalo_id",
        "stellar_mass_current",
        "stellar_halfmass_radius_kpc",
        "logsigma",
    ]
].copy()

# drop stellar_mass_current from primary if present
if "stellar_mass_current" in df_main.columns:
    print("Dropping stellar_mass_current from primary (will replace from donor)")
    df_main = df_main.drop(columns=["stellar_mass_current"])

# merge: left join keeps ALL galaxies from primary
df_merged = df_main.merge(
    df_stellar_sel,
    on="subhalo_id",
    how="left",
    validate="one_to_one",
)

# write new file only
df_merged.to_csv(OUT, index=False)

print("\n✅ Merge complete")
print("Primary rows:", len(df_main))
print("Merged rows :", len(df_merged))
print("Output file :", OUT)
print("Columns     :", list(df_merged.columns))