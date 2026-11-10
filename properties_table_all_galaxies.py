#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import h5py
import common

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "L0200N3008/THERMAL_AGN/"
MODEL_DIR = "/mnt/su3-pro/colibre/" + MODEL_NAME
SNAP_FILE = "0127"

DOR_CSV = "sfh_times_all_with_DoR_variants_corrected.csv.gz"
EXSITU_H5 = "/mnt/su3ctm/kproctor/ForMax/exsitu_summary_SnapNum_127.hdf5"

BH_LOOKUP_CSV = "corrected_bh_mass_lookup.csv"

OUTFILE = "z0_galaxy_properties_all.csv"

MASS_CUT = 1e9
Z_SUN = 0.0134


# ============================================================
# READ SOAP
# ============================================================

print("Reading SOAP...")

fields_sgn = {
    "InputHalos": (
        "HaloCatalogueIndex",
        "IsCentral",
        "HBTplus/DescendantTrackId",
        "HBTplus/TrackId"
    )
}

fields = {
    "ExclusiveSphere/50kpc": (
        "StellarMass",
        "StarFormationRate",
        "HalfMassRadiusStars",
        "LuminosityWeightedMeanStellarAge",
        "LinearMassWeightedIronOverHydrogenOfStars",
        "LinearMassWeightedMagnesiumOverHydrogenOfStars",
        "MostMassiveBlackHoleMass",
        "StellarMassFractionInMetals",
    )
}

h5data_sgn = common.read_group_data_colibre(
    MODEL_DIR, SNAP_FILE, fields_sgn
)

h5data = common.read_group_data_colibre(
    MODEL_DIR, SNAP_FILE, fields
)

(
    halo_index,
    is_central,
    desc_id,
    track_id
) = h5data_sgn

(
    stellar_mass_raw,
    sfr_raw,
    rhalf_raw,
    stellarage_lum_raw,
    Fe_lin,
    Mg_lin,
    bh_mass_raw,
    Zstar_raw
) = h5data


# ============================================================
# UNIT CONVERSIONS
# ============================================================

Mu = 1.988e43 / 1.989e33
tu = 3.086e19 / 3.154e7

stellar_mass = stellar_mass_raw * Mu
sfr = sfr_raw * Mu / tu
rhalf = rhalf_raw * 1e3
lum_weighted_age = stellarage_lum_raw * tu / 1e9

# Keep both raw and corrected versions separately
bh_mass_raw_Msun = bh_mass_raw * Mu
bh_mass = bh_mass_raw_Msun.copy()


# ============================================================
# CORRECT ZERO BH MASSES
# ============================================================

print("Applying BH correction lookup...")

bh_lookup = pd.read_csv(BH_LOOKUP_CSV)

bh_dict = dict(
    zip(
        bh_lookup["track_id"].astype(np.int64),
        bh_lookup["corrected_bh_mass"].astype(float)
    )
)

n_replaced = 0

for i, tid in enumerate(track_id.astype(np.int64)):
    if bh_mass[i] == 0 and tid in bh_dict:
        bh_mass[i] = bh_dict[tid]
        n_replaced += 1

print("BH masses replaced:", n_replaced)


# ============================================================
# SELECT Z=0 GALAXIES
# ============================================================

select = (
    (stellar_mass >= MASS_CUT) &
    (stellar_mass > 0) &
    (rhalf > 0)
)

stellar_mass = stellar_mass[select]
sfr = sfr[select]
rhalf = rhalf[select]
lum_weighted_age = lum_weighted_age[select]
Fe_lin = Fe_lin[select]
Mg_lin = Mg_lin[select]

# Select both raw and corrected BH masses
bh_mass_raw_Msun = bh_mass_raw_Msun[select]
bh_mass = bh_mass[select]

Zstar_raw = Zstar_raw[select]

halo_index = halo_index[select]
track_id = track_id[select]
is_central = np.asarray(is_central[select]).astype(bool)

print("Selected galaxies:", len(stellar_mass))


# ============================================================
# DERIVED QUANTITIES
# ============================================================

log10_stellar_mass = np.log10(stellar_mass)

log10_half_mass_radius = np.log10(rhalf)

compactness = (
    log10_stellar_mass
    - 1.5 * log10_half_mass_radius
)


# Mg/Fe
with np.errstate(divide="ignore", invalid="ignore"):
    MgFe = np.where(
        (Mg_lin > 0) & (Fe_lin > 0),
        np.log10(Mg_lin / Fe_lin) - 0.10,
        np.nan
    )


# [Z/H]
Zstar = np.asarray(Zstar_raw, dtype=float)

with np.errstate(divide="ignore", invalid="ignore"):
    Z_H = np.where(
        (Zstar > 0) & np.isfinite(Zstar),
        np.log10(Zstar / Z_SUN),
        np.nan
    )


# log10(sSFR)
with np.errstate(divide="ignore", invalid="ignore"):
    ssfr = np.where(
        (stellar_mass > 0) & np.isfinite(sfr),
        sfr / stellar_mass,
        np.nan
    )

    log10_sSFR = np.where(
        ssfr > 0,
        np.log10(ssfr),
        np.nan
    )


# ============================================================
# BH / STELLAR MASS RATIOS
# ============================================================

# Raw / uncorrected BH ratio
with np.errstate(divide="ignore", invalid="ignore"):
    bh_ratio_raw = np.where(
        (bh_mass_raw_Msun > 0) &
        (stellar_mass > 0) &
        np.isfinite(bh_mass_raw_Msun),
        bh_mass_raw_Msun / stellar_mass,
        np.nan
    )

    log10_BH_to_stellar_mass_raw = np.where(
        bh_ratio_raw > 0,
        np.log10(bh_ratio_raw),
        np.nan
    )


# Corrected BH ratio
with np.errstate(divide="ignore", invalid="ignore"):
    bh_ratio = np.where(
        (bh_mass > 0) &
        (stellar_mass > 0) &
        np.isfinite(bh_mass),
        bh_mass / stellar_mass,
        np.nan
    )

    log10_BH_to_stellar_mass = np.where(
        bh_ratio > 0,
        np.log10(bh_ratio),
        np.nan
    )


# Flag galaxies whose zero SOAP BH mass was corrected
BH_mass_corrected = (
    (bh_mass_raw_Msun == 0) &
    (bh_mass > 0)
)


# ============================================================
# DO R
# ============================================================

print("Reading DoR catalogue...")

df_dor = pd.read_csv(
    DOR_CSV,
    low_memory=False
)

# Find ID column
if "subhalo_id" in df_dor.columns:
    dor_id_col = "subhalo_id"
elif "HaloCatalogueIndex" in df_dor.columns:
    dor_id_col = "HaloCatalogueIndex"
else:
    raise RuntimeError(
        "Could not find subhalo_id or HaloCatalogueIndex in DoR CSV."
    )

df_dor["HaloCatalogueIndex"] = pd.to_numeric(
    df_dor[dor_id_col],
    errors="coerce"
)

# Use your chosen DoR definition
if "DoR_t95" in df_dor.columns:
    dor_col = "DoR_t95"
else:
    raise RuntimeError("DoR_t95 not found in DoR CSV.")

dor_lookup = (
    df_dor[
        ["HaloCatalogueIndex", dor_col]
    ]
    .dropna(subset=["HaloCatalogueIndex"])
    .drop_duplicates("HaloCatalogueIndex")
    .set_index("HaloCatalogueIndex")[dor_col]
)

DoR = (
    pd.Series(dor_lookup)
    .reindex(halo_index.astype(np.int64))
    .to_numpy(dtype=float)
)


# ============================================================
# EX-SITU FRACTION
# ============================================================

print("Reading ex-situ catalogue...")

exsitu_lookup = {}

if os.path.exists(EXSITU_H5):

    with h5py.File(EXSITU_H5, "r") as f:

        if "stars" in f:
            data = np.array(f["stars"])

            if data.ndim == 2 and data.shape[1] >= 4:

                candidate_cols = (0, 1, 2)

                overlaps = [
                    np.intersect1d(
                        data[:, c].astype(np.int64),
                        halo_index.astype(np.int64)
                    ).size
                    for c in candidate_cols
                ]

                keycol = candidate_cols[int(np.argmax(overlaps))]

                print(
                    "Using ex-situ key column:",
                    keycol,
                    "overlaps:",
                    overlaps
                )

                ids = data[:, keycol].astype(np.int64)
                exfrac = data[:, 3].astype(float)

                exsitu_lookup = dict(
                    zip(ids.tolist(), exfrac.tolist())
                )

        else:
            print("No 'stars' dataset found.")

exsitu_series = pd.Series(exsitu_lookup, dtype=float)

exsitu_frac = (
    exsitu_series
    .reindex(halo_index.astype(np.int64))
    .to_numpy(dtype=float)
)


# ============================================================
# MAKE FINAL TABLE
# ============================================================

out = pd.DataFrame({
    "HaloCatalogueIndex": halo_index.astype(np.int64),
    "track_id": track_id.astype(np.int64),
    "is_central": is_central,

    "stellar_mass_Msun": stellar_mass,
    "log10_stellar_mass": log10_stellar_mass,

    "half_mass_radius_kpc": rhalf,
    "log10_half_mass_radius": log10_half_mass_radius,

    "compactness": compactness,

    "DoR": DoR,
    "exsitu_frac": exsitu_frac,

    "lum_weighted_age_Gyr": lum_weighted_age,
    "log10_sSFR": log10_sSFR,

    "[Z/H]": Z_H,
    "[Mg/Fe]": MgFe,

    # Raw / uncorrected BH quantities
    "BH_mass_raw_Msun": bh_mass_raw_Msun,
    "log10_BH_to_stellar_mass_raw": log10_BH_to_stellar_mass_raw,

    # Corrected BH quantities
    "BH_mass_corrected_Msun": bh_mass,
    "log10_BH_to_stellar_mass_corrected": log10_BH_to_stellar_mass,

    # True if raw SOAP BH mass was zero and lookup supplied a non-zero value
    "BH_mass_corrected_flag": BH_mass_corrected,
})


# ============================================================
# SAVE
# ============================================================

out.to_csv(OUTFILE, index=False)

print()
print("Saved:", OUTFILE)
print("Number of rows:", len(out))
print("Number of columns:", len(out.columns))
print()
print(out.head())