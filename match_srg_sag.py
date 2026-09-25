#!/usr/bin/env python3
"""
match_srg_sag.py

Match compact relic galaxies (SRGs) to non-compact SAGs with similar
stellar-population properties.

Matching quantities:
    logM
    age_lum_gyr
    MgFe
    logZstar_rel
    log_ssfr

NOT used for matching:
    compactness
    DoR
    sigma
    exsitu_frac
    black-hole properties

For every SRG, find the k nearest non-compact SAGs in the
5-dimensional standardised stellar-population space.

The SAG sample can be used as a matching pool with replacement:
the same SAG may therefore be matched to several SRGs. This is
intentional and gives every SRG the same number of comparison objects.

Outputs:
    srg_sag_matches.csv
    srg_sag_match_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

SRG_DEFAULT = "relic_master_properties.csv"
SAG_DEFAULT = "noncompact_sag_master_properties.csv"

MATCH_QUANTITIES = [
    "logM",
    "age_lum_gyr",
    "MgFe",
    "logZstar_rel",
    "log_ssfr",
]


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def robust_scale_fit(srg: pd.DataFrame, sag: pd.DataFrame, quantities: list[str]):
    """
    Determine robust location and scale from the combined SRG + SAG sample.

    Location = median
    Scale    = IQR

    Using the combined sample ensures that each dimension is put on
    approximately comparable footing without being dominated by outliers.
    """
    combined = pd.concat(
        [srg[quantities], sag[quantities]],
        ignore_index=True
    )

    medians = {}
    scales = {}

    for q in quantities:
        x = pd.to_numeric(combined[q], errors="coerce")

        med = np.nanmedian(x)
        q25 = np.nanpercentile(x, 25)
        q75 = np.nanpercentile(x, 75)
        scale = q75 - q25

        # Prevent division by zero for a constant quantity.
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        medians[q] = med
        scales[q] = scale

    return medians, scales


def standardise(df: pd.DataFrame, quantities, medians, scales):
    """
    Robustly standardise each matching quantity.
    """
    arr = np.column_stack([
        (pd.to_numeric(df[q], errors="coerce").to_numpy(dtype=float)
         - medians[q]) / scales[q]
        for q in quantities
    ])

    return arr


def check_columns(df, name, quantities):
    missing = [q for q in quantities if q not in df.columns]

    if "HaloCatalogueIndex" not in df.columns:
        missing.append("HaloCatalogueIndex")

    if missing:
        raise RuntimeError(
            f"{name} is missing required columns: {missing}"
        )


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Match SRGs to non-compact SAGs using stellar-population similarity."
    )

    parser.add_argument(
        "--srg",
        default=SRG_DEFAULT,
        help="SRG master-properties CSV.",
    )

    parser.add_argument(
        "--sag",
        default=SAG_DEFAULT,
        help="Non-compact SAG master-properties CSV.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of SAG neighbours per SRG.",
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=3.0,
        help=(
            "Maximum allowed matching distance in the standardised "
            "5D space. Matches farther away are discarded."
        ),
    )

    parser.add_argument(
        "--output",
        default="srg_sag_matches.csv",
        help="Output detailed pair table.",
    )

    parser.add_argument(
        "--summary-output",
        default="srg_sag_match_summary.csv",
        help="Output per-SRG summary table.",
    )

    args = parser.parse_args()

    srg_path = Path(args.srg)
    sag_path = Path(args.sag)

    if not srg_path.exists():
        raise SystemExit(f"SRG file not found: {srg_path}")

    if not sag_path.exists():
        raise SystemExit(f"SAG file not found: {sag_path}")

    if args.k < 1:
        raise SystemExit("--k must be >= 1")

    # ------------------------------------------------------------------
    # READ DATA
    # ------------------------------------------------------------------

    print("Reading SRGs:", srg_path)
    srg = pd.read_csv(srg_path, low_memory=False)

    print("Reading SAGs:", sag_path)
    sag = pd.read_csv(sag_path, low_memory=False)

    check_columns(srg, "SRG table", MATCH_QUANTITIES)
    check_columns(sag, "SAG table", MATCH_QUANTITIES)

    print("Initial SRGs:", len(srg))
    print("Initial SAGs:", len(sag))

    # Make sure HaloCatalogueIndex is numeric.
    srg["HaloCatalogueIndex"] = pd.to_numeric(
        srg["HaloCatalogueIndex"], errors="coerce"
    )
    sag["HaloCatalogueIndex"] = pd.to_numeric(
        sag["HaloCatalogueIndex"], errors="coerce"
    )

    srg = srg[srg["HaloCatalogueIndex"].notna()].copy()
    sag = sag[sag["HaloCatalogueIndex"].notna()].copy()

    srg["HaloCatalogueIndex"] = srg["HaloCatalogueIndex"].astype(np.int64)
    sag["HaloCatalogueIndex"] = sag["HaloCatalogueIndex"].astype(np.int64)

    # ------------------------------------------------------------------
    # ONLY KEEP OBJECTS WITH FINITE VALUES IN ALL MATCHING VARIABLES
    # ------------------------------------------------------------------

    srg_finite = np.ones(len(srg), dtype=bool)
    sag_finite = np.ones(len(sag), dtype=bool)

    for q in MATCH_QUANTITIES:
        srg_finite &= np.isfinite(
            pd.to_numeric(srg[q], errors="coerce").to_numpy(dtype=float)
        )
        sag_finite &= np.isfinite(
            pd.to_numeric(sag[q], errors="coerce").to_numpy(dtype=float)
        )

    srg_match = srg.loc[srg_finite].copy().reset_index(drop=True)
    sag_match = sag.loc[sag_finite].copy().reset_index(drop=True)

    print(
        f"SRGs with complete matching properties: "
        f"{len(srg_match)}/{len(srg)}"
    )
    print(
        f"SAGs with complete matching properties: "
        f"{len(sag_match)}/{len(sag)}"
    )

    if len(sag_match) < args.k:
        raise RuntimeError(
            f"Only {len(sag_match)} usable SAGs available, "
            f"but k={args.k} was requested."
        )

    # ------------------------------------------------------------------
    # ROBUST STANDARDISATION
    # ------------------------------------------------------------------

    medians, scales = robust_scale_fit(
        srg_match,
        sag_match,
        MATCH_QUANTITIES,
    )

    print("\nMatching quantities:")
    for q in MATCH_QUANTITIES:
        print(
            f"  {q:15s}: median={medians[q]:.5g}, "
            f"IQR={scales[q]:.5g}"
        )

    X_srg = standardise(
        srg_match,
        MATCH_QUANTITIES,
        medians,
        scales,
    )

    X_sag = standardise(
        sag_match,
        MATCH_QUANTITIES,
        medians,
        scales,
    )

    # ------------------------------------------------------------------
    # BUILD KD-TREE AND FIND k NEAREST SAGs FOR EVERY SRG
    # ------------------------------------------------------------------

    tree = cKDTree(X_sag)

    distances, indices = tree.query(
        X_srg,
        k=args.k,
    )

    # scipy returns 1D arrays when k=1, so enforce 2D here.
    if args.k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # ------------------------------------------------------------------
    # CREATE DETAILED MATCH TABLE
    # ------------------------------------------------------------------

    match_rows = []

    for i in range(len(srg_match)):

        srg_row = srg_match.iloc[i]

        for rank in range(args.k):

            distance = float(distances[i, rank])
            sag_idx = int(indices[i, rank])

            # Discard poor matches.
            if distance > args.max_distance:
                continue

            sag_row = sag_match.iloc[sag_idx]

            row = {
                "SRG_HaloCatalogueIndex":
                    int(srg_row["HaloCatalogueIndex"]),

                "SAG_HaloCatalogueIndex":
                    int(sag_row["HaloCatalogueIndex"]),

                "match_rank": rank + 1,
                "matching_distance": distance,
            }

            # ----------------------------------------------------------
            # Stellar-population quantities used for matching
            # ----------------------------------------------------------

            for q in MATCH_QUANTITIES:
                srg_value = float(srg_row[q])
                sag_value = float(sag_row[q])

                row[f"SRG_{q}"] = srg_value
                row[f"SAG_{q}"] = sag_value
                row[f"delta_{q}"] = sag_value - srg_value

            # ----------------------------------------------------------
            # Quantities deliberately NOT used for matching
            # ----------------------------------------------------------

            for q in [
                "compactness",
                "DoR",
                "logR",
                "sigma",
                "log_sigma",
                "exsitu_frac",
                "log_bh_ratio_corr",
            ]:

                if q in srg_row.index:
                    row[f"SRG_{q}"] = srg_row[q]

                if q in sag_row.index:
                    row[f"SAG_{q}"] = sag_row[q]

            match_rows.append(row)

    matches = pd.DataFrame(match_rows)

    if len(matches) == 0:
        raise RuntimeError(
            "No matches survived the maximum-distance cut. "
            "Try increasing --max-distance."
        )

    # Sort naturally by SRG and match rank.
    matches = matches.sort_values(
        ["SRG_HaloCatalogueIndex", "match_rank"]
    ).reset_index(drop=True)

    matches.to_csv(args.output, index=False)

    # ------------------------------------------------------------------
    # PER-SRG SUMMARY
    # ------------------------------------------------------------------

    summary_rows = []

    for srg_id, group in matches.groupby("SRG_HaloCatalogueIndex"):

        best = group.iloc[0]

        summary_rows.append({
            "SRG_HaloCatalogueIndex": int(srg_id),

            "n_matches": len(group),

            "best_matching_distance":
                float(best["matching_distance"]),

            "median_matching_distance":
                float(group["matching_distance"].median()),

            "best_SAG_HaloCatalogueIndex":
                int(best["SAG_HaloCatalogueIndex"]),

            "SRG_compactness":
                best.get("SRG_compactness", np.nan),

            "SRG_DoR":
                best.get("SRG_DoR", np.nan),

            "best_SAG_compactness":
                best.get("SAG_compactness", np.nan),

            "best_SAG_DoR":
                best.get("SAG_DoR", np.nan),

            "delta_compactness":
                (
                    best.get("SAG_compactness", np.nan)
                    - best.get("SRG_compactness", np.nan)
                ),
        })

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        "best_matching_distance"
    ).reset_index(drop=True)

    summary.to_csv(args.summary_output, index=False)

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    print("\n==================================================")
    print("MATCHING SUMMARY")
    print("==================================================")

    print("Usable SRGs:", len(srg_match))
    print("Usable SAGs:", len(sag_match))
    print("SRGs with >=1 accepted match:", len(summary))
    print("Total accepted pairs:", len(matches))

    print(
        "Best-match distance:",
        np.nanmin(summary["best_matching_distance"])
    )

    print(
        "Median best-match distance:",
        np.nanmedian(summary["best_matching_distance"])
    )

    print(
        "90th percentile best-match distance:",
        np.nanpercentile(summary["best_matching_distance"], 90)
    )

    print("\nOutputs:")
    print("  ", args.output)
    print("  ", args.summary_output)


if __name__ == "__main__":
    main()