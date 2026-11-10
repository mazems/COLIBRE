#!/usr/bin/env python3

"""
matching_quality_check.py

Evaluate the quality of SRG-control matching.

Input:
    outputs/controls/matched_control_candidates.csv

Outputs:
    outputs/controls/matching_quality.png
    outputs/controls/matching_quality.pdf
    outputs/controls/matching_quality_by_redshift.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path(
    "outputs/controls/matched_control_candidates.csv"
)

OUTPUT_DIR = Path("outputs/controls")

MASS_TOLERANCE = 0.15
RADIUS_TOLERANCE = 0.15

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\n========================================")
print("MATCHING QUALITY DIAGNOSTICS")
print("========================================")

print("Matched pairs:", len(df))
print("Unique SRGs:", df["reference_track_id"].nunique())
print("Unique controls:", df["control_track_id"].nunique())


# ============================================================
# CALCULATE MATCHING DIFFERENCES
# ============================================================

df["delta_logM"] = (
    df["control_log10_stellar_mass"]
    - df["reference_log10_stellar_mass"]
)

df["delta_logR"] = (
    df["control_log10_rhalf"]
    - df["reference_log10_rhalf"]
)

df["matching_distance_check"] = np.sqrt(
    (df["delta_logM"] / MASS_TOLERANCE)**2
    + (df["delta_logR"] / RADIUS_TOLERANCE)**2
)


# ============================================================
# VERIFY MATCHING CONSTRAINTS
# ============================================================

mass_valid = (
    np.abs(df["delta_logM"]) <= MASS_TOLERANCE + 1e-10
)

radius_valid = (
    np.abs(df["delta_logR"]) <= RADIUS_TOLERANCE + 1e-10
)

print("\nMatching constraints:")
print("Mass tolerance satisfied:", mass_valid.sum(), "/", len(df))
print("Radius tolerance satisfied:", radius_valid.sum(), "/", len(df))

if not (mass_valid.all() and radius_valid.all()):
    print("WARNING: Some matches violate the configured tolerances!")


# ============================================================
# NUMERICAL DIAGNOSTICS
# ============================================================

for name, column in [
    ("Delta log stellar mass", "delta_logM"),
    ("Delta log radius", "delta_logR"),
    ("Matching distance", "matching_distance_check"),
]:

    values = df[column].to_numpy(dtype=float)

    print(f"\n{name}")
    print("  Median:", np.nanmedian(values))
    print(
        "  16th/84th percentiles:",
        np.nanpercentile(values, [16, 84])
    )
    print(
        "  95th percentile:",
        np.nanpercentile(values, 95)
    )
    print(
        "  Maximum absolute value:",
        np.nanmax(np.abs(values))
    )


# ============================================================
# MATCHING QUALITY BY REDSHIFT
# ============================================================

rows = []

for z, g in df.groupby("z_match"):

    rows.append({
        "redshift": z,
        "n_pairs": len(g),
        "n_reference_srgs": g["reference_track_id"].nunique(),
        "n_unique_controls": g["control_track_id"].nunique(),

        "median_delta_logM": np.nanmedian(g["delta_logM"]),
        "median_delta_logR": np.nanmedian(g["delta_logR"]),

        "median_matching_distance": np.nanmedian(
            g["matching_distance_check"]
        ),

        "p95_matching_distance": np.nanpercentile(
            g["matching_distance_check"], 95
        ),
    })

redshift_summary = pd.DataFrame(rows)

redshift_summary = redshift_summary.sort_values(
    "redshift"
)

redshift_summary.to_csv(
    OUTPUT_DIR / "matching_quality_by_redshift.csv",
    index=False
)

print("\nMatching quality by redshift:")
print(redshift_summary.to_string(index=False))


# ============================================================
# PLOTTING
# ============================================================

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 13,
})

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 5)
)

# ------------------------------------------------------------
# Panel 1: Stellar mass differences
# ------------------------------------------------------------

axes[0].hist(
    df["delta_logM"],
    bins=35,
    color="C0",
    alpha=0.75
)

axes[0].axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.5
)

axes[0].axvline(
    -MASS_TOLERANCE,
    color="red",
    linestyle=":",
    linewidth=1
)

axes[0].axvline(
    MASS_TOLERANCE,
    color="red",
    linestyle=":",
    linewidth=1
)

axes[0].set_xlabel(
    r"$\Delta\log_{10}(M_\star)$ [dex]"
)

axes[0].set_ylabel("Number of matched pairs")


# ------------------------------------------------------------
# Panel 2: Radius differences
# ------------------------------------------------------------

axes[1].hist(
    df["delta_logR"],
    bins=35,
    color="C1",
    alpha=0.75
)

axes[1].axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.5
)

axes[1].axvline(
    -RADIUS_TOLERANCE,
    color="red",
    linestyle=":",
    linewidth=1
)

axes[1].axvline(
    RADIUS_TOLERANCE,
    color="red",
    linestyle=":",
    linewidth=1
)

axes[1].set_xlabel(
    r"$\Delta\log_{10}(R_{50})$ [dex]"
)

axes[1].set_ylabel("Number of matched pairs")


# ------------------------------------------------------------
# Panel 3: Matching distance
# ------------------------------------------------------------

axes[2].hist(
    df["matching_distance_check"],
    bins=35,
    color="C2",
    alpha=0.75
)

axes[2].set_xlabel("Normalised matching distance")
axes[2].set_ylabel("Number of matched pairs")


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

for ax in axes:
    ax.grid(alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "matching_quality.png",
    dpi=250,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "matching_quality.pdf",
    dpi=250,
    bbox_inches="tight"
)

plt.close(fig)

print("\nSaved matching-quality diagnostics.")
print("Done.")