#!/usr/bin/env python3
"""COLIBRE Paper 2: central configuration (no data loading or side effects).

All paths are interpreted on the machine running the analysis (Hyades).
This file defines settings only; main.py and modules will consume them later.
"""
from pathlib import Path

# -----------------------------------------------------------------------------
# Simulation and catalogue locations
# -----------------------------------------------------------------------------
SIMULATION = {
    "name": "L0200N3008",
    "model": "THERMAL_AGN",
    "root": Path("/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN"),
    "reference_snapshot": 127,
    "reference_redshift": 0.0,
}

INPUTS = {
    "soap_dir": SIMULATION["root"] / "SOAP-HBT",
    "soap_filename_pattern": "halo_properties_{snapshot:04d}.hdf5",
    "ascii_dir": Path(
        "/mnt/su3-pro/clagos/COLIBRE/Runs/L200_m6/Thermal/ProcessedData"
    ),
    "ascii_glob": "GalaxyProperties_SFR_GE_0_z*.txt",
    "dor_sfh_csv": Path("../sfh_times_all_with_DoR_variants_corrected.csv.gz"),
    "bh_correction_csv": Path("corrected_bh_mass_lookup.csv"),
    "exsitu_z0_hdf5": Path(
        "/mnt/su3ctm/kproctor/ForMax/exsitu_summary_SnapNum_127.hdf5"
    ),
    # Optional: add only after the actual table is located and verified.
    "snapshot_redshift_table": None,
}

# -----------------------------------------------------------------------------
# Scientific reference-sample definitions
# -----------------------------------------------------------------------------
REFERENCE_SAMPLE = {
    "populations": ("SRG",),
    "allowed_populations": ("SRG", "SAG_noncompact", "SAG_all", "compact_nonrelic"),
    "dor_column": "DoR_t95",
    "dor_threshold": 0.6,
    "dor_comparison": ">",  # Paper 1 operational convention
    "compactness_threshold": 9.75,
    "compactness_exponent": 1.5,
    "srg_compactness_comparison": ">=",
    "noncompact_sag_compactness_comparison": "<",
    # Paper 1 applies these before joining to DoR catalogue.
    "minimum_stellar_mass_msun": 1.0e9,
    "require_positive_radius": True,
    # z=0 reference classifications must be based on SOAP + DoR/SFH.
    "reference_source": "SOAP-HBT",
}

# -----------------------------------------------------------------------------
# Catalogue source policy
# -----------------------------------------------------------------------------
DATA_SOURCES = {
    "reference": "SOAP-HBT",
    # For the first tracking implementation use available ASCII epochs;
    # do not silently mix unvalidated measurements from SOAP and ASCII.
    "evolution": "ASCII",
    "soap_is_authoritative": True,
    "allow_automatic_fallback": False,
    "record_provenance": True,
    "require_overlap_validation_before_mixing": True,
}

# -----------------------------------------------------------------------------
# TrackID tracking and requested physical quantities
# -----------------------------------------------------------------------------
TRACKING = {
    "method": "HBT_TrackID",
    "redshift_min": 0.0,
    "redshift_max": 10.0,
    "snapshot_selection": "available_ascii",  # later: explicit / available_soap
    "selected_redshifts": None,
    "retain_incomplete_tracks": True,
    "duplicate_trackid_policy": "raise",  # never silently choose a duplicate
    "missing_snapshot_policy": "report",  # missing track != zero-mass galaxy
}

PROPERTIES = {
    "requested": (
        "stellar_mass", "rhalf_stars", "sfr_instant", "is_central",
        "halo_catalogue_index", "host_fof_halo_id", "bh_mass",
    ),
    "derived": (
        "ssfr_instant", "log10_stellar_mass", "log10_rhalf",
        "compactness_sigma_1p5",
    ),
    "preserve_raw_values": True,
    "preserve_zero_sfr": True,
    "ssfr_plot_floor_log10_per_year": -12.0,
}

# Unit conventions MUST be established from source metadata / exporter.
# None means unknown: property ingestion must refuse scientific conversions
# until validated, rather than silently assuming a factor.
# UNITS = {
#     "target_stellar_mass": "Msun",
#     "target_radius": "physical_kpc",
#     "target_sfr": "Msun/yr",

#     "ascii_stellar_mass_to_msun": 1.0,
#     "ascii_radius_to_physical_kpc": 1000.0,
#     "ascii_sfr_to_msun_per_year": 1.0,

#     "soap_conversion_method": "legacy_paper1",
# }
UNITS = {
    "target_stellar_mass": "Msun",
    "target_radius": "physical_kpc",
    "target_sfr": "Msun/yr",

    "soap_mass_to_msun": 1.988e43 / 1.989e33,
    "soap_radius_to_comoving_kpc": 1000.0,
    "soap_sfr_to_msun_per_year": (
        (1.988e43 / 1.989e33)
        / (3.086e19 / 3.154e7)
    ),

    "ascii_stellar_mass_to_msun": 1.0,
    "ascii_radius_to_physical_kpc": 1000.0,
    "ascii_sfr_to_msun_per_year": 1.0,
}

FORMATION = {

    # --------------------------------------------------
    # Compactness criterion
    # --------------------------------------------------

    "compactness_threshold": 9.75,

    "compactness_comparison": ">=",

    # --------------------------------------------------
    # Quenching criterion
    # --------------------------------------------------

    "quenching_method": "age_of_universe",

    "quenching_factor": 0.2,

    "quenching_comparison": "<",

    # --------------------------------------------------
    # Optional stellar mass criterion
    # --------------------------------------------------

    "apply_mass_cut": False,

    "minimum_stellar_mass_msun": 1.0e10,

    # --------------------------------------------------
    # Matching epoch
    # --------------------------------------------------

    "matching_epoch_definition": "first_compact_and_quenched",

    # --------------------------------------------------
    # Cosmology
    # --------------------------------------------------

    "cosmology": "Planck13",
}

QUENCHING_HISTORY = {
    "minimum_consecutive_star_forming_snapshots": 1,
    "minimum_consecutive_quenched_snapshots": 1,
    "require_z0_quenched": True,
    "require_compact_at_final_quenching": False,
    "plot_max_tracks": 12,
}

MATCHING = {
    "n_controls_per_srg": 5,
    "mass_tolerance_dex": 0.15,
    "radius_tolerance_dex": 0.15,
    "with_replacement": True,
    "require_compact": True,
    "require_quenched": True,
    "apply_mass_cut": False,
    "minimum_matching_redshift": 0.5,
    "exclude_reference_srgs": True,
    "random_seed": 42,
}

# -----------------------------------------------------------------------------
# Pipeline stage switches: only implemented stages should be enabled.
# -----------------------------------------------------------------------------
PIPELINE = {
    "build_reference_sample": True,
    "build_track_identities": False,
    "extract_evolution_properties": False,
    "make_diagnostic_plots": False,
    "analyse_formation": False,
    "identify_red_nuggets": False,
    "construct_matched_controls": False,
}

OUTPUT = {
    "root": Path("outputs"),
    "reference_dir": Path("outputs/reference_catalogues"),
    "tracks_dir": Path("outputs/tracks"),
    "diagnostics_dir": Path("outputs/diagnostics"),
    "figures_dir": Path("outputs/figures"),
    "logs_dir": Path("logs"),
    "table_format": "csv",  # portable; parquet can be enabled later
    "overwrite": False,
    "save_configuration": True,
}

DEVELOPMENT = {
    "test_mode": False,
    "test_sample_size": 20,
    "random_seed": 42,
    "strict_input_validation": True,
}


def validate_config():
    """Check internal consistency; do not touch the filesystem."""
    valid = set(REFERENCE_SAMPLE["allowed_populations"])
    chosen = set(REFERENCE_SAMPLE["populations"])
    if not chosen or not chosen <= valid:
        raise ValueError(f"Invalid populations: {sorted(chosen - valid)}")
    if REFERENCE_SAMPLE["dor_comparison"] != ">":
        raise ValueError("Paper 1 DoR comparison is currently fixed at '>'")
    if TRACKING["redshift_min"] > TRACKING["redshift_max"]:
        raise ValueError("Tracking redshift_min must not exceed redshift_max")
    if DEVELOPMENT["test_sample_size"] < 1:
        raise ValueError("test_sample_size must be positive")
    if DATA_SOURCES["reference"] != "SOAP-HBT":
        raise ValueError("Reference classification must preserve the SOAP-HBT policy")
    return True


if __name__ == "__main__":
    validate_config()
    print("COLIBRE Paper 2 configuration is internally consistent.")
    print("This check does not validate files, units, or scientific results.")
