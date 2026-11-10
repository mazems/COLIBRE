#!/usr/bin/env python3
"""COLIBRE Paper 2 pipeline entry point (initial reference-sample stage).

From the project root:
    python3 main.py --status
    python3 main.py --dry-run
    python3 main.py
    python3 main.py --rebuild-reference  # deliberately replace existing catalogue

The reference catalogue is reused by default. Future stages are explicitly
blocked until their modules exist; no scientific calculations are placeholders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config
from modules.sample_selection import (
    build_reference_catalogue,
    save_reference_catalogue,
    select_populations,
)

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("colibre.paper2")
CATALOGUE_NAME = "reference_galaxies_z0_all.csv"
REPORT_NAME = "reference_galaxies_z0_report.json"
IMPLEMENTED_STAGES = {"build_reference_sample"}


def project_path(value: Path | str) -> Path:
    """Resolve relative project outputs independent of current working directory."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def check_stage_switches() -> None:
    unsupported = [key for key, enabled in config.PIPELINE.items()
                   if enabled and key not in IMPLEMENTED_STAGES]
    if unsupported:
        raise RuntimeError(
            "These enabled pipeline stages are not implemented yet: "
            + ", ".join(unsupported)
            + ". Disable them in config.py until their modules are available."
        )


def reference_paths() -> tuple[Path, Path]:
    directory = project_path(config.OUTPUT["reference_dir"])
    return directory / CATALOGUE_NAME, directory / REPORT_NAME


def read_saved_reference() -> tuple[pd.DataFrame, dict]:
    csv_path, report_path = reference_paths()
    if not csv_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            "Expected both saved reference files:\n"
            f"  {csv_path}\n  {report_path}\n"
            "Run 'python3 -m modules.sample_selection --write' from the "
            "project directory, or use 'python3 main.py --rebuild-reference'."
        )
    # Read TrackIDs as exact integer values, never as floats.
    frame = pd.read_csv(csv_path, dtype={"track_id": "int64", "halo_catalogue_index": "int64"})
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {"track_id", "halo_catalogue_index", "population", "compactness_z0"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Saved reference catalogue missing columns: {sorted(missing)}")
    if frame["track_id"].isna().any() or frame["track_id"].duplicated().any():
        raise ValueError("Saved reference catalogue has missing or duplicate TrackIDs")
    if len(frame) != int(report["soap_eligible"]):
        raise ValueError("Saved reference row count disagrees with its report")
    actual = {str(k): int(v) for k, v in frame["population"].value_counts().items()}
    if actual != report["population_counts"]:
        raise ValueError("Saved reference population counts disagree with its report")
    if int(report["snapshot"]) != int(config.SIMULATION["reference_snapshot"]):
        raise ValueError("Saved reference snapshot differs from config.py")
    if report["dor_column"] != config.REFERENCE_SAMPLE["dor_column"]:
        raise ValueError("Saved reference DoR column differs from config.py")
    return frame, report


def configuration_record() -> dict:
    """Snapshot configuration; path objects and tuples are JSON serializable."""
    keys = ("SIMULATION", "INPUTS", "REFERENCE_SAMPLE", "DATA_SOURCES", "TRACKING",
            "PROPERTIES", "UNITS", "PIPELINE", "OUTPUT", "DEVELOPMENT")
    return {key: getattr(config, key) for key in keys}


def json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, default=str).encode("utf-8") + b"\n"


def save_run_record(selected: pd.DataFrame, report: dict, source: str) -> Path:
    """Write an audit record, without overwriting any prior run."""
    directory = project_path(config.OUTPUT["diagnostics_dir"]) / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    settings = configuration_record()
    settings_hash = hashlib.sha256(json_bytes(settings)).hexdigest()[:12]
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_source": source,
        "reference_catalogue": str(reference_paths()[0]),
        "reference_report": str(reference_paths()[1]),
        "reference_population_counts": report["population_counts"],
        "selected_populations": list(config.REFERENCE_SAMPLE["populations"]),
        "selected_count": int(len(selected)),
        "configuration_sha256_prefix": settings_hash,
        "configuration": settings,
        "note": "Reference units follow legacy Paper 1 conventions; physical units not independently verified.",
    }
    path = directory / f"reference_run_{stamp}_{settings_hash}.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json_bytes(record).decode("utf-8"))
    return path


def run(*, dry_run: bool, rebuild_reference: bool) -> None:
    config.validate_config()
    check_stage_switches()
    csv_path, report_path = reference_paths()

    if rebuild_reference:
        LOGGER.info("Explicit rebuild requested; reading SOAP and DoR catalogues")
        # Build first, then write: failed reads do not damage the saved catalogue.
        catalogue, report = build_reference_catalogue()
        selected = select_populations(catalogue)
        if dry_run:
            LOGGER.info("DRY RUN: would replace saved reference catalogue")
        else:
            save_reference_catalogue(catalogue, report, directory=csv_path.parent, overwrite=True)
        source = "rebuilt_from_soap_and_dor"
    elif csv_path.is_file() and report_path.is_file():
        LOGGER.info("Reusing existing reference catalogue: %s", csv_path)
        catalogue, report = read_saved_reference()
        selected = select_populations(catalogue)
        source = "saved_reference"
    else:
        raise FileNotFoundError(
            "Reference catalogue is absent or incomplete. "
            "Build it first with 'python3 -m modules.sample_selection --write' "
            "or explicitly run 'python3 main.py --rebuild-reference'."
        )

    LOGGER.info("Reference galaxies: %d", len(catalogue))
    LOGGER.info("Selected %s: %d galaxies", config.REFERENCE_SAMPLE["populations"], len(selected))
    LOGGER.info("Reference counts: %s", report["population_counts"])
    if selected.empty:
        raise ValueError("Selected sample is empty; check REFERENCE_SAMPLE['populations']")
    if dry_run:
        LOGGER.info("DRY RUN: no outputs written")
        return
    path = save_run_record(selected, report, source)
    LOGGER.info("Saved run record: %s", path)
    LOGGER.info("Reference stage complete; tracking is not implemented yet")


def main() -> int:
    parser = argparse.ArgumentParser(description="COLIBRE Paper 2 modular pipeline")
    parser.add_argument("--status", action="store_true", help="Show implemented and configured stages")
    parser.add_argument("--dry-run", action="store_true", help="Validate and select without writing outputs")
    parser.add_argument("--rebuild-reference", action="store_true",
                        help="Explicitly rebuild and overwrite the saved z=0 reference catalogue")
    args = parser.parse_args()
    configure_logging()
    try:
        if args.status:
            config.validate_config()
            for stage, enabled in config.PIPELINE.items():
                state = "enabled" if enabled else "disabled"
                implemented = "implemented" if stage in IMPLEMENTED_STAGES else "NOT IMPLEMENTED"
                LOGGER.info("%-30s %-9s %s", stage, state, implemented)
            csv_path, report_path = reference_paths()
            LOGGER.info("Saved reference catalogue exists: %s", csv_path.is_file())
            LOGGER.info("Saved reference report exists: %s", report_path.is_file())
            return 0
        run(dry_run=args.dry_run, rebuild_reference=args.rebuild_reference)
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, KeyError) as exc:
        LOGGER.error("Pipeline stopped: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
