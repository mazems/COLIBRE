#!/usr/bin/env python3
"""COLIBRE Paper 2: Track matched analogue galaxies forward to z=0.

Stages
------
1. Test persistence of each unique control HBT TrackID at z=0.
2. Attach the authoritative z=0 reference classification/properties where available.
3. Extract observed control histories from their individual matching epochs to z=0.

This is *same-TrackID persistence*, NOT merger-tree descendant reconstruction.
A missing TrackID or missing reference row is NOT proof of a merger.

Run from COLIBRE_Paper2:
    python3 -m modules.descendants --dry-run --limit 20
    python3 -m modules.descendants --dry-run
    python3 -m modules.descendants --write
    python3 -m modules.descendants --write --overwrite

--limit limits reference SRGs (complete matched sets), not control rows.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import config
except ImportError as exc:
    raise SystemExit("Run from COLIBRE_Paper2: python3 -m modules.descendants ...") from exc

COLUMNS_32 = [
    "halo_catalogue_index", "is_central", "x_galaxy", "y_galaxy", "z_galaxy",
    "stellar_mass", "sfr_instant", "rhalf_stars", "h1_mass", "h2_mass",
    "kappa_co_stars", "kappa_co_gas", "disc_to_total_stars",
    "stellar_angular_momentum", "stellar_age", "gas_metallicity_lower",
    "gas_metallicity_upper", "dust_mass", "descendant_id", "track_id",
    "vel_x", "vel_y", "vel_z", "host_halo_x", "host_halo_y", "host_halo_z",
    "halo_mass_m200crit", "bh_mass", "bh_accretion_rate",
    "bh_thermal_energy_cum", "n_agn_events", "host_fof_halo_id",
]
COLUMNS_33 = COLUMNS_32[:27] + ["sfr_100myr"] + COLUMNS_32[27:]
REQUIRED_PAIR = {"reference_track_id", "control_track_id", "z_match"}
CORE = ["stellar_mass", "rhalf_stars", "sfr_instant", "bh_mass", "is_central",
        "halo_catalogue_index", "descendant_id", "halo_mass_m200crit",
        "kappa_co_stars", "disc_to_total_stars", "stellar_age", "bh_accretion_rate"]
Z_PATTERN = re.compile(r"_z([0-9]+(?:\.[0-9]+)?)\.txt$")


def jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def checked_ids(series: pd.Series, label: str) -> pd.Series:
    """Accept integer IDs printed in ASCII scientific notation, checking exactness."""
    vals = pd.to_numeric(series, errors="coerce")
    arr = vals.to_numpy(dtype=float)
    if not np.isfinite(arr).all() or (np.abs(arr) >= 2**53).any():
        raise ValueError(f"{label}: missing/unsafe identifier; inspect input")
    if not np.all(arr == np.rint(arr)):
        raise ValueError(f"{label}: nonintegral identifier")
    return pd.Series(arr.astype(np.int64), index=series.index)


def discover_ascii() -> list[tuple[float, Path]]:
    base = Path(config.INPUTS["ascii_dir"])
    pattern = config.INPUTS.get("ascii_glob", "GalaxyProperties_SFR_GE_0_z*.txt")
    found = []
    for path in base.glob(pattern):
        match = Z_PATTERN.search(path.name)
        if match:
            found.append((float(match.group(1)), path))
    found.sort(reverse=True)
    zs = [z for z, _ in found]
    if not found or len(zs) != len(set(zs)) or not any(np.isclose(z, 0) for z in zs):
        raise RuntimeError("Missing ASCII catalogues, duplicate redshifts, or missing z=0")
    return found


def get_schema(path: Path) -> list[str]:
    with path.open("r") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                ncols = len(line.split())
                break
        else:
            raise ValueError(f"Empty catalogue: {path}")
    if ncols == 32:
        return COLUMNS_32
    if ncols == 33:
        return COLUMNS_33
    raise ValueError(f"Unexpected {ncols} columns in {path}")


def read_selected(path: Path, wanted: set[int], chunksize: int) -> pd.DataFrame:
    """Chunked, memory-bounded read; only matched rows and requested fields retained."""
    schema = get_schema(path)
    wanted_cols = ["track_id"] + [c for c in CORE if c != "track_id" and c in schema]
    if "sfr_100myr" in schema:
        wanted_cols.append("sfr_100myr")
    indices = [schema.index(c) for c in wanted_cols]
    pieces = []
    for chunk in pd.read_csv(path, sep=r"\s+", header=None, comment="#",
                             usecols=indices, chunksize=chunksize):
        # Pandas returns columns in source order, not usecols order.
        chunk.columns = [schema[i] for i in sorted(indices)]
        ids = checked_ids(chunk["track_id"], f"{path.name} TrackID")
        mask = ids.isin(wanted)
        if mask.any():
            part = chunk.loc[mask].copy()
            part["track_id"] = ids.loc[mask].to_numpy()
            pieces.append(part)
    if not pieces:
        return pd.DataFrame(columns=wanted_cols)
    result = pd.concat(pieces, ignore_index=True)
    if result["track_id"].duplicated().any():
        raise ValueError(f"Duplicate TrackIDs in {path}; cannot map uniquely")
    return result


def factors() -> dict[str, float | None]:
    u = getattr(config, "UNITS", {})
    keys = {
        "stellar_mass": "ascii_stellar_mass_to_msun",
        "rhalf_stars": "ascii_radius_to_physical_kpc",
        "sfr_instant": "ascii_sfr_to_msun_per_year",
    }
    return {k: (None if u.get(v) is None else float(u[v])) for k, v in keys.items()}


def derive_physical(frame: pd.DataFrame, z: float, unit_factors: dict) -> pd.DataFrame:
    out = frame.copy()
    for src, dest in [("stellar_mass", "stellar_mass_msun"),
                      ("rhalf_stars", "rhalf_physical_kpc"),
                      ("sfr_instant", "sfr_msun_per_year")]:
        factor = unit_factors[src]
        out[dest] = (pd.to_numeric(out[src], errors="coerce") * factor
                     if factor is not None else np.nan)
    m = out["stellar_mass_msun"].to_numpy(dtype=float)
    r = out["rhalf_physical_kpc"].to_numpy(dtype=float)
    sfr = out["sfr_msun_per_year"].to_numpy(dtype=float)
    out["log10_stellar_mass"] = np.where(m > 0, np.log10(np.where(m > 0, m, np.nan)), np.nan)
    out["log10_rhalf"] = np.where(r > 0, np.log10(np.where(r > 0, r, np.nan)), np.nan)
    out["compactness_sigma_1p5"] = out["log10_stellar_mass"] - 1.5 * out["log10_rhalf"]
    out["ssfr_per_year"] = np.where((m > 0) & (sfr >= 0), sfr / m, np.nan)
    out["redshift"] = z
    return out


def load_inputs(root: Path, limit: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_path = root / "outputs/controls/matched_control_candidates.csv"
    ref_path = root / "outputs/reference_catalogues/reference_galaxies_z0_all.csv"
    pairs = pd.read_csv(pair_path)
    reference = pd.read_csv(ref_path)
    missing = REQUIRED_PAIR - set(pairs.columns)
    if missing:
        raise ValueError(f"Missing pair columns: {sorted(missing)}")
    if "track_id" not in reference or "population" not in reference:
        raise ValueError("Reference requires track_id and population")
    for col in ["reference_track_id", "control_track_id"]:
        pairs[col] = checked_ids(pairs[col], col)
    reference["track_id"] = checked_ids(reference["track_id"], "reference track_id")
    pairs["z_match"] = pd.to_numeric(pairs["z_match"], errors="raise")
    if not np.isfinite(pairs["z_match"]).all():
        raise ValueError("Invalid matching redshift")
    if reference["track_id"].duplicated().any():
        raise ValueError("Duplicate reference TrackIDs")
    if limit is not None:
        selected = pairs["reference_track_id"].drop_duplicates().head(limit)
        pairs = pairs.loc[pairs["reference_track_id"].isin(selected)].copy()
    if pairs.empty:
        raise ValueError("No matched pairs")
    return pairs, reference


def analyse(root: Path, limit: int | None, chunksize: int):
    pairs, reference = load_inputs(root, limit)
    control_ids = set(pairs["control_track_id"].astype(int))
    srg_ids = set(reference.loc[reference["population"] == "SRG", "track_id"].astype(int))
    overlap = control_ids & srg_ids
    if overlap:
        raise ValueError(f"Controls overlap reference SRGs: {sorted(overlap)[:10]}")

    inventory = discover_ascii()
    available_z = np.array([z for z, _ in inventory])
    for z in pairs["z_match"].unique():
        if not np.any(np.isclose(available_z, z, atol=1e-6, rtol=0)):
            raise ValueError(f"Matching epoch z={z} missing from ASCII inventory")
    first_match = pairs.groupby("control_track_id")["z_match"].max().to_dict()
    # A reused control can be matched at multiple epochs: retain history from earliest
    # cosmic matching time (highest redshift), but pair-level filtering is separate.
    unit_factors = factors()
    if any(v is None for v in unit_factors.values()):
        raise ValueError("Missing ASCII mass/radius/SFR factors in config.UNITS")

    histories = []
    present = None
    for z, path in inventory:
        active = {tid for tid, first_z in first_match.items() if z <= first_z + 1e-6}
        if not active:
            continue
        snap = derive_physical(read_selected(path, active, chunksize), z, unit_factors)
        if not snap.empty:
            snap["snapshot_file"] = path.name
            histories.append(snap)
        print(f"z={z:5.2f}: {len(snap):5d}/{len(active):5d} control TrackIDs present")
        if np.isclose(z, 0.0):
            present = snap.copy()
    if present is None:
        raise RuntimeError("z=0 catalogue not processed")
    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    history = history.sort_values(["track_id", "redshift"], ascending=[True, False]).reset_index(drop=True)

    unique_controls = pd.DataFrame({"control_track_id": sorted(control_ids)})
    unique_controls["first_matching_redshift"] = unique_controls["control_track_id"].map(first_match)
    z0_raw = present.rename(columns={"track_id": "control_track_id"})
    z0_raw = z0_raw.drop(columns=["redshift", "snapshot_file"], errors="ignore")
    z0_raw = z0_raw.rename(columns={c: f"ascii_z0_{c}" for c in z0_raw if c != "control_track_id"})
    status = unique_controls.merge(z0_raw, on="control_track_id", how="left", validate="one_to_one", indicator="_ascii_join")
    status["trackid_present_z0_ascii"] = status.pop("_ascii_join").eq("both")

    # SOAP-based reference catalogue is authoritative for z=0 classifications.
    z0_ref = reference.rename(columns={"track_id": "control_track_id"})
    z0_ref = z0_ref.rename(columns={c: f"reference_z0_{c}" for c in z0_ref if c != "control_track_id"})
    status = status.merge(z0_ref, on="control_track_id", how="left", validate="one_to_one", indicator="_reference_join")
    status["in_reference_z0"] = status.pop("_reference_join").eq("both")
    status["population_z0"] = status.get("reference_z0_population", pd.Series(index=status.index, dtype=object))
    status["descendant_status"] = np.select(
        [status["trackid_present_z0_ascii"] & status["in_reference_z0"],
         status["trackid_present_z0_ascii"] & ~status["in_reference_z0"],
         ~status["trackid_present_z0_ascii"] & status["in_reference_z0"]],
        ["same_trackid_present_and_classified", "same_trackid_present_reference_missing",
         "reference_present_ascii_missing"],
        default="same_trackid_absent_z0_unresolved")
    # Do NOT infer that a different TrackID is a descendant.
    status["merger_tree_descendant_verified"] = False
    status["descendant_track_id_verified"] = pd.NA

    # Preserve every original matched set and its own epoch; same control can
    # contribute to different sets with different time-zero origins.
    pair_status = pairs.merge(status[["control_track_id", "trackid_present_z0_ascii",
                                      "in_reference_z0", "population_z0", "descendant_status"]],
                              on="control_track_id", how="left", validate="many_to_one")
    pair_history = pairs[["reference_track_id", "control_track_id", "z_match"]].merge(
        history, left_on="control_track_id", right_on="track_id", how="left", validate="many_to_many")
    pair_history = pair_history.loc[pair_history["redshift"] <= pair_history["z_match"] + 1e-6].copy()
    pair_history = pair_history.drop(columns=["track_id"])
    # Optional time since matching, using same Planck13 convention as formation.py.
    try:
        from astropy.cosmology import Planck13
        pair_history["time_since_match_gyr"] = (
            Planck13.age(pair_history["redshift"].to_numpy(dtype=float)).value
            - Planck13.age(pair_history["z_match"].to_numpy(dtype=float)).value
        )
    except ImportError:
        pair_history["time_since_match_gyr"] = np.nan

    # One-row-per-control output and pair-level output have different purposes.
    report = {
        "status": "same_trackid_persistence_only_not_merger_aware",
        "n_pairs": len(pairs),
        "n_reference_srgs_in_pairs": pairs["reference_track_id"].nunique(),
        "n_unique_controls": len(control_ids),
        "n_controls_present_z0_ascii": int(status["trackid_present_z0_ascii"].sum()),
        "n_controls_in_z0_reference": int(status["in_reference_z0"].sum()),
        "n_controls_present_and_classified": int((status["trackid_present_z0_ascii"] & status["in_reference_z0"]).sum()),
        "n_controls_absent_z0_ascii": int((~status["trackid_present_z0_ascii"]).sum()),
        "n_history_rows_unique_controls": len(history),
        "n_pair_history_rows": len(pair_history),
        "n_epochs_scanned": len([1 for z, _ in inventory if any(z <= x + 1e-6 for x in first_match.values())]),
        "z0_population_counts": status["population_z0"].fillna("unclassified").value_counts().to_dict(),
        "unit_factors": unit_factors,
        "source": "ASCII snapshots; z=0 classification from SOAP-based reference catalogue",
        "caveats": [
            "Same TrackID at z=0 is persistence, not verified descendant tracing through mergers.",
            "Missing TrackID is not proof of merger or destruction.",
            "Reference catalogue non-membership is not proof of merger or non-relic outcome.",
            "DoR is attached from z=0 reference only, never interpreted as a time series.",
            "Additional raw ASCII properties retain unverified native units.",
            "Reused controls may have different z_match for different SRGs; pair histories preserve each epoch.",
            "Reference and ASCII mass/size may differ slightly, as previously validated.",
        ],
    }
    return status, pair_status, history, pair_history, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print diagnostics without saving (default)")
    mode.add_argument("--write", action="store_true", help="Save CSV and JSON outputs")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs (requires --write)")
    parser.add_argument("--limit", type=int, help="Process first N matched SRG sets")
    parser.add_argument("--chunksize", type=int, default=50000)
    args = parser.parse_args()
    if args.overwrite and not args.write:
        parser.error("--overwrite requires --write")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.chunksize <= 0:
        parser.error("--chunksize must be positive")

    root = Path(__file__).resolve().parents[1]
    outdir = root / "outputs/controls/descendants"
    names = {
        "control_descendant_status.csv": None,
        "control_descendant_pairs.csv": None,
        "control_evolutionary_properties.csv": None,
        "control_pair_evolutionary_properties.csv": None,
        "descendant_report.json": None,
    }
    if args.write and not args.overwrite:
        existing = [str(outdir / name) for name in names if (outdir / name).exists()]
        if existing:
            raise FileExistsError(f"Output(s) exist; use --write --overwrite: {existing}")

    status, pair_status, history, pair_history, report = analyse(root, args.limit, args.chunksize)
    print("\nDESCENDANT / PERSISTENCE REPORT")
    print(json.dumps(jsonable(report), indent=2))
    if not args.write:
        print("Dry run: no files written.")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    status.to_csv(outdir / "control_descendant_status.csv", index=False)
    pair_status.to_csv(outdir / "control_descendant_pairs.csv", index=False)
    history.to_csv(outdir / "control_evolutionary_properties.csv", index=False)
    pair_history.to_csv(outdir / "control_pair_evolutionary_properties.csv", index=False)
    with (outdir / "descendant_report.json").open("w") as handle:
        json.dump(jsonable(report), handle, indent=2)
    print(f"Saved 5 outputs in {outdir}")


if __name__ == "__main__":
    main()
