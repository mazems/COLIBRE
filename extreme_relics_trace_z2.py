#!/usr/bin/env python3
"""
extreme_relics_trace_z2.py

Select extreme z=0 relic galaxies from the DoR catalogue, match them to z=0
SOAP/HBT TrackIDs, follow those same TrackIDs to snapshot 0076 (z~2), and
write a compact z=0 -> z=2 comparison table.

The script also generates z=2 diagnostic plots for the SAME 490 (or however
many) z=0 extreme relics, rather than selecting a new population at z=2.

Outputs
-------
out/extreme_relics_z0_to_z2_summary.csv
out/z2_BH_ratio_extremes.png
out/z2_mass_size_extremes.png
out/z2_compactness_extremes.png
out/z2_central_fraction.txt

Optional host / velocity plots are produced only if those quantities are
available and finite.

Assumptions
-----------
- Z0_SNAP = 0127 is z=0.
- Z2_SNAP = 0076 is z~2.
- DoR catalogue IDs match the z=0 SOAP HaloCatalogueIndex.
- TrackID is HBTplus/TrackId.
- Stellar masses and BH masses are in the simulation mass unit converted by Mu.
- Half-mass radii are comoving in the raw SOAP data and are converted to
  physical kpc with 1/(1+z) * 1e3.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "L0200N3008/THERMAL_AGN/"
MODEL_DIR = "/mnt/su3-pro/colibre/" + MODEL_NAME
OUTDIR = Path("out")
OUTDIR.mkdir(parents=True, exist_ok=True)

CSV_DOR = "sfh_times_all_with_DoR_variants_corrected.csv.gz"
Z0_SNAP = "0127"
Z2_SNAP = "0076"

MIN_STELLAR_MASS = 1e9
EXTREME_DOR = 0.6
COMPACTNESS_CUT = 9.75
CHUNK = 80_000

# Same conversion convention as your existing scripts.
Mu = 1.988e43 / 1.989e33       # simulation mass unit -> Msun
tu = 3.086e19 / 3.154e7        # simulation time unit -> yr

# -----------------------------------------------------------------------------
# HDF5 helpers
# -----------------------------------------------------------------------------
def find_dataset_in_group(group: h5py.Group, candidates: Iterable[str]) -> Optional[str]:
    """Return the first matching dataset path from a list of candidate names."""
    candidates = list(candidates)

    for cand in candidates:
        if cand in group:
            return cand
        if "InputHalos" in group and cand in group["InputHalos"]:
            return "InputHalos/" + cand

    # Conservative fallback: search one level down for a name containing a
    # useful keyword. Do not make arbitrary recursive guesses.
    for key in group.keys():
        if any(token.lower() in key.lower() for token in candidates):
            return key

    return None

def find_snapshot_path(snap_label: str) -> Optional[str]:
    """Locate SOAP-HBT first, then SOAP."""
    p1 = os.path.join(MODEL_DIR, "SOAP-HBT", f"halo_properties_{snap_label}.hdf5")
    p2 = os.path.join(MODEL_DIR, "SOAP", f"halo_properties_{snap_label}.hdf5")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None

def safe_read_dataset(ds: Optional[h5py.Dataset], indices: np.ndarray) -> np.ndarray:
    """Read only selected rows from a dataset; return NaNs if unavailable."""
    n = len(indices)
    if ds is None:
        return np.full(n, np.nan)

    try:
        return np.asarray(ds[indices])
    except Exception:
        # Some HDF5 datasets dislike fancy indexing depending on ordering or
        # chunk layout. Fall back to individual reads for the few selected rows.
        out = []
        for idx in indices:
            try:
                out.append(np.asarray(ds[int(idx)]))
            except Exception:
                out.append(np.nan)
        return np.asarray(out)

def convert_raw_vel_to_kms(raw_vel: np.ndarray, z_snap: float) -> Optional[np.ndarray]:
    """Convert the simulation velocity convention used in the existing scripts."""
    if raw_vel is None:
        return None

    arr = np.asarray(raw_vel)
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape((-1, 3))
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None

    comov_to_phys = 1.0 / (1.0 + z_snap)
    return arr * comov_to_phys * 1e3 / tu

# -----------------------------------------------------------------------------
# Read DoR catalogue
# -----------------------------------------------------------------------------
if not os.path.exists(CSV_DOR):
    raise SystemExit(f"DoR CSV not found: {CSV_DOR}")

print("Loading DoR CSV:", CSV_DOR)
df_dor = pd.read_csv(CSV_DOR, low_memory=False)

id_col = None
for candidate in (
    "subhalo_id",
    "HaloCatalogueIndex",
    "subhaloId",
    "HaloIndex",
    "track_id",
    "TrackId",
):
    if candidate in df_dor.columns:
        id_col = candidate
        break

if id_col is None:
    id_col = df_dor.columns[0]
    print("Warning: no canonical ID column found; using:", id_col)

id_numeric = pd.to_numeric(
    df_dor[id_col].astype(str).str.replace("\r", "", regex=False).str.strip(),
    errors="coerce",
)
df_dor = df_dor.loc[id_numeric.notna()].copy()
df_dor["subhalo_id"] = id_numeric.loc[df_dor.index].astype(np.int64)

# Find DoR column.
dor_col = None
for candidate in ("DoR_t95", "DoR_t90", "DoR_t998", "DoR", "DoR_tfin"):
    if candidate in df_dor.columns:
        dor_col = candidate
        break
if dor_col is None:
    for col in df_dor.columns:
        if str(col).lower().startswith("dor"):
            dor_col = col
            break
if dor_col is None:
    raise SystemExit("No DoR-like column found in CSV.")

print("Using DoR column:", dor_col)

dor_lookup = (
    pd.to_numeric(df_dor[["subhalo_id", dor_col]][dor_col], errors="coerce")
)
dor_lookup = pd.Series(dor_lookup.to_numpy(), index=df_dor["subhalo_id"].to_numpy())
dor_lookup = dor_lookup.dropna().groupby(level=0).first().to_dict()
print("Loaded DoR entries:", len(dor_lookup))

# -----------------------------------------------------------------------------
# Read z=0 SOAP through common helper
# -----------------------------------------------------------------------------
print("Reading z=0 SOAP (minimal fields) via common.read_group_data_colibre...")

try:
    import common
except Exception as exc:
    raise SystemExit(
        "Couldn't import common. Run this script from your COLIBRE-analysis "
        "project where common.py is importable."
    ) from exc

fields_gal = {
    "ExclusiveSphere/50kpc": (
        "StellarMass",
        "HalfMassRadiusStars",
        "CentreOfMass",
        "MostMassiveBlackHoleMass",
        "CentreOfMassVelocity",
    )
}
fields_id = {
    "InputHalos": (
        "HaloCatalogueIndex",
        "IsCentral",
        "HBTplus/TrackId",
    )
}

h5_gal = common.read_group_data_colibre(MODEL_DIR, Z0_SNAP, fields_gal)
h5_id = common.read_group_data_colibre(MODEL_DIR, Z0_SNAP, fields_id)

m30_raw, r50_raw, centers0_raw, bh_mass0_raw, comvel0_raw = h5_gal
halo_index_all, is_central_all, track_id_all = h5_id

m30 = np.asarray(m30_raw).ravel() * Mu
r50 = np.asarray(r50_raw).ravel() * 1e3
centers0 = np.asarray(centers0_raw) * 1e3
bh_mass0 = np.asarray(bh_mass0_raw).ravel() * Mu

comvel0 = np.asarray(comvel0_raw)
if comvel0.ndim == 1 and comvel0.size % 3 == 0:
    comvel0 = comvel0.reshape((-1, 3))

halo_index_all = np.asarray(halo_index_all).ravel()
is_central_all = np.asarray(is_central_all).ravel().astype(bool)
track_id_all = np.asarray(track_id_all).ravel()

if not (
    len(m30)
    == len(r50)
    == len(centers0)
    == len(bh_mass0)
    == len(halo_index_all)
    == len(is_central_all)
    == len(track_id_all)
):
    raise RuntimeError("z=0 SOAP arrays do not have the same length.")

sel = np.where(m30 >= MIN_STELLAR_MASS)[0]
if sel.size == 0:
    raise SystemExit("No z=0 galaxies above MIN_STELLAR_MASS.")
print("Selected z=0 SOAP rows (m >= MIN_STELLAR_MASS):", sel.size)

# Match DoR to z=0 HaloCatalogueIndex.
halo_idx_selected = halo_index_all[sel].astype(np.int64)
dor_for_selected = np.array(
    [dor_lookup.get(int(h), np.nan) for h in halo_idx_selected],
    dtype=float,
)

matched_positions = np.flatnonzero(np.isfinite(dor_for_selected))
print(
    f"Matched DoR entries for selected SOAP rows: "
    f"{matched_positions.size} / {halo_idx_selected.size}"
)

# Only try +/-1 if the direct matching produced no matches, preserving your
# original fallback without silently mixing conventions.
if matched_positions.size == 0:
    for offset in (-1, +1):
        trial = np.array(
            [dor_lookup.get(int(h + offset), np.nan) for h in halo_idx_selected],
            dtype=float,
        )
        trial_matches = np.flatnonzero(np.isfinite(trial))
        if trial_matches.size:
            dor_for_selected = trial
            matched_positions = trial_matches
            print(
                f"Matched after halo index offset {offset:+d}: "
                f"{matched_positions.size}"
            )
            break

if matched_positions.size == 0:
    raise SystemExit("No z=0 SOAP rows could be matched to the DoR catalogue.")

sel_global_idx = sel[matched_positions]
z0_track_matched = track_id_all[sel_global_idx]
z0_haloidx_matched = halo_idx_selected[matched_positions]
z0_iscentral = is_central_all[sel_global_idx]
z0_mass = m30[sel_global_idx]
z0_r50 = r50[sel_global_idx]
z0_bh = bh_mass0[sel_global_idx]
z0_center = centers0[sel_global_idx]

# z=0 compactness for the matched galaxies.
# Compute it directly in log-space to minimise temporary allocations.
with np.errstate(divide="ignore", invalid="ignore"):
    logm_z0 = np.log10(z0_mass)
    logr_z0 = np.log10(z0_r50)
    compactness_z0_all = logm_z0 - 1.5 * logr_z0

# Velocity conversion is only needed for the z=0 extreme relics.
# Delay it until after the population selections to avoid creating a full second copy of the z=0 velocity catalogue at this point.

# Select the extreme z=0 relics.
mask_extreme = np.isfinite(dor_for_selected) & (dor_for_selected > EXTREME_DOR)

# z=0 compact non-relics: non-extreme objects that satisfy the same
# compactness threshold used to define SRGs.
mask_compact_nonrelic = (
    np.isfinite(dor_for_selected)
    & (dor_for_selected < EXTREME_DOR)
    & np.isfinite(compactness_z0_all)
    & (compactness_z0_all >= COMPACTNESS_CUT)
)

n_compact_nonrelic = int(mask_compact_nonrelic.sum())
print(f"Compact non-relics at z0: {n_compact_nonrelic}")
n_extreme = int(mask_extreme.sum())
print(f"Extremes at z0 (DoR>{EXTREME_DOR}): {n_extreme}")
if n_extreme == 0:
    raise SystemExit("No extreme relics matched - exiting.")

tracks_to_find = {
    int(t)
    for t in z0_track_matched[mask_extreme | mask_compact_nonrelic]
    if np.isfinite(t)
}
print(
    "Tracks to follow (extreme relics + compact non-relics):",
    len(tracks_to_find),
)
# Convert velocities only for the matched z=0 galaxies actually needed later.
comvel0_matched = comvel0[sel_global_idx]
comvel0_kms = convert_raw_vel_to_kms(comvel0_matched, 0.0)

if len(tracks_to_find) < n_extreme:
    print(
        "Warning: fewer unique TrackIDs than extreme relics: "
        f"{len(tracks_to_find)} vs {n_extreme}"
    )

# -----------------------------------------------------------------------------
# Scan z=2 SOAP/HBT by TrackID
# -----------------------------------------------------------------------------
snap_path_z2 = find_snapshot_path(Z2_SNAP)
if snap_path_z2 is None:
    raise SystemExit(
        f"Could not find z=2 snapshot file for label {Z2_SNAP} under {MODEL_DIR}"
    )

print("Scanning z=2 snapshot:", snap_path_z2)

track_candidates = [
    "HBTplus/TrackId",
    "HBT/TrackId",
    "TrackId",
    "HBTplus/track_id",
]
halo_candidates = [
    "HaloCatalogueIndex",
    "HaloIndex",
    "Halo/Index",
    "InputHalos/HaloCatalogueIndex",
]
iscen_candidates = ["IsCentral", "is_central"]

def find_galaxy_dataset(group: h5py.Group, exact_candidates: Iterable[str]) -> Optional[h5py.Dataset]:
    """Return an HDF5 dataset for an ExclusiveSphere property."""
    exact_candidates = list(exact_candidates)

    for cand in exact_candidates:
        if cand in group and isinstance(group[cand], h5py.Dataset):
            return group[cand]

    # Most commonly the top-level group contains ExclusiveSphere.
    if "ExclusiveSphere" in group:
        ex = group["ExclusiveSphere"]
        tail_names = {cand.split("/")[-1] for cand in exact_candidates}
        for tail in tail_names:
            if tail in ex and isinstance(ex[tail], h5py.Dataset):
                return ex[tail]
        if "50kpc" in ex:
            sph = ex["50kpc"]
            for tail in tail_names:
                if tail in sph and isinstance(sph[tail], h5py.Dataset):
                    return sph[tail]

    return None

found_z2_rows: list[dict] = []

with h5py.File(snap_path_z2, "r") as fh:
    ds_track_name = find_dataset_in_group(fh, track_candidates)
    ds_halo_name = find_dataset_in_group(fh, halo_candidates)
    ds_iscen_name = find_dataset_in_group(fh, iscen_candidates)

    if ds_track_name is None:
        raise SystemExit("No TrackId-like dataset found in z=2 snapshot.")
    if ds_halo_name is None:
        print("Warning: no HaloCatalogueIndex-like dataset found at z=2.")
    if ds_iscen_name is None:
        print("Warning: no IsCentral-like dataset found at z=2.")

    print("Using TrackId dataset:", ds_track_name)
    print("Using halo idx dataset:", ds_halo_name)
    print("Using IsCentral dataset:", ds_iscen_name)

    ds_mz2 = find_galaxy_dataset(
        fh,
        [
            "ExclusiveSphere/50kpc/StellarMass",
            "ExclusiveSphere/StellarMass",
            "StellarMass",
        ],
    )
    ds_r50 = find_galaxy_dataset(
        fh,
        [
            "ExclusiveSphere/50kpc/HalfMassRadiusStars",
            "ExclusiveSphere/HalfMassRadiusStars",
            "HalfMassRadiusStars",
        ],
    )
    ds_bh = find_galaxy_dataset(
        fh,
        [
            "ExclusiveSphere/50kpc/MostMassiveBlackHoleMass",
            "ExclusiveSphere/MostMassiveBlackHoleMass",
            "MostMassiveBlackHoleMass",
        ],
    )
    ds_center = find_galaxy_dataset(
        fh,
        [
            "ExclusiveSphere/50kpc/CentreOfMass",
            "ExclusiveSphere/CentreOfMass",
            "CentreOfMass",
        ],
    )
    ds_comvel = find_galaxy_dataset(
        fh,
        [
            "ExclusiveSphere/50kpc/CentreOfMassVelocity",
            "ExclusiveSphere/CentreOfMassVelocity",
            "CentreOfMassVelocity",
        ],
    )

    print("z=2 galaxy datasets:")
    print("  StellarMass:", ds_mz2.name if ds_mz2 is not None else "NOT FOUND")
    print("  HalfMassRadiusStars:", ds_r50.name if ds_r50 is not None else "NOT FOUND")
    print("  MostMassiveBlackHoleMass:", ds_bh.name if ds_bh is not None else "NOT FOUND")
    print("  CentreOfMass:", ds_center.name if ds_center is not None else "NOT FOUND")
    print("  CentreOfMassVelocity:", ds_comvel.name if ds_comvel is not None else "NOT FOUND")

    d_track = fh[ds_track_name]
    nrows = d_track.shape[0]
    print("z=2 TrackId table length:", nrows)

    tracks_remaining = set(tracks_to_find)
    abs_indices_found: list[tuple[int, int, float, object]] = []

    print("Scanning z=2 TrackIDs in chunks...")
    for start in range(0, nrows, CHUNK):
        stop = min(start + CHUNK, nrows)
        tr_chunk = np.asarray(d_track[start:stop], dtype=np.int64)

        if not tracks_remaining:
            break

        mask_in = np.isin(tr_chunk, np.fromiter(tracks_remaining, dtype=np.int64))
        if not np.any(mask_in):
            continue

        rel_idxs = np.flatnonzero(mask_in)
        abs_idxs = rel_idxs + start

        if ds_halo_name is not None:
            try:
                halo_vals = np.asarray(fh[ds_halo_name][abs_idxs])
            except Exception:
                halo_vals = np.full(len(abs_idxs), np.nan)
        else:
            halo_vals = np.full(len(abs_idxs), np.nan)

        if ds_iscen_name is not None:
            try:
                iscen_vals = np.asarray(fh[ds_iscen_name][abs_idxs])
            except Exception:
                iscen_vals = np.full(len(abs_idxs), np.nan)
        else:
            iscen_vals = np.full(len(abs_idxs), np.nan)

        for j, ai in enumerate(abs_idxs):
            trv = int(tr_chunk[rel_idxs[j]])
            halov = halo_vals[j] if j < len(halo_vals) else np.nan
            iscenv = iscen_vals[j] if j < len(iscen_vals) else np.nan

            try:
                halo_float = float(halov)
            except Exception:
                halo_float = np.nan

            abs_indices_found.append((
                int(ai),
                trv,
                halo_float,
                iscenv,
            ))
            tracks_remaining.discard(trv)

        if not tracks_remaining:
            break

    print(
        "Tracks found in z=2:", len(abs_indices_found),
        "; still missing:", len(tracks_remaining),
    )

    # Sanity check: normally one z=2 row per TrackID is expected here.
    found_track_set = {item[1] for item in abs_indices_found}
    duplicate_count = len(abs_indices_found) - len(found_track_set)
    if duplicate_count:
        print(
            "Warning:", duplicate_count,
            "duplicate z=2 TrackID matches were found. Using the first match per TrackID."
        )

    # Keep only the first row per TrackID so the later dictionary is deterministic.
    unique_found = {}
    for item in abs_indices_found:
        unique_found.setdefault(item[1], item)
    abs_indices_found = list(unique_found.values())

    if abs_indices_found:
        abs_idxs_arr = np.array([item[0] for item in abs_indices_found], dtype=np.int64)
        track_arr = np.array([item[1] for item in abs_indices_found], dtype=np.int64)
        halo_arr = np.array([item[2] for item in abs_indices_found], dtype=float)
        iscen_arr = np.array([item[3] for item in abs_indices_found], dtype=object)

        # IMPORTANT: all selective reads happen while the HDF5 file is open.
        m_z2_sel = safe_read_dataset(ds_mz2, abs_idxs_arr)
        r50_z2_sel = safe_read_dataset(ds_r50, abs_idxs_arr)
        bh_z2_sel = safe_read_dataset(ds_bh, abs_idxs_arr)
        center_z2_sel = safe_read_dataset(ds_center, abs_idxs_arr)
        comvel_z2_sel = safe_read_dataset(ds_comvel, abs_idxs_arr)

        # z=2 raw -> physical units.
        z2 = 2.0
        comov_to_phys = 1.0 / (1.0 + z2)

        m_z2_sel = np.asarray(m_z2_sel).ravel() * Mu
        r50_z2_sel = np.asarray(r50_z2_sel).ravel() * comov_to_phys * 1e3
        bh_z2_sel = np.asarray(bh_z2_sel).ravel() * Mu

        center_z2_sel = np.asarray(center_z2_sel)
        if center_z2_sel.ndim == 1 and center_z2_sel.size == 3 * len(abs_idxs_arr):
            center_z2_sel = center_z2_sel.reshape((-1, 3))
        if center_z2_sel.ndim == 2 and center_z2_sel.shape[1] == 3:
            center_z2_sel = center_z2_sel * comov_to_phys * 1e3
        else:
            center_z2_sel = np.full((len(abs_idxs_arr), 3), np.nan)

        comvel_z2_kms_sel = convert_raw_vel_to_kms(comvel_z2_sel, z2)
        if comvel_z2_kms_sel is None:
            comvel_z2_kms_sel = np.full((len(abs_idxs_arr), 3), np.nan)

        for i in range(len(abs_idxs_arr)):
            iscen_raw = iscen_arr[i]
            try:
                iscen_float = float(iscen_raw)
                iscen = bool(int(iscen_float)) if np.isfinite(iscen_float) else None
            except Exception:
                iscen = None

            row = {
                "track_id": int(track_arr[i]),
                "z2_snapshot": Z2_SNAP,
                "halo_index_z2": int(halo_arr[i]) if np.isfinite(halo_arr[i]) else np.nan,
                "is_central_z2": iscen,
                "m_z2": float(m_z2_sel[i]) if np.isfinite(m_z2_sel[i]) else np.nan,
                "r50_z2_kpc": float(r50_z2_sel[i]) if np.isfinite(r50_z2_sel[i]) else np.nan,
                "bh_z2": float(bh_z2_sel[i]) if np.isfinite(bh_z2_sel[i]) else np.nan,
                "center_x_z2": float(center_z2_sel[i, 0]) if np.isfinite(center_z2_sel[i, 0]) else np.nan,
                "center_y_z2": float(center_z2_sel[i, 1]) if np.isfinite(center_z2_sel[i, 1]) else np.nan,
                "center_z_z2": float(center_z2_sel[i, 2]) if np.isfinite(center_z2_sel[i, 2]) else np.nan,
                "v_x_z2_kms": float(comvel_z2_kms_sel[i, 0]) if np.isfinite(comvel_z2_kms_sel[i, 0]) else np.nan,
                "v_y_z2_kms": float(comvel_z2_kms_sel[i, 1]) if np.isfinite(comvel_z2_kms_sel[i, 1]) else np.nan,
                "v_z_z2_kms": float(comvel_z2_kms_sel[i, 2]) if np.isfinite(comvel_z2_kms_sel[i, 2]) else np.nan,
            }
            found_z2_rows.append(row)

    # Explicit placeholders for any missing TrackIDs.
    for missing_tr in sorted(tracks_remaining):
        found_z2_rows.append({
            "track_id": int(missing_tr),
            "z2_snapshot": Z2_SNAP,
            "halo_index_z2": np.nan,
            "is_central_z2": None,
            "m_z2": np.nan,
            "r50_z2_kpc": np.nan,
            "bh_z2": np.nan,
            "center_x_z2": np.nan,
            "center_y_z2": np.nan,
            "center_z_z2": np.nan,
            "v_x_z2_kms": np.nan,
            "v_y_z2_kms": np.nan,
            "v_z_z2_kms": np.nan,
        })

print(f"Finished z=2 scan: built {len(found_z2_rows)} entries.")

# -----------------------------------------------------------------------------
# Build final z=0 + z=2 table
# -----------------------------------------------------------------------------
map_z2 = {int(row["track_id"]): row for row in found_z2_rows}

# Keep the original df_out exactly restricted to the z=0 extreme relics.
# Separately retain z=0 compact non-relics at their z=2 positions for the
# BH-ratio plot. This avoids changing the existing mass-size/compactness plots.
compact_nonrelic_z2_rows: list[dict] = []

for i_local in np.flatnonzero(mask_compact_nonrelic):
    track_raw = z0_track_matched[i_local]
    if not np.isfinite(track_raw):
        continue

    track_id = int(track_raw)
    z2info = map_z2.get(track_id)
    if z2info is None:
        continue

    m2 = z2info.get("m_z2", np.nan)
    bh2 = z2info.get("bh_z2", np.nan)
    if not (
        np.isfinite(m2)
        and m2 > 0
        and np.isfinite(bh2)
        and bh2 > 0
    ):
        continue

    compact_nonrelic_z2_rows.append({
        "track_id": track_id,
        "log10_mstar_z2": np.log10(m2),
        "log10_bh_ratio_z2": np.log10(bh2 / m2),
    })

print(
    "z=0 compact non-relics successfully traced to z=2 with finite BH ratio:",
    len(compact_nonrelic_z2_rows),
)

rows_out: list[dict] = []

extreme_positions = np.flatnonzero(mask_extreme)
for i_local in extreme_positions:
    abs_idx = sel_global_idx[i_local]
    track_raw = z0_track_matched[i_local]
    if not np.isfinite(track_raw):
        continue
    track_id = int(track_raw)

    row = {
        "track_id": track_id,
        "dor_z0": float(dor_for_selected[i_local]),
        "halo_index_z0": int(z0_haloidx_matched[i_local]),
        "is_central_z0": bool(z0_iscentral[i_local]),
        "m_z0": float(z0_mass[i_local]),
        "r50_z0_kpc": float(z0_r50[i_local]),
        "bh_z0": float(z0_bh[i_local]) if np.isfinite(z0_bh[i_local]) else np.nan,
    }

    if z0_center.ndim == 2 and z0_center.shape[1] == 3:
        row["center_x_z0"] = float(z0_center[i_local, 0])
        row["center_y_z0"] = float(z0_center[i_local, 1])
        row["center_z_z0"] = float(z0_center[i_local, 2])
    else:
        row["center_x_z0"] = row["center_y_z0"] = row["center_z_z0"] = np.nan

    if comvel0_kms is not None and comvel0_kms.ndim == 2 and comvel0_kms.shape[1] == 3:
        matched_i = i_local
        row["v_x_z0_kms"] = float(comvel0_kms[matched_i, 0])
        row["v_y_z0_kms"] = float(comvel0_kms[matched_i, 1])
        row["v_z_z0_kms"] = float(comvel0_kms[matched_i, 2])
    else:
        row["v_x_z0_kms"] = row["v_y_z0_kms"] = row["v_z_z0_kms"] = np.nan

    z2info = map_z2.get(track_id, {})
    row.update({
        "halo_index_z2": z2info.get("halo_index_z2", np.nan),
        "is_central_z2": z2info.get("is_central_z2", None),
        "m_z2": z2info.get("m_z2", np.nan),
        "r50_z2_kpc": z2info.get("r50_z2_kpc", np.nan),
        "bh_z2": z2info.get("bh_z2", np.nan),
        "center_x_z2": z2info.get("center_x_z2", np.nan),
        "center_y_z2": z2info.get("center_y_z2", np.nan),
        "center_z_z2": z2info.get("center_z_z2", np.nan),
        "v_x_z2_kms": z2info.get("v_x_z2_kms", np.nan),
        "v_y_z2_kms": z2info.get("v_y_z2_kms", np.nan),
        "v_z_z2_kms": z2info.get("v_z_z2_kms", np.nan),
    })

    # Derived quantities at z=0 and z=2.
    for zlabel in ("z0", "z2"):
        m = row.get(f"m_{zlabel}", np.nan)
        r = row.get(f"r50_{zlabel}_kpc", np.nan)
        bh = row.get(f"bh_{zlabel}", np.nan)

        row[f"log10_mstar_{zlabel}"] = np.log10(m) if np.isfinite(m) and m > 0 else np.nan
        row[f"log10_bh_ratio_{zlabel}"] = (
            np.log10(bh / m)
            if np.isfinite(bh) and bh > 0 and np.isfinite(m) and m > 0
            else np.nan
        )
        row[f"compactness_sigma15_{zlabel}"] = (
            np.log10(m / np.power(r, 1.5))
            if np.isfinite(m) and m > 0 and np.isfinite(r) and r > 0
            else np.nan
        )

    row["delta_log10_mstar_z0_minus_z2"] = (
        row["log10_mstar_z0"] - row["log10_mstar_z2"]
        if np.isfinite(row["log10_mstar_z0"]) and np.isfinite(row["log10_mstar_z2"])
        else np.nan
    )
    row["delta_compactness_z0_minus_z2"] = (
        row["compactness_sigma15_z0"] - row["compactness_sigma15_z2"]
        if np.isfinite(row["compactness_sigma15_z0"])
        and np.isfinite(row["compactness_sigma15_z2"])
        else np.nan
    )

    rows_out.append(row)

df_out = pd.DataFrame(rows_out).sort_values("track_id").reset_index(drop=True)

out_csv = OUTDIR / "extreme_relics_z0_to_z2_summary.csv"
df_out.to_csv(out_csv, index=False)
print("Wrote summary CSV:", out_csv)

# -----------------------------------------------------------------------------
# Summary diagnostics
# -----------------------------------------------------------------------------
valid_z2 = np.isfinite(df_out["m_z2"].to_numpy())
print(f"Valid z=2 stellar masses: {valid_z2.sum()} / {len(df_out)}")
print(f"Valid z=2 half-mass radii: {np.isfinite(df_out['r50_z2_kpc']).sum()} / {len(df_out)}")
print(f"Valid z=2 BH masses: {np.isfinite(df_out['bh_z2']).sum()} / {len(df_out)}")
print(
    "Valid z=2 compact non-relics with finite BH ratio:",
    len(compact_nonrelic_z2_rows),
)

# -----------------------------------------------------------------------------
# z=2 plots -- all based on the same z=0-selected relic sample
# -----------------------------------------------------------------------------
df = df_out.loc[np.isfinite(df_out["m_z2"])].copy()

if df.empty:
    print("No objects with valid z=2 stellar masses; skipping z=2 plots.")
    sys.exit(0)

print(f"Generating z=2 plots for {len(df)} relics with valid z=2 mass.")

# ==============================================================
# 1. BH mass ratio vs stellar mass -- z ~ 2
#    Full z=2 population median + individual z=0 extreme relics
# ==============================================================

print("\nGenerating z=2 BH-ratio vs stellar-mass plot...")

# --------------------------------------------------------------
# Parameters
# --------------------------------------------------------------
# Use the same stellar-mass binning style as the z=0 analysis.
MASS_BINS = np.arange(9.0, 12.01, 0.25)

# Fine BH-ratio bins used to reconstruct median / percentiles
# from the full z=2 population without storing all 1.2e8 galaxies.
BH_RATIO_BINS = np.arange(-5.0, -0.49, 0.025)

# --------------------------------------------------------------
# Build z=2 population median / p16 / p84 relation
# --------------------------------------------------------------
print("Reading full z=2 population in chunks to build median relation...")

n_mass_bins = len(MASS_BINS) - 1
n_ratio_bins = len(BH_RATIO_BINS) - 1

# Histogram:
# rows    = stellar-mass bins
# columns = BH-ratio bins
H = np.zeros((n_mass_bins, n_ratio_bins), dtype=np.int64)

with h5py.File(snap_path_z2, "r") as fh:

    ds_mstar = fh["ExclusiveSphere/50kpc/StellarMass"]
    ds_bh = fh["ExclusiveSphere/50kpc/MostMassiveBlackHoleMass"]
    ds_r50 = fh["ExclusiveSphere/50kpc/HalfMassRadiusStars"]

    nrows = ds_mstar.shape[0]

    # Full z≈2 mass-size population for the background scatter plot
    full_logM = []
    full_logR = []

    for start in range(0, nrows, CHUNK):
        stop = min(start + CHUNK, nrows)

        m = np.asarray(ds_mstar[start:stop]).ravel() * Mu
        bh = np.asarray(ds_bh[start:stop]).ravel() * Mu

        valid = (
            np.isfinite(m)
            & np.isfinite(bh)
            & (m >= MIN_STELLAR_MASS)
            & (bh > 0)
        )

        r50 = (
            np.asarray(ds_r50[start:stop]).ravel()
            * (1.0 / (1.0 + 2.0))
            * 1e3
        )

        valid &= np.isfinite(r50) & (r50 > 0)

        if not np.any(valid):
            continue

        logM = np.log10(m[valid])
        logBHratio = np.log10(bh[valid] / m[valid])
        logR = np.log10(r50[valid])

        full_logM.append(logM)
        full_logR.append(logR)

        h, _, _ = np.histogram2d(
            logM,
            logBHratio,
            bins=[MASS_BINS, BH_RATIO_BINS],
        )

        H += h.astype(np.int64)

full_logM = np.concatenate(full_logM)
full_logR = np.concatenate(full_logR)
# --------------------------------------------------------------
# Reconstruct median / p16 / p84 from histogram
# --------------------------------------------------------------
bin_centres = 0.5 * (MASS_BINS[:-1] + MASS_BINS[1:])

medians = np.full(n_mass_bins, np.nan)
p16 = np.full(n_mass_bins, np.nan)
p84 = np.full(n_mass_bins, np.nan)

ratio_centres = 0.5 * (BH_RATIO_BINS[:-1] + BH_RATIO_BINS[1:])

for i in range(n_mass_bins):

    counts = H[i]

    if counts.sum() == 0:
        continue

    cumulative = np.cumsum(counts)
    total = cumulative[-1]

    # percentile helper
    def hist_percentile(q):
        target = q * total
        idx = np.searchsorted(cumulative, target, side="left")
        idx = min(max(idx, 0), len(ratio_centres) - 1)
        return ratio_centres[idx]

    p16[i] = hist_percentile(0.16)
    medians[i] = hist_percentile(0.50)
    p84[i] = hist_percentile(0.84)

finite_bins = np.isfinite(medians)

# --------------------------------------------------------------
# Prepare the z=0 extreme relics at z=2
# using their ORIGINAL z=0 SRG/SAG classification
# --------------------------------------------------------------

sel_ext_z2 = (
    np.isfinite(df["log10_mstar_z2"])
    & np.isfinite(df["log10_bh_ratio_z2"])
)

# Carry the ORIGINAL z=0 classification to z=2.
# A galaxy is an "SRG at z=2" here because it is an SRG
# in the z=0 sample; we are plotting its progenitor position at z=2.
sel_srg_z0class = (
    sel_ext_z2
    & np.isfinite(df["compactness_sigma15_z0"])
    & (df["compactness_sigma15_z0"] >= COMPACTNESS_CUT)
)

sel_sag_z0class = (
    sel_ext_z2
    & np.isfinite(df["compactness_sigma15_z0"])
    & (df["compactness_sigma15_z0"] < COMPACTNESS_CUT)
)

# Central / satellite split at z=2, but only for galaxies that
# were classified as SRGs at z=0, matching your original z=0 plot.
if "is_central_z2" in df.columns:
    cen_srg_z2 = (
        sel_srg_z0class
        & (df["is_central_z2"] == True)
    )

    sat_srg_z2 = (
        sel_srg_z0class
        & (df["is_central_z2"] == False)
    )
else:
    cen_srg_z2 = np.zeros(len(df), dtype=bool)
    sat_srg_z2 = np.zeros(len(df), dtype=bool)

print(
    "z=2 positions of z=0 SRGs:",
    int(sel_srg_z0class.sum())
)

print(
    "z=2 positions of z=0 SAGs:",
    int(sel_sag_z0class.sum())
)

print(
    "z=2 positions of z=0 SRGs with valid central/satellite flag:",
    int((cen_srg_z2 | sat_srg_z2).sum())
)

# --------------------------------------------------------------
# Plot
# --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Full z=2 population median relation
if np.any(finite_bins):

    x = bin_centres[finite_bins]
    med = medians[finite_bins]
    lo = med - p16[finite_bins]
    hi = p84[finite_bins] - med

    ax.errorbar(
        x,
        med,
        yerr=[lo, hi],
        fmt="o-",
        capsize=3,
        lw=1.5,
        zorder=100,
        label="z≈2 population median (16/84)",
    )

# --------------------------------------------------------------
# Individual z=0-defined populations at z=2
# --------------------------------------------------------------

# z=0 compact non-relics, shown at their z=2 positions
if compact_nonrelic_z2_rows:
    compact_nr = pd.DataFrame(compact_nonrelic_z2_rows)
    ax.scatter(
        compact_nr["log10_mstar_z2"],
        compact_nr["log10_bh_ratio_z2"],
        facecolor="lightgrey",
        edgecolor="lightgrey",
        s=10,
        alpha=0.5,
        linewidth=0.5,
        zorder=10,
        label="z=0 compact non-relics at z≈2",
    )

# z=0 SAGs, shown at their z=2 positions
if np.any(sel_sag_z0class):
    ax.scatter(
        df.loc[sel_sag_z0class, "log10_mstar_z2"],
        df.loc[sel_sag_z0class, "log10_bh_ratio_z2"],
        facecolor="C1",
        edgecolor="C1",
        s=15,
        marker="d",
        linewidth=0.7,
        zorder=110,
        label="z=0 SAGs at z≈2",
    )

# z=0 SRGs, shown at their z=2 positions
if np.any(sel_srg_z0class):
    ax.scatter(
        df.loc[sel_srg_z0class, "log10_mstar_z2"],
        df.loc[sel_srg_z0class, "log10_bh_ratio_z2"],
        facecolor="C2",
        edgecolor="C2",
        s=30,
        marker="*",
        linewidth=0.7,
        zorder=120,
        label="z=0 SRGs at z≈2",
    )

# --------------------------------------------------------------
# z=2 SRG central / satellite median markers
# --------------------------------------------------------------
for mask, edgecol, facecol, label in [
    (cen_srg_z2, "red", "green", "z≈2 SRG centrals (median)"),
    (sat_srg_z2, "blue", "green", "z≈2 SRG satellites (median)"),
]:
    if np.sum(mask) == 0:
        continue

    x_med = np.nanmedian(df.loc[mask, "log10_mstar_z2"])
    y_med = np.nanmedian(df.loc[mask, "log10_bh_ratio_z2"])

    x_std = np.nanstd(df.loc[mask, "log10_mstar_z2"])
    y_std = np.nanstd(df.loc[mask, "log10_bh_ratio_z2"])

    ax.errorbar(
        x_med,
        y_med,
        xerr=x_std,
        yerr=y_std,
        fmt="*",
        markersize=16,
        markerfacecolor=facecol,
        markeredgecolor=edgecol,
        markeredgewidth=2.0,
        ecolor=edgecol,
        elinewidth=1.8,
        capsize=4,
        zorder=200,
        label=label,
    )

# --------------------------------------------------------------
# Formatting
# --------------------------------------------------------------
ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
ax.set_ylabel(r"$\log_{10}(M_{\rm BH}/M_\star)$")

ax.set_title(r"$z\approx2$")

ax.grid(True, alpha=0.25)
ax.legend(loc="best", fontsize=9)

ax.relim()
ax.autoscale_view(True, True, True)

ymin, ymax = ax.get_ylim()
ypad = 0.06 * (ymax - ymin)
ax.set_ylim(ymin - ypad, ymax + ypad)

fig.tight_layout()

outbh = OUTDIR / "BHratio_log10_median_vs_mass_z2.pdf"
fig.savefig(outbh, dpi=200, bbox_inches="tight")

outbh_png = OUTDIR / "BHratio_log10_median_vs_mass_z2.png"
fig.savefig(outbh_png, dpi=200, bbox_inches="tight")

plt.close(fig)

print("Saved:", outbh)
print("Saved:", outbh_png)

# 2. Mass-size plane
mask = (
    np.isfinite(df["log10_mstar_z2"])
    & np.isfinite(df["r50_z2_kpc"])
    & (df["r50_z2_kpc"] > 0)
)
if mask.any():
    fig, ax = plt.subplots(figsize=(7, 6))
    # z=0 SAGs at their z≈2 positions
    ax.scatter(
        df.loc[sel_sag_z0class, "log10_mstar_z2"],
        np.log10(df.loc[sel_sag_z0class, "r50_z2_kpc"]),
        marker="d",
        s=18,
        color="C1",
        alpha=0.8,
        label="z=0 SAGs",
    )

    # z=0 SRGs at their z≈2 positions
    ax.scatter(
        df.loc[sel_srg_z0class, "log10_mstar_z2"],
        np.log10(df.loc[sel_srg_z0class, "r50_z2_kpc"]),
        marker="*",
        s=45,
        color="C2",
        alpha=0.9,
        label="z=0 SRGs",
    )
    # Compactness threshold (Sigma_1.5 = 9.75)
    xline = np.linspace(9.0, 12.0, 200)
    yline = (xline - COMPACTNESS_CUT) / 1.5

    ax.plot(
        xline,
        yline,
        "--",
        color="k",
        lw=1.5,
        label=rf"$\log\Sigma_{{1.5}}={COMPACTNESS_CUT:.2f}$",
    )
    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(R_{50}/{\rm kpc})$")
    # ax.set_title("z≈2 mass-size plane: z=0 extreme relics")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "z2_mass_size_extremes.png", dpi=200)
    plt.close(fig)
    print("Saved:", OUTDIR / "z2_mass_size_extremes.png")

# 3. Compactness distribution
mask = np.isfinite(df["compactness_sigma15_z2"])
if mask.any():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df.loc[mask, "compactness_sigma15_z2"], bins=20)
    ax.set_xlabel(r"$\log_{10}(M_\star/R_{50}^{1.5})$")
    ax.set_ylabel("N")
    ax.set_title("z≈2 compactness of z=0 extreme relics")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTDIR / "z2_compactness_extremes.png", dpi=200)
    plt.close(fig)
    print("Saved:", OUTDIR / "z2_compactness_extremes.png")

# 4. Central/satellite fraction
if "is_central_z2" in df.columns:
    central_numeric = pd.to_numeric(df["is_central_z2"], errors="coerce")
    n_central_valid = central_numeric.notna().sum()
    if n_central_valid:
        central_fraction = float((central_numeric.dropna() > 0.5).mean())
        text_path = OUTDIR / "z2_central_fraction.txt"
        text_path.write_text(
            f"z=2 central fraction among z=0 extreme relics with valid IsCentral: "
            f"{central_fraction:.6f}\n"
            f"N with valid IsCentral: {n_central_valid}\n"
        )
        print("Saved:", text_path)

# ==========================================================
# Full z≈2 mass-size plane with highlighted progenitors
# ==========================================================

fig, ax = plt.subplots(figsize=(8, 6))

# ----------------------------------------------------------
# Entire z≈2 galaxy population
# ----------------------------------------------------------
ax.scatter(
    full_logM,
    full_logR,
    s=8,
    color="lightgrey",
    alpha=0.6,
    rasterized=True,
    label=r"simulated galaxies at $z\approx2$",
)

# ----------------------------------------------------------
# Compactness threshold
# ----------------------------------------------------------
xm = np.linspace(
    np.nanmin(full_logM) - 0.2,
    np.nanmax(full_logM) + 0.2,
    400,
)

compact_line = (xm - COMPACTNESS_CUT) / 1.5

ax.plot(
    xm,
    compact_line,
    "--",
    color="black",
    lw=2,
    label=rf"compactness threshold $\log_{{10}}\Sigma_{{1.5}}={COMPACTNESS_CUT}$",
)

# ----------------------------------------------------------
# z=0 SAG progenitors at z≈2
# ----------------------------------------------------------
ax.scatter(
    df.loc[sel_sag_z0class, "log10_mstar_z2"],
    np.log10(df.loc[sel_sag_z0class, "r50_z2_kpc"]),
    marker="d",
    s=15,
    color="C1",
    edgecolors="none",
    zorder=20,
    label=fr"non-compact SAGs (DoR > {EXTREME_DOR})",
)

# ----------------------------------------------------------
# z=0 SRG progenitors at z≈2
# ----------------------------------------------------------
ax.scatter(
    df.loc[sel_srg_z0class, "log10_mstar_z2"],
    np.log10(df.loc[sel_srg_z0class, "r50_z2_kpc"]),
    marker="*",
    s=30,
    color="C2",
    edgecolors="none",
    zorder=30,
    label=fr"SRGs (DoR > {EXTREME_DOR})",
)

# ----------------------------------------------------------
# Formatting
# ----------------------------------------------------------
ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
ax.set_ylabel(r"$\log_{10}(R_{1/2,\star}/{\rm kpc})$")

ax.grid(True)

ax.legend(
    fontsize=9,
    loc="lower right",
)

fig.tight_layout()

fig.savefig(
    OUTDIR / "z2_mass_size_full_population_with_SRGs.png",
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)

print("Saved:", OUTDIR / "z2_mass_size_full_population_with_SRGs.png")

print("All requested z=2 analysis completed.")