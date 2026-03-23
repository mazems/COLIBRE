#!/usr/bin/env python3
"""
extreme_relics_trace_z2.py

Find extreme relics at z=0 (from your DoR CSV + SOAP match), follow their TrackId back to z~2
(snapshot label '0076'), gather basic properties (mass, r50, BH mass, IsCentral) at z=0 and z=2,
and write a CSV summarising the comparison.

Designed to be small, robust and memory-friendly (chunked TrackId scan where needed).
Depends on your existing `common.read_group_data_colibre` helper and on h5py/pandas/numpy.

Output:
  out/extreme_relics_z0_to_z2_summary.csv
"""
from __future__ import annotations
import os
import sys
import csv
import math
import gc
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py

# adjust to your environment
MODEL_NAME = 'L0200N3008/THERMAL_AGN/'
MODEL_DIR = '/mnt/su3-pro/colibre/' + MODEL_NAME
OUTDIR = 'out'
os.makedirs(OUTDIR, exist_ok=True)

CSV_DOR = "sfh_times_all_with_DoR_variants_corrected.csv.gz"
Z0_SNAP = '0127'   # z=0 snapshot label (used to produce your original matched arrays)
Z2_SNAP = '0076'   # z~2 snapshot label in your mapping
MIN_STELLAR_MASS = 1e9
EXTREME_DOR = 0.6

# chunk size when scanning big HDF5 datasets (tweak if needed)
CHUNK = 80_000

# Unit conversions (same convention used in your scripts)
# NOTE: these match how you convert masses, radii, times; velocity conversion below follows same pattern
Mu = 1.988e43 / 1.989e33       # sim mass unit -> Msun
tu = 3.086e19 / 3.154e7        # time unit -> yr
# comoving -> physical factor will be applied per snapshot if needed (z-dependent)
# We will treat positions scaled as comoving * (1/(1+z)) * 1e3 -> kpc and velocities similarly

# helper to try candidate dataset names in HDF5 groups
def find_dataset_in_group(group, candidates):
    for cand in candidates:
        # try direct presence
        if cand in group:
            return cand
        # try under "InputHalos" or nested names
        if "InputHalos" in group and cand in group["InputHalos"]:
            return "InputHalos/" + cand
    # fallback: try scanning top-level keys for something similar
    for k in group.keys():
        if any(s in k for s in candidates):
            return k
    return None

# read the DoR CSV and get dict subhalo_id -> DoR
if not os.path.exists(CSV_DOR):
    raise SystemExit(f"DoR CSV not found: {CSV_DOR}")
print("Loading DoR CSV:", CSV_DOR)
df_dor = pd.read_csv(CSV_DOR, low_memory=False)
# find ID column heuristically
id_col = None
for c in ("subhalo_id", "HaloCatalogueIndex", "subhaloId", "HaloIndex", "track_id", "TrackId"):
    if c in df_dor.columns:
        id_col = c
        break
if id_col is None:
    id_col = df_dor.columns[0]
    print("Warning: couldn't find canonical ID column, using:", id_col)

# normalize numeric ids
s = df_dor[id_col].astype(str).str.replace("\r", "").str.strip()
df_dor["_subid_num"] = pd.to_numeric(s, errors="coerce").astype("Int64")
df_dor = df_dor[df_dor["_subid_num"].notna()].copy()
df_dor["subhalo_id"] = df_dor["_subid_num"].astype("int64")
df_dor.drop(columns=["_subid_num"], inplace=True)

# find a DoR column
dor_col = None
for cand in ("DoR_t95", "DoR_t90", "DoR_t998", "DoR", "DoR_tfin"):
    if cand in df_dor.columns:
        dor_col = cand; break
if dor_col is None:
    for c in df_dor.columns:
        if c.lower().startswith("dor"):
            dor_col = c; break
if dor_col is None:
    raise SystemExit("No DoR-like column found in CSV.")
print("Using DoR column:", dor_col)

# build lookup
dor_lookup = {}
for _, row in df_dor.iterrows():
    try:
        sid = int(row["subhalo_id"])
        v = row.get(dor_col, np.nan)
        if pd.isna(v):
            continue
        dor_lookup[sid] = float(v)
    except Exception:
        continue
print("Loaded DoR entries:", len(dor_lookup))

# ------------------------- Read z=0 SOAP to find matched SOAP rows and TrackIds -------------------------
print("Reading z=0 SOAP (minimal fields) via common.read_group_data_colibre...")

# import your helper; script expects common to be importable (same as your pipeline)
try:
    import common
except Exception as e:
    raise SystemExit("Couldn't import `common`. Run this script from your project where `common` is available.") from e

# fields to request at z=0 (same as your original)
fields_gal = {'ExclusiveSphere/50kpc': (
    'StellarMass', 'HalfMassRadiusStars', 'CentreOfMass', 'MostMassiveBlackHoleMass', 'CentreOfMassVelocity'
)}
fields_id = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/TrackId')}

h5_gal = common.read_group_data_colibre(MODEL_DIR, Z0_SNAP, fields_gal)
h5_id = common.read_group_data_colibre(MODEL_DIR, Z0_SNAP, fields_id)

(m30, r50, centers0, bh_mass0_raw, comvel0_raw) = h5_gal
(halo_index_all, is_central_all, track_id_all) = h5_id

# coerce to numpy arrays and apply units
m30 = np.asarray(m30).ravel() * Mu
r50 = np.asarray(r50).ravel() * (1.0 / (1.0 + 0.0)) * 1e3   # z=0 -> kpc
centers0 = np.asarray(centers0) * (1.0 / (1.0 + 0.0)) * 1e3
bh_mass0 = np.asarray(bh_mass0_raw).ravel() * Mu
# velocities: keep raw array for now (will convert later if present)
comvel0 = np.asarray(comvel0_raw)
try:
    if comvel0.ndim == 1 and comvel0.size % 3 == 0:
        comvel0 = comvel0.reshape((-1,3))
except Exception:
    pass

halo_index_all = np.asarray(halo_index_all).ravel()
is_central_all = np.asarray(is_central_all).ravel().astype(bool)
track_id_all = np.asarray(track_id_all).ravel()

# select by MIN_STELLAR_MASS
sel = np.where(m30 >= MIN_STELLAR_MASS)[0]
if sel.size == 0:
    raise SystemExit("No z=0 galaxies above MIN_STELLAR_MASS.")
print("Selected z=0 SOAP rows (m >= MIN_STELLAR_MASS):", sel.size)

# map DoR (dor_lookup keys are subhalo id values that should match halo_index entries)
halo_idx_selected = halo_index_all[sel].astype(int)
# Try direct mapping halo_idx -> dor_lookup
dor_for_selected = np.array([dor_lookup.get(int(h), np.nan) for h in halo_idx_selected], dtype=float)
matched_positions = np.where(np.isfinite(dor_for_selected))[0]
print(f"Matched DoR entries for selected SOAP rows: {matched_positions.size} / {halo_idx_selected.size}")

if matched_positions.size == 0:
    # try +/-1 fallback (common off-by-one convention)
    dor_try = np.array([dor_lookup.get(int(h-1), np.nan) for h in halo_idx_selected], dtype=float)
    if np.any(np.isfinite(dor_try)):
        dor_for_selected = dor_try
        matched_positions = np.where(np.isfinite(dor_for_selected))[0]
        print("Matched after halo_idx-1 fallback:", matched_positions.size)
    else:
        dor_try2 = np.array([dor_lookup.get(int(h+1), np.nan) for h in halo_idx_selected], dtype=float)
        if np.any(np.isfinite(dor_try2)):
            dor_for_selected = dor_try2
            matched_positions = np.where(np.isfinite(dor_for_selected))[0]
            print("Matched after halo_idx+1 fallback:", matched_positions.size)

# aligned arrays (for matched subset)
sel_global_idx = sel[matched_positions]              # indices into the full SOAP arrays
z0_track_matched = track_id_all[sel_global_idx]     # TrackId for matched objects
z0_haloidx_matched = halo_idx_selected[matched_positions]
z0_iscentral = is_central_all[sel_global_idx]
z0_mass = m30[sel_global_idx]
z0_r50 = r50[sel_global_idx]
z0_bh = bh_mass0[sel_global_idx]
z0_center = centers0[sel_global_idx]
# velocities if present and shaped correctly: convert to km/s using same scaling pattern you used elsewhere.
# SOAP velocity units -> convert to km/s: we follow same recipe as in your other script:
# centre_of_mass_vel (raw) * comov_to_physical_length * 1e3 / tu  -> km/s (with z applied)
def convert_raw_vel_to_kms(raw_vel, z_snap):
    if raw_vel is None:
        return None
    arr = np.asarray(raw_vel)
    # coerce shape
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape((-1,3))
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    comov_to_physical = 1.0 / (1.0 + z_snap)
    # same pattern as you used earlier
    return arr * comov_to_physical * 1e3 / tu

if comvel0 is not None and comvel0.size > 0:
    comvel0_kms = convert_raw_vel_to_kms(comvel0, 0.0)
else:
    comvel0_kms = None

# pick extremes at z0
mask_extreme = (dor_for_selected > EXTREME_DOR) & np.isfinite(dor_for_selected)
n_extreme = int(np.sum(mask_extreme))
print(f"Extremes at z0 (DoR>{EXTREME_DOR}): {n_extreme}")
if n_extreme == 0:
    print("No extremes matched - exiting.")
    sys.exit(0)

# build a set of tracks (finite)
tracks_to_find = set(int(t) for t in z0_track_matched[mask_extreme] if np.isfinite(t))
print("Tracks to follow (sample up to 20):", list(tracks_to_find)[:20])

# ------------------------- Scan z=2 snapshot by TrackId (chunked) -------------------------
# We'll attempt to read TrackId, HaloCatalogueIndex and IsCentral from z=2 file and record matches.

def find_snapshot_path(snap_label):
    p1 = os.path.join(MODEL_DIR, "SOAP-HBT", f"halo_properties_{snap_label}.hdf5")
    p2 = os.path.join(MODEL_DIR, "SOAP",     f"halo_properties_{snap_label}.hdf5")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None

snap_path_z2 = find_snapshot_path(Z2_SNAP)
if snap_path_z2 is None:
    raise SystemExit(f"Could not find z=2 snapshot file for label {Z2_SNAP} under {MODEL_DIR}")

print("Scanning z=2 snapshot:", snap_path_z2)

# candidate dataset names to try
track_candidates = ["HBTplus/TrackId", "HBT/TrackId", "TrackId", "HBTplus/track_id"]
halo_candidates  = ["HaloCatalogueIndex", "HaloIndex", "Halo/Index", "InputHalos/HaloCatalogueIndex"]
iscen_candidates = ["IsCentral", "is_central"]

# also try to collect galaxy properties at z2 for matched tracks: StellarMass, HalfMassRadiusStars, MostMassiveBlackHoleMass, CentreOfMass
gal_candidates = ["ExclusiveSphere/50kpc/StellarMass", "ExclusiveSphere/50kpc/HalfMassRadiusStars",
                  "ExclusiveSphere/50kpc/MostMassiveBlackHoleMass", "ExclusiveSphere/50kpc/CentreOfMass"]

found_z2_rows = []   # list of dicts per found track

with h5py.File(snap_path_z2, "r") as fh:
    # locate datasets
    # TrackId dataset
    ds_track_name = None
    for cand in track_candidates:
        if cand in fh:
            ds_track_name = cand; break
        if "InputHalos" in fh and cand in fh["InputHalos"]:
            ds_track_name = "InputHalos/" + cand; break
    if ds_track_name is None:
        # try scanning keys
        for k in fh.keys():
            if "Track" in k or "track" in k:
                ds_track_name = k; break
    if ds_track_name is None:
        raise SystemExit("No TrackId-like dataset found in z=2 snapshot.")

    print("Using TrackId dataset:", ds_track_name)

    # halo idx
    ds_halo_name = find_dataset_in_group(fh, halo_candidates)
    if ds_halo_name is None:
        print("Warning: HaloCatalogueIndex-like dataset not found in z=2 snapshot; will record NaN.")
    else:
        print("Using halo idx dataset:", ds_halo_name)

    # iscentral
    ds_iscen_name = find_dataset_in_group(fh, iscen_candidates)
    if ds_iscen_name is None:
        print("Warning: IsCentral-like dataset not found in z=2 snapshot; will record NaN.")
    else:
        print("Using IsCentral dataset:", ds_iscen_name)

    # optional galaxy properties (try to get them via common helper first)
    # We'll try reading StellarMass, HalfMassRadiusStars, MostMassiveBlackHoleMass, CentreOfMass if present
    # but these may be under ExclusiveSphere/50kpc and our scanning by TrackId is chunked - reading entire arrays may be ok
    # if data is large you can adjust CHUNK or add more complicated chunk reading.
    
    # ------------------ Efficient two-pass z=2 scan (memory-friendly) ------------------
    print("Efficient two-pass z=2 scan (chunked TrackId search, then selective reads).")

    snap_path_z2 = find_snapshot_path(Z2_SNAP)
    if snap_path_z2 is None:
        raise SystemExit(f"Could not find z=2 snapshot file for label {Z2_SNAP} under {MODEL_DIR}")

    with h5py.File(snap_path_z2, "r") as fh:
        # locate track dataset name (same logic you used earlier)
        ds_track_name = None
        for cand in track_candidates:
            if cand in fh:
                ds_track_name = cand; break
            if "InputHalos" in fh and cand in fh["InputHalos"]:
                ds_track_name = "InputHalos/" + cand; break
        if ds_track_name is None:
            # try scanning keys
            for k in fh.keys():
                if "Track" in k or "track" in k:
                    ds_track_name = k; break
        if ds_track_name is None:
            raise SystemExit("No TrackId-like dataset found in z=2 snapshot.")
        print("Using TrackId dataset:", ds_track_name)

        # find halo/iscentral dataset names (may be missing)
        ds_halo_name = find_dataset_in_group(fh, halo_candidates)
        ds_iscen_name = find_dataset_in_group(fh, iscen_candidates)
        if ds_halo_name is None:
            print("Warning: HaloCatalogueIndex-like dataset not found in z=2 snapshot; will record NaN.")
        else:
            print("Using halo idx dataset:", ds_halo_name)
        if ds_iscen_name is None:
            print("Warning: IsCentral-like dataset not found in z=2 snapshot; will record NaN.")
        else:
            print("Using IsCentral dataset:", ds_iscen_name)

        # second-pass dataset candidates (ExclusiveSphere) we will read *selectively* if present
        # try to find dataset objects (not using helper to avoid big memory allocations)
        def find_dsobj(base, cand_list):
            # try exact paths first
            for cand in cand_list:
                if cand in base:
                    return base[cand]
                if "ExclusiveSphere" in base and cand.split("/")[-1] in base["ExclusiveSphere"]:
                    return base["ExclusiveSphere"][cand.split("/")[-1]]
            # scan keys for hints
            for k in base.keys():
                if any(s in k for s in ["ExclusiveSphere", "StellarMass", "HalfMassRadius", "MostMassiveBlackHole", "CentreOfMass"]):
                    obj = base[k]
                    # if it's a group, try inside
                    if isinstance(obj, h5py.Group):
                        for cand in cand_list:
                            name = cand.split("/")[-1]
                            if name in obj:
                                return obj[name]
            return None

        ds_mz2 = find_dsobj(fh, ["ExclusiveSphere/50kpc/StellarMass", "ExclusiveSphere/StellarMass", "StellarMass"])
        ds_r50 = find_dsobj(fh, ["ExclusiveSphere/50kpc/HalfMassRadiusStars", "ExclusiveSphere/HalfMassRadiusStars", "HalfMassRadiusStars"])
        ds_bh  = find_dsobj(fh, ["ExclusiveSphere/50kpc/MostMassiveBlackHoleMass", "ExclusiveSphere/MostMassiveBlackHoleMass", "MostMassiveBlackHoleMass"])
        ds_center = find_dsobj(fh, ["ExclusiveSphere/50kpc/CentreOfMass", "ExclusiveSphere/CentreOfMass", "CentreOfMass"])
        ds_comvel  = find_dsobj(fh, ["ExclusiveSphere/50kpc/CentreOfMassVelocity", "ExclusiveSphere/CentreOfMassVelocity", "CentreOfMassVelocity"])

        # CHUNK-scan TrackId to collect absolute indices for tracks_to_find
        d_track = fh[ds_track_name]
        nrows = d_track.shape[0]
        print("z=2 TrackId table length:", nrows)

        tracks_remaining = set(tracks_to_find)
        abs_indices_found = []   # list of (abs_idx, track_id, halo_val, is_central_val)

        for start in range(0, nrows, CHUNK):
            stop = min(start + CHUNK, nrows)
            tr_chunk = d_track[start:stop]
            # convert to integers for membership test without copying huge arrays if possible
            # use numpy vectorized isin against list(tracks_remaining)
            try:
                tr_chunk_int = np.asarray(tr_chunk, dtype=np.int64)
            except Exception:
                tr_chunk_int = np.asarray(tr_chunk, dtype=float)

            if len(tracks_remaining) == 0:
                break

            # build boolean mask of matches
            mask_in = np.isin(tr_chunk_int, list(tracks_remaining))
            if not np.any(mask_in):
                continue

            rel_idxs = np.nonzero(mask_in)[0]
            abs_idxs = rel_idxs + start

            # fetch halo/iscentral only for these abs_idxs (cheap small reads)
            if ds_halo_name is not None:
                try:
                    halo_vals = fh[ds_halo_name][abs_idxs]
                except Exception:
                    halo_vals = np.full(len(abs_idxs), np.nan)
            else:
                halo_vals = np.full(len(abs_idxs), np.nan)

            if ds_iscen_name is not None:
                try:
                    iscen_vals = fh[ds_iscen_name][abs_idxs]
                except Exception:
                    iscen_vals = np.full(len(abs_idxs), np.nan)
            else:
                iscen_vals = np.full(len(abs_idxs), np.nan)

            for local_i, ai in enumerate(abs_idxs):
                trv = int(tr_chunk_int[rel_idxs[local_i]])
                halov = halo_vals[local_i] if (halo_vals is not None and len(halo_vals)>local_i) else np.nan
                iscenv = iscen_vals[local_i] if (iscen_vals is not None and len(iscen_vals)>local_i) else np.nan
                abs_indices_found.append((int(ai), int(trv) if np.isfinite(trv) else trv, int(halov) if np.isfinite(halov) else np.nan, iscenv))
                # remove from remaining
                if trv in tracks_remaining:
                    tracks_remaining.remove(trv)

            if len(tracks_remaining) == 0:
                break

        print("Tracks found in z=2 (abs idxs):", len(abs_indices_found), " ; still missing:", len(tracks_remaining))

        # After gathering absolute indices, we can *selectively* read the heavy datasets only at those indices.
        # Convert to arrays of indices
        if len(abs_indices_found) == 0:
            # no matches found; add placeholders for missing tracks
            for missing_tr in list(tracks_remaining):
                found_z2_rows.append({
                    "track_id": int(missing_tr),
                    "z2_snapshot": Z2_SNAP,
                    "halo_index_z2": np.nan,
                    "is_central_z2": None,
                    "m_z2": np.nan, "r50_z2_kpc": np.nan, "bh_z2": np.nan,
                    "center_x_z2": np.nan, "center_y_z2": np.nan, "center_z_z2": np.nan,
                    "v_x_z2_kms": np.nan, "v_y_z2_kms": np.nan, "v_z_z2_kms": np.nan
                })
        else:
            abs_idxs_arr = np.array([t[0] for t in abs_indices_found], dtype=int)
            track_arr = np.array([t[1] for t in abs_indices_found], dtype=int)
            halo_arr  = np.array([t[2] for t in abs_indices_found], dtype=float)
            iscen_arr = np.array([t[3] for t in abs_indices_found])

            # Now read the heavy arrays at abs_idxs_arr (small reads)
            # If dataset missing, fill with nan
            def safe_read(ds, idxs):
                if ds is None:
                    return np.full(len(idxs), np.nan)
                try:
                    return np.asarray(ds[idxs])
                except Exception:
                    # fallback: iterative read to be extra memory-light
                    out = []
                    for ii in idxs:
                        try:
                            out.append(np.asarray(ds[int(ii)]))
                        except Exception:
                            out.append(np.nan)
                    return np.asarray(out)

            m_z2_sel = safe_read(ds_mz2, abs_idxs_arr)
            r50_z2_sel = safe_read(ds_r50, abs_idxs_arr)
            bh_z2_sel = safe_read(ds_bh, abs_idxs_arr)
            center_z2_sel = safe_read(ds_center, abs_idxs_arr)
            comvel_z2_sel = safe_read(ds_comvel, abs_idxs_arr) if ds_comvel is not None else None

            # unit conversions now (elementwise)
            comov_to_phys = 1.0 / (1.0 + 2.0)
            # masses -> Msun
            if m_z2_sel is not None:
                m_z2_sel = np.asarray(m_z2_sel).ravel() * Mu
            if r50_z2_sel is not None:
                r50_z2_sel = np.asarray(r50_z2_sel).ravel() * comov_to_phys * 1e3  # kpc
            if bh_z2_sel is not None:
                bh_z2_sel = np.asarray(bh_z2_sel).ravel() * Mu
            # centres: ensure shape Nx3
            if center_z2_sel is not None:
                cen = np.asarray(center_z2_sel)
                if cen.ndim == 1 and cen.size % 3 == 0:
                    cen = cen.reshape((-1,3))
                center_z2_sel = cen * comov_to_phys * 1e3
            # comvel -> convert to km/s using same function as earlier
            if comvel_z2_sel is not None:
                cv = np.asarray(comvel_z2_sel)
                if cv.ndim == 1 and cv.size % 3 == 0:
                    cv = cv.reshape((-1,3))
                comvel_z2_kms_sel = cv * comov_to_phys * 1e3 / tu
            else:
                comvel_z2_kms_sel = None

            # build found_z2_rows in the same order as abs_indices_found
            for idx_i in range(len(abs_idxs_arr)):
                ai = abs_idxs_arr[idx_i]
                trv = int(track_arr[idx_i])
                halov = halo_arr[idx_i] if not np.isnan(halo_arr[idx_i]) else np.nan
                iscenv = iscen_arr[idx_i] if not (iscen_arr[idx_i] is None) else None

                out = {
                    "track_id": int(trv),
                    "z2_snapshot": Z2_SNAP,
                    "halo_index_z2": int(halov) if np.isfinite(halov) else np.nan,
                    "is_central_z2": bool(iscenv) if (iscenv is not None and not (isinstance(iscenv, float) and np.isnan(iscenv))) else None
                }

                # fill selective reads safely
                try:
                    out["m_z2"] = float(m_z2_sel[idx_i]) if np.isfinite(m_z2_sel[idx_i]) else np.nan
                except Exception:
                    out["m_z2"] = np.nan
                try:
                    out["r50_z2_kpc"] = float(r50_z2_sel[idx_i]) if np.isfinite(r50_z2_sel[idx_i]) else np.nan
                except Exception:
                    out["r50_z2_kpc"] = np.nan
                try:
                    out["bh_z2"] = float(bh_z2_sel[idx_i]) if np.isfinite(bh_z2_sel[idx_i]) else np.nan
                except Exception:
                    out["bh_z2"] = np.nan
                try:
                    cx,cy,cz = center_z2_sel[idx_i]
                    out["center_x_z2"], out["center_y_z2"], out["center_z_z2"] = float(cx), float(cy), float(cz)
                except Exception:
                    out["center_x_z2"], out["center_y_z2"], out["center_z_z2"] = (np.nan, np.nan, np.nan)
                if comvel_z2_kms_sel is not None:
                    try:
                        vx,vy,vz = comvel_z2_kms_sel[idx_i]
                        out["v_x_z2_kms"], out["v_y_z2_kms"], out["v_z_z2_kms"] = float(vx), float(vy), float(vz)
                    except Exception:
                        out["v_x_z2_kms"], out["v_y_z2_kms"], out["v_z_z2_kms"] = (np.nan, np.nan, np.nan)

                found_z2_rows.append(out)

            # any tracks still missing in tracks_remaining -> placeholders
            for missing_tr in list(tracks_remaining):
                found_z2_rows.append({
                    "track_id": int(missing_tr),
                    "z2_snapshot": Z2_SNAP,
                    "halo_index_z2": np.nan,
                    "is_central_z2": None,
                    "m_z2": np.nan, "r50_z2_kpc": np.nan, "bh_z2": np.nan,
                    "center_x_z2": np.nan, "center_y_z2": np.nan, "center_z_z2": np.nan,
                    "v_x_z2_kms": np.nan, "v_y_z2_kms": np.nan, "v_z_z2_kms": np.nan
                })

    print(f"Finished efficient scanning z=2: built {len(found_z2_rows)} entries (including placeholders).")

    # now chunk-scan TrackId dataset and check for membership
    d_track = fh[ds_track_name]
    nrows = d_track.shape[0]
    print("z=2 TrackId table length:", nrows)
    tracks_remaining = set(tracks_to_find)

    # if we have full arrays of ExclusiveSphere fields, we can map by array index; else record not-found or limited info
    for start in range(0, nrows, CHUNK):
        stop = min(start + CHUNK, nrows)
        tr_chunk = np.asarray(d_track[start:stop])
        # try cast to integer-like for comparison
        try:
            tr_chunk_f = tr_chunk.astype(np.int64)
        except Exception:
            # try float then int
            tr_chunk_f = np.asarray(tr_chunk, dtype=float)
        # get indexes present
        # use numpy intersection test
        # we build mask of any tracks that are in our set
        mask_in = np.isin(tr_chunk_f, list(tracks_remaining))
        if not np.any(mask_in):
            continue
        rel_idxs = np.nonzero(mask_in)[0]
        abs_idxs = rel_idxs + start

        # read halo idx and iscentral if present
        halo_vals = None
        iscen_vals = None
        if ds_halo_name is not None:
            try:
                halo_ds = fh[ds_halo_name]
                halo_vals = np.asarray(halo_ds[abs_idxs])
            except Exception:
                halo_vals = np.full(len(abs_idxs), np.nan)
        else:
            halo_vals = np.full(len(abs_idxs), np.nan)
        if ds_iscen_name is not None:
            try:
                iscen_ds = fh[ds_iscen_name]
                iscen_vals = np.asarray(iscen_ds[abs_idxs])
            except Exception:
                iscen_vals = np.full(len(abs_idxs), np.nan)
        else:
            iscen_vals = np.full(len(abs_idxs), np.nan)

        # gather galaxy fields from helper arrays if available
        for i_local, ai in enumerate(abs_idxs):
            trval = int(tr_chunk_f[rel_idxs[i_local]])
            haloval = int(halo_vals[i_local]) if np.isfinite(halo_vals[i_local]) else np.nan
            iscen_raw = iscen_vals[i_local]
            try:
                if isinstance(iscen_raw, (np.bool_, bool)):
                    iscen = bool(iscen_raw)
                else:
                    iscen = bool(int(iscen_raw)) if np.isfinite(iscen_raw) else None
            except Exception:
                iscen = None

            out = {
                "track_id": int(trval),
                "z2_snapshot": Z2_SNAP,
                "halo_index_z2": int(haloval) if not np.isnan(haloval) else np.nan,
                "is_central_z2": iscen
            }

            if have_gal_z2:
                # ai is the absolute row index into the SOAP arrays; use to index ExclusiveSphere arrays
                try:
                    out["m_z2"] = float(m_z2_all[ai])
                except Exception:
                    out["m_z2"] = np.nan
                try:
                    out["r50_z2_kpc"] = float(r50_z2_all[ai])
                except Exception:
                    out["r50_z2_kpc"] = np.nan
                try:
                    out["bh_z2"] = float(bh_z2_all[ai])
                except Exception:
                    out["bh_z2"] = np.nan
                try:
                    out["center_x_z2"], out["center_y_z2"], out["center_z_z2"] = tuple(np.asarray(centers_z2_all[ai]))
                except Exception:
                    out["center_x_z2"], out["center_y_z2"], out["center_z_z2"] = (np.nan, np.nan, np.nan)
                if comvel_z2_kms is not None:
                    try:
                        vx, vy, vz = comvel_z2_kms[ai]
                        out["v_x_z2_kms"], out["v_y_z2_kms"], out["v_z_z2_kms"] = float(vx), float(vy), float(vz)
                    except Exception:
                        out["v_x_z2_kms"], out["v_y_z2_kms"], out["v_z_z2_kms"] = (np.nan, np.nan, np.nan)
            found_z2_rows.append(out)
            # remove track from remaining set
            if trval in tracks_remaining:
                tracks_remaining.remove(trval)

        # early break if none left
        if not tracks_remaining:
            break

    # for any tracks still not found in the entire file, record not-found rows
    for missing_tr in list(tracks_remaining):
        found_z2_rows.append({
            "track_id": int(missing_tr),
            "z2_snapshot": Z2_SNAP,
            "halo_index_z2": np.nan,
            "is_central_z2": None,
            "m_z2": np.nan, "r50_z2_kpc": np.nan, "bh_z2": np.nan,
            "center_x_z2": np.nan, "center_y_z2": np.nan, "center_z_z2": np.nan,
            "v_x_z2_kms": np.nan, "v_y_z2_kms": np.nan, "v_z_z2_kms": np.nan
        })

print(f"Finished scanning z=2: found entries for {len(found_z2_rows)} tracks (including not-found placeholders).")

# ------------------------- Build summary table combining z0 & z2 info -------------------------
# Build map track_id -> z2 row
map_z2 = {int(r["track_id"]): r for r in found_z2_rows}

rows_out = []
# iterate over extremes at z0 and assemble summary
for i_local in np.where(mask_extreme)[0]:
    # index into full SOAP arrays
    abs_idx = sel_global_idx[i_local]
    tr = int(z0_track_matched[i_local]) if np.isfinite(z0_track_matched[i_local]) else None
    if tr is None:
        continue
    z2info = map_z2.get(tr, None)
    row = {
        "track_id": int(tr),
        "halo_index_z0": int(z0_haloidx_matched[i_local]) if not np.isnan(z0_haloidx_matched[i_local]) else np.nan,
        "is_central_z0": bool(z0_iscentral[i_local]),
        "m_z0": float(z0_mass[i_local]),
        "r50_z0_kpc": float(z0_r50[i_local]),
        "bh_z0": float(z0_bh[i_local]) if np.isfinite(z0_bh[i_local]) else np.nan,
        "center_x_z0": float(z0_center[i_local,0]) if (np.isfinite(z0_center[i_local]).all()) else np.nan,
        "center_y_z0": float(z0_center[i_local,1]) if (np.isfinite(z0_center[i_local]).all()) else np.nan,
        "center_z_z0": float(z0_center[i_local,2]) if (np.isfinite(z0_center[i_local]).all()) else np.nan,
    }
    if comvel0_kms is not None:
        try:
            vx,vy,vz = comvel0_kms[abs_idx]
            row["v_x_z0_kms"], row["v_y_z0_kms"], row["v_z_z0_kms"] = float(vx), float(vy), float(vz)
        except Exception:
            row["v_x_z0_kms"], row["v_y_z0_kms"], row["v_z_z0_kms"] = (np.nan, np.nan, np.nan)

    if z2info is None:
        # not found placeholder
        row.update({
            "halo_index_z2": np.nan, "is_central_z2": None,
            "m_z2": np.nan, "r50_z2_kpc": np.nan, "bh_z2": np.nan,
            "center_x_z2": np.nan, "center_y_z2": np.nan, "center_z_z2": np.nan,
            "v_x_z2_kms": np.nan, "v_y_z2_kms": np.nan, "v_z_z2_kms": np.nan
        })
    else:
        # copy fields
        row["halo_index_z2"] = z2info.get("halo_index_z2", np.nan)
        row["is_central_z2"] = z2info.get("is_central_z2", None)
        row["m_z2"] = z2info.get("m_z2", np.nan)
        row["r50_z2_kpc"] = z2info.get("r50_z2_kpc", np.nan)
        row["bh_z2"] = z2info.get("bh_z2", np.nan)
        row["center_x_z2"] = z2info.get("center_x_z2", np.nan)
        row["center_y_z2"] = z2info.get("center_y_z2", np.nan)
        row["center_z_z2"] = z2info.get("center_z_z2", np.nan)
        row["v_x_z2_kms"] = z2info.get("v_x_z2_kms", np.nan)
        row["v_y_z2_kms"] = z2info.get("v_y_z2_kms", np.nan)
        row["v_z_z2_kms"] = z2info.get("v_z_z2_kms", np.nan)

    # compute BH ratio log columns if possible (z0/z2)
    try:
        row["log10_mstar_z0"] = float(np.log10(row["m_z0"])) if np.isfinite(row["m_z0"]) and row["m_z0"]>0 else np.nan
    except Exception:
        row["log10_mstar_z0"] = np.nan
    try:
        row["log10_mstar_z2"] = float(np.log10(row["m_z2"])) if np.isfinite(row["m_z2"]) and row["m_z2"]>0 else np.nan
    except Exception:
        row["log10_mstar_z2"] = np.nan

    try:
        row["log10_bh_ratio_z0"] = float(np.log10(row["bh_z0"]/row["m_z0"])) if np.isfinite(row["bh_z0"]) and np.isfinite(row["m_z0"]) and row["bh_z0"]>0 and row["m_z0"]>0 else np.nan
    except Exception:
        row["log10_bh_ratio_z0"] = np.nan
    try:
        row["log10_bh_ratio_z2"] = float(np.log10(row["bh_z2"]/row["m_z2"])) if np.isfinite(row["bh_z2"]) and np.isfinite(row["m_z2"]) and row["bh_z2"]>0 and row["m_z2"]>0 else np.nan
    except Exception:
        row["log10_bh_ratio_z2"] = np.nan

    rows_out.append(row)

# write CSV
out_csv = os.path.join(OUTDIR, "extreme_relics_z0_to_z2_summary.csv")
df_out = pd.DataFrame(rows_out)
df_out.to_csv(out_csv, index=False)
print("Wrote summary CSV:", out_csv)
print("Done.")

# # ==============================================================
# #               Z=2 ANALYSIS PLOTS (EXTREME RELICS)
# # ==============================================================

# import matplotlib.pyplot as plt

# print("\nGenerating z=2 analysis plots...")

# df = df_out.copy()

# # Only objects successfully found at z=2
# df = df[np.isfinite(df["m_z2"])].copy()

# if len(df) == 0:
#     print("No objects with valid z=2 data — skipping plots.")
#     sys.exit(0)

# # -----------------------------
# # 1) BH MASS RATIO PLOT (z=2)
# # -----------------------------
# with np.errstate(divide='ignore', invalid='ignore'):
#     logM_z2 = np.log10(df["m_z2"].to_numpy())
#     log_bh_ratio_z2 = df["log10_bh_ratio_z2"].to_numpy()

# mask = np.isfinite(logM_z2) & np.isfinite(log_bh_ratio_z2)

# plt.figure(figsize=(7,5))
# plt.scatter(logM_z2[mask], log_bh_ratio_z2[mask],
#             marker='*', s=120, edgecolor='k', facecolor='C1')
# plt.xlabel(r"$\log_{10}(M_\star / M_\odot)$")
# plt.ylabel(r"$\log_{10}(M_{\rm BH} / M_\star)$")
# plt.title("Extreme relics at z≈2")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(OUTDIR, "z2_BH_ratio_extremes.png"), dpi=200)
# plt.close()

# print("Saved BH ratio plot (z=2).")

# # -----------------------------
# # 2) HOST MASS HISTOGRAM (z=2)
# # -----------------------------
# if "host_mass" in df.columns:
#     hm = df["host_mass"].to_numpy()
#     mask = np.isfinite(hm) & (hm > 0)

#     if mask.sum() > 0:
#         plt.figure(figsize=(6,4))
#         plt.hist(np.log10(hm[mask]), bins=12, edgecolor='k')
#         plt.xlabel("log10(host mass) [Msun]")
#         plt.ylabel("N")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTDIR, "z2_host_mass_hist.png"), dpi=200)
#         plt.close()
#         print("Saved host mass histogram (z=2).")

# # -----------------------------
# # 3) DISTANCE / HOST RADIUS
# # -----------------------------
# if "dist_over_hostR" in df.columns:
#     x = df["host_mass"].to_numpy()
#     y = df["dist_over_hostR"].to_numpy()
#     mask = np.isfinite(x) & (x > 0) & np.isfinite(y)

#     if mask.sum() > 0:
#         plt.figure(figsize=(6,4))
#         plt.scatter(np.log10(x[mask]), y[mask], s=40)
#         plt.axhline(1.0, linestyle="--", color="k")
#         plt.xlabel("log10(host mass) [Msun]")
#         plt.ylabel("dist / host_radius")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTDIR, "z2_dist_over_hostR.png"), dpi=200)
#         plt.close()
#         print("Saved distance/host radius plot (z=2).")

# # -----------------------------
# # 4) RELATIVE VELOCITY HISTOGRAM
# # -----------------------------
# if "v_rel" in df.columns:
#     vr = df["v_rel"].to_numpy()
#     mask = np.isfinite(vr)

#     if mask.sum() > 0:
#         plt.figure(figsize=(6,4))
#         plt.hist(vr[mask], bins=25, edgecolor='k')
#         plt.xlabel("v_rel [km/s]")
#         plt.ylabel("N")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTDIR, "z2_v_rel_hist.png"), dpi=200)
#         plt.close()
#         print("Saved v_rel histogram (z=2).")

# # -----------------------------
# # 5) v_rel vs host mass
# # -----------------------------
# if "v_rel" in df.columns and "host_mass" in df.columns:
#     vr = df["v_rel"].to_numpy()
#     hm = df["host_mass"].to_numpy()
#     mask = np.isfinite(vr) & np.isfinite(hm) & (hm > 0)

#     if mask.sum() > 0:
#         plt.figure(figsize=(6,4))
#         plt.scatter(np.log10(hm[mask]), vr[mask], s=40)
#         plt.xlabel("log10(host mass) [Msun]")
#         plt.ylabel("v_rel [km/s]")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(OUTDIR, "z2_v_rel_vs_host_mass.png"), dpi=200)
#         plt.close()
#         print("Saved v_rel vs host mass (z=2).")

# print("All z=2 plots generated.")