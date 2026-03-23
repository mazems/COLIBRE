#!/usr/bin/env python3
"""
extreme_relic_satellite_hosts.py 

"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import common   # your helper; same import used in your pipeline
import h5py
import math

plt.rcParams.update({"mathtext.fontset":"stix", "font.family":"serif", "font.size":12})

# -------------------- CONFIG --------------------
csv_in = "sfh_times_all_with_DoR_variants_corrected.csv.gz"   # UCMG CSV (same as in your script)
model_name = 'L0200N3008/THERMAL_AGN/'
model_dir  = '/mnt/su3-pro/colibre/' + model_name
snap_file  = '0127'   # SOAP snapshot id used in your pipeline
outdir = "extreme_relic_satellite_hosts"
os.makedirs(outdir, exist_ok=True)

EXTREME_DOR = 0.6    # threshold for extreme relics (set as desired)
MIN_STELLAR_MASS = 1e9  # same initial selection as your scripts

# DoR column candidates (same order as in your original script)
dor_column_candidates = ["DoR_t95"]  # add more if you expect other names, e.g. ["DoR_t95","DoR_t998",...]

# -------------------- read CSV and build DoR lookup --------------------
if not os.path.exists(csv_in):
    raise SystemExit(f"CSV input not found: {csv_in}")
print("Reading CSV:", csv_in)
df_ucmg = pd.read_csv(csv_in, low_memory=False)

# choose canonical id column (same heuristic as in your script)
id_col = None
for c in ("subhalo_id", "HaloCatalogueIndex", "subhaloId", "HaloIndex", "track_id", "TrackId"):
    if c in df_ucmg.columns:
        id_col = c; break
if id_col is None:
    id_col = df_ucmg.columns[0]
    print("Warning: no id column found; using", id_col)

# normalize numeric ids -> 'subhalo_id' (int)
s = df_ucmg[id_col].astype(str).str.replace("\r", "").str.strip()
df_ucmg["_subhalo_id_numeric"] = pd.to_numeric(s, errors="coerce").astype("Int64")
n_bad = int(df_ucmg["_subhalo_id_numeric"].isna().sum())
if n_bad > 0:
    print(f"Warning: {n_bad} rows have non-numeric {id_col}; they will be ignored for matching.")
df_ucmg = df_ucmg[df_ucmg["_subhalo_id_numeric"].notna()].copy()
df_ucmg["subhalo_id"] = df_ucmg["_subhalo_id_numeric"].astype("int64")
df_ucmg.drop(columns=["_subhalo_id_numeric"], inplace=True)
df_ucmg_indexed = df_ucmg.set_index("subhalo_id", drop=False)

# find DoR column present
dor_cols_found = [c for c in dor_column_candidates if c in df_ucmg_indexed.columns]
if len(dor_cols_found) == 0:
    # fallback: any column starting with 'dor' (case-insensitive)
    for c in df_ucmg_indexed.columns:
        if c.lower().startswith("dor"):
            dor_cols_found.append(c)
if len(dor_cols_found) == 0:
    raise SystemExit("No DoR column found in CSV.")
primary_dor_col = dor_cols_found[0]
print("Using DoR column from CSV:", primary_dor_col)

# build dor_lookup dict: subhalo_id -> DoR (float)
dor_lookup = {}
for _, row in df_ucmg.iterrows():
    try:
        sid = int(row["subhalo_id"])
        v = row.get(primary_dor_col, np.nan)
        if pd.isna(v):
            continue
        dor_lookup[sid] = float(v)
    except Exception:
        continue
print("Loaded DoR entries from CSV:", len(dor_lookup))


# -------------------- read SOAP (galaxy fields + id/signpost fields + FOF arrays) --------------------
print("Reading SOAP arrays via common.read_group_data_colibre...")
fields_gal = {'ExclusiveSphere/50kpc': (
                'StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'CentreOfMass',
                'MassWeightedMeanStellarAge', 'LuminosityWeightedMeanStellarAge',
                'LinearMassWeightedIronOverHydrogenOfStars', 'LinearMassWeightedMagnesiumOverHydrogenOfStars', 'CentreOfMassVelocity'
             )}
fields_id = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')}
# request the FOF arrays under the same InputHalos group
fields_fof = {'InputHalos': ('FOF/Masses', 'FOF/Centres', 'FOF/Radii')}
soap_id = {'SOAP': ('HostHaloIndex',)}

h5data_groups = common.read_group_data_colibre(model_dir, snap_file, fields_gal)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_id)
h5data_soap = common.read_group_data_colibre(model_dir, snap_file, soap_id)

# try reading FOF arrays. helper may return them or not; attempt and validate.
try:
    h5data_fof = common.read_group_data_colibre(model_dir, snap_file, fields_fof)
except Exception as e:
    print("Warning: could not read FOF arrays via helper:", e)
    h5data_fof = None

# If helper didn't return FOF arrays, attempt to locate them inside the SOAP HDF5 file
if h5data_fof is None:
    # attempt to build path to SOAP file used by common.read_group_data_colibre above:
    # This fallback is best-effort — adjust SOAP_H5PATH to the real file if this fails.
    SOAP_H5PATH = None
    try:
        # try to infer common helper's path if available in environment (best-effort)
        # Otherwise user must set SOAP_H5PATH manually to the halo_properties_0127.hdf5 path.
        # Leave SOAP_H5PATH=None to force an explicit error (this avoids silent wrong behavior).
        if SOAP_H5PATH is None:
            raise FileNotFoundError("FOF arrays not returned by helper and SOAP_H5PATH not set.")
    except Exception:
        raise SystemExit("FOF arrays unavailable. If common.read_group_data_colibre can't find them, set SOAP_H5PATH in the script.")

# unpack galaxy + id arrays
(m30, sfr30, r50, centre_of_mass, stellarage, stellarage_lum, Fe_lin, Mg_lin, centre_of_mass_vel) = h5data_groups
(halo_index, is_central, desc_id, track_id) = h5data_idgroups
(host_halo_index) = h5data_soap

# unpack fof arrays (if returned as tuple from helper)
if h5data_fof is not None:
    try:
        fof_masses, fof_centres, fof_radii = h5data_fof
    except Exception:
        # maybe nested dict/structure — try to coerce to a sensible shape
        raise SystemExit("Unexpected structure from common.read_group_data_colibre(fields_fof). Please inspect returned object.")

# -------------------- units & selection --------------------
ztarget = 0.0
comov_to_physical_length = 1.0 / (1.0 + ztarget)
Mu = 1.988e43 / 1.989e33
tu = 3.086e19 / 3.154e7

m30 = np.asarray(m30).ravel() * Mu
sfr30 = np.asarray(sfr30).ravel() * Mu / tu
r50 = np.asarray(r50).ravel() * comov_to_physical_length * 1e3  # to kpc
centre_of_mass = np.asarray(centre_of_mass) * comov_to_physical_length * 1e3  # (N,3)
centre_of_mass_vel = np.asarray(centre_of_mass_vel)
# if centre_of_mass_vel.ndim == 1:
#     centre_of_mass_vel = centre_of_mass_vel.reshape((-1,3))
halo_index = np.asarray(halo_index).ravel()
is_central = np.asarray(is_central).ravel().astype(bool)

host_halo_index = np.asarray(host_halo_index).ravel()

# FOF guards: ensure shapes are sane
fof_masses = np.asarray(fof_masses) * Mu
fof_radii = np.asarray(fof_radii) * comov_to_physical_length * 1e3
fof_centres = np.asarray(fof_centres) * comov_to_physical_length * 1e3

# Some SOAP files store FOF arrays per-halo (N,) and centres as (N,3) — ensure this is the case:
if fof_masses.ndim != 1:
    fof_masses = fof_masses.ravel()
if fof_radii.ndim != 1:
    fof_radii = fof_radii.ravel()
if fof_centres.ndim == 1:
    # maybe flattened triples: try to reshape if length divisible by 3
    if fof_centres.size % 3 == 0:
        fof_centres = fof_centres.reshape((-1, 3))
    else:
        raise SystemExit("FOF centres shape unexpected; please check /InputHalos/FOF/Centres dataset shape.")

# after reading fof_masses, fof_radii, fof_centres (before any indexing)
print("fof_masses: n=", fof_masses.size,
      "min,max,median (raw)=", np.nanmin(fof_masses), np.nanmax(fof_masses), np.nanmedian(fof_masses))
print("fof_radii (raw): min,max,median =", np.nanmin(fof_radii), np.nanmax(fof_radii), np.nanmedian(fof_radii))
print("fof_centres shape:", fof_centres.shape)
n_zero_mass = np.sum((fof_masses == 0) | (~np.isfinite(fof_masses)))
n_zero_rad  = np.sum((fof_radii == 0)  | (~np.isfinite(fof_radii)))
print("fof zero mass:", n_zero_mass, "/", fof_masses.size, " ; zero radius:", n_zero_rad, "/", fof_radii.size)

# initial selection: stellar mass threshold & positive radius
sel = np.where(m30 >= MIN_STELLAR_MASS)[0]
if sel.size == 0:
    raise SystemExit("No galaxies meet MIN_STELLAR_MASS selection.")

# restrict arrays to selected indices (we will map DoR by halo_index later)
m = m30[sel]; r = r50[sel]; centers = centre_of_mass[sel]; center_velocities = centre_of_mass_vel[sel]
halo_idx = halo_index[sel].astype(np.int64)
is_central_sel = is_central[sel]
host_halo_idx = host_halo_index[sel].astype(np.int64)
track_sel = track_id[sel] if track_id is not None else None

print(f"Selected {sel.size} SOAP galaxies with m >= {MIN_STELLAR_MASS}.")

# -------------------- map DoR from CSV onto the selected SOAP rows --------------------
dor_series = pd.Series(dor_lookup, dtype=float)

# halo_idx may be 1-based in the CSV or SOAP; try direct reindex first, fallback to -1 shift later if necessary
halo_idx_for_lookup = halo_idx.copy()
host_halo_idx_for_lookup = host_halo_idx.copy()

dor_for_each_soap_row = dor_series.reindex(halo_idx_for_lookup).to_numpy(dtype=float)  # NaN where missing
matched_positions = np.where(np.isfinite(dor_for_each_soap_row))[0]
print(f"Matched DoR entries for selected SOAP rows: {matched_positions.size} / {halo_idx_for_lookup.size}")

# If none matched, try 1-based ↔ 0-based fallback:
if matched_positions.size == 0:
    # try shifting by +1 and -1 to detect off-by-one conventions
    possible_try = []
    try:
        dor_try = dor_series.reindex(halo_idx_for_lookup - 1).to_numpy(dtype=float)
        if np.any(np.isfinite(dor_try)):
            dor_for_each_soap_row = dor_try
            matched_positions = np.where(np.isfinite(dor_for_each_soap_row))[0]
            print(f"Matched after applying halo_idx-1 fallback: {matched_positions.size}")
    except Exception:
        pass
    if matched_positions.size == 0:
        try:
            dor_try = dor_series.reindex(halo_idx_for_lookup + 1).to_numpy(dtype=float)
            if np.any(np.isfinite(dor_try)):
                dor_for_each_soap_row = dor_try
                matched_positions = np.where(np.isfinite(dor_for_each_soap_row))[0]
                print(f"Matched after applying halo_idx+1 fallback: {matched_positions.size}")
        except Exception:
            pass

# aligned arrays for matched subset
m_matched = m[matched_positions]
r_matched = r[matched_positions]
centers_matched = centers[matched_positions]
host_halo_idx_matched = host_halo_idx_for_lookup[matched_positions].astype(np.int64)
is_central_matched = is_central_sel[matched_positions]
dor_matched = dor_for_each_soap_row[matched_positions]
track_matched = track_sel[matched_positions] if track_sel is not None else np.full(matched_positions.shape, np.nan)
v_gal_matched = center_velocities[matched_positions] 

# ---------- BEGIN: kinematic enrichment (insert here) ----------
# Assumptions: you already have arrays:
#   centres_matched    (N_matched,3)  - galaxy CoM positions in kpc (you scaled earlier)
#   host_halo_idx_matched (N_matched,) - host index (index into SOAP/subhalo arrays)
#   matched_positions  (indices into original selected arrays)
#   matched subset length = len(matched_positions)
#
# This block will:
#  - try to read per-subhalo CentreOfMassVelocity from SOAP
#  - try to read host sigma from an HDF5 in the 'extra' directory (safe fallbacks)
#  - compute v_rel and v_rel/sigma and add arrays that you can include in rows/df_out

# 1) MEMORY-SAFE: read only required sigma rows in contiguous slices ----------
extra_dir = os.path.join(model_dir, "SOAP-HBT", "extra")
sigma_ok = False
sigma_for_host = None

def contiguous_ranges(sorted_idx):
    """Return list of (start, stop) ranges (stop exclusive) for sorted unique indices."""
    if sorted_idx.size == 0:
        return []
    runs = []
    start = int(sorted_idx[0])
    prev = start
    for v in sorted_idx[1:]:
        iv = int(v)
        if iv == prev + 1:
            prev = iv
            continue
        runs.append((start, prev + 1))
        start = iv
        prev = iv
    runs.append((start, prev + 1))
    return runs

if not os.path.isdir(extra_dir):
    print("Extra dir not found:", extra_dir)
else:
    # choose the (only) HDF5 file in extra (if there are many, pick the first; change as needed)
    hdffiles = [os.path.join(extra_dir, f) for f in os.listdir(extra_dir) if f.endswith(".h5") or f.endswith(".hdf5")]
    if len(hdffiles) == 0:
        print("No HDF5 files in extra:", extra_dir)
    else:
        hdfpath = hdffiles[0]
        print("Using extra file:", hdfpath)
        try:
            with h5py.File(hdfpath, "r") as fh:
                # show top-level keys for debugging
                print("Top-level keys in extra file:", list(fh.keys())[:50])

                # common candidate names; extend if you know exact path
                candidates = [
                    "sigma", "Sigma", "VelocityDispersion", "FOF/Sigma", "FOF/VelocityDispersion",
                    "host_sigma", "halo_properties/sigma", "halo_properties/VelocityDispersion"
                ]

                ds_obj = None
                ds_name = None

                # try candidate paths
                for cand in candidates:
                    parts = cand.split("/")
                    cur = fh
                    ok = True
                    for p in parts:
                        if p in cur:
                            cur = cur[p]
                        else:
                            ok = False
                            break
                    if ok and isinstance(cur, h5py.Dataset):
                        ds_obj = cur
                        ds_name = cand
                        break

                # generic scan if still not found
                if ds_obj is None:
                    for k in fh.keys():
                        obj = fh[k]
                        if isinstance(obj, h5py.Dataset):
                            # prefer 1D float arrays
                            if obj.ndim == 1 and np.issubdtype(obj.dtype, np.floating):
                                ds_obj = obj
                                ds_name = k
                                break
                            # structured array: check fields
                            if hasattr(obj, "dtype") and obj.dtype.names is not None:
                                for nm in ("sigma", "velocity_dispersion", "vel_disp"):
                                    if nm in obj.dtype.names:
                                        ds_obj = obj
                                        ds_name = k
                                        break
                                if ds_obj is not None:
                                    break
                        elif isinstance(obj, h5py.Group):
                            # look inside group
                            for subk in obj.keys():
                                sub = obj[subk]
                                if isinstance(sub, h5py.Dataset) and sub.ndim == 1 and np.issubdtype(sub.dtype, np.floating):
                                    ds_obj = sub
                                    ds_name = f"{k}/{subk}"
                                    break
                            if ds_obj is not None:
                                break

                if ds_obj is None:
                    print("No obvious sigma dataset found in extra file. Inspect printed keys and add the correct path to `candidates`.")
                else:
                    print("Selected sigma dataset:", ds_name, " shape:", ds_obj.shape, " dtype:", ds_obj.dtype)

                    # host indices needed (these are indices into SOAP arrays)
                    host_idx = np.asarray(host_halo_idx_matched, dtype=int)
                    valid_mask = (host_idx >= 0) & (host_idx < ds_obj.shape[0])
                    if not np.any(valid_mask):
                        print("No valid host indices within dataset length; skipping sigma read.")
                        sigma_for_host = np.full(host_idx.shape, np.nan)
                        sigma_ok = False
                    else:
                        unique_idx = np.unique(host_idx[valid_mask])
                        unique_idx.sort()
                        print("Unique valid host indices needed:", unique_idx.size)

                        runs = contiguous_ranges(unique_idx)
                        print("Contiguous runs to read:", len(runs))

                        sigma_for_host = np.full(host_idx.shape, np.nan, dtype=float)

                        # tune this if memory still spikes
                        MAX_ROWS_PER_READ = 200_000

                        for (start, stop) in runs:
                            length = stop - start
                            if length <= MAX_ROWS_PER_READ:
                                vals = ds_obj[start:stop]   # read slice only
                                vals = np.asarray(vals, dtype=float)
                                # assign into sigma_for_host where host_idx in [start,stop)
                                mask_in = (host_idx >= start) & (host_idx < stop)
                                if np.any(mask_in):
                                    offsets = host_idx[mask_in] - start
                                    sigma_for_host[mask_in] = vals[offsets]
                            else:
                                # further chunk inside long run
                                sub = start
                                while sub < stop:
                                    sub_stop = min(sub + MAX_ROWS_PER_READ, stop)
                                    vals = ds_obj[sub:sub_stop]
                                    vals = np.asarray(vals, dtype=float)
                                    mask_in = (host_idx >= sub) & (host_idx < sub_stop)
                                    if np.any(mask_in):
                                        offsets = host_idx[mask_in] - sub
                                        sigma_for_host[mask_in] = vals[offsets]
                                    sub = sub_stop

                        sigma_ok = True
                        print("Done reading sigma for requested host indices.")

        except Exception as e:
            print("Exception while opening/reading extra HDF5 file:", e)
            sigma_ok = False
            sigma_for_host = np.full(len(host_halo_idx_matched), np.nan)

# If sigma_for_host still None, make a NaN array of correct length
if sigma_for_host is None:
    sigma_for_host = np.full(len(host_halo_idx_matched), np.nan)
    sigma_ok = False

print("sigma_ok =", sigma_ok, "- finite entries:", np.sum(np.isfinite(sigma_for_host)))

# ---------- 2) Build per-matched arrays for galaxy and host velocities, compute v_rel ----
Nmatched = len(matched_positions)

# ensure v_gal_matched exists and has shape (Nmatched,3)
v_gal = np.asarray(v_gal_matched, dtype=float)
if v_gal.ndim != 2 or v_gal.shape[1] != 3 or v_gal.shape[0] != Nmatched:
    raise RuntimeError("v_gal_matched has unexpected shape; expected (Nmatched,3).")

# prepare host velocities by indexing the FULL centre_of_mass_vel table
full_vel = np.asarray(centre_of_mass_vel)   # full-table (Nfull,3) created earlier
if full_vel.ndim != 2 or full_vel.shape[1] != 3:
    raise RuntimeError("centre_of_mass_vel (full) has unexpected shape; expected (Nfull,3).")
Nfull = full_vel.shape[0]

host_idx = host_halo_idx_matched.astype(int)   # indices into full table (expected)
v_host = np.full((Nmatched, 3), np.nan, dtype=float)

# valid direct indexing where host_idx is in range
valid = (host_idx >= 0) & (host_idx < Nfull)
if np.any(valid):
    v_host[valid] = full_vel[host_idx[valid]]

# optional tiny fallback for a few OOB host indices: try ±1 if that looks sensible
bad = np.where(~valid)[0]
if bad.size > 0:
    # try host_idx-1
    use_minus1 = []
    use_plus1  = []
    for i in bad:
        ih = host_idx[i]
        if (ih - 1) >= 0 and (ih - 1) < Nfull:
            use_minus1.append(i)
        elif (ih + 1) >= 0 and (ih + 1) < Nfull:
            use_plus1.append(i)
    if len(use_minus1):
        idxs = np.array(use_minus1, dtype=int)
        v_host[idxs] = full_vel[ host_idx[idxs] - 1 ]
        print(f"Applied host_idx-1 fallback for {len(use_minus1)} entries.")
    if len(use_plus1):
        idxs = np.array(use_plus1, dtype=int)
        v_host[idxs] = full_vel[ host_idx[idxs] + 1 ]
        print(f"Applied host_idx+1 fallback for {len(use_plus1)} entries.")

# compute v_rel (NaNs will propagate where v_host is missing)
diff = v_gal - v_host
v_rel = np.linalg.norm(diff, axis=1)

# --- ensure v_rel_over_sigma exists (avoid NameError later) ---
Nmatched = len(matched_positions)
v_rel_over_sigma = np.full(Nmatched, np.nan, dtype=float)

# If you read per-host sigma into `sigma_for_host` above, compute v_rel_over_sigma now.
# Use sigma_for_host (from the extra HDF5 read) — do NOT use `host_sigma_all` which may not exist.
try:
    if 'sigma_for_host' in globals() and np.any(np.isfinite(sigma_for_host)):
        sigma_arr = np.asarray(sigma_for_host, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            v_rel_over_sigma = v_rel / (sigma_arr + 1e-12)
    else:
        # nothing to compute; keep NaNs
        pass
except Exception as e:
    # keep NaNs but log the error for debugging
    print("Warning while computing v_rel_over_sigma:", e)

# optionally compute v_rel normalized by host speed (avoid divide-by-zero)
v_host_mag = np.linalg.norm(v_host, axis=1)
with np.errstate(divide='ignore', invalid='ignore'):
    v_rel_over_vhost = v_rel / (v_host_mag + 1e-12)

# prepare sigma and v_rel_over_sigma elsewhere as before (unchanged)

# 3) if sigma available, align it and compute v_rel_over_sigma
if 'sigma_for_host' in globals() and np.any(np.isfinite(sigma_for_host)):
    sigma_for_host = np.asarray(sigma_for_host, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        v_rel_over_sigma = v_rel / (sigma_for_host + 1e-12)
else:
    sigma_for_host = np.full(Nmatched, np.nan)

# 4) print quick diagnostics and attach results to rows later (or to df_out)
print("Kinematics diagnostics (extreme matched subset):")
if sigma_ok:
    print(" v_rel_over_sigma: median,16,84:",
          np.nanmedian(v_rel_over_sigma[np.isfinite(v_rel_over_sigma)]),
          np.nanpercentile(v_rel_over_sigma[np.isfinite(v_rel_over_sigma)],16) if np.any(np.isfinite(v_rel_over_sigma)) else np.nan,
          np.nanpercentile(v_rel_over_sigma[np.isfinite(v_rel_over_sigma)],84) if np.any(np.isfinite(v_rel_over_sigma)) else np.nan)
else:
    print(" host sigma not available; v_rel_over_sigma not computed.")

print("v_rel_over_sigma: finite count =", np.sum(np.isfinite(v_rel_over_sigma)))
print("v_rel: finite count =", np.sum(np.isfinite(v_rel)))

# ---------- END: kinematic enrichment ----------

# -------------------- select extreme relics (both centrals and satellites) --------------------
# Build a single mask of all extreme relics (DoR > EXTREME_DOR) and iterate once.
extreme_mask = (dor_matched > EXTREME_DOR) & np.isfinite(dor_matched)
n_extreme = int(np.sum(extreme_mask))
print(f"Extreme relics (DoR>{EXTREME_DOR}) total (centrals+satellites): {n_extreme}")
if n_extreme == 0:
    print("No extreme relics found. Exiting.")
    sys.exit(0)

# -------------------- host mapping and distance calculations (single pass) --------------------
rows = []
n_skipped_fof_oob = 0
n_skipped_badpos = 0
for i_local in np.where(extreme_mask)[0]:
    # halo index for this matched SOAP row (host_halo_idx_matched holds HostHaloIndex)
    hid = int(host_halo_idx_matched[i_local])

    # guard: check hid in bounds for fof arrays, with 1-based ↔ 0-based fallback
    if (hid < 0) or (hid >= fof_masses.size):
        hid_try = hid - 1
        if (hid_try >= 0) and (hid_try < fof_masses.size):
            hid = hid_try
        else:
            hid_try2 = hid + 1
            if (hid_try2 >= 0) and (hid_try2 < fof_masses.size):
                hid = hid_try2
            else:
                n_skipped_fof_oob += 1
                print(f"Warning: host halo index {host_halo_idx_matched[i_local]} out of bounds for FOF arrays (size {fof_masses.size}); skipping.")
                continue

    # read host FOF values, with safe conversions
    try:
        host_mass = float(fof_masses[hid])
    except Exception:
        host_mass = np.nan
    try:
        host_center = np.asarray(fof_centres[hid], dtype=float)
        if host_center.size != 3:
            raise ValueError("host centre not length 3")
    except Exception:
        host_center = np.array([np.nan, np.nan, np.nan])
    try:
        host_radius = float(fof_radii[hid])
    except Exception:
        host_radius = np.nan

    gal_center = np.asarray(centers_matched[i_local], dtype=float)
    if gal_center.size != 3 or not np.all(np.isfinite(gal_center)):
        n_skipped_badpos += 1
        print(f"Warning: galaxy center invalid for matched_index {matched_positions[i_local]}; skipping.")
        continue

    gal_r50 = float(r_matched[i_local]) if np.isfinite(r_matched[i_local]) else np.nan
    dor_val = float(dor_matched[i_local])
    gal_stellar_mass = float(m_matched[i_local]) if np.isfinite(m_matched[i_local]) else np.nan
    is_cen = bool(is_central_matched[i_local])

    # 3D distance (kpc)
    vec = gal_center - host_center
    if not np.all(np.isfinite(vec)):
        print(f"Warning: non-finite vector between galaxy and host for matched_index {matched_positions[i_local]}; skipping.")
        continue
    dist_kpc = float(np.linalg.norm(vec))
    dist_over_hostR = float(dist_kpc / host_radius) if (host_radius > 0 and np.isfinite(host_radius)) else np.nan
    dist_over_galR = float(dist_kpc / gal_r50) if (gal_r50 > 0 and np.isfinite(gal_r50)) else np.nan

    rows.append({
        "matched_index": int(matched_positions[i_local]),
        "host_halo_index": int(host_halo_idx_matched[i_local]),
        "track_id": int(track_matched[i_local]) if (not pd.isna(track_matched[i_local])) else -1,
        "DoR": dor_val,
        "is_central_flag": is_cen,
        "gal_stellar_mass": gal_stellar_mass,
        "gal_r50_kpc": gal_r50,
        "host_mass": host_mass,
        "host_radius_kpc": host_radius,
        "gal_center_x": gal_center[0], "gal_center_y": gal_center[1], "gal_center_z": gal_center[2],
        "host_center_x": host_center[0], "host_center_y": host_center[1], "host_center_z": host_center[2],
        "dist_to_host_kpc": dist_kpc,
        "dist_over_hostR": dist_over_hostR,
        "dist_over_galR50": dist_over_galR
    })

# quick skips summary
if n_skipped_fof_oob > 0:
    print(f"Skipped {n_skipped_fof_oob} extreme candidates due to FOF index out-of-bounds.")
if n_skipped_badpos > 0:
    print(f"Skipped {n_skipped_badpos} extreme candidates due to invalid galaxy/host positions.")

# -------------------- save CSV (single dataframe containing both centrals & satellites) --------------------
df_out = pd.DataFrame(rows)
csv_out = os.path.join(outdir, "extreme_relics_hosts_all.csv")
df_out.to_csv(csv_out, index=False)
print("Wrote per-object CSV (centrals+satellites):", csv_out)

# -------------------- MERGE KINEMATICS INTO df_out and PLOT --------------------
# Assumes you already computed:
#   matched_positions (length Nmatched, dtype int)
#   v_gal (Nmatched,3), v_host (Nmatched,3), v_rel (Nmatched,), v_rel_over_sigma (Nmatched,), sigma_for_host (Nmatched,)
# If any of those are missing, the arrays below will be filled with NaN.

# Build df_kin keyed by matched_index (which is the index you stored in rows['matched_index'])
try:
    Nmatched = len(matched_positions)
except NameError:
    # nothing to merge
    print("No matched_positions visible: skipping kinematics merge/plots.")
else:
    df_kin = pd.DataFrame({
        "matched_index": matched_positions.astype(int),
        "v_gal_x": np.asarray(v_gal)[:, 0] if (v_gal is not None and np.asarray(v_gal).ndim == 2) else np.nan,
        "v_gal_y": np.asarray(v_gal)[:, 1] if (v_gal is not None and np.asarray(v_gal).ndim == 2) else np.nan,
        "v_gal_z": np.asarray(v_gal)[:, 2] if (v_gal is not None and np.asarray(v_gal).ndim == 2) else np.nan,
        "v_host_x": np.asarray(v_host)[:, 0] if (v_host is not None and np.asarray(v_host).ndim == 2) else np.nan,
        "v_host_y": np.asarray(v_host)[:, 1] if (v_host is not None and np.asarray(v_host).ndim == 2) else np.nan,
        "v_host_z": np.asarray(v_host)[:, 2] if (v_host is not None and np.asarray(v_host).ndim == 2) else np.nan,
        "v_rel": np.asarray(v_rel, dtype=float),
        "sigma_for_host": np.asarray(sigma_for_host, dtype=float),
        "v_rel_over_sigma": np.asarray(v_rel_over_sigma, dtype=float)
    })

    # Merge into df_out on matched_index (left join keeps df_out order)
    df_out = df_out.merge(df_kin, on="matched_index", how="left")

    # re-save CSV (includes kinematics)
    csv_out = os.path.join(outdir, "extreme_relics_hosts_all_with_kinematics.csv")
    df_out.to_csv(csv_out, index=False)
    print("Wrote per-object CSV with kinematics:", csv_out)

    # -------------------- Useful diagnostic plots --------------------
    # 1) Histogram of v_rel (all; satellites vs centrals)
    plt.figure(figsize=(6,4))
    v_rel_all = df_out["v_rel"].to_numpy(dtype=float)
    mask_rel = np.isfinite(v_rel_all)
    if mask_rel.sum() > 0:
        # split by central flag if available
        if "is_central_flag" in df_out.columns:
            sat_mask_plot = (df_out["is_central_flag"] == False) & np.isfinite(df_out["v_rel"])
            cen_mask_plot = (df_out["is_central_flag"] == True)  & np.isfinite(df_out["v_rel"])
            if sat_mask_plot.sum() > 0:
                plt.hist(v_rel_all[sat_mask_plot.values], bins=30, alpha=0.6, edgecolor='k') #, label=f"satellites (N={int(sat_mask_plot.sum())})"
            # if cen_mask_plot.sum() > 0:
            #     plt.hist(v_rel_all[cen_mask_plot.values], bins=30, alpha=0.6, label=f"centrals (N={int(cen_mask_plot.sum())})", edgecolor='k', color='C3')
        else:
            plt.hist(v_rel_all[mask_rel], bins=30, alpha=0.7, edgecolor='k', label=f"all (N={int(mask_rel.sum())})")
        plt.xlabel("v_rel [km/s]")
        plt.ylabel("N")
        # plt.title("Relative speed between galaxy and host")
        plt.legend()
        plt.grid(True, alpha=0.3)
        p_vrel = os.path.join(outdir, "hist_v_rel.png")
        plt.tight_layout(); plt.savefig(p_vrel, dpi=200); plt.close()
        print("Saved v_rel histogram:", p_vrel)
    else:
        print("No finite v_rel values to plot.")

    # 2) Histogram of v_rel_over_sigma (if sigma present)
    if np.any(np.isfinite(df_out["v_rel_over_sigma"].to_numpy(dtype=float))):
        arr = df_out["v_rel_over_sigma"].to_numpy(dtype=float)
        mask = np.isfinite(arr)
        plt.figure(figsize=(6,4))
        plt.hist(arr[mask], bins=30, alpha=0.7, edgecolor='k')
        plt.xlabel("v_rel / sigma_host")
        plt.ylabel("N")
        plt.title("Relative speed normalized by host sigma")
        plt.grid(True, alpha=0.3)
        p_vsigma = os.path.join(outdir, "hist_v_rel_over_sigma.png")
        plt.tight_layout(); plt.savefig(p_vsigma, dpi=200); plt.close()
        print("Saved v_rel/sigma histogram:", p_vsigma)
    else:
        print("v_rel_over_sigma not available or all NaN; skipping histogram.")

    # 3) Scatter: v_rel vs log10(host_mass)
    if ("host_mass" in df_out.columns) and np.any(np.isfinite(df_out["v_rel"].to_numpy(dtype=float))):
        hm = df_out["host_mass"].to_numpy(dtype=float)
        vr = df_out["v_rel"].to_numpy(dtype=float)
        ok = np.isfinite(hm) & (hm > 0) & np.isfinite(vr)
        if ok.sum() > 0:
            plt.figure(figsize=(6,4))
            plt.scatter(np.log10(hm[ok]), vr[ok], s=30, alpha=0.8)
            plt.xlabel("log10(host mass) [Msun]")
            plt.ylabel("v_rel")
            plt.title("Relative speed vs host mass")
            plt.grid(True, alpha=0.35)
            plt.tight_layout()
            p_vrel_vs_mass = os.path.join(outdir, "v_rel_vs_host_mass.png")
            plt.savefig(p_vrel_vs_mass, dpi=200); plt.close()
            print("Saved scatter v_rel vs host_mass:", p_vrel_vs_mass)

    # 4) Scatter: v_rel vs dist_over_hostR (are faster satellites closer/infalling?)
    if ("dist_over_hostR" in df_out.columns) and np.any(np.isfinite(df_out["v_rel"].to_numpy(dtype=float))):
        d = df_out["dist_over_hostR"].to_numpy(dtype=float)
        vr = df_out["v_rel"].to_numpy(dtype=float)
        ok = np.isfinite(d) & np.isfinite(vr)
        if ok.sum() > 0:
            plt.figure(figsize=(6,4))
            plt.scatter(d[ok], vr[ok], s=30, alpha=0.8)
            plt.xlabel("dist / host_radius")
            plt.ylabel("v_rel")
            plt.title("Relative speed vs distance (in host radii)")
            plt.grid(True, alpha=0.35)
            plt.tight_layout()
            p_vrel_vs_dist = os.path.join(outdir, "v_rel_vs_dist_over_hostR.png")
            plt.savefig(p_vrel_vs_dist, dpi=200); plt.close()
            print("Saved scatter v_rel vs dist_over_hostR:", p_vrel_vs_dist)

    # 5) Optional: host sigma vs host mass (if sigma available)
    if ("sigma_for_host" in df_out.columns) and np.any(np.isfinite(df_out["sigma_for_host"].to_numpy(dtype=float))):
        sigma = df_out["sigma_for_host"].to_numpy(dtype=float)
        hm = df_out["host_mass"].to_numpy(dtype=float)
        ok = np.isfinite(sigma) & np.isfinite(hm) & (hm > 0)
        if ok.sum() > 0:
            plt.figure(figsize=(6,4))
            plt.scatter(np.log10(hm[ok]), sigma[ok], s=30, alpha=0.8)
            plt.xlabel("log10(host mass) [Msun]")
            plt.ylabel("host sigma (same units as input)")
            plt.title("Host velocity dispersion vs host mass")
            plt.grid(True, alpha=0.35)
            plt.tight_layout()
            p_sigma_vs_mass = os.path.join(outdir, "sigma_vs_host_mass.png")
            plt.savefig(p_sigma_vs_mass, dpi=200); plt.close()
            print("Saved sigma vs host_mass:", p_sigma_vs_mass)

# -------------------- split dataframe for plotting/comparison --------------------
df_sat = df_out.loc[df_out["is_central_flag"] == False].copy()
df_cen = df_out.loc[df_out["is_central_flag"] == True].copy()
print(f"Saved split: {len(df_sat)} extreme satellites, {len(df_cen)} extreme centrals (from total {len(df_out)})")

# -------------------- SAFE PLOTTING --------------------
# 1) overlayed hist: satellites (default colour) + centrals (red)
host_col = "host_mass"
if host_col in df_out.columns and df_out.shape[0] > 0:
    sat_hm = df_sat[host_col].to_numpy(dtype=float)
    cen_hm = df_cen[host_col].to_numpy(dtype=float)

    sat_mask = np.isfinite(sat_hm) & (sat_hm > 0.0)
    cen_mask = np.isfinite(cen_hm) & (cen_hm > 0.0)
    if sat_mask.sum() + cen_mask.sum() > 0:
        bins = 12
        plt.figure(figsize=(6,4))
        if sat_mask.sum() > 0:
            plt.hist(np.log10(sat_hm[sat_mask]), bins=bins, alpha=0.6, edgecolor='k', label=f"satellites (N={sat_mask.sum()})")
        if cen_mask.sum() > 0:
            plt.hist(np.log10(cen_hm[cen_mask]), bins=bins, alpha=0.6, edgecolor='k', color='red', label=f"centrals (N={cen_mask.sum()})")
        plt.xlabel("lg(host mass) [M⊙]")
        plt.ylabel("N")
        # plt.title("Host mass distribution (extreme relics)")
        plt.legend()
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        p_overlay = os.path.join(outdir, "hist_host_mass_extreme_satellites_vs_centrals.png")
        plt.savefig(p_overlay, dpi=200); plt.close()
        print("Saved overlaid histogram:", p_overlay)

# 2) scatter: dist/hostR vs host mass (safe masking)
xcol = "host_mass"
ycol = "dist_over_hostR"
if (xcol in df_out.columns) and (ycol in df_out.columns) and (df_out.shape[0] > 0):
    x = df_out[xcol].to_numpy(dtype=float)
    y = df_out[ycol].to_numpy(dtype=float)
    ok = np.isfinite(x) & (x > 0.0) & np.isfinite(y)
    if np.sum(ok) == 0:
        print("No finite positive host_mass + finite dist_over_hostR pairs to plot; skipping scatter.")
    else:
        plt.figure(figsize=(6,4))
        sc = plt.scatter(np.log10(x[ok]), y[ok], s=40)
        plt.axhline(1.0, linestyle="--", color="k", alpha=0.6, label="R_host")
        plt.xlabel("log10(host mass) [Msun]")
        plt.ylabel("dist / host_radius")
        plt.title("Satellite distance (in host radii) vs host mass")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        p2 = os.path.join(outdir, "dist_over_hostR_vs_host_mass.png")
        plt.savefig(p2, dpi=200); plt.close()
        print("Saved plot:", p2)
else:
    print("Columns for dist/hostR vs host_mass scatter not present; skipping this plot.")

# summary stats printed
print("\nSummary statistics (extreme relic satellites):")
if df_out.shape[0] > 0:
    print("N objects written:", df_out.shape[0])
    # median host mass (log10) with safe filtering
    if host_col in df_out.columns and np.any(np.isfinite(df_out[host_col].to_numpy(dtype=float)) & (df_out[host_col].to_numpy(dtype=float) > 0)):
        med_log_host = float(np.nanmedian(np.log10(df_out.loc[np.isfinite(df_out[host_col]) & (df_out[host_col] > 0), host_col])))
        print("Host mass median (log10):", med_log_host)
    else:
        print("Host mass median (log10): n/a (no positive finite host_mass).")
    if "dist_over_hostR" in df_out.columns and np.any(np.isfinite(df_out["dist_over_hostR"].to_numpy(dtype=float))):
        print("dist/hostR median:", float(np.nanmedian(df_out["dist_over_hostR"].to_numpy(dtype=float))))
    else:
        print("dist/hostR median: n/a")
    if "dist_over_galR50" in df_out.columns and np.any(np.isfinite(df_out["dist_over_galR50"].to_numpy(dtype=float))):
        print("dist/r50 median:", float(np.nanmedian(df_out["dist_over_galR50"].to_numpy(dtype=float))))
    else:
        print("dist/r50 median: n/a")
print("Done. Inspect CSV and PNGs in:", outdir)


# For e.g. first 20 extreme satellites you found (assuming df_out exists)
hm = df_out["host_radius_kpc"].to_numpy()
gr = df_out["gal_r50_kpc"].to_numpy()
print("host_radius_kpc: min,median,max:", np.nanmin(hm), np.nanmedian(hm), np.nanmax(hm))
print("gal_r50_kpc: min,median,max:", np.nanmin(gr), np.nanmedian(gr), np.nanmax(gr))
# pick 5 example rows from df_out
for i in df_out.index[:5]:
    hid = int(df_out.loc[i, "host_halo_index"])   # or 'halo_index' depending on your CSV
    print("row", i, "host_idx", hid,
          "fof_masses[hid]=", fof_masses[hid] if 0 <= hid < fof_masses.size else "OOB",
          "fof_radii[hid]=", fof_radii[hid] if 0 <= hid < fof_radii.size else "OOB")

# compare a few euclidean separations before/after scaling
gal = centers_matched[0]        # your galaxy centre (already converted to kpc in script)
host = fof_centres[host_halo_idx_matched[0]]  # whichever centre you use
print("gal centre:", gal, "host centre:", host)
print("raw distance:", np.linalg.norm(gal - host))


# --- quick diagnostics to detect misalignment ---
print("comvel_all shape:", None if 'comvel_all' not in globals() else np.asarray(comvel_all).shape)
print("halo_idx shape (selected):", halo_idx.shape)
print("matched_positions shape:", matched_positions.shape)
print("example halo_idx[matched_positions][:10]:", halo_idx[matched_positions][:10])
print("example matched_positions[:10]:", matched_positions[:10])
print("example host_halo_idx_matched[:10]:", host_halo_idx_matched[:10])
# show a few velocity rows using different indexing choices:
if 'comvel_all' in globals():
    com = np.asarray(comvel_all)
    # try index by matched_positions
    try:
        print("v_gal_by_matched[0:5]:", com[matched_positions[:5]])
    except Exception as e:
        print("v_gal_by_matched failed:", e)
    # try index by halo_idx[matched_positions]
    try:
        print("v_gal_by_haloidx[0:5]:", com[ halo_idx[matched_positions].astype(int)[:5] ])
    except Exception as e:
        print("v_gal_by_haloidx failed:", e)
    # host velocities by host_halo_idx_matched
    try:
        print("v_host_by_hostidx[0:5]:", com[ host_halo_idx_matched.astype(int)[:5] ])
    except Exception as e:
        print("v_host_by_hostidx failed:", e)

# Compare for the central objects: do host and galaxy velocities match?
if 'comvel_all' in globals():
    com = np.asarray(comvel_all)
    # pick indices of matched centrals
    cen_inds = np.where(is_central_matched)[0]
    for i in cen_inds[:10]:
        # galaxy index in full table
        full_gal_idx = halo_idx[matched_positions[i]]
        host_idx = host_halo_idx_matched[i]
        vg = None; vh = None
        try:
            vg = com[ full_gal_idx ]
        except Exception:
            pass
        try:
            vh = com[ host_idx ]
        except Exception:
            pass
        print(f"matched_local={i} matched_pos={matched_positions[i]} full_gal_idx={full_gal_idx} host_idx={host_idx}")
        print(" vg:", vg, " vh:", vh, " diff_norm:", None if (vg is None or vh is None) else np.linalg.norm(np.asarray(vg)-np.asarray(vh)))