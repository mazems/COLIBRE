#!/usr/bin/env python3
"""
compute_relicness_quantities.py

Memory-safe, lazy-read pipeline to compute relicness ingredients for subhalos in ucmg_ids.csv.

Features:
 - Opens the virtual snapshot HDF5 once and keeps dataset handles (h5py).
 - Uses PartType4/HaloCatalogueIndex (particle -> subhalo id) to select particles per subhalo.
 - Reads only needed slices from large datasets (masses, BirthScaleFactors, Ages, ElementMassFractions).
 - Computes 10 Myr bins (TIME_BIN_GYR = 0.01).
 - Implements DoR formula you supplied, with tX_span = tX - t_start.
 - Writes CSV containing subhalo_id, soap_row_index (if available), times, DoR, element totals.

USER ACTION: set paths in USER CONFIG below before running.
"""

import os
import sys
import math
import time
import numpy as np
import pandas as pd
import h5py
from astropy.cosmology import Planck15
import astropy.units as u

# -------------------- USER CONFIG --------------------
MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN'
SNAP_FILE = 'colibre_with_SOAP_membership_0127.hdf5'   # virtual snapshot (PartType4)
VIRTUAL_SNAPSHOT_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', SNAP_FILE)

SOAP_CATALOGUE_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', 'halo_properties_0127.hdf5')

UCMG_CSV = 'ucmg_ids.csv'               # input CSV (first column or 'subhalo_id')
OUTPUT_CSV = 'relicness_ingredients_fast.csv'

# Analysis settings
TIME_BIN_GYR = 0.01    # 10 Myr
TERM3_REF = 'tfin'     # choose among 'tfin','t90','t95','t998'
N_TO_PROCESS = 1       # None => all, or set small integer (e.g., 1/10) for quick test
VERBOSE = True
# -----------------------------------------------------

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# -------------------- helpers --------------------
def find_dataset_anywhere(h5file, partial_name):
    """Recursively find dataset paths whose name contains partial_name (case-insensitive)."""
    partial_lower = partial_name.lower()
    found = []
    def recurse(group, path=""):
        for key in group:
            obj = group[key]
            newpath = f"{path}/{key}" if path else key
            if isinstance(obj, h5py.Dataset):
                if partial_lower in key.lower() or partial_lower in newpath.lower():
                    found.append(newpath)
            else:
                recurse(obj, newpath)
    recurse(h5file)
    return found

# -------------------- read ucmg IDs --------------------
if not os.path.exists(UCMG_CSV):
    raise SystemExit(f"ucmg CSV not found: {UCMG_CSV}")

ucmg_df = pd.read_csv(UCMG_CSV)
if 'subhalo_id' in ucmg_df.columns:
    subhalo_ids = np.array(ucmg_df['subhalo_id'], dtype=np.int64)
else:
    subhalo_ids = np.array(ucmg_df.iloc[:, 0], dtype=np.int64)

vprint(f"Read {len(subhalo_ids)} subhalo ids from {UCMG_CSV}")

# apply quick-test truncation
if N_TO_PROCESS is not None:
    subhalo_ids = subhalo_ids[:N_TO_PROCESS]
    vprint(f"Processing only first {len(subhalo_ids)} IDs (N_TO_PROCESS={N_TO_PROCESS})")

# -------------------- load SOAP mapping (optional) --------------------
subhalo_to_row = {}
row_to_subhalo = {}
if os.path.exists(SOAP_CATALOGUE_FILE):
    with h5py.File(SOAP_CATALOGUE_FILE, 'r') as sf:
        if 'InputHalos' in sf and 'HaloCatalogueIndex' in sf['InputHalos']:
            soap_arr = sf['InputHalos']['HaloCatalogueIndex'][()]
        else:
            candidates = find_dataset_anywhere(sf, 'halocatalogueindex')
            if candidates:
                soap_arr = sf[candidates[0]][()]
                vprint("Loaded SOAP halo index from:", candidates[0])
            else:
                soap_arr = None
                vprint("Warning: Could not find HaloCatalogueIndex in SOAP catalogue file.")
    if soap_arr is not None:
        for idx, val in enumerate(soap_arr):
            subhalo_to_row[int(val)] = int(idx)
            row_to_subhalo[int(idx)] = int(val)
        vprint(f"Loaded SOAP mapping with {len(soap_arr)} rows.")
else:
    vprint("SOAP catalogue file not found; output will not include soap_row_index.")

# -------------------- open virtual snapshot lazily --------------------
if not os.path.exists(VIRTUAL_SNAPSHOT_FILE):
    raise SystemExit(f"Virtual snapshot not found: {VIRTUAL_SNAPSHOT_FILE}")

vprint("Opening virtual snapshot (PartType4) and preparing dataset handles...")
t0 = time.time()
f = h5py.File(VIRTUAL_SNAPSHOT_FILE, 'r')
if 'PartType4' not in f:
    f.close()
    raise SystemExit("PartType4 group not found in snapshot HDF5.")
p4 = f['PartType4']

# Determine mass dataset handle (do NOT load full array)
masses_ds = None
for name in ('InitialMasses', 'Masses', 'masses'):
    if name in p4:
        masses_ds = p4[name]
        vprint("Using mass dataset (lazy):", name)
        break
if masses_ds is None:
    f.close()
    raise SystemExit("No stellar mass dataset found under PartType4 (InitialMasses / Masses).")

# Load particle->halo mapping into memory (one int per particle). Usually manageable.
if 'HaloCatalogueIndex' not in p4:
    f.close()
    raise SystemExit("PartType4/HaloCatalogueIndex not found in snapshot: cannot assign particles to subhalos.")
particle_halo_index = np.array(p4['HaloCatalogueIndex'][()], dtype=np.int64)
vprint(f"Loaded particle->halo mapping: {particle_halo_index.size:,} particles")

# dataset handles for formation time info (lazy)
birth_sf_ds = p4['BirthScaleFactors'] if 'BirthScaleFactors' in p4 else None
ages_ds = p4['Ages'] if 'Ages' in p4 else None
coords_ds = p4['Coordinates'] if 'Coordinates' in p4 else None
elem_mass_fracs_ds = p4['ElementMassFractions'] if 'ElementMassFractions' in p4 else None

t_load = time.time() - t0
vprint(f"Prepared dataset handles in {t_load:.2f} s")

# Precompute constants
t_z2_gyr = Planck15.age(2.0).to(u.Gyr).value
t_uni_gyr = Planck15.age(0).to(u.Gyr).value
vprint(f"Cosmic age at z=2 = {t_z2_gyr:.4f} Gyr; universe age = {t_uni_gyr:.4f} Gyr")

# -------------------- helper: compute tX from mass histogram --------------------
def compute_mass_hist_times(tform_sel, masses_sel, time_bin_gyr=TIME_BIN_GYR):
    """
    Given formation times (Gyr) and initial masses, produce:
      total_formed, t50,t50_span, t75,t75_span, t90,t90_span, t95,t95_span, t998,t998_span
    tX returned as Gyr, spans as Gyr (tX - t_start).
    """
    if masses_sel.size == 0 or tform_sel.size == 0:
        nan = float('nan')
        return 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan
    t_start = float(np.min(tform_sel))
    t_end = float(np.max(tform_sel))
    nbins = int(np.ceil((t_end - t_start) / time_bin_gyr)) + 1
    bins = np.linspace(t_start, t_start + nbins * time_bin_gyr, nbins + 1)
    mass_per_bin, _ = np.histogram(tform_sel, bins=bins, weights=masses_sel)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    cumulative = np.cumsum(mass_per_bin)
    total_formed = float(cumulative[-1]) if cumulative.size > 0 else 0.0

    def find_tX(frac):
        if total_formed <= 0:
            return float('nan'), float('nan')
        target = frac * total_formed
        idx = int(np.searchsorted(cumulative, target))
        if idx >= len(bin_centers):
            tX = float(bin_centers[-1])
        else:
            tX = float(bin_centers[idx])
        return tX, tX - t_start

    t50, t50_span = find_tX(0.50)
    t75, t75_span = find_tX(0.75)
    t90, t90_span = find_tX(0.90)
    t95, t95_span = find_tX(0.95)
    t998, t998_span = find_tX(0.998)

    return total_formed, t50, t50_span, t75, t75_span, t90, t90_span, t95, t95_span, t998, t998_span

# -------------------- CHUNKED mapping for requested subhalo ids --------------------
# This avoids reading the entire HaloCatalogueIndex into memory.
unique_req = np.unique(subhalo_ids)
req_set = set(int(x) for x in unique_req)
vprint(f"Will collect particle indices for {len(req_set)} requested subhalo ids")

mapping = {int(sid): np.array([], dtype=np.int64) for sid in req_set}

halo_ds = p4['HaloCatalogueIndex']  # h5py dataset handle (do not call [()] on it)
npart_total = halo_ds.shape[0]
vprint(f"HaloCatalogueIndex has {npart_total:,} particles; scanning in chunks...")

chunk_size = 2_000_000   # adjust downward if memory still tight (e.g., 500k)
for start in range(0, np.int64(npart_total), chunk_size):
    stop = min(npart_total, start + chunk_size)
    # read chunk (this is a small array)
    chunk = np.array(halo_ds[start:stop], dtype=np.int64)
    # for each requested id, find matches in chunk
    # vectorized approach: loop over unique req ids is fine (req_set is small)
    for sid in req_set:
        # find locations in chunk where halo == sid
        rel = np.nonzero(chunk == sid)[0]
        if rel.size:
            abs_indices = rel + start
            # append to mapping list (efficient by concatenation of arrays)
            mapping[sid] = np.concatenate([mapping[sid], abs_indices])
    # optional progress print
    if (start // chunk_size) % 10 == 0:
        vprint(f" Scanned particles {start:,}..{stop:,} (collected so far for {len(req_set)} ids)")

# mapping now contains index arrays for requested subhalo ids (may be empty arrays)
vprint("Prepared particle-index mapping for requested subhalos (chunked scan).")

# -------------------- per-subhalo processing --------------------
out_rows = []
start_all = time.time()
for i, sid in enumerate(subhalo_ids):
    t0_loop = time.time()
    sid = int(sid)
    indices = mapping.get(sid, np.array([], dtype=int))
    soap_row = subhalo_to_row.get(sid, None)

    if indices.size == 0:
        vprint(f"[{i+1}/{len(subhalo_ids)}] subhalo {sid}: 0 star particles -> writing zeros")
        out_rows.append({
            'subhalo_id': sid,
            'soap_row_index': soap_row,
            'total_formed_mass': 0.0
        })
        continue

    # Lazy-read masses and formation-related fields for these indices only
    try:
        masses_sel = np.array(masses_ds[indices], dtype=float)
    except Exception as e:
        vprint(f"Error reading masses for subhalo {sid}: {e}")
        continue

    # build formation times tform_sel (in Gyr) from BirthScaleFactors or Ages
    if birth_sf_ds is not None:
        a_sel = np.array(birth_sf_ds[indices], dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            z_birth_sel = (1.0 / a_sel) - 1.0
        valid_z = np.isfinite(z_birth_sel) & (z_birth_sel >= 0.0)
        tform_sel = np.full_like(a_sel, np.nan, dtype=float)
        if np.any(valid_z):
            # vectorized astropy age for selected z
            tform_sel[valid_z] = Planck15.age(z_birth_sel[valid_z]).to(u.Gyr).value
        # fallback to ages dataset for invalid elements if available
        if np.any(~valid_z) and ages_ds is not None:
            ages_sel = np.array(ages_ds[indices][~valid_z], dtype=float)
            t_now = Planck15.age(0).to(u.Gyr).value
            tform_sel[~valid_z] = t_now - ages_sel
    elif ages_ds is not None:
        ages_sel = np.array(ages_ds[indices], dtype=float)
        t_now = Planck15.age(0).to(u.Gyr).value
        tform_sel = t_now - ages_sel
    else:
        vprint(f"subhalo {sid}: no BirthScaleFactors nor Ages available for formation times; skipping")
        continue

    # filter out NaN formation times
    valid_mask = np.isfinite(tform_sel)
    if not np.all(valid_mask):
        masses_sel = masses_sel[valid_mask]
        tform_sel = tform_sel[valid_mask]

    if tform_sel.size == 0 or masses_sel.size == 0:
        vprint(f"[{i+1}/{len(subhalo_ids)}] subhalo {sid}: no valid particle times after filtering")
        out_rows.append({
            'subhalo_id': sid,
            'soap_row_index': soap_row,
            'total_formed_mass': 0.0
        })
        continue

    # Compute SFH-derived times and totals
    total_formed, t50, t50_span, t75, t75_span, t90, t90_span, t95, t95_span, t998, t998_span = \
        compute_mass_hist_times(tform_sel, masses_sel, TIME_BIN_GYR)

    tfin = t998
    tfin_span = (tfin - (t50 - t50_span)) if np.isfinite(tfin) and np.isfinite(t50) and np.isfinite(t50_span) else (tfin - (t50 - t50_span) if np.isfinite(tfin) else float('nan'))
    # NOTE: The above tfin_span calculation needs to be tfin - t_start. But t_start is min(tform_sel).
    # Simpler / clearer:
    t_start_val = float(np.min(tform_sel))
    tfin_span = tfin - t_start_val if np.isfinite(tfin) else float('nan')

    # f_Mz2: fraction formed before cosmic age at z=2
    if total_formed > 0:
        f_Mz2 = float(np.sum(masses_sel[tform_sel <= t_z2_gyr]) / total_formed)
    else:
        f_Mz2 = float('nan')

    # term2: 0.5 / t75_span if t75_span > 0 else 1.0
    term2 = 0.5 / t75_span if (t75_span is not None and np.isfinite(t75_span) and t75_span > 0) else 1.0

    # term3: pick span reference
    span_map = {"tfin": tfin_span, "t90": t90_span, "t95": t95_span, "t998": t998_span}
    span_val = span_map.get(TERM3_REF, tfin_span if np.isfinite(tfin_span) else 0.0)
    term3 = (0.7 + t_uni_gyr - span_val) / t_uni_gyr if np.isfinite(span_val) else float('nan')

    term1 = float(f_Mz2) if np.isfinite(f_Mz2) else float('nan')
    dor = float((term1 + term2 + term3) / 3.0) if np.isfinite(term1) and np.isfinite(term2) and np.isfinite(term3) else float('nan')

    # element totals if ElementMassFractions present (read only per-subhalo)
    element_totals = {}
    if elem_mass_fracs_ds is not None:
        try:
            em_frac_sel = np.array(elem_mass_fracs_ds[indices], dtype=float)
            # if shape (Nsel, Nelem)
            if em_frac_sel.ndim == 2 and em_frac_sel.shape[0] == indices.size:
                # apply the same valid_mask if we filtered NaNs earlier
                if not np.all(valid_mask):
                    em_frac_sel = em_frac_sel[valid_mask]
                em_sel = em_frac_sel * masses_sel[:, None]  # absolute element masses per particle
                for ie in range(em_sel.shape[1]):
                    element_totals[f"elem_{ie}_mass"] = float(np.sum(em_sel[:, ie]))
        except Exception as e:
            vprint(f"Warning: failed to read ElementMassFractions for subhalo {sid}: {e}")

    # prepare output row
    row = {
        'subhalo_id': sid,
        'soap_row_index': soap_row,
        'total_formed_mass': float(total_formed),
        'stellar_mass_current': float(np.sum(masses_sel)),
        't_start': float(t_start_val),
        't50': float(t50), 't50_span': float(t50_span),
        't75': float(t75), 't75_span': float(t75_span),
        't90': float(t90), 't90_span': float(t90_span),
        't95': float(t95), 't95_span': float(t95_span),
        't998': float(t998), 't998_span': float(t998_span),
        'tfin': float(tfin), 'tfin_span': float(tfin_span),
        'f_Mz2': float(f_Mz2),
        'term1': float(term1),
        'term2': float(term2),
        'term3': float(term3),
        'DoR': float(dor)
    }
    row.update(element_totals)
    out_rows.append(row)

    t_loop = time.time() - t0_loop
    vprint(f"[{i+1}/{len(subhalo_ids)}] subhalo {sid} done in {t_loop:.3f}s; total_formed={total_formed:.3e}")

total_time = time.time() - start_all
vprint(f"Processed {len(out_rows)} subhalos in {total_time:.2f} s")

# -------------------- write output CSV --------------------
if len(out_rows) == 0:
    vprint("No results to write.")
else:
    df = pd.DataFrame(out_rows)
    # ensure subhalo_id, soap_row_index are first
    cols = list(df.columns)
    cols_sorted = ['subhalo_id', 'soap_row_index'] + [c for c in cols if c not in ('subhalo_id', 'soap_row_index')]
    df = df[cols_sorted]
    df.to_csv(OUTPUT_CSV, index=False)
    vprint(f"Wrote output CSV: {OUTPUT_CSV} ({df.shape[0]} rows, {df.shape[1]} columns)")

# Close file handle
try:
    f.close()
except Exception:
    pass

vprint("All done.")