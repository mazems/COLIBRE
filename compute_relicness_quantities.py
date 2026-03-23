#!/usr/bin/env python3
"""
compute_relicness_quantities.py

Memory-safe, lazy-read pipeline to compute relicness ingredients for subhalos in ucmg_ids.csv.
... (rest of docstring unchanged) ...
"""

from __future__ import annotations
import os
import sys
import math
import time
import traceback
import argparse
import gc

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
OUTPUT_PREFIX = 'relicness'             # output -> relicness_<nstart>_<nend>.csv

# Analysis settings
TIME_BIN_GYR = 0.01    # 10 Myr
TERM3_REF = 'tfin'     # choose among 'tfin','t90','t95','t998'
VERBOSE = True

# tune chunk size for HaloCatalogueIndex scan (smaller -> less memory, slightly slower)
HALO_CHUNK_SIZE = 200_000
# -----------------------------------------------------

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs, flush=True)

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
    nbins = max(1, int(np.ceil((t_end - t_start) / time_bin_gyr)) + 1)
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

# === CHANGED ===: argparse so we can run slices
parser = argparse.ArgumentParser()
parser.add_argument("nstart", type=int, nargs='?', default=None)
parser.add_argument("nend", type=int, nargs='?', default=None)
args = parser.parse_args()
# === END CHANGED ===

def main():
    t0_main = time.time()
    vprint("Starting compute_relicness_quantities.py")
    vprint("CWD:", os.getcwd())   # <-- small diagnostic so you can confirm sbatch cwd

    # -------------------- read ucmg IDs --------------------
    if not os.path.exists(UCMG_CSV):
        raise SystemExit(f"ucmg CSV not found: {UCMG_CSV}")

    ucmg_df = pd.read_csv(UCMG_CSV)
    if 'subhalo_id' in ucmg_df.columns:
        subhalo_ids_full = np.array(ucmg_df['subhalo_id'], dtype=np.int64)
    else:
        subhalo_ids_full = np.array(ucmg_df.iloc[:, 0], dtype=np.int64)

    total_ids = len(subhalo_ids_full)
    vprint(f"Read {total_ids} subhalo ids from {UCMG_CSV}")

    # apply optional slice from CLI
    if args.nstart is None and args.nend is None:
        nstart, nend = 0, total_ids
    else:
        nstart = max(0, args.nstart or 0)
        nend   = min(total_ids, args.nend or total_ids)
        if nstart >= nend:
            raise SystemExit("Invalid slice: nstart >= nend")
    subhalo_ids = subhalo_ids_full[nstart:nend]
    vprint(f"Using subhalo_ids[{nstart}:{nend}] -> {len(subhalo_ids)} halos")

    # -------------------- load SOAP mapping (optional) --------------------
    subhalo_to_row = {}
    if os.path.exists(SOAP_CATALOGUE_FILE):
        try:
            with h5py.File(SOAP_CATALOGUE_FILE, 'r') as sf:
                if 'InputHalos' in sf and 'HaloCatalogueIndex' in sf['InputHalos']:
                    soap_arr = sf['InputHalos']['HaloCatalogueIndex'][()]
                else:
                    candidates = find_dataset_anywhere(sf, 'halocatalogueindex')
                    soap_arr = sf[candidates[0]][()] if candidates else None
            if soap_arr is not None:
                for idx, val in enumerate(soap_arr):
                    subhalo_to_row[int(val)] = int(idx)
                vprint(f"Loaded SOAP mapping with {len(soap_arr)} rows.")
        except Exception as e:
            vprint("Warning: Could not load SOAP catalogue mapping:", e)
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

    # === CHANGED: mapping handling (FORCE FAST MODE: MUST HAVE .npz) ===
    unique_req = np.unique(subhalo_ids)
    req_arr = np.array([int(x) for x in unique_req], dtype=np.int64)
    req_set = set(int(x) for x in unique_req)
    vprint(f"Will collect particle indices for {len(req_set)} requested subhalo ids")

    mapping = {}  # final mapping: subhalo_id -> np.array(indices)

    # FORCED FAST MODE: mapping file must exist and contain ALL requested IDs
    mapping_npz = "/home/mzemsch/COLIBRE-analysis/ucmg_particle_index_mapping.npz"
    if not os.path.exists(mapping_npz):
        raise SystemExit(
            f"Required mapping file '{mapping_npz}' not found in working directory {os.getcwd()}.\n"
            "This script will NOT fall back to scanning HaloCatalogueIndex (slow). Please create/move the .npz file."
        )

    vprint(f"Loading precomputed particle-index mapping from '{mapping_npz}' (FAST mode)...")
    mp = np.load(mapping_npz, allow_pickle=False)

    # convert keys to ints when possible and keep only relevant keys (to reduce memory)
    for k in mp.files:
        try:
            sid = int(k)
        except Exception:
            # Skip keys that aren't integer subhalo ids
            continue
        # store all mapping arrays (we'll check completeness below)
        mapping[sid] = mp[k].astype(np.int64)

    # Check completeness: all requested IDs must be present
    missing_ids = [int(sid) for sid in req_arr if int(sid) not in mapping]
    if len(missing_ids) > 0:
        # show up to first 50 missing ids for readability
        snippet = missing_ids[:50]
        raise SystemExit(
            f"Mapping loaded but missing {len(missing_ids)} requested subhalo ids for this job. "
            f"First missing ids: {snippet}\n"
            "Please rebuild mapping to include these or adjust your job slice."
        )

    vprint(f"Loaded mapping for {len(mapping)} subhalos (FAST). All requested IDs present.")
    # === END CHANGED mapping block ===
    # --- ADD THIS DEBUG BLOCK RIGHT AFTER mapping loaded ---
    vprint("Diagnostic: reporting mapping sizes for requested subhalos (first 5 shown)...")
    for j, sid in enumerate(req_arr):
        if j >= 50:  # avoid flooding logs if many ids — adjust if desired
            break
        arr = mapping.get(int(sid), np.array([], dtype=np.int64))
        if arr is None:
            vprint(f"  subhalo {sid}: mapping key NOT present")
        else:
            if arr.size == 0:
                vprint(f"  subhalo {sid}: mapping present but EMPTY")
            else:
                sample = arr[:min(10, arr.size)]
                vprint(f"  subhalo {sid}: Nidx={arr.size} sample_idx={sample.tolist()}")
    # --- END DEBUG BLOCK ---

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
        # correct tfin_span relative to t_start
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

        # free large per-subhalo arrays to reduce memory peak
        try:
            del masses_sel
        except NameError:
            pass
        try:
            del tform_sel
        except NameError:
            pass
        try:
            del em_frac_sel
            del em_sel
        except NameError:
            pass
        try:
            del ages_sel, a_sel, z_birth_sel
        except NameError:
            pass

        gc.collect()

        t_loop = time.time() - t0_loop
        vprint(f"[{i+1}/{len(subhalo_ids)}] subhalo {sid} done in {t_loop:.3f}s; total_formed={total_formed:.3e}")

    total_time = time.time() - start_all
    vprint(f"Processed {len(out_rows)} subhalos in {total_time:.2f} s")

    # -------------------- write output CSV --------------------
    outfn = f"{OUTPUT_PREFIX}_{nstart}_{nend}.csv"
    if len(out_rows) == 0:
        vprint("No results to write.")
    else:
        df = pd.DataFrame(out_rows)
        # ensure subhalo_id, soap_row_index are first
        cols = list(df.columns)
        cols_sorted = ['subhalo_id', 'soap_row_index'] + [c for c in cols if c not in ('subhalo_id', 'soap_row_index')]
        df = df[cols_sorted]
        df.to_csv(outfn, index=False)
        vprint(f"Wrote output CSV: {outfn} ({df.shape[0]} rows, {df.shape[1]} columns)")

    # Close file handle
    try:
        f.close()
    except Exception:
        pass

    vprint("All done. Total script time: {:.2f}s".format(time.time()-t0_main))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Script crashed with exception:", e)
        traceback.print_exc()
        try:
            gc.collect()
        except Exception:
            pass
        sys.exit(3)