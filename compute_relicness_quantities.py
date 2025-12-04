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
from typing import List, Tuple, Dict

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
# === CHANGED: default workers for optional multiprocessing ===
DEFAULT_WORKERS = 1  # keep unchanged behaviour unless user requests more
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

# === CHANGED ===: argparse so we can run slices and set workers
parser = argparse.ArgumentParser()
parser.add_argument("nstart", type=int, nargs='?', default=None)
parser.add_argument("nend", type=int, nargs='?', default=None)
parser.add_argument("--workers", type=int, default=None,
                    help=f"Number of worker processes for per-subhalo processing (default={DEFAULT_WORKERS}). Use 1 to keep current behaviour.")
args = parser.parse_args()
# === END CHANGED ===

# === CHANGED: helper to coalesce indices into contiguous ranges ===
def coalesce_indices(indices: np.ndarray) -> List[Tuple[int,int]]:
    """
    Given a 1D integer array of absolute particle indices (unsorted allowed),
    return a list of (start, stop) inclusive ranges to read from HDF5.
    The returned ranges are half-open for slicing: [start, stop_exclusive).
    """
    if indices is None or indices.size == 0:
        return []
    idx = np.unique(indices.astype(np.int64))
    idx.sort()
    # find runs where consecutive difference >1
    diff = np.diff(idx)
    # boundaries where diff > 1
    breaks = np.nonzero(diff > 1)[0]
    ranges = []
    start = idx[0]
    for b in breaks:
        end = idx[b]   # inclusive
        ranges.append((int(start), int(end + 1)))  # end exclusive
        start = idx[b+1]
    # final run
    ranges.append((int(start), int(idx[-1] + 1)))
    return ranges
# === END CHANGED ===

# === CHANGED: per-halo worker function that performs coalesced reads from HDF5 ===
def process_single_halo(sid: int,
                        mapping_arr: np.ndarray,
                        snapshot_fn: str,
                        soap_row: int | None,
                        time_bin_gyr: float,
                        term3_ref: str,
                        verbose: bool) -> Dict:
    """
    Process one halo (sid). This function opens the snapshot HDF5 itself,
    coalesces indices into contiguous ranges, reads each dataset only for
    those ranges, constructs per-halo arrays and computes the same outputs
    as before. Return a row dict (same shape as before) or None on error.
    """
    # DIAGNOSTIC: record start time and npart for this halo
    t_proc_start = time.time()
    npart = int(mapping_arr.size) if mapping_arr is not None else 0

    if mapping_arr.size == 0:
        # return zero-like entry plus diagnostics
        return {'subhalo_id': int(sid),
                'soap_row_index': soap_row,
                'total_formed_mass': 0.0,
                'proc_time': 0.0,
                'npart': 0}

    try:
        with h5py.File(snapshot_fn, 'r') as f_loc:
            p4_loc = f_loc['PartType4']

            # pick dataset names as previously
            for name in ('InitialMasses', 'Masses', 'masses'):
                if name in p4_loc:
                    masses_ds_loc = p4_loc[name]; break
            else:
                raise RuntimeError("No stellar mass dataset found under PartType4")

            birth_sf_ds_loc = p4_loc['BirthScaleFactors'] if 'BirthScaleFactors' in p4_loc else None
            ages_ds_loc = p4_loc['Ages'] if 'Ages' in p4_loc else None
            elem_mass_fracs_ds_loc = p4_loc['ElementMassFractions'] if 'ElementMassFractions' in p4_loc else None

            # coalesce contiguous index ranges
            ranges = coalesce_indices(mapping_arr)
            # preallocate lists for blocks
            masses_blocks = []
            birth_blocks = [] if birth_sf_ds_loc is not None else None
            ages_blocks = [] if ages_ds_loc is not None else None
            elem_blocks = [] if elem_mass_fracs_ds_loc is not None else None

            # read blocks sequentially
            for (start, stop_excl) in ranges:
                # read slice once per dataset
                masses_blocks.append(np.array(masses_ds_loc[start:stop_excl], dtype=float))
                if birth_sf_ds_loc is not None:
                    birth_blocks.append(np.array(birth_sf_ds_loc[start:stop_excl], dtype=float))
                if ages_ds_loc is not None:
                    ages_blocks.append(np.array(ages_ds_loc[start:stop_excl], dtype=float))
                if elem_mass_fracs_ds_loc is not None:
                    elem_blocks.append(np.array(elem_mass_fracs_ds_loc[start:stop_excl], dtype=float))

            # concatenate blocks -> get arrays aligned with the concatenated block ordering
            masses_sel = np.concatenate(masses_blocks) if masses_blocks else np.array([], dtype=float)
            if birth_blocks is not None:
                birth_arr = np.concatenate(birth_blocks)
            else:
                birth_arr = None
            if ages_blocks is not None:
                ages_arr = np.concatenate(ages_blocks)
            else:
                ages_arr = None
            if elem_blocks is not None:
                elem_arr = np.concatenate(elem_blocks)
            else:
                elem_arr = None

            # BUT: mapping_arr may not correspond to simple sequential ordering inside the concatenated slice
            # We used contiguous ranges built from mapping_arr itself, so the concatenation is in the same order
            # as the sorted unique mapping_arr. To ensure correct particle-wise alignment use sorted mapping ordering:
            sorted_idx = np.argsort(np.unique(mapping_arr))
            # However, because we created ranges from np.unique(mapping_arr) and read in that sorted order,
            # the concatenation ordering equals the sorted unique indices, so we can proceed.

            # build formation times tform_sel similar to original logic
            if birth_arr is not None:
                a_sel = birth_arr
                with np.errstate(divide='ignore', invalid='ignore'):
                    z_birth_sel = (1.0 / a_sel) - 1.0
                valid_z = np.isfinite(z_birth_sel) & (z_birth_sel >= 0.0)
                tform_sel = np.full_like(a_sel, np.nan, dtype=float)
                if np.any(valid_z):
                    tform_sel[valid_z] = Planck15.age(z_birth_sel[valid_z]).to(u.Gyr).value
                if np.any(~valid_z) and ages_arr is not None:
                    t_now = Planck15.age(0).to(u.Gyr).value
                    # fill where invalid with t_now - ages
                    tform_sel[~valid_z] = t_now - ages_arr[~valid_z]
            elif ages_arr is not None:
                t_now = Planck15.age(0).to(u.Gyr).value
                tform_sel = t_now - ages_arr
            else:
                # no time info -> return zero row
                return {'subhalo_id': int(sid), 'soap_row_index': soap_row, 'total_formed_mass': 0.0}

            # filter NaNs
            valid_mask = np.isfinite(tform_sel)
            if not np.all(valid_mask):
                masses_sel = masses_sel[valid_mask]
                tform_sel = tform_sel[valid_mask]
                if elem_arr is not None:
                    elem_arr = elem_arr[valid_mask]

            if tform_sel.size == 0 or masses_sel.size == 0:
                return {'subhalo_id': int(sid), 'soap_row_index': soap_row, 'total_formed_mass': 0.0}

            # compute quantities (reuse compute_mass_hist_times)
            total_formed, t50, t50_span, t75, t75_span, t90, t90_span, t95, t95_span, t998, t998_span = \
                compute_mass_hist_times(tform_sel, masses_sel, time_bin_gyr)

            tfin = t998
            t_start_val = float(np.min(tform_sel))
            tfin_span = tfin - t_start_val if np.isfinite(tfin) else float('nan')

            if total_formed > 0:
                f_Mz2 = float(np.sum(masses_sel[tform_sel <= Planck15.age(2.0).to(u.Gyr).value]) / total_formed)
            else:
                f_Mz2 = float('nan')

            term2 = 0.5 / t75_span if (t75_span is not None and np.isfinite(t75_span) and t75_span > 0) else 1.0

            span_map = {"tfin": tfin_span, "t90": t90_span, "t95": t95_span, "t998": t998_span}
            span_val = span_map.get(term3_ref, tfin_span if np.isfinite(tfin_span) else 0.0)
            term3 = (0.7 + Planck15.age(0).to(u.Gyr).value - span_val) / Planck15.age(0).to(u.Gyr).value if np.isfinite(span_val) else float('nan')

            term1 = float(f_Mz2) if np.isfinite(f_Mz2) else float('nan')
            dor = float((term1 + term2 + term3) / 3.0) if np.isfinite(term1) and np.isfinite(term2) and np.isfinite(term3) else float('nan')

            element_totals = {}
            if elem_arr is not None:
                # elem_arr shape: (Nsel, Nelem)
                if elem_arr.ndim == 2 and elem_arr.shape[0] == masses_sel.size:
                    em_sel = elem_arr * masses_sel[:, None]
                    for ie in range(em_sel.shape[1]):
                        element_totals[f"elem_{ie}_mass"] = float(np.sum(em_sel[:, ie]))

            row = {
                'subhalo_id': int(sid),
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
            # diagnostic fields returned so caller can print timings in parallel mode
            row['proc_time'] = float(time.time() - t_proc_start)
            row['npart'] = npart
            return row
    except Exception as e:
        if verbose:
            print(f"Error in processing halo {sid}: {e}", file=sys.stderr)
            traceback.print_exc()
        return {'subhalo_id': int(sid), 'soap_row_index': soap_row, 'total_formed_mass': 0.0}
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

    # -------------------- open virtual snapshot lazily (only for mapping load; workers re-open) --------------------
    if not os.path.exists(VIRTUAL_SNAPSHOT_FILE):
        raise SystemExit(f"Virtual snapshot not found: {VIRTUAL_SNAPSHOT_FILE}")

    vprint("Opening virtual snapshot (PartType4) and preparing dataset handles (main process)...")
    t0 = time.time()
    f = h5py.File(VIRTUAL_SNAPSHOT_FILE, 'r')
    if 'PartType4' not in f:
        f.close()
        raise SystemExit("PartType4 group not found in snapshot HDF5.")
    p4 = f['PartType4']

    # Determine mass dataset handle (do NOT load full array) — main process uses it only for quick checks
    masses_ds = None
    for name in ('InitialMasses', 'Masses', 'masses'):
        if name in p4:
            masses_ds = p4[name]
            vprint("Using mass dataset (lazy):", name)
            break
    if masses_ds is None:
        f.close()
        raise SystemExit("No stellar mass dataset found under PartType4 (InitialMasses / Masses).")

    # dataset handles for formation time info (lazy) - main process not used for per-halo heavy reads
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
        snippet = missing_ids[:50]
        raise SystemExit(
            f"Mapping loaded but missing {len(missing_ids)} requested subhalo ids for this job. "
            f"First missing ids: {snippet}\n"
            "Please rebuild mapping to include these or adjust your job slice."
        )

    vprint(f"Loaded mapping for {len(mapping)} subhalos (FAST). All requested IDs present.")
    # === END CHANGED mapping block ===

    # -------------------- per-subhalo processing (optionally parallel) --------------------
    out_rows = []
    start_all = time.time()

    # decide number of workers
    workers = args.workers if args.workers is not None else DEFAULT_WORKERS
    if workers is None or workers < 1:
        workers = DEFAULT_WORKERS
    vprint(f"Using workers={workers} for per-halo processing (0: sequential fallback).")

    # Build list of tasks
    tasks = []
    for i, sid in enumerate(subhalo_ids):
        sid_i = int(sid)
        mapping_arr = mapping.get(sid_i, np.array([], dtype=np.int64))
        soap_row = subhalo_to_row.get(sid_i, None)
        tasks.append((sid_i, mapping_arr, soap_row))

    # Sequential fast path (default) to preserve original behavior
    if workers == 1:
        for i, (sid_i, mapping_arr, soap_row) in enumerate(tasks):
            # DIAGNOSTIC start
            vprint(f"[{i+1}/{len(subhalo_ids)}] START subhalo {sid_i} with Npart={int(mapping_arr.size)}")
            t0_loop = time.time()

            row = process_single_halo(sid_i, mapping_arr, VIRTUAL_SNAPSHOT_FILE, soap_row,
                                    TIME_BIN_GYR, TERM3_REF, VERBOSE)
            out_rows.append(row)

            # DIAGNOSTIC end
            proc_time = row.get('proc_time', time.time() - t0_loop)
            npart = row.get('npart', int(mapping_arr.size))
            vprint(f"[{i+1}/{len(subhalo_ids)}] END   subhalo {sid_i}: npart={npart} proc_time={proc_time:.2f}s")

            t_loop = time.time() - t0_loop
            vprint(f"[{i+1}/{len(subhalo_ids)}] subhalo {sid_i} done in {t_loop:.3f}s; total_formed={row.get('total_formed_mass',0.0):.3e}")
    else:
        # parallel execution using ProcessPoolExecutor — each process re-opens the HDF5 file
        from concurrent.futures import ProcessPoolExecutor, as_completed
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as exc:
            # submit
            for i, (sid_i, mapping_arr, soap_row) in enumerate(tasks):
                vprint(f"Submitting worker task [{i+1}/{len(tasks)}] subhalo {sid_i} (npart={int(mapping_arr.size)})")
                futures[exc.submit(process_single_halo, sid_i, mapping_arr, VIRTUAL_SNAPSHOT_FILE, soap_row,
                                TIME_BIN_GYR, TERM3_REF, VERBOSE)] = i

            # collect results
            for f in as_completed(futures):
                idx = futures[f]
                try:
                    row = f.result()
                except Exception as e:
                    vprint(f"Worker failed for task idx {idx}: {e}")
                    row = {'subhalo_id': int(subhalo_ids[idx]), 'soap_row_index': None, 'total_formed_mass': 0.0, 'proc_time': 0.0, 'npart': 0}

                out_rows.append(row)
                sub_id = row.get('subhalo_id')
                proc_time = row.get('proc_time', None)
                npart = row.get('npart', None)
                vprint(f"[{len(out_rows)}/{len(subhalo_ids)}] collected result for subhalo {sub_id} (npart={npart} proc_time={proc_time})")
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