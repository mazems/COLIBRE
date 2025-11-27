#!/usr/bin/env python3
"""
compute_relicness_quantities.py

Memory-safe, lazy-read pipeline to compute relicness ingredients for subhalos in ucmg_ids.csv.

Notes:
 - Never loads the full HaloCatalogueIndex into RAM.
 - Scans HaloCatalogueIndex in small chunks and collects indices only for requested subhalos.
 - Reads masses and formation-time fields only for selected particle indices.
 - Handles MemoryError and logs gracefully.
 - Set N_TO_PROCESS small (1) for testing; set to None for all.
"""

import os
import sys
import time
import math
import traceback
import numpy as np
import pandas as pd
import h5py
from astropy.cosmology import Planck15
import astropy.units as u
import gc

# -------------------- USER CONFIG --------------------
MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN'
SNAP_FILE = 'colibre_with_SOAP_membership_0127.hdf5'
VIRTUAL_SNAPSHOT_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', SNAP_FILE)
SOAP_CATALOGUE_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', 'halo_properties_0127.hdf5')

UCMG_CSV = 'ucmg_ids.csv'
OUTPUT_CSV = 'relicness_ingredients_fast.csv'

TIME_BIN_GYR = 0.01
TERM3_REF = 'tfin'
N_TO_PROCESS = 10     # set to None to run all
VERBOSE = True

# tune chunk size for HaloCatalogueIndex scan (smaller -> less memory, slightly slower)
HALO_CHUNK_SIZE = 200_000
# ---------------------------------------------------------------------

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def find_dataset_anywhere(h5file, partial_name):
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

# ---- main ----
def main():
    start_main = time.time()
    vprint("Starting compute_relicness_quantities.py")

    if not os.path.exists(UCMG_CSV):
        raise SystemExit(f"ucmg CSV not found: {UCMG_CSV}")
    ucmg_df = pd.read_csv(UCMG_CSV)
    if 'subhalo_id' in ucmg_df.columns:
        subhalo_ids = np.array(ucmg_df['subhalo_id'], dtype=np.int64)
    else:
        subhalo_ids = np.array(ucmg_df.iloc[:,0], dtype=np.int64)
    vprint(f"Read {len(subhalo_ids)} subhalo ids from {UCMG_CSV}")

    if N_TO_PROCESS is not None:
        subhalo_ids = subhalo_ids[:N_TO_PROCESS]
        vprint(f"Truncated to first {len(subhalo_ids)} IDs (N_TO_PROCESS={N_TO_PROCESS})")

    # SOAP mapping (optional)
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
                vprint(f"Loaded SOAP mapping ({len(soap_arr)} rows)")
        except Exception as e:
            vprint("Warning: failed to load SOAP catalogue mapping:", e)

    if not os.path.exists(VIRTUAL_SNAPSHOT_FILE):
        raise SystemExit(f"Virtual snapshot not found: {VIRTUAL_SNAPSHOT_FILE}")

    # open file handle (keep open)
    try:
        f = h5py.File(VIRTUAL_SNAPSHOT_FILE, 'r')
    except Exception as e:
        raise SystemExit(f"Unable to open snapshot HDF5: {e}")

    if 'PartType4' not in f:
        f.close()
        raise SystemExit("PartType4 not found in snapshot")
    p4 = f['PartType4']

    # mass dataset handle (lazy)
    masses_ds = None
    for name in ('InitialMasses', 'Masses', 'masses'):
        if name in p4:
            masses_ds = p4[name]
            vprint("Using mass dataset:", name)
            break
    if masses_ds is None:
        f.close()
        raise SystemExit("No mass dataset found under PartType4")

    # birth/ages dataset handles (lazy)
    birth_sf_ds = p4['BirthScaleFactors'] if 'BirthScaleFactors' in p4 else None
    ages_ds = p4['Ages'] if 'Ages' in p4 else None
    elem_mass_fracs_ds = p4['ElementMassFractions'] if 'ElementMassFractions' in p4 else None

    # compute constants
    t_z2_gyr = Planck15.age(2.0).to(u.Gyr).value
    t_uni_gyr = Planck15.age(0).to(u.Gyr).value

    # -------------------- CHUNKED mapping for requested subhalo ids (fixed + faster) --------------------
    unique_req = np.unique(subhalo_ids)
    req_arr = np.array([int(x) for x in unique_req], dtype=np.int64)
    req_set = set(int(x) for x in unique_req)
    vprint(f"Preparing mapping for {len(req_arr)} requested subhalo ids")

    mapping_lists = {sid: [] for sid in req_arr}

    halo_ds = p4['HaloCatalogueIndex']   # h5py dataset handle
    npart_total = int(halo_ds.shape[0])
    vprint(f"HaloCatalogueIndex contains {npart_total:,} particles; scanning in chunks...")

    # tune the chunk size smaller if you still see memory spikes
    chunk_size = HALO_CHUNK_SIZE  # e.g. 200_000

    for start in range(0, npart_total, chunk_size):
        stop = min(npart_total, start + chunk_size)
        chunk = np.array(halo_ds[start:stop], dtype=np.int64)   # small temp array

        # mask of particles in this chunk that belong to any requested subhalo
        mask = np.isin(chunk, req_arr)
        if not np.any(mask):
            del chunk
            if (start // chunk_size) % 20 == 0:
                vprint(f" Scanned {start:,}..{stop:,} (no requested ids found in this chunk)")
            continue

        rel_positions = np.nonzero(mask)[0]        # positions in chunk with requested ids
        vals = chunk[rel_positions]                # the corresponding subhalo ids for those positions

        # group positions by unique value present in this chunk
        unique_vals, inv = np.unique(vals, return_inverse=True)
        for j, val in enumerate(unique_vals):
            pos_for_val = rel_positions[inv == j]
            mapping_lists[int(val)].append(pos_for_val + start)

        # free chunk ASAP
        del chunk
        if (start // chunk_size) % 20 == 0:
            vprint(f" Scanned {start:,}..{stop:,} (found {unique_vals.size} requested ids)")

    # convert lists to arrays (concat once per requested id)
    mapping = {}
    for sid in req_arr:
        list_of_chunks = mapping_lists.get(int(sid), [])
        if len(list_of_chunks) == 0:
            mapping[int(sid)] = np.array([], dtype=np.int64)
        else:
            mapping[int(sid)] = np.concatenate(list_of_chunks).astype(np.int64)
        # remove the temporary list to free memory
        del mapping_lists[int(sid)]
    vprint("Prepared particle-index mapping for requested subhalos (chunked, optimized).")

    # process per subhalo
    out_rows = []
    start_all = time.time()
    for idx, sid in enumerate(subhalo_ids):
        t_loop_start = time.time()
        sid = int(sid)
        indices = mapping.get(sid, np.array([], dtype=np.int64))
        soap_row = subhalo_to_row.get(sid, None)
        if indices.size == 0:
            vprint(f"[{idx+1}/{len(subhalo_ids)}] subhalo {sid}: 0 star particles")
            out_rows.append({'subhalo_id': sid, 'soap_row_index': soap_row, 'total_formed_mass': 0.0})
            continue

        try:
            # read masses for these indices
            masses_sel = np.array(masses_ds[indices], dtype=float)
        except MemoryError:
            vprint(f"MemoryError reading masses for subhalo {sid} — skipping this halo.")
            continue
        except Exception as e:
            vprint(f"Error reading masses for subhalo {sid}: {e}")
            continue

        # build formation times for selected particles
        try:
            if birth_sf_ds is not None:
                a_sel = np.array(birth_sf_ds[indices], dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    z_birth_sel = (1.0 / a_sel) - 1.0
                valid_z = np.isfinite(z_birth_sel) & (z_birth_sel >= 0.0)
                tform_sel = np.full_like(a_sel, np.nan, dtype=float)
                if np.any(valid_z):
                    tform_sel[valid_z] = Planck15.age(z_birth_sel[valid_z]).to(u.Gyr).value
                if np.any(~valid_z) and ages_ds is not None:
                    ages_sel = np.array(ages_ds[indices][~valid_z], dtype=float)
                    t_now = Planck15.age(0).to(u.Gyr).value
                    tform_sel[~valid_z] = t_now - ages_sel
            elif ages_ds is not None:
                ages_sel = np.array(ages_ds[indices], dtype=float)
                t_now = Planck15.age(0).to(u.Gyr).value
                tform_sel = t_now - ages_sel
            else:
                vprint(f"subhalo {sid}: no birth/ages -> skipping")
                del masses_sel
                gc.collect()
                continue
        except MemoryError:
            vprint(f"MemoryError building formation times for subhalo {sid} — skipping.")
            try: del masses_sel
            except Exception: pass
            gc.collect()
            continue

        # filter NaNs
        valid_mask = np.isfinite(tform_sel)
        if not np.all(valid_mask):
            masses_sel = masses_sel[valid_mask]
            tform_sel = tform_sel[valid_mask]

        if tform_sel.size == 0 or masses_sel.size == 0:
            vprint(f"[{idx+1}/{len(subhalo_ids)}] subhalo {sid}: no valid particles after filtering")
            out_rows.append({'subhalo_id': sid, 'soap_row_index': soap_row, 'total_formed_mass': 0.0})
            try: del masses_sel, tform_sel
            except Exception: pass
            gc.collect()
            continue

        # compute hist-based times
        total_formed, t50, t50_span, t75, t75_span, t90, t90_span, t95, t95_span, t998, t998_span = \
            compute_mass_hist_times(tform_sel, masses_sel, TIME_BIN_GYR)

        t_start_val = float(np.min(tform_sel))
        tfin = t998
        tfin_span = tfin - t_start_val if np.isfinite(tfin) else float('nan')

        if total_formed > 0:
            f_Mz2 = float(np.sum(masses_sel[tform_sel <= t_z2_gyr]) / total_formed)
        else:
            f_Mz2 = float('nan')

        term2 = 0.5 / t75_span if (t75_span is not None and np.isfinite(t75_span) and t75_span > 0) else 1.0
        span_map = {"tfin": tfin_span, "t90": t90_span, "t95": t95_span, "t998": t998_span}
        span_val = span_map.get(TERM3_REF, tfin_span if np.isfinite(tfin_span) else 0.0)
        term3 = (0.7 + t_uni_gyr - span_val) / t_uni_gyr if np.isfinite(span_val) else float('nan')
        term1 = float(f_Mz2) if np.isfinite(f_Mz2) else float('nan')
        dor = float((term1 + term2 + term3) / 3.0) if np.isfinite(term1) and np.isfinite(term2) and np.isfinite(term3) else float('nan')

        # element totals (optional)
        element_totals = {}
        if elem_mass_fracs_ds is not None:
            try:
                em_frac_sel = np.array(elem_mass_fracs_ds[indices], dtype=float)
                if em_frac_sel.ndim == 2:
                    if not np.all(valid_mask):
                        em_frac_sel = em_frac_sel[valid_mask]
                    em_sel = em_frac_sel * masses_sel[:, None]
                    for ie in range(em_sel.shape[1]):
                        element_totals[f"elem_{ie}_mass"] = float(np.sum(em_sel[:, ie]))
            except Exception as e:
                vprint(f"Warning: element fractions read failed for {sid}: {e}")

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

        # cleanup per-subhalo
        try: del masses_sel, tform_sel
        except Exception: pass
        try:
            del em_frac_sel, em_sel
        except Exception:
            pass
        try:
            del ages_sel, a_sel, z_birth_sel
        except Exception:
            pass
        gc.collect()

        vprint(f"[{idx+1}/{len(subhalo_ids)}] subhalo {sid} done (total_formed={total_formed:.3e}) in {time.time()-t_loop_start:.2f}s")

    # write output CSV
    if len(out_rows) > 0:
        df = pd.DataFrame(out_rows)
        cols = list(df.columns)
        cols_sorted = ['subhalo_id', 'soap_row_index'] + [c for c in cols if c not in ('subhalo_id','soap_row_index')]
        df = df[cols_sorted]
        df.to_csv(OUTPUT_CSV, index=False)
        vprint(f"Wrote {OUTPUT_CSV} ({df.shape[0]} rows, {df.shape[1]} cols)")

    f.close()
    vprint("All done. Total time: {:.2f}s".format(time.time()-start_main))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Script crashed with exception:", e)
        traceback.print_exc()
        try:
            # attempt to close file if open
            import gc
            gc.collect()
        except Exception:
            pass
        raise