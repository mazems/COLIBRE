#!/usr/bin/env python3
"""
compute_relicness_swift.py

Minimal relicness pipeline using swiftsimio + swiftgalaxy only (NO h5py fallback).
Reads sg.stars.* fields including element channels by name (hydrogen, helium, ...),
computes SFH-derived times and DoR, and writes CSV output.

Use --test-subhalo <id> to run a single halo for diagnostics.
"""
from __future__ import annotations
import os
# avoid loading GPU/visualisation code on import (prevents numba.cuda signature errors on some nodes)
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("SWIFTSIMIO_DISABLE_VISUALISATION", "1")
import sys
import time
import argparse
import gc
import traceback

import numpy as np
import pandas as pd
import unyt as un

# import swiftsimio as sw
from swiftsimio import SWIFTDataset
from swiftgalaxy import SWIFTGalaxy, SOAP

# your utilities for lookback time (keeps behaviour consistent if you want to compare)
import utilities_statistics as us

from astropy.cosmology import Planck15, FlatLambdaCDM
import astropy.units as au

# ---------------- USER CONFIG ----------------
MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN'
SNAP_FILE = 'colibre_with_SOAP_membership_0127.hdf5'
VIRTUAL_SNAPSHOT_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', SNAP_FILE)
SOAP_CATALOGUE_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', 'halo_properties_0127.hdf5')

UCMG_CSV = 'ucmg_ids.csv'
OUTPUT_PREFIX = 'relicness'
TIME_BIN_GYR = 0.01        # histogram bin width (Gyr)
TERM3_REF = 'tfin'         # tfin | t90 | t95 | t998
VERBOSE = True

# cosmology params from COLIBRE 2025 (use these to build a FlatLambdaCDM)
COLIBRE_H = 0.681
COLIBRE_OMEGAM = 0.306
COLIBRE_OMEGAL = 0.693922

# keep a COSMO_PARAMS dict for comparsion / compatibility with utilities_statistics if needed
COSMO_PARAMS = dict(h=COLIBRE_H, omegam=COLIBRE_OMEGAM, omegal=COLIBRE_OMEGAL)

# element list: names mapped to swift attribute names on element_mass_fractions
ELEMENT_NAMES = ["H","He","C","N","O","Ne","Mg","Si","Fe","Sr","Ba","Eu"]
ELEMENT_ATTRS = ["hydrogen","helium","carbon","nitrogen","oxygen","neon",
                 "magnesium","silicon","iron","strontium","barium","europium"]
SCALE = 1e10
# ------------------------------------------------

# Build an explicit astropy cosmology matching COLIBRE 2025
cosmo_colibre = FlatLambdaCDM(H0=100.0 * COLIBRE_H, Om0=COLIBRE_OMEGAM)
t_universe_gyr = cosmo_colibre.age(0).to(au.Gyr).value

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs, flush=True)

def compute_mass_hist_times(tform_sel, masses_sel, time_bin_gyr=TIME_BIN_GYR):
    """Compute total formed mass and t50/t75/t90/t95/t998 and spans (Gyr)."""
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
        tX = float(bin_centers[-1]) if idx >= len(bin_centers) else float(bin_centers[idx])
        return tX, tX - t_start

    t50, t50_span = find_tX(0.50)
    t75, t75_span = find_tX(0.75)
    t90, t90_span = find_tX(0.90)
    t95, t95_span = find_tX(0.95)
    t998, t998_span = find_tX(0.998)
    return total_formed, t50, t50_span, t75, t75_span, t90, t90_span, t95, t95_span, t998, t998_span

def load_soap_index_map(soap_file):
    """Return dict mapping halo_id -> soap_index (position in SOAP catalogue)."""
    if not os.path.exists(soap_file):
        raise SystemExit("SOAP catalogue missing: " + soap_file)
    sd = SWIFTDataset(soap_file)
    arr = np.array(sd.input_halos.halo_catalogue_index, dtype=np.int64)
    return {int(val): int(idx) for idx, val in enumerate(arr)}

# CLI parsing
parser = argparse.ArgumentParser()
parser.add_argument("nstart", type=int, nargs='?', default=None)
parser.add_argument("nend", type=int, nargs='?', default=None)
parser.add_argument("--test-subhalo", type=int, default=None, help="Run single halo (by subhalo_id) and print diagnostics.")
args = parser.parse_args()

def process_single_halo_with_sg(subhalo_id, soap_index_map):
    """Build SWIFTGalaxy, read sg.stars fields (including element channels by name), compute metrics."""
    sid = int(subhalo_id)
    if sid not in soap_index_map:
        raise KeyError(f"{sid} not found in SOAP mapping.")
    soap_idx = int(soap_index_map[sid])
    vprint(f"Building SWIFTGalaxy for subhalo {sid} (soap_index={soap_idx})")
    sg = SWIFTGalaxy(VIRTUAL_SNAPSHOT_FILE, SOAP(SOAP_CATALOGUE_FILE, soap_index=soap_idx))

    # --- read masses (initial and current) via swiftgalaxy and convert to Msun floats ---
    m_initial_raw = sg.stars.initial_masses if hasattr(sg.stars, 'initial_masses') else None
    m_current_raw = sg.stars.masses if hasattr(sg.stars, 'masses') else None

    if m_initial_raw is None:
        m_initial = np.array([], dtype=float)
    else:
        try:
            m_initial = np.asarray(m_initial_raw.to(un.Msun).value, dtype=float)
        except Exception:
            m_initial = np.asarray(m_initial_raw, dtype=float)

    if m_current_raw is None:
        m_current = np.array([], dtype=float)
    else:
        try:
            m_current = np.asarray(m_current_raw.to(un.Msun).value, dtype=float)
        except Exception:
            m_current = np.asarray(m_current_raw, dtype=float)

    vprint(f"Read masses: Nstars={m_current.size}, sum_current={np.sum(m_current):.6e} Msun")

    # --- compute cosmic formation times (Gyr) using cosmo_colibre (COLIBRE2025) ---
    # Note: we use cosmo_colibre.lookback_time and cosmo_colibre.age consistently.
    tform = np.array([], dtype=float)
    if hasattr(sg.stars, 'birth_scale_factors'):
        birth_sf = np.array(sg.stars.birth_scale_factors, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            redshifts = (1.0 / birth_sf) - 1.0
        valid_z = np.isfinite(redshifts) & (redshifts >= 0.0)

        # --- quick diagnostic: compare Planck15.age vs cosmo_colibre.age vs us.look_back_time for a small sample
        try:
            zs = redshifts[valid_z][:10]   # small sample
            if zs.size > 0:
                print("DEBUG: sample z (first valid):", zs)
                print("DEBUG: Planck15 ages (Gyr):", Planck15.age(zs).to(au.Gyr).value)
                print("DEBUG: cosmo_colibre ages (Gyr):", cosmo_colibre.age(zs).to(au.Gyr).value)
                # attempt to show us.look_back_time conversion for comparison (may differ)
                try:
                    lb_us = us.look_back_time(zs, **COSMO_PARAMS)
                    us_ages = float(t_universe_gyr) - np.array(lb_us, dtype=float)
                    print("DEBUG: us.look_back_time -> ages (Gyr):", us_ages)
                except Exception as e_us:
                    print("DEBUG: us.look_back_time diagnostic failed:", e_us)
            else:
                print("DEBUG: no valid birth redshifts in this halo to compare.")
        except Exception as e:
            print("DEBUG: diagnostic failed:", e)

        tform_age = np.full_like(birth_sf, np.nan, dtype=float)
        if np.any(valid_z):
            # use astropy cosmo_colibre to get lookback time -> cosmic age
            lb_q = cosmo_colibre.lookback_time(redshifts[valid_z])
            lb_gyr = np.array(lb_q.to(au.Gyr).value, dtype=float)
            tform_age[valid_z] = float(t_universe_gyr) - lb_gyr

        # fallback: use sg.stars.ages for invalid entries if present
        if np.any(~valid_z) and hasattr(sg.stars, 'ages'):
            ages_arr = np.array(sg.stars.ages, dtype=float)
            tform_age[~valid_z] = float(t_universe_gyr) - ages_arr[~valid_z]
        tform = tform_age

    elif hasattr(sg.stars, 'ages'):
        # ages are stellar ages (Gyr) = lookback; cosmic age = t_universe - ages
        ages = np.array(sg.stars.ages, dtype=float)
        tform = (t_universe_gyr - ages).astype(float)
    else:
        tform = np.array([], dtype=float)

    # --- apply validity mask (both masses must be finite and tform finite) ---
    valid = np.isfinite(tform) & np.isfinite(m_initial) & np.isfinite(m_current)
    if not np.any(valid):
        # return zeros-style row
        return {
            'subhalo_id': sid,
            'soap_row_index': soap_idx,
            'total_formed_mass': 0.0,
            'stellar_mass_current': 0.0,
            't_start': float('nan'),
            't50': float('nan'),
            't50_span': float('nan'),
            't75': float('nan'),
            't75_span': float('nan'),
            't90': float('nan'),
            't90_span': float('nan'),
            't95': float('nan'),
            't95_span': float('nan'),
            't998': float('nan'),
            't998_span': float('nan'),
            'tfin': float('nan'),
            'tfin_span': float('nan'),
            'f_Mz2': float('nan'),
            'term1': float('nan'),
            'term2': float('nan'),
            'term3': float('nan'),
            'DoR': float('nan')
        }

    masses_sel = m_initial[valid]           # use initial/formed masses for SFH (as earlier)
    masses_current_sel = m_current[valid]   # current stellar masses for element multiplication & diagnostics
    tform_sel = tform[valid]

    # --- element totals: read channels by name via swiftgalaxy (pure swift approach) ---
    elem_totals = {}
    if hasattr(sg.stars, "element_mass_fractions"):
        em_parent = getattr(sg.stars, "element_mass_fractions")
        # iterate channels by name and attempt to read each one
        for nm, attr in zip(ELEMENT_NAMES, ELEMENT_ATTRS):
            try:
                ch = getattr(em_parent, attr, None)
                if ch is None:
                    vprint(f"  element channel '{attr}' missing; skipping")
                    continue
                ch_arr = np.asarray(ch, dtype=float)
                if ch_arr.ndim != 1:
                    vprint(f"  element channel '{attr}' ndim={ch_arr.ndim} != 1; skipping")
                    continue
                ch_sel = ch_arr[valid]
                if ch_sel.shape[0] != masses_current_sel.size:
                    vprint(f"  element channel '{attr}' length mismatch ({ch_sel.shape[0]} != {masses_current_sel.size}); skipping")
                    continue
                # compute total: sum_i masses_current_sel[i] * ch_sel[i]
                total = float(np.dot(masses_current_sel.astype(np.float32), ch_sel.astype(np.float32)).astype(np.float64))
                elem_totals[f"elem_{nm}_mass"] = total
            except Exception as e:
                vprint(f"  Warning: failed to read element channel '{attr}': {e}; skipping that channel.")
    else:
        vprint("sg.stars.element_mass_fractions not present on this SWIFTGalaxy; skipping elements.")
    # rescale to 10^10 solar masses
    if elem_totals:
        for k in list(elem_totals.keys()):
            elem_totals[k] = elem_totals[k] / SCALE

    # --- SFH derived metrics ---
    total_formed, t50, t50_span, t75, t75_span, t90, t90_span, t95, t95_span, t998, t998_span = \
        compute_mass_hist_times(tform_sel, masses_sel, TIME_BIN_GYR)

    tfin = t998
    t_start_val = float(np.min(tform_sel))
    tfin_span = tfin - t_start_val if np.isfinite(tfin) else float('nan')

    # fraction formed before cosmic age at z=2 (use cosmo_colibre for consistency)
    t_z2_gyr = cosmo_colibre.age(2.0).to(au.Gyr).value
    f_Mz2 = (np.sum(masses_sel[tform_sel <= t_z2_gyr]) / total_formed) if (total_formed > 0) else float('nan')

    term2 = 0.5 / t75_span if (t75_span is not None and np.isfinite(t75_span) and t75_span > 0.0) else 1.0
    span_map = {"tfin": tfin_span, "t90": t90_span, "t95": t95_span, "t998": t998_span}
    span_val = span_map.get(TERM3_REF, tfin_span if np.isfinite(tfin_span) else 0.0)

    term3 = (0.7 + t_universe_gyr - span_val) / t_universe_gyr if np.isfinite(span_val) else float('nan')
    term1 = float(f_Mz2) if np.isfinite(f_Mz2) else float('nan')
    dor = float((term1 + term2 + term3) / 3.0) if np.isfinite(term1) and np.isfinite(term2) and np.isfinite(term3) else float('nan')

    # --- prepare row and cleanup temporaries ---
    row = {
        'subhalo_id': sid,
        'soap_row_index': soap_idx,
        'total_formed_mass': float(total_formed)/SCALE, # in 10^10 solar masses
        'stellar_mass_current': float(np.sum(masses_current_sel))/SCALE, # in 10^10 solar masses
        't_start': float(t_start_val),
        't50': float(t50), 't50_span': float(t50_span),
        't75': float(t75), 't75_span': float(t75_span),
        't90': float(t90), 't90_span': float(t90_span),
        't95': float(t95), 't95_span': float(t95_span),
        't998': float(t998), 't998_span': float(t998_span),
        'tfin': float(tfin), 'tfin_span': float(tfin_span),
        'f_Mz2': float(f_Mz2),
        'term1': float(term1), 'term2': float(term2), 'term3': float(term3),
        'DoR': float(dor)
    }
    row.update(elem_totals)

    # explicit cleanup for large arrays
    try:
        del m_initial_raw, m_current_raw, m_initial, m_current
    except Exception:
        pass
    try:
        del masses_sel, masses_current_sel, tform_sel, tform
    except Exception:
        pass
    gc.collect()

    return row

def main():
    vprint("Starting pure-swift relicness script")
    soap_map = load_soap_index_map(SOAP_CATALOGUE_FILE)
    vprint(f"Loaded SOAP mapping for {len(soap_map)} halos")

    # test mode
    if args.test_subhalo is not None:
        hid = int(args.test_subhalo)
        row = process_single_halo_with_sg(hid, soap_map)
        vprint("Result for test halo:", hid)
        for k, val in row.items():
            vprint(f"  {k}: {val}")
        return

    # batch mode
    if not os.path.exists(UCMG_CSV):
        raise SystemExit("ucmg_ids.csv not found and not in test mode.")
    df = pd.read_csv(UCMG_CSV)
    subhalo_ids_full = np.array(df['subhalo_id'] if 'subhalo_id' in df.columns else df.iloc[:, 0], dtype=np.int64)
    total_ids = len(subhalo_ids_full)
    if args.nstart is None and args.nend is None:
        nstart, nend = 0, total_ids
    else:
        nstart = max(0, args.nstart or 0)
        nend = min(total_ids, args.nend or total_ids)
        if nstart >= nend:
            raise SystemExit("Invalid slice: nstart >= nend")
    subhalo_ids = subhalo_ids_full[nstart:nend]
    vprint(f"Batch mode: processing {len(subhalo_ids)} halos (slice {nstart}:{nend})")

    out_rows = []
    t_all = time.time()
    for i, sid in enumerate(subhalo_ids, start=1):
        try:
            row = process_single_halo_with_sg(sid, soap_map)
            out_rows.append(row)
            vprint(f"[{i}/{len(subhalo_ids)}] subhalo {sid} done; total_formed={row['total_formed_mass']:.3e}")
        except Exception as e:
            vprint(f"[{i}/{len(subhalo_ids)}] subhalo {sid} FAILED: {e}")
            out_rows.append({'subhalo_id': int(sid), 'soap_row_index': soap_map.get(int(sid), None), 'total_formed_mass': 0.0})
        gc.collect()

    vprint("All halos processed. Writing CSV.")
    df_out = pd.DataFrame(out_rows)
    cols = ['subhalo_id','soap_row_index'] + [c for c in df_out.columns if c not in ('subhalo_id','soap_row_index')]
    df_out = df_out[cols]
    outfn = f"{OUTPUT_PREFIX}_{nstart}_{nend}.csv"
    df_out.to_csv(outfn, index=False)
    vprint(f"Wrote {outfn} ({df_out.shape[0]} rows) in {time.time()-t_all:.1f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Script crashed:", e)
        traceback.print_exc()
        sys.exit(1)