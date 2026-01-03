#!/usr/bin/env python3
"""
compute_stellar_current_swift.py

Like your compute_stellar_current_swift.py but with a progress bar and optional periodic flush.
Outputs CSV with columns:
  subhalo_id, stellar_mass_current, stellar_halfmass_radius_kpc, logsigma

Usage examples:
  python3 compute_stellar_current_swift_progress.py
  python3 compute_stellar_current_swift_progress.py --ids 270 271 272
  python3 compute_stellar_current_swift_progress.py --flush-every 50

Notes:
 - Requires swiftgalaxy + swiftsimio available in your environment.
 - Will overwrite output file if it exists.
"""
from __future__ import annotations
import os, sys, argparse
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("SWIFTSIMIO_DISABLE_VISUALISATION", "1")

import math, time, traceback
import numpy as np
import pandas as pd

# swift I/O
from swiftsimio import SWIFTDataset
from swiftgalaxy import SWIFTGalaxy, SOAP
import unyt as un

# try tqdm, fallback to simple printer
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# ---------- user config ----------
MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN'
SNAP_FILE = 'colibre_with_SOAP_membership_0127.hdf5'
VIRTUAL_SNAPSHOT_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', SNAP_FILE)
SOAP_CATALOGUE_FILE = os.path.join(MODEL_DIR, 'SOAP-HBT', 'halo_properties_0127.hdf5')

DEFAULT_UCMG = 'ucmg_ids.csv'
DEFAULT_OUT = 'stellar_current_map_with_rhalf.csv'
# ---------------------------------

parser = argparse.ArgumentParser(description="Compute current stellar mass + half-mass radius + logsigma (swiftgalaxy)")
parser.add_argument('--ucmg', default=DEFAULT_UCMG, help='CSV with subhalo_id (or single column)')
parser.add_argument('--out', default=DEFAULT_OUT, help='Output CSV filename')
parser.add_argument('--ids', nargs='*', type=int, help='Optional manual list of subhalo ids (overrides --ucmg)')
parser.add_argument('--flush-every', type=int, default=0,
                    help='If >0: write an interim CSV every N halos (useful if you plan to cancel/restart).')
parser.add_argument('--verbose', action='store_true', help='More per-halo printouts')
args = parser.parse_args()

def vprint(*a, **k):
    if args.verbose:
        print(*a, **k, flush=True)

# read requested ids
if args.ids:
    subids = np.array(args.ids, dtype=np.int64)
else:
    if not os.path.exists(args.ucmg):
        raise SystemExit(f"ucmg CSV not found: {args.ucmg}")
    df = pd.read_csv(args.ucmg)
    if 'subhalo_id' in df.columns:
        subids = np.array(df['subhalo_id'], dtype=np.int64)
    else:
        subids = np.array(df.iloc[:,0], dtype=np.int64)

# load SOAP mapping (swiftsimio)
sd = SWIFTDataset(SOAP_CATALOGUE_FILE)
soap_ids = np.asarray(sd.input_halos.halo_catalogue_index, dtype=np.int64)
soap_map = {int(val): int(idx) for idx, val in enumerate(soap_ids)}
vprint(f"Loaded SOAP mapping ({len(soap_map)} halos)")

# prepare iteration
N = len(subids)
use_tqdm = TQDM_AVAILABLE
if use_tqdm:
    pbar = tqdm(subids, desc="halos", unit="halo")
else:
    pbar = list(subids)

out_rows = []
count = 0
t_start_all = time.time()
for sid in (pbar if use_tqdm else pbar):
    count += 1
    sid = int(sid)
    row = {'subhalo_id': sid,
           'stellar_mass_current': np.nan,
           'stellar_halfmass_radius_kpc': np.nan,
           'logsigma': np.nan}
    try:
        if sid not in soap_map:
            vprint(f"[{count}/{N}] subhalo {sid} not in SOAP catalogue; skipping")
            out_rows.append(row)
            if (args.flush_every > 0) and (count % args.flush_every == 0):
                pd.DataFrame(out_rows).to_csv(args.out, index=False)
                vprint(f"Flushed {len(out_rows)} rows to {args.out}")
            continue

        soap_idx = soap_map[sid]
        sg = SWIFTGalaxy(VIRTUAL_SNAPSHOT_FILE, SOAP(SOAP_CATALOGUE_FILE, soap_index=soap_idx))

        # read current stellar masses
        if hasattr(sg.stars, 'masses') and sg.stars.masses is not None:
            try:
                mcur = (np.asarray(sg.stars.masses.to(un.Msun).value, dtype=float)
                        if hasattr(sg.stars.masses, 'to') else np.asarray(sg.stars.masses, dtype=float))
                Mstar = float(np.sum(mcur))
            except Exception as e:
                vprint(f"[{count}/{N}] Warning reading masses for {sid}: {e}")
                Mstar = np.nan
        else:
            Mstar = np.nan

        # stellar half-mass radius: try swiftgalaxy attribute, fallback to None
        rhalf = np.nan
        try:
            # many versions expose sg.stellar_half_mass_radius as unyt quantity
            rhalf_attr = getattr(sg, 'stellar_half_mass_radius', None)
            if rhalf_attr is not None:
                # convert to kpc if possible
                try:
                    rhalf = float(rhalf_attr.to(un.kpc).value)
                except Exception:
                    # try numeric
                    rhalf = float(rhalf_attr)
        except Exception:
            # ignore and keep rhalf as nan
            rhalf = np.nan

        # compute logsigma: log10(Mstar) - 1.5*log10(rhalf)
        logsigma = np.nan
        if np.isfinite(Mstar) and Mstar > 0 and np.isfinite(rhalf) and rhalf > 0:
            logsigma = np.log10(Mstar) - 1.5 * np.log10(rhalf)

        row['stellar_mass_current'] = Mstar
        row['stellar_halfmass_radius_kpc'] = rhalf
        row['logsigma'] = logsigma

        out_rows.append(row)

        if args.flush_every > 0 and (count % args.flush_every == 0):
            pd.DataFrame(out_rows).to_csv(args.out, index=False)
            vprint(f"[{count}/{N}] Flushed {len(out_rows)} rows to {args.out}")

    except Exception as e:
        vprint(f"[{count}/{N}] ERROR processing subhalo {sid}: {type(e).__name__} {e}")
        vprint(traceback.format_exc())
        out_rows.append(row)  # append NaNs row so index order remains
    finally:
        # free fast
        try:
            del sg
        except Exception:
            pass

# final write
pd.DataFrame(out_rows).to_csv(args.out, index=False)
t_total = time.time() - t_start_all
print(f"Wrote: {args.out} ({len(out_rows)} rows) in {t_total:.1f}s")