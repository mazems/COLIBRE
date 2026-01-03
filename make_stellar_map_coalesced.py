#!/usr/bin/env python3
"""
make_stellar_map_coalesced.py

Coalesced reader (fixed units + better diagnostic)

Outputs CSV with fields:
  subhalo_id,
  stellar_mass_current   (particle-level sum, OUTPUT in 1e10 Msun),
  stellar_halfmass_radius_kpc  (SOAP ExclusiveSphere/50kpc/HalfMassRadiusStars),
  logsigma  = log10(M_star_SOAP_in_Msun) - 1.5 * log10(r_half_kpc)

Usage:
  python3 make_stellar_map_coalesced_fixed.py --ucmg ucmg_ids.csv --mapping ucmr_particle_index_mapping.npz \
    --snapshot /mnt/.../colibre_with_SOAP_membership_0127.hdf5 \
    --soap /mnt/.../halo_properties_0127.hdf5 --out relic_all_coalesced.csv
  or run with --test-subhalo 6041
"""
from __future__ import annotations
import os, sys, argparse, math, time
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from collections import defaultdict

# Defaults (adjust if needed)
DEFAULT_MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN'
DEFAULT_SOAP  = os.path.join(DEFAULT_MODEL_DIR, 'SOAP-HBT', 'halo_properties_0127.hdf5')
DEFAULT_SNAP  = os.path.join(DEFAULT_MODEL_DIR, 'SOAP-HBT', 'colibre_with_SOAP_membership_0127.hdf5')
DEFAULT_MAP   = '/home/mzemsch/COLIBRE-analysis/ucmg_particle_index_mapping.npz'

SCALE_OUT = 1e10   # output stellar_mass_current in 1e10 Msun
FLUSH_EVERY = 200
CHUNK_READ_LIMIT = 5_000_000   # max contiguous slice length to read at once (tunable)
DEFAULT_H = 0.681

# SOAP unit conversion factors you provided earlier
Mu = 1.988e43 / 1.989e33   # SOAP raw mass * Mu -> Msun (≈ 1e10)
Lu = 3.086e24 / 3.086e24   # SOAP raw length * Lu -> Mpc (here equals 1; keep for clarity)

def vprint(*a, **k):
    print(*a, **k, flush=True)

def build_soap_row_map_chunked(soap_path, wanted_set, chunk=1_000_000):
    """Chunk-scan HaloCatalogueIndex to find row indices for wanted subhalo IDs."""
    mapping = {}
    with h5py.File(soap_path, 'r') as f:
        # find canonical index dataset
        cand = ['InputHalos/HaloCatalogueIndex','HaloCatalogueIndex','InputHalos/HaloCatalogue_Index']
        idx_path = None
        for c in cand:
            if c in f:
                idx_path = c
                break
        if idx_path is None:
            # minor fallback search
            for name in f:
                if 'halocatalogueindex' in name.lower():
                    idx_path = name
                    break
        if idx_path is None:
            raise SystemExit("SOAP: HaloCatalogueIndex not found.")
        idx_ds = f[idx_path]
        nrows = int(idx_ds.shape[0])
        start = 0
        while start < nrows and len(mapping) < len(wanted_set):
            stop = min(nrows, start + chunk)
            chunk_arr = np.array(idx_ds[start:stop], dtype=np.int64)
            for rel, val in enumerate(chunk_arr):
                sid = int(val)
                if sid in wanted_set and sid not in mapping:
                    mapping[sid] = start + rel
            start = stop
    return mapping

def read_soap_values_for_rows(soap_path, row_map, h_val=DEFAULT_H):
    """
    Read ExclusiveSphere/50kpc/StellarMass and HalfMassRadiusStars for given SOAP rows.
    Apply SOAP conversions:
      - StellarMass_raw * (mf from attrs if any) * Mu -> Msun
      - HalfMassRadiusStars_raw * (rf from attrs if any) * Lu * 1e3 -> kpc
    """
    out = {}
    if len(row_map) == 0:
        return {}
    rows = [row_map[k] for k in row_map]
    subs = [int(k) for k in row_map]
    with h5py.File(soap_path, 'r') as f:
        mass_path = "ExclusiveSphere/50kpc/StellarMass"
        r_path    = "ExclusiveSphere/50kpc/HalfMassRadiusStars"
        if mass_path not in f or r_path not in f:
            raise SystemExit("SOAP missing required datasets.")
        m_ds = f[mass_path]
        r_ds = f[r_path]

        # detect factor from dataset attrs (conservative)
        def mass_attr_factor(ds):
            for k in ('Units','units','UNIT','unit'):
                if k in ds.attrs:
                    u = str(ds.attrs[k]).lower()
                    if '1e10' in u:
                        fac = 1e10
                        if 'h' in u:
                            fac = fac / h_val
                        return fac
                    if 'msun/h' in u:
                        return 1.0 / h_val
                    if 'msun' in u:
                        return 1.0
            return 1.0

        def radius_attr_factor(ds):
            for k in ('Units','units','UNIT','unit'):
                if k in ds.attrs:
                    u = str(ds.attrs[k]).lower()
                    if 'kpc' in u:
                        return 1.0
                    if 'pc' in u:
                        return 1e-3
                    if 'mpc' in u or 'm' in u and 'pc' in u:  # defensive
                        return 1e3
            return 1.0

        mf = mass_attr_factor(m_ds)
        rf = radius_attr_factor(r_ds)

        # vector read rows
        mvals = np.array(m_ds[rows], dtype=float) * mf   # now either Msun or still "raw" depending on attrs
        rvals = np.array(r_ds[rows], dtype=float) * rf   # now either kpc or still "raw"

        # **Apply SOAP canonical conversions**:
        # According to SOAP docs / your note: StellarMass raw -> multiply by Mu to get Msun
        # and HalfMassRadiusStars stored as a * L -> convert to kpc with Lu * 1e3 (and a=1 at z=0).
        # If mass_attr_factor already moved units to Msun (mf != 1), we **do not** double-apply Mu.
        if np.isfinite(np.nanmedian(np.abs(mvals))) and (mf == 1.0):
            # assume raw SOAP mass needs Mu -> Msun
            vprint("[SOAP] applying Mu conversion to StellarMass (raw -> Msun).")
            mvals = mvals * Mu

        # For radius: if rf == 1.0 (no kpc attr) assume raw stored in a*L where L is Mpc -> convert to kpc
        if rf == 1.0:
            vprint("[SOAP] applying Lu * 1e3 conversion to HalfMassRadiusStars (raw a*L -> kpc).")
            rvals = rvals * Lu * 1e3

        for sid, m, r in zip(subs, mvals, rvals):
            out[int(sid)] = (float(m) if np.isfinite(m) else float('nan'),
                             float(r) if np.isfinite(r) else float('nan'))
    return out

def accumulate_particle_sums_coalesced(snapshot_h5, mapping_npz, wanted_ids, h_val=DEFAULT_H, chunk_limit=CHUNK_READ_LIMIT):
    """
    Coalesced accumulation (vectorized):
      - loads NPZ mapping entries for the wanted_ids
      - builds arrays all_indices, owners
      - sorts by indices and performs grouped contiguous reads of Masses
      - uses np.bincount to accumulate per-owner within each read chunk
    Returns dict subhalo_id -> total_mass_Msun (float).
    """
    wanted = list(int(x) for x in wanted_ids)
    npz = np.load(mapping_npz, allow_pickle=False)

    # Build lists of indices & owners only for wanted halos
    idx_blocks = []
    owner_blocks = []
    for sid in wanted:
        sk = str(sid)
        if sk in npz.files:
            idxs = npz[sk].astype(np.int64)
        else:
            # fallback: try integer-string keys
            idxs = None
            for k in npz.files:
                try:
                    if int(k) == sid:
                        idxs = npz[k].astype(np.int64)
                        break
                except Exception:
                    continue
        if idxs is None or idxs.size == 0:
            continue
        idx_blocks.append(idxs)
        owner_blocks.append(np.full(idxs.shape, sid, dtype=np.int64))

    if len(idx_blocks) == 0:
        return {int(s): float('nan') for s in wanted}

    all_idx = np.concatenate(idx_blocks)
    all_owner = np.concatenate(owner_blocks)

    if all_idx.size == 0:
        return {int(s): 0.0 for s in wanted}

    order = np.argsort(all_idx)
    sidx = all_idx[order]
    sowner = all_owner[order]

    totals = {}
    # init totals to 0 for each wanted (so we always return key)
    for sid in wanted:
        totals[int(sid)] = 0.0

    with h5py.File(snapshot_h5, 'r') as f:
        if 'PartType4' not in f:
            raise SystemExit("Snapshot missing PartType4")
        p4 = f['PartType4']
        if 'Masses' in p4:
            mname = 'Masses'
        elif 'masses' in p4:
            mname = 'masses'
        else:
            raise SystemExit("Snapshot missing Masses dataset")
        mass_ds = p4[mname]

        # compute particle mass factor -> multiply raw dataset -> Msun
        pfac = 1.0
        try:
            # Prefer 'Conversion factor to CGS' attr if present (g / raw unit)
            if 'Conversion factor to CGS (not including cosmological corrections)' in mass_ds.attrs:
                conv = float(mass_ds.attrs['Conversion factor to CGS (not including cosmological corrections)'])
                pfac = conv / 1.989e33  # grams -> Msun
            elif 'Conversion factor to CGS' in mass_ds.attrs:
                conv = float(mass_ds.attrs['Conversion factor to CGS'])
                pfac = conv / 1.989e33
            else:
                # look for generic Units attr that might contain 1e10 etc. Fall back to 1.0
                for key in ('Units','units','unit','UNITS'):
                    if key in mass_ds.attrs:
                        u = str(mass_ds.attrs[key]).lower()
                        if '1e10' in u:
                            fac = 1e10
                            if 'h' in u:
                                fac = fac / h_val
                            pfac = fac
                        elif 'msun' in u:
                            pfac = 1.0
        except Exception:
            pfac = 1.0

        # iterate contiguous runs in sidx (but avoid super-large read slices)
        n = sidx.size
        pos = 0
        while pos < n:
            # start of run
            run_start_idx = sidx[pos]
            run_end_pos = pos + 1
            last = run_start_idx
            # grow run while contiguous and below chunk_limit
            while run_end_pos < n and (sidx[run_end_pos] == last + 1) and ((sidx[run_end_pos] - run_start_idx + 1) <= chunk_limit):
                last = sidx[run_end_pos]
                run_end_pos += 1
            run_stop_idx = sidx[run_end_pos - 1]
            # read slice
            if run_stop_idx < run_start_idx:
                pos = run_end_pos
                continue
            # read the contiguous slice once
            read_slice = np.array(mass_ds[run_start_idx:run_stop_idx + 1], dtype=float) * pfac
            local_positions = sidx[pos:run_end_pos] - run_start_idx
            local_vals = read_slice[local_positions]  # values corresponding to owners
            owners_chunk = sowner[pos:run_end_pos]

            # vectorized per-owner sum using np.bincount
            uniq_owners, inv = np.unique(owners_chunk, return_inverse=True)
            sums = np.bincount(inv, weights=local_vals, minlength=uniq_owners.size)
            # add sums to totals dict
            for owner_id, s in zip(uniq_owners, sums):
                totals[int(owner_id)] = totals.get(int(owner_id), 0.0) + float(s)

            pos = run_end_pos

    # ensure all wanted keys exist; if a wanted had no particles return NaN (or 0.0)
    out = {}
    for sid in wanted:
        val = totals.get(int(sid), None)
        if val is None:
            out[int(sid)] = float('nan')
        else:
            out[int(sid)] = float(val)
    return out

def safe_write_csv_rows(outfn, rows, append=False):
    df = pd.DataFrame(rows)
    if not append:
        df.to_csv(outfn, index=False)
    else:
        df.to_csv(outfn, index=False, mode='a', header=False)
    try:
        with open(outfn, 'ab') as fh:
            fh.flush(); os.fsync(fh.fileno())
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ucmg', default='ucmg_ids.csv')
    parser.add_argument('--soap', default=DEFAULT_SOAP)
    parser.add_argument('--snapshot', default=DEFAULT_SNAP)
    parser.add_argument('--mapping', default=DEFAULT_MAP)
    parser.add_argument('--out', default='relic_all_coalesced_fixed.csv')
    parser.add_argument('--test-subhalo', type=int, default=None)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    args = parser.parse_args()

    # read subhalo ids
    if args.test_subhalo is not None:
        wanted = np.array([int(args.test_subhalo)], dtype=np.int64)
    else:
        if not os.path.exists(args.ucmg):
            raise SystemExit("ucmg file not found: " + args.ucmg)
        df = pd.read_csv(args.ucmg)
        if 'subhalo_id' in df.columns:
            wanted = np.array(df['subhalo_id'], dtype=np.int64)
        else:
            wanted = np.array(df.iloc[:,0], dtype=np.int64)

    vprint(f"Will process {len(wanted)} subhalos (test={args.test_subhalo is not None})")
    wanted_set = set(int(x) for x in wanted)

    # map SOAP rows (chunked)
    vprint("Mapping requested subhalos -> SOAP rows (chunked scan)...")
    soap_row_map = build_soap_row_map_chunked(args.soap, wanted_set)
    vprint(f"Found SOAP rows for {len(soap_row_map)} / {len(wanted)} requested subhalos")

    # read SOAP values for those rows (apply Mu/Lu conversions)
    vprint("Reading SOAP StellarMass & HalfMassRadiusStars for requested rows (applying Mu/Lu conversions)...")
    soap_vals = read_soap_values_for_rows(args.soap, soap_row_map, h_val=args.h)

    # coalesce and sum particle masses
    vprint("Coalescing particle indices and summing particle-level masses (contiguous reads)...")
    t0 = time.time()
    particle_sums = accumulate_particle_sums_coalesced(args.snapshot, args.mapping, wanted, h_val=args.h)
    vprint(f"Particle-sums completed in {time.time()-t0:.1f}s (computed for {len(particle_sums)} halos)")

    # TEST-MODE: improved diagnostic (Msun + 1e10Msun) and logsigma computed from SOAP mass in Msun
    if args.test_subhalo is not None:
        sid = int(args.test_subhalo)
        Mpart = particle_sums.get(sid, float('nan'))    # Msun
        Mpart_1e10 = (Mpart / SCALE_OUT) if np.isfinite(Mpart) else float('nan')
        Msoap_msun, rhalf = soap_vals.get(sid, (float('nan'), float('nan')))  # Msoap_msun is in Msun after conversion
        vprint("Diagnostic (coalesced) for subhalo:", sid)
        vprint("  particle_sum (Msun):", Mpart)
        vprint("  particle_sum (1e10 Msun):", Mpart_1e10)
        vprint("  SOAP ExclusiveSphere/50kpc/StellarMass (Msun):", Msoap_msun)
        vprint("  SOAP ExclusiveSphere/50kpc/HalfMassRadiusStars (kpc):", rhalf)
        if (np.isfinite(Msoap_msun) and np.isfinite(rhalf) and rhalf>0 and Msoap_msun>0):
            logsigma = math.log10(float(Msoap_msun)) - 1.5*math.log10(float(rhalf))
            vprint("  logsigma (SOAP):", logsigma)
        else:
            vprint("  logsigma (SOAP):", float('nan'))
        return

    # assemble CSV rows and write periodically
    rows_buf = []
    written = 0
    outfn = args.out
    pbar = tqdm(wanted, desc="halos", file=sys.stdout, ncols=100)
    for i, sid in enumerate(pbar, start=1):
        sid = int(sid)
        Mpart = particle_sums.get(sid, float('nan'))  # Msun
        Mpart_out = float(Mpart / SCALE_OUT) if np.isfinite(Mpart) else float('nan')  # in 1e10 Msun
        Msoap, rhalf = soap_vals.get(sid, (float('nan'), float('nan')))
        if (not np.isfinite(Msoap)) or (not np.isfinite(rhalf)) or (rhalf <= 0) or (Msoap <= 0):
            logs = float('nan')
        else:
            try:
                logs = math.log10(float(Msoap)) - 1.5*math.log10(float(rhalf))
            except Exception:
                logs = float('nan')
        rows_buf.append({
            'subhalo_id': sid,
            'stellar_mass_current': Mpart_out,
            'stellar_halfmass_radius_kpc': float(rhalf) if np.isfinite(rhalf) else float('nan'),
            'logsigma': float(logs) if np.isfinite(logs) else float('nan')
        })
        if (i % FLUSH_EVERY) == 0 or i == len(wanted):
            safe_write_csv_rows(outfn, rows_buf, append=(written>0))
            written += len(rows_buf)
            rows_buf = []
            vprint(f"Flushed {written} rows to {outfn}")

    vprint(f"Done. Wrote {written} rows to {outfn}")

if __name__ == "__main__":
    main()