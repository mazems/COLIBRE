from __future__ import annotations
import os, sys, argparse, math, errno
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

# -------------
DEFAULT_MODEL_DIR = '/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN'
DEFAULT_SOAP = os.path.join(DEFAULT_MODEL_DIR, 'SOAP-HBT', 'halo_properties_0127.hdf5')
DEFAULT_SNAP = os.path.join(DEFAULT_MODEL_DIR, 'SOAP-HBT', 'colibre_with_SOAP_membership_0127.hdf5')
DEFAULT_MAPPING = "/home/mzemsch/COLIBRE-analysis/ucmg_particle_index_mapping.npz"
SCALE_OUT = 1e10   # convert Msun -> 1e10 Msun for output column
FLUSH_EVERY = 200
CHUNK_SIZE = 1_000_000
DEFAULT_H = 0.681
Mu = 1.988e43 / 1.989e33   # factor to convert SOAP mass raw -> Msun (≈ 1e10)
Lu = 3.086e24 / 3.086e24   # factor to convert SOAP length raw -> cMpc (user-specified; keep as your note)
# -------------

def vprint(*a, **k):
    print(*a, **k, flush=True)

# helper: convert SOAP StellarMass raw -> Msun
def soap_mass_to_msun(raw_val, mu=Mu, ds=None, ds_name="SOAP_mass"):
    """
    Convert raw SOAP StellarMass value -> Msun using Mu.
    If dataset `ds` available and has units attribute we print it.
    """
    try:
        if ds is not None:
            for k in ('Units','units','UNIT','unit'):
                if k in ds.attrs:
                    vprint(f"[SOAP UNITS] {ds_name} units attr: {ds.attrs[k]!r}")
                    break
    except Exception:
        pass

    try:
        return float(raw_val) * float(mu)
    except Exception:
        return float('nan')

# helper: convert SOAP HalfMassRadiusStars raw -> kpc
def soap_rhalf_to_kpc(raw_val, Lu_val=Lu, a_scale=1.0, comov_to_phys=1.0, ds=None, ds_name="SOAP_rhalf"):
    """
    Convert SOAP radius stored as 'a * L' to physical kpc.
      raw_val  : stored value in units of (a * L_unit)
      Lu_val   : conversion factor such that raw * Lu_val -> L_unit in Mpc (or adjust if different)
      a_scale  : cosmological scale factor a (1.0 at z=0)
      comov_to_phys : factor to convert comoving -> physical (1.0 at z=0)
    Final result is in kpc.
    """
    try:
        if ds is not None:
            for k in ('Units','units','UNIT','unit'):
                if k in ds.attrs:
                    vprint(f"[SOAP UNITS] {ds_name} units attr: {ds.attrs[k]!r}")
                    break
    except Exception:
        pass

    try:
        # raw is a * L_unit. Convert raw -> Mpc using Lu_val and comoving->physical handling.
        # According to your note: r50 = raw * Lu * comov_to_physical_length * 1e3
        # We include a_scale explicitly: if raw already includes a, divide by a to get L.
        r_mpc = float(raw_val) * float(Lu_val) * float(comov_to_phys) / float(a_scale)
        r_kpc = r_mpc * 1e3
        return r_kpc
    except Exception:
        return float('nan')

def build_soap_map_chunked(soap_path, wanted_ids, chunk_size=CHUNK_SIZE):
    """Return dict subhalo_id -> soap_row (int). Memory-safe chunk scan of HaloCatalogueIndex."""
    wanted = set(int(x) for x in wanted_ids)
    mapping = {}
    if len(wanted) == 0:
        return mapping
    with h5py.File(soap_path, 'r') as f:
        # find canonical index path
        cand_names = ['InputHalos/HaloCatalogueIndex', 'HaloCatalogueIndex', 'InputHalos/HaloCatalogue_Index']
        idx_path = None
        for c in cand_names:
            if c in f:
                idx_path = c
                break
        if idx_path is None:
            # fallback search
            for name in f:
                if 'halocatalogueindex' in name.lower():
                    idx_path = name
                    break
        if idx_path is None:
            raise SystemExit("SOAP missing HaloCatalogueIndex (cannot map subhalo ids).")
        idx_ds = f[idx_path]
        n = int(idx_ds.shape[0])
        start = 0
        while start < n and len(mapping) < len(wanted):
            stop = min(n, start + chunk_size)
            chunk = np.array(idx_ds[start:stop], dtype=np.int64)
            # compare
            for rel, val in enumerate(chunk):
                sid = int(val)
                if sid in wanted and sid not in mapping:
                    mapping[sid] = start + rel
            start = stop
    return mapping

def read_soap_values_for_rows(soap_path, row_map, h_val=DEFAULT_H):
    """Given subhalo->row map, read ExclusiveSphere/50kpc/StellarMass and HalfMassRadiusStars for those rows.
       Returns dict subhalo_id->(M_msun, r_kpc)"""
    out = {}
    if len(row_map) == 0:
        return {int(k): (float('nan'), float('nan')) for k in row_map.keys()}
    rows = [row_map[k] for k in row_map]
    subs = [int(k) for k in row_map]
    with h5py.File(soap_path, 'r') as f:
        mass_path = "ExclusiveSphere/50kpc/StellarMass"
        r_path = "ExclusiveSphere/50kpc/HalfMassRadiusStars"
        if mass_path not in f or r_path not in f:
            raise SystemExit(f"SOAP missing required datasets: {mass_path} or {r_path}")
        m_ds = f[mass_path]
        r_ds = f[r_path]

        # Read raw stored values for only requested rows (vectorized)
        raw_mvals = np.array(m_ds[rows], dtype=float)
        raw_rvals = np.array(r_ds[rows], dtype=float)

        # Convert using explicit SOAP conversion helpers (Mu/Lu) — these produce Msun and kpc
        for sid, raw_m, raw_r in zip(subs, raw_mvals, raw_rvals):
            try:
                M_msun = soap_mass_to_msun(raw_m, mu=Mu, ds=m_ds, ds_name=mass_path)
            except Exception:
                M_msun = float('nan')
            try:
                r_kpc = soap_rhalf_to_kpc(raw_r, Lu_val=Lu, a_scale=1.0, comov_to_phys=1.0, ds=r_ds, ds_name=r_path)
            except Exception:
                r_kpc = float('nan')
            out[int(sid)] = (float(M_msun), float(r_kpc))
    return out

def sum_particle_masses_from_mapping(snapshot_path, mapping_npz, wanted_ids, h_val=DEFAULT_H):
    """Sum PartType4/Masses for each wanted_id using mapping dict loaded from npz.
       Returns dict subhalo_id -> total_mass_Msun (float)."""
    # load mapping npz minimally
    npz = np.load(mapping_npz, allow_pickle=False)
    results = {}
    with h5py.File(snapshot_path, 'r') as f:
        if 'PartType4' not in f:
            raise SystemExit("Snapshot missing PartType4.")
        p4 = f['PartType4']
        if 'Masses' in p4:
            mname = 'Masses'
        elif 'masses' in p4:
            mname = 'masses'
        else:
            raise SystemExit("Snapshot PartType4 missing 'Masses' dataset.")
        mass_ds = p4[mname]

        # minimal particle-mass unit handling (dataset->Msun factor)
        def mass_factor(ds):
            for k in ('Units','units','unit','UNITS'):
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
        pf = mass_factor(mass_ds)

        for sid in wanted_ids:
            sid = int(sid)
            idxs = None
            sk = str(sid)
            print("Analysing subhalo", sid)
            if sk in npz.files:
                idxs = npz[sk].astype(np.int64)
            else:
                # try integer-keyed entries (lazy)
                for k in npz.files:
                    try:
                        if int(k) == sid:
                            idxs = npz[k].astype(np.int64)
                            break
                    except Exception:
                        pass
            if idxs is None:
                results[sid] = float('nan')
                continue
            if idxs.size == 0:
                results[sid] = 0.0
                continue
            try:
                vals = np.array(mass_ds[idxs], dtype=float)
                total = float(np.sum(vals) * pf)
                results[sid] = total
            except Exception as e:
                vprint(f"Warning: summing particles for {sid} failed: {e}")
                results[sid] = float('nan')
    return results

def safe_write_csv_rows(outfn, rows, append=False):
    """Write rows list-of-dicts to CSV; append if requested; fsync file to ensure saved."""
    df = pd.DataFrame(rows)
    if not append:
        df.to_csv(outfn, index=False)
    else:
        df.to_csv(outfn, index=False, mode='a', header=False)
    try:
        with open(outfn, 'ab') as fh:
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Simple strict stellar map builder")
    parser.add_argument('--ucmg', default='ucmg_ids.csv')
    parser.add_argument('--soap', default=DEFAULT_SOAP)
    parser.add_argument('--snapshot', default=DEFAULT_SNAP)
    parser.add_argument('--mapping', default=DEFAULT_MAPPING)
    parser.add_argument('--out', default='stellar_map_simple.csv')
    parser.add_argument('--test-subhalo', type=int, default=None)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--start', type=int, default=None,
                    help='Zero-based start index into ucmg list (use with --count)')
    parser.add_argument('--count', type=int, default=None,
                    help='Number of subhalos to process starting at --start')
    args = parser.parse_args()

    # read ids
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

    if args.start is not None:
        if args.count is None:
            raise SystemExit("If --start is supplied you must also supply --count")
        start = int(args.start)
        count = int(args.count)
        if start < 0:
            raise SystemExit("start index must be >= 0")
        wanted = np.array(wanted, dtype=np.int64)  # ensure numpy array
        # guard against slicing past end
        wanted = wanted[start:start+count]
        if wanted.size == 0:
            vprint(f"No IDs in the requested slice start={start} count={count}; exiting.")
            return  # or raise SystemExit if inside top-level function        

    vprint(f"Will process {len(wanted)} subhalos (test={args.test_subhalo is not None})")

    # 1) build SOAP row map (chunked)
    vprint("Mapping requested subhalos -> SOAP rows (chunked scan)...")
    soap_row_map = build_soap_map_chunked(args.soap, wanted)
    vprint(f"Found SOAP rows for {len(soap_row_map)} / {len(wanted)} requested subhalos")

    # 2) read SOAP values for those rows (vector read + conversions)
    vprint("Reading SOAP StellarMass & HalfMassRadiusStars for requested rows (applying Mu/Lu conversions)...")
    soap_vals = read_soap_values_for_rows(args.soap, soap_row_map, h_val=args.h)

    # 3) sum particle masses using mapping npz (particle masses -> Msun)
    vprint("Summing PartType4/Masses via mapping NPZ (particle-level current stellar mass -> Msun)...")
    particle_sums = sum_particle_masses_from_mapping(args.snapshot, args.mapping, wanted, h_val=args.h)

    # 4) assemble outputs and write (flush periodically)
    outfn = args.out
    rows_buffer = []
    written = 0
    pbar = tqdm(wanted, desc="halos", file=sys.stdout, ncols=100)
    try:
        for i, sid in enumerate(pbar, start=1):
            sid = int(sid)
            Mpart = particle_sums.get(sid, float('nan'))        # Msun
            # output requested in 1e10 Msun
            Mout = float(Mpart) if np.isfinite(Mpart) else float('nan')
            Msoap, rhalf = soap_vals.get(sid, (float('nan'), float('nan')))  # Msoap in Msun, rhalf in kpc
            if (not np.isfinite(Msoap)) or (not np.isfinite(rhalf)) or (rhalf <= 0) or (Msoap <= 0):
                logs = float('nan')
            else:
                try:
                    logs = math.log10(float(Msoap)) - 1.5 * math.log10(float(rhalf))
                except Exception:
                    logs = float('nan')
            rows_buffer.append({
                'subhalo_id': sid,
                'stellar_mass_current': Mout,
                'stellar_halfmass_radius_kpc': float(rhalf) if np.isfinite(rhalf) else float('nan'),
                'logsigma': float(logs) if np.isfinite(logs) else float('nan')
            })

            if (i % FLUSH_EVERY) == 0 or i == len(wanted):
                append = (written > 0)
                safe_write_csv_rows(outfn, rows_buffer, append=append)
                written += len(rows_buffer)
                rows_buffer = []
                vprint(f"Flushed {written} rows to {outfn}")
    except Exception as e:
        vprint("ERROR during processing loop:", e)
        if rows_buffer:
            try:
                safe_write_csv_rows(outfn, rows_buffer, append=(written>0))
                vprint("Saved partial results before exiting.")
            except Exception:
                vprint("Failed to save partial results.")
        raise

    vprint(f"Done. Wrote {written} rows to {outfn}")

if __name__ == "__main__":
    main()