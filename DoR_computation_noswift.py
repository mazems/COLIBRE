#!/usr/bin/env python3
"""
compute_sfh_times_sample.py

Compute SFH time diagnostics and DoR terms for a small sample (default) or for ALL galaxies
with --compute-all. See --help.
"""
import argparse, os, sys, random, math
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as au

# cosmology params from COLIBRE 2025
COLIBRE_H = 0.681
COLIBRE_OMEGAM = 0.306
COLIBRE_OMEGAL = 0.693922

COSMO_PARAMS = dict(h=COLIBRE_H, omegam=COLIBRE_OMEGAM, omegal=COLIBRE_OMEGAL)
cosmo_colibre = FlatLambdaCDM(H0=100.0 * COLIBRE_H, Om0=COLIBRE_OMEGAM if 'COLIBRE_OMEGAM' in globals() else COLIBRE_OMEGAM)

# NOTE: the above line tries to use your COLIBRE_OMEGAM constant; if editing, keep FlatLambdaCDM initialization consistent.
# Simpler: reinitialize correctly below to avoid name issues:
cosmo_colibre = FlatLambdaCDM(H0=100.0 * COLIBRE_H, Om0=COLIBRE_OMEGAM)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--basedir",
                   default="/mnt/su3-pro/clagos/COLIBRE/Runs/L200_m6/Thermal/ProcessedData",
                   help="base directory for GalaxyProperties / SFH / time files")
    p.add_argument("--relic_csv", default="relicness_merged_allabove9p9.csv",
                   help="UCMG CSV containing subhalo_id (used to sample ids)")
    p.add_argument("--galfile", default=None,
                   help="GalaxyProperties file (if omitted uses default inside basedir)")
    p.add_argument("--timefile", default=None,
                   help="look_back_time_info file (if omitted uses default inside basedir)")
    p.add_argument("--sfhfile", default=None,
                   help="Mstar_SFH file (if omitted uses default inside basedir)")
    p.add_argument("--n", type=int, default=2, help="number of random ids to test (sample mode)")
    p.add_argument("--seed", type=int, default=12345, help="random seed")
    p.add_argument("--lookback-z2", type=float, default=10.3,
                   help="lookback time (Gyr) corresponding to z=2 (used for f_Mz2). Default 10.3 Gyr.")
    p.add_argument("--show_matches", action="store_true", help="print matching rows from relic CSV for sampled ids")

    # compute-all related
    p.add_argument("--compute-all", action="store_true",
                   help="Compute times and DoR for ALL galaxies (writes output CSV).")
    p.add_argument("--sfh-index", default=None,
                   help="Optional text file with one halo id per SFH line (same order as SFH file).")
    p.add_argument("--out-csv", default=None,
                   help="Output CSV path for --compute-all. Default: <basedir>/sfh_times_all.csv")
    p.add_argument("--term3-ref", default="tfin",
                   help="Which span to use for term3: one of tfin/t90/t95/t998 (default tfin).")
    p.add_argument("--start", type=int, default=0,
                   help="If computing part of the full sample, start index (inclusive) of galaxy ordering (0-based).")
    p.add_argument("--end", type=int, default=None,
                   help="If computing part of the full sample, end index (exclusive) of galaxy ordering (0-based).")
    p.add_argument("--chunk-write", type=int, default=5000,
                   help="Number of rows to buffer before flushing to disk (helps memory).")
    return p.parse_args()


def count_lines(path):
    # robust line count for binary/text file
    c = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            c += chunk.count(b"\n")
    return c


def load_sfh_index(path):
    # load one integer id per line; allow whitespace, empty lines skipped
    arr = np.loadtxt(path, dtype=np.int64, ndmin=1)
    return arr


def safe_load_galprops(path):
    # try numpy.loadtxt first (fast); fallback to pandas
    try:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        return df.to_numpy()


def read_single_sfh_line(sfh_path, line_number):
    """
    Read a single line (1-based) from SFH file and return numpy 1D array floats.
    This reads file line-by-line until the requested line (efficient for small N).
    """
    with open(sfh_path, "r") as fh:
        for i, line in enumerate(fh, start=1):
            if i == line_number:
                toks = line.strip().replace(",", " ").split()
                vals = [float(x) if (x != "" and x.lower() != "nan") else 0.0 for x in toks]
                return np.array(vals, dtype=float)
    raise IndexError(f"SFH file {sfh_path} has no line {line_number}")


def quantile_from_sfh_lookback(time_bins, mass_bins, q):
    tb = np.asarray(time_bins, dtype=float)
    m = np.nan_to_num(mass_bins, nan=0.0, posinf=0.0, neginf=0.0)
    if m.size == 0 or np.sum(m) <= 0:
        return np.nan
    order = np.argsort(tb)[::-1]
    t_sorted = tb[order]
    m_sorted = m[order]
    if t_sorted.size == 1:
        return float(t_sorted[0])
    dt = np.diff(t_sorted)
    dt[dt == 0] = np.min(dt[dt > 0]) if np.any(dt > 0) else 1e-6
    edges = np.empty(t_sorted.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (t_sorted[:-1] + t_sorted[1:])
    edges[0] = t_sorted[0] + 0.5 * (t_sorted[0] - edges[1])
    edges[-1] = t_sorted[-1] - 0.5 * (edges[-2] - t_sorted[-1])
    mass_edges = np.concatenate(([0.0], np.cumsum(m_sorted)))
    total = mass_edges[-1]
    if total <= 0:
        return np.nan
    cf_edges = mass_edges / total
    qf = float(q)
    if qf <= cf_edges[0]:
        return float(edges[0])
    if qf >= cf_edges[-1]:
        return float(edges[-1])
    return float(np.interp(qf, cf_edges, edges))


def compute_all_times(time_bins, mass_bins, lookback_z2=10.3, cosmic_age=None):
    tb = np.asarray(time_bins, dtype=float)
    mass = np.nan_to_num(mass_bins, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(mass))
    out = {"total_formed_mass_Msun": total}
    if total <= 0.0:
        for k in ("t_start", "t50", "t50_span", "t75", "t75_span", "t90", "t90_span", "t95", "t95_span", "t998",
                  "t998_span", "tfin", "tfin_span"):
            out[k] = float("nan")
        out["f_Mz2"] = float("nan")
        out["term1"] = out["term2"] = out["term3"] = float("nan")
        return out
    if cosmic_age is None:
        cosmic_age_use = cosmo_colibre.age(0).to(au.Gyr).value
    else:
        cosmic_age_use = float(cosmic_age)
    nonzero_idx = np.where(mass > 0)[0]
    if nonzero_idx.size == 0:
        t_start_lb = np.nan
    else:
        t_start_lb = float(np.max(tb[nonzero_idx]))
    out["t_start"] = (cosmic_age_use - t_start_lb) if np.isfinite(t_start_lb) else float("nan")
    t50_lb = quantile_from_sfh_lookback(tb, mass, 0.50)
    t75_lb = quantile_from_sfh_lookback(tb, mass, 0.75)
    t90_lb = quantile_from_sfh_lookback(tb, mass, 0.90)
    t95_lb = quantile_from_sfh_lookback(tb, mass, 0.95)
    t998_lb = quantile_from_sfh_lookback(tb, mass, 0.998)
    tfin_lb = quantile_from_sfh_lookback(tb, mass, 0.9999)
    out["t50"] = cosmic_age_use - t50_lb if np.isfinite(t50_lb) else float("nan")
    out["t75"] = cosmic_age_use - t75_lb if np.isfinite(t75_lb) else float("nan")
    out["t90"] = cosmic_age_use - t90_lb if np.isfinite(t90_lb) else float("nan")
    out["t95"] = cosmic_age_use - t95_lb if np.isfinite(t95_lb) else float("nan")
    out["t998"] = cosmic_age_use - t998_lb if np.isfinite(t998_lb) else float("nan")
    out["tfin"] = cosmic_age_use - tfin_lb if np.isfinite(tfin_lb) else float("nan")
    if np.isfinite(out["t_start"]):
        base = out["t_start"]
        for qname in ("t50", "t75", "t90", "t95", "t998", "tfin"):
            out[f"{qname}_span"] = float(out[qname] - base) if np.isfinite(out[qname]) else float("nan")
    else:
        for qname in ("t50_span", "t75_span", "t90_span", "t95_span", "t998_span", "tfin_span"):
            out[qname] = float("nan")
    order_desc = np.argsort(tb)[::-1]
    tb_desc = tb[order_desc]
    mass_desc = mass[order_desc]
    mask_z2 = (tb_desc >= lookback_z2)
    out["f_Mz2"] = float(np.sum(mass_desc[mask_z2]) / total) if total > 0 else float("nan")
    out["term1"] = out["term2"] = out["term3"] = float("nan")
    return out


def compute_dor_from_mass(tb_arr, mass_bins, out_spans, term3_ref, cosmic_age_use):
    """
    Compute term1/term2/term3 and dor from mass/time arrays and previously computed spans (out_spans).
    tb_arr are lookback centers (Gyr), cosmic_age_use is Gyr since BB.
    """
    tb_arr = np.asarray(tb_arr, dtype=float)
    mass_bins = np.nan_to_num(mass_bins, nan=0.0, posinf=0.0, neginf=0.0)
    total_formed = float(np.sum(mass_bins))
    if total_formed <= 0:
        return {"term1": float("nan"), "term2": float("nan"), "term3": float("nan"), "dor": float("nan")}
    # formation cosmic time for each bin (Gyr since BB)
    tform = float(cosmo_colibre.age(0).to(au.Gyr).value) - tb_arr
    # term1: fraction formed before z=2 (cosmic time of z=2)
    t_z2_gyr = cosmo_colibre.age(2.0).to(au.Gyr).value
    f_Mz2 = (np.sum(mass_bins[tform <= t_z2_gyr]) / total_formed) if (total_formed > 0) else float('nan')
    # term2: 0.5 / t75_span (fallback 1.0)
    t75_span = out_spans.get("t75_span", np.nan)
    term2 = 0.5 / t75_span if (t75_span is not None and np.isfinite(t75_span) and t75_span > 0.0) else 1.0
    # term3: uses selected span; t_universe_gyr = cosmic_age_use
    t_universe_gyr = float(cosmo_colibre.age(0).to(au.Gyr).value)
    span_map = {"tfin": out_spans.get("tfin_span", np.nan),
                "t90": out_spans.get("t90_span", np.nan),
                "t95": out_spans.get("t95_span", np.nan),
                "t998": out_spans.get("t998_span", np.nan)}
    span_val = span_map.get(term3_ref, span_map.get("tfin", np.nan))
    term3 = (0.7 + t_universe_gyr - span_val) / t_universe_gyr if np.isfinite(span_val) else float('nan')
    term1 = float(f_Mz2) if np.isfinite(f_Mz2) else float('nan')
    dor = float((term1 + term2 + term3) / 3.0) if (np.isfinite(term1) and np.isfinite(term2) and np.isfinite(term3)) else float('nan')
    return {"term1": term1, "term2": term2, "term3": term3, "dor": dor}


def main():
    args = parse_args()
    random.seed(args.seed)

    # derive paths
    galfile = args.galfile if args.galfile else os.path.join(args.basedir, "GalaxyProperties_sfrGE0_z0.0.txt")
    timefile = args.timefile if args.timefile else os.path.join(args.basedir, "look_back_time_info_circular_apertures_face_on_map_dt0.01_z0.0.txt")
    sfhfile = args.sfhfile if args.sfhfile else os.path.join(args.basedir, "Mstar_SFH_ap50ckpc_circular_apertures_face_on_map_dt0.01_z0.0.txt")
    relic_csv = args.relic_csv

    # existence checks (require these files)
    for p in (galfile, timefile, sfhfile):
        if not os.path.exists(p):
            print("Missing file:", p, file=sys.stderr)
            sys.exit(1)
    # relic_csv is only needed for sample mode; we won't error now if it doesn't exist
    relic_exists = os.path.exists(relic_csv)

    # load galaxy properties (small)
    gal_arr = safe_load_galprops(galfile)
    halo_ids = gal_arr[:, 0].astype(np.int64)
    idx_by_halo = {int(h): i for i, h in enumerate(halo_ids)}
    n_gal = len(halo_ids)
    print(f"Loaded GalaxyProperties rows: {n_gal}")

    # load time bins (needed for compute-all)
    tb = np.loadtxt(timefile).astype(float)
    nbins = tb.size
    print(f"Loaded {nbins} lookback bins (Gyr)")

    # SFH file line count & optional mapping
    num_sfh_lines = count_lines(sfhfile)
    print(f"SFH file has {num_sfh_lines} lines")

    line_by_halo = None
    if args.sfh_index:
        if not os.path.exists(args.sfh_index):
            print("sfh-index file not found:", args.sfh_index, file=sys.stderr)
            sys.exit(1)
        sfh_index = load_sfh_index(args.sfh_index)
        if sfh_index.size != num_sfh_lines:
            print("Warning: sfh-index length != number of SFH lines.")
        line_by_halo = {int(h): i + 1 for i, h in enumerate(sfh_index)}

    if line_by_halo is not None:
        mapping_mode = "sfh_index"
    elif num_sfh_lines == n_gal:
        mapping_mode = "row_match"
    else:
        mapping_mode = "sfh_only"

    print("SFH mapping mode:", mapping_mode)

    # If compute-all requested, require mapping_mode != sfh_only (explicit mapping or equal lengths)
    if args.compute_all and mapping_mode == "sfh_only":
        print("ERROR: SFH file length != GalaxyProperties and no --sfh-index provided.")
        sys.exit(1)

    # -------- compute-all mode --------
    if args.compute_all:
        outpath = args.out_csv if args.out_csv else os.path.join(args.basedir, "sfh_times_all.csv")
        start = int(args.start) if args.start is not None else 0
        end = int(args.end) if args.end is not None else n_gal
        end = min(end, n_gal)
        if start < 0 or start >= end:
            print("Invalid start/end range:", start, end)
            sys.exit(1)

        print(f"Compute-all: processing galaxy indices [{start}:{end}) -> output {outpath}")
        header_written = False
        buffer = []
        flush_every = int(args.chunk_write)

        for idx in range(start, end):
            sid = int(halo_ids[idx])
            if mapping_mode == "sfh_index":
                sfh_line = line_by_halo.get(sid, None)
                if sfh_line is None:
                    # no SFH for this halo in mapping file
                    # skip quietly (avoid terminal flood)
                    continue
            else:  # row_match
                sfh_line = idx + 1

            try:
                mass_bins = read_single_sfh_line(sfhfile, sfh_line)
            except Exception as e:
                # warn and continue
                print(f"Warning: failed to read SFH for subhalo {sid} (line {sfh_line}): {e}", file=sys.stderr)
                continue

            if mass_bins.size != tb.size:
                nmin = min(mass_bins.size, tb.size)
                mass_bins = mass_bins[:nmin]
                t_use = tb[:nmin]
            else:
                t_use = tb

            cosmic_age_use = cosmo_colibre.age(0).to(au.Gyr).value
            out_times = compute_all_times(t_use, mass_bins, lookback_z2=args.lookback_z2, cosmic_age=cosmic_age_use)
            dor_dict = compute_dor_from_mass(t_use, mass_bins, out_times, args.term3_ref, cosmic_age_use)

            row = {
                "subhalo_id": sid,
                "gal_idx": idx,
                "sfh_line": sfh_line,
                "total_formed_mass_Msun": out_times.get("total_formed_mass_Msun", np.nan),
                "t_start": out_times.get("t_start", np.nan),
                "t50": out_times.get("t50", np.nan),
                "t50_span": out_times.get("t50_span", np.nan),
                "t75": out_times.get("t75", np.nan),
                "t75_span": out_times.get("t75_span", np.nan),
                "t90": out_times.get("t90", np.nan),
                "t90_span": out_times.get("t90_span", np.nan),
                "t95": out_times.get("t95", np.nan),
                "t95_span": out_times.get("t95_span", np.nan),
                "t998": out_times.get("t998", np.nan),
                "t998_span": out_times.get("t998_span", np.nan),
                "tfin": out_times.get("tfin", np.nan),
                "tfin_span": out_times.get("tfin_span", np.nan),
                "f_Mz2": out_times.get("f_Mz2", np.nan),
                "term1": dor_dict.get("term1", np.nan),
                "term2": dor_dict.get("term2", np.nan),
                "term3": dor_dict.get("term3", np.nan),
                "dor": dor_dict.get("dor", np.nan)
            }
            buffer.append(row)

            if len(buffer) >= flush_every:
                df_chunk = pd.DataFrame(buffer)
                # append to CSV safely (uncompressed)
                df_chunk.to_csv(outpath, index=False, header=not header_written, mode="a")
                header_written = True
                buffer = []

        if len(buffer) > 0:
            df_chunk = pd.DataFrame(buffer)
            df_chunk.to_csv(outpath, index=False, header=not header_written, mode="a")
            print(f"Flushed final {len(buffer)} rows -> wrote {outpath}")

        print("Compute-all done.")
        return

    # ---------- sample / comparison mode ----------
    # load time bins (already loaded above)
    # read relic CSV if present (only for sample mode)
    if relic_exists:
        df_relic = pd.read_csv(relic_csv, low_memory=False)
        id_col = None
        for c in ("subhalo_id", "HaloCatalogueIndex", "subhaloId", "HaloIndex", "track_id", "TrackId"):
            if c in df_relic.columns:
                id_col = c
                break
        if id_col is None:
            id_col = df_relic.columns[0]
            print("Warning: couldn't find canonical id column in relic CSV; using", id_col)
        relic_ids = pd.to_numeric(df_relic[id_col].astype(str).str.strip().str.replace("\r", ""), errors="coerce").dropna().astype(int).unique().tolist()
        print(f"Found {len(relic_ids)} unique ids in relic CSV (using column '{id_col}').")
    else:
        print("No relic CSV found; sample mode comparisons disabled.")
        relic_ids = []

    existing_ids = [i for i in relic_ids if i in idx_by_halo]
    if len(existing_ids) == 0:
        print("No overlap between relic ids and GalaxyProperties HaloCatalogueIndex. Abort.", file=sys.stderr)
        sys.exit(1)
    sample_n = min(args.n, len(existing_ids))
    sample_ids = random.sample(existing_ids, sample_n)
    print("Sampled ids (existing in GalaxyProperties):", sample_ids)

    if args.show_matches:
        print("\nRelic CSV rows for sampled ids (if present):")
        for sid in sample_ids:
            rows = df_relic[df_relic[id_col].astype(str).str.strip() == str(sid)]
            print(f"\n--- subhalo_id {sid} ---")
            if rows.shape[0] == 0:
                print("  (not found in relic CSV rows)")
            else:
                print(rows.to_string(index=False))

    print("\nComputed diagnostics (compared to relic CSV columns if present):")
    comp_cols = ["t50", "t75", "t90", "t95", "t998", "tfin"]
    for sid in sample_ids:
        row_idx = idx_by_halo[sid]
        sfh_line_num = row_idx + 1
        try:
            mass_bins = read_single_sfh_line(sfhfile, sfh_line_num)
        except Exception as e:
            print(f"Failed to read SFH for subhalo {sid} (line {sfh_line_num}): {e}", file=sys.stderr)
            continue
        if mass_bins.size != tb.size:
            nmin = min(mass_bins.size, tb.size)
            mass_bins = mass_bins[:nmin]
            t_use = tb[:nmin]
            print(f"Note: SFH bins ({mass_bins.size}) != time bins ({tb.size}). Truncated to {nmin}.")
        else:
            t_use = tb
        out = compute_all_times(t_use, mass_bins, lookback_z2=args.lookback_z2)
        # compute DoR for display in sample mode as well
        cosmic_age_use = cosmo_colibre.age(0).to(au.Gyr).value
        dor_dict = compute_dor_from_mass(t_use, mass_bins, out, args.term3_ref, cosmic_age_use)
        # merge dor into out for printing convenience
        out.update(dor_dict)
        print(f"\nsubhalo_id {sid}  (GalaxyProperties idx {row_idx}, SFH line {sfh_line_num})")
        for k in ("total_formed_mass_Msun", "t_start", "t50", "t50_span", "t75", "t75_span", "t90", "t90_span",
                  "t95", "t95_span", "t998", "t998_span", "tfin", "tfin_span", "f_Mz2", "term1", "term2", "term3", "dor"):
            print(f"  {k:20s} : {out.get(k)}")
        # show relic CSV values for comparison (if present)
        if relic_exists:
            relic_row = df_relic[df_relic[id_col].astype(str).str.strip() == str(sid)]
            if relic_row.shape[0] > 0:
                print("  -> Values from relic CSV (first matching row):")
                r0 = relic_row.iloc[0]
                for c in comp_cols:
                    if c in r0.index:
                        print(f"    {c:10s}: {r0[c]}")
            else:
                print("  -> No matching relic CSV row found for this id (unexpected).")

    print("\nDone.")
    return


if __name__ == "__main__":
    main()