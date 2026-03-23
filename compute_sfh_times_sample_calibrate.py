#!/usr/bin/env python3
"""
compute_sfh_times_sample_calibrate.py

Like your original script but self-contained:
 - implements quantile_from_sfh_lookback using bin-edge integration (method B)
 - calibrates cosmic_age against relic CSV span columns (grid search)
 - prints best cosmic_age and runs the sample diagnostics (--n)
"""

import argparse, os, sys, random, math
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import astropy.units as au

# ----------------- CONFIG (COLIBRE cosmology used as default) -----------------
COLIBRE_H = 0.681
COLIBRE_OMEGAM = 0.306
cosmo_colibre = FlatLambdaCDM(H0=100.0 * COLIBRE_H, Om0=COLIBRE_OMEGAM)

# ----------------- Helpers / arg parsing -------------------------------------
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
    p.add_argument("--n", type=int, default=2, help="number of random ids to test (unused for calibration)")
    p.add_argument("--seed", type=int, default=12345, help="random seed")
    p.add_argument("--lookback-z2", type=float, default=10.3,
                   help="lookback time (Gyr) corresponding to z=2 (used for f_Mz2). Default 10.3 Gyr.")
    p.add_argument("--show_matches", action="store_true", help="print matching rows from relic CSV for sampled ids")
    return p.parse_args()

def safe_load_galprops(path):
    try:
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        return df.to_numpy()

def read_single_sfh_line(sfh_path, line_number):
    with open(sfh_path, "r") as fh:
        for i, line in enumerate(fh, start=1):
            if i == line_number:
                toks = line.strip().replace(",", " ").split()
                vals = [float(x) if (x != "" and x.lower() != "nan") else 0.0 for x in toks]
                return np.array(vals, dtype=float)
    raise IndexError(f"SFH file {sfh_path} has no line {line_number}")

# ----------------- Quantile method B: bin-edge integration --------------------
def quantile_from_sfh_lookback(time_bins, mass_bins, q):
    tb = np.asarray(time_bins, dtype=float)
    m = np.nan_to_num(mass_bins, nan=0.0, posinf=0.0, neginf=0.0)
    if m.size == 0 or np.sum(m) <= 0:
        return np.nan

    order = np.argsort(tb)[::-1]      # descending lookback (oldest->youngest)
    t_sorted = tb[order]
    m_sorted = m[order]

    if t_sorted.size == 1:
        return float(t_sorted[0])

    mid = 0.5 * (t_sorted[:-1] + t_sorted[1:])
    edges = np.empty(t_sorted.size + 1, dtype=float)
    edges[1:-1] = mid
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

# ----------------- Compute diagnostics using quantile fn ---------------------
def compute_all_times(time_bins, mass_bins, lookback_z2=10.3, cosmic_age=None):
    tb = np.asarray(time_bins, dtype=float)
    mass = np.nan_to_num(mass_bins, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(mass))

    out = {"total_formed_mass_Msun": total}
    if total <= 0.0:
        for k in ("t_start","t50","t50_span","t75","t75_span","t90","t90_span",
                  "t95","t95_span","t998","t998_span","tfin","tfin_span"):
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

    # quantile lookback times (method B)
    t50_lb  = quantile_from_sfh_lookback(tb, mass, 0.50)
    t75_lb  = quantile_from_sfh_lookback(tb, mass, 0.75)
    t90_lb  = quantile_from_sfh_lookback(tb, mass, 0.90)
    t95_lb  = quantile_from_sfh_lookback(tb, mass, 0.95)
    t998_lb = quantile_from_sfh_lookback(tb, mass, 0.998)
    tfin_lb = quantile_from_sfh_lookback(tb, mass, 0.9999)

    out["t50"]  = cosmic_age_use - t50_lb  if np.isfinite(t50_lb)  else float("nan")
    out["t75"]  = cosmic_age_use - t75_lb  if np.isfinite(t75_lb)  else float("nan")
    out["t90"]  = cosmic_age_use - t90_lb  if np.isfinite(t90_lb)  else float("nan")
    out["t95"]  = cosmic_age_use - t95_lb  if np.isfinite(t95_lb)  else float("nan")
    out["t998"] = cosmic_age_use - t998_lb if np.isfinite(t998_lb) else float("nan")
    out["tfin"] = cosmic_age_use - tfin_lb if np.isfinite(tfin_lb) else float("nan")

    # spans
    if np.isfinite(out["t_start"]):
        base = out["t_start"]
        for qname in ("t50","t75","t90","t95","t998","tfin"):
            out[f"{qname}_span"] = float(out[qname] - base) if np.isfinite(out[qname]) else float("nan")
    else:
        for qname in ("t50_span","t75_span","t90_span","t95_span","t998_span","tfin_span"):
            out[qname] = float("nan")

    order_desc = np.argsort(tb)[::-1]
    tb_desc = tb[order_desc]
    mass_desc = mass[order_desc]
    mask_z2 = (tb_desc >= lookback_z2)
    out["f_Mz2"] = float(np.sum(mass_desc[mask_z2]) / total) if total>0 else float("nan")

    out["term1"] = out["term2"] = out["term3"] = float("nan")
    return out

# ----------------- Calibration helper (uses variables passed explicitly) ----------
def compute_for_id_with_age(sid, idx_by_halo, tb, sfhfile, cosmic_age_use, lookback_z2=10.3):
    row_idx = idx_by_halo[int(sid)]
    sfh_line = row_idx + 1
    mass_bins = read_single_sfh_line(sfhfile, sfh_line)
    if mass_bins.size != tb.size:
        nmin = min(mass_bins.size, tb.size)
        mass_bins = mass_bins[:nmin]
        tb_use = tb[:nmin]
    else:
        tb_use = tb
    out = compute_all_times(tb_use, mass_bins, lookback_z2=lookback_z2, cosmic_age=cosmic_age_use)
    return out

# ----------------- main -------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)

    # derive paths
    galfile = args.galfile if args.galfile else os.path.join(args.basedir, "GalaxyProperties_sfrGE0_z0.0.txt")
    timefile = args.timefile if args.timefile else os.path.join(args.basedir, "look_back_time_info_circular_apertures_face_on_map_dt0.01_z0.0.txt")
    sfhfile = args.sfhfile if args.sfhfile else os.path.join(args.basedir, "Mstar_SFH_ap50ckpc_circular_apertures_face_on_map_dt0.01_z0.0.txt")
    relic_csv = args.relic_csv

    # existence checks
    for p in (galfile, timefile, sfhfile, relic_csv):
        if not os.path.exists(p):
            print("Missing file:", p, file=sys.stderr)
            sys.exit(1)

    # load galaxy properties and build index map
    gal_arr = safe_load_galprops(galfile)
    halo_ids = gal_arr[:,0].astype(np.int64)
    idx_by_halo = {int(h): i for i,h in enumerate(halo_ids)}
    print(f"Loaded GalaxyProperties rows: {len(halo_ids)}")

    # load time bins
    tb = np.loadtxt(timefile).astype(float)
    print(f"Loaded {tb.size} lookback bins (Gyr)")

    # load relic CSV and extract id list
    df_relic = pd.read_csv(relic_csv, low_memory=False)
    id_col = None
    for c in ("subhalo_id","HaloCatalogueIndex","subhaloId","HaloIndex","track_id","TrackId"):
        if c in df_relic.columns:
            id_col = c
            break
    if id_col is None:
        id_col = df_relic.columns[0]
        print("Warning: couldn't find canonical id column in relic CSV; using", id_col)
    relic_ids = pd.to_numeric(df_relic[id_col].astype(str).str.strip().str.replace("\r",""), errors="coerce").dropna().astype(int).unique().tolist()
    print(f"Found {len(relic_ids)} unique ids in relic CSV (using column '{id_col}').")

    relic_ids_in_gal = [i for i in relic_ids if i in idx_by_halo]
    if len(relic_ids_in_gal) == 0:
        print("No overlap between relic ids and GalaxyProperties HaloCatalogueIndex. Abort.", file=sys.stderr)
        sys.exit(1)
    print(f"{len(relic_ids_in_gal)} relic ids overlap with GalaxyProperties.")

    # ---------------- Faster Calibration: preload SFHs + coarse->fine grid ----------------
    n_sample = min(50, len(relic_ids_in_gal))
    rng = np.random.default_rng(seed=12345)
    sample_ids = list(rng.choice(relic_ids_in_gal, size=n_sample, replace=False))

    span_cols = ["t50_span","t75_span","t90_span","t95_span","t998_span","tfin_span"]

    print(f"Calibration: preloading SFH lines for {len(sample_ids)} sample ids...")
    sfh_cache = {}
    for sid in sample_ids:
        row_idx = idx_by_halo[int(sid)]
        sfh_line = row_idx + 1
        try:
            mass_bins = read_single_sfh_line(sfhfile, sfh_line)
        except Exception as e:
            print(f"  Warning: failed to read SFH for id {sid}: {e}")
            continue
        sfh_cache[sid] = mass_bins

    # read CSV span values for those ids
    csv_spans_by_id = {}
    for sid in list(sfh_cache.keys()):
        rows = df_relic[df_relic[id_col].astype(str).str.strip() == str(sid)]
        if rows.shape[0] == 0:
            continue
        r0 = rows.iloc[0]
        spans = {}
        ok = False
        for c in span_cols:
            if c in r0.index:
                try:
                    spans[c] = float(r0[c])
                    ok = True
                except Exception:
                    spans[c] = np.nan
            else:
                spans[c] = np.nan
        if ok:
            csv_spans_by_id[sid] = spans

    if len(csv_spans_by_id) == 0:
        print("Calibration: no span columns found in relic CSV sample; skipping calibration.")
        calibrated_age = cosmo_colibre.age(0).to(au.Gyr).value
    else:
        coarse_grid = np.linspace(12.8, 14.6, 50)
        best_age = None; best_score = np.inf

        print("Calibration (coarse grid)...")
        for age_try in coarse_grid:
            errs = []
            for sid, csv_spans in csv_spans_by_id.items():
                mass_bins = sfh_cache.get(sid)
                if mass_bins is None:
                    continue
                if mass_bins.size != tb.size:
                    nmin = min(mass_bins.size, tb.size)
                    mb = mass_bins[:nmin]
                    tb_use = tb[:nmin]
                else:
                    mb = mass_bins; tb_use = tb
                out = compute_all_times(tb_use, mb, lookback_z2=args.lookback_z2, cosmic_age=age_try)
                for c in span_cols:
                    csvv = csv_spans.get(c, np.nan)
                    compv = out.get(c, np.nan)
                    if np.isfinite(csvv) and np.isfinite(compv):
                        errs.append((compv - csvv)**2)
            score = float(np.mean(errs)) if len(errs) else np.inf
            if score < best_score:
                best_score = score; best_age = age_try

        if best_age is None:
            calibrated_age = cosmo_colibre.age(0).to(au.Gyr).value
            print("Calibration failed to find a good age; using cosmology age:", calibrated_age)
        else:
            window = 0.25
            fine_grid = np.linspace(max(12.0, best_age - window), best_age + window, 101)
            print(f"Calibration: refining around {best_age:.4f} Gyr with finer grid...")
            best_age2 = best_age; best_score2 = best_score
            for age_try in fine_grid:
                errs = []
                for sid, csv_spans in csv_spans_by_id.items():
                    mass_bins = sfh_cache.get(sid)
                    if mass_bins is None:
                        continue
                    if mass_bins.size != tb.size:
                        nmin = min(mass_bins.size, tb.size)
                        mb = mass_bins[:nmin]
                        tb_use = tb[:nmin]
                    else:
                        mb = mass_bins; tb_use = tb
                    out = compute_all_times(tb_use, mb, lookback_z2=args.lookback_z2, cosmic_age=age_try)
                    for c in span_cols:
                        csvv = csv_spans.get(c, np.nan)
                        compv = out.get(c, np.nan)
                        if np.isfinite(csvv) and np.isfinite(compv):
                            errs.append((compv - csvv)**2)
                score = float(np.mean(errs)) if len(errs) else np.inf
                if score < best_score2:
                    best_score2 = score; best_age2 = age_try

            calibrated_age = best_age2
            print(f"Calibration done. Best cosmic_age ≈ {calibrated_age:.6f} Gyr (MSE {best_score2:.6g})")

    # ----------------- Report calibration diagnostics & per-id comparisons -----------------
    print(f"\nCalibration summary: using calibrated cosmic_age = {calibrated_age:.6f} Gyr")

    # collect residuals per span column
    resids_by_span = {c: [] for c in span_cols}
    n_done = 0

    print("\nPer-galaxy comparison (only galaxies with numeric CSV spans):")
    hdr = "id".ljust(12) + "".join([f"{c:>12s}" for c in span_cols]) + "".join([f"{c+'_cmp':>12s}" for c in span_cols])
    print(hdr)
    print("-" * len(hdr))

    for sid, csv_spans in csv_spans_by_id.items():
        mb = sfh_cache.get(sid)
        if mb is None:
            continue
        if mb.size != tb.size:
            nmin = min(mb.size, tb.size)
            mb_use = mb[:nmin]
            tb_use = tb[:nmin]
        else:
            mb_use = mb
            tb_use = tb

        out = compute_all_times(tb_use, mb_use, lookback_z2=args.lookback_z2, cosmic_age=calibrated_age)

        if not any(np.isfinite(list(csv_spans.values()))):
            continue

        n_done += 1
        line_csv = f"{str(sid):<12s}"
        line_comp = ""
        for c in span_cols:
            csvv = csv_spans.get(c, np.nan)
            compv = out.get(c, np.nan)
            s_csv = f"{csvv:10.3f}" if np.isfinite(csvv) else "     NaN  "
            s_cmp = f"{compv:10.3f}" if np.isfinite(compv) else "     NaN  "
            line_csv += f"{s_csv:>12s}"
            line_comp += f"{s_cmp:>12s}"
            if np.isfinite(csvv) and np.isfinite(compv):
                res = compv - csvv
                resids_by_span[c].append(res)
        print(line_csv + line_comp)

    if n_done == 0:
        print("No overlapping span data to show (empty csv_spans_by_id).")
    else:
        print(f"\nComputed comparisons for {n_done} galaxies. Summary residuals (comp - csv):")
        for c in span_cols:
            arr = np.array(resids_by_span[c], dtype=float)
            if arr.size == 0:
                print(f"  {c:8s}: no matched values")
                continue
            mean = float(np.nanmean(arr))
            rms = float(np.sqrt(np.nanmean(arr**2)))
            med = float(np.nanmedian(arr))
            print(f"  {c:8s}: mean={mean:+6.3f} Gyr, rms={rms:6.3f} Gyr, median={med:+6.3f} Gyr, N={arr.size}")

        all_errs = np.hstack([np.array(resids_by_span[c])**2 for c in span_cols if len(resids_by_span[c])>0])
        if all_errs.size > 0:
            mse = float(np.mean(all_errs))
            rmse = float(np.sqrt(mse))
            print(f"\nOverall MSE = {mse:.6g} (RMSE = {rmse:.6g} Gyr)")
        else:
            print("No numerical residuals collected; cannot compute MSE.")

    # End of main
    return

if __name__ == "__main__":
    main()