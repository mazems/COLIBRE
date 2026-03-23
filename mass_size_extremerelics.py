#!/usr/bin/env python3
"""
Mass-size plot + efficient tracing of zero-BH extreme relics across snapshots.

Usage:
    python mass_size_extremerelics_trace.py

Toggle behavior at the top of the file with RUN_PLOT / RUN_TRACE.
"""
from __future__ import annotations
import os
import sys
import csv
import gc
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

# ---- user/project imports ----
# make sure your PYTHONPATH includes the folder where `common` is located
import common

# ---------------- CONFIG / FLAGS ----------------
RUN_PLOT = True    # set True to create the z=0 mass-size plot
RUN_TRACE = False    # set True to run the trace of zero-BH extreme relics across snapshots

CORRECTED_DOR_CSV = "sfh_times_all_with_DoR_variants_corrected.csv.gz"
OUTDIR = "plots"
OUTNAME = "mass_size_extremes_compactness9p75.png"
OUT_CSV_TRACE = os.path.join(OUTDIR, "extreme_relics_zeroBH_central_status_by_snap.csv")

COMPACTNESS_CUT = 9.75
EXTREME_DOR = 0.6

# chunk size for scanning big HDF5 datasets (tune lower if you still get killed)
CHUNK = 80_000

# Unit conversions consistent with earlier code
Lu = 3.086e+24 / (3.086e+24)    # cMpc -> cMpc (kept like your snippet)
Mu = 1.988e+43 / 1.989e33       # simulation mass unit -> Msun (matching your conversion)
tu = 3.086e+19 / 3.154e7        # time unit -> yr

# Snapshot / model settings (match your choices)
model_name = 'L0200N3008/THERMAL_AGN/'
model_dir = '/mnt/su3-pro/colibre/' + model_name
snap_file = '0127'   # default z=0 snapshot id (used for initial SOAP read)
# Keep snap_files and zstarget aligned for the trace loop
snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0026', '0018']
zstarget   = [0.0,    0.1,    0.2,    0.5,    1.0,    2.0,    3.0,    4.0,    5.0,    6.0,    8.0,    10.0]
if len(snap_files) != len(zstarget):
    raise SystemExit("snap_files and zstarget must be the same length.")

comov_to_physical_length = 1.0 / (1.0 + 0.0)  # for z=0 usage (only for plot conversions)

# Fields to read from SOAP (minimal set)
fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')}
fields = {'ExclusiveSphere/50kpc': (
            'StellarMass', 'StarFormationRate', 'HalfMassRadiusStars',
            'MassWeightedMeanStellarAge', 'LuminosityWeightedMeanStellarAge',
            'LinearMassWeightedIronOverHydrogenOfStars',
            'LinearMassWeightedMagnesiumOverHydrogenOfStars', 'MostMassiveBlackHoleMass'
         )}

# ---------------- READ SOAP (z=0) ----------------
print("Reading SOAP with common.read_group_data_colibre(...) for snapshot", snap_file)
h5data_groups   = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
(halo_index, is_central_arr, desc_id, track_id_arr) = h5data_idgroups
(m30, sfr30, r50, stellarage, stellarage_lum, Fe_lin, Mg_lin, bh_mass_raw) = h5data_groups

# unit conversions
m30 = m30 * Mu
sfr30 = sfr30 * Mu / tu
r50 = r50 * comov_to_physical_length * 1e3    # -> kpc
stellarage_lum = stellarage_lum * tu / 1e9     # -> Gyr
# BH mass (linear Msun)
bh_mass_full = bh_mass_raw * Mu

# selection and masking (same as your large script)
sel_idx = np.where(m30 >= 1e9)[0]
if sel_idx.size == 0:
    raise SystemExit("No galaxies selected in SOAP (m >= 1e9).")

# align selected arrays and subset
m = m30[sel_idx]
r = r50[sel_idx]
halo_idx = halo_index[sel_idx]       # SOAP HaloCatalogueIndex for each selected row
track = track_id_arr[sel_idx]
sfr = sfr30[sel_idx]
Mg_lin = Mg_lin[sel_idx]
Fe_lin = Fe_lin[sel_idx]
age = stellarage_lum[sel_idx]
bh_mass = bh_mass_full[sel_idx]
is_central = is_central_arr[sel_idx]

# filter positive values
mask_pos = (m > 0) & (r > 0)
m = m[mask_pos]; r = r[mask_pos]; halo_idx = halo_idx[mask_pos]; track = track[mask_pos]
sfr = sfr[mask_pos]; Mg_lin = Mg_lin[mask_pos]; Fe_lin = Fe_lin[mask_pos]; age = age[mask_pos]; bh_mass = bh_mass[mask_pos]
is_central = is_central[mask_pos].astype(bool)
print(f"Selected SOAP galaxies after mass/radius filter: {len(m)}")

# derived
with np.errstate(divide="ignore", invalid="ignore"):
    logM = np.log10(m)
    logR = np.log10(r)
    compactness = logM - 1.5 * logR
    mgfe = np.where((Mg_lin > 0) & (Fe_lin > 0), np.log10(Mg_lin / Fe_lin) - 0.10, np.nan)
    ssfr = np.where((m > 0) & np.isfinite(sfr), sfr / m, np.nan)
    log_ssfr = np.where((ssfr > 0) & np.isfinite(ssfr), np.log10(ssfr), np.nan)
    bh_ratio = np.where((bh_mass > 0) & (m > 0) & np.isfinite(bh_mass) & np.isfinite(m),
                        bh_mass / m,
                        np.nan)
    log_bh_ratio = np.where(np.isfinite(bh_ratio) & (bh_ratio > 0), np.log10(bh_ratio), np.nan)

# ---------------- Load corrected DoR CSV and map to SOAP-selected rows ----------------
dor_lookup = {}
dor_colname = None
if os.path.exists(CORRECTED_DOR_CSV):
    print("Loading corrected DoR CSV:", CORRECTED_DOR_CSV)
    df_corr = pd.read_csv(CORRECTED_DOR_CSV, low_memory=False)

    # heuristics to find a DoR column
    for cand in ("DoR_t95", "DoR_t90", "DoR_t998", "DoR_tfin", "DoR", "dor", "DoR_csv"):
        if cand in df_corr.columns:
            dor_colname = cand
            break
    if dor_colname is None:
        for c in df_corr.columns:
            if c.lower().startswith("dor"):
                dor_colname = c
                break

    if dor_colname is None:
        print("Corrected DoR CSV found but no DoR-like column inside it. Proceeding without DoR mapping.")
    else:
        # pick ID column to map to SOAP HaloCatalogueIndex / subhalo id
        id_col = None
        for cand in ("subhalo_id", "HaloCatalogueIndex", "track_id", "TrackId", "HaloIndex"):
            if cand in df_corr.columns:
                id_col = cand
                break
        if id_col is None:
            for c in df_corr.columns:
                if pd.api.types.is_integer_dtype(df_corr[c]) or pd.api.types.is_float_dtype(df_corr[c]):
                    id_col = c
                    break
        if id_col is None:
            print("Could not find an ID column in corrected DoR CSV to map onto SOAP; skipping DoR mapping.")
            dor_colname = None
        else:
            for _, row in df_corr[[id_col, dor_colname]].iterrows():
                try:
                    idx = int(row[id_col])
                    v = row[dor_colname]
                    if pd.isna(v):
                        continue
                    dor_lookup[idx] = float(v)
                except Exception:
                    continue
            print(f"Built DoR lookup from corrected CSV using ID '{id_col}' and DoR column '{dor_colname}'. "
                  f"Entries: {len(dor_lookup)}")
else:
    print("Corrected DoR CSV not found at", CORRECTED_DOR_CSV, "-> plotting without corrected-DoR-based highlighting.")

# Map DoR onto the selected SOAP rows using halo_idx
dor_for_each_selected = np.full(len(halo_idx), np.nan, dtype=float)
if dor_lookup:
    for i, hid in enumerate(halo_idx):
        dor_for_each_selected[i] = dor_lookup.get(int(hid), np.nan)

if np.all(~np.isfinite(dor_for_each_selected)):
    print("No DoR from corrected CSV mapped to SOAP selected rows (all NaN).")
else:
    n_matched = np.sum(np.isfinite(dor_for_each_selected))
    print(f"Mapped DoR to SOAP-selected rows: {n_matched} / {len(dor_for_each_selected)} finite DoR values.")

# define extremes mask using mapped DoR
extreme_mask = np.isfinite(dor_for_each_selected) & (dor_for_each_selected > EXTREME_DOR)
extreme_mask_sat = extreme_mask & ~is_central
extreme_mask_cen = extreme_mask & is_central

# ---------------- PLOTTING (z=0 mass-size) ----------------
if RUN_PLOT:
    os.makedirs(OUTDIR, exist_ok=True)
    outpath = os.path.join(OUTDIR, OUTNAME)

    plt.rcParams.update({"font.size": 12, "figure.figsize": (8,6)})
    fig, ax = plt.subplots()

    # background: all selected galaxies (light grey)
    ax.scatter(logM, logR, s=8, color="lightgrey", alpha=0.6, label="simulated galaxies at z=0")

    # compactness reference line: logR = (logM - COMPACTNESS_CUT) / 1.5
    xm = np.linspace(np.nanmin(logM) - 0.2, np.nanmax(logM) + 0.2, 400)
    compact_line = (xm - COMPACTNESS_CUT) / 1.5
    ax.plot(xm, compact_line, linestyle="--", color="black", lw=2,
            label=fr"compactness: $\Sigma_{{1.5}} = {COMPACTNESS_CUT}$")

    # highlight extremes: big orange stars
    if np.any(extreme_mask_cen):
        ax.scatter(logM[extreme_mask_cen], logR[extreme_mask_cen],
                   facecolor='C1', edgecolor='k', s=140, marker='*', linewidth=0.6,
                   zorder=110, label=f"extreme central relics (DoR > {EXTREME_DOR})")
        print(f"Plotted {int(np.sum(extreme_mask_cen))} extreme central relics (DoR > {EXTREME_DOR}).")
    else:
        print(f"No extreme relics found with DoR > {EXTREME_DOR}.")
    
    if np.any(extreme_mask_sat):
        ax.scatter(logM[extreme_mask_sat], logR[extreme_mask_sat],
                   facecolor='C2', edgecolor='k', s=140, marker='*', linewidth=0.6,
                   zorder=110, label=f"extreme satellite relics (DoR > {EXTREME_DOR})")
        print(f"Plotted {int(np.sum(extreme_mask_sat))} extreme satellite relics (DoR > {EXTREME_DOR}).")
    else:
        print(f"No extreme relics found with DoR > {EXTREME_DOR}.")

    ax.set_xlabel(r"$\log_{10}(M_\star / M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(R_{1/2} / \mathrm{kpc})$")
    ax.grid(True)
    ax.legend(fontsize=9, loc='lower right')

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", outpath)

# ---------------- TRACE ZERO-BH EXTREMES ACROSS SNAPSHOTS ----------------
if RUN_TRACE:
    # prepare selected arrays
    bh_mass_selected = bh_mass              # linear BH mass (Msun) for the selected+filtered rows
    track_selected = track                  # TrackId aligned to selected rows
    haloidx_selected = halo_idx
    dor_selected = dor_for_each_selected

    # choose extreme relics which have DoR > EXTREME_DOR but BH mass <= 0 (or NaN)
    mask_extreme = np.isfinite(dor_selected) & (dor_selected > EXTREME_DOR)
    mask_zero_bh = mask_extreme & (~np.isfinite(bh_mass_selected) | (bh_mass_selected <= 0.0))
    idxs_zero_bh = np.where(mask_zero_bh)[0]

    print(f"Found {len(idxs_zero_bh)} extreme relic(s) with zero/missing BH mass among the selected SOAP rows.")

    if len(idxs_zero_bh) == 0:
        print("No zero-BH extreme relics to trace; skipping snapshot loop.")
    else:
        # target tracks set (only finite track ids)
        target_tracks = set(int(track_selected[i]) for i in idxs_zero_bh if np.isfinite(track_selected[i]))
        print("Target tracks (sample up to 20):", list(target_tracks)[:20])

        os.makedirs(OUTDIR, exist_ok=True)
        out_rows = []

        # helper: detect snapshot file path (SOAP-HBT preferred then SOAP)
        def find_snapshot_path(snap_label):
            path1 = os.path.join(model_dir, "SOAP-HBT", f"halo_properties_{snap_label}.hdf5")
            path2 = os.path.join(model_dir, "SOAP",     f"halo_properties_{snap_label}.hdf5")
            if os.path.exists(path1):
                return path1
            if os.path.exists(path2):
                return path2
            return None

        # chunked scanner function using h5py (reads only small chunks)
        def trace_tracks_through_snapshot_file(snapshot_path, target_tracks_set, snap_label, zsnap):
            found_rows_local = []
            if not os.path.exists(snapshot_path):
                return found_rows_local

            try:
                with h5py.File(snapshot_path, "r") as fh:
                    # try a few common dataset name locations
                    def find_first_dataset(cands):
                        for c in cands:
                            if c in fh:
                                return c
                            if "InputHalos" in fh and c in fh["InputHalos"]:
                                return "InputHalos/" + c
                        return None

                    candidates_track = ["HBTplus/TrackId", "HBT/TrackId", "TrackId", "HBTplus/track_id", "HBTplus/Track_Id"]
                    candidates_haloidx = ["HaloCatalogueIndex", "HaloIndex", "Halo/Index"]
                    candidates_iscen = ["IsCentral", "is_central"]

                    ds_track_name = find_first_dataset(candidates_track)
                    ds_halo_name  = find_first_dataset(candidates_haloidx)
                    ds_iscen_name = find_first_dataset(candidates_iscen)

                    if ds_track_name is None:
                        raise RuntimeError(f"No TrackId-like dataset found in {snapshot_path}; keys: {list(fh.keys())}")

                    d_track = fh[ds_track_name]
                    d_halo = fh[ds_halo_name] if (ds_halo_name is not None and ds_halo_name in fh) else None
                    d_iscen = fh[ds_iscen_name] if (ds_iscen_name is not None and ds_iscen_name in fh) else None

                    n_total = d_track.shape[0]
                    targets_remaining = set(target_tracks_set)
                    if not targets_remaining:
                        return found_rows_local

                    # iterate in chunks
                    for start in range(0, n_total, CHUNK):
                        stop = min(start + CHUNK, n_total)
                        # read TrackId chunk
                        track_chunk = d_track[start:stop]
                        try:
                            track_chunk_f = np.asarray(track_chunk, dtype=float)
                        except Exception:
                            track_chunk_f = np.array(track_chunk, dtype=float)

                        # check membership against remaining targets
                        # (convert targets_remaining to list for np.isin)
                        mask_in = np.isin(track_chunk_f, list(targets_remaining))
                        if not np.any(mask_in):
                            continue

                        rel_idxs = np.nonzero(mask_in)[0]
                        abs_idxs = rel_idxs + start

                        # read halo index and iscentral for matched positions
                        halo_vals = d_halo[abs_idxs] if d_halo is not None else np.full(len(abs_idxs), np.nan)
                        iscen_vals = d_iscen[abs_idxs] if d_iscen is not None else np.full(len(abs_idxs), np.nan)

                        for ai, reli in enumerate(rel_idxs):
                            trval = int(track_chunk_f[rel_idxs[ai]])
                            hid = int(halo_vals[ai]) if (d_halo is not None and np.isfinite(halo_vals[ai])) else np.nan
                            iscen_raw = iscen_vals[ai] if d_iscen is not None else np.nan
                            if isinstance(iscen_raw, (np.bool_, bool)):
                                iscen = 1 if iscen_raw else 0
                            else:
                                try:
                                    iscen = int(iscen_raw) if np.isfinite(iscen_raw) else np.nan
                                except Exception:
                                    iscen = np.nan

                            found_rows_local.append({
                                "track_id": int(trval),
                                "snapshot": snap_label,
                                "z": float(zsnap),
                                "halo_index": hid,
                                "is_central": iscen,
                                "found": True
                            })
                            if trval in targets_remaining:
                                targets_remaining.remove(trval)

                        if not targets_remaining:
                            break

                    # any remaining targets were not found in this snapshot -> mark as not found
                    for tr_missing in list(targets_remaining):
                        found_rows_local.append({
                            "track_id": int(tr_missing),
                            "snapshot": snap_label,
                            "z": float(zsnap),
                            "halo_index": np.nan,
                            "is_central": np.nan,
                            "found": False
                        })

            except Exception as e:
                print(f"Error scanning {snapshot_path}: {e}")
                # return empty list — caller will write not-found rows if needed
                return found_rows_local

            gc.collect()
            return found_rows_local

        # Outer loop over snapshots
        for snap_label, zsnap in zip(snap_files, zstarget):
            snapshot_path = find_snapshot_path(snap_label)
            if snapshot_path is None:
                print(f"Snapshot file for {snap_label} not found; marking all targets as not found for this snapshot.")
                # Mark not-found for each target track
                for t in sorted(target_tracks):
                    out_rows.append({
                        "track_id": int(t), "snapshot": snap_label, "z": float(zsnap),
                        "halo_index": np.nan, "is_central": np.nan, "found": False
                    })
                continue

            print("Scanning snapshot", snap_label, "file:", snapshot_path)
            found_here = trace_tracks_through_snapshot_file(snapshot_path, target_tracks, snap_label, zsnap)
            # if function returned rows for all targets (found or not), extend; otherwise ensure every target has entry
            if found_here:
                # build map of which tracks were covered
                covered = set([int(r["track_id"]) for r in found_here])
                # add rows for covered ones
                out_rows.extend(found_here)
                # if some target track not present in found_here (shouldn't happen, but be safe), add not-found rows
                missing_now = set(target_tracks) - covered
                for t in missing_now:
                    out_rows.append({
                        "track_id": int(t), "snapshot": snap_label, "z": float(zsnap),
                        "halo_index": np.nan, "is_central": np.nan, "found": False
                    })
            else:
                # fallback: nothing returned -> mark all as not found for this snapshot
                for t in sorted(target_tracks):
                    out_rows.append({
                        "track_id": int(t), "snapshot": snap_label, "z": float(zsnap),
                        "halo_index": np.nan, "is_central": np.nan, "found": False
                    })

        # write results to CSV
        with open(OUT_CSV_TRACE, "w", newline="") as fh:
            fieldnames = ["track_id", "snapshot", "z", "halo_index", "is_central", "found"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in out_rows:
                writer.writerow(r)

        print("Wrote central-status tracing CSV:", OUT_CSV_TRACE)

        # small summary for first 10 tracks
        summary = defaultdict(list)
        for r in out_rows:
            summary[r["track_id"]].append((r["snapshot"], r["z"], r["found"], r["halo_index"], r["is_central"]))
        for i, (trk, events) in enumerate(summary.items()):
            if i >= 10: break
            print(f"Track {trk}:")
            for ev in events:
                print("  snap", ev[0], " z=", ev[1], " found=", ev[2], " halo=", ev[3], " is_central=", ev[4])

print("Script finished.")