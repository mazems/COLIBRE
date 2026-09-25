#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_id_column(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    col = _first_existing_column(df, preferred)
    if col is None:
        raise RuntimeError(
            f"No ID column found. Tried {preferred}. Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out[col] = pd.to_numeric(out[col].astype(str).str.strip(), errors="coerce")
    bad = int(out[col].isna().sum())
    if bad > 0:
        print(f"Warning: {bad} rows had non-numeric values in '{col}' and will be dropped.")
        out = out[out[col].notna()].copy()

    out[col] = out[col].astype(np.int64)
    if col != "HaloCatalogueIndex":
        out = out.rename(columns={col: "HaloCatalogueIndex"})

    return out


def _read_first_dataset(f: h5py.File, candidates: list[str]) -> h5py.Dataset:
    for p in candidates:
        if p in f:
            return f[p]
        if p.lstrip("/") in f:
            return f[p.lstrip("/")]
    raise KeyError(f"None of these datasets were found: {candidates}")


def _read_selected_rows(ds: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    """
    Read only selected rows from a 1D dataset. h5py fancy indexing requires
    increasing indices, so we sort and then restore the original order.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return np.array([], dtype=ds.dtype)

    order = np.argsort(rows)
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)

    rows_sorted = rows[order]
    vals_sorted = ds[rows_sorted]
    vals = np.asarray(vals_sorted)[inv]
    return vals


def _load_exsitu_lookup(
    exsitu_h5: str | None,
    track_ids: np.ndarray,
    halo_ids: np.ndarray,
) -> dict[int, float]:
    if exsitu_h5 is None:
        return {}

    path = Path(exsitu_h5)
    if not path.exists():
        print(f"Ex-situ file not found: {path} (skipping)")
        return {}

    try:
        with h5py.File(path, "r") as fh:
            candidates: list[tuple[str, np.ndarray]] = []
            if "stars" in fh:
                candidates.append(("stars", np.asarray(fh["stars"])))
            else:
                for k in fh.keys():
                    try:
                        arr = np.asarray(fh[k])
                        candidates.append((k, arr))
                    except Exception:
                        pass

            best_name = None
            best_arr = None
            best_keycol = None
            best_overlap = -1

            for name, arr in candidates:
                if arr.ndim != 2 or arr.shape[1] < 4:
                    continue
                for keycol in (0, 1, 2):
                    ids = arr[:, keycol].astype(np.int64)
                    overlap_track = np.intersect1d(ids, track_ids).size
                    overlap_halo = np.intersect1d(ids, halo_ids).size
                    overlap = max(overlap_track, overlap_halo)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_name = name
                        best_arr = arr
                        best_keycol = keycol

            if best_arr is None or best_keycol is None:
                print(f"No suitable ex-situ dataset found in {path}; skipping ex-situ.")
                return {}

            ids = best_arr[:, best_keycol].astype(np.int64)
            exfrac = best_arr[:, 3].astype(float)
            lookup = dict(zip(ids.tolist(), exfrac.tolist()))
            print(
                f"Loaded {len(lookup)} ex-situ entries from {path} "
                f"(dataset '{best_name}', key col {best_keycol})."
            )
            return lookup

    except Exception as e:
        print(f"Warning: failed to read ex-situ HDF5 {path}: {e}")
        return {}


def _apply_bh_correction(track_id: np.ndarray, bh_mass_raw: np.ndarray, lookup_csv: str | None) -> np.ndarray:
    bh_corr = bh_mass_raw.copy()

    if lookup_csv is None:
        return bh_corr

    path = Path(lookup_csv)
    if not path.exists():
        print(f"BH lookup not found: {path} (skipping BH correction)")
        return bh_corr

    try:
        bh_lookup = pd.read_csv(path)
        if not {"track_id", "corrected_bh_mass"}.issubset(bh_lookup.columns):
            print(f"BH lookup {path} does not contain expected columns; skipping BH correction.")
            return bh_corr

        bh_dict = dict(
            zip(
                bh_lookup["track_id"].astype(np.int64),
                bh_lookup["corrected_bh_mass"].astype(float),
            )
        )

        n_replaced = 0
        for i, tid in enumerate(track_id.astype(np.int64)):
            if (not np.isfinite(bh_corr[i])) or (bh_corr[i] <= 0):
                new_val = bh_dict.get(int(tid), np.nan)
                if np.isfinite(new_val) and new_val > 0:
                    bh_corr[i] = float(new_val)
                    n_replaced += 1

        print(f"Replaced {n_replaced} vanished BH masses using {path}")
        return bh_corr

    except Exception as e:
        print(f"Warning: failed to apply BH correction from {path}: {e}")
        return bh_corr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SOAP-based galaxy properties for a sample defined by HaloCatalogueIndex."
    )
    parser.add_argument(
        "sample_csv",
        help="Input sample CSV, e.g. z0_relics_trackids.csv or z0_sags_trackids.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV filename. Default: <input>_properties.csv",
    )
    parser.add_argument(
        "--model-name",
        default="L0200N3008/THERMAL_AGN/",
        help="Simulation subdirectory inside /mnt/su3-pro/colibre/.",
    )
    parser.add_argument(
        "--snap",
        type=int,
        default=127,
        help="Snapshot number (e.g. 127 for z=0).",
    )
    parser.add_argument(
        "--ztarget",
        type=float,
        default=0.0,
        help="Target redshift used for physical conversion.",
    )
    parser.add_argument(
        "--exsitu-h5",
        default="/mnt/su3ctm/kproctor/ForMax/exsitu_summary_SnapNum_127.hdf5",
        help="Optional ex-situ summary HDF5 file.",
    )
    parser.add_argument(
        "--bh-lookup",
        default=str(SCRIPT_DIR / "corrected_bh_mass_lookup.csv"),
        help="Optional BH-mass correction lookup CSV.",
    )
    parser.add_argument(
        "--soap-file",
        default=None,
        help="Optional explicit SOAP HDF5 file path. Default: model SOAP-HBT/halo_properties_<snap>.hdf5",
    )
    args = parser.parse_args()

    sample_path = Path(args.sample_csv)
    if not sample_path.exists():
        raise SystemExit(f"Input CSV not found: {sample_path}")

    out_path = Path(args.output) if args.output else sample_path.with_name(sample_path.stem + "_properties.csv")
    model_dir = Path("/mnt/su3-pro/colibre") / args.model_name
    snap_file = f"{args.snap:04d}"
    soap_path = Path(args.soap_file) if args.soap_file else (model_dir / "SOAP-HBT" / f"halo_properties_{snap_file}.hdf5")
    comov_to_physical_length = 1.0 / (1.0 + args.ztarget)
    sigma_path = model_dir / "SOAP-HBT" / "extra" / f"halo_properties_{snap_file}.hdf5"
    sigma_ds = "/ExclusiveSphere/HalfMassRadiusStars/StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"

    print("Reading sample:", sample_path)
    sample = pd.read_csv(sample_path, low_memory=False)
    sample = _coerce_id_column(
        sample,
        preferred=["HaloCatalogueIndex", "subhalo_id", "subhaloId", "HaloIndex", "track_id", "TrackId", "id"],
    )

    if "TrackId" in sample.columns:
        sample["TrackId_sample"] = pd.to_numeric(sample["TrackId"], errors="coerce")
    elif "track_id" in sample.columns:
        sample["TrackId_sample"] = pd.to_numeric(sample["track_id"], errors="coerce")

    sample = sample.drop_duplicates(subset=["HaloCatalogueIndex"], keep="first").copy()
    sample_ids = sample["HaloCatalogueIndex"].to_numpy(dtype=np.int64)
    print(f"Sample size after ID cleaning: {len(sample_ids)}")

    if not soap_path.exists():
        raise SystemExit(f"SOAP file not found: {soap_path}")

    # ------------------------------------------------------------
    # Read only the SOAP halo index first, then select matching rows
    # ------------------------------------------------------------
    print("Reading SOAP index only:", soap_path)

    with h5py.File(soap_path, "r") as f:
        halo_index_ds = _read_first_dataset(
            f,
            [
                "/InputHalos/HaloCatalogueIndex",
                "InputHalos/HaloCatalogueIndex",
            ],
        )
        halo_index_all = np.asarray(halo_index_ds, dtype=np.int64)

        row_map: dict[int, int] = {}
        for i, hid in enumerate(halo_index_all):
            if int(hid) not in row_map:
                row_map[int(hid)] = i

        rows = np.array([row_map.get(int(hid), -1) for hid in sample_ids], dtype=np.int64)
        matched = rows >= 0
        n_match = int(np.sum(matched))
        n_miss = int(np.sum(~matched))
        print(f"Matched {n_match}/{len(sample_ids)} sample IDs to SOAP rows.")
        if n_miss > 0:
            print(f"Warning: {n_miss} sample IDs were not found in SOAP.")

        # keep only matched rows for extraction
        sample_matched = sample.loc[matched].copy()
        rows_matched = rows[matched]
        sample_ids_matched = sample_ids[matched]

        # read fields only for selected rows
        def r(path_candidates: list[str]) -> np.ndarray:
            ds = _read_first_dataset(f, path_candidates)
            return _read_selected_rows(ds, rows_matched)

        is_central = r([
            "/InputHalos/IsCentral",
            "InputHalos/IsCentral",
        ])
        desc_id = r([
            "/InputHalos/HBTplus/DescendantTrackId",
            "InputHalos/HBTplus/DescendantTrackId",
        ])
        track_id = r([
            "/InputHalos/HBTplus/TrackId",
            "InputHalos/HBTplus/TrackId",
        ])

        m30 = r([
            "/ExclusiveSphere/50kpc/StellarMass",
            "ExclusiveSphere/50kpc/StellarMass",
        ])
        sfr30 = r([
            "/ExclusiveSphere/50kpc/StarFormationRate",
            "ExclusiveSphere/50kpc/StarFormationRate",
        ])
        r50 = r([
            "/ExclusiveSphere/50kpc/HalfMassRadiusStars",
            "ExclusiveSphere/50kpc/HalfMassRadiusStars",
        ])
        age_mass = r([
            "/ExclusiveSphere/50kpc/MassWeightedMeanStellarAge",
            "ExclusiveSphere/50kpc/MassWeightedMeanStellarAge",
        ])
        age_lum = r([
            "/ExclusiveSphere/50kpc/LuminosityWeightedMeanStellarAge",
            "ExclusiveSphere/50kpc/LuminosityWeightedMeanStellarAge",
        ])
        Fe_lin = r([
            "/ExclusiveSphere/50kpc/LinearMassWeightedIronOverHydrogenOfStars",
            "ExclusiveSphere/50kpc/LinearMassWeightedIronOverHydrogenOfStars",
        ])
        Mg_lin = r([
            "/ExclusiveSphere/50kpc/LinearMassWeightedMagnesiumOverHydrogenOfStars",
            "ExclusiveSphere/50kpc/LinearMassWeightedMagnesiumOverHydrogenOfStars",
        ])
        bh_mass_raw = r([
            "/ExclusiveSphere/50kpc/MostMassiveBlackHoleMass",
            "ExclusiveSphere/50kpc/MostMassiveBlackHoleMass",
        ])
        Zstar_raw = r([
            "/ExclusiveSphere/50kpc/StellarMassFractionInMetals",
            "ExclusiveSphere/50kpc/StellarMassFractionInMetals",
        ])
        Z_local = r([
            "/ExclusiveSphere/1kpc/StellarMassFractionInMetals",
            "ExclusiveSphere/1kpc/StellarMassFractionInMetals",
        ])

        # Optional host halo index
        try:
            host_halo = r([
                "/SOAP/HostHaloIndex",
                "SOAP/HostHaloIndex",
            ])
        except Exception:
            host_halo = None

    # ------------------------------------------------------------
    # Units and derived quantities
    # ------------------------------------------------------------
    Mu = 1.988e43 / 1.989e33
    tu = 3.086e19 / 3.154e7

    m30 = np.asarray(m30, dtype=float) * Mu
    sfr30 = np.asarray(sfr30, dtype=float) * Mu / tu
    r50 = np.asarray(r50, dtype=float) * comov_to_physical_length * 1e3  # kpc
    age_mass = np.asarray(age_mass, dtype=float) * tu / 1e9
    age_lum = np.asarray(age_lum, dtype=float) * tu / 1e9
    bh_mass_raw = np.asarray(bh_mass_raw, dtype=float) * Mu
    Zstar_raw = np.asarray(Zstar_raw, dtype=float)
    Z_local = np.asarray(Z_local, dtype=float)

    Zsun = 0.0134
    with np.errstate(divide="ignore", invalid="ignore"):
        logZstar_rel = np.where((Zstar_raw > 0) & np.isfinite(Zstar_raw), np.log10(Zstar_raw / Zsun), np.nan)
        logZstar_loc = np.where((Z_local > 0) & np.isfinite(Z_local), np.log10(Z_local / Zsun), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        logM = np.where(m30 > 0, np.log10(m30), np.nan)
        logR = np.where(r50 > 0, np.log10(r50), np.nan)
        compactness = logM - 1.5 * logR
        mgfe = np.where((Mg_lin > 0) & (Fe_lin > 0), np.log10(Mg_lin / Fe_lin) - 0.10, np.nan)
        ssfr = np.where((m30 > 0) & np.isfinite(sfr30), sfr30 / m30, np.nan)
        log_ssfr = np.where(ssfr > 0, np.log10(ssfr), np.nan)

    bh_mass_corr = _apply_bh_correction(track_id, bh_mass_raw, args.bh_lookup)
    with np.errstate(divide="ignore", invalid="ignore"):
        bh_ratio_raw = np.where((bh_mass_raw > 0) & (m30 > 0), bh_mass_raw / m30, np.nan)
        bh_ratio_corr = np.where((bh_mass_corr > 0) & (m30 > 0), bh_mass_corr / m30, np.nan)
        log_bh_ratio_raw = np.where(bh_ratio_raw > 0, np.log10(bh_ratio_raw), np.nan)
        log_bh_ratio_corr = np.where(bh_ratio_corr > 0, np.log10(bh_ratio_corr), np.nan)

    # --------------------------------------------------------------
    # LOAD HOST VELOCITY DISPERSION (CORRECT + ALIGNED)
    # --------------------------------------------------------------
    sigma_vals = np.full(len(rows_matched), np.nan, dtype=np.float32)
    log_sigma_vals = np.full(len(rows_matched), np.nan, dtype=np.float32)

    if sigma_path.exists():
        with h5py.File(sigma_path, "r") as fs:
            ds = _read_first_dataset(
                fs,
                [sigma_ds],
            )
            print("sigma dataset shape:", ds.shape)

            # IMPORTANT: use SOAP row indices, not sample-local positions
            rows = np.asarray(ds[rows_matched, :], dtype=np.float32)

            sigma_rr = rows[:, 0]
            sigma_pphi = rows[:, 4]
            sigma_zz = rows[:, 8]

            sigma_sel = np.sqrt((sigma_rr**2 + sigma_pphi**2 + sigma_zz**2) / 3.0)

            keep = np.isfinite(sigma_sel) & (sigma_sel > 0)
            sigma_vals[keep] = sigma_sel[keep]
            log_sigma_vals[keep] = np.log10(sigma_sel[keep])

        print("Loaded sigma values:", np.isfinite(sigma_vals).sum(), "/", sigma_vals.size)
        print("N(sigma == 0):", np.count_nonzero(np.isclose(sigma_vals[np.isfinite(sigma_vals)], 0.0)))
    else:
        print(f"Sigma file not found: {sigma_path}")

    # ------------------------------------------------------------
    # Ex-situ fraction lookup
    # ------------------------------------------------------------
    exsitu_lookup = _load_exsitu_lookup(
        args.exsitu_h5,
        track_ids=np.asarray(track_id, dtype=np.int64),
        halo_ids=np.asarray(sample_ids_matched, dtype=np.int64),
    )

    exsitu_frac = np.full(len(sample_ids_matched), np.nan, dtype=float)
    if exsitu_lookup:
        exsitu_series = pd.Series(exsitu_lookup, dtype=float)
        track_reindex = exsitu_series.reindex(np.asarray(track_id, dtype=np.int64)).to_numpy(dtype=float)
        halo_reindex = exsitu_series.reindex(np.asarray(sample_ids_matched, dtype=np.int64)).to_numpy(dtype=float)

        n_track = int(np.isfinite(track_reindex).sum())
        n_halo = int(np.isfinite(halo_reindex).sum())
        if n_track >= n_halo:
            exsitu_frac = track_reindex
            print(f"Matched ex-situ using TrackId ({n_track} finite matches; halo-index matches {n_halo}).")
        else:
            exsitu_frac = halo_reindex
            print(f"Matched ex-situ using HaloCatalogueIndex ({n_halo} finite matches; track-id matches {n_track}).")
    else:
        print("No ex-situ lookup available; exsitu_frac will be NaN.")

    # ------------------------------------------------------------
    # Build output table
    # ------------------------------------------------------------
    out = sample_matched.copy()
    out["TrackId_SOAP"] = np.asarray(track_id, dtype=np.int64)
    out["DescendantTrackId"] = np.asarray(desc_id, dtype=np.int64)
    out["IsCentral"] = np.asarray(is_central).astype(bool)
    out["logM"] = logM
    out["logR"] = logR
    out["r50_kpc"] = r50
    out["compactness"] = compactness
    out["age_mass_gyr"] = age_mass
    out["age_lum_gyr"] = age_lum
    out["MgFe"] = mgfe
    out["logZstar_rel"] = logZstar_rel
    out["logZstar_loc"] = logZstar_loc
    out["log_ssfr"] = log_ssfr
    out["ssfr"] = ssfr
    out["bh_mass_raw"] = bh_mass_raw
    out["bh_mass_corr"] = bh_mass_corr
    out["log_bh_ratio_raw"] = log_bh_ratio_raw
    out["log_bh_ratio_corr"] = log_bh_ratio_corr
    out["exsitu_frac"] = exsitu_frac
    out["sigma"] = sigma_vals
    out["log_sigma"] = log_sigma_vals
    out["Zstar_raw"] = Zstar_raw
    out["Z_local_raw"] = Z_local
    if host_halo is not None:
        out["HostHaloIndex"] = np.asarray(host_halo, dtype=np.int64)

    out["matched_to_SOAP"] = True

    # Keep original sample IDs first
    first_cols = [c for c in ["HaloCatalogueIndex", "TrackId_sample", "TrackId", "DoR"] if c in out.columns]
    remaining_cols = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + remaining_cols]

    out_path = Path(args.output) if args.output else sample_path.with_name(sample_path.stem + "_properties.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()