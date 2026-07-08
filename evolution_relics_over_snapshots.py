#!/usr/bin/env python3
"""
evolution_relics_over_snapshots.py

Track z=0 relic galaxies back through redshift using TrackID matches in the
processed ASCII GalaxyProperties_SFR_GE_0* tables.

What this script does
---------------------
1. Reads a z=0 relic list (typically a CSV or ASCII table containing TrackID).
2. Scans all GalaxyProperties_SFR_GE_0* tables under a processed-data directory.
3. For each table, keeps only rows whose TrackID is in the z=0 relic sample.
4. Builds a per-galaxy time series containing (when available):
   - stellar mass
   - half-mass radius
   - compactness Sigma_1.5 = log10(Mstar / R50**1.5)
   - BH mass and BH-to-stellar-mass ratio
   - central/satellite flag
   - halo catalogue index / group index
   - redshift and lookback time
5. Writes the assembled history to CSV.
6. Produces:
   - compactness evolution plot (all galaxies, with optional highlights)
   - mass-size path plot (all galaxies, coloured by redshift)
   - optional animation if Pillow is available and --animate is used

Important note about the snapshot tables
----------------------------------------
The GalaxyProperties_SFR_GE_0* files you showed are plain numeric ASCII tables
with no header row. That means the columns are positional. This script therefore
uses an explicit 32-column schema for the processed snapshot tables.

If you later discover that your processed files use a different order for any
columns, you only need to edit SNAPSHOT_COLUMNS below in one place.

Typical usage
-------------
python evolution_relics_over_snapshots.py \
  --data-dir /mnt/su3-pro/clagos/COLIBRE/Runs/L200_m6/Thermal/ProcessedData \
  --relic-z0-file z0_relics_trackids.csv \
  --relic-trackid-col track_id \
  --output-dir relic_tracks \
  --rhalf-scale 1000 

or with stripping events tracking:

python evolution_relics_over_snapshots.py \
  --data-dir /mnt/su3-pro/clagos/COLIBRE/Runs/L200_m6/Thermal/ProcessedData \
  --relic-z0-file z0_relics_trackids.csv \
  --relic-trackid-col track_id \
  --output-dir relic_tracks \
  --rhalf-scale 1000 \
  --strip-z-cut 2.0 \
  --strip-dlogm-max -0.05 \
  --strip-dcompact-min 0.05 \
  --strip-drhalf-max -0.05 \
  --strip-require-satellite-transition

If your z=0 relic list already contains only the relic sample, no extra filter is
needed. If you want a subset of TrackIDs, just provide a file with those IDs.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

try:
    from astropy.cosmology import Planck13 as COSMO
except Exception:
    COSMO = None


# -----------------------------------------------------------------------------
# Explicit 32-column schema for the processed snapshot tables
# -----------------------------------------------------------------------------
#
# The files are positional. The names below are the convenient labels you use in
# Python after reading the table with header=None and names=SNAPSHOT_COLUMNS.
#
# The mapping follows the 32-column layout described in your README and the
# structure visible in the data rows you pasted.
#
SNAPSHOT_COLUMNS_32 = [
    "halo_catalogue_index",      # 1
    "is_central",                # 2
    "x_galaxy",                  # 3
    "y_galaxy",                  # 4
    "z_galaxy",                  # 5
    "stellar_mass",              # 6
    "sfr_instant",               # 7
    "rhalf_stars",               # 8
    "h1_mass",                   # 9
    "h2_mass",                   # 10
    "kappa_co_stars",            # 11
    "kappa_co_gas",              # 12
    "disc_to_total_stars",       # 13
    "stellar_angular_momentum",  # 14
    "stellar_age",               # 15
    "gas_metallicity_lower",     # 16
    "gas_metallicity_upper",     # 17
    "dust_mass",                 # 18
    "descendant_id",             # 19
    "track_id",                  # 20
    "vel_x",                     # 21
    "vel_y",                     # 22
    "vel_z",                     # 23
    "host_halo_x",               # 24
    "host_halo_y",               # 25
    "host_halo_z",               # 26
    "halo_mass_m200crit",        # 27
    "bh_mass",                   # 28
    "bh_accretion_rate",         # 29
    "bh_thermal_energy_cum",     # 30
    "n_agn_events",              # 31
    "host_fof_halo_id",          # 32
]

SNAPSHOT_COLUMNS_33 = [
    "halo_catalogue_index",      # 1
    "is_central",                # 2
    "x_galaxy",                  # 3
    "y_galaxy",                  # 4
    "z_galaxy",                  # 5
    "stellar_mass",              # 6
    "sfr_instant",               # 7
    "rhalf_stars",               # 8
    "h1_mass",                   # 9
    "h2_mass",                   # 10
    "kappa_co_stars",            # 11
    "kappa_co_gas",              # 12
    "disc_to_total_stars",       # 13
    "stellar_angular_momentum",  # 14
    "stellar_age",               # 15
    "gas_metallicity_lower",     # 16
    "gas_metallicity_upper",     # 17
    "dust_mass",                 # 18
    "descendant_id",             # 19
    "track_id",                  # 20
    "vel_x",                     # 21
    "vel_y",                     # 22
    "vel_z",                     # 23
    "host_halo_x",               # 24
    "host_halo_y",               # 25
    "host_halo_z",               # 26
    "halo_mass_m200crit",        # 27
    "sfr_100myr",                # 28  <-- extra low-z column
    "bh_mass",                   # 29
    "bh_accretion_rate",         # 30
    "bh_thermal_energy_cum",     # 31
    "n_agn_events",              # 32
    "host_fof_halo_id",          # 33
]

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "track_id": ["track_id"],
    "redshift": ["redshift", "z"],
    "snapshot": ["snapshot", "snap"],
    "stellar_mass": ["stellar_mass"],
    "rhalf": ["rhalf_stars", "stellar_half_mass_radius", "half_mass_radius"],
    "bh_mass": ["bh_mass"],
    "exsitu_frac": ["exsitu_frac"],
    "is_central": ["is_central"],
    "halo_index": ["halo_catalogue_index", "host_fof_halo_id"],
    "sfr": ["sfr_instant", "sfr"],
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_snapshot_columns(path: Path) -> List[str]:
    ncols = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        nrows=1,
        engine="python",
    ).shape[1]

    if ncols == 32:
        return SNAPSHOT_COLUMNS_32
    if ncols == 33:
        return SNAPSHOT_COLUMNS_33

    raise RuntimeError(f"Unexpected number of columns in {path}: {ncols}")

def find_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_to_original:
            return lower_to_original[key]
    return None


def first_existing_column(df: pd.DataFrame, logical_name: str, user_override: Optional[str] = None) -> Optional[str]:
    if user_override:
        if user_override in df.columns:
            return user_override
        for c in df.columns:
            if str(c).strip().lower() == user_override.strip().lower():
                return c
    return find_column(df, COLUMN_SYNONYMS.get(logical_name, []))


def read_table_auto(path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited ascii table with a best-effort header guess."""
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")


def read_track_ids(path: Path, track_col: str = "track_id") -> np.ndarray:
    """
    Read TrackIDs from a CSV/ASCII relic file.
    Expected columns:
        track_id, HaloCatalogueIndex, DoR

    Only track_id is used.
    """
    # CSV-aware read
    df = pd.read_csv(path, comment="#")

    # Find the track_id column case-insensitively
    col = None
    for c in df.columns:
        if str(c).strip().lower() == track_col.strip().lower():
            col = c
            break

    if col is None:
        raise ValueError(
            f"Could not find a '{track_col}' column in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    track_ids = pd.to_numeric(df[col], errors="coerce").dropna().astype(np.int64).unique()
    return np.asarray(track_ids, dtype=np.int64)


def infer_snapshot_label(filename: str) -> str:
    return Path(filename).stem


def maybe_parse_redshift_from_name(name: str) -> Optional[float]:
    lowered = name.lower()
    if "z" not in lowered:
        return None
    import re
    m = re.search(r"z[_=]?([0-9]+(?:\.[0-9]+)?)", lowered)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def get_lookback_time_gyr(z: np.ndarray) -> np.ndarray:
    if COSMO is None:
        return np.full_like(z, np.nan, dtype=float)
    return np.asarray(COSMO.lookback_time(z).value, dtype=float)


def compactness_sigma_15(mstar: np.ndarray, rhalf: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(mstar / np.power(rhalf, 1.5))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# -----------------------------------------------------------------------------
# Core loading
# -----------------------------------------------------------------------------
def extract_relic_history_from_file(
    path: Path,
    target_track_ids: set,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Load one snapshot table and keep only target TrackIDs."""
    schema = get_snapshot_columns(path)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=schema,
        engine="python",
    )

    mask = pd.to_numeric(df["track_id"], errors="coerce").isin(list(target_track_ids))
    if not bool(mask.any()):
        return pd.DataFrame()

    out = df.loc[mask].copy()
    out["track_id"] = pd.to_numeric(out["track_id"], errors="coerce").astype("Int64")
    out["snapshot_file"] = path.name
    out["snapshot_label"] = infer_snapshot_label(path.name)

    z_from_name = maybe_parse_redshift_from_name(path.name)
    out["redshift"] = z_from_name if z_from_name is not None else np.nan
    out["snapshot"] = pd.NA

    out["stellar_mass_raw"] = safe_numeric(out["stellar_mass"])
    out["stellar_mass"] = out["stellar_mass_raw"] * args.stellar_mass_scale

    out["rhalf_raw"] = safe_numeric(out["rhalf_stars"])
    out["rhalf"] = out["rhalf_raw"] * args.rhalf_scale

    out["bh_mass_raw"] = safe_numeric(out["bh_mass"])
    out["bh_mass"] = out["bh_mass_raw"] * args.bh_mass_scale

    out["bh_accretion_rate"] = safe_numeric(out["bh_accretion_rate"])
    out["bh_thermal_energy_cum"] = safe_numeric(out["bh_thermal_energy_cum"])
    out["n_agn_events"] = safe_numeric(out["n_agn_events"])
    out["halo_catalogue_index"] = safe_numeric(out["halo_catalogue_index"])
    out["host_fof_halo_id"] = safe_numeric(out["host_fof_halo_id"])
    out["is_central"] = safe_numeric(out["is_central"])
    out["sfr_instant"] = safe_numeric(out["sfr_instant"])
    out["ssfr_instant"] = out["sfr_instant"] / out["stellar_mass"]
    out["ssfr_instant"] = out["ssfr_instant"].where(out["ssfr_instant"] > 0, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["log10_ssfr_instant"] = np.log10(out["ssfr_instant"])
    out["descendant_id"] = safe_numeric(out["descendant_id"])

    out["exsitu_frac"] = np.nan

    out["bh_to_stellar_mass"] = out["bh_mass"] / out["stellar_mass"]
    out["log10_bh_to_stellar_mass"] = np.log10(out["bh_to_stellar_mass"])
    out["log10_stellar_mass"] = np.log10(out["stellar_mass"])
    out["log10_rhalf"] = np.log10(out["rhalf"])
    out["compactness_sigma_1p5"] = compactness_sigma_15(
        out["stellar_mass"].to_numpy(dtype=float),
        out["rhalf"].to_numpy(dtype=float),
    )

    return out


def load_relic_history(
    data_dir: Path,
    target_track_ids: set,
    args: argparse.Namespace,
) -> pd.DataFrame:
    pattern = str(data_dir / args.file_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")

    pieces: List[pd.DataFrame] = []
    for fname in files:
        path = Path(fname)
        try:
            df = extract_relic_history_from_file(path, target_track_ids, args)
            if not df.empty:
                pieces.append(df)
                print(f"Loaded {len(df):6d} matched rows from {path.name}")
            else:
                print(f"No target TrackIDs found in {path.name}")
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    if not pieces:
        raise RuntimeError("No matching relic rows were found in any snapshot file.")

    hist = pd.concat(pieces, ignore_index=True)
    hist["track_id"] = hist["track_id"].astype("Int64")
    hist["redshift"] = pd.to_numeric(hist["redshift"], errors="coerce")
    hist["lookback_time_gyr"] = get_lookback_time_gyr(hist["redshift"].to_numpy(dtype=float))
    hist = hist.sort_values(["track_id", "redshift"], ascending=[True, False]).reset_index(drop=True)
    return hist

def load_exsitu_lookup(exsitu_h5: str, halo_idx: np.ndarray) -> dict:
    exsitu_lookup = {}
    if not Path(exsitu_h5).exists():
        print("Ex-situ summary HDF5 not found; skipping ex-situ matching.")
        return exsitu_lookup
    try:
        with h5py.File(exsitu_h5, "r") as fh:
            if "stars" in fh:
                data = np.array(fh["stars"])
                if data.ndim == 2 and data.shape[1] >= 4:
                    candidate_cols = (0, 1, 2)
                    overlaps = []
                    for c in candidate_cols:
                        ids = data[:, c].astype(np.int64)
                        overlaps.append(np.intersect1d(ids, halo_idx).size)
                    keycol = candidate_cols[int(np.argmax(overlaps))]
                    print(f"Using ex-situ key column {keycol}; overlaps = {overlaps}")
                    ids = data[:, keycol].astype(np.int64)
                    exfrac = data[:, 3].astype(float)
                    exsitu_lookup = dict(zip(ids.tolist(), exfrac.tolist()))
                    print(f"Loaded {len(exsitu_lookup)} ex-situ entries from {exsitu_h5} (dataset 'stars').")

            else:
                for k in fh:
                    try:
                        arr = np.array(fh[k])
                        if arr.ndim == 2 and arr.shape[1] >= 4:
                            candidate_cols = (0, 1, 2)
                            overlaps = []
                            for c in candidate_cols:
                                ids = arr[:, c].astype(np.int64)
                                overlaps.append(np.intersect1d(ids, halo_idx).size)
                            keycol = candidate_cols[int(np.argmax(overlaps))]
                            print(f"Using ex-situ key column {keycol}; overlaps = {overlaps}")
                            ids = arr[:, keycol].astype(np.int64)
                            exfrac = arr[:, 3].astype(float)
                            exsitu_lookup = dict(zip(ids.tolist(), exfrac.tolist()))
                            print(f"Loaded {len(exsitu_lookup)} ex-situ entries from {exsitu_h5} (dataset '{k}').")
                            break
                    except Exception:
                        continue
    except Exception as e:
        print("Warning: failed to read ex-situ HDF5:", e)
    return exsitu_lookup

# -----------------------------------------------------------------------------
# Diagnostics / summaries
# -----------------------------------------------------------------------------
# def summarize_track_evolution(hist: pd.DataFrame) -> pd.DataFrame:
#     rows = []
#     for track_id, g in hist.groupby("track_id", sort=False):
#         g = g.sort_values("redshift", ascending=False)
#         z_first = g.iloc[0]
#         z_last = g.iloc[-1]
#         rows.append(
#             {
#                 "track_id": track_id,
#                 "n_snapshots": len(g),
#                 "z_first": z_first.get("redshift", np.nan),
#                 "z_last": z_last.get("redshift", np.nan),
#                 "log10_mstar_z0": z_last.get("log10_stellar_mass", np.nan),
#                 "log10_mstar_first": z_first.get("log10_stellar_mass", np.nan),
#                 "delta_log10_mstar": z_last.get("log10_stellar_mass", np.nan) - z_first.get("log10_stellar_mass", np.nan),
#                 "compactness_z0": z_last.get("compactness_sigma_1p5", np.nan),
#                 "compactness_first": z_first.get("compactness_sigma_1p5", np.nan),
#                 "delta_compactness": z_last.get("compactness_sigma_1p5", np.nan) - z_first.get("compactness_sigma_1p5", np.nan),
#                 "bh_to_stellar_z0": z_last.get("bh_to_stellar_mass", np.nan),
#                 "bh_to_stellar_first": z_first.get("bh_to_stellar_mass", np.nan),
#                 "central_z0": z_last.get("is_central", np.nan),
#                 "central_first": z_first.get("is_central", np.nan),
#                 "halo_index_z0": z_last.get("halo_catalogue_index", np.nan),
#                 "halo_index_first": z_first.get("halo_catalogue_index", np.nan),
#                 "host_fof_halo_id_z0": z_last.get("host_fof_halo_id", np.nan),
#                 "host_fof_halo_id_first": z_first.get("host_fof_halo_id", np.nan),
#             }
#         )
#     return pd.DataFrame(rows)

def summarize_track_evolution(hist: pd.DataFrame):
    rows = []

    for track_id, g in hist.groupby("track_id", sort=False):
        g = g.sort_values("redshift", ascending=False)

        z2_rows = g[g["snapshot_file"] == "GalaxyProperties_SFR_GE_0_z2.0.txt"]
        z0_rows = g[g["snapshot_file"] == "GalaxyProperties_SFR_GE_0_z0.0.txt"]

        if z2_rows.empty or z0_rows.empty:
            continue

        z2 = z2_rows.iloc[0]
        z0 = z0_rows.iloc[0]

        rows.append(
            {
                "track_id": track_id,
                "compactness_z2": z2["compactness_sigma_1p5"],
                "compactness_z0": z0["compactness_sigma_1p5"],
                "delta_compactness": z0["compactness_sigma_1p5"] - z2["compactness_sigma_1p5"],

                "log10_mstar_z2": z2["log10_stellar_mass"],
                "log10_mstar_z0": z0["log10_stellar_mass"],
                "delta_log10_mstar": z0["log10_stellar_mass"] - z2["log10_stellar_mass"],

                "log10_rhalf_z2": z2["log10_rhalf"],
                "log10_rhalf_z0": z0["log10_rhalf"],
                "delta_log10_rhalf": z0["log10_rhalf"] - z2["log10_rhalf"],

                "bh_to_stellar_z2": z2["bh_to_stellar_mass"],
                "bh_to_stellar_z0": z0["bh_to_stellar_mass"],

                "central_z2": z2["is_central"],
                "central_z0": z0["is_central"],

                "halo_index_z2": z2["halo_catalogue_index"],
                "halo_index_z0": z0["halo_catalogue_index"],
            }
        )

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
# def plot_compactness_evolution(hist: pd.DataFrame, outdir: Path, max_highlight: int = 12) -> None: # highlight_ids=None (for stripping events)
#     fig, ax = plt.subplots(figsize=(10, 7))
#     # highlight_ids = set() if highlight_ids is None else set(highlight_ids)

#     for track_id, g in hist.groupby("track_id"):

#         g = g.sort_values("redshift", ascending=False).reset_index(drop=True)

#         x = g["redshift"].to_numpy(dtype=float)
#         y = g["compactness_sigma_1p5"].to_numpy(dtype=float)

#         if np.all(np.isnan(y)):
#             continue

#         # -----------------------------
#         # Was this galaxy central at z=2?
#         # -----------------------------
#         z2 = g[np.isclose(g["redshift"], 2.0)]

#         if z2.empty:
#             ax.plot(x, y, lw=1.0, alpha=0.18, color="0.4")
#             continue

#         if z2.iloc[0]["is_central"] != 1:
#             ax.plot(x, y, lw=1.0, alpha=0.18, color="0.4")
#             continue

#         # ------------------------------------------
#         # Find first transition from central -> satellite
#         # ------------------------------------------
#         transition = None

#         for i in range(len(g)):
#             if g.iloc[i]["is_central"] == 0:
#                 transition = i
#                 break

#         # Never became a satellite
#         if transition is None:
#             ax.plot(x, y, lw=1.0, alpha=0.18, color="0.4")
#             continue

#         # -----------------------------
#         # Plot central phase in red
#         # -----------------------------
#         ax.plot(
#             x[:transition],
#             y[:transition],
#             color="tab:red",
#             lw=2.5,
#             alpha=0.95,
#         )

#         # -----------------------------
#         # Plot satellite phase in blue
#         # -----------------------------
#         ax.plot(
#             x[transition-1:],
#             y[transition-1:],
#             color="tab:blue",
#             lw=2.5,
#             alpha=0.95,
#             label=f"{int(track_id)}",
#         )

#         # mark the transition
#         ax.scatter(
#             x[transition],
#             y[transition],
#             marker="o",
#             s=40,
#             color="black",
#             zorder=10,
#         )
    
#     # for track_id, g in hist.groupby("track_id"):
#     #     g = g.sort_values("redshift", ascending=False)
#     #     x = g["redshift"].to_numpy(dtype=float)
#     #     y = g["compactness_sigma_1p5"].to_numpy(dtype=float)

#     #     if track_id in highlight_ids:
#     #         ax.plot(x, y, lw=2.2, alpha=0.95, label=f"{int(track_id)}")
#     #         ax.scatter(x, y, s=18)
#     #     else:
#     #         ax.plot(x, y, lw=1.0, alpha=0.18, color="0.4")
    

#     ax.set_xlabel("Redshift")
#     ax.set_ylabel(r"Compactness $\lg \Sigma_{1.5} = \lg(M_\star / R_{1/2}^{1.5})$")
#     # ax.set_title("Relic compactness evolution")
#     ax.invert_xaxis()
#     ax.grid(alpha=0.25)
#     # if highlight_ids:
#     #     ax.legend(title="Highlighted TrackID", fontsize=8, ncol=2, frameon=False)
#     fig.tight_layout()
#     fig.savefig(outdir / "compactness_evolution.png", dpi=200)
#     fig.savefig(outdir / "compactness_evolution.pdf")
#     plt.close(fig)

def exsitu_group(f):
    if pd.isna(f):
        return None
    if 0.0 <= f < 0.1:
        return "low"
    if 0.1 <= f < 0.4:
        return "mid"
    if f >= 0.4:
        return "high"
    return None

def plot_compactness_evolution(
    hist: pd.DataFrame,
    summary,
    outdir: Path,
    delta_threshold: float = 0.1,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))


    # Keep only galaxies that genuinely became more compact
    # summary = summary[summary["delta_compactness"] > delta_threshold].copy()
    summary = summary.set_index("track_id")

    def exsitu_bin(f: float):
        if pd.isna(f):
            return None
        if 0.0 <= f < 0.1:
            return "low"
        if 0.1 <= f < 0.4:
            return "mid"
        if f >= 0.4:
            return "high"
        return None

    colors = {
        "low": "tab:green",
        "mid": "tab:orange",
        "high": "tab:purple",
    }

    labels_done = set()

    for track_id, g in hist.groupby("track_id"):
        g = g.sort_values("redshift", ascending=False)
        x = g["lookback_time_gyr"].to_numpy(dtype=float)
        y = g["compactness_sigma_1p5"].to_numpy(dtype=float)

        if np.all(np.isnan(y)):
            continue

        # # Not a compactness-growth galaxy: faint grey
        # if track_id not in summary.index:
        #     ax.plot(x, y, lw=1.0, alpha=0.15, color="0.6")
        #     continue

        # Galaxy missing from the summary (should almost never happen)
        if track_id not in summary.index:
            ax.plot(
                x,
                y,
                lw=1.0,
                alpha=0.15,
                color="0.6",
            )
            continue

        exf = summary.loc[track_id, "exsitu_fraction"]
        b = exsitu_bin(exf)

        if b is None:
            ax.plot(x, y, lw=1.0, alpha=0.15, color="0.6")
            continue

        colour = colors[b]
        ax.plot(x, y, lw=2.5, alpha=0.95, color=colour)
        ax.scatter(x, y, s=18, color=colour)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="tab:green", lw=2.5, label=r"$0 \leq f_{\rm exsitu} < 0.1$"),
        Line2D([0], [0], color="tab:orange", lw=2.5, label=r"$0.1 \leq f_{\rm exsitu} < 0.4$"),
        Line2D([0], [0], color="tab:purple", lw=2.5, label=r"$f_{\rm exsitu} \geq 0.4$"),
        # Line2D([0], [0], color="0.6", lw=1.0, alpha=0.3, label=fr"$\Delta\Sigma_{{1.5}} \leq {delta_threshold:.1f}$ dex"),
        Line2D([0], [0], color="0.6", lw=1.0, alpha=0.3, label="no ex-situ measurement",),
    ]
    ax.legend(handles=legend_elements, frameon=False)

    ax.set_xlabel("Lookback time [Gyr]")
    # -------------------------------------------------------
    # Secondary x-axis: redshift
    # -------------------------------------------------------
    ax_top = ax.twiny()

    # Chosen redshift ticks
    z_ticks = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8], dtype=float)

    # Convert to lookback times
    lb_ticks = COSMO.lookback_time(z_ticks).value

    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(lb_ticks)
    ax_top.set_xticklabels([f"{z:g}" for z in z_ticks])

    ax_top.set_xlabel("Redshift")
    ax.set_ylabel(r"Compactness $\lg \Sigma_{1.5} = \lg(M_\star/R_{1/2}^{1.5})$")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "compactness_evolution.png", dpi=200)
    fig.savefig(outdir / "compactness_evolution.pdf")
    plt.close(fig)

def plot_compactness_evolution_panels(hist: pd.DataFrame, summary: pd.DataFrame, outdir: Path) -> None:
    summary = summary.copy().set_index("track_id")

    def exsitu_bin(f: float):
        if pd.isna(f):
            return None
        if 0.0 <= f < 0.1:
            return "low"
        if 0.1 <= f < 0.4:
            return "mid"
        if f >= 0.4:
            return "high"
        return None

    group_order = [
        ("low", "tab:green", r"$0 \leq f_{\rm exsitu} < 0.1$"),
        ("mid", "tab:orange", r"$0.1 \leq f_{\rm exsitu} < 0.4$"),
        ("high", "tab:purple", r"$f_{\rm exsitu} \geq 0.4$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    for ax, (target_group, color, label) in zip(axes, group_order):
        for track_id, g in hist.groupby("track_id"):
            g = g.sort_values("redshift", ascending=False)
            x = g["lookback_time_gyr"].to_numpy(dtype=float)
            y = g["compactness_sigma_1p5"].to_numpy(dtype=float)

            if np.all(np.isnan(y)):
                continue

            if track_id not in summary.index:
                ax.plot(x, y, lw=1.0, alpha=0.12, color="0.7")
                continue

            exf = summary.loc[track_id, "exsitu_fraction"]
            grp = exsitu_bin(exf)

            if grp == target_group:
                ax.plot(x, y, lw=1.5, alpha=0.95, color=color)
                ax.scatter(x, y, s=18, color=color)
            else:
                ax.plot(x, y, lw=1.0, alpha=0.10, color="0.75")

        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Lookback time [Gyr]")

    axes[0].set_ylabel(r"Compactness $\lg \Sigma_{1.5} = \lg(M_\star/R_{1/2}^{1.5})$")

    # shared top axis: redshift
    ax_top = axes[1].twiny()
    z_ticks = np.array([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8], dtype=float)
    lb_ticks = COSMO.lookback_time(z_ticks).value
    ax_top.set_xlim(axes[1].get_xlim())
    ax_top.set_xticks(lb_ticks)
    ax_top.set_xticklabels([f"{z:g}" for z in z_ticks])
    ax_top.set_xlabel("Redshift")

    fig.tight_layout()
    fig.savefig(outdir / "compactness_evolution_panels.png", dpi=200)
    fig.savefig(outdir / "compactness_evolution_panels.pdf")
    plt.close(fig)

def plot_ssfr_evolution(hist: pd.DataFrame, outdir: Path, max_highlight: int = 12) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    for track_id, g in hist.groupby("track_id"):
        g = g.sort_values("redshift", ascending=False).reset_index(drop=True)

        x = g["lookback_time_gyr"].to_numpy(dtype=float)
        y = g["log10_ssfr_instant"].to_numpy(dtype=float)

        if np.all(np.isnan(y)):
            continue

        # Keep only the evolution from z=2 to z=0
        g_post = g[g["redshift"] <= 2.0].reset_index(drop=True)

        # Need at least two snapshots
        if len(g_post) < 2:
            ax.plot(x, y, lw=1.0, alpha=0.18, color="0.4")
            continue

        # Galaxy must be central at z=2
        z2 = g_post[np.isclose(g_post["redshift"], 2.0)]
        if z2.empty or z2.iloc[0]["is_central"] != 1:
            ax.plot(x, y, lw=1.0, alpha=0.18, color="0.4")
            continue

        # Find FIRST central -> satellite transition AFTER z=2
        transition = None
        for i in range(len(g_post) - 1):
            if g_post.iloc[i]["is_central"] == 1 and g_post.iloc[i + 1]["is_central"] == 0:
                transition = i + 1
                break

        # Never became satellite after z=2
        if transition is None:
            ax.plot(x, y, lw=1.0, alpha=0.18, color="0.4")
            continue

        # Diagnostic: status history after z=2
        status_string = "".join(str(int(v)) for v in g_post["is_central"].astype(int))
        n_switches = int(np.sum(g_post["is_central"].to_numpy(dtype=int)[:-1] != g_post["is_central"].to_numpy(dtype=int)[1:]))
        print(
            f"{track_id:8d}   "
            f"{status_string}   "
            f"switches={n_switches}   "
            f"first_transition_lb={g_post.iloc[transition]['lookback_time_gyr']:.2f} Gyr"
        )

        # Plot central phase in red
        ax.plot(
            g_post["lookback_time_gyr"].to_numpy(dtype=float)[:transition],
            g_post["log10_ssfr_instant"].to_numpy(dtype=float)[:transition],
            color="tab:red",
            lw=2.5,
            alpha=0.95,
        )

        # Plot satellite phase in blue
        ax.plot(
            g_post["lookback_time_gyr"].to_numpy(dtype=float)[transition - 1:],
            g_post["log10_ssfr_instant"].to_numpy(dtype=float)[transition - 1:],
            color="tab:blue",
            lw=2.5,
            alpha=0.95,
            label=f"{int(track_id)}",
        )

        # Mark the transition
        ax.scatter(
            g_post.iloc[transition]["lookback_time_gyr"],
            g_post.iloc[transition]["log10_ssfr_instant"],
            marker="o",
            s=40,
            color="black",
            zorder=10,
        )

    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_ylabel(r"$\lg(\mathrm{sSFR_{inst}} / \mathrm{yr}^{-1})$")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "ssfr_instant_evolution.png", dpi=200)
    fig.savefig(outdir / "ssfr_instant_evolution.pdf")
    plt.close(fig)

def plot_mass_size_paths(hist: pd.DataFrame, outdir: Path, max_highlight: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    zvals = hist["redshift"].to_numpy(dtype=float)
    norm = Normalize(vmin=np.nanmin(zvals), vmax=np.nanmax(zvals))
    cmap = plt.get_cmap("viridis")

    summary = summarize_track_evolution(hist).sort_values("delta_compactness", ascending=False)
    highlight_ids = summary["track_id"].head(max_highlight).tolist()

    for track_id, g in hist.groupby("track_id"):
        g = g.sort_values("redshift", ascending=False)
        x = g["log10_stellar_mass"].to_numpy(dtype=float)
        y = g["log10_rhalf"].to_numpy(dtype=float)
        z = g["redshift"].to_numpy(dtype=float)
        if len(g) < 2:
            continue

        if track_id in highlight_ids:
            ax.plot(x, y, lw=2.0, alpha=0.95)
            ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=26, edgecolor="none")
            ax.text(x[-1], y[-1], f" {int(track_id)}", fontsize=8, alpha=0.9)
        else:
            ax.plot(x, y, lw=1.0, alpha=0.18, color="0.4")
            ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=12, alpha=0.12, edgecolor="none")

    ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
    ax.set_ylabel(r"$\log_{10}(R_{50}/\mathrm{kpc})$")
    ax.set_title("Relic mass-size tracks through redshift")
    ax.grid(alpha=0.25)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Redshift")

    fig.tight_layout()
    fig.savefig(outdir / "mass_size_tracks.png", dpi=200)
    fig.savefig(outdir / "mass_size_tracks.pdf")
    plt.close(fig)


def plot_central_satellite_transitions(hist: pd.DataFrame, outdir: Path) -> None:
    if hist["is_central"].isna().all():
        return
    g = hist.dropna(subset=["redshift", "is_central"]).copy()
    if g.empty:
        return

    g["central_flag"] = (g["is_central"] > 0.5).astype(float)
    frac = g.groupby("redshift")["central_flag"].mean().reset_index().sort_values("redshift")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(frac["redshift"], frac["central_flag"], marker="o")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Fraction central")
    ax.set_title("Central/satellite status of tracked relics")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "central_fraction_vs_redshift.png", dpi=200)
    fig.savefig(outdir / "central_fraction_vs_redshift.pdf")
    plt.close(fig)


def make_animation(hist: pd.DataFrame, outdir: Path) -> None:
    try:
        import matplotlib.animation as animation
    except Exception:
        print("Animation skipped: matplotlib.animation unavailable")
        return

    zs = np.sort(hist["redshift"].dropna().unique())
    if len(zs) == 0:
        print("Animation skipped: no finite redshifts")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    norm = Normalize(vmin=np.nanmin(hist["redshift"].to_numpy(dtype=float)), vmax=np.nanmax(hist["redshift"].to_numpy(dtype=float)))
    cmap = plt.get_cmap("viridis")
    all_tracks = {tid: g.sort_values("redshift", ascending=False) for tid, g in hist.groupby("track_id")}

    def update(frame_idx: int):
        ax.clear()
        ax.set_xlabel(r"$\log_{10}(M_\star/M_\odot)$")
        ax.set_ylabel(r"$\log_{10}(R_{50}/\mathrm{kpc})$")
        z_now = zs[frame_idx]
        ax.set_title(f"Relic mass-size evolution, z <= {z_now:.2f}")
        ax.grid(alpha=0.25)
        for _, g in all_tracks.items():
            gg = g[g["redshift"] >= z_now]
            if gg.empty:
                continue
            x = gg["log10_stellar_mass"].to_numpy(dtype=float)
            y = gg["log10_rhalf"].to_numpy(dtype=float)
            z = gg["redshift"].to_numpy(dtype=float)
            ax.plot(x, y, lw=1.5, alpha=0.8)
            ax.scatter(x, y, c=z, cmap=cmap, norm=norm, s=18, edgecolor="none")
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(zs), blit=False)
    gif_path = outdir / "mass_size_tracks.gif"
    try:
        writer = animation.PillowWriter(fps=2)
        ani.save(gif_path, writer=writer)
        print(f"Saved animation: {gif_path}")
    except Exception as exc:
        print(f"Animation skipped: {exc}")
    finally:
        plt.close(fig)

# -----------------------------------------------------------------------------
# Stripping analysis
# -----------------------------------------------------------------------------
# def find_stripping_events(
#         hist: pd.DataFrame,
#         z_cut: float = 2.0,
#         dlogm_max: float = -0.05,
#         dcompact_min: float = 0.05,
#         drhalf_max: float = -0.05,
#         require_satellite_transition: bool = False,
#     ) -> pd.DataFrame:
#         """
#         Find stripping-like events between consecutive snapshots at z <= z_cut.

#         A candidate event is flagged when, between two adjacent snapshots:
#         - compactness increases by at least dcompact_min
#         - stellar mass decreases by at least |dlogm_max|
#         - half-mass radius decreases by at least |drhalf_max|

#         If require_satellite_transition=True, also require central -> satellite.
#         """
#         events = []

#         for track_id, g in hist.groupby("track_id", sort=False):
#             g = g.dropna(subset=["redshift"]).sort_values("redshift", ascending=False)

#             # Keep only snapshots at or below z_cut, i.e. the post-z=2 phase
#             g = g[g["redshift"] <= z_cut].reset_index(drop=True)
#             if len(g) < 2:
#                 continue

#             for i in range(len(g) - 1):
#                 a = g.iloc[i]
#                 b = g.iloc[i + 1]

#                 vals = [
#                     a["log10_stellar_mass"], b["log10_stellar_mass"],
#                     a["log10_rhalf"], b["log10_rhalf"],
#                     a["compactness_sigma_1p5"], b["compactness_sigma_1p5"],
#                 ]
#                 if any(pd.isna(v) for v in vals):
#                     continue

#                 dlogm = b["log10_stellar_mass"] - a["log10_stellar_mass"]
#                 dlogr = b["log10_rhalf"] - a["log10_rhalf"]
#                 dcompact = b["compactness_sigma_1p5"] - a["compactness_sigma_1p5"]

#                 sat_transition = False
#                 if not pd.isna(a["is_central"]) and not pd.isna(b["is_central"]):
#                     sat_transition = (a["is_central"] > 0.5) and (b["is_central"] <= 0.5)

#                 if (
#                     dcompact >= dcompact_min
#                     and dlogm <= dlogm_max
#                     and dlogr <= drhalf_max
#                 ):
#                     if require_satellite_transition and not sat_transition:
#                         continue

#                     events.append(
#                         {
#                             "track_id": track_id,
#                             "z_before": a["redshift"],
#                             "z_after": b["redshift"],
#                             "lookback_before_gyr": a["lookback_time_gyr"],
#                             "lookback_after_gyr": b["lookback_time_gyr"],
#                             "log10_mstar_before": a["log10_stellar_mass"],
#                             "log10_mstar_after": b["log10_stellar_mass"],
#                             "log10_rhalf_before": a["log10_rhalf"],
#                             "log10_rhalf_after": b["log10_rhalf"],
#                             "compactness_before": a["compactness_sigma_1p5"],
#                             "compactness_after": b["compactness_sigma_1p5"],
#                             "dlog10_mstar": dlogm,
#                             "dlog10_rhalf": dlogr,
#                             "dcompactness": dcompact,
#                             "is_central_before": a["is_central"],
#                             "is_central_after": b["is_central"],
#                             "satellite_transition": sat_transition,
#                             "snapshot_before": a["snapshot_file"],
#                             "snapshot_after": b["snapshot_file"],
#                         }
#                     )
#                     break  # first post-z_cut stripping event is enough

#         return pd.DataFrame(events)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track relic galaxies across redshift using GalaxyProperties tables.")
    p.add_argument("--data-dir", type=Path, required=True, help="Directory containing GalaxyProperties_SFR_GE_0* files")
    p.add_argument("--file-glob", type=str, default="GalaxyProperties_SFR_GE_0*", help="Glob for the tables")
    p.add_argument("--relic-z0-file", type=Path, required=True, help="z=0 relic catalogue containing TrackID")
    p.add_argument("--relic-trackid-col", type=str, default="TrackID", help="TrackID column name in the z=0 relic file")
    p.add_argument("--output-dir", type=Path, default=Path("relic_evolution_out"), help="Output directory")

    p.add_argument("--stellar-mass-scale", type=float, default=1.0, help="Multiply raw stellar mass by this factor")
    p.add_argument("--rhalf-scale", type=float, default=1.0, help="Multiply raw half-mass radius by this factor")
    p.add_argument("--bh-mass-scale", type=float, default=1.0, help="Multiply raw BH mass by this factor")

    p.add_argument("--max-highlight", type=int, default=12, help="How many tracks to highlight in the overview plots")
    p.add_argument("--animate", action="store_true", help="Try to create a GIF mass-size animation")

    p.add_argument("--strip-z-cut", type=float, default=2.0,
               help="Only search for stripping events at z <= this value")
    p.add_argument("--strip-dlogm-max", type=float, default=-0.05,
                help="Minimum drop in log10 stellar mass per step")
    p.add_argument("--strip-dcompact-min", type=float, default=0.05,
                help="Minimum increase in compactness per step")
    p.add_argument("--strip-drhalf-max", type=float, default=-0.05,
                help="Minimum drop in log10 half-mass radius per step")
    p.add_argument("--strip-require-satellite-transition", action="store_true",
                help="Require a central -> satellite transition for a stripping event")
    p.add_argument("--exsitu-h5",type=Path, default=Path("/mnt/su3ctm/kproctor/ForMax/exsitu_summary_SnapNum_127.hdf5"),
                help="z=0 ex-situ summary HDF5")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    ensure_dir(args.output_dir)

    relic_track_ids = read_track_ids(args.relic_z0_file, args.relic_trackid_col)
    target_track_ids = set(int(x) for x in relic_track_ids if np.isfinite(x))
    if not target_track_ids:
        raise RuntimeError("No TrackIDs were read from the z=0 relic file.")

    print(f"Loaded {len(target_track_ids)} target TrackIDs from {args.relic_z0_file}")

    hist = load_relic_history(args.data_dir, target_track_ids, args)
    hist = hist.drop_duplicates(subset=["track_id", "redshift", "snapshot_file"], keep="last").reset_index(drop=True)

    hist_out = args.output_dir / "relic_history_tracks.csv"
    hist.to_csv(hist_out, index=False)
    print(f"Saved history table to {hist_out}")

    summary = summarize_track_evolution(hist)
    halo_idx = summary["halo_index_z0"].dropna().astype(np.int64).to_numpy()
    exsitu_lookup = load_exsitu_lookup(args.exsitu_h5, halo_idx)
    summary["exsitu_fraction"] = summary["halo_index_z0"].astype("Int64").map(exsitu_lookup)
    print(summary[["track_id", "halo_index_z0", "exsitu_fraction"]].head(10))
    print(summary["exsitu_fraction"].describe())

    summary["exsitu_group"] = summary["exsitu_fraction"].apply(exsitu_group)

    print("\nCompactness medians by ex-situ group")
    print("------------------------------------")

    group_rows = []
    for grp in ["low", "mid", "high"]:
        s = summary[summary["exsitu_group"] == grp].copy()
        if s.empty:
            print(f"{grp:4s}: no galaxies")
            continue

        med_z2 = float(np.nanmedian(s["compactness_z2"]))
        med_z0 = float(np.nanmedian(s["compactness_z0"]))
        delta_med = med_z0 - med_z2

        group_rows.append({
            "exsitu_group": grp,
            "median_compactness_z2": med_z2,
            "median_compactness_z0": med_z0,
            "delta_median_compactness": delta_med,
            "n_galaxies": len(s),
        })

        print(
            f"{grp:4s}: "
            f"z2={med_z2:.3f}, "
            f"z0={med_z0:.3f}, "
            f"delta={delta_med:.3f}, "
            f"N={len(s)}"
        )

    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(args.output_dir / "compactness_group_medians.csv", index=False)

    summary_out = args.output_dir / "relic_track_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Saved summary table to {summary_out}")

    summary["strip_like"] = (summary["delta_compactness"] > 0) & (summary["delta_log10_mstar"] < 0)
    strip_out = args.output_dir / "strip_like_candidates.csv"
    summary.sort_values(["strip_like", "delta_compactness"], ascending=[False, False]).to_csv(strip_out, index=False)

    # strip_events = find_stripping_events(
    #     hist,
    #     z_cut=args.strip_z_cut,
    #     dlogm_max=args.strip_dlogm_max,
    #     dcompact_min=args.strip_dcompact_min,
    #     drhalf_max=args.strip_drhalf_max,
    #     require_satellite_transition=args.strip_require_satellite_transition,
    # )

    # strip_events_out = args.output_dir / "stripping_events_post_z2.csv"
    # strip_events.to_csv(strip_events_out, index=False)
    # print(f"Saved stripping events to {strip_events_out}")

    # strip_ids = set(strip_events["track_id"].astype(int)) if not strip_events.empty else set()

    plot_compactness_evolution_panels(hist, summary, args.output_dir) #plot_compactness_evolution(hist, args.output_dir, highlight_ids=strip_ids)
    plot_ssfr_evolution(hist, args.output_dir, max_highlight=args.max_highlight)
    plot_mass_size_paths(hist, args.output_dir, max_highlight=args.max_highlight)
    plot_central_satellite_transitions(hist, args.output_dir)

    if args.animate:
        make_animation(hist, args.output_dir)

    print("\nTop compactness growers:")
    cols = ["track_id", "delta_compactness", "delta_log10_mstar", "central_z0", "halo_index_z0"]
    print(summary.sort_values("delta_compactness", ascending=False)[cols].head(10).to_string(index=False))

    print("\nMost stripping-like candidates:")
    print(summary[summary["strip_like"]].sort_values("delta_compactness", ascending=False)[cols].head(10).to_string(index=False))

    print("\nDone. Output directory:")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
