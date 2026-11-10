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
python3 evolution_relics_over_snapshots.py \
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
        z15_rows = g[g["snapshot_file"] == "GalaxyProperties_SFR_GE_0_z1.5.txt"]
        z0_rows = g[g["snapshot_file"] == "GalaxyProperties_SFR_GE_0_z0.0.txt"]

        if z2_rows.empty or z15_rows.empty or z0_rows.empty:
            continue

        z2 = z2_rows.iloc[0]
        z15 = z15_rows.iloc[0]
        z0 = z0_rows.iloc[0]

        rows.append(
            {
                "track_id": track_id,
                "compactness_z2": z2["compactness_sigma_1p5"],
                "compactness_z15": z15["compactness_sigma_1p5"],
                "compactness_z0": z0["compactness_sigma_1p5"],
                "delta_compactness": z0["compactness_sigma_1p5"] - z2["compactness_sigma_1p5"],
                "delta_compactness_z15": z0["compactness_sigma_1p5"] - z15["compactness_sigma_1p5"],

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

def plot_bh_mass_evolution_panels(
    hist: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path
) -> None:

    summary = summary.copy().set_index("track_id")

    group_order = [
        ("low", "tab:green", r"$0 \leq f_{\rm exsitu} < 0.1$"),
        ("mid", "tab:orange", r"$0.1 \leq f_{\rm exsitu} < 0.4$"),
        ("high", "tab:purple", r"$f_{\rm exsitu} \geq 0.4$"),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        sharex=True,
        sharey=True
    )

    for ax, (target_group, color, label) in zip(axes, group_order):

        for track_id, g in hist.groupby("track_id"):

            g = g.sort_values("redshift", ascending=False)

            x = g["lookback_time_gyr"].to_numpy(dtype=float)
            y = g["bh_mass"].to_numpy(dtype=float)

            # Plot in log-space, but preserve BH=0 as missing
            y_plot = np.where(y > 0, np.log10(y), np.nan)

            if np.all(np.isnan(y_plot)):
                continue

            if track_id not in summary.index:
                ax.plot(
                    x, y_plot,
                    lw=1.0,
                    alpha=0.10,
                    color="0.75"
                )
                continue

            exf = summary.loc[track_id, "exsitu_fraction"]
            grp = exsitu_group(exf)

            if grp == target_group:
                ax.plot(
                    x, y_plot,
                    lw=1.5,
                    alpha=0.95,
                    color=color
                )
                ax.scatter(
                    x, y_plot,
                    s=18,
                    color=color
                )
            else:
                ax.plot(
                    x, y_plot,
                    lw=1.0,
                    alpha=0.10,
                    color="0.75"
                )

        ax.set_title(label)
        ax.set_xlabel("Lookback time [Gyr]")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(
        r"$\lg(M_{\rm BH}/M_\odot)$"
    )

    # shared redshift axis
    ax_top = axes[1].twiny()

    z_ticks = np.array(
        [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8],
        dtype=float
    )

    lb_ticks = COSMO.lookback_time(z_ticks).value

    ax_top.set_xlim(axes[1].get_xlim())
    ax_top.set_xticks(lb_ticks)
    ax_top.set_xticklabels([f"{z:g}" for z in z_ticks])
    ax_top.set_xlabel("Redshift")

    fig.tight_layout()

    fig.savefig(
        outdir / "bh_mass_evolution_panels.png",
        dpi=200
    )
    fig.savefig(
        outdir / "bh_mass_evolution_panels.pdf"
    )

    plt.close(fig)

def plot_bh_evolution_panels(
    hist: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
) -> None:

    summary = summary.copy().set_index("track_id")

    group_order = [
        ("low", "tab:green", r"$0 \leq f_{\rm exsitu} < 0.1$"),
        ("mid", "tab:orange", r"$0.1 \leq f_{\rm exsitu} < 0.4$"),
        ("high", "tab:purple", r"$f_{\rm exsitu} \geq 0.4$"),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        sharex=True,
        sharey=True,
    )

    for ax, (target_group, color, label) in zip(axes, group_order):

        for track_id, g in hist.groupby("track_id"):

            g = g.sort_values("redshift", ascending=False)

            x = g["lookback_time_gyr"].to_numpy(dtype=float)
            y = g["bh_to_stellar_mass"].to_numpy(dtype=float)

            if track_id not in summary.index:
                continue

            # ex-situ group of this galaxy
            grp = summary.loc[track_id, "exsitu_group"]

            if grp == target_group:

                # Positive BH masses
                pos = np.isfinite(y) & (y > 0)

                ax.plot(
                    x[pos],
                    y[pos],
                    lw=1.5,
                    alpha=0.9,
                    color=color,
                )

                ax.scatter(
                    x[pos],
                    y[pos],
                    s=14,
                    color=color,
                )

                # Explicitly mark BH = 0 snapshots
                zero = np.isfinite(y) & (y == 0)

                if np.any(zero):
                    ax.scatter(
                        x[zero],
                        np.full(np.sum(zero), 1e-8),
                        marker="v",
                        s=25,
                        color=color,
                    )

            else:
                # Other groups in grey
                pos = np.isfinite(y) & (y > 0)

                ax.plot(
                    x[pos],
                    y[pos],
                    lw=1.0,
                    alpha=0.10,
                    color="0.75",
                )

                zero = np.isfinite(y) & (y == 0)

                if np.any(zero):
                    ax.scatter(
                        x[zero],
                        np.full(np.sum(zero), 1e-8),
                        marker="v",
                        s=12,
                        color="0.75",
                        alpha=0.10,
                    )

        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("Lookback time [Gyr]")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"$M_{\rm BH}/M_\star$")

    # Redshift axis
    ax_top = axes[1].twiny()

    z_ticks = np.array(
        [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8],
        dtype=float,
    )

    lb_ticks = COSMO.lookback_time(z_ticks).value

    ax_top.set_xlim(axes[1].get_xlim())
    ax_top.set_xticks(lb_ticks)
    ax_top.set_xticklabels([f"{z:g}" for z in z_ticks])
    ax_top.set_xlabel("Redshift")

    fig.tight_layout()

    fig.savefig(
        outdir / "bh_to_stellar_evolution_panels.png",
        dpi=200,
    )

    fig.savefig(
        outdir / "bh_to_stellar_evolution_panels.pdf"
    )

    plt.close(fig)

def plot_stellar_mass_evolution_panels(
    hist: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
) -> None:

    summary = summary.copy().set_index("track_id")

    group_order = [
        ("low", "tab:green", r"$0 \leq f_{\rm exsitu} < 0.1$"),
        ("mid", "tab:orange", r"$0.1 \leq f_{\rm exsitu} < 0.4$"),
        ("high", "tab:purple", r"$f_{\rm exsitu} \geq 0.4$"),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        sharex=True,
        sharey=True,
    )

    for ax, (target_group, color, label) in zip(axes, group_order):

        for track_id, g in hist.groupby("track_id"):

            g = g.sort_values("redshift", ascending=False)

            x = g["lookback_time_gyr"].to_numpy(dtype=float)
            y = g["log10_stellar_mass"].to_numpy(dtype=float)

            if track_id not in summary.index:
                continue

            grp = summary.loc[track_id, "exsitu_group"]

            if grp == target_group:
                ax.plot(
                    x,
                    y,
                    lw=1.5,
                    alpha=0.9,
                    color=color,
                )
                ax.scatter(
                    x,
                    y,
                    s=14,
                    color=color,
                )

            else:
                ax.plot(
                    x,
                    y,
                    lw=1.0,
                    alpha=0.10,
                    color="0.75",
                )

        ax.set_title(label)
        ax.set_xlabel("Lookback time [Gyr]")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"$\log_{10}(M_\star/M_\odot)$")

    ax_top = axes[1].twiny()

    z_ticks = np.array(
        [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8],
        dtype=float,
    )

    lb_ticks = COSMO.lookback_time(z_ticks).value

    ax_top.set_xlim(axes[1].get_xlim())
    ax_top.set_xticks(lb_ticks)
    ax_top.set_xticklabels([f"{z:g}" for z in z_ticks])
    ax_top.set_xlabel("Redshift")

    fig.tight_layout()

    fig.savefig(
        outdir / "stellar_mass_evolution_panels.png",
        dpi=200,
    )

    fig.savefig(
        outdir / "stellar_mass_evolution_panels.pdf"
    )

    plt.close(fig)

def plot_size_evolution_panels(
    hist: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
) -> None:

    summary = summary.copy().set_index("track_id")

    group_order = [
        ("low", "tab:green", r"$0 \leq f_{\rm exsitu} < 0.1$"),
        ("mid", "tab:orange", r"$0.1 \leq f_{\rm exsitu} < 0.4$"),
        ("high", "tab:purple", r"$f_{\rm exsitu} \geq 0.4$"),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6),
        sharex=True,
        sharey=True,
    )

    for ax, (target_group, color, label) in zip(axes, group_order):

        for track_id, g in hist.groupby("track_id"):

            g = g.sort_values("redshift", ascending=False)

            x = g["lookback_time_gyr"].to_numpy(dtype=float)
            y = g["log10_rhalf"].to_numpy(dtype=float)

            if track_id not in summary.index:
                continue

            grp = summary.loc[track_id, "exsitu_group"]

            if grp == target_group:
                ax.plot(
                    x,
                    y,
                    lw=1.5,
                    alpha=0.9,
                    color=color,
                )
                ax.scatter(
                    x,
                    y,
                    s=14,
                    color=color,
                )

            else:
                ax.plot(
                    x,
                    y,
                    lw=1.0,
                    alpha=0.10,
                    color="0.75",
                )

        ax.set_title(label)
        ax.set_xlabel("Lookback time [Gyr]")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"$\log_{10}(R_{50}/\mathrm{kpc})$")

    ax_top = axes[1].twiny()

    z_ticks = np.array(
        [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8],
        dtype=float,
    )

    lb_ticks = COSMO.lookback_time(z_ticks).value

    ax_top.set_xlim(axes[1].get_xlim())
    ax_top.set_xticks(lb_ticks)
    ax_top.set_xticklabels([f"{z:g}" for z in z_ticks])
    ax_top.set_xlabel("Redshift")

    fig.tight_layout()

    fig.savefig(
        outdir / "size_evolution_panels.png",
        dpi=200,
    )

    fig.savefig(
        outdir / "size_evolution_panels.pdf"
    )

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

def plot_mass_size_paths(
    hist: pd.DataFrame,
    outdir: Path,
    max_highlight: int = 20,
) -> None:
    """
    Produce two mass-size evolution figures for the tracked relics.

    Figure 1:
        Individual relic trajectories from z~2 to z=0, with arrows between
        every consecutive available snapshot. Galaxies are classified at z=0
        using compactness:
            SRG: compactness >= 9.75
            SAG: compactness < 9.75

    Figure 2:
        Median mass-size tracks for SRGs and SAGs.

    If a 'population' column is present in hist, it is used for classification.
    This allows compact non-relics to be added later without changing the
    plotting logic.

    Only snapshots with 0 <= z <= 2 are shown.
    """

    COMPACTNESS_CUT = 9.75

    # ------------------------------------------------------------------
    # Restrict to z <= 2 and require valid mass-size coordinates
    # ------------------------------------------------------------------
    h = hist.copy()

    h["redshift"] = pd.to_numeric(h["redshift"], errors="coerce")

    h = h[
        np.isfinite(h["redshift"])
        & (h["redshift"] >= 0.0)
        & (h["redshift"] <= 2.0)
        & np.isfinite(h["log10_stellar_mass"])
        & np.isfinite(h["log10_rhalf"])
    ].copy()

    if h.empty:
        print("No valid z<=2 mass-size data available; skipping mass-size evolution plots.")
        return

    # ------------------------------------------------------------------
    # Determine z=0 population class for every tracked galaxy
    # ------------------------------------------------------------------
    population_map = {}

    for track_id, g in h.groupby("track_id"):

        g = g.sort_values("redshift", ascending=True)

        # Present-day point = closest available point to z=0
        z0_idx = np.argmin(np.abs(g["redshift"].to_numpy(dtype=float)))
        z0 = g.iloc[z0_idx]

        # If an explicit population column already exists, use it.
        if "population" in g.columns:
            pop = z0.get("population", None)

            if isinstance(pop, str):
                population_map[track_id] = pop
                continue

        # Otherwise classify relics using their z=0 compactness.
        comp = z0.get("compactness_sigma_1p5", np.nan)

        if np.isfinite(comp):
            if comp >= COMPACTNESS_CUT:
                population_map[track_id] = "SRG"
            else:
                population_map[track_id] = "SAG"
        else:
            population_map[track_id] = "unknown"

    # ==================================================================
    # FIGURE 1: individual mass-size displacement arrows
    # ==================================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    # --------------------------------------------------------------
    # Full z≈2 galaxy population in the background
    # --------------------------------------------------------------
    ax.scatter(
        full_logM,
        full_logR,
        s=2,
        color="lightgrey",
        alpha=0.04,
        rasterized=True,
        zorder=0,
        label="all z≈2 galaxies",
    )

    # We use fixed class colours so the interpretation is unambiguous.
    class_colors = {
        "SRG": "C2",
        "SAG": "C1",
        "compact_nonrelic": "0.45",
        "unknown": "0.6",
    }

    class_labels_done = set()

    for track_id, g in h.groupby("track_id"):

        g = (
            g.sort_values("redshift", ascending=False)
             .drop_duplicates(subset=["redshift"], keep="first")
        )

        if len(g) < 2:
            continue

        x = g["log10_stellar_mass"].to_numpy(dtype=float)
        y = g["log10_rhalf"].to_numpy(dtype=float)

        pop = population_map.get(track_id, "unknown")
        colour = class_colors.get(pop, "0.6")

        # --------------------------------------------------------------
        # Draw faint trajectory
        # --------------------------------------------------------------
        ax.plot(
            x,
            y,
            color=colour,
            lw=0.8,
            alpha=0.08,
            zorder=1,
            rasterized=True,
        )

        # --------------------------------------------------------------
        # Draw arrows between EVERY consecutive snapshot
        # --------------------------------------------------------------
        for i in range(len(x) - 1):

            if not (
                np.isfinite(x[i])
                and np.isfinite(y[i])
                and np.isfinite(x[i + 1])
                and np.isfinite(y[i + 1])
            ):
                continue

            ax.annotate(
                "",
                xy=(x[i + 1], y[i + 1]),
                xytext=(x[i], y[i]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=colour,
                    lw=0.9,
                    alpha=0.30,
                    shrinkA=0,
                    shrinkB=0,
                ),
                zorder=2,
            )

        # --------------------------------------------------------------
        # Mark the individual snapshot positions
        # --------------------------------------------------------------
        ax.scatter(
            x,
            y,
            s=10,
            color=colour,
            alpha=0.28,
            edgecolors="none",
            zorder=3,
            rasterized=True,
        )

        # --------------------------------------------------------------
        # Make z=2 and z=0 especially obvious
        # --------------------------------------------------------------
        z2_idx = np.argmin(
            np.abs(g["redshift"].to_numpy(dtype=float) - 2.0)
        )

        z0_idx = np.argmin(
            np.abs(g["redshift"].to_numpy(dtype=float))
        )

        ax.scatter(
            x[z2_idx],
            y[z2_idx],
            s=42,
            marker="o",
            facecolor="white",
            edgecolor=colour,
            linewidth=1.2,
            zorder=10,
        )

        ax.scatter(
            x[z0_idx],
            y[z0_idx],
            s=55,
            marker="o",
            facecolor=colour,
            edgecolor="black",
            linewidth=0.8,
            zorder=11,
        )

    # ------------------------------------------------------------------
    # Compactness threshold
    # ------------------------------------------------------------------
    xline = np.linspace(
        9.0,
        12.0,
        300,
    )

    yline = (
        xline - COMPACTNESS_CUT
    ) / 1.5

    ax.plot(
        xline,
        yline,
        "--",
        color="black",
        lw=1.4,
        alpha=0.8,
        label=(
            rf"$\log_{{10}}\Sigma_{{1.5}}="
            rf"{COMPACTNESS_CUT:.2f}$"
        ),
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
        [0], [0],
        marker="o",
        markerfacecolor="lightgrey",
        markeredgecolor="none",
        color="none",
        markersize=5,
        label="all z≈2 galaxies",
        ),
        Line2D(
            [0], [0],
            color="C2",
            lw=2.0,
            label="SRGs (z=0 classification)",
        ),
        Line2D(
            [0], [0],
            color="C1",
            lw=2.0,
            label="SAGs (z=0 classification)",
        ),
        Line2D(
            [0], [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            color="none",
            markersize=7,
            label="z≈2 position",
        ),
        Line2D(
            [0], [0],
            marker="o",
            markerfacecolor="C2",
            markeredgecolor="black",
            color="none",
            markersize=7,
            label="z=0 position",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        fontsize=10,
        frameon=False,
        loc="best",
    )

    ax.set_xlabel(
        r"$\log_{10}(M_\star/M_\odot)$"
    )
    ax.set_ylabel(
        r"$\log_{10}(R_{50}/\mathrm{kpc})$"
    )

    ax.set_title(
        "Individual relic evolution through the mass-size plane"
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        outdir / "mass_size_displacements_z2_z0.png",
        dpi=250,
        bbox_inches="tight",
    )

    fig.savefig(
        outdir / "mass_size_displacements_z2_z0.pdf",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved individual mass-size displacement figure:",
        outdir / "mass_size_displacements_z2_z0.png",
    )

    # ==================================================================
    # FIGURE 2: median SRG/SAG tracks
    # ==================================================================

    # Collect all available redshifts in the z<=2 range.
    redshifts = np.sort(
        h["redshift"].dropna().unique()
    )[::-1]

    # Use a small tolerance when grouping approximately identical
    # redshifts from independent files.
    def get_median_track(population_name):

        track_rows = []

        for z in redshifts:

            vals_m = []
            vals_r = []

            for track_id, g in h.groupby("track_id"):

                if population_map.get(track_id) != population_name:
                    continue

                dz = np.abs(
                    g["redshift"].to_numpy(dtype=float) - z
                )

                if dz.size == 0:
                    continue

                idx = np.argmin(dz)

                # Require that this snapshot is genuinely the same redshift.
                if dz[idx] > 1e-4:
                    continue

                row = g.iloc[idx]

                if (
                    np.isfinite(row["log10_stellar_mass"])
                    and np.isfinite(row["log10_rhalf"])
                ):
                    vals_m.append(
                        float(row["log10_stellar_mass"])
                    )
                    vals_r.append(
                        float(row["log10_rhalf"])
                    )

            if len(vals_m) == 0:
                continue

            track_rows.append(
                {
                    "redshift": z,
                    "logM": np.nanmedian(vals_m),
                    "logR": np.nanmedian(vals_r),
                    "n": len(vals_m),
                }
            )

        return pd.DataFrame(track_rows)

    median_srg = get_median_track("SRG")
    median_sag = get_median_track("SAG")

    # ------------------------------------------------------------------
    # Median-track figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    if not median_sag.empty:

        ax.plot(
            median_sag["logM"],
            median_sag["logR"],
            color="C1",
            lw=3.0,
            marker="o",
            markersize=5,
            label="median SAG",
            zorder=10,
        )

        # Mark z=2 and z=0
        ax.scatter(
            median_sag.iloc[0]["logM"],
            median_sag.iloc[0]["logR"],
            s=80,
            marker="o",
            facecolor="white",
            edgecolor="C1",
            linewidth=2,
            zorder=20,
        )

        ax.scatter(
            median_sag.iloc[-1]["logM"],
            median_sag.iloc[-1]["logR"],
            s=90,
            marker="o",
            facecolor="C1",
            edgecolor="black",
            linewidth=1,
            zorder=21,
        )

    if not median_srg.empty:

        ax.plot(
            median_srg["logM"],
            median_srg["logR"],
            color="C2",
            lw=3.0,
            marker="o",
            markersize=5,
            label="median SRG",
            zorder=11,
        )

        # Mark z=2 and z=0
        ax.scatter(
            median_srg.iloc[0]["logM"],
            median_srg.iloc[0]["logR"],
            s=80,
            marker="o",
            facecolor="white",
            edgecolor="C2",
            linewidth=2,
            zorder=20,
        )

        ax.scatter(
            median_srg.iloc[-1]["logM"],
            median_srg.iloc[-1]["logR"],
            s=90,
            marker="o",
            facecolor="C2",
            edgecolor="black",
            linewidth=1,
            zorder=21,
        )

    # ------------------------------------------------------------------
    # Label approximate redshift positions
    # ------------------------------------------------------------------
    for track_df, colour in [
        (median_srg, "C2"),
        (median_sag, "C1"),
    ]:

        if track_df.empty:
            continue

        for target_z in [2.0, 1.0, 0.5, 0.0]:

            idx = np.argmin(
                np.abs(
                    track_df["redshift"].to_numpy(dtype=float)
                    - target_z
                )
            )

            if abs(
                track_df.iloc[idx]["redshift"] - target_z
            ) < 0.15:

                ax.annotate(
                    f"z={track_df.iloc[idx]['redshift']:.1f}",
                    (
                        track_df.iloc[idx]["logM"],
                        track_df.iloc[idx]["logR"],
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    color=colour,
                )

    # ------------------------------------------------------------------
    # Compactness threshold
    # ------------------------------------------------------------------
    xline = np.linspace(
        9.0,
        12.0,
        300,
    )

    yline = (
        xline - COMPACTNESS_CUT
    ) / 1.5

    ax.plot(
        xline,
        yline,
        "--",
        color="black",
        lw=1.4,
        alpha=0.8,
        label=(
            rf"$\log_{{10}}\Sigma_{{1.5}}="
            rf"{COMPACTNESS_CUT:.2f}$"
        ),
    )

    ax.set_xlabel(
        r"$\log_{10}(M_\star/M_\odot)$"
    )
    ax.set_ylabel(
        r"$\log_{10}(R_{50}/\mathrm{kpc})$"
    )

    ax.set_title(
        "Median relic evolution through the mass-size plane"
    )

    ax.grid(alpha=0.25)

    ax.legend(
        fontsize=10,
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        outdir / "mass_size_median_tracks_z2_z0.png",
        dpi=250,
        bbox_inches="tight",
    )

    fig.savefig(
        outdir / "mass_size_median_tracks_z2_z0.pdf",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved median mass-size evolution figure:",
        outdir / "mass_size_median_tracks_z2_z0.png",
    )


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

def plot_high_BH_ratio_galaxies(hist, top_ids, outdir):

    fig, (ax1, ax2) = plt.subplots(
        1,2,
        figsize=(12,5),
        sharex=True
    )

    for tid, g in hist.groupby("track_id"):

        if tid not in top_ids:
            continue

        g = g.sort_values("redshift", ascending=False)

        x = g["lookback_time_gyr"]

        ax1.plot(
            x,
            g["log10_stellar_mass"],
            lw=2
        )

        y = np.where(
            g["bh_mass"]>0,
            np.log10(g["bh_mass"]),
            np.nan
        )

        pos = g["bh_mass"] > 0

        ax2.plot(
            x[pos],
            np.log10(g.loc[pos, "bh_mass"]),
            lw=2,
            label=str(tid),
        )

        zero = g["bh_mass"] == 0

        if zero.any():
            ax2.scatter(
                x[zero],
                np.full(zero.sum(), 3.0),   # place them below the plotted range
                marker="v",
                s=30,
            )

    ax1.set_ylabel(r"$\log M_\star$")
    ax2.set_ylabel(r"$\log M_{\rm BH}$")

    ax1.set_xlabel("Lookback time [Gyr]")
    ax2.set_xlabel("Lookback time [Gyr]")

    ax2.legend(fontsize=6)

    fig.tight_layout()

    fig.savefig(
        outdir/"highest_BH_ratio_tracks.png",
        dpi=200
    )

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


# -----------------------------------------------------------------------------
# Stripping diagnostic figure
# -----------------------------------------------------------------------------
def plot_stripping_diagnostics(
    hist: pd.DataFrame,
    outdir: Path,
) -> None:
    """
    Diagnose whether present-day satellite relics show evolutionary signatures
    consistent with substantial post-infall stellar stripping.

    For each z=0 satellite relic with an identifiable first central -> satellite
    transition:

      - infall is defined by the first central -> satellite transition;
      - the reference stellar mass is the maximum stellar mass reached before
        that transition;
      - the reference size is the R50 measured at the snapshot where the
        pre-infall stellar mass peaks;
      - post-infall mass and size evolution are normalised to this common
        pre-infall peak state;
      - the direct structural change from the pre-infall peak state to z=0 is
        quantified in Delta log M*, Delta log R50, and Delta log Sigma_1.5.

    This deliberately distinguishes the "last central snapshot" from the
    "maximum pre-infall stellar mass". The former marks the dynamical
    transition, while the latter provides the more physically meaningful
    reference for cumulative stellar-mass loss.

    Produces:
      1. stripping_diagnostics_3panel.pdf
      2. stripping_diagnostics_3panel.png
      3. stripping_diagnostics_per_galaxy.csv
    """

    records = []
    tracks_for_plot = []
    infall_lookbacks_for_plot = []

    # ------------------------------------------------------------------
    # Identify z=0 satellites and their first central -> satellite
    # transition.
    # ------------------------------------------------------------------
    for track_id, g in hist.groupby("track_id"):

        g = (
            g.dropna(subset=["redshift", "is_central"])
             .sort_values("redshift", ascending=False)
             .reset_index(drop=True)
        )

        if len(g) < 2:
            continue

        # Present-day snapshot = row closest to z=0
        z0_idx = np.argmin(
            np.abs(g["redshift"].to_numpy(dtype=float))
        )
        z0 = g.iloc[z0_idx]

        # Only present-day satellites
        if z0["is_central"] > 0.5:
            continue

        # --------------------------------------------------------------
        # First central -> satellite transition
        # --------------------------------------------------------------
        transition = None

        for i in range(len(g) - 1):
            before = g.iloc[i]
            after = g.iloc[i + 1]

            if (
                before["is_central"] > 0.5
                and after["is_central"] <= 0.5
            ):
                transition = i
                break

        if transition is None:
            continue

        pre = g.iloc[transition]
        post = g.iloc[transition + 1]

        # --------------------------------------------------------------
        # Basic validity checks
        # --------------------------------------------------------------
        required = [
            pre["stellar_mass"],
            post["stellar_mass"],
            z0["stellar_mass"],
            pre["rhalf"],
            post["rhalf"],
            z0["rhalf"],
            pre["lookback_time_gyr"],
            post["lookback_time_gyr"],
        ]

        if not np.all(np.isfinite(required)):
            continue

        if (
            pre["stellar_mass"] <= 0
            or post["stellar_mass"] <= 0
            or z0["stellar_mass"] <= 0
            or pre["rhalf"] <= 0
            or post["rhalf"] <= 0
            or z0["rhalf"] <= 0
        ):
            continue

        # --------------------------------------------------------------
        # Infall time
        #
        # The transition occurs between the last-central snapshot and
        # the first-satellite snapshot. We use their midpoint in
        # lookback time as the estimated infall time.
        # --------------------------------------------------------------
        lb_pre = float(pre["lookback_time_gyr"])
        lb_post = float(post["lookback_time_gyr"])
        lb_infall = 0.5 * (lb_pre + lb_post)

        infall_lookbacks_for_plot.append(lb_infall)

        # Positive = time after infall
        g = g.copy()

        g["time_since_infall_gyr"] = (
            lb_infall
            - g["lookback_time_gyr"].astype(float)
        )

        # --------------------------------------------------------------
        # PRE-INFALL PEAK REFERENCE STATE
        #
        # Find the maximum stellar mass before / at the transition.
        # The radius and compactness at exactly this snapshot define the
        # corresponding structural reference state.
        # --------------------------------------------------------------
        pre_infall_part = g.iloc[:transition + 1].copy()

        m_pre_array = (
            pre_infall_part["stellar_mass"]
            .to_numpy(dtype=float)
        )

        valid_peak = (
            np.isfinite(m_pre_array)
            & (m_pre_array > 0)
        )

        if not np.any(valid_peak):
            continue

        peak_local_idx = np.where(valid_peak)[0][
            np.argmax(m_pre_array[valid_peak])
        ]

        peak = pre_infall_part.iloc[peak_local_idx]

        m_peak_pre = float(peak["stellar_mass"])
        r_peak_pre = float(peak["rhalf"])

        # Require a valid reference radius
        if not np.isfinite(r_peak_pre) or r_peak_pre <= 0:
            continue

        compact_peak_pre = float(
            peak["compactness_sigma_1p5"]
        )

        # --------------------------------------------------------------
        # z=0 values
        # --------------------------------------------------------------
        m_z0 = float(z0["stellar_mass"])
        r_z0 = float(z0["rhalf"])
        compact_z0 = float(
            z0["compactness_sigma_1p5"]
        )

        # --------------------------------------------------------------
        # Main stripping diagnostics
        # --------------------------------------------------------------

        # Fractional stellar-mass loss relative to the maximum mass
        # attained before infall.
        f_loss_peak_pre = (
            1.0 - m_z0 / m_peak_pre
        )

        # Logarithmic structural changes from the peak pre-infall state
        # to z=0.
        dlogm = np.log10(
            m_z0 / m_peak_pre
        )

        dlogr = np.log10(
            r_z0 / r_peak_pre
        )

        # Delta compactness follows directly from
        # Sigma_1.5 = M_star / R_50^1.5
        dcompact = (
            compact_z0
            - compact_peak_pre
        )

        records.append({
            "track_id": int(track_id),

            "redshift_infall_midpoint": 0.5 * (
                float(pre["redshift"])
                + float(post["redshift"])
            ),
            "lookback_infall_gyr": lb_infall,

            # Last-central quantities retained for reference
            "stellar_mass_pre_infall": float(pre["stellar_mass"]),
            "rhalf_pre_infall": float(pre["rhalf"]),

            # Peak pre-infall reference state
            "redshift_peak_pre_infall": float(
                peak["redshift"]
            ),
            "lookback_peak_pre_infall_gyr": float(
                peak["lookback_time_gyr"]
            ),
            "stellar_mass_peak_pre_infall": m_peak_pre,
            "rhalf_peak_pre_infall": r_peak_pre,
            "compactness_peak_pre_infall": compact_peak_pre,

            # Present day
            "stellar_mass_z0": m_z0,
            "rhalf_z0": r_z0,
            "compactness_z0": compact_z0,

            # Main stripping diagnostics
            "f_mass_loss_peak_pre_infall_to_z0":
                f_loss_peak_pre,

            "delta_log10_mstar":
                dlogm,

            "delta_log10_rhalf":
                dlogr,

            "delta_compactness":
                dcompact,

            # Dynamical classification
            "central_before_infall":
                float(pre["is_central"]),

            "central_after_infall":
                float(post["is_central"]),

            "central_z0":
                float(z0["is_central"]),
        })

        # --------------------------------------------------------------
        # Post-infall evolutionary track
        #
        # IMPORTANT:
        # The time coordinate is still measured from the inferred infall
        # time. The *normalisation* is now the peak pre-infall state.
        # Thus t=0 does not necessarily correspond to M/M_peak = 1.
        # --------------------------------------------------------------
        gg = g[
            np.isfinite(g["time_since_infall_gyr"])
            & np.isfinite(g["stellar_mass"])
            & np.isfinite(g["rhalf"])
            & (g["stellar_mass"] > 0)
            & (g["rhalf"] > 0)
            & (g["time_since_infall_gyr"] >= 0)
        ].copy()

        if len(gg) >= 2:

            gg["m_norm"] = (
                gg["stellar_mass"]
                / m_peak_pre
            )

            gg["r_norm"] = (
                gg["rhalf"]
                / r_peak_pre
            )

            tracks_for_plot.append(gg)

    # ------------------------------------------------------------------
    # Assemble diagnostics table
    # ------------------------------------------------------------------
    diag = pd.DataFrame(records)

    if diag.empty:
        print(
            "No suitable z=0 satellite relics with an identifiable "
            "central -> satellite transition were found."
        )
        return

    diag_path = (
        outdir /
        "stripping_diagnostics_per_galaxy.csv"
    )

    diag.to_csv(
        diag_path,
        index=False
    )

    # ------------------------------------------------------------------
    # Numerical diagnostics
    # ------------------------------------------------------------------
    print("\n====================================================")
    print("STRIPPING DIAGNOSTICS")
    print("====================================================")

    n = len(diag)

    print(
        f"Usable present-day satellite relics: {n}"
    )

    loss = (
        diag["f_mass_loss_peak_pre_infall_to_z0"]
        .to_numpy(dtype=float)
    )

    print(
        "\nStellar-mass loss relative to "
        "pre-infall peak:"
    )

    print(
        f"  median = "
        f"{100*np.nanmedian(loss):.2f}%"
    )

    print(
        f"  16th percentile = "
        f"{100*np.nanpercentile(loss, 16):.2f}%"
    )

    print(
        f"  84th percentile = "
        f"{100*np.nanpercentile(loss, 84):.2f}%"
    )

    for threshold in [0.05, 0.10, 0.20]:

        nn = int(
            np.sum(loss > threshold)
        )

        print(
            f"  > {100*threshold:.0f}% loss: "
            f"{nn}/{n} "
            f"({100*nn/n:.1f}%)"
        )

    # ------------------------------------------------------------------
    # Direct peak-pre-infall -> z=0 structural changes
    # ------------------------------------------------------------------
    dlogm = (
        diag["delta_log10_mstar"]
        .to_numpy(dtype=float)
    )

    dlogr = (
        diag["delta_log10_rhalf"]
        .to_numpy(dtype=float)
    )

    dcomp = (
        diag["delta_compactness"]
        .to_numpy(dtype=float)
    )

    print(
        "\nPeak pre-infall -> z=0 changes:"
    )

    print(
        f"  median Delta log M* = "
        f"{np.nanmedian(dlogm):.3f} dex"
    )

    print(
        f"  median Delta log R50 = "
        f"{np.nanmedian(dlogr):.3f} dex"
    )

    print(
        f"  median Delta compactness = "
        f"{np.nanmedian(dcomp):.3f} dex"
    )

    # Descriptive, not by themselves proof of stripping
    mass_loss = dlogm < 0
    size_decrease = dlogr < 0
    compactness_increase = dcomp > 0

    mass_and_size_loss = (
        mass_loss
        & size_decrease
    )

    print(
        f"  mass loss AND size decrease: "
        f"{np.sum(mass_and_size_loss)}/{n} "
        f"({100*np.mean(mass_and_size_loss):.1f}%)"
    )

    print(
        f"  compactness increase: "
        f"{np.sum(compactness_increase)}/{n} "
        f"({100*np.mean(compactness_increase):.1f}%)"
    )

    print("\nSaved diagnostics:")
    print(diag_path)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": 13,
    })

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
        gridspec_kw={"wspace": 0.28},
    )

    ax_mass, ax_size, ax_change = axes

    # ==================================================================
    # PANEL (a): stellar-mass evolution
    # ==================================================================
    for gg in tracks_for_plot:

        ax_mass.plot(
            gg["time_since_infall_gyr"],
            gg["m_norm"],
            color="0.55",
            lw=1.0,
            alpha=0.15,
            rasterized=True,
            zorder=1,
        )

    if tracks_for_plot:

        all_x = np.concatenate([
            gg["time_since_infall_gyr"]
            .to_numpy(dtype=float)
            for gg in tracks_for_plot
        ])

        all_y = np.concatenate([
            gg["m_norm"]
            .to_numpy(dtype=float)
            for gg in tracks_for_plot
        ])

        xmax = np.nanpercentile(
            all_x,
            99
        )

        bins = np.linspace(
            0,
            xmax,
            20
        )

        xc = 0.5 * (
            bins[:-1]
            + bins[1:]
        )

        med = np.full_like(
            xc,
            np.nan
        )

        lo = np.full_like(
            xc,
            np.nan
        )

        hi = np.full_like(
            xc,
            np.nan
        )

        for i in range(len(xc)):

            vals = all_y[
                (all_x >= bins[i])
                & (all_x < bins[i + 1])
            ]

            if len(vals) >= 5:

                med[i] = np.nanmedian(
                    vals
                )

                lo[i] = np.nanpercentile(
                    vals,
                    16
                )

                hi[i] = np.nanpercentile(
                    vals,
                    84
                )

        good = (
            np.isfinite(med)
            & np.isfinite(lo)
            & np.isfinite(hi)
        )

        if np.any(good):

            z0_m_norm = np.array([
                gg.iloc[np.argmin(np.abs(gg["redshift"].to_numpy(dtype=float)))]["m_norm"]
                for gg in tracks_for_plot
            ])

            median_mass_final = np.nanmedian(z0_m_norm)

            ax_mass.fill_between(
                xc[good],
                lo[good],
                hi[good],
                color="C0",
                alpha=0.20,
                zorder=4,
            )

            ax_mass.plot(
                xc[good],
                med[good],
                color="C0",
                lw=2.8,
                zorder=5,
                label=fr"median (16th--84th percentile; $z=0$: {median_mass_final:.2f})",
            )

    # Peak-mass reference
    ax_mass.axhline(
        1.0,
        color="black",
        linestyle="--",
        lw=1.2,
        zorder=3,
    )

    # Infall itself
    ax_mass.axvline(
        0.0,
        color="0.35",
        linestyle=":",
        lw=1.1,
        zorder=3,
    )

    ax_mass.set_xlabel(
        "Time since infall [Gyr]"
    )

    ax_mass.set_ylabel(
        r"$M_\star/M_{\star,\mathrm{peak,pre-infall}}$"
    )

    ax_mass.set_title(
        "(a) Stellar-mass evolution"
    )

    ax_mass.grid(
        alpha=0.25
    )

    ax_mass.legend(
        loc="best",
        fontsize=11,
        frameon=True,
        framealpha=0.9,
    )

    # ==================================================================
    # PANEL (b): size evolution
    # ==================================================================
    for gg in tracks_for_plot:

        ax_size.plot(
            gg["time_since_infall_gyr"],
            gg["r_norm"],
            color="0.55",
            lw=1.0,
            alpha=0.15,
            rasterized=True,
            zorder=1,
        )

    if tracks_for_plot:

        all_x = np.concatenate([
            gg["time_since_infall_gyr"]
            .to_numpy(dtype=float)
            for gg in tracks_for_plot
        ])

        all_y = np.concatenate([
            gg["r_norm"]
            .to_numpy(dtype=float)
            for gg in tracks_for_plot
        ])

        xmax = np.nanpercentile(
            all_x,
            99
        )

        bins = np.linspace(
            0,
            xmax,
            20
        )

        xc = 0.5 * (
            bins[:-1]
            + bins[1:]
        )

        med = np.full_like(
            xc,
            np.nan
        )

        lo = np.full_like(
            xc,
            np.nan
        )

        hi = np.full_like(
            xc,
            np.nan
        )

        for i in range(len(xc)):

            vals = all_y[
                (all_x >= bins[i])
                & (all_x < bins[i + 1])
            ]

            if len(vals) >= 5:

                med[i] = np.nanmedian(
                    vals
                )

                lo[i] = np.nanpercentile(
                    vals,
                    16
                )

                hi[i] = np.nanpercentile(
                    vals,
                    84
                )

        good = (
            np.isfinite(med)
            & np.isfinite(lo)
            & np.isfinite(hi)
        )

        if np.any(good):

            z0_r_norm = np.array([
                gg.iloc[np.argmin(np.abs(gg["redshift"].to_numpy(dtype=float)))]["r_norm"]
                for gg in tracks_for_plot
            ])

            median_size_final = np.nanmedian(z0_r_norm)

            ax_size.fill_between(
                xc[good],
                lo[good],
                hi[good],
                color="C1",
                alpha=0.20,
                zorder=4,
            )

            ax_size.plot(
                xc[good],
                med[good],
                color="C1",
                lw=2.8,
                zorder=5,
                label=fr"median (16th--84th percentile; $z=0$: {median_size_final:.2f})",
            )

    # Peak-state reference
    ax_size.axhline(
        1.0,
        color="black",
        linestyle="--",
        lw=1.2,
        zorder=3,
    )

    # Infall itself
    ax_size.axvline(
        0.0,
        color="0.35",
        linestyle=":",
        lw=1.1,
        zorder=3,
    )

    ax_size.set_xlabel(
        "Time since infall [Gyr]"
    )

    ax_size.set_ylabel(
        r"$R_{50}/R_{50,\mathrm{peak,pre-infall}}$"
    )

    ax_size.set_title(
        "(b) Size evolution"
    )

    ax_size.grid(
        alpha=0.25
    )

    ax_size.legend(
        loc="best",
        fontsize=11,
        frameon=True,
        framealpha=0.9,
    )

    # ==================================================================
    # PANEL (c): peak pre-infall -> z=0 structural change
    # ==================================================================
    x = diag[
        "delta_log10_mstar"
    ].to_numpy(dtype=float)

    y = diag[
        "delta_log10_rhalf"
    ].to_numpy(dtype=float)

    c = diag[
        "delta_compactness"
    ].to_numpy(dtype=float)

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(c)
    )

    sc = ax_change.scatter(
        x[finite],
        y[finite],
        c=c[finite],
        cmap="viridis",
        s=32,
        alpha=0.80,
        edgecolors="none",
        rasterized=True,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax_change,
        pad=0.02,
    )

    cbar.set_label(
        r"$\Delta\log_{10}\Sigma_{1.5}$ [dex]"
    )

    # Zero-change lines
    ax_change.axvline(
        0.0,
        color="0.35",
        linestyle="--",
        lw=1.0,
    )

    ax_change.axhline(
        0.0,
        color="0.35",
        linestyle="--",
        lw=1.0,
    )

    # Constant compactness:
    #
    # Delta log Sigma = Delta log M - 1.5 Delta log R = 0
    #
    # => Delta log R = Delta log M / 1.5
    if np.any(finite):

        xx = np.linspace(
            np.nanmin(x[finite]) - 0.05,
            np.nanmax(x[finite]) + 0.05,
            200,
        )

        ax_change.plot(
            xx,
            xx / 1.5,
            color="black",
            linestyle=":",
            lw=1.5,
            label="constant compactness",
        )

    ax_change.set_xlabel(
        r"$\Delta\log_{10}(M_\star)$"
    )

    ax_change.set_ylabel(
        r"$\Delta\log_{10}(R_{50})$"
    )

    ax_change.set_title(
        "(c) Peak pre-infall to $z=0$ evolution"
    )

    ax_change.grid(
        alpha=0.25
    )

    ax_change.legend(
        fontsize=10
    )

    # ------------------------------------------------------------------
    # Compact quantitative summary
    # ------------------------------------------------------------------
    median_dm = np.nanmedian(dlogm)
    median_dr = np.nanmedian(dlogr)
    median_dc = np.nanmedian(dcomp)
    median_loss = np.nanmedian(loss)

    summary_text = (
        rf"$\mathrm{{median}}\ \Delta\log M_\star"
        rf"={median_dm:.2f}$ dex" "\n"
        rf"$\mathrm{{median}}\ \Delta\log R_{{50}}"
        rf"={median_dr:.2f}$ dex" "\n"
        rf"$\mathrm{{median}}\ \Delta\Sigma_{{1.5}}"
        rf"={median_dc:.2f}$ dex" "\n"
        rf"$\mathrm{{median}}\ f_{{\rm loss}}"
        rf"={100*median_loss:.1f}\%$"
    )

    ax_change.text(
        0.03,
        0.97,
        summary_text,
        transform=ax_change.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            edgecolor="0.7",
            alpha=0.85,
            pad=3,
        ),
    )

    # ------------------------------------------------------------------
    # Final formatting
    # ------------------------------------------------------------------
    for ax in axes:

        ax.tick_params(
            axis="both",
            labelsize=12,
        )

    fig.tight_layout()

    pdf_path = (
        outdir /
        "stripping_diagnostics_3panel.pdf"
    )

    png_path = (
        outdir /
        "stripping_diagnostics_3panel.png"
    )

    fig.savefig(
        pdf_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "\nSaved stripping diagnostic figure:"
    )
    print(pdf_path)
    print(png_path)

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

    topBH = (
        summary
        .sort_values("bh_to_stellar_z0", ascending=False)
        .head(15)
    )
    print("\nHighest BH/M* galaxies")
    print(
        topBH[
            [
                "track_id",
                "bh_to_stellar_z0",
                "log10_mstar_z2",
                "log10_mstar_z0",
                "delta_log10_mstar",
            ]
        ].to_string(index=False)
    )

    top_ids = set(topBH["track_id"])

    print("\nMedian evolution by ex-situ group")
    print("---------------------------------")

    group_rows = []

    for grp in ["low", "mid", "high"]:

        s = summary[summary["exsitu_group"] == grp].copy()

        if s.empty:
            print(f"{grp:4s}: no galaxies")
            continue

        # Compactness
        compact_z2 = np.nanmedian(s["compactness_z2"])
        compact_z15 = np.nanmedian(s["compactness_z15"])
        compact_z0 = np.nanmedian(s["compactness_z0"])
        delta_compact = compact_z0 - compact_z2
        delta_compact15 = compact_z0 - compact_z15

        # Stellar mass
        mass_z2 = np.nanmedian(s["log10_mstar_z2"])
        mass_z0 = np.nanmedian(s["log10_mstar_z0"])
        delta_mass = mass_z0 - mass_z2

        # Size
        size_z2 = np.nanmedian(s["log10_rhalf_z2"])
        size_z0 = np.nanmedian(s["log10_rhalf_z0"])
        delta_size = size_z0 - size_z2

        # BH / stellar mass
        bh_z2 = np.nanmedian(s["bh_to_stellar_z2"])
        bh_z0 = np.nanmedian(s["bh_to_stellar_z0"])
        delta_bh = bh_z0 - bh_z2

        group_rows.append({
            "exsitu_group": grp,
            "n_galaxies": len(s),

            "median_compactness_z2": compact_z2,
            "median_compactness_z0": compact_z0,
            "delta_median_compactness": delta_compact,
            "median_compactness_z15": compact_z15,
            "delta_median_compactness_z15": delta_compact15,

            "median_log10_mstar_z2": mass_z2,
            "median_log10_mstar_z0": mass_z0,
            "delta_median_log10_mstar": delta_mass,

            "median_log10_rhalf_z2": size_z2,
            "median_log10_rhalf_z0": size_z0,
            "delta_median_log10_rhalf": delta_size,

            "median_bh_to_stellar_z2": bh_z2,
            "median_bh_to_stellar_z0": bh_z0,
            "delta_median_bh_to_stellar": delta_bh,
        })

        print(f"\n{grp}")
        print(f"  N                  = {len(s)}")

        print(
            f"  compactness (z=2→0):   "
            f"z2={compact_z2:.3f}, "
            f"z0={compact_z0:.3f}, "
            f"Δ={delta_compact:.3f}"
        )

        print(
            f"  compactness (z=1.5→0): "
            f"z1.5={compact_z15:.3f}, "
            f"z0={compact_z0:.3f}, "
            f"Δ={delta_compact15:.3f}"
        )

        print(
            f"  log M*:      z2={mass_z2:.3f}, "
            f"z0={mass_z0:.3f}, "
            f"delta={delta_mass:.3f}"
        )

        print(
            f"  log R50:     z2={size_z2:.3f}, "
            f"z0={size_z0:.3f}, "
            f"delta={delta_size:.3f}"
        )

        print(
            f"  BH/M*:       z2={bh_z2:.3e}, "
            f"z0={bh_z0:.3e}, "
            f"delta={delta_bh:.3e}"
        )

    group_summary = pd.DataFrame(group_rows)

    group_summary.to_csv(
        args.output_dir / "evolution_group_medians.csv",
        index=False,
    )

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
    plot_bh_mass_evolution_panels(hist, summary, args.output_dir)
    plot_bh_evolution_panels(hist, summary, args.output_dir)
    plot_stellar_mass_evolution_panels(hist, summary, args.output_dir)
    plot_size_evolution_panels(hist, summary, args.output_dir)

    plot_high_BH_ratio_galaxies(hist, top_ids, args.output_dir)

    plot_ssfr_evolution(hist, args.output_dir, max_highlight=args.max_highlight)
    plot_mass_size_paths(hist, args.output_dir, max_highlight=args.max_highlight)
    plot_central_satellite_transitions(hist, args.output_dir)

    plot_stripping_diagnostics(hist, args.output_dir)

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
