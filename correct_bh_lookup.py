#!/usr/bin/env python3

from pathlib import Path
import glob
import re
import numpy as np
import pandas as pd

DATA_DIR = "/mnt/su3-pro/clagos/COLIBRE/Runs/L200_m6/Thermal/ProcessedData"
OUTFILE = "corrected_bh_mass_lookup.csv"

SNAPSHOT_COLUMNS_32 = [
    "halo_catalogue_index","is_central","x","y","z",
    "stellar_mass","sfr_instant","rhalf_stars",
    "h1","h2","kappa_star","kappa_gas",
    "disc_to_total","jstar","stellar_age",
    "Z1","Z2","dust",
    "descendant_id","track_id",
    "vx","vy","vz",
    "hostx","hosty","hostz",
    "halo_mass",
    "bh_mass",
    "bh_accretion_rate",
    "bh_thermal_energy",
    "n_agn_events",
    "host_fof_halo_id",
]

SNAPSHOT_COLUMNS_33 = SNAPSHOT_COLUMNS_32.copy()
SNAPSHOT_COLUMNS_33.insert(27, "sfr_100myr")


def schema(fname):
    ncols = pd.read_csv(
        fname,
        sep=r"\s+",
        header=None,
        nrows=1,
        engine="python"
    ).shape[1]
    return SNAPSHOT_COLUMNS_33 if ncols == 33 else SNAPSHOT_COLUMNS_32


def parse_z(fname: str) -> float:
    m = re.search(r"_z([0-9]+(?:\.[0-9]+)?)\.txt$", Path(fname).name)
    return float(m.group(1)) if m else np.inf


files = sorted(
    glob.glob(str(Path(DATA_DIR) / "GalaxyProperties_SFR_GE_0_z*.txt")),
    key=parse_z,   # z=0 first, then 0.1, 0.2, ...
)

lookup = {}

for fname in files:
    snap = Path(fname).name
    print(snap)

    df = pd.read_csv(
        fname,
        sep=r"\s+",
        header=None,
        names=schema(fname),
        engine="python",
        usecols=["track_id", "bh_mass", "halo_catalogue_index"]
    )

    good = df["bh_mass"] > 0

    # store only the first non-zero BH mass seen for each TrackID
    for _, row in df.loc[good].iterrows():
        tid = int(row.track_id)
        if tid in lookup:
            continue

        lookup[tid] = (
            float(row.bh_mass),
            int(row.halo_catalogue_index),
            snap
        )

rows = [
    {
        "track_id": tid,
        "HaloCatalogueIndex": halo,
        "corrected_bh_mass": bh,
        "first_snapshot_with_bh": snap,
    }
    for tid, (bh, halo, snap) in lookup.items()
]

out = pd.DataFrame(rows)
out.to_csv(OUTFILE, index=False)

print(f"\nSaved {len(out)} corrected BH masses")
print(OUTFILE)

for tid in [74285, 682377, 325018]:
    idx = np.where(track_id.astype(np.int64) == tid)[0][0]
    print("tid =", tid)
    print("replacement BH mass =", bh_mass[idx])
    print("BH ratio =", log_bh_ratio[idx])


# Test if assignment of corrected black hole masses is correct

# from pathlib import Path
# import glob
# import re
# import numpy as np
# import pandas as pd

# DATA_DIR = "/mnt/su3-pro/clagos/COLIBRE/Runs/L200_m6/Thermal/ProcessedData"
# LOOKUP_CSV = "corrected_bh_mass_lookup.csv"

# TEST_TIDS = [74285, 682377, 325018]

# SNAPSHOT_COLUMNS_32 = [
#     "halo_catalogue_index","is_central","x","y","z",
#     "stellar_mass","sfr_instant","rhalf_stars",
#     "h1","h2","kappa_star","kappa_gas",
#     "disc_to_total","jstar","stellar_age",
#     "Z1","Z2","dust",
#     "descendant_id","track_id",
#     "vx","vy","vz",
#     "hostx","hosty","hostz",
#     "halo_mass",
#     "bh_mass",
#     "bh_accretion_rate",
#     "bh_thermal_energy",
#     "n_agn_events",
#     "host_fof_halo_id",
# ]

# SNAPSHOT_COLUMNS_33 = SNAPSHOT_COLUMNS_32.copy()
# SNAPSHOT_COLUMNS_33.insert(27, "sfr_100myr")

# def schema(fname):
#     ncols = pd.read_csv(
#         fname,
#         sep=r"\s+",
#         header=None,
#         nrows=1,
#         engine="python"
#     ).shape[1]
#     return SNAPSHOT_COLUMNS_33 if ncols == 33 else SNAPSHOT_COLUMNS_32

# def parse_z(fname: str) -> float:
#     m = re.search(r"_z([0-9]+(?:\.[0-9]+)?)\.txt$", Path(fname).name)
#     return float(m.group(1)) if m else np.inf

# files = sorted(
#     glob.glob(str(Path(DATA_DIR) / "GalaxyProperties_SFR_GE_0_z*.txt")),
#     key=parse_z,
# )

# lookup = {}
# history = {tid: [] for tid in TEST_TIDS}

# for fname in files:
#     snap = Path(fname).name
#     z = parse_z(fname)
#     print("Reading", snap)

#     df = pd.read_csv(
#         fname,
#         sep=r"\s+",
#         header=None,
#         names=schema(fname),
#         engine="python",
#         usecols=["track_id", "bh_mass", "halo_catalogue_index"]
#     )

#     df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").astype("Int64")
#     df["bh_mass"] = pd.to_numeric(df["bh_mass"], errors="coerce")
#     df["halo_catalogue_index"] = pd.to_numeric(df["halo_catalogue_index"], errors="coerce").astype("Int64")

#     for tid in TEST_TIDS:
#         row = df[df["track_id"] == tid]
#         if not row.empty:
#             r = row.iloc[0]
#             history[tid].append(
#                 {
#                     "snapshot": snap,
#                     "z": z,
#                     "bh_mass": float(r["bh_mass"]) if pd.notna(r["bh_mass"]) else np.nan,
#                     "halo_catalogue_index": int(r["halo_catalogue_index"]) if pd.notna(r["halo_catalogue_index"]) else None,
#                 }
#             )

#     good = df["bh_mass"] > 0
#     for _, row in df.loc[good].iterrows():
#         tid = int(row.track_id)
#         if tid in lookup:
#             continue
#         lookup[tid] = (
#             float(row.bh_mass),
#             int(row.halo_catalogue_index),
#             snap
#         )

# corrected = pd.read_csv(LOOKUP_CSV)
# corrected = corrected.set_index("track_id")

# for tid in TEST_TIDS:
#     print("\n" + "=" * 70)
#     print(f"TRACK {tid}")
#     print("-" * 70)

#     hist = pd.DataFrame(history[tid]).sort_values("z", ascending=True).reset_index(drop=True)
#     if hist.empty:
#         print("No rows found in the snapshots.")
#         continue

#     print(hist.to_string(index=False))

#     chosen = lookup.get(tid, (np.nan, None, None))
#     print("\nChosen by in-memory lookup:")
#     print(f"  bh_mass = {chosen[0]}")
#     print(f"  halo_catalogue_index = {chosen[1]}")
#     print(f"  snapshot = {chosen[2]}")

#     if tid in corrected.index:
#         print("\nStored in corrected_bh_mass_lookup.csv:")
#         print(f"  corrected_bh_mass = {corrected.loc[tid, 'corrected_bh_mass']}")
#         if "first_snapshot_with_bh" in corrected.columns:
#             print(f"  first_snapshot_with_bh = {corrected.loc[tid, 'first_snapshot_with_bh']}")