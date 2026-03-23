#!/usr/bin/env python3
"""
merge_relicness_h5py.py

Merge all relicness_*.csv files in cwd except a small set of excluded chunks.

Default excluded files:
  relicness_270_275.csv
  relicness_405_410.csv
  relicness_430_435.csv

Usage:
  python3 merge_relicness_except.py
  python3 merge_relicness_except.py --out relicness_merged.csv
  python3 merge_relicness_except.py --pattern "relicness_*.csv" --exclude relicness_270_275.csv relicness_405_410.csv
"""

from __future__ import annotations
import argparse
import glob
import os
import shutil
import time
from typing import List

import pandas as pd

DEFAULT_EXCLUDE = [
    "relicness_270_275.csv",
    "relicness_405_410.csv",
    "relicness_430_435.csv",
    "relicness_all_ucmg.csv",
    "relicness_all_ucmg_deduped.csv",
    "relicness_ingredients_fast.csv",
    "relicness_test_6041.csv",
    "relicness_merged.csv",
]

def find_files(pattern: str, exclude: List[str]) -> List[str]:
    files = sorted(glob.glob(pattern))
    files = [f for f in files if os.path.basename(f) not in exclude]
    return files

def main():
    parser = argparse.ArgumentParser(description="Merge relicness CSVs excluding a few chunk files.")
    parser.add_argument("--pattern", default="relicness_extras_*.csv", help="glob pattern to find relicness csv files")
    parser.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE, help="filenames to exclude (basename match)")
    parser.add_argument("--out", default="relicness_merged_extras.csv", help="output merged CSV filename")
    args = parser.parse_args()

    files = find_files(args.pattern, args.exclude)
    if not files:
        print("No files found to merge with pattern:", args.pattern)
        print("Excluded basenames:", args.exclude)
        return

    print(f"Found {len(files)} files to merge (excluding {len(args.exclude)}):")
    for f in files:
        print("  ", f)

    # read & concat
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'subhalo_id' not in df.columns:
                print(f"Warning: file {f} has no 'subhalo_id' column; it will still be concatenated.")
            dfs.append(df)
        except Exception as e:
            print(f"ERROR reading {f}: {e}. Skipping this file.")

    if not dfs:
        print("No readable dataframes found; aborting.")
        return

    merged = pd.concat(dfs, ignore_index=True, sort=False)

    # de-duplicate by subhalo_id if present: keep last occurrence (later files override earlier)
    if 'subhalo_id' in merged.columns:
        merged = merged.drop_duplicates(subset=['subhalo_id'], keep='last')

    # backup existing output if present
    outfn = args.out
    if os.path.exists(outfn):
        bak = f"{outfn}.{int(time.time())}.bak"
        shutil.copy2(outfn, bak)
        print(f"Existing output {outfn} backed up to {bak}")

    merged.to_csv(outfn, index=False)
    print(f"Wrote merged CSV: {outfn}  (rows={merged.shape[0]}, cols={merged.shape[1]})")

if __name__ == "__main__":
    main()