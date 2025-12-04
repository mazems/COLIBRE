#!/usr/bin/env python3
"""
make_missing_5chunk.py

Reads missing_files.txt (lines: nstart nend filename)
Splits each [nstart,nend) into subchunks of size chunk_size (default 5).
Writes missing_files_5chunk.txt with lines: nstart nend relicness_nstart_nend.csv
Skips subchunks if the output file already exists and non-empty.
"""
import sys, os

CHUNK = 5
infn = "missing_files.txt"
outfn = "missing_files_5chunk.txt"

if not os.path.exists(infn):
    print("ERROR: missing_files.txt not found", file=sys.stderr)
    sys.exit(2)

lines = [ln.strip() for ln in open(infn).read().strip().splitlines() if ln.strip()]
out_lines = []
skipped = 0
for ln in lines:
    parts = ln.split()
    if len(parts) < 2:
        print("Skipping malformed line:", ln, file=sys.stderr)
        continue
    a = int(parts[0]); b = int(parts[1])
    if a >= b:
        print("Skipping empty interval", a,b, file=sys.stderr)
        continue
    i = a
    while i < b:
        j = min(b, i + CHUNK)
        fn = f"relicness_{i}_{j}.csv"
        if os.path.exists(fn) and os.path.getsize(fn) > 0:
            skipped += 1
        else:
            out_lines.append(f"{i} {j} {fn}")
        i = j

open(outfn, "w").write("\n".join(out_lines) + ("\n" if out_lines else ""))
print(f"Wrote {outfn} with {len(out_lines)} subchunks. Skipped {skipped} already-present files.")
