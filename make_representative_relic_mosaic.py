#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_FEATURES = [
    "logM", "compactness", "DoR", "MgFe", "logZstar_rel",
    "log_ssfr", "exsitu_frac", "log_sigma",
]


def parse_args():
    p = argparse.ArgumentParser(description="Select a representative relic sample and make a face-on/edge-on mosaic.")
    p.add_argument("--input", default="relic_master_properties.csv")
    p.add_argument("--images-dir", default="images")
    p.add_argument("--output-mosaic", default="representative_relic_mosaic.png")
    p.add_argument("--output-csv", default="representative_relic_sample.csv")
    p.add_argument("--n-total", type=int, default=20)
    p.add_argument("--n-low", type=int, default=18)
    p.add_argument("--n-high", type=int, default=2)
    p.add_argument("--low-mass-min", type=float, default=10.0)
    p.add_argument("--high-mass-min", type=float, default=11.0)
    p.add_argument("--high-mass-max", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def robust_standardize(s):
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(x)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (x - med) / scale


def greedy_farthest(df, features, n, rng):
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    usable = [f for f in features if f in df.columns and pd.to_numeric(df[f], errors="coerce").notna().sum() >= 2]
    if not usable:
        idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
        return df.iloc[np.sort(idx)].copy()
    X = np.column_stack([robust_standardize(df[f]) for f in usable])
    X = np.where(np.isfinite(X), X, 0.0)
    first = int(np.argmin(np.sum(X * X, axis=1)))
    selected = [first]
    remaining = np.ones(len(df), dtype=bool)
    remaining[first] = False
    min_dist = np.linalg.norm(X - X[first], axis=1)
    min_dist[~remaining] = -np.inf
    while len(selected) < min(n, len(df)):
        mx = np.nanmax(min_dist)
        cand = np.flatnonzero(np.isclose(min_dist, mx, rtol=1e-12, atol=1e-12))
        if cand.size == 0:
            cand = np.flatnonzero(remaining)
        pick = int(cand[rng.integers(0, len(cand))])
        selected.append(pick)
        remaining[pick] = False
        d = np.linalg.norm(X - X[pick], axis=1)
        min_dist = np.minimum(min_dist, d)
        min_dist[~remaining] = -np.inf
    return df.iloc[selected].copy()


def choose_balanced(df, n, features, rng):
    df = df.copy()
    df["IsCentral"] = df["IsCentral"].astype(bool)
    parent_frac = float(df["IsCentral"].mean()) if len(df) else 0.5
    nc = min(len(df[df.IsCentral]), max(0, int(np.rint(n * parent_frac))))
    ns = min(len(df[~df.IsCentral]), n - nc)
    if nc + ns < n:
        extra = n - nc - ns
        if len(df[df.IsCentral]) - nc >= extra:
            nc += extra
        elif len(df[~df.IsCentral]) - ns >= extra:
            ns += extra
    c = greedy_farthest(df[df.IsCentral], features, nc, rng)
    s = greedy_farthest(df[~df.IsCentral], features, ns, rng)
    chosen = pd.concat([c, s])
    if len(chosen) < n:
        rem = df.loc[~df.index.isin(chosen.index)]
        chosen = pd.concat([chosen, greedy_farthest(rem, features, n - len(chosen), rng)])
    return chosen


def image_path(folder, hid, orientation):
    return Path(folder) / f"SDSS_id{int(hid)}_snap127_SDSS_original_dust_free_{orientation}.png"


def add_image(ax, path):
    ax.set_xticks([]); ax.set_yticks([]) #; ax.set_aspect("equal")
    if path.exists():
        ax.imshow(mpimg.imread(path))
    else:
        ax.set_facecolor("0.93")
        ax.text(0.5, 0.5, "IMAGE\nMISSING", ha="center", va="center", fontsize=11, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_visible(False)


def fval(x, d=2):
    try:
        x = float(x)
        return "n/a" if not np.isfinite(x) else f"{x:.{d}f}"
    except Exception:
        return "n/a"


def make_mosaic(selected, image_dir, outfile):
    """
    Compact 4 x 5 relic mosaic.

    Each galaxy:
        face-on | edge-on
        Subhalo ID / C-S
        logM / DoR / log Sigma_1.5

    The figure is deliberately shorter than a full A4 page.
    It is intended to be scaled to the desired width in LaTeX.
    """

    ncols = 4
    nrows = 5

    # Compact figure: do NOT use full A4 height here.
    fig = plt.figure(
        figsize=(8.27, 7.0),
        facecolor="white",
    )

    outer = fig.add_gridspec(
        nrows,
        ncols,
        left=0.015,
        right=0.985,
        top=0.99,
        bottom=0.01,
        wspace=0.015,
        hspace=0.015,
    )

    for i, (_, row) in enumerate(selected.iterrows()):

        rr, cc = divmod(i, ncols)

        # Image pair + compact text strip
        cell = outer[rr, cc].subgridspec(
            2,
            1,
            height_ratios=[1.0, 0.17],
            hspace=0.01,
        )

        image_row = cell[0].subgridspec(
            1,
            2,
            wspace=0.008,
        )

        axf = fig.add_subplot(image_row[0, 0])
        axe = fig.add_subplot(image_row[0, 1])
        axt = fig.add_subplot(cell[1, 0])

        hid = int(row["HaloCatalogueIndex"])

        # -------------------------
        # images
        # -------------------------
        add_image(
            axf,
            image_path(image_dir, hid, "face_on"),
        )

        add_image(
            axe,
            image_path(image_dir, hid, "edge_on"),
        )

        for ax, lab in (
            (axf, "face-on"),
            (axe, "edge-on"),
        ):
            ax.text(
                0.035,
                0.94,
                lab,
                transform=ax.transAxes,
                color="white",
                fontsize=6.5,
                ha="left",
                va="top",
                bbox=dict(
                    facecolor="black",
                    alpha=0.55,
                    edgecolor="none",
                    pad=1.2,
                ),
            )

        # -------------------------
        # labels
        # -------------------------
        axt.axis("off")

        cs = "C" if bool(row["IsCentral"]) else "S"

        line1 = f"Subhalo {hid}   [{cs}]"

        line2 = (
            rf"$\log M_\star={fval(row['logM'])}$   "
            rf"$\mathrm{{DoR}}={fval(row['DoR'])}$   "
            rf"$\log\Sigma_{{1.5}}={fval(row['compactness'])}$"
        )

        axt.text(
            0.5,
            0.68,
            line1,
            ha="center",
            va="center",
            fontsize=6.8,
            color="black",
        )

        axt.text(
            0.5,
            0.18,
            line2,
            ha="center",
            va="center",
            fontsize=6.0,
            color="black",
        )

    fig.savefig(
        outfile,
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close(fig)


def main():
    a = parse_args(); rng = np.random.default_rng(a.seed)
    inp = Path(a.input); imgdir = Path(a.images_dir)
    if not inp.exists(): raise SystemExit(f"Input CSV not found: {inp}")
    if not imgdir.exists(): raise SystemExit(f"Image directory not found: {imgdir}")
    df = pd.read_csv(inp, low_memory=False)
    for c in ["HaloCatalogueIndex", "logM", "IsCentral"]:
        if c not in df.columns: raise SystemExit(f"Required column missing: {c}")
    df["HaloCatalogueIndex"] = pd.to_numeric(df["HaloCatalogueIndex"], errors="coerce")
    df["logM"] = pd.to_numeric(df["logM"], errors="coerce")
    df = df[np.isfinite(df.HaloCatalogueIndex) & np.isfinite(df.logM)].copy()
    df["HaloCatalogueIndex"] = df.HaloCatalogueIndex.astype(np.int64)
    df["IsCentral"] = df["IsCentral"].astype(bool)
    parent = df[(df.logM >= a.low_mass_min) & (df.logM < a.high_mass_max)]
    low = parent[(parent.logM >= a.low_mass_min) & (parent.logM < a.high_mass_min)]
    high = parent[(parent.logM >= a.high_mass_min) & (parent.logM < a.high_mass_max)]
    if len(low) < a.n_low: raise SystemExit(f"Only {len(low)} low-mass relics available; need {a.n_low}.")
    if len(high) < a.n_high: raise SystemExit(f"Only {len(high)} high-mass relics available; need {a.n_high}.")
    if a.n_low + a.n_high != a.n_total: raise SystemExit("--n-low + --n-high must equal --n-total")
    features = [f for f in DEFAULT_FEATURES if f in parent.columns]
    print("Using diversity features:", ", ".join(features))
    selected = pd.concat([
        choose_balanced(low, a.n_low, features, rng),
        choose_balanced(high, a.n_high, features, rng),
    ], ignore_index=True).sort_values(["logM", "HaloCatalogueIndex"]).reset_index(drop=True)
    selected["mass_bin"] = np.where(selected.logM < 11, "10^10-10^11 Msun", "10^11-10^12 Msun")
    selected["central_satellite"] = np.where(selected.IsCentral, "Central", "Satellite")
    selected["face_on_available"] = [image_path(imgdir, h, "face_on").exists() for h in selected.HaloCatalogueIndex]
    selected["edge_on_available"] = [image_path(imgdir, h, "edge_on").exists() for h in selected.HaloCatalogueIndex]
    cols = ["HaloCatalogueIndex", "TrackId_sample", "TrackId_SOAP", "IsCentral", "central_satellite", "mass_bin",
            "logM", "logR", "r50_kpc", "compactness", "DoR", "age_mass_gyr", "age_lum_gyr", "MgFe",
            "logZstar_rel", "logZstar_loc", "log_ssfr", "exsitu_frac", "sigma", "log_sigma",
            "bh_mass_corr", "log_bh_ratio_corr", "face_on_available", "edge_on_available"]
    cols = [c for c in cols if c in selected.columns] + [c for c in selected.columns if c not in cols]
    selected[cols].to_csv(a.output_csv, index=False)
    print(f"Saved selection table: {a.output_csv}")
    print("Selected counts by mass bin:")
    print(selected.mass_bin.value_counts().sort_index().to_string())
    print(f"Parent central fraction: {parent.IsCentral.mean():.3f}")
    print(f"Selected central fraction: {selected.IsCentral.mean():.3f}")
    print(f"Missing face-on: {(~selected.face_on_available).sum()}")
    print(f"Missing edge-on: {(~selected.edge_on_available).sum()}")
    make_mosaic(selected, imgdir, a.output_mosaic)
    print(f"Saved mosaic: {a.output_mosaic}")


if __name__ == "__main__":
    main()