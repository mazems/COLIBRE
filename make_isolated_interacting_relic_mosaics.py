#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Put the HaloCatalogueIndex values you want in each mosaic here.
#
# You can freely add/remove IDs. The number of rows will
# automatically adjust while keeping 3 galaxies per row.

ISOLATED_IDS = [
    18408,
    252431,
    4323,
    129078,
    1867573,
    61127,
    104782,
    304,
    1415,
]

INTERACTING_IDS = [
    23328,
    7997,
    2348093,
    59531,
    141036,
    44523,
]

# Input catalogue containing the galaxy properties used for labels.
# The representative sample CSV is convenient because it already
# contains the selected galaxies.
PROPERTIES_CSV = "representative_relic_sample.csv"

# Folder containing the rendered relic images.
IMAGE_DIR = "images"

# Output filenames.
ISOLATED_OUTPUT = "mosaic_isolated_relics.pdf"
INTERACTING_OUTPUT = "mosaic_interacting_relics.pdf"

# Number of galaxies per row.
NCOLS = 3


# ============================================================
# HELPERS
# ============================================================

def image_path(folder: Path, hid: int, orientation: str) -> Path:
    return (
        folder
        / f"SDSS_id{int(hid)}_snap127_"
          f"SDSS_original_dust_free_{orientation}.png"
    )


def add_image(ax, path: Path):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    if path.exists():
        ax.imshow(mpimg.imread(path))
    else:
        ax.set_facecolor("0.93")
        ax.text(
            0.5,
            0.5,
            "IMAGE\nMISSING",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    for spine in ax.spines.values():
        spine.set_visible(False)


def fval(x, d=2):
    try:
        x = float(x)
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.{d}f}"
    except Exception:
        return "n/a"


def make_mosaic(
    ids,
    properties,
    image_dir,
    outfile,
):
    """
    Make a 3-column mosaic.

    Each galaxy consists of:
        face-on | edge-on
    with the galaxy label directly underneath.
    """

    if len(ids) == 0:
        print(f"No IDs supplied for {outfile}; skipping.")
        return

    ncols = NCOLS
    nrows = math.ceil(len(ids) / ncols)

    # These dimensions reproduce the compact appearance of your
    # current representative mosaic.
    fig = plt.figure(
        figsize=(4.2 * ncols, 3.0 * nrows),
        facecolor="white",
    )

    gs = fig.add_gridspec(
        nrows,
        ncols,
        left=0.015,
        right=0.985,
        top=0.995,
        bottom=0.015,
        wspace=0.035,
        hspace=0.18,
    )

    properties = properties.copy()
    properties["HaloCatalogueIndex"] = pd.to_numeric(
        properties["HaloCatalogueIndex"],
        errors="coerce",
    )

    properties = properties.dropna(
        subset=["HaloCatalogueIndex"]
    )

    properties["HaloCatalogueIndex"] = (
        properties["HaloCatalogueIndex"].astype(np.int64)
    )

    properties = properties.set_index("HaloCatalogueIndex")

    for i, hid in enumerate(ids):

        rr, cc = divmod(i, ncols)

        # Each galaxy is one horizontal pair:
        # face-on | edge-on
        sub = gs[rr, cc].subgridspec(
            1,
            2,
            wspace=0.0,
        )

        ax_face = fig.add_subplot(sub[0, 0])
        ax_edge = fig.add_subplot(sub[0, 1])

        add_image(
            ax_face,
            image_path(image_dir, hid, "face_on"),
        )
        add_image(
            ax_edge,
            image_path(image_dir, hid, "edge_on"),
        )

        # Orientation labels
        ax_face.text(
            0.025,
            0.95,
            "face-on",
            transform=ax_face.transAxes,
            color="white",
            fontsize=8,
            ha="left",
            va="top",
            bbox=dict(
                facecolor="black",
                alpha=0.45,
                edgecolor="none",
                pad=2,
            ),
        )

        ax_edge.text(
            0.025,
            0.95,
            "edge-on",
            transform=ax_edge.transAxes,
            color="white",
            fontsize=8,
            ha="left",
            va="top",
            bbox=dict(
                facecolor="black",
                alpha=0.45,
                edgecolor="none",
                pad=2,
            ),
        )

        # --------------------------------------------------------
        # Properties
        # --------------------------------------------------------
        if hid in properties.index:
            row = properties.loc[hid]

            is_central = bool(row.get("IsCentral", False))
            cs = "C" if is_central else "S"

            text = (
                f"Subhalo {hid}  [{cs}]\n"
                rf"$\log M_\star$ = {fval(row.get('logM'))}"
                f"   DoR = {fval(row.get('DoR'))}"
                rf"   $\log \Sigma_{{1.5}}$ = "
                f"{fval(row.get('compactness'))}"
            )

        else:
            text = f"Subhalo {hid}"

        # Put the text directly underneath the image pair.
        # The figure coordinates are calculated from the grid
        # position so this remains correct for any number of rows.
        x_center = (cc + 0.5) / ncols
        y = 1.0 - (rr + 1) / nrows + 0.005

        fig.text(
            x_center,
            y,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.savefig(
        outfile,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {outfile}")


# ============================================================
# MAIN
# ============================================================

def main():

    properties_path = Path(PROPERTIES_CSV)
    image_dir = Path(IMAGE_DIR)

    if not properties_path.exists():
        raise SystemExit(
            f"Properties CSV not found: {properties_path}"
        )

    if not image_dir.exists():
        raise SystemExit(
            f"Image directory not found: {image_dir}"
        )

    properties = pd.read_csv(
        properties_path,
        low_memory=False,
    )

    required = ["HaloCatalogueIndex", "logM", "DoR", "compactness"]

    missing = [
        c for c in required
        if c not in properties.columns
    ]

    if missing:
        raise SystemExit(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Remove accidental duplicates while preserving the first
    # catalogue entry.
    # --------------------------------------------------------
    properties = properties.drop_duplicates(
        subset=["HaloCatalogueIndex"],
        keep="first",
    )

    print(
        f"Creating isolated mosaic with "
        f"{len(ISOLATED_IDS)} galaxies..."
    )

    make_mosaic(
        ISOLATED_IDS,
        properties,
        image_dir,
        ISOLATED_OUTPUT,
    )

    print(
        f"Creating interacting mosaic with "
        f"{len(INTERACTING_IDS)} galaxies..."
    )

    make_mosaic(
        INTERACTING_IDS,
        properties,
        image_dir,
        INTERACTING_OUTPUT,
    )


if __name__ == "__main__":
    main()