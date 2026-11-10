#!/usr/bin/env python3

import fitz
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PLOTDIR = Path("plots_dor")

files = [
    PLOTDIR / "DoR_vs_compactness_bin_mass_10p00_11p00.pdf",
    PLOTDIR / "DoR_vs_compactness_bin_mass_11p00_12p00.pdf",
    PLOTDIR / "DoR_vs_exsitu_bin_mass_10p00_11p00.pdf",
    PLOTDIR / "DoR_vs_exsitu_bin_mass_11p00_12p00.pdf",
]

#labels = ["a)", "b)", "c)", "d)"]

outfile = "combined_DoR_panels.pdf"

# -------------------------------------------------------
# Read PDFs
# -------------------------------------------------------

images = []

for f in files:
    doc = fitz.open(str(f))
    page = doc[0]

    pix = page.get_pixmap(dpi=300, alpha=False)

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        pix.n
    )

    images.append(img)
    doc.close()

# -------------------------------------------------------
# Plot
# -------------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(11, 8),
    gridspec_kw={
        "wspace": 0.01,
        "hspace": 0.01,
    }
)

for ax, img in zip(axes.ravel(), images): #3, labels):

    ax.imshow(
        img,
        aspect="auto"
    )

    ax.axis("off")

    # # Label at exactly the same position in every panel
    # ax.text(
    #     0.12,
    #     0.92,
    #     lab,
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=11,
    #     zorder=10
    # )

plt.savefig(
    outfile,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print("Saved:", outfile)