#!/usr/bin/env python3

from pathlib import Path
import fitz



PLOT_DIR = Path("plots")

INPUT_FILES = [
    PLOT_DIR / "mass_size_z0.0_age(lum)_loess.pdf",
    PLOT_DIR / "mass_size_z0.0_metallicity_loess.pdf",
    PLOT_DIR / "mass_size_z0.0_fullMgFe_loess.pdf",
    PLOT_DIR / "mass_size_z0.0_ssfr_loess.pdf",
    PLOT_DIR / "mass_size_z0.0_sigma_loess.pdf",
    PLOT_DIR / "mass_size_z0.0_exsitu_loess.pdf",
]

OUTPUT_FILE = PLOT_DIR / "mass_size_mosaic_3x2.pdf"

PANEL_LABELS = ["a)", "b)", "c)", "d)", "e)", "f)"]

NROWS = 3
NCOLS = 2

PANEL_WIDTH = 410
PANEL_HEIGHT = 330

LEFT_MARGIN = 25
RIGHT_MARGIN = 25
TOP_MARGIN = 20
BOTTOM_MARGIN = 45

HORIZONTAL_GAP = 10
VERTICAL_GAP = 3


# Crop amounts:
# left, top, right, bottom
#
# Increase the bottom crop for the upper panels to remove their
# redundant x-axis tick labels and x-axis titles.
CROPS = [
    (10, 10, 10, 10),
    (10, 10, 10, 10),
    (10, 5,  10, 42),
    (10, 5,  10, 42),
    (10, 5,  10, 5),
    (10, 5,  10, 5),
]


page_width = (
    LEFT_MARGIN
    + NCOLS * PANEL_WIDTH
    + (NCOLS - 1) * HORIZONTAL_GAP
    + RIGHT_MARGIN
)

page_height = (
    TOP_MARGIN
    + NROWS * PANEL_HEIGHT
    + (NROWS - 1) * VERTICAL_GAP
    + BOTTOM_MARGIN
)


output_document = fitz.open()

output_page = output_document.new_page(
    width=page_width,
    height=page_height,
)


for index, input_file in enumerate(INPUT_FILES):

    if not input_file.exists():
        raise FileNotFoundError(input_file)

    row = index // NCOLS
    col = index % NCOLS

    x0 = LEFT_MARGIN + col * (PANEL_WIDTH + HORIZONTAL_GAP)
    y0 = TOP_MARGIN + row * (PANEL_HEIGHT + VERTICAL_GAP)

    destination = fitz.Rect(
        x0,
        y0,
        x0 + PANEL_WIDTH,
        y0 + PANEL_HEIGHT,
    )

    with fitz.open(input_file) as source_document:

        source_page = source_document[0]
        source_rect = source_page.rect

        crop_left, crop_top, crop_right, crop_bottom = CROPS[index]

        source_crop = fitz.Rect(
            source_rect.x0 + crop_left,
            source_rect.y0 + crop_top,
            source_rect.x1 - crop_right,
            source_rect.y1 - crop_bottom,
        )

        output_page.show_pdf_page(
            destination,
            source_document,
            pno=0,
            clip=source_crop,
            keep_proportion=True,
        )

    # Optional panel label
    output_page.insert_text(
        point=(x0 + 8, y0 + 18),
        text=PANEL_LABELS[index],
        fontsize=12,
        fontname="helv",
    )


# # Shared x-axis label
# output_page.insert_textbox(
#     fitz.Rect(
#         LEFT_MARGIN,
#         page_height - 32,
#         page_width - RIGHT_MARGIN,
#         page_height - 8,
#     ),
#     r"lg(Stellar Mass / M_sun)",
#     fontsize=12,
#     fontname="helv",
#     align=fitz.TEXT_ALIGN_CENTER,
# )


output_document.save(
    OUTPUT_FILE,
    garbage=4,
    deflate=True,
)

output_document.close()

print(f"Saved: {OUTPUT_FILE}")