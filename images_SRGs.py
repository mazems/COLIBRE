#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
from astropy.io import fits
# from swiftsimio import SWIFTDataset
# from swiftgalaxy import SWIFTGalaxy, SOAP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "PARTRIDGE"))
import partridge
from partridge import make_galaxy_image


def normalize_and_save(arr, fname):
    arr = np.asarray(arr)

    # formats: (H, W, 3) or (3, H, W)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(f"Unexpected RGB array shape: {arr.shape}")

    if arr.dtype.kind == "f":
        cols = np.clip(arr, 0.0, 1.0)
    else:
        cols = np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)

    img = Image.fromarray((cols * 255).astype(np.uint8))
    img.save(fname)


def main():
    parser = argparse.ArgumentParser(
        description="Render galaxy images from a sample CSV."
    )
    parser.add_argument("--csv", default="z0_sags_trackids.csv")
    parser.add_argument("--snap", type=int, required=True)

    # defaults set to what you want for the appendix
    parser.add_argument(
        "--orientation",
        default="edge_on",
        choices=["face_on", "edge_on", "random_x", "random_y", "random_z"],
        help="Galaxy orientation.",
    )

    parser.add_argument(
        "--model-name",
        default="L0200N3008/THERMAL_AGN/",
        help="Simulation subdirectory inside /mnt/su3-pro/colibre/.",
    )
    parser.add_argument("--pixel-size", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--output-dir", default="images_sags", help="Directory in which to save the rendered images.")
    parser.add_argument("--start", type=int, default=0,
                    help="First index in the relic list to render.")
    parser.add_argument("--end", type=int, default=None,
                    help="One-past-last index in the relic list.")
    # parser.add_argument("--index", type=int)

    # keep the option easy to switch later
    parser.add_argument(
        "--dust-free",
        action="store_true",
        default=True,
        help="Save dust-free output if available (default: on).",
    )
    parser.add_argument(
        "--no-dust-free",
        action="store_false",
        dest="dust_free",
        help="Do not request the dust-free output.",
    )

    parser.add_argument(
        "--parallelize",
        action="store_true",
        help="Parallelize the rendering.",
    )

    args = parser.parse_args()

    original_cwd = os.getcwd()
    outdir = os.path.join(original_cwd, args.output_dir)
    os.makedirs(outdir, exist_ok=True)

    model_dir = "/mnt/su3-pro/colibre/" + args.model_name

    # make sure PARTRIDGE finds its data files
    partridge_dir = os.path.dirname(partridge.__file__)
    print("Using PARTRIDGE at:", partridge_dir)
    os.chdir(partridge_dir)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, args.csv)
    df = pd.read_csv(csv_path)

    possible_id_cols = ["track_id", "HaloCatalogueIndex", "halo_track_id", "id"]
    id_col = None
    for c in possible_id_cols:
        if c in df.columns:
            id_col = c
            break

    if id_col is None:
        raise RuntimeError(f"No ID column found. Available columns: {list(df.columns)}")

    ids = df[id_col].astype(int).tolist()
    print(f"Found {len(ids)} relic IDs in {args.csv} using column '{id_col}'")
    # for individual slurm jobs
    # if args.index is not None:
    #     if args.index < 0 or args.index >= len(ids):
    #         raise IndexError(f"--index {args.index} is out of range for {len(ids)} IDs")
    #     ids = [ids[args.index]]
    # for slurm jobs in chunks
    if args.end is None or args.end > len(ids):
        args.end = len(ids)

    if args.start < 0 or args.start >= len(ids):
        raise IndexError(f"--start {args.start} is out of range for {len(ids)} IDs")

    ids = ids[args.start:args.end]
    print(f"Rendering IDs in range [{args.start}, {args.end}) -> {len(ids)} galaxies")

    n_neigh = 64 if args.orientation == "edge_on" else 512

    for i, id_target in enumerate(ids, start=1):
        try:
            # Expected output filename (dust-free by default)
            expected_file = os.path.join(
                outdir,
                f"SDSS_id{id_target}_snap{args.snap}_SDSS_original_dust_free_{args.orientation}.png",
            )

            # Skip if this galaxy has already been rendered
            if os.path.exists(expected_file):
                print(f"[{i}/{len(ids)}] Halo {id_target} already rendered, skipping.")
                continue

            print(f"[{i}/{len(ids)}] Rendering halo {id_target}")

            image_RGB = make_galaxy_image(
                simulation_path=model_dir,
                snapshot_number=args.snap,
                halo_track_id=id_target,
                image_size=args.image_size,
                pixel_size=args.pixel_size,
                bound_only=False,
                rotation_modes=[args.orientation],
                image_modes=["SDSS_original"],
                redshift=0.0,
                parallelize=args.parallelize,
                N_ngb_target=n_neigh,
                min_brightness=1,
                dust_mode="full_scattering",
                h_min=0.2,
                output_no_dust=args.dust_free,
                output_raw_brightness=False,
            )

            image_mode = list(image_RGB.keys())[0]
            halo_keys = list(image_RGB[image_mode].keys())
            if len(halo_keys) == 0:
                print(f"  -> no halo entry returned for {id_target}, skipping")
                continue

            halo_id = halo_keys[0]
            rgb_entry = image_RGB[image_mode][halo_id].get("RGB", None)
            if rgb_entry is None:
                print(f"  -> no RGB entry for {id_target}, skipping")
                continue

            if isinstance(rgb_entry, dict):
                for tag, arr in rgb_entry.items():
                    if arr is None:
                        continue

                    # Keep only the dust-free image if requested
                    if args.dust_free and tag not in ["dust_free", "nodust", "no_dust"]:
                        continue

                    fname = os.path.join(
                        outdir,
                        f"SDSS_id{id_target}_snap{args.snap}_{image_mode}_{tag}_{args.orientation}.png",
                    )
                    normalize_and_save(arr, fname)

            else:
                fname = os.path.join(
                    outdir,
                    f"SDSS_id{id_target}_snap{args.snap}_{image_mode}_{args.orientation}.png",
                )
                normalize_and_save(rgb_entry, fname)

        except Exception as e:
            print(f"  -> failed for halo {id_target}: {e}")

    os.chdir(original_cwd)
    print("Done.")


if __name__ == "__main__":
    main()