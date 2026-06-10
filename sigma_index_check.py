#!/usr/bin/env python3

import numpy as np
import h5py
import common

# ============================================================
# CONFIG
# ============================================================

model_name = 'L0200N3008/THERMAL_AGN/'
model_dir  = '/mnt/su3-pro/colibre/' + model_name
snap_file  = '0127'

sigma_path = (
    "/mnt/su3-pro/colibre/"
    "L0200N3008/THERMAL_AGN/"
    "SOAP-HBT/extra/halo_properties_0127.hdf5"
)

# ============================================================
# LOAD SOAP IDS
# ============================================================

fields_sgn = {
    'InputHalos': (
        'HaloCatalogueIndex',
    )
}

print("Loading SOAP HaloCatalogueIndex...")

(halo_catalogue_index,) = common.read_group_data_colibre(
    model_dir,
    snap_file,
    fields_sgn
)

halo_catalogue_index = np.asarray(
    halo_catalogue_index,
    dtype=np.int64
).ravel()

print("SOAP catalogue size:", halo_catalogue_index.size)

fields = {
    'ExclusiveSphere/50kpc': ('StellarMass', 'HalfMassRadiusStars')
}

(m30, r50) = common.read_group_data_colibre(model_dir, snap_file, fields)
m30 = np.asarray(m30, dtype=float).ravel()
r50 = np.asarray(r50, dtype=float).ravel()

Mu = 1.988e43 / 1.989e33
Lu = 1.0  # keep as 1.0 if you only need the selection; otherwise use your full conversion

m30 = m30 * Mu
r50 = r50 * Lu * 1e3

mask_positive = (m30 >= 1e9) & (m30 > 0) & (r50 > 0)

# ============================================================
# OPEN SIGMA FILE
# ============================================================

print("\nOpening sigma file:")
print(sigma_path)

with h5py.File(sigma_path, "r") as f:

    print("\n================ DATASETS IN FILE ================\n")

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name:90s} {obj.shape}")

    f.visititems(walk)

    # --------------------------------------------------------
    # CHECK MAIN SIGMA DATASET
    # --------------------------------------------------------

    sigma_ds = (
        "/ExclusiveSphere/"
        "HalfMassRadiusStars/"
        "StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"
    )

    print("\n=================================================")
    print("SIGMA DATASET SHAPE")
    print("=================================================\n")

    ds = f[sigma_ds]

    print("sigma dataset shape:", ds.shape)

    gal_idx = np.where(mask_positive)[0]
    sigma_zz = ds[gal_idx.astype(np.int64), 8]
    selected_ids = halo_catalogue_index[mask_positive]
    print("max selected HaloCatalogueIndex:", selected_ids.max())
    print("sigma row count minus 1:", ds.shape[0] - 1)
    print("N(sigma_zz == 0):", np.count_nonzero(np.isclose(sigma_zz[np.isfinite(sigma_zz)], 0.0)))
    print("Selected galaxies:", mask_positive.sum())

    # --------------------------------------------------------
    # TRY COMMON ID DATASETS
    # --------------------------------------------------------

    candidate_paths = [
        "InputHalos/HaloCatalogueIndex",
        "SOAP/HaloCatalogueIndex",
        "HaloCatalogueIndex",
        "InputHalos/TrackId",
        "TrackId",
    ]

    found_match = False

    for p in candidate_paths:

        if p in f:

            print("\n=================================================")
            print("FOUND POSSIBLE ID DATASET")
            print("=================================================\n")

            print("dataset:", p)

            ids = np.asarray(f[p][:], dtype=np.int64).ravel()

            print("id dataset shape:", ids.shape)

            print("\nFirst 20 IDs in sigma file:")
            print(ids[:20])

            print("\nFirst 20 SOAP HaloCatalogueIndex:")
            print(halo_catalogue_index[:20])

            same_size = (ids.size == halo_catalogue_index.size)

            print("\nSame size as SOAP catalogue?", same_size)

            if same_size:

                exact_match = np.all(
                    ids[:1000] == halo_catalogue_index[:1000]
                )

                print("First 1000 rows identical?", exact_match)

                overlap = np.intersect1d(
                    ids[:10000],
                    halo_catalogue_index[:10000]
                ).size

                print("Overlap among first 10000 IDs:", overlap)

            found_match = True

    if not found_match:

        print("\n=================================================")
        print("NO EXPLICIT ID DATASET FOUND")
        print("=================================================\n")

        print("This probably means the sigma file is row-aligned")
        print("with the SOAP catalogue.")
        print()
        print("In that case:")
        print("-> gal_idx is probably correct")
        print("-> HaloCatalogueIndex indexing is probably WRONG")