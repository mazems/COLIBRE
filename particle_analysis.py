import math
import matplotlib.pyplot as plt
import numpy as np
import os
import common
import pandas as pd
import swiftsimio as sw
from swiftgalaxy import SWIFTGalaxy, SOAP
import h5py   # CHANGED: add for reading SOAP catalogue to map indices
import gc     # CHANGED: for explicit garbage collection to reduce memory footprint
from astropy.cosmology import Planck15
import astropy.units as u

# ---------- user configuration (keep as you had) ----------
family_method = 'radial_profiles'
method = 'circular_apertures_face_on_map'

model_name = 'L0200N3008/THERMAL_AGN/'
# CHANGED: use consistent model_dir base used earlier in your workflow
model_dir = '/mnt/su3-pro/colibre/' + model_name

# snapshot configuration (keep your existing setting)
snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0026', '0018']
zstarget = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# Select only the z = 0 snapshot (as before)
snap_file = snap_files[0]   # '0127' corresponds to z=0.0
ztarget = zstarget[0]
comov_to_physical_length = 1.0 / (1.0 + ztarget)

# ---------- file names used by common/SOAP ----------
soap_catalogue_file = os.path.join(model_dir, "SOAP-HBT", f"halo_properties_{snap_file}.hdf5")
virtual_snapshot_file = os.path.join(model_dir, "SOAP-HBT", f"colibre_with_SOAP_membership_{snap_file}.hdf5")

print("Using SOAP catalogue:", soap_catalogue_file)
print("Using virtual snapshot:", virtual_snapshot_file)
if not os.path.exists(soap_catalogue_file):
    raise SystemExit("SOAP catalogue not found at: " + soap_catalogue_file)
if not os.path.exists(virtual_snapshot_file):
    raise SystemExit("Virtual snapshot not found at: " + virtual_snapshot_file)

# ---------- read your previously saved galaxy properties (optional) ----------
# If you want to verify sgn_in etc, you can re-read ProcessedData files here.
# (I keep this minimal because you already produce gal_props earlier.)
# CHANGED: read the ucmg CSV you mentioned (should contain a column 'subhalo_id')
ucmg_csv = "ucmg_ids.csv"   # change if your file name differs
if not os.path.exists(ucmg_csv):
    raise SystemExit(f"ucmg file not found: {ucmg_csv}")

ucmg_df = pd.read_csv(ucmg_csv)
if 'subhalo_id' not in ucmg_df.columns:
    # try first column fallback
    print("Warning: 'subhalo_id' column not found in CSV, using first column instead")
    subhalo_ids = ucmg_df.iloc[:, 0].to_numpy()
else:
    subhalo_ids = ucmg_df['subhalo_id'].to_numpy()

subhalo_ids = subhalo_ids.astype(np.int64)
print(f"Read {len(subhalo_ids)} IDs from {ucmg_csv} (unique: {np.unique(subhalo_ids).size})")

# ---------- Map subhalo IDs -> SOAP row indices ----------
# There are two possibilities:
# 1) Your CSV already contains SOAP row indices (rare).
# 2) Your CSV contains some halo id (HaloCatalogueIndex / subhalo id) that is a value stored
#    inside the SOAP file in dataset InputHalos/HaloCatalogueIndex (or some other column).
# We'll load that column from the SOAP file and match.

with h5py.File(soap_catalogue_file, 'r') as f:
    # CHANGED: read the common candidate column name used earlier: 'InputHalos/HaloCatalogueIndex'
    # If your SOAP file uses a different dataset name for the id, change this accordingly.
    try:
        soap_halo_index_array = f['InputHalos/HaloCatalogueIndex'][()]
    except KeyError:
        # Try alternate names, print available groups for debugging
        print("ERROR: 'InputHalos/HaloCatalogueIndex' not found in SOAP file. Available keys:")
        print(list(f.keys()))
        raise

# soap_halo_index_array is an array of the id stored per SOAP row.
# We need the row indices where soap_halo_index_array == subhalo_id.

# Build dict for mapping value -> row index (fast)
# CHANGED: if the soap array is large, use a dict for O(1) lookup
value_to_row = {int(val): int(i) for i, val in enumerate(soap_halo_index_array)}

# Now map requested subhalo_ids to soap row indices (if present)
soap_indices = []
not_found = []
for sid in subhalo_ids:
    if int(sid) in value_to_row:
        soap_indices.append(value_to_row[int(sid)])
    else:
        not_found.append(sid)

soap_indices = np.array(soap_indices, dtype=int)
print(f"Mapped {len(soap_indices)} candidates to SOAP row indices, {len(not_found)} not found.")

# Save the mapping for future reuse
map_df = pd.DataFrame({
    "subhalo_id": np.setdiff1d(subhalo_ids, not_found),
    "soap_row_index": soap_indices
})
map_outfn = "ucmg_soap_indices.csv"
map_df.to_csv(map_outfn, index=False)
print("Saved mapping to:", map_outfn)
if not_found:
    print("Some requested IDs were not found in SOAP HaloCatalogueIndex; examples:", not_found[:10])

# ---------- Decide which range to run ----------
# nstart = 0
# nend = len(soap_indices)   # process all by default
# Test of subset
nstart = 0
nend = min(10, len(soap_indices))   

print(f"Processing indices from {nstart} to {nend} (total {nend-nstart})")

# ---------- Loop over mapped SOAP indices and use SWIFTGalaxy ----------
for ii, soap_row_idx in enumerate(soap_indices[nstart:nend], start=nstart):
    print(f"\n=== [{ii}] Processing SOAP row index {soap_row_idx} ===")
    try:
        sg = SWIFTGalaxy(
            virtual_snapshot_file,
            SOAP(soap_catalogue_file, soap_index=int(soap_row_idx)),
        )
    except Exception as e:
        print("Failed to create SWIFTGalaxy for soap_index", soap_row_idx, "->", e)
        continue


 #   Now you can access particle arrays of this galaxy via sg.* attributes:
 #   stars example:
    try:
        coord_in_p4 = sg.stars.coordinates      # shape (Nstar, 3)
        vT4_in = sg.stars.velocities            # velocities
        m_part4 = sg.stars.masses               # masses
        formation_times = getattr(sg.stars, "formation_time", None)
    except Exception as e:
        print("Failed to access star particle arrays:", e)
        # Clean up object and proceed
        del sg
        gc.collect()
        continue

    # Example: compute and print simple properties
    nstars = coord_in_p4.shape[0]
    total_stellar_mass = m_part4.sum()
    print(f" nstars = {nstars:,}, total_stellar_mass (code units) = {total_stellar_mass:.6e}")

    # --- Particle-level analysis ---
    # Try to pull birth scale factors (a_birth) and initial masses
    try:
        # Most Swift versions expose these names:
        a_birth = getattr(sg.stars, "birth_scale_factors", None)
        if a_birth is None:
            a_birth = getattr(sg.stars, "BirthScaleFactors", None)

        m_birth = getattr(sg.stars, "initial_masses", None)
        if m_birth is None:
            m_birth = getattr(sg.stars, "InitialMasses", None)

        if a_birth is None or m_birth is None:
            print("   SFH warning: birth times or initial masses not found for this galaxy.")
            raise ValueError
    except Exception:
        # If SWIFTGalaxy doesn’t expose attributes, skip SFH for this galaxy
        del sg
        gc.collect()
        continue

    # Convert to numpy
    a_birth = np.array(a_birth, dtype=float)
    m_birth = np.array(m_birth, dtype=float)

    # Convert scale factor → redshift → cosmic time (Gyr)
    # z = 1/a - 1  (protect against a=0 or invalid)
    with np.errstate(divide='ignore', invalid='ignore'):
        z_birth = (1.0 / a_birth) - 1.0

    # Convert to cosmic time of formation (Gyr)
    t_form_gyr = np.array([
        Planck15.age(z).to(u.Gyr).value if np.isfinite(z) and z >= 0 else np.nan
        for z in z_birth
    ])

    # Compute lookback time (Gyr)
    t_now = Planck15.age(0).to(u.Gyr).value
    lookback = t_now - t_form_gyr

    # Remove NaNs (if any)
    mask_valid = np.isfinite(lookback) & np.isfinite(m_birth)
    lookback = lookback[mask_valid]
    m_birth = m_birth[mask_valid]

    if lookback.size == 0:
        print("   SFH warning: no valid star particles for this halo.")
        del sg
        gc.collect()
        continue

    # Sort particles by formation time
    order = np.argsort(lookback)
    lb_sorted = lookback[order]
    m_sorted = m_birth[order]

    # Cumulative stellar mass fraction curve
    cum_mass = np.cumsum(m_sorted)
    cum_frac = cum_mass / cum_mass[-1]

    # --- Plot SFH for this galaxy ---
    plt.figure(figsize=(6,4))
    plt.plot(lb_sorted, cum_frac, lw=1.5)
    plt.gca().invert_xaxis()   # optional: oldest on left
    plt.xlabel("Lookback time [Gyr]")
    plt.ylabel("Cumulative stellar mass fraction")
    plt.title(f"SFH SOAP index {soap_row_idx}")
    plt.grid(True)

    # Save per-galaxy SFH plot
    outname = f"sfh_SOAP{soap_row_idx}.png"
    plt.savefig(outname, dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved SFH:", outname)

    # Clean up & free memory before next iteration
    del sg
    gc.collect()

print("\nDone processing all requested SOAP indices.")