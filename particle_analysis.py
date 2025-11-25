import math
import matplotlib.pyplot as plt
import numpy as np
import os
import common
import pandas as pd
import swiftsimio as sw
from swiftgalaxy import SWIFTGalaxy, SOAP
import h5py 
import gc     # for explicit garbage collection to reduce memory footprint
from astropy.cosmology import Planck15
import astropy.units as u
from scipy.stats import binned_statistic_2d
import matplotlib as mpl

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


    # ---- FULL maps + SFH block ----
    
    out_map_dir = "maps_output"
    os.makedirs(out_map_dir, exist_ok=True)
    os.makedirs(os.path.join(out_map_dir, "log_age"), exist_ok=True)
    os.makedirs(os.path.join(out_map_dir, "metallicity"), exist_ok=True)
    os.makedirs(os.path.join(out_map_dir, "sfh"), exist_ok=True)

    nbins_map = 128     # map resolution (tweak if needed)
    mb_frac   = 95      # keep inner 95% most-bound particles (TNG-style)

    # --- helper to get a 1D numpy mass array (do not force a unit conversion) ---
    def _masses_to_1d(arr):
        # try common .to('Msun') usage
        try:
            if hasattr(arr, "to") and callable(arr.to):
                return np.array(arr.to("Msun").value, dtype=float)
        except Exception:
            pass
        # fallback: numpy conversion (may be in code units or 1e10 units depending on your data)
        try:
            a = np.array(arr, dtype=float)
            # ensure at least 1D
            return np.atleast_1d(a)
        except Exception:
            return np.array([], dtype=float)

    # --- read arrays safely from sg.stars ---
    try:
        coords = np.array(sg.stars.coordinates, dtype=float)    # shape (N,3)
        vels   = np.array(sg.stars.velocities, dtype=float)     # shape (N,3)
    except Exception as e:
        print("Failed to read coordinates/velocities:", e)
        del sg; gc.collect()
        continue

    # try several mass attribute names (masses, Masses, InitialMasses)
    m_raw = getattr(sg.stars, "masses", None)
    if m_raw is None:
        m_raw = getattr(sg.stars, "Masses", None)
    if m_raw is None:
        m_raw = getattr(sg.stars, "InitialMasses", None)

    masses = _masses_to_1d(m_raw)

    # ensure 1-D and dtype float
    masses = np.atleast_1d(masses).astype(float, copy=False)

    # If masses length mismatches coords length, try to reconcile
    n_coords = coords.shape[0]
    if masses.size != n_coords:
        if masses.size == 1 and n_coords > 1:
            masses = np.full(n_coords, masses.item(), dtype=float)
            print(f"Warning: broadcast single mass to {n_coords} particles.")
        elif masses.size < n_coords:
            # pad with zeros (will be harmless for maps, but warn)
            pad = n_coords - masses.size
            masses = np.concatenate([masses, np.zeros(pad, dtype=float)])
            print(f"Warning: masses shorter than coords; padded {pad} zeros.")
        else:
            masses = masses[:n_coords]
            print(f"Warning: masses longer than coords; truncated to {n_coords} entries.")

    # --- formation times to cosmic time [Gyr] (try BirthScaleFactors then Ages) ---
    t_form_gyr = None
    a_birth = getattr(sg.stars, "BirthScaleFactors", None) or getattr(sg.stars, "birth_scale_factors", None)
    if a_birth is not None:
        a_birth = np.array(a_birth, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_birth = (1.0 / a_birth) - 1.0
        t_form_gyr = np.array([Planck15.age(z).to(u.Gyr).value if np.isfinite(z) and z >= 0 else np.nan for z in z_birth])
    else:
        ages_direct = getattr(sg.stars, "Ages", None) or getattr(sg.stars, "ages", None)
        if ages_direct is not None:
            ages_direct = np.array(ages_direct, dtype=float)
            t_now = Planck15.age(0).to(u.Gyr).value
            t_form_gyr = t_now - ages_direct

    # --- metallicity (try several names) ---
    metallicity = None
    for name in ("GFM_Metallicity", "metallicity", "MetalMassFractions", "ElementMassFractions"):
        val = getattr(sg.stars, name, None)
        if val is not None:
            arr = np.array(val, dtype=float)
            # if element-wise array, sum to approximate total metallicity per particle
            if arr.ndim == 2:
                arr = arr.sum(axis=1)
            metallicity = arr
            break

    # --- compute log_age if we have formation times ---
    log_age = None
    if t_form_gyr is not None:
        t_now = Planck15.age(0).to(u.Gyr).value
        stellar_age = t_now - np.array(t_form_gyr, dtype=float)
        stellar_age = np.clip(stellar_age, 1e-6, None)
        log_age = np.log10(stellar_age)

    # quick sanity
    print(f" nstars = {n_coords:,}, total_stellar_mass (sum of masses) = {masses.sum():.6e}")

    if (log_age is None or log_age.size==0) and (metallicity is None or metallicity.size==0):
        print("   WARNING: no formation times nor metallicity — skipping maps for this halo.")
        del sg; gc.collect()
        continue

    # --- MOST-BOUND mask (95% inner fraction) ---
    radii = np.linalg.norm(coords, axis=1)
    if radii.size == 0:
        print("   No particles present -> skip")
        del sg; gc.collect()
        continue

    keep_mb = radii <= np.percentile(radii, mb_frac)
    coords = coords[keep_mb]
    masses = masses[keep_mb]
    vels = vels[keep_mb]
    if log_age is not None:
        log_age = log_age[keep_mb]
    if t_form_gyr is not None:
        t_form_gyr = np.array(t_form_gyr, dtype=float)[keep_mb]
    if metallicity is not None:
        metallicity = metallicity[keep_mb]

    if coords.shape[0] == 0:
        print("   No valid star particles left after most-bound masking.")
        del sg; gc.collect()
        continue

    # --- build map grid coordinates (use full extent from particles) ---
    x = coords[:,0]; y = coords[:,1]
    extent = [x.min(), x.max(), y.min(), y.max()]

    # --- LOG_AGE map: mass-weighted mean log_age per pixel (masked cells => white) ---
    if log_age is not None and log_age.size > 0:

        # Numerator and denominator
        H_age_num, xedges, yedges, _ = binned_statistic_2d(
            x, y, log_age * masses, statistic="sum", bins=nbins_map
        )
        H_mass, _, _, _ = binned_statistic_2d(
            x, y, masses, statistic="sum", bins=[xedges, yedges]
        )

        # ⭐ Count map (robust detection of empty bins)
        H_count, _, _, _ = binned_statistic_2d(
            x, y, None, statistic="count", bins=[xedges, yedges]
        )

        # Mass-weighted mean
        with np.errstate(divide="ignore", invalid="ignore"):
            H_age = H_age_num / (H_mass + 1e-20)

        # ⭐ Count-based masking:
        # bins with *zero particles* must be white → set to NaN
        H_age[H_count == 0] = np.nan

        # Mask invalid pixels -> white background
        H_age_masked = np.ma.masked_invalid(H_age)

        # Color map with white for masked pixels
        cmap = mpl.cm.get_cmap("viridis").copy()
        cmap.set_bad("white")

        # Contrast limits from real values only
        finite = H_age_masked.compressed()
        if finite.size > 0:
            vmin = np.percentile(finite, 5)
            vmax = np.percentile(finite, 95)
        else:
            vmin = vmax = None

        # Render map
        plt.figure(figsize=(6,5))
        im = plt.imshow(
            H_age_masked.T, origin="lower", extent=extent,
            cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal"
        )
        cbar = plt.colorbar(im)
        cbar.set_label("log Age [Gyr]")

        plt.title(f"Log Age – {soap_row_idx}")
        plt.xlabel("x [kpc]")
        plt.ylabel("y [kpc]")

        out_age = os.path.join(out_map_dir, "log_age", f"log_age_SOAP{soap_row_idx}.png")
        plt.savefig(out_age, dpi=150, bbox_inches="tight")
        plt.close()
        print("   Saved log_age map:", out_age)

    else:
        print("   WARNING: formation times not available; skipping log_age map.")

    # --- METALLICITY map: mass-weighted metallicity per pixel ---
    if metallicity is not None and metallicity.size > 0:
        H_Z_num, xedges, yedges, _ = binned_statistic_2d(x, y, metallicity * masses, statistic="sum", bins=nbins_map)
        H_mass_forZ, _, _, _ = binned_statistic_2d(x, y, masses, statistic="sum", bins=[xedges, yedges])
        with np.errstate(divide="ignore", invalid="ignore"):
            H_Z = H_Z_num / (H_mass_forZ + 1e-20)

        H_Z_masked = np.ma.masked_invalid(H_Z)
        cmap_z = mpl.cm.get_cmap("plasma").copy()
        cmap_z.set_bad("white")

        finite_z = H_Z_masked.compressed() if np.ma.is_masked(H_Z_masked) else H_Z_masked[np.isfinite(H_Z_masked)]
        if finite_z.size>0:
            vmin_z = np.percentile(finite_z, 5)
            vmax_z = np.percentile(finite_z, 95)
        else:
            vmin_z = vmax_z = None

        plt.figure(figsize=(6,5))
        imz = plt.imshow(H_Z_masked.T, origin="lower", extent=extent, cmap=cmap_z, vmin=vmin_z, vmax=vmax_z, aspect="equal")
        cbarz = plt.colorbar(imz)
        cbarz.set_label("Metallicity")
        plt.title(f"Metallicity – {soap_row_idx}")
        plt.xlabel("x [kpc]"); plt.ylabel("y [kpc]")
        out_z = os.path.join(out_map_dir, "metallicity", f"metal_SOAP{soap_row_idx}.png")
        plt.savefig(out_z, dpi=150, bbox_inches="tight")
        plt.close()
        print("   Saved metallicity map:", out_z)
    else:
        print("   WARNING: metallicity not found; skipping metallicity map.")

    # --- SFH: cumulative stellar mass fraction vs cosmic formation time (Gyr) ---
    if t_form_gyr is not None and t_form_gyr.size>0 and masses.size>0:
        order = np.argsort(t_form_gyr)
        t_sorted = t_form_gyr[order]
        m_sorted = masses[order]
        cum_mass = np.cumsum(m_sorted)
        if cum_mass.size==0 or cum_mass[-1]==0:
            cum_frac = np.zeros_like(cum_mass)
        else:
            cum_frac = cum_mass / cum_mass[-1]

        plt.figure(figsize=(6,4))
        plt.plot(t_sorted, cum_frac, lw=1.2)
        plt.xlabel("Cosmic time [Gyr]")
        plt.ylabel("Cumulative stellar mass fraction")
        plt.title(f"SFH SOAP index {soap_row_idx}")
        plt.grid(True)
        out_sfh = os.path.join(out_map_dir, "sfh", f"sfh_SOAP{soap_row_idx}.png")
        plt.savefig(out_sfh, dpi=150, bbox_inches="tight")
        plt.close()
        print("   Saved SFH:", out_sfh)
    else:
        print("   WARNING: SFH skipped (no formation times or masses).")

    # cleanup
    del sg
    gc.collect()
    # ---- end minimal maps + SFH ----