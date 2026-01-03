#!/usr/bin/env python3
"""
plot_sample_sfh_by_DoR.py

Produce a combined SFH plot for a sample of galaxies (colour-coded by DoR).

Usage examples:
  python3 plot_sample_sfh_by_DoR.py --ucmg ucmg_ids.csv --props relicness_merged.csv --out sfh_by_DoR.png --mode binned
  python3 plot_sample_sfh_by_DoR.py --ucmg ucmg_ids.csv --props relicness_merged.csv --out sfh_overlay.png --mode overlay --max-gals 200
"""
from __future__ import annotations
import os, sys, argparse, math, gc
from typing import List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# astroph stuff
from astropy.cosmology import Planck15
import astropy.units as u

# swift libs (must be on path / environment)
from swiftgalaxy import SWIFTGalaxy, SOAP

# ------------ CLI ------------
parser = argparse.ArgumentParser(description="Plot sample SFHs coloured by DoR")
parser.add_argument("--ucmg", required=True, help="CSV with subhalo_id list (ucmg_ids.csv)")
parser.add_argument("--props", required=True, help="CSV with galaxy properties that includes subhalo_id and DoR")
parser.add_argument("--soap-catalogue", required=True, help="SOAP catalogue HDF5 (halo_properties_....hdf5)")
parser.add_argument("--virtual-snap", required=True, help="Virtual snapshot HDF5 (colibre_with_SOAP_membership_....hdf5)")
parser.add_argument("--out", default="sfh_by_DoR.png", help="Output figure filename")
parser.add_argument("--mode", choices=("overlay","binned"), default="binned",
                    help="overlay = plot individual SFH lines; binned = median+percentile by DoR bins")
parser.add_argument("--nbins-time", type=int, default=200, help="Number of time bins for common grid")
parser.add_argument("--max-gals", type=int, default=None, help="Limit number of galaxies to process (for testing)")
parser.add_argument("--stride", type=int, default=1, help="Stride through input list (e.g. 2 -> every second galaxy).")
parser.add_argument("--mb-frac", type=float, default=95.0, help="Most-bound fraction to keep (percent).")
parser.add_argument("--min-particles", type=int, default=10, help="Skip galaxies with fewer star particles than this after mask.")
parser.add_argument("--bins-DoR", type=int, default=4, help="Number of DoR bins for binned mode (e.g. 4 quartiles)")
parser.add_argument("--alpha", type=float, default=0.06, help="Alpha for overlay lines")
parser.add_argument("--colormap", default="viridis", help="Matplotlib colormap")
parser.add_argument("--quiet", action="store_true", help="Less console output")
args = parser.parse_args()

def info(*a, **k):
    if not args.quiet:
        print(*a, **k, flush=True)

# ---------- read ID list and properties ----------
if not os.path.exists(args.ucmg):
    raise SystemExit("ucmg file not found: " + args.ucmg)
if not os.path.exists(args.props):
    raise SystemExit("properties file not found: " + args.props)
if not os.path.exists(args.soap_catalogue):
    info("WARNING: SOAP catalogue path not provided or not found. Using CLI value.")
if not os.path.exists(args.virtual_snap):
    info("WARNING: virtual snapshot not found.")

ucmg_df = pd.read_csv(args.ucmg)
if 'subhalo_id' in ucmg_df.columns:
    sub_ids = ucmg_df['subhalo_id'].to_numpy(dtype=np.int64)
else:
    sub_ids = ucmg_df.iloc[:,0].to_numpy(dtype=np.int64)

props = pd.read_csv(args.props)
if 'subhalo_id' not in props.columns:
    raise SystemExit("props CSV must contain a 'subhalo_id' column")
if 'DoR' not in props.columns:
    info("Warning: props CSV doesn't have 'DoR' column. Will set DoR=NaN and colour fallback to grey.")
    props['DoR'] = np.nan

# build quick lookup of DoR by subhalo_id
props_index = {int(r): float(d) if np.isfinite(d) else np.nan
               for r, d in zip(props['subhalo_id'].astype(int), pd.to_numeric(props['DoR'], errors='coerce').fillna(np.nan).to_numpy())}

# restrict to intersection of IDs
common_ids = np.intersect1d(sub_ids, props['subhalo_id'].to_numpy(dtype=np.int64))
info(f"Total IDs in UCMG file: {len(sub_ids)}. IDs present in props: {len(common_ids)}")
if common_ids.size == 0:
    raise SystemExit("No overlapping subhalo ids between ucmg and props.")

# build ordered list (apply stride & max_gals)
selected = common_ids[::args.stride]
if args.max_gals is not None:
    selected = selected[:args.max_gals]
info(f"Processing {len(selected)} galaxies (stride {args.stride}, max {args.max_gals})")

# ---------- common time grid ----------
t_now = Planck15.age(0).to(u.Gyr).value
t_grid = np.linspace(0.001, t_now, args.nbins_time)  # cosmic time [Gyr]

sfh_stack = []   # list of arrays (len = nbins_time)
do_list = []     # DoR per galaxy kept
ids_kept = []

# ---------- helper to extract SFH for one galaxy ----------
def extract_sfh_for_soap_index(soap_index: int, virtual_snapshot_file: str, soap_catalogue_file: str,
                               mb_frac: float = 95.0, min_particles: int = 10):
    """
    Return (t_form_gyr_array, masses_array) for the galaxy or (None,None) if not available.
    Uses SWIFTGalaxy + SOAP(soap_index).
    """
    sg = None
    try:
        sg = SWIFTGalaxy(virtual_snapshot_file, SOAP(soap_catalogue_file, soap_index=int(soap_index)))
    except Exception as e:
        info(f"  SWIFTGalaxy creation failed for soap_index {soap_index}: {e}")
        return None, None

    try:
        # coordinates
        coords = None
        try:
            coords = np.array(sg.stars.coordinates, dtype=float)
        except Exception:
            coords = None

        # masses: try multiple attribute names and convert to 1d Msun if possible
        m_raw = getattr(sg.stars, "masses", None)
        if m_raw is None:
            m_raw = getattr(sg.stars, "Masses", None)
        if m_raw is None:
            m_raw = getattr(sg.stars, "InitialMasses", None)

        def _masses_to_1d(arr):
            try:
                if arr is None:
                    return np.array([], dtype=float)
                # try pint-like unit conversion if available
                if hasattr(arr, "to") and callable(arr.to):
                    return np.array(arr.to("Msun").value, dtype=float)
            except Exception:
                pass
            try:
                return np.atleast_1d(np.array(arr, dtype=float))
            except Exception:
                return np.array([], dtype=float)

        masses = _masses_to_1d(m_raw)

        # formation times: BirthScaleFactors -> Ages fallback
        t_form_gyr = None
        a_birth = getattr(sg.stars, "BirthScaleFactors", None) or getattr(sg.stars, "birth_scale_factors", None)
        if a_birth is not None:
            try:
                a_birth = np.array(a_birth, dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    z_birth = (1.0 / a_birth) - 1.0
                # convert each z to cosmic time if valid
                t_form_gyr = np.array([Planck15.age(z).to(u.Gyr).value if np.isfinite(z) and (z >= 0) else np.nan for z in z_birth])
            except Exception:
                t_form_gyr = None
        else:
            ages_direct = getattr(sg.stars, "Ages", None) or getattr(sg.stars, "ages", None)
            if ages_direct is not None:
                try:
                    ages_direct = np.array(ages_direct, dtype=float)
                    t_now_local = Planck15.age(0).to(u.Gyr).value
                    t_form_gyr = t_now_local - ages_direct
                except Exception:
                    t_form_gyr = None

        # Basic existence checks
        if coords is None or masses is None or t_form_gyr is None:
            try:
                del sg
            except Exception:
                pass
            gc.collect()
            return None, None

        # mask most-bound
        radii = np.linalg.norm(coords, axis=1)
        if radii.size == 0:
            try:
                del sg
            except Exception:
                pass
            gc.collect()
            return None, None

        keep_mb_mask = radii <= np.percentile(radii, mb_frac)
        masses = np.atleast_1d(masses)[keep_mb_mask]
        t_form_gyr = np.atleast_1d(np.array(t_form_gyr, dtype=float))[keep_mb_mask]

        # ensure matched lengths
        if masses.size != t_form_gyr.size:
            nmin = min(masses.size, t_form_gyr.size)
            masses = masses[:nmin]
            t_form_gyr = t_form_gyr[:nmin]

        if masses.size < min_particles:
            try:
                del sg
            except Exception:
                pass
            gc.collect()
            return None, None

        try:
            del sg
        except Exception:
            pass
        gc.collect()
        return t_form_gyr, masses

    except Exception as e:
        info(f"  Error extracting SFH for soap_index {soap_index}: {e}")
        try:
            del sg
        except Exception:
            pass
        gc.collect()
        return None, None

# ---------- map subhalo_id -> SOAP index ----------
import h5py
with h5py.File(args.soap_catalogue, 'r') as fh:
    idx_path = None
    for cand in ('InputHalos/HaloCatalogueIndex', 'HaloCatalogueIndex', 'InputHalos/HaloCatalogue_Index'):
        if cand in fh:
            idx_path = cand
            break
    if idx_path is None:
        keys = list(fh.keys())
        raise SystemExit(f"Could not find HaloCatalogueIndex in SOAP file; available keys: {keys}")
    halo_idx_array = np.array(fh[idx_path], dtype=np.int64)

val_to_row = {int(val): int(i) for i, val in enumerate(halo_idx_array)}

# ---------- main loop: extract SFHs, interpolate to grid ----------
n_processed = 0
for sid in tqdm(selected, desc="galaxies"):
    sid_int = int(sid)
    if sid_int not in val_to_row:
        continue
    soap_row_idx = val_to_row[sid_int]
    dor = props_index.get(sid_int, np.nan)

    tform, masses = extract_sfh_for_soap_index(soap_row_idx, args.virtual_snap, args.soap_catalogue,
                                              mb_frac=args.mb_frac, min_particles=args.min_particles)
    if tform is None or masses is None:
        continue

    # sort by formation time and compute cumulative mass fraction
    order = np.argsort(tform)
    t_sorted = tform[order]
    m_sorted = masses[order]

    # drop non-finite entries
    finite_mask = np.isfinite(t_sorted) & np.isfinite(m_sorted)
    if finite_mask.sum() < 3:
        continue
    t_sorted = t_sorted[finite_mask]
    m_sorted = m_sorted[finite_mask]

    # ensure strictly increasing times (drop duplicates)
    if t_sorted.size >= 2:
        # detect strictly increasing sequence, drop duplicates
        unique_t, unique_idx = np.unique(t_sorted, return_index=True)
        if unique_t.size < t_sorted.size:
            # keep the first occurrence of each unique time
            t_sorted = t_sorted[np.sort(unique_idx)]
            m_sorted = m_sorted[np.sort(unique_idx)]

    # after cleaning, need at least 3 points to interpolate reasonably
    if t_sorted.size < 3:
        continue

    cum_mass = np.cumsum(m_sorted)
    if cum_mass.size == 0 or cum_mass[-1] == 0.0:
        continue
    cum_frac = cum_mass / cum_mass[-1]

    # clamp times into grid bounds and interpolate
    t_sorted_clamped = np.clip(t_sorted, t_grid[0], t_grid[-1])
    try:
        interp = np.interp(t_grid, t_sorted_clamped, cum_frac, left=0.0, right=1.0)
    except Exception as e:
        info(f"  Interpolation failed for soap_index {soap_row_idx}: {e}")
        continue

    sfh_stack.append(interp)
    do_list.append(dor if np.isfinite(dor) else np.nan)
    ids_kept.append(sid_int)
    n_processed += 1

info(f"Extracted SFHs for {n_processed} galaxies (kept).")

if n_processed == 0:
    raise SystemExit("No galaxies had usable SFHs. Exiting.")

sfh_array = np.vstack(sfh_stack)  # shape (N, nbins_time)
do_arr = np.array(do_list, dtype=float)

# ---------- plotting ----------
import matplotlib as mpl
cmap = plt.get_cmap(args.colormap)
# robust normalization: use 1-99 percentiles to avoid outliers crushing colormap
vmin = np.nanpercentile(do_arr, 1) if np.isfinite(do_arr).any() else 0.0
vmax = np.nanpercentile(do_arr, 99) if np.isfinite(do_arr).any() else 1.0
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

fig, ax = plt.subplots(figsize=(8,6))

if args.mode == "overlay":
    # colour each galaxy by DoR and plot with low alpha
    for row, dor in zip(sfh_array, do_arr):
        color = cmap(norm(dor)) if np.isfinite(dor) else (0.5,0.5,0.5)
        ax.plot(t_grid, row, color=color, alpha=args.alpha, lw=0.8)

    # add colorbar scaffold: create a mappable and attach to the axes explicitly
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])   # required for some Matplotlib versions
    cbar = fig.colorbar(sm, ax=ax)   # attach to the explicit Axes

    ax.set_xlabel("Cosmic time [Gyr]")
    ax.set_ylabel("Cumulative stellar mass fraction")
    ax.set_title(f"SFHs (overlay) — {sfh_array.shape[0]} galaxies")
    ax.set_ylim(-0.02, 1.02)

elif args.mode == "binned":
    fig, ax = plt.subplots(figsize=(8,6))
    # ... the rest of your binned code should use ax.plot / ax.fill_between instead of plt.plot
    # Example inside the loop:
    # ax.plot(t_grid, med, color=color, lw=2, label=...)
    # ax.fill_between(t_grid, p16, p84, color=color, alpha=0.25)
    ax.set_xlabel("Cosmic time [Gyr]")
    ax.set_ylabel("Cumulative stellar mass fraction")
    ax.set_title(f"SFH medians by DoR bins — {sfh_array.shape[0]} galaxies")
    ax.set_ylim(-0.02, 1.02)

# final save (works because fig is defined)
fig.savefig(args.out, dpi=150, bbox_inches="tight")
plt.close(fig)
info("Saved figure to", args.out)