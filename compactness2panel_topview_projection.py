import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import common   # your helper that provides read_group_data_colibre
import h5py

plt.rcParams.update({"mathtext.fontset":"stix", "font.family":"serif", "font.size":13})


# ------------------------------------------------------------------
# threshold finder
# ------------------------------------------------------------------
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree

def find_compactness_threshold(cbin_centers, cmed, compactness_all, quantity_all,
                               counts_per_bin=None,
                               sg_window=9, sg_poly=2,
                               gauss_sigma=1.0,
                               edge_frac=0.02,
                               deriv_thresh_factor=3.0,
                               min_bin_count=5,
                               bootstrap_n=0,
                               random_seed=12345,
                               cbins_edges=None): 
    """
    Return dict with threshold, method, smoothed median+derivative and bootstrap summary.
    Inputs:
      - cbin_centers, cmed : arrays from your projection (length nbins)
      - compactness_all, quantity_all : full galaxy arrays used to build the projection
      - counts_per_bin : optional nbins-length array with N per compactness bin
    Tune sg_window/sg_poly, gauss_sigma, deriv_thresh_factor, bootstrap_n.
    """
    # ensure numpy arrays
    cbin_centers = np.asarray(cbin_centers)
    cmed = np.asarray(cmed)
    n = len(cbin_centers)

    # smoothing window (must be odd and <= n)
    sw = min(sg_window if (sg_window % 2 == 1) else (sg_window+1), n if (n % 2 == 1) else (n-1))
    if sw < 3:
        sw = 3

    # replace NaN in cmed with median of finite entries for smoothing input
    finite_mask = np.isfinite(cmed)
    if finite_mask.sum() == 0:
        raise ValueError("cmed contains no finite values.")
    fill_value = float(np.nanmedian(cmed[finite_mask]))
    cmed_to_smooth = np.nan_to_num(cmed, nan=fill_value)

    # 1) smooth median
    try:
        cmed_s = savgol_filter(cmed_to_smooth, window_length=sw, polyorder=sg_poly)
    except Exception:
        cmed_s = gaussian_filter1d(cmed_to_smooth, sigma=gauss_sigma)

    # 2) derivative and smoothed derivative
    deriv = np.gradient(cmed_s, cbin_centers)
    deriv_s = gaussian_filter1d(deriv, sigma=gauss_sigma)
    abs_deriv = np.abs(deriv_s)

    # 3) robust threshold on derivative
    baseline = np.nanmedian(abs_deriv)
    mad = np.nanmedian(np.abs(abs_deriv - baseline))
    deriv_threshold = baseline + deriv_thresh_factor * (1.4826 * mad)

    # 4) candidate turning point
    idx_max = int(np.nanargmax(abs_deriv))
    left_edge_idx = int(np.floor(edge_frac * n))
    right_edge_idx = int(np.ceil((1 - edge_frac) * n)) - 1

    method = None
    threshold_value = None

    if (idx_max > left_edge_idx) and (idx_max < right_edge_idx) and (abs_deriv[idx_max] >= deriv_threshold):
        method = 'turning_point'
        threshold_value = float(cbin_centers[idx_max])
    else:
        # fallback: first bin from right (most compact) where abs_deriv > threshold and (optionally) count >= min_bin_count
        found = False
        for i in range(n-1, -1, -1):
            if abs_deriv[i] >= deriv_threshold:
                if (counts_per_bin is None) or (counts_per_bin[i] >= min_bin_count):
                    threshold_value = float(cbin_centers[i])
                    found = True
                    break
        if found:
            method = 'start_exceed'
        else:
            # last fallback: 90th percentile of compactness
            method = 'fallback_percentile'
            threshold_value = float(np.nanpercentile(compactness_all, 90))

        # 5) bootstrap: resample within compactness bins to produce threshold distribution
    rng = np.random.default_rng(seed=random_seed)
    thr_boot = []
    thr_boot_method = []

    # if user didn't request bootstrap, skip the heavy section and return bootstrap=None
    if bootstrap_n <= 0:
        bootstrap_stats = None
    else:
        # reconstruct cbins_edges from centers if not provided
        if cbins_edges is None:
            # build edges robustly from (possibly non-uniform) centers:
            if n >= 2:
                edges = np.empty(n + 1, dtype=float)
                # internal edges = midpoints between adjacent centers
                edges[1:-1] = 0.5 * (cbin_centers[:-1] + cbin_centers[1:])
                # first and last edges: extend by same half-width as nearest interval
                first_half = edges[1] - cbin_centers[0]
                last_half = cbin_centers[-1] - edges[-2]
                edges[0] = cbin_centers[0] - first_half
                edges[-1] = cbin_centers[-1] + last_half
                cbins_edges = edges
            else:
                cbins_edges = np.array([cbin_centers[0] - 0.5, cbin_centers[0] + 0.5])

        # map galaxies to bins and collect values per bin
        idxs = np.searchsorted(cbins_edges, compactness_all) - 1
        valid_mask = (idxs >= 0) & (idxs < n) & np.isfinite(quantity_all)
        if valid_mask.sum() == 0:
            bootstrap_stats = None
        else:
            # bucket values per bin
            bins_values = [[] for _ in range(n)]
            for ind, q in zip(idxs[valid_mask], quantity_all[valid_mask]):
                bins_values[ind].append(q)

            # bootstrap iterations
            for it in range(bootstrap_n):
                med_bs = np.full(n, np.nan)
                for i_bin in range(n):
                    vals = bins_values[i_bin]
                    if len(vals) == 0:
                        med_bs[i_bin] = np.nan
                    else:
                        sample = rng.choice(vals, size=len(vals), replace=True)
                        med_bs[i_bin] = np.nanmedian(sample)

                # smooth + derivative on this bootstrap median
                med_fill = float(np.nanmedian(med_bs[np.isfinite(med_bs)]) if np.any(np.isfinite(med_bs)) else np.nan)
                med_bs_to_smooth = np.nan_to_num(med_bs, nan=med_fill)
                try:
                    meds_s = savgol_filter(med_bs_to_smooth, window_length=sw, polyorder=sg_poly)
                except Exception:
                    meds_s = gaussian_filter1d(med_bs_to_smooth, sigma=gauss_sigma)
                deriv_b = np.gradient(meds_s, cbin_centers)
                deriv_bs = gaussian_filter1d(deriv_b, sigma=gauss_sigma)
                abs_bs = np.abs(deriv_bs)

                # robust threshold on this bootstrap derivative
                baseline_b = np.nanmedian(abs_bs)
                mad_b = np.nanmedian(np.abs(abs_bs - baseline_b))
                thr_b = baseline_b + deriv_thresh_factor * (1.4826 * mad_b)
                idx_max_b = int(np.nanargmax(abs_bs))

                if (idx_max_b > left_edge_idx) and (idx_max_b < right_edge_idx) and (abs_bs[idx_max_b] >= thr_b):
                    thr_boot.append(float(cbin_centers[idx_max_b]))
                    thr_boot_method.append("turning_point")
                else:
                    found_b = False
                    for j in range(n-1, -1, -1):
                        if abs_bs[j] >= thr_b and (counts_per_bin is None or counts_per_bin[j] >= min_bin_count):
                            thr_boot.append(float(cbin_centers[j]))
                            thr_boot_method.append("start_exceed")
                            found_b = True
                            break
                    if not found_b:
                        thr_boot.append(float(np.nanpercentile(compactness_all, 90)))
                        thr_boot_method.append("fallback_percentile")

            # finish bootstrap stats
            thr_arr = np.array(thr_boot)
            bootstrap_stats = {
                'median': float(np.nanmedian(thr_arr)),
                'p16': float(np.nanpercentile(thr_arr, 16)),
                'p84': float(np.nanpercentile(thr_arr, 84)),
                'raw': thr_arr,
                'methods': thr_boot_method
            }

    return {
        'threshold': threshold_value,
        'method': method,
        'cmed_smooth': cmed_s,
        'derivative': deriv_s,
        'deriv_threshold': deriv_threshold,
        'bootstrap': bootstrap_stats
    }
# ------------------------------------------------------------------

# --- Add this helper block to test multiple bin sizes (equal-count / quantile bins) ---

def thresholds_for_targets(compactness_all, quantity_all, targets_per_bin=(50,100,300,1000),
                           min_count=5, bootstrap_n=0, random_seed=12345,
                           do_bootstrap=True):
    """
    For each target number of galaxies per compactness bin, build quantile (equal-count) bins,
    compute binned medians and percentiles, call find_compactness_threshold, and return summary.
    Returns list of dicts with keys:
      'target', 'nbins', 'cbin_centers', 'cmed', 'clow', 'chigh', 'counts', 'threshold_result'
    """
    res_list = []
    compactness_all = np.asarray(compactness_all)
    quantity_all = np.asarray(quantity_all)

    N = compactness_all.size
    if N == 0:
        return []

    for target in np.atleast_1d(targets_per_bin):
        if target <= 0:
            continue
        nbins = max(3, int(np.floor(N / target)))  # at least 3 bins
        # build equal-count bin edges (quantiles)
        # we want nbins bins -> (nbins+1) edges
        quantiles = np.linspace(0.0, 1.0, nbins+1)
        edges = np.nanpercentile(compactness_all, 100.0 * quantiles)
        # ensure monotonic edges (numerical protection)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + 1e-9

        # bin centers and counts
        cbin_centers = 0.5 * (edges[:-1] + edges[1:])
        counts = np.zeros(nbins, dtype=int)
        med = np.full(nbins, np.nan)
        low = np.full(nbins, np.nan)
        high = np.full(nbins, np.nan)

        # assign indices
        idxs = np.searchsorted(edges, compactness_all, side='right') - 1
        valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(quantity_all)
        # aggregate values per bin
        from collections import defaultdict
        group = defaultdict(list)
        for ii, q in zip(idxs[valid_mask], quantity_all[valid_mask]):
            group[ii].append(q)
        for b in range(nbins):
            vals = group.get(b, [])
            counts[b] = len(vals)
            if counts[b] >= min_count:
                arr = np.asarray(vals, dtype=float)
                med[b] = np.nanmedian(arr)
                low[b] = np.nanpercentile(arr, 16)
                high[b] = np.nanpercentile(arr, 84)
            else:
                med[b] = np.nan; low[b] = np.nan; high[b] = np.nan

        # optional: call your find_compactness_threshold. If you haven't imported it here, replace with fallback:
        try:
            thr_res = find_compactness_threshold(
                cbin_centers=cbin_centers,
                cmed=med,
                compactness_all=compactness_all,
                quantity_all=quantity_all,
                counts_per_bin=counts,
                bootstrap_n=(bootstrap_n if do_bootstrap else 0),
                random_seed=random_seed,
            )
        except Exception as e:
            # fallback simple threshold: 90th percentile of compactness if nothing found
            thr_res = {'threshold': float(np.nanpercentile(compactness_all, 90)),
                       'method': 'fallback_error',
                       'bootstrap': None}

        res_list.append({
            'target_per_bin': int(target),
            'nbins': int(nbins),
            'edges': edges,
            'cbin_centers': cbin_centers,
            'counts': counts,
            'cmed': med,
            'clow': low,
            'chigh': high,
            'threshold_result': thr_res
        })

    return res_list

# ------------------ Config / dataset selection ------------------
model_name = 'L0200N3008/THERMAL_AGN/'
model_dir  = '/mnt/su3-pro/colibre/' + model_name

#h5path = '/mnt/su3ctm/kproctor/ForMax/exsitu_summary_SnapNum_127.hdf5'
from pathlib import Path

# automatically find the ex-situ file
base_dir = Path("/mnt/su3ctm/kproctor/ForMax")
matches = sorted(base_dir.glob("*exsitu*summary*.hdf5"))

if len(matches) == 0:
    raise FileNotFoundError(f"No ex-situ HDF5 file found in {base_dir}")
elif len(matches) == 1:
    h5path = str(matches[0])
else:
    # if there are several, pick the newest one
    h5path = str(max(matches, key=lambda p: p.stat().st_mtime))

print("Using ex-situ file:", h5path)

# snapshot selection (z=0 in your workflow)
snap_files = ['0127', '0119', '0114', '0102', '0092', '0076', '0064', '0056', '0048', '0040', '0026', '0018']
zstarget   = [0.0,    0.1,   0.2,   0.5,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   8.0,   10.0]
snap_file  = snap_files[0]
ztarget    = zstarget[0]
comov_to_physical_length = 1.0 / (1.0 + ztarget)

outdir = os.path.join(os.getcwd(), "plots")
os.makedirs(outdir, exist_ok=True)

# ------------------ Fields to read (same as your previous script) ------------------
fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')}
# notice we include the two linear-mass weighted element fields that you mentioned
fields = {'ExclusiveSphere/50kpc': (
            'StellarMass', 'StarFormationRate', 'HalfMassRadiusStars', 'CentreOfMass',
            'MassWeightedMeanStellarAge', 'LuminosityWeightedMeanStellarAge',
            'LinearMassWeightedIronOverHydrogenOfStars', 'LinearMassWeightedMagnesiumOverHydrogenOfStars', 'StellarMassFractionInMetals'
         )}
fields_proj = {'ProjectedAperture/50kpc/projz': ('StellarMass', 'HalfMassRadiusStars')}

h5data_groups      = common.read_group_data_colibre(model_dir, snap_file, fields)
h5data_idgroups    = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)
h5data_groups_proj = common.read_group_data_colibre(model_dir, snap_file, fields_proj)

# unpack
(m30, sfr30, r50, cp, stellarage, stellarage_lum, FeoverH, MgoverH, Zstar_raw) = h5data_groups
(m30_proj, r50_proj) = h5data_groups_proj
(sgn, is_central, desc_id, track_id) = h5data_idgroups

soap_id = {'SOAP': ('HostHaloIndex',)}
h5data_soap = common.read_group_data_colibre(model_dir, snap_file, soap_id)
(host_halo_index) = h5data_soap

# ------------------ Units conversion (same as your script) ------------------
Lu = 3.086e+24/(3.086e+24)   # cMpc -> Mpc factor (kept 1)
Mu = 1.988e+43/(1.989e+33)   # raw mass -> Msun
tu = 3.086e+19/(3.154e+7)    # time unit -> yrs

m30 = m30 * Mu
m30_proj = m30_proj * Mu
sfr30 = sfr30 * Mu / tu
r50 = r50 * Lu * comov_to_physical_length * 1e3
r50_proj = r50_proj * Lu * comov_to_physical_length * 1e3
stellarage = stellarage * tu / 1e9
stellarage_lum = stellarage_lum * tu / 1e9
cp = cp * Lu * comov_to_physical_length

Zsun = 0.0134   # AGSS09 convention
# Zsun = 0.0139 # Asplund et al. 2021 present-day photospheric value
    
Zstar = np.asarray(Zstar_raw, dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    logZstar = np.where((Zstar > 0) & np.isfinite(Zstar), np.log10(Zstar), np.nan)
    logZstar_rel = np.where((Zstar > 0) & np.isfinite(Zstar),
                            np.log10(Zstar / Zsun),
                            np.nan)

# ------------------ select galaxies (same selection) ------------------
select = np.where(m30 >= 1e9)
ngals = len(m30[select])
if ngals == 0:
    raise SystemExit("No galaxies selected (m30 >= 1e9)")

# arrays for the selected galaxies
m_in = m30[select]
r50_in = r50[select]
sgn_in = sgn[select]
# elemental quantities from the same selection
Fe_in  = FeoverH[select]
Mg_in  = MgoverH[select]
Zstar_in = Zstar[select]
logZstar_in = logZstar[select]
logZstar_rel_in = logZstar_rel[select]
stellarage_in = stellarage[select]
stellarage_lum_in = stellarage_lum[select]
sfr_in = sfr30[select] 
desc_id_in = desc_id[select]
track_id_in = track_id[select]  # IMPORTANT: this is the 'track id' that maps to ex-situ HDF5

# ensure proper shapes (flatten)
m_in = np.asarray(m_in).ravel()
r50_in = np.asarray(r50_in).ravel()
Fe_in = np.asarray(Fe_in).ravel()
Mg_in = np.asarray(Mg_in).ravel()
sgn_in = np.asarray(sgn_in).ravel()
Zstar_in = np.asarray(Zstar_in).ravel()
logZstar_in = np.asarray(logZstar_in).ravel()
logZstar_rel_in = np.asarray(logZstar_rel_in).ravel()
# ------------------ compute logs and compactness ------------------
mask_positive = (m_in > 0) & (r50_in > 0)
if not np.any(mask_positive):
    raise RuntimeError("No positive mtot/r50 values to plot after filtering selection.")

log_m = np.log10(m_in[mask_positive])
log_r = np.log10(r50_in[mask_positive])
compactness = log_m - 1.5 * log_r   

# ------------------ compute [Mg/Fe] ----------
log_MgFe_sun = +0.10
Mg = np.asarray(Mg_in[mask_positive], dtype=float)
Fe = np.asarray(Fe_in[mask_positive], dtype=float)

with np.errstate(divide="ignore", invalid="ignore"):
    MgFe_number = (Mg / Fe)
    log10_number = np.where(MgFe_number > 0, np.log10(MgFe_number), np.nan)
    mgfe = log10_number - log_MgFe_sun    # aligned with log_m / log_r / compactness

# basic info
print(f"Selected galaxies: {len(compactness)} ; Mg/Fe finite: {np.isfinite(mgfe).sum()}")

# choose targets (galaxies per compactness bin) to test
targets = (100, 300, 500, 1000, 5000, 7000, 10000)

# quick, fast scan WITHOUT bootstrap (safe)
summary_mgfe = thresholds_for_targets(compactness, mgfe, targets_per_bin=targets,
                                      min_count=5, bootstrap_n=0, do_bootstrap=False)

# print results
for s in summary_mgfe:
    thr = s['threshold_result']
    print(f"Mg/Fe target={s['target_per_bin']} -> nbins={s['nbins']} threshold={thr['threshold']:.3f} method={thr['method']}")


# ---------- load ex-situ fractions from HDF5 WITHOUT huge dict ----------
halo_selected = np.asarray(sgn_in[mask_positive], dtype=np.int64)
exsitu_fracs = np.full(halo_selected.shape, np.nan, dtype=np.float32)

if os.path.exists(h5path):
    with h5py.File(h5path, "r") as fh:
        if "stars" not in fh:
            print("HDF5 file missing dataset 'stars'; skipping ex-situ matching.")
        else:
            dset = fh["stars"]
            nrows = dset.shape[0]

            chunk = 500_000
            ids_all = np.empty(nrows, dtype=np.int64)
            fracs_all = np.empty(nrows, dtype=np.float32)

            for i0 in range(0, nrows, chunk):
                i1 = min(nrows, i0 + chunk)
                block = dset[i0:i1, :]
                ids_all[i0:i1] = block[:, 0].astype(np.int64, copy=False)
                fracs_all[i0:i1] = block[:, 3].astype(np.float32, copy=False)

            order = np.argsort(ids_all)
            ids_s = ids_all[order]
            fracs_s = fracs_all[order]

            pos = np.searchsorted(ids_s, halo_selected)
            ok = (pos < ids_s.size) & (ids_s[pos] == halo_selected)
            exsitu_fracs[ok] = fracs_s[pos[ok]]

            print(f"Loaded+matched ex-situ in vectorised mode: {ok.sum()} / {halo_selected.size}")
else:
    print("Ex-situ HDF5 not found at:", h5path)
# report match stats
n_matched = np.isfinite(exsitu_fracs).sum()
print(f"Matched ex-situ fraction for {n_matched} / {len(exsitu_fracs)} selected galaxies")

# --------------------------------------------------------------
# LOAD HOST VELOCITY DISPERSION (CORRECT + ALIGNED)
# --------------------------------------------------------------
sigma_path = "/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN/SOAP-HBT/extra/halo_properties_0127.hdf5"

# mask of galaxies you actually want to plot
mask_positive_full = (m30 >= 1e9) & (m30 > 0) & (r50 > 0)

# row positions in the SOAP catalogue
row_idx = np.flatnonzero(mask_positive_full)

# allocate full-length array if you want to keep SOAP alignment
sigma_full = np.full(m30.shape, np.nan, dtype=np.float32)

sigma_path = "/mnt/su3-pro/colibre/L0200N3008/THERMAL_AGN/SOAP-HBT/extra/halo_properties_0127.hdf5"
sigma_ds = "/ExclusiveSphere/HalfMassRadiusStars/StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"

if os.path.exists(sigma_path):
    with h5py.File(sigma_path, "r") as f:
        ds = f[sigma_ds]
        print("sigma dataset shape:", ds.shape)

        # read only the selected rows
        rows = np.asarray(ds[row_idx, :], dtype=np.float32)   # shape (N, 9)

        # diagonal components of the 3x3 tensor
        sigma_rr   = rows[:, 0]
        sigma_pphi = rows[:, 4]
        sigma_zz   = rows[:, 8]

        # your requested scalar sigma
        sigma_sel = np.sqrt((sigma_rr**2 + sigma_pphi**2 + sigma_zz**2)/3)

        # put back into full SOAP-aligned array
        sigma_full[row_idx] = sigma_sel

        # log sigma for plotting
        log_sigma_full = np.full(m30.shape, np.nan, dtype=np.float32)
        log_sigma_full[row_idx] = np.where(sigma_sel > 0, np.log10(sigma_sel), np.nan)

    print("Loaded sigma values:", np.isfinite(sigma_sel).sum(), "/", sigma_sel.size)
    print("N(sigma == 0):", np.count_nonzero(np.isclose(sigma_sel[np.isfinite(sigma_sel)], 0.0)))
else:
    print("Sigma file not found.")

sigma_vals = sigma_full[mask_positive_full]
log_sigma_vals = log_sigma_full[mask_positive_full]

# # ------------------ two-panel plotting (Mg over Fe) ------------------

# fig, axes = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={"width_ratios":[1.2,1]})
# ax_main, ax_proj = axes

# # Left: scatter of compactness vs mgfe (grey) + hexbin median overlay
# # Plot light-grey scatter of all galaxies (including NaN mgfe)
# ax_main.scatter(compactness, mgfe, s=8, color="lightgrey", alpha=0.8, label="galaxies")

# # Hexbin of median mgfe in compactness-x bins and mgfe-y bins:
# # We'll use matplotlib's hexbin but compute medians per hex via scipy / numpy aggregation.
# # Simpler and robust: use a 2D bin grid and compute median inside each bin.
# nx = 60  # horizontal bins (compactness)
# ny = 60  # vertical bins (mgfe)
# # binned grid limits
# cmin, cmax = np.nanpercentile(compactness, [1,99])
# ymin, ymax = np.nanpercentile(mgfe[~np.isnan(mgfe)], [1,99]) if np.any(np.isfinite(mgfe)) else (np.nanmin(mgfe), np.nanmax(mgfe))
# # expand a bit
# padc = 0.1 * (cmax - cmin + 1e-6)
# pady = 0.1 * (ymax - ymin + 1e-6)
# xbins = np.linspace(cmin-padc, cmax+padc, nx+1)
# ybins = np.linspace(ymin-pady, ymax+pady, ny+1)

# # indices for each point (only consider finite mgfe for median calculation)
# finite = np.isfinite(mgfe)
# xi = np.searchsorted(xbins, compactness[finite]) - 1
# yi = np.searchsorted(ybins, mgfe[finite]) - 1
# valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
# xi = xi[valid]; yi = yi[valid]
# vals = mgfe[finite][valid]

# # compute median per bin
# med_grid = np.full((ny, nx), np.nan)
# count_grid = np.zeros((ny, nx), dtype=int)
# for xb, yb, v in zip(xi, yi, vals):
#     count_grid[yb, xb] += 1
#     if np.isnan(med_grid[yb, xb]):
#         med_grid[yb, xb] = v
#     else:
#         # accumulate in list would be simpler, but avoid memory overhead: store temporarily via lists
#         # we instead collect using a dict of lists (small overhead)
#         pass

# # Because we used naive median accumulation above, do it properly with grouping:
# from collections import defaultdict
# group = defaultdict(list)
# for xb, yb, v in zip(xi, yi, vals):
#     group[(yb, xb)].append(v)
# for (yb, xb), lst in group.items():
#     med_grid[yb, xb] = np.nanmedian(lst)
#     count_grid[yb, xb] = len(lst)

# # Make the meshgrid of bin centres for plotting with pcolormesh
# xc = 0.5*(xbins[:-1] + xbins[1:])
# yc = 0.5*(ybins[:-1] + ybins[1:])
# Xc, Yc = np.meshgrid(xc, yc)

# # Plot medians on top using pcolormesh (mask empty bins)
# # med_masked = np.ma.masked_invalid(med_grid)
# # cmap = plt.get_cmap("viridis")
# # pcm = ax_main.pcolormesh(xbins, ybins, med_masked, cmap=cmap, shading='auto')
# # cbar = fig.colorbar(pcm, ax=ax_main)
# # cbar.set_label("[Mg/Fe] (dex)")

# # --- KDE density contours (robust, with KDTree masking + filled contours) ---
# x = compactness
# y = mgfe    # for the Mg/Fe panel; replace with other y as needed

# # mask finite points only
# fin = np.isfinite(x) & np.isfinite(y)
# if np.sum(fin) < 10:
#     print("Skipping KDE contours: too few finite points.")
# else:
#     xs = x[fin]; ys = y[fin]

#     pts = np.vstack([xs, ys])          # shape (2, N)
#     kde = gaussian_kde(pts, bw_method='scott')

#     # build evaluation grid exactly like your original code
#     nx_grid = 200
#     ny_grid = 200
#     x_min, x_max = np.nanpercentile(x, [1, 99])
#     y_min, y_max = np.nanpercentile(y, [1, 99])
#     xpad = 0.05 * (x_max - x_min + 1e-9)
#     ypad = 0.05 * (y_max - y_min + 1e-9)
#     xg = np.linspace(x_min - xpad, x_max + xpad, nx_grid)
#     yg = np.linspace(y_min - ypad, y_max + ypad, ny_grid)
#     Xgrid, Ygrid = np.meshgrid(xg, yg)
#     grid_pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T    # (nx*ny, 2)

#     # KDTree mask: mark grid cells far from any data point
#     tree = cKDTree(np.column_stack((xs, ys)))
#     d_grid, _ = tree.query(grid_pts, k=1)

#     # estimate typical spacing from data (like your LOESS approach)
#     d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.median(d_grid))

#     # cutoff: keep grid points within ~1.3 * typical spacing (tweak multiplier if needed)
#     cut = max(typical_spacing * 1.3, 1e-6)
#     mask_far = (d_grid > cut)

#     # evaluate KDE on grid and mask far-away cells
#     Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
#     Z_flat = Z.ravel()
#     Z_flat[mask_far] = np.nan
#     Z = Z_flat.reshape(Xgrid.shape)

#     # pick contour levels from finite Z values
#     finite_vals = Z[np.isfinite(Z)]
#     if finite_vals.size == 0:
#         print("KDE produced no finite values inside masked region; skipping contours.")
#     else:
#         levs = np.percentile(finite_vals, [50, 75, 90, 97])

#         # filled continuous density map (many levels => visually smooth)
#         cf = ax_main.contourf(Xgrid, Ygrid, Z, levels=50, cmap='viridis', antialiased=True)

#         # optional: overlay a few contour lines
#         cs = ax_main.contour(Xgrid, Ygrid, Z, levels=levs, colors='k', linewidths=0.6, alpha=0.5)

#         # colorbar attached to the filled contours
#         fig.colorbar(cf, ax=ax_main, label='Density (KDE)')

# ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_main.set_ylabel("[Mg/Fe]")
# # ax_main.set_title("Compactness vs [Mg/Fe] — 2D median")
# ax_main.grid(True)

# # Add running-median line (binned along compactness)
# # ------------------ Running-median / projection (fixed-width or quantile bins) ------------------
# # Replace your existing 'Add running-median line (binned along compactness)' block with this.

# # Configuration: choose one of the two binning modes:
# use_quantile_bins = True   # True -> equal-count (quantile) bins; False -> fixed-width bins (nbins)
# target_per_bin = 1000       # used only if use_quantile_bins is True (approx. galaxies per bin)
# nbins_fixed = 18           # used only if use_quantile_bins is False (number of fixed-width bins)
# min_count_per_bin = 5      # minimum galaxies required to compute median in a bin

# # prepare the compactness and the quantity you want to project
# # (here assume `compactness` and the current quantity array `q_arr` are defined and aligned)
# # For Mg/Fe block q_arr = mgfe; for ex-situ block q_arr = exsitu_fracs; for ages q_arr = ages; etc.
# q_arr = mgfe   # <<-- change to the actual quantity inside each block (mgfe / exsitu_fracs / ages / log_ssfr)

# # filter valid compactness (this should already be aligned in your script)
# valid_comp = np.isfinite(compactness)
# Ntot = np.sum(valid_comp)
# if Ntot == 0:
#     raise RuntimeError("No valid galaxies to bin in compactness.")

# if use_quantile_bins:
#     # compute number of bins from target_per_bin, at least 3
#     nbins = max(3, int(np.floor(Ntot / max(1, int(target_per_bin)))))
#     # compute quantile edges in compactness space (may produce very small edge steps if many duplicates)
#     quantiles = np.linspace(0.0, 100.0, nbins + 1)
#     edges = np.nanpercentile(compactness, quantiles)
#     # guard: enforce strictly increasing edges (tiny jitter if necessary)
#     for i in range(1, edges.size):
#         if edges[i] <= edges[i-1]:
#             edges[i] = edges[i-1] + 1e-9
#     cbins = edges.copy()
# else:
#     nbins = int(nbins_fixed)
#     cbins = np.linspace(np.nanmin(compactness), np.nanmax(compactness), nbins + 1)

# # centers for plotting and threshold finder compatibility
# cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])

# # allocate arrays
# cmed = np.full(nbins, np.nan)
# clow = np.full(nbins, np.nan)
# chigh = np.full(nbins, np.nan)
# counts_per_bin = np.zeros(nbins, dtype=int)

# # assign each galaxy to a bin (use side='right' like earlier behaviour)
# idxs = np.searchsorted(cbins, compactness, side='right') - 1
# # valid galaxies for bin-aggregation: index in range AND quantity finite
# valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(q_arr)

# # aggregate values per bin (small memory; robust)
# from collections import defaultdict
# group = defaultdict(list)
# for idx, q in zip(idxs[valid_mask], q_arr[valid_mask]):
#     group[int(idx)].append(float(q))

# for b in range(nbins):
#     vals = group.get(b, [])
#     counts_per_bin[b] = len(vals)
#     if counts_per_bin[b] >= min_count_per_bin:
#         arr = np.asarray(vals, dtype=float)
#         cmed[b] = np.nanmedian(arr)
#         clow[b] = np.nanpercentile(arr, 16)
#         chigh[b] = np.nanpercentile(arr, 84)
#     else:
#         cmed[b] = np.nan
#         clow[b] = np.nan
#         chigh[b] = np.nan

# ax_main.plot(cbin_centers, cmed, color="black", lw=2, label="median (binned)")
# ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15, label="16–84 pct")
# ax_main.legend(fontsize=9)

# # Right panel: projection (median mgfe vs compactness) — line + shaded percentile (cleaner)
# ax_proj.plot(cbin_centers, cmed, color="C0", lw=2)
# ax_proj.fill_between(cbin_centers, clow, chigh, color="C0", alpha=0.25)
# ax_proj.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_proj.set_ylabel("[Mg/Fe]")
# # ax_proj.set_title("Projection: median & 16–84 pct")
# ax_proj.grid(True)

# # --- call threshold finder for Mg/Fe ---
# res_mgfe = find_compactness_threshold(
#     cbin_centers=cbin_centers,
#     cmed=cmed,                      # this is the cmed you computed for mgfe
#     compactness_all=compactness,    # full galaxy compactness array
#     quantity_all=mgfe,              # raw mgfe aligned to compactness
#     counts_per_bin=counts_per_bin,  # optional; the histogram counts you computed
#     bootstrap_n=0,
#     cbins_edges=cbins
# )

# bs = res_mgfe['bootstrap']
# if bs is not None:
#     thr_raw = bs['raw']
#     methods = bs.get('methods', None)

#     if methods is not None:
#         methods = np.asarray(methods)
#         thr_turn = thr_raw[methods == 'turning_point']
#         det_rate = np.sum(methods == 'turning_point') / methods.size

#         print("detection rate:", det_rate)
#         if thr_turn.size > 0:
#             print("turning-point bootstrap median,p16,p84:",
#                   np.nanmedian(thr_turn),
#                   np.nanpercentile(thr_turn, 16),
#                   np.nanpercentile(thr_turn, 84))
#         else:
#             print("no turning-point detections in bootstrap")
#     else:
#         print("no methods recorded; inspect thr_raw histogram")

# print("Mg/Fe threshold:", res_mgfe['threshold'], "method:", res_mgfe['method'])
# if res_mgfe['bootstrap'] is not None:
#     print("Mg/Fe bootstrap median,16,84:", res_mgfe['bootstrap']['median'],
#           res_mgfe['bootstrap']['p16'], res_mgfe['bootstrap']['p84'])

# # add smoothened curve
# ax_proj.plot(
#     cbin_centers,
#     res_mgfe["cmed_smooth"],
#     linestyle="--",
#     color="C2",
#     lw=2,
#     label="smoothed median"
# )
# ax_proj.legend(fontsize=9)
# # add derivative curve
# ax_der = ax_proj.twinx()
# ax_der.plot(
#     cbin_centers,
#     res_mgfe["derivative"],
#     color="C3",
#     lw=1.4,
#     alpha=0.9,
#     label="|d(median)/d(compactness)|"
# )
# ax_der.axhline(0.0, linestyle=":", color="C3", alpha=0.6)
# ax_der.set_ylabel("Derivative (arb. units)", color="C3")
# ax_der.tick_params(axis="y", labelcolor="C3")

# # merge legends
# lines1, labels1 = ax_proj.get_legend_handles_labels()
# lines2, labels2 = ax_der.get_legend_handles_labels()
# ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
# # ---------- histogram of counts: only show for fixed-width bins ----------
# if not use_quantile_bins:
#     counts_per_bin = np.array([np.sum((compactness >= cbins[i]) & (compactness < cbins[i+1])) for i in range(nbins)])
#     ax2 = ax_proj.twinx()
#     ax2.bar(cbin_centers, counts_per_bin, width=(cbins[1]-cbins[0])*0.9, alpha=0.12, color='gray', edgecolor='none')
#     ax2.set_ylabel("N (per compactness bin)", color='gray')
#     ax2.tick_params(axis='y', labelsize=10)
# else:
#     # hide / don't draw histogram for quantile bins (optionally draw a very faint one or text)
#     pass
# # ax2.set_yticks([])

# # optional: mark threshold on the projection plot
# ax_proj.axvline(res_mgfe['threshold'], color='C2', linestyle='--', lw=1.6, label=f"thr={res_mgfe['threshold']:.2f}")
# ax_proj.legend(fontsize=8)

# fig.tight_layout()
# fig_name = os.path.join(outdir, f"compactness_mgfe_two_panel_z{ztarget:.1f}.png")
# fig.savefig(fig_name, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved two-panel compactness plot to:", fig_name)

# # -------------- two panel plotting (Ex-situ mass fraction) ------------------
# summary_exsitu = thresholds_for_targets(compactness, exsitu_fracs, targets_per_bin=targets,
#                                       min_count=5, bootstrap_n=0, do_bootstrap=False)
# for s in summary_exsitu:
#     thr = s['threshold_result']
#     print(f"exsitu target={s['target_per_bin']} -> nbins={s['nbins']} threshold={thr['threshold']:.3f} method={thr['method']}")

# fig, axes = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={"width_ratios":[1.2,1]})
# ax_main, ax_proj = axes

# print("compactness-vs-exsitu block:")
# print("finite ex-situ:", np.isfinite(exsitu_fracs).sum(), "/", exsitu_fracs.size)
# print("min/max:", np.nanmin(exsitu_fracs), np.nanmax(exsitu_fracs))

# # Left: scatter of compactness vs exsitu (grey) + hexbin median overlay
# # Plot light-grey scatter of all galaxies (including NaN exsitu)
# ax_main.scatter(compactness, exsitu_fracs, s=8, color="lightgrey", alpha=0.8, label="galaxies")

# # Hexbin of median ex-situ in compactness-x bins and exsitu-y bins:
# # We'll use matplotlib's hexbin but compute medians per hex via scipy / numpy aggregation.
# # Simpler and robust: use a 2D bin grid and compute median inside each bin.
# nx = 60  # horizontal bins (compactness)
# ny = 60  # vertical bins (ex-situ)
# # binned grid limits
# cmin, cmax = np.nanpercentile(compactness, [1,99])
# ymin, ymax = np.nanpercentile(exsitu_fracs[~np.isnan(exsitu_fracs)], [1,99]) if np.any(np.isfinite(exsitu_fracs)) else (np.nanmin(exsitu_fracs), np.nanmax(exsitu_fracs))
# # expand a bit
# padc = 0.1 * (cmax - cmin + 1e-6)
# pady = 0.1 * (ymax - ymin + 1e-6)
# xbins = np.linspace(cmin-padc, cmax+padc, nx+1)
# ybins = np.linspace(ymin-pady, ymax+pady, ny+1)

# # indices for each point (only consider finite ex-situ for median calculation)
# finite = np.isfinite(exsitu_fracs)
# xi = np.searchsorted(xbins, compactness[finite]) - 1
# yi = np.searchsorted(ybins, exsitu_fracs[finite]) - 1
# valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
# xi = xi[valid]; yi = yi[valid]
# vals = exsitu_fracs[finite][valid]

# # compute median per bin
# med_grid = np.full((ny, nx), np.nan)
# count_grid = np.zeros((ny, nx), dtype=int)
# for xb, yb, v in zip(xi, yi, vals):
#     count_grid[yb, xb] += 1
#     if np.isnan(med_grid[yb, xb]):
#         med_grid[yb, xb] = v
#     else:
#         # accumulate in list would be simpler, but avoid memory overhead: store temporarily via lists
#         # we instead collect using a dict of lists (small overhead)
#         pass

# # Because we used naive median accumulation above, do it properly with grouping:
# from collections import defaultdict
# group = defaultdict(list)
# for xb, yb, v in zip(xi, yi, vals):
#     group[(yb, xb)].append(v)
# for (yb, xb), lst in group.items():
#     med_grid[yb, xb] = np.nanmedian(lst)
#     count_grid[yb, xb] = len(lst)

# # # Make the meshgrid of bin centres for plotting with pcolormesh
# # xc = 0.5*(xbins[:-1] + xbins[1:])
# # yc = 0.5*(ybins[:-1] + ybins[1:])
# # Xc, Yc = np.meshgrid(xc, yc)

# # # Plot medians on top using pcolormesh (mask empty bins)
# # med_masked = np.ma.masked_invalid(med_grid)
# # cmap = plt.get_cmap("viridis")
# # pcm = ax_main.pcolormesh(xbins, ybins, med_masked, cmap=cmap, shading='auto')
# # cbar = fig.colorbar(pcm, ax=ax_main)
# # cbar.set_label("Ex-situ mass fraction")

# # ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# # ax_main.set_ylabel("Ex-situ mass fraction")
# # ax_main.set_title("Compactness vs Ex-situ mass fraction — 2D median")
# # ax_main.grid(True)

# # --- KDE density contours (robust, with KDTree masking + filled contours) ---
# x = compactness
# y = exsitu_fracs    # for the Mg/Fe panel; replace with other y as needed

# # mask finite points only
# fin = np.isfinite(x) & np.isfinite(y)
# if np.sum(fin) < 10:
#     print("Skipping KDE contours: too few finite points.")
# else:
#     xs = x[fin]; ys = y[fin]

#     pts = np.vstack([xs, ys])          # shape (2, N)
#     kde = gaussian_kde(pts, bw_method='scott')

#     # build evaluation grid exactly like your original code
#     nx_grid = 200
#     ny_grid = 200
#     x_min, x_max = np.nanpercentile(x, [1, 99])
#     y_min, y_max = np.nanpercentile(y, [1, 99])
#     xpad = 0.05 * (x_max - x_min + 1e-9)
#     ypad = 0.05 * (y_max - y_min + 1e-9)
#     xg = np.linspace(x_min - xpad, x_max + xpad, nx_grid)
#     yg = np.linspace(y_min - ypad, y_max + ypad, ny_grid)
#     Xgrid, Ygrid = np.meshgrid(xg, yg)
#     grid_pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T    # (nx*ny, 2)

#     # KDTree mask: mark grid cells far from any data point
#     tree = cKDTree(np.column_stack((xs, ys)))
#     d_grid, _ = tree.query(grid_pts, k=1)

#     # estimate typical spacing from data (like your LOESS approach)
#     d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.median(d_grid))

#     # cutoff: keep grid points within ~1.3 * typical spacing (tweak multiplier if needed)
#     cut = max(typical_spacing * 1.3, 1e-6)
#     mask_far = (d_grid > cut)

#     # evaluate KDE on grid and mask far-away cells
#     Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
#     Z_flat = Z.ravel()
#     Z_flat[mask_far] = np.nan
#     Z = Z_flat.reshape(Xgrid.shape)

#     # pick contour levels from finite Z values
#     finite_vals = Z[np.isfinite(Z)]
#     if finite_vals.size == 0:
#         print("KDE produced no finite values inside masked region; skipping contours.")
#     else:
#         levs = np.percentile(finite_vals, [50, 75, 90, 97])

#         # filled continuous density map (many levels => visually smooth)
#         cf = ax_main.contourf(Xgrid, Ygrid, Z, levels=50, cmap='viridis', antialiased=True)

#         # optional: overlay a few contour lines
#         cs = ax_main.contour(Xgrid, Ygrid, Z, levels=levs, colors='k', linewidths=0.6, alpha=0.5)

#         # colorbar attached to the filled contours
#         fig.colorbar(cf, ax=ax_main, label='Density (KDE)')

# ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_main.set_ylabel("Ex-situ mass fraction")
# # ax_main.set_title("Compactness vs Ex-situ mass fraction — 2D median")
# ax_main.grid(True)

# # Add running-median line (binned along compactness) -- use quantile (equal-count) bins
# use_quantile_bins = True   # True -> equal-count bins; False -> fixed-width bins
# target_per_bin = 1000       # approx galaxies per bin when using quantile bins
# nbins_fixed = 18           # used for fixed-width
# min_count_per_bin = 5

# q_arr = exsitu_fracs  # quantity for this block

# # valid compactness entries (should already be aligned)
# valid_comp = np.isfinite(compactness)
# Ntot = np.sum(valid_comp)
# if Ntot == 0:
#     raise RuntimeError("No valid galaxies to bin in compactness (exsitu).")

# if use_quantile_bins:
#     nbins = max(3, int(np.floor(Ntot / max(1, int(target_per_bin)))))
#     quantiles = np.linspace(0.0, 100.0, nbins + 1)
#     cbins = np.nanpercentile(compactness, quantiles)
#     # enforce strictly increasing edges
#     for i in range(1, cbins.size):
#         if cbins[i] <= cbins[i-1]:
#             cbins[i] = cbins[i-1] + 1e-9
# else:
#     nbins = int(nbins_fixed)
#     cbins = np.linspace(np.nanmin(compactness), np.nanmax(compactness), nbins + 1)

# cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])

# # aggregate into bins
# from collections import defaultdict
# idxs = np.searchsorted(cbins, compactness, side='right') - 1
# valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(q_arr)
# group = defaultdict(list)
# for idx, q in zip(idxs[valid_mask], q_arr[valid_mask]):
#     group[int(idx)].append(float(q))

# cmed = np.full(nbins, np.nan); clow = np.full(nbins, np.nan); chigh = np.full(nbins, np.nan)
# counts_per_bin = np.zeros(nbins, dtype=int)
# for b in range(nbins):
#     vals = group.get(b, [])
#     counts_per_bin[b] = len(vals)
#     if counts_per_bin[b] >= min_count_per_bin:
#         arr = np.asarray(vals, dtype=float)
#         cmed[b] = np.nanmedian(arr)
#         clow[b] = np.nanpercentile(arr, 16)
#         chigh[b] = np.nanpercentile(arr, 84)

# # overlay median+band on the LEFT pcolormesh
# ok = np.isfinite(cmed)
# if np.any(ok):
#     ax_main.plot(cbin_centers[ok], cmed[ok], color="black", lw=2, label="median (binned)")
#     ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15, label="16–84 pct")
#     ax_main.legend(fontsize=8, loc='upper left')

# # RIGHT panel: projection (median vs compactness)
# ax_proj.plot(cbin_centers, cmed, color="C0", lw=2)
# ax_proj.fill_between(cbin_centers, clow, chigh, color="C0", alpha=0.25)
# ax_proj.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_proj.set_ylabel(fr'$f_\text{ex-situ}$')
# # ax_proj.set_title("Projection: median & 16–84 pct")
# ax_proj.grid(True)

# # call threshold finder for exsitu
# res_exsitu = find_compactness_threshold(
#     cbin_centers=cbin_centers,
#     cmed=cmed,
#     compactness_all=compactness,
#     quantity_all=exsitu_fracs,
#     counts_per_bin=counts_per_bin,
#     bootstrap_n=0,
#     cbins_edges=cbins
# )

# # ---- diagnostics to paste after calling find_compactness_threshold ----
# res = res_exsitu
# cbin_centers = np.asarray(cbin_centers)   # ensure it's available in scope
# counts = np.asarray(counts_per_bin)       # likewise

# deriv = np.asarray(res['derivative'])
# absd = np.abs(deriv)
# n = len(cbin_centers)

# idx_max = int(np.nanargmax(absd))
# c_at_max = cbin_centers[idx_max]
# val_max = float(absd[idx_max])

# left_edge_idx = int(np.floor(0.05 * n))   # use your edge_frac if different
# right_edge_idx = int(np.ceil((1 - 0.05) * n)) - 1

# baseline = float(np.nanmedian(absd))
# mad = float(np.nanmedian(np.abs(absd - baseline)))
# # use the same deriv_thresh_factor you call the function with:
# deriv_thresh_factor = 3.0
# deriv_threshold = baseline + deriv_thresh_factor * (1.4826 * mad)

# print("=== turning-point diagnostics ===")
# print("returned method:", res['method'])
# print("idx_max:", idx_max, "center:", c_at_max)
# print("abs_deriv[idx_max]:", val_max)
# print("deriv_threshold:", deriv_threshold, "(baseline, MAD) =", baseline, mad)
# print("edge indices: left", left_edge_idx, "right", right_edge_idx,
#       "-> idx_max inside edges?", (idx_max > left_edge_idx) and (idx_max < right_edge_idx))
# print("counts_per_bin at idx_max:", counts[idx_max] if (idx_max >=0 and idx_max < len(counts)) else "n/a")
# # show top few peaks so you can compare runner-up
# order = np.argsort(absd)[-6:][::-1]
# print("top derivative peaks (index, center, value, counts):")
# for i in order:
#     print(i, f"{cbin_centers[i]:.3f}", f"{absd[i]:.6e}", "counts:", counts[i])
# #-----------------------------------------------------------------------------

# bs = res_exsitu['bootstrap']
# if bs is not None:
#     thr_raw = bs['raw']
#     methods = bs.get('methods', None)

#     if methods is not None:
#         methods = np.asarray(methods)
#         thr_turn = thr_raw[methods == 'turning_point']
#         det_rate = np.sum(methods == 'turning_point') / methods.size

#         print("detection rate:", det_rate)
#         if thr_turn.size > 0:
#             print("turning-point bootstrap median,p16,p84:",
#                   np.nanmedian(thr_turn),
#                   np.nanpercentile(thr_turn, 16),
#                   np.nanpercentile(thr_turn, 84))
#         else:
#             print("no turning-point detections in bootstrap")
#     else:
#         print("no methods recorded; inspect thr_raw histogram")

# print("exsitu threshold:", res_exsitu['threshold'], res_exsitu['method'])
# if res_exsitu['bootstrap'] is not None:
#     print("Exsitu bootstrap median,16,84:", res_exsitu['bootstrap']['median'],
#           res_exsitu['bootstrap']['p16'], res_exsitu['bootstrap']['p84'])

# # add smoothed median + derivative on projection
# if res_exsitu is not None:
#     ax_proj.plot(cbin_centers, res_exsitu["cmed_smooth"], linestyle="--", color="C2", lw=2, label="smoothed median")
#     ax_der = ax_proj.twinx()
#     ax_der.plot(cbin_centers, res_exsitu["derivative"], color="C3", lw=1.4, alpha=0.9, label="|d(median)/d(compactness)|")
#     ax_der.axhline(0.0, linestyle=":", color="C3", alpha=0.6)
#     ax_der.set_ylabel("Derivative (arb. units)", color="C3")
#     ax_der.tick_params(axis="y", labelcolor="C3")
#     # merge legends
#     lines1, labels1 = ax_proj.get_legend_handles_labels()
#     lines2, labels2 = ax_der.get_legend_handles_labels()
#     ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# # optional: histogram only for fixed-width bins
# if not use_quantile_bins:
#     counts_per_bin = np.array([np.sum((compactness >= cbins[i]) & (compactness < cbins[i+1])) for i in range(nbins)])
#     ax2 = ax_proj.twinx()
#     ax2.bar(cbin_centers, counts_per_bin, width=(cbins[1]-cbins[0])*0.9, alpha=0.12, color='gray', edgecolor='none')
#     ax2.set_ylabel("N (per compactness bin)", color='gray')
#     ax2.tick_params(axis='y', labelsize=10)

# # mark threshold on projection
# ax_proj.axvline(res_exsitu['threshold'], color='C2', linestyle='--', lw=1.6, label=f"thr={res_exsitu['threshold']:.2f}")
# ax_proj.legend(fontsize=8)

# fig.tight_layout()
# fig_name = os.path.join(outdir, f"compactness_exsitu_two_panel_z{ztarget:.1f}.png")
# fig.savefig(fig_name, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved two-panel exsitu mass fraction plot to:", fig_name)

# # ---------- Compactness vs Luminosity-weighted stellar age (two-panel) ----------
# # pick the array of ages aligned with the plotted galaxies:
# # stellarage_lum_in is the full selected array; select the mask_positive subset
# ages = np.asarray(stellarage_lum_in)[mask_positive]   # in Gyr (as in your pipeline)

# # print test results
# summary_age = thresholds_for_targets(compactness, ages, targets_per_bin=targets,
#                                       min_count=5, bootstrap_n=0, do_bootstrap=False)
# for s in summary_age:
#     thr = s['threshold_result']
#     print(f"age target={s['target_per_bin']} -> nbins={s['nbins']} threshold={thr['threshold']:.3f} method={thr['method']}")


# fig, axes = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={"width_ratios":[1.2,1]})
# ax_main, ax_proj = axes

# # scatter of all galaxies (grey), including NaNs
# ax_main.scatter(compactness, ages, s=8, color="lightgrey", alpha=0.8, label="galaxies")

# # grid parameters
# nx, ny = 60, 60
# cmin, cmax = np.nanpercentile(compactness, [1,99])
# if np.any(np.isfinite(ages)):
#     ymin, ymax = np.nanpercentile(ages[np.isfinite(ages)], [1,99])
# else:
#     ymin, ymax = np.nanmin(ages), np.nanmax(ages)
# padc = 0.1 * (cmax - cmin + 1e-6); pady = 0.1 * (ymax - ymin + 1e-6)
# xbins = np.linspace(cmin-padc, cmax+padc, nx+1)
# ybins = np.linspace(ymin-pady, ymax+pady, ny+1)

# # bin the finite points
# finite = np.isfinite(ages)
# xi = np.searchsorted(xbins, compactness[finite]) - 1
# yi = np.searchsorted(ybins, ages[finite]) - 1
# valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
# xi = xi[valid]; yi = yi[valid]; vals = ages[finite][valid]

# from collections import defaultdict
# group = defaultdict(list)
# for xb, yb, v in zip(xi, yi, vals):
#     group[(yb, xb)].append(v)

# # med_grid = np.full((ny, nx), np.nan)
# # count_grid = np.zeros((ny, nx), dtype=int)
# # for (yb, xb), lst in group.items():
# #     med_grid[yb, xb] = np.nanmedian(lst)
# #     count_grid[yb, xb] = len(lst)

# # # mask low-count cells to avoid noise
# # min_count = 5
# # med_grid_masked = med_grid.copy()
# # med_grid_masked[count_grid < min_count] = np.nan

# # pcm = ax_main.pcolormesh(xbins, ybins, med_grid_masked, cmap="viridis", shading="auto")
# # if np.any(np.isfinite(med_grid)):
# #     pcm.set_clim(float(np.nanpercentile(med_grid[np.isfinite(med_grid)], 5)),
# #                  float(np.nanpercentile(med_grid[np.isfinite(med_grid)], 95)))
# # cbar = fig.colorbar(pcm, ax=ax_main)
# # cbar.set_label("Lum-weighted age (Gyr)")

# # ax_main.set_xlabel(r"Compactness (lg[$M_\odot$] - 1.5 lg[r/kpc])")
# # ax_main.set_ylabel("Luminosity-weighted mean stellar age (Gyr)")
# # ax_main.set_title("Compactness vs Lum-weighted age — 2D median")
# # ax_main.grid(True)

# # --- KDE density contours (robust, with KDTree masking + filled contours) ---
# x = compactness
# y = ages    # for the Mg/Fe panel; replace with other y as needed

# # mask finite points only
# fin = np.isfinite(x) & np.isfinite(y)
# if np.sum(fin) < 10:
#     print("Skipping KDE contours: too few finite points.")
# else:
#     xs = x[fin]; ys = y[fin]

#     pts = np.vstack([xs, ys])          # shape (2, N)
#     kde = gaussian_kde(pts, bw_method='scott')

#     # build evaluation grid exactly like your original code
#     nx_grid = 200
#     ny_grid = 200
#     x_min, x_max = np.nanpercentile(x, [1, 99])
#     y_min, y_max = np.nanpercentile(y, [1, 99])
#     xpad = 0.05 * (x_max - x_min + 1e-9)
#     ypad = 0.05 * (y_max - y_min + 1e-9)
#     xg = np.linspace(x_min - xpad, x_max + xpad, nx_grid)
#     yg = np.linspace(y_min - ypad, y_max + ypad, ny_grid)
#     Xgrid, Ygrid = np.meshgrid(xg, yg)
#     grid_pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T    # (nx*ny, 2)

#     # KDTree mask: mark grid cells far from any data point
#     tree = cKDTree(np.column_stack((xs, ys)))
#     d_grid, _ = tree.query(grid_pts, k=1)

#     # estimate typical spacing from data (like your LOESS approach)
#     d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.median(d_grid))

#     # cutoff: keep grid points within ~1.3 * typical spacing (tweak multiplier if needed)
#     cut = max(typical_spacing * 1.3, 1e-6)
#     mask_far = (d_grid > cut)

#     # evaluate KDE on grid and mask far-away cells
#     Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
#     Z_flat = Z.ravel()
#     Z_flat[mask_far] = np.nan
#     Z = Z_flat.reshape(Xgrid.shape)

#     # pick contour levels from finite Z values
#     finite_vals = Z[np.isfinite(Z)]
#     if finite_vals.size == 0:
#         print("KDE produced no finite values inside masked region; skipping contours.")
#     else:
#         levs = np.percentile(finite_vals, [50, 75, 90, 97])

#         # filled continuous density map (many levels => visually smooth)
#         cf = ax_main.contourf(Xgrid, Ygrid, Z, levels=50, cmap='viridis', antialiased=True)

#         # optional: overlay a few contour lines
#         cs = ax_main.contour(Xgrid, Ygrid, Z, levels=levs, colors='k', linewidths=0.6, alpha=0.5)

#         # colorbar attached to the filled contours
#         fig.colorbar(cf, ax=ax_main, label='Density (KDE)')

# ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_main.set_ylabel("Lum-weighted age (Gyr)")
# # ax_main.set_title("Compactness vs Ex-situ mass fraction — 2D median")
# ax_main.grid(True)

# # ---------- Running-median / projection (age) ----------
# # Configuration: same as Mg/Fe block
# use_quantile_bins = True   # True -> equal-count bins; False -> fixed-width bins
# target_per_bin = 1000       # approx galaxies per bin when using quantile bins
# nbins_fixed = 18           # used for fixed-width
# min_count_per_bin = 5

# q_arr = ages  # quantity for this block

# # valid compactness entries (should already be aligned)
# valid_comp = np.isfinite(compactness)
# Ntot = np.sum(valid_comp)
# if Ntot == 0:
#     raise RuntimeError("No valid galaxies to bin in compactness (age).")

# if use_quantile_bins:
#     nbins = max(3, int(np.floor(Ntot / max(1, int(target_per_bin)))))
#     quantiles = np.linspace(0.0, 100.0, nbins + 1)
#     cbins = np.nanpercentile(compactness, quantiles)
#     # enforce strictly increasing edges
#     for i in range(1, cbins.size):
#         if cbins[i] <= cbins[i-1]:
#             cbins[i] = cbins[i-1] + 1e-9
# else:
#     nbins = int(nbins_fixed)
#     cbins = np.linspace(np.nanmin(compactness), np.nanmax(compactness), nbins + 1)

# cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])

# # aggregate into bins
# from collections import defaultdict
# idxs = np.searchsorted(cbins, compactness, side='right') - 1
# valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(q_arr)
# group = defaultdict(list)
# for idx, q in zip(idxs[valid_mask], q_arr[valid_mask]):
#     group[int(idx)].append(float(q))

# cmed = np.full(nbins, np.nan); clow = np.full(nbins, np.nan); chigh = np.full(nbins, np.nan)
# counts_per_bin = np.zeros(nbins, dtype=int)
# for b in range(nbins):
#     vals = group.get(b, [])
#     counts_per_bin[b] = len(vals)
#     if counts_per_bin[b] >= min_count_per_bin:
#         arr = np.asarray(vals, dtype=float)
#         cmed[b] = np.nanmedian(arr)
#         clow[b] = np.nanpercentile(arr, 16)
#         chigh[b] = np.nanpercentile(arr, 84)

# # overlay median+band on the LEFT pcolormesh
# ok = np.isfinite(cmed)
# if np.any(ok):
#     ax_main.plot(cbin_centers[ok], cmed[ok], color="black", lw=2, label="median (binned)")
#     ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15, label="16–84 pct")
#     ax_main.legend(fontsize=8, loc='upper left')

# # RIGHT panel: projection (median vs compactness)
# ax_proj.plot(cbin_centers, cmed, color="C0", lw=2)
# ax_proj.fill_between(cbin_centers, clow, chigh, color="C0", alpha=0.25)
# ax_proj.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_proj.set_ylabel("Lum-weighted age (Gyr)")
# # ax_proj.set_title("Projection: median & 16–84 pct")
# ax_proj.grid(True)

# # call threshold finder for lum-weighted age
# res_age = find_compactness_threshold(
#     cbin_centers=cbin_centers,
#     cmed=cmed,
#     compactness_all=compactness,
#     quantity_all=ages,
#     counts_per_bin=counts_per_bin,
#     bootstrap_n=0,
#     cbins_edges=cbins
# )

# bs = res_age['bootstrap']
# if bs is not None:
#     thr_raw = bs['raw']
#     methods = bs.get('methods', None)

#     if methods is not None:
#         methods = np.asarray(methods)
#         thr_turn = thr_raw[methods == 'turning_point']
#         det_rate = np.sum(methods == 'turning_point') / methods.size

#         print("detection rate:", det_rate)
#         if thr_turn.size > 0:
#             print("turning-point bootstrap median,p16,p84:",
#                   np.nanmedian(thr_turn),
#                   np.nanpercentile(thr_turn, 16),
#                   np.nanpercentile(thr_turn, 84))
#         else:
#             print("no turning-point detections in bootstrap")
#     else:
#         print("no methods recorded; inspect thr_raw histogram")

# print("age threshold:", res_age['threshold'], res_age['method'])
# if res_age['bootstrap'] is not None:
#     print("Age bootstrap median,16,84:", res_age['bootstrap']['median'],
#           res_age['bootstrap']['p16'], res_age['bootstrap']['p84'])

# # add smoothed median + derivative on projection
# if res_age is not None:
#     ax_proj.plot(cbin_centers, res_age["cmed_smooth"], linestyle="--", color="C2", lw=2, label="smoothed median")
#     ax_der = ax_proj.twinx()
#     ax_der.plot(cbin_centers, res_age["derivative"], color="C3", lw=1.4, alpha=0.9, label="|d(median)/d(compactness)|")
#     ax_der.axhline(0.0, linestyle=":", color="C3", alpha=0.6)
#     ax_der.set_ylabel("Derivative (arb. units)", color="C3")
#     ax_der.tick_params(axis="y", labelcolor="C3")
#     # merge legends
#     lines1, labels1 = ax_proj.get_legend_handles_labels()
#     lines2, labels2 = ax_der.get_legend_handles_labels()
#     ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# # optional: histogram only for fixed-width bins
# if not use_quantile_bins:
#     counts_per_bin = np.array([np.sum((compactness >= cbins[i]) & (compactness < cbins[i+1])) for i in range(nbins)])
#     ax2 = ax_proj.twinx()
#     ax2.bar(cbin_centers, counts_per_bin, width=(cbins[1]-cbins[0])*0.9, alpha=0.12, color='gray', edgecolor='none')
#     ax2.set_ylabel("N (per compactness bin)", color='gray')
#     ax2.tick_params(axis='y', labelsize=10)

# # mark threshold on projection
# ax_proj.axvline(res_age['threshold'], color='C2', linestyle='--', lw=1.6, label=f"thr={res_age['threshold']:.2f}")
# ax_proj.legend(fontsize=8)

# fig.tight_layout()
# fig_name = os.path.join(outdir, f"compactness_lumage_two_panel_z{ztarget:.1f}.png")
# fig.savefig(fig_name, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved two-panel luminosity-weighted age plot to:", fig_name)

# # ---------- Compactness vs sSFR (specific SFR, log10) ----------

# # compute sSFR
# sfr = np.asarray(sfr_in)[mask_positive]    # Msun/yr
# m   = np.asarray(m_in)[mask_positive]      # Msun
# with np.errstate(divide='ignore', invalid='ignore'):
#     ssfr = np.where(m > 0, sfr / m, np.nan)  # yr^-1

# # Safe, warning-free log10 of sSFR:
# log_ssfr = np.full_like(ssfr, np.nan, dtype=float)
# mask_pos = (ssfr > 0) & np.isfinite(ssfr)
# if np.any(mask_pos):
#     log_ssfr[mask_pos] = np.log10(ssfr[mask_pos])

# # print test results
# summary_ssfr = thresholds_for_targets(compactness, log_ssfr, targets_per_bin=targets,
#                                       min_count=5, bootstrap_n=0, do_bootstrap=False)
# for s in summary_ssfr:
#     thr = s['threshold_result']
#     print(f"ssfr target={s['target_per_bin']} -> nbins={s['nbins']} threshold={thr['threshold']:.3f} method={thr['method']}")

# fig, axes = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={"width_ratios":[1.2,1]})
# ax_main, ax_proj = axes

# ax_main.scatter(compactness, log_ssfr, s=8, color="lightgrey", alpha=0.8, label="galaxies")

# nx, ny = 60, 60
# cmin, cmax = np.nanpercentile(compactness, [1,99])
# if np.any(np.isfinite(log_ssfr)):
#     ymin, ymax = np.nanpercentile(log_ssfr[np.isfinite(log_ssfr)], [1,99])
# else:
#     ymin, ymax = np.nanmin(log_ssfr), np.nanmax(log_ssfr)
# padc = 0.1 * (cmax - cmin + 1e-6); pady = 0.1 * (ymax - ymin + 1e-6)
# xbins = np.linspace(cmin-padc, cmax+padc, nx+1)
# ybins = np.linspace(ymin-pady, ymax+pady, ny+1)

# finite = np.isfinite(log_ssfr)
# xi = np.searchsorted(xbins, compactness[finite]) - 1
# yi = np.searchsorted(ybins, log_ssfr[finite]) - 1
# valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
# xi = xi[valid]; yi = yi[valid]; vals = log_ssfr[finite][valid]

# from collections import defaultdict
# group = defaultdict(list)
# for xb, yb, v in zip(xi, yi, vals):
#     group[(yb, xb)].append(v)

# # med_grid = np.full((ny, nx), np.nan); count_grid = np.zeros((ny, nx), dtype=int)
# # for (yb, xb), lst in group.items():
# #     med_grid[yb, xb] = np.nanmedian(lst); count_grid[yb, xb] = len(lst)

# # min_count = 5
# # med_grid_masked = med_grid.copy(); med_grid_masked[count_grid < min_count] = np.nan

# # pcm = ax_main.pcolormesh(xbins, ybins, med_grid_masked, cmap="viridis", shading="auto")
# # if np.any(np.isfinite(med_grid)):
# #     pcm.set_clim(float(np.nanpercentile(med_grid[np.isfinite(med_grid)], 5)),
# #                  float(np.nanpercentile(med_grid[np.isfinite(med_grid)], 95)))
# # cbar = fig.colorbar(pcm, ax=ax_main)
# # cbar.set_label("log10(sSFR / yr$^{-1}$)")

# # ax_main.set_xlabel(r"Compactness (lg[$M_\odot$] - 1.5 lg[r/kpc])")
# # ax_main.set_ylabel("lg(sSFR / yr$^{-1}$)")
# # ax_main.set_title("Compactness vs sSFR (log) — 2D median")
# # ax_main.grid(True)

# # --- KDE density contours (robust, with KDTree masking + filled contours) ---
# x = compactness
# y = log_ssfr    # for the Mg/Fe panel; replace with other y as needed

# # mask finite points only
# fin = np.isfinite(x) & np.isfinite(y)
# if np.sum(fin) < 10:
#     print("Skipping KDE contours: too few finite points.")
# else:
#     xs = x[fin]; ys = y[fin]

#     pts = np.vstack([xs, ys])          # shape (2, N)
#     kde = gaussian_kde(pts, bw_method='scott')

#     # build evaluation grid exactly like your original code
#     nx_grid = 200
#     ny_grid = 200
#     x_min, x_max = np.nanpercentile(x, [1, 99])
#     y_min, y_max = np.nanpercentile(y, [1, 99])
#     xpad = 0.05 * (x_max - x_min + 1e-9)
#     ypad = 0.05 * (y_max - y_min + 1e-9)
#     xg = np.linspace(x_min - xpad, x_max + xpad, nx_grid)
#     yg = np.linspace(y_min - ypad, y_max + ypad, ny_grid)
#     Xgrid, Ygrid = np.meshgrid(xg, yg)
#     grid_pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T    # (nx*ny, 2)

#     # KDTree mask: mark grid cells far from any data point
#     tree = cKDTree(np.column_stack((xs, ys)))
#     d_grid, _ = tree.query(grid_pts, k=1)

#     # estimate typical spacing from data (like your LOESS approach)
#     d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.median(d_grid))

#     # cutoff: keep grid points within ~1.3 * typical spacing (tweak multiplier if needed)
#     cut = max(typical_spacing * 1.3, 1e-6)
#     mask_far = (d_grid > cut)

#     # evaluate KDE on grid and mask far-away cells
#     Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
#     Z_flat = Z.ravel()
#     Z_flat[mask_far] = np.nan
#     Z = Z_flat.reshape(Xgrid.shape)

#     # pick contour levels from finite Z values
#     finite_vals = Z[np.isfinite(Z)]
#     if finite_vals.size == 0:
#         print("KDE produced no finite values inside masked region; skipping contours.")
#     else:
#         levs = np.percentile(finite_vals, [50, 75, 90, 97])

#         # filled continuous density map (many levels => visually smooth)
#         cf = ax_main.contourf(Xgrid, Ygrid, Z, levels=50, cmap='viridis', antialiased=True)

#         # optional: overlay a few contour lines
#         cs = ax_main.contour(Xgrid, Ygrid, Z, levels=levs, colors='k', linewidths=0.6, alpha=0.5)

#         # colorbar attached to the filled contours
#         fig.colorbar(cf, ax=ax_main, label='Density (KDE)')

# ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_main.set_ylabel("lg(sSFR / yr$^{-1}$)")
# ax_main.grid(True)

# # ---------- Running-median / projection (sSFR) ----------
# use_quantile_bins = True
# target_per_bin = 1000
# nbins_fixed = 18
# min_count_per_bin = 5

# q_arr = log_ssfr  # IMPORTANT: use log_ssfr here

# valid_comp = np.isfinite(compactness)
# Ntot = np.sum(valid_comp)
# if Ntot == 0:
#     raise RuntimeError("No valid galaxies to bin in compactness (sSFR).")

# if use_quantile_bins:
#     nbins = max(3, int(np.floor(Ntot / max(1, int(target_per_bin)))))
#     quantiles = np.linspace(0.0, 100.0, nbins + 1)
#     cbins = np.nanpercentile(compactness, quantiles)
#     for i in range(1, cbins.size):
#         if cbins[i] <= cbins[i-1]:
#             cbins[i] = cbins[i-1] + 1e-9
# else:
#     nbins = int(nbins_fixed)
#     cbins = np.linspace(np.nanmin(compactness), np.nanmax(compactness), nbins + 1)

# cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])

# from collections import defaultdict
# idxs = np.searchsorted(cbins, compactness, side='right') - 1
# valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(q_arr)
# group = defaultdict(list)
# for idx, q in zip(idxs[valid_mask], q_arr[valid_mask]):
#     group[int(idx)].append(float(q))

# cmed = np.full(nbins, np.nan); clow = np.full(nbins, np.nan); chigh = np.full(nbins, np.nan)
# counts_per_bin = np.zeros(nbins, dtype=int)
# for b in range(nbins):
#     vals = group.get(b, [])
#     counts_per_bin[b] = len(vals)
#     if counts_per_bin[b] >= min_count_per_bin:
#         arr = np.asarray(vals, dtype=float)
#         cmed[b] = np.nanmedian(arr)
#         clow[b] = np.nanpercentile(arr, 16)
#         chigh[b] = np.nanpercentile(arr, 84)

# # left overlay
# ok = np.isfinite(cmed)
# if np.any(ok):
#     ax_main.plot(cbin_centers[ok], cmed[ok], color="black", lw=2, label="median (binned)")
#     ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15, label="16–84 pct")
#     ax_main.legend(fontsize=8, loc='upper left')

# # right projection
# ax_proj.plot(cbin_centers, cmed, color="C0", lw=2)
# ax_proj.fill_between(cbin_centers, clow, chigh, color="C0", alpha=0.25)
# ax_proj.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_proj.set_ylabel("lg(sSFR / yr$^{-1}$)")
# # ax_proj.set_title("Projection: median & 16–84 pct")
# ax_proj.grid(True)

# # call threshold finder (use log_ssfr as quantity_all)
# res_ssfr = find_compactness_threshold(
#     cbin_centers=cbin_centers,
#     cmed=cmed,
#     compactness_all=compactness,
#     quantity_all=log_ssfr,    
#     counts_per_bin=counts_per_bin,
#     bootstrap_n=0,
#     cbins_edges=cbins
# )

# # # determine second peak in derivative
# # deriv = np.asarray(res_ssfr["derivative"])   # smoothed derivative (can be +/−)
# # abs_deriv = np.abs(deriv)

# # # find peaks in |derivative|
# # # height and distance params are gentle defaults; tweak if necessary
# # peaks, props = find_peaks(abs_deriv, distance=2)  # require at least 2-bin separation
# # if peaks.size == 0:
# #     print("No local derivative peaks found.")
# #     c1 = c2 = np.nan
# # else:
# #     # sort peaks by peak height descending
# #     order = np.argsort(props.get("peak_heights", abs_deriv[peaks]))[::-1]
# #     sorted_peaks = peaks[order]

# #     # choose top two distinct peaks (if available)
# #     if sorted_peaks.size >= 2:
# #         p1 = sorted_peaks[0]
# #         # find next peak that is not the same bin (and not adjacent if you want stricter separation)
# #         p2 = None
# #         for pk in sorted_peaks[1:]:
# #             if abs(pk - p1) >= 1:   # require at least one-bin separation; increase to 2 if desired
# #                 p2 = pk
# #                 break
# #         if p2 is None:
# #             # only one distinct peak
# #             p2 = sorted_peaks[0]
# #     else:
# #         p1 = sorted_peaks[0]
# #         p2 = sorted_peaks[0]

# #     c1 = float(cbin_centers[p1])
# #     c2 = float(cbin_centers[p2])

# # # compute sigma_amb from runner-up separation
# # if np.isfinite(c1) and np.isfinite(c2):
# #     sigma_amb = abs(c2 - c1) / 2.0
# # else:
# #     sigma_amb = np.nan

# # # bin-resolution floor: conservative median half-bin width
# # dc = np.diff(cbin_centers)
# # if dc.size > 0:
# #     sigma_bin = 0.5 * float(np.median(dc))
# # else:
# #     sigma_bin = 0.01  # fallback small value

# # # bootstrap uncertainty: convert p16/p84 to ~1-sigma (approx)
# # bs = res_ssfr.get("bootstrap", None)
# # if bs is None:
# #     sigma_boot = np.nan
# # else:
# #     # prefer p16/p84 if available; else infer from raw
# #     p16 = bs.get("p16", None)
# #     p84 = bs.get("p84", None)
# #     if (p16 is not None) and (p84 is not None) and np.isfinite(p16) and np.isfinite(p84):
# #         sigma_boot = 0.5 * (p84 - p16)   # approx 1-sigma
# #     else:
# #         # fallback: use std of raw bootstrap if present
# #         raw = np.asarray(bs.get("raw", []))
# #         sigma_boot = float(np.nanstd(raw)) if raw.size>0 else np.nan

# # # total combined uncertainty (quadrature)
# # # choose which systematic floors to include; here we include both amb and bin floors when present
# # parts = []
# # if np.isfinite(sigma_boot): parts.append(sigma_boot)
# # if np.isfinite(sigma_amb):  parts.append(sigma_amb)
# # if np.isfinite(sigma_bin):  parts.append(sigma_bin)
# # if len(parts) == 0:
# #     sigma_total = np.nan
# # else:
# #     sigma_total = float(np.sqrt(np.sum(np.array(parts)**2)))

# # # choose final reported threshold: use res_ssfr['threshold'] (algorithm pick) and round conservatively
# # thr = float(res_ssfr['threshold'])
# # # round to 0.02–0.05 dex depending on size of sigma_total
# # if np.isfinite(sigma_total):
# #     # pick a sensible rounding: 2 significant digits in the uncertainty
# #     # e.g. sigma_total=0.123 -> round_unc=0.02 ; sigma_total=0.012 -> 0.01
# #     # compute 1st two significant digits
# #     order = int(np.floor(np.log10(sigma_total))) if sigma_total>0 else 0
# #     round_digits = max(0, -order + 1)  # conservative rounding
# #     thr_str = f"{thr:.{round_digits}f}"
# #     sig_str = f"{sigma_total:.{round_digits}f}"
# # else:
# #     thr_str = f"{thr:.2f}"
# #     sig_str = "nan"

# # # Print diagnostics
# # print("=== threshold ambiguity diagnostics ===")
# # print(f"chosen threshold (algorithm): {thr:.6f}")
# # print(f"top peak c1 (bin idx {locals().get('p1',None)}): {c1}")
# # print(f"runner-up peak c2 (bin idx {locals().get('p2',None)}): {c2}")
# # print(f"sigma_boot (approx 1σ) = {sigma_boot}")
# # print(f"sigma_amb (half separation) = {sigma_amb}")
# # print(f"sigma_bin (half median bin width) = {sigma_bin}")
# # print(f"combined sigma_total (quadrature) = {sigma_total}")
# # print(f"report: threshold = {thr_str} ± {sig_str} (compactness units)")

# # # Optionally mark the runner-up on the plot
# # try:
# #     ax_proj.axvline(c2, color='C4', linestyle=':', lw=1.2, label=f"runner-up {c2:.2f}")
# #     ax_proj.legend(fontsize=8)
# # except Exception:
# #     pass

# bs = res_ssfr['bootstrap']
# if bs is not None:
#     thr_raw = bs['raw']
#     methods = bs.get('methods', None)

#     if methods is not None:
#         methods = np.asarray(methods)
#         thr_turn = thr_raw[methods == 'turning_point']
#         det_rate = np.sum(methods == 'turning_point') / methods.size

#         print("detection rate:", det_rate)
#         if thr_turn.size > 0:
#             print("turning-point bootstrap median,p16,p84:",
#                   np.nanmedian(thr_turn),
#                   np.nanpercentile(thr_turn, 16),
#                   np.nanpercentile(thr_turn, 84))
#         else:
#             print("no turning-point detections in bootstrap")
#     else:
#         print("no methods recorded; inspect thr_raw histogram")


# print("sSFR threshold:", res_ssfr['threshold'], res_ssfr['method'])
# if res_ssfr['bootstrap'] is not None:
#     print("sSFR bootstrap median,16,84:", res_ssfr['bootstrap']['median'],
#           res_ssfr['bootstrap']['p16'], res_ssfr['bootstrap']['p84'])

# # smoothed median + derivative
# if res_ssfr is not None:
#     ax_proj.plot(cbin_centers, res_ssfr["cmed_smooth"], linestyle="--", color="C2", lw=2, label="smoothed median")
#     ax_der = ax_proj.twinx()
#     ax_der.plot(cbin_centers, res_ssfr["derivative"], color="C3", lw=1.4, alpha=0.9, label="|d(median)/d(compactness)|")
#     ax_der.axhline(0.0, linestyle=":", color="C3", alpha=0.6)
#     ax_der.set_ylabel("Derivative (arb. units)", color="C3")
#     ax_der.tick_params(axis="y", labelcolor="C3")
#     lines1, labels1 = ax_proj.get_legend_handles_labels()
#     lines2, labels2 = ax_der.get_legend_handles_labels()
#     ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# # optional histogram for fixed-width bins only
# if not use_quantile_bins:
#     counts_per_bin = np.array([np.sum((compactness >= cbins[i]) & (compactness < cbins[i+1])) for i in range(nbins)])
#     ax2 = ax_proj.twinx()
#     ax2.bar(cbin_centers, counts_per_bin, width=(cbins[1]-cbins[0])*0.9, alpha=0.12, color='gray', edgecolor='none')
#     ax2.set_ylabel("N (per compactness bin)", color='gray')
#     ax2.tick_params(axis='y', labelsize=10)

# # mark threshold
# ax_proj.axvline(res_ssfr['threshold'], color='C2', linestyle='--', lw=1.6, label=f"thr={res_ssfr['threshold']:.2f}")
# ax_proj.legend(fontsize=8)

# fig.tight_layout()
# fig_name = os.path.join(outdir, f"compactness_ssfr_two_panel_z{ztarget:.1f}.png")
# fig.savefig(fig_name, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved two-panel sSFR plot to:", fig_name)


# # ------------------ combine thresholds (minimal snippet) ------------------

# # def _get_boot_samples(res, n_draw=10000, rng=None):
# #     """Return n_draw samples for the threshold from res['bootstrap'] or fallback to approx normal."""
# #     if rng is None:
# #         rng = np.random.default_rng(12345)
# #     if res is None:
# #         return None
# #     bs = res.get("bootstrap", None)
# #     if bs is None:
# #         # fallback to single threshold value (no uncertainty)
# #         thr = res.get("threshold", np.nan)
# #         return np.full(n_draw, float(thr))
# #     raw = bs.get("raw", None)
# #     if raw is not None and len(raw) > 0:
# #         # sample with replacement from the raw bootstrap results
# #         return rng.choice(raw, size=n_draw, replace=True)
# #     # else try median/p16/p84 to build a normal approx
# #     med = bs.get("median", None)
# #     p16 = bs.get("p16", None)
# #     p84 = bs.get("p84", None)
# #     if med is None or p16 is None or p84 is None:
# #         # last-resort: use the deterministic threshold
# #         thr = res.get("threshold", np.nan)
# #         return np.full(n_draw, float(thr))
# #     # approximate sigma from (p84 - p16) ~ 2 sigma (for near-normal shape)
# #     sigma = max(1e-6, (p84 - p16) / 2.0)
# #     return rng.normal(loc=float(med), scale=float(sigma), size=n_draw)

# # # choose number of combined draws (10k is fine and cheap)
# # NCOMB = 10000
# # rng = np.random.default_rng(123456)

# # # extract bootstrap samples for each diagnostic
# # bs_mgfe = _get_boot_samples(res_mgfe, n_draw=NCOMB, rng=rng)
# # bs_age  = _get_boot_samples(res_age,  n_draw=NCOMB, rng=rng)
# # bs_ssfr = _get_boot_samples(res_ssfr, n_draw=NCOMB, rng=rng)

# # # ensure shapes
# # assert bs_mgfe.shape[0] == NCOMB and bs_age.shape[0] == NCOMB and bs_ssfr.shape[0] == NCOMB

# # # compute per-draw mean threshold over the three diagnostics
# # combined_draws = np.vstack([bs_mgfe, bs_age, bs_ssfr]).T  # shape (NCOMB, 3)
# # combined_mean_per_draw = np.nanmean(combined_draws, axis=1)  # ignore NaNs if any

# # # summarise combined distribution robustly (median and 16/84 percentiles)
# # combined_median = float(np.nanmedian(combined_mean_per_draw))
# # combined_p16 = float(np.nanpercentile(combined_mean_per_draw, 16))
# # combined_p84 = float(np.nanpercentile(combined_mean_per_draw, 84))
# # combined_mean = float(np.nanmean(combined_mean_per_draw))
# # combined_std = float(np.nanstd(combined_mean_per_draw, ddof=1))

# # print("\n--- Combined threshold summary (three diagnostics: Mg/Fe, age, sSFR) ---")
# # print(f"Combined (median of per-draw means) = {combined_median:.3f}")
# # print(f"16/84 percentiles = {combined_p16:.3f} / {combined_p84:.3f}")
# # print(f"Mean +/- std = {combined_mean:.3f} +/- {combined_std:.3f}")
# # print(f"Recommended reporting (median + p16/p84): {combined_median:.3f} (+{combined_p84-combined_median:.3f}/ -{combined_median-combined_p16:.3f})")

# # ---------- Compactness vs stellar metallicity (two-panel) ----------
# # pick the array of metallicity aligned with the plotted galaxies
# metallicity = np.asarray(logZstar_rel_in)[mask_positive]   

# mask_phys = (
#     np.isfinite(compactness) &
#     np.isfinite(metallicity) &
#     np.isfinite(log_r) &
#     (log_r > 0.0)   # adjust this threshold as needed
# )

# compactness_sel = compactness[mask_phys]
# metallicity_sel = metallicity[mask_phys]
# print("number of remaining galaxies", len(metallicity[mask_phys]))
# # print test results
# summary_metallicity = thresholds_for_targets(compactness_sel, metallicity_sel, targets_per_bin=targets,
#                                       min_count=5, bootstrap_n=0, do_bootstrap=False)
# for s in summary_metallicity:
#     thr = s['threshold_result']
#     print(f"metallicity target={s['target_per_bin']} -> nbins={s['nbins']} threshold={thr['threshold']:.3f} method={thr['method']}")


# fig, axes = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={"width_ratios":[1.2,1]})
# ax_main, ax_proj = axes

# # scatter of all galaxies (grey), including NaNs
# ax_main.scatter(compactness_sel, metallicity_sel, s=8, color="lightgrey", alpha=0.8, label="galaxies")

# # grid parameters
# nx, ny = 60, 60
# cmin, cmax = np.nanpercentile(compactness_sel, [1,99])
# if np.any(np.isfinite(metallicity_sel)):
#     ymin, ymax = np.nanpercentile(metallicity_sel[np.isfinite(metallicity_sel)], [1,99])
# else:
#     ymin, ymax = np.nanmin(metallicity_sel), np.nanmax(metallicity_sel)
# padc = 0.1 * (cmax - cmin + 1e-6); pady = 0.1 * (ymax - ymin + 1e-6)
# xbins = np.linspace(cmin-padc, cmax+padc, nx+1)
# ybins = np.linspace(ymin-pady, ymax+pady, ny+1)

# # bin the finite points
# finite = np.isfinite(metallicity_sel)
# xi = np.searchsorted(xbins, compactness_sel[finite]) - 1
# yi = np.searchsorted(ybins, metallicity_sel[finite]) - 1
# valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
# xi = xi[valid]; yi = yi[valid]; vals = metallicity_sel[finite][valid]

# from collections import defaultdict
# group = defaultdict(list)
# for xb, yb, v in zip(xi, yi, vals):
#     group[(yb, xb)].append(v)

# # --- KDE density contours (robust, with KDTree masking + filled contours) ---
# x = compactness_sel
# y = metallicity_sel    # for the Mg/Fe panel; replace with other y as needed

# # mask finite points only
# fin = np.isfinite(x) & np.isfinite(y)
# if np.sum(fin) < 10:
#     print("Skipping KDE contours: too few finite points.")
# else:
#     xs = x[fin]; ys = y[fin]

#     pts = np.vstack([xs, ys])          # shape (2, N)
#     kde = gaussian_kde(pts, bw_method='scott')

#     # build evaluation grid exactly like your original code
#     nx_grid = 200
#     ny_grid = 200
#     x_min, x_max = np.nanpercentile(x, [1, 99])
#     y_min, y_max = np.nanpercentile(y, [1, 99])
#     xpad = 0.05 * (x_max - x_min + 1e-9)
#     ypad = 0.05 * (y_max - y_min + 1e-9)
#     xg = np.linspace(x_min - xpad, x_max + xpad, nx_grid)
#     yg = np.linspace(y_min - ypad, y_max + ypad, ny_grid)
#     Xgrid, Ygrid = np.meshgrid(xg, yg)
#     grid_pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T    # (nx*ny, 2)

#     # KDTree mask: mark grid cells far from any data point
#     tree = cKDTree(np.column_stack((xs, ys)))
#     d_grid, _ = tree.query(grid_pts, k=1)

#     # estimate typical spacing from data (like your LOESS approach)
#     d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.median(d_grid))

#     # cutoff: keep grid points within ~1.3 * typical spacing (tweak multiplier if needed)
#     cut = max(typical_spacing * 1.3, 1e-6)
#     mask_far = (d_grid > cut)

#     # evaluate KDE on grid and mask far-away cells
#     Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
#     Z_flat = Z.ravel()
#     Z_flat[mask_far] = np.nan
#     Z = Z_flat.reshape(Xgrid.shape)

#     # pick contour levels from finite Z values
#     finite_vals = Z[np.isfinite(Z)]
#     if finite_vals.size == 0:
#         print("KDE produced no finite values inside masked region; skipping contours.")
#     else:
#         levs = np.percentile(finite_vals, [50, 75, 90, 97])

#         # filled continuous density map (many levels => visually smooth)
#         cf = ax_main.contourf(Xgrid, Ygrid, Z, levels=50, cmap='viridis', antialiased=True)

#         # optional: overlay a few contour lines
#         cs = ax_main.contour(Xgrid, Ygrid, Z, levels=levs, colors='k', linewidths=0.6, alpha=0.5)

#         # colorbar attached to the filled contours
#         fig.colorbar(cf, ax=ax_main, label='Density (KDE)')

# ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_main.set_ylabel(r"$\lg[Z / H]$")
# # ax_main.set_title("Compactness vs Ex-situ mass fraction — 2D median")
# ax_main.grid(True)

# # ---------- Running-median / projection (metallicity) ----------
# # Configuration: same as Mg/Fe block
# use_quantile_bins = True   # True -> equal-count bins; False -> fixed-width bins
# target_per_bin = 1000       # approx galaxies per bin when using quantile bins
# nbins_fixed = 18           # used for fixed-width
# min_count_per_bin = 5

# q_arr = metallicity_sel  # quantity for this block

# # valid compactness entries (should already be aligned)
# valid_comp = np.isfinite(compactness_sel)
# Ntot = np.sum(valid_comp)
# if Ntot == 0:
#     raise RuntimeError("No valid galaxies to bin in compactness (metallicity).")

# if use_quantile_bins:
#     nbins = max(3, int(np.floor(Ntot / max(1, int(target_per_bin)))))
#     quantiles = np.linspace(0.0, 100.0, nbins + 1)
#     cbins = np.nanpercentile(compactness, quantiles)
#     # enforce strictly increasing edges
#     for i in range(1, cbins.size):
#         if cbins[i] <= cbins[i-1]:
#             cbins[i] = cbins[i-1] + 1e-9
# else:
#     nbins = int(nbins_fixed)
#     cbins = np.linspace(np.nanmin(compactness_sel), np.nanmax(compactness_sel), nbins + 1)

# cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])

# # aggregate into bins
# from collections import defaultdict
# idxs = np.searchsorted(cbins, compactness_sel, side='right') - 1
# valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(q_arr)
# group = defaultdict(list)
# for idx, q in zip(idxs[valid_mask], q_arr[valid_mask]):
#     group[int(idx)].append(float(q))

# cmed = np.full(nbins, np.nan); clow = np.full(nbins, np.nan); chigh = np.full(nbins, np.nan)
# counts_per_bin = np.zeros(nbins, dtype=int)
# for b in range(nbins):
#     vals = group.get(b, [])
#     counts_per_bin[b] = len(vals)
#     if counts_per_bin[b] >= min_count_per_bin:
#         arr = np.asarray(vals, dtype=float)
#         cmed[b] = np.nanmedian(arr)
#         clow[b] = np.nanpercentile(arr, 16)
#         chigh[b] = np.nanpercentile(arr, 84)

# # overlay median+band on the LEFT pcolormesh
# ok = np.isfinite(cmed)
# if np.any(ok):
#     ax_main.plot(cbin_centers[ok], cmed[ok], color="black", lw=2, label="median (binned)")
#     ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15, label="16–84 pct")
#     ax_main.legend(fontsize=8, loc='upper left')

# # RIGHT panel: projection (median vs compactness)
# ax_proj.plot(cbin_centers, cmed, color="C0", lw=2)
# ax_proj.fill_between(cbin_centers, clow, chigh, color="C0", alpha=0.25)
# ax_proj.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_proj.set_ylabel(r"$\lg[Z / H]$")
# # ax_proj.set_title("Projection: median & 16–84 pct")
# ax_proj.grid(True)

# # --------------------------------------------------
# # restrict threshold search to a compactness window
# # --------------------------------------------------
# search_lo = 6.5
# search_hi = np.inf   # or set e.g. 10.2 if you want a finite window

# search_mask = np.isfinite(cbin_centers) & (cbin_centers >= search_lo) & (cbin_centers <= search_hi)
# if np.count_nonzero(search_mask) < 3:
#     raise RuntimeError("Search window too narrow: not enough compactness bins for threshold finding.")

# # slice the binned curve consistently
# i0 = np.where(search_mask)[0][0]
# i1 = np.where(search_mask)[0][-1] + 1   # +1 because edges have one extra element

# cbin_centers_search = cbin_centers[i0:i1]
# cmed_search = cmed[i0:i1]
# counts_search = counts_per_bin[i0:i1]
# cbins_search = cbins[i0:i1 + 1]

# # slice the raw data consistently as well
# raw_search_mask = np.isfinite(compactness_sel) & np.isfinite(metallicity_sel) & (compactness_sel >= search_lo) & (compactness_sel <= search_hi)
# compactness_search = compactness_sel[raw_search_mask]
# metallicity_search = metallicity_sel[raw_search_mask]

# # call threshold finder for metallicity
# # Here we use the minimum of the smoothed derivative inside the restricted window

# # use the SAME pipeline as before
# res_metallicity = find_compactness_threshold(
#     cbin_centers=cbin_centers_search,
#     cmed=cmed_search,
#     compactness_all=compactness_search,
#     quantity_all=metallicity_search,
#     counts_per_bin=counts_search,
#     bootstrap_n=0,
#     cbins_edges=cbins_search
# )

# # now OVERRIDE only the decision step
# cmed_smooth = res_metallicity["cmed_smooth"]
# derivative  = res_metallicity["derivative"]

# finite_der = np.isfinite(derivative)
# if not np.any(finite_der):
#     raise RuntimeError("No finite derivative values found.")

# i_min = np.nanargmin(np.where(finite_der, derivative, np.nan))
# threshold = float(cbin_centers_search[i_min])

# # overwrite result but keep everything else
# res_metallicity["threshold"] = threshold
# res_metallicity["method"] = "minimum_derivative"

# # # call threshold finder only on the restricted range
# # res_metallicity = find_compactness_threshold(
# #     cbin_centers=cbin_centers_search,
# #     cmed=cmed_search,
# #     compactness_all=compactness_search,
# #     quantity_all=metallicity_search,
# #     counts_per_bin=counts_search,
# #     bootstrap_n=0,
# #     cbins_edges=cbins_search
# # )

# # diagnostics
# print("metallicity threshold:", res_metallicity["threshold"], res_metallicity["method"])

# bs = res_metallicity['bootstrap']
# if bs is not None:
#     thr_raw = bs['raw']
#     methods = bs.get('methods', None)

#     if methods is not None:
#         methods = np.asarray(methods)
#         thr_turn = thr_raw[methods == 'turning_point']
#         det_rate = np.sum(methods == 'turning_point') / methods.size

#         print("detection rate:", det_rate)
#         if thr_turn.size > 0:
#             print("turning-point bootstrap median,p16,p84:",
#                   np.nanmedian(thr_turn),
#                   np.nanpercentile(thr_turn, 16),
#                   np.nanpercentile(thr_turn, 84))
#         else:
#             print("no turning-point detections in bootstrap")
#     else:
#         print("no methods recorded; inspect thr_raw histogram")

# print("metallicity threshold:", res_metallicity['threshold'], res_metallicity['method'])
# if res_metallicity['bootstrap'] is not None:
#     print("Metallicity bootstrap median,16,84:", res_metallicity['bootstrap']['median'],
#           res_metallicity['bootstrap']['p16'], res_metallicity['bootstrap']['p84'])

# # # add smoothed median + derivative on projection
# # if res_metallicity is not None:
# #     ax_proj.plot(cbin_centers, res_metallicity["cmed_smooth"], linestyle="--", color="C2", lw=2, label="smoothed median")
# #     ax_der = ax_proj.twinx()
# #     ax_der.plot(cbin_centers, res_metallicity["derivative"], color="C3", lw=1.4, alpha=0.9, label="|d(median)/d(compactness)|")
# #     ax_der.axhline(0.0, linestyle=":", color="C3", alpha=0.6)
# #     ax_der.set_ylabel("Derivative (arb. units)", color="C3")
# #     ax_der.tick_params(axis="y", labelcolor="C3")
# #     # merge legends
# #     lines1, labels1 = ax_proj.get_legend_handles_labels()
# #     lines2, labels2 = ax_der.get_legend_handles_labels()
# #     ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# # # optional: histogram only for fixed-width bins
# # if not use_quantile_bins:
# #     counts_per_bin = np.array([np.sum((compactness >= cbins[i]) & (compactness < cbins[i+1])) for i in range(nbins)])
# #     ax2 = ax_proj.twinx()
# #     ax2.bar(cbin_centers, counts_per_bin, width=(cbins[1]-cbins[0])*0.9, alpha=0.12, color='gray', edgecolor='none')
# #     ax2.set_ylabel("N (per compactness bin)", color='gray')
# #     ax2.tick_params(axis='y', labelsize=10)

# # # mark threshold on projection
# # ax_proj.axvline(res_metallicity['threshold'], color='C2', linestyle='--', lw=1.6, label=f"thr={res_metallicity['threshold']:.2f}")
# # use the restricted x-array for anything returned by the threshold finder
# x_smooth = cbin_centers_search

# # add smoothed median + derivative on projection
# if res_metallicity is not None:
#     ax_proj.plot(x_smooth, res_metallicity["cmed_smooth"],
#                  linestyle="--", color="C2", lw=2, label="smoothed median")
#     ax_der = ax_proj.twinx()
#     ax_der.plot(x_smooth, res_metallicity["derivative"],
#                 color="C3", lw=1.4, alpha=0.9, label="|d(median)/d(compactness)|")
#     ax_der.axhline(0.0, linestyle=":", color="C3", alpha=0.6)
#     ax_der.set_ylabel("Derivative (arb. units)", color="C3")
#     ax_der.tick_params(axis="y", labelcolor="C3")

#     lines1, labels1 = ax_proj.get_legend_handles_labels()
#     lines2, labels2 = ax_der.get_legend_handles_labels()
#     ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# # mark threshold on projection
# ax_proj.axvline(res_metallicity["threshold"], color="C2", linestyle="--",
#                 lw=1.6, label=f"thr={res_metallicity['threshold']:.2f}")
# ax_proj.legend(fontsize=8)

# fig.tight_layout()
# fig_name = os.path.join(outdir, f"compactness_metallicity_two_panel_z{ztarget:.1f}.png")
# fig.savefig(fig_name, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved two-panel metallicity plot to:", fig_name)

# # ---------- Compactness vs Mass-weighted stellar age (two-panel) ----------
# # pick the array of ages aligned with the plotted galaxies:
# # stellarage_in is the full selected array; select the mask_positive subset
# ages = np.asarray(stellarage_in)[mask_positive]   # in Gyr (as in your pipeline)

# # print test results
# summary_age = thresholds_for_targets(compactness, ages, targets_per_bin=targets,
#                                       min_count=5, bootstrap_n=0, do_bootstrap=False)
# for s in summary_age:
#     thr = s['threshold_result']
#     print(f"age target={s['target_per_bin']} -> nbins={s['nbins']} threshold={thr['threshold']:.3f} method={thr['method']}")


# fig, axes = plt.subplots(1, 2, figsize=(12,5), gridspec_kw={"width_ratios":[1.2,1]})
# ax_main, ax_proj = axes

# # scatter of all galaxies (grey), including NaNs
# ax_main.scatter(compactness, ages, s=8, color="lightgrey", alpha=0.8, label="galaxies")

# # grid parameters
# nx, ny = 60, 60
# cmin, cmax = np.nanpercentile(compactness, [1,99])
# if np.any(np.isfinite(ages)):
#     ymin, ymax = np.nanpercentile(ages[np.isfinite(ages)], [1,99])
# else:
#     ymin, ymax = np.nanmin(ages), np.nanmax(ages)
# padc = 0.1 * (cmax - cmin + 1e-6); pady = 0.1 * (ymax - ymin + 1e-6)
# xbins = np.linspace(cmin-padc, cmax+padc, nx+1)
# ybins = np.linspace(ymin-pady, ymax+pady, ny+1)

# # bin the finite points
# finite = np.isfinite(ages)
# xi = np.searchsorted(xbins, compactness[finite]) - 1
# yi = np.searchsorted(ybins, ages[finite]) - 1
# valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
# xi = xi[valid]; yi = yi[valid]; vals = ages[finite][valid]

# from collections import defaultdict
# group = defaultdict(list)
# for xb, yb, v in zip(xi, yi, vals):
#     group[(yb, xb)].append(v)

# # med_grid = np.full((ny, nx), np.nan)
# # count_grid = np.zeros((ny, nx), dtype=int)
# # for (yb, xb), lst in group.items():
# #     med_grid[yb, xb] = np.nanmedian(lst)
# #     count_grid[yb, xb] = len(lst)

# # # mask low-count cells to avoid noise
# # min_count = 5
# # med_grid_masked = med_grid.copy()
# # med_grid_masked[count_grid < min_count] = np.nan

# # pcm = ax_main.pcolormesh(xbins, ybins, med_grid_masked, cmap="viridis", shading="auto")
# # if np.any(np.isfinite(med_grid)):
# #     pcm.set_clim(float(np.nanpercentile(med_grid[np.isfinite(med_grid)], 5)),
# #                  float(np.nanpercentile(med_grid[np.isfinite(med_grid)], 95)))
# # cbar = fig.colorbar(pcm, ax=ax_main)
# # cbar.set_label("Lum-weighted age (Gyr)")

# # ax_main.set_xlabel(r"Compactness (lg[$M_\odot$] - 1.5 lg[r/kpc])")
# # ax_main.set_ylabel("Age (Gyr)")
# # ax_main.set_title("Compactness vs Lum-weighted age — 2D median")
# # ax_main.grid(True)

# # --- KDE density contours (robust, with KDTree masking + filled contours) ---
# x = compactness
# y = ages    # for the Mg/Fe panel; replace with other y as needed

# # mask finite points only
# fin = np.isfinite(x) & np.isfinite(y)
# if np.sum(fin) < 10:
#     print("Skipping KDE contours: too few finite points.")
# else:
#     xs = x[fin]; ys = y[fin]

#     pts = np.vstack([xs, ys])          # shape (2, N)
#     kde = gaussian_kde(pts, bw_method='scott')

#     # build evaluation grid exactly like your original code
#     nx_grid = 200
#     ny_grid = 200
#     x_min, x_max = np.nanpercentile(x, [1, 99])
#     y_min, y_max = np.nanpercentile(y, [1, 99])
#     xpad = 0.05 * (x_max - x_min + 1e-9)
#     ypad = 0.05 * (y_max - y_min + 1e-9)
#     xg = np.linspace(x_min - xpad, x_max + xpad, nx_grid)
#     yg = np.linspace(y_min - ypad, y_max + ypad, ny_grid)
#     Xgrid, Ygrid = np.meshgrid(xg, yg)
#     grid_pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T    # (nx*ny, 2)

#     # KDTree mask: mark grid cells far from any data point
#     tree = cKDTree(np.column_stack((xs, ys)))
#     d_grid, _ = tree.query(grid_pts, k=1)

#     # estimate typical spacing from data (like your LOESS approach)
#     d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
#     if d_data.ndim == 2 and d_data.shape[1] >= 2:
#         typical_spacing = float(np.nanpercentile(d_data[:, 1], 95))
#     else:
#         typical_spacing = float(np.median(d_grid))

#     # cutoff: keep grid points within ~1.3 * typical spacing (tweak multiplier if needed)
#     cut = max(typical_spacing * 1.3, 1e-6)
#     mask_far = (d_grid > cut)

#     # evaluate KDE on grid and mask far-away cells
#     Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
#     Z_flat = Z.ravel()
#     Z_flat[mask_far] = np.nan
#     Z = Z_flat.reshape(Xgrid.shape)

#     # pick contour levels from finite Z values
#     finite_vals = Z[np.isfinite(Z)]
#     if finite_vals.size == 0:
#         print("KDE produced no finite values inside masked region; skipping contours.")
#     else:
#         levs = np.percentile(finite_vals, [50, 75, 90, 97])

#         # filled continuous density map (many levels => visually smooth)
#         cf = ax_main.contourf(Xgrid, Ygrid, Z, levels=50, cmap='viridis', antialiased=True)

#         # optional: overlay a few contour lines
#         cs = ax_main.contour(Xgrid, Ygrid, Z, levels=levs, colors='k', linewidths=0.6, alpha=0.5)

#         # colorbar attached to the filled contours
#         fig.colorbar(cf, ax=ax_main, label='Density (KDE)')

# ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_main.set_ylabel("Mass-weighted age (Gyr)")
# # ax_main.set_title("Compactness vs Ex-situ mass fraction — 2D median")
# ax_main.grid(True)

# # ---------- Running-median / projection (age) ----------
# # Configuration: same as Mg/Fe block
# use_quantile_bins = True   # True -> equal-count bins; False -> fixed-width bins
# target_per_bin = 1000       # approx galaxies per bin when using quantile bins
# nbins_fixed = 18           # used for fixed-width
# min_count_per_bin = 5

# q_arr = ages  # quantity for this block

# # valid compactness entries (should already be aligned)
# valid_comp = np.isfinite(compactness)
# Ntot = np.sum(valid_comp)
# if Ntot == 0:
#     raise RuntimeError("No valid galaxies to bin in compactness (age).")

# if use_quantile_bins:
#     nbins = max(3, int(np.floor(Ntot / max(1, int(target_per_bin)))))
#     quantiles = np.linspace(0.0, 100.0, nbins + 1)
#     cbins = np.nanpercentile(compactness, quantiles)
#     # enforce strictly increasing edges
#     for i in range(1, cbins.size):
#         if cbins[i] <= cbins[i-1]:
#             cbins[i] = cbins[i-1] + 1e-9
# else:
#     nbins = int(nbins_fixed)
#     cbins = np.linspace(np.nanmin(compactness), np.nanmax(compactness), nbins + 1)

# cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])

# # aggregate into bins
# from collections import defaultdict
# idxs = np.searchsorted(cbins, compactness, side='right') - 1
# valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(q_arr)
# group = defaultdict(list)
# for idx, q in zip(idxs[valid_mask], q_arr[valid_mask]):
#     group[int(idx)].append(float(q))

# cmed = np.full(nbins, np.nan); clow = np.full(nbins, np.nan); chigh = np.full(nbins, np.nan)
# counts_per_bin = np.zeros(nbins, dtype=int)
# for b in range(nbins):
#     vals = group.get(b, [])
#     counts_per_bin[b] = len(vals)
#     if counts_per_bin[b] >= min_count_per_bin:
#         arr = np.asarray(vals, dtype=float)
#         cmed[b] = np.nanmedian(arr)
#         clow[b] = np.nanpercentile(arr, 16)
#         chigh[b] = np.nanpercentile(arr, 84)

# # overlay median+band on the LEFT pcolormesh
# ok = np.isfinite(cmed)
# if np.any(ok):
#     ax_main.plot(cbin_centers[ok], cmed[ok], color="black", lw=2, label="median (binned)")
#     ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15, label="16–84 pct")
#     ax_main.legend(fontsize=8, loc='upper left')

# # RIGHT panel: projection (median vs compactness)
# ax_proj.plot(cbin_centers, cmed, color="C0", lw=2)
# ax_proj.fill_between(cbin_centers, clow, chigh, color="C0", alpha=0.25)
# ax_proj.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
# ax_proj.set_ylabel("Mass-weighted age (Gyr)")
# # ax_proj.set_title("Projection: median & 16–84 pct")
# ax_proj.grid(True)

# # call threshold finder for mass-weighted age
# res_age = find_compactness_threshold(
#     cbin_centers=cbin_centers,
#     cmed=cmed,
#     compactness_all=compactness,
#     quantity_all=ages,
#     counts_per_bin=counts_per_bin,
#     bootstrap_n=0,
#     cbins_edges=cbins
# )

# bs = res_age['bootstrap']
# if bs is not None:
#     thr_raw = bs['raw']
#     methods = bs.get('methods', None)

#     if methods is not None:
#         methods = np.asarray(methods)
#         thr_turn = thr_raw[methods == 'turning_point']
#         det_rate = np.sum(methods == 'turning_point') / methods.size

#         print("detection rate:", det_rate)
#         if thr_turn.size > 0:
#             print("turning-point bootstrap median,p16,p84:",
#                   np.nanmedian(thr_turn),
#                   np.nanpercentile(thr_turn, 16),
#                   np.nanpercentile(thr_turn, 84))
#         else:
#             print("no turning-point detections in bootstrap")
#     else:
#         print("no methods recorded; inspect thr_raw histogram")

# print("age threshold:", res_age['threshold'], res_age['method'])
# if res_age['bootstrap'] is not None:
#     print("Age bootstrap median,16,84:", res_age['bootstrap']['median'],
#           res_age['bootstrap']['p16'], res_age['bootstrap']['p84'])

# # add smoothed median + derivative on projection
# if res_age is not None:
#     ax_proj.plot(cbin_centers, res_age["cmed_smooth"], linestyle="--", color="C2", lw=2, label="smoothed median")
#     ax_der = ax_proj.twinx()
#     ax_der.plot(cbin_centers, res_age["derivative"], color="C3", lw=1.4, alpha=0.9, label="|d(median)/d(compactness)|")
#     ax_der.axhline(0.0, linestyle=":", color="C3", alpha=0.6)
#     ax_der.set_ylabel("Derivative (arb. units)", color="C3")
#     ax_der.tick_params(axis="y", labelcolor="C3")
#     # merge legends
#     lines1, labels1 = ax_proj.get_legend_handles_labels()
#     lines2, labels2 = ax_der.get_legend_handles_labels()
#     ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# # optional: histogram only for fixed-width bins
# if not use_quantile_bins:
#     counts_per_bin = np.array([np.sum((compactness >= cbins[i]) & (compactness < cbins[i+1])) for i in range(nbins)])
#     ax2 = ax_proj.twinx()
#     ax2.bar(cbin_centers, counts_per_bin, width=(cbins[1]-cbins[0])*0.9, alpha=0.12, color='gray', edgecolor='none')
#     ax2.set_ylabel("N (per compactness bin)", color='gray')
#     ax2.tick_params(axis='y', labelsize=10)

# # mark threshold on projection
# ax_proj.axvline(res_age['threshold'], color='C2', linestyle='--', lw=1.6, label=f"thr={res_age['threshold']:.2f}")
# ax_proj.legend(fontsize=8)

# fig.tight_layout()
# fig_name = os.path.join(outdir, f"compactness_mwage_two_panel_z{ztarget:.1f}.png")
# fig.savefig(fig_name, dpi=300, bbox_inches='tight')
# plt.close(fig)
# print("Saved two-panel mass-weighted age plot to:", fig_name)

# ---------- Compactness vs velocity dispersion (two-panel) ----------

print("sigma finite:", np.isfinite(sigma_vals).sum(), "/", sigma_vals.size)
print("sigma min/max:", np.nanmin(sigma_vals), np.nanmax(sigma_vals))

# print test results
summary_sigma = thresholds_for_targets(
    compactness,
    log_sigma_vals,
    targets_per_bin=targets,
    min_count=5,
    bootstrap_n=0,
    do_bootstrap=False
)

for s in summary_sigma:
    thr = s['threshold_result']
    print(f"sigma target={s['target_per_bin']} -> nbins={s['nbins']} "
          f"threshold={thr['threshold']:.3f} method={thr['method']}")

fig, axes = plt.subplots(1, 2, figsize=(12,5),
                         gridspec_kw={"width_ratios":[1.2,1]})
ax_main, ax_proj = axes

# --- scatter ---
ax_main.scatter(compactness, log_sigma_vals,
                s=8, color="lightgrey", alpha=0.8, label="galaxies")

# --- KDE density contours ---
x = compactness
y = log_sigma_vals

fin = np.isfinite(x) & np.isfinite(y)
if np.sum(fin) >= 10:
    xs = x[fin]; ys = y[fin]

    pts = np.vstack([xs, ys])
    kde = gaussian_kde(pts, bw_method='scott')

    nx_grid = 200
    ny_grid = 200

    x_min, x_max = np.nanpercentile(x, [1, 99])
    y_min, y_max = np.nanpercentile(y, [1, 99])

    xpad = 0.05 * (x_max - x_min + 1e-9)
    ypad = 0.05 * (y_max - y_min + 1e-9)

    xg = np.linspace(x_min - xpad, x_max + xpad, nx_grid)
    yg = np.linspace(y_min - ypad, y_max + ypad, ny_grid)

    Xgrid, Ygrid = np.meshgrid(xg, yg)

    grid_pts = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T

    tree = cKDTree(np.column_stack((xs, ys)))
    d_grid, _ = tree.query(grid_pts, k=1)

    d_data, _ = tree.query(np.column_stack((xs, ys)), k=2)
    typical_spacing = float(np.nanpercentile(d_data[:,1], 95))

    cut = max(typical_spacing * 1.3, 1e-6)
    mask_far = (d_grid > cut)

    Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
    Z_flat = Z.ravel()
    Z_flat[mask_far] = np.nan
    Z = Z_flat.reshape(Xgrid.shape)

    finite_vals = Z[np.isfinite(Z)]
    if finite_vals.size > 0:
        levs = np.percentile(finite_vals, [50,75,90,97])

        cf = ax_main.contourf(Xgrid, Ygrid, Z, levels=50,
                              cmap='viridis', antialiased=True)

        ax_main.contour(Xgrid, Ygrid, Z,
                        levels=levs, colors='k',
                        linewidths=0.6, alpha=0.5)

        fig.colorbar(cf, ax=ax_main, label='Density (KDE)')

ax_main.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
ax_main.set_ylabel(r'$\lg(\sigma \, / \, \mathrm{km}\ \mathrm{s}^{-1})$')
ax_main.grid(True)

# ---------- Running-median / projection ----------

use_quantile_bins = True
target_per_bin = 1000
min_count_per_bin = 5

q_arr = log_sigma_vals

# keep only paired finite data for bin counting
valid_pair = np.isfinite(compactness) & np.isfinite(q_arr)
Ntot = np.sum(valid_pair)
if Ntot == 0:
    raise RuntimeError("No finite compactness/sigma pairs to bin.")

# optional compactness window for threshold search
search_lo = 8.0
search_hi = np.inf

raw_search_mask = valid_pair & (compactness >= search_lo) & (compactness <= search_hi)
compactness_search = compactness[raw_search_mask]
sigma_search = q_arr[raw_search_mask]

if compactness_search.size == 0:
    raise RuntimeError("No data left after applying the compactness search window.")

# build quantile bins on the SEARCHED data only
nbins = max(3, int(np.floor(compactness_search.size / max(1, int(target_per_bin)))))
cbins = np.nanpercentile(compactness_search, np.linspace(0, 100, nbins + 1))

# enforce monotonic edges
for i in range(1, len(cbins)):
    if cbins[i] <= cbins[i - 1]:
        cbins[i] = cbins[i - 1] + 1e-9

cbin_centers = 0.5 * (cbins[:-1] + cbins[1:])

# aggregate
from collections import defaultdict
idxs = np.searchsorted(cbins, compactness_search, side='right') - 1
valid_mask = (idxs >= 0) & (idxs < nbins) & np.isfinite(sigma_search)

group = defaultdict(list)
for idx, q in zip(idxs[valid_mask], sigma_search[valid_mask]):
    group[int(idx)].append(float(q))

cmed = np.full(nbins, np.nan)
clow = np.full(nbins, np.nan)
chigh = np.full(nbins, np.nan)
counts_per_bin = np.zeros(nbins, dtype=int)

for b in range(nbins):
    vals = group.get(b, [])
    counts_per_bin[b] = len(vals)
    if len(vals) >= min_count_per_bin:
        arr = np.asarray(vals, dtype=float)
        cmed[b] = np.nanmedian(arr)
        clow[b] = np.nanpercentile(arr, 16)
        chigh[b] = np.nanpercentile(arr, 84)

# plot
ok = np.isfinite(cmed)
if np.any(ok):
    ax_main.plot(cbin_centers[ok], cmed[ok], color="black", lw=2, label="median (binned)")
    ax_main.fill_between(cbin_centers, clow, chigh, color="black", alpha=0.15)

ax_proj.plot(cbin_centers, cmed, color="C0", lw=2)
ax_proj.fill_between(cbin_centers, clow, chigh, color="C0", alpha=0.25)
ax_proj.set_xlabel(r"Compactness (lg[$M_\odot \text{kpc}^{-1.5}$])")
ax_proj.set_ylabel(r'$\lg(\sigma \, / \, \mathrm{km}\ \mathrm{s}^{-1})$')
ax_proj.grid(True)

# threshold
res_sigma = find_compactness_threshold(
    cbin_centers=cbin_centers,
    cmed=cmed,
    compactness_all=compactness_search,
    quantity_all=sigma_search,
    counts_per_bin=counts_per_bin,
    bootstrap_n=0,
    cbins_edges=cbins
)

print("sigma threshold:", res_sigma["threshold"], res_sigma["method"])

ax_proj.plot(cbin_centers, res_sigma["cmed_smooth"], linestyle="--", color="C2", lw=2, label="smoothed median")

ax_der = ax_proj.twinx()
ax_der.plot(cbin_centers, res_sigma["derivative"], color="C3", lw=1.4, alpha=0.9)
ax_der.axhline(0.0, linestyle=":", color="C3", alpha=0.6)
ax_der.set_ylabel("Derivative (arb. units)", color="C3")
ax_der.tick_params(axis="y", labelcolor="C3")

ax_proj.axvline(res_sigma["threshold"], color="C2", linestyle="--", lw=1.6,
                label=f"thr={res_sigma['threshold']:.2f}")

lines1, labels1 = ax_proj.get_legend_handles_labels()
lines2, labels2 = ax_der.get_legend_handles_labels()
ax_proj.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# save
fig.tight_layout()
fig_name = os.path.join(outdir,
    f"compactness_sigma_two_panel_z{ztarget:.1f}.png")

fig.savefig(fig_name, dpi=300, bbox_inches='tight')
plt.close(fig)

print("Saved sigma plot to:", fig_name)