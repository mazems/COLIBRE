#!/usr/bin/env python3
"""
dor_bimodality_and_binmedians.py

Usage: python dor_bimodality_and_binmedians.py

Requires:
 - numpy, pandas, matplotlib, scipy, sklearn (sklearn optional; script will run without it, but GMM won't run)
 - A diagnostic CSV produced by your main script: plots_dor/soap_ucmg_matched_summary.csv
   which must contain at least these columns: DoR, MgFe, logM, log_ssfr  (names accepted:
   "DoR","MgFe","logM","log_ssfr"). Adjust column names in code if your CSV differs.

What it does:
 1) Produces DoR vs MgFe diagnostic plots (full sample and restricted mass window)
    - hexbin density (no colour variable),
    - 2D histogram,
    - scatter coloured by log_ssfr (NaN/inf sSFR shown grey)
    - optional GaussianMixture (k=2) fit and cluster overlay (prints BIC/AIC, cluster means)
 2) Produces "median lines + 16/84" plots for every quantity per mass bin,
    writing files in plots_dor/by_mass_bin/median_lines_all_bins/
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
from scipy.spatial import cKDTree as KDTree

# Optional: GaussianMixture for objective 2-component check
try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---- configuration ----
diag_csv = "plots_dor/soap_ucmg_matched_summary.csv"   # created by your main script
outdir = "plots_dor/bimodality_and_medians"
os.makedirs(outdir, exist_ok=True)
median_lines_dir = os.path.join(outdir, "by_mass_bin_median_lines")
os.makedirs(median_lines_dir, exist_ok=True)

# mass-range you asked about
focus_mass_lo = 9.6
focus_mass_hi = 10.6

# binning for mass bins (match your main code: 0.2 dex)
mass_bin_width = 0.2

# LOESS-style/median-line internals
quantile_bins_per_curve = 10  # number of x-quantiles used to make a line inside each mass-bin
min_points_per_segment = 5    # require this many points to compute median in a segment

# plotting styles
plt.rcParams.update({"font.size": 12, "figure.figsize": (7,5)})

# ---- helpers ----
def safe_load_diag(csv_path):
    if not os.path.exists(csv_path):
        raise SystemExit(f"Diagnostic CSV not found: {csv_path}\nRun your main script to produce {csv_path}")
    df = pd.read_csv(csv_path)
    # Accept alternative names: try to normalise
    names = df.columns.tolist()
    # expected fields (try find alternatives)
    colmap = {}
    def find(col_options):
        for c in col_options:
            if c in df.columns:
                return c
        return None
    colmap['DoR'] = find(['DoR','dor','DoR_csv'])
    colmap['MgFe'] = find(['MgFe','Mg_lin','MGFE','MgFe'])
    colmap['logM'] = find(['logM','lgM','mass_log','logMstar','logM'])
    colmap['logR'] = find(['logR','lgR','logR'])
    colmap['log_ssfr'] = find(['log_ssfr','logSSFR','log_sSFR','log_sfr'])
    # sanity check
    for req in ['DoR','MgFe','logM']:
        if colmap[req] is None:
            raise SystemExit(f"Column for '{req}' not found in {csv_path}; available columns: {names}")
    return df, colmap

def hexbin_plot(x, y, ax=None, gridsize=60, mincnt=1, title=None, xlabel=None, ylabel=None, cmap='viridis'):
    if ax is None:
        fig, ax = plt.subplots()
    hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=mincnt, cmap=cmap)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("counts")
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def hist2d_plot(x, y, ax=None, bins=100, range=None, title=None, xlabel=None, ylabel=None):
    if ax is None:
        fig, ax = plt.subplots()
    h = ax.hist2d(x, y, bins=bins, range=range)
    plt.colorbar(h[3], ax=ax).set_label("counts")
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def scatter_colored_by_ssfr(x, y, ssfr, ax=None, title=None, xlabel=None, ylabel=None, cmap='viridis', vmin=None, vmax=None):
    if ax is None:
        fig, ax = plt.subplots()
    finite = np.isfinite(ssfr)
    # plot missing as light grey small points behind
    ax.scatter(x[~finite], y[~finite], s=8, color='lightgrey', alpha=0.6, label='no finite sSFR')
    if np.any(finite):
        sc = ax.scatter(x[finite], y[finite], c=ssfr[finite], cmap=cmap, s=8, vmin=vmin, vmax=vmax)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("log(sSFR)")
    ax.legend(fontsize=8)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def run_gmm_check(x, y, verbose=True):
    if not SKLEARN_AVAILABLE:
        if verbose: print("sklearn not available: skipping GMM clustering check.")
        return None
    XY = np.column_stack((x, y))
    # drop NaNs
    mask = np.isfinite(XY).all(axis=1)
    XYf = XY[mask]
    if XYf.shape[0] < 20:
        if verbose: print("Too few points for GMM:", XYf.shape[0])
        return None
    # fit GMM k=1..3 and choose by BIC
    models = {}
    for k in [1,2,3]:
        gm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        gm.fit(XYf)
        models[k] = {'model': gm, 'bic': gm.bic(XYf), 'aic': gm.aic(XYf)}
    # pick best BIC
    bestk = min(models.keys(), key=lambda k: models[k]['bic'])
    if verbose:
        print("GMM BIC/AIC by k:")
        for k,v in models.items():
            print(f"  k={k}: BIC={v['bic']:.1f}, AIC={v['aic']:.1f}")
        print("Best k by BIC:", bestk)
    return models[bestk]

# LOESS internals (required for smooth maps) ---------------------------------
def polyfit_2d(x, y, z, degree=1, weights=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()
    if weights is None:
        W = np.ones_like(z, dtype=float)
    else:
        W = np.asarray(weights).ravel()
    xc = np.average(x, weights=W)
    yc = np.average(y, weights=W)
    dx = x - xc
    dy = y - yc
    if degree == 0:
        sw = W.sum()
        if sw == 0:
            return np.array([np.nan])
        a0 = (W @ z) / sw
        return np.array([a0])
    A = np.column_stack((np.ones_like(dx), dx, dy))
    ATW = (A.T * W)
    ATA = ATW @ A
    ATy = ATW @ z
    ridge = 1e-12 * np.trace(ATA) if np.isfinite(np.trace(ATA)) and np.trace(ATA) != 0 else 1e-12
    try:
        ATA[0, 0] += ridge
        beta = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(ATA) @ ATy
    return np.array(beta)

def _biweight_scale(resid):
    resid = np.abs(resid)
    if resid.size == 0:
        return 1.0
    mad = np.median(resid)
    if mad <= 0:
        return 1e-9
    return 1.4826 * mad

def loess_2d(x1, y1, z, frac=0.5, degree=1, npoints=None, xout=None, yout=None, sigz=None):
    """
    Robust local linear 2D loess. Returns (zout, wout) for xout,yout.
    If xout/yout are None, returns estimates at input points.
    """
    x1 = np.asarray(x1).ravel()
    y1 = np.asarray(y1).ravel()
    z = np.asarray(z).ravel()
    if not (x1.size == y1.size == z.size):
        raise ValueError("X, Y, Z must be same length")
    n = x1.size
    if n == 0:
        return np.array([]), np.array([])
    if npoints is None:
        npoints = int(np.ceil(frac * n))
    npoints = max(2, min(npoints, n))
    if xout is None or yout is None:
        xout = x1.copy()
        yout = y1.copy()
    xout = np.asarray(xout).ravel()
    yout = np.asarray(yout).ravel()
    if xout.size != yout.size:
        raise ValueError("xout and yout same length required")
    m = xout.size
    zout = np.empty(m, dtype=float)
    wout = np.empty(m, dtype=float)
    tree = KDTree(np.column_stack((x1, y1)))
    for j, (xx, yy) in enumerate(zip(xout, yout)):
        dists, inds = tree.query([xx, yy], k=npoints)
        if np.isscalar(dists):
            dists = np.array([dists]); inds = np.array([inds])
        rmax = np.max(dists)
        if rmax == 0:
            zout[j] = z[inds[0]]
            wout[j] = 1.0
            continue
        u = dists / rmax
        distWeights = (1.0 - u**3)**3
        distWeights = np.where(u >= 1.0, 0.0, distWeights)
        xw = x1[inds]; yw = y1[inds]; zw = z[inds]
        w_init = distWeights.copy()
        coeffs = polyfit_2d(xw, yw, zw, degree=degree, weights=w_init)
        if degree == 0:
            zfit = np.full_like(zw, coeffs[0], dtype=float)
        else:
            xc = np.average(xw, weights=w_init)
            yc = np.average(yw, weights=w_init)
            dx = xw - xc; dy = yw - yc
            a0, ax, ay = coeffs
            zfit = a0 + ax * dx + ay * dy
        biWeights = np.ones_like(zw)
        for it in range(10):
            if sigz is None:
                resid = zfit - zw
                scale = _biweight_scale(resid)
                uu = (np.abs(resid) / (6.0 * scale)) ** 2.0
            else:
                uu = ((zfit - zw) / (4.0 * sigz[inds])) ** 2.0
            uu = np.clip(uu, 0.0, 1.0)
            biWeights_new = (1.0 - uu) ** 2.0
            totWeights = distWeights * biWeights_new
            coeffs = polyfit_2d(xw, yw, zw, degree=degree, weights=totWeights)
            if degree == 0:
                zfit = np.full_like(zw, coeffs[0], dtype=float)
            else:
                xc = np.average(xw, weights=totWeights) if np.sum(totWeights) > 0 else np.mean(xw)
                yc = np.average(yw, weights=totWeights) if np.sum(totWeights) > 0 else np.mean(yw)
                dx = xw - xc; dy = yw - yc
                a0, ax, ay = coeffs
                zfit = a0 + ax * dx + ay * dy
            if np.allclose(biWeights, biWeights_new, atol=1e-6):
                biWeights = biWeights_new
                break
            biWeights = biWeights_new
        zout[j] = coeffs[0]
        wout[j] = biWeights[0] if biWeights.size > 0 else 1.0
    return zout, wout

def loess_coloured_DoR_vs_MgFe_by_ssfr(mgfe, dor, log_ssfr, outname,
                                       nx=300, ny=220, frac_loess=0.10, max_eval_pts=12000,
                                       pad_frac=0.05, min_points=6):
    """
    LOESS-smoothed map of log_sSFR on the (MgFe (x), DoR (y)) plane.
    Saves file into outdir/outname.
    """
    mask = np.isfinite(mgfe) & np.isfinite(dor) & np.isfinite(log_ssfr)
    if mask.sum() < min_points:
        print(f"LOESS ssfr map skipped ({mask.sum()} finite points).")
        return

    x_in = mgfe[mask].astype(float)
    y_in = dor[mask].astype(float)
    z_in = log_ssfr[mask].astype(float)

    N = x_in.size
    if (max_eval_pts is not None) and (N > int(max_eval_pts)):
        rng = np.random.default_rng(seed=123456)
        sel = rng.choice(N, size=int(max_eval_pts), replace=False)
        x_loess = x_in[sel].copy(); y_loess = y_in[sel].copy(); z_loess = z_in[sel].copy()
    else:
        x_loess = x_in.copy(); y_loess = y_in.copy(); z_loess = z_in.copy()

    pad_x = pad_frac * (np.nanmax(x_loess) - np.nanmin(x_loess) + 1e-9)
    pad_y = pad_frac * (np.nanmax(y_loess) - np.nanmin(y_loess) + 1e-9)
    xg = np.linspace(np.nanmin(x_loess) - pad_x, np.nanmax(x_loess) + pad_x, nx)
    yg = np.linspace(np.nanmin(y_loess) - pad_y, np.nanmax(y_loess) + pad_y, ny)
    Xg, Yg = np.meshgrid(xg, yg)
    pts_grid = np.column_stack((Xg.ravel(), Yg.ravel()))

    tree_data = KDTree(np.column_stack((x_loess, y_loess)))
    d_grid, _ = tree_data.query(pts_grid, k=1)
    d_data, _ = tree_data.query(np.column_stack((x_loess, y_loess)), k=2)
    if d_data.ndim == 2 and d_data.shape[1] >= 2:
        typical_spacing = float(np.nanpercentile(d_data[:,1], 95))
    else:
        typical_spacing = float(np.nanmedian(d_grid))
    d_thresh = max(typical_spacing * 1.3, 1e-6)
    inside_mask = (d_grid <= d_thresh)
    idx_inside = np.nonzero(inside_mask)[0]

    Zflat = np.full(pts_grid.shape[0], np.nan, dtype=float)
    if idx_inside.size > 0:
        xout = pts_grid[idx_inside, 0]
        yout = pts_grid[idx_inside, 1]
        Z_inside, _ = loess_2d(x_loess, y_loess, z_loess, frac=frac_loess, degree=1, xout=xout, yout=yout)
        Zflat[idx_inside] = Z_inside

    Zgrid = Zflat.reshape((ny, nx))
    Zmask = np.ma.masked_invalid(Zgrid)

    try:
        vmin = float(np.nanpercentile(z_in, 5)); vmax = float(np.nanpercentile(z_in, 95))
    except Exception:
        vmin, vmax = float(np.nanmin(z_in)), float(np.nanmax(z_in))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        med = float(np.nanmedian(z_in)); span = max(0.2, 0.5 * max(1e-9, abs(med)))
        vmin = med - span; vmax = med + span

    fig, ax = plt.subplots(figsize=(8,6))
    # faint scatter for orientation
    ax.scatter(x_in, y_in, s=6, color="lightgrey", alpha=0.5)
    im = ax.pcolormesh(Xg, Yg, Zmask, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("lg(sSFR / yr$^{-1}$)")
    ax.set_xlabel("[Mg/Fe]")
    ax.set_ylabel("DoR")
    # ax.set_title("LOESS map: log(sSFR) on (MgFe, DoR)")
    ax.grid(True)
    fullpath = os.path.join(outdir, outname)
    fig.savefig(fullpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved LOESS sSFR map:", fullpath)

# ---- main work ----
def main():
    df, cmap = safe_load_diag(diag_csv)
    # map columns
    DoR_col = cmap['DoR']
    MgFe_col = cmap['MgFe']
    logM_col = cmap['logM']
    logssfr_col = cmap['log_ssfr']

    # prepare arrays
    DoR = df[DoR_col].to_numpy(dtype=float)
    MgFe = df[MgFe_col].to_numpy(dtype=float)
    logM = df[logM_col].to_numpy(dtype=float)
    # log_ssfr may be missing -> set to nan if not present
    if logssfr_col is None:
        log_ssfr = np.full_like(DoR, np.nan, dtype=float)
    else:
        log_ssfr = df[logssfr_col].to_numpy(dtype=float)

    # mask for valid MgFe+DoR
    mask_valid = np.isfinite(DoR) & np.isfinite(MgFe)
    print("Total matched rows:", len(df), "Valid DoR+MgFe:", mask_valid.sum())

    # two selections: full and focus mass-range
    sel_full = mask_valid
    sel_focus = mask_valid & (logM >= focus_mass_lo) & (logM < focus_mass_hi)
    print("Focus mass bin [{:.2f},{:.2f}): {} points".format(focus_mass_lo, focus_mass_hi, np.sum(sel_focus)))

    # --- plots for both selections ---
    for sel_name, sel_mask in [('full', sel_full), ('focus', sel_focus)]:
        if sel_mask.sum() == 0:
            print("No points for selection:", sel_name)
            continue

        X = MgFe[sel_mask]; Y = DoR[sel_mask]; S = log_ssfr[sel_mask]

        # 1) hexbin
        fig, ax = plt.subplots(figsize=(7,5))
        hexbin_plot(X, Y, ax=ax, gridsize=60, mincnt=1, title=f"DoR vs MgFe (hexbin) [{sel_name}]",
                    xlabel="[Mg/Fe] (dex)", ylabel="DoR")
        fign = os.path.join(outdir, f"DoR_vs_MgFe_hexbin_{sel_name}.png")
        fig.savefig(fign, dpi=200, bbox_inches='tight'); plt.close(fig)
        print("Wrote", fign)

        # 2) hist2d
        fig, ax = plt.subplots(figsize=(7,5))
        hist2d_plot(X, Y, ax=ax, bins=120, title=f"DoR vs MgFe (hist2d) [{sel_name}]",
                    xlabel="[Mg/Fe] (dex)", ylabel="DoR")
        fign = os.path.join(outdir, f"DoR_vs_MgFe_hist2d_{sel_name}.png")
        fig.savefig(fign, dpi=200, bbox_inches='tight'); plt.close(fig)
        print("Wrote", fign)

        # 3) scatter colored by log_ssfr (if available)
        fig, ax = plt.subplots(figsize=(7,5))
        scatter_colored_by_ssfr(X, Y, S, ax=ax, title=f"DoR vs MgFe coloured by log(sSFR) [{sel_name}]",
                                xlabel="[Mg/Fe] (dex)", ylabel="DoR", vmin=-14, vmax=-8)
        fign = os.path.join(outdir, f"DoR_vs_MgFe_colored_ssfr_{sel_name}.png")
        fig.savefig(fign, dpi=200, bbox_inches='tight'); plt.close(fig)
        print("Wrote", fign)
        # LOESS map of log_sSFR on (MgFe, DoR)
        loess_coloured_DoR_vs_MgFe_by_ssfr(mgfe=X, dor=Y, log_ssfr=S,
                                        outname=f"Loess_logssfr_on_MgFe_DoR_{sel_name}.png",
                                        nx=300, ny=220, frac_loess=0.10, max_eval_pts=5000)

        # 4) GMM check (k=2) optionally
        gm_best = run_gmm_check(X, Y, verbose=True)
        if gm_best is not None:
            gm = gm_best['model']
            # assign cluster labels on the cleaned data the model used (we need to re-filter to same rows)
            XY = np.column_stack((DoR[sel_mask], MgFe[sel_mask]))
            # model was trained on non-NaN rows; but we passed non-NaN already, so consistent
            labels = gm.predict(XY)
            # overlay clusters on scatter
            fig, ax = plt.subplots(figsize=(7,5))
            for lab in np.unique(labels):
                idx = labels == lab
                ax.scatter(X[idx], Y[idx], s=8, alpha=0.7, label=f"comp {lab}")
            ax.set_ylabel("DoR"); ax.set_xlabel("[Mg/Fe]")
            ax.set_title(f"GMM (k={gm.n_components}) on DoR vs MgFe [{sel_name}]")
            ax.legend()
            fign = os.path.join(outdir, f"DoR_vs_MgFe_GMMk{gm.n_components}_{sel_name}.png")
            fig.savefig(fign, dpi=200, bbox_inches='tight'); plt.close(fig)
            print("Wrote", fign)
            print("GMM component means:", gm.means_)
            print("GMM weights:", gm.weights_)
            print("GMM covariances (shapes):", [c.shape for c in gm.covariances_])

    # -------------------------
    # PART 2: median lines + percentiles for all mass bins in one plot per quantity
    # -------------------------
    # Quantities to make DoR-vs-X median-line plots for (like your original script)
    # Map names (key->(array, xlabel, filename-friendly))
    quantities = {
        "compactness": ("compactness", None),   # if compactness not in CSV must compute: logM - 1.5*logR
        "MgFe": (MgFe_col, "[Mg/Fe] (dex)"),
        "lum_age_gyr": ("lum_age_gyr", "Lum-weighted age (Gyr)"),
        "log_ssfr": (logssfr_col, "log(sSFR / yr^-1)"),
        "exsitu_frac": ("exsitu_frac", "Ex-situ mass fraction")
    }

    # Attempt to compute missing quantities if available in CSV
    # For compactness, need logR
    if 'compactness' in df.columns:
        compactness_arr = df['compactness'].to_numpy(dtype=float)
    else:
        if 'logR' in df.columns:
            compactness_arr = df[logM_col].to_numpy(dtype=float) - 1.5 * df['logR'].to_numpy(dtype=float)
        else:
            compactness_arr = None

    # lum_age_gyr, exsitu_frac may or may not be present
    lum_age = df['lum_age_gyr'].to_numpy(dtype=float) if 'lum_age_gyr' in df.columns else None
    exsitu = df['exsitu_frac'].to_numpy(dtype=float) if 'exsitu_frac' in df.columns else None

    # create arrays dict for quantities (aligned)
    arrs = {}
    arrs['DoR'] = DoR
    arrs['MgFe'] = MgFe
    if compactness_arr is not None:
        arrs['compactness'] = compactness_arr
    if lum_age is not None:
        arrs['lum_age_gyr'] = lum_age
    arrs['log_ssfr'] = log_ssfr
    if exsitu is not None:
        arrs['exsitu_frac'] = exsitu

    # mass bins
    mmin = np.nanmin(logM); mmax = np.nanmax(logM)
    # floor/ceil to multiples of mass_bin_width
    lo0 = np.floor(mmin / mass_bin_width) * mass_bin_width
    hi0 = np.ceil(mmax / mass_bin_width) * mass_bin_width
    mass_bins = np.arange(lo0, hi0 + 1e-9, mass_bin_width)
    nbins = len(mass_bins) - 1
    print(f"Making median-line plots for {nbins} mass bins from {lo0:.2f} to {hi0:.2f} (width {mass_bin_width})")

    colors = cm.get_cmap('tab10', nbins)

    for qname, (qcol, xlabel_override) in quantities.items():
        if qname not in arrs:
            print("Skipping quantity (not found):", qname); continue
        qarr = arrs[qname]
        # create figure
        fig, ax = plt.subplots(figsize=(8,5))
        # for legend
        handles = []
        labels = []
        for ib in range(nbins):
            mlo = mass_bins[ib]; mhi = mass_bins[ib+1]
            sel_bin = (logM >= mlo) & (logM < mhi) & np.isfinite(DoR) & np.isfinite(qarr)
            nbin = int(sel_bin.sum())
            if nbin < min_points_per_segment:
                # not enough points for this mass bin
                continue
            # build quantile bins in X (the quantity qarr) within this mass bin
            q_perc = np.linspace(0, 100, quantile_bins_per_curve + 1)
            edges = np.percentile(qarr[sel_bin], q_perc)
            edges = np.unique(edges)
            if edges.size < 2:
                continue
            centers = 0.5 * (edges[:-1] + edges[1:])
            med_vals = np.full_like(centers, np.nan, dtype=float)
            p16 = np.full_like(centers, np.nan, dtype=float)
            p84 = np.full_like(centers, np.nan, dtype=float)
            for i in range(len(centers)):
                segsel = sel_bin & (qarr >= edges[i]) & (qarr < edges[i+1])
                if segsel.sum() >= min_points_per_segment:
                    v = DoR[segsel]
                    med_vals[i] = np.nanmedian(v)
                    p16[i] = np.nanpercentile(v, 16)
                    p84[i] = np.nanpercentile(v, 84)
            ok = np.isfinite(med_vals)
            if ok.sum() == 0:
                continue
            color = colors(ib % 10)
            ax.plot(centers[ok], med_vals[ok], '-', color=color, lw=1.5, label=f"{mlo:.1f}-{mhi:.1f}")
            ax.fill_between(centers[ok], p16[ok], p84[ok], color=color, alpha=0.15)
        ax.set_xlabel(xlabel_override if xlabel_override is not None else qname)
        ax.set_ylabel("DoR")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)
        ax.legend(title=f"mass bins (lg M)", fontsize=8, ncol=2, loc='best')
        fign = os.path.join(median_lines_dir, f"DoR_vs_{qname}_medianlines_all_massbins.png")
        fig.savefig(fign, dpi=200, bbox_inches='tight'); plt.close(fig)
        print("Wrote median-lines plot:", fign)

    print("All done. Outputs in:", outdir)

if __name__ == "__main__":
    main()