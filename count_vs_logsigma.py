#!/usr/bin/env python3
"""
count_vs_logsigma.py

Compute counts & fraction of (selected) galaxies as a function of logsigma threshold:
  compactness C = log10(Mstar) - 1.5 * log10(Rhalf / kpc)
For a grid of logsigma thresholds (default step = 0.1) the script computes:
  - N_compact(logsigma) : number with C >= logsigma
  - N_below(logsigma)   : number with C < logsigma
  - fraction_compact    : N_compact / N_total

Outputs a PNG in ./plots called compactness_counts_vs_logsigma.png

Adjust MODEL / SNAP / selection at top if needed.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import common

# -------- CONFIG (edit if needed) ----------
model_name = 'L0200N3008/THERMAL_AGN/'
model_dir  = '/mnt/su3-pro/colibre/' + model_name
snap_file  = '0127'   # z=0
ztarget    = 0.0
comov_to_physical_length = 1.0 / (1.0 + ztarget)

OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

# CSV cache path (use single path everywhere)
csv_path = os.path.join(OUTDIR, "compactness_counts_vs_logsigma.csv")

# selection limits (same as your script)
M_MIN = 1e9    # Msun (minimum total stellar mass to include)

# UCMG vertical line (changeable)
UCMG_LOGSIGMA = 9.9

# --------- load cached results if available ----------
if os.path.exists(csv_path):
    print(f"Loading precomputed results from {csv_path}")
    df = pd.read_csv(csv_path)
    logsigma_vals = df["logsigma"].to_numpy()
    N_compact = df["N_compact"].to_numpy(dtype=int)
    N_below = df["N_below"].to_numpy(dtype=int)
    fraction_compact = df["fraction_compact"].to_numpy(dtype=float)
    # derive N_total (should be constant across thresholds)
    if N_compact.size > 0:
        N_total = int(N_compact[0] + N_below[0])
    else:
        N_total = 0
else:
    print("CSV not found – computing compactness statistics")

    # logsigma grid
    logsigma_min = 8.0
    logsigma_max = 10.6
    logsigma_step = 0.1
    logsigma_vals = np.arange(logsigma_min, logsigma_max + 1e-9, logsigma_step)

    # -------- READ SOAP (same fields used previously) ----------
    fields_sgn = {'InputHalos': ('HaloCatalogueIndex', 'IsCentral', 'HBTplus/DescendantTrackId', 'HBTplus/TrackId')}
    fields = {'ExclusiveSphere/50kpc': (
                'StellarMass', 'StarFormationRate', 'HalfMassRadiusStars',
                'MassWeightedMeanStellarAge', 'LuminosityWeightedMeanStellarAge',
                'LinearMassWeightedIronOverHydrogenOfStars',
                'LinearMassWeightedMagnesiumOverHydrogenOfStars'
            )}

    print("Reading SOAP groups (common.read_group_data_colibre)...")
    h5data_groups   = common.read_group_data_colibre(model_dir, snap_file, fields)
    h5data_idgroups = common.read_group_data_colibre(model_dir, snap_file, fields_sgn)

    (halo_index, is_central, desc_id, track_id) = h5data_idgroups
    (m30, sfr30, r50, stellarage, stellarage_lum, Fe_lin, Mg_lin) = h5data_groups

    # unit conversions (match your script)
    Mu = 1.988e43 / 1.989e33
    tu = 3.086e19 / 3.154e7
    m30 = m30 * Mu
    sfr30 = sfr30 * Mu / tu
    r50 = r50 * comov_to_physical_length * 1e3  # -> kpc
    stellarage_lum = stellarage_lum * tu / 1e9

    # selection: mass limit
    sel = (m30 >= M_MIN)
    m_sel = m30[sel]
    r_sel = r50[sel]

    # ensure positives
    mask_pos = (m_sel > 0) & (r_sel > 0)
    m_sel = m_sel[mask_pos]
    r_sel = r_sel[mask_pos]

    if m_sel.size == 0:
        raise SystemExit("No galaxies selected (check M_MIN / data).")

    # compute compactness
    logM = np.log10(m_sel)
    logR = np.log10(r_sel)
    compactness = logM - 1.5 * logR

    # prepare results arrays
    N_compact = np.zeros_like(logsigma_vals, dtype=int)
    N_below = np.zeros_like(logsigma_vals, dtype=int)
    fraction_compact = np.zeros_like(logsigma_vals, dtype=float)
    N_total = compactness.size

    for i, s in enumerate(logsigma_vals):
        mask_compact = (compactness >= s)
        N_compact[i] = int(np.count_nonzero(mask_compact))
        N_below[i] = int(N_total - N_compact[i])
        fraction_compact[i] = float(N_compact[i]) / float(N_total) if N_total > 0 else 0.0

    # save CSV cache
    df = pd.DataFrame({
        "logsigma": logsigma_vals,
        "N_compact": N_compact,
        "N_below": N_below,
        "fraction_compact": fraction_compact
    })
    df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

# ---------- PLOTTING ----------
plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "serif", "font.size": 12})

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7,8), sharex=True,
                               gridspec_kw={"height_ratios": [2,1]})

# top: counts
ax1.plot(logsigma_vals, N_compact, marker='o', ms=4, lw=1.5, label='compact')
ax1.plot(logsigma_vals, N_below, marker='s', ms=4, lw=1.0, label='non-compact')
ax1.set_ylabel("Number of galaxies")
ax1.grid(True)
ax1.legend(fontsize=9)

# annotate total
ax1.text(0.98, 0.92, f"N_total = {N_total}", transform=ax1.transAxes, ha='right', va='top', fontsize=10)

# bottom: fraction
ax2.plot(logsigma_vals, fraction_compact, marker='o', ms=4, lw=1.5)
ax2.set_xlabel(r"$\lg\Sigma_{1.5}$")
ax2.set_ylabel("Fraction compact")
ax2.set_ylim(0, 1.05)
ax2.grid(True)

# optional: vertical line at chosen UCMG threshold
ax1.axvline(UCMG_LOGSIGMA, color='C3', linestyle='--', linewidth=1.5, label=f'UCMG threshold ($\lg\Sigma_{{1.5}}$ = {UCMG_LOGSIGMA})')
ax2.axvline(UCMG_LOGSIGMA, color='C3', linestyle='--', linewidth=1.5)
# update legend to include UCMG line
ax1.legend(fontsize=9)

# save figure
outpath = os.path.join(OUTDIR, "compactness_counts_vs_logsigma.png")
fig.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved:", outpath)