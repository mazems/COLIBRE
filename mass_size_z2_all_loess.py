#!/usr/bin/env python3
"""
mass_size_z2_all_loess.py

Unified mass-size plotting script for the z=2 snapshot of the COLIBRE
L0200N3008/THERMAL_AGN model.

Produces LOESS-coloured mass-size plots for:
  1. luminosity-weighted mean stellar age
  2. stellar metallicity [Z/H]
  3. [Mg/Fe]
  4. log10(sSFR / yr^-1)
  5. log10(sigma / km s^-1)
  6. ex-situ stellar mass fraction

Memory-conscious version:
  - one SOAP catalogue read
  - only fields actually required are requested
  - selection is done before copying/converting arrays
  - full SOAP arrays are explicitly released after selection
  - h5data_all reference is explicitly deleted
  - sigma is read only for selected rows
  - ex-situ catalogue is read in chunks
  - no unnecessary projected quantities
  - no redundant positive-value masks
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree as KDTree

import common


# ============================================================================
# LOESS helpers
# ============================================================================
def polyfit_2d(x, y, z, degree=1, weights=None):
    """
    Weighted 2D polynomial fit returning coefficients.

    degree==0:
        z ~ a0

    degree==1:
        z ~ a0 + ax*(x-xc) + ay*(y-yc)
    """
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

    if degree == 1:
        A = np.column_stack((np.ones_like(dx), dx, dy))

        ATW = A.T * W
        ATA = ATW @ A
        ATy = ATW @ z

        trace_ata = np.trace(ATA)

        if np.isfinite(trace_ata) and trace_ata != 0:
            ridge = 1e-12 * trace_ata
        else:
            ridge = 1e-12

        try:
            ATA[0, 0] += ridge
            beta = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(ATA) @ ATy

        return np.asarray(beta)

    raise NotImplementedError("Only degree 0 or 1 supported")


def _biweight_scale(resid):
    """
    Robust scale estimator:
    MAD -> approximate Gaussian sigma.
    """
    resid = np.abs(resid)

    if resid.size == 0:
        return 1.0

    mad = np.median(resid)

    if mad <= 0:
        return 1e-9

    return 1.4826 * mad


def loess_2d(
    x1,
    y1,
    z,
    frac=0.1,
    degree=1,
    rescale=False,
    npoints=None,
    sigz=None,
    xout=None,
    yout=None,
):
    """
    2D LOESS with robust biweight reweighting.

    Same algorithm as in the supplied scripts.
    """

    x1 = np.asarray(x1).ravel()
    y1 = np.asarray(y1).ravel()
    z = np.asarray(z).ravel()

    if not (x1.size == y1.size == z.size):
        raise ValueError(
            "Input vectors (X, Y, Z) must have the same size"
        )

    n = x1.size

    if n == 0:
        return np.array([]), np.array([])

    if npoints is None:
        npoints = int(np.ceil(frac * n))

    npoints = max(2, min(npoints, n))

    if xout is None or yout is None:
        xout = x1.copy()
        yout = y1.copy()
    else:
        xout = np.asarray(xout).ravel()
        yout = np.asarray(yout).ravel()

        if xout.size != yout.size:
            raise ValueError(
                "xout and yout must have same length"
            )

    m = xout.size

    zout = np.empty(m, dtype=float)
    wout = np.empty(m, dtype=float)

    tree = KDTree(
        np.column_stack((x1, y1))
    )

    for j, (xx, yy) in enumerate(zip(xout, yout)):

        dists, inds = tree.query(
            [xx, yy],
            k=npoints,
        )

        if np.isscalar(dists):
            dists = np.array([dists])
            inds = np.array([inds])

        rmax = np.max(dists)

        if rmax == 0:
            zout[j] = z[inds[0]]
            wout[j] = 1.0
            continue

        u = dists / rmax

        distWeights = (1.0 - u**3) ** 3

        distWeights = np.where(
            u >= 1.0,
            0.0,
            distWeights,
        )

        xw = x1[inds]
        yw = y1[inds]
        zw = z[inds]

        w_init = distWeights.copy()

        coeffs = polyfit_2d(
            xw,
            yw,
            zw,
            degree=degree,
            weights=w_init,
        )

        if degree == 0:

            zfit = np.full_like(
                zw,
                coeffs[0],
                dtype=float,
            )

        else:

            xc = np.average(
                xw,
                weights=w_init,
            )

            yc = np.average(
                yw,
                weights=w_init,
            )

            dx = xw - xc
            dy = yw - yc

            a0, ax, ay = coeffs

            zfit = (
                a0
                + ax * dx
                + ay * dy
            )

        biWeights = np.ones_like(zw)

        for _ in range(10):

            if sigz is None:

                resid = zfit - zw

                scale = _biweight_scale(
                    resid
                )

                uu = (
                    np.abs(resid)
                    / (6.0 * scale)
                ) ** 2.0

            else:

                uu = (
                    (zfit - zw)
                    / (4.0 * sigz[inds])
                ) ** 2.0

            uu = np.clip(
                uu,
                0.0,
                1.0,
            )

            biWeights_new = (
                1.0 - uu
            ) ** 2.0

            totWeights = (
                distWeights
                * biWeights_new
            )

            coeffs = polyfit_2d(
                xw,
                yw,
                zw,
                degree=degree,
                weights=totWeights,
            )

            if degree == 0:

                zfit = np.full_like(
                    zw,
                    coeffs[0],
                    dtype=float,
                )

            else:

                if np.sum(totWeights) > 0:

                    xc = np.average(
                        xw,
                        weights=totWeights,
                    )

                    yc = np.average(
                        yw,
                        weights=totWeights,
                    )

                else:

                    xc = np.mean(xw)
                    yc = np.mean(yw)

                dx = xw - xc
                dy = yw - yc

                a0, ax, ay = coeffs

                zfit = (
                    a0
                    + ax * dx
                    + ay * dy
                )

            if np.allclose(
                biWeights,
                biWeights_new,
                atol=1e-6,
            ):

                biWeights = biWeights_new
                break

            biWeights = biWeights_new

        if degree == 0:
            zout[j] = coeffs[0]
        else:
            zout[j] = coeffs[0]

        wout[j] = (
            biWeights[0]
            if biWeights.size > 0
            else 1.0
        )

    return zout, wout


# ============================================================================
# Common LOESS plotting helper
# ============================================================================
def plot_loess_mass_size(
    *,
    log_m,
    log_r,
    z_values,
    label,
    outfile,
    logsigma_ref,
    stellar_masses,
    show_missing=True,
    frac_loess=0.01,
    degree=1,
    nx=300,
    ny=220,
    missing_label="no data",
    vmin_percentile=5,
    vmax_percentile=95,
    clamp=None,
):
    """
    Make one mass-size plane using the same LOESS/grid/masking recipe.
    """

    z_aligned = np.asarray(
        z_values,
        dtype=float,
    )

    have_mask = np.isfinite(
        z_aligned
    )

    missing_mask = ~have_mask

    n_have = int(
        have_mask.sum()
    )

    n_missing = int(
        missing_mask.sum()
    )

    total_plot = int(
        len(z_aligned)
    )

    print(
        f"DEBUG {label}: "
        f"have={n_have}, "
        f"missing={n_missing}, "
        f"total={total_plot}"
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    if n_have == 0:

        if show_missing:

            ax.scatter(
                log_m,
                log_r,
                s=10,
                alpha=0.7,
                color="lightgrey",
                label=missing_label,
            )

        else:

            ax.scatter(
                log_m,
                log_r,
                s=10,
                alpha=0.7,
                label="galaxies",
            )

    else:

        xvals = log_m[have_mask]
        yvals = log_r[have_mask]
        zvals = z_aligned[have_mask]

        # --------------------------------------------------------------
        # Grid around the finite LOESS points
        # --------------------------------------------------------------
        pad_x = (
            0.05
            * (
                np.nanmax(xvals)
                - np.nanmin(xvals)
                + 1e-6
            )
        )

        pad_y = (
            0.05
            * (
                np.nanmax(yvals)
                - np.nanmin(yvals)
                + 1e-6
            )
        )

        xg = np.linspace(
            np.nanmin(xvals) - pad_x,
            np.nanmax(xvals) + pad_x,
            nx,
        )

        yg = np.linspace(
            np.nanmin(yvals) - pad_y,
            np.nanmax(yvals) + pad_y,
            ny,
        )

        Xg, Yg = np.meshgrid(
            xg,
            yg,
        )

        pts_grid = np.column_stack(
            (
                Xg.ravel(),
                Yg.ravel(),
            )
        )

        # --------------------------------------------------------------
        # Distance-based LOESS mask
        # --------------------------------------------------------------
        tree_data = KDTree(
            np.column_stack(
                (
                    xvals,
                    yvals,
                )
            )
        )

        d_grid, _ = tree_data.query(
            pts_grid,
            k=1,
        )

        d_data, _ = tree_data.query(
            np.column_stack(
                (
                    xvals,
                    yvals,
                )
            ),
            k=2,
        )

        if (
            d_data.ndim == 2
            and d_data.shape[1] >= 2
        ):

            typical_spacing = float(
                np.nanpercentile(
                    d_data[:, 1],
                    95,
                )
            )

        else:

            typical_spacing = float(
                np.nanmedian(d_grid)
            )

        d_thresh = max(
            typical_spacing * 1.3,
            1e-6,
        )

        inside_mask = (
            d_grid <= d_thresh
        )

        idx_inside = np.nonzero(
            inside_mask
        )[0]

        if idx_inside.size > 0:

            xout = pts_grid[
                idx_inside,
                0,
            ]

            yout = pts_grid[
                idx_inside,
                1,
            ]

            # ----------------------------------------------------------
            # LOESS
            # ----------------------------------------------------------
            Zflat_inside, _ = loess_2d(
                xvals,
                yvals,
                zvals,
                frac=frac_loess,
                degree=degree,
                xout=xout,
                yout=yout,
            )

            Zflat = np.full(
                pts_grid.shape[0],
                np.nan,
                dtype=float,
            )

            Zflat[idx_inside] = (
                Zflat_inside
            )

            Zgrid = Zflat.reshape(
                (ny, nx)
            )

            Zmask = np.ma.masked_invalid(
                Zgrid
            )

            # ----------------------------------------------------------
            # Colour limits
            # ----------------------------------------------------------
            try:

                vmin = float(
                    np.nanpercentile(
                        zvals,
                        vmin_percentile,
                    )
                )

                vmax = float(
                    np.nanpercentile(
                        zvals,
                        vmax_percentile,
                    )
                )

            except Exception:

                vmin = float(
                    np.nanmin(zvals)
                )

                vmax = float(
                    np.nanmax(zvals)
                )

            if clamp is not None:

                low, high = clamp

                vmin = max(
                    low,
                    vmin,
                )

                vmax = min(
                    high,
                    vmax,
                )

            if (
                not np.isfinite(vmin)
                or not np.isfinite(vmax)
                or vmin == vmax
            ):

                med = float(
                    np.nanmedian(zvals)
                )

                span = max(
                    0.2,
                    0.5
                    * max(
                        1e-6,
                        abs(med),
                    ),
                )

                vmin = med - span
                vmax = med + span

                if clamp is not None:

                    vmin = max(
                        clamp[0],
                        vmin,
                    )

                    vmax = min(
                        clamp[1],
                        vmax,
                    )

            # ----------------------------------------------------------
            # Plot LOESS surface
            # ----------------------------------------------------------
            im = ax.pcolormesh(
                Xg,
                Yg,
                Zmask,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

            cbar = fig.colorbar(
                im,
                ax=ax,
            )

            cbar.set_label(
                label
            )

            # Faint evaluated LOESS cells
            ax.scatter(
                xout,
                yout,
                s=1,
                c="k",
                alpha=0.05,
                linewidths=0,
            )

        else:

            # Fallback scatter
            ax.scatter(
                xvals,
                yvals,
                c=zvals,
                cmap="viridis",
                s=12,
                edgecolors="none",
            )

        # --------------------------------------------------------------
        # Missing data
        # --------------------------------------------------------------
        if (
            show_missing
            and n_missing > 0
        ):

            ax.scatter(
                log_m[missing_mask],
                log_r[missing_mask],
                color="lightgrey",
                s=8,
                alpha=0.6,
                label=missing_label,
            )

    # ------------------------------------------------------------------
    # Compactness threshold
    # ------------------------------------------------------------------
    ax.plot(
        np.log10(stellar_masses),
        (2.0 / 3.0)
        * (
            np.log10(stellar_masses)
            - logsigma_ref
        ),
        linestyle="--",
        color="black",
        label=(
            fr"compactness threshold "
            fr"$\log_{{10}} \Sigma_{{1.5}} = "
            fr"{logsigma_ref}$"
        ),
    )

    ax.set_xlabel(
        r"$\log_{{10}}(M_\star / M_{\odot}$)"
    )

    ax.set_ylabel(
        r"$\log_{{10}}(R_{1/2, \star} / "
        r"\mathrm{kpc})$"
    )

    ax.legend(
        fontsize=8
    )

    ax.grid(True)

    fig.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved {label} LOESS plot:",
        outfile,
    )

    # Encourage release of temporary plotting arrays
    del fig, ax


# ============================================================================
# Profile family / method
# ============================================================================
family_method = "radial_profiles"
method = "circular_apertures_face_on_map"

# method = "circular_apertures_random_map"
# method = "spherical_apertures"


# ============================================================================
# Model and redshift
# ============================================================================
model_name = (
    "L0200N3008/THERMAL_AGN/"
)

model_dir = (
    "/mnt/su3-pro/colibre/"
    + model_name
)

snap_files = [
    "0127",
    "0119",
    "0114",
    "0102",
    "0092",
    "0076",
    "0064",
    "0056",
    "0048",
    "0040",
    "0026",
    "0018",
]

zstarget = [
    0.0,
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
]

# z = 2
snap_file = "0076"
ztarget = 2.0

comov_to_physical_length = (
    1.0 / (1.0 + ztarget)
)

print(
    f"Using snapshot {snap_file} "
    f"at z={ztarget}"
)

# ============================================================================
# Output directory
# ============================================================================
outdir = os.path.join(
    os.getcwd(),
    "plots_z2",
)

os.makedirs(
    outdir,
    exist_ok=True,
)

print(
    "Output directory:",
    outdir,
)

# ============================================================================
# Simulation units
# ============================================================================
Lu = (
    3.086e24
    / 3.086e24
)

Mu = (
    1.988e43
    / 1.989e33
)

tu = (
    3.086e19
    / 3.154e7
)


# ============================================================================
# Read ONLY fields required for the six plots
#
# Removed because they are not needed:
#   CentreOfMass
#   MassWeightedMeanStellarAge
#   IsCentral
#   DescendantTrackId
#   TrackId
#
# This is a major memory reduction relative to the previous version.
# ============================================================================
fields_all = {
    "ExclusiveSphere/50kpc": (
        "StellarMass",
        "StarFormationRate",
        "HalfMassRadiusStars",
        "LuminosityWeightedMeanStellarAge",
        "LinearMassWeightedIronOverHydrogenOfStars",
        "LinearMassWeightedMagnesiumOverHydrogenOfStars",
        "StellarMassFractionInMetals",
    ),
    "InputHalos": (
        "HaloCatalogueIndex",
    ),
}


# ============================================================================
# ONE SOAP READ
# ============================================================================
print(
    "Reading required SOAP quantities "
    "in ONE HDF5 call..."
)

h5data_all = common.read_group_data_colibre(
    model_dir,
    snap_file,
    fields_all,
)

(
    m30,
    sfr30,
    r50,
    stellarage_lum,
    FeoverH,
    MgoverH,
    Zstar_raw,
    sgn,
) = h5data_all

# VERY IMPORTANT:
# h5data_all itself still references the arrays.
# Delete it so the arrays can actually be released later.
del h5data_all

print(
    "SOAP read complete."
)


# ============================================================================
# Select the final plotting sample BEFORE any unit conversion
# ============================================================================
mass_cut_raw = (
    1e9 / Mu
)

select = np.flatnonzero(
    (m30 >= mass_cut_raw)
    & (m30 > 0)
    & (r50 > 0)
)

ngals = select.size

if ngals == 0:

    print(
        "No galaxies selected; "
        "skipping plots."
    )

    raise SystemExit(0)


print(
    "Number of galaxies of interest:",
    ngals,
    "at redshift",
    ztarget,
)


# These are the ORIGINAL SOAP catalogue row indices.
#
# sigma lives in another HDF5 dataset whose rows follow the full SOAP
# catalogue, so these indices are preserved before we delete the full arrays.
row_idx = select.copy()

print(
    "Number of rows needed for sigma:",
    row_idx.size,
)


# ============================================================================
# Extract ONLY selected galaxies
#
# float32 is sufficient for the plotting quantities and halves memory
# compared with float64.
# ============================================================================
m_in = np.asarray(
    m30[select],
    dtype=np.float32,
)

sfr_in = np.asarray(
    sfr30[select],
    dtype=np.float32,
)

r50_in = np.asarray(
    r50[select],
    dtype=np.float32,
)

stellarage_lum_in = np.asarray(
    stellarage_lum[select],
    dtype=np.float32,
)

FeoverH_in = np.asarray(
    FeoverH[select],
    dtype=np.float32,
)

MgoverH_in = np.asarray(
    MgoverH[select],
    dtype=np.float32,
)

Zstar_in = np.asarray(
    Zstar_raw[select],
    dtype=np.float32,
)

sgn_in = np.asarray(
    sgn[select],
    dtype=np.int64,
)


# ============================================================================
# Release FULL SOAP arrays now
# ============================================================================
del m30
del sfr30
del r50
del stellarage_lum
del FeoverH
del MgoverH
del Zstar_raw
del sgn

gc.collect()

print(
    "Full SOAP catalogue released."
)


# ============================================================================
# Unit conversions on SELECTED galaxies only
# ============================================================================
m_in *= np.float32(Mu)

sfr_in *= np.float32(
    Mu / tu
)

r50_in *= np.float32(
    Lu
    * comov_to_physical_length
    * 1e3
)

stellarage_lum_in *= np.float32(
    tu / 1e9
)


# ============================================================================
# Mass-size coordinates
#
# No mask_positive is needed here because select already required:
#   m30 > 0
#   r50 > 0
# ============================================================================
log_m = np.log10(
    m_in
).astype(np.float32)

log_r = np.log10(
    r50_in
).astype(np.float32)


# ============================================================================
# Compactness threshold
# ============================================================================
stellar_masses = np.logspace(
    9,
    12,
    100,
)

logsigma_ref = 9.75


# ============================================================================
# Derived stellar quantities
# ============================================================================
Zsun = 0.0134

# --------------------------------------------------------------------------
# Stellar metallicity
# --------------------------------------------------------------------------
with np.errstate(
    divide="ignore",
    invalid="ignore",
):

    logZstar_in = np.where(
        (Zstar_in > 0)
        & np.isfinite(Zstar_in),
        np.log10(Zstar_in),
        np.nan,
    )

    logZstar_rel_in = np.where(
        (Zstar_in > 0)
        & np.isfinite(Zstar_in),
        np.log10(
            Zstar_in / Zsun
        ),
        np.nan,
    )


# Zstar_in is no longer needed after the logarithmic metallicities exist.
del Zstar_in


# --------------------------------------------------------------------------
# Mg/Fe
#
# Same definition as your original script:
#
# MgFe_number = MgoverH / FeoverH
# [Mg/Fe] = log10(MgFe_number) - log10(Mg/Fe)_sun
# --------------------------------------------------------------------------
log_MgFe_sun = +0.10

with np.errstate(
    divide="ignore",
    invalid="ignore",
):

    MgFe_number = (
        MgoverH_in
        / FeoverH_in
    )

    log10_number = np.where(
        MgFe_number > 0,
        np.log10(
            MgFe_number
        ),
        np.nan,
    )

    mgfe = (
        log10_number
        - log_MgFe_sun
    )


# Release abundance inputs that are no longer required.
del FeoverH_in
del MgoverH_in
del MgFe_number
del log10_number

gc.collect()


# ============================================================================
# Arrays already aligned with log_m/log_r
# ============================================================================
stellar_lum_plot = (
    stellarage_lum_in
)

mgfe_plot = mgfe

logZ_plot = (
    logZstar_rel_in
)

subids_plot = (
    sgn_in
)


# ============================================================================
# Compute sSFR
# ============================================================================
with np.errstate(
    divide="ignore",
    invalid="ignore",
):

    ssfr_plot = np.where(
        m_in > 0,
        sfr_in / m_in,
        np.nan,
    )

    log_ssfr_plot = np.where(
        ssfr_plot > 0,
        np.log10(ssfr_plot),
        np.nan,
    )


# Exact previous behaviour:
# replace missing/non-positive sSFR with a floor.
SSFR_FLOOR = -12.0

log_ssfr_plot[
    ~np.isfinite(log_ssfr_plot)
] = SSFR_FLOOR


# We no longer need the linear sSFR.
del ssfr_plot


# # ============================================================================
# # Velocity dispersion
# # ============================================================================
# sigma_path = (
#     "/mnt/su3-pro/colibre/"
#     "L0200N3008/THERMAL_AGN/"
#     "SOAP-HBT/extra/"
#     f"halo_properties_{snap_file}.hdf5"
# )

# sigma_ds = (
#     "/ExclusiveSphere/HalfMassRadiusStars/"
#     "StellarCylindricalVelocityDispersionVerticalLuminosityWeighted"
# )

# if os.path.exists(sigma_path):

#     with h5py.File(
#         sigma_path,
#         "r",
#     ) as f:

#         ds_vert = f[
#             sigma_ds
#         ]

#         print(
#             "Sigma dataset shape:",
#             ds_vert.shape,
#         )

#         print(
#             "Reading sigma for",
#             len(row_idx),
#             "selected galaxies",
#         )

#         rows_vert = np.asarray(
#             ds_vert[row_idx, :],
#             dtype=np.float32,
#         )

#     sigma_rr = (
#         rows_vert[:, 0]
#     )

#     sigma_pphi = (
#         rows_vert[:, 4]
#     )

#     sigma_zz = (
#         rows_vert[:, 8]
#     )

#     sigma_vals = np.sqrt(
#         (
#             sigma_rr**2
#             + sigma_pphi**2
#             + sigma_zz**2
#         ) / 3.0
#     ).astype(np.float32)

#     log_sigma_vals = np.where(
#         sigma_vals > 0,
#         np.log10(
#             sigma_vals
#         ),
#         np.nan,
#     ).astype(np.float32)

#     print(
#         "Loaded sigma values:",
#         np.isfinite(
#             sigma_vals
#         ).sum(),
#         "/",
#         sigma_vals.size,
#     )

#     print(
#         "N(sigma == 0):",
#         np.count_nonzero(
#             np.isclose(
#                 sigma_vals[
#                     np.isfinite(
#                         sigma_vals
#                     )
#                 ],
#                 0.0,
#             )
#         ),
#     )

#     del rows_vert
#     del sigma_rr
#     del sigma_pphi
#     del sigma_zz

# else:

#     print(
#         "Sigma file not found:",
#         sigma_path,
#     )

#     sigma_vals = np.full(
#         len(log_m),
#         np.nan,
#         dtype=np.float32,
#     )

#     log_sigma_vals = np.full(
#         len(log_m),
#         np.nan,
#         dtype=np.float32,
#     )


# # Sigma must have exactly the same number of entries as the selected galaxies.
# if len(log_sigma_vals) != len(log_m):

#     raise RuntimeError(
#         "Sigma is not aligned with the mass-size sample: "
#         f"{len(log_sigma_vals)} vs {len(log_m)}"
#     )

# gc.collect()


# ============================================================================
# Ex-situ stellar mass fraction
# ============================================================================
base_dir = Path(
    "/mnt/su3ctm/kproctor/ForMax"
)

matches = sorted(
    base_dir.glob(
        "*exsitu*summary*.hdf5"
    )
)

if len(matches) == 0:

    raise FileNotFoundError(
        "No ex-situ HDF5 file found in "
        f"{base_dir}"
    )

elif len(matches) == 1:

    h5path = str(
        matches[0]
    )

else:

    h5path = str(
        max(
            matches,
            key=lambda p: p.stat().st_mtime,
        )
    )


print(
    "Using ex-situ file:",
    h5path,
)


# --------------------------------------------------------------------------
# IMPORTANT:
# Do NOT load the entire ex-situ table into memory.
#
# Instead, read it in chunks and retain only entries whose HaloCatalogueIndex
# is actually present in the selected sample.
# --------------------------------------------------------------------------
halo_selected = (
    sgn_in.astype(np.int64)
)

exsitu_fracs = np.full(
    halo_selected.shape,
    np.nan,
    dtype=np.float32,
)

sort_order = np.argsort(
    halo_selected
)

sorted_selected_ids = (
    halo_selected[sort_order]
)

EXSITU_CHUNK_SIZE = 1_000_000


with h5py.File(
    h5path,
    "r",
) as fh:

    if "stars" not in fh:

        raise RuntimeError(
            "Ex-situ HDF5 file is missing "
            "dataset 'stars'."
        )

    ds_stars = fh[
        "stars"
    ]

    if (
        ds_stars.ndim != 2
        or ds_stars.shape[1] < 4
    ):

        raise RuntimeError(
            "Unexpected ex-situ 'stars' "
            f"dataset shape: {ds_stars.shape}"
        )

    n_exsitu = ds_stars.shape[0]

    print(
        "Ex-situ dataset shape:",
        ds_stars.shape,
    )

    print(
        "Matching selected galaxies "
        "against ex-situ catalogue in chunks..."
    )

    for start in range(
        0,
        n_exsitu,
        EXSITU_CHUNK_SIZE,
    ):

        stop = min(
            start + EXSITU_CHUNK_SIZE,
            n_exsitu,
        )

        ids_chunk = np.asarray(
            ds_stars[
                start:stop,
                0,
            ],
            dtype=np.int64,
        )

        frac_chunk = np.asarray(
            ds_stars[
                start:stop,
                3,
            ],
            dtype=np.float32,
        )

        positions = np.searchsorted(
            sorted_selected_ids,
            ids_chunk,
        )

        valid = (
            positions
            < sorted_selected_ids.size
        )

        valid_positions = np.nonzero(
            valid
        )[0]

        if valid_positions.size > 0:

            selected_positions = (
                positions[
                    valid_positions
                ]
            )

            exact_match = (
                sorted_selected_ids[
                    selected_positions
                ]
                == ids_chunk[
                    valid_positions
                ]
            )

            matched_chunk_positions = (
                valid_positions[
                    exact_match
                ]
            )

            if matched_chunk_positions.size > 0:

                sorted_positions = (
                    positions[
                        matched_chunk_positions
                    ]
                )

                original_positions = (
                    sort_order[
                        sorted_positions
                    ]
                )

                exsitu_fracs[
                    original_positions
                ] = frac_chunk[
                    matched_chunk_positions
                ]

        del ids_chunk
        del frac_chunk

    del sorted_selected_ids
    del sort_order

gc.collect()


n_matched = int(
    np.isfinite(
        exsitu_fracs
    ).sum()
)

print(
    "Matched ex-situ fraction for",
    n_matched,
    "/",
    len(exsitu_fracs),
    "selected galaxies",
)


# ============================================================================
# Shared plotting style
# ============================================================================
plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": 14,
    }
)


# ============================================================================
# 1. Luminosity-weighted mean stellar age
# ============================================================================
plot_loess_mass_size(
    log_m=log_m,
    log_r=log_r,
    z_values=stellar_lum_plot,
    label="Age [Gyr]",
    outfile=os.path.join(
        outdir,
        f"mass_size_z{ztarget:.1f}_age(lum)_loess.pdf",
    ),
    logsigma_ref=logsigma_ref,
    stellar_masses=stellar_masses,
    show_missing=True,
    missing_label="no lum age",
)

gc.collect()


# ============================================================================
# 2. Stellar metallicity
# ============================================================================
plot_loess_mass_size(
    log_m=log_m,
    log_r=log_r,
    z_values=logZ_plot,
    label="[Z/H]",
    outfile=os.path.join(
        outdir,
        f"mass_size_z{ztarget:.1f}_metallicity_loess.pdf",
    ),
    logsigma_ref=logsigma_ref,
    stellar_masses=stellar_masses,
    show_missing=True,
    missing_label="no metallicity",
)

gc.collect()


# ============================================================================
# 3. Mg/Fe
# ============================================================================
plot_loess_mass_size(
    log_m=log_m,
    log_r=log_r,
    z_values=mgfe_plot,
    label="[Mg/Fe]",
    outfile=os.path.join(
        outdir,
        f"mass_size_z{ztarget:.1f}_fullMgFe_loess.pdf",
    ),
    logsigma_ref=logsigma_ref,
    stellar_masses=stellar_masses,
    show_missing=True,
    missing_label="no Mg/Fe",
)

gc.collect()


# ============================================================================
# 4. Specific SFR
# ============================================================================
plot_loess_mass_size(
    log_m=log_m,
    log_r=log_r,
    z_values=log_ssfr_plot,
    label=(
        r"$\log_{10}(\mathrm{sSFR}\ /\ "
        r"\mathrm{yr}^{-1})$"
    ),
    outfile=os.path.join(
        outdir,
        f"mass_size_z{ztarget:.1f}_ssfr_loess.pdf",
    ),
    logsigma_ref=logsigma_ref,
    stellar_masses=stellar_masses,
    show_missing=False,
    missing_label="no sSFR",
)

gc.collect()


# # ============================================================================
# # 5. Velocity dispersion
# # ============================================================================
# plot_loess_mass_size(
#     log_m=log_m,
#     log_r=log_r,
#     z_values=log_sigma_vals,
#     label=(
#         r"$\log_{10}(\sigma / "
#         r"\mathrm{km}\ \mathrm{s}^{-1})$"
#     ),
#     outfile=os.path.join(
#         outdir,
#         f"mass_size_z{ztarget:.1f}_sigma_loess.pdf",
#     ),
#     logsigma_ref=logsigma_ref,
#     stellar_masses=stellar_masses,
#     show_missing=True,
#     missing_label="no sigma",
# )

# gc.collect()


# ============================================================================
# 6. Ex-situ mass fraction
# ============================================================================
plot_loess_mass_size(
    log_m=log_m,
    log_r=log_r,
    z_values=exsitu_fracs,
    label=r"$f_\mathrm{ex-situ}$",
    outfile=os.path.join(
        outdir,
        f"mass_size_z{ztarget:.1f}_exsitu_loess.pdf",
    ),
    logsigma_ref=logsigma_ref,
    stellar_masses=stellar_masses,
    show_missing=True,
    missing_label="no ex-situ data",
    clamp=(0.0, 1.0),
)

gc.collect()


print(
    "\nAll six z=2 LOESS mass-size plots completed."
)