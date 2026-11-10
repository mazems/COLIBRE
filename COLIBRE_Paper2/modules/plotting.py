#!/usr/bin/env python3
"""COLIBRE Paper 2: exploratory plots from saved evolutionary properties.

Run from COLIBRE_Paper2:
    python3 -m modules.plotting --dry-run
    python3 -m modules.plotting --write
    python3 -m modules.plotting --write --overwrite
    python3 -m modules.plotting --write --limit 20 --output-dir outputs/figures/test20

No SOAP/ASCII files are read. Population labels are the fixed z=0 labels.
Medians at high redshift describe *detected* progenitors, not a complete cohort.
The z=0 compactness cut is shown as a reference, NOT a red-nugget definition.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ------------------------------------------------------------
# Global publication-style plotting configuration
# ------------------------------------------------------------

plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (8, 6),
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

from config import OUTPUT, REFERENCE_SAMPLE

QUANTITIES = {
    'mass': ('log10_stellar_mass', r'$\log_{10}(M_\star/\mathrm{M}_\odot)$', 'stellar_mass_evolution'),
    'size': ('log10_rhalf', r'$\log_{10}(R_{1/2,\star}/\mathrm{kpc})$', 'size_evolution'),
    'compactness': ('compactness_sigma_1p5', r'$\log_{10}\Sigma_{1.5}$', 'compactness_evolution'),
    'ssfr': ('log10_ssfr_plot', r'$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$ (display floor)', 'ssfr_evolution'),
}
COLORS = {'SRG': 'C2', 'SAG_noncompact': 'C1', 'compact_nonrelic': '0.4', 'noncompact_nonrelic': 'C0'}


def project_path(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else Path(__file__).resolve().parents[1] / path).resolve()


def load_history(path=None):
    path = project_path(path or Path(OUTPUT['tracks_dir']) / 'evolutionary_properties.csv')
    if not path.is_file():
        raise FileNotFoundError(f'Missing evolutionary catalogue: {path}. Run modules.properties --write first.')
    df = pd.read_csv(path, dtype={'track_id': 'int64'})
    needed = {'track_id', 'population', 'redshift', 'log10_stellar_mass', 'log10_rhalf',
              'compactness_sigma_1p5', 'log10_ssfr_plot'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'Evolutionary catalogue missing required physical/derived columns: {sorted(missing)}')
    if df.empty or df[['track_id', 'redshift']].isna().any().any():
        raise ValueError('Empty catalogue or missing track IDs/redshifts')
    if df.duplicated(['track_id', 'redshift']).any():
        raise ValueError('Duplicate (track_id, redshift) entries; cannot plot ambiguous histories')
    if not np.isfinite(df['redshift']).all() or (df['redshift'] < 0).any():
        raise ValueError('Invalid redshift values')
    if df['population'].isna().any() or df.groupby('track_id')['population'].nunique().gt(1).any():
        raise ValueError('Population must be fixed for each reference TrackID')
    # Cross-check derived compactness against mass and size to catch stale files.
    a = df['log10_stellar_mass'].to_numpy(dtype=float)
    b = df['log10_rhalf'].to_numpy(dtype=float)
    c = df['compactness_sigma_1p5'].to_numpy(dtype=float)
    expected = a - float(REFERENCE_SAMPLE['compactness_exponent']) * b
    good = np.isfinite(expected) & np.isfinite(c)
    if good.any() and not np.allclose(expected[good], c[good], atol=1e-6, rtol=0):
        raise ValueError('Compactness does not match mass and radius; check conversions')
    return df.sort_values(['track_id', 'redshift'], ascending=[True, False]).reset_index(drop=True)

def load_quenching_epochs(path=None):
    """
    Load the revised matching epochs from quenching_history.py.

    The starting epoch is the last observed star-forming snapshot
    immediately preceding the final accepted quenching transition.

    Only galaxies with a bracketed final quenching transition
    are included.
    """

    path = project_path(
        path or Path(OUTPUT['root'])
        / 'quenching_history'
        / 'quenching_history_per_galaxy.csv'
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing quenching-history catalogue: {path}. "
            "Run modules.quenching_history --write first."
        )

    df = pd.read_csv(
        path,
        dtype={'track_id': 'int64'}
    )

    required = {
        'track_id',
        'population',
        'z_final_quenching_prior_star_forming',
        'z_final_quenching',
        'final_quenching_status',
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Quenching catalogue missing columns: {sorted(missing)}"
        )

    if df['track_id'].duplicated().any():
        raise ValueError(
            "Duplicate TrackIDs in quenching-history catalogue."
        )

    df = df[
        df['final_quenching_status']
        == 'final_quenching_bracketed'
    ].copy()

    df = df.rename(columns={
        'z_final_quenching_prior_star_forming': 'z_start'
    })

    df = df[
        np.isfinite(df['z_start'])
        & np.isfinite(df['z_final_quenching'])
    ].copy()

    return df

def load_srg_exsitu_fractions(reference_file=None, exsitu_file=None):
    """
    Load z=0 ex-situ stellar mass fractions for SRGs.

    Matching:
        z=0 TrackID -> HaloCatalogueIndex -> ex-situ HDF5

    The ex-situ fractions are cumulative z=0 measurements,
    NOT measurements at the individual quenching epochs.

    Returns
    -------
    DataFrame with:
        track_id
        exsitu_fraction
        exsitu_group
    """

    import h5py
    from config import INPUTS

    # --------------------------------------------------
    # 1. File paths
    # --------------------------------------------------

    reference_file = project_path(
        reference_file
        or Path(OUTPUT['root'])
        / 'reference_catalogues'
        / 'reference_galaxies_z0_all.csv'
    )

    exsitu_file = project_path(
        exsitu_file
        or INPUTS['exsitu_z0_hdf5']
    )

    # --------------------------------------------------
    # 2. Read z=0 SRG reference catalogue
    # --------------------------------------------------

    ref = pd.read_csv(
        reference_file,
        dtype={'track_id': 'int64'}
    )

    ref = ref[
        ref['population'] == 'SRG'
    ].copy()

    if ref['track_id'].duplicated().any():
        raise ValueError(
            "Duplicate SRG TrackIDs in reference catalogue."
        )

    if 'halo_catalogue_index' not in ref.columns:
        raise ValueError(
            "Missing halo_catalogue_index in reference catalogue."
        )

    ref['halo_index'] = pd.to_numeric(
        ref['halo_catalogue_index'],
        errors='raise'
    ).astype('int64')

    # --------------------------------------------------
    # 3. Read Paper 1 ex-situ HDF5
    # --------------------------------------------------

    with h5py.File(exsitu_file, 'r') as fh:

        if 'stars' not in fh:
            raise KeyError(
                "Ex-situ HDF5 missing dataset 'stars'."
            )

        data = np.asarray(fh['stars'])

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(
            "Unexpected ex-situ dataset structure."
        )

    # --------------------------------------------------
    # 4. Identify matching column
    # --------------------------------------------------

    candidate_cols = (0, 1, 2)

    overlaps = []

    for c in candidate_cols:

        ids = data[:, c].astype(np.int64)

        overlap = np.intersect1d(
            ids,
            ref['halo_index'].to_numpy()
        ).size

        overlaps.append(overlap)

    print("\nEx-situ identifier overlaps:")
    for col, overlap in zip(candidate_cols, overlaps):
        print(f"  Column {col}: {overlap} matches")

    if max(overlaps) == 0:
        raise ValueError(
            "No matching identifiers found in ex-situ HDF5."
        )

    if overlaps.count(max(overlaps)) > 1:
        raise ValueError(
            "Ambiguous ex-situ identifier column: "
            "multiple columns have identical maximum overlap."
        )

    keycol = candidate_cols[int(np.argmax(overlaps))]

    print(
        "Selected ex-situ identifier column:",
        keycol
    )

    ids = data[:, keycol].astype(np.int64)

    fractions = data[:, 3].astype(float)

    if pd.Series(ids).duplicated().any():
        raise ValueError(
            "Duplicate ex-situ identifiers detected."
        )

    # --------------------------------------------------
    # 5. Match fractions to SRGs
    # --------------------------------------------------

    lookup = pd.Series(
        fractions,
        index=ids
    )

    ref['exsitu_fraction'] = (
        ref['halo_index'].map(lookup)
    )

    # --------------------------------------------------
    # 6. Classify ex-situ groups
    # --------------------------------------------------

    def classify(f):

        if not np.isfinite(f):
            return 'unknown'

        if f < 0 or f > 1:
            return 'unknown'

        if f < 0.1:
            return 'low'

        if f < 0.4:
            return 'mid'

        return 'high'

    ref['exsitu_group'] = (
        ref['exsitu_fraction'].apply(classify)
    )

    # --------------------------------------------------
    # 7. Diagnostics
    # --------------------------------------------------

    print("\nSRG ex-situ matching report")
    print("=" * 50)

    print("Total SRGs:", len(ref))

    print(
        "Valid ex-situ fractions:",
        ref['exsitu_group'].ne('unknown').sum()
    )

    print(
        "Unknown ex-situ fractions:",
        ref['exsitu_group'].eq('unknown').sum()
    )

    print("\nGroup counts:")
    print(ref['exsitu_group'].value_counts())

    return ref[
        [
            'track_id',
            'exsitu_fraction',
            'exsitu_group',
        ]
    ].copy()

# ============================================================
# LOAD MATCHED CONTROL EVOLUTIONARY HISTORIES
# ============================================================

def load_matched_control_histories(
    matching_file=None,
    descendant_dir=None,
):
    """
    Load pair-specific control histories produced by descendants.py.

    Matching epoch:
        Original first compact-and-quenched epoch of each SRG.

    Grouping:
        z=0 ex-situ stellar mass fraction of the reference SRG.

    The same control can appear in multiple matched pairs.
    """

    # --------------------------------------------------------
    # 1. Resolve input paths
    # --------------------------------------------------------

    matching_file = project_path(
        matching_file
        or Path(OUTPUT['root'])
        / 'controls'
        / 'matched_control_candidates.csv'
    )

    descendant_dir = project_path(
        descendant_dir
        or Path(OUTPUT['root'])
        / 'controls'
        / 'descendants'
    )

    history_file = (
        descendant_dir
        / 'control_pair_evolutionary_properties.csv'
    )

    status_file = (
        descendant_dir
        / 'control_descendant_pairs.csv'
    )

    for path in (matching_file, history_file, status_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required control catalogue missing: {path}"
            )

    # --------------------------------------------------------
    # 2. Load original matching catalogue
    # --------------------------------------------------------

    pairs = pd.read_csv(
        matching_file,
        dtype={
            'reference_track_id': 'int64',
            'control_track_id': 'int64',
        },
    )

    required_pairs = {
        'reference_track_id',
        'control_track_id',
        'z_match',
    }

    missing = required_pairs - set(pairs.columns)

    if missing:
        raise ValueError(
            f"Matching catalogue missing: {sorted(missing)}"
        )

    if pairs.duplicated(
        ['reference_track_id', 'control_track_id', 'z_match']
    ).any():
        raise ValueError(
            "Duplicate matched-pair assignments."
        )

    # --------------------------------------------------------
    # 3. Attach SRG ex-situ groups
    # --------------------------------------------------------

    exsitu = load_srg_exsitu_fractions()

    exsitu = exsitu.rename(columns={
        'track_id': 'reference_track_id',
        'exsitu_fraction': 'reference_exsitu_fraction',
        'exsitu_group': 'reference_exsitu_group',
    })

    pairs = pairs.merge(
        exsitu,
        on='reference_track_id',
        how='left',
        validate='many_to_one',
    )

    pairs['reference_exsitu_group'] = (
        pairs['reference_exsitu_group']
        .fillna('unknown')
    )

    # --------------------------------------------------------
    # 4. Load pair-specific evolutionary histories
    # --------------------------------------------------------

    history = pd.read_csv(
        history_file,
        dtype={
            'reference_track_id': 'int64',
            'control_track_id': 'int64',
        },
    )

    required_history = {
        'reference_track_id',
        'control_track_id',
        'z_match',
        'redshift',
        'log10_stellar_mass',
        'log10_rhalf',
    }

    missing = required_history - set(history.columns)

    if missing:
        raise ValueError(
            f"Control histories missing: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # 5. Attach pair-specific ex-situ groups
    # --------------------------------------------------------

    keys = [
        'reference_track_id',
        'control_track_id',
        'z_match',
    ]

    history = history.merge(
        pairs[
            keys
            + [
                'reference_exsitu_fraction',
                'reference_exsitu_group',
            ]
        ],
        on=keys,
        how='left',
        validate='many_to_one',
    )

    # --------------------------------------------------------
    # 6. Attach z=0 persistence status
    # --------------------------------------------------------

    status = pd.read_csv(
        status_file,
        dtype={
            'reference_track_id': 'int64',
            'control_track_id': 'int64',
        },
    )

    status_required = {
        'reference_track_id',
        'control_track_id',
        'z_match',
        'trackid_present_z0_ascii',
    }

    missing = status_required - set(status.columns)

    if missing:
        raise ValueError(
            f"Control status catalogue missing: {sorted(missing)}"
        )

    status = status[
        list(status_required)
    ].copy()

    # Explicit boolean parsing prevents strings such as
    # "False" from being interpreted as True.

    status['trackid_present_z0_ascii'] = (
        status['trackid_present_z0_ascii']
        .astype('string')
        .str.lower()
        .map({
            'true': True,
            'false': False,
        })
    )

    if status['trackid_present_z0_ascii'].isna().any():
        raise ValueError(
            "Invalid or missing z=0 persistence flags."
        )

    history = history.merge(
        status,
        on=keys,
        how='left',
        validate='many_to_one',
    )

    if history['trackid_present_z0_ascii'].isna().any():
        raise ValueError(
            "Some control histories lack persistence information."
        )

    # --------------------------------------------------------
    # 7. Validate evolutionary histories
    # --------------------------------------------------------

    if history.duplicated(
        keys + ['redshift']
    ).any():
        raise ValueError(
            "Duplicate control measurements at the same epoch."
        )

    if (
        history['redshift']
        > history['z_match'] + 1e-6
    ).any():
        raise ValueError(
            "Control history contains snapshots before matching."
        )

    if history[
        'reference_exsitu_group'
    ].isna().any():
        raise ValueError(
            "Some controls lack reference SRG ex-situ groups."
        )

    # --------------------------------------------------------
    # 8. Diagnostics
    # --------------------------------------------------------

    print("\nMATCHED CONTROL HISTORY REPORT")
    print("=" * 60)

    print("Matched pairs:", len(pairs))

    print(
        "Unique controls:",
        pairs['control_track_id'].nunique()
    )

    print(
        "Pair-history rows:",
        len(history)
    )

    print("\nReference ex-situ groups:")

    print(
        pairs['reference_exsitu_group']
        .value_counts()
    )

    return pairs, history

def build_post_quenching_catalogue(history, quenching):
    """
    Construct one row per SRG with structural changes between
    the last observed star-forming snapshot and z=0.

    Uses existing physical quantities from evolutionary_properties.csv.
    """

    rows = []
    tracks = []

    q = quenching[
        quenching['population'] == 'SRG'
    ].copy()

    for _, event in q.iterrows():

        tid = int(event['track_id'])

        z_start = float(event['z_start'])
        z_final_q = float(event['z_final_quenching'])

        g = history[
            history['track_id'] == tid
        ].sort_values(
            'redshift',
            ascending=False
        ).copy()

        if g.empty:
            continue

        # --------------------------------------------------
        # Identify exact starting and final snapshots
        # --------------------------------------------------

        start = g[
            np.isclose(
                g['redshift'],
                z_start,
                atol=1e-6,
                rtol=0,
            )
        ]

        final_q = g[
            np.isclose(
                g['redshift'],
                z_final_q,
                atol=1e-6,
                rtol=0,
            )
        ]

        z0 = g[
            np.isclose(
                g['redshift'],
                0.0,
                atol=1e-6,
                rtol=0,
            )
        ]

        if len(start) != 1 or len(z0) != 1:
            continue

        a = start.iloc[0]
        b = z0.iloc[0]

        required = [
            a['log10_stellar_mass'],
            a['log10_rhalf'],
            b['log10_stellar_mass'],
            b['log10_rhalf'],
        ]

        if not np.all(np.isfinite(required)):
            continue

        # --------------------------------------------------
        # Structural changes
        # --------------------------------------------------

        dlogm = (
            b['log10_stellar_mass']
            - a['log10_stellar_mass']
        )

        dlogr = (
            b['log10_rhalf']
            - a['log10_rhalf']
        )

        dcompact = (
            b['compactness_sigma_1p5']
            - a['compactness_sigma_1p5']
        )

        rows.append({
            'track_id': tid,
            'population': 'SRG',

            'z_start': z_start,
            'z_final_quenching': z_final_q,

            'logM_start': a['log10_stellar_mass'],
            'logR_start': a['log10_rhalf'],
            'compactness_start': a['compactness_sigma_1p5'],

            'logM_z0': b['log10_stellar_mass'],
            'logR_z0': b['log10_rhalf'],
            'compactness_z0': b['compactness_sigma_1p5'],

            'delta_logM': dlogm,
            'delta_logR': dlogr,
            'delta_compactness': dcompact,

            'compact_at_start': (
                a['compactness_sigma_1p5']
                >= REFERENCE_SAMPLE['compactness_threshold']
            ),
        })

        # --------------------------------------------------
        # Retain only post-start evolutionary history
        # --------------------------------------------------

        gg = g[
            (g['redshift'] <= z_start + 1e-6)
            & (g['redshift'] >= 0)
        ].copy()

        gg['z_start'] = z_start
        gg['z_final_quenching'] = z_final_q

        tracks.append(gg)

    summary = pd.DataFrame(rows)

    if tracks:
        tracks = pd.concat(
            tracks,
            ignore_index=True
        )
    else:
        tracks = pd.DataFrame()

    return summary, tracks

def plot_post_quenching_mass_size(
    tracks,
    summary,
    outdir,
    formats=('png', 'pdf'),
):
    """
    Individual SRG mass-size tracks from each galaxy's
    last star-forming snapshot to z=0.
    """

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    for tid, g in tracks.groupby('track_id'):

        g = g.sort_values(
            'redshift',
            ascending=False
        )

        x = g['log10_stellar_mass'].to_numpy(float)
        y = g['log10_rhalf'].to_numpy(float)

        good = np.isfinite(x) & np.isfinite(y)

        if good.sum() < 2:
            continue

        x = x[good]
        y = y[good]

        # Evolutionary trajectory
        ax.plot(
            x,
            y,
            color='C2',
            lw=1.0,
            alpha=0.25,
            rasterized=True,
            zorder=2,
        )

        # Starting point: last star-forming snapshot
        ax.scatter(
            x[0],
            y[0],
            s=35,
            facecolors='white',
            edgecolors='C2',
            linewidths=1.0,
            zorder=5,
        )

        # Present-day position
        ax.scatter(
            x[-1],
            y[-1],
            s=35,
            facecolors='C2',
            edgecolors='black',
            linewidths=0.6,
            zorder=6,
        )

        # Direction arrow
        if len(x) >= 2:
            ax.annotate(
                '',
                xy=(x[-1], y[-1]),
                xytext=(x[-2], y[-2]),
                arrowprops=dict(
                    arrowstyle='->',
                    color='C2',
                    lw=1.0,
                    alpha=0.6,
                ),
            )

    # Compactness threshold
    xmin, xmax = ax.get_xlim()

    xx = np.linspace(
        xmin,
        xmax,
        300
    )

    cut = float(
        REFERENCE_SAMPLE['compactness_threshold']
    )

    exponent = float(
        REFERENCE_SAMPLE['compactness_exponent']
    )

    ax.plot(
        xx,
        (xx - cut) / exponent,
        '--',
        color='black',
        lw=1.3,
        label='Compactness threshold',
    )

    ax.set_xlabel(
        r'$\log_{10}(M_\star/\mathrm{M}_\odot)$'
    )

    ax.set_ylabel(
        r'$\log_{10}(R_{1/2,\star}/\mathrm{kpc})$'
    )

    ax.set_title(
        'SRG evolution after the final star-forming episode'
    )

    ax.grid(alpha=0.2)

    ax.legend(frameon=False)

    fig.tight_layout()

    return _save(
        fig,
        outdir / 'SRG_mass_size_post_quenching',
        formats,
    )

def plot_post_quenching_mass_size_exsitu_panels(
    tracks,
    summary,
    outdir,
    formats=('png', 'pdf'),
    epoch_mode='lastSF',
):
    """
    Three-panel SRG mass-size evolution split by z=0
    ex-situ stellar mass fraction.

    Open circles:
        Last star-forming snapshot before final quenching.

    Diamonds:
        First snapshot of final quenched phase.

    Filled circles:
        Present-day galaxy position.

    Arrows:
        Direction of evolution towards z=0.
    """

    from matplotlib.lines import Line2D

    # --------------------------------------------------
    # Select plotting conventions for the reference epoch
    # --------------------------------------------------

    if epoch_mode == 'lastSF':

        start_label = 'Last star-forming snapshot'

        show_final_quenching = True

        output_stem = (
            'SRG_mass_size_post_quenching_exsitu_panels'
        )

    elif epoch_mode == 'firstCQ':

        start_label = 'First compact-and-quenched snapshot'

        show_final_quenching = False

        output_stem = (
            'SRG_mass_size_firstCQ_exsitu_panels'
        )

    else:

        raise ValueError(
            f"Unknown epoch_mode: {epoch_mode}"
        )

    # --------------------------------------------------
    # 1. Ex-situ groups
    # --------------------------------------------------

    groups = [
        ('low', 'C2', r'$f_{\rm exsitu}<0.1$'),
        ('mid', 'C1', r'$0.1\leq f_{\rm exsitu}<0.4$'),
        ('high', 'C4', r'$f_{\rm exsitu}\geq0.4$'),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19, 6.5),
        sharex=True,
        sharey=True,
    )

    # --------------------------------------------------
    # 2. Common axis limits
    # --------------------------------------------------

    finite = (
        np.isfinite(tracks['log10_stellar_mass'])
        & np.isfinite(tracks['log10_rhalf'])
    )

    x_all = tracks.loc[
        finite,
        'log10_stellar_mass'
    ].to_numpy(float)

    y_all = tracks.loc[
        finite,
        'log10_rhalf'
    ].to_numpy(float)

    if len(x_all) == 0:
        raise ValueError(
            "No finite mass-size measurements available."
        )

    xmin, xmax = 9.0, 12.0
    ymin, ymax = -1.0, 1.8

    # --------------------------------------------------
    # 3. Plot individual trajectories
    # --------------------------------------------------

    for ax, (group, colour, label) in zip(
        axes,
        groups,
    ):

        selected = summary[
            summary['exsitu_group'] == group
        ]

        selected_ids = set(
            selected['track_id'].astype(int)
        )

        event_lookup = selected.set_index('track_id')

        for tid, g in tracks.groupby('track_id'):

            if tid not in selected_ids:
                continue

            g = g.sort_values(
                'redshift',
                ascending=False
            )

            x = g['log10_stellar_mass'].to_numpy(float)
            y = g['log10_rhalf'].to_numpy(float)
            z = g['redshift'].to_numpy(float)

            good = np.isfinite(x) & np.isfinite(y)

            if good.sum() < 2:
                continue

            # Preserve gaps rather than connecting invalid points.
            x_plot = np.where(good, x, np.nan)
            y_plot = np.where(good, y, np.nan)

            ax.plot(
                x_plot,
                y_plot,
                color=colour,
                lw=0.9,
                alpha=0.30,
                rasterized=True,
                zorder=2,
            )

            # ------------------------------------------
            # Arrows between valid adjacent snapshots
            # ------------------------------------------

            for i in range(len(x) - 1):

                if not (good[i] and good[i + 1]):
                    continue

                ax.annotate(
                    '',
                    xy=(x[i + 1], y[i + 1]),
                    xytext=(x[i], y[i]),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=colour,
                        lw=0.7,
                        alpha=0.30,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=3,
                )

            # ------------------------------------------
            # Last star-forming snapshot
            # ------------------------------------------

            z_start = float(
                event_lookup.loc[tid, 'z_start']
            )

            idx_start = np.flatnonzero(
                np.isclose(z, z_start, atol=1e-6, rtol=0)
                & good
            )

            if len(idx_start) == 1:

                j = idx_start[0]

                ax.scatter(
                    x[j],
                    y[j],
                    s=40,
                    facecolors='white',
                    edgecolors=colour,
                    linewidths=1.1,
                    zorder=6,
                )

            # ------------------------------------------
            # Final quenching snapshot (lastSF mode only)
            # ------------------------------------------

            if show_final_quenching:

                z_final_q = float(
                    event_lookup.loc[tid, 'z_final_quenching']
                )

                idx_q = np.flatnonzero(
                    np.isclose(
                        z,
                        z_final_q,
                        atol=1e-6,
                        rtol=0,
                    )
                    & good
                )

                if len(idx_q) == 1:

                    j = idx_q[0]

                    ax.scatter(
                        x[j],
                        y[j],
                        marker='D',
                        s=27,
                        facecolors=colour,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=7,
                    )

            # ------------------------------------------
            # Present-day position
            # ------------------------------------------

            idx_z0 = np.flatnonzero(
                np.isclose(z, 0.0, atol=1e-6, rtol=0)
                & good
            )

            if len(idx_z0) == 1:

                j = idx_z0[0]

                ax.scatter(
                    x[j],
                    y[j],
                    s=42,
                    facecolors=colour,
                    edgecolors='black',
                    linewidths=0.7,
                    zorder=8,
                )

        # --------------------------------------------------
        # Compactness threshold
        # --------------------------------------------------

        xx = np.linspace(xmin, xmax, 300)

        cut = float(
            REFERENCE_SAMPLE['compactness_threshold']
        )

        exponent = float(
            REFERENCE_SAMPLE['compactness_exponent']
        )

        ax.plot(
            xx,
            (xx - cut) / exponent,
            '--',
            color='black',
            lw=1.2,
            zorder=4,
        )

        ax.set_title(
            f'{label}\nN={len(selected_ids)}'
        )

        ax.set_xlabel(
            r'$\log_{10}(M_\star/\mathrm{M}_\odot)$'
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.grid(alpha=0.2)

    axes[0].set_ylabel(
        r'$\log_{10}(R_{1/2,\star}/\mathrm{kpc})$'
    )

    # --------------------------------------------------
    # 4. Shared legend
    # --------------------------------------------------

    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=7,
            label=start_label,
        ),
    ]

    if show_final_quenching:

        legend_elements.append(
            Line2D(
                [0], [0],
                marker='D',
                color='none',
                markerfacecolor='0.5',
                markeredgecolor='black',
                markersize=6,
                label='First snapshot of final quenched phase',
            )
        )

    legend_elements.extend([
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor='0.5',
            markeredgecolor='black',
            markersize=7,
            label=r'$z=0$',
        ),

        Line2D(
            [0], [0],
            color='black',
            linestyle='--',
            lw=1.2,
            label='Compactness threshold',
        ),
    ])

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(legend_elements),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return _save(
        fig,
        outdir / output_stem,
        formats,
    )

# ============================================================
# THREE-PANEL MATCHED CONTROL MASS-SIZE EVOLUTION
# ============================================================

def plot_control_mass_size_exsitu_panels(
    pairs,
    history,
    outdir,
    formats=('png', 'pdf'),
):
    """
    Plot matched-control mass-size trajectories.

    Panels are defined by the z=0 ex-situ group of the
    corresponding reference SRG.

    Open circle:
        Matching epoch.

    Filled circle:
        Same TrackID observed at z=0.

    Cross:
        Last observed snapshot if absent at z=0.
    """

    from matplotlib.lines import Line2D

    groups = [
        ('low', 'C2', r'$f_{\rm exsitu,SRG}<0.1$'),
        ('mid', 'C1', r'$0.1\leq f_{\rm exsitu,SRG}<0.4$'),
        ('high', 'C4', r'$f_{\rm exsitu,SRG}\geq0.4$'),
    ]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(19, 6.5),
        sharex=True,
        sharey=True,
    )

    finite = (
        np.isfinite(history['log10_stellar_mass'])
        & np.isfinite(history['log10_rhalf'])
    )

    x_all = history.loc[
        finite, 'log10_stellar_mass'
    ].to_numpy(float)

    y_all = history.loc[
        finite, 'log10_rhalf'
    ].to_numpy(float)

    if len(x_all) == 0:
        raise ValueError(
            "No finite control mass-size measurements."
        )

    xmin, xmax = 9.0, 12.0
    ymin, ymax = -1.0, 1.8

    # --------------------------------------------------------
    # Plot pair-specific histories
    # --------------------------------------------------------

    for ax, (group, colour, label) in zip(axes, groups):

        selected = history[
            history['reference_exsitu_group'] == group
        ]

        group_pairs = pairs[
            pairs['reference_exsitu_group'] == group
        ]

        for keys, g in selected.groupby(
            ['reference_track_id', 'control_track_id', 'z_match']
        ):

            g = g.sort_values(
                'redshift',
                ascending=False,
            )

            x = g['log10_stellar_mass'].to_numpy(float)
            y = g['log10_rhalf'].to_numpy(float)
            z = g['redshift'].to_numpy(float)

            good = np.isfinite(x) & np.isfinite(y)

            if good.sum() == 0:
                continue

            ax.plot(
                np.where(good, x, np.nan),
                np.where(good, y, np.nan),
                color=colour,
                lw=0.8,
                alpha=0.22,
                rasterized=True,
                zorder=2,
            )

            # Direction arrows
            for i in range(len(x) - 1):

                if not (good[i] and good[i + 1]):
                    continue

                ax.annotate(
                    '',
                    xy=(x[i + 1], y[i + 1]),
                    xytext=(x[i], y[i]),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=colour,
                        lw=0.6,
                        alpha=0.22,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=3,
                )

            # Matching-epoch position
            z_match = float(keys[2])

            start = np.flatnonzero(
                np.isclose(z, z_match, atol=1e-6, rtol=0)
                & good
            )

            if len(start) == 1:

                j = start[0]

                ax.scatter(
                    x[j], y[j],
                    s=28,
                    facecolors='white',
                    edgecolors=colour,
                    linewidths=0.8,
                    zorder=5,
                )

            # --------------------------------------------------
            # Present-day persistence
            # --------------------------------------------------

            present_z0 = bool(
                g['trackid_present_z0_ascii'].iloc[0]
            )

            z0 = np.flatnonzero(
                np.isclose(
                    z,
                    0.0,
                    atol=1e-6,
                    rtol=0,
                )
                & good
            )

            if present_z0 and len(z0) == 1:

                j = z0[0]

                ax.scatter(
                    x[j],
                    y[j],
                    s=32,
                    facecolors=colour,
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=6,
                )

            elif not present_z0:

                last = np.flatnonzero(good)

                if len(last):

                    j = last[-1]

                    ax.scatter(
                        x[j],
                        y[j],
                        marker='x',
                        s=30,
                        color=colour,
                        linewidths=1.0,
                        zorder=6,
                    )

        # Compactness threshold
        xx = np.linspace(xmin, xmax, 300)

        cut = float(
            REFERENCE_SAMPLE['compactness_threshold']
        )

        exponent = float(
            REFERENCE_SAMPLE['compactness_exponent']
        )

        ax.plot(
            xx,
            (xx - cut) / exponent,
            '--',
            color='black',
            lw=1.2,
        )

        n_pairs = len(group_pairs)

        n_unique = group_pairs[
            'control_track_id'
        ].nunique()

        ax.set_title(
            f'{label}\n'
            f'N pairs = {n_pairs}, '
            f'N unique = {n_unique}'
        )

        ax.set_xlabel(
            r'$\log_{10}(M_\star/\mathrm{M}_\odot)$'
        )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(
        r'$\log_{10}(R_{1/2,\star}/\mathrm{kpc})$'
    )

    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor='white',
            markeredgecolor='black',
            label=r'$z_{\mathrm{match}}=z_{\mathrm{firstCQ,SRG}}$',
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor='0.5',
            markeredgecolor='black',
            label=r'$z=0$',
        ),
        Line2D(
            [0], [0],
            marker='x',
            color='black',
            label='Last observed (absent at z=0)',
        ),
        Line2D(
            [0], [0],
            linestyle='--',
            color='black',
            label='Compactness threshold',
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return _save(
        fig,
        outdir / 'control_mass_size_exsitu_panels',
        formats,
    )

def plot_matched_srg_mass_size_exsitu_panels(
    pairs,
    outdir,
    formats=('png', 'pdf'),
):
    """
    Plot reference SRGs from their original z_firstCQ matching
    epochs to z=0, grouped by their own z=0 ex-situ fraction.
    """

    # Reuse the existing evolutionary catalogue.
    history = load_history()

    # One matching epoch per reference SRG.
    reference = (
        pairs[
            [
                'reference_track_id',
                'z_match',
                'reference_exsitu_group',
            ]
        ]
        .drop_duplicates()
    )

    if reference['reference_track_id'].duplicated().any():
        raise ValueError(
            "One SRG has multiple matching epochs."
        )

    tracks = []

    for _, row in reference.iterrows():

        tid = int(row['reference_track_id'])
        z_start = float(row['z_match'])

        g = history[
            history['track_id'] == tid
        ].copy()

        # Verify that the exact matching epoch exists.
        start_rows = g[
            np.isclose(
                g['redshift'],
                z_start,
                atol=1e-6,
                rtol=0,
            )
        ]

        if len(start_rows) != 1:
            raise ValueError(
                f"SRG {tid}: expected exactly one "
                f"snapshot at z_match={z_start}, "
                f"found {len(start_rows)}."
            )

        # Retain the trajectory from z_firstCQ to z=0.
        g = g[
            g['redshift'] <= z_start + 1e-6
        ].copy()

        g['exsitu_group'] = row[
            'reference_exsitu_group'
        ]

        if not g.empty:
            tracks.append(g)

    if not tracks:
        raise ValueError(
            "No matched SRG histories found."
        )

    tracks = pd.concat(
        tracks,
        ignore_index=True,
    )

    # Build a summary compatible with our existing
    # three-panel SRG plotting function.

    summary = reference.rename(columns={
        'reference_track_id': 'track_id',
        'reference_exsitu_group': 'exsitu_group',
        'z_match': 'z_start',
    }).copy()

    return plot_post_quenching_mass_size_exsitu_panels(
        tracks,
        summary,
        outdir,
        formats,
        epoch_mode='firstCQ',
    )

# ============================================================
# PRESENT-DAY DoR VS COMPACTNESS OF MATCHED CONTROLS
# ============================================================

def plot_control_dor_compactness(
    outdir,
    formats=('png', 'pdf'),
    overwrite=False,
):
    """
    Plot present-day DoR versus compactness for unique
    matched control galaxies.

    Generates two figures:

    1. Coloured by present-day population classification.
    2. Coloured by each control galaxy's own z=0 ex-situ
       stellar mass fraction.

    Uses SOAP-based z=0 DoR and compactness values.

    Ex-situ fractions are matched using the z=0
    HaloCatalogueIndex and the Paper 1 ex-situ HDF5.

    Each control TrackID contributes at most one point.
    """

    import h5py
    from config import INPUTS

    # ========================================================
    # 1. INPUT / OUTPUT PATHS
    # ========================================================

    input_file = project_path(
        Path(OUTPUT['root'])
        / 'controls'
        / 'descendants'
        / 'control_descendant_status.csv'
    )

    exsitu_file = project_path(
        INPUTS['exsitu_z0_hdf5']
    )

    outdir = project_path(outdir)

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Missing control descendant catalogue: {input_file}"
        )

    if not exsitu_file.is_file():
        raise FileNotFoundError(
            f"Missing ex-situ catalogue: {exsitu_file}"
        )

    # ========================================================
    # 2. LOAD CONTROL DESCENDANT CATALOGUE
    # ========================================================

    df = pd.read_csv(
        input_file,
        dtype={'control_track_id': 'int64'},
    )

    required = {
        'control_track_id',
        'reference_z0_DoR_t95',
        'reference_z0_compactness_z0',
        'reference_z0_population',
        'reference_z0_halo_catalogue_index',
        'in_reference_z0',
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Control catalogue missing columns: {sorted(missing)}"
        )

    if df['control_track_id'].duplicated().any():
        raise ValueError(
            "Duplicate control TrackIDs detected."
        )

    n_total = len(df)

    # ========================================================
    # 3. SELECT CONTROLS IDENTIFIED AT z=0
    # ========================================================

    present = (
        df['in_reference_z0']
        .astype('string')
        .str.lower()
        .map({
            'true': True,
            'false': False,
        })
    )

    if present.isna().any():
        raise ValueError(
            "Invalid in_reference_z0 flags."
        )

    df = df.loc[present].copy()

    # ========================================================
    # 4. EXTRACT PRESENT-DAY PROPERTIES
    # ========================================================

    df['dor'] = pd.to_numeric(
        df['reference_z0_DoR_t95'],
        errors='coerce',
    )

    df['compactness'] = pd.to_numeric(
        df['reference_z0_compactness_z0'],
        errors='coerce',
    )

    df['population'] = (
        df['reference_z0_population']
        .astype('string')
    )

    df['halo_index'] = pd.to_numeric(
        df['reference_z0_halo_catalogue_index'],
        errors='coerce',
    )

    # ========================================================
    # 5. LOAD EX-SITU STELLAR MASS FRACTIONS
    # ========================================================

    with h5py.File(exsitu_file, 'r') as fh:

        if 'stars' not in fh:
            raise KeyError(
                "Ex-situ HDF5 missing dataset 'stars'."
            )

        data = np.asarray(fh['stars'])

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(
            "Unexpected ex-situ dataset structure."
        )

    # Previously validated Paper 1 column definitions:
    #
    # Column 0: HaloCatalogueIndex
    # Column 3: Ex-situ stellar mass fraction

    exsitu_ids = data[:, 0].astype(np.int64)

    exsitu_fractions = data[:, 3].astype(float)

    if pd.Series(exsitu_ids).duplicated().any():
        raise ValueError(
            "Duplicate identifiers in ex-situ catalogue."
        )

    lookup = pd.Series(
        exsitu_fractions,
        index=exsitu_ids,
    )

    # ========================================================
    # 6. MATCH CONTROL GALAXIES TO EX-SITU FRACTIONS
    # ========================================================

    df['exsitu_fraction'] = np.nan

    valid_halo = (
        df['halo_index'].notna()
    )

    df.loc[
        valid_halo,
        'exsitu_fraction'
    ] = (
        df.loc[valid_halo, 'halo_index']
        .astype('int64')
        .map(lookup)
    )

    # Reject invalid fractions.

    invalid_fraction = (
        ~np.isfinite(df['exsitu_fraction'])
        | (df['exsitu_fraction'] < 0)
        | (df['exsitu_fraction'] > 1)
    )

    df.loc[
        invalid_fraction,
        'exsitu_fraction'
    ] = np.nan

    # ========================================================
    # 7. SELECT VALID GALAXIES
    # ========================================================

    valid = df[
        np.isfinite(df['dor'])
        & np.isfinite(df['compactness'])
        & df['population'].notna()
    ].copy()

    valid_exsitu = valid[
        np.isfinite(valid['exsitu_fraction'])
    ].copy()

    # ========================================================
    # 8. REPORT
    # ========================================================

    print("\nCONTROL DoR–COMPACTNESS ANALYSIS")
    print("=" * 65)

    print("Unique matched controls:", n_total)

    print(
        "Controls identified at z=0:",
        len(df)
    )

    print(
        "Valid DoR and compactness:",
        len(valid)
    )

    print(
        "Valid ex-situ fractions:",
        len(valid_exsitu)
    )

    print(
        "Controls without z=0 identification:",
        n_total - len(df)
    )

    print("\nPresent-day classifications:")

    print(
        valid['population']
        .value_counts()
        .to_string()
    )

    print("\nEx-situ fraction percentiles:")

    print(
        valid_exsitu['exsitu_fraction'].quantile(
            [0, 0.16, 0.5, 0.84, 1.0]
        )
    )

    # ========================================================
    # 9. CORRELATIONS
    # ========================================================

    if len(valid_exsitu) >= 3:

        rho_compact, p_compact = spearmanr(
            valid_exsitu['exsitu_fraction'],
            valid_exsitu['compactness'],
        )

        rho_dor, p_dor = spearmanr(
            valid_exsitu['exsitu_fraction'],
            valid_exsitu['dor'],
        )

    else:

        rho_compact = np.nan
        p_compact = np.nan

        rho_dor = np.nan
        p_dor = np.nan

    print("\nSPEARMAN CORRELATIONS")
    print("=" * 65)

    print(
        f"Ex-situ vs compactness: "
        f"rho = {rho_compact:.4f}, "
        f"p = {p_compact:.4g}"
    )

    print(
        f"Ex-situ vs DoR: "
        f"rho = {rho_dor:.4f}, "
        f"p = {p_dor:.4g}"
    )

    # ========================================================
    # 10. OUTPUT FILE CHECK
    # ========================================================

    stems = [
        'control_dor_compactness_z0',
        'control_dor_compactness_exsitu_z0',
    ]

    expected = [
        outdir / f'{stem}.{fmt}'
        for stem in stems
        for fmt in formats
    ]

    expected += [
        outdir / 'control_dor_compactness_z0.csv',
        outdir / 'control_dor_compactness_correlations.csv',
    ]

    existing = [
        str(path)
        for path in expected
        if path.exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Control DoR-compactness outputs already exist; "
            f"use --overwrite: {existing}"
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 11. SAVE NUMERICAL RESULTS
    # ========================================================

    valid[
        [
            'control_track_id',
            'dor',
            'compactness',
            'population',
            'exsitu_fraction',
        ]
    ].to_csv(
        outdir / 'control_dor_compactness_z0.csv',
        index=False,
    )

    correlations = pd.DataFrame([
        {
            'quantity': 'compactness',
            'n_galaxies': len(valid_exsitu),
            'spearman_rho': rho_compact,
            'spearman_p': p_compact,
        },
        {
            'quantity': 'DoR',
            'n_galaxies': len(valid_exsitu),
            'spearman_rho': rho_dor,
            'spearman_p': p_dor,
        },
    ])

    correlations.to_csv(
        outdir / 'control_dor_compactness_correlations.csv',
        index=False,
    )

    # ========================================================
    # 12. CLASSIFICATION-COLOURED FIGURE
    # ========================================================

    population_styles = {
        'SRG': ('C2', 'SRG'),
        'SAG_noncompact': ('C1', 'Noncompact SAG'),
        'compact_nonrelic': ('C4', 'Compact non-relic'),
        'noncompact_nonrelic': ('0.5', 'Noncompact non-relic'),
    }

    dor_cut = 0.6

    compactness_cut = float(
        REFERENCE_SAMPLE['compactness_threshold']
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    for population, (colour, label) in population_styles.items():

        subset = valid[
            valid['population'] == population
        ]

        if subset.empty:
            continue

        ax.scatter(
            subset['dor'],
            subset['compactness'],
            s=35,
            color=colour,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.35,
            label=f'{label} (N={len(subset)})',
            rasterized=True,
        )

    ax.axvline(
        dor_cut,
        color='black',
        linestyle='--',
        linewidth=1.3,
    )

    ax.axhline(
        compactness_cut,
        color='black',
        linestyle='--',
        linewidth=1.3,
    )

    ax.set_xlabel(
        r'$\mathrm{DoR}(z=0)$'
    )

    ax.set_ylabel(
        r'$\log_{10}\Sigma_{1.5}(z=0)$'
    )

    ax.set_title(
        'Present-day properties of matched control galaxies'
    )

    ax.set_xlim(0, 1)

    ax.grid(alpha=0.2)

    ax.legend(
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()

    for fmt in formats:

        fig.savefig(
            outdir / f'control_dor_compactness_z0.{fmt}',
            dpi=250,
            bbox_inches='tight',
        )

    plt.close(fig)

    # ========================================================
    # 13. EX-SITU-COLOURED FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    scatter = ax.scatter(
        valid_exsitu['dor'],
        valid_exsitu['compactness'],
        c=valid_exsitu['exsitu_fraction'],
        cmap='viridis',
        vmin=0,
        vmax=1,
        s=40,
        alpha=0.85,
        edgecolors='black',
        linewidths=0.3,
        rasterized=True,
    )

    # Classification thresholds

    ax.axvline(
        dor_cut,
        color='black',
        linestyle='--',
        linewidth=1.3,
    )

    ax.axhline(
        compactness_cut,
        color='black',
        linestyle='--',
        linewidth=1.3,
    )

    # Colour bar

    cbar = fig.colorbar(
        scatter,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        r'$f_{\mathrm{exsitu}}(z=0)$'
    )

    # Axis labels

    ax.set_xlabel(
        r'$\mathrm{DoR}(z=0)$'
    )

    ax.set_ylabel(
        r'$\log_{10}\Sigma_{1.5}(z=0)$'
    )

    ax.set_title(
        'Present-day ex-situ fractions of matched controls'
    )

    ax.set_xlim(0, 1)

    ax.grid(alpha=0.2)

    # Correlation annotation

    annotation = (
        f"N = {len(valid_exsitu)}\n"
        f"$\\rho_{{f_{{ex}},\\Sigma}}$ = {rho_compact:.3f}\n"
        f"$\\rho_{{f_{{ex}},DoR}}$ = {rho_dor:.3f}"
    )

    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=11,
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.85,
        ),
    )

    fig.tight_layout()

    for fmt in formats:

        fig.savefig(
            outdir / f'control_dor_compactness_exsitu_z0.{fmt}',
            dpi=250,
            bbox_inches='tight',
        )

    plt.close(fig)

    print("\nSaved control DoR-compactness outputs to:", outdir)

    return valid

def plot_structural_displacements(
    summary,
    outdir,
    formats=('png', 'pdf'),
):
    """
    Structural changes from the last star-forming snapshot
    to z=0.
    """

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    x = summary['delta_logM'].to_numpy(float)
    y = summary['delta_logR'].to_numpy(float)

    ax.scatter(
        x,
        y,
        s=30,
        color='C2',
        alpha=0.7,
        edgecolors='none',
        rasterized=True,
    )

    ax.axhline(
        0,
        color='0.5',
        ls='--',
        lw=1,
    )

    ax.axvline(
        0,
        color='0.5',
        ls='--',
        lw=1,
    )

    # Constant compactness:
    # Delta log R = Delta log M / 1.5

    xmin, xmax = ax.get_xlim()

    xx = np.linspace(
        xmin,
        xmax,
        300,
    )

    exponent = float(
        REFERENCE_SAMPLE['compactness_exponent']
    )

    ax.plot(
        xx,
        xx / exponent,
        'k:',
        lw=1.5,
        label='Constant compactness',
    )

    ax.set_xlabel(
        r'$\Delta\log_{10}(M_\star)$'
    )

    ax.set_ylabel(
        r'$\Delta\log_{10}(R_{1/2,\star})$'
    )

    ax.set_title(
        'Structural evolution after the last star-forming episode'
    )

    ax.grid(alpha=0.2)

    ax.legend(frameon=False)

    fig.tight_layout()

    return _save(
        fig,
        outdir / 'SRG_structural_displacements',
        formats,
    )    

# ============================================================
# FORMATION EPOCHS VS PRESENT-DAY EX-SITU FRACTION
# ============================================================

def plot_formation_exsitu_correlations(
    quenching_file=None,
    outdir=None,
    formats=('png', 'pdf'),
    overwrite=False,
    limit=None,
):
    """
    Investigate correlations between formation/quenching epochs
    and present-day ex-situ stellar mass fractions of SRGs.

    Epoch definitions
    -----------------
    z_firstCQ:
        Earliest observed compact-and-quenched snapshot.

    z_lastSF:
        Last star-forming snapshot immediately preceding
        the final accepted quenching transition.

    Ex-situ:
        Cumulative stellar ex-situ fraction measured at z=0.

    Outputs
    -------
    formation_exsitu_two_panels.png/pdf
    formation_exsitu_combined.png/pdf
    formation_exsitu_correlations.csv
    formation_exsitu_galaxies.csv
    """

    # --------------------------------------------------------
    # 1. Resolve paths
    # --------------------------------------------------------

    quenching_file = project_path(
        quenching_file
        or Path(OUTPUT['root'])
        / 'quenching_history'
        / 'quenching_history_per_galaxy.csv'
    )

    outdir = project_path(
        outdir
        or Path(OUTPUT['figures_dir'])
        / 'formation_exsitu'
    )

    if not quenching_file.is_file():
        raise FileNotFoundError(
            f"Quenching-history file not found: {quenching_file}"
        )

    # --------------------------------------------------------
    # 2. Load quenching histories
    # --------------------------------------------------------

    q = pd.read_csv(
        quenching_file,
        dtype={'track_id': 'int64'}
    )

    required = {
        'track_id',
        'population',
        'z_first_compact_quenched',
        'z_final_quenching_prior_star_forming',
        'final_quenching_status',
    }

    missing = required - set(q.columns)

    if missing:
        raise ValueError(
            f"Quenching catalogue missing columns: {sorted(missing)}"
        )

    q = q[
        q['population'] == 'SRG'
    ].copy()

    if q['track_id'].duplicated().any():
        raise ValueError(
            "Duplicate TrackIDs in quenching-history catalogue."
        )

    # --------------------------------------------------------
    # 3. Load previously validated z=0 ex-situ fractions
    # --------------------------------------------------------

    exsitu = load_srg_exsitu_fractions()

    df = q.merge(
        exsitu,
        on='track_id',
        how='left',
        validate='one_to_one',
    )

    # --------------------------------------------------------
    # 4. Rename the two epoch definitions
    # --------------------------------------------------------

    df['z_firstCQ'] = pd.to_numeric(
        df['z_first_compact_quenched'],
        errors='coerce'
    )

    df['z_lastSF'] = pd.to_numeric(
        df['z_final_quenching_prior_star_forming'],
        errors='coerce'
    )

    # Only accept last-SF epochs belonging to a valid
    # bracketed final quenching transition.

    valid_transition = (
        df['final_quenching_status']
        == 'final_quenching_bracketed'
    )

    df.loc[
        ~valid_transition,
        'z_lastSF'
    ] = np.nan

    df['exsitu_fraction'] = pd.to_numeric(
        df['exsitu_fraction'],
        errors='coerce'
    )

    # Reject invalid ex-situ fractions.

    invalid_fraction = (
        ~np.isfinite(df['exsitu_fraction'])
        | (df['exsitu_fraction'] < 0)
        | (df['exsitu_fraction'] > 1)
    )

    df.loc[
        invalid_fraction,
        'exsitu_fraction'
    ] = np.nan

    # --------------------------------------------------------
    # 5. Optional limited sample
    # --------------------------------------------------------

    if limit is not None:

        ids = sorted(
            df['track_id'].unique()
        )[:limit]

        df = df[
            df['track_id'].isin(ids)
        ].copy()

    # --------------------------------------------------------
    # 6. Use the same galaxies for both correlations
    # --------------------------------------------------------

    valid = df[
        np.isfinite(df['z_firstCQ'])
        & np.isfinite(df['z_lastSF'])
        & np.isfinite(df['exsitu_fraction'])
    ].copy()

    print("\nFORMATION–EX-SITU ANALYSIS")
    print("=" * 60)

    print("Total SRGs:", len(df))

    print(
        "Valid first-CQ epochs:",
        df['z_firstCQ'].notna().sum()
    )

    print(
        "Valid last-SF epochs:",
        df['z_lastSF'].notna().sum()
    )

    print(
        "Valid ex-situ fractions:",
        df['exsitu_fraction'].notna().sum()
    )

    print(
        "Galaxies with all three measurements:",
        len(valid)
    )

    if len(valid) < 3:
        raise ValueError(
            "Insufficient galaxies for correlation analysis."
        )

    # --------------------------------------------------------
    # 7. Calculate correlation coefficients
    # --------------------------------------------------------

    x = valid[
        'exsitu_fraction'
    ].to_numpy(dtype=float)

    epoch_definitions = {
        'First compact + quenched': 'z_firstCQ',
        'Last star-forming': 'z_lastSF',
    }

    results = []

    for label, column in epoch_definitions.items():

        y = valid[column].to_numpy(dtype=float)

        if np.unique(x).size < 2 or np.unique(y).size < 2:
            r, p_pearson = np.nan, np.nan
            rho, p_spearman = np.nan, np.nan
        else:
            r, p_pearson = pearsonr(x, y)
            rho, p_spearman = spearmanr(x, y)

        results.append({
            'epoch_definition': label,
            'n_galaxies': len(valid),
            'pearson_r': float(r),
            'pearson_p': float(p_pearson),
            'spearman_rho': float(rho),
            'spearman_p': float(p_spearman),
            'median_redshift': float(np.median(y)),
            'p16_redshift': float(np.percentile(y, 16)),
            'p84_redshift': float(np.percentile(y, 84)),
        })

    correlations = pd.DataFrame(results)

    print("\nCORRELATION RESULTS")
    print("=" * 60)

    print(
        correlations.to_string(
            index=False,
            float_format=lambda v: f"{v:.4g}"
        )
    )

    # --------------------------------------------------------
    # 8. Dry-run support
    # --------------------------------------------------------

    if outdir is None:
        raise ValueError("Output directory is undefined.")

    if not overwrite:
        expected = [
            outdir / f'{stem}.{ext}'
            for stem in (
                'formation_exsitu_two_panels',
                'formation_exsitu_combined',
            )
            for ext in formats
        ]

        expected += [
            outdir / 'formation_exsitu_correlations.csv',
            outdir / 'formation_exsitu_galaxies.csv',
        ]

        existing = [
            str(path)
            for path in expected
            if path.exists()
        ]

        if existing:
            raise FileExistsError(
                "Formation-exsitu outputs already exist; "
                f"use --overwrite: {existing}"
            )

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 9. Save numerical catalogues
    # --------------------------------------------------------

    correlations.to_csv(
        outdir / 'formation_exsitu_correlations.csv',
        index=False
    )

    valid[
        [
            'track_id',
            'exsitu_fraction',
            'exsitu_group',
            'z_firstCQ',
            'z_lastSF',
        ]
    ].to_csv(
        outdir / 'formation_exsitu_galaxies.csv',
        index=False
    )

    # --------------------------------------------------------
    # 10. Two-panel correlation figure
    # --------------------------------------------------------

    styles = {
        'First compact + quenched': {
            'color': 'C2',
            'marker': 'o',
        },
        'Last star-forming': {
            'color': 'C1',
            'marker': 'D',
        },
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
    )

    for ax, (label, column) in zip(
        axes,
        epoch_definitions.items()
    ):

        style = styles[label]

        y = valid[column].to_numpy(float)

        ax.scatter(
            x,
            y,
            s=38,
            marker=style['marker'],
            facecolors=style['color'],
            edgecolors='black',
            linewidths=0.35,
            alpha=0.65,
            rasterized=True,
        )

        row = correlations.loc[
            correlations['epoch_definition'] == label
        ].iloc[0]

        annotation = (
            f"N = {len(valid)}\n"
            f"Pearson $r$ = {row['pearson_r']:.3f}\n"
            f"Spearman $\\rho$ = {row['spearman_rho']:.3f}"
        )

        ax.text(
            0.04,
            0.96,
            annotation,
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=11,
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.85,
            ),
        )

        ax.set_title(label)

        ax.set_xlabel(
            r'$f_{\mathrm{exsitu}}(z=0)$'
        )

        ax.grid(alpha=0.2)

    axes[0].set_ylabel(
        'Reference redshift'
    )

    fig.suptitle(
        'Formation epochs versus present-day ex-situ fraction',
        fontsize=16,
    )

    fig.tight_layout()

    for ext in formats:
        fig.savefig(
            outdir / f'formation_exsitu_two_panels.{ext}',
            dpi=250,
            bbox_inches='tight',
        )

    plt.close(fig)

    # --------------------------------------------------------
    # 11. Combined correlation figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    for label, column in epoch_definitions.items():

        style = styles[label]

        ax.scatter(
            x,
            valid[column].to_numpy(float),
            s=38,
            marker=style['marker'],
            facecolors=style['color'],
            edgecolors='black',
            linewidths=0.35,
            alpha=0.55,
            label=label,
            rasterized=True,
        )

    ax.set_xlabel(
        r'$f_{\mathrm{exsitu}}(z=0)$'
    )

    ax.set_ylabel(
        'Reference redshift'
    )

    ax.set_title(
        'Alternative formation-epoch definitions'
    )

    ax.grid(alpha=0.2)

    ax.legend(
        frameon=False,
        loc='upper right',
    )

    fig.tight_layout()

    for ext in formats:
        fig.savefig(
            outdir / f'formation_exsitu_combined.{ext}',
            dpi=250,
            bbox_inches='tight',
        )

    plt.close(fig)

    print("\nSaved formation-exsitu outputs to:", outdir)

    return correlations, valid

def epoch_statistics(df, column):
    """One contribution per detected TrackID per epoch; do not fabricate missing data."""
    rows = []
    for (pop, z), g in df.groupby(['population', 'redshift'], sort=True):
        values = pd.to_numeric(g[column], errors='coerce').to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append({'population': pop, 'redshift': float(z), 'n_detected': len(g),
                     'n_valid': len(finite),
                     'median': float(np.median(finite)) if len(finite) else np.nan,
                     'p16': float(np.percentile(finite, 16)) if len(finite) else np.nan,
                     'p84': float(np.percentile(finite, 84)) if len(finite) else np.nan})
    return pd.DataFrame(rows).sort_values(['population', 'redshift']).reset_index(drop=True)


def _save(fig, base, formats):
    paths = []
    for ext in formats:
        dest = base.with_suffix('.' + ext)
        fig.savefig(dest, dpi=220, bbox_inches='tight')
        paths.append(dest)
    plt.close(fig)
    return paths


def plot_quantity(df, key, outdir, formats=('png', 'pdf')):
    col, ylabel, stem = QUANTITIES[key]
    fig, ax = plt.subplots(figsize=(9, 6))
    # Each galaxy is equally weighted; missing snapshots remain gaps.
    for (tid, pop), g in df.groupby(['track_id', 'population'], sort=False):
        g = g.sort_values('redshift')
        x = g['redshift'].to_numpy(dtype=float)
        y = g[col].to_numpy(dtype=float)
        ok = np.isfinite(y)
        if not ok.any():
            continue
        ax.plot(x, y, color=COLORS.get(pop, '0.5'), alpha=0.11, lw=0.8,
                rasterized=True, zorder=1)
    for pop, g in df.groupby('population', sort=True):
        stats = epoch_statistics(g, col).sort_values('redshift')
        ok = stats['n_valid'].ge(1) & np.isfinite(stats['median'])
        s = stats.loc[ok]
        if s.empty:
            continue
        color = COLORS.get(pop, 'C0')
        ax.fill_between(s['redshift'].to_numpy(), s['p16'].to_numpy(),
                        s['p84'].to_numpy(), color=color, alpha=0.20, zorder=3)
        ax.plot(s['redshift'], s['median'], color=color, lw=2.5, marker='o', ms=3.5,
                label=f'{pop}: median (16th–84th)', zorder=4)
    if key == 'compactness':
        cut = float(REFERENCE_SAMPLE['compactness_threshold'])
        ax.axhline(cut, ls='--', lw=1.2, color='black',
                   label=f'z=0 reference compactness cut ({cut:g}); not a red-nugget cut')
    if key == 'ssfr':
        ax.text(0.02, 0.02, 'Zero SFR shown at plotting floor; missing SFR omitted',
                transform=ax.transAxes, fontsize=9, va='bottom')
    ax.set_xlabel('Redshift (higher z = earlier)')
    ax.set_ylabel(ylabel)
    ax.invert_xaxis()
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('Detected progenitors of z=0 populations')
    fig.tight_layout()
    return _save(fig, outdir / stem, formats)


def plot_mass_size(df, outdir, formats=('png', 'pdf')):
    fig, ax = plt.subplots(figsize=(9, 7))
    for (tid, pop), g in df.groupby(['track_id', 'population'], sort=False):
        g = g.sort_values('redshift', ascending=False)
        x = g['log10_stellar_mass'].to_numpy(dtype=float)
        y = g['log10_rhalf'].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        if not good.any():
            continue
        color = COLORS.get(pop, '0.5')
        # Masked values break lines, preventing false links across invalid measurements.
        ax.plot(np.where(good, x, np.nan), np.where(good, y, np.nan),
                color=color, lw=0.9, alpha=0.16, rasterized=True)
        for z, marker, face in [(2.0, 'o', 'none'), (0.0, 'o', color)]:
            sub = g.loc[np.isclose(g['redshift'], z, atol=1e-8, rtol=0)]
            if len(sub) == 1:
                xx = float(sub['log10_stellar_mass'].iloc[0]); yy = float(sub['log10_rhalf'].iloc[0])
                if np.isfinite(xx) and np.isfinite(yy):
                    ax.scatter(xx, yy, marker=marker, s=25 if z else 30,
                               facecolors=face, edgecolors=color if z else 'black',
                               linewidths=0.7, alpha=0.65, rasterized=True)
    # Cohort medians by redshift (medians need not correspond to one galaxy).
    for pop, g in df.groupby('population'):
        s = g.groupby('redshift').agg(
            logm=('log10_stellar_mass', 'median'), logr=('log10_rhalf', 'median'),
            n=('track_id', 'size')).reset_index().sort_values('redshift', ascending=False)
        s = s[np.isfinite(s['logm']) & np.isfinite(s['logr'])]
        if len(s):
            ax.plot(s['logm'], s['logr'], color=COLORS.get(pop, 'C0'), lw=2.4,
                    marker='o', ms=4, label=f'{pop}: epoch medians')
    xmin, xmax = ax.get_xlim()
    xx = np.linspace(xmin, xmax, 200)
    cut = float(REFERENCE_SAMPLE['compactness_threshold'])
    exponent = float(REFERENCE_SAMPLE['compactness_exponent'])
    ax.plot(xx, (xx - cut) / exponent, ls='--', color='black', lw=1.2,
            label='z=0 reference compactness cut')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'$\log_{10}(M_\star/\mathrm{M}_\odot)$')
    ax.set_ylabel(r'$\log_{10}(R_{1/2,\star}/\mathrm{kpc})$')
    ax.set_title('Mass–size evolution: z=2 open; z=0 filled')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _save(fig, outdir / 'mass_size_tracks', formats)


def plot_completeness(df, outdir, formats=('png', 'pdf')):
    fig, ax = plt.subplots(figsize=(8, 5))
    for pop, g in df.groupby('population'):
        n_targets = g['track_id'].nunique()
        counts = g.groupby('redshift')['track_id'].nunique()
        # Missing epochs from the full inventory are not represented in this figure.
        ax.plot(counts.index, counts.to_numpy() / n_targets, marker='o', ms=4,
                color=COLORS.get(pop, 'C0'), label=f'{pop} (N={n_targets})')
    ax.set_ylim(0, 1.04)
    ax.invert_xaxis()
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Fraction of selected tracks detected')
    ax.set_title('Detection completeness at represented epochs')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, outdir / 'tracking_completeness_observed_epochs', formats)


def run_plots(df, outdir, formats=('png', 'pdf'), overwrite=False):
    outdir = project_path(outdir)
    names = [v[2] for v in QUANTITIES.values()] + ['mass_size_tracks', 'tracking_completeness_observed_epochs']
    expected = [outdir / f'{stem}.{fmt}' for stem in names for fmt in formats]
    expected += [outdir / f'epoch_statistics_{key}.csv' for key in QUANTITIES]
    existing = [str(p) for p in expected if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Plot output(s) already exist; use --overwrite: {existing}')
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for key, (col, _, _) in QUANTITIES.items():
        written.extend(plot_quantity(df, key, outdir, formats))
        path = outdir / f'epoch_statistics_{key}.csv'
        epoch_statistics(df, col).to_csv(path, index=False)
        written.append(path)
    written.extend(plot_mass_size(df, outdir, formats))
    written.extend(plot_completeness(df, outdir, formats))
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--write', action='store_true', help='Generate figures and epoch summary tables')
    actions.add_argument('--dry-run', action='store_true', help='Validate and summarize without writing (default)')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing plot outputs (requires --write)')
    parser.add_argument('--limit', type=int, help='Plot first N TrackIDs for testing')
    parser.add_argument('--input-file', type=Path, help='Override saved evolutionary catalogue path')
    parser.add_argument('--output-dir', type=Path, help='Override output directory')
    parser.add_argument('--formation-only', action='store_true', help='Plot formation diagnostics from saved formation catalogues only')
    parser.add_argument('--post-quenching', action='store_true', help='Plot SRG mass-size evolution from individual final star-forming epochs')
    parser.add_argument('--formation-exsitu', action='store_true', help='Correlate SRG formation epochs with z=0 ex-situ fractions')
    parser.add_argument('--control-mass-size', action='store_true', help='Plot matched control and SRG mass-size tracks from original matching epochs')
    parser.add_argument('--control-dor-compactness', action='store_true', help='Plot z=0 DoR versus compactness for matched controls')
    parser.add_argument('--formation-dir', type=Path, help='Override saved formation catalogue directory')
    parser.add_argument('--png-only', action='store_true', help='Save only PNG rather than PNG and PDF')
    args = parser.parse_args()
    if args.overwrite and not args.write:
        parser.error('--overwrite requires --write')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')

    # ========================================================
    # Formation epoch versus ex-situ fraction
    # ========================================================
    if args.formation_exsitu:

        outdir = project_path(
            args.output_dir
            or Path(OUTPUT['figures_dir'])
            / 'formation_exsitu'
        )

        # ----------------------------------------------------
        # Dry run: calculate correlations without saving
        # ----------------------------------------------------

        if not args.write:

            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp:

                correlations, valid = (
                    plot_formation_exsitu_correlations(
                        outdir=Path(tmp),
                        formats=('png',) if args.png_only
                        else ('png', 'pdf'),
                        overwrite=True,
                        limit=args.limit,
                    )
                )

            print("\nDry run completed.")
            print("No permanent files written.")

            return

        # ----------------------------------------------------
        # Write figures and numerical catalogues
        # ----------------------------------------------------

        plot_formation_exsitu_correlations(
            outdir=outdir,
            formats=('png',) if args.png_only
            else ('png', 'pdf'),
            overwrite=args.overwrite,
            limit=args.limit,
        )

        return

    # ========================================================
    # PRESENT-DAY CONTROL DoR VS COMPACTNESS
    # ========================================================

    if args.control_dor_compactness:

        outdir = project_path(
            args.output_dir
            or Path(OUTPUT['root'])
            / 'controls'
            / 'figures'
        )

        formats = (
            ('png',)
            if args.png_only
            else ('png', 'pdf')
        )

        if not args.write:

            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp:

                plot_control_dor_compactness(
                    outdir=Path(tmp),
                    formats=formats,
                    overwrite=True,
                )

            print("\nDry run completed.")
            print("No permanent files written.")

            return

        plot_control_dor_compactness(
            outdir=outdir,
            formats=formats,
            overwrite=args.overwrite,
        )

        return

    # ========================================================
    # MATCHED CONTROL MASS-SIZE EVOLUTION
    # ========================================================

    if args.control_mass_size:

        pairs, history = load_matched_control_histories()

        print(
            f"\nMatched pairs: {len(pairs)}"
        )

        print(
            f"Control history rows: {len(history)}"
        )

        if args.limit is not None:

            ids = sorted(
                pairs['reference_track_id'].unique()
            )[:args.limit]

            pairs = pairs[
                pairs['reference_track_id'].isin(ids)
            ].copy()

            history = history[
                history['reference_track_id'].isin(ids)
            ].copy()

        if not args.write:
            print("Dry run: no files written.")
            return

        outdir = project_path(
            args.output_dir
            or Path(OUTPUT['root'])
            / 'control_mass_size'
        )

        formats = (
            ('png',)
            if args.png_only
            else ('png', 'pdf')
        )

        control_dir = outdir / 'controls'
        srg_dir = outdir / 'matched_srgs'

        control_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        srg_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        plot_control_mass_size_exsitu_panels(
            pairs,
            history,
            control_dir,
            formats,
        )

        plot_matched_srg_mass_size_exsitu_panels(
            pairs,
            srg_dir,
            formats,
        )

        print("\nSaved matched mass-size comparison.")

        return

    if args.post_quenching:

        history = load_history(args.input_file)

        quenching = load_quenching_epochs()

        if args.limit is not None:
            ids = sorted(
                quenching['track_id'].unique()
            )[:args.limit]

            quenching = quenching[
                quenching['track_id'].isin(ids)
            ].copy()

        summary, tracks = build_post_quenching_catalogue(
            history,
            quenching,
        )

        # --------------------------------------------------
        # Attach z=0 ex-situ fractions from Paper 1
        # --------------------------------------------------

        exsitu = load_srg_exsitu_fractions()

        summary = summary.merge(
            exsitu,
            on='track_id',
            how='left',
            validate='one_to_one',
        )

        summary['exsitu_group'] = (
            summary['exsitu_group']
            .fillna('unknown')
        )

        print(
            f"\nSRGs with valid post-quenching tracks: "
            f"{len(summary)}"
        )

        print("\nPost-quenching ex-situ groups:")
        print(summary['exsitu_group'].value_counts())

        print(
            "\nSRGs with valid ex-situ fractions:",
            summary['exsitu_group'].ne('unknown').sum(),
            "/",
            len(summary),
        )

        if not args.write:
            print("Dry run: no files written.")
            return

        outdir = project_path(
            args.output_dir
            or Path(OUTPUT['root']) / 'post_quenching_evolution'
        )

        outdir.mkdir(
            parents=True,
            exist_ok=True
        )

        summary.to_csv(
            outdir / 'SRG_post_quenching_summary.csv',
            index=False,
        )

        formats = (
            ('png',)
            if args.png_only
            else ('png', 'pdf')
        )

        plot_post_quenching_mass_size(
            tracks,
            summary,
            outdir,
            formats,
        )

        plot_structural_displacements(
            summary,
            outdir,
            formats,
        )

        plot_post_quenching_mass_size_exsitu_panels(
            tracks,
            summary,
            outdir,
            formats,
        )

        print("Saved post-quenching evolution analysis.")

        return
    if args.formation_only:
        match, snap = load_formation_tables(args.formation_dir)
        if args.limit is not None:
            ids = sorted(match['track_id'].unique())[:args.limit]
            match = match[match['track_id'].isin(ids)].copy()
            snap = snap[snap['track_id'].isin(ids)].copy()
        print(f'Formation: {len(match)} galaxies; {int(match.red_nugget_detected.sum())} detected; {int(match.red_nugget_detected.eq(False).sum())} never detected')
        print('Caution: first detection is snapshot-limited; occupancy fractions use observed classifiable progenitors.')
        if not args.write:
            print('Dry run: no files written.')
            return
        outdir = args.output_dir or Path(OUTPUT['figures_dir']) / 'formation'
        for path in plot_formation_diagnostics(match, snap, outdir, formats=('png',) if args.png_only else ('png', 'pdf'), overwrite=args.overwrite):
            print('Saved:', path)
        return
    df = load_history(args.input_file)
    if args.limit is not None:
        ids = sorted(df['track_id'].unique())[:args.limit]
        df = df[df['track_id'].isin(ids)].copy()
    print(f'Loaded {len(df)} galaxy–epoch rows; {df.track_id.nunique()} tracks; '
          f'{df.redshift.nunique()} represented redshifts')
    for pop, g in df.groupby('population'):
        print(f'  {pop}: {g.track_id.nunique()} galaxies, {len(g)} records')
    for key, (col, _, _) in QUANTITIES.items():
        print(f'  {key}: {np.isfinite(pd.to_numeric(df[col], errors="coerce")).sum()} finite values')
    print('Caution: epoch medians at high redshift are conditional on progenitor detection.')
    if not args.write:
        print('Dry run: no files written.')
        return
    outdir = args.output_dir or OUTPUT['figures_dir']
    for path in run_plots(df, outdir, formats=('png',) if args.png_only else ('png', 'pdf'),
                          overwrite=args.overwrite):
        print('Saved:', path)


# ============================================================
# FORMATION DIAGNOSTICS (drop-in addition to modules/plotting.py)
# ============================================================

def _formation_bool(series, name):
    """Parse saved CSV boolean columns without treating 'False' as truthy."""
    values = series.astype('string').str.strip().str.lower()
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    invalid = values.notna() & ~values.isin(mapping)
    if invalid.any():
        raise ValueError(f'Invalid values in {name}: {values[invalid].unique().tolist()}')
    return values.map(mapping).astype('boolean')


def load_formation_tables(directory=None):
    """Load the two saved outputs of modules.formation, checking consistency."""
    directory = project_path(directory or Path(OUTPUT['root']) / 'formation')
    match_path = directory / 'red_nugget_matching_epochs.csv'
    snap_path = directory / 'red_nugget_snapshots.csv'
    if not match_path.is_file() or not snap_path.is_file():
        raise FileNotFoundError('Formation catalogues missing; run python3 -m modules.formation --write first')
    match = pd.read_csv(match_path, dtype={'track_id': 'int64'})
    snap = pd.read_csv(snap_path, dtype={'track_id': 'int64'})
    required_match = {'track_id', 'population', 'z_match', 'red_nugget_detected',
                      'z_first_compact', 'z_first_quenched', 'onset_uncertain'}
    required_snap = {'track_id', 'population', 'redshift', 'is_red_nugget'}
    for name, df, required in [('matching epochs', match, required_match),
                               ('snapshots', snap, required_snap)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'{name} missing columns: {sorted(missing)}')
    if match.empty or match['track_id'].duplicated().any():
        raise ValueError('Matching epochs must have one row per galaxy')
    if snap.duplicated(['track_id', 'redshift']).any():
        raise ValueError('Duplicate track/redshift in formation snapshots')
    if set(snap['track_id']) != set(match['track_id']):
        raise ValueError('Formation snapshot and matching TrackID sets differ')
    for key in ['red_nugget_detected', 'onset_uncertain']:
        match[key] = _formation_bool(match[key], key)
    snap['is_red_nugget'] = _formation_bool(snap['is_red_nugget'], 'is_red_nugget')
    if match['red_nugget_detected'].isna().any():
        raise ValueError('Missing detection flag')
    if (match['red_nugget_detected'].to_numpy(dtype=bool) !=
            match['z_match'].notna().to_numpy()).any():
        raise ValueError('Detection flags disagree with z_match')
    first_from_snap = (snap.loc[snap['is_red_nugget'].eq(True).fillna(False)]
                       .groupby('track_id')['redshift'].max())
    detected = match.loc[match['red_nugget_detected'].eq(True)].set_index('track_id')
    if set(first_from_snap.index) != set(detected.index):
        raise ValueError('Detection flags disagree between formation files')
    if not np.allclose(first_from_snap.reindex(detected.index).to_numpy(dtype=float),
                       detected['z_match'].to_numpy(dtype=float), atol=1e-6):
        raise ValueError('First detected redshifts disagree between formation files')
    return match, snap


def _formation_save(fig, destination, formats):
    fig.tight_layout()
    saved = []
    for fmt in formats:
        path = destination.with_suffix('.' + fmt)
        fig.savefig(path, dpi=250, bbox_inches='tight')
        saved.append(path)
    plt.close(fig)
    return saved


def plot_formation_diagnostics(match, snap, outdir, formats=('png', 'pdf'), overwrite=False):
    """Four formation diagnostics and two reproducible numerical tables."""
    outdir = project_path(outdir)
    stems = ['first_compact_quenched_redshift', 'compact_quenched_fraction_vs_redshift',
             'cumulative_first_detection', 'first_compact_vs_first_quenched']
    expected = [outdir / f'{stem}.{fmt}' for stem in stems for fmt in formats]
    expected += [outdir / 'formation_onset_counts.csv', outdir / 'formation_epoch_counts.csv']
    existing = [str(path) for path in expected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Formation plot output(s) exist; use --overwrite: {existing}')
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    populations = sorted(match['population'].unique())
    all_z = np.sort(snap['redshift'].dropna().unique())[::-1]
    first_counts = (match.dropna(subset=['z_match'])
                    .groupby(['population', 'z_match']).size().rename('n_first').reset_index())
    grid = pd.MultiIndex.from_product([populations, all_z], names=['population', 'redshift']).to_frame(index=False)
    onset = grid.merge(first_counts.rename(columns={'z_match': 'redshift'}),
                       on=['population', 'redshift'], how='left')
    onset['n_first'] = onset['n_first'].fillna(0).astype(int)
    onset.to_csv(outdir / 'formation_onset_counts.csv', index=False)
    written.append(outdir / 'formation_onset_counts.csv')

    # A: first detection counts (each galaxy contributes once).
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(all_z))
    width = 0.82 / len(populations)
    for j, pop in enumerate(populations):
        subset = onset[onset['population'] == pop].set_index('redshift').reindex(all_z)
        ax.bar(x + (j - (len(populations)-1)/2)*width, subset['n_first'].to_numpy(),
               width=width, label=pop, color=COLORS.get(pop, f'C{j}'))
    ax.set_xticks(x, [f'{z:g}' for z in all_z], rotation=45, ha='right')
    ax.set_xlabel('First observed compact-and-quenched redshift')
    ax.set_ylabel('Number of galaxies (first detection only)')
    ax.set_title('First compact-and-quenched epoch')
    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    undetected = int(match['red_nugget_detected'].eq(False).sum())
    ax.text(0.99, 0.97, f'Never detected: {undetected}', ha='right', va='top',
            transform=ax.transAxes, fontsize=10)
    written.extend(_formation_save(fig, outdir / stems[0], formats))

    # B: instantaneous occupancy among observed progenitors, NOT onset.
    records = []
    for (pop, z), group in snap.groupby(['population', 'redshift'], sort=True):
        states = group['is_red_nugget']
        n_yes = int(states.eq(True).sum())
        n_known = int(states.notna().sum())
        n_observed = group['track_id'].nunique()
        records.append({'population': pop, 'redshift': z, 'n_compact_quenched': n_yes,
                        'n_classifiable': n_known, 'n_observed': n_observed,
                        'fraction_classifiable': n_yes / n_known if n_known else np.nan})
    occupancy = pd.DataFrame(records).sort_values(['population', 'redshift'])
    occupancy.to_csv(outdir / 'formation_epoch_counts.csv', index=False)
    written.append(outdir / 'formation_epoch_counts.csv')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, pop in enumerate(populations):
        g = occupancy[occupancy['population'] == pop].sort_values('redshift')
        ax.plot(g['redshift'], g['fraction_classifiable'], '-o', ms=4,
                color=COLORS.get(pop, f'C{j}'), label=pop)
    ax.set_ylim(-0.03, 1.03)
    ax.invert_xaxis()
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Compact and quenched / classifiable progenitors')
    ax.set_title('Compact-and-quenched fraction among observed progenitors')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    written.extend(_formation_save(fig, outdir / stems[1], formats))

    # C: cumulative first detection by each epoch; denominator is ALL targets.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, pop in enumerate(populations):
        g = onset[onset['population'] == pop].set_index('redshift').reindex(all_z)
        cumulative = np.cumsum(g['n_first'].to_numpy(dtype=int))
        n_total = int((match['population'] == pop).sum())
        ax.step(all_z, cumulative / n_total, where='post', lw=2,
                color=COLORS.get(pop, f'C{j}'), label=f'{pop} (N={n_total})')
        ax.scatter(all_z, cumulative / n_total, s=13, color=COLORS.get(pop, f'C{j}'))
    ax.invert_xaxis()
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Fraction of z=0 reference galaxies detected by this epoch')
    ax.set_title('Cumulative first compact-and-quenched detections')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    written.extend(_formation_save(fig, outdir / stems[2], formats))

    # D: first compact vs first quenched; points can overlap on discrete epochs.
    fig, ax = plt.subplots(figsize=(7, 6.5))
    both = match.dropna(subset=['z_first_compact', 'z_first_quenched'])
    for j, pop in enumerate(populations):
        g = both[both['population'] == pop]
        if not g.empty:
            ax.scatter(g['z_first_compact'], g['z_first_quenched'], s=25,
                       alpha=0.35, color=COLORS.get(pop, f'C{j}'), label=f'{pop} (N={len(g)})')
    if not both.empty:
        bound = max(both['z_first_compact'].max(), both['z_first_quenched'].max()) + 0.2
        ax.plot([0, bound], [0, bound], '--', color='black', lw=1, label='Same first observed epoch')
        ax.set_xlim(-0.1, bound)
        ax.set_ylim(-0.1, bound)
    ax.set_xlabel('First observed compact redshift')
    ax.set_ylabel('First observed quenched redshift')
    ax.set_title('Ordering of compactness and quenching')
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    written.extend(_formation_save(fig, outdir / stems[3], formats))
    return written


if __name__ == '__main__':
    main()