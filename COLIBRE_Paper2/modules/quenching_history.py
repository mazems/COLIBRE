#!/usr/bin/env python3
"""Analyse observed quenching, rejuvenation, and alternative matching epochs.

Run from COLIBRE_Paper2:
    python3 -m modules.quenching_history --dry-run --limit 20
    python3 -m modules.quenching_history --dry-run
    python3 -m modules.quenching_history --write
    python3 -m modules.quenching_history --write --overwrite

Reads the existing evolutionary_properties.csv and reuses formation.py's
physical classification (including the age-dependent sSFR threshold).
Never changes existing formation or matching files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT
from .formation import classify_snapshots, load_history, project_path, settings as formation_settings, _cosmology

DEFAULT_QUENCHING_HISTORY = {
    "minimum_consecutive_star_forming_snapshots": 1,
    "minimum_consecutive_quenched_snapshots": 1,
    "require_z0_quenched": True,
    "require_compact_at_final_quenching": False,
    "plot_max_tracks": 12,
}


def settings():
    try:
        from config import QUENCHING_HISTORY
    except ImportError:
        QUENCHING_HISTORY = {}
    cfg = {**DEFAULT_QUENCHING_HISTORY, **QUENCHING_HISTORY}
    for key in ("minimum_consecutive_star_forming_snapshots", "minimum_consecutive_quenched_snapshots", "plot_max_tracks"):
        value = cfg[key]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        cfg[key] = int(value)
    for key in ("require_z0_quenched", "require_compact_at_final_quenching"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be boolean")
    return cfg


def _runs(states):
    """Contiguous runs of 1=SF, 0=Q, -1=unknown, with inclusive endpoints."""
    if not len(states):
        return []
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(states) - 1]
    return [(int(a), int(b), int(states[a])) for a, b in zip(starts, ends)]


def _state(value):
    if pd.isna(value):
        return -1
    return 0 if bool(value) else 1  # is_quenched -> SF/Q


def _first_epoch(g, mask):
    selected = g.loc[mask]
    return float(selected.iloc[0]["redshift"]) if len(selected) else np.nan


def analyse(history: pd.DataFrame, fcfg=None, qcfg=None):
    """Return classified epochs, per-galaxy events, population report.

    Final quenching is defined from the LAST qualifying SF run, followed by
    >=min_q consecutive Q observations and NO later SF observations.
    An unknown observation after the transition makes persistence uncertain.
    If the earliest observed state is already Q and there is no SF episode,
    the final quenching transition is left undefined (left-censored).
    """
    fcfg = formation_settings() if fcfg is None else fcfg
    qcfg = settings() if qcfg is None else qcfg
    classified = classify_snapshots(history, fcfg)
    cosmo = _cosmology(fcfg["cosmology"])
    min_sf = qcfg["minimum_consecutive_star_forming_snapshots"]
    min_q = qcfg["minimum_consecutive_quenched_snapshots"]
    rows = []
    for tid, g0 in classified.groupby("track_id", sort=True):
        g = g0.sort_values("redshift", ascending=False).reset_index(drop=True)
        states = np.array([_state(x) for x in g["is_quenched"]], dtype=np.int8)
        runs = _runs(states)
        sf_runs = [(a, b) for a, b, s in runs if s == 1 and b - a + 1 >= min_sf]
        q_runs = [(a, b) for a, b, s in runs if s == 0 and b - a + 1 >= min_q]
        # Count Q -> SF transitions only when both adjacent observations are known.
        rejuvenation_indices = [i for i in range(1, len(states)) if states[i-1] == 0 and states[i] == 1]
        qualifying_rejuvenations = [a for a, b in sf_runs if a > 0 and states[a-1] == 0]
        last_sf_idx = sf_runs[-1][1] if sf_runs else None
        final_q_idx = None
        status = "not_determined"
        final_q_persistent = False
        if last_sf_idx is None:
            status = "already_quenched_no_observed_sf" if np.any(states == 0) else "no_qualifying_sf_run"
        else:
            # Only a Q run after last qualifying SF run may define final Q.
            later_q = [(a, b) for a, b in q_runs if a > last_sf_idx]
            if not later_q:
                status = "no_qualifying_quenched_run_after_last_sf"
            else:
                candidate = later_q[0][0]
                tail = states[candidate:]
                if np.any(tail == 1):
                    status = "later_star_formation_detected"
                elif np.any(tail == -1):
                    status = "unknown_later_quenching_state"
                else:
                    final_q_idx = candidate
                    final_q_persistent = True
                    status = "final_quenching_bracketed" if candidate > 0 and states[candidate-1] == 1 else "final_quenching_unbracketed"
        # Optional strict z=0 requirement. Do not call a z=0.1 measurement z=0.
        z0_rows = g[np.isclose(g["redshift"].to_numpy(float), 0.0, atol=1e-6)]
        z0_q = (bool(z0_rows.iloc[0]["is_quenched"]) if len(z0_rows) and pd.notna(z0_rows.iloc[0]["is_quenched"]) else None)
        if qcfg["require_z0_quenched"] and z0_q is not True:
            final_q_idx = None
            final_q_persistent = False
            status = "z0_not_observed_quenched"
        final_row = g.iloc[final_q_idx] if final_q_idx is not None else None
        first_cq = g.loc[g["is_red_nugget"].fillna(False)]
        first_cq_row = first_cq.iloc[0] if len(first_cq) else None
        first_q_z = _first_epoch(g, g["is_quenched"].fillna(False))
        first_c_z = _first_epoch(g, g["is_compact"].fillna(False))
        z_old = float(first_cq_row["redshift"]) if first_cq_row is not None else np.nan
        z_new = float(final_row["redshift"]) if final_row is not None else np.nan
        compact_final = (bool(final_row["is_compact"]) if final_row is not None and pd.notna(final_row["is_compact"]) else None)
        match_eligible = bool(final_row is not None and (compact_final is True or not qcfg["require_compact_at_final_quenching"]))
        if np.isfinite(z_old) and np.isfinite(z_new):
            dt = float(cosmo.age(z_new).value - cosmo.age(z_old).value)
        else:
            dt = np.nan
        # An SF run can end immediately before a Q run; the true event is between snapshots.
        bracket_z_sf = float(g.iloc[final_q_idx-1]["redshift"]) if final_q_idx is not None and final_q_idx > 0 and states[final_q_idx-1] == 1 else np.nan
        rows.append({
            "track_id": int(tid), "population": str(g.iloc[0]["population"]),
            "n_observed_epochs": int(len(g)), "n_unknown_quenching_epochs": int((states == -1).sum()),
            "z_first_compact": first_c_z, "z_first_quenched": first_q_z,
            "z_first_compact_quenched": z_old,
            "z_last_star_forming": float(g.iloc[last_sf_idx]["redshift"]) if last_sf_idx is not None else np.nan,
            "z_final_quenching": z_new,
            "z_final_quenching_prior_star_forming": bracket_z_sf,
            "final_quenching_status": status,
            "final_quenching_persistent_observed": final_q_persistent,
            "compact_at_final_quenching": compact_final,
            "compact_quenched_at_final_quenching": bool(final_row is not None and compact_final is True),
            "eligible_for_alternative_matching": match_eligible,
            "n_rejuvenation_transitions_observed": len(rejuvenation_indices),
            "n_rejuvenation_episodes_qualifying": len(qualifying_rejuvenations),
            "has_observed_rejuvenation": bool(rejuvenation_indices),
            "has_qualifying_rejuvenation": bool(qualifying_rejuvenations),
            "quenched_at_z0": z0_q,
            "delta_t_finalQ_minus_firstCQ_gyr": dt,
            "stellar_mass_finalQ_msun": float(final_row["stellar_mass_msun"]) if final_row is not None else np.nan,
            "rhalf_finalQ_kpc": float(final_row["rhalf_physical_kpc"]) if final_row is not None else np.nan,
            "compactness_finalQ": float(final_row["compactness_sigma_1p5"]) if final_row is not None else np.nan,
            "ssfr_finalQ_per_year": float(final_row["ssfr_instant"]) if final_row is not None else np.nan,
        })
    events = pd.DataFrame(rows)
    report = {
        "n_galaxies": int(len(events)), "n_epoch_rows": int(len(classified)),
        "n_with_first_compact_quenched": int(events["z_first_compact_quenched"].notna().sum()),
        "n_with_final_quenching_transition": int(events["z_final_quenching"].notna().sum()),
        "n_with_qualifying_rejuvenation": int(events["has_qualifying_rejuvenation"].sum()),
        "n_with_any_observed_Q_to_SF_transition": int(events["has_observed_rejuvenation"].sum()),
        "n_eligible_for_alternative_matching": int(events["eligible_for_alternative_matching"].sum()),
        "n_compact_at_final_quenching": int(events["compact_quenched_at_final_quenching"].sum()),
        "final_quenching_status_counts": {str(k): int(v) for k, v in events["final_quenching_status"].value_counts().items()},
        "quenching_history_configuration": qcfg,
        "formation_configuration": fcfg,
        "note": "Discrete observed epochs only; no continuous transition times. First-CQ and final-Q are different scientific definitions. No existing matching files modified.",
    }
    return classified, events, report


def make_plots(classified, events, outdir, qcfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)

    # Revised matching-epoch histogram
    plot_matching_epoch_histogram(
        events,
        classified,
        outdir,
    )

    valid = events.dropna(subset=["z_first_compact_quenched", "z_final_quenching"])
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(valid):
        ax.scatter(valid["z_first_compact_quenched"], valid["z_final_quenching"], s=24, alpha=.55)
        top = max(float(valid["z_first_compact_quenched"].max()), float(valid["z_final_quenching"].max()))
        ax.plot([0, top], [0, top], "k--", lw=1, label="same observed epoch")
        ax.legend(frameon=False)
    ax.set(xlabel="First compact + quenched redshift", ylabel="Final quenching redshift", title=f"Epoch comparison (N={len(valid)})")
    ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"firstCQ_vs_finalQ.{suffix}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    dt = events["delta_t_finalQ_minus_firstCQ_gyr"].dropna().to_numpy(float)
    if len(dt):
        ax.hist(dt, bins=min(35, max(8, int(np.sqrt(len(dt))*2))), edgecolor="white")
    ax.axvline(0, ls="--", color="black", label="same epoch")
    ax.set(xlabel="Cosmic time(final Q) - time(first CQ) [Gyr]", ylabel="Number of galaxies", title=f"Reference epoch shift (N={len(dt)})")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"matching_epoch_time_shift.{suffix}", dpi=220)
    plt.close(fig)

    # Rejuvenating tracks are most informative; deterministic selection.
    ids = events.loc[events["has_observed_rejuvenation"], "track_id"].astype(int).head(qcfg["plot_max_tracks"]).tolist()
    if not ids:
        ids = events["track_id"].astype(int).head(qcfg["plot_max_tracks"]).tolist()
    fig, ax = plt.subplots(figsize=(9, 6))
    for tid in ids:
        g = classified.loc[classified["track_id"] == tid].sort_values("redshift", ascending=False)
        s = g["ssfr_instant"].to_numpy(float)
        positive = np.isfinite(s) & (s > 0)
        y = np.full(len(s), np.nan)
        y[positive] = np.log10(s[positive])
        ax.plot(g["redshift"], y, marker="o", ms=3, lw=1.2, label=str(tid))
    zgrid = np.linspace(0, max(0.1, float(classified["redshift"].max())), 250)
    cosmo = _cosmology(formation_settings()["cosmology"])
    qfactor = float(formation_settings()["quenching_factor"])
    threshold = qfactor / (cosmo.age(zgrid).value * 1e9)
    ax.plot(zgrid, np.log10(threshold), "k--", lw=2, label="quenching threshold")
    ax.set(xlabel="Redshift", ylabel=r"$\log_{10}(\mathrm{sSFR}/\mathrm{yr}^{-1})$", title="Selected observed sSFR histories (zero SFR omitted on log axis)")
    ax.invert_xaxis()
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(outdir / f"rejuvenation_ssfr_examples.{suffix}", dpi=220)
    plt.close(fig)

def plot_matching_epoch_histogram(events, classified, outdir):
    """
    Histogram of the last observed star-forming snapshot immediately
    preceding the final accepted quenching transition.

    Uses the actual available snapshot redshifts as discrete categories.

    Only galaxies with a bracketed final quenching transition are
    included. Missing matching epochs are reported separately.

    Produces:
        last_star_forming_epoch_histogram.png
        last_star_forming_epoch_histogram.pdf
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Available snapshot redshifts
    # ----------------------------------------------------------

    redshifts = np.sort(
        classified["redshift"]
        .dropna()
        .unique()
    )[::-1]

    # ----------------------------------------------------------
    # 2. Select valid alternative matching epochs
    # ----------------------------------------------------------

    valid = events[
        (events["final_quenching_status"] == "final_quenching_bracketed")
        & events["z_final_quenching_prior_star_forming"].notna()
    ].copy()

    z_match = valid[
        "z_final_quenching_prior_star_forming"
    ].to_numpy(dtype=float)

    # ----------------------------------------------------------
    # 3. Count galaxies at each redshift
    # ----------------------------------------------------------

    counts = np.array([
        np.count_nonzero(
            np.isclose(z_match, z, atol=1e-6)
        )
        for z in redshifts
    ])

    # ----------------------------------------------------------
    # 4. Plot
    # ----------------------------------------------------------

    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": 13,
    })

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    x = np.arange(len(redshifts))

    ax.bar(
        x,
        counts,
        width=0.82,
        color="C1",
        label="SRG",
    )

    # ----------------------------------------------------------
    # 5. Formatting
    # ----------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        [f"{z:g}" for z in redshifts],
        rotation=45,
        ha="right",
    )

    ax.set_xlabel(
        "Last star-forming redshift before final quenching"
    )

    ax.set_ylabel(
        "Number of galaxies"
    )

    ax.set_title(
        "Last star-forming epoch before final quenching"
    )

    n_total = len(events)
    n_valid = len(valid)
    n_missing = n_total - n_valid

    ax.text(
        0.98,
        0.96,
        f"No accepted transition: {n_missing}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.set_axisbelow(True)
    fig.tight_layout()

    # ----------------------------------------------------------
    # 6. Save
    # ----------------------------------------------------------

    for suffix in ("png", "pdf"):
        fig.savefig(
            outdir / f"last_star_forming_epoch_histogram.{suffix}",
            dpi=250,
            bbox_inches="tight",
        )
    plt.close(fig)

    # ----------------------------------------------------------
    # 7. Numerical summary
    # ----------------------------------------------------------

    print("\nLAST STAR-FORMING MATCHING EPOCH")
    print("=" * 50)

    print("Total SRGs:", n_total)
    print("Valid revised matching epochs:", n_valid)
    print("Without accepted transition:", n_missing)
    if n_valid:
        print(
            "Median matching redshift:",
            np.median(z_match)
        )
        print(
            "16th/84th percentiles:",
            np.percentile(z_match, [16, 84])
        )
    print("\nCounts by redshift:")
    for z, n in zip(redshifts, counts):
        if n > 0:
            print(f"z={z:5.2f}: {n:4d} galaxies")

def save_outputs(classified, events, report, outdir, overwrite=False):
    outdir = project_path(outdir)
    paths = [outdir / "quenching_history_snapshots.csv", outdir / "quenching_history_per_galaxy.csv", outdir / "quenching_history_report.json"]
    figures = [outdir / f"{stem}.{ext}" for stem in ("firstCQ_vs_finalQ", "matching_epoch_time_shift", "rejuvenation_ssfr_examples", "last_star_forming_epoch_histogram") for ext in ("png", "pdf")]
    existing = [str(p) for p in paths + figures if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output(s) already exist; use --write --overwrite: {existing}")
    outdir.mkdir(parents=True, exist_ok=True)
    classified.to_csv(paths[0], index=False)
    events.to_csv(paths[1], index=False)
    paths[2].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_plots(classified, events, outdir, report["quenching_history_configuration"])
    return paths + figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.overwrite and not args.write:
        parser.error("--overwrite requires --write")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.write and args.limit is not None:
        parser.error("Refusing to write a limited sample as production output")
    fcfg, qcfg = formation_settings(), settings()
    history = load_history(args.input_file)
    if args.limit is not None:
        ids = np.sort(history["track_id"].unique())[:args.limit]
        history = history.loc[history["track_id"].isin(ids)].copy()
    classified, events, report = analyse(history, fcfg, qcfg)
    print("QUENCHING HISTORY REPORT\n" + json.dumps(report, indent=2))
    print("\nFirst 10 galaxies:\n" + events[["track_id", "z_first_compact_quenched", "z_last_star_forming", "z_final_quenching", "final_quenching_status", "n_rejuvenation_episodes_qualifying"]].head(10).to_string(index=False))
    if args.write:
        directory = args.output_dir or Path(OUTPUT["root"]) / "quenching_history"
        for path in save_outputs(classified, events, report, directory, args.overwrite):
            print("Saved:", path)
    else:
        print("Dry run: no files written.")


if __name__ == "__main__":
    main()