"""
radar_plots.py
==============
Generates publication-quality radar (spider) plots of speech biomarker
deviation profiles, matching the style of published clinical studies
(e.g., Parkinson's / Alzheimer's speech deviation profiling papers).

Coordinate system
-----------------
The radar axes use SIGNED z-scores, shifted by +MAX_Z into non-negative
display space so Matplotlib's polar projection can render them:

    display_r = z_score + MAX_Z

  Centre of radar  →  z = 0    (display_r = MAX_Z)
  Inner dashed ring →  z = -2σ  (display_r = MAX_Z - 2)
  Outer dashed ring →  z = +2σ  (display_r = MAX_Z + 2)

This exactly replicates the reference plot's concentric ring labels
(−2σ at centre, 0 at mid-ring, +2σ at outer ring).

Aggregation
-----------
Before plotting, recordings are grouped by speaker_id and their z-scores
are averaged, yielding one profile per participant (not per file).

Overcrowding guard
------------------
At most MAX_INDIVIDUAL_TRACES participant lines are drawn. The group mean
is always computed from ALL participants.

Plots produced
--------------
  radar_hc.png        – HC individual (blue, faint) + HC mean (blue, thick) + band
  radar_ad.png        – AD individual (red, faint)  + AD mean (red, thick)  + band
  radar_comparison.png – HC mean vs AD mean + band
"""

import logging
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Axis labels
# ---------------------------------------------------------------------------
RADAR_FEATURE_LABELS = {
    "f0_mean":             "F0 mean",
    "f0_std":              "F0 SD",
    "f0_min":              "F0 min",
    "f0_max":              "F0 max",
    "hnr":                 "HNR",
    "jitter_local":        "Jitter\n(local)",
    "jitter_absolute":     "Jitter\n(abs)",
    "jitter_rap":          "Jitter\n(RAP)",
    "jitter_ppq5":         "Jitter\n(ppq5)",
    "shimmer_local":       "Shimmer\n(local)",
    "shimmer_db":          "Shimmer\n(dB)",
    "shimmer_apq3":        "Shimmer\n(apq3)",
    "shimmer_apq5":        "Shimmer\n(apq5)",
    "shimmer_apq11":       "Shimmer\n(apq11)",
    "f1_mean":             "F1",
    "f2_mean":             "F2",
    "f3_mean":             "F3",
    "f4_mean":             "F4",
    "recording_duration":  "Rec\nDur",
    "speaking_duration":   "Speak\nDur",
    "n_pauses":            "N\nPauses",
    "mean_pause_duration": "Mean\nPause",
    "max_pause_duration":  "Max\nPause",
    "total_pause_time":    "Total\nPause",
    "silence_ratio":       "Silence\nRatio",
    "speech_rate":         "Speech\nRate",
    "articulation_rate":   "Artic\nRate",
    "phonation_time_ratio":"Phonation\nRatio",
    "total_words":         "N Words",
    "unique_words":        "Unique\nWords",
    "type_token_ratio":    "TTR",
    "mean_utterance_len":  "MUL",
    "avg_word_length":     "Avg Word\nLen",
}

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
HC_COLOR  = "#2166AC"   # blue
AD_COLOR  = "#D6604D"   # red
BAND_COLOR = "#4DAF4A"  # green

INDIVIDUAL_ALPHA = 0.15   # transparency of individual participant traces
INDIVIDUAL_LW    = 0.9    # line-width of individual traces
MEAN_LW          = 2.8    # line-width of group mean trace

MAX_Z                 = 4.0   # z-score axis limit (data units)
MAX_INDIVIDUAL_TRACES = 30    # cap on individual lines drawn

PANEL_BG = "#E8E8E8"   # light grey background matching reference style

RANDOM_SEED = 42  # reproducible sampling


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_all_radar_plots(
    deviation_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: str | Path,
) -> None:
    """
    Generate and save all three radar plots.

    Parameters
    ----------
    deviation_df : pd.DataFrame
        Output of deviation_scoring.compute_deviations().
        Must contain 'speaker_id', 'group', and 'z_<feature>' columns.
    feature_columns : list[str]
        Features to plot (same list used in deviation scoring).
    output_dir : str | Path
        Directory where PNG files are saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify z-score columns that actually exist
    z_cols = [f"z_{f}" for f in feature_columns if f"z_{f}" in deviation_df.columns]
    feat_names = [c[2:] for c in z_cols]   # strip 'z_'

    if not feat_names:
        logger.error("No z-score columns found in deviation DataFrame — skipping radar plots.")
        return

    labels = [RADAR_FEATURE_LABELS.get(f, f) for f in feat_names]

    # Aggregate to participant level (mean across recordings per speaker_id)
    hc_participants = _aggregate_participants(deviation_df, "HC", z_cols)
    ad_participants = _aggregate_participants(deviation_df, "AD", z_cols)

    n_hc = len(hc_participants)
    n_ad = len(ad_participants)
    logger.info("Participants after aggregation: HC=%d  AD=%d", n_hc, n_ad)

    # Compute group means from ALL participants (before sampling)
    hc_mean = _group_mean(hc_participants)
    ad_mean = _group_mean(ad_participants)

    # Sample for display only
    hc_display = _sample_for_display(hc_participants)
    ad_display  = _sample_for_display(ad_participants)

    # ------------------------------------------------------------------
    # Plot 1: Healthy Control profile
    # ------------------------------------------------------------------
    fig, ax = _make_radar_fig(
        title=f"HC\n(n={n_hc})",
        bg_color=PANEL_BG,
    )
    _draw_radar(
        ax, hc_display, hc_mean, labels,
        individual_color=HC_COLOR,
        mean_color=HC_COLOR,
        mean_label=f"HC mean (n={n_hc})",
        draw_band=True,
    )
    _finalize_legend(ax, extra_handles=[
        mpatches.Patch(facecolor=BAND_COLOR, alpha=0.45,
                       edgecolor=BAND_COLOR, linestyle="--",
                       label="95% healthy range"),
        plt.Line2D([0], [0], color=HC_COLOR, linewidth=MEAN_LW,
                   label=f"HC mean (n={n_hc})"),
    ])
    _save(fig, output_dir / "radar_hc.png")
    logger.info("Saved radar_hc.png")

    # ------------------------------------------------------------------
    # Plot 2: Alzheimer's Disease profile
    # ------------------------------------------------------------------
    fig, ax = _make_radar_fig(
        title=f"AD\n(n={n_ad})",
        bg_color=PANEL_BG,
    )
    _draw_radar(
        ax, ad_display, ad_mean, labels,
        individual_color=AD_COLOR,
        mean_color=AD_COLOR,
        mean_label=f"AD mean (n={n_ad})",
        draw_band=True,
    )
    _finalize_legend(ax, extra_handles=[
        mpatches.Patch(facecolor=BAND_COLOR, alpha=0.45,
                       edgecolor=BAND_COLOR, linestyle="--",
                       label="95% healthy range"),
        plt.Line2D([0], [0], color=AD_COLOR, linewidth=MEAN_LW,
                   label=f"AD mean (n={n_ad})"),
    ])
    _save(fig, output_dir / "radar_ad.png")
    logger.info("Saved radar_ad.png")

    # ------------------------------------------------------------------
    # Plot 3: Comparison
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # Plot 3: Parkinson-style side-by-side comparison
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(16, 8), dpi=150, facecolor=PANEL_BG)

    ax_hc = fig.add_subplot(121, polar=True, facecolor=PANEL_BG)
    ax_ad = fig.add_subplot(122, polar=True, facecolor=PANEL_BG)

    ax_hc.set_title(
        f"HC\n(n={n_hc})",
        fontsize=15,
        fontweight="bold",
        pad=22,
    )

    ax_ad.set_title(
        f"AD\n(n={n_ad})",
        fontsize=15,
        fontweight="bold",
        pad=22,
    )

    _draw_radar(
        ax_hc,
        hc_display,
        hc_mean,
        labels,
        individual_color=HC_COLOR,
        mean_color=HC_COLOR,
        mean_label=f"HC mean (n={n_hc})",
        draw_band=True,
    )

    _draw_radar(
        ax_ad,
        ad_display,
        ad_mean,
        labels,
        individual_color=AD_COLOR,
        mean_color=AD_COLOR,
        mean_label=f"AD mean (n={n_ad})",
        draw_band=True,
    )

    ax_hc.legend(
        handles=[
            mpatches.Patch(
                facecolor=BAND_COLOR,
                alpha=0.45,
                edgecolor=BAND_COLOR,
                linestyle="--",
                label="95% healthy range",
            ),
            plt.Line2D(
                [0], [0],
                color=HC_COLOR,
                linewidth=MEAN_LW,
                label=f"HC mean (n={n_hc})",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.32, 1.12),
        fontsize=8,
    )

    ax_ad.legend(
        handles=[
            mpatches.Patch(
                facecolor=BAND_COLOR,
                alpha=0.45,
                edgecolor=BAND_COLOR,
                linestyle="--",
                label="95% healthy range",
            ),
            plt.Line2D(
                [0], [0],
                color=AD_COLOR,
                linewidth=MEAN_LW,
                label=f"AD mean (n={n_ad})",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(1.32, 1.12),
        fontsize=8,
    )

    fig.suptitle(
        "AD Speech Deviation Profile\n(In-Corpus Referenced)",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "radar_comparison.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=PANEL_BG,
    )
    plt.close(fig)

    logger.info("Saved radar_comparison.png")




# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_participants(
    deviation_df: pd.DataFrame,
    group: str,
    z_cols: list[str],
) -> np.ndarray:
    """
    Group recordings by speaker_id, average z-scores, return N_participants × F array.

    NaN z-scores (feature extraction failed for that recording) are excluded
    from the mean via nanmean — a participant's axis is only NaN if ALL their
    recordings failed for that feature.
    """
    grp_df = deviation_df[deviation_df["group"] == group].copy()
    if grp_df.empty:
        logger.warning("No recordings for group '%s'.", group)
        return np.empty((0, len(z_cols)))

    numeric = grp_df[["speaker_id"] + z_cols].copy()
    for c in z_cols:
        numeric[c] = pd.to_numeric(numeric[c], errors="coerce")

    # Mean per speaker across their recordings
    agg = numeric.groupby("speaker_id")[z_cols].apply(
        lambda x: np.nanmean(x.values, axis=0)
    )

    # agg is a Series of arrays; stack into matrix
    mat = np.vstack(agg.values)          # shape: (n_participants, n_features)

    # Clip to axis limits (signed, so ±MAX_Z)
    mat = np.clip(mat, -MAX_Z, MAX_Z)

    # Remaining NaN → 0 (sits at healthy centre; won't distort means)
    mat = np.where(np.isnan(mat), 0.0, mat)

    return mat


def _group_mean(participant_matrix: np.ndarray) -> np.ndarray | None:
    """Return mean profile across all participants, or None if empty."""
    if participant_matrix.shape[0] == 0:
        return None
    return np.mean(participant_matrix, axis=0)


def _sample_for_display(participant_matrix: np.ndarray) -> np.ndarray:
    """Randomly sample ≤ MAX_INDIVIDUAL_TRACES rows for visual rendering."""
    n = participant_matrix.shape[0]
    if n <= MAX_INDIVIDUAL_TRACES:
        return participant_matrix
    rng = random.Random(RANDOM_SEED)
    indices = rng.sample(range(n), MAX_INDIVIDUAL_TRACES)
    return participant_matrix[indices]


# ---------------------------------------------------------------------------
# Coordinate system helpers
# ---------------------------------------------------------------------------
# Matplotlib polar axes cannot display negative radii cleanly.
# We shift every z-score by +MAX_Z so:
#
#   z = -MAX_Z  →  display_r = 0           (absolute pole)
#   z = -2      →  display_r = MAX_Z - 2   (inner reference ring)
#   z =  0      →  display_r = MAX_Z       (healthy centre ring)
#   z = +2      →  display_r = MAX_Z + 2   (outer reference ring)
#   z = +MAX_Z  →  display_r = 2*MAX_Z     (axis boundary)
#
# All ytick labels are then set to show the actual z-score values.

def _z_to_r(z: float | np.ndarray) -> float | np.ndarray:
    """Convert signed z-score(s) to non-negative display radius."""
    return z + MAX_Z


def _make_angles(n_feats: int) -> list[float]:
    """Return angles list (closed loop) for n_feats radar axes."""
    angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
    return angles + angles[:1]


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_band(ax, n_feats: int) -> None:
    """
    Draw the ±2σ healthy reference band and ring markers.

    Inner ring  →  z = −2  (display_r = MAX_Z − 2)
    Outer ring  →  z = +2  (display_r = MAX_Z + 2)
    Shaded fill between them.
    """
    angles = _make_angles(n_feats)
    inner_r = _z_to_r(-2.0)
    outer_r = _z_to_r(+2.0)

    inner_vals = [inner_r] * (n_feats + 1)
    outer_vals = [outer_r] * (n_feats + 1)

    # Green shaded band
    ax.fill_between(angles, inner_vals, outer_vals,
                    color=BAND_COLOR, alpha=0.30)
    # Dashed boundary lines
    ax.plot(angles, inner_vals, color=BAND_COLOR, linewidth=1.2,
            linestyle="--", alpha=0.9)
    ax.plot(angles, outer_vals, color=BAND_COLOR, linewidth=1.2,
            linestyle="--", alpha=0.9)


def _draw_individual_traces(
    ax,
    display_matrix: np.ndarray,
    color: str,
) -> None:
    """Draw faint individual participant lines."""
    if display_matrix.shape[0] == 0:
        return
    n_feats = display_matrix.shape[1]
    angles  = _make_angles(n_feats)
    for row in display_matrix:
        r_vals = _z_to_r(row).tolist() + [_z_to_r(row[0])]
        ax.plot(angles, r_vals, color=color,
                alpha=INDIVIDUAL_ALPHA, linewidth=INDIVIDUAL_LW)


def _draw_mean_line(
    ax,
    mean_vec: np.ndarray | None,
    labels: list[str],
    color: str,
    label: str,
) -> None:
    """Draw the thick group mean line."""
    if mean_vec is None or len(mean_vec) == 0:
        return
    n_feats = len(labels)
    angles  = _make_angles(n_feats)
    r_vals  = _z_to_r(mean_vec).tolist() + [_z_to_r(mean_vec[0])]
    ax.plot(angles, r_vals, color=color, linewidth=MEAN_LW,
            label=label, zorder=10)
    ax.fill(angles, r_vals, color=color, alpha=0.08)


def _draw_radar(
    ax,
    display_matrix: np.ndarray,
    mean_vec: np.ndarray | None,
    labels: list[str],
    individual_color: str,
    mean_color: str,
    mean_label: str,
    draw_band: bool,
) -> None:
    """Compose a full radar: optional band, individual traces, mean line, axis formatting."""
    n_feats = len(labels)

    if draw_band:
        _draw_band(ax, n_feats)

    _draw_individual_traces(ax, display_matrix, individual_color)
    _draw_mean_line(ax, mean_vec, labels, mean_color, mean_label)

    # Axis ticks and labels
    angles = _make_angles(n_feats)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, color="#222222")

    # Radial axis: display range [0, 2*MAX_Z]; label with actual z values
    r_min = 0.0           # display pole  = z = -MAX_Z
    r_max = 2.0 * MAX_Z   # display edge  = z = +MAX_Z

    ax.set_ylim(r_min, r_max)

    # Tick positions in display coords → z labels
    z_ticks = [-2, 0, 2]          # z-score values to label
    r_ticks = [_z_to_r(z) for z in z_ticks]
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{z:+d}σ" if z != 0 else "0" for z in z_ticks],
                       fontsize=8, color="dimgrey")

    ax.grid(color="grey", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines["polar"].set_visible(False)


# ---------------------------------------------------------------------------
# Figure / legend / save
# ---------------------------------------------------------------------------

def _make_radar_fig(title: str, bg_color: str = "white") -> tuple:
    """Create a polar-projection figure with styled background."""
    fig = plt.figure(figsize=(9, 9), dpi=150, facecolor=bg_color)
    ax  = fig.add_subplot(111, polar=True, facecolor=bg_color)
    ax.set_facecolor(bg_color)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=22,
                 color="#111111")
    return fig, ax


def _finalize_legend(ax, extra_handles: list) -> None:
    """Place a clean legend in the upper-right area."""
    ax.legend(
        handles=extra_handles,
        loc="upper right",
        bbox_to_anchor=(1.38, 1.18),
        fontsize=9,
        framealpha=0.88,
        edgecolor="#CCCCCC",
    )


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)