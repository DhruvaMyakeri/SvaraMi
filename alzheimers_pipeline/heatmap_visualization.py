"""
heatmap_visualization.py
========================
Generates:
  1. deviation_heatmap.png – z-score heatmap (participants × features)
  2. boxplots/            – per-feature boxplot comparing HC vs AD
  3. violinplots/         – per-feature violin plot comparing HC vs AD

Design decisions:
- Heatmap uses a diverging colormap (coolwarm) centred at z=0; ±4 are the
  colour extremes. Missing values are shown as light grey.
- Participants are sorted by group (HC first) then by abs_z_mean within group,
  so clinical severity gradient is visible.
- Distribution plots are saved individually per feature to keep file sizes
  manageable and allow selective inclusion in reports.
- Seaborn's violin estimator uses kernel density estimation (default
  bandwidth) which can widen for small samples — this is expected.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

HC_COLOR = "#2166AC"
AD_COLOR = "#D6604D"
HEATMAP_VMAX = 4.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_heatmap(
    deviation_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: str | Path,
) -> None:
    """
    Create and save the deviation z-score heatmap.

    Parameters
    ----------
    deviation_df : pd.DataFrame
        Output of deviation_scoring.compute_deviations().
    feature_columns : list[str]
        Features to include (those with z_ columns).
    output_dir : str | Path
        Directory where deviation_heatmap.png is saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z_cols = [f"z_{f}" for f in feature_columns if f"z_{f}" in deviation_df.columns]
    if not z_cols:
        logger.error("No z-score columns found; skipping heatmap.")
        return

    # Sort: HC first, then by abs_z_mean ascending within group
    df = deviation_df.copy()
    df["_sort_group"] = (df["group"] == "AD").astype(int)
    df = df.sort_values(["_sort_group", "abs_z_mean"])

    z_mat = df[z_cols].apply(pd.to_numeric, errors="coerce").values
    z_mat = np.clip(z_mat, -HEATMAP_VMAX, HEATMAP_VMAX)

    row_labels = [f"{r['group'][:1]}_{r['speaker_id']}"
                  for _, r in df.iterrows()]
    col_labels = [c[2:] for c in z_cols]  # strip 'z_'

    n_rows, n_cols = z_mat.shape
    fig_height = max(8, n_rows * 0.22)
    fig_width = max(12, n_cols * 0.45)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)

    im = ax.imshow(
        z_mat,
        aspect="auto",
        cmap="coolwarm",
        vmin=-HEATMAP_VMAX,
        vmax=HEATMAP_VMAX,
        interpolation="nearest",
    )

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("z-score (deviation from HC mean)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Axes labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Participant  (HC | AD)", fontsize=11)
    ax.set_title(
        "Speech Biomarker Deviation Heatmap\n(z-scores relative to HC reference)",
        fontsize=13, fontweight="bold",
    )

    # Horizontal separator between HC and AD
    n_hc = (df["group"] == "HC").sum()
    if 0 < n_hc < n_rows:
        ax.axhline(n_hc - 0.5, color="black", linewidth=1.5, linestyle="--")
        ax.text(n_cols - 0.5, n_hc - 0.6, " AD", fontsize=8,
                color="black", va="bottom", ha="right")
        ax.text(n_cols - 0.5, n_hc + 0.1, " HC", fontsize=8,
                color="black", va="top", ha="right")

    fig.tight_layout()
    out_path = output_dir / "deviation_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_distribution_plots(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: str | Path,
) -> None:
    """
    Save per-feature boxplot and violin plot to separate subdirectories.

    Parameters
    ----------
    features_df : pd.DataFrame
        Full feature table with 'group' column.
    feature_columns : list[str]
        Feature names to plot.
    output_dir : str | Path
        Parent output directory; subfolders 'boxplots' and 'violinplots'
        are created automatically.
    """
    output_dir = Path(output_dir)
    box_dir = output_dir / "boxplots"
    vio_dir = output_dir / "violinplots"
    box_dir.mkdir(parents=True, exist_ok=True)
    vio_dir.mkdir(parents=True, exist_ok=True)

    palette = {"HC": HC_COLOR, "AD": AD_COLOR}

    for feat in feature_columns:
        if feat not in features_df.columns:
            continue

        plot_df = features_df[["group", feat]].copy()
        plot_df[feat] = pd.to_numeric(plot_df[feat], errors="coerce")
        plot_df = plot_df.dropna(subset=[feat])

        if plot_df.empty:
            continue

        # --- Boxplot ---
        fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
        sns.boxplot(
            data=plot_df, x="group", y=feat,
            order=["HC", "AD"], palette=palette,
            width=0.5, linewidth=1.2,
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
            ax=ax,
        )
        sns.stripplot(
            data=plot_df, x="group", y=feat,
            order=["HC", "AD"], palette=palette,
            size=3, alpha=0.3, jitter=True, ax=ax,
        )
        ax.set_title(f"{feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        fig.tight_layout()
        fig.savefig(box_dir / f"{feat}_boxplot.png", dpi=120,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # --- Violin plot ---
        fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
        _n_hc = (plot_df["group"] == "HC").sum()
        _n_ad = (plot_df["group"] == "AD").sum()
        if _n_hc >= 4 and _n_ad >= 4:
            sns.violinplot(
                data=plot_df, x="group", y=feat,
                order=["HC", "AD"], palette=palette,
                inner="quartile", linewidth=1.0,
                ax=ax,
            )
        else:
            # Fall back to boxplot if too few points for KDE
            sns.boxplot(
                data=plot_df, x="group", y=feat,
                order=["HC", "AD"], palette=palette, ax=ax,
            )
        sns.stripplot(
            data=plot_df, x="group", y=feat,
            order=["HC", "AD"], palette=palette,
            size=3, alpha=0.3, jitter=True, ax=ax,
        )
        ax.set_title(f"{feat}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        fig.tight_layout()
        fig.savefig(vio_dir / f"{feat}_violin.png", dpi=120,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    logger.info(
        "Saved %d boxplots and %d violin plots.",
        len(list(box_dir.glob("*.png"))),
        len(list(vio_dir.glob("*.png"))),
    )
