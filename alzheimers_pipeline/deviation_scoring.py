"""
deviation_scoring.py
====================
Computes individual z-score deviation profiles relative to the healthy
control reference intervals.

For each participant × feature:

    z = (feature_value − HC_mean) / HC_std

Interpretation:
  • z ≈ 0   → within healthy range
  • z > +2   → abnormally HIGH compared to healthy speakers
  • z < −2   → abnormally LOW compared to healthy speakers

Additional summary statistics per participant:
  • abs_z_mean  – mean |z| across all scored features (global severity proxy)
  • max_abs_z   – maximum |z| (most deviant single feature)

Design decisions:
- Features where HC std = 0 or HC mean = NaN yield NaN z-scores and are
  excluded from abs_z_mean computation automatically via nanmean.
- Absolute z-scores (not signed) form the basis of the mean deviation score,
  matching the convention in published AD speech biomarker studies where
  direction of deviation varies by feature.
- The raw signed z-scores are retained for visualisation (radar plots).
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_deviations(
    features_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Compute z-score deviation scores for every participant.

    Parameters
    ----------
    features_df : pd.DataFrame
        Full feature table (all groups). Must contain 'speaker_id',
        'file_name', 'group', and all columns in feature_columns.
    reference_df : pd.DataFrame
        Output of healthy_reference.build_reference(), indexed by feature name.
    feature_columns : list[str]
        Features to score (must be a subset of reference_df.index).

    Returns
    -------
    pd.DataFrame with columns:
        speaker_id, file_name, group,
        z_<feature>  for each feature,
        abs_z_mean, max_abs_z
    """
    logger.info(
        "Computing deviation scores for %d participants × %d features.",
        len(features_df), len(feature_columns),
    )

    result_rows = []

    for _, row in features_df.iterrows():
        out = dict(
            speaker_id=row["speaker_id"],
            file_name=row["file_name"],
            group=row["group"],
        )

        z_values = []
        for feat in feature_columns:
            if feat not in reference_df.index:
                out[f"z_{feat}"] = np.nan
                continue

            hc_mean = reference_df.loc[feat, "mean"]
            hc_std = reference_df.loc[feat, "std"]
            val = pd.to_numeric(row.get(feat), errors="coerce")

            if pd.isna(hc_mean) or pd.isna(hc_std) or hc_std == 0 or pd.isna(val):
                out[f"z_{feat}"] = np.nan
            else:
                z = (val - hc_mean) / hc_std
                out[f"z_{feat}"] = float(z)
                z_values.append(abs(z))

        out["abs_z_mean"] = float(np.nanmean(z_values)) if z_values else np.nan
        out["max_abs_z"] = float(np.nanmax(z_values)) if z_values else np.nan

        result_rows.append(out)

    df = pd.DataFrame(result_rows)
    _log_deviation_summary(df)
    return df


def _log_deviation_summary(df: pd.DataFrame) -> None:
    """Log mean deviation scores by group."""
    for grp, grp_df in df.groupby("group"):
        valid = grp_df["abs_z_mean"].dropna()
        logger.info(
            "Group %s: mean |z| = %.3f ± %.3f  (n=%d)",
            grp, valid.mean(), valid.std(), len(valid),
        )
