"""
healthy_reference.py
====================
Constructs healthy control reference intervals for every speech feature.

These intervals serve as the normative baseline against which Alzheimer's
Disease participants are evaluated — analogous to clinical reference ranges
in laboratory medicine.

Statistics computed per feature (HC group only):
  • mean, std
  • median, IQR (Q25, Q75)
  • lower_95_bound  = mean − 1.96 × std
  • upper_95_bound  = mean + 1.96 × std

Design decisions:
- Only non-NaN observations from the HC group are used.
- Features with fewer than 3 valid HC observations generate a warning and are
  excluded from normative scoring (their reference stats are still saved).
- The 95% bounds are parametric (Gaussian assumption). This is standard in
  clinical speech studies where the Gaussian approximation is broadly
  acceptable. For small samples (<30), bootstrap CIs could be substituted.
- Reference statistics are also used as the z-score denominator in deviation
  scoring; features with std=0 are set to NaN to avoid division by zero.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_HC_OBSERVATIONS = 3  # warn if fewer valid HCs for a feature


def build_reference(features_df: pd.DataFrame,
                    feature_columns: list[str]) -> pd.DataFrame:
    """
    Build healthy control reference intervals.

    Parameters
    ----------
    features_df : pd.DataFrame
        Full feature table (all groups).  Must contain a 'group' column.
    feature_columns : list[str]
        Names of columns to include in the reference table.

    Returns
    -------
    pd.DataFrame indexed by feature name, with columns:
        mean, std, median, q25, q75, iqr,
        lower_95_bound, upper_95_bound, n_valid
    """
    hc_df = features_df.loc[features_df["group"] == "HC", feature_columns].copy()

    if hc_df.empty:
        raise ValueError(
            "No HC (Healthy Control) recordings found in feature table. "
            "Cannot build reference intervals."
        )

    logger.info(
        "Building reference intervals from %d HC recordings over %d features.",
        len(hc_df), len(feature_columns),
    )

    records = []
    for feat in feature_columns:
        col = pd.to_numeric(hc_df[feat], errors="coerce").dropna()
        n = len(col)

        if n < MIN_HC_OBSERVATIONS:
            logger.warning(
                "Feature '%s' has only %d valid HC observations (min=%d); "
                "reference stats may be unreliable.",
                feat, n, MIN_HC_OBSERVATIONS,
            )

        if n == 0:
            records.append(_empty_reference_row(feat))
            continue

        mean_ = float(col.mean())
        std_ = float(col.std(ddof=1)) if n > 1 else 0.0
        median_ = float(col.median())
        q25 = float(col.quantile(0.25))
        q75 = float(col.quantile(0.75))
        iqr = q75 - q25

        # Guard against zero std (constant feature)
        if std_ == 0.0:
            logger.warning(
                "Feature '%s' has std=0 among HC; z-scores will be NaN.", feat
            )
            lower_95 = mean_
            upper_95 = mean_
        else:
            lower_95 = mean_ - 1.96 * std_
            upper_95 = mean_ + 1.96 * std_

        records.append(dict(
            feature=feat,
            mean=mean_,
            std=std_,
            median=median_,
            q25=q25,
            q75=q75,
            iqr=iqr,
            lower_95_bound=lower_95,
            upper_95_bound=upper_95,
            n_valid=n,
        ))

    ref_df = pd.DataFrame(records).set_index("feature")
    logger.info("Reference table built: %d features.", len(ref_df))
    return ref_df


def _empty_reference_row(feature_name: str) -> dict:
    return dict(
        feature=feature_name,
        mean=np.nan, std=np.nan, median=np.nan,
        q25=np.nan, q75=np.nan, iqr=np.nan,
        lower_95_bound=np.nan, upper_95_bound=np.nan,
        n_valid=0,
    )
