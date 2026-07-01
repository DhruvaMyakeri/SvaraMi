"""
run_pipeline.py
===============
Main entry point for the Alzheimer's Disease Speech Biomarker Analysis Pipeline.

Usage
-----
    python run_pipeline.py --ad_dir /path/to/pitt_corpus --hc_dir /path/to/hc_data

Arguments
---------
    --ad_dir       Path to Alzheimer's Disease recordings (Pitt Corpus)
    --hc_dir       Path to Healthy Control recordings
    --output_dir   Output directory (default: ./outputs)
    --skip_dist    Skip per-feature distribution plots (faster for large datasets)
    --log_level    Logging verbosity: DEBUG | INFO | WARNING (default: INFO)

Pipeline stages
---------------
  1. Data discovery   – scan directories, build manifest
  2. Feature extraction – acoustic features via Parselmouth
  3. Transcript features – CHAT (.cha) parsing (if available)
  4. Feature merge     – combine acoustic + transcript into single table
  5. HC reference      – build normative reference intervals
  6. Deviation scoring – compute z-scores for all participants
  7. Radar plots       – 3 radar plot figures
  8. Heatmap           – z-score heatmap
  9. Distribution plots – per-feature boxplots and violin plots
  10. Report           – save summary statistics text file
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from data_loader import discover_dataset
from feature_extractor import extract_all_features, ACOUSTIC_FEATURE_COLUMNS
from transcript_feature_extractor import (
    extract_transcript_features, TRANSCRIPT_FEATURE_COLUMNS,
)
from healthy_reference import build_reference
from deviation_scoring import compute_deviations
from radar_plots import make_all_radar_plots
from heatmap_visualization import make_heatmap, make_distribution_plots


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="w"),
        ],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alzheimer's Disease Speech Biomarker Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ad_dir", required=True,
        help="Path to AD (Pitt Corpus) WAV directory",
    )
    parser.add_argument(
        "--hc_dir", required=True,
        help="Path to Healthy Control WAV directory",
    )
    parser.add_argument(
        "--output_dir", default="./outputs",
        help="Output directory (default: ./outputs)",
    )
    parser.add_argument(
        "--skip_dist", action="store_true",
        help="Skip per-feature distribution plots (faster)",
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(
    ad_dir: str,
    hc_dir: str,
    output_dir: str = "./outputs",
    skip_dist: bool = False,
) -> None:
    """
    Execute the full pipeline programmatically.
    Can be called from scripts, notebooks, or run_pipeline.py CLI.
    """
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "boxplots").mkdir(exist_ok=True)
    (out / "violinplots").mkdir(exist_ok=True)
    (out / "reports").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # STAGE 1: Discover files
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1/10  Data Discovery")
    logger.info("=" * 60)
    manifest = discover_dataset(ad_dir, hc_dir)
    manifest.to_csv(out / "manifest.csv", index=False)

    # ------------------------------------------------------------------
    # STAGE 2: Acoustic feature extraction
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2/10  Acoustic Feature Extraction")
    logger.info("=" * 60)
    acoustic_df = extract_all_features(manifest)

    # ------------------------------------------------------------------
    # STAGE 3: Transcript features
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3/10  Transcript Feature Extraction")
    logger.info("=" * 60)
    transcript_df = extract_transcript_features(manifest)

    # ------------------------------------------------------------------
    # STAGE 4: Merge into single feature table
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 4/10  Merging Features")
    logger.info("=" * 60)
    merge_keys = ["speaker_id", "file_name", "group"]
    features_df = acoustic_df.merge(
        transcript_df.drop(columns=["group"]),
        on=["speaker_id", "file_name"],
        how="left",
    )

    # Determine which feature columns are usable (exist and have at least one value)
    candidate_cols = ACOUSTIC_FEATURE_COLUMNS + TRANSCRIPT_FEATURE_COLUMNS
    feature_columns = [
        c for c in candidate_cols
        if c in features_df.columns and features_df[c].notna().any()
    ]
    logger.info("Active features: %d", len(feature_columns))

    # Save
    save_cols = merge_keys + feature_columns
    features_df[save_cols].to_csv(out / "extracted_features.csv", index=False)
    logger.info("Saved extracted_features.csv")

    # ------------------------------------------------------------------
    # STAGE 5: Healthy reference intervals
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 5/10  Building Healthy Reference Intervals")
    logger.info("=" * 60)
    reference_df = build_reference(features_df, feature_columns)
    reference_df.to_csv(out / "healthy_reference.csv")
    logger.info("Saved healthy_reference.csv")

    # ------------------------------------------------------------------
    # STAGE 6: Deviation scoring
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 6/10  Computing Deviation Scores")
    logger.info("=" * 60)
    deviation_df = compute_deviations(features_df, reference_df, feature_columns)
    deviation_df.to_csv(out / "deviation_scores.csv", index=False)
    logger.info("Saved deviation_scores.csv")

    # ------------------------------------------------------------------
    # STAGE 7: Radar plots
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 7/10  Radar Plots")
    logger.info("=" * 60)
    make_all_radar_plots(deviation_df, feature_columns, out)

    # ------------------------------------------------------------------
    # STAGE 8: Heatmap
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 8/10  Deviation Heatmap")
    logger.info("=" * 60)
    make_heatmap(deviation_df, feature_columns, out)

    # ------------------------------------------------------------------
    # STAGE 9: Distribution plots
    # ------------------------------------------------------------------
    if not skip_dist:
        logger.info("=" * 60)
        logger.info("STAGE 9/10  Distribution Plots")
        logger.info("=" * 60)
        make_distribution_plots(features_df, feature_columns, out)
    else:
        logger.info("STAGE 9/10  Distribution plots skipped (--skip_dist).")

    # ------------------------------------------------------------------
    # STAGE 10: Summary report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 10/10  Summary Report")
    logger.info("=" * 60)
    _write_report(features_df, reference_df, deviation_df,
                  feature_columns, out / "reports")

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1f seconds.", elapsed)
    logger.info("All outputs saved to: %s", out.resolve())


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    features_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    deviation_df: pd.DataFrame,
    feature_columns: list[str],
    report_dir: Path,
) -> None:
    """Write a plain-text summary of key statistics."""
    lines = [
        "=" * 70,
        "ALZHEIMER'S SPEECH BIOMARKER ANALYSIS — SUMMARY REPORT",
        "=" * 70,
        "",
        "SAMPLE",
        "-" * 40,
    ]

    for grp in ["HC", "AD"]:
        n = (features_df["group"] == grp).sum()
        lines.append(f"  {grp}: {n} recordings")
    lines.append("")

    lines += [
        "FEATURE COVERAGE",
        "-" * 40,
        f"  Active features: {len(feature_columns)}",
        "",
        "HEALTHY REFERENCE  (mean ± std)",
        "-" * 40,
    ]
    for feat in feature_columns:
        if feat in reference_df.index:
            m = reference_df.loc[feat, "mean"]
            s = reference_df.loc[feat, "std"]
            n = reference_df.loc[feat, "n_valid"]
            lines.append(f"  {feat:<30s}  {m:>10.4f}  ±  {s:.4f}  (n={n})")
    lines.append("")

    lines += [
        "MEAN DEVIATION SCORE (|z| mean across features)",
        "-" * 40,
    ]
    for grp, grp_df in deviation_df.groupby("group"):
        v = grp_df["abs_z_mean"].dropna()
        lines.append(
            f"  {grp}:  mean={v.mean():.3f}  std={v.std():.3f}  "
            f"min={v.min():.3f}  max={v.max():.3f}  (n={len(v)})"
        )
    lines.append("")
    lines.append("=" * 70)

    report_path = report_dir / "summary_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved summary_report.txt")

    # Also save per-group feature statistics
    group_stats = (
        features_df
        .groupby("group")[feature_columns]
        .describe()
        .T
    )
    group_stats.to_csv(report_dir / "group_feature_stats.csv")
    logger.info("Saved group_feature_stats.csv")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(args.log_level)

    try:
        run(
            ad_dir=args.ad_dir,
            hc_dir=args.hc_dir,
            output_dir=args.output_dir,
            skip_dist=args.skip_dist,
        )
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)
