"""
Module 1c — Parkinson's Telemonitoring Dataset Loader
Maps pre-extracted features from the UCI Telemonitoring PD dataset
to the standardised Botelho et al. (2024) feature names.

Input:  parkinsons+telemonitoring/parkinsons_updrs.data
Output: outputs/features/telemonitoring_pd_features.csv

Feature mapping:
  Jitter(%)         → localJitter
  Jitter(Abs)       → localabsoluteJitter
  Jitter:RAP        → rapJitter
  Jitter:PPQ5       → ppq5Jitter
  Shimmer           → localShimmer
  Shimmer(dB)       → localdbShimmer
  Shimmer:APQ3      → apq3Shimmer
  Shimmer:APQ5      → aqpq5Shimmer
  Shimmer:APQ11     → apq11Shimmer
  HNR               → HNR
  age               → covariate
  sex               → covariate
  motor_UPDRS       → severity label
  total_UPDRS       → severity label

All subjects are PD patients (no healthy controls in this dataset).
Controls for deviation scoring come from CLAC reference intervals.
"""

import os
import pandas as pd
import numpy as np


def load_telemonitoring(data_path, output_dir):
    """
    Load telemonitoring dataset and map to standardised feature names.
    Aggregates multiple recordings per subject to one row per subject (mean).
    Returns a DataFrame in the same format as feature_extractor.py output.
    """
    print("\n" + "="*60)
    print("Loading Parkinson's Telemonitoring Dataset")
    print("="*60)

    df_raw = pd.read_csv(data_path)
    print(f"  Raw shape   : {df_raw.shape}")
    print(f"  Subjects    : {df_raw['subject#'].nunique()}")
    print(f"  Recordings  : {len(df_raw)}")
    print(f"  Columns     : {list(df_raw.columns)}")

    # ── Feature mapping ──────────────────────────────────────────────────────
    df_mapped = pd.DataFrame()

    df_mapped["speaker_id"] = df_raw["subject#"].astype(str).apply(
        lambda x: f"tele_spk{x.zfill(3)}"
    )
    df_mapped["task"]       = "sustained_vowel"
    df_mapped["condition"]  = "PD"   # all subjects are PD patients
    df_mapped["dataset"]    = "telemonitoring"

    # Covariates
    df_mapped["age"] = df_raw["age"]
    df_mapped["sex"] = df_raw["sex"].map({0: "M", 1: "F"})  # 0=male, 1=female per dataset docs

    # Severity labels
    df_mapped["motor_UPDRS"] = df_raw["motor_UPDRS"]
    df_mapped["total_UPDRS"] = df_raw["total_UPDRS"]

    # Direct feature mappings
    df_mapped["localJitter"]          = df_raw["Jitter(%)"]
    df_mapped["localabsoluteJitter"]  = df_raw["Jitter(Abs)"]
    df_mapped["rapJitter"]            = df_raw["Jitter:RAP"]
    df_mapped["ppq5Jitter"]           = df_raw["Jitter:PPQ5"]
    df_mapped["localShimmer"]         = df_raw["Shimmer"]
    df_mapped["localdbShimmer"]       = df_raw["Shimmer(dB)"]
    df_mapped["apq3Shimmer"]          = df_raw["Shimmer:APQ3"]
    df_mapped["aqpq5Shimmer"]         = df_raw["Shimmer:APQ5"]
    df_mapped["apq11Shimmer"]         = df_raw["Shimmer:APQ11"]
    df_mapped["HNR"]                  = df_raw["HNR"]

    # Features not available in this dataset — set to NaN
    unavailable = [
        "meanF0", "stdevF0",
        "F1_mean","F1_median","F2_mean","F2_median",
        "F3_mean","F3_median","F4_mean","F4_median",
        "speech_rate","articulation_rate","avg_syllable_duration",
        "mean_pause_duration","mean_speech_duration",
        "silence_rate","silence_to_speech_ratio","mean_silence_count",
        "SNR"
    ]
    for feat in unavailable:
        df_mapped[feat] = np.nan

    # ── Per-speaker aggregation ───────────────────────────────────────────────
    # ~200 recordings per patient — aggregate to one row per subject
    feature_cols = [
        c for c in df_mapped.columns
        if c not in ["speaker_id","task","condition","dataset","sex"]
    ]

    df_agg = df_mapped.groupby(
        ["speaker_id","task","condition","dataset","sex"]
    )[feature_cols].mean().reset_index()

    print(f"\n  After aggregation : {len(df_agg)} unique subjects")
    print(f"  Age range         : {df_agg['age'].min():.0f} – {df_agg['age'].max():.0f} years")
    print(f"  motor_UPDRS range : {df_agg['motor_UPDRS'].min():.1f} – {df_agg['motor_UPDRS'].max():.1f}")
    print(f"  total_UPDRS range : {df_agg['total_UPDRS'].min():.1f} – {df_agg['total_UPDRS'].max():.1f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "telemonitoring_pd_features.csv")
    df_agg.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")

    # Sanity check — print feature means
    for feat in ["HNR", "localJitter", "localShimmer", "apq11Shimmer"]:
        print(f"  {feat} mean = {df_agg[feat].mean():.4f}  "
              f"std = {df_agg[feat].std():.4f}")

    return df_agg


if __name__ == "__main__":

    BASE      = r"D:\PROJECTS\Research\Speech-Disease-Observation"
    DATA_PATH = os.path.join(BASE, "parkinsons+telemonitoring", "parkinsons_updrs.data")
    OUT_DIR   = os.path.join(BASE, "outputs", "features")

    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
    else:
        df = load_telemonitoring(DATA_PATH, OUT_DIR)
        print(f"\nDone. Shape: {df.shape}")