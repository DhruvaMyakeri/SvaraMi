"""
Module 1b — Oxford Parkinson's Dataset Loader
Maps pre-extracted features from the UCI Oxford PD dataset
to the standardised Botelho et al. (2024) feature names.

Input:  parkinsons/parkinsons.data
Output: outputs/features/oxford_pd_features.csv

Feature mapping:
  MDVP:Fo(Hz)       → meanF0
  MDVP:Jitter(%)    → localJitter
  MDVP:Jitter(Abs)  → localabsoluteJitter
  MDVP:RAP          → rapJitter
  MDVP:PPQ          → ppq5Jitter
  MDVP:Shimmer      → localShimmer
  MDVP:Shimmer(dB)  → localdbShimmer
  Shimmer:APQ3      → apq3Shimmer
  Shimmer:APQ5      → aqpq5Shimmer
  MDVP:APQ          → apq11Shimmer
  HNR               → HNR
  status            → label (0=healthy, 1=PD)

Not mapped (not in Botelho feature set):
  MDVP:Fhi, MDVP:Flo, NHR, RPDE, DFA, spread1, spread2, D2, PPE, Jitter:DDP
"""

import os
import pandas as pd
import numpy as np


def load_oxford_pd(data_path, output_dir):
    """
    Load Oxford PD dataset and map to standardised feature names.
    Returns a DataFrame in the same format as feature_extractor.py output.
    """
    print("\n" + "="*60)
    print("Loading Oxford Parkinson's Dataset")
    print("="*60)

    df_raw = pd.read_csv(data_path)
    print(f"  Raw shape: {df_raw.shape}")
    print(f"  Columns: {list(df_raw.columns)}")
    print(f"  PD subjects  : {(df_raw['status'] == 1).sum()}")
    print(f"  HC subjects  : {(df_raw['status'] == 0).sum()}")

    # ── Feature mapping ──────────────────────────────────────────────────────
    df_mapped = pd.DataFrame()

    # Speaker ID — extract from name column (e.g. "phon_R01_S01_1" → "R01_S01")
    df_mapped["speaker_id"] = df_raw["name"].apply(
        lambda x: "_".join(str(x).split("_")[1:3]) if "_" in str(x) else str(x)
    )
    df_mapped["task"]      = "sustained_vowel"
    df_mapped["condition"] = df_raw["status"].map({1: "PD", 0: "HC"})
    df_mapped["dataset"]   = "oxford_pd"

    # Direct feature mappings
    df_mapped["meanF0"]            = df_raw["MDVP:Fo(Hz)"]
    df_mapped["HNR"]               = df_raw["HNR"]
    df_mapped["localJitter"]       = df_raw["MDVP:Jitter(%)"]
    df_mapped["localabsoluteJitter"]= df_raw["MDVP:Jitter(Abs)"]
    df_mapped["rapJitter"]         = df_raw["MDVP:RAP"]
    df_mapped["ppq5Jitter"]        = df_raw["MDVP:PPQ"]
    df_mapped["localShimmer"]      = df_raw["MDVP:Shimmer"]
    df_mapped["localdbShimmer"]    = df_raw["MDVP:Shimmer(dB)"]
    df_mapped["apq3Shimmer"]       = df_raw["Shimmer:APQ3"]
    df_mapped["aqpq5Shimmer"]      = df_raw["Shimmer:APQ5"]
    df_mapped["apq11Shimmer"]      = df_raw["MDVP:APQ"]

    # Features not available in this dataset — set to NaN
    unavailable = [
        "stdevF0",
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
    # Oxford PD has multiple recordings per speaker — aggregate to one row per speaker
    feature_cols = [c for c in df_mapped.columns
                    if c not in ["speaker_id","task","condition","dataset"]]

    df_agg = df_mapped.groupby(
        ["speaker_id","task","condition","dataset"]
    )[feature_cols].mean().reset_index()

    print(f"\n  After aggregation: {len(df_agg)} unique speakers")
    print(f"  PD : {(df_agg['condition'] == 'PD').sum()}")
    print(f"  HC : {(df_agg['condition'] == 'HC').sum()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "oxford_pd_features.csv")
    df_agg.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")

    # Sanity check
    for feat in ["meanF0", "HNR", "localJitter", "localShimmer"]:
        print(f"  {feat}: PD mean={df_agg[df_agg.condition=='PD'][feat].mean():.4f}  "
              f"HC mean={df_agg[df_agg.condition=='HC'][feat].mean():.4f}")

    return df_agg


if __name__ == "__main__":

    BASE      = r"D:\PROJECTS\Research\Speech-Disease-Observation"
    DATA_PATH = os.path.join(BASE, "parkinsons", "parkinsons.data")
    OUT_DIR   = os.path.join(BASE, "outputs", "features")

    if not os.path.exists(DATA_PATH):
        print(f"File not found: {DATA_PATH}")
    else:
        df = load_oxford_pd(DATA_PATH, OUT_DIR)
        print(f"\nDone. Shape: {df.shape}")