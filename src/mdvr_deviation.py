"""
Module 3d — MDVR-KCL Deviation Scoring (Full Features including Rhythm)
In-corpus referencing: HC from each task used as reference for PD.
Produces deviation scores for ReadText and SpontaneousDialogue separately,
then combines into one MDVR-KCL deviation CSV.

Run: python src/mdvr_deviation.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

BASE    = r"D:\PROJECTS\Research\Speech-Disease-Observation"
FEAT    = os.path.join(BASE, "outputs", "features")
OUT_DIR = os.path.join(BASE, "outputs", "deviation_scores")
os.makedirs(OUT_DIR, exist_ok=True)

ALL_FEATURES = [
    # Voice quality
    "meanF0", "stdevF0", "HNR",
    "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter",
    "localShimmer", "localdbShimmer", "apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer",
    # Vocal tract
    "F1_mean", "F1_median", "F2_mean", "F2_median",
    "F3_mean", "F3_median", "F4_mean", "F4_median",
    # Rhythm
    "speech_rate", "articulation_rate", "avg_syllable_duration",
    "mean_pause_duration", "mean_speech_duration",
    "silence_rate", "silence_to_speech_ratio", "mean_silence_count",
]

VOICE_QUALITY = [
    "meanF0", "HNR",
    "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter",
    "localShimmer", "localdbShimmer", "apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer",
]

RHYTHM = [
    "speech_rate", "articulation_rate", "avg_syllable_duration",
    "mean_pause_duration", "mean_speech_duration",
    "silence_rate", "silence_to_speech_ratio", "mean_silence_count",
]


def compute_deviation(hc_df, pd_df, features, task_label):
    """In-corpus deviation scoring using HC as reference."""
    ref = {}
    for feat in features:
        if feat not in hc_df.columns:
            continue
        vals = hc_df[feat].dropna()
        if len(vals) >= 2:
            ref[feat] = (vals.mean(), vals.std())

    records = []
    for df, cond in [(hc_df, "HC"), (pd_df, "PD")]:
        for _, row in df.iterrows():
            rec = {
                "speaker_id": row.get("speaker_id", "unknown"),
                "condition":  cond,
                "task":       task_label,
                "dataset":    "mdvr_kcl",
            }
            for feat in features:
                if feat not in row.index or pd.isna(row[feat]) or feat not in ref:
                    rec[f"dev_{feat}"] = np.nan
                    continue
                mean, std = ref[feat]
                rec[f"dev_{feat}"] = float((row[feat] - mean) / std) if std > 0 else np.nan
            records.append(rec)

    return pd.DataFrame(records)


def print_summary(dev_df, features, label):
    hc  = dev_df[dev_df["condition"] == "HC"]
    pd_ = dev_df[dev_df["condition"] == "PD"]
    print(f"\n  {'Feature':<28} {'HC dev':>8} {'PD dev':>8}  Direction")
    print(f"  {'-'*65}")
    for feat in features:
        col = f"dev_{feat}"
        if col not in dev_df.columns:
            continue
        hc_m  = hc[col].mean()
        pd_m  = pd_[col].mean()
        direction = "above healthy" if pd_m > 0.3 else "below healthy" if pd_m < -0.3 else "within range"
        print(f"  {feat:<28} {hc_m:>8.3f} {pd_m:>8.3f}  {direction}")


all_records = []

for task in ["readtext", "spontaneous"]:
    hc_path = os.path.join(FEAT, f"mdvr_mdvr_{task}_hc_full_features.csv")
    pd_path = os.path.join(FEAT, f"mdvr_mdvr_{task}_pd_full_features.csv")

    if not os.path.exists(hc_path) or not os.path.exists(pd_path):
        print(f"  Skipping {task} — files not found")
        continue

    hc_df = pd.read_csv(hc_path)
    pd_df = pd.read_csv(pd_path)

    print(f"\n{'='*65}")
    print(f"MDVR-KCL {task.upper()} — n_HC={len(hc_df)}, n_PD={len(pd_df)}")
    print(f"{'='*65}")

    dev_df = compute_deviation(hc_df, pd_df, ALL_FEATURES, task)
    all_records.append(dev_df)

    print(f"\n  VOICE QUALITY:")
    print_summary(dev_df, VOICE_QUALITY, task)
    print(f"\n  RHYTHM:")
    print_summary(dev_df, RHYTHM, task)

# Combine both tasks
combined = pd.concat(all_records, ignore_index=True)
out_path = os.path.join(OUT_DIR, "mdvr_kcl_deviation.csv")
combined.to_csv(out_path, index=False)
print(f"\nSaved combined deviation scores: {out_path}")
print(f"Total rows: {len(combined)}")

# PD only summary across both tasks
pd_combined = combined[combined["condition"] == "PD"]
print(f"\n{'='*65}")
print("COMBINED PD DEVIATION (both tasks averaged)")
print(f"{'='*65}")
for feat in VOICE_QUALITY + RHYTHM:
    col = f"dev_{feat}"
    if col not in pd_combined.columns:
        continue
    mean = pd_combined[col].mean()
    direction = "above healthy" if mean > 0.3 else "below healthy" if mean < -0.3 else "within range"
    print(f"  {feat:<28} {mean:>8.3f}  {direction}")