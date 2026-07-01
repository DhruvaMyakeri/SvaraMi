"""
extract_ah_features.py
======================
Extract acoustic features from HC_AH and PD_AH sustained vowel recordings
using the CLAC feature extractor (src/feature_extractor.py).

Output: outputs/features/ah_acoustic_features.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from feature_extractor import extract_all_acoustic
from tqdm import tqdm

BASE = r"D:\PROJECTS\Research\Speech-Disease-Observation"

GROUPS = {
    "HC": os.path.join(BASE, "HC_AH", "HC_AH"),
    "PD": os.path.join(BASE, "PD_AH", "PD_AH"),
}

OUTPUT_DIR = os.path.join(BASE, "outputs", "features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

records = []
errors = 0

for group, folder in GROUPS.items():
    wav_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
    print(f"\n{'='*60}")
    print(f"Group: {group}  |  Folder: {folder}  |  Files: {len(wav_files)}")
    print(f"{'='*60}")

    for wav_file in tqdm(wav_files, desc=group):
        wav_path = os.path.join(folder, wav_file)
        feats = extract_all_acoustic(wav_path, task="max_phonation")

        if feats is not None:
            feats["speaker_id"] = wav_file.replace(".wav", "")
            feats["group"] = group
            feats["task"] = "AH"
            records.append(feats)
        else:
            errors += 1

print(f"\n{'='*60}")
print(f"Total extracted: {len(records)}  |  Errors: {errors}")

if records:
    df = pd.DataFrame(records)
    cols = ["speaker_id", "group", "task"] + [
        c for c in df.columns if c not in ["speaker_id", "group", "task"]
    ]
    df = df[cols]

    out_path = os.path.join(OUTPUT_DIR, "ah_acoustic_features.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"Shape: {df.shape}")

    print("\nPer-group NaN rates for key features:")
    for feat in ["meanF0", "stdevF0", "HNR", "localJitter", "localShimmer"]:
        if feat in df.columns:
            for grp in ["HC", "PD"]:
                sub = df[df["group"] == grp][feat]
                print(f"  {grp} {feat}: mean={sub.mean():.4f}  NaN={sub.isna().mean():.1%}")
