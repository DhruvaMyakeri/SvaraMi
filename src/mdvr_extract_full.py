"""
Re-extract MDVR-KCL features with rhythm forced on for all tasks.
Run: python src/mdvr_extract_full.py
"""
import sys, os
sys.path.insert(0, 'src')
import parselmouth
import numpy as np
import pandas as pd
from tqdm import tqdm
from feature_extractor import (
    extract_f0, extract_hnr, extract_jitter,
    extract_shimmer, extract_formants,
    extract_rhythm, extract_snr
)

BASE = r"D:\PROJECTS\Research\Speech-Disease-Observation\26_29_09_2017_KCL\26-29_09_2017_KCL"
OUT  = r"D:\PROJECTS\Research\Speech-Disease-Observation\outputs\features"

FOLDERS = [
    ("mdvr_readtext_hc",    os.path.join(BASE, "ReadText",           "HC")),
    ("mdvr_readtext_pd",    os.path.join(BASE, "ReadText",           "PD")),
    ("mdvr_spontaneous_hc", os.path.join(BASE, "SpontaneousDialogue","HC")),
    ("mdvr_spontaneous_pd", os.path.join(BASE, "SpontaneousDialogue","PD")),
]

for task_label, folder in FOLDERS:
    wavs    = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
    records = []

    for wav in tqdm(wavs, desc=task_label):
        path = os.path.join(folder, wav)
        try:
            sound = parselmouth.Sound(path)
            feats = {"speaker_id": wav.replace(".wav",""), "task": task_label}
            feats.update(extract_f0(sound))
            feats.update(extract_hnr(sound))
            feats.update(extract_jitter(sound))
            feats.update(extract_shimmer(sound))
            feats.update(extract_formants(sound))
            feats.update(extract_rhythm(sound))   # forced on for all tasks
            feats.update(extract_snr(sound))
            records.append(feats)
        except Exception as e:
            print(f"  ERROR: {wav} — {e}")

    df       = pd.DataFrame(records)
    out_path = os.path.join(OUT, f"mdvr_{task_label}_full_features.csv")
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} rows -> {out_path}")
    print(f"  speech_rate mean        = {df['speech_rate'].mean():.4f}  "
          f"NaN: {df['speech_rate'].isna().mean():.0%}")
    print(f"  mean_pause_duration     = {df['mean_pause_duration'].mean():.4f}  "
          f"NaN: {df['mean_pause_duration'].isna().mean():.0%}")
    print(f"  silence_rate            = {df['silence_rate'].mean():.4f}  "
          f"NaN: {df['silence_rate'].isna().mean():.0%}")
    print(f"  localShimmer mean       = {df['localShimmer'].mean():.4f}")
    print(f"  meanF0 mean             = {df['meanF0'].mean():.4f}")

print("\nDone.")