"""
Module 1d — CLAC Metadata Merge + Outlier Removal
Merges speaker metadata (age, gender, education, symptoms) onto
extracted acoustic features, then removes outliers using
Mahalanobis distance following Botelho et al. (2024) Step 3.

Input:
  outputs/features/clac_cookie_theft_acoustic_features.csv
  outputs/features/clac_max_phonation_acoustic_features.csv
  outputs/features/clac_picnic_acoustic_features.csv
  CLAC-Dataset/CLAC-Dataset/metadata.xlsx

Output:
  outputs/features/clac_cookie_theft_merged.csv
  outputs/features/clac_max_phonation_merged.csv
  outputs/features/clac_picnic_merged.csv
  outputs/features/clac_all_merged.csv  (combined)
"""

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
import warnings
warnings.filterwarnings('ignore')


BASE        = r"D:\PROJECTS\Research\Speech-Disease-Observation"
FEATURE_DIR = os.path.join(BASE, "outputs", "features")
META_PATH   = os.path.join(BASE, "CLAC-Dataset", "CLAC-Dataset", "metadata.xlsx")
OUT_DIR     = os.path.join(BASE, "outputs", "features")


# ─── LOAD METADATA ────────────────────────────────────────────────────────────

def load_metadata(meta_path):
    df = pd.read_excel(meta_path)
    df = df.rename(columns={
        "speakerID":          "speaker_id",
        "age (years)":        "age",
        "gender":             "gender",
        "education (years)":  "education",
        "symptoms":           "symptoms"
    })
    df = df[["speaker_id","age","gender","education","symptoms",
             "worker_country","worker_city"]]

    print(f"\nMetadata loaded: {len(df)} speakers")
    print(f"  Gender distribution: {df['gender'].value_counts().to_dict()}")
    print(f"  Age range: {df['age'].min():.0f} – {df['age'].max():.0f} years")
    print(f"  Symptoms=yes: {(df['symptoms']=='yes').sum()}")
    print(f"  Education NaN: {df['education'].isna().sum()}")
    return df


# ─── OUTLIER REMOVAL ──────────────────────────────────────────────────────────

def remove_outliers_mahalanobis(df, feature_cols, threshold=3.0):
    """
    Remove outliers using Mahalanobis distance from population mean.
    Threshold: 3 standard deviations from mean of Mahalanobis distances.
    Following Botelho et al. Step 3 — conducted separately per task,
    using rhythm + voice quality + vocal tract features only
    (not content features, to ensure ASR-independence).

    Returns: (df_clean, df_outliers, n_removed)
    """
    # Use only numeric acoustic features, drop NaN rows for distance computation
    X = df[feature_cols].dropna(axis=1, how='any')
    valid_cols = X.columns.tolist()
    X = X.dropna()

    if len(X) < 10:
        print("  Too few valid rows for Mahalanobis — skipping outlier removal")
        return df, pd.DataFrame(), 0

    # Compute covariance matrix and its inverse
    mu   = X.mean().values
    cov  = np.cov(X.values.T)

    # Regularise covariance matrix to avoid singularity
    cov  = cov + np.eye(cov.shape[0]) * 1e-6
    try:
        cov_inv = inv(cov)
    except np.linalg.LinAlgError:
        print("  Singular covariance matrix — using diagonal regularisation")
        cov     = np.diag(np.diag(cov))
        cov_inv = inv(cov)

    # Compute Mahalanobis distance for each sample
    distances = []
    for idx in X.index:
        row  = X.loc[idx].values
        dist = mahalanobis(row, mu, cov_inv)
        distances.append((idx, dist))

    dist_df            = pd.DataFrame(distances, columns=["index","mahal_dist"])
    dist_mean          = dist_df["mahal_dist"].mean()
    dist_std           = dist_df["mahal_dist"].std()
    cutoff             = dist_mean + threshold * dist_std

    outlier_indices    = dist_df[dist_df["mahal_dist"] > cutoff]["index"].tolist()
    df_clean           = df[~df.index.isin(outlier_indices)].copy()
    df_outliers        = df[df.index.isin(outlier_indices)].copy()
    n_removed          = len(outlier_indices)

    return df_clean, df_outliers, n_removed


# ─── PROCESS ONE TASK ─────────────────────────────────────────────────────────

def process_task(task_name, metadata_df):
    """
    Load features for one task, merge metadata, remove outliers,
    filter out speakers who reported symptoms.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {task_name}")
    print(f"{'='*60}")

    feat_path = os.path.join(FEATURE_DIR, f"clac_{task_name}_acoustic_features.csv")
    if not os.path.exists(feat_path):
        print(f"  File not found: {feat_path}")
        return None

    df = pd.read_csv(feat_path)
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")

    # ── Merge metadata ────────────────────────────────────────────────────────
    df = df.merge(metadata_df, on="speaker_id", how="left")
    n_no_meta = df["gender"].isna().sum()
    print(f"  Merged metadata. Missing metadata: {n_no_meta} speakers")

    # ── Remove speakers who reported symptoms ─────────────────────────────────
    # Botelho et al. used self-reported healthy speakers only
    n_before = len(df)
    df = df[df["symptoms"] != "yes"].copy()
    n_removed_symptoms = n_before - len(df)
    print(f"  Removed {n_removed_symptoms} speakers who reported symptoms")

    # ── Age partitioning flag ─────────────────────────────────────────────────
    # Botelho: partitions by gender; age partition desired but data insufficient
    # We add age_group for future use
    df["age_group"] = df["age"].apply(
        lambda x: "lt50" if x < 50 else "gte50" if not pd.isna(x) else np.nan
    )

    # ── Outlier removal via Mahalanobis distance ──────────────────────────────
    # Use rhythm + voice quality + vocal tract features (not SNR, not content)
    acoustic_feature_cols = [
        "meanF0","stdevF0","HNR",
        "localJitter","localabsoluteJitter","rapJitter","ppq5Jitter",
        "localShimmer","localdbShimmer","apq3Shimmer","aqpq5Shimmer","apq11Shimmer",
        "F1_mean","F1_median","F2_mean","F2_median",
        "F3_mean","F3_median","F4_mean","F4_median",
        "speech_rate","articulation_rate","avg_syllable_duration",
        "mean_pause_duration","mean_speech_duration",
        "silence_rate","silence_to_speech_ratio","mean_silence_count"
    ]
    # Keep only columns that exist in this task's feature set
    feature_cols = [c for c in acoustic_feature_cols if c in df.columns]

    print(f"  Running Mahalanobis outlier removal on {len(feature_cols)} features...")
    df_clean, df_outliers, n_removed = remove_outliers_mahalanobis(
        df, feature_cols, threshold=3.0
    )

    pct_removed = n_removed / len(df) * 100 if len(df) > 0 else 0
    print(f"  Outliers removed : {n_removed} ({pct_removed:.1f}%)")
    print(f"  Clean speakers   : {len(df_clean)}")

    # ── Gender breakdown ──────────────────────────────────────────────────────
    gender_counts = df_clean["gender"].value_counts()
    print(f"  Gender breakdown : {gender_counts.to_dict()}")

    # ── Age breakdown ─────────────────────────────────────────────────────────
    age_counts = df_clean["age_group"].value_counts()
    print(f"  Age groups       : {age_counts.to_dict()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, f"clac_{task_name}_merged.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"  Saved to         : {out_path}")

    return df_clean


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load metadata
    metadata_df = load_metadata(META_PATH)

    # Process all three tasks
    tasks   = ["cookie_theft", "max_phonation", "picnic"]
    results = {}

    for task in tasks:
        df_task = process_task(task, metadata_df)
        if df_task is not None:
            results[task] = df_task

    # Combine all tasks into one master file
    if results:
        df_all = pd.concat(results.values(), ignore_index=True)
        all_path = os.path.join(OUT_DIR, "clac_all_merged.csv")
        df_all.to_csv(all_path, index=False)

        print(f"\n{'='*60}")
        print("MERGE + OUTLIER REMOVAL COMPLETE")
        print(f"{'='*60}")
        for task, df in results.items():
            print(f"  {task:20s}  {len(df):4d} speakers")
        print(f"  {'combined':20s}  {len(df_all):4d} rows total")
        print(f"\n  Master file saved to: {all_path}")

        # Final feature summary
        print(f"\nFeature columns in merged files:")
        feat_cols = [c for c in df_all.columns if c not in [
            "speaker_id","task","gender","age","education",
            "symptoms","worker_country","worker_city","age_group"
        ]]
        print(f"  {len(feat_cols)} features: {feat_cols}")