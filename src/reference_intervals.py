"""
Module 2 — Reference Interval Estimation
Computes reference intervals (2.5th - 97.5th percentiles) for each feature
from the CLAC healthy reference population, partitioned by gender and task.
Bootstrap 90% confidence intervals on RI limits (1000 resamples).
Follows Botelho et al. (2024) Step 5 exactly.

Input:  outputs/features/clac_all_merged.csv
Output: outputs/reference_intervals/reference_intervals.csv
        outputs/reference_intervals/reference_intervals_summary.txt
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


BASE    = r"D:\PROJECTS\Research\Speech-Disease-Observation"
IN_PATH = os.path.join(BASE, "outputs", "features", "clac_all_merged.csv")
OUT_DIR = os.path.join(BASE, "outputs", "reference_intervals")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── ACOUSTIC FEATURE COLUMNS ─────────────────────────────────────────────────

FEATURE_COLS = [
    # Voice quality
    "meanF0", "stdevF0", "HNR",
    "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter",
    "localShimmer", "localdbShimmer", "apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer",
    # Vocal tract
    "F1_mean", "F1_median", "F2_mean", "F2_median",
    "F3_mean", "F3_median", "F4_mean", "F4_median",
    # Rhythm (spontaneous speech tasks only)
    "speech_rate", "articulation_rate", "avg_syllable_duration",
    "mean_pause_duration", "mean_speech_duration",
    "silence_rate", "silence_to_speech_ratio", "mean_silence_count",
]


# ─── BOOTSTRAP CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(values, stat_fn, n_boot=1000, ci=90):
    """
    Compute bootstrap confidence interval on a statistic.
    Returns (lower_ci, upper_ci) at the given CI level.
    """
    values   = values[~np.isnan(values)]
    if len(values) < 10:
        return np.nan, np.nan

    boot_stats = []
    rng        = np.random.default_rng(seed=42)
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(stat_fn(sample))

    alpha  = (100 - ci) / 2
    lo     = np.percentile(boot_stats, alpha)
    hi     = np.percentile(boot_stats, 100 - alpha)
    return lo, hi


# ─── COMPUTE REFERENCE INTERVALS ──────────────────────────────────────────────

def compute_reference_intervals(df, feature_cols, n_boot=1000):
    """
    For each (task, gender, feature) combination compute:
      - RI lower limit (2.5th percentile)
      - RI upper limit (97.5th percentile)
      - Bootstrap 90% CI on both limits
      - Mean and std of the feature
      - N (sample size)

    Excludes 'other' gender as per Botelho (male/female only).
    Returns a DataFrame of reference intervals.
    """
    records = []

    # Only male and female — Botelho partitions by these two
    genders = ["male", "female"]
    tasks   = df["task"].unique()

    for task in tasks:
        for gender in genders:
            mask   = (df["task"] == task) & (df["gender"] == gender)
            subset = df[mask]
            n      = len(subset)

            print(f"\n  task={task}  gender={gender}  n={n}")

            for feat in feature_cols:
                vals = subset[feat].dropna().values

                if len(vals) < 10:
                    # Not enough data
                    records.append({
                        "task": task, "gender": gender, "feature": feat,
                        "n": n, "n_valid": len(vals),
                        "mean": np.nan, "std": np.nan,
                        "RI_lower": np.nan, "RI_upper": np.nan,
                        "CI_lower_lo": np.nan, "CI_lower_hi": np.nan,
                        "CI_upper_lo": np.nan, "CI_upper_hi": np.nan,
                    })
                    continue

                mean_val = float(np.mean(vals))
                std_val  = float(np.std(vals))
                ri_lo    = float(np.percentile(vals, 2.5))
                ri_hi    = float(np.percentile(vals, 97.5))

                # Bootstrap CIs on the RI limits
                ci_lo_lo, ci_lo_hi = bootstrap_ci(
                    vals, lambda x: np.percentile(x, 2.5),  n_boot=n_boot
                )
                ci_hi_lo, ci_hi_hi = bootstrap_ci(
                    vals, lambda x: np.percentile(x, 97.5), n_boot=n_boot
                )

                records.append({
                    "task":        task,
                    "gender":      gender,
                    "feature":     feat,
                    "n":           n,
                    "n_valid":     len(vals),
                    "mean":        mean_val,
                    "std":         std_val,
                    "RI_lower":    ri_lo,
                    "RI_upper":    ri_hi,
                    "CI_lower_lo": ci_lo_lo,
                    "CI_lower_hi": ci_lo_hi,
                    "CI_upper_lo": ci_hi_lo,
                    "CI_upper_hi": ci_hi_hi,
                })

    return pd.DataFrame(records)


# ─── VISUALISE ────────────────────────────────────────────────────────────────

def plot_ri_summary(ri_df, out_dir):
    """
    Plot RI width per feature per gender for a quick visual sanity check.
    Wider RI = more variable feature in healthy population.
    """
    ri_df["RI_width"] = ri_df["RI_upper"] - ri_df["RI_lower"]

    for task in ri_df["task"].unique():
        subset = ri_df[ri_df["task"] == task].copy()
        subset = subset.dropna(subset=["RI_width"])

        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 6))
        colors  = {"male": "#4C72B0", "female": "#DD8452"}
        x       = np.arange(len(subset["feature"].unique()))
        feats   = sorted(subset["feature"].unique())
        width   = 0.35

        for i, gender in enumerate(["male", "female"]):
            g_data  = subset[subset["gender"] == gender].set_index("feature")
            heights = [g_data.loc[f, "RI_width"] if f in g_data.index else 0
                       for f in feats]
            ax.bar(x + i * width, heights, width,
                   label=gender, color=colors[gender], alpha=0.8)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("RI Width (97.5th - 2.5th percentile)")
        ax.set_title(f"Reference Interval Width per Feature — {task}")
        ax.legend()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"ri_width_{task}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved plot: {out_path}")


def print_summary(ri_df):
    """Print a readable summary of key reference intervals."""
    print(f"\n{'='*70}")
    print("REFERENCE INTERVAL SUMMARY — Key Features")
    print(f"{'='*70}")

    key_features = ["meanF0", "HNR", "localJitter", "localShimmer", "speech_rate"]
    tasks        = ri_df["task"].unique()

    for task in tasks:
        print(f"\nTask: {task}")
        print(f"  {'Feature':<25} {'Gender':<8} {'N':>5}  "
              f"{'Mean':>8}  {'RI Lower':>10}  {'RI Upper':>10}")
        print(f"  {'-'*70}")
        for feat in key_features:
            for gender in ["male", "female"]:
                row = ri_df[
                    (ri_df["task"] == task) &
                    (ri_df["gender"] == gender) &
                    (ri_df["feature"] == feat)
                ]
                if row.empty or pd.isna(row["RI_lower"].values[0]):
                    continue
                print(f"  {feat:<25} {gender:<8} {row['n'].values[0]:>5}  "
                      f"{row['mean'].values[0]:>8.4f}  "
                      f"{row['RI_lower'].values[0]:>10.4f}  "
                      f"{row['RI_upper'].values[0]:>10.4f}")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading CLAC merged data...")
    df = pd.read_csv(IN_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Tasks: {df['task'].unique()}")
    print(f"  Genders: {df['gender'].value_counts().to_dict()}")

    # Only use features available in this dataset
    feat_cols = [f for f in FEATURE_COLS if f in df.columns]
    print(f"\nComputing reference intervals for {len(feat_cols)} features...")
    print(f"Partitioned by: task × gender")
    print(f"Bootstrap resamples: 1000")
    print(f"RI: 2.5th – 97.5th percentile")
    print(f"CI: 90% on RI limits\n")

    ri_df = compute_reference_intervals(df, feat_cols, n_boot=1000)

    # Save
    out_path = os.path.join(OUT_DIR, "reference_intervals.csv")
    ri_df.to_csv(out_path, index=False)
    print(f"\nReference intervals saved to: {out_path}")
    print(f"Total rows: {len(ri_df)}  "
          f"({ri_df['task'].nunique()} tasks × "
          f"{ri_df['gender'].nunique()} genders × "
          f"{ri_df['feature'].nunique()} features)")

    # Summary
    print_summary(ri_df)

    # Plots
    print("\nGenerating RI width plots...")
    plot_ri_summary(ri_df, OUT_DIR)

    print(f"\n{'='*60}")
    print("MODULE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Reference intervals: {out_path}")
    print(f"  Plots saved to     : {OUT_DIR}")