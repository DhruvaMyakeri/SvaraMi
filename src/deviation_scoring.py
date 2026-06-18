"""
Module 3b — Deviation Scoring (In-Corpus Control Referencing)
Computes signed z-score deviation scores for PD patients
using in-corpus healthy controls as the reference population.

Confound gate result: RED — cross-corpus CLAC reference contaminated
by recording conditions. Using Oxford PD healthy controls (n=8) as
reference for both Oxford PD and Telemonitoring datasets.

Input:
  outputs/features/oxford_pd_features.csv
  outputs/features/telemonitoring_pd_features.csv

Output:
  outputs/deviation_scores/oxford_pd_deviation.csv
  outputs/deviation_scores/telemonitoring_pd_deviation.csv
  outputs/deviation_scores/pd_combined_deviation.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')


BASE       = r"D:\PROJECTS\Research\Speech-Disease-Observation"
OXFORD_PATH = os.path.join(BASE, "outputs", "features", "oxford_pd_features.csv")
TELE_PATH   = os.path.join(BASE, "outputs", "features", "telemonitoring_pd_features.csv")
OUT_DIR     = os.path.join(BASE, "outputs", "deviation_scores")
PLOT_DIR    = os.path.join(BASE, "outputs", "figures")
os.makedirs(OUT_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# Features available in both PD datasets
SHARED_FEATURES = [
    "meanF0", "HNR",
    "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter",
    "localShimmer", "localdbShimmer", "apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer",
]

# Friendly short names for plots
FEAT_LABELS = {
    "meanF0":              "F0 mean",
    "HNR":                 "HNR",
    "localJitter":         "Jitter\n(local)",
    "localabsoluteJitter": "Jitter\n(abs)",
    "rapJitter":           "Jitter\n(RAP)",
    "ppq5Jitter":          "Jitter\n(ppq5)",
    "localShimmer":        "Shimmer\n(local)",
    "localdbShimmer":      "Shimmer\n(dB)",
    "apq3Shimmer":         "Shimmer\n(apq3)",
    "aqpq5Shimmer":        "Shimmer\n(apq5)",
    "apq11Shimmer":        "Shimmer\n(apq11)",
}


# ─── IN-CORPUS REFERENCE STATS ────────────────────────────────────────────────

def compute_incorpus_reference(df, features, condition_col="condition", hc_label="HC"):
    """
    Compute mean and std of each feature from in-corpus healthy controls.
    Partitioned by gender where possible.
    Returns a dict: {(gender, feature): (mean, std)}
    """
    hc_df  = df[df[condition_col] == hc_label].copy()
    n_hc   = len(hc_df)
    print(f"  In-corpus HC speakers: {n_hc}")

    if n_hc == 0:
        raise ValueError("No healthy controls found in dataset.")

    ref_stats = {}
    genders   = ["male", "female"]

    for gender in genders:
        g_df = hc_df[hc_df["gender"] == gender] if "gender" in hc_df.columns else hc_df
        if len(g_df) < 2:
            # Fall back to all HC if gender subset too small
            g_df = hc_df
            print(f"  Warning: <2 HC for gender={gender}, using all HC as fallback")

        for feat in features:
            if feat not in g_df.columns:
                ref_stats[(gender, feat)] = (np.nan, np.nan)
                continue
            vals = g_df[feat].dropna()
            if len(vals) < 2:
                ref_stats[(gender, feat)] = (np.nan, np.nan)
                continue
            ref_stats[(gender, feat)] = (float(vals.mean()), float(vals.std()))

    return ref_stats, n_hc


def compute_deviation_scores(df, ref_stats, features, fallback_gender="male"):
    """
    Compute signed z-score deviations for each speaker.
    z = (value - HC_mean) / HC_std
    Positive = above HC mean, Negative = below HC mean.
    """
    records = []

    for _, row in df.iterrows():
        gender  = str(row.get("gender", fallback_gender)).lower()
        if gender not in ["male", "female"]:
            gender = fallback_gender

        record  = {
            "speaker_id": row.get("speaker_id", "unknown"),
            "condition":  row.get("condition",  "unknown"),
            "gender":     gender,
            "dataset":    row.get("dataset",    "unknown"),
        }

        # Add covariates if available
        for cov in ["age", "motor_UPDRS", "total_UPDRS"]:
            if cov in row.index:
                record[cov] = row[cov]

        for feat in features:
            if feat not in row.index or pd.isna(row[feat]):
                record[f"dev_{feat}"] = np.nan
                continue

            mean, std = ref_stats.get((gender, feat), (np.nan, np.nan))

            # Fallback to other gender if this gender's stats unavailable
            if np.isnan(mean):
                other = "female" if gender == "male" else "male"
                mean, std = ref_stats.get((other, feat), (np.nan, np.nan))

            if np.isnan(mean) or np.isnan(std) or std == 0:
                record[f"dev_{feat}"] = np.nan
                continue

            record[f"dev_{feat}"] = float((row[feat] - mean) / std)

        records.append(record)

    return pd.DataFrame(records)


# ─── RADAR PLOT ───────────────────────────────────────────────────────────────

def plot_radar(deviation_df, features, title, out_path, condition_colors=None):
    """
    Radar/spider plot of mean deviation per condition.
    Replicates Botelho et al. Figure 4 style.
    Green band = reference interval (±1.96 SD = 95% of healthy).
    Each line = one speaker's deviation profile.
    """
    if condition_colors is None:
        condition_colors = {"HC": "#4C72B0", "PD": "#DD4444"}

    labels     = [FEAT_LABELS.get(f, f) for f in features]
    n          = len(features)
    angles     = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles    += angles[:1]  # close the polygon

    fig, axes = plt.subplots(
        1, len(deviation_df["condition"].unique()),
        figsize=(6 * len(deviation_df["condition"].unique()), 6),
        subplot_kw=dict(polar=True)
    )
    if not hasattr(axes, '__len__'):
        axes = [axes]

    conditions = sorted(deviation_df["condition"].unique(),
                        key=lambda x: (x != "HC", x))

    for ax, cond in zip(axes, conditions):
        cond_df = deviation_df[deviation_df["condition"] == cond]
        color   = condition_colors.get(cond, "#888888")

        # Draw reference band (±1.96 = 95% healthy range in z-score space)
        ref_vals = [1.96] * n + [1.96]
        ax.fill(angles, ref_vals, alpha=0.15, color="green")
        ax.plot(angles, ref_vals, color="green", linewidth=1.5,
                linestyle="--", label="95% healthy range")
        ref_neg  = [-1.96] * n + [-1.96]
        ax.fill(angles, ref_neg, alpha=0.0)
        ax.plot(angles, ref_neg, color="green", linewidth=1.5, linestyle="--")

        # Draw each speaker
        for _, row in cond_df.iterrows():
            vals = [row.get(f"dev_{f}", np.nan) for f in features]
            # Skip if all NaN
            if all(np.isnan(v) for v in vals):
                continue
            # Replace NaN with 0 for plotting
            vals = [v if not np.isnan(v) else 0 for v in vals]
            vals += vals[:1]
            ax.plot(angles, vals, color=color, alpha=0.3, linewidth=0.8)

        # Draw mean deviation line
        mean_vals = [cond_df[f"dev_{f}"].mean() for f in features]
        mean_vals = [v if not np.isnan(v) else 0 for v in mean_vals]
        mean_vals += mean_vals[:1]
        ax.plot(angles, mean_vals, color=color, linewidth=2.5,
                label=f"{cond} mean (n={len(cond_df)})")
        ax.fill(angles, mean_vals, alpha=0.1, color=color)

        # Style
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(-4, 4)
        ax.set_yticks([-2, 0, 2])
        ax.set_yticklabels(["-2σ", "0", "+2σ"], fontsize=7)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_title(f"{cond}\n(n={len(cond_df)})",
                     fontsize=11, fontweight="bold", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Radar plot saved: {out_path}")


# ─── SUMMARY STATS ────────────────────────────────────────────────────────────

def print_deviation_summary(dev_df, features, dataset_name):
    """Print mean deviation per condition per feature."""
    print(f"\n{'='*70}")
    print(f"DEVIATION SUMMARY — {dataset_name}")
    print(f"{'='*70}")
    print(f"  {'Feature':<25} {'HC mean dev':>12} {'PD mean dev':>12} "
          f"{'Direction':>12}")
    print(f"  {'-'*65}")

    conditions = dev_df["condition"].unique()
    hc_df = dev_df[dev_df["condition"] == "HC"] if "HC" in conditions else None
    pd_df = dev_df[dev_df["condition"] == "PD"] if "PD" in conditions else dev_df

    for feat in features:
        col       = f"dev_{feat}"
        if col not in dev_df.columns:
            continue
        hc_mean   = hc_df[col].mean()  if hc_df  is not None else np.nan
        pd_mean   = pd_df[col].mean()  if pd_df  is not None else np.nan
        direction = ("↑ above healthy" if pd_mean > 0.3
                     else "↓ below healthy" if pd_mean < -0.3
                     else "→ within range")
        print(f"  {feat:<25} {hc_mean:>12.4f} {pd_mean:>12.4f} {direction:>12}")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("Loading datasets...")
    oxford_df = pd.read_csv(OXFORD_PATH)
    tele_df   = pd.read_csv(TELE_PATH)

    print(f"  Oxford PD : {len(oxford_df)} speakers "
          f"(PD={( oxford_df['condition']=='PD').sum()}, "
          f"HC={(oxford_df['condition']=='HC').sum()})")
    print(f"  Telemonitoring: {len(tele_df)} speakers (all PD)")

    # ── Oxford PD — in-corpus referencing ────────────────────────────────────
    print(f"\n{'='*60}")
    print("Oxford PD — In-Corpus Deviation Scoring")
    print(f"{'='*60}")

    ref_stats_oxford, n_hc = compute_incorpus_reference(
        oxford_df, SHARED_FEATURES
    )

    oxford_dev = compute_deviation_scores(
        oxford_df, ref_stats_oxford, SHARED_FEATURES
    )

    oxford_out = os.path.join(OUT_DIR, "oxford_pd_deviation.csv")
    oxford_dev.to_csv(oxford_out, index=False)
    print(f"  Saved: {oxford_out}")
    print_deviation_summary(oxford_dev, SHARED_FEATURES, "Oxford PD")

    # ── Telemonitoring — use Oxford HC as reference ───────────────────────────
    print(f"\n{'='*60}")
    print("Telemonitoring PD — Using Oxford HC as Reference")
    print(f"{'='*60}")
    print("  (Telemonitoring has no healthy controls — using Oxford HC stats)")

    tele_dev = compute_deviation_scores(
        tele_df, ref_stats_oxford, SHARED_FEATURES,
        fallback_gender="male"
    )

    tele_out = os.path.join(OUT_DIR, "telemonitoring_pd_deviation.csv")
    tele_dev.to_csv(tele_out, index=False)
    print(f"  Saved: {tele_out}")
    print_deviation_summary(tele_dev, SHARED_FEATURES, "Telemonitoring PD")

    # ── Combined PD deviation dataset ─────────────────────────────────────────
    combined = pd.concat([oxford_dev, tele_dev], ignore_index=True)
    combined_out = os.path.join(OUT_DIR, "pd_combined_deviation.csv")
    combined.to_csv(combined_out, index=False)
    print(f"\n  Combined saved: {combined_out}")
    print(f"  Total rows: {len(combined)}")

    # ── Radar plots ───────────────────────────────────────────────────────────
    print(f"\nGenerating radar plots...")

    # Oxford PD radar
    plot_radar(
        oxford_dev, SHARED_FEATURES,
        title="Oxford PD — Deviation from Healthy Controls\n(Sustained Vowel /a/)",
        out_path=os.path.join(PLOT_DIR, "radar_oxford_pd.png"),
        condition_colors={"HC": "#4C72B0", "PD": "#DD4444"}
    )

    # Telemonitoring radar (PD only)
    plot_radar(
        tele_dev, SHARED_FEATURES,
        title="Telemonitoring PD — Deviation from Oxford HC Reference\n(Sustained Vowel /a/)",
        out_path=os.path.join(PLOT_DIR, "radar_telemonitoring_pd.png"),
        condition_colors={"PD": "#DD4444"}
    )

    # Combined PD radar
    plot_radar(
        combined[combined["dataset"] == "oxford_pd"],
        SHARED_FEATURES,
        title="PD Speech Deviation Profile\n(In-Corpus Referenced, Sustained Vowel /a/)",
        out_path=os.path.join(PLOT_DIR, "radar_pd_combined.png"),
        condition_colors={"HC": "#4C72B0", "PD": "#DD4444"}
    )

    print(f"\n{'='*60}")
    print("MODULE 3b COMPLETE")
    print(f"{'='*60}")
    print(f"  Oxford deviation   : {oxford_out}")
    print(f"  Tele deviation     : {tele_out}")
    print(f"  Combined           : {combined_out}")
    print(f"  Radar plots        : {PLOT_DIR}")
    print(f"\nNext step: Module 4 — Characterisation analysis")
    print(f"  (Wait for ADReSS + Bridge2AI before running Module 4)")
    