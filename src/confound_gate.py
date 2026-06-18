"""
Module 3a — Confound Gate
Checks whether CLAC healthy speakers and Oxford PD healthy controls
already separate on deviation scores BEFORE adding any patient data.

If they do separate significantly — the cross-corpus reference is
contaminated by age/recording conditions, not disease.
If they don't — we can proceed with CLAC as the universal reference.

Following Botelho et al. recommendation + Fable 5 critique.

Input:
  outputs/features/clac_all_merged.csv
  outputs/features/oxford_pd_features.csv
  outputs/reference_intervals/reference_intervals.csv

Output:
  outputs/confound_gate/confound_gate_results.csv
  outputs/confound_gate/confound_gate_summary.txt
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


BASE    = r"D:\PROJECTS\Research\Speech-Disease-Observation"
OUT_DIR = os.path.join(BASE, "outputs", "confound_gate")
os.makedirs(OUT_DIR, exist_ok=True)

RI_PATH      = os.path.join(BASE, "outputs", "reference_intervals", "reference_intervals.csv")
CLAC_PATH    = os.path.join(BASE, "outputs", "features", "clac_all_merged.csv")
OXFORD_PATH  = os.path.join(BASE, "outputs", "features", "oxford_pd_features.csv")


# ─── FEATURE COLUMNS TO CHECK ────────────────────────────────────────────────

# Only features available in Oxford PD dataset
SHARED_FEATURES = [
    "meanF0", "HNR",
    "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter",
    "localShimmer", "localdbShimmer", "apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer",
]


# ─── COMPUTE SIGNED DEVIATION SCORES ─────────────────────────────────────────

def compute_deviation_scores(df, ri_df, features, task="max_phonation"):
    """
    Compute signed z-score deviation from CLAC reference intervals.
    z = (value - RI_mean) / RI_std
    Positive = above healthy mean, Negative = below healthy mean.

    For subjects with unknown gender, use the overall mean/std.
    """
    records = []

    for _, row in df.iterrows():
        speaker_id = row.get("speaker_id", "unknown")
        gender     = str(row.get("gender", "")).lower()
        condition  = row.get("condition", "unknown")

        if gender not in ["male", "female"]:
            gender = "male"  # fallback

        deviations = {"speaker_id": speaker_id, "condition": condition, "gender": gender}

        for feat in features:
            if feat not in row.index or pd.isna(row[feat]):
                deviations[f"dev_{feat}"] = np.nan
                continue

            # Get reference mean and std for this feature/gender/task
            ri_row = ri_df[
                (ri_df["task"]    == task) &
                (ri_df["gender"]  == gender) &
                (ri_df["feature"] == feat)
            ]

            if ri_row.empty or pd.isna(ri_row["mean"].values[0]):
                deviations[f"dev_{feat}"] = np.nan
                continue

            ref_mean = ri_row["mean"].values[0]
            ref_std  = ri_row["std"].values[0]

            if ref_std == 0 or np.isnan(ref_std):
                deviations[f"dev_{feat}"] = np.nan
                continue

            z_score = (row[feat] - ref_mean) / ref_std
            deviations[f"dev_{feat}"] = float(z_score)

        records.append(deviations)

    return pd.DataFrame(records)


# ─── CONFOUND GATE TEST ───────────────────────────────────────────────────────

def run_confound_gate(clac_df, oxford_df, ri_df):
    """
    Compare CLAC healthy vs Oxford PD healthy controls on deviation scores.
    Uses Mann-Whitney U test (non-parametric, same as Botelho).
    If p < 0.05 for many features — confound is significant.
    """
    print("\nComputing deviation scores for CLAC healthy speakers...")

    # Use max_phonation task for comparison (sustained vowel — same as Oxford PD task)
    clac_vowel = clac_df[clac_df["task"] == "max_phonation"].copy()
    clac_vowel["condition"] = "CLAC_HC"

    print(f"  CLAC vowel speakers: {len(clac_vowel)}")

    # Oxford PD healthy controls only
    oxford_hc = oxford_df[oxford_df["condition"] == "HC"].copy()
    print(f"  Oxford HC speakers : {len(oxford_hc)}")

    if len(oxford_hc) == 0:
        print("  No healthy controls in Oxford PD dataset — cannot run confound gate")
        return None

    # Compute deviation scores for both groups
    clac_dev   = compute_deviation_scores(clac_vowel,  ri_df, SHARED_FEATURES, "max_phonation")
    oxford_dev = compute_deviation_scores(oxford_hc,   ri_df, SHARED_FEATURES, "max_phonation")

    # ── Statistical comparison ────────────────────────────────────────────────
    results = []
    print(f"\n{'='*70}")
    print("CONFOUND GATE RESULTS — CLAC Healthy vs Oxford PD Healthy Controls")
    print(f"{'='*70}")
    print(f"  {'Feature':<25} {'CLAC mean':>10} {'Oxford HC mean':>15} "
          f"{'p-value':>10} {'Significant':>12}")
    print(f"  {'-'*70}")

    n_significant = 0

    for feat in SHARED_FEATURES:
        dev_col = f"dev_{feat}"
        if dev_col not in clac_dev.columns or dev_col not in oxford_dev.columns:
            continue

        clac_vals   = clac_dev[dev_col].dropna().values
        oxford_vals = oxford_dev[dev_col].dropna().values

        if len(clac_vals) < 5 or len(oxford_vals) < 5:
            continue

        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(clac_vals, oxford_vals, alternative="two-sided")
        sig      = p < 0.05
        if sig:
            n_significant += 1

        print(f"  {feat:<25} {clac_vals.mean():>10.4f} {oxford_vals.mean():>15.4f} "
              f"{p:>10.4f} {'*** YES' if sig else 'no':>12}")

        results.append({
            "feature":        feat,
            "clac_mean_dev":  clac_vals.mean(),
            "oxford_hc_mean_dev": oxford_vals.mean(),
            "p_value":        p,
            "significant":    sig
        })

    results_df = pd.DataFrame(results)

    # ── Verdict ───────────────────────────────────────────────────────────────
    n_total = len(results)
    pct_sig  = n_significant / n_total * 100 if n_total > 0 else 0

    print(f"\n  Significant features: {n_significant}/{n_total} ({pct_sig:.0f}%)")
    print(f"\n{'='*70}")

    if pct_sig < 30:
        verdict = "GREEN"
        message = ("CLAC and Oxford HC do not substantially separate on deviation "
                   "scores. Cross-corpus reference is viable. Proceed with CLAC "
                   "as the universal reference baseline.")
    elif pct_sig < 60:
        verdict = "AMBER"
        message = ("Moderate separation detected. Some features may be confounded "
                   "by recording conditions or demographics. Use in-corpus control "
                   "referencing for those specific features. Proceed with caution.")
    else:
        verdict = "RED"
        message = ("Strong separation detected. Cross-corpus reference is "
                   "contaminated. Switch to in-corpus control referencing for "
                   "all cross-condition claims.")

    print(f"  CONFOUND GATE VERDICT: {verdict}")
    print(f"  {message}")
    print(f"{'='*70}")

    # Save results
    out_path = os.path.join(OUT_DIR, "confound_gate_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    # Save verdict to text file
    txt_path = os.path.join(OUT_DIR, "confound_gate_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"CONFOUND GATE VERDICT: {verdict}\n\n")
        f.write(f"Significant features: {n_significant}/{n_total} ({pct_sig:.0f}%)\n\n")
        f.write(f"{message}\n\n")
        f.write(results_df.to_string())
    print(f"  Summary saved to : {txt_path}")

    # ── Plot deviation distributions ──────────────────────────────────────────
    plot_confound_distributions(clac_dev, oxford_dev, results_df, OUT_DIR)

    return results_df, verdict


def plot_confound_distributions(clac_dev, oxford_dev, results_df, out_dir):
    """
    Plot deviation score distributions for CLAC vs Oxford HC
    for each feature — visual confirmation of confound gate.
    """
    sig_features = results_df[results_df["significant"]]["feature"].tolist()
    all_features = results_df["feature"].tolist()

    n_cols = 4
    n_rows = int(np.ceil(len(all_features) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(all_features):
        ax      = axes[i]
        dev_col = f"dev_{feat}"
        clac_v  = clac_dev[dev_col].dropna()
        ox_v    = oxford_dev[dev_col].dropna()

        ax.hist(clac_v,  bins=20, alpha=0.6, color="#4C72B0", label="CLAC HC")
        ax.hist(ox_v,    bins=10, alpha=0.6, color="#DD8452", label="Oxford HC")
        ax.axvline(0,    color="black", linestyle="--", alpha=0.5, linewidth=0.8)

        is_sig = feat in sig_features
        ax.set_title(f"{feat}\n{'*** SIGNIFICANT' if is_sig else 'ok'}",
                     fontsize=7, color="red" if is_sig else "black")
        ax.set_xlabel("Deviation (z-score)", fontsize=6)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for j in range(len(all_features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confound Gate: CLAC Healthy vs Oxford PD Healthy Controls\n"
                 "Deviation scores should overlap if no confound",
                 fontsize=10, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "confound_gate_distributions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Distribution plot : {out_path}")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading data...")
    clac_df   = pd.read_csv(CLAC_PATH)
    oxford_df = pd.read_csv(OXFORD_PATH)
    ri_df     = pd.read_csv(RI_PATH)

    print(f"  CLAC  : {len(clac_df)} rows")
    print(f"  Oxford: {len(oxford_df)} rows "
          f"(PD={( oxford_df['condition']=='PD').sum()}, "
          f"HC={(oxford_df['condition']=='HC').sum()})")
    print(f"  RI    : {len(ri_df)} reference intervals")

    results, verdict = run_confound_gate(clac_df, oxford_df, ri_df)

    print(f"\nFinal verdict: {verdict}")
    print("Next step:")
    if verdict == "GREEN":
        print("  Proceed to Module 3b — deviation scoring for all PD datasets")
    elif verdict == "AMBER":
        print("  Proceed with caution — flag affected features in results")
    else:
        print("  Switch to in-corpus control referencing before proceeding")