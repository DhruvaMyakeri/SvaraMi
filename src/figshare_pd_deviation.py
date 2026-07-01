"""
Module 3c — Figshare AH Dataset: Confound Check + Deviation Scoring
Third independent PD dataset (University of Arkansas, telephone recordings,
sustained vowel /a/, raw audio). Used as a replication check against
Oxford PD and Telemonitoring PD.

Input:
  outputs/features/ah_acoustic_features.csv  (already extracted)
  demographics file (age, sex, HC/PwPD label) — update DEMO_PATH below

Output:
  outputs/deviation_scores/figshare_pd_deviation.csv
  outputs/figures/radar_figshare_pd.png
  Console: confound check vs CLAC, and 3-way replication comparison
  vs Oxford PD and Telemonitoring
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE      = r"D:\PROJECTS\Research\Speech-Disease-Observation"
FEAT_PATH = os.path.join(BASE, "outputs", "features", "ah_acoustic_features.csv")
DEMO_PATH = os.path.join(BASE, "data", "raw", "figshare_pd", "demographics.csv")  # UPDATE if named differently
OUT_DIR   = os.path.join(BASE, "outputs", "deviation_scores")
FIG_DIR   = os.path.join(BASE, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SHARED_FEATURES = [
    "meanF0", "HNR",
    "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter",
    "localShimmer", "localdbShimmer", "apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer",
]

FEAT_LABELS = {
    "meanF0": "F0 mean", "HNR": "HNR",
    "localJitter": "Jitter\n(local)", "localabsoluteJitter": "Jitter\n(abs)",
    "rapJitter": "Jitter\n(RAP)", "ppq5Jitter": "Jitter\n(ppq5)",
    "localShimmer": "Shimmer\n(local)", "localdbShimmer": "Shimmer\n(dB)",
    "apq3Shimmer": "Shimmer\n(apq3)", "aqpq5Shimmer": "Shimmer\n(apq5)",
    "apq11Shimmer": "Shimmer\n(apq11)",
}


# ── LOAD + MERGE DEMOGRAPHICS ─────────────────────────────────────────────────

def load_and_merge():
    df = pd.read_csv(FEAT_PATH)
    print(f"  Features loaded: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    if os.path.exists(DEMO_PATH):
        demo = pd.read_csv(DEMO_PATH)
        print(f"  Demographics loaded: {demo.shape}, columns: {demo.columns.tolist()}")
        merge_key = None
        for candidate in ["speaker_id", "Sample ID", "sample_id", "SampleID", "name"]:
            if candidate in demo.columns and candidate in df.columns:
                merge_key = candidate
                break
        if merge_key:
            df = df.merge(demo, on=merge_key, how="left")
            print(f"  Merged on: {merge_key}")
        else:
            print("  WARNING: could not auto-match merge key — check demographics columns manually")
            print(f"  Feature file columns: {df.columns.tolist()}")
            print(f"  Demographics columns: {demo.columns.tolist()}")
    else:
        print(f"  WARNING: demographics file not found at {DEMO_PATH}")
        print("  Proceeding with condition/gender info already in features CSV if present")

    return df


def standardize_condition_column(df):
    """Ensure a 'condition' column exists with values HC / PD."""
    if "condition" in df.columns:
        return df
    for candidate in ["label", "Label", "status", "group", "Group"]:
        if candidate in df.columns:
            df["condition"] = df[candidate].astype(str).str.upper().map(
                lambda x: "PD" if "PD" in x or x == "1" else "HC"
            )
            print(f"  Derived condition from column: {candidate}")
            return df
    print("  WARNING: no condition/label column found — cannot proceed without group labels")
    return df


# ── CONFOUND CHECK VS CLAC ────────────────────────────────────────────────────

def confound_check_vs_clac(df, clac_path):
    """Quick check: does this dataset's HC group separate from CLAC on raw values?"""
    if not os.path.exists(clac_path):
        print("  CLAC merged file not found — skipping confound check")
        return

    clac = pd.read_csv(clac_path)
    clac_vowel = clac[clac["task"] == "max_phonation"]

    hc = df[df["condition"] == "HC"]
    print(f"\n{'='*60}")
    print("CONFOUND CHECK — Figshare HC vs CLAC (raw values)")
    print(f"{'='*60}")
    print(f"  {'Feature':<20} {'CLAC mean':>12} {'Figshare HC mean':>18} {'p-value':>10}")

    for feat in SHARED_FEATURES:
        if feat not in clac_vowel.columns or feat not in hc.columns:
            continue
        c_vals = clac_vowel[feat].dropna()
        f_vals = hc[feat].dropna()
        if len(c_vals) < 5 or len(f_vals) < 5:
            continue
        _, p = stats.mannwhitneyu(c_vals, f_vals, alternative="two-sided")
        flag = "***" if p < 0.05 else ""
        print(f"  {feat:<20} {c_vals.mean():>12.4f} {f_vals.mean():>18.4f} {p:>10.4f} {flag}")


# ── IN-CORPUS DEVIATION SCORING ───────────────────────────────────────────────

def compute_deviation(df, features):
    hc = df[df["condition"] == "HC"]
    print(f"\n  In-corpus HC reference: n={len(hc)}")

    ref_stats = {}
    for feat in features:
        if feat not in hc.columns:
            continue
        vals = hc[feat].dropna()
        if len(vals) < 2:
            continue
        ref_stats[feat] = (vals.mean(), vals.std())

    records = []
    for _, row in df.iterrows():
        rec = {
            "speaker_id": row.get("speaker_id", row.get("name", "unknown")),
            "condition":  row.get("condition", "unknown"),
            "dataset":    "figshare_ah",
        }
        for feat in features:
            if feat not in row.index or pd.isna(row[feat]) or feat not in ref_stats:
                rec[f"dev_{feat}"] = np.nan
                continue
            mean, std = ref_stats[feat]
            rec[f"dev_{feat}"] = (row[feat] - mean) / std if std > 0 else np.nan
        records.append(rec)

    return pd.DataFrame(records)


def print_deviation_summary(dev_df, features):
    print(f"\n{'='*70}")
    print("DEVIATION SUMMARY — Figshare AH Dataset")
    print(f"{'='*70}")
    hc = dev_df[dev_df["condition"] == "HC"]
    pd_ = dev_df[dev_df["condition"] == "PD"]
    print(f"  {'Feature':<25} {'HC dev':>10} {'PD dev':>10} {'Direction':>15}")
    for feat in features:
        col = f"dev_{feat}"
        if col not in dev_df.columns:
            continue
        hc_m = hc[col].mean()
        pd_m = pd_[col].mean()
        direction = "above healthy" if pd_m > 0.3 else "below healthy" if pd_m < -0.3 else "within range"
        print(f"  {feat:<25} {hc_m:>10.4f} {pd_m:>10.4f} {direction:>15}")


def three_way_comparison(figshare_dev, features):
    """Compare Figshare deviation vs Oxford and Telemonitoring for replication check."""
    oxford_path = os.path.join(OUT_DIR, "oxford_pd_deviation.csv")
    tele_path   = os.path.join(OUT_DIR, "telemonitoring_pd_deviation.csv")

    if not (os.path.exists(oxford_path) and os.path.exists(tele_path)):
        print("\n  Oxford/Telemonitoring deviation files not found — skipping 3-way comparison")
        return

    oxford = pd.read_csv(oxford_path)
    tele   = pd.read_csv(tele_path)
    oxford_pd = oxford[oxford["condition"] == "PD"]
    fig_pd    = figshare_dev[figshare_dev["condition"] == "PD"]

    print(f"\n{'='*80}")
    print("THREE-WAY REPLICATION CHECK — Oxford vs Telemonitoring vs Figshare AH")
    print(f"{'='*80}")
    print(f"  {'Feature':<25} {'Oxford PD':>12} {'Tele PD':>12} {'Figshare PD':>14} {'Consistent?':>12}")
    print(f"  {'-'*78}")

    for feat in features:
        col = f"dev_{feat}"
        if col not in oxford_pd.columns or col not in tele.columns or col not in fig_pd.columns:
            continue
        ox_m  = oxford_pd[col].mean()
        te_m  = tele[col].mean()
        fg_m  = fig_pd[col].mean()

        signs = [np.sign(x) for x in [ox_m, te_m, fg_m] if not np.isnan(x)]
        consistent = len(set(signs)) == 1 if len(signs) == 3 else "partial"

        print(f"  {feat:<25} {ox_m:>12.3f} {te_m:>12.3f} {fg_m:>14.3f} "
              f"{'YES' if consistent is True else 'NO' if consistent is False else '?':>12}")


def plot_radar(dev_df, features, out_path):
    labels  = [FEAT_LABELS.get(f, f) for f in features]
    n       = len(features)
    angles  = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    conditions = ["HC", "PD"]
    colors     = {"HC": "#4C72B0", "PD": "#DD4444"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

    for ax, cond in zip(axes, conditions):
        sub = dev_df[dev_df["condition"] == cond]
        color = colors[cond]

        ref = [1.96]*n + [1.96]
        ax.fill(angles, ref, alpha=0.15, color="green")
        ax.plot(angles, ref, color="green", linewidth=1.5, linestyle="--")
        ref_neg = [-1.96]*n + [-1.96]
        ax.plot(angles, ref_neg, color="green", linewidth=1.5, linestyle="--")

        for _, row in sub.iterrows():
            vals = [row.get(f"dev_{f}", 0) for f in features]
            vals = [v if not np.isnan(v) else 0 for v in vals]
            vals += vals[:1]
            ax.plot(angles, vals, color=color, alpha=0.25, linewidth=0.8)

        mean_vals = [sub[f"dev_{f}"].mean() for f in features]
        mean_vals = [v if not np.isnan(v) else 0 for v in mean_vals]
        mean_vals += mean_vals[:1]
        ax.plot(angles, mean_vals, color=color, linewidth=2.5, label=f"{cond} mean (n={len(sub)})")
        ax.fill(angles, mean_vals, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(-4, 4)
        ax.set_title(f"{cond}\n(n={len(sub)})", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

    plt.suptitle("Figshare AH Dataset -- PD Deviation Profile\n(Telephone recordings, sustained vowel /a/)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Radar plot saved: {out_path}")


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading Figshare AH data...")
    df = load_and_merge()
    df = standardize_condition_column(df)

    if "condition" not in df.columns or df["condition"].isna().all():
        print("\nSTOP: condition/label column could not be determined.")
        print("Please check your demographics file and feature CSV column names,")
        print("then update the merge logic in load_and_merge() accordingly.")
    else:
        print(f"\n  Condition counts: {df['condition'].value_counts().to_dict()}")

        clac_path = os.path.join(BASE, "outputs", "features", "clac_all_merged.csv")
        confound_check_vs_clac(df, clac_path)

        dev_df = compute_deviation(df, SHARED_FEATURES)
        out_path = os.path.join(OUT_DIR, "figshare_pd_deviation.csv")
        dev_df.to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")

        print_deviation_summary(dev_df, SHARED_FEATURES)

        three_way_comparison(dev_df, SHARED_FEATURES)

        plot_radar(dev_df, SHARED_FEATURES, os.path.join(FIG_DIR, "radar_figshare_pd.png"))

        print(f"\n{'='*60}")
        print("MODULE 3c COMPLETE")
        print(f"{'='*60}")