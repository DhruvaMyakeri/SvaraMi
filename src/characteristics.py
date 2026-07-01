"""
Module 4 — Cross-Condition Characterisation Analysis
Produces the paper's core contribution:
- Per-feature Cohen's d effect size per condition (PD, AD)
- Feature bucket classification: shared-concordant, shared-discordant,
  condition-specific (PD), condition-specific (AD), null
- Deviation heatmap (rows=features, columns=conditions)
- Overlaid radar plot comparing PD and AD mean profiles

Input:
  outputs/deviation_scores/oxford_pd_deviation.csv
  outputs/deviation_scores/telemonitoring_pd_deviation.csv
  alzheimers_pipeline/outputs/deviation_scores.csv

Output:
  outputs/characterisation/feature_buckets.csv
  outputs/characterisation/effect_sizes.csv
  outputs/figures/deviation_heatmap.png
  outputs/figures/radar_pd_vs_ad.png
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

BASE     = r"D:\PROJECTS\Research\Speech-Disease-Observation"
OUT_DIR  = os.path.join(BASE, "outputs", "characterisation")
FIG_DIR  = os.path.join(BASE, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Column mapping: AD pipeline names → standard names ───────────────────────
AD_COL_MAP = {
    "z_f0_mean":           "dev_meanF0",
    "z_f0_std":            "dev_stdevF0",
    "z_f0_min":            "dev_minF0",
    "z_f0_max":            "dev_maxF0",
    "z_hnr":               "dev_HNR",
    "z_jitter_local":      "dev_localJitter",
    "z_jitter_absolute":   "dev_localabsoluteJitter",
    "z_jitter_rap":        "dev_rapJitter",
    "z_jitter_ppq5":       "dev_ppq5Jitter",
    "z_shimmer_local":     "dev_localShimmer",
    "z_shimmer_apq3":      "dev_apq3Shimmer",
    "z_shimmer_apq5":      "dev_aqpq5Shimmer",
    "z_shimmer_apq11":     "dev_apq11Shimmer",
    "z_f1_mean":           "dev_F1_mean",
    "z_f2_mean":           "dev_F2_mean",
    "z_f3_mean":           "dev_F3_mean",
    "z_f4_mean":           "dev_F4_mean",
    "z_speech_rate":       "dev_speech_rate",
    "z_articulation_rate": "dev_articulation_rate",
    "z_n_pauses":          "dev_n_pauses",
    "z_mean_pause_duration":"dev_mean_pause_duration",
    "z_max_pause_duration": "dev_max_pause_duration",
    "z_total_pause_time":   "dev_total_pause_time",
    "z_silence_ratio":      "dev_silence_ratio",
    "z_phonation_time_ratio":"dev_phonation_time_ratio",
    "z_speaking_duration":  "dev_speaking_duration",
    "z_recording_duration": "dev_recording_duration",
}

# Shared intersection features — available in BOTH PD and AD
SHARED_FEATURES = [
    "dev_meanF0",
    "dev_HNR",
    "dev_localJitter",
    "dev_localabsoluteJitter",
    "dev_rapJitter",
    "dev_ppq5Jitter",
    "dev_localShimmer",
    "dev_apq3Shimmer",
    "dev_aqpq5Shimmer",
    "dev_apq11Shimmer",
]

# AD-only features (not in PD pre-extracted CSVs)
AD_ONLY_FEATURES = [
    "dev_stdevF0",
    "dev_F1_mean", "dev_F2_mean", "dev_F3_mean", "dev_F4_mean",
    "dev_speech_rate", "dev_articulation_rate",
    "dev_n_pauses", "dev_mean_pause_duration",
    "dev_max_pause_duration", "dev_total_pause_time",
    "dev_silence_ratio", "dev_phonation_time_ratio",
]

# Friendly labels for plots
FEAT_LABELS = {
    "dev_meanF0":              "F0 mean",
    "dev_HNR":                 "HNR",
    "dev_localJitter":         "Jitter (local)",
    "dev_localabsoluteJitter": "Jitter (abs)",
    "dev_rapJitter":           "Jitter (RAP)",
    "dev_ppq5Jitter":          "Jitter (ppq5)",
    "dev_localShimmer":        "Shimmer (local)",
    "dev_apq3Shimmer":         "Shimmer (apq3)",
    "dev_aqpq5Shimmer":        "Shimmer (apq5)",
    "dev_apq11Shimmer":        "Shimmer (apq11)",
    "dev_stdevF0":             "F0 std",
    "dev_F1_mean":             "F1",
    "dev_F2_mean":             "F2",
    "dev_F3_mean":             "F3",
    "dev_F4_mean":             "F4",
    "dev_speech_rate":         "Speech Rate",
    "dev_articulation_rate":   "Artic Rate",
    "dev_n_pauses":            "N Pauses",
    "dev_mean_pause_duration": "Mean Pause",
    "dev_max_pause_duration":  "Max Pause",
    "dev_total_pause_time":    "Total Pause",
    "dev_silence_ratio":       "Silence Ratio",
    "dev_phonation_time_ratio":"Phonation Ratio",
}


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_pd_deviations():
    """Load and combine Oxford PD + Telemonitoring deviation scores."""
    oxford_path = os.path.join(BASE, "outputs", "deviation_scores", "oxford_pd_deviation.csv")
    tele_path   = os.path.join(BASE, "outputs", "deviation_scores", "telemonitoring_pd_deviation.csv")

    oxford = pd.read_csv(oxford_path)
    tele   = pd.read_csv(tele_path)

    # Keep PD patients only from Oxford (HC used as reference, not subjects)
    oxford_pd = oxford[oxford["condition"] == "PD"].copy()
    oxford_hc = oxford[oxford["condition"] == "HC"].copy()
    tele_pd   = tele.copy()  # all PD

    print(f"  Oxford PD patients : {len(oxford_pd)}")
    print(f"  Oxford HC controls : {len(oxford_hc)}")
    print(f"  Telemonitoring PD  : {len(tele_pd)}")

    # Combine all PD
    pd_combined = pd.concat([oxford_pd, tele_pd], ignore_index=True)
    print(f"  Total PD subjects  : {len(pd_combined)}")

    return pd_combined, oxford_hc


def load_ad_deviations():
    """Load AD deviation scores and aggregate to one row per speaker."""
    ad_path = os.path.join(BASE, "alzheimers_pipeline", "outputs", "deviation_scores.csv")
    df      = pd.read_csv(ad_path)

    # Rename columns to standard names
    df = df.rename(columns=AD_COL_MAP)
    df = df.rename(columns={"group": "condition"})

    # Aggregate multiple recordings per speaker to one row per speaker
    dev_cols = [c for c in df.columns if c.startswith("dev_")]
    df_agg   = df.groupby(["speaker_id", "condition"])[dev_cols].mean().reset_index()

    ad_df = df_agg[df_agg["condition"] == "AD"].copy()
    hc_df = df_agg[df_agg["condition"] == "HC"].copy()

    print(f"  AD unique speakers : {len(ad_df)}")
    print(f"  HC unique speakers : {len(hc_df)}")

    return ad_df, hc_df


# ── EFFECT SIZE ───────────────────────────────────────────────────────────────

def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    group1 = group1.dropna()
    group2 = group2.dropna()
    if len(group1) < 2 or len(group2) < 2:
        return np.nan
    n1, n2   = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return np.nan
    return (group1.mean() - group2.mean()) / pooled_std


def bootstrap_cohens_d(group1, group2, n_boot=1000, ci=95):
    """Bootstrap CI on Cohen's d."""
    group1 = group1.dropna().values
    group2 = group2.dropna().values
    if len(group1) < 2 or len(group2) < 2:
        return np.nan, np.nan
    rng       = np.random.default_rng(seed=42)
    boot_ds   = []
    for _ in range(n_boot):
        s1 = rng.choice(group1, size=len(group1), replace=True)
        s2 = rng.choice(group2, size=len(group2), replace=True)
        n1, n2 = len(s1), len(s2)
        pooled  = np.sqrt(((n1-1)*s1.var(ddof=1) + (n2-1)*s2.var(ddof=1))/(n1+n2-2))
        if pooled > 0:
            boot_ds.append((s1.mean() - s2.mean()) / pooled)
    alpha = (100 - ci) / 2
    return np.percentile(boot_ds, alpha), np.percentile(boot_ds, 100-alpha)


# ── CHARACTERISATION ──────────────────────────────────────────────────────────

def characterise_features(pd_df, pd_hc_df, ad_df, ad_hc_df, features, n_boot=500):
    """
    For each feature compute:
    - Cohen's d for PD vs HC
    - Cohen's d for AD vs HC
    - Mann-Whitney p-value for each
    - Bootstrap CIs on Cohen's d
    - Feature bucket classification
    """
    records = []
    print(f"\nComputing effect sizes for {len(features)} features...")

    for feat in features:
        if feat not in pd_df.columns and feat not in ad_df.columns:
            continue

        row = {"feature": feat, "label": FEAT_LABELS.get(feat, feat)}

        # PD effect
        if feat in pd_df.columns and feat in pd_hc_df.columns:
            pd_vals  = pd_df[feat].dropna()
            hc_pd    = pd_hc_df[feat].dropna()
            d_pd     = cohens_d(pd_vals, hc_pd)
            ci_lo_pd, ci_hi_pd = bootstrap_cohens_d(pd_vals, hc_pd, n_boot=n_boot)
            _, p_pd  = stats.mannwhitneyu(pd_vals, hc_pd, alternative="two-sided")
            row.update({
                "d_PD": d_pd, "d_PD_ci_lo": ci_lo_pd, "d_PD_ci_hi": ci_hi_pd,
                "p_PD": p_pd, "sig_PD": p_pd < 0.05,
                "n_PD": len(pd_vals), "n_HC_PD": len(hc_pd),
                "mean_PD": pd_vals.mean(), "mean_HC_PD": hc_pd.mean()
            })
        else:
            row.update({"d_PD": np.nan, "p_PD": np.nan, "sig_PD": False})

        # AD effect
        if feat in ad_df.columns and feat in ad_hc_df.columns:
            ad_vals  = ad_df[feat].dropna()
            hc_ad    = ad_hc_df[feat].dropna()
            d_ad     = cohens_d(ad_vals, hc_ad)
            ci_lo_ad, ci_hi_ad = bootstrap_cohens_d(ad_vals, hc_ad, n_boot=n_boot)
            _, p_ad  = stats.mannwhitneyu(ad_vals, hc_ad, alternative="two-sided")
            row.update({
                "d_AD": d_ad, "d_AD_ci_lo": ci_lo_ad, "d_AD_ci_hi": ci_hi_ad,
                "p_AD": p_ad, "sig_AD": p_ad < 0.05,
                "n_AD": len(ad_vals), "n_HC_AD": len(hc_ad),
                "mean_AD": ad_vals.mean(), "mean_HC_AD": hc_ad.mean()
            })
        else:
            row.update({"d_AD": np.nan, "p_AD": np.nan, "sig_AD": False})

        records.append(row)

    df = pd.DataFrame(records)

    # ── Feature bucket classification ─────────────────────────────────────────
    def classify(row):
        sig_pd = row.get("sig_PD", False)
        sig_ad = row.get("sig_AD", False)
        d_pd   = row.get("d_PD",   np.nan)
        d_ad   = row.get("d_AD",   np.nan)

        if not sig_pd and not sig_ad:
            return "null"
        if sig_pd and not sig_ad:
            return "PD-specific"
        if sig_ad and not sig_pd:
            return "AD-specific"
        if sig_pd and sig_ad:
            # Both significant — check direction
            if not np.isnan(d_pd) and not np.isnan(d_ad):
                same_dir = (d_pd > 0 and d_ad > 0) or (d_pd < 0 and d_ad < 0)
                return "shared-concordant" if same_dir else "shared-discordant"
        return "unclear"

    df["bucket"] = df.apply(classify, axis=1)
    return df


# ── HEATMAP ───────────────────────────────────────────────────────────────────

def plot_heatmap(results_df, features, out_path):
    """
    Deviation heatmap — rows=features, columns=conditions
    Cells = signed Cohen's d (effect size)
    Diverging colormap: red=above healthy, blue=below healthy, grey=non-significant
    """
    # Build matrix
    labels   = [FEAT_LABELS.get(f, f) for f in features]
    d_pd     = results_df.set_index("feature")["d_PD"].reindex(features).values
    d_ad     = results_df.set_index("feature")["d_AD"].reindex(features).values
    sig_pd   = results_df.set_index("feature")["sig_PD"].reindex(features).values
    sig_ad   = results_df.set_index("feature")["sig_AD"].reindex(features).values

    matrix   = np.array([d_pd, d_ad]).T  # shape: (n_features, 2)
    sig_mat  = np.array([sig_pd, sig_ad]).T

    n_feat   = len(features)
    fig_h    = max(6, n_feat * 0.6)
    fig, ax  = plt.subplots(figsize=(7.5, fig_h))
    vmax     = np.nanmax(np.abs(matrix[~np.isnan(matrix)])) if not np.all(np.isnan(matrix)) else 1.0
    vmax     = max(vmax, 0.3)
    norm     = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Grey background for non-significant cells FIRST (so it sits under everything)
    ax.imshow(np.ones_like(matrix), cmap="Greys", vmin=0, vmax=10, aspect="auto", zorder=0)

    # Masked significant-only matrix drawn on top
    masked   = np.where(sig_mat, matrix, np.nan)
    im = ax.imshow(masked, cmap="RdBu_r", norm=norm, aspect="auto", zorder=1)

    # Draw grey squares explicitly for non-sig cells (crisp, no overlap issues)
    for i in range(n_feat):
        for j in range(2):
            if not sig_mat[i, j] or np.isnan(matrix[i, j]):
                ax.add_patch(plt.Rectangle(
                    (j-0.5, i-0.5), 1, 1,
                    fill=True, facecolor="#E8E8E8", edgecolor="white",
                    linewidth=1.5, zorder=2
                ))

    # Cell borders for all cells (clean grid look)
    for i in range(n_feat):
        for j in range(2):
            ax.add_patch(plt.Rectangle(
                (j-0.5, i-0.5), 1, 1,
                fill=False, edgecolor="white", linewidth=2, zorder=3
            ))

    # Annotate cells — value only, asterisk as separate small marker
    for i in range(n_feat):
        for j in range(2):
            val = matrix[i, j]
            is_sig = sig_mat[i, j] and not np.isnan(val)
            if np.isnan(val):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=9, color="#999999", zorder=4)
                continue
            color = "white" if (is_sig and abs(val) > vmax*0.55) else \
                    ("#1a1a1a" if is_sig else "#888888")
            txt = f"{val:+.2f}"
            ax.text(j, i - 0.10, txt, ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold" if is_sig else "normal",
                    zorder=4)
            if is_sig:
                ax.text(j, i + 0.22, "significant (p<0.05)", ha="center", va="center",
                        fontsize=6.5, color=color, style="italic", zorder=4)

    # Column headers — short, non-overlapping, placed above the plot area
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Parkinson's\nDisease", "Alzheimer's\nDisease"],
                       fontsize=12, fontweight="bold")
    ax.xaxis.set_ticks_position("none")
    ax.tick_params(axis="x", pad=10)

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(n_feat - 0.5, -0.5)

    # Move x labels to top
    ax.xaxis.set_label_position("top")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.08, aspect=15)
    cbar.set_label("Cohen's d\n(+ = above healthy, \u2212 = below healthy)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("SvaraMi \u2014 Speech Deviation Heatmap", fontsize=15,
                 fontweight="bold", y=1.01)
    ax.set_title("Parkinson's Disease vs Alzheimer's Disease (Cohen's d effect size)",
                 fontsize=10, color="#444444", pad=14)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#B2182B", label="Above healthy (+d)"),
        Patch(facecolor="#2166AC", label="Below healthy (\u2212d)"),
        Patch(facecolor="#E8E8E8", label="Non-significant (p>0.05)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.04), fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved: {out_path}")


# ── OVERLAID RADAR ────────────────────────────────────────────────────────────

def plot_overlaid_radar(pd_df, ad_df, features, out_path):
    """
    Overlaid radar plot — PD and AD mean deviation profiles on same axes.
    Shows directly which features are shared vs condition-specific.
    """
    labels  = [FEAT_LABELS.get(f, f) for f in features]
    n       = len(features)
    angles  = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    pd_means = [pd_df[f].mean() if f in pd_df.columns else 0 for f in features]
    ad_means = [ad_df[f].mean() if f in ad_df.columns else 0 for f in features]
    pd_means += pd_means[:1]
    ad_means += ad_means[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    # Reference band ±1.96
    ref = [1.96]*n + [1.96]
    ax.fill(angles, ref,  alpha=0.12, color="green")
    ax.plot(angles, ref,  color="green", linewidth=1.5, linestyle="--",
            label="95% healthy range")
    ref_neg = [-1.96]*n + [-1.96]
    ax.plot(angles, ref_neg, color="green", linewidth=1.5, linestyle="--")

    # PD line
    ax.plot(angles, pd_means, color="#DD4444", linewidth=2.5,
            label=f"PD mean (n={len(pd_df)})")
    ax.fill(angles, pd_means, alpha=0.12, color="#DD4444")

    # AD line
    ax.plot(angles, ad_means, color="#4444DD", linewidth=2.5,
            label=f"AD mean (n={len(ad_df)})")
    ax.fill(angles, ad_means, alpha=0.12, color="#4444DD")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(-4, 5)
    ax.set_yticks([-2, 0, 2, 4])
    ax.set_yticklabels(["-2σ", "0", "+2σ", "+4σ"], fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_title("SvaraMi — PD vs AD Deviation Profiles\n(Shared Intersection Features)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Overlaid radar saved: {out_path}")


# ── PRINT RESULTS ─────────────────────────────────────────────────────────────

def print_results(results_df):
    buckets = ["shared-concordant", "shared-discordant",
               "PD-specific", "AD-specific", "null"]

    print(f"\n{'='*70}")
    print("CHARACTERISATION RESULTS — Feature Bucket Classification")
    print(f"{'='*70}")

    for bucket in buckets:
        subset = results_df[results_df["bucket"] == bucket]
        if subset.empty:
            continue
        print(f"\n  {bucket.upper()} ({len(subset)} features):")
        for _, row in subset.iterrows():
            d_pd = f"{row['d_PD']:.3f}" if not np.isnan(row.get('d_PD', np.nan)) else "N/A"
            d_ad = f"{row['d_AD']:.3f}" if not np.isnan(row.get('d_AD', np.nan)) else "N/A"
            print(f"    {row['label']:<25}  PD d={d_pd:>8}  AD d={d_ad:>8}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for bucket in buckets:
        n = (results_df["bucket"] == bucket).sum()
        print(f"  {bucket:<25}: {n} features")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("="*60)
    print("MODULE 4 — CROSS-CONDITION CHARACTERISATION")
    print("="*60)

    # Load data
    print("\nLoading PD deviation scores...")
    pd_df, pd_hc_df = load_pd_deviations()

    print("\nLoading AD deviation scores...")
    ad_df, ad_hc_df = load_ad_deviations()

    # ── Level 1: Shared intersection features (PD + AD)
    print(f"\n{'─'*60}")
    print("LEVEL 1 — Shared Intersection Features (PD + AD)")
    print(f"{'─'*60}")
    results_shared = characterise_features(
        pd_df, pd_hc_df, ad_df, ad_hc_df,
        SHARED_FEATURES, n_boot=500
    )
    print_results(results_shared)

    shared_out = os.path.join(OUT_DIR, "characterisation_shared_features.csv")
    results_shared.to_csv(shared_out, index=False)
    print(f"\n  Saved: {shared_out}")

    # ── Diagnostic: full stats with CIs and p-values for every shared feature ──
    print(f"\n{'='*90}")
    print("DIAGNOSTIC — Full Effect Size Table with Bootstrap CIs and p-values")
    print(f"{'='*90}")
    print(f"  {'Feature':<20} {'d_PD':>8} {'CI_PD':>18} {'p_PD':>8} {'sig':>5}  |  "
          f"{'d_AD':>8} {'CI_AD':>18} {'p_AD':>8} {'sig':>5}")
    print(f"  {'-'*100}")
    for _, row in results_shared.iterrows():
        d_pd    = row.get('d_PD', np.nan)
        d_ad    = row.get('d_AD', np.nan)
        p_pd    = row.get('p_PD', np.nan)
        p_ad    = row.get('p_AD', np.nan)
        ci_pd   = f"[{row.get('d_PD_ci_lo', np.nan):.2f}, {row.get('d_PD_ci_hi', np.nan):.2f}]" if not np.isnan(row.get('d_PD_ci_lo', np.nan)) else "N/A"
        ci_ad   = f"[{row.get('d_AD_ci_lo', np.nan):.2f}, {row.get('d_AD_ci_hi', np.nan):.2f}]" if not np.isnan(row.get('d_AD_ci_lo', np.nan)) else "N/A"
        sig_pd  = "YES" if row.get('sig_PD', False) else "no"
        sig_ad  = "YES" if row.get('sig_AD', False) else "no"
        print(f"  {row['label']:<20} {d_pd:>8.3f} {ci_pd:>18} {p_pd:>8.4f} {sig_pd:>5}  |  "
              f"{d_ad:>8.3f} {ci_ad:>18} {p_ad:>8.4f} {sig_ad:>5}")

    # Specifically flag F0 mean for inspection
    f0_row = results_shared[results_shared["feature"] == "dev_meanF0"]
    if not f0_row.empty:
        r = f0_row.iloc[0]
        print(f"\n  >>> F0 MEAN DIAGNOSTIC <<<")
        print(f"      PD: d={r.get('d_PD'):.4f}  p={r.get('p_PD'):.6f}  "
              f"n_PD={r.get('n_PD')}  n_HC={r.get('n_HC_PD')}  "
              f"significant={r.get('sig_PD')}")
        print(f"      AD: d={r.get('d_AD'):.4f}  p={r.get('p_AD'):.6f}  "
              f"n_AD={r.get('n_AD')}  n_HC={r.get('n_HC_AD')}  "
              f"significant={r.get('sig_AD')}")
        if r.get('p_PD') > 0.05 and r.get('p_PD') < 0.10:
            print(f"      NOTE: PD p-value is between 0.05-0.10 — borderline non-significant,")
            print(f"            likely due to small Oxford HC sample (n=8). Interpret with caution.")

    # ── Level 2: AD-only features
    print(f"\n{'─'*60}")
    print("LEVEL 2 — AD-Only Features (rhythm, formants)")
    print(f"{'─'*60}")
    ad_only_avail = [f for f in AD_ONLY_FEATURES if f in ad_df.columns]
    results_ad_only = characterise_features(
        pd_df, pd_hc_df, ad_df, ad_hc_df,
        ad_only_avail, n_boot=500
    )

    print(f"\n  AD-Only Feature Deviations:")
    for _, row in results_ad_only.iterrows():
        d_ad = f"{row['d_AD']:.3f}" if not np.isnan(row.get('d_AD', np.nan)) else "N/A"
        sig  = "* sig" if row.get("sig_AD", False) else "  ns"
        print(f"    {row['label']:<25}  AD d={d_ad:>8}  {sig}")

    ad_only_out = os.path.join(OUT_DIR, "characterisation_ad_only_features.csv")
    results_ad_only.to_csv(ad_only_out, index=False)
    print(f"\n  Saved: {ad_only_out}")

    # ── Figures
    print(f"\n{'─'*60}")
    print("Generating figures...")
    print(f"{'─'*60}")

    # Heatmap — shared features only
    plot_heatmap(
        results_shared,
        SHARED_FEATURES,
        os.path.join(FIG_DIR, "deviation_heatmap_shared.png")
    )

    # Heatmap — all AD features
    all_feats = SHARED_FEATURES + [f for f in ad_only_avail if f not in SHARED_FEATURES]
    plot_heatmap(
        pd.concat([results_shared, results_ad_only], ignore_index=True),
        all_feats,
        os.path.join(FIG_DIR, "deviation_heatmap_full.png")
    )

    # Overlaid radar — shared features
    plot_overlaid_radar(
        pd_df, ad_df,
        SHARED_FEATURES,
        os.path.join(FIG_DIR, "radar_pd_vs_ad.png")
    )

    print(f"\n{'='*60}")
    print("MODULE 4 COMPLETE")
    print(f"{'='*60}")
    print(f"  Shared features results : {shared_out}")
    print(f"  AD-only results         : {ad_only_out}")
    print(f"  Heatmap (shared)        : outputs/figures/deviation_heatmap_shared.png")
    print(f"  Heatmap (full)          : outputs/figures/deviation_heatmap_full.png")
    print(f"  Overlaid radar          : outputs/figures/radar_pd_vs_ad.png")
    print(f"\nNext step: Module 5 — XGBoost + SHAP + NAMs")