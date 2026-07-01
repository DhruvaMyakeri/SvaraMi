"""
Module 4 v2 — Cross-Condition Characterisation (Updated)
Now includes MDVR-KCL as third PD dataset (adds rhythm features).
PD pool: Oxford PD + Telemonitoring + MDVR-KCL (ReadText + Spontaneous)
AD pool: Pitt Corpus (teammate pipeline)

New shared intersection now includes rhythm features available in MDVR-KCL:
  mean_pause_duration, silence_to_speech_ratio, articulation_rate,
  avg_syllable_duration, mean_speech_duration, silence_rate, mean_silence_count

Run: python src/characterisation_v2.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

BASE    = r"D:\PROJECTS\Research\Speech-Disease-Observation"
OUT_DIR = os.path.join(BASE, "outputs", "characterisation")
FIG_DIR = os.path.join(BASE, "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── AD column mapping ─────────────────────────────────────────────────────────
AD_COL_MAP = {
    "z_f0_mean":            "dev_meanF0",
    "z_f0_std":             "dev_stdevF0",
    "z_hnr":                "dev_HNR",
    "z_jitter_local":       "dev_localJitter",
    "z_jitter_absolute":    "dev_localabsoluteJitter",
    "z_jitter_rap":         "dev_rapJitter",
    "z_jitter_ppq5":        "dev_ppq5Jitter",
    "z_shimmer_local":      "dev_localShimmer",
    "z_shimmer_apq3":       "dev_apq3Shimmer",
    "z_shimmer_apq5":       "dev_aqpq5Shimmer",
    "z_shimmer_apq11":      "dev_apq11Shimmer",
    "z_f1_mean":            "dev_F1_mean",
    "z_f2_mean":            "dev_F2_mean",
    "z_f3_mean":            "dev_F3_mean",
    "z_f4_mean":            "dev_F4_mean",
    "z_speech_rate":        "dev_speech_rate",
    "z_articulation_rate":  "dev_articulation_rate",
    "z_n_pauses":           "dev_n_pauses",
    "z_mean_pause_duration":"dev_mean_pause_duration",
    "z_max_pause_duration": "dev_max_pause_duration",
    "z_total_pause_time":   "dev_total_pause_time",
    "z_silence_ratio":      "dev_silence_ratio",
    "z_phonation_time_ratio":"dev_phonation_time_ratio",
    "z_speaking_duration":  "dev_speaking_duration",
    "z_recording_duration": "dev_recording_duration",
}

# ── Feature sets ──────────────────────────────────────────────────────────────
VOICE_QUALITY = [
    "dev_meanF0", "dev_HNR",
    "dev_localJitter", "dev_localabsoluteJitter", "dev_rapJitter", "dev_ppq5Jitter",
    "dev_localShimmer", "dev_localdbShimmer", "dev_apq3Shimmer",
    "dev_aqpq5Shimmer", "dev_apq11Shimmer",
]

RHYTHM_PD_AVAILABLE = [
    "dev_mean_pause_duration", "dev_total_pause_time",
    "dev_silence_to_speech_ratio", "dev_speech_rate",
    "dev_articulation_rate", "dev_avg_syllable_duration",
    "dev_mean_speech_duration", "dev_silence_rate", "dev_mean_silence_count",
]

AD_ONLY = [
    "dev_stdevF0", "dev_F1_mean", "dev_F2_mean", "dev_F3_mean", "dev_F4_mean",
    "dev_n_pauses", "dev_total_pause_time", "dev_phonation_time_ratio",
]

# AD rhythm columns (may differ in naming from PD)
AD_RHYTHM_MAP = {
    "dev_mean_pause_duration":    "z_mean_pause_duration",
    "dev_max_pause_duration":     "z_max_pause_duration",
    "dev_total_pause_time":       "z_total_pause_time",
    "dev_silence_rate":           "z_silence_ratio",
    "dev_speech_rate":            "z_speech_rate",
    "dev_articulation_rate":      "z_articulation_rate",
    "dev_n_pauses":               "z_n_pauses",
    "dev_phonation_time_ratio":   "z_phonation_time_ratio",
    "dev_speaking_duration":      "z_speaking_duration",
    "dev_recording_duration":     "z_recording_duration",
}

FEAT_LABELS = {
    "dev_meanF0":                "F0 mean",
    "dev_HNR":                   "HNR",
    "dev_localJitter":           "Jitter (local)",
    "dev_localabsoluteJitter":   "Jitter (abs)",
    "dev_rapJitter":             "Jitter (RAP)",
    "dev_ppq5Jitter":            "Jitter (ppq5)",
    "dev_localShimmer":          "Shimmer (local)",
    "dev_localdbShimmer":        "Shimmer (dB)",
    "dev_apq3Shimmer":           "Shimmer (apq3)",
    "dev_aqpq5Shimmer":          "Shimmer (apq5)",
    "dev_apq11Shimmer":          "Shimmer (apq11)",
    "dev_mean_pause_duration":   "Mean Pause Dur.",
    "dev_total_pause_time":      "Total Pause Time",
    "dev_silence_to_speech_ratio":"Silence/Speech",
    "dev_speech_rate":           "Speech Rate",
    "dev_articulation_rate":     "Artic. Rate",
    "dev_avg_syllable_duration": "Avg Syl. Dur.",
    "dev_mean_speech_duration":  "Mean Speech Dur.",
    "dev_silence_rate":          "Silence Rate",
    "dev_mean_silence_count":    "Silence Count",
    "dev_stdevF0":               "F0 std",
    "dev_F1_mean":               "F1",
    "dev_F2_mean":               "F2",
    "dev_F3_mean":               "F3",
    "dev_F4_mean":               "F4",
}


# ── LOAD PD DATA ──────────────────────────────────────────────────────────────

def load_pd():
    DEV = os.path.join(BASE, "outputs", "deviation_scores")
    dfs = []

    # Oxford PD
    p = os.path.join(DEV, "oxford_pd_deviation.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df = df[df["condition"] == "PD"]
        dfs.append(df)
        print(f"  Oxford PD patients    : {len(df)}")

    # Telemonitoring
    p = os.path.join(DEV, "telemonitoring_pd_deviation.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        dfs.append(df)
        print(f"  Telemonitoring PD     : {len(df)}")

    # MDVR-KCL (has rhythm features)
    p = os.path.join(DEV, "mdvr_kcl_deviation.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df = df[df["condition"] == "PD"]
        dfs.append(df)
        print(f"  MDVR-KCL PD          : {len(df)}")

    pd_combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total PD subjects     : {len(pd_combined)}")
    return pd_combined


def load_pd_hc():
    DEV = os.path.join(BASE, "outputs", "deviation_scores")
    dfs = []
    p = os.path.join(DEV, "oxford_pd_deviation.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        hc = df[df["condition"] == "HC"]
        dfs.append(hc)
        print(f"  Oxford HC controls    : {len(hc)}")

    # MDVR-KCL HC
    p = os.path.join(DEV, "mdvr_kcl_deviation.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        hc = df[df["condition"] == "HC"]
        dfs.append(hc)
        print(f"  MDVR-KCL HC controls : {len(hc)}")

    hc_combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total PD HC           : {len(hc_combined)}")
    return hc_combined


def load_ad():
    ad_path = os.path.join(BASE, "alzheimers_pipeline", "outputs", "deviation_scores.csv")
    df = pd.read_csv(ad_path)
    df = df.rename(columns=AD_COL_MAP)
    df = df.rename(columns={"group": "condition"})
    dev_cols = [c for c in df.columns if c.startswith("dev_")]
    df_agg = df.groupby(["speaker_id", "condition"])[dev_cols].mean().reset_index()
    ad = df_agg[df_agg["condition"] == "AD"]
    hc = df_agg[df_agg["condition"] == "HC"]
    print(f"  AD unique speakers    : {len(ad)}")
    print(f"  HC unique speakers    : {len(hc)}")
    return ad, hc


# ── EFFECT SIZE ───────────────────────────────────────────────────────────────

def cohens_d(g1, g2):
    g1, g2 = g1.dropna(), g2.dropna()
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.var(ddof=1) + (n2-1)*g2.var(ddof=1)) / (n1+n2-2))
    return float((g1.mean() - g2.mean()) / pooled) if pooled > 0 else np.nan


def bootstrap_d(g1, g2, n_boot=500, ci=95):
    g1, g2 = g1.dropna().values, g2.dropna().values
    if len(g1) < 2 or len(g2) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    ds = []
    for _ in range(n_boot):
        s1 = rng.choice(g1, len(g1), replace=True)
        s2 = rng.choice(g2, len(g2), replace=True)
        n1, n2 = len(s1), len(s2)
        p = np.sqrt(((n1-1)*s1.var(ddof=1)+(n2-1)*s2.var(ddof=1))/(n1+n2-2))
        if p > 0:
            ds.append((s1.mean()-s2.mean())/p)
    a = (100-ci)/2
    return (np.percentile(ds,a), np.percentile(ds,100-a)) if ds else (np.nan, np.nan)


def characterise(pd_df, pd_hc_df, ad_df, ad_hc_df, features, n_boot=500):
    records = []
    for feat in features:
        row = {"feature": feat, "label": FEAT_LABELS.get(feat, feat)}

        # PD
        if feat in pd_df.columns and feat in pd_hc_df.columns:
            pv = pd_df[feat].dropna()
            hv = pd_hc_df[feat].dropna()
            if len(pv) >= 2 and len(hv) >= 2:
                d = cohens_d(pv, hv)
                ci_lo, ci_hi = bootstrap_d(pv, hv, n_boot)
                _, p = stats.mannwhitneyu(pv, hv, alternative="two-sided")
                row.update({"d_PD": d, "d_PD_ci_lo": ci_lo, "d_PD_ci_hi": ci_hi,
                             "p_PD": p, "sig_PD": p < 0.05,
                             "n_PD": len(pv), "n_HC_PD": len(hv)})
            else:
                row.update({"d_PD": np.nan, "p_PD": np.nan, "sig_PD": False})
        else:
            row.update({"d_PD": np.nan, "p_PD": np.nan, "sig_PD": False})

        # AD
        if feat in ad_df.columns and feat in ad_hc_df.columns:
            av = ad_df[feat].dropna()
            hv = ad_hc_df[feat].dropna()
            if len(av) >= 2 and len(hv) >= 2:
                d = cohens_d(av, hv)
                ci_lo, ci_hi = bootstrap_d(av, hv, n_boot)
                _, p = stats.mannwhitneyu(av, hv, alternative="two-sided")
                row.update({"d_AD": d, "d_AD_ci_lo": ci_lo, "d_AD_ci_hi": ci_hi,
                             "p_AD": p, "sig_AD": p < 0.05,
                             "n_AD": len(av), "n_HC_AD": len(hv)})
            else:
                row.update({"d_AD": np.nan, "p_AD": np.nan, "sig_AD": False})
        else:
            row.update({"d_AD": np.nan, "p_AD": np.nan, "sig_AD": False})

        records.append(row)

    df = pd.DataFrame(records)

    def bucket(r):
        sp = r.get("sig_PD", False)
        sa = r.get("sig_AD", False)
        dp = r.get("d_PD",   np.nan)
        da = r.get("d_AD",   np.nan)
        if not sp and not sa:          return "null"
        if sp and not sa:              return "PD-specific"
        if sa and not sp:              return "AD-specific"
        if sp and sa:
            if not (np.isnan(dp) or np.isnan(da)):
                return "shared-concordant" if (dp>0)==(da>0) else "shared-discordant"
        return "unclear"

    df["bucket"] = df.apply(bucket, axis=1)
    return df


# ── HEATMAP ───────────────────────────────────────────────────────────────────

def plot_heatmap(res_df, features, title_suffix, filename):
    labels  = [FEAT_LABELS.get(f, f) for f in features]
    n       = len(features)
    d_pd    = res_df.set_index("feature")["d_PD"].reindex(features).values
    d_ad    = res_df.set_index("feature")["d_AD"].reindex(features).values
    sig_pd  = res_df.set_index("feature")["sig_PD"].reindex(features).values
    sig_ad  = res_df.set_index("feature")["sig_AD"].reindex(features).values
    matrix  = np.array([d_pd, d_ad]).T
    sig_mat = np.array([sig_pd, sig_ad]).T

    fig_h = max(8, n * 0.62)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    vmax = max(np.nanmax(np.abs(matrix[~np.isnan(matrix)])) if not np.all(np.isnan(matrix)) else 1, 0.3)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    ax.imshow(np.ones_like(matrix), cmap="Greys", vmin=0, vmax=10, aspect="auto", zorder=0)
    masked = np.where(sig_mat, matrix, np.nan)
    im = ax.imshow(masked, cmap="RdBu_r", norm=norm, aspect="auto", zorder=1)

    for i in range(n):
        for j in range(2):
            if not sig_mat[i, j] or np.isnan(matrix[i, j]):
                ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,
                    fill=True,facecolor="#E8E8E8",edgecolor="white",linewidth=1.5,zorder=2))
            ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,
                fill=False,edgecolor="white",linewidth=2,zorder=3))
            val = matrix[i, j]
            is_sig = sig_mat[i, j] and not np.isnan(val)
            if np.isnan(val):
                ax.text(j,i,"n/a",ha="center",va="center",fontsize=8,color="#999999",zorder=4)
            else:
                color = "white" if (is_sig and abs(val)>vmax*0.55) else ("#1a1a1a" if is_sig else "#888888")
                ax.text(j,i-0.10,f"{val:+.2f}",ha="center",va="center",
                        fontsize=10,color=color,fontweight="bold" if is_sig else "normal",zorder=4)
                if is_sig:
                    ax.text(j,i+0.22,"p<0.05",ha="center",va="center",
                            fontsize=6.5,color=color,style="italic",zorder=4)

    ax.set_xticks([0,1])
    ax.set_xticklabels(["Parkinson's\nDisease","Alzheimer's\nDisease"],fontsize=12,fontweight="bold")
    ax.xaxis.set_ticks_position("none")
    ax.tick_params(axis="x",pad=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels,fontsize=10)
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(n-0.5,-0.5)
    ax.xaxis.set_label_position("top")
    ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im,ax=ax,shrink=0.5,pad=0.08,aspect=15)
    cbar.set_label("Cohen's d\n(+ = above healthy, \u2212 = below)",fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"SvaraMi \u2014 Speech Deviation Heatmap\n{title_suffix}",
                 fontsize=14,fontweight="bold",y=1.01)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#B2182B",label="Above healthy (+d)"),
        Patch(facecolor="#2166AC",label="Below healthy (\u2212d)"),
        Patch(facecolor="#E8E8E8",label="Non-significant (p>0.05)"),
    ],loc="lower center",ncol=3,bbox_to_anchor=(0.5,-0.04),fontsize=9,frameon=False)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out,dpi=200,bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── OVERLAID RADAR ────────────────────────────────────────────────────────────

def plot_radar(pd_df, ad_df, features, filename):
    labels = [FEAT_LABELS.get(f,f) for f in features]
    n = len(features)
    angles = np.linspace(0,2*np.pi,n,endpoint=False).tolist()
    angles += angles[:1]

    pd_m = [pd_df[f].mean() if f in pd_df.columns else 0 for f in features]
    ad_m = [ad_df[f].mean() if f in ad_df.columns else 0 for f in features]
    pd_m = [v if not np.isnan(v) else 0 for v in pd_m] + [pd_m[0] if not np.isnan(pd_m[0]) else 0]
    ad_m = [v if not np.isnan(v) else 0 for v in ad_m] + [ad_m[0] if not np.isnan(ad_m[0]) else 0]

    fig, ax = plt.subplots(figsize=(9,9),subplot_kw=dict(polar=True))
    ref = [1.96]*n+[1.96]
    ax.fill(angles,ref,alpha=0.12,color="green")
    ax.plot(angles,ref,color="green",linewidth=1.5,linestyle="--",label="95% healthy range")
    ax.plot(angles,[-1.96]*n+[-1.96],color="green",linewidth=1.5,linestyle="--")
    ax.plot(angles,pd_m,color="#DD4444",linewidth=2.5,label=f"PD mean (n={len(pd_df)})")
    ax.fill(angles,pd_m,alpha=0.12,color="#DD4444")
    ax.plot(angles,ad_m,color="#4444DD",linewidth=2.5,label=f"AD mean (n={len(ad_df)})")
    ax.fill(angles,ad_m,alpha=0.12,color="#4444DD")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels,fontsize=8)
    ax.set_ylim(-4,5)
    ax.set_title("SvaraMi \u2014 PD vs AD Deviation Profiles",fontsize=13,fontweight="bold",pad=20)
    ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.1),fontsize=9)
    plt.tight_layout()
    out = os.path.join(FIG_DIR,filename)
    plt.savefig(out,dpi=200,bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── PRINT RESULTS ─────────────────────────────────────────────────────────────

def print_results(res_df, title):
    print(f"\n{'='*70}")
    print(f"CHARACTERISATION — {title}")
    print(f"{'='*70}")
    for bucket in ["shared-concordant","shared-discordant","PD-specific","AD-specific","null"]:
        sub = res_df[res_df["bucket"]==bucket]
        if sub.empty: continue
        print(f"\n  {bucket.upper()} ({len(sub)}):")
        for _,r in sub.iterrows():
            dp = f"{r['d_PD']:.3f}" if not np.isnan(r.get('d_PD',np.nan)) else "N/A"
            da = f"{r['d_AD']:.3f}" if not np.isnan(r.get('d_AD',np.nan)) else "N/A"
            print(f"    {r['label']:<28} PD d={dp:>8}  AD d={da:>8}")
    print(f"\n  Summary:")
    for b in ["shared-concordant","shared-discordant","PD-specific","AD-specific","null"]:
        print(f"    {b:<25}: {(res_df['bucket']==b).sum()}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("MODULE 4 v2 — UPDATED CHARACTERISATION")
    print("PD: Oxford + Telemonitoring + MDVR-KCL")
    print("AD: Pitt Corpus")
    print("="*60)

    print("\nLoading PD deviation scores...")
    pd_df    = load_pd()
    pd_hc_df = load_pd_hc()

    print("\nLoading AD deviation scores...")
    ad_df, ad_hc_df = load_ad()

    # ── Level 1: Voice quality (all PD datasets contribute) ──────────────────
    print(f"\n{'─'*60}")
    print("LEVEL 1 — Voice Quality Features")
    res_vq = characterise(pd_df, pd_hc_df, ad_df, ad_hc_df, VOICE_QUALITY)
    print_results(res_vq, "Voice Quality")
    res_vq.to_csv(os.path.join(OUT_DIR,"char_voice_quality.csv"),index=False)

    # ── Level 2: Rhythm (MDVR-KCL PD + Pitt AD) ─────────────────────────────
    print(f"\n{'─'*60}")
    print("LEVEL 2 — Rhythm Features (MDVR-KCL PD vs Pitt AD)")

    # For rhythm, only MDVR-KCL has PD rhythm data — use it as PD source
    mdvr_dev  = os.path.join(BASE,"outputs","deviation_scores","mdvr_kcl_deviation.csv")
    mdvr_df   = pd.read_csv(mdvr_dev)
    mdvr_pd   = mdvr_df[mdvr_df["condition"]=="PD"]
    mdvr_hc   = mdvr_df[mdvr_df["condition"]=="HC"]

    # AD rhythm — rename z_ columns to dev_ names for alignment
    ad_rhythm_df    = ad_df.copy()
    ad_hc_rhythm_df = ad_hc_df.copy()

    # Rename z_ rhythm columns to dev_ standard names
    for dev_name, z_name in AD_RHYTHM_MAP.items():
        if z_name in ad_rhythm_df.columns:
            ad_rhythm_df    = ad_rhythm_df.rename(columns={z_name: dev_name})
            ad_hc_rhythm_df = ad_hc_rhythm_df.rename(columns={z_name: dev_name})

    res_rh = characterise(mdvr_pd, mdvr_hc, ad_rhythm_df, ad_hc_rhythm_df, RHYTHM_PD_AVAILABLE)
    print_results(res_rh, "Rhythm Features (MDVR-KCL PD vs Pitt AD)")
    res_rh.to_csv(os.path.join(OUT_DIR,"char_rhythm.csv"),index=False)

    # ── Level 3: AD-only features ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("LEVEL 3 — AD-Only Features")
    res_ad = characterise(pd_df, pd_hc_df, ad_df, ad_hc_df, AD_ONLY)
    print_results(res_ad, "AD-Only Features")
    res_ad.to_csv(os.path.join(OUT_DIR,"char_ad_only.csv"),index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Generating figures...")

    # Heatmap 1: Voice quality only
    plot_heatmap(res_vq, VOICE_QUALITY,
                 "Voice Quality Features — PD (Oxford+Tele+MDVR) vs AD (Pitt)",
                 "heatmap_voice_quality.png")

    # Heatmap 2: Full — voice quality + rhythm combined
    all_res  = pd.concat([res_vq, res_rh], ignore_index=True)
    all_feat = VOICE_QUALITY + RHYTHM_PD_AVAILABLE
    plot_heatmap(all_res, all_feat,
                 "Voice Quality + Rhythm — PD vs AD",
                 "heatmap_full_v2.png")

    # Radar: voice quality comparison
    plot_radar(pd_df, ad_df, VOICE_QUALITY, "radar_pd_vs_ad_vq.png")

    # Radar: rhythm comparison (MDVR-KCL PD vs AD)
    plot_radar(mdvr_pd, ad_rhythm_df, RHYTHM_PD_AVAILABLE, "radar_pd_vs_ad_rhythm.png")

    print(f"\n{'='*60}")
    print("MODULE 4 v2 COMPLETE")
    print(f"{'='*60}")
    print(f"  Voice quality results : outputs/characterisation/char_voice_quality.csv")
    print(f"  Rhythm results        : outputs/characterisation/char_rhythm.csv")
    print(f"  Heatmap (VQ)          : outputs/figures/heatmap_voice_quality.png")
    print(f"  Heatmap (full)        : outputs/figures/heatmap_full_v2.png")
    print(f"  Radar (VQ)            : outputs/figures/radar_pd_vs_ad_vq.png")
    print(f"  Radar (rhythm)        : outputs/figures/radar_pd_vs_ad_rhythm.png")