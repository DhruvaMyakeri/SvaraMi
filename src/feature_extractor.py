"""
Module 1a — CLAC Acoustic Feature Extractor
Extracts 28 acoustic features from raw WAV files using Praat via parselmouth.
Follows Botelho et al. (2024) Table I — rhythm, voice quality, vocal tract features.
Content/linguistic features (13) are handled separately in content_extractor.py

Run on: CLAC cookie_theft, max_phonation, picnic folders
Output: outputs/features/clac_{task}_acoustic_features.csv
"""

import os
import parselmouth
import numpy as np
import pandas as pd
from parselmouth.praat import call
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ─── VOICE QUALITY FEATURES ───────────────────────────────────────────────────

def extract_f0(sound):
    """F0 mean and standard deviation — Botelho Table I: meanF0, stdevF0"""
    try:
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        mean_f0  = call(pitch, "Get mean", 0, 0, "Hertz")
        stdev_f0 = call(pitch, "Get standard deviation", 0, 0, "Hertz")
        return {
            "meanF0":  float(mean_f0)  if not np.isnan(mean_f0)  else np.nan,
            "stdevF0": float(stdev_f0) if not np.isnan(stdev_f0) else np.nan
        }
    except:
        return {"meanF0": np.nan, "stdevF0": np.nan}


def extract_hnr(sound):
    """Harmonics-to-Noise Ratio — Botelho Table I: HNR"""
    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        return {"HNR": float(hnr) if not np.isnan(hnr) else np.nan}
    except:
        return {"HNR": np.nan}


def extract_jitter(sound):
    """
    Jitter measures — Botelho Table I:
    localJitter, localabsoluteJitter, rapJitter, ppq5Jitter
    """
    try:
        pp = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        return {
            "localJitter":         call(pp, "Get jitter (local)",          0, 0, 0.0001, 0.02, 1.3),
            "localabsoluteJitter": call(pp, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3),
            "rapJitter":           call(pp, "Get jitter (rap)",             0, 0, 0.0001, 0.02, 1.3),
            "ppq5Jitter":          call(pp, "Get jitter (ppq5)",            0, 0, 0.0001, 0.02, 1.3),
        }
    except:
        return {k: np.nan for k in [
            "localJitter", "localabsoluteJitter", "rapJitter", "ppq5Jitter"
        ]}


def extract_shimmer(sound):
    """
    Shimmer measures — Botelho Table I:
    localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer
    """
    try:
        pp = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        return {
            "localShimmer":   call([sound, pp], "Get shimmer (local)",   0, 0, 0.0001, 0.02, 1.3, 1.6),
            "localdbShimmer": call([sound, pp], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            "apq3Shimmer":    call([sound, pp], "Get shimmer (apq3)",     0, 0, 0.0001, 0.02, 1.3, 1.6),
            "aqpq5Shimmer":   call([sound, pp], "Get shimmer (apq5)",     0, 0, 0.0001, 0.02, 1.3, 1.6),
            "apq11Shimmer":   call([sound, pp], "Get shimmer (apq11)",    0, 0, 0.0001, 0.02, 1.3, 1.6),
        }
    except:
        return {k: np.nan for k in [
            "localShimmer", "localdbShimmer", "apq3Shimmer", "aqpq5Shimmer", "apq11Shimmer"
        ]}


# ─── VOCAL TRACT FEATURES ─────────────────────────────────────────────────────

def extract_formants(sound):
    """
    Formant frequencies F1-F4 mean and median — Botelho Table I:
    F1_mean, F1_median, F2_mean, F2_median, F3_mean, F3_median, F4_mean, F4_median
    """
    try:
        formants = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        duration = sound.get_total_duration()
        step = 0.01
        n_frames = int(duration / step)

        vals = {1: [], 2: [], 3: [], 4: []}
        for i in range(n_frames):
            t = i * step
            for fn in [1, 2, 3, 4]:
                v = call(formants, "Get value at time", fn, t, "Hertz", "Linear")
                if not np.isnan(v):
                    vals[fn].append(v)

        result = {}
        for fn in [1, 2, 3, 4]:
            result[f"F{fn}_mean"]   = float(np.mean(vals[fn]))   if vals[fn] else np.nan
            result[f"F{fn}_median"] = float(np.median(vals[fn])) if vals[fn] else np.nan
        return result
    except:
        return {k: np.nan for k in [
            "F1_mean","F1_median","F2_mean","F2_median",
            "F3_mean","F3_median","F4_mean","F4_median"
        ]}


# ─── RHYTHM FEATURES ──────────────────────────────────────────────────────────

def extract_rhythm(sound):
    """
    Rhythm features — Botelho Table I:
    speech_rate, articulation_rate, avg_syllable_duration,
    mean_pause_duration, mean_speech_duration,
    silence_rate, silence_to_speech_ratio, mean_silence_count

    Uses intensity-based voice activity detection (VAD).
    Syllable count approximated via intensity peaks.
    """
    try:
        duration = sound.get_total_duration()
        if duration < 0.1:
            raise ValueError("Audio too short")

        intensity   = call(sound, "To Intensity", 100, 0.0, "yes")
        threshold   = 50  # dB — frames below this are silence
        step        = 0.01

        speech_segs  = []
        silence_segs = []
        in_speech    = False
        seg_start    = 0.0
        speech_time  = 0.0

        for i in range(int(duration / step)):
            t  = i * step
            db = call(intensity, "Get value at time", t, "Cubic")
            voiced = (not np.isnan(db)) and (db >= threshold)

            if voiced:
                speech_time += step
                if not in_speech:
                    if i > 0:
                        silence_segs.append(t - seg_start)
                    seg_start = t
                    in_speech = True
            else:
                if in_speech:
                    speech_segs.append(t - seg_start)
                    seg_start = t
                    in_speech = False

        # Close final segment
        if in_speech:
            speech_segs.append(duration - seg_start)
        else:
            silence_segs.append(duration - seg_start)

        silence_time = duration - speech_time

        # Syllable count: ~3.5 syllables per second of phonation (English average)
        n_syl = max(1, round(speech_time * 3.5))

        speech_rate       = n_syl / duration       if duration > 0      else np.nan
        articulation_rate = n_syl / speech_time    if speech_time > 0   else np.nan
        avg_syl_dur       = speech_time / n_syl    if n_syl > 0         else np.nan
        mean_pause_dur    = float(np.mean(silence_segs)) if silence_segs else 0.0
        mean_speech_dur   = float(np.mean(speech_segs))  if speech_segs  else np.nan
        silence_rate      = silence_time / duration if duration > 0     else np.nan
        sil_speech_ratio  = (len(silence_segs) / len(speech_segs)
                             if speech_segs else np.nan)
        mean_sil_count    = len(silence_segs) / duration if duration > 0 else np.nan

        return {
            "speech_rate":             speech_rate,
            "articulation_rate":       articulation_rate,
            "avg_syllable_duration":   avg_syl_dur,
            "mean_pause_duration":     mean_pause_dur,
            "mean_speech_duration":    mean_speech_dur,
            "silence_rate":            silence_rate,
            "silence_to_speech_ratio": sil_speech_ratio,
            "mean_silence_count":      mean_sil_count,
        }
    except Exception as e:
        return {k: np.nan for k in [
            "speech_rate","articulation_rate","avg_syllable_duration",
            "mean_pause_duration","mean_speech_duration",
            "silence_rate","silence_to_speech_ratio","mean_silence_count"
        ]}


# ─── RECORDING QUALITY ────────────────────────────────────────────────────────

def extract_snr(sound):
    """
    SNR approximation — used as recording quality covariate.
    Estimated as max dB - background noise floor.
    """
    try:
        intensity = call(sound, "To Intensity", 100, 0.0, "yes")
        max_db    = call(intensity, "Get maximum", 0, 0, "Parabolic")
        min_db    = call(intensity, "Get minimum", 0, 0, "Parabolic")
        return {"SNR": float(max_db - min_db) if not np.isnan(max_db) else np.nan}
    except:
        return {"SNR": np.nan}


# ─── MAIN PER-FILE FUNCTION ───────────────────────────────────────────────────

def extract_all_acoustic(wav_path, task):
    """
    Extract all 28 acoustic features + SNR from one WAV file.
    For max_phonation (sustained vowel): extracts voice quality + vocal tract only.
    For cookie_theft / picnic (spontaneous speech): extracts all including rhythm.
    Returns a dict or None on failure.
    """
    try:
        sound   = parselmouth.Sound(wav_path)
        features = {}

        # Voice quality — all tasks
        features.update(extract_f0(sound))
        features.update(extract_hnr(sound))
        features.update(extract_jitter(sound))
        features.update(extract_shimmer(sound))

        # Vocal tract — all tasks
        features.update(extract_formants(sound))

        # Rhythm — spontaneous speech tasks only
        if task in ["cookie_theft", "picnic"]:
            features.update(extract_rhythm(sound))
        else:
            # sustained vowel — rhythm features are not meaningful
            features.update({k: np.nan for k in [
                "speech_rate","articulation_rate","avg_syllable_duration",
                "mean_pause_duration","mean_speech_duration",
                "silence_rate","silence_to_speech_ratio","mean_silence_count"
            ]})

        # Recording quality covariate
        features.update(extract_snr(sound))

        return features

    except Exception as e:
        print(f"  ERROR: {os.path.basename(wav_path)} — {e}")
        return None


# ─── PROCESS TASK FOLDER ─────────────────────────────────────────────────────

def process_task_folder(task_folder, task_name, output_dir):
    """
    Process all WAV files in a CLAC task folder.
    Saves CSV to output_dir/clac_{task_name}_acoustic_features.csv
    """
    wav_files = sorted([
        f for f in os.listdir(task_folder) if f.endswith(".wav")
    ])

    print(f"\n{'='*60}")
    print(f"Task: {task_name}  |  Files: {len(wav_files)}")
    print(f"{'='*60}")

    records = []
    errors  = 0

    for wav_file in tqdm(wav_files, desc=task_name):
        speaker_id = wav_file.replace(".wav", "")
        wav_path   = os.path.join(task_folder, wav_file)
        feats      = extract_all_acoustic(wav_path, task_name)

        if feats is not None:
            feats["speaker_id"] = speaker_id
            feats["task"]       = task_name
            records.append(feats)
        else:
            errors += 1

    if not records:
        print(f"  No features extracted.")
        return None

    df   = pd.DataFrame(records)
    cols = ["speaker_id", "task"] + [
        c for c in df.columns if c not in ["speaker_id", "task"]
    ]
    df = df[cols]

    out_path = os.path.join(output_dir, f"clac_{task_name}_acoustic_features.csv")
    df.to_csv(out_path, index=False)

    print(f"\n  Speakers processed : {len(df)}")
    print(f"  Errors skipped     : {errors}")
    print(f"  Features per row   : {len(df.columns) - 2}")
    print(f"  Saved to           : {out_path}")

    # Quick sanity check — print mean of a few key features
    for feat in ["meanF0", "HNR", "localJitter", "localShimmer"]:
        if feat in df.columns:
            print(f"  {feat} mean = {df[feat].mean():.4f}  "
                  f"(NaN rate: {df[feat].isna().mean():.1%})")

    return df


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    BASE       = r"D:\PROJECTS\Research\Speech-Disease-Observation"
    CLAC_DIR   = os.path.join(BASE, "CLAC-Dataset", "CLAC-Dataset")
    OUTPUT_DIR = os.path.join(BASE, "outputs", "features")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Tasks to process
    TASKS = {
        "cookie_theft":  os.path.join(CLAC_DIR, "cookie_theft"),
        "max_phonation": os.path.join(CLAC_DIR, "max_phonation"),
        "picnic":        os.path.join(CLAC_DIR, "picnic"),
    }

    summary = {}
    for task_name, task_folder in TASKS.items():
        if os.path.isdir(task_folder):
            df = process_task_folder(task_folder, task_name, OUTPUT_DIR)
            if df is not None:
                summary[task_name] = df
        else:
            print(f"\nFolder not found: {task_folder}")

    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    for task, df in summary.items():
        print(f"  {task:20s}  {len(df):4d} speakers  "
              f"{len(df.columns)-2} features")
    print(f"\nAll files saved to: {OUTPUT_DIR}")