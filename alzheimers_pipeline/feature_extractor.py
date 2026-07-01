"""
feature_extractor.py
====================
Extracts clinically interpretable acoustic speech biomarkers from WAV files
using Parselmouth (Python binding for Praat).

Feature set mirrors published clinical studies on neurodegenerative disease:
  • Fundamental frequency (F0) statistics
  • Harmonics-to-Noise Ratio (HNR)
  • Jitter variants (local, absolute, RAP, PPQ5)
  • Shimmer variants (local, dB, APQ3, APQ5, APQ11)
  • Formant means (F1–F4)
  • Timing & fluency (duration, pauses, rates)

Design decisions:
- Silence threshold: -25 dB (Praat default, appropriate for studio/semi-studio
  recordings; adjust SILENCE_THRESHOLD if needed for noisier corpora).
- Pitch floor/ceiling: 75–500 Hz (broad range covering male and female voices).
- Formant analysis uses up to 5.5 kHz maximum (Burg method, 5 formants).
- All exceptions are caught per-file; a NaN row is returned on failure.
- tqdm progress bars work both in terminal and Jupyter.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Praat analysis constants
# ---------------------------------------------------------------------------
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 500.0
SILENCE_THRESHOLD_DB = -25.0       # dB below peak for silence detection
MIN_SILENCE_DURATION_S = 0.15      # minimum pause length (s)
MIN_SOUNDING_DURATION_S = 0.05     # minimum voiced segment (s)
FORMANT_MAX_FREQ_HZ = 5500.0       # Burg formant ceiling
N_FORMANTS = 5                     # number of formants to track
JITTER_MAX_PERIOD = 0.02           # 20 ms max period for jitter (Praat default)
JITTER_MAX_FACTOR = 1.3
SHIMMER_MAX_DB = 1.3               # Praat default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_all_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over all recordings in the manifest and extract acoustic features.

    Parameters
    ----------
    manifest : pd.DataFrame
        Output of data_loader.discover_dataset().

    Returns
    -------
    pd.DataFrame – one row per recording, all acoustic features as columns.
    """
    rows = []
    for _, rec in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting features"):
        row = _extract_one(rec)
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(
        "Feature extraction complete: %d/%d succeeded.",
        df["f0_mean"].notna().sum(), len(df),
    )
    return df


def extract_features_from_file(file_path: str | Path) -> dict:
    """Public single-file entry point (useful for quick tests)."""
    rec = pd.Series(
        dict(speaker_id="test", file_name=Path(file_path).name,
             file_path=str(file_path), group="unknown", cha_path=None)
    )
    return _extract_one(rec)


# ---------------------------------------------------------------------------
# Per-file extraction
# ---------------------------------------------------------------------------

def _extract_one(rec: pd.Series) -> dict:
    """Extract all acoustic features from one recording. Never raises."""
    base = dict(
        speaker_id=rec["speaker_id"],
        file_name=rec["file_name"],
        group=rec["group"],
    )

    try:
        sound = parselmouth.Sound(rec["file_path"])
    except Exception as exc:
        logger.warning("Cannot load %s: %s", rec["file_path"], exc)
        return {**base, **_nan_row()}

    try:
        features = {}
        features.update(_extract_f0(sound))
        features.update(_extract_hnr(sound))
        features.update(_extract_jitter(sound))
        features.update(_extract_shimmer(sound))
        features.update(_extract_formants(sound))
        features.update(_extract_timing(sound))
        return {**base, **features}
    except Exception as exc:
        logger.warning("Feature extraction failed for %s: %s", rec["file_path"], exc)
        return {**base, **_nan_row()}


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

def _extract_f0(sound: parselmouth.Sound) -> dict:
    """Fundamental frequency statistics (voiced frames only)."""
    try:
        pitch = sound.to_pitch(
            pitch_floor=PITCH_FLOOR_HZ,
            pitch_ceiling=PITCH_CEILING_HZ,
        )
        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]

        if voiced.size == 0:
            return dict(f0_mean=np.nan, f0_std=np.nan,
                        f0_min=np.nan, f0_max=np.nan)
        return dict(
            f0_mean=float(np.mean(voiced)),
            f0_std=float(np.std(voiced)),
            f0_min=float(np.min(voiced)),
            f0_max=float(np.max(voiced)),
        )
    except Exception:
        return dict(f0_mean=np.nan, f0_std=np.nan,
                    f0_min=np.nan, f0_max=np.nan)


def _extract_hnr(sound: parselmouth.Sound) -> dict:
    """Harmonics-to-Noise Ratio (mean over voiced frames)."""
    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01,
                           PITCH_FLOOR_HZ, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        return dict(hnr=float(hnr) if np.isfinite(hnr) else np.nan)
    except Exception:
        return dict(hnr=np.nan)


def _extract_jitter(sound: parselmouth.Sound) -> dict:
    """
    Jitter variants: local, absolute, RAP, PPQ5.
    Computed from PointProcess (glottal pulse train).
    """
    result = dict(
        jitter_local=np.nan, jitter_absolute=np.nan,
        jitter_rap=np.nan, jitter_ppq5=np.nan,
    )
    try:
        pitch = sound.to_pitch(
            pitch_floor=PITCH_FLOOR_HZ,
            pitch_ceiling=PITCH_CEILING_HZ,
        )
        pp = call(sound, "To PointProcess (periodic, cc)",
                  PITCH_FLOOR_HZ, PITCH_CEILING_HZ)

        dur = sound.get_total_duration()

        def safe_jitter(method):
            try:
                val = call(pp, method, 0, dur,
                           JITTER_MAX_PERIOD / PITCH_CEILING_HZ,
                           JITTER_MAX_PERIOD, JITTER_MAX_FACTOR)
                return float(val) if np.isfinite(val) else np.nan
            except Exception:
                return np.nan

        result["jitter_local"] = safe_jitter("Get jitter (local)")
        result["jitter_absolute"] = safe_jitter("Get jitter (local, absolute)")
        result["jitter_rap"] = safe_jitter("Get jitter (rap)")
        result["jitter_ppq5"] = safe_jitter("Get jitter (ppq5)")
    except Exception:
        pass
    return result


def _extract_shimmer(sound: parselmouth.Sound) -> dict:
    """
    Shimmer variants: local, dB, APQ3, APQ5, APQ11.
    Computed from glottal PointProcess.
    """
    result = dict(
        shimmer_local=np.nan, shimmer_db=np.nan,
        shimmer_apq3=np.nan, shimmer_apq5=np.nan, shimmer_apq11=np.nan,
    )
    try:
        pp = call(sound, "To PointProcess (periodic, cc)",
                  PITCH_FLOOR_HZ, PITCH_CEILING_HZ)
        dur = sound.get_total_duration()

        def safe_shimmer(method, extra=None):
            try:
                args = [pp, sound, method, 0, dur,
                        JITTER_MAX_PERIOD / PITCH_CEILING_HZ,
                        JITTER_MAX_PERIOD, JITTER_MAX_FACTOR]
                if extra is not None:
                    args.append(extra)
                val = call(args[0:2], method, *args[2:])
                return float(val) if np.isfinite(val) else np.nan
            except Exception:
                return np.nan

        # Praat shimmer functions require both PointProcess and Sound
        dur = sound.get_total_duration()
        min_p = 1.0 / PITCH_CEILING_HZ
        max_p = JITTER_MAX_PERIOD

        def shimmer_call(method, *extra_args):
            try:
                val = call([pp, sound], method,
                           0, dur, min_p, max_p,
                           JITTER_MAX_FACTOR, SHIMMER_MAX_DB,
                           *extra_args)
                return float(val) if np.isfinite(val) else np.nan
            except Exception:
                return np.nan

        result["shimmer_local"] = shimmer_call("Get shimmer (local)")
        result["shimmer_db"] = shimmer_call("Get shimmer (local, dB)")
        result["shimmer_apq3"] = shimmer_call("Get shimmer (apq3)")
        result["shimmer_apq5"] = shimmer_call("Get shimmer (apq5)")
        result["shimmer_apq11"] = shimmer_call("Get shimmer (apq11)")
    except Exception:
        pass
    return result


def _extract_formants(sound: parselmouth.Sound) -> dict:
    """Mean formant frequencies F1–F4 using LPC Burg method."""
    result = {f"f{i}_mean": np.nan for i in range(1, 5)}
    try:
        formants = call(
            sound, "To Formant (burg)",
            0.0,                # time step (0 = auto)
            N_FORMANTS,
            FORMANT_MAX_FREQ_HZ,
            0.025,              # window length (s)
            50.0,               # pre-emphasis from (Hz)
        )
        dur = sound.get_total_duration()
        for i in range(1, 5):
            try:
                val = call(formants, f"Get mean", i, 0, dur, "hertz")
                result[f"f{i}_mean"] = float(val) if np.isfinite(val) else np.nan
            except Exception:
                pass
    except Exception:
        pass
    return result


def _extract_timing(sound: parselmouth.Sound) -> dict:
    """
    Timing and fluency features derived from TextGrid silence detection.

    Approach: Use Praat's 'To TextGrid (silences)' to segment the recording
    into sounding and silent intervals. This avoids any VAD model dependency.
    """
    result = dict(
        recording_duration=np.nan,
        speaking_duration=np.nan,
        n_pauses=np.nan,
        mean_pause_duration=np.nan,
        max_pause_duration=np.nan,
        total_pause_time=np.nan,
        silence_ratio=np.nan,
        speech_rate=np.nan,       # syllables/s – approximated via voiced segments
        articulation_rate=np.nan, # speech frames / speaking time
        phonation_time_ratio=np.nan,
    )
    try:
        total_dur = sound.get_total_duration()
        result["recording_duration"] = float(total_dur)

        # Intensity-based silence detection
        intensity = sound.to_intensity(minimum_pitch=PITCH_FLOOR_HZ)
        textgrid = call(
            intensity, "To TextGrid (silences)",
            SILENCE_THRESHOLD_DB,
            MIN_SILENCE_DURATION_S,
            MIN_SOUNDING_DURATION_S,
            "silent",
            "sounding",
        )

        n_intervals = call(textgrid, "Get number of intervals", 1)

        silent_durations = []
        sounding_durations = []

        for i in range(1, n_intervals + 1):
            label = call(textgrid, "Get label of interval", 1, i)
            t_start = call(textgrid, "Get start time of interval", 1, i)
            t_end = call(textgrid, "Get end time of interval", 1, i)
            dur_i = t_end - t_start
            if label == "silent":
                silent_durations.append(dur_i)
            else:
                sounding_durations.append(dur_i)

        speaking_dur = sum(sounding_durations)
        total_pause = sum(silent_durations)
        n_pauses = len(silent_durations)

        result.update(
            speaking_duration=float(speaking_dur),
            n_pauses=int(n_pauses),
            mean_pause_duration=float(np.mean(silent_durations)) if silent_durations else 0.0,
            max_pause_duration=float(np.max(silent_durations)) if silent_durations else 0.0,
            total_pause_time=float(total_pause),
            silence_ratio=float(total_pause / total_dur) if total_dur > 0 else np.nan,
            phonation_time_ratio=float(speaking_dur / total_dur) if total_dur > 0 else np.nan,
        )

        # Articulation rate: voiced frames per second of speaking time.
        # Speech rate: approximated as n_voiced_frames / total_duration.
        pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
        voiced = (pitch.selected_array["frequency"] > 0).sum()
        dt = pitch.dt

        if speaking_dur > 0:
            result["articulation_rate"] = float((voiced * dt) / speaking_dur)
        if total_dur > 0:
            result["speech_rate"] = float((voiced * dt) / total_dur)

    except Exception as exc:
        logger.debug("Timing extraction failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# NaN skeleton
# ---------------------------------------------------------------------------

def _nan_row() -> dict:
    """Return a dict of all feature names mapped to NaN for failed files."""
    keys = [
        "f0_mean", "f0_std", "f0_min", "f0_max",
        "hnr",
        "jitter_local", "jitter_absolute", "jitter_rap", "jitter_ppq5",
        "shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11",
        "f1_mean", "f2_mean", "f3_mean", "f4_mean",
        "recording_duration", "speaking_duration",
        "n_pauses", "mean_pause_duration", "max_pause_duration",
        "total_pause_time", "silence_ratio",
        "speech_rate", "articulation_rate", "phonation_time_ratio",
    ]
    return {k: np.nan for k in keys}


# ---------------------------------------------------------------------------
# Column ordering for downstream use
# ---------------------------------------------------------------------------
ACOUSTIC_FEATURE_COLUMNS = list(_nan_row().keys())
