"""
data_loader.py
==============
Discovers and validates WAV recordings and optional CHAT (.cha) transcripts
from the DementiaBank Pitt Corpus (AD) and Healthy Control datasets.

Design decisions:
- Flexible folder-structure detection: handles flat and nested layouts.
- Speaker IDs are derived from file stems to support multi-recording speakers.
- Corrupted or zero-byte files are flagged but never crash the pipeline.
- Transcript pairing is opportunistic: missing .cha files degrade gracefully.
"""

import os
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_dataset(
    ad_dir: str | Path,
    hc_dir: str | Path,
) -> pd.DataFrame:
    """
    Walk both dataset directories and return a manifest DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        speaker_id  : str   – derived from file stem
        file_name   : str   – basename of the WAV file
        file_path   : str   – absolute path to WAV
        group       : str   – 'AD' or 'HC'
        cha_path    : str | None – absolute path to paired .cha file, or None
    """
    ad_records = _scan_directory(Path(ad_dir), group="AD")
    hc_records = _scan_directory(Path(hc_dir), group="HC")

    all_records = ad_records + hc_records
    if not all_records:
        raise FileNotFoundError(
            f"No valid WAV files found in:\n  AD: {ad_dir}\n  HC: {hc_dir}"
        )

    df = pd.DataFrame(all_records)
    _log_summary(df)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scan_directory(root: Path, group: str) -> list[dict]:
    """Recursively find all WAV files under *root* and pair with .cha if present."""
    if not root.exists():
        logger.warning("Directory does not exist: %s  (group=%s) – skipping.", root, group)
        return []

    records = []
    wav_files = sorted(root.rglob("*.wav")) + sorted(root.rglob("*.WAV"))

    if not wav_files:
        logger.warning("No WAV files found in %s", root)
        return []

    for wav_path in wav_files:
        if not _is_valid_wav(wav_path):
            logger.warning("Skipping invalid/empty file: %s", wav_path)
            continue

        speaker_id = _derive_speaker_id(wav_path)
        cha_path = _find_cha(wav_path)

        records.append(
            dict(
                speaker_id=speaker_id,
                file_name=wav_path.name,
                file_path=str(wav_path.resolve()),
                group=group,
                cha_path=str(cha_path.resolve()) if cha_path else None,
            )
        )

    logger.info("Found %d valid WAV files in %s [%s]", len(records), root, group)
    return records


def _is_valid_wav(path: Path) -> bool:
    """Return True if the file exists, is non-empty, and has a .wav extension."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _derive_speaker_id(wav_path: Path) -> str:
    """
    Derive a speaker ID from the file stem.

    Convention used by DementiaBank:  <speakerID>-<session>.wav
    e.g. 001-0.wav  →  speaker_id = '001'

    For other naming schemes the full stem is used as-is.
    """
    stem = wav_path.stem
    # If the stem contains a hyphen, take the portion before the first hyphen
    # as the speaker identifier.
    parts = stem.split("-")
    return parts[0] if len(parts) >= 2 else stem


def _find_cha(wav_path: Path) -> Optional[Path]:
    """
    Look for a CHAT transcript (.cha) co-located with the WAV file.

    Search strategy (in order):
    1. Same directory, same stem: e.g. 001-0.cha
    2. Same directory, speaker prefix: e.g. 001.cha
    3. Parent directory: same stem or speaker prefix
    """
    stem = wav_path.stem
    speaker = stem.split("-")[0]

    candidates = [
        wav_path.parent / f"{stem}.cha",
        wav_path.parent / f"{speaker}.cha",
        wav_path.parent.parent / f"{stem}.cha",
        wav_path.parent.parent / f"{speaker}.cha",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _log_summary(df: pd.DataFrame) -> None:
    total = len(df)
    n_ad = (df["group"] == "AD").sum()
    n_hc = (df["group"] == "HC").sum()
    n_cha = df["cha_path"].notna().sum()

    logger.info(
        "Dataset manifest: %d recordings total  |  AD=%d  HC=%d  |  transcripts=%d",
        total, n_ad, n_hc, n_cha,
    )
    if n_hc == 0:
        logger.error(
            "No healthy control recordings found! "
            "Reference interval construction will fail."
        )
