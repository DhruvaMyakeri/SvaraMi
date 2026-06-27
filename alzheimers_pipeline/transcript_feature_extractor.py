"""
transcript_feature_extractor.py
================================
Extracts lexical and discourse features from CHAT-format (.cha) transcripts
used in the DementiaBank Pitt Corpus.

Features extracted (all clinical interpretable):
  • total_words         – total word token count
  • unique_words        – vocabulary size (type count)
  • type_token_ratio    – TTR = unique / total  (lexical diversity)
  • mean_utterance_len  – mean words per utterance
  • avg_word_length     – mean character length of words

CHAT format notes:
  - Participant utterances start with '*PAR:' (participant/patient).
  - Investigator lines start with '*INV:' and are excluded.
  - Header lines start with '@' and are excluded.
  - Continuation lines begin with a tab (part of the previous utterance).
  - Special CHAT markers ([//], [/], <...>, etc.) are stripped before counting.

Design decisions:
- Only *PAR: lines are analysed (the subject's own speech).
- Regex-based CHAT marker removal is intentionally conservative; edge cases
  produce slightly inflated word counts rather than crashes.
- TTR is well-known to be length-dependent. For research purposes, consider
  MATTR or VOCD; this pipeline provides raw TTR as a simple starting point.
"""

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CHAT line parsing patterns
# ---------------------------------------------------------------------------

# Participant utterance line (any tier label, not investigator)
_PAR_LINE = re.compile(r"^\*PAR:\s*(.+)", re.IGNORECASE)

# CHAT special annotation patterns to strip:
_CHAT_NOISE = re.compile(
    r"""
    \[[\s\S]*?\]      |  # [bracketed annotations] e.g. [/], [//], [?]
    &\S+              |  # filled pauses / disfluencies: &uh &um
    <[^>]*>           |  # retracing scope markers: <like that>
    \+[\/\\.]+        |  # +//, +/., etc.
    [^\w\s''\-]          # punctuation except apostrophes and hyphens
    """,
    re.VERBOSE,
)

# Collapse whitespace
_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_transcript_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Extract lexical features for every recording that has a .cha transcript.

    Parameters
    ----------
    manifest : pd.DataFrame
        Must contain columns: speaker_id, file_name, group, cha_path.

    Returns
    -------
    pd.DataFrame
        One row per recording. Rows without transcripts have NaN features.
        Columns: speaker_id, file_name, group,
                 total_words, unique_words, type_token_ratio,
                 mean_utterance_len, avg_word_length
    """
    has_cha = manifest["cha_path"].notna()
    n_total = len(manifest)
    n_cha = has_cha.sum()
    logger.info(
        "Transcript extraction: %d/%d recordings have .cha files.", n_cha, n_total
    )

    rows = []
    for _, rec in tqdm(manifest.iterrows(), total=n_total, desc="Parsing transcripts"):
        base = dict(
            speaker_id=rec["speaker_id"],
            file_name=rec["file_name"],
            group=rec["group"],
        )

        if pd.isna(rec.get("cha_path")):
            rows.append({**base, **_nan_transcript_row()})
            continue

        feats = _parse_cha(rec["cha_path"])
        rows.append({**base, **feats})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def _parse_cha(cha_path: str) -> dict:
    """Parse a single CHAT file and return lexical feature dict."""
    try:
        text = Path(cha_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read %s: %s", cha_path, exc)
        return _nan_transcript_row()

    utterances = _extract_participant_utterances(text)

    if not utterances:
        logger.debug("No *PAR: utterances found in %s", cha_path)
        return _nan_transcript_row()

    all_words = []
    utt_lengths = []

    for utt in utterances:
        words = _tokenise(utt)
        if words:
            all_words.extend(words)
            utt_lengths.append(len(words))

    if not all_words:
        return _nan_transcript_row()

    total_words = len(all_words)
    unique_words = len(set(w.lower() for w in all_words))
    ttr = unique_words / total_words if total_words > 0 else np.nan
    mean_utt_len = float(np.mean(utt_lengths)) if utt_lengths else np.nan
    avg_word_len = float(np.mean([len(w) for w in all_words]))

    return dict(
        total_words=int(total_words),
        unique_words=int(unique_words),
        type_token_ratio=float(ttr),
        mean_utterance_len=mean_utt_len,
        avg_word_length=avg_word_len,
    )


def _extract_participant_utterances(text: str) -> list[str]:
    """
    Extract participant (*PAR:) utterance text from raw CHAT file content.
    Handles multi-line utterances joined by tab-continuation lines.
    """
    lines = text.splitlines()
    utterances = []
    current_utt = None

    for line in lines:
        if _PAR_LINE.match(line):
            if current_utt is not None:
                utterances.append(current_utt)
            current_utt = _PAR_LINE.match(line).group(1).strip()
        elif line.startswith("\t") and current_utt is not None:
            # Continuation of current utterance
            current_utt += " " + line.strip()
        else:
            # Any other tier (@header, *INV:, %tier) ends current utterance
            if current_utt is not None:
                utterances.append(current_utt)
                current_utt = None

    if current_utt is not None:
        utterances.append(current_utt)

    return utterances


def _tokenise(raw: str) -> list[str]:
    """
    Clean CHAT annotations and return a list of word tokens.
    """
    cleaned = _CHAT_NOISE.sub(" ", raw)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    words = [w for w in cleaned.split() if _is_real_word(w)]
    return words


def _is_real_word(token: str) -> bool:
    """
    Filter out non-lexical tokens remaining after CHAT cleaning.
    Accepts tokens that contain at least one alphabetic character.
    """
    return bool(re.search(r"[a-zA-Z]", token))


# ---------------------------------------------------------------------------
# NaN skeleton
# ---------------------------------------------------------------------------

def _nan_transcript_row() -> dict:
    return dict(
        total_words=np.nan,
        unique_words=np.nan,
        type_token_ratio=np.nan,
        mean_utterance_len=np.nan,
        avg_word_length=np.nan,
    )


TRANSCRIPT_FEATURE_COLUMNS = list(_nan_transcript_row().keys())
