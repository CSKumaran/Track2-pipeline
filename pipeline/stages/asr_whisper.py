"""Stage 1 – ASR transcription using Whisper (local)."""

from __future__ import annotations

import json
import logging
import os

import pandas as pd

from pipeline.utils.ffmpeg_utils import extract_audio
from pipeline.utils.io_utils import ensure_dir, safe_write_csv

logger = logging.getLogger(__name__)


def run_asr_whisper(
    video_path: str,
    output_dir: str,
    model_name: str = "medium",
) -> dict:
    """Run Whisper ASR on *video_path* and write word / segment CSVs.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_dir : str
        Directory where outputs will be written.
    model_name : str
        Whisper model size (e.g. "tiny", "base", "small", "medium", "large").

    Returns
    -------
    dict
        ASR statistics: n_words, n_segments, total_duration,
        mean_words_per_segment.
    """
    ensure_dir(output_dir)

    # 0. Check for cached outputs ─────────────────────────────────────
    words_path = os.path.join(output_dir, "transcript_words.csv")
    segs_path = os.path.join(output_dir, "transcript_segments.csv")
    improved_path = os.path.join(output_dir, "transcript_segments_improved.csv")
    if all(os.path.exists(p) for p in [words_path, segs_path, improved_path]):
        logger.info("ASR outputs already exist, skipping transcription.")
        words_df = pd.read_csv(words_path)
        segs_df = pd.read_csv(segs_path)
        improved_df = pd.read_csv(improved_path)
        n_words = len(words_df)
        total_duration = float(segs_df["end_time"].max()) if len(segs_df) else 0.0
        return {
            "n_words": n_words,
            "n_segments_original": len(segs_df),
            "n_segments_improved": len(improved_df),
            "total_duration": round(total_duration, 2),
            "mean_words_per_segment": round(n_words / len(improved_df), 2) if len(improved_df) else 0.0,
        }

    # 1. Extract audio ────────────────────────────────────────────────
    audio_path = extract_audio(video_path, output_dir)

    # 2. Transcribe with faster-whisper (preferred) or openai-whisper ─
    segments_raw, words_raw = _transcribe(audio_path, model_name)

    # 3. Build DataFrames ─────────────────────────────────────────────
    word_rows = []
    for wid, w in enumerate(words_raw):
        word_rows.append({
            "word_id": wid,
            "word": w["word"],
            "start_time": round(w["start"], 4),
            "end_time": round(w["end"], 4),
        })
    words_df = pd.DataFrame(word_rows)

    seg_rows = []
    for sid, seg in enumerate(segments_raw):
        seg_rows.append({
            "segment_id": sid,
            "text": seg["text"].strip(),
            "start_time": round(seg["start"], 4),
            "end_time": round(seg["end"], 4),
        })
    segments_df = pd.DataFrame(seg_rows)

    # 4. Write original Whisper outputs ──────────────────────────────
    safe_write_csv(words_df, os.path.join(output_dir, "transcript_words.csv"))
    safe_write_csv(segments_df, os.path.join(output_dir, "transcript_segments.csv"))

    raw_json_path = os.path.join(output_dir, "whisper_raw.json")
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump({"segments": segments_raw, "words": words_raw}, f,
                  ensure_ascii=False, indent=2)
    logger.info("Wrote raw Whisper JSON → %s", raw_json_path)

    # 5. Build improved segments (sentence-boundary re-segmentation) ─
    improved_df = _resegment_into_sentences(words_df)
    safe_write_csv(improved_df,
                   os.path.join(output_dir, "transcript_segments_improved.csv"))

    # 6. Compute statistics ───────────────────────────────────────────
    n_words = len(words_df)
    n_segments_orig = len(segments_df)
    n_segments_improved = len(improved_df)
    total_duration = float(segments_df["end_time"].max()) if n_segments_orig else 0.0
    mean_wps = n_words / n_segments_improved if n_segments_improved else 0.0

    stats = {
        "n_words": n_words,
        "n_segments_original": n_segments_orig,
        "n_segments_improved": n_segments_improved,
        "total_duration": round(total_duration, 2),
        "mean_words_per_segment": round(mean_wps, 2),
    }
    logger.info("ASR stats: %s", stats)
    return stats


# ── Internal transcription dispatcher ────────────────────────────────

def _transcribe(audio_path: str, model_name: str):
    """Try faster-whisper first, fall back to openai-whisper.

    Returns (segments_list, words_list) where each element is a plain dict.
    """
    try:
        return _transcribe_faster_whisper(audio_path, model_name)
    except ImportError:
        logger.info("faster-whisper not installed; falling back to openai-whisper.")
        return _transcribe_openai_whisper(audio_path, model_name)


def _transcribe_faster_whisper(audio_path: str, model_name: str):
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="auto", compute_type="auto")
    raw_segments, _info = model.transcribe(
        audio_path, beam_size=5, word_timestamps=True
    )

    segments_out: list[dict] = []
    words_out: list[dict] = []

    for seg in raw_segments:
        segments_out.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        })
        if seg.words:
            for w in seg.words:
                words_out.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                })

    return segments_out, words_out


def _transcribe_openai_whisper(audio_path: str, model_name: str):
    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, word_timestamps=True)

    segments_out: list[dict] = []
    words_out: list[dict] = []

    for seg in result.get("segments", []):
        segments_out.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        })
        for w in seg.get("words", []):
            words_out.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"],
            })

    return segments_out, words_out


# ── Sentence-boundary re-segmentation ────────────────────────────────

import re

# Sentence-ending punctuation pattern
_SENT_END = re.compile(r'[.!?]$')

# Minimum words per segment — avoids orphaned tiny fragments
_MIN_WORDS_PER_SEGMENT = 4


def _resegment_into_sentences(words_df: pd.DataFrame) -> pd.DataFrame:
    """Re-segment word-level transcript into clean sentence-level segments.

    Two-pass approach:
      1. **Sentence split**: accumulate words until sentence-ending
         punctuation is reached, then emit a segment.
      2. **Merge short**: any segment with fewer than _MIN_WORDS_PER_SEGMENT
         words is merged into its predecessor.

    Parameters
    ----------
    words_df : pd.DataFrame
        Must have columns: word, start_time, end_time.

    Returns
    -------
    pd.DataFrame
        Columns: segment_id, text, start_time, end_time.
    """
    if words_df.empty:
        return pd.DataFrame(columns=["segment_id", "text", "start_time", "end_time"])

    # ── Pass 1: split on sentence-ending punctuation ─────────────────
    raw_segments: list[dict] = []
    current_words: list[str] = []
    seg_start = None

    for _, row in words_df.iterrows():
        word = str(row["word"]).strip()
        if not word:
            continue

        if seg_start is None:
            seg_start = row["start_time"]
        current_words.append(word)

        if _SENT_END.search(word):
            raw_segments.append({
                "text": " ".join(current_words),
                "start_time": round(seg_start, 4),
                "end_time": round(row["end_time"], 4),
                "n_words": len(current_words),
            })
            current_words = []
            seg_start = None

    # Flush remaining words (last sentence may lack final punctuation)
    if current_words and seg_start is not None:
        raw_segments.append({
            "text": " ".join(current_words),
            "start_time": round(seg_start, 4),
            "end_time": round(float(words_df.iloc[-1]["end_time"]), 4),
            "n_words": len(current_words),
        })

    if not raw_segments:
        return pd.DataFrame(columns=["segment_id", "text", "start_time", "end_time"])

    # ── Pass 2: merge short segments into predecessor ────────────────
    merged: list[dict] = [raw_segments[0]]

    for seg in raw_segments[1:]:
        if seg["n_words"] < _MIN_WORDS_PER_SEGMENT:
            # Merge into previous segment
            prev = merged[-1]
            prev["text"] = prev["text"] + " " + seg["text"]
            prev["end_time"] = seg["end_time"]
            prev["n_words"] += seg["n_words"]
        else:
            merged.append(seg)

    # Build final DataFrame
    rows = []
    for sid, seg in enumerate(merged):
        rows.append({
            "segment_id": sid,
            "text": seg["text"],
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
        })

    improved_df = pd.DataFrame(rows)
    logger.info("Re-segmentation: %d words → %d sentences (pass1=%d, after merge=%d)",
                len(words_df), len(improved_df),
                len(raw_segments), len(merged))
    return improved_df
