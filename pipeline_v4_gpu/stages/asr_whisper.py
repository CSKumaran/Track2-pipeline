"""Stage 1: ASR Transcription (WhisperX + pySBD + sanity checker) [v4.0]."""

import logging
import os
import pandas as pd
import pysbd

# PyTorch 2.6 compat patch is applied in __init__.py (package level)

from ..config import Config
from ..utils.ffmpeg_utils import extract_audio
from ..utils.io_utils import save_csv, cache_exists
from ..utils.timestamp_validator import validate_timestamps

logger = logging.getLogger(__name__)

CACHE_FILES = [
    "transcript_words.csv",
    "transcript_segments.csv",
    "transcript_segments_improved.csv",
]


def run_stage1(video_path: str, output_dir: str, cfg: Config, diag=None) -> dict:
    """Run ASR transcription. Returns dict with paths to output CSVs."""
    if cache_exists(output_dir, CACHE_FILES):
        logger.info("Stage 1: cache hit, skipping ASR")
        paths = {f: os.path.join(output_dir, f) for f in CACHE_FILES}
        # Still produce diagnostics from cached files
        if diag is not None:
            _write_diagnostics_from_cache(output_dir, diag)
        return paths

    wav_path = extract_audio(video_path, output_dir)

    asr_backend_used = None
    if cfg.ASR_BACKEND == "whisperx":
        try:
            words_df, segments_df = _run_whisperx(wav_path, cfg)
            asr_backend_used = "whisperx"
            logger.info("ASR backend: whisperx")
        except Exception as e:
            logger.warning("WhisperX failed (%s), falling back to faster-whisper", e)
            words_df, segments_df = _run_faster_whisper(wav_path, cfg)
            asr_backend_used = "faster-whisper (fallback)"
            logger.info("ASR backend: faster-whisper (fallback)")
    else:
        words_df, segments_df = _run_faster_whisper(wav_path, cfg)
        asr_backend_used = "faster-whisper"
        logger.info("ASR backend: faster-whisper")

    # pySBD re-segmentation
    improved_df = _resegment_pysbd(words_df, segments_df, cfg)

    # Timestamp sanity checker [v2.1]
    words_df = validate_timestamps(words_df)

    # Save
    paths = {}
    for name, df in [
        ("transcript_words.csv", words_df),
        ("transcript_segments.csv", segments_df),
        ("transcript_segments_improved.csv", improved_df),
    ]:
        p = os.path.join(output_dir, name)
        save_csv(df, p)
        paths[name] = p

    # === Diagnostics [v2.2] ===
    if diag is not None:
        _write_diagnostics(words_df, segments_df, improved_df, asr_backend_used, diag)

    return paths


def _run_whisperx(wav_path: str, cfg: Config) -> tuple:
    """Run WhisperX with forced alignment."""
    import torch
    import whisperx

    logger.info("torch.load patched at module level for PyTorch 2.6 compat")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ctranslate2 on CPU doesn't support float16; auto-select
    if device == "cpu":
        compute_type = "int8"
    else:
        compute_type = cfg.WHISPER_COMPUTE_TYPE
    language = cfg.WHISPER_LANGUAGE if cfg.WHISPER_LANGUAGE else None
    logger.info("Loading WhisperX model: %s (device=%s, compute=%s, beam=%d, lang=%s)",
                cfg.WHISPER_MODEL, device, compute_type, cfg.WHISPER_BEAM_SIZE,
                language or "auto")
    model = whisperx.load_model(
        cfg.WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
        language=language,
        asr_options={"beam_size": cfg.WHISPER_BEAM_SIZE},
    )

    logger.info("Transcribing (batch_size=%d, beam_size=%d)...",
                cfg.WHISPER_BATCH_SIZE, cfg.WHISPER_BEAM_SIZE)
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(
        audio,
        batch_size=cfg.WHISPER_BATCH_SIZE,
    )

    # Forced alignment
    lang_code = language or result.get("language", "en")
    logger.info("Running forced alignment (language=%s)...", lang_code)
    align_model, metadata = whisperx.load_align_model(
        language_code=lang_code, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False
    )

    # Free memory
    del model, align_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Build words DataFrame
    words = []
    word_id = 0
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({
                "word_id": word_id,
                "word": w.get("word", ""),
                "start_time": w.get("start", None),
                "end_time": w.get("end", None),
            })
            word_id += 1

    words_df = pd.DataFrame(words)

    # Build segments DataFrame
    segments = []
    for i, seg in enumerate(result["segments"]):
        segments.append({
            "segment_id": i,
            "text": seg.get("text", ""),
            "start_time": seg.get("start", 0),
            "end_time": seg.get("end", 0),
        })
    segments_df = pd.DataFrame(segments)

    return words_df, segments_df


def _run_faster_whisper(wav_path: str, cfg: Config) -> tuple:
    """Fallback: faster-whisper without forced alignment."""
    from faster_whisper import WhisperModel

    # Use GPU if available, fallback to CPU with safe compute type
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"
    logger.info("Loading faster-whisper: %s (device=%s, compute=%s)", cfg.WHISPER_MODEL, device, compute_type)
    model = WhisperModel(cfg.WHISPER_MODEL, device=device, compute_type=compute_type)

    segments_raw, info = model.transcribe(wav_path, word_timestamps=True)
    segments_list = list(segments_raw)

    del model

    words = []
    word_id = 0
    segments = []
    for i, seg in enumerate(segments_list):
        segments.append({
            "segment_id": i,
            "text": seg.text.strip(),
            "start_time": seg.start,
            "end_time": seg.end,
        })
        for w in (seg.words or []):
            words.append({
                "word_id": word_id,
                "word": w.word.strip(),
                "start_time": w.start,
                "end_time": w.end,
            })
            word_id += 1

    words_df = pd.DataFrame(words)
    segments_df = pd.DataFrame(segments)
    return words_df, segments_df


def _resegment_pysbd(words_df: pd.DataFrame, segments_df: pd.DataFrame,
                     cfg: Config = None) -> pd.DataFrame:
    """Re-segment transcript using pySBD for better sentence boundaries.

    [v2.2] Added force-split for segments exceeding PYSBD_MAX_SEGMENT_WORDS.
    Splits at the best clause boundary (comma, semicolon, conjunction) near
    the midpoint, falling back to midpoint if no punctuation found.
    """
    import re

    if words_df.empty:
        return segments_df.copy()

    max_words = cfg.PYSBD_MAX_SEGMENT_WORDS if cfg else 40

    # Full transcript text
    full_text = " ".join(words_df["word"].astype(str).tolist())

    # pySBD segmentation
    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = segmenter.segment(full_text)

    # Map sentences back to word timestamps
    improved_segments = []
    word_idx = 0
    for seg_id, sentence in enumerate(sentences):
        sent_words = sentence.split()
        if not sent_words:
            continue

        # Find matching words in words_df
        start_idx = word_idx
        matched = 0
        while word_idx < len(words_df) and matched < len(sent_words):
            word_idx += 1
            matched += 1
        end_idx = word_idx - 1

        if start_idx >= len(words_df):
            break

        start_time = words_df.iloc[start_idx]["start_time"]
        end_time = words_df.iloc[min(end_idx, len(words_df) - 1)]["end_time"]

        improved_segments.append({
            "segment_id": seg_id,
            "text": sentence.strip(),
            "start_time": start_time,
            "end_time": end_time,
            "n_words": len(sent_words),
            "_word_start_idx": start_idx,  # track for force-split
            "_word_end_idx": min(end_idx, len(words_df) - 1),
        })

    improved_df = pd.DataFrame(improved_segments)

    # Merge short segments (< 4 words) into predecessor
    if len(improved_df) > 1:
        merged = []
        for _, row in improved_df.iterrows():
            if merged and row["n_words"] < 4:
                merged[-1]["text"] += " " + row["text"]
                merged[-1]["end_time"] = row["end_time"]
                merged[-1]["n_words"] += row["n_words"]
                merged[-1]["_word_end_idx"] = row["_word_end_idx"]
            else:
                merged.append(row.to_dict())

        improved_df = pd.DataFrame(merged)
        improved_df["segment_id"] = range(len(improved_df))

    # [v2.2] Force-split segments exceeding max_words
    if max_words and len(improved_df) > 0:
        split_segments = []
        n_splits = 0
        for _, row in improved_df.iterrows():
            if row["n_words"] <= max_words:
                split_segments.append(row.to_dict())
                continue

            # This segment is too long — force-split it
            ws = int(row["_word_start_idx"])
            we = int(row["_word_end_idx"])
            seg_words_df = words_df.iloc[ws:we + 1]

            # Find split points: clause boundaries (comma, semicolon, "and", "but", "or", "so", "because", "however")
            clause_pattern = re.compile(r'[,;]$')
            conjunctions = {"and", "but", "or", "so", "because", "however", "although", "while", "when", "then"}

            # Collect candidate split positions (word index within this segment)
            candidates = []
            seg_word_list = seg_words_df["word"].astype(str).tolist()
            for j in range(len(seg_word_list)):
                word_clean = seg_word_list[j].strip()
                if clause_pattern.search(word_clean):
                    candidates.append(j + 1)  # split AFTER this word
                elif word_clean.lower() in conjunctions and j > 0:
                    candidates.append(j)  # split BEFORE conjunction

            # Recursively split into chunks <= max_words
            chunk_boundaries = _find_split_points(0, len(seg_word_list), max_words, candidates)

            for ci, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
                abs_start = ws + chunk_start
                abs_end = ws + chunk_end - 1
                if abs_start >= len(words_df) or abs_end < abs_start:
                    continue
                chunk_text = " ".join(seg_word_list[chunk_start:chunk_end])
                split_segments.append({
                    "text": chunk_text.strip(),
                    "start_time": words_df.iloc[abs_start]["start_time"],
                    "end_time": words_df.iloc[min(abs_end, len(words_df) - 1)]["end_time"],
                    "n_words": chunk_end - chunk_start,
                    "_word_start_idx": abs_start,
                    "_word_end_idx": abs_end,
                })
            n_splits += 1

        if n_splits > 0:
            logger.info("pySBD force-split %d oversized segments (max_words=%d)", n_splits, max_words)
            improved_df = pd.DataFrame(split_segments)
            improved_df["segment_id"] = range(len(improved_df))

    # Drop internal tracking columns
    improved_df = improved_df.drop(columns=["_word_start_idx", "_word_end_idx"], errors="ignore")

    return improved_df


def _find_split_points(start: int, end: int, max_words: int, candidates: list) -> list:
    """Recursively find split points for a word range [start, end) to keep chunks <= max_words.

    Returns list of (chunk_start, chunk_end) tuples.
    """
    length = end - start
    if length <= max_words:
        return [(start, end)]

    midpoint = start + length // 2

    # Find best candidate split near midpoint
    best = None
    best_dist = float("inf")
    for c in candidates:
        if start < c < end:  # must be strictly inside
            dist = abs(c - midpoint)
            if dist < best_dist:
                best = c
                best_dist = dist

    # Fallback: split at midpoint if no clause boundary found
    if best is None:
        best = midpoint

    left = _find_split_points(start, best, max_words, candidates)
    right = _find_split_points(best, end, max_words, candidates)
    return left + right


# =====================================================================
# Diagnostics [v2.2]
# =====================================================================

def _write_diagnostics(words_df, segments_df, improved_df, asr_backend_used, diag):
    """Write detailed Stage 1 diagnostics."""
    n_words = len(words_df)
    n_segments_raw = len(segments_df)
    n_segments_improved = len(improved_df)

    # Timestamp reliability analysis
    if "timestamp_reliable" in words_df.columns:
        n_reliable = int(words_df["timestamp_reliable"].sum())
        n_flagged = n_words - n_reliable
    else:
        n_reliable = n_words
        n_flagged = 0

    pct_unreliable = (n_flagged / n_words * 100) if n_words > 0 else 0.0

    # Flag reason breakdown — re-run the validation logic to categorize
    flag_reasons = {
        "duration_out_of_range": 0,
        "large_gap_over_3s": 0,
        "non_monotonic": 0,
        "missing_timestamp": 0,
    }
    if not words_df.empty and "timestamp_reliable" in words_df.columns:
        wdf = words_df.copy()
        wdf["start_time"] = pd.to_numeric(wdf["start_time"], errors="coerce")
        wdf["end_time"] = pd.to_numeric(wdf["end_time"], errors="coerce")

        for i in range(len(wdf)):
            if wdf.iloc[i]["timestamp_reliable"]:
                continue
            start = wdf.iloc[i]["start_time"]
            end = wdf.iloc[i]["end_time"]
            if pd.isna(start) or pd.isna(end):
                flag_reasons["missing_timestamp"] += 1
            elif (end - start) < 0.05 or (end - start) > 2.0:
                flag_reasons["duration_out_of_range"] += 1
            elif i > 0:
                prev_end = wdf.iloc[i - 1]["end_time"]
                prev_start = wdf.iloc[i - 1]["start_time"]
                if pd.notna(prev_end) and (start - prev_end) > 3.0:
                    flag_reasons["large_gap_over_3s"] += 1
                elif pd.notna(prev_start) and start < prev_start:
                    flag_reasons["non_monotonic"] += 1
                else:
                    flag_reasons["duration_out_of_range"] += 1

    # Word duration statistics
    dur_stats = {}
    if not words_df.empty:
        starts = pd.to_numeric(words_df["start_time"], errors="coerce")
        ends = pd.to_numeric(words_df["end_time"], errors="coerce")
        durations = ends - starts
        valid_dur = durations.dropna()
        if len(valid_dur) > 0:
            dur_stats = {
                "mean_s": round(float(valid_dur.mean()), 4),
                "median_s": round(float(valid_dur.median()), 4),
                "min_s": round(float(valid_dur.min()), 4),
                "max_s": round(float(valid_dur.max()), 4),
                "std_s": round(float(valid_dur.std()), 4),
            }

    # Inter-word gap statistics
    gap_stats = {}
    if len(words_df) > 1:
        starts = pd.to_numeric(words_df["start_time"], errors="coerce")
        ends = pd.to_numeric(words_df["end_time"], errors="coerce")
        gaps = starts.iloc[1:].values - ends.iloc[:-1].values
        valid_gaps = pd.Series(gaps).dropna()
        if len(valid_gaps) > 0:
            gap_stats = {
                "mean_s": round(float(valid_gaps.mean()), 4),
                "median_s": round(float(valid_gaps.median()), 4),
                "min_s": round(float(valid_gaps.min()), 4),
                "max_s": round(float(valid_gaps.max()), 4),
                "n_gaps_over_3s": int((valid_gaps > 3.0).sum()),
                "n_negative_gaps": int((valid_gaps < 0).sum()),
            }

    # Improved segment statistics
    seg_stats = {}
    if not improved_df.empty:
        seg_starts = pd.to_numeric(improved_df["start_time"], errors="coerce")
        seg_ends = pd.to_numeric(improved_df["end_time"], errors="coerce")
        seg_durations = seg_ends - seg_starts
        valid_seg = seg_durations.dropna()
        if len(valid_seg) > 0:
            seg_stats = {
                "mean_duration_s": round(float(valid_seg.mean()), 2),
                "min_duration_s": round(float(valid_seg.min()), 2),
                "max_duration_s": round(float(valid_seg.max()), 2),
            }
        if "n_words" in improved_df.columns:
            seg_stats["mean_n_words"] = round(float(improved_df["n_words"].mean()), 1)
            seg_stats["min_n_words"] = int(improved_df["n_words"].min())
            seg_stats["max_n_words"] = int(improved_df["n_words"].max())

    # Transcript time coverage
    time_coverage = {}
    if not words_df.empty:
        starts = pd.to_numeric(words_df["start_time"], errors="coerce")
        ends = pd.to_numeric(words_df["end_time"], errors="coerce")
        first_t = float(starts.min()) if starts.notna().any() else None
        last_t = float(ends.max()) if ends.notna().any() else None
        time_coverage = {
            "first_word_time": first_t,
            "last_word_time": last_t,
            "transcript_span_s": round(last_t - first_t, 2) if first_t is not None and last_t is not None else None,
        }

    # Sample of flagged words (first 20)
    flagged_samples = []
    if "timestamp_reliable" in words_df.columns:
        flagged = words_df[~words_df["timestamp_reliable"]].head(20)
        for _, row in flagged.iterrows():
            flagged_samples.append({
                "word_id": int(row["word_id"]),
                "word": str(row["word"]),
                "start_time": row["start_time"],
                "end_time": row["end_time"],
            })

    diag_data = {
        "asr_backend": asr_backend_used,
        "n_words_total": n_words,
        "n_words_reliable": n_reliable,
        "n_words_flagged": n_flagged,
        "pct_unreliable": round(pct_unreliable, 2),
        "flag_reasons": flag_reasons,
        "n_segments_raw_whisper": n_segments_raw,
        "n_segments_after_pysbd": n_segments_improved,
        "n_segments_merged_short": n_segments_raw - n_segments_improved,
        "word_duration_stats": dur_stats,
        "inter_word_gap_stats": gap_stats,
        "segment_stats": seg_stats,
        "time_coverage": time_coverage,
        "flagged_word_samples": flagged_samples,
    }

    diag.write_json("stage1_asr.json", diag_data)


def _write_diagnostics_from_cache(output_dir, diag):
    """Produce diagnostics from cached CSV files."""
    try:
        words_df = pd.read_csv(os.path.join(output_dir, "transcript_words.csv"))
        segments_df = pd.read_csv(os.path.join(output_dir, "transcript_segments.csv"))
        improved_df = pd.read_csv(os.path.join(output_dir, "transcript_segments_improved.csv"))
        _write_diagnostics(words_df, segments_df, improved_df, "cached", diag)
    except Exception as e:
        logger.warning("Could not produce diagnostics from cache: %s", e)
