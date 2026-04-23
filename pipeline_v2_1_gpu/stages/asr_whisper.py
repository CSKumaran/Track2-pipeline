"""Stage 1: ASR Transcription (WhisperX + pySBD + sanity checker)."""

import logging
import os
import pandas as pd
import pysbd

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


def run_stage1(video_path: str, output_dir: str, cfg: Config) -> dict:
    """Run ASR transcription. Returns dict with paths to output CSVs."""
    if cache_exists(output_dir, CACHE_FILES):
        logger.info("Stage 1: cache hit, skipping ASR")
        return {f: os.path.join(output_dir, f) for f in CACHE_FILES}

    wav_path = extract_audio(video_path, output_dir)

    if cfg.ASR_BACKEND == "whisperx":
        try:
            words_df, segments_df = _run_whisperx(wav_path, cfg)
            logger.info("ASR backend: whisperx")
        except Exception as e:
            logger.warning("WhisperX failed (%s), falling back to faster-whisper", e)
            words_df, segments_df = _run_faster_whisper(wav_path, cfg)
            logger.info("ASR backend: faster-whisper (fallback)")
    else:
        words_df, segments_df = _run_faster_whisper(wav_path, cfg)
        logger.info("ASR backend: faster-whisper")

    # pySBD re-segmentation
    improved_df = _resegment_pysbd(words_df, segments_df)

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

    return paths


def _run_whisperx(wav_path: str, cfg: Config) -> tuple:
    """Run WhisperX with forced alignment."""
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ctranslate2 on CPU doesn't support float16; auto-select
    if device == "cpu":
        compute_type = "int8"
    else:
        compute_type = cfg.WHISPER_COMPUTE_TYPE
    logger.info("Loading WhisperX model: %s (device=%s, compute=%s)", cfg.WHISPER_MODEL, device, compute_type)
    model = whisperx.load_model(
        cfg.WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
    )

    logger.info("Transcribing...")
    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, batch_size=cfg.WHISPER_BATCH_SIZE)

    # Forced alignment
    logger.info("Running forced alignment...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result.get("language", "en"), device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False
    )

    # Free memory
    del model, align_model
    import torch
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


def _resegment_pysbd(words_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    """Re-segment transcript using pySBD for better sentence boundaries."""
    if words_df.empty:
        return segments_df.copy()

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
            else:
                merged.append(row.to_dict())

        improved_df = pd.DataFrame(merged)
        improved_df["segment_id"] = range(len(improved_df))

    return improved_df
