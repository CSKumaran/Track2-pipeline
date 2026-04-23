"""Stage 2 – Scene detection via fixed-rate frame extraction + SSIM filtering.

Approach (designed for instructional / animated videos):
  1. Extract one frame every `sample_interval` seconds using ffmpeg.
  2. Compute SSIM between consecutive frames.
  3. When SSIM drops below a threshold → mark a scene boundary.
  4. Merge adjacent frames with SSIM above threshold into one scene.
  5. Filter out scenes shorter than `min_duration`.

This replaces PySceneDetect, which misses subtle slide / animation changes.
"""

from __future__ import annotations

import logging
import os
import subprocess

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

from pipeline.utils.ffmpeg_utils import extract_frame_at_time
from pipeline.utils.io_utils import ensure_dir, safe_write_csv

logger = logging.getLogger(__name__)

# ── OCR engine (lazy-loaded) ─────────────────────────────────────────
_ocr_reader = None
_ocr_available = None  # None = not yet checked


def _get_ocr_reader():
    """Lazy-load EasyOCR reader. Returns None if not available."""
    global _ocr_reader, _ocr_available
    if _ocr_available is not None:
        return _ocr_reader
    try:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        _ocr_available = True
        logger.info("EasyOCR loaded successfully (GPU=%s)", _ocr_reader.device)
    except ImportError:
        logger.warning("EasyOCR not installed – OCR features disabled. "
                       "Install with: pip install easyocr")
        _ocr_available = False
    except Exception as e:
        logger.warning("EasyOCR init failed (%s) – OCR features disabled.", e)
        _ocr_available = False
    return _ocr_reader


def _run_ocr_on_frame(frame_path: str, min_confidence: float = 0.3) -> str:
    """Run OCR on a single frame image. Returns extracted text or ''."""
    reader = _get_ocr_reader()
    if reader is None:
        return ""
    try:
        results = reader.readtext(frame_path)
        words = []
        for (bbox, text, conf) in results:
            if conf >= min_confidence and text.strip():
                words.append(text.strip())
        return " ".join(words)
    except Exception as e:
        logger.warning("OCR failed on %s: %s", frame_path, e)
        return ""


def _normalize_ocr_words(text) -> set[str]:
    """Normalize OCR text into a set of lowercase words (strip punctuation)."""
    import re
    if not text or not isinstance(text, str):
        return set()
    # lowercase, keep only alphanumeric and spaces
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    words = {w for w in cleaned.split() if len(w) >= 2}
    return words


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _extract_frames_fixed_rate(
    video_path: str, output_dir: str, interval: float
) -> list[tuple[float, str]]:
    """Extract frames at fixed intervals. Returns list of (timestamp, path)."""
    frames_dir = os.path.join(output_dir, "sampled_frames")
    ensure_dir(frames_dir)

    duration = _get_video_duration(video_path)
    # Stop 0.5s before end to avoid ffmpeg seek-past-end errors
    timestamps = np.arange(0, duration - 0.5, interval).tolist()

    frame_list: list[tuple[float, str]] = []
    for i, t in enumerate(timestamps):
        fp = os.path.join(frames_dir, f"frame_{i:05d}.jpg")
        extract_frame_at_time(video_path, t, fp)
        frame_list.append((round(t, 4), fp))

    logger.info("Extracted %d frames at %.1fs intervals (duration=%.1fs)",
                len(frame_list), interval, duration)
    return frame_list


def _compute_ssim_series(
    frame_list: list[tuple[float, str]],
) -> list[dict]:
    """Compute SSIM between consecutive frames.

    Returns a list of dicts with: idx, time, ssim_to_prev, frame_path.
    """
    records: list[dict] = []
    prev_gray = None

    for i, (t, fp) in enumerate(frame_list):
        img = cv2.imread(fp)
        if img is None:
            logger.warning("Could not read frame: %s", fp)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize for consistent SSIM computation (faster + resolution-independent)
        gray = cv2.resize(gray, (320, 240))

        if prev_gray is None:
            ssim_val = 0.0  # first frame always starts a new scene
        else:
            ssim_val = float(ssim(prev_gray, gray))

        records.append({
            "idx": i,
            "time": t,
            "ssim_to_prev": round(ssim_val, 4),
            "frame_path": fp,
        })
        prev_gray = gray

    return records


def _compute_ocr_series(
    frame_list: list[tuple[float, str]],
    min_confidence: float = 0.3,
) -> list[dict]:
    """Run OCR on each frame and compute new words vs previous frame.

    Returns a list of dicts with: idx, time, ocr_text, new_words.
    """
    records: list[dict] = []
    prev_words: set[str] = set()

    for i, (t, fp) in enumerate(frame_list):
        ocr_text = _run_ocr_on_frame(fp, min_confidence)
        current_words = _normalize_ocr_words(ocr_text)
        new_words = current_words - prev_words
        records.append({
            "idx": i,
            "time": t,
            "ocr_text": ocr_text,
            "new_words": " ".join(sorted(new_words)) if new_words else "",
            "n_new_words": len(new_words),
        })
        if current_words:
            prev_words = current_words

    return records


def _build_scenes_from_ssim(
    ssim_records: list[dict],
    ssim_threshold: float,
    min_duration: float,
    sample_interval: float,
    ocr_records: list[dict] | None = None,
    min_new_words: int = 1,
) -> list[dict]:
    """Group frames into scenes based on SSIM threshold OR OCR text changes.

    A new scene starts whenever:
      - ssim_to_prev < ssim_threshold, OR
      - n_new_words >= min_new_words (new OCR text detected)
    """
    if not ssim_records:
        return []

    # Build an index of OCR new-word counts keyed by frame idx
    ocr_new_counts: dict[int, int] = {}
    if ocr_records:
        for rec in ocr_records:
            ocr_new_counts[rec["idx"]] = rec.get("n_new_words", 0)

    # Find scene boundary indices
    boundaries = [0]  # first frame always starts scene 0
    for rec in ssim_records[1:]:
        is_ssim_change = rec["ssim_to_prev"] < ssim_threshold
        is_ocr_change = ocr_new_counts.get(rec["idx"], 0) >= min_new_words
        if is_ssim_change or is_ocr_change:
            boundaries.append(rec["idx"])

    # Build OCR lookup by frame index for fast access
    ocr_by_idx: dict[int, dict] = {}
    if ocr_records:
        for orec in ocr_records:
            ocr_by_idx[orec["idx"]] = orec

    # Build scene rows from boundary pairs
    scenes: list[dict] = []
    sid = 0
    prev_scene_words: set[str] = set()  # OCR words from previous scene's boundary
    for b_idx in range(len(boundaries)):
        start_rec = ssim_records[boundaries[b_idx]]
        if b_idx + 1 < len(boundaries):
            # Scene ends just before next boundary
            end_idx = boundaries[b_idx + 1] - 1
        else:
            # Last scene goes to end
            end_idx = len(ssim_records) - 1
        end_rec = ssim_records[end_idx]

        t_start = start_rec["time"]
        t_end = end_rec["time"] + sample_interval  # extend to cover last frame
        dur = t_end - t_start

        if dur < min_duration:
            continue

        # t_vis = timestamp of FIRST frame where new content appears
        # (the transition frame), not the scene midpoint
        t_vis = t_start

        # Collect new OCR words: compare this scene's boundary frame
        # against the PREVIOUS SCENE's boundary frame (not the previous
        # individual frame). This captures ALL words that appeared between
        # scenes, not just the single word that triggered the boundary.
        # Words are preserved in their original OCR reading order.
        new_ocr_words = ""
        if ocr_records:
            boundary_idx = start_rec["idx"]
            # Find OCR text at this boundary frame
            current_ocr = ""
            for orec in ocr_records:
                if orec["idx"] == boundary_idx:
                    val = orec.get("ocr_text", "")
                    current_ocr = str(val) if val and not (isinstance(val, float)) else ""
                    break
            current_words = _normalize_ocr_words(current_ocr)
            new_word_set = current_words - prev_scene_words
            # Preserve original OCR order: walk the raw text and keep
            # only words that are in the new_word_set
            if new_word_set:
                import re
                raw_tokens = current_ocr.split()
                ordered_new = []
                seen = set()
                for tok in raw_tokens:
                    norm = re.sub(r"[^a-z0-9]", "", tok.lower())
                    if norm in new_word_set and norm not in seen:
                        ordered_new.append(tok)  # keep original casing
                        seen.add(norm)
                new_ocr_words = " ".join(ordered_new)
            # Update previous scene's word set for next iteration
            if current_words:
                prev_scene_words = current_words

        scenes.append({
            "scene_id": sid,
            "t_start": round(t_start, 4),
            "t_end": round(t_end, 4),
            "duration": round(dur, 4),
            "t_vis": round(t_vis, 4),
            "start_frame_idx": start_rec["idx"],
            "end_frame_idx": end_idx,
            "new_ocr_words": new_ocr_words,
        })
        sid += 1

    return scenes


def run_scene_detection(
    video_path: str,
    output_dir: str,
    thresholds: list[float],
    min_duration: float,
    sample_interval: float = 2.0,
    ocr_enabled: bool = True,
    ocr_min_confidence: float = 0.3,
    min_new_words_for_scene: int = 1,
) -> pd.DataFrame:
    """Detect scenes using fixed-rate extraction + SSIM change detection + OCR.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    output_dir : str
        Per-video output directory.
    thresholds : list[float]
        SSIM thresholds to evaluate (e.g. [0.85, 0.90, 0.95]).
        Lower = fewer scenes (only big visual changes).
        Higher = more scenes (catches subtle changes).
    min_duration : float
        Minimum scene duration in seconds.
    sample_interval : float
        Seconds between extracted frames (default 2.0).
    ocr_enabled : bool
        Whether to run OCR on frames for text-change detection.
    ocr_min_confidence : float
        Minimum OCR confidence to accept a word.
    min_new_words_for_scene : int
        Minimum new OCR words to trigger a scene boundary.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns: threshold, scene_id, t_start,
        t_end, duration, t_vis.
    """
    ensure_dir(output_dir)

    # Step 1: Extract frames at fixed rate (done once, reused across thresholds)
    frame_list = _extract_frames_fixed_rate(video_path, output_dir, sample_interval)

    # Step 2: Compute SSIM series (done once)
    ssim_records = _compute_ssim_series(frame_list)

    # Save SSIM series for inspection
    ssim_df = pd.DataFrame(ssim_records)
    safe_write_csv(ssim_df[["idx", "time", "ssim_to_prev"]],
                   os.path.join(output_dir, "ssim_series.csv"))

    # Step 2b: Run OCR on each frame (done once, cached)
    ocr_records: list[dict] | None = None
    ocr_csv_path = os.path.join(output_dir, "ocr_per_frame.csv")
    if ocr_enabled:
        # Check for cached OCR results
        if os.path.isfile(ocr_csv_path):
            cached_ocr_df = pd.read_csv(ocr_csv_path)
            if len(cached_ocr_df) == len(frame_list):
                logger.info("Reusing cached OCR results: %s (%d frames)",
                            ocr_csv_path, len(cached_ocr_df))
                # Reconstruct n_new_words from new_words text column
                if "n_new_words" not in cached_ocr_df.columns:
                    cached_ocr_df["n_new_words"] = cached_ocr_df["new_words"].fillna("").apply(
                        lambda x: len(str(x).split()) if str(x).strip() else 0
                    )
                ocr_records = cached_ocr_df.to_dict("records")
            else:
                logger.info("Cached OCR frame count mismatch, re-running OCR.")

        if ocr_records is None:
            if _get_ocr_reader() is not None:
                logger.info("Running OCR on %d frames...", len(frame_list))
                ocr_records = _compute_ocr_series(frame_list, ocr_min_confidence)
                # Save OCR results
                ocr_df = pd.DataFrame(ocr_records)
                safe_write_csv(
                    ocr_df[["idx", "time", "ocr_text", "new_words", "n_new_words"]],
                    ocr_csv_path,
                )
                n_with_text = sum(1 for r in ocr_records if r["ocr_text"])
                n_with_new = sum(1 for r in ocr_records if r["n_new_words"] > 0)
                logger.info("OCR complete: %d/%d frames have text, "
                            "%d frames have new words",
                            n_with_text, len(ocr_records), n_with_new)
            else:
                logger.warning("OCR enabled but EasyOCR not available. "
                               "Proceeding with SSIM-only scene detection.")

    # Step 3: Build scenes at each SSIM threshold (+ OCR transitions)
    all_scenes: list[pd.DataFrame] = []
    stats_rows: list[dict] = []

    for T in thresholds:
        logger.info("Building scenes with SSIM threshold=%.2f", T)
        scene_rows = _build_scenes_from_ssim(
            ssim_records, T, min_duration, sample_interval,
            ocr_records=ocr_records,
            min_new_words=min_new_words_for_scene,
        )

        # Add threshold column
        for row in scene_rows:
            row["threshold"] = T

        df = pd.DataFrame(scene_rows)

        # Save per-threshold CSV
        csv_path = os.path.join(output_dir, f"scenes_threshold_{T}.csv")
        safe_write_csv(df, csv_path)

        # Extract representative frame at t_vis for each scene
        frames_dir = os.path.join(output_dir, f"frames_threshold_{T}")
        ensure_dir(frames_dir)
        for _, row in df.iterrows():
            frame_path = os.path.join(
                frames_dir, f"scene_{int(row['scene_id'])}.jpg"
            )
            extract_frame_at_time(video_path, row["t_vis"], frame_path)

        all_scenes.append(df)

        n_ssim_boundaries = len([r for r in ssim_records if r["ssim_to_prev"] < T])
        n_ocr_boundaries = 0
        if ocr_records:
            n_ocr_boundaries = len([r for r in ocr_records
                                    if r["n_new_words"] >= min_new_words_for_scene])
        stats_rows.append({
            "threshold": T,
            "n_scenes": len(df),
            "mean_duration": round(df["duration"].mean(), 2) if len(df) else 0,
            "median_duration": round(df["duration"].median(), 2) if len(df) else 0,
            "n_ssim_boundaries": n_ssim_boundaries,
            "n_ocr_boundaries": n_ocr_boundaries,
            "n_filtered_short": max(0, n_ssim_boundaries + n_ocr_boundaries - len(df)),
        })
        logger.info("SSIM threshold %.2f: %d scenes "
                     "(SSIM boundaries=%d, OCR boundaries=%d)",
                     T, len(df), n_ssim_boundaries, n_ocr_boundaries)

    # Save stats
    stats_df = pd.DataFrame(stats_rows)
    safe_write_csv(stats_df, os.path.join(output_dir, "scene_stats.csv"))

    combined = pd.concat(all_scenes, ignore_index=True) if all_scenes else pd.DataFrame()

    # Guarantee expected columns even when empty
    expected_cols = ["scene_id", "t_start", "t_end", "duration", "t_vis", "threshold"]
    if combined.empty:
        combined = pd.DataFrame(columns=expected_cols)

    return combined
