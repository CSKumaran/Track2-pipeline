"""Stage 2: Scene Detection (Multi-Signal Fusion)."""

import logging
import os
import numpy as np
import pandas as pd
from PIL import Image

from ..config import Config
from ..utils.ffmpeg_utils import extract_frames
from ..utils.dinov2_utils import embed_frames, compute_consecutive_distances, find_centroid_frame, unload_dinov2
from ..utils.ocr_utils import run_ocr, extract_words_from_ocr, jaccard_distance
from ..utils.io_utils import save_csv, cache_exists

logger = logging.getLogger(__name__)


def run_stage2(video_path: str, output_dir: str, cfg: Config) -> dict:
    """Run scene detection. Returns dict with paths to outputs."""
    cache_files = ["scenes.csv", "ocr_per_frame.csv", "dinov2_distances.csv"]
    if cache_exists(output_dir, cache_files):
        logger.info("Stage 2: cache hit, skipping scene detection")
        return {f: os.path.join(output_dir, f) for f in cache_files}

    # Extract frames
    frame_data = extract_frames(video_path, output_dir, cfg.SAMPLE_INTERVAL)
    times = [t for t, _ in frame_data]
    paths = [p for _, p in frame_data]

    if not paths:
        logger.error("No frames extracted!")
        return {}

    logger.info("Stage 2: Processing %d frames", len(paths))

    # Signal A: PySceneDetect
    signal_a = _pyscenedetect_signal(video_path, times)

    # Signal B: DINOv2 embedding distances
    logger.info("Computing DINOv2 embeddings...")
    dinov2_embeddings = embed_frames(paths, cfg.DINOV2_MODEL, cfg.DINOV2_BATCH_SIZE)
    dinov2_dists = compute_consecutive_distances(dinov2_embeddings)
    # Pad to match frame count (first frame has no predecessor)
    signal_b = np.concatenate([[0.0], dinov2_dists])
    # Normalize
    if signal_b.max() > 0:
        signal_b = signal_b / signal_b.max()

    # Signal C: OCR Jaccard distance (every 3rd frame on GPU, every 5th on CPU)
    ocr_sample_step = max(1, int(1.5 / cfg.SAMPLE_INTERVAL))  # every ~1.5s on GPU
    logger.info("Running OCR on every %dth frame (%d frames)...",
                ocr_sample_step, len(range(0, len(paths), ocr_sample_step)))
    signal_c, ocr_per_frame = _ocr_jaccard_signal(paths, times, cfg, sample_step=ocr_sample_step)

    # Save OCR per frame
    ocr_df = pd.DataFrame(ocr_per_frame)
    save_csv(ocr_df, os.path.join(output_dir, "ocr_per_frame.csv"))

    # Save DINOv2 distances
    dinov2_df = pd.DataFrame({
        "frame_time": times[:len(signal_b)],
        "distance": signal_b[:len(times)],
    })
    save_csv(dinov2_df, os.path.join(output_dir, "dinov2_distances.csv"))

    # Fuse signals: 2-signal if OCR all zeros, 3-signal otherwise
    n = min(len(signal_a), len(signal_b), len(signal_c))
    if signal_c[:n].max() > 0:
        combined = (
            cfg.SCENE_SIGNAL_W1 * signal_a[:n]
            + cfg.SCENE_SIGNAL_W2 * signal_b[:n]
            + cfg.SCENE_SIGNAL_W3 * signal_c[:n]
        )
    else:
        # No OCR signal, reweight A and B
        w_total = cfg.SCENE_SIGNAL_W1 + cfg.SCENE_SIGNAL_W2
        combined = (
            (cfg.SCENE_SIGNAL_W1 / w_total) * signal_a[:n]
            + (cfg.SCENE_SIGNAL_W2 / w_total) * signal_b[:n]
        )

    # Adaptive threshold
    threshold = _adaptive_threshold(combined, cfg.SCENE_THRESHOLD_K)
    logger.info("Adaptive threshold: %.4f", threshold)

    # Find boundaries
    boundaries = _find_boundaries(combined, threshold, times[:n])

    # Minimum scene count guard
    from ..utils.ffmpeg_utils import get_video_duration
    duration = get_video_duration(video_path)
    if len(boundaries) < 3 and duration > 60:
        lower_threshold = threshold - 0.5 * np.std(combined)
        boundaries = _find_boundaries(combined, lower_threshold, times[:n])
        logger.info("Lowered threshold to %.4f, got %d boundaries", lower_threshold, len(boundaries))

    # Build scenes
    scenes = _build_scenes(boundaries, times[:n], paths[:n], dinov2_embeddings[:n], cfg, output_dir)

    # Save scenes
    scenes_df = pd.DataFrame(scenes)
    save_csv(scenes_df, os.path.join(output_dir, "scenes.csv"))

    # Unload DINOv2 to free memory
    unload_dinov2()

    logger.info("Stage 2: %d scenes detected", len(scenes))
    return {f: os.path.join(output_dir, f) for f in cache_files}


def _pyscenedetect_signal(video_path: str, times: list) -> np.ndarray:
    """Signal A: PySceneDetect AdaptiveDetector scores."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import AdaptiveDetector

        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(AdaptiveDetector(window_width=5))
        manager.detect_scenes(video)
        scene_list = manager.get_scene_list()

        # Convert scene boundaries to per-frame signal
        signal = np.zeros(len(times))
        for scene_start, scene_end in scene_list:
            t = scene_start.get_seconds()
            # Find closest frame
            idx = np.argmin(np.abs(np.array(times) - t))
            signal[idx] = 1.0

        return signal
    except Exception as e:
        logger.warning("PySceneDetect failed: %s, using zeros", e)
        return np.zeros(len(times))


def _ocr_jaccard_signal(paths: list, times: list, cfg: Config,
                        sample_step: int = 1) -> tuple:
    """Signal C: OCR Jaccard distance between sampled frames."""
    signal = np.zeros(len(paths))
    ocr_records = []
    prev_words = set()
    prev_sampled_idx = None

    for i, (t, p) in enumerate(zip(times, paths)):
        # Only run OCR on sampled frames
        if i % sample_step != 0 and not cfg.OCR_ENABLED:
            ocr_records.append({
                "frame_time": t, "words": "", "n_words": 0, "mean_confidence": 0,
            })
            continue

        if cfg.OCR_ENABLED and i % sample_step == 0:
            ocr_results = run_ocr(p, cfg.OCR_ENGINE, cfg.OCR_MIN_CONFIDENCE)
            words = extract_words_from_ocr(ocr_results)
            ocr_text = " ".join(sorted(words))
            confidences = [c for _, c, _ in ocr_results]
        else:
            words = set()
            ocr_text = ""
            confidences = []

        ocr_records.append({
            "frame_time": t,
            "words": ocr_text,
            "n_words": len(words),
            "mean_confidence": np.mean(confidences) if confidences else 0,
        })

        if prev_sampled_idx is not None:
            jd = jaccard_distance(prev_words, words)
            if jd > cfg.OCR_JACCARD_THRESHOLD:
                # Spread signal across skipped frames
                for j in range(prev_sampled_idx + 1, i + 1):
                    if j < len(signal):
                        signal[j] = jd

        if i % sample_step == 0:
            prev_words = words
            prev_sampled_idx = i

    # Normalize
    if signal.max() > 0:
        signal = signal / signal.max()

    return signal, ocr_records


def _adaptive_threshold(scores: np.ndarray, k: float) -> float:
    """Compute adaptive threshold: mean + k * std, or Otsu-like."""
    if len(scores) == 0:
        return 0.5

    # Try Otsu-like: maximize between-class variance
    try:
        sorted_scores = np.sort(scores)
        best_threshold = sorted_scores[len(sorted_scores) // 2]
        best_variance = 0

        for t_candidate in np.linspace(scores.min(), scores.max(), 50):
            below = scores[scores <= t_candidate]
            above = scores[scores > t_candidate]
            if len(below) == 0 or len(above) == 0:
                continue
            w0 = len(below) / len(scores)
            w1 = len(above) / len(scores)
            variance = w0 * w1 * (below.mean() - above.mean()) ** 2
            if variance > best_variance:
                best_variance = variance
                best_threshold = t_candidate

        if best_variance > 0:
            return float(best_threshold)
    except Exception:
        pass

    # Fallback: mean + k * std
    return float(np.mean(scores) + k * np.std(scores))


def _find_boundaries(scores: np.ndarray, threshold: float, times: list) -> list:
    """Find scene boundaries via thresholding + NMS."""
    candidates = []
    for i in range(len(scores)):
        if scores[i] >= threshold:
            candidates.append((i, scores[i], times[i]))

    if not candidates:
        return []

    # Non-maximum suppression: keep peaks only
    nms_boundaries = []
    group = [candidates[0]]
    for c in candidates[1:]:
        if c[0] - group[-1][0] <= 2:  # adjacent frames
            group.append(c)
        else:
            # Pick best in group
            best = max(group, key=lambda x: x[1])
            nms_boundaries.append(best)
            group = [c]
    best = max(group, key=lambda x: x[1])
    nms_boundaries.append(best)

    return nms_boundaries


def _build_scenes(boundaries: list, times: list, paths: list,
                  embeddings: np.ndarray, cfg: Config, output_dir: str) -> list:
    """Build scene list from boundaries. Select keyframes via DINOv2 centroid."""
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Add start and end as implicit boundaries
    scene_starts = [0] + [b[0] for b in boundaries]
    scene_ends = [b[0] for b in boundaries] + [len(times) - 1]

    scenes = []
    for sid, (s_idx, e_idx) in enumerate(zip(scene_starts, scene_ends)):
        if s_idx >= len(times) or e_idx >= len(times):
            continue

        t_start = times[s_idx]
        t_end = times[e_idx]
        duration = t_end - t_start

        # Scene embeddings
        scene_emb = embeddings[s_idx:e_idx + 1]
        n_frames = len(scene_emb)

        # Keyframe selection: centroid, avoiding first/last 10%
        if n_frames >= 3:
            margin = max(1, int(n_frames * 0.1))
            inner_emb = scene_emb[margin:-margin] if margin < n_frames // 2 else scene_emb
            inner_offset = margin if margin < n_frames // 2 else 0
            kf_local = find_centroid_frame(inner_emb)
            kf_idx = s_idx + inner_offset + kf_local
        else:
            kf_idx = s_idx + n_frames // 2

        kf_idx = min(kf_idx, len(times) - 1)
        t_keyframe = times[kf_idx]

        # Copy keyframe
        src_path = paths[kf_idx]
        dst_path = os.path.join(frames_dir, f"scene_{sid:03d}.jpg")
        try:
            import shutil
            shutil.copy2(src_path, dst_path)
        except Exception:
            dst_path = src_path

        # Get boundary score
        boundary_score = 0.0
        for b in boundaries:
            if b[0] == s_idx or (sid > 0 and b[0] == scene_starts[sid]):
                boundary_score = b[1]
                break

        # OCR words from keyframe
        if cfg.OCR_ENABLED:
            from ..utils.ocr_utils import run_ocr, extract_words_from_ocr
            ocr_results = run_ocr(paths[kf_idx], cfg.OCR_ENGINE, cfg.OCR_MIN_CONFIDENCE)
            ocr_words = extract_words_from_ocr(ocr_results)
        else:
            ocr_words = set()

        scenes.append({
            "scene_id": sid,
            "t_start": t_start,
            "t_end": t_end,
            "duration": duration,
            "t_keyframe": t_keyframe,
            "keyframe_path": dst_path,
            "detection_signal": "multi_fusion",
            "boundary_score": boundary_score,
            "ocr_words": " ".join(sorted(ocr_words)),
            "n_ocr_words": len(ocr_words),
        })

    # Smart merge: merge short scenes into predecessor if visually similar
    if len(scenes) > 1:
        scenes = _smart_merge(scenes, embeddings, times, cfg)

    return scenes


def _smart_merge(scenes: list, embeddings: np.ndarray, times: list, cfg: Config) -> list:
    """Merge scenes shorter than MIN_SCENE_MERGE_DURATION if visually similar to predecessor."""
    from sklearn.metrics.pairwise import cosine_similarity

    merged = [scenes[0]]
    for scene in scenes[1:]:
        if scene["duration"] < cfg.MIN_SCENE_MERGE_DURATION and merged:
            # Check DINOv2 similarity with predecessor
            prev = merged[-1]
            prev_kf_idx = np.argmin(np.abs(np.array(times) - prev["t_keyframe"]))
            curr_kf_idx = np.argmin(np.abs(np.array(times) - scene["t_keyframe"]))
            if prev_kf_idx < len(embeddings) and curr_kf_idx < len(embeddings):
                sim = cosine_similarity(
                    embeddings[prev_kf_idx:prev_kf_idx + 1],
                    embeddings[curr_kf_idx:curr_kf_idx + 1]
                )[0, 0]
                if sim > cfg.MERGE_SIMILARITY_THRESHOLD:
                    # Merge into predecessor
                    merged[-1]["t_end"] = scene["t_end"]
                    merged[-1]["duration"] = merged[-1]["t_end"] - merged[-1]["t_start"]
                    continue
        merged.append(scene)

    # Re-number
    for i, s in enumerate(merged):
        s["scene_id"] = i

    return merged
