"""Stage 2: Scene Detection (Multi-Signal Fusion) [v2.2].

V2.2 improvements over V2.1:
- P2 FIX: Density-aware threshold guard (EXPECTED_SCENES_PER_MINUTE)
- P2 FIX: Force-split scenes exceeding MAX_SCENE_DURATION
- P2 FIX: Lowered fallback threshold (SCENE_THRESHOLD_K_FALLBACK = 1.0)
- P3 FIX: OCR at scene start + keyframe (OCR_SAMPLE_SCENE_START)
- Full diagnostics output
"""

import logging
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image

from ..config import Config
from ..utils.ffmpeg_utils import extract_frames, get_video_duration
from ..utils.dinov2_utils import (
    embed_frames, compute_consecutive_distances,
    find_centroid_frame, unload_dinov2,
)
from ..utils.ocr_utils import run_ocr, extract_words_from_ocr, jaccard_distance
from ..utils.io_utils import save_csv, cache_exists

logger = logging.getLogger(__name__)

CACHE_FILES = ["scenes.csv", "ocr_per_frame.csv", "dinov2_distances.csv"]


def run_stage2(video_path: str, output_dir: str, cfg: Config, diag=None) -> dict:
    """Run scene detection. Returns dict with paths to output CSVs."""
    if cache_exists(output_dir, CACHE_FILES):
        logger.info("Stage 2: cache hit, skipping scene detection")
        paths = {f: os.path.join(output_dir, f) for f in CACHE_FILES}
        if diag is not None:
            _write_diagnostics_from_cache(output_dir, diag)
        return paths

    # Extract frames
    frame_data = extract_frames(video_path, output_dir, cfg.SAMPLE_INTERVAL)
    times = [t for t, _ in frame_data]
    paths_list = [p for _, p in frame_data]

    if not paths_list:
        logger.error("No frames extracted!")
        return {}

    duration = get_video_duration(video_path)
    logger.info("Stage 2: %d frames over %.1fs", len(paths_list), duration)

    # =========================================================
    # Signal A: PySceneDetect
    # =========================================================
    signal_a = _pyscenedetect_signal(video_path, times)

    # =========================================================
    # Signal B: DINOv2 embedding distances
    # =========================================================
    logger.info("Computing DINOv2 embeddings (batch=%d)...", cfg.DINOV2_BATCH_SIZE)
    dinov2_embeddings = embed_frames(paths_list, cfg.DINOV2_MODEL, cfg.DINOV2_BATCH_SIZE)
    dinov2_dists = compute_consecutive_distances(dinov2_embeddings)
    signal_b = np.concatenate([[0.0], dinov2_dists])
    signal_b_raw = signal_b.copy()
    if signal_b.max() > 0:
        signal_b = signal_b / signal_b.max()

    # =========================================================
    # Signal C: OCR Jaccard distance
    # =========================================================
    # [v3] OCR every frame (0.5s) for sub-scene temporal resolution
    if getattr(cfg, "OCR_SAMPLE_EVERY_FRAME", False):
        ocr_sample_step = 1
    else:
        ocr_sample_step = max(1, int(1.5 / cfg.SAMPLE_INTERVAL))
    logger.info("Running OCR every %dth frame (%d frames)...",
                ocr_sample_step, len(range(0, len(paths_list), ocr_sample_step)))
    signal_c, ocr_per_frame = _ocr_jaccard_signal(
        paths_list, times, cfg, sample_step=ocr_sample_step
    )

    # Save OCR per frame
    ocr_df = pd.DataFrame(ocr_per_frame)
    save_csv(ocr_df, os.path.join(output_dir, "ocr_per_frame.csv"))

    # Save DINOv2 distances
    dinov2_df = pd.DataFrame({
        "frame_time": times[:len(signal_b)],
        "distance": signal_b_raw[:len(times)],
        "distance_normalized": signal_b[:len(times)],
    })
    save_csv(dinov2_df, os.path.join(output_dir, "dinov2_distances.csv"))

    # =========================================================
    # Fuse signals
    # =========================================================
    n = min(len(signal_a), len(signal_b), len(signal_c))
    ocr_has_signal = signal_c[:n].max() > 0
    if ocr_has_signal:
        combined = (
            cfg.SCENE_SIGNAL_W1 * signal_a[:n]
            + cfg.SCENE_SIGNAL_W2 * signal_b[:n]
            + cfg.SCENE_SIGNAL_W3 * signal_c[:n]
        )
        fusion_mode = "3-signal"
    else:
        w_total = cfg.SCENE_SIGNAL_W1 + cfg.SCENE_SIGNAL_W2
        combined = (
            (cfg.SCENE_SIGNAL_W1 / w_total) * signal_a[:n]
            + (cfg.SCENE_SIGNAL_W2 / w_total) * signal_b[:n]
        )
        fusion_mode = "2-signal (no OCR)"
    logger.info("Signal fusion: %s", fusion_mode)

    # =========================================================
    # Adaptive threshold
    # =========================================================
    threshold = _adaptive_threshold(combined, cfg.SCENE_THRESHOLD_K)
    threshold_method = "otsu"

    # Find boundaries
    boundaries = _find_boundaries(combined, threshold, times[:n])
    n_boundaries_initial = len(boundaries)

    # [v2.2] Density-aware threshold guard
    expected_scenes = cfg.EXPECTED_SCENES_PER_MINUTE * (duration / 60.0)
    threshold_adjustments = []

    if len(boundaries) + 1 < expected_scenes * 0.5:
        # Too few scenes — progressively lower threshold
        for factor in [0.75, 0.5, 0.25]:
            new_threshold = threshold * factor
            new_boundaries = _find_boundaries(combined, new_threshold, times[:n])
            threshold_adjustments.append({
                "factor": factor,
                "threshold": round(float(new_threshold), 4),
                "n_boundaries": len(new_boundaries),
            })
            # Check if max scene duration is acceptable
            test_scenes = _build_scenes_quick(new_boundaries, times[:n])
            max_dur = max((s[1] - s[0] for s in test_scenes), default=0)
            if max_dur <= cfg.MAX_SCENE_DURATION or len(new_boundaries) >= expected_scenes * 0.8:
                boundaries = new_boundaries
                threshold = new_threshold
                threshold_method = f"density-guard (factor={factor})"
                logger.info("Density guard: lowered threshold to %.4f (factor=%.2f), %d boundaries",
                            threshold, factor, len(boundaries))
                break
        else:
            # Use lowest threshold if nothing else worked
            if threshold_adjustments:
                best = threshold_adjustments[-1]
                boundaries = _find_boundaries(combined, best["threshold"], times[:n])
                threshold = best["threshold"]
                threshold_method = "density-guard (forced)"
                logger.info("Density guard: forced threshold %.4f, %d boundaries",
                            threshold, len(boundaries))

    # Legacy fallback guard
    elif len(boundaries) < 3 and duration > 60:
        fallback_threshold = float(np.mean(combined) + cfg.SCENE_THRESHOLD_K_FALLBACK * np.std(combined))
        if fallback_threshold < threshold:
            boundaries = _find_boundaries(combined, fallback_threshold, times[:n])
            threshold = fallback_threshold
            threshold_method = "fallback (mean + k_fallback*std)"
            logger.info("Fallback threshold: %.4f, %d boundaries", threshold, len(boundaries))

    n_boundaries_after_guard = len(boundaries)

    # [v2.2] Scene detection mode flag for sensitivity analysis
    if threshold_method == "otsu":
        scene_detection_mode = "natural"
    elif "density-guard" in threshold_method:
        scene_detection_mode = "density-forced"
    else:
        scene_detection_mode = "fallback"

    # [v2.2] Pathological config guard
    if expected_scenes < 3:
        logger.warning("Expected scenes (%.1f) < 3 — config may be inconsistent with video length (%.1fs)",
                        expected_scenes, duration)
    max_possible_scenes = duration / cfg.MIN_SCENE_MERGE_DURATION if cfg.MIN_SCENE_MERGE_DURATION > 0 else 999
    if expected_scenes > max_possible_scenes:
        logger.warning("Expected scenes (%.1f) > max possible (%.1f) — EXPECTED_SCENES_PER_MINUTE too high",
                        expected_scenes, max_possible_scenes)

    logger.info("Threshold: %.4f (%s), %d boundaries [mode: %s]",
                threshold, threshold_method, len(boundaries), scene_detection_mode)

    # =========================================================
    # Build scenes
    # =========================================================
    scenes = _build_scenes(
        boundaries, times[:n], paths_list[:n],
        dinov2_embeddings[:n], cfg, output_dir
    )
    n_scenes_before_merge = len(scenes)

    # Smart merge short scenes
    if len(scenes) > 1:
        scenes = _smart_merge(scenes, dinov2_embeddings, times, cfg)
    n_scenes_after_merge = len(scenes)
    n_merged = n_scenes_before_merge - n_scenes_after_merge

    # [v2.2] Force-split long scenes (now using DINOv2 + OCR combined signal)
    n_force_splits = 0
    if cfg.MAX_SCENE_DURATION > 0:
        scenes, n_force_splits = _force_split_long_scenes(
            scenes, times[:n], paths_list[:n], dinov2_embeddings[:n],
            signal_b[:n], cfg, output_dir, signal_c=signal_c[:n]
        )
    n_scenes_final = len(scenes)

    # [v2.2] OCR at scene start + keyframe
    if cfg.OCR_SAMPLE_SCENE_START:
        _add_scene_start_ocr(scenes, times[:n], paths_list[:n], cfg)

    # Save scenes
    scenes_df = pd.DataFrame(scenes)
    save_csv(scenes_df, os.path.join(output_dir, "scenes.csv"))

    # Unload DINOv2
    unload_dinov2()

    logger.info("Stage 2: %d scenes (merged %d, split %d)", n_scenes_final, n_merged, n_force_splits)

    # =========================================================
    # Diagnostics [v2.2]
    # =========================================================
    if diag is not None:
        _write_diagnostics(
            scenes, times[:n], signal_a[:n], signal_b[:n], signal_c[:n],
            combined, threshold, threshold_method, fusion_mode,
            duration, n_boundaries_initial, n_boundaries_after_guard,
            n_scenes_before_merge, n_merged, n_force_splits, n_scenes_final,
            threshold_adjustments, ocr_has_signal, cfg, diag,
            scene_detection_mode=scene_detection_mode,
        )

    return {f: os.path.join(output_dir, f) for f in CACHE_FILES}


# =====================================================================
# Signal Computation
# =====================================================================

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

        signal = np.zeros(len(times))
        for scene_start, _ in scene_list:
            t = scene_start.get_seconds()
            idx = np.argmin(np.abs(np.array(times) - t))
            signal[idx] = 1.0

        logger.info("PySceneDetect: %d boundaries", len(scene_list))
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
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        })

        if prev_sampled_idx is not None and i % sample_step == 0:
            jd = jaccard_distance(prev_words, words)
            if jd > cfg.OCR_JACCARD_THRESHOLD:
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


# =====================================================================
# Thresholding & Boundaries
# =====================================================================

def _adaptive_threshold(scores: np.ndarray, k: float) -> float:
    """Otsu-like threshold, fallback to mean + k*std."""
    if len(scores) == 0:
        return 0.5

    try:
        best_threshold = float(np.median(scores))
        best_variance = 0.0

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
                best_threshold = float(t_candidate)

        if best_variance > 0:
            return best_threshold
    except Exception:
        pass

    return float(np.mean(scores) + k * np.std(scores))


def _find_boundaries(scores: np.ndarray, threshold: float, times: list) -> list:
    """Find boundaries via thresholding + NMS."""
    candidates = []
    for i in range(len(scores)):
        if scores[i] >= threshold:
            candidates.append((i, float(scores[i]), times[i]))

    if not candidates:
        return []

    # NMS: group adjacent frames, keep peak
    nms_boundaries = []
    group = [candidates[0]]
    for c in candidates[1:]:
        if c[0] - group[-1][0] <= 2:
            group.append(c)
        else:
            best = max(group, key=lambda x: x[1])
            nms_boundaries.append(best)
            group = [c]
    best = max(group, key=lambda x: x[1])
    nms_boundaries.append(best)

    return nms_boundaries


def _build_scenes_quick(boundaries, times):
    """Quick scene interval computation (for threshold guard checks)."""
    scene_starts = [0.0] + [b[2] for b in boundaries]
    scene_ends = [b[2] for b in boundaries] + [times[-1] if times else 0.0]
    return list(zip(scene_starts, scene_ends))


# =====================================================================
# Scene Construction
# =====================================================================

def _build_scenes(boundaries: list, times: list, paths: list,
                  embeddings: np.ndarray, cfg: Config, output_dir: str) -> list:
    """Build scenes with keyframe selection via DINOv2 centroid."""
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    scene_starts = [0] + [b[0] for b in boundaries]
    scene_ends = [b[0] for b in boundaries] + [len(times) - 1]

    scenes = []
    for sid, (s_idx, e_idx) in enumerate(zip(scene_starts, scene_ends)):
        if s_idx >= len(times) or e_idx >= len(times):
            continue

        t_start = times[s_idx]
        t_end = times[e_idx]
        scene_duration = t_end - t_start

        scene_emb = embeddings[s_idx:e_idx + 1]
        n_frames = len(scene_emb)

        # Keyframe: centroid of inner 80%
        if n_frames >= 3:
            margin = max(1, int(n_frames * 0.1))
            if margin < n_frames // 2:
                inner_emb = scene_emb[margin:-margin]
                inner_offset = margin
            else:
                inner_emb = scene_emb
                inner_offset = 0
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
            shutil.copy2(src_path, dst_path)
        except Exception:
            dst_path = src_path

        # OCR on keyframe
        if cfg.OCR_ENABLED:
            ocr_results = run_ocr(paths[kf_idx], cfg.OCR_ENGINE, cfg.OCR_MIN_CONFIDENCE)
            ocr_words = extract_words_from_ocr(ocr_results)
        else:
            ocr_words = set()

        scenes.append({
            "scene_id": sid,
            "t_start": round(t_start, 3),
            "t_end": round(t_end, 3),
            "duration": round(scene_duration, 3),
            "t_keyframe": round(t_keyframe, 3),
            "keyframe_path": dst_path,
            "n_frames": n_frames,
            "ocr_words": " ".join(sorted(ocr_words)),
            "n_ocr_words": len(ocr_words),
        })

    return scenes


# =====================================================================
# Smart Merge [v2.1]
# =====================================================================

def _smart_merge(scenes: list, embeddings: np.ndarray,
                 times: list, cfg: Config) -> list:
    """Merge scenes < MIN_SCENE_MERGE_DURATION if DINOv2-similar to predecessor."""
    from sklearn.metrics.pairwise import cosine_similarity

    merged = [scenes[0]]
    for scene in scenes[1:]:
        if scene["duration"] < cfg.MIN_SCENE_MERGE_DURATION and merged:
            prev = merged[-1]
            prev_kf_idx = min(np.argmin(np.abs(np.array(times) - prev["t_keyframe"])), len(embeddings) - 1)
            curr_kf_idx = min(np.argmin(np.abs(np.array(times) - scene["t_keyframe"])), len(embeddings) - 1)
            if prev_kf_idx < len(embeddings) and curr_kf_idx < len(embeddings):
                sim = cosine_similarity(
                    embeddings[prev_kf_idx:prev_kf_idx + 1],
                    embeddings[curr_kf_idx:curr_kf_idx + 1]
                )[0, 0]
                if sim > cfg.MERGE_SIMILARITY_THRESHOLD:
                    merged[-1]["t_end"] = scene["t_end"]
                    merged[-1]["duration"] = round(
                        merged[-1]["t_end"] - merged[-1]["t_start"], 3
                    )
                    # Union OCR words
                    prev_ocr = set(merged[-1]["ocr_words"].split()) if merged[-1]["ocr_words"] else set()
                    curr_ocr = set(scene["ocr_words"].split()) if scene["ocr_words"] else set()
                    all_ocr = prev_ocr | curr_ocr
                    merged[-1]["ocr_words"] = " ".join(sorted(all_ocr))
                    merged[-1]["n_ocr_words"] = len(all_ocr)
                    continue
        merged.append(scene)

    for i, s in enumerate(merged):
        s["scene_id"] = i

    return merged


# =====================================================================
# Force-Split Long Scenes [v2.2 — P2 FIX]
# =====================================================================

def _force_split_long_scenes(scenes: list, times: list, paths: list,
                             embeddings: np.ndarray, signal_b: np.ndarray,
                             cfg: Config, output_dir: str,
                             signal_c: np.ndarray = None) -> tuple:
    """Split scenes exceeding MAX_SCENE_DURATION at highest combined peak.

    [v2.2 improvement] Uses DINOv2 + OCR Jaccard combined signal for split
    point selection. Text-heavy slides with similar backgrounds benefit from
    OCR Jaccard spikes even when DINOv2 is flat.
    """
    frames_dir = os.path.join(output_dir, "frames")
    new_scenes = []
    n_splits = 0

    for scene in scenes:
        if scene["duration"] <= cfg.MAX_SCENE_DURATION:
            new_scenes.append(scene)
            continue

        # Find frame indices for this scene
        s_idx = np.argmin(np.abs(np.array(times) - scene["t_start"]))
        e_idx = min(np.argmin(np.abs(np.array(times) - scene["t_end"])), len(times) - 1)

        if e_idx - s_idx < 4:
            new_scenes.append(scene)
            continue

        # [v2.2] Combined split signal: DINOv2 + OCR Jaccard
        margin = max(1, (e_idx - s_idx) // 10)
        dino_slice = signal_b[s_idx + margin:e_idx - margin + 1]
        if len(dino_slice) == 0:
            new_scenes.append(scene)
            continue

        if signal_c is not None and len(signal_c) > s_idx + margin:
            ocr_slice = signal_c[s_idx + margin:s_idx + margin + len(dino_slice)]
            if len(ocr_slice) == len(dino_slice) and ocr_slice.max() > 0:
                # Combine: 60% DINOv2 + 40% OCR (OCR more reliable for text changes)
                scene_signal = 0.6 * dino_slice + 0.4 * ocr_slice
                logger.info("Force-split using DINOv2+OCR combined signal")
            else:
                scene_signal = dino_slice
        else:
            scene_signal = dino_slice

        split_local = int(np.argmax(scene_signal))
        split_idx = s_idx + margin + split_local

        logger.info("Force-split scene %d (%.1fs) at t=%.1f",
                    scene["scene_id"], scene["duration"], times[split_idx])

        # Build two sub-scenes
        for sub_s, sub_e in [(s_idx, split_idx), (split_idx, e_idx)]:
            sub_emb = embeddings[sub_s:sub_e + 1]
            n_frames = len(sub_emb)
            if n_frames == 0:
                continue

            # Keyframe via centroid
            if n_frames >= 3:
                m = max(1, int(n_frames * 0.1))
                if m < n_frames // 2:
                    inner = sub_emb[m:-m]
                    offset = m
                else:
                    inner = sub_emb
                    offset = 0
                kf_local = find_centroid_frame(inner)
                kf_idx = sub_s + offset + kf_local
            else:
                kf_idx = sub_s + n_frames // 2

            kf_idx = min(kf_idx, len(times) - 1)
            sid = len(new_scenes)

            # Copy keyframe
            dst_path = os.path.join(frames_dir, f"scene_{sid:03d}.jpg")
            try:
                shutil.copy2(paths[kf_idx], dst_path)
            except Exception:
                dst_path = paths[kf_idx]

            # OCR on keyframe
            if cfg.OCR_ENABLED:
                ocr_results = run_ocr(paths[kf_idx], cfg.OCR_ENGINE, cfg.OCR_MIN_CONFIDENCE)
                ocr_words = extract_words_from_ocr(ocr_results)
            else:
                ocr_words = set()

            new_scenes.append({
                "scene_id": sid,
                "t_start": round(times[sub_s], 3),
                "t_end": round(times[sub_e], 3),
                "duration": round(times[sub_e] - times[sub_s], 3),
                "t_keyframe": round(times[kf_idx], 3),
                "keyframe_path": dst_path,
                "n_frames": n_frames,
                "ocr_words": " ".join(sorted(ocr_words)),
                "n_ocr_words": len(ocr_words),
            })

        n_splits += 1

    # Re-number
    for i, s in enumerate(new_scenes):
        s["scene_id"] = i

    if n_splits > 0:
        logger.info("Force-split %d scenes exceeding %.0fs", n_splits, cfg.MAX_SCENE_DURATION)

    return new_scenes, n_splits


# =====================================================================
# OCR at Scene Start [v2.2 — P3 FIX]
# =====================================================================

def _add_scene_start_ocr(scenes: list, times: list, paths: list, cfg: Config):
    """Add OCR words from scene start frame, union with keyframe OCR."""
    if not cfg.OCR_ENABLED:
        return

    for scene in scenes:
        s_idx = np.argmin(np.abs(np.array(times) - scene["t_start"]))
        kf_idx = np.argmin(np.abs(np.array(times) - scene["t_keyframe"]))

        # Skip if start frame IS the keyframe
        if s_idx == kf_idx:
            continue

        if s_idx < len(paths):
            start_ocr = run_ocr(paths[s_idx], cfg.OCR_ENGINE, cfg.OCR_MIN_CONFIDENCE)
            start_words = extract_words_from_ocr(start_ocr)

            existing = set(scene["ocr_words"].split()) if scene["ocr_words"] else set()
            all_words = existing | start_words
            scene["ocr_words"] = " ".join(sorted(all_words))
            scene["n_ocr_words"] = len(all_words)


# =====================================================================
# Diagnostics [v2.2]
# =====================================================================

def _write_diagnostics(scenes, times, signal_a, signal_b, signal_c,
                       combined, threshold, threshold_method, fusion_mode,
                       duration, n_boundaries_initial, n_boundaries_after_guard,
                       n_scenes_before_merge, n_merged, n_force_splits,
                       n_scenes_final, threshold_adjustments,
                       ocr_has_signal, cfg, diag,
                       scene_detection_mode="natural"):
    """Write comprehensive Stage 2 diagnostics."""

    # Scene duration stats
    durations = [s["duration"] for s in scenes]
    dur_stats = {}
    if durations:
        dur_stats = {
            "mean_s": round(float(np.mean(durations)), 2),
            "median_s": round(float(np.median(durations)), 2),
            "min_s": round(float(np.min(durations)), 2),
            "max_s": round(float(np.max(durations)), 2),
            "std_s": round(float(np.std(durations)), 2),
        }

    # OCR coverage
    n_scenes_with_ocr = sum(1 for s in scenes if s["n_ocr_words"] > 0)
    total_ocr_words = sum(s["n_ocr_words"] for s in scenes)

    # Signal statistics
    signal_stats = {
        "signal_a_pyscenedetect": {
            "n_nonzero": int(np.count_nonzero(signal_a)),
            "max": round(float(signal_a.max()), 4) if len(signal_a) > 0 else 0,
        },
        "signal_b_dinov2": {
            "mean": round(float(signal_b.mean()), 4) if len(signal_b) > 0 else 0,
            "median": round(float(np.median(signal_b)), 4) if len(signal_b) > 0 else 0,
            "max": round(float(signal_b.max()), 4) if len(signal_b) > 0 else 0,
            "std": round(float(signal_b.std()), 4) if len(signal_b) > 0 else 0,
        },
        "signal_c_ocr_jaccard": {
            "has_signal": ocr_has_signal,
            "n_nonzero": int(np.count_nonzero(signal_c)),
            "max": round(float(signal_c.max()), 4) if len(signal_c) > 0 else 0,
        },
        "combined": {
            "mean": round(float(combined.mean()), 4) if len(combined) > 0 else 0,
            "median": round(float(np.median(combined)), 4) if len(combined) > 0 else 0,
            "max": round(float(combined.max()), 4) if len(combined) > 0 else 0,
            "std": round(float(combined.std()), 4) if len(combined) > 0 else 0,
        },
    }

    # Per-scene summary
    scene_summaries = []
    for s in scenes:
        scene_summaries.append({
            "scene_id": s["scene_id"],
            "t_start": s["t_start"],
            "t_end": s["t_end"],
            "duration": s["duration"],
            "t_keyframe": s["t_keyframe"],
            "n_ocr_words": s["n_ocr_words"],
            "ocr_words_preview": s["ocr_words"][:100] if s["ocr_words"] else "",
        })

    diag_data = {
        "video_duration_s": round(duration, 2),
        "n_frames_sampled": len(times),
        "sample_interval": cfg.SAMPLE_INTERVAL,
        "fusion_mode": fusion_mode,
        "threshold": round(float(threshold), 4),
        "threshold_method": threshold_method,
        "scene_detection_mode": scene_detection_mode,
        "n_boundaries_initial": n_boundaries_initial,
        "n_boundaries_after_guard": n_boundaries_after_guard,
        "threshold_adjustments": threshold_adjustments,
        "n_scenes_before_merge": n_scenes_before_merge,
        "n_scenes_merged": n_merged,
        "n_scenes_force_split": n_force_splits,
        "n_scenes_final": n_scenes_final,
        "scene_duration_stats": dur_stats,
        "max_scene_duration_limit": cfg.MAX_SCENE_DURATION,
        "expected_scenes_per_minute": cfg.EXPECTED_SCENES_PER_MINUTE,
        "ocr_engine": cfg.OCR_ENGINE,
        "ocr_sample_scene_start": cfg.OCR_SAMPLE_SCENE_START,
        "n_scenes_with_ocr": n_scenes_with_ocr,
        "total_ocr_words": total_ocr_words,
        "signal_stats": signal_stats,
        "scene_summaries": scene_summaries,
    }

    diag.write_json("stage2_scene_detection.json", diag_data)


def _write_diagnostics_from_cache(output_dir, diag):
    """Produce diagnostics from cached files."""
    try:
        scenes_df = pd.read_csv(os.path.join(output_dir, "scenes.csv"))
        scenes = scenes_df.to_dict("records")
        durations = [s["duration"] for s in scenes]
        dur_stats = {}
        if durations:
            dur_stats = {
                "mean_s": round(float(np.mean(durations)), 2),
                "median_s": round(float(np.median(durations)), 2),
                "min_s": round(float(np.min(durations)), 2),
                "max_s": round(float(np.max(durations)), 2),
                "std_s": round(float(np.std(durations)), 2),
            }
        n_scenes_with_ocr = sum(1 for s in scenes if s.get("n_ocr_words", 0) > 0)
        diag_data = {
            "source": "cached",
            "n_scenes_final": len(scenes),
            "scene_duration_stats": dur_stats,
            "n_scenes_with_ocr": n_scenes_with_ocr,
        }
        diag.write_json("stage2_scene_detection.json", diag_data)
    except Exception as e:
        logger.warning("Could not produce Stage 2 diagnostics from cache: %s", e)
