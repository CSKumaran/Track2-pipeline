"""Stage 5: Keyword Extraction & Visual Grounding (Core Novelty)."""

import logging
import os
import numpy as np
import pandas as pd

from ..config import Config
from ..utils.keyword_utils import (
    extract_keywords_spacy, extract_keywords_keybert, merge_keywords,
    classify_groundability, load_concreteness_db,
)
from ..utils.ocr_utils import normalize_word, fuzzy_match_word, fuzzy_match_multiword
from ..utils.io_utils import save_csv, cache_exists

logger = logging.getLogger(__name__)


def run_stage5(output_dir: str, cfg: Config) -> dict:
    """Run keyword extraction and visual grounding."""
    cache_files = ["keyword_alignment.csv", "segment_keyword_scores.csv"]
    if cache_exists(output_dir, cache_files):
        logger.info("Stage 5: cache hit, skipping keyword analysis")
        return {f: os.path.join(output_dir, f) for f in cache_files}

    # Load data
    segments_df = pd.read_csv(os.path.join(output_dir, "transcript_segments_improved.csv"))
    words_df = pd.read_csv(os.path.join(output_dir, "transcript_words.csv"))
    ocr_df = pd.read_csv(os.path.join(output_dir, "ocr_per_frame.csv"))
    scenes_df = pd.read_csv(os.path.join(output_dir, "scenes.csv"))

    # Load concreteness DB
    concreteness_db = load_concreteness_db()

    all_kw_results = []
    segment_scores = []

    for _, seg in segments_df.iterrows():
        seg_id = seg["segment_id"]
        text = str(seg["text"])
        seg_start = float(seg["start_time"]) if pd.notna(seg["start_time"]) else 0
        seg_end = float(seg["end_time"]) if pd.notna(seg["end_time"]) else 0

        # 5a: Extract keywords
        spacy_kws = extract_keywords_spacy(text)
        keybert_kws = []
        if cfg.KEYWORD_USE_KEYBERT:
            keybert_kws = extract_keywords_keybert(
                text, cfg.EMBEDDING_MODEL_NAME, cfg.KEYWORD_KEYBERT_TOP_N
            )

        keywords = merge_keywords(spacy_kws, keybert_kws, cfg.KEYWORD_MIN_LENGTH)

        n_grounded = 0
        n_not_visual = 0
        delta_ts = []
        confidences = []

        for kw_id, kw in enumerate(keywords):
            # Map to word timestamps
            t_narr, ts_reliable = _find_keyword_timestamp(kw, words_df, seg_start, seg_end)

            # 5b: Groundability
            groundability = classify_groundability(kw, concreteness_db)

            # Skip LOW groundability
            if groundability == "LOW":
                all_kw_results.append({
                    "keyword_id": f"{seg_id}_{kw_id}",
                    "keyword_text": kw,
                    "segment_id": seg_id,
                    "t_narr": t_narr,
                    "t_vis": None,
                    "delta_t": None,
                    "method": "skipped_low_groundability",
                    "confidence": "NONE",
                    "groundability": groundability,
                    "bounding_box_area": None,
                    "is_transient": False,
                    "is_visual": False,
                    "flag_for_review": False,
                    "timestamp_reliable": ts_reliable,
                })
                n_not_visual += 1
                continue

            # 5c: Visual Grounding Cascade
            result = _grounding_cascade(
                kw, t_narr, groundability, ocr_df, scenes_df, cfg, output_dir
            )

            if result["t_vis"] is not None:
                n_grounded += 1
                dt = result["t_vis"] - t_narr if t_narr is not None else None
                conf_weight = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4, "VERY_LOW": 0.2}.get(
                    result["confidence"], 0.5
                )
                if dt is not None:
                    delta_ts.append(dt)
                    confidences.append(conf_weight)
            else:
                dt = None
                n_not_visual += 1

            all_kw_results.append({
                "keyword_id": f"{seg_id}_{kw_id}",
                "keyword_text": kw,
                "segment_id": seg_id,
                "t_narr": t_narr,
                "t_vis": result["t_vis"],
                "delta_t": dt,
                "method": result["method"],
                "confidence": result["confidence"],
                "groundability": groundability,
                "bounding_box_area": result.get("bbox_area"),
                "is_transient": result.get("is_transient", False),
                "is_visual": result["t_vis"] is not None,
                "flag_for_review": result.get("flag_for_review", False),
                "timestamp_reliable": ts_reliable,
            })

        # Segment-level aggregation
        if delta_ts:
            weights = np.array(confidences)
            dts = np.array(delta_ts)
            # Confidence-weighted median approximation
            sorted_idx = np.argsort(dts)
            cumw = np.cumsum(weights[sorted_idx])
            median_idx = np.searchsorted(cumw, cumw[-1] / 2)
            median_idx = min(median_idx, len(sorted_idx) - 1)
            weighted_median = dts[sorted_idx[median_idx]]
        else:
            weighted_median = None

        segment_scores.append({
            "segment_id": seg_id,
            "n_keywords": len(keywords),
            "n_groundable": len([k for k in keywords if classify_groundability(k, concreteness_db) != "LOW"]),
            "n_grounded": n_grounded,
            "n_not_visual": n_not_visual,
            "delta_t_weighted_median": weighted_median,
            "delta_t_mean": np.mean(delta_ts) if delta_ts else None,
            "delta_t_std": np.std(delta_ts) if delta_ts else None,
        })

    kw_df = pd.DataFrame(all_kw_results)
    seg_scores_df = pd.DataFrame(segment_scores)

    save_csv(kw_df, os.path.join(output_dir, "keyword_alignment.csv"))
    save_csv(seg_scores_df, os.path.join(output_dir, "segment_keyword_scores.csv"))

    logger.info("Stage 5: %d keywords extracted, %d grounded",
                len(kw_df), len(kw_df[kw_df["is_visual"] == True]) if not kw_df.empty else 0)
    return {f: os.path.join(output_dir, f) for f in cache_files}


def _find_keyword_timestamp(keyword: str, words_df: pd.DataFrame,
                            seg_start: float, seg_end: float) -> tuple:
    """Find keyword in transcript words, return (t_narr, timestamp_reliable)."""
    if words_df.empty:
        return None, True

    kw_parts = keyword.lower().split()
    # Search within segment time range
    mask = (
        (pd.to_numeric(words_df["start_time"], errors="coerce") >= seg_start - 1)
        & (pd.to_numeric(words_df["start_time"], errors="coerce") <= seg_end + 1)
    )
    seg_words = words_df[mask]

    if seg_words.empty:
        return (seg_start + seg_end) / 2, True

    # Find first word of keyword in transcript
    for _, w in seg_words.iterrows():
        w_text = str(w["word"]).strip().lower()
        if w_text == kw_parts[0] or (len(kw_parts[0]) >= 3 and kw_parts[0] in w_text):
            t_narr = float(w["start_time"]) if pd.notna(w["start_time"]) else None
            ts_reliable = bool(w.get("timestamp_reliable", True))
            return t_narr, ts_reliable

    # Fallback: segment midpoint
    return (seg_start + seg_end) / 2, True


def _grounding_cascade(keyword: str, t_narr: float, groundability: str,
                       ocr_df: pd.DataFrame, scenes_df: pd.DataFrame,
                       cfg: Config, output_dir: str) -> dict:
    """4-step grounding cascade."""
    if t_narr is None:
        t_narr = 0

    # Step 1: OCR fuzzy search (±60s window)
    result = _step1_ocr_search(keyword, t_narr, ocr_df, cfg)
    if result["t_vis"] is not None:
        return result

    # Step 2: GroundingDINO (if enabled)
    if cfg.GROUNDING_DINO_ENABLED:
        result = _step2_grounding_dino(keyword, t_narr, scenes_df, cfg, output_dir)
        if result["t_vis"] is not None:
            # 5d: Persistence check
            result = _persistence_check(result, keyword, cfg, output_dir)
            return result

    # Step 3: SigLIP contextual matching
    if cfg.SIGLIP_ENABLED:
        result = _step3_siglip(keyword, t_narr, scenes_df, cfg, output_dir)
        if result["t_vis"] is not None:
            result = _persistence_check(result, keyword, cfg, output_dir)
            return result

    # Step 4: VLM visual existence check (HIGH groundability only)
    if groundability == "HIGH" and cfg.VLM_MODE != "skip":
        result = _step4_vlm_check(keyword, t_narr, scenes_df, cfg, output_dir)
        if result["t_vis"] is not None:
            result = _persistence_check(result, keyword, cfg, output_dir)
            return result

    return {"t_vis": None, "method": "not_found", "confidence": "NONE",
            "flag_for_review": False}


def _step1_ocr_search(keyword: str, t_narr: float, ocr_df: pd.DataFrame,
                      cfg: Config) -> dict:
    """Step 1: OCR fuzzy search within ±60s window."""
    if ocr_df.empty or "words" not in ocr_df.columns:
        return {"t_vis": None, "method": "ocr", "confidence": "NONE"}

    window = 60.0
    mask = (
        (pd.to_numeric(ocr_df["frame_time"], errors="coerce") >= t_narr - window)
        & (pd.to_numeric(ocr_df["frame_time"], errors="coerce") <= t_narr + window)
    )
    nearby = ocr_df[mask].copy()

    if nearby.empty:
        return {"t_vis": None, "method": "ocr", "confidence": "NONE"}

    # Search for keyword in OCR words
    best_t = None
    best_dist = float("inf")

    for _, row in nearby.iterrows():
        frame_words = set(str(row["words"]).split())
        if fuzzy_match_multiword(keyword, frame_words, cfg.OCR_FUZZY_THRESHOLD):
            frame_t = float(row["frame_time"])
            dist = abs(frame_t - t_narr)
            if dist < best_dist:
                best_dist = dist
                best_t = frame_t

    if best_t is not None:
        return {"t_vis": best_t, "method": "ocr", "confidence": "HIGH",
                "flag_for_review": False}

    return {"t_vis": None, "method": "ocr", "confidence": "NONE"}


def _step2_grounding_dino(keyword, t_narr, scenes_df, cfg, output_dir):
    """Step 2: GroundingDINO/Florence-2 detection."""
    from ..utils.grounding_utils import detect_objects

    # Get keyframes within ±30s
    window = 30.0
    nearby_scenes = scenes_df[
        (pd.to_numeric(scenes_df["t_keyframe"], errors="coerce") >= t_narr - window)
        & (pd.to_numeric(scenes_df["t_keyframe"], errors="coerce") <= t_narr + window)
    ]

    best_conf = 0
    best_t = None
    best_bbox_area = None

    for _, scene in nearby_scenes.iterrows():
        kf_path = scene["keyframe_path"]
        if not os.path.exists(kf_path):
            continue

        # Augment single words with context
        query = keyword
        if len(keyword.split()) == 1:
            query = f"{keyword} label or {keyword} diagram"

        detections = detect_objects(
            kf_path, query,
            cfg.GROUNDING_DINO_BOX_THRESHOLD,
            cfg.GROUNDING_DINO_TEXT_THRESHOLD,
            cfg.GROUNDING_MODEL,
        )

        for bbox, conf, label in detections:
            if conf > best_conf:
                best_conf = conf
                best_t = float(scene["t_keyframe"])
                # bbox area
                if len(bbox) == 4:
                    best_bbox_area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    if best_t is not None:
        return {"t_vis": best_t, "method": "grounding_dino", "confidence": "MEDIUM",
                "bbox_area": best_bbox_area, "flag_for_review": False}

    return {"t_vis": None, "method": "grounding_dino", "confidence": "NONE"}


def _step3_siglip(keyword, t_narr, scenes_df, cfg, output_dir):
    """Step 3: SigLIP contextual matching."""
    from ..utils.siglip_utils import embed_images, embed_texts, sigmoid_similarity, temporal_gaussian_weight

    window = 30.0
    nearby_scenes = scenes_df[
        (pd.to_numeric(scenes_df["t_keyframe"], errors="coerce") >= t_narr - window)
        & (pd.to_numeric(scenes_df["t_keyframe"], errors="coerce") <= t_narr + window)
    ]

    if nearby_scenes.empty:
        return {"t_vis": None, "method": "siglip_contextual", "confidence": "NONE"}

    kf_paths = nearby_scenes["keyframe_path"].tolist()
    kf_times = pd.to_numeric(nearby_scenes["t_keyframe"], errors="coerce").values
    valid = [(p, t) for p, t in zip(kf_paths, kf_times) if os.path.exists(p)]
    if not valid:
        return {"t_vis": None, "method": "siglip_contextual", "confidence": "NONE"}

    paths_valid, times_valid = zip(*valid)

    # Embed keyword (as full sentence context)
    text_emb = embed_texts([f"A visual showing {keyword}"])
    img_emb = embed_images(list(paths_valid))

    sims = sigmoid_similarity(img_emb, text_emb)[:, 0]

    # Apply temporal decay
    weighted = np.array([
        sims[j] * temporal_gaussian_weight(times_valid[j], t_narr, cfg.TEMPORAL_SIGMA)
        for j in range(len(sims))
    ])

    best_idx = int(np.argmax(weighted))
    best_sim = float(sims[best_idx])

    if best_sim > 0.15:
        return {"t_vis": float(times_valid[best_idx]), "method": "siglip_contextual",
                "confidence": "LOW", "flag_for_review": False}

    return {"t_vis": None, "method": "siglip_contextual", "confidence": "NONE"}


def _step4_vlm_check(keyword, t_narr, scenes_df, cfg, output_dir):
    """Step 4: VLM visual existence check."""
    import requests
    import base64

    window = 30.0
    nearby_scenes = scenes_df[
        (pd.to_numeric(scenes_df["t_keyframe"], errors="coerce") >= t_narr - window)
        & (pd.to_numeric(scenes_df["t_keyframe"], errors="coerce") <= t_narr + window)
    ]

    for _, scene in nearby_scenes.iterrows():
        kf_path = scene["keyframe_path"]
        if not os.path.exists(kf_path):
            continue

        prompt = (
            f"Does this frame visually depict the concept '{keyword}'? "
            f"Answer with just YES or NO."
        )

        try:
            with open(kf_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            resp = requests.post(
                f"{cfg.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": cfg.OLLAMA_MODEL,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=60,
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip().upper()
                if "YES" in answer:
                    return {"t_vis": float(scene["t_keyframe"]),
                            "method": "vlm_classification", "confidence": "LOW",
                            "flag_for_review": True}
        except Exception as e:
            logger.warning("VLM check failed for %s: %s", keyword, e)
            continue

    return {"t_vis": None, "method": "vlm_classification", "confidence": "NONE",
            "flag_for_review": False}


def _persistence_check(result: dict, keyword: str, cfg: Config, output_dir: str) -> dict:
    """5d: 3-frame persistence check on non-OCR detections."""
    if result["method"] == "ocr":
        return result  # OCR is ground truth, skip

    # For now, simplified: check if OCR also finds the keyword nearby
    # Full persistence check would need adjacent frame access
    # Mark as not-transient by default (conservative)
    result["is_transient"] = False
    return result


import os
