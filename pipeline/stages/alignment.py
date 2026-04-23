"""Stage 4b – Align scene concepts to narration using 3-track priority cascade:
  - Track A: exact OCR word matching against transcript
  - Track B: CLIP vision-to-text alignment (NEW)
  - Track C: semantic cosine similarity (global best match)
"""

from __future__ import annotations

import logging
import os
import re
import string
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.utils.embedding_utils import get_text_embeddings
from pipeline.utils.io_utils import ensure_dir, safe_write_csv
from pipeline.stages.text_units import (
    load_word_level_data,
    get_words_for_segment,
    build_word_windows,
)

if TYPE_CHECKING:
    from pipeline.config import Config

logger = logging.getLogger(__name__)

# These are now configurable via Config but kept as module-level defaults
# for backward compatibility with direct function calls.
MIN_GLOBAL_SIM = 0.20
MIN_WORD_MATCHES_FOR_TRACK_A = 1


def _normalize_word(w: str) -> str:
    """Lowercase and strip punctuation from a word."""
    return re.sub(r"[^a-z0-9]", "", w.lower())


# Common English function words to exclude from Track A matching.
# These appear so frequently in transcripts that they add noise, not signal.
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "is", "it", "its", "be", "are", "was", "were", "been", "has",
    "have", "had", "do", "does", "did", "will", "can", "may", "not", "no",
    "so", "if", "by", "as", "we", "he", "she", "you", "they", "this",
    "that", "with", "from", "than", "then", "into", "also", "very", "more",
    "about", "dont", "only", "just", "some", "when", "how", "what", "where",
    "there", "here", "all", "each", "every", "both", "one", "two", "new",
    "get", "got", "our", "out", "way", "own", "well",
}


def _run_track_a_exact_match(
    ocr_text: str,
    words_df: pd.DataFrame,
    t_vis: float,
    temporal_window: float = 30.0,
) -> dict:
    """Track A: exact word matching between OCR text and transcript words.

    For each OCR word (excluding stop words), find the closest exact match
    in transcript_words.csv within ±temporal_window seconds of t_vis.
    Compute per-word delta_t, then take the median.

    Parameters
    ----------
    temporal_window : float
        Maximum seconds from t_vis to consider a transcript word match.
        Words whose closest occurrence is farther away are skipped.

    Returns
    -------
    dict with keys: n_word_matches, trackA_delta_t, matched_words
    """
    if not ocr_text or not ocr_text.strip():
        return {"n_word_matches": 0, "trackA_delta_t": None, "matched_words": []}

    # Normalize OCR words — skip stop words and very short words
    ocr_words = {
        _normalize_word(w) for w in ocr_text.split()
        if len(_normalize_word(w)) >= 3 and _normalize_word(w) not in _STOP_WORDS
    }
    if not ocr_words:
        return {"n_word_matches": 0, "trackA_delta_t": None, "matched_words": []}

    # Normalize transcript words for lookup
    transcript_words_norm = words_df["word"].astype(str).apply(_normalize_word)

    matched_deltas: list[float] = []
    matched_words: list[str] = []

    for ocr_w in ocr_words:
        # Find all transcript words that match exactly
        mask = transcript_words_norm == ocr_w
        if not mask.any():
            continue
        matching_rows = words_df[mask]
        # FILTER: only consider words within temporal window of t_vis
        nearby = matching_rows[(matching_rows["t_word"] - t_vis).abs() <= temporal_window]
        if nearby.empty:
            continue  # no nearby occurrence — skip this word
        # Pick the closest match by timestamp to t_vis
        distances = (nearby["t_word"] - t_vis).abs()
        closest_idx = distances.idxmin()
        closest_row = nearby.loc[closest_idx]
        dt = float(closest_row["t_word"]) - t_vis
        matched_deltas.append(dt)
        matched_words.append(ocr_w)

    if not matched_deltas:
        return {"n_word_matches": 0, "trackA_delta_t": None, "matched_words": []}

    median_dt = float(np.median(matched_deltas))
    return {
        "n_word_matches": len(matched_deltas),
        "trackA_delta_t": round(median_dt, 4),
        "matched_words": matched_words,
    }


def align_scenes_to_narration(
    scenes_with_concepts_df: pd.DataFrame,
    segment_meta_df: pd.DataFrame,
    segment_embeddings: np.ndarray,
    config: "Config",
    output_dir: str = "",
    frames_dir: str = "",
) -> pd.DataFrame:
    """For each scene, run three alignment tracks and pick the best:

    **Track A (exact word match)**:
      - Take OCR words from new_ocr_words
      - Search transcript_words.csv for exact matches
      - Compute per-word delta_t, take median

    **Track B (CLIP vision-to-text)**:
      - Embed keyframe image + transcript segments with CLIP
      - Temporal Gaussian decay, word-level drill-down

    **Track C (semantic match)** (formerly Track B):
      - Embed concept_text, cosine similarity against segments
      - Word-level drill-down for precise t_narr

    **Priority cascade**: Track A → Track B (CLIP) → Track C → no_match

    Parameters
    ----------
    scenes_with_concepts_df : pd.DataFrame
        Must have: threshold, scene_id, t_vis, concept_text.
        Optionally: ocr_text (for Track A).
    segment_meta_df : pd.DataFrame
        Must have: segment_id, text, start_time, end_time, t_seg.
    segment_embeddings : np.ndarray
        Shape (N_segments, d).
    config : Config
        Pipeline configuration.
    output_dir : str
        Per-video output directory (for loading word data).
    frames_dir : str
        Path to keyframe images (for CLIP Track B).

    Returns
    -------
    pd.DataFrame
        One row per scene with alignment columns including match_track.
    """
    empty_cols = [
        "threshold", "scene_id", "t_vis", "concept_text",
        "match_type", "match_track", "best_segment_id", "best_segment_text",
        "t_narr_segment", "t_narr_word", "t_narr", "delta_t",
        "sim_segment", "sim_words", "best_word_window",
        "n_word_matches", "trackA_delta_t",
        "clip_sim", "clip_best_seg_text", "clip_word_window", "trackB_clip_delta_t",
        "trackC_delta_t",
        "scene_type", "scene_type_conf",
    ]
    if len(scenes_with_concepts_df) == 0:
        return pd.DataFrame(columns=empty_cols)

    # Load word-level data for word-window drill-down and Track A
    words_df = load_word_level_data(output_dir)

    # Pre-compute all concept embeddings in one batch (for Track C)
    concept_texts = scenes_with_concepts_df["concept_text"].astype(str).tolist()
    concept_embeddings = get_text_embeddings(concept_texts, config)

    seg_ids = segment_meta_df["segment_id"].values
    seg_texts = segment_meta_df["text"].astype(str).values
    seg_texts_list = list(seg_texts)
    seg_starts = segment_meta_df["start_time"].values
    seg_ends = segment_meta_df["end_time"].values
    t_segs = segment_meta_df["t_seg"].values

    # Check if ocr_text column exists
    has_ocr = "ocr_text" in scenes_with_concepts_df.columns

    # Use config values (with module-level defaults as fallback)
    min_global_sim = getattr(config, "MIN_GLOBAL_SIM", MIN_GLOBAL_SIM)
    min_word_matches = getattr(config, "MIN_WORD_MATCHES", MIN_WORD_MATCHES_FOR_TRACK_A)
    temporal_sigma = getattr(config, "TEMPORAL_SIGMA", 15.0)

    # CLIP settings
    clip_enabled = getattr(config, "CLIP_ENABLED", False) and bool(frames_dir)
    clip_min_sim = getattr(config, "CLIP_MIN_SIM", 0.20)

    if clip_enabled:
        try:
            from pipeline.utils.clip_utils import (
                compute_clip_alignment,
                classify_scene_type,
            )
            logger.info("CLIP Track B enabled (model=%s, min_sim=%.2f)",
                        config.CLIP_MODEL_NAME, clip_min_sim)
        except ImportError:
            logger.warning("CLIP not available — disabling Track B")
            clip_enabled = False

    results: list[dict] = []

    for i, (_, row) in enumerate(scenes_with_concepts_df.iterrows()):
        t_vis = row["t_vis"]
        concept = str(row["concept_text"])
        sid = int(row["scene_id"])

        # ── Skip NON_CONTENT scenes ────────────────────────────────────
        if concept.upper().strip() == "NON_CONTENT":
            results.append({
                "threshold": row["threshold"],
                "scene_id": sid,
                "t_vis": round(t_vis, 4),
                "concept_text": concept,
                "match_type": "non_content",
                "match_track": None,
                "best_segment_id": None,
                "best_segment_text": None,
                "t_narr_segment": None,
                "t_narr_word": None,
                "t_narr": None,
                "delta_t": None,
                "sim_segment": None,
                "sim_words": None,
                "best_word_window": None,
                "n_word_matches": 0,
                "trackA_delta_t": None,
                "clip_sim": None,
                "clip_best_seg_text": None,
                "clip_word_window": None,
                "trackB_clip_delta_t": None,
                "trackC_delta_t": None,
                "scene_type": None,
                "scene_type_conf": None,
            })
            continue

        # ── Track A: exact OCR word matching ───────────────────────────
        _raw_nw = row.get("new_ocr_words", "")
        new_ocr_words = str(_raw_nw).strip() if pd.notna(_raw_nw) else ""
        _raw_ocr = row.get("ocr_text", "")
        ocr_text_full = str(_raw_ocr).strip() if pd.notna(_raw_ocr) else ""
        track_a_text = new_ocr_words if new_ocr_words else ocr_text_full
        track_a = _run_track_a_exact_match(
            track_a_text, words_df, t_vis,
            temporal_window=config.TRACK_A_TEMPORAL_WINDOW,
        )

        # ── Track B: CLIP vision-to-text alignment ─────────────────────
        clip_result = {
            "clip_sim": 0.0,
            "clip_best_seg_id": None,
            "clip_best_seg_text": None,
            "clip_t_narr": None,
            "clip_delta_t": None,
            "clip_word_window": None,
        }
        scene_type_info = {"scene_type": None, "scene_type_conf": None}

        if clip_enabled:
            keyframe_path = os.path.join(frames_dir, f"scene_{sid}.jpg")
            clip_result = compute_clip_alignment(
                image_path=keyframe_path,
                segment_texts=seg_texts_list,
                t_segs=t_segs,
                seg_starts=seg_starts,
                seg_ends=seg_ends,
                t_vis=t_vis,
                temporal_sigma=temporal_sigma,
                words_df=words_df,
                config=config,
            )
            scene_type_info = classify_scene_type(keyframe_path, config)

        # ── Track C: semantic matching (formerly Track B) ──────────────
        c_emb = concept_embeddings[i].reshape(1, -1)
        raw_sims = cosine_similarity(c_emb, segment_embeddings).flatten()

        time_offsets = np.abs(t_segs - t_vis)
        temporal_weights = np.exp(-0.5 * (time_offsets / temporal_sigma) ** 2)
        weighted_sims = raw_sims * temporal_weights

        best_idx = int(np.argmax(weighted_sims))
        sim_segment = float(raw_sims[best_idx])
        best_seg_id = int(seg_ids[best_idx])
        best_seg_text = str(seg_texts[best_idx])
        t_narr_segment = float(t_segs[best_idx])

        # Track C word-level drill-down
        trackC_delta_t = None
        t_narr_word_c = None
        sim_words_best = 0.0
        best_ww_text = None

        if sim_segment >= min_global_sim:
            seg_start = float(seg_starts[best_idx])
            seg_end = float(seg_ends[best_idx])
            words_in_seg = get_words_for_segment(words_df, seg_start, seg_end)
            word_windows = build_word_windows(words_in_seg, window_size=3)

            if word_windows:
                ww_texts = [ww["text"] for ww in word_windows]
                ww_embeddings = get_text_embeddings(ww_texts, config)
                ww_sims = cosine_similarity(c_emb, ww_embeddings).flatten()
                best_ww_idx = int(np.argmax(ww_sims))
                sim_words_best = float(ww_sims[best_ww_idx])
                t_narr_word_c = word_windows[best_ww_idx]["t_center"]
                best_ww_text = word_windows[best_ww_idx]["text"]

            t_narr_c = t_narr_word_c if t_narr_word_c is not None else t_narr_segment
            trackC_delta_t = round(t_narr_c - t_vis, 4)

        # ── Priority cascade: Track A → Track B → Track C → no_match ──
        clip_sim = clip_result.get("clip_sim", 0.0) or 0.0

        if (track_a["n_word_matches"] >= min_word_matches
                and track_a["trackA_delta_t"] is not None):
            # Track A wins
            match_track = "word_exact"
            t_narr = t_vis + track_a["trackA_delta_t"]
            delta_t = track_a["trackA_delta_t"]
            match_type = "matched"
        elif clip_enabled and clip_sim >= clip_min_sim and clip_result.get("clip_t_narr") is not None:
            # Track B (CLIP) wins
            match_track = "clip_vision"
            t_narr = clip_result["clip_t_narr"]
            delta_t = clip_result["clip_delta_t"]
            match_type = "matched"
        elif sim_segment >= min_global_sim:
            # Track C (semantic) wins
            match_track = "semantic"
            t_narr_final = t_narr_word_c if t_narr_word_c is not None else t_narr_segment
            t_narr = t_narr_final
            delta_t = t_narr - t_vis
            match_type = "matched"
        else:
            # No match from any track
            match_track = None
            t_narr = None
            delta_t = None
            match_type = "no_match"

        # Show matched words from whichever track was used
        if match_track == "word_exact":
            display_word_match = ", ".join(track_a["matched_words"])
        elif match_track == "clip_vision":
            display_word_match = clip_result.get("clip_word_window") or best_ww_text
        else:
            display_word_match = best_ww_text

        results.append({
            "threshold": row["threshold"],
            "scene_id": sid,
            "t_vis": round(t_vis, 4),
            "concept_text": row["concept_text"],
            "match_type": match_type,
            "match_track": match_track,
            "best_segment_id": best_seg_id,
            "best_segment_text": best_seg_text,
            "t_narr_segment": round(t_narr_segment, 4),
            "t_narr_word": round(t_narr_word_c, 4) if t_narr_word_c is not None else None,
            "t_narr": round(t_narr, 4) if t_narr is not None else None,
            "delta_t": round(delta_t, 4) if delta_t is not None else None,
            "sim_segment": round(sim_segment, 4),
            "sim_words": round(sim_words_best, 4) if sim_words_best > 0 else None,
            "best_word_window": display_word_match,
            "n_word_matches": track_a["n_word_matches"],
            "trackA_delta_t": track_a["trackA_delta_t"],
            "clip_sim": round(clip_sim, 4) if clip_sim > 0 else None,
            "clip_best_seg_text": clip_result.get("clip_best_seg_text"),
            "clip_word_window": clip_result.get("clip_word_window"),
            "trackB_clip_delta_t": clip_result.get("clip_delta_t"),
            "trackC_delta_t": trackC_delta_t,
            "scene_type": scene_type_info.get("scene_type"),
            "scene_type_conf": scene_type_info.get("scene_type_conf"),
        })

    alignment_df = pd.DataFrame(results)

    # Diagnostics
    _log_alignment_diagnostics(alignment_df)

    return alignment_df


def save_alignment_results(
    alignment_df: pd.DataFrame, output_dir: str
) -> None:
    """Save alignment CSVs per threshold into *output_dir*."""
    for threshold, group in alignment_df.groupby("threshold"):
        csv_path = os.path.join(
            output_dir, f"alignment_events_threshold_{threshold}.csv"
        )
        safe_write_csv(group, csv_path)


def _log_alignment_diagnostics(alignment_df: pd.DataFrame) -> None:
    """Log match statistics and Δt diagnostics."""
    total = len(alignment_df)
    if total == 0:
        return

    for threshold, group in alignment_df.groupby("threshold"):
        n = len(group)
        matched = group[group["match_type"] == "matched"]
        no_match = group[group["match_type"] == "no_match"]

        logger.info("── Alignment diagnostics (threshold=%s) ──", threshold)
        logger.info("  matched: %d (%.1f%%)", len(matched), 100 * len(matched) / n)
        logger.info("  no_match: %d (%.1f%%)", len(no_match), 100 * len(no_match) / n)

        if len(matched) > 0:
            dt = matched["delta_t"].astype(float)
            logger.info(
                "  Δt: mean=%.2f, SD=%.2f, min=%.2f, max=%.2f",
                dt.mean(), dt.std(), dt.min(), dt.max(),
            )
            sim_seg = matched["sim_segment"].astype(float)
            logger.info(
                "  Segment sim: mean=%.3f, min=%.3f, max=%.3f",
                sim_seg.mean(), sim_seg.min(), sim_seg.max(),
            )
            ww = matched["sim_words"].dropna().astype(float)
            if len(ww) > 0:
                logger.info(
                    "  Word-window sim: mean=%.3f, min=%.3f, max=%.3f",
                    ww.mean(), ww.min(), ww.max(),
                )
