"""Stage 4: Transcript-First Alignment [v2.2].

For each groundable keyword, search for its visual counterpart using a priority
cascade. This implements Cases A-H from the architectural plan.

Search Priority Order (per keyword):
  1. Case A: OCR at t_narr → visual present now → delta_t ≤ 0 → Score 100
  2. Case B: Forward search +10s → visual appears late → delta_t > 0 → penalized
  3. Case C: Backward search -10s → visual expired → delta_t capped at search_window
  4. Case G: SigLIP similarity → visual match without text → reduced alpha
  5. Case F: No visual correlate → excluded from scoring

Asymmetric scoring:
  - delta_t = t_vis - t_narr
  - delta_t ≤ 0 → visual already present when concept spoken → NO penalty
  - delta_t > 0 → visual appears AFTER narration → penalized

Outputs:
    keyword_alignment.csv — per-keyword: t_narr, t_vis, delta_t, match_case, alpha, ...
    segment_alignment.csv — per-segment aggregates
    stage4_alignment.json — diagnostics
"""

import logging
import os
import numpy as np
import pandas as pd

from ..config import Config
from ..utils.ocr_utils import (
    normalize_word, fuzzy_match_word, fuzzy_match_multiword,
)
from ..utils.io_utils import save_csv, cache_exists

logger = logging.getLogger(__name__)

# Search window (seconds) around t_narr
SEARCH_WINDOW = 10.0
# OCR time tolerance for "at t_narr" check (seconds)
OCR_AT_TNARR_TOLERANCE = 1.0


def run_stage4(output_dir: str, cfg: Config, diag=None) -> dict:
    """Transcript-first alignment: for each keyword, find visual onset (t_vis).

    Requires Stage 2 (scenes.csv, ocr_per_frame.csv) and Stage 5 (keywords.csv).
    """
    cache_files = ["keyword_alignment.csv", "segment_alignment.csv"]
    if cache_exists(output_dir, cache_files):
        logger.info("Stage 4: cache hit, skipping alignment")
        return {f: os.path.join(output_dir, f) for f in cache_files}

    # Load inputs
    keywords_df = pd.read_csv(os.path.join(output_dir, "keywords.csv"))
    scenes_df = pd.read_csv(os.path.join(output_dir, "scenes.csv"))
    ocr_df = pd.read_csv(os.path.join(output_dir, "ocr_per_frame.csv"))
    segments_df = pd.read_csv(
        os.path.join(output_dir, "transcript_segments_improved.csv")
    )

    if keywords_df.empty:
        logger.warning("No keywords for alignment")
        save_csv(pd.DataFrame(), os.path.join(output_dir, "keyword_alignment.csv"))
        save_csv(pd.DataFrame(), os.path.join(output_dir, "segment_alignment.csv"))
        return {f: os.path.join(output_dir, f) for f in cache_files}

    # Parse numeric columns
    ocr_df["frame_time"] = pd.to_numeric(ocr_df["frame_time"], errors="coerce")
    scenes_df["t_start"] = pd.to_numeric(scenes_df["t_start"], errors="coerce")
    scenes_df["t_end"] = pd.to_numeric(scenes_df["t_end"], errors="coerce")
    scenes_df["t_keyframe"] = pd.to_numeric(scenes_df["t_keyframe"], errors="coerce")

    # Pre-parse OCR words per frame for fast lookup
    ocr_word_sets = _build_ocr_word_index(ocr_df)

    # Search window from config
    search_window = getattr(cfg, 'TRACK_A_TEMPORAL_WINDOW', SEARCH_WINDOW)
    ocr_fuzzy_threshold = getattr(cfg, 'OCR_FUZZY_THRESHOLD', 0.8)

    # Process each keyword
    alignment_results = []
    diag_details = []

    # Check if SigLIP is available (for Case G)
    siglip_available = cfg.SIGLIP_ENABLED
    siglip_loaded = False

    for _, kw_row in keywords_df.iterrows():
        kw_id = kw_row["keyword_id"]
        kw_text = str(kw_row["keyword_text"])
        seg_id = kw_row["segment_id"]
        t_narr = float(kw_row["t_narr"]) if pd.notna(kw_row["t_narr"]) else None
        groundability = str(kw_row.get("groundability", "MEDIUM"))
        is_groundable = bool(kw_row.get("is_groundable", True))

        # Skip non-groundable keywords
        if not is_groundable:
            alignment_results.append(_no_match_result(
                kw_id, kw_text, seg_id, t_narr, groundability,
                reason="low_groundability"
            ))
            continue

        if t_narr is None:
            alignment_results.append(_no_match_result(
                kw_id, kw_text, seg_id, t_narr, groundability,
                reason="no_t_narr"
            ))
            continue

        # Find which scene contains t_narr
        current_scene = _find_scene_at_time(scenes_df, t_narr)

        # === Case A: OCR at t_narr (visual currently on screen) ===
        result = _case_a_visual_present(
            kw_text, t_narr, current_scene, ocr_word_sets, ocr_df,
            ocr_fuzzy_threshold
        )
        if result is not None:
            result.update({
                "keyword_id": kw_id, "keyword_text": kw_text,
                "segment_id": seg_id, "groundability": groundability,
            })
            alignment_results.append(result)
            diag_details.append({"kw_id": kw_id, "case": "A", "success": True})
            continue

        # === Case B: Forward search +10s (visual appears late) ===
        result = _case_b_forward_search(
            kw_text, t_narr, search_window, scenes_df, ocr_word_sets, ocr_df,
            ocr_fuzzy_threshold
        )
        if result is not None:
            result.update({
                "keyword_id": kw_id, "keyword_text": kw_text,
                "segment_id": seg_id, "groundability": groundability,
            })
            alignment_results.append(result)
            diag_details.append({"kw_id": kw_id, "case": "B", "success": True})
            continue

        # === Case C: Backward search -10s (visual expired) ===
        result = _case_c_backward_search(
            kw_text, t_narr, search_window, scenes_df, ocr_word_sets, ocr_df,
            ocr_fuzzy_threshold
        )
        if result is not None:
            result.update({
                "keyword_id": kw_id, "keyword_text": kw_text,
                "segment_id": seg_id, "groundability": groundability,
            })
            alignment_results.append(result)
            diag_details.append({"kw_id": kw_id, "case": "C", "success": True})
            continue

        # === Case G: SigLIP visual match (no OCR text, reduced alpha) ===
        if siglip_available and groundability in ("HIGH", "MEDIUM"):
            if not siglip_loaded:
                try:
                    from ..utils.siglip_utils import load_siglip
                    load_siglip(cfg.SIGLIP_MODEL_NAME, cfg.SIGLIP_PRETRAINED)
                    siglip_loaded = True
                except Exception as e:
                    logger.warning("SigLIP not available: %s", e)
                    siglip_available = False

            if siglip_available:
                result = _case_g_siglip_match(
                    kw_text, t_narr, search_window, scenes_df, cfg
                )
                if result is not None:
                    result.update({
                        "keyword_id": kw_id, "keyword_text": kw_text,
                        "segment_id": seg_id, "groundability": groundability,
                    })
                    alignment_results.append(result)
                    diag_details.append({"kw_id": kw_id, "case": "G", "success": True})
                    continue

        # === Case F: No visual correlate ===
        # Sub-classify: why did we fail to match?
        f_reason = _classify_no_match_reason(
            kw_text, t_narr, search_window, ocr_word_sets, current_scene
        )
        alignment_results.append(_no_match_result(
            kw_id, kw_text, seg_id, t_narr, groundability,
            reason=f_reason
        ))
        diag_details.append({"kw_id": kw_id, "case": "F", "success": False,
                             "reason": f_reason})

    # Unload SigLIP if loaded
    if siglip_loaded:
        from ..utils.siglip_utils import unload_siglip
        unload_siglip()

    # Build alignment dataframe
    kw_align_df = pd.DataFrame(alignment_results)

    # Compute per-segment aggregates
    seg_align_df = _compute_segment_aggregates(kw_align_df, segments_df)

    # Save outputs
    save_csv(kw_align_df, os.path.join(output_dir, "keyword_alignment.csv"))
    save_csv(seg_align_df, os.path.join(output_dir, "segment_alignment.csv"))

    # Log summary
    n_matched = len(kw_align_df[kw_align_df["match_case"] != "F"]) if not kw_align_df.empty else 0
    n_total = len(kw_align_df)
    logger.info(
        "Stage 4: %d/%d keywords matched (%.1f%%)",
        n_matched, n_total, 100 * n_matched / max(n_total, 1)
    )

    # Case distribution
    if not kw_align_df.empty:
        case_dist = kw_align_df["match_case"].value_counts().to_dict()
        logger.info("  Case distribution: %s", case_dist)

    # Diagnostics
    if diag is not None:
        _write_diagnostics(diag, kw_align_df, seg_align_df, diag_details)

    return {f: os.path.join(output_dir, f) for f in cache_files}


# =====================================================================
# OCR Word Index
# =====================================================================

def _build_ocr_word_index(ocr_df: pd.DataFrame) -> dict:
    """Build time → word_set index from ocr_per_frame.csv for fast lookup."""
    index = {}
    for _, row in ocr_df.iterrows():
        t = float(row["frame_time"]) if pd.notna(row["frame_time"]) else None
        if t is None:
            continue
        words_str = str(row.get("words", ""))
        if words_str and words_str != "nan":
            word_set = set(words_str.split())
        else:
            word_set = set()
        index[t] = word_set
    return index


def _find_ocr_at_time(t: float, ocr_word_sets: dict, tolerance: float = 1.0) -> set:
    """Find OCR words at time t (nearest frame within tolerance)."""
    best_t = None
    best_dist = float("inf")
    for frame_t in ocr_word_sets:
        dist = abs(frame_t - t)
        if dist <= tolerance and dist < best_dist:
            best_dist = dist
            best_t = frame_t
    if best_t is not None:
        return ocr_word_sets[best_t]
    return set()


def _get_scene_ocr_union(scene: pd.Series, ocr_word_sets: dict) -> set:
    """Get UNION of all OCR words across all frames within a scene.

    A scene represents one visual state (defined by scene detection).
    Using the union compensates for per-frame OCR noise — if text appears
    on ANY frame within the scene, it's present in that visual state.
    """
    if scene is None:
        return set()
    t_start = float(scene["t_start"])
    t_end = float(scene["t_end"])
    union = set()
    for frame_t, word_set in ocr_word_sets.items():
        if t_start <= frame_t <= t_end and word_set:
            union |= word_set
    return union


def _search_ocr_in_range(kw_text: str, t_start: float, t_end: float,
                         ocr_word_sets: dict, threshold: float) -> tuple:
    """Search for keyword in OCR frames within [t_start, t_end].

    Uses two strategies:
    1. Full multi-word match (all words present) — HIGH confidence
    2. Any content word match (at least one word ≥4 chars) — MEDIUM confidence

    Returns (frame_time, found, match_quality) where match_quality is
    "full" or "partial".
    """
    best_t_full = None
    best_dist_full = float("inf")
    best_t_partial = None
    best_dist_partial = float("inf")

    # Extract content words from keyword (≥4 chars, not stop words)
    kw_words = [normalize_word(w) for w in kw_text.split()
                if len(normalize_word(w)) >= 4]

    for frame_t, word_set in ocr_word_sets.items():
        if t_start <= frame_t <= t_end and word_set:
            # Strategy 1: Full multi-word match
            if fuzzy_match_multiword(kw_text, word_set, threshold):
                dist = frame_t - t_start
                if dist < best_dist_full:
                    best_dist_full = dist
                    best_t_full = frame_t

            # Strategy 2: Any content word match
            if kw_words and best_t_full is None:
                ocr_normalized = {normalize_word(w) for w in word_set if len(w) >= 3}
                for kw_w in kw_words:
                    if fuzzy_match_word(kw_w, ocr_normalized, threshold):
                        dist = frame_t - t_start
                        if dist < best_dist_partial:
                            best_dist_partial = dist
                            best_t_partial = frame_t
                        break

    # Prefer full match over partial
    if best_t_full is not None:
        return best_t_full, True, "full"
    if best_t_partial is not None:
        return best_t_partial, True, "partial"
    return None, False, None


# =====================================================================
# Scene Lookup
# =====================================================================

def _find_scene_at_time(scenes_df: pd.DataFrame, t: float) -> pd.Series:
    """Find the scene that contains time t."""
    if scenes_df.empty:
        return None
    mask = (scenes_df["t_start"] <= t) & (scenes_df["t_end"] >= t)
    matches = scenes_df[mask]
    if not matches.empty:
        return matches.iloc[0]
    # Fallback: nearest scene
    mid = (scenes_df["t_start"] + scenes_df["t_end"]) / 2
    closest_idx = (mid - t).abs().idxmin()
    return scenes_df.loc[closest_idx]


# =====================================================================
# Case A: Visual Present at t_narr
# =====================================================================

def _case_a_visual_present(kw_text: str, t_narr: float,
                           current_scene: pd.Series,
                           ocr_word_sets: dict, ocr_df: pd.DataFrame,
                           threshold: float) -> dict:
    """Case A: Keyword visible in current scene (visual on screen at t_narr).

    [v2.2 fix] Uses UNION of all OCR across the current scene instead of
    single-frame lookup. Rationale: a scene = one visual state. If text
    appears on ANY frame within the scene, it's present throughout.
    Single-frame OCR misses text due to per-frame noise (blur, partial
    detection during transitions). Scene-level union is robust.

    If found:
    - t_vis = scene.t_start (when visual first appeared)
    - delta_t = t_vis - t_narr ≤ 0 → Score 100 (no penalty)

    Sub-case D (progressive reveal): keyword appears mid-scene, not from
    start. Uses the earliest OCR frame that has the keyword.
    """
    if current_scene is None:
        return None

    scene_t_start = float(current_scene["t_start"])
    scene_t_end = float(current_scene["t_end"])

    # [v2.2] Use union of ALL OCR words in current scene (not single frame)
    # This compensates for per-frame OCR noise and gives maximum vocabulary
    scene_ocr_union = _get_scene_ocr_union(current_scene, ocr_word_sets)
    if not scene_ocr_union:
        return None

    # Try full multi-word match first, then individual word match
    is_full_match = fuzzy_match_multiword(kw_text, scene_ocr_union, threshold)
    is_word_match = False
    if not is_full_match:
        # Check if any content word (≥4 chars) from keyword appears in OCR
        kw_words = [normalize_word(w) for w in kw_text.split()
                    if len(normalize_word(w)) >= 4]
        ocr_normalized = {normalize_word(w) for w in scene_ocr_union if len(w) >= 3}
        for kw_w in kw_words:
            if fuzzy_match_word(kw_w, ocr_normalized, threshold):
                is_word_match = True
                break

    if not is_full_match and not is_word_match:
        return None

    # Keyword IS present in the current scene's visual state!

    # Determine match quality for alpha
    match_quality = "full" if is_full_match else "partial"
    alpha = 1.0 if is_full_match else 0.8  # partial word match slightly lower
    confidence = "HIGH" if is_full_match else "MEDIUM"

    # [v3] ALWAYS find the earliest frame where keyword appears in scene.
    # This preserves sub-scene temporal resolution and detects delays.
    # The scene OCR union is used only as a FILTER (does keyword exist
    # anywhere in scene?). The actual t_vis comes from frame-level lookup.
    earliest_t = _find_earliest_ocr_in_scene(
        kw_text, scene_t_start, scene_t_end, ocr_word_sets, threshold
    )

    if earliest_t is not None:
        t_vis = earliest_t
        # Case A vs D: was keyword present within first 1.0s of scene?
        # (tighter than old 30% check — 1.0s matches OCR sampling resolution)
        if t_vis <= scene_t_start + 1.0:
            match_case = "A"
            sub_scene_onset = None
        else:
            match_case = "D"
            sub_scene_onset = t_vis
    else:
        # Fallback: keyword in union but not on any individual frame
        # (OCR noise — word split across frames). Use scene start.
        t_vis = scene_t_start
        match_case = "A"
        sub_scene_onset = None

    delta_t = t_vis - t_narr

    return {
        "t_narr": t_narr,
        "t_vis": t_vis,
        "delta_t": delta_t,
        "match_case": match_case,
        "match_method": f"ocr_at_tnarr_{match_quality}",
        "alpha": alpha,
        "scene_id": int(current_scene.get("scene_id", -1)),
        "scene_t_start": scene_t_start,
        "scene_t_end": scene_t_end,
        "sub_scene_onset": sub_scene_onset,
        "confidence": confidence,
    }


def _find_earliest_ocr_in_scene(kw_text: str, scene_start: float,
                                scene_end: float, ocr_word_sets: dict,
                                threshold: float) -> float:
    """Find earliest frame in [scene_start, scene_end] with keyword.

    [v3] Changed: searches entire scene [scene_start, scene_end] instead of
    just [scene_start, t_narr]. This ensures we find the true first-appearance
    time even if it's after t_narr (for forward-looking within scene).
    """
    candidates = []
    kw_words = [normalize_word(w) for w in kw_text.split()
                if len(normalize_word(w)) >= 4]

    for frame_t, word_set in ocr_word_sets.items():
        if scene_start <= frame_t <= scene_end and word_set:
            # Full match
            if fuzzy_match_multiword(kw_text, word_set, threshold):
                candidates.append(frame_t)
            # Word-level match
            elif kw_words:
                ocr_norm = {normalize_word(w) for w in word_set if len(w) >= 3}
                for kw_w in kw_words:
                    if fuzzy_match_word(kw_w, ocr_norm, threshold):
                        candidates.append(frame_t)
                        break
    if candidates:
        return min(candidates)
    return None


# =====================================================================
# Case B: Forward Search (visual appears late)
# =====================================================================

def _case_b_forward_search(kw_text: str, t_narr: float,
                           search_window: float,
                           scenes_df: pd.DataFrame,
                           ocr_word_sets: dict, ocr_df: pd.DataFrame,
                           threshold: float) -> dict:
    """Case B: Keyword NOT at t_narr, found in later scene within +search_window.

    Scan forward through OCR frames in [t_narr, t_narr + search_window].
    If found: t_vis = that scene's t_start, delta_t > 0 → penalized.
    """
    t_end = t_narr + search_window

    # Search OCR frames forward
    first_frame_t, found, match_quality = _search_ocr_in_range(
        kw_text, t_narr + OCR_AT_TNARR_TOLERANCE, t_end,
        ocr_word_sets, threshold
    )

    if not found:
        return None

    # Find the scene that contains this frame
    scene = _find_scene_at_time(scenes_df, first_frame_t)
    if scene is None:
        return None

    # t_vis = scene start (when visual first appeared)
    t_vis = float(scene["t_start"])

    # But if keyword only appears mid-scene (not at scene start),
    # use the actual frame time
    ocr_at_scene_start = _find_ocr_at_time(t_vis, ocr_word_sets, OCR_AT_TNARR_TOLERANCE)
    if ocr_at_scene_start and fuzzy_match_multiword(kw_text, ocr_at_scene_start, threshold):
        t_vis = float(scene["t_start"])
    else:
        t_vis = first_frame_t  # progressive reveal in future scene

    delta_t = t_vis - t_narr  # > 0, penalized

    alpha = 1.0 if match_quality == "full" else 0.8
    confidence = "HIGH" if match_quality == "full" else "MEDIUM"

    return {
        "t_narr": t_narr,
        "t_vis": t_vis,
        "delta_t": delta_t,
        "match_case": "B",
        "match_method": f"ocr_forward_{match_quality}",
        "alpha": alpha,
        "scene_id": int(scene.get("scene_id", -1)),
        "scene_t_start": float(scene["t_start"]),
        "scene_t_end": float(scene["t_end"]),
        "sub_scene_onset": None,
        "confidence": confidence,
    }


# =====================================================================
# Case C: Backward Search (visual expired)
# =====================================================================

def _case_c_backward_search(kw_text: str, t_narr: float,
                            search_window: float,
                            scenes_df: pd.DataFrame,
                            ocr_word_sets: dict, ocr_df: pd.DataFrame,
                            threshold: float) -> dict:
    """Case C: Keyword was shown earlier but has disappeared.

    Search backward up to t_narr - search_window.
    If found in a scene that ENDED before t_narr:
    - Mark as visual_expired
    - delta_t = search_window (capped, severe violation)
    """
    t_start = t_narr - search_window

    # Search OCR frames backward
    best_frame_t = None
    best_dist = float("inf")

    for frame_t, word_set in ocr_word_sets.items():
        if t_start <= frame_t < t_narr - OCR_AT_TNARR_TOLERANCE and word_set:
            if fuzzy_match_multiword(kw_text, word_set, threshold):
                dist = t_narr - frame_t
                if dist < best_dist:
                    best_dist = dist
                    best_frame_t = frame_t

    if best_frame_t is None:
        return None

    # Find the scene that contained this frame
    scene = _find_scene_at_time(scenes_df, best_frame_t)
    if scene is None:
        return None

    scene_t_end = float(scene["t_end"])

    # Only mark as expired if the scene has ENDED before t_narr
    # (if scene still active, Case A should have caught it)
    if scene_t_end >= t_narr:
        # Scene still active — this shouldn't happen (Case A should catch)
        # But handle gracefully: treat as Case A
        return None

    # Visual expired — cap delta_t at search_window
    gap_since_expired = t_narr - scene_t_end
    delta_t = search_window  # capped penalty

    return {
        "t_narr": t_narr,
        "t_vis": float(scene["t_start"]),
        "delta_t": delta_t,
        "match_case": "C",
        "match_method": "ocr_backward_expired",
        "alpha": 0.8,  # slightly reduced confidence for expired visuals
        "scene_id": int(scene.get("scene_id", -1)),
        "scene_t_start": float(scene["t_start"]),
        "scene_t_end": scene_t_end,
        "sub_scene_onset": None,
        "confidence": "MEDIUM",
        "gap_since_expired": gap_since_expired,
    }


# =====================================================================
# Case G: SigLIP Visual Match (no OCR text)
# =====================================================================

def _case_g_siglip_match(kw_text: str, t_narr: float,
                         search_window: float,
                         scenes_df: pd.DataFrame,
                         cfg: Config) -> dict:
    """Case G: No OCR match but visual match via SigLIP.

    For animations, real-life demos, diagrams without text labels.
    Included with reduced alpha (0.3-0.5).
    """
    from ..utils.siglip_utils import embed_images, embed_texts, sigmoid_similarity
    from ..utils.siglip_utils import temporal_gaussian_weight

    # Get scenes within ±search_window of t_narr
    nearby = scenes_df[
        (scenes_df["t_keyframe"] >= t_narr - search_window) &
        (scenes_df["t_keyframe"] <= t_narr + search_window)
    ]

    if nearby.empty:
        return None

    # Get valid keyframe paths
    kf_paths = []
    kf_times = []
    kf_scene_ids = []
    for _, scene in nearby.iterrows():
        kf_path = str(scene.get("keyframe_path", ""))
        if kf_path and os.path.exists(kf_path):
            kf_paths.append(kf_path)
            kf_times.append(float(scene["t_keyframe"]))
            kf_scene_ids.append(int(scene["scene_id"]))

    if not kf_paths:
        return None

    # Embed keyword as visual query
    text_emb = embed_texts([f"A visual showing {kw_text}"])
    img_emb = embed_images(kf_paths)

    # Compute similarities with temporal decay
    raw_sims = sigmoid_similarity(img_emb, text_emb)[:, 0]
    sigma = cfg.TEMPORAL_SIGMA

    weighted_sims = np.array([
        raw_sims[j] * temporal_gaussian_weight(kf_times[j], t_narr, sigma)
        for j in range(len(raw_sims))
    ])

    best_idx = int(np.argmax(weighted_sims))
    best_sim = float(raw_sims[best_idx])

    # Threshold check
    min_sim = cfg.SIGLIP_KEYWORD_MIN_SIM
    if best_sim < min_sim:
        return None

    # Find the scene
    best_scene_id = kf_scene_ids[best_idx]
    scene = scenes_df[scenes_df["scene_id"] == best_scene_id].iloc[0]

    # t_vis = scene start
    t_vis = float(scene["t_start"])
    delta_t = t_vis - t_narr

    # Reduced alpha for non-OCR matches
    # Check if scene has OCR — if yes, down-weight further
    scene_ocr_words = str(scene.get("ocr_words", ""))
    if scene_ocr_words and scene_ocr_words.strip() and scene_ocr_words != "nan":
        alpha = cfg.SIGLIP_WEIGHT_WITH_OCR * 0.5  # 0.25 when OCR present but keyword not in OCR
    else:
        alpha = 0.4  # no OCR at all → moderate confidence

    # Scale alpha by similarity
    alpha = alpha * min(1.0, (best_sim - min_sim) / (0.6 - min_sim))
    alpha = max(0.1, min(0.5, alpha))

    return {
        "t_narr": t_narr,
        "t_vis": t_vis,
        "delta_t": delta_t,
        "match_case": "G",
        "match_method": "siglip_visual",
        "alpha": alpha,
        "scene_id": int(scene["scene_id"]),
        "scene_t_start": float(scene["t_start"]),
        "scene_t_end": float(scene["t_end"]),
        "sub_scene_onset": None,
        "confidence": "LOW",
        "siglip_sim": best_sim,
    }


# =====================================================================
# Case F Sub-Classification
# =====================================================================

def _classify_no_match_reason(kw_text: str, t_narr: float,
                              search_window: float,
                              ocr_word_sets: dict,
                              current_scene) -> str:
    """Sub-classify WHY no visual match was found (Case F).

    Distinguishes between:
    - no_ocr_nearby: No OCR text at all in ±search_window → likely talking head / animation
    - ocr_vocabulary_mismatch: OCR text exists but keyword not found → concept not on screen
    - no_visual_correlate: default / fallback
    """
    if t_narr is None:
        return "no_t_narr"

    # Check if there's ANY OCR text within ±search_window
    t_lo = t_narr - search_window
    t_hi = t_narr + search_window

    has_any_ocr = False
    for frame_t, word_set in ocr_word_sets.items():
        if t_lo <= frame_t <= t_hi and word_set:
            has_any_ocr = True
            break

    if not has_any_ocr:
        return "no_ocr_nearby"  # No text on screen at all → abstract/talking head
    else:
        return "ocr_vocabulary_mismatch"  # Text exists but keyword not in it


# =====================================================================
# No Match (Case F)
# =====================================================================

def _no_match_result(kw_id, kw_text, seg_id, t_narr, groundability,
                     reason="no_visual_correlate"):
    """Case F: No visual correlate found — excluded from TC scoring."""
    return {
        "keyword_id": kw_id,
        "keyword_text": kw_text,
        "segment_id": seg_id,
        "t_narr": t_narr,
        "t_vis": None,
        "delta_t": None,
        "match_case": "F",
        "match_method": reason,
        "alpha": 0.0,
        "scene_id": None,
        "scene_t_start": None,
        "scene_t_end": None,
        "sub_scene_onset": None,
        "confidence": "NONE",
        "groundability": groundability,
    }


# =====================================================================
# Segment-Level Aggregation
# =====================================================================

def _compute_segment_aggregates(kw_align_df: pd.DataFrame,
                                segments_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate keyword alignment results per segment."""
    if kw_align_df.empty:
        return pd.DataFrame()

    rows = []
    for seg_id in segments_df["segment_id"].unique():
        seg_kws = kw_align_df[kw_align_df["segment_id"] == seg_id]
        matched = seg_kws[seg_kws["match_case"] != "F"]
        unmatched = seg_kws[seg_kws["match_case"] == "F"]

        n_keywords = len(seg_kws)
        n_matched = len(matched)

        if n_matched > 0 and "delta_t" in matched.columns:
            delta_ts = matched["delta_t"].dropna().values
            alphas = matched["alpha"].dropna().values

            if len(delta_ts) > 0:
                # Alpha-weighted mean delta_t
                if len(alphas) == len(delta_ts) and np.sum(alphas) > 0:
                    mean_delta_t = float(np.average(delta_ts, weights=alphas))
                else:
                    mean_delta_t = float(np.mean(delta_ts))
                median_delta_t = float(np.median(delta_ts))
                min_delta_t = float(np.min(delta_ts))
                max_delta_t = float(np.max(delta_ts))
            else:
                mean_delta_t = median_delta_t = min_delta_t = max_delta_t = None
        else:
            mean_delta_t = median_delta_t = min_delta_t = max_delta_t = None

        # Case distribution
        case_counts = seg_kws["match_case"].value_counts().to_dict()

        seg_row = segments_df[segments_df["segment_id"] == seg_id]
        seg_text = str(seg_row.iloc[0]["text"]) if not seg_row.empty else ""

        rows.append({
            "segment_id": seg_id,
            "text_preview": seg_text[:80],
            "n_keywords": n_keywords,
            "n_matched": n_matched,
            "n_unmatched": len(unmatched),
            "match_rate": n_matched / max(n_keywords, 1),
            "mean_delta_t": mean_delta_t,
            "median_delta_t": median_delta_t,
            "min_delta_t": min_delta_t,
            "max_delta_t": max_delta_t,
            "n_case_A": case_counts.get("A", 0),
            "n_case_B": case_counts.get("B", 0),
            "n_case_C": case_counts.get("C", 0),
            "n_case_D": case_counts.get("D", 0),
            "n_case_G": case_counts.get("G", 0),
            "n_case_F": case_counts.get("F", 0),
        })

    return pd.DataFrame(rows)


# =====================================================================
# Diagnostics
# =====================================================================

def _write_diagnostics(diag, kw_align_df, seg_align_df, diag_details):
    """Write alignment diagnostics JSON."""
    if kw_align_df.empty:
        diag.write_json("stage4_alignment.json", {"n_keywords": 0})
        return

    # Case distribution
    case_dist = kw_align_df["match_case"].value_counts().to_dict()

    # Delta_t statistics (only matched keywords)
    matched = kw_align_df[kw_align_df["match_case"] != "F"]
    delta_ts = matched["delta_t"].dropna().values

    dt_stats = {}
    if len(delta_ts) > 0:
        dt_stats = {
            "mean": float(np.mean(delta_ts)),
            "median": float(np.median(delta_ts)),
            "std": float(np.std(delta_ts)),
            "min": float(np.min(delta_ts)),
            "max": float(np.max(delta_ts)),
            "n_negative": int(np.sum(delta_ts <= 0)),
            "n_positive": int(np.sum(delta_ts > 0)),
            "n_zero_or_near": int(np.sum(np.abs(delta_ts) <= 1.0)),
            "pct_visual_before_narration": float(
                100 * np.sum(delta_ts <= 0) / len(delta_ts)
            ),
        }

    # Confidence distribution
    conf_dist = kw_align_df["confidence"].value_counts().to_dict()

    # Alpha statistics
    alphas = matched["alpha"].dropna().values
    alpha_stats = {}
    if len(alphas) > 0:
        alpha_stats = {
            "mean": float(np.mean(alphas)),
            "median": float(np.median(alphas)),
            "min": float(np.min(alphas)),
            "max": float(np.max(alphas)),
        }

    # Per-segment summary
    seg_summaries = []
    if not seg_align_df.empty:
        for _, row in seg_align_df.iterrows():
            seg_summaries.append({
                "segment_id": int(row["segment_id"]),
                "n_keywords": int(row["n_keywords"]),
                "n_matched": int(row["n_matched"]),
                "match_rate": float(row["match_rate"]),
                "mean_delta_t": row["mean_delta_t"],
                "cases": {
                    "A": int(row["n_case_A"]), "B": int(row["n_case_B"]),
                    "C": int(row["n_case_C"]), "D": int(row["n_case_D"]),
                    "G": int(row["n_case_G"]), "F": int(row["n_case_F"]),
                }
            })

    diag.write_json("stage4_alignment.json", {
        "n_keywords_total": len(kw_align_df),
        "n_matched": len(matched),
        "n_unmatched": int(case_dist.get("F", 0)),
        "match_rate_pct": float(100 * len(matched) / max(len(kw_align_df), 1)),
        "case_distribution": case_dist,
        "confidence_distribution": conf_dist,
        "delta_t_statistics": dt_stats,
        "alpha_statistics": alpha_stats,
        "segment_summaries": seg_summaries,
    })
