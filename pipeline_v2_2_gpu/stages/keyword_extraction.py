"""Stage 5: Keyword Extraction & Timestamp Assignment [v2.2 Transcript-First].

In the transcript-first architecture, this stage runs BEFORE alignment (Stage 4).
It extracts keywords from each transcript segment and assigns word-level timestamps
(t_narr) using WhisperX forced alignment data.

Visual grounding (finding where keywords appear visually) happens in Stage 4 (Alignment).

Outputs:
    keywords.csv — keyword_id, keyword_text, segment_id, t_narr, groundability, ...
    stage5_keywords.json — diagnostics
"""

import logging
import os
import numpy as np
import pandas as pd

from ..config import Config
from ..utils.keyword_utils import (
    extract_keywords_spacy, extract_keywords_keybert, merge_keywords,
    classify_groundability, load_concreteness_db, unload_keyword_models,
)
from ..utils.ocr_utils import normalize_word
from ..utils.io_utils import save_csv, cache_exists

logger = logging.getLogger(__name__)


def run_stage5(output_dir: str, cfg: Config, diag=None) -> dict:
    """Extract keywords from transcript segments and assign word-level timestamps.

    This is the first step in the transcript-first pipeline:
    For each segment → extract keywords → find t_narr from word timestamps.

    Returns dict of output file paths.
    """
    cache_files = ["keywords.csv"]
    if cache_exists(output_dir, cache_files):
        logger.info("Stage 5: cache hit, skipping keyword extraction")
        return {f: os.path.join(output_dir, f) for f in cache_files}

    # Load data
    segments_df = pd.read_csv(
        os.path.join(output_dir, "transcript_segments_improved.csv")
    )
    words_df = pd.read_csv(os.path.join(output_dir, "transcript_words.csv"))

    # Load concreteness DB for groundability classification
    concreteness_db = load_concreteness_db()

    all_keywords = []
    kw_global_id = 0

    # Per-segment diagnostics
    diag_segments = []

    for _, seg in segments_df.iterrows():
        seg_id = seg["segment_id"]
        text = str(seg["text"]) if pd.notna(seg["text"]) else ""
        seg_start = float(seg["start_time"]) if pd.notna(seg.get("start_time")) else 0
        seg_end = float(seg["end_time"]) if pd.notna(seg.get("end_time")) else 0

        if not text.strip():
            diag_segments.append({
                "segment_id": seg_id, "n_spacy": 0, "n_keybert": 0,
                "n_merged": 0, "n_groundable": 0,
            })
            continue

        # --- 5a: Extract keywords ---
        spacy_kws = extract_keywords_spacy(text)

        keybert_kws = []
        if cfg.KEYWORD_USE_KEYBERT and cfg.KEYWORD_ENABLED:
            keybert_kws = extract_keywords_keybert(
                text, cfg.EMBEDDING_MODEL_NAME, cfg.KEYWORD_KEYBERT_TOP_N
            )

        keywords = merge_keywords(spacy_kws, keybert_kws, cfg.KEYWORD_MIN_LENGTH)

        n_groundable = 0

        for kw in keywords:
            # --- 5b: Find word-level timestamp (t_narr) ---
            t_narr, ts_reliable, match_info = _find_keyword_timestamp(
                kw, words_df, seg_start, seg_end
            )

            # --- 5c: Groundability classification ---
            groundability = classify_groundability(kw, concreteness_db)
            is_groundable = groundability != "LOW"
            if is_groundable:
                n_groundable += 1

            all_keywords.append({
                "keyword_id": kw_global_id,
                "keyword_text": kw,
                "segment_id": seg_id,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "t_narr": t_narr,
                "t_narr_method": match_info["method"],
                "timestamp_reliable": ts_reliable,
                "groundability": groundability,
                "is_groundable": is_groundable,
                "n_words": len(kw.split()),
                "source": match_info.get("source", "spacy"),
            })
            kw_global_id += 1

        diag_segments.append({
            "segment_id": seg_id,
            "text_preview": text[:80],
            "n_spacy": len(spacy_kws),
            "n_keybert": len(keybert_kws),
            "n_merged": len(keywords),
            "n_groundable": n_groundable,
            "keywords": [kw for kw in keywords],
        })

    # Build output dataframe
    kw_df = pd.DataFrame(all_keywords)

    # Add global frequency: how many segments mention each keyword
    if not kw_df.empty:
        freq = kw_df.groupby("keyword_text")["segment_id"].nunique().rename("global_segment_freq")
        kw_df = kw_df.merge(freq, on="keyword_text", how="left")
    else:
        kw_df["global_segment_freq"] = []

    # Save
    save_csv(kw_df, os.path.join(output_dir, "keywords.csv"))

    # Unload models to free GPU memory
    unload_keyword_models()

    # Stats
    n_total = len(kw_df)
    n_groundable = len(kw_df[kw_df["is_groundable"]]) if not kw_df.empty else 0
    n_segments_with_kw = kw_df["segment_id"].nunique() if not kw_df.empty else 0

    logger.info(
        "Stage 5: %d keywords from %d segments (%d groundable, %d LOW skipped)",
        n_total, n_segments_with_kw, n_groundable, n_total - n_groundable
    )

    # Diagnostics
    if diag is not None:
        groundability_dist = {}
        if not kw_df.empty:
            groundability_dist = kw_df["groundability"].value_counts().to_dict()

        t_narr_method_dist = {}
        if not kw_df.empty:
            t_narr_method_dist = kw_df["t_narr_method"].value_counts().to_dict()

        diag.write_json("stage5_keywords.json", {
            "n_segments_total": len(segments_df),
            "n_segments_with_keywords": n_segments_with_kw,
            "n_keywords_total": n_total,
            "n_groundable": n_groundable,
            "n_low_groundability": n_total - n_groundable,
            "groundability_distribution": groundability_dist,
            "t_narr_method_distribution": t_narr_method_dist,
            "keywords_per_segment": {
                "mean": float(np.mean([s["n_merged"] for s in diag_segments])) if diag_segments else 0,
                "median": float(np.median([s["n_merged"] for s in diag_segments])) if diag_segments else 0,
                "max": max([s["n_merged"] for s in diag_segments]) if diag_segments else 0,
                "min": min([s["n_merged"] for s in diag_segments]) if diag_segments else 0,
            },
            "segment_details": diag_segments,
        })

    return {f: os.path.join(output_dir, f) for f in cache_files}


def _find_keyword_timestamp(keyword: str, words_df: pd.DataFrame,
                            seg_start: float, seg_end: float) -> tuple:
    """Find keyword's word-level timestamp from WhisperX forced alignment.

    Search strategy:
    1. Exact match of keyword's first word in transcript_words.csv within segment range
    2. Fuzzy match (normalized) if exact fails
    3. Fallback: segment midpoint

    Returns:
        (t_narr, timestamp_reliable, match_info)
    """
    if words_df.empty:
        t_mid = (seg_start + seg_end) / 2
        return t_mid, True, {"method": "fallback_empty_words", "source": "spacy"}

    kw_parts = keyword.lower().split()
    first_word = kw_parts[0]
    first_word_norm = normalize_word(first_word)

    # Search within segment time range (±1s buffer for alignment jitter)
    times = pd.to_numeric(words_df["start_time"], errors="coerce")
    mask = (times >= seg_start - 1.0) & (times <= seg_end + 1.0)
    seg_words = words_df[mask]

    if seg_words.empty:
        t_mid = (seg_start + seg_end) / 2
        return t_mid, True, {"method": "fallback_no_words_in_range", "source": "spacy"}

    # Strategy 1: Exact match on first word
    for _, w in seg_words.iterrows():
        w_text = str(w["word"]).strip().lower()
        if w_text == first_word:
            t_narr = float(w["start_time"]) if pd.notna(w["start_time"]) else None
            if t_narr is not None:
                ts_reliable = bool(w.get("timestamp_reliable", True))
                return t_narr, ts_reliable, {"method": "exact_word_match", "source": "spacy"}

    # Strategy 2: Normalized match (strip punctuation)
    if len(first_word_norm) >= 3:
        for _, w in seg_words.iterrows():
            w_norm = normalize_word(str(w["word"]))
            if w_norm == first_word_norm:
                t_narr = float(w["start_time"]) if pd.notna(w["start_time"]) else None
                if t_narr is not None:
                    ts_reliable = bool(w.get("timestamp_reliable", True))
                    return t_narr, ts_reliable, {"method": "normalized_match", "source": "spacy"}

    # Strategy 3: Substring match (keyword part appears within a transcript word)
    if len(first_word_norm) >= 4:
        for _, w in seg_words.iterrows():
            w_norm = normalize_word(str(w["word"]))
            if first_word_norm in w_norm or w_norm in first_word_norm:
                t_narr = float(w["start_time"]) if pd.notna(w["start_time"]) else None
                if t_narr is not None:
                    ts_reliable = bool(w.get("timestamp_reliable", True))
                    return t_narr, ts_reliable, {"method": "substring_match", "source": "spacy"}

    # Fallback: segment midpoint
    t_mid = (seg_start + seg_end) / 2
    return t_mid, True, {"method": "fallback_midpoint", "source": "spacy"}
