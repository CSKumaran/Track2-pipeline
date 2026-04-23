"""Stage 5: Keyword Extraction & Timestamp Assignment [v4.0 Transcript-First].

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

    # [v4.0] First-mention detection
    if not kw_df.empty:
        first_mention_window = getattr(cfg, 'FIRST_MENTION_WINDOW_S', 60.0)
        kw_df = _add_first_mention_flags(kw_df, first_mention_window)

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

        n_first_mention = int(kw_df["is_first_mention"].sum()) if "is_first_mention" in kw_df.columns else 0

        diag.write_json("stage5_keywords.json", {
            "n_segments_total": len(segments_df),
            "n_segments_with_keywords": n_segments_with_kw,
            "n_keywords_total": n_total,
            "n_groundable": n_groundable,
            "n_low_groundability": n_total - n_groundable,
            "n_first_mention": n_first_mention,
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


def _add_first_mention_flags(kw_df: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """[v4.0] Add is_first_mention column to keywords dataframe.

    A keyword is a first mention if its lemma (or close morphological variant)
    has NOT appeared in any earlier segment beyond the lookback window.

    Uses spaCy lemmatization for robust matching across morphological forms
    (e.g. "optimize" / "optimization" / "optimized" share the lemma "optimize").
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception:
        logger.warning("[v4.0] spaCy model not available, falling back to surface-form first-mention")
        nlp = None

    # Sort by t_narr to process in temporal order
    kw_sorted = kw_df.sort_values("t_narr").copy()

    # Build lemma for each keyword
    if nlp is not None:
        # Batch lemmatize all keywords at once for efficiency
        all_texts = kw_sorted["keyword_text"].tolist()
        lemmas = []
        for doc in nlp.pipe(all_texts, batch_size=64):
            # Use set of token lemmas (handles multi-word keywords)
            lemma_set = frozenset(tok.lemma_.lower() for tok in doc
                                  if not tok.is_stop and len(tok.text) >= 3)
            lemmas.append(lemma_set)
        kw_sorted["_lemma_set"] = lemmas
    else:
        # Fallback: lowercase surface forms
        kw_sorted["_lemma_set"] = kw_sorted["keyword_text"].apply(
            lambda t: frozenset(w.lower() for w in t.split() if len(w) >= 3)
        )

    # Track seen lemma sets with their earliest t_narr
    seen_lemmas = {}  # lemma_set -> earliest t_narr
    is_first_mention = []

    for _, row in kw_sorted.iterrows():
        lemma_set = row["_lemma_set"]
        t_narr = row["t_narr"]

        if not lemma_set:
            is_first_mention.append(False)
            continue

        # Check if any matching lemma set was seen within the window
        found_prior = False
        for seen_set, seen_time in seen_lemmas.items():
            # Match if substantial overlap (>50% of lemmas shared)
            if seen_set and lemma_set:
                overlap = len(seen_set & lemma_set)
                min_len = min(len(seen_set), len(lemma_set))
                if min_len > 0 and overlap / min_len >= 0.5:
                    # Prior occurrence exists — but is it within the window?
                    if t_narr - seen_time <= window_s:
                        found_prior = True
                        break

        is_first_mention.append(not found_prior)

        # Update seen set with this keyword's lemmas (track earliest time)
        if lemma_set not in seen_lemmas:
            seen_lemmas[lemma_set] = t_narr

    kw_sorted["is_first_mention"] = is_first_mention

    # Drop temp column, restore original order
    kw_sorted = kw_sorted.drop(columns=["_lemma_set"])
    result = kw_sorted.sort_values("keyword_id").reset_index(drop=True)

    n_first = result["is_first_mention"].sum()
    logger.info("[v4.0] First-mention detection: %d/%d keywords are first mentions",
                n_first, len(result))

    return result
