"""Stage 4a – Build text units (segment + word-level metadata & embeddings)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pipeline.utils.embedding_utils import get_text_embeddings
from pipeline.utils.io_utils import ensure_dir, safe_write_csv

if TYPE_CHECKING:
    from pipeline.config import Config

logger = logging.getLogger(__name__)


def build_text_units_and_embeddings(
    transcript_segments_path: str,
    output_dir: str,
    config: "Config",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load transcript segments, compute midpoint and embeddings.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        (segment_meta DataFrame, embeddings array of shape (N, d)).
    """
    ensure_dir(output_dir)

    seg_df = pd.read_csv(transcript_segments_path)
    seg_df["t_seg"] = 0.5 * (seg_df["start_time"] + seg_df["end_time"])

    # Compute segment-level embeddings
    texts = seg_df["text"].astype(str).tolist()
    embeddings = get_text_embeddings(texts, config)

    # Save outputs
    meta_path = os.path.join(output_dir, "segment_meta.csv")
    safe_write_csv(
        seg_df[["segment_id", "text", "start_time", "end_time", "t_seg"]],
        meta_path,
    )

    emb_path = os.path.join(output_dir, "segment_embeddings.npy")
    np.save(emb_path, embeddings)
    logger.info("Saved embeddings (%s) → %s", embeddings.shape, emb_path)

    return seg_df, embeddings


def load_word_level_data(
    output_dir: str,
) -> pd.DataFrame:
    """Load transcript_words.csv and return as DataFrame.

    Expected columns: word_id, word, start_time, end_time.
    """
    words_path = os.path.join(output_dir, "transcript_words.csv")
    words_df = pd.read_csv(words_path)
    # Compute per-word midpoint
    words_df["t_word"] = 0.5 * (words_df["start_time"] + words_df["end_time"])
    logger.info("Loaded %d words from %s", len(words_df), words_path)
    return words_df


def get_words_for_segment(
    words_df: pd.DataFrame,
    seg_start: float,
    seg_end: float,
    tolerance: float = 0.3,
) -> pd.DataFrame:
    """Return words that fall within a segment's time range.

    Uses word start_time with a small tolerance to handle boundary cases.
    """
    mask = (
        (words_df["start_time"] >= seg_start - tolerance)
        & (words_df["start_time"] <= seg_end + tolerance)
    )
    return words_df[mask].copy()


def build_word_windows(
    words_in_segment: pd.DataFrame,
    window_size: int = 3,
) -> list[dict]:
    """Build sliding windows of consecutive words within a segment.

    Parameters
    ----------
    words_in_segment : pd.DataFrame
        Words belonging to one segment (sorted by time).
    window_size : int
        Number of consecutive words per window.

    Returns
    -------
    list[dict]
        Each dict has: 'text' (joined words), 't_center' (mean of word
        midpoints), 'start_time', 'end_time'.
    """
    words_sorted = words_in_segment.sort_values("start_time").reset_index(drop=True)
    n = len(words_sorted)

    if n == 0:
        return []

    # If fewer words than window_size, use all words as one window
    if n <= window_size:
        text = " ".join(words_sorted["word"].astype(str).tolist())
        t_center = words_sorted["t_word"].mean()
        return [{
            "text": text,
            "t_center": float(t_center),
            "start_time": float(words_sorted["start_time"].iloc[0]),
            "end_time": float(words_sorted["end_time"].iloc[-1]),
        }]

    windows = []
    for i in range(n - window_size + 1):
        chunk = words_sorted.iloc[i : i + window_size]
        text = " ".join(chunk["word"].astype(str).tolist())
        t_center = chunk["t_word"].mean()
        windows.append({
            "text": text,
            "t_center": float(t_center),
            "start_time": float(chunk["start_time"].iloc[0]),
            "end_time": float(chunk["end_time"].iloc[-1]),
        })

    return windows
