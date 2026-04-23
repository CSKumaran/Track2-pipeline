"""Stage 4a: Text Units & Embeddings."""

import logging
import os
import numpy as np
import pandas as pd

from ..config import Config
from ..utils.embedding_utils import embed_texts, unload_embedding_model
from ..utils.io_utils import save_csv, save_npy, cache_exists

logger = logging.getLogger(__name__)


def run_stage4a(output_dir: str, cfg: Config) -> dict:
    """Compute segment embeddings. Returns paths to outputs."""
    cache_files = ["segment_meta.csv", "segment_embeddings.npy"]
    if cache_exists(output_dir, cache_files):
        logger.info("Stage 4a: cache hit, skipping embeddings")
        return {f: os.path.join(output_dir, f) for f in cache_files}

    seg_path = os.path.join(output_dir, "transcript_segments_improved.csv")
    segments_df = pd.read_csv(seg_path)

    if segments_df.empty:
        logger.warning("No segments to embed")
        meta = pd.DataFrame(columns=["segment_id", "text", "start_time", "end_time", "t_mid"])
        save_csv(meta, os.path.join(output_dir, "segment_meta.csv"))
        save_npy(np.array([]), os.path.join(output_dir, "segment_embeddings.npy"))
        return {f: os.path.join(output_dir, f) for f in cache_files}

    # Compute midpoints
    segments_df["t_mid"] = (
        pd.to_numeric(segments_df["start_time"], errors="coerce")
        + pd.to_numeric(segments_df["end_time"], errors="coerce")
    ) / 2.0

    # Embed segments
    texts = segments_df["text"].astype(str).tolist()
    logger.info("Embedding %d segments with %s", len(texts), cfg.EMBEDDING_MODEL_NAME)
    embeddings = embed_texts(texts, cfg.EMBEDDING_MODEL_NAME)

    # Save
    meta = segments_df[["segment_id", "text", "start_time", "end_time", "t_mid"]].copy()
    save_csv(meta, os.path.join(output_dir, "segment_meta.csv"))
    save_npy(embeddings, os.path.join(output_dir, "segment_embeddings.npy"))

    logger.info("Stage 4a: %d segments embedded, shape=%s", len(texts), embeddings.shape)
    return {f: os.path.join(output_dir, f) for f in cache_files}
