"""CLIP-based visual alignment and scene classification utilities.

Uses OpenAI CLIP (ViT-B/16) to:
  - Embed frame images and transcript text in a shared space
  - Compute frame-to-transcript alignment (Track B)
  - Zero-shot classify scene types for dashboard display
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from pipeline.config import Config

logger = logging.getLogger(__name__)

# ── Lazy-loaded CLIP model (singleton) ───────────────────────────────
_clip_model = None
_clip_preprocess = None
_clip_device = "cpu"


def _load_clip(model_name: str = "ViT-B/16"):
    """Lazy-load CLIP model + preprocess transform, cached globally."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        return _clip_model, _clip_preprocess

    import clip as clip_lib
    import torch

    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading CLIP %s on %s …", model_name, _clip_device)
    _clip_model, _clip_preprocess = clip_lib.load(model_name, device=_clip_device)
    _clip_model.eval()
    logger.info("CLIP %s loaded successfully.", model_name)
    return _clip_model, _clip_preprocess


# ── Image / text embeddings ──────────────────────────────────────────

def get_clip_image_embedding(image_path: str, config: "Config") -> np.ndarray:
    """Encode a single image with CLIP vision encoder → (512,) float32."""
    import torch
    from PIL import Image

    model, preprocess = _load_clip(config.CLIP_MODEL_NAME)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(_clip_device)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
    return emb.cpu().numpy().astype(np.float32).flatten()


def get_clip_text_embeddings(texts: list[str], config: "Config") -> np.ndarray:
    """Encode a list of texts with CLIP text encoder → (N, 512) float32.

    CLIP tokenizer has a 77-token limit; long texts are truncated automatically.
    """
    import clip as clip_lib
    import torch

    model, _ = _load_clip(config.CLIP_MODEL_NAME)
    tokens = clip_lib.tokenize(texts, truncate=True).to(_clip_device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


# ── Scene type classification ────────────────────────────────────────

_SCENE_TYPE_LABELS = [
    ("text_slide", "a text slide with words and bullet points"),
    ("diagram", "a technical diagram with labels and arrows"),
    ("animation", "an animation or visual simulation"),
    ("real_world", "a real-world photograph or video footage"),
    ("code", "a code editor or programming interface"),
]


def classify_scene_type(
    image_path: str, config: "Config",
) -> dict:
    """Zero-shot scene classification using CLIP.

    Returns dict with 'scene_type' (str) and 'scene_type_conf' (float).
    """
    if not os.path.exists(image_path):
        return {"scene_type": "unknown", "scene_type_conf": 0.0}

    img_emb = get_clip_image_embedding(image_path, config)  # (512,)
    label_texts = [desc for _, desc in _SCENE_TYPE_LABELS]
    txt_embs = get_clip_text_embeddings(label_texts, config)  # (5, 512)

    sims = (img_emb @ txt_embs.T).flatten()  # cosine sim (already L2-normed)
    best_idx = int(np.argmax(sims))
    return {
        "scene_type": _SCENE_TYPE_LABELS[best_idx][0],
        "scene_type_conf": round(float(sims[best_idx]), 4),
    }


# ── CLIP alignment (Track B) ────────────────────────────────────────

def compute_clip_alignment(
    image_path: str,
    segment_texts: list[str],
    t_segs: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    t_vis: float,
    temporal_sigma: float,
    words_df,
    config: "Config",
) -> dict:
    """Align a scene keyframe to transcript segments using CLIP.

    Steps:
      1. Embed keyframe image with CLIP vision encoder
      2. Embed all transcript segments with CLIP text encoder
      3. Apply temporal Gaussian decay (same σ as semantic track)
      4. Find best matching segment
      5. Word-level drill-down within best segment

    Returns dict with: clip_sim, clip_best_seg_id, clip_t_narr, clip_delta_t,
                       clip_word_window, clip_best_seg_text
    """
    from pipeline.stages.text_units import get_words_for_segment, build_word_windows

    empty = {
        "clip_sim": 0.0,
        "clip_best_seg_id": None,
        "clip_best_seg_text": None,
        "clip_t_narr": None,
        "clip_delta_t": None,
        "clip_word_window": None,
    }

    if not os.path.exists(image_path) or len(segment_texts) == 0:
        return empty

    # Step 1-2: embed image + segments
    img_emb = get_clip_image_embedding(image_path, config)  # (512,)
    txt_embs = get_clip_text_embeddings(segment_texts, config)  # (N, 512)

    # Step 3: raw similarity + temporal decay
    raw_sims = (img_emb @ txt_embs.T).flatten()
    time_offsets = np.abs(t_segs - t_vis)
    temporal_weights = np.exp(-0.5 * (time_offsets / temporal_sigma) ** 2)
    weighted_sims = raw_sims * temporal_weights

    best_idx = int(np.argmax(weighted_sims))
    clip_sim = float(raw_sims[best_idx])
    clip_best_seg_text = segment_texts[best_idx]

    # Step 5: word-level drill-down
    t_narr = float(t_segs[best_idx])
    clip_word_window = None

    if words_df is not None and len(words_df) > 0:
        seg_start = float(seg_starts[best_idx])
        seg_end = float(seg_ends[best_idx])
        words_in_seg = get_words_for_segment(words_df, seg_start, seg_end)
        word_windows = build_word_windows(words_in_seg, window_size=3)

        if word_windows:
            ww_texts = [ww["text"] for ww in word_windows]
            ww_embs = get_clip_text_embeddings(ww_texts, config)
            ww_sims = (img_emb @ ww_embs.T).flatten()
            best_ww_idx = int(np.argmax(ww_sims))
            t_narr = word_windows[best_ww_idx]["t_center"]
            clip_word_window = word_windows[best_ww_idx]["text"]

    return {
        "clip_sim": round(clip_sim, 4),
        "clip_best_seg_id": best_idx,
        "clip_best_seg_text": clip_best_seg_text,
        "clip_t_narr": round(t_narr, 4),
        "clip_delta_t": round(t_narr - t_vis, 4),
        "clip_word_window": clip_word_window,
    }
