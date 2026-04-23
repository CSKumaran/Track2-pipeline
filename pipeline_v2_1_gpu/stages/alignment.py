"""Stage 4b: Narration Alignment (3-Track Cascade)."""

import logging
import os
import numpy as np
import pandas as pd

from ..config import Config
from ..utils.io_utils import save_csv, cache_exists, load_npy
from ..utils.ocr_utils import normalize_word

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "as", "into", "about", "between",
    "through", "after", "before", "during", "and", "but", "or", "nor",
    "not", "so", "yet", "both", "either", "neither", "this", "that",
    "these", "those", "it", "its", "we", "you", "he", "she", "they",
    "i", "me", "my", "our", "your", "his", "her", "their", "them",
}


def run_stage4b(output_dir: str, cfg: Config) -> dict:
    """Run 3-track alignment cascade. Returns path to alignment_events.csv."""
    cache_file = "alignment_events.csv"
    if cache_exists(output_dir, [cache_file]):
        logger.info("Stage 4b: cache hit, skipping alignment")
        return {cache_file: os.path.join(output_dir, cache_file)}

    scenes_df = pd.read_csv(os.path.join(output_dir, "scenes.csv"))
    concepts_df = pd.read_csv(os.path.join(output_dir, "scene_concepts.csv"))
    words_df = pd.read_csv(os.path.join(output_dir, "transcript_words.csv"))
    seg_meta = pd.read_csv(os.path.join(output_dir, "segment_meta.csv"))
    seg_embeddings = load_npy(os.path.join(output_dir, "segment_embeddings.npy"))

    if scenes_df.empty or seg_meta.empty:
        logger.warning("No scenes or segments for alignment")
        empty = pd.DataFrame()
        save_csv(empty, os.path.join(output_dir, cache_file))
        return {cache_file: os.path.join(output_dir, cache_file)}

    # Merge scenes with concepts
    scenes = scenes_df.merge(concepts_df, on="scene_id", how="left")

    results = []
    for _, scene in scenes.iterrows():
        scene_id = scene["scene_id"]
        # t_vis = when visual first appears (scene start), NOT keyframe centroid
        t_vis = float(scene["t_start"])
        t_kf = float(scene["t_keyframe"])  # still used for concept extraction/matching
        concept_text = str(scene.get("concept_text", ""))
        is_content = scene.get("is_content", True)
        frame_type = str(scene.get("frame_type", ""))
        frame_type_conf = float(scene.get("frame_type_confidence", 0))

        if not is_content:
            results.append(_no_match_result(scene_id, t_vis, concept_text,
                                            frame_type, frame_type_conf,
                                            reason="non_content"))
            continue

        # Track A: OCR word matching + context validation
        result = _track_a(scene, words_df, seg_meta, seg_embeddings, cfg)
        if result is not None:
            # Recompute delta_t relative to t_vis (scene start)
            if result.get("t_narr") is not None:
                result["delta_t"] = result["t_narr"] - t_vis
            result["t_vis"] = t_vis
            result.update({"scene_id": scene_id, "scene_type": frame_type,
                          "scene_type_conf": frame_type_conf})
            results.append(result)
            continue

        # Track B: SigLIP vision-to-text
        if cfg.SIGLIP_ENABLED:
            result = _track_b(scene, seg_meta, cfg, output_dir)
            if result is not None:
                if result.get("t_narr") is not None:
                    result["delta_t"] = result["t_narr"] - t_vis
                result["t_vis"] = t_vis
                result.update({"scene_id": scene_id, "scene_type": frame_type,
                              "scene_type_conf": frame_type_conf})
                results.append(result)
                continue

        # Track C: Semantic cosine similarity
        result = _track_c(scene, seg_meta, seg_embeddings, cfg)
        if result is not None:
            if result.get("t_narr") is not None:
                result["delta_t"] = result["t_narr"] - t_vis
            result["t_vis"] = t_vis
            result.update({"scene_id": scene_id, "scene_type": frame_type,
                          "scene_type_conf": frame_type_conf})
            results.append(result)
            continue

        # No match
        results.append(_no_match_result(scene_id, t_vis, concept_text,
                                        frame_type, frame_type_conf))

    df = pd.DataFrame(results)
    save_csv(df, os.path.join(output_dir, cache_file))
    logger.info("Stage 4b: %d alignment events", len(df))
    return {cache_file: os.path.join(output_dir, cache_file)}


def _track_a(scene, words_df, seg_meta, seg_embeddings, cfg):
    """Track A: OCR word matching + context validation."""
    from ..utils.embedding_utils import embed_texts, cosine_similarity_matrix

    t_kf = float(scene["t_keyframe"])
    concept_text = str(scene.get("concept_text", ""))
    ocr_words_str = str(scene.get("ocr_words", ""))

    if not ocr_words_str.strip():
        return None

    ocr_words = [w for w in ocr_words_str.split() if len(w) >= 3 and w not in STOP_WORDS]
    if not ocr_words:
        return None

    # Search transcript words within ±30s
    window = cfg.TRACK_A_TEMPORAL_WINDOW
    mask = (
        (pd.to_numeric(words_df["start_time"], errors="coerce") >= t_kf - window)
        & (pd.to_numeric(words_df["start_time"], errors="coerce") <= t_kf + window)
    )
    nearby_words = words_df[mask].copy()
    if nearby_words.empty:
        return None

    # Match OCR words in transcript
    matches = []
    for ocr_w in ocr_words:
        ocr_norm = normalize_word(ocr_w)
        for _, tw in nearby_words.iterrows():
            tw_norm = normalize_word(str(tw["word"]))
            if ocr_norm == tw_norm and len(ocr_norm) >= 3:
                t_word = float(tw["start_time"])
                matches.append(t_word - t_kf)
                break

    if not matches:
        return None

    # Context validation: check semantic similarity
    if len(seg_meta) > 0 and len(seg_embeddings) > 0 and concept_text.strip():
        concept_emb = embed_texts([concept_text], cfg.EMBEDDING_MODEL_NAME)

        # Find the segment closest to median match
        t_narr_candidate = t_kf + float(np.median(matches))
        t_mids = pd.to_numeric(seg_meta["t_mid"], errors="coerce").values
        closest_seg_idx = int(np.argmin(np.abs(t_mids - t_narr_candidate)))

        if closest_seg_idx < len(seg_embeddings):
            seg_emb = seg_embeddings[closest_seg_idx:closest_seg_idx + 1]
            sim = cosine_similarity_matrix(concept_emb, seg_emb)[0, 0]
            if sim < cfg.TRACK_A_CONTEXT_MIN_SIM:
                return None  # Context validation failed

    delta_t = float(np.median(matches))
    t_narr = t_kf + delta_t

    return {
        "t_keyframe": t_kf,
        "concept_text": concept_text,
        "match_type": "word_exact",
        "match_track": "A",
        "t_narr": t_narr,
        "delta_t": delta_t,
        "n_word_matches": len(matches),
        "siglip_sim": None,
        "semantic_sim": None,
        "alpha": 1.0,
    }


def _track_b(scene, seg_meta, cfg, output_dir):
    """Track B: SigLIP vision-to-text."""
    from ..utils.siglip_utils import embed_images, embed_texts as siglip_embed_texts
    from ..utils.siglip_utils import sigmoid_similarity, temporal_gaussian_weight

    t_kf = float(scene["t_keyframe"])
    kf_path = scene["keyframe_path"]
    concept_text = str(scene.get("concept_text", ""))

    if seg_meta.empty:
        return None

    # Embed keyframe
    img_emb = embed_images([kf_path])

    # Embed all transcript segments
    seg_texts = seg_meta["text"].astype(str).tolist()
    text_emb = siglip_embed_texts(seg_texts)

    # Compute similarities with temporal decay
    raw_sims = sigmoid_similarity(img_emb, text_emb)[0]
    t_mids = pd.to_numeric(seg_meta["t_mid"], errors="coerce").values

    weighted_sims = np.array([
        raw_sims[j] * temporal_gaussian_weight(t_mids[j], t_kf, cfg.TEMPORAL_SIGMA)
        for j in range(len(raw_sims))
    ])

    best_idx = int(np.argmax(weighted_sims))
    best_sim = float(raw_sims[best_idx])

    if best_sim < cfg.SIGLIP_MIN_SIM:
        return None

    t_narr = float(t_mids[best_idx])
    delta_t = t_narr - t_kf

    # Alpha
    alpha = 0.5 + 0.5 * (best_sim - cfg.SIGLIP_ALPHA_LOW) / (cfg.SIGLIP_ALPHA_HIGH - cfg.SIGLIP_ALPHA_LOW)
    alpha = max(0.5, min(1.0, alpha))

    return {
        "t_keyframe": t_kf,
        "concept_text": concept_text,
        "match_type": "siglip_vision",
        "match_track": "B",
        "t_narr": t_narr,
        "delta_t": delta_t,
        "n_word_matches": 0,
        "siglip_sim": best_sim,
        "semantic_sim": None,
        "alpha": alpha,
    }


def _track_c(scene, seg_meta, seg_embeddings, cfg):
    """Track C: Semantic cosine similarity."""
    from ..utils.embedding_utils import embed_texts, cosine_similarity_matrix

    t_kf = float(scene["t_keyframe"])
    concept_text = str(scene.get("concept_text", ""))

    if not concept_text.strip() or seg_meta.empty or len(seg_embeddings) == 0:
        return None

    # Embed concept text
    concept_emb = embed_texts([concept_text], cfg.EMBEDDING_MODEL_NAME)

    # Cosine similarity with temporal decay
    raw_sims = cosine_similarity_matrix(concept_emb, seg_embeddings)[0]
    t_mids = pd.to_numeric(seg_meta["t_mid"], errors="coerce").values

    from ..utils.siglip_utils import temporal_gaussian_weight
    weighted_sims = np.array([
        raw_sims[j] * temporal_gaussian_weight(t_mids[j], t_kf, cfg.TEMPORAL_SIGMA)
        for j in range(len(raw_sims))
    ])

    best_idx = int(np.argmax(weighted_sims))
    best_sim = float(raw_sims[best_idx])

    if best_sim < cfg.MIN_GLOBAL_SIM:
        return None

    t_narr = float(t_mids[best_idx])
    delta_t = t_narr - t_kf

    # Alpha
    alpha = (best_sim - cfg.ALPHA_SIM_LOW) / (cfg.ALPHA_SIM_HIGH - cfg.ALPHA_SIM_LOW)
    alpha = max(0.0, min(1.0, alpha))

    return {
        "t_keyframe": t_kf,
        "concept_text": concept_text,
        "match_type": "text_semantic",
        "match_track": "C",
        "t_narr": t_narr,
        "delta_t": delta_t,
        "n_word_matches": 0,
        "siglip_sim": None,
        "semantic_sim": best_sim,
        "alpha": alpha,
    }


def _no_match_result(scene_id, t_vis, concept_text, frame_type, frame_type_conf,
                     reason="no_match"):
    return {
        "scene_id": scene_id,
        "t_vis": t_vis,
        "t_keyframe": t_vis,
        "concept_text": concept_text,
        "match_type": reason,
        "match_track": "none",
        "t_narr": None,
        "delta_t": None,
        "n_word_matches": 0,
        "siglip_sim": None,
        "semantic_sim": None,
        "scene_type": frame_type,
        "scene_type_conf": frame_type_conf,
        "alpha": 0.0,
    }
