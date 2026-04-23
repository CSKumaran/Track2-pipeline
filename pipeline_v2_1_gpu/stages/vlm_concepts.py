"""Stage 3: Visual Concept Extraction (VLM + OCR)."""

import logging
import os
import json
import pandas as pd

from ..config import Config
from ..utils.ocr_utils import run_ocr
from ..utils.io_utils import save_csv, cache_exists

logger = logging.getLogger(__name__)

VLM_PROMPT = (
    "Describe the educational content visible in this frame. "
    "List any text, formulas, diagrams, charts, code, or visual concepts shown."
)


def run_stage3(output_dir: str, cfg: Config) -> dict:
    """Run visual concept extraction on scene keyframes."""
    cache_file = "scene_concepts.csv"
    if cache_exists(output_dir, [cache_file]):
        logger.info("Stage 3: cache hit, skipping VLM concepts")
        return {cache_file: os.path.join(output_dir, cache_file)}

    scenes_df = pd.read_csv(os.path.join(output_dir, "scenes.csv"))
    results = []

    for _, scene in scenes_df.iterrows():
        kf_path = scene["keyframe_path"]
        scene_id = scene["scene_id"]

        # OCR on keyframe
        if cfg.OCR_ENABLED:
            ocr_results = run_ocr(kf_path, cfg.OCR_ENGINE, cfg.OCR_MIN_CONFIDENCE)
            ocr_text = " ".join([t for t, _, _ in ocr_results])
        else:
            ocr_text = ""

        # VLM concept extraction
        vlm_text = ""
        vlm_backend = "none"

        if cfg.VLM_MODE == "ollama":
            vlm_text = _run_ollama_vlm(kf_path, cfg)
            vlm_backend = f"ollama/{cfg.OLLAMA_MODEL}"
        elif cfg.VLM_MODE == "gemini":
            vlm_text = _run_gemini_vlm(kf_path, cfg)
            vlm_backend = f"gemini/{cfg.GEMINI_MODEL}"

        concept_text = f"{ocr_text} {vlm_text}".strip()

        # Frame type classification via SigLIP
        is_content, frame_type, frame_type_conf = True, "content slide", 0.5
        if cfg.SIGLIP_ENABLED:
            try:
                from ..utils.siglip_utils import classify_frame_type
                is_content, frame_type, frame_type_conf = classify_frame_type(
                    kf_path, cfg.CONTENT_LABELS, cfg.NON_CONTENT_LABELS
                )
                # Low-confidence classification (< 0.55) is unreliable —
                # default to content to avoid dropping valid scenes
                if frame_type_conf < 0.55:
                    logger.info("Scene %d: low SigLIP confidence %.3f for '%s', defaulting to content",
                                scene_id, frame_type_conf, frame_type)
                    is_content = True
                    frame_type = f"{frame_type} (low_conf)"
            except Exception as e:
                logger.warning("SigLIP classification failed: %s", e)

        # If OCR found text, it's almost certainly a content frame
        if ocr_text.strip():
            is_content = True

        results.append({
            "scene_id": scene_id,
            "ocr_text": ocr_text,
            "vlm_text": vlm_text,
            "concept_text": concept_text,
            "is_content": is_content,
            "frame_type": frame_type,
            "frame_type_confidence": frame_type_conf,
            "vlm_backend": vlm_backend,
        })

    df = pd.DataFrame(results)
    path = os.path.join(output_dir, cache_file)
    save_csv(df, path)
    logger.info("Stage 3: %d scene concepts extracted", len(df))
    return {cache_file: path}


def _run_ollama_vlm(image_path: str, cfg: Config) -> str:
    """Run local VLM via Ollama API."""
    import requests
    import base64

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        resp = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": cfg.OLLAMA_MODEL,
                "prompt": VLM_PROMPT,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        else:
            logger.warning("Ollama returned %d: %s", resp.status_code, resp.text[:200])
            return ""
    except Exception as e:
        logger.warning("Ollama VLM failed: %s", e)
        return ""


def _run_gemini_vlm(image_path: str, cfg: Config) -> str:
    """Run Gemini API for concept extraction."""
    if not cfg.GEMINI_API_KEY:
        logger.warning("No Gemini API key, skipping Gemini VLM")
        return ""

    try:
        from ..utils.gemini_utils import query_gemini_with_image
        return query_gemini_with_image(image_path, VLM_PROMPT, cfg)
    except Exception as e:
        logger.warning("Gemini VLM failed: %s", e)
        return ""
