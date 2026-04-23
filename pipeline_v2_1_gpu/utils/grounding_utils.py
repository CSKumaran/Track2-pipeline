"""GroundingDINO / Florence-2 wrapper (optional, disabled by default on CPU)."""

import logging

logger = logging.getLogger(__name__)

_grounding_model = None


def is_grounding_available(model_name: str = "grounding_dino") -> bool:
    """Check if grounding model is available."""
    if model_name == "grounding_dino":
        try:
            import groundingdino
            return True
        except ImportError:
            return False
    elif model_name == "florence2":
        try:
            from transformers import AutoModelForCausalLM
            return True
        except ImportError:
            return False
    return False


def detect_objects(image_path: str, text_query: str,
                   box_threshold: float = 0.25,
                   text_threshold: float = 0.25,
                   model_name: str = "grounding_dino") -> list:
    """
    Detect objects matching text query in image.
    Returns list of (bbox, confidence, label).
    bbox format: [x1, y1, x2, y2] normalized.
    """
    if not is_grounding_available(model_name):
        logger.warning("Grounding model %s not available", model_name)
        return []

    try:
        if model_name == "grounding_dino":
            return _detect_grounding_dino(image_path, text_query, box_threshold, text_threshold)
        elif model_name == "florence2":
            return _detect_florence2(image_path, text_query, box_threshold)
    except Exception as e:
        logger.warning("Object detection failed: %s", e)
        return []

    return []


def _detect_grounding_dino(image_path, text_query, box_threshold, text_threshold):
    """GroundingDINO detection."""
    # This requires groundingdino package which may not be installed
    logger.info("GroundingDINO detection: '%s'", text_query)
    try:
        from groundingdino.util.inference import load_model, load_image, predict
        import groundingdino.config
        import os

        # Load model (cached)
        global _grounding_model
        if _grounding_model is None:
            config_path = os.path.join(os.path.dirname(groundingdino.config.__file__),
                                       "GroundingDINO_SwinT_OGC.py")
            _grounding_model = load_model(config_path, "groundingdino_swint_ogc.pth")

        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=_grounding_model,
            image=image,
            caption=text_query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        results = []
        for bbox, conf, label in zip(boxes.tolist(), logits.tolist(), phrases):
            results.append((bbox, float(conf), label))
        return results
    except Exception as e:
        logger.warning("GroundingDINO failed: %s", e)
        return []


def _detect_florence2(image_path, text_query, box_threshold):
    """Florence-2 detection (via transformers)."""
    logger.info("Florence-2 detection: '%s'", text_query)
    # Florence-2 is heavy; stub for now
    return []
