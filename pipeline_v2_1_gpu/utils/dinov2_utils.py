"""DINOv2 frame embeddings and distances."""

import logging
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_model = None
_processor = None


def load_dinov2(model_name: str = "facebook/dinov2-base"):
    global _model, _processor
    if _model is not None:
        return _model, _processor
    from transformers import AutoImageProcessor, AutoModel
    logger.info("Loading DINOv2: %s", model_name)
    _processor = AutoImageProcessor.from_pretrained(model_name)
    _model = AutoModel.from_pretrained(model_name)
    _model.eval()
    return _model, _processor


def unload_dinov2():
    global _model, _processor
    if _model is not None:
        del _model, _processor
        _model = None
        _processor = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Unloaded DINOv2")


def embed_frames(frame_paths: list, model_name: str = "facebook/dinov2-base",
                 batch_size: int = 4) -> np.ndarray:
    """Embed frames with DINOv2. Returns (N, 768) array."""
    model, processor = load_dinov2(model_name)
    all_embeddings = []

    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning("Failed to load %s: %s", p, e)
                images.append(Image.new("RGB", (224, 224)))

        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


def compute_consecutive_distances(embeddings: np.ndarray) -> np.ndarray:
    """Cosine distance between consecutive frame embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity
    if len(embeddings) < 2:
        return np.array([])
    distances = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i:i+1], embeddings[i+1:i+2])[0, 0]
        distances.append(1.0 - sim)
    return np.array(distances)


def find_centroid_frame(embeddings: np.ndarray) -> int:
    """Find frame index closest to centroid embedding."""
    if len(embeddings) == 0:
        return 0
    centroid = embeddings.mean(axis=0, keepdims=True)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(centroid, embeddings)[0]
    return int(np.argmax(sims))
