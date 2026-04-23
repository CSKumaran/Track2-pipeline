"""SigLIP embeddings, alignment, and scene classification."""

import logging
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_model = None
_preprocess = None
_tokenizer = None


def load_siglip(model_name: str = "ViT-B-16-SigLIP", pretrained: str = "webli"):
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return _model, _preprocess, _tokenizer
    import open_clip
    logger.info("Loading SigLIP: %s/%s", model_name, pretrained)
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    _tokenizer = open_clip.get_tokenizer(model_name)
    _model.eval()
    return _model, _preprocess, _tokenizer


def unload_siglip():
    global _model, _preprocess, _tokenizer
    if _model is not None:
        del _model, _preprocess, _tokenizer
        _model = None
        _preprocess = None
        _tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Unloaded SigLIP")


def embed_images(image_paths: list, batch_size: int = 4) -> np.ndarray:
    """Embed images with SigLIP vision encoder."""
    model, preprocess, _ = load_siglip()
    all_emb = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        images = []
        for p in batch:
            try:
                img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                images.append(img)
            except Exception as e:
                logger.warning("Failed to load %s: %s", p, e)
                images.append(torch.zeros(1, 3, 224, 224))
        images_tensor = torch.cat(images, dim=0)
        with torch.no_grad():
            emb = model.encode_image(images_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


def embed_texts(texts: list, batch_size: int = 4) -> np.ndarray:
    """Embed texts with SigLIP text encoder."""
    model, _, tokenizer = load_siglip()
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch)
        with torch.no_grad():
            emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


def sigmoid_similarity(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Compute SigLIP sigmoid similarity."""
    # SigLIP uses sigmoid instead of softmax
    logits = image_emb @ text_emb.T
    return 1.0 / (1.0 + np.exp(-logits))


def classify_frame_type(image_path: str, content_labels: list,
                        non_content_labels: list) -> tuple:
    """Zero-shot classify frame as content vs non-content.
    Returns (is_content, frame_type, confidence)."""
    all_labels = content_labels + non_content_labels
    img_emb = embed_images([image_path])
    text_emb = embed_texts(all_labels)
    sims = sigmoid_similarity(img_emb, text_emb)[0]
    best_idx = int(np.argmax(sims))
    best_label = all_labels[best_idx]
    is_content = best_idx < len(content_labels)
    return is_content, best_label, float(sims[best_idx])


def temporal_gaussian_weight(t_seg: float, t_ref: float, sigma: float = 15.0) -> float:
    """Gaussian temporal decay weight."""
    return float(np.exp(-0.5 * ((t_seg - t_ref) / sigma) ** 2))
