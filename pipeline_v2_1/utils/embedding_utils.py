"""bge-large-en-v1.5 embedding wrapper."""

import logging
import numpy as np

logger = logging.getLogger(__name__)

_model = None


def load_embedding_model(model_name: str = "BAAI/bge-large-en-v1.5"):
    global _model
    if _model is not None:
        return _model
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", model_name)
    _model = SentenceTransformer(model_name)
    return _model


def unload_embedding_model():
    global _model
    if _model is not None:
        del _model
        _model = None
        logger.info("Unloaded embedding model")


def embed_texts(texts: list, model_name: str = "BAAI/bge-large-en-v1.5",
                prefix: str = "Represent this sentence: ",
                batch_size: int = 4) -> np.ndarray:
    """Embed texts with bge-large. Returns (N, 1024) array."""
    model = load_embedding_model(model_name)
    # bge instruction prefix
    prefixed = [prefix + t for t in texts]
    embeddings = model.encode(prefixed, batch_size=batch_size,
                              show_progress_bar=False, normalize_embeddings=True)
    return np.array(embeddings)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(a, b)
