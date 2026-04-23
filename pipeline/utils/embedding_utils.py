"""Text embedding utilities — offline (sentence-transformers) or API stub."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pipeline.config import Config

logger = logging.getLogger(__name__)

# Module-level cache so the model is loaded only once per process.
_model_cache: dict[str, object] = {}


def _load_st_model(model_name: str):
    """Lazily load a sentence-transformers model (cached)."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model: %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _get_embeddings_api_stub(texts: list[str], config: "Config") -> np.ndarray:
    """Placeholder for API-based embeddings (e.g. OpenAI, Cohere, Voyage).

    To implement:
        1. Call your embedding API with *texts* in batches.
        2. Collect the resulting vectors.
        3. Return an (N, d) numpy array.
    """
    raise NotImplementedError(
        "API embeddings are not yet implemented. "
        "Set USE_API_EMBEDDINGS=False to use the local sentence-transformers model."
    )


def get_text_embeddings(texts: list[str], config: "Config") -> np.ndarray:
    """Return an (N, d) array of embeddings for the given *texts*.

    Uses the local sentence-transformers model by default.
    If ``config.USE_API_EMBEDDINGS`` is True, delegates to an API stub.
    """
    if config.USE_API_EMBEDDINGS:
        return _get_embeddings_api_stub(texts, config)

    model = _load_st_model(config.EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=False,
                              convert_to_numpy=True)
    return np.asarray(embeddings, dtype=np.float32)
