"""Gemini API utilities with local file caching and rate-limit retry."""

import logging
import os
import json
import hashlib
import time

from ..config import Config

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds between retries on 429


def _cache_key(prompt: str, model: str) -> str:
    """Generate cache key from prompt + model."""
    h = hashlib.md5()
    h.update(prompt.encode())
    h.update(model.encode())
    return h.hexdigest()


def _get_cache(cache_dir: str, key: str):
    """Load cached response if exists."""
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _set_cache(cache_dir: str, key: str, data):
    """Save response to cache."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f)


def query_gemini_text(prompt: str, cfg: Config) -> str:
    """Query Gemini with text-only prompt. Returns response text."""
    if not cfg.GEMINI_API_KEY:
        logger.warning("No Gemini API key configured")
        return ""

    cache_key = _cache_key(prompt, cfg.GEMINI_MODEL)
    cached = _get_cache(cfg.GEMINI_CACHE_DIR, cache_key)
    if cached is not None:
        logger.debug("Gemini cache hit")
        return cached.get("text", "")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=cfg.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=cfg.GEMINI_TEMPERATURE,
                ),
            )
            text = response.text
            _set_cache(cfg.GEMINI_CACHE_DIR, cache_key, {"text": text})
            return text
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES:
                logger.warning("Rate limited (attempt %d/%d), waiting %ds...",
                               attempt, MAX_RETRIES, RETRY_DELAY * attempt)
                time.sleep(RETRY_DELAY * attempt)
            else:
                logger.warning("Gemini text query failed: %s", e)
                return ""
    return ""
