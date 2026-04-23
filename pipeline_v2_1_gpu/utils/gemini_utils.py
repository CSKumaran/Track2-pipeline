"""Gemini API + structured output + caching."""

import logging
import os
import json
import hashlib
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)


def _cache_key(prompt: str, image_path: str, model: str) -> str:
    """Generate cache key from prompt + image + model."""
    h = hashlib.md5()
    h.update(prompt.encode())
    h.update(model.encode())
    if image_path and os.path.exists(image_path):
        h.update(image_path.encode())
        h.update(str(os.path.getmtime(image_path)).encode())
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
    """Query Gemini with text-only prompt."""
    if not cfg.GEMINI_API_KEY:
        logger.warning("No Gemini API key configured")
        return ""

    cache_key = _cache_key(prompt, "", cfg.GEMINI_MODEL)
    cached = _get_cache(cfg.GEMINI_CACHE_DIR, cache_key)
    if cached is not None:
        return cached.get("text", "")

    try:
        import google.generativeai as genai
        genai.configure(api_key=cfg.GEMINI_API_KEY)

        model = genai.GenerativeModel(cfg.GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": cfg.GEMINI_TEMPERATURE},
        )
        text = response.text
        _set_cache(cfg.GEMINI_CACHE_DIR, cache_key, {"text": text})
        return text
    except Exception as e:
        logger.warning("Gemini text query failed: %s", e)
        return ""


def query_gemini_with_image(image_path: str, prompt: str, cfg: Config) -> str:
    """Query Gemini with image + text prompt."""
    if not cfg.GEMINI_API_KEY:
        logger.warning("No Gemini API key configured")
        return ""

    cache_key = _cache_key(prompt, image_path, cfg.GEMINI_MODEL)
    cached = _get_cache(cfg.GEMINI_CACHE_DIR, cache_key)
    if cached is not None:
        return cached.get("text", "")

    try:
        import google.generativeai as genai
        from PIL import Image
        genai.configure(api_key=cfg.GEMINI_API_KEY)

        model = genai.GenerativeModel(cfg.GEMINI_MODEL)
        img = Image.open(image_path)
        response = model.generate_content(
            [prompt, img],
            generation_config={"temperature": cfg.GEMINI_TEMPERATURE},
        )
        text = response.text
        _set_cache(cfg.GEMINI_CACHE_DIR, cache_key, {"text": text})
        return text
    except Exception as e:
        logger.warning("Gemini image query failed: %s", e)
        return ""


def query_gemini_structured(prompt: str, schema: dict, cfg: Config) -> dict:
    """Query Gemini with structured output mode."""
    if not cfg.GEMINI_API_KEY:
        logger.warning("No Gemini API key configured")
        return {}

    cache_key = _cache_key(prompt + json.dumps(schema), "", cfg.GEMINI_MODEL)
    cached = _get_cache(cfg.GEMINI_CACHE_DIR, cache_key)
    if cached is not None:
        return cached

    try:
        import google.generativeai as genai
        genai.configure(api_key=cfg.GEMINI_API_KEY)

        model = genai.GenerativeModel(cfg.GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": cfg.GEMINI_TEMPERATURE,
                "response_mime_type": "application/json",
                "response_schema": schema,
            },
        )
        data = json.loads(response.text)
        _set_cache(cfg.GEMINI_CACHE_DIR, cache_key, data)
        return data
    except Exception as e:
        logger.warning("Gemini structured query failed: %s", e)
        return {}
