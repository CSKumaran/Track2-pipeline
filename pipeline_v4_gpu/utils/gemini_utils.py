"""Gemini API utilities with local file caching and rate-limit retry.

[v4.0] Added query_gemini_vision() for image+text prompts (frame understanding).
[v4.0] Added query_gemini_multimodal() for multi-image+text prompts (importance with visual context).
"""

import logging
import os
import json
import hashlib
import time
import base64

from ..config import Config

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 15  # seconds between retries on 429


def _cache_key(prompt: str, model: str, image_paths: list = None) -> str:
    """Generate cache key from prompt + model + optional image paths."""
    h = hashlib.md5()
    h.update(prompt.encode())
    h.update(model.encode())
    if image_paths:
        for p in sorted(image_paths):
            h.update(p.encode())
            # Include file mtime so cache invalidates if image changes
            try:
                h.update(str(os.path.getmtime(p)).encode())
            except OSError:
                pass
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


def _load_image_part(image_path: str):
    """Load an image file and return a Gemini Part object."""
    from google.genai import types

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".webp": "image/webp", ".gif": "image/gif"}
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        data = f.read()

    return types.Part.from_bytes(data=data, mime_type=mime_type)


def query_gemini_vision(prompt: str, image_path: str, cfg: Config,
                        cache_dir: str = None) -> str:
    """Query Gemini with a single image + text prompt. Returns response text.

    [v4.0] Used for frame understanding — extracting concepts from keyframes.
    """
    if not cfg.GEMINI_API_KEY:
        logger.warning("No Gemini API key configured")
        return ""

    if not os.path.exists(image_path):
        logger.warning("Image not found: %s", image_path)
        return ""

    _cdir = cache_dir or cfg.GEMINI_FRAME_CACHE_DIR
    cache_k = _cache_key(prompt, cfg.GEMINI_MODEL, [image_path])
    cached = _get_cache(_cdir, cache_k)
    if cached is not None:
        logger.debug("Gemini vision cache hit: %s", os.path.basename(image_path))
        return cached.get("text", "")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)
    image_part = _load_image_part(image_path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=cfg.GEMINI_MODEL,
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=cfg.GEMINI_TEMPERATURE,
                ),
            )
            text = response.text
            _set_cache(_cdir, cache_k, {"text": text})
            return text
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES:
                logger.warning("Rate limited (attempt %d/%d), waiting %ds...",
                               attempt, MAX_RETRIES, RETRY_DELAY * attempt)
                time.sleep(RETRY_DELAY * attempt)
            else:
                logger.warning("Gemini vision query failed: %s", e)
                return ""
    return ""


def query_gemini_multimodal(prompt: str, image_paths: list, cfg: Config,
                            cache_dir: str = None) -> str:
    """Query Gemini with multiple images + text prompt. Returns response text.

    [v4.0] Used for importance rating with visual context — sends all
    overlapping scene keyframes for a transcript segment.
    """
    if not cfg.GEMINI_API_KEY:
        logger.warning("No Gemini API key configured")
        return ""

    # Filter to existing images
    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        logger.debug("No valid images for multimodal query, falling back to text-only")
        return query_gemini_text(prompt, cfg)

    _cdir = cache_dir or cfg.GEMINI_CACHE_DIR
    cache_k = _cache_key(prompt, cfg.GEMINI_MODEL, valid_paths)
    cached = _get_cache(_cdir, cache_k)
    if cached is not None:
        logger.debug("Gemini multimodal cache hit (%d images)", len(valid_paths))
        return cached.get("text", "")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=cfg.GEMINI_API_KEY)

    # Build contents: images first, then text prompt
    contents = []
    for p in valid_paths:
        try:
            contents.append(_load_image_part(p))
        except Exception as e:
            logger.warning("Failed to load image %s: %s", p, e)
    contents.append(prompt)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=cfg.GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=cfg.GEMINI_TEMPERATURE,
                ),
            )
            text = response.text
            _set_cache(_cdir, cache_k, {"text": text})
            return text
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES:
                logger.warning("Rate limited (attempt %d/%d), waiting %ds...",
                               attempt, MAX_RETRIES, RETRY_DELAY * attempt)
                time.sleep(RETRY_DELAY * attempt)
            else:
                logger.warning("Gemini multimodal query failed: %s", e)
                return ""
    return ""
