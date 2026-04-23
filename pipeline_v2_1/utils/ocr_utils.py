"""PaddleOCR wrapper + fuzzy matching utilities."""

import logging
import re
from Levenshtein import ratio as levenshtein_ratio

logger = logging.getLogger(__name__)

_ocr_engine = None
_ocr_engine_type = None  # Track which engine was actually loaded


def get_ocr_engine(engine_name: str = "easyocr"):
    """Lazy-load OCR engine. Returns (engine, engine_type)."""
    global _ocr_engine, _ocr_engine_type
    if _ocr_engine is not None and _ocr_engine_type == engine_name:
        return _ocr_engine, _ocr_engine_type
    if _ocr_engine is not None and _ocr_engine_type != engine_name:
        _ocr_engine = None
        _ocr_engine_type = None

    if engine_name == "paddleocr":
        try:
            from paddleocr import PaddleOCR
            _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")
            _ocr_engine_type = "paddleocr"
            logger.info("Loaded PaddleOCR")
            return _ocr_engine, _ocr_engine_type
        except Exception as e:
            logger.warning("PaddleOCR failed: %s, trying easyocr", e)
            engine_name = "easyocr"

    if engine_name == "easyocr":
        try:
            import easyocr
            import torch
            _ocr_engine = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
            _ocr_engine_type = "easyocr"
            logger.info("Loaded EasyOCR")
            return _ocr_engine, _ocr_engine_type
        except Exception as e:
            logger.error("EasyOCR also failed: %s", e)
            return None, None

    return None, None


def run_ocr(image_path: str, engine_name: str = "paddleocr",
            min_confidence: float = 0.3) -> list:
    """Run OCR on an image. Returns list of (text, confidence, bbox)."""
    engine, actual_type = get_ocr_engine(engine_name)
    if engine is None:
        return []

    results = []
    try:
        if actual_type == "paddleocr":
            # PaddleOCR v3.x uses .predict(), older uses .ocr()
            if hasattr(engine, 'predict'):
                raw = engine.predict(image_path)
                # New API returns dict with 'rec_texts', 'rec_scores', 'dt_polys'
                if raw and isinstance(raw, dict):
                    texts = raw.get('rec_texts', [])
                    scores = raw.get('rec_scores', [])
                    polys = raw.get('dt_polys', [None] * len(texts))
                    for t, s, p in zip(texts, scores, polys):
                        if s >= min_confidence and len(t.strip()) > 0:
                            results.append((t.strip(), float(s), p))
                elif raw and isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, dict):
                            texts = item.get('rec_texts', [])
                            scores = item.get('rec_scores', [])
                            polys = item.get('dt_polys', [None] * len(texts))
                            for t, s, p in zip(texts, scores, polys):
                                if s >= min_confidence and len(t.strip()) > 0:
                                    results.append((t.strip(), float(s), p))
            else:
                raw = engine.ocr(image_path, cls=True)
                if raw and raw[0]:
                    for line in raw[0]:
                        bbox, (text, conf) = line[0], line[1]
                        if conf >= min_confidence and len(text.strip()) > 0:
                            results.append((text.strip(), float(conf), bbox))
        else:
            # EasyOCR
            raw = engine.readtext(image_path)
            for bbox, text, conf in raw:
                if conf >= min_confidence and len(text.strip()) > 0:
                    results.append((text.strip(), float(conf), bbox))
    except Exception as e:
        logger.warning("OCR failed on %s: %s", image_path, e)

    return results


def extract_words_from_ocr(ocr_results: list) -> set:
    """Extract normalized word set from OCR results."""
    words = set()
    for text, _, _ in ocr_results:
        for w in text.split():
            w_norm = normalize_word(w)
            if len(w_norm) >= 2:
                words.add(w_norm)
    return words


def normalize_word(word: str) -> str:
    """Lowercase, strip non-alphanumeric."""
    return re.sub(r'[^a-z0-9]', '', word.lower())


def jaccard_distance(set_a: set, set_b: set) -> float:
    """Compute Jaccard distance between two word sets."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return 1.0 - len(intersection) / len(union)


def fuzzy_match_word(keyword: str, ocr_words: set, threshold: float = 0.8) -> bool:
    """Check if keyword fuzzy-matches any OCR word."""
    kw_norm = normalize_word(keyword)
    if not kw_norm:
        return False
    for w in ocr_words:
        if levenshtein_ratio(kw_norm, w) > threshold:
            return True
    return False


def fuzzy_match_multiword(keyword: str, ocr_words: set, threshold: float = 0.8) -> bool:
    """Check if all words in a multi-word keyword appear in OCR words."""
    parts = keyword.lower().split()
    if len(parts) <= 1:
        return fuzzy_match_word(keyword, ocr_words, threshold)
    # All component words must appear
    for part in parts:
        part_norm = normalize_word(part)
        if len(part_norm) < 2:
            continue
        found = False
        for w in ocr_words:
            if levenshtein_ratio(part_norm, w) > threshold:
                found = True
                break
        if not found:
            return False
    return True
