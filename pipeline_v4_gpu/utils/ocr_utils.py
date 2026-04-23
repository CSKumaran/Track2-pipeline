"""OCR utilities [v4.0] — Surya primary, EasyOCR fallback + fuzzy matching.

[v4.0] Added transcript-informed spellcheck and abbreviation expansion.
"""

import logging
import os
import re
from Levenshtein import ratio as levenshtein_ratio

logger = logging.getLogger(__name__)

_ocr_engine = None
_ocr_engine_type = None
_ocr_failed_permanently = False


# =====================================================================
# Engine Loading
# =====================================================================

def get_ocr_engine(engine_name: str = "surya"):
    """Lazy-load OCR engine. Returns (engine, engine_type).

    Priority: surya → easyocr
    PaddleOCR removed due to PIR/oneDNN incompatibility with PyTorch 2.6.
    """
    global _ocr_engine, _ocr_engine_type, _ocr_failed_permanently

    if _ocr_engine is not None and not _ocr_failed_permanently:
        return _ocr_engine, _ocr_engine_type

    if _ocr_failed_permanently:
        _ocr_engine = None
        _ocr_engine_type = None
        _ocr_failed_permanently = False

    # Map legacy paddleocr requests to surya
    if engine_name == "paddleocr":
        logger.info("PaddleOCR requested but deprecated — using Surya instead")
        engine_name = "surya"

    if engine_name == "surya":
        try:
            # Surya >=0.17 uses pipeline-based API
            try:
                from surya.pipeline import SuryaPipeline
                pipeline = SuryaPipeline()
                _ocr_engine = {"pipeline": pipeline, "api_version": "0.17+"}
                _ocr_engine_type = "surya"
                logger.info("Loaded Surya OCR v0.17+ pipeline (GPU-accelerated)")
                return _ocr_engine, _ocr_engine_type
            except (ImportError, Exception) as e_new:
                logger.debug("Surya v0.17+ pipeline not available: %s", e_new)

            # Surya <0.17 uses separate predictors
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            det_predictor = DetectionPredictor()
            rec_predictor = RecognitionPredictor()
            _ocr_engine = {
                "det_predictor": det_predictor,
                "rec_predictor": rec_predictor,
                "api_version": "0.16",
            }
            _ocr_engine_type = "surya"
            logger.info("Loaded Surya OCR v0.16 (GPU-accelerated)")
            return _ocr_engine, _ocr_engine_type
        except ImportError:
            logger.warning("Surya not installed, trying EasyOCR")
            engine_name = "easyocr"
        except Exception as e:
            logger.warning("Surya failed to load: %s, trying EasyOCR", e)
            engine_name = "easyocr"

    if engine_name == "easyocr":
        try:
            import easyocr
            import torch
            gpu = torch.cuda.is_available()
            _ocr_engine = easyocr.Reader(["en"], gpu=gpu)
            _ocr_engine_type = "easyocr"
            logger.info("Loaded EasyOCR (GPU=%s)", gpu)
            return _ocr_engine, _ocr_engine_type
        except Exception as e:
            logger.error("EasyOCR also failed: %s", e)
            return None, None

    return None, None


# =====================================================================
# Single Image OCR
# =====================================================================

def run_ocr(image_path: str, engine_name: str = "surya",
            min_confidence: float = 0.3) -> list:
    """Run OCR on a single image. Returns list of (text, confidence, bbox).

    For batch processing of multiple images, use run_ocr_batch() instead.
    """
    global _ocr_failed_permanently
    engine, actual_type = get_ocr_engine(engine_name)
    if engine is None:
        return []

    try:
        if actual_type == "surya":
            return _run_surya_single(engine, image_path, min_confidence)
        else:
            return _run_easyocr_single(engine, image_path, min_confidence)
    except Exception as e:
        if actual_type == "surya":
            logger.warning("Surya failed on %s: %s — switching to EasyOCR", image_path, e)
            _ocr_failed_permanently = True
            _fallback_to_easyocr()
            engine, actual_type = get_ocr_engine("easyocr")
            if engine is not None:
                return run_ocr(image_path, "easyocr", min_confidence)
        else:
            logger.warning("OCR failed on %s: %s", image_path, e)
        return []


def _run_surya_single(engine: dict, image_path: str,
                      min_confidence: float) -> list:
    """Run Surya OCR on a single image."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    if engine.get("api_version") == "0.17+":
        # Surya v0.17+ pipeline API
        pipeline = engine["pipeline"]
        result = pipeline.ocr([img], languages=["en"])
        return _extract_surya_results(result, 0, min_confidence)
    else:
        # Surya v0.16 separate predictors API
        from surya.recognition import run_recognition
        from surya.detection import run_detection

        det_predictor = engine["det_predictor"]
        rec_predictor = engine["rec_predictor"]

        det_results = run_detection([img], det_predictor)
        rec_results = run_recognition(
            [img], det_results, rec_predictor, ["en"]
        )
        return _extract_surya_results(rec_results, 0, min_confidence)


def _run_easyocr_single(engine, image_path: str,
                        min_confidence: float) -> list:
    """Run EasyOCR on a single image."""
    raw = engine.readtext(image_path)
    results = []
    for bbox, text, conf in raw:
        if conf >= min_confidence and len(text.strip()) > 0:
            results.append((text.strip(), float(conf), bbox))
    return results


# =====================================================================
# Batch OCR (Surya-optimized)
# =====================================================================

def run_ocr_batch(image_paths: list, engine_name: str = "surya",
                  min_confidence: float = 0.3) -> list:
    """Run OCR on a batch of images. Returns list of list of (text, confidence, bbox).

    Surya processes all images in one GPU call for maximum efficiency.
    EasyOCR falls back to serial processing.
    """
    engine, actual_type = get_ocr_engine(engine_name)
    if engine is None:
        return [[] for _ in image_paths]

    if not image_paths:
        return []

    try:
        if actual_type == "surya":
            return _run_surya_batch(engine, image_paths, min_confidence)
        else:
            # EasyOCR: serial fallback
            return [_run_easyocr_single(engine, p, min_confidence)
                    for p in image_paths]
    except Exception as e:
        logger.warning("Batch OCR failed: %s — falling back to serial", e)
        results = []
        for p in image_paths:
            try:
                results.append(run_ocr(p, engine_name, min_confidence))
            except Exception:
                results.append([])
        return results


def _extract_surya_results(rec_results, page_idx: int,
                           min_confidence: float) -> list:
    """Extract text results from Surya output (works for both API versions)."""
    results = []
    if rec_results and page_idx < len(rec_results):
        page = rec_results[page_idx]
        for line in page.text_lines:
            text = line.text.strip()
            conf = line.confidence
            bbox = line.bbox if hasattr(line, 'bbox') else None
            if conf >= min_confidence and len(text) > 0:
                results.append((text, float(conf), bbox))
    return results


def _run_surya_batch(engine: dict, image_paths: list,
                     min_confidence: float) -> list:
    """Run Surya OCR on a batch of images in one GPU call."""
    from PIL import Image

    # Load all images
    images = []
    valid_indices = []
    for i, p in enumerate(image_paths):
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            valid_indices.append(i)
        except Exception as e:
            logger.warning("Could not load image %s: %s", p, e)

    if not images:
        return [[] for _ in image_paths]

    logger.info("Surya batch OCR: %d images", len(images))

    if engine.get("api_version") == "0.17+":
        # Surya v0.17+ pipeline API
        pipeline = engine["pipeline"]
        rec_results = pipeline.ocr(images, languages=["en"])
    else:
        # Surya v0.16 separate predictors API
        from surya.recognition import run_recognition
        from surya.detection import run_detection

        det_predictor = engine["det_predictor"]
        rec_predictor = engine["rec_predictor"]

        det_results = run_detection(images, det_predictor)
        languages = [["en"]] * len(images)
        rec_results = run_recognition(images, det_results, rec_predictor, languages)

    # Build results indexed by original position
    all_results = [[] for _ in image_paths]
    for batch_idx, orig_idx in enumerate(valid_indices):
        if batch_idx < len(rec_results):
            page = rec_results[batch_idx]
            for line in page.text_lines:
                text = line.text.strip()
                conf = line.confidence
                bbox = line.bbox if hasattr(line, 'bbox') else None
                if conf >= min_confidence and len(text) > 0:
                    all_results[orig_idx].append((text, float(conf), bbox))

    return all_results


# =====================================================================
# Fallback Helper
# =====================================================================

def _fallback_to_easyocr():
    """Reset engine state so next get_ocr_engine call loads EasyOCR."""
    global _ocr_engine, _ocr_engine_type, _ocr_failed_permanently
    logger.warning("Switching to EasyOCR permanently for this session")
    _ocr_engine = None
    _ocr_engine_type = None
    _ocr_failed_permanently = False


def unload_ocr():
    """Free OCR engine resources."""
    global _ocr_engine, _ocr_engine_type
    if _ocr_engine is not None:
        logger.info("Unloaded OCR engine: %s", _ocr_engine_type)
        _ocr_engine = None
        _ocr_engine_type = None


# =====================================================================
# Word Extraction & Matching
# =====================================================================

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


# =====================================================================
# [v4.0] Transcript-Informed OCR Post-Processing
# =====================================================================

def build_transcript_vocabulary(transcript_words_path: str) -> set:
    """Build a vocabulary set from transcript words for OCR spellcheck.

    Returns a set of normalized words from the transcript, which can be
    used to correct garbled OCR output (e.g. "Iocal" → "local").
    """
    import pandas as pd

    vocab = set()
    try:
        words_df = pd.read_csv(transcript_words_path)
        if "word" in words_df.columns:
            for w in words_df["word"].dropna():
                w_norm = normalize_word(str(w))
                if len(w_norm) >= 3:
                    vocab.add(w_norm)
    except Exception as e:
        logger.warning("Could not build transcript vocabulary: %s", e)

    logger.info("[v4.0] Transcript vocabulary: %d unique words", len(vocab))
    return vocab


def correct_ocr_with_transcript(ocr_words: set, transcript_vocab: set,
                                threshold: float = 0.75) -> set:
    """Correct OCR words using transcript vocabulary as reference.

    For each OCR word that doesn't match any transcript word exactly,
    find the closest transcript word by Levenshtein ratio. If above threshold,
    replace the OCR word with the transcript word.

    Returns: corrected word set (union of originals and corrections).
    """
    corrected = set(ocr_words)

    for ocr_w in list(ocr_words):
        if ocr_w in transcript_vocab:
            continue  # already matches

        best_match = None
        best_ratio = 0

        for tv_w in transcript_vocab:
            # Quick length filter: skip if lengths differ by more than 30%
            if abs(len(ocr_w) - len(tv_w)) > max(len(ocr_w), len(tv_w)) * 0.3:
                continue
            r = levenshtein_ratio(ocr_w, tv_w)
            if r > best_ratio:
                best_ratio = r
                best_match = tv_w

        if best_match and best_ratio >= threshold:
            corrected.add(best_match)

    return corrected


def build_abbreviation_map(transcript_words_path: str) -> dict:
    """Build abbreviation → expansion map from transcript.

    Detects common abbreviation patterns:
    - All-caps words (KNN, SVM, CNN) mapped to their lowercase form
    - Words with internal caps (BackProp) mapped to separated form

    Returns dict: normalized_abbrev → set of possible expansions.
    """
    import pandas as pd

    abbrev_map = {}
    try:
        words_df = pd.read_csv(transcript_words_path)
        if "word" not in words_df.columns:
            return abbrev_map

        all_words = [str(w).strip() for w in words_df["word"].dropna()]

        for w in all_words:
            if len(w) < 2:
                continue

            w_norm = normalize_word(w)

            # All-caps abbreviation (KNN, SVM)
            if w.isupper() and len(w) >= 2:
                # Map "knn" → {"knn"} plus hyphenated variant "k-nn"
                abbrev_map.setdefault(w_norm, set()).add(w_norm)
                # Common variants: k-nn, k.n.n.
                no_sep = w_norm
                with_hyphens = "-".join(w_norm)
                abbrev_map.setdefault(no_sep, set()).add(with_hyphens)

    except Exception as e:
        logger.warning("Could not build abbreviation map: %s", e)

    return abbrev_map


def expand_abbreviations_in_ocr(ocr_words: set, abbrev_map: dict) -> set:
    """Expand abbreviations found in OCR using the abbreviation map.

    If OCR contains "k-nn" or "knn" and transcript has "KNN", add all variants.
    """
    expanded = set(ocr_words)

    for ocr_w in list(ocr_words):
        # Check if OCR word matches any abbreviation variant
        ocr_stripped = normalize_word(ocr_w)
        for abbrev, variants in abbrev_map.items():
            if ocr_stripped == abbrev or ocr_stripped in variants:
                expanded.update(variants)
                expanded.add(abbrev)

    return expanded
