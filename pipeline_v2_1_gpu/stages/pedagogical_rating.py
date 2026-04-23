"""Stage 6: Pedagogical Importance Rating (3-tier backend)."""

import logging
import os
import json
import numpy as np
import pandas as pd

from ..config import Config
from ..utils.io_utils import save_csv, cache_exists

logger = logging.getLogger(__name__)

IMPORTANCE_PROMPT_TEMPLATE = """You are an expert instructional designer. For each transcript segment, rate its pedagogical importance on a scale of 1-5:

1 = Low: intro/outro, greetings, transitions, filler
2 = Below Average: recap of known material, tangential examples
3 = Average: supporting explanation, context building
4 = Above Average: key concept introduction, important examples
5 = Critical: core concept derivation, formula explanation, step-by-step procedure, definition of fundamental terms

Consider: conceptual density, visual complexity, sequential dependency, assessment relevance.

Segments:
{segments}

Return a JSON array:
[{{"segment_id": 1, "importance": 3, "reason": "brief reason"}}, ...]"""


def run_stage6(output_dir: str, cfg: Config) -> dict:
    """Run pedagogical importance rating."""
    cache_file = "pedagogical_importance.csv"
    if cache_exists(output_dir, [cache_file]):
        logger.info("Stage 6: cache hit, skipping importance rating")
        return {cache_file: os.path.join(output_dir, cache_file)}

    segments_df = pd.read_csv(os.path.join(output_dir, "transcript_segments_improved.csv"))

    if segments_df.empty:
        logger.warning("No segments for importance rating")
        save_csv(pd.DataFrame(), os.path.join(output_dir, cache_file))
        return {cache_file: os.path.join(output_dir, cache_file)}

    # Build prompt
    seg_lines = []
    for _, seg in segments_df.iterrows():
        seg_lines.append(
            f'[segment_id: {seg["segment_id"]}] '
            f'({seg["start_time"]:.1f}s - {seg["end_time"]:.1f}s) '
            f'"{seg["text"]}"'
        )
    segments_text = "\n".join(seg_lines)
    prompt = IMPORTANCE_PROMPT_TEMPLATE.format(segments=segments_text)

    # Try tiers in order
    backend = cfg.IMPORTANCE_BACKEND
    ratings = None

    if backend in ("auto", "gemini"):
        ratings = _tier1_gemini(prompt, segments_df, cfg)
        if ratings is not None:
            logger.info("Importance backend: gemini")

    if ratings is None and backend in ("auto", "local_llm"):
        ratings = _tier2_local_llm(prompt, segments_df, cfg)
        if ratings is not None:
            logger.info("Importance backend: local_llm")

    if ratings is None:
        ratings = _tier3_heuristic(segments_df, output_dir, cfg)
        logger.info("Importance backend: heuristic")

    df = pd.DataFrame(ratings)
    save_csv(df, os.path.join(output_dir, cache_file))
    logger.info("Stage 6: %d segments rated", len(df))
    return {cache_file: os.path.join(output_dir, cache_file)}


def _tier1_gemini(prompt: str, segments_df: pd.DataFrame, cfg: Config) -> list:
    """Tier 1: Gemini API."""
    if not cfg.GEMINI_API_KEY:
        return None

    try:
        from ..utils.gemini_utils import query_gemini_text

        # Run 1
        resp1 = query_gemini_text(prompt, cfg)
        ratings1 = _parse_importance_json(resp1, segments_df)

        if cfg.IMPORTANCE_DOUBLE_RUN:
            # Run 2
            resp2 = query_gemini_text(prompt + "\n(Second run for reliability)", cfg)
            ratings2 = _parse_importance_json(resp2, segments_df)

            # Merge: average when disagree
            result = []
            for r1 in ratings1:
                r2_match = next((r for r in ratings2 if r["segment_id"] == r1["segment_id"]), None)
                r2_imp = r2_match["importance"] if r2_match else r1["importance"]
                is_reliable = abs(r1["importance"] - r2_imp) <= cfg.IMPORTANCE_DISAGREEMENT_THRESHOLD
                result.append({
                    "segment_id": r1["segment_id"],
                    "importance": round((r1["importance"] + r2_imp) / 2),
                    "reason": r1.get("reason", ""),
                    "backend": "gemini",
                    "is_reliable": is_reliable,
                    "run1_rating": r1["importance"],
                    "run2_rating": r2_imp,
                })
            return result

        return [{
            "segment_id": r["segment_id"],
            "importance": r["importance"],
            "reason": r.get("reason", ""),
            "backend": "gemini",
            "is_reliable": True,
            "run1_rating": r["importance"],
            "run2_rating": None,
        } for r in ratings1]

    except Exception as e:
        logger.warning("Gemini importance rating failed: %s", e)
        return None


def _tier2_local_llm(prompt: str, segments_df: pd.DataFrame, cfg: Config) -> list:
    """Tier 2: Local LLM via Ollama."""
    import requests

    try:
        resp = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": cfg.LOCAL_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=300,
        )
        if resp.status_code == 200:
            text = resp.json().get("response", "")
            ratings = _parse_importance_json(text, segments_df)
            return [{
                "segment_id": r["segment_id"],
                "importance": r["importance"],
                "reason": r.get("reason", ""),
                "backend": "local_llm",
                "is_reliable": True,
                "run1_rating": r["importance"],
                "run2_rating": None,
            } for r in ratings]
    except Exception as e:
        logger.warning("Local LLM importance failed: %s", e)

    return None


def _tier3_heuristic(segments_df: pd.DataFrame, output_dir: str, cfg: Config) -> list:
    """Tier 3: Heuristic fallback."""
    # Load keyword data if available
    kw_path = os.path.join(output_dir, "segment_keyword_scores.csv")
    kw_data = {}
    if os.path.exists(kw_path):
        kw_df = pd.read_csv(kw_path)
        for _, row in kw_df.iterrows():
            kw_data[row["segment_id"]] = row

    results = []
    all_scores = []

    for _, seg in segments_df.iterrows():
        seg_id = seg["segment_id"]
        text = str(seg["text"])
        duration = float(seg["end_time"]) - float(seg["start_time"]) if pd.notna(seg["end_time"]) else 1
        n_words = len(text.split())

        # Keyword density
        kw_info = kw_data.get(seg_id, {})
        n_kw = kw_info.get("n_keywords", 0) if isinstance(kw_info, dict) else getattr(kw_info, "n_keywords", 0)
        kw_density = n_kw / max(duration, 0.1)

        # Speech rate (words per second)
        speech_rate = n_words / max(duration, 0.1)

        # Simple heuristic score
        score = 0.4 * min(kw_density / 2.0, 1.0) + 0.3 * min(speech_rate / 4.0, 1.0) + 0.3 * min(n_words / 20, 1.0)
        all_scores.append(score)

    # Map to 1-5 via percentile bins
    if all_scores:
        percentiles = np.percentile(all_scores, [20, 40, 60, 80])
        for i, (_, seg) in enumerate(segments_df.iterrows()):
            score = all_scores[i]
            if score <= percentiles[0]:
                imp = 1
            elif score <= percentiles[1]:
                imp = 2
            elif score <= percentiles[2]:
                imp = 3
            elif score <= percentiles[3]:
                imp = 4
            else:
                imp = 5

            results.append({
                "segment_id": seg["segment_id"],
                "importance": imp,
                "reason": "heuristic",
                "backend": "heuristic",
                "is_reliable": False,
                "run1_rating": imp,
                "run2_rating": None,
            })
    else:
        for _, seg in segments_df.iterrows():
            results.append({
                "segment_id": seg["segment_id"],
                "importance": 3,
                "reason": "default",
                "backend": "heuristic",
                "is_reliable": False,
                "run1_rating": 3,
                "run2_rating": None,
            })

    return results


def _parse_importance_json(text: str, segments_df: pd.DataFrame) -> list:
    """Parse JSON from LLM response for importance ratings."""
    # Try to extract JSON array from response
    text = text.strip()

    # Find JSON array in text
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            if isinstance(data, list):
                result = []
                for item in data:
                    seg_id = item.get("segment_id", 0)
                    imp = item.get("importance", 3)
                    imp = max(1, min(5, int(imp)))
                    reason = item.get("reason", "")
                    result.append({
                        "segment_id": seg_id,
                        "importance": imp,
                        "reason": reason,
                    })
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: default rating 3 for all segments
    logger.warning("Could not parse importance JSON, defaulting to 3")
    return [{"segment_id": int(row["segment_id"]), "importance": 3, "reason": "parse_error"}
            for _, row in segments_df.iterrows()]
