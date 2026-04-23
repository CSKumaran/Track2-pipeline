"""Stage 6: Pedagogical Importance Rating [v2.2].

3-tier backend:
  1. Gemini API (if API key available)
  2. Local LLM via Ollama (if running)
  3. Heuristic fallback (always available)

[v2.2] Changes from v2.1:
  - Uses keywords.csv from Stage 5 for heuristic keyword density
  - Compressed weight range for heuristic backend (0.8-1.2 vs 0.3-2.0)
  - Diagnostics output with per-segment scoring breakdown
  - is_reliable flag: False for ALL heuristic ratings (honest about uncertainty)
"""

import logging
import os
import json
import numpy as np
import pandas as pd

from ..config import Config
from ..utils.io_utils import save_csv, cache_exists

logger = logging.getLogger(__name__)

IMPORTANCE_PROMPT_TEMPLATE = """You are an expert instructional designer analyzing an educational video transcript. For each transcript segment, rate its pedagogical importance on a scale of 1-5:

1 = Low: intro/outro, greetings, transitions, filler, social pleasantries
2 = Below Average: recap of known material, tangential examples, repetition
3 = Average: supporting explanation, context building, examples of concepts
4 = Above Average: key concept introduction, important worked examples, critical distinctions
5 = Critical: core concept derivation, formula explanation, step-by-step procedure, definition of fundamental terms

Consider:
- Conceptual density: how many new ideas per segment?
- Visual dependency: does this segment reference something on screen?
- Sequential dependency: would skipping this break understanding of later segments?
- Assessment relevance: would this appear on an exam?

Segments:
{segments}

Return ONLY a JSON array, no other text:
[{{"segment_id": 1, "importance": 3, "reason": "brief reason"}}, ...]"""


def run_stage6(output_dir: str, cfg: Config, diag=None) -> dict:
    """Run pedagogical importance rating."""
    cache_file = "pedagogical_importance.csv"
    if cache_exists(output_dir, [cache_file]):
        logger.info("Stage 6: cache hit, skipping importance rating")
        return {cache_file: os.path.join(output_dir, cache_file)}

    segments_df = pd.read_csv(
        os.path.join(output_dir, "transcript_segments_improved.csv")
    )

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
            logger.info("Importance backend: gemini (%d segments)", len(ratings))

    if ratings is None and backend in ("auto", "local_llm"):
        ratings = _tier2_local_llm(prompt, segments_df, cfg)
        if ratings is not None:
            logger.info("Importance backend: local_llm (%d segments)", len(ratings))

    if ratings is None:
        ratings = _tier3_heuristic(segments_df, output_dir, cfg)
        logger.info("Importance backend: heuristic (%d segments)", len(ratings))

    df = pd.DataFrame(ratings)
    save_csv(df, os.path.join(output_dir, cache_file))
    logger.info("Stage 6: %d segments rated", len(df))

    # Distribution summary
    if not df.empty:
        dist = df["importance"].value_counts().sort_index()
        for level, count in dist.items():
            logger.info("  Importance %d: %d segments (%.1f%%)",
                        level, count, 100.0 * count / len(df))

    # Diagnostics
    if diag and not df.empty:
        diag_data = {
            "n_segments": len(df),
            "backend": df["backend"].iloc[0] if "backend" in df.columns else "unknown",
            "distribution": {
                str(k): int(v) for k, v in
                df["importance"].value_counts().sort_index().items()
            },
            "mean_importance": float(df["importance"].mean()),
            "n_reliable": int(df["is_reliable"].sum()) if "is_reliable" in df.columns else 0,
            "n_unreliable": int((~df["is_reliable"]).sum()) if "is_reliable" in df.columns else len(df),
            "per_segment": [
                {
                    "segment_id": int(row["segment_id"]),
                    "importance": int(row["importance"]),
                    "reason": str(row.get("reason", "")),
                    "backend": str(row.get("backend", "")),
                    "is_reliable": bool(row.get("is_reliable", False)),
                }
                for _, row in df.iterrows()
            ],
        }
        diag.write_json("stage6_importance.json", diag_data)

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
            resp2 = query_gemini_text(prompt + "\n(Second independent run for reliability check)", cfg)
            ratings2 = _parse_importance_json(resp2, segments_df)

            # Merge: average when disagree
            result = []
            r2_map = {r["segment_id"]: r for r in ratings2}
            for r1 in ratings1:
                r2_match = r2_map.get(r1["segment_id"])
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
    """Tier 3: Heuristic fallback.

    [v2.2] Uses keywords.csv from Stage 5 for keyword density.
    All ratings marked is_reliable=False (honest about heuristic uncertainty).
    """
    # Load keyword data from Stage 5
    kw_path = os.path.join(output_dir, "keywords.csv")
    kw_per_segment = {}
    if os.path.exists(kw_path):
        kw_df = pd.read_csv(kw_path)
        if not kw_df.empty and "segment_id" in kw_df.columns:
            kw_per_segment = kw_df.groupby("segment_id").size().to_dict()
            logger.info("Heuristic: using %d keywords across %d segments",
                        len(kw_df), len(kw_per_segment))

    results = []
    all_scores = []

    for _, seg in segments_df.iterrows():
        seg_id = seg["segment_id"]
        text = str(seg["text"])
        duration = float(seg["end_time"]) - float(seg["start_time"]) if pd.notna(seg["end_time"]) else 1
        n_words = len(text.split())

        # Keyword density from Stage 5
        n_kw = kw_per_segment.get(seg_id, 0)
        kw_density = n_kw / max(duration, 0.1)

        # Speech rate (words per second)
        speech_rate = n_words / max(duration, 0.1)

        # Heuristic features
        # 1. Keyword density (normalized) — 40%
        f_kw = min(kw_density / 2.0, 1.0)

        # 2. Speech rate (normalized) — 30%
        f_rate = min(speech_rate / 4.0, 1.0)

        # 3. Word count (longer segments tend to be more substantive) — 20%
        f_words = min(n_words / 25, 1.0)

        # 4. Technical indicator: presence of domain terms, numbers — 10%
        has_technical = any(c.isdigit() for c in text) or \
            any(w in text.lower() for w in [
                "algorithm", "function", "equation", "formula", "definition",
                "theorem", "step", "procedure", "method", "optimization",
                "convergence", "objective", "constraint", "parameter",
            ])
        f_tech = 1.0 if has_technical else 0.0

        score = 0.40 * f_kw + 0.30 * f_rate + 0.20 * f_words + 0.10 * f_tech
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
                "heuristic_score": round(all_scores[i], 4),
            })
    else:
        for _, seg in segments_df.iterrows():
            results.append({
                "segment_id": seg["segment_id"],
                "importance": 3,
                "reason": "default (no segments)",
                "backend": "heuristic",
                "is_reliable": False,
                "run1_rating": 3,
                "run2_rating": None,
                "heuristic_score": 0.0,
            })

    return results


def _parse_importance_json(text: str, segments_df: pd.DataFrame) -> list:
    """Parse JSON from LLM response for importance ratings."""
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
    return [
        {"segment_id": int(row["segment_id"]), "importance": 3, "reason": "parse_error"}
        for _, row in segments_df.iterrows()
    ]
