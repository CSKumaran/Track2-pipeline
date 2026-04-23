"""Stage 6: Pedagogical Importance Rating [v4.0].

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

# [v4.0] Calibrated prompt for local LLMs (7B models over-rate without distribution guidance)
IMPORTANCE_PROMPT_LOCAL_LLM = """You are an expert instructional designer analyzing an educational video transcript. For each transcript segment, rate its pedagogical importance on a scale of 1-5:

1 = Low: intro/outro, greetings, transitions, filler, social pleasantries
2 = Below Average: recap of known material, tangential examples, repetition
3 = Average: supporting explanation, context building, examples of concepts
4 = Above Average: key concept introduction, important worked examples, critical distinctions
5 = Critical: core concept derivation, formula explanation, step-by-step procedure, definition of fundamental terms

IMPORTANT calibration guidance:
- Use the FULL range 1-5. Rating everything 4 or 5 is not helpful.
- Rating 3 (Average) should be the most common — most content is supporting material.
- Reserve 5 (Critical) for segments where skipping them would make later content incomprehensible.
- Reserve 4 (Above Average) for key concept introductions and critical worked examples.
- Use 1-2 for transitions, greetings, repetition, and tangential material.
- Be discriminating: in a well-structured lecture, only a few segments contain the core ideas.

Consider:
- Conceptual density: how many new ideas per segment?
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

    # [v4.0] Load scenes for visual context (if available)
    scenes_df = None
    scenes_path = os.path.join(output_dir, "scenes.csv")
    if os.path.exists(scenes_path):
        scenes_df = pd.read_csv(scenes_path)
        scenes_df["t_start"] = pd.to_numeric(scenes_df["t_start"], errors="coerce")
        scenes_df["t_end"] = pd.to_numeric(scenes_df["t_end"], errors="coerce")

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
        ratings = _tier1_gemini(prompt, segments_df, cfg,
                                scenes_df=scenes_df, output_dir=output_dir)
        if ratings is not None:
            logger.info("Importance backend: gemini (%d segments)", len(ratings))

    if ratings is None and backend in ("auto", "local_llm"):
        ratings = _tier2_local_llm(prompt, segments_df, cfg,
                                   scenes_df=scenes_df, output_dir=output_dir)
        if ratings is not None:
            logger.info("Importance backend: local_vlm (%d segments)", len(ratings))

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


def _tier1_gemini(prompt: str, segments_df: pd.DataFrame, cfg: Config,
                  scenes_df=None, output_dir: str = None) -> list:
    """Tier 1: Gemini API.

    [v4.0] When scenes_df is available, sends keyframe images alongside the
    transcript for multimodal importance rating. Each segment gets all
    overlapping scene keyframes for complete visual context.
    """
    if not cfg.GEMINI_API_KEY:
        return None

    try:
        # [v4.0] Try multimodal first if scenes available
        if scenes_df is not None and not scenes_df.empty:
            image_paths = _collect_segment_keyframes(segments_df, scenes_df)
            if image_paths:
                from ..utils.gemini_utils import query_gemini_multimodal
                enhanced_prompt = (
                    "The following images are keyframes from the instructional video, "
                    "shown in chronological order. Use them to judge whether each segment "
                    "requires the learner to actively integrate visual and verbal information.\n\n"
                    + prompt
                )
                resp1 = query_gemini_multimodal(enhanced_prompt, image_paths, cfg)
                ratings1 = _parse_importance_json(resp1, segments_df)
                if ratings1:
                    logger.info("[v4.0] Gemini multimodal importance: %d images, %d segments",
                                len(image_paths), len(ratings1))
                    # Tag as multimodal backend
                    return [{
                        "segment_id": r["segment_id"],
                        "importance": r["importance"],
                        "reason": r.get("reason", ""),
                        "backend": "gemini_multimodal",
                        "is_reliable": True,
                        "run1_rating": r["importance"],
                        "run2_rating": None,
                    } for r in ratings1]

        # Fallback to text-only Gemini
        from ..utils.gemini_utils import query_gemini_text
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


def _tier2_local_llm(prompt: str, segments_df: pd.DataFrame, cfg: Config,
                     scenes_df=None, output_dir: str = None) -> list:
    """Tier 2: Local VLM via vLLM/SGLang (OpenAI-compatible) or Ollama fallback.

    [v4.0] Supports three backends:
    - vLLM/SGLang: OpenAI-compatible API at LOCAL_VLM_ENDPOINT (recommended).
      Serves Qwen2.5-VL-7B/72B with 3-5x faster throughput than Ollama.
      Supports multimodal (images + text) via OpenAI vision API format.
    - Ollama: Legacy support at OLLAMA_BASE_URL (slower, limited batching).
    - auto: Try vLLM first, fall back to Ollama.

    Recommended HPC setup:
      # Qwen2.5-VL-72B (best quality, needs >=40GB VRAM or 2x GPU):
      vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ --tensor-parallel-size 2
      # Qwen2.5-VL-7B (fits any GPU):
      vllm serve Qwen/Qwen2.5-VL-7B-Instruct
    """
    import requests

    backend = getattr(cfg, 'LOCAL_VLM_BACKEND', 'auto')
    result = None

    # --- vLLM/SGLang (OpenAI-compatible API) ---
    if backend in ("vllm", "auto"):
        result = _tier2_vllm(prompt, segments_df, cfg, scenes_df, output_dir)
        if result is not None:
            return result
        if backend == "vllm":
            return None  # Don't fall through if explicitly set to vllm

    # --- Ollama fallback ---
    if backend in ("ollama", "auto"):
        result = _tier2_ollama(prompt, segments_df, cfg)
        if result is not None:
            return result

    return None


def _tier2_vllm(prompt: str, segments_df: pd.DataFrame, cfg: Config,
                scenes_df=None, output_dir: str = None) -> list:
    """Tier 2a: Local LLM via vLLM/SGLang — Text-Only.

    Sends all segments in one API call (~1900 tokens, fits 4096 context).
    Uses calibrated prompt (IMPORTANCE_PROMPT_LOCAL_LLM) with distribution
    guidance to prevent 7B models from over-rating everything to 4-5.

    Visual analysis is left to Tier 1 (Gemini multimodal).
    """
    import requests

    endpoint = getattr(cfg, 'LOCAL_VLM_ENDPOINT', 'http://localhost:8000/v1')
    model_id = getattr(cfg, 'LOCAL_VLM_MODEL_ID', 'Qwen/Qwen2.5-7B-Instruct')

    # Build calibrated prompt for local LLM (stronger distribution guidance)
    seg_lines = []
    for _, seg in segments_df.iterrows():
        seg_lines.append(
            f'[segment_id: {seg["segment_id"]}] '
            f'({seg["start_time"]:.1f}s - {seg["end_time"]:.1f}s) '
            f'"{seg["text"]}"'
        )
    local_prompt = IMPORTANCE_PROMPT_LOCAL_LLM.format(segments="\n".join(seg_lines))

    # Estimate input tokens (~4 chars/token) and cap max_tokens to fit context
    max_ctx = getattr(cfg, 'LOCAL_VLM_MAX_CTX', 4096)
    est_input_tokens = len(local_prompt) // 3  # conservative estimate
    max_output = min(2048, max_ctx - est_input_tokens - 100)  # 100 token safety margin
    if max_output < 256:
        logger.warning("vLLM: prompt too long (%d est tokens) for %d context, skipping",
                       est_input_tokens, max_ctx)
        return None
    logger.info("vLLM: ~%d input tokens, max_tokens=%d (context=%d)",
                est_input_tokens, max_output, max_ctx)

    try:
        resp = requests.post(
            f"{endpoint}/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": local_prompt}],
                "temperature": 0,
                "max_tokens": max_output,
            },
            timeout=600,
        )

        logger.info("vLLM response status: %d", resp.status_code)
        if resp.status_code != 200:
            logger.warning("vLLM HTTP %d: %s", resp.status_code,
                           resp.text[:500] if resp.text else "no body")
            return None

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        logger.info("vLLM response: %d chars", len(text))

        ratings = _parse_importance_json(text, segments_df)
        if not ratings:
            logger.warning("vLLM: parsing returned no ratings. Preview: %s", text[:500])
            return None

        logger.info("vLLM (text-only): rated %d segments", len(ratings))

        # Log distribution for monitoring calibration
        dist = {}
        for r in ratings:
            dist[r["importance"]] = dist.get(r["importance"], 0) + 1
        logger.info("vLLM distribution: %s", dict(sorted(dist.items())))

        return [{
            "segment_id": r["segment_id"],
            "importance": r["importance"],
            "reason": r.get("reason", ""),
            "backend": "local_llm_vllm",
            "is_reliable": True,
            "run1_rating": r["importance"],
            "run2_rating": None,
        } for r in ratings]

    except requests.ConnectionError:
        logger.warning("vLLM endpoint not reachable at %s", endpoint)
        return None
    except Exception as e:
        logger.warning("vLLM failed: %s (type: %s)", e, type(e).__name__)
        return None


def _tier2_ollama(prompt: str, segments_df: pd.DataFrame, cfg: Config) -> list:
    """Tier 2b: Local LLM via Ollama (legacy fallback)."""
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
            if ratings:
                return [{
                    "segment_id": r["segment_id"],
                    "importance": r["importance"],
                    "reason": r.get("reason", ""),
                    "backend": "local_llm",
                    "is_reliable": True,
                    "run1_rating": r["importance"],
                    "run2_rating": None,
                } for r in ratings]
    except requests.ConnectionError:
        logger.debug("Ollama not available at %s", cfg.OLLAMA_BASE_URL)
    except Exception as e:
        logger.warning("Ollama importance failed: %s", e)

    return None


def _tier3_heuristic(segments_df: pd.DataFrame, output_dir: str, cfg: Config) -> list:
    """Tier 3: Heuristic fallback.

    [v4.0] Enhanced with CPIP-inspired signals:
    - Instructional verb detection (dependency-parsed, not just word lists)
    - Discourse signaling phrases (explicit visual-reference cues)
    - First-mention density (from Stage 5 keywords.csv)

    All ratings marked is_reliable=False (honest about heuristic uncertainty).
    """
    # Load keyword data from Stage 5
    kw_path = os.path.join(output_dir, "keywords.csv")
    kw_per_segment = {}
    first_mention_per_segment = {}
    if os.path.exists(kw_path):
        kw_df = pd.read_csv(kw_path)
        if not kw_df.empty and "segment_id" in kw_df.columns:
            kw_per_segment = kw_df.groupby("segment_id").size().to_dict()
            # [v4.0] First-mention counts per segment
            if "is_first_mention" in kw_df.columns:
                fm = kw_df[kw_df["is_first_mention"] == True]
                first_mention_per_segment = fm.groupby("segment_id").size().to_dict()
            logger.info("Heuristic: using %d keywords across %d segments (%d first mentions)",
                        len(kw_df), len(kw_per_segment),
                        sum(first_mention_per_segment.values()))

    # [v4.0] Load spaCy for instructional verb detection (dep parsing)
    nlp = None
    if getattr(cfg, 'INSTRUCTIONAL_VERB_ENABLED', True):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:
            logger.debug("spaCy not available for instructional verb detection")

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

        # === Feature 1: Keyword density (normalized) — 30% ===
        f_kw = min(kw_density / 2.0, 1.0)

        # === Feature 2: Speech rate (normalized) — 20% ===
        f_rate = min(speech_rate / 4.0, 1.0)

        # === Feature 3: Word count — 10% ===
        f_words = min(n_words / 25, 1.0)

        # === Feature 4: Instructional verb + technical object — 15% [v4.0] ===
        f_instr_verb = _detect_instructional_verb(text, nlp)

        # === Feature 5: Discourse signaling phrases — 10% [v4.0] ===
        f_discourse = _detect_discourse_signals(text)

        # === Feature 6: First-mention density — 15% [v4.0] ===
        n_first = first_mention_per_segment.get(seg_id, 0)
        f_first_mention = min(n_first / max(n_kw, 1), 1.0) if n_kw > 0 else 0.0

        score = (0.30 * f_kw + 0.20 * f_rate + 0.10 * f_words +
                 0.15 * f_instr_verb + 0.10 * f_discourse + 0.15 * f_first_mention)
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
                "reason": "heuristic_v4",
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


# =====================================================================
# [v4.0] CPIP Signal Detection Helpers
# =====================================================================

# Bloom's taxonomy verbs that signal instructional intent
_BLOOM_VERBS = {
    "define", "describe", "identify", "list", "name", "recall", "recognize",
    "explain", "summarize", "interpret", "classify", "compare", "contrast",
    "apply", "calculate", "compute", "demonstrate", "solve", "use",
    "analyze", "differentiate", "distinguish", "examine", "investigate",
    "evaluate", "assess", "justify", "argue", "critique",
    "create", "design", "develop", "formulate", "construct",
    # Common instructional verbs in STEM lectures
    "notice", "observe", "consider", "note", "look", "see", "show",
    "derive", "prove", "verify", "check", "substitute", "simplify",
}


def _detect_instructional_verb(text: str, nlp=None) -> float:
    """Detect instructional verbs governing technical objects via dependency parse.

    Returns 1.0 if instructional pattern found, 0.0 otherwise.

    With spaCy: checks if a Bloom's verb has a direct object or subject that
    contains a technical/non-common noun (not a pronoun, not a stop word).
    Without spaCy: falls back to simple keyword matching (less accurate).
    """
    text_lower = text.lower()

    if nlp is not None:
        doc = nlp(text_lower)
        for token in doc:
            if token.lemma_ in _BLOOM_VERBS and token.pos_ == "VERB":
                # Check if this verb governs a technical-looking object
                for child in token.children:
                    if child.dep_ in ("dobj", "nsubj", "attr", "pobj"):
                        # Not a pronoun, not a stop word, not too short
                        if (not child.is_stop and child.pos_ in ("NOUN", "PROPN")
                                and len(child.text) >= 4):
                            return 1.0
                        # Check children of the object (e.g., "the gradient vector")
                        for grandchild in child.children:
                            if (not grandchild.is_stop
                                    and grandchild.pos_ in ("NOUN", "PROPN", "ADJ")
                                    and len(grandchild.text) >= 4):
                                return 1.0
        return 0.0
    else:
        # Fallback: simple word matching (less precise, more recall)
        for verb in _BLOOM_VERBS:
            if verb in text_lower:
                return 0.5  # reduced score for non-parsed detection
        return 0.0


# Discourse signaling phrases that indicate visual-verbal integration demand
_DISCOURSE_PATTERNS = [
    r"\bas you can see\b",
    r"\blook(?:ing)? at\b",
    r"\bnotice (?:that|how|the)\b",
    r"\bthis is important\b",
    r"\bthe key (?:idea|point|concept|thing)\b",
    r"\bpay attention to\b",
    r"\bas (?:shown|illustrated|depicted)\b",
    r"\bin (?:this|the) (?:diagram|figure|graph|chart|table|slide|image)\b",
    r"\bon (?:the|this) (?:screen|slide|board)\b",
    r"\bhere (?:we|you|I) (?:can )see\b",
    r"\blet me show\b",
    r"\bif you look\b",
    r"\bwhat (?:we|you) see\b",
    r"\bthe formula (?:is|shows|says)\b",
]


def _detect_discourse_signals(text: str) -> float:
    """Detect discourse signaling phrases that indicate visual-verbal integration.

    Returns 1.0 if any pattern found, 0.0 otherwise.
    """
    import re
    text_lower = text.lower()
    for pattern in _DISCOURSE_PATTERNS:
        if re.search(pattern, text_lower):
            return 1.0
    return 0.0


def _collect_segment_keyframes(segments_df, scenes_df) -> list:
    """[v4.0] Collect unique keyframe paths that overlap with any segment.

    Returns a deduplicated, chronologically-ordered list of keyframe paths.
    These are sent as images in the multimodal Gemini importance prompt.
    """
    import os

    keyframes = []
    seen_paths = set()

    for _, scene in scenes_df.sort_values("t_start").iterrows():
        kf_path = str(scene.get("keyframe_path", ""))
        if kf_path and kf_path not in seen_paths and os.path.exists(kf_path):
            # Check if any segment overlaps with this scene
            s_start = float(scene["t_start"])
            s_end = float(scene["t_end"])
            overlaps = segments_df[
                (segments_df["start_time"] < s_end) &
                (segments_df["end_time"] > s_start)
            ]
            if not overlaps.empty:
                keyframes.append(kf_path)
                seen_paths.add(kf_path)

    logger.debug("[v4.0] Collected %d keyframes for multimodal importance", len(keyframes))
    return keyframes


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
