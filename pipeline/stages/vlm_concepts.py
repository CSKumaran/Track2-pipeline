"""Stage 3 – VLM concept labelling + OCR for detected scenes."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

import pandas as pd

from pipeline.utils.ffmpeg_utils import extract_frame_at_time
from pipeline.utils.io_utils import ensure_dir, safe_write_csv
from pipeline.utils.viz_reports import generate_concepts_html_report
from pipeline.stages.scene_detection import _run_ocr_on_frame, _get_ocr_reader

logger = logging.getLogger(__name__)

VLM_PROMPT = (
    "What text and key concepts are visible in this instructional video frame? "
    "List the main readable text and any technical concepts shown. "
    "Be specific and concise."
)

# Marker for non-content frames
NON_CONTENT_MARKER = "NON_CONTENT"


# ── Public API ───────────────────────────────────────────────────────

def label_scene_concepts(
    scenes_df: pd.DataFrame,
    video_path: str,
    video_output_dir: str,
    vlm_mode: str = "offline_llava",
    transcript_segments_df: Optional[pd.DataFrame] = None,
    skip_vlm: bool = False,
    ocr_enabled: bool = True,
    ocr_min_confidence: float = 0.3,
) -> pd.DataFrame:
    """Label each scene with OCR text and/or a VLM concept sentence.

    For each scene:
      a. Run OCR on the keyframe -> extract readable text (ocr_text)
      b. Run VLM on the keyframe -> describe visual content (vlm_text)
      c. Merge: combined concept_text = ocr_text + vlm_text

    Parameters
    ----------
    scenes_df : pd.DataFrame
        Must contain: threshold, scene_id, t_start, t_end, t_vis, duration.
    video_path : str
        Original video (used to extract additional frames if needed).
    video_output_dir : str
        Per-video output root.
    vlm_mode : str
        ``"offline_llava"``, ``"ollama"``, or ``"api"``.
    transcript_segments_df : pd.DataFrame | None
        If provided, used to annotate the HTML report with nearby transcript.
    skip_vlm : bool
        If True, skip VLM but still run OCR.
    ocr_enabled : bool
        Whether to run OCR on keyframes.
    ocr_min_confidence : float
        Minimum OCR confidence to accept a word.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added ``ocr_text``, ``vlm_text``, and
        ``concept_text`` (combined) columns.
    """
    results: list[pd.DataFrame] = []

    for threshold, group in scenes_df.groupby("threshold"):
        # ── Cache check: reuse existing concept CSV if it matches ──────
        cached_csv = os.path.join(
            video_output_dir, f"scene_concepts_threshold_{threshold}.csv"
        )
        if os.path.isfile(cached_csv):
            cached = pd.read_csv(cached_csv)
            # Reuse if same number of scenes AND has the new columns
            if (len(cached) == len(group)
                    and "concept_text" in cached.columns
                    and "ocr_text" in cached.columns
                    and "vlm_text" in cached.columns):
                logger.info("Reusing cached concepts: %s", cached_csv)
                sub = group.copy()
                sub["ocr_text"] = cached["ocr_text"].values
                sub["vlm_text"] = cached["vlm_text"].values
                sub["concept_text"] = cached["concept_text"].values
                results.append(sub)
                continue

        frames_dir = os.path.join(video_output_dir, f"frames_threshold_{threshold}")
        ensure_dir(frames_dir)

        ocr_texts: list[str] = []
        vlm_texts: list[str] = []
        combined_texts: list[str] = []

        for _, row in group.iterrows():
            sid = int(row["scene_id"])
            dur = row["duration"]
            t_start = row["t_start"]

            # Compute 3 sample times within the scene
            t1 = t_start + 0.1 * dur
            t2 = t_start + 0.5 * dur
            t3 = t_start + 0.9 * dur

            # Ensure frames exist
            frame_paths: list[str] = []
            for idx, t in enumerate([t1, t2, t3]):
                fp = os.path.join(frames_dir, f"scene_{sid}_sample_{idx}.jpg")
                extract_frame_at_time(video_path, t, fp)
                frame_paths.append(fp)

            # ── OCR on keyframe at t_vis (matches displayed frame) ─────
            ocr_text = ""
            if ocr_enabled and _get_ocr_reader() is not None:
                # Use the scene keyframe (extracted at t_vis) so OCR text
                # matches the thumbnail shown in the dashboard report.
                keyframe_path = os.path.join(frames_dir, f"scene_{sid}.jpg")
                if os.path.exists(keyframe_path):
                    ocr_text = _run_ocr_on_frame(keyframe_path, ocr_min_confidence)
                else:
                    # Fallback to middle sample frame
                    mid_frame = frame_paths[len(frame_paths) // 2]
                    ocr_text = _run_ocr_on_frame(mid_frame, ocr_min_confidence)

            # ── VLM on keyframe ───────────────────────────────────────
            vlm_text = ""
            if not skip_vlm:
                raw_response = call_vlm_on_frames(frame_paths, VLM_PROMPT, vlm_mode)
                vlm_text = _postprocess_vlm_response(raw_response)
            else:
                vlm_text = "PLACEHOLDER - VLM not run"

            # ── Combine ───────────────────────────────────────────────
            parts = []
            if ocr_text.strip():
                parts.append(ocr_text.strip())
            if vlm_text.strip() and vlm_text != "PLACEHOLDER - VLM not run":
                parts.append(vlm_text.strip())
            concept = "; ".join(parts) if parts else vlm_text

            logger.info("  Scene %d OCR: %s", sid, ocr_text[:60] if ocr_text else "(none)")
            logger.info("  Scene %d VLM: %s", sid, vlm_text[:60])

            ocr_texts.append(ocr_text)
            vlm_texts.append(vlm_text)
            combined_texts.append(concept)

        sub = group.copy()
        sub["ocr_text"] = ocr_texts
        sub["vlm_text"] = vlm_texts
        sub["concept_text"] = combined_texts

        # Save per-threshold CSV with all columns
        csv_path = os.path.join(
            video_output_dir, f"scene_concepts_threshold_{threshold}.csv"
        )
        save_cols = [
            "threshold", "scene_id", "t_start", "t_end", "t_vis",
            "ocr_text", "vlm_text", "concept_text", "new_ocr_words",
        ]
        # Only include columns that exist
        save_cols = [c for c in save_cols if c in sub.columns]
        safe_write_csv(sub[save_cols], csv_path)

        # Generate HTML report
        html_path = os.path.join(
            video_output_dir, f"reports_concepts_threshold_{threshold}.html"
        )
        generate_concepts_html_report(
            sub, frames_dir, html_path,
            transcript_segments_df=transcript_segments_df,
        )

        results.append(sub)

    if results:
        return pd.concat(results, ignore_index=True)
    return scenes_df.assign(ocr_text="", vlm_text="", concept_text="")


# ── Post-processing ──────────────────────────────────────────────────

def _postprocess_vlm_response(raw: str) -> str:
    """Clean VLM response: detect NON_CONTENT or extract clean keywords.

    Two-phase approach:
    1. NON_CONTENT detection via heuristics on the response text.
    2. If content, clean up boilerplate to extract the useful terms.
    """
    text = raw.strip()

    # ── Phase 1: Non-content detection ────────────────────────────────
    # Check if VLM explicitly said non-content
    if "NON_CONTENT" in text.upper().replace("NON-CONTENT", "NON_CONTENT"):
        return NON_CONTENT_MARKER

    text_lower = text.lower()

    # Check for responses that indicate the VLM couldn't find content
    refusal_signals = [
        "doesn't contain any text",
        "does not contain any text",
        "no text or visual elements",
        "appears to be a graphic rather",
        "no readable text",
        "cannot determine",
        "i cannot",
        "no instructional content",
    ]
    for signal in refusal_signals:
        if signal in text_lower:
            return NON_CONTENT_MARKER

    # Check for frames that are purely non-content (channel branding, etc.)
    # Only flag if the ENTIRE response is about non-content things
    non_content_only_signals = [
        "channel logo", "subscribe button", "intro screen",
        "outro screen", "end screen", "channel name only",
    ]
    # Only trigger if the response is short AND matches
    if len(text) < 80:
        for signal in non_content_only_signals:
            if signal in text_lower:
                return NON_CONTENT_MARKER

    # ── Phase 2: Clean up to extract useful concept text ──────────────
    cleaned = text

    # Strip common VLM boilerplate preambles
    boilerplate_prefixes = [
        "the primary concept being explained",
        "the key technical terms",
        "the instructional content shows",
        "the main readable text",
        "the text visible in this",
        "the image contains the following text:",
        "the visible text reads:",
        "this frame shows",
        "the frame shows",
        "in this frame,",
        "here are the key",
        "key terms:",
        "terms:",
    ]
    for prefix in boilerplate_prefixes:
        idx = cleaned.lower().find(prefix)
        if idx != -1:
            after = cleaned[idx + len(prefix):]
            # Find the end of the preamble
            for sep in [":", " is ", " are ", ". ", "- "]:
                sep_idx = after.find(sep)
                if sep_idx != -1 and sep_idx < 30:
                    candidate = after[sep_idx + len(sep):].strip()
                    if len(candidate) > 5:
                        cleaned = candidate
                    break

    # Remove surrounding quotes
    cleaned = cleaned.strip('"\'')

    # Remove leading "The text" / "The image" if still present
    for starter in ["the text ", "the image ", "the frame "]:
        if cleaned.lower().startswith(starter):
            rest = cleaned[len(starter):]
            for verb in ["reads ", "shows ", "contains ", "displays "]:
                if rest.lower().startswith(verb):
                    cleaned = rest[len(verb):].strip('"\'.:- ')
                    break

    # Truncate very long responses to the most useful part
    if len(cleaned) > 250:
        lines = cleaned.split("\n")
        # Keep lines that look like keyword lists or short descriptions
        useful = [l.strip().strip("-•*").strip() for l in lines
                  if 5 < len(l.strip()) < 150]
        if useful:
            cleaned = "; ".join(useful[:5])

    return cleaned if cleaned else text


# ── VLM dispatch ─────────────────────────────────────────────────────

def call_vlm_on_frames(
    frame_paths: list[str],
    prompt: str,
    mode: str,
) -> str:
    """Call a VLM to describe the concept shown in *frame_paths*.

    Parameters
    ----------
    frame_paths : list[str]
        JPEG paths for frames sampled from one scene.
    prompt : str
        The instruction prompt.
    mode : str
        ``"offline_llava"`` or ``"api"``.

    Returns
    -------
    str
        The VLM's concept sentence.
    """
    if mode == "ollama":
        return _call_ollama(frame_paths, prompt)
    elif mode == "offline_llava":
        return _call_llava_offline(frame_paths, prompt)
    elif mode == "api":
        return _call_vlm_api(frame_paths, prompt)
    else:
        raise ValueError(f"Unknown VLM mode: {mode!r}")


# ── Ollama (local) ───────────────────────────────────────────────────

# Read Ollama settings from config at module level is tricky, so we
# accept them via environment or use defaults.  The label_scene_concepts
# function passes them through; for direct calls we fall back to defaults.

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llava:7b")


def _call_ollama(
    frame_paths: list[str],
    prompt: str,
    model: str | None = None,
    base_url: str | None = None,
) -> str:
    """Call a local Ollama vision model (e.g. llava:13b).

    Ollama's LLaVA supports one image per request, so we send the
    middle frame (most representative of the scene).
    """
    model = model or _OLLAMA_MODEL
    base_url = (base_url or _OLLAMA_BASE_URL).rstrip("/")
    url = f"{base_url}/api/generate"

    # Use the middle frame (index 1 of 3) as the most representative
    mid_frame = frame_paths[len(frame_paths) // 2]
    with open(mid_frame, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 150,
        },
    }).encode("utf-8")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body.get("response", "").strip()
            if not text:
                logger.warning("Ollama returned empty response for %s", mid_frame)
                text = "EMPTY_VLM_RESPONSE"
            return text
        except Exception as e:
            # Read error body if available (HTTP 500 etc.)
            err_body = ""
            if hasattr(e, "read"):
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
            logger.warning("Ollama attempt %d/%d failed: %s %s",
                           attempt, max_retries, e, err_body[:200])
            if attempt == max_retries:
                raise ConnectionError(
                    f"Ollama failed after {max_retries} attempts at {base_url}.\n"
                    f"Is Ollama running?  Start with: ollama serve\n"
                    f"Model: ollama pull {model}\n"
                    f"Last error: {e}\n{err_body[:500]}"
                ) from e
            import time
            time.sleep(5 * attempt)  # back off: 5s, 10s


# ── Offline LLaVA placeholder ────────────────────────────────────────

def _call_llava_offline(frame_paths: list[str], prompt: str) -> str:
    """Placeholder for local LLaVA inference.

    To implement with llava / llama-cpp-python / transformers:
    ─────────────────────────────────────────────────────────
    from llava.model import LlavaForConditionalGeneration
    from llava.conversation import conv_templates
    from PIL import Image

    model = LlavaForConditionalGeneration.from_pretrained(
        "liuhaotian/llava-v1.5-7b", device_map="auto"
    )
    processor = ...  # load matching processor

    images = [Image.open(p) for p in frame_paths]
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=128)
    concept_text = processor.decode(output_ids[0], skip_special_tokens=True)
    return concept_text
    ─────────────────────────────────────────────────────────
    """
    raise NotImplementedError(
        "Offline LLaVA inference is not yet wired up. "
        "See the docstring of _call_llava_offline() for the integration pattern. "
        "You can also set --vlm-mode api to use an API-based VLM."
    )


# ── API VLM stub ─────────────────────────────────────────────────────

def _call_vlm_api(frame_paths: list[str], prompt: str) -> str:
    """Stub for calling a cloud VLM API (Claude, GPT-4o, Gemini).

    To implement:
    ─────────────────────────────────────────────────────────
    import anthropic, base64

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    image_contents = []
    for path in frame_paths:
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        image_contents.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": image_contents + [{"type": "text", "text": prompt}],
        }],
    )
    return response.content[0].text
    ─────────────────────────────────────────────────────────
    """
    raise NotImplementedError(
        "API VLM is not yet implemented. "
        "See the docstring of _call_vlm_api() for the integration pattern."
    )
