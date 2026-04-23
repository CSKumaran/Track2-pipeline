"""CLI entry point – run the full temporal contiguity pipeline on one or more videos."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from pipeline.config import Config, config_from_args
from pipeline.utils.io_utils import video_output_dir, video_stem

from pipeline.stages.asr_whisper import run_asr_whisper
from pipeline.stages.scene_detection import run_scene_detection
from pipeline.stages.vlm_concepts import label_scene_concepts
from pipeline.stages.text_units import build_text_units_and_embeddings
from pipeline.stages.alignment import align_scenes_to_narration, save_alignment_results
from pipeline.stages.scoring import (
    compute_scores_for_alignments,
    compute_video_aggregates,
    save_scoring_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Temporal Contiguity Scoring Pipeline for instructional videos."
    )
    p.add_argument(
        "--video", required=True, nargs="+",
        help="Path(s) to input video file(s).",
    )
    p.add_argument(
        "--output-root", default=None,
        help="Root output directory (default: outputs).",
    )
    p.add_argument(
        "--scene-thresholds", type=float, nargs="+", default=None,
        help="SSIM thresholds for scene change detection (default: 0.85 0.90 0.95). "
             "Lower = fewer scenes, higher = more scenes.",
    )
    p.add_argument(
        "--sample-interval", type=float, default=None,
        help="Seconds between extracted frames (default: 2.0).",
    )
    p.add_argument(
        "--omega", type=float, default=None,
        help="Half-window in seconds for narration search (default: 5.0).",
    )
    p.add_argument(
        "--use-api-embeddings", action="store_true",
        help="Use API embeddings instead of local sentence-transformers.",
    )
    p.add_argument(
        "--vlm-mode", choices=["ollama", "offline_llava", "api"], default=None,
        help="VLM backend (default: ollama).",
    )
    p.add_argument(
        "--whisper-model", default=None,
        help="Whisper model size (default: medium).",
    )
    p.add_argument(
        "--skip-vlm", action="store_true",
        help="Skip Stage 3 (VLM concept labelling) if VLM is not available. "
             "OCR will still run if enabled.",
    )
    p.add_argument(
        "--no-ocr", action="store_true",
        help="Disable OCR-based features (scene detection and concept extraction).",
    )
    p.add_argument(
        "--no-clip", action="store_true",
        help="Disable CLIP-based visual alignment (Track B).",
    )
    p.add_argument(
        "--share-asr-from", type=str, default=None,
        help="Reuse ASR outputs from another video (e.g., A0). "
             "Copies transcript files from that video's output dir "
             "so all delay-variant videos share identical word timestamps.",
    )
    return p


def _uniform_fallback_scenes(
    total_duration: float, config: Config, segment_dur: float = 10.0
) -> pd.DataFrame:
    """Create uniform fixed-length scenes when PySceneDetect finds nothing.

    One set of scenes is created per threshold so downstream code that
    groups by threshold still works.
    """
    import numpy as np
    from pipeline.utils.io_utils import safe_write_csv

    boundaries = np.arange(0, total_duration, segment_dur)
    rows = []
    for T in config.SCENE_THRESHOLDS:
        for sid, t_start in enumerate(boundaries):
            t_end = min(t_start + segment_dur, total_duration)
            if t_end - t_start < 1.0:
                continue
            rows.append({
                "scene_id": sid,
                "t_start": round(t_start, 4),
                "t_end": round(t_end, 4),
                "duration": round(t_end - t_start, 4),
                "t_vis": round(0.5 * (t_start + t_end), 4),
                "threshold": T,
            })
    df = pd.DataFrame(rows)
    logger.info("Uniform fallback: %d scenes per threshold (%.0fs each)",
                len(boundaries), segment_dur)
    return df


def _copy_asr_outputs(source_dir: str, target_dir: str) -> None:
    """Copy ASR transcript files from source video for shared-audio experiments."""
    import shutil
    files = ["transcript_words.csv", "transcript_segments.csv",
             "transcript_segments_improved.csv"]
    for fname in files:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(target_dir, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source ASR file not found: {src}")
        shutil.copy2(src, dst)
        logger.info("  Shared ASR: copied %s", fname)


def process_video(video_path: str, config: Config, skip_vlm: bool = False,
                  share_asr_from: str | None = None) -> None:
    """Run the full pipeline on a single video."""
    vid_stem = video_stem(video_path)
    out_dir = video_output_dir(config.OUTPUT_ROOT, video_path)
    logger.info("═══ Processing: %s → %s ═══", vid_stem, out_dir)

    # ── Stage 1: ASR ─────────────────────────────────────────────────
    if share_asr_from:
        source_dir = os.path.join(config.OUTPUT_ROOT, share_asr_from)
        os.makedirs(out_dir, exist_ok=True)
        logger.info("── Stage 1: Sharing ASR from %s ──", share_asr_from)
        _copy_asr_outputs(source_dir, out_dir)
    else:
        logger.info("── Stage 1: ASR (Whisper %s) ──", config.WHISPER_MODEL)
    asr_stats = run_asr_whisper(video_path, out_dir, model_name=config.WHISPER_MODEL)
    logger.info("ASR complete: %d words, %d→%d segments (original→improved), %.1fs",
                asr_stats["n_words"],
                asr_stats["n_segments_original"],
                asr_stats["n_segments_improved"],
                asr_stats["total_duration"])

    # ── Stage 2: Scene detection ─────────────────────────────────────
    logger.info("── Stage 2: Scene detection (thresholds=%s, OCR=%s) ──",
                config.SCENE_THRESHOLDS, config.OCR_ENABLED)
    scenes_df = run_scene_detection(
        video_path, out_dir,
        thresholds=config.SCENE_THRESHOLDS,
        min_duration=config.MIN_SCENE_DURATION,
        sample_interval=config.SAMPLE_INTERVAL,
        ocr_enabled=config.OCR_ENABLED,
        ocr_min_confidence=config.OCR_MIN_CONFIDENCE,
        min_new_words_for_scene=config.MIN_NEW_WORDS_FOR_SCENE,
    )
    for t in config.SCENE_THRESHOLDS:
        sub = scenes_df[scenes_df["threshold"] == t] if "threshold" in scenes_df.columns else scenes_df
        logger.info("  threshold=%.2f → %d scenes", t, len(sub))

    # If no scenes were detected at any threshold, create uniform segments
    if scenes_df.empty or len(scenes_df) == 0:
        logger.warning(
            "No scenes detected at any threshold. "
            "Falling back to uniform %.0fs segments.", config.MIN_SCENE_DURATION * 3
        )
        scenes_df = _uniform_fallback_scenes(
            asr_stats["total_duration"], config
        )

    # ── Stage 3: VLM concept labelling ───────────────────────────────
    # Use the improved (sentence-boundary) segmentation for all downstream stages
    transcript_seg_path = os.path.join(out_dir, "transcript_segments_improved.csv")
    transcript_seg_df = pd.read_csv(transcript_seg_path)

    if skip_vlm:
        logger.info("── Stage 3: VLM skipped, running OCR only (OCR=%s) ──",
                     config.OCR_ENABLED)
    else:
        logger.info("── Stage 3: VLM concept labelling (mode=%s) + OCR ──",
                     config.VLM_MODE)

    try:
        scenes_df = label_scene_concepts(
            scenes_df, video_path, out_dir,
            vlm_mode=config.VLM_MODE,
            transcript_segments_df=transcript_seg_df,
            skip_vlm=skip_vlm,
            ocr_enabled=config.OCR_ENABLED,
            ocr_min_confidence=config.OCR_MIN_CONFIDENCE,
        )
    except NotImplementedError as e:
        logger.warning("VLM not available (%s). Using placeholder concepts.", e)
        scenes_df = scenes_df.copy()
        scenes_df["ocr_text"] = ""
        scenes_df["vlm_text"] = "PLACEHOLDER - VLM not available"
        scenes_df["concept_text"] = "PLACEHOLDER - VLM not available"

    # ── Stage 4a: Text units & embeddings ────────────────────────────
    logger.info("── Stage 4a: Text units & embeddings ──")
    segment_meta_df, segment_embeddings = build_text_units_and_embeddings(
        transcript_seg_path, out_dir, config,
    )

    # ── Stage 4b: Alignment (global best + word-level precision) ─────
    # Build frames_dir for CLIP Track B (uses per-threshold keyframes)
    # Since alignment runs across all thresholds, pass the first threshold's
    # frames dir — keyframes are named by scene_id which is unique per threshold.
    # We'll pass frames_dir per-threshold below.
    logger.info("── Stage 4b: Narration alignment (word-level, CLIP=%s) ──",
                config.CLIP_ENABLED)

    # Run alignment per-threshold so each gets its own frames_dir
    alignment_parts = []
    for T in config.SCENE_THRESHOLDS:
        t_scenes = scenes_df[scenes_df["threshold"] == T] if "threshold" in scenes_df.columns else scenes_df
        if len(t_scenes) == 0:
            continue
        t_frames_dir = os.path.join(out_dir, f"frames_threshold_{T}")
        part = align_scenes_to_narration(
            t_scenes, segment_meta_df, segment_embeddings, config,
            output_dir=out_dir,
            frames_dir=t_frames_dir,
        )
        alignment_parts.append(part)
    alignment_df = pd.concat(alignment_parts, ignore_index=True) if alignment_parts else pd.DataFrame()
    save_alignment_results(alignment_df, out_dir)

    # ── Stage 5: Scoring ─────────────────────────────────────────────
    logger.info("── Stage 5: Scoring ──")
    scores_df = compute_scores_for_alignments(alignment_df, config)
    video_agg_df = compute_video_aggregates(scores_df, vid_stem)

    # Build concept_texts lookup for the dashboard report
    # Pass OCR, VLM, and new_words separately for distinct columns
    concept_texts: dict[int, dict] = {}
    if "concept_text" in scenes_df.columns:
        for _, row in scenes_df.iterrows():
            sid = int(row["scene_id"])
            concept_texts[sid] = {
                "ocr_text": str(row.get("ocr_text", "")).strip(),
                "vlm_text": str(row.get("vlm_text", "")).strip(),
                "new_words": str(row.get("new_ocr_words", "")).strip(),
            }

    save_scoring_results(
        scores_df, video_agg_df, out_dir,
        video_name=vid_stem,
        vlm_mode=config.VLM_MODE,
        concept_texts=concept_texts,
    )

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("═══ Summary for %s ═══", vid_stem)
    for _, agg_row in video_agg_df.iterrows():
        dt_str = ""
        if agg_row.get("mean_delta_t") is not None:
            dt_str = (f", Δt: mean={agg_row['mean_delta_t']:.2f}, "
                      f"SD={agg_row['sd_delta_t']:.2f}")
        n_nc = int(agg_row.get("n_non_content", 0))
        nc_str = f", {n_nc} non-content" if n_nc > 0 else ""
        logger.info(
            "  threshold=%s: %d scenes (%d content, %d matched, %d no_match%s), "
            "mean_S_final=%.1f%s, "
            "%%Optimal=%.0f, %%Suboptimal=%.0f, "
            "%%Disruptive=%.0f, %%Unacceptable=%.0f",
            agg_row["threshold"], agg_row["n_scenes"],
            agg_row.get("n_content", agg_row["n_matched"]),
            agg_row["n_matched"], agg_row["n_no_match"], nc_str,
            agg_row["mean_S_final"], dt_str,
            agg_row["pct_Optimal"], agg_row["pct_Suboptimal"],
            agg_row["pct_Disruptive"], agg_row["pct_Unacceptable"],
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)

    skip_vlm = getattr(args, "skip_vlm", False)
    share_asr_from = getattr(args, "share_asr_from", None)

    for vpath in args.video:
        if not os.path.isfile(vpath):
            logger.error("Video not found: %s", vpath)
            sys.exit(1)
        process_video(vpath, config, skip_vlm=skip_vlm,
                      share_asr_from=share_asr_from)

    logger.info("All videos processed.")


if __name__ == "__main__":
    main()
