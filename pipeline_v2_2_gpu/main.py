"""Pipeline v2.2 -- CLI entry point & stage orchestration."""

import argparse
import logging
import os
import sys
import time

from .config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_v2_2")


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline v2.2 -- Temporal Contiguity Analysis"
    )
    # Input/Output
    p.add_argument("--video", nargs="+", required=True, help="Input video file(s)")
    p.add_argument("--output-root", default="outputs_v2_2", help="Root output dir")

    # Stage control
    p.add_argument("--stage", type=int, default=None,
                   help="Run only this stage (1-7). Default: run all.")

    # ASR
    p.add_argument("--asr-backend", default="whisperx",
                   choices=["whisperx", "faster-whisper"])
    p.add_argument("--whisper-model", default="medium")
    p.add_argument("--validate-timestamps-mfa", action="store_true")

    # Scene Detection
    p.add_argument("--sample-interval", type=float, default=0.5)
    p.add_argument("--max-scene-duration", type=float, default=30.0)

    # OCR
    p.add_argument("--ocr-engine", default="surya",
                   choices=["surya", "easyocr", "paddleocr"])
    p.add_argument("--no-ocr", action="store_true")

    # VLM
    p.add_argument("--vlm-mode", default="skip",
                   choices=["ollama", "gemini", "skip"])
    p.add_argument("--ollama-model", default="llava:7b")
    p.add_argument("--skip-vlm", action="store_true")

    # Gemini
    p.add_argument("--gemini-api-key", default="")
    p.add_argument("--gemini-model", default="gemini-2.0-flash")

    # Alignment [v2.2]
    p.add_argument("--no-siglip", action="store_true")
    p.add_argument("--track-c-mode", default="text_semantic")
    p.add_argument("--temporal-sigma", type=float, default=5.0)
    p.add_argument("--use-v21-timing", action="store_true",
                   help="Revert to V2.1 t_start for delta_t (for comparison)")

    # Keywords
    p.add_argument("--no-keywords", action="store_true")
    p.add_argument("--no-grounding-dino", action="store_true", default=True)
    p.add_argument("--grounding-model", default="grounding_dino")

    # Importance
    p.add_argument("--no-importance", action="store_true")
    p.add_argument("--importance-backend", default="auto",
                   choices=["gemini", "local_llm", "heuristic", "auto"])

    # Scoring [v2.2]
    p.add_argument("--score-tau", type=float, default=2.5)
    p.add_argument("--scoring-mode", default="both",
                   choices=["gaussian", "piecewise", "both"])

    # Diagnostics [v2.2]
    p.add_argument("--no-diagnostics", action="store_true")

    # Monotonicity [v2.2]
    p.add_argument("--no-monotonic-check", action="store_true")

    return p.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    cfg.ASR_BACKEND = args.asr_backend
    cfg.WHISPER_MODEL = args.whisper_model
    cfg.VALIDATE_TIMESTAMPS_MFA = args.validate_timestamps_mfa
    cfg.SAMPLE_INTERVAL = args.sample_interval
    cfg.MAX_SCENE_DURATION = args.max_scene_duration
    cfg.OCR_ENABLED = not args.no_ocr
    cfg.OCR_ENGINE = args.ocr_engine
    cfg.VLM_MODE = "skip" if args.skip_vlm else args.vlm_mode
    cfg.OLLAMA_MODEL = args.ollama_model
    cfg.GEMINI_API_KEY = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    cfg.GEMINI_MODEL = args.gemini_model
    cfg.SIGLIP_ENABLED = not args.no_siglip
    cfg.TRACK_C_MODE = args.track_c_mode
    cfg.TEMPORAL_SIGMA = args.temporal_sigma
    cfg.USE_KEYFRAME_AS_TVIS = not args.use_v21_timing
    cfg.KEYWORD_ENABLED = not args.no_keywords
    cfg.GROUNDING_DINO_ENABLED = not args.no_grounding_dino
    cfg.GROUNDING_MODEL = args.grounding_model
    cfg.IMPORTANCE_ENABLED = not args.no_importance
    cfg.IMPORTANCE_BACKEND = args.importance_backend
    cfg.SCORE_TAU = args.score_tau
    cfg.SCORING_MODE = args.scoring_mode
    cfg.OUTPUT_ROOT = args.output_root
    cfg.DIAGNOSTICS_ENABLED = not args.no_diagnostics
    cfg.MONOTONIC_CHECK_ENABLED = not args.no_monotonic_check
    return cfg


def run_pipeline(video_path: str, cfg: Config, stage_only: int = None):
    """Run pipeline on a single video. If stage_only is set, run only that stage."""
    from .utils.io_utils import get_output_dir, save_json
    from .utils.diagnostics import DiagnosticsWriter

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = get_output_dir(cfg.OUTPUT_ROOT, video_path)
    logger.info("=" * 60)
    logger.info("Processing: %s", video_name)
    logger.info("Output: %s", output_dir)
    if stage_only:
        logger.info("Running ONLY Stage %d", stage_only)
    logger.info("=" * 60)

    # Diagnostics writer
    diag = DiagnosticsWriter(output_dir, enabled=cfg.DIAGNOSTICS_ENABLED)
    timings = {}

    def should_run(stage_num):
        return stage_only is None or stage_only == stage_num

    # Stage 1: ASR
    if should_run(1):
        t0 = time.time()
        logger.info("--- Stage 1: ASR Transcription ---")
        from .stages.asr_whisper import run_stage1
        run_stage1(video_path, output_dir, cfg, diag=diag)
        timings["stage1_asr"] = time.time() - t0

    # Stage 2: Scene Detection
    if should_run(2):
        t0 = time.time()
        logger.info("--- Stage 2: Scene Detection ---")
        from .stages.scene_detection import run_stage2
        run_stage2(video_path, output_dir, cfg, diag=diag)
        timings["stage2_scene_detection"] = time.time() - t0

    # Stage 5: Keyword Extraction [v2.2: runs BEFORE alignment]
    # In transcript-first architecture: 1→2→5→3→4→6→7
    if should_run(5):
        t0 = time.time()
        logger.info("--- Stage 5: Keyword Extraction ---")
        from .stages.keyword_extraction import run_stage5
        run_stage5(output_dir, cfg, diag=diag)
        timings["stage5_keywords"] = time.time() - t0

    # Stage 4: Alignment [v2.2: transcript-first, asymmetric]
    if should_run(4):
        t0 = time.time()
        logger.info("--- Stage 4: Alignment (Transcript-First) ---")
        from .stages.alignment import run_stage4
        run_stage4(output_dir, cfg, diag=diag)
        timings["stage4_alignment"] = time.time() - t0

    # Stage 6: Pedagogical Importance Rating
    if should_run(6):
        t0 = time.time()
        logger.info("--- Stage 6: Pedagogical Importance ---")
        from .stages.pedagogical_rating import run_stage6
        run_stage6(output_dir, cfg, diag=diag)
        timings["stage6_importance"] = time.time() - t0

    # Stage 7: Scoring & Aggregation
    if should_run(7):
        t0 = time.time()
        logger.info("--- Stage 7: Scoring & Aggregation ---")
        from .stages.scoring import run_stage7
        run_stage7(output_dir, cfg, diag=diag)
        timings["stage7_scoring"] = time.time() - t0

    # Dashboard: Generate after Stage 7 (or when running all stages)
    if should_run(7) or stage_only is None:
        try:
            t0 = time.time()
            logger.info("--- Generating Dashboard ---")
            from .utils.viz_reports import generate_dashboard
            dashboard_path = generate_dashboard(output_dir, cfg=cfg)
            timings["dashboard"] = time.time() - t0
            logger.info("Dashboard: %s", dashboard_path)
        except Exception as e:
            logger.warning("Dashboard generation failed: %s", e, exc_info=True)

    # Save timings
    timings["total"] = sum(timings.values())
    save_json(timings, os.path.join(output_dir, "timings.json"))
    logger.info("Done: %s (total: %.1fs)", video_name, timings["total"])
    return output_dir


def main():
    args = parse_args()
    cfg = build_config(args)

    logger.info("Pipeline v2.2 starting")
    logger.info("Config: ASR=%s, VLM=%s, SigLIP=%s, Keywords=%s, Importance=%s",
                cfg.ASR_BACKEND, cfg.VLM_MODE, cfg.SIGLIP_ENABLED,
                cfg.KEYWORD_ENABLED, cfg.IMPORTANCE_ENABLED)
    logger.info("[v2.2] TEMPORAL_SIGMA=%.1f, USE_KEYFRAME_AS_TVIS=%s, SCORING_MODE=%s, DIAGNOSTICS=%s",
                cfg.TEMPORAL_SIGMA, cfg.USE_KEYFRAME_AS_TVIS, cfg.SCORING_MODE, cfg.DIAGNOSTICS_ENABLED)

    for video_path in args.video:
        if not os.path.exists(video_path):
            logger.error("Video not found: %s", video_path)
            continue
        try:
            run_pipeline(video_path, cfg, stage_only=args.stage)
        except Exception as e:
            logger.error("Failed on %s: %s", video_path, e, exc_info=True)


if __name__ == "__main__":
    main()
