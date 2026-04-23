"""Pipeline v2.1 — CLI entry point & stage orchestration."""

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
logger = logging.getLogger("pipeline_v2_1")


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline v2.1 — Temporal Contiguity Analysis"
    )
    # Input/Output
    p.add_argument("--video", nargs="+", required=True, help="Input video file(s)")
    p.add_argument("--output-root", default="outputs_v2_1", help="Root output dir")

    # ASR
    p.add_argument("--asr-backend", default="whisperx", choices=["whisperx", "faster-whisper"])
    p.add_argument("--whisper-model", default="medium")
    p.add_argument("--validate-timestamps-mfa", action="store_true")

    # Scene Detection
    p.add_argument("--sample-interval", type=float, default=0.5)

    # OCR
    p.add_argument("--ocr-engine", default="paddleocr", choices=["paddleocr", "easyocr"])
    p.add_argument("--no-ocr", action="store_true")

    # VLM
    p.add_argument("--vlm-mode", default="ollama", choices=["ollama", "gemini", "skip"])
    p.add_argument("--ollama-model", default="llava:7b")
    p.add_argument("--skip-vlm", action="store_true")

    # Gemini
    p.add_argument("--gemini-api-key", default="")
    p.add_argument("--gemini-model", default="gemini-2.0-flash")

    # Alignment
    p.add_argument("--no-siglip", action="store_true")
    p.add_argument("--track-c-mode", default="text_semantic")

    # Keywords
    p.add_argument("--no-keywords", action="store_true")
    p.add_argument("--no-grounding-dino", action="store_true", default=True)
    p.add_argument("--grounding-model", default="grounding_dino")

    # Importance
    p.add_argument("--no-importance", action="store_true")
    p.add_argument("--importance-backend", default="auto",
                   choices=["gemini", "local_llm", "heuristic", "auto"])

    # Scoring
    p.add_argument("--score-tau", type=float, default=2.5)

    return p.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    cfg.ASR_BACKEND = args.asr_backend
    cfg.WHISPER_MODEL = args.whisper_model
    cfg.VALIDATE_TIMESTAMPS_MFA = args.validate_timestamps_mfa
    cfg.SAMPLE_INTERVAL = args.sample_interval
    cfg.OCR_ENABLED = not args.no_ocr
    cfg.OCR_ENGINE = args.ocr_engine
    cfg.VLM_MODE = "skip" if args.skip_vlm else args.vlm_mode
    cfg.OLLAMA_MODEL = args.ollama_model
    cfg.GEMINI_API_KEY = args.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    cfg.GEMINI_MODEL = args.gemini_model
    cfg.SIGLIP_ENABLED = not args.no_siglip
    cfg.TRACK_C_MODE = args.track_c_mode
    cfg.KEYWORD_ENABLED = not args.no_keywords
    cfg.GROUNDING_DINO_ENABLED = not args.no_grounding_dino
    cfg.GROUNDING_MODEL = args.grounding_model
    cfg.IMPORTANCE_ENABLED = not args.no_importance
    cfg.IMPORTANCE_BACKEND = args.importance_backend
    cfg.SCORE_TAU = args.score_tau
    cfg.OUTPUT_ROOT = args.output_root
    return cfg


def run_pipeline(video_path: str, cfg: Config):
    """Run full pipeline on a single video."""
    from .utils.io_utils import get_output_dir

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = get_output_dir(cfg.OUTPUT_ROOT, video_path)
    logger.info("=" * 60)
    logger.info("Processing: %s", video_name)
    logger.info("Output: %s", output_dir)
    logger.info("=" * 60)

    timings = {}

    # Stage 1: ASR
    t0 = time.time()
    logger.info("--- Stage 1: ASR Transcription ---")
    from .stages.asr_whisper import run_stage1
    run_stage1(video_path, output_dir, cfg)
    timings["stage1_asr"] = time.time() - t0

    # Stage 2: Scene Detection
    t0 = time.time()
    logger.info("--- Stage 2: Scene Detection ---")
    from .stages.scene_detection import run_stage2
    run_stage2(video_path, output_dir, cfg)
    timings["stage2_scenes"] = time.time() - t0

    # Stage 3: Visual Concept Extraction
    t0 = time.time()
    logger.info("--- Stage 3: Visual Concepts ---")
    from .stages.vlm_concepts import run_stage3
    run_stage3(output_dir, cfg)
    timings["stage3_vlm"] = time.time() - t0

    # Stage 4a: Text Units & Embeddings
    t0 = time.time()
    logger.info("--- Stage 4a: Text Embeddings ---")
    from .stages.text_units import run_stage4a
    run_stage4a(output_dir, cfg)
    timings["stage4a_embeddings"] = time.time() - t0

    # Stage 4b: Narration Alignment (all 3 tracks enabled on GPU)
    t0 = time.time()
    logger.info("--- Stage 4b: Alignment (Track A+B+C) ---")
    from .stages.alignment import run_stage4b
    run_stage4b(output_dir, cfg)
    timings["stage4b_alignment"] = time.time() - t0

    # Stage 5: Keyword Extraction & Grounding
    if cfg.KEYWORD_ENABLED:
        t0 = time.time()
        logger.info("--- Stage 5: Keywords ---")
        from .stages.keyword_analysis import run_stage5
        run_stage5(output_dir, cfg)
        timings["stage5_keywords"] = time.time() - t0

    # Stage 6: Pedagogical Importance
    if cfg.IMPORTANCE_ENABLED:
        t0 = time.time()
        logger.info("--- Stage 6: Importance ---")
        from .stages.pedagogical_rating import run_stage6
        run_stage6(output_dir, cfg)
        timings["stage6_importance"] = time.time() - t0

    # Stage 7: Scoring
    t0 = time.time()
    logger.info("--- Stage 7: Scoring ---")
    from .stages.scoring import run_scoring
    run_scoring(output_dir, cfg)
    timings["stage7_scoring"] = time.time() - t0

    # Dashboard
    t0 = time.time()
    logger.info("--- Generating Dashboard ---")
    from .utils.viz_reports import generate_dashboard
    generate_dashboard(output_dir, video_name, cfg)
    timings["dashboard"] = time.time() - t0

    # Save timings
    from .utils.io_utils import save_json
    timings["total"] = sum(timings.values())
    save_json(timings, os.path.join(output_dir, "timings.json"))

    logger.info("Done: %s (total: %.1fs)", video_name, timings["total"])
    return output_dir


def main():
    args = parse_args()
    cfg = build_config(args)

    logger.info("Pipeline v2.1 starting")
    logger.info("Config: ASR=%s, VLM=%s, SigLIP=%s, Keywords=%s, Importance=%s",
                cfg.ASR_BACKEND, cfg.VLM_MODE, cfg.SIGLIP_ENABLED,
                cfg.KEYWORD_ENABLED, cfg.IMPORTANCE_ENABLED)

    for video_path in args.video:
        if not os.path.exists(video_path):
            logger.error("Video not found: %s", video_path)
            continue
        try:
            run_pipeline(video_path, cfg)
        except Exception as e:
            logger.error("Failed on %s: %s", video_path, e, exc_info=True)


if __name__ == "__main__":
    main()
