"""Central configuration for the temporal contiguity scoring pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Scene detection (SSIM-based) ────────────────────────────────
    SCENE_THRESHOLDS: list[float] = field(default_factory=lambda: [0.85, 0.90, 0.95])
    MIN_SCENE_DURATION: float = 3.0   # seconds
    SAMPLE_INTERVAL: float = 0.333    # seconds between extracted frames (3 fps)

    # ── OCR settings ───────────────────────────────────────────────────
    OCR_ENABLED: bool = True
    OCR_MIN_CONFIDENCE: float = 0.3
    MIN_NEW_WORDS_FOR_SCENE: int = 1   # min new OCR words to trigger a scene

    # ── Alignment parameters ─────────────────────────────────────────
    OMEGA_SECONDS: float = 5.0       # half-window for narration search
    THETA_WINDOW: float = 0.35       # min cosine sim for window match
    THETA_GLOBAL: float = 0.30       # min cosine sim for global fallback
    MIN_GLOBAL_SIM: float = 0.20     # below this, no match is accepted
    MIN_WORD_MATCHES: int = 1        # min exact word matches for Track A
    TEMPORAL_SIGMA: float = 15.0     # Gaussian decay σ for temporal weighting
    TRACK_A_TEMPORAL_WINDOW: float = 30.0  # max seconds from t_vis for Track A word matching

    # ── Embedding model ──────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    USE_API_EMBEDDINGS: bool = False  # if True, use placeholder API function

    # ── VLM ──────────────────────────────────────────────────────────
    VLM_MODE: str = "ollama"          # "ollama", "offline_llava", or "api"
    OLLAMA_MODEL: str = "llava:7b"    # Ollama model name for vision tasks (7b fits in ~5GB RAM)
    OLLAMA_BASE_URL: str = "http://localhost:11434"  # Ollama server URL

    # ── Whisper ──────────────────────────────────────────────────────
    WHISPER_MODEL: str = "medium"

    # ── Paths ────────────────────────────────────────────────────────
    OUTPUT_ROOT: str = "outputs"

    # ── CLIP (visual alignment – Track B) ────────────────────────────
    CLIP_ENABLED: bool = True
    CLIP_MODEL_NAME: str = "ViT-B/16"
    CLIP_MIN_SIM: float = 0.20       # min CLIP similarity to accept alignment
    CLIP_ALPHA_LOW: float = 0.20     # CLIP sim maps to α=0.5
    CLIP_ALPHA_HIGH: float = 0.60    # CLIP sim maps to α=1.0

    # ── Alpha mapping (cosine sim → semantic weight) ─────────────────
    ALPHA_SIM_LOW: float = 0.30      # sim values <= this map to α=0
    ALPHA_SIM_HIGH: float = 0.80     # sim values >= this map to α=1


def config_from_args(args) -> Config:
    """Build a Config from parsed argparse namespace, overriding defaults."""
    cfg = Config()
    if getattr(args, "scene_thresholds", None):
        cfg.SCENE_THRESHOLDS = args.scene_thresholds
    if getattr(args, "sample_interval", None) is not None:
        cfg.SAMPLE_INTERVAL = args.sample_interval
    if getattr(args, "omega", None) is not None:
        cfg.OMEGA_SECONDS = args.omega
    if getattr(args, "output_root", None):
        cfg.OUTPUT_ROOT = args.output_root
    if getattr(args, "use_api_embeddings", False):
        cfg.USE_API_EMBEDDINGS = True
    if getattr(args, "vlm_mode", None):
        cfg.VLM_MODE = args.vlm_mode
    if getattr(args, "whisper_model", None):
        cfg.WHISPER_MODEL = args.whisper_model
    if getattr(args, "no_ocr", False):
        cfg.OCR_ENABLED = False
    if getattr(args, "no_clip", False):
        cfg.CLIP_ENABLED = False
    return cfg
