"""Pipeline v2.1 — Config dataclass with all tunable parameters."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # === ASR ===
    ASR_BACKEND: str = "whisperx"
    WHISPER_MODEL: str = "medium"
    WHISPER_COMPUTE_TYPE: str = "float16"  # GPU
    WHISPER_BATCH_SIZE: int = 16  # GPU batch
    VALIDATE_TIMESTAMPS_MFA: bool = False

    # === Scene Detection ===
    SAMPLE_INTERVAL: float = 0.5
    SCENE_SIGNAL_W1: float = 0.35  # PySceneDetect weight
    SCENE_SIGNAL_W2: float = 0.45  # DINOv2 weight
    SCENE_SIGNAL_W3: float = 0.20  # OCR Jaccard weight
    SCENE_THRESHOLD_K: float = 1.5  # adaptive: mean + k*std
    MIN_SCENE_MERGE_DURATION: float = 2.0
    MERGE_SIMILARITY_THRESHOLD: float = 0.85

    # === OCR ===
    OCR_ENABLED: bool = True
    OCR_ENGINE: str = "paddleocr"  # paddleocr | easyocr
    OCR_MIN_CONFIDENCE: float = 0.3
    OCR_JACCARD_THRESHOLD: float = 0.5
    OCR_FUZZY_THRESHOLD: float = 0.8  # Levenshtein ratio

    # === DINOv2 ===
    DINOV2_MODEL: str = "facebook/dinov2-base"
    DINOV2_BATCH_SIZE: int = 16  # GPU batch

    # === Embeddings ===
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"

    # === SigLIP ===
    SIGLIP_ENABLED: bool = True
    SIGLIP_MODEL_NAME: str = "ViT-B-16-SigLIP"
    SIGLIP_PRETRAINED: str = "webli"
    SIGLIP_MIN_SIM: float = 0.20
    SIGLIP_ALPHA_LOW: float = 0.20
    SIGLIP_ALPHA_HIGH: float = 0.60

    # === Alignment ===
    TEMPORAL_SIGMA: float = 15.0
    TRACK_A_TEMPORAL_WINDOW: float = 30.0
    TRACK_A_CONTEXT_MIN_SIM: float = 0.25
    MIN_GLOBAL_SIM: float = 0.20
    ALPHA_SIM_LOW: float = 0.30
    ALPHA_SIM_HIGH: float = 0.80
    TRACK_C_MODE: str = "text_semantic"

    # === VLM ===
    VLM_MODE: str = "skip"  # ollama | gemini | skip (no ollama on HPC, gemini needs API key)
    OLLAMA_MODEL: str = "llava:7b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # === Gemini ===
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_CACHE_DIR: str = ".gemini_cache"
    GEMINI_TEMPERATURE: float = 0.0
    GEMINI_USE_STRUCTURED_OUTPUT: bool = True

    # === Keywords ===
    KEYWORD_ENABLED: bool = True
    KEYWORD_MIN_LENGTH: int = 3
    KEYWORD_USE_KEYBERT: bool = True
    KEYWORD_KEYBERT_TOP_N: int = 5

    # === Grounding ===
    GROUNDING_MODEL: str = "grounding_dino"
    GROUNDING_DINO_ENABLED: bool = False  # disabled by default on CPU
    GROUNDING_DINO_BOX_THRESHOLD: float = 0.25
    GROUNDING_DINO_TEXT_THRESHOLD: float = 0.25
    PERSISTENCE_CHECK_FRAMES: int = 3

    # === Groundability ===
    GROUNDABILITY_HIGH_THRESHOLD: float = 4.0
    GROUNDABILITY_MEDIUM_THRESHOLD: float = 2.5

    # === Pedagogical Importance ===
    IMPORTANCE_ENABLED: bool = True
    IMPORTANCE_BACKEND: str = "auto"  # gemini | local_llm | heuristic | auto
    LOCAL_LLM_MODEL: str = "llava:7b"
    IMPORTANCE_WEIGHTS: dict = field(default_factory=lambda: {
        1: 0.3, 2: 0.6, 3: 1.0, 4: 1.5, 5: 2.0
    })
    IMPORTANCE_DOUBLE_RUN: bool = True
    IMPORTANCE_DISAGREEMENT_THRESHOLD: int = 1

    # === Scoring ===
    SCORE_TAU: float = 2.5

    # === Paths ===
    OUTPUT_ROOT: str = "outputs_v2_1"

    # === Non-content frame labels ===
    CONTENT_LABELS: list = field(default_factory=lambda: [
        "content slide", "diagram", "code editor", "whiteboard",
        "animation frame", "demonstration"
    ])
    NON_CONTENT_LABELS: list = field(default_factory=lambda: [
        "title slide", "logo screen", "blank screen",
        "loading screen", "transition effect", "section divider",
        "talking head with no visual aids"
    ])
