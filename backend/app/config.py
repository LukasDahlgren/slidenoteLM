from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_RECORDING_EXTENSIONS: tuple[str, ...] = (
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".opus",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
)

DEFAULT_FRONTEND_ORIGIN = "http://localhost:5173"
DEFAULT_MAX_PDF_BYTES = 100 * 1024 * 1024  # 100 MB
DEFAULT_MAX_RECORDING_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
DEFAULT_WORKER_CONCURRENCY = 2

# Multi-lecture pipelined queue defaults
DEFAULT_QUEUE_MAX_IN_FLIGHT_JOBS = 4
DEFAULT_QUEUE_LANE_AUDIO_PREP_CONCURRENCY = 2
DEFAULT_QUEUE_LANE_TRANSCRIPTION_CONCURRENCY = 2
DEFAULT_QUEUE_LANE_OCR_CONCURRENCY = 3
DEFAULT_QUEUE_LANE_LLM_CONCURRENCY = 3
DEFAULT_QUEUE_OCR_OVERLAP_WITH_LLM = "full"
VALID_QUEUE_OCR_OVERLAP_MODES: frozenset[str] = frozenset({"paced", "strict", "full"})
DEFAULT_QUEUE_LOOKAHEAD_OCR_WORKERS = 2
DEFAULT_GEMINI_COMBINED_RATE_LIMIT_RPM = 120
DEFAULT_QUEUE_AUTO_PAUSE_ON_AUTH_ERROR = True
DEFAULT_CLEANUP_RAW_RECORDING_AFTER_AUDIO_PREP = True
DEFAULT_MAX_QUEUE_DEPTH = 50
DEFAULT_MIN_FREE_DISK_BYTES = 512 * 1024 * 1024  # 512 MiB

# Audio normalization & Groq Whisper transcription defaults
DEFAULT_TRANSCRIBE_TARGET_BITRATE = "32k"
DEFAULT_TRANSCRIBE_MAX_UPLOAD_BYTES = 24_000_000  # Safely below Groq's 25 MB limit
DEFAULT_TRANSCRIBE_CHUNK_HEADROOM_PCT = 90
DEFAULT_TRANSCRIBE_MIN_CHUNK_SECONDS = 300
DEFAULT_TRANSCRIBE_PARALLEL_WORKERS = 4
DEFAULT_TRANSCRIBE_GLOBAL_MAX_CONCURRENT = 6
DEFAULT_TRANSCRIBE_RATE_LIMIT_RPM = 20
DEFAULT_TRANSCRIBE_RETRY_ATTEMPTS = 3
DEFAULT_TRANSCRIBE_RETRY_BASE_DELAY_SECONDS = 3.0
DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS = 180.0

# Google Gemini Slide OCR defaults
DEFAULT_GEMINI_MODEL = "gemini-4-neon"
DEFAULT_OCR_MODEL_GOOGLE = DEFAULT_GEMINI_MODEL
DEFAULT_OCR_THINKING_LEVEL = "MEDIUM"
DEFAULT_OCR_MAX_PAGES_PER_BATCH = 8
DEFAULT_OCR_MAX_BATCH_BYTES = 18 * 1024 * 1024  # 18 MiB (safely below Gemini's 20 MB inline limit)
DEFAULT_OCR_PARALLEL_WORKERS = 8
DEFAULT_OCR_PARALLEL_MIN_PAGES = 4
DEFAULT_OCR_GLOBAL_MAX_CONCURRENT = 16
DEFAULT_OCR_RATE_LIMIT_RPM = 60
DEFAULT_OCR_RETRY_ATTEMPTS = 3
DEFAULT_OCR_RETRY_BASE_DELAY_SECONDS = 2.0
DEFAULT_OCR_TIMEOUT_SECONDS = 180.0
DEFAULT_OCR_MAX_OUTPUT_TOKENS = 32768

# Google Gemini LLM defaults (Phase 6: Alignment, Section Notes, Coverage Audit)
DEFAULT_PIPELINE_LLM_MODE = "gemini"
DEFAULT_LLM_ALIGN_MODEL = DEFAULT_GEMINI_MODEL
DEFAULT_LLM_ALIGN_THINKING_LEVEL = "HIGH"
DEFAULT_LLM_ALIGN_MAX_OUTPUT_TOKENS = 32768
DEFAULT_LLM_NOTES_MODEL = DEFAULT_GEMINI_MODEL
DEFAULT_LLM_NOTES_THINKING_LEVEL = "HIGH"
DEFAULT_LLM_NOTES_MAX_OUTPUT_TOKENS = 49152
DEFAULT_LLM_COVERAGE_MODEL = DEFAULT_GEMINI_MODEL
DEFAULT_LLM_COVERAGE_THINKING_LEVEL = "MEDIUM"
DEFAULT_LLM_COVERAGE_MAX_OUTPUT_TOKENS = 16384
DEFAULT_LLM_GLOBAL_MAX_CONCURRENT = 24
DEFAULT_LLM_RATE_LIMIT_RPM = 90
DEFAULT_LLM_RETRY_ATTEMPTS = 4
DEFAULT_LLM_RETRY_BASE_DELAY_SECONDS = 2.0
DEFAULT_LLM_NOTES_PARALLEL_WORKERS = 9
DEFAULT_LLM_COVERAGE_PARALLEL_WORKERS = 4
DEFAULT_NOTES_MAX_SLIDES_PER_SECTION = 8
DEFAULT_NOTES_MAX_SECTION_DURATION_SECONDS = 900.0
DEFAULT_NOTES_MAX_SECTION_TRANSCRIPT_CHARS = 14000
DEFAULT_NOTES_INLINE_OFF_SLIDE_MAX_SECONDS = 35.0
DEFAULT_NOTES_INLINE_OFF_SLIDE_MAX_CHARS = 300
DEFAULT_NOTES_MIN_AUDIT_TRANSCRIPT_CHARS = 200
DEFAULT_LLM_TIMEOUT_SECONDS = 300.0

VALID_THINKING_LEVELS: frozenset[str] = frozenset(
    {"MINIMAL", "LOW", "MEDIUM", "HIGH", "XHIGH"}
)
THINKING_LEVEL_OFF_ALIASES: frozenset[str] = frozenset({"OFF", "NONE"})
THINKING_LEVEL_DEFAULT_ALIASES: frozenset[str] = frozenset({"DEFAULT", "UNSPECIFIED"})

# HTTP API rate-limiting defaults (sliding window per client IP; 0 disables a bucket)
DEFAULT_RATE_LIMIT_ENABLED = True
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_RATE_LIMIT_TRUST_PROXY_HEADERS = True
DEFAULT_RATE_LIMIT_AUTH_LOGIN_PER_WINDOW = 10
DEFAULT_RATE_LIMIT_AUTH_PER_WINDOW = 60
DEFAULT_RATE_LIMIT_JOBS_CREATE_PER_WINDOW = 30
DEFAULT_RATE_LIMIT_JOBS_READ_PER_WINDOW = 120
DEFAULT_RATE_LIMIT_QUEUE_CONTROL_PER_WINDOW = 60
DEFAULT_RATE_LIMIT_HEALTH_PER_WINDOW = 120
DEFAULT_RATE_LIMIT_DEFAULT_PER_WINDOW = 120


def _parse_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        parsed = int(raw.strip())
    except ValueError:
        return default
    return max(minimum, parsed)


def _parse_float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        parsed = float(raw.strip())
    except ValueError:
        return default
    return max(minimum, parsed)


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_ocr_overlap_mode_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in VALID_QUEUE_OCR_OVERLAP_MODES:
        return normalized
    logger.warning(
        "Ignoring invalid %s=%r (expected one of %s); using %r.",
        name,
        raw,
        ", ".join(sorted(VALID_QUEUE_OCR_OVERLAP_MODES)),
        default,
    )
    return default


def _parse_thinking_level_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().upper()
    if normalized in THINKING_LEVEL_OFF_ALIASES:
        # Gemini 4 cannot disable thinking completely; map OFF/NONE to the lowest tier (MINIMAL).
        return "MINIMAL"
    if normalized in THINKING_LEVEL_DEFAULT_ALIASES:
        return ""
    if normalized in VALID_THINKING_LEVELS:
        return normalized
    logger.warning(
        "Ignoring invalid %s=%r (expected one of %s, OFF/NONE, or DEFAULT); using %r.",
        name,
        raw,
        ", ".join(sorted(VALID_THINKING_LEVELS)),
        default,
    )
    return default


def _parse_frontend_origins(raw_value: str | None) -> tuple[str, ...]:
    raw = (raw_value or "").strip() or DEFAULT_FRONTEND_ORIGIN
    origins: list[str] = []
    for part in raw.split(","):
        origin = part.strip().rstrip("/")
        if origin and origin not in origins:
            origins.append(origin)

    if not origins:
        origins.append(DEFAULT_FRONTEND_ORIGIN)

    # Mirror localhost <-> 127.0.0.1 for local dev so browsers on either hostname work.
    expanded = list(origins)
    for origin in origins:
        if "://localhost" in origin:
            mirror = origin.replace("://localhost", "://127.0.0.1", 1)
            if mirror not in expanded:
                expanded.append(mirror)
        elif "://127.0.0.1" in origin:
            mirror = origin.replace("://127.0.0.1", "://localhost", 1)
            if mirror not in expanded:
                expanded.append(mirror)

    return tuple(expanded)


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    jobs_dir: Path
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    frontend_origin: str = DEFAULT_FRONTEND_ORIGIN
    frontend_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )
    pipeline_llm_mode: str = DEFAULT_PIPELINE_LLM_MODE
    groq_api_key: str = ""
    transcription_model: str = "whisper-large-v3-turbo"
    gemini_api_key: str = ""
    ocr_model_google: str = DEFAULT_OCR_MODEL_GOOGLE
    ocr_thinking_level: str = DEFAULT_OCR_THINKING_LEVEL
    llm_align_model: str = DEFAULT_LLM_ALIGN_MODEL
    llm_align_thinking_level: str = DEFAULT_LLM_ALIGN_THINKING_LEVEL
    llm_align_max_output_tokens: int = DEFAULT_LLM_ALIGN_MAX_OUTPUT_TOKENS
    llm_notes_model: str = DEFAULT_LLM_NOTES_MODEL
    llm_notes_thinking_level: str = DEFAULT_LLM_NOTES_THINKING_LEVEL
    llm_notes_max_output_tokens: int = DEFAULT_LLM_NOTES_MAX_OUTPUT_TOKENS
    llm_coverage_model: str = DEFAULT_LLM_COVERAGE_MODEL
    llm_coverage_thinking_level: str = DEFAULT_LLM_COVERAGE_THINKING_LEVEL
    llm_coverage_max_output_tokens: int = DEFAULT_LLM_COVERAGE_MAX_OUTPUT_TOKENS
    llm_global_max_concurrent: int = DEFAULT_LLM_GLOBAL_MAX_CONCURRENT
    llm_rate_limit_rpm: int = DEFAULT_LLM_RATE_LIMIT_RPM
    llm_retry_attempts: int = DEFAULT_LLM_RETRY_ATTEMPTS
    llm_retry_base_delay_seconds: float = DEFAULT_LLM_RETRY_BASE_DELAY_SECONDS
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    google_genai_use_vertexai: bool = False
    google_cloud_project: str = ""
    google_cloud_location: str = "us-central1"
    max_pdf_bytes: int = DEFAULT_MAX_PDF_BYTES
    max_recording_bytes: int = DEFAULT_MAX_RECORDING_BYTES
    worker_concurrency: int = DEFAULT_WORKER_CONCURRENCY
    allowed_recording_extensions: tuple[str, ...] = ALLOWED_RECORDING_EXTENSIONS
    transcribe_target_bitrate: str = DEFAULT_TRANSCRIBE_TARGET_BITRATE
    transcribe_max_upload_bytes: int = DEFAULT_TRANSCRIBE_MAX_UPLOAD_BYTES
    transcribe_chunk_headroom_pct: int = DEFAULT_TRANSCRIBE_CHUNK_HEADROOM_PCT
    transcribe_min_chunk_seconds: int = DEFAULT_TRANSCRIBE_MIN_CHUNK_SECONDS
    transcribe_parallel_workers: int = DEFAULT_TRANSCRIBE_PARALLEL_WORKERS
    transcribe_global_max_concurrent: int = DEFAULT_TRANSCRIBE_GLOBAL_MAX_CONCURRENT
    transcribe_rate_limit_rpm: int = DEFAULT_TRANSCRIBE_RATE_LIMIT_RPM
    transcribe_retry_attempts: int = DEFAULT_TRANSCRIBE_RETRY_ATTEMPTS
    transcribe_retry_base_delay_seconds: float = DEFAULT_TRANSCRIBE_RETRY_BASE_DELAY_SECONDS
    transcribe_timeout_seconds: float = DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS
    ocr_max_pages_per_batch: int = DEFAULT_OCR_MAX_PAGES_PER_BATCH
    ocr_max_batch_bytes: int = DEFAULT_OCR_MAX_BATCH_BYTES
    ocr_parallel_workers: int = DEFAULT_OCR_PARALLEL_WORKERS
    ocr_parallel_min_pages: int = DEFAULT_OCR_PARALLEL_MIN_PAGES
    ocr_global_max_concurrent: int = DEFAULT_OCR_GLOBAL_MAX_CONCURRENT
    ocr_rate_limit_rpm: int = DEFAULT_OCR_RATE_LIMIT_RPM
    ocr_retry_attempts: int = DEFAULT_OCR_RETRY_ATTEMPTS
    ocr_retry_base_delay_seconds: float = DEFAULT_OCR_RETRY_BASE_DELAY_SECONDS
    ocr_timeout_seconds: float = DEFAULT_OCR_TIMEOUT_SECONDS
    ocr_max_output_tokens: int = DEFAULT_OCR_MAX_OUTPUT_TOKENS
    ffmpeg_path: str = ""
    ffprobe_path: str = ""
    app_password: str = ""
    app_session_secret: str = ""
    rate_limit_enabled: bool = DEFAULT_RATE_LIMIT_ENABLED
    rate_limit_window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS
    rate_limit_trust_proxy_headers: bool = DEFAULT_RATE_LIMIT_TRUST_PROXY_HEADERS
    rate_limit_auth_login_per_window: int = DEFAULT_RATE_LIMIT_AUTH_LOGIN_PER_WINDOW
    rate_limit_auth_per_window: int = DEFAULT_RATE_LIMIT_AUTH_PER_WINDOW
    rate_limit_jobs_create_per_window: int = DEFAULT_RATE_LIMIT_JOBS_CREATE_PER_WINDOW
    rate_limit_jobs_read_per_window: int = DEFAULT_RATE_LIMIT_JOBS_READ_PER_WINDOW
    rate_limit_queue_control_per_window: int = DEFAULT_RATE_LIMIT_QUEUE_CONTROL_PER_WINDOW
    rate_limit_health_per_window: int = DEFAULT_RATE_LIMIT_HEALTH_PER_WINDOW
    rate_limit_default_per_window: int = DEFAULT_RATE_LIMIT_DEFAULT_PER_WINDOW

    queue_max_in_flight_jobs: int = DEFAULT_QUEUE_MAX_IN_FLIGHT_JOBS
    queue_lane_audio_prep_concurrency: int = DEFAULT_QUEUE_LANE_AUDIO_PREP_CONCURRENCY
    queue_lane_transcription_concurrency: int = DEFAULT_QUEUE_LANE_TRANSCRIPTION_CONCURRENCY
    queue_lane_ocr_concurrency: int = DEFAULT_QUEUE_LANE_OCR_CONCURRENCY
    queue_lane_llm_concurrency: int = DEFAULT_QUEUE_LANE_LLM_CONCURRENCY
    queue_ocr_overlap_with_llm: str = DEFAULT_QUEUE_OCR_OVERLAP_WITH_LLM
    queue_lookahead_ocr_workers: int = DEFAULT_QUEUE_LOOKAHEAD_OCR_WORKERS
    gemini_combined_rate_limit_rpm: int = DEFAULT_GEMINI_COMBINED_RATE_LIMIT_RPM
    queue_auto_pause_on_auth_error: bool = DEFAULT_QUEUE_AUTO_PAUSE_ON_AUTH_ERROR
    cleanup_raw_recording_after_audio_prep: bool = DEFAULT_CLEANUP_RAW_RECORDING_AFTER_AUDIO_PREP
    max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH
    min_free_disk_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES

    llm_notes_parallel_workers: int = DEFAULT_LLM_NOTES_PARALLEL_WORKERS
    llm_coverage_parallel_workers: int = DEFAULT_LLM_COVERAGE_PARALLEL_WORKERS
    notes_max_slides_per_section: int = DEFAULT_NOTES_MAX_SLIDES_PER_SECTION
    notes_max_section_duration_seconds: float = DEFAULT_NOTES_MAX_SECTION_DURATION_SECONDS
    notes_max_section_transcript_chars: int = DEFAULT_NOTES_MAX_SECTION_TRANSCRIPT_CHARS
    notes_inline_off_slide_max_seconds: float = DEFAULT_NOTES_INLINE_OFF_SLIDE_MAX_SECONDS
    notes_inline_off_slide_max_chars: int = DEFAULT_NOTES_INLINE_OFF_SLIDE_MAX_CHARS
    notes_min_audit_transcript_chars: int = DEFAULT_NOTES_MIN_AUDIT_TRANSCRIPT_CHARS

    @property
    def transcribe_safe_upload_bytes(self) -> int:
        headroom_pct = max(10, min(99, self.transcribe_chunk_headroom_pct))
        return max(1, int(self.transcribe_max_upload_bytes * (headroom_pct / 100.0)))

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def load_settings(
    *,
    env_file: Path | None = None,
    base_dir: Path = REPO_ROOT,
) -> Settings:
    dotenv_path = env_file if env_file is not None else (base_dir / ".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    data_dir = _resolve_path(os.getenv("DATA_DIR", "./data"), base_dir)
    raw_db_path = os.getenv("SQLITE_DB_PATH", "").strip()
    db_path = _resolve_path(raw_db_path, base_dir) if raw_db_path else (data_dir / "slidenotelm.db")
    jobs_dir = data_dir / "jobs"
    frontend_origins = _parse_frontend_origins(os.getenv("FRONTEND_ORIGIN"))

    return Settings(
        data_dir=data_dir,
        db_path=db_path,
        jobs_dir=jobs_dir,
        backend_host=os.getenv("BACKEND_HOST", "127.0.0.1").strip() or "127.0.0.1",
        backend_port=_parse_int_env("BACKEND_PORT", 8000, minimum=1),
        frontend_origin=frontend_origins[0],
        frontend_origins=frontend_origins,
        pipeline_llm_mode=(
            os.getenv("PIPELINE_LLM_MODE", DEFAULT_PIPELINE_LLM_MODE).strip().lower()
            or DEFAULT_PIPELINE_LLM_MODE
        ),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        transcription_model=os.getenv("TRANSCRIPTION_MODEL", "whisper-large-v3-turbo").strip()
        or "whisper-large-v3-turbo",
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        ocr_model_google=os.getenv("OCR_MODEL_GOOGLE", DEFAULT_OCR_MODEL_GOOGLE).strip()
        or DEFAULT_OCR_MODEL_GOOGLE,
        ocr_thinking_level=_parse_thinking_level_env(
            "OCR_THINKING_LEVEL", DEFAULT_OCR_THINKING_LEVEL
        ),
        llm_align_model=os.getenv("LLM_ALIGN_MODEL", DEFAULT_LLM_ALIGN_MODEL).strip()
        or DEFAULT_LLM_ALIGN_MODEL,
        llm_align_thinking_level=_parse_thinking_level_env(
            "LLM_ALIGN_THINKING_LEVEL", DEFAULT_LLM_ALIGN_THINKING_LEVEL
        ),
        llm_align_max_output_tokens=_parse_int_env(
            "LLM_ALIGN_MAX_OUTPUT_TOKENS",
            DEFAULT_LLM_ALIGN_MAX_OUTPUT_TOKENS,
            minimum=1024,
        ),
        llm_notes_model=os.getenv("LLM_NOTES_MODEL", DEFAULT_LLM_NOTES_MODEL).strip()
        or DEFAULT_LLM_NOTES_MODEL,
        llm_notes_thinking_level=_parse_thinking_level_env(
            "LLM_NOTES_THINKING_LEVEL", DEFAULT_LLM_NOTES_THINKING_LEVEL
        ),
        llm_notes_max_output_tokens=_parse_int_env(
            "LLM_NOTES_MAX_OUTPUT_TOKENS",
            DEFAULT_LLM_NOTES_MAX_OUTPUT_TOKENS,
            minimum=1024,
        ),
        llm_coverage_model=os.getenv("LLM_COVERAGE_MODEL", DEFAULT_LLM_COVERAGE_MODEL).strip()
        or DEFAULT_LLM_COVERAGE_MODEL,
        llm_coverage_thinking_level=_parse_thinking_level_env(
            "LLM_COVERAGE_THINKING_LEVEL", DEFAULT_LLM_COVERAGE_THINKING_LEVEL
        ),
        llm_coverage_max_output_tokens=_parse_int_env(
            "LLM_COVERAGE_MAX_OUTPUT_TOKENS",
            DEFAULT_LLM_COVERAGE_MAX_OUTPUT_TOKENS,
            minimum=1024,
        ),
        llm_global_max_concurrent=_parse_int_env(
            "LLM_GLOBAL_MAX_CONCURRENT",
            DEFAULT_LLM_GLOBAL_MAX_CONCURRENT,
            minimum=1,
        ),
        llm_rate_limit_rpm=_parse_int_env(
            "LLM_RATE_LIMIT_RPM",
            DEFAULT_LLM_RATE_LIMIT_RPM,
            minimum=0,
        ),
        llm_retry_attempts=_parse_int_env(
            "LLM_RETRY_ATTEMPTS",
            DEFAULT_LLM_RETRY_ATTEMPTS,
            minimum=1,
        ),
        llm_retry_base_delay_seconds=_parse_float_env(
            "LLM_RETRY_BASE_DELAY_SECONDS",
            DEFAULT_LLM_RETRY_BASE_DELAY_SECONDS,
            minimum=0.0,
        ),
        llm_notes_parallel_workers=_parse_int_env(
            "LLM_NOTES_PARALLEL_WORKERS", DEFAULT_LLM_NOTES_PARALLEL_WORKERS, minimum=1
        ),
        llm_coverage_parallel_workers=_parse_int_env(
            "LLM_COVERAGE_PARALLEL_WORKERS", DEFAULT_LLM_COVERAGE_PARALLEL_WORKERS, minimum=1
        ),
        notes_max_slides_per_section=_parse_int_env(
            "NOTES_MAX_SLIDES_PER_SECTION", DEFAULT_NOTES_MAX_SLIDES_PER_SECTION, minimum=1
        ),
        notes_max_section_duration_seconds=_parse_float_env(
            "NOTES_MAX_SECTION_DURATION_SECONDS", DEFAULT_NOTES_MAX_SECTION_DURATION_SECONDS, minimum=30.0
        ),
        notes_max_section_transcript_chars=_parse_int_env(
            "NOTES_MAX_SECTION_TRANSCRIPT_CHARS", DEFAULT_NOTES_MAX_SECTION_TRANSCRIPT_CHARS, minimum=500
        ),
        notes_inline_off_slide_max_seconds=_parse_float_env(
            "NOTES_INLINE_OFF_SLIDE_MAX_SECONDS", DEFAULT_NOTES_INLINE_OFF_SLIDE_MAX_SECONDS, minimum=0.0
        ),
        notes_inline_off_slide_max_chars=_parse_int_env(
            "NOTES_INLINE_OFF_SLIDE_MAX_CHARS", DEFAULT_NOTES_INLINE_OFF_SLIDE_MAX_CHARS, minimum=0
        ),
        notes_min_audit_transcript_chars=_parse_int_env(
            "NOTES_MIN_AUDIT_TRANSCRIPT_CHARS", DEFAULT_NOTES_MIN_AUDIT_TRANSCRIPT_CHARS, minimum=0
        ),
        llm_timeout_seconds=_parse_float_env(
            "LLM_TIMEOUT_SECONDS",
            DEFAULT_LLM_TIMEOUT_SECONDS,
            minimum=5.0,
        ),
        google_genai_use_vertexai=_parse_bool_env("GOOGLE_GENAI_USE_VERTEXAI", False),
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT", "").strip(),
        google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
        or "us-central1",
        max_pdf_bytes=_parse_int_env("MAX_PDF_BYTES", DEFAULT_MAX_PDF_BYTES, minimum=1024),
        max_recording_bytes=_parse_int_env(
            "MAX_RECORDING_BYTES", DEFAULT_MAX_RECORDING_BYTES, minimum=1024
        ),
        worker_concurrency=_parse_int_env(
            "WORKER_CONCURRENCY", DEFAULT_WORKER_CONCURRENCY, minimum=1
        ),
        transcribe_target_bitrate=os.getenv(
            "TRANSCRIBE_TARGET_BITRATE", DEFAULT_TRANSCRIBE_TARGET_BITRATE
        ).strip()
        or DEFAULT_TRANSCRIBE_TARGET_BITRATE,
        transcribe_max_upload_bytes=_parse_int_env(
            "TRANSCRIBE_MAX_UPLOAD_BYTES",
            DEFAULT_TRANSCRIBE_MAX_UPLOAD_BYTES,
            minimum=1024,
        ),
        transcribe_chunk_headroom_pct=min(
            99,
            _parse_int_env(
                "TRANSCRIBE_CHUNK_HEADROOM_PCT",
                DEFAULT_TRANSCRIBE_CHUNK_HEADROOM_PCT,
                minimum=10,
            ),
        ),
        transcribe_min_chunk_seconds=_parse_int_env(
            "TRANSCRIBE_MIN_CHUNK_SECONDS",
            DEFAULT_TRANSCRIBE_MIN_CHUNK_SECONDS,
            minimum=1,
        ),
        transcribe_parallel_workers=_parse_int_env(
            "TRANSCRIBE_PARALLEL_WORKERS",
            DEFAULT_TRANSCRIBE_PARALLEL_WORKERS,
            minimum=1,
        ),
        transcribe_global_max_concurrent=_parse_int_env(
            "TRANSCRIBE_GLOBAL_MAX_CONCURRENT",
            DEFAULT_TRANSCRIBE_GLOBAL_MAX_CONCURRENT,
            minimum=1,
        ),
        transcribe_rate_limit_rpm=_parse_int_env(
            "TRANSCRIBE_RATE_LIMIT_RPM",
            DEFAULT_TRANSCRIBE_RATE_LIMIT_RPM,
            minimum=0,
        ),
        transcribe_retry_attempts=_parse_int_env(
            "TRANSCRIBE_RETRY_ATTEMPTS",
            DEFAULT_TRANSCRIBE_RETRY_ATTEMPTS,
            minimum=1,
        ),
        transcribe_retry_base_delay_seconds=_parse_float_env(
            "TRANSCRIBE_RETRY_BASE_DELAY_SECONDS",
            DEFAULT_TRANSCRIBE_RETRY_BASE_DELAY_SECONDS,
            minimum=0.0,
        ),
        transcribe_timeout_seconds=_parse_float_env(
            "TRANSCRIBE_TIMEOUT_SECONDS",
            DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS,
            minimum=5.0,
        ),
        ocr_max_pages_per_batch=_parse_int_env(
            "OCR_MAX_PAGES_PER_BATCH",
            DEFAULT_OCR_MAX_PAGES_PER_BATCH,
            minimum=1,
        ),
        ocr_max_batch_bytes=_parse_int_env(
            "OCR_MAX_BATCH_BYTES",
            DEFAULT_OCR_MAX_BATCH_BYTES,
            minimum=1024,
        ),
        ocr_parallel_workers=_parse_int_env(
            "OCR_PARALLEL_WORKERS",
            DEFAULT_OCR_PARALLEL_WORKERS,
            minimum=1,
        ),
        ocr_parallel_min_pages=_parse_int_env(
            "OCR_PARALLEL_MIN_PAGES",
            DEFAULT_OCR_PARALLEL_MIN_PAGES,
            minimum=2,
        ),
        ocr_global_max_concurrent=_parse_int_env(
            "OCR_GLOBAL_MAX_CONCURRENT",
            DEFAULT_OCR_GLOBAL_MAX_CONCURRENT,
            minimum=1,
        ),
        ocr_rate_limit_rpm=_parse_int_env(
            "OCR_RATE_LIMIT_RPM",
            DEFAULT_OCR_RATE_LIMIT_RPM,
            minimum=0,
        ),
        ocr_retry_attempts=_parse_int_env(
            "OCR_RETRY_ATTEMPTS",
            DEFAULT_OCR_RETRY_ATTEMPTS,
            minimum=1,
        ),
        ocr_retry_base_delay_seconds=_parse_float_env(
            "OCR_RETRY_BASE_DELAY_SECONDS",
            DEFAULT_OCR_RETRY_BASE_DELAY_SECONDS,
            minimum=0.0,
        ),
        ocr_timeout_seconds=_parse_float_env(
            "OCR_TIMEOUT_SECONDS",
            DEFAULT_OCR_TIMEOUT_SECONDS,
            minimum=5.0,
        ),
        ocr_max_output_tokens=_parse_int_env(
            "OCR_MAX_OUTPUT_TOKENS",
            DEFAULT_OCR_MAX_OUTPUT_TOKENS,
            minimum=1024,
        ),
        ffmpeg_path=os.getenv("FFMPEG_PATH", "").strip(),
        ffprobe_path=os.getenv("FFPROBE_PATH", "").strip(),
        app_password=os.getenv("APP_PASSWORD", "").strip(),
        app_session_secret=os.getenv("APP_SESSION_SECRET", "").strip(),
        rate_limit_enabled=_parse_bool_env(
            "RATE_LIMIT_ENABLED",
            DEFAULT_RATE_LIMIT_ENABLED,
        ),
        rate_limit_window_seconds=_parse_int_env(
            "RATE_LIMIT_WINDOW_SECONDS",
            DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
            minimum=1,
        ),
        rate_limit_trust_proxy_headers=_parse_bool_env(
            "RATE_LIMIT_TRUST_PROXY_HEADERS",
            DEFAULT_RATE_LIMIT_TRUST_PROXY_HEADERS,
        ),
        rate_limit_auth_login_per_window=_parse_int_env(
            "RATE_LIMIT_AUTH_LOGIN_PER_MINUTE",
            DEFAULT_RATE_LIMIT_AUTH_LOGIN_PER_WINDOW,
            minimum=0,
        ),
        rate_limit_auth_per_window=_parse_int_env(
            "RATE_LIMIT_AUTH_PER_MINUTE",
            DEFAULT_RATE_LIMIT_AUTH_PER_WINDOW,
            minimum=0,
        ),
        rate_limit_jobs_create_per_window=_parse_int_env(
            "RATE_LIMIT_JOBS_CREATE_PER_MINUTE",
            DEFAULT_RATE_LIMIT_JOBS_CREATE_PER_WINDOW,
            minimum=0,
        ),
        rate_limit_jobs_read_per_window=_parse_int_env(
            "RATE_LIMIT_JOBS_READ_PER_MINUTE",
            DEFAULT_RATE_LIMIT_JOBS_READ_PER_WINDOW,
            minimum=0,
        ),
        rate_limit_queue_control_per_window=_parse_int_env(
            "RATE_LIMIT_QUEUE_CONTROL_PER_MINUTE",
            DEFAULT_RATE_LIMIT_QUEUE_CONTROL_PER_WINDOW,
            minimum=0,
        ),
        rate_limit_health_per_window=_parse_int_env(
            "RATE_LIMIT_HEALTH_PER_MINUTE",
            DEFAULT_RATE_LIMIT_HEALTH_PER_WINDOW,
            minimum=0,
        ),
        rate_limit_default_per_window=_parse_int_env(
            "RATE_LIMIT_DEFAULT_PER_MINUTE",
            DEFAULT_RATE_LIMIT_DEFAULT_PER_WINDOW,
            minimum=0,
        ),
        queue_max_in_flight_jobs=_parse_int_env(
            "QUEUE_MAX_IN_FLIGHT_JOBS",
            DEFAULT_QUEUE_MAX_IN_FLIGHT_JOBS,
            minimum=1,
        ),
        queue_lane_audio_prep_concurrency=_parse_int_env(
            "QUEUE_LANE_AUDIO_PREP_CONCURRENCY",
            DEFAULT_QUEUE_LANE_AUDIO_PREP_CONCURRENCY,
            minimum=1,
        ),
        queue_lane_transcription_concurrency=_parse_int_env(
            "QUEUE_LANE_TRANSCRIPTION_CONCURRENCY",
            DEFAULT_QUEUE_LANE_TRANSCRIPTION_CONCURRENCY,
            minimum=1,
        ),
        queue_lane_ocr_concurrency=_parse_int_env(
            "QUEUE_LANE_OCR_CONCURRENCY",
            DEFAULT_QUEUE_LANE_OCR_CONCURRENCY,
            minimum=1,
        ),
        queue_lane_llm_concurrency=_parse_int_env(
            "QUEUE_LANE_LLM_CONCURRENCY",
            DEFAULT_QUEUE_LANE_LLM_CONCURRENCY,
            minimum=1,
        ),
        queue_ocr_overlap_with_llm=_parse_ocr_overlap_mode_env(
            "QUEUE_OCR_OVERLAP_WITH_LLM",
            DEFAULT_QUEUE_OCR_OVERLAP_WITH_LLM,
        ),
        queue_lookahead_ocr_workers=_parse_int_env(
            "QUEUE_LOOKAHEAD_OCR_WORKERS",
            DEFAULT_QUEUE_LOOKAHEAD_OCR_WORKERS,
            minimum=1,
        ),
        gemini_combined_rate_limit_rpm=_parse_int_env(
            "GEMINI_COMBINED_RATE_LIMIT_RPM",
            DEFAULT_GEMINI_COMBINED_RATE_LIMIT_RPM,
            minimum=0,
        ),
        queue_auto_pause_on_auth_error=_parse_bool_env(
            "QUEUE_AUTO_PAUSE_ON_AUTH_ERROR",
            DEFAULT_QUEUE_AUTO_PAUSE_ON_AUTH_ERROR,
        ),
        cleanup_raw_recording_after_audio_prep=_parse_bool_env(
            "CLEANUP_RAW_RECORDING_AFTER_AUDIO_PREP",
            DEFAULT_CLEANUP_RAW_RECORDING_AFTER_AUDIO_PREP,
        ),
        max_queue_depth=_parse_int_env(
            "MAX_QUEUE_DEPTH",
            DEFAULT_MAX_QUEUE_DEPTH,
            minimum=0,
        ),
        min_free_disk_bytes=_parse_int_env(
            "MIN_FREE_DISK_BYTES",
            DEFAULT_MIN_FREE_DISK_BYTES,
            minimum=0,
        ),
    )
