from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

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

# Audio normalization & Groq Whisper transcription defaults
DEFAULT_TRANSCRIBE_TARGET_BITRATE = "32k"
DEFAULT_TRANSCRIBE_MAX_UPLOAD_BYTES = 24_000_000  # Safely below Groq's 25 MB limit
DEFAULT_TRANSCRIBE_CHUNK_HEADROOM_PCT = 90
DEFAULT_TRANSCRIBE_MIN_CHUNK_SECONDS = 300
DEFAULT_TRANSCRIBE_PARALLEL_WORKERS = 4
DEFAULT_TRANSCRIBE_GLOBAL_MAX_CONCURRENT = 6
DEFAULT_TRANSCRIBE_RETRY_ATTEMPTS = 3
DEFAULT_TRANSCRIBE_RETRY_BASE_DELAY_SECONDS = 3.0
DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS = 180.0

# Google Gemini Slide OCR defaults
DEFAULT_OCR_MAX_PAGES_PER_BATCH = 20
DEFAULT_OCR_MAX_BATCH_BYTES = 18 * 1024 * 1024  # 18 MiB (safely below Gemini's 20 MB inline limit)
DEFAULT_OCR_PARALLEL_WORKERS = 2
DEFAULT_OCR_PARALLEL_MIN_PAGES = 4
DEFAULT_OCR_GLOBAL_MAX_CONCURRENT = 4
DEFAULT_OCR_RETRY_ATTEMPTS = 3
DEFAULT_OCR_RETRY_BASE_DELAY_SECONDS = 2.0
DEFAULT_OCR_TIMEOUT_SECONDS = 180.0
DEFAULT_OCR_MAX_OUTPUT_TOKENS = 32768


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
    pipeline_llm_mode: str = "stub"
    groq_api_key: str = ""
    transcription_model: str = "whisper-large-v3-turbo"
    gemini_api_key: str = ""
    ocr_model_google: str = "gemini-2.5-flash"
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
    transcribe_retry_attempts: int = DEFAULT_TRANSCRIBE_RETRY_ATTEMPTS
    transcribe_retry_base_delay_seconds: float = DEFAULT_TRANSCRIBE_RETRY_BASE_DELAY_SECONDS
    transcribe_timeout_seconds: float = DEFAULT_TRANSCRIBE_TIMEOUT_SECONDS
    ocr_max_pages_per_batch: int = DEFAULT_OCR_MAX_PAGES_PER_BATCH
    ocr_max_batch_bytes: int = DEFAULT_OCR_MAX_BATCH_BYTES
    ocr_parallel_workers: int = DEFAULT_OCR_PARALLEL_WORKERS
    ocr_parallel_min_pages: int = DEFAULT_OCR_PARALLEL_MIN_PAGES
    ocr_global_max_concurrent: int = DEFAULT_OCR_GLOBAL_MAX_CONCURRENT
    ocr_retry_attempts: int = DEFAULT_OCR_RETRY_ATTEMPTS
    ocr_retry_base_delay_seconds: float = DEFAULT_OCR_RETRY_BASE_DELAY_SECONDS
    ocr_timeout_seconds: float = DEFAULT_OCR_TIMEOUT_SECONDS
    ocr_max_output_tokens: int = DEFAULT_OCR_MAX_OUTPUT_TOKENS
    ffmpeg_path: str = ""
    ffprobe_path: str = ""
    app_password: str = ""
    app_session_secret: str = ""

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
        pipeline_llm_mode=(os.getenv("PIPELINE_LLM_MODE", "stub").strip().lower() or "stub"),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        transcription_model=os.getenv("TRANSCRIPTION_MODEL", "whisper-large-v3-turbo").strip()
        or "whisper-large-v3-turbo",
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        ocr_model_google=os.getenv("OCR_MODEL_GOOGLE", "gemini-2.5-flash").strip()
        or "gemini-2.5-flash",
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
    )
