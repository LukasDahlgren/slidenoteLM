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


def _parse_int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        parsed = int(raw.strip())
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
    )
