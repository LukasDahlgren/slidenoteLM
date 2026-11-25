from __future__ import annotations

import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .auth import (
    AuthLoginRequest,
    AuthStatusResponse,
    clear_session_cookie,
    create_session_token,
    is_request_authenticated,
    set_session_cookie,
    verify_password,
)
from .config import REPO_ROOT, Settings, load_settings
from .db import JobStore, generate_job_token
from .pipeline import StageHandler
from .schemas import (
    ARTIFACT_SLIDES_PDF,
    PIPELINE_STAGES,
    STAGE_LABELS,
    HealthResponse,
    JobCreateResponse,
    JobErrorResponse,
    JobPreviewResponse,
    JobStatsResponse,
    JobStatusResponse,
    PipelineStage,
    StageStateResponse,
)
from .worker import JobWorker

UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MiB
PDF_MAGIC_HEADER = b"%PDF-"
PDF_HEADER_SCAN_BYTES = 1024
PREVIEW_MAX_CHARS = 50_000
_SLUG_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def sanitize_filename(filename: str | None, fallback: str) -> str:
    if not filename:
        return fallback
    cleaned = Path(filename).name.strip()
    return cleaned or fallback


def build_download_filename(slides_filename: str) -> str:
    stem = Path(slides_filename).stem.lower()
    slug = _SLUG_NON_ALNUM_RE.sub("-", stem).strip("-")
    base = slug[:80].strip("-") or "lecture"
    return f"{base}-notes.md"


def _compute_progress_pct(job: dict) -> int:
    if job["status"] == "completed":
        return 100
    stages = job.get("stages") or []
    total = len(PIPELINE_STAGES)
    if total == 0:
        return 0
    score = 0.0
    for stage in stages:
        stage_status = stage.get("status")
        if stage_status == "completed":
            score += 1.0
        elif stage_status == "running":
            score += 0.5
    pct = int(round((score / total) * 100))
    return max(0, min(99, pct))


def _resolve_notes_path(job: dict) -> Path | None:
    artifacts = job.get("artifacts") or {}
    notes_rel = artifacts.get("notes_md")
    if not notes_rel:
        return None
    job_dir = Path(job["job_dir"]).resolve()
    candidate = (job_dir / notes_rel).resolve()
    try:
        candidate.relative_to(job_dir)
    except ValueError:
        return None
    return candidate


async def _save_upload_file(
    upload: UploadFile,
    destination: Path,
    *,
    max_bytes: int,
    label: str,
    require_pdf_header: bool = False,
) -> int:
    total_bytes = 0
    header_found = not require_pdf_header
    header_window = bytearray()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out_file:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break

            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"{label} exceeds the maximum allowed size of {max_bytes} bytes.",
                )

            if not header_found:
                remaining = PDF_HEADER_SCAN_BYTES - len(header_window)
                if remaining > 0:
                    header_window.extend(chunk[:remaining])
                    if PDF_MAGIC_HEADER in header_window:
                        header_found = True
                    elif len(header_window) >= PDF_HEADER_SCAN_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Uploaded slides file is not a valid PDF document.",
                        )

            out_file.write(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{label} is empty. Please upload a non-empty file.",
        )

    if require_pdf_header and not header_found:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded slides file is not a valid PDF document.",
        )

    return total_bytes


def _job_to_status_response(job: dict) -> JobStatusResponse:
    token = str(job["token"])
    current_stage: PipelineStage = job["current_stage"]
    stages = [
        StageStateResponse(
            stage=s["stage"],
            label=STAGE_LABELS[s["stage"]],
            status=s["status"],
            started_at=s["started_at"],
            completed_at=s["completed_at"],
            message=s["message"],
        )
        for s in job["stages"]
    ]

    error: JobErrorResponse | None = None
    if job.get("error_stage") and job.get("error_message"):
        err_stage: PipelineStage = job["error_stage"]
        error = JobErrorResponse(
            stage=err_stage,
            stage_label=STAGE_LABELS[err_stage],
            message=str(job["error_message"]),
            hint=str(job.get("error_hint") or "Retry the upload or check backend logs."),
        )

    notes_path = _resolve_notes_path(job)
    download_ready = (
        job["status"] == "completed"
        and notes_path is not None
        and notes_path.is_file()
    )
    download_filename = build_download_filename(str(job["slides_filename"]))

    return JobStatusResponse(
        token=token,
        result_url=f"/jobs/{token}",
        status=job["status"],
        current_stage=current_stage,
        current_stage_label=STAGE_LABELS[current_stage],
        progress_pct=_compute_progress_pct(job),
        slides_filename=str(job["slides_filename"]),
        recording_filename=str(job["recording_filename"]),
        created_at=str(job["created_at"]),
        updated_at=str(job["updated_at"]),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        download_ready=download_ready,
        download_url=f"/api/jobs/{token}/download" if download_ready else None,
        download_filename=download_filename if download_ready else None,
        stages=stages,
        error=error,
        artifacts=dict(job.get("artifacts") or {}),
        stats=JobStatsResponse.model_validate(job.get("stats") or {}),
        warnings=list(job.get("warnings") or []),
    )


def _require_completed_notes_path(store: JobStore, token: str) -> tuple[dict, Path]:
    job = store.get_job(token)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found. Check that your private result link is complete.",
        )

    if job["status"] != "completed":
        if job["status"] == "failed":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Processing failed before Markdown notes were generated.",
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Markdown notes are not ready yet. Wait for processing to complete.",
        )

    notes_path = _resolve_notes_path(job)
    if notes_path is None or not notes_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Generated notes file could not be found on disk.",
        )

    return job, notes_path


class AppPasswordMiddleware:
    """Enforce optional APP_PASSWORD cookie authentication on /api/jobs* routes."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self.settings.app_password:
            path = scope.get("path", "")
            method = scope.get("method", "GET").upper()
            if method != "OPTIONS" and (path == "/api/jobs" or path.startswith("/api/jobs/")):
                request = Request(scope, receive=receive)
                if not is_request_authenticated(request, self.settings):
                    response = JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Authentication required."},
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


def _has_unsafe_path_syntax(raw_path: str) -> bool:
    if "\x00" in raw_path or "\\" in raw_path:
        return True
    return ".." in PurePosixPath(raw_path).parts


def _resolve_static_file(dist_dir: Path, raw_path: str) -> Path | None:
    normalized = raw_path.lstrip("/")
    if not normalized or _has_unsafe_path_syntax(normalized):
        return None
    candidate = (dist_dir / normalized).resolve()
    try:
        candidate.relative_to(dist_dir)
    except ValueError:
        return None
    if candidate.is_file():
        return candidate
    return None


def create_app(
    *,
    settings: Settings | None = None,
    stage_handlers: dict[PipelineStage, StageHandler] | None = None,
    frontend_dist_dir: Path | None = None,
) -> FastAPI:
    resolved_settings = settings or load_settings()
    resolved_dist_dir = (
        frontend_dist_dir
        if frontend_dist_dir is not None
        else (REPO_ROOT / "frontend" / "dist")
    ).resolve()
    store = JobStore(resolved_settings.db_path)
    worker = JobWorker(store, resolved_settings, stage_handlers=stage_handlers)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        resolved_settings.ensure_directories()
        store.init_db()
        worker.recover_interrupted_jobs()
        try:
            yield
        finally:
            worker.shutdown(wait=False)

    application = FastAPI(
        title="slidenoteLM API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.store = store
    application.state.worker = worker
    application.state.frontend_dist_dir = resolved_dist_dir

    application.add_middleware(AppPasswordMiddleware, settings=resolved_settings)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.frontend_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(pipeline_llm_mode=resolved_settings.pipeline_llm_mode)

    @application.get("/api/auth/status", response_model=AuthStatusResponse)
    def auth_status(request: Request) -> AuthStatusResponse:
        auth_required = bool(resolved_settings.app_password)
        authenticated = (
            True
            if not auth_required
            else is_request_authenticated(request, resolved_settings)
        )
        return AuthStatusResponse(
            auth_required=auth_required,
            authenticated=authenticated,
        )

    @application.post("/api/auth/login", response_model=AuthStatusResponse)
    def auth_login(
        payload: AuthLoginRequest,
        request: Request,
        response: Response,
    ) -> AuthStatusResponse:
        if not resolved_settings.app_password:
            return AuthStatusResponse(auth_required=False, authenticated=True)
        if not verify_password(payload.password, resolved_settings.app_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid password.",
            )
        token = create_session_token(
            resolved_settings.app_password,
            resolved_settings.app_session_secret,
        )
        set_session_cookie(response, request, token)
        return AuthStatusResponse(auth_required=True, authenticated=True)

    @application.post("/api/auth/logout", response_model=AuthStatusResponse)
    def auth_logout(request: Request, response: Response) -> AuthStatusResponse:
        clear_session_cookie(response, request)
        auth_required = bool(resolved_settings.app_password)
        return AuthStatusResponse(
            auth_required=auth_required,
            authenticated=not auth_required,
        )

    @application.post(
        "/api/jobs",
        response_model=JobCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_job(
        slides: UploadFile = File(...),
        recording: UploadFile = File(...),
    ) -> JobCreateResponse:
        slides_name = sanitize_filename(slides.filename, "slides.pdf")
        recording_name = sanitize_filename(recording.filename, "recording")

        slides_ext = Path(slides_name).suffix.lower()
        if slides_ext != ".pdf":
            await slides.close()
            await recording.close()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Slides upload must be a .pdf file.",
            )

        recording_ext = Path(recording_name).suffix.lower()
        if recording_ext not in resolved_settings.allowed_recording_extensions:
            await slides.close()
            await recording.close()
            allowed_list = ", ".join(resolved_settings.allowed_recording_extensions)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported recording format '{recording_ext or 'none'}'. "
                    f"Allowed formats: {allowed_list}"
                ),
            )

        token = generate_job_token()
        job_dir = resolved_settings.jobs_dir / token
        slides_rel = ARTIFACT_SLIDES_PDF
        recording_rel = f"input/recording{recording_ext}"

        try:
            (job_dir / "input").mkdir(parents=True, exist_ok=True)
            (job_dir / "artifacts").mkdir(parents=True, exist_ok=True)

            await _save_upload_file(
                slides,
                job_dir / slides_rel,
                max_bytes=resolved_settings.max_pdf_bytes,
                label="Slides PDF",
                require_pdf_header=True,
            )
            await _save_upload_file(
                recording,
                job_dir / recording_rel,
                max_bytes=resolved_settings.max_recording_bytes,
                label="Recording file",
                require_pdf_header=False,
            )
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        finally:
            await slides.close()
            await recording.close()

        job = store.create_job(
            slides_filename=slides_name,
            recording_filename=recording_name,
            job_dir=job_dir,
            artifacts={
                "slides_pdf": slides_rel,
                "recording": recording_rel,
            },
            token=token,
        )
        worker.submit(token)

        return JobCreateResponse(
            token=token,
            result_url=f"/jobs/{token}",
            status=job["status"],
            current_stage=job["current_stage"],
            created_at=str(job["created_at"]),
        )

    @application.get("/api/jobs/{token}", response_model=JobStatusResponse)
    def get_job_status(token: str) -> JobStatusResponse:
        job = store.get_job(token)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found. Check that your private result link is complete.",
            )
        return _job_to_status_response(job)

    @application.get("/api/jobs/{token}/preview", response_model=JobPreviewResponse)
    def preview_notes(token: str) -> JobPreviewResponse:
        job, notes_path = _require_completed_notes_path(store, token)
        full_text = notes_path.read_text(encoding="utf-8")
        truncated = len(full_text) > PREVIEW_MAX_CHARS
        content = full_text[:PREVIEW_MAX_CHARS] if truncated else full_text
        return JobPreviewResponse(
            token=token,
            filename=build_download_filename(str(job["slides_filename"])),
            content=content,
            word_count=len(full_text.split()),
            truncated=truncated,
        )

    @application.get("/api/jobs/{token}/download")
    def download_notes(token: str) -> FileResponse:
        job, notes_path = _require_completed_notes_path(store, token)
        return FileResponse(
            path=notes_path,
            media_type="text/markdown; charset=utf-8",
            filename=build_download_filename(str(job["slides_filename"])),
        )

    index_html = resolved_dist_dir / "index.html"
    if index_html.is_file():

        @application.get("/{full_path:path}", include_in_schema=False)
        def serve_spa(full_path: str) -> FileResponse:
            normalized = full_path.lstrip("/")
            if normalized == "api" or normalized.startswith("api/"):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Not Found",
                )

            if _has_unsafe_path_syntax(normalized):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Not Found",
                )

            if normalized:
                raw_candidate = resolved_dist_dir / normalized
                resolved_candidate = raw_candidate.resolve()
                try:
                    resolved_candidate.relative_to(resolved_dist_dir)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Not Found",
                    ) from exc

                static_file = _resolve_static_file(resolved_dist_dir, normalized)
                if static_file is not None:
                    headers = (
                        {"Cache-Control": "no-cache"}
                        if static_file == index_html
                        else None
                    )
                    return FileResponse(path=static_file, headers=headers)

                if normalized == "assets" or normalized.startswith("assets/"):
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Not Found",
                    )

            return FileResponse(path=index_html, headers={"Cache-Control": "no-cache"})

    return application


app = create_app()
