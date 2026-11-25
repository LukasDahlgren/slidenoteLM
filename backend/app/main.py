from __future__ import annotations

import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import Settings, load_settings
from .db import JobStore, generate_job_token
from .schemas import (
    PIPELINE_STAGES,
    STAGE_LABELS,
    HealthResponse,
    JobCreateResponse,
    JobErrorResponse,
    JobStatusResponse,
    PipelineStage,
    StageStateResponse,
)
from .worker import JobWorker, StageHandler

UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MiB
PDF_MAGIC_HEADER = b"%PDF-"
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
    header_checked = not require_pdf_header
    header_buffer = bytearray()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out_file:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break

            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"{label} exceeds the maximum allowed size of {max_bytes} bytes.",
                )

            if not header_checked:
                header_buffer.extend(chunk[: len(PDF_MAGIC_HEADER) - len(header_buffer)])
                if len(header_buffer) >= len(PDF_MAGIC_HEADER):
                    if bytes(header_buffer) != PDF_MAGIC_HEADER:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Uploaded slides file is not a valid PDF document.",
                        )
                    header_checked = True

            out_file.write(chunk)

    if total_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{label} is empty. Please upload a non-empty file.",
        )

    if require_pdf_header and not header_checked:
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
    )


def create_app(
    *,
    settings: Settings | None = None,
    stage_handlers: dict[PipelineStage, StageHandler] | None = None,
) -> FastAPI:
    resolved_settings = settings or load_settings()
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

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved_settings.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(pipeline_llm_mode=resolved_settings.pipeline_llm_mode)

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
        slides_rel = "input/slides.pdf"
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

    @application.get("/api/jobs/{token}/download")
    def download_notes(token: str) -> FileResponse:
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

        return FileResponse(
            path=notes_path,
            media_type="text/markdown; charset=utf-8",
            filename=build_download_filename(str(job["slides_filename"])),
        )

    return application


app = create_app()
