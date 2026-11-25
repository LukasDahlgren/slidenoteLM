from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "processing", "completed", "failed"]
StageStatus = Literal["pending", "running", "completed", "failed"]
PipelineStage = Literal[
    "upload",
    "audio_prep",
    "transcription",
    "ocr",
    "alignment",
    "notes",
]

PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    "upload",
    "audio_prep",
    "transcription",
    "ocr",
    "alignment",
    "notes",
)

BACKGROUND_STAGES: tuple[PipelineStage, ...] = (
    "audio_prep",
    "transcription",
    "ocr",
    "alignment",
    "notes",
)

STAGE_LABELS: dict[PipelineStage, str] = {
    "upload": "Upload",
    "audio_prep": "Audio Prep",
    "transcription": "Transcription (Groq)",
    "ocr": "Slide OCR (Gemini)",
    "alignment": "Alignment",
    "notes": "Markdown Notes",
}


class StageError(RuntimeError):
    """Structured pipeline failure carrying the failing stage and an actionable hint."""

    def __init__(self, stage: PipelineStage, message: str, hint: str) -> None:
        super().__init__(message)
        self.stage: PipelineStage = stage
        self.message = message
        self.hint = hint


@dataclass(frozen=True)
class JobContext:
    token: str
    slides_filename: str
    recording_filename: str
    job_dir: Path
    input_dir: Path
    artifacts_dir: Path
    slides_path: Path
    recording_path: Path
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    message: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "slidenoteLM"
    pipeline_llm_mode: str


class JobCreateResponse(BaseModel):
    token: str
    result_url: str
    status: JobStatus
    current_stage: PipelineStage
    created_at: str


class StageStateResponse(BaseModel):
    stage: PipelineStage
    label: str
    status: StageStatus
    started_at: str | None = None
    completed_at: str | None = None
    message: str | None = None


class JobErrorResponse(BaseModel):
    stage: PipelineStage
    stage_label: str
    message: str
    hint: str


class JobStatusResponse(BaseModel):
    token: str
    result_url: str
    status: JobStatus
    current_stage: PipelineStage
    current_stage_label: str
    progress_pct: int = Field(ge=0, le=100)
    slides_filename: str
    recording_filename: str
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    download_ready: bool
    download_url: str | None = None
    download_filename: str | None = None
    stages: list[StageStateResponse]
    error: JobErrorResponse | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
