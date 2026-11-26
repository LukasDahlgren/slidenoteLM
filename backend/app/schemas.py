from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "processing", "completed", "failed", "cancelled"]
StageStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
PipelineRole = Literal["primary", "preparing"]
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

# Canonical job-relative artifact paths shared across pipeline stages.
ARTIFACT_SLIDES_PDF = "input/slides.pdf"
ARTIFACT_AUDIO_NORMALIZED = "artifacts/audio_normalized.mp3"
ARTIFACT_TRANSCRIPT = "artifacts/transcript.json"
ARTIFACT_SLIDES_OCR = "artifacts/slides_ocr.json"
ARTIFACT_ALIGNMENT = "artifacts/alignment.json"
ARTIFACT_NOTES_MD = "artifacts/notes.md"

ProgressCallback = Callable[[str], None]


class StageError(RuntimeError):
    """Structured pipeline failure carrying the failing stage and an actionable hint."""

    def __init__(
        self,
        stage: PipelineStage,
        message: str,
        hint: str,
        *,
        fatal_for_queue: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage: PipelineStage = stage
        self.message = message
        self.hint = hint
        self.fatal_for_queue = fatal_for_queue


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
    progress_callback: ProgressCallback | None = field(default=None, repr=False, compare=False)
    cancel_event: threading.Event | None = field(default=None, repr=False, compare=False)
    progress_fraction_callback: Callable[[float], None] | None = field(default=None, repr=False, compare=False)

    @property
    def audio_normalized_path(self) -> Path:
        return self.job_dir / ARTIFACT_AUDIO_NORMALIZED

    @property
    def transcript_path(self) -> Path:
        return self.job_dir / ARTIFACT_TRANSCRIPT

    @property
    def slides_ocr_path(self) -> Path:
        return self.job_dir / ARTIFACT_SLIDES_OCR

    @property
    def alignment_path(self) -> Path:
        return self.job_dir / ARTIFACT_ALIGNMENT

    @property
    def notes_path(self) -> Path:
        return self.job_dir / ARTIFACT_NOTES_MD

    def is_cancelled(self) -> bool:
        """Return True if cooperative cancellation has been requested for this job."""
        return self.cancel_event is not None and self.cancel_event.is_set()

    def report_progress(self, message: str) -> None:
        """Emit a live status update for the currently executing stage."""
        if self.progress_callback is not None and message.strip():
            self.progress_callback(message.strip())


@dataclass(frozen=True)
class StageResult:
    message: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --- Persisted Pipeline Artifact Contracts (Phases 2-4) ---


class TranscriptSegment(BaseModel):
    index: int = Field(ge=0)
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str


class TranscriptArtifact(BaseModel):
    source_recording: str
    model: str = ""
    duration_seconds: float | None = Field(default=None, ge=0.0)
    segments: list[TranscriptSegment] = Field(default_factory=list)


class SlideOCRRecord(BaseModel):
    slide: int = Field(ge=1)
    visible_text: str = ""
    visual_context: str = ""
    tables_code_formulas: str = ""
    raw_text: str = ""


class SlidesOCRArtifact(BaseModel):
    source_pdf: str
    model: str = ""
    slide_count: int = Field(default=0, ge=0)
    slides: list[SlideOCRRecord] = Field(default_factory=list)


class AlignmentEntry(BaseModel):
    slide: int | None = Field(default=None, ge=1)
    start_segment: int = Field(ge=0)
    end_segment: int = Field(ge=0)
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    uncertain: bool = False
    reason: str = ""


class AlignmentIssue(BaseModel):
    code: str
    message: str
    slides: list[int] = Field(default_factory=list)
    severity: Literal["warning", "error"] = "warning"


class AlignmentArtifact(BaseModel):
    mode: str = "stub"
    entries: list[AlignmentEntry] = Field(default_factory=list)
    alignment_issues: list[AlignmentIssue] = Field(default_factory=list)


# --- HTTP API Response Models ---


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
    queue_position: int | None = None
    batch_id: str | None = None


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


class JobStatsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    duration_seconds: float | None = None
    slide_count: int | None = None
    transcript_segments: int | None = None
    word_count: int | None = None


class JobStatusResponse(BaseModel):
    token: str
    result_url: str
    status: JobStatus
    current_stage: PipelineStage
    current_stage_label: str
    progress_pct: int = Field(ge=0, le=100)
    slides_filename: str
    recording_filename: str
    queue_label: str | None = None
    batch_id: str | None = None
    queue_position: int | None = None
    queue_index: int | None = None
    pipeline_role: PipelineRole | None = None
    waiting_reason: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    download_ready: bool
    download_url: str | None = None
    download_filename: str | None = None
    can_cancel: bool = False
    can_retry: bool = False
    can_retry_from_scratch: bool = False
    stages: list[StageStateResponse]
    error: JobErrorResponse | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    stats: JobStatsResponse = Field(default_factory=JobStatsResponse)
    warnings: list[str] = Field(default_factory=list)


class QueueItemResponse(BaseModel):
    token: str
    result_url: str
    status: JobStatus
    current_stage: PipelineStage
    current_stage_label: str
    progress_pct: int = Field(ge=0, le=100)
    slides_filename: str
    recording_filename: str
    queue_label: str | None = None
    batch_id: str | None = None
    queue_position: int | None = None
    queue_index: int | None = None
    pipeline_role: PipelineRole | None = None
    waiting_reason: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    download_ready: bool
    download_url: str | None = None
    download_filename: str | None = None
    can_cancel: bool = False
    can_retry: bool = False
    can_retry_from_scratch: bool = False
    error: JobErrorResponse | None = None
    stats: JobStatsResponse = Field(default_factory=JobStatsResponse)


class QueueCountsResponse(BaseModel):
    processing: int = 0
    queued: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0


class QueueSnapshotResponse(BaseModel):
    paused: bool
    pause_reason: str | None = None
    max_in_flight_jobs: int = 2
    max_queue_depth: int = 50
    counts: QueueCountsResponse = Field(default_factory=QueueCountsResponse)
    active: list[JobStatusResponse] = Field(default_factory=list)
    queued: list[QueueItemResponse] = Field(default_factory=list)
    recent: list[QueueItemResponse] = Field(default_factory=list)


class QueueOrderRequest(BaseModel):
    tokens: list[str] = Field(default_factory=list)


class JobCancelRequest(BaseModel):
    delete_files: bool | None = None


class JobRetryRequest(BaseModel):
    from_scratch: bool = False
    position: Literal["end", "front"] = "end"


class JobPreviewResponse(BaseModel):
    token: str
    filename: str
    content: str
    word_count: int = Field(ge=0)
    truncated: bool = False
