from __future__ import annotations

from typing import Callable

from ..config import Settings
from ..schemas import JobContext, PipelineStage, StageResult
from .align import run_alignment_stage
from .audio import run_audio_prep_stage
from .notes import run_notes_stage
from .ocr import run_ocr_stage
from .transcribe import run_transcription_stage

StageHandler = Callable[[JobContext, Settings], StageResult]


def build_default_stage_handlers() -> dict[PipelineStage, StageHandler]:
    """Return the active pipeline stage handlers for all background stages."""
    return {
        "audio_prep": run_audio_prep_stage,
        "transcription": run_transcription_stage,
        "ocr": run_ocr_stage,
        "alignment": run_alignment_stage,
        "notes": run_notes_stage,
    }
