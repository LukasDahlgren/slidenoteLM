from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .config import Settings
from .db import JobStore
from .pipeline import StageHandler, build_default_stage_handlers
from .schemas import (
    ARTIFACT_SLIDES_PDF,
    BACKGROUND_STAGES,
    STAGE_LABELS,
    JobContext,
    PipelineStage,
    StageError,
)

logger = logging.getLogger(__name__)


class JobWorker:
    """Runs lecture processing jobs asynchronously in a background thread pool."""

    def __init__(
        self,
        store: JobStore,
        settings: Settings,
        stage_handlers: dict[PipelineStage, StageHandler] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self._handlers: dict[PipelineStage, StageHandler] = build_default_stage_handlers()
        if stage_handlers:
            self._handlers.update(stage_handlers)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.worker_concurrency,
            thread_name_prefix="slidenotelm-worker",
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future[None]] = {}

    def recover_interrupted_jobs(self) -> int:
        count = self.store.mark_interrupted_jobs()
        if count:
            logger.warning("Marked %d interrupted job(s) as failed on startup.", count)
        return count

    def submit(self, token: str) -> Future[None]:
        future = self._executor.submit(self.run_job, token)
        with self._lock:
            self._futures[token] = future

        def _cleanup(_done: Future[None]) -> None:
            with self._lock:
                self._futures.pop(token, None)

        future.add_done_callback(_cleanup)
        return future

    def wait_for_job(self, token: str, timeout: float = 10.0) -> None:
        with self._lock:
            future = self._futures.get(token)
        if future is not None:
            future.result(timeout=timeout)

    def run_job(self, token: str) -> None:
        job = self.store.get_job(token)
        if job is None:
            logger.error("Cannot run unknown job token=%s", token)
            return

        job_dir = Path(job["job_dir"])
        input_dir = job_dir / "input"
        artifacts_dir = job_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, str] = dict(job.get("artifacts") or {})
        slides_rel = artifacts.get("slides_pdf", ARTIFACT_SLIDES_PDF)
        recording_rel = artifacts.get("recording", "")
        slides_path = job_dir / slides_rel
        recording_path = job_dir / recording_rel if recording_rel else input_dir / "recording"

        self.store.mark_job_started(token)

        for stage in BACKGROUND_STAGES:
            handler = self._handlers[stage]
            self.store.start_stage(token, stage)

            def _emit_progress(message: str, *, _stage: PipelineStage = stage) -> None:
                self.store.update_stage_message(token, _stage, message)

            context = JobContext(
                token=token,
                slides_filename=str(job["slides_filename"]),
                recording_filename=str(job["recording_filename"]),
                job_dir=job_dir,
                input_dir=input_dir,
                artifacts_dir=artifacts_dir,
                slides_path=slides_path,
                recording_path=recording_path,
                artifacts=dict(artifacts),
                progress_callback=_emit_progress,
            )
            try:
                result = handler(context, self.settings)
            except StageError as exc:
                logger.warning(
                    "Job %s failed at stage %s: %s",
                    token,
                    exc.stage,
                    exc.message,
                )
                self.store.mark_job_failed(
                    token,
                    stage=exc.stage,
                    message=exc.message,
                    hint=exc.hint,
                )
                return
            except Exception as exc:
                logger.exception("Unexpected error in job %s at stage %s", token, stage)
                stage_label = STAGE_LABELS[stage]
                self.store.mark_job_failed(
                    token,
                    stage=stage,
                    message=f"Unexpected error during {stage_label}: {exc}",
                    hint="Check backend server logs and retry the upload.",
                )
                return

            if result.artifacts:
                artifacts.update(result.artifacts)
            self.store.complete_stage(
                token,
                stage,
                message=result.message or f"{STAGE_LABELS[stage]} completed.",
                artifacts=result.artifacts or None,
                stats=result.stats or None,
                warnings=result.warnings or None,
            )

        self.store.mark_job_completed(token, artifacts=artifacts)

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)


__all__ = ["JobWorker", "StageHandler", "build_default_stage_handlers"]
