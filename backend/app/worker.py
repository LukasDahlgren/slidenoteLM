from __future__ import annotations

import dataclasses
import logging
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .db import JobStore
from .pipeline import StageHandler, build_default_stage_handlers
from .schemas import (
    ARTIFACT_AUDIO_NORMALIZED,
    ARTIFACT_SLIDES_PDF,
    BACKGROUND_STAGES,
    STAGE_LABELS,
    JobContext,
    PipelineRole,
    PipelineStage,
    StageError,
    StageResult,
)

logger = logging.getLogger(__name__)

STAGE_LANE: dict[PipelineStage, str] = {
    "audio_prep": "audio_prep",
    "transcription": "transcription",
    "ocr": "ocr",
    "alignment": "llm",
    "notes": "llm",
}

STAGE_DEPS: dict[PipelineStage, frozenset[PipelineStage]] = {
    "audio_prep": frozenset(),
    "ocr": frozenset(),
    "transcription": frozenset({"audio_prep"}),
    "alignment": frozenset({"transcription", "ocr"}),
    "notes": frozenset({"alignment"}),
}

STAGE_DISPATCH_ORDER: tuple[PipelineStage, ...] = (
    "audio_prep",
    "ocr",
    "transcription",
    "alignment",
    "notes",
)

ALL_LANES: tuple[str, ...] = ("audio_prep", "transcription", "ocr", "llm")
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


@dataclass
class _ActiveJobState:
    token: str
    queue_position: int
    slides_filename: str
    recording_filename: str
    job_dir: Path
    input_dir: Path
    artifacts_dir: Path
    slides_path: Path
    recording_path: Path
    artifacts: dict[str, str]
    completed: set[PipelineStage]
    running_stages: set[PipelineStage] = field(default_factory=set)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    failed: bool = False
    waiting_reason: str | None = None

    @property
    def audio_normalized_path(self) -> Path:
        return self.job_dir / ARTIFACT_AUDIO_NORMALIZED


class JobWorker:
    """SQLite-backed 4-lane multi-slot stage-pipelined coordinator for lecture queues.

    Runs up to `settings.queue_max_in_flight_jobs` lectures concurrently across 4
    independent resource lanes with per-lane concurrency limits:
      - `audio_prep` (local ffmpeg CPU/disk)
      - `transcription` (Groq Whisper)
      - `ocr` (Google Gemini OCR, rate-limit coordinated with `llm`)
      - `llm` (Google Gemini alignment -> notes)
    """

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

        self._lane_capacity: dict[str, int] = {
            "audio_prep": max(1, int(getattr(settings, "queue_lane_audio_prep_concurrency", 2))),
            "transcription": max(
                1, int(getattr(settings, "queue_lane_transcription_concurrency", 2))
            ),
            "ocr": max(1, int(getattr(settings, "queue_lane_ocr_concurrency", 3))),
            "llm": max(1, int(getattr(settings, "queue_lane_llm_concurrency", 3))),
        }

        self._cond = threading.Condition(threading.RLock())
        self._stopping = False
        self._started = False
        self._coordinator_thread: threading.Thread | None = None
        self._stage_pool = ThreadPoolExecutor(
            max_workers=max(len(ALL_LANES), sum(self._lane_capacity.values())),
            thread_name_prefix="slidenotelm-lane",
        )
        self._in_flight: dict[str, _ActiveJobState] = {}
        self._lane_active: dict[str, dict[str, PipelineStage]] = {
            lane: {} for lane in ALL_LANES
        }
        self._done_events: dict[str, threading.Event] = {}

    def start(self) -> None:
        with self._cond:
            if self._started or self._stopping:
                return
            self._started = True
            self._coordinator_thread = threading.Thread(
                target=self._coordinator_loop,
                name="slidenotelm-queue-coordinator",
                daemon=True,
            )
            self._coordinator_thread.start()

    def recover_interrupted_jobs(self) -> int:
        count = self.store.mark_interrupted_jobs()
        if count:
            logger.warning("Marked %d interrupted job(s) as failed on startup.", count)
        self.start()
        self.notify()
        return count

    def notify(self) -> None:
        with self._cond:
            if not self._started and not self._stopping:
                self.start()
            self._cond.notify_all()

    def submit(self, token: str) -> None:
        """Notify the coordinator that a persisted queued job is ready to be scheduled."""
        with self._cond:
            self._done_events[token] = threading.Event()
            if not self._started and not self._stopping:
                self.start()
            self._cond.notify_all()

    def wait_for_job(self, token: str, timeout: float = 10.0) -> None:
        with self._cond:
            in_flight = token in self._in_flight
            event = self._done_events.get(token)
            if event is None:
                event = threading.Event()
                self._done_events[token] = event
        if not in_flight:
            job = self.store.get_job(token)
            if job is None or job["status"] in TERMINAL_STATUSES:
                event.set()
                return
        if not event.wait(timeout=timeout):
            raise TimeoutError(f"Timed out waiting for job {token} after {timeout:.1f}s.")

    def run_job(self, token: str, *, timeout: float = 300.0) -> None:
        """Synchronously run a persisted job to completion (backward-compatible helper)."""
        self.submit(token)
        self.wait_for_job(token, timeout=timeout)

    def wait_for_idle(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                has_active = bool(self._in_flight)
                has_runnable_queued = (
                    not self.store.is_queue_paused()
                    and bool(self.store.queued_job_tokens())
                )
                if not has_active and not has_runnable_queued:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for queue idle after {timeout:.1f}s.")
                self._cond.wait(timeout=min(0.1, remaining))

    def _primary_tokens_locked(self, ordered: list[_ActiveJobState]) -> set[str]:
        llm_slots = max(1, self._lane_capacity["llm"])
        return {js.token for js in ordered[:llm_slots]}

    def get_runtime_job_meta(self, token: str) -> tuple[PipelineRole | None, str | None]:
        """Return (pipeline_role, waiting_reason) for an in-flight job."""
        with self._cond:
            if token not in self._in_flight:
                return None, None
            ordered = self._ordered_in_flight_locked()
            primaries = self._primary_tokens_locked(ordered)
            role: PipelineRole = "primary" if token in primaries else "preparing"
            return role, self._in_flight[token].waiting_reason

    def get_all_runtime_meta(self) -> dict[str, tuple[PipelineRole, str | None]]:
        """Return runtime metadata for all currently in-flight jobs."""
        with self._cond:
            ordered = self._ordered_in_flight_locked()
            if not ordered:
                return {}
            primaries = self._primary_tokens_locked(ordered)
            return {
                js.token: (
                    "primary" if js.token in primaries else "preparing",
                    js.waiting_reason,
                )
                for js in ordered
            }

    def cancel_job(
        self,
        token: str,
        *,
        delete_files: bool | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Cancel a queued or active job.

        Returns a status tag ('cancelled', 'cancelling', 'not_found', 'not_cancellable')
        plus the latest job dict (when found).
        """
        job = self.store.get_job(token)
        if job is None:
            return "not_found", None

        # Fast path: atomically cancel if still queued.
        if self.store.cancel_queued_job(token):
            should_delete = True if delete_files is None else bool(delete_files)
            if should_delete:
                shutil.rmtree(Path(str(job["job_dir"])), ignore_errors=True)
            with self._cond:
                self._signal_done_locked(token)
                self._cond.notify_all()
            return "cancelled", self.store.get_job(token)

        # Otherwise check if active in flight.
        with self._cond:
            js = self._in_flight.get(token)
            if js is not None:
                js.cancel_event.set()
                if not js.running_stages:
                    self.store.mark_job_cancelled(token)
                    self._in_flight.pop(token, None)
                    self._signal_done_locked(token)
                    self._cond.notify_all()
                    return "cancelled", self.store.get_job(token)
                self._cond.notify_all()
                return "cancelling", self.store.get_job(token)

        reloaded = self.store.get_job(token)
        if reloaded is None:
            return "not_found", None
        if reloaded["status"] == "cancelled":
            return "cancelled", reloaded
        if reloaded["status"] == "processing":
            if self.store.mark_job_cancelled(token):
                with self._cond:
                    self._signal_done_locked(token)
                    self._cond.notify_all()
                return "cancelled", self.store.get_job(token)
        return "not_cancellable", reloaded

    def shutdown(self, *, wait: bool = False) -> None:
        with self._cond:
            self._stopping = True
            if not wait:
                for js in self._in_flight.values():
                    js.cancel_event.set()
            self._cond.notify_all()
            coord = self._coordinator_thread

        if wait and coord is not None and coord.is_alive():
            coord.join(timeout=5.0)
        self._stage_pool.shutdown(wait=wait, cancel_futures=False)

    # --- Internal Coordinator Logic ---

    def _ordered_in_flight_locked(self) -> list[_ActiveJobState]:
        return sorted(self._in_flight.values(), key=lambda js: (js.queue_position, js.token))

    def _lane_has_capacity_locked(self, lane: str) -> bool:
        return len(self._lane_active[lane]) < self._lane_capacity[lane]

    def _other_job_holds_llm_locked(self, token: str | None) -> bool:
        return any(owner != token for owner in self._lane_active["llm"])

    def _signal_done_locked(self, token: str) -> None:
        event = self._done_events.get(token)
        if event is not None:
            event.set()

    def _can_admit_lookahead_locked(self, paused: bool) -> bool:
        if paused:
            return False
        max_in_flight = max(1, int(getattr(self.settings, "queue_max_in_flight_jobs", 4)))
        if len(self._in_flight) >= max_in_flight:
            return False
        if not self._in_flight:
            return True

        peek = self.store.peek_next_queued_job()
        if peek is None:
            return False

        completed = {
            s["stage"] for s in (peek.get("stages") or []) if s["status"] == "completed"
        }
        ready_stages = [
            stage
            for stage in STAGE_DISPATCH_ORDER
            if stage not in completed and STAGE_DEPS[stage] <= completed
        ]
        if not ready_stages:
            return True

        # Lazy admission: keep waiting jobs cleanly 'queued' (reorderable/cancellable)
        # until an entry lane they need actually has a free slot.
        if "audio_prep" in ready_stages:
            if self._lane_has_capacity_locked("audio_prep"):
                return True
            # When OCR allows multi-lecture concurrency (>1), admit immediately so OCR can
            # run while waiting for an audio_prep slot; when OCR is single-slot, wait for
            # audio_prep so J2 doesn't claim prematurely before J1 finishes ffmpeg.
            if (
                "ocr" in ready_stages
                and self._lane_capacity["ocr"] > 1
                and self._lane_has_capacity_locked("ocr")
                and not (
                    self._other_job_holds_llm_locked(None)
                    and self.settings.queue_ocr_overlap_with_llm == "strict"
                )
            ):
                return True
            return False

        if "transcription" in ready_stages and self._lane_has_capacity_locked("transcription"):
            return True
        if "ocr" in ready_stages and self._lane_has_capacity_locked("ocr"):
            if (
                self._other_job_holds_llm_locked(None)
                and self.settings.queue_ocr_overlap_with_llm == "strict"
            ):
                return False
            return True
        if ("alignment" in ready_stages or "notes" in ready_stages) and self._lane_has_capacity_locked("llm"):
            return True
        return False

    def _build_active_job_state(self, job: dict[str, Any]) -> _ActiveJobState:
        job_dir = Path(str(job["job_dir"]))
        input_dir = job_dir / "input"
        artifacts_dir = job_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        artifacts: dict[str, str] = dict(job.get("artifacts") or {})
        slides_rel = artifacts.get("slides_pdf", ARTIFACT_SLIDES_PDF)
        recording_rel = artifacts.get("recording", "")
        slides_path = job_dir / slides_rel
        recording_path = job_dir / recording_rel if recording_rel else input_dir / "recording"
        completed: set[PipelineStage] = {
            s["stage"]
            for s in (job.get("stages") or [])
            if s["status"] == "completed" and s["stage"] in STAGE_DEPS
        }
        queue_pos = int(job["queue_position"]) if job.get("queue_position") is not None else 0

        return _ActiveJobState(
            token=str(job["token"]),
            queue_position=queue_pos,
            slides_filename=str(job["slides_filename"]),
            recording_filename=str(job["recording_filename"]),
            job_dir=job_dir,
            input_dir=input_dir,
            artifacts_dir=artifacts_dir,
            slides_path=slides_path,
            recording_path=recording_path,
            artifacts=artifacts,
            completed=completed,
        )

    def _effective_settings_for_stage_locked(
        self,
        js: _ActiveJobState,
        stage: PipelineStage,
    ) -> Settings:
        if (
            stage == "ocr"
            and self._other_job_holds_llm_locked(js.token)
            and self.settings.queue_ocr_overlap_with_llm == "paced"
        ):
            lookahead_cap = max(1, int(self.settings.queue_lookahead_ocr_workers))
            capped = min(self.settings.ocr_parallel_workers, lookahead_cap)
            if capped != self.settings.ocr_parallel_workers:
                return dataclasses.replace(self.settings, ocr_parallel_workers=capped)
        return self.settings

    def _stage_context(self, js: _ActiveJobState, stage: PipelineStage) -> JobContext:
        return JobContext(
            token=js.token,
            slides_filename=js.slides_filename,
            recording_filename=js.recording_filename,
            job_dir=js.job_dir,
            input_dir=js.input_dir,
            artifacts_dir=js.artifacts_dir,
            slides_path=js.slides_path,
            recording_path=js.recording_path,
            artifacts=dict(js.artifacts),
            progress_callback=lambda message: self.store.update_stage_message(
                js.token, stage, message
            ),
            cancel_event=js.cancel_event,
        )

    def _coordinator_loop(self) -> None:
        while True:
            dispatched_any = False
            with self._cond:
                if self._stopping:
                    return

                paused = self.store.is_queue_paused()

                # 1. Admit eligible queued jobs lazily into the active pipeline window.
                while not self._stopping and self._can_admit_lookahead_locked(paused):
                    claimed = self.store.claim_next_queued_job()
                    if claimed is None:
                        break
                    token = str(claimed["token"])
                    if token not in self._done_events:
                        self._done_events[token] = threading.Event()
                    js = self._build_active_job_state(claimed)
                    if js.completed >= set(BACKGROUND_STAGES):
                        self.store.mark_job_completed(token, artifacts=js.artifacts)
                        self._signal_done_locked(token)
                        dispatched_any = True
                        continue
                    self._in_flight[token] = js
                    dispatched_any = True

                # 2. Dispatch ready stages strictly in queue_position ASC order.
                ordered = self._ordered_in_flight_locked()
                primaries = self._primary_tokens_locked(ordered)

                for js in ordered:
                    js.waiting_reason = None
                    if js.failed or js.cancel_event.is_set():
                        if not js.running_stages:
                            if js.cancel_event.is_set():
                                self.store.mark_job_cancelled(js.token)
                            self._in_flight.pop(js.token, None)
                            self._signal_done_locked(js.token)
                            dispatched_any = True
                        continue

                    if js.completed >= set(BACKGROUND_STAGES):
                        if not js.running_stages:
                            self.store.mark_job_completed(js.token, artifacts=js.artifacts)
                            self._in_flight.pop(js.token, None)
                            self._signal_done_locked(js.token)
                            dispatched_any = True
                        continue

                    # When paused, allow primary jobs to finish cleanly, but do not start
                    # new stages on lookahead ('preparing') jobs.
                    if paused and js.token not in primaries:
                        if not js.running_stages:
                            js.waiting_reason = "Queue paused."
                        continue

                    ready_stages = [
                        stage
                        for stage in STAGE_DISPATCH_ORDER
                        if stage not in js.completed
                        and stage not in js.running_stages
                        and STAGE_DEPS[stage] <= js.completed
                    ]

                    block_reasons: list[str] = []
                    for stage in ready_stages:
                        lane = STAGE_LANE[stage]
                        if not self._lane_has_capacity_locked(lane):
                            active_stages = list(self._lane_active[lane].values())
                            owner_label = (
                                STAGE_LABELS[active_stages[0]] if active_stages else lane
                            )
                            block_reasons.append(
                                f"Waiting for an active lecture to finish {owner_label}."
                            )
                            continue

                        if (
                            stage == "ocr"
                            and self._other_job_holds_llm_locked(js.token)
                            and self.settings.queue_ocr_overlap_with_llm == "strict"
                        ):
                            llm_stages = list(self._lane_active["llm"].values())
                            owner_label = (
                                STAGE_LABELS[llm_stages[0]] if llm_stages else "LLM stage"
                            )
                            block_reasons.append(
                                f"Waiting for active lecture to finish {owner_label} (strict Gemini pacing)."
                            )
                            continue

                        effective_settings = self._effective_settings_for_stage_locked(js, stage)
                        self._lane_active[lane][js.token] = stage
                        js.running_stages.add(stage)
                        self.store.start_stage(js.token, stage)
                        ctx = self._stage_context(js, stage)
                        try:
                            future = self._stage_pool.submit(
                                self._handlers[stage], ctx, effective_settings
                            )
                        except RuntimeError:
                            self._lane_active[lane].pop(js.token, None)
                            js.running_stages.discard(stage)
                            return

                        future.add_done_callback(
                            lambda fut, _tok=js.token, _st=stage, _ln=lane: self._on_stage_done(
                                _tok, _st, _ln, fut
                            )
                        )
                        dispatched_any = True

                    if not js.running_stages and block_reasons:
                        js.waiting_reason = block_reasons[0]

                if not dispatched_any and not self._stopping:
                    self._cond.wait(timeout=2.0)

    def _cleanup_raw_recording_if_configured(self, js: _ActiveJobState) -> None:
        if not self.settings.cleanup_raw_recording_after_audio_prep:
            return
        try:
            normalized = js.audio_normalized_path
            raw = js.recording_path
            if (
                normalized.is_file()
                and normalized.stat().st_size > 0
                and raw.is_file()
                and raw.resolve() != normalized.resolve()
            ):
                raw.unlink(missing_ok=True)
                logger.info("Removed raw recording %s after audio normalization.", raw.name)
        except OSError:
            logger.debug("Ignoring error removing raw recording for job %s.", js.token, exc_info=True)

    def _on_stage_done(
        self,
        token: str,
        stage: PipelineStage,
        lane: str,
        future: Future[StageResult],
    ) -> None:
        result: StageResult | None = None
        exc: BaseException | None = None
        try:
            result = future.result()
        except BaseException as err:  # noqa: BLE001 - capture any stage failure cleanly
            exc = err

        with self._cond:
            if self._lane_active[lane].get(token) == stage:
                self._lane_active[lane].pop(token, None)

            js = self._in_flight.get(token)
            if js is None:
                self._cond.notify_all()
                return

            js.running_stages.discard(stage)

            if js.cancel_event.is_set():
                if not js.running_stages:
                    self.store.mark_job_cancelled(token)
                    self._in_flight.pop(token, None)
                    self._signal_done_locked(token)
                self._cond.notify_all()
                return

            if exc is not None:
                js.failed = True
                if isinstance(exc, StageError):
                    logger.warning("Job %s failed at stage %s: %s", token, exc.stage, exc.message)
                    self.store.mark_job_failed(
                        token,
                        stage=exc.stage,
                        message=exc.message,
                        hint=exc.hint,
                    )
                    if exc.fatal_for_queue and self.settings.queue_auto_pause_on_auth_error:
                        logger.warning(
                            "Auto-pausing queue due to systemic error in job %s: %s",
                            token,
                            exc.message,
                        )
                        self.store.set_queue_paused(True, reason=exc.message)
                else:
                    logger.exception("Unexpected error in job %s at stage %s", token, stage)
                    self.store.mark_job_failed(
                        token,
                        stage=stage,
                        message=f"Unexpected error during {STAGE_LABELS[stage]}: {exc}",
                        hint="Check backend server logs and retry the upload.",
                    )

                if not js.running_stages:
                    self._in_flight.pop(token, None)
                    self._signal_done_locked(token)
                self._cond.notify_all()
                return

            assert result is not None
            self.store.complete_stage(
                token,
                stage,
                message=result.message or f"{STAGE_LABELS[stage]} completed.",
                artifacts=result.artifacts,
                stats=result.stats,
                warnings=result.warnings,
            )
            js.artifacts.update(result.artifacts)
            js.completed.add(stage)

            if stage == "audio_prep":
                self._cleanup_raw_recording_if_configured(js)

            if js.failed:
                if not js.running_stages:
                    self._in_flight.pop(token, None)
                    self._signal_done_locked(token)
            elif js.completed >= set(BACKGROUND_STAGES) and not js.running_stages:
                self.store.mark_job_completed(token, artifacts=js.artifacts)
                self._in_flight.pop(token, None)
                self._signal_done_locked(token)

            self._cond.notify_all()


__all__ = [
    "ALL_LANES",
    "JobWorker",
    "STAGE_DEPS",
    "STAGE_LANE",
    "StageHandler",
    "build_default_stage_handlers",
]
