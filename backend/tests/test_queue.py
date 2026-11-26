from __future__ import annotations

import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend.app.config import Settings, load_settings
from backend.app.main import create_app
from backend.app.pipeline import StageHandler
from backend.app.ratelimit import RateLimitRule, resolve_api_rate_limit_rule
from backend.app.schemas import (
    ARTIFACT_ALIGNMENT,
    ARTIFACT_AUDIO_NORMALIZED,
    ARTIFACT_NOTES_MD,
    ARTIFACT_SLIDES_OCR,
    ARTIFACT_TRANSCRIPT,
    AlignmentArtifact,
    JobContext,
    PipelineStage,
    SlidesOCRArtifact,
    StageError,
    StageResult,
    TranscriptArtifact,
)


def _make_settings(tmp_dir: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "data_dir": tmp_dir,
        "db_path": tmp_dir / "slidenotelm.db",
        "jobs_dir": tmp_dir / "jobs",
        "pipeline_llm_mode": "stub",
        "worker_concurrency": 1,
        "queue_max_in_flight_jobs": 2,
        "queue_lane_audio_prep_concurrency": 1,
        "queue_lane_transcription_concurrency": 1,
        "queue_lane_ocr_concurrency": 1,
        "queue_lane_llm_concurrency": 1,
        "queue_ocr_overlap_with_llm": "paced",
        "queue_lookahead_ocr_workers": 2,
        "cleanup_raw_recording_after_audio_prep": True,
        "queue_auto_pause_on_auth_error": True,
        "min_free_disk_bytes": 0,
        "rate_limit_enabled": False,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def _stub_audio_prep(context: JobContext, _settings: Settings) -> StageResult:
    context.audio_normalized_path.parent.mkdir(parents=True, exist_ok=True)
    context.audio_normalized_path.write_bytes(b"ID3-normalized-mp3")
    return StageResult(
        message="Audio normalized.",
        artifacts={"audio_normalized": ARTIFACT_AUDIO_NORMALIZED},
    )


def _stub_transcription(context: JobContext, settings: Settings) -> StageResult:
    artifact = TranscriptArtifact(
        source_recording=context.recording_filename,
        model=settings.transcription_model,
        duration_seconds=10.0,
        segments=[],
    )
    context.transcript_path.write_text(artifact.model_dump_json(), encoding="utf-8")
    return StageResult(
        message="Transcribed.",
        artifacts={"transcript": ARTIFACT_TRANSCRIPT},
        stats={"duration_seconds": 10.0, "transcript_segments": 0},
    )


def _stub_ocr(context: JobContext, settings: Settings) -> StageResult:
    artifact = SlidesOCRArtifact(
        source_pdf=context.slides_filename,
        model=settings.ocr_model_google,
        slide_count=1,
        slides=[],
    )
    context.slides_ocr_path.write_text(artifact.model_dump_json(), encoding="utf-8")
    return StageResult(
        message="OCR done.",
        artifacts={"slides_ocr": ARTIFACT_SLIDES_OCR},
        stats={"slide_count": 1},
    )


def _stub_alignment(context: JobContext, _settings: Settings) -> StageResult:
    artifact = AlignmentArtifact(mode="stub", entries=[], alignment_issues=[])
    context.alignment_path.write_text(artifact.model_dump_json(), encoding="utf-8")
    return StageResult(
        message="Aligned.",
        artifacts={"alignment": ARTIFACT_ALIGNMENT},
    )


def _stub_notes(context: JobContext, _settings: Settings) -> StageResult:
    context.notes_path.write_text(f"# {context.slides_filename}\n\nNotes content.\n", encoding="utf-8")
    return StageResult(
        message="Notes generated.",
        artifacts={"notes_md": ARTIFACT_NOTES_MD},
        stats={"word_count": 4},
    )


def _stub_handlers(**overrides: StageHandler) -> dict[PipelineStage, StageHandler]:
    handlers: dict[PipelineStage, StageHandler] = {
        "audio_prep": _stub_audio_prep,
        "transcription": _stub_transcription,
        "ocr": _stub_ocr,
        "alignment": _stub_alignment,
        "notes": _stub_notes,
    }
    for name, fn in overrides.items():
        handlers[name] = fn  # type: ignore[index]
    return handlers


def _upload_pair(
    client: TestClient,
    name: str,
    *,
    batch_id: str | None = None,
    label: str | None = None,
) -> dict:
    data: dict[str, str] = {}
    if batch_id is not None:
        data["batch_id"] = batch_id
    if label is not None:
        data["label"] = label
    resp = client.post(
        "/api/jobs",
        files={
            "slides": (f"{name}.pdf", b"%PDF-1.4\n%test", "application/pdf"),
            "recording": (f"{name}.mp4", b"fake-video-bytes", "video/mp4"),
        },
        data=data or None,
    )
    assert resp.status_code == 202, resp.text
    return resp.json()


class QueuePipeliningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-queue-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cross_job_stage_pipelining_starts_j2_audio_when_j1_audio_finishes_and_bounds_lookahead(self) -> None:
        j1_audio_started = threading.Event()
        release_j1_audio = threading.Event()
        j1_notes_started = threading.Event()
        release_j1_notes = threading.Event()
        j2_audio_started = threading.Event()
        j2_transcription_done = threading.Event()
        j2_ocr_done = threading.Event()
        j3_audio_started = threading.Event()

        def audio_prep(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "lec1.pdf":
                j1_audio_started.set()
                self.assertTrue(release_j1_audio.wait(timeout=5.0))
            elif ctx.slides_filename == "lec2.pdf":
                j2_audio_started.set()
            elif ctx.slides_filename == "lec3.pdf":
                j3_audio_started.set()
            return _stub_audio_prep(ctx, settings)

        def transcription(ctx: JobContext, settings: Settings) -> StageResult:
            res = _stub_transcription(ctx, settings)
            if ctx.slides_filename == "lec2.pdf":
                j2_transcription_done.set()
            return res

        def ocr(ctx: JobContext, settings: Settings) -> StageResult:
            res = _stub_ocr(ctx, settings)
            if ctx.slides_filename == "lec2.pdf":
                j2_ocr_done.set()
            return res

        def notes(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "lec1.pdf":
                j1_notes_started.set()
                self.assertTrue(release_j1_notes.wait(timeout=5.0))
            return _stub_notes(ctx, settings)

        app = create_app(
            settings=_make_settings(self.tmp_dir, queue_max_in_flight_jobs=2),
            stage_handlers=_stub_handlers(
                audio_prep=audio_prep,
                transcription=transcription,
                ocr=ocr,
                notes=notes,
            ),
        )

        with TestClient(app) as client:
            j1 = _upload_pair(client, "lec1", batch_id="batch_1")
            self.assertTrue(j1_audio_started.wait(timeout=5.0))

            j2 = _upload_pair(client, "lec2", batch_id="batch_1")
            j3 = _upload_pair(client, "lec3", batch_id="batch_1")

            # While J1 holds audio_prep, J2 and J3 stay cleanly queued (lazy admission).
            snap_before = client.get("/api/queue").json()
            self.assertEqual([a["token"] for a in snap_before["active"]], [j1["token"]])
            self.assertEqual([q["token"] for q in snap_before["queued"]], [j2["token"], j3["token"]])
            self.assertFalse(j2_audio_started.is_set())

            # Release J1 audio_prep -> J2 must IMMEDIATELY start audio_prep and pre-warm
            # transcription + ocr while J1 is still blocked in notes!
            release_j1_audio.set()
            self.assertTrue(j1_notes_started.wait(timeout=5.0))
            self.assertTrue(j2_audio_started.wait(timeout=5.0))
            self.assertTrue(j2_transcription_done.wait(timeout=5.0))
            self.assertTrue(j2_ocr_done.wait(timeout=5.0))

            # J3 must remain queued because max_in_flight_jobs == 2.
            self.assertFalse(j3_audio_started.is_set())

            snap_mid = client.get("/api/queue").json()
            self.assertEqual([a["token"] for a in snap_mid["active"]], [j1["token"], j2["token"]])
            self.assertEqual(snap_mid["active"][0]["pipeline_role"], "primary")
            self.assertEqual(snap_mid["active"][1]["pipeline_role"], "preparing")
            self.assertEqual([q["token"] for q in snap_mid["queued"]], [j3["token"]])
            self.assertEqual(snap_mid["queued"][0]["queue_index"], 1)

            # Release J1 notes -> J1 completes, J2 runs alignment+notes, J3 runs to completion.
            release_j1_notes.set()
            app.state.worker.wait_for_job(j1["token"], timeout=5.0)
            app.state.worker.wait_for_job(j2["token"], timeout=5.0)
            app.state.worker.wait_for_job(j3["token"], timeout=5.0)

            self.assertTrue(j3_audio_started.is_set())
            snap_end = client.get("/api/queue").json()
            self.assertEqual(snap_end["counts"]["completed"], 3)
            self.assertEqual(snap_end["active"], [])
            self.assertEqual(snap_end["queued"], [])

    def test_multi_slot_lanes_process_multiple_lectures_simultaneously(self) -> None:
        # With default multi-slot lanes (audio=2, transcription=2, ocr=3, llm=3, max_in_flight=4):
        # J1, J2, J3 must genuinely run `notes` in parallel while J4 pre-warms and J5 stays queued.
        notes_lock = threading.Lock()
        active_notes: set[str] = set()
        three_in_notes = threading.Event()
        release_notes = threading.Event()
        j4_prepared = threading.Event()
        j5_started = threading.Event()

        def audio_prep(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "lec5.pdf":
                j5_started.set()
            return _stub_audio_prep(ctx, settings)

        def transcription(ctx: JobContext, settings: Settings) -> StageResult:
            res = _stub_transcription(ctx, settings)
            if ctx.slides_filename == "lec4.pdf":
                j4_prepared.set()
            return res

        def notes(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename in {"lec1.pdf", "lec2.pdf", "lec3.pdf"}:
                with notes_lock:
                    active_notes.add(ctx.slides_filename)
                    if len(active_notes) == 3:
                        three_in_notes.set()
                self.assertTrue(release_notes.wait(timeout=5.0))
            return _stub_notes(ctx, settings)

        app = create_app(
            settings=_make_settings(
                self.tmp_dir,
                queue_max_in_flight_jobs=4,
                queue_lane_audio_prep_concurrency=2,
                queue_lane_transcription_concurrency=2,
                queue_lane_ocr_concurrency=3,
                queue_lane_llm_concurrency=3,
                queue_ocr_overlap_with_llm="full",
            ),
            stage_handlers=_stub_handlers(
                audio_prep=audio_prep,
                transcription=transcription,
                notes=notes,
            ),
        )

        with TestClient(app) as client:
            jobs = [_upload_pair(client, f"lec{i}") for i in range(1, 6)]

            # All 3 primary lectures reach `notes` concurrently AND J4 pre-warms in parallel!
            self.assertTrue(three_in_notes.wait(timeout=5.0))
            self.assertTrue(j4_prepared.wait(timeout=5.0))
            self.assertFalse(j5_started.is_set())

            snap = client.get("/api/queue").json()
            self.assertEqual(len(snap["active"]), 4)
            roles = [item["pipeline_role"] for item in snap["active"]]
            self.assertEqual(roles, ["primary", "primary", "primary", "preparing"])
            self.assertEqual([q["token"] for q in snap["queued"]], [jobs[4]["token"]])

            release_notes.set()
            for j in jobs:
                app.state.worker.wait_for_job(j["token"], timeout=5.0)

            self.assertTrue(j5_started.is_set())
            final_snap = client.get("/api/queue").json()
            self.assertEqual(final_snap["counts"]["completed"], 5)
            self.assertEqual(final_snap["active"], [])
            self.assertEqual(final_snap["queued"], [])

    def test_strict_serial_mode_max_in_flight_1_waits_for_j1_to_complete(self) -> None:
        j1_notes_started = threading.Event()
        release_j1_notes = threading.Event()
        j2_audio_started = threading.Event()

        def audio_prep(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "lec2.pdf":
                j2_audio_started.set()
            return _stub_audio_prep(ctx, settings)

        def notes(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "lec1.pdf":
                j1_notes_started.set()
                self.assertTrue(release_j1_notes.wait(timeout=5.0))
            return _stub_notes(ctx, settings)

        app = create_app(
            settings=_make_settings(self.tmp_dir, queue_max_in_flight_jobs=1),
            stage_handlers=_stub_handlers(audio_prep=audio_prep, notes=notes),
        )
        with TestClient(app) as client:
            j1 = _upload_pair(client, "lec1")
            j2 = _upload_pair(client, "lec2")
            self.assertTrue(j1_notes_started.wait(timeout=5.0))
            self.assertFalse(j2_audio_started.wait(timeout=0.15))

            release_j1_notes.set()
            app.state.worker.wait_for_job(j1["token"], timeout=5.0)
            app.state.worker.wait_for_job(j2["token"], timeout=5.0)
            self.assertTrue(j2_audio_started.is_set())

    def test_ocr_overlap_policies_strict_vs_paced(self) -> None:
        # 1) strict mode: J2 runs audio+transcription while J1 is in notes, but holds OCR.
        j1_notes_started = threading.Event()
        release_j1_notes = threading.Event()
        j2_transcribed = threading.Event()
        j2_ocr_started = threading.Event()

        def strict_notes(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "lec1.pdf":
                j1_notes_started.set()
                self.assertTrue(release_j1_notes.wait(timeout=5.0))
            return _stub_notes(ctx, settings)

        def strict_transcription(ctx: JobContext, settings: Settings) -> StageResult:
            res = _stub_transcription(ctx, settings)
            if ctx.slides_filename == "lec2.pdf":
                j2_transcribed.set()
            return res

        def strict_ocr(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "lec2.pdf":
                j2_ocr_started.set()
            return _stub_ocr(ctx, settings)

        strict_app = create_app(
            settings=_make_settings(
                self.tmp_dir / "strict",
                queue_max_in_flight_jobs=2,
                queue_ocr_overlap_with_llm="strict",
            ),
            stage_handlers=_stub_handlers(
                notes=strict_notes,
                transcription=strict_transcription,
                ocr=strict_ocr,
            ),
        )
        with TestClient(strict_app) as client:
            j1 = _upload_pair(client, "lec1")
            self.assertTrue(j1_notes_started.wait(timeout=5.0))
            j2 = _upload_pair(client, "lec2")
            self.assertTrue(j2_transcribed.wait(timeout=5.0))
            # OCR must NOT start while J1 holds llm lane in strict mode.
            self.assertFalse(j2_ocr_started.wait(timeout=0.15))

            release_j1_notes.set()
            strict_app.state.worker.wait_for_job(j1["token"], timeout=5.0)
            strict_app.state.worker.wait_for_job(j2["token"], timeout=5.0)
            self.assertTrue(j2_ocr_started.is_set())

        # 2) paced mode: J2 runs OCR while J1 is in notes, with ocr_parallel_workers capped.
        p_j1_notes = threading.Event()
        p_release_j1 = threading.Event()
        p_j2_ocr_workers: list[int] = []
        p_j2_ocr_done = threading.Event()

        def paced_notes(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "lec1.pdf":
                p_j1_notes.set()
                self.assertTrue(p_release_j1.wait(timeout=5.0))
            return _stub_notes(ctx, settings)

        def paced_ocr(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "lec2.pdf":
                p_j2_ocr_workers.append(settings.ocr_parallel_workers)
                res = _stub_ocr(ctx, settings)
                p_j2_ocr_done.set()
                return res
            return _stub_ocr(ctx, settings)

        paced_app = create_app(
            settings=_make_settings(
                self.tmp_dir / "paced",
                queue_max_in_flight_jobs=2,
                queue_ocr_overlap_with_llm="paced",
                ocr_parallel_workers=8,
                queue_lookahead_ocr_workers=2,
            ),
            stage_handlers=_stub_handlers(notes=paced_notes, ocr=paced_ocr),
        )
        with TestClient(paced_app) as client:
            j1 = _upload_pair(client, "lec1")
            self.assertTrue(p_j1_notes.wait(timeout=5.0))
            j2 = _upload_pair(client, "lec2")
            self.assertTrue(p_j2_ocr_done.wait(timeout=5.0))
            self.assertEqual(p_j2_ocr_workers, [2])
            p_release_j1.set()
            paced_app.state.worker.wait_for_job(j1["token"], timeout=5.0)
            paced_app.state.worker.wait_for_job(j2["token"], timeout=5.0)

    def test_raw_recording_deleted_after_audio_prep_and_checkpointed_retry_skips_completed_stages(self) -> None:
        call_counts = {"audio_prep": 0, "transcription": 0, "ocr": 0, "alignment": 0, "notes": 0}
        fail_notes_once = {"should_fail": True}

        def count_audio(ctx: JobContext, settings: Settings) -> StageResult:
            call_counts["audio_prep"] += 1
            return _stub_audio_prep(ctx, settings)

        def count_transcribe(ctx: JobContext, settings: Settings) -> StageResult:
            call_counts["transcription"] += 1
            return _stub_transcription(ctx, settings)

        def count_ocr(ctx: JobContext, settings: Settings) -> StageResult:
            call_counts["ocr"] += 1
            return _stub_ocr(ctx, settings)

        def count_align(ctx: JobContext, settings: Settings) -> StageResult:
            call_counts["alignment"] += 1
            return _stub_alignment(ctx, settings)

        def flaky_notes(ctx: JobContext, settings: Settings) -> StageResult:
            call_counts["notes"] += 1
            if fail_notes_once["should_fail"]:
                fail_notes_once["should_fail"] = False
                raise StageError(
                    stage="notes",
                    message="Transient Gemini outage.",
                    hint="Click Retry to resume.",
                )
            return _stub_notes(ctx, settings)

        app = create_app(
            settings=_make_settings(
                self.tmp_dir,
                cleanup_raw_recording_after_audio_prep=True,
            ),
            stage_handlers=_stub_handlers(
                audio_prep=count_audio,
                transcription=count_transcribe,
                ocr=count_ocr,
                alignment=count_align,
                notes=flaky_notes,
            ),
        )
        with TestClient(app) as client:
            created = _upload_pair(client, "lec_retry")
            token = created["token"]
            app.state.worker.wait_for_job(token, timeout=5.0)

            failed = client.get(f"/api/jobs/{token}").json()
            self.assertEqual(failed["status"], "failed")
            self.assertTrue(failed["can_retry"])
            self.assertFalse(failed["can_retry_from_scratch"])

            # Raw recording was cleaned up right after audio_prep, normalized MP3 remains.
            job_dir = app.state.settings.jobs_dir / token
            self.assertFalse((job_dir / "input" / "recording.mp4").exists())
            self.assertTrue((job_dir / ARTIFACT_AUDIO_NORMALIZED).is_file())

            # Retry from scratch must fail cleanly with 409 because raw video was deleted.
            scratch_resp = client.post(
                f"/api/jobs/{token}/retry",
                json={"from_scratch": True},
            )
            self.assertEqual(scratch_resp.status_code, 409)

            # Checkpointed retry resumes cleanly at notes without re-running earlier stages!
            retry_resp = client.post(f"/api/jobs/{token}/retry", json={"from_scratch": False})
            self.assertEqual(retry_resp.status_code, 202)
            app.state.worker.wait_for_job(token, timeout=5.0)

            done = client.get(f"/api/jobs/{token}").json()
            self.assertEqual(done["status"], "completed")
            self.assertTrue(done["download_ready"])
            self.assertEqual(call_counts["audio_prep"], 1)
            self.assertEqual(call_counts["transcription"], 1)
            self.assertEqual(call_counts["ocr"], 1)
            self.assertEqual(call_counts["alignment"], 1)
            self.assertEqual(call_counts["notes"], 2)

    def test_reorder_queued_jobs_and_ignore_non_queued_gracefully(self) -> None:
        order_executed: list[str] = []
        j1_started = threading.Event()
        release_j1 = threading.Event()

        def audio_prep(ctx: JobContext, settings: Settings) -> StageResult:
            order_executed.append(ctx.slides_filename)
            if ctx.slides_filename == "lec1.pdf":
                j1_started.set()
                self.assertTrue(release_j1.wait(timeout=5.0))
            return _stub_audio_prep(ctx, settings)

        app = create_app(
            settings=_make_settings(self.tmp_dir, queue_max_in_flight_jobs=1),
            stage_handlers=_stub_handlers(audio_prep=audio_prep),
        )
        with TestClient(app) as client:
            j1 = _upload_pair(client, "lec1")
            self.assertTrue(j1_started.wait(timeout=5.0))
            j2 = _upload_pair(client, "lec2")
            j3 = _upload_pair(client, "lec3")
            j4 = _upload_pair(client, "lec4")

            # Reorder puts J4 first, J2 second; J1 (active) is ignored; J3 (omitted) appends to tail.
            reorder_resp = client.patch(
                "/api/queue/order",
                json={"tokens": [j1["token"], j4["token"], j2["token"]]},
            )
            self.assertEqual(reorder_resp.status_code, 200)
            queued_tokens = [q["token"] for q in reorder_resp.json()["queued"]]
            self.assertEqual(queued_tokens, [j4["token"], j2["token"], j3["token"]])

            release_j1.set()
            for tok in (j1["token"], j2["token"], j3["token"], j4["token"]):
                app.state.worker.wait_for_job(tok, timeout=5.0)

            self.assertEqual(order_executed, ["lec1.pdf", "lec4.pdf", "lec2.pdf", "lec3.pdf"])

    def test_cancel_queued_and_active_jobs(self) -> None:
        j1_started = threading.Event()
        j1_cancelled_seen = threading.Event()
        j2_ran = threading.Event()

        def audio_prep(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "lec1.pdf":
                j1_started.set()
                assert ctx.cancel_event is not None
                if ctx.cancel_event.wait(timeout=5.0):
                    j1_cancelled_seen.set()
                    raise StageError(stage="audio_prep", message="Cancelled", hint="Cancelled")
            elif ctx.slides_filename == "lec2.pdf":
                j2_ran.set()
            return _stub_audio_prep(ctx, settings)

        app = create_app(
            settings=_make_settings(self.tmp_dir, queue_max_in_flight_jobs=1),
            stage_handlers=_stub_handlers(audio_prep=audio_prep),
        )
        with TestClient(app) as client:
            j1 = _upload_pair(client, "lec1")
            self.assertTrue(j1_started.wait(timeout=5.0))
            j2 = _upload_pair(client, "lec2")
            j3 = _upload_pair(client, "lec3")

            # Cancel queued J2 -> immediately cancelled and disk folder deleted by default.
            cancel_j2 = client.post(f"/api/jobs/{j2['token']}/cancel")
            self.assertEqual(cancel_j2.status_code, 200)
            self.assertEqual(cancel_j2.json()["status"], "cancelled")
            self.assertFalse((app.state.settings.jobs_dir / j2["token"]).exists())

            # Cancel active J1 -> cooperative cancellation unwinds and J3 starts and completes.
            cancel_j1 = client.post(f"/api/jobs/{j1['token']}/cancel")
            self.assertEqual(cancel_j1.status_code, 200)
            self.assertTrue(j1_cancelled_seen.wait(timeout=5.0))
            app.state.worker.wait_for_job(j1["token"], timeout=5.0)
            app.state.worker.wait_for_job(j3["token"], timeout=5.0)

            self.assertEqual(client.get(f"/api/jobs/{j1['token']}").json()["status"], "cancelled")
            self.assertEqual(client.get(f"/api/jobs/{j3['token']}").json()["status"], "completed")
            self.assertFalse(j2_ran.is_set())

            # Cannot cancel already-completed J3 -> 409.
            self.assertEqual(client.post(f"/api/jobs/{j3['token']}/cancel").status_code, 409)

    def test_failure_isolation_vs_systemic_auto_pause_and_resume_across_restart(self) -> None:
        # 1) Non-fatal per-lecture failure: J1 fails, J2 still completes automatically.
        def non_fatal_ocr(ctx: JobContext, settings: Settings) -> StageResult:
            if ctx.slides_filename == "bad.pdf":
                raise StageError(stage="ocr", message="Corrupt PDF", hint="Re-export PDF")
            return _stub_ocr(ctx, settings)

        app1 = create_app(
            settings=_make_settings(self.tmp_dir / "iso", queue_max_in_flight_jobs=1),
            stage_handlers=_stub_handlers(ocr=non_fatal_ocr),
        )
        with TestClient(app1) as client:
            j1 = _upload_pair(client, "bad")
            j2 = _upload_pair(client, "good")
            app1.state.worker.wait_for_job(j1["token"], timeout=5.0)
            app1.state.worker.wait_for_job(j2["token"], timeout=5.0)
            self.assertEqual(client.get(f"/api/jobs/{j1['token']}").json()["status"], "failed")
            self.assertEqual(client.get(f"/api/jobs/{j2['token']}").json()["status"], "completed")
            self.assertFalse(client.get("/api/queue").json()["paused"])

        # 2) Systemic auth error (fatal_for_queue=True): auto-pauses queue and persists across restart.
        fail_auth = {"enabled": True}

        def auth_failing_ocr(ctx: JobContext, settings: Settings) -> StageResult:
            if fail_auth["enabled"]:
                raise StageError(
                    stage="ocr",
                    message="GEMINI_API_KEY is not configured.",
                    hint="Set GEMINI_API_KEY and resume.",
                    fatal_for_queue=True,
                )
            return _stub_ocr(ctx, settings)

        settings2 = _make_settings(self.tmp_dir / "autopause", queue_max_in_flight_jobs=1)
        app2 = create_app(settings=settings2, stage_handlers=_stub_handlers(ocr=auth_failing_ocr))
        with TestClient(app2) as client:
            # Pause before enqueuing so both jobs exist before J1 runs.
            client.post("/api/queue/pause")
            j1 = _upload_pair(client, "lec1")
            j2 = _upload_pair(client, "lec2")
            client.post("/api/queue/resume")

            app2.state.worker.wait_for_job(j1["token"], timeout=5.0)
            snap = client.get("/api/queue").json()
            self.assertTrue(snap["paused"])
            self.assertIn("GEMINI_API_KEY", snap["pause_reason"])
            self.assertEqual([q["token"] for q in snap["queued"]], [j2["token"]])

        # Simulate server restart: queue MUST remain paused until explicitly resumed!
        fail_auth["enabled"] = False
        app2_restarted = create_app(
            settings=settings2,
            stage_handlers=_stub_handlers(ocr=auth_failing_ocr),
        )
        with TestClient(app2_restarted) as client:
            snap_restarted = client.get("/api/queue").json()
            self.assertTrue(snap_restarted["paused"])
            self.assertEqual([q["token"] for q in snap_restarted["queued"]], [j2["token"]])

            resume_resp = client.post("/api/queue/resume")
            self.assertEqual(resume_resp.status_code, 200)
            self.assertFalse(resume_resp.json()["paused"])
            app2_restarted.state.worker.wait_for_job(j2["token"], timeout=5.0)
            self.assertEqual(client.get(f"/api/jobs/{j2['token']}").json()["status"], "completed")

    def test_delete_job_and_queue_depth_and_disk_guardrails(self) -> None:
        settings = _make_settings(self.tmp_dir, max_queue_depth=2, queue_max_in_flight_jobs=1)
        app = create_app(settings=settings, stage_handlers=_stub_handlers())
        with TestClient(app) as client:
            client.post("/api/queue/pause")
            j1 = _upload_pair(client, "lec1")
            _upload_pair(client, "lec2")

            # 3rd upload exceeds max_queue_depth=2 -> 409 Conflict
            over = client.post(
                "/api/jobs",
                files={
                    "slides": ("lec3.pdf", b"%PDF-1.4\n%test", "application/pdf"),
                    "recording": ("lec3.mp4", b"video", "video/mp4"),
                },
            )
            self.assertEqual(over.status_code, 409)

            # Cannot DELETE a queued job -> 409
            self.assertEqual(client.delete(f"/api/jobs/{j1['token']}").status_code, 409)

            # Resume, finish, then DELETE succeeds with 204 and removes disk dir.
            client.post("/api/queue/resume")
            app.state.worker.wait_for_job(j1["token"], timeout=5.0)
            del_resp = client.delete(f"/api/jobs/{j1['token']}")
            self.assertEqual(del_resp.status_code, 204)
            self.assertFalse((settings.jobs_dir / j1["token"]).exists())
            self.assertEqual(client.get(f"/api/jobs/{j1['token']}").status_code, 404)

        # Low free disk space -> 507 Insufficient Storage
        disk_app = create_app(
            settings=_make_settings(self.tmp_dir / "disk", min_free_disk_bytes=1024 * 1024 * 1024),
            stage_handlers=_stub_handlers(),
        )
        fake_usage = shutil._ntuple_diskusage(total=1000, used=900, free=100)
        with mock.patch("backend.app.main.shutil.disk_usage", return_value=fake_usage):
            with TestClient(disk_app) as client:
                resp = client.post(
                    "/api/jobs",
                    files={
                        "slides": ("lec.pdf", b"%PDF-1.4\n%test", "application/pdf"),
                        "recording": ("lec.mp4", b"video", "video/mp4"),
                    },
                )
                self.assertEqual(resp.status_code, 507)

    def test_auth_middleware_rate_limits_and_env_parsing_for_queue(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            app_password="queue-password",
            rate_limit_enabled=True,
            rate_limit_window_seconds=60,
            rate_limit_jobs_read_per_window=120,
            rate_limit_queue_control_per_window=60,
        )
        # Rate limit rule mapping checks
        self.assertEqual(
            resolve_api_rate_limit_rule("GET", "/api/queue", settings),
            RateLimitRule("jobs_read", 120, 60),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("PATCH", "/api/queue/order", settings),
            RateLimitRule("queue_control", 60, 60),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("POST", "/api/queue/pause", settings),
            RateLimitRule("queue_control", 60, 60),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("POST", "/api/jobs/tok_1/cancel", settings),
            RateLimitRule("queue_control", 60, 60),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("POST", "/api/jobs/tok_1/retry", settings),
            RateLimitRule("queue_control", 60, 60),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("DELETE", "/api/jobs/tok_1", settings),
            RateLimitRule("queue_control", 60, 60),
        )

        # Auth middleware protects all /api/queue* and /api/jobs/* control routes
        app = create_app(settings=settings, stage_handlers=_stub_handlers())
        with TestClient(app) as client:
            for method, path in [
                ("GET", "/api/queue"),
                ("PATCH", "/api/queue/order"),
                ("POST", "/api/queue/pause"),
                ("POST", "/api/queue/resume"),
                ("POST", "/api/jobs/tok_1/cancel"),
                ("POST", "/api/jobs/tok_1/retry"),
                ("DELETE", "/api/jobs/tok_1"),
            ]:
                self.assertEqual(client.request(method, path).status_code, 401, msg=f"{method} {path}")

            login = client.post("/api/auth/login", json={"password": "queue-password"})
            self.assertEqual(login.status_code, 200)
            self.assertEqual(client.get("/api/queue").status_code, 200)

        # Env var parsing check
        env_patch = {
            "QUEUE_MAX_IN_FLIGHT_JOBS": "5",
            "QUEUE_LANE_AUDIO_PREP_CONCURRENCY": "3",
            "QUEUE_LANE_TRANSCRIPTION_CONCURRENCY": "2",
            "QUEUE_LANE_OCR_CONCURRENCY": "4",
            "QUEUE_LANE_LLM_CONCURRENCY": "4",
            "QUEUE_OCR_OVERLAP_WITH_LLM": "strict",
            "QUEUE_LOOKAHEAD_OCR_WORKERS": "4",
            "GEMINI_COMBINED_RATE_LIMIT_RPM": "150",
            "QUEUE_AUTO_PAUSE_ON_AUTH_ERROR": "false",
            "CLEANUP_RAW_RECORDING_AFTER_AUDIO_PREP": "false",
            "MAX_QUEUE_DEPTH": "25",
            "MIN_FREE_DISK_BYTES": "1048576",
            "RATE_LIMIT_QUEUE_CONTROL_PER_MINUTE": "45",
        }
        with mock.patch.dict(os.environ, env_patch, clear=False):
            cfg = load_settings(env_file=self.tmp_dir / "nonexistent.env")
        self.assertEqual(cfg.queue_max_in_flight_jobs, 5)
        self.assertEqual(cfg.queue_lane_audio_prep_concurrency, 3)
        self.assertEqual(cfg.queue_lane_transcription_concurrency, 2)
        self.assertEqual(cfg.queue_lane_ocr_concurrency, 4)
        self.assertEqual(cfg.queue_lane_llm_concurrency, 4)
        self.assertEqual(cfg.queue_ocr_overlap_with_llm, "strict")
        self.assertEqual(cfg.queue_lookahead_ocr_workers, 4)
        self.assertEqual(cfg.gemini_combined_rate_limit_rpm, 150)
        self.assertFalse(cfg.queue_auto_pause_on_auth_error)
        self.assertFalse(cfg.cleanup_raw_recording_after_audio_prep)
        self.assertEqual(cfg.max_queue_depth, 25)
        self.assertEqual(cfg.min_free_disk_bytes, 1048576)
        self.assertEqual(cfg.rate_limit_queue_control_per_window, 45)


if __name__ == "__main__":
    unittest.main()
