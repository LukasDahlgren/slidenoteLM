from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pymupdf
from fastapi.testclient import TestClient

from backend.app.config import Settings, load_settings
from backend.app.db import (
    INTERRUPTED_HINT,
    INTERRUPTED_MESSAGE,
    SCHEMA_VERSION,
    JobStore,
    generate_job_token,
)
from backend.app.main import build_download_filename, create_app
from backend.app.pipeline import StageHandler
from backend.app.pipeline.audio import FFmpegBinaries
from backend.app.schemas import (
    ARTIFACT_ALIGNMENT,
    ARTIFACT_AUDIO_NORMALIZED,
    ARTIFACT_NOTES_MD,
    ARTIFACT_SLIDES_OCR,
    ARTIFACT_TRANSCRIPT,
    AlignmentArtifact,
    PIPELINE_STAGES,
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
        "worker_concurrency": 2,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def _stub_audio_prep(context: JobContext, _settings: Settings) -> StageResult:
    context.audio_normalized_path.write_bytes(b"stub-mp3")
    return StageResult(
        message="Audio prep stub completed.",
        artifacts={"audio_normalized": "artifacts/audio_normalized.mp3"},
    )


def _stub_transcription(context: JobContext, settings: Settings) -> StageResult:
    artifact = TranscriptArtifact(
        source_recording=context.recording_filename,
        model=settings.transcription_model,
        duration_seconds=0.0,
        segments=[],
    )
    context.transcript_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return StageResult(
        message="Transcription stub completed.",
        artifacts={"transcript": ARTIFACT_TRANSCRIPT},
        stats={"duration_seconds": 0.0, "transcript_segments": 0},
    )


def _stub_ocr(context: JobContext, settings: Settings) -> StageResult:
    artifact = SlidesOCRArtifact(
        source_pdf=context.slides_filename,
        model=settings.ocr_model_google,
        slide_count=0,
        slides=[],
    )
    context.slides_ocr_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return StageResult(
        message="OCR stub completed.",
        artifacts={"slides_ocr": ARTIFACT_SLIDES_OCR},
        stats={"slide_count": 0},
    )


def _phase1_stub_handlers(**overrides: StageHandler) -> dict[PipelineStage, StageHandler]:
    handlers: dict[PipelineStage, StageHandler] = {
        "audio_prep": _stub_audio_prep,
        "transcription": _stub_transcription,
        "ocr": _stub_ocr,
    }
    for stage_name, handler in overrides.items():
        handlers[stage_name] = handler  # type: ignore[index]
    return handlers


def _build_pdf_bytes(page_texts: list[str]) -> bytes:
    doc = pymupdf.open()
    try:
        for text in page_texts:
            page = doc.new_page(width=640, height=360)
            page.insert_text((48, 72), text, fontsize=16)
        return bytes(doc.tobytes())
    finally:
        doc.close()


class Phase1BackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-p1-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_health_endpoint(self) -> None:
        app = create_app(settings=_make_settings(self.tmp_dir))
        with TestClient(app) as client:
            resp = client.get("/api/health")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(
                resp.json(),
                {"status": "ok", "service": "slidenoteLM", "pipeline_llm_mode": "stub"},
            )

    def test_private_token_entropy_and_uniqueness(self) -> None:
        tokens = {generate_job_token() for _ in range(50)}
        self.assertEqual(len(tokens), 50)
        for token in tokens:
            self.assertGreaterEqual(len(token), 32)

    def test_download_filename_sanitization(self) -> None:
        self.assertEqual(
            build_download_filename("Lecture 03: Neural Nets & Backprop.PDF"),
            "lecture-03-neural-nets-backprop-notes.md",
        )
        self.assertEqual(build_download_filename("...pdf"), "lecture-notes.md")

    def test_upload_validation_rejects_bad_inputs_and_cleans_up(self) -> None:
        settings = _make_settings(self.tmp_dir, max_pdf_bytes=64, max_recording_bytes=128)
        app = create_app(settings=settings)
        with TestClient(app) as client:
            # Non-PDF extension for slides
            resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("slides.txt", b"%PDF-1.4 fake", "text/plain"),
                    "recording": ("lecture.mp3", b"audio-bytes", "audio/mpeg"),
                },
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn(".pdf", resp.json()["detail"])

            # Invalid PDF magic header
            resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("slides.pdf", b"NOT-A-PDF-HEADER", "application/pdf"),
                    "recording": ("lecture.mp3", b"audio-bytes", "audio/mpeg"),
                },
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("valid PDF", resp.json()["detail"])

            # Unsupported recording format
            resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("slides.pdf", b"%PDF-1.4\nbody", "application/pdf"),
                    "recording": ("lecture.exe", b"binary", "application/octet-stream"),
                },
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("Unsupported recording format", resp.json()["detail"])

            # Empty recording file
            resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("slides.pdf", b"%PDF-1.4\nbody", "application/pdf"),
                    "recording": ("lecture.mp4", b"", "video/mp4"),
                },
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn("empty", resp.json()["detail"].lower())

            # Oversized PDF
            resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("slides.pdf", b"%PDF-1.4" + (b"x" * 100), "application/pdf"),
                    "recording": ("lecture.mp3", b"audio-bytes", "audio/mpeg"),
                },
            )
            self.assertEqual(resp.status_code, 413)

            # Oversized recording
            resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("slides.pdf", b"%PDF-1.4\nbody", "application/pdf"),
                    "recording": ("lecture.mp3", b"a" * 200, "audio/mpeg"),
                },
            )
            self.assertEqual(resp.status_code, 413)

        # Ensure failed uploads left no orphan job directories
        self.assertEqual(list(settings.jobs_dir.iterdir()), [])

    def test_unknown_token_returns_404(self) -> None:
        app = create_app(settings=_make_settings(self.tmp_dir))
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/jobs/nonexistent-token").status_code, 404)
            self.assertEqual(client.get("/api/jobs/nonexistent-token/download").status_code, 404)

    def test_upload_runs_background_stages_and_serves_markdown_download(self) -> None:
        gate = threading.Event()
        ocr_started = threading.Event()
        transcript_written = threading.Event()

        def gated_audio_prep(context: JobContext, _settings: Settings) -> StageResult:
            self.assertTrue(gate.wait(timeout=5.0))
            self.assertTrue(context.recording_path.is_file())
            return _stub_audio_prep(context, _settings)

        def transcription(context: JobContext, settings: Settings) -> StageResult:
            self.assertEqual(context.artifacts["audio_normalized"], ARTIFACT_AUDIO_NORMALIZED)
            result = _stub_transcription(context, settings)
            transcript_written.set()
            return result

        def gated_ocr(context: JobContext, settings: Settings) -> StageResult:
            ocr_started.set()
            self.assertTrue(transcript_written.wait(timeout=5.0))
            return _stub_ocr(context, settings)

        app = create_app(
            settings=_make_settings(self.tmp_dir),
            stage_handlers=_phase1_stub_handlers(
                audio_prep=gated_audio_prep, transcription=transcription, ocr=gated_ocr,
            ),
        )

        with TestClient(app) as client:
            create_resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("Week 02 Intro.pdf", b"%PDF-1.4\n%test", "application/pdf"),
                    "recording": ("recording.mp4", b"fake-mp4-data", "video/mp4"),
                },
            )
            self.assertEqual(create_resp.status_code, 202)
            created = create_resp.json()
            token = created["token"]
            self.assertGreaterEqual(len(token), 32)
            self.assertEqual(created["result_url"], f"/jobs/{token}")

            self.assertTrue(ocr_started.wait(timeout=5.0))
            in_progress = client.get(f"/api/jobs/{token}")
            self.assertEqual(in_progress.status_code, 200)
            in_progress_data = in_progress.json()
            self.assertEqual(in_progress_data["status"], "processing")
            self.assertEqual(in_progress_data["current_stage"], "audio_prep")
            running = {s["stage"] for s in in_progress_data["stages"] if s["status"] == "running"}
            self.assertEqual(running, {"ocr", "audio_prep"})
            self.assertFalse(in_progress_data["download_ready"])
            self.assertIsNone(in_progress_data["download_url"])

            # Downloading before completion returns 409 Conflict
            early_dl = client.get(f"/api/jobs/{token}/download")
            self.assertEqual(early_dl.status_code, 409)

            # Release worker and wait for completion
            gate.set()
            app.state.worker.wait_for_job(token, timeout=5.0)

            done_resp = client.get(f"/api/jobs/{token}")
            self.assertEqual(done_resp.status_code, 200)
            done = done_resp.json()
            self.assertEqual(done["status"], "completed")
            self.assertEqual(done["progress_pct"], 100)
            self.assertTrue(done["download_ready"])
            self.assertEqual(done["download_url"], f"/api/jobs/{token}/download")
            self.assertEqual(done["download_filename"], "week-02-intro-notes.md")
            self.assertIsNone(done["error"])
            self.assertEqual([s["stage"] for s in done["stages"]], list(PIPELINE_STAGES))
            self.assertTrue(all(s["status"] == "completed" for s in done["stages"]))

            # Download the generated Markdown
            dl_resp = client.get(done["download_url"])
            self.assertEqual(dl_resp.status_code, 200)
            self.assertIn("text/markdown", dl_resp.headers["content-type"])
            self.assertIn("week-02-intro-notes.md", dl_resp.headers["content-disposition"])
            self.assertIn("# Week 02 Intro", dl_resp.text)

    def test_stage_error_reports_exact_stage_and_actionable_hint(self) -> None:
        failure_saved = threading.Event()

        def gated_audio_prep(context: JobContext, settings: Settings) -> StageResult:
            self.assertTrue(failure_saved.wait(timeout=5.0))
            return _stub_audio_prep(context, settings)

        def failing_ocr(_context: JobContext, _settings: Settings) -> StageResult:
            raise StageError(
                stage="ocr",
                message="GEMINI_API_KEY is missing.",
                hint="Set GEMINI_API_KEY in .env and restart the backend.",
            )

        app = create_app(
            settings=_make_settings(self.tmp_dir),
            stage_handlers=_phase1_stub_handlers(audio_prep=gated_audio_prep, ocr=failing_ocr),
        )
        mark_failed = app.state.store.mark_job_failed

        def record_failure(*args, **kwargs):
            mark_failed(*args, **kwargs)
            failure_saved.set()

        with mock.patch.object(app.state.store, "mark_job_failed", record_failure), TestClient(app) as client:
            create_resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("lecture.pdf", b"%PDF-1.4\n%test", "application/pdf"),
                    "recording": ("lecture.wav", b"RIFFDATA", "audio/wav"),
                },
            )
            self.assertEqual(create_resp.status_code, 202)
            token = create_resp.json()["token"]
            app.state.worker.wait_for_job(token, timeout=5.0)

            status_resp = client.get(f"/api/jobs/{token}")
            self.assertEqual(status_resp.status_code, 200)
            data = status_resp.json()
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["current_stage"], "ocr")
            self.assertFalse(data["download_ready"])
            self.assertEqual(
                data["error"],
                {
                    "stage": "ocr",
                    "stage_label": "Slide OCR (Gemini)",
                    "message": "GEMINI_API_KEY is missing.",
                    "hint": "Set GEMINI_API_KEY in .env and restart the backend.",
                },
            )

            stages_by_name = {s["stage"]: s for s in data["stages"]}
            self.assertEqual(stages_by_name["upload"]["status"], "completed")
            self.assertEqual(stages_by_name["audio_prep"]["status"], "completed")
            self.assertEqual(stages_by_name["transcription"]["status"], "pending")
            self.assertEqual(stages_by_name["ocr"]["status"], "failed")
            self.assertEqual(stages_by_name["alignment"]["status"], "pending")
            self.assertEqual(stages_by_name["notes"]["status"], "pending")

            # Download on failed job returns 409
            dl_resp = client.get(f"/api/jobs/{token}/download")
            self.assertEqual(dl_resp.status_code, 409)

    def test_unexpected_stage_exception_fails_gracefully(self) -> None:
        def crashing_alignment(_context: JobContext, _settings: Settings) -> StageResult:
            raise ValueError("malformed payload")

        app = create_app(
            settings=_make_settings(self.tmp_dir),
            stage_handlers=_phase1_stub_handlers(alignment=crashing_alignment),
        )
        with TestClient(app) as client:
            create_resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("lecture.pdf", b"%PDF-1.4\n%test", "application/pdf"),
                    "recording": ("lecture.mp3", b"ID3DATA", "audio/mpeg"),
                },
            )
            self.assertEqual(create_resp.status_code, 202)
            token = create_resp.json()["token"]
            app.state.worker.wait_for_job(token, timeout=5.0)

            status_resp = client.get(f"/api/jobs/{token}")
            self.assertEqual(status_resp.status_code, 200)
            data = status_resp.json()
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["current_stage"], "alignment")
            self.assertEqual(data["error"]["stage"], "alignment")
            self.assertIn("malformed payload", data["error"]["message"])
            self.assertTrue(data["error"]["hint"])

    def test_missing_upload_field_returns_422(self) -> None:
        app = create_app(settings=_make_settings(self.tmp_dir))
        with TestClient(app) as client:
            resp = client.post(
                "/api/jobs",
                files={"slides": ("lecture.pdf", b"%PDF-1.4\n%test", "application/pdf")},
            )
            self.assertEqual(resp.status_code, 422)

    def test_restart_marks_interrupted_in_progress_jobs_clearly(self) -> None:
        settings = _make_settings(self.tmp_dir)
        settings.ensure_directories()
        store = JobStore(settings.db_path)
        store.init_db()

        # Simulate a job that was mid-transcription when the server died
        job_dir = settings.jobs_dir / "interrupted-token"
        (job_dir / "input").mkdir(parents=True, exist_ok=True)
        store.create_job(
            slides_filename="slides.pdf",
            recording_filename="audio.mp3",
            job_dir=job_dir,
            artifacts={"slides_pdf": "input/slides.pdf", "recording": "input/recording.mp3"},
            token="interrupted-token",
        )
        store.mark_job_started("interrupted-token")
        store.complete_stage("interrupted-token", "audio_prep", message="Done")
        store.start_stage("interrupted-token", "transcription", message="Transcribing...")
        store.start_stage("interrupted-token", "ocr", message="Reading slides...")

        # This queued job already has a saved OCR checkpoint.
        queued_dir = settings.jobs_dir / "queued-token"
        (queued_dir / "input").mkdir(parents=True, exist_ok=True)
        store.create_job(
            slides_filename="slides2.pdf",
            recording_filename="audio2.mp3",
            job_dir=queued_dir,
            artifacts={"slides_pdf": "input/slides.pdf", "recording": "input/recording.mp3"},
            token="queued-token",
        )
        (queued_dir / "artifacts").mkdir()
        (queued_dir / ARTIFACT_SLIDES_OCR).write_text(
            SlidesOCRArtifact(source_pdf="slides2.pdf").model_dump_json(), encoding="utf-8",
        )
        store.complete_stage("queued-token", "ocr", artifacts={"slides_ocr": ARTIFACT_SLIDES_OCR})
        unexpected_ocr = mock.Mock(side_effect=AssertionError("Completed OCR ran twice"))

        # Booting a fresh FastAPI app runs lifespan recovery
        app = create_app(settings=settings, stage_handlers=_phase1_stub_handlers(ocr=unexpected_ocr))
        with TestClient(app) as client:
            resp = client.get("/api/jobs/interrupted-token")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["current_stage"], "transcription")
            self.assertEqual(data["error"]["stage"], "transcription")
            self.assertEqual(data["error"]["message"], INTERRUPTED_MESSAGE)
            self.assertEqual(data["error"]["hint"], INTERRUPTED_HINT)
            failed = {s["stage"] for s in data["stages"] if s["status"] == "failed"}
            self.assertEqual(failed, {"transcription", "ocr"})

            app.state.worker.wait_for_job("queued-token", timeout=5.0)
            queued_resp = client.get("/api/jobs/queued-token")
            self.assertEqual(queued_resp.status_code, 200)
            queued_data = queued_resp.json()
            self.assertEqual(queued_data["status"], "completed", queued_data["error"])
            unexpected_ocr.assert_not_called()

    def test_sub_stage_progress_stats_warnings_and_preview(self) -> None:
        progress_written = threading.Event()
        release_stage = threading.Event()

        def chunked_transcription(context: JobContext, settings: Settings) -> StageResult:
            context.report_progress("Transcribing chunk 2 of 4...")
            progress_written.set()
            release_stage.wait(timeout=5.0)
            artifact = TranscriptArtifact(
                source_recording=context.recording_filename,
                model=settings.transcription_model,
                duration_seconds=3612.5,
                segments=[],
            )
            context.transcript_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
            return StageResult(
                message="Transcribed 4 chunks.",
                artifacts={"transcript": ARTIFACT_TRANSCRIPT},
                stats={"duration_seconds": 3612.5, "transcript_segments": 420},
                warnings=["Chunk 3 retried once after rate limit."],
            )

        def alignment_with_warning(context: JobContext, settings: Settings) -> StageResult:
            artifact = AlignmentArtifact(mode=settings.pipeline_llm_mode, entries=[], alignment_issues=[])
            context.alignment_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
            return StageResult(
                message="Aligned with 1 warning.",
                artifacts={"alignment": ARTIFACT_ALIGNMENT},
                warnings=[
                    "Chunk 3 retried once after rate limit.",  # duplicate should be deduplicated
                    "Slide 12 alignment uncertain; falling back to timestamps.",
                ],
            )

        app = create_app(
            settings=_make_settings(self.tmp_dir),
            stage_handlers=_phase1_stub_handlers(
                transcription=chunked_transcription,
                alignment=alignment_with_warning,
            ),
        )

        with TestClient(app) as client:
            resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("Distributed Systems.pdf", b"%PDF-1.7\nbody", "application/pdf"),
                    "recording": ("lecture.m4a", b"audio-bytes", "audio/mp4"),
                },
            )
            self.assertEqual(resp.status_code, 202)
            token = resp.json()["token"]

            self.assertTrue(progress_written.wait(timeout=5.0))
            live_status = client.get(f"/api/jobs/{token}").json()
            stages_by_name = {s["stage"]: s for s in live_status["stages"]}
            self.assertEqual(stages_by_name["transcription"]["status"], "running")
            self.assertEqual(
                stages_by_name["transcription"]["message"],
                "Transcribing chunk 2 of 4...",
            )
            self.assertEqual(client.get(f"/api/jobs/{token}/preview").status_code, 409)

            release_stage.set()
            app.state.worker.wait_for_job(token, timeout=5.0)

            done = client.get(f"/api/jobs/{token}").json()
            self.assertEqual(done["status"], "completed")
            self.assertEqual(done["stats"]["duration_seconds"], 3612.5)
            self.assertEqual(done["stats"]["transcript_segments"], 420)
            self.assertEqual(done["stats"]["slide_count"], 0)
            self.assertGreater(done["stats"]["word_count"], 0)
            self.assertEqual(
                done["warnings"],
                [
                    "Chunk 3 retried once after rate limit.",
                    "Slide 12 alignment uncertain; falling back to timestamps.",
                ],
            )

            preview_resp = client.get(f"/api/jobs/{token}/preview")
            self.assertEqual(preview_resp.status_code, 200)
            preview = preview_resp.json()
            self.assertEqual(preview["token"], token)
            self.assertEqual(preview["filename"], "distributed-systems-notes.md")
            self.assertIn("# Distributed Systems", preview["content"])
            self.assertEqual(preview["word_count"], done["stats"]["word_count"])
            self.assertFalse(preview["truncated"])

    def test_pdf_header_allowed_within_first_1024_bytes(self) -> None:
        app = create_app(
            settings=_make_settings(self.tmp_dir),
            stage_handlers=_phase1_stub_handlers(),
        )
        with TestClient(app) as client:
            # Valid PDF with leading preamble bytes (< 1024 bytes before %PDF-)
            ok_resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("preamble.pdf", (b" " * 64) + b"%PDF-1.7\nbody", "application/pdf"),
                    "recording": ("rec.mp3", b"ID3", "audio/mpeg"),
                },
            )
            self.assertEqual(ok_resp.status_code, 202)
            app.state.worker.wait_for_job(ok_resp.json()["token"], timeout=5.0)

            # Invalid PDF where %PDF- only appears after the 1024-byte spec window
            bad_resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("late.pdf", (b"x" * 1024) + b"%PDF-1.7", "application/pdf"),
                    "recording": ("rec.mp3", b"ID3", "audio/mpeg"),
                },
            )
            self.assertEqual(bad_resp.status_code, 400)

    def test_sqlite_schema_migrates_legacy_v1_database(self) -> None:
        db_path = self.tmp_dir / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE jobs (
                token TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                current_stage TEXT NOT NULL,
                slides_filename TEXT NOT NULL,
                recording_filename TEXT NOT NULL,
                job_dir TEXT NOT NULL,
                artifacts_json TEXT NOT NULL DEFAULT '{}',
                error_stage TEXT,
                error_message TEXT,
                error_hint TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE TABLE job_stages (
                job_token TEXT NOT NULL,
                stage TEXT NOT NULL,
                stage_order INTEGER NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                started_at TEXT,
                completed_at TEXT,
                PRIMARY KEY (job_token, stage)
            );
            INSERT INTO jobs (
                token, status, current_stage, slides_filename, recording_filename,
                job_dir, artifacts_json, created_at, updated_at
            ) VALUES (
                'legacy-tok', 'completed', 'notes', 'old.pdf', 'old.mp3',
                '/tmp/old', '{}', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
            );
            PRAGMA user_version = 1;
            """
        )
        conn.commit()
        conn.close()

        store = JobStore(db_path)
        store.init_db()

        verify_conn = sqlite3.connect(str(db_path))
        version = verify_conn.execute("PRAGMA user_version").fetchone()[0]
        verify_conn.close()
        self.assertEqual(version, SCHEMA_VERSION)

        job = store.get_job("legacy-tok")
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job["stats"], {})
        self.assertEqual(job["warnings"], [])

    def test_load_settings_parses_vertex_and_multi_origin_cors(self) -> None:
        env_patch = {
            "DATA_DIR": str(self.tmp_dir / "custom-data"),
            "FRONTEND_ORIGIN": "http://localhost:5173, http://localhost:4173",
            "GOOGLE_GENAI_USE_VERTEXAI": "true",
            "GOOGLE_CLOUD_PROJECT": "my-gcp-project",
            "GOOGLE_CLOUD_LOCATION": "europe-west4",
            "OCR_MAX_PAGES_PER_BATCH": "12",
            "OCR_MAX_BATCH_BYTES": "10485760",
            "OCR_PARALLEL_WORKERS": "3",
            "OCR_PARALLEL_MIN_PAGES": "6",
            "OCR_GLOBAL_MAX_CONCURRENT": "5",
            "OCR_RETRY_ATTEMPTS": "4",
            "OCR_RETRY_BASE_DELAY_SECONDS": "1.5",
            "OCR_TIMEOUT_SECONDS": "120",
            "OCR_MAX_OUTPUT_TOKENS": "16384",
        }
        with mock.patch.dict(os.environ, env_patch, clear=False):
            cfg = load_settings(env_file=self.tmp_dir / "nonexistent.env")

        self.assertTrue(cfg.google_genai_use_vertexai)
        self.assertEqual(cfg.google_cloud_project, "my-gcp-project")
        self.assertEqual(cfg.google_cloud_location, "europe-west4")
        self.assertEqual(cfg.ocr_max_pages_per_batch, 12)
        self.assertEqual(cfg.ocr_max_batch_bytes, 10_485_760)
        self.assertEqual(cfg.ocr_parallel_workers, 3)
        self.assertEqual(cfg.ocr_parallel_min_pages, 6)
        self.assertEqual(cfg.ocr_global_max_concurrent, 5)
        self.assertEqual(cfg.ocr_retry_attempts, 4)
        self.assertEqual(cfg.ocr_retry_base_delay_seconds, 1.5)
        self.assertEqual(cfg.ocr_timeout_seconds, 120.0)
        self.assertEqual(cfg.ocr_max_output_tokens, 16384)
        self.assertIn("http://localhost:5173", cfg.frontend_origins)
        self.assertIn("http://127.0.0.1:5173", cfg.frontend_origins)
        self.assertIn("http://localhost:4173", cfg.frontend_origins)
        self.assertIn("http://127.0.0.1:4173", cfg.frontend_origins)

    def test_full_pipeline_end_to_end_via_api_in_stub_mode(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            groq_api_key="gsk-test-e2e",
            gemini_api_key="AIza-test-e2e",
            pipeline_llm_mode="stub",
        )
        pdf_bytes = _build_pdf_bytes(
            [
                "Lecture 4: Hash Tables",
                "Collision Resolution: Chaining vs Open Addressing",
            ]
        )

        def fake_normalize(_src, output_path: Path, **_kwargs) -> Path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"ID3-normalized-audio")
            return output_path

        fake_groq_client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(
                        segments=[
                            SimpleNamespace(
                                start=0.0,
                                end=18.5,
                                text="Welcome to Lecture 4 on hash tables and constant-time lookup.",
                            ),
                            SimpleNamespace(
                                start=18.5,
                                end=42.0,
                                text="When two keys collide, chaining stores a bucket list while open addressing probes.",
                            ),
                        ]
                    )
                )
            )
        )
        fake_gemini_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_kwargs: SimpleNamespace(
                    text=json.dumps(
                        [
                            {
                                "slide": 1,
                                "text": (
                                    "VISIBLE_TEXT:\nLecture 4: Hash Tables\nO(1) average lookup\n\n"
                                    "VISUAL_CONTEXT:\n- Diagram mapping keys through h(k) into bucket array."
                                ),
                            },
                            {
                                "slide": 2,
                                "text": (
                                    "VISIBLE_TEXT:\nCollision Resolution\nChaining vs Open Addressing\n\n"
                                    "TABLES_CODE_FORMULAS:\nload_factor = n / m"
                                ),
                            },
                        ]
                    ),
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=320,
                        candidates_token_count=110,
                    ),
                )
            )
        )

        # No stage_handlers override: exercises the full default wired pipeline end-to-end.
        app = create_app(settings=settings)
        with (
            mock.patch(
                "backend.app.pipeline.audio.resolve_ffmpeg_binaries",
                return_value=FFmpegBinaries(ffmpeg="ffmpeg", ffprobe="ffprobe"),
            ),
            mock.patch(
                "backend.app.pipeline.audio.normalize_audio_to_mp3",
                side_effect=fake_normalize,
            ),
            mock.patch(
                "backend.app.pipeline.audio.probe_audio_duration",
                return_value=42.0,
            ),
            mock.patch(
                "backend.app.pipeline.transcribe._default_groq_client_factory",
                return_value=fake_groq_client,
            ),
            mock.patch(
                "backend.app.pipeline.transcribe._try_probe_duration_best_effort",
                return_value=42.0,
            ),
            mock.patch(
                "backend.app.pipeline.ocr._default_genai_client_factory",
                return_value=fake_gemini_client,
            ),
        ):
            with TestClient(app) as client:
                create_resp = client.post(
                    "/api/jobs",
                    files={
                        "slides": ("Week 04 Hash Tables.pdf", pdf_bytes, "application/pdf"),
                        "recording": ("lecture.mp4", b"fake-mp4-video", "video/mp4"),
                    },
                )
                self.assertEqual(create_resp.status_code, 202)
                token = create_resp.json()["token"]
                app.state.worker.wait_for_job(token, timeout=10.0)

                status_resp = client.get(f"/api/jobs/{token}")
                self.assertEqual(status_resp.status_code, 200)
                status_data = status_resp.json()
                self.assertEqual(status_data["status"], "completed")
                self.assertEqual(status_data["progress_pct"], 100)
                self.assertTrue(status_data["download_ready"])
                self.assertIsNone(status_data["error"])
                self.assertTrue(all(s["status"] == "completed" for s in status_data["stages"]))

                self.assertEqual(status_data["artifacts"]["audio_normalized"], ARTIFACT_AUDIO_NORMALIZED)
                self.assertEqual(status_data["artifacts"]["transcript"], ARTIFACT_TRANSCRIPT)
                self.assertEqual(status_data["artifacts"]["slides_ocr"], ARTIFACT_SLIDES_OCR)
                self.assertEqual(status_data["artifacts"]["alignment"], ARTIFACT_ALIGNMENT)
                self.assertEqual(status_data["artifacts"]["notes_md"], ARTIFACT_NOTES_MD)

                self.assertEqual(status_data["stats"]["slide_count"], 2)
                self.assertEqual(status_data["stats"]["transcript_segments"], 2)
                self.assertEqual(status_data["stats"]["duration_seconds"], 42.0)
                self.assertGreaterEqual(status_data["stats"]["section_count"], 1)
                self.assertGreater(status_data["stats"]["word_count"], 20)

                dl_resp = client.get(f"/api/jobs/{token}/download")
                self.assertEqual(dl_resp.status_code, 200)
                self.assertIn("Hash Tables", dl_resp.text)
                self.assertIn("Diagram mapping keys through h(k)", dl_resp.text)
                self.assertIn("load_factor = n / m", dl_resp.text)
                self.assertIn("open addressing probes", dl_resp.text)

                preview_resp = client.get(f"/api/jobs/{token}/preview")
                self.assertEqual(preview_resp.status_code, 200)
                preview_data = preview_resp.json()
                self.assertEqual(preview_data["filename"], "week-04-hash-tables-notes.md")
                self.assertEqual(
                    preview_data["content"],
                    dl_resp.text.replace("\r\n", "\n"),
                )

    def test_strict_ocr_failure_via_api_when_gemini_credentials_missing(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            groq_api_key="gsk-test-strict-ocr",
            gemini_api_key="",
            google_genai_use_vertexai=False,
        )
        pdf_bytes = _build_pdf_bytes(["Slide text that must never leak via pdfplumber fallback"])

        def fake_normalize(_src, output_path: Path, **_kwargs) -> Path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"ID3-normalized-audio")
            return output_path

        fake_groq_client = SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(
                    create=lambda **_kwargs: SimpleNamespace(
                        segments=[SimpleNamespace(start=0.0, end=3.0, text="Hello class.")]
                    )
                )
            )
        )

        # Audio work may be in flight when the OCR credential check fails.
        app = create_app(settings=settings)
        with (
            mock.patch(
                "backend.app.pipeline.audio.resolve_ffmpeg_binaries",
                return_value=FFmpegBinaries(ffmpeg="ffmpeg", ffprobe="ffprobe"),
            ),
            mock.patch(
                "backend.app.pipeline.audio.normalize_audio_to_mp3",
                side_effect=fake_normalize,
            ),
            mock.patch(
                "backend.app.pipeline.audio.probe_audio_duration",
                return_value=3.0,
            ),
            mock.patch(
                "backend.app.pipeline.transcribe._default_groq_client_factory",
                return_value=fake_groq_client,
            ),
            mock.patch(
                "backend.app.pipeline.transcribe._try_probe_duration_best_effort",
                return_value=3.0,
            ),
        ):
            with TestClient(app) as client:
                create_resp = client.post(
                    "/api/jobs",
                    files={
                        "slides": ("strict.pdf", pdf_bytes, "application/pdf"),
                        "recording": ("lecture.mp3", b"ID3-bytes", "audio/mpeg"),
                    },
                )
                self.assertEqual(create_resp.status_code, 202)
                token = create_resp.json()["token"]
                app.state.worker.wait_for_job(token, timeout=10.0)

                status_resp = client.get(f"/api/jobs/{token}")
                self.assertEqual(status_resp.status_code, 200)
                data = status_resp.json()
                self.assertEqual(data["status"], "failed")
                self.assertEqual(data["current_stage"], "ocr")
                self.assertFalse(data["download_ready"])
                self.assertEqual(data["error"]["stage"], "ocr")
                self.assertEqual(data["error"]["stage_label"], "Slide OCR (Gemini)")
                self.assertIn("GEMINI_API_KEY", data["error"]["message"])
                self.assertIn("GEMINI_API_KEY", data["error"]["hint"])

                stages_by_name = {s["stage"]: s["status"] for s in data["stages"]}
                self.assertEqual(stages_by_name["audio_prep"], "completed")
                self.assertIn(stages_by_name["transcription"], ("pending", "completed"))
                self.assertEqual(stages_by_name["ocr"], "failed")
                self.assertEqual(stages_by_name["alignment"], "pending")
                self.assertEqual(stages_by_name["notes"], "pending")

                # Confirm zero fallback artifact was created on disk.
                slides_ocr_file = settings.jobs_dir / token / ARTIFACT_SLIDES_OCR
                self.assertFalse(slides_ocr_file.exists())
                self.assertEqual(client.get(f"/api/jobs/{token}/download").status_code, 409)


if __name__ == "__main__":
    unittest.main()
