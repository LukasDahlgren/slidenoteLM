from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.db import INTERRUPTED_HINT, INTERRUPTED_MESSAGE, JobStore, generate_job_token
from backend.app.main import build_download_filename, create_app
from backend.app.schemas import PIPELINE_STAGES, JobContext, StageError, StageResult


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
        stage_started = threading.Event()

        def gated_audio_prep(context: JobContext, _settings: Settings) -> StageResult:
            stage_started.set()
            gate.wait(timeout=5.0)
            self.assertTrue(context.recording_path.is_file())
            return StageResult(message="Audio normalized.")

        app = create_app(
            settings=_make_settings(self.tmp_dir),
            stage_handlers={"audio_prep": gated_audio_prep},
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

            # Wait until worker enters audio_prep so we can inspect in-progress state
            self.assertTrue(stage_started.wait(timeout=5.0))
            in_progress = client.get(f"/api/jobs/{token}")
            self.assertEqual(in_progress.status_code, 200)
            in_progress_data = in_progress.json()
            self.assertEqual(in_progress_data["status"], "processing")
            self.assertEqual(in_progress_data["current_stage"], "audio_prep")
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
        def failing_ocr(_context: JobContext, _settings: Settings) -> StageResult:
            raise StageError(
                stage="ocr",
                message="GEMINI_API_KEY is missing.",
                hint="Set GEMINI_API_KEY in .env and restart the backend.",
            )

        app = create_app(
            settings=_make_settings(self.tmp_dir),
            stage_handlers={"ocr": failing_ocr},
        )
        with TestClient(app) as client:
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
            self.assertEqual(stages_by_name["transcription"]["status"], "completed")
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
            stage_handlers={"alignment": crashing_alignment},
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

        # Also simulate a queued job that hadn't started any background stage yet
        queued_dir = settings.jobs_dir / "queued-token"
        (queued_dir / "input").mkdir(parents=True, exist_ok=True)
        store.create_job(
            slides_filename="slides2.pdf",
            recording_filename="audio2.mp3",
            job_dir=queued_dir,
            artifacts={"slides_pdf": "input/slides.pdf", "recording": "input/recording.mp3"},
            token="queued-token",
        )

        # Booting a fresh FastAPI app runs lifespan recovery
        app = create_app(settings=settings)
        with TestClient(app) as client:
            resp = client.get("/api/jobs/interrupted-token")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["current_stage"], "transcription")
            self.assertEqual(data["error"]["stage"], "transcription")
            self.assertEqual(data["error"]["message"], INTERRUPTED_MESSAGE)
            self.assertEqual(data["error"]["hint"], INTERRUPTED_HINT)

            queued_resp = client.get("/api/jobs/queued-token")
            self.assertEqual(queued_resp.status_code, 200)
            queued_data = queued_resp.json()
            self.assertEqual(queued_data["status"], "failed")
            self.assertEqual(queued_data["current_stage"], "audio_prep")
            self.assertEqual(queued_data["error"]["stage"], "audio_prep")
            self.assertEqual(queued_data["error"]["message"], INTERRUPTED_MESSAGE)


if __name__ == "__main__":
    unittest.main()
