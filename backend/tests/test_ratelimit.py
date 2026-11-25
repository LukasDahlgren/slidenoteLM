from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pymupdf
from fastapi.testclient import TestClient

from backend.app.config import Settings, load_settings
from backend.app.main import create_app
from backend.app.pipeline import StageHandler
from backend.app.pipeline.ocr import extract_slides_with_gemini_ocr
from backend.app.pipeline.transcribe import transcribe_audio_file
from backend.app.ratelimit import (
    OutboundRateLimiter,
    RateLimitRule,
    SlidingWindowRateLimiter,
    extract_client_ip,
    resolve_api_rate_limit_rule,
)
from backend.app.schemas import (
    ARTIFACT_SLIDES_OCR,
    ARTIFACT_TRANSCRIPT,
    JobContext,
    PipelineStage,
    SlidesOCRArtifact,
    StageResult,
    TranscriptArtifact,
)


class _FakeClock:
    def __init__(self, initial: float = 1000.0) -> None:
        self.now = initial

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_settings(tmp_dir: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "data_dir": tmp_dir,
        "db_path": tmp_dir / "slidenotelm.db",
        "jobs_dir": tmp_dir / "jobs",
        "pipeline_llm_mode": "stub",
        "worker_concurrency": 1,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def _stub_handlers() -> dict[PipelineStage, StageHandler]:
    def _audio(context: JobContext, _settings: Settings) -> StageResult:
        context.audio_normalized_path.write_bytes(b"stub-mp3")
        return StageResult(
            message="Audio prep stub completed.",
            artifacts={"audio_normalized": "artifacts/audio_normalized.mp3"},
        )

    def _transcribe(context: JobContext, settings: Settings) -> StageResult:
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

    def _ocr(context: JobContext, settings: Settings) -> StageResult:
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

    return {
        "audio_prep": _audio,
        "transcription": _transcribe,
        "ocr": _ocr,
    }


def _create_sample_pdf(path: Path, pages: int) -> Path:
    doc = pymupdf.open()
    try:
        for idx in range(1, pages + 1):
            page = doc.new_page(width=640, height=360)
            page.insert_text((48, 72), f"Slide {idx}", fontsize=16)
        doc.save(str(path))
    finally:
        doc.close()
    return path


class RateLimiterUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-rl-unit-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_sliding_window_enforces_limit_resets_and_isolates_clients(self) -> None:
        clock = _FakeClock(100.0)
        limiter = SlidingWindowRateLimiter(clock=clock)
        rule = RateLimitRule(name="auth_login", limit=2, window_seconds=60)

        d1 = limiter.check(rule, "203.0.113.10")
        self.assertTrue(d1.allowed)
        self.assertEqual(d1.remaining, 1)

        clock.advance(10.0)
        d2 = limiter.check(rule, "203.0.113.10")
        self.assertTrue(d2.allowed)
        self.assertEqual(d2.remaining, 0)

        # Third request within 60s window is rejected with Retry-After ~50s
        d3 = limiter.check(rule, "203.0.113.10")
        self.assertFalse(d3.allowed)
        self.assertEqual(d3.remaining, 0)
        self.assertEqual(d3.retry_after_seconds, 50)

        # Another IP is unaffected
        other = limiter.check(rule, "203.0.113.11")
        self.assertTrue(other.allowed)
        self.assertEqual(other.remaining, 1)

        # Once the first request ages out of the 60s window, another slot opens
        clock.advance(50.1)
        d4 = limiter.check(rule, "203.0.113.10")
        self.assertTrue(d4.allowed)
        self.assertEqual(d4.remaining, 0)

        # Disabled rule (limit=0) always allows
        unlimited = limiter.check(RateLimitRule("open", limit=0, window_seconds=60), "203.0.113.10")
        self.assertTrue(unlimited.allowed)

    def test_outbound_rate_limiter_paces_calls_when_window_full(self) -> None:
        clock = _FakeClock(500.0)
        limiter = OutboundRateLimiter(max_requests=2, window_seconds=60.0, clock=clock)
        slept: list[float] = []
        notified: list[float] = []

        def fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            clock.advance(seconds)

        self.assertEqual(limiter.acquire(sleep_fn=fake_sleep, on_wait=notified.append), 0.0)
        clock.advance(15.0)
        self.assertEqual(limiter.acquire(sleep_fn=fake_sleep, on_wait=notified.append), 0.0)

        # 3rd call must wait 45s until the first timestamp exits the 60s window
        waited = limiter.acquire(sleep_fn=fake_sleep, on_wait=notified.append)
        self.assertAlmostEqual(waited, 45.0, places=3)
        self.assertEqual(len(slept), 1)
        self.assertEqual(len(notified), 1)

        # Disabled outbound limiter never sleeps
        disabled = OutboundRateLimiter(max_requests=0, window_seconds=60.0, clock=clock)
        self.assertEqual(disabled.acquire(sleep_fn=fake_sleep), 0.0)
        self.assertEqual(len(slept), 1)

    def test_extract_client_ip_and_route_rule_resolution(self) -> None:
        scope_forwarded = {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"198.51.100.42, 10.0.0.1")],
            "client": ("127.0.0.1", 5555),
        }
        self.assertEqual(extract_client_ip(scope_forwarded, trust_proxy_headers=True), "198.51.100.42")
        self.assertEqual(extract_client_ip(scope_forwarded, trust_proxy_headers=False), "127.0.0.1")

        scope_real_ip = {
            "type": "http",
            "headers": [(b"x-real-ip", b"198.51.100.99")],
            "client": ("127.0.0.1", 5555),
        }
        self.assertEqual(extract_client_ip(scope_real_ip, trust_proxy_headers=True), "198.51.100.99")

        settings = _make_settings(
            self.tmp_dir,
            rate_limit_window_seconds=45,
            rate_limit_auth_login_per_window=5,
            rate_limit_auth_per_window=30,
            rate_limit_jobs_create_per_window=4,
            rate_limit_jobs_read_per_window=90,
            rate_limit_health_per_window=100,
            rate_limit_default_per_window=80,
        )

        self.assertIsNone(resolve_api_rate_limit_rule("OPTIONS", "/api/jobs", settings))
        self.assertIsNone(resolve_api_rate_limit_rule("GET", "/", settings))
        self.assertIsNone(resolve_api_rate_limit_rule("GET", "/assets/app.js", settings))

        self.assertEqual(
            resolve_api_rate_limit_rule("GET", "/api/health", settings),
            RateLimitRule("health", 100, 45),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("POST", "/api/auth/login", settings),
            RateLimitRule("auth_login", 5, 45),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("GET", "/api/auth/status", settings),
            RateLimitRule("auth", 30, 45),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("POST", "/api/auth/logout", settings),
            RateLimitRule("auth", 30, 45),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("POST", "/api/jobs", settings),
            RateLimitRule("jobs_create", 4, 45),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("GET", "/api/jobs/tok_123", settings),
            RateLimitRule("jobs_read", 90, 45),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("GET", "/api/jobs/tok_123/download", settings),
            RateLimitRule("jobs_read", 90, 45),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("GET", "/api/jobs/tok_123/preview", settings),
            RateLimitRule("jobs_read", 90, 45),
        )
        self.assertEqual(
            resolve_api_rate_limit_rule("GET", "/api/unknown", settings),
            RateLimitRule("api_default", 80, 45),
        )

    def test_load_settings_parses_rate_limit_env_vars(self) -> None:
        env_patch = {
            "RATE_LIMIT_ENABLED": "false",
            "RATE_LIMIT_WINDOW_SECONDS": "30",
            "RATE_LIMIT_TRUST_PROXY_HEADERS": "false",
            "RATE_LIMIT_AUTH_LOGIN_PER_MINUTE": "5",
            "RATE_LIMIT_AUTH_PER_MINUTE": "25",
            "RATE_LIMIT_JOBS_CREATE_PER_MINUTE": "3",
            "RATE_LIMIT_JOBS_READ_PER_MINUTE": "150",
            "RATE_LIMIT_HEALTH_PER_MINUTE": "0",
            "RATE_LIMIT_DEFAULT_PER_MINUTE": "75",
            "TRANSCRIBE_RATE_LIMIT_RPM": "12",
            "OCR_RATE_LIMIT_RPM": "9",
        }
        with mock.patch.dict(os.environ, env_patch, clear=False):
            cfg = load_settings(env_file=self.tmp_dir / "missing.env")

        self.assertFalse(cfg.rate_limit_enabled)
        self.assertEqual(cfg.rate_limit_window_seconds, 30)
        self.assertFalse(cfg.rate_limit_trust_proxy_headers)
        self.assertEqual(cfg.rate_limit_auth_login_per_window, 5)
        self.assertEqual(cfg.rate_limit_auth_per_window, 25)
        self.assertEqual(cfg.rate_limit_jobs_create_per_window, 3)
        self.assertEqual(cfg.rate_limit_jobs_read_per_window, 150)
        self.assertEqual(cfg.rate_limit_health_per_window, 0)
        self.assertEqual(cfg.rate_limit_default_per_window, 75)
        self.assertEqual(cfg.transcribe_rate_limit_rpm, 12)
        self.assertEqual(cfg.ocr_rate_limit_rpm, 9)


class RateLimitHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-rl-http-"))
        self.dist_dir = self.tmp_dir / "dist"
        (self.dist_dir / "assets").mkdir(parents=True)
        (self.dist_dir / "index.html").write_text("<html>SPA</html>", encoding="utf-8")
        (self.dist_dir / "assets" / "app.js").write_text("console.log(1);", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_auth_login_rate_limit_returns_429_and_recovers_after_window(self) -> None:
        clock = _FakeClock(1000.0)
        limiter = SlidingWindowRateLimiter(clock=clock)
        settings = _make_settings(
            self.tmp_dir,
            app_password="correct-password",
            rate_limit_window_seconds=60,
            rate_limit_auth_login_per_window=2,
        )
        app = create_app(
            settings=settings,
            stage_handlers=_stub_handlers(),
            frontend_dist_dir=self.dist_dir,
            rate_limiter=limiter,
        )

        with TestClient(app) as client:
            r1 = client.post("/api/auth/login", json={"password": "wrong"})
            self.assertEqual(r1.status_code, 401)
            self.assertEqual(r1.headers["x-ratelimit-limit"], "2")
            self.assertEqual(r1.headers["x-ratelimit-remaining"], "1")
            self.assertIn("x-ratelimit-reset", r1.headers)

            r2 = client.post("/api/auth/login", json={"password": "wrong"})
            self.assertEqual(r2.status_code, 401)
            self.assertEqual(r2.headers["x-ratelimit-remaining"], "0")

            # 3rd login attempt is blocked with 429 + Retry-After
            r3 = client.post(
                "/api/auth/login",
                json={"password": "correct-password"},
                headers={"Origin": "http://localhost:5173"},
            )
            self.assertEqual(r3.status_code, 429)
            self.assertEqual(r3.headers["retry-after"], "60")
            self.assertEqual(r3.headers["x-ratelimit-remaining"], "0")
            self.assertEqual(r3.headers["access-control-allow-origin"], "http://localhost:5173")
            self.assertIn("Retry-After", r3.headers["access-control-expose-headers"])
            self.assertIn("Rate limit exceeded", r3.json()["detail"])

            # Different client IP can still log in
            other_ip = client.post(
                "/api/auth/login",
                json={"password": "correct-password"},
                headers={"X-Forwarded-For": "203.0.113.55"},
            )
            self.assertEqual(other_ip.status_code, 200)

            # After the window elapses, original IP can log in again
            clock.advance(60.1)
            r4 = client.post("/api/auth/login", json={"password": "correct-password"})
            self.assertEqual(r4.status_code, 200)

    def test_jobs_create_read_health_and_spa_rate_limits(self) -> None:
        clock = _FakeClock(2000.0)
        limiter = SlidingWindowRateLimiter(clock=clock)
        settings = _make_settings(
            self.tmp_dir,
            rate_limit_window_seconds=60,
            rate_limit_jobs_create_per_window=1,
            rate_limit_jobs_read_per_window=2,
            rate_limit_health_per_window=2,
            rate_limit_default_per_window=1,
        )
        app = create_app(
            settings=settings,
            stage_handlers=_stub_handlers(),
            frontend_dist_dir=self.dist_dir,
            rate_limiter=limiter,
        )

        with TestClient(app) as client:
            # First upload succeeds
            upload1 = client.post(
                "/api/jobs",
                files={
                    "slides": ("lec.pdf", b"%PDF-1.4\nbody", "application/pdf"),
                    "recording": ("lec.mp3", b"ID3-audio", "audio/mpeg"),
                },
            )
            self.assertEqual(upload1.status_code, 202)
            self.assertEqual(upload1.headers["x-ratelimit-limit"], "1")
            self.assertEqual(upload1.headers["x-ratelimit-remaining"], "0")
            token = upload1.json()["token"]
            app.state.worker.wait_for_job(token, timeout=5.0)

            # Second upload immediately hits 429
            upload2 = client.post(
                "/api/jobs",
                files={
                    "slides": ("lec2.pdf", b"%PDF-1.4\nbody", "application/pdf"),
                    "recording": ("lec2.mp3", b"ID3-audio", "audio/mpeg"),
                },
            )
            self.assertEqual(upload2.status_code, 429)

            # Jobs read bucket covers status, preview, and download
            self.assertEqual(client.get(f"/api/jobs/{token}").status_code, 200)
            self.assertEqual(client.get(f"/api/jobs/{token}/download").status_code, 200)
            self.assertEqual(client.get(f"/api/jobs/{token}/preview").status_code, 429)

            # Health endpoint uses its own bucket
            self.assertEqual(client.get("/api/health").status_code, 200)
            self.assertEqual(client.get("/api/health").status_code, 200)
            self.assertEqual(client.get("/api/health").status_code, 429)

            # Unknown /api/* routes are also rate-limited by api_default
            self.assertEqual(client.get("/api/does-not-exist").status_code, 404)
            self.assertEqual(client.get("/api/does-not-exist").status_code, 429)

            # Static SPA routes are NOT rate-limited and do not emit X-RateLimit headers
            for _ in range(5):
                spa_resp = client.get("/")
                self.assertEqual(spa_resp.status_code, 200)
                self.assertNotIn("x-ratelimit-limit", spa_resp.headers)
                asset_resp = client.get("/assets/app.js")
                self.assertEqual(asset_resp.status_code, 200)

    def test_rate_limit_disabled_flag_allows_unlimited_requests(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            rate_limit_enabled=False,
            rate_limit_health_per_window=1,
        )
        app = create_app(settings=settings, stage_handlers=_stub_handlers())
        with TestClient(app) as client:
            for _ in range(5):
                resp = client.get("/api/health")
                self.assertEqual(resp.status_code, 200)
                self.assertNotIn("x-ratelimit-limit", resp.headers)

    def test_unauthenticated_api_flood_is_rate_limited_before_auth_middleware(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            app_password="secret",
            rate_limit_jobs_read_per_window=2,
        )
        app = create_app(settings=settings, stage_handlers=_stub_handlers())
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/jobs/tok_any").status_code, 401)
            self.assertEqual(client.get("/api/jobs/tok_any").status_code, 401)
            self.assertEqual(client.get("/api/jobs/tok_any").status_code, 429)


class OutboundPipelineRateLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-rl-outbound-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_transcribe_audio_file_applies_outbound_rate_limiter(self) -> None:
        clock = _FakeClock(1000.0)
        limiter = OutboundRateLimiter(max_requests=1, window_seconds=60.0, clock=clock)
        sleeps: list[float] = []
        progress: list[str] = []

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock.advance(seconds)

        mp3_path = self.tmp_dir / "audio.mp3"
        mp3_path.write_bytes(b"ID3" + b"0" * 100)

        attempts = {"count": 0}

        def fake_create(**_kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                err = RuntimeError("Too many requests")
                err.status_code = 429  # type: ignore[attr-defined]
                raise err
            return SimpleNamespace(
                segments=[SimpleNamespace(start=0.0, end=2.0, text="Hello world")]
            )

        fake_client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=fake_create))
        )
        settings = _make_settings(
            self.tmp_dir,
            groq_api_key="gsk-test",
            transcribe_retry_attempts=2,
            transcribe_retry_base_delay_seconds=0.0,
        )

        artifact, warnings, chunk_count = transcribe_audio_file(
            mp3_path,
            settings,
            source_recording="lecture.mp3",
            progress_callback=progress.append,
            client_factory=lambda _cfg: fake_client,
            rate_limiter=limiter,
            sleep_fn=fake_sleep,
            duration_seconds=2.0,
        )
        self.assertEqual(chunk_count, 1)
        self.assertEqual(len(artifact.segments), 1)
        self.assertEqual(attempts["count"], 2)
        # 1st attempt used the 1/60s slot; 2nd attempt waited ~60s via OutboundRateLimiter
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 60.0, places=2)
        self.assertTrue(any("rate limit pacing" in msg.lower() for msg in progress))
        self.assertTrue(warnings)

    def test_gemini_ocr_applies_outbound_rate_limiter_across_batches(self) -> None:
        clock = _FakeClock(2000.0)
        limiter = OutboundRateLimiter(max_requests=1, window_seconds=60.0, clock=clock)
        sleeps: list[float] = []
        progress: list[str] = []

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock.advance(seconds)

        pdf_path = _create_sample_pdf(self.tmp_dir / "slides.pdf", pages=2)

        def fake_generate_content(**_kwargs):
            return SimpleNamespace(
                text=json.dumps([{"slide": 1, "text": "VISIBLE_TEXT:\nSlide content"}]),
                usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5),
            )

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=fake_generate_content)
        )
        settings = _make_settings(
            self.tmp_dir,
            gemini_api_key="AIza-test",
            ocr_max_pages_per_batch=1,
            ocr_parallel_workers=1,
            ocr_retry_attempts=1,
        )

        artifact, _warnings, stats = extract_slides_with_gemini_ocr(
            pdf_path,
            settings,
            source_pdf="slides.pdf",
            progress_callback=progress.append,
            client_factory=lambda _cfg: fake_client,
            rate_limiter=limiter,
            sleep_fn=fake_sleep,
        )
        self.assertEqual(artifact.slide_count, 2)
        self.assertEqual(stats["ocr_batches"], 2)
        # Batch 1 consumed the single slot, Batch 2 paused 60s via OutboundRateLimiter
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 60.0, places=2)
        self.assertTrue(any("rate limit pacing" in msg.lower() for msg in progress))


if __name__ == "__main__":
    unittest.main()
