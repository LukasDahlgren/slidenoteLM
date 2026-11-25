from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException, Response
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.requests import Request

from backend.app.auth import (
    AuthLoginRequest,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    create_session_token,
    derive_signing_key,
    verify_password,
    verify_session_token,
)
from backend.app.config import REPO_ROOT, Settings, load_settings
from backend.app.main import (
    AppPasswordMiddleware,
    _has_unsafe_path_syntax,
    _resolve_static_file,
    create_app,
)
from backend.app.pipeline import StageHandler
from backend.app.schemas import (
    ARTIFACT_SLIDES_OCR,
    ARTIFACT_TRANSCRIPT,
    JobContext,
    PipelineStage,
    SlidesOCRArtifact,
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


def _stub_handlers() -> dict[PipelineStage, StageHandler]:
    return {
        "audio_prep": _stub_audio_prep,
        "transcription": _stub_transcription,
        "ocr": _stub_ocr,
    }


class AuthAndConfigUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-auth-unit-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_password_and_session_token_helpers(self) -> None:
        self.assertTrue(verify_password("secret-123", "secret-123"))
        self.assertFalse(verify_password("wrong", "secret-123"))
        self.assertTrue(verify_password("anything", ""))

        # Deterministic signing when APP_SESSION_SECRET is empty (survives restarts)
        key_a = derive_signing_key("secret-123", "")
        key_b = derive_signing_key("secret-123", "")
        self.assertEqual(key_a, key_b)

        tok_a = create_session_token("secret-123", "")
        tok_b = create_session_token("secret-123", "")
        self.assertEqual(tok_a, tok_b)
        self.assertTrue(verify_session_token(tok_a, "secret-123", ""))
        self.assertFalse(verify_session_token(tok_a, "rotated-password", ""))
        self.assertFalse(verify_session_token("v1.tampered", "secret-123", ""))
        self.assertFalse(verify_session_token(None, "secret-123", ""))
        self.assertFalse(verify_session_token(tok_a, "", ""))

        # Explicit APP_SESSION_SECRET produces distinct signature and invalidates on rotation
        tok_secret = create_session_token("secret-123", "custom-signing-secret")
        self.assertNotEqual(tok_secret, tok_a)
        self.assertTrue(verify_session_token(tok_secret, "secret-123", "custom-signing-secret"))
        self.assertFalse(verify_session_token(tok_secret, "secret-123", "different-secret"))
        self.assertFalse(verify_session_token(tok_secret, "rotated-password", "custom-signing-secret"))

    def test_load_settings_parses_auth_env_vars(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            default_cfg = load_settings(env_file=self.tmp_dir / "missing.env")
        self.assertEqual(default_cfg.app_password, "")
        self.assertEqual(default_cfg.app_session_secret, "")

        with mock.patch.dict(
            os.environ,
            {
                "APP_PASSWORD": "my-lecture-password",
                "APP_SESSION_SECRET": "my-session-secret",
            },
            clear=False,
        ):
            cfg = load_settings(env_file=self.tmp_dir / "missing.env")
        self.assertEqual(cfg.app_password, "my-lecture-password")
        self.assertEqual(cfg.app_session_secret, "my-session-secret")

    def test_static_path_traversal_helpers(self) -> None:
        dist = self.tmp_dir / "dist"
        (dist / "assets").mkdir(parents=True)
        index_html = dist / "index.html"
        app_js = dist / "assets" / "app.js"
        outside_secret = self.tmp_dir / "secret.txt"
        index_html.write_text("<html>ok</html>", encoding="utf-8")
        app_js.write_text("console.log('ok')", encoding="utf-8")
        outside_secret.write_text("TOP SECRET", encoding="utf-8")

        self.assertTrue(_has_unsafe_path_syntax("../secret.txt"))
        self.assertTrue(_has_unsafe_path_syntax("assets/../../secret.txt"))
        self.assertTrue(_has_unsafe_path_syntax("assets\\app.js"))
        self.assertTrue(_has_unsafe_path_syntax("assets/app.js\x00.png"))
        self.assertFalse(_has_unsafe_path_syntax("assets/app.js"))

        self.assertEqual(_resolve_static_file(dist.resolve(), "assets/app.js"), app_js.resolve())
        self.assertIsNone(_resolve_static_file(dist.resolve(), "../secret.txt"))
        self.assertIsNone(_resolve_static_file(dist.resolve(), "assets/../../secret.txt"))
        self.assertIsNone(_resolve_static_file(dist.resolve(), "assets/missing.js"))

    def test_railway_and_docker_configs_present_and_valid(self) -> None:
        railway_path = REPO_ROOT / "railway.json"
        dockerfile_path = REPO_ROOT / "Dockerfile"
        dockerignore_path = REPO_ROOT / ".dockerignore"

        self.assertTrue(railway_path.is_file())
        self.assertTrue(dockerfile_path.is_file())
        self.assertTrue(dockerignore_path.is_file())

        railway_cfg = json.loads(railway_path.read_text(encoding="utf-8"))
        self.assertEqual(railway_cfg["build"]["builder"], "DOCKERFILE")
        self.assertEqual(railway_cfg["deploy"]["healthcheckPath"], "/api/health")
        self.assertEqual(railway_cfg["deploy"]["restartPolicyType"], "ON_FAILURE")

        dockerfile_text = dockerfile_path.read_text(encoding="utf-8")
        self.assertIn("node:22-slim", dockerfile_text)
        self.assertIn("python:3.12-slim", dockerfile_text)
        self.assertIn("ffmpeg", dockerfile_text)
        self.assertIn("ARG TARGETARCH", dockerfile_text)
        self.assertIn("ARG BWS_VERSION=2.1.0", dockerfile_text)
        self.assertIn("https://github.com/bitwarden/sdk-sm/releases/download/bws-v${BWS_VERSION}/bws-${BWS_ARCH}-unknown-linux-gnu-${BWS_VERSION}.zip", dockerfile_text)
        self.assertIn("chmod +x /usr/local/bin/bws", dockerfile_text)
        self.assertIn("bws --version", dockerfile_text)
        self.assertIn("${PORT:-8000}", dockerfile_text)

        cmd_lines = [line for line in dockerfile_text.splitlines() if line.startswith("CMD ")]
        self.assertEqual(len(cmd_lines), 1)
        cmd_args = json.loads(cmd_lines[0][4:])
        self.assertEqual(cmd_args[:2], ["sh", "-c"])
        entrypoint_script = cmd_args[2]
        self.assertIn("BWS_ACCESS_TOKEN", entrypoint_script)
        self.assertIn("BWS_PROJECT_ID", entrypoint_script)
        self.assertIn("exec bws run --project-id \"$BWS_PROJECT_ID\" -- uvicorn backend.app.main:app", entrypoint_script)
        self.assertIn("exec uvicorn backend.app.main:app", entrypoint_script)
        self.assertIn(">&2; exit 1", entrypoint_script)

    def test_app_password_middleware_and_routes_without_sockets(self) -> None:
        """Exercise AppPasswordMiddleware, /api/auth/* handlers, and SPA route directly."""

        def _step_sync(coro: object) -> object:
            try:
                coro.send(None)  # type: ignore[attr-defined]
            except StopIteration as stop:
                return stop.value
            raise AssertionError("Expected synchronous completion without event loop I/O.")

        def _make_request(
            path: str,
            *,
            method: str = "GET",
            scheme: str = "http",
            headers: list[tuple[bytes, bytes]] | None = None,
        ) -> Request:
            scope = {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": method,
                "scheme": scheme,
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": b"",
                "headers": headers or [],
                "server": ("testserver", 80),
                "client": ("127.0.0.1", 12345),
            }
            return Request(scope)

        dist = self.tmp_dir / "dist"
        (dist / "assets").mkdir(parents=True)
        (dist / "index.html").write_text("<html>SPA</html>", encoding="utf-8")
        (dist / "assets" / "app.js").write_text("console.log(1);", encoding="utf-8")
        (dist / "favicon.ico").write_bytes(b"ICO")
        (self.tmp_dir / "secret.txt").write_text("SECRET", encoding="utf-8")

        settings = _make_settings(self.tmp_dir, app_password="top-secret-password")
        app = create_app(settings=settings, frontend_dist_dir=dist)

        routes = {
            (route.path, tuple(sorted(route.methods or ()))) : route.endpoint
            for route in app.routes
            if isinstance(route, APIRoute)
        }
        auth_status_ep = routes[("/api/auth/status", ("GET",))]
        auth_login_ep = routes[("/api/auth/login", ("POST",))]
        auth_logout_ep = routes[("/api/auth/logout", ("POST",))]
        spa_ep = routes[("/{full_path:path}", ("GET",))]

        # Auth status unauthenticated
        res_unauth = auth_status_ep(_make_request("/api/auth/status"))
        self.assertEqual(res_unauth.model_dump(), {"auth_required": True, "authenticated": False})

        # Invalid login raises 401
        with self.assertRaises(HTTPException) as bad_ctx:
            auth_login_ep(
                AuthLoginRequest(password="wrong"),
                _make_request("/api/auth/login", method="POST"),
                Response(),
            )
        self.assertEqual(bad_ctx.exception.status_code, 401)

        # Valid login sets HttpOnly, SameSite=lax, Max-Age, and Secure on X-Forwarded-Proto: https
        https_resp = Response()
        ok_res = auth_login_ep(
            AuthLoginRequest(password="top-secret-password"),
            _make_request(
                "/api/auth/login",
                method="POST",
                headers=[(b"x-forwarded-proto", b"https")],
            ),
            https_resp,
        )
        self.assertEqual(ok_res.model_dump(), {"auth_required": True, "authenticated": True})
        set_cookie = https_resp.headers.get("set-cookie", "")
        self.assertIn(f"{SESSION_COOKIE_NAME}=", set_cookie)
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=lax", set_cookie)
        self.assertIn(f"Max-Age={SESSION_MAX_AGE_SECONDS}", set_cookie)
        self.assertIn("Secure", set_cookie)

        cookie_val = create_session_token("top-secret-password", "")
        cookie_header = [(b"cookie", f"{SESSION_COOKIE_NAME}={cookie_val}".encode("ascii"))]

        # Auth status authenticated with cookie
        res_auth = auth_status_ep(_make_request("/api/auth/status", headers=cookie_header))
        self.assertEqual(res_auth.model_dump(), {"auth_required": True, "authenticated": True})

        # Logout clears cookie
        logout_resp = Response()
        logout_data = auth_logout_ep(_make_request("/api/auth/logout", method="POST"), logout_resp)
        self.assertEqual(logout_data.model_dump(), {"auth_required": True, "authenticated": False})
        self.assertIn("Max-Age=0", logout_resp.headers.get("set-cookie", ""))

        # Middleware blocks unauthenticated /api/jobs*, passes /api/health, /api/auth/*, OPTIONS, and valid cookie
        called: list[str] = []

        async def downstream(scope, _receive, send) -> None:  # type: ignore[no-untyped-def]
            called.append(scope["path"])
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        mw = AppPasswordMiddleware(downstream, settings)

        def _invoke_mw(path: str, *, method: str = "GET", headers: list[tuple[bytes, bytes]] | None = None) -> tuple[int, bytes]:
            messages: list[dict] = []

            async def receive() -> dict:
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(msg: dict) -> None:
                messages.append(msg)

            scope = _make_request(path, method=method, headers=headers).scope
            _step_sync(mw(scope, receive, send))
            status_code = next(m["status"] for m in messages if m["type"] == "http.response.start")
            body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
            return status_code, body

        for blocked_method, blocked_path in [
            ("GET", "/api/jobs/tok_1"),
            ("POST", "/api/jobs"),
            ("GET", "/api/jobs/tok_1/download"),
            ("GET", "/api/jobs/tok_1/preview"),
        ]:
            code, body = _invoke_mw(blocked_path, method=blocked_method)
            self.assertEqual(code, 401)
            self.assertEqual(json.loads(body), {"detail": "Authentication required."})

        self.assertEqual(_invoke_mw("/api/health")[0], 200)
        self.assertEqual(_invoke_mw("/api/auth/status")[0], 200)
        self.assertEqual(_invoke_mw("/api/jobs", method="OPTIONS")[0], 200)
        self.assertEqual(_invoke_mw("/api/jobs/tok_1", headers=cookie_header)[0], 200)

        # SPA route handler checks
        self.assertEqual(Path(spa_ep("").path), (dist / "index.html").resolve())
        self.assertEqual(Path(spa_ep("jobs/tok_abc").path), (dist / "index.html").resolve())
        self.assertEqual(Path(spa_ep("assets/app.js").path), (dist / "assets" / "app.js").resolve())
        self.assertEqual(Path(spa_ep("favicon.ico").path), (dist / "favicon.ico").resolve())

        for not_found_path in [
            "api",
            "api/nonexistent",
            "assets/missing.js",
            "../secret.txt",
            "assets/../../secret.txt",
        ]:
            with self.assertRaises(HTTPException) as nf_ctx:
                spa_ep(not_found_path)
            self.assertEqual(nf_ctx.exception.status_code, 404, msg=not_found_path)


class AuthAndStaticHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-auth-http-"))
        self.dist_dir = self.tmp_dir / "dist"
        (self.dist_dir / "assets").mkdir(parents=True)
        (self.dist_dir / "index.html").write_text(
            "<!doctype html><html><body><div id='root'>SPA</div></body></html>",
            encoding="utf-8",
        )
        (self.dist_dir / "assets" / "index-abc.js").write_text(
            "console.log('slidenotelm');",
            encoding="utf-8",
        )
        (self.dist_dir / "favicon.ico").write_bytes(b"ICO")
        (self.tmp_dir / "secret.txt").write_text("DO NOT LEAK", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_auth_disabled_when_app_password_empty(self) -> None:
        app = create_app(
            settings=_make_settings(self.tmp_dir, app_password=""),
            stage_handlers=_stub_handlers(),
            frontend_dist_dir=self.dist_dir,
        )
        with TestClient(app) as client:
            status_resp = client.get("/api/auth/status")
            self.assertEqual(status_resp.status_code, 200)
            self.assertEqual(
                status_resp.json(),
                {"auth_required": False, "authenticated": True},
            )
            self.assertEqual(client.get("/api/health").status_code, 200)
            # Jobs routes are accessible (404 for unknown token, NOT 401)
            self.assertEqual(client.get("/api/jobs/unknown-tok").status_code, 404)

    def test_auth_gate_protects_jobs_persists_across_restarts_and_supports_logout(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            app_password="lecture-secret-pw",
            app_session_secret="",
        )
        app = create_app(
            settings=settings,
            stage_handlers=_stub_handlers(),
            frontend_dist_dir=self.dist_dir,
        )

        with TestClient(app) as client:
            # Health and auth status remain open
            self.assertEqual(client.get("/api/health").status_code, 200)
            self.assertEqual(
                client.get("/api/auth/status").json(),
                {"auth_required": True, "authenticated": False},
            )

            # All /api/jobs* routes reject unauthenticated requests with 401
            for method, path in [
                ("GET", "/api/jobs/tok_123"),
                ("GET", "/api/jobs/tok_123/preview"),
                ("GET", "/api/jobs/tok_123/download"),
                ("POST", "/api/jobs"),
            ]:
                resp = client.request(method, path)
                self.assertEqual(resp.status_code, 401, msg=f"{method} {path} should be 401")
                self.assertEqual(resp.json(), {"detail": "Authentication required."})

            # Wrong password rejected with 401 and no cookie
            bad_login = client.post("/api/auth/login", json={"password": "wrong-password"})
            self.assertEqual(bad_login.status_code, 401)
            self.assertNotIn(SESSION_COOKIE_NAME, client.cookies)

            # Valid password sets HttpOnly, SameSite=Lax cookie with 30-day Max-Age
            ok_login = client.post("/api/auth/login", json={"password": "lecture-secret-pw"})
            self.assertEqual(ok_login.status_code, 200)
            self.assertEqual(
                ok_login.json(),
                {"auth_required": True, "authenticated": True},
            )
            set_cookie = ok_login.headers.get("set-cookie", "")
            self.assertIn(f"{SESSION_COOKIE_NAME}=", set_cookie)
            self.assertIn("HttpOnly", set_cookie)
            self.assertIn("SameSite=lax", set_cookie)
            self.assertIn("Path=/", set_cookie)
            self.assertIn(f"Max-Age={SESSION_MAX_AGE_SECONDS}", set_cookie)
            self.assertNotIn("Secure", set_cookie)

            saved_cookie = client.cookies.get(SESSION_COOKIE_NAME)
            self.assertIsNotNone(saved_cookie)
            assert saved_cookie is not None

            self.assertEqual(
                client.get("/api/auth/status").json(),
                {"auth_required": True, "authenticated": True},
            )

            # Authenticated upload succeeds
            upload_resp = client.post(
                "/api/jobs",
                files={
                    "slides": ("week1.pdf", b"%PDF-1.4\nbody", "application/pdf"),
                    "recording": ("week1.mp3", b"ID3-audio", "audio/mpeg"),
                },
            )
            self.assertEqual(upload_resp.status_code, 202)
            token = upload_resp.json()["token"]
            app.state.worker.wait_for_job(token, timeout=5.0)
            self.assertEqual(client.get(f"/api/jobs/{token}").status_code, 200)

            # Logout clears cookie and locks /api/jobs* again
            logout_resp = client.post("/api/auth/logout")
            self.assertEqual(logout_resp.status_code, 200)
            self.assertEqual(
                logout_resp.json(),
                {"auth_required": True, "authenticated": False},
            )
            self.assertEqual(client.get(f"/api/jobs/{token}").status_code, 401)

        # Simulate container restart with the same APP_PASSWORD and empty APP_SESSION_SECRET
        restarted_app = create_app(
            settings=settings,
            stage_handlers=_stub_handlers(),
            frontend_dist_dir=self.dist_dir,
        )
        with TestClient(restarted_app) as client2:
            client2.cookies.set(SESSION_COOKIE_NAME, saved_cookie)
            self.assertEqual(
                client2.get("/api/auth/status").json(),
                {"auth_required": True, "authenticated": True},
            )
            self.assertEqual(client2.get(f"/api/jobs/{token}").status_code, 200)

    def test_secure_cookie_flag_set_when_https_or_x_forwarded_proto(self) -> None:
        app = create_app(
            settings=_make_settings(self.tmp_dir, app_password="pw"),
            frontend_dist_dir=self.dist_dir,
        )
        with TestClient(app) as client:
            resp = client.post(
                "/api/auth/login",
                json={"password": "pw"},
                headers={"X-Forwarded-Proto": "https"},
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Secure", resp.headers.get("set-cookie", ""))

    def test_spa_static_serving_does_not_shadow_api_and_blocks_traversal(self) -> None:
        app = create_app(
            settings=_make_settings(self.tmp_dir),
            frontend_dist_dir=self.dist_dir,
        )
        with TestClient(app) as client:
            # Root and client-side SPA routes serve index.html
            root_resp = client.get("/")
            self.assertEqual(root_resp.status_code, 200)
            self.assertIn("<div id='root'>SPA</div>", root_resp.text)

            job_page_resp = client.get("/jobs/tok_any_client_route")
            self.assertEqual(job_page_resp.status_code, 200)
            self.assertIn("<div id='root'>SPA</div>", job_page_resp.text)

            # Built assets & static root files served directly
            js_resp = client.get("/assets/index-abc.js")
            self.assertEqual(js_resp.status_code, 200)
            self.assertIn("console.log('slidenotelm')", js_resp.text)

            ico_resp = client.get("/favicon.ico")
            self.assertEqual(ico_resp.status_code, 200)
            self.assertEqual(ico_resp.content, b"ICO")

            # Missing asset under /assets/ returns 404 (not index.html)
            missing_asset = client.get("/assets/does-not-exist.js")
            self.assertEqual(missing_asset.status_code, 404)

            # Unknown /api/* routes ALWAYS return 404 JSON, never index.html
            unknown_api = client.get("/api/does-not-exist")
            self.assertEqual(unknown_api.status_code, 404)
            self.assertIn("application/json", unknown_api.headers.get("content-type", ""))
            self.assertEqual(unknown_api.json(), {"detail": "Not Found"})

            unknown_api_root = client.get("/api")
            self.assertEqual(unknown_api_root.status_code, 404)
            self.assertEqual(unknown_api_root.json(), {"detail": "Not Found"})

            # Path traversal attempts never leak files outside dist
            for traversal_url in [
                "/..%2fsecret.txt",
                "/assets/..%2f..%2fsecret.txt",
                "/assets/%2e%2e/%2e%2e/secret.txt",
            ]:
                resp = client.get(traversal_url)
                self.assertEqual(resp.status_code, 404, msg=f"Traversal {traversal_url} must be 404")
                self.assertNotIn("DO NOT LEAK", resp.text)

    def test_spa_not_mounted_when_dist_missing(self) -> None:
        missing_dist = self.tmp_dir / "nonexistent-dist"
        app = create_app(
            settings=_make_settings(self.tmp_dir),
            frontend_dist_dir=missing_dist,
        )
        with TestClient(app) as client:
            self.assertEqual(client.get("/api/health").status_code, 200)
            self.assertEqual(client.get("/").status_code, 404)


if __name__ == "__main__":
    unittest.main()
