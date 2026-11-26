from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pymupdf
from google.genai import errors as genai_errors
from google.genai import types

from backend.app.config import Settings, load_settings
from backend.app.pipeline.align import (
    align_with_llm,
    normalize_alignment_entries,
    run_alignment_stage,
)
from backend.app.pipeline.gemini_client import (
    build_thinking_config,
    call_gemini_text,
    compute_retry_delay_seconds,
    create_gemini_client,
    extract_retry_after_seconds,
    is_gemini_auth_error,
    is_gemini_live_mode,
    is_transient_gemini_error,
    supports_thinking_level,
    validate_gemini_credentials,
)
from backend.app.pipeline.notes import (
    _unwrap_outer_markdown_fence,
    check_coverage_with_llm,
    generate_notes,
    generate_section_notes_with_llm,
    run_notes_stage,
)
from backend.app.pipeline.ocr import extract_slides_with_gemini_ocr
from backend.app.pipeline.prompts import (
    ALIGNMENT_RESPONSE_JSON_SCHEMA,
    COVERAGE_CHECK_RESPONSE_JSON_SCHEMA,
    CoverageCheckInput,
    SectionPromptSegment,
    build_alignment_prompt,
    parse_alignment_response,
    parse_coverage_check_response,
)
from backend.app.ratelimit import OutboundRateLimiter
from backend.app.schemas import (
    AlignmentArtifact,
    AlignmentEntry,
    JobContext,
    SlideOCRRecord,
    SlidesOCRArtifact,
    StageError,
    TranscriptArtifact,
    TranscriptSegment,
)


def _make_settings(tmp_dir: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "data_dir": tmp_dir,
        "db_path": tmp_dir / "slidenotelm.db",
        "jobs_dir": tmp_dir / "jobs",
        "pipeline_llm_mode": "gemini",
        "gemini_api_key": "AIzaSy-phase6-test-key",
        "llm_retry_base_delay_seconds": 0.0,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def _make_context(tmp_dir: Path) -> JobContext:
    job_dir = tmp_dir / "job-phase6"
    input_dir = job_dir / "input"
    artifacts_dir = job_dir / "artifacts"
    input_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    slides_path = input_dir / "slides.pdf"
    slides_path.write_bytes(b"%PDF-1.7\n")
    recording_path = input_dir / "recording.mp3"
    recording_path.write_bytes(b"fake-audio")
    return JobContext(
        token="tok-phase6-gemini",
        slides_filename="neural_nets.pdf",
        recording_filename="neural_nets.mp3",
        job_dir=job_dir,
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        slides_path=slides_path,
        recording_path=recording_path,
    )


def _sample_slides() -> SlidesOCRArtifact:
    return SlidesOCRArtifact(
        source_pdf="neural_nets.pdf",
        model="gemini-4-neon",
        slide_count=2,
        slides=[
            SlideOCRRecord(
                slide=1,
                visible_text="Backpropagation\nChain rule through layers",
                visual_context="Computation graph with forward arrows in blue and backward gradients in red.",
                tables_code_formulas="dL/dw = dL/dy * dy/dw",
            ),
            SlideOCRRecord(
                slide=2,
                visible_text="Vanishing Gradients\nSigmoid saturation regions",
                visual_context="Plot of sigmoid derivative peaking at 0.25 at x=0 and flattening near 0.",
                tables_code_formulas="sigma'(x) = sigma(x)(1 - sigma(x)) <= 0.25",
            ),
        ],
    )


def _sample_transcript() -> TranscriptArtifact:
    return TranscriptArtifact(
        source_recording="neural_nets.mp3",
        model="whisper-large-v3-turbo",
        duration_seconds=90.0,
        segments=[
            TranscriptSegment(
                index=0,
                start=0.0,
                end=20.0,
                text="Look at the red backward arrows on the computation graph: we apply the chain rule step by step.",
            ),
            TranscriptSegment(
                index=1,
                start=20.0,
                end=45.0,
                text="Notice why multiplying many sigmoid derivatives fails: each factor is at most 0.25 so gradients vanish.",
            ),
            TranscriptSegment(
                index=2,
                start=45.0,
                end=75.0,
                text="Student asks whether ReLU completely eliminates this; lecturer explains dying ReLU for negative inputs.",
            ),
        ],
    )


class _FakeAPIError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.response = SimpleNamespace(status_code=code, headers={})


class _FakeTextClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.closed = False
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def close(self) -> None:
        self.closed = True

    def _generate_content(self, *, model: str, contents, config=None):
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        if not self._responses:
            raise AssertionError("Unexpected extra Gemini API call.")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if isinstance(nxt, str):
            return SimpleNamespace(
                text=nxt,
                candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
                prompt_feedback=None,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=1200,
                    candidates_token_count=400,
                    thoughts_token_count=250,
                ),
            )
        return nxt


class TestPhase6GeminiConfigAndThinking(unittest.TestCase):
    def test_load_settings_defaults_to_gemini_4_neon_varied_thinking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clean_env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(
                    ("PIPELINE_LLM_", "OCR_", "LLM_", "GEMINI_", "GOOGLE_")
                )
            }
            with mock.patch.dict(os.environ, clean_env, clear=True):
                cfg = load_settings(env_file=Path(tmp) / "missing.env")

        self.assertEqual(cfg.pipeline_llm_mode, "gemini")
        self.assertEqual(cfg.ocr_model_google, "gemini-4-neon")
        self.assertEqual(cfg.ocr_thinking_level, "LOW")
        self.assertEqual(cfg.llm_align_model, "gemini-4-neon")
        self.assertEqual(cfg.llm_align_thinking_level, "HIGH")
        self.assertEqual(cfg.llm_notes_model, "gemini-4-neon")
        self.assertEqual(cfg.llm_notes_thinking_level, "HIGH")
        self.assertEqual(cfg.llm_coverage_model, "gemini-4-neon")
        self.assertEqual(cfg.llm_coverage_thinking_level, "MEDIUM")
        self.assertEqual(cfg.llm_retry_attempts, 4)

    def test_load_settings_parses_custom_thinking_levels_off_and_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_patch = {
                "PIPELINE_LLM_MODE": "live",
                "OCR_MODEL_GOOGLE": "gemini-4-neon",
                "OCR_THINKING_LEVEL": "off",
                "LLM_ALIGN_THINKING_LEVEL": "xhigh",
                "LLM_NOTES_THINKING_LEVEL": "medium",
                "LLM_COVERAGE_THINKING_LEVEL": "default",
            }
            with mock.patch.dict(os.environ, env_patch, clear=False):
                cfg = load_settings(env_file=Path(tmp) / "missing.env")

        self.assertTrue(is_gemini_live_mode(cfg.pipeline_llm_mode))
        # OFF maps to MINIMAL (lowest thinking tier on Gemini 4); DEFAULT omits thinking_config.
        self.assertEqual(cfg.ocr_thinking_level, "MINIMAL")
        self.assertEqual(cfg.llm_align_thinking_level, "XHIGH")
        self.assertEqual(cfg.llm_notes_thinking_level, "MEDIUM")
        self.assertEqual(cfg.llm_coverage_thinking_level, "")

    def test_supports_thinking_level_and_build_thinking_config(self) -> None:
        self.assertTrue(supports_thinking_level("gemini-4-neon"))
        self.assertTrue(supports_thinking_level("publishers/google/models/gemini-4-carbon"))
        self.assertTrue(supports_thinking_level("gemini-3.8-flash"))
        self.assertFalse(supports_thinking_level("gemini-2.5-flash"))
        self.assertFalse(supports_thinking_level("publishers/google/models/gemini-2.5-pro"))
        self.assertFalse(supports_thinking_level("gemini-pro-latest"))
        self.assertFalse(supports_thinking_level("gemma-3-27b-it"))

        neon_cfg = build_thinking_config("LOW", model="gemini-4-neon")
        self.assertIsNotNone(neon_cfg)
        assert neon_cfg is not None
        self.assertEqual(neon_cfg.thinking_level, types.ThinkingLevel.LOW)

        off_cfg = build_thinking_config("off", model="gemini-4-neon")
        self.assertIsNotNone(off_cfg)
        assert off_cfg is not None
        self.assertEqual(off_cfg.thinking_level, types.ThinkingLevel.MINIMAL)

        default_cfg = build_thinking_config("default", model="gemini-4-neon")
        self.assertIsNone(default_cfg)

        legacy_cfg = build_thinking_config("LOW", model="gemini-2.5-flash")
        self.assertIsNone(legacy_cfg)

    def test_vertex_credentials_validation_and_http_timeout_ms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_vertex = _make_settings(
                Path(tmp),
                gemini_api_key="",
                google_genai_use_vertexai=True,
                google_cloud_project="",
            )
            with self.assertRaises(StageError) as ctx:
                validate_gemini_credentials(bad_vertex, stage="alignment")
            self.assertEqual(ctx.exception.stage, "alignment")
            self.assertIn("GOOGLE_CLOUD_PROJECT", ctx.exception.message)

            good_settings = _make_settings(Path(tmp), llm_timeout_seconds=123.5)
            with mock.patch("backend.app.pipeline.gemini_client.genai.Client") as mock_ctor:
                create_gemini_client(good_settings)
            mock_ctor.assert_called_once()
            http_opts = mock_ctor.call_args.kwargs["http_options"]
            self.assertEqual(http_opts.timeout, 123500)


class TestPhase6GeminiErrorsSafetyAndLifecycle(unittest.TestCase):
    def test_real_429_client_error_with_billing_text_retries_using_retry_info_delay(self) -> None:
        err_429 = genai_errors.ClientError(
            429,
            {
                "error": {
                    "code": 429,
                    "message": "You exceeded your current quota, please check your plan and billing details.",
                    "status": "RESOURCE_EXHAUSTED",
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "37s",
                        }
                    ],
                }
            },
        )
        self.assertFalse(is_gemini_auth_error(err_429))
        self.assertTrue(is_transient_gemini_error(err_429))
        self.assertEqual(extract_retry_after_seconds(err_429), 37.0)
        self.assertEqual(compute_retry_delay_seconds(1, 2.0, retry_after=37.0), 37.0)

        # Real 400 with API_KEY_INVALID IS an auth error; generic 400 is neither auth nor transient.
        err_400_key = genai_errors.ClientError(
            400,
            {
                "error": {
                    "code": 400,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )
        self.assertTrue(is_gemini_auth_error(err_400_key))

        err_400_bad_req = genai_errors.ClientError(
            400,
            {
                "error": {
                    "code": 400,
                    "message": "Invalid JSON schema argument.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )
        self.assertFalse(is_gemini_auth_error(err_400_bad_req))
        self.assertFalse(is_transient_gemini_error(err_400_bad_req))

        sleeps: list[float] = []
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), llm_retry_attempts=3)
            client = _FakeTextClient([err_429, "Recovered after quota wait"])
            res = call_gemini_text(
                stage="notes",
                task_label="quota retry test",
                settings=settings,
                model="gemini-4-neon",
                thinking_level="HIGH",
                system_instruction="sys",
                user_prompt="user",
                max_output_tokens=4096,
                client=client,
                sleep_fn=sleeps.append,
            )

        self.assertEqual(res.text, "Recovered after quota wait")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(sleeps, [37.0])
        # Injected client must NOT be closed by call_gemini_text.
        self.assertFalse(client.closed)

    def test_finish_reason_max_tokens_and_safety_blocks_raise_non_retryable_stage_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), llm_retry_attempts=3)

            max_tokens_client = _FakeTextClient(
                [
                    SimpleNamespace(
                        text="## Partial truncated section...",
                        candidates=[SimpleNamespace(finish_reason=types.FinishReason.MAX_TOKENS)],
                        prompt_feedback=None,
                    )
                ]
            )
            with self.assertRaises(StageError) as max_err:
                call_gemini_text(
                    stage="notes",
                    task_label="truncated section",
                    settings=settings,
                    model="gemini-4-neon",
                    thinking_level="HIGH",
                    system_instruction="sys",
                    user_prompt="user",
                    max_output_tokens=4096,
                    client=max_tokens_client,
                    sleep_fn=lambda _s: None,
                )
            self.assertIn("MAX_TOKENS", max_err.exception.message)
            self.assertEqual(len(max_tokens_client.calls), 1)

            safety_client = _FakeTextClient(
                [
                    SimpleNamespace(
                        text=None,
                        candidates=[SimpleNamespace(finish_reason=types.FinishReason.SAFETY)],
                        prompt_feedback=None,
                    )
                ]
            )
            with self.assertRaises(StageError) as safety_err:
                call_gemini_text(
                    stage="notes",
                    task_label="safety stop",
                    settings=settings,
                    model="gemini-4-neon",
                    thinking_level="HIGH",
                    system_instruction="sys",
                    user_prompt="user",
                    max_output_tokens=4096,
                    client=safety_client,
                    sleep_fn=lambda _s: None,
                )
            self.assertIn("SAFETY", safety_err.exception.message)
            self.assertEqual(len(safety_client.calls), 1)

            blocked_client = _FakeTextClient(
                [
                    SimpleNamespace(
                        text=None,
                        candidates=[],
                        prompt_feedback=SimpleNamespace(
                            block_reason=types.BlockedReason.PROHIBITED_CONTENT
                        ),
                    )
                ]
            )
            with self.assertRaises(StageError) as block_err:
                call_gemini_text(
                    stage="alignment",
                    task_label="blocked prompt",
                    settings=settings,
                    model="gemini-4-neon",
                    thinking_level="HIGH",
                    system_instruction="sys",
                    user_prompt="user",
                    max_output_tokens=4096,
                    client=blocked_client,
                    sleep_fn=lambda _s: None,
                )
            self.assertIn("PROHIBITED_CONTENT", block_err.exception.message)
            self.assertEqual(len(blocked_client.calls), 1)

    def test_empty_thought_only_response_retries_and_legacy_model_uses_zero_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), llm_retry_attempts=2)

            thought_only = SimpleNamespace(
                text=None,
                candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
                prompt_feedback=None,
            )
            client = _FakeTextClient([thought_only, "Valid answer"])
            res = call_gemini_text(
                stage="notes",
                task_label="thought-only retry",
                settings=settings,
                model="gemini-4-neon",
                thinking_level="default",
                system_instruction="sys",
                user_prompt="user",
                max_output_tokens=4096,
                client=client,
                sleep_fn=lambda _s: None,
            )
            self.assertEqual(res.text, "Valid answer")
            self.assertEqual(len(client.calls), 2)
            # Even with DEFAULT (no thinking_config), gemini-4-neon keeps default temperature (None).
            self.assertIsNone(client.calls[0]["config"].thinking_config)
            self.assertIsNone(client.calls[0]["config"].temperature)

            # Legacy gemini-2.5-flash omits thinking_config and sets temperature=0.0.
            legacy_client = _FakeTextClient(["Legacy ok"])
            call_gemini_text(
                stage="notes",
                task_label="legacy model test",
                settings=settings,
                model="gemini-2.5-flash",
                thinking_level="HIGH",
                system_instruction="sys",
                user_prompt="user",
                max_output_tokens=4096,
                client=legacy_client,
                sleep_fn=lambda _s: None,
            )
            self.assertIsNone(legacy_client.calls[0]["config"].thinking_config)
            self.assertEqual(legacy_client.calls[0]["config"].temperature, 0.0)

    def test_internally_created_client_is_closed_and_rate_limiter_invoked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp))
            internal_client = _FakeTextClient(["Closed cleanly"])
            limiter = mock.MagicMock(spec=OutboundRateLimiter)
            with mock.patch(
                "backend.app.pipeline.gemini_client.create_gemini_client",
                return_value=internal_client,
            ):
                res = call_gemini_text(
                    stage="alignment",
                    task_label="close test",
                    settings=settings,
                    model="gemini-4-neon",
                    thinking_level="HIGH",
                    system_instruction="sys",
                    user_prompt="user",
                    max_output_tokens=4096,
                    rate_limiter=limiter,
                )
            self.assertEqual(res.text, "Closed cleanly")
            self.assertTrue(internal_client.closed)
            limiter.acquire.assert_called_once()


class TestPhase6GeminiOCRThinking(unittest.TestCase):
    def test_ocr_passes_gemini_4_neon_and_low_thinking_without_forcing_zero_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "slides.pdf"
            doc = pymupdf.open()
            try:
                page = doc.new_page(width=640, height=360)
                page.insert_text((48, 72), "Slide 1: Transformers", fontsize=16)
                doc.save(str(pdf_path))
            finally:
                doc.close()

            settings = _make_settings(
                tmp_dir,
                ocr_model_google="gemini-4-neon",
                ocr_thinking_level="LOW",
            )
            fake_client = _FakeTextClient(
                [json.dumps([{"slide": 1, "text": "VISIBLE_TEXT:\nTransformers"}])]
            )
            artifact, warnings, _stats = extract_slides_with_gemini_ocr(
                pdf_path,
                settings,
                source_pdf="slides.pdf",
                client_factory=lambda _s: fake_client,
            )

        self.assertEqual(artifact.model, "gemini-4-neon")
        self.assertEqual(warnings, [])
        self.assertEqual(len(fake_client.calls), 1)
        call = fake_client.calls[0]
        self.assertEqual(call["model"], "gemini-4-neon")
        cfg = call["config"]
        self.assertIsNotNone(cfg.thinking_config)
        self.assertEqual(cfg.thinking_config.thinking_level, types.ThinkingLevel.LOW)
        self.assertIsNone(cfg.temperature)


class TestPhase6GeminiAlignment(unittest.TestCase):
    def test_alignment_prompt_parser_and_string_bool_normalization(self) -> None:
        slides = _sample_slides()
        transcript = _sample_transcript()
        prompt = build_alignment_prompt(slides, transcript)
        self.assertIn("Slide 1", prompt.user)
        self.assertIn("VISUAL_CONTEXT:", prompt.user)
        self.assertIn("seg 0", prompt.user)

        parsed_obj = parse_alignment_response(
            '```json\n{"entries": [{"slide": 1, "start_segment": 0, "end_segment": 0, "uncertain": "false"}]}\n```'
        )
        self.assertEqual(len(parsed_obj), 1)
        normalized, issues = normalize_alignment_entries(parsed_obj, transcript, slides)
        self.assertEqual(issues, [])
        self.assertFalse(normalized[0].uncertain)

        parsed_arr = parse_alignment_response(
            '[{"slide": null, "start_segment": 0, "end_segment": 2, "uncertain": "true", "reason": "Q&A"}]'
        )
        normalized_arr, _ = normalize_alignment_entries(parsed_arr, transcript, slides)
        self.assertTrue(normalized_arr[0].uncertain)

    def test_align_with_llm_enforces_schema_retries_malformed_json_and_tracks_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="gemini", llm_retry_attempts=3)
            slides = _sample_slides()
            transcript = _sample_transcript()
            fake_client = _FakeTextClient(
                [
                    "{not valid json",
                    json.dumps(
                        {
                            "entries": [
                                {
                                    "slide": 1,
                                    "start_segment": 0,
                                    "end_segment": 0,
                                    "uncertain": False,
                                    "reason": "",
                                },
                                {
                                    "slide": 2,
                                    "start_segment": 1,
                                    "end_segment": 1,
                                    "uncertain": False,
                                    "reason": "",
                                },
                                {
                                    "slide": None,
                                    "start_segment": 2,
                                    "end_segment": 2,
                                    "uncertain": False,
                                    "reason": "Student Q&A on ReLU",
                                },
                            ]
                        }
                    ),
                ]
            )
            usage: dict[str, int] = {}
            artifact = align_with_llm(
                slides,
                transcript,
                settings,
                client_factory=lambda _s: fake_client,
                usage_out=usage,
            )

        self.assertEqual(artifact.mode, "gemini")
        self.assertEqual(len(artifact.entries), 3)
        self.assertEqual(len(fake_client.calls), 2)
        cfg = fake_client.calls[0]["config"]
        self.assertEqual(cfg.response_mime_type, "application/json")
        self.assertEqual(cfg.response_json_schema, ALIGNMENT_RESPONSE_JSON_SCHEMA)
        self.assertEqual(cfg.thinking_config.thinking_level, types.ThinkingLevel.HIGH)
        self.assertIsNone(cfg.temperature)
        self.assertEqual(usage["align_input_tokens"], 1200)
        self.assertEqual(usage["align_output_tokens"], 400)
        self.assertEqual(usage["align_thinking_tokens"], 250)

    def test_align_short_circuits_empty_transcript_and_empty_slides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="gemini")
            fake_client = _FakeTextClient([])

            empty_tx = TranscriptArtifact(
                source_recording="empty.mp3",
                model="whisper-large-v3-turbo",
                duration_seconds=0.0,
                segments=[],
            )
            res_empty_tx = align_with_llm(
                _sample_slides(),
                empty_tx,
                settings,
                client_factory=lambda _s: fake_client,
            )
            self.assertEqual(res_empty_tx.entries, [])
            self.assertEqual(len(fake_client.calls), 0)

            empty_slides = SlidesOCRArtifact(
                source_pdf="empty.pdf",
                model="gemini-4-neon",
                slide_count=0,
                slides=[],
            )
            res_empty_slides = align_with_llm(
                empty_slides,
                _sample_transcript(),
                settings,
                client_factory=lambda _s: fake_client,
            )
            self.assertEqual(len(res_empty_slides.entries), 1)
            self.assertIsNone(res_empty_slides.entries[0].slide)
            self.assertEqual(len(fake_client.calls), 0)

    def test_align_missing_api_key_raises_actionable_stage_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), gemini_api_key="", pipeline_llm_mode="gemini")
            with self.assertRaises(StageError) as ctx:
                align_with_llm(_sample_slides(), _sample_transcript(), settings)
        self.assertEqual(ctx.exception.stage, "alignment")
        self.assertIn("GEMINI_API_KEY", ctx.exception.message)

    def test_run_alignment_stage_persists_gemini_artifact_and_token_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ctx = _make_context(tmp_dir)
            ctx.slides_ocr_path.write_text(_sample_slides().model_dump_json(), encoding="utf-8")
            ctx.transcript_path.write_text(
                _sample_transcript().model_dump_json(), encoding="utf-8"
            )
            settings = _make_settings(tmp_dir, pipeline_llm_mode="live")
            fake_client = _FakeTextClient(
                [
                    json.dumps(
                        {
                            "entries": [
                                {"slide": 1, "start_segment": 0, "end_segment": 0},
                                {"slide": 2, "start_segment": 1, "end_segment": 2},
                            ]
                        }
                    )
                ]
            )
            result = run_alignment_stage(ctx, settings, client_factory=lambda _s: fake_client)
            persisted = AlignmentArtifact.model_validate_json(
                ctx.alignment_path.read_text(encoding="utf-8")
            )

        self.assertEqual(result.stats["alignment_mode"], "gemini")
        self.assertEqual(result.stats["align_input_tokens"], 1200)
        self.assertEqual(result.stats["align_output_tokens"], 400)
        self.assertEqual(result.stats["align_thinking_tokens"], 250)
        self.assertEqual(persisted.mode, "gemini")
        self.assertEqual(len(persisted.entries), 2)


class TestPhase6GeminiNotesAndCoverage(unittest.TestCase):
    def test_unwrap_outer_markdown_fence_edge_cases(self) -> None:
        # Labeled markdown fence wrapping inner python code block -> unwrap cleanly.
        wrapped = (
            "```markdown\n"
            "## Backpropagation (Slide 1) [00:00–00:20]\n\n"
            "```python\n"
            "grad = dL_dy * dy_dw\n"
            "```\n"
            "```"
        )
        unwrapped = _unwrap_outer_markdown_fence(wrapped)
        self.assertTrue(unwrapped.startswith("## Backpropagation"))
        self.assertIn("```python", unwrapped)
        self.assertTrue(unwrapped.endswith("```"))

        # Unlabeled fence wrapping a pure heading section (no inner fences) -> unwrap cleanly.
        unlabeled_md = "```\n## Gradient Descent (Slide 1) [00:00–00:20]\n\n- Step 1\n```"
        self.assertEqual(
            _unwrap_outer_markdown_fence(unlabeled_md),
            "## Gradient Descent (Slide 1) [00:00–00:20]\n\n- Step 1",
        )

        # Critical regression: unlabeled code blocks at BOTH ends must NOT be damaged!
        two_blocks = "```\nprint(1)\n```\n\nSome text\n\n```\nprint(2)\n```"
        self.assertEqual(_unwrap_outer_markdown_fence(two_blocks), two_blocks)

        comment_blocks = "```\n# comment\nprint(1)\n```\n\nText\n\n```\nprint(2)\n```"
        self.assertEqual(_unwrap_outer_markdown_fence(comment_blocks), comment_blocks)

    def test_parse_coverage_check_response_raises_value_error_on_bad_int_type(self) -> None:
        with self.assertRaises(ValueError):
            parse_coverage_check_response(
                {"passed": True, "total_segment_count": {"bad": 1}, "missing_items": []}
            )

    def test_generate_notes_uses_high_medium_schemas_reuses_client_and_degrades_audit_gracefully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="gemini", llm_retry_attempts=2)
            slides = _sample_slides()
            transcript = _sample_transcript()
            alignment = AlignmentArtifact(
                mode="gemini",
                entries=[
                    AlignmentEntry(
                        slide=1, start_segment=0, end_segment=0, start_time=0.0, end_time=20.0
                    ),
                    AlignmentEntry(
                        slide=2, start_segment=1, end_segment=1, start_time=20.0, end_time=45.0
                    ),
                    AlignmentEntry(
                        slide=None,
                        start_segment=2,
                        end_segment=2,
                        start_time=45.0,
                        end_time=75.0,
                        reason="Student Q&A on ReLU",
                    ),
                ],
            )
            fake_client = _FakeTextClient(
                [
                    # Section 1 (Slides 1-2) notes
                    (
                        "```markdown\n"
                        "## Backpropagation and Vanishing Gradients (Slides 1–2) [00:00–00:45]\n\n"
                        "- Red backward arrows propagate dL/dw via the chain rule.\n"
                        "- Sigmoid derivatives peak at 0.25, causing gradients to shrink across layers.\n"
                        "```"
                    ),
                    # Section 1 coverage audit: malformed JSON on attempt 1, valid on attempt 2
                    "{broken json",
                    json.dumps(
                        {
                            "passed": True,
                            "covered_segment_count": 2,
                            "total_segment_count": 2,
                            "missing_items": [],
                            "notes": "All substantive steps covered.",
                        }
                    ),
                    # Section 2 (off-slide Q&A) notes
                    (
                        "## Student Q&A: ReLU and Dying ReLU [00:45–01:15]\n\n"
                        "- ReLU avoids sigmoid saturation for positive inputs, but negative inputs can cause dying ReLU.\n"
                    ),
                    # Section 2 coverage audit fails both attempts -> must NOT discard generated notes!
                    "{still broken 1",
                    "{still broken 2",
                ]
            )
            factory_calls = 0

            def counting_factory(_s: Settings) -> _FakeTextClient:
                nonlocal factory_calls
                factory_calls += 1
                return fake_client

            result = generate_notes(
                slides,
                transcript,
                alignment,
                settings,
                client_factory=counting_factory,
            )

        self.assertEqual(factory_calls, 1)
        self.assertEqual(len(fake_client.calls), 6)

        notes_call_1 = fake_client.calls[0]
        cov_call_1 = fake_client.calls[1]
        self.assertEqual(notes_call_1["model"], "gemini-4-neon")
        self.assertEqual(notes_call_1["config"].thinking_config.thinking_level, types.ThinkingLevel.HIGH)
        self.assertIsNone(notes_call_1["config"].response_mime_type)

        self.assertEqual(cov_call_1["model"], "gemini-4-neon")
        self.assertEqual(cov_call_1["config"].thinking_config.thinking_level, types.ThinkingLevel.MEDIUM)
        self.assertEqual(cov_call_1["config"].response_mime_type, "application/json")
        self.assertEqual(
            cov_call_1["config"].response_json_schema,
            COVERAGE_CHECK_RESPONSE_JSON_SCHEMA,
        )

        # Notes preserved even though section 2 coverage audit failed!
        self.assertIn("- **Notes Mode:** `gemini`", result.markdown)
        self.assertNotIn("```markdown", result.markdown)
        self.assertIn("## Backpropagation and Vanishing Gradients", result.markdown)
        self.assertIn("## Student Q&A: ReLU and Dying ReLU", result.markdown)
        self.assertFalse(result.coverage_result.passed)
        self.assertTrue(any("Coverage audit unavailable" in w for w in result.warnings))
        self.assertEqual(result.notes_input_tokens, 2400)
        self.assertEqual(result.coverage_input_tokens, 1200)

    def test_generate_notes_closes_internally_created_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="gemini")
            slides = _sample_slides()
            transcript = _sample_transcript()
            alignment = AlignmentArtifact(
                mode="gemini",
                entries=[
                    AlignmentEntry(
                        slide=1, start_segment=0, end_segment=2, start_time=0.0, end_time=75.0
                    )
                ],
            )
            internal_client = _FakeTextClient(
                [
                    "## Backprop (Slide 1) [00:00–01:15]\n\n- Notes body.",
                    json.dumps(
                        {
                            "passed": True,
                            "covered_segment_count": 3,
                            "total_segment_count": 3,
                            "missing_items": [],
                            "notes": "Ok.",
                        }
                    ),
                ]
            )
            with mock.patch(
                "backend.app.pipeline.notes.create_gemini_client",
                return_value=internal_client,
            ):
                generate_notes(slides, transcript, alignment, settings)
            self.assertTrue(internal_client.closed)

    def test_direct_coverage_check_raises_stage_error_on_repeated_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="gemini", llm_retry_attempts=2)
            cov_input = CoverageCheckInput(
                lecture_title="Test",
                section_index=1,
                total_sections=1,
                section_kind="aligned",
                start_time=0.0,
                end_time=20.0,
                segments=(SectionPromptSegment(index=0, start=0.0, end=20.0, text="Hello"),),
                generated_markdown="## Intro\n- Hello",
                slide_numbers=(1,),
            )
            bad_client = _FakeTextClient(["not json 1", "not json 2"])
            with self.assertRaises(StageError) as ctx:
                check_coverage_with_llm(
                    cov_input,
                    settings,
                    client_factory=lambda _s: bad_client,
                )
            self.assertEqual(ctx.exception.stage, "notes")
            self.assertEqual(len(bad_client.calls), 2)

    def test_run_notes_stage_writes_markdown_and_surfaces_token_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ctx = _make_context(tmp_dir)
            slides = _sample_slides()
            transcript = _sample_transcript()
            alignment = AlignmentArtifact(
                mode="gemini",
                entries=[
                    AlignmentEntry(
                        slide=1, start_segment=0, end_segment=2, start_time=0.0, end_time=75.0
                    )
                ],
            )
            ctx.slides_ocr_path.write_text(slides.model_dump_json(), encoding="utf-8")
            ctx.transcript_path.write_text(transcript.model_dump_json(), encoding="utf-8")
            ctx.alignment_path.write_text(alignment.model_dump_json(), encoding="utf-8")
            settings = _make_settings(tmp_dir, pipeline_llm_mode="gemini")
            fake_client = _FakeTextClient(
                [
                    "## Backpropagation (Slide 1) [00:00–01:15]\n\n- Detailed chain rule explanation.",
                    json.dumps(
                        {
                            "passed": True,
                            "covered_segment_count": 3,
                            "total_segment_count": 3,
                            "missing_items": [],
                            "notes": "Complete.",
                        }
                    ),
                ]
            )

            stage_result = run_notes_stage(
                ctx, settings, client_factory=lambda _s: fake_client
            )
            written = ctx.notes_path.read_text(encoding="utf-8")

        self.assertTrue(stage_result.stats["coverage_passed"])
        self.assertEqual(stage_result.stats["notes_input_tokens"], 1200)
        self.assertEqual(stage_result.stats["notes_output_tokens"], 400)
        self.assertEqual(stage_result.stats["notes_thinking_tokens"], 250)
        self.assertEqual(stage_result.stats["coverage_input_tokens"], 1200)
        self.assertIn("Detailed chain rule explanation.", written)

    def test_call_gemini_text_fails_fast_on_401_and_retries_429(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), llm_retry_attempts=3)
            retrying_client = _FakeTextClient(
                [
                    _FakeAPIError("RESOURCE_EXHAUSTED 429", code=429),
                    "Recovered output",
                ]
            )
            res = call_gemini_text(
                stage="notes",
                task_label="test retry",
                settings=settings,
                model="gemini-4-neon",
                thinking_level="HIGH",
                system_instruction="sys",
                user_prompt="user",
                max_output_tokens=4096,
                client=retrying_client,
                sleep_fn=lambda _s: None,
            )
            self.assertEqual(res.text, "Recovered output")
            self.assertEqual(len(retrying_client.calls), 2)

            auth_client = _FakeTextClient([_FakeAPIError("API_KEY_INVALID", code=401)])
            with self.assertRaises(StageError) as auth_err:
                call_gemini_text(
                    stage="notes",
                    task_label="test auth",
                    settings=settings,
                    model="gemini-4-neon",
                    thinking_level="HIGH",
                    system_instruction="sys",
                    user_prompt="user",
                    max_output_tokens=4096,
                    client=auth_client,
                    sleep_fn=lambda _s: None,
                )
            self.assertEqual(auth_err.exception.stage, "notes")
            self.assertEqual(len(auth_client.calls), 1)


if __name__ == "__main__":
    unittest.main()
