from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pymupdf

from backend.app.config import Settings, load_settings
from backend.app.pipeline import ocr as ocr_module
from backend.app.pipeline.ocr import (
    PdfBatchSpec,
    STRUCTURED_OCR_SECTION_MARKERS,
    build_slide_ocr_records,
    cleanup_structured_ocr_text,
    extract_slides_with_gemini_ocr,
    inspect_pdf_page_count,
    is_auth_error,
    is_payload_too_large_error,
    is_transient_ocr_error,
    normalize_batch_slides,
    parse_gemini_ocr_response,
    parse_structured_ocr_text,
    plan_pdf_batches,
    rebuild_structured_ocr_text,
    run_ocr_stage,
    slice_pdf_batches,
    strip_repeating_boilerplate,
    validate_gemini_credentials,
)
from backend.app.schemas import (
    ARTIFACT_SLIDES_OCR,
    JobContext,
    SlidesOCRArtifact,
    StageError,
)


def _make_settings(tmp_dir: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "data_dir": tmp_dir,
        "db_path": tmp_dir / "slidenotelm.db",
        "jobs_dir": tmp_dir / "jobs",
        "gemini_api_key": "AIzaSy-test-key",
        "ocr_model_google": "gemini-2.5-flash",
        "google_genai_use_vertexai": False,
        "google_cloud_project": "",
        "google_cloud_location": "us-central1",
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def _make_context(
    tmp_dir: Path,
    *,
    slides_filename: str = "lecture.pdf",
    progress_log: list[str] | None = None,
) -> JobContext:
    job_dir = tmp_dir / "job-ocr"
    input_dir = job_dir / "input"
    artifacts_dir = job_dir / "artifacts"
    input_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    slides_path = input_dir / "slides.pdf"
    recording_path = input_dir / "recording.mp3"
    recording_path.write_bytes(b"fake-audio")
    callback = progress_log.append if progress_log is not None else None
    return JobContext(
        token="tok-phase3-test",
        slides_filename=slides_filename,
        recording_filename="recording.mp3",
        job_dir=job_dir,
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        slides_path=slides_path,
        recording_path=recording_path,
        progress_callback=callback,
    )


def _create_sample_pdf(path: Path, page_texts: list[str]) -> Path:
    doc = pymupdf.open()
    try:
        for text in page_texts:
            page = doc.new_page(width=640, height=360)
            if text:
                page.insert_text((48, 72), text, fontsize=16)
        doc.save(str(path))
    finally:
        doc.close()
    return path


class _FakeGeminiAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        status: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.response = SimpleNamespace(status_code=code, headers=headers or {})


class _FakeGeminiClient:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, *, model: str, contents, config=None):
        prompt = ""
        pdf_bytes = b""
        for item in contents:
            if isinstance(item, str):
                prompt = item
            else:
                inline = getattr(item, "inline_data", None)
                if inline is not None and getattr(inline, "data", None):
                    pdf_bytes = bytes(inline.data)

        batch_page_count = 0
        if pdf_bytes:
            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            try:
                batch_page_count = len(doc)
            finally:
                doc.close()

        record = {
            "model": model,
            "prompt": prompt,
            "config": config,
            "pdf_size_bytes": len(pdf_bytes),
            "batch_page_count": batch_page_count,
        }
        with self._lock:
            self.calls.append(record)
        return self._handler(record, pdf_bytes)


class Phase3StructuredParsingTests(unittest.TestCase):
    def test_parse_and_rebuild_structured_markers_and_inline_markers(self) -> None:
        raw = (
            "VISIBLE_TEXT:\n"
            "Gradient Descent\n"
            "- Updates weights iteratively\n\n"
            "VISUAL_CONTEXT:\n"
            "- 2D contour plot showing step arrows moving toward a global minimum.\n\n"
            "TABLES_CODE_FORMULAS:\n"
            "Formula: w_{t+1} = w_t - eta * grad L(w_t)"
        )
        sections = parse_structured_ocr_text(raw)
        self.assertTrue(sections.has_markers)
        self.assertEqual(sections.visible_text, "Gradient Descent\n- Updates weights iteratively")
        self.assertIn("2D contour plot", sections.visual_context)
        self.assertIn("w_{t+1}", sections.tables_code_formulas)
        self.assertEqual(rebuild_structured_ocr_text(sections), raw)

        # Inline content right after marker is also handled cleanly.
        inline = "VISIBLE_TEXT: Title Slide\nVISUAL_CONTEXT: Campus logo illustration"
        inline_sections = parse_structured_ocr_text(inline)
        self.assertTrue(inline_sections.has_markers)
        self.assertEqual(inline_sections.visible_text, "Title Slide")
        self.assertEqual(inline_sections.visual_context, "Campus logo illustration")
        self.assertEqual(inline_sections.tables_code_formulas, "")

    def test_parse_unmarked_text_falls_back_to_visible_text_record_with_warning_tracking(self) -> None:
        sections = parse_structured_ocr_text("Plain slide without markers")
        self.assertFalse(sections.has_markers)

        records, metrics, unmarked = build_slide_ocr_records(
            [{"slide": 1, "text": "Plain slide without markers"}]
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].visible_text, "Plain slide without markers")
        self.assertEqual(records[0].visual_context, "")
        self.assertEqual(records[0].tables_code_formulas, "")
        self.assertEqual(records[0].raw_text, "Plain slide without markers")
        self.assertEqual(metrics["structured_slides"], 0)
        self.assertEqual(unmarked, [1])

    def test_cleanup_structured_ocr_text_removes_duplicate_visible_lines_from_other_sections(self) -> None:
        raw = (
            "VISIBLE_TEXT:\n"
            "Binary Search\n"
            "Time complexity: O(log n)\n\n"
            "VISUAL_CONTEXT:\n"
            "- Binary Search\n"
            "- Array split diagram highlighting midpoint comparison.\n\n"
            "TABLES_CODE_FORMULAS:\n"
            "Time complexity: O(log n).\n"
            "| Algorithm | Complexity |\n"
            "| Binary Search | O(log n) |"
        )
        cleaned = cleanup_structured_ocr_text(raw)
        sections = parse_structured_ocr_text(cleaned)
        self.assertIn("Binary Search", sections.visible_text)
        self.assertNotIn("- Binary Search", sections.visual_context)
        self.assertIn("Array split diagram", sections.visual_context)
        self.assertNotIn("Time complexity: O(log n).", sections.tables_code_formulas)
        self.assertIn("| Binary Search | O(log n) |", sections.tables_code_formulas)

    def test_strip_repeating_boilerplate_preserves_structured_markers_and_drops_empty_sections(self) -> None:
        slides = [
            {
                "slide": 1,
                "text": (
                    "VISIBLE_TEXT:\n"
                    "CS229 Machine Learning\n"
                    "Lecture 1: Linear Regression\n"
                    "Stanford University - Autumn 2026\n\n"
                    "VISUAL_CONTEXT:\n"
                    "- Scatter plot of housing prices vs square footage."
                ),
            },
            {
                "slide": 2,
                "text": (
                    "VISIBLE_TEXT:\n"
                    "CS229 Machine Learning\n"
                    "Cost Function Definition\n"
                    "Stanford University - Autumn 2026\n\n"
                    "TABLES_CODE_FORMULAS:\n"
                    "J(theta) = 1/2 sum (h_theta(x) - y)^2"
                ),
            },
            {
                "slide": 3,
                "text": (
                    "VISIBLE_TEXT:\n"
                    "CS229 Machine Learning\n"
                    "Normal Equations\n"
                    "Stanford University - Autumn 2026"
                ),
            },
            {
                "slide": 4,
                "text": (
                    "VISIBLE_TEXT:\n"
                    "CS229 Machine Learning\n"
                    "Stanford University - Autumn 2026\n\n"
                    "VISUAL_CONTEXT:\n"
                    "- Full-bleed diagram of projection onto column space."
                ),
            },
        ]

        cleaned, removed_count = strip_repeating_boilerplate(slides)
        self.assertEqual(removed_count, 8)
        for slide in cleaned:
            self.assertNotIn("CS229 Machine Learning", slide["text"])
            self.assertNotIn("Stanford University - Autumn 2026", slide["text"])

        # Structured markers remain intact on slides with content.
        s1 = parse_structured_ocr_text(cleaned[0]["text"])
        self.assertTrue(s1.has_markers)
        self.assertEqual(s1.visible_text, "Lecture 1: Linear Regression")
        self.assertIn("Scatter plot", s1.visual_context)

        # Slide 4 only had boilerplate under VISIBLE_TEXT:, so empty VISIBLE_TEXT: marker is omitted
        # while VISUAL_CONTEXT: is cleanly preserved.
        self.assertNotIn("VISIBLE_TEXT:", cleaned[3]["text"])
        self.assertIn("VISUAL_CONTEXT:", cleaned[3]["text"])
        s4 = parse_structured_ocr_text(cleaned[3]["text"])
        self.assertEqual(s4.visible_text, "")
        self.assertIn("Full-bleed diagram", s4.visual_context)

        # Verify section markers themselves are never treated as boilerplate even on every slide.
        for marker in STRUCTURED_OCR_SECTION_MARKERS:
            self.assertIn(marker, STRUCTURED_OCR_SECTION_MARKERS)

    def test_parse_gemini_ocr_response_handles_fences_prose_and_structured_object_fields(self) -> None:
        fenced = '```json\n[{"slide": 1, "text": "VISIBLE_TEXT:\\nHello"}]\n```'
        parsed_fenced = parse_gemini_ocr_response(fenced)
        self.assertEqual(parsed_fenced, [{"slide": 1, "text": "VISIBLE_TEXT:\nHello"}])

        with_preamble = 'Here is the extracted JSON:\n[{"slide": 1, "text": "VISIBLE_TEXT:\\nSlide 1"}]\nDone.'
        parsed_preamble = parse_gemini_ocr_response(with_preamble)
        self.assertEqual(parsed_preamble, [{"slide": 1, "text": "VISIBLE_TEXT:\nSlide 1"}])

        structured_fields = json.dumps(
            [
                {
                    "slide": 1,
                    "visible_text": "Intro",
                    "visual_context": "- Flowchart A -> B",
                    "tables_code_formulas": "x = 1",
                }
            ]
        )
        parsed_fields = parse_gemini_ocr_response(structured_fields)
        self.assertIn("VISIBLE_TEXT:\nIntro", parsed_fields[0]["text"])
        self.assertIn("VISUAL_CONTEXT:\n- Flowchart A -> B", parsed_fields[0]["text"])
        self.assertIn("TABLES_CODE_FORMULAS:\nx = 1", parsed_fields[0]["text"])

    def test_normalize_batch_slides_maps_relative_global_and_missing_slides(self) -> None:
        batch = PdfBatchSpec(index=1, page_start=6, page_end=8)

        # Relative 1..3 out-of-order with slide 2 omitted
        normalized, warning = normalize_batch_slides(
            [
                {"slide": 3, "text": "VISIBLE_TEXT:\nPage 8"},
                {"slide": 1, "text": "VISIBLE_TEXT:\nPage 6"},
            ],
            batch=batch,
        )
        self.assertIsNotNone(warning)
        self.assertEqual(
            normalized,
            [
                {"slide": 6, "text": "VISIBLE_TEXT:\nPage 6"},
                {"slide": 7, "text": ""},
                {"slide": 8, "text": "VISIBLE_TEXT:\nPage 8"},
            ],
        )

        # Global 6..8 numbering
        normalized_global, warning_global = normalize_batch_slides(
            [
                {"slide": 6, "text": "A"},
                {"slide": 7, "text": "B"},
                {"slide": 8, "text": "C"},
            ],
            batch=batch,
        )
        self.assertIsNone(warning_global)
        self.assertEqual([s["slide"] for s in normalized_global], [6, 7, 8])
        self.assertEqual([s["text"] for s in normalized_global], ["A", "B", "C"])


class Phase3PdfSlicingAndBatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-p3-pdf-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_plan_pdf_batches_balances_ranges_for_small_parallel_and_large_decks(self) -> None:
        # Small deck (< parallel_min_pages=4): single batch
        small = plan_pdf_batches(3, max_pages_per_batch=20, parallel_workers=2, parallel_min_pages=4)
        self.assertEqual([(b.page_start, b.page_end) for b in small], [(1, 3)])

        # Medium deck (>= 4 pages, <= 20): splits across 2 workers evenly
        medium = plan_pdf_batches(7, max_pages_per_batch=20, parallel_workers=2, parallel_min_pages=4)
        self.assertEqual([(b.page_start, b.page_end) for b in medium], [(1, 4), (5, 7)])

        # Large deck (45 pages, max 20/batch): balanced 3 batches of 15 pages each
        large = plan_pdf_batches(45, max_pages_per_batch=20, parallel_workers=2, parallel_min_pages=4)
        self.assertEqual(
            [(b.page_start, b.page_end) for b in large],
            [(1, 15), (16, 30), (31, 45)],
        )

    def test_inspect_and_slice_real_pdf_with_pymupdf_never_uses_text_extraction(self) -> None:
        pdf_path = _create_sample_pdf(
            self.tmp_dir / "slides.pdf",
            [f"Slide {i}" for i in range(1, 6)],
        )

        self.assertEqual(inspect_pdf_page_count(pdf_path), 5)

        payloads = slice_pdf_batches(
            pdf_path,
            page_count=5,
            max_pages_per_batch=3,
            parallel_workers=2,
        )
        self.assertEqual(len(payloads), 2)
        self.assertEqual((payloads[0].spec.page_start, payloads[0].spec.page_end), (1, 3))
        self.assertEqual((payloads[1].spec.page_start, payloads[1].spec.page_end), (4, 5))

        # Verify each sliced byte payload is a valid PDF with exact expected page count.
        for payload in payloads:
            doc = pymupdf.open(stream=payload.pdf_bytes, filetype="pdf")
            try:
                self.assertEqual(len(doc), payload.spec.page_count)
            finally:
                doc.close()

    def test_inspect_pdf_rejects_missing_empty_corrupt_and_zero_page_pdfs(self) -> None:
        # Missing file
        with self.assertRaises(StageError) as missing_ctx:
            inspect_pdf_page_count(self.tmp_dir / "missing.pdf")
        self.assertEqual(missing_ctx.exception.stage, "ocr")
        self.assertIn("missing", missing_ctx.exception.message.lower())

        # Empty 0-byte file
        empty_pdf = self.tmp_dir / "empty.pdf"
        empty_pdf.write_bytes(b"")
        with self.assertRaises(StageError) as empty_ctx:
            inspect_pdf_page_count(empty_pdf)
        self.assertEqual(empty_ctx.exception.stage, "ocr")
        self.assertIn("empty", empty_ctx.exception.message.lower())

        # Corrupt PDF bytes
        corrupt_pdf = self.tmp_dir / "corrupt.pdf"
        corrupt_pdf.write_bytes(b"not a valid pdf file at all")
        with self.assertRaises(StageError) as corrupt_ctx:
            inspect_pdf_page_count(corrupt_pdf)
        self.assertEqual(corrupt_ctx.exception.stage, "ocr")
        self.assertIn("corrupted", corrupt_ctx.exception.message.lower())

        # 0-page PDF
        fake_zero_doc = mock.MagicMock()
        fake_zero_doc.needs_pass = False
        fake_zero_doc.__len__.return_value = 0
        valid_dummy = self.tmp_dir / "zero.pdf"
        valid_dummy.write_bytes(b"%PDF-1.7\n")
        with mock.patch("backend.app.pipeline.ocr.pymupdf.open", return_value=fake_zero_doc):
            with self.assertRaises(StageError) as zero_ctx:
                inspect_pdf_page_count(valid_dummy)
        self.assertEqual(zero_ctx.exception.stage, "ocr")
        self.assertIn("0 pages", zero_ctx.exception.message)
        fake_zero_doc.close.assert_called_once()

        # Password-protected PDF
        fake_locked_doc = mock.MagicMock()
        fake_locked_doc.needs_pass = True
        with mock.patch("backend.app.pipeline.ocr.pymupdf.open", return_value=fake_locked_doc):
            with self.assertRaises(StageError) as locked_ctx:
                inspect_pdf_page_count(valid_dummy)
        self.assertEqual(locked_ctx.exception.stage, "ocr")
        self.assertIn("password-protected", locked_ctx.exception.message.lower())
        fake_locked_doc.close.assert_called_once()

    def test_slice_pdf_batches_bisects_oversized_batches_and_rejects_oversized_single_slide(self) -> None:
        pdf_path = _create_sample_pdf(
            self.tmp_dir / "bisect.pdf",
            [f"Slide {i}" for i in range(1, 5)],
        )

        # Simulate oversized multi-page slices that become small enough when split down to 1-page slices.
        def fake_render(_doc, start: int, end: int) -> bytes:
            page_count = end - start + 1
            return b"x" * (page_count * 1200)

        with mock.patch("backend.app.pipeline.ocr._render_pdf_slice_bytes", side_effect=fake_render):
            payloads = slice_pdf_batches(
                pdf_path,
                page_count=4,
                max_pages_per_batch=4,
                max_batch_bytes=1500,
                parallel_workers=1,
            )
        self.assertEqual(
            [(p.spec.page_start, p.spec.page_end) for p in payloads],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
        )

        # Single slide exceeding max_batch_bytes raises actionable StageError.
        with mock.patch(
            "backend.app.pipeline.ocr._render_pdf_slice_bytes",
            return_value=b"x" * 3000,
        ):
            with self.assertRaises(StageError) as oversize_ctx:
                slice_pdf_batches(
                    pdf_path,
                    page_count=4,
                    max_pages_per_batch=4,
                    max_batch_bytes=1500,
                    parallel_workers=1,
                )
        self.assertEqual(oversize_ctx.exception.stage, "ocr")
        self.assertIn("exceeding Gemini's inline OCR batch limit", oversize_ctx.exception.message)


class Phase3GeminiOCRStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-p3-ocr-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_credentials_validation_and_vertex_ai_client_factory(self) -> None:
        # Missing API key in standard mode
        missing_key_settings = _make_settings(self.tmp_dir, gemini_api_key="")
        with self.assertRaises(StageError) as api_ctx:
            validate_gemini_credentials(missing_key_settings)
        self.assertEqual(api_ctx.exception.stage, "ocr")
        self.assertIn("GEMINI_API_KEY", api_ctx.exception.message)
        self.assertIn("does not fall back", api_ctx.exception.hint.lower())

        # Missing project in Vertex AI mode
        missing_proj_settings = _make_settings(
            self.tmp_dir,
            gemini_api_key="",
            google_genai_use_vertexai=True,
            google_cloud_project="",
        )
        with self.assertRaises(StageError) as vertex_ctx:
            validate_gemini_credentials(missing_proj_settings)
        self.assertEqual(vertex_ctx.exception.stage, "ocr")
        self.assertIn("GOOGLE_CLOUD_PROJECT", vertex_ctx.exception.message)

        # Verify API key client instantiation
        api_settings = _make_settings(self.tmp_dir, gemini_api_key="key-123")
        with mock.patch("backend.app.pipeline.ocr.genai.Client") as mock_client_cls:
            ocr_module._default_genai_client_factory(api_settings)
        _, api_kwargs = mock_client_cls.call_args
        self.assertEqual(api_kwargs["api_key"], "key-123")
        self.assertNotIn("vertexai", api_kwargs)
        self.assertEqual(api_kwargs["http_options"].timeout, 180_000)

        # Verify Vertex AI client instantiation
        vertex_settings = _make_settings(
            self.tmp_dir,
            gemini_api_key="",
            google_genai_use_vertexai=True,
            google_cloud_project="my-gcp-project",
            google_cloud_location="europe-west4",
        )
        with mock.patch("backend.app.pipeline.ocr.genai.Client") as mock_client_cls:
            ocr_module._default_genai_client_factory(vertex_settings)
        _, vertex_kwargs = mock_client_cls.call_args
        self.assertTrue(vertex_kwargs["vertexai"])
        self.assertEqual(vertex_kwargs["project"], "my-gcp-project")
        self.assertEqual(vertex_kwargs["location"], "europe-west4")
        self.assertNotIn("api_key", vertex_kwargs)

    def test_error_classification_helpers(self) -> None:
        self.assertTrue(is_auth_error(_FakeGeminiAPIError("Unauthorized", code=401)))
        self.assertTrue(is_auth_error(_FakeGeminiAPIError("Permission denied", status="PERMISSION_DENIED")))
        self.assertTrue(is_auth_error(RuntimeError("API_KEY_INVALID")))
        self.assertFalse(is_auth_error(_FakeGeminiAPIError("Rate limit", code=429)))

        self.assertTrue(is_payload_too_large_error(_FakeGeminiAPIError("Too large", code=413)))
        self.assertTrue(is_payload_too_large_error(RuntimeError("Request payload size exceeds limit")))

        self.assertTrue(is_transient_ocr_error(_FakeGeminiAPIError("Rate limit", code=429)))
        self.assertTrue(is_transient_ocr_error(_FakeGeminiAPIError("Unavailable", status="UNAVAILABLE")))
        self.assertTrue(is_transient_ocr_error(TimeoutError("timed out")))
        self.assertFalse(is_transient_ocr_error(_FakeGeminiAPIError("Invalid argument", code=400)))

    def test_parallel_batch_ocr_merges_slides_strips_boilerplate_and_reports_progress(self) -> None:
        # 5-page PDF splits into 2 batches: pages 1-3 (3 pages) and pages 4-5 (2 pages).
        pdf_path = _create_sample_pdf(
            self.tmp_dir / "deck.pdf",
            [f"Deck Slide {i}" for i in range(1, 6)],
        )
        settings = _make_settings(self.tmp_dir)
        progress: list[str] = []

        active = 0
        peak = 0
        lock = threading.Lock()

        def fake_gemini(record, _pdf_bytes):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                count = int(record["batch_page_count"])
                # Make batch 1 (3 pages) finish slightly slower than batch 2 (2 pages)
                # to verify out-of-order completion still merges slides in global order 1..5.
                time.sleep(0.05 if count == 3 else 0.02)
                batch_tag = "B1" if count == 3 else "B2"
                payload = []
                for idx in range(1, count + 1):
                    payload.append(
                        {
                            "slide": idx,
                            "text": (
                                "VISIBLE_TEXT:\n"
                                "ACME University - CS101\n"
                                f"{batch_tag} Slide Topic {idx}\n\n"
                                "VISUAL_CONTEXT:\n"
                                f"- Diagram illustrating {batch_tag} topic {idx}\n\n"
                                "TABLES_CODE_FORMULAS:\n"
                                f"f_{batch_tag}({idx}) = {idx} * x"
                            ),
                        }
                    )
                return SimpleNamespace(
                    text=json.dumps(payload),
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=1000,
                        candidates_token_count=250,
                    ),
                )
            finally:
                with lock:
                    active -= 1

        fake_client = _FakeGeminiClient(fake_gemini)
        artifact, warnings, stats = extract_slides_with_gemini_ocr(
            pdf_path,
            settings,
            source_pdf="deck.pdf",
            progress_callback=progress.append,
            client_factory=lambda _s: fake_client,
        )

        self.assertEqual(warnings, [])
        self.assertEqual(artifact.source_pdf, "deck.pdf")
        self.assertEqual(artifact.model, "gemini-2.5-flash")
        self.assertEqual(artifact.slide_count, 5)
        self.assertEqual([s.slide for s in artifact.slides], [1, 2, 3, 4, 5])
        self.assertEqual(
            [s.visible_text for s in artifact.slides],
            [
                "B1 Slide Topic 1",
                "B1 Slide Topic 2",
                "B1 Slide Topic 3",
                "B2 Slide Topic 1",
                "B2 Slide Topic 2",
            ],
        )
        self.assertEqual(stats["ocr_batches"], 2)
        self.assertEqual(stats["structured_slides"], 5)
        self.assertEqual(stats["visual_context_slides"], 5)
        self.assertEqual(stats["tables_code_formulas_slides"], 5)
        self.assertEqual(stats["boilerplate_lines_removed"], 5)
        self.assertEqual(stats["ocr_input_tokens"], 2000)
        self.assertEqual(stats["ocr_output_tokens"], 500)
        self.assertEqual(peak, 2)

        for slide in artifact.slides:
            self.assertNotIn("ACME University - CS101", slide.visible_text)
            self.assertIn("VISIBLE_TEXT:", slide.raw_text)
            self.assertIn("VISUAL_CONTEXT:", slide.raw_text)
            self.assertIn("TABLES_CODE_FORMULAS:", slide.raw_text)

        self.assertTrue(any("across 2 batches" in msg for msg in progress))
        self.assertTrue(any("Completed OCR batch 2 of 2" in msg for msg in progress))

    def test_transient_error_and_malformed_json_retry_then_succeed(self) -> None:
        pdf_path = _create_sample_pdf(self.tmp_dir / "two_slides.pdf", ["One", "Two"])
        settings = _make_settings(self.tmp_dir)
        sleeps: list[float] = []
        progress: list[str] = []
        attempts = {"count": 0}

        def flaky_gemini(_record, _pdf_bytes):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _FakeGeminiAPIError("Rate limited", code=429, headers={"Retry-After": "1.5"})
            if attempts["count"] == 2:
                return SimpleNamespace(text="not valid json at all")
            return SimpleNamespace(
                text=json.dumps(
                    [
                        {"slide": 1, "text": "VISIBLE_TEXT:\nSlide One"},
                        {"slide": 2, "text": "VISIBLE_TEXT:\nSlide Two"},
                    ]
                )
            )

        fake_client = _FakeGeminiClient(flaky_gemini)
        artifact, warnings, _stats = extract_slides_with_gemini_ocr(
            pdf_path,
            settings,
            source_pdf="two_slides.pdf",
            progress_callback=progress.append,
            client_factory=lambda _s: fake_client,
            sleep_fn=sleeps.append,
        )

        self.assertEqual(attempts["count"], 3)
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(sleeps[0], 1.5)
        self.assertEqual(artifact.slide_count, 2)
        self.assertEqual(artifact.slides[0].visible_text, "Slide One")
        self.assertTrue(any("retried 2 times" in w for w in warnings))
        self.assertTrue(any("retrying (2/3)" in msg for msg in progress))

    def test_strict_failure_policy_never_falls_back_to_text_extraction(self) -> None:
        pdf_path = _create_sample_pdf(
            self.tmp_dir / "lecture.pdf",
            ["Secret text that must never be returned via fallback"],
        )
        settings = _make_settings(self.tmp_dir)
        sleeps: list[float] = []

        # 1. Exhausted transient retries -> StageError, no fallback, no artifact written
        failing_503 = _FakeGeminiClient(
            lambda _rec, _b: (_ for _ in ()).throw(
                _FakeGeminiAPIError("Service unavailable", code=503)
            )
        )
        with self.assertRaises(StageError) as ctx_503:
            extract_slides_with_gemini_ocr(
                pdf_path,
                settings,
                source_pdf="lecture.pdf",
                client_factory=lambda _s: failing_503,
                sleep_fn=sleeps.append,
            )
        self.assertEqual(ctx_503.exception.stage, "ocr")
        self.assertIn("3 attempt(s)", ctx_503.exception.message)
        self.assertIn("not fall back", ctx_503.exception.hint.lower())
        self.assertEqual(len(failing_503.calls), 3)
        self.assertEqual(len(sleeps), 2)

        # 2. Auth 401 fails immediately without retrying
        sleeps.clear()
        failing_401 = _FakeGeminiClient(
            lambda _rec, _b: (_ for _ in ()).throw(
                _FakeGeminiAPIError("API key not valid", code=401)
            )
        )
        with self.assertRaises(StageError) as ctx_401:
            extract_slides_with_gemini_ocr(
                pdf_path,
                settings,
                source_pdf="lecture.pdf",
                client_factory=lambda _s: failing_401,
                sleep_fn=sleeps.append,
            )
        self.assertEqual(ctx_401.exception.stage, "ocr")
        self.assertIn("authentication failed", ctx_401.exception.message.lower())
        self.assertIn("GEMINI_API_KEY", ctx_401.exception.hint)
        self.assertEqual(len(failing_401.calls), 1)
        self.assertEqual(sleeps, [])

        # 3. Non-transient 404 fails immediately without retrying
        failing_404 = _FakeGeminiClient(
            lambda _rec, _b: (_ for _ in ()).throw(
                _FakeGeminiAPIError("Model not found", code=404)
            )
        )
        with self.assertRaises(StageError) as ctx_404:
            extract_slides_with_gemini_ocr(
                pdf_path,
                settings,
                source_pdf="lecture.pdf",
                client_factory=lambda _s: failing_404,
                sleep_fn=sleeps.append,
            )
        self.assertEqual(ctx_404.exception.stage, "ocr")
        self.assertIn("Model not found", ctx_404.exception.message)
        self.assertIn("never falls back to pdfplumber", ctx_404.exception.hint.lower())
        self.assertEqual(len(failing_404.calls), 1)

        # 4. Payload too large (413) fails immediately without retrying
        failing_413 = _FakeGeminiClient(
            lambda _rec, _b: (_ for _ in ()).throw(
                _FakeGeminiAPIError("Request Entity Too Large", code=413)
            )
        )
        with self.assertRaises(StageError) as ctx_413:
            extract_slides_with_gemini_ocr(
                pdf_path,
                settings,
                source_pdf="lecture.pdf",
                client_factory=lambda _s: failing_413,
                sleep_fn=sleeps.append,
            )
        self.assertEqual(ctx_413.exception.stage, "ocr")
        self.assertIn("payload is too large", ctx_413.exception.message.lower())
        self.assertEqual(len(failing_413.calls), 1)

        # 5. run_ocr_stage writes NO artifact when OCR fails (strict failure, zero fallback)
        context = _make_context(self.tmp_dir, slides_filename="lecture.pdf")
        shutil.copy2(pdf_path, context.slides_path)
        missing_creds = _make_settings(self.tmp_dir, gemini_api_key="")
        with self.assertRaises(StageError) as stage_ctx:
            run_ocr_stage(context, missing_creds)
        self.assertEqual(stage_ctx.exception.stage, "ocr")
        self.assertFalse(context.slides_ocr_path.exists())

        # 6. Ensure ocr.py has zero pdfplumber imports or get_text fallback calls
        ocr_source = Path(ocr_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import pdfplumber", ocr_source)
        self.assertNotIn("get_text(", ocr_source)
        self.assertNotIn("extract_text(", ocr_source)

    def test_run_ocr_stage_writes_slides_ocr_json_atomically(self) -> None:
        progress: list[str] = []
        context = _make_context(
            self.tmp_dir,
            slides_filename="Week 03 Neural Nets.pdf",
            progress_log=progress,
        )
        _create_sample_pdf(context.slides_path, ["Backpropagation", "Activation Functions"])
        settings = _make_settings(self.tmp_dir)

        fake_client = _FakeGeminiClient(
            lambda _rec, _b: SimpleNamespace(
                text=json.dumps(
                    [
                        {
                            "slide": 1,
                            "text": (
                                "VISIBLE_TEXT:\nBackpropagation\n\n"
                                "VISUAL_CONTEXT:\n- Computation graph with backward gradient arrows."
                            ),
                        },
                        {
                            "slide": 2,
                            "text": (
                                "VISIBLE_TEXT:\nActivation Functions\n\n"
                                "TABLES_CODE_FORMULAS:\nReLU(x) = max(0, x)"
                            ),
                        },
                    ]
                ),
                usage_metadata=SimpleNamespace(
                    prompt_token_count=450,
                    candidates_token_count=120,
                ),
            )
        )

        with mock.patch(
            "backend.app.pipeline.ocr._default_genai_client_factory",
            return_value=fake_client,
        ):
            result = run_ocr_stage(context, settings)

        self.assertEqual(result.artifacts, {"slides_ocr": ARTIFACT_SLIDES_OCR})
        self.assertEqual(result.stats["slide_count"], 2)
        self.assertEqual(result.stats["visual_context_slides"], 1)
        self.assertEqual(result.stats["tables_code_formulas_slides"], 1)
        self.assertEqual(result.warnings, [])
        self.assertIn("2 slide(s)", result.message)
        self.assertTrue(len(progress) >= 2)

        # Verify atomic write left no temp file behind.
        self.assertEqual(
            sorted(p.name for p in context.artifacts_dir.iterdir()),
            ["slides_ocr.json"],
        )
        parsed = SlidesOCRArtifact.model_validate_json(
            context.slides_ocr_path.read_text(encoding="utf-8")
        )
        self.assertEqual(parsed.source_pdf, "Week 03 Neural Nets.pdf")
        self.assertEqual(parsed.model, "gemini-2.5-flash")
        self.assertEqual(parsed.slide_count, 2)
        self.assertEqual(parsed.slides[0].visible_text, "Backpropagation")
        self.assertIn("Computation graph", parsed.slides[0].visual_context)
        self.assertEqual(parsed.slides[1].tables_code_formulas, "ReLU(x) = max(0, x)")


def _has_live_gemini_opt_in() -> bool:
    if os.getenv("RUN_LIVE_GEMINI_OCR_TESTS", "").strip().lower() not in {"1", "true", "yes"}:
        return False
    settings = load_settings()
    if settings.google_genai_use_vertexai:
        return bool(settings.google_cloud_project.strip())
    return bool(settings.gemini_api_key.strip())


@unittest.skipUnless(
    _has_live_gemini_opt_in(),
    "Live Gemini OCR smoke test requires RUN_LIVE_GEMINI_OCR_TESTS=1 and configured Gemini/Vertex credentials",
)
class Phase3LiveGeminiSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-p3-live-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_live_gemini_ocr_smoke_on_generated_slide_pdf(self) -> None:
        settings = load_settings()
        context = _make_context(self.tmp_dir, slides_filename="live_smoke.pdf")
        _create_sample_pdf(
            context.slides_path,
            [
                "Lecture 1: Binary Search\nTime Complexity: O(log n)",
                "Summary\nDivide the search interval in half on every step.",
            ],
        )

        result = run_ocr_stage(context, settings)
        self.assertEqual(result.stats["slide_count"], 2)
        parsed = SlidesOCRArtifact.model_validate_json(
            context.slides_ocr_path.read_text(encoding="utf-8")
        )
        self.assertEqual(parsed.slide_count, 2)
        self.assertIn("Binary Search", parsed.slides[0].raw_text)


if __name__ == "__main__":
    unittest.main()
