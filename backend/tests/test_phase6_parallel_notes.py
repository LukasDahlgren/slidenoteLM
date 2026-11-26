from __future__ import annotations

import json
import os
import re
import unittest
from concurrent.futures import Future
from pathlib import Path
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.config import load_settings
from backend.app.pipeline.notes import generate_notes
from backend.app.pipeline.ocr import plan_pdf_batches
from backend.app.pipeline.prompts import build_section_notes_prompt
from backend.app.schemas import (
    AlignmentArtifact, AlignmentEntry, SlideOCRRecord, SlidesOCRArtifact, StageError,
    TranscriptArtifact, TranscriptSegment,
)
from backend.tests.test_phase6_gemini_llm import _FakeTextClient, _make_settings


def lecture(numbers, durations=None):
    durations = durations or [10] * len(numbers)
    slides = sorted({n for n in numbers if n is not None})
    deck = SlidesOCRArtifact(
        source_pdf="deck.pdf", slide_count=len(slides),
        slides=[SlideOCRRecord(slide=n, visible_text=f"Topic {n}") for n in reversed(slides)],
    )
    segments, entries = [], []
    start = 0
    for i, (n, duration) in enumerate(zip(numbers, durations)):
        end = start + duration
        segments.append(TranscriptSegment(index=i, start=start, end=end, text=f"Explanation {i}"))
        entries.append(AlignmentEntry(slide=n, start_segment=i, end_segment=i,
                                      start_time=start, end_time=end))
        start = end
    return (deck, TranscriptArtifact(source_recording="speech.wav", segments=segments,
                                     duration_seconds=start), AlignmentArtifact(entries=entries))


class ParallelNotesTests(unittest.TestCase):
    def test_settings(self):
        defaults = {
            "LLM_NOTES_PARALLEL_WORKERS": (9, 1), "LLM_COVERAGE_PARALLEL_WORKERS": (4, 1),
            "OCR_MAX_PAGES_PER_BATCH": (8, 1), "OCR_PARALLEL_WORKERS": (8, 1),
            "OCR_GLOBAL_MAX_CONCURRENT": (16, 1), "LLM_GLOBAL_MAX_CONCURRENT": (24, 1),
            "NOTES_MAX_SLIDES_PER_SECTION": (8, 1), "NOTES_MAX_SECTION_DURATION_SECONDS": (900, 30),
            "NOTES_MAX_SECTION_TRANSCRIPT_CHARS": (14000, 500),
            "NOTES_INLINE_OFF_SLIDE_MAX_SECONDS": (35, 0), "NOTES_INLINE_OFF_SLIDE_MAX_CHARS": (300, 0),
            "NOTES_MIN_AUDIT_TRANSCRIPT_CHARS": (200, 0),
        }
        for override in (None, "1337", "-1"):
            with self.subTest(override=override), patch.dict(
                os.environ, {k: override for k in defaults if override is not None}, clear=True
            ):
                cfg = load_settings(env_file=Path("absent.env"))
                for key, (default, minimum) in defaults.items():
                    expected = default if override is None else max(minimum, int(override))
                    self.assertEqual(getattr(cfg, key.lower()), expected)
                self.assertEqual((cfg.llm_notes_max_output_tokens, cfg.llm_rate_limit_rpm), (49152, 90))

    def test_pdf_batch_sizes(self):
        for count, expected in ((5, [3, 2]), (51, [8, 8, 7, 7, 7, 7, 7])):
            batches = plan_pdf_batches(count)
            self.assertEqual([b.page_end - b.page_start + 1 for b in batches], expected)

    def test_flanked_short_off_slide_segment_keeps_its_anchor_and_order(self):
        result = generate_notes(
            *lecture([None, 1, None, 2, None, 3, None], [10, 10, 35, 10, 36, 10, 10]),
            _make_settings(Path("."), pipeline_llm_mode="stub"),
        )
        self.assertEqual([s.kind for s in result.sections],
                         ["off_slide", "aligned", "off_slide", "aligned", "off_slide"])
        self.assertEqual([s.slide for s in result.sections[1].segments], [1, None, 2])
        self.assertIn("[Off-slide]: Explanation 2", build_section_notes_prompt(result.prompt_inputs[1]).user)
        # Stub Markdown preserves chronological order (00:10 -> 00:20 [Off-slide] -> 00:55) and [Off-slide] label.
        md = result.markdown
        self.assertIn("[00:20–00:55] [Off-slide] Explanation 2", md)
        self.assertLess(md.index("Explanation 1"), md.index("Explanation 2"))
        self.assertLess(md.index("Explanation 2"), md.index("Explanation 3"))

        # Adjacent short off-slide entries between aligned slides are evaluated as one combined run.
        combined = generate_notes(
            *lecture([1, None, None, 2], [10, 10, 10, 10]),
            _make_settings(Path("."), pipeline_llm_mode="stub"),
        )
        self.assertEqual([s.kind for s in combined.sections], ["aligned"])
        self.assertEqual([s.slide for s in combined.sections[0].segments], [1, None, None, 2])

    def test_concurrent_drafts_pipeline_audits_and_assemble_in_order(self):
        rendezvous = Barrier(2)
        second_audit_started = Event()
        lock = Lock()
        active = peak = 0

        def respond(*, model, contents, config):
            nonlocal active, peak
            idx = int(re.search(r"Section: (\d+) of", contents)[1])
            if "### Source Transcript Segments" in contents:
                if idx == 2:
                    second_audit_started.set()
                    raise RuntimeError("audit offline")
                response = json.dumps({"passed": True, "covered_segment_count": 1})
            else:
                with lock:
                    active += 1
                    peak = max(peak, active)
                rendezvous.wait(timeout=5)
                if idx == 1:
                    self.assertTrue(second_audit_started.wait(5))
                with lock:
                    active -= 1
                response = f"## Finished section {idx}"
            return _FakeTextClient([response]).models.generate_content(
                model=model, contents=contents, config=config,
            )

        client = SimpleNamespace(models=SimpleNamespace(generate_content=respond))
        progress = []
        result = generate_notes(
            *lecture([1, 1, 1, 1, 1, 2]),
            _make_settings(Path("."), notes_max_slides_per_section=1, notes_min_audit_transcript_chars=0,
                           llm_retry_attempts=1, llm_rate_limit_rpm=0),
            client_factory=lambda _: client,
            progress_fraction_callback=progress.append,
        )
        self.assertEqual(progress, [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(peak, 2)
        self.assertLess(result.markdown.index("## Finished section 1"), result.markdown.index("## Finished section 2"))
        first, second = result.prompt_inputs
        self.assertEqual(first.lecture_outline, ("Slide 1: Topic 1", "Slide 2: Topic 2"))
        self.assertEqual(second.lecture_outline, first.lecture_outline)
        self.assertEqual(first.prior_boundary_segments, ())
        self.assertEqual([s.index for s in second.prior_boundary_segments], [1, 2, 3, 4])
        self.assertEqual(second.rolling_context.prior_section_headings, ["Topic 1"])
        self.assertIn("Topic", second.rolling_context.key_terms)
        self.assertEqual((second.rolling_context.last_slide, second.rolling_context.last_timestamp), (1, 50))
        self.assertFalse(result.coverage_result.passed)
        self.assertTrue(any("Coverage audit unavailable on section 2" in w for w in result.warnings))
        self.assertEqual(result.notes_input_tokens, 2400)

    def test_small_sections_skip_api_audit_without_false_certification(self):
        for workers in (1, 2):
            client = _FakeTextClient(["## Draft"] * 2)
            progress = []
            result = generate_notes(
                *lecture([1, 2]), _make_settings(Path("."), notes_max_slides_per_section=1,
                                                 llm_notes_parallel_workers=workers, llm_rate_limit_rpm=0),
                client_factory=lambda _: client,
                progress_fraction_callback=progress.append,
            )
            self.assertEqual(progress, [0.5, 1.0])
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(result.coverage_input_tokens, 0)
            self.assertEqual(result.coverage_result.checked_sections, 0)
            self.assertEqual(result.coverage_result.covered_segment_count, 0)
            self.assertIn("unaudited", result.coverage_result.notes)

    def test_uncertain_and_off_slide_parallel_prompts_omit_numbered_outline(self):
        deck, transcript, alignment = lecture([1, 2, None], [10, 10, 60])
        alignment = AlignmentArtifact(
            entries=[
                alignment.entries[0],
                alignment.entries[1].model_copy(update={"uncertain": True, "reason": "ambiguous"}),
                alignment.entries[2],
            ]
        )
        client = _FakeTextClient(["## Sec 1", "## Sec 2", "## Sec 3"])
        result = generate_notes(
            deck,
            transcript,
            alignment,
            _make_settings(
                Path("."),
                notes_max_slides_per_section=1,
                llm_notes_parallel_workers=3,
                llm_rate_limit_rpm=0,
            ),
            client_factory=lambda _: client,
        )
        aligned_prompt, uncertain_prompt, off_slide_prompt = result.prompt_inputs
        self.assertEqual(aligned_prompt.lecture_outline, ("Slide 1: Topic 1", "Slide 2: Topic 2"))
        self.assertEqual(uncertain_prompt.lecture_outline, ())
        self.assertEqual(off_slide_prompt.lecture_outline, ())
        uncertain_user = build_section_notes_prompt(uncertain_prompt).user
        self.assertNotIn("Slide 2: Topic 2", uncertain_user)
        self.assertNotIn("[Slide 1]", uncertain_user)

    def test_injected_coverage_callable_runs_sequentially_in_order(self):
        seen: list[int] = []
        active = peak = 0
        lock = Lock()

        def coverage_cb(_messages, payload, _settings):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                seen.append(payload.section_index)
            with lock:
                active -= 1
            return {"passed": True, "covered_segment_count": len(payload.segments)}

        client = _FakeTextClient(["## Section 1", "## Section 2", "## Section 3"])
        generate_notes(
            *lecture([1, 2, 3]),
            _make_settings(
                Path("."),
                notes_max_slides_per_section=1,
                llm_notes_parallel_workers=3,
                llm_rate_limit_rpm=0,
            ),
            coverage_llm_callable=coverage_cb,
            client_factory=lambda _: client,
        )
        self.assertEqual(peak, 1)
        self.assertEqual(seen, [1, 2, 3])

    def test_draft_failure_interrupts_sibling_retry_backoff_immediately(self):
        from google.genai import errors as genai_errors

        retrying_started = Event()
        call_counts = {1: 0, 2: 0}

        def respond(*, model, contents, config):
            idx = int(re.search(r"Section: (\d+) of", contents)[1])
            call_counts[idx] += 1
            if idx == 1:
                retrying_started.set()
                raise genai_errors.ClientError(
                    429,
                    {
                        "error": {
                            "code": 429,
                            "message": "Quota exceeded",
                            "status": "RESOURCE_EXHAUSTED",
                            "details": [
                                {
                                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                                    "retryDelay": "30s",
                                }
                            ],
                        }
                    },
                )
            self.assertTrue(retrying_started.wait(5))
            raise genai_errors.ClientError(
                401, {"error": {"code": 401, "message": "Invalid key", "status": "UNAUTHENTICATED"}}
            )

        client = SimpleNamespace(models=SimpleNamespace(generate_content=respond))
        import time as _time

        t0 = _time.monotonic()
        with self.assertRaises(StageError):
            generate_notes(
                *lecture([1, 2]),
                _make_settings(
                    Path("."),
                    notes_max_slides_per_section=1,
                    llm_notes_parallel_workers=2,
                    llm_retry_attempts=3,
                    llm_retry_base_delay_seconds=1.0,
                    llm_rate_limit_rpm=0,
                ),
                client_factory=lambda _: client,
            )
        elapsed = _time.monotonic() - t0
        # Must interrupt the 30s retry wait quickly rather than sleeping 30s.
        self.assertLess(elapsed, 5.0)
        self.assertEqual(call_counts[1], 1)

    def test_draft_failure_cancels_queued_draft_and_audit(self):
        class ControlledPool:
            def __init__(self, *, max_workers):
                self.tasks = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, fn, *args):
                future = Future()
                self.tasks.append(future)
                return future

        writers = ControlledPool(max_workers=2)
        reviewers = ControlledPool(max_workers=1)

        def complete(futures):
            first, failed, pending = list(futures)
            first.set_result(("## First", None))
            yield first
            failed.set_exception(StageError(stage="notes", message="draft failure", hint="Retry"))
            yield failed

        with patch("backend.app.pipeline.notes.ThreadPoolExecutor", side_effect=[writers, reviewers]), \
             patch("backend.app.pipeline.notes.as_completed", side_effect=complete), \
             self.assertRaisesRegex(StageError, "draft failure"):
            generate_notes(*lecture([1, 2, 3]), _make_settings(
                Path("."), notes_max_slides_per_section=1, notes_min_audit_transcript_chars=0,
            ), client_factory=lambda _: object())
        self.assertTrue(writers.tasks[-1].cancelled())
        self.assertTrue(reviewers.tasks[0].cancelled())
