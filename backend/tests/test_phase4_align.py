from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.config import Settings
from backend.app.pipeline.align import (
    align_deterministic_stub,
    align_with_llm,
    alignment_issues,
    annotate_alignment_entries,
    check_alignment_quality,
    format_alignment_warnings,
    load_slides_ocr_artifact,
    load_transcript_artifact,
    normalize_alignment_entries,
    run_alignment_stage,
)
from backend.app.schemas import (
    ARTIFACT_ALIGNMENT,
    AlignmentArtifact,
    AlignmentEntry,
    AlignmentIssue,
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
        "pipeline_llm_mode": "stub",
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def _make_context(
    tmp_dir: Path,
    *,
    progress_log: list[str] | None = None,
) -> JobContext:
    job_dir = tmp_dir / "job-align"
    input_dir = job_dir / "input"
    artifacts_dir = job_dir / "artifacts"
    input_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    slides_path = input_dir / "slides.pdf"
    slides_path.write_bytes(b"%PDF-1.7\n")
    recording_path = input_dir / "recording.mp3"
    recording_path.write_bytes(b"fake-audio")
    callback = progress_log.append if progress_log is not None else None
    return JobContext(
        token="tok-phase4-align",
        slides_filename="lecture.pdf",
        recording_filename="lecture.mp3",
        job_dir=job_dir,
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        slides_path=slides_path,
        recording_path=recording_path,
        progress_callback=callback,
    )


def _make_slides(texts: list[str], *, start_num: int = 1) -> SlidesOCRArtifact:
    records = [
        SlideOCRRecord(
            slide=start_num + idx,
            visible_text=text,
            visual_context="",
            tables_code_formulas="",
            raw_text=text,
        )
        for idx, text in enumerate(texts)
    ]
    return SlidesOCRArtifact(
        source_pdf="lecture.pdf",
        model="gemini-2.5-flash",
        slide_count=len(records),
        slides=records,
    )


def _make_transcript(
    texts: list[str],
    *,
    segment_seconds: float = 15.0,
) -> TranscriptArtifact:
    segments = [
        TranscriptSegment(
            index=idx,
            start=round(idx * segment_seconds, 2),
            end=round((idx + 1) * segment_seconds, 2),
            text=text,
        )
        for idx, text in enumerate(texts)
    ]
    duration = round(len(segments) * segment_seconds, 2)
    return TranscriptArtifact(
        source_recording="lecture.mp3",
        model="whisper-large-v3-turbo",
        duration_seconds=duration,
        segments=segments,
    )


class TestAlignmentQualityChecks(unittest.TestCase):
    def test_clean_monotonic_alignment_produces_no_issues(self) -> None:
        slides = _make_slides(
            [
                "Gradient descent optimization basics for neural networks and convex loss functions",
                "Backpropagation chain rule derivatives across hidden layers in deep learning",
                "Regularization techniques including weight decay dropout and early stopping",
            ]
        )
        transcript = _make_transcript(
            [
                "First we discuss gradient descent optimization.",
                "We minimize the convex loss function iteratively.",
                "Next we apply backpropagation using the chain rule.",
                "Gradients flow backward through hidden layers.",
                "Finally regularization prevents overfitting.",
                "Dropout and weight decay improve generalization.",
            ],
            segment_seconds=15.0,
        )
        entries = [
            AlignmentEntry(slide=1, start_segment=0, end_segment=1, start_time=0.0, end_time=30.0),
            AlignmentEntry(slide=2, start_segment=2, end_segment=3, start_time=30.0, end_time=60.0),
            AlignmentEntry(slide=3, start_segment=4, end_segment=5, start_time=60.0, end_time=90.0),
        ]

        issues = check_alignment_quality(entries, transcript, slides)
        self.assertEqual(issues, [])
        self.assertEqual(alignment_issues(entries, transcript, slides), [])

    def test_out_of_order_slide_jump_and_regression_marks_uncertain(self) -> None:
        slides = _make_slides(["Slide 1", "Slide 2", "Slide 3", "Slide 4"])
        transcript = _make_transcript([f"Segment {i}" for i in range(8)], segment_seconds=12.0)
        entries = [
            AlignmentEntry(slide=1, start_segment=0, end_segment=1, start_time=0.0, end_time=24.0),
            AlignmentEntry(slide=4, start_segment=2, end_segment=3, start_time=24.0, end_time=48.0),
            AlignmentEntry(slide=2, start_segment=4, end_segment=5, start_time=48.0, end_time=72.0),
            AlignmentEntry(slide=3, start_segment=6, end_segment=7, start_time=72.0, end_time=96.0),
        ]

        annotated, issues = annotate_alignment_entries(entries, transcript, slides)
        codes = {issue.code for issue in issues}
        self.assertIn("out_of_order_slide", codes)

        # Slide 1 is clean; jump to 4, regression to 2, and still-below-max 3 are uncertain.
        self.assertFalse(annotated[0].uncertain)
        self.assertTrue(annotated[1].uncertain)
        self.assertTrue(annotated[2].uncertain)
        self.assertTrue(annotated[3].uncertain)
        self.assertIn("timestamp-only", annotated[2].reason.lower())
        self.assertIn("out-of-order", annotated[2].reason.lower())

    def test_duplicate_slide_flagged_and_marked_uncertain(self) -> None:
        slides = _make_slides(["Slide 1", "Slide 2", "Slide 3"])
        transcript = _make_transcript([f"Segment {i}" for i in range(8)], segment_seconds=12.0)
        entries = [
            AlignmentEntry(slide=1, start_segment=0, end_segment=1, start_time=0.0, end_time=24.0),
            AlignmentEntry(slide=2, start_segment=2, end_segment=3, start_time=24.0, end_time=48.0),
            AlignmentEntry(slide=3, start_segment=4, end_segment=5, start_time=48.0, end_time=72.0),
            AlignmentEntry(slide=2, start_segment=6, end_segment=7, start_time=72.0, end_time=96.0),
        ]

        annotated, issues = annotate_alignment_entries(entries, transcript, slides)
        codes = [issue.code for issue in issues]
        self.assertIn("duplicate_slide", codes)
        self.assertIn("out_of_order_slide", codes)
        self.assertFalse(annotated[0].uncertain)
        self.assertFalse(annotated[1].uncertain)
        self.assertFalse(annotated[2].uncertain)
        self.assertTrue(annotated[3].uncertain)
        self.assertIn("duplicate", annotated[3].reason.lower())

    def test_overlapping_and_non_monotonic_segment_ranges_detected(self) -> None:
        slides = _make_slides(["Slide 1", "Slide 2", "Slide 3"])
        transcript = _make_transcript([f"Segment {i}" for i in range(6)], segment_seconds=12.0)
        entries = [
            AlignmentEntry(slide=1, start_segment=0, end_segment=2, start_time=0.0, end_time=36.0),
            AlignmentEntry(slide=2, start_segment=2, end_segment=4, start_time=24.0, end_time=60.0),
            AlignmentEntry(slide=3, start_segment=1, end_segment=5, start_time=12.0, end_time=72.0),
        ]

        annotated, issues = annotate_alignment_entries(entries, transcript, slides)
        codes = {issue.code for issue in issues}
        self.assertIn("overlapping_segment_range", codes)
        self.assertIn("non_monotonic_segments", codes)
        self.assertTrue(all(entry.uncertain for entry in annotated))
        severities = {issue.code: issue.severity for issue in issues}
        self.assertEqual(severities["overlapping_segment_range"], "error")
        self.assertEqual(severities["non_monotonic_segments"], "error")

    def test_invalid_segment_range_and_unknown_slide_handling(self) -> None:
        slides = _make_slides(["Slide 1", "Slide 2"])
        transcript = _make_transcript(["Seg 0", "Seg 1", "Seg 2"], segment_seconds=12.0)
        entries = [
            AlignmentEntry(slide=1, start_segment=2, end_segment=1, start_time=24.0, end_time=12.0),
            AlignmentEntry(slide=99, start_segment=0, end_segment=5, start_time=0.0, end_time=60.0),
        ]

        annotated, issues = annotate_alignment_entries(entries, transcript, slides)
        codes = {issue.code for issue in issues}
        self.assertIn("invalid_segment_range", codes)
        self.assertIn("unknown_slide", codes)
        self.assertIn("uncovered_slides", codes)

        self.assertTrue(annotated[0].uncertain)
        # Unknown slide number must be cleared to slide=None for downstream timestamp-only fallback.
        self.assertIsNone(annotated[1].slide)
        self.assertTrue(annotated[1].uncertain)
        self.assertIn("does not exist", annotated[1].reason.lower())

    def test_zero_duration_short_substantive_and_empty_span_text(self) -> None:
        slides = _make_slides(
            [
                "This slide has more than eight substantive technical words explaining convolution",
                "Another substantive slide with more than eight technical words about pooling layers",
            ]
        )
        transcript = TranscriptArtifact(
            source_recording="lecture.mp3",
            duration_seconds=10.0,
            segments=[
                TranscriptSegment(index=0, start=0.0, end=0.0, text="   "),
                TranscriptSegment(index=1, start=1.0, end=5.0, text="Quick mention of pooling."),
            ],
        )
        entries = [
            AlignmentEntry(slide=1, start_segment=0, end_segment=0, start_time=0.0, end_time=0.0),
            AlignmentEntry(slide=2, start_segment=1, end_segment=1, start_time=1.0, end_time=5.0),
        ]

        annotated, issues = annotate_alignment_entries(entries, transcript, slides)
        codes = {issue.code for issue in issues}
        self.assertIn("zero_duration", codes)
        self.assertIn("empty_span_text", codes)
        self.assertIn("short_substantive_window", codes)

        # Zero-duration span is marked uncertain; short substantive window is warning-only.
        self.assertTrue(annotated[0].uncertain)
        self.assertFalse(annotated[1].uncertain)

    def test_suspiciously_wide_span_and_uncovered_slides_and_segments(self) -> None:
        slides = _make_slides(["Slide 1", "Slide 2", "Slide 3", "Slide 4"])
        transcript = _make_transcript([f"Speech {i}" for i in range(12)], segment_seconds=12.0)
        entries = [
            AlignmentEntry(slide=1, start_segment=0, end_segment=8, start_time=0.0, end_time=108.0),
            AlignmentEntry(slide=2, start_segment=9, end_segment=9, start_time=108.0, end_time=120.0),
        ]

        issues = check_alignment_quality(entries, transcript, slides)
        issue_by_code = {issue.code: issue for issue in issues}
        self.assertIn("suspiciously_wide_span", issue_by_code)
        self.assertEqual(issue_by_code["suspiciously_wide_span"].slides, [1])
        self.assertIn("uncovered_slides", issue_by_code)
        self.assertEqual(issue_by_code["uncovered_slides"].slides, [3, 4])
        self.assertIn("uncovered_transcript_segments", issue_by_code)

    def test_explicit_off_slide_spans_do_not_break_monotonicity(self) -> None:
        slides = _make_slides(["Slide 1", "Slide 2"])
        transcript = _make_transcript([f"Speech {i}" for i in range(6)], segment_seconds=15.0)
        entries = [
            AlignmentEntry(slide=1, start_segment=0, end_segment=1, start_time=0.0, end_time=30.0),
            AlignmentEntry(
                slide=None,
                start_segment=2,
                end_segment=3,
                start_time=30.0,
                end_time=60.0,
                uncertain=False,
                reason="Live coding demo and student Q&A",
            ),
            AlignmentEntry(slide=2, start_segment=4, end_segment=5, start_time=60.0, end_time=90.0),
        ]

        annotated, issues = annotate_alignment_entries(entries, transcript, slides)
        self.assertEqual(issues, [])
        self.assertIsNone(annotated[1].slide)
        self.assertFalse(annotated[1].uncertain)
        self.assertEqual(annotated[1].reason, "Live coding demo and student Q&A")

    def test_empty_transcript_and_empty_slides_issues(self) -> None:
        empty_slides = _make_slides([])
        empty_transcript = _make_transcript([])

        issues = check_alignment_quality([], empty_transcript, empty_slides)
        codes = {issue.code for issue in issues}
        self.assertIn("empty_transcript", codes)
        self.assertIn("empty_slides", codes)


class TestDeterministicStubAlignment(unittest.TestCase):
    def test_stub_zero_segments(self) -> None:
        slides = _make_slides(["Intro", "Methods"])
        transcript = _make_transcript([])
        entries = align_deterministic_stub(slides, transcript)
        self.assertEqual(entries, [])

        annotated, issues = annotate_alignment_entries(entries, transcript, slides)
        self.assertEqual(annotated, [])
        codes = {issue.code for issue in issues}
        self.assertIn("empty_transcript", codes)
        self.assertIn("uncovered_slides", codes)

    def test_stub_zero_slides_emits_off_slide_span(self) -> None:
        slides = _make_slides([])
        transcript = _make_transcript(["Opening remark", "Main demo", "Student question"])
        entries = align_deterministic_stub(slides, transcript)

        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].slide)
        self.assertEqual(entries[0].start_segment, 0)
        self.assertEqual(entries[0].end_segment, 2)
        self.assertEqual(entries[0].start_time, 0.0)
        self.assertEqual(entries[0].end_time, 45.0)
        self.assertIn("off-slide", entries[0].reason.lower())

        annotated, issues = annotate_alignment_entries(entries, transcript, slides)
        self.assertEqual(len(annotated), 1)
        self.assertEqual([issue.code for issue in issues], ["empty_slides"])

    def test_stub_fewer_segments_than_slides_is_monotonic_without_overlap(self) -> None:
        slides = _make_slides(
            [
                "Course syllabus logistics",
                "Quantum entanglement bell inequality",
                "Classical mechanics newton",
                "Thermodynamics entropy carnot cycle",
            ]
        )
        transcript = _make_transcript(
            [
                "Today we focus on quantum entanglement and the bell inequality.",
                "Then we jump to thermodynamics entropy and the carnot cycle.",
            ],
            segment_seconds=20.0,
        )

        entries = align_deterministic_stub(slides, transcript)
        self.assertEqual(len(entries), 2)
        self.assertEqual([e.slide for e in entries], [2, 4])
        self.assertEqual([(e.start_segment, e.end_segment) for e in entries], [(0, 0), (1, 1)])

        annotated, issues = annotate_alignment_entries(entries, transcript, slides)
        self.assertTrue(all(not e.uncertain for e in annotated))
        self.assertEqual([issue.code for issue in issues], ["uncovered_slides"])
        self.assertEqual(issues[0].slides, [1, 3])

    def test_stub_more_segments_than_slides_uses_lexical_hints_deterministically(self) -> None:
        slides = _make_slides(
            [
                "Fourier transform frequency spectrum sinusoidal decomposition",
                "Nyquist sampling theorem aliasing reconstruction",
                "Butterworth lowpass filter cutoff attenuation",
            ]
        )
        transcript = _make_transcript(
            [
                "The Fourier transform decomposes signals into frequency spectrum components.",
                "Sinusoidal decomposition reveals harmonics clearly.",
                "Computing the frequency spectrum helps analyze periodic waves.",
                "Another Fourier transform example shows phase shifts.",
                "Next the Nyquist sampling theorem prevents aliasing during reconstruction.",
                "Finally a Butterworth lowpass filter shapes cutoff attenuation.",
            ],
            segment_seconds=15.0,
        )

        first_run = align_deterministic_stub(slides, transcript)
        second_run = align_deterministic_stub(slides, transcript)
        self.assertEqual(first_run, second_run)

        self.assertEqual([e.slide for e in first_run], [1, 2, 3])
        # Lexical overlap shifts slide 1 to cover segments 0..3 instead of naive 0..1.
        self.assertEqual(
            [(e.start_segment, e.end_segment) for e in first_run],
            [(0, 3), (4, 4), (5, 5)],
        )

        issues = check_alignment_quality(first_run, transcript, slides)
        self.assertEqual(issues, [])

    def test_stub_proportional_fallback_without_lexical_overlap(self) -> None:
        slides = _make_slides(["Alpha", "Beta", "Gamma"])
        transcript = _make_transcript(
            ["One", "Two", "Three", "Four", "Five", "Six"],
            segment_seconds=12.0,
        )

        entries = align_deterministic_stub(slides, transcript)
        self.assertEqual([e.slide for e in entries], [1, 2, 3])
        self.assertEqual(
            [(e.start_segment, e.end_segment) for e in entries],
            [(0, 1), (2, 3), (4, 5)],
        )


class TestLLMHookAndStageExecution(unittest.TestCase):
    def test_non_stub_mode_without_llm_callable_raises_actionable_stage_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="openai")
            slides = _make_slides(["Intro"])
            transcript = _make_transcript(["Hello class"])

            with self.assertRaises(StageError) as ctx:
                align_with_llm(slides, transcript, settings)

            self.assertEqual(ctx.exception.stage, "alignment")
            self.assertIn("openai", ctx.exception.message)
            self.assertIn("Phase 6", ctx.exception.hint)
            self.assertIn("PIPELINE_LLM_MODE=stub", ctx.exception.hint)

    def test_injected_llm_callable_normalizes_and_annotates_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="live")
            slides = _make_slides(["Slide 1", "Slide 2", "Slide 3"])
            transcript = _make_transcript(
                ["Seg 0", "Seg 1", "Seg 2", "Seg 3", "Seg 4", "Seg 5"],
                segment_seconds=12.0,
            )

            def fake_llm(
                _slides: SlidesOCRArtifact,
                _transcript: TranscriptArtifact,
                _settings: Settings,
            ) -> list[dict[str, object]]:
                return [
                    {"slide": 1, "start_segment": 0, "end_segment": 1},
                    {"slide": None, "start_segment": 2, "end_segment": 2, "reason": "Student Q&A"},
                    {"slide": 3, "start_segment": 3, "end_segment": 4},
                    {"slide": 2, "start_segment": 5, "end_segment": 5},
                ]

            artifact = align_with_llm(slides, transcript, settings, llm_callable=fake_llm)
            self.assertEqual(artifact.mode, "live")
            self.assertEqual(len(artifact.entries), 4)
            # Timestamps were auto-resolved from transcript segments.
            self.assertEqual(artifact.entries[0].start_time, 0.0)
            self.assertEqual(artifact.entries[0].end_time, 24.0)
            self.assertIsNone(artifact.entries[1].slide)
            self.assertEqual(artifact.entries[1].reason, "Student Q&A")
            # Out-of-order jump (1 -> 3 -> 2) marked uncertain.
            self.assertTrue(artifact.entries[2].uncertain)
            self.assertTrue(artifact.entries[3].uncertain)
            self.assertIn("out_of_order_slide", {i.code for i in artifact.alignment_issues})

    def test_injected_llm_returning_alignment_artifact_preserves_and_deduplicates_issues(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="live")
            slides = _make_slides(["Slide 1", "Slide 2"])
            transcript = _make_transcript(["Seg 0", "Seg 1"], segment_seconds=15.0)
            pre_issue = AlignmentIssue(
                code="llm_low_confidence",
                message="Model reported low confidence at slide boundary.",
                slides=[2],
                severity="warning",
            )

            def artifact_llm(
                _slides: SlidesOCRArtifact,
                _transcript: TranscriptArtifact,
                _settings: Settings,
            ) -> AlignmentArtifact:
                return AlignmentArtifact(
                    mode="custom-llm",
                    entries=[
                        AlignmentEntry(
                            slide=1, start_segment=0, end_segment=0, start_time=0.0, end_time=15.0
                        ),
                        AlignmentEntry(
                            slide=2,
                            start_segment=1,
                            end_segment=1,
                            start_time=15.0,
                            end_time=30.0,
                            uncertain=True,
                            reason="Speaker jumped topics quickly.",
                        ),
                    ],
                    alignment_issues=[pre_issue, pre_issue],
                )

            artifact = align_with_llm(slides, transcript, settings, llm_callable=artifact_llm)
            self.assertEqual(artifact.mode, "custom-llm")
            self.assertTrue(artifact.entries[1].uncertain)
            self.assertEqual(artifact.entries[1].reason, "Speaker jumped topics quickly.")
            self.assertEqual(len(artifact.alignment_issues), 1)
            self.assertEqual(artifact.alignment_issues[0].code, "llm_low_confidence")

    def test_injected_llm_exception_wrapped_in_stage_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="live")
            slides = _make_slides(["Slide 1"])
            transcript = _make_transcript(["Seg 0"])

            def failing_llm(*_args: object) -> list[AlignmentEntry]:
                raise RuntimeError("upstream timeout")

            with self.assertRaises(StageError) as ctx:
                align_with_llm(slides, transcript, settings, llm_callable=failing_llm)

            self.assertEqual(ctx.exception.stage, "alignment")
            self.assertIn("upstream timeout", ctx.exception.message)

    def test_normalize_raw_entries_handles_malformed_and_negative_values(self) -> None:
        slides = _make_slides(["Slide 1"])
        transcript = _make_transcript(["Seg 0", "Seg 1"])
        entries, issues = normalize_alignment_entries(
            [
                "not-a-dict",  # type: ignore[list-item]
                {"slide": 0, "start_segment": -2, "end_segment": 1},
                {"slide": 1, "start_segment": 0},
            ],
            transcript,
            slides,
        )
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0].slide)
        self.assertEqual(entries[0].start_segment, 0)
        self.assertTrue(entries[0].uncertain)
        codes = {issue.code for issue in issues}
        self.assertIn("malformed_alignment_entry", codes)
        self.assertIn("unknown_slide", codes)
        self.assertIn("invalid_segment_range", codes)

    def test_run_alignment_stage_persists_artifact_and_reports_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = _make_settings(tmp_path, pipeline_llm_mode="stub")
            progress_log: list[str] = []
            context = _make_context(tmp_path, progress_log=progress_log)

            slides = _make_slides(
                [
                    "Linear regression least squares objective",
                    "Logistic regression sigmoid probability classification",
                ]
            )
            transcript = _make_transcript(
                [
                    "Linear regression minimizes the least squares objective.",
                    "We fit coefficients to minimize squared residuals.",
                    "Logistic regression uses the sigmoid function for classification.",
                    "Outputs represent class probabilities.",
                ],
                segment_seconds=15.0,
            )
            context.slides_ocr_path.write_text(slides.model_dump_json(indent=2), encoding="utf-8")
            context.transcript_path.write_text(
                transcript.model_dump_json(indent=2), encoding="utf-8"
            )

            result = run_alignment_stage(context, settings)

            self.assertTrue(context.alignment_path.exists())
            self.assertEqual(result.artifacts, {"alignment": ARTIFACT_ALIGNMENT})
            self.assertEqual(result.warnings, [])
            self.assertEqual(result.stats["alignment_mode"], "stub")
            self.assertEqual(result.stats["aligned_entries"], 2)
            self.assertEqual(result.stats["aligned_slides"], 2)
            self.assertEqual(result.stats["uncertain_entries"], 0)
            self.assertEqual(result.stats["off_slide_entries"], 0)
            self.assertGreaterEqual(len(progress_log), 2)

            persisted = AlignmentArtifact.model_validate_json(
                context.alignment_path.read_text(encoding="utf-8")
            )
            self.assertEqual(persisted.mode, "stub")
            self.assertEqual([e.slide for e in persisted.entries], [1, 2])
            self.assertEqual(persisted.alignment_issues, [])

    def test_run_alignment_stage_surfaces_warnings_from_alignment_issues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = _make_settings(tmp_path, pipeline_llm_mode="stub")
            context = _make_context(tmp_path)

            slides = _make_slides(["Slide 1", "Slide 2", "Slide 3"])
            transcript = _make_transcript(["Only one spoken segment."], segment_seconds=20.0)
            context.slides_ocr_path.write_text(slides.model_dump_json(indent=2), encoding="utf-8")
            context.transcript_path.write_text(
                transcript.model_dump_json(indent=2), encoding="utf-8"
            )

            result = run_alignment_stage(context, settings)
            self.assertEqual(result.stats["alignment_issue_count"], 1)
            self.assertEqual(len(result.warnings), 1)
            self.assertIn("uncovered_slides", result.warnings[0])

    def test_run_alignment_stage_missing_or_corrupt_inputs_raise_stage_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = _make_settings(tmp_path, pipeline_llm_mode="stub")
            context = _make_context(tmp_path)

            # Missing transcript.json
            with self.assertRaises(StageError) as missing_tx:
                run_alignment_stage(context, settings)
            self.assertEqual(missing_tx.exception.stage, "alignment")
            self.assertIn("transcript.json", missing_tx.exception.message)

            # Corrupt transcript.json
            context.transcript_path.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(StageError) as bad_tx:
                load_transcript_artifact(context.transcript_path)
            self.assertEqual(bad_tx.exception.stage, "alignment")

            # Valid transcript, missing slides_ocr.json
            transcript = _make_transcript(["Hello"])
            context.transcript_path.write_text(
                transcript.model_dump_json(indent=2), encoding="utf-8"
            )
            with self.assertRaises(StageError) as missing_ocr:
                run_alignment_stage(context, settings)
            self.assertEqual(missing_ocr.exception.stage, "alignment")
            self.assertIn("slides_ocr.json", missing_ocr.exception.message)

            # Corrupt slides_ocr.json
            context.slides_ocr_path.write_text(json.dumps({"wrong": "schema"}), encoding="utf-8")
            with self.assertRaises(StageError) as bad_ocr:
                load_slides_ocr_artifact(context.slides_ocr_path)
            self.assertEqual(bad_ocr.exception.stage, "alignment")

    def test_format_alignment_warnings(self) -> None:
        formatted = format_alignment_warnings(
            [
                AlignmentIssue(
                    code="uncovered_slides",
                    message="1 slide(s) were not covered: slide(s) 3.",
                    slides=[3],
                    severity="warning",
                )
            ]
        )
        self.assertEqual(formatted, ["1 slide(s) were not covered: slide(s) 3. (uncovered_slides)"])


if __name__ == "__main__":
    unittest.main()
