from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.app.config import Settings
from backend.app.pipeline.notes import (
    SectionChunkConfig,
    build_lecture_sections,
    check_coverage_with_llm,
    check_section_coverage_stub,
    generate_notes,
    generate_section_notes_with_llm,
    load_notes_inputs,
    render_section_notes_stub,
    run_notes_stage,
)
from backend.app.pipeline.prompts import (
    AI_FIRST_MARKDOWN_RULES,
    COVERAGE_CHECK_SYSTEM_PROMPT,
    SECTION_NOTES_SYSTEM_PROMPT,
    CoverageCheckInput,
    CoverageCheckResult,
    CoverageGap,
    PromptMessages,
    RollingContext,
    SectionPromptInput,
    SectionPromptSegment,
    SectionPromptSlide,
    build_coverage_check_prompt,
    build_section_notes_prompt,
    format_timestamp,
    format_timestamp_range,
    parse_coverage_check_response,
)
from backend.app.schemas import (
    ARTIFACT_NOTES_MD,
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
    slides_filename: str = "distributed_systems_week3.pdf",
    recording_filename: str = "week3_lecture.mp4",
    progress_log: list[str] | None = None,
) -> JobContext:
    job_dir = tmp_dir / "job-notes-1"
    input_dir = job_dir / "input"
    artifacts_dir = job_dir / "artifacts"
    input_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    slides_path = input_dir / "slides.pdf"
    slides_path.write_bytes(b"%PDF-1.7\n")
    recording_path = input_dir / "recording.mp4"
    recording_path.write_bytes(b"fake-video")
    callback = progress_log.append if progress_log is not None else None
    return JobContext(
        token="tok-phase4-notes",
        slides_filename=slides_filename,
        recording_filename=recording_filename,
        job_dir=job_dir,
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        slides_path=slides_path,
        recording_path=recording_path,
        progress_callback=callback,
    )


def _write_artifacts(
    context: JobContext,
    slides_ocr: SlidesOCRArtifact,
    transcript: TranscriptArtifact,
    alignment: AlignmentArtifact,
) -> None:
    context.slides_ocr_path.write_text(slides_ocr.model_dump_json(indent=2), encoding="utf-8")
    context.transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    context.alignment_path.write_text(alignment.model_dump_json(indent=2), encoding="utf-8")


def _sample_lecture_artifacts() -> tuple[SlidesOCRArtifact, TranscriptArtifact, AlignmentArtifact]:
    slides_ocr = SlidesOCRArtifact(
        source_pdf="raft_consensus_lecture.pdf",
        model="gemini-2.5-flash",
        slide_count=5,
        slides=[
            SlideOCRRecord(
                slide=1,
                visible_text="Raft Consensus Algorithm\nLeader Election and Log Replication",
                visual_context="State diagram with Follower, Candidate, and Leader nodes.",
                tables_code_formulas="",
            ),
            SlideOCRRecord(
                slide=2,
                visible_text="Election Timeouts & Split Votes\nRandomized timeout range: 150ms–300ms",
                visual_context="Timeline showing two candidates requesting votes simultaneously.",
                tables_code_formulas="timeout ~ Uniform(150ms, 300ms)",
            ),
            SlideOCRRecord(
                slide=3,
                visible_text="AppendEntries RPC\nLog matching property",
                visual_context="Two horizontal log arrays with index 1..5 and term numbers.",
                tables_code_formulas="AppendEntries(term, leaderId, prevLogIndex, prevLogTerm, entries[])",
            ),
            SlideOCRRecord(
                slide=4,
                visible_text="Safety Proof Sketch\nLeader Completeness Property",
                visual_context="Diagram comparing committed entry on majority quorum against candidate log.",
                tables_code_formulas="quorum = floor(N / 2) + 1",
            ),
            SlideOCRRecord(
                slide=5,
                visible_text="Further Reading\nOngaro & Ousterhout (2014)",
                visual_context="",
                tables_code_formulas="",
            ),
        ],
    )
    transcript = TranscriptArtifact(
        source_recording="raft_lecture.mp4",
        model="whisper-large-v3-turbo",
        duration_seconds=360.0,
        segments=[
            TranscriptSegment(
                index=0,
                start=0.0,
                end=35.0,
                text=(
                    "Today we examine the Raft Consensus Algorithm, focusing on how Follower, "
                    "Candidate, and Leader states coordinate state machine replication."
                ),
            ),
            TranscriptSegment(
                index=1,
                start=35.0,
                end=95.0,
                text=(
                    "Notice the election timeout on this slide: randomizing between 150ms and "
                    "300ms breaks symmetric split votes when two candidates start an election together."
                ),
            ),
            TranscriptSegment(
                index=2,
                start=95.0,
                end=150.0,
                text=(
                    "Student question: What happens if a network partition isolates the current "
                    "leader? Answer: The isolated leader cannot reach a quorum of nodes, so its "
                    "new writes remain uncommitted while the majority side elects a higher-term leader."
                ),
            ),
            TranscriptSegment(
                index=3,
                start=150.0,
                end=230.0,
                text=(
                    "Now walking through AppendEntries RPC step by step: step 1 checks prevLogIndex, "
                    "step 2 checks prevLogTerm, and step 3 fails whenever the follower's entry has "
                    "a conflicting term, forcing the leader to decrement nextIndex and retry."
                ),
            ),
            TranscriptSegment(
                index=4,
                start=230.0,
                end=310.0,
                text=(
                    "Because every committed entry is stored on at least floor(N/2)+1 servers, "
                    "any future elected leader quorum must intersect that majority."
                ),
            ),
        ],
    )
    alignment = AlignmentArtifact(
        mode="stub",
        entries=[
            AlignmentEntry(slide=1, start_segment=0, end_segment=0, start_time=0.0, end_time=35.0),
            AlignmentEntry(slide=2, start_segment=1, end_segment=1, start_time=35.0, end_time=95.0),
            AlignmentEntry(
                slide=None,
                start_segment=2,
                end_segment=2,
                start_time=95.0,
                end_time=150.0,
                uncertain=False,
                reason="Student Q&A on network partitions",
            ),
            AlignmentEntry(slide=3, start_segment=3, end_segment=3, start_time=150.0, end_time=230.0),
            AlignmentEntry(
                slide=4,
                start_segment=4,
                end_segment=4,
                start_time=230.0,
                end_time=310.0,
                uncertain=True,
                reason="Out-of-order jump detected near proof discussion",
            ),
        ],
        alignment_issues=[
            AlignmentIssue(
                code="uncovered_slides",
                message="Slide 5 had no aligned transcript segments.",
                slides=[5],
                severity="warning",
            ),
            AlignmentIssue(
                code="out_of_order_jump",
                message="Uncertain jump around slide 4.",
                slides=[4],
                severity="warning",
            ),
        ],
    )
    return slides_ocr, transcript, alignment


class TestPromptsSpecification(unittest.TestCase):
    def test_all_nine_ai_first_rules_encoded(self) -> None:
        self.assertEqual(len(AI_FIRST_MARKDOWN_RULES), 9)
        self.assertEqual([rule.number for rule in AI_FIRST_MARKDOWN_RULES], list(range(1, 10)))

        prompt = SECTION_NOTES_SYSTEM_PROMPT.lower()
        # Rule 1: length driven by teaching, no bullet caps / fixed quotas
        self.assertIn("no fixed sentence counts", prompt.replace("never impose fixed sentence counts", "no fixed sentence counts"))
        self.assertIn("bullet caps", prompt)
        self.assertIn("word quotas", prompt)
        # Rule 2: explicit over stylistic, pronoun replacement for NotebookLM
        self.assertIn("notebooklm", prompt)
        self.assertIn("ambiguous pronouns", prompt)
        # Rule 3: preserve reasoning chains in sequence
        self.assertIn("reasoning chains", prompt)
        self.assertIn("why it failed", prompt)
        # Rule 4: keep substantive spoken content, strip filler
        self.assertIn("student q&a", prompt)
        self.assertIn("self-corrections", prompt)
        self.assertIn("spoken filler", prompt)
        # Rule 5: connect speech to visuals using VISUAL_CONTEXT + ambiguity
        self.assertIn("visual_context", prompt)
        self.assertIn("ambiguity", prompt)
        # Rule 6: exam guidance only when explicitly stated
        self.assertIn("time spent on a topic alone is never evidence", prompt)
        # Rule 7: strict grounding
        self.assertIn("never invent missing logical steps", prompt)
        self.assertIn("contradictions", prompt)
        # Rule 8: chunk-friendly Markdown + timestamp-only fallback on uncertainty
        self.assertIn("[mm:ss–mm:ss]", prompt)
        self.assertIn("timestamp alone instead of forcing a dubious slide number", prompt)
        # Rule 9: rolling context + transcript coverage check
        self.assertIn("rolling context", prompt)
        self.assertIn("coverage", prompt)

    def test_timestamp_formatting_uses_en_dash(self) -> None:
        self.assertEqual(format_timestamp(0.0), "00:00")
        self.assertEqual(format_timestamp(65.4), "01:05")
        self.assertEqual(format_timestamp(3725.0), "62:05")
        self.assertEqual(format_timestamp_range(95.0, 150.0), "[01:35–02:30]")

    def test_build_section_notes_prompt_variants(self) -> None:
        rolling = RollingContext(
            prior_section_headings=["Introduction to Raft (Slide 1) [00:00–00:35]"],
            key_terms=["Raft", "Quorum", "AppendEntries"],
            ongoing_threads=["00:35 Comparing split votes across 3-node cluster"],
            last_section_summary="Introduced Follower, Candidate, and Leader states.",
            last_slide=1,
            last_timestamp=35.0,
        )

        aligned_input = SectionPromptInput(
            lecture_title="Raft Consensus",
            section_index=2,
            total_sections=4,
            section_kind="aligned",
            start_time=35.0,
            end_time=95.0,
            slide_numbers=(2,),
            slides=(
                SectionPromptSlide(
                    slide=2,
                    visible_text="Election Timeouts",
                    visual_context="Timeline diagram with two candidates.",
                    tables_code_formulas="timeout ~ Uniform(150ms, 300ms)",
                ),
            ),
            segments=(
                SectionPromptSegment(index=1, start=35.0, end=95.0, text="Randomized timeouts prevent split votes.", slide=2),
            ),
            rolling_context=rolling,
        )
        aligned_msgs = build_section_notes_prompt(aligned_input)
        self.assertIn("Slide 2", aligned_msgs.user)
        self.assertIn("[00:35–01:35]", aligned_msgs.user)
        self.assertIn("VISUAL_CONTEXT:", aligned_msgs.user)
        self.assertIn("Timeline diagram with two candidates.", aligned_msgs.user)
        self.assertIn("Raft, Quorum, AppendEntries", aligned_msgs.user)

        uncertain_input = SectionPromptInput(
            lecture_title="Raft Consensus",
            section_index=3,
            total_sections=4,
            section_kind="uncertain",
            start_time=230.0,
            end_time=310.0,
            slide_numbers=(),
            slides=(),
            segments=(
                SectionPromptSegment(
                    index=4,
                    start=230.0,
                    end=310.0,
                    text="Quorum intersection ensures leader completeness.",
                    slide=None,
                    uncertain=True,
                    uncertainty_reason="Out-of-order jump detected",
                ),
            ),
            uncertainty_reasons=("Out-of-order jump detected",),
        )
        uncertain_msgs = build_section_notes_prompt(uncertain_input)
        self.assertIn("UNCERTAIN ALIGNMENT SECTION", uncertain_msgs.user)
        self.assertIn("timestamp ALONE", uncertain_msgs.user)
        self.assertIn("[03:50–05:10]", uncertain_msgs.user)
        self.assertNotIn("#### Slide", uncertain_msgs.user)

        off_slide_input = SectionPromptInput(
            lecture_title="Raft Consensus",
            section_index=4,
            total_sections=4,
            section_kind="off_slide",
            start_time=95.0,
            end_time=150.0,
            segments=(
                SectionPromptSegment(index=2, start=95.0, end=150.0, text="Student question on partitions.", slide=None),
            ),
        )
        off_slide_msgs = build_section_notes_prompt(off_slide_input)
        self.assertIn("OFF-SLIDE / DEMO / Q&A SECTION", off_slide_msgs.user)
        self.assertIn("Do NOT attach or invent any slide number", off_slide_msgs.user)

    def test_coverage_check_prompt_and_response_parser(self) -> None:
        cov_input = CoverageCheckInput(
            lecture_title="Raft Consensus",
            section_index=1,
            total_sections=1,
            section_kind="aligned",
            start_time=0.0,
            end_time=60.0,
            slide_numbers=(1,),
            segments=(
                SectionPromptSegment(index=0, start=0.0, end=30.0, text="Leader election uses randomized timers.", slide=1),
                SectionPromptSegment(index=1, start=30.0, end=60.0, text="Student asked about split brain scenarios.", slide=1),
            ),
            generated_markdown="## Raft Leader Election (Slide 1) [00:00–01:00]\n- Uses randomized timers.",
        )
        msgs = build_coverage_check_prompt(cov_input)
        self.assertEqual(msgs.system, COVERAGE_CHECK_SYSTEM_PROMPT)
        self.assertIn("Student asked about split brain scenarios.", msgs.user)

        fenced_json = """```json
        {
          "passed": false,
          "covered_segment_count": 1,
          "total_segment_count": 2,
          "missing_items": [
            {
              "category": "student_qa",
              "start_time": 30.0,
              "end_time": 60.0,
              "segment_indices": [1],
              "summary": "Omitted student Q&A about split brain scenarios.",
              "severity": "warning"
            }
          ],
          "notes": "One student Q&A segment was not covered."
        }
        ```"""
        parsed = parse_coverage_check_response(fenced_json, total_segment_count=2)
        self.assertFalse(parsed.passed)
        self.assertEqual(parsed.covered_segment_count, 1)
        self.assertEqual(parsed.total_segment_count, 2)
        self.assertEqual(len(parsed.missing_items), 1)
        self.assertEqual(parsed.missing_items[0].category, "student_qa")
        self.assertIn("[00:30–01:00]", parsed.to_warnings()[0])

        with self.assertRaises(ValueError):
            parse_coverage_check_response("not valid json")


class TestSectionChunking(unittest.TestCase):
    def test_chunking_respects_max_slides_duration_and_chars(self) -> None:
        slides = SlidesOCRArtifact(
            source_pdf="deck.pdf",
            slide_count=6,
            slides=[SlideOCRRecord(slide=i, visible_text=f"Topic {i}") for i in range(1, 7)],
        )
        segments = [
            TranscriptSegment(
                index=i - 1,
                start=(i - 1) * 60.0,
                end=i * 60.0,
                text=f"Explanation for slide {i} " + ("detail " * 15),
            )
            for i in range(1, 7)
        ]
        transcript = TranscriptArtifact(source_recording="rec.mp3", duration_seconds=360.0, segments=segments)
        alignment = AlignmentArtifact(
            entries=[
                AlignmentEntry(
                    slide=i,
                    start_segment=i - 1,
                    end_segment=i - 1,
                    start_time=(i - 1) * 60.0,
                    end_time=i * 60.0,
                )
                for i in range(1, 7)
            ]
        )

        # Max 2 slides per section -> 3 sections of 2 slides each.
        by_slides = build_lecture_sections(
            slides,
            transcript,
            alignment,
            config=SectionChunkConfig(max_slides_per_section=2, max_duration_seconds=600.0, max_transcript_chars=10000),
        )
        self.assertEqual([sec.slide_numbers for sec in by_slides], [(1, 2), (3, 4), (5, 6)])

        # Max duration 90s -> each 60s slide stays in its own section (60 + 60 = 120 > 90).
        by_duration = build_lecture_sections(
            slides,
            transcript,
            alignment,
            config=SectionChunkConfig(max_slides_per_section=6, max_duration_seconds=90.0, max_transcript_chars=10000),
        )
        self.assertEqual(len(by_duration), 6)

        # Max transcript chars 250 -> each segment (~130 chars) groups in pairs (<=250? 260 > 250 -> 6 sections).
        by_chars = build_lecture_sections(
            slides,
            transcript,
            alignment,
            config=SectionChunkConfig(max_slides_per_section=6, max_duration_seconds=600.0, max_transcript_chars=200),
        )
        self.assertEqual(len(by_chars), 6)

    def test_off_slide_and_uncovered_segments_preserved_as_explicit_sections(self) -> None:
        slides_ocr, transcript, alignment = _sample_lecture_artifacts()
        # Add an unreferenced trailing segment (index 5) not listed in alignment.entries.
        transcript = transcript.model_copy(
            update={
                "segments": [
                    *transcript.segments,
                    TranscriptSegment(
                        index=5,
                        start=315.0,
                        end=355.0,
                        text="Closing reminder: Lab 2 implements RequestVote RPC and log compaction.",
                    ),
                ]
            }
        )
        sections = build_lecture_sections(
            slides_ocr,
            transcript,
            alignment,
            config=SectionChunkConfig(max_slides_per_section=4),
        )
        # Expected chronological order:
        # 1: aligned Slides 1-2
        # 2: off_slide Q&A (seg 2)
        # 3: aligned Slide 3 (seg 3)
        # 4: uncertain region (seg 4, originally slide=4 uncertain=True)
        # 5: off_slide unreferenced trailing gap (seg 5)
        self.assertEqual(
            [sec.kind for sec in sections],
            ["aligned", "off_slide", "aligned", "uncertain", "off_slide"],
        )
        self.assertEqual(sections[0].slide_numbers, (1, 2))
        self.assertEqual(sections[1].slide_numbers, ())
        self.assertEqual(sections[2].slide_numbers, (3,))
        self.assertEqual(sections[3].slide_numbers, ())
        self.assertEqual(sections[4].slide_numbers, ())
        self.assertEqual([seg.index for seg in sections[4].segments], [5])

    def test_long_off_slide_demo_splits_at_segment_boundaries(self) -> None:
        slides_ocr = SlidesOCRArtifact(source_pdf="demo.pdf", slide_count=1, slides=[SlideOCRRecord(slide=1, visible_text="Intro")])
        transcript = TranscriptArtifact(
            source_recording="demo.mp4",
            duration_seconds=300.0,
            segments=[
                TranscriptSegment(index=0, start=0.0, end=80.0, text="Live demo part one inspecting kernel logs."),
                TranscriptSegment(index=1, start=80.0, end=160.0, text="Live demo part two reproducing the deadlock."),
                TranscriptSegment(index=2, start=160.0, end=240.0, text="Live demo part three applying the mutex ordering fix."),
            ],
        )
        alignment = AlignmentArtifact(
            entries=[
                AlignmentEntry(slide=None, start_segment=0, end_segment=2, start_time=0.0, end_time=240.0, reason="Live terminal demo")
            ]
        )
        sections = build_lecture_sections(
            slides_ocr,
            transcript,
            alignment,
            config=SectionChunkConfig(max_duration_seconds=100.0),
        )
        self.assertEqual(len(sections), 3)
        self.assertTrue(all(sec.kind == "off_slide" for sec in sections))
        self.assertEqual([sec.segments[0].index for sec in sections], [0, 1, 2])


class TestUncertaintyFallback(unittest.TestCase):
    def test_uncertain_entry_and_unresolved_issue_use_timestamp_only_anchors(self) -> None:
        slides_ocr = SlidesOCRArtifact(
            source_pdf="physics.pdf",
            slide_count=4,
            slides=[
                SlideOCRRecord(slide=1, visible_text="Newton's Laws"),
                SlideOCRRecord(slide=2, visible_text="Momentum Conservation"),
                SlideOCRRecord(slide=3, visible_text="Angular Momentum"),
                SlideOCRRecord(slide=4, visible_text="Gyroscopic Precession"),
            ],
        )
        transcript = TranscriptArtifact(
            source_recording="physics.mp3",
            duration_seconds=240.0,
            segments=[
                TranscriptSegment(index=0, start=0.0, end=60.0, text="Newton's second law states force equals mass times acceleration."),
                TranscriptSegment(index=1, start=60.0, end=120.0, text="Explicit uncertain entry discussing impulse and momentum transfer."),
                TranscriptSegment(index=2, start=120.0, end=180.0, text="Unresolved issue flagged slide discussing torque and rotational inertia."),
                TranscriptSegment(index=3, start=180.0, end=240.0, text="Nonexistent slide reference discussing precession frequency."),
            ],
        )
        alignment = AlignmentArtifact(
            entries=[
                AlignmentEntry(slide=1, start_segment=0, end_segment=0, start_time=0.0, end_time=60.0, uncertain=False),
                AlignmentEntry(
                    slide=2,
                    start_segment=1,
                    end_segment=1,
                    start_time=60.0,
                    end_time=120.0,
                    uncertain=True,
                    reason="Low confidence slide transition",
                ),
                AlignmentEntry(
                    slide=3,
                    start_segment=2,
                    end_segment=2,
                    start_time=120.0,
                    end_time=180.0,
                    uncertain=False,
                ),
                AlignmentEntry(
                    slide=99,
                    start_segment=3,
                    end_segment=3,
                    start_time=180.0,
                    end_time=240.0,
                    uncertain=False,
                ),
            ],
            alignment_issues=[
                AlignmentIssue(
                    code="out_of_order_jump",
                    message="Slide 3 jumped out of monotonic order.",
                    slides=[3],
                    severity="warning",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp))
            result = generate_notes(slides_ocr, transcript, alignment, settings)

        # Slide 1 aligned; entries for slides 2, 3, 99 all fall back to uncertain timestamp-only.
        self.assertEqual(result.sections[0].kind, "aligned")
        self.assertEqual(result.sections[0].slide_numbers, (1,))
        uncertain_sections = [sec for sec in result.sections if sec.kind == "uncertain"]
        self.assertGreaterEqual(len(uncertain_sections), 1)
        for sec in uncertain_sections:
            self.assertEqual(sec.slide_numbers, ())
            self.assertEqual(sec.slides, ())

        # Verify rendered Markdown has timestamp-only anchors for uncertain spans and NEVER forces Slide 2, 3, or 99.
        self.assertIn("Slide 1", result.markdown)
        self.assertIn("[01:00–04:00]", result.markdown)
        self.assertNotIn("Slide 2", result.markdown)
        self.assertNotIn("Slide 3", result.markdown)
        self.assertNotIn("Slide 99", result.markdown)
        # Slide 4 had no entry at all, so it appears only in the slide-only unaligned section.
        self.assertIn("## Unaligned Slides (Slide Content Only)", result.markdown)
        self.assertIn("### Slide 4: Gyroscopic Precession", result.markdown)

    def test_informational_and_localized_align_issues_preserve_confirmed_slides(self) -> None:
        slides_ocr = SlidesOCRArtifact(
            source_pdf="algo.pdf",
            slide_count=2,
            slides=[
                SlideOCRRecord(slide=1, visible_text="Dynamic Programming Overview"),
                SlideOCRRecord(slide=2, visible_text="Edit Distance Walkthrough"),
            ],
        )
        transcript = TranscriptArtifact(
            source_recording="algo.mp3",
            duration_seconds=800.0,
            segments=[
                TranscriptSegment(index=0, start=0.0, end=700.0, text="Detailed 12-minute walkthrough of Dynamic Programming."),
                TranscriptSegment(index=1, start=700.0, end=760.0, text="Edit Distance recurrence table explanation."),
                TranscriptSegment(index=2, start=760.0, end=800.0, text="Brief regression back to earlier slide."),
            ],
        )
        alignment = AlignmentArtifact(
            entries=[
                AlignmentEntry(slide=1, start_segment=0, end_segment=0, start_time=0.0, end_time=700.0, uncertain=False),
                AlignmentEntry(slide=2, start_segment=1, end_segment=1, start_time=700.0, end_time=760.0, uncertain=False),
                AlignmentEntry(
                    slide=1,
                    start_segment=2,
                    end_segment=2,
                    start_time=760.0,
                    end_time=800.0,
                    uncertain=True,
                    reason="Out-of-order regression from slide 2 to slide 1",
                ),
            ],
            alignment_issues=[
                AlignmentIssue(
                    code="suspiciously_wide_span",
                    message="Slide 1 covers a long span.",
                    slides=[1],
                    severity="warning",
                ),
                AlignmentIssue(
                    code="out_of_order_slide",
                    message="Out-of-order slide regression from slide 2 to slide 1.",
                    slides=[2, 1],
                    severity="warning",
                ),
            ],
        )
        sections = build_lecture_sections(
            slides_ocr, transcript, alignment, config=SectionChunkConfig(max_duration_seconds=480)
        )
        # Confirmed slides 1 and 2 remain aligned; only the localized regressing entry is uncertain.
        self.assertEqual([sec.kind for sec in sections], ["aligned", "aligned", "uncertain"])
        self.assertEqual(sections[0].slide_numbers, (1,))
        self.assertEqual(sections[1].slide_numbers, (2,))
        self.assertEqual(sections[2].slide_numbers, ())


class TestRollingContextCarryOver(unittest.TestCase):
    def test_rolling_context_handoff_across_section_boundaries(self) -> None:
        slides_ocr = SlidesOCRArtifact(
            source_pdf="ml_backprop.pdf",
            slide_count=4,
            slides=[
                SlideOCRRecord(slide=1, visible_text="Computational Graphs\nForward Pass"),
                SlideOCRRecord(slide=2, visible_text="Chain Rule Derivation\nLocal Gradients"),
                SlideOCRRecord(slide=3, visible_text="Vanishing Gradients\nSigmoid Saturation"),
                SlideOCRRecord(slide=4, visible_text="Residual Connections\nGradient Highways"),
            ],
        )
        transcript = TranscriptArtifact(
            source_recording="ml.mp3",
            duration_seconds=240.0,
            segments=[
                TranscriptSegment(
                    index=0,
                    start=0.0,
                    end=60.0,
                    text="Let's start an example walkthrough of Backpropagation on a two-layer Computational Graph.",
                ),
                TranscriptSegment(
                    index=1,
                    start=60.0,
                    end=120.0,
                    text="Continuing step 2 of the example, we multiply upstream gradient by the Local Gradient.",
                ),
                TranscriptSegment(
                    index=2,
                    start=120.0,
                    end=180.0,
                    text="When Sigmoid saturates, repeated multiplication causes Vanishing Gradients.",
                ),
                TranscriptSegment(
                    index=3,
                    start=180.0,
                    end=240.0,
                    text="Residual Connections add an identity shortcut so gradients flow directly backward.",
                ),
            ],
        )
        alignment = AlignmentArtifact(
            entries=[
                AlignmentEntry(slide=i, start_segment=i - 1, end_segment=i - 1, start_time=(i - 1) * 60.0, end_time=i * 60.0)
                for i in range(1, 5)
            ]
        )

        captured_messages: list[PromptMessages] = []

        def fake_section_llm(
            messages: PromptMessages,
            prompt_input: SectionPromptInput,
            settings: Settings,
        ) -> str:
            captured_messages.append(messages)
            return (
                f"## Section {prompt_input.section_index}: {prompt_input.slides[0].visible_text.splitlines()[0]} "
                f"(Slide {prompt_input.slide_numbers[0]}) {format_timestamp_range(prompt_input.start_time, prompt_input.end_time)}\n\n"
                f"- Uses `grad_fn_{prompt_input.section_index}`.\n"
                f"- {prompt_input.segments[0].text}"
            )

        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="stub")
            result = generate_notes(
                slides_ocr,
                transcript,
                alignment,
                settings,
                chunk_config=SectionChunkConfig(max_slides_per_section=1),
                section_llm_callable=fake_section_llm,
            )

        self.assertEqual(len(result.prompt_inputs), 4)
        # Section 1 has empty rolling context.
        self.assertTrue(result.prompt_inputs[0].rolling_context.is_empty())
        self.assertIn("first section of the lecture", captured_messages[0].user)

        # Section 2 sees Section 1's heading, term, thread, slide, and timestamp.
        ctx_2 = result.prompt_inputs[1].rolling_context
        self.assertFalse(ctx_2.is_empty())
        self.assertEqual(len(ctx_2.prior_section_headings), 1)
        self.assertIn("Computational Graphs", ctx_2.prior_section_headings[0])
        self.assertIn("grad_fn_1", ctx_2.key_terms)
        self.assertTrue(any("walkthrough" in t.lower() or "example" in t.lower() for t in ctx_2.ongoing_threads))
        self.assertEqual(ctx_2.last_slide, 1)
        self.assertEqual(ctx_2.last_timestamp, 60.0)
        self.assertIn("grad_fn_1", captured_messages[1].user)

        # Section 4 accumulates prior headings/terms without mutating earlier snapshots.
        self.assertEqual(len(result.prompt_inputs[0].rolling_context.prior_section_headings), 0)
        self.assertEqual(len(result.prompt_inputs[3].rolling_context.prior_section_headings), 3)
        self.assertIn("grad_fn_3", result.prompt_inputs[3].rolling_context.key_terms)
        self.assertEqual(result.prompt_inputs[3].rolling_context.last_slide, 3)


class TestCoverageCheck(unittest.TestCase):
    def test_stub_coverage_check_detects_missing_substantive_content_and_ignores_filler(self) -> None:
        cov_input = CoverageCheckInput(
            lecture_title="Databases",
            section_index=1,
            total_sections=1,
            section_kind="aligned",
            start_time=0.0,
            end_time=90.0,
            slide_numbers=(1,),
            segments=(
                SectionPromptSegment(index=0, start=0.0, end=10.0, text="Can everyone hear me? Okay, um, let's begin.", slide=1),
                SectionPromptSegment(
                    index=1,
                    start=10.0,
                    end=45.0,
                    text="Write-ahead logging persists redo records sequentially before dirty pages flush to disk.",
                    slide=1,
                ),
                SectionPromptSegment(
                    index=2,
                    start=45.0,
                    end=90.0,
                    text="Student question: Why does checkpointing bound crash recovery time?",
                    slide=1,
                ),
            ),
            generated_markdown=(
                "## Write-Ahead Logging (Slide 1) [00:00–01:30]\n\n"
                "- Write-ahead logging writes redo records sequentially before dirty pages flush to disk."
            ),
        )
        result = check_section_coverage_stub(cov_input)
        self.assertFalse(result.passed)
        # Segment 0 is operational chatter (ignored); segment 1 covered; segment 2 missing!
        self.assertEqual(result.total_segment_count, 2)
        self.assertEqual(result.covered_segment_count, 1)
        self.assertEqual(len(result.missing_items), 1)
        self.assertEqual(result.missing_items[0].category, "student_qa")
        self.assertEqual(result.missing_items[0].segment_indices, (2,))
        self.assertTrue(result.to_warnings())


class TestPhase6LLMHooks(unittest.TestCase):
    def test_non_stub_mode_without_callable_raises_actionable_stage_error(self) -> None:
        prompt_input = SectionPromptInput(
            lecture_title="Compilers",
            section_index=1,
            total_sections=1,
            section_kind="aligned",
            start_time=0.0,
            end_time=30.0,
            slide_numbers=(1,),
            slides=(SectionPromptSlide(slide=1, visible_text="Lexing"),),
            segments=(SectionPromptSegment(index=0, start=0.0, end=30.0, text="Lexers tokenize source characters.", slide=1),),
        )
        cov_input = CoverageCheckInput(
            lecture_title="Compilers",
            section_index=1,
            total_sections=1,
            section_kind="aligned",
            start_time=0.0,
            end_time=30.0,
            segments=prompt_input.segments,
            generated_markdown="## Lexing (Slide 1) [00:00–00:30]\n- Lexers tokenize source characters.",
            slide_numbers=(1,),
        )

        with tempfile.TemporaryDirectory() as tmp:
            live_settings = _make_settings(Path(tmp), pipeline_llm_mode="openai")

            with self.assertRaises(StageError) as notes_err:
                generate_section_notes_with_llm(prompt_input, live_settings)
            self.assertEqual(notes_err.exception.stage, "notes")
            self.assertIn("Phase 6", notes_err.exception.hint)
            self.assertIn("PIPELINE_LLM_MODE=stub", notes_err.exception.hint)

            with self.assertRaises(StageError) as cov_err:
                check_coverage_with_llm(cov_input, live_settings)
            self.assertEqual(cov_err.exception.stage, "notes")
            self.assertIn("Phase 6", cov_err.exception.hint)

    def test_injectable_callables_run_in_non_stub_mode_and_surface_coverage_warnings(self) -> None:
        slides_ocr, transcript, alignment = _sample_lecture_artifacts()

        def section_llm(messages: PromptMessages, payload: SectionPromptInput, settings: Settings) -> str:
            return render_section_notes_stub(payload)

        def coverage_llm(messages: PromptMessages, payload: CoverageCheckInput, settings: Settings) -> dict[str, object]:
            if payload.section_kind == "off_slide":
                return {
                    "passed": False,
                    "covered_segment_count": 0,
                    "total_segment_count": len(payload.segments),
                    "missing_items": [
                        {
                            "category": "student_qa",
                            "start_time": payload.start_time,
                            "end_time": payload.end_time,
                            "segment_indices": [seg.index for seg in payload.segments],
                            "summary": "Verify full quorum partition answer is retained.",
                            "severity": "warning",
                        }
                    ],
                    "notes": "Flagged for verification.",
                }
            return {
                "passed": True,
                "covered_segment_count": len(payload.segments),
                "total_segment_count": len(payload.segments),
                "missing_items": [],
                "notes": "Complete.",
            }

        with tempfile.TemporaryDirectory() as tmp:
            live_settings = _make_settings(Path(tmp), pipeline_llm_mode="anthropic")
            result = generate_notes(
                slides_ocr,
                transcript,
                alignment,
                live_settings,
                section_llm_callable=section_llm,
                coverage_llm_callable=coverage_llm,
            )

        self.assertFalse(result.coverage_result.passed)
        self.assertEqual(len(result.coverage_result.missing_items), 1)
        self.assertTrue(any("student_qa" in w for w in result.warnings))

    def test_empty_llm_output_raises_stage_error(self) -> None:
        prompt_input = SectionPromptInput(
            lecture_title="Empty Test",
            section_index=1,
            total_sections=1,
            section_kind="off_slide",
            start_time=0.0,
            end_time=10.0,
            segments=(SectionPromptSegment(index=0, start=0.0, end=10.0, text="Demo content"),),
        )
        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(Path(tmp), pipeline_llm_mode="live")
            with self.assertRaises(StageError) as ctx:
                generate_section_notes_with_llm(
                    prompt_input,
                    settings,
                    llm_callable=lambda _m, _p, _s: "   ",
                )
        self.assertEqual(ctx.exception.stage, "notes")


class TestRunNotesStage(unittest.TestCase):
    def test_run_notes_stage_end_to_end_stub_mode(self) -> None:
        slides_ocr, transcript, alignment = _sample_lecture_artifacts()
        progress_log: list[str] = []

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = _make_settings(tmp_path, pipeline_llm_mode="stub")
            context = _make_context(tmp_path, progress_log=progress_log)
            _write_artifacts(context, slides_ocr, transcript, alignment)

            stage_result = run_notes_stage(context, settings)

            self.assertTrue(context.notes_path.is_file())
            markdown = context.notes_path.read_text(encoding="utf-8")

        self.assertEqual(stage_result.artifacts, {"notes_md": ARTIFACT_NOTES_MD})
        self.assertGreater(stage_result.stats["word_count"], 80)
        self.assertEqual(stage_result.stats["section_count"], 4)
        self.assertEqual(stage_result.stats["aligned_sections"], 2)
        self.assertEqual(stage_result.stats["off_slide_sections"], 1)
        self.assertEqual(stage_result.stats["uncertain_sections"], 1)
        self.assertTrue(stage_result.stats["coverage_passed"])
        self.assertTrue(progress_log)
        self.assertTrue(any("out_of_order_jump" in w for w in stage_result.warnings))

        # Check NotebookLM-ready Markdown structure & content fidelity.
        self.assertTrue(markdown.startswith("# Raft Consensus Algorithm"))
        self.assertIn("VISUAL_CONTEXT", markdown)
        self.assertIn("timeout ~ Uniform(150ms, 300ms)", markdown)
        self.assertIn("AppendEntries(term, leaderId, prevLogIndex, prevLogTerm, entries[])", markdown)
        self.assertIn("Off-Slide Discussion & Student Q&A [01:35–02:30]", markdown)
        self.assertIn("Unanchored Lecture Segment (Timestamp-Only) [03:50–05:10]", markdown)
        # Dubious slide 4 must NOT be forced as a slide anchor in notes.md.
        self.assertNotIn("Slide 4", markdown)
        # Uncovered Slide 5 is preserved as slide-only content.
        self.assertIn("## Unaligned Slides (Slide Content Only)", markdown)
        self.assertIn("### Slide 5: Further Reading", markdown)

    def test_missing_or_corrupt_upstream_artifacts_raise_stage_error(self) -> None:
        slides_ocr, transcript, alignment = _sample_lecture_artifacts()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = _make_settings(tmp_path)
            context = _make_context(tmp_path)

            # Missing slides_ocr.json
            with self.assertRaises(StageError) as err_ocr:
                load_notes_inputs(context)
            self.assertEqual(err_ocr.exception.stage, "notes")
            self.assertIn("ocr", err_ocr.exception.hint.lower())

            # Write valid slides_ocr, missing transcript.json
            context.slides_ocr_path.write_text(slides_ocr.model_dump_json(), encoding="utf-8")
            with self.assertRaises(StageError) as err_tr:
                load_notes_inputs(context)
            self.assertEqual(err_tr.exception.stage, "notes")
            self.assertIn("transcription", err_tr.exception.hint.lower())

            # Write valid transcript, corrupt alignment.json
            context.transcript_path.write_text(transcript.model_dump_json(), encoding="utf-8")
            context.alignment_path.write_text("{not-valid-json", encoding="utf-8")
            with self.assertRaises(StageError) as err_al:
                run_notes_stage(context, settings)
            self.assertEqual(err_al.exception.stage, "notes")
            self.assertIn("alignment", err_al.exception.hint.lower())

    def test_empty_lecture_artifacts_produce_valid_markdown(self) -> None:
        slides_ocr = SlidesOCRArtifact(source_pdf="empty_deck.pdf", slide_count=0, slides=[])
        transcript = TranscriptArtifact(source_recording="silent.mp3", duration_seconds=0.0, segments=[])
        alignment = AlignmentArtifact(mode="stub", entries=[], alignment_issues=[])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings = _make_settings(tmp_path, pipeline_llm_mode="stub")
            context = _make_context(tmp_path, slides_filename="empty_deck.pdf")
            _write_artifacts(context, slides_ocr, transcript, alignment)

            result = run_notes_stage(context, settings)
            content = context.notes_path.read_text(encoding="utf-8")

        self.assertEqual(result.stats["section_count"], 0)
        self.assertGreater(result.stats["word_count"], 0)
        self.assertIn("# empty deck", content)


if __name__ == "__main__":
    unittest.main()
