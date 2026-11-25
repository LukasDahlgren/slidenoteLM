from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import ValidationError

from ..config import Settings
from ..schemas import (
    ARTIFACT_ALIGNMENT,
    AlignmentArtifact,
    AlignmentEntry,
    AlignmentIssue,
    JobContext,
    PipelineStage,
    ProgressCallback,
    SlideOCRRecord,
    SlidesOCRArtifact,
    StageError,
    StageResult,
    TranscriptArtifact,
)

STAGE_NAME: PipelineStage = "alignment"

# Structural alignment check thresholds (adapted & simplified from TeachersNote).
SUBSTANTIVE_SLIDE_MIN_WORDS = 8
MIN_SUBSTANTIVE_DURATION_SECONDS = 10.0
WIDE_SPAN_MIN_SEGMENTS = 30
WIDE_SPAN_SEGMENT_MULTIPLIER = 4.0
WIDE_SPAN_MAX_FRACTION = 0.6

_TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "and",
        "are",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "can",
        "could",
        "does",
        "doing",
        "each",
        "for",
        "from",
        "have",
        "having",
        "here",
        "into",
        "its",
        "just",
        "more",
        "most",
        "not",
        "now",
        "only",
        "other",
        "our",
        "out",
        "over",
        "same",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)

AlignmentLLMResult = AlignmentArtifact | Sequence[AlignmentEntry] | Sequence[dict[str, Any]]
AlignmentLLMCallable = Callable[[SlidesOCRArtifact, TranscriptArtifact, Settings], AlignmentLLMResult]


def slide_ocr_text(slide: SlideOCRRecord) -> str:
    """Return combined structured OCR text for a slide (falling back to raw_text)."""
    parts = [
        slide.visible_text.strip(),
        slide.visual_context.strip(),
        slide.tables_code_formulas.strip(),
    ]
    combined = "\n".join(part for part in parts if part)
    return combined if combined else slide.raw_text.strip()


def _ordered_slides(slides: SlidesOCRArtifact) -> list[SlideOCRRecord]:
    """Return slides sorted by ascending slide number, deduplicating repeated slide numbers."""
    seen: set[int] = set()
    ordered: list[SlideOCRRecord] = []
    for record in sorted(slides.slides, key=lambda item: item.slide):
        if record.slide in seen:
            continue
        seen.add(record.slide)
        ordered.append(record)
    return ordered


def _tokenize_content(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOPWORDS
    }


def _append_reason(existing: str, addition: str) -> str:
    clean_existing = existing.strip()
    clean_addition = addition.strip()
    if not clean_addition:
        return clean_existing
    if not clean_existing:
        return clean_addition
    if clean_addition in clean_existing:
        return clean_existing
    return f"{clean_existing}; {clean_addition}"


def load_transcript_artifact(path: Path) -> TranscriptArtifact:
    """Load and validate artifacts/transcript.json or raise an actionable StageError."""
    if not path.exists() or not path.is_file():
        raise StageError(
            stage=STAGE_NAME,
            message=f"Required transcript artifact is missing: {path.name}.",
            hint=(
                "Ensure the Transcription (Groq) stage completed successfully and wrote "
                "artifacts/transcript.json before running alignment."
            ),
        )
    try:
        raw_text = path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
        return TranscriptArtifact.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise StageError(
            stage=STAGE_NAME,
            message=f"Transcript artifact {path.name} is invalid: {exc}",
            hint="Re-run the Transcription (Groq) stage to regenerate a valid artifacts/transcript.json.",
        ) from exc


def load_slides_ocr_artifact(path: Path) -> SlidesOCRArtifact:
    """Load and validate artifacts/slides_ocr.json or raise an actionable StageError."""
    if not path.exists() or not path.is_file():
        raise StageError(
            stage=STAGE_NAME,
            message=f"Required slide OCR artifact is missing: {path.name}.",
            hint=(
                "Ensure the Slide OCR (Gemini) stage completed successfully and wrote "
                "artifacts/slides_ocr.json before running alignment."
            ),
        )
    try:
        raw_text = path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
        return SlidesOCRArtifact.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise StageError(
            stage=STAGE_NAME,
            message=f"Slide OCR artifact {path.name} is invalid: {exc}",
            hint="Re-run the Slide OCR (Gemini) stage to regenerate a valid artifacts/slides_ocr.json.",
        ) from exc


def _resolve_times_for_segment_range(
    transcript: TranscriptArtifact,
    start_segment: int,
    end_segment: int,
) -> tuple[float, float]:
    segments = transcript.segments
    if not segments:
        return 0.0, 0.0
    if 0 <= start_segment < len(segments):
        start_time = round(max(0.0, float(segments[start_segment].start)), 2)
    else:
        start_time = 0.0
    if 0 <= end_segment < len(segments):
        end_time = round(max(0.0, float(segments[end_segment].end)), 2)
    else:
        end_time = start_time
    return start_time, end_time


def normalize_alignment_entries(
    raw_entries: Sequence[AlignmentEntry | dict[str, Any]],
    transcript: TranscriptArtifact,
    slides: SlidesOCRArtifact,
) -> tuple[list[AlignmentEntry], list[AlignmentIssue]]:
    """Coerce raw entry dicts or AlignmentEntry objects into validated entries plus parse issues."""
    del slides  # Kept for signature symmetry; slide membership is validated in quality checks.
    normalized: list[AlignmentEntry] = []
    parse_issues: list[AlignmentIssue] = []

    for idx, item in enumerate(raw_entries):
        if isinstance(item, AlignmentEntry):
            normalized.append(item.model_copy())
            continue
        if not isinstance(item, dict):
            parse_issues.append(
                AlignmentIssue(
                    code="malformed_alignment_entry",
                    message=f"Alignment entry at index {idx} was not an object and was skipped.",
                    slides=[],
                    severity="error",
                )
            )
            continue

        raw_slide = item.get("slide", None)
        slide_val: int | None = None
        uncertain = bool(item.get("uncertain", False))
        reason = str(item.get("reason", "") or "").strip()

        if raw_slide is not None:
            try:
                parsed_slide = int(raw_slide)
            except (TypeError, ValueError):
                uncertain = True
                reason = _append_reason(
                    reason,
                    f"Non-numeric slide identifier '{raw_slide}'; cleared to timestamp-only anchor.",
                )
                parse_issues.append(
                    AlignmentIssue(
                        code="unknown_slide",
                        message=f"Alignment entry {idx} had non-numeric slide '{raw_slide}'.",
                        slides=[],
                        severity="error",
                    )
                )
            else:
                if parsed_slide < 1:
                    uncertain = True
                    reason = _append_reason(
                        reason,
                        f"Invalid slide number {parsed_slide}; cleared to timestamp-only anchor.",
                    )
                    parse_issues.append(
                        AlignmentIssue(
                            code="unknown_slide",
                            message=f"Alignment entry {idx} referenced invalid slide number {parsed_slide}.",
                            slides=[],
                            severity="error",
                        )
                    )
                else:
                    slide_val = parsed_slide

        try:
            raw_start_seg = int(item["start_segment"])
            raw_end_seg = int(item["end_segment"])
        except (KeyError, TypeError, ValueError):
            parse_issues.append(
                AlignmentIssue(
                    code="malformed_alignment_entry",
                    message=f"Alignment entry {idx} is missing integer start_segment/end_segment.",
                    slides=[slide_val] if slide_val is not None else [],
                    severity="error",
                )
            )
            continue

        if raw_start_seg < 0 or raw_end_seg < 0:
            uncertain = True
            reason = _append_reason(
                reason,
                f"Negative segment bounds ({raw_start_seg}..{raw_end_seg}); clamped for validation.",
            )
            parse_issues.append(
                AlignmentIssue(
                    code="invalid_segment_range",
                    message=(
                        f"Alignment entry {idx} had negative segment bounds "
                        f"[{raw_start_seg}, {raw_end_seg}]."
                    ),
                    slides=[slide_val] if slide_val is not None else [],
                    severity="error",
                )
            )
            start_seg = max(0, raw_start_seg)
            end_seg = max(0, raw_end_seg)
        else:
            start_seg = raw_start_seg
            end_seg = raw_end_seg

        inferred_start, inferred_end = _resolve_times_for_segment_range(
            transcript, start_seg, end_seg
        )
        raw_start_time = item.get("start_time")
        raw_end_time = item.get("end_time")
        try:
            start_time = (
                round(max(0.0, float(raw_start_time)), 2)
                if raw_start_time is not None
                else inferred_start
            )
            end_time = (
                round(max(0.0, float(raw_end_time)), 2)
                if raw_end_time is not None
                else inferred_end
            )
        except (TypeError, ValueError):
            start_time = inferred_start
            end_time = inferred_end

        normalized.append(
            AlignmentEntry(
                slide=slide_val,
                start_segment=start_seg,
                end_segment=end_seg,
                start_time=start_time,
                end_time=end_time,
                uncertain=uncertain,
                reason=reason,
            )
        )

    return normalized, parse_issues


def _inspect_alignment_quality(
    entries: Sequence[AlignmentEntry],
    transcript: TranscriptArtifact,
    slides: SlidesOCRArtifact,
    *,
    substantive_min_words: int = SUBSTANTIVE_SLIDE_MIN_WORDS,
    min_substantive_duration_seconds: float = MIN_SUBSTANTIVE_DURATION_SECONDS,
    wide_span_min_segments: int = WIDE_SPAN_MIN_SEGMENTS,
    wide_span_multiplier: float = WIDE_SPAN_SEGMENT_MULTIPLIER,
    wide_span_max_fraction: float = WIDE_SPAN_MAX_FRACTION,
) -> tuple[list[AlignmentIssue], dict[int, list[str]], set[int]]:
    """Run structural checks and return (issues, per_entry_uncertain_reasons, clear_slide_indices)."""
    ordered_slides = _ordered_slides(slides)
    expected_slide_numbers = [record.slide for record in ordered_slides]
    expected_slide_set = set(expected_slide_numbers)
    substantive_slides = {
        record.slide
        for record in ordered_slides
        if len(slide_ocr_text(record).split()) >= substantive_min_words
    }

    segments = transcript.segments
    num_segments = len(segments)
    num_slides = len(expected_slide_numbers)

    issues: list[AlignmentIssue] = []
    uncertain_reasons: dict[int, list[str]] = {}
    clear_slide_indices: set[int] = set()

    def _mark_uncertain(entry_idx: int, reason_text: str) -> None:
        bucket = uncertain_reasons.setdefault(entry_idx, [])
        if reason_text not in bucket:
            bucket.append(reason_text)

    if num_segments == 0:
        issues.append(
            AlignmentIssue(
                code="empty_transcript",
                message="Transcript contains 0 segments; slide alignment cannot anchor to spoken audio.",
                slides=[],
                severity="warning",
            )
        )

    if num_slides == 0:
        issues.append(
            AlignmentIssue(
                code="empty_slides",
                message="Slide deck contains 0 OCR slides; transcript spans will use timestamp-only anchors.",
                slides=[],
                severity="warning",
            )
        )

    if num_segments > 0 and num_slides > 0 and not entries:
        issues.append(
            AlignmentIssue(
                code="no_alignment_entries",
                message=(
                    f"Alignment produced 0 entries for {num_slides} slide(s) and "
                    f"{num_segments} transcript segment(s)."
                ),
                slides=list(expected_slide_numbers),
                severity="warning",
            )
        )

    covered_slides: set[int] = set()
    seen_slide_first_idx: dict[int, int] = {}
    covered_segments: set[int] = set()

    prev_valid_entry_idx: int | None = None
    prev_valid_entry: AlignmentEntry | None = None

    prev_slide: int | None = None
    prev_slide_entry_idx: int | None = None
    prev_slide_was_forward_jump = False
    max_seen_slide: int | None = None

    avg_segments_per_slide = (num_segments / num_slides) if num_slides > 0 else float(num_segments)

    for idx, entry in enumerate(entries):
        slide_num = entry.slide
        slide_list = [slide_num] if slide_num is not None else []
        label = f"Slide {slide_num}" if slide_num is not None else f"Off-slide entry #{idx + 1}"

        # 1. Unknown slide membership check
        slide_known = True
        if slide_num is not None and slide_num not in expected_slide_set:
            slide_known = False
            issues.append(
                AlignmentIssue(
                    code="unknown_slide",
                    message=(
                        f"Entry #{idx + 1} references slide {slide_num}, which is not in "
                        f"slides_ocr.json ({num_slides} slide(s)); falling back to timestamp-only."
                    ),
                    slides=[slide_num],
                    severity="error",
                )
            )
            _mark_uncertain(
                idx,
                f"Slide {slide_num} does not exist in slide deck; cleared to timestamp-only anchor.",
            )
            clear_slide_indices.add(idx)

        # 2. Segment range validity
        bounds_valid = (
            num_segments > 0
            and 0 <= entry.start_segment <= entry.end_segment < num_segments
        )
        if not bounds_valid:
            issues.append(
                AlignmentIssue(
                    code="invalid_segment_range",
                    message=(
                        f"{label} has invalid segment range [{entry.start_segment}, {entry.end_segment}] "
                        f"for transcript with {num_segments} segment(s)."
                    ),
                    slides=slide_list,
                    severity="error",
                )
            )
            _mark_uncertain(
                idx,
                (
                    f"Invalid segment range [{entry.start_segment}, {entry.end_segment}] "
                    f"(transcript has {num_segments} segment(s)); use timestamp-only anchor."
                ),
            )
        else:
            covered_segments.update(range(entry.start_segment, entry.end_segment + 1))

            # 3. Time range, zero-duration, short substantive window, empty text, wide span
            seg_start_time = float(segments[entry.start_segment].start)
            seg_end_time = float(segments[entry.end_segment].end)
            duration = seg_end_time - seg_start_time

            if entry.end_time < entry.start_time:
                issues.append(
                    AlignmentIssue(
                        code="invalid_time_range",
                        message=(
                            f"{label} has inverted timestamps "
                            f"[{entry.start_time:.2f}s, {entry.end_time:.2f}s]."
                        ),
                        slides=slide_list,
                        severity="warning",
                    )
                )
                _mark_uncertain(
                    idx,
                    f"Inverted timestamps ({entry.start_time:.2f}s > {entry.end_time:.2f}s); use timestamp-only anchor.",
                )

            if duration <= 0.0:
                issues.append(
                    AlignmentIssue(
                        code="zero_duration",
                        message=(
                            f"{label} covers a zero-duration transcript span "
                            f"(segments {entry.start_segment}..{entry.end_segment})."
                        ),
                        slides=slide_list,
                        severity="warning",
                    )
                )
                _mark_uncertain(
                    idx,
                    "Zero-duration aligned span; use timestamp-only anchor.",
                )
            elif (
                slide_num is not None
                and slide_known
                and slide_num in substantive_slides
                and duration < min_substantive_duration_seconds
            ):
                issues.append(
                    AlignmentIssue(
                        code="short_substantive_window",
                        message=(
                            f"Slide {slide_num} is substantive (>={substantive_min_words} words) "
                            f"but aligned to only {duration:.1f}s (< {min_substantive_duration_seconds:.0f}s)."
                        ),
                        slides=[slide_num],
                        severity="warning",
                    )
                )

            span_has_text = any(
                segments[seg_idx].text.strip()
                for seg_idx in range(entry.start_segment, entry.end_segment + 1)
            )
            if not span_has_text:
                issues.append(
                    AlignmentIssue(
                        code="empty_span_text",
                        message=(
                            f"{label} spans segments {entry.start_segment}..{entry.end_segment} "
                            "with no non-empty spoken text."
                        ),
                        slides=slide_list,
                        severity="warning",
                    )
                )

            span_segments = entry.end_segment - entry.start_segment + 1
            if slide_num is not None and slide_known and num_slides >= 2:
                wide_by_count = span_segments > max(
                    wide_span_min_segments,
                    avg_segments_per_slide * wide_span_multiplier,
                )
                wide_by_fraction = (
                    num_slides >= 3
                    and num_segments >= 10
                    and (span_segments / num_segments) > wide_span_max_fraction
                )
                if wide_by_count or wide_by_fraction:
                    pct = (span_segments / num_segments) * 100.0
                    issues.append(
                        AlignmentIssue(
                            code="suspiciously_wide_span",
                            message=(
                                f"Slide {slide_num} covers {span_segments}/{num_segments} segments "
                                f"({pct:.0f}% of transcript); verify whether slide transitions were missed."
                            ),
                            slides=[slide_num],
                            severity="warning",
                        )
                    )

            # 4. Segment ordering & overlap against previous valid entry
            if prev_valid_entry is not None and prev_valid_entry_idx is not None:
                involved_slides = [
                    s
                    for s in (prev_valid_entry.slide, slide_num)
                    if s is not None
                ]
                if entry.start_segment < prev_valid_entry.start_segment:
                    issues.append(
                        AlignmentIssue(
                            code="non_monotonic_segments",
                            message=(
                                f"{label} starts at segment {entry.start_segment}, earlier than "
                                f"previous entry start segment {prev_valid_entry.start_segment}."
                            ),
                            slides=involved_slides,
                            severity="error",
                        )
                    )
                    _mark_uncertain(
                        idx,
                        (
                            f"Non-monotonic segment order (start {entry.start_segment} < "
                            f"prior start {prev_valid_entry.start_segment}); use timestamp-only anchor."
                        ),
                    )
                    _mark_uncertain(
                        prev_valid_entry_idx,
                        "Adjacent to non-monotonic segment range; use timestamp-only anchor.",
                    )
                elif entry.start_segment <= prev_valid_entry.end_segment:
                    issues.append(
                        AlignmentIssue(
                            code="overlapping_segment_range",
                            message=(
                                f"{label} (segments {entry.start_segment}..{entry.end_segment}) "
                                f"overlaps prior entry (segments "
                                f"{prev_valid_entry.start_segment}..{prev_valid_entry.end_segment})."
                            ),
                            slides=involved_slides,
                            severity="error",
                        )
                    )
                    _mark_uncertain(
                        idx,
                        (
                            f"Overlaps prior segment range "
                            f"[{prev_valid_entry.start_segment}, {prev_valid_entry.end_segment}]; "
                            "use timestamp-only anchor."
                        ),
                    )
                    _mark_uncertain(
                        prev_valid_entry_idx,
                        (
                            f"Overlaps following segment range "
                            f"[{entry.start_segment}, {entry.end_segment}]; "
                            "use timestamp-only anchor."
                        ),
                    )

            prev_valid_entry = entry
            prev_valid_entry_idx = idx

        # 5. Slide monotonicity & duplicate slide checks (for known slide entries only)
        if slide_num is not None and slide_known:
            covered_slides.add(slide_num)

            if slide_num in seen_slide_first_idx:
                issues.append(
                    AlignmentIssue(
                        code="duplicate_slide",
                        message=(
                            f"Slide {slide_num} appears more than once in alignment "
                            f"(entries #{seen_slide_first_idx[slide_num] + 1} and #{idx + 1})."
                        ),
                        slides=[slide_num],
                        severity="warning",
                    )
                )
                _mark_uncertain(
                    idx,
                    f"Duplicate slide {slide_num} assignment; use timestamp-only anchor.",
                )
            else:
                seen_slide_first_idx[slide_num] = idx

            if prev_slide is not None and slide_num < prev_slide:
                issues.append(
                    AlignmentIssue(
                        code="out_of_order_slide",
                        message=(
                            f"Out-of-order slide regression from slide {prev_slide} to slide {slide_num} "
                            f"at entry #{idx + 1}; use timestamp-only anchor."
                        ),
                        slides=[prev_slide, slide_num],
                        severity="warning",
                    )
                )
                _mark_uncertain(
                    idx,
                    (
                        f"Out-of-order slide jump from slide {prev_slide} to slide {slide_num}; "
                        "use timestamp-only anchor."
                    ),
                )
                if prev_slide_was_forward_jump and prev_slide_entry_idx is not None:
                    _mark_uncertain(
                        prev_slide_entry_idx,
                        (
                            f"Out-of-order forward jump to slide {prev_slide} before regressing "
                            f"to slide {slide_num}; use timestamp-only anchor."
                        ),
                    )
            elif max_seen_slide is not None and slide_num < max_seen_slide:
                issues.append(
                    AlignmentIssue(
                        code="out_of_order_slide",
                        message=(
                            f"Slide {slide_num} at entry #{idx + 1} remains out of order after "
                            f"earlier slide {max_seen_slide}; use timestamp-only anchor."
                        ),
                        slides=[max_seen_slide, slide_num],
                        severity="warning",
                    )
                )
                _mark_uncertain(
                    idx,
                    (
                        f"Out-of-order region (slide {slide_num} follows earlier slide "
                        f"{max_seen_slide}); use timestamp-only anchor."
                    ),
                )

            if prev_slide is None:
                prev_slide_was_forward_jump = (
                    bool(expected_slide_numbers) and slide_num > expected_slide_numbers[0]
                )
            else:
                prev_slide_was_forward_jump = slide_num > prev_slide + 1

            prev_slide = slide_num
            prev_slide_entry_idx = idx
            if max_seen_slide is None or slide_num > max_seen_slide:
                max_seen_slide = slide_num

    # 6. Uncovered / skipped slides check
    if expected_slide_numbers and (entries or num_segments == 0):
        missing_slides = [s for s in expected_slide_numbers if s not in covered_slides]
        if missing_slides:
            slide_preview = ", ".join(str(s) for s in missing_slides)
            issues.append(
                AlignmentIssue(
                    code="uncovered_slides",
                    message=(
                        f"{len(missing_slides)} slide(s) were not covered by any alignment entry: "
                        f"slide(s) {slide_preview}."
                    ),
                    slides=missing_slides,
                    severity="warning",
                )
            )

    # 7. Uncovered transcript segments check
    if num_segments > 0 and entries:
        uncovered_count = num_segments - len(covered_segments)
        if uncovered_count > 0:
            issues.append(
                AlignmentIssue(
                    code="uncovered_transcript_segments",
                    message=(
                        f"{uncovered_count} of {num_segments} transcript segment(s) are not "
                        "covered by any alignment entry."
                    ),
                    slides=[],
                    severity="warning",
                )
            )

    return issues, uncertain_reasons, clear_slide_indices


def check_alignment_quality(
    entries: Sequence[AlignmentEntry],
    transcript: TranscriptArtifact,
    slides: SlidesOCRArtifact,
    *,
    substantive_min_words: int = SUBSTANTIVE_SLIDE_MIN_WORDS,
    min_substantive_duration_seconds: float = MIN_SUBSTANTIVE_DURATION_SECONDS,
    wide_span_min_segments: int = WIDE_SPAN_MIN_SEGMENTS,
    wide_span_multiplier: float = WIDE_SPAN_SEGMENT_MULTIPLIER,
    wide_span_max_fraction: float = WIDE_SPAN_MAX_FRACTION,
) -> list[AlignmentIssue]:
    """Inspect alignment entries for structural defects and quality warnings."""
    issues, _, _ = _inspect_alignment_quality(
        entries,
        transcript,
        slides,
        substantive_min_words=substantive_min_words,
        min_substantive_duration_seconds=min_substantive_duration_seconds,
        wide_span_min_segments=wide_span_min_segments,
        wide_span_multiplier=wide_span_multiplier,
        wide_span_max_fraction=wide_span_max_fraction,
    )
    return issues


def alignment_issues(
    entries: Sequence[AlignmentEntry],
    transcript: TranscriptArtifact,
    slides: SlidesOCRArtifact,
    *,
    substantive_min_words: int = SUBSTANTIVE_SLIDE_MIN_WORDS,
    min_substantive_duration_seconds: float = MIN_SUBSTANTIVE_DURATION_SECONDS,
    wide_span_min_segments: int = WIDE_SPAN_MIN_SEGMENTS,
    wide_span_multiplier: float = WIDE_SPAN_SEGMENT_MULTIPLIER,
    wide_span_max_fraction: float = WIDE_SPAN_MAX_FRACTION,
) -> list[AlignmentIssue]:
    """Alias for check_alignment_quality matching TeachersNote terminology."""
    return check_alignment_quality(
        entries,
        transcript,
        slides,
        substantive_min_words=substantive_min_words,
        min_substantive_duration_seconds=min_substantive_duration_seconds,
        wide_span_min_segments=wide_span_min_segments,
        wide_span_multiplier=wide_span_multiplier,
        wide_span_max_fraction=wide_span_max_fraction,
    )


def annotate_alignment_entries(
    entries: Sequence[AlignmentEntry],
    transcript: TranscriptArtifact,
    slides: SlidesOCRArtifact,
    *,
    extra_issues: Sequence[AlignmentIssue] = (),
    substantive_min_words: int = SUBSTANTIVE_SLIDE_MIN_WORDS,
    min_substantive_duration_seconds: float = MIN_SUBSTANTIVE_DURATION_SECONDS,
    wide_span_min_segments: int = WIDE_SPAN_MIN_SEGMENTS,
    wide_span_multiplier: float = WIDE_SPAN_SEGMENT_MULTIPLIER,
    wide_span_max_fraction: float = WIDE_SPAN_MAX_FRACTION,
) -> tuple[list[AlignmentEntry], list[AlignmentIssue]]:
    """Run quality checks and mark uncertain/out-of-order entries for timestamp-only fallback."""
    detected_issues, uncertain_reasons, clear_slide_indices = _inspect_alignment_quality(
        entries,
        transcript,
        slides,
        substantive_min_words=substantive_min_words,
        min_substantive_duration_seconds=min_substantive_duration_seconds,
        wide_span_min_segments=wide_span_min_segments,
        wide_span_multiplier=wide_span_multiplier,
        wide_span_max_fraction=wide_span_max_fraction,
    )

    annotated: list[AlignmentEntry] = []
    for idx, entry in enumerate(entries):
        reasons_for_entry = uncertain_reasons.get(idx, [])
        slide_val = None if idx in clear_slide_indices else entry.slide
        is_uncertain = entry.uncertain or bool(reasons_for_entry)
        reason_text = entry.reason.strip()

        for extra_reason in reasons_for_entry:
            reason_text = _append_reason(reason_text, extra_reason)

        if is_uncertain and not reason_text:
            reason_text = "Uncertain slide alignment; use timestamp-only anchor."
        elif slide_val is None and not reason_text:
            reason_text = "Off-slide / unaligned transcript span (timestamp-only anchor)."

        annotated.append(
            AlignmentEntry(
                slide=slide_val,
                start_segment=entry.start_segment,
                end_segment=entry.end_segment,
                start_time=entry.start_time,
                end_time=entry.end_time,
                uncertain=is_uncertain,
                reason=reason_text,
            )
        )

    combined_issues: list[AlignmentIssue] = []
    seen_issue_keys: set[tuple[str, tuple[int, ...], str]] = set()
    for issue in (*extra_issues, *detected_issues):
        key = (issue.code, tuple(issue.slides), issue.message)
        if key in seen_issue_keys:
            continue
        seen_issue_keys.add(key)
        combined_issues.append(issue)
    return annotated, combined_issues


def format_alignment_warnings(issues: Sequence[AlignmentIssue]) -> list[str]:
    """Format AlignmentIssue objects into human-readable StageResult warning strings."""
    return [f"{issue.message} ({issue.code})" for issue in issues]


def align_deterministic_stub(
    slides: SlidesOCRArtifact,
    transcript: TranscriptArtifact,
) -> list[AlignmentEntry]:
    """Deterministically align transcript segments to slides using lexical overlap + proportional DP.

    Handles all edge cases cleanly:
    - 0 transcript segments -> []
    - 0 slides -> single explicit off-slide entry (slide=None) covering all segments
    - Fewer segments than slides -> 1 segment per best-matching monotonic slide subset (no overlaps)
    - Segments >= slides -> contiguous monotonic partition covering all slides and all segments
    """
    segments = transcript.segments
    num_segments = len(segments)
    if num_segments == 0:
        return []

    ordered_slides = _ordered_slides(slides)
    num_slides = len(ordered_slides)
    if num_slides == 0:
        return [
            AlignmentEntry(
                slide=None,
                start_segment=0,
                end_segment=num_segments - 1,
                start_time=round(max(0.0, float(segments[0].start)), 2),
                end_time=round(max(0.0, float(segments[-1].end)), 2),
                uncertain=False,
                reason="No slides available; treating full transcript as timestamp-only off-slide span.",
            )
        ]

    if num_slides == 1:
        return [
            AlignmentEntry(
                slide=ordered_slides[0].slide,
                start_segment=0,
                end_segment=num_segments - 1,
                start_time=round(max(0.0, float(segments[0].start)), 2),
                end_time=round(max(0.0, float(segments[-1].end)), 2),
                uncertain=False,
                reason="",
            )
        ]

    slide_tokens = [_tokenize_content(slide_ocr_text(record)) for record in ordered_slides]
    seg_tokens = [_tokenize_content(segment.text) for segment in segments]

    def _cell_score(seg_idx: int, slide_idx: int) -> float:
        overlap = float(min(4, len(slide_tokens[slide_idx] & seg_tokens[seg_idx])))
        expected_slide = ((seg_idx + 0.5) * num_slides / num_segments) - 0.5
        proportional_affinity = -0.01 * ((slide_idx - expected_slide) ** 2)
        return overlap + proportional_affinity

    neg_inf = float("-inf")

    # Case 1: Fewer segments than slides -> assign each segment to a strictly increasing slide.
    if num_segments < num_slides:
        dp = [[neg_inf] * num_slides for _ in range(num_segments)]
        parent = [[-1] * num_slides for _ in range(num_segments)]

        for k in range(num_slides - num_segments + 1):
            dp[0][k] = _cell_score(0, k)

        for i in range(1, num_segments):
            best_prev_score = neg_inf
            best_prev_k = -1
            max_k = num_slides - (num_segments - i)
            for k in range(i, max_k + 1):
                if dp[i - 1][k - 1] > best_prev_score:
                    best_prev_score = dp[i - 1][k - 1]
                    best_prev_k = k - 1
                if best_prev_k != -1:
                    dp[i][k] = best_prev_score + _cell_score(i, k)
                    parent[i][k] = best_prev_k

        best_end_k = max(range(num_segments - 1, num_slides), key=lambda k: dp[num_segments - 1][k])
        chosen_slides = [0] * num_segments
        curr_k = best_end_k
        for i in range(num_segments - 1, -1, -1):
            chosen_slides[i] = curr_k
            curr_k = parent[i][curr_k]

        return [
            AlignmentEntry(
                slide=ordered_slides[chosen_slides[i]].slide,
                start_segment=i,
                end_segment=i,
                start_time=round(max(0.0, float(segments[i].start)), 2),
                end_time=round(max(0.0, float(segments[i].end)), 2),
                uncertain=False,
                reason="",
            )
            for i in range(num_segments)
        ]

    # Case 2: At least as many segments as slides -> contiguous monotonic partition covering all slides.
    dp = [[neg_inf] * num_slides for _ in range(num_segments)]
    parent = [[-1] * num_slides for _ in range(num_segments)]
    dp[0][0] = _cell_score(0, 0)

    for i in range(1, num_segments):
        min_k = max(0, num_slides - (num_segments - i))
        max_k = min(i, num_slides - 1)
        for k in range(min_k, max_k + 1):
            stay_score = dp[i - 1][k]
            advance_score = dp[i - 1][k - 1] if k > 0 else neg_inf
            if advance_score > stay_score:
                dp[i][k] = advance_score + _cell_score(i, k)
                parent[i][k] = k - 1
            elif stay_score > neg_inf:
                dp[i][k] = stay_score + _cell_score(i, k)
                parent[i][k] = k

    assigned_slide_idx = [0] * num_segments
    curr_k = num_slides - 1
    for i in range(num_segments - 1, -1, -1):
        assigned_slide_idx[i] = curr_k
        if i > 0:
            curr_k = parent[i][curr_k]

    entries: list[AlignmentEntry] = []
    span_start = 0
    for i in range(1, num_segments + 1):
        if i == num_segments or assigned_slide_idx[i] != assigned_slide_idx[span_start]:
            slide_idx = assigned_slide_idx[span_start]
            span_end = i - 1
            entries.append(
                AlignmentEntry(
                    slide=ordered_slides[slide_idx].slide,
                    start_segment=span_start,
                    end_segment=span_end,
                    start_time=round(max(0.0, float(segments[span_start].start)), 2),
                    end_time=round(max(0.0, float(segments[span_end].end)), 2),
                    uncertain=False,
                    reason="",
                )
            )
            span_start = i

    return entries


def align_with_llm(
    slides: SlidesOCRArtifact,
    transcript: TranscriptArtifact,
    settings: Settings,
    *,
    llm_callable: AlignmentLLMCallable | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AlignmentArtifact:
    """Align transcript segments to slides via injected LLM callable or deterministic stub mode.

    When PIPELINE_LLM_MODE != 'stub' and no llm_callable is wired, raises an actionable
    StageError noting that Phase 6 wires the chosen LLM provider.
    """
    mode = (settings.pipeline_llm_mode or "stub").strip().lower() or "stub"

    if llm_callable is not None:
        if progress_callback is not None:
            progress_callback("Aligning transcript segments to slides with LLM...")
        try:
            raw_result = llm_callable(slides, transcript, settings)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                stage=STAGE_NAME,
                message=f"Alignment LLM call failed: {exc}",
                hint=(
                    "Verify the LLM provider credentials and response format, or switch "
                    "PIPELINE_LLM_MODE=stub to test end-to-end with deterministic alignment."
                ),
            ) from exc

        extra_issues: list[AlignmentIssue] = []
        if isinstance(raw_result, AlignmentArtifact):
            raw_entries: Sequence[AlignmentEntry | dict[str, Any]] = raw_result.entries
            extra_issues.extend(raw_result.alignment_issues)
            artifact_mode = raw_result.mode or (mode if mode != "stub" else "llm")
        else:
            raw_entries = raw_result
            artifact_mode = mode if mode != "stub" else "llm"

        normalized_entries, parse_issues = normalize_alignment_entries(
            raw_entries, transcript, slides
        )
        annotated_entries, all_issues = annotate_alignment_entries(
            normalized_entries,
            transcript,
            slides,
            extra_issues=[*extra_issues, *parse_issues],
        )
        return AlignmentArtifact(
            mode=artifact_mode,
            entries=annotated_entries,
            alignment_issues=all_issues,
        )

    if mode == "stub":
        if progress_callback is not None:
            progress_callback("Running deterministic stub alignment...")
        stub_entries = align_deterministic_stub(slides, transcript)
        annotated_entries, all_issues = annotate_alignment_entries(
            stub_entries, transcript, slides
        )
        return AlignmentArtifact(
            mode="stub",
            entries=annotated_entries,
            alignment_issues=all_issues,
        )

    raise StageError(
        stage=STAGE_NAME,
        message=f"Alignment LLM mode '{settings.pipeline_llm_mode}' is not wired yet.",
        hint=(
            "Phase 6 will wire the chosen LLM provider into backend/app/pipeline/align.py "
            "(align_with_llm). Set PIPELINE_LLM_MODE=stub in .env to run the pipeline "
            "end-to-end using deterministic stub alignment."
        ),
    )


def run_alignment_stage(
    context: JobContext,
    settings: Settings,
    *,
    llm_callable: AlignmentLLMCallable | None = None,
) -> StageResult:
    """Execute the alignment pipeline stage and persist artifacts/alignment.json."""
    context.report_progress("Loading transcript and slide OCR artifacts...")
    transcript = load_transcript_artifact(context.transcript_path)
    slides = load_slides_ocr_artifact(context.slides_ocr_path)

    artifact = align_with_llm(
        slides,
        transcript,
        settings,
        llm_callable=llm_callable,
        progress_callback=context.report_progress,
    )

    context.alignment_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = context.alignment_path.with_name(f".{context.alignment_path.stem}.tmp.json")
    temp_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temp_path, context.alignment_path)

    warnings = format_alignment_warnings(artifact.alignment_issues)
    aligned_slides = len(
        {entry.slide for entry in artifact.entries if entry.slide is not None and not entry.uncertain}
    )
    uncertain_entries = sum(1 for entry in artifact.entries if entry.uncertain)
    off_slide_entries = sum(1 for entry in artifact.entries if entry.slide is None)

    summary = (
        f"Aligned {len(artifact.entries)} span(s) across {aligned_slides} slide(s) "
        f"in {artifact.mode} mode"
    )
    if warnings:
        summary += f" with {len(warnings)} warning(s)."
    else:
        summary += "."
    context.report_progress(summary)

    return StageResult(
        message=summary,
        artifacts={"alignment": ARTIFACT_ALIGNMENT},
        stats={
            "alignment_mode": artifact.mode,
            "aligned_entries": len(artifact.entries),
            "aligned_slides": aligned_slides,
            "uncertain_entries": uncertain_entries,
            "off_slide_entries": off_slide_entries,
            "alignment_issue_count": len(artifact.alignment_issues),
        },
        warnings=warnings,
    )


__all__ = [
    "AlignmentLLMCallable",
    "AlignmentLLMResult",
    "MIN_SUBSTANTIVE_DURATION_SECONDS",
    "STAGE_NAME",
    "SUBSTANTIVE_SLIDE_MIN_WORDS",
    "WIDE_SPAN_MAX_FRACTION",
    "WIDE_SPAN_MIN_SEGMENTS",
    "WIDE_SPAN_SEGMENT_MULTIPLIER",
    "align_deterministic_stub",
    "align_with_llm",
    "alignment_issues",
    "annotate_alignment_entries",
    "check_alignment_quality",
    "format_alignment_warnings",
    "load_slides_ocr_artifact",
    "load_transcript_artifact",
    "normalize_alignment_entries",
    "run_alignment_stage",
    "slide_ocr_text",
]
