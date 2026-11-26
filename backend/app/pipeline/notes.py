from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Sequence

from pydantic import ValidationError

from ..config import (
    DEFAULT_LLM_NOTES_PARALLEL_WORKERS,
    DEFAULT_NOTES_MAX_SLIDES_PER_SECTION,
    DEFAULT_NOTES_MAX_SECTION_DURATION_SECONDS,
    DEFAULT_NOTES_MAX_SECTION_TRANSCRIPT_CHARS,
    DEFAULT_NOTES_INLINE_OFF_SLIDE_MAX_SECONDS,
    DEFAULT_NOTES_INLINE_OFF_SLIDE_MAX_CHARS,
    DEFAULT_LLM_NOTES_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_NOTES_MODEL,
    DEFAULT_LLM_NOTES_THINKING_LEVEL,
    Settings,
)
from ..schemas import (
    ARTIFACT_NOTES_MD,
    AlignmentArtifact,
    AlignmentEntry,
    AlignmentIssue,
    JobContext,
    ProgressCallback,
    SlideOCRRecord,
    SlidesOCRArtifact,
    StageError,
    StageResult,
    TranscriptArtifact,
    TranscriptSegment,
)
from .gemini_client import (
    GeminiTextResult,
    GenAIClientFactory,
    call_gemini_text,
    close_gemini_client,
    create_gemini_client,
    is_gemini_live_mode,
    validate_gemini_credentials,
)
from .prompts import (
    PromptMessages,
    RollingContext,
    SectionKind,
    SectionPromptInput,
    SectionPromptSegment,
    SectionPromptSlide,
    build_section_notes_prompt,
    format_timestamp,
    format_timestamp_range,
)

DEFAULT_MAX_SLIDES_PER_SECTION = DEFAULT_NOTES_MAX_SLIDES_PER_SECTION
DEFAULT_MAX_SECTION_DURATION_SECONDS = DEFAULT_NOTES_MAX_SECTION_DURATION_SECONDS
DEFAULT_MAX_SECTION_TRANSCRIPT_CHARS = DEFAULT_NOTES_MAX_SECTION_TRANSCRIPT_CHARS
DEFAULT_INLINE_OFF_SLIDE_MAX_SECONDS = DEFAULT_NOTES_INLINE_OFF_SLIDE_MAX_SECONDS
DEFAULT_INLINE_OFF_SLIDE_MAX_CHARS = DEFAULT_NOTES_INLINE_OFF_SLIDE_MAX_CHARS

SectionNotesLLMCallable = Callable[[PromptMessages, SectionPromptInput, Settings], str]

# Issue codes that describe global/informational deck conditions rather than making an
# already-aligned slide's placement dubious.
_NON_DUBIOUS_ISSUE_CODES = frozenset(
    {
        "uncovered_slides",
        "uncovered_slide",
        "uncovered_transcript_segments",
        "missing_slides",
        "empty_slides",
        "empty_transcript",
        "empty_span_text",
        "no_alignment_entries",
        "short_substantive_window",
        "suspiciously_wide_span",
        "stub_mode",
    }
)

_CAPITALIZED_TERM_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}(?:\s+[A-Z][A-Za-z0-9_-]{2,})?\b")
_BACKTICK_TERM_RE = re.compile(r"`([^`]{2,48})`")
_THREAD_SIGNAL_RE = re.compile(
    r"\b(?:example|walkthrough|derivation|proof|demo|question|q&a|step\s+\d+|continue|next slide)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SectionChunkConfig:
    """Limits governing how contiguous aligned slides and spans are grouped into sections."""

    max_slides_per_section: int = DEFAULT_MAX_SLIDES_PER_SECTION
    max_duration_seconds: float = DEFAULT_MAX_SECTION_DURATION_SECONDS
    max_transcript_chars: int = DEFAULT_MAX_SECTION_TRANSCRIPT_CHARS

    inline_off_slide_max_seconds: float = DEFAULT_INLINE_OFF_SLIDE_MAX_SECONDS
    inline_off_slide_max_chars: int = DEFAULT_INLINE_OFF_SLIDE_MAX_CHARS

    def __post_init__(self) -> None:
        if self.inline_off_slide_max_seconds < 0 or self.inline_off_slide_max_chars < 0:
            raise ValueError("inline off-slide limits must be nonnegative")
        if self.max_slides_per_section < 1:
            raise ValueError("max_slides_per_section must be >= 1")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be > 0")
        if self.max_transcript_chars < 1:
            raise ValueError("max_transcript_chars must be >= 1")


@dataclass(frozen=True)
class _ResolvedSpan:
    kind: SectionKind
    slide: int | None
    start_time: float
    end_time: float
    segments: tuple[SectionPromptSegment, ...]
    uncertainty_reasons: tuple[str, ...] = ()
    alignment_warnings: tuple[str, ...] = ()

    @property
    def transcript_chars(self) -> int:
        return sum(len(seg.text) for seg in self.segments)


@dataclass(frozen=True)
class LectureSection:
    """One manageable lecture section ready for prompt building and Markdown generation."""

    section_index: int
    total_sections: int
    kind: SectionKind
    start_time: float
    end_time: float
    slide_numbers: tuple[int, ...]
    slides: tuple[SectionPromptSlide, ...]
    segments: tuple[SectionPromptSegment, ...]
    uncertainty_reasons: tuple[str, ...] = ()
    alignment_warnings: tuple[str, ...] = ()

    @property
    def transcript_chars(self) -> int:
        return sum(len(seg.text) for seg in self.segments)


@dataclass(frozen=True)
class NotesGenerationResult:
    """Full output of the notes pipeline stage (Markdown and metadata)."""

    markdown: str
    sections: tuple[LectureSection, ...]
    prompt_inputs: tuple[SectionPromptInput, ...]
    warnings: tuple[str, ...]
    word_count: int
    notes_input_tokens: int = 0
    notes_output_tokens: int = 0
    notes_thinking_tokens: int = 0


def _load_json_artifact(
    path: Path,
    model_cls: type[SlidesOCRArtifact] | type[TranscriptArtifact] | type[AlignmentArtifact],
    *,
    artifact_label: str,
    upstream_stage: str,
) -> Any:
    if not path.is_file():
        raise StageError(
            stage="notes",
            message=f"Required {artifact_label} artifact is missing ({path.name}).",
            hint=(
                f"Ensure the {upstream_stage} stage completed successfully and wrote "
                f"{path.name} before running Markdown notes generation."
            ),
        )
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StageError(
            stage="notes",
            message=f"Could not read {artifact_label} artifact ({path.name}): {exc}",
            hint=f"Check file permissions for {path.name} or re-run the job from the start.",
        ) from exc

    try:
        return model_cls.model_validate_json(raw_text)
    except ValidationError as exc:
        raise StageError(
            stage="notes",
            message=f"Invalid {artifact_label} artifact ({path.name}): {exc.errors()[0]['msg']}.",
            hint=(
                f"Re-run the {upstream_stage} stage so a valid {path.name} matching "
                "the pipeline schema is produced."
            ),
        ) from exc


def load_notes_inputs(
    context: JobContext,
) -> tuple[SlidesOCRArtifact, TranscriptArtifact, AlignmentArtifact]:
    """Load and validate slides_ocr.json, transcript.json, and alignment.json."""
    slides_ocr: SlidesOCRArtifact = _load_json_artifact(
        context.slides_ocr_path,
        SlidesOCRArtifact,
        artifact_label="slide OCR",
        upstream_stage="ocr",
    )
    transcript: TranscriptArtifact = _load_json_artifact(
        context.transcript_path,
        TranscriptArtifact,
        artifact_label="transcript",
        upstream_stage="transcription",
    )
    alignment: AlignmentArtifact = _load_json_artifact(
        context.alignment_path,
        AlignmentArtifact,
        artifact_label="alignment",
        upstream_stage="alignment",
    )
    return slides_ocr, transcript, alignment


def derive_lecture_title(
    slides_ocr: SlidesOCRArtifact,
    *,
    fallback_filename: str = "",
) -> str:
    """Derive a clean human-readable lecture title from Slide 1 OCR or source filename."""
    if slides_ocr.slides:
        first_slide = sorted(slides_ocr.slides, key=lambda item: item.slide)[0]
        for raw_line in first_slide.visible_text.splitlines():
            cleaned = re.sub(r"^[#*\-\s]+", "", raw_line).strip()
            if len(cleaned) >= 3 and not re.fullmatch(r"(?:slide\s*)?\d+", cleaned, re.IGNORECASE):
                return cleaned[:120]

    source_name = slides_ocr.source_pdf.strip() or fallback_filename.strip()
    if source_name:
        stem = Path(source_name).stem
        readable = re.sub(r"[_\-]+", " ", stem).strip()
        if readable:
            return readable
    return "Lecture Notes"


def _build_dubious_slide_issues_map(
    alignment_issues: Sequence[AlignmentIssue],
    entries: Sequence[AlignmentEntry],
) -> dict[int, list[str]]:
    """Map slide numbers flagged by unlocalized dubious-alignment issues to their messages.

    If ``align.py`` already localized an issue onto specific ``uncertain=True`` entry/entries
    for the referenced slide(s), avoid clobbering unrelated valid entries (e.g., preserving the
    valid first occurrence of a slide while keeping its duplicate/regressing entry uncertain).
    """
    explicitly_uncertain_slides = {
        entry.slide for entry in entries if entry.uncertain and entry.slide is not None
    }
    dubious: dict[int, list[str]] = {}
    for issue in alignment_issues:
        code_norm = issue.code.strip().lower()
        if issue.severity != "error" and code_norm in _NON_DUBIOUS_ISSUE_CODES:
            continue
        if issue.slides and any(slide_num in explicitly_uncertain_slides for slide_num in issue.slides):
            continue
        for slide_num in issue.slides:
            dubious.setdefault(slide_num, []).append(f"[{issue.code}] {issue.message}")
    return dubious


def _resolve_entry_segments(
    entry: AlignmentEntry,
    segments_by_index: dict[int, TranscriptSegment],
    ordered_segments: Sequence[TranscriptSegment],
) -> list[TranscriptSegment]:
    if not ordered_segments:
        return []
    start_idx = min(entry.start_segment, entry.end_segment)
    end_idx = max(entry.start_segment, entry.end_segment)

    matched: list[TranscriptSegment] = []
    for seg_idx in range(start_idx, end_idx + 1):
        seg = segments_by_index.get(seg_idx)
        if seg is not None:
            matched.append(seg)

    if matched:
        return matched

    # Fallback when segment indices are sparse/non-contiguous: match by index range.
    return [seg for seg in ordered_segments if start_idx <= seg.index <= end_idx]


def _split_long_span(
    span: _ResolvedSpan,
    config: SectionChunkConfig,
) -> list[_ResolvedSpan]:
    """Split an overly long off-slide or uncertain span cleanly at segment boundaries."""
    if len(span.segments) <= 1:
        return [span]

    duration = max(0.0, span.end_time - span.start_time)
    if duration <= config.max_duration_seconds and span.transcript_chars <= config.max_transcript_chars:
        return [span]

    parts: list[_ResolvedSpan] = []
    current_segs: list[SectionPromptSegment] = []
    current_chars = 0

    for seg in span.segments:
        seg_len = len(seg.text)
        if current_segs:
            projected_duration = max(0.0, seg.end - current_segs[0].start)
            projected_chars = current_chars + seg_len
            if (
                projected_duration > config.max_duration_seconds
                or projected_chars > config.max_transcript_chars
            ):
                parts.append(
                    _ResolvedSpan(
                        kind=span.kind,
                        slide=span.slide,
                        start_time=current_segs[0].start,
                        end_time=current_segs[-1].end,
                        segments=tuple(current_segs),
                        uncertainty_reasons=span.uncertainty_reasons,
                        alignment_warnings=span.alignment_warnings,
                    )
                )
                current_segs = []
                current_chars = 0

        current_segs.append(seg)
        current_chars += seg_len

    if current_segs:
        parts.append(
            _ResolvedSpan(
                kind=span.kind,
                slide=span.slide,
                start_time=current_segs[0].start,
                end_time=current_segs[-1].end,
                segments=tuple(current_segs),
                uncertainty_reasons=span.uncertainty_reasons,
                alignment_warnings=span.alignment_warnings,
            )
        )

    return parts


def _to_prompt_slide(record: SlideOCRRecord) -> SectionPromptSlide:
    return SectionPromptSlide(
        slide=record.slide,
        visible_text=record.visible_text,
        visual_context=record.visual_context,
        tables_code_formulas=record.tables_code_formulas,
    )


def _extract_uncovered_segment_spans(
    ordered_segments: Sequence[TranscriptSegment],
    covered_indices: set[int],
    *,
    default_kind: SectionKind,
    reason: str,
) -> list[_ResolvedSpan]:
    """Group any transcript segments not referenced by alignment entries into chronological spans."""
    spans: list[_ResolvedSpan] = []
    current_group: list[TranscriptSegment] = []

    def _flush() -> None:
        if not current_group:
            return
        prompt_segs = tuple(
            SectionPromptSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                slide=None,
                uncertain=(default_kind == "uncertain"),
                uncertainty_reason=reason if default_kind == "uncertain" else "",
            )
            for seg in current_group
        )
        spans.append(
            _ResolvedSpan(
                kind=default_kind,
                slide=None,
                start_time=prompt_segs[0].start,
                end_time=prompt_segs[-1].end,
                segments=prompt_segs,
                uncertainty_reasons=(reason,) if default_kind == "uncertain" and reason else (),
            )
        )
        current_group.clear()

    for seg in ordered_segments:
        if seg.index in covered_indices:
            _flush()
        else:
            current_group.append(seg)
    _flush()
    return spans


def build_lecture_sections(
    slides_ocr: SlidesOCRArtifact,
    transcript: TranscriptArtifact,
    alignment: AlignmentArtifact,
    *,
    config: SectionChunkConfig | None = None,
) -> list[LectureSection]:
    """Group aligned slides, off-slide spans, and uncertain regions into manageable sections."""
    chunk_cfg = config or SectionChunkConfig()
    slides_by_num: dict[int, SlideOCRRecord] = {rec.slide: rec for rec in slides_ocr.slides}
    ordered_segments = sorted(transcript.segments, key=lambda seg: (seg.start, seg.index))
    segments_by_index: dict[int, TranscriptSegment] = {seg.index: seg for seg in ordered_segments}
    dubious_slide_issues = _build_dubious_slide_issues_map(
        alignment.alignment_issues,
        alignment.entries,
    )

    resolved_spans: list[_ResolvedSpan] = []
    covered_segment_indices: set[int] = set()

    sorted_entries = sorted(
        alignment.entries,
        key=lambda item: (item.start_time, item.start_segment, item.end_time),
    )

    for entry in sorted_entries:
        matched_raw = _resolve_entry_segments(entry, segments_by_index, ordered_segments)
        # Keep only segments not already assigned to an earlier span to prevent duplicate notes,
        # while still keeping a span alive if it carries slide/timing metadata.
        unique_raw = [seg for seg in matched_raw if seg.index not in covered_segment_indices]
        for seg in unique_raw:
            covered_segment_indices.add(seg.index)

        entry_issues: list[str] = []
        if entry.slide is not None and entry.slide in dubious_slide_issues:
            entry_issues.extend(dubious_slide_issues[entry.slide])

        missing_slide_record = entry.slide is not None and entry.slide not in slides_by_num
        inverted_bounds = (
            entry.start_segment > entry.end_segment or entry.start_time > entry.end_time
        )
        is_uncertain = bool(
            entry.uncertain or entry_issues or missing_slide_record or inverted_bounds
        )

        reasons: list[str] = []
        if entry.reason.strip():
            reasons.append(entry.reason.strip())
        if missing_slide_record and entry.slide is not None:
            reasons.append(f"Referenced slide {entry.slide} is missing from slides_ocr.json.")
        if inverted_bounds:
            reasons.append("Alignment entry has inverted segment or timestamp bounds.")
        if entry_issues:
            reasons.extend(entry_issues)

        if is_uncertain:
            kind: SectionKind = "uncertain"
            confirmed_slide: int | None = None
        elif entry.slide is None:
            kind = "off_slide"
            confirmed_slide = None
        else:
            kind = "aligned"
            confirmed_slide = entry.slide

        reason_summary = "; ".join(dict.fromkeys(reasons))
        prompt_segments = tuple(
            SectionPromptSegment(
                index=seg.index,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                slide=confirmed_slide,
                uncertain=(kind == "uncertain"),
                uncertainty_reason=reason_summary if kind == "uncertain" else "",
            )
            for seg in unique_raw
        )

        # Skip degenerate entries that had all segments consumed and have no slide content.
        if not prompt_segments and confirmed_slide is None and matched_raw:
            continue

        start_time = prompt_segments[0].start if prompt_segments else entry.start_time
        end_time = prompt_segments[-1].end if prompt_segments else max(entry.start_time, entry.end_time)

        span = _ResolvedSpan(
            kind=kind,
            slide=confirmed_slide,
            start_time=start_time,
            end_time=max(start_time, end_time),
            segments=prompt_segments,
            uncertainty_reasons=tuple(dict.fromkeys(reasons)) if kind == "uncertain" else (),
            alignment_warnings=tuple(dict.fromkeys(entry_issues)),
        )
        resolved_spans.extend(
            _split_long_span(span, chunk_cfg) if kind != "aligned" else [span]
        )

    # Ensure unreferenced transcript segments are never silently dropped.
    if ordered_segments and len(covered_segment_indices) < len(ordered_segments):
        has_slides = bool(slides_ocr.slides)
        fallback_kind: SectionKind = (
            "uncertain" if (not sorted_entries and has_slides) else "off_slide"
        )
        fallback_reason = (
            "Transcript could not be aligned to slides because alignment entries were empty."
            if fallback_kind == "uncertain"
            else "Unaligned spoken span outside mapped slide entries."
        )
        uncovered_spans = _extract_uncovered_segment_spans(
            ordered_segments,
            covered_segment_indices,
            default_kind=fallback_kind,
            reason=fallback_reason,
        )
        for span in uncovered_spans:
            resolved_spans.extend(_split_long_span(span, chunk_cfg))

    # Re-sort chronologically after merging uncovered gaps.
    resolved_spans.sort(
        key=lambda sp: (
            sp.start_time,
            sp.segments[0].index if sp.segments else -1,
            sp.end_time,
        )
    )

    # Group contiguous spans of the same kind into LectureSection chunks.
    raw_groups: list[list[_ResolvedSpan]] = []
    pos = 0
    while pos < len(resolved_spans):
        span = resolved_spans[pos]
        if not raw_groups:
            raw_groups.append([span])
            pos += 1
            continue

        current_group = raw_groups[-1]
        current_kind = current_group[0].kind

        group_start = current_group[0].start_time
        current_chars = sum(item.transcript_chars for item in current_group)
        projected_duration = max(0.0, span.end_time - group_start)
        projected_chars = current_chars + span.transcript_chars

        exceeds_duration = projected_duration > chunk_cfg.max_duration_seconds
        exceeds_chars = projected_chars > chunk_cfg.max_transcript_chars
        if span.kind != current_kind:
            if span.kind == "off_slide" and current_kind == "aligned":
                run_end = pos
                while run_end < len(resolved_spans) and resolved_spans[run_end].kind == "off_slide":
                    run_end += 1
                run = resolved_spans[pos:run_end]
                flanked_by_aligned = (
                    run_end < len(resolved_spans) and resolved_spans[run_end].kind == "aligned"
                )
                run_duration = max(0.0, run[-1].end_time - run[0].start_time)
                run_chars = sum(item.transcript_chars for item in run)
                run_projected_duration = max(0.0, run[-1].end_time - group_start)
                run_projected_chars = current_chars + run_chars
                if (
                    flanked_by_aligned
                    and run_duration <= chunk_cfg.inline_off_slide_max_seconds
                    and run_chars <= chunk_cfg.inline_off_slide_max_chars
                    and run_projected_duration <= chunk_cfg.max_duration_seconds
                    and run_projected_chars <= chunk_cfg.max_transcript_chars
                ):
                    current_group.extend(run)
                    pos = run_end
                    continue
            raw_groups.append([span])
            pos += 1
            continue

        exceeds_slides = False
        if current_kind == "aligned":
            unique_slides = {item.slide for item in current_group if item.slide is not None}
            if span.slide is not None:
                unique_slides.add(span.slide)
            exceeds_slides = len(unique_slides) > chunk_cfg.max_slides_per_section

        if exceeds_slides or exceeds_duration or exceeds_chars:
            raw_groups.append([span])
        else:
            current_group.append(span)
        pos += 1

    total_sections = len(raw_groups)
    sections: list[LectureSection] = []
    for idx, group in enumerate(raw_groups, start=1):
        kind = group[0].kind
        start_time = group[0].start_time
        end_time = max(item.end_time for item in group)

        segments_flat: list[SectionPromptSegment] = []
        uncertainty_reasons: list[str] = []
        alignment_warnings: list[str] = []
        slide_nums_ordered: list[int] = []

        for item in group:
            segments_flat.extend(item.segments)
            uncertainty_reasons.extend(item.uncertainty_reasons)
            alignment_warnings.extend(item.alignment_warnings)
            if kind == "aligned" and item.slide is not None and item.slide not in slide_nums_ordered:
                slide_nums_ordered.append(item.slide)

        section_slides: tuple[SectionPromptSlide, ...] = ()
        if kind == "aligned":
            section_slides = tuple(
                _to_prompt_slide(slides_by_num[num])
                for num in slide_nums_ordered
                if num in slides_by_num
            )

        sections.append(
            LectureSection(
                section_index=idx,
                total_sections=total_sections,
                kind=kind,
                start_time=start_time,
                end_time=max(start_time, end_time),
                slide_numbers=tuple(slide_nums_ordered) if kind == "aligned" else (),
                slides=section_slides,
                segments=tuple(segments_flat),
                uncertainty_reasons=tuple(dict.fromkeys(uncertainty_reasons)),
                alignment_warnings=tuple(dict.fromkeys(alignment_warnings)),
            )
        )

    return sections


def _extract_heading_from_markdown(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            return stripped[3:].strip()
    return fallback


def _extract_key_terms(section: LectureSection, markdown: str) -> list[str]:
    terms: list[str] = []

    for match in _BACKTICK_TERM_RE.findall(markdown):
        candidate = match.strip()
        if candidate and candidate not in terms:
            terms.append(candidate)

    source_blobs: list[str] = []
    for slide in section.slides:
        first_line = next(
            (ln.strip() for ln in slide.visible_text.splitlines() if ln.strip()),
            "",
        )
        if first_line:
            source_blobs.append(first_line)
    source_blobs.append(markdown[:1200])

    stop_phrases = {
        "Slide",
        "Slides",
        "Visible",
        "Visual",
        "Context",
        "Spoken",
        "Section",
        "Discussion",
        "Timestamp",
        "Tables",
        "Code",
        "Formulas",
        "Lecture",
        "Notes",
        "Anchor",
        "Uncertain",
        "Off-Slide",
    }

    for blob in source_blobs:
        for match in _CAPITALIZED_TERM_RE.findall(blob):
            cleaned = match.strip()
            if cleaned in stop_phrases or cleaned.split()[0] in stop_phrases:
                continue
            if len(cleaned) < 3:
                continue
            if cleaned not in terms:
                terms.append(cleaned)
            if len(terms) >= 12:
                return terms

    return terms


def _summarize_section_for_context(section: LectureSection, heading: str) -> str:
    ts_range = format_timestamp_range(section.start_time, section.end_time)
    if section.segments:
        preview_bits: list[str] = []
        char_budget = 220
        used = 0
        for seg in section.segments:
            text = " ".join(seg.text.split())
            if not text:
                continue
            if used + len(text) > char_budget and preview_bits:
                break
            preview_bits.append(text)
            used += len(text) + 1
        snippet = " ".join(preview_bits).strip()
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "..."
        return f"{heading} {ts_range}: {snippet}"
    return f"{heading} {ts_range}."


def update_rolling_context(
    rolling_context: RollingContext,
    section: LectureSection,
    generated_markdown: str,
) -> RollingContext:
    """Update rolling context in place after generating one section so boundaries stay coherent."""
    ts_range = format_timestamp_range(section.start_time, section.end_time)
    fallback_heading = f"Section {section.section_index} {ts_range}"
    heading = _extract_heading_from_markdown(generated_markdown, fallback=fallback_heading)

    rolling_context.prior_section_headings.append(heading)
    if len(rolling_context.prior_section_headings) > 12:
        rolling_context.prior_section_headings = rolling_context.prior_section_headings[-12:]

    for term in _extract_key_terms(section, generated_markdown):
        if term not in rolling_context.key_terms:
            rolling_context.key_terms.append(term)
    if len(rolling_context.key_terms) > 25:
        rolling_context.key_terms = rolling_context.key_terms[-25:]

    thread_candidates: list[str] = []
    for seg in section.segments:
        if _THREAD_SIGNAL_RE.search(seg.text):
            cleaned = " ".join(seg.text.split())
            if len(cleaned) > 140:
                cleaned = cleaned[:137].rstrip() + "..."
            thread_candidates.append(f"{format_timestamp(seg.start)} {cleaned}")
            if len(thread_candidates) >= 2:
                break
    for thread in thread_candidates:
        if thread not in rolling_context.ongoing_threads:
            rolling_context.ongoing_threads.append(thread)
    if len(rolling_context.ongoing_threads) > 6:
        rolling_context.ongoing_threads = rolling_context.ongoing_threads[-6:]

    rolling_context.last_section_summary = _summarize_section_for_context(section, heading)
    if section.kind == "aligned" and section.slide_numbers:
        rolling_context.last_slide = section.slide_numbers[-1]
    rolling_context.last_timestamp = section.end_time
    return rolling_context


def _slide_heading_from_ocr(slide: SectionPromptSlide) -> str:
    for raw_line in slide.visible_text.splitlines():
        cleaned = re.sub(r"^[#*\-\s]+", "", raw_line).strip()
        if cleaned:
            return cleaned[:100]
    return f"Slide {slide.slide}"


def _section_topic_title(prompt_input: SectionPromptInput) -> str:
    if prompt_input.section_kind == "aligned":
        titles: list[str] = []
        for slide in prompt_input.slides:
            title = _slide_heading_from_ocr(slide)
            if title and title not in titles:
                titles.append(title)
        if titles:
            if len(titles) == 1:
                return titles[0]
            return f"{titles[0]} — {titles[-1]}"
        return f"Lecture Section {prompt_input.section_index}"

    if prompt_input.section_kind == "off_slide":
        combined = " ".join(seg.text for seg in prompt_input.segments).lower()
        if "?" in combined or "question" in combined:
            return "Off-Slide Discussion & Student Q&A"
        if any(word in combined for word in ("demo", "terminal", "code", "live", "browser")):
            return "Off-Slide Live Demonstration & Walkthrough"
        return "Off-Slide Discussion"

    return "Unanchored Lecture Segment (Timestamp-Only)"


def _format_slide_anchor_suffix(slide_numbers: tuple[int, ...]) -> str:
    ordered = sorted(set(slide_numbers))
    if not ordered:
        return ""
    if len(ordered) == 1:
        return f" (Slide {ordered[0]})"
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f" (Slides {ordered[0]}–{ordered[-1]})"
    return " (Slides " + ", ".join(str(n) for n in ordered) + ")"


def render_section_notes_stub(prompt_input: SectionPromptInput) -> str:
    """Deterministically assemble NotebookLM-ready Markdown for one section in stub mode."""
    ts_range = format_timestamp_range(prompt_input.start_time, prompt_input.end_time)
    topic = _section_topic_title(prompt_input)

    if prompt_input.section_kind == "aligned":
        slide_suffix = _format_slide_anchor_suffix(prompt_input.slide_numbers)
        heading = f"## {topic}{slide_suffix} {ts_range}"
    else:
        # Both off_slide and uncertain sections strictly use timestamp-only anchors.
        heading = f"## {topic} {ts_range}"

    lines: list[str] = [heading, ""]

    if not prompt_input.rolling_context.is_empty():
        prev_heading = (
            prompt_input.rolling_context.prior_section_headings[-1]
            if prompt_input.rolling_context.prior_section_headings
            else ""
        )
        carry_parts: list[str] = []
        if prev_heading:
            carry_parts.append(f"Continues from: {prev_heading}")
        if prompt_input.rolling_context.key_terms:
            recent_terms = ", ".join(prompt_input.rolling_context.key_terms[-6:])
            carry_parts.append(f"Active concepts: {recent_terms}")
        if carry_parts:
            lines.append("> **Context Carry-Over:** " + " | ".join(carry_parts))
            lines.append("")

    if prompt_input.section_kind == "aligned":
        segments_by_slide: dict[int, list[SectionPromptSegment]] = {}
        unassigned_segments: list[SectionPromptSegment] = []
        for seg in prompt_input.segments:
            if seg.slide is not None:
                segments_by_slide.setdefault(seg.slide, []).append(seg)
            else:
                unassigned_segments.append(seg)

        pending_off_slide = sorted(unassigned_segments, key=lambda s: (s.start, s.index))

        def _flush_off_slide_before(cutoff_time: float | None) -> None:
            nonlocal pending_off_slide
            if not pending_off_slide:
                return
            if cutoff_time is None:
                to_emit = pending_off_slide
                pending_off_slide = []
            else:
                to_emit = [seg for seg in pending_off_slide if seg.start <= cutoff_time]
                pending_off_slide = [seg for seg in pending_off_slide if seg.start > cutoff_time]
            if not to_emit:
                return
            lines.append("**Off-Slide Addition (Timestamped):**")
            for seg in to_emit:
                seg_range = format_timestamp_range(seg.start, seg.end)
                lines.append(f"- {seg_range} [Off-slide] {seg.text.strip()}")
            lines.append("")

        for slide in prompt_input.slides:
            slide_segs = segments_by_slide.get(slide.slide, [])
            if slide_segs:
                _flush_off_slide_before(slide_segs[0].start)

            slide_title = _slide_heading_from_ocr(slide)
            slide_ts = (
                f" {format_timestamp_range(slide_segs[0].start, slide_segs[-1].end)}"
                if slide_segs
                else ""
            )
            lines.append(f"### Slide {slide.slide}: {slide_title}{slide_ts}")
            lines.append("")

            lines.append("**Slide Content (Visible Text):**")
            if slide.visible_text.strip():
                lines.append(slide.visible_text.strip())
            else:
                lines.append("- *(No visible slide text captured by OCR.)*")
            lines.append("")

            if slide.visual_context.strip():
                lines.append("**Visual / Diagram Context (`VISUAL_CONTEXT`):**")
                lines.append(slide.visual_context.strip())
                lines.append("")

            if slide.tables_code_formulas.strip():
                lines.append("**Tables, Code & Formulas:**")
                lines.append(slide.tables_code_formulas.strip())
                lines.append("")

            if slide_segs:
                lines.append("**Spoken Explanation & Additions (Timestamped):**")
                for seg in slide_segs:
                    seg_range = format_timestamp_range(seg.start, seg.end)
                    lines.append(f"- {seg_range} {seg.text.strip()}")
                lines.append("")

        _flush_off_slide_before(None)

    elif prompt_input.section_kind == "off_slide":
        lines.append(
            "> **Spoken-Only / Off-Slide Section:** This discussion, demonstration, or Q&A "
            f"occurred without an anchored slide ({ts_range})."
        )
        lines.append("")
        lines.append("**Spoken Transcript & Explanations (Timestamped):**")
        if prompt_input.segments:
            for seg in prompt_input.segments:
                seg_range = format_timestamp_range(seg.start, seg.end)
                lines.append(f"- {seg_range} {seg.text.strip()}")
        else:
            lines.append("- *(No transcript segments recorded for this span.)*")
        lines.append("")

    else:
        lines.append(
            "> **Alignment Uncertainty Fallback:** Slide alignment for this time span is "
            f"uncertain; anchored by timestamp {ts_range} only (dubious slide numbers omitted)."
        )
        lines.append("")
        lines.append("**Spoken Transcript & Explanations (Timestamped):**")
        if prompt_input.segments:
            for seg in prompt_input.segments:
                seg_range = format_timestamp_range(seg.start, seg.end)
                lines.append(f"- {seg_range} {seg.text.strip()}")
        else:
            lines.append("- *(No transcript segments recorded for this span.)*")
        lines.append("")

    return "\n".join(lines).strip()


def _unwrap_outer_markdown_fence(raw_text: str) -> str:
    """Unwrap an outer ```markdown ... ``` fence if the LLM wrapped the entire section."""
    cleaned = (raw_text or "").strip()
    if not cleaned.startswith("```") or not cleaned.endswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if len(lines) < 2:
        return cleaned
    opening = lines[0].strip()
    closing = lines[-1].strip()
    if not opening.startswith("```") or closing != "```":
        return cleaned
    language = opening[3:].strip().lower()
    if language not in {"", "md", "markdown"}:
        return cleaned

    body_lines = lines[1:-1]
    inner_fence_lines = sum(1 for line in body_lines if line.lstrip().startswith("```"))
    body = "\n".join(body_lines).strip()

    if language == "":
        if inner_fence_lines > 0:
            return cleaned
        first_non_empty = next((line.lstrip() for line in body_lines if line.strip()), "")
        if not first_non_empty.startswith("## "):
            return cleaned
        return body

    if inner_fence_lines % 2 != 0:
        return cleaned
    return body


def _generate_section_notes_with_usage(
    prompt_input: SectionPromptInput,
    settings: Settings,
    *,
    llm_callable: SectionNotesLLMCallable | None = None,
    client: Any | None = None,
    client_factory: GenAIClientFactory | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: Event | None = None,
) -> tuple[str, GeminiTextResult | None]:
    prompt_messages = build_section_notes_prompt(prompt_input)

    if llm_callable is not None:
        try:
            rendered = llm_callable(prompt_messages, prompt_input, settings)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                stage="notes",
                message=(
                    f"LLM section note generation failed on section "
                    f"{prompt_input.section_index}/{prompt_input.total_sections}: {exc}"
                ),
                hint="Check your LLM provider credentials, model configuration, and rate limits, then retry.",
            ) from exc

        cleaned = _unwrap_outer_markdown_fence(rendered)
        if not cleaned:
            raise StageError(
                stage="notes",
                message=(
                    f"LLM returned empty Markdown notes for section "
                    f"{prompt_input.section_index}/{prompt_input.total_sections}."
                ),
                hint="Inspect the configured LLM provider response or retry with PIPELINE_LLM_MODE=stub.",
            )
        return cleaned, None

    mode = (settings.pipeline_llm_mode or "gemini").strip().lower()
    if mode == "stub":
        return render_section_notes_stub(prompt_input), None

    if is_gemini_live_mode(mode):
        model = (
            str(getattr(settings, "llm_notes_model", DEFAULT_LLM_NOTES_MODEL) or "").strip()
            or DEFAULT_LLM_NOTES_MODEL
        )
        thinking_level = str(
            getattr(settings, "llm_notes_thinking_level", DEFAULT_LLM_NOTES_THINKING_LEVEL) or ""
        )
        max_output_tokens = int(
            getattr(settings, "llm_notes_max_output_tokens", DEFAULT_LLM_NOTES_MAX_OUTPUT_TOKENS)
            or DEFAULT_LLM_NOTES_MAX_OUTPUT_TOKENS
        )
        result = call_gemini_text(
            stage="notes",
            task_label=(
                f"notes section {prompt_input.section_index}/{prompt_input.total_sections} "
                f"({model})"
            ),
            settings=settings,
            model=model,
            thinking_level=thinking_level,
            system_instruction=prompt_messages.system,
            user_prompt=prompt_messages.user,
            max_output_tokens=max_output_tokens,
            response_mime_type=None,
            client=client,
            client_factory=client_factory,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        cleaned = _unwrap_outer_markdown_fence(result.text)
        if not cleaned:
            raise StageError(
                stage="notes",
                message=(
                    f"Gemini ({model}) returned empty Markdown notes for section "
                    f"{prompt_input.section_index}/{prompt_input.total_sections}."
                ),
                hint="Retry the job or test with PIPELINE_LLM_MODE=stub.",
            )
        return cleaned, result

    raise StageError(
        stage="notes",
        message=(
            f"Markdown note generation LLM mode '{settings.pipeline_llm_mode}' is not supported."
        ),
        hint=(
            "Phase 6 wired Google Gemini into generate_section_notes_with_llm(). "
            "Set PIPELINE_LLM_MODE=gemini (or PIPELINE_LLM_MODE=live) with GEMINI_API_KEY, "
            "set PIPELINE_LLM_MODE=stub in .env to run deterministic local scaffolding, "
            "or pass an explicit llm_callable."
        ),
    )


def generate_section_notes_with_llm(
    prompt_input: SectionPromptInput,
    settings: Settings,
    *,
    llm_callable: SectionNotesLLMCallable | None = None,
    client: Any | None = None,
    client_factory: GenAIClientFactory | None = None,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Generate one section's NotebookLM-optimized Markdown notes via Gemini, callable, or stub."""
    markdown, _ = _generate_section_notes_with_usage(
        prompt_input,
        settings,
        llm_callable=llm_callable,
        client=client,
        client_factory=client_factory,
        progress_callback=progress_callback,
    )
    return markdown


def _render_uncovered_slides_section(
    slides_ocr: SlidesOCRArtifact,
    alignment: AlignmentArtifact,
    aligned_slide_numbers: set[int],
) -> str:
    """Render slides that have zero entries in alignment.json (slide-only content)."""
    referenced_in_alignment = {
        entry.slide for entry in alignment.entries if entry.slide is not None
    }
    uncovered = [
        slide
        for slide in sorted(slides_ocr.slides, key=lambda rec: rec.slide)
        if slide.slide not in aligned_slide_numbers and slide.slide not in referenced_in_alignment
    ]
    if not uncovered:
        return ""

    lines: list[str] = [
        "## Unaligned Slides (Slide Content Only)",
        "",
        "> **Slide-Only Content:** No spoken transcript segments were aligned to the following slide(s).",
        "",
    ]
    for slide in uncovered:
        prompt_slide = _to_prompt_slide(slide)
        title = _slide_heading_from_ocr(prompt_slide)
        lines.append(f"### Slide {slide.slide}: {title}")
        lines.append("")
        if slide.visible_text.strip():
            lines.append("**Slide Content (Visible Text):**")
            lines.append(slide.visible_text.strip())
            lines.append("")
        if slide.visual_context.strip():
            lines.append("**Visual / Diagram Context (`VISUAL_CONTEXT`):**")
            lines.append(slide.visual_context.strip())
            lines.append("")
        if slide.tables_code_formulas.strip():
            lines.append("**Tables, Code & Formulas:**")
            lines.append(slide.tables_code_formulas.strip())
            lines.append("")
    return "\n".join(lines).strip()


def _assemble_document_markdown(
    *,
    lecture_title: str,
    slides_ocr: SlidesOCRArtifact,
    transcript: TranscriptArtifact,
    alignment: AlignmentArtifact,
    section_markdowns: Sequence[str],
    uncovered_slides_markdown: str,
    mode: str,
) -> str:
    duration = transcript.duration_seconds
    if duration is None and transcript.segments:
        duration = max(seg.end for seg in transcript.segments)
    duration_label = format_timestamp(duration) if duration is not None else "00:00"
    slide_count = slides_ocr.slide_count or len(slides_ocr.slides)

    header_lines: list[str] = [
        f"# {lecture_title}",
        "",
        "## Lecture Metadata",
        f"- **Slides PDF:** `{slides_ocr.source_pdf or 'slides.pdf'}` ({slide_count} slide(s))",
        f"- **Recording:** `{transcript.source_recording or 'recording'}` (Duration: {duration_label})",
        f"- **Sections:** {len(section_markdowns)}",
        f"- **Notes Mode:** `{mode}`",
    ]

    if alignment.alignment_issues:
        header_lines.extend(
            [
                "",
                "> **Alignment Notice:** Some lecture spans had uncertain slide alignment and "
                "use `[MM:SS–MM:SS]` timestamp-only anchors instead of dubious slide numbers.",
            ]
        )

    blocks: list[str] = ["\n".join(header_lines).strip()]
    for md in section_markdowns:
        cleaned = md.strip()
        if cleaned:
            blocks.append(cleaned)

    if uncovered_slides_markdown.strip():
        blocks.append(uncovered_slides_markdown.strip())

    if len(blocks) == 1:
        blocks.append(
            "## Lecture Content\n\n"
            "*(No slide text or spoken transcript segments were available for this lecture.)*"
        )

    return "\n\n---\n\n".join(blocks).rstrip() + "\n"


def _make_section_prompt(
    section: LectureSection,
    lecture_title: str,
    rolling_context: RollingContext,
    outline: tuple[str, ...] = (),
    boundary: tuple[SectionPromptSegment, ...] = (),
) -> SectionPromptInput:
    return SectionPromptInput(
        lecture_title=lecture_title,
        section_index=section.section_index,
        total_sections=section.total_sections,
        section_kind=section.kind,
        start_time=section.start_time,
        end_time=section.end_time,
        slide_numbers=section.slide_numbers,
        slides=section.slides,
        segments=section.segments,
        rolling_context=rolling_context.snapshot(),
        alignment_warnings=section.alignment_warnings,
        lecture_outline=outline,
        prior_boundary_segments=boundary,
        uncertainty_reasons=section.uncertainty_reasons,
    )


def _build_parallel_prompt_inputs(
    sections: Sequence[LectureSection], lecture_title: str, slides_ocr: SlidesOCRArtifact
) -> list[SectionPromptInput]:
    outline = tuple(
        f"Slide {slide.slide}: {_slide_heading_from_ocr(_to_prompt_slide(slide))}"
        for slide in sorted(slides_ocr.slides, key=lambda slide: slide.slide)
    )
    context = RollingContext()
    boundary: tuple[SectionPromptSegment, ...] = ()
    prompts = []
    for section in sections:
        section_outline = outline if section.kind == "aligned" else ()
        prompt = _make_section_prompt(section, lecture_title, context, section_outline, boundary)
        prompts.append(prompt)
        update_rolling_context(context, section, "## " + _section_topic_title(prompt))
        boundary = tuple(sorted((*boundary, *section.segments), key=lambda seg: (seg.start, seg.index))[-4:])
    return prompts


class _CombinedCancelEvent:
    """Views a local section-failure event and an optional job-level cancel event together."""

    def __init__(self, local: Event, external: Event | None) -> None:
        self._local = local
        self._external = external

    def is_set(self) -> bool:
        return self._local.is_set() or (self._external is not None and self._external.is_set())

    def wait(self, timeout: float | None = None) -> bool:
        if self._external is None:
            return self._local.wait(timeout=timeout)
        if self.is_set():
            return True
        if timeout is None:
            while not self.is_set():
                if self._local.wait(timeout=0.1) or self._external.is_set():
                    return True
            return True
        import time as _time
        deadline = _time.monotonic() + max(0.0, timeout)
        while not self.is_set():
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                return self.is_set()
            if self._local.wait(timeout=min(0.1, remaining)) or self._external.is_set():
                return True
        return True


def generate_notes(
    slides_ocr: SlidesOCRArtifact,
    transcript: TranscriptArtifact,
    alignment: AlignmentArtifact,
    settings: Settings,
    *,
    lecture_title: str | None = None,
    chunk_config: SectionChunkConfig | None = None,
    section_llm_callable: SectionNotesLLMCallable | None = None,
    client_factory: GenAIClientFactory | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_fraction_callback: Callable[[float], None] | None = None,
    cancel_event: Event | None = None,
) -> NotesGenerationResult:
    """Orchestrate section chunking, rolling context carry-over, and section generation."""
    mode = (settings.pipeline_llm_mode or "gemini").strip().lower()
    use_gemini = is_gemini_live_mode(mode)

    if mode != "stub" and not use_gemini and section_llm_callable is None:
        raise StageError(
            stage="notes",
            message=(
                f"Markdown note generation LLM mode '{settings.pipeline_llm_mode}' is not supported."
            ),
            hint=(
                "Phase 6 wired Google Gemini into generate_section_notes_with_llm(). "
                "Set PIPELINE_LLM_MODE=gemini (or PIPELINE_LLM_MODE=live) with GEMINI_API_KEY, "
                "set PIPELINE_LLM_MODE=stub in .env to run deterministic local scaffolding, "
                "or pass an explicit section_llm_callable."
            ),
        )

    shared_client: Any | None = None
    owns_client = False
    if use_gemini and section_llm_callable is None:
        validate_gemini_credentials(settings, stage="notes")
        factory = client_factory or create_gemini_client
        try:
            shared_client = factory(settings)
            owns_client = client_factory is None
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                stage="notes",
                message=f"Failed to initialize Google Gemini client for notes stage: {exc}",
                hint=(
                    "Verify GEMINI_API_KEY (or Vertex AI config) in .env, restart the backend, "
                    "and retry."
                ),
            ) from exc

    resolved_title = (lecture_title or "").strip() or derive_lecture_title(slides_ocr)

    rolling_context = RollingContext()
    prompt_inputs: list[SectionPromptInput] = []
    section_markdowns: list[str] = []
    aligned_slide_numbers: set[int] = set()

    notes_input_tokens = 0
    notes_output_tokens = 0
    notes_thinking_tokens = 0

    try:
        if progress_callback is not None:
            progress_callback("Chunking aligned slides and transcript into lecture sections...")

        sections = build_lecture_sections(
            slides_ocr,
            transcript,
            alignment,
            config=chunk_config or SectionChunkConfig(
                max_slides_per_section=settings.notes_max_slides_per_section,
                max_duration_seconds=settings.notes_max_section_duration_seconds,
                max_transcript_chars=settings.notes_max_section_transcript_chars,
                inline_off_slide_max_seconds=settings.notes_inline_off_slide_max_seconds,
                inline_off_slide_max_chars=settings.notes_inline_off_slide_max_chars,
            ),
        )

        total_sections = len(sections)
        notes_workers = getattr(settings, "llm_notes_parallel_workers", DEFAULT_LLM_NOTES_PARALLEL_WORKERS)
        parallel = (
            use_gemini
            and section_llm_callable is None
            and notes_workers > 1
            and total_sections > 1
        )
        local_cancel = Event()
        effective_cancel: Any = (
            _CombinedCancelEvent(local_cancel, cancel_event) if cancel_event is not None else local_cancel
        )

        def generate(prompt: SectionPromptInput):
            return _generate_section_notes_with_usage(
                prompt, settings, llm_callable=section_llm_callable, client=shared_client,
                client_factory=client_factory, progress_callback=report, cancel_event=effective_cancel,
            )

        callback_lock = Lock()

        def report(message: str) -> None:
            if progress_callback:
                with callback_lock:
                    progress_callback(message)

        notes = {}
        if parallel:
            prompt_inputs = _build_parallel_prompt_inputs(sections, resolved_title, slides_ocr)
            with ThreadPoolExecutor(max_workers=min(notes_workers, total_sections)) as writers:
                drafts = {writers.submit(generate, p): section for p, section in zip(prompt_inputs, sections)}
                try:
                    for future in as_completed(drafts):
                        section = drafts[future]
                        idx = section.section_index
                        notes[idx] = future.result()
                        if progress_fraction_callback is not None:
                            progress_fraction_callback(len(notes) / total_sections)
                        report(f"Generated notes for {len(notes)}/{total_sections} sections...")
                except BaseException:
                    local_cancel.set()
                    for future in drafts:
                        future.cancel()
                    raise
        else:
            for section in sections:
                if effective_cancel.is_set():
                    raise StageError(
                        stage="notes",
                        message="Note generation cancelled by user.",
                        hint="Retry the lecture if you still want Markdown notes.",
                    )
                idx = section.section_index
                report(f"Generating notes for section {idx} of {total_sections}...")
                prompt = _make_section_prompt(section, resolved_title, rolling_context)
                prompt_inputs.append(prompt)
                notes[idx] = generate(prompt)
                if progress_fraction_callback is not None:
                    progress_fraction_callback(len(notes) / total_sections)
                markdown = notes[idx][0]
                update_rolling_context(rolling_context, section, markdown)

        # Workers return usage with their results. Only the coordinator mutates totals.
        for section in sections:
            idx = section.section_index
            markdown, usage = notes[idx]
            section_markdowns.append(markdown)
            aligned_slide_numbers.update(section.slide_numbers)
            if usage:
                notes_input_tokens += usage.input_tokens
                notes_output_tokens += usage.output_tokens
                notes_thinking_tokens += usage.thinking_tokens
    finally:
        if owns_client:
            close_gemini_client(shared_client)

    if progress_callback is not None:
        progress_callback("Assembling final NotebookLM-ready Markdown document...")

    uncovered_slides_md = _render_uncovered_slides_section(
        slides_ocr,
        alignment,
        aligned_slide_numbers,
    )
    resolved_mode = "gemini" if (use_gemini and section_llm_callable is None) else mode
    markdown = _assemble_document_markdown(
        lecture_title=resolved_title,
        slides_ocr=slides_ocr,
        transcript=transcript,
        alignment=alignment,
        section_markdowns=section_markdowns,
        uncovered_slides_markdown=uncovered_slides_md,
        mode=resolved_mode,
    )

    warnings: list[str] = []
    for issue in alignment.alignment_issues:
        warning_text = f"Alignment ({issue.code}): {issue.message}"
        if warning_text not in warnings:
            warnings.append(warning_text)
    word_count = len(markdown.split())
    return NotesGenerationResult(
        markdown=markdown,
        sections=tuple(sections),
        prompt_inputs=tuple(prompt_inputs),
        warnings=tuple(warnings),
        word_count=word_count,
        notes_input_tokens=notes_input_tokens,
        notes_output_tokens=notes_output_tokens,
        notes_thinking_tokens=notes_thinking_tokens,
    )


def run_notes_stage(
    context: JobContext,
    settings: Settings,
    *,
    client_factory: GenAIClientFactory | None = None,
) -> StageResult:
    """Pipeline stage handler: generate artifacts/notes.md from OCR, transcript, and alignment."""
    context.report_progress("Loading slide OCR, transcript, and alignment artifacts...")
    slides_ocr, transcript, alignment = load_notes_inputs(context)

    result = generate_notes(
        slides_ocr,
        transcript,
        alignment,
        settings,
        lecture_title=derive_lecture_title(
            slides_ocr,
            fallback_filename=context.slides_filename,
        ),
        client_factory=client_factory,
        progress_callback=context.report_progress,
        progress_fraction_callback=context.progress_fraction_callback,
        cancel_event=context.cancel_event,
    )

    context.notes_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = context.notes_path.with_name(f".{context.notes_path.stem}.tmp.md")
    temp_path.write_text(result.markdown, encoding="utf-8")
    os.replace(temp_path, context.notes_path)

    aligned_count = sum(1 for sec in result.sections if sec.kind == "aligned")
    off_slide_count = sum(1 for sec in result.sections if sec.kind == "off_slide")
    uncertain_count = sum(1 for sec in result.sections if sec.kind == "uncertain")

    section_count = len(result.sections)
    return StageResult(
        message=(
            f"Generated NotebookLM-ready Markdown notes "
            f"({section_count} section(s), {result.word_count} words)."
        ),
        artifacts={"notes_md": ARTIFACT_NOTES_MD},
        stats={
            "word_count": result.word_count,
            "section_count": section_count,
            "aligned_sections": aligned_count,
            "off_slide_sections": off_slide_count,
            "uncertain_sections": uncertain_count,
            "notes_input_tokens": result.notes_input_tokens,
            "notes_output_tokens": result.notes_output_tokens,
            "notes_thinking_tokens": result.notes_thinking_tokens,
        },
        warnings=list(result.warnings),
    )


__all__ = [
    "DEFAULT_MAX_SECTION_DURATION_SECONDS",
    "DEFAULT_MAX_SECTION_TRANSCRIPT_CHARS",
    "DEFAULT_MAX_SLIDES_PER_SECTION",
    "LectureSection",
    "NotesGenerationResult",
    "SectionChunkConfig",
    "SectionNotesLLMCallable",
    "build_lecture_sections",
    "derive_lecture_title",
    "generate_notes",
    "generate_section_notes_with_llm",
    "load_notes_inputs",
    "render_section_notes_stub",
    "run_notes_stage",
    "update_rolling_context",
]
