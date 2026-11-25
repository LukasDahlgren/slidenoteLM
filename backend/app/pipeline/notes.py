from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import ValidationError

from ..config import Settings
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
from .prompts import (
    CoverageCheckInput,
    CoverageCheckResult,
    CoverageGap,
    PromptMessages,
    RollingContext,
    SectionKind,
    SectionPromptInput,
    SectionPromptSegment,
    SectionPromptSlide,
    build_coverage_check_prompt,
    build_section_notes_prompt,
    format_timestamp,
    format_timestamp_range,
    parse_coverage_check_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_SLIDES_PER_SECTION = 4
DEFAULT_MAX_SECTION_DURATION_SECONDS = 480.0
DEFAULT_MAX_SECTION_TRANSCRIPT_CHARS = 6000

SectionNotesLLMCallable = Callable[[PromptMessages, SectionPromptInput, Settings], str]
CoverageCheckLLMCallable = Callable[
    [PromptMessages, CoverageCheckInput, Settings],
    CoverageCheckResult | dict[str, Any] | str,
]

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

# Common spoken filler words stripped when checking substantive coverage in stub mode.
_FILLER_TOKENS = frozenset(
    {
        "a",
        "ah",
        "all",
        "am",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "do",
        "does",
        "er",
        "for",
        "from",
        "go",
        "going",
        "got",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "him",
        "his",
        "hmm",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "know",
        "let",
        "like",
        "me",
        "my",
        "no",
        "not",
        "now",
        "of",
        "oh",
        "ok",
        "okay",
        "on",
        "one",
        "or",
        "our",
        "out",
        "really",
        "right",
        "say",
        "see",
        "she",
        "so",
        "some",
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
        "to",
        "uh",
        "um",
        "up",
        "us",
        "very",
        "was",
        "we",
        "well",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "would",
        "yeah",
        "yes",
        "you",
        "your",
    }
)

_OPERATIONAL_CHATTER_RE = re.compile(
    r"^(?:can (?:you|everyone) hear me|testing(?: one two)?|is the mic on|"
    r"let'?s take a (?:short |quick )?break|we'?ll be right back|"
    r"welcome back everyone|all right(?: let'?s get started)?|hello everyone)\b",
    flags=re.IGNORECASE,
)

_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./+-]*")
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

    def __post_init__(self) -> None:
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
    """Full output of the notes pipeline stage (Markdown + coverage + metadata)."""

    markdown: str
    sections: tuple[LectureSection, ...]
    prompt_inputs: tuple[SectionPromptInput, ...]
    coverage_result: CoverageCheckResult
    warnings: tuple[str, ...]
    word_count: int


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
    for span in resolved_spans:
        if not raw_groups:
            raw_groups.append([span])
            continue

        current_group = raw_groups[-1]
        current_kind = current_group[0].kind
        if span.kind != current_kind:
            raw_groups.append([span])
            continue

        group_start = current_group[0].start_time
        projected_duration = max(0.0, span.end_time - group_start)
        projected_chars = sum(item.transcript_chars for item in current_group) + span.transcript_chars

        exceeds_duration = projected_duration > chunk_cfg.max_duration_seconds
        exceeds_chars = projected_chars > chunk_cfg.max_transcript_chars
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

        for slide in prompt_input.slides:
            slide_segs = segments_by_slide.get(slide.slide, [])
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

        if unassigned_segments:
            lines.append("**Additional Spoken Explanation (Timestamped):**")
            for seg in unassigned_segments:
                seg_range = format_timestamp_range(seg.start, seg.end)
                lines.append(f"- {seg_range} {seg.text.strip()}")
            lines.append("")

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


def _extract_substantive_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _WORD_TOKEN_RE.findall(text.lower()):
        cleaned = raw.strip(".,;:!?()[]{}'\"")
        if len(cleaned) < 3:
            continue
        if cleaned in _FILLER_TOKENS:
            continue
        tokens.add(cleaned)
    return tokens


def _is_substantive_segment(text: str) -> bool:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return False
    if _OPERATIONAL_CHATTER_RE.match(cleaned) and len(cleaned.split()) <= 10:
        return False
    tokens = _extract_substantive_tokens(cleaned)
    return len(tokens) >= 2


def check_section_coverage_stub(payload: CoverageCheckInput) -> CoverageCheckResult:
    """Deterministic lexical coverage check verifying substantive spoken segments appear in notes."""
    markdown_tokens = _extract_substantive_tokens(payload.generated_markdown)
    total_substantive = 0
    covered_substantive = 0
    gaps: list[CoverageGap] = []

    for seg in payload.segments:
        if not _is_substantive_segment(seg.text):
            continue
        total_substantive += 1
        seg_tokens = _extract_substantive_tokens(seg.text)
        overlap = seg_tokens & markdown_tokens
        ratio = (len(overlap) / len(seg_tokens)) if seg_tokens else 1.0
        if ratio >= 0.4 or (len(seg_tokens) <= 3 and len(overlap) >= 1):
            covered_substantive += 1
        else:
            category = "student_qa" if "?" in seg.text else "explanation"
            snippet = " ".join(seg.text.split())
            if len(snippet) > 120:
                snippet = snippet[:117].rstrip() + "..."
            gaps.append(
                CoverageGap(
                    category=category,
                    summary=f"Transcript segment {seg.index} may be missing from notes: \"{snippet}\"",
                    start_time=seg.start,
                    end_time=seg.end,
                    segment_indices=(seg.index,),
                    severity="warning",
                )
            )

    passed = len(gaps) == 0
    notes = (
        f"Deterministic stub coverage check passed ({covered_substantive}/{total_substantive} "
        "substantive segment(s) covered)."
        if passed
        else (
            f"Deterministic stub coverage check flagged {len(gaps)} potentially missing "
            f"substantive segment(s) out of {total_substantive}."
        )
    )
    return CoverageCheckResult(
        passed=passed,
        checked_sections=1,
        covered_segment_count=covered_substantive,
        total_segment_count=total_substantive,
        missing_items=tuple(gaps),
        notes=notes,
    )


def generate_section_notes_with_llm(
    prompt_input: SectionPromptInput,
    settings: Settings,
    *,
    llm_callable: SectionNotesLLMCallable | None = None,
) -> str:
    """Phase 6 LLM hook for generating one section's NotebookLM-optimized Markdown notes."""
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

        cleaned = (rendered or "").strip()
        if not cleaned:
            raise StageError(
                stage="notes",
                message=(
                    f"LLM returned empty Markdown notes for section "
                    f"{prompt_input.section_index}/{prompt_input.total_sections}."
                ),
                hint="Inspect the configured LLM provider response or retry with PIPELINE_LLM_MODE=stub.",
            )
        return cleaned

    mode = (settings.pipeline_llm_mode or "stub").strip().lower()
    if mode == "stub":
        return render_section_notes_stub(prompt_input)

    raise StageError(
        stage="notes",
        message=(
            f"Markdown note generation LLM provider is not wired yet for "
            f"PIPELINE_LLM_MODE='{settings.pipeline_llm_mode}'."
        ),
        hint=(
            "Phase 6 wires the chosen LLM provider into generate_section_notes_with_llm(). "
            "Set PIPELINE_LLM_MODE=stub in .env to run deterministic Phase 4 scaffolding, "
            "or pass an explicit llm_callable."
        ),
    )


def check_coverage_with_llm(
    coverage_input: CoverageCheckInput,
    settings: Settings,
    *,
    llm_callable: CoverageCheckLLMCallable | None = None,
) -> CoverageCheckResult:
    """Phase 6 LLM hook for auditing transcript coverage against generated section notes."""
    prompt_messages = build_coverage_check_prompt(coverage_input)

    if llm_callable is not None:
        try:
            raw_result = llm_callable(prompt_messages, coverage_input, settings)
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                stage="notes",
                message=(
                    f"LLM transcript coverage check failed on section "
                    f"{coverage_input.section_index}/{coverage_input.total_sections}: {exc}"
                ),
                hint="Verify the configured LLM provider and JSON response format, then retry.",
            ) from exc

        if isinstance(raw_result, CoverageCheckResult):
            return raw_result
        try:
            return parse_coverage_check_response(
                raw_result,
                total_segment_count=len(coverage_input.segments),
            )
        except ValueError as exc:
            raise StageError(
                stage="notes",
                message=(
                    f"Invalid coverage check output on section "
                    f"{coverage_input.section_index}/{coverage_input.total_sections}: {exc}"
                ),
                hint="Ensure the coverage-check LLM returns valid JSON matching the coverage schema.",
            ) from exc

    mode = (settings.pipeline_llm_mode or "stub").strip().lower()
    if mode == "stub":
        return check_section_coverage_stub(coverage_input)

    raise StageError(
        stage="notes",
        message=(
            f"Transcript coverage check LLM provider is not wired yet for "
            f"PIPELINE_LLM_MODE='{settings.pipeline_llm_mode}'."
        ),
        hint=(
            "Phase 6 wires the chosen LLM provider into check_coverage_with_llm(). "
            "Set PIPELINE_LLM_MODE=stub in .env to run deterministic Phase 4 scaffolding, "
            "or pass an explicit llm_callable."
        ),
    )


def _merge_coverage_results(results: Sequence[CoverageCheckResult]) -> CoverageCheckResult:
    if not results:
        return CoverageCheckResult(
            passed=True,
            checked_sections=0,
            covered_segment_count=0,
            total_segment_count=0,
            missing_items=(),
            notes="No transcript sections to audit.",
        )

    all_gaps: list[CoverageGap] = []
    covered = 0
    total = 0
    passed = True
    for item in results:
        all_gaps.extend(item.missing_items)
        covered += item.covered_segment_count
        total += item.total_segment_count
        if not item.passed:
            passed = False

    if all_gaps:
        passed = False

    summary = (
        f"Audited {len(results)} section(s): {covered}/{total} substantive segment(s) covered, "
        f"{len(all_gaps)} gap(s) flagged."
    )
    return CoverageCheckResult(
        passed=passed,
        checked_sections=len(results),
        covered_segment_count=covered,
        total_segment_count=total,
        missing_items=tuple(all_gaps),
        notes=summary,
    )


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


def generate_notes(
    slides_ocr: SlidesOCRArtifact,
    transcript: TranscriptArtifact,
    alignment: AlignmentArtifact,
    settings: Settings,
    *,
    lecture_title: str | None = None,
    chunk_config: SectionChunkConfig | None = None,
    section_llm_callable: SectionNotesLLMCallable | None = None,
    coverage_llm_callable: CoverageCheckLLMCallable | None = None,
    progress_callback: ProgressCallback | None = None,
) -> NotesGenerationResult:
    """Orchestrate section chunking, rolling context carry-over, LLM/stub generation, and coverage check."""
    mode = (settings.pipeline_llm_mode or "stub").strip().lower()
    if mode != "stub" and section_llm_callable is None:
        raise StageError(
            stage="notes",
            message=(
                f"Markdown note generation LLM provider is not wired yet for "
                f"PIPELINE_LLM_MODE='{settings.pipeline_llm_mode}'."
            ),
            hint=(
                "Phase 6 wires the chosen LLM provider into generate_section_notes_with_llm(). "
                "Set PIPELINE_LLM_MODE=stub in .env to run deterministic Phase 4 scaffolding, "
                "or pass an explicit section_llm_callable."
            ),
        )

    resolved_title = (lecture_title or "").strip() or derive_lecture_title(slides_ocr)

    if progress_callback is not None:
        progress_callback("Chunking aligned slides and transcript into lecture sections...")

    sections = build_lecture_sections(
        slides_ocr,
        transcript,
        alignment,
        config=chunk_config,
    )

    rolling_context = RollingContext()
    prompt_inputs: list[SectionPromptInput] = []
    section_markdowns: list[str] = []
    coverage_results: list[CoverageCheckResult] = []
    aligned_slide_numbers: set[int] = set()

    total_sections = len(sections)
    for section in sections:
        if progress_callback is not None:
            progress_callback(
                f"Generating notes for section {section.section_index} of {total_sections}..."
            )

        context_snapshot = rolling_context.snapshot()
        prompt_input = SectionPromptInput(
            lecture_title=resolved_title,
            section_index=section.section_index,
            total_sections=section.total_sections,
            section_kind=section.kind,
            start_time=section.start_time,
            end_time=section.end_time,
            slide_numbers=section.slide_numbers,
            slides=section.slides,
            segments=section.segments,
            rolling_context=context_snapshot,
            alignment_warnings=section.alignment_warnings,
            uncertainty_reasons=section.uncertainty_reasons,
        )
        prompt_inputs.append(prompt_input)

        section_md = generate_section_notes_with_llm(
            prompt_input,
            settings,
            llm_callable=section_llm_callable,
        )
        section_markdowns.append(section_md)

        if section.kind == "aligned":
            aligned_slide_numbers.update(section.slide_numbers)

        if section.segments:
            coverage_input = CoverageCheckInput(
                lecture_title=resolved_title,
                section_index=section.section_index,
                total_sections=section.total_sections,
                section_kind=section.kind,
                start_time=section.start_time,
                end_time=section.end_time,
                segments=section.segments,
                generated_markdown=section_md,
                slide_numbers=section.slide_numbers,
            )
            coverage_results.append(
                check_coverage_with_llm(
                    coverage_input,
                    settings,
                    llm_callable=coverage_llm_callable,
                )
            )

        update_rolling_context(rolling_context, section, section_md)

    if progress_callback is not None:
        progress_callback("Assembling final NotebookLM-ready Markdown document...")

    merged_coverage = _merge_coverage_results(coverage_results)
    uncovered_slides_md = _render_uncovered_slides_section(
        slides_ocr,
        alignment,
        aligned_slide_numbers,
    )
    markdown = _assemble_document_markdown(
        lecture_title=resolved_title,
        slides_ocr=slides_ocr,
        transcript=transcript,
        alignment=alignment,
        section_markdowns=section_markdowns,
        uncovered_slides_markdown=uncovered_slides_md,
        mode=(settings.pipeline_llm_mode or "stub").strip().lower(),
    )

    warnings: list[str] = []
    for issue in alignment.alignment_issues:
        warning_text = f"Alignment ({issue.code}): {issue.message}"
        if warning_text not in warnings:
            warnings.append(warning_text)
    for cov_warning in merged_coverage.to_warnings():
        if cov_warning not in warnings:
            warnings.append(cov_warning)

    word_count = len(markdown.split())
    return NotesGenerationResult(
        markdown=markdown,
        sections=tuple(sections),
        prompt_inputs=tuple(prompt_inputs),
        coverage_result=merged_coverage,
        warnings=tuple(warnings),
        word_count=word_count,
    )


def run_notes_stage(context: JobContext, settings: Settings) -> StageResult:
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
        progress_callback=context.report_progress,
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
            "coverage_passed": result.coverage_result.passed,
            "coverage_gaps": len(result.coverage_result.missing_items),
        },
        warnings=list(result.warnings),
    )


__all__ = [
    "CoverageCheckLLMCallable",
    "DEFAULT_MAX_SECTION_DURATION_SECONDS",
    "DEFAULT_MAX_SECTION_TRANSCRIPT_CHARS",
    "DEFAULT_MAX_SLIDES_PER_SECTION",
    "LectureSection",
    "NotesGenerationResult",
    "SectionChunkConfig",
    "SectionNotesLLMCallable",
    "build_lecture_sections",
    "check_coverage_with_llm",
    "check_section_coverage_stub",
    "derive_lecture_title",
    "generate_notes",
    "generate_section_notes_with_llm",
    "load_notes_inputs",
    "render_section_notes_stub",
    "run_notes_stage",
    "update_rolling_context",
]
