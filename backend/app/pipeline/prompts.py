from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from ..schemas import SlidesOCRArtifact, TranscriptArtifact

SectionKind = Literal["aligned", "off_slide", "uncertain"]
CoverageSeverity = Literal["warning", "error"]


@dataclass(frozen=True)
class PromptSpecRule:
    """Single rule from the AI-First Markdown Specification for NotebookLM."""

    number: int
    name: str
    guidance: str


AI_FIRST_MARKDOWN_RULES: tuple[PromptSpecRule, ...] = (
    PromptSpecRule(
        number=1,
        name="Length driven by teaching",
        guidance=(
            "Never impose fixed sentence counts, bullet caps, or per-slide word quotas. "
            "Scale detail strictly to teaching depth: a title or transition slide may take "
            "one concise line, whereas a 12-minute diagram walkthrough or derivation requires "
            "dense, detailed, multi-part coverage."
        ),
    ),
    PromptSpecRule(
        number=2,
        name="Explicit over stylistic",
        guidance=(
            "Optimize for NotebookLM AI retrieval and Q&A rather than literary prose. "
            "Use crisp, structured, information-dense statements, step-by-step sequences, "
            "and explicit causal links. Replace ambiguous pronouns ('it', 'this', 'that', "
            "'they', 'that method', 'the former') with explicit concept names so every "
            "retrieved chunk stays self-contained and unambiguous."
        ),
    ),
    PromptSpecRule(
        number=3,
        name="Preserve reasoning chains in sequence",
        guidance=(
            "Record multi-step derivations, proofs, debugging walkthroughs, and worked "
            "examples in exact logical sequence. If the lecturer walks through three steps "
            "and explains why the third step fails, capture every step, what happened, and "
            "why it failed. Never compress a worked example into a vague summary such as "
            "'the lecturer demonstrated a failure'."
        ),
    ),
    PromptSpecRule(
        number=4,
        name="Keep substantive spoken content",
        guidance=(
            "Preserve all substantive spoken teaching: concrete examples, analogies, "
            "definitions, exceptions, edge cases, self-corrections, caveats, and "
            "substantive student Q&A together with the lecturer's answers. Strip only "
            "spoken filler ('um', 'uh', 'you know'), false starts, operational chatter "
            "(microphone checks, room logistics), and empty repetition—while retaining "
            "any new nuance or clarification added by a second explanation."
        ),
    ),
    PromptSpecRule(
        number=5,
        name="Connect speech to visuals",
        guidance=(
            "Use Gemini OCR VISUAL_CONTEXT (together with VISIBLE_TEXT and "
            "TABLES_CODE_FORMULAS) to ground spoken deictic references such as 'this arrow', "
            "'the second curve', or 'the box on the right' to the specific visual element "
            "when supported by the slide and recording. When the intended visual target is "
            "ambiguous or not confirmed by OCR, explicitly note the ambiguity instead of guessing."
        ),
    ),
    PromptSpecRule(
        number=6,
        name="Accurate exam and study guidance",
        guidance=(
            "Tag exam hints, assessment priorities, or study-focus callouts ONLY when the "
            "lecturer explicitly states them (for example, 'this will be on the midterm' or "
            "'expect a question like this on the final'). Time spent on a topic alone is "
            "never evidence that it will be assessed."
        ),
    ),
    PromptSpecRule(
        number=7,
        name="Strict grounding",
        guidance=(
            "Preserve technical terms, equations, symbols, numbers, units, code snippets, "
            "and qualifications verbatim and accurately. Never invent missing logical steps, "
            "never silently resolve contradictions between a slide and spoken speech (record "
            "both and note the discrepancy explicitly), and never present a paraphrase as a "
            "direct quotation."
        ),
    ),
    PromptSpecRule(
        number=8,
        name="Chunk-friendly Markdown structure",
        guidance=(
            "Structure notes for clean chunking and retrieval: use a clear lecture title "
            "and descriptive topic headings naming the concrete concept explained; include "
            "confirmed slide numbers and [MM:SS–MM:SS] timestamps where available; allow "
            "topics to span multiple slides when the explanation spans multiple slides; "
            "give substantive off-slide discussion, live demos, and student Q&A dedicated "
            "timestamped sections; distinguish slide content from spoken-only additions "
            "where helpful for grounding; and whenever slide alignment is uncertain "
            "(unresolved alignment warnings, out-of-order jumps, or unaligned spans), use "
            "the [MM:SS–MM:SS] timestamp alone instead of forcing a dubious slide number."
        ),
    ),
    PromptSpecRule(
        number=9,
        name="Long-lecture reliability",
        guidance=(
            "Maintain continuity across section boundaries using the supplied rolling "
            "context (prior section headings, introduced key terms, and ongoing multi-slide "
            "examples or derivations) so multi-slide explanations remain coherent. Preserve "
            "all substantive spoken content so downstream transcript coverage checks find "
            "no silently dropped explanations, worked examples, corrections, or student Q&A."
        ),
    ),
)


def _render_spec_rules_block() -> str:
    lines = [
        f"{rule.number}. **{rule.name}:** {rule.guidance}"
        for rule in AI_FIRST_MARKDOWN_RULES
    ]
    return "\n".join(lines)


ALIGNMENT_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slide": {"type": ["integer", "null"]},
                    "start_segment": {"type": "integer"},
                    "end_segment": {"type": "integer"},
                    "uncertain": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["slide", "start_segment", "end_segment", "uncertain", "reason"],
            },
        }
    },
    "required": ["entries"],
}

COVERAGE_CHECK_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "covered_segment_count": {"type": "integer"},
        "total_segment_count": {"type": "integer"},
        "missing_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "start_time": {"type": ["number", "null"]},
                    "end_time": {"type": ["number", "null"]},
                    "segment_indices": {"type": "array", "items": {"type": "integer"}},
                    "summary": {"type": "string"},
                    "severity": {"type": "string", "enum": ["warning", "error"]},
                },
                "required": ["category", "summary", "severity"],
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["passed", "covered_segment_count", "total_segment_count", "missing_items", "notes"],
}

ALIGNMENT_SYSTEM_PROMPT: str = (
    "You are an expert lecture alignment engine. Your job is to align chronological spoken "
    "transcript segments to the exact 1-indexed lecture slides the lecturer is actively "
    "teaching or discussing.\n\n"
    "Rules:\n"
    "1. Cover EVERY transcript segment from index 0 through N-1 in strict chronological "
    "order with contiguous, non-overlapping spans (each span has integer start_segment <= "
    "end_segment, and the next span begins at previous end_segment + 1).\n"
    "2. Assign each span to the exact 1-indexed slide number ('slide': <int>) matching the "
    "lecturer's spoken explanation, using visible text, visual/diagram context, and "
    "tables/code/formulas.\n"
    "3. For substantive off-slide discussion (live coding/demos, whiteboard derivations not "
    "on slides, extended tangents, student Q&A not tied to a slide, or lecture intro/outro "
    "logistics), set \"slide\": null, \"uncertain\": false, and briefly describe the span "
    "in \"reason\" (e.g. \"Live coding demo\" or \"Student Q&A on grading\").\n"
    "4. If the lecturer genuinely jumps backward or skips ahead and the target slide is "
    "clearly supported by the speech, assign that slide. Whenever the true slide is genuinely "
    "ambiguous or unclear, set \"uncertain\": true and explain why in \"reason\" so "
    "downstream note generation uses timestamp-only anchors instead of guessing.\n"
    "5. Never invent slide numbers outside the provided deck. Slides skipped in speech "
    "should simply not appear in any span.\n\n"
    "Return ONLY valid JSON matching this exact schema:\n"
    "{\n"
    '  "entries": [\n'
    "    {\n"
    '      "slide": <int 1-indexed slide number or null>,\n'
    '      "start_segment": <int>,\n'
    '      "end_segment": <int>,\n'
    '      "uncertain": true | false,\n'
    '      "reason": "<empty string or brief reason when off-slide/uncertain>"\n'
    "    }\n"
    "  ]\n"
    "}"
)


SECTION_NOTES_SYSTEM_PROMPT: str = (
    "You are an expert technical note-taker generating English Markdown lecture notes "
    "optimized specifically for **NotebookLM (AI retrieval and Q&A)**.\n"
    "Your primary reader is an AI retrieval system answering detailed factual and "
    "reasoning questions—not a human skimming decorative prose. Prioritize factual "
    "completeness, explicitness, self-contained chunks, and zero ambiguity over "
    "stylistic flow or brevity.\n\n"
    "Follow ALL 9 rules of the AI-First Markdown Specification strictly:\n"
    f"{_render_spec_rules_block()}\n\n"
    "Output requirements:\n"
    "- Output clean English Markdown for the requested lecture section only (do not wrap "
    "the entire response in a markdown code fence).\n"
    "- Start the section with a descriptive `##` heading naming the concrete concept(s) "
    "taught, followed by the appropriate anchor (`Slide N` / `Slides A–B` + `[MM:SS–MM:SS]` "
    "for confirmed aligned sections, or `[MM:SS–MM:SS]` ONLY for off-slide and uncertain sections).\n"
    "- Do NOT invent slide numbers when a section is marked off-slide or uncertain.\n"
    "- Do NOT cap bullet counts, do NOT truncate off-slide discussion, and do NOT omit "
    "spoken-only nuance, worked-example steps, corrections, or student Q&A."
)


COVERAGE_CHECK_SYSTEM_PROMPT: str = (
    "You are a strict transcript coverage auditor for NotebookLM-optimized lecture notes.\n"
    "Your job is to compare one lecture section's chronological transcript against the "
    "generated Markdown notes for that section and detect any **silently dropped "
    "substantive content**.\n\n"
    "What counts as substantive (MUST be captured in the notes):\n"
    "- Concept definitions, explanations, causal links, and reasoning chains (every step "
    "and why a step succeeds or fails).\n"
    "- Worked examples, analogies, edge cases, exceptions, caveats, and numerical/code details.\n"
    "- Lecturer self-corrections or contradictions between spoken speech and slide text.\n"
    "- Visual explanations referencing slide diagrams, curves, arrows, tables, or formulas.\n"
    "- Substantive student questions and the lecturer's answers (Q&A).\n"
    "- Explicit exam/assessment guidance stated by the lecturer.\n\n"
    "What is NOT a coverage gap (ignore these):\n"
    "- Spoken filler ('um', 'uh', 'you know'), false starts, and verbal stumbles.\n"
    "- Operational/logistical chatter (audio checks, breaks, seating, file-sharing logistics).\n"
    "- Pure empty repetition that adds no new detail or nuance.\n\n"
    "Return ONLY valid JSON matching this exact schema:\n"
    "{\n"
    '  "passed": true | false,\n'
    '  "covered_segment_count": <int>,\n'
    '  "total_segment_count": <int>,\n'
    '  "missing_items": [\n'
    "    {\n"
    '      "category": "explanation" | "worked_example" | "student_qa" | "definition" | '
    '"correction" | "visual_reference" | "exam_guidance" | "other",\n'
    '      "start_time": <float seconds or null>,\n'
    '      "end_time": <float seconds or null>,\n'
    '      "segment_indices": [<int>, ...],\n'
    '      "summary": "<concise explicit description of what was omitted>",\n'
    '      "severity": "warning" | "error"\n'
    "    }\n"
    "  ],\n"
    '  "notes": "<brief audit summary>"\n'
    "}"
)


def format_timestamp(seconds: float) -> str:
    """Format a non-negative timestamp in seconds as MM:SS."""
    total_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp_range(start_seconds: float, end_seconds: float) -> str:
    """Format a timestamp span as [MM:SS–MM:SS] using an en-dash."""
    start = max(0.0, float(start_seconds))
    end = max(start, float(end_seconds))
    return f"[{format_timestamp(start)}–{format_timestamp(end)}]"


@dataclass(frozen=True)
class PromptMessages:
    """System + user prompt pair passed to the LLM hook."""

    system: str
    user: str


@dataclass
class RollingContext:
    """Context carried across section boundaries for long-lecture coherence."""

    prior_section_headings: list[str] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)
    ongoing_threads: list[str] = field(default_factory=list)
    last_section_summary: str = ""
    last_slide: int | None = None
    last_timestamp: float | None = None

    def is_empty(self) -> bool:
        return not (
            self.prior_section_headings
            or self.key_terms
            or self.ongoing_threads
            or self.last_section_summary.strip()
        )

    def snapshot(self) -> RollingContext:
        """Return an independent copy representing state at a specific section boundary."""
        return RollingContext(
            prior_section_headings=list(self.prior_section_headings),
            key_terms=list(self.key_terms),
            ongoing_threads=list(self.ongoing_threads),
            last_section_summary=self.last_section_summary,
            last_slide=self.last_slide,
            last_timestamp=self.last_timestamp,
        )

    def to_prompt_block(self) -> str:
        """Render rolling context into a concise block for downstream section prompts."""
        if self.is_empty():
            return "None (this is the first section of the lecture)."

        lines: list[str] = []
        if self.prior_section_headings:
            lines.append("Prior section headings:")
            for heading in self.prior_section_headings:
                lines.append(f"- {heading}")
        if self.key_terms:
            lines.append(f"Key terms / concepts already introduced: {', '.join(self.key_terms)}")
        if self.ongoing_threads:
            lines.append("Ongoing multi-slide examples / reasoning threads:")
            for thread in self.ongoing_threads:
                lines.append(f"- {thread}")
        if self.last_section_summary.strip():
            lines.append(f"Previous section summary: {self.last_section_summary.strip()}")
        boundary_bits: list[str] = []
        if self.last_slide is not None:
            boundary_bits.append(f"Last confirmed slide: Slide {self.last_slide}")
        if self.last_timestamp is not None:
            boundary_bits.append(f"Last timestamp: {format_timestamp(self.last_timestamp)}")
        if boundary_bits:
            lines.append("Boundary state: " + " | ".join(boundary_bits))
        return "\n".join(lines)


@dataclass(frozen=True)
class SectionPromptSlide:
    """Structured OCR payload for one slide inside a section prompt."""

    slide: int
    visible_text: str = ""
    visual_context: str = ""
    tables_code_formulas: str = ""


@dataclass(frozen=True)
class SectionPromptSegment:
    """One chronological transcript segment inside a section prompt."""

    index: int
    start: float
    end: float
    text: str
    slide: int | None = None
    uncertain: bool = False
    uncertainty_reason: str = ""


@dataclass(frozen=True)
class SectionPromptInput:
    """Structured input contract for generating one lecture section's Markdown notes."""

    lecture_title: str
    section_index: int
    total_sections: int
    section_kind: SectionKind
    start_time: float
    end_time: float
    slide_numbers: tuple[int, ...] = ()
    slides: tuple[SectionPromptSlide, ...] = ()
    segments: tuple[SectionPromptSegment, ...] = ()
    rolling_context: RollingContext = field(default_factory=RollingContext)
    alignment_warnings: tuple[str, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageGap:
    """Single omitted substantive item identified by the transcript coverage check."""

    category: str
    summary: str
    start_time: float | None = None
    end_time: float | None = None
    segment_indices: tuple[int, ...] = ()
    severity: CoverageSeverity = "warning"

    def format_warning(self) -> str:
        ts_prefix = ""
        if self.start_time is not None:
            end_val = self.end_time if self.end_time is not None else self.start_time
            ts_prefix = f"{format_timestamp_range(self.start_time, end_val)} "
        return f"Coverage gap ({self.category}): {ts_prefix}{self.summary}".strip()


@dataclass(frozen=True)
class CoverageCheckInput:
    """Structured input contract for checking transcript coverage of generated notes."""

    lecture_title: str
    section_index: int
    total_sections: int
    section_kind: SectionKind
    start_time: float
    end_time: float
    segments: tuple[SectionPromptSegment, ...]
    generated_markdown: str
    slide_numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class CoverageCheckResult:
    """Structured result returned by the transcript coverage checker."""

    passed: bool
    checked_sections: int = 1
    covered_segment_count: int = 0
    total_segment_count: int = 0
    missing_items: tuple[CoverageGap, ...] = ()
    notes: str = ""

    def to_warnings(self) -> list[str]:
        return [gap.format_warning() for gap in self.missing_items]


def _format_slide_range_label(slide_numbers: tuple[int, ...]) -> str:
    if not slide_numbers:
        return "None (timestamp-only anchor)"
    ordered = sorted(set(slide_numbers))
    if len(ordered) == 1:
        return f"Slide {ordered[0]}"
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"Slides {ordered[0]}–{ordered[-1]}"
    return "Slides " + ", ".join(str(num) for num in ordered)


def build_section_notes_prompt(payload: SectionPromptInput) -> PromptMessages:
    """Build the system + user prompts for one section of NotebookLM-optimized Markdown notes."""
    ts_range = format_timestamp_range(payload.start_time, payload.end_time)

    if payload.section_kind == "aligned":
        anchor_instruction = (
            f"Confirmed slide anchor(s): {_format_slide_range_label(payload.slide_numbers)} "
            f"covering {ts_range}. Include confirmed slide numbers and {ts_range} timestamps, "
            "connect spoken references to VISUAL_CONTEXT, and distinguish slide content from "
            "spoken-only additions."
        )
    elif payload.section_kind == "off_slide":
        anchor_instruction = (
            f"OFF-SLIDE / DEMO / Q&A SECTION covering {ts_range}. "
            "Use a dedicated timestamped section anchored ONLY by [MM:SS–MM:SS] timestamps. "
            "Do NOT attach or invent any slide number."
        )
    else:
        reasons_text = "; ".join(payload.uncertainty_reasons) if payload.uncertainty_reasons else (
            "Unresolved alignment warning or out-of-order jump."
        )
        anchor_instruction = (
            f"UNCERTAIN ALIGNMENT SECTION covering {ts_range}. "
            f"Reason: {reasons_text} "
            "CRITICAL: Use the [MM:SS–MM:SS] timestamp ALONE as the anchor. "
            "Do NOT force, guess, or print dubious slide numbers in headings or anchors."
        )

    lines: list[str] = [
        f"Lecture Title: {payload.lecture_title}",
        f"Section: {payload.section_index} of {payload.total_sections}",
        f"Section Kind: {payload.section_kind}",
        f"Time Range: {ts_range}",
        f"Anchoring Rule: {anchor_instruction}",
    ]

    if payload.alignment_warnings:
        lines.append("")
        lines.append("### Alignment Warnings")
        for warning in payload.alignment_warnings:
            lines.append(f"- {warning}")

    lines.append("")
    lines.append("### Rolling Context From Prior Sections")
    lines.append(payload.rolling_context.to_prompt_block())

    if payload.section_kind == "aligned" and payload.slides:
        lines.append("")
        lines.append("### Slide OCR Context (Gemini OCR)")
        for slide in payload.slides:
            lines.append(f"#### Slide {slide.slide}")
            lines.append("VISIBLE_TEXT:")
            lines.append(slide.visible_text.strip() or "(none)")
            lines.append("VISUAL_CONTEXT:")
            lines.append(slide.visual_context.strip() or "(none)")
            lines.append("TABLES_CODE_FORMULAS:")
            lines.append(slide.tables_code_formulas.strip() or "(none)")
            lines.append("")
    elif payload.section_kind == "off_slide":
        lines.append("")
        lines.append("### Slide OCR Context")
        lines.append("No slide is anchored to this off-slide / live demo / Q&A span.")
    elif payload.section_kind == "uncertain":
        lines.append("")
        lines.append("### Slide OCR Context")
        lines.append(
            "Slide alignment is uncertain for this span; slide anchors are intentionally "
            "withheld so notes rely solely on verified speech and [MM:SS–MM:SS] timestamps."
        )

    lines.append("")
    lines.append("### Chronological Transcript Segments")
    if payload.segments:
        for seg in payload.segments:
            seg_range = format_timestamp_range(seg.start, seg.end)
            slide_tag = (
                f" [Slide {seg.slide}]"
                if payload.section_kind == "aligned" and seg.slide is not None and not seg.uncertain
                else ""
            )
            lines.append(f"- {seg_range} (seg {seg.index}){slide_tag}: {seg.text.strip()}")
    else:
        lines.append("(No spoken transcript segments aligned to this section.)")

    lines.append("")
    lines.append(
        "Generate the complete, information-dense, NotebookLM-ready English Markdown for "
        "this section now, following all 9 rules of the AI-First Markdown Specification."
    )

    return PromptMessages(
        system=SECTION_NOTES_SYSTEM_PROMPT,
        user="\n".join(lines),
    )


def build_coverage_check_prompt(payload: CoverageCheckInput) -> PromptMessages:
    """Build the system + user prompts to verify no substantive spoken content was dropped."""
    ts_range = format_timestamp_range(payload.start_time, payload.end_time)
    anchor_desc = (
        f"{_format_slide_range_label(payload.slide_numbers)} {ts_range}"
        if payload.section_kind == "aligned" and payload.slide_numbers
        else ts_range
    )

    lines: list[str] = [
        f"Lecture Title: {payload.lecture_title}",
        f"Section: {payload.section_index} of {payload.total_sections} ({payload.section_kind})",
        f"Anchor: {anchor_desc}",
        "",
        "### Source Transcript Segments",
    ]
    if payload.segments:
        for seg in payload.segments:
            seg_range = format_timestamp_range(seg.start, seg.end)
            lines.append(f"- {seg_range} (seg {seg.index}): {seg.text.strip()}")
    else:
        lines.append("(No transcript segments in this section.)")

    lines.extend(
        [
            "",
            "### Generated Markdown Notes To Audit",
            payload.generated_markdown.strip() or "(empty)",
            "",
            "Audit every substantive explanation, reasoning chain, worked example, "
            "definition, exception, self-correction, visual reference, explicit exam hint, "
            "and student Q&A in the transcript against the generated Markdown and return "
            "ONLY the required JSON object.",
        ]
    )

    return PromptMessages(
        system=COVERAGE_CHECK_SYSTEM_PROMPT,
        user="\n".join(lines),
    )


def _strip_json_code_fence(raw_text: str) -> str:
    cleaned = raw_text.strip()
    fence_match = re.match(
        r"^```(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```\s*$",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def build_alignment_prompt(
    slides: SlidesOCRArtifact,
    transcript: TranscriptArtifact,
) -> PromptMessages:
    """Build the system + user prompts for aligning full transcript segments to lecture slides."""
    ordered_slides = sorted(slides.slides, key=lambda item: item.slide)
    lines: list[str] = [
        f"Source PDF: {slides.source_pdf or 'slides.pdf'} ({len(ordered_slides)} slide(s))",
        f"Transcript Segments: {len(transcript.segments)} (indices 0..{max(0, len(transcript.segments) - 1)})",
        "",
        "### Lecture Slides (Gemini OCR)",
    ]
    if ordered_slides:
        for slide in ordered_slides:
            lines.append(f"#### Slide {slide.slide}")
            visible = slide.visible_text.strip()
            visual = slide.visual_context.strip()
            tables = slide.tables_code_formulas.strip()
            if visible:
                lines.append("VISIBLE_TEXT:")
                lines.append(visible)
            if visual:
                lines.append("VISUAL_CONTEXT:")
                lines.append(visual)
            if tables:
                lines.append("TABLES_CODE_FORMULAS:")
                lines.append(tables)
            if not (visible or visual or tables):
                raw = slide.raw_text.strip()
                lines.append(raw if raw else "(blank slide)")
            lines.append("")
    else:
        lines.append("(No slides available; mark all segments off-slide with slide=null.)")
        lines.append("")

    lines.append("### Chronological Transcript Segments")
    for idx, seg in enumerate(transcript.segments):
        seg_range = format_timestamp_range(seg.start, seg.end)
        lines.append(f"- seg {idx} {seg_range}: {seg.text.strip()}")

    lines.extend(
        [
            "",
            "Align every segment index from 0 to "
            f"{max(0, len(transcript.segments) - 1)} in chronological order and return ONLY "
            "the JSON object with the 'entries' array.",
        ]
    )

    return PromptMessages(
        system=ALIGNMENT_SYSTEM_PROMPT,
        user="\n".join(lines),
    )


def parse_alignment_response(raw_response: str | dict[str, Any] | Sequence[Any]) -> list[dict[str, Any]]:
    """Parse a JSON alignment response into a list of raw alignment entry dicts."""
    if isinstance(raw_response, str):
        cleaned = _strip_json_code_fence(raw_response)
        try:
            data: Any = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Alignment response is not valid JSON: {exc}") from exc
    else:
        data = raw_response

    if isinstance(data, dict):
        if isinstance(data.get("entries"), list):
            entries_raw = data["entries"]
        elif isinstance(data.get("alignment"), list):
            entries_raw = data["alignment"]
        else:
            raise ValueError("Alignment JSON object must contain an 'entries' list.")
    elif isinstance(data, list):
        entries_raw = data
    else:
        raise ValueError(
            f"Alignment response must decode to a JSON object or array, got {type(data).__name__}."
        )

    return list(entries_raw)


def parse_coverage_check_response(
    raw_response: str | dict[str, Any],
    *,
    total_segment_count: int = 0,
) -> CoverageCheckResult:
    """Parse a JSON string or dict returned by a coverage-check LLM into CoverageCheckResult."""
    if isinstance(raw_response, str):
        cleaned = _strip_json_code_fence(raw_response)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Coverage check response is not valid JSON: {exc}") from exc
    elif isinstance(raw_response, dict):
        data = raw_response
    else:
        raise ValueError(
            f"Unsupported coverage check response type: {type(raw_response).__name__}"
        )

    if not isinstance(data, dict):
        raise ValueError("Coverage check response must decode to a JSON object.")

    raw_missing = data.get("missing_items") or []
    if not isinstance(raw_missing, list):
        raise ValueError("Coverage check 'missing_items' field must be a list.")

    gaps: list[CoverageGap] = []
    for item in raw_missing:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or "").strip()
        if not summary:
            continue
        category = str(item.get("category") or "other").strip() or "other"
        raw_severity = str(item.get("severity") or "warning").strip().lower()
        severity: CoverageSeverity = "error" if raw_severity == "error" else "warning"
        start_raw = item.get("start_time")
        end_raw = item.get("end_time")
        start_time = float(start_raw) if isinstance(start_raw, (int, float)) else None
        end_time = float(end_raw) if isinstance(end_raw, (int, float)) else None
        raw_indices = item.get("segment_indices") or []
        seg_indices = tuple(
            int(idx) for idx in raw_indices if isinstance(idx, int) and idx >= 0
        )
        gaps.append(
            CoverageGap(
                category=category,
                summary=summary,
                start_time=start_time,
                end_time=end_time,
                segment_indices=seg_indices,
                severity=severity,
            )
        )

    try:
        raw_total = data.get("total_segment_count", total_segment_count)
        total_count = int(total_segment_count if raw_total is None else raw_total)
        covered_default = max(0, total_count - len(gaps))
        raw_covered = data.get("covered_segment_count", covered_default)
        covered_count = int(covered_default if raw_covered is None else raw_covered)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Coverage check segment counts must be integers: {exc}") from exc
    passed_raw = data.get("passed")
    passed = bool(passed_raw) if isinstance(passed_raw, bool) else (len(gaps) == 0)
    if gaps:
        passed = False
    notes = str(data.get("notes") or "").strip()

    return CoverageCheckResult(
        passed=passed,
        checked_sections=1,
        covered_segment_count=max(0, covered_count),
        total_segment_count=max(0, total_count),
        missing_items=tuple(gaps),
        notes=notes,
    )


__all__ = [
    "AI_FIRST_MARKDOWN_RULES",
    "ALIGNMENT_RESPONSE_JSON_SCHEMA",
    "ALIGNMENT_SYSTEM_PROMPT",
    "COVERAGE_CHECK_RESPONSE_JSON_SCHEMA",
    "COVERAGE_CHECK_SYSTEM_PROMPT",
    "CoverageCheckInput",
    "CoverageCheckResult",
    "CoverageGap",
    "CoverageSeverity",
    "PromptMessages",
    "PromptSpecRule",
    "RollingContext",
    "SECTION_NOTES_SYSTEM_PROMPT",
    "SectionKind",
    "SectionPromptInput",
    "SectionPromptSegment",
    "SectionPromptSlide",
    "build_alignment_prompt",
    "build_coverage_check_prompt",
    "build_section_notes_prompt",
    "format_timestamp",
    "format_timestamp_range",
    "parse_alignment_response",
    "parse_coverage_check_response",
]
