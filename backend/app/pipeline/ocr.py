from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx
import pymupdf
from google import genai
from google.genai import types

from ..config import (
    DEFAULT_GEMINI_COMBINED_RATE_LIMIT_RPM,
    DEFAULT_OCR_GLOBAL_MAX_CONCURRENT,
    DEFAULT_OCR_MAX_PAGES_PER_BATCH,
    DEFAULT_OCR_MODEL_GOOGLE,
    DEFAULT_OCR_PARALLEL_WORKERS,
    DEFAULT_OCR_THINKING_LEVEL,
    Settings,
)
from ..ratelimit import OutboundRateLimiter
from ..schemas import (
    ARTIFACT_SLIDES_OCR,
    JobContext,
    ProgressCallback,
    SlideOCRRecord,
    SlidesOCRArtifact,
    StageError,
    StageResult,
)
from .gemini_client import (
    build_thinking_config,
    extract_retry_after_seconds as _shared_extract_retry_after_seconds,
    get_global_gemini_combined_rate_limiter,
    supports_thinking_level,
)

logger = logging.getLogger(__name__)

GenAIClientFactory = Callable[[Settings], Any]
SleepFn = Callable[[float], None]
JitterFn = Callable[[float, float], float]

DEFAULT_OCR_MODEL = DEFAULT_OCR_MODEL_GOOGLE
DEFAULT_OCR_MAX_BATCH_BYTES = 18 * 1024 * 1024  # Safely below Gemini's 20 MB inline payload limit
DEFAULT_OCR_PARALLEL_MIN_PAGES = 4
DEFAULT_OCR_RATE_LIMIT_RPM = 20
DEFAULT_OCR_RETRY_ATTEMPTS = 3
DEFAULT_OCR_RETRY_BASE_DELAY_SECONDS = 2.0
DEFAULT_OCR_TIMEOUT_SECONDS = 180.0
DEFAULT_OCR_MAX_OUTPUT_TOKENS = 32768
DEFAULT_BOILERPLATE_MIN_FRACTION = 0.4
DEFAULT_BOILERPLATE_MIN_COUNT = 3

SECTION_NAME_TO_LABEL: dict[str, str] = {
    "visible_text": "VISIBLE_TEXT:",
    "visual_context": "VISUAL_CONTEXT:",
    "tables_code_formulas": "TABLES_CODE_FORMULAS:",
}
STRUCTURED_OCR_SECTION_ORDER: tuple[str, ...] = (
    "visible_text",
    "visual_context",
    "tables_code_formulas",
)
STRUCTURED_OCR_SECTION_MARKERS = frozenset(SECTION_NAME_TO_LABEL.values())
SECTION_LABEL_TO_NAME: dict[str, str] = {
    label: name for name, label in SECTION_NAME_TO_LABEL.items()
}

_LEADING_LIST_MARKER_RE = re.compile(r"^(?:[-*•]+|\d+[.)])\s+")
_INLINE_SECTION_MARKER_RE = re.compile(
    r"^(VISIBLE_TEXT|VISUAL_CONTEXT|TABLES_CODE_FORMULAS):\s*(.*)$"
)

_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_AUTH_HTTP_STATUSES = frozenset({401, 403})
_MAX_RETRY_AFTER_SECONDS = 60.0

_semaphore_lock = threading.Lock()
_global_ocr_semaphores: dict[int, threading.BoundedSemaphore] = {}
_rate_limiter_lock = threading.Lock()
_global_ocr_rate_limiters: dict[int, OutboundRateLimiter] = {}


@dataclass(frozen=True)
class StructuredOCRSections:
    visible_text: str = ""
    visual_context: str = ""
    tables_code_formulas: str = ""
    has_markers: bool = False
    original_text: str = ""


@dataclass(frozen=True)
class PdfBatchSpec:
    index: int
    page_start: int  # 1-indexed inclusive
    page_end: int  # 1-indexed inclusive

    @property
    def page_count(self) -> int:
        return self.page_end - self.page_start + 1

    @property
    def page_range_label(self) -> str:
        if self.page_start == self.page_end:
            return f"slide {self.page_start}"
        return f"slides {self.page_start}-{self.page_end}"


@dataclass(frozen=True)
class PdfBatchPayload:
    spec: PdfBatchSpec
    pdf_bytes: bytes

    @property
    def size_bytes(self) -> int:
        return len(self.pdf_bytes)


class _MalformedOCRResponseError(ValueError):
    """Raised when Gemini returns empty or unparseable JSON output."""


def _get_setting_int(settings: Settings, name: str, default: int, *, minimum: int = 1) -> int:
    raw = getattr(settings, name, default)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _get_setting_float(
    settings: Settings,
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
) -> float:
    raw = getattr(settings, name, default)
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _get_global_ocr_semaphore(max_concurrent: int) -> threading.BoundedSemaphore:
    limit = max(1, int(max_concurrent))
    with _semaphore_lock:
        semaphore = _global_ocr_semaphores.get(limit)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _global_ocr_semaphores[limit] = semaphore
        return semaphore


def _get_global_ocr_rate_limiter(max_rpm: int) -> OutboundRateLimiter | None:
    limit = max(0, int(max_rpm))
    if limit <= 0:
        return None
    with _rate_limiter_lock:
        limiter = _global_ocr_rate_limiters.get(limit)
        if limiter is None:
            limiter = OutboundRateLimiter(max_requests=limit, window_seconds=60.0)
            _global_ocr_rate_limiters[limit] = limiter
        return limiter


def validate_gemini_credentials(settings: Settings) -> None:
    """Validate Google Gemini / Vertex AI credentials before starting OCR."""
    if settings.google_genai_use_vertexai:
        if not settings.google_cloud_project.strip():
            raise StageError(
                stage="ocr",
                message=(
                    "GOOGLE_GENAI_USE_VERTEXAI is enabled, but GOOGLE_CLOUD_PROJECT "
                    "is not configured."
                ),
                hint=(
                    "Set GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION) in .env "
                    "with valid Google Cloud Application Default Credentials, or set "
                    "GOOGLE_GENAI_USE_VERTEXAI=false and provide GEMINI_API_KEY."
                ),
                fatal_for_queue=True,
            )
        return

    if not settings.gemini_api_key.strip():
        raise StageError(
            stage="ocr",
            message="GEMINI_API_KEY is not configured for slide OCR.",
            hint=(
                "Add your Google Gemini API key to GEMINI_API_KEY in .env "
                "(or set GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT for Vertex AI), "
                "restart the backend, and retry. Slide OCR does not fall back to plain text extraction."
            ),
            fatal_for_queue=True,
        )


def _default_genai_client_factory(settings: Settings) -> genai.Client:
    timeout_seconds = _get_setting_float(
        settings,
        "ocr_timeout_seconds",
        DEFAULT_OCR_TIMEOUT_SECONDS,
        minimum=5.0,
    )
    http_options = types.HttpOptions(timeout=int(timeout_seconds * 1000))

    if settings.google_genai_use_vertexai:
        location = settings.google_cloud_location.strip() or "us-central1"
        return genai.Client(
            vertexai=True,
            project=settings.google_cloud_project.strip(),
            location=location,
            http_options=http_options,
        )

    return genai.Client(
        api_key=settings.gemini_api_key.strip(),
        http_options=http_options,
    )


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int):
            return resp_status
    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    parsed = _shared_extract_retry_after_seconds(exc)
    if parsed is None:
        return None
    return min(parsed, _MAX_RETRY_AFTER_SECONDS)


def is_auth_error(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code in _AUTH_HTTP_STATUSES:
        return True
    status = str(getattr(exc, "status", "") or "").upper()
    if status in {"UNAUTHENTICATED", "PERMISSION_DENIED"}:
        return True
    message = str(exc).lower()
    return (
        "api key not valid" in message
        or "api_key_invalid" in message
        or "invalid api key" in message
        or "unauthenticated" in message
        or "unauthorized" in message
        or "permission denied" in message
        or "permission_denied" in message
        or "default credentials" in message
        or "could not automatically determine credentials" in message
    )


def is_payload_too_large_error(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code == 413:
        return True
    message = str(exc).lower()
    return (
        "request entity too large" in message
        or "payload size exceeds" in message
        or "request payload size exceeds" in message
        or "request_too_large" in message
    )


def is_transient_ocr_error(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            _MalformedOCRResponseError,
            TimeoutError,
            ConnectionError,
            httpx.TimeoutException,
            httpx.NetworkError,
        ),
    ):
        return True
    status_code = _extract_status_code(exc)
    if status_code in _TRANSIENT_HTTP_STATUSES:
        return True
    status = str(getattr(exc, "status", "") or "").upper()
    if status in {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "ABORTED"}:
        return True
    message = str(exc).lower()
    return (
        "resource_exhausted" in message
        or "rate limit" in message
        or "too many requests" in message
        or "quota exceeded" in message
        or "temporarily unavailable" in message
        or "service unavailable" in message
        or "internal server error" in message
        or "bad gateway" in message
        or "deadline exceeded" in message
        or "timed out" in message
        or "overloaded" in message
    )


def compute_retry_delay_seconds(
    attempt: int,
    base_delay_seconds: float,
    *,
    retry_after: float | None = None,
    jitter_fn: JitterFn = random.uniform,
) -> float:
    if retry_after is not None and retry_after > 0:
        return retry_after
    if base_delay_seconds <= 0:
        return 0.0
    base = base_delay_seconds * (2 ** max(0, attempt - 1))
    return max(0.0, base * jitter_fn(0.85, 1.15))


# --- Structured OCR Section Parsing & Cleanup ---


def _normalize_structured_ocr_line(line: str) -> str:
    normalized = line.strip()
    if not normalized:
        return ""
    normalized = _LEADING_LIST_MARKER_RE.sub("", normalized)
    normalized = " ".join(normalized.split())
    normalized = normalized.strip("`'\"")
    normalized = normalized.rstrip(".,;:")
    return normalized.casefold()


def parse_structured_ocr_text(text: str | None) -> StructuredOCRSections:
    """Parse VISIBLE_TEXT / VISUAL_CONTEXT / TABLES_CODE_FORMULAS markers from OCR output."""
    raw_text = text or ""
    lines = raw_text.splitlines()

    has_marker = False
    for line in lines:
        stripped = line.strip()
        if stripped in SECTION_LABEL_TO_NAME or _INLINE_SECTION_MARKER_RE.match(stripped):
            has_marker = True
            break

    if not has_marker:
        return StructuredOCRSections(original_text=raw_text)

    sections: dict[str, list[str]] = {
        section_name: [] for section_name in STRUCTURED_OCR_SECTION_ORDER
    }
    current_section = "visible_text"

    for line in lines:
        stripped = line.strip()
        if stripped in SECTION_LABEL_TO_NAME:
            current_section = SECTION_LABEL_TO_NAME[stripped]
            continue
        inline_match = _INLINE_SECTION_MARKER_RE.match(stripped)
        if inline_match:
            label = f"{inline_match.group(1)}:"
            current_section = SECTION_LABEL_TO_NAME[label]
            remainder = inline_match.group(2).strip()
            if remainder:
                sections[current_section].append(remainder)
            continue
        sections[current_section].append(line)

    return StructuredOCRSections(
        visible_text="\n".join(sections["visible_text"]).strip(),
        visual_context="\n".join(sections["visual_context"]).strip(),
        tables_code_formulas="\n".join(sections["tables_code_formulas"]).strip(),
        has_markers=True,
        original_text=raw_text,
    )


def rebuild_structured_ocr_text(sections: StructuredOCRSections) -> str:
    if not sections.has_markers:
        return sections.original_text.strip()

    parts: list[str] = []
    for section_name in STRUCTURED_OCR_SECTION_ORDER:
        content = getattr(sections, section_name).strip()
        if not content:
            continue
        parts.append(f"{SECTION_NAME_TO_LABEL[section_name]}\n{content}")
    return "\n\n".join(parts).strip()


def _remove_overlapping_lines(text: str, normalized_visible_lines: set[str]) -> str:
    kept_lines = [
        line
        for line in text.splitlines()
        if _normalize_structured_ocr_line(line) not in normalized_visible_lines
    ]
    return "\n".join(kept_lines).strip()


def cleanup_structured_ocr_text(text: str | None) -> str:
    """Remove duplicate visible-text lines that bled into visual/table sections."""
    sections = parse_structured_ocr_text(text)
    if not sections.has_markers:
        return sections.original_text.strip()

    normalized_visible_lines = {
        normalized
        for normalized in (
            _normalize_structured_ocr_line(line)
            for line in sections.visible_text.splitlines()
        )
        if normalized
    }
    if not normalized_visible_lines:
        return rebuild_structured_ocr_text(sections)

    cleaned = StructuredOCRSections(
        visible_text=sections.visible_text,
        visual_context=_remove_overlapping_lines(
            sections.visual_context,
            normalized_visible_lines,
        ),
        tables_code_formulas=_remove_overlapping_lines(
            sections.tables_code_formulas,
            normalized_visible_lines,
        ),
        has_markers=True,
        original_text=sections.original_text,
    )
    return rebuild_structured_ocr_text(cleaned)


def _strip_boilerplate_from_text(text: str, boilerplate: set[str]) -> tuple[str, int]:
    removed = 0
    kept_lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped in boilerplate and stripped not in STRUCTURED_OCR_SECTION_MARKERS:
            removed += 1
            continue
        kept_lines.append(raw_line)
    return "\n".join(kept_lines).strip(), removed


def strip_repeating_boilerplate(
    slides: Sequence[dict[str, Any]],
    *,
    min_fraction: float = DEFAULT_BOILERPLATE_MIN_FRACTION,
    min_count: int = DEFAULT_BOILERPLATE_MIN_COUNT,
) -> tuple[list[dict[str, Any]], int]:
    """Remove repeating headers/footers/watermarks across slides without losing structured markers."""
    slide_count = len(slides)
    if slide_count == 0:
        return [], 0

    threshold = max(min_count, int(slide_count * min_fraction))
    parsed_slides: list[tuple[dict[str, Any], StructuredOCRSections]] = []
    line_counts: Counter[str] = Counter()

    for slide in slides:
        raw_text = str(slide.get("text", "") or "")
        sections = parse_structured_ocr_text(raw_text)
        parsed_slides.append((slide, sections))

        seen_on_slide: set[str] = set()
        candidate_blocks = (
            (sections.visible_text, sections.visual_context, sections.tables_code_formulas)
            if sections.has_markers
            else (raw_text,)
        )
        for block in candidate_blocks:
            for raw_line in block.splitlines():
                stripped = raw_line.strip()
                if (
                    len(stripped) > 2
                    and stripped not in STRUCTURED_OCR_SECTION_MARKERS
                    and stripped not in seen_on_slide
                ):
                    line_counts[stripped] += 1
                    seen_on_slide.add(stripped)

    boilerplate = {line for line, count in line_counts.items() if count >= threshold}
    boilerplate -= STRUCTURED_OCR_SECTION_MARKERS
    if not boilerplate:
        return [dict(slide) for slide in slides], 0

    cleaned_slides: list[dict[str, Any]] = []
    total_removed = 0
    for slide, sections in parsed_slides:
        raw_text = str(slide.get("text", "") or "")
        if sections.has_markers:
            visible, r1 = _strip_boilerplate_from_text(sections.visible_text, boilerplate)
            visual, r2 = _strip_boilerplate_from_text(sections.visual_context, boilerplate)
            tables, r3 = _strip_boilerplate_from_text(sections.tables_code_formulas, boilerplate)
            total_removed += r1 + r2 + r3
            rebuilt = rebuild_structured_ocr_text(
                StructuredOCRSections(
                    visible_text=visible,
                    visual_context=visual,
                    tables_code_formulas=tables,
                    has_markers=True,
                    original_text=raw_text,
                )
            )
            cleaned_slides.append({"slide": slide["slide"], "text": rebuilt})
        else:
            filtered_text, removed = _strip_boilerplate_from_text(raw_text, boilerplate)
            total_removed += removed
            cleaned_slides.append({"slide": slide["slide"], "text": filtered_text})

    return cleaned_slides, total_removed


def build_slide_ocr_records(
    slides: Sequence[dict[str, Any]],
) -> tuple[list[SlideOCRRecord], dict[str, int], list[int]]:
    """Convert cleaned per-slide text dicts into validated SlideOCRRecord models."""
    records: list[SlideOCRRecord] = []
    structured_slides = 0
    visual_context_slides = 0
    tables_code_formulas_slides = 0
    empty_slides = 0
    unmarked_slides: list[int] = []

    for fallback_index, slide in enumerate(slides, start=1):
        try:
            slide_number = int(slide.get("slide", fallback_index))
        except (TypeError, ValueError):
            slide_number = fallback_index
        if slide_number < 1:
            slide_number = fallback_index

        raw_text = str(slide.get("text", "") or "").strip()
        sections = parse_structured_ocr_text(raw_text)

        if not raw_text:
            empty_slides += 1
            records.append(SlideOCRRecord(slide=slide_number))
            continue

        if sections.has_markers:
            structured_slides += 1
            if sections.visual_context:
                visual_context_slides += 1
            if sections.tables_code_formulas:
                tables_code_formulas_slides += 1
            canonical_raw = rebuild_structured_ocr_text(sections)
            records.append(
                SlideOCRRecord(
                    slide=slide_number,
                    visible_text=sections.visible_text,
                    visual_context=sections.visual_context,
                    tables_code_formulas=sections.tables_code_formulas,
                    raw_text=canonical_raw,
                )
            )
        else:
            unmarked_slides.append(slide_number)
            records.append(
                SlideOCRRecord(
                    slide=slide_number,
                    visible_text=raw_text,
                    visual_context="",
                    tables_code_formulas="",
                    raw_text=raw_text,
                )
            )

    metrics = {
        "slide_count": len(records),
        "structured_slides": structured_slides,
        "visual_context_slides": visual_context_slides,
        "tables_code_formulas_slides": tables_code_formulas_slides,
        "empty_slides": empty_slides,
    }
    return records, metrics, unmarked_slides


# --- PDF Inspection & Slicing with PyMuPDF (no text extraction) ---


def inspect_pdf_page_count(pdf_path: Path) -> int:
    """Inspect PDF validity and page count using PyMuPDF without extracting text."""
    path = Path(pdf_path)
    if not path.is_file():
        raise StageError(
            stage="ocr",
            message="Uploaded slide PDF is missing from job storage.",
            hint="Re-upload the slide PDF and recording to start a fresh job.",
        )
    if path.stat().st_size == 0:
        raise StageError(
            stage="ocr",
            message="Uploaded slide PDF is empty (0 bytes).",
            hint="Export a valid, non-empty PDF slide deck and upload it again.",
        )

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise StageError(
            stage="ocr",
            message=f"Slide PDF could not be opened or is corrupted ({exc}).",
            hint="Verify the file is a valid PDF that opens locally, re-export it if needed, and retry.",
        ) from exc

    try:
        if getattr(doc, "needs_pass", False):
            raise StageError(
                stage="ocr",
                message="Slide PDF is password-protected and cannot be read for OCR.",
                hint="Remove the password protection or print/export an unlocked copy of the PDF, then retry.",
            )
        page_count = len(doc)
    finally:
        doc.close()

    if page_count <= 0:
        raise StageError(
            stage="ocr",
            message="Slide PDF contains 0 pages.",
            hint="Upload a slide PDF containing at least one page.",
        )
    return page_count


def plan_pdf_batches(
    page_count: int,
    *,
    max_pages_per_batch: int = DEFAULT_OCR_MAX_PAGES_PER_BATCH,
    parallel_workers: int = DEFAULT_OCR_PARALLEL_WORKERS,
    parallel_min_pages: int = DEFAULT_OCR_PARALLEL_MIN_PAGES,
) -> list[PdfBatchSpec]:
    """Partition a PDF into balanced 1-indexed inclusive page ranges for Gemini OCR."""
    if page_count <= 0:
        raise ValueError("page_count must be positive")

    resolved_max_pages = max(1, int(max_pages_per_batch))
    resolved_workers = max(1, int(parallel_workers))
    resolved_min_pages = max(2, int(parallel_min_pages))

    min_batches = math.ceil(page_count / resolved_max_pages)
    if min_batches == 1 and resolved_workers > 1 and page_count >= resolved_min_pages:
        min_pages_per_split = max(2, resolved_min_pages // 2)
        batch_count = min(resolved_workers, max(1, page_count // min_pages_per_split))
    else:
        batch_count = min(min_batches, page_count)

    base_pages, remainder = divmod(page_count, batch_count)
    batches: list[PdfBatchSpec] = []
    page_start = 1
    for idx in range(batch_count):
        size = base_pages + (1 if idx < remainder else 0)
        if size <= 0:
            continue
        page_end = page_start + size - 1
        batches.append(PdfBatchSpec(index=idx, page_start=page_start, page_end=page_end))
        page_start = page_end + 1
    return batches


def _render_pdf_slice_bytes(source_doc: pymupdf.Document, page_start: int, page_end: int) -> bytes:
    """Slice 1-indexed inclusive pages from an open PyMuPDF document into PDF bytes."""
    batch_doc = pymupdf.open()
    try:
        batch_doc.insert_pdf(source_doc, from_page=page_start - 1, to_page=page_end - 1)
        return bytes(batch_doc.tobytes(garbage=3, deflate=True))
    finally:
        batch_doc.close()


def slice_pdf_batches(
    pdf_path: Path,
    *,
    page_count: int,
    max_pages_per_batch: int = DEFAULT_OCR_MAX_PAGES_PER_BATCH,
    max_batch_bytes: int = DEFAULT_OCR_MAX_BATCH_BYTES,
    parallel_workers: int = DEFAULT_OCR_PARALLEL_WORKERS,
    parallel_min_pages: int = DEFAULT_OCR_PARALLEL_MIN_PAGES,
) -> list[PdfBatchPayload]:
    """Create size-bounded PDF byte payloads for each planned page range using PyMuPDF."""
    planned = plan_pdf_batches(
        page_count,
        max_pages_per_batch=max_pages_per_batch,
        parallel_workers=parallel_workers,
        parallel_min_pages=parallel_min_pages,
    )
    safe_max_bytes = max(1024, int(max_batch_bytes))

    # Fast path: single batch that already fits under the inline byte budget.
    if len(planned) == 1:
        raw_bytes = Path(pdf_path).read_bytes()
        if len(raw_bytes) <= safe_max_bytes:
            return [PdfBatchPayload(spec=planned[0], pdf_bytes=raw_bytes)]

    try:
        source_doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise StageError(
            stage="ocr",
            message=f"Slide PDF could not be opened while preparing OCR batches ({exc}).",
            hint="Verify the uploaded PDF is intact and re-upload it.",
        ) from exc

    payloads: list[PdfBatchPayload] = []
    try:
        pending_ranges: list[tuple[int, int]] = [
            (spec.page_start, spec.page_end) for spec in planned
        ]
        while pending_ranges:
            start, end = pending_ranges.pop(0)
            slice_bytes = _render_pdf_slice_bytes(source_doc, start, end)
            if len(slice_bytes) > safe_max_bytes and end > start:
                mid = (start + end) // 2
                pending_ranges.insert(0, (mid + 1, end))
                pending_ranges.insert(0, (start, mid))
                continue
            if len(slice_bytes) > safe_max_bytes:
                raise StageError(
                    stage="ocr",
                    message=(
                        f"Slide {start} alone is {len(slice_bytes):,} bytes, exceeding Gemini's "
                        f"inline OCR batch limit ({safe_max_bytes:,} bytes)."
                    ),
                    hint=(
                        "Compress high-resolution embedded images in the slide PDF "
                        "(or re-export at 150-200 DPI) and retry."
                    ),
                )
            payloads.append(
                PdfBatchPayload(
                    spec=PdfBatchSpec(index=len(payloads), page_start=start, page_end=end),
                    pdf_bytes=slice_bytes,
                )
            )
    finally:
        source_doc.close()

    return payloads


# --- Gemini Prompt & Response Parsing ---


def build_ocr_prompt(expected_pages: int) -> str:
    """Build the structured Gemini slide OCR prompt."""
    return f"""Extract lecture slide content from every page of this PDF with high fidelity.

This PDF contains exactly {expected_pages} page(s). Return a JSON array with exactly {expected_pages} object(s), numbered 1 through {expected_pages} in page order.

For each slide, produce a compact but information-rich structured text representation that preserves visible text, technical details, and visual information for downstream AI-first lecture note generation.

For each slide, include:
1. All meaningful visible text (titles, headers, body, bullet points, labels, legends, callouts)
2. Chart/diagram/image content that plain text extraction would miss:
   - Describe diagrams, arrows, flows, axes, legends, annotations, screenshots, UI/code screenshots, and meaningful images
   - Capture important relationships such as cause/effect, sequence, hierarchy, data flow, before/after, comparisons, and mappings
   - For charts, include chart type, axes, series/legend labels, visible trends, and notable values if readable
   - For screenshots or UI images, include visible UI labels, states, selected items, and the concept demonstrated
3. Code snippets, commands, pseudocode, formulas, and tables (use markdown-like rows for tables; preserve formatting and symbols)

Return ONLY a JSON array in this exact format:
[
  {{"slide": 1, "text": "VISIBLE_TEXT:\\n...\\n\\nVISUAL_CONTEXT:\\n- Diagram/chart/image description...\\n\\nTABLES_CODE_FORMULAS:\\n..."}},
  {{"slide": 2, "text": "VISIBLE_TEXT:\\n..."}}
]

Guidelines:
- Put ordinary slide text under VISIBLE_TEXT:, visual/diagram descriptions under VISUAL_CONTEXT:, and tables/code/formulas under TABLES_CODE_FORMULAS:
- Use the exact section headers shown above (each on its own line) whenever that section has content
- Omit a section header if that section has no meaningful content for that slide
- Preserve original language and technical notation for visible text, code, and formulas; write visual descriptions clearly in English
- Preserve meaningful line breaks
- If a slide has no visible text but has meaningful visuals, describe them under VISUAL_CONTEXT:
- If a slide is completely blank or has no meaningful content, use empty string ""
- Ignore purely decorative backgrounds, repeated logos, and repeated footer/header boilerplate unless educationally relevant
- Do not invent content that is not visible on the slide
- Do not wrap the response in markdown code blocks
- Return valid JSON only"""


def _unwrap_full_json_fence(response_text: str) -> str:
    stripped = response_text.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return response_text

    first_line_end = stripped.find("\n")
    if first_line_end == -1:
        return response_text

    opening = stripped[:first_line_end].strip()
    language = opening[3:].strip().lower()
    if language not in {"", "json"}:
        return response_text

    return stripped[first_line_end + 1 : -3].strip()


def _extract_json_array(response_text: str) -> str | None:
    start_index: int | None = None
    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(response_text):
        if in_string:
            if escape_next:
                escape_next = False
            elif char == "\\":
                escape_next = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "[":
            if depth == 0:
                start_index = index
            depth += 1
            continue

        if char == "]" and depth:
            depth -= 1
            if depth == 0 and start_index is not None:
                return response_text[start_index : index + 1]

    return None


def _coerce_slide_entry_text(entry: dict[str, Any]) -> str:
    if "text" in entry and entry["text"] is not None:
        return str(entry["text"])

    structured_keys = ("visible_text", "visual_context", "tables_code_formulas")
    if any(key in entry for key in structured_keys):
        sections = StructuredOCRSections(
            visible_text=str(entry.get("visible_text") or "").strip(),
            visual_context=str(entry.get("visual_context") or "").strip(),
            tables_code_formulas=str(entry.get("tables_code_formulas") or "").strip(),
            has_markers=True,
        )
        return rebuild_structured_ocr_text(sections)

    raise _MalformedOCRResponseError("Slide entry is missing a 'text' field.")


def parse_gemini_ocr_response(response_text: str) -> list[dict[str, Any]]:
    """Parse and validate a Gemini JSON array response into slide dicts."""
    if not response_text or not response_text.strip():
        raise _MalformedOCRResponseError("Empty response from Gemini OCR.")

    cleaned = _unwrap_full_json_fence(response_text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        array_text = _extract_json_array(cleaned)
        if array_text is None:
            raise _MalformedOCRResponseError(
                f"Failed to parse Gemini OCR JSON response: {exc}"
            ) from exc
        try:
            parsed = json.loads(array_text)
        except json.JSONDecodeError as array_exc:
            raise _MalformedOCRResponseError(
                f"Failed to parse Gemini OCR JSON array: {array_exc}"
            ) from array_exc

    if isinstance(parsed, dict) and isinstance(parsed.get("slides"), list):
        parsed = parsed["slides"]

    if not isinstance(parsed, list):
        raise _MalformedOCRResponseError(
            f"Expected JSON array from Gemini OCR, got {type(parsed).__name__}."
        )

    slides: list[dict[str, Any]] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise _MalformedOCRResponseError(f"Slide item {index} is not a JSON object.")
        if "slide" not in item:
            raise _MalformedOCRResponseError(f"Slide item {index} is missing 'slide' number.")
        slides.append(
            {
                "slide": item["slide"],
                "text": _coerce_slide_entry_text(item),
            }
        )

    return slides


def _try_parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def normalize_batch_slides(
    raw_slides: Sequence[dict[str, Any]],
    *,
    batch: PdfBatchSpec,
) -> tuple[list[dict[str, Any]], str | None]:
    """Align returned batch slide numbers to global 1-indexed deck page numbers."""
    expected_count = batch.page_count
    warning: str | None = None
    if len(raw_slides) != expected_count:
        warning = (
            f"Gemini OCR returned {len(raw_slides)} slide(s) for {batch.page_range_label} "
            f"(expected {expected_count}); normalized to match PDF page count."
        )

    parsed_numbers = [_try_parse_int(item.get("slide")) for item in raw_slides]

    # Case 1: Unique 1..expected_count relative numbers
    if (
        raw_slides
        and all(num is not None and 1 <= num <= expected_count for num in parsed_numbers)
        and len(set(parsed_numbers)) == len(parsed_numbers)
    ):
        by_relative = {
            int(num): str(item.get("text", "") or "")
            for num, item in zip(parsed_numbers, raw_slides)
            if num is not None
        }
        return [
            {"slide": batch.page_start + offset, "text": by_relative.get(offset + 1, "")}
            for offset in range(expected_count)
        ], warning

    # Case 2: Unique global page_start..page_end numbers
    if (
        raw_slides
        and all(
            num is not None and batch.page_start <= num <= batch.page_end
            for num in parsed_numbers
        )
        and len(set(parsed_numbers)) == len(parsed_numbers)
    ):
        by_global = {
            int(num): str(item.get("text", "") or "")
            for num, item in zip(parsed_numbers, raw_slides)
            if num is not None
        }
        return [
            {
                "slide": batch.page_start + offset,
                "text": by_global.get(batch.page_start + offset, ""),
            }
            for offset in range(expected_count)
        ], warning

    # Case 3: Fallback positional ordering sorted by numeric slide key when present
    ordered = sorted(
        enumerate(raw_slides),
        key=lambda pair: (
            parsed_numbers[pair[0]] if parsed_numbers[pair[0]] is not None else pair[0] + 1,
            pair[0],
        ),
    )
    normalized: list[dict[str, Any]] = []
    for offset in range(expected_count):
        text = str(ordered[offset][1].get("text", "") or "") if offset < len(ordered) else ""
        normalized.append({"slide": batch.page_start + offset, "text": text})
    return normalized, warning


def _extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    try:
        text = getattr(response, "text", None)
    except ValueError as exc:
        raise _MalformedOCRResponseError(
            f"Gemini OCR response did not contain readable text ({exc})."
        ) from exc
    return str(text or "")


def _extract_usage_tokens(response: Any) -> tuple[int, int]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return 0, 0
    input_tokens = int(getattr(metadata, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(metadata, "candidates_token_count", 0) or 0)
    return max(0, input_tokens), max(0, output_tokens)


# --- Gemini Execution & Retries ---


def _ocr_single_batch_once(
    client: Any,
    batch_payload: PdfBatchPayload,
    *,
    model: str,
    max_output_tokens: int,
    thinking_level: str = DEFAULT_OCR_THINKING_LEVEL,
    semaphore: threading.BoundedSemaphore,
    rate_limiter: OutboundRateLimiter | None = None,
    combined_rate_limiter: OutboundRateLimiter | None = None,
    batch_label: str = "",
    progress_callback: ProgressCallback | None = None,
    sleep_fn: SleepFn = time.sleep,
) -> tuple[list[dict[str, Any]], int, int]:
    prompt = build_ocr_prompt(batch_payload.spec.page_count)
    pdf_part = types.Part.from_bytes(
        data=batch_payload.pdf_bytes,
        mime_type="application/pdf",
    )
    thinking_config = build_thinking_config(thinking_level, model=model)
    config_kwargs: dict[str, Any] = {
        "max_output_tokens": max_output_tokens,
        "response_mime_type": "application/json",
    }
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    # Gemini 3+/4 reasoning models recommend keeping the default temperature (1.0).
    if not supports_thinking_level(model):
        config_kwargs["temperature"] = 0.0
    config = types.GenerateContentConfig(**config_kwargs)

    if rate_limiter is not None:
        def _notify_wait(wait_seconds: float) -> None:
            wait_msg = (
                f"Gemini OCR rate limit pacing ({batch_label}); waiting {wait_seconds:.1f}s..."
                if batch_label
                else f"Gemini OCR rate limit pacing; waiting {wait_seconds:.1f}s..."
            )
            logger.info("%s", wait_msg)
            if progress_callback is not None:
                progress_callback(wait_msg)

        rate_limiter.acquire(sleep_fn=sleep_fn, on_wait=_notify_wait)

    if combined_rate_limiter is not None:
        def _notify_combined_wait(wait_seconds: float) -> None:
            wait_msg = (
                f"Shared Gemini quota pacing ({batch_label}); waiting {wait_seconds:.1f}s..."
                if batch_label
                else f"Shared Gemini quota pacing; waiting {wait_seconds:.1f}s..."
            )
            logger.info("%s", wait_msg)
            if progress_callback is not None:
                progress_callback(wait_msg)

        combined_rate_limiter.acquire(sleep_fn=sleep_fn, on_wait=_notify_combined_wait)

    with semaphore:
        response = client.models.generate_content(
            model=model,
            contents=[prompt, pdf_part],
            config=config,
        )

    raw_text = _extract_response_text(response)
    slides = parse_gemini_ocr_response(raw_text)
    input_tokens, output_tokens = _extract_usage_tokens(response)
    return slides, input_tokens, output_tokens


def _auth_error_hint(settings: Settings) -> str:
    if settings.google_genai_use_vertexai:
        return (
            "Verify GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and your Google Cloud "
            "Application Default Credentials (Vertex AI user permissions), then retry."
        )
    return (
        "Verify that GEMINI_API_KEY in .env is valid and has access to the configured "
        "OCR_MODEL_GOOGLE model, restart the backend, and retry."
    )


def _ocr_single_batch_with_retries(
    client: Any,
    batch_payload: PdfBatchPayload,
    *,
    total_batches: int,
    settings: Settings,
    model: str,
    semaphore: threading.BoundedSemaphore,
    rate_limiter: OutboundRateLimiter | None = None,
    combined_rate_limiter: OutboundRateLimiter | None = None,
    progress_callback: ProgressCallback | None = None,
    warnings: list[str] | None = None,
    warnings_lock: threading.Lock | None = None,
    sleep_fn: SleepFn = time.sleep,
    cancel_event: threading.Event | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    attempts = _get_setting_int(settings, "ocr_retry_attempts", DEFAULT_OCR_RETRY_ATTEMPTS)
    base_delay = _get_setting_float(
        settings,
        "ocr_retry_base_delay_seconds",
        DEFAULT_OCR_RETRY_BASE_DELAY_SECONDS,
    )
    max_output_tokens = _get_setting_int(
        settings,
        "ocr_max_output_tokens",
        DEFAULT_OCR_MAX_OUTPUT_TOKENS,
        minimum=1024,
    )
    thinking_level = str(getattr(settings, "ocr_thinking_level", DEFAULT_OCR_THINKING_LEVEL) or "")

    batch_label = (
        f"Batch {batch_payload.spec.index + 1}/{total_batches} "
        f"({batch_payload.spec.page_range_label})"
        if total_batches > 1
        else batch_payload.spec.page_range_label
    )

    def _record_warning(note: str) -> None:
        if warnings is None:
            return
        if warnings_lock is not None:
            with warnings_lock:
                warnings.append(note)
        else:
            warnings.append(note)

    def _check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise StageError(
                stage="ocr",
                message="Slide OCR cancelled by user.",
                hint="Retry the lecture if you still want Markdown notes.",
            )

    def _effective_sleep(seconds: float) -> None:
        _check_cancelled()
        if seconds <= 0:
            return
        if cancel_event is not None and sleep_fn is time.sleep:
            cancel_event.wait(timeout=seconds)
        else:
            sleep_fn(seconds)
        _check_cancelled()

    for attempt in range(1, attempts + 1):
        _check_cancelled()
        try:
            raw_slides, input_tokens, output_tokens = _ocr_single_batch_once(
                client,
                batch_payload,
                model=model,
                max_output_tokens=max_output_tokens,
                thinking_level=thinking_level,
                semaphore=semaphore,
                rate_limiter=rate_limiter,
                combined_rate_limiter=combined_rate_limiter,
                batch_label=batch_label,
                progress_callback=progress_callback,
                sleep_fn=_effective_sleep,
            )
            normalized_slides, mismatch_warning = normalize_batch_slides(
                raw_slides,
                batch=batch_payload.spec,
            )
            if mismatch_warning:
                _record_warning(mismatch_warning)
            if attempt > 1:
                retries_used = attempt - 1
                retry_word = "once" if retries_used == 1 else f"{retries_used} times"
                _record_warning(
                    f"Gemini OCR {batch_label} retried {retry_word} after a transient error."
                )
            return normalized_slides, input_tokens, output_tokens
        except StageError:
            raise
        except Exception as exc:
            if is_auth_error(exc):
                raise StageError(
                    stage="ocr",
                    message=(
                        f"Google Gemini OCR authentication failed ({batch_label}: {exc})."
                    ),
                    hint=_auth_error_hint(settings),
                    fatal_for_queue=True,
                ) from exc
            if is_payload_too_large_error(exc):
                raise StageError(
                    stage="ocr",
                    message=(
                        f"Gemini OCR rejected {batch_label} because the PDF payload is too large "
                        f"({batch_payload.size_bytes:,} bytes)."
                    ),
                    hint=(
                        "Compress embedded images in the PDF or reduce OCR batch size, then retry. "
                        "Slide OCR does not fall back to plain text extraction."
                    ),
                ) from exc
            if not is_transient_ocr_error(exc):
                raise StageError(
                    stage="ocr",
                    message=f"Google Gemini OCR failed ({batch_label}: {exc}).",
                    hint=(
                        f"Check OCR_MODEL_GOOGLE ('{model}') and your Google Gemini / Vertex AI "
                        "account configuration, then retry. Slide OCR never falls back to pdfplumber."
                    ),
                ) from exc
            if attempt >= attempts:
                raise StageError(
                    stage="ocr",
                    message=(
                        f"Google Gemini OCR failed after {attempts} attempt(s) "
                        f"({batch_label}: {exc})."
                    ),
                    hint=(
                        "Gemini appears temporarily unavailable, rate-limited, or returned "
                        "malformed JSON repeatedly. Wait a moment and retry; slidenoteLM will "
                        "not fall back to plain PDF text extraction."
                    ),
                ) from exc

            delay = compute_retry_delay_seconds(
                attempt,
                base_delay,
                retry_after=_extract_retry_after_seconds(exc),
            )
            retry_msg = (
                f"Gemini OCR transient error ({batch_label}); "
                f"retrying ({attempt + 1}/{attempts}) in {delay:.1f}s..."
            )
            logger.warning("%s (%s)", retry_msg, exc)
            if progress_callback is not None:
                progress_callback(retry_msg)
            if delay > 0:
                _effective_sleep(delay)

    raise AssertionError("Unreachable retry loop exit.")  # pragma: no cover


def extract_slides_with_gemini_ocr(
    pdf_path: Path,
    settings: Settings,
    *,
    source_pdf: str,
    progress_callback: ProgressCallback | None = None,
    client_factory: GenAIClientFactory | None = None,
    rate_limiter: OutboundRateLimiter | None = None,
    sleep_fn: SleepFn = time.sleep,
    cancel_event: threading.Event | None = None,
) -> tuple[SlidesOCRArtifact, list[str], dict[str, int]]:
    """Run strict Google Gemini OCR across a slide PDF and return the structured artifact."""
    validate_gemini_credentials(settings)

    if progress_callback is not None:
        progress_callback("Inspecting slide PDF...")
    page_count = inspect_pdf_page_count(pdf_path)

    model = settings.ocr_model_google.strip() or DEFAULT_OCR_MODEL
    max_pages_per_batch = _get_setting_int(
        settings,
        "ocr_max_pages_per_batch",
        DEFAULT_OCR_MAX_PAGES_PER_BATCH,
    )
    max_batch_bytes = _get_setting_int(
        settings,
        "ocr_max_batch_bytes",
        DEFAULT_OCR_MAX_BATCH_BYTES,
        minimum=1024,
    )
    parallel_workers = _get_setting_int(
        settings,
        "ocr_parallel_workers",
        DEFAULT_OCR_PARALLEL_WORKERS,
    )
    parallel_min_pages = _get_setting_int(
        settings,
        "ocr_parallel_min_pages",
        DEFAULT_OCR_PARALLEL_MIN_PAGES,
        minimum=2,
    )
    global_max_concurrent = _get_setting_int(
        settings,
        "ocr_global_max_concurrent",
        DEFAULT_OCR_GLOBAL_MAX_CONCURRENT,
    )

    batch_payloads = slice_pdf_batches(
        pdf_path,
        page_count=page_count,
        max_pages_per_batch=max_pages_per_batch,
        max_batch_bytes=max_batch_bytes,
        parallel_workers=parallel_workers,
        parallel_min_pages=parallel_min_pages,
    )
    total_batches = len(batch_payloads)

    factory = client_factory or _default_genai_client_factory
    try:
        client = factory(settings)
    except StageError:
        raise
    except Exception as exc:
        raise StageError(
            stage="ocr",
            message=f"Failed to initialize Google Gemini client ({exc}).",
            hint=_auth_error_hint(settings),
        ) from exc

    semaphore = _get_global_ocr_semaphore(global_max_concurrent)
    ocr_rate_limit_rpm = _get_setting_int(
        settings,
        "ocr_rate_limit_rpm",
        DEFAULT_OCR_RATE_LIMIT_RPM,
        minimum=0,
    )
    combined_rpm = _get_setting_int(
        settings,
        "gemini_combined_rate_limit_rpm",
        DEFAULT_GEMINI_COMBINED_RATE_LIMIT_RPM,
        minimum=0,
    )
    if rate_limiter is not None:
        resolved_rate_limiter: OutboundRateLimiter | None = rate_limiter
        combined_rate_limiter: OutboundRateLimiter | None = None
    elif client_factory is not None:
        resolved_rate_limiter = (
            OutboundRateLimiter(max_requests=ocr_rate_limit_rpm, window_seconds=60.0)
            if ocr_rate_limit_rpm > 0
            else None
        )
        combined_rate_limiter = None
    else:
        resolved_rate_limiter = _get_global_ocr_rate_limiter(ocr_rate_limit_rpm)
        combined_rate_limiter = get_global_gemini_combined_rate_limiter(combined_rpm)
    warnings: list[str] = []
    warnings_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed_batches = 0

    if progress_callback is not None:
        if total_batches > 1:
            progress_callback(
                f"Running Gemini OCR ({model}) on {page_count} slides across {total_batches} batches..."
            )
        else:
            progress_callback(
                f"Running Gemini OCR ({model}) on {page_count} slide(s)..."
            )

    batch_results: list[tuple[int, list[dict[str, Any]], int, int]] = []

    def _run_batch(payload: PdfBatchPayload) -> tuple[int, list[dict[str, Any]], int, int]:
        slides, in_tok, out_tok = _ocr_single_batch_with_retries(
            client,
            payload,
            total_batches=total_batches,
            settings=settings,
            model=model,
            semaphore=semaphore,
            rate_limiter=resolved_rate_limiter,
            combined_rate_limiter=combined_rate_limiter,
            progress_callback=progress_callback,
            warnings=warnings,
            warnings_lock=warnings_lock,
            sleep_fn=sleep_fn,
            cancel_event=cancel_event,
        )
        return payload.spec.page_start, slides, in_tok, out_tok

    worker_count = min(parallel_workers, total_batches)
    if worker_count > 1:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="slidenotelm-ocr",
        ) as pool:
            future_to_payload = {
                pool.submit(_run_batch, payload): payload for payload in batch_payloads
            }
            try:
                for future in as_completed(future_to_payload):
                    payload = future_to_payload[future]
                    batch_results.append(future.result())
                    with progress_lock:
                        completed_batches += 1
                        done = completed_batches
                    if progress_callback is not None:
                        progress_callback(
                            f"Completed OCR batch {done} of {total_batches} "
                            f"({payload.spec.page_range_label})..."
                        )
            except Exception:
                for pending in future_to_payload:
                    pending.cancel()
                raise
    else:
        for payload in batch_payloads:
            batch_results.append(_run_batch(payload))
            completed_batches += 1
            if progress_callback is not None and total_batches > 1:
                progress_callback(
                    f"Completed OCR batch {completed_batches} of {total_batches} "
                    f"({payload.spec.page_range_label})..."
                )

    merged_slides: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    for _, slides, in_tok, out_tok in sorted(batch_results, key=lambda item: item[0]):
        merged_slides.extend(slides)
        total_input_tokens += in_tok
        total_output_tokens += out_tok

    if progress_callback is not None:
        progress_callback("Cleaning structured OCR output and stripping repeating boilerplate...")

    deduped_slides = [
        {"slide": slide["slide"], "text": cleanup_structured_ocr_text(slide.get("text", ""))}
        for slide in merged_slides
    ]
    cleaned_slides, boilerplate_removed = strip_repeating_boilerplate(deduped_slides)
    records, metrics, unmarked_slides = build_slide_ocr_records(cleaned_slides)

    if unmarked_slides:
        warnings.append(
            "Some slides lacked structured OCR section markers and were stored as visible text: "
            + ", ".join(str(num) for num in unmarked_slides)
        )

    artifact = SlidesOCRArtifact(
        source_pdf=source_pdf,
        model=model,
        slide_count=len(records),
        slides=records,
    )
    stats = {
        "slide_count": len(records),
        "ocr_batches": total_batches,
        "structured_slides": metrics["structured_slides"],
        "visual_context_slides": metrics["visual_context_slides"],
        "tables_code_formulas_slides": metrics["tables_code_formulas_slides"],
        "boilerplate_lines_removed": boilerplate_removed,
        "ocr_input_tokens": total_input_tokens,
        "ocr_output_tokens": total_output_tokens,
    }
    return artifact, warnings, stats


def run_ocr_stage(context: JobContext, settings: Settings) -> StageResult:
    """Pipeline stage handler for strict Google Gemini Slide OCR."""
    artifact, warnings, stats = extract_slides_with_gemini_ocr(
        context.slides_path,
        settings,
        source_pdf=context.slides_filename,
        progress_callback=context.report_progress,
        cancel_event=context.cancel_event,
    )

    context.slides_ocr_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = context.slides_ocr_path.with_name(f".{context.slides_ocr_path.stem}.tmp.json")
    temp_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temp_path, context.slides_ocr_path)

    batch_count = stats.get("ocr_batches", 1)
    batch_suffix = f" across {batch_count} batches" if batch_count > 1 else ""
    return StageResult(
        message=(
            f"Extracted OCR for {artifact.slide_count} slide(s){batch_suffix} "
            f"with Gemini ({artifact.model})."
        ),
        artifacts={"slides_ocr": ARTIFACT_SLIDES_OCR},
        stats=stats,
        warnings=warnings,
    )


__all__ = [
    "DEFAULT_OCR_MAX_BATCH_BYTES",
    "DEFAULT_OCR_MAX_PAGES_PER_BATCH",
    "DEFAULT_OCR_MODEL",
    "GenAIClientFactory",
    "PdfBatchPayload",
    "PdfBatchSpec",
    "SECTION_LABEL_TO_NAME",
    "SECTION_NAME_TO_LABEL",
    "STRUCTURED_OCR_SECTION_MARKERS",
    "STRUCTURED_OCR_SECTION_ORDER",
    "StructuredOCRSections",
    "build_ocr_prompt",
    "build_slide_ocr_records",
    "cleanup_structured_ocr_text",
    "compute_retry_delay_seconds",
    "extract_slides_with_gemini_ocr",
    "inspect_pdf_page_count",
    "is_auth_error",
    "is_payload_too_large_error",
    "is_transient_ocr_error",
    "normalize_batch_slides",
    "parse_gemini_ocr_response",
    "parse_structured_ocr_text",
    "plan_pdf_batches",
    "rebuild_structured_ocr_text",
    "run_ocr_stage",
    "slice_pdf_batches",
    "strip_repeating_boilerplate",
    "validate_gemini_credentials",
]
