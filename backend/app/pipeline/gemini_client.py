from __future__ import annotations

import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import httpx
from google import genai
from google.genai import types

from ..config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LLM_GLOBAL_MAX_CONCURRENT,
    DEFAULT_LLM_RATE_LIMIT_RPM,
    DEFAULT_LLM_RETRY_ATTEMPTS,
    DEFAULT_LLM_RETRY_BASE_DELAY_SECONDS,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    THINKING_LEVEL_DEFAULT_ALIASES,
    THINKING_LEVEL_OFF_ALIASES,
    VALID_THINKING_LEVELS,
    Settings,
)
from ..ratelimit import OutboundRateLimiter
from ..schemas import PipelineStage, ProgressCallback, StageError

logger = logging.getLogger(__name__)

T = TypeVar("T")

GenAIClientFactory = Callable[[Settings], Any]
SleepFn = Callable[[float], None]
JitterFn = Callable[[float, float], float]

GEMINI_LIVE_MODES: frozenset[str] = frozenset({"gemini", "live"})

_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_AUTH_HTTP_STATUSES = frozenset({401, 403})
_TRANSIENT_RPC_STATUSES = frozenset(
    {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "ABORTED"}
)
_AUTH_RPC_STATUSES = frozenset({"UNAUTHENTICATED", "PERMISSION_DENIED"})
_MAX_RETRY_AFTER_SECONDS = 90.0

_GEMINI_MAJOR_VERSION_RE = re.compile(r"^gemini-(\d+)(?:[.\-]|$)", re.IGNORECASE)
_RETRY_DELAY_SECONDS_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s\s*$", re.IGNORECASE)

_NON_RETRYABLE_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "MODEL_ARMOR",
    }
)

_semaphore_lock = threading.Lock()
_global_llm_semaphores: dict[int, threading.BoundedSemaphore] = {}
_rate_limiter_lock = threading.Lock()
_global_llm_rate_limiters: dict[int, OutboundRateLimiter] = {}


class _RetryableOutputError(Exception):
    """Internal exception for empty or unparseable model output that should be retried."""


@dataclass(frozen=True)
class GeminiTextResult:
    """Text response, optional parsed value, and token accounting from a Gemini call."""

    text: str
    model: str
    thinking_level: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    parsed: Any = None


def is_gemini_live_mode(raw_mode: str | None) -> bool:
    """Return True when PIPELINE_LLM_MODE selects the live Google Gemini provider."""
    mode = (raw_mode or "").strip().lower()
    return mode in GEMINI_LIVE_MODES


def normalize_thinking_level(raw_level: str | None) -> str:
    """Normalize a thinking_level string to uppercase, mapping OFF/NONE -> MINIMAL."""
    if not raw_level:
        return ""
    cleaned = raw_level.strip().upper()
    if cleaned in THINKING_LEVEL_OFF_ALIASES:
        return "MINIMAL"
    if cleaned in THINKING_LEVEL_DEFAULT_ALIASES:
        return ""
    if cleaned in VALID_THINKING_LEVELS:
        return cleaned
    return ""


def supports_thinking_level(model: str | None) -> bool:
    """Return True for Gemini 3+ / Gemini 4 models that accept ThinkingConfig.thinking_level."""
    cleaned = (model or "").strip().lower()
    if not cleaned:
        return True
    basename = cleaned.rsplit("/", 1)[-1]
    match = _GEMINI_MAJOR_VERSION_RE.match(basename)
    if match:
        return int(match.group(1)) >= 3
    # Non-versioned aliases (gemini-pro-latest, gemma-*, etc.) do not accept thinking_level.
    return False


def build_thinking_config(
    thinking_level: str | None,
    *,
    model: str | None = None,
) -> types.ThinkingConfig | None:
    """Build a Gemini 4 ThinkingConfig for a valid thinking_level, or None when unset/unsupported."""
    if not supports_thinking_level(model):
        return None
    level = normalize_thinking_level(thinking_level)
    if not level:
        return None
    return types.ThinkingConfig(thinking_level=level)


def validate_gemini_credentials(settings: Settings, *, stage: PipelineStage) -> None:
    """Validate Google Gemini / Vertex AI credentials before running a Gemini-backed stage."""
    if settings.google_genai_use_vertexai:
        if not settings.google_cloud_project.strip():
            raise StageError(
                stage=stage,
                message=(
                    "GOOGLE_GENAI_USE_VERTEXAI is enabled, but GOOGLE_CLOUD_PROJECT "
                    "is not configured."
                ),
                hint=(
                    "Set GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION) in .env "
                    "with valid Google Cloud Application Default Credentials, or set "
                    "GOOGLE_GENAI_USE_VERTEXAI=false and provide GEMINI_API_KEY."
                ),
            )
        return

    if not settings.gemini_api_key.strip():
        stage_label = "slide OCR" if stage == "ocr" else f"the {stage} stage"
        extra = (
            " Slide OCR does not fall back to plain text extraction."
            if stage == "ocr"
            else " Set PIPELINE_LLM_MODE=stub to run deterministic local scaffolding instead."
        )
        raise StageError(
            stage=stage,
            message=f"GEMINI_API_KEY is not configured for {stage_label}.",
            hint=(
                "Add your Google Gemini API key to GEMINI_API_KEY in .env "
                "(or set GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT for Vertex AI), "
                f"restart the backend, and retry.{extra}"
            ),
        )


def create_gemini_client(
    settings: Settings,
    *,
    timeout_seconds: float | None = None,
) -> genai.Client:
    """Construct a Google GenAI SDK client for Gemini Developer API or Vertex AI."""
    resolved_timeout = max(
        5.0,
        float(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "llm_timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS)
        ),
    )
    http_options = types.HttpOptions(timeout=int(resolved_timeout * 1000))

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


def close_gemini_client(client: Any | None) -> None:
    """Safely close a Google GenAI client if it exposes a close() method."""
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            logger.debug("Ignoring error while closing Gemini client.", exc_info=True)


def _get_global_llm_semaphore(max_concurrent: int) -> threading.BoundedSemaphore:
    limit = max(1, int(max_concurrent))
    with _semaphore_lock:
        semaphore = _global_llm_semaphores.get(limit)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _global_llm_semaphores[limit] = semaphore
        return semaphore


def _get_global_llm_rate_limiter(max_rpm: int) -> OutboundRateLimiter | None:
    limit = max(0, int(max_rpm))
    if limit <= 0:
        return None
    with _rate_limiter_lock:
        limiter = _global_llm_rate_limiters.get(limit)
        if limiter is None:
            limiter = OutboundRateLimiter(max_requests=limit, window_seconds=60.0)
            _global_llm_rate_limiters[limit] = limiter
        return limiter


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


def _parse_retry_delay_value(raw_delay: Any) -> float | None:
    if isinstance(raw_delay, (int, float)):
        parsed = float(raw_delay)
        return parsed if parsed > 0 else None
    if isinstance(raw_delay, str):
        match = _RETRY_DELAY_SECONDS_RE.match(raw_delay)
        if match:
            parsed = float(match.group(1))
            return parsed if parsed > 0 else None
        try:
            parsed = float(raw_delay.strip())
            return parsed if parsed > 0 else None
        except ValueError:
            return None
    return None


def extract_retry_after_seconds(exc: Exception) -> float | None:
    """Extract retry delay from HTTP Retry-After header or Google RPC RetryInfo details."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is not None:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        parsed = _parse_retry_delay_value(raw)
        if parsed is not None:
            return min(parsed, _MAX_RETRY_AFTER_SECONDS)

    # Google API errors embed RetryInfo inside exc.details["error"]["details"].
    details_root = getattr(exc, "details", None)
    if isinstance(details_root, dict):
        error_obj = details_root.get("error")
        detail_list = (
            error_obj.get("details")
            if isinstance(error_obj, dict)
            else details_root.get("details")
        )
        if isinstance(detail_list, list):
            for entry in detail_list:
                if not isinstance(entry, dict):
                    continue
                entry_type = str(entry.get("@type") or "")
                if entry_type.endswith("RetryInfo") or "retryDelay" in entry:
                    parsed = _parse_retry_delay_value(entry.get("retryDelay"))
                    if parsed is not None:
                        return min(parsed, _MAX_RETRY_AFTER_SECONDS)
    return None


def is_gemini_auth_error(exc: Exception) -> bool:
    """Identify authentication/permission failures without misclassifying 429 quota errors."""
    status_code = _extract_status_code(exc)
    rpc_status = str(getattr(exc, "status", "") or "").upper()
    message = str(exc).lower()

    if status_code is not None:
        if status_code in _AUTH_HTTP_STATUSES:
            return True
        if status_code == 400:
            return (
                "api_key_invalid" in message
                or "api key not valid" in message
                or "invalid api key" in message
            )
        # Known non-auth HTTP status (e.g. 429 RESOURCE_EXHAUSTED whose body mentions "billing").
        return False

    if rpc_status:
        if rpc_status in _AUTH_RPC_STATUSES:
            return True
        if rpc_status in _TRANSIENT_RPC_STATUSES:
            return False

    auth_markers = (
        "api key not valid",
        "api_key_invalid",
        "invalid api key",
        "unauthenticated",
        "permission_denied",
        "permission denied",
        "default credentials",
        "could not automatically determine credentials",
        "credentials were not found",
    )
    return any(marker in message for marker in auth_markers)


def is_transient_gemini_error(exc: Exception) -> bool:
    """Identify transient transport, rate-limit, quota, and server errors."""
    if isinstance(
        exc,
        (
            _RetryableOutputError,
            TimeoutError,
            ConnectionError,
            httpx.TimeoutException,
            httpx.NetworkError,
        ),
    ):
        return True

    status_code = _extract_status_code(exc)
    if status_code is not None:
        return status_code in _TRANSIENT_HTTP_STATUSES

    rpc_status = str(getattr(exc, "status", "") or "").upper()
    if rpc_status in _TRANSIENT_RPC_STATUSES:
        return True

    message = str(exc).lower()
    transient_markers = (
        "resource_exhausted",
        "rate limit",
        "too many requests",
        "quota exceeded",
        "temporarily unavailable",
        "service unavailable",
        "overloaded",
        "deadline exceeded",
        "timed out",
        "timeout",
        "connection reset",
        "bad gateway",
        "gateway timeout",
        "internal server error",
    )
    return any(marker in message for marker in transient_markers)


def compute_retry_delay_seconds(
    attempt: int,
    base_delay: float,
    *,
    retry_after: float | None = None,
    jitter_fn: JitterFn = random.uniform,
) -> float:
    if retry_after is not None and retry_after > 0:
        return min(retry_after, _MAX_RETRY_AFTER_SECONDS)
    if base_delay <= 0:
        return 0.0
    exponential = base_delay * (2 ** max(0, attempt - 1))
    jitter = jitter_fn(0.0, min(1.0, base_delay * 0.5))
    return min(exponential + max(0.0, jitter), _MAX_RETRY_AFTER_SECONDS)


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    return str(value).strip().upper()


def _check_response_safety_and_finish(
    response: Any,
    *,
    stage: PipelineStage,
    task_label: str,
    model: str,
    max_output_tokens: int,
) -> None:
    """Fail loudly on blocked prompts, safety/recitation halts, or MAX_TOKENS truncation."""
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = _enum_name(getattr(prompt_feedback, "block_reason", None))
    if block_reason and block_reason not in {"BLOCKED_REASON_UNSPECIFIED", "BLOCK_REASON_UNSPECIFIED", "NONE"}:
        raise StageError(
            stage=stage,
            message=f"Google Gemini blocked the prompt for {task_label} (block_reason={block_reason}).",
            hint=(
                "Inspect the lecture slide or transcript text around this span for content "
                "triggering safety filters, or adjust the input and retry."
            ),
        )

    candidates = getattr(response, "candidates", None)
    if not candidates or not isinstance(candidates, (list, tuple)):
        return
    first = candidates[0]
    finish_reason = _enum_name(getattr(first, "finish_reason", None))
    if not finish_reason or finish_reason in {"FINISH_REASON_UNSPECIFIED", "STOP"}:
        return

    if finish_reason == "MAX_TOKENS":
        raise StageError(
            stage=stage,
            message=(
                f"Google Gemini truncated output for {task_label} (finish_reason=MAX_TOKENS "
                f"at limit {max_output_tokens})."
            ),
            hint=(
                "Thinking tokens count toward max_output_tokens on Gemini 4 models. "
                "Increase the stage's LLM_*_MAX_OUTPUT_TOKENS limit or lower its "
                "LLM_*_THINKING_LEVEL (e.g. MEDIUM or LOW) in .env and retry."
            ),
        )

    if finish_reason in _NON_RETRYABLE_FINISH_REASONS:
        raise StageError(
            stage=stage,
            message=(
                f"Google Gemini stopped generating {task_label} "
                f"(finish_reason={finish_reason}, model={model})."
            ),
            hint=(
                "Gemini halted generation due to content/policy filtering. Review the source "
                "span or retry with adjusted lecture inputs."
            ),
        )


def extract_response_text(response: Any) -> str:
    """Safely extract non-thought text from a Google GenAI GenerateContentResponse."""
    if response is None:
        return ""
    try:
        text = getattr(response, "text", None)
    except ValueError:
        return ""
    return str(text or "")


def extract_usage_tokens(response: Any) -> tuple[int, int, int]:
    """Return (input_tokens, output_tokens, thinking_tokens) from usage_metadata."""
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return 0, 0, 0
    input_tokens = max(0, int(getattr(metadata, "prompt_token_count", 0) or 0))
    output_tokens = max(0, int(getattr(metadata, "candidates_token_count", 0) or 0))
    thinking_tokens = max(0, int(getattr(metadata, "thoughts_token_count", 0) or 0))
    return input_tokens, output_tokens, thinking_tokens


def _auth_error_hint(settings: Settings, *, model: str) -> str:
    if settings.google_genai_use_vertexai:
        return (
            "Verify GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and your Google Cloud "
            f"Application Default Credentials for model '{model}', then retry."
        )
    return (
        f"Verify that GEMINI_API_KEY in .env is valid and has access/billing enabled for "
        f"model '{model}'. To change models, update OCR_MODEL_GOOGLE, LLM_ALIGN_MODEL, "
        "LLM_NOTES_MODEL, and LLM_COVERAGE_MODEL in .env, restart the backend, and retry."
    )


def call_gemini_text(
    *,
    stage: PipelineStage,
    task_label: str,
    settings: Settings,
    model: str,
    thinking_level: str,
    system_instruction: str,
    user_prompt: str,
    max_output_tokens: int,
    response_mime_type: str | None = None,
    response_json_schema: dict[str, Any] | None = None,
    parse_fn: Callable[[str], T] | None = None,
    client: Any | None = None,
    client_factory: GenAIClientFactory | None = None,
    rate_limiter: OutboundRateLimiter | None = None,
    progress_callback: ProgressCallback | None = None,
    sleep_fn: SleepFn = time.sleep,
) -> GeminiTextResult:
    """Call Google Gemini generate_content with Gemini 4 thinking_level, pacing, and retries."""
    validate_gemini_credentials(settings, stage=stage)

    resolved_model = (model or "").strip() or DEFAULT_GEMINI_MODEL
    resolved_level = normalize_thinking_level(thinking_level)
    resolved_max_tokens = max(1024, int(max_output_tokens))
    thinking_config = build_thinking_config(resolved_level, model=resolved_model)

    config_kwargs: dict[str, Any] = {
        "system_instruction": system_instruction,
        "max_output_tokens": resolved_max_tokens,
    }
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type
    if response_json_schema is not None:
        config_kwargs["response_json_schema"] = response_json_schema
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    # Gemini 3+/4 reasoning models recommend keeping the default temperature (1.0).
    if not supports_thinking_level(resolved_model):
        config_kwargs["temperature"] = 0.0
    config = types.GenerateContentConfig(**config_kwargs)

    owns_client = False
    if client is None:
        factory = client_factory or create_gemini_client
        try:
            client = factory(settings)
            owns_client = client_factory is None
        except StageError:
            raise
        except Exception as exc:
            raise StageError(
                stage=stage,
                message=f"Failed to initialize Google Gemini client for {task_label}: {exc}",
                hint=_auth_error_hint(settings, model=resolved_model),
            ) from exc

    attempts = max(1, int(getattr(settings, "llm_retry_attempts", DEFAULT_LLM_RETRY_ATTEMPTS)))
    base_delay = max(
        0.0,
        float(
            getattr(
                settings,
                "llm_retry_base_delay_seconds",
                DEFAULT_LLM_RETRY_BASE_DELAY_SECONDS,
            )
        ),
    )
    max_concurrent = max(
        1,
        int(getattr(settings, "llm_global_max_concurrent", DEFAULT_LLM_GLOBAL_MAX_CONCURRENT)),
    )
    rate_limit_rpm = max(
        0,
        int(getattr(settings, "llm_rate_limit_rpm", DEFAULT_LLM_RATE_LIMIT_RPM)),
    )

    semaphore = _get_global_llm_semaphore(max_concurrent)
    if rate_limiter is not None:
        resolved_limiter: OutboundRateLimiter | None = rate_limiter
    elif client_factory is not None:
        resolved_limiter = (
            OutboundRateLimiter(max_requests=rate_limit_rpm, window_seconds=60.0)
            if rate_limit_rpm > 0
            else None
        )
    else:
        resolved_limiter = _get_global_llm_rate_limiter(rate_limit_rpm)

    try:
        for attempt in range(1, attempts + 1):
            try:
                if resolved_limiter is not None:
                    def _notify_wait(wait_seconds: float) -> None:
                        wait_msg = (
                            f"Gemini rate limit pacing ({task_label}); waiting {wait_seconds:.1f}s..."
                        )
                        logger.info("%s", wait_msg)
                        if progress_callback is not None:
                            progress_callback(wait_msg)

                    resolved_limiter.acquire(sleep_fn=sleep_fn, on_wait=_notify_wait)

                with semaphore:
                    response = client.models.generate_content(
                        model=resolved_model,
                        contents=user_prompt,
                        config=config,
                    )

                _check_response_safety_and_finish(
                    response,
                    stage=stage,
                    task_label=task_label,
                    model=resolved_model,
                    max_output_tokens=resolved_max_tokens,
                )

                text = extract_response_text(response).strip()
                if not text:
                    raise _RetryableOutputError(
                        f"Gemini returned an empty response for {task_label}."
                    )

                parsed_value: Any = None
                if parse_fn is not None:
                    try:
                        parsed_value = parse_fn(text)
                    except ValueError as parse_exc:
                        raise _RetryableOutputError(str(parse_exc)) from parse_exc

                in_tok, out_tok, think_tok = extract_usage_tokens(response)
                return GeminiTextResult(
                    text=text,
                    model=resolved_model,
                    thinking_level=resolved_level,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    thinking_tokens=think_tok,
                    parsed=parsed_value,
                )
            except StageError:
                raise
            except Exception as exc:
                if is_gemini_auth_error(exc):
                    raise StageError(
                        stage=stage,
                        message=f"Google Gemini authentication failed ({task_label}): {exc}",
                        hint=_auth_error_hint(settings, model=resolved_model),
                    ) from exc

                if not is_transient_gemini_error(exc):
                    raise StageError(
                        stage=stage,
                        message=f"Google Gemini call failed ({task_label}): {exc}",
                        hint=(
                            f"Check model '{resolved_model}' and thinking level "
                            f"'{resolved_level or 'default'}', or set PIPELINE_LLM_MODE=stub "
                            "to run deterministic local scaffolding."
                        ),
                    ) from exc

                if attempt >= attempts:
                    raise StageError(
                        stage=stage,
                        message=(
                            f"Google Gemini failed after {attempts} attempt(s) "
                            f"({task_label}): {exc}"
                        ),
                        hint=(
                            "Gemini appears temporarily unavailable, rate-limited, or returned "
                            "empty/invalid output repeatedly. Wait a moment and retry."
                        ),
                    ) from exc

                delay = compute_retry_delay_seconds(
                    attempt,
                    base_delay,
                    retry_after=extract_retry_after_seconds(exc),
                )
                retry_msg = (
                    f"Gemini transient error ({task_label}); "
                    f"retrying ({attempt + 1}/{attempts}) in {delay:.1f}s..."
                )
                logger.warning("%s (%s)", retry_msg, exc)
                if progress_callback is not None:
                    progress_callback(retry_msg)
                if delay > 0:
                    sleep_fn(delay)

        raise AssertionError("Unreachable retry loop exit.")  # pragma: no cover
    finally:
        if owns_client:
            close_gemini_client(client)


__all__ = [
    "GEMINI_LIVE_MODES",
    "GeminiTextResult",
    "GenAIClientFactory",
    "build_thinking_config",
    "call_gemini_text",
    "close_gemini_client",
    "compute_retry_delay_seconds",
    "create_gemini_client",
    "extract_response_text",
    "extract_retry_after_seconds",
    "extract_usage_tokens",
    "is_gemini_auth_error",
    "is_gemini_live_mode",
    "is_transient_gemini_error",
    "normalize_thinking_level",
    "supports_thinking_level",
    "validate_gemini_credentials",
]
