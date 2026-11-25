from __future__ import annotations

import logging
import os
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx
from groq import Groq

from ..config import Settings
from ..ratelimit import OutboundRateLimiter
from ..schemas import (
    ARTIFACT_TRANSCRIPT,
    JobContext,
    ProgressCallback,
    StageError,
    StageResult,
    TranscriptArtifact,
    TranscriptSegment,
)
from .audio import (
    AudioChunkSpec,
    extract_audio_chunk,
    plan_audio_chunks,
    probe_audio_duration,
    resolve_ffmpeg_binaries,
)

logger = logging.getLogger(__name__)

GroqClientFactory = Callable[[Settings], Any]
SleepFn = Callable[[float], None]
JitterFn = Callable[[float, float], float]

_HALLUCINATION_PATTERNS: tuple[str, ...] = (
    "textning.nu",
    "texting.nu",
    "undertextning.nu",
)

_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_AUTH_HTTP_STATUSES = frozenset({401, 403})
_MAX_RETRY_AFTER_SECONDS = 60.0

_semaphore_lock = threading.Lock()
_global_transcribe_semaphores: dict[int, threading.BoundedSemaphore] = {}
_rate_limiter_lock = threading.Lock()
_global_transcribe_rate_limiters: dict[int, OutboundRateLimiter] = {}


def _get_global_transcribe_semaphore(max_concurrent: int) -> threading.BoundedSemaphore:
    limit = max(1, int(max_concurrent))
    with _semaphore_lock:
        semaphore = _global_transcribe_semaphores.get(limit)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(limit)
            _global_transcribe_semaphores[limit] = semaphore
        return semaphore


def _get_global_transcribe_rate_limiter(max_rpm: int) -> OutboundRateLimiter | None:
    limit = max(0, int(max_rpm))
    if limit <= 0:
        return None
    with _rate_limiter_lock:
        limiter = _global_transcribe_rate_limiters.get(limit)
        if limiter is None:
            limiter = OutboundRateLimiter(max_requests=limit, window_seconds=60.0)
            _global_transcribe_rate_limiters[limit] = limiter
        return limiter


def _default_groq_client_factory(settings: Settings) -> Groq:
    return Groq(
        api_key=settings.groq_api_key,
        timeout=settings.transcribe_timeout_seconds,
        max_retries=0,  # Retries and backoff are managed explicitly with job warnings/progress.
    )


def _extract_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int):
            return resp_status
    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        parsed = float(str(raw).strip())
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return min(parsed, _MAX_RETRY_AFTER_SECONDS)


def is_auth_error(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code in _AUTH_HTTP_STATUSES:
        return True
    message = str(exc).lower()
    return (
        "invalid api key" in message
        or "invalid_api_key" in message
        or "authentication" in message
        or "unauthorized" in message
    )


def is_request_too_large_error(exc: Exception) -> bool:
    status_code = _extract_status_code(exc)
    if status_code == 413:
        return True
    message = str(exc).lower()
    return (
        "error code: 413" in message
        or "request entity too large" in message
        or "request_too_large" in message
        or "maximum content size limit" in message
    )


def is_transient_transcription_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    status_code = _extract_status_code(exc)
    if status_code in _TRANSIENT_HTTP_STATUSES:
        return True
    message = str(exc).lower()
    return (
        "error code 524" in message
        or "a timeout occurred" in message
        or "timed out" in message
        or "temporarily unavailable" in message
        or "service unavailable" in message
        or "internal server error" in message
        or "bad gateway" in message
        or "rate limit" in message
        or "too many requests" in message
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


def is_hallucinated_segment(text: str) -> bool:
    lowered = text.lower().strip().strip(".")
    return any(pattern in lowered for pattern in _HALLUCINATION_PATTERNS)


def _field_value(obj: Any, key: str, default: Any) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj[key]
    except Exception:
        return default


def extract_raw_segments(
    response: Any,
    *,
    fallback_duration: float = 0.0,
) -> list[dict[str, Any]]:
    raw_segments = _field_value(response, "segments", None)
    normalized: list[dict[str, Any]] = []

    if raw_segments:
        for seg in raw_segments:
            text = str(_field_value(seg, "text", "") or "").strip()
            if not text or is_hallucinated_segment(text):
                continue
            try:
                start = max(0.0, float(_field_value(seg, "start", 0.0) or 0.0))
                end = max(start, float(_field_value(seg, "end", start) or start))
            except (TypeError, ValueError):
                start = 0.0
                end = 0.0
            normalized.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text,
                }
            )

    if not normalized:
        fallback_text = str(_field_value(response, "text", "") or "").strip()
        if fallback_text and not is_hallucinated_segment(fallback_text):
            normalized.append(
                {
                    "start": 0.0,
                    "end": round(max(0.0, float(fallback_duration or 0.0)), 2),
                    "text": fallback_text,
                }
            )

    return normalized


def merge_chunk_segments(
    chunk_results: Sequence[tuple[AudioChunkSpec, Sequence[dict[str, Any]]]],
) -> list[TranscriptSegment]:
    """Shift per-chunk timestamps by each chunk's start offset and sort chronologically."""
    merged_rows: list[tuple[float, float, int, int, str]] = []
    for chunk_spec, raw_segments in sorted(chunk_results, key=lambda item: item[0].index):
        offset = float(chunk_spec.start_seconds)
        for seg_pos, seg in enumerate(raw_segments):
            text = str(seg.get("text", "")).strip()
            if not text or is_hallucinated_segment(text):
                continue
            raw_start = max(0.0, float(seg.get("start", 0.0) or 0.0))
            raw_end = max(raw_start, float(seg.get("end", raw_start) or raw_start))
            shifted_start = round(max(0.0, offset + raw_start), 2)
            shifted_end = round(max(shifted_start, offset + raw_end), 2)
            merged_rows.append((shifted_start, shifted_end, chunk_spec.index, seg_pos, text))

    merged_rows.sort(key=lambda row: (row[0], row[1], row[2], row[3]))

    return [
        TranscriptSegment(
            index=idx,
            start=row[0],
            end=row[1],
            text=row[4],
        )
        for idx, row in enumerate(merged_rows)
    ]


def _transcribe_single_file_once(
    client: Any,
    mp3_path: Path,
    *,
    model: str,
    semaphore: threading.BoundedSemaphore,
    rate_limiter: OutboundRateLimiter | None = None,
    chunk_label: str | None = None,
    progress_callback: ProgressCallback | None = None,
    sleep_fn: SleepFn = time.sleep,
    fallback_duration: float = 0.0,
) -> list[dict[str, Any]]:
    if rate_limiter is not None:
        def _notify_wait(wait_seconds: float) -> None:
            wait_msg = (
                f"Groq rate limit pacing ({chunk_label}); waiting {wait_seconds:.1f}s..."
                if chunk_label
                else f"Groq rate limit pacing; waiting {wait_seconds:.1f}s..."
            )
            logger.info("%s", wait_msg)
            if progress_callback is not None:
                progress_callback(wait_msg)

        rate_limiter.acquire(sleep_fn=sleep_fn, on_wait=_notify_wait)

    with semaphore:
        with mp3_path.open("rb") as audio_handle:
            response = client.audio.transcriptions.create(
                model=model,
                file=(mp3_path.name, audio_handle),
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
    return extract_raw_segments(response, fallback_duration=fallback_duration)


def _transcribe_single_file_with_retries(
    client: Any,
    mp3_path: Path,
    *,
    settings: Settings,
    semaphore: threading.BoundedSemaphore,
    rate_limiter: OutboundRateLimiter | None = None,
    chunk_label: str | None = None,
    fallback_duration: float = 0.0,
    progress_callback: ProgressCallback | None = None,
    warnings: list[str] | None = None,
    warnings_lock: threading.Lock | None = None,
    sleep_fn: SleepFn = time.sleep,
) -> list[dict[str, Any]]:
    attempts = max(1, settings.transcribe_retry_attempts)
    label_prefix = f"{chunk_label}: " if chunk_label else ""

    for attempt in range(1, attempts + 1):
        try:
            segments = _transcribe_single_file_once(
                client,
                mp3_path,
                model=settings.transcription_model,
                semaphore=semaphore,
                rate_limiter=rate_limiter,
                chunk_label=chunk_label,
                progress_callback=progress_callback,
                sleep_fn=sleep_fn,
                fallback_duration=fallback_duration,
            )
            if attempt > 1 and warnings is not None:
                retries_used = attempt - 1
                retry_word = "once" if retries_used == 1 else f"{retries_used} times"
                note = (
                    f"{chunk_label} retried {retry_word} after a transient Groq API error."
                    if chunk_label
                    else f"Transcription retried {retry_word} after a transient Groq API error."
                )
                if warnings_lock is not None:
                    with warnings_lock:
                        warnings.append(note)
                else:
                    warnings.append(note)
            return segments
        except StageError:
            raise
        except Exception as exc:
            if is_request_too_large_error(exc):
                raise
            if is_auth_error(exc):
                raise StageError(
                    stage="transcription",
                    message="Groq API authentication failed while transcribing audio.",
                    hint="Verify that GROQ_API_KEY in .env is valid and active, restart the backend, and retry.",
                ) from exc
            if not is_transient_transcription_error(exc):
                raise StageError(
                    stage="transcription",
                    message=f"Groq Whisper transcription failed ({label_prefix}{exc}).",
                    hint="Check your TRANSCRIPTION_MODEL setting and Groq account status, then retry.",
                ) from exc
            if attempt >= attempts:
                raise StageError(
                    stage="transcription",
                    message=(
                        f"Groq Whisper transcription failed after {attempts} attempt(s) "
                        f"({label_prefix}{exc})."
                    ),
                    hint="Groq appears temporarily unavailable or rate-limited. Wait a moment and retry.",
                ) from exc

            delay = compute_retry_delay_seconds(
                attempt,
                settings.transcribe_retry_base_delay_seconds,
                retry_after=_extract_retry_after_seconds(exc),
            )
            retry_msg = (
                f"Groq API transient error ({chunk_label}); retrying "
                f"({attempt + 1}/{attempts}) in {delay:.1f}s..."
                if chunk_label
                else f"Groq API transient error; retrying ({attempt + 1}/{attempts}) in {delay:.1f}s..."
            )
            logger.warning("%s (%s)", retry_msg, exc)
            if progress_callback is not None:
                progress_callback(retry_msg)
            if delay > 0:
                sleep_fn(delay)

    raise AssertionError("Unreachable retry loop exit.")  # pragma: no cover


def _try_probe_duration_best_effort(
    mp3_path: Path,
    settings: Settings,
) -> float | None:
    try:
        binaries = resolve_ffmpeg_binaries(settings, stage="transcription")
        return probe_audio_duration(
            mp3_path,
            ffprobe_bin=binaries.ffprobe,
            stage="transcription",
        )
    except StageError:
        return None


def _transcribe_in_chunks(
    *,
    client: Any,
    mp3_path: Path,
    duration_seconds: float,
    chunk_specs: list[AudioChunkSpec],
    ffmpeg_bin: str,
    settings: Settings,
    semaphore: threading.BoundedSemaphore,
    rate_limiter: OutboundRateLimiter | None = None,
    progress_callback: ProgressCallback | None = None,
    warnings: list[str],
    sleep_fn: SleepFn = time.sleep,
) -> list[TranscriptSegment]:
    total_chunks = len(chunk_specs)
    warnings_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed_count = 0

    if progress_callback is not None:
        progress_callback(f"Transcribing audio across {total_chunks} chunks with Groq Whisper...")

    chunk_results: list[tuple[AudioChunkSpec, list[dict[str, Any]]]] = []

    with tempfile.TemporaryDirectory(prefix="slidenotelm-chunks-") as tmp_dir:
        tmp_root = Path(tmp_dir)

        def _process_chunk(spec: AudioChunkSpec) -> tuple[AudioChunkSpec, list[dict[str, Any]]]:
            chunk_path = tmp_root / f"chunk-{spec.index:04d}.mp3"
            label = f"Chunk {spec.index + 1}/{total_chunks}"
            try:
                extract_audio_chunk(
                    mp3_path,
                    chunk_path,
                    chunk=spec,
                    ffmpeg_bin=ffmpeg_bin,
                    target_bitrate=settings.transcribe_target_bitrate,
                    stage="transcription",
                )
                raw_segments = _transcribe_single_file_with_retries(
                    client,
                    chunk_path,
                    settings=settings,
                    semaphore=semaphore,
                    rate_limiter=rate_limiter,
                    chunk_label=label,
                    fallback_duration=spec.duration_seconds,
                    progress_callback=progress_callback,
                    warnings=warnings,
                    warnings_lock=warnings_lock,
                    sleep_fn=sleep_fn,
                )
            except StageError:
                raise
            except Exception as exc:
                if is_request_too_large_error(exc):
                    raise StageError(
                        stage="transcription",
                        message=(
                            f"{label} exceeded Groq's maximum upload size even after splitting "
                            f"(duration={duration_seconds:.1f}s)."
                        ),
                        hint=(
                            "Lower TRANSCRIBE_MAX_UPLOAD_BYTES or TRANSCRIBE_TARGET_BITRATE "
                            "in .env and retry."
                        ),
                    ) from exc
                raise
            finally:
                chunk_path.unlink(missing_ok=True)
            return spec, raw_segments

        worker_count = min(max(1, settings.transcribe_parallel_workers), total_chunks)
        if worker_count > 1:
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="slidenotelm-groq",
            ) as pool:
                futures = [pool.submit(_process_chunk, spec) for spec in chunk_specs]
                try:
                    for future in as_completed(futures):
                        chunk_results.append(future.result())
                        with progress_lock:
                            completed_count += 1
                            done = completed_count
                        if progress_callback is not None:
                            progress_callback(f"Transcribed chunk {done} of {total_chunks}...")
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
        else:
            for spec in chunk_specs:
                chunk_results.append(_process_chunk(spec))
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(f"Transcribed chunk {completed_count} of {total_chunks}...")

    return merge_chunk_segments(chunk_results)


def transcribe_audio_file(
    mp3_path: Path,
    settings: Settings,
    *,
    source_recording: str,
    progress_callback: ProgressCallback | None = None,
    client_factory: GroqClientFactory | None = None,
    rate_limiter: OutboundRateLimiter | None = None,
    sleep_fn: SleepFn = time.sleep,
    duration_seconds: float | None = None,
) -> tuple[TranscriptArtifact, list[str], int]:
    """Transcribe a normalized MP3 file via Groq Whisper with auto-chunking and retries."""
    if not settings.groq_api_key.strip():
        raise StageError(
            stage="transcription",
            message="GROQ_API_KEY is not configured.",
            hint="Add your Groq API key to GROQ_API_KEY in .env, restart the backend, and retry.",
        )

    audio_path = Path(mp3_path)
    if not audio_path.is_file():
        raise StageError(
            stage="transcription",
            message="Normalized audio file is missing before transcription.",
            hint="Re-upload the recording so Audio Prep can generate audio_normalized.mp3.",
        )

    file_size_bytes = audio_path.stat().st_size
    if file_size_bytes == 0:
        raise StageError(
            stage="transcription",
            message="Normalized audio file is empty.",
            hint="Verify the uploaded recording contains audible sound and retry.",
        )

    factory = client_factory or _default_groq_client_factory
    client = factory(settings)
    semaphore = _get_global_transcribe_semaphore(settings.transcribe_global_max_concurrent)
    transcribe_rpm = max(0, int(getattr(settings, "transcribe_rate_limit_rpm", 0)))
    if rate_limiter is not None:
        resolved_rate_limiter: OutboundRateLimiter | None = rate_limiter
    elif client_factory is not None:
        resolved_rate_limiter = (
            OutboundRateLimiter(max_requests=transcribe_rpm, window_seconds=60.0)
            if transcribe_rpm > 0
            else None
        )
    else:
        resolved_rate_limiter = _get_global_transcribe_rate_limiter(transcribe_rpm)
    warnings: list[str] = []

    resolved_duration = duration_seconds
    needs_chunking = file_size_bytes > settings.transcribe_safe_upload_bytes
    force_split = False

    if not needs_chunking:
        if progress_callback is not None:
            progress_callback("Transcribing recording with Groq Whisper...")
        try:
            raw_segments = _transcribe_single_file_with_retries(
                client,
                audio_path,
                settings=settings,
                semaphore=semaphore,
                rate_limiter=resolved_rate_limiter,
                fallback_duration=resolved_duration or 0.0,
                progress_callback=progress_callback,
                warnings=warnings,
                sleep_fn=sleep_fn,
            )
            segments = merge_chunk_segments(
                [
                    (
                        AudioChunkSpec(
                            index=0,
                            start_seconds=0.0,
                            duration_seconds=resolved_duration or 0.0,
                        ),
                        raw_segments,
                    )
                ]
            )
            if resolved_duration is None:
                resolved_duration = _try_probe_duration_best_effort(audio_path, settings)
            if resolved_duration is None:
                resolved_duration = max((seg.end for seg in segments), default=0.0)
            artifact = TranscriptArtifact(
                source_recording=source_recording,
                model=settings.transcription_model,
                duration_seconds=round(max(0.0, resolved_duration), 2),
                segments=segments,
            )
            return artifact, warnings, 1
        except Exception as exc:
            if not is_request_too_large_error(exc):
                raise
            force_split = True
            warnings.append(
                "Recording exceeded Groq's single-request upload limit; retried with chunked uploads."
            )
            logger.warning("Groq returned 413 for %s; falling back to chunked transcription.", audio_path)
            if progress_callback is not None:
                progress_callback("Audio exceeded single-upload limit; splitting into chunks...")

    binaries = resolve_ffmpeg_binaries(settings, stage="transcription")
    if resolved_duration is None:
        resolved_duration = probe_audio_duration(
            audio_path,
            ffprobe_bin=binaries.ffprobe,
            stage="transcription",
        )

    chunk_specs = plan_audio_chunks(
        file_size_bytes=file_size_bytes,
        duration_seconds=resolved_duration,
        safe_max_bytes=settings.transcribe_safe_upload_bytes,
        min_chunk_seconds=settings.transcribe_min_chunk_seconds,
        force_split=force_split,
    )

    segments = _transcribe_in_chunks(
        client=client,
        mp3_path=audio_path,
        duration_seconds=resolved_duration,
        chunk_specs=chunk_specs,
        ffmpeg_bin=binaries.ffmpeg,
        settings=settings,
        semaphore=semaphore,
        rate_limiter=resolved_rate_limiter,
        progress_callback=progress_callback,
        warnings=warnings,
        sleep_fn=sleep_fn,
    )

    artifact = TranscriptArtifact(
        source_recording=source_recording,
        model=settings.transcription_model,
        duration_seconds=round(max(0.0, resolved_duration), 2),
        segments=segments,
    )
    return artifact, warnings, len(chunk_specs)


def run_transcription_stage(context: JobContext, settings: Settings) -> StageResult:
    artifact, warnings, chunk_count = transcribe_audio_file(
        context.audio_normalized_path,
        settings,
        source_recording=context.recording_filename,
        progress_callback=context.report_progress,
    )

    context.transcript_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = context.transcript_path.with_name(f".{context.transcript_path.stem}.tmp.json")
    temp_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temp_path, context.transcript_path)

    segment_count = len(artifact.segments)
    chunk_suffix = f" across {chunk_count} chunks" if chunk_count > 1 else ""
    return StageResult(
        message=f"Transcribed {segment_count} segment(s){chunk_suffix} with Groq Whisper.",
        artifacts={"transcript": ARTIFACT_TRANSCRIPT},
        stats={
            "duration_seconds": artifact.duration_seconds,
            "transcript_segments": segment_count,
            "transcription_chunks": chunk_count,
        },
        warnings=warnings,
    )


__all__ = [
    "GroqClientFactory",
    "compute_retry_delay_seconds",
    "extract_raw_segments",
    "is_auth_error",
    "is_hallucinated_segment",
    "is_request_too_large_error",
    "is_transient_transcription_error",
    "merge_chunk_segments",
    "run_transcription_stage",
    "transcribe_audio_file",
]
