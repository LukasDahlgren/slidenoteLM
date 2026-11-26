from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import DEFAULT_TRANSCRIBE_TARGET_BITRATE, Settings
from ..schemas import (
    ARTIFACT_AUDIO_NORMALIZED,
    JobContext,
    PipelineStage,
    StageError,
    StageResult,
)

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
DEFAULT_TARGET_BITRATE = DEFAULT_TRANSCRIBE_TARGET_BITRATE
FFMPEG_STALL_SECONDS = 180

FFMPEG_INSTALL_HINT = (
    "Install FFmpeg so both 'ffmpeg' and 'ffprobe' are available on PATH "
    "(Windows: 'winget install Gyan.FFmpeg', macOS: 'brew install ffmpeg'), "
    "restart your terminal and backend, then retry."
)


@dataclass(frozen=True)
class FFmpegBinaries:
    ffmpeg: str
    ffprobe: str


@dataclass(frozen=True)
class AudioChunkSpec:
    index: int
    start_seconds: float
    duration_seconds: float

    @property
    def end_seconds(self) -> float:
        return round(self.start_seconds + self.duration_seconds, 3)


@dataclass(frozen=True)
class AudioChunkFile:
    index: int
    start_seconds: float
    duration_seconds: float
    path: Path
    size_bytes: int


def _resolve_single_binary(name: str, configured_path: str = "") -> str | None:
    cleaned = configured_path.strip()
    if cleaned:
        candidate = Path(cleaned)
        if candidate.is_file():
            return str(candidate)
        return shutil.which(cleaned)
    return shutil.which(name)


def resolve_ffmpeg_binaries(
    settings: Settings | None = None,
    *,
    stage: PipelineStage = "audio_prep",
) -> FFmpegBinaries:
    ffmpeg_override = settings.ffmpeg_path if settings is not None else ""
    ffprobe_override = settings.ffprobe_path if settings is not None else ""

    ffmpeg_bin = _resolve_single_binary("ffmpeg", ffmpeg_override)
    ffprobe_bin = _resolve_single_binary("ffprobe", ffprobe_override)

    missing: list[str] = []
    if not ffmpeg_bin:
        missing.append("ffmpeg")
    if not ffprobe_bin:
        missing.append("ffprobe")

    if missing:
        missing_str = " and ".join(missing)
        raise StageError(
            stage=stage,
            message=f"Required media tool(s) not found on PATH: {missing_str}.",
            hint=FFMPEG_INSTALL_HINT,
            fatal_for_queue=True,
        )

    assert ffmpeg_bin is not None
    assert ffprobe_bin is not None
    return FFmpegBinaries(ffmpeg=ffmpeg_bin, ffprobe=ffprobe_bin)


def _decode_stderr(stderr: str | bytes | None) -> str:
    if not stderr:
        return ""
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", errors="replace")
    return str(stderr)


def _format_stderr_snippet(stderr: str | bytes | None) -> str:
    text = _decode_stderr(stderr)
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    snippet = " | ".join(lines[-2:])
    if len(snippet) > 300:
        snippet = snippet[-300:]
    return snippet


def _is_no_audio_stream_error(stderr: str) -> bool:
    lowered = stderr.lower()
    return (
        "matches no streams" in lowered
        or "does not contain any stream" in lowered
        or "stream map '0:a:0'" in lowered
        or "stream specifier ':a:0'" in lowered
    )


def _format_duration_label(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _parse_positive_duration(raw_output: str) -> float | None:
    for line in raw_output.splitlines():
        token = line.strip()
        if not token or token.upper() == "N/A":
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def probe_audio_duration(
    media_path: Path,
    *,
    ffprobe_bin: str = "ffprobe",
    stage: PipelineStage = "audio_prep",
) -> float:
    path = Path(media_path)
    if not path.is_file():
        raise StageError(
            stage=stage,
            message=f"Audio file could not be found for duration probing: {path.name}.",
            hint="Re-upload the recording and try again.",
        )

    queries = (
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
    )

    last_stderr = ""
    for command in queries:
        try:
            proc = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise StageError(
                stage=stage,
                message="Required media tool 'ffprobe' could not be executed.",
                hint=FFMPEG_INSTALL_HINT,
            ) from exc
        except subprocess.CalledProcessError as exc:
            last_stderr = _format_stderr_snippet(exc.stderr)
            continue

        duration = _parse_positive_duration(proc.stdout or "")
        if duration is not None:
            return duration

    detail = f" ({last_stderr})" if last_stderr else ""
    raise StageError(
        stage=stage,
        message=f"Could not determine a valid audio duration for '{path.name}'{detail}.",
        hint="Verify that the uploaded file is a valid, uncorrupted audio or video recording with sound.",
    )


def _run_ffmpeg(command: list[str], output: Path, stage: PipelineStage) -> None:
    # Jobs run unattended. In particular, FFmpeg must never poll the server's console.
    with subprocess.Popen(
        [command[0], "-nostdin", *command[1:]],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    ) as proc:
        size = 0
        while True:
            try:
                _, stderr = proc.communicate(timeout=FFMPEG_STALL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                current_size = output.stat().st_size if output.exists() else 0
                if current_size > size:
                    size = current_size
                    continue
                proc.kill()
                proc.communicate()
                output.unlink(missing_ok=True)
                raise StageError(
                    stage=stage,
                    message=f"FFmpeg wrote no audio for {FFMPEG_STALL_SECONDS} seconds.",
                    hint="Check that the recording plays through and that the job disk has free space.",
                )
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, command, stderr=stderr)


def normalize_audio_to_mp3(
    source_path: Path,
    target_mp3_path: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    target_bitrate: str = DEFAULT_TARGET_BITRATE,
    stage: PipelineStage = "audio_prep",
) -> Path:
    source = Path(source_path)
    target = Path(target_mp3_path)
    if not source.is_file():
        raise StageError(
            stage=stage,
            message="Uploaded recording file is missing from job storage.",
            hint="Re-upload the slide PDF and recording to start a fresh job.",
        )

    resolved_bitrate = str(target_bitrate or "").strip() or DEFAULT_TARGET_BITRATE
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_name(f".{target.stem}.tmp.mp3")

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "libmp3lame",
        "-b:a",
        resolved_bitrate,
        str(temp_target),
    ]

    try:
        _run_ffmpeg(command, temp_target, stage)
    except FileNotFoundError as exc:
        temp_target.unlink(missing_ok=True)
        raise StageError(
            stage=stage,
            message="Required media tool 'ffmpeg' could not be executed.",
            hint=FFMPEG_INSTALL_HINT,
        ) from exc
    except subprocess.CalledProcessError as exc:
        temp_target.unlink(missing_ok=True)
        raw_stderr = _decode_stderr(exc.stderr)
        if _is_no_audio_stream_error(raw_stderr):
            raise StageError(
                stage=stage,
                message="The uploaded recording does not contain an audio track.",
                hint="Upload an audio file or a video recording that includes spoken audio.",
            ) from exc
        stderr = _format_stderr_snippet(raw_stderr)
        detail = f": {stderr}" if stderr else "."
        raise StageError(
            stage=stage,
            message=f"ffmpeg failed to extract and normalize audio from '{source.name}'{detail}",
            hint="Confirm the recording is not corrupted or DRM-protected, then try re-uploading.",
        ) from exc

    if not temp_target.is_file() or temp_target.stat().st_size == 0:
        temp_target.unlink(missing_ok=True)
        raise StageError(
            stage=stage,
            message="ffmpeg produced an empty audio file during normalization.",
            hint="Verify the recording contains audible sound and re-upload.",
        )

    os.replace(temp_target, target)
    return target


def plan_audio_chunks(
    *,
    file_size_bytes: int,
    duration_seconds: float,
    safe_max_bytes: int,
    min_chunk_seconds: int = 300,
    force_split: bool = False,
) -> list[AudioChunkSpec]:
    """Plan contiguous, balanced chunk specs covering [0, duration_seconds]."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be a positive finite number.")
    if safe_max_bytes <= 0:
        raise ValueError("safe_max_bytes must be positive.")
    if file_size_bytes < 0:
        raise ValueError("file_size_bytes cannot be negative.")

    total_duration = round(duration_seconds, 3)
    if total_duration <= 0:
        total_duration = duration_seconds

    if not force_split and file_size_bytes <= safe_max_bytes:
        return [AudioChunkSpec(index=0, start_seconds=0.0, duration_seconds=total_duration)]

    bytes_per_second = file_size_bytes / max(duration_seconds, 0.001)
    if bytes_per_second <= 0:
        chunk_count = 2 if (force_split and duration_seconds > 1.0) else 1
    else:
        fit_seconds = int(safe_max_bytes / bytes_per_second)
        if fit_seconds >= min_chunk_seconds:
            max_chunk_seconds = fit_seconds
        else:
            max_chunk_seconds = max(1, fit_seconds)
        chunk_count = max(1, math.ceil(duration_seconds / max_chunk_seconds))
        if force_split and duration_seconds > 1.0:
            chunk_count = max(2, chunk_count)

    # Cap chunk count so each chunk is at least ~0.5s on very short clips.
    max_sane_chunks = max(1, int(math.floor(duration_seconds / 0.5)))
    chunk_count = min(chunk_count, max_sane_chunks)

    boundaries = [round((idx * total_duration) / chunk_count, 3) for idx in range(chunk_count)]
    boundaries.append(total_duration)

    chunks: list[AudioChunkSpec] = []
    for idx in range(chunk_count):
        start = boundaries[idx]
        end = boundaries[idx + 1]
        seg_duration = round(end - start, 3)
        if seg_duration <= 0:
            continue
        chunks.append(
            AudioChunkSpec(
                index=len(chunks),
                start_seconds=start,
                duration_seconds=seg_duration,
            )
        )

    if not chunks:
        return [AudioChunkSpec(index=0, start_seconds=0.0, duration_seconds=total_duration)]
    return chunks


def extract_audio_chunk(
    source_mp3_path: Path,
    output_chunk_path: Path,
    *,
    chunk: AudioChunkSpec,
    ffmpeg_bin: str = "ffmpeg",
    target_bitrate: str = DEFAULT_TARGET_BITRATE,
    stage: PipelineStage = "transcription",
) -> AudioChunkFile:
    source = Path(source_mp3_path)
    output = Path(output_chunk_path)
    resolved_bitrate = str(target_bitrate or "").strip() or DEFAULT_TARGET_BITRATE
    output.parent.mkdir(parents=True, exist_ok=True)

    # Placing -ss after -i performs decode-accurate seeking so timestamp offsets stay exact.
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ss",
        f"{chunk.start_seconds:.3f}",
        "-t",
        f"{chunk.duration_seconds:.3f}",
        "-vn",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "libmp3lame",
        "-b:a",
        resolved_bitrate,
        str(output),
    ]

    try:
        _run_ffmpeg(command, output, stage)
    except FileNotFoundError as exc:
        output.unlink(missing_ok=True)
        raise StageError(
            stage=stage,
            message="Required media tool 'ffmpeg' could not be executed while splitting audio.",
            hint=FFMPEG_INSTALL_HINT,
        ) from exc
    except subprocess.CalledProcessError as exc:
        output.unlink(missing_ok=True)
        stderr = _format_stderr_snippet(exc.stderr)
        detail = f": {stderr}" if stderr else "."
        raise StageError(
            stage=stage,
            message=f"ffmpeg failed while extracting audio chunk {chunk.index + 1}{detail}",
            hint="Verify the recording is intact and retry the upload.",
        ) from exc

    if not output.is_file() or output.stat().st_size == 0:
        output.unlink(missing_ok=True)
        raise StageError(
            stage=stage,
            message=f"ffmpeg produced an empty file for audio chunk {chunk.index + 1}.",
            hint="Verify the recording contains audible sound throughout and retry.",
        )

    return AudioChunkFile(
        index=chunk.index,
        start_seconds=chunk.start_seconds,
        duration_seconds=chunk.duration_seconds,
        path=output,
        size_bytes=output.stat().st_size,
    )


def run_audio_prep_stage(context: JobContext, settings: Settings) -> StageResult:
    if not context.recording_path.is_file():
        raise StageError(
            stage="audio_prep",
            message="Uploaded recording file is missing from job storage.",
            hint="Re-upload the slide PDF and recording to start a fresh job.",
        )

    binaries = resolve_ffmpeg_binaries(settings, stage="audio_prep")
    context.report_progress("Extracting and normalizing audio to 16 kHz mono MP3...")
    normalized_path = normalize_audio_to_mp3(
        context.recording_path,
        context.audio_normalized_path,
        ffmpeg_bin=binaries.ffmpeg,
        target_bitrate=settings.transcribe_target_bitrate,
        stage="audio_prep",
    )

    context.report_progress("Measuring normalized audio duration...")
    duration_seconds = probe_audio_duration(
        normalized_path,
        ffprobe_bin=binaries.ffprobe,
        stage="audio_prep",
    )
    audio_size_bytes = normalized_path.stat().st_size
    planned_chunks = plan_audio_chunks(
        file_size_bytes=audio_size_bytes,
        duration_seconds=duration_seconds,
        safe_max_bytes=settings.transcribe_safe_upload_bytes,
        min_chunk_seconds=settings.transcribe_min_chunk_seconds,
    )

    size_mb = audio_size_bytes / (1024 * 1024)
    duration_label = _format_duration_label(duration_seconds)
    chunk_note = (
        f"; will split into {len(planned_chunks)} chunks for Groq Whisper"
        if len(planned_chunks) > 1
        else ""
    )

    return StageResult(
        message=f"Audio normalized to 16 kHz mono MP3 ({duration_label}, {size_mb:.1f} MB{chunk_note}).",
        artifacts={"audio_normalized": ARTIFACT_AUDIO_NORMALIZED},
        stats={
            "duration_seconds": round(duration_seconds, 2),
            "audio_size_bytes": audio_size_bytes,
        },
    )


__all__ = [
    "AudioChunkFile",
    "AudioChunkSpec",
    "DEFAULT_TARGET_BITRATE",
    "FFMPEG_INSTALL_HINT",
    "FFmpegBinaries",
    "TARGET_CHANNELS",
    "TARGET_SAMPLE_RATE",
    "extract_audio_chunk",
    "normalize_audio_to_mp3",
    "plan_audio_chunks",
    "probe_audio_duration",
    "resolve_ffmpeg_binaries",
    "run_audio_prep_stage",
]
