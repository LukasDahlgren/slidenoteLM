from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.app.config import Settings, load_settings
from backend.app.db import JobStore
from backend.app.pipeline.audio import (
    AudioChunkFile,
    AudioChunkSpec,
    FFmpegBinaries,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    extract_audio_chunk,
    normalize_audio_to_mp3,
    plan_audio_chunks,
    probe_audio_duration,
    resolve_ffmpeg_binaries,
    run_audio_prep_stage,
)
from backend.app.pipeline.transcribe import (
    compute_retry_delay_seconds,
    extract_raw_segments,
    is_auth_error,
    is_request_too_large_error,
    is_transient_transcription_error,
    merge_chunk_segments,
    run_transcription_stage,
    transcribe_audio_file,
)
from backend.app.schemas import (
    ARTIFACT_AUDIO_NORMALIZED,
    ARTIFACT_SLIDES_OCR,
    ARTIFACT_TRANSCRIPT,
    JobContext,
    SlideOCRRecord,
    SlidesOCRArtifact,
    StageError,
    StageResult,
    TranscriptArtifact,
)
from backend.app.worker import JobWorker


def _make_settings(tmp_dir: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "data_dir": tmp_dir,
        "db_path": tmp_dir / "slidenotelm.db",
        "jobs_dir": tmp_dir / "jobs",
        "pipeline_llm_mode": "stub",
        "groq_api_key": "gsk_test_key_123",
        "transcription_model": "whisper-large-v3-turbo",
        "transcribe_retry_attempts": 3,
        "transcribe_retry_base_delay_seconds": 0.01,
        "transcribe_parallel_workers": 3,
        "transcribe_global_max_concurrent": 4,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def _make_context(
    tmp_dir: Path,
    *,
    recording_filename: str = "lecture.mp4",
    progress_log: list[str] | None = None,
) -> JobContext:
    job_dir = tmp_dir / "job-1"
    input_dir = job_dir / "input"
    artifacts_dir = job_dir / "artifacts"
    input_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    slides_path = input_dir / "slides.pdf"
    slides_path.write_bytes(b"%PDF-1.7\nbody")
    ext = Path(recording_filename).suffix or ".mp4"
    recording_path = input_dir / f"recording{ext}"
    callback = progress_log.append if progress_log is not None else None
    return JobContext(
        token="tok-phase2-test",
        slides_filename="lecture.pdf",
        recording_filename=recording_filename,
        job_dir=job_dir,
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        slides_path=slides_path,
        recording_path=recording_path,
        progress_callback=callback,
    )


def _write_sine_wav(
    path: Path,
    *,
    duration_seconds: float = 4.0,
    sample_rate: int = 44100,
    channels: int = 2,
    freq_hz: float = 440.0,
) -> Path:
    frame_count = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for i in range(frame_count):
            sample = int(16000 * math.sin(2.0 * math.pi * freq_hz * (i / sample_rate)))
            packed = struct.pack("<h", sample)
            frames.extend(packed * channels)
        wav_file.writeframes(bytes(frames))
    return path


class _FakeHTTPError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code, headers=headers or {})


class _FakeGroqClient:
    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[dict[str, object]] = []
        self._lock = threading.Lock()
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        file_arg = kwargs.get("file")
        filename = ""
        payload_bytes = b""
        if isinstance(file_arg, tuple):
            filename, handle = file_arg
            payload_bytes = handle.read()
        elif hasattr(file_arg, "read"):
            filename = getattr(file_arg, "name", "")
            payload_bytes = file_arg.read()
        record = {
            "model": kwargs.get("model"),
            "response_format": kwargs.get("response_format"),
            "timestamp_granularities": kwargs.get("timestamp_granularities"),
            "filename": filename,
            "size_bytes": len(payload_bytes),
        }
        with self._lock:
            self.calls.append(record)
        return self._handler(record, payload_bytes)


class Phase2AudioUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-p2-audio-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_resolve_ffmpeg_binaries_missing_raises_actionable_stage_error(self) -> None:
        settings = _make_settings(self.tmp_dir)
        with mock.patch("backend.app.pipeline.audio.shutil.which", return_value=None):
            with self.assertRaises(StageError) as ctx:
                resolve_ffmpeg_binaries(settings, stage="audio_prep")

        self.assertEqual(ctx.exception.stage, "audio_prep")
        self.assertIn("ffmpeg", ctx.exception.message)
        self.assertIn("ffprobe", ctx.exception.message)
        self.assertIn("winget install Gyan.FFmpeg", ctx.exception.hint)
        self.assertIn("brew install ffmpeg", ctx.exception.hint)

    def test_resolve_ffmpeg_binaries_honors_explicit_paths(self) -> None:
        fake_ffmpeg = self.tmp_dir / "custom-ffmpeg.exe"
        fake_ffprobe = self.tmp_dir / "custom-ffprobe.exe"
        fake_ffmpeg.write_bytes(b"bin")
        fake_ffprobe.write_bytes(b"bin")
        settings = _make_settings(
            self.tmp_dir,
            ffmpeg_path=str(fake_ffmpeg),
            ffprobe_path=str(fake_ffprobe),
        )

        with mock.patch("backend.app.pipeline.audio.shutil.which", return_value=None):
            resolved = resolve_ffmpeg_binaries(settings)

        self.assertEqual(resolved.ffmpeg, str(fake_ffmpeg))
        self.assertEqual(resolved.ffprobe, str(fake_ffprobe))

    def test_normalize_audio_invokes_ffmpeg_16khz_mono_mp3_and_replaces_atomically(self) -> None:
        source = self.tmp_dir / "lecture.mkv"
        source.write_bytes(b"video-data")
        target = self.tmp_dir / "artifacts" / "audio_normalized.mp3"
        recorded_cmds: list[list[str]] = []

        def fake_launch(cmd, **kwargs):
            recorded_cmds.append(list(cmd))
            self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
            Path(cmd[-1]).write_bytes(b"ID3-normalized-mp3-bytes")
            process = mock.MagicMock()
            process.__enter__.return_value = process
            process.communicate.return_value = (None, b"")
            process.returncode = 0
            return process

        with mock.patch("backend.app.pipeline.audio.subprocess.Popen", side_effect=fake_launch):
            out = normalize_audio_to_mp3(
                source,
                target,
                ffmpeg_bin="/usr/bin/ffmpeg",
                target_bitrate="32k",
            )

        self.assertEqual(out, target)
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_bytes(), b"ID3-normalized-mp3-bytes")
        # Ensure no temp file remains after atomic replace
        self.assertEqual([p.name for p in target.parent.iterdir()], ["audio_normalized.mp3"])

        cmd = recorded_cmds[0]
        self.assertEqual(cmd[0], "/usr/bin/ffmpeg")
        self.assertIn("-nostdin", cmd)
        self.assertIn("-vn", cmd)
        self.assertIn("0:a:0", cmd)
        self.assertEqual(cmd[cmd.index("-ac") + 1], str(TARGET_CHANNELS))
        self.assertEqual(cmd[cmd.index("-ar") + 1], str(TARGET_SAMPLE_RATE))
        self.assertEqual(cmd[cmd.index("-b:a") + 1], "32k")

    def test_normalize_audio_detects_video_without_audio_track(self) -> None:
        source = self.tmp_dir / "silent.mp4"
        source.write_bytes(b"mp4-without-audio")
        target = self.tmp_dir / "out.mp3"

        err = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="Stream map '0:a:0' matches no streams.\n",
        )
        with mock.patch("backend.app.pipeline.audio._run_ffmpeg", side_effect=err):
            with self.assertRaises(StageError) as ctx:
                normalize_audio_to_mp3(source, target)

        self.assertEqual(ctx.exception.stage, "audio_prep")
        self.assertIn("does not contain an audio track", ctx.exception.message)
        self.assertFalse(target.exists())

    def test_stalled_encoder_is_stopped_and_partial_output_removed(self) -> None:
        source = self.tmp_dir / "recording.mp3"
        source.touch()
        target = self.tmp_dir / "output.mp3"
        partial = self.tmp_dir / ".output.tmp.mp3"
        partial.write_bytes(b"partial audio")
        with mock.patch("backend.app.pipeline.audio.subprocess.Popen") as popen:
            proc = popen.return_value.__enter__.return_value
            timeout = subprocess.TimeoutExpired("ffmpeg", 180)
            proc.communicate.side_effect = [timeout, timeout, (None, b"")]
            with self.assertRaisesRegex(StageError, "wrote no audio"):
                normalize_audio_to_mp3(source, target)
            proc.kill.assert_called_once()
        self.assertFalse(partial.exists())
        self.assertFalse(target.exists())

    def test_probe_audio_duration_falls_back_to_stream_duration_when_format_is_na(self) -> None:
        mp3_path = self.tmp_dir / "sample.mp3"
        mp3_path.write_bytes(b"mp3")
        responses = iter(
            [
                subprocess.CompletedProcess(["ffprobe"], 0, stdout="N/A\n", stderr=""),
                subprocess.CompletedProcess(["ffprobe"], 0, stdout="123.456\n", stderr=""),
            ]
        )

        with mock.patch("backend.app.pipeline.audio.subprocess.run", side_effect=lambda *a, **k: next(responses)):
            duration = probe_audio_duration(mp3_path)

        self.assertAlmostEqual(duration, 123.456, places=3)

    def test_plan_audio_chunks_single_and_balanced_multi_chunk_splits(self) -> None:
        # Fits within safe limit -> single chunk
        single = plan_audio_chunks(
            file_size_bytes=10_000_000,
            duration_seconds=2500.0,
            safe_max_bytes=21_600_000,
        )
        self.assertEqual(len(single), 1)
        self.assertEqual(single[0], AudioChunkSpec(index=0, start_seconds=0.0, duration_seconds=2500.0))

        # 3-hour lecture (~43.2 MB at 32k) with 21.6 MB safe limit -> balanced chunks, zero gaps
        multi = plan_audio_chunks(
            file_size_bytes=45_000_000,
            duration_seconds=10800.0,
            safe_max_bytes=21_600_000,
        )
        self.assertEqual(len(multi), 3)
        self.assertEqual(multi[0].start_seconds, 0.0)
        self.assertEqual(multi[-1].end_seconds, 10800.0)
        for prev, nxt in zip(multi, multi[1:]):
            self.assertEqual(prev.end_seconds, nxt.start_seconds)
            self.assertEqual(prev.duration_seconds, nxt.duration_seconds)

        # Force split (e.g. 413 fallback on small file) -> at least 2 balanced chunks
        forced = plan_audio_chunks(
            file_size_bytes=5_000_000,
            duration_seconds=600.0,
            safe_max_bytes=21_600_000,
            force_split=True,
        )
        self.assertEqual(len(forced), 2)
        self.assertEqual(forced[0], AudioChunkSpec(index=0, start_seconds=0.0, duration_seconds=300.0))
        self.assertEqual(forced[1], AudioChunkSpec(index=1, start_seconds=300.0, duration_seconds=300.0))

        # Tight safe_max_bytes smaller than min_chunk_seconds would allow -> still splits safely
        tight = plan_audio_chunks(
            file_size_bytes=100_000,
            duration_seconds=100.0,
            safe_max_bytes=30_000,
            min_chunk_seconds=300,
        )
        self.assertGreaterEqual(len(tight), 4)
        self.assertEqual(tight[-1].end_seconds, 100.0)

    def test_load_settings_parses_phase2_transcription_env_vars(self) -> None:
        env_patch = {
            "DATA_DIR": str(self.tmp_dir / "data"),
            "GROQ_API_KEY": "gsk_env_value",
            "TRANSCRIPTION_MODEL": "whisper-large-v3",
            "TRANSCRIBE_TARGET_BITRATE": "48k",
            "TRANSCRIBE_MAX_UPLOAD_BYTES": "20000000",
            "TRANSCRIBE_CHUNK_HEADROOM_PCT": "85",
            "TRANSCRIBE_MIN_CHUNK_SECONDS": "180",
            "TRANSCRIBE_PARALLEL_WORKERS": "6",
            "TRANSCRIBE_GLOBAL_MAX_CONCURRENT": "8",
            "TRANSCRIBE_RETRY_ATTEMPTS": "5",
            "TRANSCRIBE_RETRY_BASE_DELAY_SECONDS": "1.5",
            "TRANSCRIBE_TIMEOUT_SECONDS": "240",
            "FFMPEG_PATH": "/opt/bin/ffmpeg",
            "FFPROBE_PATH": "/opt/bin/ffprobe",
        }
        with mock.patch.dict(os.environ, env_patch, clear=False):
            cfg = load_settings(env_file=self.tmp_dir / "missing.env")

        self.assertEqual(cfg.groq_api_key, "gsk_env_value")
        self.assertEqual(cfg.transcription_model, "whisper-large-v3")
        self.assertEqual(cfg.transcribe_target_bitrate, "48k")
        self.assertEqual(cfg.transcribe_max_upload_bytes, 20_000_000)
        self.assertEqual(cfg.transcribe_chunk_headroom_pct, 85)
        self.assertEqual(cfg.transcribe_safe_upload_bytes, 17_000_000)
        self.assertEqual(cfg.transcribe_min_chunk_seconds, 180)
        self.assertEqual(cfg.transcribe_parallel_workers, 6)
        self.assertEqual(cfg.transcribe_global_max_concurrent, 8)
        self.assertEqual(cfg.transcribe_retry_attempts, 5)
        self.assertAlmostEqual(cfg.transcribe_retry_base_delay_seconds, 1.5)
        self.assertAlmostEqual(cfg.transcribe_timeout_seconds, 240.0)
        self.assertEqual(cfg.ffmpeg_path, "/opt/bin/ffmpeg")
        self.assertEqual(cfg.ffprobe_path, "/opt/bin/ffprobe")


class Phase2TranscriptionUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-p2-tx-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_error_classification_and_retry_backoff(self) -> None:
        self.assertTrue(is_auth_error(_FakeHTTPError("Unauthorized", status_code=401)))
        self.assertTrue(is_request_too_large_error(_FakeHTTPError("Too large", status_code=413)))
        self.assertTrue(is_transient_transcription_error(_FakeHTTPError("Rate limit", status_code=429)))
        self.assertTrue(is_transient_transcription_error(TimeoutError("timed out")))
        self.assertFalse(is_transient_transcription_error(_FakeHTTPError("Bad request", status_code=400)))

        # Exponential backoff with deterministic jitter
        d1 = compute_retry_delay_seconds(1, 2.0, jitter_fn=lambda a, b: 1.0)
        d2 = compute_retry_delay_seconds(2, 2.0, jitter_fn=lambda a, b: 1.0)
        d3 = compute_retry_delay_seconds(3, 2.0, jitter_fn=lambda a, b: 1.0)
        self.assertEqual((d1, d2, d3), (2.0, 4.0, 8.0))
        # Explicit Retry-After takes precedence
        self.assertEqual(compute_retry_delay_seconds(1, 2.0, retry_after=7.5), 7.5)

    def test_extract_and_merge_chunk_segments_shifts_timestamps_and_filters_hallucinations(self) -> None:
        # Supports object-style and dict-style segments and filters hallucinations/blanks
        raw_obj = SimpleNamespace(
            segments=[
                SimpleNamespace(start=0.0, end=2.5, text="  Welcome to lecture one. "),
                SimpleNamespace(start=2.5, end=4.0, text="   "),
                SimpleNamespace(start=4.0, end=5.0, text="Textning.nu"),
            ],
            text="Welcome to lecture one.",
        )
        extracted_0 = extract_raw_segments(raw_obj)
        self.assertEqual(extracted_0, [{"start": 0.0, "end": 2.5, "text": "Welcome to lecture one."}])

        # Fallback to top-level text if segments array is empty
        fallback_obj = SimpleNamespace(segments=[], text="Short clip summary.")
        self.assertEqual(
            extract_raw_segments(fallback_obj, fallback_duration=3.2),
            [{"start": 0.0, "end": 3.2, "text": "Short clip summary."}],
        )

        # Out-of-order chunk completion still merges in exact chronological order with shifted offsets
        chunk_0 = AudioChunkSpec(index=0, start_seconds=0.0, duration_seconds=300.0)
        chunk_1 = AudioChunkSpec(index=1, start_seconds=300.0, duration_seconds=300.0)
        chunk_2 = AudioChunkSpec(index=2, start_seconds=600.0, duration_seconds=120.0)

        merged = merge_chunk_segments(
            [
                (chunk_2, [{"start": 1.25, "end": 4.5, "text": "Final takeaway."}]),
                (chunk_0, [{"start": 0.5, "end": 3.0, "text": "Intro remark."}]),
                (
                    chunk_1,
                    [
                        {"start": 0.0, "end": 5.2, "text": "Core proof step one."},
                        {"start": 5.2, "end": 9.8, "text": "Core proof step two."},
                    ],
                ),
            ]
        )

        self.assertEqual(len(merged), 4)
        self.assertEqual([s.index for s in merged], [0, 1, 2, 3])
        self.assertEqual((merged[0].start, merged[0].end, merged[0].text), (0.5, 3.0, "Intro remark."))
        self.assertEqual((merged[1].start, merged[1].end, merged[1].text), (300.0, 305.2, "Core proof step one."))
        self.assertEqual((merged[2].start, merged[2].end, merged[2].text), (305.2, 309.8, "Core proof step two."))
        self.assertEqual((merged[3].start, merged[3].end, merged[3].text), (601.25, 604.5, "Final takeaway."))

    def test_missing_groq_api_key_fails_clearly_at_transcription_stage(self) -> None:
        settings = _make_settings(self.tmp_dir, groq_api_key="")
        mp3_path = self.tmp_dir / "audio.mp3"
        mp3_path.write_bytes(b"mp3-bytes")

        with self.assertRaises(StageError) as ctx:
            transcribe_audio_file(mp3_path, settings, source_recording="lecture.mp3")

        self.assertEqual(ctx.exception.stage, "transcription")
        self.assertIn("GROQ_API_KEY", ctx.exception.message)
        self.assertIn(".env", ctx.exception.hint)

    def test_auth_error_fails_without_retrying(self) -> None:
        settings = _make_settings(self.tmp_dir, transcribe_retry_attempts=4)
        mp3_path = self.tmp_dir / "audio.mp3"
        mp3_path.write_bytes(b"small-mp3")
        sleeps: list[float] = []

        def fail_auth(_record, _payload):
            raise _FakeHTTPError("Invalid API key", status_code=401)

        fake_client = _FakeGroqClient(fail_auth)
        with self.assertRaises(StageError) as ctx:
            transcribe_audio_file(
                mp3_path,
                settings,
                source_recording="lecture.mp3",
                client_factory=lambda _s: fake_client,
                sleep_fn=sleeps.append,
                duration_seconds=12.0,
            )

        self.assertEqual(ctx.exception.stage, "transcription")
        self.assertIn("authentication", ctx.exception.message.lower())
        self.assertEqual(len(fake_client.calls), 1)
        self.assertEqual(sleeps, [])

    def test_transient_rate_limit_retries_with_backoff_and_emits_warning(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            transcribe_retry_attempts=3,
            transcribe_retry_base_delay_seconds=1.0,
        )
        mp3_path = self.tmp_dir / "audio.mp3"
        mp3_path.write_bytes(b"small-mp3")
        sleeps: list[float] = []
        progress: list[str] = []
        attempts = {"count": 0}

        def flaky_groq(_record, _payload):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise _FakeHTTPError(
                    "Rate limit exceeded",
                    status_code=429,
                    headers={"Retry-After": "2.5"},
                )
            return SimpleNamespace(
                segments=[SimpleNamespace(start=0.0, end=4.2, text="Recovered cleanly.")]
            )

        fake_client = _FakeGroqClient(flaky_groq)
        artifact, warnings, chunk_count = transcribe_audio_file(
            mp3_path,
            settings,
            source_recording="lecture.mp3",
            progress_callback=progress.append,
            client_factory=lambda _s: fake_client,
            sleep_fn=sleeps.append,
            duration_seconds=4.2,
        )

        self.assertEqual(chunk_count, 1)
        self.assertEqual(len(fake_client.calls), 2)
        self.assertEqual(sleeps, [2.5])
        self.assertEqual(len(artifact.segments), 1)
        self.assertEqual(artifact.segments[0].text, "Recovered cleanly.")
        self.assertEqual(len(warnings), 1)
        self.assertIn("retried once", warnings[0])
        self.assertTrue(any("retrying" in msg.lower() for msg in progress))

    def test_parallel_chunked_transcription_bounds_concurrency_and_merges_offsets(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            transcribe_max_upload_bytes=1000,
            transcribe_chunk_headroom_pct=90,  # safe = 900 bytes
            transcribe_min_chunk_seconds=10,
            transcribe_parallel_workers=2,
            transcribe_global_max_concurrent=2,
        )
        mp3_path = self.tmp_dir / "long_audio.mp3"
        mp3_path.write_bytes(b"x" * 3000)  # Exceeds 900 bytes -> 4 chunks across 400s

        active = 0
        peak = 0
        state_lock = threading.Lock()
        progress: list[str] = []
        extracted_chunks: list[AudioChunkSpec] = []

        def fake_extract(_src, out_path: Path, *, chunk: AudioChunkSpec, **_kwargs) -> AudioChunkFile:
            extracted_chunks.append(chunk)
            payload = f"chunk:{chunk.index}".encode("utf-8")
            out_path.write_bytes(payload)
            return AudioChunkFile(
                index=chunk.index,
                start_seconds=chunk.start_seconds,
                duration_seconds=chunk.duration_seconds,
                path=out_path,
                size_bytes=len(payload),
            )

        def parallel_groq(_record, payload: bytes):
            nonlocal active, peak
            with state_lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.05)
                idx = int(payload.decode("utf-8").split(":")[1])
                return SimpleNamespace(
                    segments=[
                        SimpleNamespace(start=1.0, end=4.0, text=f"Segment A of chunk {idx}"),
                        SimpleNamespace(start=4.5, end=8.0, text=f"Segment B of chunk {idx}"),
                    ]
                )
            finally:
                with state_lock:
                    active -= 1

        fake_client = _FakeGroqClient(parallel_groq)
        with (
            mock.patch(
                "backend.app.pipeline.transcribe.resolve_ffmpeg_binaries",
                return_value=FFmpegBinaries(ffmpeg="ffmpeg", ffprobe="ffprobe"),
            ),
            mock.patch(
                "backend.app.pipeline.transcribe.probe_audio_duration",
                return_value=400.0,
            ),
            mock.patch(
                "backend.app.pipeline.transcribe.extract_audio_chunk",
                side_effect=fake_extract,
            ),
        ):
            artifact, warnings, chunk_count = transcribe_audio_file(
                mp3_path,
                settings,
                source_recording="long.mp4",
                progress_callback=progress.append,
                client_factory=lambda _s: fake_client,
            )

        self.assertEqual(chunk_count, 4)
        self.assertEqual(len(extracted_chunks), 4)
        self.assertEqual(warnings, [])
        self.assertEqual(artifact.duration_seconds, 400.0)
        # Verify bounded parallelism happened (>1 in flight, never exceeding worker cap of 2)
        self.assertEqual(peak, 2)

        self.assertEqual(len(artifact.segments), 8)
        self.assertEqual([s.index for s in artifact.segments], list(range(8)))
        # Chunks start at 0, 100, 200, 300
        self.assertEqual(artifact.segments[0].start, 1.0)
        self.assertEqual(artifact.segments[2].start, 101.0)
        self.assertEqual(artifact.segments[4].start, 201.0)
        self.assertEqual(artifact.segments[6].start, 301.0)
        self.assertEqual(artifact.segments[7].text, "Segment B of chunk 3")
        self.assertTrue(any("4 of 4" in msg for msg in progress))

    def test_413_on_single_upload_automatically_falls_back_to_chunked_uploads(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            transcribe_max_upload_bytes=10_000,
            transcribe_parallel_workers=2,
        )
        mp3_path = self.tmp_dir / "borderline.mp3"
        mp3_path.write_bytes(b"small-enough-locally")

        def fake_extract(_src, out_path: Path, *, chunk: AudioChunkSpec, **_kwargs) -> AudioChunkFile:
            payload = f"split:{chunk.index}".encode("utf-8")
            out_path.write_bytes(payload)
            return AudioChunkFile(
                index=chunk.index,
                start_seconds=chunk.start_seconds,
                duration_seconds=chunk.duration_seconds,
                path=out_path,
                size_bytes=len(payload),
            )

        def groq_rejects_full_accepts_chunks(_record, payload: bytes):
            if payload == b"small-enough-locally":
                raise _FakeHTTPError("Request Entity Too Large", status_code=413)
            idx = int(payload.decode("utf-8").split(":")[1])
            return SimpleNamespace(
                segments=[SimpleNamespace(start=0.5, end=2.0, text=f"Chunk {idx} text")]
            )

        fake_client = _FakeGroqClient(groq_rejects_full_accepts_chunks)
        with (
            mock.patch(
                "backend.app.pipeline.transcribe.resolve_ffmpeg_binaries",
                return_value=FFmpegBinaries(ffmpeg="ffmpeg", ffprobe="ffprobe"),
            ),
            mock.patch(
                "backend.app.pipeline.transcribe.probe_audio_duration",
                return_value=120.0,
            ),
            mock.patch(
                "backend.app.pipeline.transcribe.extract_audio_chunk",
                side_effect=fake_extract,
            ),
        ):
            artifact, warnings, chunk_count = transcribe_audio_file(
                mp3_path,
                settings,
                source_recording="borderline.m4a",
                client_factory=lambda _s: fake_client,
            )

        self.assertEqual(chunk_count, 2)
        self.assertEqual(len(artifact.segments), 2)
        self.assertEqual(artifact.segments[0].start, 0.5)
        self.assertEqual(artifact.segments[1].start, 60.5)
        self.assertTrue(any("chunked uploads" in w.lower() for w in warnings))

    def test_exhausted_retries_and_non_transient_errors_raise_stage_error(self) -> None:
        mp3_path = self.tmp_dir / "audio.mp3"
        mp3_path.write_bytes(b"small-mp3")

        # 1. Transient 503 fails across all configured attempts
        settings = _make_settings(
            self.tmp_dir,
            transcribe_retry_attempts=2,
            transcribe_retry_base_delay_seconds=0.5,
        )
        sleeps: list[float] = []
        client_503 = _FakeGroqClient(
            lambda _rec, _pay: (_ for _ in ()).throw(
                _FakeHTTPError("Service unavailable", status_code=503)
            )
        )
        with self.assertRaises(StageError) as ctx_503:
            transcribe_audio_file(
                mp3_path,
                settings,
                source_recording="lecture.mp3",
                client_factory=lambda _s: client_503,
                sleep_fn=sleeps.append,
                duration_seconds=5.0,
            )
        self.assertEqual(ctx_503.exception.stage, "transcription")
        self.assertIn("2 attempt(s)", ctx_503.exception.message)
        self.assertEqual(len(client_503.calls), 2)
        self.assertEqual(len(sleeps), 1)

        # 2. Non-transient 400 fails immediately without retrying
        sleeps.clear()
        client_400 = _FakeGroqClient(
            lambda _rec, _pay: (_ for _ in ()).throw(
                _FakeHTTPError("Invalid model", status_code=400)
            )
        )
        with self.assertRaises(StageError) as ctx_400:
            transcribe_audio_file(
                mp3_path,
                settings,
                source_recording="lecture.mp3",
                client_factory=lambda _s: client_400,
                sleep_fn=sleeps.append,
                duration_seconds=5.0,
            )
        self.assertEqual(ctx_400.exception.stage, "transcription")
        self.assertIn("Invalid model", ctx_400.exception.message)
        self.assertEqual(len(client_400.calls), 1)
        self.assertEqual(sleeps, [])

    def test_413_after_chunking_raises_actionable_stage_error(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            transcribe_max_upload_bytes=100,
            transcribe_chunk_headroom_pct=90,
            transcribe_parallel_workers=1,
        )
        mp3_path = self.tmp_dir / "huge.mp3"
        mp3_path.write_bytes(b"x" * 200)

        def fake_extract(_src, out_path: Path, *, chunk: AudioChunkSpec, **_kwargs) -> AudioChunkFile:
            out_path.write_bytes(b"still-too-big")
            return AudioChunkFile(
                index=chunk.index,
                start_seconds=chunk.start_seconds,
                duration_seconds=chunk.duration_seconds,
                path=out_path,
                size_bytes=13,
            )

        client_413 = _FakeGroqClient(
            lambda _rec, _pay: (_ for _ in ()).throw(
                _FakeHTTPError("Request Entity Too Large", status_code=413)
            )
        )
        with (
            mock.patch(
                "backend.app.pipeline.transcribe.resolve_ffmpeg_binaries",
                return_value=FFmpegBinaries(ffmpeg="ffmpeg", ffprobe="ffprobe"),
            ),
            mock.patch(
                "backend.app.pipeline.transcribe.probe_audio_duration",
                return_value=60.0,
            ),
            mock.patch(
                "backend.app.pipeline.transcribe.extract_audio_chunk",
                side_effect=fake_extract,
            ),
        ):
            with self.assertRaises(StageError) as ctx:
                transcribe_audio_file(
                    mp3_path,
                    settings,
                    source_recording="huge.mp4",
                    client_factory=lambda _s: client_413,
                )

        self.assertEqual(ctx.exception.stage, "transcription")
        self.assertIn("exceeded Groq's maximum upload size", ctx.exception.message)
        self.assertIn("TRANSCRIBE_MAX_UPLOAD_BYTES", ctx.exception.hint)

    def test_run_transcription_stage_writes_transcript_atomically(self) -> None:
        context = _make_context(self.tmp_dir, recording_filename="seminar.m4a")
        context.audio_normalized_path.write_bytes(b"normalized-mp3-data")
        settings = _make_settings(self.tmp_dir)
        fake_client = _FakeGroqClient(
            lambda _rec, _pay: SimpleNamespace(
                segments=[
                    SimpleNamespace(start=0.0, end=2.25, text="Hello class."),
                    SimpleNamespace(start=2.25, end=5.5, text="Today we cover hashing."),
                ]
            )
        )

        with (
            mock.patch(
                "backend.app.pipeline.transcribe._default_groq_client_factory",
                return_value=fake_client,
            ),
            mock.patch(
                "backend.app.pipeline.transcribe._try_probe_duration_best_effort",
                return_value=5.5,
            ),
        ):
            result = run_transcription_stage(context, settings)

        self.assertEqual(result.artifacts, {"transcript": ARTIFACT_TRANSCRIPT})
        self.assertEqual(result.stats["transcript_segments"], 2)
        self.assertEqual(result.stats["duration_seconds"], 5.5)
        self.assertEqual(result.stats["transcription_chunks"], 1)
        self.assertEqual(
            sorted(p.name for p in context.artifacts_dir.iterdir()),
            ["audio_normalized.mp3", "transcript.json"],
        )
        parsed = TranscriptArtifact.model_validate_json(
            context.transcript_path.read_text(encoding="utf-8")
        )
        self.assertEqual(len(parsed.segments), 2)
        self.assertEqual(parsed.segments[1].text, "Today we cover hashing.")


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "Real ffmpeg/ffprobe binaries are not available on PATH",
)
class Phase2RealFFmpegIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="slidenotelm-p2-ffmpeg-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _ffprobe_stream_info(self, media_path: Path) -> dict[str, object]:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_name,sample_rate,channels:format=duration",
                "-of",
                "json",
                str(media_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(proc.stdout)

    def test_real_wav_normalization_and_chunk_splitting(self) -> None:
        progress: list[str] = []
        context = _make_context(self.tmp_dir, recording_filename="lecture.wav", progress_log=progress)
        _write_sine_wav(context.recording_path, duration_seconds=4.0, sample_rate=44100, channels=2)
        settings = _make_settings(self.tmp_dir)

        result = run_audio_prep_stage(context, settings)

        self.assertEqual(result.artifacts["audio_normalized"], ARTIFACT_AUDIO_NORMALIZED)
        self.assertTrue(context.audio_normalized_path.is_file())
        self.assertGreater(context.audio_normalized_path.stat().st_size, 1000)
        self.assertAlmostEqual(float(result.stats["duration_seconds"]), 4.0, delta=0.25)

        # Verify actual MP3 encoding properties via ffprobe
        info = self._ffprobe_stream_info(context.audio_normalized_path)
        audio_stream = info["streams"][0]  # type: ignore[index]
        self.assertEqual(audio_stream["codec_name"], "mp3")
        self.assertEqual(int(audio_stream["sample_rate"]), TARGET_SAMPLE_RATE)
        self.assertEqual(int(audio_stream["channels"]), TARGET_CHANNELS)

        # Split the real normalized MP3 into 2 chunks and verify each chunk's duration
        chunks = plan_audio_chunks(
            file_size_bytes=context.audio_normalized_path.stat().st_size,
            duration_seconds=float(result.stats["duration_seconds"]),
            safe_max_bytes=10_000_000,
            force_split=True,
        )
        self.assertEqual(len(chunks), 2)
        for chunk in chunks:
            out_chunk = self.tmp_dir / f"chunk-{chunk.index}.mp3"
            extracted = extract_audio_chunk(
                context.audio_normalized_path,
                out_chunk,
                chunk=chunk,
                target_bitrate=settings.transcribe_target_bitrate,
            )
            self.assertGreater(extracted.size_bytes, 500)
            chunk_dur = probe_audio_duration(extracted.path)
            self.assertAlmostEqual(chunk_dur, chunk.duration_seconds, delta=0.2)

    def test_real_mp4_video_audio_extraction_and_video_without_audio_rejection(self) -> None:
        settings = _make_settings(self.tmp_dir)

        # 1. Generate a real 2-second MP4 with both video and audio streams
        mp4_with_audio = self.tmp_dir / "lecture_with_audio.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=160x90:d=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:v",
                "mpeg4",
                "-c:a",
                "aac",
                "-shortest",
                str(mp4_with_audio),
            ],
            check=True,
            capture_output=True,
        )

        ctx_ok = _make_context(self.tmp_dir / "with-audio", recording_filename="lecture.mp4")
        shutil.copy2(mp4_with_audio, ctx_ok.recording_path)
        ok_result = run_audio_prep_stage(ctx_ok, settings)
        self.assertTrue(ctx_ok.audio_normalized_path.is_file())
        self.assertAlmostEqual(float(ok_result.stats["duration_seconds"]), 2.0, delta=0.25)

        # 2. Generate a real 1-second video-only MP4 (no audio track)
        mp4_silent = self.tmp_dir / "video_only.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=160x90:d=1",
                "-c:v",
                "mpeg4",
                str(mp4_silent),
            ],
            check=True,
            capture_output=True,
        )

        ctx_silent = _make_context(self.tmp_dir / "no-audio", recording_filename="video_only.mp4")
        shutil.copy2(mp4_silent, ctx_silent.recording_path)
        with self.assertRaises(StageError) as err_ctx:
            run_audio_prep_stage(ctx_silent, settings)

        self.assertEqual(err_ctx.exception.stage, "audio_prep")
        self.assertIn("does not contain an audio track", err_ctx.exception.message)

    def test_worker_end_to_end_with_real_ffmpeg_and_mocked_groq_persists_transcript(self) -> None:
        settings = _make_settings(
            self.tmp_dir,
            transcribe_max_upload_bytes=10_000,  # Forces multi-chunk split on ~16 KB 4-second MP3!
            transcribe_chunk_headroom_pct=80,  # safe = 8,000 bytes -> 2+ chunks
            transcribe_min_chunk_seconds=1,
            transcribe_parallel_workers=2,
        )
        settings.ensure_directories()
        store = JobStore(settings.db_path)
        store.init_db()

        token = "worker-p2-e2e-token"
        job_dir = settings.jobs_dir / token
        (job_dir / "input").mkdir(parents=True, exist_ok=True)
        (job_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (job_dir / "input" / "slides.pdf").write_bytes(b"%PDF-1.7\nbody")
        _write_sine_wav(job_dir / "input" / "recording.wav", duration_seconds=4.0)

        store.create_job(
            slides_filename="Week 04 Graphs.pdf",
            recording_filename="lecture.wav",
            job_dir=job_dir,
            artifacts={
                "slides_pdf": "input/slides.pdf",
                "recording": "input/recording.wav",
            },
            token=token,
        )

        def fake_groq(record, _payload):
            chunk_name = str(record["filename"])
            return SimpleNamespace(
                segments=[
                    SimpleNamespace(start=0.2, end=1.4, text=f"Spoken note from {chunk_name}")
                ]
            )

        def _stub_ocr(ctx: JobContext, cfg: Settings) -> StageResult:
            ocr_artifact = SlidesOCRArtifact(
                source_pdf=ctx.slides_filename,
                model=cfg.ocr_model_google,
                slide_count=1,
                slides=[SlideOCRRecord(slide=1, visible_text="Graphs", raw_text="VISIBLE_TEXT:\nGraphs")],
            )
            ctx.slides_ocr_path.write_text(ocr_artifact.model_dump_json(indent=2), encoding="utf-8")
            return StageResult(
                message="OCR stub completed for Phase 2 worker test.",
                artifacts={"slides_ocr": ARTIFACT_SLIDES_OCR},
                stats={"slide_count": 1},
            )

        fake_client = _FakeGroqClient(fake_groq)
        worker = JobWorker(store, settings, stage_handlers={"ocr": _stub_ocr})
        try:
            with mock.patch(
                "backend.app.pipeline.transcribe._default_groq_client_factory",
                return_value=fake_client,
            ):
                worker.run_job(token)
        finally:
            worker.shutdown(wait=True)

        job = store.get_job(token)
        self.assertIsNotNone(job)
        assert job is not None
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["artifacts"]["audio_normalized"], ARTIFACT_AUDIO_NORMALIZED)
        self.assertEqual(job["artifacts"]["transcript"], ARTIFACT_TRANSCRIPT)
        self.assertGreaterEqual(int(job["stats"]["transcription_chunks"]), 2)
        self.assertGreaterEqual(int(job["stats"]["transcript_segments"]), 2)

        transcript_path = job_dir / ARTIFACT_TRANSCRIPT
        self.assertTrue(transcript_path.is_file())
        parsed = TranscriptArtifact.model_validate_json(transcript_path.read_text(encoding="utf-8"))
        self.assertEqual(parsed.source_recording, "lecture.wav")
        self.assertEqual(parsed.model, "whisper-large-v3-turbo")
        self.assertAlmostEqual(float(parsed.duration_seconds or 0.0), 4.0, delta=0.25)
        self.assertGreaterEqual(len(parsed.segments), 2)
        # Verify second chunk segment timestamp was shifted past 0.2s
        self.assertGreater(parsed.segments[-1].start, parsed.segments[0].start)


if __name__ == "__main__":
    unittest.main()
