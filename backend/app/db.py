from __future__ import annotations

import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from .schemas import (
    ARTIFACT_ALIGNMENT,
    ARTIFACT_AUDIO_NORMALIZED,
    ARTIFACT_NOTES_MD,
    ARTIFACT_SLIDES_OCR,
    ARTIFACT_SLIDES_PDF,
    ARTIFACT_TRANSCRIPT,
    PIPELINE_STAGES,
    PipelineStage,
)

SCHEMA_VERSION = 4
QUEUE_POSITION_STEP = 1000
INTERRUPTED_MESSAGE = "Processing was interrupted because the server restarted."
INTERRUPTED_HINT = "Click Retry to resume from the last completed stage, or upload again."

_STAGE_ARTIFACT_KEYS: dict[str, tuple[str, str]] = {
    "audio_prep": ("audio_normalized", ARTIFACT_AUDIO_NORMALIZED),
    "transcription": ("transcript", ARTIFACT_TRANSCRIPT),
    "ocr": ("slides_ocr", ARTIFACT_SLIDES_OCR),
    "alignment": ("alignment", ARTIFACT_ALIGNMENT),
    "notes": ("notes_md", ARTIFACT_NOTES_MD),
}

_STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "audio_prep": (),
    "ocr": (),
    "transcription": ("audio_prep",),
    "alignment": ("transcription", "ocr"),
    "notes": ("alignment",),
}


def check_job_retry_eligibility(job: dict[str, Any]) -> tuple[bool, bool, set[str]]:
    """Return (can_retry, can_retry_from_scratch, reusable_completed_stages) for a job dict."""
    if job.get("status") not in {"failed", "cancelled"}:
        return False, False, set()

    job_dir = Path(str(job["job_dir"]))
    artifacts = dict(job.get("artifacts") or {})
    slides_rel = artifacts.get("slides_pdf", ARTIFACT_SLIDES_PDF)
    recording_rel = artifacts.get("recording", "")

    has_slides = bool(slides_rel) and (job_dir / slides_rel).is_file()
    has_recording = bool(recording_rel) and (job_dir / recording_rel).is_file()

    stage_statuses = {str(s["stage"]): str(s["status"]) for s in (job.get("stages") or [])}
    reusable: set[str] = set()

    # Evaluate in topological order so downstream stages are only reusable if upstream stages are too.
    for stage in ("audio_prep", "ocr", "transcription", "alignment", "notes"):
        if stage_statuses.get(stage) != "completed":
            continue
        if any(dep not in reusable for dep in _STAGE_DEPENDENCIES[stage]):
            continue
        art_key, default_rel = _STAGE_ARTIFACT_KEYS[stage]
        rel = artifacts.get(art_key) or default_rel
        art_path = job_dir / rel
        if art_path.is_file() and art_path.stat().st_size > 0:
            reusable.add(stage)

    can_retry_from_scratch = has_slides and has_recording
    can_retry = (("audio_prep" in reusable) or has_recording) and (
        ("ocr" in reusable) or has_slides
    )
    return can_retry, can_retry_from_scratch, reusable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def generate_job_token() -> str:
    """Return a high-entropy, URL-safe private job token (~192 bits)."""
    return secrets.token_urlsafe(24)


class JobStore:
    """SQLite-backed persistence for jobs, per-stage history, errors, artifacts, and stats."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    token TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    slides_filename TEXT NOT NULL,
                    recording_filename TEXT NOT NULL,
                    job_dir TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    error_stage TEXT,
                    error_message TEXT,
                    error_hint TEXT,
                    queue_position INTEGER,
                    batch_id TEXT,
                    queue_label TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS job_stages (
                    job_token TEXT NOT NULL REFERENCES jobs(token) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    stage_order INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    PRIMARY KEY (job_token, stage)
                );

                CREATE TABLE IF NOT EXISTS queue_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    paused INTEGER NOT NULL DEFAULT 0,
                    pause_reason TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_job_stages_order
                    ON job_stages(job_token, stage_order);
                """
            )
            self._migrate_schema(conn)

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        stage_columns = {row["name"] for row in conn.execute("PRAGMA table_info(job_stages)")}
        if "work_fraction" not in stage_columns:
            conn.execute("ALTER TABLE job_stages ADD COLUMN work_fraction REAL")
        columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "metadata_json" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")
        if "warnings_json" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN warnings_json TEXT NOT NULL DEFAULT '[]'")
        if "queue_position" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN queue_position INTEGER")
        if "batch_id" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN batch_id TEXT")
        if "queue_label" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN queue_label TEXT")

        conn.execute(
            f"UPDATE jobs SET queue_position = rowid * {QUEUE_POSITION_STEP} "
            "WHERE queue_position IS NULL"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queue_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                paused INTEGER NOT NULL DEFAULT 0,
                pause_reason TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO queue_state (id, paused, pause_reason, updated_at)
            VALUES (1, 0, NULL, ?)
            """,
            (utc_now_iso(),),
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_queue_order "
            "ON jobs(status, queue_position ASC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_created_at "
            "ON jobs(created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_batch_id ON jobs(batch_id)"
        )
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def create_job(
        self,
        *,
        slides_filename: str,
        recording_filename: str,
        job_dir: Path,
        artifacts: dict[str, str],
        token: str | None = None,
        batch_id: str | None = None,
        queue_label: str | None = None,
    ) -> dict[str, Any]:
        resolved_token = token or generate_job_token()
        now = utc_now_iso()
        artifacts_json = json.dumps(artifacts, sort_keys=True)
        cleaned_batch = (batch_id or "").strip() or None
        cleaned_label = (queue_label or "").strip() or None

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            max_row = conn.execute(
                "SELECT COALESCE(MAX(queue_position), 0) AS max_pos FROM jobs"
            ).fetchone()
            next_pos = int(max_row["max_pos"]) + QUEUE_POSITION_STEP

            conn.execute(
                """
                INSERT INTO jobs (
                    token,
                    status,
                    current_stage,
                    slides_filename,
                    recording_filename,
                    job_dir,
                    artifacts_json,
                    metadata_json,
                    warnings_json,
                    queue_position,
                    batch_id,
                    queue_label,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, '{}', '[]', ?, ?, ?, ?, ?)
                """,
                (
                    resolved_token,
                    "queued",
                    "upload",
                    slides_filename,
                    recording_filename,
                    str(job_dir),
                    artifacts_json,
                    next_pos,
                    cleaned_batch,
                    cleaned_label,
                    now,
                    now,
                ),
            )

            for order, stage in enumerate(PIPELINE_STAGES):
                if stage == "upload":
                    stage_status = "completed"
                    stage_message = "Files uploaded and validated."
                    stage_started = now
                    stage_completed = now
                else:
                    stage_status = "pending"
                    stage_message = None
                    stage_started = None
                    stage_completed = None

                conn.execute(
                    """
                    INSERT INTO job_stages (
                        job_token,
                        stage,
                        stage_order,
                        status,
                        message,
                        started_at,
                        completed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_token,
                        stage,
                        order,
                        stage_status,
                        stage_message,
                        stage_started,
                        stage_completed,
                    ),
                )

        job = self.get_job(resolved_token)
        if job is None:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Failed to reload newly created job {resolved_token}")
        return job

    def get_job(self, token: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute("BEGIN")  # Read the job label and its stages from the same snapshot.
            return self._get_job_in_conn(conn, token)

    def _get_job_in_conn(self, conn: sqlite3.Connection, token: str) -> dict[str, Any] | None:
        row = conn.execute("SELECT rowid, * FROM jobs WHERE token = ?", (token,)).fetchone()
        if row is None:
            return None

        stage_rows = conn.execute(
            """
            SELECT stage, status, message, started_at, completed_at, work_fraction
            FROM job_stages
            WHERE job_token = ?
            ORDER BY stage_order ASC
            """,
            (token,),
        ).fetchall()

        queue_index: int | None = None
        if row["status"] == "queued":
            idx_row = conn.execute(
                """
                SELECT COUNT(*) AS ahead
                FROM jobs
                WHERE status = 'queued'
                  AND (
                      COALESCE(queue_position, rowid * ?) < COALESCE(?, ? * ?)
                      OR (
                          COALESCE(queue_position, rowid * ?) = COALESCE(?, ? * ?)
                          AND rowid < ?
                      )
                  )
                """,
                (
                    QUEUE_POSITION_STEP,
                    row["queue_position"],
                    row["rowid"],
                    QUEUE_POSITION_STEP,
                    QUEUE_POSITION_STEP,
                    row["queue_position"],
                    row["rowid"],
                    QUEUE_POSITION_STEP,
                    row["rowid"],
                ),
            ).fetchone()
            queue_index = int(idx_row["ahead"]) + 1 if idx_row is not None else 1

        return self._row_to_job(row, stage_rows, queue_index=queue_index)

    def mark_job_started(self, token: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'processing',
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE token = ? AND status IN ('queued', 'processing')
                """,
                (now, now, token),
            )

    def start_stage(self, token: str, stage: PipelineStage, message: str | None = None) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE job_stages
                SET status = 'running',
                    message = COALESCE(?, message),
                    started_at = COALESCE(started_at, ?),
                    completed_at = NULL
                WHERE job_token = ? AND stage = ?
                """,
                (message, now, token, stage),
            )
            conn.execute(
                """
                UPDATE jobs
                SET status = 'processing',
                    current_stage = ?,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE token = ?
                """,
                (self._current_stage(conn, token, stage), now, now, token),
            )

    @staticmethod
    def _current_stage(conn: sqlite3.Connection, token: str, fallback: PipelineStage) -> str:
        """Use the earliest running stage for clients that display a single label."""
        row = conn.execute(
            """SELECT stage FROM job_stages WHERE job_token = ? AND status = 'running'
               ORDER BY stage_order LIMIT 1""",
            (token,),
        ).fetchone()
        return row["stage"] if row else fallback

    def update_stage_fraction(self, token: str, stage: PipelineStage, fraction: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE job_stages SET work_fraction = ?
                   WHERE job_token = ? AND stage = ? AND status = 'running'""",
                (fraction, token, stage),
            )

    def update_stage_message(self, token: str, stage: PipelineStage, message: str) -> None:
        """Update the live status message for a running stage."""
        cleaned = message.strip()
        if not cleaned:
            return
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE job_stages
                SET message = ?
                WHERE job_token = ? AND stage = ?
                """,
                (cleaned, token, stage),
            )
            conn.execute(
                "UPDATE jobs SET updated_at = ? WHERE token = ?",
                (now, token),
            )

    def complete_stage(
        self,
        token: str,
        stage: PipelineStage,
        *,
        message: str | None = None,
        artifacts: dict[str, str] | None = None,
        stats: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE job_stages
                SET status = 'completed',
                    message = COALESCE(?, message),
                    started_at = COALESCE(started_at, ?),
                    completed_at = ?
                WHERE job_token = ? AND stage = ?
                """,
                (message, now, now, token, stage),
            )
            merged_artifacts = self._load_json_dict_locked(conn, token, "artifacts_json")
            if artifacts:
                merged_artifacts.update(artifacts)

            merged_stats = self._load_json_dict_locked(conn, token, "metadata_json")
            if stats:
                merged_stats.update(stats)

            merged_warnings = self._merge_warnings(
                self._load_json_list_locked(conn, token, "warnings_json"),
                warnings,
            )

            conn.execute(
                """
                UPDATE jobs
                SET current_stage = COALESCE(error_stage, ?),
                    artifacts_json = ?,
                    metadata_json = ?,
                    warnings_json = ?,
                    updated_at = ?
                WHERE token = ?
                """,
                (
                    self._current_stage(conn, token, stage),
                    json.dumps(merged_artifacts, sort_keys=True),
                    json.dumps(merged_stats, sort_keys=True),
                    json.dumps(merged_warnings),
                    now,
                    token,
                ),
            )

    def mark_job_completed(
        self,
        token: str,
        *,
        artifacts: dict[str, str] | None = None,
        stats: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            merged_artifacts = self._load_json_dict_locked(conn, token, "artifacts_json")
            if artifacts:
                merged_artifacts.update(artifacts)

            merged_stats = self._load_json_dict_locked(conn, token, "metadata_json")
            if stats:
                merged_stats.update(stats)

            merged_warnings = self._merge_warnings(
                self._load_json_list_locked(conn, token, "warnings_json"),
                warnings,
            )

            conn.execute(
                """
                UPDATE jobs
                SET status = 'completed',
                    current_stage = 'notes',
                    artifacts_json = ?,
                    metadata_json = ?,
                    warnings_json = ?,
                    error_stage = NULL,
                    error_message = NULL,
                    error_hint = NULL,
                    updated_at = ?,
                    completed_at = ?
                WHERE token = ? AND status NOT IN ('failed', 'cancelled')
                """,
                (
                    json.dumps(merged_artifacts, sort_keys=True),
                    json.dumps(merged_stats, sort_keys=True),
                    json.dumps(merged_warnings),
                    now,
                    now,
                    token,
                ),
            )

    def mark_job_failed(
        self,
        token: str,
        *,
        stage: PipelineStage,
        message: str,
        hint: str,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'failed',
                    current_stage = ?,
                    error_stage = ?,
                    error_message = ?,
                    error_hint = ?,
                    updated_at = ?,
                    completed_at = ?
                WHERE token = ? AND status NOT IN ('failed', 'cancelled')
                """,
                (stage, stage, message, hint, now, now, token),
            )
            conn.execute(
                """
                UPDATE job_stages
                SET status = 'failed',
                    message = ?,
                    started_at = COALESCE(started_at, ?),
                    completed_at = ?
                WHERE job_token = ? AND stage = ? AND status != 'cancelled'
                """,
                (message, now, now, token, stage),
            )

    def queued_job_tokens(self) -> list[str]:
        with self._connect() as conn:
            return [
                str(row["token"])
                for row in conn.execute(
                    """
                    SELECT token
                    FROM jobs
                    WHERE status = 'queued'
                    ORDER BY COALESCE(queue_position, rowid * ?) ASC, rowid ASC
                    """,
                    (QUEUE_POSITION_STEP,),
                )
            ]

    def get_queue_state(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT paused, pause_reason, updated_at FROM queue_state WHERE id = 1"
            ).fetchone()
            if row is None:
                return {"paused": False, "pause_reason": None, "updated_at": utc_now_iso()}
            return {
                "paused": bool(row["paused"]),
                "pause_reason": row["pause_reason"],
                "updated_at": str(row["updated_at"]),
            }

    def is_queue_paused(self) -> bool:
        return bool(self.get_queue_state()["paused"])

    def set_queue_paused(self, paused: bool, *, reason: str | None = None) -> dict[str, Any]:
        now = utc_now_iso()
        cleaned_reason = (reason or "").strip() or None if paused else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO queue_state (id, paused, pause_reason, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    paused = excluded.paused,
                    pause_reason = excluded.pause_reason,
                    updated_at = excluded.updated_at
                """,
                (1 if paused else 0, cleaned_reason, now),
            )
        return {"paused": bool(paused), "pause_reason": cleaned_reason, "updated_at": now}

    def count_active_and_queued_jobs(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM jobs WHERE status IN ('queued', 'processing')"
            ).fetchone()
            return int(row["cnt"]) if row is not None else 0

    def peek_next_queued_job(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            conn.execute("BEGIN")
            row = conn.execute(
                """
                SELECT token
                FROM jobs
                WHERE status = 'queued'
                ORDER BY COALESCE(queue_position, rowid * ?) ASC, rowid ASC
                LIMIT 1
                """,
                (QUEUE_POSITION_STEP,),
            ).fetchone()
            if row is None:
                return None
            return self._get_job_in_conn(conn, str(row["token"]))

    def claim_next_queued_job(self, *, allow_when_paused: bool = False) -> dict[str, Any] | None:
        """Atomically claim the head queued job for processing, respecting queue pause state."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if not allow_when_paused:
                state_row = conn.execute(
                    "SELECT paused FROM queue_state WHERE id = 1"
                ).fetchone()
                if state_row is not None and bool(state_row["paused"]):
                    return None

            row = conn.execute(
                """
                SELECT token
                FROM jobs
                WHERE status = 'queued'
                ORDER BY COALESCE(queue_position, rowid * ?) ASC, rowid ASC
                LIMIT 1
                """,
                (QUEUE_POSITION_STEP,),
            ).fetchone()
            if row is None:
                return None

            token = str(row["token"])
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = 'processing',
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE token = ? AND status = 'queued'
                """,
                (now, now, token),
            )
            if cursor.rowcount != 1:
                return None
            return self._get_job_in_conn(conn, token)

    def reorder_queued_jobs(self, ordered_tokens: list[str]) -> list[str]:
        """Atomically reorder waiting queued jobs; ignores tokens no longer queued."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing_rows = conn.execute(
                """
                SELECT token
                FROM jobs
                WHERE status = 'queued'
                ORDER BY COALESCE(queue_position, rowid * ?) ASC, rowid ASC
                """,
                (QUEUE_POSITION_STEP,),
            ).fetchall()
            existing_order = [str(r["token"]) for r in existing_rows]
            existing_set = set(existing_order)

            seen: set[str] = set()
            final_order: list[str] = []
            for tok in ordered_tokens:
                cleaned = str(tok).strip()
                if cleaned in existing_set and cleaned not in seen:
                    seen.add(cleaned)
                    final_order.append(cleaned)
            for tok in existing_order:
                if tok not in seen:
                    seen.add(tok)
                    final_order.append(tok)

            max_non_queued_row = conn.execute(
                """
                SELECT COALESCE(MAX(queue_position), 0) AS max_pos
                FROM jobs
                WHERE status != 'queued'
                """
            ).fetchone()
            base_pos = int(max_non_queued_row["max_pos"]) if max_non_queued_row is not None else 0

            for idx, tok in enumerate(final_order, start=1):
                new_pos = base_pos + (idx * QUEUE_POSITION_STEP)
                conn.execute(
                    "UPDATE jobs SET queue_position = ?, updated_at = ? WHERE token = ? AND status = 'queued'",
                    (new_pos, now, tok),
                )

            return final_order

    def cancel_queued_job(self, token: str, *, message: str = "Cancelled before processing.") -> bool:
        """Atomically cancel a job iff it is still in 'queued' status."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = 'cancelled',
                    error_stage = NULL,
                    error_message = NULL,
                    error_hint = NULL,
                    updated_at = ?,
                    completed_at = ?
                WHERE token = ? AND status = 'queued'
                """,
                (now, now, token),
            )
            if cursor.rowcount != 1:
                return False
            conn.execute(
                """
                UPDATE job_stages
                SET status = 'cancelled',
                    message = COALESCE(message, ?),
                    completed_at = COALESCE(completed_at, ?)
                WHERE job_token = ? AND status = 'pending'
                """,
                (message, now, token),
            )
            return True

    def mark_job_cancelled(self, token: str, *, message: str = "Cancelled by user.") -> bool:
        """Mark a queued or processing job as cancelled."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = 'cancelled',
                    error_stage = NULL,
                    error_message = NULL,
                    error_hint = NULL,
                    updated_at = ?,
                    completed_at = ?
                WHERE token = ? AND status IN ('queued', 'processing')
                """,
                (now, now, token),
            )
            if cursor.rowcount != 1:
                return False
            conn.execute(
                """
                UPDATE job_stages
                SET status = 'cancelled',
                    message = ?,
                    completed_at = ?
                WHERE job_token = ? AND status = 'running'
                """,
                (message, now, token),
            )
            return True

    def retry_job(
        self,
        token: str,
        *,
        from_scratch: bool = False,
        position: Literal["end", "front"] = "end",
    ) -> dict[str, Any]:
        """Re-enqueue a failed/cancelled job, reusing valid completed stage checkpoints by default."""
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            job = self._get_job_in_conn(conn, token)
            if job is None:
                raise KeyError(f"Job not found: {token}")
            if job["status"] not in {"failed", "cancelled"}:
                raise ValueError(
                    f"Only failed or cancelled jobs can be retried (current status: {job['status']})."
                )

            can_retry, can_retry_from_scratch, reusable = check_job_retry_eligibility(job)
            if from_scratch:
                if not can_retry_from_scratch:
                    raise ValueError(
                        "Cannot retry from scratch because the original uploaded files are no longer on disk."
                    )
                keep_completed: set[str] = {"upload"}
            else:
                if not can_retry:
                    raise ValueError(
                        "Required input files are missing from disk; please upload this lecture again."
                    )
                keep_completed = {"upload", *reusable}

            last_completed: PipelineStage = "upload"
            for stage in PIPELINE_STAGES:
                if stage in keep_completed:
                    last_completed = stage
                else:
                    conn.execute(
                        """
                        UPDATE job_stages
                        SET status = 'pending',
                            work_fraction = NULL,
                            message = NULL,
                            started_at = NULL,
                            completed_at = NULL
                        WHERE job_token = ? AND stage = ?
                        """,
                        (token, stage),
                    )

            if position == "front":
                min_row = conn.execute(
                    """
                    SELECT MIN(COALESCE(queue_position, rowid * ?)) AS min_pos
                    FROM jobs
                    WHERE status = 'queued'
                    """,
                    (QUEUE_POSITION_STEP,),
                ).fetchone()
                if min_row is not None and min_row["min_pos"] is not None:
                    new_pos = int(min_row["min_pos"]) - QUEUE_POSITION_STEP
                else:
                    max_row = conn.execute(
                        "SELECT COALESCE(MAX(queue_position), 0) AS max_pos FROM jobs"
                    ).fetchone()
                    new_pos = int(max_row["max_pos"]) + QUEUE_POSITION_STEP
            else:
                max_row = conn.execute(
                    "SELECT COALESCE(MAX(queue_position), 0) AS max_pos FROM jobs"
                ).fetchone()
                new_pos = int(max_row["max_pos"]) + QUEUE_POSITION_STEP

            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued',
                    current_stage = ?,
                    queue_position = ?,
                    error_stage = NULL,
                    error_message = NULL,
                    error_hint = NULL,
                    completed_at = NULL,
                    updated_at = ?
                WHERE token = ?
                """,
                (last_completed, new_pos, now, token),
            )
            reloaded = self._get_job_in_conn(conn, token)
            if reloaded is None:  # pragma: no cover - defensive guard
                raise RuntimeError(f"Failed to reload retried job {token}")
            return reloaded

    def delete_job(self, token: str) -> Path | None:
        """Delete a terminal job from SQLite and return its job_dir for filesystem cleanup."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, job_dir FROM jobs WHERE token = ?",
                (token,),
            ).fetchone()
            if row is None:
                return None
            if row["status"] in {"queued", "processing"}:
                raise ValueError(
                    f"Cannot delete job while status is '{row['status']}'. Cancel it first."
                )
            job_dir = Path(str(row["job_dir"]))
            conn.execute("DELETE FROM jobs WHERE token = ?", (token,))
            return job_dir

    def get_queue_snapshot(
        self,
        *,
        recent_limit: int = 50,
        batch_id: str | None = None,
    ) -> dict[str, Any]:
        """Read queue pause state, counts, active jobs (with stages), queued jobs, and recent jobs."""
        resolved_limit = max(0, min(200, int(recent_limit)))
        cleaned_batch = (batch_id or "").strip() or None

        with self._connect() as conn:
            conn.execute("BEGIN")
            state_row = conn.execute(
                "SELECT paused, pause_reason, updated_at FROM queue_state WHERE id = 1"
            ).fetchone()
            queue_state = {
                "paused": bool(state_row["paused"]) if state_row is not None else False,
                "pause_reason": state_row["pause_reason"] if state_row is not None else None,
                "updated_at": str(state_row["updated_at"]) if state_row is not None else utc_now_iso(),
            }

            counts = {
                "processing": 0,
                "queued": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
            }
            if cleaned_batch:
                count_rows = conn.execute(
                    "SELECT status, COUNT(*) AS cnt FROM jobs WHERE batch_id = ? GROUP BY status",
                    (cleaned_batch,),
                ).fetchall()
            else:
                count_rows = conn.execute(
                    "SELECT status, COUNT(*) AS cnt FROM jobs GROUP BY status"
                ).fetchall()
            for r in count_rows:
                st = str(r["status"])
                if st in counts:
                    counts[st] = int(r["cnt"])

            batch_clause = "AND batch_id = ?" if cleaned_batch else ""
            batch_params: tuple[Any, ...] = (cleaned_batch,) if cleaned_batch else ()

            active_rows = conn.execute(
                f"""
                SELECT rowid, *
                FROM jobs
                WHERE status = 'processing' {batch_clause}
                ORDER BY COALESCE(queue_position, rowid * ?) ASC, rowid ASC
                """,
                (*batch_params, QUEUE_POSITION_STEP),
            ).fetchall()

            queued_rows = conn.execute(
                f"""
                SELECT rowid, *
                FROM jobs
                WHERE status = 'queued' {batch_clause}
                ORDER BY COALESCE(queue_position, rowid * ?) ASC, rowid ASC
                """,
                (*batch_params, QUEUE_POSITION_STEP),
            ).fetchall()

            recent_rows: list[sqlite3.Row] = []
            if resolved_limit > 0:
                recent_rows = conn.execute(
                    f"""
                    SELECT rowid, *
                    FROM jobs
                    WHERE status IN ('completed', 'failed', 'cancelled') {batch_clause}
                    ORDER BY COALESCE(completed_at, updated_at) DESC, rowid DESC
                    LIMIT ?
                    """,
                    (*batch_params, resolved_limit),
                ).fetchall()

            all_tokens = [
                str(r["token"]) for r in (*active_rows, *queued_rows, *recent_rows)
            ]
            stages_by_token: dict[str, list[sqlite3.Row]] = {t: [] for t in all_tokens}
            if all_tokens:
                placeholders = ",".join("?" for _ in all_tokens)
                stage_rows = conn.execute(
                    f"""
                    SELECT job_token, stage, status, message, started_at, completed_at, work_fraction
                    FROM job_stages
                    WHERE job_token IN ({placeholders})
                    ORDER BY job_token, stage_order ASC
                    """,
                    tuple(all_tokens),
                ).fetchall()
                for srow in stage_rows:
                    stages_by_token[str(srow["job_token"])].append(srow)

            active_jobs = [
                self._row_to_job(r, stages_by_token[str(r["token"])])
                for r in active_rows
            ]
            queued_jobs = [
                self._row_to_job(r, stages_by_token[str(r["token"])], queue_index=idx)
                for idx, r in enumerate(queued_rows, start=1)
            ]
            recent_jobs = [
                self._row_to_job(r, stages_by_token[str(r["token"])])
                for r in recent_rows
            ]

        return {
            "queue_state": queue_state,
            "counts": counts,
            "active": active_jobs,
            "queued": queued_jobs,
            "recent": recent_jobs,
        }

    def mark_interrupted_jobs(self) -> int:
        """Mark in-progress jobs as failed after a server restart."""
        now = utc_now_iso()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT token, current_stage
                FROM jobs
                WHERE status = 'processing'
                """
            ).fetchall()

            for row in rows:
                token = str(row["token"])
                running_stage_row = conn.execute(
                    """
                    SELECT stage
                    FROM job_stages
                    WHERE job_token = ? AND status = 'running'
                    ORDER BY stage_order ASC
                    LIMIT 1
                    """,
                    (token,),
                ).fetchone()

                if running_stage_row is not None:
                    failed_stage = str(running_stage_row["stage"])
                else:
                    next_pending_row = conn.execute(
                        """
                        SELECT stage
                        FROM job_stages
                        WHERE job_token = ? AND status = 'pending'
                        ORDER BY stage_order ASC
                        LIMIT 1
                        """,
                        (token,),
                    ).fetchone()
                    failed_stage = (
                        str(next_pending_row["stage"])
                        if next_pending_row is not None
                        else str(row["current_stage"])
                    )

                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed',
                        current_stage = ?,
                        error_stage = ?,
                        error_message = ?,
                        error_hint = ?,
                        updated_at = ?,
                        completed_at = ?
                    WHERE token = ?
                    """,
                    (
                        failed_stage,
                        failed_stage,
                        INTERRUPTED_MESSAGE,
                        INTERRUPTED_HINT,
                        now,
                        now,
                        token,
                    ),
                )
                conn.execute(
                    """
                    UPDATE job_stages
                    SET status = 'failed',
                        message = ?,
                        started_at = COALESCE(started_at, ?),
                        completed_at = ?
                    WHERE job_token = ? AND stage = ?
                    """,
                    (INTERRUPTED_MESSAGE, now, now, token, failed_stage),
                )

            # A failed job may also have had a sibling stage in flight at shutdown.
            conn.execute(
                """UPDATE job_stages SET status = 'failed', message = ?, completed_at = ?
                   WHERE status = 'running' AND job_token IN
                       (SELECT token FROM jobs WHERE status = 'failed')""",
                (INTERRUPTED_MESSAGE, now),
            )

        return len(rows)

    @staticmethod
    def _load_json_dict_locked(
        conn: sqlite3.Connection,
        token: str,
        column: str,
    ) -> dict[str, Any]:
        row = conn.execute(f"SELECT {column} FROM jobs WHERE token = ?", (token,)).fetchone()
        if row is None or not row[column]:
            return {}
        try:
            loaded = json.loads(row[column])
        except json.JSONDecodeError:
            return {}
        return dict(loaded) if isinstance(loaded, dict) else {}

    @staticmethod
    def _load_json_list_locked(
        conn: sqlite3.Connection,
        token: str,
        column: str,
    ) -> list[str]:
        row = conn.execute(f"SELECT {column} FROM jobs WHERE token = ?", (token,)).fetchone()
        if row is None or not row[column]:
            return []
        try:
            loaded = json.loads(row[column])
        except json.JSONDecodeError:
            return []
        if not isinstance(loaded, list):
            return []
        return [str(item) for item in loaded if str(item).strip()]

    @staticmethod
    def _merge_warnings(existing: list[str], incoming: list[str] | None) -> list[str]:
        merged = list(existing)
        if not incoming:
            return merged
        for warning in incoming:
            cleaned = str(warning).strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)
        return merged

    @staticmethod
    def _row_to_job(
        row: sqlite3.Row,
        stage_rows: list[sqlite3.Row],
        *,
        queue_index: int | None = None,
    ) -> dict[str, Any]:
        def _parse_dict(raw: Any) -> dict[str, Any]:
            try:
                loaded = json.loads(raw or "{}")
            except json.JSONDecodeError:
                return {}
            return loaded if isinstance(loaded, dict) else {}

        def _parse_list(raw: Any) -> list[str]:
            try:
                loaded = json.loads(raw or "[]")
            except json.JSONDecodeError:
                return []
            if not isinstance(loaded, list):
                return []
            return [str(item) for item in loaded if str(item).strip()]

        keys = set(row.keys())
        return {
            "token": row["token"],
            "status": row["status"],
            "current_stage": row["current_stage"],
            "slides_filename": row["slides_filename"],
            "recording_filename": row["recording_filename"],
            "job_dir": row["job_dir"],
            "artifacts": _parse_dict(row["artifacts_json"]),
            "stats": _parse_dict(row["metadata_json"]),
            "warnings": _parse_list(row["warnings_json"]),
            "error_stage": row["error_stage"],
            "error_message": row["error_message"],
            "error_hint": row["error_hint"],
            "queue_position": row["queue_position"] if "queue_position" in keys else None,
            "queue_index": queue_index,
            "batch_id": row["batch_id"] if "batch_id" in keys else None,
            "queue_label": row["queue_label"] if "queue_label" in keys else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "stages": [
                {
                    "stage": s["stage"],
                    "status": s["status"],
                    "message": s["message"],
                    "started_at": s["started_at"],
                    "completed_at": s["completed_at"],
                    "work_fraction": s["work_fraction"],
                }
                for s in stage_rows
            ],
        }
