from __future__ import annotations

import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .schemas import PIPELINE_STAGES, PipelineStage

SCHEMA_VERSION = 2
INTERRUPTED_MESSAGE = "Processing was interrupted because the server restarted."
INTERRUPTED_HINT = "Start a new upload to re-run this lecture."


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

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_job_stages_order
                    ON job_stages(job_token, stage_order);
                """
            )
            self._migrate_schema(conn)

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"]) for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "metadata_json" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")
        if "warnings_json" not in columns:
            conn.execute("ALTER TABLE jobs ADD COLUMN warnings_json TEXT NOT NULL DEFAULT '[]'")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def create_job(
        self,
        *,
        slides_filename: str,
        recording_filename: str,
        job_dir: Path,
        artifacts: dict[str, str],
        token: str | None = None,
    ) -> dict[str, Any]:
        resolved_token = token or generate_job_token()
        now = utc_now_iso()
        artifacts_json = json.dumps(artifacts, sort_keys=True)

        with self._connect() as conn:
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
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, '{}', '[]', ?, ?)
                """,
                (
                    resolved_token,
                    "queued",
                    "upload",
                    slides_filename,
                    recording_filename,
                    str(job_dir),
                    artifacts_json,
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
            row = conn.execute("SELECT * FROM jobs WHERE token = ?", (token,)).fetchone()
            if row is None:
                return None

            stage_rows = conn.execute(
                """
                SELECT stage, status, message, started_at, completed_at
                FROM job_stages
                WHERE job_token = ?
                ORDER BY stage_order ASC
                """,
                (token,),
            ).fetchall()

        return self._row_to_job(row, stage_rows)

    def mark_job_started(self, token: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'processing',
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE token = ?
                """,
                (now, now, token),
            )

    def start_stage(self, token: str, stage: PipelineStage, message: str | None = None) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'processing',
                    current_stage = ?,
                    started_at = COALESCE(started_at, ?),
                    updated_at = ?
                WHERE token = ?
                """,
                (stage, now, now, token),
            )
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
                SET current_stage = ?,
                    artifacts_json = ?,
                    metadata_json = ?,
                    warnings_json = ?,
                    updated_at = ?
                WHERE token = ?
                """,
                (
                    stage,
                    json.dumps(merged_artifacts, sort_keys=True),
                    json.dumps(merged_stats, sort_keys=True),
                    json.dumps(merged_warnings),
                    now,
                    token,
                ),
            )
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
                WHERE token = ?
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
                WHERE token = ?
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
                WHERE job_token = ? AND stage = ?
                """,
                (message, now, now, token, stage),
            )

    def mark_interrupted_jobs(self) -> int:
        """Mark any queued or in-progress jobs as failed after a server restart."""
        now = utc_now_iso()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT token, current_stage
                FROM jobs
                WHERE status IN ('queued', 'processing')
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
    def _row_to_job(row: sqlite3.Row, stage_rows: list[sqlite3.Row]) -> dict[str, Any]:
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
                }
                for s in stage_rows
            ],
        }
