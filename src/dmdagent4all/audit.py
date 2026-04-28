from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  event_type TEXT NOT NULL,
  user_request TEXT,
  tool TEXT,
  risk INTEGER,
  approved INTEGER,
  model TEXT,
  result_status TEXT,
  metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  tool TEXT NOT NULL,
  risk INTEGER,
  args_json TEXT NOT NULL,
  decision TEXT NOT NULL,
  result_status TEXT
);

CREATE TABLE IF NOT EXISTS approvals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  tool TEXT NOT NULL,
  risk INTEGER,
  status TEXT NOT NULL,
  reason TEXT,
  expires_at TEXT,
  args_json TEXT NOT NULL DEFAULT '{}',
  request_reason TEXT,
  decision_reason TEXT
);

CREATE TABLE IF NOT EXISTS connectors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  name TEXT NOT NULL UNIQUE,
  enabled INTEGER NOT NULL,
  status TEXT NOT NULL,
  metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  local INTEGER NOT NULL,
  metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  path TEXT NOT NULL,
  action TEXT NOT NULL,
  source TEXT,
  metadata_json TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    user_request: str | None = None
    tool: str | None = None
    risk: int | None = None
    approved: bool | None = None
    model: str | None = None
    result_status: str | None = None
    metadata: dict[str, Any] | None = None


class AuditStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def record_event(self, event: AuditEvent) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_logs (
                  created_at, event_type, user_request, tool, risk, approved,
                  model, result_status, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    event.event_type,
                    event.user_request,
                    event.tool,
                    event.risk,
                    None if event.approved is None else int(event.approved),
                    event.model,
                    event.result_status,
                    json.dumps(event.metadata or {}, sort_keys=True),
                ),
            )

    def record_tool_call(
        self,
        *,
        tool: str,
        risk: int | None,
        args: dict[str, Any],
        decision: str,
        result_status: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tool_calls (
                  created_at, tool, risk, args_json, decision, result_status
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    tool,
                    risk,
                    json.dumps(args, sort_keys=True),
                    decision,
                    result_status,
                ),
            )

    def record_approval(
        self,
        *,
        tool: str,
        risk: int | None,
        args: dict[str, Any],
        request_reason: str,
        decision_reason: str,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO approvals (
                  created_at, tool, risk, status, reason, args_json,
                  request_reason, decision_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    tool,
                    risk,
                    "pending",
                    decision_reason,
                    json.dumps(args, sort_keys=True),
                    request_reason,
                    decision_reason,
                ),
            )
            return int(cursor.lastrowid)

    def list_approvals(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        where = ""
        params: tuple[Any, ...] = (limit,)
        if status is not None:
            where = "WHERE status = ?"
            params = (status, limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, created_at, tool, risk, status, reason, expires_at,
                       args_json, request_reason, decision_reason
                FROM approvals
                {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            {
                "id": row[0],
                "created_at": row[1],
                "tool": row[2],
                "risk": row[3],
                "status": row[4],
                "reason": row[5],
                "expires_at": row[6],
                "args": json.loads(row[7] or "{}"),
                "request_reason": row[8],
                "decision_reason": row[9],
            }
            for row in rows
        ]

    def get_approval(self, approval_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, created_at, tool, risk, status, reason, expires_at,
                       args_json, request_reason, decision_reason
                FROM approvals
                WHERE id = ?
                """,
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "created_at": row[1],
            "tool": row[2],
            "risk": row[3],
            "status": row[4],
            "reason": row[5],
            "expires_at": row[6],
            "args": json.loads(row[7] or "{}"),
            "request_reason": row[8],
            "decision_reason": row[9],
        }

    def set_approval_status(self, approval_id: int, status: str) -> bool:
        if status not in {"approved", "denied", "cancelled", "executed", "failed"}:
            raise ValueError(
                "Approval status must be approved, denied, cancelled, executed, or failed."
            )
        with self._connect() as connection:
            status_filter = "('pending', 'approved')" if status in {"executed", "failed"} else "('pending')"
            cursor = connection.execute(
                f"""
                UPDATE approvals
                SET status = ?
                WHERE id = ? AND status IN {status_filter}
                """,
                (status, approval_id),
            )
            return cursor.rowcount > 0

    def list_recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, event_type, user_request, tool, risk, approved,
                       model, result_status, metadata_json
                FROM audit_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "created_at": row[0],
                "event_type": row[1],
                "user_request": row[2],
                "tool": row[3],
                "risk": row[4],
                "approved": None if row[5] is None else bool(row[5]),
                "model": row[6],
                "result_status": row[7],
                "metadata": json.loads(row[8]),
            }
            for row in rows
        ]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            _ensure_column(connection, "approvals", "args_json", "TEXT NOT NULL DEFAULT '{}'")
            _ensure_column(connection, "approvals", "request_reason", "TEXT")
            _ensure_column(connection, "approvals", "decision_reason", "TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_column(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    columns = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
