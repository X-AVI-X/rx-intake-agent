"""SQLite-backed intake records, human review queue and append-only audit log."""

from __future__ import annotations

import json
import sqlite3
import threading
import time

from .schemas import IntakeResult, Status

_SCHEMA = """
CREATE TABLE IF NOT EXISTS intakes (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    intake_id TEXT NOT NULL,
    action TEXT NOT NULL,
    actor TEXT NOT NULL,
    detail TEXT,
    at REAL NOT NULL
);
"""


class ReviewError(Exception):
    pass


class Store:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def save(self, result: IntakeResult) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO intakes (id, status, result_json, created_at) VALUES (?, ?, ?, ?)",
                (result.id, result.status.value, result.model_dump_json(), time.time()),
            )
            self._audit(result.id, "created", f"system:{result.extractor}", result.status.value)

    def get(self, intake_id: str) -> IntakeResult | None:
        row = self._conn.execute("SELECT result_json FROM intakes WHERE id = ?", (intake_id,)).fetchone()
        return IntakeResult.model_validate_json(row[0]) if row else None

    def pending_reviews(self) -> list[IntakeResult]:
        rows = self._conn.execute(
            "SELECT result_json FROM intakes WHERE status = ? ORDER BY created_at", (Status.NEEDS_REVIEW.value,)
        ).fetchall()
        return [IntakeResult.model_validate_json(r[0]) for r in rows]

    def decide(self, intake_id: str, approve: bool, reviewer: str, reason: str) -> IntakeResult:
        """Record a pharmacist's decision. Only items still in review can be decided."""
        if not reviewer.strip() or not reason.strip():
            raise ReviewError("reviewer and reason are required")
        with self._lock, self._conn:
            row = self._conn.execute("SELECT result_json, status FROM intakes WHERE id = ?", (intake_id,)).fetchone()
            if row is None:
                raise ReviewError("not found")
            if row[1] != Status.NEEDS_REVIEW.value:
                raise ReviewError(f"intake is {row[1]}, not awaiting review")
            result = IntakeResult.model_validate_json(row[0])
            result.status = Status.ACCEPTED if approve else Status.REJECTED
            self._conn.execute(
                "UPDATE intakes SET status = ?, result_json = ? WHERE id = ?",
                (result.status.value, result.model_dump_json(), intake_id),
            )
            self._audit(intake_id, "approved" if approve else "rejected", f"human:{reviewer}", reason)
        return result

    def audit_trail(self, intake_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT action, actor, detail, at FROM audit_log WHERE intake_id = ? ORDER BY seq", (intake_id,)
        ).fetchall()
        return [{"action": a, "actor": b, "detail": c, "at": d} for a, b, c, d in rows]

    def _audit(self, intake_id: str, action: str, actor: str, detail: str) -> None:
        self._conn.execute(
            "INSERT INTO audit_log (intake_id, action, actor, detail, at) VALUES (?, ?, ?, ?, ?)",
            (intake_id, action, actor, json.dumps(detail) if not isinstance(detail, str) else detail, time.time()),
        )
