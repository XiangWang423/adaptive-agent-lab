from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrajectoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    step_index INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id, id);
                """
            )

    def start_run(self, task: str) -> str:
        run_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs (id, task, status, started_at) VALUES (?, ?, ?, ?)",
                (run_id, task, "running", _now()),
            )
        return run_id

    def append_step(
        self,
        run_id: str,
        step_index: int,
        kind: str,
        payload: dict[str, Any],
        duration_ms: float = 0.0,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO steps
                   (run_id, step_index, kind, payload, duration_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, step_index, kind, json.dumps(payload), duration_ms, _now()),
            )

    def finish_run(
        self,
        run_id: str,
        status: str,
        answer: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """UPDATE runs
                   SET status = ?, answer = ?, error = ?, finished_at = ?
                   WHERE id = ?""",
                (status, answer, error, _now(), run_id),
            )

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT r.*,
                          COUNT(s.id) AS event_count,
                          SUM(CASE WHEN s.kind = 'tool_result' THEN 1 ELSE 0 END) AS tool_calls
                   FROM runs r LEFT JOIN steps s ON s.run_id = r.id
                   GROUP BY r.id ORDER BY r.started_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            run = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            steps = connection.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        result = dict(run)
        result["steps"] = [
            {**dict(step), "payload": json.loads(step["payload"])} for step in steps
        ]
        return result
