from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_agent_lab.trajectory import TrajectoryStore


class TrajectoryStoreTests(unittest.TestCase):
    def test_closes_each_database_connection(self) -> None:
        real_connect = sqlite3.connect
        connections: list[sqlite3.Connection] = []

        def tracked_connect(path: Path) -> sqlite3.Connection:
            connection = real_connect(path)
            connections.append(connection)
            return connection

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "adaptive_agent_lab.trajectory.sqlite3.connect",
                side_effect=tracked_connect,
            ):
                store = TrajectoryStore(Path(directory) / "runs.db")
                run_id = store.start_run("test task")
                store.append_step(run_id, 0, "test", {"ok": True})
                store.finish_run(run_id, "completed", answer="done")
                store.list_runs()
                store.get_run(run_id)

        self.assertGreater(len(connections), 0)
        for connection in connections:
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()
