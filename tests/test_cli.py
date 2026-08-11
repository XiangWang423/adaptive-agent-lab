import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_agent_lab.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_run_command_accepts_task_and_model(self) -> None:
        args = build_parser().parse_args(
            ["run", "Count these words", "--model", "test-model"]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.task, "Count these words")
        self.assertEqual(args.model, "test-model")

    def test_run_command_requires_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "runs.db"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    SystemExit, "Provide --model or set OPENAI_MODEL"
                ):
                    main(["--db", str(db_path), "run", "Say hello"])


if __name__ == "__main__":
    unittest.main()
