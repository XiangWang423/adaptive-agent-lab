import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adaptive_agent_lab.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_run_command_accepts_task_and_model(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "Count these words",
                "--model",
                "test-model",
                "--provider",
                "openrouter",
                "--max-steps",
                "2",
            ]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.task, "Count these words")
        self.assertEqual(args.model, "test-model")
        self.assertEqual(args.provider, "openrouter")
        self.assertEqual(args.max_steps, 2)

    def test_run_command_requires_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "runs.db"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    SystemExit, "Provide --model or set AGENT_MODEL"
                ):
                    main(["--db", str(db_path), "run", "Say hello"])

    def test_live_eval_accepts_model_and_case_file(self) -> None:
        args = build_parser().parse_args(
            [
                "eval-live",
                "--model",
                "test-model",
                "--provider",
                "openrouter",
                "--cases",
                "cases.jsonl",
                "--max-steps",
                "3",
            ]
        )

        self.assertEqual(args.command, "eval-live")
        self.assertEqual(args.model, "test-model")
        self.assertEqual(args.provider, "openrouter")
        self.assertEqual(args.cases, Path("cases.jsonl"))
        self.assertEqual(args.max_steps, 3)

    def test_live_eval_requires_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "runs.db"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    SystemExit, "Provide --model or set AGENT_MODEL"
                ):
                    main(["--db", str(db_path), "eval-live"])


if __name__ == "__main__":
    unittest.main()
