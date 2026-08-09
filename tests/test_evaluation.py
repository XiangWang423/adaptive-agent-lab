import tempfile
import unittest
from pathlib import Path

from adaptive_agent_lab.evaluation import BASELINE_CASES, run_baseline


class EvaluationTests(unittest.TestCase):
    def test_baseline_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_baseline(Path(directory) / "eval.db")

        self.assertEqual(report["summary"]["cases"], len(BASELINE_CASES))
        self.assertEqual(report["summary"]["success_rate"], 1.0)
        self.assertTrue(all(result["passed"] for result in report["results"]))


if __name__ == "__main__":
    unittest.main()
