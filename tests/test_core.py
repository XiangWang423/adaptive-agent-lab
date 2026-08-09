import tempfile
import unittest
from pathlib import Path

from adaptive_agent_lab.core import AgentRunner
from adaptive_agent_lab.models import BaselineModel
from adaptive_agent_lab.tools import calculate, default_tools
from adaptive_agent_lab.trajectory import TrajectoryStore


class CoreTests(unittest.TestCase):
    def test_calculator_accepts_arithmetic_and_rejects_code(self) -> None:
        self.assertEqual(calculate("2 + 3 * 4"), 14)
        with self.assertRaises(ValueError):
            calculate("__import__('os').getcwd()")

    def test_agent_records_action_tool_result_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(Path(directory) / "runs.db")
            result = AgentRunner(BaselineModel(), default_tools(), store).run(
                "What is 6 * 7?"
            )

            self.assertEqual(result.answer, "42")
            self.assertEqual(result.tool_calls, 1)
            trace = store.get_run(result.run_id)
            self.assertIsNotNone(trace)
            assert trace is not None
            self.assertEqual(
                [step["kind"] for step in trace["steps"]],
                ["model_action", "tool_result", "final"],
            )


if __name__ == "__main__":
    unittest.main()
