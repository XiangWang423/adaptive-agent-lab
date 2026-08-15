import tempfile
import unittest
from pathlib import Path

from adaptive_agent_lab.core import AgentRunner
from adaptive_agent_lab.models import BaselineModel
from adaptive_agent_lab.tools import calculate, default_tools, word_count
from adaptive_agent_lab.trajectory import TrajectoryStore


class InterruptedModel:
    def decide(self, messages, tools):
        raise KeyboardInterrupt("cancelled by user")


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
    def test_word_count_uses_whitespace_boundaries(self) -> None:
        self.assertEqual(word_count("hello   world"), 2)
        self.assertEqual(word_count(""), 0)

    def test_tool_validates_arguments(self) -> None:
        word_count_tool = next(
            tool for tool in default_tools() if tool.name == "word_count"
        )

        with self.assertRaisesRegex(ValueError, "Missing required argument: text"):
            word_count_tool.invoke({})

        with self.assertRaisesRegex(ValueError, "Unexpected argument: typo"):
            word_count_tool.invoke({"text": "hello", "typo": True})

        with self.assertRaisesRegex(ValueError, "Argument text must be a string"):
            word_count_tool.invoke({"text": 123})

    def test_lookup_schema_explains_the_key_format(self) -> None:
        lookup_tool = next(tool for tool in default_tools() if tool.name == "lookup")
        description = lookup_tool.parameters["properties"]["key"]["description"]

        self.assertIn("category:entity", description)
        self.assertIn("capital:Japan", description)

    def test_agent_records_an_interrupted_run_before_reraising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(Path(directory) / "runs.db")
            runner = AgentRunner(InterruptedModel(), default_tools(), store)

            with self.assertRaisesRegex(KeyboardInterrupt, "cancelled by user"):
                runner.run("A task interrupted with Ctrl-C")

            runs = store.list_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["status"], "interrupted")
            self.assertIn("KeyboardInterrupt", runs[0]["error"])


if __name__ == "__main__":
    unittest.main()
