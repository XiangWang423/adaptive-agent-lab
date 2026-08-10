from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Sequence, Union

from adaptive_agent_lab.core import (
    Action,
    AgentRunner,
    FinalAnswer,
    Message,
    Tool,
)
from adaptive_agent_lab.tools import default_tools
from adaptive_agent_lab.trajectory import TrajectoryStore


class RecoveringModel:
    def decide(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
    ) -> Union[Action, FinalAnswer]:
        latest = messages[-1]

        if latest.role == "user":
            return Action("word_count", {"text": 123})

        if latest.role == "tool" and latest.content["error"]:
            return Action("word_count", {"text": "hello world"})

        return FinalAnswer(str(latest.content["output"]))


class RecoveryTests(unittest.TestCase):
    def test_agent_recovers_after_invalid_tool_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(Path(directory) / "runs.db")
            runner = AgentRunner(RecoveringModel(), default_tools(), store)

            result = runner.run("Count the words after correcting the arguments")
            trace = store.get_run(result.run_id)

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.answer, "2")
            self.assertEqual(result.tool_calls, 2)
            self.assertIsNotNone(trace)
            assert trace is not None
            self.assertEqual(
                [step["kind"] for step in trace["steps"]],
                [
                    "model_action",
                    "tool_result",
                    "model_action",
                    "tool_result",
                    "final",
                ],
            )


if __name__ == "__main__":
    unittest.main()
