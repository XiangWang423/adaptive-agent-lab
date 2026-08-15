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
from adaptive_agent_lab.memory import TrajectoryMemory
from adaptive_agent_lab.tools import default_tools
from adaptive_agent_lab.trajectory import TrajectoryStore


class LearningModel:
    """A deterministic stand-in for a model that can use recalled experience."""

    def decide(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
    ) -> Union[Action, FinalAnswer]:
        latest = messages[-1]

        if latest.role == "user":
            memories = [message for message in messages if message.role == "memory"]
            if memories:
                return Action("word_count", {"text": "hello world"})
            return Action("word_count", {"text": 123})

        if latest.role == "tool" and latest.content["error"]:
            return Action("word_count", {"text": "hello world"})

        return FinalAnswer(str(latest.content["output"]))


class TrajectoryMemoryTests(unittest.TestCase):
    def test_similar_recovered_run_reduces_repeated_tool_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(Path(directory) / "runs.db")

            first = AgentRunner(LearningModel(), default_tools(), store).run(
                "Count words after correcting invalid arguments"
            )
            second = AgentRunner(
                LearningModel(),
                default_tools(),
                store,
                memory=TrajectoryMemory(store),
            ).run("Count words after correcting bad arguments")
            trace = store.get_run(second.run_id)

            self.assertEqual(first.tool_calls, 2)
            self.assertEqual(second.status, "completed")
            self.assertEqual(second.answer, "2")
            self.assertEqual(second.tool_calls, 1)
            self.assertIsNotNone(trace)
            assert trace is not None
            self.assertEqual(trace["steps"][0]["kind"], "memory_recall")
            self.assertEqual(trace["steps"][0]["payload"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
