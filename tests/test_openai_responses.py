import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adaptive_agent_lab.core import Action, AgentRunner, FinalAnswer, Message
from adaptive_agent_lab.openai_model import OpenAIResponsesModel
from adaptive_agent_lab.tools import default_tools
from adaptive_agent_lab.trajectory import TrajectoryStore


class FakeResponses:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, *responses):
        self.responses = FakeResponses(*responses)


class OpenAIResponsesModelTests(unittest.TestCase):
    def test_converts_text_response_to_final_answer(self) -> None:
        response = SimpleNamespace(
            id="resp_123",
            output=[],
            output_text="Hello from the model",
        )
        client = FakeClient(response)
        model = OpenAIResponsesModel(client, model="test-model")

        decision = model.decide(
            [Message("user", "Say hello")],
            default_tools(),
        )

        self.assertIsInstance(decision, FinalAnswer)
        self.assertEqual(decision.content, "Hello from the model")
        self.assertEqual(client.responses.requests[0]["model"], "test-model")

    def test_converts_function_call_to_action(self) -> None:
        response = SimpleNamespace(
            id="resp_123",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="word_count",
                    arguments='{"text": "hello world"}',
                    call_id="call_123",
                )
            ],
            output_text="",
        )
        client = FakeClient(response)
        model = OpenAIResponsesModel(client, model="test-model")

        decision = model.decide(
            [Message("user", "Count these words")],
            default_tools(),
        )

        self.assertIsInstance(decision, Action)
        self.assertEqual(decision.tool_name, "word_count")
        self.assertEqual(decision.arguments, {"text": "hello world"})
        self.assertEqual(decision.call_id, "call_123")
        self.assertEqual(decision.response_id, "resp_123")

        request = client.responses.requests[0]
        tool_names = {tool["name"] for tool in request["tools"]}
        self.assertIn("word_count", tool_names)

    def test_completes_tool_call_round_trip(self) -> None:
        function_call = SimpleNamespace(
            id="resp_123",
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="word_count",
                    arguments='{"text": "hello world"}',
                    call_id="call_123",
                )
            ],
            output_text="",
        )
        final_response = SimpleNamespace(
            id="resp_456",
            output=[],
            output_text="2",
        )
        client = FakeClient(function_call, final_response)
        model = OpenAIResponsesModel(client, model="test-model")

        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(Path(directory) / "runs.db")
            result = AgentRunner(model, default_tools(), store).run(
                "Count the words in hello world"
            )
            trace = store.get_run(result.run_id)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "2")
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(len(client.responses.requests), 2)

        second_request = client.responses.requests[1]
        self.assertEqual(second_request["previous_response_id"], "resp_123")
        tool_output = second_request["input"][0]
        self.assertEqual(tool_output["type"], "function_call_output")
        self.assertEqual(tool_output["call_id"], "call_123")
        self.assertEqual(
            json.loads(tool_output["output"]),
            {"output": 2, "error": None},
        )

        self.assertIsNotNone(trace)
        assert trace is not None
        first_tool_result = trace["steps"][1]["payload"]
        self.assertEqual(first_tool_result["call_id"], "call_123")
        self.assertEqual(first_tool_result["response_id"], "resp_123")


if __name__ == "__main__":
    unittest.main()
