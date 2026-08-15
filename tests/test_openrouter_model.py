import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adaptive_agent_lab.core import Action, AgentRunner, FinalAnswer, Message
from adaptive_agent_lab.openrouter_model import OpenRouterChatModel
from adaptive_agent_lab.tools import default_tools
from adaptive_agent_lab.trajectory import TrajectoryStore


class FakeCompletions:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, *responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(*responses))


def text_response(content: str):
    message = SimpleNamespace(content=content, tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_response(name: str, arguments: str, call_id: str):
    tool_call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class OpenRouterChatModelTests(unittest.TestCase):
    def test_converts_text_response_to_final_answer(self) -> None:
        client = FakeClient(text_response("Hello from OpenRouter"))
        model = OpenRouterChatModel(client, model="test-model")

        decision = model.decide([Message("user", "Say hello")], default_tools())

        self.assertIsInstance(decision, FinalAnswer)
        self.assertEqual(decision.content, "Hello from OpenRouter")
        request = client.chat.completions.requests[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["messages"][0]["role"], "system")
        self.assertIn("return the tool output faithfully", request["messages"][0]["content"])

    def test_converts_tool_call_to_action(self) -> None:
        client = FakeClient(
            tool_response("word_count", '{"text": "hello world"}', "call_123")
        )
        model = OpenRouterChatModel(client, model="test-model")

        decision = model.decide(
            [Message("user", "Count these words")], default_tools()
        )

        self.assertIsInstance(decision, Action)
        self.assertEqual(decision.tool_name, "word_count")
        self.assertEqual(decision.arguments, {"text": "hello world"})
        self.assertEqual(decision.call_id, "call_123")

        request = client.chat.completions.requests[0]
        tool_names = {tool["function"]["name"] for tool in request["tools"]}
        self.assertIn("word_count", tool_names)
        self.assertFalse(request["parallel_tool_calls"])

    def test_sends_recalled_memory_as_system_context(self) -> None:
        client = FakeClient(text_response("done"))
        model = OpenRouterChatModel(client, model="test-model")

        model.decide(
            [
                Message("memory", "A similar call used the text argument."),
                Message("user", "Count words"),
            ],
            default_tools(),
        )

        messages = client.chat.completions.requests[0]["messages"]
        self.assertEqual([message["role"] for message in messages[:3]], [
            "system",
            "system",
            "user",
        ])
        self.assertIn("similar call", messages[1]["content"])

    def test_sends_complete_history_after_tool_result(self) -> None:
        client = FakeClient(
            tool_response("word_count", '{"text": "hello world"}', "call_123"),
            text_response("2"),
        )
        model = OpenRouterChatModel(client, model="test-model")

        with tempfile.TemporaryDirectory() as directory:
            store = TrajectoryStore(Path(directory) / "runs.db")
            result = AgentRunner(model, default_tools(), store).run(
                "Count the words in hello world"
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "2")
        self.assertEqual(result.tool_calls, 1)
        self.assertEqual(len(client.chat.completions.requests), 2)

        messages = client.chat.completions.requests[1]["messages"]
        self.assertEqual([message["role"] for message in messages], [
            "system",
            "user",
            "assistant",
            "tool",
        ])
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "call_123")
        self.assertEqual(messages[3]["tool_call_id"], "call_123")
        self.assertEqual(
            json.loads(messages[3]["content"]),
            {"output": 2, "error": None},
        )


if __name__ == "__main__":
    unittest.main()
