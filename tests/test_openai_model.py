import unittest

from adaptive_agent_lab.core import Action


class OpenAIModelTests(unittest.TestCase):
    def test_action_keeps_provider_call_id(self) -> None:
        action = Action(
            "word_count",
            {"text": "hello world"},
            call_id="call_123",
            response_id="resp_123",
        )
        self.assertEqual(action.call_id, "call_123")
        self.assertEqual(action.response_id, "resp_123")


if __name__ == "__main__":
    unittest.main()
