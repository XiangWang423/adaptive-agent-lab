from __future__ import annotations

import json
import os
from typing import Any, Sequence, Union

from .core import Action, FinalAnswer, Message, Tool


SYSTEM_PROMPT = (
    "Use the available tools when they can answer the user's request. "
    "After a successful tool result, return the tool output faithfully without "
    "adding labels, prefixes, or explanations. If a tool reports an error, inspect "
    "the error and retry with corrected arguments when possible."
)


class OpenRouterChatModel:
    """OpenRouter adapter using its OpenAI-compatible Chat Completions API."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    @classmethod
    def from_env(cls, model: str) -> "OpenRouterChatModel":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenRouter support is not installed. Run: pip install -e '.[openai]'"
            ) from exc

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        return cls(client, model)

    def decide(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
    ) -> Union[Action, FinalAnswer]:
        request_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *(self._convert_message(message) for message in messages),
        ]
        tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=request_messages,
            tools=tool_definitions,
            parallel_tool_calls=False,
        )
        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Function call arguments must be a JSON object")
            return Action(
                tool_name=tool_call.function.name,
                arguments=arguments,
                call_id=tool_call.id,
            )

        return FinalAnswer(message.content or "")

    @staticmethod
    def _convert_message(message: Message) -> dict[str, Any]:
        if message.role == "memory":
            return {"role": "system", "content": str(message.content)}

        if message.role == "user":
            return {"role": "user", "content": str(message.content)}

        if message.role == "assistant" and isinstance(message.content, Action):
            action = message.content
            if not action.call_id:
                raise ValueError("Assistant action is missing a tool call ID")
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": action.call_id,
                        "type": "function",
                        "function": {
                            "name": action.tool_name,
                            "arguments": json.dumps(
                                action.arguments, ensure_ascii=False
                            ),
                        },
                    }
                ],
            }

        if message.role == "tool":
            call_id = message.content.get("call_id")
            if not call_id:
                raise ValueError("Tool result is missing a tool call ID")
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    {
                        "output": message.content.get("output"),
                        "error": message.content.get("error"),
                    },
                    ensure_ascii=False,
                ),
            }

        raise ValueError(f"Unsupported message role: {message.role}")
