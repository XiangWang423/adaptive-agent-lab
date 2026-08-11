from __future__ import annotations

import json
import os
from typing import Any, Sequence, Union

from .core import Action, FinalAnswer, Message, Tool


class OpenAIResponsesModel:
    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    @classmethod
    def from_env(cls, model: str) -> "OpenAIResponsesModel":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI support is not installed. Run: pip install -e '.[openai]'"
            ) from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        return cls(OpenAI(), model)

    def decide(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
    ) -> Union[Action, FinalAnswer]:
        latest = messages[-1]
        previous_response_id = None

        if latest.role == "tool":
            call_id = latest.content.get("call_id")
            previous_response_id = latest.content.get("response_id")
            if not call_id or not previous_response_id:
                raise ValueError("Tool result is missing OpenAI correlation IDs")
            output = json.dumps(
                {
                    "output": latest.content.get("output"),
                    "error": latest.content.get("error"),
                },
                ensure_ascii=False,
            )
            input_items = [
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            ]
        else:
            input_items = [
                {
                    "role": "user",
                    "content": str(message.content),
                }
                for message in messages
                if message.role == "user"
            ]

        tool_definitions = []

        for tool in tools:
            tool_definitions.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": False,
                }
            )

        request = {
            "model": self.model,
            "input": input_items,
            "tools": tool_definitions,
        }
        if previous_response_id is not None:
            request["previous_response_id"] = previous_response_id

        response = self.client.responses.create(
            **request,
        )

        for item in response.output:
            if item.type == "function_call":
                arguments = json.loads(item.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Function call arguments must be a JSON object")
                return Action(
                    tool_name=item.name,
                    arguments=arguments,
                    call_id=item.call_id,
                    response_id=response.id,
                )

        return FinalAnswer(response.output_text)
