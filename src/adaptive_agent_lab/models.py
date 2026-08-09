from __future__ import annotations

import re
from typing import Sequence, Union

from .core import Action, FinalAnswer, Message, Tool


class BaselineModel:
    """A deterministic policy used to test the runtime without API cost."""

    _ARITHMETIC = re.compile(r"(-?\d+(?:\.\d+)?\s*[+\-*/%]\s*-?\d+(?:\.\d+)?)")

    def decide(
        self, messages: Sequence[Message], tools: Sequence[Tool]
    ) -> Union[Action, FinalAnswer]:
        latest = messages[-1]
        if latest.role == "tool":
            if latest.content["error"]:
                return FinalAnswer(f"Tool failed: {latest.content['error']}")
            return FinalAnswer(str(latest.content["output"]))

        task = str(messages[0].content).lower()
        arithmetic = self._ARITHMETIC.search(task)
        if arithmetic:
            return Action("calculator", {"expression": arithmetic.group(1)})
        if "capital of france" in task:
            return Action("lookup", {"key": "capital:france"})
        if "capital of japan" in task:
            return Action("lookup", {"key": "capital:japan"})
        if "language" in task and "brazil" in task:
            return Action("lookup", {"key": "language:brazil"})
        return FinalAnswer("I do not know how to solve this task.")
