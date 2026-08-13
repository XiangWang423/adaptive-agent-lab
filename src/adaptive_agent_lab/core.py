from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Protocol, Sequence, Union

from .trajectory import TrajectoryStore


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def invoke(self, arguments: dict[str, Any]) -> Any:
        required = self.parameters.get("required", [])
        allowed = self.parameters.get("properties", {})

        for name in required:
            if name not in arguments:
                raise ValueError(f"Missing required argument: {name}")

        for name in arguments:
            if name not in allowed:
                raise ValueError(f"Unexpected argument: {name}")

        for name, value in arguments.items():
            parameter = allowed[name]
            if parameter.get("type") == "string" and not isinstance(value, str):
                raise ValueError(f"Argument {name} must be a string")

        return self.handler(**arguments)


@dataclass(frozen=True)
class Action:
    tool_name: str
    arguments: dict[str, Any]
    call_id: str | None = None
    response_id: str | None = None


@dataclass(frozen=True)
class FinalAnswer:
    content: str


Decision = Union[Action, FinalAnswer]


@dataclass(frozen=True)
class Message:
    role: str
    content: Any


class ChatModel(Protocol):
    def decide(self, messages: Sequence[Message], tools: Sequence[Tool]) -> Decision:
        """Return one tool action or a final answer."""


@dataclass(frozen=True)
class AgentResult:
    run_id: str
    status: str
    answer: str | None
    steps: int
    tool_calls: int
    error: str | None = None


class AgentRunner:
    """A deliberately small agent state machine with observable transitions."""

    def __init__(
        self,
        model: ChatModel,
        tools: Sequence[Tool],
        store: TrajectoryStore,
        max_steps: int = 8,
    ) -> None:
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.store = store
        self.max_steps = max_steps

    def run(self, task: str) -> AgentResult:
        run_id = self.store.start_run(task)
        messages = [Message("user", task)]
        tool_calls = 0

        try:
            for step_index in range(self.max_steps):
                started = perf_counter()
                decision = self.model.decide(messages, list(self.tools.values()))
                decision_ms = (perf_counter() - started) * 1000

                if isinstance(decision, FinalAnswer):
                    self.store.append_step(
                        run_id,
                        step_index,
                        "final",
                        {"content": decision.content},
                        duration_ms=decision_ms,
                    )
                    self.store.finish_run(run_id, "completed", decision.content)
                    return AgentResult(
                        run_id, "completed", decision.content, step_index + 1, tool_calls
                    )

                self.store.append_step(
                    run_id,
                    step_index,
                    "model_action",
                    {
                        "tool_name": decision.tool_name,
                        "arguments": decision.arguments,
                        "call_id": decision.call_id,
                        "response_id": decision.response_id,
                    },
                    duration_ms=decision_ms,
                )
                tool = self.tools.get(decision.tool_name)
                if tool is None:
                    raise ValueError(f"Unknown tool: {decision.tool_name}")

                tool_started = perf_counter()
                try:
                    output = tool.invoke(decision.arguments)
                    tool_error = None
                except Exception as exc:
                    output = None
                    tool_error = f"{type(exc).__name__}: {exc}"
                tool_ms = (perf_counter() - tool_started) * 1000
                tool_calls += 1
                observation = {
                    "tool_name": tool.name,
                    "output": output,
                    "error": tool_error,
                    "call_id": decision.call_id,
                    "response_id": decision.response_id,
                }
                self.store.append_step(
                    run_id,
                    step_index,
                    "tool_result",
                    observation,
                    duration_ms=tool_ms,
                )
                messages.append(Message("assistant", decision))
                messages.append(Message("tool", observation))

            error = f"Maximum step count ({self.max_steps}) reached"
            self.store.finish_run(run_id, "failed", error=error)
            return AgentResult(run_id, "failed", None, self.max_steps, tool_calls, error)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.store.finish_run(run_id, "failed", error=error)
            return AgentResult(run_id, "failed", None, len(messages), tool_calls, error)
