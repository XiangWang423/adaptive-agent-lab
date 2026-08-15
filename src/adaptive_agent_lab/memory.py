from __future__ import annotations

import re
from typing import Any

from .trajectory import TrajectoryStore


_WORD = re.compile(r"[a-z0-9_]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "after",
    "in",
    "of",
    "the",
    "to",
    "use",
    "what",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _WORD.findall(text.lower())
        if token not in _STOP_WORDS
    }


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class TrajectoryMemory:
    """Retrieve similar runs that recovered from a tool error."""

    def __init__(
        self,
        store: TrajectoryStore,
        limit: int = 3,
        candidate_limit: int = 100,
        minimum_similarity: float = 0.2,
    ) -> None:
        self.store = store
        self.limit = limit
        self.candidate_limit = candidate_limit
        self.minimum_similarity = minimum_similarity

    def recall(self, task: str) -> list[dict[str, Any]]:
        candidates: list[tuple[float, dict[str, Any]]] = []

        for run in self.store.list_runs(self.candidate_limit):
            if run["status"] != "completed":
                continue

            score = _similarity(task, run["task"])
            if score < self.minimum_similarity:
                continue

            trace = self.store.get_run(run["id"])
            if trace is None:
                continue
            recovery = self._extract_recovery(trace["steps"])
            if recovery is None:
                continue

            candidates.append(
                (
                    score,
                    {
                        "similarity": round(score, 3),
                        **recovery,
                    },
                )
            )

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return [memory for _, memory in candidates[: self.limit]]

    @staticmethod
    def _extract_recovery(
        steps: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        latest_action: dict[str, Any] | None = None
        failed_call: dict[str, Any] | None = None

        for step in steps:
            payload = step["payload"]
            if step["kind"] == "model_action":
                latest_action = payload
                continue

            if step["kind"] != "tool_result" or latest_action is None:
                continue

            if payload.get("error"):
                failed_call = {
                    "tool_name": latest_action["tool_name"],
                    "arguments": latest_action["arguments"],
                    "error": payload["error"],
                }
                continue

            if failed_call is not None:
                return {
                    "failed_call": failed_call,
                    "successful_call": {
                        "tool_name": latest_action["tool_name"],
                        "arguments": latest_action["arguments"],
                    },
                }

        return None
