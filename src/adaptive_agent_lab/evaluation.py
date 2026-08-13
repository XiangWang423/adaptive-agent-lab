from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from math import ceil
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Sequence

from .core import AgentRunner, ChatModel
from .models import BaselineModel
from .tools import default_tools
from .trajectory import TrajectoryStore


@dataclass(frozen=True)
class EvalCase:
    id: str
    task: str
    expected: str
    expected_tool: str | None = None


BASELINE_CASES = [
    EvalCase("arithmetic-multiply", "What is 12 * 7?", "84", "calculator"),
    EvalCase("arithmetic-add", "Calculate 19 + 23.", "42", "calculator"),
    EvalCase("capital-france", "What is the capital of France?", "Paris", "lookup"),
    EvalCase("capital-japan", "What is the capital of Japan?", "Tokyo", "lookup"),
    EvalCase(
        "language-brazil",
        "What language is mainly spoken in Brazil?",
        "Portuguese",
        "lookup",
    ),
    EvalCase(
        "word-count",
        "Count the words in: I love building agents",
        "4",
        "word_count",
    ),
]


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                case = EvalCase(**json.loads(line))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid eval case on line {line_number}: {exc}") from exc
            if case.id in seen_ids:
                raise ValueError(f"Duplicate eval case ID: {case.id}")
            seen_ids.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("Evaluation requires at least one case")
    return cases


def _trajectory_metrics(store: TrajectoryStore, run_id: str) -> dict[str, Any]:
    trace = store.get_run(run_id)
    if trace is None:
        return {
            "selected_tools": [],
            "tool_errors": 0,
            "first_tool_call_succeeded": None,
        }
    selected_tools = [
        step["payload"]["tool_name"]
        for step in trace["steps"]
        if step["kind"] == "model_action"
    ]
    tool_results = [
        step["payload"]
        for step in trace["steps"]
        if step["kind"] == "tool_result"
    ]
    return {
        "selected_tools": selected_tools,
        "tool_errors": sum(result.get("error") is not None for result in tool_results),
        "first_tool_call_succeeded": (
            None if not tool_results else tool_results[0].get("error") is None
        ),
    }


_NUMBER = re.compile(r"(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])")


def _score_answer(expected: str, actual: str | None) -> tuple[bool, str]:
    """Score model output without paying for a second model as a judge."""

    if actual is None:
        return False, "missing"

    try:
        expected_number = Decimal(expected.strip())
    except InvalidOperation:
        normalized_expected = expected.strip().rstrip(".!?").casefold()
        normalized_actual = actual.strip().rstrip(".!?").casefold()
        return normalized_actual == normalized_expected, "normalized_text"

    actual_numbers = _NUMBER.findall(actual)
    if len(actual_numbers) != 1:
        return False, "single_number"
    try:
        return Decimal(actual_numbers[0]) == expected_number, "single_number"
    except InvalidOperation:
        return False, "single_number"


def run_evaluation(
    model: ChatModel,
    cases: Sequence[EvalCase],
    db_path: str | Path,
    max_steps: int = 8,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("Evaluation requires at least one case")

    store = TrajectoryStore(db_path)
    runner = AgentRunner(model, default_tools(), store, max_steps=max_steps)
    results: list[dict[str, Any]] = []

    for case in cases:
        started = perf_counter()
        agent_result = runner.run(case.task)
        latency_ms = (perf_counter() - started) * 1000
        answer_matched, scoring_method = _score_answer(
            case.expected, agent_result.answer
        )
        passed = agent_result.status == "completed" and answer_matched
        trajectory = _trajectory_metrics(store, agent_result.run_id)
        selected_tools = trajectory["selected_tools"]
        selected_tool = selected_tools[0] if selected_tools else None
        tool_errors = trajectory["tool_errors"]
        recovered_after_tool_error = (
            agent_result.status == "completed" and tool_errors > 0
        )
        tool_selection_passed = (
            None
            if case.expected_tool is None
            else selected_tool == case.expected_tool
        )
        results.append(
            {
                "case": asdict(case),
                "passed": passed,
                "scoring_method": scoring_method,
                "tool_selection_passed": tool_selection_passed,
                "run_id": agent_result.run_id,
                "actual": agent_result.answer,
                "selected_tool": selected_tool,
                "selected_tools": selected_tools,
                "first_tool_call_succeeded": trajectory[
                    "first_tool_call_succeeded"
                ],
                "tool_errors": tool_errors,
                "recovered_after_tool_error": recovered_after_tool_error,
                "status": agent_result.status,
                "steps": agent_result.steps,
                "tool_calls": agent_result.tool_calls,
                "latency_ms": round(latency_ms, 3),
                "error": agent_result.error,
            }
        )

    passed_count = sum(result["passed"] for result in results)
    tool_results = [
        result["tool_selection_passed"]
        for result in results
        if result["tool_selection_passed"] is not None
    ]
    correct_tool_count = sum(tool_results)
    first_attempt_results = [
        result["first_tool_call_succeeded"]
        for result in results
        if result["first_tool_call_succeeded"] is not None
    ]
    first_attempt_successes = sum(first_attempt_results)
    cases_with_tool_errors = sum(result["tool_errors"] > 0 for result in results)
    recovered_cases = sum(result["recovered_after_tool_error"] for result in results)
    latencies = sorted(result["latency_ms"] for result in results)
    p95_index = ceil(0.95 * len(latencies)) - 1
    return {
        "summary": {
            "cases": len(results),
            "passed": passed_count,
            "success_rate": passed_count / len(results),
            "tool_selection_cases": len(tool_results),
            "correct_tool_selections": correct_tool_count,
            "tool_selection_accuracy": (
                correct_tool_count / len(tool_results) if tool_results else None
            ),
            "first_tool_call_success_rate": (
                first_attempt_successes / len(first_attempt_results)
                if first_attempt_results
                else None
            ),
            "total_tool_errors": sum(result["tool_errors"] for result in results),
            "cases_with_tool_errors": cases_with_tool_errors,
            "recovered_cases": recovered_cases,
            "recovery_rate": (
                recovered_cases / cases_with_tool_errors
                if cases_with_tool_errors
                else None
            ),
            "average_steps": round(mean(result["steps"] for result in results), 3),
            "average_tool_calls": round(mean(result["tool_calls"] for result in results), 3),
            "average_latency_ms": round(mean(result["latency_ms"] for result in results), 3),
            "p95_latency_ms": latencies[p95_index],
        },
        "results": results,
    }


def run_baseline(db_path: str | Path) -> dict[str, Any]:
    return run_evaluation(BaselineModel(), BASELINE_CASES, db_path)
