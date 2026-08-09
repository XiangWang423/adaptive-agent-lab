from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from .core import AgentRunner
from .models import BaselineModel
from .tools import default_tools
from .trajectory import TrajectoryStore


@dataclass(frozen=True)
class EvalCase:
    id: str
    task: str
    expected: str


BASELINE_CASES = [
    EvalCase("arithmetic-multiply", "What is 12 * 7?", "84"),
    EvalCase("arithmetic-add", "Calculate 19 + 23.", "42"),
    EvalCase("capital-france", "What is the capital of France?", "Paris"),
    EvalCase("capital-japan", "What is the capital of Japan?", "Tokyo"),
    EvalCase("language-brazil", "What language is mainly spoken in Brazil?", "Portuguese"),
]


def run_baseline(db_path: str | Path) -> dict[str, Any]:
    store = TrajectoryStore(db_path)
    runner = AgentRunner(BaselineModel(), default_tools(), store)
    results: list[dict[str, Any]] = []

    for case in BASELINE_CASES:
        started = perf_counter()
        agent_result = runner.run(case.task)
        latency_ms = (perf_counter() - started) * 1000
        passed = agent_result.status == "completed" and agent_result.answer == case.expected
        results.append(
            {
                "case": asdict(case),
                "passed": passed,
                "run_id": agent_result.run_id,
                "actual": agent_result.answer,
                "status": agent_result.status,
                "steps": agent_result.steps,
                "tool_calls": agent_result.tool_calls,
                "latency_ms": round(latency_ms, 3),
                "error": agent_result.error,
            }
        )

    passed_count = sum(result["passed"] for result in results)
    return {
        "summary": {
            "cases": len(results),
            "passed": passed_count,
            "success_rate": passed_count / len(results),
            "average_steps": round(mean(result["steps"] for result in results), 3),
            "average_tool_calls": round(mean(result["tool_calls"] for result in results), 3),
            "average_latency_ms": round(mean(result["latency_ms"] for result in results), 3),
        },
        "results": results,
    }
