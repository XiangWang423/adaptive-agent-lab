from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .core import AgentRunner
from .evaluation import load_eval_cases, run_baseline, run_evaluation
from .memory import TrajectoryMemory
from .openai_model import OpenAIResponsesModel
from .openrouter_model import OpenRouterChatModel
from .tools import default_tools
from .trajectory import TrajectoryStore


def default_db_path() -> Path:
    return Path(os.environ.get("ADAPTIVE_AGENT_DB", ".adaptive-agent-lab/trajectories.db"))


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _load_live_model(provider: str, model: str):
    if provider == "openrouter":
        return OpenRouterChatModel.from_env(model)
    return OpenAIResponsesModel.from_env(model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-agent-lab")
    parser.add_argument("--db", type=Path, default=default_db_path())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("eval", help="Run the deterministic baseline benchmark.")
    live_eval = subparsers.add_parser(
        "eval-live", help="Evaluate a live model provider with JSONL cases."
    )
    default_model = os.environ.get("AGENT_MODEL") or os.environ.get("OPENAI_MODEL")
    live_eval.add_argument("--model", default=default_model)
    live_eval.add_argument(
        "--provider",
        choices=("openai", "openrouter"),
        default=os.environ.get("AGENT_PROVIDER", "openai"),
    )
    live_eval.add_argument(
        "--cases", type=Path, default=Path("evals/live_cases.jsonl")
    )
    live_eval.add_argument("--max-steps", type=int, default=8)
    run = subparsers.add_parser("run", help="Run one task with a live model provider.")
    run.add_argument("task")
    run.add_argument("--model", default=default_model)
    run.add_argument(
        "--provider",
        choices=("openai", "openrouter"),
        default=os.environ.get("AGENT_PROVIDER", "openai"),
    )
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument(
        "--memory",
        action="store_true",
        help="Recall similar recovered trajectories before the model decides.",
    )
    runs = subparsers.add_parser("runs", help="List recent agent runs.")
    runs.add_argument("--limit", type=int, default=10)
    show = subparsers.add_parser("show", help="Show one execution trajectory.")
    show.add_argument("run_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "eval":
        _print(run_baseline(args.db))
        return 0

    if args.command == "eval-live":
        if not args.model:
            raise SystemExit("Provide --model or set AGENT_MODEL")
        try:
            model = _load_live_model(args.provider, args.model)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        cases = load_eval_cases(args.cases)
        _print(run_evaluation(model, cases, args.db, max_steps=args.max_steps))
        return 0

    store = TrajectoryStore(args.db)
    if args.command == "run":
        if not args.model:
            raise SystemExit("Provide --model or set AGENT_MODEL")
        try:
            model = _load_live_model(args.provider, args.model)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        result = AgentRunner(
            model,
            default_tools(),
            store,
            max_steps=args.max_steps,
            memory=TrajectoryMemory(store) if args.memory else None,
        ).run(args.task)
        _print(asdict(result))
        return 0 if result.status == "completed" else 1
    if args.command == "runs":
        _print(store.list_runs(args.limit))
        return 0
    if args.command == "show":
        run = store.get_run(args.run_id)
        if run is None:
            raise SystemExit(f"Unknown run ID: {args.run_id}")
        _print(run)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
