from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from .evaluation import run_baseline
from .trajectory import TrajectoryStore


def default_db_path() -> Path:
    return Path(os.environ.get("ADAPTIVE_AGENT_DB", ".adaptive-agent-lab/trajectories.db"))


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-agent-lab")
    parser.add_argument("--db", type=Path, default=default_db_path())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("eval", help="Run the deterministic baseline benchmark.")
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

    store = TrajectoryStore(args.db)
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
