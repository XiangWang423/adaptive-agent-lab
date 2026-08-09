#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "src"))

from adaptive_agent_lab.evaluation import run_baseline  # noqa: E402
from adaptive_agent_lab.trajectory import TrajectoryStore  # noqa: E402


DB_PATH = Path(os.environ.get("ADAPTIVE_AGENT_DB", ".adaptive-agent-lab/trajectories.db"))


TOOLS = [
    {
        "name": "run_baseline_eval",
        "description": "Run the deterministic Adaptive Agent Lab benchmark.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_agent_runs",
        "description": "List recent stored agent runs.",
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
        },
    },
    {
        "name": "get_agent_run",
        "description": "Return one agent run and its ordered execution steps.",
        "inputSchema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    },
]


def _tool_result(value: Any, is_error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
        "isError": is_error,
    }


def _handle(method: str, params: dict[str, Any]) -> Any:
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "adaptive-agent-lab", "version": "0.1.0"},
        }
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        store = TrajectoryStore(DB_PATH)
        if name == "run_baseline_eval":
            return _tool_result(run_baseline(DB_PATH))
        if name == "list_agent_runs":
            return _tool_result(store.list_runs(int(arguments.get("limit", 10))))
        if name == "get_agent_run":
            run = store.get_run(str(arguments.get("run_id", "")))
            return _tool_result(run or {"error": "Run not found"}, is_error=run is None)
        return _tool_result({"error": f"Unknown tool: {name}"}, is_error=True)
    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    for line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(line)
            if "id" not in request:
                continue
            response = {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": _handle(request.get("method", ""), request.get("params", {})),
            }
        except Exception as exc:
            response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if isinstance(request, dict) else None,
                "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"},
            }
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
