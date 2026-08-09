---
name: run-agent-evals
description: Run Adaptive Agent Lab's deterministic benchmark, compare success and efficiency metrics, and inspect SQLite execution traces. Use when Codex needs to evaluate this agent runtime, diagnose a failed benchmark task, verify a change to the agent loop or tools, or explain an execution trajectory.
---

# Run Agent Evals

Use the bundled MCP tools when available. Fall back to the project CLI when working directly in the repository.

## Workflow

1. Run `run_baseline_eval` without changing the implementation.
2. Report success rate, average steps, average tool calls, and failed case IDs.
3. For each failure, call `get_agent_run` with its run ID and identify the first incorrect decision or tool result.
4. Make only the smallest relevant code change when the user asks for a fix.
5. Run the complete benchmark again and compare before/after metrics.
6. Treat a lower success rate as a regression. Do not accept a change only because one example improves.

## CLI fallback

Run from the plugin root:

```bash
python3 -m adaptive_agent_lab.cli eval
python3 -m adaptive_agent_lab.cli runs --limit 10
python3 -m adaptive_agent_lab.cli show <run-id>
```

If the package has not been installed, prefix commands with `PYTHONPATH=src`.

## Interpretation rules

- Success rate is the release gate; efficiency metrics are secondary.
- A tool error is different from a wrong model decision; name the failing layer.
- Keep raw task inputs and expected outputs fixed while comparing implementations.
- Preserve failed traces. They are inputs to the later skill-learning phase.
