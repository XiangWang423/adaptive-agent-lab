# Adaptive Agent Lab

Adaptive Agent Lab is a small, observable agent runtime built to answer a practical question:
can an agent learn from failed trajectories without silently regressing on tasks it already solves?

Phase 1 establishes the measurable baseline: a framework-free agent loop, typed tools, SQLite
trajectories, deterministic evaluations, trace inspection, and a Codex MCP/Skill interface. Later
phases will add a real model provider, memory retrieval, versioned skill generation, evaluation
gates, and automatic rollback.

## Architecture

```text
Task -> AgentRunner -> Model decision -> Tool -> Observation -> Final answer
             |                         |                    |
             +---------- SQLite trajectory store ----------+
                                      |
                            Eval runner / MCP tools
```

`AgentRunner` is intentionally a small state machine rather than a wrapper around an agent
framework. Every transition is persisted, so a benchmark failure can be traced to either a model
decision, a tool execution, or the final answer.

## Quick start

Python 3.9 or later is required. The runtime has no third-party dependencies.

```bash
PYTHONPATH=src python3 -m adaptive_agent_lab.cli eval
PYTHONPATH=src python3 -m adaptive_agent_lab.cli runs --limit 5
```

Inspect a run using the ID printed by the evaluation:

```bash
PYTHONPATH=src python3 -m adaptive_agent_lab.cli show <run-id>
```

Run the standard-library tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

If `pytest` is installed, `pytest` also works.

## Why the baseline model is deterministic

The first benchmark uses a rule-based model so runtime behavior can be tested without API cost,
network access, or model drift. This does not pretend to be an intelligent agent. It isolates the
orchestration and observability layers before a real LLM is introduced in Phase 2.

## Roadmap

- Phase 2: OpenAI-compatible model provider and episodic/semantic memory.
- Phase 3: Generate versioned skills from failed trajectories.
- Phase 4: Evaluation gate, canary release, and automatic rollback.
- Phase 5: Dashboard, security policy, deployment, and portfolio demo.

Inspired by the learning-loop ideas in Nous Research's Hermes Agent. This project is an independent
educational implementation and does not copy Hermes source code.
