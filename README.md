# Adaptive Agent Lab

[![CI](https://github.com/XiangWang423/adaptive-agent-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/XiangWang423/adaptive-agent-lab/actions/workflows/ci.yml)

Adaptive Agent Lab is a small, observable agent runtime built to answer a practical question:
can an agent learn from failed trajectories without silently regressing on tasks it already solves?

Phase 1 establishes the measurable baseline: a framework-free agent loop, typed tools, SQLite
trajectories, deterministic and live-model evaluations, trace inspection, OpenAI/OpenRouter
provider adapters, and a Codex MCP/Skill interface. Phase 2 adds opt-in retrieval of similar
recovered trajectories. Later phases will add versioned skill generation, evaluation gates, and
automatic rollback.

## Architecture

```text
Task -> AgentRunner -> Model decision -> Tool -> Observation -> Final answer
             |                         |                    |
             +---------- SQLite trajectory store ----------+
              |                       |
              +-- trajectory memory <-+-- Eval runner / MCP tools
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

## Run with the OpenAI Responses API

The real-model provider is optional, so the deterministic benchmark and CI do not require an API
key or network access. Install the provider extra when you want to run a live task:

```bash
python3 -m pip install -e '.[openai]'
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="your-supported-model-id"
adaptive-agent-lab run "Count the words in: reliable agents need good traces"
```

The provider sends the registered tools as function definitions, converts function calls into the
runtime's `Action` objects, executes them through `AgentRunner`, and returns
`function_call_output` with the matching `call_id`. It uses `previous_response_id` to continue the
same Responses API interaction. Every action and tool result remains visible in the SQLite
trajectory store.

Never commit API keys. `.env` files are ignored by Git, and the CLI reads credentials from the
process environment through the official OpenAI SDK.

## Run with OpenRouter

The OpenRouter adapter uses its OpenAI-compatible Chat Completions API while keeping orchestration
inside `AgentRunner`. Unlike the OpenAI Responses adapter, it sends the complete user/action/tool
history on every turn.

```bash
python3 -m pip install -e '.[openai]'
read -s "OPENROUTER_API_KEY?Paste OpenRouter key: "
export OPENROUTER_API_KEY
echo

PYTHONPATH=src python3 -m adaptive_agent_lab.cli \
  --db /tmp/openrouter-run.db \
  run \
  --provider openrouter \
  --model openai/gpt-5-nano \
  --max-steps 2 \
  "Count the words in: I love building agents"
```

`--max-steps` is a cost and safety guard. It caps model decisions, including retries after tool
errors.

## Recall recovered trajectories

Add `--memory` to a live `run` command to retrieve similar completed runs that first encountered a
tool error and later recovered. Retrieval uses local token overlap, so it does not make another API
request. The model receives a compact failed-call/successful-call example before its first decision,
and the recall is recorded as a `memory_recall` trajectory event.

```bash
PYTHONPATH=src python3 -m adaptive_agent_lab.cli \
  --db .adaptive-agent-lab/trajectories.db \
  run \
  --provider openrouter \
  --model openai/gpt-5-nano \
  --memory \
  "Count words after correcting invalid arguments"
```

Memory is opt-in and is not enabled inside evaluation by default. This avoids contaminating a test
set with earlier answers from the same benchmark database; a future phase will add explicit
train/evaluation memory splits and A/B metrics.

## Live evaluation

JSONL cases score final-answer correctness separately from tool selection. Reports also include
first-tool-call success, tool errors, recovery rate, average tool calls, and latency percentiles.
The numeric scorer accepts one unambiguous number inside a short response without paying for a
second model as a judge.

```bash
PYTHONPATH=src python3 -m adaptive_agent_lab.cli \
  --db /tmp/openrouter-smoke.db \
  eval-live \
  --provider openrouter \
  --model openai/gpt-5-nano \
  --cases evals/smoke_cases.jsonl \
  --max-steps 2
```

### Isolated memory evaluation

Live evaluation can read recovered trajectories from a separate historical database. The memory
database must already exist and cannot be the same file as the evaluation result database. This
prevents the current test runs from becoming their own memory and contaminating the benchmark.

Run the control group without memory:

```bash
PYTHONPATH=src python3 -m adaptive_agent_lab.cli \
  --db /tmp/memory-ab-control.db \
  eval-live \
  --provider openrouter \
  --model openai/gpt-5-nano \
  --cases evals/smoke_cases.jsonl \
  --max-steps 2 > /tmp/memory-ab-control.json
```

Run the treatment group against a database of earlier, recovered trajectories:

```bash
PYTHONPATH=src python3 -m adaptive_agent_lab.cli \
  --db /tmp/memory-ab-treatment.db \
  eval-live \
  --provider openrouter \
  --model openai/gpt-5-nano \
  --cases evals/smoke_cases.jsonl \
  --max-steps 2 \
  --memory-db .adaptive-agent-lab/trajectories.db \
  > /tmp/memory-ab-treatment.json
```

The report includes `memory_enabled`, `cases_with_memory_recall`, `memory_recall_rate`, and a
per-case recalled-memory count alongside correctness, first-call success, tool errors, tool calls,
and latency. Run control and treatment against the same held-out cases, but never use trajectories
from those held-out cases as memory.

### Evaluation-driven debugging example

On 2026-08-13, a three-case OpenRouter smoke run selected the correct tool in all three cases but
returned `capital:Tokyo` instead of the tool output `Tokyo` in one case. SQLite trajectory
inspection then exposed a second inefficiency: the model tried `country:Japan` and `entity:Japan`
before recovering with `capital:Japan`.

The fix strengthened the provider system prompt, documented the lookup key contract in its JSON
Schema, added first-attempt and recovery metrics, and introduced a two-step cost guard. A focused
regression run improved that case from three tool calls, two tool errors, and 39.4 seconds to one
tool call, zero tool errors, and 7.7 seconds. This is a single-case regression result, not a claim
about the full benchmark.

## Why the baseline model is deterministic

The first benchmark uses a rule-based model so runtime behavior can be tested without API cost,
network access, or model drift. This does not pretend to be an intelligent agent. It isolates the
orchestration and observability layers before a real LLM is introduced in Phase 2.

## Roadmap

- Phase 2: OpenAI Responses and OpenRouter providers, live evaluation, and first episodic
  trajectory retrieval (implemented); semantic retrieval and isolated memory evaluation remain.
- Phase 3: Generate versioned skills from failed trajectories.
- Phase 4: Evaluation gate, canary release, and automatic rollback.
- Phase 5: Dashboard, security policy, deployment, and portfolio demo.

Inspired by the learning-loop ideas in Nous Research's Hermes Agent. This project is an independent
educational implementation and does not copy Hermes source code.
