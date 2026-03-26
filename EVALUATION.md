# Evaluation: LNL Runtime vs OpenClaw Baseline

This document describes how to run and compare the two evaluation modes:

1. **LNL Runtime** (`evaluate`) — Multi-object paradigm where LLM-objects communicate via a message bus
2. **OpenClaw Baseline** (`evaluate_baseline`) — Single OpenClaw agent handling the entire workflow

Both use the same test cases, the same judge, and produce the same output schema for apples-to-apples comparison.

## Prerequisites

### LNL Runtime

```bash
source .venv/bin/activate
# Requires OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env
```

### OpenClaw Baseline

Requires Node 22+ and Python 3.11+.

```bash
# 1. Install Node 22 (if needed)
brew install node@22
export PATH="/opt/homebrew/opt/node@22/bin:$PATH"

# 2. Install OpenClaw
curl -fsSL https://openclaw.ai/install.sh | bash

# 3. Onboard (interactive — configure model provider + API key)
openclaw onboard --install-daemon

# 4. Verify gateway is running
openclaw gateway status

# 5. Install Python SDK
source .venv/bin/activate
uv pip install openclaw-sdk
```

## Running the Evaluations

### LNL Runtime (multi-object)

```bash
python -m src.data.evaluate \
    -i outputs/data/zapier/20260322_010211/test_cases.jsonl \
    --runs 3 \
    --model gpt-4o
```

Use a separate model for the LLM judge (e.g., a stronger model to judge a cheaper object model):

```bash
python -m src.data.evaluate \
    -i outputs/data/zapier/20260322_010211/test_cases.jsonl \
    --model gpt-4o \
    --judge-model claude-sonnet-4-6
```

Add `--verbose` / `-v` to see per-event details during the run — what the judge expected, the evidence it saw, and its reasoning:

```bash
python -m src.data.evaluate \
    -i outputs/data/zapier/20260322_010211/test_cases.jsonl \
    --model gpt-4o --verbose
```

Debug specific test cases:

```bash
# Run test case at position 3 (1-based index), full run (steps + modifications + events)
python -m src.data.evaluate \
    -i outputs/data/zapier/20260322_010211/test_cases.jsonl \
    --tc 3 --verbose --debug-messages --model gpt-4o

# Run multiple test cases by index or ID
python -m src.data.evaluate \
    -i outputs/data/zapier/20260322_010211/test_cases.jsonl \
    --tc 1 3 5 TC007 --verbose --model gpt-4o

# Run only steps (no modifications/events) for baseline behavior
python -m src.data.evaluate \
    -i outputs/data/zapier/20260322_010211/test_cases.jsonl \
    --tc 1 --steps-only --debug-messages --model gpt-4o
```

Output: `test_cases_eval.jsonl` (next to input file)

#### Mock tool execution

The LNL evaluator supports **in-process mock tools** — scripted implementations of external APIs (Slack, Email, HubSpot, Jira, etc.) that LLM-objects call directly via `tool_calls`, bypassing the message bus.

Mock tools are wired automatically when a test case has events with `triggered_by` set: the evaluator derives a `MockToolDef` for each unique tool name, and when an LLM-object calls that tool, the corresponding event input is injected into the target object via `inject_event`. All tool calls are logged and included as evidence for the judge.

For additional tool coverage, pass one or more YAML mock config files:

```bash
python -m src.data.evaluate \
    -i test_cases.jsonl \
    --model gpt-4o \
    --mock-config config/mocks/lnl/email.yaml \
    --mock-config config/mocks/lnl/slack.yaml
```

**Priority order** (highest wins on `tool_name` collision):

| Layer | Source | When to use |
|---|---|---|
| `tc.mock_tools` | Inline in each TestCase | Per-test-case scripted responses |
| `--mock-config` | Shared YAML files | Reusable boilerplate responses |
| `triggered_by` auto-derived | `Event.triggered_by` fields | Orchestration: tool call → event injection |

Any tool the LLM calls that isn't covered by the above layers hits a **PassthroughExecutor** fallback — it returns a generic success and logs the call for judge evidence, so evaluation never errors on unknown tools.

**Mock config YAML format** (`config/mocks/lnl/*.yaml`):

```yaml
tools:
  - tool_name: email.send
    description: Send an email to a recipient.
    arguments_schema:
      type: object
      properties:
        to: {type: string, description: Recipient email address}
        subject: {type: string, description: Subject line}
        body: {type: string, description: Email body}
      required: [to, subject, body]
    response_template: "Email sent to {to} (subject: '{subject}'). Message queued."
    # scripted_responses: consumed FIFO per call; {call_index} = 1-based call number
    # triggers: dispatch events to other objects when the tool fires
    triggers:
      - target_object_id: slack-notifier
        message_template: "[Email Sent] To: {to} | Subject: {subject}"
        source: email
```

#### Test case selection and debugging

**`--tc N [N2 ...]`** — Run specific test cases by 1-based index or ID:
- `--tc 3` — run test case at position 3
- `--tc 1 3 5` — run test cases 1, 3, and 5
- `--tc TC007 TC015` — run test cases by ID
- `--tc 2 TC010 5` — mix indices and IDs

Overrides `--limit`. Useful for isolating flaky or incomplete test cases.

**`--steps-only`** — Run only the steps (baseline behavior section); skip modifications and events. Useful for:
- Testing initialization behavior without scenario changes
- Debugging what happens when external systems first contact the objects
- Verifying that seed_data and object definitions are correct before applying test logic

**`--debug-messages`** — Print all messages flowing through the message bus, including JSON envelopes for external events and internal peer communication. Shows sender, recipient, message type, and content.

### OpenClaw Baseline (single agent)

```bash
python -m src.data.evaluate_baseline \
    -i outputs/data/zapier/20260322_010211/test_cases.jsonl \
    --mock-server \
    --runs 3
```

Output: `test_cases_baseline.jsonl` (next to input file)

#### Mock external system integration (`--mock-server`)

When `--mock-server` is passed, the runner automatically:

1. **Starts a MockServer** (FastAPI, `localhost:18888`) before the first test case and stops it after the last.
2. **Per test case**: scans object `skills` and `event_sources` for system keywords (`slack`, `email`, `jira`, `webhook`) and loads matching scripts from `config/mocks/`.
3. **Per test case**: builds orchestration triggers from events that have `triggered_by` set — these events are injected into the agent session only when the agent actually calls the corresponding tool, not unconditionally.
4. **The OpenClaw plugin** (`plugins/openclaw-mock-external`) must be installed once beforehand — it registers the mock tools with the OpenClaw agent.

**One-time plugin install:**
```bash
cd plugins/openclaw-mock-external && npm install && npm run build
openclaw plugin install .
```

**Mock system scripts** (`config/mocks/`): define boilerplate immediate responses per tool (delivery ACKs, channel listings, ticket IDs). Generic across test cases.

**Orchestration** is derived from `Event.triggered_by` fields in each test case — the test-case-specific content (what Slack/email says back) lives in `Event.input`.

**`time_scale`** compresses simulated delays: `0.01` means 1 simulated minute = 0.6 real seconds. Configured in `config/mocks/orchestration/*.yaml` or set on `OrchestratorScript`.

```bash
# Script-driven (default)
python -m src.data.evaluate_baseline -i test_cases.jsonl --mock-server

# LLM-powered mock responses
python -m src.data.evaluate_baseline -i test_cases.jsonl --mock-server --mock-llm-mode

# Custom OpenClaw gateway
python -m src.data.evaluate_baseline -i test_cases.jsonl --mock-server --openclaw-http-url http://localhost:18789
```

## CLI Flags

| Flag | LNL (`evaluate`) | Baseline (`evaluate_baseline`) |
|---|---|---|
| `--input`, `-i` | Test cases JSONL (required) | Test cases JSONL (required) |
| `--output`, `-o` | Output path | Output path |
| `--runs` | Runs per test case (default: 1) | Runs per test case (default: 1) |
| `--timeout` | Wall-clock seconds per step/event (default: 60) | Seconds per run (default: 120) |
| `--model`, `-m` | Model for LLM-objects | N/A (configured in OpenClaw) |
| `--provider`, `-p` | `openai` or `anthropic` | N/A |
| `--judge-model` | Judge model (default: same as `--model`; strongly prefer a separate stronger model) | Judge model (default: `gpt-4o-mini`) |
| `--judge-provider` | Judge provider (inferred from model name) | Judge provider (default: `openai`) |
| `--verbose`, `-v` | Print per-event evidence, expected, and judge reasoning | N/A |
| `--agent-id` | N/A | OpenClaw agent ID (default: `lnl-baseline`) |
| `--gateway-url` | N/A | OpenClaw gateway URL (default: auto-detect) |
| `--limit`, `-n` | First N test cases only | First N test cases only |
| `--tc` | Specific test cases by 1-based index or ID (overrides `--limit`) | N/A |
| `--steps-only` | Run only steps; skip modifications and events | N/A |
| `--debug-messages` | Print messages exchanged between LLM-objects | N/A |
| `--mock-config` | YAML file(s) with shared mock tool definitions (repeatable) | N/A |

## Comparing Results

The last line of each output JSONL is an `EvalSummary` with aggregate metrics:

```bash
tail -1 outputs/.../test_cases_eval.jsonl | python -m json.tool
tail -1 outputs/.../test_cases_baseline.jsonl | python -m json.tool
```

Key metrics in `EvalSummary`:

| Metric | Description |
|---|---|
| `mean_pass_rate` | Average correctness across all test case runs |
| `pass_rate_std` | Behavioral consistency (std dev across runs per test case) |
| `mean_event_input_tokens` | Average input tokens per event |
| `mean_event_output_tokens` | Average output tokens per event |
| `mean_event_latency_ms` | Average latency per event |
| `mean_mod_input_tokens` | Average input tokens per modification |
| `mean_mod_output_tokens` | Average output tokens per modification |
| `mean_mod_latency_ms` | Average latency per modification |

Per-test-case results (`TestCaseResult` lines) include per-event pass/fail with reasoning, token costs, the `expected` assertion condition, and the `evidence` text that was presented to the judge.

## How They Differ

**LNL Runtime**: Creates separate LLM-objects for each component (e.g., `hubspot`, `quote-approvals`, `slack`). Each object has its own system prompt, state, and conversation history. Messages route through a `MessageBus` — when one object processes an event, it can send messages to peers, triggering a chain of LLM calls.

**OpenClaw Baseline**: A single OpenClaw agent receives ALL component definitions in one system prompt. Steps, modifications, and events are sent as sequential messages in one conversation. The agent tracks all state internally and describes what actions it took.

The judge is identical: an LLM evaluates whether the expected assertion holds given the observable evidence (replies, actions, state).
