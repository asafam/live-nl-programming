# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
# Create venv (one-time)
[ -d .venv ] || uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Requires a `.env` file with `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`.

## Common Commands

```bash
# Run interactive actor system
python -m src.app --provider openai --model gpt-4o-mini
python -m src.app --provider anthropic --model claude-3-5-sonnet-latest

# Run tests
pytest tests/ -v
pytest tests/test_mediator.py::test_mediator_basic -v  # single test

# Data generation pipeline (two stages)
python -m src.data.generate_samples -i data/zapier/raw/templates.yaml --samples-per-template 1
python -m src.data.generate_test_cases -i outputs/data/zapier/generated/samples.jsonl --scenario-count 1
```

## Architecture

This project has two main subsystems:

### Actor System (`src/system/`)

An actor-based message bus where actors communicate via natural language. All domain logic is LLM-driven, not hardcoded.

- **MessageBus** (`message_bus.py`) — Routes REQUEST, EVENT, and RESPONSE messages between actors. Supports pub/sub subscriptions by topic and broadcast.
- **CoordinatorActor** (`actors/coordinator_actor.py`) — The entry point actor. Dynamically creates other actors and mediators from natural language requests, routes tasks to them.
- **Actor** (`actors/base.py`) — Base class. Each actor has a name, state dict, LLM instance, and system prompt. `receive()` processes messages → LLM generates structured responses with `{response, state_updates, messages}`.
- **MediatorActor** (`actors/mediator_actor.py`) — Subscribes to events, evaluates rules semantically via LLM, dispatches requests to target actors. Holds orchestration logic, not domain data.
- **LLM clients** (`llm/`) — `AbstractLLM` interface with OpenAI and Anthropic implementations. Both support structured output via JSON schema.

**Flow:** User → Coordinator → creates actors via MessageBus → actors exchange NL messages → LLM generates structured responses → state updates applied.

### Data Generation Pipeline (`src/data/`)

Two-stage LLM pipeline generating test cases from automation templates:

- **Stage 1** (`generate_samples.py`): Raw YAML templates → concrete sample instances (JSONL)
- **Stage 2** (`generate_test_cases.py`): Samples → test cases with modifications and events (JSONL)

Key design: `mod_type` and `ambiguity` are **script-controlled**, not LLM-generated. The LLM produces `GeneratedModification` (id, when, intent only). The script assigns `mod_type` and `ambiguity` during `scenario_to_test_case` conversion. For `--mod-type mixed` or `--ambiguity random`, the script samples values per iteration.

**Schemas** (`schema.py`): `GeneratedModification` (LLM output) vs `Modification` (final output with script-assigned fields). `Scenario` uses `GeneratedModification`; `TestCase` uses `Modification`.

**Output path** is derived from input filename, mod-type, and ambiguity (e.g., `samples__temporal__vague.jsonl`).

## Configuration

- `config/system.yaml` — LLM model, temperature, seed, heartbeat, feature flags (proactivity, self_reflection)
- `config/prompts/system/` — Actor, coordinator, mediator system prompts
- `config/prompts/data-gen/` — Data generation prompt templates (use `{PLACEHOLDER}` substitution)

## Skills

- `/commit` — Creates a git commit using haiku (cheaper/faster model). Accepts optional message guidance: `/commit fix ambiguity handling`.

## Principles

- Never hardcode domain-specific logic — keep code generic, configurable, LLM-driven
- Prefer YAML configs over hardcoded values
- Maintain clean actor separation with message passing via MessageBus
- All domain behavior should be configurable or user-specified
