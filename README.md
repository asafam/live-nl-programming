# Live NL Prog Shopping List

Concise overview of the actor-based, natural-language message bus demo with tool calls.

## Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI and/or Anthropic API keys

## Setup

1. Create the virtual environment (only needed once):
   ```bash
   # Skip if directory already exists
   [ -d live-nl-programming ] || uv venv live-nl-programming
   ```

2. Activate the environment (run in each new shell):
   ```bash
   source live-nl-programming/bin/activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```bash
   cat > .env << 'EOF'
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   EOF
   ```

## Run

```bash
# OpenAI default model
python -m src.app --provider openai --model gpt-4o-mini

# Anthropic
python -m src.app --provider anthropic --model claude-3-5-sonnet-latest

# Custom base URL for OpenAI-compatible endpoints
python -m src.app --provider openai --model gpt-4o-mini --openai-base-url https://api.openai.com/v1
```

## How it works

- `MessageBus` delivers messages between actors.
- `ToolExecutor` implements two tools: `create_actor` and `send_message`.
- `CoordinatorActor` is the only concrete actor; the LLM can request tools to spin up new actors and route messages.
- `Actor` exposes a default `send_message` tool so any actor can ask the bus to deliver NL messages.
- `OpenAIChatLLM` surfaces tool calls via `tool_calls`, and actors dispatch them through `ToolExecutor`.

## Demo script (example)

You can interactively drive the coordinator with natural language prompts like:

- "Create an actor Shopper to handle buying items." (LLM should call `create_actor`)
- "Send to Shopper: please list items we need for pasta dinner." (LLM should call `send_message`)
- "Send to Shopper: update the list with 2 boxes of pasta and 1 jar of sauce." (routes via bus)

The coordinator maintains lightweight metadata of created actors in state. All communication stays in natural language; tool calls are handled automatically when the LLM requests them.
