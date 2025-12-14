# Live NL Prog Shopping List

Concise overview of the actor-based, natural-language message bus demo with tool calls.

## Setup

1. Install deps (needs `openai`, `anthropic`, `python-dotenv`):
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` with your API keys:
   ```bash
   echo 'OPENAI_API_KEY=sk-...' >> .env
   echo 'ANTHROPIC_API_KEY=ak-...' >> .env
   # optional: OPENAI_BASE_URL=https://api.openai.com/v1
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
