# Mock Server

A lightweight HTTP server that simulates external tool APIs (Slack, email, Jira, etc.) for agent evaluation. Drop it into any project and run it — no configuration required.

## Quick start

```bash
# Copy to your project
cp -r mock/ /your-project/mock/

# Install dependencies
pip install -r mock/requirements.txt

# Run
python mock/server.py
# → MockServer ready on port 18888
```

## Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/health` | Readiness probe — returns `{"status":"ok"}` |
| `POST` | `/configure` | Set session key and mock script before a run |
| `GET` | `/log` | Retrieve recorded tool call log |
| `POST` | `/tool/{method}` | Receive a tool invocation, return a scripted response |

## How it works

Any client (gateway plugin, test harness, etc.) routes tool calls to `POST /tool/{method}`. The server looks up the method in the active **MockScript**, interpolates the response template with the call arguments, and returns it. Optionally it injects a follow-up message back into the agent session via OpenClaw's `/hooks/wake`.

### Configure a run

```bash
curl -X POST http://localhost:18888/configure \
  -H 'Content-Type: application/json' \
  -d '{
    "session_key": "session-abc",
    "slot_id": "default",
    "mock_script": {
      "systems": [{
        "system": "slack",
        "tools": [{
          "method": "slack_send_message",
          "immediate": {"template": "Message {tool_call_id} delivered to {channel}"}
        }]
      }]
    }
  }'
```

### Call a tool

```bash
curl -X POST http://localhost:18888/tool/slack_send_message \
  -H 'Content-Type: application/json' \
  -d '{"channel": "#general", "text": "Hello"}'
# → {"status": "ok", "result": "Message a1b2c3d4 delivered to #general"}
```

### Retrieve the call log

```bash
curl http://localhost:18888/log
# → {"calls": [{"method": "slack_send_message", "args": {...}, "result": "...", ...}]}
```

## Built-in system configs

`config/` contains ready-made mock definitions for common external systems. They are loaded automatically by the evaluation harness based on keyword matching — no setup needed.

| File | System |
|------|--------|
| `slack.yaml` | Slack messaging |
| `email.yaml` | Email / Gmail |
| `jira.yaml` | Jira issue tracking |
| `github.yaml` | GitHub |
| `stripe.yaml` | Stripe payments |
| `google_calendar.yaml` | Google Calendar |
| `google_sheets.yaml` | Google Sheets |
| `hubspot.yaml` | HubSpot CRM |
| `salesforce.yaml` | Salesforce |
| `airtable.yaml` | Airtable |
| `asana.yaml` | Asana |
| `notion.yaml` | Notion |
| `monday.yaml` | Monday.com |
| `twilio.yaml` | Twilio SMS |
| `zapier.yaml` | Zapier |
| `generic_webhook.yaml` | Generic webhook |

## Concurrent slot isolation

For parallel test runs, pass a `slot_id` to keep sessions isolated:

```json
{"__slot_id__": "tc-42", "__session_key__": "session-abc", "channel": "#alerts"}
```

Each slot gets its own call log, mock script, and orchestration state.

## Orchestration

Place YAML scripts in `config/orchestration/` to define event chains — when a tool fires, automatically inject follow-up messages into the agent session after a configurable delay.

```yaml
name: approval-flow
time_scale: 0.01          # 1 sim-minute = 0.6 real seconds
triggers:
  - tool: email_send
    match:
      subject: ".*[Aa]pproval.*"
    reactions:
      - source: slack
        message: "@manager approved the request"
        after_minutes: 2
```

## CLI options

```
python mock/server.py [--port PORT] [--openclaw-url URL]

  --port           Port to listen on (default: 18888)
  --openclaw-url   OpenClaw gateway URL for wake callbacks
                   (default: http://localhost:18789; only needed for callbacks)
```

## Python API

Use `MockServer` directly in tests or evaluation scripts:

```python
from mock.server import MockServer
from mock.schema import MockScript, MockSystemDef, MockMethodDef, MockImmediateResponse

script = MockScript(systems=[
    MockSystemDef(system="slack", tools=[
        MockMethodDef(method="slack_send_message",
                      immediate=MockImmediateResponse(template="ok: {tool_call_id}"))
    ])
])

server = MockServer(mock_script=script, port=18888)
server.start()
server.wait_ready()

# ... run your test ...

log = server.get_log()
server.stop()
```
