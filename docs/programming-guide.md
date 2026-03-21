# LNL Programming Guide

A practical guide to building programs in the Live Natural Language Programming paradigm.

## What is an LNL Program?

An LNL program is a collection of **LLM-objects** that communicate via natural language messages through a **message bus**. Each object has a definition (what it is), state (what it knows), and a brain (an LLM that processes messages). The key property: **definitions can change at runtime while state persists**.

There is no hardcoded logic. All behavior is described in natural language and executed by the LLM.

## Creating LLM-Objects

### From the API

```python
from src.lnl import Runtime, OpenAIBrain, ObjectDefinition, PeerDeclaration

brain = OpenAIBrain(model="gpt-4o-mini")
rt = Runtime(brain, strict_peers=False)

rt.create_object(ObjectDefinition(
    object_id="shopping-list",
    role="Manages a shopping list of items with quantity, name, and unit price.",
    state_description="Track current items (each with name, quantity, unit price).",
    behavior=(
        "When items are added, append them to the list. "
        "When items are removed, remove them."
    ),
    peers=[PeerDeclaration("budget-manager", "Tracks spending against a budget")],
))
```

### From Markdown Files

Objects can be defined in `.md` files:

```markdown
# Shopping List

## Role

Manages a shopping list of items with quantity, name, and unit price.

## State

Track current items (each with name, quantity, unit price).

## Behavior

When items are added, append them to the list.
When items are removed, remove them.

## Peers

- budget-manager: Tracks spending against a budget
```

Load a single file or an entire directory:

```python
rt.load_file("programs/shopping-list.md")
rt.load_directory("programs/")
```

### Definition Fields

| Field | Required | Purpose |
|-------|----------|---------|
| `object_id` | Yes | Unique identifier (slug format: `shopping-list`) |
| `role` | Yes | What this object is and does |
| `state_description` | No | What the object should track in its state |
| `behavior` | No | Rules for how the object responds to messages |
| `peers` | No | Other objects this one can communicate with |
| `skills` | No | Named capabilities the object has |
| `subscriptions` | No | Topics this object listens to |

## Writing Effective Definitions

### Role

Keep it concise — one or two sentences describing the object's purpose:

```
Manages a shopping list of items with quantity, name, and unit price.
Enforces any dietary constraints when adding items.
```

### State Description

Tell the LLM **what** to track, not **how**. The LLM decides the format:

```
Track two things: (1) current items, each with name, quantity, and unit price;
(2) a list of active dietary constraints. Always include both sections, even when empty.
```

The more specific you are about structure, the more consistent the state will be across interactions.

### Behavior

Describe rules as **when/then** patterns. Be explicit about what should trigger outgoing messages:

```
When items are added, append them to the list and send exactly ONE message
to budget-manager with ONLY the newly added items and their total cost.
When a dietary constraint is received, add it to the constraints list AND
remove any conflicting items. Always keep constraints for future enforcement.
```

Key principles:
- **Be explicit about message content** — say what the outgoing message should contain
- **Specify incremental vs. full** — "send only the change, not the full list"
- **Prevent loops** — "Never send outgoing messages" for passive trackers
- **State is source of truth** — the LLM's current state already reflects all prior interactions

### Peers

Peers are **directional** — they declare which objects this one can **send messages to**. An object can always *receive* messages from others without declaring them as peers.

```python
# Shopping list sends notifications TO budget-manager
rt.create_object(ObjectDefinition(
    object_id="shopping-list",
    role="Manages a shopping list.",
    peers=[PeerDeclaration("budget-manager", "Tracks spending against a budget")],
))

# Budget manager receives from shopping-list but never sends back — no peers needed
rt.create_object(ObjectDefinition(
    object_id="budget-manager",
    role="Passively tracks a household budget.",
))
```

If an object has **no peers**, it cannot send outgoing messages — making it a passive receiver by design.

When `strict_peers=True` (the default), the bus blocks domain messages to non-peers. Messages from `__user__` and `__system__` always bypass peer validation.

## Sending Messages

```python
# User sends to an object
results = rt.send("shopping-list", "Add 3 apples at $2 each")

# System event
results = rt.send_event("email-monitor", "New email from Sarah: going vegan!")

# Broadcast to all
results = rt.broadcast("System shutting down in 5 minutes")

# Publish to topic subscribers
results = rt.publish("dietary-updates", "New constraint: no dairy")
```

Every `send()` returns a list of `ProcessingResult` — one per object that processed a message, including any chained responses:

```python
for r in results:
    print(f"[{r.object_id}] {r.reply}")
    print(f"  State: {r.state_after}")
```

### Message Chains

If object A sends a message to B, and B's response includes an outgoing message to C, the bus delivers to C before returning. A single `send()` call returns results from the entire chain:

```
User -> shopping-list -> budget-manager
         result[0]         result[1]
```

The chain depth limit (default 10) prevents infinite loops.

## Modifying Objects at Runtime

The core "live" property — change an object's definition while its state persists:

```python
# Change behavior mid-session
rt.modify("shopping-list", behavior="Also suggest recipe ideas when items are added.")

# Add a new peer relationship
rt.add_peer("shopping-list", "recipe-bot", "Suggests recipes")

# Remove a peer
rt.remove_peer("shopping-list", "recipe-bot")
```

The object keeps its accumulated state and history. On the next message, the LLM sees the updated definition.

### Save and Reload

Modifications are in-memory until explicitly saved:

```python
rt.has_unsaved_modifications("shopping-list")  # True
rt.save_object("shopping-list", "programs/shopping-list.md")
rt.has_unsaved_modifications("shopping-list")  # False
```

## Inspecting State

```python
# Current NL state string
print(rt.state("shopping-list"))

# Full debug snapshot
print(rt.snapshot("shopping-list"))

# Communication graph
print(rt.topology())
# {'shopping-list': ['budget-manager'], 'budget-manager': ['shopping-list']}

# Message log
for entry in rt.message_log:
    print(f"{entry.message.sender} -> {entry.message.recipient}: {entry.message.content[:50]}")

# Aggregate metrics
print(f"Messages routed: {rt.metrics.messages_routed}")
```

## Common Patterns

### Notifier → Tracker

One object does work and notifies a passive tracker:

```python
# Shopping list: active, has budget-manager as peer → sends updates
peers=[PeerDeclaration("budget-manager", "Tracks spending")]
behavior="After adding items, notify budget-manager with the cost."

# Budget manager: passive, has NO peers → cannot send outgoing messages
peers=[]  # or simply omit
behavior="Update spent/remaining when notified."
```

The passive tracker pattern is enforced by **not declaring any peers** — no peers means no outgoing messages.

### Monitor → Actor

An external event monitor forwards relevant information:

```python
# Email monitor: filters and forwards
behavior="When an email arrives with dietary info, forward it to shopping-list."

# Shopping list: reacts to forwarded events
behavior="When a dietary constraint arrives, enforce it on the list."
```

### Topic Pub/Sub

Objects subscribe to topics for decoupled communication:

```python
rt.create_object(ObjectDefinition(
    object_id="logger",
    role="Logs all order events.",
    subscriptions=["order-events"],
))

# Any object can publish to the topic
rt.publish("order-events", "Order #123 completed")
```

## Debugging Tips

- Use `rt.snapshot(object_id)` to see full object state including definition and history length
- Check `rt.message_log` to see what was delivered, what was blocked, and why
- If an object's state drifts, check its `state_description` — more specific descriptions produce more consistent state
- If chained messages cause duplicates, add "Never send outgoing messages" to passive objects
- If the LLM re-processes history, check that your `behavior` says "only act on the new message"

---

## Using the CLI

The CLI provides interactive access to the runtime from the terminal.

### Setup

```bash
source .venv/bin/activate
```

All CLI commands follow the pattern:
```bash
python -m src.lnl.cli --provider <provider> [--model <model>] <command> [args]
```

Providers: `openai`, `anthropic`, `mock`

### Loading Objects

```bash
# Load all .md files from a directory
python -m src.lnl.cli --provider openai load programs/

# Load a single file
python -m src.lnl.cli --provider openai new programs/shopping-list.md
```

### Sending Messages

```bash
# Send a message to an object
python -m src.lnl.cli --provider openai send shopping-list "Add 3 apples at $2 each"

# Send an event
python -m src.lnl.cli --provider openai event email-monitor "New email from Sarah: going vegan"
```

### Inspecting State

```bash
# View current state
python -m src.lnl.cli --provider openai state shopping-list

# Full snapshot (JSON)
python -m src.lnl.cli --provider openai snapshot shopping-list

# Communication topology
python -m src.lnl.cli --provider openai topology

# Message log
python -m src.lnl.cli --provider openai log
```

### Modifying Objects

```bash
python -m src.lnl.cli --provider openai modify shopping-list \
    --role "Senior shopping list manager" \
    --behavior "Also suggest substitutes for out-of-stock items"
```

### Saving

```bash
python -m src.lnl.cli --provider openai save shopping-list --path out/shopping-list.md
```

### Running Benchmarks

```bash
# Run a single scenario
python -m src.lnl.cli --provider openai run scenarios/proactive-event/

# Run all scenarios in a directory
python -m src.lnl.cli --provider openai run scenarios/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `openai` | LLM provider (`openai`, `anthropic`, `mock`) |
| `--model` | provider default | Model name |
| `--strict-peers` | off | Enforce peer validation on domain messages |
| `--max-chain-depth` | 10 | Maximum message chain depth |
