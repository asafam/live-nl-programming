# Mediator Actor

## Overview

The **Mediator Actor** is a specialized actor type that functions as a rule engine for orchestrating interactions between other actors in the live NL programming system. Unlike domain actors that hold data, Mediators hold **LOGIC** (rules) and translate signals into requests.

## Key Characteristics

- **Stateless for Domain Data**: Does not track domain state
- **Rule-Based**: Holds a list of trigger-action rules
- **Event-Driven**: Listens to signals from specified actors
- **Request Generator**: Translates matched signals into requests to target actors
- **Compensating Transactions**: Can handle rollback logic when target actors fail

## Architecture

```
┌─────────────┐         EVENT          ┌──────────────┐
│   Source    │ ──────────────────────> │   Mediator   │
│   Actor     │                         │  (Rule Eng.) │
│  (ActorA)   │                         │              │
└─────────────┘                         └──────────────┘
                                               │
                                               │ REQUEST
                                               │ (based on rule)
                                               ▼
                                        ┌──────────────┐
                                        │   Target     │
                                        │   Actor      │
                                        │  (ActorB)    │
                                        └──────────────┘
```

## State Structure

```json
{
  "purpose": "Orchestrate synchronization between ActorA and ActorB",
  "listening_to": ["ActorA", "ActorC"],
  "rules": [
    {
      "trigger_condition": "Data changed in ActorA",
      "action_instruction": "Update corresponding data in ActorB",
      "target_actor": "ActorB"
    }
  ],
  "constraints": []
}
```

## Usage

### Creating a Mediator

```python
from src.actors.mediator_actor import MediatorActor
from src.llm.openai_client import OpenAIChatLLM

# Define initial rules
initial_rules = [
    {
        "trigger_condition": "Data changed in source actor",
        "action_instruction": "Update corresponding data in target actor",
        "target_actor": "TargetActor"
    }
]

# Create mediator
mediator = MediatorActor(
    name="DataSyncOrchestrator",
    llm=OpenAIChatLLM(),
    purpose="Orchestrate data synchronization between actors",
    initial_rules=initial_rules,
    listening_to=["SourceActor"],
    enable_self_reflection=False
)

# Register with message bus
bus.register(mediator)
```

### Managing Rules Dynamically

```python
# Add a rule
mediator.add_rule(
    trigger_condition="Temperature exceeds 100 degrees",
    action_instruction="Send alert to monitoring system",
    target_actor="AlertManager"
)

# Remove a rule by index
mediator.remove_rule(0)

# Add listening target
mediator.add_listening_target("TemperatureSensor")

# Remove listening target
mediator.remove_listening_target("TemperatureSensor")
```

## How It Works

### 1. Signal Processing

When a Mediator receives an EVENT message:

1. **Semantic Matching**: Compares the event content against all `trigger_condition` rules
2. **Pattern Recognition**: Uses LLM for semantic understanding (doesn't require exact wording)
3. **Multi-Match**: Can trigger multiple rules if multiple conditions match

Example:
- Signal: `"Updated field X to value Y"`
- Rule trigger: `"Data changed in source actor"`
- **Match**: Yes (semantic equivalence)

### 2. Request Generation

When a rule matches:

1. **Extract Details**: Parse relevant information from the signal
2. **Formulate Request**: Create a specific request with context
3. **Route to Target**: Send REQUEST to the target actor specified in the rule

Example:
```
Signal: "Updated field X to value Y"
Rule: {trigger: "Data changed", action: "Sync data", target: "TargetActor"}
→ Request to TargetActor: "Based on source update, please update your field X to value Y"
```

### 3. Failure Handling

When a target actor fails or rejects:

1. **Detect Failure**: Recognize failure/rejection in response
2. **Find Compensation**: Look for rollback rule
3. **Execute Compensation**: Send compensating request
4. **Notify User**: Report failure if no compensation available

Example:
```
TargetActor responds: "Cannot apply update"
→ Mediator sends to SourceActor: "Rollback the last change due to target rejection"
→ Mediator notifies User: "Update could not be synchronized - rollback initiated"
```

## Message Types

### Incoming Messages

- **EVENT**: Signals from watched actors that may trigger rules
- **REQUEST**: Commands to modify rules, listening config, or handle specific scenarios

### Outgoing Messages

- **REQUEST**: Most common - sends commands to target actors when rules match
- **RESPONSE**: Acknowledges signals that don't match rules or confirms rule updates

**Important**: Mediators do NOT send EVENT messages. They translate events into requests.

## Use Cases

### 1. Data Synchronization

```python
rules = [{
    "trigger_condition": "Data added to source",
    "action_instruction": "Add corresponding data to target",
    "target_actor": "TargetActor"
}, {
    "trigger_condition": "Data removed from source",
    "action_instruction": "Remove corresponding data from target",
    "target_actor": "TargetActor"
}]
```

### 2. Alert Management

```python
rules = [{
    "trigger_condition": "Temperature exceeds threshold",
    "action_instruction": "Send critical alert",
    "target_actor": "AlertSystem"
}, {
    "trigger_condition": "System error detected",
    "action_instruction": "Log error and notify administrator",
    "target_actor": "LoggingService"
}]
```

### 3. Workflow Orchestration

```python
rules = [{
    "trigger_condition": "Task initiated",
    "action_instruction": "Begin processing",
    "target_actor": "ProcessorActor"
}, {
    "trigger_condition": "Processing completed",
    "action_instruction": "Update status and notify completion",
    "target_actor": "StatusActor"
}]
```

## Benefits

1. **Separation of Concerns**: Domain actors focus on data, mediators focus on coordination
2. **Flexible Orchestration**: Rules can be added/removed/modified dynamically
3. **Natural Language**: All rules are expressed in natural language
4. **LLM-Powered**: Semantic matching handles variations in event descriptions
5. **Composable**: Multiple mediators can handle different aspects of orchestration
6. **Testable**: Rule evaluation is deterministic given the same inputs

## Testing

See [tests/test_mediator.py](../tests/test_mediator.py) for examples:

- `test_mediator_basic()`: Demonstrates basic rule-based orchestration
- `test_mediator_rule_management()`: Shows dynamic rule manipulation

## Configuration

The Mediator uses the standard actor reflection prompt from `config/prompts/actor.yaml` but has its own specialized system prompt defined in `config/prompts/mediator.yaml`.

## API Reference

### MediatorActor Class

**Constructor**:
```python
MediatorActor(
    name: str,
    llm: AbstractLLM,
    purpose: str,
    initial_rules: Optional[List[Dict]] = None,
    listening_to: Optional[List[str]] = None,
    enable_self_reflection: bool = False
)
```

**Methods**:
- `add_rule(trigger_condition, action_instruction, target_actor)`: Add a new rule
- `remove_rule(rule_index)`: Remove a rule by index
- `add_listening_target(actor_name)`: Start listening to an actor
- `remove_listening_target(actor_name)`: Stop listening to an actor

**Properties**:
- `state`: Current mediator state (rules, listening_to, etc.)
- `system_prompt`: Generated system prompt with current state

## Future Enhancements

Potential improvements:

1. **Rule Priorities**: Add priority levels to rules for conflict resolution
2. **Conditional Logic**: Support AND/OR conditions in trigger patterns
3. **Time-Based Rules**: Trigger rules based on time elapsed or schedules
4. **Rule Chains**: Support multi-step orchestrations
5. **Rule Templates**: Pre-defined rule patterns for common scenarios
6. **Rule Validation**: Validate rules before adding them
7. **Rule Metrics**: Track rule execution frequency and success rates
