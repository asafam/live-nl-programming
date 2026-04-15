# NoviCode Personal Assistant Dataset

Expanded workflow dataset derived from the [NoviCode](https://huggingface.co/datasets/biu-nlp/NoviCode) research corpus — a collection of natural language programming utterances by non-programmer users targeting personal assistant APIs (Siri/Alexa-style).

## Overview

| Field | Value |
|---|---|
| Examples | 137 |
| Source | `biu-nlp/NoviCode` on HuggingFace |
| Domains | 10 (clock, calendar, reminders, messaging, weather, navigation, music, shopping, smart home, events) |
| Actors per example | 1–3 (75% have exactly 2) |
| Steps per example | 5–9 (avg 6.6) |
| File | `raw/personal_assistant_samples.yaml` |

## Purpose

Each example pairs a **seed utterance** (the original NL user request from NoviCode) with a **multi-step workflow** showing how a distributed personal assistant system — composed of domain-specific LLM-objects — would handle the request.

The expanded workflows serve as input to the data generation pipeline:
- **Stage 1**: Grounding + LLM-object system design
- **Stage 2**: Test case generation (modifications + events)
- **Stage 3**: Evaluation via the LNL runtime

## Schema

```yaml
- id: rainy-morning-alarm-weather-conditional
  name: "Rainy Morning Alarm Adjuster"
  source_type: "NoviCode/PersonalAssistant"
  seed_utterance: "If it's raining tomorrow morning, set my alarm for 7:30..."
  actors:
    - name: WeatherAgent
      domain: weather
    - name: ClockAgent
      domain: clock
  raw_steps:
    - "WeatherAgent checks tomorrow morning's forecast for Chicago..."
    - "WeatherAgent finds rain is expected between 6 AM and 9 AM..."
    - "WeatherAgent notifies ClockAgent of the rainy forecast..."
    - "ClockAgent sets the alarm for 7:30 AM..."
    - "ClockAgent confirms: alarm set for 7:30 AM tomorrow (rainy day)."
```

### Fields

| Field | Description |
|---|---|
| `id` | Kebab-case identifier (unique) |
| `name` | Human-readable automation title |
| `source_type` | Always `"NoviCode/PersonalAssistant"` |
| `seed_utterance` | Original NL utterance from the NoviCode dataset (ground truth) |
| `actors` | List of domain agents involved (name + domain) |
| `raw_steps` | 5–9 narrative steps showing the actor handoff flow |

## Actor Catalog

| Agent | Domain | Count |
|---|---|---|
| MessagingAgent | messaging | 57 |
| WeatherAgent | weather | 38 |
| CalendarAgent | calendar | 35 |
| RemindersAgent | reminders | 31 |
| ClockAgent | clock | 24 |
| SmartHomeAgent | smart_home | 17 |
| NavigationAgent | navigation | 16 |
| ShoppingAgent | shopping | 14 |
| EventsAgent | events | 12 |
| MusicAgent | music | 10 |

## Domain Distribution

| Domain | Examples |
|---|---|
| messaging | 57 |
| weather | 38 |
| calendar | 35 |
| reminders | 31 |
| clock | 24 |
| smart_home | 17 |
| navigation | 16 |
| shopping | 14 |
| events | 12 |
| music | 10 |

## Difference from Zapier Templates

| Aspect | Zapier (`data/zapier/`) | NoviCode (`data/novicode/`) |
|---|---|---|
| User type | Developer / business user | Non-programmer |
| Request style | Multi-app automation (Slack → ClickUp) | Personal assistant (Siri/Alexa) |
| `link` field | Zapier template URL | Not present — replaced by `seed_utterance` |
| `actors` field | Not present | Explicit list of domain agents |
| Complexity | Enterprise workflows, 7–9 steps | Personal tasks, 5–7 steps |
| Domains | CRM, ticketing, HR, sales | Weather, calendar, messaging, home |

## Running the Pipeline

```bash
# Content validity check (LLM judge)
python -m src.data.validate_novicode \
    --input data/novicode/raw/personal_assistant_samples.yaml \
    --model claude-opus-4-6

# Full pipeline (Stage 1 + 2)
python -m src.data.pipeline \
    -i data/novicode/raw/personal_assistant_samples.yaml \
    --target-dir outputs/novicode/run1

# Evaluation (Stage 3)
python -m src.data.evaluate \
    -i outputs/novicode/run1/test_cases.jsonl \
    --runs 3
```

## Quality Notes

The `raw_steps` in this dataset have been through multiple LLM-based audit passes covering:
- Faithfulness to the original seed utterance
- Naturalness (no API names, condition codes, GPS coordinates, device IDs)
- Actor-step consistency (no agent acting without being declared)
- Correct agent types (EventsAgent for tickets/concerts, not ShoppingAgent)
- Explicit user confirmation before any purchase or booking
- Date/day-of-week correctness (April 2026 calendar)
