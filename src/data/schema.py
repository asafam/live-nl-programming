from pydantic import BaseModel
from typing import Literal
from enum import Enum

class ModType(str, Enum):
    temporal = "temporal"
    contextual = "contextual"
    exception = "exception"
    correction = "correction"
    expansion = "expansion"
    removal = "removal"

class Ambiguity(str, Enum):
    precise = "precise"
    semantic = "semantic"
    vague = "vague"
    implicit = "implicit"

class EventExpect(BaseModel):
    action: str
    reason: str

class Event(BaseModel):
    id: str
    source: str
    input: str
    when: str  # Same format as modification: "W02-1T09:00"
    expect: EventExpect

class Modification(BaseModel):
    id: str
    when: str
    mod_type: ModType
    intent: str
    ambiguity: Ambiguity

class TestCase(BaseModel):
    id: str
    name: str
    domain: str
    source_type: str
    link: str
    steps: list[str]
    modifications: list[Modification]
    events: list[Event]

class TestCases(BaseModel):
    test_cases: list[TestCase]

# Sample generation schemas
class Sample(BaseModel):
    id: str
    name: str
    domain: str
    source_type: str
    link: str
    raw_steps: list[str]
    steps: list[str]

class Samples(BaseModel):
    samples: list[Sample]

# Scenario generation schemas (LLM output before merging with instance metadata)
class Scenario(BaseModel):
    id: str
    sample_id: str
    description: str
    modifications: list[Modification]
    events: list[Event]

class Scenarios(BaseModel):
    scenarios: list[Scenario]

