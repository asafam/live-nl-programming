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

class GeneratedModification(BaseModel):
    """LLM output schema — mod_type and ambiguity are set by the script, not the LLM."""
    id: str
    when: str
    intent: str

class Modification(BaseModel):
    """Full modification with script-assigned mod_type and ambiguity."""
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
class ObjectDeclaration(BaseModel):
    name: str
    category: str  # "platform" or "business_logic"
    responsibility: str
    communicates_with: list[str]  # e.g., ["queries KnowledgeBase", "sends through Slack"]

class Sample(BaseModel):
    id: str
    name: str
    domain: str
    source_type: str
    link: str
    raw_steps: list[str]
    objects: list[ObjectDeclaration]
    steps: list[str]

class Samples(BaseModel):
    samples: list[Sample]

# Scenario generation schemas (LLM output before merging with instance metadata)
class Scenario(BaseModel):
    id: str
    sample_id: str
    description: str
    modifications: list[GeneratedModification]
    events: list[Event]

class Scenarios(BaseModel):
    scenarios: list[Scenario]

