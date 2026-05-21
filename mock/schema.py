"""
Mock server schema — Pydantic types for the HTTP mock protocol.

Self-contained: no imports from src/. Safe to copy alongside server.py.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class MockImmediateResponse(BaseModel):
    """Synchronous response returned to the tool caller."""
    template: str   # e.g. "message_id: {tool_call_id}, delivered to #{channel}"
    status: str = "ok"


class MockCallback(BaseModel):
    """Optional follow-up message injected back into the agent session."""
    delay_seconds: float = 0.5
    message_template: str   # interpolated with tool call args; ignored in LLM mode
    source: str             # e.g. "slack" — for log grouping


class MockMethodDef(BaseModel):
    """Behaviour definition for one tool method."""
    method: str                          # e.g. "slack_send_message"
    immediate: MockImmediateResponse
    callback: Optional[MockCallback] = None
    llm_persona: Optional[str] = None   # if set, use LLM mode for this method


class MockSystemDef(BaseModel):
    """Complete mock definition for one external system."""
    system: str
    tools: list[MockMethodDef]


class MockScript(BaseModel):
    """Collection of mock system definitions for one evaluation run."""
    systems: list[MockSystemDef]

    def get_method(self, method: str) -> Optional[MockMethodDef]:
        for sys in self.systems:
            for tool in sys.tools:
                if tool.method == method:
                    return tool
        return None


class OrchestratorReaction(BaseModel):
    """A single action to fire after a trigger matches."""
    source: str                         # e.g. "slack", "email" — appears in injection prefix
    message: str                        # template with {arg} interpolation from tool call args
    after_seconds: float = 0.0          # real-time delay (scaled by time_scale)
    after_minutes: float = 0.0          # simulated minutes (scaled by time_scale)


class OrchestratorTrigger(BaseModel):
    """Rule: when tool `tool` fires and args match `match`, schedule `reactions`."""
    tool: str                           # tool method name, e.g. "email_send"
    match: dict[str, str] = Field(default_factory=dict)  # arg key → regex pattern (empty = match all)
    reactions: list[OrchestratorReaction]
    fire_once: bool = True              # if True, fires only on the first matching call per session


class OrchestratorScript(BaseModel):
    """Named scenario script defining cross-system event chains."""
    name: str
    time_scale: float = 1.0             # compress time: 0.01 → 1 simulated min = 0.6 real sec
    triggers: list[OrchestratorTrigger]


class EventTrigger(BaseModel):
    """Describes how a TC Event is triggered by an external tool call."""
    tool: str
    match: dict[str, str] = Field(default_factory=dict)
    after_seconds: float = 0.0
    after_minutes: float = 0.0
