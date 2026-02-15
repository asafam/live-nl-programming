"""Output schema for evaluation records."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ExpectedOutcome(BaseModel):
    action: str
    reason: str


class EvalRecord(BaseModel):
    """One record per input (step, modification, or event) in a test case."""
    test_case_id: str
    input_id: str
    input_type: str  # "step", "modification", or "event"
    input_text: str
    when: Optional[str] = None
    actual_response: str
    expected: Optional[ExpectedOutcome] = None
    tokens: TokenUsage = TokenUsage()
    latency_ms: int = 0
    actor_states: Dict[str, Any] = {}
