"""LLM provider abstraction — Brain interface and implementations."""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from .types import (
    InferenceMetrics,
    LLMResponse,
    Message,
    ObjectDefinition,
    OutgoingMessage,
)

# JSON schema for the LLM response format
LLM_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "updated_state": {
            "type": "string",
            "description": "The complete updated state after processing the message.",
        },
        "reply": {
            "type": "string",
            "description": "Your reply to the sender of the message.",
        },
        "outgoing_messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "The object_id of the recipient.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the message.",
                    },
                },
                "required": ["recipient", "content"],
                "additionalProperties": False,
            },
            "description": "Messages to send to other objects.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief internal reasoning about what you did and why.",
        },
    },
    "required": ["updated_state", "reply", "outgoing_messages", "reasoning"],
    "additionalProperties": False,
}


_PROMPT_CONFIG: Optional[dict] = None


def _load_prompt_config() -> dict:
    """Load the prompt config from config/prompts/lnl/object.yaml."""
    global _PROMPT_CONFIG
    if _PROMPT_CONFIG is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "prompts" / "lnl" / "object.yaml"
        with open(config_path) as f:
            _PROMPT_CONFIG = yaml.safe_load(f)
    return _PROMPT_CONFIG


def build_system_prompt(definition: ObjectDefinition, current_state: str) -> str:
    """Build the system prompt from the YAML template and an ObjectDefinition."""
    config = _load_prompt_config()
    template = config["system_prompt"]

    peers = ""
    if definition.peers:
        peers = "\n".join(f"- {p.object_id}: {p.relationship}" for p in definition.peers)

    skills = ""
    if definition.skills:
        skills = "\n".join(f"- {s}" for s in definition.skills)

    return template.format(
        object_id=definition.object_id,
        role=definition.role,
        behavior=definition.behavior or "(none)",
        peers=peers or "(none)",
        skills=skills or "(none)",
        state_description=definition.state_description or "(none)",
        current_state=current_state or "(empty)",
    )


def get_history_prefix() -> str:
    """Get the history prefix text from config."""
    config = _load_prompt_config()
    return config.get("history_prefix", "").strip()


def _build_chat_messages(
    sys_prompt: str,
    history: Sequence[Message],
    message: Message,
) -> list[dict[str, str]]:
    """Build the chat message list with labeled history and new message."""
    msgs: list[dict[str, str]] = [{"role": "system", "content": sys_prompt}]
    if history:
        prefix = get_history_prefix()
        history_lines = [f"  [{msg.sender}]: {msg.content}" for msg in history]
        msgs.append({"role": "user", "content": f"{prefix}\n" + "\n".join(history_lines)})
        msgs.append({"role": "assistant", "content": "Understood, I see the past context. What is the new message?"})
    msgs.append({"role": "user", "content": f"[NEW MESSAGE] [{message.sender}]: {message.content}"})
    return msgs


class LLMBrain(ABC):
    """Abstract interface for LLM processing backends."""

    @abstractmethod
    def process(
        self,
        definition: ObjectDefinition,
        current_state: str,
        message: Message,
        history: Sequence[Message],
    ) -> tuple[LLMResponse, InferenceMetrics]:
        """Process a message and return the LLM response with metrics."""
        ...


class OpenAIBrain(LLMBrain):
    """Brain backed by the OpenAI API (self-contained, no config files)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        seed: Optional[int] = 42,
    ) -> None:
        import os

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self.model = model
        self._temperature = temperature
        self._seed = seed
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def process(
        self,
        definition: ObjectDefinition,
        current_state: str,
        message: Message,
        history: Sequence[Message],
    ) -> tuple[LLMResponse, InferenceMetrics]:
        sys_prompt = build_system_prompt(definition, current_state)
        messages = _build_chat_messages(sys_prompt, history, message)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "llm_response",
                    "schema": LLM_RESPONSE_SCHEMA,
                    "strict": True,
                },
            },
        }
        if self._seed is not None:
            kwargs["seed"] = self._seed

        t0 = time.time()
        resp = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        metrics = InferenceMetrics(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            latency_ms=latency_ms,
            model=self.model,
        )

        raw = json.loads(resp.choices[0].message.content or "{}")
        return _parse_llm_result(raw), metrics


class AnthropicBrain(LLMBrain):
    """Brain backed by the Anthropic API (self-contained, no config files)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        import os

        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self.model = model
        self._temperature = temperature
        self._client = _anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    @staticmethod
    def _enforce_strict_schema(schema: dict) -> None:
        """Recursively set additionalProperties: false on all object types."""
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)
        for key in ("properties", "$defs"):
            if key in schema:
                for sub in schema[key].values():
                    if isinstance(sub, dict):
                        AnthropicBrain._enforce_strict_schema(sub)
        for key in ("items", "anyOf", "oneOf", "allOf"):
            if key in schema:
                target = schema[key]
                if isinstance(target, dict):
                    AnthropicBrain._enforce_strict_schema(target)
                elif isinstance(target, list):
                    for item in target:
                        if isinstance(item, dict):
                            AnthropicBrain._enforce_strict_schema(item)

    def process(
        self,
        definition: ObjectDefinition,
        current_state: str,
        message: Message,
        history: Sequence[Message],
    ) -> tuple[LLMResponse, InferenceMetrics]:
        sys_prompt = build_system_prompt(definition, current_state)
        # Anthropic: system is separate, only pass non-system messages
        all_msgs = _build_chat_messages(sys_prompt, history, message)
        messages = [m for m in all_msgs if m["role"] != "system"]

        schema = json.loads(json.dumps(LLM_RESPONSE_SCHEMA))
        self._enforce_strict_schema(schema)

        t0 = time.time()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self._temperature,
            system=sys_prompt,
            messages=messages,
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                },
            },
        )
        latency_ms = (time.time() - t0) * 1000

        metrics = InferenceMetrics(
            input_tokens=getattr(resp.usage, "input_tokens", 0) if resp.usage else 0,
            output_tokens=getattr(resp.usage, "output_tokens", 0) if resp.usage else 0,
            latency_ms=latency_ms,
            model=self.model,
        )

        content_str = ""
        for block in resp.content:
            if hasattr(block, "text"):
                content_str += block.text

        raw = json.loads(content_str or "{}")
        return _parse_llm_result(raw), metrics


@dataclass
class _ScriptEntry:
    response: LLMResponse
    metrics: InferenceMetrics = field(
        default_factory=lambda: InferenceMetrics(model="mock")
    )


@dataclass
class CallRecord:
    """Record of a call made to MockBrain."""
    object_id: str
    definition: ObjectDefinition
    current_state: str
    message: Message


class MockBrain(LLMBrain):
    """Deterministic scripted brain for testing."""

    def __init__(self) -> None:
        self._scripts: dict[str, list[_ScriptEntry]] = {}
        self._default_response: Optional[LLMResponse] = None
        self.call_log: list[CallRecord] = []

    def script(
        self,
        object_id: str,
        response: LLMResponse,
        metrics: Optional[InferenceMetrics] = None,
    ) -> None:
        """Add a scripted response for an object. Responses are consumed in order."""
        entry = _ScriptEntry(
            response=response,
            metrics=metrics or InferenceMetrics(model="mock"),
        )
        self._scripts.setdefault(object_id, []).append(entry)

    def set_default(self, response: LLMResponse) -> None:
        """Set a default response for any unscripted calls."""
        self._default_response = response

    def process(
        self,
        definition: ObjectDefinition,
        current_state: str,
        message: Message,
        history: Sequence[Message],
    ) -> tuple[LLMResponse, InferenceMetrics]:
        self.call_log.append(
            CallRecord(
                object_id=definition.object_id,
                definition=definition,
                current_state=current_state,
                message=message,
            )
        )

        entries = self._scripts.get(definition.object_id, [])
        if entries:
            entry = entries.pop(0)
            return entry.response, entry.metrics

        if self._default_response is not None:
            return self._default_response, InferenceMetrics(model="mock")

        # Fallback: echo back with no state change
        return (
            LLMResponse(
                updated_state=current_state,
                reply=f"Echo: {message.content}",
                outgoing_messages=[],
                reasoning="No script configured",
            ),
            InferenceMetrics(model="mock"),
        )


def _parse_llm_result(result: Any) -> LLMResponse:
    """Parse the raw LLM result (dict or StructuredResponse) into LLMResponse."""
    if isinstance(result, dict):
        data = result
    else:
        # StructuredResponse from Anthropic — has .response, .state, .messages
        data = {
            "updated_state": getattr(result, "state", "") or "",
            "reply": getattr(result, "response", "") or "",
            "outgoing_messages": getattr(result, "messages", []) or [],
            "reasoning": "",
        }

    outgoing = []
    for m in data.get("outgoing_messages", []):
        if isinstance(m, dict):
            outgoing.append(OutgoingMessage(recipient=m["recipient"], content=m["content"]))
        elif isinstance(m, OutgoingMessage):
            outgoing.append(m)

    return LLMResponse(
        updated_state=data.get("updated_state", ""),
        reply=data.get("reply", ""),
        outgoing_messages=outgoing,
        reasoning=data.get("reasoning", ""),
    )
