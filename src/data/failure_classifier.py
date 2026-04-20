"""LLM-as-classifier — assigns a structured failure category to a failed event.

Runs post-hoc on the judge's verdict + evidence. Categories are defined in
config/prompts/data-gen/failure_classifier.yaml.
"""
from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_PROMPT_PATH = Path(__file__).parent.parent.parent / "config" / "prompts" / "data-gen" / "failure_classifier.yaml"
_CACHE_DIR = Path(__file__).parent.parent.parent / "outputs" / "data" / ".failure_cache"

CATEGORIES = [
    "missing_peer_action",
    "missing_tool_action",
    "wrong_arguments",
    "wrong_recipient",
    "partial_chain",
    "state_not_persisted",
    "cross_context_contamination",
    "timeout",
    "step_failed",
    "other",
]

_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "confidence": {"type": "number"},
        "short_rationale": {"type": "string"},
    },
    "required": ["category", "confidence", "short_rationale"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ClassifierResult:
    category: str
    confidence: float
    short_rationale: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "short_rationale": self.short_rationale,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached": self.cached,
        }


def _load_system_prompt() -> str:
    with open(_PROMPT_PATH) as f:
        return yaml.safe_load(f)["system_prompt"].strip()


def _cache_key(condition: str, reasoning: str, evidence: str, model: str) -> str:
    h = hashlib.sha256()
    for part in (model, condition, reasoning, evidence):
        h.update(part.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_read(key: str) -> Optional[dict]:
    path = _CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _cache_write(key: str, payload: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(payload, f)


def _user_msg(condition: str, reasoning: str, evidence: str, prior_context: str = "") -> str:
    parts = [f"Condition: {condition}", f"Judge reasoning: {reasoning}"]
    if prior_context:
        parts.append(f"Prior context:\n{prior_context[:2000]}")
    parts.append(f"Evidence:\n{evidence[:6000]}")
    return "\n\n".join(parts)


def _heuristic_timeout(reasoning: str) -> bool:
    r = reasoning.lower()
    return r.startswith("timeout after") or " timeout after " in r


def _heuristic_step_failed(event_id: str) -> bool:
    return isinstance(event_id, str) and event_id.startswith("S")


class FailureClassifier(ABC):
    """Classify a failed EventResult into one of CATEGORIES."""

    model: str = "unknown"

    def classify(
        self,
        *,
        condition: str,
        reasoning: str,
        evidence: str,
        prior_context: str = "",
        event_id: str = "",
    ) -> ClassifierResult:
        # Fast-path heuristics — skip LLM for unambiguous cases
        if _heuristic_timeout(reasoning):
            return ClassifierResult("timeout", 1.0, "Timeout text in reasoning", cached=True)

        key = _cache_key(condition, reasoning, evidence, self.model)
        cached = _cache_read(key)
        if cached is not None:
            return ClassifierResult(
                category=cached["category"],
                confidence=float(cached.get("confidence", 0.0)),
                short_rationale=cached.get("short_rationale", ""),
                input_tokens=int(cached.get("input_tokens", 0)),
                output_tokens=int(cached.get("output_tokens", 0)),
                cached=True,
            )

        category, confidence, rationale, in_tok, out_tok = self._call(
            condition, reasoning, evidence, prior_context
        )

        # Post-hoc cascade check: step events are always step_failed if not timeout
        if _heuristic_step_failed(event_id) and category == "other":
            category = "step_failed"

        if category not in CATEGORIES:
            category = "other"

        payload = {
            "category": category,
            "confidence": confidence,
            "short_rationale": rationale,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }
        _cache_write(key, payload)
        return ClassifierResult(
            category=category,
            confidence=confidence,
            short_rationale=rationale,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached=False,
        )

    @abstractmethod
    def _call(
        self, condition: str, reasoning: str, evidence: str, prior_context: str
    ) -> tuple[str, float, str, int, int]:
        """Return (category, confidence, short_rationale, input_tokens, output_tokens)."""


class OpenAIFailureClassifier(FailureClassifier):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        self.model = model
        self._client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._system = _load_system_prompt()

    def _call(self, condition, reasoning, evidence, prior_context):
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": _user_msg(condition, reasoning, evidence, prior_context)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = _safe_json_loads(resp.choices[0].message.content or "{}")
        in_tok = resp.usage.prompt_tokens if resp.usage else 0
        out_tok = resp.usage.completion_tokens if resp.usage else 0
        return (
            str(raw.get("category", "other")),
            float(raw.get("confidence", 0.0) or 0.0),
            str(raw.get("short_rationale", "")),
            in_tok,
            out_tok,
        )


class AnthropicFailureClassifier(FailureClassifier):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: Optional[str] = None) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        self.model = model
        self._client = _anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"],
            timeout=120.0,
        )
        self._system = _load_system_prompt()

    def _call(self, condition, reasoning, evidence, prior_context):
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0.0,
            system=self._system,
            messages=[{"role": "user", "content": _user_msg(condition, reasoning, evidence, prior_context)}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
        content = "".join(block.text for block in resp.content if hasattr(block, "text"))
        raw = _safe_json_loads(content or "{}")
        in_tok = resp.usage.input_tokens if resp.usage else 0
        out_tok = resp.usage.output_tokens if resp.usage else 0
        return (
            str(raw.get("category", "other")),
            float(raw.get("confidence", 0.0) or 0.0),
            str(raw.get("short_rationale", "")),
            in_tok,
            out_tok,
        )


def make_classifier(provider: str, model: Optional[str] = None) -> FailureClassifier:
    provider = provider.lower()
    if provider == "openai":
        return OpenAIFailureClassifier(model=model or "gpt-4o-mini")
    if provider == "anthropic":
        return AnthropicFailureClassifier(model=model or "claude-haiku-4-5-20251001")
    raise ValueError(f"Unknown classifier provider: {provider}")


def _safe_json_loads(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(text)
            return result
        return {}
