"""LLMObject — the single runtime entity in the LNL system."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict

from .brain import (
    LLM_RESPONSE_SCHEMA,
    LLM_RESPONSE_SCHEMA_WITH_TOOLS,
    LLMBrain,
    _build_chat_messages,
    build_system_prompt,
)
from .tools import ToolRegistry
from .types import (
    InferenceMetrics,
    Message,
    ObjectDefinition,
    ProcessingResult,
)


class LLMObject:
    """An LLM-object: definition + brain + mutable NL state."""

    MAX_TOOL_ROUNDS = 5
    # Maximum number of past messages kept in history. The object's state is
    # the canonical summary of all prior processing, so old messages add noise
    # without adding information. None means unbounded (kept for compatibility).
    MAX_HISTORY = 6

    def __init__(
        self,
        definition: ObjectDefinition,
        brain: LLMBrain,
        tool_registry: ToolRegistry | None = None,
        tool_context_factory: object = None,
    ) -> None:
        self._definition = definition
        self._brain = brain
        self._state: dict = {}  # mutable runtime state; seed_data is static and kept on definition
        self._history: list[Message] = []
        self._mailbox: deque[Message] = deque()
        self._tool_registry = tool_registry
        self._tool_context_factory = tool_context_factory

    # --- Properties ---

    @property
    def object_id(self) -> str:
        return self._definition.object_id

    @property
    def state(self) -> dict:
        return self._state

    @property
    def definition(self) -> ObjectDefinition:
        return self._definition

    @property
    def peer_ids(self) -> list[str]:
        return [p.object_id for p in self._definition.peers]

    @property
    def subscriptions(self) -> list[str]:
        return list(self._definition.subscriptions)

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    # --- Mailbox ---

    @property
    def has_pending(self) -> bool:
        """True if the mailbox has messages waiting to be processed."""
        return bool(self._mailbox)

    @property
    def mailbox(self) -> deque[Message]:
        return self._mailbox

    def deliver(self, message: Message) -> None:
        """Put a message in this object's mailbox."""
        self._mailbox.append(message)

    def process_next(self) -> ProcessingResult | None:
        """Process the next message from the mailbox."""
        if not self._mailbox:
            return None
        message = self._mailbox.popleft()
        return self.process_message(message)

    # --- Core Processing (ReAct loop) ---

    def process_message(self, message: Message) -> ProcessingResult:
        """Process an incoming message via a ReAct loop: think → act → observe → repeat."""
        state_before = self._state

        # Assemble context once
        schema = LLM_RESPONSE_SCHEMA_WITH_TOOLS if self._tool_registry else LLM_RESPONSE_SCHEMA
        tools_desc = self._tool_registry.describe() if self._tool_registry else ""
        sys_prompt = build_system_prompt(self._definition, self._state, tools=tools_desc)
        messages = _build_chat_messages(sys_prompt, self._history, message)

        total_metrics = InferenceMetrics(model="")
        response = None
        tool_rounds = 0

        while True:
            response, metrics = self._brain.call(messages, schema, object_id=self.object_id)
            total_metrics = _accumulate_metrics(total_metrics, metrics)

            if not response.tool_calls or not self._tool_registry or tool_rounds >= self.MAX_TOOL_ROUNDS:
                break  # final response — no more tool calls (or limit reached)

            tool_rounds += 1
            # Execute tools and append results to the growing context
            ctx = self._tool_context_factory(self) if self._tool_context_factory else {}
            results = [self._tool_registry.execute(tc, ctx) for tc in response.tool_calls]

            messages.append({"role": "assistant", "content": json.dumps({
                "tool_calls": [
                    {"id": tc.id, "tool": tc.tool, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
                "reasoning": response.reasoning,
            })})
            results_text = "\n".join(
                f"[{r.id}] output: {r.output}" + (f"\nerror: {r.error}" if r.error else "")
                for r in results
            )
            messages.append({"role": "user", "content": f"[Tool results]:\n{results_text}"})

        # Apply final response
        self._state = response.updated_state
        self._history.append(message)
        if self.MAX_HISTORY is not None and len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY:]

        return ProcessingResult(
            object_id=self.object_id,
            reply=response.reply,
            outgoing_messages=response.outgoing_messages,
            state_before=state_before,
            state_after=self._state,
            metrics=total_metrics,
            in_reply_to=message.sender,
            source_message_type=message.type,
            external_actions=response.external_actions,
        )

    # --- Live Modification ---

    def modify_definition(self, **updates: object) -> None:
        """Change definition fields WITHOUT resetting state."""
        for key, value in updates.items():
            if not hasattr(self._definition, key):
                raise AttributeError(f"ObjectDefinition has no field '{key}'")
            setattr(self._definition, key, value)

    # --- Testing / Debugging ---

    def set_state(self, state: dict) -> None:
        """Set state directly (for testing)."""
        self._state = state

    def snapshot(self) -> dict:
        """Return a debug snapshot of the object."""
        return {
            "object_id": self.object_id,
            "state": self._state,
            "definition": asdict(self._definition),
            "history_length": len(self._history),
        }


def _accumulate_metrics(base: InferenceMetrics, add: InferenceMetrics) -> InferenceMetrics:
    """Combine metrics from multiple LLM calls."""
    return InferenceMetrics(
        input_tokens=base.input_tokens + add.input_tokens,
        output_tokens=base.output_tokens + add.output_tokens,
        latency_ms=base.latency_ms + add.latency_ms,
        model=base.model or add.model,
    )
