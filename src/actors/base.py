from __future__ import annotations

import json
from typing import Dict, List, Optional

from src.llm.base import AbstractLLM, ChatMessage, system_message, user_message


class BaseActor:
    """Marker base class for all actors."""

    def __init__(self, name: str) -> None:
        self.name = name

    def receive(self, message: str, from_actor: str) -> str:  # pragma: no cover - interface hook
        raise NotImplementedError


class Actor(BaseActor):
    """Concrete actor with messaging and state management."""

    def __init__(
        self,
        name: str,
        llm: AbstractLLM,
        system_prompt: str,
        initial_state: Optional[Dict] = None,
    ) -> None:
        super().__init__(name=name)
        self.llm = llm
        self.system_prompt = system_prompt
        self.state: Dict = initial_state or {}
        self.message_bus = None  # set by MessageBus upon registration
        self.message_history: List[tuple[str, str]] = []  # List of (from_actor, message)
        # No tools - state updates and messaging are handled via structured response parsing

    def set_message_bus(self, bus) -> None:
        self.message_bus = bus

    def set_tool_executor(self, executor) -> None:
        self.tool_executor = executor

    def update_system_prompt(self, new_prompt: str) -> None:
        """Update the actor's system prompt"""
        self.system_prompt = new_prompt

    # ---- Generic state helpers -------------------------------------------------
    def get_state(self) -> Dict:
        return self.state

    def set_state_field(self, key: str, value) -> None:
        print(f"[State] {self.name}: set {key} = {value}")
        self.state[key] = value

    def update_state(self, updates: Dict) -> None:
        print(f"[State] {self.name}: update {updates}")
        self.state.update(updates)

    def speak(self, content: str) -> str:
        """LLM-backed natural language response with actor persona."""
        state_for_prompt = {k: v for k, v in self.state.items() if k != "known_actors"}
        full_prompt = self.system_prompt + f"\n\nCurrent state: {json.dumps(state_for_prompt)}. Use this current state to inform your response."
        messages: List[ChatMessage] = [system_message(full_prompt), user_message(content)]
        return self.llm.generate_text(messages)

    def _format_known_actors(self) -> Optional[str]:
        known = self.state.get("known_actors", {})
        if not known:
            return None
        lines = ["Known actors:"]
        for name, meta in known.items():
            desc = meta.get("purpose") or meta.get("description") or ""
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def receive(self, message: str, from_actor: str) -> str:
        """Handle incoming natural-language message and parse structured response from LLM."""
        state_for_prompt = {k: v for k, v in self.state.items() if k != "known_actors"}
        chat: List[ChatMessage] = [system_message(self.system_prompt + f"\n\nCurrent state: {json.dumps(state_for_prompt)}")]

        known_ctx = self._format_known_actors()
        if known_ctx:
            chat.append(system_message(known_ctx))

        chat.append(user_message(f"From {from_actor}: {message}"))

        # Define structured response schema
        response_schema = {
            "type": "object",
            "properties": {
                "state": {
                    "type": "object",
                    "description": "State updates to apply (JSON object that will be merged into current state)",
                    "additionalProperties": True  # Allow any additional properties
                },
                "messages": {
                    "type": "array",
                    "description": "Messages to send to other actors",
                    "items": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "description": "Target actor name"},
                            "message": {"type": "string", "description": "Message content"},
                            "message_type": {"type": "string", "description": "Type of the message, defaults to 'default'"}
                        },
                        "required": ["to", "message"],
                        "additionalProperties": False
                    }
                }
            },
            "required": []
        }

        # Get structured response from LLM
        structured_response = self.llm.generate_structured(chat, response_schema)

        # Apply state changes if any
        if structured_response.state:
            self.update_state(structured_response.state)

        # Send messages if any
        if structured_response.messages and self.message_bus:
            for msg in structured_response.messages:
                if isinstance(msg, dict) and 'to' in msg and 'message' in msg:
                    msg.setdefault('message_type', 'default')
                    self.message_bus.send(from_actor=self.name, to_actor=msg['to'], message=msg['message'])

        return ""

