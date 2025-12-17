from __future__ import annotations

from typing import Dict, Union

from src.actors.base import Actor, BaseActor, MessageType


class MessageBus:
    """Central router that holds actor instances and delivers messages."""

    def __init__(self) -> None:
        self.actors: Dict[str, BaseActor] = {}

    def register(self, actor: BaseActor) -> None:
        self.actors[actor.name] = actor
        if hasattr(actor, "set_message_bus"):
            actor.set_message_bus(self)

    def send_request(self, from_actor: str, to_actor: str, message: str) -> str:
        return self._send(from_actor, to_actor, message, MessageType.REQUEST)

    def notify_event(self, from_actor: str, to_actor: str, message: str) -> str:
        return self._send(from_actor, to_actor, message, MessageType.EVENT)

    def _send(self, from_actor: str, to_actor: str, message: str, message_type: MessageType = MessageType.REQUEST) -> str:
        # Special handling for User
        if to_actor == "User":
            print(f"[MessageBus] {from_actor} -> {to_actor} [{message_type.value}]: {message}")
            return ""

        if to_actor not in self.actors:
            return f"Unknown actor: {to_actor}"
        # Don't send empty messages
        if not message or not message.strip():
            return ""
        print(f"[MessageBus] {from_actor} -> {to_actor} [{message_type.value}]: {message}")
        recipient = self.actors[to_actor]
        response = recipient.receive(message, from_actor, message_type)
        if response:
            print(f"[MessageBus] {to_actor} -> {from_actor}: {response}")
        return response

    def broadcast(self, from_actor: str, message: str, message_type: MessageType = MessageType.REQUEST) -> Dict[str, str]:
        # Don't send empty messages
        if not message or not message.strip():
            return {}
        responses = {}
        for actor_name, actor in self.actors.items():
            if actor_name != from_actor:
                print(f"[MessageBus] {from_actor} -> {actor_name} [{message_type.value}]: {message}")
                response = actor.receive(message, from_actor, message_type)
                print(f"[MessageBus] {actor_name} -> {from_actor}: {response}")
                responses[actor_name] = response
        return responses
