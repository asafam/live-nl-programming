from __future__ import annotations

from typing import Dict, Set

from src.system.actors.base import Actor, BaseActor, MessageType


class MessageBus:
    """Central router that holds actor instances and delivers messages."""

    def __init__(self) -> None:
        self.actors: Dict[str, BaseActor] = {}
        # Topic subscriptions: topic_name -> set of subscriber actor names
        self.subscriptions: Dict[str, Set[str]] = {}

    def register(self, actor: BaseActor) -> None:
        self.actors[actor.name] = actor
        if hasattr(actor, "set_message_bus"):
            actor.set_message_bus(self)

        # Auto-subscribe mediators to their listening_to actors
        if hasattr(actor, 'state') and 'listening_to' in actor.state:
            for source_actor in actor.state['listening_to']:
                self.subscribe(actor.name, source_actor)

    def subscribe(self, subscriber: str, topic: str) -> None:
        """Subscribe an actor to a topic (typically another actor's name)."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        self.subscriptions[topic].add(subscriber)
        print(f"[MessageBus] {subscriber} subscribed to topic '{topic}'")

    def unsubscribe(self, subscriber: str, topic: str) -> None:
        """Unsubscribe an actor from a topic."""
        if topic in self.subscriptions and subscriber in self.subscriptions[topic]:
            self.subscriptions[topic].remove(subscriber)
            print(f"[MessageBus] {subscriber} unsubscribed from topic '{topic}'")
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]

    def send_request(self, from_actor: str, to_actor: str, message: str) -> str:
        return self._send(from_actor, to_actor, message, MessageType.REQUEST)

    def notify_event(self, from_actor: str, to_actor: str, message: str) -> str:
        """Send an event to a specific actor and also publish to subscribers of from_actor."""
        # Special handling for MessageBus - just publish to subscribers
        if to_actor == "MessageBus":
            self.publish(from_actor, message, MessageType.EVENT)
            return ""

        # Send direct message
        response = self._send(from_actor, to_actor, message, MessageType.EVENT)

        # Also publish to subscribers of the from_actor (as a topic)
        self.publish(from_actor, message, MessageType.EVENT)

        return response

    def send_response(self, from_actor: str, to_actor: str, message: str) -> str:
        return self._send(from_actor, to_actor, message, MessageType.RESPONSE)

    def publish(self, topic: str, message: str, message_type: MessageType = MessageType.EVENT) -> Dict[str, str]:
        """Publish a message to all subscribers of a topic."""
        if not message or not message.strip():
            return {}

        # Get subscribers for this topic
        subscribers = self.subscriptions.get(topic, set())
        if not subscribers:
            return {}

        responses = {}
        for subscriber_name in subscribers:
            if subscriber_name in self.actors:
                print(f"[MessageBus] {topic} ~> {subscriber_name} [{message_type.value}]: {message}")
                actor = self.actors[subscriber_name]
                response = actor.receive(message, topic, message_type)
                if response:
                    print(f"[MessageBus] {subscriber_name} -> {topic}: {response}")
                    responses[subscriber_name] = response

        return responses

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
