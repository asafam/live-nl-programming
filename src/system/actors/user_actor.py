from __future__ import annotations

from .base import BaseActor


class UserActor(BaseActor):
    """Special actor representing the external user. Forwards messages to the Coordinator."""

    def __init__(self, name: str = "User") -> None:
        super().__init__(name=name)

    def send_user_message(self, message: str) -> str:
        """Send a user message to the Coordinator and return the response."""
        if not self.message_bus:
            return "No message bus available."
        return self.message_bus.send_request(from_actor=self.name, to_actor="Coordinator", message=message)

    def receive(self, message: str, from_actor: str) -> str:
        # User actor doesn't receive messages in the normal flow; it's the sender
        return f"User actor received unexpected message: {message}"