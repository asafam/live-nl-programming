from typing import Optional, Dict
from src.actors.listener import Listener
from src.actors.base import MessageType

class MockEmailMonitor(Listener):
    def __init__(self, name: str, email_message: Optional[str] = None, trigger_count: int = 2) -> None:
        super().__init__(name=name)
        self.state = {"interaction_count": 0}
        self.trigger_count = trigger_count
        self.email_message = email_message or "New email from Sarah: 'I'm going vegan, please cancel the meat'."

    def listen(self) -> None:
        """Simulate checking emails and triggering an event."""
        self.check_emails()

    def check_emails(self) -> None:
        """Simulate checking emails and triggering an event."""
        self.state["interaction_count"] += 1
        if self.state["interaction_count"] >= self.trigger_count:
            if self.message_bus:
                print(f"[{self.name}] Injecting email event...")
                # Broadcast to MessageBus so mediators can subscribe
                self.message_bus.notify_event(
                    from_actor=self.name,
                    to_actor="MessageBus",
                    message=self.email_message
                )
