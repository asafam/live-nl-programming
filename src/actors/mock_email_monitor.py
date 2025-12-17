from typing import Optional, Dict
from src.actors.listener import Listener
from src.actors.base import MessageType

class MockEmailMonitor(Listener):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.state = {"interaction_count": 0}
        self.trigger_count = 2  # Trigger after 2 interactions

    def listen(self) -> None:
        """Simulate checking emails and triggering an event."""
        self.check_emails()

    def check_emails(self) -> None:
        """Simulate checking emails and triggering an event."""
        self.state["interaction_count"] += 1
        if self.state["interaction_count"] >= self.trigger_count:
            if self.message_bus:
                print(f"[{self.name}] Injecting email event...")
                self.message_bus.notify_event(
                    from_actor=self.name,
                    to_actor="Coordinator",
                    message="New email from Sarah: 'I'm going vegan, please cancel the meat'."
                )
