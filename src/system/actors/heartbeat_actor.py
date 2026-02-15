from __future__ import annotations

import threading
import time

from .base import BaseActor


class HeartbeatActor(BaseActor):
    """Actor that periodically broadcasts a heartbeat message."""

    def __init__(self, name: str, interval: int) -> None:
        super().__init__(name=name)
        self.interval = interval
        self.message_bus = None

    def set_message_bus(self, bus) -> None:
        self.message_bus = bus

    def receive(self, message: str, from_actor: str) -> str:
        """Handle incoming messages. For heartbeat, just acknowledge."""
        return f"Heartbeat actor received: {message}"

    def start(self) -> None:
        """Start the heartbeat thread."""
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def run(self) -> None:
        """Run the heartbeat loop."""
        while True:
            time.sleep(self.interval)
            if self.message_bus:
                self.message_bus.broadcast(self.name, "heartbeat")