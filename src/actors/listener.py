from abc import ABC, abstractmethod
from typing import Optional
from src.actors.base import BaseActor, MessageType

class Listener(BaseActor, ABC):
    """
    Abstract base class for actors that listen to external APIs/events 
    and push messages to the bus, without using an LLM.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.message_bus = None  # Set by MessageBus upon registration

    def set_message_bus(self, bus) -> None:
        self.message_bus = bus

    def receive(self, message: str, from_actor: str, message_type: MessageType = MessageType.REQUEST) -> str:
        """
        Listeners do not process incoming messages.
        """
        return ""

    @abstractmethod
    def listen(self) -> None:
        """
        Check for external events and push them to the message bus if needed.
        This method should be called periodically or triggered by the system.
        """
        pass
