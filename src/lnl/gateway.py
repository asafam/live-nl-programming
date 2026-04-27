"""EventGateway — dispatches external events to Runtime-managed event sources."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .runtime import Runtime
    from .types import ProcessingResult


class EventGateway:
    """Dispatches external events to Runtime-managed event sources.

    Sits between the external world (test harness, webhooks, adapters) and
    the Runtime. Resolves which event source to fire into based on the
    object's declared event_sources.
    """

    def __init__(self, rt: Runtime) -> None:
        self._rt = rt

    def dispatch(
        self,
        recipient: str,
        content: str,
        descriptor: str | None = None,
        source: str = "__external__",
    ) -> list[ProcessingResult]:
        """Route an external event to the recipient's event source and process.

        Resolution order:
        1. If descriptor provided, use that specific event source
        2. Otherwise, use the object's first declared event source
        3. Fall back to inject_event if no event source exists
        """
        event_source = None

        if descriptor is not None:
            event_source = self._rt.get_event_source(recipient, descriptor)
        else:
            # Look up first declared event source for this object
            registry = self._rt.event_registry
            descriptors = registry.get(recipient)
            if descriptors:
                event_source = self._rt.get_event_source(recipient, descriptors[0])

        if event_source is not None:
            event_source.fire(content, source=source)
            return self._rt.process_pending()

        return self._rt.inject_event(recipient, content, source=source)

    def dispatch_many(
        self,
        items: list[tuple[str, str, str]],
    ) -> list[ProcessingResult]:
        """Dispatch multiple external events simultaneously in one transaction.

        Each item is (recipient, content, source). All events are injected into
        their recipients' event sources before the wave settles — true concurrent
        dispatch without serialization between items.
        """
        for recipient, content, source in items:
            registry = self._rt.event_registry
            descriptors = registry.get(recipient)
            if descriptors:
                event_source = self._rt.get_event_source(recipient, descriptors[0])
                if event_source is not None:
                    event_source.fire(content, source=source)
                    continue
            # fallback: inject directly
            self._rt._event_sources.inject(recipient, content, source)
        return self._rt.process_pending()
