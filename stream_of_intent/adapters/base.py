from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Iterator

from stream_of_intent.config import StreamConfig
from stream_of_intent.types import Event

logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """Interface that every stream adapter must implement.

    Subclasses provide transport-specific :meth:`connect`, :meth:`send`,
    and :meth:`disconnect` implementations. The :meth:`stream` convenience
    method drives the full lifecycle.
    """

    def __init__(self, config: StreamConfig) -> None:
        self.config = config

    @abstractmethod
    def connect(self) -> None:
        """Establish the transport connection."""

    @abstractmethod
    def send(self, event: Event) -> None:
        """Serialize and transmit a single event."""

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully shut down the transport."""

    def stream(self, events: Iterator[Event]) -> None:
        """Connect, iterate over *events*, send each one, then disconnect.

        Args:
            events: Potentially infinite iterator of events to stream.
        """
        self.connect()
        try:
            for event in events:
                self.send(event)
        except KeyboardInterrupt:
            logger.info("Stream interrupted by user.")
        finally:
            self.disconnect()

    def _serialize_event(self, event: Event) -> str:
        """Serialize an :class:`Event` to a JSON string.

        Args:
            event: The event to serialize.

        Returns:
            JSON string representation.
        """
        return json.dumps({
            "case_id": event.case_id,
            "activity": event.activity,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "arrival_timestamp": event.arrival_timestamp,
        })
