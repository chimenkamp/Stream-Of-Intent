from __future__ import annotations

import sys

from stream_of_intent.adapters.base import BaseAdapter, logger
from stream_of_intent.config import StreamConfig
from stream_of_intent.types import Event


class ConsoleAdapter(BaseAdapter):
    """Writes each event as a single JSON line to *stdout*."""

    def __init__(self, config: StreamConfig) -> None:
        super().__init__(config)

    def connect(self) -> None:
        logger.info("ConsoleAdapter: streaming events to stdout …")

    def send(self, event: Event) -> None:
        sys.stdout.write(self._serialize_event(event) + "\n")
        sys.stdout.flush()

    def disconnect(self) -> None:
        logger.info("ConsoleAdapter: stream ended.")
