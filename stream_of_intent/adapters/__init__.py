"""Pluggable stream transport adapters.

Provides a factory function :func:`create_adapter` that returns the
appropriate :class:`~stream_of_intent.adapters.base.BaseAdapter` subclass
based on the :class:`~stream_of_intent.config.StreamConfig`.
"""

from __future__ import annotations

from stream_of_intent.adapters.base import BaseAdapter
from stream_of_intent.config import StreamConfig


def create_adapter(config: StreamConfig) -> BaseAdapter:
    """Instantiate the adapter specified by *config.adapter*.

    Args:
        config: Streaming configuration with the adapter type and
            connection parameters.

    Returns:
        A concrete :class:`BaseAdapter` subclass ready for use.

    Raises:
        ValueError: If the adapter type is unknown.
    """
    if config.adapter == "console":
        from stream_of_intent.adapters.console import ConsoleAdapter

        return ConsoleAdapter(config)

    if config.adapter == "websocket":
        from stream_of_intent.adapters.websocket import WebSocketAdapter

        return WebSocketAdapter(config)

    if config.adapter == "kafka":
        from stream_of_intent.adapters.kafka import KafkaAdapter

        return KafkaAdapter(config)

    raise ValueError(f"Unknown adapter type: {config.adapter!r}")


__all__ = ["BaseAdapter", "create_adapter"]
