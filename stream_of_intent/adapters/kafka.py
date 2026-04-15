from __future__ import annotations

from stream_of_intent.adapters.base import BaseAdapter, logger
from stream_of_intent.config import StreamConfig
from stream_of_intent.types import Event


class KafkaAdapter(BaseAdapter):
    """Publishes each event as a JSON message to a Kafka topic.

    Requires a running Kafka broker reachable at ``config.host:config.port``.
    Uses the ``kafka-python-ng`` package (``kafka.KafkaProducer``).
    """

    def __init__(self, config: StreamConfig) -> None:
        super().__init__(config)
        self._producer = None

    def connect(self) -> None:
        from kafka import KafkaProducer  # type: ignore[import-untyped]

        bootstrap = f"{self.config.host}:{self.config.port}"
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: v.encode("utf-8"),
        )
        logger.info(
            "KafkaAdapter: connected to %s, topic=%s",
            bootstrap, self.config.topic,
        )

    def send(self, event: Event) -> None:
        if self._producer is None:
            raise RuntimeError("KafkaAdapter.connect() must be called first.")
        self._producer.send(self.config.topic, value=self._serialize_event(event))

    def disconnect(self) -> None:
        if self._producer is not None:
            self._producer.flush()
            self._producer.close()
            self._producer = None
        logger.info("KafkaAdapter: producer closed.")
