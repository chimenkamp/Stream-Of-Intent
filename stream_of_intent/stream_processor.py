from __future__ import annotations

import logging
from typing import Iterator, Optional

from stream_of_intent.config import StreamConfig
from stream_of_intent.model_representation import (
    AbstractModelRepresentation,
    ReplayableProcessModel,
)
from stream_of_intent.postprocessors import apply_postprocessors
from stream_of_intent.simulation import SimulationEngine
from stream_of_intent.types import Event

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Replays an optimised model as a configurable-loop event stream.

    The processor reconstructs a :class:`ReplayableProcessModel` from the
    stored transition matrices and activities, runs the
    :class:`SimulationEngine` through postprocessors, and loops the
    simulation according to :attr:`config.max_loops`.

    Attributes:
        model: The abstract model representation to replay.
        config: Streaming configuration (loop count, etc.).
    """

    def __init__(
        self,
        model: AbstractModelRepresentation,
        config: Optional[StreamConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or StreamConfig()

    def run(self) -> Iterator[Event]:
        """Generate an event stream by replaying the stored model."""
        params = self.model.optimized_params
        static = self.model.static_params
        replay_model = ReplayableProcessModel.from_representation(self.model)

        loop_count = 0
        time_offset = 0.0
        case_id_offset = 0

        while not self._should_stop_looping(loop_count):
            engine = self._create_loop_engine(
                replay_model, params, static, loop_count, case_id_offset,
            )
            stream = self._build_postprocessed_stream(engine, params, static, loop_count)
            last_arrival, events_in_loop = yield from self._yield_shifted_events(
                stream, time_offset,
            )

            case_id_offset = engine._case_counter
            time_offset = last_arrival
            loop_count += 1
            logger.debug(
                "Loop %d complete — %d events, time_offset=%.2f",
                loop_count, events_in_loop, time_offset,
            )

    def _should_stop_looping(self, loop_count: int) -> bool:
        """Check whether the replay loop limit has been reached."""
        return self.config.max_loops > 0 and loop_count >= self.config.max_loops

    def _create_loop_engine(
        self,
        replay_model: ReplayableProcessModel,
        params,
        static,
        loop_count: int,
        case_id_offset: int,
    ) -> SimulationEngine:
        """Create a fresh SimulationEngine for one loop iteration."""
        engine = SimulationEngine(
            model=replay_model,
            params=params.simulation,
            seed=(
                (static.random_seed + loop_count)
                if static.random_seed is not None
                else None
            ),
        )
        engine._case_counter = case_id_offset
        return engine

    def _build_postprocessed_stream(
        self,
        engine: SimulationEngine,
        params,
        static,
        loop_count: int,
    ) -> Iterator[Event]:
        """Run the engine and apply postprocessors for one loop iteration."""
        raw_stream = engine.run(max_events=static.stream_length)
        return apply_postprocessors(
            raw_stream,
            params.postprocessor,
            seed=(
                (static.random_seed + loop_count)
                if static.random_seed is not None
                else None
            ),
        )

    @staticmethod
    def _yield_shifted_events(
        stream: Iterator[Event],
        time_offset: float,
    ) -> Iterator[Event]:
        """Shift event timestamps by the accumulated offset and yield them."""
        last_arrival = time_offset
        events_in_loop = 0
        for event in stream:
            shifted = Event(
                case_id=event.case_id,
                activity=event.activity,
                timestamp=event.timestamp + time_offset,
                event_type=event.event_type,
                arrival_timestamp=event.arrival_timestamp + time_offset,
            )
            last_arrival = max(last_arrival, shifted.arrival_timestamp)
            yield shifted
            events_in_loop += 1
        return last_arrival, events_in_loop

    @classmethod
    def from_file(
        cls,
        path: str,
        config: Optional[StreamConfig] = None,
    ) -> StreamProcessor:
        """Load a model representation from a JSON file and build a processor.

        Args:
            path: Path to the ``.model.json`` file.
            config: Optional streaming configuration.

        Returns:
            A ready-to-run :class:`StreamProcessor`.
        """
        rep = AbstractModelRepresentation.load(path)
        return cls(model=rep, config=config)
