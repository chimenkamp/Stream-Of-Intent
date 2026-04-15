from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from stream_of_intent.types import Event, PostprocessorParams


class OutOfOrderInjector:
    """Injects out-of-order behavior by delaying arrival timestamps.

    For a configurable fraction of events, the ``arrival_timestamp`` is
    displaced forward relative to the ``timestamp``, creating events that
    arrive later than their natural temporal order would dictate.

    The transformed stream remains ordered by ``arrival_timestamp``.

    Attributes:
        fraction: Proportion of events whose arrival is delayed.
        max_displacement: Maximum delay added to arrival_timestamp.
    """

    def __init__(
        self,
        fraction: float = 0.1,
        max_displacement: float = 10.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the out-of-order injector.

        Args:
            fraction: Fraction of events to displace (0.0 to 1.0).
            max_displacement: Maximum delay in time units.
            seed: Random seed for reproducibility.
        """
        self.fraction = np.clip(fraction, 0.0, 1.0)
        self.max_displacement = max(max_displacement, 0.0)
        self.rng = np.random.RandomState(seed)

    def transform(self, stream: Iterator[Event]) -> Iterator[Event]:
        """Apply out-of-order displacement to the event stream.

        Buffers events and re-orders them by their (possibly displaced)
        arrival timestamps.  Uses a sliding buffer to limit memory usage
        while ensuring correct ordering.

        Args:
            stream: Input event stream ordered by arrival_timestamp.

        Yields:
            Events with displaced arrival timestamps, re-ordered by
            the new arrival_timestamp values.
        """
        if self.fraction < 1e-6 or self.max_displacement < 1e-6:
            yield from stream
            return

        buffer: List[Tuple[float, int, Event]] = []
        counter = 0
        flush_threshold = max(int(self.max_displacement * 2), 50)

        for event in stream:
            if self.rng.random() < self.fraction:
                delay = self.rng.uniform(0, self.max_displacement)
                displaced = Event(
                    case_id=event.case_id,
                    activity=event.activity,
                    timestamp=event.timestamp,
                    event_type=event.event_type,
                    arrival_timestamp=event.arrival_timestamp + delay,
                )
                heapq.heappush(buffer, (displaced.arrival_timestamp, counter, displaced))
            else:
                heapq.heappush(buffer, (event.arrival_timestamp, counter, event))

            counter += 1

            while len(buffer) > flush_threshold:
                _, _, buffered_event = heapq.heappop(buffer)
                yield buffered_event

        while buffer:
            _, _, buffered_event = heapq.heappop(buffer)
            yield buffered_event


class FractalInjector:
    """Amplifies fractal behavior by replicating trace patterns at different scales.

    Scans the stream for completed traces (sequences of events for a case)
    and probabilistically replicates them as new subprocess cases with
    scaled timestamps, creating self-similar nested patterns.

    Attributes:
        scale_factor: Temporal scaling factor for replicated patterns.
        replication_probability: Probability that a completed trace is
            replicated at a different scale.
    """

    def __init__(
        self,
        scale_factor: float = 0.5,
        replication_probability: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the fractal injector.

        Args:
            scale_factor: Temporal scaling for replicated patterns.
                Values < 1 compress time (faster subprocess), > 1 expand.
            replication_probability: Probability of replicating each
                completed trace.
            seed: Random seed for reproducibility.
        """
        self.scale_factor = scale_factor
        self.replication_probability = np.clip(replication_probability, 0.0, 1.0)
        self.rng = np.random.RandomState(seed)
        self._replica_counter = 0

    def transform(self, stream: Iterator[Event]) -> Iterator[Event]:
        """Apply fractal pattern replication to the event stream."""
        if self.replication_probability < 1e-6:
            yield from stream
            return

        case_events: Dict[str, List[Event]] = defaultdict(list)
        pending_starts: Dict[str, int] = defaultdict(int)
        pending_replicas: List[Tuple[float, int, Event]] = []
        counter = 0

        for event in stream:
            yield from self._flush_due_replicas(pending_replicas, event)
            yield event
            counter += 1
            counter = self._track_and_replicate(
                event, case_events, pending_starts, pending_replicas, counter,
            )

        yield from self._flush_remaining_replicas(pending_replicas)

    @staticmethod
    def _flush_due_replicas(
        pending_replicas: List[Tuple[float, int, Event]],
        current_event: Event,
    ) -> Iterator[Event]:
        """Yield all pending replicas due before the current event's arrival."""
        while pending_replicas and pending_replicas[0][0] <= current_event.arrival_timestamp:
            _, _, replica_event = heapq.heappop(pending_replicas)
            yield replica_event

    def _track_and_replicate(
        self,
        event: Event,
        case_events: Dict[str, List[Event]],
        pending_starts: Dict[str, int],
        pending_replicas: List[Tuple[float, int, Event]],
        counter: int,
    ) -> int:
        """Track case lifecycle and queue replicas on case completion."""
        case_events[event.case_id].append(event)

        if event.event_type == "start":
            pending_starts[event.case_id] += 1
        elif event.event_type == "end":
            pending_starts[event.case_id] -= 1
            if self._should_replicate(event.case_id, case_events, pending_starts):
                counter = self._queue_replicas(
                    event, case_events, pending_starts, pending_replicas, counter,
                )

        return counter

    def _should_replicate(
        self,
        case_id: str,
        case_events: Dict[str, List[Event]],
        pending_starts: Dict[str, int],
    ) -> bool:
        """Determine whether a completed case should be replicated."""
        return (
            pending_starts[case_id] <= 0
            and len(case_events[case_id]) >= 2
            and self.rng.random() < self.replication_probability
        )

    def _queue_replicas(
        self,
        event: Event,
        case_events: Dict[str, List[Event]],
        pending_starts: Dict[str, int],
        pending_replicas: List[Tuple[float, int, Event]],
        counter: int,
    ) -> int:
        """Create trace replicas and push them onto the pending heap."""
        replicas = self._replicate_trace(
            case_events[event.case_id],
            event.arrival_timestamp,
        )
        for replica in replicas:
            heapq.heappush(
                pending_replicas,
                (replica.arrival_timestamp, counter, replica),
            )
            counter += 1

        del case_events[event.case_id]
        del pending_starts[event.case_id]
        return counter

    @staticmethod
    def _flush_remaining_replicas(
        pending_replicas: List[Tuple[float, int, Event]],
    ) -> Iterator[Event]:
        """Yield all remaining replicas after the input stream is exhausted."""
        while pending_replicas:
            _, _, replica_event = heapq.heappop(pending_replicas)
            yield replica_event

    def _replicate_trace(
        self,
        events: List[Event],
        base_time: float,
    ) -> List[Event]:
        """Create a scaled replica of a trace's events.

        Args:
            events: Original events of the trace to replicate.
            base_time: Start time for the replica.

        Returns:
            List of replica events with scaled timestamps and new case IDs.
        """
        self._replica_counter += 1
        parent_id = events[0].case_id
        replica_id = f"{parent_id}.fractal_{self._replica_counter}"

        if not events:
            return []

        origin_time = events[0].timestamp
        replicas: List[Event] = []

        for event in events:
            offset = (event.timestamp - origin_time) * self.scale_factor
            new_timestamp = base_time + offset

            arrival_offset = (event.arrival_timestamp - origin_time) * self.scale_factor
            new_arrival = base_time + arrival_offset

            replicas.append(Event(
                case_id=replica_id,
                activity=event.activity,
                timestamp=new_timestamp,
                event_type=event.event_type,
                arrival_timestamp=new_arrival,
            ))

        return replicas


def apply_postprocessors(
    stream: Iterator[Event],
    params: PostprocessorParams,
    seed: Optional[int] = None,
) -> Iterator[Event]:
    """Apply all postprocessors to a stream based on parameter configuration.

    Constructs and chains the appropriate postprocessors based on the
    parameter values.  Postprocessors with zero-strength parameters
    (fraction = 0, probability = 0) are skipped.

    Args:
        stream: Raw event stream from the simulation engine.
        params: Postprocessor parameters controlling injection strengths.
        seed: Random seed for reproducibility.

    Returns:
        Transformed event stream with postprocessors applied.
    """
    current_stream = stream

    if params.fractal_replication_probability > 1e-6:
        fractal = FractalInjector(
            scale_factor=params.fractal_scale_factor,
            replication_probability=params.fractal_replication_probability,
            seed=seed,
        )
        current_stream = fractal.transform(current_stream)

    if params.ooo_fraction > 1e-6 and params.ooo_max_displacement > 1e-6:
        ooo = OutOfOrderInjector(
            fraction=params.ooo_fraction,
            max_displacement=params.ooo_max_displacement,
            seed=(seed + 1) if seed is not None else None,
        )
        current_stream = ooo.transform(current_stream)

    return current_stream
