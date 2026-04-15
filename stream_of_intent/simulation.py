from __future__ import annotations

import heapq
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from stream_of_intent.process_model import ProcessModel, _START_SYMBOL, _END_SYMBOL
from stream_of_intent.types import Event, SimulationParams


class _ActiveCase:
    """Internal state for a single active process case.

    Attributes:
        case_id: Unique identifier for this case.
        history: List of completed activity labels in execution order.
        pending_end: Optional tuple of (end_timestamp, activity) for the
            currently executing activity awaiting its end event.
        model: The ProcessModel governing this case's behavior.
        depth: Nesting depth (0 for top-level cases).
        temporal_scale: Multiplicative factor for activity durations,
            used to scale subprocess timing relative to the parent.
    """

    def __init__(
        self,
        case_id: str,
        model: ProcessModel,
        depth: int = 0,
        temporal_scale: float = 1.0,
    ) -> None:
        """Initialize an active case.

        Args:
            case_id: Unique identifier for this case.
            model: Process model governing transitions.
            depth: Nesting depth (0 for top-level).
            temporal_scale: Duration scaling factor.
        """
        self.case_id = case_id
        self.history: List[str] = []
        self.pending_end: Optional[Tuple[float, str]] = None
        self.model = model
        self.depth = depth
        self.temporal_scale = temporal_scale
        self.finished = False


class SimulationEngine:
    """Event stream simulation engine with variable-order Markov transitions.

    Generates an unbounded stream of :class:`Event` objects ordered by
    ``arrival_timestamp``.  Multiple cases execute concurrently, each
    following a variable-order Markov model extracted from its process
    model.

    The engine uses a priority queue (min-heap) keyed on timestamp to
    determine which event fires next across all active cases.
    """

    def __init__(
        self,
        model: ProcessModel,
        params: SimulationParams,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the simulation engine.

        Args:
            model: The process model to simulate.
            params: Simulation parameters controlling timing, concurrency,
                Markov order, and subprocess behavior.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.params = params
        self.rng = np.random.RandomState(seed)
        self._case_counter = 0
        self._event_heap: List[Tuple[float, int, Event]] = []
        self._heap_counter = 0

    def run(self, max_events: int = 0) -> Iterator[Event]:
        """Generate an event stream.

        Yields events ordered by arrival_timestamp.  If ``max_events`` is 0,
        the stream is unbounded (runs until all cases complete and no new
        arrivals are scheduled).

        Args:
            max_events: Maximum number of events to yield. 0 for unlimited.

        Yields:
            Event objects ordered by arrival_timestamp.
        """
        active_cases: List[_ActiveCase] = []
        global_clock = 0.0
        next_arrival_time = 0.0
        event_count = 0

        transition_matrix = self.model.get_transition_matrix(
            self.params.markov_order
        )

        while True:
            if max_events > 0 and event_count >= max_events:
                break

            while (
                next_arrival_time <= global_clock
                and len(active_cases) < self.params.max_concurrent_cases
            ):
                case = self._create_case(self.model)
                active_cases.append(case)
                self._schedule_next_activity(case, next_arrival_time, transition_matrix)

                inter_arrival = self.rng.exponential(
                    1.0 / max(self.params.case_arrival_rate, 1e-6)
                )
                next_arrival_time += inter_arrival

            if not self._event_heap and not active_cases:
                break

            if not self._event_heap:
                if next_arrival_time > global_clock:
                    global_clock = next_arrival_time
                    continue
                break

            timestamp, _, event = heapq.heappop(self._event_heap)
            global_clock = max(global_clock, timestamp)

            yield event
            event_count += 1

            if max_events > 0 and event_count >= max_events:
                break

            case = self._find_case(active_cases, event.case_id)
            if case is None:
                continue

            if event.event_type == "start":
                duration = self._sample_duration(case.temporal_scale)
                end_time = event.timestamp + duration
                end_event = Event(
                    case_id=case.case_id,
                    activity=event.activity,
                    timestamp=end_time,
                    event_type="end",
                    arrival_timestamp=end_time,
                )
                self._push_event(end_event)

            elif event.event_type == "end":
                case.history.append(event.activity)

                if (
                    self.params.subprocess_probability > 0
                    and case.depth < self.params.subprocess_max_depth
                    and self.rng.random() < self.params.subprocess_probability
                ):
                    child_model = case.model.get_subprocess_model(
                        self.rng, simplify=True
                    )
                    child_case = self._create_case(
                        child_model,
                        parent_id=case.case_id,
                        depth=case.depth + 1,
                        temporal_scale=(
                            case.temporal_scale
                            * self.params.subprocess_temporal_scale
                        ),
                    )
                    active_cases.append(child_case)
                    child_tm = child_model.get_transition_matrix(
                        self.params.markov_order
                    )
                    self._schedule_next_activity(
                        child_case, event.timestamp, child_tm
                    )

                self._schedule_next_activity(
                    case, event.timestamp, transition_matrix
                )

                if case.finished:
                    active_cases = [c for c in active_cases if c is not case]

    def _create_case(
        self,
        model: ProcessModel,
        parent_id: Optional[str] = None,
        depth: int = 0,
        temporal_scale: float = 1.0,
    ) -> _ActiveCase:
        """Create a new active case with a unique ID.

        Args:
            model: Process model for this case.
            parent_id: Parent case ID (for subprocess hierarchy).
            depth: Nesting depth.
            temporal_scale: Duration scaling factor.

        Returns:
            A new _ActiveCase instance.
        """
        self._case_counter += 1
        if parent_id is not None:
            case_id = f"{parent_id}.{self._case_counter}"
        else:
            case_id = f"case_{self._case_counter}"

        return _ActiveCase(
            case_id=case_id,
            model=model,
            depth=depth,
            temporal_scale=temporal_scale,
        )

    def _schedule_next_activity(
        self,
        case: _ActiveCase,
        current_time: float,
        transition_matrix: Dict[Tuple[str, ...], Dict[str, float]],
    ) -> None:
        """Determine and schedule the next activity for a case.

        Uses the variable-order Markov transition matrix.  If the context
        is not found exactly, falls back to shorter contexts progressively
        until a match is found.

        Args:
            case: The active case to advance.
            current_time: Current simulation time.
            transition_matrix: Variable-order transition probability matrix.
        """
        order = self.params.markov_order
        next_activity = self._select_next_activity(
            case, order, transition_matrix
        )

        if next_activity is None or next_activity == _END_SYMBOL:
            case.finished = True
            return

        start_event = Event(
            case_id=case.case_id,
            activity=next_activity,
            timestamp=current_time,
            event_type="start",
            arrival_timestamp=current_time,
        )
        self._push_event(start_event)

    def _select_next_activity(
        self,
        case: _ActiveCase,
        order: int,
        transition_matrix: Dict[Tuple[str, ...], Dict[str, float]],
    ) -> Optional[str]:
        """Select the next activity using variable-order Markov with fallback.

        Tries the full order context first; if no match, progressively
        reduces order until a match is found.  Blends with context-sensitive
        adjustments when ``context_sensitivity`` > 0.

        Args:
            case: The active case whose history determines the context.
            order: Maximum Markov order to try.
            transition_matrix: The transition probability matrix.

        Returns:
            The selected next activity label, or None if no valid transition.
        """
        padded_history = [_START_SYMBOL] * order + case.history

        for current_order in range(order, 0, -1):
            context = tuple(padded_history[-current_order:])
            if context in transition_matrix:
                dist = transition_matrix[context]
                return self._sample_from_distribution(dist, case)

        start_context = tuple([_START_SYMBOL] * order)
        if start_context in transition_matrix:
            dist = transition_matrix[start_context]
            return self._sample_from_distribution(dist, case)

        if self.model.activities:
            return str(self.rng.choice(self.model.activities))

        return None

    def _sample_from_distribution(
        self,
        dist: Dict[str, float],
        case: _ActiveCase,
    ) -> str:
        """Sample an activity from a probability distribution.

        Applies context sensitivity blending: with probability
        ``context_sensitivity``, the distribution is modified based on
        whether certain activities have appeared earlier in the case history,
        boosting rare transitions to increase non-linear dependency.

        Args:
            dist: Dictionary mapping activity labels to probabilities.
            case: The active case (for history-based adjustments).

        Returns:
            The selected activity label.
        """
        activities = list(dist.keys())
        probs = np.array([dist[a] for a in activities], dtype=float)

        if (
            self.params.context_sensitivity > 0
            and len(case.history) > 1
        ):
            history_set = set(case.history)
            adjustment = np.ones(len(activities), dtype=float)
            for i, act in enumerate(activities):
                if act == _END_SYMBOL:
                    continue
                if act in history_set:
                    adjustment[i] *= (1.0 - 0.5 * self.params.context_sensitivity)
                else:
                    adjustment[i] *= (1.0 + 0.3 * self.params.context_sensitivity)

            probs = probs * adjustment

        prob_sum = probs.sum()
        if prob_sum < 1e-12:
            return activities[0]
        probs = probs / prob_sum

        idx = self.rng.choice(len(activities), p=probs)
        return activities[idx]

    def _sample_duration(self, temporal_scale: float) -> float:
        """Sample an activity duration from an exponential distribution.

        Args:
            temporal_scale: Multiplicative scale factor (for subprocesses).

        Returns:
            Duration in time units.
        """
        base_duration = self.rng.exponential(self.params.mean_activity_duration)
        variance_factor = max(self.params.duration_variance_factor, 0.01)
        duration = base_duration * variance_factor * temporal_scale
        return max(duration, 0.01)

    def _push_event(self, event: Event) -> None:
        """Push an event onto the priority queue.

        Args:
            event: Event to schedule.
        """
        self._heap_counter += 1
        heapq.heappush(
            self._event_heap,
            (event.arrival_timestamp, self._heap_counter, event),
        )

    @staticmethod
    def _find_case(
        cases: List[_ActiveCase],
        case_id: str,
    ) -> Optional[_ActiveCase]:
        """Find an active case by its ID.

        Args:
            cases: List of currently active cases.
            case_id: The case ID to search for.

        Returns:
            The matching _ActiveCase, or None if not found.
        """
        for case in cases:
            if case.case_id == case_id:
                return case
        return None
