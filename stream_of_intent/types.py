from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pm4py

class FeatureType(str, Enum):
    """Enumeration of the five event-stream feature dimensions."""

    TEMPORAL_DEPENDENCY = "temporal_dependency"
    LONG_TERM_DEPENDENCY = "long_term_dependency"
    NON_LINEAR_DEPENDENCY = "non_linear_dependency"
    OUT_OF_ORDER = "out_of_order"
    FRACTAL_BEHAVIOR = "fractal_behavior"

FEATURE_ORDER: List[FeatureType] = list(FeatureType)

@dataclass(frozen=True)
class Expression:
    """A simple comparison expression for use with :class:`Condition`.

    Supports ``<``, ``>``, ``<=``, ``>=``, and ``between`` operators.
    For ``between``, both *value* (lower bound) and *upper* (upper bound)
    are required.

    Attributes:
        operator: Comparison operator string.
        value: Threshold or lower-bound value.
        upper: Upper-bound value (only used when *operator* is ``"between"``).
    """

    operator: Literal["<", ">", "<=", ">=", "between"]
    value: float
    upper: Optional[float] = None

    def satisfied(self, x: float) -> bool:
        """Return True if *x* satisfies this expression."""
        if self.operator == "<":
            return x < self.value
        if self.operator == ">":
            return x > self.value
        if self.operator == "<=":
            return x <= self.value
        if self.operator == ">=":
            return x >= self.value
        if self.operator == "between":
            return self.value <= x <= (self.upper or self.value)
        return False

    def penalty(self, x: float) -> float:
        """Distance from the nearest satisfying value (0 if satisfied)."""
        if self.satisfied(x):
            return 0.0

        if self.operator in ("<", "<="):
            return x - self.value
        if self.operator in (">", ">="):
            return self.value - x
        if self.operator == "between":
            lo = self.value
            hi = self.upper if self.upper is not None else lo
            if x < lo:
                return lo - x
            return x - hi

        return 0.0

    def __str__(self) -> str:
        if self.operator == "between":
            return f"between({self.value}, {self.upper})"
        return f"{self.operator} {self.value}"


@dataclass(frozen=True)
class Target:
    """Optimise a feature to match a precise numeric value.

    Attributes:
        feature: Which feature this operator applies to.
        value: The desired value in [0, 1].
    """

    feature: FeatureType
    value: float


@dataclass(frozen=True)
class Symbol:
    """Let the optimiser freely determine this feature's value.

    The feature is *not* penalised in the objective function.  The optimiser
    will set pipeline parameters however it likes for this dimension — its
    only purpose is to help satisfy the :class:`Target` and
    :class:`Condition` operators on other features.

    Attributes:
        feature: Which feature this operator applies to.
    """

    feature: FeatureType


@dataclass(frozen=True)
class Condition:
    """Constrain a feature with a relational expression instead of a point value.

    The optimiser receives zero penalty when the expression is satisfied
    and a penalty proportional to the distance from the nearest satisfying
    boundary otherwise.

    Attributes:
        feature: Which feature this operator applies to.
        expression: The constraint expression (e.g. ``Expression("<", 0.5)``).
    """

    feature: FeatureType
    expression: Expression


@dataclass(frozen=True)
class Exclude:
    """Completely ignore a feature during optimisation and reporting.

    Attributes:
        feature: Which feature this operator applies to.
    """

    feature: FeatureType


FeatureOperator = Union[Target, Symbol, Condition, Exclude]

@dataclass
class OptimizationSpec:
    """Describes how each feature dimension participates in the objective.

    Holds a list of :class:`FeatureOperator` items — one per feature.  Any
    feature not explicitly listed defaults to :class:`Exclude`.

    Provides :meth:`distance` which replaces the old Euclidean distance on
    the full five-dimensional vector with an operator-aware objective value.
    """

    operators: List[FeatureOperator] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._map: Dict[FeatureType, FeatureOperator] = {}
        for op in self.operators:
            self._map[op.feature] = op

    def get(self, feature: FeatureType) -> FeatureOperator:
        """Return the operator for *feature*, defaulting to Exclude."""
        return self._map.get(feature, Exclude(feature))

    def distance(self, achieved: FeatureVector) -> float:
        """Compute the operator-aware objective value.

        - :class:`Target`: squared error ``(achieved - target)²``
        - :class:`Condition`: squared penalty from nearest boundary
        - :class:`Symbol` / :class:`Exclude`: 0 (no contribution)

        Returns the square root of the sum (i.e. an Euclidean-like norm
        over the active dimensions).
        """
        total = 0.0
        for ft in FEATURE_ORDER:
            op = self.get(ft)
            val = achieved.get(ft)
            if isinstance(op, Target):
                total += (val - op.value) ** 2
            elif isinstance(op, Condition):
                total += op.expression.penalty(val) ** 2
            # Symbol and Exclude contribute nothing.
        return math.sqrt(total)

    def targets_as_feature_vector(self) -> FeatureVector:
        """Build a FeatureVector using target values where available.

        For :class:`Target` operators the target value is used.  For all
        other operator types the value is 0.0 (placeholder).
        """
        values = {}
        for ft in FEATURE_ORDER:
            op = self.get(ft)
            if isinstance(op, Target):
                values[ft.value] = op.value  # ft.value is the str name
            else:
                values[ft.value] = 0.0
        return FeatureVector(**values)

    def to_metadata(self) -> Dict:
        """Serialise the spec into a JSON-friendly dict for model metadata."""
        entries = {}
        for ft in FEATURE_ORDER:
            op = self.get(ft)
            if isinstance(op, Target):
                entries[ft.value] = {"type": "target", "value": op.value}
            elif isinstance(op, Symbol):
                entries[ft.value] = {"type": "symbol"}
            elif isinstance(op, Condition):
                entries[ft.value] = {
                    "type": "condition",
                    "expression": str(op.expression),
                }
            else:
                entries[ft.value] = {"type": "exclude"}
        return entries

    @classmethod
    def from_targets(cls, targets: FeatureVector) -> OptimizationSpec:
        """Create a spec where every feature is a Target (legacy compat)."""
        return cls([
            Target(ft, getattr(targets, ft.value))
            for ft in FEATURE_ORDER
        ])


@dataclass(frozen=True)
class Event:
    """A single event in an event stream.

    Following the interval-based event stream formalization from the paper,
    each activity is represented by a pair of start/end events.  The stream
    is ordered by ``arrival_timestamp``, which may differ from ``timestamp``
    to model out-of-order observation.

    Attributes:
        case_id: Identifier of the process instance this event belongs to.
        activity: Name of the activity being performed.
        timestamp: When the event actually occurred in the process.
        event_type: Whether this marks the start or end of an activity.
        arrival_timestamp: When the event was observed by the analysis system.
    """

    case_id: str
    activity: str
    timestamp: float
    event_type: Literal["start", "end"]
    arrival_timestamp: float

    def get_pm4py_event(self) -> pm4py.objects.log.obj.Event:
        """Convert to a pm4py-event."""
        return pm4py.objects.log.obj.Event(
            case_id=self.case_id,
            activity=self.activity,
            timestamp=self.timestamp,
            event_type=self.event_type,
            arrival_timestamp=self.arrival_timestamp,
        )



@dataclass
class FeatureVector:
    """Five-dimensional feature vector characterizing an event stream window.

    All values are normalized to [0, 1].

    Attributes:
        temporal_dependency: Degree to which the next activity depends on the
            immediately preceding activity (short-range predictability).
        long_term_dependency: Degree to which distant past events influence
            current activity selection (requires memory beyond last event).
        non_linear_dependency: Degree of complex branching, parallelism, and
            context-aware decisions in the process.
        out_of_order: Proportion and magnitude of events arriving out of their
            true temporal order.
        fractal_behavior: Degree of self-similar, nested subprocess patterns
            repeating at different scales.
    """

    temporal_dependency: float = 0.0
    long_term_dependency: float = 0.0
    non_linear_dependency: float = 0.0
    out_of_order: float = 0.0
    fractal_behavior: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to a numpy array of shape (5,).

        Returns:
            Array with the five feature values in canonical order.
        """
        return np.array([
            self.temporal_dependency,
            self.long_term_dependency,
            self.non_linear_dependency,
            self.out_of_order,
            self.fractal_behavior,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> FeatureVector:
        """Construct a FeatureVector from a numpy array of shape (5,).

        Args:
            arr: Array with five feature values in canonical order.

        Returns:
            A new FeatureVector instance.
        """
        return cls(
            temporal_dependency=float(arr[0]),
            long_term_dependency=float(arr[1]),
            non_linear_dependency=float(arr[2]),
            out_of_order=float(arr[3]),
            fractal_behavior=float(arr[4]),
        )

    def distance(self, other: FeatureVector) -> float:
        """Euclidean distance to another feature vector.

        Args:
            other: The target feature vector to compare against.

        Returns:
            Euclidean distance in the 5D feature space.
        """
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def get(self, feature: FeatureType) -> float:
        """Look up a feature value by :class:`FeatureType`.

        Args:
            feature: The feature dimension to retrieve.

        Returns:
            The scalar value for that dimension.
        """
        return float(getattr(self, feature.value))


@dataclass
class ModelParams:
    """Tunable parameters for process model generation.

    These parameters are subject to Bayesian optimization to steer the
    generated process model toward desired stream characteristics.

    Attributes:
        num_activities: Number of distinct activities in the process model.
        sequence_weight: Probability weight for sequential operator in process
            tree generation. Higher values yield more linear processes.
        parallel_weight: Probability weight for parallel operator. Higher
            values yield more concurrent behavior.
        loop_weight: Probability weight for loop operator. Higher values yield
            more repetitive behavior.
        choice_weight: Probability weight for choice (XOR) operator. Higher
            values yield more branching.
        nesting_depth: Maximum depth of nested subprocess structures. Controls
            hierarchical complexity.
        loop_probability: Probability that a loop iteration continues vs exits.
        skip_connection_probability: Probability of adding long-range skip
            connections in the model, enabling dependencies beyond immediate
            neighbors.
        silent_transition_probability: Probability of invisible (tau)
            transitions, enabling optional behavior.
    """

    num_activities: int = 10
    sequence_weight: float = 0.3
    parallel_weight: float = 0.2
    loop_weight: float = 0.2
    choice_weight: float = 0.3
    nesting_depth: int = 3
    loop_probability: float = 0.3
    skip_connection_probability: float = 0.1
    silent_transition_probability: float = 0.1


@dataclass
class SimulationParams:
    """Tunable parameters for the simulation engine.

    Control how events are generated from the process model, including
    timing, concurrency, and memory depth.

    Attributes:
        markov_order: Order of the Markov model used for activity transitions.
            Order 1 gives standard first-order chain; higher orders enable
            long-term dependencies by conditioning on more history.
        case_arrival_rate: Mean rate of new case arrivals (cases per time unit).
            Controls how many cases are active concurrently.
        mean_activity_duration: Mean duration of an activity instance. Actual
            durations are drawn from an exponential distribution with this mean.
        duration_variance_factor: Multiplicative factor controlling the variance
            of activity durations relative to the mean.
        max_concurrent_cases: Maximum number of concurrently active cases.
        subprocess_probability: Probability that completing an activity spawns
            a child subprocess case (for fractal behavior).
        subprocess_temporal_scale: Temporal scaling factor applied to child
            subprocess durations relative to the parent.
        subprocess_max_depth: Maximum nesting depth for subprocess spawning.
        context_sensitivity: Degree to which transition probabilities depend on
            full case history vs only the last N events. 0.0 = pure Markov,
            1.0 = fully context-aware.
    """

    markov_order: int = 1
    case_arrival_rate: float = 1.0
    mean_activity_duration: float = 5.0
    duration_variance_factor: float = 1.0
    max_concurrent_cases: int = 10
    subprocess_probability: float = 0.0
    subprocess_temporal_scale: float = 0.5
    subprocess_max_depth: int = 2
    context_sensitivity: float = 0.0


@dataclass
class PostprocessorParams:
    """Tunable parameters for stream postprocessors.

    Control the injection of out-of-order behavior and fractal patterns
    after initial stream generation.

    Attributes:
        ooo_fraction: Fraction of events whose arrival_timestamp is displaced
            from their timestamp to create out-of-order observations.
        ooo_max_displacement: Maximum delay (in time units) that can be added
            to an event's arrival_timestamp during out-of-order injection.
        fractal_scale_factor: Temporal scaling factor for fractal pattern
            replication at different scales.
        fractal_replication_probability: Probability that a trace pattern is
            replicated at a different temporal scale by the fractal injector.
    """

    ooo_fraction: float = 0.0
    ooo_max_displacement: float = 10.0
    fractal_scale_factor: float = 0.5
    fractal_replication_probability: float = 0.0


@dataclass
class StaticParams:
    """Non-optimized configuration parameters.

    These values remain fixed during optimization and define the overall
    scope and structure of the generated stream.

    Attributes:
        window_size: Number of events per tumbling window for feature
            extraction.
        num_activities: Number of distinct activities in the process.
        stream_length: Total number of events to generate for each
            optimization evaluation.
        num_windows_for_evaluation: Number of windows to average over when
            computing feature distance during optimization.
        random_seed: Random seed for reproducibility. None for non-deterministic.
    """

    window_size: int = 1000
    num_activities: int = 10
    stream_length: int = 5000
    num_windows_for_evaluation: int = 5
    random_seed: Optional[int] = 42


@dataclass
class OptimizableParams:
    """Combined parameter container passed through the optimization loop.

    Bundles all parameters that the Bayesian optimizer can tune.

    Attributes:
        model: Process model generation parameters.
        simulation: Simulation engine parameters.
        postprocessor: Stream postprocessor parameters.
    """

    model: ModelParams = field(default_factory=ModelParams)
    simulation: SimulationParams = field(default_factory=SimulationParams)
    postprocessor: PostprocessorParams = field(default_factory=PostprocessorParams)
