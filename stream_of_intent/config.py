from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from stream_of_intent.types import FeatureVector, OptimizationSpec, StaticParams


@dataclass
class StreamConfig:
    """Configuration for streaming an abstract model representation.

    Attributes:
        adapter: Transport adapter to use for the event stream output.
        host: Network address the adapter binds or connects to.
        port: Network port for the adapter.
        topic: Logical channel name (Kafka topic, WebSocket path, …).
        max_loops: How many times the simulation is replayed.
            ``0`` means infinite looping.
        event_format: Wire serialization format for individual events.
    """

    adapter: Literal["console", "websocket", "kafka"] = "console"
    host: str = "localhost"
    port: int = 8765
    topic: str = "stream-of-intent"
    max_loops: int = 0
    event_format: Literal["json"] = "json"


@dataclass
class RunConfig:
    """Complete configuration for a Stream of Intent run.

    Attributes:
        spec: Symbolic optimisation specification describing how each
            feature dimension participates in the objective.  When
            provided, *targets* is ignored.
        targets: (Legacy) Desired feature vector the optimiser tries to
            match.  Converted to an :class:`OptimizationSpec` of all
            :class:`Target` operators when *spec* is ``None``.
        static: Fixed parameters that are NOT subject to optimisation
            (stream size, window size, activity count, seed, …).
        max_iterations: Maximum number of Bayesian optimisation trials.
        output_path: CSV file path for the generated stream.
            ``None`` writes to stdout.
        verbose: If ``True``, log at DEBUG level; otherwise INFO.
        stream: Optional streaming configuration.  When provided, the
            optimised model can be played out as a live event stream
            through the specified adapter.
    """

    spec: Optional[OptimizationSpec] = None
    targets: FeatureVector = field(default_factory=FeatureVector)
    static: StaticParams = field(default_factory=StaticParams)
    max_iterations: int = 50
    output_path: Optional[str] = None
    verbose: bool = False
    stream: Optional[StreamConfig] = None

    @property
    def effective_spec(self) -> OptimizationSpec:
        """Return the active spec, converting *targets* if needed."""
        if self.spec is not None:
            return self.spec
        return OptimizationSpec.from_targets(self.targets)
