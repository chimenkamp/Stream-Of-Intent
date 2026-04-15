from stream_of_intent.config import RunConfig, StreamConfig
from stream_of_intent.model_representation import AbstractModelRepresentation
from stream_of_intent.stream_processor import StreamProcessor
from stream_of_intent.types import (
    Condition,
    Event,
    Exclude,
    Expression,
    FeatureType,
    FeatureVector,
    FEATURE_ORDER,
    ModelParams,
    OptimizationSpec,
    SimulationParams,
    StaticParams,
    Symbol,
    Target,
)

__all__ = [
    "AbstractModelRepresentation",
    "Condition",
    "Event",
    "Exclude",
    "Expression",
    "FeatureType",
    "FeatureVector",
    "FEATURE_ORDER",
    "ModelParams",
    "OptimizationSpec",
    "RunConfig",
    "SimulationParams",
    "StaticParams",
    "StreamConfig",
    "StreamProcessor",
    "Symbol",
    "Target",
    "generate_intentional_stream",
]


def generate_intentional_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Lazy import wrapper. See :func:`stream_of_intent.optimization.generate_intentional_stream`."""
    from stream_of_intent.optimization import generate_intentional_stream as _fn

    return _fn(*args, **kwargs)
