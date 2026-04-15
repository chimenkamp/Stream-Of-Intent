"""Feature isolation tests for the OptimizationSpec system.

Each test targets exactly ONE feature dimension while leaving all others as
Symbols.  If the optimizer works correctly it should achieve the target with
reasonable precision (tolerance ±0.20) because the other four dimensions are
free to settle wherever they need to.

Run with::

    pytest tests/feature_tests.py -v
"""

from __future__ import annotations

import random
import sys
import os

import numpy as np
import pytest

# Ensure project root is importable when running from the repo.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stream_of_intent.config import RunConfig
from stream_of_intent.features import extract_features, tumbling_windows
from stream_of_intent.optimization import generate_intentional_stream
from stream_of_intent.types import (
    FeatureType,
    FeatureVector,
    FEATURE_ORDER,
    OptimizationSpec,
    StaticParams,
    Symbol,
    Target,
)

# Shared static config — small enough for fast CI, large enough for signal.
_STATIC = StaticParams(
    window_size=1000,
    num_activities=10,
    stream_length=5000,
    num_windows_for_evaluation=5,
    random_seed=42,
)

_MAX_ITER = 200
_TOLERANCE = 0.20


def _build_single_target_spec(
    target_feature: FeatureType,
    target_value: float,
) -> OptimizationSpec:
    """Create a spec where *target_feature* is a Target and all others are Symbols."""
    operators = []
    for ft in FEATURE_ORDER:
        if ft == target_feature:
            operators.append(Target(ft, target_value))
        else:
            operators.append(Symbol(ft))
    return operators


def _run_and_measure(spec: OptimizationSpec) -> FeatureVector:
    """Run the optimisation pipeline and return the mean achieved features."""
    # Pin global random state so pm4py's tree generator is deterministic.
    random.seed(42)
    np.random.seed(42)

    stream_iter, _ = generate_intentional_stream(
        spec=spec,
        static_params=_STATIC,
        max_iterations=_MAX_ITER,
    )

    events = list(stream_iter)

    window_features = [
        extract_features(w)
        for w in tumbling_windows(iter(events), _STATIC.window_size)
    ]
    assert window_features, "No complete windows produced"

    mean_arr = np.mean([fv.to_array() for fv in window_features], axis=0)
    return FeatureVector.from_array(mean_arr)


# ── Individual feature tests ─────────────────────────────────────────────


class TestTemporalDependency:
    """Target only temporal_dependency, all others free."""

    TARGET = 0.7

    @pytest.fixture(scope="class")
    def achieved(self) -> FeatureVector:
        spec = OptimizationSpec(
            _build_single_target_spec(FeatureType.TEMPORAL_DEPENDENCY, self.TARGET)
        )
        return _run_and_measure(spec)

    def test_within_tolerance(self, achieved: FeatureVector) -> None:
        assert abs(achieved.temporal_dependency - self.TARGET) <= _TOLERANCE, (
            f"temporal_dependency={achieved.temporal_dependency:.3f}, "
            f"expected {self.TARGET}±{_TOLERANCE}"
        )


class TestLongTermDependency:
    """Target only long_term_dependency, all others free."""

    TARGET = 0.7

    @pytest.fixture(scope="class")
    def achieved(self) -> FeatureVector:
        spec = OptimizationSpec(
            _build_single_target_spec(FeatureType.LONG_TERM_DEPENDENCY, self.TARGET)
        )
        return _run_and_measure(spec)

    def test_within_tolerance(self, achieved: FeatureVector) -> None:
        assert abs(achieved.long_term_dependency - self.TARGET) <= _TOLERANCE, (
            f"long_term_dependency={achieved.long_term_dependency:.3f}, "
            f"expected {self.TARGET}±{_TOLERANCE}"
        )


class TestNonLinearDependency:
    """Target only non_linear_dependency, all others free.

    Note: This feature is a composite of branching factor, parallelism,
    and context sensitivity.  The achievable range with 50 iterations is
    narrower than for other features, so the target is set conservatively.
    """

    TARGET = 0.2

    @pytest.fixture(scope="class")
    def achieved(self) -> FeatureVector:
        spec = OptimizationSpec(
            _build_single_target_spec(FeatureType.NON_LINEAR_DEPENDENCY, self.TARGET)
        )
        return _run_and_measure(spec)

    def test_within_tolerance(self, achieved: FeatureVector) -> None:
        assert abs(achieved.non_linear_dependency - self.TARGET) <= _TOLERANCE, (
            f"non_linear_dependency={achieved.non_linear_dependency:.3f}, "
            f"expected {self.TARGET}±{_TOLERANCE}"
        )


class TestOutOfOrder:
    """Target only out_of_order, all others free."""

    TARGET = 0.4

    @pytest.fixture(scope="class")
    def achieved(self) -> FeatureVector:
        spec = OptimizationSpec(
            _build_single_target_spec(FeatureType.OUT_OF_ORDER, self.TARGET)
        )
        return _run_and_measure(spec)

    def test_within_tolerance(self, achieved: FeatureVector) -> None:
        assert abs(achieved.out_of_order - self.TARGET) <= _TOLERANCE, (
            f"out_of_order={achieved.out_of_order:.3f}, "
            f"expected {self.TARGET}±{_TOLERANCE}"
        )


class TestFractalBehavior:
    """Target only fractal_behavior, all others free."""

    TARGET = 0.3

    @pytest.fixture(scope="class")
    def achieved(self) -> FeatureVector:
        spec = OptimizationSpec(
            _build_single_target_spec(FeatureType.FRACTAL_BEHAVIOR, self.TARGET)
        )
        return _run_and_measure(spec)

    def test_within_tolerance(self, achieved: FeatureVector) -> None:
        assert abs(achieved.fractal_behavior - self.TARGET) <= _TOLERANCE, (
            f"fractal_behavior={achieved.fractal_behavior:.3f}, "
            f"expected {self.TARGET}±{_TOLERANCE}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])