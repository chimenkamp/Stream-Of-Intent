from __future__ import annotations

import logging
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
import optuna

from stream_of_intent.features import extract_features, tumbling_windows
from stream_of_intent.model_representation import (
    AbstractModelRepresentation,
    encode_transition_matrices,
)
from stream_of_intent.postprocessors import apply_postprocessors
from stream_of_intent.process_model import ProcessModelGenerator
from stream_of_intent.simulation import SimulationEngine
from stream_of_intent.types import (
    Event,
    Exclude,
    FeatureType,
    FeatureVector,
    ModelParams,
    OptimizableParams,
    OptimizationSpec,
    PostprocessorParams,
    SimulationParams,
    StaticParams,
)

from typing import NamedTuple


class _ExclusionFlags(NamedTuple):
    """Boolean flags indicating which feature dimensions are excluded."""

    fractal: bool
    out_of_order: bool
    long_term: bool
    non_linear: bool

logger = logging.getLogger(__name__)


def generate_intentional_stream(
    targets: FeatureVector | OptimizationSpec | None = None,
    static_params: Optional[StaticParams] = None,
    max_iterations: int = 50,
    *,
    spec: Optional[OptimizationSpec] = None,
    study_callbacks: Optional[List[callable]] = None,
) -> Tuple[Iterator[Event], AbstractModelRepresentation]:
    """Generate an event stream matching target feature values.

    Runs Bayesian optimization to find pipeline parameters that produce
    streams whose extracted features are closest to the given targets,
    then returns an unbounded stream from the best configuration together
    with an :class:`AbstractModelRepresentation` that can be serialised
    and replayed independently.

    Args:
        targets: Legacy target feature vector.  Converted internally to an
            :class:`OptimizationSpec` of all :class:`Target` operators.
            Ignored when *spec* is provided.
        static_params: Non-optimized configuration (window size, stream
            length, etc.).  Uses defaults if not provided.
        max_iterations: Maximum number of Bayesian optimization iterations.
        spec: Symbolic optimisation specification.  Takes precedence over
            *targets* when both are supplied.

    Returns:
        A tuple of ``(event_iterator, model_representation)``.
    """
    # Resolve the effective spec.
    if spec is not None:
        effective_spec = spec
    elif isinstance(targets, OptimizationSpec):
        effective_spec = targets
    elif isinstance(targets, FeatureVector):
        effective_spec = OptimizationSpec.from_targets(targets)
    else:
        effective_spec = OptimizationSpec()

    if static_params is None:
        static_params = StaticParams()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=static_params.random_seed,
        ),
    )

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: operator-aware distance to spec.

        Args:
            trial: Optuna trial providing parameter suggestions.

        Returns:
            Objective value computed by the OptimizationSpec.
        """
        params = _suggest_params(trial, static_params, effective_spec)
        distance = _evaluate_pipeline(params, effective_spec, static_params)
        return distance

    study.optimize(
        objective,
        n_trials=max_iterations,
        show_progress_bar=False,
        callbacks=study_callbacks or [],
    )

    best_params = _reconstruct_params(study.best_trial, static_params, effective_spec)
    logger.info(
        "Optimization complete. Best distance: %.4f",
        study.best_value,
    )

    model_rep = _build_model_representation(
        best_params, static_params, effective_spec, study.best_value,
    )

    return _run_pipeline(best_params, static_params), model_rep


def _suggest_params(
    trial: optuna.Trial,
    static: StaticParams,
    spec: OptimizationSpec | None = None,
) -> OptimizableParams:
    """Suggest a set of optimizable parameters for one trial."""
    flags = _compute_exclusion_flags(spec)
    model = _suggest_model_params(trial, static, flags)
    simulation = _suggest_simulation_params(trial, flags)
    postprocessor = _suggest_postprocessor_params(trial, flags)
    return OptimizableParams(model=model, simulation=simulation, postprocessor=postprocessor)


def _suggest_model_params(
    trial: optuna.Trial,
    static: StaticParams,
    flags: _ExclusionFlags,
) -> ModelParams:
    """Suggest model generation parameters for one trial."""
    return ModelParams(
        num_activities=static.num_activities,
        sequence_weight=trial.suggest_float("sequence_weight", 0.05, 0.8),
        parallel_weight=(
            0.0 if flags.non_linear
            else trial.suggest_float("parallel_weight", 0.0, 0.6)
        ),
        loop_weight=trial.suggest_float("loop_weight", 0.0, 0.6),
        choice_weight=trial.suggest_float("choice_weight", 0.05, 0.6),
        nesting_depth=trial.suggest_int("nesting_depth", 1, 5),
        loop_probability=trial.suggest_float("loop_probability", 0.0, 0.8),
        skip_connection_probability=trial.suggest_float(
            "skip_connection_probability", 0.0, 0.5,
        ),
        silent_transition_probability=trial.suggest_float(
            "silent_transition_probability", 0.0, 0.3,
        ),
    )


def _suggest_simulation_params(
    trial: optuna.Trial,
    flags: _ExclusionFlags,
) -> SimulationParams:
    """Suggest simulation parameters for one trial."""
    return SimulationParams(
        markov_order=(
            1 if flags.long_term
            else trial.suggest_int("markov_order", 1, 4)
        ),
        case_arrival_rate=trial.suggest_float("case_arrival_rate", 0.1, 5.0),
        mean_activity_duration=trial.suggest_float(
            "mean_activity_duration", 1.0, 20.0,
        ),
        duration_variance_factor=trial.suggest_float(
            "duration_variance_factor", 0.1, 3.0,
        ),
        max_concurrent_cases=trial.suggest_int("max_concurrent_cases", 1, 20),
        subprocess_probability=(
            0.0 if flags.fractal
            else trial.suggest_float("subprocess_probability", 0.0, 0.5)
        ),
        subprocess_temporal_scale=(
            0.1 if flags.fractal
            else trial.suggest_float("subprocess_temporal_scale", 0.1, 1.0)
        ),
        subprocess_max_depth=(
            0 if flags.fractal
            else trial.suggest_int("subprocess_max_depth", 0, 3)
        ),
        context_sensitivity=(
            0.0 if flags.non_linear
            else trial.suggest_float("context_sensitivity", 0.0, 1.0)
        ),
    )


def _suggest_postprocessor_params(
    trial: optuna.Trial,
    flags: _ExclusionFlags,
) -> PostprocessorParams:
    """Suggest postprocessor parameters for one trial."""
    return PostprocessorParams(
        ooo_fraction=(
            0.0 if flags.out_of_order
            else trial.suggest_float("ooo_fraction", 0.0, 0.8)
        ),
        ooo_max_displacement=(
            0.0 if flags.out_of_order
            else trial.suggest_float("ooo_max_displacement", 0.0, 50.0)
        ),
        fractal_scale_factor=(
            0.1 if flags.fractal
            else trial.suggest_float("fractal_scale_factor", 0.1, 2.0)
        ),
        fractal_replication_probability=(
            0.0 if flags.fractal
            else trial.suggest_float("fractal_replication_probability", 0.0, 0.5)
        ),
    )


def _excluded_features(spec: OptimizationSpec | None) -> set[FeatureType]:
    """Return the set of feature types that are :class:`Exclude`-d in *spec*."""
    if spec is None:
        return set()
    return {
        ft for ft in FeatureType if isinstance(spec.get(ft), Exclude)
    }


def _compute_exclusion_flags(spec: OptimizationSpec | None) -> _ExclusionFlags:
    """Determine which feature dimensions are excluded from optimization."""
    excluded = _excluded_features(spec)
    return _ExclusionFlags(
        fractal=FeatureType.FRACTAL_BEHAVIOR in excluded,
        out_of_order=FeatureType.OUT_OF_ORDER in excluded,
        long_term=FeatureType.LONG_TERM_DEPENDENCY in excluded,
        non_linear=FeatureType.NON_LINEAR_DEPENDENCY in excluded,
    )


def _reconstruct_params(
    trial: optuna.trial.FrozenTrial,
    static: StaticParams,
    spec: OptimizationSpec | None = None,
) -> OptimizableParams:
    """Reconstruct OptimizableParams from a completed trial's parameters."""
    flags = _compute_exclusion_flags(spec)
    p = trial.params
    model = _reconstruct_model_params(p, static, flags)
    simulation = _reconstruct_simulation_params(p, flags)
    postprocessor = _reconstruct_postprocessor_params(p, flags)
    return OptimizableParams(model=model, simulation=simulation, postprocessor=postprocessor)


def _reconstruct_model_params(
    p: dict,
    static: StaticParams,
    flags: _ExclusionFlags,
) -> ModelParams:
    """Reconstruct model parameters from a trial's parameter dictionary."""
    return ModelParams(
        num_activities=static.num_activities,
        sequence_weight=p["sequence_weight"],
        parallel_weight=0.0 if flags.non_linear else p["parallel_weight"],
        loop_weight=p["loop_weight"],
        choice_weight=p["choice_weight"],
        nesting_depth=p["nesting_depth"],
        loop_probability=p["loop_probability"],
        skip_connection_probability=p["skip_connection_probability"],
        silent_transition_probability=p["silent_transition_probability"],
    )


def _reconstruct_simulation_params(
    p: dict,
    flags: _ExclusionFlags,
) -> SimulationParams:
    """Reconstruct simulation parameters from a trial's parameter dictionary."""
    return SimulationParams(
        markov_order=1 if flags.long_term else p["markov_order"],
        case_arrival_rate=p["case_arrival_rate"],
        mean_activity_duration=p["mean_activity_duration"],
        duration_variance_factor=p["duration_variance_factor"],
        max_concurrent_cases=p["max_concurrent_cases"],
        subprocess_probability=0.0 if flags.fractal else p["subprocess_probability"],
        subprocess_temporal_scale=0.1 if flags.fractal else p["subprocess_temporal_scale"],
        subprocess_max_depth=0 if flags.fractal else p["subprocess_max_depth"],
        context_sensitivity=0.0 if flags.non_linear else p["context_sensitivity"],
    )


def _reconstruct_postprocessor_params(
    p: dict,
    flags: _ExclusionFlags,
) -> PostprocessorParams:
    """Reconstruct postprocessor parameters from a trial's parameter dictionary."""
    return PostprocessorParams(
        ooo_fraction=0.0 if flags.out_of_order else p["ooo_fraction"],
        ooo_max_displacement=0.0 if flags.out_of_order else p["ooo_max_displacement"],
        fractal_scale_factor=0.1 if flags.fractal else p["fractal_scale_factor"],
        fractal_replication_probability=0.0 if flags.fractal else p["fractal_replication_probability"],
    )


def _evaluate_pipeline(
    params: OptimizableParams,
    spec: OptimizationSpec,
    static: StaticParams,
) -> float:
    """Run the full pipeline and compute distance to the optimisation spec.

    Generates a stream, extracts features over multiple windows, and
    computes the operator-aware distance defined by the spec.

    Args:
        params: Pipeline parameters (model + simulation + postprocessor).
        spec: Symbolic optimisation specification.
        static: Static configuration parameters.

    Returns:
        Objective value computed by the spec's distance function.
    """
    try:
        stream = _run_pipeline(params, static)

        feature_vectors: List[FeatureVector] = []
        event_count = 0
        for window in tumbling_windows(stream, static.window_size):
            fv = extract_features(window)
            feature_vectors.append(fv)
            event_count += len(window)

            if len(feature_vectors) >= static.num_windows_for_evaluation:
                break
            if event_count >= static.stream_length:
                break

        if not feature_vectors:
            return 2.236

        mean_features = FeatureVector.from_array(
            np.mean([fv.to_array() for fv in feature_vectors], axis=0)
        )

        return spec.distance(mean_features)

    except Exception as exc:
        logger.debug("Pipeline evaluation failed: %s", exc)
        return 2.236


def _run_pipeline(
    params: OptimizableParams,
    static: StaticParams,
) -> Iterator[Event]:
    """Execute the full generation pipeline with given parameters.

    Creates the process model, runs the simulation engine, and applies
    postprocessors.

    Args:
        params: Pipeline parameters.
        static: Static configuration.

    Returns:
        Iterator of Event objects from the complete pipeline.
    """
    generator = ProcessModelGenerator()
    model = generator.generate(params.model, seed=static.random_seed)

    engine = SimulationEngine(
        model=model,
        params=params.simulation,
        seed=static.random_seed,
    )

    raw_stream = engine.run(max_events=static.stream_length)

    return apply_postprocessors(
        raw_stream,
        params.postprocessor,
        seed=static.random_seed,
    )


def _build_model_representation(
    params: OptimizableParams,
    static: StaticParams,
    spec: OptimizationSpec,
    best_distance: float,
) -> AbstractModelRepresentation:
    """Build an :class:`AbstractModelRepresentation` from the best trial.

    Re-generates the process model from the best parameters to capture
    its transition matrices and activity list.

    Args:
        params: Best-found optimisable parameters.
        static: Static configuration.
        spec: The optimisation specification used during optimisation.
        best_distance: Objective value achieved by the best trial.

    Returns:
        A fully populated :class:`AbstractModelRepresentation`.
    """
    generator = ProcessModelGenerator()
    model = generator.generate(params.model, seed=static.random_seed)

    encoded_matrices = _extract_encoded_transition_matrices(
        model, params.simulation.markov_order,
    )
    sub_activities, sub_matrices = _extract_subprocess_data(model, params, static)

    return AbstractModelRepresentation(
        optimized_params=params,
        static_params=static,
        activities=model.activities,
        transition_matrices=encoded_matrices,
        subprocess_activities=sub_activities,
        subprocess_transition_matrices=sub_matrices,
        metadata={
            "spec": spec.to_metadata(),
            "best_distance": best_distance,
            "generation_timestamp": time.time(),
        },
    )


def _extract_encoded_transition_matrices(
    model: ProcessModelGenerator,
    max_order: int,
) -> dict:
    """Extract and encode transition matrices for all orders up to max_order."""
    raw_matrices = {}
    for order in range(1, max_order + 1):
        raw_matrices[order] = model.get_transition_matrix(order)
    return encode_transition_matrices(raw_matrices)


def _extract_subprocess_data(
    model: ProcessModelGenerator,
    params: OptimizableParams,
    static: StaticParams,
) -> tuple:
    """Extract subprocess model activities and transition matrices if active."""
    if params.simulation.subprocess_probability <= 1e-6:
        return None, None
    rng = np.random.RandomState(static.random_seed)
    sub_model = model.get_subprocess_model(rng, simplify=True)
    sub_activities = sub_model.activities
    sub_matrices = _extract_encoded_transition_matrices(
        sub_model, params.simulation.markov_order,
    )
    return sub_activities, sub_matrices
