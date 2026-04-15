"""Run Stream of Intent with full configuration control.

All settings are configured via dataclasses.  Edit the ``RunConfig`` at the
bottom of this file to customise a run, then execute::

    python main.py
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import List, Optional, TextIO

import numpy as np

from stream_of_intent.adapters import create_adapter
from stream_of_intent.config import RunConfig, StreamConfig
from stream_of_intent.features import extract_features, tumbling_windows
from stream_of_intent.optimization import generate_intentional_stream
from stream_of_intent.stream_processor import StreamProcessor
from stream_of_intent.types import (
    Condition,
    Exclude,
    Expression,
    FeatureType,
    FeatureVector,
    FEATURE_ORDER,
    OptimizationSpec,
    StaticParams,
    Symbol,
    Target,
)


def run(config: RunConfig) -> None:
    """Execute the full Stream of Intent pipeline."""
    log = _setup_logging(config)
    _log_config(log, config)

    stream, model_rep = _run_optimization(config, log)
    _save_model_representation(model_rep, config, log)
    collected = _write_stream_csv(stream, config, log)

    _log_achieved(log, collected, config)
    log.info("=" * 60)
    log.info("Done.")


def _setup_logging(config: RunConfig) -> logging.Logger:
    """Configure logging and return the application logger."""
    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return logging.getLogger("stream_of_intent")


def _run_optimization(
    config: RunConfig,
    log: logging.Logger,
) -> tuple:
    """Run Bayesian optimisation and return the stream and model representation."""
    spec = config.effective_spec
    log.info("-" * 60)
    log.info("Starting Bayesian optimisation (%d iterations) …",
             config.max_iterations)
    t0 = time.time()

    stream, model_rep = generate_intentional_stream(
        spec=spec,
        static_params=config.static,
        max_iterations=config.max_iterations,
    )

    elapsed = time.time() - t0
    log.info("Optimisation completed in %.1f s.", elapsed)
    return stream, model_rep


def _save_model_representation(
    model_rep,
    config: RunConfig,
    log: logging.Logger,
) -> None:
    """Persist the abstract model representation alongside the CSV."""
    model_path = _model_path_for(config.output_path)
    model_rep.save(model_path)
    log.info("Saved model representation to %s", model_path)


def _write_stream_csv(
    stream,
    config: RunConfig,
    log: logging.Logger,
) -> List:
    """Write the event stream to CSV and return collected events."""
    log.info("-" * 60)
    log.info("Writing output …")

    output_file: Optional[TextIO] = None
    collected: List = []
    try:
        if config.output_path:
            output_file = open(config.output_path, "w", newline="")
            writer = csv.writer(output_file)
        else:
            writer = csv.writer(sys.stdout)

        writer.writerow([
            "case_id",
            "activity",
            "timestamp",
            "event_type",
            "arrival_timestamp",
        ])

        for event in stream:
            writer.writerow([
                event.case_id,
                event.activity,
                f"{event.timestamp:.6f}",
                event.event_type,
                f"{event.arrival_timestamp:.6f}",
            ])
            collected.append(event)

    finally:
        if output_file is not None:
            output_file.close()

    log.info("Wrote %d events.", len(collected))
    return collected


def _log_config(log: logging.Logger, config: RunConfig) -> None:
    """Log the full run configuration.

    Args:
        log: Logger instance.
        config: The run configuration to log.
    """
    log.info("=" * 60)
    log.info("Stream of Intent — Intentional Event Stream Generator")
    log.info("=" * 60)

    spec = config.effective_spec
    log.info("Optimisation specification:")
    for ft in FEATURE_ORDER:
        op = spec.get(ft)
        if isinstance(op, Target):
            log.info("  %-24s = Target(%.3f)", ft.value, op.value)
        elif isinstance(op, Symbol):
            log.info("  %-24s = Symbol (free)", ft.value)
        elif isinstance(op, Condition):
            log.info("  %-24s = Condition(%s)", ft.value, op.expression)
        elif isinstance(op, Exclude):
            log.info("  %-24s = Exclude", ft.value)

    log.info("Static configuration:")
    for key, value in asdict(config.static).items():
        log.info("  %-24s = %s", key, value)

    log.info("Optimisation:")
    log.info("  max_iterations         = %d", config.max_iterations)
    log.info("  output_path            = %s", config.output_path or "<stdout>")


def _log_achieved(
    log: logging.Logger,
    events: List,
    config: RunConfig,
) -> None:
    """Compute and log achieved features vs. optimisation spec.

    Args:
        log: Logger instance.
        events: All generated events.
        config: The run configuration (for spec and window size).
    """
    if not events or len(events) < config.static.window_size:
        return

    log.info("-" * 60)
    log.info("Achieved features (per-window mean):")

    window_features = []
    for window in tumbling_windows(iter(events), config.static.window_size):
        window_features.append(extract_features(window))

    if not window_features:
        return

    mean_arr = np.mean([fv.to_array() for fv in window_features], axis=0)
    achieved = FeatureVector.from_array(mean_arr)
    spec = config.effective_spec

    for ft in FEATURE_ORDER:
        op = spec.get(ft)
        a = achieved.get(ft)

        if isinstance(op, Target):
            log.info(
                "  %-24s = %.3f  (target: %.3f, Δ %+.3f)",
                ft.value, a, op.value, a - op.value,
            )
        elif isinstance(op, Symbol):
            log.info("  %-24s = %.3f  (symbol — free)", ft.value, a)
        elif isinstance(op, Condition):
            ok = "✓" if op.expression.satisfied(a) else "✗"
            log.info(
                "  %-24s = %.3f  (condition: %s %s)",
                ft.value, a, op.expression, ok,
            )
        elif isinstance(op, Exclude):
            log.info("  %-24s = %.3f  (excluded)", ft.value, a)

    log.info("  %-24s = %.4f", "objective_distance", spec.distance(achieved))


def _model_path_for(output_path: Optional[str]) -> str:
    """Derive the model-representation file path from the CSV output path."""
    if output_path:
        base, _ = os.path.splitext(output_path)
        return base + ".model.json"
    return "stream.model.json"


def stream(config: RunConfig) -> None:
    """Stream events from an (existing or freshly optimised) model.

    If a model file already exists for the configured output path it is
    loaded directly; otherwise the optimisation pipeline is run first.
    Events are emitted through the adapter specified in
    ``config.stream``.

    Args:
        config: Complete run configuration with a ``stream`` section.
    """
    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("stream_of_intent")

    stream_cfg = config.stream or StreamConfig()

    model_path = _model_path_for(config.output_path)
    if os.path.isfile(model_path):
        log.info("Loading model representation from %s", model_path)
        processor = StreamProcessor.from_file(model_path, config=stream_cfg)
    else:
        log.info("No model file found — running optimisation first …")
        _, model_rep = generate_intentional_stream(
            spec=config.effective_spec,
            static_params=config.static,
            max_iterations=config.max_iterations,
        )
        model_rep.save(model_path)
        log.info("Saved model representation to %s", model_path)
        processor = StreamProcessor(model=model_rep, config=stream_cfg)

    adapter = create_adapter(stream_cfg)
    log.info("Streaming via %s adapter …", stream_cfg.adapter)
    adapter.stream(processor.run())


if __name__ == "__main__":
    run(RunConfig(
        spec=OptimizationSpec([
            Target(FeatureType.TEMPORAL_DEPENDENCY, 0.7),
            Symbol(FeatureType.LONG_TERM_DEPENDENCY),
            Symbol(FeatureType.NON_LINEAR_DEPENDENCY),
            Symbol(FeatureType.OUT_OF_ORDER),
            Symbol(FeatureType.FRACTAL_BEHAVIOR),
        ]),
        static=StaticParams(
            window_size=1000,
            num_activities=10,
            stream_length=5000,
            num_windows_for_evaluation=5,
            random_seed=42,
        ),
        max_iterations=50,
        output_path="stream.csv",
        verbose=True,
    ))
