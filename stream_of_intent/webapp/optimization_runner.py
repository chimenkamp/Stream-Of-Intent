from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import optuna

from stream_of_intent.features import extract_features, tumbling_windows
from stream_of_intent.model_representation import AbstractModelRepresentation
from stream_of_intent.optimization import generate_intentional_stream
from stream_of_intent.types import (
    FEATURE_ORDER,
    FeatureVector,
    OptimizationSpec,
    StaticParams,
)


@dataclass
class TrialResult:
    trial_number: int
    distance: float
    best_distance: float
    params: Dict[str, Any]
    timestamp: float


@dataclass
class OptimizationRun:
    """Manages a background optimization with progress tracking."""

    spec: OptimizationSpec
    static_params: StaticParams
    max_iterations: int

    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _progress: List[TrialResult] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _result: Optional[AbstractModelRepresentation] = field(default=None, init=False, repr=False)
    _error: Optional[str] = field(default=None, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    _achieved_features: Optional[FeatureVector] = field(default=None, init=False, repr=False)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def progress(self) -> List[TrialResult]:
        with self._lock:
            return list(self._progress)

    @property
    def result(self) -> Optional[AbstractModelRepresentation]:
        return self._result

    @property
    def achieved_features(self) -> Optional[FeatureVector]:
        return self._achieved_features

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def elapsed(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._cancelled = False
        self._error = None
        self._result = None
        self._achieved_features = None
        self._start_time = time.time()
        with self._lock:
            self._progress.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancelled = True

    def _run(self) -> None:
        try:
            best_so_far = float("inf")

            def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                nonlocal best_so_far
                if self._cancelled:
                    study.stop()
                    return
                dist = trial.value if trial.value is not None else float("inf")
                if dist < best_so_far:
                    best_so_far = dist
                result = TrialResult(
                    trial_number=trial.number,
                    distance=dist,
                    best_distance=best_so_far,
                    params=dict(trial.params),
                    timestamp=time.time(),
                )
                with self._lock:
                    self._progress.append(result)

            events_iter, model_rep = generate_intentional_stream(
                spec=self.spec,
                static_params=self.static_params,
                max_iterations=self.max_iterations,
                study_callbacks=[_callback],
            )

            # Consume a window to compute achieved features for metadata.
            events = []
            for ev in events_iter:
                events.append(ev)
                if len(events) >= self.static_params.window_size:
                    break

            if events:
                fv = extract_features(events)
                self._achieved_features = fv
                model_rep.metadata["achieved_features"] = {
                    ft.value: fv.get(ft) for ft in FEATURE_ORDER
                }

            self._result = model_rep

        except Exception as exc:
            self._error = str(exc)
        finally:
            self._running = False


# Singleton for the current run (one optimization at a time per server).
_current_run: Optional[OptimizationRun] = None


def get_current_run() -> Optional[OptimizationRun]:
    return _current_run


def start_run(
    spec: OptimizationSpec,
    static_params: StaticParams,
    max_iterations: int,
) -> OptimizationRun:
    global _current_run
    if _current_run is not None and _current_run.is_running:
        _current_run.cancel()
    _current_run = OptimizationRun(
        spec=spec,
        static_params=static_params,
        max_iterations=max_iterations,
    )
    _current_run.start()
    return _current_run
