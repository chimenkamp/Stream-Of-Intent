from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stream_of_intent.types import (
    ModelParams,
    OptimizableParams,
    PostprocessorParams,
    SimulationParams,
    StaticParams,
)

_KEY_SEPARATOR = "|"


@dataclass
class AbstractModelRepresentation:
    """Serialisable snapshot of an optimised process model.

    Stores everything needed to replay the event stream without
    re-running process-tree generation or Bayesian optimisation.

    Attributes:
        optimized_params: Best-found optimisable parameters.
        static_params: Non-optimised fixed configuration.
        activities: Sorted list of activity labels.
        transition_matrices: Variable-order Markov matrices keyed by
            order.  Each value maps a ``"|"``-joined context string to a
            dict of ``{activity: probability}``.
        subprocess_activities: Activity list for the subprocess model
            (if any).
        subprocess_transition_matrices: Transition matrices for the
            subprocess model (if any).
        metadata: Free-form metadata (timestamps, distances, …).
    """

    optimized_params: OptimizableParams = field(default_factory=OptimizableParams)
    static_params: StaticParams = field(default_factory=StaticParams)
    activities: List[str] = field(default_factory=list)
    transition_matrices: Dict[int, Dict[str, Dict[str, float]]] = field(
        default_factory=dict,
    )
    subprocess_activities: Optional[List[str]] = None
    subprocess_transition_matrices: Optional[
        Dict[int, Dict[str, Dict[str, float]]]
    ] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the representation to a JSON-friendly dictionary."""
        return {
            "optimized_params": asdict(self.optimized_params),
            "static_params": asdict(self.static_params),
            "activities": self.activities,
            "transition_matrices": {
                str(order): matrix
                for order, matrix in self.transition_matrices.items()
            },
            "subprocess_activities": self.subprocess_activities,
            "subprocess_transition_matrices": (
                {
                    str(order): matrix
                    for order, matrix in self.subprocess_transition_matrices.items()
                }
                if self.subprocess_transition_matrices is not None
                else None
            ),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str) -> None:
        """Write the JSON representation to *path*."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AbstractModelRepresentation:
        """Reconstruct from a plain dictionary."""
        op = data["optimized_params"]
        optimized_params = OptimizableParams(
            model=ModelParams(**op["model"]),
            simulation=SimulationParams(**op["simulation"]),
            postprocessor=PostprocessorParams(**op["postprocessor"]),
        )
        static_params = StaticParams(**data["static_params"])

        transition_matrices: Dict[int, Dict[str, Dict[str, float]]] = {
            int(order): matrix
            for order, matrix in data["transition_matrices"].items()
        }

        sub_tm: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None
        if data.get("subprocess_transition_matrices") is not None:
            sub_tm = {
                int(order): matrix
                for order, matrix in data["subprocess_transition_matrices"].items()
            }

        return cls(
            optimized_params=optimized_params,
            static_params=static_params,
            activities=data["activities"],
            transition_matrices=transition_matrices,
            subprocess_activities=data.get("subprocess_activities"),
            subprocess_transition_matrices=sub_tm,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> AbstractModelRepresentation:
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: str) -> AbstractModelRepresentation:
        """Read an :class:`AbstractModelRepresentation` from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())


def _encode_context(context: Tuple[str, ...]) -> str:
    """Encode a tuple context key to a ``"|"``-separated string."""
    return _KEY_SEPARATOR.join(context)


def _decode_context(key: str) -> Tuple[str, ...]:
    """Decode a ``"|"``-separated string back to a tuple."""
    return tuple(key.split(_KEY_SEPARATOR))


def encode_transition_matrices(
    matrices: Dict[int, Dict[Tuple[str, ...], Dict[str, float]]],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Convert tuple-keyed transition matrices to string-keyed form for JSON."""
    encoded: Dict[int, Dict[str, Dict[str, float]]] = {}
    for order, matrix in matrices.items():
        encoded[order] = {
            _encode_context(ctx): dist for ctx, dist in matrix.items()
        }
    return encoded


class ReplayableProcessModel:
    """Lightweight stand-in for :class:`~stream_of_intent.process_model.ProcessModel`.

    Implements the interface that :class:`~stream_of_intent.simulation.SimulationEngine`
    relies on — ``activities``, ``get_transition_matrix(order)``, and
    ``get_subprocess_model(rng)`` — without requiring pm4py.
    """

    def __init__(
        self,
        activities: List[str],
        transition_matrices: Dict[int, Dict[str, Dict[str, float]]],
        subprocess_activities: Optional[List[str]] = None,
        subprocess_transition_matrices: Optional[
            Dict[int, Dict[str, Dict[str, float]]]
        ] = None,
    ) -> None:
        self.activities = activities
        self._str_matrices = transition_matrices
        self._tuple_matrices: Dict[
            int, Dict[Tuple[str, ...], Dict[str, float]]
        ] = {}
        self._subprocess_activities = subprocess_activities
        self._subprocess_str_matrices = subprocess_transition_matrices

    def get_transition_matrix(
        self, order: int,
    ) -> Dict[Tuple[str, ...], Dict[str, float]]:
        """Return a tuple-keyed transition matrix for the given Markov *order*."""
        if order in self._tuple_matrices:
            return self._tuple_matrices[order]

        str_matrix = self._str_matrices.get(order, {})
        decoded: Dict[Tuple[str, ...], Dict[str, float]] = {
            _decode_context(k): v for k, v in str_matrix.items()
        }
        self._tuple_matrices[order] = decoded
        return decoded

    def get_subprocess_model(
        self,
        rng: np.random.RandomState,
        simplify: bool = True,
    ) -> ReplayableProcessModel:
        """Return a subprocess model for fractal behaviour."""
        if self._subprocess_activities and self._subprocess_str_matrices:
            return ReplayableProcessModel(
                activities=self._subprocess_activities,
                transition_matrices=self._subprocess_str_matrices,
            )
        
        subset_size = max(2, len(self.activities) // 2)
        subset = list(rng.choice(self.activities, size=subset_size, replace=False))
        return ReplayableProcessModel(
            activities=subset,
            transition_matrices=self._str_matrices,
        )

    @classmethod
    def from_representation(
        cls, rep: AbstractModelRepresentation,
    ) -> ReplayableProcessModel:
        """Construct directly from an :class:`AbstractModelRepresentation`."""
        return cls(
            activities=rep.activities,
            transition_matrices=rep.transition_matrices,
            subprocess_activities=rep.subprocess_activities,
            subprocess_transition_matrices=rep.subprocess_transition_matrices,
        )
