from __future__ import annotations

from typing import Any, Dict, List, Optional

from stream_of_intent.model_representation import AbstractModelRepresentation

_START = "__START__"
_END = "__END__"


def build_elements(
    rep: AbstractModelRepresentation,
    order: int = 1,
    edge_threshold: float = 0.0,
    include_subprocess: bool = True,
) -> List[Dict[str, Any]]:
    """Convert a model representation to Cytoscape elements.

    Returns a list of node and edge dicts compatible with
    ``dash_cytoscape.Cytoscape(elements=...)``.
    """
    elements: List[Dict[str, Any]] = []
    activities = rep.activities or []

    elements.append({
        "data": {"id": _START, "label": "START", "node_type": "control"},
        "classes": "control-node start-node",
    })
    elements.append({
        "data": {"id": _END, "label": "END", "node_type": "control"},
        "classes": "control-node end-node",
    })
    for act in activities:
        elements.append({
            "data": {"id": act, "label": act, "node_type": "activity"},
            "classes": "activity-node",
        })

    matrices = rep.transition_matrices or {}
    matrix = matrices.get(order, matrices.get(str(order), {}))
    if isinstance(matrix, dict):
        _add_edges(elements, matrix, edge_threshold, "main")

    if include_subprocess and rep.subprocess_activities:
        for act in rep.subprocess_activities:
            node_id = f"sub_{act}"
            elements.append({
                "data": {"id": node_id, "label": act, "node_type": "subprocess"},
                "classes": "subprocess-node",
            })
        sub_matrices = rep.subprocess_transition_matrices or {}
        sub_matrix = sub_matrices.get(order, sub_matrices.get(str(order), {}))
        if isinstance(sub_matrix, dict):
            _add_edges(elements, sub_matrix, edge_threshold, "sub", prefix="sub_")

    return elements


def _add_edges(
    elements: List[Dict[str, Any]],
    matrix: Dict[str, Dict[str, float]],
    threshold: float,
    group: str,
    prefix: str = "",
) -> None:
    """Add directed weighted edges from a transition matrix."""
    edge_id = 0
    for context_key, transitions in matrix.items():
        if not isinstance(transitions, dict):
            continue

        parts = context_key.split("|") if context_key else []
        source = parts[-1] if parts else None
        if source is None:
            continue

        if source in ("__START__", "<START>"):
            source = _START
        elif source in ("__END__", "<END>"):
            source = _END
        else:
            source = f"{prefix}{source}" if prefix else source

        for target_act, prob in transitions.items():
            if prob < threshold:
                continue

            target = target_act
            if target in ("__START__", "<START>"):
                target = _START
            elif target in ("__END__", "<END>"):
                target = _END
            else:
                target = f"{prefix}{target}" if prefix else target

            elements.append({
                "data": {
                    "id": f"e_{group}_{edge_id}",
                    "source": source,
                    "target": target,
                    "weight": round(prob, 4),
                    "label": f"{prob:.2f}",
                    "edge_group": group,
                },
                "classes": f"{group}-edge",
            })
            edge_id += 1


def get_available_orders(rep: AbstractModelRepresentation) -> List[int]:
    """Return sorted list of Markov orders present in the model."""
    if not rep.transition_matrices:
        return [1]
    return sorted(int(k) for k in rep.transition_matrices)


def build_heatmap_data(
    rep: AbstractModelRepresentation,
    order: int = 1,
) -> Optional[Dict[str, Any]]:
    """Build a transition probability matrix for Plotly heatmap.

    Returns dict with keys ``z``, ``x``, ``y`` (activity labels).
    """
    matrices = rep.transition_matrices or {}
    matrix = matrices.get(order, matrices.get(str(order)))
    if matrix is None:
        return None

    activities = sorted(rep.activities) if rep.activities else []
    if not activities:
        return None

    n = len(activities)
    idx = {a: i for i, a in enumerate(activities)}
    z = [[0.0] * n for _ in range(n)]

    for context_key, transitions in matrix.items():
        if not isinstance(transitions, dict):
            continue
        parts = context_key.split("|") if context_key else []
        source = parts[-1] if parts else None
        if source not in idx:
            continue
        for target_act, prob in transitions.items():
            if target_act in idx:
                z[idx[source]][idx[target_act]] = round(prob, 4)

    return {"z": z, "x": activities, "y": activities}
