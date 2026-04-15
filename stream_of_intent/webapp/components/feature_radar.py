from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go

from stream_of_intent.types import FEATURE_ORDER, FeatureType
from stream_of_intent.webapp.components.theme import NORD, FEATURE_COLORS, MODEL_COLORS, plotly_layout

_LABELS = [
    "Temporal\nDep.",
    "Long-Term\nDep.",
    "Non-Linear\nDep.",
    "Out of\nOrder",
    "Fractal\nBehavior",
]


def build_radar(
    achieved: Optional[Dict[str, float]] = None,
    target: Optional[Dict[str, float]] = None,
    title: str = "Feature Profile",
) -> go.Figure:
    """Build a single-model radar chart comparing achieved vs target."""
    fig = go.Figure()
    cats = _LABELS + [_LABELS[0]]  # Close the polygon.

    if target:
        vals = [target.get(ft.value, 0) for ft in FEATURE_ORDER]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name="Target",
            line=dict(color=NORD["nord13"], width=2, dash="dash"),
            fill="none", opacity=0.8,
        ))

    if achieved:
        vals = [achieved.get(ft.value, 0) for ft in FEATURE_ORDER]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name="Achieved",
            line=dict(color=NORD["nord8"], width=2.5),
            fill="toself", fillcolor="rgba(136,192,208,0.12)",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=NORD["nord0"],
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=NORD["nord2"], linecolor=NORD["nord2"],
                tickfont=dict(color=NORD["nord3"], size=10),
            ),
            angularaxis=dict(
                gridcolor=NORD["nord2"], linecolor=NORD["nord2"],
                tickfont=dict(color=NORD["nord4"], size=10),
            ),
        ),
        paper_bgcolor=NORD["nord1"],
        font=dict(color=NORD["nord4"], family="Inter, system-ui, sans-serif"),
        title=dict(text=title, font=dict(size=13, color=NORD["nord5"])),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=NORD["nord4"], size=11)),
        height=320,
        margin=dict(l=60, r=60, t=50, b=40),
    )
    return fig


def build_comparison_radar(
    models: List[Dict],
    title: str = "Feature Comparison",
) -> go.Figure:
    """Build an overlay radar chart for multiple models.

    Args:
        models: List of dicts with keys ``name`` and ``features``
            (dict mapping feature name → value).
    """
    fig = go.Figure()
    cats = _LABELS + [_LABELS[0]]

    for i, model in enumerate(models):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        feats = model.get("features", {})
        vals = [feats.get(ft.value, 0) for ft in FEATURE_ORDER]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=model.get("name", f"Model {i+1}"),
            line=dict(color=color, width=2),
            fill="toself", fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=NORD["nord0"],
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=NORD["nord2"], linecolor=NORD["nord2"],
                tickfont=dict(color=NORD["nord3"], size=10),
            ),
            angularaxis=dict(
                gridcolor=NORD["nord2"], linecolor=NORD["nord2"],
                tickfont=dict(color=NORD["nord4"], size=10),
            ),
        ),
        paper_bgcolor=NORD["nord1"],
        font=dict(color=NORD["nord4"], family="Inter, system-ui, sans-serif"),
        title=dict(text=title, font=dict(size=13, color=NORD["nord5"])),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=NORD["nord4"], size=11)),
        height=380,
        margin=dict(l=60, r=60, t=50, b=40),
    )
    return fig


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R,G,B' string."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
