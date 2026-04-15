from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

from stream_of_intent.webapp.components.theme import NORD, plotly_layout


def create_loss_plot() -> html.Div:
    """Create the loss plot container with an empty figure."""
    fig = _empty_figure()
    return html.Div(
        [
            dcc.Graph(id="loss-plot", figure=fig, config={"displayModeBar": False}),
        ],
    )


def build_loss_figure(progress: list) -> go.Figure:
    """Build a Plotly figure from optimisation progress data.

    Args:
        progress: List of TrialResult-like dicts with keys
            trial_number, distance, best_distance.
    """
    if not progress:
        return _empty_figure()

    trials = [p.trial_number if hasattr(p, "trial_number") else p["trial_number"] for p in progress]
    distances = [p.distance if hasattr(p, "distance") else p["distance"] for p in progress]
    best = [p.best_distance if hasattr(p, "best_distance") else p["best_distance"] for p in progress]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trials, y=distances,
        mode="markers",
        name="Trial Distance",
        marker=dict(color=NORD["nord3"], size=5, opacity=0.6),
    ))

    fig.add_trace(go.Scatter(
        x=trials, y=best,
        mode="lines",
        name="Best So Far",
        line=dict(color=NORD["nord8"], width=2.5),
    ))

    fig.update_layout(
        **plotly_layout(
            title=dict(text="Optimization Progress", font=dict(size=14, color=NORD["nord5"])),
            xaxis=dict(title="Trial", gridcolor=NORD["nord2"], zerolinecolor=NORD["nord2"]),
            yaxis=dict(title="Distance (Loss)", gridcolor=NORD["nord2"], zerolinecolor=NORD["nord2"]),
            height=280,
            margin=dict(l=50, r=20, t=40, b=40),
        )
    )
    return fig


def _empty_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **plotly_layout(
            title=dict(text="Optimization Progress", font=dict(size=14, color=NORD["nord5"])),
            xaxis=dict(title="Trial", gridcolor=NORD["nord2"]),
            yaxis=dict(title="Distance (Loss)", gridcolor=NORD["nord2"]),
            height=280,
            margin=dict(l=50, r=20, t=40, b=40),
            annotations=[dict(
                text="Start training to see progress",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=NORD["nord3"], size=14),
            )],
        )
    )
    return fig
