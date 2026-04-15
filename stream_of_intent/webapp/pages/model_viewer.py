from __future__ import annotations

import json

import plotly.graph_objects as go
from dash import dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from stream_of_intent.types import FEATURE_ORDER
from stream_of_intent.webapp.components.feature_radar import build_radar
from stream_of_intent.webapp.components.model_graph import create_model_graph
from stream_of_intent.webapp.components.theme import NORD, plotly_layout
from stream_of_intent.webapp.db import get_model, list_models, load_representation
from stream_of_intent.webapp.graph_builder import (
    build_elements,
    build_heatmap_data,
    get_available_orders,
)


def layout():
    return html.Div(
        [
            html.Div(
                [html.Span("◈", className="title-icon"), "Model Viewer"],
                className="page-title",
            ),
            # ---- Two-panel layout: settings left, content right ----
            html.Div(
                [
                    # ---- Left panel: settings ----
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Model", className="nord-card-header"),
                                    html.Label("Select Model", className="nord-label"),
                                    dcc.Dropdown(
                                        id="viewer-model-select",
                                        options=[],
                                        placeholder="Choose a model…",
                                        className="mb-3",
                                    ),
                                    html.Label("Markov Order", className="nord-label"),
                                    dcc.Dropdown(
                                        id="viewer-order-select",
                                        options=[{"label": f"Order {i}", "value": i} for i in range(1, 5)],
                                        value=1,
                                        clearable=False,
                                        className="mb-3",
                                    ),
                                    html.Label("Graph Layout", className="nord-label"),
                                    dcc.Dropdown(
                                        id="viewer-layout-select",
                                        options=[
                                            {"label": "Force-Directed (CoSE)", "value": "cose"},
                                            {"label": "Hierarchical (Dagre)", "value": "dagre"},
                                            {"label": "Circle", "value": "circle"},
                                            {"label": "Grid", "value": "grid"},
                                            {"label": "Concentric", "value": "concentric"},
                                        ],
                                        value="cose",
                                        clearable=False,
                                        className="mb-3",
                                    ),
                                    html.Label("Edge Probability Threshold", className="nord-label"),
                                    dcc.Slider(
                                        id="viewer-edge-threshold",
                                        min=0, max=0.5, step=0.01, value=0.0,
                                        marks={0: "0", 0.1: ".1", 0.2: ".2", 0.5: ".5"},
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                className="nord-card",
                            ),
                            # Feature radar
                            html.Div(
                                [
                                    html.Div("Feature Profile", className="nord-card-header"),
                                    dcc.Graph(id="viewer-radar", figure=build_radar(),
                                              config={"displayModeBar": False}),
                                ],
                                className="nord-card",
                            ),
                            # Metadata
                            html.Div(
                                [
                                    html.Div("Metadata", className="nord-card-header"),
                                    html.Div(id="viewer-metadata"),
                                ],
                                className="nord-card",
                            ),
                        ],
                        className="panel-left",
                    ),
                    # ---- Right panel: content ----
                    html.Div(
                        [
                            # Graph
                            html.Div(
                                [
                                    html.Div("Process Model Graph", className="nord-card-header"),
                                    html.Div(id="viewer-graph-container", children=[
                                        create_model_graph("viewer-cytoscape", height="520px"),
                                    ]),
                                    html.Div(id="viewer-node-detail", className="mt-2"),
                                ],
                                className="nord-card",
                            ),
                            # Heatmap
                            html.Div(
                                [
                                    html.Div("Transition Matrix", className="nord-card-header"),
                                    dcc.Graph(id="viewer-heatmap", config={"displayModeBar": False}),
                                ],
                                className="nord-card",
                            ),
                            # Parameters
                            html.Div(
                                [
                                    html.Div("Model Parameters", className="nord-card-header"),
                                    html.Div(id="viewer-params-detail"),
                                ],
                                className="nord-card",
                            ),
                        ],
                        className="panel-right",
                    ),
                ],
                className="panel-layout",
            ),
        ],
    )


def register_callbacks(app):
    """Register callbacks for the model viewer page."""

    # Populate model dropdown
    @app.callback(
        Output("viewer-model-select", "options"),
        Input("viewer-model-select", "id"),  # fires on mount
    )
    def populate_models(_):
        models = list_models()
        return [{"label": f"{m['name']} (dist: {m.get('best_distance', '?'):.4f})" if m.get("best_distance") else m["name"],
                 "value": m["id"]} for m in models]

    # Main model load callback
    @app.callback(
        Output("viewer-cytoscape", "elements"),
        Output("viewer-cytoscape", "layout"),
        Output("viewer-heatmap", "figure"),
        Output("viewer-radar", "figure"),
        Output("viewer-params-detail", "children"),
        Output("viewer-metadata", "children"),
        Output("viewer-order-select", "options"),
        Input("viewer-model-select", "value"),
        Input("viewer-order-select", "value"),
        Input("viewer-layout-select", "value"),
        Input("viewer-edge-threshold", "value"),
    )
    def load_model(model_id, order, layout_name, threshold):
        if model_id is None:
            empty_fig = go.Figure()
            empty_fig.update_layout(**plotly_layout(height=300,
                annotations=[dict(text="Select a model", xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False, font=dict(color=NORD["nord3"], size=14))]))
            return [], _layout_dict("cose"), empty_fig, build_radar(), html.Div("No model selected.", className="text-muted-nord"), html.Div(), [{"label": "Order 1", "value": 1}]

        rep = load_representation(model_id)
        if rep is None:
            return [], _layout_dict("cose"), go.Figure(), build_radar(), html.Div("Model not found."), html.Div(), []

        # Graph elements
        elements = build_elements(rep, order=order or 1, edge_threshold=threshold or 0.0)

        # Available orders
        orders = get_available_orders(rep)
        order_opts = [{"label": f"Order {o}", "value": o} for o in orders]

        # Heatmap
        hm_data = build_heatmap_data(rep, order=order or 1)
        if hm_data:
            hm_fig = go.Figure(go.Heatmap(
                z=hm_data["z"], x=hm_data["x"], y=hm_data["y"],
                colorscale=[
                    [0, NORD["nord0"]], [0.5, NORD["nord10"]], [1, NORD["nord8"]],
                ],
                hoverongaps=False,
            ))
            hm_fig.update_layout(**plotly_layout(
                title=dict(text=f"Transition Probabilities (Order {order or 1})", font=dict(size=13, color=NORD["nord5"])),
                height=350, xaxis=dict(title="To", tickangle=-45),
                yaxis=dict(title="From", autorange="reversed"),
            ))
        else:
            hm_fig = go.Figure()
            hm_fig.update_layout(**plotly_layout(height=350))

        # Radar
        meta = rep.metadata or {}
        achieved = meta.get("achieved_features")
        spec_meta = meta.get("spec", {})
        targets = {}
        for ft in FEATURE_ORDER:
            entry = spec_meta.get(ft.value, {})
            if entry.get("type") == "target":
                targets[ft.value] = entry["value"]
        radar_fig = build_radar(achieved=achieved, target=targets if targets else None, title="Feature Profile")

        # Params
        params_div = _render_params(rep)

        # Metadata
        meta_div = _render_metadata(meta)

        return elements, _layout_dict(layout_name or "cose"), hm_fig, radar_fig, params_div, meta_div, order_opts

    # Node click detail
    @app.callback(
        Output("viewer-node-detail", "children"),
        Input("viewer-cytoscape", "tapNodeData"),
        State("viewer-model-select", "value"),
        State("viewer-order-select", "value"),
    )
    def node_click(node_data, model_id, order):
        if not node_data or not model_id:
            return html.Div()

        rep = load_representation(model_id)
        if rep is None:
            return html.Div()

        node_id = node_data.get("id", "")
        label = node_data.get("label", node_id)

        matrices = rep.transition_matrices or {}
        matrix = matrices.get(order or 1, matrices.get(str(order or 1), {}))

        # Find outgoing transitions
        outgoing = {}
        for ctx_key, transitions in matrix.items():
            parts = ctx_key.split("|") if ctx_key else []
            if parts and parts[-1] == node_id:
                outgoing.update(transitions)

        if not outgoing:
            return html.Div(
                f"Node: {label} — no outgoing transitions in matrix.",
                style={"color": NORD["nord3"], "fontSize": "0.82rem"},
            )

        sorted_out = sorted(outgoing.items(), key=lambda x: x[1], reverse=True)
        rows = [
            html.Div(
                [
                    html.Span(f"→ {act}", style={"color": NORD["nord4"], "fontSize": "0.8rem"}),
                    html.Span(f"{prob:.3f}", style={"color": NORD["nord8"], "fontSize": "0.8rem",
                                                     "fontWeight": "600", "fontVariantNumeric": "tabular-nums"}),
                ],
                style={"display": "flex", "justifyContent": "space-between", "padding": "0.15rem 0"},
            )
            for act, prob in sorted_out
        ]
        return html.Div(
            [
                html.Div(f"Outgoing from: {label}", style={"color": NORD["nord8"], "fontWeight": "600",
                                                             "fontSize": "0.82rem", "marginBottom": "0.3rem"}),
                *rows,
            ],
            style={"backgroundColor": NORD["nord0"], "padding": "0.7rem",
                   "borderRadius": "4px", "border": f"1px solid {NORD['nord2']}"},
        )


def _layout_dict(name: str) -> dict:
    return {
        "name": name,
        "animate": True,
        "animationDuration": 300,
        "nodeRepulsion": 6000,
        "idealEdgeLength": 120,
        "edgeElasticity": 50,
        "gravity": 0.3,
        "padding": 30,
    }


def _render_params(rep) -> html.Div:
    params = rep.optimized_params
    sections = []

    for label, obj in [("Model", params.model), ("Simulation", params.simulation), ("Postprocessor", params.postprocessor)]:
        items = []
        for k, v in sorted(vars(obj).items()):
            val = f"{v:.4f}" if isinstance(v, float) else str(v)
            items.append(
                html.Div(
                    [
                        html.Span(k, style={"color": NORD["nord4"], "fontSize": "0.78rem"}),
                        html.Span(val, style={"color": NORD["nord8"], "fontSize": "0.78rem",
                                              "fontWeight": "600", "fontVariantNumeric": "tabular-nums"}),
                    ],
                    style={"display": "flex", "justifyContent": "space-between",
                           "padding": "0.15rem 0", "borderBottom": f"1px solid {NORD['nord2']}"},
                )
            )
        sections.append(
            html.Div(
                [
                    html.Div(label, style={"color": NORD["nord9"], "fontWeight": "600",
                                           "fontSize": "0.82rem", "marginBottom": "0.3rem"}),
                    *items,
                ],
                className="mb-3",
            )
        )

    return html.Div(sections, style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "1rem"})


def _render_metadata(meta: dict) -> html.Div:
    if not meta:
        return html.Div("No metadata available.", className="text-muted-nord")

    items = []
    for k, v in sorted(meta.items()):
        if k in ("spec", "achieved_features"):
            continue
        if k == "generation_timestamp":
            import datetime
            v = datetime.datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M:%S")
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        items.append(
            html.Div(
                [
                    html.Span(k, style={"color": NORD["nord4"], "fontSize": "0.8rem"}),
                    html.Span(val, style={"color": NORD["nord8"], "fontSize": "0.8rem", "fontWeight": "600"}),
                ],
                style={"display": "flex", "justifyContent": "space-between",
                       "padding": "0.2rem 0", "borderBottom": f"1px solid {NORD['nord2']}"},
            )
        )

    # Spec summary
    spec_data = meta.get("spec", {})
    if spec_data:
        spec_rows = []
        for ft_name, entry in spec_data.items():
            desc = entry.get("type", "?")
            if desc == "target":
                desc = f"Target → {entry.get('value', '?')}"
            elif desc == "condition":
                desc = f"Condition: {entry.get('expression', '?')}"
            spec_rows.append(
                html.Div(
                    [
                        html.Span(ft_name.replace("_", " ").title(),
                                  style={"color": NORD["nord4"], "fontSize": "0.8rem"}),
                        html.Span(desc, style={"color": NORD["nord13"], "fontSize": "0.8rem"}),
                    ],
                    style={"display": "flex", "justifyContent": "space-between", "padding": "0.15rem 0"},
                )
            )
        items.append(html.Hr(style={"borderColor": NORD["nord2"], "margin": "0.5rem 0"}))
        items.append(html.Div("Optimization Spec", style={"color": NORD["nord9"], "fontWeight": "600",
                                                            "fontSize": "0.82rem", "marginBottom": "0.3rem"}))
        items.extend(spec_rows)

    return html.Div(items)
