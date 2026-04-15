from __future__ import annotations

import itertools
import json
from dataclasses import asdict

import plotly.graph_objects as go
from dash import dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from stream_of_intent.config import StreamConfig
from stream_of_intent.features import extract_features
from stream_of_intent.stream_processor import StreamProcessor
from stream_of_intent.types import FEATURE_ORDER
from stream_of_intent.webapp.components.feature_radar import build_comparison_radar
from stream_of_intent.webapp.components.model_graph import create_model_graph
from stream_of_intent.webapp.components.theme import NORD, MODEL_COLORS, plotly_layout
from stream_of_intent.webapp.db import list_models, load_representation
from stream_of_intent.webapp.graph_builder import build_elements


def layout():
    return html.Div(
        [
            html.Div(
                [html.Span("⊞", className="title-icon"), "Compare Models"],
                className="page-title",
            ),
            html.Div(
                [
                    html.Label("Select Models to Compare (2+)", className="nord-label"),
                    dcc.Dropdown(
                        id="compare-model-select",
                        options=[],
                        multi=True,
                        placeholder="Choose models…",
                    ),
                    dbc.Button("Compare", id="compare-btn", className="btn-nord-primary mt-2", size="sm"),
                ],
                className="nord-card",
            ),
            html.Div(id="compare-results"),
        ],
    )


def register_callbacks(app):
    """Register callbacks for the compare page."""

    @app.callback(
        Output("compare-model-select", "options"),
        Input("compare-model-select", "id"),
    )
    def populate_models(_):
        models = list_models()
        return [{"label": m["name"], "value": m["id"]} for m in models]

    @app.callback(
        Output("compare-results", "children"),
        Input("compare-btn", "n_clicks"),
        State("compare-model-select", "value"),
        prevent_initial_call=True,
    )
    def run_comparison(n, model_ids):
        if not n or not model_ids or len(model_ids) < 2:
            return dbc.Alert("Select at least 2 models.",
                             style={"backgroundColor": NORD["nord13"], "color": NORD["nord0"],
                                    "border": "none", "fontSize": "0.82rem"})

        models_data = []
        reps = {}
        for mid in model_ids:
            rep = load_representation(mid)
            if rep is None:
                continue
            reps[mid] = rep
            meta = rep.metadata or {}
            achieved = meta.get("achieved_features", {})
            models_data.append({
                "id": mid,
                "name": meta.get("name", f"Model {mid}"),
                "features": achieved,
                "rep": rep,
                "best_distance": meta.get("best_distance", None),
            })

        from stream_of_intent.webapp.db import get_model
        for md in models_data:
            record = get_model(md["id"])
            if record:
                md["name"] = record.name

        if len(models_data) < 2:
            return html.Div("Not enough valid models found.", className="text-muted-nord")

        sections = []

        radar_fig = build_comparison_radar(models_data, title="Feature Comparison")
        sections.append(
            html.Div(
                [
                    html.Div("Feature Comparison", className="nord-card-header"),
                    dcc.Graph(figure=radar_fig, config={"displayModeBar": False}),
                ],
                className="nord-card",
            )
        )

        sections.append(
            html.Div(
                [
                    html.Div("Parameter Comparison", className="nord-card-header"),
                    _build_param_diff(models_data),
                ],
                className="nord-card",
            )
        )

        sections.append(
            html.Div(
                [
                    html.Div("Optimization Spec Diff", className="nord-card-header"),
                    _build_spec_diff(models_data),
                ],
                className="nord-card",
            )
        )

        graph_cols = []
        for i, md in enumerate(models_data):
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            elements = build_elements(md["rep"], order=1, edge_threshold=0.05)
            graph_cols.append(
                html.Div(
                    [
                        html.Div(md["name"],
                                 style={"color": color, "fontWeight": "600",
                                        "fontSize": "0.85rem", "marginBottom": "0.5rem"}),
                        create_model_graph(
                            graph_id=f"compare-graph-{md['id']}",
                            elements=elements,
                            height="380px",
                        ),
                    ],
                    style={"flex": "1", "minWidth": "0"},
                )
            )

        sections.append(
            html.Div(
                [
                    html.Div("Process Model Graphs", className="nord-card-header"),
                    html.Div(graph_cols, style={"display": "flex", "gap": "1rem"}),
                ],
                className="nord-card",
            )
        )

        stream_cols = []
        for i, md in enumerate(models_data):
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            rep = md["rep"]
            processor = StreamProcessor(rep, StreamConfig(max_loops=1))
            events = list(itertools.islice(processor.run(), 50))

            rows = []
            for ev in events[:30]:
                rows.append(
                    html.Tr([
                        html.Td(ev.case_id, style={"color": NORD["nord8"], "fontSize": "0.75rem"}),
                        html.Td(ev.activity, style={"fontSize": "0.75rem"}),
                        html.Td(f"{ev.timestamp:.1f}", style={"fontSize": "0.75rem"}),
                        html.Td(ev.event_type, style={"fontSize": "0.75rem"}),
                    ])
                )

            t = html.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Case"), html.Th("Activity"), html.Th("Time"), html.Th("Type"),
                    ], style={"backgroundColor": NORD["nord2"], "color": NORD["nord8"],
                              "fontSize": "0.7rem", "textTransform": "uppercase"})),
                    html.Tbody(rows),
                ],
                style={"width": "100%", "borderCollapse": "collapse"},
            )

            stream_cols.append(
                html.Div(
                    [
                        html.Div(md["name"],
                                 style={"color": color, "fontWeight": "600",
                                        "fontSize": "0.85rem", "marginBottom": "0.3rem"}),
                        html.Div(t, style={"maxHeight": "300px", "overflowY": "auto",
                                           "border": f"1px solid {NORD['nord2']}", "borderRadius": "4px"}),
                    ],
                    style={"flex": "1", "minWidth": "0"},
                )
            )

        sections.append(
            html.Div(
                [
                    html.Div("Stream Samples", className="nord-card-header"),
                    html.Div(stream_cols, style={"display": "flex", "gap": "1rem"}),
                ],
                className="nord-card",
            )
        )

        return html.Div(sections)


def _build_param_diff(models_data: list) -> html.Div:
    """Build a parameter comparison table highlighting differences."""
    if not models_data:
        return html.Div()

    all_params = {}
    for md in models_data:
        rep = md["rep"]
        for section_name, section_obj in [("model", rep.optimized_params.model),
                                           ("simulation", rep.optimized_params.simulation),
                                           ("postprocessor", rep.optimized_params.postprocessor)]:
            for k, v in vars(section_obj).items():
                key = f"{section_name}.{k}"
                if key not in all_params:
                    all_params[key] = []
                all_params[key].append(v)

    header = html.Tr(
        [html.Th("Parameter", style={"minWidth": "200px"})] +
        [html.Th(md["name"], style={"color": MODEL_COLORS[i % len(MODEL_COLORS)]})
         for i, md in enumerate(models_data)],
        style={"backgroundColor": NORD["nord2"], "color": NORD["nord8"],
               "fontSize": "0.78rem", "textTransform": "uppercase"},
    )

    rows = []
    for key in sorted(all_params):
        values = all_params[key]
        differs = len(set(str(v) for v in values)) > 1

        cells = [html.Td(key, style={"color": NORD["nord4"], "fontSize": "0.78rem"})]
        for v in values:
            fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
            style = {"fontSize": "0.78rem", "fontVariantNumeric": "tabular-nums"}
            if differs:
                style["color"] = NORD["nord13"]
                style["fontWeight"] = "600"
            else:
                style["color"] = NORD["nord4"]
            cells.append(html.Td(fmt, style=style))
        rows.append(html.Tr(cells))

    return html.Div(
        html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.8rem"},
        ),
        style={"maxHeight": "400px", "overflowY": "auto"},
    )


def _build_spec_diff(models_data: list) -> html.Div:
    """Build an optimization spec comparison."""
    rows = []
    for ft in FEATURE_ORDER:
        cells = [html.Td(ft.value.replace("_", " ").title(),
                         style={"color": NORD["nord4"], "fontSize": "0.8rem", "fontWeight": "600"})]
        for i, md in enumerate(models_data):
            meta = md["rep"].metadata or {}
            spec = meta.get("spec", {})
            entry = spec.get(ft.value, {"type": "exclude"})
            desc = entry.get("type", "?")
            if desc == "target":
                desc = f"Target → {entry.get('value', '?'):.2f}"
            elif desc == "condition":
                desc = f"Cond: {entry.get('expression', '?')}"
            cells.append(html.Td(desc, style={"fontSize": "0.8rem", "color": NORD["nord13"]}))
        rows.append(html.Tr(cells))

    header = html.Tr(
        [html.Th("Feature")] +
        [html.Th(md["name"], style={"color": MODEL_COLORS[i % len(MODEL_COLORS)]})
         for i, md in enumerate(models_data)],
        style={"backgroundColor": NORD["nord2"], "color": NORD["nord8"],
               "fontSize": "0.78rem", "textTransform": "uppercase"},
    )

    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )
