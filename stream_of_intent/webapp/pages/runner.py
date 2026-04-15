from __future__ import annotations

import itertools
import json
import threading
import time
from typing import Optional

import plotly.graph_objects as go
from dash import dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from stream_of_intent.config import StreamConfig
from stream_of_intent.model_representation import AbstractModelRepresentation
from stream_of_intent.stream_processor import StreamProcessor
from stream_of_intent.types import FEATURE_ORDER
from stream_of_intent.features import extract_features
from stream_of_intent.webapp.components.feature_radar import build_radar
from stream_of_intent.webapp.components.theme import NORD, plotly_layout
from stream_of_intent.webapp.db import list_models, load_representation

# Streaming state
_stream_thread: Optional[threading.Thread] = None
_stream_stop = threading.Event()
_stream_event_count = 0
_stream_lock = threading.Lock()


def layout():
    return html.Div(
        [
            html.Div(
                [html.Span("▶", className="title-icon"), "Run & Stream"],
                className="page-title",
            ),
            # Model selector
            html.Div(
                [
                    html.Label("Select Model", className="nord-label"),
                    dcc.Dropdown(
                        id="runner-model-select",
                        options=[],
                        placeholder="Choose a model…",
                    ),
                ],
                className="nord-card",
            ),
            # ===== Stream Preview Section =====
            html.Div(
                [
                    html.Div("Stream Preview", className="nord-card-header"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Preview Events", className="nord-label"),
                                    dbc.Input(
                                        id="runner-preview-count", type="number",
                                        min=10, max=5000, step=10, value=200,
                                        className="nord-input",
                                    ),
                                ],
                                style={"width": "140px"},
                            ),
                            dbc.Button("Generate Preview", id="runner-preview-btn",
                                       className="btn-nord-primary ms-3", size="sm"),
                        ],
                        className="d-flex align-items-end mb-3",
                    ),
                    # Event table
                    html.Div(id="runner-event-table"),
                    # Timeline
                    html.Div(
                        dcc.Graph(id="runner-timeline", config={"displayModeBar": False}),
                        className="mt-2",
                    ),
                    # Feature summary
                    html.Div(
                        dcc.Graph(id="runner-feat-radar", figure=build_radar(),
                                  config={"displayModeBar": False}),
                        className="mt-2",
                    ),
                ],
                className="nord-card",
            ),
            # ===== Network Streaming Section =====
            html.Div(
                [
                    html.Div("Network Streaming", className="nord-card-header"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Adapter", className="nord-label"),
                                    dcc.Dropdown(
                                        id="runner-adapter",
                                        options=[
                                            {"label": "Console", "value": "console"},
                                            {"label": "WebSocket", "value": "websocket"},
                                            {"label": "Kafka", "value": "kafka"},
                                        ],
                                        value="console",
                                        clearable=False,
                                    ),
                                ],
                                className="col-3",
                            ),
                            html.Div(
                                [
                                    html.Label("Host", className="nord-label"),
                                    dbc.Input(id="runner-host", type="text", value="localhost",
                                              className="nord-input"),
                                ],
                                className="col-3",
                            ),
                            html.Div(
                                [
                                    html.Label("Port", className="nord-label"),
                                    dbc.Input(id="runner-port", type="number", value=8765,
                                              className="nord-input"),
                                ],
                                className="col-2",
                            ),
                            html.Div(
                                [
                                    html.Label("Topic", className="nord-label"),
                                    dbc.Input(id="runner-topic", type="text",
                                              value="stream-of-intent", className="nord-input"),
                                ],
                                className="col-2",
                            ),
                            html.Div(
                                [
                                    html.Label("Max Loops", className="nord-label"),
                                    dbc.Input(id="runner-max-loops", type="number",
                                              min=0, value=0, className="nord-input"),
                                ],
                                className="col-2",
                            ),
                        ],
                        className="row g-2 mb-3",
                    ),
                    html.Div(
                        [
                            dbc.Button("▶ Start Streaming", id="runner-stream-start",
                                       className="btn-nord-success me-2", size="sm"),
                            dbc.Button("■ Stop Streaming", id="runner-stream-stop",
                                       className="btn-nord-danger me-2", size="sm", disabled=True),
                            html.Span(id="runner-stream-status",
                                      style={"fontSize": "0.82rem", "color": NORD["nord4"], "marginLeft": "1rem"}),
                        ],
                        className="d-flex align-items-center mb-2",
                    ),
                    html.Div(id="runner-stream-info"),
                ],
                className="nord-card",
            ),
            # Polling interval
            dcc.Interval(id="runner-stream-poll", interval=2000, disabled=True),
        ],
    )


def register_callbacks(app):
    """Register callbacks for the runner page."""

    # Populate model selector
    @app.callback(
        Output("runner-model-select", "options"),
        Input("runner-model-select", "id"),
    )
    def populate_models(_):
        models = list_models()
        return [{"label": m["name"], "value": m["id"]} for m in models]

    # Generate preview
    @app.callback(
        Output("runner-event-table", "children"),
        Output("runner-timeline", "figure"),
        Output("runner-feat-radar", "figure"),
        Input("runner-preview-btn", "n_clicks"),
        State("runner-model-select", "value"),
        State("runner-preview-count", "value"),
        prevent_initial_call=True,
    )
    def generate_preview(n, model_id, count):
        if not model_id:
            return html.Div("Select a model first.", className="text-muted-nord"), go.Figure(), build_radar()

        rep = load_representation(model_id)
        if rep is None:
            return html.Div("Model not found.", className="text-muted-nord"), go.Figure(), build_radar()

        count = count or 200
        processor = StreamProcessor(rep, StreamConfig(max_loops=1))
        events = list(itertools.islice(processor.run(), count))

        if not events:
            return html.Div("No events generated.", className="text-muted-nord"), go.Figure(), build_radar()

        # Event table (first 100 rows)
        table_rows = []
        for ev in events[:100]:
            table_rows.append(
                html.Tr([
                    html.Td(ev.case_id, style={"color": NORD["nord8"]}),
                    html.Td(ev.activity),
                    html.Td(f"{ev.timestamp:.2f}"),
                    html.Td(ev.event_type),
                    html.Td(f"{ev.arrival_timestamp:.2f}"),
                ])
            )

        table = html.Div(
            html.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Case ID"), html.Th("Activity"), html.Th("Timestamp"),
                        html.Th("Type"), html.Th("Arrival"),
                    ], style={"backgroundColor": NORD["nord2"], "color": NORD["nord8"],
                              "fontSize": "0.78rem", "textTransform": "uppercase"})),
                    html.Tbody(table_rows),
                ],
                style={"width": "100%", "fontSize": "0.8rem", "borderCollapse": "collapse"},
            ),
            style={"maxHeight": "350px", "overflowY": "auto", "border": f"1px solid {NORD['nord2']}",
                   "borderRadius": "4px"},
        )

        # Timeline figure (Gantt-like)
        timeline_fig = _build_timeline(events)

        # Feature summary
        fv = extract_features(events)
        achieved = {ft.value: fv.get(ft) for ft in FEATURE_ORDER}
        radar = build_radar(achieved=achieved, title="Preview Features")

        return table, timeline_fig, radar

    # Start streaming
    @app.callback(
        Output("runner-stream-start", "disabled"),
        Output("runner-stream-stop", "disabled"),
        Output("runner-stream-poll", "disabled"),
        Output("runner-stream-status", "children"),
        Output("runner-stream-info", "children"),
        Input("runner-stream-start", "n_clicks"),
        State("runner-model-select", "value"),
        State("runner-adapter", "value"),
        State("runner-host", "value"),
        State("runner-port", "value"),
        State("runner-topic", "value"),
        State("runner-max-loops", "value"),
        prevent_initial_call=True,
    )
    def start_streaming(n, model_id, adapter, host, port, topic, max_loops):
        global _stream_thread, _stream_event_count
        if not model_id:
            return False, True, True, "Select a model first.", ""

        rep = load_representation(model_id)
        if rep is None:
            return False, True, True, "Model not found.", ""

        _stream_stop.clear()
        _stream_event_count = 0

        config = StreamConfig(
            adapter=adapter or "console",
            host=host or "localhost",
            port=port or 8765,
            topic=topic or "stream-of-intent",
            max_loops=max_loops or 0,
        )

        def _stream_worker():
            global _stream_event_count
            from stream_of_intent.adapters import create_adapter
            ad = create_adapter(config)
            processor = StreamProcessor(rep, config)
            try:
                ad.connect()
                for event in processor.run():
                    if _stream_stop.is_set():
                        break
                    ad.send(event)
                    with _stream_lock:
                        _stream_event_count += 1
            finally:
                ad.disconnect()

        _stream_thread = threading.Thread(target=_stream_worker, daemon=True)
        _stream_thread.start()

        info = ""
        if adapter == "websocket":
            info = html.Div([
                html.Span("Connect at: ", style={"color": NORD["nord4"]}),
                html.Code(f"ws://{host}:{port}", style={"color": NORD["nord8"]}),
            ])

        return True, False, False, "Streaming…", info

    # Stop streaming
    @app.callback(
        Output("runner-stream-start", "disabled", allow_duplicate=True),
        Output("runner-stream-stop", "disabled", allow_duplicate=True),
        Output("runner-stream-poll", "disabled", allow_duplicate=True),
        Output("runner-stream-status", "children", allow_duplicate=True),
        Input("runner-stream-stop", "n_clicks"),
        prevent_initial_call=True,
    )
    def stop_streaming(n):
        _stream_stop.set()
        return False, True, True, "Stopped."

    # Poll stream status
    @app.callback(
        Output("runner-stream-status", "children", allow_duplicate=True),
        Input("runner-stream-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_stream(n):
        with _stream_lock:
            count = _stream_event_count
        alive = _stream_thread is not None and _stream_thread.is_alive()
        if alive:
            return f"Streaming… {count:,} events sent"
        return f"Stream ended — {count:,} events total"


def _build_timeline(events: list) -> go.Figure:
    """Build a simple Gantt-like timeline from events."""
    # Group start/end pairs by case_id + activity
    cases = {}
    for ev in events:
        key = (ev.case_id, ev.activity)
        if key not in cases:
            cases[key] = {"start": None, "end": None, "case_id": ev.case_id, "activity": ev.activity}
        if ev.event_type == "start":
            cases[key]["start"] = ev.timestamp
        elif ev.event_type == "end":
            cases[key]["end"] = ev.timestamp

    fig = go.Figure()
    case_ids = sorted(set(ev.case_id for ev in events))
    case_y = {cid: i for i, cid in enumerate(case_ids)}

    colors = [NORD["nord8"], NORD["nord9"], NORD["nord10"], NORD["nord15"],
              NORD["nord14"], NORD["nord12"], NORD["nord13"], NORD["nord7"]]
    act_set = sorted(set(ev.activity for ev in events))
    act_color = {a: colors[i % len(colors)] for i, a in enumerate(act_set)}

    for entry in cases.values():
        if entry["start"] is None or entry["end"] is None:
            continue
        y = case_y.get(entry["case_id"], 0)
        fig.add_trace(go.Bar(
            x=[entry["end"] - entry["start"]],
            y=[entry["case_id"]],
            base=[entry["start"]],
            orientation="h",
            marker=dict(color=act_color.get(entry["activity"], NORD["nord8"]), opacity=0.8),
            name=entry["activity"],
            showlegend=False,
            hovertemplate=f"<b>{entry['activity']}</b><br>Case: {entry['case_id']}<br>"
                          f"Start: {entry['start']:.2f}<br>End: {entry['end']:.2f}<extra></extra>",
        ))

    fig.update_layout(
        **plotly_layout(
            title=dict(text="Event Timeline", font=dict(size=13, color=NORD["nord5"])),
            height=max(200, len(case_ids) * 28 + 80),
            barmode="overlay",
            xaxis=dict(title="Time", gridcolor=NORD["nord2"]),
            yaxis=dict(title="", gridcolor=NORD["nord2"], autorange="reversed",
                       tickfont=dict(size=10)),
            margin=dict(l=100, r=20, t=40, b=40),
        )
    )
    return fig
