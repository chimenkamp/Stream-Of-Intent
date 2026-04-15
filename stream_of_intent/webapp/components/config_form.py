from __future__ import annotations

from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc


def create_config_form() -> html.Div:
    """Build the RunConfig / StaticParams configuration form."""
    return html.Div(
        [
            html.Div("Configuration", className="nord-card-header"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Window Size", className="nord-label"),
                                    dbc.Input(
                                        id="cfg-window-size", type="number",
                                        min=100, max=50000, step=100, value=1000,
                                        className="nord-input",
                                    ),
                                ],
                                className="col-6",
                            ),
                            html.Div(
                                [
                                    html.Label("Num Activities", className="nord-label"),
                                    dbc.Input(
                                        id="cfg-num-activities", type="number",
                                        min=2, max=100, step=1, value=10,
                                        className="nord-input",
                                    ),
                                ],
                                className="col-6",
                            ),
                        ],
                        className="row g-2 mb-2",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Stream Length", className="nord-label"),
                                    dbc.Input(
                                        id="cfg-stream-length", type="number",
                                        min=500, max=500000, step=500, value=5000,
                                        className="nord-input",
                                    ),
                                ],
                                className="col-6",
                            ),
                            html.Div(
                                [
                                    html.Label("Eval Windows", className="nord-label"),
                                    dbc.Input(
                                        id="cfg-eval-windows", type="number",
                                        min=1, max=50, step=1, value=5,
                                        className="nord-input",
                                    ),
                                ],
                                className="col-6",
                            ),
                        ],
                        className="row g-2 mb-2",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Random Seed", className="nord-label"),
                                    dbc.Input(
                                        id="cfg-seed", type="number",
                                        min=0, max=999999, step=1, value=42,
                                        className="nord-input",
                                    ),
                                ],
                                className="col-6",
                            ),
                            html.Div(
                                [
                                    html.Label("Max Iterations", className="nord-label"),
                                    dbc.Input(
                                        id="cfg-max-iter", type="number",
                                        min=5, max=5000, step=5, value=200,
                                        className="nord-input",
                                    ),
                                ],
                                className="col-6",
                            ),
                        ],
                        className="row g-2",
                    ),
                ],
            ),
            dcc.Store(id="config-store", data={}),
        ],
        className="nord-card",
    )


def register_config_callbacks(app):
    """Register callback to collect config into config-store."""

    @app.callback(
        Output("config-store", "data"),
        Input("cfg-window-size", "value"),
        Input("cfg-num-activities", "value"),
        Input("cfg-stream-length", "value"),
        Input("cfg-eval-windows", "value"),
        Input("cfg-seed", "value"),
        Input("cfg-max-iter", "value"),
    )
    def assemble_config(window_size, num_act, stream_len, eval_win, seed, max_iter):
        return {
            "window_size": window_size or 1000,
            "num_activities": num_act or 10,
            "stream_length": stream_len or 5000,
            "num_windows_for_evaluation": eval_win or 5,
            "random_seed": seed if seed is not None else 42,
            "max_iterations": max_iter or 200,
        }
