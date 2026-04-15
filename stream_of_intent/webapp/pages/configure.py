"""Configure & Train page — spec editor, config, real-time training viz."""

from __future__ import annotations

import json
import time

from dash import dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from stream_of_intent.types import (
    FEATURE_ORDER,
    Condition,
    Exclude,
    Expression,
    FeatureType,
    OptimizationSpec,
    StaticParams,
    Symbol,
    Target,
)
from stream_of_intent.webapp.components.config_form import create_config_form
from stream_of_intent.webapp.components.feature_radar import build_radar
from stream_of_intent.webapp.components.loss_plot import build_loss_figure, create_loss_plot
from stream_of_intent.webapp.components.spec_editor import create_spec_editor
from stream_of_intent.webapp.components.theme import NORD
from stream_of_intent.webapp.optimization_runner import get_current_run, start_run


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.Span("⚙", className="title-icon"), "Configure & Train"],
                        className="page-title",
                    ),
                    html.Div(
                        [
                            html.Div(id="train-fitness-badge"),
                            dbc.Button(
                                "▶  Start Training",
                                id="btn-start-train",
                                className="btn-nord-primary",
                            ),
                            dbc.Button(
                                "■  Stop",
                                id="btn-stop-train",
                                className="btn-nord-danger",
                                disabled=True,
                            ),
                            dbc.Button(
                                "💾  Save Model",
                                id="btn-save-model",
                                className="btn-nord-success",
                                disabled=True,
                            ),
                        ],
                        className="header-actions",
                    ),
                ],
                className="page-header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            create_spec_editor(),
                            create_config_form(),
                        ],
                        className="panel-left",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(id="train-status-badge", children="Idle",
                                              style={"display": "inline-block", "padding": "0.2rem 0.7rem",
                                                     "borderRadius": "3px", "fontSize": "0.75rem",
                                                     "fontWeight": "600", "backgroundColor": NORD["nord3"],
                                                     "color": NORD["nord5"]}),
                                    html.Span(id="train-status-text", className="ms-3",
                                              style={"fontSize": "0.82rem", "color": NORD["nord4"]}),
                                ],
                                className="nord-card mb-2",
                                style={"padding": "0.7rem 1rem", "display": "flex", "alignItems": "center"},
                            ),
                            html.Div(create_loss_plot(), className="nord-card"),
                            html.Div(
                                [
                                    dcc.Graph(id="train-radar", figure=build_radar(),
                                              config={"displayModeBar": False}),
                                ],
                                className="nord-card",
                            ),
                            html.Div(
                                [
                                    html.Div("Best Parameters", className="nord-card-header"),
                                    html.Div(id="best-params-table"),
                                ],
                                className="nord-card",
                            ),
                        ],
                        className="panel-right",
                    ),
                ],
                className="panel-layout",
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Save Trained Model"), close_button=True),
                    dbc.ModalBody(
                        [
                            html.Label("Model Name", className="nord-label"),
                            dbc.Input(
                                id="save-model-name", type="text",
                                placeholder="my-model-v1",
                                className="nord-input mb-2",
                            ),
                            html.Label("Description", className="nord-label"),
                            dbc.Textarea(
                                id="save-model-desc",
                                placeholder="Describe the model...",
                                className="nord-textarea mb-2",
                                style={"height": "80px"},
                            ),
                            html.Label("Tags", className="nord-label"),
                            dbc.Input(
                                id="save-model-tags", type="text",
                                placeholder="experiment, v1, temporal",
                                className="nord-input",
                            ),
                            html.Div(id="save-feedback", className="mt-3"),
                        ],
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Save", id="btn-confirm-save", className="btn-nord-primary"),
                            dbc.Button("Cancel", id="btn-cancel-save", className="btn-nord-secondary"),
                        ],
                    ),
                ],
                id="save-modal",
                is_open=False,
                centered=True,
            ),
            dcc.Store(id="has-model-store", data=False),
            dcc.Interval(id="train-poll", interval=1500, disabled=True),
        ],
    )


def register_callbacks(app):
    """Register all callbacks for the configure & train page."""

    from stream_of_intent.webapp.components.spec_editor import register_spec_callbacks
    from stream_of_intent.webapp.components.config_form import register_config_callbacks

    register_spec_callbacks(app)
    register_config_callbacks(app)

    @app.callback(
        Output("train-poll", "disabled"),
        Output("btn-start-train", "disabled"),
        Output("btn-stop-train", "disabled"),
        Output("has-model-store", "data"),
        Input("btn-start-train", "n_clicks"),
        State("spec-store", "data"),
        State("config-store", "data"),
        prevent_initial_call=True,
    )
    def start_training(n_clicks, spec_data, config_data):
        if not n_clicks:
            return no_update, no_update, no_update, no_update

        spec = _build_spec(spec_data or {})
        cfg = config_data or {}
        static = StaticParams(
            window_size=cfg.get("window_size", 1000),
            num_activities=cfg.get("num_activities", 10),
            stream_length=cfg.get("stream_length", 5000),
            num_windows_for_evaluation=cfg.get("num_windows_for_evaluation", 5),
            random_seed=cfg.get("random_seed", 42),
        )
        max_iter = cfg.get("max_iterations", 200)

        start_run(spec, static, max_iter)

        return False, True, False, False
    
    @app.callback(
        Output("btn-save-model", "disabled"),
        Input("has-model-store", "data"),
    )
    def toggle_save_button(has_model):
        return not has_model

    @app.callback(
        Output("btn-stop-train", "disabled", allow_duplicate=True),
        Input("btn-stop-train", "n_clicks"),
        prevent_initial_call=True,
    )
    def stop_training(n_clicks):
        run = get_current_run()
        if run:
            run.cancel()
        return True

    @app.callback(
        Output("loss-plot", "figure"),
        Output("train-radar", "figure"),
        Output("train-status-badge", "children"),
        Output("train-status-badge", "style"),
        Output("train-status-text", "children"),
        Output("best-params-table", "children"),
        Output("train-fitness-badge", "children"),
        Output("train-poll", "disabled", allow_duplicate=True),
        Output("btn-start-train", "disabled", allow_duplicate=True),
        Output("btn-stop-train", "disabled", allow_duplicate=True),
        Output("has-model-store", "data", allow_duplicate=True),
        Input("train-poll", "n_intervals"),
        State("spec-store", "data"),
        prevent_initial_call=True,
    )
    def poll_progress(n, spec_data):
        run = get_current_run()
        if run is None:
            return (no_update,) * 11

        progress = run.progress
        loss_fig = build_loss_figure(progress)

        target_feats = _extract_targets(spec_data or {})
        achieved_feats = None
        if run.achieved_features:
            achieved_feats = {ft.value: run.achieved_features.get(ft) for ft in FEATURE_ORDER}

        radar_fig = build_radar(achieved=achieved_feats, target=target_feats)

        fitness_badge = _build_fitness_badge(spec_data or {}, achieved_feats, progress)

        if run.is_running:
            n_trials = len(progress)
            best_dist = progress[-1].best_distance if progress else float("inf")
            elapsed = run.elapsed
            badge_style = {"display": "inline-block", "padding": "0.2rem 0.7rem",
                           "borderRadius": "3px", "fontSize": "0.75rem",
                           "fontWeight": "600", "backgroundColor": NORD["nord13"],
                           "color": NORD["nord0"]}
            status_text = f"Trial {n_trials}  |  Best distance: {best_dist:.4f}  |  {elapsed:.0f}s"
            return loss_fig, radar_fig, "Running", badge_style, status_text, _format_params(progress), fitness_badge, False, True, False, False
        else:
            badge_color = NORD["nord14"] if run.error is None else NORD["nord11"]
            badge_label = "Complete" if run.error is None else "Error"
            badge_style = {"display": "inline-block", "padding": "0.2rem 0.7rem",
                           "borderRadius": "3px", "fontSize": "0.75rem",
                           "fontWeight": "600", "backgroundColor": badge_color,
                           "color": NORD["nord0"]}
            status_text = run.error if run.error else f"Finished — {len(progress)} trials, {run.elapsed:.0f}s"

            if run.result and not achieved_feats:
                meta = run.result.metadata or {}
                af = meta.get("achieved_features")
                if af:
                    achieved_feats = af
                    radar_fig = build_radar(achieved=achieved_feats, target=target_feats)
                    fitness_badge = _build_fitness_badge(spec_data or {}, achieved_feats, progress)

            has_model = run.result is not None
            return loss_fig, radar_fig, badge_label, badge_style, status_text, _format_params(progress), fitness_badge, True, False, True, has_model

    @app.callback(
        Output("save-modal", "is_open"),
        Input("btn-save-model", "n_clicks"),
        Input("btn-cancel-save", "n_clicks"),
        Input("btn-confirm-save", "n_clicks"),
        State("save-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_save_modal(open_clicks, cancel, confirm, is_open):
        from dash import ctx as dash_ctx
        trigger = dash_ctx.triggered_id
        if trigger == "btn-save-model":
            return True
        return False

    @app.callback(
        Output("save-feedback", "children"),
        Input("btn-confirm-save", "n_clicks"),
        State("save-model-name", "value"),
        State("save-model-desc", "value"),
        State("save-model-tags", "value"),
        prevent_initial_call=True,
    )
    def save_model(n_clicks, name, desc, tags):
        if not n_clicks or not name:
            return dbc.Alert("Please enter a model name.", color="warning",
                             style={"backgroundColor": NORD["nord13"], "color": NORD["nord0"],
                                    "border": "none", "fontSize": "0.82rem"})

        run = get_current_run()
        if run is None or run.result is None:
            return dbc.Alert("No trained model available.", color="danger",
                             style={"backgroundColor": NORD["nord11"], "color": NORD["nord6"],
                                    "border": "none", "fontSize": "0.82rem"})

        from stream_of_intent.webapp.db import save_model as db_save
        try:
            model_id = db_save(
                name=name.strip(),
                representation=run.result,
                description=(desc or "").strip(),
                tags=(tags or "").strip(),
            )
            return dbc.Alert(
                f"Model '{name}' saved (ID: {model_id}).",
                color="success",
                style={"backgroundColor": NORD["nord14"], "color": NORD["nord0"],
                       "border": "none", "fontSize": "0.82rem"},
                duration=5000,
            )
        except Exception as exc:
            return dbc.Alert(
                f"Error: {exc}",
                color="danger",
                style={"backgroundColor": NORD["nord11"], "color": NORD["nord6"],
                       "border": "none", "fontSize": "0.82rem"},
            )


def _build_spec(spec_data: dict) -> OptimizationSpec:
    """Convert the spec-store dict to an OptimizationSpec."""
    operators = []
    for ft in FEATURE_ORDER:
        entry = spec_data.get(ft.value, {"type": "exclude"})
        op_type = entry.get("type", "exclude")
        if op_type == "target":
            operators.append(Target(ft, entry.get("value", 0.5)))
        elif op_type == "symbol":
            operators.append(Symbol(ft))
        elif op_type == "condition":
            expr_op = entry.get("operator", ">=")
            val = entry.get("value", 0.3)
            upper = entry.get("upper")
            operators.append(Condition(ft, Expression(expr_op, val, upper)))
        else:
            operators.append(Exclude(ft))
    return OptimizationSpec(operators)


def _extract_targets(spec_data: dict) -> dict:
    """Extract target values from spec-store for radar display."""
    targets = {}
    for ft in FEATURE_ORDER:
        entry = spec_data.get(ft.value, {})
        if entry.get("type") == "target":
            targets[ft.value] = entry.get("value", 0)
    return targets


def _format_params(progress: list) -> html.Div:
    """Format best trial parameters as small cards."""
    if not progress:
        return html.Div("No trials completed yet.", className="text-muted-nord")

    best = min(progress, key=lambda p: p.best_distance if hasattr(p, "best_distance") else p["best_distance"])
    params = best.params if hasattr(best, "params") else best.get("params", {})

    rows = []
    for k, v in sorted(params.items()):
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        rows.append(
            html.Div(
                [
                    html.Span(k, style={"color": NORD["nord4"], "fontSize": "0.78rem", "flex": "1"}),
                    html.Span(val, style={"color": NORD["nord8"], "fontSize": "0.78rem",
                                          "fontVariantNumeric": "tabular-nums", "fontWeight": "600"}),
                ],
                style={"display": "flex", "justifyContent": "space-between",
                       "padding": "0.2rem 0", "borderBottom": f"1px solid {NORD['nord2']}"},
            )
        )
    return html.Div(rows, style={"maxHeight": "300px", "overflowY": "auto"})


def _build_fitness_badge(spec_data: dict, achieved_feats: dict | None, progress: list) -> html.Div:
    """Build an inline fitness summary showing avg delta to target values."""
    items = []

    if progress:
        best_dist = min(p.best_distance if hasattr(p, "best_distance") else p["best_distance"] for p in progress)
        items.append(
            html.Div(
                [
                    html.Span("Distance ", className="fitness-label"),
                    html.Span(f"{best_dist:.4f}", className="fitness-value"),
                ],
                className="fitness-inline",
            )
        )

    if achieved_feats and spec_data:
        deltas = []
        for ft in FEATURE_ORDER:
            entry = spec_data.get(ft.value, {})
            if entry.get("type") == "target":
                target_val = entry.get("value", 0)
                achieved_val = achieved_feats.get(ft.value, 0)
                deltas.append(abs(achieved_val - target_val))
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            color = NORD["nord14"] if avg_delta < 0.1 else NORD["nord13"] if avg_delta < 0.25 else NORD["nord11"]
            items.append(
                html.Div(
                    [
                        html.Span("Avg Δ target ", className="fitness-label"),
                        html.Span(f"{avg_delta:.3f}", style={"color": color, "fontWeight": "700"}),
                    ],
                    className="fitness-inline",
                )
            )

    if not items:
        return html.Div()

    return html.Div(items, style={"display": "flex", "gap": "0.5rem", "alignItems": "center"})
