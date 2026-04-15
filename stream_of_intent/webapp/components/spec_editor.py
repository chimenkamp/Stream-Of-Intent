from __future__ import annotations

from dash import dcc, html, callback, Input, Output, State, ALL, MATCH, ctx
import dash_bootstrap_components as dbc

from stream_of_intent.types import FEATURE_ORDER, FeatureType

_FEATURE_LABELS = {
    FeatureType.TEMPORAL_DEPENDENCY: "Temporal Dependency",
    FeatureType.LONG_TERM_DEPENDENCY: "Long-Term Dependency",
    FeatureType.NON_LINEAR_DEPENDENCY: "Non-Linear Dependency",
    FeatureType.OUT_OF_ORDER: "Out of Order",
    FeatureType.FRACTAL_BEHAVIOR: "Fractal Behavior",
}

_FEATURE_DESCRIPTIONS = {
    FeatureType.TEMPORAL_DEPENDENCY: "Short-range predictability — how well the next activity is predicted by the previous one.",
    FeatureType.LONG_TERM_DEPENDENCY: "Long-range memory — influence of distant past events on current decisions.",
    FeatureType.NON_LINEAR_DEPENDENCY: "Complex branching, parallelism, and context-aware process decisions.",
    FeatureType.OUT_OF_ORDER: "Proportion and magnitude of events arriving out of their true temporal order.",
    FeatureType.FRACTAL_BEHAVIOR: "Self-similar nested subprocess patterns repeating at different temporal scales.",
}

OPERATOR_TYPES = ["target", "symbol", "condition", "exclude"]


def create_spec_editor() -> html.Div:
    """Build the full OptimizationSpec editor component."""
    feature_cards = []
    for ft in FEATURE_ORDER:
        ft_id = ft.value
        label = _FEATURE_LABELS[ft]

        card = html.Div(
            [
                html.Div(label, className="feature-name"),
                html.Div(
                    [
                        html.Label("Operator", className="nord-label"),
                        dcc.Dropdown(
                            id={"type": "op-selector", "feature": ft_id},
                            options=[
                                {"label": "🎯 Target", "value": "target"},
                                {"label": "✧ Symbol", "value": "symbol"},
                                {"label": "⊳ Condition", "value": "condition"},
                                {"label": "⊘ Exclude", "value": "exclude"},
                            ],
                            value="symbol",
                            clearable=False,
                            className="mb-2",
                        ),
                    ],
                    style={"marginBottom": "0.3rem"},
                ),

                html.Div(
                    [
                        html.Label("Target Value", className="nord-label"),
                        dcc.Slider(
                            id={"type": "target-slider", "feature": ft_id},
                            min=0, max=1, step=0.01, value=0.5,
                            marks={0: "0", 0.5: ".5", 1: "1"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    id={"type": "target-container", "feature": ft_id},
                    style={"display": "none"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Comparison", className="nord-label"),
                                dcc.Dropdown(
                                    id={"type": "cond-op", "feature": ft_id},
                                    options=[
                                        {"label": "<", "value": "<"},
                                        {"label": ">", "value": ">"},
                                        {"label": "≤", "value": "<="},
                                        {"label": "≥", "value": ">="},
                                        {"label": "between", "value": "between"},
                                    ],
                                    value=">=",
                                    clearable=False,
                                ),
                            ],
                            style={"marginBottom": "0.3rem"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Value", className="nord-label"),
                                                dbc.Input(
                                                    id={"type": "cond-val", "feature": ft_id},
                                                    type="number", min=0, max=1, step=0.01, value=0.3,
                                                    className="nord-input",
                                                ),
                                            ],
                                            className="col-6",
                                        ),
                                        html.Div(
                                            [
                                                html.Label("Upper", className="nord-label"),
                                                dbc.Input(
                                                    id={"type": "cond-upper", "feature": ft_id},
                                                    type="number", min=0, max=1, step=0.01, value=0.7,
                                                    className="nord-input", disabled=True,
                                                ),
                                            ],
                                            className="col-6",
                                            id={"type": "cond-upper-wrap", "feature": ft_id},
                                        ),
                                    ],
                                    className="row g-2",
                                ),
                            ],
                        ),
                    ],
                    id={"type": "cond-container", "feature": ft_id},
                    style={"display": "none"},
                ),
            ],
            id={"type": "op-card", "feature": ft_id},
            className="operator-card operator-symbol",
        )
        feature_cards.append(card)

    return html.Div(
        [
            html.Div("Optimization Specification", className="nord-card-header"),
            html.Div(feature_cards, className="feature-grid"),
            dcc.Store(id="spec-store", data={}),
        ],
        className="nord-card",
    )

def register_spec_callbacks(app):
    """Register all callbacks for the spec editor."""

    @app.callback(
        Output({"type": "target-container", "feature": MATCH}, "style"),
        Output({"type": "cond-container", "feature": MATCH}, "style"),
        Output({"type": "op-card", "feature": MATCH}, "className"),
        Input({"type": "op-selector", "feature": MATCH}, "value"),
    )
    def toggle_operator_controls(op_type):
        show = {"display": "block"}
        hide = {"display": "none"}
        base = "operator-card"
        if op_type == "target":
            return show, hide, f"{base} operator-target"
        elif op_type == "condition":
            return hide, show, f"{base} operator-condition"
        elif op_type == "symbol":
            return hide, hide, f"{base} operator-symbol"
        else:
            return hide, hide, f"{base} operator-exclude"

    @app.callback(
        Output({"type": "cond-upper", "feature": MATCH}, "disabled"),
        Input({"type": "cond-op", "feature": MATCH}, "value"),
    )
    def toggle_upper(cond_op):
        return cond_op != "between"

    inputs = []
    for ft in FEATURE_ORDER:
        fid = ft.value
        inputs.extend([
            Input({"type": "op-selector", "feature": fid}, "value"),
            Input({"type": "target-slider", "feature": fid}, "value"),
            Input({"type": "cond-op", "feature": fid}, "value"),
            Input({"type": "cond-val", "feature": fid}, "value"),
            Input({"type": "cond-upper", "feature": fid}, "value"),
        ])

    @app.callback(Output("spec-store", "data"), inputs)
    def assemble_spec(*values):
        spec = {}
        for i, ft in enumerate(FEATURE_ORDER):
            base = i * 5
            op_type = values[base]
            target_val = values[base + 1]
            cond_op = values[base + 2]
            cond_val = values[base + 3]
            cond_upper = values[base + 4]

            if op_type == "target":
                spec[ft.value] = {"type": "target", "value": target_val or 0.5}
            elif op_type == "condition":
                entry = {"type": "condition", "operator": cond_op or ">=", "value": cond_val or 0.3}
                if cond_op == "between":
                    entry["upper"] = cond_upper or 0.7
                spec[ft.value] = entry
            elif op_type == "symbol":
                spec[ft.value] = {"type": "symbol"}
            else:
                spec[ft.value] = {"type": "exclude"}
        return spec
