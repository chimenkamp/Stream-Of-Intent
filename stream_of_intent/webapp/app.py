from __future__ import annotations

from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

from stream_of_intent.webapp.components.theme import NORD
from stream_of_intent.webapp.pages import configure, model_viewer, library, runner, compare


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder="assets",
        title="Stream of Intent",
        update_title=None,
        external_stylesheets=[dbc.themes.DARKLY],
    )

    sidebar = html.Div(
        [
            html.Div(
                [
                    html.H4("Stream of Intent"),
                    html.Small("Intentional Event Stream Generation"),
                ],
                className="brand",
            ),
            dbc.Nav(
                [
                    dbc.NavLink(
                        [html.Span("⚙", className="nav-icon"), "Configure & Train"],
                        href="/", active="exact",
                    ),
                    dbc.NavLink(
                        [html.Span("◈", className="nav-icon"), "Model Viewer"],
                        href="/model-viewer", active="exact",
                    ),
                    dbc.NavLink(
                        [html.Span("☰", className="nav-icon"), "Library"],
                        href="/library", active="exact",
                    ),
                    dbc.NavLink(
                        [html.Span("▶", className="nav-icon"), "Run & Stream"],
                        href="/runner", active="exact",
                    ),
                    dbc.NavLink(
                        [html.Span("⊞", className="nav-icon"), "Compare"],
                        href="/compare", active="exact",
                    ),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        id="sidebar",
    )

    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="viewer-model-select-store"),
            html.Div(
                [
                    sidebar,
                    html.Div(id="page-content"),
                ],
                id="app-container",
            ),
        ],
    )

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def display_page(pathname):
        if pathname == "/model-viewer":
            return model_viewer.layout()
        elif pathname == "/library":
            return library.layout()
        elif pathname == "/runner":
            return runner.layout()
        elif pathname == "/compare":
            return compare.layout()
        else:
            return configure.layout()

    configure.register_callbacks(app)
    model_viewer.register_callbacks(app)
    library.register_callbacks(app)
    runner.register_callbacks(app)
    compare.register_callbacks(app)

    return app
