from __future__ import annotations

import base64
import json
import time

from dash import dcc, html, callback, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc

from stream_of_intent.webapp.components.theme import NORD
from stream_of_intent.webapp.db import (
    delete_model,
    import_model_json,
    list_models,
    get_model,
    update_model,
)


def layout():
    return html.Div(
        [
            html.Div(
                [html.Span("☰", className="title-icon"), "Model Library"],
                className="page-title",
            ),

            html.Div(
                [
                    dbc.Button("↻ Refresh", id="lib-refresh", className="btn-nord-secondary me-2", size="sm"),
                    dbc.Button("⬆ Import .model.json", id="lib-import-btn", className="btn-nord-primary me-2", size="sm"),
                    dbc.Button("Compare Selected", id="lib-compare-btn", className="btn-nord-primary me-2", size="sm"),
                    dbc.Button("🗑 Delete Selected", id="lib-delete-btn", className="btn-nord-danger", size="sm"),
                ],
                className="d-flex mb-3",
            ),

            dbc.Modal(
                [
                    dbc.ModalHeader(
                        dbc.ModalTitle("Import Model"),
                        style={"backgroundColor": NORD["nord1"], "borderColor": NORD["nord2"]},
                    ),
                    dbc.ModalBody(
                        [
                            html.Label("Model Name", className="nord-label"),
                            dbc.Input(id="lib-import-name", type="text",
                                      placeholder="imported-model", className="nord-input mb-2"),
                            html.Label("Description", className="nord-label"),
                            dbc.Input(id="lib-import-desc", type="text",
                                      placeholder="Optional description", className="nord-input mb-2"),
                            html.Label("Upload .model.json", className="nord-label"),
                            dcc.Upload(
                                id="lib-upload",
                                children=html.Div([
                                    "Drag & drop or ",
                                    html.A("click to select", style={"color": NORD["nord8"]}),
                                ]),
                                style={
                                    "borderWidth": "1px", "borderStyle": "dashed",
                                    "borderColor": NORD["nord3"], "borderRadius": "6px",
                                    "padding": "1.5rem", "textAlign": "center",
                                    "backgroundColor": NORD["nord0"], "color": NORD["nord4"],
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={"backgroundColor": NORD["nord1"]},
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("Import", id="lib-import-confirm", className="btn-nord-primary"),
                            dbc.Button("Cancel", id="lib-import-cancel", className="btn-nord-secondary"),
                        ],
                        style={"backgroundColor": NORD["nord1"], "borderColor": NORD["nord2"]},
                    ),
                ],
                id="lib-import-modal",
                is_open=False,
                centered=True,
            ),
            # Feedback
            html.Div(id="lib-feedback"),
            # DataTable
            html.Div(
                [
                    dash_table.DataTable(
                        id="lib-table",
                        columns=[
                            {"name": "ID", "id": "id"},
                            {"name": "Name", "id": "name", "editable": True},
                            {"name": "Best Distance", "id": "best_distance", "type": "numeric",
                             "format": {"specifier": ".4f"}},
                            {"name": "Tags", "id": "tags", "editable": True},
                            {"name": "Created", "id": "created_at"},
                            {"name": "Description", "id": "description", "editable": True},
                        ],
                        data=[],
                        row_selectable="multi",
                        sort_action="native",
                        filter_action="native",
                        page_action="native",
                        page_size=15,
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": NORD["nord2"],
                            "color": NORD["nord8"],
                            "fontWeight": "600",
                            "fontSize": "0.78rem",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.03em",
                            "border": f"1px solid {NORD['nord3']}",
                        },
                        style_cell={
                            "backgroundColor": NORD["nord1"],
                            "color": NORD["nord4"],
                            "border": f"1px solid {NORD['nord2']}",
                            "fontSize": "0.82rem",
                            "padding": "0.5rem",
                            "textAlign": "left",
                        },
                        style_data_conditional=[
                            {
                                "if": {"state": "active"},
                                "backgroundColor": NORD["nord2"],
                                "border": f"1px solid {NORD['nord8']}",
                            },
                            {
                                "if": {"state": "selected"},
                                "backgroundColor": f"rgba(94,129,172,0.15)",
                                "border": f"1px solid {NORD['nord10']}",
                            },
                        ],
                        style_filter={
                            "backgroundColor": NORD["nord0"],
                            "color": NORD["nord4"],
                        },
                    ),
                ],
                className="nord-card",
            ),
            # Row action buttons
            html.Div(
                [
                    dbc.Button("View Selected", id="lib-view-btn", className="btn-nord-primary me-2", size="sm"),
                    dbc.Button("⬇ Download JSON", id="lib-download-btn", className="btn-nord-secondary me-2", size="sm"),
                    dcc.Download(id="lib-download"),
                ],
                className="mt-2",
            ),
            # Hidden store for navigation
            dcc.Store(id="lib-compare-store"),
        ],
    )


def register_callbacks(app):
    """Register callbacks for the library page."""

    # Refresh table
    @app.callback(
        Output("lib-table", "data"),
        Input("lib-refresh", "n_clicks"),
        Input("lib-import-confirm", "n_clicks"),
        Input("lib-delete-btn", "n_clicks"),
    )
    def refresh_table(*_):
        import datetime
        models = list_models()
        for m in models:
            if m.get("created_at"):
                m["created_at"] = datetime.datetime.fromtimestamp(m["created_at"]).strftime("%Y-%m-%d %H:%M")
            m["best_distance"] = m.get("best_distance") or 0.0
        return models

    # Import modal toggle
    @app.callback(
        Output("lib-import-modal", "is_open"),
        Input("lib-import-btn", "n_clicks"),
        Input("lib-import-cancel", "n_clicks"),
        Input("lib-import-confirm", "n_clicks"),
        State("lib-import-modal", "is_open"),
        prevent_initial_call=True,
    )
    def toggle_import_modal(open_c, cancel_c, confirm_c, is_open):
        from dash import ctx
        trigger = ctx.triggered_id
        if trigger == "lib-import-btn":
            return True
        return False

    # Import file
    @app.callback(
        Output("lib-feedback", "children"),
        Input("lib-import-confirm", "n_clicks"),
        State("lib-import-name", "value"),
        State("lib-import-desc", "value"),
        State("lib-upload", "contents"),
        State("lib-upload", "filename"),
        prevent_initial_call=True,
    )
    def do_import(n, name, desc, contents, filename):
        if not n or not contents or not name:
            return dbc.Alert("Please provide a name and upload a file.",
                             style={"backgroundColor": NORD["nord13"], "color": NORD["nord0"],
                                    "border": "none", "fontSize": "0.82rem"}, duration=4000)
        try:
            _, content_string = contents.split(",")
            decoded = base64.b64decode(content_string).decode("utf-8")
            model_id = import_model_json(name.strip(), decoded, description=(desc or "").strip())
            return dbc.Alert(f"Imported '{name}' (ID: {model_id})",
                             style={"backgroundColor": NORD["nord14"], "color": NORD["nord0"],
                                    "border": "none", "fontSize": "0.82rem"}, duration=4000)
        except Exception as exc:
            return dbc.Alert(f"Import error: {exc}",
                             style={"backgroundColor": NORD["nord11"], "color": NORD["nord6"],
                                    "border": "none", "fontSize": "0.82rem"}, duration=6000)

    # Delete selected
    @app.callback(
        Output("lib-feedback", "children", allow_duplicate=True),
        Input("lib-delete-btn", "n_clicks"),
        State("lib-table", "selected_rows"),
        State("lib-table", "data"),
        prevent_initial_call=True,
    )
    def delete_selected(n, selected, data):
        if not n or not selected:
            return no_update
        count = 0
        for idx in selected:
            if idx < len(data):
                delete_model(data[idx]["id"])
                count += 1
        return dbc.Alert(f"Deleted {count} model(s).",
                         style={"backgroundColor": NORD["nord11"], "color": NORD["nord6"],
                                "border": "none", "fontSize": "0.82rem"}, duration=3000)

    # Edit cell (rename, tags, description)
    @app.callback(
        Output("lib-feedback", "children", allow_duplicate=True),
        Input("lib-table", "data_timestamp"),
        State("lib-table", "data"),
        State("lib-table", "data_previous"),
        prevent_initial_call=True,
    )
    def cell_edit(ts, data, prev_data):
        if not data or not prev_data:
            return no_update
        for row, prev_row in zip(data, prev_data):
            if row != prev_row:
                update_model(
                    row["id"],
                    name=row.get("name"),
                    description=row.get("description"),
                    tags=row.get("tags"),
                )
                return no_update
        return no_update

    # Download JSON
    @app.callback(
        Output("lib-download", "data"),
        Input("lib-download-btn", "n_clicks"),
        State("lib-table", "selected_rows"),
        State("lib-table", "data"),
        prevent_initial_call=True,
    )
    def download_json(n, selected, data):
        if not n or not selected:
            return no_update
        idx = selected[0]
        if idx >= len(data):
            return no_update
        record = get_model(data[idx]["id"])
        if record is None:
            return no_update
        return dcc.send_string(record.model_json, filename=f"{record.name}.model.json")

    # View selected → navigate to model viewer
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("viewer-model-select-store", "data"),
        Input("lib-view-btn", "n_clicks"),
        State("lib-table", "selected_rows"),
        State("lib-table", "data"),
        prevent_initial_call=True,
    )
    def view_selected(n, selected, data):
        if not n or not selected:
            return no_update, no_update
        idx = selected[0]
        if idx >= len(data):
            return no_update, no_update
        return "/model-viewer", data[idx]["id"]

    # Compare selected → navigate to compare page
    @app.callback(
        Output("url", "pathname", allow_duplicate=True),
        Output("lib-compare-store", "data"),
        Input("lib-compare-btn", "n_clicks"),
        State("lib-table", "selected_rows"),
        State("lib-table", "data"),
        prevent_initial_call=True,
    )
    def compare_selected(n, selected, data):
        if not n or not selected or len(selected) < 2:
            return no_update, no_update
        ids = [data[i]["id"] for i in selected if i < len(data)]
        return "/compare", ids
