from __future__ import annotations

import dash_cytoscape as cyto
from dash import html

from stream_of_intent.webapp.components.theme import NORD

try:
    cyto.load_extra_layouts()
except Exception:
    pass

_STYLESHEET = [
    {
        "selector": ".activity-node",
        "style": {
            "label": "data(label)",
            "background-color": NORD["nord10"],
            "color": NORD["nord5"],
            "font-size": "11px",
            "font-weight": "600",
            "text-halign": "center",
            "text-valign": "center",
            "width": 48,
            "height": 48,
            "border-width": 2,
            "border-color": NORD["nord9"],
        },
    },
    {
        "selector": ".control-node",
        "style": {
            "label": "data(label)",
            "shape": "diamond",
            "width": 38,
            "height": 38,
            "font-size": "9px",
            "font-weight": "700",
            "text-halign": "center",
            "text-valign": "center",
            "color": NORD["nord0"],
        },
    },
    {
        "selector": ".start-node",
        "style": {
            "background-color": NORD["nord14"],
            "border-color": NORD["nord14"],
            "border-width": 2,
        },
    },
    {
        "selector": ".end-node",
        "style": {
            "background-color": NORD["nord11"],
            "border-color": NORD["nord11"],
            "border-width": 2,
        },
    },
    {
        "selector": ".subprocess-node",
        "style": {
            "label": "data(label)",
            "background-color": NORD["nord15"],
            "color": NORD["nord5"],
            "font-size": "10px",
            "text-halign": "center",
            "text-valign": "center",
            "width": 40,
            "height": 40,
            "border-width": 2,
            "border-color": NORD["nord15"],
            "border-style": "dashed",
            "shape": "round-rectangle",
        },
    },
    {
        "selector": ".main-edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-color": NORD["nord8"],
            "line-color": NORD["nord8"],
            "width": "mapData(weight, 0, 1, 0.8, 5)",
            "opacity": "mapData(weight, 0, 1, 0.25, 0.9)",
            "label": "data(label)",
            "font-size": "8px",
            "color": NORD["nord3"],
            "text-opacity": 0.8,
            "text-background-color": NORD["nord1"],
            "text-background-opacity": 0.7,
            "text-background-padding": "2px",
        },
    },
    {
        "selector": ".sub-edge",
        "style": {
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-color": NORD["nord15"],
            "line-color": NORD["nord15"],
            "width": "mapData(weight, 0, 1, 0.6, 4)",
            "opacity": "mapData(weight, 0, 1, 0.2, 0.8)",
            "line-style": "dashed",
            "font-size": "8px",
            "color": NORD["nord3"],
        },
    },
    {
        "selector": ":selected",
        "style": {
            "border-color": NORD["nord13"],
            "border-width": 3,
            "line-color": NORD["nord13"],
            "target-arrow-color": NORD["nord13"],
        },
    },
]


def create_model_graph(
    graph_id: str = "model-cytoscape",
    elements: list | None = None,
    layout_name: str = "cose",
    height: str = "500px",
) -> html.Div:
    """Create a Cytoscape graph component for process model visualisation."""
    return html.Div(
        [
            cyto.Cytoscape(
                id=graph_id,
                elements=elements or [],
                stylesheet=_STYLESHEET,
                layout={
                    "name": layout_name,
                    "animate": True,
                    "animationDuration": 300,
                    "nodeRepulsion": 6000,
                    "idealEdgeLength": 120,
                    "edgeElasticity": 50,
                    "gravity": 0.3,
                    "padding": 30,
                },
                style={"width": "100%", "height": height},
                responsive=True,
            ),
        ],
        className="cytoscape-container",
    )
