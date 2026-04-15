from __future__ import annotations

NORD = {
    "nord0": "#2E3440", "nord1": "#3B4252", "nord2": "#434C5E", "nord3": "#4C566A",
    "nord4": "#D8DEE9", "nord5": "#E5E9F0", "nord6": "#ECEFF4",
    "nord7": "#8FBCBB", "nord8": "#88C0D0", "nord9": "#81A1C1", "nord10": "#5E81AC",
    "nord11": "#BF616A", "nord12": "#D08770", "nord13": "#EBCB8B",
    "nord14": "#A3BE8C", "nord15": "#B48EAD",
}

FEATURE_COLORS = [NORD["nord8"], NORD["nord9"], NORD["nord15"], NORD["nord12"], NORD["nord14"]]

MODEL_COLORS = [NORD["nord8"], NORD["nord11"], NORD["nord14"], NORD["nord13"], NORD["nord15"], NORD["nord12"], NORD["nord9"], NORD["nord7"]]


def plotly_layout(**overrides) -> dict:
    """Return a base Plotly layout dict with NORD styling."""
    base = dict(
        paper_bgcolor=NORD["nord1"],
        plot_bgcolor=NORD["nord0"],
        font=dict(color=NORD["nord4"], family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=50, r=20, t=40, b=40),
        xaxis=dict(gridcolor=NORD["nord2"], zerolinecolor=NORD["nord2"]),
        yaxis=dict(gridcolor=NORD["nord2"], zerolinecolor=NORD["nord2"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=NORD["nord4"], size=11)),
    )
    base.update(overrides)
    return base
