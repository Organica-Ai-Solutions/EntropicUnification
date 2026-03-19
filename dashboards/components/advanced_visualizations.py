"""
Advanced visualization components for the EntropicUnification Dashboard.

Provides 3D entropy surface, spacetime diagram, quantum-state bar chart,
and entanglement network — all with a consistent dark space theme.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html
import dash_bootstrap_components as dbc

# ── brand palette (matches custom.css) ───────────────────────────────────────
TEAL   = "#1D9E75"
AMBER  = "#FAC775"
CORAL  = "#993C1D"
BLUE   = "#4d9fff"
BG     = "#030609"
CARD   = "rgba(10,18,28,0.72)"

_GRAPH_CFG = {"displayModeBar": True, "scrollZoom": True, "responsive": True}


# ─────────────────────────────────────────────────────────────────────────────
# Panel layout
# ─────────────────────────────────────────────────────────────────────────────

def create_advanced_visualizations_panel():
    """Create the Advanced Visualizations tab panel."""

    def _card(title, subtitle, graph_id, height=440, is_3d=False):
        h = height if not is_3d else 520
        return dbc.Card(
            dbc.CardBody([
                html.H5(title, className="mb-1", style={"color": TEAL, "fontFamily": "Space Mono, monospace", "letterSpacing": "0.04em"}),
                html.P(subtitle, className="text-muted small mb-2"),
                dcc.Graph(
                    id=graph_id,
                    figure=create_empty_figure(),
                    config=_GRAPH_CFG,
                    style={"height": f"{h}px"},
                ),
            ]),
            className="glass-card mb-4",
            style={"background": CARD, "border": f"1px solid rgba(29,158,117,0.25)"},
        )

    return html.Div(
        [
            # ── header row ──────────────────────────────────────────────────
            dbc.Row(
                dbc.Col([
                    html.H3("Advanced Visualizations",
                            style={"color": TEAL, "fontFamily": "Space Mono, monospace"}),
                    html.P(
                        "Deeper insights into the quantum-geometric relationship. "
                        "Run a simulation or click Load Demo to populate all four plots.",
                        className="text-muted mb-3",
                    ),
                    dbc.Button(
                        "Load Demo Data",
                        id="btn-load-demo-viz",
                        color="outline-info",
                        size="sm",
                        className="mb-4",
                        style={"borderColor": TEAL, "color": TEAL},
                    ),
                ], width=12),
            ),

            # ── row 1: 3D entropy + spacetime ───────────────────────────────
            dbc.Row([
                dbc.Col(
                    _card("3D Entropy Distribution",
                          "Spatial distribution of entanglement entropy (area-law surface).",
                          "graph-3d-entropy", is_3d=True),
                    width=12, lg=6,
                ),
                dbc.Col(
                    _card("Spacetime Diagram",
                          "Minkowski diagram with inertial worldlines and a geodesic bent by r_s.",
                          "graph-spacetime-diagram"),
                    width=12, lg=6,
                ),
            ]),

            # ── row 2: quantum state + entanglement network ─────────────────
            dbc.Row([
                dbc.Col(
                    _card("Quantum State",
                          "Measurement probabilities and phases for the initial quantum state.",
                          "graph-quantum-state"),
                    width=12, lg=6,
                ),
                dbc.Col(
                    _card("Entanglement Network",
                          "Pairwise entanglement strength between qubits (edge width = strength).",
                          "graph-entanglement-network"),
                    width=12, lg=6,
                ),
            ]),
        ],
        className="p-4",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_empty_figure(message="Run a simulation or click Load Demo"):
    """Dark placeholder figure."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=20),
        annotations=[dict(
            text=message,
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            font=dict(size=13, color="rgba(255,255,255,0.4)", family="Space Mono, monospace"),
        )],
    )
    return fig


def _base_layout(title, plot_style, **kwargs):
    """Shared dark layout kwargs."""
    base = dict(
        title=dict(text=title, font=dict(family="Space Mono, monospace", size=14, color=AMBER)),
        template=plot_style,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", color="rgba(255,255,255,0.75)"),
        margin=dict(l=30, r=20, t=50, b=30),
    )
    base.update(kwargs)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 1. 3D Entropy Surface
# ─────────────────────────────────────────────────────────────────────────────

def create_3d_entropy_visualization(results, plot_style="plotly_dark"):
    """3D entropy surface. Scales with area_law_coefficient from results."""
    x = np.linspace(-5, 5, 28)
    y = np.linspace(-5, 5, 28)
    X, Y = np.meshgrid(x, y)

    coeff = 0.1
    if results:
        coeff = (
            results.get("analysis", {})
                   .get("area_law", {})
                   .get("area_law_coefficient", 0.1)
        ) * 0.4

    Z = coeff * (X**2 + Y**2) * np.exp(-(X**2 + Y**2) / 30)
    suffix = f" — coeff {coeff:.3f}" if results else " — demo"

    fig = go.Figure(data=[go.Surface(
        z=Z, x=X, y=Y,
        colorscale=[[0, BG], [0.35, TEAL], [0.75, AMBER], [1.0, "#fff0b0"]],
        showscale=True,
        colorbar=dict(thickness=12, len=0.6, tickfont=dict(size=10, color="rgba(255,255,255,0.5)")),
        opacity=0.93,
        hovertemplate="x=%{x:.1f}  y=%{y:.1f}<br>S=%{z:.4f}<extra></extra>",
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor=AMBER, project_z=True, width=1),
        ),
    )])
    layout = _base_layout(f"3D Entropy Distribution{suffix}", plot_style)
    layout["margin"] = dict(l=0, r=0, b=0, t=50)
    fig.update_layout(
        **layout,
        scene=dict(
            xaxis=dict(title="X", gridcolor="rgba(255,255,255,0.06)", backgroundcolor=BG, zerolinecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Y", gridcolor="rgba(255,255,255,0.06)", backgroundcolor=BG, zerolinecolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Entropy Density", gridcolor="rgba(255,255,255,0.06)", backgroundcolor=BG, zerolinecolor="rgba(255,255,255,0.1)"),
            bgcolor=BG,
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spacetime Diagram
# ─────────────────────────────────────────────────────────────────────────────

def create_spacetime_diagram(results, plot_style="plotly_dark"):
    """Minkowski diagram with geodesic bent by fitted r_s."""
    t = np.linspace(0, 10, 150)

    r_s = 0.448
    if results:
        r_s = results.get("schwarzschild", {}).get("r_s", 0.448)

    fig = go.Figure()

    # Light cones
    for sign, name in [(1, "Future light cone"), (-1, "Past light cone")]:
        fig.add_trace(go.Scatter(
            x=sign * t, y=t, mode="lines",
            line=dict(color="rgba(250,100,100,0.50)", width=1.5, dash="dash"),
            name=name,
        ))

    # Inertial worldlines
    for i, xi in enumerate(range(-4, 5, 2)):
        fig.add_trace(go.Scatter(
            x=[xi] * len(t), y=t, mode="lines",
            line=dict(color="rgba(77,159,255,0.22)", width=1),
            name="Inertial worldlines" if i == 0 else "",
            showlegend=(i == 0),
        ))

    # Geodesic bent by r_s
    geo_x = np.sin(t) * (1 + r_s * np.exp(-t / 3))
    fig.add_trace(go.Scatter(
        x=geo_x, y=t, mode="lines",
        line=dict(color=TEAL, width=2.5),
        name=f"Geodesic  r_s={r_s:.3f}",
    ))

    fig.update_layout(
        **_base_layout(f"Spacetime Diagram — r_s = {r_s:.3f}", plot_style,
                       xaxis_title="Space", yaxis_title="Time"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11),
            bgcolor="rgba(3,6,9,0.6)",
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Quantum State
# ─────────────────────────────────────────────────────────────────────────────

def create_quantum_state_visualization(results, plot_style="plotly_dark"):
    """Bar chart: probabilities + phases for the initial quantum state."""
    initial_state = "bell"
    if results:
        initial_state = results.get("initial_state", "bell")

    _STATE_DATA = {
        "bell": {
            "states": ["|00⟩", "|01⟩", "|10⟩", "|11⟩"],
            "probs":  [0.5, 0, 0, 0.5],
            "phases": [0, 0, 0, np.pi],
        },
        "ghz": {
            "states": ["|000⟩", "|001⟩", "|010⟩", "|011⟩", "|100⟩", "|101⟩", "|110⟩", "|111⟩"],
            "probs":  [0.5, 0, 0, 0, 0, 0, 0, 0.5],
            "phases": [0] * 8,
        },
        "random": {
            "states": ["|00⟩", "|01⟩", "|10⟩", "|11⟩"],
            "probs":  [0.25, 0.35, 0.15, 0.25],
            "phases": [0.3, 1.1, 2.4, 0.8],
        },
    }
    d = _STATE_DATA.get(initial_state, _STATE_DATA["bell"])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Measurement Probabilities", "Phases (rad)"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Bar(
        x=d["states"], y=d["probs"],
        marker=dict(color=TEAL, line=dict(color=AMBER, width=0.5)),
        name="Probability",
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=d["states"], y=d["phases"],
        marker=dict(color=AMBER, line=dict(color=TEAL, width=0.5)),
        name="Phase",
        hovertemplate="%{x}: %{y:.3f} rad<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        **_base_layout(f"Quantum State — {initial_state.upper()}", plot_style,
                       height=420, showlegend=False),
    )
    # Dark axes for both subplots
    for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
        fig.update_layout(**{axis: dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)")})
    # Subplot title colour
    for ann in fig.layout.annotations:
        ann.font.color = "rgba(255,255,255,0.55)"
        ann.font.size = 12

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Entanglement Network
# ─────────────────────────────────────────────────────────────────────────────

def create_entanglement_network(results, plot_style="plotly_dark"):
    """Circular entanglement network; edge width encodes pairwise strength."""
    num_qubits   = 4
    total_entropy = 0.693

    if results:
        cfg = results.get("config", {})
        num_qubits   = cfg.get("num_qubits") or cfg.get("quantum", {}).get("num_qubits", 4)
        total_entropy = (
            results.get("analysis", {})
                   .get("entropy_components", {})
                   .get("total", 0.693)
        )

    num_qubits = max(2, int(num_qubits))

    theta = np.linspace(0, 2 * np.pi, num_qubits, endpoint=False)
    qx = np.cos(theta)
    qy = np.sin(theta)

    fig = go.Figure()

    # Draw each edge individually so widths/colours differ
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            strength = 0.3 + 0.7 * abs(np.sin((i + j) / num_qubits * np.pi)) * total_entropy
            alpha    = min(0.9, strength)
            fig.add_trace(go.Scatter(
                x=[qx[i], qx[j], None],
                y=[qy[i], qy[j], None],
                mode="lines",
                line=dict(
                    width=max(0.5, strength * 5),
                    color=f"rgba(29,158,117,{alpha:.2f})",
                ),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Qubit nodes
    fig.add_trace(go.Scatter(
        x=qx, y=qy,
        mode="markers+text",
        text=[f"Q{i}" for i in range(num_qubits)],
        textposition="middle center",
        marker=dict(
            size=32,
            color=TEAL,
            line=dict(width=2, color=AMBER),
        ),
        textfont=dict(color=BG, size=11, family="Space Mono, monospace"),
        name="Qubits",
        hovertemplate="Qubit %{text}<extra></extra>",
    ))

    layout = _base_layout(
        f"Entanglement Network — {num_qubits} qubits   S={total_entropy:.3f}",
        plot_style,
        hovermode="closest",
    )
    layout["margin"] = dict(l=10, r=10, t=50, b=10)
    fig.update_layout(
        **layout,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.35, 1.35]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-1.35, 1.35], scaleanchor="x", scaleratio=1),
        showlegend=False,
    )
    return fig
