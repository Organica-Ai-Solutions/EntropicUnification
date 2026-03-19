"""
Control Panel Component — EntropicUnification Dashboard
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

TEAL  = "#1D9E75"
AMBER = "#FAC775"
BG    = "#030609"
CARD  = "rgba(10,18,28,0.72)"

_CARD_STYLE  = {"background": CARD, "border": "1px solid rgba(29,158,117,0.22)"}
_H_STYLE     = {"color": TEAL, "fontFamily": "Space Mono, monospace",
                "fontSize": "0.78rem", "textTransform": "uppercase", "letterSpacing": "0.08em"}


def _section_head(text):
    return html.P(text, className="mb-2 mt-3", style=_H_STYLE)


def create_control_panel():
    """Create the control panel component."""
    return dbc.Container(
        [
            # ── Header ────────────────────────────────────────────────────────
            dbc.Row(dbc.Col([
                html.H3("Simulation Controls",
                        style={"color": TEAL, "fontFamily": "Space Mono, monospace"}),
                html.P("Configure and run EntropicUnification simulations.",
                       className="text-muted mb-4"),
            ], width=12)),

            # ── Row 1: Quantum + Spacetime ────────────────────────────────────
            dbc.Row([
                # Quantum Parameters
                dbc.Col([
                    _section_head("Quantum Parameters"),
                    dbc.Card(dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Qubits", style={"fontSize": "0.82rem"}),
                                dbc.Input(id="input-quantum-qubits", type="number",
                                          min=2, max=10, step=1, value=4),
                                dbc.FormText("2 – 10"),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Circuit Depth", style={"fontSize": "0.82rem"}),
                                dbc.Input(id="input-quantum-depth", type="number",
                                          min=1, max=10, step=1, value=4),
                                dbc.FormText("1 – 10"),
                            ], width=6),
                        ]),
                        dbc.Row(dbc.Col([
                            dbc.Label("Initial State", className="mt-3",
                                      style={"fontSize": "0.82rem"}),
                            dcc.Dropdown(
                                id="dropdown-initial-state",
                                options=[
                                    {"label": "Bell State (S = ln 2 ≈ 0.693)", "value": "bell"},
                                    {"label": "GHZ State",                      "value": "ghz"},
                                    {"label": "Random State",                   "value": "random"},
                                ],
                                value="bell", clearable=False,
                            ),
                        ], width=12), className="mt-1"),
                    ]), className="glass-card", style=_CARD_STYLE),
                ], width=12, lg=6),

                # Spacetime Parameters
                dbc.Col([
                    _section_head("Spacetime Parameters"),
                    dbc.Card(dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Dimensions", style={"fontSize": "0.82rem"}),
                                dbc.Input(id="input-spacetime-dimensions", type="number",
                                          min=2, max=4, step=1, value=2),
                                dbc.FormText("2 – 4"),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Lattice Size", style={"fontSize": "0.82rem"}),
                                dbc.Input(id="input-spacetime-lattice", type="number",
                                          min=16, max=128, step=16, value=32),
                                dbc.FormText("16 – 128 (64 = H3 default)"),
                            ], width=6),
                        ]),
                        dbc.Row(dbc.Col([
                            dbc.Label("Stress Tensor Formulation", className="mt-3",
                                      style={"fontSize": "0.82rem"}),
                            dcc.Dropdown(
                                id="dropdown-stress-form",
                                options=[
                                    {"label": "Massless (traceless, E=pc)",    "value": "massless"},
                                    {"label": "Faulkner (holographic modular)", "value": "faulkner"},
                                    {"label": "Jacobson (thermodynamic)",       "value": "jacobson"},
                                    {"label": "Canonical (symmetric)",          "value": "canonical"},
                                    {"label": "Lagrangian (covariant)",         "value": "lagrangian"},
                                ],
                                value="massless", clearable=False,
                            ),
                        ], width=12), className="mt-1"),
                    ]), className="glass-card", style=_CARD_STYLE),
                ], width=12, lg=6),
            ], className="mb-2"),

            # ── Row 2: Optimization ───────────────────────────────────────────
            dbc.Row(dbc.Col([
                _section_head("Optimization Parameters"),
                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Iterations", style={"fontSize": "0.82rem"}),
                            dbc.Input(id="input-optimization-steps", type="number",
                                      min=50, max=5000, step=50, value=300),
                            dbc.FormText("50 – 5 000  (H3 used 1 500)"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Learning Rate", style={"fontSize": "0.82rem"}),
                            dbc.Input(id="input-optimization-lr", type="number",
                                      min=0.0001, max=0.1, step=0.0005, value=0.001),
                            dbc.FormText("0.0001 – 0.1"),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Optimizer", style={"fontSize": "0.82rem"}),
                            dcc.Dropdown(
                                id="dropdown-optimization-strategy",
                                options=[
                                    {"label": "Adam (default)",    "value": "adam"},
                                    {"label": "SGD",               "value": "sgd"},
                                    {"label": "L-BFGS",            "value": "lbfgs"},
                                ],
                                value="adam", clearable=False,
                            ),
                            dbc.FormText("Adam was used for all H3 runs"),
                        ], width=4),
                    ]),
                ]), className="glass-card", style=_CARD_STYLE),
            ], width=12), className="mb-2"),

            # ── Run Controls ──────────────────────────────────────────────────
            dbc.Row(dbc.Col(
                html.Div([
                    dbc.Button("Run Simulation", id="btn-run-simulation",
                               color="success", size="lg", className="me-2"),
                    dbc.Button("Stop", id="btn-stop-simulation",
                               color="danger", size="lg", className="me-2", disabled=True),
                    dbc.Button("Reset", id="btn-reset-simulation",
                               color="secondary", size="lg", outline=True),
                ], className="d-flex justify-content-center my-4"),
                width=12,
            )),

            # ── Progress ──────────────────────────────────────────────────────
            dbc.Row(dbc.Col([
                dbc.Progress(id="progress-simulation", value=0, label="0%",
                             striped=True, animated=True,
                             style={"height": "22px"}, className="mb-2"),
                html.Div(id="simulation-status-text", className="text-center mb-4"),
            ], width=12)),

            # ── Load Previous Results ─────────────────────────────────────────
            dbc.Row(dbc.Col([
                _section_head("Load Previous Results"),
                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Result Directory", style={"fontSize": "0.82rem"}),
                            dcc.Dropdown(
                                id="dropdown-result-dir",
                                options=[],          # populated by callback
                                placeholder="Select a run directory…",
                                clearable=True,
                            ),
                            dbc.FormText(
                                "Directories under schwarzschild_results/ — "
                                "sorted newest first.",
                                style={"color": "rgba(255,255,255,0.4)"},
                            ),
                        ], width=9),
                        dbc.Col(
                            html.Div(
                                dbc.Button("Load", id="btn-load-results",
                                           color="info", className="mt-4 w-100"),
                                className="d-flex align-items-end h-100",
                            ),
                            width=3,
                        ),
                    ]),
                    # Load status feedback
                    html.Div(id="load-results-status", className="mt-2"),
                    # PNG image viewer (shown when selected dir has images)
                    html.Div(id="results-image-viewer", className="mt-3"),
                ]), className="glass-card", style=_CARD_STYLE),
            ], width=12), className="mb-4"),
        ],
        fluid=True,
        className="p-3",
    )
