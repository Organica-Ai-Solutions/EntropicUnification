"""
Control Panel Component for EntropicUnification Dashboard

This module provides the control panel for setting simulation parameters
and running simulations.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

def create_control_panel():
    """Create the control panel component."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Simulation Controls", className="mb-4"),
                            html.P(
                                "Configure and run EntropicUnification simulations.",
                                className="text-muted",
                            ),
                        ],
                        width=12,
                    )
                ]
            ),
            
            # Quantum Parameters
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Quantum Parameters", className="mt-3"),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Number of Qubits"),
                                                        dbc.Input(
                                                            id="input-quantum-qubits",
                                                            type="number",
                                                            min=2,
                                                            max=10,
                                                            step=1,
                                                            value=4,
                                                        ),
                                                        dbc.FormText(
                                                            "Number of qubits in the quantum circuit (2-10)"
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Circuit Depth"),
                                                        dbc.Input(
                                                            id="input-quantum-depth",
                                                            type="number",
                                                            min=1,
                                                            max=10,
                                                            step=1,
                                                            value=4,
                                                        ),
                                                        dbc.FormText(
                                                            "Depth of the quantum circuit (1-10)"
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Initial State"),
                                                        dcc.Dropdown(
                                                            id="dropdown-initial-state",
                                                            options=[
                                                                {"label": "Bell State", "value": "bell"},
                                                                {"label": "GHZ State", "value": "ghz"},
                                                                {"label": "Random State", "value": "random"},
                                                            ],
                                                            value="bell",
                                                        ),
                                                        dbc.FormText(
                                                            "Initial quantum state for the simulation"
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ],
                                            className="mt-3",
                                        ),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        ],
                        width=6,
                    ),
                    
                    # Spacetime Parameters
                    dbc.Col(
                        [
                            html.H4("Spacetime Parameters", className="mt-3"),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Dimensions"),
                                                        dbc.Input(
                                                            id="input-spacetime-dimensions",
                                                            type="number",
                                                            min=2,
                                                            max=4,
                                                            step=1,
                                                            value=4,
                                                        ),
                                                        dbc.FormText(
                                                            "Number of spacetime dimensions (2-4)"
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Lattice Size"),
                                                        dbc.Input(
                                                            id="input-spacetime-lattice",
                                                            type="number",
                                                            min=16,
                                                            max=128,
                                                            step=16,
                                                            value=64,
                                                        ),
                                                        dbc.FormText(
                                                            "Size of the spacetime lattice (16-128)"
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Stress Tensor Formulation"),
                                                        dcc.Dropdown(
                                                            id="dropdown-stress-form",
                                                            options=[
                                                                {"label": "Jacobson", "value": "jacobson"},
                                                                {"label": "Canonical", "value": "canonical"},
                                                                {"label": "Faulkner", "value": "faulkner"},
                                                                {"label": "Modified", "value": "modified"},
                                                            ],
                                                            value="jacobson",
                                                        ),
                                                        dbc.FormText(
                                                            "Formulation of the entropic stress-energy tensor"
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ],
                                            className="mt-3",
                                        ),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            
            # Optimization Parameters
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Optimization Parameters", className="mt-3"),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Optimization Steps"),
                                                        dbc.Input(
                                                            id="input-optimization-steps",
                                                            type="number",
                                                            min=10,
                                                            max=1000,
                                                            step=10,
                                                            value=100,
                                                        ),
                                                        dbc.FormText(
                                                            "Number of optimization steps (10-1000)"
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Learning Rate"),
                                                        dbc.Input(
                                                            id="input-optimization-lr",
                                                            type="number",
                                                            min=0.0001,
                                                            max=0.1,
                                                            step=0.001,
                                                            value=0.001,
                                                        ),
                                                        dbc.FormText(
                                                            "Learning rate for optimization (0.0001-0.1)"
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Optimization Strategy"),
                                                        dcc.Dropdown(
                                                            id="dropdown-optimization-strategy",
                                                            options=[
                                                                {"label": "Standard", "value": "standard"},
                                                                {"label": "Basin Hopping", "value": "basin_hopping"},
                                                                {"label": "Simulated Annealing", "value": "simulated_annealing"},
                                                            ],
                                                            value="standard",
                                                        ),
                                                        dbc.FormText(
                                                            "Strategy for optimization"
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ],
                                            className="mt-3",
                                        ),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            
            # Experimental Features
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Experimental Features", className="mt-3"),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Checklist(
                                                            id="checklist-experimental",
                                                            options=[
                                                                {"label": "Edge Modes", "value": "edge_modes"},
                                                                {"label": "Non-Conformal Matter", "value": "non_conformal"},
                                                                {"label": "Higher Curvature Terms", "value": "higher_curvature"},
                                                            ],
                                                            value=["edge_modes"],
                                                            inline=True,
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            
            # Run Controls
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                "Run Simulation",
                                id="btn-run-simulation",
                                color="primary",
                                size="lg",
                                className="me-3",
                            ),
                            dbc.Button(
                                "Stop Simulation",
                                id="btn-stop-simulation",
                                color="danger",
                                size="lg",
                                className="me-3",
                            ),
                            dbc.Button(
                                "Reset",
                                id="btn-reset-simulation",
                                color="secondary",
                                size="lg",
                            ),
                        ],
                        width=12,
                        className="d-flex justify-content-center mb-4",
                    ),
                ]
            ),
            
            # Progress Bar
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Simulation Progress", className="mt-3"),
                            dbc.Progress(
                                id="progress-simulation",
                                value=0,
                                label="0%",
                                striped=True,
                                animated=True,
                                className="mb-3",
                            ),
                            html.Div(id="simulation-status-text", className="text-center mb-4"),
                        ],
                        width=12,
                    ),
                ]
            ),
            
            # Load Results Section
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Load Previous Results", className="mt-3"),
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Result Directory"),
                                                        dcc.Dropdown(
                                                            id="dropdown-result-dir",
                                                            options=[
                                                                {"label": "Bell State (2023-10-20)", "value": "results/bell/20231020_120000"},
                                                                {"label": "GHZ State (2023-10-19)", "value": "results/ghz/20231019_150000"},
                                                                {"label": "Random State (2023-10-18)", "value": "results/random/20231018_140000"},
                                                            ],
                                                        ),
                                                    ],
                                                    width=9,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            dbc.Button(
                                                                "Load",
                                                                id="btn-load-results",
                                                                color="info",
                                                                className="mt-4",
                                                            ),
                                                            className="d-flex align-items-end h-100",
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                                className="mb-4",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
        ],
        fluid=True,
    )
