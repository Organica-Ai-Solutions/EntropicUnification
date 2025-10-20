"""
Results Panel Component for EntropicUnification Dashboard

This module provides the results panel for visualizing simulation results.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

def create_results_panel():
    """Create the results panel component."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Simulation Results", className="mb-4"),
                            html.P(
                                "Visualize and analyze simulation results.",
                                className="text-muted",
                            ),
                        ],
                        width=12,
                    )
                ]
            ),
            
            # Plot Controls
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Plot Style"),
                                                        dcc.Dropdown(
                                                            id="dropdown-plot-style",
                                                            options=[
                                                                {"label": "Plotly", "value": "plotly"},
                                                                {"label": "Plotly White", "value": "plotly_white"},
                                                                {"label": "Plotly Dark", "value": "plotly_dark"},
                                                                {"label": "Seaborn", "value": "seaborn"},
                                                            ],
                                                            value="plotly_white",
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Plot Type"),
                                                        dcc.Dropdown(
                                                            id="dropdown-plot-type",
                                                            options=[
                                                                {"label": "All", "value": "all"},
                                                                {"label": "Loss Curves", "value": "loss"},
                                                                {"label": "Entropy vs Area", "value": "entropy_area"},
                                                                {"label": "Entropy Components", "value": "entropy_components"},
                                                                {"label": "Metric Evolution", "value": "metric_evolution"},
                                                            ],
                                                            value="all",
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Download Format"),
                                                        dcc.Dropdown(
                                                            id="dropdown-download-format",
                                                            options=[
                                                                {"label": "PNG", "value": "png"},
                                                                {"label": "SVG", "value": "svg"},
                                                                {"label": "PDF", "value": "pdf"},
                                                            ],
                                                            value="png",
                                                        ),
                                                    ],
                                                    width=4,
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            [
                                                                dbc.Button(
                                                                    "Download All Plots",
                                                                    id="btn-download-plots",
                                                                    color="success",
                                                                    className="mt-3 me-2",
                                                                ),
                                                                dbc.Button(
                                                                    "Export Data",
                                                                    id="btn-export-data",
                                                                    color="info",
                                                                    className="mt-3",
                                                                ),
                                                            ],
                                                            className="d-flex justify-content-end",
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
            
            # Loss Curves Plot
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Loss Curves", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="graph-loss-curves",
                                                config={"responsive": True},
                                                style={"height": "400px"},
                                            ),
                                        ]
                                    ),
                                    dbc.CardFooter(
                                        html.P(
                                            "Loss curves show the optimization progress of the entropic field equations. "
                                            "Decreasing trend indicates convergence toward a consistent solution.",
                                            className="text-muted mb-0",
                                        )
                                    ),
                                ],
                                className="mb-4 shadow-sm",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            
            # Entropy vs Area and Entropy Components Plots
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Entropy vs Area", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="graph-entropy-area",
                                                config={"responsive": True},
                                                style={"height": "400px"},
                                            ),
                                        ]
                                    ),
                                    dbc.CardFooter(
                                        html.P(
                                            "This plot shows the relationship between entanglement entropy and boundary area. "
                                            "In holographic theories, entropy is expected to be proportional to area (S ∝ A).",
                                            className="text-muted mb-0",
                                        )
                                    ),
                                ],
                                className="mb-4 shadow-sm",
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Entropy Components", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="graph-entropy-components",
                                                config={"responsive": True},
                                                style={"height": "400px"},
                                            ),
                                        ]
                                    ),
                                    dbc.CardFooter(
                                        html.P(
                                            "This chart shows the relative contributions to the total entanglement entropy. "
                                            "Edge modes represent gauge degrees of freedom at the entangling surface.",
                                            className="text-muted mb-0",
                                        )
                                    ),
                                ],
                                className="mb-4 shadow-sm",
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            
            # Metric Evolution Plot
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Metric Evolution", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="graph-metric-evolution",
                                                config={"responsive": True},
                                                style={"height": "500px"},
                                            ),
                                        ]
                                    ),
                                    dbc.CardFooter(
                                        html.P(
                                            "These heatmaps show the evolution of the spacetime metric tensor (g_μν) during optimization. "
                                            "Changes in the metric reflect how spacetime geometry responds to entanglement entropy.",
                                            className="text-muted mb-0",
                                        )
                                    ),
                                ],
                                className="mb-4 shadow-sm",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            
            # Summary Table
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5("Simulation Summary", className="mb-0")
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                id="div-summary-table",
                                                children=[
                                                    dbc.Table(
                                                        [
                                                            html.Thead(
                                                                html.Tr(
                                                                    [
                                                                        html.Th("Metric"),
                                                                        html.Th("Value"),
                                                                        html.Th("Description"),
                                                                    ]
                                                                )
                                                            ),
                                                            html.Tbody(
                                                                [
                                                                    html.Tr(
                                                                        [
                                                                            html.Td("Area Law Coefficient"),
                                                                            html.Td("0.2534"),
                                                                            html.Td("Proportionality constant between entropy and area"),
                                                                        ]
                                                                    ),
                                                                    html.Tr(
                                                                        [
                                                                            html.Td("R² Value"),
                                                                            html.Td("0.9876"),
                                                                            html.Td("Goodness of fit for the area law"),
                                                                        ]
                                                                    ),
                                                                    html.Tr(
                                                                        [
                                                                            html.Td("Final Loss"),
                                                                            html.Td("0.0012"),
                                                                            html.Td("Final value of the loss function"),
                                                                        ]
                                                                    ),
                                                                    html.Tr(
                                                                        [
                                                                            html.Td("Ricci Scalar"),
                                                                            html.Td("0.0034"),
                                                                            html.Td("Average Ricci scalar curvature"),
                                                                        ]
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        bordered=True,
                                                        hover=True,
                                                        responsive=True,
                                                        striped=True,
                                                    ),
                                                ],
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-4 shadow-sm",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
        ],
        fluid=True,
    )
