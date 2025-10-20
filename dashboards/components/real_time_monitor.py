"""
Real-Time Monitoring Component for EntropicUnification Dashboard

This module provides real-time monitoring components for the dashboard.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time

def create_system_monitor():
    """Create a system monitor component."""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5("System Monitor", className="mb-0")
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.H6("CPU Usage", className="mb-2"),
                                            dbc.Progress(
                                                id="cpu-usage-progress",
                                                value=0,
                                                label="0%",
                                                color="success",
                                                className="mb-3",
                                            ),
                                        ]
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.H6("Memory Usage", className="mb-2"),
                                            dbc.Progress(
                                                id="memory-usage-progress",
                                                value=0,
                                                label="0%",
                                                color="info",
                                                className="mb-3",
                                            ),
                                        ]
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
                                    html.Div(
                                        [
                                            html.H6("GPU Usage", className="mb-2"),
                                            dbc.Progress(
                                                id="gpu-usage-progress",
                                                value=0,
                                                label="0%",
                                                color="warning",
                                                className="mb-3",
                                            ),
                                        ]
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.H6("Disk I/O", className="mb-2"),
                                            dbc.Progress(
                                                id="disk-io-progress",
                                                value=0,
                                                label="0%",
                                                color="danger",
                                                className="mb-3",
                                            ),
                                        ]
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
                                    html.Div(
                                        id="system-monitor-stats",
                                        children=[
                                            dbc.Table(
                                                [
                                                    html.Thead(
                                                        html.Tr(
                                                            [
                                                                html.Th("Metric"),
                                                                html.Th("Value"),
                                                            ]
                                                        )
                                                    ),
                                                    html.Tbody(
                                                        [
                                                            html.Tr(
                                                                [
                                                                    html.Td("Uptime"),
                                                                    html.Td("00:00:00", id="uptime-value"),
                                                                ]
                                                            ),
                                                            html.Tr(
                                                                [
                                                                    html.Td("Iterations/sec"),
                                                                    html.Td("0", id="iterations-per-second-value"),
                                                                ]
                                                            ),
                                                            html.Tr(
                                                                [
                                                                    html.Td("Current Loss"),
                                                                    html.Td("0.0000", id="current-loss-value"),
                                                                ]
                                                            ),
                                                            html.Tr(
                                                                [
                                                                    html.Td("Estimated Time Remaining"),
                                                                    html.Td("00:00:00", id="time-remaining-value"),
                                                                ]
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                bordered=True,
                                                hover=True,
                                                size="sm",
                                                className="mb-0",
                                            ),
                                        ],
                                    ),
                                ],
                                width=12,
                            ),
                        ],
                        className="mt-3",
                    ),
                ]
            ),
        ],
        className="mb-4 shadow-sm",
    )

def create_real_time_plots():
    """Create real-time plots component."""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5("Real-Time Metrics", className="mb-0")
            ),
            dbc.CardBody(
                [
                    dcc.Graph(
                        id="real-time-metrics-graph",
                        config={"responsive": True},
                        style={"height": "400px"},
                    ),
                ]
            ),
            dbc.CardFooter(
                html.P(
                    "This plot shows real-time metrics during the simulation. "
                    "The data is updated every second.",
                    className="text-muted mb-0",
                )
            ),
        ],
        className="mb-4 shadow-sm",
    )

def create_log_viewer():
    """Create a log viewer component."""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5("Simulation Log", className="mb-0")
            ),
            dbc.CardBody(
                [
                    dbc.Textarea(
                        id="simulation-log",
                        className="code-block",
                        style={"height": "300px", "fontFamily": "monospace"},
                        readOnly=True,
                    ),
                ]
            ),
            dbc.CardFooter(
                html.Div(
                    [
                        dbc.Button(
                            "Clear Log",
                            id="btn-clear-log",
                            color="secondary",
                            size="sm",
                            className="me-2",
                        ),
                        dbc.Button(
                            "Download Log",
                            id="btn-download-log",
                            color="info",
                            size="sm",
                        ),
                        dbc.FormText(
                            "Real-time simulation log output.",
                            className="ms-3",
                        ),
                    ],
                    className="d-flex align-items-center",
                )
            ),
        ],
        className="mb-4 shadow-sm",
    )

def create_real_time_monitor_panel():
    """Create the real-time monitor panel component."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Real-Time Monitoring", className="mb-4"),
                            html.P(
                                "Monitor the simulation in real-time with system metrics and live updates.",
                                className="text-muted",
                            ),
                        ],
                        width=12,
                    )
                ]
            ),
            
            # System Monitor and Real-Time Plots
            dbc.Row(
                [
                    dbc.Col(
                        [
                            create_system_monitor(),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            create_real_time_plots(),
                        ],
                        width=8,
                    ),
                ]
            ),
            
            # Log Viewer
            dbc.Row(
                [
                    dbc.Col(
                        [
                            create_log_viewer(),
                        ],
                        width=12,
                    ),
                ]
            ),
        ],
        fluid=True,
    )

def create_real_time_metrics_figure(data=None, plot_style="plotly_white"):
    """Create a real-time metrics figure."""
    if data is None:
        # Create sample data for demonstration
        timestamps = [time.time() - i for i in range(60, 0, -1)]
        loss_values = [np.exp(-i/20) for i in range(60)]
        entropy_values = [0.2 + 0.1 * np.sin(i/10) for i in range(60)]
        gradient_norm_values = [0.5 * np.exp(-i/30) for i in range(60)]
    else:
        # Use actual data
        timestamps = data.get("timestamps", [])
        loss_values = data.get("loss_values", [])
        entropy_values = data.get("entropy_values", [])
        gradient_norm_values = data.get("gradient_norm_values", [])
    
    # Create the figure with subplots
    fig = make_subplots(
        rows=1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{"secondary_y": True}]],
    )
    
    # Add loss trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=loss_values,
            mode="lines",
            name="Loss",
            line=dict(color="red", width=2),
        ),
        row=1, col=1,
        secondary_y=False,
    )
    
    # Add entropy trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=entropy_values,
            mode="lines",
            name="Entropy",
            line=dict(color="blue", width=2),
        ),
        row=1, col=1,
        secondary_y=True,
    )
    
    # Add gradient norm trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=gradient_norm_values,
            mode="lines",
            name="Gradient Norm",
            line=dict(color="green", width=2),
        ),
        row=1, col=1,
        secondary_y=False,
    )
    
    # Update layout
    fig.update_layout(
        title="Real-Time Simulation Metrics",
        template=plot_style,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=60, t=60, b=60),
        hovermode="closest",
    )
    
    # Update axes
    fig.update_xaxes(
        title_text="Time",
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0, 0, 0, 0.1)",
    )
    
    fig.update_yaxes(
        title_text="Loss / Gradient Norm",
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(0, 0, 0, 0.1)",
        secondary_y=False,
    )
    
    fig.update_yaxes(
        title_text="Entropy",
        row=1, col=1,
        showgrid=False,
        secondary_y=True,
    )
    
    return fig
