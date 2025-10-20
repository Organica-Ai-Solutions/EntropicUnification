#!/usr/bin/env python3
"""
EntropicUnification Dashboard (Standalone Version)

This is a standalone version of the dashboard that doesn't require
any of the core modules, making it easier to run and test.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
from pathlib import Path

# Import dashboard components
try:
    from components.control_panel import create_control_panel
    from components.results_panel import create_results_panel
    from components.explanation_panel import create_explanation_panel
except ImportError:
    # Try relative imports if the above fails
    from dashboards.components.control_panel import create_control_panel
    from dashboards.components.results_panel import create_results_panel
    from dashboards.components.explanation_panel import create_explanation_panel

# Get the absolute path to the assets folder
assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    assets_folder=assets_path
)
app.title = "EntropicUnification Dashboard"
server = app.server

# Create the app layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.Img(src="assets/entropic.jpg", height="100px", className="mr-3"),
                        html.H1("EntropicUnification Dashboard", className="display-4"),
                        html.P(
                            "Exploring the connections between quantum entanglement and spacetime geometry",
                            className="lead",
                        ),
                    ],
                    className="text-center my-4",
                ),
                width=12,
            )
        ),
        
        # Tabs for different sections
        dbc.Tabs(
            [
                # Control Console Tab
                dbc.Tab(
                    create_control_panel(),
                    label="Control Console",
                    tab_id="tab-control",
                ),
                
                # Results Dashboard Tab
                dbc.Tab(
                    create_results_panel(),
                    label="Results Dashboard",
                    tab_id="tab-results",
                ),
                
                # Explanations Tab
                dbc.Tab(
                    create_explanation_panel(),
                    label="Explanations",
                    tab_id="tab-explanations",
                ),
            ],
            id="tabs",
            active_tab="tab-control",
        ),
        
        # Footer
        dbc.Row(
            dbc.Col(
                html.Footer(
                    [
                        html.Hr(),
                        html.P(
                            "EntropicUnification Dashboard v1.0 | © 2025",
                            className="text-center text-muted",
                        ),
                    ]
                ),
                width=12,
            )
        ),
        
        # Store components for sharing data between callbacks
        dcc.Store(id="simulation-config", storage_type="session"),
        dcc.Store(id="simulation-results", storage_type="session"),
        dcc.Store(id="simulation-status", storage_type="session"),
        
        # Interval for updating real-time data
        dcc.Interval(id="interval-update", interval=1000, n_intervals=0),
    ],
    fluid=True,
    className="p-4",
)

# Callback to update simulation status
@app.callback(
    Output("simulation-status", "data"),
    Input("interval-update", "n_intervals"),
    State("simulation-status", "data"),
)
def update_simulation_status(n_intervals, current_status):
    """Update the simulation status."""
    if current_status is None or "running" not in current_status:
        return {"running": False, "progress": 0, "message": "Ready to start simulation"}
    
    if current_status["running"]:
        # In a real implementation, this would check the actual simulation progress
        # For now, we'll just simulate progress
        progress = min(100, current_status.get("progress", 0) + 2)
        
        if progress >= 100:
            return {
                "running": False,
                "progress": 100,
                "message": "Simulation completed",
                "completed": True,
            }
        
        return {
            "running": True,
            "progress": progress,
            "message": f"Running simulation... {progress}% complete",
        }
    
    return current_status

# Callback to start a simulation
@app.callback(
    Output("simulation-config", "data"),
    Output("simulation-status", "data"),
    Input("btn-run-simulation", "n_clicks"),
    State("simulation-config", "data"),
    State("input-quantum-qubits", "value"),
    State("input-quantum-depth", "value"),
    State("input-spacetime-dimensions", "value"),
    State("input-spacetime-lattice", "value"),
    State("dropdown-stress-form", "value"),
    State("dropdown-initial-state", "value"),
    State("input-optimization-steps", "value"),
    prevent_initial_call=True,
)
def start_simulation(
    n_clicks, current_config, qubits, depth, dimensions, lattice, stress_form, initial_state, steps
):
    """Start a simulation with the specified parameters."""
    if n_clicks is None:
        return current_config, {"running": False, "progress": 0, "message": "Ready to start simulation"}
    
    # Create a new configuration
    config = {
        "quantum": {
            "num_qubits": qubits,
            "circuit_depth": depth,
        },
        "spacetime": {
            "dimensions": dimensions,
            "lattice_size": lattice,
        },
        "coupling": {
            "stress_form": stress_form,
        },
        "optimization": {
            "steps": steps,
        },
        "initial_state": initial_state,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # In a real implementation, this would start the actual simulation
    # For now, we'll just return the config and update the status
    
    return config, {"running": True, "progress": 0, "message": "Starting simulation..."}

# Callback to update the progress bar and status text
@app.callback(
    Output("progress-simulation", "value"),
    Output("progress-simulation", "label"),
    Output("simulation-status-text", "children"),
    Input("simulation-status", "data"),
)
def update_progress_bar(status):
    """Update the progress bar based on simulation status."""
    if status is None or "progress" not in status:
        return 0, "0%", "Ready to start simulation"
    
    progress = status.get("progress", 0)
    message = status.get("message", "")
    
    # Create a status alert with appropriate color
    if status.get("completed", False):
        alert_color = "success"
    elif status.get("error", False):
        alert_color = "danger"
    elif status.get("running", False):
        alert_color = "info"
    else:
        alert_color = "secondary"
    
    status_alert = dbc.Alert(
        message,
        color=alert_color,
        className="mb-0"
    )
    
    return progress, f"{progress}%", status_alert

# Callback to update the available result directories
@app.callback(
    Output("dropdown-result-dir", "options"),
    Input("interval-update", "n_intervals"),
)
def update_result_directories(n_intervals):
    """Update the available result directories."""
    # In a real implementation, this would scan the results directory
    # For now, we'll just return some dummy options
    return [
        {"label": "Bell State (2025-10-20)", "value": "bell_20251020"},
        {"label": "GHZ State (2025-10-19)", "value": "ghz_20251019"},
        {"label": "Random State (2025-10-18)", "value": "random_20251018"},
        {"label": "Current Simulation", "value": "current"},
    ]

# Callback to load simulation results (simplified for demo)
@app.callback(
    Output("simulation-results", "data"),
    Input("btn-load-results", "n_clicks"),
    State("dropdown-result-dir", "value"),
    prevent_initial_call=True,
)
def load_simulation_results(n_clicks, result_dir):
    """Load simulation results (demo version)."""
    if n_clicks is None or result_dir is None:
        return None
    
    # Create dummy results based on the selected directory
    if "bell" in result_dir:
        # Bell state results
        results = {
            "history": {
                "total_loss": [np.exp(-i/20) for i in range(100)],
                "einstein_loss": [np.exp(-i/15) for i in range(100)],
                "entropy_loss": [np.exp(-i/25) for i in range(100)],
            },
            "analysis": {
                "area_law": {
                    "area_law_coefficient": 0.253,
                    "r_squared": 0.987,
                },
                "entropy_components": {
                    "bulk": 0.7,
                    "edge_modes": 0.2,
                    "uv_correction": 0.1,
                    "total": 1.0,
                },
            }
        }
    elif "ghz" in result_dir:
        # GHZ state results
        results = {
            "history": {
                "total_loss": [np.exp(-i/15) for i in range(100)],
                "einstein_loss": [np.exp(-i/10) for i in range(100)],
                "entropy_loss": [np.exp(-i/20) for i in range(100)],
            },
            "analysis": {
                "area_law": {
                    "area_law_coefficient": 0.312,
                    "r_squared": 0.954,
                },
                "entropy_components": {
                    "bulk": 0.6,
                    "edge_modes": 0.3,
                    "uv_correction": 0.1,
                    "total": 1.0,
                },
            }
        }
    else:
        # Random state or current simulation results
        results = {
            "history": {
                "total_loss": [np.exp(-i/25) for i in range(100)],
                "einstein_loss": [np.exp(-i/20) for i in range(100)],
                "entropy_loss": [np.exp(-i/30) for i in range(100)],
            },
            "analysis": {
                "area_law": {
                    "area_law_coefficient": 0.195,
                    "r_squared": 0.921,
                },
                "entropy_components": {
                    "bulk": 0.5,
                    "edge_modes": 0.3,
                    "uv_correction": 0.2,
                    "total": 1.0,
                },
            }
        }
    
    return results

# Callback to update the results plots
@app.callback(
    Output("graph-loss-curves", "figure"),
    Output("graph-entropy-area", "figure"),
    Output("graph-entropy-components", "figure"),
    Output("graph-metric-evolution", "figure"),
    Input("simulation-results", "data"),
    Input("dropdown-plot-style", "value"),
)
def update_results_plots(results, plot_style):
    """Update the results plots based on loaded data."""
    if results is None:
        # Return empty figures
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data loaded",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # In a real implementation, this would create actual plots from the results
    # For now, we'll just create dummy plots
    
    # Loss curves plot
    loss_fig = go.Figure()
    iterations = list(range(100))
    loss_fig.add_trace(go.Scatter(x=iterations, y=[np.exp(-i/20) for i in iterations], name="Total Loss", line=dict(color="blue")))
    loss_fig.add_trace(go.Scatter(x=iterations, y=[np.exp(-i/15) for i in iterations], name="Einstein Loss", line=dict(color="red")))
    loss_fig.add_trace(go.Scatter(x=iterations, y=[np.exp(-i/25) for i in iterations], name="Entropy Loss", line=dict(color="green")))
    
    loss_fig.update_layout(
        title="Loss Curves",
        xaxis_title="Iterations",
        yaxis_title="Loss Value",
        yaxis_type="log",
        template=plot_style or "plotly_white",
    )
    
    # Entropy vs Area plot
    entropy_area_fig = go.Figure()
    areas = np.linspace(1, 10, 20)
    entropies = 0.25 * areas + 0.1 * np.random.randn(20)
    
    entropy_area_fig.add_trace(go.Scatter(x=areas, y=entropies, mode="markers", name="Data Points", marker=dict(color="blue")))
    
    # Add best fit line
    coef = np.polyfit(areas, entropies, 1)
    line_y = coef[0] * areas + coef[1]
    entropy_area_fig.add_trace(go.Scatter(x=areas, y=line_y, mode="lines", name=f"Best Fit: S = {coef[0]:.4f}A + {coef[1]:.4f}", line=dict(color="red", dash="dash")))
    
    # Add ideal line
    entropy_area_fig.add_trace(go.Scatter(x=areas, y=0.25 * areas, mode="lines", name="Ideal: S = A/4", line=dict(color="green", dash="dot")))
    
    entropy_area_fig.update_layout(
        title="Entropy vs Area",
        xaxis_title="Boundary Area",
        yaxis_title="Entanglement Entropy",
        template=plot_style or "plotly_white",
    )
    
    # Entropy components plot
    entropy_components_fig = go.Figure(data=[
        go.Pie(
            labels=["Bulk Entropy", "Edge Modes", "UV Correction"],
            values=[0.7, 0.2, 0.1],
            hole=0.3,
            marker=dict(colors=["#3498db", "#e74c3c", "#2ecc71"]),
        )
    ])
    
    entropy_components_fig.update_layout(
        title="Entropy Components",
        template=plot_style or "plotly_white",
    )
    
    # Metric evolution plot
    metric_evolution_fig = make_subplots(rows=1, cols=3, subplot_titles=["Initial", "Middle", "Final"])
    
    # Generate some dummy metric data
    def generate_metric(t):
        x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        return np.exp(-t * (x**2 + y**2))
    
    metrics = [generate_metric(t) for t in [0.5, 1.0, 2.0]]
    
    for i, metric in enumerate(metrics):
        metric_evolution_fig.add_trace(
            go.Heatmap(z=metric, colorscale="Viridis"),
            row=1, col=i+1
        )
    
    metric_evolution_fig.update_layout(
        title="Metric Evolution",
        template=plot_style or "plotly_white",
    )
    
    return loss_fig, entropy_area_fig, entropy_components_fig, metric_evolution_fig

# Run the app
if __name__ == "__main__":
    print("\n" + "="*50)
    print("EntropicUnification Dashboard (Standalone Version)")
    print("="*50)
    print("\nStarting dashboard server...")
    print("Open your web browser and navigate to: http://127.0.0.1:8050/")
    print("\nPress Ctrl+C to stop the server.")
    app.run(debug=True, port=8050)
