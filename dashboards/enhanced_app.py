#!/usr/bin/env python3
"""
EntropicUnification Dashboard (Enhanced Version)

This is an enhanced version of the dashboard with improved UI, themes,
interactive plots, and help tooltips.
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
from dash import dcc, html, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from datetime import datetime
from pathlib import Path

# Import dashboard components
try:
    from components.control_panel import create_control_panel
    from components.results_panel import create_results_panel
    from components.explanation_panel import create_explanation_panel
    from components.settings_panel import create_settings_panel
    from components.help_tooltips import get_help_tooltip, HELP_TOOLTIPS
    from components.interactive_plots import (
        create_plot_container,
        create_enhanced_loss_curves,
        create_enhanced_entropy_area,
        create_enhanced_entropy_components,
        create_enhanced_metric_evolution,
        fig_to_uri
    )
except ImportError:
    # Try relative imports if the above fails
    from dashboards.components.control_panel import create_control_panel
    from dashboards.components.results_panel import create_results_panel
    from dashboards.components.explanation_panel import create_explanation_panel
    from dashboards.components.settings_panel import create_settings_panel
    from dashboards.components.help_tooltips import get_help_tooltip, HELP_TOOLTIPS
    from dashboards.components.interactive_plots import (
        create_plot_container,
        create_enhanced_loss_curves,
        create_enhanced_entropy_area,
        create_enhanced_entropy_components,
        create_enhanced_metric_evolution,
        fig_to_uri
    )

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
        # Theme CSS (for dynamic theme changes)
        html.Div(id="theme-css", style={"display": "none"}),
        
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
                    className="text-center my-4 dashboard-header",
                ),
                width=12,
            )
        ),
        
        # Settings Panel
        create_settings_panel(),
        
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
        
        # Download component for plot downloads
        dcc.Download(id="download-data"),
        
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

# Callback to start a simulation
@app.callback(
    Output("simulation-config", "data"),
    Output("simulation-status", "data", allow_duplicate=True),
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

# Callback to load simulation results
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
                    "areas": np.linspace(1, 10, 20).tolist(),
                    "entropies": (0.25 * np.linspace(1, 10, 20) + 0.1 * np.random.randn(20)).tolist(),
                    "area_law_coefficient": 0.253,
                    "intercept": 0.021,
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
                    "areas": np.linspace(1, 10, 20).tolist(),
                    "entropies": (0.31 * np.linspace(1, 10, 20) + 0.15 * np.random.randn(20)).tolist(),
                    "area_law_coefficient": 0.312,
                    "intercept": 0.045,
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
                    "areas": np.linspace(1, 10, 20).tolist(),
                    "entropies": (0.19 * np.linspace(1, 10, 20) + 0.2 * np.random.randn(20)).tolist(),
                    "area_law_coefficient": 0.195,
                    "intercept": 0.078,
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

# Callback to update the results plots with enhanced versions
@app.callback(
    Output("graph-loss-curves", "figure"),
    Output("graph-entropy-area", "figure"),
    Output("graph-entropy-components", "figure"),
    Output("graph-metric-evolution", "figure"),
    Input("simulation-results", "data"),
    Input("plot-style-dropdown", "value"),
    Input("interactive-plots-switch", "value"),
)
def update_results_plots(results, plot_style, interactive):
    """Update the results plots based on loaded data."""
    # Set default plot style if not provided
    if not plot_style:
        plot_style = "plotly_white"
    
    # Create empty figures if no results
    if results is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data loaded",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=plot_style,
        )
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    # Create enhanced plots
    loss_fig = create_enhanced_loss_curves(results, plot_style)
    entropy_area_fig = create_enhanced_entropy_area(results, plot_style)
    entropy_components_fig = create_enhanced_entropy_components(results, plot_style)
    metric_evolution_fig = create_enhanced_metric_evolution(results, plot_style)
    
    # Set interactivity
    config = {
        "displayModeBar": interactive,
        "responsive": True,
    }
    
    for fig in [loss_fig, entropy_area_fig, entropy_components_fig, metric_evolution_fig]:
        fig.update_layout(hovermode="closest" if interactive else False)
    
    return loss_fig, entropy_area_fig, entropy_components_fig, metric_evolution_fig

# Callback for plot downloads
@app.callback(
    Output("download-data", "data"),
    [
        Input("btn-download-loss", "n_clicks"),
        Input("btn-download-entropy-area", "n_clicks"),
        Input("btn-download-entropy-components", "n_clicks"),
        Input("btn-download-metric", "n_clicks"),
    ],
    [
        State("graph-loss-curves", "figure"),
        State("graph-entropy-area", "figure"),
        State("graph-entropy-components", "figure"),
        State("graph-metric-evolution", "figure"),
        State("download-format-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def download_plot(n1, n2, n3, n4, loss_fig, entropy_area_fig, entropy_components_fig, metric_fig, format_type):
    """Download a plot based on which button was clicked."""
    triggered_id = ctx.triggered_id if ctx.triggered_id else None
    if not triggered_id:
        return dash.no_update
    
    if not format_type:
        format_type = "png"
    
    if triggered_id == "btn-download-loss":
        return dict(
            content=fig_to_uri(loss_fig, format_type),
            filename=f"loss_curves.{format_type}"
        )
    elif triggered_id == "btn-download-entropy-area":
        return dict(
            content=fig_to_uri(entropy_area_fig, format_type),
            filename=f"entropy_area.{format_type}"
        )
    elif triggered_id == "btn-download-entropy-components":
        return dict(
            content=fig_to_uri(entropy_components_fig, format_type),
            filename=f"entropy_components.{format_type}"
        )
    elif triggered_id == "btn-download-metric":
        return dict(
            content=fig_to_uri(metric_fig, format_type),
            filename=f"metric_evolution.{format_type}"
        )
    
    return dash.no_update

# Callback to toggle settings panel
@app.callback(
    Output("settings-panel", "className"),
    Input("settings-toggle", "n_clicks"),
    State("settings-panel", "className"),
    prevent_initial_call=True,
)
def toggle_settings_panel(n_clicks, current_class):
    """Toggle the settings panel open/closed."""
    if n_clicks is None:
        return current_class
    
    if "open" in current_class:
        return "settings-panel"
    else:
        return "settings-panel open"

# Callback to update theme
@app.callback(
    Output("theme-css", "children"),
    Input("dark-mode-switch", "value"),
    prevent_initial_call=True,
)
def update_theme(dark_mode):
    """Update the theme based on the dark mode switch."""
    if dark_mode:
        return "body { background-color: #2C3E50; color: #ECF0F1; }"
    else:
        return ""

# Callback to update interval
@app.callback(
    Output("interval-update", "interval"),
    Input("refresh-interval-input", "value"),
    Input("auto-refresh-switch", "value"),
    prevent_initial_call=True,
)
def update_interval(interval_seconds, auto_refresh):
    """Update the interval for real-time updates."""
    if not auto_refresh:
        return 1000 * 60 * 60  # 1 hour (effectively disabled)
    
    if interval_seconds is None or interval_seconds < 1:
        interval_seconds = 1
    
    return interval_seconds * 1000  # Convert to milliseconds

# Run the app
if __name__ == "__main__":
    print("\n" + "="*50)
    print("EntropicUnification Dashboard (Enhanced Version)")
    print("="*50)
    print("\nStarting dashboard server...")
    print("Open your web browser and navigate to: http://127.0.0.1:8050/")
    print("\nPress Ctrl+C to stop the server.")
    app.run(debug=True, port=8050)
