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
    from components.advanced_visualizations import (
        create_advanced_visualizations_panel,
        create_3d_entropy_visualization,
        create_spacetime_diagram,
        create_quantum_state_visualization,
        create_entanglement_network
    )
    from components.real_time_monitor import (
        create_real_time_monitor_panel,
        create_real_time_metrics_figure,
        create_system_monitor,
        create_real_time_plots,
        create_log_viewer
    )
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
    from dashboards.components.advanced_visualizations import (
        create_advanced_visualizations_panel,
        create_3d_entropy_visualization,
        create_spacetime_diagram,
        create_quantum_state_visualization,
        create_entanglement_network
    )
    from dashboards.components.real_time_monitor import (
        create_real_time_monitor_panel,
        create_real_time_metrics_figure,
        create_system_monitor,
        create_real_time_plots,
        create_log_viewer
    )
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
    assets_folder=assets_path,
    suppress_callback_exceptions=True  # Prevent callback exceptions during partial loading
)
app.title = "EntropicUnification Dashboard"
server = app.server

# Configure Dash to work better with React
app.config.update({
    'suppress_callback_exceptions': True,
    'prevent_initial_callbacks': True
})

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
                    
                    # Advanced Visualizations Tab
                    dbc.Tab(
                        create_advanced_visualizations_panel(),
                        label="Advanced Visualizations",
                        tab_id="tab-advanced",
                    ),
                    
                    # Real-Time Monitoring Tab
                    dbc.Tab(
                        create_real_time_monitor_panel(),
                        label="Real-Time Monitoring",
                        tab_id="tab-monitoring",
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
    [
        Output("simulation-status", "data"),
        Output("simulation-results", "data", allow_duplicate=True),
    ],
    Input("interval-update", "n_intervals"),
    [
        State("simulation-status", "data"),
        State("simulation-results", "data"),
        State("simulation-config", "data"),
    ],
)
def update_simulation_status(n_intervals, current_status, current_results, config):
    """Update the simulation status and generate results when completed."""
    if current_status is None or "running" not in current_status:
        return {"running": False, "progress": 0, "message": "Ready to start simulation"}, current_results
    
    if current_status["running"]:
        # In a real implementation, this would check the actual simulation progress
        # For now, we'll just simulate progress
        progress = min(100, current_status.get("progress", 0) + 2)
        
        if progress >= 100:
            # Simulation completed, generate results
            results = generate_simulation_results(config)
            
            return {
                "running": False,
                "progress": 100,
                "message": "Simulation completed",
                "completed": True,
            }, results
        
        return {
            "running": True,
            "progress": progress,
            "message": f"Running simulation... {progress}% complete",
        }, current_results
    
    return current_status, current_results

def generate_simulation_results(config):
    """Generate simulation results based on the configuration."""
    if config is None:
        return None
        
    # Get configuration values
    initial_state = config.get("initial_state", "bell")
    
    # Create dummy results based on the selected state
    if initial_state == "bell":
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
    elif initial_state == "ghz":
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
        # Random state or other results
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
    [
        Output("simulation-config", "data"),
        Output("simulation-status", "data", allow_duplicate=True),
        Output("simulation-results", "data", allow_duplicate=True),
    ],
    Input("btn-run-simulation", "n_clicks"),
    [
        State("simulation-config", "data"),
        State("input-quantum-qubits", "value"),
        State("input-quantum-depth", "value"),
        State("input-spacetime-dimensions", "value"),
        State("input-spacetime-lattice", "value"),
        State("dropdown-stress-form", "value"),
        State("dropdown-initial-state", "value"),
        State("input-optimization-steps", "value"),
    ],
    prevent_initial_call=True,
)
def start_simulation(
    n_clicks, current_config, qubits, depth, dimensions, lattice, stress_form, initial_state, steps
):
    """Start a simulation with the specified parameters."""
    if n_clicks is None:
        return current_config, {"running": False, "progress": 0, "message": "Ready to start simulation"}, None
    
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
    # Clear any existing results when starting a new simulation
    
    return config, {"running": True, "progress": 0, "message": "Starting simulation..."}, None

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
    Output("simulation-results", "data", allow_duplicate=True),
    Input("btn-load-results", "n_clicks"),
    [
        State("dropdown-result-dir", "value"),
        State("simulation-config", "data"),
    ],
    prevent_initial_call=True,
)
def load_simulation_results(n_clicks, result_dir, config):
    """Load simulation results (demo version)."""
    if n_clicks is None or result_dir is None:
        return None
    
    # Create a mock config based on the selected directory
    mock_config = {
        "initial_state": "bell" if "bell" in result_dir else "ghz" if "ghz" in result_dir else "random"
    }
    
    # Use the generate_simulation_results function to create results
    return generate_simulation_results(mock_config)

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

# Define simplified versions of the advanced visualization functions
def create_simple_3d_entropy_visualization(results, plot_style="plotly_white"):
    """Create a simplified 3D visualization of entropy distribution."""
    import plotly.graph_objects as go
    import numpy as np
    
    # Create sample data for demonstration
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    
    # Create a function that resembles entropy distribution
    Z = 0.1 * (X**2 + Y**2) * np.exp(-(X**2 + Y**2) / 50)
    
    # Create the 3D surface plot
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    
    fig.update_layout(
        title="3D Entropy Distribution",
        scene=dict(
            xaxis_title="Spatial Dimension X",
            yaxis_title="Spatial Dimension Y",
            zaxis_title="Entropy Density",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
        ),
        template=plot_style,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    
    return fig

def create_simple_spacetime_diagram(results, plot_style="plotly_white"):
    """Create a simplified spacetime diagram visualization."""
    import plotly.graph_objects as go
    import numpy as np
    
    # Create sample data for demonstration
    t = np.linspace(0, 10, 100)
    
    # Create a figure
    fig = go.Figure()
    
    # Add light cone
    fig.add_trace(
        go.Scatter(
            x=t, y=t,
            mode="lines",
            line=dict(color="rgba(255, 0, 0, 0.5)", width=2, dash="dash"),
            name="Future Light Cone",
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=t, y=-t,
            mode="lines",
            line=dict(color="rgba(255, 0, 0, 0.5)", width=2, dash="dash"),
            name="Past Light Cone",
        )
    )
    
    # Add worldlines
    for i in range(-4, 5, 2):
        fig.add_trace(
            go.Scatter(
                x=[i] * len(t), y=t,
                mode="lines",
                line=dict(color="rgba(0, 0, 255, 0.5)", width=1),
                name=f"Worldline x={i}" if i == -4 else "",
                showlegend=i == -4,
            )
        )
    
    # Add geodesic
    fig.add_trace(
        go.Scatter(
            x=np.sin(t), y=t,
            mode="lines",
            line=dict(color="green", width=3),
            name="Geodesic",
        )
    )
    
    fig.update_layout(
        title="Spacetime Diagram",
        xaxis_title="Space",
        yaxis_title="Time",
        template=plot_style,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    return fig

def create_simple_quantum_state_visualization(results, plot_style="plotly_white"):
    """Create a simplified visualization of the quantum state."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Create sample data for demonstration
    states = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    
    # For Bell state
    probabilities = [0.5, 0, 0, 0.5]
    phases = [0, 0, 0, np.pi]
    
    # Create the bar chart for probabilities
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Probabilities", "Phases"))
    
    fig.add_trace(
        go.Bar(
            x=states,
            y=probabilities,
            marker_color='rgb(55, 83, 109)',
            name="Probability"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=states,
            y=phases,
            marker_color='rgb(26, 118, 255)',
            name="Phase"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Quantum State Visualization",
        template=plot_style,
        height=400,
    )
    
    return fig

def create_simple_entanglement_network(results, plot_style="plotly_white"):
    """Create a simplified visualization of the entanglement network."""
    import plotly.graph_objects as go
    import numpy as np
    
    # Create sample data for demonstration
    num_qubits = 4
    
    # Create node positions in a circle
    theta = np.linspace(0, 2*np.pi, num_qubits, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Create edges (connections between nodes)
    edge_x = []
    edge_y = []
    edge_colors = []
    
    # Create a fully connected network with varying entanglement strengths
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            # Add the line between nodes i and j
            edge_x.extend([x[i], x[j], None])
            edge_y.extend([y[i], y[j], None])
            
            # Calculate entanglement strength (just a demo)
            strength = 0.5 + 0.5 * np.sin((i+j)/num_qubits * np.pi)
            edge_colors.extend([strength, strength, strength])
    
    # Create the edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color=edge_colors, colorscale='Viridis'),
        hoverinfo='none',
        mode='lines',
        name='Entanglement'
    )
    
    # Create the node trace
    node_trace = go.Scatter(
        x=x, y=y,
        mode='markers+text',
        text=[f'Q{i}' for i in range(num_qubits)],
        textposition="middle center",
        marker=dict(
            showscale=False,
            color='rgba(255, 0, 0, 0.8)',
            size=20,
            line=dict(width=2, color='DarkSlateGrey')
        ),
        name='Qubits'
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="Quantum Entanglement Network",
        showlegend=True,
        hovermode='closest',
        template=plot_style,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    return fig

# Callback to update advanced visualizations
@app.callback(
    [
        Output("graph-3d-entropy", "figure"),
        Output("graph-spacetime-diagram", "figure"),
        Output("graph-quantum-state", "figure"),
        Output("graph-entanglement-network", "figure"),
    ],
    [
        Input("simulation-results", "data"),
        Input("plot-style-dropdown", "value"),
    ],
)
def update_advanced_visualizations(results, plot_style):
    """Update the advanced visualization plots."""
    # Set default plot style if not provided
    if not plot_style:
        plot_style = "plotly_white"
    
    try:
        # Create the advanced visualizations using the simplified functions
        entropy_3d_fig = create_simple_3d_entropy_visualization(results, plot_style)
        spacetime_fig = create_simple_spacetime_diagram(results, plot_style)
        quantum_state_fig = create_simple_quantum_state_visualization(results, plot_style)
        entanglement_network_fig = create_simple_entanglement_network(results, plot_style)
        
        return entropy_3d_fig, spacetime_fig, quantum_state_fig, entanglement_network_fig
    except Exception as e:
        # If there's an error, return empty figures
        import plotly.graph_objects as go
        
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="No data available",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template=plot_style,
            annotations=[dict(
                text=f"Waiting for simulation data...",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5
            )]
        )
        
        return empty_fig, empty_fig, empty_fig, empty_fig

# Callback to update real-time metrics
@app.callback(
    Output("real-time-metrics-graph", "figure"),
    Input("interval-update", "n_intervals"),
    State("simulation-status", "data"),
    State("plot-style-dropdown", "value"),
)
def update_real_time_metrics(n_intervals, status, plot_style):
    """Update the real-time metrics graph."""
    # Set default plot style if not provided
    if not plot_style:
        plot_style = "plotly_white"
    
    # Check if simulation is running
    if status and status.get("running", False):
        # In a real implementation, this would use actual data
        # For now, we'll just create sample data
        import time
        import numpy as np
        
        timestamps = [time.time() - i for i in range(60, 0, -1)]
        
        # Add some randomness to make it look real-time
        loss_values = [np.exp(-i/20) + 0.05 * np.random.randn() for i in range(60)]
        entropy_values = [0.2 + 0.1 * np.sin(i/10) + 0.02 * np.random.randn() for i in range(60)]
        gradient_norm_values = [0.5 * np.exp(-i/30) + 0.03 * np.random.randn() for i in range(60)]
        
        data = {
            "timestamps": timestamps,
            "loss_values": loss_values,
            "entropy_values": entropy_values,
            "gradient_norm_values": gradient_norm_values,
        }
    else:
        data = None
    
    return create_real_time_metrics_figure(data, plot_style)

# Callback to update system monitor
@app.callback(
    [
        Output("cpu-usage-progress", "value"),
        Output("cpu-usage-progress", "label"),
        Output("memory-usage-progress", "value"),
        Output("memory-usage-progress", "label"),
        Output("gpu-usage-progress", "value"),
        Output("gpu-usage-progress", "label"),
        Output("disk-io-progress", "value"),
        Output("disk-io-progress", "label"),
    ],
    Input("interval-update", "n_intervals"),
    State("simulation-status", "data"),
)
def update_system_monitor(n_intervals, status):
    """Update the system monitor."""
    # Check if simulation is running
    if status and status.get("running", False):
        # In a real implementation, this would use actual system data
        # For now, we'll just create sample data
        import random
        
        cpu_usage = min(100, max(0, 30 + random.randint(-5, 5)))
        memory_usage = min(100, max(0, 45 + random.randint(-3, 3)))
        gpu_usage = min(100, max(0, 70 + random.randint(-7, 7)))
        disk_io = min(100, max(0, 20 + random.randint(-2, 2)))
    else:
        cpu_usage = 10
        memory_usage = 20
        gpu_usage = 5
        disk_io = 2
    
    return (
        cpu_usage, f"{cpu_usage}%",
        memory_usage, f"{memory_usage}%",
        gpu_usage, f"{gpu_usage}%",
        disk_io, f"{disk_io}%",
    )

# Callback to update system monitor stats
@app.callback(
    [
        Output("uptime-value", "children"),
        Output("iterations-per-second-value", "children"),
        Output("current-loss-value", "children"),
        Output("time-remaining-value", "children"),
    ],
    Input("interval-update", "n_intervals"),
    State("simulation-status", "data"),
)
def update_system_monitor_stats(n_intervals, status):
    """Update the system monitor stats."""
    # Check if simulation is running
    if status and status.get("running", False):
        # In a real implementation, this would use actual data
        # For now, we'll just create sample data
        import random
        import time
        from datetime import timedelta
        
        # Calculate uptime (just use n_intervals as seconds for demo)
        uptime = timedelta(seconds=n_intervals)
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        # Calculate iterations per second
        iterations_per_second = 10 + random.randint(0, 5)
        
        # Calculate current loss
        current_loss = 0.1 * np.exp(-n_intervals/100) + 0.01 * random.random()
        current_loss_str = f"{current_loss:.6f}"
        
        # Calculate estimated time remaining
        progress = status.get("progress", 0)
        if progress > 0:
            time_per_percent = n_intervals / progress
            remaining_percent = 100 - progress
            remaining_seconds = int(time_per_percent * remaining_percent)
            time_remaining = timedelta(seconds=remaining_seconds)
            time_remaining_str = str(time_remaining).split('.')[0]  # Remove microseconds
        else:
            time_remaining_str = "Calculating..."
    else:
        uptime_str = "00:00:00"
        iterations_per_second = "0"
        current_loss_str = "0.000000"
        time_remaining_str = "00:00:00"
    
    return uptime_str, str(iterations_per_second), current_loss_str, time_remaining_str

# Callback to update simulation log
@app.callback(
    Output("simulation-log", "value"),
    [
        Input("interval-update", "n_intervals"),
        Input("btn-clear-log", "n_clicks"),
    ],
    [
        State("simulation-log", "value"),
        State("simulation-status", "data"),
    ],
)
def update_simulation_log(n_intervals, clear_clicks, current_log, status):
    """Update the simulation log."""
    # Clear log if button was clicked
    if ctx.triggered_id == "btn-clear-log":
        return ""
    
    # Check if simulation is running
    if status and status.get("running", False):
        # In a real implementation, this would use actual log data
        # For now, we'll just create sample log entries
        import random
        import time
        from datetime import datetime
        
        log_entries = [
            f"[{datetime.now().strftime('%H:%M:%S')}] Iteration {n_intervals}: Loss = {0.1 * np.exp(-n_intervals/100):.6f}",
            f"[{datetime.now().strftime('%H:%M:%S')}] Entropy = {0.2 + 0.1 * np.sin(n_intervals/10):.6f}",
            f"[{datetime.now().strftime('%H:%M:%S')}] Gradient norm = {0.5 * np.exp(-n_intervals/30):.6f}",
        ]
        
        # Add a random log entry occasionally
        if random.random() < 0.2:
            messages = [
                "Optimizing metric tensor components",
                "Recalculating entropy gradient",
                "Updating coupling terms",
                "Checking Bianchi identity constraints",
                "Evaluating area law relationship",
                "Computing edge mode contributions",
                "Applying higher curvature corrections",
            ]
            log_entries.append(f"[{datetime.now().strftime('%H:%M:%S')}] {random.choice(messages)}")
        
        # Append to current log
        new_log = current_log or ""
        for entry in log_entries:
            if random.random() < 0.3:  # Only add some entries to avoid too much text
                new_log += entry + "\n"
        
        # Limit log size
        log_lines = new_log.split("\n")
        if len(log_lines) > 100:
            log_lines = log_lines[-100:]
        
        return "\n".join(log_lines)
    
    return current_log or ""

# Run the app
if __name__ == "__main__":
    print("\n" + "="*50)
    print("EntropicUnification Dashboard (Enhanced Version)")
    print("="*50)
    print("\nStarting dashboard server...")
    print("Open your web browser and navigate to: http://127.0.0.1:8050/")
    print("\nPress Ctrl+C to stop the server.")
    app.run(debug=True, port=8050)
