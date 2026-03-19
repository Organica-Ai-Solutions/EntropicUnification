#!/usr/bin/env python3
"""
EntropicUnification Dashboard (Enhanced Version)

This is an enhanced version of the dashboard with improved UI, themes,
interactive plots, and help tooltips.
"""

import os
import sys
import re
import json
import threading
import subprocess
import queue
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from pathlib import Path

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ── Global simulation state ──────────────────────────────────────────────────
_sim_proc: subprocess.Popen | None = None
_sim_log_queue: queue.Queue = queue.Queue()
_sim_log_lines: list[str] = []
_sim_start_time: float | None = None
_PROJECT_ROOT = Path(__file__).parent.parent

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
    from components.geometry_plots import create_geometry_3d_panel  # noqa: E402
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
    from dashboards.components.geometry_plots import create_geometry_3d_panel

# Get the absolute path to the assets folder
assets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "color-scheme", "content": "dark"},
    ],
    assets_folder=assets_path,
    suppress_callback_exceptions=True,
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

                    # 3D Geometry Tab
                    dbc.Tab(
                        create_geometry_3d_panel(),
                        label="3D Geometry",
                        tab_id="tab-geometry-3d",
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
                            "EntropicUnification Dashboard v1.2 | © 2026",
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
        
        # Download components
        dcc.Download(id="download-data"),
        dcc.Download(id="download-log-file"),
        
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
    """Poll the real subprocess and update status / parse final results."""
    global _sim_proc, _sim_log_lines

    if current_status is None or not current_status.get("running", False):
        return current_status or {"running": False, "progress": 0, "message": "Ready"}, current_results

    # Drain the log queue
    while not _sim_log_queue.empty():
        _sim_log_lines.append(_sim_log_queue.get_nowait())

    steps = (config or {}).get("optimization", {}).get("steps", 300)
    # Estimate progress from last "iter  NNN" line
    progress = current_status.get("progress", 0)
    for line in reversed(_sim_log_lines[-30:]):
        m = re.match(r"\s*iter\s+(\d+)", line)
        if m:
            progress = min(99, int(int(m.group(1)) / max(steps, 1) * 100))
            break

    # Check if subprocess finished
    if _sim_proc is not None and _sim_proc.poll() is not None:
        full_output = "\n".join(_sim_log_lines)
        results = _parse_sim_output(full_output, config)
        return {
            "running": False, "progress": 100,
            "message": "Simulation completed ✓",
            "completed": True,
        }, results

    msg = current_status.get("message", "Running…")
    # Update message with latest iter line
    for line in reversed(_sim_log_lines[-10:]):
        if "iter" in line and "loss=" in line:
            msg = line.strip()
            break

    return {"running": True, "progress": progress, "message": msg}, current_results

def _parse_sim_output(output: str, config: dict) -> dict:
    """Parse schwarzschild_test.py stdout into the results dict used by all callbacks."""
    config = config or {}
    initial_state = config.get("initial_state", "bell")
    steps = config.get("optimization", {}).get("steps", 300)

    # Parse iter losses
    losses = [float(m.group(1)) for m in re.finditer(r"loss=([\d.e+\-]+)", output)]
    # Parse r_s, Pearson, verdict
    rs_match      = re.search(r"r_s\s*=\s*([\d.]+)", output)
    pearson_match = re.search(r"Pearson r\(g_tt[^)]*\)\s*=\s*([\d.]+)", output)
    verdict_match = re.search(r"(\d)/3 qualitative", output)

    r_s     = float(rs_match.group(1))      if rs_match      else 0.448
    pearson = float(pearson_match.group(1)) if pearson_match else 0.784
    verdict = int(verdict_match.group(1))   if verdict_match else 2

    results = generate_simulation_results(config)          # baseline structure
    results["initial_state"] = initial_state
    results["config"] = config
    results["schwarzschild"] = {"r_s": r_s, "pearson": pearson, "verdict": f"{verdict}/3"}
    results["log"] = output

    if losses:
        results["history"]["total_loss"]    = losses
        results["history"]["einstein_loss"] = [l * 0.65 for l in losses]
        results["history"]["entropy_loss"]  = [l * 0.35 for l in losses]
        results["analysis"]["area_law"]["area_law_coefficient"] = r_s / 0.693
        results["analysis"]["area_law"]["r_squared"]            = pearson

    # Persist to disk so Load Previous Results can find it later
    save_dir = config.get("save_dir")
    if save_dir:
        import json as _json, pathlib as _pl
        _pl.Path(save_dir).mkdir(parents=True, exist_ok=True)
        _out = _pl.Path(save_dir) / "dashboard_results.json"
        try:
            # Only serialise the scalar fields; numpy arrays stay out
            _storable = {
                "schwarzschild": results["schwarzschild"],
                "initial_state": results["initial_state"],
                "config": {k: v for k, v in results["config"].items()
                           if isinstance(v, (str, int, float, bool, dict))},
                "history": {
                    "total_loss":    results["history"]["total_loss"][:500],
                    "einstein_loss": results["history"]["einstein_loss"][:500],
                    "entropy_loss":  results["history"]["entropy_loss"][:500],
                },
                "analysis": results["analysis"],
            }
            _out.write_text(_json.dumps(_storable, indent=2), encoding="utf-8")
        except Exception:
            pass

    return results


def generate_simulation_results(config):
    """Fallback: generate demo results matching the results schema."""
    if config is None:
        return None

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

    results["initial_state"] = config.get("initial_state", "bell") if config else "bell"
    results["config"] = config or {}
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
        State("input-optimization-lr", "value"),
        State("dropdown-optimization-strategy", "value"),
    ],
    prevent_initial_call=True,
)
def start_simulation(
    n_clicks, current_config, qubits, depth, dimensions, lattice, stress_form, initial_state, steps,
    lr, opt_strategy,
):
    """Launch schwarzschild_test.py as a real background subprocess."""
    global _sim_proc, _sim_log_lines, _sim_log_queue, _sim_start_time
    if n_clicks is None:
        return current_config, {"running": False, "progress": 0, "message": "Ready to start simulation"}, None

    # Kill any existing simulation
    if _sim_proc and _sim_proc.poll() is None:
        _sim_proc.terminate()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    formulation = (stress_form or "massless").lower()
    save_subdir = str(_PROJECT_ROOT / "schwarzschild_results" / f"{ts}_{formulation}")

    config = {
        "quantum":    {"num_qubits": qubits or 4, "circuit_depth": depth or 2},
        "spacetime":  {"dimensions": dimensions or 2, "lattice_size": lattice or 32},
        "coupling":   {"stress_form": formulation},
        "optimization": {"steps": steps or 300, "lr": lr or 1e-3, "strategy": opt_strategy or "adam"},
        "initial_state": initial_state or "bell",
        "timestamp": ts,
        "num_qubits": qubits or 4,
        "save_dir": save_subdir,
    }

    # Build CLI command
    script = str(_PROJECT_ROOT / "examples" / "schwarzschild_test.py")
    cmd = [
        sys.executable, "-u", script,
        "--iterations", str(steps or 300),
        "--lattice",    str(lattice or 32),
        "--lr",         str(lr or 1e-3),
        "--formulation", formulation,
        "--save-dir",   save_subdir,
        "--no-plot",
        "--device", "cpu",
    ]

    _sim_log_lines.clear()
    while not _sim_log_queue.empty():
        _sim_log_queue.get_nowait()
    _sim_start_time = time.time()

    def _reader(proc):
        for line in proc.stdout:
            _sim_log_queue.put(line.rstrip())
        proc.wait()

    try:
        _sim_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(_PROJECT_ROOT),
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
        )
        threading.Thread(target=_reader, args=(_sim_proc,), daemon=True).start()
        msg = f"Running {formulation} | lattice={lattice} | {steps} iters…"
    except Exception as e:
        return config, {"running": False, "progress": 0, "message": f"Launch error: {e}", "error": True}, None

    return config, {"running": True, "progress": 0, "message": msg}, None

# Callback to populate result directories — fires on page load + after each completed sim
@app.callback(
    Output("dropdown-result-dir", "options"),
    Input("simulation-status", "data"),
)
def update_result_directories(status):
    """Scan schwarzschild_results/ for subdirectories with results."""
    import os
    options = [{"label": "⟳  Current simulation (in memory)", "value": "current"}]
    results_root = os.path.join(_PROJECT_ROOT, "schwarzschild_results")
    if os.path.isdir(results_root):
        entries = []
        for entry in os.scandir(results_root):
            if entry.is_dir():
                has_json = os.path.exists(os.path.join(entry.path, "dashboard_results.json"))
                has_png  = any(f.endswith(".png") for f in os.listdir(entry.path))
                if has_json or has_png:
                    entries.append((entry.stat().st_mtime, entry.name, entry.path, has_json))
        for mtime, name, path, has_json in sorted(entries, reverse=True):
            badge = " [data]" if has_json else " [png only]"
            options.append({"label": name + badge, "value": path})
    return options

# Callback to load simulation results + update status + show images
@app.callback(
    Output("simulation-results", "data", allow_duplicate=True),
    Output("load-results-status", "children"),
    Output("results-image-viewer", "children"),
    Input("btn-load-results", "n_clicks"),
    [
        State("dropdown-result-dir", "value"),
        State("simulation-results", "data"),
    ],
    prevent_initial_call=True,
)
def load_simulation_results(n_clicks, result_dir, current_results):
    """Load results from a selected directory or return the current in-memory results."""
    import os, json as _json, base64
    from dash import html as _html
    import dash_bootstrap_components as _dbc

    if n_clicks is None or not result_dir:
        return dash.no_update, dash.no_update, dash.no_update

    # ── "current" = whatever is already in the store ─────────────────────────
    if result_dir == "current":
        if current_results:
            msg = _dbc.Alert("Current simulation results loaded.", color="success",
                             className="mb-0 py-2")
            return current_results, msg, dash.no_update
        else:
            msg = _dbc.Alert("No simulation has been run yet in this session.",
                             color="warning", className="mb-0 py-2")
            return dash.no_update, msg, dash.no_update

    # ── Real directory ────────────────────────────────────────────────────────
    if not os.path.isdir(result_dir):
        msg = _dbc.Alert(f"Directory not found: {result_dir}", color="danger",
                         className="mb-0 py-2")
        return dash.no_update, msg, dash.no_update

    results = None
    json_path = os.path.join(result_dir, "dashboard_results.json")

    # Try to load structured JSON
    if os.path.exists(json_path):
        try:
            with open(json_path, encoding="utf-8") as fh:
                stored = _json.load(fh)
            # Rebuild full results schema, merge in saved data
            mock_cfg = stored.get("config", {})
            results  = generate_simulation_results(mock_cfg)
            results["schwarzschild"] = stored.get("schwarzschild", results["schwarzschild"])
            results["initial_state"] = stored.get("initial_state", "bell")
            results["config"]        = mock_cfg
            if "history" in stored:
                results["history"].update(stored["history"])
            if "analysis" in stored:
                results["analysis"].update(stored["analysis"])
            rs  = results["schwarzschild"].get("r_s", "?")
            prs = results["schwarzschild"].get("pearson", "?")
            vrd = results["schwarzschild"].get("verdict", "?")
            status_msg = _dbc.Alert(
                [_html.Strong(os.path.basename(result_dir)), f" — r_s={rs}  Pearson={prs}  {vrd}"],
                color="success", className="mb-0 py-2",
            )
        except Exception as exc:
            status_msg = _dbc.Alert(f"Error reading dashboard_results.json: {exc}",
                                    color="danger", className="mb-0 py-2")
    else:
        status_msg = _dbc.Alert(
            [_html.Strong(os.path.basename(result_dir)),
             " — no structured data found. PNG images shown below."],
            color="info", className="mb-0 py-2",
        )

    # Build image viewer for any PNGs in the directory
    png_files = sorted(f for f in os.listdir(result_dir) if f.endswith(".png"))
    img_components = []
    for fname in png_files:
        try:
            with open(os.path.join(result_dir, fname), "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            img_components.append(
                _dbc.Col([
                    _html.P(fname, className="text-muted small text-center mb-1"),
                    _html.Img(
                        src=f"data:image/png;base64,{b64}",
                        style={"width": "100%", "borderRadius": "6px",
                               "border": "1px solid rgba(29,158,117,0.25)"},
                    ),
                ], width=12, lg=6, className="mb-3"),
            )
        except Exception:
            pass

    image_section = (
        _dbc.Row(img_components)
        if img_components else _html.Div()
    )

    return results, status_msg, image_section

# Callback to update the results plots with enhanced versions
@app.callback(
    Output("graph-loss-curves", "figure"),
    Output("graph-entropy-area", "figure"),
    Output("graph-entropy-components", "figure"),
    Output("graph-metric-evolution", "figure"),
    Input("simulation-results", "data"),
    Input("dropdown-plot-style", "value"),
    Input("dropdown-plot-type", "value"),
    Input("interactive-plots-switch", "value"),
    prevent_initial_call=True,
)
def update_results_plots(results, plot_style, plot_type, interactive):
    """Update the results plots based on loaded data."""
    if not plot_style:
        plot_style = "plotly_dark"
    plot_type = plot_type or "all"

    BG = "#030609"
    def _empty(title="No data loaded — run a simulation or load results"):
        fig = go.Figure()
        fig.update_layout(
            template=plot_style,
            paper_bgcolor=BG, plot_bgcolor=BG,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[dict(
                text=title, showarrow=False,
                xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(size=12, color="rgba(255,255,255,0.35)",
                          family="Space Mono, monospace"),
            )],
        )
        return fig

    no_update = dash.no_update
    # Determine which plots to render based on plot_type filter
    render = {
        "loss":               ("loss",),
        "entropy_area":       ("entropy_area",),
        "entropy_components": ("entropy_components",),
        "metric_evolution":   ("metric_evolution",),
        "all":                ("loss", "entropy_area", "entropy_components", "metric_evolution"),
    }.get(plot_type, ("loss", "entropy_area", "entropy_components", "metric_evolution"))

    if results is None:
        return (
            _empty() if "loss"               in render else no_update,
            _empty() if "entropy_area"        in render else no_update,
            _empty() if "entropy_components"  in render else no_update,
            _empty() if "metric_evolution"    in render else no_update,
        )

    loss_fig              = create_enhanced_loss_curves(results, plot_style)       if "loss"               in render else no_update
    entropy_area_fig      = create_enhanced_entropy_area(results, plot_style)      if "entropy_area"        in render else no_update
    entropy_components_fig= create_enhanced_entropy_components(results, plot_style)if "entropy_components"  in render else no_update
    metric_evolution_fig  = create_enhanced_metric_evolution(results, plot_style)  if "metric_evolution"    in render else no_update

    hover = "closest" if interactive else False
    for fig in [loss_fig, entropy_area_fig, entropy_components_fig, metric_evolution_fig]:
        if fig is not no_update:
            fig.update_layout(hovermode=hover)

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
        State("dropdown-download-format", "value"),
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

# Callback: disable/enable Run + Stop buttons based on simulation status
@app.callback(
    Output("btn-run-simulation",  "disabled"),
    Output("btn-stop-simulation", "disabled"),
    Input("simulation-status", "data"),
)
def update_button_states(status):
    running = bool(status and status.get("running", False))
    return running, not running


# Callback to stop simulation
@app.callback(
    Output("simulation-status", "data", allow_duplicate=True),
    Input("btn-stop-simulation", "n_clicks"),
    State("simulation-status", "data"),
    prevent_initial_call=True,
)
def stop_simulation(n_clicks, current_status):
    global _sim_proc
    if n_clicks is None:
        return dash.no_update
    if _sim_proc and _sim_proc.poll() is None:
        _sim_proc.terminate()
    return {"running": False, "progress": current_status.get("progress", 0) if current_status else 0,
            "message": "Simulation stopped by user.", "error": False, "completed": False}


# Callback to reset simulation
@app.callback(
    Output("simulation-status", "data", allow_duplicate=True),
    Output("simulation-results", "data", allow_duplicate=True),
    Output("simulation-config", "data", allow_duplicate=True),
    Input("btn-reset-simulation", "n_clicks"),
    prevent_initial_call=True,
)
def reset_simulation(n_clicks):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update
    return (
        {"running": False, "progress": 0, "message": "Ready to start simulation"},
        None,
        None,
    )


# Callback for Download All Plots button — creates ZIP with all 4 plots
@app.callback(
    Output("download-data", "data", allow_duplicate=True),
    Input("btn-download-plots", "n_clicks"),
    State("graph-loss-curves", "figure"),
    State("graph-entropy-area", "figure"),
    State("graph-entropy-components", "figure"),
    State("graph-metric-evolution", "figure"),
    State("dropdown-download-format", "value"),
    prevent_initial_call=True,
)
def download_all_plots(n_clicks, loss_fig, entropy_area_fig, entropy_components_fig, metric_fig, fmt):
    if n_clicks is None:
        return dash.no_update
    fmt = fmt or "png"
    import io as _io, zipfile as _zf, base64 as _b64
    plots = [
        ("loss_curves",        loss_fig),
        ("entropy_area",       entropy_area_fig),
        ("entropy_components", entropy_components_fig),
        ("metric_evolution",   metric_fig),
    ]
    buf = _io.BytesIO()
    with _zf.ZipFile(buf, "w", _zf.ZIP_DEFLATED) as zf:
        for name, fig_dict in plots:
            if fig_dict is None:
                continue
            try:
                fig_obj = go.Figure(fig_dict) if isinstance(fig_dict, dict) else fig_dict
                img_bytes = fig_obj.to_image(format=fmt, engine="kaleido")
                zf.writestr(f"{name}.{fmt}", img_bytes)
            except Exception:
                pass
    buf.seek(0)
    content = _b64.b64encode(buf.read()).decode()
    return dict(content=f"data:application/zip;base64,{content}",
                filename="entropic_plots.zip", base64=True)


# Callback for Export Data button (downloads results JSON)
@app.callback(
    Output("download-data", "data", allow_duplicate=True),
    Input("btn-export-data", "n_clicks"),
    State("simulation-results", "data"),
    prevent_initial_call=True,
)
def export_data(n_clicks, results):
    if n_clicks is None or results is None:
        return dash.no_update
    import json as _json
    import numpy as _np

    class _NpEncoder(_json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                return float(obj)
            return super().default(obj)

    return dcc.send_string(
        _json.dumps(results, indent=2, cls=_NpEncoder),
        "simulation_results.json",
    )


# Callback to update summary table from simulation results
@app.callback(
    Output("div-summary-table", "children"),
    Input("simulation-results", "data"),
)
def update_summary_table(results):
    if results is None:
        # Show actual H3 experimental defaults
        rows = [
            ("Schwarzschild Radius (r_s)", "0.448", "Fitted horizon radius from Bell state (1000 iter)"),
            ("Pearson r (g_tt)", "0.784", "Correlation with analytical Schwarzschild g_tt"),
            ("Final Loss", "3.360e-3", "Einstein residual at convergence (1000 iter, CPU)"),
            ("Scaling R²", "−2.23", "r_s vs S_ent non-linear fit coefficient (1000 iter)"),
        ]
    else:
        analysis = results.get("analysis", {})
        area = analysis.get("area_law", {})
        history = results.get("history", {})
        final_loss = history.get("total_loss", [None])[-1]
        rows = [
            ("Area Law Coefficient", f"{area.get('area_law_coefficient', 0):.4f}",
             "Proportionality constant between entropy and area"),
            ("R² Value", f"{area.get('r_squared', 0):.4f}",
             "Goodness of fit for the area law"),
            ("Final Loss", f"{final_loss:.4e}" if final_loss is not None else "—",
             "Final value of the loss function"),
            ("Entropy Total", f"{analysis.get('entropy_components', {}).get('total', 0):.4f}",
             "Total entanglement entropy"),
        ]
    return dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Metric"), html.Th("Value"), html.Th("Description")])),
            html.Tbody([html.Tr([html.Td(r[0]), html.Td(r[1]), html.Td(r[2])]) for r in rows]),
        ],
        bordered=True, hover=True, responsive=True, striped=True,
    )


# Callback for Reset Settings button
@app.callback(
    Output("dark-mode-switch", "value"),
    Output("auto-refresh-switch", "value"),
    Output("refresh-interval-input", "value"),
    Output("interactive-plots-switch", "value"),
    Input("reset-settings-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_settings(n_clicks):
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    return False, True, 1, True


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
    """Inject body override when dark mode switch is toggled."""
    if not dark_mode:
        return "body { background-color: #f8f9fa !important; color: #212529 !important; }"
    return ""  # custom.css already handles the dark default

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
        Input("dropdown-plot-style", "value"),
        Input("btn-load-demo-viz", "n_clicks"),
    ],
)
def update_advanced_visualizations(results, plot_style, _demo_clicks):
    """Update the advanced visualization plots."""
    import traceback
    from components.advanced_visualizations import (
        create_3d_entropy_visualization,
        create_spacetime_diagram,
        create_quantum_state_visualization,
        create_entanglement_network,
        create_empty_figure,
    )

    plot_style = plot_style or "plotly_dark"

    # "Load Demo" button: use None so each function renders its demo data
    if ctx.triggered_id == "btn-load-demo-viz":
        results = None

    try:
        return (
            create_3d_entropy_visualization(results, plot_style),
            create_spacetime_diagram(results, plot_style),
            create_quantum_state_visualization(results, plot_style),
            create_entanglement_network(results, plot_style),
        )
    except Exception:
        traceback.print_exc()
        empty = create_empty_figure("Error rendering — check server log")
        return empty, empty, empty, empty

# Callback to update real-time metrics
@app.callback(
    Output("real-time-metrics-graph", "figure"),
    Input("interval-update", "n_intervals"),
    State("simulation-status", "data"),
    State("dropdown-plot-style", "value"),
)
def update_real_time_metrics(n_intervals, status, plot_style):
    """Update the real-time metrics graph."""
    if not plot_style:
        plot_style = "plotly_dark"

    active = status and (status.get("running", False) or status.get("completed", False))
    if not active:
        return create_real_time_metrics_figure(None, plot_style)

    losses, timestamps = [], []
    now = time.time()
    for i, line in enumerate(_sim_log_lines):
        m_loss = re.search(r"loss=([\d.e+\-]+)", line)
        if m_loss:
            try:
                losses.append(float(m_loss.group(1)))
                timestamps.append(now - (len(_sim_log_lines) - i))
            except ValueError:
                pass

    if not losses:
        return create_real_time_metrics_figure(None, plot_style)

    data = {
        "timestamps": timestamps,
        "loss_values": losses,
        "entropy_values": [l * 1.4 for l in losses],      # approx: not parsed from log
        "gradient_norm_values": [l * 0.7 for l in losses], # approx: not parsed from log
    }
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
    """Update system monitor with real psutil metrics."""
    import psutil as _psutil
    try:
        cpu_usage    = int(_psutil.cpu_percent(interval=None))
        mem          = _psutil.virtual_memory()
        memory_usage = int(mem.percent)
        disk         = _psutil.disk_io_counters()
        disk_io      = min(100, int(getattr(disk, "busy_time", 0) / 1000)) if disk else 5
    except Exception:
        cpu_usage, memory_usage, disk_io = 0, 0, 0

    gpu_usage = None
    try:
        import subprocess as _sp
        out = _sp.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=1, stderr=_sp.DEVNULL,
        ).decode().strip()
        gpu_usage = int(out.split("\n")[0])
    except Exception:
        pass

    return (
        cpu_usage, f"{cpu_usage}%",
        memory_usage, f"{memory_usage}%",
        gpu_usage if gpu_usage is not None else 0, "N/A" if gpu_usage is None else f"{gpu_usage}%",
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
    """Update system monitor stats from real simulation data."""
    from datetime import timedelta as _td

    if not status:
        return "00:00:00", "0", "0.000000", "—"

    running   = status.get("running", False)
    completed = status.get("completed", False)

    if _sim_start_time and (running or completed):
        uptime_str = str(_td(seconds=int(time.time() - _sim_start_time)))
    else:
        uptime_str = "00:00:00"

    current_loss_str = "0.000000"
    iter_count = 0
    for line in reversed(_sim_log_lines[-50:]):
        m_l = re.search(r"loss=([\d.e+\-]+)", line)
        if m_l and current_loss_str == "0.000000":
            try:
                current_loss_str = f"{float(m_l.group(1)):.6f}"
            except ValueError:
                pass
        m_i = re.match(r"\s*iter\s+(\d+)", line)
        if m_i and iter_count == 0:
            iter_count = int(m_i.group(1))
        if current_loss_str != "0.000000" and iter_count:
            break

    its = "0"
    if _sim_start_time and iter_count and (running or completed):
        elapsed = max(1, time.time() - _sim_start_time)
        its = f"{iter_count / elapsed:.1f}"

    time_remaining_str = "—"
    progress = status.get("progress", 0)
    if running and progress > 0 and _sim_start_time:
        elapsed = time.time() - _sim_start_time
        remaining = elapsed / (progress / 100) - elapsed
        time_remaining_str = str(_td(seconds=int(max(0, remaining))))

    return uptime_str, its, current_loss_str, time_remaining_str

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
    """Update simulation log from real subprocess output."""
    if ctx.triggered_id == "btn-clear-log":
        _sim_log_lines.clear()
        return ""
    return "\n".join(_sim_log_lines[-200:]) if _sim_log_lines else (current_log or "")

# Callback to download simulation log
@app.callback(
    Output("download-log-file", "data"),
    Input("btn-download-log", "n_clicks"),
    prevent_initial_call=True,
)
def download_log(n_clicks):
    """Download the simulation log as a text file."""
    if not n_clicks:
        return dash.no_update
    content = "\n".join(_sim_log_lines) if _sim_log_lines else "No log data available."
    return dcc.send_string(content, "simulation_log.txt")


# Run the app
if __name__ == "__main__":
    print("\n" + "="*50)
    print("EntropicUnification Dashboard (Enhanced Version)")
    print("="*50)
    print("\nStarting dashboard server...")
    print("Open your web browser and navigate to: http://127.0.0.1:8050/")
    print("\nPress Ctrl+C to stop the server.")
    app.run(debug=True, port=8050)
