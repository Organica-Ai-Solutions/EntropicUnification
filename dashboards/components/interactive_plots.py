"""
Interactive Plots Component for EntropicUnification Dashboard

This module provides enhanced interactive plots with download options.
"""

import base64
import io
import json
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_plot_container(graph_id, title=None, download_id=None, height="400px"):
    """
    Create a container for a plot with download button.
    
    Args:
        graph_id: ID for the graph component
        title: Optional title for the plot
        download_id: ID for the download button
        height: Height of the plot
        
    Returns:
        HTML div containing the plot and download button
    """
    components = []
    
    # Add title if provided
    if title:
        components.append(html.H5(title, className="card-title"))
    
    # Add download button if ID provided
    if download_id:
        components.append(
            html.Button(
                html.I(className="fas fa-download"),
                id=download_id,
                className="plot-download-btn",
                title="Download Plot",
            )
        )
    
    # Add graph component
    components.append(
        dcc.Graph(
            id=graph_id,
            config={
                "responsive": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"{graph_id}",
                    "height": 800,
                    "width": 1200,
                    "scale": 2
                }
            },
            style={"height": height},
        )
    )
    
    return html.Div(
        components,
        className="plot-container position-relative mb-4",
    )

def create_enhanced_loss_curves(data, plot_style="plotly_white"):
    """
    Create enhanced loss curves plot.
    
    Args:
        data: Dictionary containing loss history data
        plot_style: Plotly template to use
        
    Returns:
        Plotly figure object
    """
    if not data or "history" not in data:
        # Return empty figure
        return go.Figure().update_layout(
            title="No loss data available",
            template=plot_style,
        )
    
    history = data["history"]
    iterations = list(range(len(history["total_loss"])))
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=history["total_loss"],
            name="Total Loss",
            line=dict(color="#3498db", width=3),
            hovertemplate="Iteration: %{x}<br>Total Loss: %{y:.6f}<extra></extra>",
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=history["einstein_loss"],
            name="Einstein Loss",
            line=dict(color="#e74c3c", width=2, dash="solid"),
            hovertemplate="Iteration: %{x}<br>Einstein Loss: %{y:.6f}<extra></extra>",
        ),
        secondary_y=True,
    )
    
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=history["entropy_loss"],
            name="Entropy Loss",
            line=dict(color="#2ecc71", width=2, dash="solid"),
            hovertemplate="Iteration: %{x}<br>Entropy Loss: %{y:.6f}<extra></extra>",
        ),
        secondary_y=True,
    )
    
    if "regularity_loss" in history:
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=history["regularity_loss"],
                name="Regularity Loss",
                line=dict(color="#9b59b6", width=2, dash="dot"),
                hovertemplate="Iteration: %{x}<br>Regularity Loss: %{y:.6f}<extra></extra>",
            ),
            secondary_y=True,
        )
    
    # Add moving average of total loss
    window_size = max(1, len(iterations) // 10)
    moving_avg = [sum(history["total_loss"][max(0, i-window_size):i+1]) / 
                 min(i+1, window_size+1) for i in range(len(iterations))]
    
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=moving_avg,
            name=f"Moving Avg ({window_size} steps)",
            line=dict(color="#f39c12", width=2, dash="dash"),
            hovertemplate="Iteration: %{x}<br>Moving Avg: %{y:.6f}<extra></extra>",
        ),
        secondary_y=False,
    )
    
    # Add annotations for key points
    min_loss_idx = history["total_loss"].index(min(history["total_loss"]))
    fig.add_annotation(
        x=min_loss_idx,
        y=history["total_loss"][min_loss_idx],
        text="Minimum Loss",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#3498db",
        ax=20,
        ay=-40,
    )
    
    # Update layout
    fig.update_layout(
        title="Loss Curves",
        xaxis_title="Optimization Iteration",
        yaxis_title="Total Loss",
        yaxis2_title="Component Losses",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        hovermode="x unified",
        template=plot_style,
        margin=dict(l=50, r=50, t=80, b=80),
    )
    
    # Update axes
    fig.update_yaxes(
        title_text="Total Loss",
        type="log",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.1)",
    )
    
    fig.update_yaxes(
        title_text="Component Losses",
        type="log",
        secondary_y=True,
        showgrid=False,
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.1)",
    )
    
    # Add explanatory text
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.15,
        text="Loss curves show the optimization progress of the entropic field equations.<br>"
             "Einstein Loss: Measures consistency between geometry and entropic stress-energy.<br>"
             "Entropy Loss: Measures alignment of entropy gradients with target values.<br>"
             "Decreasing trend indicates convergence toward a consistent solution.",
        showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
        borderpad=6,
    )
    
    return fig

def create_enhanced_entropy_area(data, plot_style="plotly_white"):
    """
    Create enhanced entropy vs area plot.
    
    Args:
        data: Dictionary containing analysis data
        plot_style: Plotly template to use
        
    Returns:
        Plotly figure object
    """
    if not data or "analysis" not in data or "area_law" not in data["analysis"]:
        # Return empty figure
        return go.Figure().update_layout(
            title="No entropy-area data available",
            template=plot_style,
        )
    
    area_law = data["analysis"]["area_law"]
    areas = area_law["areas"]
    entropies = area_law["entropies"]
    coefficient = area_law["area_law_coefficient"]
    intercept = area_law["intercept"]
    r_squared = area_law["r_squared"]
    
    fig = go.Figure()
    
    # Add scatter plot for data points
    fig.add_trace(
        go.Scatter(
            x=areas,
            y=entropies,
            mode="markers",
            name="Data Points",
            marker=dict(
                color="#3498db",
                size=10,
                line=dict(color="#2980b9", width=2),
            ),
            hovertemplate="Area: %{x:.4f}<br>Entropy: %{y:.4f}<extra></extra>",
        )
    )
    
    # Add best fit line
    x_line = [min(areas) * 0.9, max(areas) * 1.1]
    y_line = [coefficient * x + intercept for x in x_line]
    
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Best Fit: S = {coefficient:.4f}A + {intercept:.4f}",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            hovertemplate="Area: %{x:.4f}<br>Entropy: %{y:.4f}<extra></extra>",
        )
    )
    
    # Add ideal line (S = A/4)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=[0.25 * x for x in x_line],
            mode="lines",
            name="Ideal: S = A/4",
            line=dict(color="#2ecc71", width=2, dash="dot"),
            hovertemplate="Area: %{x:.4f}<br>Entropy: %{y:.4f}<extra></extra>",
        )
    )
    
    # Add confidence band around best fit line
    # (This is a simplified version, in a real implementation you would use standard error)
    upper_y = [y + 0.05 * max(entropies) for y in y_line]
    lower_y = [y - 0.05 * max(entropies) for y in y_line]
    
    fig.add_trace(
        go.Scatter(
            x=x_line + x_line[::-1],
            y=upper_y + lower_y[::-1],
            fill="toself",
            fillcolor="rgba(231, 76, 60, 0.2)",
            line=dict(color="rgba(231, 76, 60, 0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    
    # Add R² annotation
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text=f"R² = {r_squared:.4f}",
        showarrow=False,
        font=dict(size=14, color="#e74c3c"),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
        borderpad=6,
    )
    
    # Update layout
    fig.update_layout(
        title="Entropy vs Area Relationship",
        xaxis_title="Boundary Area (normalized)",
        yaxis_title="Entanglement Entropy (normalized)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        hovermode="closest",
        template=plot_style,
        margin=dict(l=50, r=50, t=80, b=80),
    )
    
    # Add explanatory text
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.15,
        text="This plot shows the relationship between entanglement entropy and boundary area.<br>"
             f"The fitted coefficient ({coefficient:.4f}) represents the area law proportionality constant.<br>"
             f"R² value of {r_squared:.4f} indicates how well the data follows the area law.<br>"
             "In holographic theories, entropy is expected to be proportional to area (S ∝ A).",
        showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
        borderpad=6,
    )
    
    return fig

def create_enhanced_entropy_components(data, plot_style="plotly_white"):
    """
    Create enhanced entropy components plot.
    
    Args:
        data: Dictionary containing analysis data
        plot_style: Plotly template to use
        
    Returns:
        Plotly figure object
    """
    if not data or "analysis" not in data or "entropy_components" not in data["analysis"]:
        # Return empty figure
        return go.Figure().update_layout(
            title="No entropy components data available",
            template=plot_style,
        )
    
    components = data["analysis"]["entropy_components"]
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, 
        cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=["Entropy Components Distribution", "Entropy Components Values"],
    )
    
    # Add pie chart
    labels = ["Bulk Entropy", "Edge Modes", "UV Correction"]
    values = [components["bulk"], components["edge_modes"], components["uv_correction"]]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo="label+percent",
            textposition="inside",
            insidetextorientation="radial",
            marker=dict(colors=colors),
            hovertemplate="%{label}: %{value:.4f} (%{percent})<extra></extra>",
        ),
        row=1, col=1,
    )
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=labels + ["Total"],
            y=values + [components["total"]],
            marker=dict(
                color=colors + ["#f39c12"],
                line=dict(color="#ffffff", width=1),
            ),
            text=[f"{v:.4f}" for v in values + [components["total"]]],
            textposition="auto",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )
    
    # Update layout
    fig.update_layout(
        title="Entropy Components",
        showlegend=False,
        template=plot_style,
        margin=dict(l=50, r=50, t=80, b=80),
    )
    
    # Update axes
    fig.update_yaxes(title_text="Entropy Value", row=1, col=2)
    
    # Add explanatory text
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.15,
        text="This chart shows the relative contributions to the total entanglement entropy.<br>"
             "Bulk Entropy: Standard von Neumann entropy from quantum state bipartition.<br>"
             "Edge Modes: Contribution from gauge degrees of freedom at the entangling surface.<br>"
             "UV Correction: Regularization effects from the UV cutoff.",
        showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
        borderpad=6,
    )
    
    return fig

def create_enhanced_metric_evolution(data, plot_style="plotly_white"):
    """
    Create enhanced metric evolution plot.
    
    Args:
        data: Dictionary containing simulation data
        plot_style: Plotly template to use
        
    Returns:
        Plotly figure object
    """
    # For demo purposes, we'll create a dummy metric evolution
    # In a real implementation, this would use actual metric data from checkpoints
    
    # Generate some dummy metric data
    def generate_metric(t):
        import numpy as np
        x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        return np.exp(-t * (x**2 + y**2))
    
    metrics = [generate_metric(t) for t in [0.5, 1.0, 2.0]]
    titles = ["Initial", "Middle", "Final"]
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=titles,
    )
    
    # Add heatmaps
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        fig.add_trace(
            go.Heatmap(
                z=metric,
                colorscale="Viridis",
                showscale=i == len(metrics) - 1,  # Only show colorbar for the last plot
                hovertemplate="x: %{x}<br>y: %{y}<br>g_μν: %{z:.4f}<extra></extra>",
            ),
            row=1, col=i + 1,
        )
    
    # Update layout
    fig.update_layout(
        title="Metric Evolution",
        template=plot_style,
        margin=dict(l=50, r=50, t=80, b=80),
    )
    
    # Update axes
    for i in range(1, 4):
        fig.update_xaxes(title_text="μ", row=1, col=i)
        fig.update_yaxes(title_text="ν", row=1, col=i)
    
    # Add explanatory text
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.15,
        text="These heatmaps show the evolution of the spacetime metric tensor (g_μν) during optimization.<br>"
             "Color intensity represents the metric component values at each point.<br>"
             "Changes in the metric reflect how spacetime geometry responds to entanglement entropy.",
        showarrow=False,
        font=dict(size=12),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.2)",
        borderwidth=1,
        borderpad=6,
    )
    
    return fig

def fig_to_uri(fig, format="png"):
    """
    Convert a plotly figure to a URI for download.
    
    Args:
        fig: Plotly figure object
        format: Image format (png, svg, jpeg, pdf)
        
    Returns:
        URI string
    """
    img_bytes = fig.to_image(format=format, engine="kaleido")
    encoding = base64.b64encode(img_bytes).decode()
    return f"data:image/{format};base64,{encoding}"
