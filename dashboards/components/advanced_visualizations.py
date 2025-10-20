"""
Advanced visualization components for the EntropicUnification Dashboard.

This module provides advanced visualizations for quantum states, entropy distributions,
spacetime diagrams, and entanglement networks.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc

def create_advanced_visualizations_panel():
    """Create the advanced visualizations panel."""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Advanced Visualizations"),
                            html.P(
                                "These visualizations provide deeper insights into the quantum-geometric relationship.",
                                className="lead",
                            ),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("3D Entropy Distribution"),
                            html.P("Spatial distribution of entanglement entropy."),
                            dcc.Graph(
                                id="graph-3d-entropy",
                                figure=create_empty_figure("No simulation data loaded"),
                                config={"responsive": True},
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                    dbc.Col(
                        [
                            html.H4("Spacetime Diagram"),
                            html.P("Visualization of spacetime geometry."),
                            dcc.Graph(
                                id="graph-spacetime-diagram",
                                figure=create_empty_figure("No simulation data loaded"),
                                config={"responsive": True},
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Quantum State Visualization"),
                            html.P("Visualization of the quantum state."),
                            dcc.Graph(
                                id="graph-quantum-state",
                                figure=create_empty_figure("No simulation data loaded"),
                                config={"responsive": True},
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                    dbc.Col(
                        [
                            html.H4("Entanglement Network"),
                            html.P("Network visualization of entanglement between qubits."),
                            dcc.Graph(
                                id="graph-entanglement-network",
                                figure=create_empty_figure("No simulation data loaded"),
                                config={"responsive": True},
                            ),
                        ],
                        width=12,
                        lg=6,
                    ),
                ],
            ),
        ],
        className="p-4",
    )

def create_empty_figure(message="No data available"):
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        title=message,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[dict(
            text=message,
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5
        )]
    )
    return fig

def create_3d_entropy_visualization(results, plot_style="plotly_white"):
    """Create a 3D visualization of entropy distribution."""
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

def create_spacetime_diagram(results, plot_style="plotly_white"):
    """Create a spacetime diagram visualization."""
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

def create_quantum_state_visualization(results, plot_style="plotly_white"):
    """Create a visualization of the quantum state."""
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

def create_entanglement_network(results, plot_style="plotly_white"):
    """Create a visualization of the entanglement network."""
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