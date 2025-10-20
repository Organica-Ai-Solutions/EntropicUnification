"""
Help Tooltips Component for EntropicUnification Dashboard

This module provides help tooltips for the dashboard components.
"""

import dash_bootstrap_components as dbc
from dash import html

def create_help_icon(tooltip_id, tooltip_text):
    """
    Create a help icon with a tooltip.
    
    Args:
        tooltip_id: ID for the tooltip
        tooltip_text: Text to display in the tooltip
        
    Returns:
        HTML component with the help icon and tooltip
    """
    return html.Span(
        [
            html.I(
                className="fas fa-question-circle help-icon",
                id=f"help-icon-{tooltip_id}",
            ),
            dbc.Tooltip(
                tooltip_text,
                target=f"help-icon-{tooltip_id}",
                placement="right",
            ),
        ],
        className="ms-1",
    )

# Dictionary of help tooltips for different components
HELP_TOOLTIPS = {
    "quantum-qubits": "Number of qubits in the quantum circuit. More qubits allow for more complex quantum states but increase computational complexity.",
    
    "quantum-depth": "Depth of the quantum circuit. Higher depth allows for more complex operations but may lead to more noise in real quantum hardware.",
    
    "initial-state": "The initial quantum state for the simulation. Bell states are maximally entangled 2-qubit states. GHZ states are multi-qubit entangled states.",
    
    "spacetime-dimensions": "Number of spacetime dimensions. Standard spacetime has 4 dimensions (3 space + 1 time).",
    
    "spacetime-lattice": "Size of the lattice used to discretize spacetime. Larger values give higher resolution but increase computational complexity.",
    
    "stress-form": "Formulation of the entropic stress-energy tensor. Different formulations represent different theoretical approaches to connecting entropy and geometry.",
    
    "optimization-steps": "Number of optimization iterations. More steps may lead to better convergence but take longer to compute.",
    
    "optimization-strategy": "Strategy for optimization. Different strategies may be better suited for different types of problems.",
    
    "experimental-features": "Experimental features that can be enabled or disabled. These represent cutting-edge theoretical concepts that are still being explored.",
    
    "loss-curves": "Shows how the different components of the loss function evolve during optimization. Decreasing trends indicate convergence.",
    
    "entropy-area": "Shows the relationship between entanglement entropy and boundary area. In holographic theories, entropy is expected to be proportional to area.",
    
    "entropy-components": "Shows the relative contributions to the total entanglement entropy from different sources.",
    
    "metric-evolution": "Shows how the spacetime metric evolves during optimization. Changes in the metric reflect how spacetime geometry responds to entanglement entropy.",
    
    "simulation-summary": "Provides a summary of key metrics from the simulation, including area law analysis and convergence status.",
}

def get_help_tooltip(component_id):
    """
    Get a help tooltip for a specific component.
    
    Args:
        component_id: ID of the component
        
    Returns:
        Help tooltip component or None if not found
    """
    if component_id in HELP_TOOLTIPS:
        return create_help_icon(component_id, HELP_TOOLTIPS[component_id])
    return None
