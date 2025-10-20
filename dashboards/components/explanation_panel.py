"""
Explanation Panel Component for EntropicUnification Dashboard

This module provides the explanation panel with detailed explanations
of the simulation concepts and results.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

def create_explanation_panel():
    """Create the explanation panel component."""
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H3("Explanations & Insights", className="mb-4"),
                            html.P(
                                "Understand the concepts and results of the EntropicUnification framework.",
                                className="text-muted",
                            ),
                        ],
                        width=12,
                    )
                ]
            ),
            
            # Tabs for different explanations
            dbc.Tabs(
                [
                    # Framework Tab
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4("EntropicUnification Framework", className="mt-4"),
                                            html.P(
                                                [
                                                    "EntropicUnification is an exploratory computational framework that investigates potential connections between ",
                                                    html.Strong("quantum entanglement entropy"),
                                                    " and ",
                                                    html.Strong("spacetime geometry"),
                                                    " — a testbed for exploring how quantum information might relate to gravitational dynamics through differentiable programming."
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    "The framework proposes a speculative connection inspired by several strands of research including holographic entanglement entropy, thermodynamic derivations of gravity, and information geometry:"
                                                ]
                                            ),
                                            dbc.Card(
                                                dbc.CardBody(
                                                    html.P(
                                                        "G_μν ∝ ∇_μ∇_ν S_ent",
                                                        className="text-center fs-4",
                                                    )
                                                ),
                                                className="mb-3",
                                            ),
                                            html.P(
                                                [
                                                    "where G_μν is the Einstein tensor encoding spacetime curvature and S_ent is the entanglement entropy."
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    "This heuristic relationship draws inspiration from Jacobson's thermodynamic derivation of gravity, the Ryu-Takayanagi formula, and Faulkner's work on entanglement and linearized Einstein equations."
                                                ]
                                            ),
                                            html.H5("Key Components", className="mt-4"),
                                            dbc.ListGroup(
                                                [
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Quantum Engine: "),
                                                            "Prepares and evolves quantum states, calculating entanglement entropy."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Geometry Engine: "),
                                                            "Handles spacetime metric operations and calculates curvature tensors."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Entropy Module: "),
                                                            "Computes entanglement entropy and its gradients."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Coupling Layer: "),
                                                            "Connects entropy gradients to spacetime curvature."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Optimizer: "),
                                                            "Finds metrics that minimize inconsistency between entropy and curvature."
                                                        ]
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Framework Overview",
                        tab_id="tab-framework",
                    ),
                    
                    # Quantum Tab
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4("Quantum Concepts", className="mt-4"),
                                            html.P(
                                                [
                                                    "The quantum aspects of EntropicUnification focus on preparing quantum states and calculating their entanglement properties."
                                                ]
                                            ),
                                            
                                            html.H5("Quantum States", className="mt-4"),
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H6("Bell State"),
                                                        html.P(
                                                            [
                                                                "A maximally entangled two-qubit state:",
                                                                html.Br(),
                                                                "|Ψ⟩ = (|00⟩ + |11⟩)/√2"
                                                            ]
                                                        ),
                                                        html.H6("GHZ State"),
                                                        html.P(
                                                            [
                                                                "A multi-qubit entangled state:",
                                                                html.Br(),
                                                                "|Ψ⟩ = (|00...0⟩ + |11...1⟩)/√2"
                                                            ]
                                                        ),
                                                        html.H6("Random State"),
                                                        html.P(
                                                            [
                                                                "A randomly generated quantum state with varying degrees of entanglement."
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                                className="mb-3",
                                            ),
                                            
                                            html.H5("Entanglement Entropy", className="mt-4"),
                                            html.P(
                                                [
                                                    "Entanglement entropy quantifies the amount of quantum entanglement between different parts of a quantum system. For a bipartite system divided into subsystems A and B, the entanglement entropy is:"
                                                ]
                                            ),
                                            dbc.Card(
                                                dbc.CardBody(
                                                    html.P(
                                                        "S_A = -Tr(ρ_A log ρ_A)",
                                                        className="text-center fs-4",
                                                    )
                                                ),
                                                className="mb-3",
                                            ),
                                            html.P(
                                                [
                                                    "where ρ_A is the reduced density matrix of subsystem A, obtained by tracing out subsystem B from the full density matrix."
                                                ]
                                            ),
                                            
                                            html.H5("Edge Modes", className="mt-4"),
                                            html.P(
                                                [
                                                    "Edge modes are gauge degrees of freedom that become physical at entangling surfaces. They contribute additional entropy beyond the standard von Neumann entropy calculation."
                                                ]
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Quantum Concepts",
                        tab_id="tab-quantum",
                    ),
                    
                    # Geometry Tab
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4("Geometric Concepts", className="mt-4"),
                                            html.P(
                                                [
                                                    "The geometric aspects of EntropicUnification focus on spacetime metrics, curvature tensors, and their relationship to entropy."
                                                ]
                                            ),
                                            
                                            html.H5("Spacetime Metric", className="mt-4"),
                                            html.P(
                                                [
                                                    "The metric tensor g_μν defines the geometry of spacetime, determining distances, angles, and causal structure."
                                                ]
                                            ),
                                            
                                            html.H5("Curvature Tensors", className="mt-4"),
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H6("Christoffel Symbols"),
                                                        html.P(
                                                            [
                                                                "Γ^λ_μν = (1/2)g^λσ(∂_μg_νσ + ∂_νg_μσ - ∂_σg_μν)"
                                                            ]
                                                        ),
                                                        html.H6("Riemann Tensor"),
                                                        html.P(
                                                            [
                                                                "R^λ_μνσ = ∂_νΓ^λ_μσ - ∂_σΓ^λ_μν + Γ^λ_νρΓ^ρ_μσ - Γ^λ_σρΓ^ρ_μν"
                                                            ]
                                                        ),
                                                        html.H6("Ricci Tensor"),
                                                        html.P(
                                                            [
                                                                "R_μν = R^λ_μλν"
                                                            ]
                                                        ),
                                                        html.H6("Ricci Scalar"),
                                                        html.P(
                                                            [
                                                                "R = g^μνR_μν"
                                                            ]
                                                        ),
                                                        html.H6("Einstein Tensor"),
                                                        html.P(
                                                            [
                                                                "G_μν = R_μν - (1/2)Rg_μν"
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                                className="mb-3",
                                            ),
                                            
                                            html.H5("Higher Curvature Terms", className="mt-4"),
                                            html.P(
                                                [
                                                    "Extensions to general relativity beyond the Einstein-Hilbert action, such as:"
                                                ]
                                            ),
                                            dbc.ListGroup(
                                                [
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Gauss-Bonnet: "),
                                                            "R² - 4R_μνR^μν + R_μνρσR^μνρσ"
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Weyl Tensor: "),
                                                            "C_μνρσ = R_μνρσ + terms involving Ricci tensor and scalar"
                                                        ]
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Geometric Concepts",
                        tab_id="tab-geometry",
                    ),
                    
                    # Results Interpretation Tab
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4("Interpreting Results", className="mt-4"),
                                            html.P(
                                                [
                                                    "This section explains how to interpret the various plots and metrics produced by the EntropicUnification framework."
                                                ]
                                            ),
                                            
                                            html.H5("Loss Curves", className="mt-4"),
                                            html.P(
                                                [
                                                    "Loss curves show the optimization progress of the entropic field equations:"
                                                ]
                                            ),
                                            dbc.ListGroup(
                                                [
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Total Loss: "),
                                                            "Combined loss from all components."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Einstein Loss: "),
                                                            "Measures consistency between geometry and entropic stress-energy."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Entropy Loss: "),
                                                            "Measures alignment of entropy gradients with target values."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Regularity Loss: "),
                                                            "Penalizes non-physical or irregular metrics."
                                                        ]
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            html.P(
                                                [
                                                    "A decreasing trend indicates convergence toward a consistent solution. Plateaus may indicate local minima."
                                                ]
                                            ),
                                            
                                            html.H5("Entropy vs Area", className="mt-4"),
                                            html.P(
                                                [
                                                    "This plot shows the relationship between entanglement entropy and boundary area:"
                                                ]
                                            ),
                                            dbc.ListGroup(
                                                [
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Area Law Coefficient: "),
                                                            "The proportionality constant between entropy and area. In holographic theories, this should approach 1/4 in natural units."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("R² Value: "),
                                                            "Goodness of fit for the area law. Values close to 1 indicate strong adherence to the area law."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Intercept: "),
                                                            "Non-zero intercept may indicate additional constant contributions to entropy."
                                                        ]
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            
                                            html.H5("Entropy Components", className="mt-4"),
                                            html.P(
                                                [
                                                    "This chart shows the relative contributions to the total entanglement entropy:"
                                                ]
                                            ),
                                            dbc.ListGroup(
                                                [
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Bulk Entropy: "),
                                                            "Standard von Neumann entropy from quantum state bipartition."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("Edge Modes: "),
                                                            "Contribution from gauge degrees of freedom at the entangling surface."
                                                        ]
                                                    ),
                                                    dbc.ListGroupItem(
                                                        [
                                                            html.Strong("UV Correction: "),
                                                            "Regularization effects from the UV cutoff."
                                                        ]
                                                    ),
                                                ],
                                                className="mb-4",
                                            ),
                                            
                                            html.H5("Metric Evolution", className="mt-4"),
                                            html.P(
                                                [
                                                    "These heatmaps show the evolution of the spacetime metric tensor (g_μν) during optimization:"
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    "Color intensity represents the metric component values at each point. Changes in the metric reflect how spacetime geometry responds to entanglement entropy."
                                                ]
                                            ),
                                            html.P(
                                                [
                                                    "For Lorentzian metrics, expect the time-time component (g_00) to be negative, while spatial components are positive."
                                                ]
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Results Interpretation",
                        tab_id="tab-interpretation",
                    ),
                    
                    # Theoretical Background Tab
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4("Theoretical Background", className="mt-4"),
                                            html.P(
                                                [
                                                    "The relationship between quantum information and spacetime geometry represents one of the most profound open questions in theoretical physics. Several key developments motivate this exploration:"
                                                ]
                                            ),
                                            
                                            html.H5("Black Hole Thermodynamics", className="mt-4"),
                                            html.P(
                                                [
                                                    "The Bekenstein-Hawking entropy formula (S = A/4G) suggests that black hole entropy is proportional to horizon area, hinting at a deep connection between information and geometry."
                                                ]
                                            ),
                                            
                                            html.H5("Holographic Principle", className="mt-4"),
                                            html.P(
                                                [
                                                    "The AdS/CFT correspondence demonstrates that gravitational physics in a higher-dimensional space can be encoded in a quantum field theory on its boundary."
                                                ]
                                            ),
                                            
                                            html.H5("Entanglement and Spacetime", className="mt-4"),
                                            html.P(
                                                [
                                                    "Work by Van Raamsdonk, Maldacena, and others suggests that quantum entanglement between boundary regions may be responsible for the connectedness of the bulk spacetime."
                                                ]
                                            ),
                                            
                                            html.H5("Thermodynamic Gravity", className="mt-4"),
                                            html.P(
                                                [
                                                    "Jacobson's derivation of Einstein's equations from thermodynamic principles indicates that spacetime dynamics might emerge from more fundamental information-theoretic considerations."
                                                ]
                                            ),
                                            
                                            html.H5("Disclaimer", className="mt-4"),
                                            dbc.Card(
                                                dbc.CardBody(
                                                    html.P(
                                                        [
                                                            "EntropicUnification is presented as an exploratory computational framework, not a validated physical theory. It serves as a testbed for investigating potential connections between quantum information and spacetime geometry through differentiable programming techniques."
                                                        ]
                                                    )
                                                ),
                                                className="mb-3 border-warning",
                                            ),
                                            html.P(
                                                [
                                                    "The mathematical formulations, particularly the \"entropic stress-energy tensor,\" are heuristic rather than derived from first principles. Known theoretical challenges in relating entanglement entropy to geometry (edge modes, gauge fields, non-conformal matter) are not fully addressed in the current implementation."
                                                ]
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Theoretical Background",
                        tab_id="tab-theory",
                    ),
                ],
                id="explanation-tabs",
                active_tab="tab-framework",
            ),
        ],
        fluid=True,
    )
