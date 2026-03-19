"""
Settings Panel Component for EntropicUnification Dashboard
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

TEAL  = "#1D9E75"
AMBER = "#FAC775"


def create_settings_panel():
    """Create the settings panel component."""
    return html.Div(
        [
            # Settings Toggle Button
            html.Div(
                html.I(className="fas fa-cog fa-lg"),
                id="settings-toggle",
                className="settings-toggle",
                n_clicks=0,
            ),

            # Settings Panel
            html.Div(
                [
                    html.H4("Dashboard Settings", className="mb-4",
                            style={"color": TEAL, "fontFamily": "Space Mono, monospace"}),

                    # ── Appearance ─────────────────────────────────────────
                    html.Div(
                        [
                            html.H5("Appearance", className="mb-3",
                                    style={"color": AMBER, "fontSize": "0.85rem",
                                           "textTransform": "uppercase", "letterSpacing": "0.08em"}),
                            dbc.Row(
                                [
                                    dbc.Col(html.P("Dark Mode", className="mb-0"), width=7),
                                    dbc.Col(
                                        html.Label(
                                            [
                                                dbc.Checkbox(
                                                    id="dark-mode-switch",
                                                    className="theme-switch-input",
                                                ),
                                                html.Span(className="theme-slider"),
                                            ],
                                            className="theme-switch",
                                        ),
                                        width=5,
                                        className="d-flex justify-content-end",
                                    ),
                                ],
                                className="mb-3 align-items-center",
                            ),
                        ],
                        className="mb-4",
                    ),

                    # ── Plot Settings ───────────────────────────────────────
                    html.Div(
                        [
                            html.H5("Plot Settings", className="mb-3",
                                    style={"color": AMBER, "fontSize": "0.85rem",
                                           "textTransform": "uppercase", "letterSpacing": "0.08em"}),
                            dbc.Row(
                                [
                                    dbc.Col(html.P("Interactive Plots", className="mb-0"), width=7),
                                    dbc.Col(
                                        dbc.Switch(id="interactive-plots-switch", value=True),
                                        width=5,
                                        className="d-flex justify-content-end",
                                    ),
                                ],
                                className="mb-3 align-items-center",
                            ),
                        ],
                        className="mb-4",
                    ),

                    # ── Auto-Refresh ────────────────────────────────────────
                    html.Div(
                        [
                            html.H5("Refresh", className="mb-3",
                                    style={"color": AMBER, "fontSize": "0.85rem",
                                           "textTransform": "uppercase", "letterSpacing": "0.08em"}),
                            dbc.Row(
                                [
                                    dbc.Col(html.P("Auto-refresh", className="mb-0"), width=7),
                                    dbc.Col(
                                        dbc.Switch(id="auto-refresh-switch", value=True),
                                        width=5,
                                        className="d-flex justify-content-end",
                                    ),
                                ],
                                className="mb-3 align-items-center",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(html.P("Interval (s)", className="mb-0"), width=7),
                                    dbc.Col(
                                        dbc.Input(
                                            id="refresh-interval-input",
                                            type="number",
                                            min=1, max=60, step=1, value=1,
                                        ),
                                        width=5,
                                    ),
                                ],
                                className="mb-3 align-items-center",
                            ),
                        ],
                        className="mb-4",
                    ),

                    # ── Reset ───────────────────────────────────────────────
                    dbc.Button(
                        "Reset to Defaults",
                        id="reset-settings-button",
                        color="secondary",
                        outline=True,
                        className="w-100 mb-3",
                    ),
                ],
                id="settings-panel",
                className="settings-panel",
            ),
        ]
    )
