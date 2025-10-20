"""
Settings Panel Component for EntropicUnification Dashboard

This module provides the settings panel for customizing the dashboard appearance and behavior.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

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
                    html.H4("Dashboard Settings", className="mb-4"),
                    
                    # Theme Settings
                    html.Div(
                        [
                            html.H5("Theme", className="mb-3"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.P("Dark Mode"),
                                        width=6,
                                    ),
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
                                        width=6,
                                        className="d-flex justify-content-end",
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.P("Color Theme"),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="color-theme-dropdown",
                                            options=[
                                                {"label": "Default (Flatly)", "value": "FLATLY"},
                                                {"label": "Darkly", "value": "DARKLY"},
                                                {"label": "Cyborg", "value": "CYBORG"},
                                                {"label": "Journal", "value": "JOURNAL"},
                                                {"label": "Lumen", "value": "LUMEN"},
                                                {"label": "Superhero", "value": "SUPERHERO"},
                                            ],
                                            value="FLATLY",
                                            clearable=False,
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="mb-3",
                            ),
                        ],
                        className="mb-4",
                    ),
                    
                    # Plot Settings
                    html.Div(
                        [
                            html.H5("Plot Settings", className="mb-3"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.P("Plot Style"),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="plot-style-dropdown",
                                            options=[
                                                {"label": "Plotly", "value": "plotly"},
                                                {"label": "Plotly White", "value": "plotly_white"},
                                                {"label": "Plotly Dark", "value": "plotly_dark"},
                                                {"label": "Seaborn", "value": "seaborn"},
                                                {"label": "Ggplot2", "value": "ggplot2"},
                                                {"label": "Simple White", "value": "simple_white"},
                                            ],
                                            value="plotly_white",
                                            clearable=False,
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.P("Download Format"),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dcc.Dropdown(
                                            id="download-format-dropdown",
                                            options=[
                                                {"label": "PNG", "value": "png"},
                                                {"label": "SVG", "value": "svg"},
                                                {"label": "PDF", "value": "pdf"},
                                                {"label": "JPEG", "value": "jpeg"},
                                            ],
                                            value="png",
                                            clearable=False,
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.P("Interactive Plots"),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dbc.Switch(
                                            id="interactive-plots-switch",
                                            value=True,
                                        ),
                                        width=6,
                                        className="d-flex justify-content-end",
                                    ),
                                ],
                                className="mb-3",
                            ),
                        ],
                        className="mb-4",
                    ),
                    
                    # UI Settings
                    html.Div(
                        [
                            html.H5("UI Settings", className="mb-3"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.P("Show Help Tooltips"),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dbc.Switch(
                                            id="help-tooltips-switch",
                                            value=True,
                                        ),
                                        width=6,
                                        className="d-flex justify-content-end",
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.P("Auto-refresh"),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dbc.Switch(
                                            id="auto-refresh-switch",
                                            value=True,
                                        ),
                                        width=6,
                                        className="d-flex justify-content-end",
                                    ),
                                ],
                                className="mb-3",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.P("Refresh Interval (s)"),
                                        width=6,
                                    ),
                                    dbc.Col(
                                        dbc.Input(
                                            id="refresh-interval-input",
                                            type="number",
                                            min=1,
                                            max=60,
                                            step=1,
                                            value=1,
                                        ),
                                        width=6,
                                    ),
                                ],
                                className="mb-3",
                            ),
                        ],
                        className="mb-4",
                    ),
                    
                    # Reset Button
                    dbc.Button(
                        "Reset to Defaults",
                        id="reset-settings-button",
                        color="secondary",
                        className="w-100 mb-3",
                    ),
                ],
                id="settings-panel",
                className="settings-panel",
            ),
        ]
    )
