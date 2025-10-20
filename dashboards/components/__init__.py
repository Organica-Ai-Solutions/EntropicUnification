"""
Dashboard components for EntropicUnification Framework.
"""

from .control_panel import create_control_panel
from .results_panel import create_results_panel
from .explanation_panel import create_explanation_panel
from .settings_panel import create_settings_panel
from .help_tooltips import get_help_tooltip, HELP_TOOLTIPS
from .advanced_visualizations import (
    create_advanced_visualizations_panel,
    create_3d_entropy_visualization,
    create_spacetime_diagram,
    create_quantum_state_visualization,
    create_entanglement_network
)
from .real_time_monitor import (
    create_real_time_monitor_panel,
    create_real_time_metrics_figure,
    create_system_monitor,
    create_real_time_plots,
    create_log_viewer
)
from .interactive_plots import (
    create_plot_container,
    create_enhanced_loss_curves,
    create_enhanced_entropy_area,
    create_enhanced_entropy_components,
    create_enhanced_metric_evolution,
    fig_to_uri
)
