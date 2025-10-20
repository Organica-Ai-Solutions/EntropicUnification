#!/usr/bin/env python3
"""
Test script to verify that all dashboard components can be imported correctly.
"""

import os
import sys
import importlib

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test importing all dashboard components."""
    try:
        print("Testing imports...")
        
        # Try importing components directly
        from dashboards.components.control_panel import create_control_panel
        print("✓ control_panel")
        
        from dashboards.components.results_panel import create_results_panel
        print("✓ results_panel")
        
        from dashboards.components.explanation_panel import create_explanation_panel
        print("✓ explanation_panel")
        
        from dashboards.components.settings_panel import create_settings_panel
        print("✓ settings_panel")
        
        from dashboards.components.help_tooltips import get_help_tooltip, HELP_TOOLTIPS
        print("✓ help_tooltips")
        
        from dashboards.components.advanced_visualizations import (
            create_advanced_visualizations_panel,
            create_3d_entropy_visualization,
            create_spacetime_diagram,
            create_quantum_state_visualization,
            create_entanglement_network
        )
        print("✓ advanced_visualizations")
        
        from dashboards.components.real_time_monitor import (
            create_real_time_monitor_panel,
            create_real_time_metrics_figure,
            create_system_monitor,
            create_real_time_plots,
            create_log_viewer
        )
        print("✓ real_time_monitor")
        
        from dashboards.components.interactive_plots import (
            create_plot_container,
            create_enhanced_loss_curves,
            create_enhanced_entropy_area,
            create_enhanced_entropy_components,
            create_enhanced_metric_evolution,
            fig_to_uri
        )
        print("✓ interactive_plots")
        
        # Try importing from enhanced_app
        from dashboards import enhanced_app
        print("✓ enhanced_app")
        
        # Check if tabs are defined in enhanced_app
        tabs = [tab['tab_id'] for tab in enhanced_app.app.layout.children[2].children]
        print(f"Tabs defined in enhanced_app: {tabs}")
        
        print("\nAll imports successful!")
        return True
    except Exception as e:
        print(f"\nError during imports: {e}")
        return False

if __name__ == "__main__":
    test_imports()
