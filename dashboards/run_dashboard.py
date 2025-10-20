#!/usr/bin/env python3
"""
Script to run the EntropicUnification Dashboard.

This script checks which dashboard version to run based on the available dependencies.
"""

import os
import sys
import importlib
from pathlib import Path

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def main():
    """Run the appropriate dashboard version."""
    print("\n" + "="*50)
    print("EntropicUnification Dashboard Runner")
    print("="*50)
    
    # Check if the required packages are installed
    required_packages = ["dash", "dash_bootstrap_components", "plotly", "numpy", "pandas"]
    missing_packages = [pkg for pkg in required_packages if not check_import(pkg)]
    
    if missing_packages:
        print("\nSome required packages are missing. Please install them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return 1
    
    # Check if the core modules are available
    core_available = all([
        check_import("core.quantum_engine"),
        check_import("core.geometry_engine"),
        check_import("core.entropy_module"),
        check_import("core.coupling_layer"),
        check_import("core.loss_functions"),
        check_import("core.optimizer")
    ])
    
    # Check if enhanced components are available
    enhanced_available = all([
        os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'components', 'settings_panel.py')),
        os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'components', 'help_tooltips.py')),
        os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'components', 'interactive_plots.py')),
        os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'custom.css')),
        os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'dashboard.js'))
    ])
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='EntropicUnification Dashboard Runner')
    parser.add_argument('--version', choices=['full', 'standalone', 'enhanced', 'auto'], default='auto',
                        help='Dashboard version to run (default: auto)')
    args = parser.parse_args()
    
    # Determine which dashboard to run
    if args.version == 'full':
        if core_available:
            module_name = "app"
        else:
            print("\nWarning: Core modules not found. Cannot run full dashboard.")
            print("Falling back to standalone dashboard.")
            module_name = "standalone_app"
    elif args.version == 'standalone':
        module_name = "standalone_app"
    elif args.version == 'enhanced':
        if enhanced_available:
            module_name = "enhanced_app"
        else:
            print("\nWarning: Enhanced components not found. Cannot run enhanced dashboard.")
            print("Falling back to standalone dashboard.")
            module_name = "standalone_app"
    else:  # auto
        if core_available:
            if enhanced_available:
                print("\nCore modules and enhanced components found. Running enhanced dashboard...")
                module_name = "enhanced_app"
            else:
                print("\nCore modules found. Running full dashboard...")
                module_name = "app"
        else:
            if enhanced_available:
                print("\nEnhanced components found. Running enhanced standalone dashboard...")
                module_name = "enhanced_app"
            else:
                print("\nRunning basic standalone dashboard...")
                module_name = "standalone_app"
    
    # Run the selected dashboard
    try:
        dashboard_module = importlib.import_module(module_name, package="dashboards")
        print(f"\nStarting dashboard from {module_name}.py...")
        print("Open your web browser and navigate to: http://127.0.0.1:8050/")
        print("\nPress Ctrl+C to stop the server.")
        
        # Access the app object from the module and run it
        app = getattr(dashboard_module, "app")
        app.run(debug=True, port=8050)
        
        return 0
    except Exception as e:
        print(f"\nError running dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
