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
    
    # Determine which dashboard to run
    if core_available:
        print("\nCore modules found. Running full dashboard...")
        module_name = "app"
    else:
        print("\nCore modules not found. Running standalone dashboard...")
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
