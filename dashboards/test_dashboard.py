#!/usr/bin/env python3
"""
Test script for the EntropicUnification Dashboard.

This script checks if all the necessary components are available
and working properly.
"""

import os
import sys
import importlib
from pathlib import Path

def check_import(module_name, package=None):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name, package)
        return True
    except ImportError as e:
        print(f"Error importing {module_name}: {e}")
        return False

def main():
    """Run tests for the dashboard components."""
    print("Testing EntropicUnification Dashboard components...")
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check required packages
    print("\nChecking required packages:")
    required_packages = [
        "dash",
        "dash_bootstrap_components",
        "plotly",
        "numpy",
        "pandas",
    ]
    
    all_packages_available = True
    for package in required_packages:
        available = check_import(package)
        print(f"  {package}: {'✓' if available else '✗'}")
        all_packages_available = all_packages_available and available
    
    if not all_packages_available:
        print("\nSome required packages are missing. Please install them with:")
        print("  pip install dash dash-bootstrap-components plotly numpy pandas")
    
    # Check dashboard components
    print("\nChecking dashboard components:")
    components = [
        ("components.control_panel", None),
        ("components.results_panel", None),
        ("components.explanation_panel", None),
        ("utils.simulation_runner", None),
        ("utils.result_loader", None),
    ]
    
    all_components_available = True
    for module_name, package in components:
        try:
            # Try direct import first
            available = check_import(module_name, package)
            if not available:
                # Try with dashboards prefix
                available = check_import(f"dashboards.{module_name}", None)
            
            print(f"  {module_name}: {'✓' if available else '✗'}")
            all_components_available = all_components_available and available
        except Exception as e:
            print(f"  {module_name}: ✗ (Error: {e})")
            all_components_available = False
    
    # Check assets directory
    print("\nChecking assets directory:")
    assets_dir = Path(__file__).parent / "assets"
    assets_exist = assets_dir.exists()
    print(f"  assets directory: {'✓' if assets_exist else '✗'}")
    
    if assets_exist:
        logo_file = assets_dir / "entropic.jpg"
        logo_exists = logo_file.exists()
        print(f"  logo file: {'✓' if logo_exists else '✗'}")
    else:
        print("  logo file: ✗ (assets directory not found)")
    
    # Check app files
    print("\nChecking app files:")
    app_file = Path(__file__).parent / "app.py"
    app_exists = app_file.exists()
    print(f"  app.py: {'✓' if app_exists else '✗'}")
    
    simple_app_file = Path(__file__).parent / "simple_app.py"
    simple_app_exists = simple_app_file.exists()
    print(f"  simple_app.py: {'✓' if simple_app_exists else '✗'}")
    
    # Summary
    print("\nSummary:")
    all_checks_passed = all_packages_available and all_components_available and assets_exist and logo_exists and app_exists and simple_app_exists
    
    if all_checks_passed:
        print("All checks passed! The dashboard should work properly.")
        print("\nTo run the dashboard, use one of the following commands:")
        print("  python dashboards/app.py")
        print("  python dashboards/simple_app.py")
    else:
        print("Some checks failed. Please fix the issues before running the dashboard.")
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
