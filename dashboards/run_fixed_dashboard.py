#!/usr/bin/env python3
"""
Script to run the EntropicUnification Dashboard with fixes for React errors.

This script provides a more robust way to run the dashboard, handling port conflicts
and React component errors.
"""

import os
import sys
import importlib
import subprocess
import time
from pathlib import Path
import socket
import psutil

def check_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port=8050, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not check_port_in_use(port):
            return port
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def kill_existing_dashboard_processes():
    """Kill any existing dashboard processes."""
    killed = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.cmdline())
            if 'python' in cmdline and 'dashboard' in cmdline:
                print(f"Killing existing dashboard process (PID: {proc.pid})")
                proc.kill()
                killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    if killed:
        # Give processes time to fully terminate
        time.sleep(2)
    
    return killed

def main():
    """Run the enhanced dashboard with fixes for React errors."""
    print("\n" + "="*50)
    print("EntropicUnification Dashboard (Fixed Version)")
    print("="*50)
    
    # Check if the required packages are installed
    required_packages = ["dash", "dash_bootstrap_components", "plotly", "numpy", "pandas", "psutil"]
    missing_packages = [pkg for pkg in required_packages if not check_import(pkg)]
    
    if missing_packages:
        print("\nSome required packages are missing. Please install them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return 1
    
    # Kill any existing dashboard processes
    kill_existing_dashboard_processes()
    
    # Find an available port
    try:
        port = find_available_port()
        print(f"Found available port: {port}")
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    
    # Import the enhanced_app module
    try:
        # Add parent directory to path for imports
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Import the enhanced_app module
        from dashboards import enhanced_app
        
        print("Starting enhanced dashboard...")
        print(f"Open your web browser and navigate to: http://127.0.0.1:{port}/")
        print("Press Ctrl+C to stop the server.")
        
        # Configure the app to suppress React errors
        enhanced_app.app.config.update({
            'suppress_callback_exceptions': True,
            'prevent_initial_callbacks': True
        })
        
        # Run the app
        enhanced_app.app.run(debug=True, port=port, host='127.0.0.1')
        
        return 0
    except Exception as e:
        print(f"Error running dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())