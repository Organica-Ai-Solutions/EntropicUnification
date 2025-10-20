#!/usr/bin/env python3
"""
Simple script to run the EntropicUnification Dashboard.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced_app module
try:
    from dashboards import enhanced_app
    
    print("\n" + "="*50)
    print("EntropicUnification Dashboard (Simple Run)")
    print("="*50)
    
    print("Starting dashboard...")
    print("Open your web browser and navigate to: http://127.0.0.1:8070/")
    print("Press Ctrl+C to stop the server.")
    
    # Configure the app to suppress React errors
    enhanced_app.app.config.update({
        'suppress_callback_exceptions': True,
        'prevent_initial_callbacks': True
    })
    
    # Run the app
    enhanced_app.app.run(debug=False, port=8070, host='127.0.0.1')
    
except Exception as e:
    print(f"Error running dashboard: {e}")
    sys.exit(1)
