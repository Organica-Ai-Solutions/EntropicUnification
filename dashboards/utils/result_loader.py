"""
Result Loader for EntropicUnification Dashboard

This module provides utilities for loading simulation results
for display in the dashboard.
"""

import os
import sys
import json
import yaml
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ResultLoader:
    """Class for loading EntropicUnification simulation results."""
    
    def __init__(self):
        """Initialize the ResultLoader."""
        pass
    
    def load_results(self, result_dir):
        """
        Load simulation results from a directory.
        
        Args:
            result_dir: Path to the results directory
            
        Returns:
            Dictionary containing the loaded results
        """
        result_dir = Path(result_dir)
        
        # Check if directory exists
        if not result_dir.exists():
            print(f"Results directory does not exist: {result_dir}")
            return None
        
        try:
            # Load configuration
            config_path = result_dir / "config.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
            
            # Load history
            history_path = result_dir / "data" / "history.json"
            if history_path.exists():
                with open(history_path, "r") as f:
                    history = json.load(f)
            else:
                history = {}
            
            # Load analysis
            analysis_path = result_dir / "data" / "analysis.json"
            if analysis_path.exists():
                with open(analysis_path, "r") as f:
                    analysis = json.load(f)
            else:
                analysis = {}
            
            # Load final state
            final_state_path = result_dir / "data" / "final_state.npy"
            if final_state_path.exists():
                final_state = np.load(final_state_path)
            else:
                final_state = None
            
            # Load final metric
            final_metric_path = result_dir / "data" / "final_metric.npy"
            if final_metric_path.exists():
                final_metric = np.load(final_metric_path)
            else:
                final_metric = None
            
            # Load plot metadata
            plot_metadata_path = result_dir / "plot_metadata.json"
            if plot_metadata_path.exists():
                with open(plot_metadata_path, "r") as f:
                    plot_metadata = json.load(f)
            else:
                plot_metadata = {}
            
            # Construct results dictionary
            results = {
                "config": config,
                "history": history,
                "analysis": analysis,
                "final_state": final_state,
                "final_metric": final_metric,
                "plot_metadata": plot_metadata,
                "result_dir": str(result_dir),
            }
            
            return results
            
        except Exception as e:
            print(f"Error loading results: {e}")
            return None
    
    def list_available_results(self, base_dir="results"):
        """
        List available simulation results.
        
        Args:
            base_dir: Base directory to search for results
            
        Returns:
            List of available result directories
        """
        base_dir = Path(base_dir)
        
        # Check if directory exists
        if not base_dir.exists():
            print(f"Base directory does not exist: {base_dir}")
            return []
        
        # Find all result directories
        result_dirs = []
        
        # Check for dashboard results
        dashboard_dir = base_dir / "dashboard"
        if dashboard_dir.exists():
            # Look for state directories
            for state_dir in dashboard_dir.iterdir():
                if state_dir.is_dir():
                    # Look for timestamp directories
                    for timestamp_dir in state_dir.iterdir():
                        if timestamp_dir.is_dir():
                            # Check if this is a valid result directory
                            if (timestamp_dir / "config.yaml").exists() or (timestamp_dir / "data").exists():
                                result_dirs.append({
                                    "path": str(timestamp_dir),
                                    "state": state_dir.name,
                                    "timestamp": timestamp_dir.name,
                                    "label": f"{state_dir.name.capitalize()} ({timestamp_dir.name})",
                                    "source": "dashboard",
                                })
        
        # Check for other results
        for state_dir in base_dir.iterdir():
            if state_dir.is_dir() and state_dir.name != "dashboard":
                # Look for timestamp directories
                for timestamp_dir in state_dir.iterdir():
                    if timestamp_dir.is_dir():
                        # Check if this is a valid result directory
                        if (timestamp_dir / "config.yaml").exists() or (timestamp_dir / "data").exists():
                            result_dirs.append({
                                "path": str(timestamp_dir),
                                "state": state_dir.name,
                                "timestamp": timestamp_dir.name,
                                "label": f"{state_dir.name.capitalize()} ({timestamp_dir.name})",
                                "source": "simulation",
                            })
        
        # Sort by timestamp (newest first)
        result_dirs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return result_dirs
    
    def get_result_summary(self, result_dir):
        """
        Get a summary of simulation results.
        
        Args:
            result_dir: Path to the results directory
            
        Returns:
            Dictionary containing a summary of the results
        """
        results = self.load_results(result_dir)
        if results is None:
            return None
        
        # Extract key metrics
        analysis = results.get("analysis", {})
        history = results.get("history", {})
        config = results.get("config", {})
        
        # Area law metrics
        area_law = analysis.get("area_law", {})
        area_law_coefficient = area_law.get("area_law_coefficient", 0.0)
        r_squared = area_law.get("r_squared", 0.0)
        
        # Entropy components
        entropy_components = analysis.get("entropy_components", {})
        total_entropy = entropy_components.get("total", 0.0)
        
        # Convergence metrics
        convergence = analysis.get("convergence", {})
        final_loss = convergence.get("final_loss", 0.0)
        converged = convergence.get("converged", False)
        
        # Configuration
        quantum_config = config.get("quantum", {})
        num_qubits = quantum_config.get("num_qubits", 0)
        circuit_depth = quantum_config.get("circuit_depth", 0)
        
        coupling_config = config.get("coupling", {})
        stress_form = coupling_config.get("stress_form", "unknown")
        
        # Construct summary
        summary = {
            "area_law_coefficient": area_law_coefficient,
            "r_squared": r_squared,
            "total_entropy": total_entropy,
            "final_loss": final_loss,
            "converged": converged,
            "num_qubits": num_qubits,
            "circuit_depth": circuit_depth,
            "stress_form": stress_form,
        }
        
        return summary
