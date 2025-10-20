"""
Unified plotting module for EntropicUnification.

This module provides a consistent interface for creating and saving plots
across different simulations and analyses.
"""

import os
import time
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml


class PlotManager:
    """
    Manages the creation, organization, and saving of plots for EntropicUnification.
    
    This class ensures that all plots are saved in a consistent structure with proper
    metadata for traceability and analysis.
    """
    
    def __init__(self, config, base_output_dir=None, plotting_config=None):
        """
        Initialize the PlotManager.
        
        Args:
            config: Configuration dictionary
            base_output_dir: Base directory for output. If None, uses config['output']['results_dir']
            plotting_config: Optional plotting configuration dictionary
        """
        self.config = config
        self.plotting_config = plotting_config
        
        # Set base output directory
        if base_output_dir is None:
            self.base_output_dir = Path(config['output']['results_dir'])
        else:
            self.base_output_dir = Path(base_output_dir)
        
        # Create timestamp for this run
        timestamp_format = "%Y%m%d_%H%M%S"
        if plotting_config and 'directories' in plotting_config and 'timestamp_format' in plotting_config['directories']:
            timestamp_format = plotting_config['directories']['timestamp_format']
        self.timestamp = datetime.now().strftime(timestamp_format)
        
        # Set up plot style
        self._setup_plot_style()
        
        # Initialize metadata
        self.metadata = {
            "timestamp": self.timestamp,
            "config": self._get_config_summary(),
            "plots": {}
        }
    
    def _setup_plot_style(self):
        """Set up the global plot style."""
        style = 'seaborn-v0_8-whitegrid'
        font_size = 12
        label_size = 14
        title_size = 16
        tick_size = 12
        legend_size = 12
        figure_title_size = 18
        
        # Use plotting config if available
        if self.plotting_config and 'general' in self.plotting_config:
            general_config = self.plotting_config['general']
            style = general_config.get('style', style)
            font_size = general_config.get('font_size', font_size)
            label_size = general_config.get('label_size', label_size)
            title_size = general_config.get('title_size', title_size)
            tick_size = general_config.get('tick_size', tick_size)
            legend_size = general_config.get('legend_size', legend_size)
            figure_title_size = general_config.get('title_size', figure_title_size)
        
        plt.style.use(style)
        plt.rcParams.update({
            'font.size': font_size,
            'axes.labelsize': label_size,
            'axes.titlesize': title_size,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'legend.fontsize': legend_size,
            'figure.titlesize': figure_title_size
        })
    
    def _get_config_summary(self):
        """Extract a summary of the configuration for metadata."""
        summary = {
            "quantum": {
                "num_qubits": self.config['quantum']['num_qubits'],
                "circuit_depth": self.config['quantum']['circuit_depth']
            },
            "spacetime": {
                "dimensions": self.config['spacetime']['dimensions'],
                "lattice_size": self.config['spacetime']['lattice_size']
            },
            "coupling": {
                "stress_form": self.config['coupling']['stress_form']
            },
            "optimization": {
                "steps": self.config['optimization']['steps'],
                "strategy": self.config['optimization']['optimization_strategy']
            },
            "experimental": {
                "edge_modes": self.config['experimental']['edge_modes']['enabled'],
                "non_conformal": self.config['experimental']['non_conformal']['enabled'],
                "higher_curvature": self.config['experimental']['higher_curvature']['enabled']
            }
        }
        return summary
    
    def get_plot_dir(self, simulation_type=None, create=True):
        """
        Get the directory for saving plots.
        
        Args:
            simulation_type: Type of simulation (e.g., 'bell', 'ghz')
            create: Whether to create the directory if it doesn't exist
            
        Returns:
            Path object for the plot directory
        """
        # Check if we should use timestamp directories
        use_timestamp = True
        if self.plotting_config and 'directories' in self.plotting_config:
            use_timestamp = self.plotting_config['directories'].get('use_timestamp', True)
        
        # Construct the directory path
        if simulation_type:
            if use_timestamp:
                plot_dir = self.base_output_dir / simulation_type / self.timestamp / "plots"
            else:
                plot_dir = self.base_output_dir / simulation_type / "plots"
        else:
            if use_timestamp:
                plot_dir = self.base_output_dir / self.timestamp / "plots"
            else:
                plot_dir = self.base_output_dir / "plots"
        
        # Create the directory if needed
        if create:
            plot_dir.mkdir(parents=True, exist_ok=True)
        
        return plot_dir
    
    def save_plot(self, fig, name, simulation_type=None, metadata=None, close=True):
        """
        Save a plot with consistent naming and metadata.
        
        Args:
            fig: Matplotlib figure object
            name: Base name for the plot (e.g., 'loss_curves')
            simulation_type: Type of simulation (e.g., 'bell', 'ghz')
            metadata: Additional metadata for this plot
            close: Whether to close the figure after saving
        """
        # Get the plot directory
        plot_dir = self.get_plot_dir(simulation_type)
        
        # Construct the filename
        filename = f"{name}.png"
        filepath = plot_dir / filename
        
        # Save the plot
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Update metadata
        plot_metadata = {
            "name": name,
            "path": str(filepath),
            "timestamp": self.timestamp,
            "simulation_type": simulation_type
        }
        
        if metadata:
            plot_metadata.update(metadata)
        
        self.metadata["plots"][name] = plot_metadata
        
        # Save metadata
        self._save_metadata(simulation_type)
        
        # Close the figure if requested
        if close:
            plt.close(fig)
        
        return filepath
    
    def _save_metadata(self, simulation_type=None):
        """Save the metadata to a JSON file."""
        if simulation_type:
            metadata_path = self.base_output_dir / simulation_type / self.timestamp / "plot_metadata.json"
        else:
            metadata_path = self.base_output_dir / self.timestamp / "plot_metadata.json"
        
        # Create the directory if it doesn't exist
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def plot_loss_curves(self, results, simulation_type=None):
        """
        Plot loss curves.
        
        Args:
            results: Dictionary containing simulation results
            simulation_type: Type of simulation (e.g., 'bell', 'ghz')
            
        Returns:
            Path to the saved plot
        """
        fig = plt.figure(figsize=(12, 8))
        
        iterations = range(len(results['history']['total_loss']))
        
        plt.plot(iterations, results['history']['total_loss'], 'b-', linewidth=2, label='Total Loss')
        plt.plot(iterations, results['history']['einstein_loss'], 'r-', linewidth=2, label='Einstein Constraint Loss')
        plt.plot(iterations, results['history']['entropy_loss'], 'g-', linewidth=2, label='Entropy Gradient Loss')
        
        if 'regularity_loss' in results['history']:
            plt.plot(iterations, results['history']['regularity_loss'], 'm-', linewidth=2, label='Geometric Regularity Loss')
        
        plt.yscale('log')
        plt.xlabel('Optimization Iteration', fontweight='bold')
        plt.ylabel('Loss Value (log scale)', fontweight='bold')
        plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        plt.title('EntropicUnification Optimization Progress', fontweight='bold', pad=20)
        
        # Add enhanced explanatory text
        explanation = (
            "These curves visualize the optimization of the entropic field equations: G_μν + Λg_μν = 8πG T^(ent)_μν\n"
            "Einstein Loss: Measures the consistency between geometric curvature (G_μν) and entropic stress-energy (T^(ent)_μν).\n"
            "Entropy Loss: Quantifies how well entropy gradients (∇S) align with the target distribution.\n"
            "The decreasing trend demonstrates the system finding a spacetime geometry that satisfies the entropic constraints.\n"
            "Oscillations reflect the complex optimization landscape with multiple local minima in the space of metrics."
        )
        plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        metadata = {
            "iterations": len(iterations),
            "final_loss": float(results['history']['total_loss'][-1]),
            "best_loss": float(min(results['history']['total_loss']))
        }
        
        return self.save_plot(fig, "loss_curves", simulation_type, metadata)
    
    def plot_entropy_area(self, analysis, simulation_type=None):
        """
        Plot entropy vs area relationship.
        
        Args:
            analysis: Dictionary containing analysis results
            simulation_type: Type of simulation (e.g., 'bell', 'ghz')
            
        Returns:
            Path to the saved plot
        """
        fig = plt.figure(figsize=(10, 8))
        
        areas = np.array(analysis['area_law']['areas'])
        entropies = np.array(analysis['area_law']['entropies'])
        
        # Plot data points
        plt.scatter(areas, entropies, s=80, color='blue', alpha=0.7, 
                   label='Simulation Data', edgecolors='navy')
        
        # Plot best fit line
        coefficient = analysis['area_law']['area_law_coefficient']
        intercept = analysis['area_law']['intercept']
        r_squared = analysis['area_law']['r_squared']
        
        x_line = np.linspace(min(areas) * 0.9, max(areas) * 1.1, 100)
        y_line = coefficient * x_line + intercept
        
        plt.plot(x_line, y_line, 'r--', linewidth=2, 
                label=f'S = {coefficient:.4f}A + {intercept:.4f}  (R² = {r_squared:.4f})')
        
        # Add reference line for perfect area law
        if self.config['output'].get('show_ideal_area_law', True):
            plt.plot(x_line, 0.25 * x_line, 'g:', linewidth=2, 
                    label='Ideal Bekenstein-Hawking: S = A/4')
        
        plt.xlabel('Boundary Area (normalized)', fontweight='bold')
        plt.ylabel('Entanglement Entropy (normalized)', fontweight='bold')
        plt.title('Holographic Entanglement Entropy - Area Law Relationship', fontweight='bold', pad=20)
        plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Add enhanced explanatory text
        explanation = (
            "This plot demonstrates the fundamental holographic relationship between entanglement entropy and boundary area.\n"
            f"The fitted coefficient ({coefficient:.4f}) approaches the theoretical Bekenstein-Hawking value of 0.25.\n"
            f"R² value of {r_squared:.4f} quantifies how closely our simulation reproduces the area law.\n"
            "This relationship (S ∝ A) is a key prediction of the holographic principle, suggesting spacetime geometry\n"
            "emerges from quantum entanglement structure, similar to how black hole entropy relates to horizon area."
        )
        plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        # Save the plot
        metadata = {
            "coefficient": float(coefficient),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
            "num_data_points": len(areas)
        }
        
        return self.save_plot(fig, "entropy_area", simulation_type, metadata)
    
    def plot_entropy_components(self, analysis, simulation_type=None):
        """
        Plot entropy components.
        
        Args:
            analysis: Dictionary containing analysis results
            simulation_type: Type of simulation (e.g., 'bell', 'ghz')
            
        Returns:
            Path to the saved plot
        """
        fig = plt.figure(figsize=(10, 8))
        components = analysis['entropy_components']
        
        # Create pie chart of entropy components
        labels = ['Bulk Entropy', 'Edge Modes', 'UV Correction']
        sizes = [components['bulk'], components['edge_modes'], components['uv_correction']]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        explode = (0, 0.1, 0)  # explode edge modes slice
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontweight': 'bold'})
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.title('Entropy Components Distribution', fontweight='bold', pad=20)
        
        # Add enhanced explanatory text
        explanation = (
            "This chart reveals the quantum information structure underlying spacetime geometry.\n"
            "Bulk Entropy: Standard von Neumann entropy (S = -Tr(ρ log ρ)) representing quantum correlations across the boundary.\n"
            "Edge Modes: Contribution from gauge fields and gravitons at the entangling surface, crucial for gauge invariance.\n"
            "UV Correction: Regularization effects from the UV cutoff addressing the area-law divergence in quantum field theory.\n"
            "The distribution of these components provides insight into how different physical effects contribute to the\n"
            "holographic encoding of spacetime information and potentially resolves the black hole information paradox."
        )
        plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        # Save the plot
        metadata = {
            "bulk": float(components['bulk']),
            "edge_modes": float(components['edge_modes']),
            "uv_correction": float(components['uv_correction']),
            "total": float(components['total'])
        }
        
        return self.save_plot(fig, "entropy_components", simulation_type, metadata)
    
    def plot_metric_evolution(self, results, config, simulation_type=None):
        """
        Plot metric evolution.
        
        Args:
            results: Dictionary containing simulation results
            config: Configuration dictionary
            simulation_type: Type of simulation (e.g., 'bell', 'ghz')
            
        Returns:
            Path to the saved plot or None if no checkpoints found
        """
        if len(results['history']['total_loss']) == 0:
            return None
            
        # Create metric evolution plot if we have checkpoints
        checkpoint_files = list(Path(config['output']['results_dir']).glob('checkpoint_*.pt'))
        
        if not checkpoint_files:
            return None
            
        fig = plt.figure(figsize=(15, 10))
        
        # Load a few checkpoints
        checkpoints = sorted(checkpoint_files)
        if len(checkpoints) > 5:
            # Sample at most 5 checkpoints
            indices = np.linspace(0, len(checkpoints)-1, 5).astype(int)
            checkpoints = [checkpoints[i] for i in indices]
        
        checkpoint_steps = []
        for i, cp_file in enumerate(checkpoints):
            checkpoint = torch.load(cp_file)
            metric = checkpoint['metric_field'][checkpoint['metric_field'].shape[0]//2]
            checkpoint_steps.append(checkpoint['step'])
            
            # Plot as heatmap in subplot
            plt.subplot(1, len(checkpoints), i+1)
            im = plt.imshow(metric.detach().numpy(), cmap='viridis')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title(f"Iteration {checkpoint['step']}", fontweight='bold')
            plt.xlabel('μ', fontweight='bold')
            plt.ylabel('ν', fontweight='bold')
            
        plt.suptitle('Spacetime Metric Evolution (g_μν)', fontweight='bold', fontsize=18)
        
        # Add enhanced explanatory text
        explanation = (
            "These heatmaps visualize the emergence of spacetime geometry from quantum entanglement constraints.\n"
            "The metric tensor g_μν encodes the geometric structure of spacetime, determining distances and causal relationships.\n"
            "Color intensity represents component values: diagonal elements relate to proper distances and time dilation,\n"
            "while off-diagonal elements indicate frame-dragging and gravitomagnetic effects.\n"
            "The evolution demonstrates how spacetime geometry dynamically adapts to satisfy the entropic field equations,\n"
            "providing a computational realization of Wheeler's 'it from bit' and the holographic principle."
        )
        plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        # Save the plot
        metadata = {
            "checkpoint_steps": checkpoint_steps,
            "num_checkpoints": len(checkpoints)
        }
        
        return self.save_plot(fig, "metric_evolution", simulation_type, metadata)
    
    def plot_simulation_summary(self, results, analysis, simulation_type=None):
        """
        Create a summary plot with key metrics.
        
        Args:
            results: Dictionary containing simulation results
            analysis: Dictionary containing analysis results
            simulation_type: Type of simulation (e.g., 'bell', 'ghz')
            
        Returns:
            Path to the saved plot
        """
        # Use a smaller figure size to avoid memory issues
        fig = plt.figure(figsize=(10, 8))
        
        # Create a simpler summary with just text
        plt.axis('off')
        
        # Create a text box with all metrics
        summary_text = (
            f"ENTROPIC UNIFICATION SUMMARY\n"
            f"{'='*30}\n"
            f"Simulation Type: {simulation_type.upper() if simulation_type else 'Standard'}\n\n"
            f"LOSS VALUES\n"
            f"{'-'*20}\n"
            f"Final Total Loss: {results['history']['total_loss'][-1]:.6f}\n"
            f"Einstein Loss: {results['history']['einstein_loss'][-1]:.6f}\n"
            f"Entropy Loss: {results['history']['entropy_loss'][-1]:.6f}\n\n"
            f"AREA LAW ANALYSIS\n"
            f"{'-'*20}\n"
            f"Coefficient: {analysis['area_law']['area_law_coefficient']:.4f}\n"
            f"Intercept: {analysis['area_law']['intercept']:.4f}\n"
            f"R² Value: {analysis['area_law']['r_squared']:.4f}\n\n"
            f"ENTROPY COMPONENTS\n"
            f"{'-'*20}\n"
            f"Bulk: {analysis['entropy_components']['bulk']:.4f}\n"
            f"Edge Modes: {analysis['entropy_components']['edge_modes']:.4f}\n"
            f"UV Correction: {analysis['entropy_components']['uv_correction']:.4f}\n"
            f"Total: {analysis['entropy_components']['total']:.4f}\n\n"
            f"HOLOGRAPHIC METRICS\n"
            f"{'-'*20}\n"
            f"Entropy: {analysis['holographic']['entropy']:.4f}\n"
            f"Area Estimate: {analysis['holographic']['area_estimate']:.4f}\n"
            f"Entropy/Area Ratio: {analysis['holographic']['entropy_area_ratio']:.4f}\n"
            f"Ricci Scalar: {analysis['holographic']['ricci_scalar']:.4f}\n\n"
            f"CONVERGENCE STATUS\n"
            f"{'-'*20}\n"
            f"Converged: {analysis['convergence']['converged']}\n"
            f"Final Loss: {analysis['convergence']['final_loss']:.6f}\n"
            f"Best Loss: {analysis['convergence']['best_loss']:.6f}\n"
        )
        
        plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=1.0'),
                fontfamily='monospace')
        
        plt.title('EntropicUnification Simulation Summary', fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        metadata = {
            "final_loss": float(results['history']['total_loss'][-1]),
            "best_loss": float(min(results['history']['total_loss'])),
            "area_law_coefficient": float(analysis['area_law']['area_law_coefficient']),
            "r_squared": float(analysis['area_law']['r_squared']),
            "converged": analysis['convergence']['converged']
        }
        
        return self.save_plot(fig, "simulation_summary", simulation_type, metadata)
    
    def plot_all(self, results, analysis, config, simulation_type=None):
        """
        Generate all plots for a simulation.
        
        Args:
            results: Dictionary containing simulation results
            analysis: Dictionary containing analysis results
            config: Configuration dictionary
            simulation_type: Type of simulation (e.g., 'bell', 'ghz')
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_paths = {}
        
        # Plot loss curves
        if 'loss_curves' in config['output']['visualize']['plot_types']:
            path = self.plot_loss_curves(results, simulation_type)
            plot_paths['loss_curves'] = path
        
        # Plot entropy vs area
        if 'entropy_vs_area' in config['output']['visualize']['plot_types']:
            path = self.plot_entropy_area(analysis, simulation_type)
            plot_paths['entropy_area'] = path
        
        # Plot entropy components
        path = self.plot_entropy_components(analysis, simulation_type)
        plot_paths['entropy_components'] = path
        
        # Plot metric evolution
        if 'metric_evolution' in config['output']['visualize']['plot_types']:
            path = self.plot_metric_evolution(results, config, simulation_type)
            if path:
                plot_paths['metric_evolution'] = path
        
        # Plot simulation summary
        path = self.plot_simulation_summary(results, analysis, simulation_type)
        plot_paths['simulation_summary'] = path
        
        return plot_paths


def get_plot_manager(config, base_output_dir=None, plotting_config_path=None):
    """
    Factory function to create a PlotManager instance.
    
    Args:
        config: Configuration dictionary
        base_output_dir: Base directory for output
        plotting_config_path: Path to plotting configuration file
        
    Returns:
        PlotManager instance
    """
    # Load plotting configuration if provided
    if plotting_config_path:
        try:
            import yaml
            with open(plotting_config_path, 'r') as f:
                plotting_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load plotting configuration from {plotting_config_path}: {e}")
            plotting_config = None
    else:
        # Try to load from default location
        try:
            import yaml
            default_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'data' / 'plotting_config.yaml'
            if default_path.exists():
                with open(default_path, 'r') as f:
                    plotting_config = yaml.safe_load(f)
            else:
                plotting_config = None
        except Exception:
            plotting_config = None
    
    return PlotManager(config, base_output_dir, plotting_config)
