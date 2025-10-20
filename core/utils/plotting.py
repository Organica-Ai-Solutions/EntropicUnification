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
        
        # Add explanatory text
        explanation = (
            "Loss curves show the optimization progress of the entropic field equations.\n"
            "Einstein Loss: Measures consistency between geometry and entropic stress-energy.\n"
            "Entropy Loss: Measures alignment of entropy gradients with target values.\n"
            "Decreasing trend indicates convergence toward a consistent solution."
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
        
        # Add explanatory text
        explanation = (
            "This plot shows the relationship between entanglement entropy and boundary area.\n"
            f"The fitted coefficient ({coefficient:.4f}) represents the area law proportionality constant.\n"
            f"R² value of {r_squared:.4f} indicates how well the data follows the area law.\n"
            "In holographic theories, entropy is expected to be proportional to area (S ∝ A)."
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
        
        # Add explanatory text
        explanation = (
            "This chart shows the relative contributions to the total entanglement entropy.\n"
            "Bulk Entropy: Standard von Neumann entropy from quantum state bipartition.\n"
            "Edge Modes: Contribution from gauge degrees of freedom at the entangling surface.\n"
            "UV Correction: Regularization effects from the UV cutoff."
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
        
        # Add explanatory text
        explanation = (
            "These heatmaps show the evolution of the spacetime metric tensor (g_μν) during optimization.\n"
            "Color intensity represents the metric component values at each point.\n"
            "Changes in the metric reflect how spacetime geometry responds to entanglement entropy."
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
        fig = plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        
        # Plot final loss values
        loss_labels = ['Total', 'Einstein', 'Entropy', 'Regularity']
        loss_values = [
            results['history']['total_loss'][-1],
            results['history']['einstein_loss'][-1],
            results['history']['entropy_loss'][-1],
            results['history'].get('regularity_loss', [0])[-1]
        ]
        
        bars = plt.bar(loss_labels, loss_values, color=['blue', 'red', 'green', 'purple'])
        plt.yscale('log')
        plt.title('Final Loss Values', fontweight='bold')
        plt.ylabel('Loss (log scale)', fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2e}',
                    ha='center', va='bottom', rotation=0, fontsize=10)
        
        # Plot entropy components
        plt.subplot(2, 2, 2)
        components = analysis['entropy_components']
        component_labels = ['Bulk', 'Edge Modes', 'UV Correction', 'Total']
        component_values = [
            components['bulk'],
            components['edge_modes'],
            components['uv_correction'],
            components['total']
        ]
        
        bars = plt.bar(component_labels, component_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        plt.title('Entropy Components', fontweight='bold')
        plt.ylabel('Entropy Value', fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Plot holographic metrics
        plt.subplot(2, 2, 3)
        holo_labels = ['Entropy', 'Area Est.', 'S/A Ratio', 'Ricci Scalar']
        holo_values = [
            analysis['holographic']['entropy'],
            analysis['holographic']['area_estimate'],
            analysis['holographic']['entropy_area_ratio'],
            analysis['holographic']['ricci_scalar']
        ]
        
        bars = plt.bar(holo_labels, holo_values, color=['#9b59b6', '#34495e', '#1abc9c', '#d35400'])
        plt.title('Holographic Metrics', fontweight='bold')
        plt.ylabel('Value', fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Plot area law metrics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # Create a text box with area law metrics
        area_law_text = (
            "AREA LAW ANALYSIS\n"
            "------------------------\n"
            f"Coefficient: {analysis['area_law']['area_law_coefficient']:.4f}\n"
            f"Intercept: {analysis['area_law']['intercept']:.4f}\n"
            f"R² Value: {analysis['area_law']['r_squared']:.4f}\n\n"
            "CONVERGENCE STATUS\n"
            "------------------------\n"
            f"Converged: {analysis['convergence']['converged']}\n"
            f"Final Loss: {analysis['convergence']['final_loss']:.6f}\n"
            f"Best Loss: {analysis['convergence']['best_loss']:.6f}\n"
        )
        
        plt.text(0.5, 0.5, area_law_text, ha='center', va='center', fontsize=12,
                bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=0.8'),
                fontfamily='monospace')
        
        plt.suptitle('EntropicUnification Simulation Summary', fontweight='bold', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
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
