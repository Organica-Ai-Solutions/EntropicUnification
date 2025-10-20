#!/usr/bin/env python3
"""
EntropicUnification Enhanced Framework Example

This script demonstrates the enhanced EntropicUnification framework with:
- Edge mode support for gauge fields
- Non-conformal matter field corrections
- Higher-order curvature terms
- Improved optimization strategies

The simulation explores the relationship between quantum entanglement
and spacetime geometry, implementing the theoretical insights from
the research literature on entanglement-gravity connections.
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum_engine import QuantumEngine, QuantumConfig
from core.utils.plotting import get_plot_manager
from core.geometry_engine import GeometryEngine, BoundaryCondition
from core.entropy_module import EntropyModule
from core.coupling_layer import CouplingLayer, StressTensorFormulation
from core.loss_functions import LossFunctions, LossFormulation
from core.optimizer import EntropicOptimizer, OptimizerConfig, OptimizationStrategy, PartitionStrategy


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_quantum_config(config):
    """Create quantum configuration from loaded config."""
    return QuantumConfig(
        num_qubits=config['quantum']['num_qubits'],
        depth=config['quantum']['circuit_depth'],
        device=config['quantum']['device'],
        interface="torch",
        shots=config['quantum']['shots'],
        seed=config['quantum']['seed']
    )


def create_optimizer_config(config):
    """Create optimizer configuration from loaded config."""
    return OptimizerConfig(
        learning_rate=config['optimization']['learning_rate'],
        steps=config['optimization']['steps'],
        checkpoint_interval=config['optimization']['checkpoint_interval'],
        log_interval=config['optimization']['log_interval'],
        metric_grad_clip=config['optimization']['metric_grad_clip'],
        results_path=config['output']['results_dir'],
        optimization_strategy=config['optimization']['optimization_strategy'],
        partition_strategy=config['optimization']['partition_strategy'],
        loss_formulation=config['optimization']['loss_formulation'],
        stress_formulation=config['coupling']['stress_form'],
        lr_schedule=config['optimization']['lr_schedule'],
        convergence_threshold=config['optimization']['convergence_threshold'],
        patience=config['optimization']['patience'],
        include_edge_modes=config['entropy']['include_edge_modes'],
        include_higher_curvature=config['geometry']['include_higher_curvature'],
        conformal_invariance=config['entropy']['conformal_invariance'],
        uv_cutoff=config['entropy']['uv_cutoff'],
        regularization_scheme=config['entropy']['regularization_scheme'],
        basin_hopping_params=config['optimization']['basin_hopping'],
        annealing_schedule=config['optimization']['annealing']
    )


def setup_components(config):
    """Set up all components of the EntropicUnification framework."""
    # Create quantum engine
    quantum_config = create_quantum_config(config)
    quantum_engine = QuantumEngine(quantum_config)
    
    # Create geometry engine
    geometry_engine = GeometryEngine(
        dimensions=config['spacetime']['dimensions'],
        lattice_size=config['spacetime']['lattice_size'],
        dx=config['spacetime']['dx'],
        regularization=config['geometry']['metric_regularization'],
        initial_metric=config['geometry']['initial_metric'],
        boundary_condition=config['geometry']['boundary_condition'],
        signature=config['spacetime']['signature'],
    )
    
    # Create entropy module
    entropy_module = EntropyModule(
        quantum_engine=quantum_engine,
        epsilon=1e-12,
        uv_cutoff=config['entropy']['uv_cutoff'],
        include_edge_modes=config['entropy']['include_edge_modes'],
        conformal_invariance=config['entropy']['conformal_invariance'],
        regularization_scheme=config['entropy']['regularization_scheme']
    )
    
    # Configure edge mode parameters if enabled
    if config['entropy']['include_edge_modes']:
        entropy_module.set_edge_mode_parameters(
            dimension=config['entropy']['edge_mode_dimension'],
            entropy_factor=config['entropy']['edge_mode_entropy_factor']
        )
    
    # Create coupling layer
    coupling_layer = CouplingLayer(
        geometry_engine=geometry_engine,
        entropy_module=entropy_module,
        coupling_strength=config['coupling']['coupling_strength'],
        stress_form=config['coupling']['stress_form'],
        include_edge_modes=config['coupling']['include_edge_modes'],
        include_higher_curvature=config['coupling']['include_higher_curvature'],
        conformal_invariance=config['coupling']['conformal_invariance'],
        hbar_factor=config['coupling']['hbar_factor']
    )
    
    # Set higher curvature parameters if enabled
    if config['coupling']['include_higher_curvature']:
        coupling_layer.set_higher_curvature_parameters(
            alpha_gb=config['coupling']['higher_curvature']['alpha_gb'],
            lambda_cosmo=config['coupling']['higher_curvature']['lambda_cosmo']
        )
    
    # Create loss functions
    loss_functions = LossFunctions(
        coupling_layer=coupling_layer,
        regularization_weight=config['optimization']['loss_weights']['smoothness'],
        curvature_weight=config['optimization']['loss_weights']['curvature'],
        formulation=config['optimization']['loss_formulation'],
        basin_hopping=config['optimization']['optimization_strategy'] == 'basin_hopping',
        annealing_schedule=config['optimization']['annealing']
    )
    
    # Create optimizer
    optimizer_config = create_optimizer_config(config)
    optimizer = EntropicOptimizer(
        quantum_engine=quantum_engine,
        geometry_engine=geometry_engine,
        entropy_module=entropy_module,
        coupling_layer=coupling_layer,
        loss_functions=loss_functions,
        config=optimizer_config
    )
    
    return {
        'quantum_engine': quantum_engine,
        'geometry_engine': geometry_engine,
        'entropy_module': entropy_module,
        'coupling_layer': coupling_layer,
        'loss_functions': loss_functions,
        'optimizer': optimizer
    }


def prepare_initial_state(quantum_engine, state_type='bell'):
    """Prepare initial quantum state."""
    if state_type == 'bell':
        return quantum_engine.bell_state()
    elif state_type == 'ghz':
        return quantum_engine.ghz_state()
    else:
        # Random state
        state = torch.randn(2**quantum_engine.num_qubits, dtype=torch.complex128)
        state = state / torch.norm(state)
        return state


def run_simulation(components, config, state_type='bell'):
    """Run the entropic simulation."""
    print(f"Starting EntropicUnification simulation with {state_type} state...")
    
    # Get components
    quantum_engine = components['quantum_engine']
    optimizer = components['optimizer']
    
    # Prepare initial state
    initial_state = prepare_initial_state(quantum_engine, state_type)
    
    # Create parameters and times
    num_qubits = quantum_engine.num_qubits
    depth = quantum_engine.depth
    
    # Calculate total parameters needed based on the quantum engine's configuration
    params_per_qubit = quantum_engine.params_per_qubit  # 4: RY, RZ, RX, time evolution
    params_per_entangler = quantum_engine.params_per_entangler  # 2: CRX, CRZ
    total_params = (params_per_qubit * num_qubits + params_per_entangler * (num_qubits - 1)) * depth
    
    # Create parameter tensor with the correct shape
    parameters = torch.randn(total_params, dtype=torch.float64)
    times = torch.linspace(0, 1, 10, dtype=torch.float64)
    
    # Define partition (first half of qubits)
    partition = list(range(num_qubits // 2))
    
    # Define loss weights
    weights = config['optimization']['loss_weights']
    
    # Run optimization
    start_time = time.time()
    results = optimizer.train(
        parameters=parameters,
        times=times,
        partition=partition,
        weights=weights
    )
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Final loss: {results['history']['total_loss'][-1]:.6f}")
    
    return results


def analyze_results(components, results, config):
    """Analyze simulation results."""
    print("\nAnalyzing results...")
    
    # Get components
    optimizer = components['optimizer']
    entropy_module = components['entropy_module']
    geometry_engine = components['geometry_engine']
    
    # Get final state
    final_state = results['final_state']
    
    # Analyze convergence
    convergence = optimizer.analyze_convergence()
    print(f"Convergence analysis:")
    print(f"  Converged: {convergence['converged']}")
    print(f"  Final loss: {convergence['final_loss']:.6f}")
    print(f"  Best loss: {convergence['best_loss']:.6f}")
    
    # Compute entropy for different partitions
    num_qubits = components['quantum_engine'].num_qubits
    partitions = [
        list(range(k)) for k in range(1, num_qubits)
    ]
    
    # Analyze area law
    area_law = optimizer.compute_entropic_area_law(final_state, partitions)
    print(f"\nArea law analysis:")
    print(f"  Coefficient: {area_law['area_law_coefficient']:.4f}")
    print(f"  R²: {area_law['r_squared']:.4f}")
    
    # Compute holographic metrics
    holographic = optimizer.compute_holographic_metrics(final_state, partitions[0])
    print(f"\nHolographic metrics:")
    print(f"  Entropy: {holographic['entropy']:.4f}")
    print(f"  Area estimate: {holographic['area_estimate']:.4f}")
    print(f"  Entropy/Area ratio: {holographic['entropy_area_ratio']:.4f}")
    print(f"  Ricci scalar: {holographic['ricci_scalar']:.4f}")
    
    # Get entropy components
    entropy_components = entropy_module.get_entropy_components()
    print(f"\nEntropy components:")
    print(f"  Bulk: {entropy_components['bulk']:.4f}")
    print(f"  Edge modes: {entropy_components['edge_modes']:.4f}")
    print(f"  UV correction: {entropy_components['uv_correction']:.4f}")
    print(f"  Total: {entropy_components['total']:.4f}")
    
    return {
        'convergence': convergence,
        'area_law': area_law,
        'holographic': holographic,
        'entropy_components': entropy_components
    }


def compare_results(all_results, all_analyses, output_dir):
    """Generate comparative analysis of different quantum states."""
    print("\nGenerating comparative analysis...")
    
    # Create output directory
    plot_dir = Path(output_dir) / "comparative"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Set global plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })
    
    # Compare loss convergence
    plt.figure(figsize=(12, 8))
    
    for state_type, results in all_results.items():
        iterations = range(len(results['history']['total_loss']))
        plt.plot(iterations, results['history']['total_loss'], linewidth=2, 
                label=f'{state_type.upper()} State')
    
    plt.yscale('log')
    plt.xlabel('Optimization Iteration', fontweight='bold')
    plt.ylabel('Total Loss (log scale)', fontweight='bold')
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.title('Loss Convergence Comparison Across Quantum States', fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanatory text
    explanation = (
        "This plot compares optimization convergence for different initial quantum states.\n"
        "Lower final loss values indicate better alignment between entropy and geometry.\n"
        "Convergence patterns reveal how different quantum states affect the entropic-geometric coupling."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(plot_dir / 'loss_comparison.png', dpi=300)
    
    # Compare area law coefficients
    plt.figure(figsize=(10, 8))
    
    state_types = list(all_analyses.keys())
    coefficients = [all_analyses[s]['area_law']['area_law_coefficient'] for s in state_types]
    r_squared = [all_analyses[s]['area_law']['r_squared'] for s in state_types]
    
    x = np.arange(len(state_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, coefficients, width, label='Area Law Coefficient', color='blue')
    bars2 = ax.bar(x + width/2, r_squared, width, label='R² Value', color='red')
    
    ax.set_xlabel('Quantum State Type', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Area Law Metrics Comparison', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in state_types])
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)
    
    # Add explanatory text
    explanation = (
        "This chart compares how different quantum states conform to the area law relationship.\n"
        "Area Law Coefficient: Proportionality constant between entropy and boundary area.\n"
        "R² Value: Statistical measure of how well the data fits the area law (higher is better).\n"
        "Ideal holographic systems would have coefficient = 0.25 (Bekenstein-Hawking) and R² = 1.0."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(plot_dir / 'area_law_comparison.png', dpi=300)
    
    # Compare entropy components
    plt.figure(figsize=(15, 10))
    
    # Extract entropy components for each state
    bulk_values = [all_analyses[s]['entropy_components']['bulk'] for s in state_types]
    edge_values = [all_analyses[s]['entropy_components']['edge_modes'] for s in state_types]
    uv_values = [all_analyses[s]['entropy_components']['uv_correction'] for s in state_types]
    
    # Set up the bar chart
    x = np.arange(len(state_types))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width, bulk_values, width, label='Bulk Entropy', color='#3498db')
    bars2 = ax.bar(x, edge_values, width, label='Edge Modes', color='#e74c3c')
    bars3 = ax.bar(x + width, uv_values, width, label='UV Correction', color='#2ecc71')
    
    ax.set_xlabel('Quantum State Type', fontweight='bold')
    ax.set_ylabel('Entropy Value', fontweight='bold')
    ax.set_title('Entropy Components Comparison', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in state_types])
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)
    
    # Add explanatory text
    explanation = (
        "This chart compares entropy components across different quantum states.\n"
        "Bulk Entropy: Standard von Neumann entropy from quantum state bipartition.\n"
        "Edge Modes: Contribution from gauge degrees of freedom at the entangling surface.\n"
        "UV Correction: Regularization effects from the UV cutoff.\n"
        "The distribution of components reveals how different states encode information geometrically."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(plot_dir / 'entropy_components_comparison.png', dpi=300)
    
    # Create summary table
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Collect summary data
    data = []
    columns = ['State Type', 'Final Loss', 'Area Coeff.', 'R²', 'S/A Ratio', 'Ricci Scalar', 
              'Bulk Entropy', 'Edge Modes', 'Converged']
    
    for state in state_types:
        results = all_results[state]
        analysis = all_analyses[state]
        
        data.append([
            state.upper(),
            f"{results['history']['total_loss'][-1]:.6f}",
            f"{analysis['area_law']['area_law_coefficient']:.4f}",
            f"{analysis['area_law']['r_squared']:.4f}",
            f"{analysis['holographic']['entropy_area_ratio']:.4f}",
            f"{analysis['holographic']['ricci_scalar']:.4f}",
            f"{analysis['entropy_components']['bulk']:.4f}",
            f"{analysis['entropy_components']['edge_modes']:.4f}",
            str(analysis['convergence']['converged'])
        ])
    
    # Create table
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i, key in enumerate(table._cells):
        cell = table._cells[key]
        if i < len(columns):  # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#34495e')
        else:  # Data rows
            cell.set_facecolor('#f8f9fa')
            if key[1] == 0:  # State type column
                cell.set_text_props(weight='bold')
    
    plt.title('EntropicUnification Comparative Results', fontweight='bold', fontsize=20, pad=20)
    
    # Add explanatory text
    explanation = (
        "This table summarizes key metrics across different quantum states.\n"
        "Compare how different initial states affect the entropic-geometric coupling and area law relationship.\n"
        "Lower loss values and higher R² values indicate better performance in the entropic field equations."
    )
    plt.figtext(0.5, 0.05, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(plot_dir / 'comparative_summary.png', dpi=300)
    
    print(f"Comparative analysis plots saved to {plot_dir}")


def plot_results(results, analysis, config, output_dir, simulation_type=None):
    """Plot simulation results with enhanced labels and explanations."""
    if not config['output']['visualize']['save_plots']:
        return
        
    print("\nGenerating enhanced plots...")
    
    # Use the unified plotting system
    plot_manager = get_plot_manager(config, output_dir)
    plot_paths = plot_manager.plot_all(results, analysis, config, simulation_type)
    
    print(f"Enhanced plots saved to {plot_manager.get_plot_dir(simulation_type)}")
    
    return plot_paths
    
    # The plotting is now handled by the PlotManager


def main():
    """Main function to run the simulation."""
    parser = argparse.ArgumentParser(description='EntropicUnification Simulation')
    parser.add_argument('--config', type=str, default='data/configs.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--state', type=str, default='bell',
                        choices=['bell', 'ghz', 'random', 'all'],
                        help='Initial quantum state type (use "all" to run all types)')
    parser.add_argument('--stress', type=str, default=None,
                        choices=['jacobson', 'canonical', 'faulkner', 'modified'],
                        help='Stress tensor formulation to use')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Override number of optimization iterations')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update output directory
    config['output']['results_dir'] = args.output
    
    # Override iterations if specified
    if args.iterations is not None:
        config['optimization']['steps'] = args.iterations
        print(f"Overriding iterations to {args.iterations}")
    
    # Override stress tensor formulation if specified
    if args.stress is not None:
        config['coupling']['stress_form'] = args.stress
        print(f"Using {args.stress} stress tensor formulation")
    
    # Set up components
    components = setup_components(config)
    
    if args.state == 'all':
        # Run simulations for all state types
        states = ['bell', 'ghz', 'random']
        all_results = {}
        all_analyses = {}
        
        for state_type in states:
            print(f"\n{'='*50}")
            print(f"Running simulation with {state_type.upper()} state")
            print(f"{'='*50}")
            
            # Create state-specific output directory
            state_output = f"{args.output}/{state_type}"
            os.makedirs(state_output, exist_ok=True)
            
            # Update output directory in config
            config['output']['results_dir'] = state_output
            
            # Run simulation for this state
            results = run_simulation(components, config, state_type)
            analysis = analyze_results(components, results, config)
            
                    # Plot results
                    plot_results(results, analysis, config, state_output, state_type)
            
            # Store results and analysis
            all_results[state_type] = results
            all_analyses[state_type] = analysis
        
        # Generate comparative analysis
        print("\nGenerating comparative analysis...")
        compare_results(all_results, all_analyses, args.output)
        
    else:
        # Run simulation for the specified state
        results = run_simulation(components, config, args.state)
        
        # Analyze results
        analysis = analyze_results(components, results, config)
        
                # Plot results
                plot_results(results, analysis, config, args.output, args.state)
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
