#!/usr/bin/env python3
"""
Demonstrate advanced optimization techniques for the EntropicUnification framework.

This script compares different optimization strategies and learning rate schedules
to find the most effective approach for the entropic field equations.
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
from core.geometry_engine import GeometryEngine, BoundaryCondition
from core.entropy_module import EntropyModule
from core.coupling_layer import CouplingLayer, StressTensorFormulation
from core.loss_functions import LossFunctions, LossFormulation
from core.advanced_optimizer import AdvancedEntropicOptimizer, AdvancedOptimizerConfig, OptimizerType, LRScheduleType


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


def create_advanced_optimizer_config(config, optimizer_type, lr_schedule):
    """Create advanced optimizer configuration."""
    return AdvancedOptimizerConfig(
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
        lr_schedule=lr_schedule,
        optimizer_type=optimizer_type,
        early_stopping=True,
        patience=15,
        basin_hopping=optimizer_type == "basin_hopping",
        beta1=0.9,
        beta2=0.999,
        lr_decay_rate=0.9,
        lr_decay_steps=20
    )


def setup_components(config, optimizer_type, lr_schedule):
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
    optimizer_config = create_advanced_optimizer_config(config, optimizer_type, lr_schedule)
    optimizer = AdvancedEntropicOptimizer(
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
    convergence = optimizer.analyze_convergence(results['history'])
    print(f"Convergence analysis:")
    print(f"  Converged: {convergence['converged']}")
    print(f"  Final loss: {convergence['final_loss']:.6f}")
    print(f"  Best loss: {convergence['best_loss']:.6f}")
    print(f"  Convergence rate: {convergence['convergence_rate']:.6f}")
    
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


def plot_optimizer_comparison(all_results, output_dir):
    """Plot comparison of different optimizer configurations."""
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
    
    # Create output directory
    plot_dir = Path(output_dir) / "optimizer_comparison"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(12, 8))
    
    for name, results in all_results.items():
        if 'history' in results and 'total_loss' in results['history']:
            iterations = range(len(results['history']['total_loss']))
            plt.plot(iterations, results['history']['total_loss'], linewidth=2, label=name)
    
    plt.yscale('log')
    plt.xlabel('Optimization Iteration', fontweight='bold')
    plt.ylabel('Total Loss (log scale)', fontweight='bold')
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.title('Optimizer Comparison: Loss Convergence', fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanatory text
    explanation = (
        "This plot compares loss convergence for different optimization strategies.\n"
        "Faster convergence and lower final loss indicate more effective optimization.\n"
        "ADAM typically converges faster than SGD due to adaptive learning rates."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(plot_dir / 'loss_comparison.png', dpi=300)
    
    # Plot learning rate schedules
    plt.figure(figsize=(12, 8))
    
    for name, results in all_results.items():
        if 'history' in results and 'learning_rate' in results['history']:
            iterations = range(len(results['history']['learning_rate']))
            plt.plot(iterations, results['history']['learning_rate'], linewidth=2, label=name)
    
    plt.xlabel('Optimization Iteration', fontweight='bold')
    plt.ylabel('Learning Rate', fontweight='bold')
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.title('Learning Rate Schedules', fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanatory text
    explanation = (
        "This plot shows how learning rates change during optimization.\n"
        "Different schedules balance exploration (high LR) and exploitation (low LR).\n"
        "Adaptive schedules like cosine annealing can help escape local minima."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(plot_dir / 'learning_rate_schedules.png', dpi=300)
    
    # Create summary table
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis('off')
    
    # Collect summary data
    data = []
    columns = ['Optimizer', 'Final Loss', 'Best Loss', 'Iterations', 'Converged', 
              'Area Coeff.', 'R²', 'Entropy/Area']
    
    for name, results in all_results.items():
        if 'analysis' not in results:
            continue
            
        analysis = results['analysis']
        history = results['history']
        
        data.append([
            name,
            f"{history['total_loss'][-1]:.6f}",
            f"{min(history['total_loss']):.6f}",
            str(len(history['total_loss'])),
            str(analysis['convergence']['converged']),
            f"{analysis['area_law']['area_law_coefficient']:.4f}",
            f"{analysis['area_law']['r_squared']:.4f}",
            f"{analysis['holographic']['entropy_area_ratio']:.4f}"
        ])
    
    # Create table
    if data:
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
                if key[1] == 0:  # Optimizer column
                    cell.set_text_props(weight='bold')
    
    plt.title('Optimizer Performance Comparison', fontweight='bold', fontsize=20, pad=20)
    
    # Add explanatory text
    explanation = (
        "This table summarizes key metrics across different optimization strategies.\n"
        "Compare convergence speed, final loss values, and physical metrics like the area law coefficient.\n"
        "The best optimizer balances computational efficiency with physical accuracy."
    )
    plt.figtext(0.5, 0.05, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(plot_dir / 'optimizer_summary.png', dpi=300)
    
    print(f"Optimizer comparison plots saved to {plot_dir}")


def main():
    """Main function to run the optimization comparison."""
    parser = argparse.ArgumentParser(description='Advanced Optimization Comparison')
    parser.add_argument('--config', type=str, default='data/configs.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results/advanced_optimization',
                        help='Output directory')
    parser.add_argument('--state', type=str, default='bell',
                        choices=['bell', 'ghz', 'random'],
                        help='Initial quantum state type')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update output directory
    config['output']['results_dir'] = args.output
    
    # Set up optimizer configurations to test
    optimizer_configs = [
        ("SGD", OptimizerType.SGD, LRScheduleType.CONSTANT),
        ("SGD + Momentum", OptimizerType.SGD, LRScheduleType.STEP),
        ("ADAM", OptimizerType.ADAM, LRScheduleType.CONSTANT),
        ("ADAM + Cosine", OptimizerType.ADAM, LRScheduleType.COSINE),
        ("RMSProp", OptimizerType.RMSPROP, LRScheduleType.EXPONENTIAL),
    ]
    
    # Run simulations for each optimizer configuration
    all_results = {}
    
    for name, opt_type, lr_schedule in optimizer_configs:
        print(f"\n{'='*50}")
        print(f"Running simulation with {name}")
        print(f"{'='*50}")
        
        # Create optimizer-specific output directory
        opt_output = f"{args.output}/{name.replace(' ', '_').lower()}"
        os.makedirs(opt_output, exist_ok=True)
        
        # Update output directory in config
        config['output']['results_dir'] = opt_output
        
        # Set up components with this optimizer
        components = setup_components(config, opt_type, lr_schedule)
        
        # Run simulation
        results = run_simulation(components, config, args.state)
        
        # Analyze results
        analysis = analyze_results(components, results, config)
        
        # Store results
        all_results[name] = {
            'history': results['history'],
            'analysis': analysis
        }
    
    # Generate comparison plots
    plot_optimizer_comparison(all_results, args.output)
    
    print("\nOptimization comparison complete!")


if __name__ == "__main__":
    main()
