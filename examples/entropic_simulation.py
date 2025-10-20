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
    parameters = torch.randn(quantum_engine.depth, 2, num_qubits, dtype=torch.float64)
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


def plot_results(results, analysis, config, output_dir):
    """Plot simulation results."""
    if not config['output']['visualize']['save_plots']:
        return
        
    print("\nGenerating plots...")
    
    # Create output directory
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves
    if 'loss_curves' in config['output']['visualize']['plot_types']:
        plt.figure(figsize=(10, 6))
        plt.plot(results['history']['total_loss'], label='Total Loss')
        plt.plot(results['history']['einstein_loss'], label='Einstein Loss')
        plt.plot(results['history']['entropy_loss'], label='Entropy Loss')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Entropic Optimization Progress')
        plt.savefig(plot_dir / 'loss_curves.png')
        
    # Plot entropy vs area
    if 'entropy_vs_area' in config['output']['visualize']['plot_types']:
        plt.figure(figsize=(8, 6))
        plt.scatter(analysis['area_law']['areas'], analysis['area_law']['entropies'])
        
        # Plot best fit line
        areas = np.array(analysis['area_law']['areas'])
        coefficient = analysis['area_law']['area_law_coefficient']
        intercept = analysis['area_law']['intercept']
        plt.plot(areas, coefficient * areas + intercept, 'r--', 
                label=f'S = {coefficient:.4f}A + {intercept:.4f}')
        
        plt.xlabel('Boundary Area')
        plt.ylabel('Entanglement Entropy')
        plt.title('Entropy-Area Relationship')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_dir / 'entropy_area.png')
        
    # Plot metric evolution
    if 'metric_evolution' in config['output']['visualize']['plot_types'] and len(results['history']['total_loss']) > 0:
        # Create metric evolution plot if we have checkpoints
        checkpoint_files = list(Path(config['output']['results_dir']).glob('checkpoint_*.pt'))
        
        if checkpoint_files:
            plt.figure(figsize=(12, 8))
            
            # Load a few checkpoints
            checkpoints = sorted(checkpoint_files)
            if len(checkpoints) > 5:
                # Sample at most 5 checkpoints
                indices = np.linspace(0, len(checkpoints)-1, 5).astype(int)
                checkpoints = [checkpoints[i] for i in indices]
            
            for i, cp_file in enumerate(checkpoints):
                checkpoint = torch.load(cp_file)
                metric = checkpoint['metric_field'][checkpoint['metric_field'].shape[0]//2]
                
                # Plot as heatmap in subplot
                plt.subplot(1, len(checkpoints), i+1)
                plt.imshow(metric.numpy(), cmap='viridis')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.title(f"Step {checkpoint['step']}")
                
            plt.tight_layout()
            plt.savefig(plot_dir / 'metric_evolution.png')
    
    print(f"Plots saved to {plot_dir}")


def main():
    """Main function to run the simulation."""
    parser = argparse.ArgumentParser(description='EntropicUnification Simulation')
    parser.add_argument('--config', type=str, default='data/configs.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--state', type=str, default='bell',
                        choices=['bell', 'ghz', 'random'],
                        help='Initial quantum state type')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update output directory
    config['output']['results_dir'] = args.output
    
    # Set up components
    components = setup_components(config)
    
    # Run simulation
    results = run_simulation(components, config, args.state)
    
    # Analyze results
    analysis = analyze_results(components, results, config)
    
    # Plot results
    plot_results(results, analysis, config, args.output)
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
