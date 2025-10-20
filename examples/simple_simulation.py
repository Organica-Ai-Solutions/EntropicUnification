#!/usr/bin/env python3
"""
Simple EntropicUnification simulation with minimal dependencies.
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_simple_simulation():
    """Run a simple simulation of entropic unification."""
    print("Running simple EntropicUnification simulation...")
    
    # Parameters
    num_qubits = 2
    lattice_size = 10
    dimensions = 4
    num_iterations = 50
    
    # Create a simple quantum state (Bell state)
    state = torch.zeros(2**num_qubits, dtype=torch.complex128)
    state[0] = 1 / np.sqrt(2)
    state[-1] = 1 / np.sqrt(2)
    
    # Create a simple metric (Minkowski)
    metric = torch.zeros(lattice_size, dimensions, dimensions)
    for i in range(lattice_size):
        metric[i] = torch.eye(dimensions)
        metric[i, 0, 0] = -1.0  # Lorentzian signature
    
    # Create a simple partition
    partition = [0]  # First qubit
    
    # Compute reduced density matrix
    rho = torch.outer(state, state.conj())
    rho_A = torch.zeros((2, 2), dtype=torch.complex128)
    
    # Partial trace manually
    rho_A[0, 0] = rho[0, 0] + rho[1, 1]
    rho_A[0, 1] = rho[0, 2] + rho[1, 3]
    rho_A[1, 0] = rho[2, 0] + rho[3, 1]
    rho_A[1, 1] = rho[2, 2] + rho[3, 3]
    
    # Compute entanglement entropy
    eigenvalues = torch.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    entropy = -torch.sum(eigenvalues * torch.log(eigenvalues))
    
    # Simulate optimization
    print("\nSimulating optimization...")
    
    # Store metrics
    loss_history = []
    entropy_history = []
    metric_history = []
    
    for i in range(num_iterations):
        # Store current state
        metric_history.append(metric.clone())
        entropy_history.append(entropy.item())
        
        # Compute a simple loss (just for demonstration)
        loss = torch.abs(1.0 - entropy)
        loss_history.append(loss.item())
        
        # Update metric (simple gradient descent)
        if i < num_iterations - 1:  # Don't update on the last iteration
            # Simple update: move metric towards making entropy closer to 1
            if entropy < 1.0:
                # Increase entropy by making spacetime more curved
                for j in range(lattice_size):
                    metric[j, 1, 1] += 0.01
                    metric[j, 2, 2] += 0.01
                    metric[j, 3, 3] += 0.01
            else:
                # Decrease entropy by making spacetime flatter
                for j in range(lattice_size):
                    metric[j, 1, 1] -= 0.01
                    metric[j, 2, 2] -= 0.01
                    metric[j, 3, 3] -= 0.01
            
            # Recompute entropy (in a real simulation this would involve quantum state evolution)
            entropy = torch.tensor(1.0 - loss.item() * 0.9)  # Simple update for demonstration
        
        # Print progress
        if i % 10 == 0 or i == num_iterations - 1:
            print(f"Iteration {i+1}/{num_iterations}: Loss = {loss.item():.6f}, Entropy = {entropy.item():.6f}")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Create output directory
    output_dir = Path('results/simple_simulation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_iterations + 1), loss_history, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.title('Loss Curve', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=300)
    plt.close()
    
    # Plot entropy curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_iterations + 1), entropy_history, 'r-', linewidth=2)
    plt.xlabel('Iteration', fontweight='bold')
    plt.ylabel('Entropy', fontweight='bold')
    plt.title('Entropy Evolution', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'entropy_curve.png', dpi=300)
    plt.close()
    
    # Plot metric evolution (just the spatial components)
    plt.figure(figsize=(12, 8))
    spatial_components = []
    for metrics in metric_history:
        # Average the spatial components
        spatial_avg = (metrics[:, 1, 1] + metrics[:, 2, 2] + metrics[:, 3, 3]) / 3
        spatial_components.append(spatial_avg.mean().item())
    
    plt.plot(range(1, num_iterations + 1), spatial_components, 'g-', linewidth=2)
    plt.xlabel('Iteration', fontweight='bold')
    plt.ylabel('Average Spatial Metric Component', fontweight='bold')
    plt.title('Metric Evolution', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_evolution.png', dpi=300)
    plt.close()
    
    # Create summary plot
    plt.figure(figsize=(15, 10))
    
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_iterations + 1), loss_history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True, alpha=0.3)
    
    # Entropy
    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_iterations + 1), entropy_history, 'r-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title('Entropy Evolution')
    plt.grid(True, alpha=0.3)
    
    # Metric
    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_iterations + 1), spatial_components, 'g-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Metric')
    plt.title('Metric Evolution')
    plt.grid(True, alpha=0.3)
    
    # Final metrics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    info_text = (
        f"SIMULATION SUMMARY\n"
        f"------------------------\n"
        f"Initial Loss: {loss_history[0]:.6f}\n"
        f"Final Loss: {loss_history[-1]:.6f}\n"
        f"Improvement: {(1 - loss_history[-1]/loss_history[0])*100:.2f}%\n\n"
        f"Initial Entropy: {entropy_history[0]:.6f}\n"
        f"Final Entropy: {entropy_history[-1]:.6f}\n"
        f"Change: {(entropy_history[-1] - entropy_history[0]):.6f}\n\n"
        f"Initial Metric: {spatial_components[0]:.6f}\n"
        f"Final Metric: {spatial_components[-1]:.6f}\n"
        f"Change: {(spatial_components[-1] - spatial_components[0]):.6f}"
    )
    plt.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12,
            bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=0.8'),
            fontfamily='monospace')
    
    plt.suptitle('EntropicUnification Simulation Summary', fontweight='bold', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / 'summary.png', dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_dir}")
    
    return {
        'loss_history': loss_history,
        'entropy_history': entropy_history,
        'metric_history': metric_history
    }

if __name__ == "__main__":
    run_simple_simulation()
