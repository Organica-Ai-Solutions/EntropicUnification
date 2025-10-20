#!/usr/bin/env python3
"""
Test script for the fixed original geometry engine.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry_engine import GeometryEngine, BoundaryCondition
from core.utils.plotting import get_plot_manager

def test_schwarzschild():
    """Test the fixed geometry engine with a Schwarzschild-like metric."""
    print("Testing fixed original geometry engine with Schwarzschild-like metric...")
    
    # Create a geometry engine
    geometry = GeometryEngine(
        dimensions=4,
        lattice_size=20,
        dx=0.1,
        boundary_condition=BoundaryCondition.PERIODIC,
        higher_curvature_terms=True,
        alpha_GB=0.1
    )
    
    # Create a Schwarzschild-like metric
    # ds^2 = -(1 - 2M/r) dt^2 + (1 - 2M/r)^(-1) dr^2 + r^2 dΩ^2
    r = torch.linspace(2.1, 10.0, geometry.lattice_size)  # Start outside the horizon
    M = 1.0  # Mass parameter
    
    # Create a new metric tensor
    with torch.no_grad():
        metric = torch.zeros_like(geometry.metric_field)
        
        for i in range(geometry.lattice_size):
            # Start with Minkowski metric
            metric[i] = torch.eye(geometry.dimensions)
            
            # Set the Schwarzschild components
            metric[i, 0, 0] = -(1.0 - 2.0 * M / r[i])  # g_tt
            metric[i, 1, 1] = 1.0 / (1.0 - 2.0 * M / r[i])  # g_rr
            metric[i, 2, 2] = r[i]**2  # g_θθ
            metric[i, 3, 3] = r[i]**2 * torch.sin(torch.tensor(np.pi/4))**2  # g_φφ (at θ = π/4)
        
        # Copy to the geometry engine's metric field
        geometry.metric_field.data.copy_(metric)
    
    # Compute curvature tensors
    print("Computing curvature tensors...")
    christoffel = geometry.compute_christoffel_symbols()
    riemann = geometry.compute_riemann_tensor()
    ricci = geometry.compute_ricci_tensor()
    scalar = geometry.compute_ricci_scalar()
    einstein = geometry.compute_einstein_tensor()
    
    # Print some results
    print(f"Christoffel symbols shape: {christoffel.shape}")
    print(f"Riemann tensor shape: {riemann.shape}")
    print(f"Ricci tensor shape: {ricci.shape}")
    print(f"Ricci scalar shape: {scalar.shape}")
    print(f"Einstein tensor shape: {einstein.shape}")
    
    print(f"Ricci scalar at r = {r[0].item():.2f}: {scalar[0].item():.6e}")
    print(f"Ricci scalar at r = {r[-1].item():.2f}: {scalar[-1].item():.6e}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot Ricci scalar
    plt.subplot(2, 2, 1)
    plt.plot(r.numpy(), scalar.detach().numpy(), 'b-', linewidth=2)
    plt.xlabel('r', fontweight='bold')
    plt.ylabel('Ricci Scalar', fontweight='bold')
    plt.title('Ricci Scalar vs. Radius', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot metric components
    plt.subplot(2, 2, 2)
    g_tt = torch.zeros(geometry.lattice_size)
    g_rr = torch.zeros(geometry.lattice_size)
    for i in range(geometry.lattice_size):
        g_tt[i] = geometry.metric_field[i, 0, 0]
        g_rr[i] = geometry.metric_field[i, 1, 1]
    
    plt.plot(r.numpy(), g_tt.detach().numpy(), 'r-', linewidth=2, label='g_tt')
    plt.plot(r.numpy(), g_rr.detach().numpy(), 'g-', linewidth=2, label='g_rr')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('r', fontweight='bold')
    plt.ylabel('Metric Component', fontweight='bold')
    plt.title('Metric Components vs. Radius', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Einstein tensor components
    plt.subplot(2, 2, 3)
    G_tt = torch.zeros(geometry.lattice_size)
    G_rr = torch.zeros(geometry.lattice_size)
    for i in range(geometry.lattice_size):
        G_tt[i] = einstein[i, 0, 0]
        G_rr[i] = einstein[i, 1, 1]
    
    plt.plot(r.numpy(), G_tt.detach().numpy(), 'r-', linewidth=2, label='G_tt')
    plt.plot(r.numpy(), G_rr.detach().numpy(), 'g-', linewidth=2, label='G_rr')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('r', fontweight='bold')
    plt.ylabel('Einstein Tensor Component', fontweight='bold')
    plt.title('Einstein Tensor Components vs. Radius', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Riemann tensor components
    plt.subplot(2, 2, 4)
    R_trtr = torch.zeros(geometry.lattice_size)
    R_tθtθ = torch.zeros(geometry.lattice_size)
    for i in range(geometry.lattice_size):
        R_trtr[i] = riemann[i, 0, 1, 0, 1]
        R_tθtθ[i] = riemann[i, 0, 2, 0, 2]
    
    plt.plot(r.numpy(), R_trtr.detach().numpy(), 'r-', linewidth=2, label='R_trtr')
    plt.plot(r.numpy(), R_tθtθ.detach().numpy(), 'g-', linewidth=2, label='R_tθtθ')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('r', fontweight='bold')
    plt.ylabel('Riemann Tensor Component', fontweight='bold')
    plt.title('Riemann Tensor Components vs. Radius', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('Fixed Original Geometry Engine: Schwarzschild-like Metric Analysis', fontweight='bold', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Create output directory
    output_dir = Path('results/original_geometry')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'schwarzschild_analysis.png', dpi=300)
    plt.close()
    
    return {
        'r': r,
        'scalar': scalar,
        'einstein': einstein,
        'riemann': riemann
    }

def test_wave():
    """Test the fixed geometry engine with a wave-like metric perturbation."""
    print("\nTesting fixed original geometry engine with wave-like metric perturbation...")
    
    # Create a geometry engine
    geometry = GeometryEngine(
        dimensions=4,
        lattice_size=50,
        dx=0.1,
        boundary_condition=BoundaryCondition.PERIODIC,
        higher_curvature_terms=False
    )
    
    # Create a wave-like metric perturbation
    x = torch.linspace(0, 2*np.pi, geometry.lattice_size)
    amplitude = 0.1
    
    # Create a new metric tensor
    with torch.no_grad():
        metric = torch.zeros_like(geometry.metric_field)
        
        for i in range(geometry.lattice_size):
            # Start with Minkowski metric
            metric[i] = torch.eye(geometry.dimensions)
            metric[i, 0, 0] = -1.0
            
            # Add a wave-like perturbation to the spatial components
            perturbation = amplitude * torch.sin(x[i])
            metric[i, 1, 1] += perturbation
            metric[i, 2, 2] += perturbation
            metric[i, 3, 3] += perturbation
        
        # Copy to the geometry engine's metric field
        geometry.metric_field.data.copy_(metric)
    
    # Compute curvature tensors
    print("Computing curvature tensors...")
    christoffel = geometry.compute_christoffel_symbols()
    riemann = geometry.compute_riemann_tensor()
    ricci = geometry.compute_ricci_tensor()
    scalar = geometry.compute_ricci_scalar()
    einstein = geometry.compute_einstein_tensor()
    
    # Print some results
    print(f"Ricci scalar at x = {x[0].item():.2f}: {scalar[0].item():.6e}")
    print(f"Ricci scalar at x = {x[geometry.lattice_size//4].item():.2f}: {scalar[geometry.lattice_size//4].item():.6e}")
    print(f"Ricci scalar at x = {x[geometry.lattice_size//2].item():.2f}: {scalar[geometry.lattice_size//2].item():.6e}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot metric perturbation
    plt.subplot(3, 1, 1)
    perturbation = torch.zeros(geometry.lattice_size)
    for i in range(geometry.lattice_size):
        perturbation[i] = geometry.metric_field[i, 1, 1] - 1.0
    
    plt.plot(x.numpy(), perturbation.detach().numpy(), 'b-', linewidth=2)
    plt.xlabel('x', fontweight='bold')
    plt.ylabel('Metric Perturbation', fontweight='bold')
    plt.title('Wave-like Metric Perturbation', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Ricci scalar
    plt.subplot(3, 1, 2)
    plt.plot(x.numpy(), scalar.detach().numpy(), 'r-', linewidth=2)
    plt.xlabel('x', fontweight='bold')
    plt.ylabel('Ricci Scalar', fontweight='bold')
    plt.title('Ricci Scalar vs. Position', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Einstein tensor components
    plt.subplot(3, 1, 3)
    G_tt = torch.zeros(geometry.lattice_size)
    G_xx = torch.zeros(geometry.lattice_size)
    for i in range(geometry.lattice_size):
        G_tt[i] = einstein[i, 0, 0]
        G_xx[i] = einstein[i, 1, 1]
    
    plt.plot(x.numpy(), G_tt.detach().numpy(), 'r-', linewidth=2, label='G_tt')
    plt.plot(x.numpy(), G_xx.detach().numpy(), 'g-', linewidth=2, label='G_xx')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('x', fontweight='bold')
    plt.ylabel('Einstein Tensor Component', fontweight='bold')
    plt.title('Einstein Tensor Components vs. Position', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle('Fixed Original Geometry Engine: Wave-like Metric Perturbation Analysis', fontweight='bold', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Create output directory
    output_dir = Path('results/original_geometry')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'wave_analysis.png', dpi=300)
    plt.close()
    
    return {
        'x': x,
        'perturbation': perturbation,
        'scalar': scalar,
        'einstein': einstein
    }

def main():
    """Run the tests."""
    # Test with Schwarzschild-like metric
    schwarzschild_results = test_schwarzschild()
    
    # Test with wave-like metric perturbation
    wave_results = test_wave()
    
    # Print summary
    print("\n" + "="*50)
    print("Fixed Original Geometry Engine Test Summary")
    print("="*50)
    
    print("\nSchwarzschildlike Metric:")
    print(f"  Ricci scalar range: [{schwarzschild_results['scalar'].min().item():.6e}, {schwarzschild_results['scalar'].max().item():.6e}]")
    print(f"  Einstein tensor magnitude: {torch.norm(schwarzschild_results['einstein'].reshape(-1)).item():.6e}")
    
    print("\nWave-like Metric Perturbation:")
    print(f"  Ricci scalar range: [{wave_results['scalar'].min().item():.6e}, {wave_results['scalar'].max().item():.6e}]")
    print(f"  Einstein tensor magnitude: {torch.norm(wave_results['einstein'].reshape(-1)).item():.6e}")
    
    print("\nTests completed. Results saved to results/original_geometry/")

if __name__ == "__main__":
    main()
