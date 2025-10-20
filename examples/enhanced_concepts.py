#!/usr/bin/env python3
"""
Demonstration of enhanced concepts in the EntropicUnification framework.

This script demonstrates key concepts from our improvements:
1. Advanced finite difference methods
2. Physical constraint enforcement
3. Multiple optimization strategies
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

def demonstrate_finite_difference_accuracy():
    """Demonstrate the accuracy of different finite difference methods."""
    print("Demonstrating finite difference accuracy...")
    
    # Create a test function with known derivatives
    x = torch.linspace(0, 2*np.pi, 100)
    f = torch.sin(x)
    
    # Analytical derivatives
    df_analytical = torch.cos(x)
    d2f_analytical = -torch.sin(x)
    
    # 2nd order central difference for first derivative
    df_2nd = torch.zeros_like(f)
    for i in range(1, len(f) - 1):
        df_2nd[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
    df_2nd[0] = (f[1] - f[0]) / (x[1] - x[0])  # Forward difference for first point
    df_2nd[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])  # Backward difference for last point
    
    # 4th order central difference for first derivative
    df_4th = torch.zeros_like(f)
    for i in range(2, len(f) - 2):
        df_4th[i] = (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12 * (x[i+1] - x[i]))
    # Handle boundaries with lower order methods
    df_4th[0] = df_2nd[0]
    df_4th[1] = df_2nd[1]
    df_4th[-2] = df_2nd[-2]
    df_4th[-1] = df_2nd[-1]
    
    # Spectral method for first derivative (using FFT)
    n = len(x)
    dx = x[1] - x[0]
    k = 2 * np.pi * torch.fft.rfftfreq(n, dx)
    f_hat = torch.fft.rfft(f)
    df_spectral = torch.fft.irfft(1j * k * f_hat, n=n)
    
    # Compute errors
    error_2nd = torch.abs(df_2nd - df_analytical).mean().item()
    error_4th = torch.abs(df_4th - df_analytical).mean().item()
    error_spectral = torch.abs(df_spectral - df_analytical).mean().item()
    
    # Print results
    print(f"First derivative errors:")
    print(f"  2nd order:  {error_2nd:.6e}")
    print(f"  4th order:  {error_4th:.6e}")
    print(f"  Spectral:   {error_spectral:.6e}")
    print(f"  Improvement from 2nd to 4th: {error_2nd/error_4th:.2f}x")
    print(f"  Improvement from 4th to spectral: {error_4th/error_spectral:.2f}x")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    plt.plot(x, df_analytical, 'k-', linewidth=2, label='Analytical')
    plt.plot(x, df_2nd, 'r--', linewidth=1.5, label='2nd Order')
    plt.plot(x, df_4th, 'g--', linewidth=1.5, label='4th Order')
    plt.plot(x, df_spectral, 'b--', linewidth=1.5, label='Spectral')
    
    plt.xlabel('x', fontweight='bold')
    plt.ylabel('df/dx', fontweight='bold')
    plt.title('First Derivative Comparison', fontweight='bold', pad=20)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add error annotations
    plt.annotate(f"2nd Order Error: {error_2nd:.2e}", xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.annotate(f"4th Order Error: {error_4th:.2e}", xy=(0.02, 0.90), xycoords='axes fraction',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.annotate(f"Spectral Error: {error_spectral:.2e}", xy=(0.02, 0.85), xycoords='axes fraction',
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Create output directory
    output_dir = Path('results/enhanced_concepts')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'finite_difference_comparison.png', dpi=300)
    plt.close()
    
    return {
        'error_2nd': error_2nd,
        'error_4th': error_4th,
        'error_spectral': error_spectral
    }

def demonstrate_bianchi_identity():
    """Demonstrate the enforcement of the Bianchi identity."""
    print("\nDemonstrating Bianchi identity enforcement...")
    
    # Create a simple 4D Riemann tensor (not necessarily satisfying the Bianchi identity)
    d = 4  # Dimensions
    riemann = torch.zeros(1, d, d, d, d)  # [batch, a, b, c, d]
    
    # Fill with random values
    torch.manual_seed(42)  # For reproducibility
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for d_idx in range(d):
                    riemann[0, a, b, c, d_idx] = torch.randn(1).item()
    
    # Enforce antisymmetry in first two indices: R_{abcd} = -R_{bacd}
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for d_idx in range(d):
                    riemann[0, a, b, c, d_idx] = 0.5 * (riemann[0, a, b, c, d_idx] - riemann[0, b, a, c, d_idx])
    
    # Enforce antisymmetry in last two indices: R_{abcd} = -R_{abdc}
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for d_idx in range(d):
                    riemann[0, a, b, c, d_idx] = 0.5 * (riemann[0, a, b, c, d_idx] - riemann[0, a, b, d_idx, c])
    
    # Check Bianchi identity violation before enforcement
    bianchi_violation_before = torch.zeros(1)
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for d_idx in range(d):
                    # R_{abcd} + R_{acdb} + R_{adbc}
                    cyclic_sum = (
                        riemann[0, a, b, c, d_idx] + 
                        riemann[0, a, c, d_idx, b] + 
                        riemann[0, a, d_idx, b, c]
                    )
                    bianchi_violation_before += torch.abs(cyclic_sum)
    
    # Enforce Bianchi identity
    riemann_after = riemann.clone()
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for d_idx in range(d):
                    # Compute the cyclic sum
                    cyclic_sum = (
                        riemann[0, a, b, c, d_idx] + 
                        riemann[0, a, c, d_idx, b] + 
                        riemann[0, a, d_idx, b, c]
                    )
                    
                    # Distribute the correction equally among the three terms
                    correction = cyclic_sum / 3.0
                    
                    riemann_after[0, a, b, c, d_idx] -= correction
                    riemann_after[0, a, c, d_idx, b] -= correction
                    riemann_after[0, a, d_idx, b, c] -= correction
    
    # Check Bianchi identity violation after enforcement
    bianchi_violation_after = torch.zeros(1)
    for a in range(d):
        for b in range(d):
            for c in range(d):
                for d_idx in range(d):
                    # R_{abcd} + R_{acdb} + R_{adbc}
                    cyclic_sum = (
                        riemann_after[0, a, b, c, d_idx] + 
                        riemann_after[0, a, c, d_idx, b] + 
                        riemann_after[0, a, d_idx, b, c]
                    )
                    bianchi_violation_after += torch.abs(cyclic_sum)
    
    # Print results
    print(f"Bianchi identity violation before: {bianchi_violation_before.item():.6e}")
    print(f"Bianchi identity violation after:  {bianchi_violation_after.item():.6e}")
    print(f"Improvement factor: {bianchi_violation_before.item() / (bianchi_violation_after.item() + 1e-10):.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    labels = ['Before Enforcement', 'After Enforcement']
    values = [bianchi_violation_before.item(), bianchi_violation_after.item()]
    
    bars = plt.bar(labels, values, color=['#e74c3c', '#2ecc71'])
    plt.yscale('log')
    plt.ylabel('Bianchi Identity Violation (log scale)', fontweight='bold')
    plt.title('Effect of Bianchi Identity Enforcement', fontweight='bold', pad=20)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}',
                ha='center', va='bottom', fontsize=10)
    
    # Add explanation
    explanation = (
        "The Bianchi identity (R_{abcd} + R_{acdb} + R_{adbc} = 0) is a fundamental constraint\n"
        "that the Riemann curvature tensor must satisfy. Enforcing this constraint ensures\n"
        "that the geometric calculations are physically consistent."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Create output directory
    output_dir = Path('results/enhanced_concepts')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'bianchi_identity.png', dpi=300)
    plt.close()
    
    return {
        'violation_before': bianchi_violation_before.item(),
        'violation_after': bianchi_violation_after.item()
    }

def demonstrate_optimization_strategies():
    """Demonstrate different optimization strategies."""
    print("\nDemonstrating optimization strategies...")
    
    # Define a challenging 2D function with multiple local minima
    def rosenbrock(x, y, a=1, b=100):
        return (a - x)**2 + b * (y - x**2)**2
    
    # Create a grid of points
    x = torch.linspace(-2, 2, 100)
    y = torch.linspace(-1, 3, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Z = rosenbrock(X, Y)
    
    # Starting point
    x0, y0 = -1.5, 2.5
    
    # Define different optimization strategies
    strategies = {
        'SGD': {
            'lr': 0.001,
            'momentum': 0.0,
            'adaptive': False,
            'color': 'r',
            'label': 'SGD'
        },
        'SGD+Momentum': {
            'lr': 0.001,
            'momentum': 0.9,
            'adaptive': False,
            'color': 'g',
            'label': 'SGD with Momentum'
        },
        'Adam': {
            'lr': 0.01,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'adaptive': True,
            'color': 'b',
            'label': 'Adam'
        }
    }
    
    # Run optimizations
    results = {}
    for name, params in strategies.items():
        print(f"Running {name} optimization...")
        
        # Initialize
        x_history = [x0]
        y_history = [y0]
        loss_history = [rosenbrock(x0, y0)]
        
        # Parameters
        x_current = x0
        y_current = y0
        iterations = 100
        
        # Initialize momentum/adaptive terms
        vx, vy = 0, 0  # Momentum
        mx, my = 0, 0  # First moment (Adam)
        vx_adam, vy_adam = 0, 0  # Second moment (Adam)
        
        # Run optimization
        for i in range(iterations):
            # Compute gradients
            with torch.no_grad():
                # Numerical gradients
                eps = 1e-6
                dx = (rosenbrock(x_current + eps, y_current) - rosenbrock(x_current - eps, y_current)) / (2 * eps)
                dy = (rosenbrock(x_current, y_current + eps) - rosenbrock(x_current, y_current - eps)) / (2 * eps)
                
                # Apply optimization strategy
                if name == 'SGD':
                    # Simple SGD
                    x_current -= params['lr'] * dx
                    y_current -= params['lr'] * dy
                    
                elif name == 'SGD+Momentum':
                    # SGD with momentum
                    vx = params['momentum'] * vx - params['lr'] * dx
                    vy = params['momentum'] * vy - params['lr'] * dy
                    x_current += vx
                    y_current += vy
                    
                elif name == 'Adam':
                    # Adam optimizer
                    t = i + 1
                    
                    # Update biased first moment estimate
                    mx = params['beta1'] * mx + (1 - params['beta1']) * dx
                    my = params['beta1'] * my + (1 - params['beta1']) * dy
                    
                    # Update biased second raw moment estimate
                    vx_adam = params['beta2'] * vx_adam + (1 - params['beta2']) * dx**2
                    vy_adam = params['beta2'] * vy_adam + (1 - params['beta2']) * dy**2
                    
                    # Bias correction
                    mx_hat = mx / (1 - params['beta1']**t)
                    my_hat = my / (1 - params['beta1']**t)
                    vx_hat = vx_adam / (1 - params['beta2']**t)
                    vy_hat = vy_adam / (1 - params['beta2']**t)
                    
                    # Update parameters
                    x_current -= params['lr'] * mx_hat / (torch.sqrt(torch.tensor(vx_hat)) + params['epsilon'])
                    y_current -= params['lr'] * my_hat / (torch.sqrt(torch.tensor(vy_hat)) + params['epsilon'])
            
            # Record history
            x_history.append(x_current)
            y_history.append(y_current)
            loss_history.append(rosenbrock(x_current, y_current))
        
        # Store results
        results[name] = {
            'x_history': x_history,
            'y_history': y_history,
            'loss_history': loss_history,
            'params': params
        }
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot loss curves
    plt.subplot(2, 2, 1)
    for name, res in results.items():
        plt.plot(res['loss_history'], color=res['params']['color'], label=res['params']['label'])
    plt.yscale('log')
    plt.xlabel('Iteration', fontweight='bold')
    plt.ylabel('Loss (log scale)', fontweight='bold')
    plt.title('Loss Curves', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2D trajectories
    plt.subplot(2, 2, 2)
    
    # Plot contour of the function
    levels = torch.logspace(0, 4, 20)
    plt.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=levels, alpha=0.3, cmap='viridis')
    
    # Plot optimization trajectories
    for name, res in results.items():
        plt.plot(res['x_history'], res['y_history'], color=res['params']['color'], 
                 label=res['params']['label'], marker='o', markersize=3)
        plt.plot(res['x_history'][-1], res['y_history'][-1], color=res['params']['color'], 
                 marker='*', markersize=10)
    
    plt.xlabel('x', fontweight='bold')
    plt.ylabel('y', fontweight='bold')
    plt.title('Optimization Trajectories', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot final positions
    plt.subplot(2, 2, 3)
    final_x = [res['x_history'][-1] for name, res in results.items()]
    final_y = [res['y_history'][-1] for name, res in results.items()]
    final_loss = [res['loss_history'][-1] for name, res in results.items()]
    
    bars = plt.bar([res['params']['label'] for name, res in results.items()], final_loss, 
                  color=[res['params']['color'] for name, res in results.items()])
    
    plt.yscale('log')
    plt.xlabel('Optimization Strategy', fontweight='bold')
    plt.ylabel('Final Loss (log scale)', fontweight='bold')
    plt.title('Final Loss Comparison', fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2e}',
                ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add summary table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Create summary text
    summary_text = "OPTIMIZATION COMPARISON\n"
    summary_text += "------------------------\n\n"
    
    for name, res in results.items():
        summary_text += f"{res['params']['label']}:\n"
        summary_text += f"  Initial Loss: {res['loss_history'][0]:.2e}\n"
        summary_text += f"  Final Loss: {res['loss_history'][-1]:.2e}\n"
        summary_text += f"  Improvement: {res['loss_history'][0]/res['loss_history'][-1]:.2f}x\n"
        summary_text += f"  Final Position: ({res['x_history'][-1]:.4f}, {res['y_history'][-1]:.4f})\n\n"
    
    plt.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10,
            bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=0.8'),
            fontfamily='monospace')
    
    plt.suptitle('Comparison of Optimization Strategies', fontweight='bold', fontsize=16)
    
    # Add explanation
    explanation = (
        "This demonstration compares different optimization strategies on the Rosenbrock function,\n"
        "a challenging non-convex optimization problem with a narrow valley. The plots show how\n"
        "advanced optimization techniques like momentum and adaptive learning rates (Adam)\n"
        "can significantly improve convergence compared to basic gradient descent."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Create output directory
    output_dir = Path('results/enhanced_concepts')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / 'optimization_comparison.png', dpi=300)
    plt.close()
    
    return results

def main():
    """Run all demonstrations."""
    # Demonstrate finite difference accuracy
    fd_results = demonstrate_finite_difference_accuracy()
    
    # Demonstrate Bianchi identity enforcement
    bianchi_results = demonstrate_bianchi_identity()
    
    # Demonstrate optimization strategies
    opt_results = demonstrate_optimization_strategies()
    
    # Print summary
    print("\n" + "="*50)
    print("Enhanced Concepts Demonstration Summary")
    print("="*50)
    
    print("\nFinite Difference Accuracy:")
    print(f"  2nd order error: {fd_results['error_2nd']:.2e}")
    print(f"  4th order error: {fd_results['error_4th']:.2e}")
    print(f"  Spectral error:  {fd_results['error_spectral']:.2e}")
    print(f"  Improvement from 2nd to 4th: {fd_results['error_2nd']/fd_results['error_4th']:.2f}x")
    print(f"  Improvement from 4th to spectral: {fd_results['error_4th']/fd_results['error_spectral']:.2f}x")
    
    print("\nBianchi Identity Enforcement:")
    print(f"  Violation before: {bianchi_results['violation_before']:.2e}")
    print(f"  Violation after:  {bianchi_results['violation_after']:.2e}")
    print(f"  Improvement factor: {bianchi_results['violation_before']/(bianchi_results['violation_after']+1e-10):.2f}x")
    
    print("\nOptimization Strategies:")
    for name, res in opt_results.items():
        print(f"  {name}:")
        print(f"    Initial loss: {res['loss_history'][0]:.2e}")
        print(f"    Final loss:   {res['loss_history'][-1]:.2e}")
        print(f"    Improvement:  {res['loss_history'][0]/res['loss_history'][-1]:.2f}x")
    
    print("\nAll demonstrations completed. Results saved to results/enhanced_concepts/")

if __name__ == "__main__":
    main()
