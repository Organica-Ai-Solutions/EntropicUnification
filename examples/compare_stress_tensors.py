#!/usr/bin/env python3
"""
Compare different stress tensor formulations in the EntropicUnification framework.

This script analyzes the results from simulations run with different stress tensor
formulations and generates comparative visualizations.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import pandas as pd
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(base_dir):
    """Load results from different stress tensor formulations."""
    formulations = ['jacobson', 'canonical', 'faulkner', 'modified']
    results = {}
    analyses = {}
    
    for form in formulations:
        form_dir = Path(base_dir) / form
        
        # Load history
        checkpoint_files = list(form_dir.glob('checkpoint_*.pt'))
        if checkpoint_files:
            latest_checkpoint = sorted(checkpoint_files)[-1]
            checkpoint = torch.load(latest_checkpoint)
            results[form] = {
                'history': {
                    'total_loss': checkpoint.get('loss_history', {}).get('total_loss', []),
                    'einstein_loss': checkpoint.get('loss_history', {}).get('einstein_loss', []),
                    'entropy_loss': checkpoint.get('loss_history', {}).get('entropy_loss', []),
                }
            }
        
        # Try to load analysis data
        try:
            with open(form_dir / 'analysis.json', 'r') as f:
                analyses[form] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If analysis.json doesn't exist, create a minimal structure
            analyses[form] = {
                'area_law': {
                    'area_law_coefficient': 0.0,
                    'r_squared': 0.0,
                    'intercept': 0.0
                },
                'holographic': {
                    'entropy': 0.0,
                    'area_estimate': 0.0,
                    'entropy_area_ratio': 0.0,
                    'ricci_scalar': 0.0
                },
                'entropy_components': {
                    'bulk': 0.0,
                    'edge_modes': 0.0,
                    'uv_correction': 0.0,
                    'total': 0.0
                },
                'convergence': {
                    'converged': False,
                    'final_loss': 0.0,
                    'best_loss': 0.0
                }
            }
            
            # Try to extract some data from the checkpoint
            if form in results:
                if results[form]['history']['total_loss']:
                    analyses[form]['convergence']['final_loss'] = results[form]['history']['total_loss'][-1]
                    analyses[form]['convergence']['best_loss'] = min(results[form]['history']['total_loss'])
    
    return results, analyses


def compare_stress_tensors(results, analyses, output_dir):
    """Generate comparative analysis of different stress tensor formulations."""
    print("\nGenerating stress tensor comparison...")
    
    # Create output directory
    plot_dir = Path(output_dir)
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
    
    for form, res in results.items():
        if 'history' in res and 'total_loss' in res['history'] and res['history']['total_loss']:
            iterations = range(len(res['history']['total_loss']))
            plt.plot(iterations, res['history']['total_loss'], linewidth=2, 
                    label=f'{form.capitalize()} Formulation')
    
    plt.yscale('log')
    plt.xlabel('Optimization Iteration', fontweight='bold')
    plt.ylabel('Total Loss (log scale)', fontweight='bold')
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.title('Loss Convergence Across Stress Tensor Formulations', fontweight='bold', pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanatory text
    explanation = (
        "This plot compares optimization convergence for different stress tensor formulations.\n"
        "Lower final loss values indicate better alignment between entropy and geometry.\n"
        "The convergence pattern reveals which formulation best captures the entropic-geometric coupling."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(plot_dir / 'loss_comparison.png', dpi=300)
    
    # Compare area law coefficients
    plt.figure(figsize=(10, 8))
    
    formulations = list(analyses.keys())
    coefficients = [analyses[f]['area_law']['area_law_coefficient'] for f in formulations]
    r_squared = [analyses[f]['area_law']['r_squared'] for f in formulations]
    
    x = np.arange(len(formulations))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, coefficients, width, label='Area Law Coefficient', color='blue')
    bars2 = ax.bar(x + width/2, r_squared, width, label='R² Value', color='red')
    
    # Add reference line for ideal Bekenstein-Hawking coefficient (0.25)
    ax.axhline(y=0.25, color='green', linestyle='--', alpha=0.7, 
              label='Ideal B-H Coefficient (0.25)')
    
    ax.set_xlabel('Stress Tensor Formulation', fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Area Law Metrics Comparison', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in formulations])
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
        "This chart compares how different stress tensor formulations conform to the area law relationship.\n"
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
    
    # Extract entropy components for each formulation
    bulk_values = [analyses[f]['entropy_components']['bulk'] for f in formulations]
    edge_values = [analyses[f]['entropy_components']['edge_modes'] for f in formulations]
    uv_values = [analyses[f]['entropy_components']['uv_correction'] for f in formulations]
    total_values = [analyses[f]['entropy_components']['total'] for f in formulations]
    
    # Set up the bar chart
    x = np.arange(len(formulations))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width*1.5, bulk_values, width, label='Bulk Entropy', color='#3498db')
    bars2 = ax.bar(x - width*0.5, edge_values, width, label='Edge Modes', color='#e74c3c')
    bars3 = ax.bar(x + width*0.5, uv_values, width, label='UV Correction', color='#2ecc71')
    bars4 = ax.bar(x + width*1.5, total_values, width, label='Total Entropy', color='#f39c12')
    
    ax.set_xlabel('Stress Tensor Formulation', fontweight='bold')
    ax.set_ylabel('Entropy Value', fontweight='bold')
    ax.set_title('Entropy Components Comparison', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in formulations])
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
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
        "This chart compares entropy components across different stress tensor formulations.\n"
        "Bulk Entropy: Standard von Neumann entropy from quantum state bipartition.\n"
        "Edge Modes: Contribution from gauge degrees of freedom at the entangling surface.\n"
        "UV Correction: Regularization effects from the UV cutoff.\n"
        "The distribution reveals how different formulations handle the entropy-geometry relationship."
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
    columns = ['Formulation', 'Final Loss', 'Area Coeff.', 'R²', 'S/A Ratio', 'Ricci Scalar', 
              'Bulk Entropy', 'Edge Modes', 'Total Entropy']
    
    for form in formulations:
        result = results.get(form, {})
        analysis = analyses.get(form, {})
        
        final_loss = 0.0
        if 'history' in result and 'total_loss' in result['history'] and result['history']['total_loss']:
            final_loss = result['history']['total_loss'][-1]
        
        data.append([
            form.capitalize(),
            f"{final_loss:.6f}",
            f"{analysis['area_law']['area_law_coefficient']:.4f}",
            f"{analysis['area_law']['r_squared']:.4f}",
            f"{analysis['holographic']['entropy_area_ratio']:.4f}",
            f"{analysis['holographic']['ricci_scalar']:.4f}",
            f"{analysis['entropy_components']['bulk']:.4f}",
            f"{analysis['entropy_components']['edge_modes']:.4f}",
            f"{analysis['entropy_components']['total']:.4f}"
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
            if key[1] == 0:  # Formulation column
                cell.set_text_props(weight='bold')
    
    plt.title('Stress Tensor Formulation Comparison', fontweight='bold', fontsize=20, pad=20)
    
    # Add explanatory text
    explanation = (
        "This table summarizes key metrics across different stress tensor formulations.\n"
        "Compare how different formulations affect the entropic-geometric coupling and area law relationship.\n"
        "The modified formulation includes corrections for non-conformal fields and edge modes."
    )
    plt.figtext(0.5, 0.05, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(plot_dir / 'formulation_summary.png', dpi=300)
    
    # Create radar chart for overall comparison
    categories = ['Area Law Fit (R²)', 'Loss Performance', 'Entropy/Area Ratio', 'Edge Mode Integration']
    
    # Normalize values for radar chart
    r_squared_norm = [analyses[f]['area_law']['r_squared'] for f in formulations]
    
    # For loss, lower is better, so we invert and normalize
    loss_values = []
    for form in formulations:
        if 'history' in results[form] and results[form]['history']['total_loss']:
            loss_values.append(1.0 / (results[form]['history']['total_loss'][-1] + 0.01))
        else:
            loss_values.append(0.0)
    
    max_loss = max(loss_values) if loss_values and max(loss_values) > 0 else 1.0
    loss_norm = [val / max_loss if max_loss > 0 else 0.0 for val in loss_values]
    
    # For entropy/area ratio, closer to 1 is better
    entropy_area_norm = [1.0 - abs(analyses[f]['holographic']['entropy_area_ratio'] - 1.0) for f in formulations]
    
    # For edge mode integration, we use a simple metric based on edge mode contribution
    edge_mode_norm = [analyses[f]['entropy_components']['edge_modes'] / analyses[f]['entropy_components']['total'] 
                      if analyses[f]['entropy_components']['total'] > 0 else 0 
                      for f in formulations]
    
    # Combine metrics
    values = np.array([r_squared_norm, loss_norm, entropy_area_norm, edge_mode_norm]).T
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of variables
    N = len(categories)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Plot for each formulation
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for i, form in enumerate(formulations):
        values_form = values[i].tolist()
        values_form += values_form[:1]  # Close the loop
        
        ax.plot(angles, values_form, linewidth=2, linestyle='solid', label=form.capitalize(), color=colors[i])
        ax.fill(angles, values_form, alpha=0.1, color=colors[i])
    
    # Set category labels
    plt.xticks(angles[:-1], categories, fontweight='bold')
    
    # Remove radial labels
    ax.set_yticklabels([])
    
    # Add legend
    if formulations:  # Only add legend if we have data
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.title('Stress Tensor Formulation Performance Comparison', fontweight='bold', pad=20)
    
    # Add explanatory text
    explanation = (
        "This radar chart compares the overall performance of different stress tensor formulations.\n"
        "Area Law Fit: How well the formulation conforms to the expected area law relationship.\n"
        "Loss Performance: How effectively the formulation minimizes the entropic field equations.\n"
        "Entropy/Area Ratio: How close the ratio is to the expected value of 1.0.\n"
        "Edge Mode Integration: How well the formulation incorporates boundary effects."
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(plot_dir / 'radar_comparison.png', dpi=300)
    
    print(f"Stress tensor comparison plots saved to {plot_dir}")


def main():
    """Main function to run the comparison."""
    import argparse
    parser = argparse.ArgumentParser(description='Compare stress tensor formulations')
    parser.add_argument('--input', type=str, default='results/stress_tensors',
                        help='Input directory containing stress tensor simulation results')
    parser.add_argument('--output', type=str, default='results/stress_tensors/comparison',
                        help='Output directory for comparison plots')
    args = parser.parse_args()
    
    # Load results
    results, analyses = load_results(args.input)
    
    # Generate comparison
    compare_stress_tensors(results, analyses, args.output)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
