# EntropicUnification Examples

This directory contains example scripts that demonstrate the EntropicUnification framework. These examples showcase the enhanced capabilities of the framework, including support for edge modes, non-conformal matter fields, and higher-order curvature corrections.

## Available Examples

### 1. Basic Entropic Simulation (`entropic_simulation.py`)

This script demonstrates the core functionality of the EntropicUnification framework. It:

- Sets up the quantum, geometric, and entropic components
- Runs an optimization to find a spacetime metric consistent with quantum entanglement
- Analyzes the results, including area law behavior and holographic metrics
- Generates visualizations of the optimization process

#### Usage

```bash
python entropic_simulation.py --config ../data/configs.yaml --output ../results --state bell
```

#### Parameters

- `--config`: Path to the configuration file (default: `data/configs.yaml`)
- `--output`: Directory to save results (default: `results`)
- `--state`: Initial quantum state type (choices: `bell`, `ghz`, `random`, default: `bell`)

#### Output

The script generates:

- Training logs and checkpoints in the specified output directory
- Analysis of convergence, area law behavior, and holographic metrics
- Plots of loss curves, entropy-area relationship, and metric evolution

## Key Concepts Demonstrated

### Edge Modes

Edge modes represent gauge degrees of freedom that become physical at boundaries. The framework now properly accounts for these contributions to entanglement entropy, which are especially important for gauge fields and gravitons.

```python
# Configure edge mode parameters
entropy_module.set_edge_mode_parameters(
    dimension=config['entropy']['edge_mode_dimension'],
    entropy_factor=config['entropy']['edge_mode_entropy_factor']
)
```

### Non-Conformal Matter Fields

The framework now handles non-conformal matter fields, which require special treatment in the entropic stress-energy tensor. This addresses Speranza's observations about the subtleties in deriving Einstein equations from entanglement equilibrium.

```python
# Set stress tensor formulation that handles non-conformal fields
coupling_layer.set_stress_tensor_formulation("modified")
```

### Higher-Order Curvature Terms

Support for higher-order curvature corrections, including Gauss-Bonnet terms and Weyl tensor contributions, allows for more accurate modeling of spacetime geometry beyond the Einstein tensor.

```python
# Enable higher-order curvature terms
coupling_layer.set_higher_curvature_parameters(
    alpha_gb=0.1,  # Gauss-Bonnet coupling
    lambda_cosmo=0.001  # Cosmological constant
)
```

### Advanced Optimization Strategies

The framework includes several optimization strategies to handle the complex loss landscape:

- **Basin Hopping**: Helps escape local minima by occasionally perturbing the metric
- **Simulated Annealing**: Gradually reduces temperature to balance exploration and exploitation
- **Adaptive Learning Rates**: Adjusts learning rates based on optimization progress

```python
# Configure optimizer for basin hopping
optimizer_config = OptimizerConfig(
    optimization_strategy="basin_hopping",
    basin_hopping_params={
        "hop_threshold": 50,
        "max_hops": 5,
        "temperature": 0.1,
    }
)
```

## Theoretical Background

These examples implement the insights from recent research on entanglement-gravity connections:

1. **Jacobson's Thermodynamic Derivation**: Einstein equations can emerge from thermodynamic principles applied to entanglement entropy.

2. **Ryu-Takayanagi Formula**: Entanglement entropy in holographic theories is proportional to the area of minimal surfaces in the bulk.

3. **Faulkner's Linearized Einstein Dynamics**: The first law of entanglement implies linearized Einstein equations.

4. **Speranza's Analysis**: Deriving Einstein equations from entanglement requires careful treatment of non-conformal matter and edge modes.

## Extending the Examples

You can modify these examples to explore different aspects of the framework:

- Change the quantum state preparation to investigate different entanglement structures
- Modify the geometric setup to explore different spacetime topologies
- Adjust the optimization parameters to improve convergence
- Implement custom loss functions to test different physical hypotheses

## References

For more details on the theoretical background and implementation, see the [WHITEPAPER.md](../WHITEPAPER.md) and [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) files in the main directory.
