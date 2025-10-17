# EntropicUnification - Quick Start Guide

## Overview

EntropicUnification is a computational framework that explores how spacetime geometry emerges from quantum entanglement. This guide will help you get started quickly.

## Installation (5 minutes)

1. **Navigate to the project directory:**
```bash
cd ~/Desktop/EntropicUnification
```

2. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Your First Simulation (10 minutes)

### Option 1: Run the Example Notebook

```bash
jupyter notebook notebooks/experiments.ipynb
```

Then execute cells sequentially to see:
- Quantum state evolution
- Entropy calculations
- Geometric coupling
- Real-time visualization

### Option 2: Python Script

Create a file `run_simulation.py`:

```python
import torch
import yaml
from core.quantum_engine import QuantumEngine
from core.geometry_engine import GeometryEngine
from core.entropy_module import EntropyModule
from core.coupling_layer import CouplingLayer
from core.loss_functions import LossFunctions
from core.optimizer import EntropicOptimizer

# Load configuration
with open('data/configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize system
NUM_QUBITS = 4
quantum_engine = QuantumEngine(NUM_QUBITS, circuit_depth=3)
geometry_engine = GeometryEngine(dimensions=4, lattice_size=32)
entropy_module = EntropyModule(quantum_engine)
coupling_layer = CouplingLayer(geometry_engine, entropy_module)
loss_functions = LossFunctions(coupling_layer)

optimizer = EntropicOptimizer(
    quantum_engine, geometry_engine, entropy_module,
    coupling_layer, loss_functions
)

# Prepare initial state (Bell state for 2 qubits, extended)
initial_state = torch.zeros(2**NUM_QUBITS)
initial_state[0] = 1/torch.sqrt(torch.tensor(2.0))
initial_state[-1] = 1/torch.sqrt(torch.tensor(2.0))
initial_state.requires_grad = True

# Define partition and target
partition = list(range(NUM_QUBITS // 2))
target_gradient = torch.randn_like(initial_state)
target_gradient = target_gradient / torch.norm(target_gradient)

# Run optimization
print("Starting entropic optimization...")
results = optimizer.train(
    initial_state,
    partition,
    target_gradient,
    n_steps=100,  # Short run for demo
    learning_rate=1e-3
)

print(f"\nTraining completed in {results['training_time']:.2f} seconds")
print(f"Final loss: {results['history']['total_loss'][-1]:.6f}")
print("\nFinal metric tensor:")
print(results['final_metric'])
```

Run with:
```bash
python run_simulation.py
```

## Understanding the Results

After running a simulation, check the `results/` directory:

- **`training_logs.json`**: Loss evolution over time
- **`checkpoint_*.pt`**: Saved states at intervals
- **`final_results.pt`**: Complete final state

### Loading and Analyzing Results

```python
import torch
import json
import matplotlib.pyplot as plt

# Load training history
with open('results/training_logs.json', 'r') as f:
    history = json.load(f)

# Plot loss evolution
plt.figure(figsize=(10, 6))
plt.plot(history['total_loss'], label='Total Loss')
plt.plot(history['einstein_loss'], label='Einstein Loss')
plt.plot(history['entropy_loss'], label='Entropy Loss')
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.title('Entropic Optimization Progress')
plt.savefig('results/loss_curves.png')
plt.show()

# Load final results
results = torch.load('results/final_results.pt')
print("Final metric:")
print(results['final_metric'])
```

## Key Concepts

### 1. Quantum Layer
- Prepares and evolves quantum states
- Generates entanglement between qubits

### 2. Entropy Module
- Computes von Neumann entropy
- Calculates entropy gradients

### 3. Geometry Layer
- Represents spacetime metric tensor
- Computes curvature (Riemann, Ricci, Einstein tensors)

### 4. Coupling Layer
- Links entropy gradients to geometric curvature
- Enforces consistency: G_μν ∝ ∇S_ent

### 5. Optimizer
- Minimizes inconsistency between quantum and geometric layers
- Uses gradient descent to find optimal metric

## Customization

### Adjust Quantum System Size

In `data/configs.yaml`:
```yaml
quantum:
  num_qubits: 6  # Change from 4 to 6
  circuit_depth: 4  # Deeper circuits
```

### Tune Optimization

```yaml
optimization:
  learning_rate: 5.0e-4  # Smaller for stability
  max_iterations: 5000
  loss_weights:
    einstein: 2.0  # Emphasize Einstein constraint
    entropy: 1.0
    regularity: 0.1
```

### Change Physical Constants

```yaml
constants:
  gravitational_constant: 6.67430e-11
  reduced_planck_constant: 1.054571817e-34
```

## Common Issues

### Issue: "Out of memory"
**Solution**: Reduce `num_qubits` (each additional qubit doubles memory usage)

### Issue: "Loss not converging"
**Solution**: 
- Decrease `learning_rate`
- Increase `regularity` weight
- Check initial state is normalized

### Issue: "NaN values in metric"
**Solution**:
- Add metric regularization
- Check for degenerate eigenvalues
- Enable gradient clipping

## Next Steps

1. **Read the White Paper**: See [WHITEPAPER.md](WHITEPAPER.md) for theoretical details

2. **Explore Examples**: Check `notebooks/experiments.ipynb` for more experiments

3. **Modify Core Modules**: Extend quantum circuits, add custom loss functions

4. **Visualize Results**: Create plots of entropy vs. curvature evolution

5. **Run Longer Simulations**: Increase `max_iterations` for convergence

## Scientific Context

This framework tests the hypothesis that:
> **Spacetime geometry emerges from quantum entanglement through an optimization process that minimizes entropic imbalance.**

Key ideas:
- Quantum entanglement entropy → geometric curvature
- Einstein equations emerge from entropy gradients
- Universe as an information-processing optimizer

## Support

For detailed documentation:
- **Theoretical**: [WHITEPAPER.md](WHITEPAPER.md)
- **Technical**: [README.md](README.md)
- **API**: Check docstrings in `core/` modules

## Citation

If you use this framework in your research, please cite:

```
EntropicUnification: A Differentiable Framework for Learning 
Spacetime Geometry from Quantum Entanglement (2025)
```

---

**Happy exploring the quantum origins of spacetime!** 🚀✨
