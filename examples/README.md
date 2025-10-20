# EntropicUnification Examples

This directory contains example scripts demonstrating the EntropicUnification framework.

## Main Examples

### 1. Entropic Simulation (`entropic_simulation.py`)

The primary example that demonstrates a complete simulation using the EntropicUnification framework.

```bash
python examples/entropic_simulation.py --config data/configs.yaml --output results/my_simulation
```

Options:
- `--config`: Path to configuration file (default: data/configs.yaml)
- `--output`: Output directory for results (default: results)
- `--state`: Initial quantum state type (choices: bell, ghz, random, all; default: bell)
- `--stress`: Stress tensor formulation to use (choices: jacobson, canonical, faulkner, modified)
- `--iterations`: Override number of optimization iterations

### 2. Compare Stress Tensors (`compare_stress_tensors.py`)

Compares different formulations of the entropic stress-energy tensor.

```bash
python examples/compare_stress_tensors.py --input results/stress_comparison --output results/comparison
```

Options:
- `--input`: Directory containing simulation results for different stress tensor formulations
- `--output`: Output directory for comparison plots

### 3. Test Original Geometry (`test_original_geometry.py`)

Tests the geometry engine with different metric configurations.

```bash
python examples/test_original_geometry.py --output results/geometry_test
```

Options:
- `--output`: Output directory for test results (default: results/original_geometry)
- `--metric`: Metric type to test (choices: schwarzschild, wave; default: both)

### 4. Advanced Optimization (`advanced_optimization.py`)

Demonstrates advanced optimization techniques.

```bash
python examples/advanced_optimization.py --optimizer adam --lr_schedule cosine
```

Options:
- `--optimizer`: Optimizer type (choices: sgd, adam, rmsprop; default: adam)
- `--lr_schedule`: Learning rate schedule (choices: constant, step, exponential, cosine; default: constant)

## 5. Simple Simulation (`simple_simulation.py`)

A simplified example for quick experimentation.

```bash
python examples/simple_simulation.py
```

## 6. Enhanced Concepts (`enhanced_concepts.py`)

Demonstrates key concepts of the enhanced framework.

```bash
python examples/enhanced_concepts.py --concept finite_difference
```

Options:
- `--concept`: Concept to demonstrate (choices: finite_difference, bianchi, optimization; default: all)

## Output

All examples save their output to the `results/` directory by default. This includes:
- Figures and plots
- Metric evolution data
- Entropy and curvature data
- Performance metrics

## Configuration

Examples use the configuration file at `data/configs.yaml` by default. You can modify this file or provide a custom configuration file using the `--config` option.