# EntropicUnification Unified Plotting System

This document describes the unified plotting system for EntropicUnification, which ensures that all plots are saved consistently, organized properly, and include metadata for traceability.

## Overview

The unified plotting system is implemented in `core/utils/plotting.py` and provides a consistent interface for creating and saving plots across different simulations and analyses. The system:

1. Organizes plots by simulation type and timestamp
2. Adds metadata to plots for better traceability
3. Provides a consistent look and feel for all plots
4. Supports configuration through a YAML file

## Directory Structure

Plots are saved in a consistent directory structure:

```
results/
  ├── [simulation_type]/
  │   ├── [timestamp]/
  │   │   ├── plots/
  │   │   │   ├── loss_curves.png
  │   │   │   ├── entropy_area.png
  │   │   │   ├── entropy_components.png
  │   │   │   ├── metric_evolution.png
  │   │   │   └── simulation_summary.png
  │   │   └── plot_metadata.json
  │   └── ...
  └── ...
```

- `simulation_type`: Type of simulation (e.g., 'bell', 'ghz', 'random')
- `timestamp`: Timestamp of the simulation run (format: YYYYMMDD_HHMMSS)

## Usage

### Basic Usage

```python
from core.utils.plotting import get_plot_manager

# Get a plot manager
plot_manager = get_plot_manager(config, output_dir)

# Generate all plots
plot_paths = plot_manager.plot_all(results, analysis, config, simulation_type)
```

### Custom Plots

```python
# Get a plot manager
plot_manager = get_plot_manager(config, output_dir)

# Create a custom plot
fig, ax = plt.subplots(figsize=(10, 8))
# ... create your plot ...

# Save the plot with metadata
metadata = {"custom_value": 42}
plot_manager.save_plot(fig, "custom_plot", simulation_type, metadata)
```

## Configuration

The plotting system can be configured through `data/plotting_config.yaml`. This file allows you to customize:

- Plot styles
- Colors
- Sizes
- Directory structure
- Metadata settings

Example configuration:

```yaml
# General plotting settings
general:
  style: "seaborn-v0_8-whitegrid"
  dpi: 300
  save_format: "png"
  font_family: "sans-serif"
  font_size: 12
  title_size: 18
  
# Directory structure settings
directories:
  use_timestamp: true
  timestamp_format: "%Y%m%d_%H%M%S"
```

## Metadata

Each plot is saved with metadata in a JSON file (`plot_metadata.json`). This metadata includes:

- Timestamp
- Configuration summary
- Plot-specific metadata (e.g., loss values, R² values)

Example metadata:

```json
{
  "timestamp": "20251020_002530",
  "config": {
    "quantum": {
      "num_qubits": 4,
      "circuit_depth": 4
    },
    "spacetime": {
      "dimensions": 4,
      "lattice_size": 64
    },
    "coupling": {
      "stress_form": "jacobson"
    }
  },
  "plots": {
    "loss_curves": {
      "name": "loss_curves",
      "path": "results/bell/20251020_002530/plots/loss_curves.png",
      "timestamp": "20251020_002530",
      "simulation_type": "bell",
      "iterations": 100,
      "final_loss": 0.00123,
      "best_loss": 0.00098
    },
    "entropy_area": {
      "name": "entropy_area",
      "path": "results/bell/20251020_002530/plots/entropy_area.png",
      "timestamp": "20251020_002530",
      "simulation_type": "bell",
      "coefficient": 0.2534,
      "intercept": 0.0123,
      "r_squared": 0.9876,
      "num_data_points": 10
    }
  }
}
```

## Available Plot Types

The following plot types are available:

1. **Loss Curves**: Shows the optimization progress of the entropic field equations
2. **Entropy vs Area**: Shows the relationship between entanglement entropy and boundary area
3. **Entropy Components**: Shows the relative contributions to the total entanglement entropy
4. **Metric Evolution**: Shows the evolution of the spacetime metric tensor during optimization
5. **Simulation Summary**: Shows a summary of key metrics from the simulation

## Comparison Tools

The plotting system also includes tools for comparing results across different runs:

- Compare different stress tensor formulations
- Compare different initial quantum states
- Compare different optimization strategies

## Customization

You can customize the plotting system by:

1. Modifying `data/plotting_config.yaml`
2. Extending the `PlotManager` class with new plot types
3. Creating custom plots and saving them with `plot_manager.save_plot()`

## Best Practices

1. Always use the `PlotManager` for saving plots
2. Include relevant metadata with each plot
3. Use consistent naming conventions for plots
4. Use the `plot_all()` method to generate all standard plots
5. Use the configuration file to customize plot appearance
