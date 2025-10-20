# EntropicUnification Dashboard

This document describes the EntropicUnification Dashboard, a web-based interface for controlling simulations and visualizing results.

## Overview

The EntropicUnification Dashboard provides an intuitive interface for:

1. Setting up and running simulations
2. Visualizing simulation results
3. Understanding the theoretical concepts behind the framework

## Getting Started

To start the dashboard, run:

```bash
python dashboards/run_dashboard.py
```

This script will automatically determine which dashboard version to run based on the available dependencies:

- If the core modules are available, it will run the full dashboard (`app.py`)
- If the core modules are not available, it will run the standalone dashboard (`standalone_app.py`)

Then open your browser and navigate to `http://localhost:8050`.

### Alternative Methods

You can also run a specific dashboard version directly:

```bash
# Run the full dashboard (requires core modules)
cd dashboards
python app.py

# Run the standalone dashboard (no dependencies on core modules)
cd dashboards
python standalone_app.py
```

## Dashboard Structure

The dashboard is organized into three main tabs:

### 1. Control Console

The Control Console allows you to:

- Configure quantum parameters (number of qubits, circuit depth, initial state)
- Configure spacetime parameters (dimensions, lattice size, stress tensor formulation)
- Configure optimization parameters (steps, learning rate, strategy)
- Enable/disable experimental features (edge modes, non-conformal matter, higher curvature terms)
- Run, stop, and reset simulations
- Monitor simulation progress
- Load previous simulation results

### 2. Results Dashboard

The Results Dashboard displays:

- Loss curves showing optimization progress
- Entropy vs Area plot demonstrating the area law relationship
- Entropy components pie chart
- Metric evolution heatmaps
- Summary table with key metrics

You can customize the plots by:
- Changing the plot style
- Selecting specific plot types
- Downloading plots in different formats

### 3. Explanations

The Explanations tab provides detailed information about:

- Framework Overview: The basic concepts and components of EntropicUnification
- Quantum Concepts: Explanations of quantum states, entanglement entropy, and edge modes
- Geometric Concepts: Explanations of spacetime metrics, curvature tensors, and higher curvature terms
- Results Interpretation: How to interpret the various plots and metrics
- Theoretical Background: The scientific context and significance of the framework

## Features

### Real-time Monitoring

The dashboard provides real-time monitoring of simulation progress, including:

- Progress bar showing completion percentage
- Status messages
- Live updates of loss values

### Interactive Visualizations

All visualizations are interactive, allowing you to:

- Zoom in/out
- Pan
- Hover for detailed information
- Toggle visibility of different data series
- Download as images

### Result Management

The dashboard makes it easy to manage simulation results:

- Load previous simulation results
- Compare results across different runs
- Export data for further analysis

## Configuration

The dashboard uses the same configuration system as the core framework, with additional options for visualization. You can:

- Set default parameters in `data/configs.yaml`
- Override parameters through the user interface
- Save and load custom configurations

## Technical Details

The dashboard is built using:

- Dash: A Python framework for building web applications
- Plotly: For interactive visualizations
- Bootstrap: For responsive layout and styling

The dashboard architecture follows a modular design:

- `app.py`: Main application entry point
- `components/`: UI components for different dashboard sections
- `utils/`: Utilities for simulation and result management

## Best Practices

When using the dashboard:

1. Start with simple configurations (few qubits, small lattice size) to ensure quick execution
2. Use the explanations tab to understand the theoretical concepts
3. Save interesting results for later comparison
4. Experiment with different parameter combinations to explore the framework's capabilities

## Troubleshooting

If you encounter issues:

- Check the terminal for error messages
- Ensure all dependencies are installed (`pip install -r requirements.txt`)
- Verify that the core framework is functioning correctly
- Try restarting the dashboard server

## Future Enhancements

Planned enhancements for the dashboard include:

- User authentication and result sharing
- Advanced comparison tools for multiple simulations
- Integration with external compute resources for larger simulations
- Export of animations showing metric evolution
- Integration with experiment tracking platforms like Weights & Biases
