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

- If the core modules and enhanced components are available, it will run the enhanced dashboard (`enhanced_app.py`)
- If only the core modules are available, it will run the full dashboard (`app.py`)
- If only the enhanced components are available, it will run the enhanced standalone dashboard (`enhanced_app.py`)
- If neither are available, it will run the basic standalone dashboard (`standalone_app.py`)

Then open your browser and navigate to `http://localhost:8050`.

### Command Line Options

You can specify which dashboard version to run using the `--version` flag:

```bash
# Auto-detect the best version to run (default)
python dashboards/run_dashboard.py --version auto

# Run the full dashboard (requires core modules)
python dashboards/run_dashboard.py --version full

# Run the standalone dashboard (no dependencies on core modules)
python dashboards/run_dashboard.py --version standalone

# Run the enhanced dashboard (with improved UI and features)
python dashboards/run_dashboard.py --version enhanced
```

### Alternative Methods

You can also run a specific dashboard version directly:

```bash
# Run the full dashboard (requires core modules)
cd dashboards
python app.py

# Run the standalone dashboard (no dependencies on core modules)
cd dashboards
python standalone_app.py

# Run the enhanced dashboard (with improved UI and features)
cd dashboards
python enhanced_app.py
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

### Basic Dashboard

The dashboard consists of three main sections:

1. **Control Console**: Configure and run simulations
2. **Results Dashboard**: Visualize simulation results
3. **Explanations**: Learn about the theoretical concepts

### Enhanced Dashboard

The enhanced dashboard includes all the features of the basic dashboard, plus:

1. **Settings Panel**: Customize the dashboard appearance and behavior
   - Theme switching (light/dark mode)
   - Plot style selection
   - Download format options
   - Refresh interval settings

2. **Interactive Plots**: Enhanced visualization features
   - Detailed tooltips and annotations
   - Download options for each plot
   - Explanatory text for each visualization
   - Improved styling and responsiveness

3. **Help System**: Contextual help throughout the interface
   - Help tooltips for each parameter
   - Detailed explanations of concepts
   - Quick reference guides

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
