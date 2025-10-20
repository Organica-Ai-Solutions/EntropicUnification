# EntropicUnification Dashboard User Guide

This guide explains how to use the EntropicUnification Dashboard to run simulations, visualize results, and gain insights into the quantum-geometric relationship.

## Getting Started

### Running the Dashboard

To start the dashboard, run the following command from the project root directory:

```bash
python dashboards/run_fixed_dashboard.py
```

This script will:
1. Kill any existing dashboard processes
2. Find an available port (starting from 8050)
3. Start the dashboard server
4. Open a browser window with the dashboard

The terminal will show the URL where the dashboard is running (typically http://127.0.0.1:8050/).

### Dashboard Layout

The dashboard is organized into several tabs:

1. **Control Console** - Configure and run simulations
2. **Results Dashboard** - View simulation results and plots
3. **Advanced Visualizations** - Explore detailed visualizations of quantum states and geometry
4. **Real-Time Monitoring** - Monitor simulation progress and system resources
5. **Explanations** - Learn about the EntropicUnification framework

## Control Console

The Control Console is where you configure and run simulations.

### Quantum Parameters

- **Number of Qubits** - Set the number of qubits in the quantum system (2-10)
- **Circuit Depth** - Set the depth of the quantum circuit (1-10)
- **Initial State** - Choose the initial quantum state (Bell, GHZ, or Random)

### Spacetime Parameters

- **Dimensions** - Set the number of spacetime dimensions (2-4)
- **Lattice Size** - Set the size of the spatial lattice (10-100)
- **Stress Tensor Form** - Choose the formulation of the stress-energy tensor:
  - **Jacobson** - Original Jacobson thermodynamic formulation
  - **Canonical** - Simple outer product of gradients
  - **Faulkner** - Faulkner's linearized Einstein formulation
  - **Modified** - Modified formulation with edge mode corrections

### Optimization Parameters

- **Optimization Steps** - Set the number of optimization steps (100-10000)
- **Learning Rate** - Set the learning rate for optimization (0.0001-0.1)
- **Optimization Strategy** - Choose the optimization strategy:
  - **Standard** - Standard gradient descent
  - **Adaptive** - Adaptive learning rate
  - **Annealed** - Simulated annealing
  - **Basin Hopping** - Basin hopping for exploring multiple minima

### Experimental Features

- **Edge Modes** - Include edge mode contributions in entropy calculations
- **Non-Conformal Matter** - Include non-conformal matter field corrections
- **Higher Curvature** - Include higher-order curvature terms

### Running a Simulation

1. Configure the parameters according to your needs
2. Click the "Run Simulation" button
3. Monitor the progress bar and status messages
4. When the simulation completes, switch to the Results Dashboard tab

### Loading Previous Results

1. Select a result directory from the dropdown
2. Click the "Load Results" button
3. The results will be loaded and displayed in the Results Dashboard

## Results Dashboard

The Results Dashboard displays the results of the simulation.

### Loss Curves

This plot shows the convergence of the loss function during optimization. It includes:
- **Total Loss** - Overall optimization objective
- **Einstein Loss** - Mismatch between Einstein tensor and stress tensor
- **Entropy Loss** - Deviation from target entropy gradient
- **Curvature Loss** - Regularization term for curvature
- **Smoothness Loss** - Regularization term for metric smoothness

### Entropy-Area Relationship

This plot shows the relationship between entanglement entropy and boundary area, which is a key aspect of the holographic principle. The plot includes:
- **Data Points** - Entropy values for different boundary areas
- **Linear Fit** - Best-fit line showing the area law relationship
- **Area Law Coefficient** - Slope of the best-fit line
- **R² Value** - Goodness of fit

### Entropy Components

This plot shows the breakdown of entropy contributions from different sources:
- **Bulk Entropy** - Entropy from the bulk quantum state
- **Edge Modes** - Entropy contribution from edge modes
- **UV Correction** - Entropy contribution from UV regularization

### Metric Evolution

This plot shows how the spacetime metric evolves during optimization:
- **Metric Components** - Values of the metric tensor components
- **Initial Metric** - Starting metric configuration
- **Final Metric** - Optimized metric configuration

### Plot Controls

- **Download** - Download plots in various formats (PNG, SVG, PDF)
- **Plot Style** - Change the visual style of the plots
- **Interactive Plots** - Toggle interactive features (zooming, panning, etc.)

## Advanced Visualizations

The Advanced Visualizations tab provides more detailed visualizations of the quantum-geometric relationship.

### 3D Entropy Distribution

This visualization shows the spatial distribution of entanglement entropy across the lattice.

### Spacetime Diagram

This visualization shows the spacetime geometry, including:
- **Light Cones** - Future and past light cones
- **Worldlines** - Paths of particles through spacetime
- **Geodesics** - Shortest paths in curved spacetime

### Quantum State Visualization

This visualization shows the quantum state in terms of:
- **Probabilities** - Probabilities of different basis states
- **Phases** - Quantum phases of different basis states

### Entanglement Network

This visualization shows the network of entanglement between qubits:
- **Nodes** - Individual qubits
- **Edges** - Entanglement connections between qubits
- **Edge Colors** - Strength of entanglement

## Real-Time Monitoring

The Real-Time Monitoring tab allows you to monitor the simulation progress and system resources.

### Metrics Graph

This graph shows real-time metrics during simulation:
- **Loss** - Current value of the loss function
- **Entropy** - Current value of the entanglement entropy
- **Gradient Norm** - Norm of the gradient

### System Monitor

This panel shows system resource usage:
- **CPU Usage** - Percentage of CPU being used
- **Memory Usage** - Percentage of memory being used
- **GPU Usage** - Percentage of GPU being used (if available)
- **Disk I/O** - Disk input/output activity

### Simulation Log

This panel shows a log of simulation events and messages.

## Settings Panel

The Settings Panel (accessible via the gear icon) allows you to customize the dashboard:

### Theme

- **Dark Mode** - Toggle between light and dark themes

### Refresh Settings

- **Auto Refresh** - Toggle automatic refreshing of data
- **Refresh Interval** - Set the interval for data refreshing (in seconds)

### Plot Settings

- **Plot Style** - Choose the default style for all plots
- **Download Format** - Choose the default format for plot downloads

## Tips and Best Practices

1. **Start Simple** - Begin with a small number of qubits (2-4) and a small lattice size (10-20)
2. **Explore Different States** - Try different initial quantum states to see how they affect the results
3. **Compare Stress Tensor Forms** - Compare different stress tensor formulations to see how they affect the coupling
4. **Use Adaptive Optimization** - For complex simulations, use adaptive optimization strategies
5. **Monitor Convergence** - Keep an eye on the loss curves to ensure the simulation is converging
6. **Save Interesting Results** - Use the plot download feature to save interesting results
7. **Experiment with Parameters** - Try different parameter combinations to explore the parameter space

## Troubleshooting

### Dashboard Won't Start

- Ensure all required packages are installed: `pip install -r requirements.txt`
- Check if another process is using the default port (8050)
- Try running with a specific port: `python dashboards/run_fixed_dashboard.py --port 8051`

### Simulation Errors

- Check the simulation log for error messages
- Try reducing the number of qubits or lattice size
- Try a different optimization strategy
- Check the system monitor to ensure you have enough resources

### Visualization Issues

- If plots are not displaying, try refreshing the page
- If 3D visualizations are slow, try reducing the lattice size
- If you see React errors in the console, try restarting the dashboard

### Performance Issues

- For large simulations, increase the memory available to Python
- For complex visualizations, use a computer with a dedicated GPU
- Consider running simulations in batch mode and loading results later

## Advanced Usage

### Custom Quantum States

You can create custom quantum states by modifying the `quantum_engine.py` file:

```python
def create_custom_state(num_qubits):
    # Create your custom state here
    return state
```

### Custom Stress Tensor Formulations

You can create custom stress tensor formulations by modifying the `coupling_layer.py` file:

```python
def compute_custom_stress_tensor(self, entropy_gradient, metric):
    # Compute your custom stress tensor here
    return stress_tensor
```

### Batch Simulations

For running multiple simulations with different parameters, you can use the `examples/batch_simulation.py` script:

```bash
python examples/batch_simulation.py --config batch_config.yaml
```

## Further Reading

For more information about the EntropicUnification framework, see the following documentation:

- [README.md](../README.md) - Overview of the project
- [WHITEPAPER.md](../WHITEPAPER.md) - Detailed theoretical background
- [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - Codebase structure and architecture
- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide
