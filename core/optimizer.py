"""
Training loop for the entropic-unification engine with enhanced optimization capabilities.

This module provides the main optimization loop for the EntropicUnification framework,
coordinating the interaction between quantum, geometric, and entropic components.
The enhanced version includes:

- Support for advanced optimization strategies
- Improved convergence monitoring
- Adaptive learning rates
- Multiple partition strategies
- Comprehensive logging and visualization
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from tqdm import tqdm

from .coupling_layer import CouplingLayer, StressTensorFormulation
from .entropy_module import EntropyModule
from .geometry_engine import GeometryEngine, BoundaryCondition
from .loss_functions import LossFunctions, LossFormulation
from .quantum_engine import QuantumEngine


class OptimizationStrategy(str, Enum):
    """Different optimization strategies."""
    STANDARD = "standard"     # Standard gradient descent
    ADAPTIVE = "adaptive"     # Adaptive learning rate
    ANNEALED = "annealed"     # Simulated annealing
    BASIN_HOPPING = "basin_hopping"  # Basin hopping


class PartitionStrategy(str, Enum):
    """Different strategies for choosing partitions."""
    FIXED = "fixed"           # Use a fixed partition
    ROTATING = "rotating"     # Rotate through different partitions
    RANDOM = "random"         # Randomly select partitions
    ADAPTIVE = "adaptive"     # Adaptively choose partitions


@dataclass
class OptimizerConfig:
    """Configuration for the EntropicOptimizer."""
    learning_rate: float = 1e-3
    steps: int = 1000
    checkpoint_interval: int = 100
    log_interval: int = 10
    metric_grad_clip: Optional[float] = 10.0
    results_path: str = "results"
    
    # Enhanced optimization parameters
    optimization_strategy: Union[str, OptimizationStrategy] = OptimizationStrategy.STANDARD
    partition_strategy: Union[str, PartitionStrategy] = PartitionStrategy.FIXED
    loss_formulation: Union[str, LossFormulation] = LossFormulation.STANDARD
    stress_formulation: Union[str, StressTensorFormulation] = StressTensorFormulation.JACOBSON
    
    # Learning rate schedule
    lr_schedule: Dict[str, float] = field(default_factory=lambda: {
        "initial_lr": 1e-3,
        "final_lr": 1e-5,
        "decay_rate": 0.95,
    })
    
    # Convergence criteria
    convergence_threshold: float = 1e-6
    patience: int = 50
    
    # Advanced features
    include_edge_modes: bool = True
    include_higher_curvature: bool = False
    conformal_invariance: bool = False
    
    # Regularization
    uv_cutoff: float = 1e-6
    regularization_scheme: str = "lattice"
    
    # Basin hopping parameters
    basin_hopping_params: Dict[str, float] = field(default_factory=lambda: {
        "hop_threshold": 50,
        "max_hops": 5,
        "temperature": 0.1,
    })
    
    # Annealing schedule
    annealing_schedule: Dict[str, float] = field(default_factory=lambda: {
        "initial_temp": 1.0,
        "final_temp": 0.01,
        "decay_rate": 0.95,
    })


class EntropicOptimizer:
    """Optimizer for the EntropicUnification framework with enhanced capabilities."""
    
    def __init__(
        self,
        quantum_engine: QuantumEngine,
        geometry_engine: GeometryEngine,
        entropy_module: EntropyModule,
        coupling_layer: CouplingLayer,
        loss_functions: LossFunctions,
        config: OptimizerConfig,
    ) -> None:
        """Initialize the entropic optimizer.
        
        Args:
            quantum_engine: Quantum engine for state evolution
            geometry_engine: Geometry engine for metric calculations
            entropy_module: Entropy module for entanglement calculations
            coupling_layer: Coupling layer connecting entropy and geometry
            loss_functions: Loss functions for optimization
            config: Optimizer configuration
        """
        self.quantum = quantum_engine
        self.geometry = geometry_engine
        self.entropy = entropy_module
        self.coupling = coupling_layer
        self.loss = loss_functions
        self.config = config
        
        # Convert string enums to enum types if needed
        if isinstance(config.optimization_strategy, str):
            self.optimization_strategy = OptimizationStrategy(config.optimization_strategy.lower())
        else:
            self.optimization_strategy = config.optimization_strategy
            
        if isinstance(config.partition_strategy, str):
            self.partition_strategy = PartitionStrategy(config.partition_strategy.lower())
        else:
            self.partition_strategy = config.partition_strategy
            
        # Configure components based on config
        self._configure_components()
        
        # Initialize history
        self.history: Dict[str, List[float]] = {
            "total_loss": [],
            "einstein_loss": [],
            "entropy_loss": [],
            "curvature_loss": [],
            "smoothness_loss": [],
            "consistency": [],
            "learning_rate": [],
            "temperature": [],
        }
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.best_metric = None
        self.steps_without_improvement = 0
        
        # Current learning rate
        self.current_lr = config.lr_schedule["initial_lr"]
        
        # Results path
        self.results_path = Path(config.results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Available partitions for rotating/random strategies
        self._generate_partition_options()

    def _configure_components(self) -> None:
        """Configure components based on optimizer config."""
        # Configure entropy module
        self.entropy.uv_cutoff = self.config.uv_cutoff
        self.entropy.include_edge_modes = self.config.include_edge_modes
        self.entropy.conformal_invariance = self.config.conformal_invariance
        self.entropy.regularization_scheme = self.config.regularization_scheme
        
        # Configure coupling layer
        self.coupling.include_edge_modes = self.config.include_edge_modes
        self.coupling.include_higher_curvature = self.config.include_higher_curvature
        self.coupling.conformal_invariance = self.config.conformal_invariance
        self.coupling.set_stress_tensor_formulation(self.config.stress_formulation)
        
        # Configure loss functions
        self.loss.set_formulation(self.config.loss_formulation)
        
        # Set up basin hopping if needed
        if self.optimization_strategy == OptimizationStrategy.BASIN_HOPPING:
            self.loss.set_basin_hopping(True)
            
        # Set up annealing if needed
        if self.optimization_strategy == OptimizationStrategy.ANNEALED:
            self.loss.set_annealing_schedule(self.config.annealing_schedule)

    def _generate_partition_options(self) -> None:
        """Generate partition options for different strategies."""
        num_qubits = self.quantum.num_qubits
        
        # Generate all possible bipartitions
        self.partition_options = []
        
        # Start with simple bipartitions (first k qubits)
        for k in range(1, num_qubits):
            self.partition_options.append(list(range(k)))
            
        # Add some more complex partitions if we have enough qubits
        if num_qubits >= 4:
            # Every other qubit
            self.partition_options.append(list(range(0, num_qubits, 2)))
            
            # First and last qubit
            self.partition_options.append([0, num_qubits-1])
        
        self.current_partition_idx = 0

    def _get_next_partition(self, current_partition: List[int]) -> List[int]:
        """Get the next partition based on the partition strategy.
        
        Args:
            current_partition: Current partition
            
        Returns:
            Next partition to use
        """
        if self.partition_strategy == PartitionStrategy.FIXED:
            # Keep using the same partition
            return current_partition
            
        elif self.partition_strategy == PartitionStrategy.ROTATING:
            # Rotate to the next partition
            self.current_partition_idx = (self.current_partition_idx + 1) % len(self.partition_options)
            return self.partition_options[self.current_partition_idx]
            
        elif self.partition_strategy == PartitionStrategy.RANDOM:
            # Choose a random partition
            return self.partition_options[np.random.randint(len(self.partition_options))]
            
        elif self.partition_strategy == PartitionStrategy.ADAPTIVE:
            # Choose partition based on current loss landscape
            # For simplicity, we'll rotate if we haven't improved for a while
            if self.steps_without_improvement > self.config.patience // 2:
                self.current_partition_idx = (self.current_partition_idx + 1) % len(self.partition_options)
                return self.partition_options[self.current_partition_idx]
            else:
                return current_partition
                
        # Default to current partition
        return current_partition

    def _update_learning_rate(self, step: int) -> None:
        """Update the learning rate based on the schedule.
        
        Args:
            step: Current optimization step
        """
        if self.optimization_strategy == OptimizationStrategy.ADAPTIVE:
            # Exponential decay schedule
            self.current_lr = self.config.lr_schedule["initial_lr"] * (
                self.config.lr_schedule["decay_rate"] ** (step / 100)
            )
            # Clamp to final learning rate
            self.current_lr = max(self.current_lr, self.config.lr_schedule["final_lr"])
            
        elif self.optimization_strategy == OptimizationStrategy.ANNEALED:
            # Use higher learning rate when temperature is high
            temp_factor = self.loss.temperature / self.config.annealing_schedule["initial_temp"]
            base_lr = self.config.lr_schedule["initial_lr"] * (
                self.config.lr_schedule["decay_rate"] ** (step / 100)
            )
            self.current_lr = base_lr * (1.0 + temp_factor)
            
            # Clamp to reasonable range
            min_lr = self.config.lr_schedule["final_lr"]
            max_lr = self.config.lr_schedule["initial_lr"] * 2.0
            self.current_lr = max(min(self.current_lr, max_lr), min_lr)

    def _check_convergence(self, loss: float) -> bool:
        """Check if optimization has converged.
        
        Args:
            loss: Current loss value
            
        Returns:
            True if converged, False otherwise
        """
        # Check if loss is below threshold
        if loss < self.config.convergence_threshold:
            return True
            
        # Check if we've improved
        if loss < self.best_loss:
            self.best_loss = loss
            self.steps_without_improvement = 0
            
            # Save best metric
            self.best_metric = self.geometry.metric.detach().clone()
        else:
            self.steps_without_improvement += 1
            
        # Check if we've gone too long without improvement
        if self.steps_without_improvement >= self.config.patience:
            # For basin hopping, we'll let it continue
            if self.optimization_strategy == OptimizationStrategy.BASIN_HOPPING:
                return False
            else:
                return True
                
        return False

    # ------------------------------------------------------------------
    # Training step and loop
    # ------------------------------------------------------------------
    def optimization_step(
        self,
        state: torch.Tensor,
        partition: List[int],
        target_gradient: Optional[torch.Tensor],
        weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        """Perform one optimization step.
        
        Args:
            state: Quantum state
            partition: Partition defining entanglement region
            target_gradient: Optional target entropy gradient
            weights: Loss weights
            
        Returns:
            Dictionary with optimization results
        """
        # Compute coupling terms
        terms = self.coupling.compute_coupling_terms(state, partition)
        
        # Compute losses
        losses = self.loss.total_loss(terms, target_gradient, weights)
        
        # Update coupling with current learning rate
        update_info = self.coupling.update_coupling(
            state,
            partition,
            learning_rate=self.current_lr,
            metric_grad_clip=self.config.metric_grad_clip,
        )
        
        # Combine results
        results = {**terms.__dict__, **losses, **update_info}
        results["learning_rate"] = torch.tensor(self.current_lr)
        
        return results

    def train(
        self,
        parameters: torch.Tensor,
        times: torch.Tensor,
        partition: List[int],
        initial_states: Optional[torch.Tensor] = None,
        target_gradient: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Train the model to find optimal metric.
        
        Args:
            parameters: Quantum circuit parameters
            times: Time points for evolution
            partition: Initial partition defining entanglement region
            initial_states: Optional initial quantum states
            target_gradient: Optional target entropy gradient
            weights: Optional loss weights
            
        Returns:
            Dictionary with training results
        """
        if weights is None:
            weights = {
                "einstein": 1.0,
                "entropy": 1.0,
                "curvature": 0.1,
                "smoothness": 0.1,
            }

        # Evolve quantum states
        states = self.quantum.time_evolve_batch(parameters, times, initial_states)
        state = states[0]
        
        # Initialize current partition
        current_partition = partition
        
        # Start training
        start = time.time()
        converged = False
        
        for step in tqdm(range(self.config.steps), desc="Entropic optimization"):
            # Update learning rate
            self._update_learning_rate(step)
            
            # Get next partition if needed
            if step > 0 and self.partition_strategy != PartitionStrategy.FIXED:
                current_partition = self._get_next_partition(current_partition)
            
            # Perform optimization step
            results = self.optimization_step(state, current_partition, target_gradient, weights)
            
            # Update history
            if (step + 1) % self.config.log_interval == 0:
                self._update_history(results)
            
            # Check for convergence
            if self._check_convergence(results["total_loss"].item()):
                print(f"Converged after {step+1} steps!")
                converged = True
                break
            
            # Save checkpoint if needed
            if (step + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(step + 1, state, results)
                self.save_training_curves()

        end = time.time()
        
        # If we have a best metric and we didn't converge naturally, use it
        if self.best_metric is not None and not converged:
            with torch.no_grad():
                self.geometry.metric_field[self.geometry.active_index].copy_(self.best_metric)
        
        # Save final results
        final_results = {
            "final_state": state.detach(),
            "final_metric": self.geometry.metric.detach(),
            "training_time": end - start,
            "history": self.history,
            "converged": converged,
            "steps": step + 1,
        }
        
        self.save_results(final_results)
        return final_results

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    def _update_history(self, results: Dict[str, torch.Tensor]) -> None:
        """Update training history.
        
        Args:
            results: Dictionary with results from optimization step
        """
        for key in self.history:
            tensor = results.get(key)
            if tensor is not None:
                self.history[key].append(float(tensor.detach().cpu()))

    def save_checkpoint(
        self, 
        step: int, 
        state: torch.Tensor, 
        results: Dict[str, torch.Tensor]
    ) -> None:
        """Save a checkpoint.
        
        Args:
            step: Current optimization step
            state: Current quantum state
            results: Dictionary with results from optimization step
        """
        payload = {
            "step": step,
            "state": state.detach().cpu(),
            "metric_field": self.geometry.metric_field.detach().cpu(),
            "results": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in results.items()},
            "current_lr": self.current_lr,
            "best_loss": self.best_loss,
            "steps_without_improvement": self.steps_without_improvement,
        }
        
        # Save best metric if available
        if self.best_metric is not None:
            payload["best_metric"] = self.best_metric.detach().cpu()
            
        torch.save(payload, self.results_path / f"checkpoint_{step}.pt")

    def save_training_curves(self) -> None:
        """Save training curves to JSON file."""
        with open(self.results_path / "training_logs.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def save_results(self, results: Dict[str, torch.Tensor]) -> None:
        """Save final results.
        
        Args:
            results: Dictionary with final results
        """
        # Convert tensors to CPU
        tensor_results = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in results.items()
        }
        
        # Save as PyTorch file
        torch.save(tensor_results, self.results_path / "final_results.pt")
        
        # Save a summary in JSON format
        summary = {
            "training_time": results["training_time"],
            "converged": results.get("converged", False),
            "steps": results.get("steps", self.config.steps),
            "final_loss": self.history["total_loss"][-1] if self.history["total_loss"] else None,
            "best_loss": self.best_loss,
        }
        
        with open(self.results_path / "results_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------
    def analyze_convergence(self) -> Dict[str, float]:
        """Analyze convergence properties.
        
        Returns:
            Dictionary with convergence metrics
        """
        if not self.history["total_loss"]:
            return {"converged": False}
            
        # Compute convergence metrics
        final_loss = self.history["total_loss"][-1]
        best_loss = min(self.history["total_loss"])
        loss_std = np.std(self.history["total_loss"][-10:]) if len(self.history["total_loss"]) >= 10 else np.nan
        
        # Check if converged
        converged = final_loss < self.config.convergence_threshold
        
        # Compute convergence rate (if enough data)
        if len(self.history["total_loss"]) >= 20:
            # Use exponential fit to estimate convergence rate
            steps = np.arange(len(self.history["total_loss"]))
            log_loss = np.log(np.array(self.history["total_loss"]) + 1e-10)
            
            # Simple linear regression on log loss
            slope, _ = np.polyfit(steps[-20:], log_loss[-20:], 1)
            convergence_rate = np.exp(slope)
        else:
            convergence_rate = np.nan
            
        return {
            "converged": converged,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "loss_std": loss_std,
            "convergence_rate": convergence_rate,
        }
        
    def compute_entropic_area_law(
        self, 
        state: torch.Tensor,
        partitions: List[List[int]]
    ) -> Dict[str, float]:
        """Compute entropic area law coefficient.
        
        Args:
            state: Quantum state
            partitions: List of different partitions
            
        Returns:
            Dictionary with area law metrics
        """
        # Compute entropy for each partition
        entropies = []
        areas = []
        
        for partition in partitions:
            # Compute entropy
            entropy = self.entropy.compute_entanglement_entropy(
                state, 
                partition, 
                include_edge=self.config.include_edge_modes
            )
            entropies.append(entropy.item())
            
            # Use boundary size as proxy for area
            area = len(partition)
            areas.append(area)
            
        # Fit linear relationship
        areas_array = np.array(areas)
        entropies_array = np.array(entropies)
        
        # Simple linear regression
        slope, intercept = np.polyfit(areas_array, entropies_array, 1)
        
        # Compute R² to assess fit quality
        y_pred = slope * areas_array + intercept
        ss_total = np.sum((entropies_array - np.mean(entropies_array))**2)
        ss_residual = np.sum((entropies_array - y_pred)**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        return {
            "area_law_coefficient": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "areas": areas,
            "entropies": entropies,
        }
        
    def compute_holographic_metrics(
        self, 
        state: torch.Tensor,
        partition: List[int]
    ) -> Dict[str, float]:
        """Compute metrics related to holographic principles.
        
        Args:
            state: Quantum state
            partition: Partition defining entanglement region
            
        Returns:
            Dictionary with holographic metrics
        """
        # Compute entropy
        entropy = self.entropy.compute_entanglement_entropy(
            state, 
            partition, 
            include_edge=self.config.include_edge_modes
        ).item()
        
        # Get metric at active index
        metric = self.geometry.metric
        
        # Compute determinant (volume element)
        det = torch.det(metric).item()
        
        # Compute Ricci scalar (curvature)
        ricci_scalar = self.geometry.compute_ricci_scalar()[self.geometry.active_index].item()
        
        # Compute area estimate (simplified)
        boundary_size = len(partition)
        area_estimate = boundary_size * abs(det)**(1/4)
        
        # Check if entropy ~ area relationship holds
        # S = Area/4G in natural units
        G = 1.0  # Newton's constant in natural units
        predicted_entropy = area_estimate / (4.0 * G)
        entropy_area_ratio = entropy / area_estimate
        
        return {
            "entropy": entropy,
            "area_estimate": area_estimate,
            "predicted_entropy": predicted_entropy,
            "entropy_area_ratio": entropy_area_ratio,
            "ricci_scalar": ricci_scalar,
            "metric_determinant": det,
        }