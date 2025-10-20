"""
Loss functions enforcing entropic-geometry consistency with improved optimization.

This module provides loss functions for the EntropicUnification framework with
enhanced capabilities to handle multiple local minima and improve convergence
in the optimization landscape. It includes:

- Multiple loss formulations for different physical scenarios
- Adaptive weighting of loss components
- Regularization techniques to avoid poor local minima
- Annealing schedules for optimization
- Basin hopping capabilities to explore multiple minima
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import numpy as np

from .coupling_layer import CouplingLayer


class LossFormulation(str, Enum):
    """Different formulations of the loss function."""
    STANDARD = "standard"       # Standard Einstein constraint
    RELAXED = "relaxed"         # Relaxed constraint with regularization
    ADAPTIVE = "adaptive"       # Adaptive weighting based on training progress
    ANNEALED = "annealed"       # Simulated annealing for better exploration


class LossFunctions:
    """Loss functions with improved handling of optimization landscape."""
    
    def __init__(
        self,
        coupling_layer: CouplingLayer,
        entropy_target: Optional[torch.Tensor] = None,
        regularization_weight: float = 1e-3,
        curvature_weight: float = 1.0,
        formulation: Union[str, LossFormulation] = LossFormulation.STANDARD,
        basin_hopping: bool = False,
        annealing_schedule: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the loss functions.
        
        Args:
            coupling_layer: The coupling layer connecting entropy and geometry
            entropy_target: Optional target entropy gradient
            regularization_weight: Weight for regularization terms
            curvature_weight: Weight for curvature terms
            formulation: Which loss formulation to use
            basin_hopping: Whether to use basin hopping for exploration
            annealing_schedule: Optional annealing schedule parameters
        """
        self.coupling = coupling_layer
        self.entropy_target = entropy_target
        self.regularization_weight = regularization_weight
        self.curvature_weight = curvature_weight
        
        # Convert string to enum if needed
        if isinstance(formulation, str):
            self.formulation = LossFormulation(formulation.lower())
        else:
            self.formulation = formulation
            
        self.basin_hopping = basin_hopping
        
        # Set up annealing schedule
        if annealing_schedule is None:
            self.annealing_schedule = {
                "initial_temp": 1.0,
                "final_temp": 0.01,
                "decay_rate": 0.95,
            }
        else:
            self.annealing_schedule = annealing_schedule
            
        # Current annealing temperature
        self.temperature = self.annealing_schedule["initial_temp"]
        
        # Training history for adaptive weighting
        self.history = {
            "einstein_loss": [],
            "entropy_loss": [],
            "curvature_loss": [],
            "smoothness_loss": [],
            "step": 0,
        }
        
        # For basin hopping
        self.best_metric = None
        self.best_loss = float('inf')
        self.last_hop_step = 0
        self.hop_count = 0
        self.hop_threshold = 50  # Steps without improvement before hopping
        self.max_hops = 5  # Maximum number of basin hops

    def einstein_constraint_loss(
        self, 
        terms: Dict[str, torch.Tensor],
        use_relaxed: bool = False
    ) -> torch.Tensor:
        """Compute the Einstein constraint loss.
        
        Args:
            terms: Dictionary of coupling terms
            use_relaxed: Whether to use a relaxed constraint
            
        Returns:
            Einstein constraint loss
        """
        residual = terms["coupling_residual"]
        
        if use_relaxed or self.formulation == LossFormulation.RELAXED:
            # Relaxed constraint: L2 norm with smoothing
            # This helps avoid sharp valleys in the loss landscape
            squared_norm = torch.sum(residual**2)
            return torch.sqrt(squared_norm + 1e-6)
        else:
            # Standard constraint: Frobenius norm
            return torch.norm(residual)

    def entropy_gradient_loss(
        self, 
        terms: Dict[str, torch.Tensor], 
        target_gradient: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute the entropy gradient loss.
        
        Args:
            terms: Dictionary of coupling terms
            target_gradient: Optional target entropy gradient
            
        Returns:
            Entropy gradient loss
        """
        if target_gradient is None:
            if self.entropy_target is None:
                return torch.tensor(0.0, device=terms["entropy_gradient"].device)
            target_gradient = self.entropy_target
            
        # Compute difference between actual and target gradient
        diff = terms["entropy_gradient"] - target_gradient
        
        # Use Huber loss for robustness to outliers
        delta = 1.0
        huber_diff = torch.where(
            torch.abs(diff) < delta,
            0.5 * diff**2,
            delta * (torch.abs(diff) - 0.5 * delta)
        )
        
        return torch.sum(huber_diff)

    def curvature_regularization(
        self, 
        use_gauss_bonnet: bool = False
    ) -> torch.Tensor:
        """Compute curvature regularization.
        
        Args:
            use_gauss_bonnet: Whether to use Gauss-Bonnet term
            
        Returns:
            Curvature regularization loss
        """
        if use_gauss_bonnet and hasattr(self.coupling.geometry, "compute_gauss_bonnet_term"):
            # Use Gauss-Bonnet term for more comprehensive regularization
            gb_term = self.coupling.geometry.compute_gauss_bonnet_term()
            return self.curvature_weight * torch.abs(gb_term[self.coupling.geometry.active_index])
        else:
            # Use simple Ricci scalar regularization
            scalar = self.coupling.geometry.compute_ricci_scalar()
            return self.curvature_weight * torch.abs(scalar[self.coupling.geometry.active_index])

    def metric_smoothness(self, use_weyl: bool = False) -> torch.Tensor:
        """Compute metric smoothness regularization.
        
        Args:
            use_weyl: Whether to use Weyl tensor for conformal regularization
            
        Returns:
            Metric smoothness loss
        """
        field = self.coupling.geometry.metric_field
        grad = self.coupling.geometry._finite_difference(field)
        
        if use_weyl and hasattr(self.coupling.geometry, "compute_weyl_tensor"):
            # Use Weyl tensor to penalize conformal irregularities
            weyl = self.coupling.geometry.compute_weyl_tensor()
            weyl_norm = torch.norm(weyl[self.coupling.geometry.active_index])
            
            # Combine gradient smoothness with Weyl tensor regularization
            return self.regularization_weight * (torch.norm(grad) + weyl_norm)
        else:
            # Standard gradient smoothness
            return self.regularization_weight * torch.norm(grad)

    def update_annealing_temperature(self, step: int) -> None:
        """Update the annealing temperature based on the current step.
        
        Args:
            step: Current optimization step
        """
        if self.formulation == LossFormulation.ANNEALED:
            # Exponential decay schedule
            self.temperature = self.annealing_schedule["initial_temp"] * (
                self.annealing_schedule["decay_rate"] ** (step / 100)
            )
            # Clamp to final temperature
            self.temperature = max(self.temperature, self.annealing_schedule["final_temp"])

    def compute_adaptive_weights(self) -> Dict[str, float]:
        """Compute adaptive weights based on training history.
        
        Returns:
            Dictionary of adaptive weights
        """
        if len(self.history["einstein_loss"]) < 10 or self.formulation != LossFormulation.ADAPTIVE:
            # Not enough history or not using adaptive formulation
            return {
                "einstein": 1.0,
                "entropy": 1.0,
                "curvature": 0.1,
                "smoothness": 0.1,
            }
            
        # Compute recent loss statistics
        recent = 10  # Consider last 10 steps
        losses = {
            "einstein": np.mean(self.history["einstein_loss"][-recent:]),
            "entropy": np.mean(self.history["entropy_loss"][-recent:]),
            "curvature": np.mean(self.history["curvature_loss"][-recent:]),
            "smoothness": np.mean(self.history["smoothness_loss"][-recent:]),
        }
        
        # Compute relative magnitudes
        total = sum(losses.values()) + 1e-10  # Avoid division by zero
        relative = {k: v / total for k, v in losses.items()}
        
        # Invert the weights to focus on larger losses
        inverted = {k: 1.0 / (v + 1e-10) for k, v in relative.items()}
        
        # Normalize to sum to appropriate values
        total_inverted = sum(inverted.values())
        weights = {
            "einstein": inverted["einstein"] / total_inverted,
            "entropy": inverted["entropy"] / total_inverted,
            "curvature": 0.1 * inverted["curvature"] / total_inverted,
            "smoothness": 0.1 * inverted["smoothness"] / total_inverted,
        }
        
        return weights

    def check_basin_hopping(
        self, 
        total_loss: torch.Tensor, 
        terms: Dict[str, torch.Tensor]
    ) -> bool:
        """Check if basin hopping should be performed.
        
        Args:
            total_loss: Current total loss
            terms: Dictionary of coupling terms
            
        Returns:
            True if basin hopping should be performed, False otherwise
        """
        if not self.basin_hopping:
            return False
            
        # Update best loss if current loss is better
        # Ensure we're using the real part of the loss for comparison
        loss_val = float(torch.real(total_loss).item())
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_metric = self.coupling.geometry.metric.detach().clone()
            self.last_hop_step = self.history["step"]
            return False
            
        # Check if we've been stuck for too long
        steps_since_improvement = self.history["step"] - self.last_hop_step
        if steps_since_improvement > self.hop_threshold and self.hop_count < self.max_hops:
            self.hop_count += 1
            self.last_hop_step = self.history["step"]
            return True
            
        return False

    def perform_basin_hop(self) -> None:
        """Perform a basin hop by perturbing the metric."""
        if self.best_metric is not None:
            # Start from the best metric found so far
            with torch.no_grad():
                self.coupling.geometry.metric_field[self.coupling.geometry.active_index].copy_(
                    self.best_metric
                )
                
            # Add random perturbation scaled by temperature
            perturbation = torch.randn_like(self.best_metric) * self.temperature * 0.1
            
            # Apply perturbation while preserving symmetry
            with torch.no_grad():
                perturbed = self.best_metric + perturbation
                perturbed = 0.5 * (perturbed + perturbed.t())
                self.coupling.geometry.metric_field[self.coupling.geometry.active_index].copy_(
                    perturbed
                )

    def total_loss(
        self,
        terms: Dict[str, torch.Tensor],
        target_gradient: Optional[torch.Tensor],
        weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        """Compute the total loss with improved handling of local minima.
        
        Args:
            terms: Dictionary of coupling terms
            target_gradient: Optional target entropy gradient
            weights: Dictionary of loss weights
            
        Returns:
            Dictionary with loss components
        """
        # Update step counter
        self.history["step"] += 1
        
        # Update annealing temperature
        self.update_annealing_temperature(self.history["step"])
        
        # Compute loss components
        use_relaxed = self.formulation in [LossFormulation.RELAXED, LossFormulation.ANNEALED]
        use_advanced = self.formulation != LossFormulation.STANDARD
        
        einstein_loss = self.einstein_constraint_loss(terms, use_relaxed=use_relaxed)
        entropy_loss = self.entropy_gradient_loss(terms, target_gradient)
        curvature_loss = self.curvature_regularization(use_gauss_bonnet=use_advanced)
        smoothness_loss = self.metric_smoothness(use_weyl=use_advanced)
        
        # Use adaptive weights if requested
        if self.formulation == LossFormulation.ADAPTIVE:
            adaptive_weights = self.compute_adaptive_weights()
            weights = {
                "einstein": weights.get("einstein", 1.0) * adaptive_weights["einstein"],
                "entropy": weights.get("entropy", 1.0) * adaptive_weights["entropy"],
                "curvature": weights.get("curvature", 0.1) * adaptive_weights["curvature"],
                "smoothness": weights.get("smoothness", 0.1) * adaptive_weights["smoothness"],
            }

        # Compute total loss
        total = (
            weights.get("einstein", 1.0) * einstein_loss
            + weights.get("entropy", 1.0) * entropy_loss
            + weights.get("curvature", 0.1) * curvature_loss
            + weights.get("smoothness", 0.1) * smoothness_loss
        )
        
        # Apply annealing if using annealed formulation
        if self.formulation == LossFormulation.ANNEALED:
            # Add noise scaled by temperature to help escape local minima
            noise = torch.randn(1, device=total.device) * self.temperature
            total = total + noise
        
        # Update history
        self.history["einstein_loss"].append(einstein_loss.item())
        self.history["entropy_loss"].append(entropy_loss.item())
        self.history["curvature_loss"].append(curvature_loss.item())
        self.history["smoothness_loss"].append(smoothness_loss.item())
        
        # Check if basin hopping should be performed
        if self.check_basin_hopping(total, terms):
            self.perform_basin_hop()
        
        return {
            "total_loss": total,
            "einstein_loss": einstein_loss,
            "entropy_loss": entropy_loss,
            "curvature_loss": curvature_loss,
            "smoothness_loss": smoothness_loss,
            "temperature": torch.tensor(self.temperature),
        }
    
    # ------------------------------------------------------------------
    # Configuration methods
    # ------------------------------------------------------------------
    def set_formulation(self, formulation: Union[str, LossFormulation]) -> None:
        """Set the loss formulation to use.
        
        Args:
            formulation: Loss formulation to use
        """
        if isinstance(formulation, str):
            self.formulation = LossFormulation(formulation.lower())
        else:
            self.formulation = formulation
            
    def set_basin_hopping(self, enable: bool) -> None:
        """Enable or disable basin hopping.
        
        Args:
            enable: Whether to enable basin hopping
        """
        self.basin_hopping = enable
        if enable:
            # Reset basin hopping state
            self.best_loss = float('inf')
            self.best_metric = None
            self.last_hop_step = self.history["step"]
            self.hop_count = 0
            
    def set_annealing_schedule(self, schedule: Dict[str, float]) -> None:
        """Set the annealing schedule.
        
        Args:
            schedule: Dictionary with annealing parameters
        """
        self.annealing_schedule = schedule
        self.temperature = schedule["initial_temp"]