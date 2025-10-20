"""
Coupling layer linking entanglement gradients to spacetime curvature.

The layer accepts the quantum entropy module and the geometry engine and
returns the tensors needed by the loss module. It also exposes convenient
helpers for computing entropic stress tensors and the Einstein tensor.

This implementation includes:
- Multiple stress-energy tensor formulations (Jacobson, canonical, Faulkner)
- Support for non-conformal matter fields
- Edge mode contributions
- Higher-order curvature corrections
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import torch

from .entropy_module import EntropyModule
from .geometry_engine import GeometryEngine


class StressTensorFormulation(str, Enum):
    """Different formulations of the entropic stress-energy tensor."""
    JACOBSON = "jacobson"  # Original Jacobson thermodynamic formulation
    CANONICAL = "canonical"  # Simple outer product of gradients
    FAULKNER = "faulkner"   # Faulkner's linearized Einstein formulation
    MODIFIED = "modified"   # Modified formulation with edge mode corrections


@dataclass
class CouplingTerms:
    """Container for all coupling-related tensors and quantities."""
    entropy_gradient: torch.Tensor
    stress_tensor: torch.Tensor
    einstein_tensor: torch.Tensor
    coupling_residual: torch.Tensor
    edge_mode_contribution: Optional[torch.Tensor] = None
    higher_curvature_terms: Optional[torch.Tensor] = None
    
    def __getitem__(self, key):
        """Make the class subscriptable to access its fields."""
        return getattr(self, key)


class CouplingLayer:
    """Couples quantum entanglement entropy with spacetime geometry."""
    
    def __init__(
        self,
        geometry_engine: GeometryEngine,
        entropy_module: EntropyModule,
        coupling_strength: float = 1.0,
        stress_form: Union[str, StressTensorFormulation] = StressTensorFormulation.JACOBSON,
        include_edge_modes: bool = True,
        include_higher_curvature: bool = False,
        conformal_invariance: bool = False,
        hbar_factor: float = 1.0/(2.0*3.14159),  # ℏ/(2π) in natural units
    ) -> None:
        """Initialize the coupling layer.
        
        Args:
            geometry_engine: The geometry engine for metric and curvature calculations
            entropy_module: The entropy module for entanglement calculations
            coupling_strength: Overall coupling strength (analogous to 8πG)
            stress_form: Which formulation of stress-energy tensor to use
            include_edge_modes: Whether to include edge mode contributions
            include_higher_curvature: Whether to include higher-order curvature terms
            conformal_invariance: Whether to assume conformal invariance
            hbar_factor: Factor of ℏ/(2π) in natural units
        """
        self.geometry = geometry_engine
        self.entropy = entropy_module
        self.coupling_strength = coupling_strength
        
        # Convert string to enum if needed
        if isinstance(stress_form, str):
            self.stress_form = StressTensorFormulation(stress_form.lower())
        else:
            self.stress_form = stress_form
            
        self.include_edge_modes = include_edge_modes
        self.include_higher_curvature = include_higher_curvature
        self.conformal_invariance = conformal_invariance
        self.hbar_factor = hbar_factor
        
        # Parameters for higher-order curvature corrections
        self.alpha_GB = 0.0  # Gauss-Bonnet coupling
        self.lambda_cosmo = 0.0  # Cosmological constant

    # ------------------------------------------------------------------
    # Stress-energy tensors induced by entropy gradients
    # ------------------------------------------------------------------
    def compute_entropy_stress_tensor(
        self, 
        entropy_gradient: torch.Tensor,
        metric: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute the stress-energy tensor from entropy gradients.
        
        Args:
            entropy_gradient: Gradient of entropy with respect to state parameters
            metric: Optional metric tensor (uses geometry engine's metric if None)
            
        Returns:
            Tuple of (stress_tensor, edge_mode_contribution)
        """
        if metric is None:
            metric = self.geometry.metric
            
        edge_contribution = None
        
        # Compute the basic stress tensor based on selected formulation
        if self.stress_form == StressTensorFormulation.JACOBSON:
            # Jacobson's thermodynamic formulation:
            # T_μν = (ℏ/2π)[∇_μS ∇_νS - (1/2)g_μν (∇S)²]
            
            # Ensure entropy gradient has the right dimensions for spacetime
            # We need to project the quantum state gradient to spacetime dimensions
            dim = self.geometry.dimensions
            if entropy_gradient.shape[0] != dim:
                # Project to spacetime dimensions using a simple mapping
                # This is a heuristic approach for demonstration purposes
                entropy_grad_spacetime = torch.zeros(dim, dtype=entropy_gradient.dtype)
                # Use the first dim components or pad with zeros
                entropy_grad_spacetime[:min(dim, entropy_gradient.shape[0])] = entropy_gradient[:min(dim, entropy_gradient.shape[0])]
            else:
                entropy_grad_spacetime = entropy_gradient
                
            contraction = torch.dot(entropy_grad_spacetime, entropy_grad_spacetime)
            T = torch.outer(entropy_grad_spacetime, entropy_grad_spacetime)
            T = self.hbar_factor * (T - 0.5 * metric * contraction)
            
        elif self.stress_form == StressTensorFormulation.CANONICAL:
            # Simple outer product:
            # T_μν = (ℏ/2π)∇_μS ∇_νS
            
            # Ensure entropy gradient has the right dimensions for spacetime
            dim = self.geometry.dimensions
            if entropy_gradient.shape[0] != dim:
                entropy_grad_spacetime = torch.zeros(dim, dtype=entropy_gradient.dtype)
                entropy_grad_spacetime[:min(dim, entropy_gradient.shape[0])] = entropy_gradient[:min(dim, entropy_gradient.shape[0])]
            else:
                entropy_grad_spacetime = entropy_gradient
                
            T = self.hbar_factor * torch.outer(entropy_grad_spacetime, entropy_grad_spacetime)
            
        elif self.stress_form == StressTensorFormulation.FAULKNER:
            # Faulkner's linearized Einstein formulation:
            # T_μν = (ℏ/2π)[∇_μ∇_νS - (□S)g_μν]
            # This requires computing second derivatives of entropy
            # For now, we approximate with a modified Jacobson form
            
            # Ensure entropy gradient has the right dimensions for spacetime
            dim = self.geometry.dimensions
            if entropy_gradient.shape[0] != dim:
                entropy_grad_spacetime = torch.zeros(dim, dtype=entropy_gradient.dtype)
                entropy_grad_spacetime[:min(dim, entropy_gradient.shape[0])] = entropy_gradient[:min(dim, entropy_gradient.shape[0])]
            else:
                entropy_grad_spacetime = entropy_gradient
                
            contraction = torch.dot(entropy_grad_spacetime, entropy_grad_spacetime)
            T = torch.outer(entropy_grad_spacetime, entropy_grad_spacetime)
            
            # Add a term that approximates the effect of second derivatives
            trace_term = contraction * metric
            T = self.hbar_factor * (T - trace_term)
            
        elif self.stress_form == StressTensorFormulation.MODIFIED:
            # Modified formulation with corrections for non-conformal fields:
            # T_μν = (ℏ/2π)[∇_μS ∇_νS - (1/2)g_μν (∇S)² + α R_μν]
            # where α is a non-conformality parameter
            
            # Ensure entropy gradient has the right dimensions for spacetime
            dim = self.geometry.dimensions
            if entropy_gradient.shape[0] != dim:
                entropy_grad_spacetime = torch.zeros(dim, dtype=entropy_gradient.dtype)
                entropy_grad_spacetime[:min(dim, entropy_gradient.shape[0])] = entropy_gradient[:min(dim, entropy_gradient.shape[0])]
            else:
                entropy_grad_spacetime = entropy_gradient
                
            contraction = torch.dot(entropy_grad_spacetime, entropy_grad_spacetime)
            T = torch.outer(entropy_grad_spacetime, entropy_grad_spacetime)
            
            # Basic Jacobson term
            T = self.hbar_factor * (T - 0.5 * metric * contraction)
            
            # Add correction for non-conformal fields if needed
            if not self.conformal_invariance:
                # Get Ricci tensor for the correction
                ricci = self.geometry.compute_ricci_tensor()
                
                # Non-conformality parameter (could be made configurable)
                alpha = 0.1
                
                # Add correction term
                T = T + alpha * self.hbar_factor * ricci[self.geometry.active_index]
        else:
            raise ValueError(f"Unknown stress tensor formulation: {self.stress_form}")
            
        # Add edge mode contribution if requested
        if self.include_edge_modes and not self.conformal_invariance:
            # Edge modes contribute an additional boundary stress-energy
            # This is a simplified model - in reality, edge mode contribution
            # depends on the specific gauge theory and boundary conditions
            
            # For simplicity, we model it as a small correction to the stress tensor
            # proportional to the metric (like a cosmological constant term)
            edge_factor = 0.01  # Small contribution factor
            edge_contribution = edge_factor * self.hbar_factor * metric
            
            # Add to the stress tensor
            T = T + edge_contribution
            
        # Apply overall coupling strength
        T = self.coupling_strength * T
            
        return T, edge_contribution

    # ------------------------------------------------------------------
    # Einstein tensor and higher curvature terms
    # ------------------------------------------------------------------
    def compute_einstein_tensor(
        self,
        include_higher_curvature: Optional[bool] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute the Einstein tensor and optional higher-order curvature terms.
        
        Args:
            include_higher_curvature: Whether to include higher-order curvature terms
                (defaults to self.include_higher_curvature)
                
        Returns:
            Tuple of (einstein_tensor, higher_curvature_terms)
        """
        include_higher_curvature = (
            self.include_higher_curvature if include_higher_curvature is None 
            else include_higher_curvature
        )
        
        # Compute basic Einstein tensor
        ricci = self.geometry.compute_ricci_tensor()
        R = self.geometry.compute_ricci_scalar()
        g = self.geometry.metric
        
        # G_μν = R_μν - (1/2)R g_μν
        # Ensure dimensions match for the calculation
        active_idx = self.geometry.active_index
        ricci_active = ricci[active_idx]
        
        # For scalar * tensor multiplication, we need to reshape the scalar
        # to ensure broadcasting works correctly
        R_active = R[active_idx]
        
        # Create the Einstein tensor with proper broadcasting
        G = ricci_active - 0.5 * R_active * g
        
        # Add cosmological constant if non-zero
        if abs(self.lambda_cosmo) > 1e-10:
            G = G + self.lambda_cosmo * g
        
        higher_curvature_terms = None
        
        # Compute higher-order curvature terms if requested
        if include_higher_curvature:
            # Simplified Gauss-Bonnet term: α(R_μαβγ R_ν^αβγ - 2R_μα R_ν^α + (1/2)R² g_μν)
            # For simplicity, we'll approximate this with a term proportional to
            # the Ricci tensor squared minus trace squared
            
            # We already have ricci_active from above
            ricci_squared = torch.matmul(ricci_active, ricci_active)
            ricci_trace = torch.sum(torch.diagonal(ricci_active, dim1=0, dim2=1))
            
            # Approximate Gauss-Bonnet contribution
            gb_term = ricci_squared - 0.5 * (ricci_trace**2) * g
            
            higher_curvature_terms = self.alpha_GB * gb_term
            
            # Add to Einstein tensor
            G = G + higher_curvature_terms
            
        return G, higher_curvature_terms

    # ------------------------------------------------------------------
    # Main coupling computation
    # ------------------------------------------------------------------
    def compute_coupling_terms(
        self, 
        state: torch.Tensor, 
        partition: list
    ) -> CouplingTerms:
        """Compute all coupling terms between entropy and geometry.
        
        Args:
            state: Quantum state vector
            partition: Partition defining the entanglement region
            
        Returns:
            CouplingTerms object containing all relevant tensors
        """
        # Compute entropy gradient with edge mode handling
        entropy_grad = self.entropy.entropy_gradient(
            state, 
            partition, 
            include_edge=self.include_edge_modes,
            apply_uv_cutoff=True
        )
        
        # Compute stress tensor
        T, edge_contribution = self.compute_entropy_stress_tensor(entropy_grad)
        
        # Compute Einstein tensor
        G, higher_curvature = self.compute_einstein_tensor()
        
        # Compute residual (mismatch between geometry and entropy)
        residual = G - T
        
        return CouplingTerms(
            entropy_grad, 
            T, 
            G, 
            residual,
            edge_contribution,
            higher_curvature
        )

    def compute_coupling_consistency(self, state: torch.Tensor, partition: list) -> torch.Tensor:
        """Compute the consistency between entropy gradient and spacetime curvature.
        
        Args:
            state: Quantum state vector
            partition: Partition defining the entanglement region
            
        Returns:
            Consistency measure (lower is better)
        """
        terms = self.compute_coupling_terms(state, partition)
        return torch.norm(terms.coupling_residual)

    def update_coupling(
        self,
        state: torch.Tensor,
        partition: list,
        learning_rate: float,
        metric_grad_clip: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Update the metric to improve coupling consistency.
        
        Args:
            state: Quantum state vector
            partition: Partition defining the entanglement region
            learning_rate: Learning rate for gradient descent
            metric_grad_clip: Optional clipping value for metric gradients
            
        Returns:
            Dictionary with updated tensors and metrics
        """
        terms = self.compute_coupling_terms(state, partition)
        consistency = torch.norm(terms.coupling_residual)

        # Compute gradient of consistency with respect to metric
        metric_gradient = torch.autograd.grad(
            consistency,
            self.geometry.metric_field,
            retain_graph=True,
            create_graph=True,
        )[0]

        # Extract active component and apply gradient clipping if needed
        active_grad = metric_gradient[self.geometry.active_index]
        if metric_grad_clip is not None:
            active_grad = torch.clamp(active_grad, -metric_grad_clip, metric_grad_clip)

        # Update the metric
        self.geometry.update_metric(active_grad, learning_rate)

        # Return all relevant tensors and metrics
        result = {
            "entropy_gradient": terms.entropy_gradient,
            "stress_tensor": terms.stress_tensor,
            "einstein_tensor": terms.einstein_tensor,
            "coupling_residual": terms.coupling_residual,
            "metric_gradient": active_grad,
            "consistency": consistency,
        }
        
        # Add optional components if present
        if terms.edge_mode_contribution is not None:
            result["edge_mode_contribution"] = terms.edge_mode_contribution
            
        if terms.higher_curvature_terms is not None:
            result["higher_curvature_terms"] = terms.higher_curvature_terms
            
        return result
        
    # ------------------------------------------------------------------
    # Configuration methods
    # ------------------------------------------------------------------
    def set_stress_tensor_formulation(
        self, 
        formulation: Union[str, StressTensorFormulation]
    ) -> None:
        """Set the stress tensor formulation to use.
        
        Args:
            formulation: Stress tensor formulation to use
        """
        if isinstance(formulation, str):
            self.stress_form = StressTensorFormulation(formulation.lower())
        else:
            self.stress_form = formulation
            
    def set_higher_curvature_parameters(
        self,
        alpha_gb: float = 0.0,
        lambda_cosmo: float = 0.0
    ) -> None:
        """Set parameters for higher-order curvature terms.
        
        Args:
            alpha_gb: Gauss-Bonnet coupling parameter
            lambda_cosmo: Cosmological constant
        """
        self.alpha_GB = alpha_gb
        self.lambda_cosmo = lambda_cosmo
        
    def set_conformal_invariance(self, conformal: bool) -> None:
        """Set whether to assume conformal invariance.
        
        Args:
            conformal: Whether fields are conformally invariant
        """
        self.conformal_invariance = conformal
        # Update entropy module as well to ensure consistency
        self.entropy.conformal_invariance = conformal