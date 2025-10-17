"""
Coupling layer linking entanglement gradients to spacetime curvature.

The layer accepts the quantum entropy module and the geometry engine and
returns the tensors needed by the loss module.  It also exposes convenient
helpers for computing Jacobson-style stress tensors and the Einstein tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .entropy_module import EntropyModule
from .geometry_engine import GeometryEngine


@dataclass
class CouplingTerms:
    entropy_gradient: torch.Tensor
    stress_tensor: torch.Tensor
    einstein_tensor: torch.Tensor
    coupling_residual: torch.Tensor


class CouplingLayer:
    def __init__(
        self,
        geometry_engine: GeometryEngine,
        entropy_module: EntropyModule,
        coupling_strength: float = 1.0,
        stress_form: str = "jacobson",
    ) -> None:
        self.geometry = geometry_engine
        self.entropy = entropy_module
        self.coupling_strength = coupling_strength
        self.stress_form = stress_form.lower()

    # ------------------------------------------------------------------
    # Stress-energy tensors induced by entropy gradients
    # ------------------------------------------------------------------
    def compute_entropy_stress_tensor(self, entropy_gradient: torch.Tensor) -> torch.Tensor:
        g = self.geometry.metric

        if self.stress_form == "jacobson":
            contraction = torch.dot(entropy_gradient, entropy_gradient)
            T = torch.outer(entropy_gradient, entropy_gradient)
            return self.coupling_strength * (T - 0.5 * g * contraction)

        if self.stress_form == "canonical":
            return self.coupling_strength * torch.outer(entropy_gradient, entropy_gradient)

        raise ValueError(f"Unknown stress tensor form '{self.stress_form}'")

    # ------------------------------------------------------------------
    # Einstein tensor and coupling residuals
    # ------------------------------------------------------------------
    def compute_einstein_tensor(self) -> torch.Tensor:
        ricci = self.geometry.compute_ricci_tensor()
        R = self.geometry.compute_ricci_scalar()
        g = self.geometry.metric
        return ricci[self.geometry.active_index] - 0.5 * R * g

    def compute_coupling_terms(self, state: torch.Tensor, partition: list) -> CouplingTerms:
        entropy_grad = self.entropy.entropy_gradient(state, partition)
        T = self.compute_entropy_stress_tensor(entropy_grad)
        G = self.compute_einstein_tensor()
        residual = G - T
        return CouplingTerms(entropy_grad, T, G, residual)

    def compute_coupling_consistency(self, state: torch.Tensor, partition: list) -> torch.Tensor:
        terms = self.compute_coupling_terms(state, partition)
        return torch.norm(terms.coupling_residual)

    def update_coupling(
        self,
        state: torch.Tensor,
        partition: list,
        learning_rate: float,
        metric_grad_clip: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        terms = self.compute_coupling_terms(state, partition)
        consistency = torch.norm(terms.coupling_residual)

        metric_gradient = torch.autograd.grad(
            consistency,
            self.geometry.metric_field,
            retain_graph=True,
            create_graph=True,
        )[0]

        active_grad = metric_gradient[self.geometry.active_index]
        if metric_grad_clip is not None:
            active_grad = torch.clamp(active_grad, -metric_grad_clip, metric_grad_clip)

        self.geometry.update_metric(active_grad, learning_rate)

        return {
            "entropy_gradient": terms.entropy_gradient,
            "stress_tensor": terms.stress_tensor,
            "einstein_tensor": terms.einstein_tensor,
            "coupling_residual": terms.coupling_residual,
            "metric_gradient": active_grad,
            "consistency": consistency,
        }
