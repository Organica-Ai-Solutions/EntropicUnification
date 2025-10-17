"""Loss functions enforcing entropic-geometry consistency."""

from __future__ import annotations

from typing import Optional, Dict, List

import torch

from .coupling_layer import CouplingLayer


class LossFunctions:
    def __init__(
        self,
        coupling_layer: CouplingLayer,
        entropy_target: Optional[torch.Tensor] = None,
        regularization_weight: float = 1e-3,
        curvature_weight: float = 1.0,
    ) -> None:
        self.coupling = coupling_layer
        self.entropy_target = entropy_target
        self.regularization_weight = regularization_weight
        self.curvature_weight = curvature_weight

    def einstein_constraint_loss(self, terms: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.norm(terms["coupling_residual"])

    def entropy_gradient_loss(
        self, terms: Dict[str, torch.Tensor], target_gradient: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if target_gradient is None:
            if self.entropy_target is None:
                return torch.tensor(0.0, device=terms["entropy_gradient"].device)
            target_gradient = self.entropy_target
        return torch.norm(terms["entropy_gradient"] - target_gradient)

    def curvature_regularization(self) -> torch.Tensor:
        scalar = self.coupling.geometry.compute_ricci_scalar()
        return self.curvature_weight * scalar.abs()

    def metric_smoothness(self) -> torch.Tensor:
        field = self.coupling.geometry.metric_field
        grad = self.coupling.geometry._finite_difference(field)
        return self.regularization_weight * torch.norm(grad)

    def total_loss(
        self,
        terms: Dict[str, torch.Tensor],
        target_gradient: Optional[torch.Tensor],
        weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        einstein_loss = self.einstein_constraint_loss(terms)
        entropy_loss = self.entropy_gradient_loss(terms, target_gradient)
        curvature_loss = self.curvature_regularization()
        smoothness_loss = self.metric_smoothness()

        total = (
            weights.get("einstein", 1.0) * einstein_loss
            + weights.get("entropy", 1.0) * entropy_loss
            + weights.get("curvature", 0.1) * curvature_loss
            + weights.get("smoothness", 0.1) * smoothness_loss
        )

        return {
            "total_loss": total,
            "einstein_loss": einstein_loss,
            "entropy_loss": entropy_loss,
            "curvature_loss": curvature_loss,
            "smoothness_loss": smoothness_loss,
        }
