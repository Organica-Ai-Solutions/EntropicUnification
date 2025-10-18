"""
Geometry Engine: Represents and manipulates the spacetime metric gμν(t, x).

This implementation treats the metric as a differentiable field defined on a 1-D
spatial lattice (sufficient for 1+1 or 3+1 toy models where the metric varies
along a single spatial coordinate).  The engine provides finite-difference
derivatives, Christoffel symbols, and curvature tensors that are all compatible
with PyTorch autograd so they can participate in the global optimisation loop.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional


class GeometryEngine(nn.Module):
    """Differentiable spacetime metric on a 1-D lattice."""

    def __init__(
        self,
        dimensions: int,
        lattice_size: int,
        dx: float = 1.0,
        regularization: float = 1e-4,
        initial_metric: str = "minkowski",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if dimensions < 2:
            raise ValueError("GeometryEngine requires at least 2 spacetime dimensions")

        self.dimensions = dimensions
        self.lattice_size = lattice_size
        self.dx = torch.as_tensor(dx, dtype=dtype, device=device or torch.device("cpu"))
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.regularization = regularization

        # Base Lorentzian metric (diag(-1, +1, +1, ...))
        base = torch.diag(
            torch.tensor([-1.0] + [1.0] * (dimensions - 1), dtype=dtype, device=self.device)
        )
        field = base.repeat(lattice_size, 1, 1)

        self.metric_field = nn.Parameter(field)
        self.active_index = lattice_size // 2

        self._apply_initial_metric(initial_metric)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    @property
    def metric(self) -> torch.Tensor:
        """Return the metric tensor at the active lattice index."""

        return self.metric_field[self.active_index]

    def set_active_index(self, index: int) -> None:
        if not 0 <= index < self.lattice_size:
            raise IndexError("Active metric index out of bounds")
        self.active_index = index

    def _apply_initial_metric(self, name: str) -> None:
        with torch.no_grad():
            if name.lower() == "minkowski":
                # Already initialised to Minkowski; nothing else to do
                pass
            elif name.lower() == "frw":
                # Simple flat FRW with scale factor a(x) = 1 + ε x
                epsilon = 1e-3
                x = torch.linspace(-1.0, 1.0, self.lattice_size, device=self.device)
                a = 1.0 + epsilon * x
                for i in range(self.lattice_size):
                    g = self.metric_field[i]
                    g[0, 0] = -1.0
                    for j in range(1, self.dimensions):
                        g[j, j] = a[i] ** 2
            else:
                raise ValueError(f"Unknown initial metric '{name}'")
            self._enforce_symmetry()

    def _enforce_symmetry(self) -> None:
        with torch.no_grad():
            sym_field = 0.5 * (self.metric_field + self.metric_field.transpose(-1, -2))
            self.metric_field.copy_(sym_field)

    # ------------------------------------------------------------------
    # Finite differences
    # ------------------------------------------------------------------
    def _finite_difference(self, tensor: torch.Tensor, order: int = 1, axis: int = 0) -> torch.Tensor:
        """Higher-order finite differences along specified axis.
        
        Args:
            tensor: Input tensor to take derivatives of
            order: Order of the derivative (1 or 2 supported)
            axis: Axis along which to take the derivative
            
        Returns:
            Tensor containing the finite difference approximation
        """
        if order not in (1, 2):
            raise ValueError("Only 1st and 2nd order derivatives are supported")
            
        if tensor.dim() < 3:
            raise ValueError("Input tensor must have at least 3 dimensions")
            
        dx = self.dx
        ndim = tensor.dim()
        
        # Pad the tensor for boundary handling
        pad_size = 2 if order == 2 else 1
        pad = [(0, 0)] * ndim
        pad[axis] = (pad_size, pad_size)
        padded = torch.nn.functional.pad(tensor, [p for dim in reversed(pad) for p in dim])
        
        if order == 1:
            # 4th order central difference for 1st derivative
            derivative = (
                -padded.narrow(axis, 4, tensor.size(axis)) 
                + 8 * padded.narrow(axis, 3, tensor.size(axis))
                - 8 * padded.narrow(axis, 1, tensor.size(axis))
                + padded.narrow(axis, 0, tensor.size(axis))
            ) / (12.0 * dx)
        else:
            # 2nd order central difference for 2nd derivative
            derivative = (
                -padded.narrow(axis, 4, tensor.size(axis)) 
                + 16 * padded.narrow(axis, 3, tensor.size(axis))
                - 30 * padded.narrow(axis, 2, tensor.size(axis))
                + 16 * padded.narrow(axis, 1, tensor.size(axis))
                - padded.narrow(axis, 0, tensor.size(axis))
            ) / (12.0 * dx ** 2)
            
        return derivative
        
    def compute_christoffel(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute the Christoffel symbols from the metric tensor.
        
        Args:
            metric: Metric tensor g_μν of shape [d, d] or [..., d, d]
            
        Returns:
            Christoffel symbols Γ^λ_μν of shape [d, d, d] or [..., d, d, d]
        """
        # Compute inverse metric g^μν
        g_inv = torch.linalg.inv(metric)
        
        # Compute derivatives of the metric
        dg = torch.stack([
            self._finite_difference(metric, order=1, axis=i)
            for i in range(metric.dim() - 2)
        ], dim=-3)  # Shape: [d, ..., d, d]
        
        # Compute Christoffel symbols: Γ^λ_μν = 0.5 g^λρ (∂_μ g_νρ + ∂_ν g_μρ - ∂_ρ g_μν)
        term1 = dg.unsqueeze(-1)  # ∂_μ g_νρ
        term1 = term1.permute(0, 1, 2, 3, 4)  # Reorder to match indices
        term2 = term1.permute(1, 0, 2, 3, 4)  # ∂_ν g_μρ
        term3 = term1.permute(2, 1, 0, 3, 4)  # ∂_ρ g_μν
        
        christoffel = 0.5 * torch.einsum('...λρ,...μνρ->...λμν', g_inv, term1 + term2 - term3)
        return christoffel
        
    def compute_riemann_tensor(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute the Riemann curvature tensor from the metric.
        
        The Riemann tensor is computed as:
        R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ
        
        Args:
            metric: Metric tensor g_μν of shape [d, d] or [..., d, d]
            
        Returns:
            Riemann curvature tensor R^ρ_σμν of shape [d, d, d, d] or [..., d, d, d, d]
        """
        # First compute Christoffel symbols and their derivatives
        gamma = self.compute_christoffel(metric)  # Γ^λ_μν
        
        # Compute derivatives of Christoffel symbols
        dgamma = torch.stack([
            self._finite_difference(gamma, order=1, axis=i)
            for i in range(gamma.dim() - 3)
        ], dim=-4)  # Shape: [d, ..., d, d, d]
        
        # Compute the Riemann tensor components
        # R^ρ_σμν = ∂_μ Γ^ρ_νσ - ∂_ν Γ^ρ_μσ + Γ^ρ_μλ Γ^λ_νσ - Γ^ρ_νλ Γ^λ_μσ
        dgamma1 = dgamma.permute(0, 1, 3, 2, 4)  # ∂_μ Γ^ρ_νσ
        dgamma2 = dgamma.permute(1, 0, 3, 2, 4)  # ∂_ν Γ^ρ_μσ
        
        # Contract Christoffel symbols: Γ^ρ_μλ Γ^λ_νσ
        gamma_contract1 = torch.einsum('...ρμλ,...λνσ->...ρμνσ', gamma, gamma)
        gamma_contract2 = torch.einsum('...ρνλ,...λμσ->...ρμνσ', gamma, gamma)
        
        # Combine all terms
        riemann = (dgamma1 - dgamma2.permute(0, 1, 3, 2, 4) + 
                  gamma_contract1 - gamma_contract2)
                  
        return riemann
        
    def compute_ricci_tensor(self, riemann: torch.Tensor) -> torch.Tensor:
        """Compute the Ricci tensor from the Riemann tensor.
        
        The Ricci tensor is the contraction: R_μν = R^λ_μλν
        
        Args:
            riemann: Riemann tensor R^ρ_σμν of shape [d, d, d, d] or [..., d, d, d, d]
            
        Returns:
            Ricci tensor R_μν of shape [d, d] or [..., d, d]
        """
        return torch.einsum('...ρμρν->...μν', riemann)
        
    def compute_ricci_scalar(self, ricci: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
        """Compute the Ricci scalar from the Ricci tensor and metric.
        
        The Ricci scalar is the contraction: R = g^μν R_μν
        
        Args:
            ricci: Ricci tensor R_μν of shape [d, d] or [..., d, d]
            metric: Metric tensor g_μν of shape [d, d] or [..., d, d]
            
        Returns:
            Ricci scalar R of shape [] or [...]
        """
        g_inv = torch.linalg.inv(metric)
        return torch.einsum('...μν,...μν->...', g_inv, ricci)
        
    def compute_einstein_tensor(self, ricci: torch.Tensor, ricci_scalar: torch.Tensor, 
                              metric: torch.Tensor) -> torch.Tensor:
        """Compute the Einstein tensor G_μν = R_μν - 1/2 R g_μν.
        
        Args:
            ricci: Ricci tensor R_μν of shape [d, d] or [..., d, d]
            ricci_scalar: Ricci scalar R of shape [] or [...]
            metric: Metric tensor g_μν of shape [d, d] or [..., d, d]
            
        Returns:
            Einstein tensor G_μν of shape [d, d] or [..., d, d]
        """
        return ricci - 0.5 * ricci_scalar.unsqueeze(-1).unsqueeze(-1) * metric
        derivative[1:-1] = (tensor[2:] - tensor[:-2]) / (2.0 * dx)

        # One-sided differences for the boundaries
        derivative[0] = (tensor[1] - tensor[0]) / dx
        derivative[-1] = (tensor[-1] - tensor[-2]) / dx

        return derivative

    # ------------------------------------------------------------------
    # Curvature calculations
    # ------------------------------------------------------------------
    def compute_metric_derivatives(self, metric: Optional[torch.Tensor] = None, order: int = 1) -> torch.Tensor:
        """Compute ∂_μ g_{σν} for each lattice point.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            order: Order of the derivative (1 or 2)
            
        Returns:
            Tensor of metric derivatives with shape [lattice_size, d, d, d] for 1st order
            or [lattice_size, d, d, d, d] for 2nd order (where d is number of dimensions)
        """
        if metric is None:
            metric = self.metric_field
            
        if order == 1:
            # First derivatives
            deriv = torch.zeros(
                (self.lattice_size, self.dimensions, self.dimensions, self.dimensions),
                dtype=self.dtype,
                device=self.device,
            )
            
            # Compute derivatives along each dimension
            for mu in range(self.dimensions):
                deriv[:, mu] = self._finite_difference(metric, order=1, axis=0) / self.dx
                
        elif order == 2:
            # Second derivatives
            deriv = torch.zeros(
                (self.lattice_size, self.dimensions, self.dimensions, 
                 self.dimensions, self.dimensions),
                dtype=self.dtype,
                device=self.device,
            )
            
            # Compute second derivatives
            for mu in range(self.dimensions):
                for nu in range(self.dimensions):
                    # Get first derivative along nu
                    dg_dnu = self._finite_difference(metric, order=1, axis=0) / self.dx
                    # Take derivative of dg_dnu along mu
                    deriv[:, mu, nu] = self._finite_difference(dg_dnu, order=1, axis=0) / self.dx
        else:
            raise ValueError("Only 1st and 2nd order derivatives are supported")
            
        return deriv

    def compute_christoffel_symbols(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Γ^α_{μν} for every lattice point.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Christoffel symbols with shape [lattice_size, d, d, d]
            where Γ^α_{μν} = christoffel[i,α,μ,ν]
        """
        if metric is None:
            metric = self.metric_field
            
        dg = self.compute_metric_derivatives(metric, order=1)
        g_inv = torch.linalg.inv(metric)
        
        # Pre-allocate Christoffel symbols
        christoffel = torch.zeros_like(dg)
        
        # Compute Christoffel symbols: Γ^α_{μν} = 0.5 g^{αβ} (∂_μ g_{νβ} + ∂_ν g_{μβ} - ∂_β g_{μν})
        for i in range(self.lattice_size):
            for alpha in range(self.dimensions):
                for mu in range(self.dimensions):
                    for nu in range(self.dimensions):
                        term = 0.0
                        for beta in range(self.dimensions):
                            dg_mu_nu_beta = dg[i, mu, nu, beta] if mu < self.dimensions else 0.0
                            dg_nu_mu_beta = dg[i, nu, mu, beta] if nu < self.dimensions else 0.0
                            dg_beta_mu_nu = dg[i, beta, mu, nu] if beta < self.dimensions else 0.0
                            
                            term += 0.5 * g_inv[i, alpha, beta] * (
                                dg_mu_nu_beta + dg_nu_mu_beta - dg_beta_mu_nu
                            )
                        christoffel[i, alpha, mu, nu] = term
        
        return christoffel
        
    def compute_riemann_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Riemann curvature tensor R^α_{βμν}.
        
        The Riemann tensor is computed as:
        R^α_{βμν} = ∂_μ Γ^α_{νβ} - ∂_ν Γ^α_{μβ} + Γ^α_{μλ} Γ^λ_{νβ} - Γ^α_{νλ} Γ^λ_{μβ}
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Riemann curvature tensor with shape [lattice_size, d, d, d, d]
            where R^α_{βμν} = riemann[i,α,β,μ,ν]
        """
        if metric is None:
            metric = self.metric_field
            
        # Compute Christoffel symbols and their derivatives
        gamma = self.compute_christoffel_symbols(metric)
        d_gamma = self.compute_metric_derivatives(gamma, order=1)
        
        # Pre-allocate Riemann tensor
        riemann = torch.zeros(
            (self.lattice_size, self.dimensions, self.dimensions, 
             self.dimensions, self.dimensions),
            dtype=self.dtype,
            device=self.device,
        )
        
        # Compute Riemann tensor components
        for i in range(self.lattice_size):
            for alpha in range(self.dimensions):
                for beta in range(self.dimensions):
                    for mu in range(self.dimensions):
                        for nu in range(self.dimensions):
                            # ∂_μ Γ^α_{νβ} - ∂_ν Γ^α_{μβ}
                            term1 = d_gamma[i, mu, alpha, nu, beta] - d_gamma[i, nu, alpha, mu, beta]
                            
                            # Γ^α_{μλ} Γ^λ_{νβ} - Γ^α_{νλ} Γ^λ_{μβ}
                            term2 = 0.0
                            for lam in range(self.dimensions):
                                term2 += (
                                    gamma[i, alpha, mu, lam] * gamma[i, lam, nu, beta] -
                                    gamma[i, alpha, nu, lam] * gamma[i, lam, mu, beta]
                                )
                            
                            riemann[i, alpha, beta, mu, nu] = term1 + term2
        
        return riemann
    
    def compute_ricci_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Ricci tensor R_{μν} by contracting the Riemann tensor.
        
        The Ricci tensor is obtained by contracting the first and third indices
        of the Riemann tensor: R_{μν} = R^α_{μαν}
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Ricci tensor with shape [lattice_size, d, d]
        """
        riemann = self.compute_riemann_tensor(metric)
        # Contract first and third indices: R_{μν} = R^α_{μαν}
        return torch.einsum('iabic->ibc', riemann)
    
    def compute_ricci_scalar(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Ricci scalar R by contracting the Ricci tensor.
        
        The Ricci scalar is obtained by contracting the Ricci tensor with the
        inverse metric: R = g^{μν} R_{μν}
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Ricci scalar with shape [lattice_size]
        """
        if metric is None:
            metric = self.metric_field
            
        ricci = self.compute_ricci_tensor(metric)
        g_inv = torch.linalg.inv(metric)
        
        # Contract with inverse metric: R = g^{μν} R_{μν}
        return torch.einsum('iμν,iμν->i', g_inv, ricci)
    
    def compute_einstein_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Einstein tensor G_{μν} = R_{μν} - 1/2 R g_{μν}.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Einstein tensor with shape [lattice_size, d, d]
        """
        if metric is None:
            metric = self.metric_field
            
        ricci = self.compute_ricci_tensor(metric)
        ricci_scalar = self.compute_ricci_scalar(metric)
        
        # Reshape for broadcasting
        ricci_scalar = ricci_scalar.view(-1, 1, 1)
        
        # G_{μν} = R_{μν} - 1/2 R g_{μν}
        return ricci - 0.5 * ricci_scalar * metric
                            term = term + 0.5 * g_inv[alpha, sigma] * (
                                dg[mu, sigma, nu]
                                + dg[nu, sigma, mu]
                                - dg[sigma, mu, nu]
                            )
                        christoffel[i, alpha, mu, nu] = term

        return christoffel

    def compute_riemann_tensor(self) -> torch.Tensor:
        """Compute R^α_{βμν} for each lattice point."""

        christoffel = self.compute_christoffel_symbols()
        gamma_deriv = torch.zeros(
            (self.lattice_size, self.dimensions, self.dimensions, self.dimensions, self.dimensions),
            dtype=self.dtype,
            device=self.device,
        )
        gamma_deriv[:, 1, :, :, :] = self._finite_difference(christoffel)

        riemann = torch.zeros(
            (self.lattice_size, self.dimensions, self.dimensions, self.dimensions, self.dimensions),
            dtype=self.dtype,
            device=self.device,
        )

        for i in range(self.lattice_size):
            for alpha in range(self.dimensions):
                for beta in range(self.dimensions):
                    for mu in range(self.dimensions):
                        for nu in range(self.dimensions):
                            value = (
                                gamma_deriv[i, mu, alpha, beta, nu]
                                - gamma_deriv[i, nu, alpha, beta, mu]
                            )
                            for sigma in range(self.dimensions):
                                value = value + (
                                    christoffel[i, alpha, sigma, mu]
                                    * christoffel[i, sigma, beta, nu]
                                    - christoffel[i, alpha, sigma, nu]
                                    * christoffel[i, sigma, beta, mu]
                                )
                            riemann[i, alpha, beta, mu, nu] = value

        return riemann

    def compute_ricci_tensor(self) -> torch.Tensor:
        """Compute R_{βν} for each lattice point."""

        riemann = self.compute_riemann_tensor()
        ricci = torch.einsum("iαβαν->iβν", riemann)
        return ricci

    def compute_ricci_scalar(self) -> torch.Tensor:
        """Compute scalar curvature R at the active lattice index."""

        ricci = self.compute_ricci_tensor()
        g_inv = torch.linalg.inv(self.metric)
        active_ricci = ricci[self.active_index]
        scalar = torch.einsum("μν,μν->", g_inv, active_ricci)
        return scalar

    # ------------------------------------------------------------------
    # Metric updates
    # ------------------------------------------------------------------
    def update_metric(self, gradient: torch.Tensor, learning_rate: float) -> None:
        """Gradient descent update on the active metric component."""

        with torch.no_grad():
            updated = self.metric_field[self.active_index] - learning_rate * gradient
            updated = 0.5 * (updated + updated.t())

            # Keep metric close to the Lorentzian base to avoid degeneracy
            base = torch.diag(
                torch.tensor([-1.0] + [1.0] * (self.dimensions - 1), dtype=self.dtype, device=self.device)
            )
            delta = updated - base
            delta = torch.clamp(delta, -self.regularization, self.regularization)
            self.metric_field[self.active_index].copy_(base + delta)

