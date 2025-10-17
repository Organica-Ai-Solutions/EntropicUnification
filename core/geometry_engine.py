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
    def _finite_difference(self, tensor: torch.Tensor) -> torch.Tensor:
        """Central finite difference along the lattice axis."""

        derivative = torch.zeros_like(tensor)
        dx = self.dx

        # Central differences for interior points
        derivative[1:-1] = (tensor[2:] - tensor[:-2]) / (2.0 * dx)

        # One-sided differences for the boundaries
        derivative[0] = (tensor[1] - tensor[0]) / dx
        derivative[-1] = (tensor[-1] - tensor[-2]) / dx

        return derivative

    # ------------------------------------------------------------------
    # Curvature calculations
    # ------------------------------------------------------------------
    def compute_metric_derivatives(self) -> torch.Tensor:
        """∂_μ g_{σν} for each lattice point (only spatial derivative non-zero)."""

        deriv = torch.zeros(
            (self.lattice_size, self.dimensions, self.dimensions, self.dimensions),
            dtype=self.dtype,
            device=self.device,
        )

        spatial_grad = self._finite_difference(self.metric_field)
        # Assume derivative along coordinate index 1 corresponds to lattice axis
        deriv[:, 1, :, :] = spatial_grad
        return deriv

    def compute_christoffel_symbols(self) -> torch.Tensor:
        """Compute Γ^α_{μν} for every lattice point."""

        metric_derivatives = self.compute_metric_derivatives()
        christoffel = torch.zeros(
            (self.lattice_size, self.dimensions, self.dimensions, self.dimensions),
            dtype=self.dtype,
            device=self.device,
        )

        for i in range(self.lattice_size):
            g = self.metric_field[i]
            g_inv = torch.linalg.inv(g)
            dg = metric_derivatives[i]

            for alpha in range(self.dimensions):
                for mu in range(self.dimensions):
                    for nu in range(self.dimensions):
                        term = 0.0
                        for sigma in range(self.dimensions):
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

