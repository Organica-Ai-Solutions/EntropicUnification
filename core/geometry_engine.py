"""
Geometry Engine: Represents and manipulates the spacetime metric gμν(t, x).

This implementation treats the metric as a differentiable field defined on a 1-D
spatial lattice (sufficient for 1+1 or 3+1 toy models where the metric varies
along a single spatial coordinate). The engine provides finite-difference
derivatives, Christoffel symbols, and curvature tensors that are all compatible
with PyTorch autograd so they can participate in the global optimisation loop.

Enhanced version includes:
- Higher-order curvature tensors (Weyl, Gauss-Bonnet)
- Support for non-Lorentzian signatures
- Improved numerical stability
- Configurable boundary conditions
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class BoundaryCondition(str, Enum):
    """Boundary conditions for the metric field."""
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ABSORBING = "absorbing"


class GeometryEngine(nn.Module):
    """Differentiable spacetime metric on a 1-D lattice with higher-order curvature support."""

    def __init__(
        self,
        dimensions: int,
        lattice_size: int,
        dx: float = 1.0,
        regularization: float = 1e-4,
        initial_metric: str = "minkowski",
        boundary_condition: Union[str, BoundaryCondition] = BoundaryCondition.PERIODIC,
        signature: List[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Initialize the geometry engine.
        
        Args:
            dimensions: Number of spacetime dimensions
            lattice_size: Size of the spatial lattice
            dx: Spatial step size
            regularization: Regularization parameter for metric updates
            initial_metric: Initial metric configuration ("minkowski", "frw", "schwarzschild")
            boundary_condition: Boundary condition for finite differences
            signature: Metric signature, e.g. [-1, 1, 1, 1] for Lorentzian
            device: PyTorch device to use
            dtype: PyTorch data type to use
        """
        super().__init__()
        if dimensions < 2:
            raise ValueError("GeometryEngine requires at least 2 spacetime dimensions")

        self.dimensions = dimensions
        self.lattice_size = lattice_size
        self.dx = torch.as_tensor(dx, dtype=dtype, device=device or torch.device("cpu"))
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.regularization = regularization
        
        # Set boundary condition
        if isinstance(boundary_condition, str):
            self.boundary_condition = BoundaryCondition(boundary_condition.lower())
        else:
            self.boundary_condition = boundary_condition

        # Set metric signature
        if signature is None:
            # Default to Lorentzian signature (-,+,+,+,...)
            self.signature = [-1] + [1] * (dimensions - 1)
        else:
            if len(signature) != dimensions:
                raise ValueError(f"Signature must have {dimensions} elements")
            self.signature = signature

        # Base metric from signature
        base = torch.diag(
            torch.tensor(self.signature, dtype=dtype, device=self.device)
        )
        field = base.repeat(lattice_size, 1, 1)

        self.metric_field = nn.Parameter(field)
        self.active_index = lattice_size // 2
        
        # Initialize higher-order curvature tensors
        self._weyl_tensor = None
        self._gauss_bonnet_term = None
        
        # Cache for expensive computations
        self._cache = {}
        
        # Initialize the metric field
        self._apply_initial_metric(initial_metric)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    @property
    def metric(self) -> torch.Tensor:
        """Return the metric tensor at the active lattice index."""
        return self.metric_field[self.active_index]

    def set_active_index(self, index: int) -> None:
        """Set the active lattice index.
        
        Args:
            index: New active index
        """
        if not 0 <= index < self.lattice_size:
            raise IndexError("Active metric index out of bounds")
        self.active_index = index
        # Clear cache when changing active index
        self._clear_cache()

    def _apply_initial_metric(self, name: str) -> None:
        """Initialize the metric field with a specific configuration.
        
        Args:
            name: Name of the initial metric configuration
        """
        with torch.no_grad():
            if name.lower() == "minkowski":
                # Already initialized to Minkowski; nothing else to do
                pass
            elif name.lower() == "frw":
                # Simple flat FRW with scale factor a(x) = 1 + ε x
                epsilon = 1e-3
                x = torch.linspace(-1.0, 1.0, self.lattice_size, device=self.device)
                a = 1.0 + epsilon * x
                for i in range(self.lattice_size):
                    g = self.metric_field[i]
                    g[0, 0] = self.signature[0]  # Time component
                    for j in range(1, self.dimensions):
                        g[j, j] = self.signature[j] * a[i] ** 2  # Spatial components
            elif name.lower() == "schwarzschild":
                # Simple Schwarzschild-like metric with a central mass
                # ds^2 = -(1-2M/r)dt^2 + (1-2M/r)^(-1)dr^2 + r^2 dΩ^2
                mass = 0.1  # Small mass parameter
                r = torch.linspace(2.1*mass, 10.0*mass, self.lattice_size, device=self.device)
                for i in range(self.lattice_size):
                    g = self.metric_field[i]
                    # Time component
                    g[0, 0] = -1.0 * (1.0 - 2.0*mass/r[i])
                    # Radial component
                    g[1, 1] = 1.0 / (1.0 - 2.0*mass/r[i])
                    # Angular components (if dimensions > 2)
                    if self.dimensions > 2:
                        g[2, 2] = r[i]**2  # θ component
                    if self.dimensions > 3:
                        g[3, 3] = r[i]**2 * torch.sin(torch.tensor(0.5*torch.pi))**2  # φ component
            else:
                raise ValueError(f"Unknown initial metric '{name}'")
            self._enforce_symmetry()

    def _enforce_symmetry(self) -> None:
        """Enforce symmetry of the metric tensor."""
        with torch.no_grad():
            sym_field = 0.5 * (self.metric_field + self.metric_field.transpose(-1, -2))
            self.metric_field.copy_(sym_field)
            
    def _clear_cache(self) -> None:
        """Clear the computation cache."""
        self._cache = {}
        self._weyl_tensor = None
        self._gauss_bonnet_term = None

    # ------------------------------------------------------------------
    # Finite differences with improved boundary handling
    # ------------------------------------------------------------------
    def _finite_difference(
        self, 
        tensor: torch.Tensor, 
        order: int = 1, 
        axis: int = 0
    ) -> torch.Tensor:
        """Compute finite differences with configurable boundary conditions.
        
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
        
        # Pad the tensor according to boundary condition
        pad_size = 2 if order == 2 else 1
        pad = [(0, 0)] * ndim
        pad[axis] = (pad_size, pad_size)
        
        if self.boundary_condition == BoundaryCondition.PERIODIC:
            # For periodic boundaries, wrap around
            padded = torch.nn.functional.pad(
                tensor, 
                [p for dim in reversed(pad) for p in dim], 
                mode='circular'
            )
        elif self.boundary_condition == BoundaryCondition.DIRICHLET:
            # For Dirichlet boundaries, use zero padding
            padded = torch.nn.functional.pad(
                tensor, 
                [p for dim in reversed(pad) for p in dim], 
                mode='constant', 
                value=0.0
            )
        elif self.boundary_condition == BoundaryCondition.NEUMANN:
            # For Neumann boundaries, use reflection padding
            padded = torch.nn.functional.pad(
                tensor, 
                [p for dim in reversed(pad) for p in dim], 
                mode='reflect'
            )
        else:  # Default to absorbing boundary
            # For absorbing boundaries, use replication padding
            padded = torch.nn.functional.pad(
                tensor, 
                [p for dim in reversed(pad) for p in dim], 
                mode='replicate'
            )
        
        if order == 1:
            # 4th order central difference for 1st derivative
            derivative = (
                -padded.narrow(axis, 4, tensor.size(axis)) 
                + 8 * padded.narrow(axis, 3, tensor.size(axis))
                - 8 * padded.narrow(axis, 1, tensor.size(axis))
                + padded.narrow(axis, 0, tensor.size(axis))
            ) / (12.0 * dx)
        else:
            # 4th order central difference for 2nd derivative
            derivative = (
                -padded.narrow(axis, 4, tensor.size(axis)) 
                + 16 * padded.narrow(axis, 3, tensor.size(axis))
                - 30 * padded.narrow(axis, 2, tensor.size(axis))
                + 16 * padded.narrow(axis, 1, tensor.size(axis))
                - padded.narrow(axis, 0, tensor.size(axis))
            ) / (12.0 * dx ** 2)
            
        return derivative

    # ------------------------------------------------------------------
    # Curvature calculations
    # ------------------------------------------------------------------
    def compute_christoffel_symbols(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Christoffel symbols from the metric tensor.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Christoffel symbols with shape [lattice_size, d, d, d]
        """
        cache_key = "christoffel"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        if metric is None:
            metric = self.metric_field
            
        # Compute metric derivatives
        dg = torch.zeros(
            (self.lattice_size, self.dimensions, self.dimensions, self.dimensions),
            dtype=self.dtype,
            device=self.device,
        )
        
        # Compute derivatives along each dimension
        for mu in range(self.dimensions):
            dg[:, mu] = self._finite_difference(metric, order=1, axis=0)
            
        # Compute inverse metric
        g_inv = torch.linalg.inv(metric)
        
        # Pre-allocate Christoffel symbols
        christoffel = torch.zeros(
            (self.lattice_size, self.dimensions, self.dimensions, self.dimensions),
            dtype=self.dtype,
            device=self.device,
        )
        
        # Compute Christoffel symbols: Γ^α_{μν} = 0.5 g^{αβ} (∂_μ g_{νβ} + ∂_ν g_{μβ} - ∂_β g_{μν})
        for i in range(self.lattice_size):
            for alpha in range(self.dimensions):
                for mu in range(self.dimensions):
                    for nu in range(self.dimensions):
                        term = 0.0
                        for beta in range(self.dimensions):
                            term += 0.5 * g_inv[i, alpha, beta] * (
                                dg[i, mu, nu, beta] + dg[i, nu, mu, beta] - dg[i, beta, mu, nu]
                            )
                        christoffel[i, alpha, mu, nu] = term
        
        # Cache the result
        self._cache[cache_key] = christoffel
        return christoffel

    def compute_riemann_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Riemann curvature tensor R^α_{βμν}.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Riemann tensor with shape [lattice_size, d, d, d, d]
        """
        cache_key = "riemann"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        if metric is None:
            metric = self.metric_field
            
        # Compute Christoffel symbols
        gamma = self.compute_christoffel_symbols(metric)
        
        # Compute derivatives of Christoffel symbols
        dgamma = torch.zeros(
            (self.lattice_size, self.dimensions, self.dimensions, 
             self.dimensions, self.dimensions),
            dtype=self.dtype,
            device=self.device,
        )
        
        for mu in range(self.dimensions):
            dgamma[:, mu] = self._finite_difference(gamma, order=1, axis=0)
        
        # Pre-allocate Riemann tensor
        riemann = torch.zeros(
            (self.lattice_size, self.dimensions, self.dimensions, 
             self.dimensions, self.dimensions),
            dtype=self.dtype,
            device=self.device,
        )
        
        # Compute Riemann tensor components
        # R^α_{βμν} = ∂_μ Γ^α_{νβ} - ∂_ν Γ^α_{μβ} + Γ^α_{μλ} Γ^λ_{νβ} - Γ^α_{νλ} Γ^λ_{μβ}
        for i in range(self.lattice_size):
            for alpha in range(self.dimensions):
                for beta in range(self.dimensions):
                    for mu in range(self.dimensions):
                        for nu in range(self.dimensions):
                            # ∂_μ Γ^α_{νβ} - ∂_ν Γ^α_{μβ}
                            term1 = dgamma[i, mu, alpha, nu, beta] - dgamma[i, nu, alpha, mu, beta]
                            
                            # Γ^α_{μλ} Γ^λ_{νβ} - Γ^α_{νλ} Γ^λ_{μβ}
                            term2 = 0.0
                            for lam in range(self.dimensions):
                                term2 += (
                                    gamma[i, alpha, mu, lam] * gamma[i, lam, nu, beta] -
                                    gamma[i, alpha, nu, lam] * gamma[i, lam, mu, beta]
                                )
                            
                            riemann[i, alpha, beta, mu, nu] = term1 + term2
        
        # Cache the result
        self._cache[cache_key] = riemann
        return riemann
    
    def compute_ricci_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Ricci tensor R_{μν} by contracting the Riemann tensor.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Ricci tensor with shape [lattice_size, d, d]
        """
        cache_key = "ricci"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        riemann = self.compute_riemann_tensor(metric)
        
        # Contract first and third indices: R_{μν} = R^α_{μαν}
        ricci = torch.einsum('iabic->ibc', riemann)
        
        # Cache the result
        self._cache[cache_key] = ricci
        return ricci
    
    def compute_ricci_scalar(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Ricci scalar R by contracting the Ricci tensor.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Ricci scalar with shape [lattice_size]
        """
        cache_key = "ricci_scalar"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        if metric is None:
            metric = self.metric_field
            
        ricci = self.compute_ricci_tensor(metric)
        g_inv = torch.linalg.inv(metric)
        
        # Contract with inverse metric: R = g^{μν} R_{μν}
        scalar = torch.einsum('iμν,iμν->i', g_inv, ricci)
        
        # Cache the result
        self._cache[cache_key] = scalar
        return scalar
    
    def compute_einstein_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Einstein tensor G_{μν} = R_{μν} - 1/2 R g_{μν}.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Einstein tensor with shape [lattice_size, d, d]
        """
        cache_key = "einstein"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        if metric is None:
            metric = self.metric_field
            
        ricci = self.compute_ricci_tensor(metric)
        ricci_scalar = self.compute_ricci_scalar(metric)
        
        # Reshape for broadcasting
        ricci_scalar = ricci_scalar.view(-1, 1, 1)
        
        # G_{μν} = R_{μν} - 1/2 R g_{μν}
        einstein = ricci - 0.5 * ricci_scalar * metric
        
        # Cache the result
        self._cache[cache_key] = einstein
        return einstein
    
    # ------------------------------------------------------------------
    # Higher-order curvature tensors
    # ------------------------------------------------------------------
    def compute_weyl_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Weyl conformal curvature tensor.
        
        The Weyl tensor is the traceless part of the Riemann tensor:
        C_{αβμν} = R_{αβμν} - (g_{α[μ}R_{ν]β} - g_{β[μ}R_{ν]α}) + (1/3)R g_{α[μ}g_{ν]β}
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Weyl tensor with shape [lattice_size, d, d, d, d]
        """
        if self._weyl_tensor is not None:
            return self._weyl_tensor
            
        if metric is None:
            metric = self.metric_field
            
        riemann = self.compute_riemann_tensor(metric)
        ricci = self.compute_ricci_tensor(metric)
        scalar = self.compute_ricci_scalar(metric).view(-1, 1, 1)
        
        # Pre-allocate Weyl tensor
        weyl = torch.zeros_like(riemann)
        
        # Dimension-dependent factor
        n = self.dimensions
        factor = 2.0 / ((n-1) * (n-2))
        
        # Compute Weyl tensor components
        for i in range(self.lattice_size):
            g = metric[i]
            R = ricci[i]
            
            for a in range(self.dimensions):
                for b in range(self.dimensions):
                    for m in range(self.dimensions):
                        for n in range(self.dimensions):
                            # Riemann part
                            weyl[i, a, b, m, n] = riemann[i, a, b, m, n]
                            
                            # Ricci part (antisymmetrized)
                            ricci_term = (
                                g[a, m] * R[n, b] - g[a, n] * R[m, b] -
                                g[b, m] * R[n, a] + g[b, n] * R[m, a]
                            )
                            weyl[i, a, b, m, n] -= factor * ricci_term
                            
                            # Scalar part (antisymmetrized)
                            scalar_term = scalar[i] * (
                                g[a, m] * g[n, b] - g[a, n] * g[m, b]
                            )
                            weyl[i, a, b, m, n] += factor/3.0 * scalar_term
        
        self._weyl_tensor = weyl
        return weyl
    
    def compute_gauss_bonnet_term(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Gauss-Bonnet term: R² - 4R_{μν}R^{μν} + R_{μναβ}R^{μναβ}.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Gauss-Bonnet term with shape [lattice_size]
        """
        if self._gauss_bonnet_term is not None:
            return self._gauss_bonnet_term
            
        if metric is None:
            metric = self.metric_field
            
        # Get necessary tensors
        riemann = self.compute_riemann_tensor(metric)
        ricci = self.compute_ricci_tensor(metric)
        scalar = self.compute_ricci_scalar(metric)
        g_inv = torch.linalg.inv(metric)
        
        # Compute squared terms
        ricci_squared = torch.einsum('iμν,iμσ,iνσ->i', ricci, g_inv, g_inv)
        riemann_squared = torch.einsum(
            'iαβμν,iγδρσ,iαγ,iβδ,iμρ,iνσ->i', 
            riemann, riemann, g_inv, g_inv, g_inv, g_inv
        )
        
        # Gauss-Bonnet term: R² - 4R_{μν}R^{μν} + R_{μναβ}R^{μναβ}
        gb_term = scalar**2 - 4*ricci_squared + riemann_squared
        
        self._gauss_bonnet_term = gb_term
        return gb_term
    
    # ------------------------------------------------------------------
    # Metric updates
    # ------------------------------------------------------------------
    def update_metric(self, gradient: torch.Tensor, learning_rate: float) -> None:
        """Gradient descent update on the active metric component.
        
        Args:
            gradient: Gradient of the loss with respect to the metric
            learning_rate: Learning rate for the update
        """
        with torch.no_grad():
            # Apply gradient descent update
            updated = self.metric_field[self.active_index] - learning_rate * gradient
            
            # Enforce symmetry
            updated = 0.5 * (updated + updated.t())

            # Keep metric close to the signature base to avoid degeneracy
            base = torch.diag(
                torch.tensor(self.signature, dtype=self.dtype, device=self.device)
            )
            delta = updated - base
            delta = torch.clamp(delta, -self.regularization, self.regularization)
            self.metric_field[self.active_index].copy_(base + delta)
            
            # Clear cache after metric update
            self._clear_cache()
    
    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def compute_determinant(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the determinant of the metric tensor.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Determinant with shape [lattice_size]
        """
        if metric is None:
            metric = self.metric_field
            
        return torch.linalg.det(metric)
    
    def compute_proper_volume(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the proper volume element sqrt(|g|).
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Volume element with shape [lattice_size]
        """
        det = self.compute_determinant(metric)
        return torch.sqrt(torch.abs(det))
    
    def compute_geodesic_equation(
        self, 
        position: torch.Tensor, 
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute the geodesic equation for a given position and velocity.
        
        d²x^α/dλ² + Γ^α_{μν} (dx^μ/dλ) (dx^ν/dλ) = 0
        
        Args:
            position: Position vector with shape [d]
            velocity: Velocity vector with shape [d]
            
        Returns:
            Acceleration vector with shape [d]
        """
        # Get Christoffel symbols at the active index
        gamma = self.compute_christoffel_symbols()[self.active_index]
        
        # Compute acceleration from geodesic equation
        acceleration = torch.zeros_like(position)
        
        for alpha in range(self.dimensions):
            term = 0.0
            for mu in range(self.dimensions):
                for nu in range(self.dimensions):
                    term -= gamma[alpha, mu, nu] * velocity[mu] * velocity[nu]
            acceleration[alpha] = term
            
        return acceleration
    
    def is_flat(self, tolerance: float = 1e-6) -> bool:
        """Check if the metric is flat (zero curvature).
        
        Args:
            tolerance: Tolerance for considering curvature as zero
            
        Returns:
            True if the metric is flat, False otherwise
        """
        ricci_scalar = self.compute_ricci_scalar()[self.active_index]
        return torch.abs(ricci_scalar) < tolerance