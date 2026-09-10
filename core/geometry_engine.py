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

from .utils.finite_difference import fixed_finite_difference
from .validation import (GateConfig, ValidationReport, check_finite,
                         check_meaningful_dimension, check_metric_resolved,
                         check_riemann_identities)


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
        higher_curvature_terms: bool = False,
        alpha_GB: float = 0.1,
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
            higher_curvature_terms: Whether to include higher-order curvature terms
            alpha_GB: Gauss-Bonnet coupling constant
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
        self.higher_curvature_terms = higher_curvature_terms
        self.alpha_GB = alpha_GB

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
        """dx-normalized finite difference along the lattice axis.

        Delegates to core.utils.finite_difference.fixed_finite_difference so
        that first derivatives carry units of 1/dx and second derivatives
        1/dx**2. (An earlier version of this method omitted the dx division,
        which made every curvature quantity dimensionally wrong.)
        """
        return fixed_finite_difference(tensor, order=order, axis=axis, dx=float(self.dx))

    # ------------------------------------------------------------------
    # Spectral methods for higher accuracy
    # ------------------------------------------------------------------
    def spectral_derivative(self, tensor: torch.Tensor, order: int = 1, axis: int = 0) -> torch.Tensor:
        """Compute derivatives using spectral methods (Fourier transform).
        
        This is highly accurate for periodic boundaries but requires the tensor
        to have periodic boundary conditions.
        
        Args:
            tensor: Input tensor to take derivatives of
            order: Order of the derivative (1 or 2 supported)
            axis: Axis along which to take the derivative
            
        Returns:
            Tensor containing the spectral derivative approximation
        """
        if self.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("Spectral derivatives only work with periodic boundary conditions")
            
        if order not in (1, 2):
            raise ValueError("Only 1st and 2nd order derivatives are supported")
            
        # Get shape along derivative axis
        n = tensor.shape[axis]
        
        # Compute frequency components
        k = torch.fft.rfftfreq(n, d=self.dx) * 2 * torch.pi
        
        # Reshape k for proper broadcasting
        shape = [1] * tensor.dim()
        shape[axis] = k.shape[0]
        k = k.reshape(shape)
        
        # Convert to frequency domain
        ft = torch.fft.rfft(tensor, dim=axis)
        
        # Apply frequency domain derivative
        if order == 1:
            # First derivative: multiply by i*k
            ft = 1j * k * ft
        elif order == 2:
            # Second derivative: multiply by -k²
            ft = -(k**2) * ft
            
        # Convert back to spatial domain
        result = torch.fft.irfft(ft, n=n, dim=axis)
        
        return result
    
    def spectral_laplacian(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute the Laplacian using spectral methods.
        
        This computes ∇²f for a scalar field f.
        
        Args:
            tensor: Input tensor (scalar field)
            
        Returns:
            Laplacian of the input tensor
        """
        result = torch.zeros_like(tensor)
        
        # Compute second derivative along each axis
        for axis in range(self.dimensions):
            result += self.spectral_derivative(tensor, order=2, axis=axis)
            
        return result

    # ------------------------------------------------------------------
    # Covariant derivatives
    # ------------------------------------------------------------------
    def covariant_derivative(self, tensor: torch.Tensor, index: int) -> torch.Tensor:
        """Compute the covariant derivative of a tensor.
        
        This handles the proper transformation rules based on tensor rank.
        Currently supports scalar and vector fields.
        
        Args:
            tensor: Input tensor (scalar or vector field)
            index: Index to take the derivative with respect to
            
        Returns:
            Covariant derivative of the tensor
        """
        # Compute Christoffel symbols if not already cached
        christoffel = self.compute_christoffel_symbols()
        
        # Get partial derivative
        partial = self._finite_difference(tensor, order=1, axis=index)
        
        # For scalar fields, covariant derivative equals partial derivative
        if tensor.dim() == self.dimensions + 1:  # [lattice_size, ...]
            return partial
            
        # For vector fields, add Christoffel symbol terms
        elif tensor.dim() == self.dimensions + 2:  # [lattice_size, vector_dim, ...]
            result = partial.clone()
            
            # Add Christoffel terms: ∇_i V^j = ∂_i V^j + Γ^j_ik V^k
            for j in range(self.dimensions):
                for k in range(self.dimensions):
                    result[..., j, :] += christoffel[..., j, index, k] * tensor[..., k, :]
                    
            return result
            
        else:
            raise ValueError(f"Unsupported tensor rank: {tensor.dim() - self.dimensions}")
            
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
        use_cache = self._cacheable(metric)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if metric is None:
            metric = self.metric_field
            
        lattice_size = metric.shape[0]
        dim = self.dimensions

        # Batched inverse metric g^μσ
        g_inv = torch.linalg.inv(metric)

        # The metric field is static and homogeneous in every coordinate
        # except x^1, the coordinate the lattice discretizes.  Therefore
        # ∂_α g_μν = 0 for α ≠ 1, and ∂_1 g_μν is the lattice finite
        # difference.  (An earlier version copied the lattice derivative
        # into *every* coordinate slot, which does not correspond to any
        # metric ansatz.)
        dg = self._finite_difference(metric, order=1, axis=0)  # (N, d, d)
        dg_full = torch.zeros(
            (lattice_size, dim, dim, dim), dtype=metric.dtype, device=metric.device
        )
        dg_full[:, 1] = dg  # dg_full[i, α, μ, ν] = ∂_α g_μν

        # Γ^μ_αβ = (1/2) g^μσ (∂_α g_σβ + ∂_β g_σα − ∂_σ g_αβ)
        # Build S[i, σ, α, β] = ∂_α g_σβ + ∂_β g_σα − ∂_σ g_αβ
        s_term = (
            dg_full.permute(0, 2, 1, 3)      # [i, σ, α, β] = ∂_α g_σβ
            + dg_full.permute(0, 2, 3, 1)    # [i, σ, α, β] = ∂_β g_σα
            - dg_full                        # [i, σ, α, β] = ∂_σ g_αβ
        )
        christoffel = 0.5 * torch.einsum("ims,isab->imab", g_inv, s_term)

        # Cache the result
        if use_cache:
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
        use_cache = self._cacheable(metric)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if metric is None:
            metric = self.metric_field
            
        # Compute Christoffel symbols
        gamma = self.compute_christoffel_symbols(metric)

        dim = self.dimensions
        lattice_size = metric.shape[0]

        # ∂_α Γ: nonzero only for α = 1 (the lattice coordinate) — same
        # static, single-coordinate ansatz as compute_christoffel_symbols.
        dgamma_lattice = self._finite_difference(gamma, order=1, axis=0)  # (N, d, d, d)
        dgamma = torch.zeros(
            (lattice_size, dim, dim, dim, dim), dtype=metric.dtype, device=metric.device
        )
        dgamma[:, 1] = dgamma_lattice  # dgamma[i, α, μ, ν, β] = ∂_α Γ^μ_νβ

        # R^α_{βμν} = ∂_μ Γ^α_{νβ} − ∂_ν Γ^α_{μβ} + Γ^α_{μλ} Γ^λ_{νβ} − Γ^α_{νλ} Γ^λ_{μβ}
        # term1[i, α, β, μ, ν] = ∂_μ Γ^α_{νβ}: dgamma dims are (i, deriv, up, low1, low2)
        dG = dgamma.permute(0, 2, 4, 1, 3)  # [i, α, β, μ, ν] = dgamma[i, μ, α, ν, β]
        term1 = dG - dG.transpose(3, 4)

        # term2[i, α, β, μ, ν] = Γ^α_{μλ} Γ^λ_{νβ} (antisymmetrized in μ↔ν)
        gg = torch.einsum("iaml,ilnb->iabmn", gamma, gamma)
        term2 = gg - gg.transpose(3, 4)

        # No symmetry "enforcement": the mixed-index R^α_{βμν} does not carry
        # the fully-lowered symmetries, and projecting them in silently
        # corrupts the computed curvature.  Use riemann_identity_violations()
        # to *measure* how well the lowered tensor satisfies its identities.
        riemann = term1 + term2

        # Cache the result
        if use_cache:
            self._cache[cache_key] = riemann
        return riemann

    def lower_riemann_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the fully-lowered Riemann tensor R_{αβμν} = g_{αλ} R^λ_{βμν}."""
        if metric is None:
            metric = self.metric_field
        riemann = self.compute_riemann_tensor(metric)
        return torch.einsum("ial,ilbmn->iabmn", metric, riemann)

    def riemann_identity_violations(
        self, metric: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Diagnostic: measure how badly the lowered Riemann tensor violates
        its algebraic identities (antisymmetry, pair symmetry, first Bianchi).

        Returns relative Frobenius-norm violations.  Large values indicate
        discretization error or an unphysical metric state — they are
        reported, never silently projected away.
        """
        rl = self.lower_riemann_tensor(metric).detach()
        norm = torch.linalg.norm(rl.flatten()) + 1e-30
        antisym_first = torch.linalg.norm((rl + rl.transpose(1, 2)).flatten()) / norm
        antisym_last = torch.linalg.norm((rl + rl.transpose(3, 4)).flatten()) / norm
        pair_sym = torch.linalg.norm((rl - rl.permute(0, 3, 4, 1, 2)).flatten()) / norm
        bianchi = torch.linalg.norm(
            (rl + rl.permute(0, 1, 3, 4, 2) + rl.permute(0, 1, 4, 2, 3)).flatten()
        ) / norm
        return {
            "antisymmetry_first_pair": float(antisym_first),
            "antisymmetry_last_pair": float(antisym_last),
            "pair_symmetry": float(pair_sym),
            "first_bianchi": float(bianchi),
        }
    
    def _cacheable(self, metric: Optional[torch.Tensor]) -> bool:
        """Whether a result computed from `metric` may use the shared cache.

        The curvature caches are keyed by name only ("christoffel", "riemann"),
        so they are valid solely for `self.metric_field`.  Passing a different
        metric and hitting a warm cache silently mixes tensors from two
        different metrics — the lowering step would combine cached Christoffels
        of one with the metric of another, producing a result belonging to
        neither.  An explicitly supplied foreign metric therefore bypasses the
        cache entirely.
        """
        return metric is None or metric is self.metric_field

    def metric_truncation_scale(
        self, metric: Optional[torch.Tensor] = None
    ) -> float:
        """Dimensionless truncation parameter ``||d2g|| dx^2 / ||g||``.

        This is ``(dx/lambda)^2`` for a metric varying on characteristic
        length ``lambda`` — the size of the second-order finite-difference
        error in every derived tensor.  It is what makes the Riemann identity
        gate resolution-independent: the violations are compared against the
        error the metric's own smoothness predicts, rather than against a
        constant that is too tight on a coarse lattice and too loose on a
        fine one.

        A smooth, well-resolved metric gives a small value that falls as the
        lattice refines.  Per-site noise gives a large value that does not.
        """
        m = self.metric_field if metric is None else metric
        with torch.no_grad():
            m = m.detach()
            if m.shape[0] < 3:
                return 0.0
            # second difference in lattice units; the dx^2 of the derivative
            # and the dx^2 of the error cancel, so this is already the
            # dimensionless ratio.
            d2 = m[2:] - 2.0 * m[1:-1] + m[:-2]
            return float(
                torch.linalg.norm(d2.flatten())
                / (torch.linalg.norm(m[1:-1].flatten()) + 1e-30)
            )

    def validate_curvature(
        self,
        metric: Optional[torch.Tensor] = None,
        gates: Optional[GateConfig] = None,
        report: Optional[ValidationReport] = None,
    ) -> ValidationReport:
        """Gate the curvature pipeline: identities, finiteness, dimension.

        Call this before trusting anything computed from the metric. Unlike
        `riemann_identity_violations`, which reports numbers a human has to
        notice, this raises when a tolerance is exceeded — v1.2's curvature
        was wrong while every printed diagnostic looked fine, because the
        symmetries had been projected in rather than checked.
        """
        cfg = GateConfig() if gates is None else gates
        rep = ValidationReport() if report is None else report
        m = self.metric_field if metric is None else metric

        check_meaningful_dimension(self.dimensions, cfg, rep)
        check_finite("metric", m, cfg, rep)
        truncation = self.metric_truncation_scale(m)
        check_metric_resolved(truncation, cfg, rep)
        check_riemann_identities(self.riemann_identity_violations(m), cfg, rep,
                                 truncation=truncation)
        check_finite("einstein_tensor", self.compute_einstein_tensor(m), cfg, rep)
        return rep

    def compute_ricci_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Ricci tensor by contracting the Riemann tensor.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Ricci tensor with shape [lattice_size, d, d]
        """
        cache_key = "ricci"
        use_cache = self._cacheable(metric)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        if metric is None:
            metric = self.metric_field
        
        # Get Riemann tensor
        riemann = self.compute_riemann_tensor(metric)

        # R_μν = R^λ_{μλν} — contraction of the first and third indices.
        # Computed honestly, with no symmetrization: for a valid metric the
        # result is symmetric up to discretization error, and any asymmetry
        # is a diagnostic worth seeing rather than hiding.
        ricci = torch.einsum("ilmln->imn", riemann)

        # Cache the result
        if use_cache:
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
        use_cache = self._cacheable(metric)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
            
        if metric is None:
            metric = self.metric_field
            
        ricci = self.compute_ricci_tensor(metric)
        g_inv = torch.linalg.inv(metric)
        
        # Contract with inverse metric: R = g^{μν} R_{μν}
        scalar = torch.einsum("imn,imn->i", g_inv, ricci)

        # Cache the result
        if use_cache:
            self._cache[cache_key] = scalar
        return scalar
    
    def compute_higher_curvature_terms(
        self, 
        metric: torch.Tensor,
        ricci: torch.Tensor,
        scalar: torch.Tensor
    ) -> torch.Tensor:
        """
        Heuristic higher-curvature correction (NOT the true Gauss-Bonnet
        variation): H_μν = R_μλ R^λ_ν − (1/4) g_μν R².  Off by default;
        for the honest Gauss-Bonnet scalar see compute_gauss_bonnet_term().

        Args:
            metric: Metric tensor
            ricci: Ricci tensor
            scalar: Ricci scalar
            
        Returns:
            Higher curvature terms with shape [lattice_size, d, d]
        """
        lattice_size = metric.shape[0]
        dim = self.dimensions
        higher_curvature = torch.zeros((lattice_size, dim, dim), dtype=metric.dtype, device=metric.device)
        
        # Compute simple Gauss-Bonnet-like term
        for i in range(lattice_size):
            # Compute Ricci tensor squared
            ricci_squared = torch.matmul(ricci[i], ricci[i])
            
            # Gauss-Bonnet contribution: H_μν = R_μλ R^λ_ν - (1/4) g_μν R^2
            for mu in range(dim):
                for nu in range(dim):
                    higher_curvature[i, mu, nu] = ricci_squared[mu, nu] - 0.25 * (scalar[i]**2) * metric[i, mu, nu]
        
        return self.alpha_GB * higher_curvature
    
    def compute_einstein_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Einstein tensor G_{μν} = R_{μν} - 1/2 R g_{μν}.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Einstein tensor with shape [lattice_size, d, d]
        """
        cache_key = "einstein"
        use_cache = self._cacheable(metric)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
            
        if metric is None:
            metric = self.metric_field
            
        ricci = self.compute_ricci_tensor(metric)
        ricci_scalar = self.compute_ricci_scalar(metric)
        
        # Reshape for broadcasting
        ricci_scalar = ricci_scalar.view(-1, 1, 1)
        
        # G_{μν} = R_{μν} - 1/2 R g_{μν}
        einstein = ricci - 0.5 * ricci_scalar * metric
        
        # Add higher curvature terms if enabled
        if self.higher_curvature_terms:
            higher_curvature = self.compute_higher_curvature_terms(metric, ricci, ricci_scalar.squeeze())
            einstein = einstein + higher_curvature
        
        # Cache the result
        if use_cache:
            self._cache[cache_key] = einstein
        return einstein
    
    # ------------------------------------------------------------------
    # Higher-order curvature tensors
    # ------------------------------------------------------------------
    def compute_weyl_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the fully-lowered Weyl conformal curvature tensor C_{abmn}.

        C_{abmn} = R_{abmn}
                   - (1/(n-2)) (g_{am} R_{nb} - g_{an} R_{mb}
                                - g_{bm} R_{na} + g_{bn} R_{ma})
                   + (R/((n-1)(n-2))) (g_{am} g_{nb} - g_{an} g_{mb})

        The Weyl tensor vanishes identically for n <= 3, so zeros are
        returned in that case.

        Args:
            metric: Input metric tensor. If None, uses self.metric_field

        Returns:
            Weyl tensor with shape [lattice_size, d, d, d, d]
        """
        if self._weyl_tensor is not None:
            return self._weyl_tensor

        if metric is None:
            metric = self.metric_field

        n = self.dimensions
        rl = self.lower_riemann_tensor(metric)  # fully lowered R_{abmn}

        if n <= 3:
            # Weyl is identically zero in 2 and 3 dimensions.
            self._weyl_tensor = torch.zeros_like(rl)
            return self._weyl_tensor

        ricci = self.compute_ricci_tensor(metric)
        scalar = self.compute_ricci_scalar(metric)

        g = metric
        # ricci_part[i,a,b,m,n] = g_am R_nb - g_an R_mb - g_bm R_na + g_bn R_ma
        ricci_part = (
            torch.einsum("iam,inb->iabmn", g, ricci)
            - torch.einsum("ian,imb->iabmn", g, ricci)
            - torch.einsum("ibm,ina->iabmn", g, ricci)
            + torch.einsum("ibn,ima->iabmn", g, ricci)
        )
        # scalar_part[i,a,b,m,n] = g_am g_nb - g_an g_mb
        scalar_part = (
            torch.einsum("iam,inb->iabmn", g, g)
            - torch.einsum("ian,imb->iabmn", g, g)
        )

        weyl = (
            rl
            - ricci_part / (n - 2)
            + scalar.view(-1, 1, 1, 1, 1) * scalar_part / ((n - 1) * (n - 2))
        )

        self._weyl_tensor = weyl
        return weyl

    def compute_gauss_bonnet_term(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Gauss-Bonnet scalar: R**2 - 4 R_mn R^mn + R_abmn R^abmn.

        Args:
            metric: Input metric tensor. If None, uses self.metric_field

        Returns:
            Gauss-Bonnet term with shape [lattice_size]
        """
        if self._gauss_bonnet_term is not None:
            return self._gauss_bonnet_term

        if metric is None:
            metric = self.metric_field

        ricci = self.compute_ricci_tensor(metric)
        scalar = self.compute_ricci_scalar(metric)
        g_inv = torch.linalg.inv(metric)
        rl = self.lower_riemann_tensor(metric)  # fully lowered R_{abmn}

        # R_mn R^mn = g^ma g^nb R_mn R_ab
        ricci_squared = torch.einsum(
            "imn,ima,inb,iab->i", ricci, g_inv, g_inv, ricci
        )
        # R_abmn R^abmn = g^ap g^bq g^mr g^ns R_abmn R_pqrs
        riemann_squared = torch.einsum(
            "iabmn,iap,ibq,imr,ins,ipqrs->i", rl, g_inv, g_inv, g_inv, g_inv, rl
        )

        gb_term = scalar**2 - 4 * ricci_squared + riemann_squared

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