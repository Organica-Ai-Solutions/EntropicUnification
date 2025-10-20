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
        """
        A robust implementation of finite difference that handles different tensor dimensions.
        
        Args:
            tensor: Input tensor to take derivatives of
            order: Order of the derivative (1 or 2)
            axis: Axis along which to take the derivative
            
        Returns:
            Tensor containing the finite difference approximation
        """
        # Get tensor shape
        shape = tensor.shape
        
        # Check if the axis is valid
        if axis >= len(shape):
            raise ValueError(f"Axis {axis} is out of range for tensor with {len(shape)} dimensions")
        
        # Initialize result tensor
        result = torch.zeros_like(tensor)
        
        # Get the size along the derivative axis
        axis_size = shape[axis]
        
        if order == 1:
            # First derivative
            # For interior points, use central difference
            for i in range(1, axis_size - 1):
                # Create slices for i-1, i, and i+1
                s_prev = [slice(None)] * len(shape)
                s_curr = [slice(None)] * len(shape)
                s_next = [slice(None)] * len(shape)
                
                s_prev[axis] = i - 1
                s_curr[axis] = i
                s_next[axis] = i + 1
                
                # Central difference
                result[tuple(s_curr)] = (tensor[tuple(s_next)] - tensor[tuple(s_prev)]) / 2.0
            
            # Forward difference for first point
            if axis_size > 1:
                s_first = [slice(None)] * len(shape)
                s_second = [slice(None)] * len(shape)
                s_first[axis] = 0
                s_second[axis] = 1
                
                result[tuple(s_first)] = tensor[tuple(s_second)] - tensor[tuple(s_first)]
            
            # Backward difference for last point
            if axis_size > 1:
                s_last = [slice(None)] * len(shape)
                s_second_last = [slice(None)] * len(shape)
                s_last[axis] = axis_size - 1
                s_second_last[axis] = axis_size - 2
                
                result[tuple(s_last)] = tensor[tuple(s_last)] - tensor[tuple(s_second_last)]
        
        elif order == 2:
            # Second derivative
            # For interior points, use central difference
            for i in range(1, axis_size - 1):
                # Create slices for i-1, i, and i+1
                s_prev = [slice(None)] * len(shape)
                s_curr = [slice(None)] * len(shape)
                s_next = [slice(None)] * len(shape)
                
                s_prev[axis] = i - 1
                s_curr[axis] = i
                s_next[axis] = i + 1
                
                # Central difference for second derivative
                result[tuple(s_curr)] = tensor[tuple(s_next)] - 2 * tensor[tuple(s_curr)] + tensor[tuple(s_prev)]
            
            # Forward difference for first point
            if axis_size > 2:
                s_first = [slice(None)] * len(shape)
                s_second = [slice(None)] * len(shape)
                s_third = [slice(None)] * len(shape)
                s_first[axis] = 0
                s_second[axis] = 1
                s_third[axis] = 2
                
                result[tuple(s_first)] = tensor[tuple(s_third)] - 2 * tensor[tuple(s_second)] + tensor[tuple(s_first)]
            
            # Backward difference for last point
            if axis_size > 2:
                s_last = [slice(None)] * len(shape)
                s_second_last = [slice(None)] * len(shape)
                s_third_last = [slice(None)] * len(shape)
                s_last[axis] = axis_size - 1
                s_second_last[axis] = axis_size - 2
                s_third_last[axis] = axis_size - 3
                
                result[tuple(s_last)] = tensor[tuple(s_last)] - 2 * tensor[tuple(s_second_last)] + tensor[tuple(s_third_last)]
        
        else:
            raise ValueError("Only 1st and 2nd order derivatives are supported")
        
        return result
    
    def _handle_boundaries_first_derivative(self, tensor, result, axis, slices):
        """Handle boundary points for first derivative with appropriate one-sided differences."""
        axis_size = tensor.shape[axis]
        
        # Create slices for boundary regions
        for i in range(3):  # First 3 points
            s = [slice(None)] * tensor.dim()
            s[axis] = i
            boundary = tuple(s)
            
            if i == 0:  # First point: Forward difference (4th order)
                # f'(x) ≈ (-25f(x) + 48f(x+h) - 36f(x+2h) + 16f(x+3h) - 3f(x+4h)) / (12h)
                result[boundary] = (
                    -25 * tensor[slices[0]] + 
                    48 * tensor[slices[1]] - 
                    36 * tensor[slices[2]] + 
                    16 * tensor[slices[3]] - 
                    3 * tensor[tuple(self._get_offset_slice(tensor, axis, 4))]
                ) / (12.0 * self.dx)
                
            elif i == 1:  # Second point: Forward-biased difference (4th order)
                # f'(x) ≈ (-3f(x-1h) - 10f(x) + 18f(x+1h) - 6f(x+2h) + f(x+3h)) / (12h)
                result[boundary] = (
                    -3 * tensor[slices[-1]] - 
                    10 * tensor[slices[0]] + 
                    18 * tensor[slices[1]] - 
                    6 * tensor[slices[2]] + 
                    tensor[slices[3]]
                ) / (12.0 * self.dx)
                
            elif i == 2:  # Third point: Central difference (4th order)
                # f'(x) ≈ (-f(x-2h) + 8f(x-h) - 8f(x+h) + f(x+2h)) / (12h)
                result[boundary] = (
                    -tensor[slices[-2]] + 
                    8 * tensor[slices[-1]] - 
                    8 * tensor[slices[1]] + 
                    tensor[slices[2]]
                ) / (12.0 * self.dx)
        
        # Last 3 points
        for i in range(3):
            s = [slice(None)] * tensor.dim()
            s[axis] = axis_size - 3 + i
            boundary = tuple(s)
            
            if i == 0:  # Third-to-last point: Central difference (4th order)
                # f'(x) ≈ (-f(x-2h) + 8f(x-h) - 8f(x+h) + f(x+2h)) / (12h)
                result[boundary] = (
                    -tensor[tuple(self._get_offset_slice(tensor, axis, -2))] + 
                    8 * tensor[tuple(self._get_offset_slice(tensor, axis, -1))] - 
                    8 * tensor[tuple(self._get_offset_slice(tensor, axis, 1))] + 
                    tensor[tuple(self._get_offset_slice(tensor, axis, 2))]
                ) / (12.0 * self.dx)
                
            elif i == 1:  # Second-to-last point: Backward-biased difference (4th order)
                # f'(x) ≈ (-f(x-3h) + 6f(x-2h) - 18f(x-h) + 10f(x) + 3f(x+h)) / (12h)
                result[boundary] = (
                    -tensor[tuple(self._get_offset_slice(tensor, axis, -3))] + 
                    6 * tensor[tuple(self._get_offset_slice(tensor, axis, -2))] - 
                    18 * tensor[tuple(self._get_offset_slice(tensor, axis, -1))] + 
                    10 * tensor[slices[0]] + 
                    3 * tensor[tuple(self._get_offset_slice(tensor, axis, 1))]
                ) / (12.0 * self.dx)
                
            elif i == 2:  # Last point: Backward difference (4th order)
                # f'(x) ≈ (3f(x-4h) - 16f(x-3h) + 36f(x-2h) - 48f(x-h) + 25f(x)) / (12h)
                result[boundary] = (
                    3 * tensor[tuple(self._get_offset_slice(tensor, axis, -4))] - 
                    16 * tensor[tuple(self._get_offset_slice(tensor, axis, -3))] + 
                    36 * tensor[tuple(self._get_offset_slice(tensor, axis, -2))] - 
                    48 * tensor[tuple(self._get_offset_slice(tensor, axis, -1))] + 
                    25 * tensor[slices[0]]
                ) / (12.0 * self.dx)
    
    def _handle_boundaries_second_derivative(self, tensor, result, axis, slices):
        """Handle boundary points for second derivative with appropriate one-sided differences."""
        axis_size = tensor.shape[axis]
        
        # Create slices for boundary regions
        for i in range(3):  # First 3 points
            s = [slice(None)] * tensor.dim()
            s[axis] = i
            boundary = tuple(s)
            
            if i == 0:  # First point: Forward difference (4th order)
                # f''(x) ≈ (45f(x) - 154f(x+h) + 214f(x+2h) - 156f(x+3h) + 61f(x+4h) - 10f(x+5h)) / (12h²)
                result[boundary] = (
                    45 * tensor[slices[0]] - 
                    154 * tensor[slices[1]] + 
                    214 * tensor[slices[2]] - 
                    156 * tensor[slices[3]] + 
                    61 * tensor[tuple(self._get_offset_slice(tensor, axis, 4))] - 
                    10 * tensor[tuple(self._get_offset_slice(tensor, axis, 5))]
                ) / (12.0 * self.dx ** 2)
                
            elif i == 1:  # Second point: Forward-biased difference (4th order)
                # f''(x) ≈ (10f(x-1h) - 15f(x) - 4f(x+1h) + 14f(x+2h) - 6f(x+3h) + f(x+4h)) / (12h²)
                result[boundary] = (
                    10 * tensor[slices[-1]] - 
                    15 * tensor[slices[0]] - 
                    4 * tensor[slices[1]] + 
                    14 * tensor[slices[2]] - 
                    6 * tensor[slices[3]] + 
                    tensor[tuple(self._get_offset_slice(tensor, axis, 4))]
                ) / (12.0 * self.dx ** 2)
                
            elif i == 2:  # Third point: Central-biased difference (4th order)
                # f''(x) ≈ (f(x-2h) - 16f(x-h) + 30f(x) - 16f(x+h) + f(x+2h)) / (12h²)
                result[boundary] = (
                    tensor[slices[-2]] - 
                    16 * tensor[slices[-1]] + 
                    30 * tensor[slices[0]] - 
                    16 * tensor[slices[1]] + 
                    tensor[slices[2]]
                ) / (12.0 * self.dx ** 2)
        
        # Last 3 points
        for i in range(3):
            s = [slice(None)] * tensor.dim()
            s[axis] = axis_size - 3 + i
            boundary = tuple(s)
            
            if i == 0:  # Third-to-last point: Central-biased difference (4th order)
                # f''(x) ≈ (f(x-2h) - 16f(x-h) + 30f(x) - 16f(x+h) + f(x+2h)) / (12h²)
                result[boundary] = (
                    tensor[tuple(self._get_offset_slice(tensor, axis, -2))] - 
                    16 * tensor[tuple(self._get_offset_slice(tensor, axis, -1))] + 
                    30 * tensor[slices[0]] - 
                    16 * tensor[tuple(self._get_offset_slice(tensor, axis, 1))] + 
                    tensor[tuple(self._get_offset_slice(tensor, axis, 2))]
                ) / (12.0 * self.dx ** 2)
                
            elif i == 1:  # Second-to-last point: Backward-biased difference (4th order)
                # f''(x) ≈ (f(x-4h) - 6f(x-3h) + 14f(x-2h) - 4f(x-h) - 15f(x) + 10f(x+h)) / (12h²)
                result[boundary] = (
                    tensor[tuple(self._get_offset_slice(tensor, axis, -4))] - 
                    6 * tensor[tuple(self._get_offset_slice(tensor, axis, -3))] + 
                    14 * tensor[tuple(self._get_offset_slice(tensor, axis, -2))] - 
                    4 * tensor[tuple(self._get_offset_slice(tensor, axis, -1))] - 
                    15 * tensor[slices[0]] + 
                    10 * tensor[tuple(self._get_offset_slice(tensor, axis, 1))]
                ) / (12.0 * self.dx ** 2)
                
            elif i == 2:  # Last point: Backward difference (4th order)
                # f''(x) ≈ (-10f(x-5h) + 61f(x-4h) - 156f(x-3h) + 214f(x-2h) - 154f(x-h) + 45f(x)) / (12h²)
                result[boundary] = (
                    -10 * tensor[tuple(self._get_offset_slice(tensor, axis, -5))] + 
                    61 * tensor[tuple(self._get_offset_slice(tensor, axis, -4))] - 
                    156 * tensor[tuple(self._get_offset_slice(tensor, axis, -3))] + 
                    214 * tensor[tuple(self._get_offset_slice(tensor, axis, -2))] - 
                    154 * tensor[tuple(self._get_offset_slice(tensor, axis, -1))] + 
                    45 * tensor[slices[0]]
                ) / (12.0 * self.dx ** 2)
    
    def _get_offset_slice(self, tensor, axis, offset):
        """Get a slice with the given offset, handling boundary conditions."""
        s = [slice(None)] * tensor.dim()
        axis_size = tensor.shape[axis]
        
        # Handle boundaries based on boundary condition
        if 0 <= offset < axis_size:
            s[axis] = offset
        elif self.boundary_condition == BoundaryCondition.PERIODIC:
            s[axis] = offset % axis_size
        elif self.boundary_condition == BoundaryCondition.DIRICHLET:
            s[axis] = max(0, min(offset, axis_size - 1))
        elif self.boundary_condition == BoundaryCondition.NEUMANN:
            if offset < 0:
                s[axis] = -offset - 1
            elif offset >= axis_size:
                s[axis] = 2 * axis_size - offset - 1
            else:
                s[axis] = offset
        else:  # ABSORBING or default
            s[axis] = max(0, min(offset, axis_size - 1))
            
        return s
    
    def _finite_difference_simple(self, tensor, order, axis):
        """Simplified finite difference implementation as fallback for small tensors."""
        result = torch.zeros_like(tensor)
        
        if order == 1:
            # Simple central difference for first derivative
            if tensor.shape[axis] > 2:
                # Get slices for i+1 and i-1
                slice_plus = [slice(None)] * tensor.dim()
                slice_minus = [slice(None)] * tensor.dim()
                slice_center = [slice(None)] * tensor.dim()
                
                # For interior points
                slice_plus[axis] = slice(2, None)
                slice_center[axis] = slice(1, -1)
                slice_minus[axis] = slice(0, -2)
                
                # Central difference: (f(x+h) - f(x-h)) / 2h
                result[tuple(slice_center)] = (tensor[tuple(slice_plus)] - tensor[tuple(slice_minus)]) / (2.0 * self.dx)
                
                # For boundary points, use forward/backward difference
                # Forward difference at the first point
                first_slice = [slice(None)] * tensor.dim()
                first_slice[axis] = 0
                next_slice = [slice(None)] * tensor.dim()
                next_slice[axis] = 1
                result[tuple(first_slice)] = (tensor[tuple(next_slice)] - tensor[tuple(first_slice)]) / self.dx
                
                # Backward difference at the last point
                last_slice = [slice(None)] * tensor.dim()
                last_slice[axis] = -1
                prev_slice = [slice(None)] * tensor.dim()
                prev_slice[axis] = -2
                result[tuple(last_slice)] = (tensor[tuple(last_slice)] - tensor[tuple(prev_slice)]) / self.dx
        else:  # order == 2
            # Simple central difference for second derivative
            if tensor.shape[axis] > 2:
                # Get slices for i+1, i, and i-1
                slice_plus = [slice(None)] * tensor.dim()
                slice_center = [slice(None)] * tensor.dim()
                slice_minus = [slice(None)] * tensor.dim()
                
                # For interior points
                slice_plus[axis] = slice(2, None)
                slice_center[axis] = slice(1, -1)
                slice_minus[axis] = slice(0, -2)
                
                # Central difference: (f(x+h) - 2f(x) + f(x-h)) / h²
                result[tuple(slice_center)] = (
                    tensor[tuple(slice_plus)] 
                    - 2 * tensor[tuple(slice_center)] 
                    + tensor[tuple(slice_minus)]
                ) / (self.dx ** 2)
                
                # For boundary points, use forward/backward approximations
                # First point: Forward difference
                first_slice = [slice(None)] * tensor.dim()
                first_slice[axis] = 0
                first_plus1 = [slice(None)] * tensor.dim()
                first_plus1[axis] = 1
                first_plus2 = [slice(None)] * tensor.dim()
                first_plus2[axis] = 2
                result[tuple(first_slice)] = (
                    2 * tensor[tuple(first_slice)] 
                    - 5 * tensor[tuple(first_plus1)] 
                    + 4 * tensor[tuple(first_plus2)] 
                    - tensor[tuple(self._get_offset_slice(tensor, axis, 3))]
                ) / (self.dx ** 2)
                
                # Last point: Backward difference
                last_slice = [slice(None)] * tensor.dim()
                last_slice[axis] = -1
                last_minus1 = [slice(None)] * tensor.dim()
                last_minus1[axis] = -2
                last_minus2 = [slice(None)] * tensor.dim()
                last_minus2[axis] = -3
                result[tuple(last_slice)] = (
                    2 * tensor[tuple(last_slice)] 
                    - 5 * tensor[tuple(last_minus1)] 
                    + 4 * tensor[tuple(last_minus2)] 
                    - tensor[tuple(self._get_offset_slice(tensor, axis, -4))]
                ) / (self.dx ** 2)
                
        return result

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
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        if metric is None:
            metric = self.metric_field
            
        # Initialize Christoffel symbols
        lattice_size = metric.shape[0]
        dim = self.dimensions
        christoffel = torch.zeros((lattice_size, dim, dim, dim), dtype=metric.dtype, device=metric.device)
        
        # Compute inverse metric
        g_inv = torch.zeros_like(metric)
        for i in range(lattice_size):
            g_inv[i] = torch.linalg.inv(metric[i])
        
        # Compute derivatives of the metric along the lattice dimension
        dg = self._finite_difference(metric, order=1, axis=0)
        
        # Reshape derivatives for use in the Christoffel symbol calculation
        # We'll use a simplified approach where we only consider derivatives along the first dimension
        dg_reshaped = torch.zeros((lattice_size, dim, dim, dim), dtype=metric.dtype, device=metric.device)
        
        # For simplicity, we'll set all spatial derivatives to be the same as the time derivative
        for i in range(lattice_size):
            for alpha in range(dim):
                for mu in range(dim):
                    for nu in range(dim):
                        dg_reshaped[i, alpha, mu, nu] = dg[i, mu, nu]
        
        # Compute Christoffel symbols
        for i in range(lattice_size):
            for mu in range(dim):
                for alpha in range(dim):
                    for beta in range(dim):
                        # Sum over sigma
                        for sigma in range(dim):
                            # Γ^μ_αβ = (1/2) g^μσ (∂_α g_σβ + ∂_β g_σα - ∂_σ g_αβ)
                            christoffel[i, mu, alpha, beta] += 0.5 * g_inv[i, mu, sigma] * (
                                dg_reshaped[i, alpha, sigma, beta] + 
                                dg_reshaped[i, beta, sigma, alpha] - 
                                dg_reshaped[i, sigma, alpha, beta]
                            )
        
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
        
        # Enforce Riemann tensor symmetries and Bianchi identity
        riemann = self.enforce_tensor_symmetries(riemann, "riemann")
        
        # Cache the result
        self._cache[cache_key] = riemann
        return riemann
    
    def compute_ricci_tensor(self, metric: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the Ricci tensor by contracting the Riemann tensor.
        
        Args:
            metric: Input metric tensor. If None, uses self.metric_field
            
        Returns:
            Ricci tensor with shape [lattice_size, d, d]
        """
        cache_key = "ricci"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if metric is None:
            metric = self.metric_field
        
        # Get Riemann tensor
        riemann = self.compute_riemann_tensor(metric)
        
        # Initialize Ricci tensor
        lattice_size = metric.shape[0]
        dim = self.dimensions
        ricci = torch.zeros((lattice_size, dim, dim), dtype=metric.dtype, device=metric.device)
        
        # Compute Ricci tensor by contracting Riemann tensor
        for i in range(lattice_size):
            for mu in range(dim):
                for nu in range(dim):
                    # R_μν = R^λ_μλν
                    for lamb in range(dim):
                        ricci[i, mu, nu] += riemann[i, lamb, mu, lamb, nu]
        
        # Enforce Ricci tensor symmetry
        ricci = self.enforce_tensor_symmetries(ricci, "ricci")
        
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
        # Use explicit loops instead of einsum with Greek letters
        lattice_size = metric.shape[0]
        scalar = torch.zeros(lattice_size, dtype=metric.dtype, device=metric.device)
        
        for i in range(lattice_size):
            # Manual contraction of g_inv and ricci
            scalar[i] = torch.sum(g_inv[i] * ricci[i])
        
        # Cache the result
        self._cache[cache_key] = scalar
        return scalar
    
    def compute_higher_curvature_terms(
        self, 
        metric: torch.Tensor,
        ricci: torch.Tensor,
        scalar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute higher curvature terms (e.g., Gauss-Bonnet).
        
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
        
        # Add higher curvature terms if enabled
        if self.higher_curvature_terms:
            higher_curvature = self.compute_higher_curvature_terms(metric, ricci, ricci_scalar.squeeze())
            einstein = einstein + higher_curvature
        
        # Cache the result
        self._cache[cache_key] = einstein
        return einstein
    
    # ------------------------------------------------------------------
    # Physical constraint enforcement
    # ------------------------------------------------------------------
    def enforce_tensor_symmetries(self, tensor: torch.Tensor, tensor_type: str) -> torch.Tensor:
        """Enforce the symmetry properties of various tensors.
        
        Args:
            tensor: Input tensor to enforce symmetries on
            tensor_type: Type of tensor ('riemann', 'weyl', etc.)
            
        Returns:
            Tensor with enforced symmetries
        """
        if tensor_type == "riemann" or tensor_type == "weyl":
            # Enforce antisymmetry in first two indices: R_{abcd} = -R_{bacd}
            tensor = 0.5 * (tensor - tensor.transpose(1, 2))
            
            # Enforce antisymmetry in last two indices: R_{abcd} = -R_{abdc}
            tensor = 0.5 * (tensor - tensor.transpose(3, 4))
            
            # Enforce pair symmetry: R_{abcd} = R_{cdab}
            tensor = 0.5 * (tensor + tensor.permute(0, 3, 4, 1, 2))
            
            if tensor_type == "riemann":
                # Enforce Bianchi identity: R_{abcd} + R_{acdb} + R_{adbc} = 0
                tensor = self.enforce_bianchi_identity(tensor)
                
        elif tensor_type == "ricci":
            # Enforce symmetry: R_{ab} = R_{ba}
            tensor = 0.5 * (tensor + tensor.transpose(1, 2))
            
        return tensor
        
    def enforce_bianchi_identity(self, riemann_tensor: torch.Tensor) -> torch.Tensor:
        """Enforce the Bianchi identity on the Riemann tensor.
        
        The first Bianchi identity states:
        R_{abcd} + R_{acdb} + R_{adbc} = 0
        
        Args:
            riemann_tensor: Input Riemann tensor
            
        Returns:
            Riemann tensor with enforced Bianchi identity
        """
        d = self.dimensions
        corrected_tensor = riemann_tensor.clone()
        
        # For each set of indices, enforce the cyclic identity
        for a in range(d):
            for b in range(d):
                for c in range(d):
                    for d_idx in range(d):
                        # Compute the cyclic sum
                        cyclic_sum = (
                            riemann_tensor[:, a, b, c, d_idx] + 
                            riemann_tensor[:, a, c, d_idx, b] + 
                            riemann_tensor[:, a, d_idx, b, c]
                        )
                        
                        # Distribute the correction equally among the three terms
                        correction = cyclic_sum / 3.0
                        
                        corrected_tensor[:, a, b, c, d_idx] -= correction
                        corrected_tensor[:, a, c, d_idx, b] -= correction
                        corrected_tensor[:, a, d_idx, b, c] -= correction
        
        return corrected_tensor
    
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
        
        # Enforce Weyl tensor symmetries
        weyl = self.enforce_tensor_symmetries(weyl, "weyl")
        
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