"""Entropy measurements for quantum subsystems with support for edge modes and non-conformal fields."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .quantum_engine import QuantumEngine


class EntropyModule:
    def __init__(
        self, 
        quantum_engine: QuantumEngine, 
        epsilon: float = 1e-12,
        uv_cutoff: float = 1e-6,
        include_edge_modes: bool = True,
        conformal_invariance: bool = False,
        regularization_scheme: str = "lattice"
    ) -> None:
        """Initialize the entropy module with advanced features.
        
        Args:
            quantum_engine: The quantum engine for state preparation and manipulation
            epsilon: Small value for numerical stability in entropy calculations
            uv_cutoff: Ultraviolet cutoff parameter for entropy regularization
            include_edge_modes: Whether to include edge modes in entropy calculations
            conformal_invariance: Whether to assume conformal invariance in the field theory
            regularization_scheme: Regularization scheme for entropy calculations
                Options: "lattice", "dimensional", "entanglement", "holographic"
        """
        self.quantum_engine = quantum_engine
        self.epsilon = epsilon
        self.uv_cutoff = uv_cutoff
        self.include_edge_modes = include_edge_modes
        self.conformal_invariance = conformal_invariance
        self.regularization_scheme = regularization_scheme
        
        # Edge mode contribution parameters
        self.edge_mode_dimension = 1  # Default dimension of edge Hilbert space
        self.edge_mode_entropy_factor = 0.5  # Contribution factor of edge modes
        
        # For tracking entropy contributions
        self.last_entropy_components = {
            "bulk": 0.0,
            "edge_modes": 0.0,
            "uv_correction": 0.0,
            "total": 0.0
        }

    def compute_density_matrix(self, state: torch.Tensor) -> torch.Tensor:
        """Compute density matrix from pure state vector."""
        state = state.reshape(-1, 1)
        return state @ state.conj().t()

    def partial_trace(
        self, 
        state: torch.Tensor, 
        keep_qubits: Sequence[int],
        include_edge: bool = None
    ) -> torch.Tensor:
        """Compute reduced density matrix with optional edge mode handling.
        
        Args:
            state: Pure quantum state vector
            keep_qubits: Indices of qubits to keep (trace out the rest)
            include_edge: Whether to include edge modes (defaults to self.include_edge_modes)
            
        Returns:
            Reduced density matrix
        """
        include_edge = self.include_edge_modes if include_edge is None else include_edge
        
        # Standard partial trace from quantum engine
        rho_A = self.quantum_engine.reduced_density_matrix(state, keep_qubits)
        
        if include_edge and not self.conformal_invariance:
            # Add edge mode contribution following Donnelly-Wall prescription
            # This models gauge degrees of freedom that become physical at the boundary
            edge_dim = self.edge_mode_dimension
            if edge_dim > 1:
                # Tensor product with maximally mixed edge mode state
                edge_state = torch.eye(edge_dim, dtype=rho_A.dtype, device=rho_A.device) / edge_dim
                rho_A_with_edge = torch.kron(rho_A, edge_state)
                return rho_A_with_edge
        
        return rho_A

    def von_neumann_entropy(
        self, 
        rho: torch.Tensor, 
        apply_uv_cutoff: bool = True
    ) -> torch.Tensor:
        """Compute von Neumann entropy with UV regularization.
        
        Args:
            rho: Density matrix
            apply_uv_cutoff: Whether to apply UV cutoff regularization
            
        Returns:
            Entropy value as tensor
        """
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvalsh(rho)
        
        # Apply UV regularization if requested
        if apply_uv_cutoff and self.regularization_scheme == "lattice":
            # Lattice cutoff: truncate eigenvalues below cutoff
            eigenvals = eigenvals.clamp(min=self.uv_cutoff)
        elif apply_uv_cutoff and self.regularization_scheme == "dimensional":
            # Dimensional regularization: smooth modification of small eigenvalues
            eigenvals = eigenvals + self.uv_cutoff * torch.exp(-eigenvals/self.uv_cutoff)
            
        # Ensure numerical stability
        eigenvals = eigenvals.clamp(min=self.epsilon)
        
        # Compute entropy
        entropy = -torch.sum(eigenvals * torch.log(eigenvals))
        
        return entropy

    def compute_entanglement_entropy(
        self, 
        state: torch.Tensor, 
        partition: Sequence[int],
        include_edge: bool = None,
        apply_uv_cutoff: bool = True
    ) -> torch.Tensor:
        """Compute entanglement entropy with edge mode and UV regularization support.
        
        Args:
            state: Pure quantum state vector
            partition: Indices of qubits to keep (trace out the rest)
            include_edge: Whether to include edge modes
            apply_uv_cutoff: Whether to apply UV cutoff regularization
            
        Returns:
            Entanglement entropy value
        """
        include_edge = self.include_edge_modes if include_edge is None else include_edge
        
        # Compute reduced density matrix (without edge modes first)
        rho_A = self.partial_trace(state, partition, include_edge=False)
        
        # Compute bulk entropy contribution
        bulk_entropy = self.von_neumann_entropy(rho_A, apply_uv_cutoff)
        
        # Initialize component tracking
        self.last_entropy_components = {
            "bulk": bulk_entropy.item(),
            "edge_modes": 0.0,
            "uv_correction": 0.0,
            "total": bulk_entropy.item()
        }
        
        # Add edge mode contribution if requested
        if include_edge and not self.conformal_invariance:
            # Compute edge mode entropy contribution
            # For gauge fields and gravitons, edge modes contribute additional entropy
            boundary_size = len(partition)
            edge_entropy = self.edge_mode_entropy_factor * boundary_size * torch.log(
                torch.tensor(self.edge_mode_dimension, dtype=bulk_entropy.dtype, device=bulk_entropy.device)
            )
            
            self.last_entropy_components["edge_modes"] = edge_entropy.item()
            self.last_entropy_components["total"] += edge_entropy.item()
            
            return bulk_entropy + edge_entropy
        
        # For conformal fields, no edge mode contribution
        return bulk_entropy

    def entropy_gradient(
        self, 
        state: torch.Tensor, 
        partition: Sequence[int],
        include_edge: bool = None,
        apply_uv_cutoff: bool = True
    ) -> torch.Tensor:
        """Compute gradient of entanglement entropy with respect to state parameters.
        
        Args:
            state: Pure quantum state vector
            partition: Indices of qubits to keep (trace out the rest)
            include_edge: Whether to include edge modes
            apply_uv_cutoff: Whether to apply UV cutoff regularization
            
        Returns:
            Gradient of entropy with respect to state parameters
        """
        # Ensure state requires gradient
        if not state.requires_grad:
            state = state.detach().clone().requires_grad_(True)

        # Compute entropy with tracking enabled
        entropy = self.compute_entanglement_entropy(
            state, partition, include_edge, apply_uv_cutoff
        )

        # Compute gradient (create_graph=True so callers can compute second derivatives)
        grad = torch.autograd.grad(entropy, state, create_graph=True, retain_graph=True)[0]

        # Store state reference so Faulkner formulation can compute the Hessian
        self._last_state = state

        return grad

    def entropy_flow(
        self, 
        states: torch.Tensor, 
        partition: Sequence[int], 
        times: torch.Tensor,
        include_edge: bool = None,
        apply_uv_cutoff: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute entropy and its time derivative for a batch of states.
        
        Args:
            states: Batch of quantum state vectors
            partition: Indices of qubits to keep (trace out the rest)
            times: Time points corresponding to each state
            include_edge: Whether to include edge modes
            apply_uv_cutoff: Whether to apply UV cutoff regularization
            
        Returns:
            Tuple of (entropies, entropy_derivatives)
        """
        # Compute entropies for all states
        entropies = torch.stack(
            [self.compute_entanglement_entropy(
                state, partition, include_edge, apply_uv_cutoff
            ) for state in states]
        )
        
        # Compute time derivative using central finite differences.
        # torch.gradient() has inconsistent API across PyTorch versions;
        # manual implementation is more portable.
        dt = times[1:] - times[:-1]  # (N-1,) interval widths
        # Interior points: central difference
        dS_dt = torch.zeros_like(entropies)
        dS_dt[1:-1] = (entropies[2:] - entropies[:-2]) / (times[2:] - times[:-2])
        # Boundary points: one-sided difference
        dS_dt[0] = (entropies[1] - entropies[0]) / dt[0].clamp(min=1e-12)
        dS_dt[-1] = (entropies[-1] - entropies[-2]) / dt[-1].clamp(min=1e-12)
        
        return entropies, dS_dt

    def area_law_coefficient(
        self, 
        state: torch.Tensor, 
        partitions: List[List[int]]
    ) -> float:
        """Estimate area law coefficient from multiple partitions.
        
        In holographic theories, entanglement entropy follows an area law:
        S ~ α × Area, where α is the coefficient we're estimating.
        
        Args:
            state: Pure quantum state vector
            partitions: List of different partitions to use for estimation
            
        Returns:
            Estimated area law coefficient
        """
        areas = []
        entropies = []
        
        for partition in partitions:
            # Estimate "area" as boundary size
            area = len(partition)
            areas.append(area)
            
            # Compute entropy
            entropy = self.compute_entanglement_entropy(state, partition)
            entropies.append(entropy.item())
        
        # Convert to tensors for linear regression
        areas_tensor = torch.tensor(areas, dtype=torch.float)
        entropies_tensor = torch.tensor(entropies, dtype=torch.float)
        
        # Simple linear regression to find coefficient
        mean_area = torch.mean(areas_tensor)
        mean_entropy = torch.mean(entropies_tensor)
        
        numerator = torch.sum((areas_tensor - mean_area) * (entropies_tensor - mean_entropy))
        denominator = torch.sum((areas_tensor - mean_area) ** 2)
        
        coefficient = numerator / denominator
        
        return coefficient.item()
        
    def holographic_entropy(
        self,
        state: torch.Tensor,
        partition: Sequence[int],
        metric_tensor: torch.Tensor,
        G_N: float = 1.0,
        hbar: float = 1.0,
    ) -> torch.Tensor:
        """Compute holographic entanglement entropy via the Ryu-Takayanagi formula.

        S_A = L(γ_A) / (4 G_N ℏ)

        where L(γ_A) is the length of the minimal geodesic in the bulk connecting
        the boundary endpoints of region A (Ryu-Takayanagi surface in 1+1D).

        Implementation
        --------------
        For a 1+1D bulk with metric g_μν = diag(g_tt, g_rr), on a constant-time
        slice the geodesic length between two radial positions r_a and r_b is:

            L = ∫_{r_a}^{r_b} √g_rr(r) dr

        The qubit partition is mapped to a spatial interval by assuming that
        qubits are uniformly distributed over the lattice.

        Args:
            state: Pure quantum state vector (used to infer partition size).
            partition: Qubit indices defining boundary region A.
            metric_tensor: Either
                - (lattice_size, dim, dim) full metric field, or
                - (dim, dim) metric at a single lattice point (used as constant).
            G_N: Newton's constant in natural units (default 1).
            hbar: Reduced Planck constant in natural units (default 1).

        Returns:
            Holographic entropy estimate S_RT as a scalar tensor.
        """
        if metric_tensor.dim() == 2:
            # Single-point metric: treat as constant bulk geometry
            g_rr = metric_tensor[1, 1].abs()
            # Geodesic length ≈ √g_rr * (fraction of total system covered by partition)
            frac = len(partition) / self.quantum_engine.num_qubits
            geodesic_length = torch.sqrt(g_rr) * frac
            return geodesic_length / (4.0 * G_N * hbar)

        # Full lattice metric: shape (lattice_size, dim, dim)
        lattice_size = metric_tensor.shape[0]
        n_qubits = self.quantum_engine.num_qubits

        # Map partition to lattice interval [i_a, i_b]
        if len(partition) == 0:
            return torch.tensor(0.0, dtype=metric_tensor.dtype, device=metric_tensor.device)

        # Qubit i spans lattice positions [i*L/n, (i+1)*L/n)
        # Boundary region A = contiguous block from min(partition) to max(partition)
        q_min = min(partition)
        q_max = max(partition)
        i_a = int(q_min * lattice_size / n_qubits)
        i_b = int((q_max + 1) * lattice_size / n_qubits)
        i_a = max(0, min(i_a, lattice_size - 1))
        i_b = max(i_a + 1, min(i_b, lattice_size))

        # Spatial step size (assumed uniform)
        dr = 1.0 / lattice_size  # normalized units; scale by actual dx if known

        # Geodesic length L = Σ √g_rr(r_i) * dr  for r_i in [r_a, r_b]
        g_rr_values = metric_tensor[i_a:i_b, 1, 1].abs()
        geodesic_length = torch.sum(torch.sqrt(g_rr_values)) * dr

        # RT formula: S = L / (4 G_N ℏ)
        return geodesic_length / (4.0 * G_N * hbar)
        
    def get_entropy_components(self) -> Dict[str, float]:
        """Get the components of the last entropy calculation.
        
        Returns:
            Dictionary with entropy components: bulk, edge_modes, uv_correction, total
        """
        return self.last_entropy_components
        
    def set_edge_mode_parameters(
        self, 
        dimension: int = 1, 
        entropy_factor: float = 0.5
    ) -> None:
        """Configure edge mode parameters.
        
        Args:
            dimension: Dimension of edge mode Hilbert space
            entropy_factor: Contribution factor of edge modes to entropy
        """
        self.edge_mode_dimension = max(1, dimension)
        self.edge_mode_entropy_factor = max(0.0, entropy_factor)