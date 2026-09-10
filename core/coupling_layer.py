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

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import torch

from .entropy_module import EntropyModule
from .geometry_engine import GeometryEngine
from .utils.finite_difference import fixed_finite_difference
from .validation import (EntropyField, GateConfig, ProvenanceError,
                         ValidationReport,
                         check_entropy_field_sanity, check_entropy_provenance,
                         check_finite, check_trace_identity,
                         check_tracelessness)


class StressTensorFormulation(str, Enum):
    """Different formulations of the entropic stress-energy tensor."""
    JACOBSON = "jacobson"    # Original Jacobson thermodynamic formulation
    CANONICAL = "canonical"  # Simple outer product of gradients
    FAULKNER = "faulkner"    # Faulkner's linearized Einstein formulation
    MODIFIED = "modified"    # Modified formulation with edge mode corrections
    LAGRANGIAN = "lagrangian"  # Hilbert-variation derived from explicit action
    MASSLESS = "massless"      # Traceless form enforcing E=pc massless constraint


@dataclass
class CouplingTerms:
    """Container for all coupling-related tensors and quantities."""
    entropy_gradient: torch.Tensor
    stress_tensor: torch.Tensor
    einstein_tensor: torch.Tensor
    coupling_residual: torch.Tensor
    edge_mode_contribution: Optional[torch.Tensor] = None
    higher_curvature_terms: Optional[torch.Tensor] = None
    tracelessness_violation: Optional[torch.Tensor] = None  # g^μν T_μν, should be ~0 for massless

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
        include_edge_modes: bool = False,
        allow_legacy_heuristic: bool = False,
        covariant_hessian: bool = True,
        include_higher_curvature: bool = False,
        conformal_invariance: bool = False,
        hbar_factor: float = 1.0 / (2.0 * math.pi),  # ℏ/(2π) in natural units
    ) -> None:
        """Initialize the coupling layer.
        
        Args:
            geometry_engine: The geometry engine for metric and curvature calculations
            entropy_module: The entropy module for entanglement calculations
            coupling_strength: Overall coupling strength (analogous to 8πG)
            stress_form: Which formulation of stress-energy tensor to use
            include_edge_modes: Whether to add the toy edge-mode term. This is
                a hardcoded Λ-like correction (0.01 · ℏ/2π · g_μν), NOT a
                derived edge-mode stress tensor — off by default.
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
        # Opt-in for the legacy v1.2 state-space-gradient stack
        # (compute_entropy_stress_tensor / compute_coupling_terms). Off by
        # default: those paths relabel state-parameter derivatives as
        # spacetime components and must never produce a reported number.
        # Setting it here, once, keeps the acknowledgement visible at the
        # construction site instead of scattered through call sites.
        self.allow_legacy_heuristic = allow_legacy_heuristic
        # FAULKNER's Hessian. True (default) uses the covariant
        # ∇_μ∇_νS = ∂_μ∂_νS − Γ^λ_{μν}∂_λS, which is what the formulation
        # actually names. False restores the pre-v1.4 coordinate second
        # derivative, kept only so the two can be compared.
        self.covariant_hessian = covariant_hessian

    # ------------------------------------------------------------------
    # Stress-energy tensors induced by entropy gradients
    # ------------------------------------------------------------------
    def compute_entropy_stress_tensor(
        self,
        entropy_gradient: torch.Tensor,
        metric: Optional[torch.Tensor] = None,
        allow_legacy_heuristic: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """LEGACY DEMO PATH — this is the v1.2 defect, kept only for comparison.

        `entropy_gradient` here is a gradient with respect to quantum **state
        parameters**, whose components are then used as if they were spacetime
        indices.  That relabelling is precisely what invalidated the v1.2
        results.  It is not physics and must never produce a reported number.

        Raises `ProvenanceError` unless `allow_legacy_heuristic=True` is passed
        explicitly at the call site, so using it is visible in the diff.
        Use `compute_stress_tensor_field()` for the real pipeline.

        
        Args:
            entropy_gradient: Gradient of entropy with respect to state parameters
            metric: Optional metric tensor (uses geometry engine's metric if None)
            
        Returns:
            Tuple of (stress_tensor, edge_mode_contribution)

        Warning:
            This method interprets a gradient taken with respect to *quantum
            state amplitudes* as if its first `dim` components were spacetime
            derivatives ∇_μ S.  That identification is a demo heuristic with
            no physical justification.  For a physically meaningful stress
            tensor built from an entropy field S(x) on the lattice, use
            compute_stress_tensor_field() instead.
        """
        if allow_legacy_heuristic is None:
            allow_legacy_heuristic = self.allow_legacy_heuristic
        if not allow_legacy_heuristic:
            raise ProvenanceError(
                "compute_entropy_stress_tensor() is the legacy state-space-"
                "gradient heuristic: it relabels derivatives with respect to "
                "state parameters as spacetime components, which is the v1.2 "
                "defect. Use compute_stress_tensor_field() with an EntropyField, "
                "or pass allow_legacy_heuristic=True to run it knowingly."
            )

        if not getattr(self, "_projection_warned", False):
            warnings.warn(
                "compute_entropy_stress_tensor() projects a state-space "
                "gradient onto spacetime indices — a demo heuristic, not "
                "physics. Prefer compute_stress_tensor_field() with a "
                "spatial entropy profile.",
                stacklevel=2,
            )
            self._projection_warned = True

        if metric is None:
            metric = self.geometry.metric

        # Inverse metric for covariant contraction (∇S)² = g^μν ∂_μS ∂_νS.
        # An earlier version used the Euclidean dot product, which breaks
        # the trace identities every formulation below relies on.
        try:
            metric_inv = torch.linalg.inv(metric)
        except Exception:
            metric_inv = torch.linalg.pinv(metric)

        # State-space gradients are complex (d S / d amplitude); only the
        # real part is used in the heuristic spacetime projection below.
        if entropy_gradient.is_complex():
            entropy_gradient = entropy_gradient.real
        entropy_gradient = entropy_gradient.to(dtype=metric.dtype)

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
                entropy_grad_spacetime = torch.zeros(dim, dtype=entropy_gradient.dtype, device=entropy_gradient.device)
                # Use the first dim components or pad with zeros
                entropy_grad_spacetime[:min(dim, entropy_gradient.shape[0])] = entropy_gradient[:min(dim, entropy_gradient.shape[0])]
            else:
                entropy_grad_spacetime = entropy_gradient
                
            contraction = entropy_grad_spacetime @ metric_inv @ entropy_grad_spacetime
            T = torch.outer(entropy_grad_spacetime, entropy_grad_spacetime)
            T = self.hbar_factor * (T - 0.5 * metric * contraction)
            
        elif self.stress_form == StressTensorFormulation.CANONICAL:
            # Simple outer product:
            # T_μν = (ℏ/2π)∇_μS ∇_νS
            
            # Ensure entropy gradient has the right dimensions for spacetime
            dim = self.geometry.dimensions
            if entropy_gradient.shape[0] != dim:
                entropy_grad_spacetime = torch.zeros(dim, dtype=entropy_gradient.dtype, device=entropy_gradient.device)
                entropy_grad_spacetime[:min(dim, entropy_gradient.shape[0])] = entropy_gradient[:min(dim, entropy_gradient.shape[0])]
            else:
                entropy_grad_spacetime = entropy_gradient
                
            T = self.hbar_factor * torch.outer(entropy_grad_spacetime, entropy_grad_spacetime)
            
        elif self.stress_form == StressTensorFormulation.FAULKNER:
            # Faulkner's linearized Einstein formulation (arXiv:1312.7856):
            #   T_μν = (ℏ/2π)[ ∇_μ∇_νS − (□S) g_μν ]
            # where □S = g^μν ∇_μ∇_νS is the d'Alembertian of entropy.
            #
            # Implementation: compute the Hessian of S w.r.t. the quantum state
            # projected to spacetime dimensions.  entropy_gradient() stores the
            # state it differentiated against as entropy._last_state with
            # create_graph=True, enabling second-order autograd here.

            dim = self.geometry.dimensions
            if entropy_gradient.shape[0] != dim:
                entropy_grad_spacetime = torch.zeros(dim, dtype=entropy_gradient.dtype, device=entropy_gradient.device)
                entropy_grad_spacetime[:min(dim, entropy_gradient.shape[0])] = (
                    entropy_gradient[:min(dim, entropy_gradient.shape[0])]
                )
            else:
                entropy_grad_spacetime = entropy_gradient

            # Attempt to compute the spacetime Hessian H_μν = ∂²S/∂x_μ∂x_ν
            state_ref = getattr(self.entropy, "_last_state", None)
            hessian = torch.zeros(dim, dim, dtype=entropy_grad_spacetime.dtype, device=entropy_grad_spacetime.device)

            if state_ref is not None and entropy_grad_spacetime.requires_grad:
                # Second-order autograd: differentiate each component of the
                # projected gradient to obtain the Hessian rows.
                for mu in range(dim):
                    if entropy_grad_spacetime[mu].grad_fn is not None:
                        row_full = torch.autograd.grad(
                            entropy_grad_spacetime[mu],
                            state_ref,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if row_full is not None:
                            row_real = row_full.real
                            hessian[mu, :] = row_real[:dim]
            else:
                # Fallback: symmetric outer product (= Jacobson Hessian approximation)
                hessian = torch.outer(entropy_grad_spacetime, entropy_grad_spacetime)

            # □S = Tr(H) in flat-metric approximation (proper: g^μν H_μν)
            try:
                metric_inv = torch.linalg.inv(metric)
            except Exception:
                metric_inv = torch.linalg.pinv(metric)
            box_S = torch.sum(metric_inv * hessian)

            T = self.hbar_factor * (hessian - box_S * metric)
            
        elif self.stress_form == StressTensorFormulation.MODIFIED:
            # Modified formulation with corrections for non-conformal fields:
            # T_μν = (ℏ/2π)[∇_μS ∇_νS - (1/2)g_μν (∇S)² + α R_μν]
            # where α is a non-conformality parameter

            # Ensure entropy gradient has the right dimensions for spacetime
            dim = self.geometry.dimensions
            if entropy_gradient.shape[0] != dim:
                entropy_grad_spacetime = torch.zeros(dim, dtype=entropy_gradient.dtype, device=entropy_gradient.device)
                entropy_grad_spacetime[:min(dim, entropy_gradient.shape[0])] = entropy_gradient[:min(dim, entropy_gradient.shape[0])]
            else:
                entropy_grad_spacetime = entropy_gradient

            contraction = entropy_grad_spacetime @ metric_inv @ entropy_grad_spacetime
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

        elif self.stress_form == StressTensorFormulation.LAGRANGIAN:
            # Derived from the Hilbert action via variational principle.
            #
            # Action:
            #   S = ∫d^n x √(-g) [ R/(16πG) - (ℏ/4π)(∇_μS)(∇^μS) ]
            #
            # Varying the matter part L_m = -(ℏ/4π) g^αβ ∂_αS ∂_βS with
            # respect to g^μν using T_μν = -(2/√(-g)) δ(√(-g) L_m)/δg^μν:
            #
            #   δ(√(-g) L_m)/δg^μν
            #     = (-½ √(-g) g_μν) L_m + √(-g)(-(ℏ/4π) ∂_μS ∂_νS)
            #
            #   T_μν = -(2)[(-½ g_μν)(-(ℏ/4π)(∇S)²) - (ℏ/4π) ∂_μS ∂_νS]
            #         = (ℏ/2π) ∂_μS ∂_νS - ½ g_μν (ℏ/2π)(∇S)²
            #
            # This is mathematically identical to JACOBSON but is *derived*,
            # not postulated.  Trace in n dimensions:
            #   g^μν T_μν = (ℏ/2π)(∇S)²(1 - n/2)
            # → zero only in n=2; non-zero in n=4 (tracelessness violation).

            dim = self.geometry.dimensions
            if entropy_gradient.shape[0] != dim:
                entropy_grad_spacetime = torch.zeros(dim, dtype=entropy_gradient.dtype, device=entropy_gradient.device)
                entropy_grad_spacetime[:min(dim, entropy_gradient.shape[0])] = entropy_gradient[:min(dim, entropy_gradient.shape[0])]
            else:
                entropy_grad_spacetime = entropy_gradient

            contraction = entropy_grad_spacetime @ metric_inv @ entropy_grad_spacetime
            T = self.hbar_factor * (
                torch.outer(entropy_grad_spacetime, entropy_grad_spacetime)
                - 0.5 * metric * contraction
            )

        elif self.stress_form == StressTensorFormulation.MASSLESS:
            # Traceless stress tensor — enforces the E=pc massless constraint.
            #
            # Motivation: entropy gradients propagate at c (pure information,
            # no rest mass), so they must satisfy the same tracelessness
            # condition as the electromagnetic stress tensor:
            #   g^μν T_μν = 0
            #
            # The JACOBSON/LAGRANGIAN form has trace (ℏ/2π)(∇S)²(1 - n/2),
            # which vanishes only in n=2.  Replacing the (1/2) prefactor with
            # (1/n) makes the tensor traceless in any dimension:
            #
            #   T_μν = (ℏ/2π)[ ∂_μS ∂_νS - (1/n) g_μν (∇S)² ]
            #
            # Trace check:
            #   g^μν T_μν = (ℏ/2π)[ (∇S)² - (1/n)·n·(∇S)² ] = 0  ✓
            #
            # This is the leading-order conformal (traceless) completion of
            # the Lagrangian-derived form.

            dim = self.geometry.dimensions
            if entropy_gradient.shape[0] != dim:
                entropy_grad_spacetime = torch.zeros(dim, dtype=entropy_gradient.dtype, device=entropy_gradient.device)
                entropy_grad_spacetime[:min(dim, entropy_gradient.shape[0])] = entropy_gradient[:min(dim, entropy_gradient.shape[0])]
            else:
                entropy_grad_spacetime = entropy_gradient

            contraction = entropy_grad_spacetime @ metric_inv @ entropy_grad_spacetime
            T = self.hbar_factor * (
                torch.outer(entropy_grad_spacetime, entropy_grad_spacetime)
                - (1.0 / dim) * metric * contraction
            )

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

    def compute_stress_tensor_field(
        self,
        entropy_field: Union[torch.Tensor, "EntropyField"],
        metric_field: Optional[torch.Tensor] = None,
        formulation: Optional[Union[str, StressTensorFormulation]] = None,
        gates: Optional[GateConfig] = None,
        report: Optional[ValidationReport] = None,
    ) -> torch.Tensor:
        """Compute T_μν(x) over the whole lattice from an entropy field S(x).

        This is the physically meaningful path.  S(x) must be a scalar field
        on the lattice — e.g. the entanglement entropy of the qubits inside
        radius x, computed from an actual quantum state — and ∇_μ S is its
        honest spacetime derivative: the field is static, so ∂_t S = 0, and
        ∂_1 S is the dx-normalized lattice finite difference.  Nothing about
        the spatial structure of the source is inserted by hand, and no
        state-space gradient components are relabeled as spacetime indices.

        All contractions use the inverse metric, so the MASSLESS form is
        traceless exactly and identically: g^μν T_μν = 0 by construction.

        The entropy field is gated before use: it must be an `EntropyField`
        whose provenance records derivation from partial traces of a real
        quantum state.  A bare tensor is refused, because nothing about a
        tensor records whether its spatial structure was computed or inserted
        by hand — and an inserted profile is precisely what invalidated the
        v1.2 results.  Pass `gates=GateConfig(require_derived_entropy=False)`
        to bypass this deliberately.

        Args:
            entropy_field: S(x) as an `EntropyField` (shape (lattice_size,)).
                A raw tensor is accepted only when the provenance gate is
                explicitly disabled.
            metric_field: g_μν(x), shape (lattice_size, dim, dim); defaults to
                the geometry engine's current metric field (autograd flows
                through it).
            formulation: Stress tensor formulation; defaults to self.stress_form.
            gates: Gate configuration; defaults to strict `GateConfig()`.
            report: Optional `ValidationReport` collecting gate results.

        Returns:
            T_μν(x) with shape (lattice_size, dim, dim).

        Raises:
            ProvenanceError: the entropy field is not derived from a state.
            ValidationError: the field is degenerate, or the resulting tensor
                is non-finite or fails tracelessness for a traceless form.
        """
        cfg = GateConfig() if gates is None else gates
        check_entropy_provenance(entropy_field, cfg, report)
        check_entropy_field_sanity(entropy_field, cfg, report)
        if isinstance(entropy_field, EntropyField):
            entropy_field = entropy_field.values

        g = self.geometry.metric_field if metric_field is None else metric_field
        dim = self.geometry.dimensions
        n_points = g.shape[0]
        dx = float(self.geometry.dx)

        if entropy_field.shape[0] != n_points:
            raise ValueError(
                f"entropy_field has {entropy_field.shape[0]} points but the "
                f"lattice has {n_points}"
            )

        form = self.stress_form if formulation is None else (
            StressTensorFormulation(formulation.lower())
            if isinstance(formulation, str) else formulation
        )

        s_field = entropy_field.to(dtype=g.dtype, device=g.device)

        # ∇_μ S: static scalar field varying along the lattice coordinate x¹
        dS = fixed_finite_difference(s_field, order=1, axis=0, dx=dx)  # (N,)
        grads = torch.zeros((n_points, dim), dtype=g.dtype, device=g.device)
        grads[:, 1] = dS

        g_inv = torch.linalg.inv(g)                                    # (N, d, d)
        outer = torch.einsum("ni,nj->nij", grads, grads)               # (N, d, d)
        # (∇S)² = g^μν ∂_μS ∂_νS — covariant contraction
        contraction = torch.einsum("nab,na,nb->n", g_inv, grads, grads)

        if form == StressTensorFormulation.MASSLESS:
            # T_μν = (ℏ/2π)[∂_μS ∂_νS − (1/n) g_μν (∇S)²] — traceless exactly
            T = self.hbar_factor * (
                outer - (1.0 / dim) * g * contraction.view(n_points, 1, 1)
            )
        elif form in (StressTensorFormulation.LAGRANGIAN,
                      StressTensorFormulation.JACOBSON):
            # T_μν = (ℏ/2π)[∂_μS ∂_νS − (1/2) g_μν (∇S)²]
            T = self.hbar_factor * (
                outer - 0.5 * g * contraction.view(n_points, 1, 1)
            )
        elif form == StressTensorFormulation.CANONICAL:
            T = self.hbar_factor * outer
        elif form == StressTensorFormulation.FAULKNER:
            # T_μν = (ℏ/2π)[∇_μ∇_νS − (□S) g_μν] with the *covariant* Hessian
            #
            #     ∇_μ∇_νS = ∂_μ∂_νS − Γ^λ_{μν} ∂_λS
            #
            # The Christoffel term is not optional bookkeeping.  Dropping it
            # (as this code did before v1.4) leaves the coordinate second
            # derivative, whose only nonzero entry is ∂₁∂₁S — so ∇₀∇₀S reads
            # as zero even where Γ¹₀₀ ∂₁S is not, and the tensor is not the
            # one the docstring names.  It only coincides with the covariant
            # Hessian on a flat metric, which is exactly the configuration
            # the optimizer moves away from.
            d2S = fixed_finite_difference(s_field, order=2, axis=0, dx=dx)
            coord_hessian = torch.zeros((n_points, dim, dim), dtype=g.dtype,
                                        device=g.device)
            coord_hessian[:, 1, 1] = d2S
            if self.covariant_hessian:
                # gamma[n, a, b, c] = Γ^a_{bc}
                gamma = self.geometry.compute_christoffel_symbols(g)
                connection = torch.einsum("nlmv,nl->nmv", gamma, grads)
                hessian = coord_hessian - connection
            else:
                hessian = coord_hessian
            box_S = torch.einsum("nab,nab->n", g_inv, hessian)
            T = self.hbar_factor * (hessian - box_S.view(n_points, 1, 1) * g)
            # FAULKNER is NOT traceless: g^uv[∇_u∇_vS − (□S)g_uv] = (1−n)□S.
            # Verified against that prediction rather than against zero.
            faulkner_trace = (1 - dim) * box_S * self.hbar_factor
        elif form == StressTensorFormulation.MODIFIED:
            ricci = self.geometry.compute_ricci_tensor(g)
            alpha = 0.1  # non-conformality parameter
            T = self.hbar_factor * (
                outer - 0.5 * g * contraction.view(n_points, 1, 1) + alpha * ricci
            )
        else:
            raise ValueError(f"Unknown stress tensor formulation: {form}")

        T = self.coupling_strength * T

        # Output gates: a wrong contraction shows up as a non-zero trace in a
        # formulation that is traceless by construction, and a diverged run
        # shows up as NaN. Both must stop the pipeline, not decorate it.
        check_finite("stress_tensor", T, cfg, report)
        if form == StressTensorFormulation.FAULKNER:
            check_trace_identity(T, g, faulkner_trace * self.coupling_strength,
                                 cfg, report, label="faulkner ((1-n)□S)")
        else:
            check_tracelessness(T, g, form.value, cfg, report)
        return T

    def compute_tracelessness_violation(
        self,
        stress_tensor: torch.Tensor,
        metric: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute g^μν T_μν — the trace of the stress tensor.

        For a massless field (E=pc) this must be zero.  Non-zero values
        indicate the stress tensor is sourcing a massive (non-conformal)
        field, which violates the information-propagates-at-c assumption.

        Args:
            stress_tensor: T_μν as a (dim × dim) tensor.
            metric: g_μν (uses geometry engine's metric if None).

        Returns:
            Scalar tensor holding g^μν T_μν.
        """
        if metric is None:
            metric = self.geometry.metric

        # g^μν is the matrix inverse of g_μν.
        # For a diagonal metric this is just 1/diag, but we invert
        # generally so the check works for any metric state.
        try:
            metric_inv = torch.linalg.inv(metric)
        except Exception:
            # Fallback: use pseudo-inverse if metric is singular
            metric_inv = torch.linalg.pinv(metric)

        # Trace = g^μν T_μν = sum_μ sum_ν metric_inv[μ,ν] * T[μ,ν]
        trace = torch.sum(metric_inv * stress_tensor)
        return trace

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
        partition: list,
        allow_legacy_heuristic: Optional[bool] = None,
    ) -> CouplingTerms:
        """Compute all coupling terms between entropy and geometry.
        
        Args:
            state: Quantum state vector
            partition: Partition defining the entanglement region
            
        Returns:
            CouplingTerms object containing all relevant tensors

        Note:
            This path runs the LEGACY state-space-gradient heuristic (the v1.2
            defect) and therefore raises unless `allow_legacy_heuristic=True`.
            It is retained for comparison against the honest pipeline in
            `compute_stress_tensor_field()`, not for producing results.
        """
        # Compute entropy gradient with edge mode handling
        entropy_grad = self.entropy.entropy_gradient(
            state, 
            partition, 
            include_edge=self.include_edge_modes,
            apply_uv_cutoff=True
        )
        
        # Compute stress tensor
        T, edge_contribution = self.compute_entropy_stress_tensor(
            entropy_grad, allow_legacy_heuristic=allow_legacy_heuristic)
        
        # Compute Einstein tensor
        G, higher_curvature = self.compute_einstein_tensor()
        
        # Compute residual (mismatch between geometry and entropy)
        residual = G - T

        # Check massless constraint: g^μν T_μν should be 0 for E=pc fields
        trace_violation = self.compute_tracelessness_violation(T)

        return CouplingTerms(
            entropy_grad,
            T,
            G,
            residual,
            edge_contribution,
            higher_curvature,
            trace_violation,
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

        if terms.tracelessness_violation is not None:
            result["tracelessness_violation"] = terms.tracelessness_violation

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