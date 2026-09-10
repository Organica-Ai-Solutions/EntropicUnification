#!/usr/bin/env python3
"""Reproduce every tolerance in docs/VALIDATION.md.

The document's thesis is that tolerances are measured rather than guessed, so
the measurement has to be in the tree and re-runnable:

    venv/bin/python scripts/calibrate_gates.py

Prints the three calibration tables: the Riemann error-ratio sweep, the noise
comparison behind `metric_resolution_tol`, and the FAULKNER trace identity.

Note on this machine: `import pennylane` takes ~21 minutes, so it is stubbed —
nothing measured here touches it. `QuantumEngine.reduced_density_matrix` is
pure torch and is bound onto a stub carrying only `num_qubits`.
"""
from __future__ import annotations

import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_fake = types.ModuleType("pennylane")
_fake.__version__ = "stub"
sys.modules.setdefault("pennylane", _fake)

import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from core.coupling_layer import CouplingLayer  # noqa: E402
from core.entropy_module import EntropyModule  # noqa: E402
from core.geometry_engine import GeometryEngine  # noqa: E402
from core.quantum_engine import QuantumEngine  # noqa: E402
from core.utils.finite_difference import fixed_finite_difference  # noqa: E402
from core.validation import GateConfig  # noqa: E402


class _StubQE:
    """num_qubits plus the genuine pure-torch partial trace."""

    def __init__(self, n: int) -> None:
        self.num_qubits = n

    reduced_density_matrix = QuantumEngine.reduced_density_matrix


def smooth_metric(n: int, dims: int, amp: float, k: int = 1) -> GeometryEngine:
    """Flat metric plus a long-wavelength sinusoid: smooth, with a continuum limit."""
    g = GeometryEngine(lattice_size=n, dimensions=dims)
    x = torch.linspace(0, 1, n, dtype=torch.float64)
    bump = amp * torch.sin(2 * math.pi * k * x)
    with torch.no_grad():
        for a in range(dims):
            g.metric_field[:, a, a] += bump
    g._clear_cache()
    return g


def noisy_metric(n: int, dims: int, amp: float, seed: int = 0) -> GeometryEngine:
    """Per-site Gaussian noise: no continuum limit at any resolution."""
    g = GeometryEngine(lattice_size=n, dimensions=dims)
    torch.manual_seed(seed)
    with torch.no_grad():
        g.metric_field += amp * torch.randn_like(g.metric_field)
        g.metric_field.copy_(0.5 * (g.metric_field + g.metric_field.transpose(1, 2)))
    g._clear_cache()
    return g


def riemann_ratio_table() -> None:
    print("\n== riemann_error_ratio: violation / predicted truncation error ==")
    print(f"{'case':22s}" + "".join(f"{('N=' + str(n)):>9s}" for n in (16, 32, 64, 128, 256)))
    worst_lo, worst_hi = float("inf"), 0.0
    for dims in (2, 4):
        for amp in (0.01, 0.1):
            cells = []
            for n in (16, 32, 64, 128, 256):
                g = smooth_metric(n, dims, amp)
                eps = g.metric_truncation_scale()
                viol = max(g.riemann_identity_violations().values())
                ratio = viol / eps if eps > 0 else float("nan")
                worst_lo, worst_hi = min(worst_lo, ratio), max(worst_hi, ratio)
                cells.append(f"{ratio:9.2f}")
            print(f"dim {dims}, amp {amp:<8}" + "".join(cells))
    print(f"observed ratio range: {worst_lo:.2f} .. {worst_hi:.2f}  "
          f"(GateConfig default {GateConfig().riemann_error_ratio})")

    print("\n-- absolute violations, showing O(dx^2) convergence --")
    prev = None
    for n in (16, 32, 64, 128, 256):
        v = max(smooth_metric(n, 2, 0.1).riemann_identity_violations().values())
        fall = "" if prev is None else f"  ({prev / v:.2f}x)"
        print(f"  N={n:4d}: {v:.3e}{fall}")
        prev = v


def resolution_table() -> None:
    print("\n== metric_resolution_tol: smooth vs per-site noise ==")
    tol = GateConfig().metric_resolution_tol
    print(f"tolerance = {tol:g}")
    print("\n-- per-site noise at N=64, dim 2 --")
    for amp in (0.0001, 0.001, 0.01, 0.05):
        g = noisy_metric(64, 2, amp)
        eps = g.metric_truncation_scale()
        viol = max(g.riemann_identity_violations().values())
        print(f"  amp={amp:<8} eps={eps:.3e}  viol={viol:.3e}  "
              f"ratio={viol / eps:.2f}  {'FAIL' if eps > tol else 'pass'}")
    print("\n-- the honest limitation: noise eps does NOT fall with refinement --")
    for n in (32, 64, 128):
        print(f"  noise amp=0.001  N={n:4d}: eps={noisy_metric(n, 2, 0.001).metric_truncation_scale():.3e}")
    for n in (32, 64, 128):
        print(f"  smooth amp=0.1   N={n:4d}: eps={smooth_metric(n, 2, 0.1).metric_truncation_scale():.3e}")


def faulkner_trace_table() -> None:
    print("\n== FAULKNER trace identity: g^uv T_uv should equal (1-n) BoxS ==")
    em = EntropyModule(_StubQE(4))
    amps = torch.zeros(16, dtype=torch.complex128)
    amps[0] = amps[-1] = 1 / math.sqrt(2)          # GHZ
    n_pts = 64
    r = torch.linspace(1.0, 5.0, n_pts, dtype=torch.float64)
    field = em.entropy_field(amps, torch.linspace(1.5, 3.5, 4).tolist(), r)
    relax = GateConfig(strict=False)
    for dims in (2, 4):
        geo = GeometryEngine(lattice_size=n_pts, dimensions=dims)
        cl = CouplingLayer(geo, em)
        T = cl.compute_stress_tensor_field(field, formulation="faulkner", gates=relax)
        g = geo.metric_field
        with torch.no_grad():
            g_inv = torch.linalg.inv(g)
            trace = torch.einsum("nab,nab->n", g_inv, T)
            d2S = fixed_finite_difference(field.values.to(g.dtype), order=2,
                                          axis=0, dx=float(geo.dx))
            hess = torch.zeros((n_pts, dims, dims), dtype=g.dtype)
            hess[:, 1, 1] = d2S
            box = torch.einsum("nab,nab->n", g_inv, hess)
            pred = (1 - dims) * box * cl.hbar_factor * cl.coupling_strength
            mismatch = float((trace - pred).abs().max() / (pred.abs().max() + 1e-30))
        print(f"  n={dims}: max|trace|={float(trace.abs().max()):.3e}  "
              f"predicted={float(pred.abs().max()):.3e}  "
              f"relative mismatch={mismatch:.2e}")
    print("\n  Caveat: the Hessian here is the coordinate second derivative,")
    print("  not the covariant one (no Christoffel term), so this verifies")
    print("  internal algebraic consistency of the implemented formula.")


if __name__ == "__main__":
    riemann_ratio_table()
    resolution_table()
    faulkner_trace_table()
    print("\ndone")
