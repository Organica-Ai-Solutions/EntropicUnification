"""Is the entropy source differentiable to the order the formulation uses?

    venv/bin/python scripts/source_regularity.py

S(r) with interpolation="linear" is piecewise linear: C0, not C1. This
measures the two consequences — dS is stencil-dependent at the kinks, and
d2S diverges as 1/dx, which makes FAULKNER ill-posed on this source.

The FD fix was verified on exp(sin 3x) — analytic and smooth. But S(r) with
interpolation="linear" is piecewise linear: C0, not C1, with kinks at the
qubit positions. High-order stencils amplify non-smooth data rather than
improving on it, so the fix may be correct for the metric and wrong for the
source.
"""
import sys, math, types
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("pennylane", types.ModuleType("pennylane"))
import torch; torch.set_default_dtype(torch.float64)
from core.quantum_engine import QuantumEngine
from core.entropy_module import EntropyModule
from core.utils.finite_difference import fixed_finite_difference as fd

class QE:
    def __init__(s,n): s.num_qubits=n
    reduced_density_matrix = QuantumEngine.reduced_density_matrix

def two_point(f, dx):
    """The OLD edge behaviour, for comparison."""
    d = fd(f,1,0,dx).clone()
    d[0]=(f[1]-f[0])/dx; d[-1]=(f[-1]-f[-2])/dx
    return d

em=EntropyModule(QE(4))
amps=torch.zeros(16,dtype=torch.complex128); amps[0]=amps[-1]=1/math.sqrt(2)
print("S(r) is piecewise linear; ||dS||^2 drives the stress tensor.\n", flush=True)
print(f"{'N':>6s}{'interp':>10s}{'|dS|^2 new':>14s}{'|dS|^2 old':>14s}{'ratio':>8s}", flush=True)
for interp in ("linear","steps"):
    for N in (32,64,128,256):
        r=torch.linspace(0.5,5.0,N,dtype=torch.float64); dx=float(r[1]-r[0])
        S=em.entropy_field(amps, torch.linspace(0.5,2.0,4).tolist(), r,
                           interpolation=interp).values
        a=float((fd(S,1,0,dx)**2).sum()); b=float((two_point(S,dx)**2).sum())
        print(f"{N:6d}{interp:>10s}{a:14.4e}{b:14.4e}{(a/b if b else float('nan')):8.2f}", flush=True)

print("\nSecond derivative (FAULKNER uses this) on a piecewise-linear field:", flush=True)
print("d2S is a sum of deltas at the kinks — it does NOT converge.", flush=True)
print(f"{'N':>6s}{'max|d2S|':>14s}", flush=True)
for N in (32,64,128,256,512):
    r=torch.linspace(0.5,5.0,N,dtype=torch.float64); dx=float(r[1]-r[0])
    S=em.entropy_field(amps, torch.linspace(0.5,2.0,4).tolist(), r).values
    print(f"{N:6d}{float(fd(S,2,0,dx).abs().max()):14.4e}", flush=True)
print("\nDONE", flush=True)
