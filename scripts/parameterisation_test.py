"""Does a smooth-by-construction parameterisation have a continuum limit?

    venv/bin/python scripts/parameterisation_test.py

The bare experiment treats all N lattice values as free parameters, so the
optimiser can reach a pointwise solution that is not a field; a smoothness
penalty does not fix that (scripts/smoothness_sweep.py). This expands the
metric in a small Chebyshev basis and optimises the COEFFICIENTS instead.

The decisive measurement optimises ONCE and then resamples the same
coefficient vector at increasing lattice sizes, so the optimiser is held
fixed and any deviation from a 4x fall per doubling is pure discretisation.

The bare experiment treats all N lattice values as free parameters, so the
optimiser can find a pointwise solution that is not a field. A smoothness
penalty does not fix that (scripts/smoothness_sweep.py).

This tries the structural alternative: expand the metric in a small Chebyshev
basis and optimise the COEFFICIENTS. Any field so built is smooth by
construction, and refining the lattice resamples the same smooth function —
so eps must fall as dx^2 if the machinery is sound.

Acceptance criterion is the same one the repo uses everywhere:
refinement ratio >= 2, eps <= 0.02 at the finest lattice.
"""
import sys, math, types
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules.setdefault("pennylane", types.ModuleType("pennylane"))
import torch; torch.set_default_dtype(torch.float64)
from core.quantum_engine import QuantumEngine
from core.geometry_engine import GeometryEngine
from core.entropy_module import EntropyModule
from core.coupling_layer import CouplingLayer, StressTensorFormulation
from core.validation import GateConfig

class QE:
    def __init__(s,n): s.num_qubits=n
    reduced_density_matrix = QuantumEngine.reduced_density_matrix

R_MIN, R_MAX, NQ, CLUSTER, LR, ITERS = 0.5, 5.0, 4, 1.5, 1e-2, 400
relax = GateConfig(strict=False)

def cheb_basis(N, k):
    """T_0..T_{k-1} sampled on the lattice, mapped to [-1,1]."""
    x = torch.linspace(-1.0, 1.0, N, dtype=torch.float64)
    B = [torch.ones_like(x), x]
    while len(B) < k:
        B.append(2*x*B[-1] - B[-2])
    return torch.stack(B[:k], dim=1)          # (N, k)

def run(N, k, iters=ITERS):
    geo = GeometryEngine(lattice_size=N, dimensions=2,
                         dx=float(R_MAX-R_MIN)/(N-1), initial_metric="minkowski")
    em = EntropyModule(QE(NQ)); cl = CouplingLayer(geo, em)
    amps = torch.zeros(2**NQ, dtype=torch.complex128); amps[0]=amps[-1]=1/math.sqrt(2)
    r = torch.linspace(R_MIN, R_MAX, N, dtype=torch.float64)
    S = em.entropy_field(amps, torch.linspace(R_MIN, R_MIN+CLUSTER, NQ).tolist(), r).detach()

    B = cheb_basis(N, k)
    # metric = Minkowski + sum_j c[j,mu] * T_j(x), diagonal perturbation only
    c = torch.zeros(k, 2, dtype=torch.float64, requires_grad=True)
    base = torch.zeros(N, 2, 2, dtype=torch.float64)
    base[:,0,0] = -1.0; base[:,1,1] = 1.0

    opt = torch.optim.Adam([c], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=LR/10)
    for _ in range(iters):
        opt.zero_grad(); geo._clear_cache()
        pert = B @ c                                    # (N,2)
        g = base.clone()
        g[:,0,0] = base[:,0,0] + pert[:,0]
        g[:,1,1] = base[:,1,1] + pert[:,1]
        G = geo.compute_einstein_tensor(g)
        T = cl.compute_stress_tensor_field(S, metric_field=g,
                formulation=StressTensorFormulation.MASSLESS, gates=relax)
        loss = torch.sum((G-T)**2)
        loss.backward(); opt.step(); sched.step()

    with torch.no_grad():
        pert = B @ c
        g = base.clone()
        g[:,0,0] += pert[:,0]; g[:,1,1] += pert[:,1]
        geo._clear_cache()
        eps = geo.metric_truncation_scale(g)
        gtt_rng = float(g[:,0,0].max()-g[:,0,0].min())
    return eps, gtt_rng, float(loss), c.detach()

def eps_of(coeffs, N):
    """Resample ONE coefficient vector onto a lattice of size N and measure eps.

    This is the clean test: the same smooth function, sampled more finely.
    Any deviation from a 4x fall per doubling is a property of the
    discretisation alone, with the optimiser held fixed.
    """
    geo = GeometryEngine(lattice_size=N, dimensions=2,
                         dx=float(R_MAX-R_MIN)/(N-1), initial_metric="minkowski")
    B = cheb_basis(N, coeffs.shape[0])
    g = torch.zeros(N,2,2,dtype=torch.float64)
    g[:,0,0] = -1.0 + (B @ coeffs)[:,0]
    g[:,1,1] =  1.0 + (B @ coeffs)[:,1]
    return geo.metric_truncation_scale(g)

print("Optimise ONCE at N=128, then resample the same coefficients.", flush=True)
print("A fixed smooth function must give exactly ~4x per doubling.\n", flush=True)
print(f"{'k':>4s}" + "".join(f"{('N='+str(n)):>12s}" for n in (32,64,128,256,512)), flush=True)
store={}
for k in (4, 8, 16):
    _,_,_,c = run(128, k)
    store[k]=c
    print(f"{k:4d}" + "".join(f"{eps_of(c,N):12.3e}" for N in (32,64,128,256,512)), flush=True)

print("\nRatios eps(N)/eps(2N) — pure discretisation, optimiser held fixed", flush=True)
for k,c in store.items():
    e={N:eps_of(c,N) for N in (32,64,128,256,512)}
    rs=[e[N]/e[2*N] if e[2*N] else float('nan') for N in (32,64,128,256)]
    print(f"  k={k:3d}  " + "  ".join(f"{N}->{2*N}:{r:6.2f}" for N,r in zip((32,64,128,256),rs)), flush=True)
print("\nDONE", flush=True)
