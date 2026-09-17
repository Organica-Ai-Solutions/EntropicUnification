"""Does the OPTIMISED metric have a continuum limit?

    venv/bin/python scripts/convergence_test.py

Answers the question `check_metric_resolved` cannot answer from a single
resolution. Run this before trusting any number extracted from an optimised
metric.

A fixed eps threshold cannot distinguish "unresolved" from "varies fast".
The discriminator is refinement: a smooth field's truncation parameter falls
~4x per lattice doubling; a field with no continuum limit does not move.

Replicates schwarzschild_test's loop exactly (Minkowski init, Adam 1e-3,
cosine schedule, grad clip 1.0, symmetry enforcement) at several lattice
sizes, with gates relaxed so nothing aborts, and reports eps vs N.
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

R_MIN, R_MAX, NQ, CLUSTER, LR, CLIP = 0.5, 5.0, 4, 1.5, 1e-3, 1.0
relax = GateConfig(strict=False)

def run(N, iters, record_at, lam=0.0):
    geo = GeometryEngine(lattice_size=N, dimensions=2,
                         dx=float(R_MAX-R_MIN)/(N-1), initial_metric="minkowski")
    em = EntropyModule(QE(NQ)); cl = CouplingLayer(geo, em)
    amps = torch.zeros(2**NQ, dtype=torch.complex128)
    amps[0]=amps[-1]=1/math.sqrt(2)                      # GHZ
    r = torch.linspace(R_MIN, R_MAX, N, dtype=torch.float64)
    pos = torch.linspace(R_MIN, R_MIN+CLUSTER, NQ).tolist()
    S = em.entropy_field(amps, pos, r).detach()
    opt = torch.optim.Adam([geo.metric_field], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=LR/10)
    out = {}
    for it in range(iters):
        opt.zero_grad(); geo._clear_cache()
        G = geo.compute_einstein_tensor()
        T = cl.compute_stress_tensor_field(S, metric_field=geo.metric_field,
                                           formulation=StressTensorFormulation.MASSLESS,
                                           gates=relax)
        loss = torch.sum((G-T)**2)
        if lam:
            g = geo.metric_field
            d2 = g[2:] - 2.0*g[1:-1] + g[:-2]
            loss = loss + lam * torch.sum(d2**2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([geo.metric_field], CLIP)
        opt.step(); sched.step(); geo._enforce_symmetry()
        if (it+1) in record_at:
            geo._clear_cache()
            out[it+1] = geo.metric_truncation_scale()
    return out

ITERS = 400
RECORD = {400}
print(f"eps after {ITERS} iters (tol {GateConfig().metric_resolution_tol:g})", flush=True)
print(f"{'lambda':>10s}" + "".join(f"{('N='+str(n)):>12s}" for n in (32,64,128))
      + f"{'gtt_range':>12s}", flush=True)
res={}
for lam in (0.0, 1e-2, 1.0, 1e2, 1e4):
    res[lam]={}; row=""
    for N in (32,64,128):
        res[lam][N]=run(N, ITERS, RECORD, lam=lam)[ITERS]
        row+=f"{res[lam][N]:12.3e}"
    print(f"{lam:10.0e}{row}", flush=True)
print("\nRefinement ratio (need >=2 AND eps<=0.02 at N=128)", flush=True)
for lam in res:
    r1=res[lam][32]/res[lam][64] if res[lam][64] else float('nan')
    r2=res[lam][64]/res[lam][128] if res[lam][128] else float('nan')
    ok = "RESOLVED" if (r1>=2 and r2>=2 and res[lam][128]<=0.02) else "unresolved"
    print(f"  lam={lam:8.0e}  32->64:{r1:5.2f}  64->128:{r2:5.2f}   {ok}", flush=True)
print("\nDONE", flush=True)
