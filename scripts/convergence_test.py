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

def run(N, iters, record_at):
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_([geo.metric_field], CLIP)
        opt.step(); sched.step(); geo._enforce_symmetry()
        if (it+1) in record_at:
            geo._clear_cache()
            out[it+1] = geo.metric_truncation_scale()
    return out

ITERS = 400
RECORD = {50, 100, 200, 400}
print(f"eps = ||d2g||/||g||  (tolerance {GateConfig().metric_resolution_tol:g})")
print(f"{'N':>6s}" + "".join(f"{('it'+str(i)):>12s}" for i in sorted(RECORD)), flush=True)
res = {}
for N in (32, 64, 128, 256):
    res[N] = run(N, ITERS, RECORD)
    print(f"{N:6d}" + "".join(f"{res[N][i]:12.3e}" for i in sorted(RECORD)), flush=True)

print("\nRefinement ratio eps(N) / eps(2N)  — smooth field should give ~4")
for i in sorted(RECORD):
    row = []
    for N in (32, 64, 128):
        row.append(res[N][i] / res[2*N][i] if res[2*N][i] else float('nan'))
    print(f"  it{i:<4d} " + "  ".join(f"{N}->{2*N}: {v:5.2f}" for N, v in zip((32,64,128), row)), flush=True)
print("\nDONE", flush=True)
