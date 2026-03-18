"""
Scaling Experiment: r_s vs S_ent (Bekenstein-Hawking proportionality test)

Tests whether the fitted Schwarzschild radius r_s scales linearly with
entanglement entropy S_ent across a range of quantum states.

Physical hypothesis (Bekenstein-Hawking analog):
    r_s = k * S_ent

where k is a proportionality constant. If confirmed, this is a quantitative
Bekenstein-Hawking-like relationship derived entirely from entropic optimization.

States used:
    |psi(theta)> = cos(theta)|00> + sin(theta)|11>   (2-qubit generalized Bell)

    theta=0:    S = 0           (product state)
    theta=pi/6: S ~ 0.276       (weak entanglement)
    theta=pi/5: S ~ 0.410
    theta=pi/4: S = ln(2)~0.693 (maximal Bell)

Usage:
    python examples/scaling_experiment.py --device auto
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.quantum_engine import QuantumEngine, QuantumConfig
from core.geometry_engine import GeometryEngine
from core.entropy_module import EntropyModule
from core.coupling_layer import CouplingLayer, StressTensorFormulation
from examples.schwarzschild_test import fit_schwarzschild_radius


# ---------------------------------------------------------------------------
# Parameterized quantum states
# ---------------------------------------------------------------------------

def partial_bell_state(theta: float, num_qubits: int = 4) -> torch.Tensor:
    """
    |psi(theta)> = cos(theta)|00...0> + sin(theta)|11...1>

    For a partition into [first half | second half], the entanglement entropy is:
        S = -cos^2(theta) * ln(cos^2(theta)) - sin^2(theta) * ln(sin^2(theta))

    theta=0     -> S=0 (product state)
    theta=pi/4  -> S=ln(2) (maximally entangled Bell / GHZ)
    """
    psi = torch.zeros(2 ** num_qubits, dtype=torch.complex128)
    psi[0]  = float(np.cos(theta))   # |00...0>
    psi[-1] = float(np.sin(theta))   # |11...1>
    return psi


def theoretical_entropy(theta: float) -> float:
    """Von Neumann entropy of the reduced state for partial Bell."""
    c2 = np.cos(theta) ** 2
    s2 = np.sin(theta) ** 2
    eps = 1e-12
    S = 0.0
    if c2 > eps:
        S -= c2 * np.log(c2)
    if s2 > eps:
        S -= s2 * np.log(s2)
    return float(S)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(theta: float, cfg: dict) -> dict:
    """Run Schwarzschild test for a given entanglement angle theta."""
    device = torch.device(cfg["device"])
    dtype  = torch.float64

    geometry = GeometryEngine(
        dimensions=2,
        lattice_size=cfg["lattice_size"],
        dx=float(cfg["r_max"] - cfg["r_min"]) / (cfg["lattice_size"] - 1),
        initial_metric="minkowski",
        boundary_condition="dirichlet",
        dtype=dtype,
        device=device,
    )

    num_qubits = cfg["num_qubits"]
    qe = QuantumEngine(QuantumConfig(num_qubits=num_qubits, depth=2))
    entropy_module = EntropyModule(qe, include_edge_modes=False, conformal_invariance=True)
    coupling = CouplingLayer(
        geometry_engine=geometry,
        entropy_module=entropy_module,
        coupling_strength=1.0,
        stress_form=StressTensorFormulation.MASSLESS,
        include_edge_modes=False,
        conformal_invariance=True,
    )

    # Build parameterized state
    psi = partial_bell_state(theta, num_qubits=num_qubits)
    n_half = num_qubits // 2
    partition = list(range(n_half))

    # Compute entropy and gradient
    S_ent = entropy_module.compute_entanglement_entropy(psi, partition).item()
    grad  = entropy_module.entropy_gradient(psi, partition).real.detach().to(device)

    # Project gradient to spacetime dim
    dim = geometry.dimensions
    bg = grad[:dim] if grad.shape[0] >= dim else torch.cat(
        [grad, torch.zeros(dim - grad.shape[0], dtype=grad.dtype, device=device)]
    )

    r_grid  = torch.linspace(cfg["r_min"], cfg["r_max"], cfg["lattice_size"],
                              dtype=dtype, device=device)
    sigma   = cfg["localization_sigma"]
    weights = torch.exp(-r_grid ** 2 / (2.0 * sigma ** 2))
    weights = weights / (weights.max() + 1e-12)

    hbar_factor = coupling.hbar_factor * coupling.coupling_strength
    N = cfg["lattice_size"]

    optimizer = torch.optim.Adam([geometry.metric_field], lr=cfg["learning_rate"])

    for _ in range(cfg["n_iterations"]):
        optimizer.zero_grad()
        geometry._clear_cache()

        G_all = geometry.compute_einstein_tensor()
        grads_field = weights.unsqueeze(1) * bg.unsqueeze(0)
        outer       = torch.einsum("ni,nj->nij", grads_field, grads_field)
        contraction = torch.sum(grads_field ** 2, dim=1)
        g           = geometry.metric_field

        T_all = hbar_factor * (outer - (1.0 / dim) * g * contraction.view(N, 1, 1))
        residual  = G_all - T_all
        total_loss = torch.sum(residual ** 2)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([geometry.metric_field], 1.0)
        optimizer.step()
        geometry._enforce_symmetry()

    g_tt_final = geometry.metric_field[:, 0, 0].detach().cpu().numpy()
    r_np       = r_grid.cpu().numpy()
    r_s_fit    = fit_schwarzschild_radius(r_np, g_tt_final)

    # Explicit CUDA memory cleanup — prevents OOM on successive runs
    del geometry, qe, entropy_module, coupling, optimizer, r_grid, weights, bg, grad, psi

    return {"theta": theta, "S_ent": S_ent, "r_s": r_s_fit}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="r_s vs S_ent scaling experiment"
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lattice",    type=int, default=32)
    parser.add_argument("--device",     type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-dir",   type=str, default="schwarzschild_results/scaling")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    cfg = {
        "lattice_size":       args.lattice,
        "r_min":              0.5,
        "r_max":              5.0,
        "num_qubits":         4,
        "localization_sigma": 0.8,
        "n_iterations":       args.iterations,
        "learning_rate":      1e-3,
        "device":             device,
    }

    # Sweep: product state -> half -> quarter -> Bell
    thetas = [0.0, np.pi/8, np.pi/6, np.pi/5, np.pi/4]

    print(f"\n=== Bekenstein-Hawking Scaling Experiment ===")
    print(f"Sweeping entanglement angle theta = 0 -> pi/4")
    print(f"Lattice: {cfg['lattice_size']}, Iterations: {cfg['n_iterations']}, Device: {device}\n")
    print(f"  {'theta':>8}  {'S_ent':>8}  {'r_s':>10}  {'r_s/S_ent':>10}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")

    # Run each theta in a fresh subprocess to isolate CUDA contexts
    import subprocess
    import json as _json
    import tempfile as _tf

    results = []
    for theta in thetas:
        S_theory = theoretical_entropy(theta)
        if S_theory < 1e-6:
            print(f"  {theta:>8.4f}  {0.0:>8.6f}  {'(skip)':>10}  {'(skip)':>10}")
            results.append({"theta": theta, "S_ent": 0.0, "r_s": 0.0})
            continue

        # Write a tiny worker script to a temp file and run it.
        # Inline the run_one logic to avoid circular import issues.
        worker_code = f"""
import sys, json, os
sys.path.insert(0, {repr(str(Path(__file__).parent.parent))})
os.chdir({repr(str(Path(__file__).parent.parent))})
import numpy as np
import torch
from core.quantum_engine import QuantumEngine, QuantumConfig
from core.geometry_engine import GeometryEngine
from core.entropy_module import EntropyModule
from core.coupling_layer import CouplingLayer, StressTensorFormulation
from examples.schwarzschild_test import fit_schwarzschild_radius

theta = {theta}
cfg = {repr(cfg)}

device = torch.device(cfg["device"])
dtype  = torch.float64
geometry = GeometryEngine(dimensions=2, lattice_size=cfg["lattice_size"],
    dx=float(cfg["r_max"]-cfg["r_min"])/(cfg["lattice_size"]-1),
    initial_metric="minkowski", boundary_condition="dirichlet", dtype=dtype, device=device)
num_qubits = cfg["num_qubits"]
qe = QuantumEngine(QuantumConfig(num_qubits=num_qubits, depth=2))
em = EntropyModule(qe, include_edge_modes=False, conformal_invariance=True)
coupling = CouplingLayer(geometry_engine=geometry, entropy_module=em, coupling_strength=1.0,
    stress_form=StressTensorFormulation.MASSLESS, include_edge_modes=False, conformal_invariance=True)

psi = torch.zeros(2**num_qubits, dtype=torch.complex128)
psi[0]  = float(np.cos(theta))
psi[-1] = float(np.sin(theta))
partition = list(range(num_qubits//2))
S_ent = em.compute_entanglement_entropy(psi, partition).item()
grad  = em.entropy_gradient(psi, partition).real.detach().to(device)
dim   = geometry.dimensions
bg    = grad[:dim] if grad.shape[0] >= dim else torch.cat([grad, torch.zeros(dim-grad.shape[0], dtype=grad.dtype, device=device)])

r_grid  = torch.linspace(cfg["r_min"], cfg["r_max"], cfg["lattice_size"], dtype=dtype, device=device)
weights = torch.exp(-r_grid**2/(2.0*cfg["localization_sigma"]**2))
weights = weights/(weights.max()+1e-12)
hbar_f  = coupling.hbar_factor * coupling.coupling_strength
N = cfg["lattice_size"]
opt = torch.optim.Adam([geometry.metric_field], lr=cfg["learning_rate"])

for _ in range(cfg["n_iterations"]):
    opt.zero_grad(); geometry._clear_cache()
    G     = geometry.compute_einstein_tensor()
    gf    = weights.unsqueeze(1)*bg.unsqueeze(0)
    outer = torch.einsum("ni,nj->nij", gf, gf)
    cont  = torch.sum(gf**2, dim=1)
    g     = geometry.metric_field
    T     = hbar_f*(outer-(1.0/dim)*g*cont.view(N,1,1))
    loss  = torch.sum((G-T)**2)
    loss.backward(); torch.nn.utils.clip_grad_norm_([geometry.metric_field],1.0); opt.step()
    geometry._enforce_symmetry()

gtt = geometry.metric_field[:,0,0].detach().cpu().numpy()
r_np = r_grid.cpu().numpy()
r_s  = fit_schwarzschild_radius(r_np, gtt)
print("RESULT:" + json.dumps({{"theta":theta,"S_ent":S_ent,"r_s":r_s}}))
"""
        with _tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(worker_code)
            worker_path = f.name

        proc = subprocess.run(
            [sys.executable, worker_path],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        Path(worker_path).unlink(missing_ok=True)

        # Parse result from stdout
        r = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT:"):
                r = _json.loads(line[7:])
                break
        if r is None:
            print(f"  {theta:>8.4f}  ERROR (subprocess failed)")
            if proc.stderr:
                print(f"    stderr: {proc.stderr[:200]}")
            continue

        ratio = r["r_s"] / r["S_ent"] if r["S_ent"] > 1e-8 else float("nan")
        print(f"  {theta:>8.4f}  {r['S_ent']:>8.6f}  {r['r_s']:>10.6f}  {ratio:>10.6f}")
        results.append(r)

    # Fit linear model r_s = k * S_ent (no intercept)
    S_vals = np.array([r["S_ent"] for r in results if r["S_ent"] > 1e-6])
    rs_vals = np.array([r["r_s"] for r in results if r["S_ent"] > 1e-6])

    if len(S_vals) >= 2:
        # Least-squares fit through origin: k = sum(S*r_s) / sum(S^2)
        k_fit = float(np.dot(S_vals, rs_vals) / np.dot(S_vals, S_vals))
        residuals = rs_vals - k_fit * S_vals
        r2 = 1.0 - np.var(residuals) / (np.var(rs_vals) + 1e-12)
        print(f"\nLinear fit r_s = k * S_ent:")
        print(f"  k = {k_fit:.6f}")
        print(f"  R^2 = {r2:.6f}")

        if r2 > 0.90:
            print("  STRONG linear scaling confirmed (R^2 > 0.90) -- Bekenstein-Hawking analog.")
        elif r2 > 0.70:
            print("  Moderate linear scaling (R^2 > 0.70) -- promising, needs more iterations.")
        else:
            print("  Weak scaling -- may need more iterations or a different range.")

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(save_dir / "scaling_results.npy"), results)
    print(f"\nResults saved -> {save_dir / 'scaling_results.npy'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(S_vals, rs_vals, s=80, zorder=5, label="Simulation")
        if len(S_vals) >= 2:
            S_line = np.linspace(0, S_vals.max() * 1.1, 100)
            ax.plot(S_line, k_fit * S_line, "r--", lw=1.5,
                    label=f"r_s = {k_fit:.4f} * S  (R^2={r2:.3f})")
        ax.set_xlabel("Entanglement entropy $S_{\\rm ent}$", fontsize=12)
        ax.set_ylabel("Fitted Schwarzschild radius $r_s$", fontsize=12)
        ax.set_title("$r_s$ vs $S_{\\rm ent}$ — Bekenstein-Hawking Scaling", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        out = save_dir / "rs_vs_sent.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved -> {out}")
    except ImportError:
        print("matplotlib not available - skipping plot")


if __name__ == "__main__":
    main()
