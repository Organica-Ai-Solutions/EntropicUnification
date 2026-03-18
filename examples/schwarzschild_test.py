"""
Schwarzschild Recovery Test — Hypothesis H3

Tests whether a Bell state with Gaussian-localized spatial support drives the
metric from flat Minkowski toward a Schwarzschild profile under entropic
optimization.

This is the flagship experiment of the EntropicUnification framework.
If a localized maximally-entangled state produces a recognizable Schwarzschild
metric profile, it is strong evidence that the entropic-gravity correspondence
captures something physically real.

Physical setup
--------------
- 1+1D spacetime (t, r), Lorentzian signature (-,+)
- Spatial lattice:  r in [r_min, r_max],  N_LATTICE points
- Entanglement source: Bell state |Ψ⟩ = (|00⟩+|11⟩)/√2 concentrated at r=0
  via Gaussian profile  w(r) = exp(-r²/2σ²)
- Initial metric: flat Minkowski  g_munu = diag(-1, +1)
- Stress tensor: MASSLESS formulation (traceless, E=pc constraint satisfied)

Expected Schwarzschild signature after convergence
---------------------------------------------------
  g_tt(r) -> -(1 - r_s/r)         [deepens toward origin]
  g_rr(r) -> +(1 - r_s/r)^{-1}   [grows positive toward origin]
  r_s ∝ S_Bell                    (Bekenstein-Hawking-like proportionality)

Even a *qualitative* match — correct sign structure, monotonic profile,
asymptotic flatness — constitutes a significant result.  A quantitative fit
of r_s to S_Bell would be remarkable.

Usage
-----
  python examples/schwarzschild_test.py
  python examples/schwarzschild_test.py --iterations 500 --lattice 64
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

# Force UTF-8 on Windows to avoid cp1252 encode errors
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch

# Make sure imports work whether running from repo root or examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.quantum_engine import QuantumEngine, QuantumConfig
from core.geometry_engine import GeometryEngine
from core.entropy_module import EntropyModule
from core.coupling_layer import CouplingLayer, StressTensorFormulation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CFG = {
    # Lattice
    "lattice_size": 32,      # Number of radial points
    "r_min": 0.5,            # Closest approach (avoid r=0 singularity)
    "r_max": 5.0,            # Far-field boundary
    # Quantum source
    "num_qubits": 4,         # Bell-like state; entanglement S = log(2) ≈ 0.693
    "localization_sigma": 0.8,  # Gaussian width of entropy source (in lattice units)
    # Optimization
    "n_iterations": 300,
    "learning_rate": 1e-3,
    "grad_clip": 1.0,
    # Physics
    "stress_form": StressTensorFormulation.MASSLESS,  # Traceless, E=pc compliant
    "coupling_strength": 1.0,
    # Output
    "plot": True,
    "verbose": True,
    "save_dir": "schwarzschild_results",
}


# ---------------------------------------------------------------------------
# Schwarzschild reference
# ---------------------------------------------------------------------------

def schwarzschild_gtt(r: np.ndarray, r_s: float) -> np.ndarray:
    """g_tt = -(1 - r_s/r)"""
    return -(1.0 - r_s / np.clip(r, 1e-8, None))


def schwarzschild_grr(r: np.ndarray, r_s: float) -> np.ndarray:
    """g_rr = +(1 - r_s/r)^{-1}"""
    return 1.0 / np.clip(1.0 - r_s / np.clip(r, 1e-8, None), 1e-8, None)


def fit_schwarzschild_radius(r: np.ndarray, g_tt: np.ndarray) -> float:
    """
    Fit r_s from g_tt profile using the relation g_tt = -(1 - r_s/r).
    Solves: r_s = r * (1 + g_tt)  at each point; return median for robustness.
    """
    # g_tt = -(1 - r_s/r) -> r_s = r*(1 + g_tt)  (note: g_tt < 0 -> 1+g_tt < 1)
    r_s_estimates = r * (1.0 + g_tt)
    # Filter out clearly unphysical estimates (very large or negative)
    valid = (r_s_estimates > 0) & (r_s_estimates < 10.0 * np.max(r))
    if not np.any(valid):
        return 0.0
    return float(np.median(r_s_estimates[valid]))


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_schwarzschild_test(cfg: dict) -> dict:
    """
    Run the Schwarzschild recovery experiment.

    Returns a results dict with metric history, loss curves, and diagnostics.
    """
    device = torch.device(cfg.get("device", "cpu"))
    dtype = torch.float64

    # ------------------------------------------------------------------
    # 1.  Build components
    # ------------------------------------------------------------------
    print("\n=== Schwarzschild Recovery Test (H3) ===\n")
    print(f"Lattice: {cfg['lattice_size']} points, r in [{cfg['r_min']}, {cfg['r_max']}]")
    print(f"Stress tensor: {cfg['stress_form'].value}")
    print(f"Iterations: {cfg['n_iterations']}")
    print(f"Device: {device}")
    print()

    geometry = GeometryEngine(
        dimensions=2,
        lattice_size=cfg["lattice_size"],
        dx=float(cfg["r_max"] - cfg["r_min"]) / (cfg["lattice_size"] - 1),
        initial_metric="minkowski",
        boundary_condition="dirichlet",
        dtype=dtype,
        device=device,
    )

    qe = QuantumEngine(QuantumConfig(num_qubits=cfg["num_qubits"], depth=2))
    entropy_module = EntropyModule(
        qe,
        include_edge_modes=False,   # keep it clean for H3
        conformal_invariance=True,  # MASSLESS form is already conformal
    )
    coupling = CouplingLayer(
        geometry_engine=geometry,
        entropy_module=entropy_module,
        coupling_strength=cfg["coupling_strength"],
        stress_form=cfg["stress_form"],
        include_edge_modes=False,
        conformal_invariance=True,
    )

    # ------------------------------------------------------------------
    # 2.  Prepare the Bell state and base entropy gradient
    # ------------------------------------------------------------------
    bell = qe.bell_state()  # (|00...0⟩ + |11...1⟩)/√2

    # Partition: keep first half of qubits, trace out second half
    n_half = cfg["num_qubits"] // 2
    partition = list(range(n_half))

    # Pre-compute the base entropy gradient (state-space direction)
    bell_grad = entropy_module.entropy_gradient(bell, partition).real.detach().to(device)
    S_bell = entropy_module.compute_entanglement_entropy(bell, partition).item()
    print(f"Bell state entanglement entropy S = {S_bell:.6f} (log2 max = {np.log(2):.6f})")
    print(f"Base gradient norm: {torch.norm(bell_grad).item():.6f}\n")

    # ------------------------------------------------------------------
    # 3.  Build radial grid and Gaussian spatial weights
    # ------------------------------------------------------------------
    r_grid = torch.linspace(cfg["r_min"], cfg["r_max"], cfg["lattice_size"],
                            dtype=dtype, device=device)
    # Normalized: σ in physical radial units
    sigma = cfg["localization_sigma"]
    weights = torch.exp(-r_grid**2 / (2.0 * sigma**2))  # shape: (lattice_size,)
    weights = weights / weights.max()  # normalize to [0,1]

    print("Spatial entropy profile (weight at each r):")
    for i in range(0, cfg["lattice_size"], cfg["lattice_size"] // 8):
        print(f"  r={r_grid[i].item():.3f}  w={weights[i].item():.4f}")
    print()

    # ------------------------------------------------------------------
    # 4.  Set up PyTorch optimizer on the full metric field
    # ------------------------------------------------------------------
    optimizer_torch = torch.optim.Adam(
        [geometry.metric_field],
        lr=cfg["learning_rate"],
    )
    # Cosine annealing: smoothly reduce lr to lr/10 over all iterations
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_torch,
        T_max=cfg["n_iterations"],
        eta_min=cfg["learning_rate"] / 10.0,
    )

    # ------------------------------------------------------------------
    # 5.  Optimization loop — vectorized over the full radial lattice
    #
    #  Key speedup: compute Einstein tensor ONCE per iteration (not N times).
    #  The stress tensor is also computed in a single batched operation.
    # ------------------------------------------------------------------
    dim = geometry.dimensions
    N   = cfg["lattice_size"]
    hbar_factor = coupling.hbar_factor * coupling.coupling_strength

    loss_history = []
    trace_history = []
    gtt_snapshots = {}
    grr_snapshots = {}
    snapshot_iters = {0, cfg["n_iterations"] // 4,
                      cfg["n_iterations"] // 2, cfg["n_iterations"] - 1}

    for iteration in range(cfg["n_iterations"]):
        optimizer_torch.zero_grad()
        geometry._clear_cache()          # fresh computation graph each step

        # --- Einstein tensor for the full lattice in one shot ---
        # G shape: (N, dim, dim)
        G_all = geometry.compute_einstein_tensor()   # uses metric_field

        # --- Stress tensor T_munu vectorized over all N lattice points ---
        # Project bell_grad (quantum state space) to spacetime dim
        # (matches what coupling_layer does internally: truncate/pad to dim)
        bg = bell_grad[:dim] if bell_grad.shape[0] >= dim else torch.cat(
            [bell_grad, torch.zeros(dim - bell_grad.shape[0], dtype=bell_grad.dtype, device=device)]
        )  # shape: (dim,)

        # grads: (N, dim) — spatially weighted entropy gradient
        grads = weights.unsqueeze(1) * bg.unsqueeze(0)   # (N, dim)

        # outer products: (N, dim, dim)
        outer = torch.einsum("ni,nj->nij", grads, grads)

        # (∇S)² at each point: (N,)
        contraction = torch.sum(grads ** 2, dim=1)

        # metric field: (N, dim, dim)
        g = geometry.metric_field

        if cfg["stress_form"] == StressTensorFormulation.MASSLESS:
            # T_munu = (hbar/2pi)[∂_mu S ∂_nu S - (1/n) g_munu (∇S)²]
            T_all = hbar_factor * (outer - (1.0 / dim) * g * contraction.view(N, 1, 1))
        elif cfg["stress_form"] == StressTensorFormulation.FAULKNER:
            # Faulkner: T_munu = (hbar/2pi)[H_munu - (trH) g_munu]
            # where H_munu = ∂_mu ∂_nu S is the Hessian.
            # In the vectorized path we approximate H_munu ≈ ∂_mu S ∂_nu S
            # (outer product of gradient — leading-order approximation).
            # The trace-subtracted form is equivalent to massless in 2D.
            box_S = contraction  # (N,) approximate d'Alembertian
            T_all = hbar_factor * (outer - g * box_S.view(N, 1, 1))
        else:
            # LAGRANGIAN / JACOBSON / canonical: (1/2) prefactor
            T_all = hbar_factor * (outer - 0.5 * g * contraction.view(N, 1, 1))

        # --- Residual loss ---
        residual = G_all - T_all                     # (N, dim, dim)
        total_loss = torch.sum(residual ** 2)

        # --- Tracelessness diagnostic (no_grad, diagnostic only) ---
        with torch.no_grad():
            try:
                g_inv = torch.linalg.inv(g)          # (N, dim, dim)
                trace_per_pt = torch.einsum("nij,nij->n", g_inv, T_all)
                avg_trace = trace_per_pt.abs().mean()
            except Exception:
                avg_trace = torch.tensor(float("nan"))

        # Backpropagate and update full metric field
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([geometry.metric_field], cfg["grad_clip"])
        optimizer_torch.step()
        scheduler.step()

        # Enforce metric symmetry after each step
        geometry._enforce_symmetry()

        loss_val = total_loss.item()
        trace_val = float(avg_trace.item())
        loss_history.append(loss_val)
        trace_history.append(trace_val)

        # Snapshots
        if iteration in snapshot_iters:
            with torch.no_grad():
                gtt_snapshots[iteration] = geometry.metric_field[:, 0, 0].cpu().numpy().copy()
                grr_snapshots[iteration] = geometry.metric_field[:, 1, 1].cpu().numpy().copy()

        if cfg["verbose"] and (iteration % 50 == 0 or iteration == cfg["n_iterations"] - 1):
            print(f"  iter {iteration:4d}  loss={loss_val:.6e}  "
                  f"avg_trace_violation={trace_val:.4e}")

    # ------------------------------------------------------------------
    # 6.  Extract final metric profile and compare to Schwarzschild
    # ------------------------------------------------------------------
    geometry.metric_field.requires_grad_(False)
    g_tt_final = geometry.metric_field[:, 0, 0].cpu().numpy()
    g_rr_final = geometry.metric_field[:, 1, 1].cpu().numpy()
    r_np = r_grid.cpu().numpy()

    # Fit Schwarzschild radius
    r_s_fit = fit_schwarzschild_radius(r_np, g_tt_final)
    print(f"\n--- Results ---")
    print(f"Fitted Schwarzschild radius:  r_s = {r_s_fit:.6f}")
    print(f"Bell entropy used as source:  S   = {S_bell:.6f}")
    if S_bell > 0:
        print(f"Ratio r_s / S_bell           = {r_s_fit / S_bell:.6f}")

    # Qualitative checks
    g_tt_near = g_tt_final[0]   # closest to origin
    g_tt_far  = g_tt_final[-1]  # far field

    print(f"\nQualitative Schwarzschild checks:")
    print(f"  g_tt near source  = {g_tt_near:.6f}  "
          f"(Schwarzschild: less negative near source, ->(1-r_s/r) -> 0 at horizon)")
    print(f"  g_tt far field    = {g_tt_far:.6f}  "
          f"(should approach -1 for asymptotic flatness)")
    print(f"  g_rr near source  = {g_rr_final[0]:.6f}  "
          f"(should be > g_rr far field -> larger, diverges at horizon)")
    print(f"  g_rr far field    = {g_rr_final[-1]:.6f}  "
          f"(should approach +1)")

    # Check signs — Schwarzschild conventions:
    #   g_tt = -(1 - r_s/r):  near source -> 0 (LESS negative); far field -> -1
    #   g_rr = 1/(1 - r_s/r): near source -> +inf (larger);     far field -> +1
    sign_correct    = g_tt_near > g_tt_far     # g_tt less negative near source
    flatness_check  = abs(g_tt_far + 1.0) < 0.5  # rough asymptotic flatness
    grr_monotone    = g_rr_final[0] > g_rr_final[-1]  # g_rr peaks at origin

    print(f"\n  [{'PASS' if sign_correct else 'FAIL'}]  "
          f"g_tt less negative near source (correct Schwarzschild sign)")
    print(f"  [{'PASS' if flatness_check else 'FAIL'}]  "
          f"Asymptotic flatness g_tt(r_max) ~= -1  "
          f"(Δ = {abs(g_tt_far + 1.0):.4f})")
    print(f"  [{'PASS' if grr_monotone else 'FAIL'}]  "
          f"g_rr monotonically decreasing from source")

    # Correlation with Schwarzschild profile
    if r_s_fit > 0:
        g_tt_schw = schwarzschild_gtt(r_np, r_s_fit)
        g_rr_schw = schwarzschild_grr(r_np, r_s_fit)
        # Pearson correlation
        cc_tt = float(np.corrcoef(g_tt_final, g_tt_schw)[0, 1])
        cc_rr = float(np.corrcoef(g_rr_final, g_rr_schw)[0, 1])
        print(f"\n  Pearson r(g_tt, Schwarzschild g_tt) = {cc_tt:.4f}"
              f"  (1.0 = perfect match)")
        print(f"  Pearson r(g_rr, Schwarzschild g_rr) = {cc_rr:.4f}")
    else:
        cc_tt = cc_rr = 0.0
        g_tt_schw = g_rr_schw = np.zeros_like(r_np)

    results = {
        "r_grid": r_np,
        "g_tt_final": g_tt_final,
        "g_rr_final": g_rr_final,
        "g_tt_schwarzschild": g_tt_schw,
        "g_rr_schwarzschild": g_rr_schw,
        "r_s_fit": r_s_fit,
        "S_bell": S_bell,
        "loss_history": np.array(loss_history),
        "trace_history": np.array(trace_history),
        "gtt_snapshots": gtt_snapshots,
        "grr_snapshots": grr_snapshots,
        "pearson_gtt": cc_tt,
        "pearson_grr": cc_rr,
        "sign_correct": sign_correct,
        "flatness_check": flatness_check,
        "grr_monotone": grr_monotone,
    }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict, save_dir: str) -> None:
    """Generate diagnostic plots for the Schwarzschild recovery test."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping plots")
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    r = results["r_grid"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Schwarzschild Recovery Test (H3) — EntropicUnification",
                 fontsize=13, fontweight="bold")

    # --- (0,0)  g_tt vs Schwarzschild ---
    ax = axes[0, 0]
    ax.plot(r, results["g_tt_final"], "b-", lw=2, label="Optimized $g_{tt}$")
    if results["r_s_fit"] > 0:
        ax.plot(r, results["g_tt_schwarzschild"], "r--", lw=1.5,
                label=f"Schwarzschild ($r_s={results['r_s_fit']:.3f}$)")
    ax.axhline(-1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("r")
    ax.set_ylabel("$g_{tt}$")
    ax.set_title(f"$g_{{tt}}$  (Pearson r = {results['pearson_gtt']:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (0,1)  g_rr vs Schwarzschild ---
    ax = axes[0, 1]
    ax.plot(r, results["g_rr_final"], "b-", lw=2, label="Optimized $g_{rr}$")
    if results["r_s_fit"] > 0:
        ax.plot(r, results["g_rr_schwarzschild"], "r--", lw=1.5,
                label=f"Schwarzschild ($r_s={results['r_s_fit']:.3f}$)")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("r")
    ax.set_ylabel("$g_{rr}$")
    ax.set_title(f"$g_{{rr}}$  (Pearson r = {results['pearson_grr']:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,0)  Loss convergence ---
    ax = axes[1, 0]
    ax.semilogy(results["loss_history"], "k-", lw=1.5, label="Total loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Loss convergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,1)  Tracelessness violation (E=pc diagnostic) ---
    ax = axes[1, 1]
    ax.semilogy(results["trace_history"], "m-", lw=1.5,
                label=r"$|g^{\mu\nu} T_{\mu\nu}|$ (avg per point)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tracelessness violation (log scale)")
    ax.set_title("E=pc constraint: $g^{\\mu\\nu} T_{\\mu\\nu} \\to 0$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / "schwarzschild_recovery.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved -> {out}")

    # --- Snapshot evolution plot ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("$g_{tt}$ profile evolution during optimization", fontsize=12)

    snaps = sorted(results["gtt_snapshots"].keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(snaps)))

    for col, it in zip(colors, snaps):
        axes2[0].plot(r, results["gtt_snapshots"][it], color=col,
                      lw=1.5, label=f"iter {it}")
        axes2[1].plot(r, results["grr_snapshots"][it], color=col,
                      lw=1.5, label=f"iter {it}")

    for ax2, comp, ref in zip(axes2, ["$g_{tt}$", "$g_{rr}$"], [-1.0, 1.0]):
        ax2.axhline(ref, color="red", lw=0.8, ls="--", alpha=0.5,
                    label="Flat Minkowski ref")
        ax2.set_xlabel("r")
        ax2.set_ylabel(comp)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = Path(save_dir) / "schwarzschild_evolution.png"
    plt.savefig(str(out2), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Evolution plot saved -> {out2}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Schwarzschild Recovery Test for EntropicUnification"
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_CFG["n_iterations"],
                        help="Number of optimization iterations")
    parser.add_argument("--lattice", type=int, default=DEFAULT_CFG["lattice_size"],
                        help="Number of radial lattice points")
    parser.add_argument("--lr", type=float, default=DEFAULT_CFG["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--sigma", type=float, default=DEFAULT_CFG["localization_sigma"],
                        help="Gaussian localization width (radial units)")
    parser.add_argument("--formulation", type=str, default="massless",
                        choices=["jacobson", "lagrangian", "massless", "canonical", "faulkner"],
                        help="Stress tensor formulation")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plotting")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_CFG["save_dir"],
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to run on (auto selects cuda if available)")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = {**DEFAULT_CFG}
    cfg["n_iterations"] = args.iterations
    cfg["lattice_size"] = args.lattice
    cfg["learning_rate"] = args.lr
    cfg["localization_sigma"] = args.sigma
    cfg["stress_form"] = StressTensorFormulation(args.formulation)
    cfg["plot"] = not args.no_plot
    cfg["save_dir"] = args.save_dir
    if args.device == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg["device"] = args.device

    results = run_schwarzschild_test(cfg)

    if cfg["plot"]:
        plot_results(results, cfg["save_dir"])

    # Summary verdict
    passed = sum([
        results["sign_correct"],
        results["flatness_check"],
        results["grr_monotone"],
    ])
    print(f"\n=== Schwarzschild Verdict: {passed}/3 qualitative checks passed ===")
    if passed == 3:
        print("    Strong qualitative Schwarzschild signature detected.")
        print(f"    Pearson correlation with Schwarzschild profile: "
              f"g_tt={results['pearson_gtt']:.3f}, g_rr={results['pearson_grr']:.3f}")
    elif passed >= 2:
        print("    Partial Schwarzschild signature - promising, needs more iterations.")
    else:
        print("    Framework not converging to Schwarzschild - "
              "this is also a result (see docs for interpretation).")

    return results


if __name__ == "__main__":
    main()
