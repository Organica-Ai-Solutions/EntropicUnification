# EntropicUnification

<p align="center">
  <img src="docs/images/entropic.jpg" alt="EntropicUnification" width="600"/>
</p>

<p align="center">
  <b>A differentiable computational framework for learning spacetime geometry from quantum entanglement entropy</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.3-blue" />
  <img src="https://img.shields.io/badge/python-3.9%2B-green" />
  <img src="https://img.shields.io/badge/framework-PyTorch%20%7C%20PennyLane-orange" />
  <img src="https://img.shields.io/badge/status-active%20research-purple" />
  <img src="https://img.shields.io/badge/part%20of-NIS%20Protocol%20Ecosystem-teal" />
</p>

---

> *"The universe doesn't obey mathematical laws — it computes them."*

---

## What This Is

EntropicUnification is a **differentiable toy-model sandbox** inspired by a fundamental question: can spacetime geometry emerge from quantum information?

The framework implements the conjecture that entanglement entropy gradients source spacetime curvature via an entropic stress tensor $T^{(\text{ent})}_{\mu\nu}$ built from $\nabla_\mu S$, where $S(x)$ is an entanglement entropy field computed from partial traces of an actual quantum state. The stress tensor's *form* follows from Hilbert variation of a scalar-field action (see [Theoretical Foundation](#theoretical-foundation)); whether entanglement entropy actually behaves as such a field is the conjecture under test, not a derived fact.

The pipeline is: quantum state → entropy field $S(r)$ (partial trace at every cut radius) → spatial gradient $\nabla S$ → stress tensor → metric optimization against the Einstein tensor on a 1-D lattice. It is a research testbed for exploring this idea numerically — **not** a validated theory of quantum gravity, and not evidence for one.

---

## Results Status (v1.3)

**All quantitative results previously reported here (v1.2 and earlier) are withdrawn.**

The v1.2 implementation had defects that invalidated its headline numbers:

1. **The spatial entropy profile was inserted by hand.** The "entropy gradient" was a gradient with respect to quantum state *amplitudes*, relabeled as spacetime components and multiplied by a hand-placed Gaussian $w(r)$. The recovered $g_{tt}$ well was therefore a restatement of the input Gaussian, not an emergent property of entanglement. The Schwarzschild fits, the $r_s/S$ ratios, and the entanglement-scaling curves all inherited this circularity.
2. **Curvature was not GR curvature.** Christoffel symbols copied the lattice derivative into *every* coordinate slot; finite differences omitted the $1/dx$ normalization; and Riemann-tensor symmetries were projected onto the mixed-index tensor (where they do not hold), silently corrupting the result.
3. **Contractions used the Euclidean dot product** instead of $g^{\mu\nu}\partial_\mu S\,\partial_\nu S$, breaking the trace identities the MASSLESS formulation depends on.

v1.3 fixes all of the above:

- $S(r)$ is now computed **from the quantum state**: qubits are placed at radial positions, and $S(r)$ is the entanglement entropy of the qubits inside radius $r$, obtained by partial trace at every lattice point. For a pure state $S(r)$ vanishes below the innermost and above the outermost qubit — any localized bump is emergent, not assumed.
- Christoffel, Riemann, Ricci, and Einstein tensors are computed honestly for the declared metric ansatz (static, varying along the single lattice coordinate), with dx-normalized derivatives and **no symmetry projection** — `riemann_identity_violations()` reports identity violations instead of hiding them.
- All contractions use the inverse metric; the MASSLESS stress tensor is traceless exactly, by construction.
- The FAULKNER formulation uses the real spatial Hessian $\partial_r^2 S$, not an outer-product surrogate.

Re-run experiments with:

```bash
python examples/schwarzschild_test.py --iterations 1000 --lattice 64
python examples/scaling_experiment.py --iterations 1000
```

and report whatever they produce — including a null result. A negative outcome under the honest implementation is a meaningful data point about the conjecture in this toy setting.

### First v1.3 runs (July 2026, CPU, honest pipeline)

**Schwarzschild test** (1000 iterations, lattice 64, MASSLESS): 2/3 qualitative checks pass — $g_{tt}$ has the correct sign structure (less negative toward the source) and rough asymptotic flatness, but $g_{rr}$ moves the *wrong* way. Pearson correlation with a fitted Schwarzschild profile: $r(g_{tt}) = 0.50$, $r(g_{rr}) = 0.02$. The v1.2 claim of 0.78 does not survive the corrections. Tracelessness violation is now at machine precision (~1e-17), as it should be by construction.

**Entanglement scaling** (1000 iterations, lattice 32): no clean $r_s \propto S$ relation — a through-origin linear fit gives negative $R^2$. Short runs (50 iterations) look linear, but the relationship does not stabilize with convergence. The v1.2 "genuinely non-linear, monotonically decreasing ratio" claim also does not reproduce.

In short: **the current honest 1+1D toy shows a weak, partial signature at best.** This is the real starting point. (Recall that in 1+1D the continuum Einstein tensor vanishes identically — a meaningful H3 test needs the framework extended to ≥ 3+1D first.)

---

## Theoretical Foundation

### The Stress Tensor Ansatz

*If* one postulates that the entropy field contributes to the action like a massless scalar,

$$S = \int \sqrt{-g} \left[ \frac{R}{16\pi G} - \frac{\hbar}{4\pi} (\nabla S)^2 \right] d^n x$$

then Hilbert variation with respect to $g^{\mu\nu}$ yields:

$$T^{(\text{ent})}_{\mu\nu} = \frac{\hbar}{2\pi} \left[ \nabla_\mu S \, \nabla_\nu S - \frac{1}{2} g_{\mu\nu} (\nabla S)^2 \right]$$

The variation itself is standard scalar-field algebra. What is *not* derived — and is the actual conjecture this sandbox explores — is the premise that entanglement entropy enters the gravitational action this way at all. As of v1.3, $\nabla_\mu S$ in the code is a genuine spacetime derivative of an entropy field computed from partial traces of a quantum state, so the implementation at least matches the equation being tested.

### Massless Constraint (E = pc)

Entanglement entropy is pure information — it propagates at $c$ with no rest mass. This imposes tracelessness on $T^{(\text{ent})}_{\mu\nu}$:

$$g^{\mu\nu} T^{(\text{ent})}_{\mu\nu} = 0$$

The **MASSLESS formulation** enforces this exactly by replacing $\frac{1}{2}$ with $\frac{1}{n}$ (valid in any dimension):

$$T^{(\text{ent})}_{\mu\nu} = \frac{\hbar}{2\pi} \left[ \nabla_\mu S \, \nabla_\nu S - \frac{1}{n} g_{\mu\nu} (\nabla S)^2 \right]$$

A tracelessness diagnostic runs automatically every simulation. Zero = massless field satisfied.

### Three Stress Tensor Formulations

| Formulation | Basis | Traceless | Use Case |
|---|---|---|---|
| `LAGRANGIAN` | Hilbert variation of covariant action | No (massive analog) | Baseline derivation |
| `MASSLESS` | Lagrangian + E=pc constraint ($1/n$) | Yes | Default — physically motivated |
| `FAULKNER` | Linearized Einstein from Hessian: $\nabla_\mu\nabla_\nu S - (\Box S)g_{\mu\nu}$ | Yes | Closest to Faulkner (2013) |

---

## Scientific Framework

Three domains connected by a single differentiable pipeline:

```
Quantum Information ──► Thermodynamics ──► Geometry ──► Learning Dynamics
     (ψ, S_ent)            (∇S, T_μν)       (G_μν, g_μν)     (∂ℒ/∂g_μν)
```

**Quantum Information → Thermodynamics**: Entanglement entropy computed via von Neumann formula from PennyLane quantum circuits.

**Thermodynamics → Geometry**: Entropy gradients map to spacetime curvature through the entropic field equation $G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G \, T^{(\text{ent})}_{\mu\nu}$.

**Geometry → Learning**: The metric tensor is optimized via PyTorch autograd to minimize inconsistency between geometric curvature and entropic flow. The universe as optimizer.

---

## Three Experimental Hypotheses

| | Hypothesis | Status |
|---|---|---|
| H1 | Higher entanglement → larger curvature | ⬜ Open — prior "confirmation" used the invalidated v1.2 pipeline |
| H2 | Optimization converges to modified Einstein equations | ⬜ Open — must be re-run on v1.3 |
| H3 | Localized entanglement source recovers Schwarzschild metric | ⬜ Open — v1.2 result was circular (hand-placed Gaussian source) |

---

## Architecture

```
EntropicUnification/
├── core/
│   ├── quantum_engine.py       # Quantum state evolution ψ(t) — O(2ⁿ) partial trace
│   ├── geometry_engine.py      # Spacetime metric, Christoffel, Riemann, Einstein tensors
│   ├── entropy_module.py       # Von Neumann entropy, RT geodesic integral, entropy flow
│   ├── coupling_layer.py       # T_μν formulations (LAGRANGIAN / MASSLESS / FAULKNER)
│   ├── loss_functions.py       # Einstein constraint, entropy flow, regularity
│   ├── optimizer.py            # Training loop, convergence, checkpoints
│   ├── advanced_optimizer.py   # Basin hopping, simulated annealing, adaptive LR
│   └── utils/
│       └── finite_difference.py  # dx-normalized finite difference (1st and 2nd order)
│
├── examples/
│   ├── schwarzschild_test.py   # H3: GHZ entropy field S(r) → metric optimization
│   ├── scaling_experiment.py   # r_s vs S_ent Bekenstein-Hawking scaling sweep
│   ├── entropic_simulation.py  # Full simulation pipeline
│   ├── compare_stress_tensors.py  # Formulation comparison
│   └── test_original_geometry.py
│
├── dashboards/
│   ├── enhanced_app.py         # Interactive Dash dashboard
│   └── run_fixed_dashboard.py  # Port-conflict-safe launcher
│
├── notebooks/
│   └── experiments.ipynb       # Interactive visualization
│
├── WHITEPAPER.md               # Full theoretical treatment
├── QUICKSTART.md               # Up and running in 10 minutes
└── README.md                   # This file
```

---

## Quick Start

```bash
git clone https://github.com/Organica-Ai-Solutions/EntropicUnification.git
cd EntropicUnification
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Run the Schwarzschild test:**
```bash
python examples/schwarzschild_test.py
# With options:
python examples/schwarzschild_test.py --iterations 500 --lattice 64 --formulation massless
```

**Compare all three stress tensor formulations:**
```bash
python examples/compare_stress_tensors.py
```

**Launch the interactive dashboard:**
```bash
python dashboards/run_fixed_dashboard.py
```

**Run a full simulation:**
```bash
python entropic_unification.py
```

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| PyTorch | ≥ 2.0.0 | Autograd, tensor ops, metric optimization |
| PennyLane | ≥ 0.30.0 | Quantum circuit simulation |
| NumPy | ≥ 1.21.0 | Numerical computations |
| SciPy | ≥ 1.7.0 | Scientific algorithms |
| Matplotlib | ≥ 3.4.0 | Visualization |
| Dash / Plotly | latest | Interactive dashboard |
| NetworkX | latest | Graph-based entropy calculations |

---

## Theoretical Context

This framework sits at the intersection of four established research programs:

**Ryu-Takayanagi (2006)** — Entanglement entropy equals minimal surface area in AdS/CFT: $S_A = \text{Area}(\gamma_A) / 4G_N\hbar$. EntropicUnification implements this as a proper geodesic integral on the lattice.

**Jacobson (1995)** — Einstein equations derived from thermodynamic principles applied to local Rindler horizons. EntropicUnification provides a computational realization of this derivation.

**Van Raamsdonk / Maldacena (2010–2013)** — Quantum entanglement between boundary regions is responsible for the connectedness of bulk spacetime (ER = EPR).

**Faulkner et al. (2013)** — Linearized Einstein equations from entanglement, arXiv:1312.7856. The FAULKNER formulation in `coupling_layer.py` implements $T_{\mu\nu} = \frac{\hbar}{2\pi}[\nabla_\mu\nabla_\nu S - (\Box S)g_{\mu\nu}]$ via second-order autograd.

**Bianconi (2025)** — Independent derivation of gravity from quantum relative entropy (Phys. Rev. D). Converges on similar conclusions from a pure theory direction.

---

## Honest Caveats

This is a research testbed, not a validated theory of quantum gravity.

- **The central premise is a conjecture.** Treating entanglement entropy as a massless scalar field in the gravitational action is postulated, not derived. The Hilbert variation only tells you what stress tensor that postulate implies.
- **The entropy field depends on the qubit placement.** S(r) is genuinely computed from the state, but where the qubits sit on the lattice is a modeling choice, and the mapping from Hilbert space partitions to spatial regions has no canonical form.
- **The geometry is a 1+1D toy.** The metric varies along a single coordinate; in 2D the Einstein tensor vanishes identically in the continuum, so any structure in G_munu here is discretization effect plus gauge. Interpret 2D "recoveries" accordingly. Higher-dimensional runs are needed for any physical claim.
- **S(r) for few qubits is piecewise constant.** The "linear" interpolation between qubit positions is a declared discretization choice; with 4 qubits the gradient structure is coarse. More qubits give a smoother, more meaningful field.
- **`holographic_entropy()`** implements the RT geodesic integral in 1+1D only — full minimal surface solvers for higher dimensions are future work.
- **Results are sensitive** to lattice size, iteration count, learning rate, and qubit cluster geometry.
- The legacy `compute_entropy_stress_tensor()` state-space-projection path is retained for backward compatibility but emits a warning and should not be used for results.

These limitations are tracked and documented. The framework is intended to be honest about what it does and does not demonstrate.

---

## Connection to NIS Protocol

EntropicUnification is the **fundamental physics layer** of the [NIS Protocol](https://github.com/Organica-Ai-Solutions/NIS_Protocol) ecosystem.

Where NIS Protocol implements cognitive intelligence — multi-agent reasoning, memory, action — EntropicUnification investigates the informational substrate beneath physical reality. Both share a core architectural principle: intelligence and physics as optimization processes over information structures.

The long-term vision: NIS agents grounded in physics that is itself grounded in information theory, all the way down.

---

## Roadmap

- [x] Stress tensor ansatz from Hilbert variation (LAGRANGIAN / MASSLESS / FAULKNER)
- [x] Entropy field $S(r)$ computed from partial traces of the actual state (v1.3)
- [x] Honest curvature pipeline: dx-normalized derivatives, single-coordinate ansatz, no symmetry projection (v1.3)
- [x] Exact tracelessness of MASSLESS form via $g^{\mu\nu}$ contraction (v1.3)
- [x] Real spatial Hessian for FAULKNER formulation (v1.3)
- [x] O(2ⁿ) partial trace via tensor reshape
- [ ] Re-run H1–H3 on the v1.3 pipeline and publish results (positive or null)
- [ ] Move beyond 1+1D — in 2D the continuum Einstein tensor vanishes identically, so H3 needs ≥ 3+1D to be meaningful
- [ ] Many-qubit chains (8–12) for smoother $S(r)$ profiles
- [ ] Real quantum hardware integration (IBM Quantum / IonQ)
- [ ] Cosmological simulations — early universe dynamics
- [ ] Black hole information paradox testbed
- [ ] Higher curvature corrections (Gauss-Bonnet)

---

## Citation

```bibtex
@software{entropicunification2025,
  title     = {EntropicUnification: A Differentiable Framework for Learning
               Spacetime Geometry from Quantum Entanglement},
  author    = {Organica AI Solutions},
  year      = {2025},
  version   = {1.3},
  url       = {https://github.com/Organica-Ai-Solutions/EntropicUnification},
  note      = {Part of the NIS Protocol ecosystem}
}
```

---

## References

1. Ryu, S. & Takayanagi, T. (2006). Holographic derivation of entanglement entropy from AdS/CFT. *Phys. Rev. Lett.* 96, 181602.
2. Jacobson, T. (1995). Thermodynamics of spacetime: the Einstein equation of state. *Phys. Rev. Lett.* 75, 1260.
3. Van Raamsdonk, M. (2010). Building up spacetime with quantum entanglement. *Gen. Rel. Grav.* 42, 2323.
4. Faulkner, T. et al. (2013). Gravitation from entanglement in holographic CFTs. arXiv:1312.7856.
5. Maldacena, J. & Susskind, L. (2013). Cool horizons for entangled black holes. *Fortsch. Phys.* 61, 781.
6. Bianconi, G. (2025). Gravity from entropy. *Phys. Rev. D.*
7. Wheeler, J.A. (1990). Information, physics, quantum: the search for links. In *Complexity, Entropy, and the Physics of Information.*

---

*Version 1.3 — July 2026*
*Organica AI Solutions — [organicaai.com](https://organicaai.com)*
