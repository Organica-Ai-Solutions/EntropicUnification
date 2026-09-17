# EntropicUnification — Progress Report

**Status as of v1.4 (September 2026).** This document was rewritten from
scratch at v1.4: every prior version described the v1.1/v1.2 codebase and
listed as achievements several things v1.3 deliberately *removed*. If you are
looking for the old text, it is in git history — treat it as a record of what
we believed, not of what the code does.

## What the framework is

A differentiable toy model asking whether spacetime geometry can emerge from
entanglement entropy. The pipeline:

```
quantum state  ->  S(r) by partial trace  ->  ∇S  ->  T_μν  ->  metric fit vs G_μν
```

It is a numerical sandbox on a 1-D lattice, not a theory and not evidence for
one.

## The v1.2 retraction (the central fact about this project)

All quantitative results from v1.2 and earlier are **withdrawn**. Three
independent defects:

1. **The entropy field's spatial structure was inserted by hand.** The
   "entropy gradient" was a gradient with respect to quantum state
   *amplitudes*, relabelled as spacetime components and multiplied by a
   hand-placed Gaussian `w(r)`. The recovered `g_tt` well restated that
   Gaussian. Every downstream number — the Schwarzschild fits, the `r_s/S`
   ratios, the scaling curves — inherited the circularity.
2. **Curvature was not GR curvature.** Christoffel symbols copied one lattice
   derivative into every coordinate slot; finite differences omitted the
   `1/dx` normalisation; Riemann symmetries were *projected onto the
   mixed-index tensor*, where they do not hold.
3. **Contractions used the Euclidean dot product** rather than
   `g^μν ∂_μS ∂_νS`, breaking the trace identities MASSLESS depends on.

The most instructive part: **none of the diagnostics caught any of it.**
Defect 2 made the identity checks read *better*, because the symmetries were
being imposed rather than measured.

## What v1.3 fixed (July 2026)

- `S(r)` computed from partial traces of the actual state at every cut radius
  (`EntropyModule.entropy_profile`). For a pure state `S(r)` vanishes below the
  innermost and above the outermost qubit, so any bump is emergent.
- `CouplingLayer.compute_stress_tensor_field` builds `T_μν` from the genuine
  spatial derivative of `S(r)` with `g^μν` contractions. MASSLESS is traceless
  to machine precision; FAULKNER uses the real spatial Hessian.
- Christoffel/Riemann/Ricci implement the declared static single-coordinate
  ansatz. Finite differences are `dx`-normalised throughout.
- Symmetry projection and Bianchi "enforcement" **removed**; replaced by
  `riemann_identity_violations()`, which measures and reports.
- Weyl/Gauss-Bonnet computed on the properly lowered Riemann tensor.

## What v1.4 adds (September 2026): gates, not warnings

v1.3 fixed the physics but left every consistency check *advisory* — a number
printed for a human to notice. That is the same failure mode that let v1.2
ship. v1.4 introduces `core/validation.py`, which turns them into gates that
**abort the run**:

| gate | catches |
|---|---|
| `check_entropy_provenance` | an entropy field whose structure was inserted rather than derived — **the v1.2 defect, at its source** |
| `check_entropy_field_sanity` | constant or negative `S(x)`: no source, nothing to test |
| `check_tracelessness` | a wrong contraction, relative to `‖T‖`, for the formulations that are traceless by construction |
| `check_riemann_identities` | curvature violating the identities by more than discretization explains |
| `check_metric_resolved` | a metric varying on the scale of the lattice — no continuum limit |
| `check_finite` | NaN/Inf reaching the loss — a diverged run that would still print plots |
| `check_meaningful_dimension` | a run claiming a curvature result in 2D, where `G_μν ≡ 0` identically |

Provenance is carried by the data: `EntropyModule.entropy_field()` returns an
`EntropyField` recording that every value came from a partial trace of a named
state. `CouplingLayer` **refuses a bare tensor** — not because a tensor is
necessarily wrong, but because nothing about it records whether its spatial
structure was computed or hand-placed, and that distinction is the whole
lesson of v1.2.

Any gate can be relaxed (`GateConfig(strict=False)`, or a named `allow_*`
flag). Relaxing one is a deliberate, visible act. That is the design.

## Independent review of v1.4

The gating layer was reviewed by a separate agent with no stake in it, and the
first pass found the gates were wired at the wrong point: both experiments
validated only the **initial** metric, which is Minkowski — exactly flat, so
truncation is zero, identity violations are zero, and the curvature gates
could not fail. Good machinery aimed at the one input incapable of tripping
it. Also found: the geometry caches were keyed by name only, so passing a
metric argument with a warm cache mixed tensors from two different metrics;
`EntropyField` was forgeable in one line and mutable after construction; and
the legacy v1.2 state-space-gradient path was still fully reachable and
completely ungated.

All of those are fixed. The review's remaining findings are recorded as known
limitations rather than quietly dropped:

| limitation | status |
|---|---|
| missing `1/dx` is invisible to the gates (relative violations are scale-invariant) | documented; covered by the derivative tests instead |
| `check_metric_resolved` passes noise below ~0.8% of \|g\| | documented; `converges_under_refinement()` added as the real test, not wired in by default (needs two resolutions) |
| provenance is attestation + tamper-evidence, not proof | documented |
| FAULKNER used the coordinate Hessian, not the covariant one | **fixed in v1.4** — the Christoffel term is included; flat metrics agree exactly, curved ones differ by ~0.2%, and `grad_0 grad_0 S` (previously forced to zero) is nonzero |

## v1.3 results withdrawn (v1.4.2)

The v1.3 numbers below are **withdrawn**. On the first experiment the v1.4
gates were pointed at, the optimised metric failed `check_metric_resolved`
(eps = 1.20e-01 vs 2e-2), and a refinement sweep
(`scripts/convergence_test.py`) put the convergence ratio at **1.45** where a
resolved field needs ~4 — so the field has no continuum limit and its
curvature approximates nothing. Cause: the experiment uses a bare pointwise
residual loss with no neighbour coupling. **Adding that coupling does not fix
it** — `scripts/smoothness_sweep.py` sweeps a second-difference penalty over
five orders of magnitude and no value reaches a refinement ratio of 2; above
λ≈1 the ratios invert, because the useful λ becomes a function of lattice
size. So the problem is not merely a misconfigured experiment: pointwise
optimisation of a metric against a curvature residual does not produce a field
with a continuum limit in this setup.

H1, H2 and H3 were already open. They remain open, and the framework now has
**no quantitative result of its own that has passed its own gates.** That is
the accurate status.

## Withdrawn v1.3 results (kept for the record)

**Schwarzschild test** (1000 iterations, lattice 64, MASSLESS): 2/3 qualitative
checks. `g_tt` has the right sign structure and rough asymptotic flatness;
`g_rr` moves the wrong way. Pearson `r(g_tt) = 0.50`, `r(g_rr) = 0.02`. The
v1.2 claim of 0.78 does not survive. Tracelessness violation ~1e-17.

**Entanglement scaling** (1000 iterations, lattice 32): no clean `r_s ∝ S`
relation; a through-origin linear fit gives negative R². Short runs look
linear but the relationship does not survive convergence.

**H1, H2, H3: all open.** Prior confirmations used the invalidated pipeline.

## The limitation that dominates everything else

The framework runs in **1+1D, where the continuum Einstein tensor vanishes
identically** — `G_μν ≡ 0` for every metric in two dimensions. The optimiser is
fitting a target that is structurally zero. No amount of numerical care fixes
this; H3 is not merely unconfirmed here, it is **untestable** here.

`check_meaningful_dimension` exists to make this impossible to forget, and is
off by default only because turning it on would (correctly) fail every run the
framework can currently perform.

## Next steps, in order of honesty-weighted value

0. ~~Find a parameterisation with a continuum limit by construction.~~
   **Done** (`scripts/parameterisation_test.py`): a Chebyshev-coefficient
   metric converges at exactly second order — ratios 4.01 / 3.96 / 3.81 at
   k = 4 / 8 / 16 with the optimiser held fixed. The representation is no
   longer the blocker.
1. **Port `schwarzschild_test.py` and `scaling_experiment.py` onto the
   Chebyshev basis and re-run.** This is now the immediate next step: the
   framework has a configuration with a continuum limit but has never been
   run in it, so it still has no number that passes its own gates. Report the
   fit loss alongside eps — k=4 passes the smoothness gate most cleanly while
   fitting G = T about 25x worse than k=16, and a smooth metric that does not
   satisfy the field equation recovers nothing.
1. **≥3+1D.** Everything else is secondary. Until then no result here bears on
   the conjecture.
2. **Make dimension an output, not an input.** A framework claiming geometry
   emerges from entanglement currently *assumes* the lattice, the coordinate
   and the dimension. A tensor-network formulation, where geometry is read off
   network connectivity, would not.
3. **A benchmark with a known answer.** "Recover Schwarzschild" is loosely
   posed, which is how v1.2 slid into circularity. The fluid/gravity
   correspondence supplies cases where the dual metric is derivable
   independently — a target that can actually be missed.
4. Many-qubit chains (8–12) for smoother `S(r)`.
5. GPU/sparse partial traces; the O(2ⁿ) trace is the scaling wall.

Real quantum hardware and cosmological simulations remain on the roadmap and
remain, at this stage, premature.
