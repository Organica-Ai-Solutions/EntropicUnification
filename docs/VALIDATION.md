# Pipeline validation

## Why this exists

v1.2 of this project reported a Schwarzschild recovery with a Pearson
correlation of 0.78. The number was an artifact: the "entropy field" had a
Gaussian profile placed into it by hand, so the metric well the optimiser
found was a restatement of the input, not an emergent property of
entanglement.

Nothing in the code objected. Every consistency check was **advisory** — a
float printed to stdout that a human had to read, interpret, and act on. Worse,
one defect actively suppressed its own detection: Riemann symmetries were
*projected onto* the mixed-index tensor (where they do not hold), so the
identity diagnostics reported near-zero violations precisely because the
curvature was being corrupted.

The lesson is not "be more careful." It is that a check nobody is forced to
read is not a check. `core/validation.py` turns them into gates that abort.

## A worked example of why the rule matters

The boundary-stencil bug fixed in v1.4.4 was found by refusing to loosen a
tolerance. The Chebyshev experiment failed `check_riemann_identities` at ratio
5.23 against a 4.0 threshold, and the obvious diagnosis — "my tolerance was
calibrated on single-mode sinusoids and is too tight here" — was both
plausible and, as it turns out, partly true.

Instead of acting on it, the criterion was fixed **in advance**: a truncation
error gives a ratio that is *constant* under refinement whatever its value; a
ratio that *grows* means a genuine defect. It grew — ~1.4x per lattice
doubling across every mode count, while the single-sinusoid reference stayed
pinned at 1.42.

That pointed at `core/utils/finite_difference.py`, where the boundary used
two-point one-sided differences (O(dx)) against an O(dx²) interior. The
lower order dominated the global error norm wherever the field varied near
the boundary, degrading the whole scheme to roughly O(dx^1.5). Chebyshev
polynomials concentrate their variation at the endpoints, which is why they
exposed it and a sinusoid never did.

Two lessons, both uncomfortable:

1. **The convenient explanation was available and wrong-ish.** Raising the
   tolerance would have made the run pass and left a bug that had been
   costing half an order of accuracy in every curvature number this project
   has ever produced — including the ones already withdrawn.
2. **The calibration set was too narrow.** One function family, chosen
   without asking where it stresses the code. A sinusoid on [0,1] barely
   varies at the endpoints, so a first-order edge stencil was invisible to it.

## The rule

> A defect that would invalidate a published number must **raise**, not warn.

Everything else follows from that.

## Gates

### `check_entropy_provenance` — the v1.2 gate

An entropy field is admissible as physics only if it was computed from partial
traces of an actual quantum state. This is enforced by making the data carry
the claim:

```python
S = entropy_module.entropy_field(ghz, qubit_positions, r_grid)
S.provenance          # partial_trace [derived] (num_qubits=6, state_norm=1.0, ...)
```

`CouplingLayer.compute_stress_tensor_field` **refuses a bare tensor**:

```python
coupling.compute_stress_tensor_field(S.values)   # ProvenanceError
coupling.compute_stress_tensor_field(S)          # fine
```

A raw tensor is not rejected because tensors are wrong. It is rejected because
nothing about a tensor records whether its spatial structure was *computed* or
*inserted*, and that distinction is the entire difference between v1.2 and
v1.3. Fields that are knowingly not derived are constructed explicitly and
still gated:

```python
EntropyField.unverified(profile, "analytic")       # honest, still refused by default
EntropyField.unverified(profile, "partial_trace")  # ValueError: cannot fake derivation
EntropyField(profile, Provenance("partial_trace")) # ProvenanceError: needs the state
```

`EntropyField` is frozen, and its constructor rejects a derived provenance
supplied directly — only `from_partial_traces` (which requires the state) can
mint one. A `values_digest` recorded at construction is re-checked by the gate,
so swapping the numbers afterwards is detected.

**What this is and is not.** It is an attestation about how the object was
built, plus a tamper-evidence check on the values. It is *not* a proof that the
numbers are entanglement entropies — a sufficiently determined caller can still
reach into the module internals. The goal is to make the v1.2 mistake
impossible to make *by accident* and impossible to make *invisibly*, not to
defeat an adversary.

### `check_trace_identity` and `check_tracelessness`

Every stress tensor formulation has a trace fixed by its own algebra:

| formulation | predicted `g^uv T_uv` |
|---|---|
| MASSLESS | `0` — the `1/n` coefficient makes it traceless in any dimension |
| FAULKNER | `(1-n) BoxS` — **not** traceless |
| LAGRANGIAN, CANONICAL, MODIFIED | no fixed value; gate reports as skipped |

Checking the *predicted* value is strictly stronger than checking for zero,
and it catches the same defect: v1.2 used a Euclidean dot product where
`g^uv` belongs, which breaks the trace identity whatever that identity is.
Measured relative to `||T||`, so the check is scale-free. Exact float64
algebra lands at ~1e-17; the tolerance is 1e-10.

**This gate found a real error on its first run.** The README described
FAULKNER as traceless while printing, in the same table row, the formula
`grad_u grad_v S - (BoxS) g_uv` — whose trace is `(1-n) BoxS`, zero only in
`n=1`. The implementation was correct; the documentation was not. Measured
against the prediction, the match is exact:

| dimension | max abs trace | predicted `(1-n) BoxS` | relative mismatch |
|---|---|---|---|
| n=2 | 1.103e-1 | 1.103e-1 | 0.00 |
| n=4 | 3.310e-1 | 3.310e-1 | 5.2e-18 |

**Resolved in v1.4.** This gate originally verified only internal algebraic
consistency: the implemented Hessian was the *coordinate* second derivative
`d2S/dr2`, not the covariant

    grad_mu grad_nu S = d_mu d_nu S - Gamma^lambda_{mu nu} d_lambda S

and `faulkner_trace` was built from the same `box_S` as `T`, so the two agreed
by construction whatever the Hessian was. The Christoffel term is now
included. Measured consequences:

| check | dim 2 | dim 4 |
|---|---|---|
| flat metric: covariant vs coordinate | 0.000e+00 | 0.000e+00 |
| curved metric: relative difference | 2.32e-3 | 2.04e-3 |
| `grad_0 grad_0 S` (forced to zero before) | 1.24e-3 | 1.24e-3 |
| trace vs `(1-n) BoxS` | 1.34e-16 | 1.79e-16 |

The two agree exactly where `Gamma` vanishes, differ where it does not, and
the trace identity survives the correction. `CouplingLayer(...,
covariant_hessian=False)` restores the old behaviour for comparison.

Note what did *not* happen: the formulation was not quietly switched to a
traceless variant to make the gate pass. Changing `(BoxS) g_uv` to
`(1/n)(BoxS) g_uv` would be a physics decision about which tensor to test,
not a bug fix, and it would have destroyed the evidence that the docs and the
code disagreed.

### `check_riemann_identities`

The **lowered** Riemann tensor must satisfy antisymmetry in each index pair,
pair-exchange symmetry, and the first Bianchi identity. Measured by
`GeometryEngine.riemann_identity_violations()` and gated here — never imposed.

On a lattice these identities are not exact: they hold up to second-order
finite-difference error, so an absolute threshold is meaningless on its own —
it is too tight on a coarse lattice and too loose on a fine one. What is
meaningful is the **ratio** of the measured violation to the truncation error
the metric's own smoothness predicts. A correct implementation sits near 1 at
every resolution; a wrong one does not, however fine the lattice.

### `check_metric_resolved`

The lattice must actually resolve the metric's variation. The gate quantity is
the dimensionless `eps = ||d2g|| dx^2 / ||g||`, i.e. `(dx/lambda)^2` for a
metric varying on length `lambda`.

This is a *different* failure from the one above. A field of per-site noise
has no continuum limit at all — there is no metric there to take the curvature
of, and every derived tensor is meaningless no matter how carefully computed.
Such a field produces identity violations that are perfectly "consistent with
truncation error" (ratio ≈ 1.1), so the Riemann gate waves it through. This
gate is what stops it.

### `check_finite`

No NaN or Inf may enter the loss. A diverged optimiser otherwise runs to
completion, writes a loss history, and renders plots — all of NaN.

### `check_entropy_field_sanity`

`S(x)` must be non-negative (von Neumann entropy cannot be negative) and not
constant. A constant field has no gradient, hence no source; any curvature the
optimiser then finds is fitting numerical noise.

### `check_meaningful_dimension`

In two dimensions the Einstein tensor **vanishes identically** — `G_μν ≡ 0` for
every metric, not approximately, not usually. A 1+1D run is fitting a target
that is structurally zero, so it cannot test whether entanglement sources
curvature.

This gate is **off by default**, because switching it on fails every run the
framework can currently perform. That is an accurate description of the
project's status, and the gate exists so the fact cannot be quietly forgotten.
Turning it on is how a run asserts it is meant to be physically conclusive.

## Calibration

Tolerances are only meaningful if they were measured rather than guessed —
and the measurement has to be re-runnable, so it lives in the tree:

```bash
venv/bin/python scripts/calibrate_gates.py
```

Every table below is that script's output.

### `traceless_tol = 1e-10` — principled

The MASSLESS trace is exactly zero in exact arithmetic; observed float64
values sit at ~1e-17. A real failure is a bug by many orders of magnitude, so
the threshold's precise value is not load-bearing.

### `riemann_error_ratio = 4.0` — measured, and spectrum-dependent

Ratio of the worst identity violation to the predicted truncation error
`eps`, for a smooth metric (flat + a long-wavelength sinusoid):

| case | N=16 | N=32 | N=64 | N=128 | N=256 |
|---|---|---|---|---|---|
| dim 2, amp 0.01 | 1.19 | 1.35 | 1.39 | 1.40 | 1.41 |
| dim 2, amp 0.1  | 1.19 | 1.35 | 1.39 | 1.40 | 1.41 |
| dim 4, amp 0.01 | 0.69 | 0.78 | 0.80 | 0.81 | 0.81 |
| dim 4, amp 0.1  | 0.68 | 0.77 | 0.80 | 0.81 | 0.81 |

Absolute violations fall 3.80x, 4.03x, 4.04x, 4.02x per doubling — clean
second order, with a flat metric giving identically zero at every resolution.
Observed range 0.68–1.41, so 4.0 leaves roughly 3x margin.

**These numbers are post-v1.4.4.** Before the boundary-stencil fix the ratios
read 0.82–1.45 and were flat from N=16; they now rise to the same asymptote
from below, which is the healthier signature — coarse-lattice error dies off
instead of being masked by a first-order edge term.

**The ratio depends on the metric's spectral content, and 4.0 does not cover
every case.** `eps` is built from a *second* difference, while the Riemann
identity violation also involves higher derivatives. A metric with more
high-frequency content therefore produces a larger violation per unit of
proxy — legitimately. Measured on Chebyshev metrics of `k` modes, after the
boundary fix:

| modes | N=32 | N=64 | N=128 | N=256 | N=512 |
|---|---|---|---|---|---|
| 4 | 2.75 | 2.54 | 2.38 | 2.28 | 2.23 |
| 8 | 2.91 | 3.33 | 3.31 | 3.13 | 2.95 |
| 16 | 2.48 | 8.95 | 11.69 | 11.99 | 11.29 |

k=4 and k=8 sit under 4.0; **k=16 plateaus near 11.5 and would fail the gate
despite being perfectly well resolved.** A plateau is the signature of
truncation error with a large coefficient, not of a defect.

So for high-mode metrics the absolute threshold is the wrong instrument, and
`converges_under_refinement` — which asks whether the ratio is *constant*
rather than whether it is small — is the authoritative test. The default is
deliberately **not** raised to accommodate k=16: loosening a tolerance to make
a run pass is how v1.2's checks became decorative. Raise it consciously, per
run, with the mode count recorded.

### `metric_resolution_tol = 2e-2` — measured, and the weakest gate here

Per-site Gaussian noise added to a flat metric, N=64, dim 2:

| noise amp | eps | violation | ratio | verdict |
|---|---|---|---|---|
| 0.0001 | 2.56e-4 | 2.59e-4 | 1.01 | **passes** |
| 0.001  | 2.56e-3 | 2.59e-3 | 1.01 | **passes** |
| 0.01   | 2.55e-2 | 2.59e-2 | 1.01 | fails |
| 0.05   | 1.27e-1 | 1.30e-1 | 1.02 | fails |

Note the ratio: ≈1.01 throughout. **Noise passes the Riemann gate**, because
its identity violations really are just truncation error — of a field with no
continuum limit. So a second gate is needed. But be clear about how much it
buys:

- `eps` scales with **amplitude as well as resolution**. It is
  `dx²‖g''‖/‖g‖`, proportional to `(dx/λ)²` only at fixed amplitude. A
  single-resolution threshold therefore cannot separate "unresolved" from
  "small" in general.
- Consequently **noise below roughly 0.8% of ‖g‖ passes this gate.** At
  amp 1e-4 and 1e-3 it sails through at every resolution tested.
- The threshold sits between the roughest legitimate case tested (smooth,
  N=16, amp 0.1: eps = 1.26e-2) and noise at amp 0.01 (eps = 2.55e-2). That is
  a 2x margin — but it is a margin between two *amplitudes*, not between
  resolved and unresolved fields, which is a weaker statement than it looks.

**The real discriminator is convergence, not magnitude.** Under refinement a
smooth field's eps falls ~4x per doubling while noise does not move:

| | N=32 | N=64 | N=128 |
|---|---|---|---|
| smooth, amp 0.1 | 2.94e-3 | 7.07e-4 | 1.73e-4 |
| noise, amp 0.001 | 2.74e-4 | 2.56e-4 | 2.93e-4 |

`converges_under_refinement(eps_coarse, eps_fine, cfg)` implements that test
and is the one to use when it matters. It is not wired into
`validate_curvature`, because that would require computing the metric at two
resolutions on every call. Treat `check_metric_resolved` as a cheap screen for
grossly unresolved fields, not as proof that a field is well resolved.

## Relaxing a gate

Every gate can be turned off:

```python
GateConfig(strict=False)                      # report everything, raise nothing
GateConfig(require_derived_entropy=False)     # allow an inserted profile
GateConfig(riemann_tol=1e-2)                  # loosen one tolerance
```

This is intentional — exploratory work needs to run unverified things. The
design goal was never to make bypassing impossible; it was to make bypassing
**explicit, local, and visible in the diff**, instead of being the default
state of the code.

## Reporting

Gates accumulate into a `ValidationReport`:

```python
report = ValidationReport()
geometry.validate_curvature(gates=cfg, report=report)
print(report.summary())
```

```
validation: 4/4 gates passed
  [PASS] meaningful_dimension  2.000e+00 (tol 4.0e+00)  dim=2: the continuum Einstein tensor vanishes identically in 2D...
  [PASS] finite:metric  0.000e+00 (tol 0.0e+00)  metric is finite
  [PASS] riemann_identities  ...
  [PASS] finite:einstein_tensor  ...
```

`report.to_dict()` serialises for storage alongside results, so a published
number can be accompanied by the gates it cleared.
