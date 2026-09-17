"""Pipeline gates: make the physics either provably right or loudly wrong.

v1.2 reported a Schwarzschild recovery that was an artifact of a Gaussian
inserted by hand into the "entropy" field.  Nothing in the code objected,
because every consistency check was advisory — a printed number a human had
to notice.  This module turns those checks into gates that *fail the run*.

The design rule: a defect that would invalidate a published number must raise,
not warn.  Three gates matter most.

Provenance
    An entropy field is only admissible as physics if it was computed from
    partial traces of an actual quantum state.  `EntropyField` carries that
    fact with it, and the coupling layer refuses a bare tensor.  This is the
    gate that would have caught the v1.2 defect at the source.

Tracelessness
    The MASSLESS and FAULKNER formulations are traceless by construction.
    A non-zero trace means the contraction is wrong (v1.2 used a Euclidean
    dot product instead of g^μν), so it is an implementation bug, not a
    physical finding.  Checked relative to the tensor's own scale.

Curvature identities
    The lowered Riemann tensor must satisfy antisymmetry, pair symmetry and
    the first Bianchi identity.  v1.2 *projected* these onto the mixed-index
    tensor, where they do not hold, corrupting the curvature while making the
    diagnostics look perfect.  Here they are measured and gated, never imposed.

Every gate can be relaxed explicitly — `GateConfig(strict=False)` to report
without raising, or an individual `require_*` / tolerance field — for
exploratory work.  Relaxing one is a deliberate, visible act; that is the
whole point.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence

import torch


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class ValidationError(RuntimeError):
    """A pipeline gate failed. The run is not physically meaningful."""


class ProvenanceError(ValidationError):
    """A quantity was used as physics without a verified derivation."""


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

#: Sources that count as genuinely derived from a quantum state.
DERIVED_SOURCES = frozenset({"partial_trace"})



@dataclass(frozen=True)
class Provenance:
    """Where a field came from, recorded at construction time.

    `derived` is the load-bearing bit: True only when every value traces back
    to a partial trace of an actual quantum state.
    """

    source: str
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def derived(self) -> bool:
        return self.source in DERIVED_SOURCES

    def __str__(self) -> str:
        bits = ", ".join(f"{k}={v}" for k, v in self.detail.items())
        tag = "derived" if self.derived else "NOT DERIVED"
        return f"{self.source} [{tag}]" + (f" ({bits})" if bits else "")


_DERIVATION_TOKEN = object()


@dataclass(frozen=True)
class EntropyField:
    """An entropy field S(x) that knows how it was produced.

    Frozen, and its constructor refuses a derived provenance supplied
    directly: a field may only claim `partial_trace` if it was built through
    `from_partial_traces`, which requires the state it came from. That closes
    the obvious forgery —

        EntropyField(hand_placed_gaussian, Provenance("partial_trace"))

    — which would otherwise walk past the provenance gate and reproduce the
    v1.2 defect with the gate's own blessing. Freezing stops the values being
    swapped out after the claim is recorded.

    This remains an attestation about how the object was constructed, not a
    proof that the numbers are what they claim; `values_digest` binds the
    claim to the actual values so later substitution is at least detectable.
    """

    values: torch.Tensor
    provenance: Provenance
    _token: Any = None

    def __post_init__(self) -> None:
        if self.values.dim() != 1:
            raise ValueError(
                f"entropy field must be 1-D over the lattice, got shape "
                f"{tuple(self.values.shape)}"
            )
        if self.provenance.derived and self._token is not _DERIVATION_TOKEN:
            raise ProvenanceError(
                f"cannot construct an EntropyField claiming "
                f"{self.provenance.source!r} directly — a derived field must be "
                f"built by EntropyModule.entropy_field() or "
                f"EntropyField.from_partial_traces(), which require the quantum "
                f"state. Use EntropyField.unverified(...) for a hand-built field."
            )

    # -- tensor-ish conveniences -------------------------------------------
    @property
    def shape(self) -> torch.Size:
        return self.values.shape

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def __getitem__(self, idx):
        return self.values[idx]

    def to(self, *a, **kw) -> "EntropyField":
        return EntropyField(self.values.to(*a, **kw), self.provenance,
                            _DERIVATION_TOKEN)

    def detach(self) -> "EntropyField":
        return EntropyField(self.values.detach(), self.provenance,
                            _DERIVATION_TOKEN)

    def tensor(self) -> torch.Tensor:
        return self.values

    @staticmethod
    def _digest(t: torch.Tensor) -> str:
        with torch.no_grad():
            return hashlib.sha256(
                t.detach().to(torch.float64).cpu().numpy().tobytes()
            ).hexdigest()[:16]

    def values_digest(self) -> str:
        """Digest of the values, so a recorded claim can be re-checked."""
        return self._digest(self.values)

    def verify_digest(self) -> bool:
        """True if the values still match the digest taken at construction."""
        recorded = self.provenance.detail.get("values_digest")
        return recorded is None or recorded == self.values_digest()

    @classmethod
    def from_partial_traces(
        cls, values: torch.Tensor, *, num_qubits: int,
        qubit_positions: Sequence[float], interpolation: str,
        state_norm: float, state_digest: str = "",
    ) -> "EntropyField":
        return cls(values, Provenance("partial_trace", {
            "num_qubits": num_qubits,
            "qubit_positions": [round(float(p), 6) for p in qubit_positions],
            "interpolation": interpolation,
            "state_norm": round(float(state_norm), 12),
            "state_digest": state_digest,
            "values_digest": cls._digest(values),
        }), _DERIVATION_TOKEN)

    @classmethod
    def unverified(cls, values: torch.Tensor, source: str = "synthetic",
                   **detail: Any) -> "EntropyField":
        """Wrap a field that is *not* derived from a state. Gated by default."""
        if source in DERIVED_SOURCES:
            raise ValueError(
                f"{source!r} claims derivation; construct it with "
                f"from_partial_traces() so the claim is backed by real inputs"
            )
        return cls(values, Provenance(source, dict(detail)))



# --------------------------------------------------------------------------
# Gate configuration
# --------------------------------------------------------------------------

@dataclass
class GateConfig:
    """Tolerances and switches for the pipeline gates.

    Defaults are strict: a run that passes with these defaults has cleared
    every check whose failure invalidated v1.2.
    """

    strict: bool = True

    #: |g^μν T_μν| / ‖T‖ for formulations that are traceless by construction.
    #: Exact algebra in float64 lands near 1e-16; 1e-10 is generous headroom
    #: while still catching a wrong contraction by many orders of magnitude.
    traceless_tol: float = 1e-10

    #: Ratio of the measured Riemann identity violation to the truncation
    #: error the metric's own smoothness implies. Values near 1 mean the
    #: violation is exactly the expected O(dx²) finite-difference error; a
    #: much larger ratio means the curvature code is wrong, independent of
    #: resolution. Calibrated at ~1.15 across N=16..256; 4.0 leaves margin.
    riemann_error_ratio: float = 4.0

    #: Absolute fallback when the metric's smoothness cannot be estimated.
    riemann_tol: float = 1e-3

    #: Maximum relative second difference eps = ||d2g|| / ||g||, which equals
    #: dx^2 ||g''|| / ||g|| — proportional to (dx/lambda)^2 at fixed amplitude,
    #: but it also scales with the perturbation amplitude, so it is a proxy for
    #: resolution only when amplitude is held fixed. Large-amplitude per-site
    #: noise fails here; *small*-amplitude noise does not (see
    #: docs/VALIDATION.md). The real discriminator is that eps fails to fall
    #: under refinement, which a single-resolution gate cannot see.
    metric_resolution_tol: float = 2e-2

    #: Fail if any tensor entering the loss is non-finite.
    require_finite: bool = True

    #: Fail if the entropy field is not derived from a quantum state.
    require_derived_entropy: bool = True

    #: Fail if the lattice dimension cannot support a curvature test at all.
    require_meaningful_dimension: bool = False


@dataclass
class GateResult:
    name: str
    passed: bool
    value: Optional[float]
    tolerance: Optional[float]
    message: str

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        num = "" if self.value is None else f"  {self.value:.3e}"
        tol = "" if self.tolerance is None else f" (tol {self.tolerance:.1e})"
        return f"[{mark}] {self.name}{num}{tol}  {self.message}"


class ValidationReport:
    """Accumulates gate results so a run can print what it actually verified."""

    def __init__(self) -> None:
        self.results: list[GateResult] = []

    def add(self, r: GateResult) -> GateResult:
        self.results.append(r)
        return r

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]

    def summary(self) -> str:
        head = (f"validation: {len(self.results) - len(self.failures)}"
                f"/{len(self.results)} gates passed")
        return "\n".join([head, *(f"  {r}" for r in self.results)])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": [
                {"name": r.name, "passed": r.passed, "value": r.value,
                 "tolerance": r.tolerance, "message": r.message}
                for r in self.results
            ],
        }


def _fail(cfg: GateConfig, result: GateResult, exc=ValidationError) -> GateResult:
    if not result.passed and cfg.strict:
        raise exc(str(result))
    return result


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def check_entropy_provenance(
    entropy: Any, cfg: GateConfig, report: Optional[ValidationReport] = None
) -> GateResult:
    """Gate: S(x) must be derived from partial traces of a quantum state.

    This is the v1.2 gate. A bare tensor is rejected outright — not because a
    tensor is necessarily wrong, but because nothing about it records whether
    its spatial structure was computed or inserted, and that distinction is
    exactly what invalidated the earlier results.
    """
    if isinstance(entropy, EntropyField):
        prov = entropy.provenance
        if not entropy.verify_digest():
            r = GateResult(
                "entropy_provenance", False, None, None,
                "entropy values do not match the digest recorded when the "
                "provenance was created: the field was substituted after "
                "derivation, so the claim no longer describes these numbers",
            )
            if report:
                report.add(r)
            return _fail(cfg, r, ProvenanceError)
        ok = prov.derived or not cfg.require_derived_entropy
        msg = (f"entropy provenance: {prov}" if ok else
               f"entropy field is {prov} — its spatial structure was not "
               f"computed from a quantum state, so any geometry recovered from "
               f"it restates its own input (this is the v1.2 defect). Pass "
               f"GateConfig(require_derived_entropy=False) to explore anyway.")
        r = GateResult("entropy_provenance", ok, None, None, msg)
    else:
        ok = not cfg.require_derived_entropy
        r = GateResult(
            "entropy_provenance", ok, None, None,
            "bare tensor supplied as the entropy field: no provenance recorded. "
            "Build it with EntropyModule.entropy_field(...) so the derivation is "
            "carried with the data."
            if not ok else "provenance check disabled",
        )
    if report:
        report.add(r)
    return _fail(cfg, r, ProvenanceError)


def check_trace_identity(
    stress_tensor: torch.Tensor,
    metric: torch.Tensor,
    expected_trace: Optional[torch.Tensor],
    cfg: GateConfig,
    report: Optional[ValidationReport] = None,
    label: str = "",
) -> GateResult:
    """Gate: g^μν T_μν must equal what the formulation's algebra predicts.

    This is stronger than checking tracelessness, and applies to more of the
    formulations.  Each stress tensor has a trace fixed by its own definition:

      MASSLESS   0                  (the 1/n coefficient makes it traceless)
      FAULKNER   (1 - n) □S         (∇_μ∇_νS - (□S)g_μν is *not* traceless)

    Verifying the predicted value catches a wrong contraction — v1.2 used a
    Euclidean dot product where g^μν belongs — without requiring the answer to
    be zero.  Compared relative to ‖T‖, so the check is scale-free.

    Pass `expected_trace=None` for formulations with no fixed trace
    (LAGRANGIAN, CANONICAL, MODIFIED); the gate then reports as skipped.
    """
    if expected_trace is None:
        r = GateResult("trace_identity", True, None, None,
                       f"no fixed trace for {label or 'this formulation'}")
        if report:
            report.add(r)
        return r

    with torch.no_grad():
        try:
            g_inv = torch.linalg.inv(metric)
        except Exception:
            # A degenerate metric is itself a finding, not a crash.
            g_inv = torch.linalg.pinv(metric)
        trace = torch.einsum("nab,nab->n", g_inv, stress_tensor)
        scale = torch.linalg.norm(stress_tensor.flatten()) / math.sqrt(
            max(stress_tensor.shape[0], 1)
        )
        rel = float((trace - expected_trace).abs().max() / (scale + 1e-300))

    ok = rel <= cfg.traceless_tol
    r = GateResult(
        "trace_identity", ok, rel, cfg.traceless_tol,
        f"g^μν T_μν matches the {label or 'declared'} algebra"
        + ("" if ok else " — the contraction is wrong (v1.2 used a Euclidean "
                         "dot product here); this is a bug, not a result"),
    )
    if report:
        report.add(r)
    return _fail(cfg, r)


def check_tracelessness(
    stress_tensor: torch.Tensor,
    metric: torch.Tensor,
    formulation: str,
    cfg: GateConfig,
    report: Optional[ValidationReport] = None,
) -> GateResult:
    """Gate: MASSLESS must be exactly traceless.

    Only MASSLESS is traceless by construction. FAULKNER is *not* — its trace
    is (1 - n)□S, verified separately by `check_trace_identity`; the claim
    that it is traceless was a documentation error, caught by this gate.
    """
    form = str(formulation).lower().split(".")[-1]
    if form != "massless":
        r = GateResult("tracelessness", True, None, None,
                       f"not applicable to {form} (only MASSLESS is traceless "
                       f"by construction)")
        if report:
            report.add(r)
        return r
    return check_trace_identity(
        stress_tensor, metric,
        torch.zeros(stress_tensor.shape[0], dtype=stress_tensor.dtype,
                    device=stress_tensor.device),
        cfg, report, label="massless (traceless)",
    )


def check_riemann_identities(
    violations: Dict[str, float],
    cfg: GateConfig,
    report: Optional[ValidationReport] = None,
    truncation: Optional[float] = None,
) -> GateResult:
    """Gate: the lowered Riemann tensor must satisfy its algebraic identities.

    Measured, never projected. v1.2 imposed these symmetries on the
    *mixed-index* tensor where they do not hold, which corrupted the curvature
    while making every diagnostic read zero.

    On a lattice these identities are not exact — they hold up to
    second-order finite-difference error — so an absolute threshold is
    meaningless on its own. What *is* meaningful is the ratio of the measured
    violation to the truncation error the metric's own smoothness predicts
    (`GeometryEngine.metric_truncation_scale()`). A correct implementation
    sits near 1 at every resolution; a wrong one does not, no matter how fine
    the lattice.
    """
    if not violations:
        r = GateResult("riemann_identities", False, None, None,
                       "no identity violations reported — the diagnostic "
                       "returned nothing, so nothing was actually checked")
        if report:
            report.add(r)
        return _fail(cfg, r)
    worst_name, worst = max(violations.items(), key=lambda kv: kv[1])
    # An (almost) flat metric has truncation ~0, which would make the scaled
    # tolerance tighter than float64 noise. Floor it at the absolute value.
    if truncation is not None and truncation * cfg.riemann_error_ratio < cfg.riemann_tol:
        truncation = None
    if truncation is None or truncation <= 0.0:
        tol, basis = cfg.riemann_tol, "absolute fallback"
        ratio = None
    else:
        tol = cfg.riemann_error_ratio * truncation
        ratio = worst / truncation
        basis = (f"{cfg.riemann_error_ratio:g}x predicted truncation error "
                 f"{truncation:.2e} (ratio {ratio:.2f})")
    ok = worst <= tol
    r = GateResult(
        "riemann_identities", ok, worst, tol,
        f"worst violation: {worst_name} [{basis}]"
        + ("" if ok else " — violation exceeds what discretization explains, "
                         "so the curvature computation is wrong rather than "
                         "merely coarse"),
    )
    if report:
        report.add(r)
    return _fail(cfg, r)


def check_metric_resolved(
    truncation: float,
    cfg: GateConfig,
    report: Optional[ValidationReport] = None,
) -> GateResult:
    """Gate: the lattice must actually resolve the metric's variation.

    `truncation` is the relative second difference ||d2g|| / ||g||. A smooth
    metric on an adequate lattice gives a small number that shrinks as the
    lattice refines; per-site noise gives one that does not shrink, because it
    has no continuum limit — there is no metric there to take the curvature
    of, and every derived tensor is meaningless however carefully computed.

    Limitation, stated plainly: this gate thresholds a single-resolution
    magnitude, and that magnitude scales with perturbation amplitude as well
    as with resolution. Noise below roughly 0.8% of ||g|| passes. Catching
    *any* unresolved field requires comparing eps at two resolutions and
    requiring it to fall ~4x; use `converges_under_refinement()` for that.
    """
    ok = truncation <= cfg.metric_resolution_tol
    r = GateResult(
        "metric_resolved", ok, truncation, cfg.metric_resolution_tol,
        "lattice resolves the metric's variation" if ok else
        "metric varies on the scale of the lattice spacing: it has no "
        "continuum limit, so its curvature is not approximating anything. "
        "Refine the lattice or smooth the field.",
    )
    if report:
        report.add(r)
    return _fail(cfg, r)


def converges_under_refinement(
    eps_coarse: float, eps_fine: float, cfg: GateConfig,
    report: Optional[ValidationReport] = None,
) -> GateResult:
    """Gate: the truncation parameter must fall under lattice refinement.

    This is the real test of whether a field has a continuum limit, and the
    one `check_metric_resolved` cannot perform from a single resolution.
    Halving dx should cut a second-order truncation error by ~4x; noise does
    not improve at all. Requiring a factor of 2 leaves room for the metric
    not being perfectly smooth while still failing a non-convergent field.

    Args:
        eps_coarse: metric_truncation_scale() at lattice spacing 2*dx.
        eps_fine: metric_truncation_scale() at lattice spacing dx.
    """
    ratio = eps_coarse / (eps_fine + 1e-300)
    ok = ratio >= 2.0
    r = GateResult(
        "refinement_convergence", ok, ratio, 2.0,
        f"truncation fell {ratio:.2f}x under refinement"
        + ("" if ok else " — the field does not converge, so it has no "
                         "continuum limit and its curvature approximates "
                         "nothing (second-order error should fall ~4x)"),
    )
    if report:
        report.add(r)
    return _fail(cfg, r)


def check_source_differentiable(
    second_derivative_scale: Dict[int, float],
    cfg: GateConfig,
    report: Optional[ValidationReport] = None,
) -> GateResult:
    """Gate: the entropy field must be differentiable to the order used.

    S(r) built with interpolation="linear" is piecewise linear — C0 but not
    C1, with kinks at the qubit positions. Its first derivative is ambiguous
    at those kinks (finite differences just pick a resolution of the
    ambiguity), and its second derivative is a sum of delta functions that
    diverges as 1/dx under refinement.

    That makes the FAULKNER formulation, which needs d2S, ill-posed on the
    default source: refining the lattice makes the answer worse without
    bound. Measured max|d2S| on the default GHZ profile — 65.8, 271, 1104,
    4452, 17876 at N = 32..512, i.e. 4x per doubling.

    Args:
        second_derivative_scale: {lattice_size: max|d2S|} at two or more
            resolutions. A well-posed source gives a bounded sequence.
    """
    sizes = sorted(second_derivative_scale)
    if len(sizes) < 2:
        r = GateResult("source_differentiable", True, None, None,
                       "need two resolutions to judge; not checked")
        if report:
            report.add(r)
        return r
    coarse, fine = second_derivative_scale[sizes[0]], second_derivative_scale[sizes[-1]]
    growth = fine / (coarse + 1e-300)
    ok = growth <= 1.5
    r = GateResult(
        "source_differentiable", ok, growth, 1.5,
        f"max|d2S| grew {growth:.2f}x from N={sizes[0]} to N={sizes[-1]}"
        + ("" if ok else " — the source is not twice differentiable, so any "
                         "formulation using d2S (FAULKNER) is ill-posed on it: "
                         "refining the lattice makes the answer diverge. Use a "
                         "smooth interpolation, or a formulation needing only dS."),
    )
    if report:
        report.add(r)
    return _fail(cfg, r)


def check_finite(
    name: str,
    tensor: torch.Tensor,
    cfg: GateConfig,
    report: Optional[ValidationReport] = None,
) -> GateResult:
    """Gate: no NaN/Inf may enter the loss.

    A diverged optimizer that quietly produces NaN will still print a loss
    history and a plot; this makes it stop instead.
    """
    if not cfg.require_finite:
        r = GateResult(f"finite:{name}", True, None, None, "finiteness check disabled")
        if report:
            report.add(r)
        return r
    with torch.no_grad():
        bad = int((~torch.isfinite(tensor)).sum())
    ok = bad == 0
    r = GateResult(f"finite:{name}", ok, float(bad), 0.0,
                   f"{name} has {bad} non-finite entries" if not ok
                   else f"{name} is finite")
    if report:
        report.add(r)
    return _fail(cfg, r)


def check_meaningful_dimension(
    dimensions: int,
    cfg: GateConfig,
    report: Optional[ValidationReport] = None,
) -> GateResult:
    """Gate: in 2D the Einstein tensor vanishes identically.

    G_μν ≡ 0 for *every* metric in two dimensions, so a 1+1D run cannot test
    whether entanglement sources curvature — the target it is fitting is
    structurally zero. The framework currently runs in 1+1D, so this gate is
    off by default; turning it on is how you assert a run is meant to be
    physically conclusive.
    """
    ok = dimensions >= 4 or not cfg.require_meaningful_dimension
    r = GateResult(
        "meaningful_dimension", ok, float(dimensions), 4.0,
        f"dim={dimensions}: the continuum Einstein tensor vanishes identically "
        f"in 2D, so H3 cannot be tested here"
        if dimensions < 4 else f"dim={dimensions} supports a curvature test",
    )
    if report:
        report.add(r)
    return _fail(cfg, r)


def check_entropy_field_sanity(
    entropy: Any,
    cfg: GateConfig,
    report: Optional[ValidationReport] = None,
) -> GateResult:
    """Gate: S(x) must be non-negative and not identically zero.

    A pure state's entropy field vanishing everywhere means the qubits carry
    no entanglement across any cut — there is no source, and any curvature the
    optimizer finds is fitting noise.
    """
    values = entropy.values if isinstance(entropy, EntropyField) else entropy
    with torch.no_grad():
        min_v = float(values.min())
        span = float(values.max() - values.min())
    if min_v < -1e-9:
        r = GateResult("entropy_sanity", False, min_v, 0.0,
                       "entropy field has negative values — von Neumann "
                       "entropy cannot be negative")
    elif span <= 1e-12:
        r = GateResult("entropy_sanity", False, span, 1e-12,
                       "entropy field is constant: no gradient, hence no "
                       "source — the run cannot test the conjecture")
    else:
        r = GateResult("entropy_sanity", True, span, 1e-12,
                       f"S(x) non-negative, varies by {span:.3e}")
    if report:
        report.add(r)
    return _fail(cfg, r)


__all__ = [
    "ValidationError", "ProvenanceError", "Provenance", "EntropyField",
    "DERIVED_SOURCES", "GateConfig", "GateResult",
    "ValidationReport", "check_entropy_provenance", "check_tracelessness",
    "check_trace_identity",
    "check_riemann_identities", "check_metric_resolved",
    "converges_under_refinement", "check_source_differentiable",
    "check_finite",
    "check_meaningful_dimension",
    "check_entropy_field_sanity",
]
