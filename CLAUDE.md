# EntropicUnification — working notes

Differentiable framework for learning spacetime geometry from quantum
entanglement entropy. A research sandbox, not a result generator.

## The rule that matters

**v1.3 withdrew every v1.2 result as circular** — the pipeline had been
recovering the geometry it was implicitly given. That withdrawal is the standard
this project is held to.

So: any result showing a known metric being recovered is **presumed circular
until proven otherwise**. Before reporting that Schwarzschild came out, prove the
answer wasn't an input — check what the entropy field was seeded with, what the
loss actually constrains, and whether an ablation that should break the recovery
does break it. A result that survives only under the exact configuration that
produced it is not a result.

Withdraw findings that turn out wrong, in full and in writing. Don't soften it.

## Layout

- `core/entropy_module.py` — the entropy field. Where circularity enters.
- `core/geometry_engine.py` — metric/curvature
- `core/loss_functions.py` — what is actually being constrained
- `core/validation.py` — consistency checks. As of v1.4 these are **gates that
  abort**, not warnings. Don't downgrade a gate to a warning to make a run finish.
- `scripts/calibrate_gates.py` — regenerates every tolerance table in
  `docs/VALIDATION.md`. Change a tolerance, rerun this, update the doc.
- `core/training_loop.py`, `optimizer.py`, `advanced_optimizer.py`
- `paper/`, `ARXIV_DRAFT.md` — the writeup. Must not outrun what's verified.

## History

`v1.3-honest-physics` has been merged; `main` carries v1.3 and v1.4. v1.4 turned
consistency checks into aborting gates; v1.4.1 made FAULKNER use the covariant
Hessian. Older commits ("fix all mathematical calculations") predate the honest
pipeline — don't mine them for approaches.

## Environment

Never build the venv on the anaconda interpreter — `.pth` files aren't processed
there and editable installs silently fail to import. `uv venv --python 3.12`.

**Imports on this machine are pathologically slow** (the volume runs ~98% full):
`import torch` takes minutes, `import pennylane` was measured at **21 minutes**.
Budget for it — run anything torch-touching with `nohup ... &` and poll the log
rather than waiting on a foreground timeout.

Nothing outside `core/quantum_engine.py` needs pennylane, so stub it when
testing the rest:

```python
import sys, types
sys.modules.setdefault("pennylane", types.ModuleType("pennylane"))
```

`QuantumEngine.reduced_density_matrix` is pure torch and can be bound onto a
stub class carrying only `num_qubits` — that gives real partial traces with no
pennylane import. `scripts/calibrate_gates.py` does exactly this.

## Workflow

Plan/execute/judge: strongest model plans and judges, Sonnet/Haiku subagents
execute. The judge step is not optional here — v1.4's gates were found to be
validating the *initial flat* metric (which cannot fail them) only because an
independent reviewer looked.

## Claims

Numbers in the README, the paper, or a report must trace to a run that can be
reproduced. If it can't be rerun, it doesn't get claimed.
