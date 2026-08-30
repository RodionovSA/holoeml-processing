# holoEML processing

Processing tools for experimental interferogram data from the holoEML project.

## Overview

This repository extracts phase information from stacks of phase-shifted
interferograms — recovering the wrapped phase map, fringe amplitude, and
background from frames where the exact phase-shift steps aren't precisely
known. Phase unwrapping and related tooling are planned as the project grows.

## Project structure

- `phase/` — phase-processing library.
  - `solver.py` — `PhaseSolver`/`PhaseConfig`/`PhaseResult`: the main entry
    point. Configure a `PhaseConfig` (which algorithm to run and how) and
    call `PhaseSolver(config).fit(stack)` to recover phase.
  - `methods/` — one module per phase-recovery algorithm, registered in
    `methods/__init__.py`'s `METHOD_REGISTRY` (`phase.solver.METHODS` is
    derived from it). Currently: `aia.py` — an Advanced Iterative Algorithm
    implementation for blind phase-shift extraction (Wang & Han 2004;
    enhanced per Chen & Kemao, *Optics Express* 27(26), 37634-37651, 2019).
  - `utils.py` — `measure_frame_contrast`/`measure_frame_visibility`,
    per-frame fringe-gain estimation shared by every method.
  - `carrier.py` — `remove_carrier`, estimating/removing a spatial
    carrier and (optionally) defocus from a wrapped phase map.
  - `reference.py` — `subtract_reference`, resolving the phase sign-branch
    ambiguity between a sample and reference phase map.
  - `combine.py` — `combine_acquisitions`, averaging repeated independent
    acquisitions of the same object.
  - `ripple.py` — `estimate_phase_ripple` / `apply_phase_ripple`, correcting
    a phase-locked error `eps(phi)`.
  - `backend.py` — NumPy/CuPy array-module dispatch shared by all of the
    above (see GPU section below).
- `docs/interference_model.md` — the interferometry model (Eq. 8) every
  module in `phase/` is written against.
- `main.py` — entry-point stub.
- `tests/` — unit tests (synthetic data; run in seconds).
- `scripts/test/` — notebook-based checks against real acquisitions.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

Installs `phase` in editable mode plus its dependencies into `.venv`. Run
`uv sync` again any time `pyproject.toml`/`uv.lock` change. Run project
commands with `uv run <command>` (e.g. `uv run pytest`, `uv run jupyter lab`),
or `source .venv/bin/activate` first.

## Quick usage

```python
from phase import PhaseSolver, PhaseConfig

# stack: np.ndarray, shape (N, H, W) — N phase-shifted interferogram frames
solver = PhaseSolver(PhaseConfig()).fit(stack)

phi = solver.phi_   # wrapped phase map, (H, W), in (-pi, pi]
b   = solver.b_     # fringe modulation amplitude map, (H, W)
a   = solver.a_     # background intensity map, (H, W)

print(solver.reconstruction_error_)      # RMSE of the fit, method-agnostic
print(solver.method_param_)              # diagnostics specific to the method used
```

`PhaseConfig` selects and configures the algorithm (`method="aia"` by
default; see `phase.solver.METHODS` for what's registered) and controls the
shared normalization/gain-estimation steps (`use_alpha`, `use_g`, `g`);
`solver.method_param_` carries whatever diagnostics that method reports —
for `"aia"`, an `AIAParam` with `kappa_p`/`kappa_ps` (condition-number
diagnostics from Chen & Kemao 2019: large values flag a poorly conditioned
acquisition whose result shouldn't be trusted, even if `converged` is
`True`), `predicted_rms` (the paper's predicted phase error in radians),
`iters_run`, and `converged`. See `phase/solver.py` and
`phase/methods/aia.py` for full parameter/field documentation.

## GPU (CuPy)

Every public function in this package accepts a `device="auto"|"cpu"|"cuda"`
keyword (`PhaseSolver`'s is a constructor argument; the rest take it directly).
`"auto"` (the default) uses a GPU via [CuPy](https://cupy.dev/) if one is
installed and available, else falls back to NumPy on the CPU — no other code
changes needed either way. Result arrays stay on whichever device ran the
computation (they are *not* downloaded automatically), so chaining calls on
the GPU doesn't round-trip large arrays over PCIe in between; call
`phase.backend.asnumpy(x)` to bring one back to the host explicitly (e.g.
before plotting or `np.savez`).

To enable it:

```bash
uv sync --extra cuda
```

This installs `cupy-cuda12x`, and requires an NVIDIA GPU with a matching
CUDA driver — irrelevant on a machine without one (e.g. Apple Silicon or any
non-NVIDIA GPU), where `device="auto"` already runs on NumPy/CPU with no
setup needed. Match the CUDA *major* version to what your driver can
actually run, not just what `nvidia-smi` reports as its maximum — a driver
can advertise CUDA 13 support while the installed GPU (anything below
compute capability 7.5: Maxwell/Pascal/Volta) cannot execute CUDA 13
binaries, since that toolkit dropped offline compilation for those
architectures. If in doubt, use `cupy-cuda12x`.

The large `(N, P)`-shaped arrays default to `float32` (`dtype=` on
`PhaseSolver`/`measure_frame_contrast` overrides this); the small
per-iteration linear algebra (condition numbers, normal-equation solves,
residual reductions) always runs in `float64` regardless. This was verified
against an equivalent float64-throughout run at <1e-4 degrees RMS — far
below the ~1.15 degrees/acquisition scatter floor measured in
`combine_acquisitions`'s docstring.

## Status

Early / work in progress.
