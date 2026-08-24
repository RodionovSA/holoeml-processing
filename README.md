# holoEML processing

Processing tools for experimental interferogram data from the holoEML project.

## Overview

This repository extracts phase information from stacks of phase-shifted
interferograms — recovering the wrapped phase map, fringe amplitude, and
background from frames where the exact phase-shift steps aren't precisely
known. Phase unwrapping and related tooling are planned as the project grows.

## Project structure

- `phase/` — phase-processing library.
  - `aia.py` — `aia`, an Advanced Iterative Algorithm implementation for
    blind phase-shift extraction (Wang & Han 2004; enhanced per Chen & Kemao,
    *Optics Express* 27(26), 37634-37651, 2019), plus `measure_frame_contrast`.
  - `carrier.py` — `remove_carrier`, estimating/removing a spatial
    carrier and (optionally) defocus from a wrapped phase map.
  - `reference.py` — `subtract_reference`, resolving the AIA sign-branch
    ambiguity between a sample and reference phase map.
  - `combine.py` — `combine_acquisitions`, averaging repeated independent
    acquisitions of the same object.
  - `ripple.py` — `estimate_phase_ripple` / `apply_phase_ripple`, correcting
    a phase-locked error `eps(phi)`.
  - `backend.py` — NumPy/CuPy array-module dispatch shared by all of the
    above (see GPU section below).
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

## Quick usage

```python
from phase import aia

# stack: np.ndarray, shape (N, H, W) — N phase-shifted interferogram frames
result = aia(stack)

phi = result.phi   # wrapped phase map, (H, W), in (-pi, pi]
b   = result.b     # fringe modulation amplitude map, (H, W)
a   = result.a     # background intensity map, (H, W)

print(result.converged, result.iters_run)
print(result.kappa_p, result.kappa_ps, result.predicted_rms)
```

`kappa_p` and `kappa_ps` are condition-number diagnostics (Chen & Kemao 2019):
large values flag a poorly conditioned acquisition (bad phase-shift spacing or
insufficient phase coverage) whose result shouldn't be trusted, even if
`converged` is `True`. `predicted_rms` is the paper's predicted phase error
in radians.

## GPU (CuPy)

Every public function (`aia`, `remove_carrier`, `subtract_reference`,
`combine_acquisitions`, `estimate_phase_ripple`) accepts a
`device="auto"|"cpu"|"cuda"` keyword. `"auto"` (the default) uses a GPU via
[CuPy](https://cupy.dev/) if one is installed and available, else falls back
to NumPy on the CPU — no other code changes needed either way. Result arrays
stay on whichever device ran the computation (they are *not* downloaded
automatically), so chaining calls on the GPU doesn't round-trip large arrays
over PCIe in between; call `phase.backend.asnumpy(x)` to bring one back to
the host explicitly (e.g. before plotting or `np.savez`).

To enable it:

```bash
uv sync --extra cuda
```

This installs `cupy-cuda12x`. Match the CUDA *major* version to what your
driver can actually run, not just what `nvidia-smi` reports as its maximum —
a driver can advertise CUDA 13 support while the installed GPU (anything
below compute capability 7.5: Maxwell/Pascal/Volta) cannot execute CUDA 13
binaries, since that toolkit dropped offline compilation for those
architectures. If in doubt, use `cupy-cuda12x`.

The large `(N, P)`-shaped arrays default to `float32` (`dtype=` on `aia` /
`measure_frame_contrast` overrides this); the small per-iteration linear
algebra (condition numbers, normal-equation solves, residual reductions)
always runs in `float64` regardless. This was verified against a `float64`
run on real data at <1e-4 degrees RMS — far below this setup's measured
~1.15 degrees/acquisition scatter.

## Status

Early / work in progress.
