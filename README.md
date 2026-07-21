# holoEML processing

Processing tools for experimental interferogram data from the holoEML project.

## Overview

This repository extracts phase information from stacks of phase-shifted
interferograms — recovering the wrapped phase map, fringe amplitude, and
background from frames where the exact phase-shift steps aren't precisely
known. Phase unwrapping and related tooling are planned as the project grows.

## Project structure

- `phase/` — phase-processing library.
  - `utils.py` — `aia`, an Advanced Iterative Algorithm implementation for
    blind phase-shift extraction (Wang & Han 2004; enhanced per Chen & Kemao,
    *Optics Express* 27(26), 37634-37651, 2019).
- `main.py` — entry-point stub.
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
from phase.utils import aia

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

## Status

Early / work in progress.
