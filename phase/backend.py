"""Array-module dispatch (NumPy/CuPy) shared across :mod:`phase`.

Every function in this package is written against ``xp = get_array_module(...)``
and calls ``xp.`` throughout, so it runs unchanged on NumPy (CPU, always
available) or CuPy (GPU, only if installed and a CUDA device is present) --
whichever module the arrays handed to it already live on. This module is the
only place that imports cupy, and it does so optionally: everything works
with only numpy installed.
"""

import numpy as np

try:
    import cupy as _cp
except ImportError:  # pragma: no cover - exercised only where cupy is absent
    _cp = None

CUPY_AVAILABLE = _cp is not None


def get_array_module(*arrays):
    """Return the array module (``numpy`` or ``cupy``) that ``arrays`` live on.

    Thin wrapper around ``cupy.get_array_module`` that also works when cupy
    isn't installed at all (falls back to ``numpy`` unconditionally, rather
    than raising). Every public function in this package starts with
    ``xp = get_array_module(...)`` on its array inputs and uses ``xp.``
    throughout, rather than importing numpy/cupy directly -- that's what
    makes the same source run on both. Do not mix numpy and cupy arrays in
    one call; move everything to one device first (see :func:`to_device`).
    """
    if CUPY_AVAILABLE:
        return _cp.get_array_module(*arrays)
    return np


def is_cupy(xp) -> bool:
    """True if ``xp`` (as returned by :func:`get_array_module`) is cupy."""
    return CUPY_AVAILABLE and xp is _cp


def to_device(x, device: str = "auto", dtype=None):
    """Move array-like ``x`` onto the requested device, returning an ndarray.

    Only meant for host->device staging at a public entry point (e.g.
    :meth:`phase.solver.PhaseSolver.fit`, :func:`phase.carrier.remove_carrier`,
    :func:`phase.combine.combine_acquisitions`) -- everything else should
    just call :func:`get_array_module` on whatever array it's handed rather
    than moving data around mid-pipeline.

    Parameters
    ----------
    x : array-like
        Data to move (numpy array, cupy array, or anything ``asarray``
        accepts).
    device : {"auto", "cpu", "cuda"}, default "auto"
        "cpu" always returns a numpy array. "cuda" uploads to the GPU via
        cupy, raising if cupy isn't installed. "auto" uses cupy if it's
        installed (regardless of whether ``x`` already lives there), else
        falls back to numpy -- so code that always calls this with
        ``device="auto"`` runs on the GPU when one is available and on the
        CPU otherwise, with no other changes.
    dtype : dtype, optional
        Cast while moving, avoiding a separate pass over the array.
    """
    if device == "cpu":
        xp = np
    elif device == "cuda":
        if not CUPY_AVAILABLE:
            raise RuntimeError("device='cuda' requested but cupy is not installed")
        xp = _cp
    elif device == "auto":
        xp = _cp if CUPY_AVAILABLE else np
    else:
        raise ValueError(f"device must be 'cpu', 'cuda', or 'auto', got {device!r}")

    return xp.asarray(x, dtype=dtype) if dtype is not None else xp.asarray(x)


def asnumpy(x):
    """Bring ``x`` back to a plain numpy array (no-op if it already is one)."""
    if CUPY_AVAILABLE and isinstance(x, _cp.ndarray):
        return _cp.asnumpy(x)
    return np.asarray(x)


def wrap(x):
    """Wrap ``x`` (radians) into ``(-pi, pi]``, on whichever array module owns it.

    Numerically equivalent (to float64 machine precision) to
    ``xp.angle(xp.exp(1j*x))``, which is the pattern used throughout this
    package for wrap-safe arithmetic -- but computed directly, without a
    complex round-trip, which is both cheaper and (via ``cupy.fuse``, see
    :func:`fused_wrap`) fusable into one GPU kernel instead of two transcendental
    passes plus a temporary complex array.
    """
    xp = get_array_module(x)
    two_pi = 2 * xp.pi
    return x - two_pi * xp.round(x / two_pi)


def _wrap_diff_impl(phi_a, phi_b, sign):
    """``wrap(phi_a + sign*phi_b)`` via the complex product, not raw subtraction.

    Used for the several call sites that need e.g. ``wrap(phi - phi_ref)``
    starting from two already-wrapped angles: multiplying
    ``exp(i*phi_a) * exp(i*sign*phi_b)`` and taking ``angle`` is exact and
    wrap-safe regardless of how phi_a/phi_b were themselves wrapped, whereas
    ``wrap(phi_a + sign*phi_b)`` on the raw floats is *not* always equal to
    it once phi_a/phi_b are taken modulo 2pi independently. Kept as a
    non-fused (angle/exp) path since it needs the actual complex product,
    not just a scalar mod.
    """
    xp = get_array_module(phi_a, phi_b)
    return xp.angle(xp.exp(1j * phi_a) * xp.exp(1j * sign * phi_b))


def wrap_add(phi_a, phi_b):
    """``wrap(phi_a + phi_b)``, safe when phi_a/phi_b are each already wrapped."""
    return _wrap_diff_impl(phi_a, phi_b, 1.0)


def wrap_sub(phi_a, phi_b):
    """``wrap(phi_a - phi_b)``, safe when phi_a/phi_b are each already wrapped."""
    return _wrap_diff_impl(phi_a, phi_b, -1.0)


if CUPY_AVAILABLE:
    _fused_wrap_cuda = _cp.ElementwiseKernel(
        "float64 x", "float64 y",
        "y = x - 6.283185307179586 * rint(x * 0.15915494309189535)",
        "phase_wrap",
    )

    def fused_wrap(x):
        """Single-kernel version of :func:`wrap` for cupy float64 arrays.

        Plain :func:`wrap` already fuses fine under cupy's default kernel
        fusion for typical expressions, but this ``ElementwiseKernel`` avoids
        relying on that and is the one used inside hot loops (e.g.
        :mod:`phase.carrier`'s tilt refine) where every kernel launch counts.
        Falls back to :func:`wrap` for numpy input.
        """
        xp = get_array_module(x)
        if is_cupy(xp) and x.dtype == _cp.float64:
            return _fused_wrap_cuda(x)
        return wrap(x)
else:  # pragma: no cover - exercised only where cupy is absent
    fused_wrap = wrap


def default_dtype(xp, complex_: bool = False):
    """Default working (real or complex) dtype: float32/complex64.

    Applies to the large (N, P)-shaped arrays throughout this package.
    Small per-iteration linear-algebra objects (3x3 Gram matrices, frame-count
    vectors, condition numbers, reduction accumulators) should stay float64
    regardless of this default -- see the ``dtype`` parameter of
    :func:`phase.methods.aia.aia` for why float32 there is safe and float64
    for the big arrays is not (memory, and FP64 throughput on non-datacenter
    GPUs).
    """
    return xp.complex64 if complex_ else xp.float32
