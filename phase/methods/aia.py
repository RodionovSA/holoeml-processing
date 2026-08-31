"""Advanced Iterative Algorithm (AIA) for phase-shifting interferometry."""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import backend as _backend
from ..backend import get_array_module, wrap
from .base import MethodParam, _fmt_value


@dataclass
class AIAParam(MethodParam):
    """AIA's per-method diagnostics, carried on ``PhaseResult.method_param``.

    Attributes
    ----------
    kappa_p : float
        Condition number of the spatial (pixel-step) normal matrix
        ``A_p`` -- how well the phase-shift distribution conditions the
        per-pixel ``(a, u, v)`` solve. Enters the accuracy prediction
        formula directly.
    kappa_ps : float
        Condition number of the paper's *normalized* temporal design
        (unit-circle directions ``cos(phi), sin(phi)``, i.e. amplitude
        ``b`` divided out) -- how well the recovered phase pattern alone
        covers the unit circle. Theoretically bounded below by 2, with 2
        achieved when the phase is evenly distributed over ``2*pi`` (Chen
        & Kemao 2019, Eq. 12/24). Large values mean the field has too
        little phase variation (less than roughly one fringe) for the
        frame step to reliably separate ``delta_n`` from noise -- this
        can happen even when the loop reports ``converged``. Note this is
        computed independently of the actual (amplitude-weighted) frame
        step solve, so it can be *optimistic* when much of the field is
        unmodulated (low ``b``), since flat background pixels contribute
        spurious phase there without contributing real signal.
    predicted_rms : float
        Predicted RMS phase error, in radians, from the accuracy model
        of Chen & Kemao (2019), Eq. 28/40.
    iters_run : int
        Number of alternating least-squares iterations actually run.
    converged : bool
        Whether the loop stopped because ``tol`` was reached (True) or
        because ``iters`` was exhausted without reaching it (False).
    """

    kappa_p: float
    kappa_ps: float
    predicted_rms: float
    iters_run: int
    converged: bool

    def print_summary(self) -> None:
        """Print converged, kappa_p, kappa_ps, predicted_rms -- in that order, one per line."""
        print(f"converged:     {_fmt_value(self.converged)}")
        print(f"kappa_p:       {_fmt_value(self.kappa_p)}")
        print(f"kappa_ps:      {_fmt_value(self.kappa_ps)}")
        print(f"predicted_rms: {_fmt_value(self.predicted_rms)}")


def _cond3(M, xp):
    """2-norm condition number of a small square matrix ``M`` (here 3x3).

    Equivalent to ``np.linalg.cond(M)`` (its default, ``p=None``, is exactly
    ``smax/smin`` of the SVD for a square matrix) but implemented directly
    via ``xp.linalg.svd`` rather than ``xp.linalg.cond`` -- cupy's ``linalg``
    module doesn't provide ``cond``, while ``svd`` is available on both, so
    this one implementation runs unchanged on numpy and cupy.
    """
    s = xp.linalg.svd(M, compute_uv=False)
    smin = float(s.min())
    if smin <= 0:
        return float("inf")
    return float(s.max()) / smin


def _chunked_sigma(I, A, X, xp, chunk: int = 1_000_000):
    """RMS of ``I - A @ X`` without ever materializing the full residual.

    ``I`` is ``(N, P)`` in the working dtype, ``A`` is ``(N, 3)`` float64,
    ``X`` is ``(3, P)`` in ``I``'s dtype. The direct
    ``resid = I - A @ X; sqrt(mean(resid**2))`` allocates a second full-size
    ``(N, P)`` array purely to reduce it to one scalar -- for a large stack
    this is the single largest transient allocation in :func:`aia`, since it
    scales the same way ``I`` itself does. Streaming over pixel chunks with
    a float64 accumulator gives a bit-identical result at a small, fixed
    peak memory, and each chunk's matmul is small enough to stay well under
    a display-GPU's watchdog kernel-timeout.
    """
    N, P = I.shape
    A_work = A.astype(I.dtype)
    ssq = 0.0
    for s in range(0, P, chunk):
        resid = I[:, s:s + chunk] - A_work @ X[:, s:s + chunk]
        ssq += float(xp.sum(resid.astype(xp.float64) ** 2))
    return float(np.sqrt(ssq / (N * P)))


def aia(stack: np.ndarray, g: np.ndarray, delta0: Optional[np.ndarray] = None,
        iters: int = 30, tol: float = 1e-4, dtype=None):
    """Advanced Iterative Algorithm (AIA) for phase-shifting interferometry.

    Recovers the wrapped phase map from a stack of phase-shifted
    interferograms whose phase-step sizes are not precisely known, by
    jointly estimating the per-pixel fringe pattern and the per-frame
    phase steps.

    Model
    -----
    Each frame is assumed to follow the standard phase-shifting model::

        I_n(x, y) = a(x, y) + g_n * b(x, y) * cos(phi(x, y) + delta_n)
                  = a(x, y) + g_n * [u(x, y) * cos(delta_n) + v(x, y) * sin(delta_n)]

    where ``u = b*cos(phi)``, ``v = -b*sin(phi)``, and ``g_n`` is each
    frame's fringe contrast relative to the shared map ``b`` (see the ``g``
    parameter). For fixed ``delta_n`` and ``g_n`` this is linear in
    ``(a, u, v)``, and for fixed ``(u, v)``
    it is linear in ``delta_n`` -- but not linear in both at once, so the
    unknowns are recovered by alternating least squares.

    Algorithm
    ---------
    Each iteration alternates two linear solves, until the largest
    per-frame phase-step update falls below ``tol`` (or ``iters`` is
    reached):

    1. **Pixel step** -- with ``delta_n`` fixed, solve a per-pixel linear
       regression across the ``N`` frames for the background ``a`` and
       quadrature components ``u, v``. The normal matrix of this solve is
       ``A_p`` (Chen & Kemao's notation).
    2. **Frame step** -- with ``(a, u, v)`` fixed, solve a per-frame
       linear regression across the ``P = H*W`` pixels for each frame's
       phase step ``delta_n``. A free per-frame offset is kept in this
       fit to absorb the background ``a`` and any frame-to-frame
       brightness drift, since it is re-derived from the raw data each
       iteration rather than subtracted explicitly -- subtracting the
       (still-imperfect, mid-iteration) ``a`` estimate was tried and
       found empirically to be less robust, letting its error feed back
       into the fit and increase, rather than reduce, leakage into ``a``.
       The normal matrix of this solve is ``A_ps``.

    Phase steps are re-referenced to frame 0 (``delta[0] = 0``) every
    iteration, since the model has a phase-origin ambiguity that would
    otherwise let the iteration drift.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted interferogram frames, already alpha-normalized and on
        the target device (:meth:`phase.solver.PhaseSolver.fit` does both
        before dispatching here).
    g : np.ndarray, shape (N,)
        Per-frame fringe contrast ``g_n`` (see Model), already resolved by
        the caller -- all ones if gain estimation is disabled
        (``PhaseConfig(use_g=False)``), or
        :func:`phase.utils.measure_frame_contrast`'s output otherwise. This
        function does not estimate gain itself: ``g_n`` must be supplied
        already measured, not fit jointly with ``(u, v)`` here -- like
        ``delta_n``, ``g_n`` enters the model bilinearly with ``(u, v)``,
        and this implementation only alternates between the two blocks
        described under Algorithm above.
    delta0 : np.ndarray, shape (N,), optional
        Initial guess for the phase step of each frame, in radians. If
        not given, defaults to evenly-spaced steps
        ``delta0[i] = i * 2*pi / N``, which minimizes ``kappa_ps`` (its
        theoretical lower bound is 2) and gives the most reliable
        convergence when the true phase-shift distribution is unknown.
    iters : int, default 30
        Maximum number of alternating least-squares iterations.
    tol : float, default 1e-4
        Convergence tolerance, in radians, on the largest per-frame
        change in the estimated phase step between iterations (Chen &
        Kemao's recommended default).
    dtype : numpy/cupy dtype, optional
        Working dtype for the large ``(N, P)``-shaped arrays (the interferogram
        stack reshaped and every per-pixel quantity derived from it).
        Defaults to ``float32`` (see :func:`phase.backend.default_dtype`) --
        the camera already writes float32, and float64 here both doubles
        memory for no benefit and runs at 1/32 throughput on non-datacenter
        GPUs. The small per-iteration linear algebra (the two 3-unknown
        normal-equation solves, condition numbers, and the residual
        reduction used for ``predicted_rms``) always runs in float64
        regardless of this setting, so accuracy is governed by the model,
        not by this dtype -- verified against an equivalent float64-throughout
        run at <1e-4 degrees RMS.

    Returns
    -------
    a, b, phi, delta : np.ndarray
        The Eq. (8) fields recovered by this method -- ``a`` and ``b`` shape
        ``(H, W)``, ``phi`` shape ``(H, W)`` wrapped to ``(-pi, pi]``,
        ``delta`` shape ``(N,)``. Numpy or cupy arrays matching ``stack``'s
        array module -- not forced back to the host, so that chaining
        ``aia`` -> :func:`~phase.combine.combine_acquisitions` on a GPU
        doesn't round-trip large arrays over PCIe in between; call
        :func:`phase.backend.asnumpy` yourself when you need a
        guaranteed-numpy array.
    method_param : AIAParam
        ``kappa_p, kappa_ps, predicted_rms`` are diagnostics for whether
        this acquisition (frame count, phase-shift distribution, noise
        level) supports a trustworthy result; ``iters_run, converged``
        describe convergence. See :class:`AIAParam`.

    References
    ----------
    Z. Wang and B. Han, "Advanced iterative algorithm for phase
    extraction of randomly phase-shifted interferograms," Optics and
    Lasers in Engineering (2004).

    Y. Chen and Q. Kemao, "Advanced iterative algorithm for phase
    extraction: performance evaluation and enhancement," Optics Express
    27(26), 37634-37651 (2019). Establishes that accuracy is governed by
    the condition numbers of the two least-squares steps (``kappa_p``,
    ``kappa_ps``, computed here) and the accuracy prediction formula used
    for ``predicted_rms``; also shows the background term decouples from
    the fringe terms when phase-shifts are well distributed, i.e. good
    conditioning -- not subtracting ``a`` -- is the correct lever for
    accuracy (see Notes).

    """
    xp = get_array_module(stack)
    N, H, W = stack.shape
    work_dtype = dtype if dtype is not None else _backend.default_dtype(xp)
    I = stack.reshape(N, -1).astype(work_dtype, copy=False)        # (N, P)
    P = I.shape[1]

    if delta0 is None:
        delta0 = xp.arange(N) * 2 * xp.pi / N
    delta = xp.asarray(delta0, dtype=xp.float64).copy()
    g = xp.asarray(g, dtype=xp.float64)

    # (P,3) working buffer for the frame-step design matrix, allocated once
    # and overwritten in place each iteration (u, v change; the constant
    # column doesn't) rather than rebuilt via column_stack every pass.
    UV = xp.empty((P, 3), dtype=work_dtype, order="F")
    UV[:, 0] = 1

    u = v = a = None
    A = None
    converged = False
    it = 0
    for it in range(iters):
        # pixel step: I_n = a + g_n*(u cosδ_n + v sinδ_n),  u=b cosφ, v=-b sinφ
        c, s = xp.cos(delta), xp.sin(delta)
        A = xp.column_stack([xp.ones(N), g * c, g * s])                # (N,3) float64
        X = xp.linalg.pinv(A).astype(work_dtype) @ I                   # (3,P)
        a, u, v = X[0], X[1], X[2]

        # frame step: fit [alpha_n, Pn, Qn] against the fixed u,v patterns
        UV[:, 1] = u
        UV[:, 2] = v
        BtB = (UV.T @ UV).astype(xp.float64)                          # (3,3)
        IB = (I @ UV).astype(xp.float64)                              # (N,3)
        x = xp.linalg.solve(BtB, IB.T)                                # (3,N)
        Pn, Qn = x[1], x[2]
        new_delta = xp.arctan2(Qn, Pn)

        new_delta = new_delta - new_delta[0]                          # pin phase origin
        step = float(xp.abs(wrap(new_delta - delta)).max())
        delta = new_delta
        if step < tol:
            converged = True
            break

    phi = xp.arctan2(-v, u).reshape(H, W)
    u64, v64 = u.astype(xp.float64), v.astype(xp.float64)
    b   = xp.sqrt(u64**2 + v64**2).reshape(H, W)
    a_map = a.reshape(H, W)

    # diagnostics (Chen & Kemao 2019): condition numbers of the two
    # normal matrices, and the accuracy they predict.
    kappa_p = _cond3(A.T @ A, xp)

    # kappa_ps: paper's *normalized* A_ps (Eq. 12), built from unit-circle
    # directions cos(phi), sin(phi) with amplitude b divided out. This is
    # deliberately different from the actual (amplitude-weighted) frame-step
    # solve matrix B.T@B -- normalizing is what makes the >=2 bound and the
    # "large is bad" threshold below meaningful; the amplitude-weighted
    # version is skewed by fringe-visibility variation, not just phase
    # coverage (see AIAParam.kappa_ps docstring). Assembled here from five
    # scalar reductions rather than materializing a (P,3) design matrix
    # ``C`` just to form ``C.T @ C`` -- same 3x3 Gram matrix, at a small
    # fraction of the peak memory for a large acquisition.
    r = xp.maximum(xp.sqrt(u64**2 + v64**2), xp.finfo(xp.float64).eps)
    cphi, sphi = u64 / r, -v64 / r
    Scp, Ssp = float(cphi.sum()), float(sphi.sum())
    Scc, Sss = float((cphi * cphi).sum()), float((sphi * sphi).sum())
    Scs = float((cphi * sphi).sum())
    CtC = xp.asarray([[float(P), Scp, Ssp], [Scp, Scc, Scs], [Ssp, Scs, Sss]])
    kappa_ps = _cond3(CtC, xp)

    sigma = _chunked_sigma(I, A, xp.vstack([a, u, v]), xp)
    b_amp = max(float(xp.median(b)), np.finfo(float).eps)
    predicted_rms = 0.42 * (np.sqrt(kappa_p) + 2) * (sigma / b_amp) / np.sqrt(N)

    if kappa_p > 20:
        warnings.warn(
            f"aia: poorly conditioned phase-shift distribution "
            f"(kappa_p={kappa_p:.1f}); accuracy is unreliable. Consider "
            f"more evenly-spaced phase shifts and/or more frames.",
            stacklevel=2,
        )
    if kappa_ps > 20:
        warnings.warn(
            f"aia: poor phase coverage (kappa_ps={kappa_ps:.1f}); the "
            f"field spans too little phase (roughly less than one fringe) "
            f"for the frame step to reliably separate delta_n from noise, "
            f"even though the iteration converged. Consider adding phase "
            f"diversity (e.g. tilt/carrier fringes) or using calibrated "
            f"phase steps instead of blind estimation.",
            stacklevel=2,
        )

    method_param = AIAParam(
        kappa_p=kappa_p, kappa_ps=kappa_ps, predicted_rms=predicted_rms,
        iters_run=it + 1, converged=converged,
    )
    return a_map, b, phi, delta, method_param
