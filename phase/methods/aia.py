"""Advanced Iterative Algorithm (AIA) for phase-shifting interferometry."""

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .. import backend as _backend
from ..backend import get_array_module, wrap
from .base import MethodParam, _fmt_value


@dataclass
class AIAParam(MethodParam):
    """AIA's per-method diagnostics, carried on ``PhaseResult.method_param``.

    See ``docs/aia.md`` §"Accuracy diagnostics" for the derivation of
    ``kappa_p``, ``kappa_ps``, and ``predicted_rms``.

    Attributes
    ----------
    kappa_p : float
        Condition number of the pixel-step normal matrix ``A_p`` -- large
        values (>20) mean an unreliable solve; consider more evenly-spaced
        phase shifts or more frames.
    kappa_ps : float
        Condition number of the normalized frame-step design (unit-circle
        directions, amplitude divided out), bounded below by 2. Large
        values mean too little phase variation (less than ~one fringe) for
        the frame step to reliably separate ``delta_n`` from noise, even
        when the loop reports ``converged``.
    predicted_rms : float
        Predicted RMS phase error, in radians (Chen & Kemao 2019, Eq. 28/40).
    iters_run : int
        Number of alternating least-squares iterations actually run.
    converged : bool
        Whether the loop stopped because ``tol`` was reached (True) or
        ``iters`` was exhausted (False).
    g_fit : np.ndarray, shape (N,)
        Per-frame fringe contrast that ``(a, u, v)`` were actually fit
        against -- one iteration behind this call's own updated ``g`` when
        ``fit_gain=True``, same convention as ``kappa_p``/``kappa_ps``.
    c_fit : np.ndarray, shape (N,)
        Per-frame residual offset ``(a, u, v)`` were fit against,
        gauge-fixed to ``mean(c_fit) = 0``. All zero when ``fit_gain=False``;
        large otherwise signals ``PhaseConfig.use_alpha`` hasn't fully
        removed frame-to-frame brightness drift.
    g_min_ratio : float
        ``min(g_fit) / median(g_fit)``. Very small means some frame is
        nearly uncorrelated with the recovered fringe pattern and should
        probably be dropped from the acquisition.
    """

    kappa_p: float
    kappa_ps: float
    predicted_rms: float
    iters_run: int
    converged: bool
    g_fit: np.ndarray
    c_fit: np.ndarray
    g_min_ratio: float

    def print_summary(self) -> None:
        """Print converged, kappa_p, kappa_ps, predicted_rms, g_min_ratio -- in that order, one per line."""
        print(f"converged:     {_fmt_value(self.converged)}")
        print(f"kappa_p:       {_fmt_value(self.kappa_p)}")
        print(f"kappa_ps:      {_fmt_value(self.kappa_ps)}")
        print(f"predicted_rms: {_fmt_value(self.predicted_rms)}")
        print(f"g_min_ratio:   {_fmt_value(self.g_min_ratio)}")


def _cond3(M, xp):
    """2-norm condition number of a small square matrix ``M`` (here 3x3).

    Implemented via ``xp.linalg.svd`` rather than ``xp.linalg.cond`` --
    cupy's ``linalg`` has no ``cond``, but both provide ``svd``.
    """
    s = xp.linalg.svd(M, compute_uv=False)
    smin = float(s.min())
    if smin <= 0:
        return float("inf")
    return float(s.max()) / smin


def _whiten_uv(u: np.ndarray, v: np.ndarray, xp) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate/shear ``(u, v)`` to have equal pixel-sum energy and be orthogonal.

    Required after every pixel step when ``fit_gain`` is True, to fix a
    gauge freedom that only appears once ``g_n`` is free -- see
    ``docs/aia.md`` §"A gauge freedom that only appears once g_n is free".
    Forcing ``sum(u**2) == sum(v**2)`` and ``sum(u*v) == 0`` collapses that
    freedom from ``GL(2)`` down to ``O(2)``, the same ambiguity plain AIA
    already resolves elsewhere (phase origin, sign branch). This does not
    change the objective any :func:`aia_pixel_step` call already minimized
    -- the very next :func:`aia_frame_step` call reaches a joint residual at
    least as low as before whitening.

    Parameters
    ----------
    u, v : np.ndarray, shape (P,)
        Quadrature components from :func:`aia_pixel_step`.
    xp : module
        ``numpy`` or ``cupy``, matching ``u``/``v``.

    Returns
    -------
    u, v : np.ndarray, shape (P,)
        Whitened quadrature components, same dtype as the inputs.
    """
    u64, v64 = u.astype(xp.float64), v.astype(xp.float64)
    Suu = float(xp.sum(u64 * u64))
    Svv = float(xp.sum(v64 * v64))
    Suv = float(xp.sum(u64 * v64))
    eps = np.finfo(float).eps
    scale = np.sqrt(max((Suu + Svv) / 2, eps))

    K = np.array([[Suu, Suv], [Suv, Svv]])
    w, V = np.linalg.eigh(K)
    w = np.maximum(w, eps)
    M = (V * (scale / np.sqrt(w))) @ V.T             # symmetric, scale * K^(-1/2)
    m00, m01, m11 = float(M[0, 0]), float(M[0, 1]), float(M[1, 1])

    u_new = u * m00 + v * m01
    v_new = u * m01 + v * m11
    return u_new, v_new


def _pixel_design(delta: np.ndarray, g: np.ndarray, xp) -> np.ndarray:
    """Assemble the ``(N, 3)`` pixel-step design matrix ``[1, g*cos(delta), g*sin(delta)]``.

    Shared by :func:`aia_pixel_step` and :func:`aia` (for the ``kappa_p``
    diagnostic) so the two never drift apart.
    """
    return xp.column_stack([xp.ones_like(delta), g * xp.cos(delta), g * xp.sin(delta)])


def _chunked_sigma(I, A, X, xp, c=None, chunk: int = 1_000_000):
    """RMS of ``I - c - A @ X`` without ever materializing the full residual.

    Streams over pixel chunks with a float64 accumulator instead of
    allocating a second full ``(N, P)`` residual array just to reduce it to
    one scalar -- otherwise the largest transient allocation in :func:`aia`
    for a large stack.

    ``I`` is ``(N, P)`` in the working dtype, ``A`` is ``(N, 3)`` float64,
    ``X`` is ``(3, P)`` in ``I``'s dtype, ``c`` is an optional ``(N,)``
    per-frame offset (``None`` for the ``fit_gain=False`` residual).
    """
    N, P = I.shape
    A_work = A.astype(I.dtype)
    c_work = None if c is None else c.astype(I.dtype)[:, None]
    ssq = 0.0
    for s in range(0, P, chunk):
        resid = I[:, s:s + chunk] - A_work @ X[:, s:s + chunk]
        if c_work is not None:
            resid = resid - c_work
        ssq += float(xp.sum(resid.astype(xp.float64) ** 2))
    return float(np.sqrt(ssq / (N * P)))

def _aia_diagnostics(I, delta_fit, g, a, u, v, N, xp, iters_run: int, converged: bool,
                      c=None) -> AIAParam:
    """Assemble :class:`AIAParam` from a solved ``(a, u, v)`` and the piston
    ``delta`` it was fit against.

    Factored out of :func:`aia` so :func:`phase.methods.step_field.aia_step_field`
    can recompute the same diagnostics for its own final, refined solution --
    see :class:`AIAParam` and ``docs/aia.md`` for the derivations.

    Parameters
    ----------
    I : np.ndarray, shape (N, P)
        Flattened interferogram stack.
    delta_fit : np.ndarray, shape (N,)
        Piston phase steps that ``(a, u, v)`` were actually fit against.
    g : np.ndarray, shape (N,)
        Per-frame fringe contrast that ``(a, u, v)`` were actually fit
        against -- reported unchanged as :attr:`AIAParam.g_fit`.
    a, u, v : np.ndarray, shape (P,)
        Background and quadrature components.
    N : int
        Number of frames.
    xp : module
        ``numpy`` or ``cupy``, matching the other arguments.
    iters_run : int
        Value to report as :attr:`AIAParam.iters_run`.
    converged : bool
        Value to report as :attr:`AIAParam.converged`.
    c : np.ndarray, shape (N,), optional
        Per-frame residual offset from the joint-gain frame step, reported
        as :attr:`AIAParam.c_fit` and subtracted from ``I`` before measuring
        the residual used for ``predicted_rms``. ``None`` (the
        ``fit_gain=False`` default) reports an all-zero ``c_fit`` and
        leaves ``I`` uncorrected.

    Returns
    -------
    AIAParam
    """
    P = I.shape[1]

    A = _pixel_design(delta_fit, g, xp)
    kappa_p = _cond3(A.T @ A, xp)

    # kappa_ps uses the *normalized* design (unit-circle directions, b
    # divided out) -- see AIAParam.kappa_ps. Built from five scalar
    # reductions rather than a (P,3) design matrix, for memory.
    u64, v64 = u.astype(xp.float64), v.astype(xp.float64)
    r = xp.maximum(xp.sqrt(u64**2 + v64**2), xp.finfo(xp.float64).eps)
    cphi, sphi = u64 / r, -v64 / r
    Scp, Ssp = float(cphi.sum()), float(sphi.sum())
    Scc, Sss = float((cphi * cphi).sum()), float((sphi * sphi).sum())
    Scs = float((cphi * sphi).sum())
    CtC = xp.asarray([[float(P), Scp, Ssp], [Scp, Scc, Scs], [Ssp, Scs, Sss]])
    kappa_ps = _cond3(CtC, xp)

    sigma = _chunked_sigma(I, A, xp.vstack([a, u, v]), xp, c=c)
    b = xp.sqrt(u64**2 + v64**2)
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

    c_fit = xp.zeros(N, dtype=xp.float64) if c is None else xp.asarray(c, dtype=xp.float64)
    g_min_ratio = float(xp.min(g)) / max(float(xp.median(g)), np.finfo(float).eps)
    if g_min_ratio < 0.1:
        warnings.warn(
            f"aia: at least one frame's fitted gain is far below the "
            f"median (g_min_ratio={g_min_ratio:.3f}); that frame is nearly "
            f"uncorrelated with the recovered fringe pattern and its "
            f"delta_n is poorly determined. Consider dropping it from the "
            f"acquisition.",
            stacklevel=2,
        )

    return AIAParam(
        kappa_p=kappa_p, kappa_ps=kappa_ps, predicted_rms=predicted_rms,
        iters_run=iters_run, converged=converged,
        g_fit=xp.asarray(g, dtype=xp.float64), c_fit=c_fit, g_min_ratio=g_min_ratio,
    )


def aia_pixel_step(stack: np.ndarray, delta: np.ndarray, g: Optional[np.ndarray] = None,
                    dtype=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel least-squares solve for the background and quadrature fields.

    With ``delta_n``/``g_n`` fixed, each pixel's ``N`` samples follow
    ``I_n = a + g_n*(u*cos(delta_n) + v*sin(delta_n))``, linear in
    ``(a, u, v)`` -- solved for every pixel at once via one shared
    pseudoinverse of the ``(N, 3)`` design matrix. See ``docs/aia.md``
    §"Pixel step". Here ``u = b*cos(phi)``, ``v = -b*sin(phi)``; recovering
    ``phi`` needs this paired with a ``delta_n`` estimate (e.g.
    :func:`aia_frame_step`).

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P`` pixels each.
    delta : np.ndarray, shape (N,)
        Phase step of each frame, in radians.
    g : np.ndarray, shape (N,), optional
        Per-frame fringe contrast. Defaults to all ones (no frame-to-frame
        contrast variation).
    dtype : numpy/cupy dtype, optional
        Working dtype for the returned ``(P,)`` fields. Defaults to
        ``stack``'s array module's :func:`phase.backend.default_dtype`. The
        design matrix and pseudoinverse are always computed in float64
        regardless of this setting.

    Returns
    -------
    a, u, v : np.ndarray, shape (P,)
        Background and quadrature components, in ``dtype``.
    """
    if len(stack.shape) != 2:
        raise ValueError(f"Stack shape must be have 2 dims, but got {len(stack.shape)}")
    if len(delta.shape) != 1:
        raise ValueError(f"Delta shape must be have 1 dim, but got {len(delta.shape)}")
    if stack.shape[0] != len(delta):
        raise ValueError("First dimensions of stack and delta must be equal")
    if g is not None and len(g) != len(delta):
        raise ValueError("g must have the same length as delta")

    xp = get_array_module(stack)
    N = stack.shape[0]
    work_dtype = dtype if dtype is not None else _backend.default_dtype(xp)
    delta = xp.asarray(delta, dtype=xp.float64)
    g = xp.ones(N, dtype=xp.float64) if g is None else xp.asarray(g, dtype=xp.float64)

    A = _pixel_design(delta, g, xp)                                 # (N,3) float64
    X = xp.linalg.pinv(A).astype(work_dtype) @ stack                # (3,P)
    return X[0], X[1], X[2]


def aia_frame_step(stack: np.ndarray, u: np.ndarray, v: np.ndarray,
                    precise_reduce: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame least-squares solve for each frame's phase step and gain.

    With ``(u, v)`` fixed, each frame's ``P`` pixels follow
    ``I_n ~ c_n + P_n*u + Q_n*v`` for a free offset ``c_n`` and
    ``(P_n, Q_n) = g_n*(cos(delta_n), sin(delta_n))``, giving
    ``delta_n = atan2(Q_n, P_n)``, ``g_n = hypot(P_n, Q_n)`` -- the
    transpose of :func:`aia_pixel_step`'s solve. See ``docs/aia.md``
    §"Frame step".

    ``c_n`` is left free rather than folded into the pixel step's ``a``:
    re-deriving it from raw data every call was found more robust than
    carrying forward a mid-iteration estimate of ``a``.

    Returned ``delta`` is absolute (not pinned to a phase origin) and ``g``
    is unnormalized -- a caller iterating on ``delta`` or wanting
    ``median(g) = 1`` (see :attr:`phase.solver.PhaseResult.g`) must do so
    itself.

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P`` pixels each.
    u, v : np.ndarray, shape (P,)
        Quadrature components, e.g. as returned by :func:`aia_pixel_step`.
    precise_reduce : bool, default True
        See :attr:`phase.solver.PhaseConfig.precise_reduce`.

    Returns
    -------
    delta : np.ndarray, shape (N,), float64
        Estimated phase step of each frame, in radians.
    g : np.ndarray, shape (N,), float64
        Estimated fringe contrast of each frame, unnormalized.
    c : np.ndarray, shape (N,), float64
        Estimated per-frame offset (background level plus brightness
        drift), unnormalized.
    """
    if len(stack.shape) != 2:
        raise ValueError(f"Stack shape must be have 2 dims, but got {len(stack.shape)}")
    if len(u.shape) != 1 or len(v.shape) != 1:
        raise ValueError("u and v must each have 1 dim")
    if len(u) != stack.shape[1] or len(v) != stack.shape[1]:
        raise ValueError("u and v must have the same length as stack's second dimension")

    xp = get_array_module(stack, u, v)
    P = stack.shape[1]
    u64 = xp.asarray(u, dtype=xp.float64)
    v64 = xp.asarray(v, dtype=xp.float64)

    # (P,)-sized, so always float64 regardless of precise_reduce -- never
    # touches stack, costs nothing.
    Su, Sv = float(xp.sum(u64)), float(xp.sum(v64))
    Suu = float(xp.sum(u64 * u64))
    Svv = float(xp.sum(v64 * v64))
    Suv = float(xp.sum(u64 * v64))
    BtB = xp.asarray([[float(P), Su, Sv], [Su, Suu, Suv], [Sv, Suv, Svv]])

    # precise_reduce controls whether stack @ u/v promotes stack to float64
    # (True) or runs at stack's own dtype (False) -- the one place in this
    # function that actually costs memory.
    u_mm = u64 if precise_reduce else xp.asarray(u, dtype=stack.dtype)
    v_mm = v64 if precise_reduce else xp.asarray(v, dtype=stack.dtype)
    IB = xp.stack([xp.sum(stack, axis=1).astype(xp.float64),
                    (stack @ u_mm).astype(xp.float64),
                    (stack @ v_mm).astype(xp.float64)], axis=1)      # (N,3)

    x = xp.linalg.solve(BtB, IB.T)                                  # (3,N)
    c, Pn, Qn = x[0], x[1], x[2]
    delta = xp.arctan2(Qn, Pn)
    g = xp.sqrt(Pn * Pn + Qn * Qn)
    return delta, g, c


def aia(stack: np.ndarray, g: np.ndarray, fit_gain: bool = False,
        delta0: Optional[np.ndarray] = None,
        iters: int = 30, tol: float = 1e-4, dtype=None,
        precise_reduce: bool = True):
    """Advanced Iterative Algorithm (AIA) for phase-shifting interferometry.

    Recovers the wrapped phase map, fringe amplitude, background, and
    per-frame phase steps from a stack of phase-shifted interferograms
    whose step sizes aren't precisely known, by alternating a per-pixel
    least-squares solve (:func:`aia_pixel_step`) with a per-frame one
    (:func:`aia_frame_step`) until convergence. See ``docs/aia.md`` for the
    full derivation and the numbered algorithm, including the ``fit_gain``
    joint-gain extension and its whitening step (:func:`_whiten_uv`).

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted interferogram frames, already alpha-normalized and on
        the target device (:meth:`phase.solver.PhaseSolver.fit` does both).
    g : np.ndarray, shape (N,)
        Per-frame fringe contrast. Fixed when ``fit_gain=False``, initial
        guess when ``fit_gain=True``.
    fit_gain : bool, default False
        Recover ``g_n`` jointly with ``delta_n`` instead of holding it
        fixed. Prefer this over an out-of-band estimate (e.g.
        :func:`phase.utils.measure_frame_contrast`, which needs a spatial
        carrier and fails on circular/carrier-free fringes) whenever
        contrast drifts frame-to-frame.
    delta0 : np.ndarray, shape (N,), optional
        Initial guess for each frame's phase step, in radians. Defaults to
        evenly-spaced steps, which minimizes ``kappa_ps`` (see
        ``docs/aia.md`` §"Accuracy diagnostics").
    iters : int, default 30
        Maximum number of alternating least-squares iterations.
    tol : float, default 1e-4
        Convergence tolerance on the largest per-frame change in ``delta``
        (and, when ``fit_gain`` is True, in ``g``) between iterations.
    dtype : numpy/cupy dtype, optional
        Working dtype for the ``(N, P)``-shaped arrays. Defaults to
        ``float32`` (see :func:`phase.backend.default_dtype`); the small
        per-iteration linear algebra always runs in float64 regardless of
        this setting.
    precise_reduce : bool, default True
        See :attr:`phase.solver.PhaseConfig.precise_reduce`.

    Returns
    -------
    a, b, phi, delta, g : np.ndarray
        ``a``, ``b``, ``phi`` shape ``(H, W)`` (``phi`` wrapped to
        ``(-pi, pi]``); ``delta``, ``g`` shape ``(N,)``. ``g`` is the input
        ``g`` unchanged when ``fit_gain=False``, or the jointly fitted
        contrast (``median(g) = 1``) when True. Numpy or cupy arrays
        matching ``stack``'s array module, not forced back to the host --
        call :func:`phase.backend.asnumpy` yourself if needed.
    method_param : AIAParam
        Convergence and accuracy diagnostics -- see :class:`AIAParam`.

    References
    ----------
    Z. Wang and B. Han, "Advanced iterative algorithm for phase
    extraction of randomly phase-shifted interferograms," Optics and
    Lasers in Engineering (2004).

    Y. Chen and Q. Kemao, "Advanced iterative algorithm for phase
    extraction: performance evaluation and enhancement," Optics Express
    27(26), 37634-37651 (2019).
    """
    xp = get_array_module(stack)
    N, H, W = stack.shape
    work_dtype = dtype if dtype is not None else _backend.default_dtype(xp)
    I = stack.reshape(N, -1).astype(work_dtype, copy=False)        # (N, P)

    if delta0 is None:
        delta0 = xp.arange(N) * 2 * xp.pi / N
    delta = xp.asarray(delta0, dtype=xp.float64).copy()
    g = xp.asarray(g, dtype=xp.float64)
    c = xp.zeros(N, dtype=xp.float64)

    u = v = a = None
    delta_fit = delta
    g_fit = g
    converged = False
    it = 0
    for it in range(iters):
        delta_fit = delta
        g_fit = g
        c_fit = c
        # No-op (c stays zero) unless fit_gain -- see docs/aia.md.
        I_pixel = I - c_fit.astype(work_dtype)[:, None] if fit_gain else I
        a, u, v = aia_pixel_step(I_pixel, delta_fit, g_fit, dtype=work_dtype)
        if fit_gain:
            u, v = _whiten_uv(u, v, xp)
        new_delta, new_g, new_c = aia_frame_step(I, u, v, precise_reduce=precise_reduce)

        new_delta = new_delta - new_delta[0]                          # pin phase origin
        step = float(xp.abs(wrap(new_delta - delta)).max())

        if fit_gain:
            new_c = new_c - xp.mean(new_c)                            # gauge fix
            new_g = new_g / max(float(xp.median(new_g)), np.finfo(float).eps)
            step = max(step, float(xp.abs(new_g - g).max()))
            g = new_g
            c = new_c

        delta = new_delta
        if step < tol:
            converged = True
            break

    phi = xp.arctan2(-v, u).reshape(H, W)
    u64, v64 = u.astype(xp.float64), v.astype(xp.float64)
    b   = xp.sqrt(u64**2 + v64**2).reshape(H, W)
    a_map = a.reshape(H, W)

    # Diagnostics use what (a, u, v) were actually fit against
    # (delta_fit/g_fit/c_fit), not the final frame step's update.
    method_param = _aia_diagnostics(I, delta_fit, g_fit, a, u, v, N, xp, it + 1, converged,
                                     c=(c_fit if fit_gain else None))
    return a_map, b, phi, delta, g, method_param
