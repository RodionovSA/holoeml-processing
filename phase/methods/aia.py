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
    g_fit : np.ndarray, shape (N,)
        Per-frame fringe contrast that ``(a, u, v)`` -- and hence ``b``,
        ``phi`` -- were actually fit against: the *previous* iteration's
        frame-step output when ``fit_gain=True`` (normalized so
        ``median(g_fit) = 1``), or the input ``g`` unchanged when
        ``fit_gain=False``. Not the same as this call's own updated ``g``
        (the one this function returns), by the same one-iteration-behind
        convention as ``kappa_p``/``kappa_ps``/``predicted_rms`` below.
    c_fit : np.ndarray, shape (N,)
        Per-frame residual offset that ``(a, u, v)`` were actually fit
        against, gauge-fixed so ``mean(c_fit) = 0`` (see
        :func:`aia_frame_step`) -- same one-iteration-behind convention as
        ``g_fit``. All zero when ``fit_gain=False``. Should be small when
        ``PhaseConfig.use_alpha`` has already removed frame-to-frame
        brightness drift; a large value here is itself the signal that it
        has not.
    g_min_ratio : float
        ``min(g_fit) / median(g_fit)``. A frame whose data is nearly
        uncorrelated with the recovered fringe pattern ``(u, v)`` gets
        ``g_n -> 0`` -- useful automatic downweighting in the pixel step,
        but it also means that frame's ``delta_n`` is poorly determined.
        A very small ratio flags that a frame should probably be dropped
        from the acquisition rather than trusted.
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


def _whiten_uv(u: np.ndarray, v: np.ndarray, xp) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate/shear ``(u, v)`` to have equal pixel-sum energy and be orthogonal.

    Required after every pixel step when ``fit_gain`` is True (see
    :func:`aia`), to fix a gauge freedom that only appears once ``g_n`` is
    free. With ``g_n`` fixed at 1, ``aia_frame_step``'s ``(P_n, Q_n)`` is
    implicitly constrained to the unit circle (``P_n=cos(delta_n),
    Q_n=sin(delta_n)``), which is what makes plain AIA identifiable up to
    just a global phase origin and the documented sign branch. Once
    ``g_n`` floats, ``(P_n, Q_n)`` is unconstrained in the plane, and the
    model ``a(x) + c_n + P_n*u(x) + Q_n*v(x)`` is then invariant under
    *any* invertible linear reparametrization ``(u, v) -> (u, v) @ M``,
    ``(P, Q) -> (P, Q) @ M^-T`` -- because ``{1, u, v}`` and
    ``{1, u@M, v@M}`` span the same subspace for any invertible ``M``, not
    just a rotation. Left alone, the alternating solve can converge to any
    basis of that subspace: it fits the data exactly as well (often
    *better*, since a generic basis has more freedom to explain noise) but
    ``(P_n, Q_n)`` no longer traces ``(g_n*cos(delta_n), g_n*sin(delta_n))``
    for any physically meaningful ``delta_n``, so the recovered phase can
    be badly wrong even though ``reconstruction_error`` looks good --
    verified on synthetic data: without this step, a fit that halves the
    residual relative to the correctly-identified (fixed-``g``) solution
    can still land tens of degrees off in phase.

    Forcing ``sum(u**2) == sum(v**2)`` and ``sum(u*v) == 0`` (the pixel-sum
    inner product) collapses the residual gauge from the full invertible
    group down to just rotations and reflections (``O(2)``) -- exactly the
    ambiguity plain AIA already has and already resolves elsewhere (phase
    origin via ``delta[0] = 0``, sign via the documented ``(phi, delta) ->
    (-phi, -delta)`` branch), rather than the two of them plus an
    additional two-parameter shear/scale family. Total pixel-sum energy
    (the Gram matrix's trace) is preserved, so this does not change ``g``'s
    overall scale -- only :func:`aia`'s own ``g /= median(g)`` step does.

    This does not change the objective any :func:`aia_pixel_step` call
    already minimized: ``(u, v)`` and their whitened version span the same
    3-D column space (together with the constant term), so the very next
    :func:`aia_frame_step` call -- an unconstrained per-frame regression
    against whichever basis it is given -- always reaches a joint residual
    at least as low as before whitening. The eigendecomposition of the
    tiny (2, 2) Gram matrix is done via plain ``numpy`` regardless of
    ``xp`` (cupy has no advantage on a 2x2 matrix); only the elementwise
    combination of ``(u, v)`` with the resulting scalar coefficients runs
    on ``xp``.

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

    Shared by :func:`aia_pixel_step` (to solve for ``a, u, v``) and :func:`aia`
    (to recompute the same matrix for the ``kappa_p`` / residual diagnostics)
    so the two never drift apart.
    """
    return xp.column_stack([xp.ones_like(delta), g * xp.cos(delta), g * xp.sin(delta)])


def _chunked_sigma(I, A, X, xp, c=None, chunk: int = 1_000_000):
    """RMS of ``I - c - A @ X`` without ever materializing the full residual.

    ``I`` is ``(N, P)`` in the working dtype, ``A`` is ``(N, 3)`` float64,
    ``X`` is ``(3, P)`` in ``I``'s dtype, ``c`` is an optional ``(N,)``
    per-frame offset (the joint-gain fit's ``c_n``; omitted or ``None``
    for the piston-only, ``fit_gain=False`` residual). The direct
    ``resid = I - c[:, None] - A @ X; sqrt(mean(resid**2))`` allocates a
    second full-size ``(N, P)`` array purely to reduce it to one scalar --
    for a large stack this is the single largest transient allocation in
    :func:`aia`, since it scales the same way ``I`` itself does. Streaming
    over pixel chunks with a float64 accumulator gives a bit-identical
    result at a small, fixed peak memory, and each chunk's matmul is small
    enough to stay well under a display-GPU's watchdog kernel-timeout.
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
    can recompute the same ``kappa_p``/``kappa_ps``/``predicted_rms``
    diagnostics for its own final, tilt-refined solution, using the exact
    formula ``aia`` itself uses -- see :class:`AIAParam` for what each field
    means and :func:`aia`'s Algorithm/Notes sections for the derivations.

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
        Per-frame residual offset from the joint-gain frame step (see
        :func:`aia_frame_step`), reported as :attr:`AIAParam.c_fit` and
        subtracted from ``I`` before measuring the residual used for
        ``predicted_rms``. ``None`` (the ``fit_gain=False`` default)
        reports an all-zero ``c_fit`` and leaves ``I`` uncorrected.

    Returns
    -------
    AIAParam
    """
    P = I.shape[1]

    # diagnostics (Chen & Kemao 2019): condition numbers of the two
    # normal matrices, and the accuracy they predict.
    A = _pixel_design(delta_fit, g, xp)
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

    With the per-frame phase steps ``delta_n`` (and contrasts ``g_n``) held
    fixed, each pixel's ``N`` samples follow::

        I_n = a + g_n * (u*cos(delta_n) + v*sin(delta_n))

    which is linear in ``(a, u, v)``, so every pixel is solved independently
    by one shared pseudoinverse of the ``(N, 3)`` design matrix. Here
    ``u = b*cos(phi)``, ``v = -b*sin(phi)`` for fringe amplitude ``b`` and
    phase ``phi``; recovering ``phi`` requires this to be paired with an
    estimate of ``delta_n`` from elsewhere (e.g. :func:`aia_frame_step`).

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


def aia_frame_step(stack: np.ndarray, u: np.ndarray, v: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame least-squares solve for each frame's phase step and gain.

    With the quadrature fields ``(u, v)`` held fixed, each frame's ``P``
    pixels follow ``I_n ~ c_n + P_n*u + Q_n*v`` for a free per-frame offset
    ``c_n`` and coefficients ``(P_n, Q_n) = g_n*(cos(delta_n), sin(delta_n))``,
    from which ``delta_n = atan2(Q_n, P_n)`` and ``g_n = hypot(P_n, Q_n)``.
    This is the transpose of :func:`aia_pixel_step`'s solve: a linear
    regression across pixels instead of across frames.

    ``c_n`` is left free rather than fixed to (or subtracted as) the
    background field ``a`` from the pixel step: it absorbs the background
    and any frame-to-frame brightness drift, and is re-derived from the raw
    data on every call rather than carrying forward a mid-iteration estimate
    of ``a`` -- feeding that estimate's own error back into this fit was
    found empirically to be less robust. Callers that want a single joint
    objective across both steps (see :func:`aia`'s ``fit_gain`` path) instead
    feed this ``c_n`` *back into* the next pixel step, rather than folding it
    into ``a`` here.

    The returned phase steps are absolute: this function does not resolve
    the model's phase-origin ambiguity (a constant that can be added to
    every ``delta_n``), so a caller comparing or iterating on ``delta``
    should pin it to a reference (e.g. ``delta[0] = 0``) itself. Likewise
    ``g_n`` is returned at whatever scale ``(u, v)`` happen to carry -- a
    caller that wants the ``median(g) = 1`` convention used elsewhere in
    this package (e.g. :func:`phase.utils.measure_frame_contrast`,
    :attr:`phase.solver.PhaseResult.g`) should normalize it itself.

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P`` pixels each.
    u, v : np.ndarray, shape (P,)
        Quadrature components, e.g. as returned by :func:`aia_pixel_step`.

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

    # Five scalar reductions build the (3,3) Gram matrix without ever
    # materializing a (P,3) design matrix -- the same pattern used for
    # kappa_ps in aia().
    Su, Sv = float(xp.sum(u64)), float(xp.sum(v64))
    Suu = float(xp.sum(u64 * u64))
    Svv = float(xp.sum(v64 * v64))
    Suv = float(xp.sum(u64 * v64))
    BtB = xp.asarray([[float(P), Su, Sv], [Su, Suu, Suv], [Sv, Suv, Svv]])

    IB = xp.stack([xp.sum(stack, axis=1).astype(xp.float64),
                    (stack @ u64).astype(xp.float64),
                    (stack @ v64).astype(xp.float64)], axis=1)      # (N,3)

    x = xp.linalg.solve(BtB, IB.T)                                  # (3,N)
    c, Pn, Qn = x[0], x[1], x[2]
    delta = xp.arctan2(Qn, Pn)
    g = xp.sqrt(Pn * Pn + Qn * Qn)
    return delta, g, c


def aia(stack: np.ndarray, g: np.ndarray, fit_gain: bool = False,
        delta0: Optional[np.ndarray] = None,
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
    and ``fit_gain`` parameters). For fixed ``delta_n`` and ``g_n`` this is
    linear in ``(a, u, v)``, and for fixed ``(u, v)`` it is linear in
    ``delta_n`` -- but not linear in both at once, so the unknowns are
    recovered by alternating least squares.

    Algorithm
    ---------
    Each iteration alternates two linear solves, until the largest
    per-frame update falls below ``tol`` (or ``iters`` is reached):

    1. **Pixel step** (:func:`aia_pixel_step`) -- with ``delta_n`` (and
       ``g_n``) fixed, recover the background ``a`` and quadrature
       components ``u, v``, from ``stack - c_n`` when ``fit_gain`` is True
       so this step and the frame step below minimize the same joint
       residual. The normal matrix of this solve is ``A_p`` (Chen &
       Kemao's notation).
    2. **Whiten** (:func:`_whiten_uv`, only when ``fit_gain`` is True) --
       rotate/shear ``(u, v)`` to equal pixel-sum energy and zero
       correlation. With ``g_n`` free, step 1's ``(u, v)`` is only
       determined up to an arbitrary invertible linear reparametrization
       (a gauge that plain, fixed-``g`` AIA does not have, since there
       ``(cos(delta_n), sin(delta_n))`` is already pinned to the unit
       circle); left uncorrected, the iteration can converge to a
       differently-fitting basis that reproduces the data just as well
       (or better) but is no longer the true phase. This step is exact
       and cost-free with respect to the objective below -- see its
       docstring.
    3. **Frame step** (:func:`aia_frame_step`) -- with ``(a, u, v)``
       fixed, recover each frame's phase step ``delta_n`` and, when
       ``fit_gain`` is True, its contrast ``g_n`` (normalized so
       ``median(g) = 1``) and residual offset ``c_n`` (gauge-fixed so
       ``mean(c) = 0``). The normal matrix of this solve is ``A_ps``.

    Phase steps are re-referenced to frame 0 (``delta[0] = 0``) every
    iteration, since the model has a phase-origin ambiguity that would
    otherwise let the iteration drift. With ``fit_gain=True`` the joint
    residual ``||stack - a - c - g*(u*cos(delta) + v*sin(delta))||`` then
    decreases monotonically every iteration -- steps 1 and 3 are exact
    least-squares minimizers of that one objective, and step 2 changes
    nothing about the achievable value of that objective (it only fixes
    which basis of the same subspace step 3 is handed) -- so a stall
    (``iters`` exhausted without ``converged``) is a genuine local optimum
    of the model, not the two steps chasing different targets.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted interferogram frames, already alpha-normalized and on
        the target device (:meth:`phase.solver.PhaseSolver.fit` does both
        before dispatching here).
    g : np.ndarray, shape (N,)
        Per-frame fringe contrast ``g_n`` (see Model). Used as the fixed,
        already-resolved contrast when ``fit_gain=False`` (all ones if gain
        estimation is disabled, ``PhaseConfig(gain_mode="none")``) and as
        the initial guess when ``fit_gain=True``.
    fit_gain : bool, default False
        If True, recover ``g_n`` jointly with ``delta_n`` in the frame step
        instead of holding it fixed at the input ``g`` -- the extension
        described under Algorithm above. Prefer this over an out-of-band
        gain estimate (e.g. :func:`phase.utils.measure_frame_contrast`,
        which assumes a spatial carrier and fails on circular or
        carrier-free fringes) whenever contrast drifts frame-to-frame.
    delta0 : np.ndarray, shape (N,), optional
        Initial guess for the phase step of each frame, in radians. If
        not given, defaults to evenly-spaced steps
        ``delta0[i] = i * 2*pi / N``, which minimizes ``kappa_ps`` (its
        theoretical lower bound is 2) and gives the most reliable
        convergence when the true phase-shift distribution is unknown.
    iters : int, default 30
        Maximum number of alternating least-squares iterations.
    tol : float, default 1e-4
        Convergence tolerance, in radians for ``delta`` and in ``g``'s own
        (median-1) units, on the largest per-frame change in ``delta`` (and,
        when ``fit_gain`` is True, in ``g``) between iterations (Chen &
        Kemao's recommended default for ``delta``).
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
    a, b, phi, delta, g : np.ndarray
        The Eq. (8) fields recovered by this method -- ``a`` and ``b`` shape
        ``(H, W)``, ``phi`` shape ``(H, W)`` wrapped to ``(-pi, pi]``,
        ``delta`` and ``g`` shape ``(N,)``. ``g`` is the input ``g``
        unchanged when ``fit_gain=False``, or the jointly fitted contrast
        (``median(g) = 1``) when True. Numpy or cupy arrays matching
        ``stack``'s array module -- not forced back to the host, so that
        chaining ``aia`` -> :func:`~phase.combine.combine_acquisitions` on a
        GPU doesn't round-trip large arrays over PCIe in between; call
        :func:`phase.backend.asnumpy` yourself when you need a
        guaranteed-numpy array.
    method_param : AIAParam
        ``kappa_p, kappa_ps, predicted_rms`` are diagnostics for whether
        this acquisition (frame count, phase-shift distribution, noise
        level) supports a trustworthy result; ``iters_run, converged``
        describe convergence; ``g_fit, c_fit, g_min_ratio`` describe the
        joint-gain fit (all zero/one when ``fit_gain=False``). See
        :class:`AIAParam`.

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
        # With fit_gain, both steps must minimize the same joint residual
        # (Eq. (8) plus the per-frame offset c_n) for the iteration to be a
        # true alternating least-squares descent -- so the pixel step sees
        # stack minus the frame step's own c_n, not the raw stack. Without
        # fit_gain this is a no-op (c stays zero throughout) and the loop
        # is bit-for-bit identical to the piston-only path.
        I_pixel = I - c_fit.astype(work_dtype)[:, None] if fit_gain else I
        a, u, v = aia_pixel_step(I_pixel, delta_fit, g_fit, dtype=work_dtype)
        if fit_gain:
            # Fixes a gauge freedom that only exists once g_n is free --
            # see _whiten_uv's docstring. Not needed when g is fixed: the
            # unit-circle constraint on (cos(delta), sin(delta)) already
            # pins this gauge in that case.
            u, v = _whiten_uv(u, v, xp)
        new_delta, new_g, new_c = aia_frame_step(I, u, v)

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

    # Rebuilt from delta_fit/g_fit/c_fit (what (a, u, v) were actually fit
    # against), not the updated delta/g/c from the final frame step.
    method_param = _aia_diagnostics(I, delta_fit, g_fit, a, u, v, N, xp, it + 1, converged,
                                     c=(c_fit if fit_gain else None))
    return a_map, b, phi, delta, g, method_param
