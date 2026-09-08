"""AIA with iterative, arbitrary-degree step-field refinement.

Builds on :mod:`phase.methods.aia`: :func:`fit_step_field` and
:func:`step_field_quality` are the per-frame step-field-coefficient
regression and its quality check derived in
``docs/step_field_residuals.md`` (its §8, Eqs. E1-E4); :func:`aia_step_field`
is the full solve that alternates the piston-only AIA pixel/frame step with
this fit until the fit stops improving. ``degree=1`` recovers the pure
linear-tilt model; higher degrees add curvature and beyond.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

from .. import backend as _backend
from ..backend import get_array_module
from .aia import AIAParam, _aia_diagnostics, _whiten_uv, aia, aia_frame_step, aia_pixel_step
from .base import MethodParam, _fmt_value


@lru_cache(maxsize=32)
def _poly_basis(H: int, W: int, degree: int, xp):
    """Orthonormal polynomial basis ``p_1..p_J`` of Eq. (T1), shape ``(J, H*W)``.

    Monomials of total degree ``1..degree`` in ``(x, y)`` on centered,
    unit-normalized coordinates, mean-subtracted (the field-mean-zero
    gauge, Eq. T3) and Gram-Schmidt-orthonormalized in ascending degree
    order -- see ``docs/step_field_residuals.md`` §1.2(i). ``degree=0``
    returns the empty ``(0, H*W)`` basis (no step field, the piston-only
    model). Cached per ``(H, W, degree, xp)``: :func:`aia_step_field` calls
    this every refinement iteration at the same shape/degree.

    Parameters
    ----------
    H, W : int
        Frame height and width.
    degree : int
        Highest total polynomial degree ``M`` to include, ``>= 0``.
    xp : module
        ``numpy`` or ``cupy``.

    Returns
    -------
    np.ndarray, shape (J, H*W), float64
        ``J = (degree+1)*(degree+2)//2 - 1`` basis rows (``J=0`` at
        ``degree=0``), flattened row-major to match ``stack``'s pixel
        flattening elsewhere in this package.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")

    yy, xx = xp.meshgrid(xp.arange(H, dtype=xp.float64), xp.arange(W, dtype=xp.float64),
                          indexing="ij")
    x = ((xx - xx.mean()) / max(W / 2.0, 1.0)).ravel()
    y = ((yy - yy.mean()) / max(H / 2.0, 1.0)).ravel()

    # Monomial exponent pairs (ex, ey) of total degree 1..degree, ascending
    # so lower-order terms are fixed before higher-order ones build on them.
    exponents = [(d - i, i) for d in range(1, degree + 1) for i in range(d + 1)]
    J = len(exponents)
    P = H * W

    basis = xp.empty((J, P), dtype=xp.float64)
    for j, (ex, ey) in enumerate(exponents):
        col = (x ** ex) * (y ** ey)
        col = col - col.mean()                          # Eq. (T3): zero field mean
        for k in range(j):                               # modified Gram-Schmidt
            col = col - (col @ basis[k]) * basis[k]
        norm = float(xp.sqrt(xp.sum(col * col)))
        basis[j] = col / norm
    return basis


def _cond_batch(M, xp):
    """2-norm condition number of a batch of square matrices, shape ``(..., k)``.

    Batched analogue of :func:`phase.methods.aia._cond3`; a singular batch
    element reports ``inf`` rather than dividing by zero.
    """
    s = xp.linalg.svd(M, compute_uv=False)
    smin = s.min(axis=-1)
    smax = s.max(axis=-1)
    return xp.where(smin > 0, smax / xp.where(smin > 0, smin, 1.0), xp.inf)


def fit_step_field(stack: np.ndarray, a: np.ndarray, u: np.ndarray, v: np.ndarray,
                    delta: np.ndarray, basis: np.ndarray,
                    g: Optional[np.ndarray] = None,
                    chunk: int = 1_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Fit each frame's step-field coefficients from the AIA pixel-step residual.

    A spatially-varying phase-step error leaves a first-order residual in
    the piston-model AIA fit, linear in each frame's coefficients
    ``c_1n..c_Jn`` -- the transpose of :func:`aia_pixel_step`'s per-pixel
    regression across frames. See ``docs/step_field_residuals.md`` §8,
    Eq. (E1) for the derivation (``G^(n) c = -h^(n)`` solved independently
    per frame).

    ``G^(n)`` is built from the basis's pairwise products in pixel chunks,
    rather than materializing the full pair-product array at once -- the
    same idiom :func:`phase.methods.aia._chunked_sigma` uses, for the same
    reason.

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P = H*W`` pixels each.
    a, u, v : np.ndarray, shape (P,)
        Background and quadrature components of the piston-model AIA
        solution, e.g. as returned by :func:`aia_pixel_step`.
    delta : np.ndarray, shape (N,)
        Piston phase step of each frame, in radians.
    basis : np.ndarray, shape (J, P)
        Step-field basis, e.g. from :func:`_poly_basis`.
    g : np.ndarray, shape (N,), optional
        Per-frame fringe contrast, as used in the pixel-step solution.
        Defaults to all ones (no frame-to-frame contrast variation).
    chunk : int, default 1_000_000
        Pixels processed per chunk while accumulating ``G^(n)``, ``h^(n)``.

    Returns
    -------
    coeffs : np.ndarray, shape (J, N), float64
        Per-frame step-field coefficients ``c_jn``.
    cond : np.ndarray, shape (N,), float64
        Per-frame condition number of ``G^(n)`` (Eq. E3) -- a large value
        flags that frame's fit as unreliable regardless of how small the
        resulting residual looks.
    """
    if len(stack.shape) != 2:
        raise ValueError(f"Stack shape must be have 2 dims, but got {len(stack.shape)}")
    if a.shape != u.shape or a.shape != v.shape:
        raise ValueError("a, u, and v must have the same shape")
    if len(a.shape) != 1 or a.shape[0] != stack.shape[1]:
        raise ValueError("a, u, and v must be 1-D with length equal to stack's second dimension")
    if len(delta.shape) != 1 or delta.shape[0] != stack.shape[0]:
        raise ValueError("delta must be 1-D with length equal to stack's first dimension")
    if len(basis.shape) != 2 or basis.shape[1] != stack.shape[1]:
        raise ValueError("basis must be 2-D with second dimension equal to stack's second dimension")
    if g is not None and len(g) != len(delta):
        raise ValueError("g must have the same length as delta")

    xp = get_array_module(stack, a, u, v, delta, basis)
    N = stack.shape[0]
    P = stack.shape[1]
    J = basis.shape[0]
    delta = xp.asarray(delta, dtype=xp.float64)
    g = xp.ones(N, dtype=xp.float64) if g is None else xp.asarray(g, dtype=xp.float64)
    c, s = xp.cos(delta), xp.sin(delta)

    # Unique (j, j') pairs (j<=j') covering G's upper triangle, in plain
    # Python since J is small (a handful of basis terms at most).
    iu = [jj for jj in range(J) for _ in range(jj, J)]
    ju = [kk for jj in range(J) for kk in range(jj, J)]
    iu, ju = xp.asarray(iu), xp.asarray(ju)
    K = len(iu)

    G_flat = xp.zeros((N, K), dtype=xp.float64)
    h = xp.zeros((N, J), dtype=xp.float64)

    for start in range(0, P, chunk):
        sl = slice(start, start + chunk)
        a_c, u_c, v_c = a[sl], u[sl], v[sl]
        basis_c = basis[:, sl]                                          # (J, Pc)
        stack_c = stack[:, sl].astype(xp.float64)                       # (N, Pc)

        model_c = a_c[None, :] + g[:, None] * (xp.outer(c, u_c) + xp.outer(s, v_c))
        resid_c = stack_c - model_c                                     # (N, Pc)
        w_c = g[:, None] * (xp.outer(s, u_c) - xp.outer(c, v_c))        # (N, Pc)

        pp_c = basis_c[iu, :] * basis_c[ju, :]                          # (K, Pc)
        G_flat += (w_c * w_c) @ pp_c.T                                  # (N, K)
        h += (w_c * resid_c) @ basis_c.T                                # (N, J)

    G = xp.zeros((N, J, J), dtype=xp.float64)
    G[:, iu, ju] = G_flat
    G[:, ju, iu] = G_flat
    cond = _cond_batch(G, xp)

    # xp.linalg.solve's batched gufunc needs rhs's last two axes as its own
    # (m, n) core dims -- the singleton axis is squeezed back off after.
    sol = xp.linalg.solve(G, -h[..., None])[..., 0]                     # (N, J)
    return sol.T, cond


def step_field_quality(stack: np.ndarray, a: np.ndarray, u: np.ndarray, v: np.ndarray,
                        delta: np.ndarray, coeffs: np.ndarray, basis: np.ndarray,
                        H: int, W: int, g: Optional[np.ndarray] = None,
                        crop: int = 100, precise_reduce: bool = True) -> Tuple[float, np.ndarray]:
    """Measure how much of the AIA residual the fitted step field actually explains.

    Reconstructs each frame at the step-corrected phase step
    ``delta_n + coeffs[:,n] @ basis`` and reports the ratio of that
    residual's RMS to the raw data's RMS, over a border-cropped region (the
    fit is least reliable near the edges). Intended to be called once per
    refinement iteration alongside :func:`fit_step_field`: ``rms_frac``
    dropping iteration to iteration is the signal that the fitted step
    field is a real correction rather than fitted noise.

    Parameters
    ----------
    stack : np.ndarray, shape (N, P)
        Interferogram frames flattened to ``P = H*W`` pixels each.
    a, u, v : np.ndarray, shape (P,)
        Background and quadrature components of the piston-model AIA
        solution, as passed to :func:`fit_step_field`.
    delta : np.ndarray, shape (N,)
        Piston phase step of each frame, in radians.
    coeffs : np.ndarray, shape (J, N)
        Per-frame step-field coefficients, e.g. as returned by
        :func:`fit_step_field`.
    basis : np.ndarray, shape (J, P)
        Step-field basis matching ``coeffs``, e.g. from :func:`_poly_basis`.
    H, W : int
        Frame height and width, ``P = H*W``.
    g : np.ndarray, shape (N,), optional
        Per-frame fringe contrast, as used in the pixel-step solution.
        Defaults to all ones (no frame-to-frame contrast variation).
    crop : int, default 100
        Pixels excluded from each edge of the field before computing either
        RMS. ``crop=0`` compares over the full field.
    precise_reduce : bool, default True
        See :attr:`phase.solver.PhaseConfig.precise_reduce`. Controls the
        dtype of every operand feeding the ``(N, P)`` reconstruction below,
        and hence of the returned ``resid`` itself.

    Returns
    -------
    rms_frac : float
        RMS of the step-corrected-model residual divided by the RMS of
        ``stack``, both over the cropped region.
    resid : np.ndarray, shape (N, P)
        The step-corrected-model residual (uncropped), float64 if
        ``precise_reduce`` (the default) else ``stack``'s own dtype.
    """
    if len(stack.shape) != 2:
        raise ValueError(f"Stack shape must be have 2 dims, but got {len(stack.shape)}")
    if a.shape != u.shape or a.shape != v.shape:
        raise ValueError("a, u, and v must have the same shape")
    if len(a.shape) != 1 or a.shape[0] != stack.shape[1]:
        raise ValueError("a, u, and v must be 1-D with length equal to stack's second dimension")
    if len(delta.shape) != 1 or delta.shape[0] != stack.shape[0]:
        raise ValueError("delta must be 1-D with length equal to stack's first dimension")
    if coeffs.shape[1] != len(delta):
        raise ValueError("coeffs' second dimension must equal len(delta)")
    if basis.shape[0] != coeffs.shape[0] or basis.shape[1] != stack.shape[1]:
        raise ValueError("basis must have shape (coeffs.shape[0], stack.shape[1])")
    if stack.shape[1] != H * W:
        raise ValueError(f"stack's second dimension ({stack.shape[1]}) must equal H*W ({H * W})")
    if g is not None and len(g) != len(delta):
        raise ValueError("g must have the same length as delta")
    if crop < 0 or 2 * crop >= H or 2 * crop >= W:
        raise ValueError(f"crop={crop} leaves no pixels for frame size {H}x{W}")

    xp = get_array_module(stack, a, u, v, delta, coeffs, basis)
    N = stack.shape[0]
    # Every operand feeding Delta_field/dn/model must already be at
    # calc_dtype before those (N, P) arrays are built -- casting resid
    # afterward is too late. basis is always float64 from _poly_basis, and
    # u/v are already float64 in aia_step_field's actual usage, so without
    # this they'd force float64 here regardless of precise_reduce.
    calc_dtype = xp.float64 if precise_reduce else stack.dtype
    delta = xp.asarray(delta, dtype=calc_dtype)
    coeffs = xp.asarray(coeffs, dtype=calc_dtype)
    g = xp.ones(N, dtype=calc_dtype) if g is None else xp.asarray(g, dtype=calc_dtype)
    basis = basis.astype(calc_dtype, copy=False)
    a = a.astype(calc_dtype, copy=False)
    u = u.astype(calc_dtype, copy=False)
    v = v.astype(calc_dtype, copy=False)

    Delta_field = coeffs.T @ basis                                       # (N, P) in calc_dtype
    dn = delta[:, None] + Delta_field                                    # (N, P) corrected step
    model = a[None, :] + g[:, None] * (u[None, :] * xp.cos(dn) + v[None, :] * xp.sin(dn))
    resid = stack - model                                                # calc_dtype (stack promotes to it if needed)

    # Border crop as a boolean mask over the flattened field -- crop=0
    # leaves it all-True (`mask[-0:] = False` would zero the whole field,
    # since Python's -0 == 0, so this is guarded by `if crop > 0`).
    mask = xp.ones((H, W), dtype=bool)
    if crop > 0:
        mask[:crop] = mask[-crop:] = mask[:, :crop] = mask[:, -crop:] = False
    mask = mask.ravel()

    if precise_reduce:
        rms_frac = float(xp.std(resid[:, mask].astype(xp.float64))
                          / xp.std(stack[:, mask].astype(xp.float64)))
    else:
        rms_frac = float(xp.std(resid[:, mask]) / xp.std(stack[:, mask]))
    return rms_frac, resid


@dataclass
class StepFieldParam(MethodParam):
    """Diagnostics for :func:`aia_step_field`'s AIA-with-step-field-refinement solve.

    Attributes
    ----------
    aia_param : AIAParam
        ``kappa_p``, ``kappa_ps``, ``predicted_rms`` recomputed against the
        *final*, step-field-refined ``(a, u, v, delta)``. ``iters_run``/
        ``converged`` instead describe the initial
        :func:`phase.methods.aia.aia` call's own loop -- the outer
        refinement loop has its own ``refine_iters_run``/``refine_converged``
        below.
    degree : int
        Highest total polynomial degree ``M`` fit (see :func:`_poly_basis`);
        ``1`` is a pure linear tilt.
    coeffs : np.ndarray, shape (J, N)
        Per-frame step-field coefficients ``c_jn`` from the *best* round
        (see ``best_iter``), fit against the ``(a, u, v, delta)`` this
        result reports. Raw per-frame least-squares fit, not gauge-fixed
        (``docs/step_field_residuals.md`` Eq. T3b) -- subtract
        ``coeffs.mean(axis=1, keepdims=True)`` yourself before reading a
        row as physical per-frame drift.
    coeffs_rms : np.ndarray, shape (J,)
        RMS of each row of ``coeffs`` across frames -- a quick "was there
        meaningful step-field error, and in which order" summary.
    kappa_fit : float
        ``max`` over frames of the per-frame fit's condition number
        (Eq. E3) -- large values flag that ``degree`` has outrun what the
        recorded fringe pattern can resolve, even if ``rms_frac`` looks good.
    rms_frac : float
        The best round's :func:`step_field_quality` value
        (``== min(rms_frac_history)``).
    rms_frac_history : list of float
        ``rms_frac`` at every refinement iteration, in order, including any
        round that made it worse.
    refine_iters_run : int
        Number of refinement iterations actually run.
    refine_converged : bool
        Whether the loop stopped because a round's non-negative
        round-over-round improvement in ``rms_frac`` fell below
        ``refine_tol`` (True) or ``refine_iters`` was exhausted (False).
    best_iter : int
        0-indexed round that ``coeffs``/``kappa_fit``/``rms_frac`` were
        taken from -- not necessarily the last one run. ``-1`` if
        ``refine_iters=0``.
    precise_reduce : bool
        See :attr:`phase.solver.PhaseConfig.precise_reduce`. Carried here
        (not just as a call argument) so :meth:`phase_step_field`, called
        generically by :meth:`phase.solver.PhaseSolver.fit`, can honor it.
    work_dtype
        The working dtype ``aia_step_field`` actually solved in -- what
        :meth:`phase_step_field` casts down to when ``precise_reduce`` is
        False.
    """

    aia_param: AIAParam
    degree: int
    coeffs: np.ndarray
    coeffs_rms: np.ndarray
    kappa_fit: float
    rms_frac: float
    rms_frac_history: List[float]
    refine_iters_run: int
    refine_converged: bool
    best_iter: int
    precise_reduce: bool
    work_dtype: object

    def print_summary(self) -> None:
        """Delegate to the inner AIAParam, then print the refinement diagnostics."""
        self.aia_param.print_summary()
        print(f"degree:           {_fmt_value(self.degree)}")
        print(f"refine_converged: {_fmt_value(self.refine_converged)}")
        print(f"refine_iters_run: {_fmt_value(self.refine_iters_run)}")
        print(f"best_iter:        {_fmt_value(self.best_iter)}")
        print(f"rms_frac:         {_fmt_value(self.rms_frac)}")
        print(f"kappa_fit:        {_fmt_value(self.kappa_fit)}")
        print(f"coeffs_rms:       {_fmt_value(self.coeffs_rms)}")

    def phase_step_field(self, delta, H, W, xp):
        """Piston ``delta_n`` plus the fitted per-frame step field ``coeffs[:,n] @ basis``.

        Overrides :meth:`phase.methods.base.MethodParam.phase_step_field`'s
        plain broadcast so :meth:`phase.solver.PhaseSolver.fit`'s
        reconstruction check sees the spatially-varying phase step this
        method recovers. Honors ``self.precise_reduce`` exactly as
        :func:`step_field_quality` does (float64 vs. ``self.work_dtype``).
        """
        calc_dtype = xp.float64 if self.precise_reduce else self.work_dtype
        basis = _poly_basis(H, W, self.degree, xp).astype(calc_dtype, copy=False)  # (J, P)
        coeffs = xp.asarray(self.coeffs, dtype=calc_dtype)
        N = delta.shape[0]
        field = delta[:, None] + coeffs.T @ basis                       # (N, P)
        return field.reshape(N, H, W)


def aia_step_field(stack: np.ndarray, g: np.ndarray, fit_gain: bool = False,
                    delta0: Optional[np.ndarray] = None,
                    iters: int = 30, tol: float = 1e-4, dtype=None,
                    degree: int = 1, refine_iters: int = 5, refine_tol: float = 1e-3,
                    crop: int = 100, precise_reduce: bool = True):
    """Advanced Iterative Algorithm with iterative, arbitrary-degree step-field refinement.

    Runs the piston-only :func:`phase.methods.aia.aia` to convergence, then
    alternates fitting the per-frame step-field residual
    (:func:`fit_step_field`), scoring it (:func:`step_field_quality`), and
    removing its estimated contribution from the *original* stack before
    re-running the pixel/frame step -- until the round-over-round
    improvement in ``rms_frac`` falls below ``refine_tol`` or
    ``refine_iters`` is spent. See ``docs/step_field_residuals.md`` §8.2
    ("The algorithm") for the full step-by-step derivation, and
    ``docs/aia.md`` for the inner piston-only solve.

    Each round's fitted coefficients are gauge-fixed (subtracting each
    basis term's frame mean, Eq. T3b/E4) before being used to correct the
    data, so the static part of any term is left for the next pixel/frame
    step to absorb into ``Phi`` instead. The *reported* ``coeffs`` (see
    :class:`StepFieldParam`) are the raw, non-gauge-fixed fit.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted interferogram frames -- see :func:`phase.methods.aia.aia`.
    g : np.ndarray, shape (N,)
        Per-frame fringe contrast -- see :func:`phase.methods.aia.aia`.
    fit_gain : bool, default False
        If True, recover ``g`` jointly rather than holding it fixed --
        forwarded to the initial :func:`phase.methods.aia.aia` call and
        kept fitted (re-estimated each refinement round) throughout.
    delta0, iters, tol, dtype
        Passed through to the initial :func:`phase.methods.aia.aia` call.
    degree : int, default 1
        Highest total polynomial degree to fit the step field to (see
        :func:`_poly_basis`); ``0`` disables the step-field correction
        entirely, returning the plain-``aia`` result. ``1`` is a pure
        linear tilt (registered separately as ``"aia_tilt"``), ``2`` adds
        curvature. ``docs/step_field_residuals.md`` §7 recommends keeping
        this small (2-3).
    refine_iters : int, default 5
        Maximum number of refinement rounds. ``0`` skips refinement
        entirely, returning the plain ``aia`` result (``coeffs`` all zero).
    refine_tol : float, default 1e-3
        Stop refining once a round's non-negative round-over-round drop in
        :func:`step_field_quality`'s ``rms_frac`` is below this. A round
        that makes ``rms_frac`` worse keeps the loop running but is never
        returned -- see :attr:`StepFieldParam.best_iter`.
    crop : int, default 100
        Pixels excluded from each edge of the field when computing
        ``rms_frac`` (see :func:`step_field_quality`).
    precise_reduce : bool, default True
        See :attr:`phase.solver.PhaseConfig.precise_reduce`. Forwarded to
        the initial ``aia`` call and to every ``aia_frame_step``/
        ``step_field_quality`` call in the refinement loop below.

    Returns
    -------
    a, b, phi, delta, g, method_param
        Same contract as :func:`phase.methods.aia.aia`; ``method_param`` is
        a :class:`StepFieldParam`.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")

    xp = get_array_module(stack)
    N, H, W = stack.shape
    work_dtype = dtype if dtype is not None else _backend.default_dtype(xp)
    I = stack.reshape(N, -1).astype(work_dtype, copy=False)        # (N, P)
    g = xp.asarray(g, dtype=xp.float64)

    a_map, b0, phi0, delta, g, aia_param0 = aia(stack, g, fit_gain=fit_gain,
                                                 delta0=delta0, iters=iters, tol=tol, dtype=dtype,
                                                 precise_reduce=precise_reduce)
    a = a_map.reshape(-1)
    u = (b0 * xp.cos(phi0)).reshape(-1)
    v = (-b0 * xp.sin(phi0)).reshape(-1)
    # (a, u, v) above were fit against I minus aia()'s own c_fit (zero when
    # fit_gain=False) -- the step-field regression below must match.
    c = aia_param0.c_fit

    basis = _poly_basis(H, W, degree, xp)                            # (J, P)
    J = basis.shape[0]
    coeffs = xp.zeros((J, N), dtype=xp.float64)
    kappa_fit = float("nan")
    rms_frac = float("nan")
    rms_history: List[float] = []
    prev_rms = None
    refine_converged = False
    best = None            # (a, u, v, delta, g, c, coeffs, kappa_fit, rms_frac) of the best round
    best_iter = -1
    it = -1

    # degree=0 -> J=0: no step field to fit, so this loop is skipped and the
    # plain aia() result passes through unchanged (same as refine_iters=0).
    for it in range(refine_iters if J > 0 else 0):
        Ic = I - c.astype(work_dtype)[:, None] if fit_gain else I
        coeffs_it, cond = fit_step_field(Ic, a, u, v, delta, basis, g=g)
        kappa_it = float(xp.max(cond))
        rms_frac_it, _ = step_field_quality(Ic, a, u, v, delta, coeffs_it, basis, H, W, g=g, crop=crop,
                                             precise_reduce=precise_reduce)
        rms_history.append(rms_frac_it)

        # (a, u, v, delta, g, c) are the values *before* this round's
        # correction -- the state coeffs_it/rms_frac_it were actually fit
        # and scored against, so the snapshot is self-consistent.
        if best is None or rms_frac_it < best[-1]:
            best = (a, u, v, delta, g, c, coeffs_it, kappa_it, rms_frac_it)
            best_iter = it

        # Only a non-negative improvement counts as converged -- a worse
        # round must not look "converged" just because the drop is negative.
        if prev_rms is not None and 0 <= (prev_rms - rms_frac_it) < refine_tol:
            refine_converged = True
            break
        prev_rms = rms_frac_it

        coeffs_fixed = coeffs_it - coeffs_it.mean(axis=1, keepdims=True)  # gauge-fix (Eq. T3b/E4)

        # Correct the *original* stack (not a running buffer -- each round's
        # coeffs estimate the total step field, not an increment). Eq. (E1)
        # fits resid ~= -w*Delta, so recovering it means adding it back.
        cd, sd = xp.cos(delta), xp.sin(delta)
        w = g[:, None] * (xp.outer(sd, u) - xp.outer(cd, v))
        Delta_field = coeffs_fixed.T @ basis                          # (N, P)
        corrected = I + w * Delta_field                               # raw, for the frame step
        corrected_Ic = corrected - c.astype(work_dtype)[:, None] if fit_gain else corrected

        a, u, v = aia_pixel_step(corrected_Ic, delta, g, dtype=work_dtype)
        if fit_gain:
            u, v = _whiten_uv(u, v, xp)
        new_delta, new_g, new_c = aia_frame_step(corrected, u, v, precise_reduce=precise_reduce)
        delta = new_delta - new_delta[0]
        if fit_gain:
            c = new_c - xp.mean(new_c)                                # gauge fix
            g = new_g / max(float(xp.median(new_g)), np.finfo(float).eps)

    refine_iters_run = it + 1

    if best is not None:
        a, u, v, delta, g, c, coeffs, kappa_fit, rms_frac = best

    aia_param = _aia_diagnostics(I, delta, g, a, u, v, N, xp,
                                  aia_param0.iters_run, aia_param0.converged,
                                  c=(c if fit_gain else None))
    coeffs_rms = xp.sqrt(xp.mean(coeffs ** 2, axis=1))
    method_param = StepFieldParam(
        aia_param=aia_param, degree=degree, coeffs=coeffs, coeffs_rms=coeffs_rms,
        kappa_fit=kappa_fit, rms_frac=rms_frac, rms_frac_history=rms_history,
        refine_iters_run=refine_iters_run, refine_converged=refine_converged,
        best_iter=best_iter, precise_reduce=precise_reduce, work_dtype=work_dtype,
    )

    phi = xp.arctan2(-v, u).reshape(H, W)
    u64, v64 = u.astype(xp.float64), v.astype(xp.float64)
    b = xp.sqrt(u64 ** 2 + v64 ** 2).reshape(H, W)
    a_map = a.reshape(H, W)
    return a_map, b, phi, delta, g, method_param
