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
from .aia import AIAParam, _aia_diagnostics, aia, aia_frame_step, aia_pixel_step
from .base import MethodParam, _fmt_value


@lru_cache(maxsize=32)
def _poly_basis(H: int, W: int, degree: int, xp):
    """Orthonormal polynomial basis ``p_1..p_J`` of Eq. (T1), shape ``(J, H*W)``.

    Enumerates the monomials of total degree ``1..degree`` in ``(x, y)`` on
    centered, unit-normalized coordinates (``x, y`` each range roughly over
    ``[-1, 1]``, keeping higher-degree terms from blowing up in magnitude),
    mean-subtracts each one (the field-mean-zero gauge, Eq. T3), then
    modified-Gram-Schmidt-orthonormalizes them in ascending degree order
    against the field's pixel inner product ``sum_{x,y} f*g``. Mean-zero is
    preserved under Gram-Schmidt (the difference of two zero-mean vectors is
    zero-mean), so every returned row still satisfies Eq. (T3) exactly.

    At ``degree=1`` this returns (unit-normalized) ``x, y`` themselves: on a
    centered rectangular grid ``sum_{x,y} x*y = 0`` already (the grid is
    separable), so the two are already orthogonal and Gram-Schmidt is a
    no-op beyond normalization. At ``degree=0`` there are no degree-``>=1``
    monomials to include (``docs/step_field_residuals.md`` Eq. T1's basis
    starts at degree 1 by construction -- degree 0 is the piston, handled
    separately), so this returns the empty ``(0, H*W)`` basis: no step
    field at all, the piston-only model.

    Orthonormalizing (rather than using the raw Taylor monomials directly)
    only changes the numerical conditioning of the per-frame regression in
    :func:`fit_step_field` -- it does not change what the fit can represent,
    since it spans the identical space of degree-``<=degree`` polynomials.

    Cached per ``(H, W, degree, xp)``: the outer refinement loop in
    :func:`aia_step_field` calls this once per iteration at the same shape
    and degree, and Gram-Schmidt is ``O(J^2 * H*W)``, not free to redo.

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
        ``degree=0``), each flattened row-major to match ``stack``'s pixel
        flattening elsewhere in this package.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")

    yy, xx = xp.meshgrid(xp.arange(H, dtype=xp.float64), xp.arange(W, dtype=xp.float64),
                          indexing="ij")
    x = ((xx - xx.mean()) / max(W / 2.0, 1.0)).ravel()
    y = ((yy - yy.mean()) / max(H / 2.0, 1.0)).ravel()

    # (ex, ey) exponent pairs for every monomial x^ex * y^ey of total degree
    # 1..degree, ascending degree so lower-order terms are orthonormalized
    # (and hence fixed) before higher-order ones are built on top of them.
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

    Batched analogue of :func:`phase.methods.aia._cond3`; a matrix whose
    smallest singular value is exactly zero (or the batch element is
    otherwise singular) reports ``inf`` rather than dividing by zero.
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

    ``docs/step_field_residuals.md`` §8 shows that a spatially-varying
    phase-step error ``delta_n(x,y) = delta_n + sum_j c_jn*p_j(x,y)`` leaves
    a first-order residual in the piston-model AIA fit (its Eq. T7)::

        r_n = I_n - (a + g_n*(u*cos(delta_n) + v*sin(delta_n))) ~= -w_n * sum_j c_jn*p_j

    with ``w_n = u*sin(delta_n) - v*cos(delta_n)``. This is linear in
    ``c_1n..c_Jn`` for each frame, so -- the transpose of
    :func:`aia_pixel_step`'s per-pixel regression across frames -- each
    frame's coefficients are recovered by an independent weighted linear
    regression across pixels, onto the ``J`` basis images ``w_n * p_j``
    (Eq. E1): ``G^(n) c = -h^(n)`` with ``G^(n)_jj' = sum_xy w_n^2*p_j*p_j'``,
    ``h^(n)_j = sum_xy w_n*r_n*p_j``.

    ``G^(n)`` is built from the basis's pairwise products in pixel chunks
    (rather than materializing the full ``(H*W, J*(J+1)/2)`` pair-product
    array at once, which for ``J=5`` on a several-megapixel field is a few
    hundred MB) -- the same reduction-via-matmul idiom
    :func:`phase.methods.aia._chunked_sigma` uses for the same reason.

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

    # Unique (j, j') pairs (j<=j') covering G's upper triangle -- computed in
    # plain Python (J is small, at most a handful of basis terms) rather
    # than relying on a particular xp providing triu_indices.
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

    # xp.linalg.solve's batched gufunc requires rhs's last two axes to be
    # its own (m, n) core dims, not a trailing vector -- the explicit
    # singleton axis is squeezed back off after solving.
    sol = xp.linalg.solve(G, -h[..., None])[..., 0]                     # (N, J)
    return sol.T, cond


def step_field_quality(stack: np.ndarray, a: np.ndarray, u: np.ndarray, v: np.ndarray,
                        delta: np.ndarray, coeffs: np.ndarray, basis: np.ndarray,
                        H: int, W: int, g: Optional[np.ndarray] = None,
                        crop: int = 100) -> Tuple[float, np.ndarray]:
    """Measure how much of the AIA residual the fitted step field actually explains.

    Reconstructs each frame with the step-corrected phase step
    ``delta_n + coeffs[:,n] @ basis`` in place of the piston-only
    ``delta_n`` (see :func:`fit_step_field` and
    ``docs/step_field_residuals.md``), and reports the ratio of that
    residual's RMS to the raw data's RMS, both computed over a
    border-cropped region (the fit and any smoothing artifacts are least
    reliable near the edges). Intended to be called once per refinement
    iteration, alongside :func:`fit_step_field`: ``rms_frac`` dropping from
    one iteration to the next is the signal that the fitted step field is a
    real correction rather than fitted noise.

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

    Returns
    -------
    rms_frac : float
        RMS of the step-corrected-model residual divided by the RMS of
        ``stack``, both over the cropped region.
    resid : np.ndarray, shape (N, P)
        The step-corrected-model residual (uncropped), in ``stack``'s dtype.
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
    delta = xp.asarray(delta, dtype=xp.float64)
    coeffs = xp.asarray(coeffs, dtype=xp.float64)
    g = xp.ones(N, dtype=xp.float64) if g is None else xp.asarray(g, dtype=xp.float64)

    Delta_field = coeffs.T @ basis                                       # (N, P)
    dn = delta[:, None] + Delta_field                                    # (N, P) corrected step
    model = a[None, :] + g[:, None] * (u[None, :] * xp.cos(dn) + v[None, :] * xp.sin(dn))
    resid = stack - model

    # Border crop, as a boolean mask over the flattened field -- crop=0
    # leaves the mask all-True (the naive `mask[-0:] = False` would instead
    # zero out the whole field, since Python's -0 == 0).
    mask = xp.ones((H, W), dtype=bool)
    if crop > 0:
        mask[:crop] = mask[-crop:] = mask[:, :crop] = mask[:, -crop:] = False
    mask = mask.ravel()

    rms_frac = float(xp.std(resid[:, mask].astype(xp.float64))
                      / xp.std(stack[:, mask].astype(xp.float64)))
    return rms_frac, resid


@dataclass
class StepFieldParam(MethodParam):
    """Diagnostics for :func:`aia_step_field`'s AIA-with-step-field-refinement solve.

    Attributes
    ----------
    aia_param : AIAParam
        ``kappa_p``, ``kappa_ps``, and ``predicted_rms`` recomputed against
        the *final*, step-field-refined ``(a, u, v, delta)`` -- not the
        initial piston-only solve. ``iters_run`` and ``converged`` instead
        describe that initial :func:`phase.methods.aia.aia` call's own
        alternating pixel/frame-step loop (see
        :class:`phase.methods.aia.AIAParam`); the outer refinement loop has
        its own ``refine_iters_run`` and ``refine_converged`` below.
    degree : int
        Highest total polynomial degree ``M`` fit (see :func:`_poly_basis`);
        ``1`` is a pure linear tilt.
    coeffs : np.ndarray, shape (J, N)
        Per-frame step-field coefficients ``c_jn`` from the *best* round
        (see ``best_iter`` below), fit against the ``(a, u, v, delta)`` this
        result reports -- not necessarily the last round run: a round that
        makes ``rms_frac`` worse is kept in ``rms_frac_history`` but never
        used for the reported fields. These are the raw per-frame
        least-squares fit, not gauge-fixed (``docs/step_field_residuals.md``
        Eq. T3b) -- each row's frame mean can carry an arbitrary offset
        degenerate with ``phi``; subtract ``coeffs.mean(axis=1,
        keepdims=True)`` yourself before interpreting a row as a physical
        per-frame drift.
    coeffs_rms : np.ndarray, shape (J,)
        RMS of each row of ``coeffs`` across frames -- a quick "was there
        meaningful step-field error at all, and in which order" summary,
        independent of the per-frame detail in ``coeffs`` itself.
    kappa_fit : float
        ``max`` over frames of the per-frame fit's condition number
        (Eq. E3) -- large values flag that ``degree`` has outrun what the
        recorded fringe pattern can resolve, even if ``rms_frac`` still
        looks good.
    rms_frac : float
        The best round's :func:`step_field_quality` value (``==
        min(rms_frac_history)``): RMS of the step-corrected-model residual
        as a fraction of the raw data's RMS, over the cropped region.
    rms_frac_history : list of float
        ``rms_frac`` at every refinement iteration, in order, including any
        round that made it worse -- lets a caller inspect the full
        convergence trace, not just the best round's value.
    refine_iters_run : int
        Number of refinement iterations actually run.
    refine_converged : bool
        Whether the refinement loop stopped because a round's non-negative
        round-over-round improvement in ``rms_frac`` fell below
        ``refine_tol`` (True) or because ``refine_iters`` was exhausted
        without that happening (False). A round that made ``rms_frac``
        worse never sets this True on its own -- see ``best_iter``.
    best_iter : int
        0-indexed round that ``coeffs``/``kappa_fit``/``rms_frac`` were
        taken from -- the round with the lowest ``rms_frac_history`` entry,
        not necessarily the last one run. ``-1`` if ``refine_iters=0``.
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
        plain broadcast -- this is what lets
        :meth:`phase.solver.PhaseSolver.fit`'s reconstruction check see the
        spatially-varying phase step this method actually recovers.
        Reconstructs the basis via :func:`_poly_basis` from ``(H, W,
        self.degree)`` rather than storing it on this dataclass -- the
        basis is deterministic from those three values and cached there.
        """
        basis = _poly_basis(H, W, self.degree, xp)                      # (J, P)
        coeffs = xp.asarray(self.coeffs, dtype=xp.float64)
        N = delta.shape[0]
        field = delta[:, None] + coeffs.T @ basis                       # (N, P)
        return field.reshape(N, H, W)


def aia_step_field(stack: np.ndarray, g: np.ndarray, delta0: Optional[np.ndarray] = None,
                    iters: int = 30, tol: float = 1e-4, dtype=None,
                    degree: int = 1, refine_iters: int = 5, refine_tol: float = 1e-3,
                    crop: int = 100):
    """Advanced Iterative Algorithm with iterative, arbitrary-degree step-field refinement.

    Runs the piston-only :func:`phase.methods.aia.aia` to convergence, then
    alternates fitting the per-frame step-field residual
    (:func:`fit_step_field`), scoring it (:func:`step_field_quality`), and
    removing its estimated intensity contribution from the *original* stack
    before re-running the fast pixel/frame step
    (:func:`~phase.methods.aia.aia_pixel_step`,
    :func:`~phase.methods.aia.aia_frame_step`) -- until the round-over-round
    improvement in ``rms_frac`` falls below ``refine_tol`` or
    ``refine_iters`` is spent. See ``docs/step_field_residuals.md`` for the
    step-field model this refines against, and ``docs/aia.md`` for the
    inner piston-only solve.

    Before building the correction fed into that re-solve, each round's
    fitted coefficients are gauge-fixed (subtracting each basis term's
    frame mean, ``docs/step_field_residuals.md`` Eq. T3b/E4) -- an exact
    invariance of the forward model, so this changes nothing about what the
    correction removes from the data, but it keeps the static part of any
    basis term's coefficient out of the correction and lets the next
    pixel/frame-step re-solve absorb it into ``Phi`` instead, where it
    physically belongs (a per-frame-constant step-field term is
    indistinguishable from part of ``phi_carrier``). The *reported*
    ``coeffs`` (see :class:`StepFieldParam`) are the raw, non-gauge-fixed
    fit against the final ``(a, u, v, delta)`` -- see that class's docstring.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted interferogram frames -- see :func:`phase.methods.aia.aia`.
    g : np.ndarray, shape (N,)
        Per-frame fringe contrast -- see :func:`phase.methods.aia.aia`.
    delta0, iters, tol, dtype
        Passed through to the initial :func:`phase.methods.aia.aia` call;
        see that function for their meaning.
    degree : int, default 1
        Highest total polynomial degree to fit the step field to (see
        :func:`_poly_basis`); ``0`` disables the step-field correction
        entirely (no refinement round ever runs, regardless of
        ``refine_iters``), returning the plain-``aia`` result through this
        same code path -- a baseline to compare higher degrees against.
        ``1`` is a pure linear tilt (registered separately as
        ``"aia_tilt"`` in :data:`phase.methods.METHOD_REGISTRY`), ``2``
        adds curvature. ``docs/step_field_residuals.md`` §7 recommends
        keeping this small (2-3): a higher degree couples more strongly into
        the frame step's own estimate of ``delta_n``.
    refine_iters : int, default 5
        Maximum number of refinement rounds. ``refine_iters=0`` skips
        refinement entirely, returning the plain ``aia`` result (with
        ``coeffs`` all zero) wrapped in a :class:`StepFieldParam`.
    refine_tol : float, default 1e-3
        Stop refining once a round's *non-negative* round-over-round drop in
        :func:`step_field_quality`'s ``rms_frac`` is below this. A round
        that makes ``rms_frac`` worse does not stop the loop early and is
        never returned: refinement keeps running (up to ``refine_iters``),
        and whichever round had the lowest ``rms_frac`` overall is what is
        ultimately returned -- see :attr:`StepFieldParam.best_iter`.
    crop : int, default 100
        Pixels excluded from each edge of the field when computing
        ``rms_frac`` (see :func:`step_field_quality`).

    Returns
    -------
    a, b, phi, delta, method_param
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

    a_map, b0, phi0, delta, aia_param0 = aia(stack, g, delta0=delta0, iters=iters, tol=tol, dtype=dtype)
    a = a_map.reshape(-1)
    u = (b0 * xp.cos(phi0)).reshape(-1)
    v = (-b0 * xp.sin(phi0)).reshape(-1)

    basis = _poly_basis(H, W, degree, xp)                            # (J, P)
    J = basis.shape[0]
    coeffs = xp.zeros((J, N), dtype=xp.float64)
    kappa_fit = float("nan")
    rms_frac = float("nan")
    rms_history: List[float] = []
    prev_rms = None
    refine_converged = False
    best = None            # (a, u, v, delta, coeffs, kappa_fit, rms_frac) of the best round
    best_iter = -1
    it = -1

    # degree=0 has no coefficients to fit (J=0, no step-field basis at all)
    # -- gate the loop on J too, so this is exactly the plain aia() result,
    # the same short-circuit already documented for refine_iters=0. This
    # also sidesteps feeding an empty basis into fit_step_field/_cond_batch,
    # whose batched svd/solve calls aren't meant to handle a 0-column Gram
    # matrix (no singular values to reduce over).
    for it in range(refine_iters if J > 0 else 0):
        coeffs_it, cond = fit_step_field(I, a, u, v, delta, basis, g=g)
        kappa_it = float(xp.max(cond))
        rms_frac_it, _ = step_field_quality(I, a, u, v, delta, coeffs_it, basis, H, W, g=g, crop=crop)
        rms_history.append(rms_frac_it)

        # (a, u, v, delta) here are the values *before* this round's
        # correction -- the state coeffs_it/rms_frac_it were actually fit
        # and scored against, so the snapshot is self-consistent.
        if best is None or rms_frac_it < best[-1]:
            best = (a, u, v, delta, coeffs_it, kappa_it, rms_frac_it)
            best_iter = it

        # Only a genuine (non-negative) improvement below refine_tol counts
        # as convergence -- a *worse* round must not look like "converged"
        # just because prev_rms - rms_frac_it is negative and hence also
        # less than a positive refine_tol. A regression keeps the loop
        # running (up to refine_iters); it just isn't the best round, and
        # the final answer below is always the best one seen, not the last.
        if prev_rms is not None and 0 <= (prev_rms - rms_frac_it) < refine_tol:
            refine_converged = True
            break
        prev_rms = rms_frac_it

        # Gauge-fix (Eq. T3b/E4) before building the correction: remove each
        # basis term's frame mean so only the frame-to-frame-varying part of
        # the fit is subtracted from the data, leaving the static part for
        # the next pixel/frame-step re-solve to absorb into Phi.
        coeffs_fixed = coeffs_it - coeffs_it.mean(axis=1, keepdims=True)

        # Remove this round's fitted (gauge-fixed) step field from the
        # *original* stack (not a running-corrected buffer -- each round's
        # coeffs already estimates the total current-best step field, not
        # an increment) and re-run the fast, shared-pseudoinverse
        # pixel/frame step on the result. fit_step_field's own normal
        # equations (Eq. E1) fit resid ~= -w*Delta, so removing that
        # contribution means adding w*Delta back, not subtracting it.
        c, s = xp.cos(delta), xp.sin(delta)
        w = g[:, None] * (xp.outer(s, u) - xp.outer(c, v))
        Delta_field = coeffs_fixed.T @ basis                          # (N, P)
        corrected = I + w * Delta_field

        a, u, v = aia_pixel_step(corrected, delta, g, dtype=work_dtype)
        new_delta = aia_frame_step(corrected, u, v)
        delta = new_delta - new_delta[0]

    refine_iters_run = it + 1

    if best is not None:
        a, u, v, delta, coeffs, kappa_fit, rms_frac = best

    aia_param = _aia_diagnostics(I, delta, g, a, u, v, N, xp,
                                  aia_param0.iters_run, aia_param0.converged)
    coeffs_rms = xp.sqrt(xp.mean(coeffs ** 2, axis=1))
    method_param = StepFieldParam(
        aia_param=aia_param, degree=degree, coeffs=coeffs, coeffs_rms=coeffs_rms,
        kappa_fit=kappa_fit, rms_frac=rms_frac, rms_frac_history=rms_history,
        refine_iters_run=refine_iters_run, refine_converged=refine_converged,
        best_iter=best_iter,
    )

    phi = xp.arctan2(-v, u).reshape(H, W)
    u64, v64 = u.astype(xp.float64), v.astype(xp.float64)
    b = xp.sqrt(u64 ** 2 + v64 ** 2).reshape(H, W)
    a_map = a.reshape(H, W)
    return a_map, b, phi, delta, method_param
