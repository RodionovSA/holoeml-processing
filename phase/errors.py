"""Per-pixel phase-error maps for :class:`phase.solver.PhaseResult.phi_error`.

Computed from a method's *output* (``b, phi, delta, g``), not from inside the
method itself -- see ``docs/aia.md`` "Direct phase-error computation" for the
derivation (Eq. 21-38) and ``docs/step_field_residuals.md`` §9 (Eq. E5-E9)
for the ``aia_step_field``/``aia_tilt`` extension. :func:`compute_phi_error`
is the single entry point :class:`phase.solver.PhaseSolver` calls; it
dispatches on ``method`` and currently computes a result for ``"aia"`` and
``"aia_step_field"``/``"aia_tilt"`` -- every other registered method gets
``None``, since no closed-form error expression has been derived for it.
"""

from typing import Optional, Tuple

import numpy as np

from .methods.step_field import _poly_basis


def compute_phi_error(method: str, b: np.ndarray, phi: np.ndarray, delta: np.ndarray,
                       g: np.ndarray, fit_gain: bool, noise_std: np.ndarray,
                       simplified: bool, method_param, xp) -> Optional[np.ndarray]:
    """Dispatch to the per-method phase-error implementation.

    Parameters
    ----------
    method : str
        ``PhaseConfig.method``. ``"aia"`` and ``"aia_step_field"``/
        ``"aia_tilt"`` (case-insensitive, matching
        :data:`phase.methods.METHOD_REGISTRY`'s own aliasing) are
        implemented; every other value returns ``None``.
    b, phi : np.ndarray, shape (H, W)
        Fitted fringe amplitude and wrapped phase.
    delta, g : np.ndarray, shape (N,)
        Fitted per-frame phase steps and gain.
    fit_gain : bool
        Whether ``g`` was jointly fitted (``docs/aia.md``'s Stage 3) or held
        fixed (Stage 2) -- selects which correction term Eq. (38) includes.
    noise_std : np.ndarray, shape (H, W)
        Per-pixel camera noise std, ``docs/aia.md``'s ``sigma_0(x, y)``
        (Eq. 27a) -- either ``PhaseConfig.noise_std``, or, when that's
        ``None``, :meth:`PhaseSolver.fit`'s own per-pixel residual map
        (the frame-averaged reconstruction residual, kept as a map rather
        than pooled into the scalar ``reconstruction_error``). Shape must
        match ``phi``.
    simplified : bool
        If True, return Eq. (22)'s baseline only, dropping the ``delta_n``/
        ``g_n`` uncertainty correction (Eq. 34/38) and, for
        ``aia_step_field``, the step-field leverage discount (Eq. E9) --
        both are ``O(1/N_p)`` relative to the baseline.
    method_param : MethodParam
        The fitted result's own diagnostics (e.g. an
        :class:`phase.methods.step_field.StepFieldParam` for
        ``method="aia_step_field"``) -- only its ``.degree`` is used here,
        to rebuild the step-field basis.
    xp : module
        ``numpy`` or ``cupy``, matching ``b``/``phi``/``delta``/``g``.

    Returns
    -------
    np.ndarray, shape (H, W), or None
        ``sigma_Phi(x, y)`` in radians for a supported method; ``None``
        otherwise.

    Raises
    ------
    ValueError
        If ``noise_std``'s shape does not match ``phi``.
    """
    if noise_std.shape != phi.shape:
        raise ValueError(
            f"noise_std shape {noise_std.shape} does not match phi shape {phi.shape}"
        )
    method = method.lower()
    if method == "aia":
        phi_var, _ = _aia_phi_error_parts(b, phi, delta, g, fit_gain, noise_std,
                                           simplified, xp)
        return xp.sqrt(phi_var)
    if method in ("aia_step_field", "aia_tilt"):
        return _aia_step_field_phi_error(b, phi, delta, g, fit_gain, noise_std,
                                          simplified, method_param.degree, xp)
    return None


def _aia_phi_error_parts(b: np.ndarray, phi: np.ndarray, delta: np.ndarray, g: np.ndarray,
                          fit_gain: bool, sigma0, simplified: bool, xp
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """``docs/aia.md``'s AIA phase-error map, Eq. (22) plus Eq. (34)/(38).

    Factored out of :func:`compute_phi_error`'s plain-``aia`` branch so
    :func:`_aia_step_field_phi_error` can reuse ``w_dot_k`` (built from
    Eq. 29's leverage vectors) rather than recomputing it -- the step-field
    discount (``docs/step_field_residuals.md`` Eq. E9) needs the same
    quantity as Eq. (34)/(38)'s own correction term.

    Parameters
    ----------
    b, phi : np.ndarray, shape (H, W)
    delta, g : np.ndarray, shape (N,)
    fit_gain : bool
        Include Eq. (38)'s ``g_n`` term (Stage 3) when True, Eq. (34)'s
        ``delta_n``-only term (Stage 2) when False.
    sigma0 : np.ndarray, shape (H, W), or float
    simplified : bool
        Skip the correction term entirely when True.
    xp : module

    Returns
    -------
    phi_var : np.ndarray, shape (H, W)
        ``sigma_Phi(x, y)^2``, Eq. (22) alone if ``simplified`` else plus
        Eq. (34)/(38).
    w_dot_k : np.ndarray, shape (N, H, W), or None
        ``w(x,y) . k_n`` (Eq. 29's leverage vectors dotted with
        ``(sin(phi), cos(phi))``), working dtype; ``None`` if
        ``simplified`` (nothing downstream needs it then).
    """
    work = b.dtype
    delta64 = xp.asarray(delta, dtype=xp.float64)
    g64 = xp.asarray(g, dtype=xp.float64)
    N = delta64.shape[0]

    # Frame-side quantities (Eqs. 20a, 21, 29) -- small, float64 regardless
    # of b/phi's working dtype, same convention as phase.methods.aia's own
    # normal-equation solves. Cast to `work` right before use against a big
    # (H, W)/(N, H, W) array, same as aia_pixel_step's own
    # `pinv(A).astype(work_dtype) @ stack` -- otherwise ordinary promotion
    # silently turns every big array below float64 too.
    d = xp.stack([xp.cos(delta64), xp.sin(delta64)])                     # (2, N)
    R = xp.mean(g64[None, :] * d, axis=1)                                # (2,)
    C = xp.mean(g64[None, :] ** 2 * d[:, None, :] * d[None, :, :], axis=2) \
        - xp.outer(R, R)                                                 # (2, 2)
    C_inv = xp.linalg.pinv(C)
    C_inv_w = C_inv.astype(work)

    eps = xp.asarray(xp.finfo(xp.float64).eps, dtype=work)
    b2 = xp.maximum(b ** 2, eps)
    w = xp.stack([xp.sin(phi), xp.cos(phi)])                             # (2, H, W)

    # Eq. (22): Stage-1 baseline.
    phi_var = sigma0 ** 2 / (N * b2) * xp.einsum('ahw,ab,bhw->hw', w, C_inv_w, w)

    if simplified:
        return phi_var, None

    Np = b.size
    b2_mean = xp.mean(b2)
    sigma_eff_sq = xp.mean(sigma0 ** 2 * b2) / b2_mean                    # Eq. (33)
    g_w = g64.astype(work)
    delta_w = delta64.astype(work)

    r = g64[None, :] * d - R[:, None]                                    # (2, N)
    k = ((C_inv @ r) / N).astype(work)                                   # (2, N), Eq. (29)
    w_dot_k = xp.einsum('ahw,an->nhw', w, k)                             # (N, H, W)

    dI_ddelta = -g_w[:, None, None] * b[None, :, :] \
        * xp.sin(phi[None, :, :] + delta_w[:, None, None])               # (N, H, W), Eq. (28)
    term = dI_ddelta ** 2 / g_w[:, None, None] ** 2
    if fit_gain:
        dI_dg = b[None, :, :] * xp.cos(phi[None, :, :] + delta_w[:, None, None])  # Eq. (35)
        term = term + dI_dg ** 2

    correction = (2 * sigma_eff_sq / (Np * b2_mean * b2)) \
        * xp.sum(term * w_dot_k ** 2, axis=0)                            # Eq. (34)/(38)
    return phi_var + correction, w_dot_k


def _aia_step_field_phi_error(b: np.ndarray, phi: np.ndarray, delta: np.ndarray,
                               g: np.ndarray, fit_gain: bool, sigma0,
                               simplified: bool, degree: int, xp) -> np.ndarray:
    """``docs/step_field_residuals.md``'s AIA-with-step-field phase-error map, Eq. (E9).

    ``docs/aia.md``'s Eq. (22)/(34)/(38) (via :func:`_aia_phi_error_parts`)
    discounted by the step-field fit's own leverage (Eq. E7): each frame's
    effective noise at a pixel is *reduced* by ``(1 - h_n(x,y))``, since
    Eq. (E1)'s per-frame fit is built from that same pixel's own residual.
    Reuses Eq. (29)'s leverage vectors ``k`` rather than a per-pixel 3x3
    solve -- an exact algebraic identity (`D_n = sigma0^2(1-h_n)` splits
    linearly, and the ``h_n`` piece collapses onto the same ``k``), not an
    approximation.

    Parameters
    ----------
    b, phi : np.ndarray, shape (H, W)
    delta, g : np.ndarray, shape (N,)
    fit_gain : bool
    sigma0 : np.ndarray, shape (H, W), or float
    simplified : bool
        Skip both Eq. (34)/(38)'s correction and this function's own
        leverage discount when True -- both are ``O(1/N_p)``, dropped
        together for consistency.
    degree : int
        Highest total polynomial degree of the fitted step field (
        :attr:`phase.methods.step_field.StepFieldParam.degree`); ``0``
        means no step field (:func:`phase.methods.step_field._poly_basis`
        returns an empty basis), so the discount is identically zero and
        this reduces to the plain-``aia`` result.
    xp : module

    Returns
    -------
    np.ndarray, shape (H, W)
        ``sigma_Phi(x, y)``, in radians.
    """
    phi_var, w_dot_k = _aia_phi_error_parts(b, phi, delta, g, fit_gain, sigma0, simplified, xp)
    if simplified:
        return xp.sqrt(phi_var)

    H, W = phi.shape
    work = b.dtype
    basis = _poly_basis(H, W, degree, xp).astype(work, copy=False)        # (J, P)
    J = basis.shape[0]
    if J == 0:
        return xp.sqrt(phi_var)

    delta64 = xp.asarray(delta, dtype=xp.float64)
    g64 = xp.asarray(g, dtype=xp.float64)
    P_n = (g64 * xp.cos(delta64)).astype(work)
    Q_n = (g64 * xp.sin(delta64)).astype(work)
    u = (b * xp.cos(phi)).reshape(-1)
    v = (-b * xp.sin(phi)).reshape(-1)
    w_n = Q_n[:, None] * u[None, :] - P_n[:, None] * v[None, :]           # (N, P), Eq. (E5)
    w_n_sq = w_n ** 2

    # Eq. (E1)'s own w_n^2-weighted Gram matrix, one (J, J) solve per frame
    # -- the same computation fit_step_field does internally, recomputed
    # here from data already in hand rather than threading a new return
    # value through phase/methods/step_field.py. Kept in the working dtype
    # throughout (unlike the frame-side-only C/C_inv above): G is a
    # reduction over the big (N, P) data, not a small per-iteration object,
    # so the same "cast big arrays down, not small ones up" rule as
    # _aia_phi_error_parts applies here to the reduction's inputs.
    G = xp.einsum('np,jp,kp->njk', w_n_sq, basis, basis)                  # (N, J, J)
    G_inv = xp.linalg.pinv(G)
    h = w_n_sq * xp.einsum('jp,njk,kp->np', basis, G_inv, basis)          # (N, P), Eq. (E7)
    h = h.reshape(-1, H, W)

    b2 = xp.maximum(b ** 2, xp.asarray(xp.finfo(xp.float64).eps, dtype=work))
    discount = (sigma0 ** 2 / b2) * xp.sum(h * w_dot_k ** 2, axis=0)

    return xp.sqrt(xp.maximum(phi_var - discount, 0))
