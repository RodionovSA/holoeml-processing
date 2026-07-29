""" Utility functions and tools for phase processing"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AIAResult:
    """Output of :func:`aia`.

    Attributes
    ----------
    phi : np.ndarray, shape (H, W)
        Wrapped phase map, in ``(-pi, pi]``.
    b : np.ndarray, shape (H, W)
        Fringe modulation amplitude (contrast) map.
    a : np.ndarray, shape (H, W)
        Background intensity map.
    delta : np.ndarray, shape (N,)
        Converged per-frame phase steps, in radians, referenced to
        ``delta[0] = 0``.
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

    phi: np.ndarray
    b: np.ndarray
    a: np.ndarray
    delta: np.ndarray
    kappa_p: float
    kappa_ps: float
    predicted_rms: float
    iters_run: int
    converged: bool


def aia(stack: np.ndarray, delta0: Optional[np.ndarray] = None,
        iters: int = 30, tol: float = 1e-4) -> AIAResult:
    """Advanced Iterative Algorithm (AIA) for phase-shifting interferometry.

    Recovers the wrapped phase map from a stack of phase-shifted
    interferograms whose phase-step sizes are not precisely known, by
    jointly estimating the per-pixel fringe pattern and the per-frame
    phase steps.

    Model
    -----
    Each frame is assumed to follow the standard phase-shifting model::

        I_n(x, y) = a(x, y) + b(x, y) * cos(phi(x, y) + delta_n)
                  = a(x, y) + u(x, y) * cos(delta_n) + v(x, y) * sin(delta_n)

    where ``u = b*cos(phi)`` and ``v = -b*sin(phi)``. This is linear in
    ``(a, u, v)`` for fixed ``delta_n``, and linear in ``delta_n`` for
    fixed ``(u, v)`` -- but not linear in both at once, so the unknowns
    are recovered by alternating least squares.

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
        Phase-shifted interferogram frames.
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

    Returns
    -------
    AIAResult
        See :class:`AIAResult` for field descriptions. Physical outputs
        are ``phi, b, a, delta``; ``kappa_p, kappa_ps, predicted_rms``
        are diagnostics for whether this acquisition (frame count,
        phase-shift distribution, noise level) supports a trustworthy
        result; ``iters_run, converged`` describe convergence.

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

    Notes
    -----
    An earlier version of this function also estimated a per-frame
    amplitude-modulation factor ``g_n``, and separately tried subtracting
    the background ``a`` in the frame step. Both were removed: on real
    data, ``g_n`` made convergence less robust (it could lock onto a
    self-consistent but incorrect fixed point), and subtracting ``a``
    increased leaked oscillations into the recovered background rather
    than reducing them. Chen & Kemao (2019) explain why the latter isn't
    the right fix: the background decouples from the fringe terms when
    the phase-shift distribution is well conditioned, so accuracy should
    be controlled via ``delta0`` spacing and frame count -- monitored
    here with ``kappa_p``/``kappa_ps`` -- rather than by subtracting a
    still-imperfect, mid-iteration background estimate.
    """
    N, H, W = stack.shape
    I = stack.reshape(N, -1).astype(float)          # (N, P)
    if delta0 is None:
        delta0 = np.arange(N) * 2 * np.pi / N
    delta = np.asarray(delta0, float).copy()
    u = v = a = None
    A = B = None
    converged = False
    it = 0
    for it in range(iters):
        # pixel step: I_n = a + u cosδ_n + v sinδ_n,  u=b cosφ, v=-b sinφ
        c, s = np.cos(delta), np.sin(delta)
        A = np.column_stack([np.ones(N), c, s])                       # (N,3)
        a, u, v = np.linalg.pinv(A) @ I                                # (3,P)

        # frame step: fit [alpha_n, Pn, Qn] against the fixed u,v patterns
        B = np.column_stack([np.ones(u.size), u, v])                  # (P,3)
        x = np.linalg.solve(B.T @ B, (I @ B).T)                       # (3,N)
        Pn, Qn = x[1], x[2]
        new_delta = np.arctan2(Qn, Pn)

        new_delta -= new_delta[0]                                     # pin phase origin
        step = np.abs(np.angle(np.exp(1j*(new_delta - delta)))).max()
        delta = new_delta
        if step < tol:
            converged = True
            break

    phi = np.arctan2(-v, u).reshape(H, W)
    b   = np.sqrt(u**2 + v**2).reshape(H, W)
    a_map = a.reshape(H, W)

    # diagnostics (Chen & Kemao 2019): condition numbers of the two
    # normal matrices, and the accuracy they predict.
    kappa_p = float(np.linalg.cond(A.T @ A))

    # kappa_ps: paper's *normalized* A_ps (Eq. 12), built from unit-circle
    # directions cos(phi), sin(phi) with amplitude b divided out. This is
    # deliberately different from the actual (amplitude-weighted) frame-step
    # solve matrix B.T@B -- normalizing is what makes the >=2 bound and the
    # "large is bad" threshold below meaningful; the amplitude-weighted
    # version is skewed by fringe-visibility variation, not just phase
    # coverage (see AIAResult.kappa_ps docstring).
    r = np.sqrt(u**2 + v**2)
    r = np.maximum(r, np.finfo(float).eps)
    cphi, sphi = u / r, -v / r
    C = np.column_stack([np.ones(cphi.size), cphi, sphi])         # (P,3)
    kappa_ps = float(np.linalg.cond(C.T @ C))

    resid = I - A @ np.vstack([a, u, v])
    sigma = float(np.sqrt(np.mean(resid**2)))
    b_amp = max(float(np.median(b)), np.finfo(float).eps)
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

    return AIAResult(
        phi=phi, b=b, a=a_map, delta=delta,
        kappa_p=kappa_p, kappa_ps=kappa_ps, predicted_rms=predicted_rms,
        iters_run=it + 1, converged=converged,
    )


@dataclass
class CarrierResult:
    """Output of :func:`remove_carrier`.

    Attributes
    ----------
    phi : np.ndarray, shape (H, W)
        Wrapped phase map with the linear carrier (and, if requested, the
        defocus term) and piston removed, in ``(-pi, pi]``.
    kx, ky : float
        Fitted carrier spatial frequency, in rad/pixel, along the column
        (x) and row (y) axes respectively.
    fx, fy : float
        Same carrier, in cycles/pixel (``kx = 2*pi*fx``, ``ky = 2*pi*fy``).
    kxx, kyy, kxy : float
        Fitted quadratic curvature coefficients, in rad/pixel^2, for the
        term ``kxx*x^2 + kyy*y^2 + kxy*x*y``. All ``0.0`` when
        ``defocus=False``. ``kxx == kyy`` and ``kxy == 0`` corresponds to
        isotropic defocus (a spherical wavefront); ``kxx != kyy`` and/or
        ``kxy != 0`` describes astigmatic/cross-coupled curvature.
    piston : float
        Fitted constant phase offset, in radians, removed after
        demodulation.
    """

    phi: np.ndarray
    kx: float
    ky: float
    fx: float
    fy: float
    kxx: float
    kyy: float
    kxy: float
    piston: float


def _estimate_tilt(c: np.ndarray, w: np.ndarray, X: np.ndarray, Y: np.ndarray,
                    window: bool, refine_iters: int) -> tuple:
    """Estimate the dominant linear carrier ``(fx, fy)`` of a complex field.

    Shared by :func:`remove_carrier`'s final full-field estimate and its
    per-block defocus pre-estimation -- see that function's Algorithm
    section (steps 1-2) for the coarse-FFT + wrap-safe-refine method.
    ``X, Y`` must be same-shape coordinate grids for ``c`` (absolute
    origin doesn't matter -- only integer pixel spacing does).
    """
    H, W = c.shape
    if window and H > 1 and W > 1:
        win = np.outer(np.hanning(H), np.hanning(W))
    else:
        win = np.ones((H, W))
    F = np.fft.fft2(w * c * win)
    iy, ix = np.unravel_index(np.argmax(np.abs(F)), F.shape)
    fy = float(np.fft.fftfreq(H)[iy])
    fx = float(np.fft.fftfreq(W)[ix])

    for _ in range(refine_iters):
        d = c * np.exp(-1j * 2 * np.pi * (fx * X + fy * Y))
        dfx = dfy = 0.0
        if W > 1:
            wpx = w[:, 1:] * w[:, :-1]
            gx = np.angle(np.sum(wpx * d[:, 1:] * np.conj(d[:, :-1])))
            dfx = gx / (2 * np.pi)
        if H > 1:
            wpy = w[1:, :] * w[:-1, :]
            gy = np.angle(np.sum(wpy * d[1:, :] * np.conj(d[:-1, :])))
            dfy = gy / (2 * np.pi)
        fx += dfx
        fy += dfy
        if abs(dfx) < 1e-8 and abs(dfy) < 1e-8:
            break

    return fx, fy


def _estimate_curvature(c: np.ndarray, w: np.ndarray, window: bool,
                         refine_iters: int, n_blocks: int) -> tuple:
    """Estimate quadratic curvature ``(kxx, kyy, kxy)`` via block-wise tilt.

    For ``phi = piston + kx*x + ky*y + kxx*x^2 + kyy*y^2 + kxy*x*y``, the
    local instantaneous frequency is linear in position:
    ``fx_local(x, y) = fx0 + (kxx/pi)*x + (kxy/(2*pi))*y``,
    ``fy_local(x, y) = fy0 + (kxy/(2*pi))*x + (kyy/pi)*y``. Splits the
    field into an ``n_blocks x n_blocks`` grid, measures a local
    ``(fx, fy)`` per block with :func:`_estimate_tilt`, and fits that
    (coupled, since ``kxy`` appears in both) linear relationship by least
    squares across all blocks to recover ``kxx, kyy, kxy``. Returns all
    zeros (with a warning) if fewer than 4 blocks have usable weight,
    since that under-determines the 5-unknown fit.
    """
    H, W = c.shape
    row_groups = [g for g in np.array_split(np.arange(H), n_blocks) if len(g) > 1]
    col_groups = [g for g in np.array_split(np.arange(W), n_blocks) if len(g) > 1]

    total_w = w.sum()
    floor = 0.01 * total_w / max(len(row_groups) * len(col_groups), 1)

    xc_list, yc_list, fx_list, fy_list = [], [], [], []
    for rows in row_groups:
        for cols in col_groups:
            idx = np.ix_(rows, cols)
            w_block = w[idx]
            if w_block.sum() < floor:
                continue
            c_block = c[idx]
            Xb, Yb = np.meshgrid(np.arange(len(cols), dtype=float),
                                  np.arange(len(rows), dtype=float))
            fx_i, fy_i = _estimate_tilt(c_block, w_block, Xb, Yb, window, refine_iters)
            xc_list.append(float(cols.mean()))
            yc_list.append(float(rows.mean()))
            fx_list.append(fx_i)
            fy_list.append(fy_i)

    if len(xc_list) < 4:
        warnings.warn(
            "remove_carrier: fewer than 4 blocks had usable weight; "
            "cannot fit a curvature term, skipping it (kxx=kyy=kxy=0). "
            "Try a smaller n_blocks or check weight/mask coverage.",
            stacklevel=3,
        )
        return 0.0, 0.0, 0.0

    n = len(xc_list)
    # unknowns: [fx0, fy0, A=kxx/pi, B=kxy/(2*pi), C=kyy/pi]
    M = np.zeros((2 * n, 5))
    rhs = np.zeros(2 * n)
    M[0::2, 0] = 1.0
    M[0::2, 2] = xc_list
    M[0::2, 3] = yc_list
    rhs[0::2] = fx_list
    M[1::2, 1] = 1.0
    M[1::2, 3] = xc_list
    M[1::2, 4] = yc_list
    rhs[1::2] = fy_list
    _, _, A, B, C = np.linalg.lstsq(M, rhs, rcond=None)[0]
    kxx = float(np.pi * A)
    kyy = float(np.pi * C)
    kxy = float(2 * np.pi * B)
    return kxx, kyy, kxy


def remove_carrier(phi: np.ndarray, weight: Optional[np.ndarray] = None,
                    mask: Optional[np.ndarray] = None, refine_iters: int = 5,
                    window: bool = True, defocus: bool = False,
                    n_blocks: int = 6) -> CarrierResult:
    """Estimate and remove a linear phase carrier from a wrapped phase map.

    Off-axis / tilted-reference interferometry (and piezo-scanning setups
    with a residual tilt) superimpose a spatial carrier on the recovered
    phase::

        phi(x, y) = wrap(phase_obj(x, y) + kx*x + ky*y
                         + kxx*x^2 + kyy*y^2 + kxy*x*y + piston)

    This removes the linear ramp ``kx*x + ky*y``, the constant ``piston``,
    and -- if ``defocus=True`` -- the quadratic curvature term
    ``kxx*x^2 + kyy*y^2 + kxy*x*y``, leaving ``wrap(phase_obj)``. It never
    unwraps: all estimation happens on the complex field ``exp(i*phi)``,
    so it is insensitive to how many fringes the carrier spans.

    Algorithm
    ---------
    1. **Coarse (FFT).** FFT the (optionally Hann-windowed, to suppress
       edge leakage) weighted complex field ``weight * exp(i*phi)``; the
       carrier shows up as the dominant off-DC peak. Its bin gives an
       integer-pixel estimate of the carrier frequency ``(fx, fy)`` in
       cycles/pixel. This step is what makes the method robust to a high
       carrier (many fringes across the field, near the Nyquist limit) --
       a pixel-to-pixel gradient estimate alone would alias there.
    2. **Sub-pixel refinement (wrap-safe).** Demodulate by the current
       ``(fx, fy)`` estimate and measure the *residual* tilt by vector
       averaging of neighboring-pixel phase differences,
       ``angle(sum(d[..., 1:] * conj(d[..., :-1])))`` along each axis --
       equivalent to a weighted circular mean of local gradients, so wraps
       in the residual don't bias the estimate the way a naive
       finite-difference-then-average would. Repeated ``refine_iters``
       times (each step's residual is much smaller than the last, since
       the coarse step already removed the bulk of the tilt). Steps 1-2
       are :func:`_estimate_tilt`.
    3. **Curvature pre-correction (if ``defocus=True``), before step 1-2's
       final pass.** A quadratic phase is a 2D chirp: its local
       instantaneous frequency varies linearly with position (and, if the
       curvature is anisotropic, ``fx`` also varies with ``y`` and vice
       versa via the coupled ``kxy`` term). Splitting the field into an
       ``n_blocks x n_blocks`` grid and running steps 1-2 *within each
       block* gives a local ``(fx, fy)`` sample at each block's center;
       jointly regressing those samples against block position recovers
       ``kxx, kyy, kxy`` (:func:`_estimate_curvature`). The field is
       demodulated by ``kxx*x^2 + kyy*y^2 + kxy*x*y`` before the final,
       full-field tilt/piston pass, which then only has to polish the
       (typically much smaller) residual linear term.
    4. **Demodulate + piston.** Divide out the final carrier (and defocus,
       if applicable), then remove the constant phase offset via the
       (weighted) mean resultant vector.

    Parameters
    ----------
    phi : np.ndarray, shape (H, W)
        Wrapped phase map, in ``(-pi, pi]`` (e.g. ``AIAResult.phi``).
    weight : np.ndarray, shape (H, W), optional
        Per-pixel reliability used only for *estimating* the carrier (e.g.
        ``AIAResult.b``, the modulation map) -- down-weights noisy,
        low-modulation pixels so they don't bias the fit. Negative values
        are clipped to 0. Does not affect the returned ``phi``, which is
        always computed from the unweighted field.
    mask : np.ndarray, shape (H, W), optional
        Boolean (or 0/1) map; pixels where it is falsey are excluded from
        estimation, same effect as ``weight=0`` there. Combined with
        ``weight`` if both are given.
    refine_iters : int, default 5
        Number of sub-pixel refinement iterations after each coarse FFT
        step (both the final full-field estimate and, if ``defocus=True``,
        each per-block estimate).
    window : bool, default True
        Apply a 2D Hann window before each coarse FFT to reduce spectral
        leakage from the field's edges (recommended; skipped automatically
        if the field being estimated has a size-1 axis).
    defocus : bool, default False
        Also fit and remove a quadratic curvature term
        ``kxx*x^2 + kyy*y^2 + kxy*x*y``, via the block-regression method
        above. Off by default so existing linear-only behavior is
        unchanged. Worth enabling if: the field spans a large FOV, the
        setup uses a non-collimated/point-source reference beam, or (a
        useful diagnostic) the fitted carrier is noticeably sensitive to
        ``weight``/``mask`` choices even after excluding genuine
        background -- that sensitivity is the signature of fitting a
        plane to a curved surface, not a bug in the weighting.
    n_blocks : int, default 6
        Grid size (``n_blocks x n_blocks``) for the curvature pre-estimate.
        Only used when ``defocus=True``. Needs at least 4 blocks with
        usable weight to fit (5 unknowns, 2 equations/block); falls back
        to ``kxx=kyy=kxy=0`` (with a warning) if fewer are available --
        e.g. too small an image for the requested grid, or ``weight``/
        ``mask`` leaving too little usable area.

    Returns
    -------
    CarrierResult
        See :class:`CarrierResult`. ``phi`` is the object phase with the
        carrier (and curvature, if requested) and piston removed; ``kx,
        ky`` (rad/pixel) / ``fx, fy`` (cycles/pixel) describe the removed
        linear carrier; ``kxx, kyy, kxy`` (rad/pixel^2) the removed
        curvature terms (all ``0.0`` if not requested); ``piston`` the
        removed constant offset.

    Notes
    -----
    The block-regression estimate is exact for a pure carrier+curvature
    field (verified against synthetic data with no other structure); on
    real data any genuine object phase with local structure comparable to
    the block size will mildly bias individual blocks' tilt estimates,
    the usual resolution/robustness trade-off of a block size -- fewer,
    larger blocks average out more of that bias but track less localized
    curvature. If ``kxx`` and ``kyy`` come back similar and ``kxy`` near
    0, the curvature is essentially isotropic (a spherical wavefront); if
    they differ substantially, that's astigmatism or a tilted/off-axis
    reference, both handled by this general form.
    """
    H, W = phi.shape
    c = np.exp(1j * phi)

    w = np.ones((H, W)) if weight is None else np.clip(np.asarray(weight, float), 0, None)
    if mask is not None:
        w = w * np.asarray(mask, bool)
    if w.sum() <= 0:
        w = np.ones((H, W))

    X, Y = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))

    kxx = kyy = kxy = 0.0
    if defocus:
        kxx, kyy, kxy = _estimate_curvature(c, w, window, refine_iters, n_blocks)
        if kxx or kyy or kxy:
            c = c * np.exp(-1j * (kxx * X**2 + kyy * Y**2 + kxy * X * Y))

    fx, fy = _estimate_tilt(c, w, X, Y, window, refine_iters)

    # demodulate the (curvature-corrected) carrier, then remove the piston
    d = c * np.exp(-1j * 2 * np.pi * (fx * X + fy * Y))
    piston = float(np.angle(np.sum(w * d)))
    d = d * np.exp(-1j * piston)
    phi_flat = np.angle(d)

    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    return CarrierResult(phi=phi_flat, kx=kx, ky=ky, fx=fx, fy=fy,
                          kxx=kxx, kyy=kyy, kxy=kxy, piston=piston)


@dataclass
class DifferenceResult:
    """Output of :func:`subtract_reference`.

    Attributes
    ----------
    phi : np.ndarray, shape (H, W)
        Resolved phase difference, in ``(-pi, pi]``.
    sign : int
        ``+1`` if ``phi_ref`` was used as-is (``phi - phi_ref``); ``-1``
        if ``phi_ref``'s sign branch was flipped before combining
        (``phi + phi_ref``) -- see :func:`subtract_reference` for why.
    spread_same : float
        Weighted circular RMS spread, in radians, of ``phi - phi_ref``.
    spread_flipped : float
        Weighted circular RMS spread, in radians, of ``phi + phi_ref``.
    ambiguous : bool
        True if ``spread_same`` and ``spread_flipped`` were too close to
        confidently pick a branch (see ``ambiguous_ratio``) -- the choice
        of ``sign`` shouldn't be trusted without visually checking ``phi``.
    """

    phi: np.ndarray
    sign: int
    spread_same: float
    spread_flipped: float
    ambiguous: bool


def subtract_reference(phi: np.ndarray, phi_ref: np.ndarray,
                        weight: Optional[np.ndarray] = None,
                        mask: Optional[np.ndarray] = None,
                        ambiguous_ratio: float = 1.5) -> DifferenceResult:
    """Combine a sample phase map with an independently-recovered reference.

    A reference measurement (same setup, no sample) is meant to capture
    whatever tilt/curvature/background aberration comes purely from the
    setup, so that subtracting it from a sample measurement's phase
    cancels that shared aberration and leaves only the sample-induced
    phase. This only works if ``phi`` and ``phi_ref`` share a common sign
    convention -- and :func:`aia` cannot guarantee that on its own.

    Why not: the phase-shifting model ``I_n = a + b*cos(phi + delta_n)``
    is exactly invariant under ``(phi, delta) -> (-phi, -delta)`` for
    every frame at once (cosine is even), so ``aia`` has no way to tell
    ``+phi`` from ``-phi`` from intensity data alone. Each independent
    ``aia()`` call converges to *one* of these two mirror-image branches,
    and nothing forces two separate runs (e.g. sample vs. reference) to
    land on the same one. If they land on opposite branches, naive
    ``phi - phi_ref`` *adds* the shared aberration instead of canceling
    it (``phi_true - (-phi_true) = 2*phi_true``), which can make the
    result visibly worse than not subtracting at all.

    This function resolves the branch automatically: it computes both
    ``phi - phi_ref`` and ``phi + phi_ref`` (wrap-safe, in the complex
    domain) and keeps whichever has lower spread -- the shared aberration
    is normally the dominant term in both ``phi`` and ``phi_ref``, so the
    correctly-signed combination should cancel most of it and come out
    with dramatically lower spread than the wrong one.

    Parameters
    ----------
    phi : np.ndarray, shape (H, W)
        Sample phase map, in ``(-pi, pi]`` (e.g. ``AIAResult.phi``).
    phi_ref : np.ndarray, shape (H, W)
        Reference phase map, same shape as ``phi``.
    weight : np.ndarray, shape (H, W), optional
        Per-pixel reliability used to weight the spread comparison (e.g.
        the sample's modulation map) -- does not affect the returned
        ``phi`` itself, only which branch is judged better. Negative
        values are clipped to 0.
    mask : np.ndarray, shape (H, W), optional
        Boolean (or 0/1) map; pixels where it is falsey are excluded from
        the spread comparison. Combined with ``weight`` if both are given.
    ambiguous_ratio : float, default 1.5
        If the larger of the two spreads is less than this factor times
        the smaller, the branches aren't clearly distinguishable and
        ``ambiguous=True`` is returned (with a warning) -- e.g. because
        the shared aberration is small relative to noise/sample signal,
        or the two measurements didn't actually share much of a common
        aberration to cancel.

    Returns
    -------
    DifferenceResult
        See :class:`DifferenceResult`.

    Notes
    -----
    This resolves the sign ambiguity, not general drift between the two
    measurements. If the setup moved between the sample and reference
    shots, some residual tilt/curvature can remain in ``phi`` even after
    picking the right branch -- :func:`remove_carrier` can clean that up
    as a separate step, run on the *difference* map, not on ``phi``/
    ``phi_ref`` individually beforehand.
    """
    if phi.shape != phi_ref.shape:
        raise ValueError(f"phi.shape {phi.shape} != phi_ref.shape {phi_ref.shape}")

    H, W = phi.shape
    w = np.ones((H, W)) if weight is None else np.clip(np.asarray(weight, float), 0, None)
    if mask is not None:
        w = w * np.asarray(mask, bool)
    if w.sum() <= 0:
        w = np.ones((H, W))

    def spread(d: np.ndarray) -> float:
        mean_angle = np.angle(np.sum(w * np.exp(1j * d)))
        resid = np.angle(np.exp(1j * (d - mean_angle)))
        return float(np.sqrt(np.sum(w * resid**2) / np.sum(w)))

    diff_same = np.angle(np.exp(1j * phi) * np.exp(-1j * phi_ref))
    diff_flipped = np.angle(np.exp(1j * phi) * np.exp(1j * phi_ref))
    spread_same = spread(diff_same)
    spread_flipped = spread(diff_flipped)

    if spread_same <= spread_flipped:
        phi_out, sign = diff_same, 1
    else:
        phi_out, sign = diff_flipped, -1

    lo, hi = sorted([spread_same, spread_flipped])
    ambiguous = hi < ambiguous_ratio * lo
    if ambiguous:
        warnings.warn(
            f"subtract_reference: spread_same={spread_same:.4f} and "
            f"spread_flipped={spread_flipped:.4f} are too close to "
            f"confidently resolve the sign branch (ratio "
            f"{hi / lo:.2f} < {ambiguous_ratio}); check phi visually "
            f"before trusting it.",
            stacklevel=2,
        )

    return DifferenceResult(phi=phi_out, sign=sign, spread_same=spread_same,
                             spread_flipped=spread_flipped, ambiguous=ambiguous)
