"""Advanced Iterative Algorithm (AIA) for phase-shifting interferometry."""

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
    g : np.ndarray, shape (N,)
        Per-frame fringe contrast (gain) used in the pixel-step model,
        normalized so ``median(g) = 1``. All ones if ``aia`` was called
        with ``gain=None`` (the default). See ``aia``'s ``gain`` parameter
        for why this must be supplied/measured rather than fit.
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
    g: np.ndarray
    kappa_p: float
    kappa_ps: float
    predicted_rms: float
    iters_run: int
    converged: bool


def measure_frame_contrast(stack: np.ndarray, dc_radius: int = 8,
                            halfwin: tuple = (3, 4)) -> np.ndarray:
    """Measure each frame's fringe contrast directly from its spatial carrier.

    ``aia`` assumes every frame shares one fringe-modulation map ``b(x,y)``;
    in practice illumination drift, source-coherence roll-off over a long
    scan, or per-shot exposure variation make the *true* per-frame contrast
    ``g_n`` deviate from 1 -- sometimes by tens of percent. Forcing a shared
    ``b`` then makes the least-squares fit trade contrast error off against
    phase, producing an error that is a deterministic function of the local
    phase (see ``aia``'s ``gain`` parameter). This function measures ``g_n``
    directly from the data, independent of the AIA solve, so it can be
    supplied as a fixed input rather than estimated jointly with phase.

    Method: the fringe pattern shows up as a carrier peak in each frame's
    2D spectrum (present even with no deliberate off-axis tilt, since the
    interferometer's residual mechanical tilt/curvature already produces
    one -- exactly what :func:`~phase.carrier.remove_carrier` also relies
    on). The carrier's *location* in frequency is fixed by the optics, but
    a per-frame amplitude drift, tilt jitter, or defocus jitter can shift
    its exact bin slightly frame to frame; a single-bin readout would then
    conflate that jitter with a real contrast change. Integrating ``|F|^2``
    over a small neighborhood around the peak (located once, from the
    frame-summed spectrum) makes the estimate robust to that jitter while
    still being insensitive to phase content elsewhere in the field.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted interferogram frames (same input as ``aia``).
    dc_radius : int, default 8
        Half-size, in FFT bins, of the neighborhood around DC excluded
        before locating the carrier peak (avoids picking up the strong
        low-frequency background/illumination term instead of the
        fringes).
    halfwin : (int, int), default (3, 4)
        Half-size, in FFT bins along (row, column), of the neighborhood
        integrated around the located carrier peak.

    Returns
    -------
    np.ndarray, shape (N,)
        Per-frame contrast, normalized so ``median(g) = 1``. Pass this as
        ``aia(..., gain=g)``, or use ``gain="auto"`` to have ``aia`` call
        this internally.
    """
    N, H, W = stack.shape
    win = np.outer(np.hanning(H), np.hanning(W)) if H > 1 and W > 1 else np.ones((H, W))
    F = np.fft.rfft2(stack.astype(float) * win, axes=(1, 2))   # (N, H, W//2+1)

    P = np.abs(F).sum(0)
    P[:dc_radius, :dc_radius] = 0
    P[-dc_radius:, :dc_radius] = 0
    iy, ix = np.unravel_index(np.argmax(P), P.shape)

    hy, hx = halfwin
    rows = [(iy + k) % F.shape[1] for k in range(-hy, hy + 1)]
    c0, c1 = max(ix - hx, 0), min(ix + hx + 1, F.shape[2])
    amp = np.sqrt((np.abs(F[:, rows, :][:, :, c0:c1]) ** 2).sum(axis=(1, 2)))

    return amp / np.median(amp)


def aia(stack: np.ndarray, delta0: Optional[np.ndarray] = None,
        iters: int = 30, tol: float = 1e-4, gain=None) -> AIAResult:
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
    frame's fringe contrast relative to the shared map ``b`` (``g_n = 1``
    for all ``n`` if not supplied -- see ``gain``). For fixed ``delta_n``
    and ``g_n`` this is linear in ``(a, u, v)``, and for fixed ``(u, v)``
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
    gain : None, "auto", or np.ndarray of shape (N,), default None
        Per-frame fringe contrast ``g_n`` (see Model). ``None`` assumes
        ``g_n = 1`` for every frame (original behavior). ``"auto"`` calls
        :func:`measure_frame_contrast` internally. An array supplies
        pre-measured values directly (e.g. from a calibration shot, or
        computed once and reused). This must be *measured independently*,
        not fit jointly with ``(u, v)`` -- see Notes for why.

    Returns
    -------
    AIAResult
        See :class:`AIAResult` for field descriptions. Physical outputs
        are ``phi, b, a, delta, g``; ``kappa_p, kappa_ps, predicted_rms``
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
    An earlier version of this function also tried subtracting the
    background ``a`` in the frame step; this was removed because it
    increased leaked oscillations into the recovered background rather
    than reducing them. Chen & Kemao (2019) explain why: the background
    decouples from the fringe terms when the phase-shift distribution is
    well conditioned, so accuracy should be controlled via ``delta0``
    spacing and frame count -- monitored here with ``kappa_p``/
    ``kappa_ps`` -- rather than by subtracting a still-imperfect,
    mid-iteration background estimate.

    An earlier version also tried estimating ``g_n`` *jointly* with
    ``(u, v)`` -- adding it as a free per-frame unknown in the frame step,
    analogous to how ``delta_n`` is estimated -- and abandoned it for
    locking onto "a self-consistent but incorrect fixed point". The reason
    is structural, not a tuning issue: with ``g_n`` free, the pixel-step
    model ``I_n = a + P_n*u + Q_n*v`` (writing ``P_n = g_n*cos(delta_n)``,
    ``Q_n = g_n*sin(delta_n)``) is invariant under *any* invertible linear
    transform ``T``: replacing ``(u, v) -> T(u, v)`` and
    ``(P_n, Q_n) -> (P_n, Q_n)*T^-1`` for every frame leaves every
    predicted ``I_n`` unchanged. Pinning the phase origin
    (``delta[0] = 0``) and a contrast scale (e.g. ``median(g) = 1``) fixes
    only 2 of that group's 4 degrees of freedom; the remaining 2 (a shear
    plus a rotation/reflection ambiguity beyond the intended
    ``phi -> -phi`` one, see :func:`~phase.reference.subtract_reference`)
    are free to drift, and on real data measurably do: the recovered
    ``(P_n, Q_n)`` collapse onto a sheared, anisotropic ellipse rather than
    the intended circle, which corrupts ``phi = arctan2(-v, u)``
    everywhere. This is why ``gain`` above requires ``g_n`` to be supplied
    or measured independently of the AIA solve (see
    :func:`measure_frame_contrast`) rather than fit alongside it -- an
    independent measurement breaks the invariance the free-parameter
    version could not.

    A spatially-varying per-frame phase step,
    ``delta_n(x, y) = d_n + p_n*x + q_n*y`` (e.g. from a reference mirror
    tilting slightly as a piezo stage steps), was also investigated as a
    possible source of residual error and is *not* worth modeling: measured
    directly from real data (by coherently averaging each frame's
    Takeda-demodulated field, relative to frame 0, over a coarse patch grid,
    then plane-fitting the patch phases -- plain per-pixel or per-bin
    gradient estimates are unusable here, since that field spans ~13 orders
    of magnitude in amplitude and a naive weighted-gradient sum is dominated
    by a handful of near-zero-amplitude pixels), the tilt was only ~5-8
    degrees peak-to-peak, and holding ``(p_n, q_n)`` fixed at those measured
    values in a per-pixel version of this solve changed a real null-test
    residual (two independent bare-glass acquisitions, true difference
    zero) by 0.002 degrees -- noise-level, not worth the added cost. The
    residual that remains after ``gain`` is corrected is dominated by
    genuine acquisition-to-acquisition random scatter (measured at ~1.15
    degrees per acquisition on this setup) rather than any per-frame model
    term; average multiple independent acquisitions of the same object
    (see :func:`~phase.combine.combine_acquisitions`) to bring it down
    further -- it was verified to follow the expected ``1/sqrt(k)`` scaling.
    """
    N, H, W = stack.shape
    I = stack.reshape(N, -1).astype(float)          # (N, P)
    if delta0 is None:
        delta0 = np.arange(N) * 2 * np.pi / N
    delta = np.asarray(delta0, float).copy()

    if gain is None:
        g = np.ones(N)
    elif isinstance(gain, str) and gain == "auto":
        g = measure_frame_contrast(stack)
    else:
        g = np.asarray(gain, float)

    u = v = a = None
    A = B = None
    converged = False
    it = 0
    for it in range(iters):
        # pixel step: I_n = a + g_n*(u cosδ_n + v sinδ_n),  u=b cosφ, v=-b sinφ
        c, s = np.cos(delta), np.sin(delta)
        A = np.column_stack([np.ones(N), g * c, g * s])                # (N,3)
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
        phi=phi, b=b, a=a_map, delta=delta, g=g,
        kappa_p=kappa_p, kappa_ps=kappa_ps, predicted_rms=predicted_rms,
        iters_run=it + 1, converged=converged,
    )
