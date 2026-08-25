"""Advanced Iterative Algorithm (AIA) for phase-shifting interferometry."""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import backend as _backend
from .backend import get_array_module, to_device, wrap


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

    Notes
    -----
    ``phi, b, a, delta, g`` are numpy or cupy arrays depending on ``aia``'s
    ``device`` argument -- see its docstring. They are not forced back to
    the host, so that passing them straight into
    :func:`~phase.carrier.remove_carrier` or
    :func:`~phase.combine.combine_acquisitions` keeps large arrays resident
    on the GPU. Call :func:`phase.backend.asnumpy` on a field yourself (e.g.
    before plotting or ``np.savez``) when you need a guaranteed-numpy array.
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
    ``(N, P)`` array purely to reduce it to one scalar -- at the real
    acquisition size that is the single largest transient allocation in
    ``aia`` (measured: ~3.5s and another 1.7GB on top of ``I`` itself).
    Streaming over pixel chunks with a float64 accumulator gives a
    bit-identical result (verified on real data) at a small, fixed peak
    memory, and each chunk's matmul is small enough to stay well under a
    display-GPU's watchdog kernel-timeout.
    """
    N, P = I.shape
    A_work = A.astype(I.dtype)
    ssq = 0.0
    for s in range(0, P, chunk):
        resid = I[:, s:s + chunk] - A_work @ X[:, s:s + chunk]
        ssq += float(xp.sum(resid.astype(xp.float64) ** 2))
    return float(np.sqrt(ssq / (N * P)))


def _carrier_dc_amplitudes(stack: np.ndarray, dc_radius: int = 8,
                            halfwin: tuple = (3, 4), frame_chunk: int = 8,
                            dtype=None) -> tuple:
    """Per-frame carrier and DC amplitude, shared by :func:`measure_frame_contrast`
    and :func:`measure_frame_visibility`.

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
    still being insensitive to phase content elsewhere in the field. The
    DC bin (``F[:, 0, 0]``, read off in the same pass, no extra FFT) is the
    windowed-field analog of the background term ``a`` -- see
    :func:`measure_frame_visibility`.

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
    frame_chunk : int, default 8
        Number of frames' FFTs held resident at once. The original
        implementation computed ``rfft2`` over the whole stack in one call;
        at the real acquisition size (30 frames, full ROI) that is >1.7 GB of
        complex output alone, which does not fit a memory-constrained GPU
        (e.g. a 4 GB Quadro). Streaming over frame chunks -- first to locate
        the carrier peak from a running frame-summed spectrum, then again to
        read off each frame's amplitude at that peak -- gives the identical
        result (verified) at a small, fixed peak memory regardless of ``N``.
        Irrelevant to correctness; lower only matters for constrained VRAM.
    dtype : numpy/cupy dtype, optional
        Working (real) dtype for the per-chunk FFT input; the FFT itself then
        runs in the matching complex dtype (e.g. float32 in, complex64 out).
        Defaults to ``float32`` (see :func:`phase.backend.default_dtype`).
        The frame-summed peak-location accumulator and the per-frame energy
        accumulator always stay float64 regardless of this setting, since
        they're tiny (``(H, W//2+1)`` and ``(N,)``) -- verified against a
        float64-throughout run on real data at 2e-8 relative error in ``g``,
        far below the per-frame contrast swings (tens of percent) this is
        meant to track.

    Returns
    -------
    carrier_amp, dc_amp : np.ndarray, each shape (N,)
        Per-frame carrier-peak amplitude and DC (zero-frequency) amplitude,
        both un-normalized (raw FFT-domain magnitudes, not divided by
        anything).
    """
    xp = get_array_module(stack)
    work_dtype = dtype if dtype is not None else _backend.default_dtype(xp)
    N, H, W = stack.shape
    win = (xp.outer(xp.hanning(H), xp.hanning(W)) if H > 1 and W > 1
           else xp.ones((H, W))).astype(work_dtype)
    Wc = W // 2 + 1

    # pass 1: locate the carrier peak from the frame-summed spectrum,
    # streamed over frame chunks so at most `frame_chunk` frames' FFTs are
    # resident at once.
    Psum = xp.zeros((H, Wc), dtype=xp.float64)
    for s in range(0, N, frame_chunk):
        block = stack[s:s + frame_chunk].astype(work_dtype) * win
        Psum += xp.abs(xp.fft.rfft2(block, axes=(1, 2))).astype(xp.float64).sum(0)
    Psum[:dc_radius, :dc_radius] = 0
    Psum[-dc_radius:, :dc_radius] = 0
    iy, ix = xp.unravel_index(xp.argmax(Psum), Psum.shape)
    iy, ix = int(iy), int(ix)

    hy, hx = halfwin
    rows = xp.asarray([(iy + k) % H for k in range(-hy, hy + 1)])
    c0, c1 = max(ix - hx, 0), min(ix + hx + 1, Wc)

    # pass 2: per-frame amplitude at the carrier peak and at DC, same chunking.
    amp_sq = xp.empty(N, dtype=xp.float64)
    dc_amp = xp.empty(N, dtype=xp.float64)
    for s in range(0, N, frame_chunk):
        block = stack[s:s + frame_chunk].astype(work_dtype) * win
        Fc = xp.fft.rfft2(block, axes=(1, 2))
        amp_sq[s:s + block.shape[0]] = (
            xp.abs(Fc[:, rows, :][:, :, c0:c1]).astype(xp.float64) ** 2
        ).sum(axis=(1, 2))
        dc_amp[s:s + block.shape[0]] = xp.abs(Fc[:, 0, 0]).astype(xp.float64)

    return xp.sqrt(amp_sq), dc_amp


def measure_frame_contrast(stack: np.ndarray, dc_radius: int = 8,
                            halfwin: tuple = (3, 4), frame_chunk: int = 8,
                            dtype=None) -> np.ndarray:
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

    See :func:`_carrier_dc_amplitudes` for the carrier-peak method and all
    parameters (identical here).

    Returns
    -------
    np.ndarray, shape (N,)
        Per-frame contrast, normalized so ``median(g) = 1``. Pass this as
        ``aia(..., gain=g)``, or use ``gain="auto"`` to have ``aia`` call
        this internally. Relative *within this stack only* -- use
        :func:`measure_frame_visibility` instead to compare contrast across
        separately-captured frames or stacks (e.g. a piezo coherence scan).
    """
    xp = get_array_module(stack)
    amp, _ = _carrier_dc_amplitudes(stack, dc_radius, halfwin, frame_chunk, dtype)
    return amp / xp.median(amp)


def measure_frame_visibility(stack: np.ndarray, dc_radius: int = 8,
                              halfwin: tuple = (3, 4), frame_chunk: int = 8,
                              dtype=None) -> np.ndarray:
    """Measure each frame's fringe visibility, absolutely (not stack-relative).

    ``aia`` fits ``I_n = a + g_n*b*cos(phi + delta_n)``, so the *true*
    visibility of a fringe pattern is ``b/a`` (Michelson's
    ``(Imax-Imin)/(Imax+Imin)``). :func:`measure_frame_contrast` measures
    something proportional to ``b`` but normalizes it to ``median=1`` within
    its input stack, which makes it useless for comparing contrast *across*
    stacks -- e.g. one frame per position of a piezo coherence scan, where
    the whole point is to compare visibility across many separate captures.

    This returns ``2 * carrier_amp / dc_amp`` per frame (see
    :func:`_carrier_dc_amplitudes`): ``dc_amp`` is the windowed field's
    zero-frequency term, the same-pass analog of ``a``; the factor of 2
    accounts for a real cosine's energy splitting between the ``+`` and
    ``-`` carrier frequency, of which only the ``+`` side is integrated.
    For a field that is exactly ``a + b*cos(carrier)`` with no other
    spatial structure, this equals ``b/a`` exactly modulo windowing.

    In practice it is *proportional* to ``b/a``, not exactly equal --
    the Hann window used for peak-location suppresses the carrier and DC
    terms by different net factors, and real object phase structure (not
    just a pure linear carrier) spreads some of the carrier peak's energy
    into neighboring bins outside ``halfwin``. Both effects are constant
    for one fixed setup and ROI, so the value is safe to compare against
    itself over time or across piezo position -- e.g. as the per-frame
    metric for a coherence scan, or a coherence-zone drift log -- but is
    not a calibrated absolute number to compare across different setups
    or ROIs.

    See :func:`_carrier_dc_amplitudes` for all parameters (identical here).

    Returns
    -------
    np.ndarray, shape (N,)
        Per-frame visibility, proportional to ``b/a``. Not normalized --
        comparable across stacks, unlike :func:`measure_frame_contrast`.
    """
    xp = get_array_module(stack)
    amp, dc_amp = _carrier_dc_amplitudes(stack, dc_radius, halfwin, frame_chunk, dtype)
    dc_amp = xp.where(dc_amp > 0, dc_amp, xp.asarray(xp.finfo(xp.float64).eps))
    return 2.0 * amp / dc_amp


def aia(stack: np.ndarray, delta0: Optional[np.ndarray] = None,
        iters: int = 30, tol: float = 1e-4, gain=None,
        device: str = "auto", dtype=None) -> AIAResult:
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
    device : {"auto", "cpu", "cuda"}, default "auto"
        Where to run. ``"auto"`` uses a GPU (via cupy) if one is installed,
        else the CPU (via numpy) -- both give the same result to the
        precision of ``dtype`` below. ``"cpu"``/``"cuda"`` force one or the
        other, uploading ``stack`` if needed (raises if ``"cuda"`` is
        requested without cupy installed). The result's array fields
        (``phi, b, a, delta, g``) live on whichever device ran the
        computation -- pass ``device="cpu"`` for a guaranteed-numpy result,
        or call :func:`phase.backend.asnumpy` on them yourself; they are
        *not* forced back to the host automatically, so that chaining
        ``aia`` -> :func:`~phase.combine.combine_acquisitions` on a GPU
        doesn't round-trip large arrays over PCIe in between.
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
        not by this dtype -- verified against a float64 run on real data at
        <1e-4 degrees RMS.

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

    """
    stack = to_device(stack, device=device)
    xp = get_array_module(stack)
    N, H, W = stack.shape
    work_dtype = dtype if dtype is not None else _backend.default_dtype(xp)
    I = stack.reshape(N, -1).astype(work_dtype, copy=False)        # (N, P)
    P = I.shape[1]

    if delta0 is None:
        delta0 = xp.arange(N) * 2 * xp.pi / N
    delta = xp.asarray(delta0, dtype=xp.float64).copy()

    if gain is None:
        g = xp.ones(N, dtype=xp.float64)
    elif isinstance(gain, str) and gain == "auto":
        g = measure_frame_contrast(stack, dtype=work_dtype)
    else:
        g = xp.asarray(gain, dtype=xp.float64)

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
    # coverage (see AIAResult.kappa_ps docstring). Assembled here from five
    # scalar reductions rather than materializing a (P,3) design matrix
    # ``C`` just to form ``C.T @ C`` -- same 3x3 Gram matrix, ~200x less
    # peak memory at the real acquisition size.
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

    return AIAResult(
        phi=phi, b=b, a=a_map, delta=delta, g=g,
        kappa_p=kappa_p, kappa_ps=kappa_ps, predicted_rms=predicted_rms,
        iters_run=it + 1, converged=converged,
    )
