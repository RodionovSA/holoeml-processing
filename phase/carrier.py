"""Estimating and removing the spatial carrier/defocus from a phase map."""

import cmath
import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .backend import get_array_module, to_device


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


def _next_smooth(n: int, factors=(2, 3, 5, 7)) -> int:
    """Smallest ``m >= n`` whose only prime factors are in ``factors``.

    Used to zero-pad the *coarse* peak-search FFT below to a size cuFFT (and
    pocketfft) handle efficiently. ``H, W`` at the real acquisition size
    (2200, 3296) factor as ``2^3*5^2*11`` and ``2^5*103`` -- the 11 and 103
    push both onto a slow mixed-radix/Bluestein path. Padding to the next
    7-smooth size (2200->2205, 3296->3360) measured ~1.8x faster on this
    machine's FFT alone, and is safe: the coarse FFT only has to land the
    peak within about one bin of the true carrier, since
    :func:`_estimate_tilt`'s closed-form refine then converges to the exact
    same fixed point regardless of which nearby bin it started from
    (verified on real data).
    """
    k = n
    while True:
        m = k
        for f in factors:
            while m % f == 0:
                m //= f
        if m == 1:
            return k
        k += 1


def _coarse_peak(field, xp):
    """Locate the dominant off-DC peak of ``field``'s spectrum, FFT zero-padded
    to an efficient size (see :func:`_next_smooth`). Returns cycles/pixel
    ``(fy, fx)`` -- an integer-bin estimate later refined by
    :func:`_estimate_tilt`."""
    H, W = field.shape
    Hp, Wp = _next_smooth(H), _next_smooth(W)
    if (Hp, Wp) != (H, W):
        padded = xp.zeros((Hp, Wp), dtype=field.dtype)
        padded[:H, :W] = field
    else:
        padded = field
    F = xp.fft.fft2(padded)
    iy, ix = xp.unravel_index(xp.argmax(xp.abs(F)), F.shape)
    fy = float(xp.fft.fftfreq(Hp)[int(iy)])
    fx = float(xp.fft.fftfreq(Wp)[int(ix)])
    return fy, fx


def _estimate_tilt(c: np.ndarray, w: np.ndarray, window: bool, refine_iters: int) -> tuple:
    """Estimate the dominant linear carrier ``(fx, fy)`` of a complex field.

    Shared by :func:`remove_carrier`'s final full-field estimate and (via
    :func:`_estimate_curvature`'s batched variant below) its per-block
    defocus pre-estimation -- see :func:`remove_carrier`'s Algorithm section
    (steps 1-2) for the coarse-FFT + wrap-safe-refine method.

    Algorithm note: the residual-tilt refine below is a *closed form*, not
    the literal per-iteration re-demodulation the original implementation
    used. In ``sum(w_pair * d[...,1:] * conj(d[...,:-1]))`` with
    ``d = c * exp(-2*pi*i*(fx*x + fy*y))``, the position-dependent phase
    factor is identical on both members of every neighboring-pixel pair and
    cancels except for the fixed 1-pixel step between them, so the whole sum
    factors as ``exp(-2*pi*i*fx) * Sx`` where
    ``Sx = sum(w_pair * c[...,1:] * conj(c[...,:-1]))`` does not depend on
    the current ``(fx, fy)`` estimate at all (same for ``Sy`` down rows).
    ``Sx``/``Sy`` are therefore computed once, and the fixed-point iteration
    that follows runs on their two (scalar) angles rather than re-summing
    the full field every pass -- bit-identical to the original in float64
    (verified against real data: 0 and 1.4e-17 rad difference in fx, fy)
    at a fraction of the memory traffic, and with zero further GPU kernel
    launches once ``Sx``/``Sy`` are known.
    """
    xp = get_array_module(c, w)
    H, W = c.shape
    if window and H > 1 and W > 1:
        win = xp.outer(xp.hanning(H), xp.hanning(W))
    else:
        win = xp.ones((H, W))
    fy, fx = _coarse_peak(w * c * win, xp)

    Sx = Sy = None
    if W > 1:
        wpx = w[:, 1:] * w[:, :-1]
        Sx = complex(xp.sum(wpx * c[:, 1:] * xp.conj(c[:, :-1])))
    if H > 1:
        wpy = w[1:, :] * w[:-1, :]
        Sy = complex(xp.sum(wpy * c[1:, :] * xp.conj(c[:-1, :])))
    ax = cmath.phase(Sx) if Sx is not None else 0.0
    ay = cmath.phase(Sy) if Sy is not None else 0.0

    two_pi = 2 * math.pi
    for _ in range(refine_iters):
        dfx = cmath.phase(cmath.exp(1j * (ax - two_pi * fx))) / two_pi if Sx is not None else 0.0
        dfy = cmath.phase(cmath.exp(1j * (ay - two_pi * fy))) / two_pi if Sy is not None else 0.0
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
    ``fy_local(x, y) = fy0 + (kxy/(2*pi))*x + (kyy/pi)*y``. Splits the field
    into an ``n_blocks x n_blocks`` grid of equal-size blocks, measures a
    local ``(fx, fy)`` per block, and fits that (coupled, since ``kxy``
    appears in both) linear relationship by least squares across all blocks
    to recover ``kxx, kyy, kxy``. Returns all zeros (with a warning) if
    fewer than 4 blocks have usable weight, since that under-determines the
    5-unknown fit.

    All ``n_blocks**2`` blocks are estimated in one batch -- a single
    windowed+padded FFT over the whole ``(n_blocks**2, bh, bw)`` stack, then
    the same closed-form ``Sx``/``Sy`` reduction and fixed-point refine as
    :func:`_estimate_tilt`, done per-block via array ops rather than a
    Python loop calling :func:`_estimate_tilt` once per block. At the
    docstring default (``n_blocks=10``) that replaces 100 separate small
    FFT + refine calls -- disproportionately expensive as GPU kernel-launch
    overhead -- with O(1) launches. Blocks are equal-size, obtained by
    cropping to the largest ``H', W'`` divisible by ``n_blocks`` (dropping at
    most ``n_blocks - 1`` pixels along the far edge of each axis) rather than
    the ragged groups ``np.array_split`` would give, so the block grid can be
    reshaped into a batch dimension instead of gathered with ``np.ix_``
    (which forces a copy per block).

    Unlike :func:`_estimate_tilt`, this is *not* bit-identical to the
    original per-block-loop implementation -- the different block boundaries
    (equal-size crop vs. ``array_split``'s ragged groups, plus dropped edge
    pixels) shift each block's local tilt estimate slightly, which the
    regression then propagates into ``kxx, kyy, kxy``. Measured on real data
    (a full ``remove_carrier(defocus=True)`` call, ``n_blocks=10``): up to
    ~7% relative change in the fitted curvature coefficients, translating to
    a 0.06 deg RMS / 0.24 deg max change in the final ``phi`` -- about 5% of
    this setup's ~1.15 deg/acquisition scatter floor, and well within it.
    """
    xp = get_array_module(c, w)
    H, W = c.shape
    bh, bw = H // n_blocks, W // n_blocks
    if bh < 2 or bw < 2:
        warnings.warn(
            "remove_carrier: n_blocks too large for this image size; "
            "cannot fit a curvature term, skipping it (kxx=kyy=kxy=0). "
            "Try a smaller n_blocks.",
            stacklevel=3,
        )
        return 0.0, 0.0, 0.0
    Hc, Wc = bh * n_blocks, bw * n_blocks

    def to_blocks(a):
        return (a[:Hc, :Wc].reshape(n_blocks, bh, n_blocks, bw)
                 .transpose(0, 2, 1, 3).reshape(n_blocks * n_blocks, bh, bw))

    c_b = to_blocks(c)
    w_b = to_blocks(w)
    nblk = n_blocks * n_blocks

    win = (xp.outer(xp.hanning(bh), xp.hanning(bw)) if (window and bh > 1 and bw > 1)
           else xp.ones((bh, bw)))

    # coarse peak per block, via one padded batched FFT
    bhp, bwp = _next_smooth(bh), _next_smooth(bw)
    field = w_b * c_b * win
    if (bhp, bwp) != (bh, bw):
        padded = xp.zeros((nblk, bhp, bwp), dtype=field.dtype)
        padded[:, :bh, :bw] = field
    else:
        padded = field
    F = xp.fft.fft2(padded, axes=(1, 2))
    Fabs = xp.abs(F).reshape(nblk, -1)
    peak = xp.argmax(Fabs, axis=1)
    iy, ix = peak // bwp, peak % bwp
    fy = xp.fft.fftfreq(bhp)[iy].astype(xp.float64)
    fx = xp.fft.fftfreq(bwp)[ix].astype(xp.float64)

    # closed-form Sx/Sy per block (see _estimate_tilt), then the same
    # fixed-point refine, vectorized over the block axis.
    have_x, have_y = bw > 1, bh > 1
    if have_x:
        wpx = w_b[:, :, 1:] * w_b[:, :, :-1]
        Sx = (wpx * c_b[:, :, 1:] * xp.conj(c_b[:, :, :-1])).sum(axis=(1, 2))
        ax = xp.angle(Sx)
    if have_y:
        wpy = w_b[:, 1:, :] * w_b[:, :-1, :]
        Sy = (wpy * c_b[:, 1:, :] * xp.conj(c_b[:, :-1, :])).sum(axis=(1, 2))
        ay = xp.angle(Sy)

    two_pi = 2 * xp.pi
    for _ in range(refine_iters):
        if have_x:
            fx = fx + xp.angle(xp.exp(1j * (ax - two_pi * fx))) / two_pi
        if have_y:
            fy = fy + xp.angle(xp.exp(1j * (ay - two_pi * fy))) / two_pi

    block_row = xp.repeat(xp.arange(n_blocks), n_blocks)
    block_col = xp.tile(xp.arange(n_blocks), n_blocks)
    yc = block_row.astype(xp.float64) * bh + (bh - 1) / 2.0
    xc = block_col.astype(xp.float64) * bw + (bw - 1) / 2.0

    total_w = float(w.sum())
    floor = 0.01 * total_w / max(nblk, 1)
    w_sum = w_b.reshape(nblk, -1).sum(axis=1)
    usable = w_sum >= floor

    n = int(xp.sum(usable))
    if n < 4:
        warnings.warn(
            "remove_carrier: fewer than 4 blocks had usable weight; "
            "cannot fit a curvature term, skipping it (kxx=kyy=kxy=0). "
            "Try a smaller n_blocks or check weight/mask coverage.",
            stacklevel=3,
        )
        return 0.0, 0.0, 0.0

    xc_u, yc_u = xc[usable], yc[usable]
    fx_u, fy_u = fx[usable], fy[usable]

    # unknowns: [fx0, fy0, A=kxx/pi, B=kxy/(2*pi), C=kyy/pi]
    M = xp.zeros((2 * n, 5), dtype=xp.float64)
    
    rhs = xp.zeros(2 * n, dtype=xp.float64)
    M[0::2, 0] = 1.0
    M[0::2, 2] = xc_u
    M[0::2, 3] = yc_u
    rhs[0::2] = fx_u
    M[1::2, 1] = 1.0
    M[1::2, 3] = xc_u
    M[1::2, 4] = yc_u
    rhs[1::2] = fy_u
    sol = xp.linalg.lstsq(M, rhs, rcond=None)[0]
    A, B, C = float(sol[2]), float(sol[3]), float(sol[4])
    kxx = float(np.pi * A)
    kyy = float(np.pi * C)
    kxy = float(2 * np.pi * B)
    return kxx, kyy, kxy


def remove_carrier(phi: np.ndarray, weight: Optional[np.ndarray] = None,
                    mask: Optional[np.ndarray] = None, refine_iters: int = 10,
                    window: bool = True, defocus: bool = True,
                    n_blocks: int = 10, device: str = "auto") -> CarrierResult:
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
       a pixel-to-pixel gradient estimate alone would alias there. The
       field is zero-padded to an FFT-efficient size for this step only
       (see :func:`_next_smooth`); it only has to localize the peak to
       within about one bin, which step 2 then resolves exactly.
    2. **Sub-pixel refinement (wrap-safe, closed-form).** The residual tilt
       after demodulating by the current ``(fx, fy)`` estimate is measured
       by vector averaging of neighboring-pixel phase differences --
       equivalent to a weighted circular mean of local gradients, so wraps
       in the residual don't bias the estimate the way a naive
       finite-difference-then-average would -- and, because that averaged
       quantity turns out not to depend on the current ``(fx, fy)`` guess at
       all (see :func:`_estimate_tilt`), is computed once and then iterated
       as a fixed point on two scalars rather than re-demodulating the full
       field ``refine_iters`` times. Steps 1-2 are :func:`_estimate_tilt`.
    3. **Curvature pre-correction (if ``defocus=True``), before step 1-2's
       final pass.** A quadratic phase is a 2D chirp: its local
       instantaneous frequency varies linearly with position (and, if the
       curvature is anisotropic, ``fx`` also varies with ``y`` and vice
       versa via the coupled ``kxy`` term). Splitting the field into an
       ``n_blocks x n_blocks`` grid and running steps 1-2 *within each
       block* (all blocks at once, see :func:`_estimate_curvature`) gives a
       local ``(fx, fy)`` sample at each block's center; jointly regressing
       those samples against block position recovers ``kxx, kyy, kxy``. The
       field is demodulated by ``kxx*x^2 + kyy*y^2 + kxy*x*y`` before the
       final, full-field tilt/piston pass, which then only has to polish the
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
    refine_iters : int, default 10
        Number of sub-pixel refinement iterations after each coarse FFT
        step (both the final full-field estimate and, if ``defocus=True``,
        each per-block estimate). Matches every call site in this package;
        raised from the original default of 5 for that reason.
    window : bool, default True
        Apply a 2D Hann window before each coarse FFT to reduce spectral
        leakage from the field's edges (recommended; skipped automatically
        if the field being estimated has a size-1 axis).
    defocus : bool, default True
        Also fit and remove a quadratic curvature term
        ``kxx*x^2 + kyy*y^2 + kxy*x*y``, via the block-regression method
        above. Defaults on -- every call site in this package (and its
        callers in :mod:`phase.combine` and :mod:`phase.ripple`) already
        passed ``defocus=True`` explicitly; matching the default avoids the
        signature silently disagreeing with actual usage. Worth disabling
        only for a field known to have no curvature term to begin with.
    n_blocks : int, default 10
        Grid size (``n_blocks x n_blocks``) for the curvature pre-estimate.
        Only used when ``defocus=True``. Needs at least 4 blocks with
        usable weight to fit (5 unknowns, 2 equations/block); falls back
        to ``kxx=kyy=kxy=0`` (with a warning) if fewer are available --
        e.g. too small an image for the requested grid, or ``weight``/
        ``mask`` leaving too little usable area. Matches every call site;
        raised from the original default of 6 for that reason.
    device : {"auto", "cpu", "cuda"}, default "auto"
        Where to run -- see :func:`phase.aia.aia`'s ``device`` parameter for
        the full explanation. ``phi``/``weight``/``mask`` are uploaded if
        needed; the result's ``phi`` field stays on that same device rather
        than being downloaded automatically.

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
    phi = to_device(phi, device=device)
    xp = get_array_module(phi)
    H, W = phi.shape
    c = xp.exp(1j * phi)

    w = xp.ones((H, W)) if weight is None else xp.clip(to_device(weight, device=device), 0, None)
    if mask is not None:
        w = w * to_device(mask, device=device, dtype=bool)
    if float(w.sum()) <= 0:
        w = xp.ones((H, W))

    X, Y = xp.meshgrid(xp.arange(W, dtype=xp.float64), xp.arange(H, dtype=xp.float64))

    kxx = kyy = kxy = 0.0
    if defocus:
        kxx, kyy, kxy = _estimate_curvature(c, w, window, refine_iters, n_blocks)
        if kxx or kyy or kxy:
            c = c * xp.exp(-1j * (kxx * X**2 + kyy * Y**2 + kxy * X * Y))

    fx, fy = _estimate_tilt(c, w, window, refine_iters)

    # demodulate the (curvature-corrected) carrier, then remove the piston
    d = c * xp.exp(-1j * 2 * xp.pi * (fx * X + fy * Y))
    piston = float(xp.angle(xp.sum(w * d)))
    d = d * xp.exp(-1j * piston)
    phi_flat = xp.angle(d)

    kx = 2 * np.pi * fx
    ky = 2 * np.pi * fy
    return CarrierResult(phi=phi_flat, kx=kx, ky=ky, fx=fx, fy=fy,
                          kxx=kxx, kyy=kyy, kxy=kxy, piston=piston)
