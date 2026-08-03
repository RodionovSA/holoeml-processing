"""Estimating and removing the spatial carrier/defocus from a phase map."""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np


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
