"""Estimating and removing a phase-locked ripple error, eps(phi)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .backend import get_array_module, to_device, wrap
from .carrier import remove_carrier


@dataclass
class RippleResult:
    """Output of :func:`estimate_phase_ripple`.

    Attributes
    ----------
    coeffs : dict
        ``{0: c0}`` plus ``{k: (a_k, b_k) for k in orders}``, the fitted
        Fourier series ``eps(phi) = c0 + sum_k a_k*cos(k*phi) + b_k*sin(k*phi)``.
    orders : tuple of int
        Harmonic orders that were fit (as passed in).
    bin_centers : np.ndarray, shape (nbins,)
        Phase bin centers used for the fit, in ``(-pi, pi]``.
    lut : np.ndarray, shape (nbins,)
        The measured (pre-fit) circular-mean residual at each bin center --
        a diagnostic for inspecting/plotting the raw ripple shape and
        judging whether ``orders`` captures it.
    bin_weight : np.ndarray, shape (nbins,)
        Total weight that landed in each bin (0 where ``lut`` is undefined).
    rms_before : float
        RMS of the input residual, in radians, before fitting.
    rms_after : float
        RMS of ``residual - eps(phi)`` evaluated at every sample used for
        the fit, in radians -- how much of the ripple the fit captured.
    """

    coeffs: dict
    orders: tuple
    bin_centers: np.ndarray
    lut: np.ndarray
    bin_weight: np.ndarray
    rms_before: float
    rms_after: float


def estimate_phase_ripple(phi: np.ndarray, mask: np.ndarray,
                           weight: Optional[np.ndarray] = None,
                           orders: tuple = (1, 2, 3, 4), nbins: int = 180,
                           carrier_kwargs: Optional[dict] = None,
                           device: str = "auto") -> RippleResult:
    """Estimate a phase-locked ripple error from a region of known-flat phase.

    ``aia`` can leave a residual error that is a deterministic function of
    the recovered phase itself, ``eps(phi)``, rather than of position --
    the signature of an imperfect frame model (e.g. per-frame contrast
    treated as constant when it isn't, see ``aia``'s ``gain`` parameter).
    Because it tracks phase, not position, it doesn't average out spatially
    and isn't separable from real structure by a spatial filter (e.g. FFT)
    when the two overlap in spatial frequency -- but it *is* separable in
    the phase domain, where the real object signal has no reason to
    correlate with wrapped phase value while the ripple is defined by it.

    This estimates ``eps(phi)`` from a region where the true phase is known
    to be smooth (e.g. bare substrate surrounding a structure): after
    :func:`~phase.carrier.remove_carrier` strips the smooth tilt/defocus/
    piston trend, whatever is left in that region should be ~0 plus
    ``eps(phi)`` plus noise. Binning that leftover by the *original*
    (pre-carrier-removal) wrapped phase and fitting a low-order Fourier
    series isolates ``eps(phi)``, which :func:`apply_phase_ripple` then
    removes from the *entire* map (mask region and structure alike) -- the
    correction is evaluated pointwise from each pixel's own phase value,
    so it does not blur, smooth, or otherwise touch spatial resolution.

    Apply this to the sample and reference phase maps *separately*, before
    :func:`~phase.reference.subtract_reference` -- their ripples generally
    differ (different per-frame contrast/step sequences), so in the
    difference they beat against each other into a low spatial frequency
    instead of canceling; correcting each map on its own is what actually
    removes it.

    Parameters
    ----------
    phi : np.ndarray, shape (H, W)
        Wrapped phase map to estimate the ripple from (e.g. ``AIAResult.phi``,
        *before* carrier removal).
    mask : np.ndarray, shape (H, W)
        Boolean (or 0/1) map selecting the known-flat region (e.g. thresholded
        modulation map, or a hand-drawn ROI excluding the structure).
    weight : np.ndarray, shape (H, W), optional
        Per-pixel reliability (e.g. the modulation map ``b``), combined with
        ``mask``. Negative values are clipped to 0.
    orders : tuple of int, default (1, 2, 3, 4)
        Harmonic orders to fit. Start with low orders and inspect
        ``RippleResult.lut`` vs. the fit before adding more -- high orders
        fit noise if the true ripple is simple.
    nbins : int, default 180
        Number of phase bins (2 degrees/bin at the default) used to
        circular-mean-average samples before fitting. This is a robustness
        step for the *fit* only (it reduces millions of noisy per-pixel
        samples to `nbins` robust points) -- :func:`apply_phase_ripple`
        evaluates the fitted analytic ``eps(phi)`` at each pixel's exact
        phase value, so the correction itself is not binned/quantized.
    carrier_kwargs : dict, optional
        Extra keyword arguments forwarded to the internal
        :func:`~phase.carrier.remove_carrier` call (defaults to
        ``defocus=True, refine_iters=10, n_blocks=10``, matching
        ``remove_carrier``'s own defaults).
    device : {"auto", "cpu", "cuda"}, default "auto"
        Where to run -- see :func:`phase.aia.aia`'s ``device`` parameter.

    Returns
    -------
    RippleResult
        See :class:`RippleResult`. Pass to :func:`apply_phase_ripple`.
    """
    phi = to_device(phi, device=device)
    xp = get_array_module(phi)
    mask = to_device(mask, device=device, dtype=bool)
    w = xp.ones(phi.shape) if weight is None else xp.clip(to_device(weight, device=device), 0, None)

    ckw = dict(defocus=True, refine_iters=10, n_blocks=10)
    if carrier_kwargs:
        ckw.update(carrier_kwargs)
    flat = remove_carrier(phi, weight=w, mask=mask, device=device, **ckw).phi

    phi_w = wrap(phi)
    idx = mask & (w > 0)
    x = phi_w[idx]
    y = flat[idx]
    wx = w[idx]
    rms_before = float(xp.sqrt(xp.average(y**2, weights=wx)))

    # circular-mean-bin the residual against phase before fitting -- robust
    # to per-pixel noise and keeps the regression small regardless of ROI size.
    edges = xp.linspace(-xp.pi, xp.pi, nbins + 1)
    bin_idx = xp.clip(xp.digitize(x, edges) - 1, 0, nbins - 1)
    bin_weight = xp.bincount(bin_idx, weights=wx, minlength=nbins)
    bin_c = xp.bincount(bin_idx, weights=wx * xp.cos(y), minlength=nbins)
    bin_s = xp.bincount(bin_idx, weights=wx * xp.sin(y), minlength=nbins)
    valid = bin_weight > 0
    lut = xp.full(nbins, xp.nan)
    lut[valid] = xp.arctan2(bin_s[valid], bin_c[valid])
    bin_centers = -xp.pi + (xp.arange(nbins) + 0.5) * 2 * xp.pi / nbins

    cols = [xp.ones(int(xp.sum(valid)))]
    for k in orders:
        cols += [xp.cos(k * bin_centers[valid]), xp.sin(k * bin_centers[valid])]
    D = xp.column_stack(cols)
    sw = xp.sqrt(bin_weight[valid])
    sol, *_ = xp.linalg.lstsq(D * sw[:, None], lut[valid] * sw, rcond=None)

    coeffs = {0: float(sol[0])}
    for i, k in enumerate(orders):
        coeffs[k] = (float(sol[1 + 2 * i]), float(sol[2 + 2 * i]))

    eps = _eval_ripple(x, coeffs, orders)
    rms_after = float(xp.sqrt(xp.average(wrap(y - eps)**2, weights=wx)))

    return RippleResult(coeffs=coeffs, orders=tuple(orders), bin_centers=bin_centers,
                         lut=lut, bin_weight=bin_weight, rms_before=rms_before,
                         rms_after=rms_after)


def _eval_ripple(phi: np.ndarray, coeffs: dict, orders: tuple) -> np.ndarray:
    """Evaluate the fitted ``eps(phi) = c0 + sum_k a_k*cos(k*phi) + b_k*sin(k*phi)``."""
    xp = get_array_module(phi)
    eps = xp.full(phi.shape, coeffs[0], dtype=xp.float64)
    for k in orders:
        a_k, b_k = coeffs[k]
        eps = eps + a_k * xp.cos(k * phi) + b_k * xp.sin(k * phi)
    return eps


def apply_phase_ripple(phi: np.ndarray, ripple: RippleResult) -> np.ndarray:
    """Remove a fitted phase-locked ripple from a wrapped phase map.

    Evaluates ``eps(phi)`` from :class:`RippleResult` at every pixel's own
    (exact, unbinned) phase value and subtracts it, wrap-safe. This is a
    pointwise correction -- each output pixel depends only on that pixel's
    own input value, so it changes no spatial frequency content and cannot
    blur or otherwise affect resolution; it is applied at every pixel,
    including over structure, not just where it was estimated from.

    Parameters
    ----------
    phi : np.ndarray, shape (H, W)
        Wrapped phase map to correct (e.g. the same map passed to
        :func:`estimate_phase_ripple`, before carrier removal).
    ripple : RippleResult
        Output of :func:`estimate_phase_ripple`.

    Returns
    -------
    np.ndarray, shape (H, W)
        Corrected phase map, wrapped to ``(-pi, pi]``.
    """
    phi_w = wrap(phi)
    eps = _eval_ripple(phi_w, ripple.coeffs, ripple.orders)
    return wrap(phi_w - eps)
