"""Resolving the +/-phi sign branch between a sample and reference phase map."""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np


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
    convention -- and :func:`~phase.aia.aia` cannot guarantee that on its
    own.

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
    picking the right branch -- :func:`~phase.carrier.remove_carrier` can
    clean that up as a separate step, run on the *difference* map, not on
    ``phi``/``phi_ref`` individually beforehand.
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
