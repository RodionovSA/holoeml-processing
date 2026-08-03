"""Averaging repeated, independent phase acquisitions of the same object."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .carrier import remove_carrier
from .reference import subtract_reference


@dataclass
class CombinedResult:
    """Output of :func:`combine_acquisitions`.

    Attributes
    ----------
    phi : np.ndarray, shape (H, W)
        Combined phase map, in ``(-pi, pi]``.
    scatter : np.ndarray, shape (H, W)
        Per-pixel circular standard deviation across the input acquisitions,
        in radians (``sqrt(-2*log(R))`` where ``R`` is the mean resultant
        length) -- the empirical, per-pixel uncertainty of ``phi``. Rises
        wherever the inputs disagree, e.g. low-modulation regions or spots
        where one acquisition had a local defect.
    mean_resultant : np.ndarray, shape (H, W)
        Per-pixel mean resultant length ``R`` in ``[0, 1]`` (1 = perfect
        agreement across acquisitions). ``scatter`` is a monotonic function
        of this; kept separately since ``R`` is also directly useful as a
        reliability weight (e.g. as ``weight=`` to
        :func:`~phase.carrier.remove_carrier`).
    n : int
        Number of acquisitions combined.
    sign_flips : list of int
        Indices (into the input ``phis``) whose sign branch was flipped
        (``-phi``) to match ``phis[reference]`` before averaging.
    """

    phi: np.ndarray
    scatter: np.ndarray
    mean_resultant: np.ndarray
    n: int
    sign_flips: list


def combine_acquisitions(phis, weights=None, align_carrier: bool = True,
                          reference: int = 0, carrier_kwargs: Optional[dict] = None
                          ) -> CombinedResult:
    """Average independent, repeated phase measurements of the same object.

    A single ``aia()`` run's accuracy is limited by real acquisition-to-
    acquisition randomness (vibration, air currents, source noise) that does
    not average out within one run, only across independent runs -- unlike
    per-frame model errors (contrast, phase-step), which are systematic
    within a run and need a better model rather than averaging (see
    :func:`~phase.aia.aia`'s ``gain`` parameter and its Notes on per-frame
    tilt). Verified on real repeated bare-glass acquisitions: this random
    component was ~1.15 degrees per acquisition and averaging brought a
    null-test difference down following the expected ``1/sqrt(k)`` scaling
    (1.81 degrees for 1 acquisition vs. 1.30 degrees for an average of 2).

    Three things must be resolved before a plain average of wrapped phase
    maps means anything, all handled here using existing functions in this
    package:

    1. **Sign branch** -- each ``aia()`` run independently lands on ``+phi``
       or ``-phi`` (see :func:`~phase.reference.subtract_reference`). Every
       map is resolved against ``phis[reference]`` the same way
       ``subtract_reference`` does, and flipped if that gives lower spread;
       see ``sign_flips``. This is done on the *raw* input maps, before
       carrier removal (step 2) -- the shared carrier (from the raw
       object/reference beam tilt, often many fringes) is a strong,
       unambiguous discriminant for the sign, whereas what two acquisitions
       of the same object still share *after* their own carriers are
       independently removed is only the real surface figure, which can be
       comparable in size to acquisition-to-acquisition noise and too
       marginal a signal to reliably resolve the branch from (this was
       verified to produce spurious flips when tried in that order).
    2. **Inter-acquisition drift** -- tilt/piston (and, if ``align_carrier``
       includes ``defocus``, curvature) generally differ slightly between
       acquisitions of the same nominal setup. Each sign-resolved map is
       then passed through :func:`~phase.carrier.remove_carrier` so the
       average isn't blurred by chasing a moving carrier.
    3. **Circular averaging** -- phase is combined as the complex mean
       ``mean(weight * exp(i*phi))``, never an arithmetic mean of the
       wrapped angle.

    Parameters
    ----------
    phis : sequence of np.ndarray, each shape (H, W)
        Independently recovered phase maps of the *same* object (e.g. one
        per repeated scan). Pass ``AIAResult.phi`` from separate ``aia()``
        calls -- run ``aia`` on one stack at a time and keep only ``phi``/
        ``b``, rather than holding every raw stack in memory at once.
    weights : sequence of np.ndarray, each shape (H, W), optional
        Per-acquisition, per-pixel reliability (e.g. each acquisition's
        ``AIAResult.b``). Used both for the carrier-removal step and the
        final weighted circular mean. Defaults to uniform weight.
    align_carrier : bool, default True
        Run :func:`~phase.carrier.remove_carrier` (with ``defocus=True``)
        on each map before combining. Turn off only if the maps are
        already known to share one carrier (e.g. already differenced
        against a reference).
    reference : int, default 0
        Index into ``phis`` that fixes the sign convention; every other map
        is oriented to agree with this one.
    carrier_kwargs : dict, optional
        Extra keyword arguments forwarded to the internal
        :func:`~phase.carrier.remove_carrier` calls (defaults to
        ``defocus=True, refine_iters=10, n_blocks=10``).

    Returns
    -------
    CombinedResult
        See :class:`CombinedResult`.
    """
    n = len(phis)
    if n < 2:
        raise ValueError(f"combine_acquisitions: need at least 2 acquisitions, got {n}")

    H, W = phis[0].shape
    ws = [np.ones((H, W)) if weights is None else np.clip(np.asarray(weights[i], float), 0, None)
          for i in range(n)]

    # Sign resolution must happen on the *raw* maps, before carrier removal:
    # the shared carrier (typically many cycles, since it comes from the raw
    # tilt between object and reference beam) is a strong, unambiguous
    # discriminant for the +/-phi branch. Once each map's own carrier has
    # been independently removed, what two same-object acquisitions still
    # share is only the much smaller real surface figure -- comparable in
    # size to acquisition-to-acquisition noise, and too marginal a signal
    # for reliable branch resolution (verified: doing it post-carrier-removal
    # produced spurious flips even on genuinely unflipped inputs).
    sign_flips = []
    resolved = list(phis)
    for i in range(n):
        if i == reference:
            continue
        dr = subtract_reference(phis[reference], phis[i], weight=ws[reference])
        if dr.sign == -1:
            resolved[i] = np.angle(np.exp(-1j * phis[i]))
            sign_flips.append(i)

    aligned = resolved
    if align_carrier:
        ckw = dict(defocus=True, refine_iters=10, n_blocks=10)
        if carrier_kwargs:
            ckw.update(carrier_kwargs)
        aligned = [remove_carrier(p, weight=ws[i], **ckw).phi for i, p in enumerate(aligned)]

    stack = np.stack([w * np.exp(1j * p) for w, p in zip(ws, aligned)], axis=0)
    total_w = np.sum(ws, axis=0)
    total_w = np.where(total_w > 0, total_w, np.finfo(float).eps)
    mean_field = stack.sum(0) / total_w

    phi = np.angle(mean_field)
    R = np.clip(np.abs(mean_field), 0, 1)
    scatter = np.sqrt(np.clip(-2 * np.log(np.maximum(R, np.finfo(float).eps)), 0, None))

    return CombinedResult(phi=phi, scatter=scatter, mean_resultant=R,
                           n=n, sign_flips=sign_flips)
