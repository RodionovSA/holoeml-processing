"""Per-frame carrier/DC-amplitude estimation, shared by every phase-recovery method.

Holds :func:`_carrier_dc_amplitudes` and the two measurements built on it,
:func:`measure_frame_contrast` and :func:`measure_frame_visibility`. Used by
:meth:`phase.solver.PhaseSolver._estimate_gain` to resolve each frame's
fringe gain ``g_n`` (Eq. (8) of ``docs/interference_model.md``) before
dispatching to a phase-recovery method.
"""

import warnings

import numpy as np

from .backend import default_dtype, get_array_module

def _carrier_dc_amplitudes(stack: np.ndarray, dc_radius: int = 8,
                            halfwin: tuple = (3, 4), frame_chunk: int = 8,
                            dtype=None) -> tuple:
    """Per-frame carrier and DC amplitude, for gain/visibility estimation.

    The fringe pattern shows up as a carrier peak in each frame's 2D spectrum;
    integrating ``|F|^2`` over a small neighborhood around that peak
    (located once, from the frame-summed spectrum) gives a per-frame amplitude
    robust to small tilt/defocus jitter. The DC bin (``F[:, 0, 0]``),
    read off in the same pass, is the windowed-field analog of the background term ``a``.

    Parameters
    ----------
    stack : np.ndarray, shape (N, H, W)
        Phase-shifted interferogram frames.
    dc_radius : int, default 8
        Half-size, in FFT bins, of the neighborhood around DC excluded
        before locating the carrier peak.
    halfwin : (int, int), default (3, 4)
        Half-size, in FFT bins along (row, column), of the neighborhood
        integrated around the located carrier peak.
    frame_chunk : int, default 8
        Number of frames' FFTs held resident at once (bounds peak memory
        regardless of ``N``; irrelevant to the result).
    dtype : numpy/cupy dtype, optional
        Working (real) dtype for the per-chunk FFT input. Defaults to
        ``float32`` (see :func:`phase.backend.default_dtype`).

    Returns
    -------
    carrier_amp, dc_amp : np.ndarray, each shape (N,)
        Per-frame carrier-peak amplitude and DC (zero-frequency) amplitude,
        both un-normalized.

    Warns
    -----
    UserWarning
        If the located peak sits on the ``dc_radius`` exclusion boundary --
        a sign the true carrier frequency is too close to DC for this
        ``dc_radius``, so the "peak" found is likely background leakage
        rather than the genuine carrier, and the returned amplitudes are
        unreliable. Use a smaller ``dc_radius`` or verify the carrier
        location.
    """
    xp = get_array_module(stack)
    work_dtype = dtype if dtype is not None else default_dtype(xp)
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

    if ix <= dc_radius and (iy <= dc_radius or iy >= H - dc_radius - 1):
        warnings.warn(
            f"_carrier_dc_amplitudes: located carrier peak at (row={iy}, col={ix}) "
            f"sits on the dc_radius={dc_radius} exclusion boundary -- likely DC/background "
            f"leakage rather than a genuine carrier peak, not the true fringe frequency. "
            f"Consider a smaller dc_radius, or verify the carrier location.",
            stacklevel=2,
        )

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

    Phase-recovery methods generally assume every frame shares one
    fringe-modulation map ``b(x, y)`` (Eq. (8) of
    ``docs/interference_model.md``); in practice illumination drift,
    source-coherence roll-off over a long scan, or per-shot exposure
    variation make the *true* per-frame contrast ``g_n`` deviate from 1 --
    sometimes by tens of percent. Forcing a shared ``b`` then makes the
    least-squares fit trade contrast error off against phase, producing an
    error that is a deterministic function of the local phase. This function
    measures ``g_n`` directly from the data, independent of any particular
    solve, so it can be supplied as a fixed input rather than estimated
    jointly with phase.

    See :func:`_carrier_dc_amplitudes` for the carrier-peak method and all
    parameters (identical here).

    Returns
    -------
    np.ndarray, shape (N,)
        Per-frame contrast, normalized so ``median(g) = 1`` -- this is what
        :meth:`phase.solver.PhaseSolver._estimate_gain` passes as ``g`` to
        the selected phase-recovery method (see
        :class:`phase.solver.PhaseConfig`'s ``use_g``). Relative *within
        this stack only* -- use :func:`measure_frame_visibility` instead to
        compare contrast across separately-captured frames or stacks (e.g.
        a piezo coherence scan).
    """
    xp = get_array_module(stack)
    amp, _ = _carrier_dc_amplitudes(stack, dc_radius, halfwin, frame_chunk, dtype)
    return amp / xp.median(amp)

def measure_frame_visibility(stack: np.ndarray, dc_radius: int = 8,
                              halfwin: tuple = (3, 4), frame_chunk: int = 8,
                              dtype=None) -> np.ndarray:
    """Measure each frame's fringe visibility, absolutely (not stack-relative).

    Eq. (8) of ``docs/interference_model.md`` models each frame as
    ``I_n = alpha_n * [a + g_n*b*cos(phi + delta_n)]``, so the *true*
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