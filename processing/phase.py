"""Phase pipeline: sample and reference interferogram stacks -> sample-minus-reference phase.

Wraps :mod:`phase_shift`. For each sample/reference pair:

1. both stacks are solved with :class:`phase_shift.PhaseSolver`;
2. the reference phase is subtracted with :func:`phase_shift.subtract_reference`,
   which resolves the ``phi -> -phi`` branch between the two solves;
3. optionally, a polynomial carrier is removed with :func:`phase_shift.remove_carrier`,
   optionally fitted only on an automatically found uniform region (usually the bare
   substrate), see :func:`fit_carrier`.

For a wavelength scan the final per-wavelength maps (after step 3 when enabled) are
then put on one common branch with :func:`phase_shift.align_signs`. Aligning after
carrier removal keeps neighbouring wavelengths comparable even when the residual
sample-vs-reference tilt drifts between them. What is left is a single global sign
ambiguity for the whole scan, which the caller fixes by negating the output if
needed.

Inputs must already be dark-frame (black) corrected; no intensity preprocessing is
done here.

Usage::

    from processing.phase import PhasePipeline

    pipe = PhasePipeline.from_yaml("configs/phase.yaml")
    phi = pipe.process(sample, reference)              # (Y, X) or (Wvl, Y, X)
    res = pipe.process(sample, reference, output="full")
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import yaml
from phase_shift import (
    CarrierResult,
    DifferenceResult,
    PhaseConfig,
    PhaseResult,
    PhaseSolver,
    SignAlignment,
    align_signs,
    asnumpy,
    remove_carrier,
    subtract_reference,
)
from phase_shift.backend import get_array_module, to_device
from phase_shift.utils import wrap

Output = Literal["phi", "full"]
WEIGHTS = ("b_ref", "none")
OUTPUTS = ("phi", "full")


# --------------------------------------------------------------------------- config

@dataclass
class UniformMaskConfig:
    """Restrict the carrier fit to a uniform region found from the phase itself.

    Structures with a large or non-uniform phase change bias the carrier fit. With
    this enabled, :func:`fit_carrier` alternates: fit the carrier on the current
    mask, remove it, find the dominant phase level (histogram peak, assumed to be the
    substrate), and keep only pixels within ``threshold`` of it. See
    :func:`fit_carrier`.

    Attributes
    ----------
    enabled : bool, default False
        Use the automatic mask at all.
    threshold : float, default 0.3
        Keep pixels whose carrier-removed phase is within this many radians of the
        substrate level. Must exceed the noise and any carrier term above
        ``degree``, or substrate gets rejected too.
    grow : int, default 0
        Also reject pixels within this many pixels of a rejected one, to stay away
        from structure edges.
    iters : int, default 5
        Maximum fit/mask rounds; stops earlier once the mask no longer changes.
    min_fraction : float, default 0.05
        Smallest acceptable kept fraction of the allowed pixels. If a round would
        keep less, it is discarded with a warning and the previous mask is used.
    """

    enabled: bool = False
    threshold: float = 0.3
    grow: int = 0
    iters: int = 5
    min_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")
        if self.grow < 0 or self.iters < 1:
            raise ValueError(f"need grow >= 0 and iters >= 1, got {self.grow}, {self.iters}")


@dataclass
class CarrierConfig:
    """Optional carrier removal after reference subtraction.

    ``degree``, ``window``, ``subsample`` and ``max_iter`` are passed to
    :func:`phase_shift.remove_carrier`.

    Attributes
    ----------
    enabled : bool, default False
        Remove the carrier at all.
    degree : int, default 1
        Highest total polynomial degree of the carrier: ``0`` piston only,
        ``1`` tilt, ``2`` adds defocus/astigmatism, ...
    window : int, default 5
        Neighbour-product window of the starting estimate, positive odd pixels.
    subsample : int, default 4
        Stride of the grid the fit starts on; the phase must change by less than
        ``pi`` over this many pixels. ``1`` fits on the full grid only.
    max_iter : int, default 20
        Maximum Newton steps per grid.
    uniform_mask : UniformMaskConfig
        Optional automatic restriction of the fit to the substrate; off by default.
    """

    enabled: bool = False
    degree: int = 1
    window: int = 5
    subsample: int = 4
    max_iter: int = 20
    uniform_mask: UniformMaskConfig = field(default_factory=UniformMaskConfig)

    def __post_init__(self) -> None:
        if isinstance(self.uniform_mask, dict):
            self.uniform_mask = UniformMaskConfig(**self.uniform_mask)

    def kwargs(self) -> dict:
        """Keyword arguments for :func:`phase_shift.remove_carrier`."""
        return {"degree": self.degree, "window": self.window,
                "subsample": self.subsample, "max_iter": self.max_iter}


@dataclass
class PipelineConfig:
    """Settings for :class:`PhasePipeline`.

    Attributes
    ----------
    solver : PhaseConfig
        phase-shift solver settings (method, gain mode, method kwargs, ...).
    carrier : CarrierConfig
        Optional carrier removal after reference subtraction; off by default.
    device : {"auto", "cpu", "cuda"}, default "auto"
        Device the solves and carrier fits run on. Sign alignment across
        wavelengths always runs on the CPU, so the full ``(Wvl, Y, X)`` stack never
        has to fit on the GPU.
    precision : {"single", "double", "fast"}, default "single"
        See :class:`phase_shift.Precision`.
    weight : {"b_ref", "none"}, default "b_ref"
        Per-pixel weight used for sign-branch choices (reference subtraction and
        wavelength alignment) and for the carrier fit: the reference fringe
        amplitude ``b``, or uniform.
    ambiguous_ratio : float, default 1.5
        Passed to :func:`phase_shift.subtract_reference` and
        :func:`phase_shift.align_signs`; a branch choice whose two spreads are
        within this factor is flagged as ambiguous.
    """

    solver: PhaseConfig = field(default_factory=PhaseConfig)
    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    device: str = "auto"
    precision: str = "single"
    weight: str = "b_ref"
    ambiguous_ratio: float = 1.5

    def __post_init__(self) -> None:
        if self.weight not in WEIGHTS:
            raise ValueError(f"unknown weight {self.weight!r}, expected one of {WEIGHTS}")

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        """Build from a plain dict; ``solver`` holds :class:`PhaseConfig` fields and
        ``carrier`` holds :class:`CarrierConfig` fields."""
        data = dict(data)
        solver = dict(data.pop("solver", None) or {})
        for key in ("g", "noise_std"):
            if solver.get(key) is not None:
                solver[key] = np.asarray(solver[key], dtype=float)
        carrier = CarrierConfig(**(data.pop("carrier", None) or {}))
        return cls(solver=PhaseConfig(**solver), carrier=carrier, **data)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> PipelineConfig:
        """Load a YAML file laid out like ``configs/phase.yaml``."""
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "precision": self.precision,
            "weight": self.weight,
            "ambiguous_ratio": self.ambiguous_ratio,
            "solver": self.solver.to_dict(),
            "carrier": asdict(self.carrier),
        }

    def to_yaml(self, path: str | os.PathLike) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


# --------------------------------------------------------------------------- results

@dataclass
class StackResult:
    """Full output for one sample/reference pair, all arrays on the host.

    Attributes
    ----------
    phi : np.ndarray, shape (Y, X)
        Final phase of this pair, wrapped, in radians: ``carrier.phi`` if carrier
        removal ran, else ``difference.phi`` (the same array, not a copy). Not
        sign-aligned to other wavelengths; see :attr:`WvlStackResult.phi` for that.
    sample, reference : PhaseResult
        The two solves (``phi``, ``a``, ``b``, ``delta``, ``g``, ``alpha``,
        ``phi_error``, diagnostics). ``method_param`` is passed through as the
        method returned it and may still hold GPU arrays.
    difference : DifferenceResult
        Reference subtraction: the raw difference ``phi``, the chosen ``sign`` of
        the reference branch, both spreads, and the ``ambiguous`` flag.
    carrier : CarrierResult or None
        Carrier fit (``eta`` coefficients, ``piston``, ``cond``, ...); ``None``
        when carrier removal is disabled.
    carrier_mask : np.ndarray of bool, shape (Y, X), or None
        Pixels the carrier was fitted on when the uniform mask is enabled (already
        combined with the caller's ``mask``); ``None`` otherwise.
    """

    phi: np.ndarray
    sample: PhaseResult
    reference: PhaseResult
    difference: DifferenceResult
    carrier: CarrierResult | None = None
    carrier_mask: np.ndarray | None = None


@dataclass
class WvlStackResult:
    """Full output for a wavelength scan.

    Attributes
    ----------
    phi : np.ndarray, shape (Wvl, Y, X)
        Final phase with every wavelength on the sign branch of wavelength 0, so
        the whole scan shares one global sign:
        ``phi[k] = alignment.signs[k] * stacks[k].phi``. Same array as
        ``alignment.phi``.
    alignment : SignAlignment
        ``signs`` applied per wavelength, ``ambiguous`` flags, and the
        ``reference`` index each wavelength was compared with.
    stacks : list[StackResult]
        Per-wavelength results before alignment.
    """

    phi: np.ndarray
    alignment: SignAlignment
    stacks: list[StackResult]


# --------------------------------------------------------------------------- pipeline

class PhasePipeline:
    """Solve sample/reference stacks for sample-minus-reference phase.

    Parameters
    ----------
    config : PipelineConfig, optional
        Defaults to ``PipelineConfig()``.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.solver = PhaseSolver(self.config.solver, device=self.config.device,
                                  precision=self.config.precision)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> PhasePipeline:
        return cls(PipelineConfig.from_yaml(path))

    def process(self, sample, reference, output: Output = "phi",
                mask: np.ndarray | None = None, progress: bool = True):
        """Dispatch on dimensionality: ``(N, Y, X)`` -> :meth:`process_stack`,
        ``(Wvl, N, Y, X)`` -> :meth:`process_wvl_stack`."""
        ndim = np.ndim(sample)
        if ndim == 3:
            return self.process_stack(sample, reference, output=output, mask=mask)
        if ndim == 4:
            return self.process_wvl_stack(sample, reference, output=output, mask=mask,
                                          progress=progress)
        raise ValueError(f"sample must be (N, Y, X) or (Wvl, N, Y, X), got shape "
                         f"{np.shape(sample)}")

    def process_stack(self, sample, reference, output: Output = "phi",
                      mask: np.ndarray | None = None) -> np.ndarray | StackResult:
        """Solve one sample/reference pair.

        Parameters
        ----------
        sample, reference : array, shape (N, Y, X)
            Dark-corrected interferogram stacks. Frame counts may differ; ``(Y, X)``
            must match.
        output : {"phi", "full"}, default "phi"
            ``"phi"`` returns the ``(Y, X)`` phase as a NumPy array; ``"full"``
            returns a :class:`StackResult`.
        mask : np.ndarray, shape (Y, X), optional
            Pixels where falsey are ignored when choosing the reference sign branch
            and when fitting the carrier (e.g. keep only bare substrate). Does not
            affect the solves; the carrier is still removed everywhere.
        """
        _check_output(output)
        sample, reference = np.asarray(sample), np.asarray(reference)
        if sample.ndim != 3 or reference.ndim != 3:
            raise ValueError(f"sample and reference must be (N, Y, X), got "
                             f"{sample.shape} and {reference.shape}")
        if sample.shape[1:] != reference.shape[1:]:
            raise ValueError(f"sample and reference frame shapes differ: "
                             f"{sample.shape[1:]} vs {reference.shape[1:]}")

        res = self._solve_pair(sample, reference, mask)
        self._free_gpu()
        return res if output == "full" else res.phi

    def process_wvl_stack(self, sample, reference, output: Output = "phi",
                          mask: np.ndarray | None = None,
                          progress: bool = True) -> np.ndarray | WvlStackResult:
        """Solve a wavelength scan and put all wavelengths on one sign branch.

        Wavelengths are processed one at a time, so ``sample``/``reference`` may be
        memory-mapped (e.g. ``np.load(..., mmap_mode="r")``). They must be ordered
        so neighbouring wavelengths have similar phase maps.

        Parameters
        ----------
        sample, reference : array, shape (Wvl, N, Y, X)
            Dark-corrected interferogram stacks. ``Wvl`` and ``(Y, X)`` must match.
        output : {"phi", "full"}, default "phi"
            ``"phi"`` returns the aligned ``(Wvl, Y, X)`` phase; ``"full"`` returns a
            :class:`WvlStackResult`.
        mask : np.ndarray, shape (Y, X), optional
            Pixels where falsey are ignored in every sign-branch choice and carrier
            fit.
        progress : bool, default True
            Show a tqdm progress bar over wavelengths.

        Warns
        -----
        UserWarning
            If any wavelength's alignment to its neighbour was ambiguous.
        """
        _check_output(output)
        s_shape, r_shape = np.shape(sample), np.shape(reference)
        if len(s_shape) != 4 or len(r_shape) != 4:
            raise ValueError(f"sample and reference must be (Wvl, N, Y, X), got "
                             f"{s_shape} and {r_shape}")
        if s_shape[0] != r_shape[0] or s_shape[2:] != r_shape[2:]:
            raise ValueError(f"sample and reference must share Wvl and (Y, X): "
                             f"{s_shape} vs {r_shape}")

        n_wvl = s_shape[0]
        phis = weights = None                   # (Wvl, Y, X) host arrays
        stacks: list[StackResult] = []
        for k in _progress(range(n_wvl), progress):
            res = self._solve_pair(np.asarray(sample[k]), np.asarray(reference[k]), mask)
            if phis is None:
                phis = np.empty((n_wvl,) + res.phi.shape, dtype=res.phi.dtype)
                if self.config.weight == "b_ref":
                    weights = np.empty((n_wvl,) + res.phi.shape, dtype=res.reference.b.dtype)
            phis[k] = res.phi
            if weights is not None:
                weights[k] = res.reference.b
            if output == "full":
                stacks.append(res)
            del res
            self._free_gpu()

        alignment = align_signs(phis, weight=weights,
                                mask=None if mask is None else asnumpy(mask),
                                ambiguous_ratio=self.config.ambiguous_ratio, device="cpu")
        del phis, weights
        if alignment.ambiguous.any():
            idx = np.flatnonzero(alignment.ambiguous).tolist()
            warnings.warn(f"sign alignment was ambiguous for wavelength indices {idx}; "
                          f"check those maps visually.", stacklevel=2)

        if output == "phi":
            return alignment.phi
        return WvlStackResult(phi=alignment.phi, alignment=alignment, stacks=stacks)

    # ----------------------------------------------------------------------- internals

    def _solve_pair(self, sample: np.ndarray, reference: np.ndarray,
                    mask: np.ndarray | None) -> StackResult:
        """Fit both stacks, subtract the reference, optionally remove the carrier.

        Returns a host-side :class:`StackResult`.
        """
        cfg = self.config
        rs = self.solver.fit(sample).result
        rr = self.solver.fit(reference).result
        weight = rr.b if cfg.weight == "b_ref" else None
        if mask is not None:
            mask = to_device(mask, device=cfg.device)
        diff = subtract_reference(rs.phi, rr.phi, weight=weight, mask=mask,
                                  ambiguous_ratio=cfg.ambiguous_ratio, device=cfg.device)
        carrier = carrier_mask = None
        if cfg.carrier.enabled:
            carrier, carrier_mask = fit_carrier(diff.phi, cfg.carrier, weight=weight,
                                                mask=mask, device=cfg.device,
                                                precision=cfg.precision)
            carrier.phi = asnumpy(carrier.phi)
            if carrier_mask is not None:
                carrier_mask = asnumpy(carrier_mask)
        diff.phi = asnumpy(diff.phi)
        return StackResult(phi=diff.phi if carrier is None else carrier.phi,
                           sample=rs.to_device("cpu"), reference=rr.to_device("cpu"),
                           difference=diff, carrier=carrier, carrier_mask=carrier_mask)

    def _free_gpu(self) -> None:
        self.solver.result = None
        if self.config.device == "cpu":
            return
        try:
            import cupy
        except ImportError:
            return
        cupy.get_default_memory_pool().free_all_blocks()


# --------------------------------------------------------------------------- carrier

def fit_carrier(phi, config: CarrierConfig, weight=None, mask=None, device: str = "auto",
                precision: str | None = None
                ) -> tuple[CarrierResult, np.ndarray | None]:
    """Remove the polynomial carrier, optionally fitting it on a uniform region only.

    Without ``config.uniform_mask.enabled`` this is a single
    :func:`phase_shift.remove_carrier` call on ``mask``. With it, each round

    1. fits and removes the carrier using the current mask (initially ``mask``, or
       every pixel);
    2. takes the substrate level as the peak of the histogram of the result over
       ``mask``, refined by the circular mean of the pixels near the peak;
    3. keeps pixels of ``mask`` within ``threshold`` of that level, then drops those
       within ``grow`` px of a rejected one;

    until the mask stops changing or ``iters`` rounds ran, and finally refits on the
    last mask. The carrier is always removed from the whole map; the mask only
    selects the pixels the fit uses.

    Parameters
    ----------
    phi : array, shape (Y, X)
        Wrapped phase, in radians (e.g. the sample-minus-reference difference).
    config : CarrierConfig
        Fit settings; ``config.enabled`` is not checked here.
    weight : array, shape (Y, X), optional
        Per-pixel fit weight, e.g. the reference fringe amplitude.
    mask : array of bool, shape (Y, X), optional
        Pixels allowed in the fit at all; the automatic mask is a subset of it.
    device, precision
        Passed to :func:`phase_shift.remove_carrier`.

    Returns
    -------
    carrier : CarrierResult
        Fit on the final mask; ``carrier.phi`` is the carrier-removed map.
    fit_mask : array of bool, shape (Y, X), or None
        Final mask (on the same device as ``carrier.phi``), or None when the uniform
        mask is disabled.

    Warns
    -----
    UserWarning
        If a round would keep less than ``min_fraction`` of the allowed pixels.
    """
    kw = dict(device=device, precision=precision, **config.kwargs())
    um = config.uniform_mask
    if not um.enabled:
        return remove_carrier(phi, weight=weight, mask=mask, **kw), None

    phi = to_device(phi, device=device)
    xp = get_array_module(phi)
    allowed = (xp.ones(phi.shape, dtype=bool) if mask is None
               else to_device(mask, device=device).astype(bool))
    n_allowed = int(allowed.sum())

    fit_mask = allowed
    carrier = remove_carrier(phi, weight=weight, mask=fit_mask, **kw)
    for _ in range(um.iters):
        new = allowed & (xp.abs(wrap(carrier.phi - _substrate_level(carrier.phi, allowed,
                                                                    um.threshold)))
                         < um.threshold)
        if um.grow:
            new &= ~_dilate(~new, um.grow)
        if int(new.sum()) < um.min_fraction * n_allowed:
            warnings.warn(f"uniform carrier mask would keep {int(new.sum())} of {n_allowed} "
                          f"pixels (< min_fraction={um.min_fraction}); keeping the previous "
                          f"mask. Increase threshold or reduce grow.", stacklevel=3)
            break
        if bool(xp.all(new == fit_mask)):
            break
        fit_mask = new
        carrier = remove_carrier(phi, weight=weight, mask=fit_mask, **kw)
    return carrier, fit_mask


def _substrate_level(phi, region, threshold: float) -> float:
    """Dominant phase level of ``phi`` over ``region``: histogram peak, refined by the
    circular mean of the pixels within ``threshold`` of it."""
    xp = get_array_module(phi)
    vals = phi[region]
    counts, edges = xp.histogram(vals, bins=360, range=(-np.pi, np.pi))
    i = int(xp.argmax(counts))
    peak = float(edges[i] + edges[i + 1]) / 2
    near = vals[xp.abs(wrap(vals - peak)) < threshold]
    return float(xp.angle(xp.mean(xp.exp(1j * near)))) if near.size else peak


def _dilate(m, r: int):
    """True where any pixel of the ``(2r+1)^2`` window around it is True."""
    xp = get_array_module(m)
    h, w = m.shape
    c = xp.zeros((h + 2 * r + 1, w + 2 * r + 1), dtype=xp.int32)
    c[r + 1:r + 1 + h, r + 1:r + 1 + w] = m
    c = c.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]
    return s > 0


def _check_output(output: str) -> None:
    if output not in OUTPUTS:
        raise ValueError(f"unknown output {output!r}, expected one of {OUTPUTS}")


def _progress(it, enabled: bool):
    if not enabled:
        return it
    from tqdm.auto import tqdm
    return tqdm(it, desc="wavelengths")
