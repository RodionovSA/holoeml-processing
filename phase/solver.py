"""Solver class for phase-shifting phase extraction methods.

:class:`PhaseResult` holds the fields that a solver produces, matching
the per-frame model of ``docs/interference_model.md`` Eq. (8)::

    I_n(x, y) = alpha_n * [a(x, y) + g_n * b(x, y) * cos(phi(x, y) + delta_n)]

"""

from dataclasses import dataclass, field
from typing import Optional, Dict

import numpy as np
import yaml

from .backend import get_array_module, to_device
from .methods import METHOD_REGISTRY, MethodParam
from .methods.base import _fmt_value

METHODS = list(METHOD_REGISTRY)
GAIN_MODES = ("none", "joint")


@dataclass
class PhaseResult:
    """Fields from :class:`PhaseSolver`'s output.

    Attributes
    ----------
    phi : np.ndarray, shape (H, W)
        Wrapped phase map, in ``(-pi, pi]``.
    a : np.ndarray, shape (H, W)
        Background intensity map.
    b : np.ndarray, shape (H, W)
        Fringe amplitude (contrast) map.
    delta : np.ndarray, shape (N,)
        Per-frame phase steps, in radians, referenced to ``delta[0] = 0``.
    g : np.ndarray, shape (N,)
        Per-frame fringe gain, normalized so ``median(g) = 1``.
    alpha : np.ndarray, shape (N,)
        Per-frame common source-power factor, scaling ``a`` and ``b``
        together.
    method_param : MethodParam
        Diagnostics specific to whichever algorithm produced this result
        (e.g. an :class:`phase.methods.aia.AIAParam` for ``method="aia"``)
        -- see :data:`METHODS` and :class:`MethodParam`.
    reconstruction_error : float
        RMSE, in the input stack's original units, between the input stack
        and Eq. (8) evaluated at ``phi, a, b, delta, g, alpha`` -- a
        method-agnostic fit-quality check computed the same way regardless
        of ``method`` (see :meth:`PhaseSolver.fit`).

    Notes
    -----
    Fields are numpy or cupy arrays depending on the solver's ``device``
    argument -- they are not forced back to the host, so that passing them
    on to another GPU-aware step keeps large arrays resident on the GPU.
    Call :func:`phase.backend.asnumpy` on a field yourself when you need a
    guaranteed-numpy array.
    """

    phi: np.ndarray
    a: np.ndarray
    b: np.ndarray
    delta: np.ndarray
    g: np.ndarray
    alpha: np.ndarray
    method_param: MethodParam
    reconstruction_error: float

    def to_device(self, device: str = "auto", dtype=None) -> "PhaseResult":
        """Return a copy with every array field moved to the given device.

        Parameters
        ----------
        device : {"auto", "cpu", "cuda"}, default "auto"
            See :func:`phase.backend.to_device`.
        dtype : dtype, optional
            Cast while moving; see :func:`phase.backend.to_device`.

        Returns
        -------
        PhaseResult
            A new instance with ``phi, a, b, delta, g, alpha`` moved to
            ``device`` (and cast to ``dtype`` if given). ``method_param``
            and ``reconstruction_error`` are plain Python scalars for every
            currently registered method, so they're carried over unchanged.
        """
        return PhaseResult(
            phi=to_device(self.phi, device=device, dtype=dtype),
            a=to_device(self.a, device=device, dtype=dtype),
            b=to_device(self.b, device=device, dtype=dtype),
            delta=to_device(self.delta, device=device, dtype=dtype),
            g=to_device(self.g, device=device, dtype=dtype),
            alpha=to_device(self.alpha, device=device, dtype=dtype),
            method_param=self.method_param,
            reconstruction_error=self.reconstruction_error,
        )

    def print_summary(self) -> None:
        """Print method_param's summary, then reconstruction_error.

        Delegates the method-specific diagnostics to
        :meth:`MethodParam.print_summary` (overridden per method), then
        prints ``reconstruction_error`` -- always a :class:`PhaseResult`
        field, so identical across every method.
        """
        self.method_param.print_summary()
        print(f"reconstruction_error: {_fmt_value(self.reconstruction_error)}")

@dataclass
class PhaseConfig:
    """Configuration for :class:`PhaseSolver`, validated once at construction.

    Attributes
    ----------
    use_alpha : bool, default True
        If True, estimate and divide out the per-frame factor ``alpha_n``
        (:meth:`PhaseSolver._normalize`) before solving. If False, skip
        normalization and fix ``alpha_n = 1`` for every frame.
    gain_mode : {"none", "joint"}, default "joint"
        How to resolve the per-frame fringe gain ``g_n`` when ``g`` is not
        given. ``"joint"`` fits it jointly with the phase step inside the
        chosen method's own iteration (``fit_gain=True``, e.g.
        :func:`phase.methods.aia.aia`) -- this makes no assumption about the
        fringe pattern's spatial frequency, so it works on circular or
        otherwise carrier-free fringes where the older FFT-based estimate
        (:func:`phase.utils.measure_frame_contrast`) does not. ``"none"``
        fixes ``g_n = 1`` for every frame. Ignored when ``g`` is given.
    g : np.ndarray, shape (N,), optional
        Precomputed per-frame fringe gain, e.g. from a calibration shot, or
        from calling :func:`phase.utils.measure_frame_contrast` yourself and
        reusing the result across several fits. When given, this is used
        directly as the fixed gain and ``gain_mode`` is ignored.
    method : str, default "aia"
        Which registered algorithm to dispatch to -- must be one of
        :data:`METHODS` (case-insensitive), checked here at construction
        time rather than at every :meth:`PhaseSolver.fit` call.
    method_kwargs : dict, default {}
        Extra keyword arguments passed through to the selected method
        (e.g. ``{"iters": 50, "tol": 1e-5}`` for ``method="aia"``) -- see
        the chosen method's function for what it accepts.

    See :meth:`to_yaml`/:meth:`from_yaml` to save/load a configuration as
    a YAML file.
    """

    use_alpha: bool = True
    gain_mode: str = "joint"
    g: Optional[np.ndarray] = None
    method: str = "aia"
    method_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.method.lower() not in METHODS:
            raise ValueError(f"unknown method {self.method!r}, expected one of {METHODS}")
        if self.gain_mode not in GAIN_MODES:
            raise ValueError(
                f"unknown gain_mode {self.gain_mode!r}, expected one of {GAIN_MODES}"
            )

    def to_dict(self) -> Dict:
        """Return this configuration as a plain, YAML-safe dict.

        ``g`` is converted to a plain list (or ``None``), since YAML has no
        native array type; reconstructed by :meth:`from_yaml`.
        ``method_kwargs``' values are included as-is and must themselves be
        YAML-safe (numbers, strings, lists, nested dicts of the same) for
        :meth:`to_yaml` to succeed on the result -- e.g. a numpy array in
        there (such as a custom ``delta0`` for ``"aia"``) is not supported
        and will raise from PyYAML when dumped.

        Returns
        -------
        dict
            Keys match :class:`PhaseConfig`'s fields, in declaration order.
        """
        data = {
            "use_alpha": self.use_alpha,
            "gain_mode": self.gain_mode,
            "g": self.g.tolist() if self.g is not None else None,
            "method": self.method,
            "method_kwargs": self.method_kwargs,
        }

        return data
    
    def to_yaml(self, path) -> None:
        """Save this configuration as a YAML file.

        Converts to a plain dict via :meth:`to_dict` and writes it with
        ``yaml.safe_dump`` (not ``yaml.dump``, which can serialize
        arbitrary Python objects) -- this is a config file, plausibly
        hand-edited or shared, so it stays restricted to plain data. Read
        back with :meth:`from_yaml`.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file path.
        """
        data = self.to_dict()
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path) -> "PhaseConfig":
        """Load a configuration previously written by :meth:`to_yaml`.

        Parameters
        ----------
        path : str or os.PathLike
            Source file path.

        Returns
        -------
        PhaseConfig
            Reconstructed from the file; a key missing from the file falls
            back to that field's normal default, an unrecognized key
            raises ``TypeError``, and an invalid ``method`` or ``gain_mode``
            raises ``ValueError`` (same validation as constructing one
            directly).

        Raises
        ------
        ValueError
            If the file contains ``use_g``, ``dc_radius``, ``halfwin``, or
            ``frame_chunk`` -- the FFT-based gain estimator these configured
            (:func:`phase.utils.measure_frame_contrast`) has been replaced
            by the ``gain_mode`` field (``"joint"`` fits gain inside the
            method's own iteration instead), so a config written for the
            old estimator would otherwise silently change meaning rather
            than fail loudly.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        removed = {"use_g", "dc_radius", "halfwin", "frame_chunk"} & data.keys()
        if removed:
            raise ValueError(
                f"{path!r} uses removed PhaseConfig field(s) {sorted(removed)} "
                f"from the old FFT-based gain estimator -- replace them with "
                f"gain_mode='joint' (fit gain jointly, the new default) or "
                f"gain_mode='none' (fix g=1)."
            )
        if data.get("g") is not None:
            data["g"] = np.asarray(data["g"], dtype=float)
        return cls(**data)


class PhaseSolver:
    """Front end for phase-shifting phase extraction, dispatching by ``method``.

    Wraps the pipeline described in ``docs/interference_model.md`` --
    :meth:`fit` normalizes the input stack (:meth:`_normalize`, gated by
    ``config.use_alpha``), then hands off to the algorithm named by
    ``config.method`` (see :data:`METHODS` for the recognized names, and
    :data:`phase.methods.METHOD_REGISTRY` for their implementations) to
    recover ``phi, a, b, delta`` -- and, when ``config.gain_mode ==
    "joint"`` (the default) and no explicit ``config.g`` is given, ``g``
    jointly with them, inside that method's own iteration. Results are read
    back
    from the fitted ``PhaseSolver`` via the ``phi_``, ``a_``, ``b_``,
    ``delta_``, ``g_``, ``alpha_``, ``method_param_``, ``reconstruction_error_``
    properties -- see :class:`PhaseResult` for their definitions.
    """

    def __init__(self, config: PhaseConfig, device: str = "auto", dtype=None):
        """
        Parameters
        ----------
        config : PhaseConfig
            Which method to run and how (see :class:`PhaseConfig`);
            validated at construction, immutable afterward.
        device : {"auto", "cpu", "cuda"}, default "auto"
            See :func:`phase.backend.to_device`.
        dtype : dtype, optional
            Working dtype for the staged stack. Defaults to whatever
            :func:`phase.backend.to_device` chooses when left unset.
        """
        self.config = config
        self.device = device
        self.dtype = dtype
        self.result_: Optional[PhaseResult] = None

    def fit(self, stack: np.ndarray) -> "PhaseSolver":
        """Recover phase from an interferogram stack.

        Parameters
        ----------
        stack : np.ndarray, shape (N, H, W)
            Stack of ``N`` phase-shifted interferograms.

        Returns
        -------
        self
            For chaining, e.g. ``solver.fit(stack).phi_``.

        Raises
        ------
        ValueError
            If ``stack`` is not 3-D. (An unrecognized ``config.method`` is
            instead caught earlier, at :class:`PhaseConfig` construction.)
        """
        if stack.ndim != 3:
            raise ValueError(f"stack must be 3-D (N, H, W), got shape {stack.shape}")
        stack = to_device(stack, device=self.device, dtype=self.dtype)
        xp = get_array_module(stack)
        if self.config.use_alpha:
            normalized_stack, alpha = self._normalize(stack)
        else:
            normalized_stack = stack
            alpha = xp.ones(stack.shape[0], dtype=xp.float64)

        if self.config.g is not None:
            g = xp.asarray(self.config.g, dtype=xp.float64)
            fit_gain = False
        else:
            g = xp.ones_like(alpha)
            fit_gain = self.config.gain_mode == "joint"
        a, b, phi, delta, g, method_param = self._solve(normalized_stack, g, fit_gain)

        # Eq. (8) evaluated at the fitted parameters, vs. the raw input --
        # a method-agnostic fit-quality check (works the same for any
        # method, since it only depends on the shared a/b/phi/delta/g/alpha
        # contract, not on how they were produced). The phase-step term
        # goes through method_param.phase_step_field rather than a plain
        # delta[:, newaxis, newaxis] broadcast, so a method whose recovered
        # step varies spatially (e.g. a per-frame tilt) is reconstructed
        # correctly too -- see MethodParam.phase_step_field.
        H, W = phi.shape
        delta_field = method_param.phase_step_field(delta, H, W, xp)          # (N, H, W)
        carrier = g[:, xp.newaxis, xp.newaxis] * b[xp.newaxis, :, :] \
            * xp.cos(phi[xp.newaxis, :, :] + delta_field)
        rec_stack = alpha[:, xp.newaxis, xp.newaxis] * (a[xp.newaxis, :, :] + carrier)
        rmse = float(xp.sqrt(xp.mean((stack - rec_stack) ** 2)))

        self.result_ = PhaseResult(phi, a, b, delta, g, alpha, method_param, rmse)
        return self
    
    def _solve(self, stack: np.ndarray, g: np.ndarray, fit_gain: bool):
        """Dispatch to the configured method and recover its Eq. (8) fields.

        Looks up ``config.method`` in :data:`phase.methods.METHOD_REGISTRY`
        (already validated to exist by :class:`PhaseConfig`) and calls it
        with the normalized ``stack``, initial ``g``, ``fit_gain``
        (``config.gain_mode == "joint"`` and no explicit ``config.g``, see
        :meth:`fit`), this solver's ``dtype``, and ``config.method_kwargs``.

        Returns
        -------
        a, b, phi, delta, g, method_param
            See the chosen method's function for details (e.g.
            :func:`phase.methods.aia.aia` for ``method="aia"``); ``g`` is
            the input ``g`` unchanged when ``fit_gain=False``, or the
            jointly fitted gain otherwise.
        """
        solve_fn = METHOD_REGISTRY[self.config.method.lower()]
        return solve_fn(stack, g, fit_gain=fit_gain, dtype=self.dtype, **self.config.method_kwargs)

    def _check_fitted(self):
        if self.result_ is None:
            raise RuntimeError("call fit(stack) before reading results")
        
    def _normalize(self, stack: np.ndarray):
        """Normalize each frame's intensity and extract ``alpha``.

        Divides out only the frame-to-frame fluctuation in overall
        intensity -- the common source-power factor ``alpha_n`` of Eq. (8)
        -- so the returned stack keeps the input's absolute scale rather
        than being rescaled to unit mean; ``a`` and ``b`` fit from it stay
        in the same (e.g. camera) units as the input. This estimates
        ``alpha_n`` from the per-frame mean, which equals ``alpha_n * a``
        only insofar as the cosine term averages out over the field --
        accurate when the field carries many fringes, biased when it
        carries less than roughly one.

        Parameters
        ----------
        stack : np.ndarray, shape (N, H, W)
            Interferogram stack, already moved to the target device/dtype
            by :meth:`fit`.

        Returns
        -------
        normalized_stack : np.ndarray, shape (N, H, W)
            ``stack`` with each frame divided by its ``alpha``.
        alpha : np.ndarray, shape (N,)
            Per-frame common source-power factor, normalized so
            ``median(alpha) = 1``.
        """
        xp = get_array_module(stack)
        m = xp.mean(stack, axis=(1, 2))
        if not float(xp.min(m)) > 0:
            raise ValueError("every frame must have a positive mean intensity")
        alpha = m / xp.median(m)
        return stack / alpha[:, None, None], alpha

    @property
    def phi_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.phi

    @property
    def a_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.a

    @property
    def b_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.b

    @property
    def delta_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.delta

    @property
    def g_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.g

    @property
    def alpha_(self) -> np.ndarray:
        self._check_fitted()
        return self.result_.alpha

    @property
    def method_param_(self) -> MethodParam:
        self._check_fitted()
        return self.result_.method_param

    @property
    def reconstruction_error_(self) -> float:
        self._check_fitted()
        return self.result_.reconstruction_error
