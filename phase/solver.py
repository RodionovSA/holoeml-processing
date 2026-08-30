"""Solver class for phase-shifting phase extraction methods.

:class:`PhaseResult` holds the fields that a solver produces, matching
the per-frame model of ``docs/interference_model.md`` Eq. (8)::

    I_n(x, y) = alpha_n * [a(x, y) + g_n * b(x, y) * cos(phi(x, y) + delta_n)]

"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .backend import get_array_module, to_device
from .utils import measure_frame_contrast
from .methods import METHOD_REGISTRY, MethodParam

METHODS = list(METHOD_REGISTRY)


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

@dataclass(frozen=True)
class PhaseConfig:
    """Configuration for :class:`PhaseSolver`, validated once at construction.

    Attributes
    ----------
    use_alpha : bool, default True
        If True, estimate and divide out the per-frame factor ``alpha_n``
        (:meth:`PhaseSolver._normalize`) before solving. If False, skip
        normalization and fix ``alpha_n = 1`` for every frame.
    use_g : bool, default True
        If True and ``g`` is not given, estimate the per-frame fringe gain
        ``g_n`` from the spatial carrier (:meth:`PhaseSolver._estimate_gain`)
        before solving. If False, fix ``g_n = 1`` for every frame. Ignored
        when ``g`` is given.
    g : np.ndarray, shape (N,), optional
        Precomputed per-frame fringe gain, e.g. from a calibration shot, or
        from calling :func:`phase.utils.measure_frame_contrast` yourself and
        reusing the result across several fits. When given, this is used
        directly and neither estimated nor defaulted to ones, regardless of
        ``use_g``.
    dc_radius, halfwin, frame_chunk
        Passed through to :func:`phase.utils.measure_frame_contrast` when
        gain is being estimated (``use_g`` is True and ``g`` is not given);
        see that function for their meaning. Unused otherwise.
    method : str, default "aia"
        Which registered algorithm to dispatch to -- must be one of
        :data:`METHODS` (case-insensitive), checked here at construction
        time rather than at every :meth:`PhaseSolver.fit` call.
    method_kwargs : dict, default {}
        Extra keyword arguments passed through to the selected method
        (e.g. ``{"iters": 50, "tol": 1e-5}`` for ``method="aia"``) -- see
        the chosen method's function for what it accepts.
    """

    use_alpha: bool = True
    use_g: bool = True
    g: Optional[np.ndarray] = None
    dc_radius: int = 8
    halfwin: Tuple[int, int] = (3, 4)
    frame_chunk: int = 8
    method: str = "aia"
    method_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.method.lower() not in METHODS:
            raise ValueError(f"unknown method {self.method!r}, expected one of {METHODS}")


class PhaseSolver:
    """Front end for phase-shifting phase extraction, dispatching by ``method``.

    Wraps the pipeline described in ``docs/interference_model.md`` --
    :meth:`fit` normalizes the input stack (:meth:`_normalize`, gated by
    ``config.use_alpha``), estimates each frame's fringe gain
    (:meth:`_estimate_gain`, gated by ``config.use_g``), then hands off to
    the algorithm named by ``config.method`` (see :data:`METHODS` for the
    recognized names, and :data:`phase.methods.METHOD_REGISTRY` for their
    implementations) to recover ``phi, a, b, delta``. Results are read back
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
        elif self.config.use_g:
            g = self._estimate_gain(normalized_stack,
                                    self.config.dc_radius,
                                    self.config.halfwin,
                                    self.config.frame_chunk)
        else:
            g = xp.ones_like(alpha)
        a, b, phi, delta, method_param = self._solve(normalized_stack, g)

        # Eq. (8) evaluated at the fitted parameters, vs. the raw input --
        # a method-agnostic fit-quality check (works the same for any
        # method, since it only depends on the shared a/b/phi/delta/g/alpha
        # contract, not on how they were produced).
        carrier = g[:, xp.newaxis, xp.newaxis] * b[xp.newaxis, :, :] \
            * xp.cos(phi[xp.newaxis, :, :] + delta[:, xp.newaxis, xp.newaxis])
        rec_stack = alpha[:, xp.newaxis, xp.newaxis] * (a[xp.newaxis, :, :] + carrier)
        rmse = float(xp.sqrt(xp.mean((stack - rec_stack) ** 2)))

        self.result_ = PhaseResult(phi, a, b, delta, g, alpha, method_param, rmse)
        return self
    
    def _solve(self, stack: np.ndarray, g: np.ndarray):
        """Dispatch to the configured method and recover its Eq. (8) fields.

        Looks up ``config.method`` in :data:`phase.methods.METHOD_REGISTRY`
        (already validated to exist by :class:`PhaseConfig`) and calls it
        with the normalized ``stack``, resolved ``g``, this solver's
        ``dtype``, and ``config.method_kwargs``.

        Returns
        -------
        a, b, phi, delta, method_param
            See the chosen method's function for details (e.g.
            :func:`phase.methods.aia.aia` for ``method="aia"``).
        """
        solve_fn = METHOD_REGISTRY[self.config.method.lower()]
        return solve_fn(stack, g, dtype=self.dtype, **self.config.method_kwargs)

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

    def _estimate_gain(self, stack: np.ndarray, dc_radius: int = 8,
                        halfwin: tuple = (3, 4), frame_chunk: int = 8) -> np.ndarray:
        """Measure each frame's fringe gain ``g_n`` from its spatial carrier.

        Parameters
        ----------
        stack : np.ndarray, shape (N, H, W)
            Interferogram stack, already moved to the target device/dtype
            by :meth:`fit`.
        dc_radius, halfwin, frame_chunk
            See :func:`phase.utils.measure_frame_contrast`.

        Returns
        -------
        np.ndarray, shape (N,)
            Per-frame fringe gain, normalized so ``median(g) = 1``.
        """
        return measure_frame_contrast(stack, dc_radius, halfwin, frame_chunk, self.dtype)


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
