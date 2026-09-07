"""Registry of phase-recovery method implementations for :class:`phase.solver.PhaseSolver`.

Each entry maps a method name to a callable
``(stack, g, fit_gain=False, dtype=None, **method_kwargs) -> (a, b, phi, delta, g, method_param)``,
matching Eq. (8) of ``docs/interference_model.md``. The returned ``g`` is the
input ``g`` passed through unchanged when ``fit_gain=False``, or the
per-frame contrast jointly recovered alongside ``delta`` (normalized to
``median(g) = 1``) when ``fit_gain=True`` -- see :func:`phase.methods.aia.aia`
for the reference implementation. Add a method by writing such a function in
its own module here and registering it below -- :data:`phase.solver.METHODS`
is derived from this dict's keys, nothing else changes.
"""

from .base import MethodParam
from .aia import aia
from .step_field import aia_step_field

METHOD_REGISTRY = {
    "aia": aia,
    "aia_step_field": aia_step_field,
    # Kept as an alias (degree defaults to 1, a pure linear tilt) so
    # existing configs/notebooks written against the old name keep working.
    "aia_tilt": aia_step_field,
}
