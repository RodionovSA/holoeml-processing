"""Registry of phase-recovery method implementations for :class:`phase.solver.PhaseSolver`.

Each entry maps a method name to a callable
``(stack, g, dtype=None, **method_kwargs) -> (a, b, phi, delta, method_param)``,
matching Eq. (8) of ``docs/interference_model.md``. Add a method by writing such
a function in its own module here and registering it below --
:data:`phase.solver.METHODS` is derived from this dict's keys, nothing else changes.
"""

from .base import MethodParam
from .aia import aia

METHOD_REGISTRY = {
    "aia": aia,
}
