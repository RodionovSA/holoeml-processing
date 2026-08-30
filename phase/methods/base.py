"""Shared base type for per-method diagnostics.

Kept separate from :mod:`phase.solver` so the dependency between the two
packages stays one-directional: ``phase.solver`` imports from
``phase.methods``, never the reverse -- a method module (e.g.
:mod:`phase.methods.aia`) importing :class:`MethodParam` from here instead
of from ``phase.solver`` avoids a circular import.
"""

from dataclasses import dataclass


@dataclass
class MethodParam:
    """Base marker type for a method's per-fit diagnostics.

    Each entry in :data:`phase.solver.METHODS` returns its own subclass
    (e.g. ``aia`` returns :class:`phase.methods.aia.AIAParam`) carrying
    whatever diagnostics are specific to that algorithm (convergence,
    condition numbers, ...). Stored on
    :attr:`phase.solver.PhaseResult.method_param`.
    """
