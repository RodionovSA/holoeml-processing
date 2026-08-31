"""Shared base type for per-method diagnostics.

Kept separate from :mod:`phase.solver` so the dependency between the two
packages stays one-directional: ``phase.solver`` imports from
``phase.methods``, never the reverse -- a method module (e.g.
:mod:`phase.methods.aia`) importing :class:`MethodParam` from here instead
of from ``phase.solver`` avoids a circular import.
"""

from dataclasses import dataclass, fields


def _fmt_value(value) -> str:
    """Format a diagnostic value: 6 significant figures for floats, ``str`` otherwise."""
    return f"{value:.6g}" if isinstance(value, float) else str(value)


@dataclass
class MethodParam:
    """Base marker type for a method's per-fit diagnostics.

    Each entry in :data:`phase.solver.METHODS` returns its own subclass
    (e.g. ``aia`` returns :class:`phase.methods.aia.AIAParam`) carrying
    whatever diagnostics are specific to that algorithm (convergence,
    condition numbers, ...). Stored on
    :attr:`phase.solver.PhaseResult.method_param`.
    """

    def print_summary(self) -> None:
        """Print this method's diagnostic fields, one per line.

        Generic default, in dataclass declaration order -- a method
        overrides this for a specific order or subset (e.g.
        :meth:`phase.methods.aia.AIAParam.print_summary`).
        """
        for f in fields(self):
            print(f"{f.name}: {_fmt_value(getattr(self, f.name))}")
