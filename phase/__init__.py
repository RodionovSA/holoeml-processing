"""Phase processing: interferogram stack -> wrapped phase -> object phase.

Public API, re-exported here regardless of which submodule a name lives in:

- :mod:`phase.aia` -- ``aia``, ``measure_frame_contrast``,
  ``measure_frame_visibility``, ``AIAResult``
- :mod:`phase.carrier` -- ``remove_carrier``, ``CarrierResult``
- :mod:`phase.reference` -- ``subtract_reference``, ``DifferenceResult``
- :mod:`phase.ripple` -- ``estimate_phase_ripple``, ``apply_phase_ripple``,
  ``RippleResult``
- :mod:`phase.combine` -- ``combine_acquisitions``, ``CombinedResult``
- :mod:`phase.backend` -- NumPy/CuPy array-module dispatch shared by all of
  the above; every function accepts a ``device="auto"|"cpu"|"cuda"``
  argument (see e.g. :func:`aia`'s docstring) and returns result arrays on
  whichever device it ran on. Use ``phase.backend.asnumpy`` to bring a
  result field back to the host explicitly.
"""

from . import backend
from .aia import AIAResult, aia, measure_frame_contrast, measure_frame_visibility
from .backend import asnumpy
from .carrier import CarrierResult, remove_carrier
from .combine import CombinedResult, combine_acquisitions
from .reference import DifferenceResult, subtract_reference
from .ripple import RippleResult, apply_phase_ripple, estimate_phase_ripple

__all__ = [
    "AIAResult",
    "aia",
    "measure_frame_contrast",
    "measure_frame_visibility",
    "CarrierResult",
    "remove_carrier",
    "DifferenceResult",
    "subtract_reference",
    "RippleResult",
    "estimate_phase_ripple",
    "apply_phase_ripple",
    "CombinedResult",
    "combine_acquisitions",
    "backend",
    "asnumpy",
]
