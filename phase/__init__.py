"""Phase processing: interferogram stack -> wrapped phase -> object phase.

Public API, re-exported here regardless of which submodule a name lives in:

- :mod:`phase.aia` -- ``aia``, ``measure_frame_contrast``, ``AIAResult``
- :mod:`phase.carrier` -- ``remove_carrier``, ``CarrierResult``
- :mod:`phase.reference` -- ``subtract_reference``, ``DifferenceResult``
- :mod:`phase.ripple` -- ``estimate_phase_ripple``, ``apply_phase_ripple``,
  ``RippleResult``
- :mod:`phase.combine` -- ``combine_acquisitions``, ``CombinedResult``
"""

from .aia import AIAResult, aia, measure_frame_contrast
from .carrier import CarrierResult, remove_carrier
from .combine import CombinedResult, combine_acquisitions
from .reference import DifferenceResult, subtract_reference
from .ripple import RippleResult, apply_phase_ripple, estimate_phase_ripple

__all__ = [
    "AIAResult",
    "aia",
    "measure_frame_contrast",
    "CarrierResult",
    "remove_carrier",
    "DifferenceResult",
    "subtract_reference",
    "RippleResult",
    "estimate_phase_ripple",
    "apply_phase_ripple",
    "CombinedResult",
    "combine_acquisitions",
]
