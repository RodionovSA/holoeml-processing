"""Reference spectra for converting relative spectrometer measurements to absolute.

Reflection is measured relative to a mirror: ``R_file = R_sample / R_mirror``, so
``R_sample = R_file * mirror_reflectance(wavelength)``.

Data live in ``processing/refdata/`` and are regenerated from the vendor workbook with
``scripts/extract_mirror_reference.py``.

Usage::

    from processing.references import mirror_reflectance

    R_abs = s_r["square1"] / 100 * mirror_reflectance(s_r.wavelength)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REFDATA = Path(__file__).parent / "refdata"
_POL = {"p": 1, "s": 2, "unpol": 3}


def load_mirror(aoi: int = 12, name: str = "thorlabs_p01_silver") -> np.ndarray:
    """Raw table ``(P, 4)``: ``wavelength [nm], R_p, R_s, R_unpol`` (fractions)."""
    path = REFDATA / f"{name}_aoi{aoi}.npy"
    if not path.exists():
        have = sorted(p.stem for p in REFDATA.glob("*.npy"))
        raise FileNotFoundError(f"no reference {path.name}; have {have}")
    return np.load(path)


def mirror_reflectance(wavelength, aoi: int = 12, pol: str = "unpol",
                       name: str = "thorlabs_p01_silver") -> np.ndarray:
    """Mirror reflectance (fraction) linearly interpolated at ``wavelength`` [nm].

    Parameters
    ----------
    wavelength : array_like
        nm.
    aoi : int, default 12
        Angle of incidence of the vendor curve: 12 (near-normal, what a
        normal-incidence spectrometer setup sees) or 45.
    pol : {"unpol", "p", "s"}
    name : str
        Mirror data set in ``processing/refdata/``.

    Raises
    ------
    ValueError
        If ``wavelength`` reaches outside the tabulated range.
    """
    if pol not in _POL:
        raise ValueError(f"pol must be one of {list(_POL)}, got {pol!r}")
    t = load_mirror(aoi, name)
    w = np.asarray(wavelength, dtype=float)
    if w.min() < t[0, 0] or w.max() > t[-1, 0]:
        raise ValueError(f"wavelength {w.min()}..{w.max()} outside tabulated "
                         f"{t[0, 0]}..{t[-1, 0]} nm")
    return np.interp(w, t[:, 0], t[:, _POL[pol]])
