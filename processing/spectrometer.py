"""Reading OceanView spectrometer text exports into NumPy arrays.

One file is one spectrum: a ``Key: value`` header, then ``>>>>>Begin Spectral Data<<<<<``,
then ``wavelength<TAB>counts`` rows. File names follow OceanView's
``<base>__<index>__<HH-MM-SS-mmm>.txt``, where ``<base>`` is what was typed in, here
``<experiment>_<sample>_<spot>_<Mode>`` (e.g. ``HoloEML281026_LPCVD_meta125_Reflection``).

The spot name (``meta125``, ``square1``, ``glass``) is taken as the last ``_`` token of
``<base>`` that is not a measurement mode, and split into a text ``group`` and a numeric
``value`` (``meta``/``125.0``, ``square``/``1.0``, ``glass``/``nan``). Taking it from the
right keeps it correct when a leading token is missing from a file name.

Usage::

    from processing.spectrometer import read_folder

    s = read_folder(".../spectrometer/LPCVD/R")
    s.wavelength            # (P,) nm, shared by all files
    s.counts                # (N, P)
    s.names                 # ["meta125", "meta150", ..., "square1", ...]

    meta = s.select(group="meta")      # only meta*, sorted by value
    meta.values                        # [125., 150., 175., 200., 225.]
    s["glass"]                         # (P,) or (k, P) if measured k times
    s.per_second()                     # counts / integration time
    s.resample(400, 800, 5)            # (N, 81) on 400, 405, ..., 800 nm
    read_folder(path, grid=(400, 800, 5), method="mean")   # same, at read time
    s.smooth(12)                       # Savitzky-Golay, 12 nm window, native pixels
    read_folder(path, smooth=12, grid=(400, 800, 5))       # smooth, then resample

Spectra with different integration times are not comparable as raw counts;
:func:`read_folder` warns when that is the case, see :meth:`SpectraSet.per_second`.
"""

from __future__ import annotations

import io
import math
import os
import re
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_MARKER = ">>>>>Begin Spectral Data<<<<<"
END_MARKER = ">>>>>End Spectral Data<<<<<"
MODES = ("Reflection", "Transmission", "Absorbance", "Irradiance", "Scope", "Counts")

_SUFFIX = re.compile(r"^(?P<base>.+?)__(?P<index>\d+)__(?P<time>\d{2}-\d{2}-\d{2}-\d{3})$")
_SPOT = re.compile(r"^(?P<group>.*?)(?P<value>[-+]?\d+(?:\.\d+)?)$")


# --------------------------------------------------------------------------- names

@dataclass(frozen=True)
class FileName:
    """What a spectrum's file name says about it.

    Attributes
    ----------
    stem : str
        File name without extension.
    tokens : tuple[str, ...]
        ``<base>`` split on ``_``.
    name : str
        Spot name: last token that is not in :data:`MODES`.
    group : str
        Text part of ``name`` before a trailing number (``"meta"``); all of ``name``
        if it has none.
    value : float
        Trailing number of ``name`` (``125.0``), ``nan`` if it has none.
    mode : str or None
        Measurement mode token (``"Reflection"``), if present.
    index : int
        OceanView's file counter; ``-1`` if the name has no OceanView suffix.
    time : str or None
        OceanView's ``HH-MM-SS-mmm`` save time.
    """

    stem: str
    tokens: tuple[str, ...]
    name: str
    group: str
    value: float
    mode: str | None
    index: int
    time: str | None


def parse_name(path: str | os.PathLike, modes=MODES) -> FileName:
    """Split a spectrum file name into its parts; see :class:`FileName`.

    Names without the OceanView ``__<index>__<time>`` suffix are accepted; the whole
    stem is then ``<base>``.
    """
    stem = Path(path).stem
    m = _SUFFIX.match(stem)
    base, index, time = (m["base"], int(m["index"]), m["time"]) if m else (stem, -1, None)
    tokens = tuple(t for t in base.split("_") if t)
    mode = next((t for t in reversed(tokens) if t in modes), None)
    rest = [t for t in tokens if t not in modes]
    name = rest[-1] if rest else base
    s = _SPOT.match(name)
    group, value = (s["group"], float(s["value"])) if s and s["group"] else (name, math.nan)
    return FileName(stem, tokens, name, group, value, mode, index, time)


# --------------------------------------------------------------------------- single file

@dataclass
class Spectrum:
    """One spectrometer file.

    Attributes
    ----------
    wavelength : np.ndarray
        ``(P,)`` x axis, nm (or whatever the header's ``XAxis mode`` says).
    counts : np.ndarray
        ``(P,)`` y values as saved (raw counts, or R/T if OceanView computed it).
    header : dict
        Header ``Key: value`` pairs, numbers/booleans converted; ``Date`` parsed into
        ``datetime`` when possible.
    file : FileName
        Parsed file name.
    path : Path
    """

    wavelength: np.ndarray
    counts: np.ndarray
    header: dict
    file: FileName
    path: Path

    @property
    def integration_time(self) -> float:
        """Integration time in seconds, ``nan`` if missing from the header."""
        return float(self.header.get("Integration Time (sec)", math.nan))


def read_spectrum(path: str | os.PathLike, dtype=np.float64) -> Spectrum:
    """Read one OceanView ``.txt`` export."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    head, sep, body = text.partition(DATA_MARKER)
    if not sep:
        raise ValueError(f"{path.name}: no '{DATA_MARKER}' line, not an OceanView export?")
    body = body.split(END_MARKER, 1)[0]
    data = np.loadtxt(io.StringIO(body), dtype=dtype, ndmin=2)
    if data.shape[1] < 2:
        raise ValueError(f"{path.name}: expected two data columns, got {data.shape[1]}")

    header = {}
    for line in head.splitlines():
        key, colon, val = line.partition(":")
        if colon and key.strip():
            header[key.strip()] = _convert(key.strip(), val.strip())

    n = header.get("Number of Pixels in Spectrum")
    if isinstance(n, int) and n != len(data):
        warnings.warn(f"{path.name}: header says {n} pixels, file has {len(data)}", stacklevel=2)
    return Spectrum(data[:, 0].copy(), data[:, 1].copy(), header, parse_name(path), path)


def _convert(key: str, val: str):
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    if key == "Date":
        # "Sun Nov 08 10:34:46 AST 2026": drop the zone name, strptime can't parse it.
        parts = val.split()
        try:
            return datetime.strptime(" ".join(parts[:4] + parts[5:]), "%a %b %d %H:%M:%S %Y")
        except (ValueError, IndexError):
            pass
    return val


# --------------------------------------------------------------------------- folder

@dataclass
class SpectraSet:
    """``N`` spectra on one shared wavelength axis.

    Per-spectrum fields are aligned along axis 0 of ``counts``. Build with
    :func:`read_folder`; :meth:`select` and :meth:`sorted` return new sets.

    Attributes
    ----------
    wavelength : np.ndarray
        ``(P,)``.
    counts : np.ndarray
        ``(N, P)``.
    files : list[FileName]
    headers : list[dict]
    paths : list[Path]
    """

    wavelength: np.ndarray
    counts: np.ndarray
    files: list[FileName]
    headers: list[dict]
    paths: list[Path] = field(default_factory=list)

    # -- per-spectrum arrays
    @property
    def names(self) -> list[str]:
        return [f.name for f in self.files]

    @property
    def groups(self) -> list[str]:
        return [f.group for f in self.files]

    @property
    def values(self) -> np.ndarray:
        """``(N,)`` trailing number of each spot name, ``nan`` where there is none."""
        return np.array([f.value for f in self.files])

    @property
    def integration_time(self) -> np.ndarray:
        """``(N,)`` seconds."""
        return np.array([float(h.get("Integration Time (sec)", math.nan)) for h in self.headers])

    @property
    def dates(self) -> list:
        return [h.get("Date") for h in self.headers]

    def __len__(self) -> int:
        return len(self.files)

    def __repr__(self) -> str:
        return (f"SpectraSet(N={len(self)}, P={self.wavelength.size}, "
                f"wavelength={self.wavelength[0]:.1f}..{self.wavelength[-1]:.1f}, "
                f"names={self.names})")

    # -- access
    def __getitem__(self, key):
        """``s["glass"]`` -> counts of that spot, ``(P,)``, or ``(k, P)`` if measured
        ``k`` times. Integers, slices, index arrays and boolean masks return a
        :class:`SpectraSet`."""
        if isinstance(key, str):
            idx = [i for i, n in enumerate(self.names) if n == key]
            if not idx:
                raise KeyError(f"no spectrum named {key!r}; have {sorted(set(self.names))}")
            return self.counts[idx[0]] if len(idx) == 1 else self.counts[idx]
        idx = np.arange(len(self))[key]
        return self._take(np.atleast_1d(idx))

    def _take(self, idx) -> SpectraSet:
        idx = [int(i) for i in idx]
        return replace(self, counts=self.counts[idx],
                       files=[self.files[i] for i in idx],
                       headers=[self.headers[i] for i in idx],
                       paths=[self.paths[i] for i in idx] if self.paths else [])

    def select(self, group: str | None = None, names=None, mode: str | None = None) -> SpectraSet:
        """Subset by spot ``group`` (``"meta"``), explicit ``names``, and/or ``mode``."""
        keep = np.ones(len(self), bool)
        if group is not None:
            keep &= np.array([g == group for g in self.groups])
        if names is not None:
            names = {names} if isinstance(names, str) else set(names)
            keep &= np.array([n in names for n in self.names])
        if mode is not None:
            keep &= np.array([f.mode == mode for f in self.files])
        return self._take(np.flatnonzero(keep))

    def sorted(self) -> SpectraSet:
        """Order by group, then numeric value, then save time."""
        def key(i):
            f, d = self.files[i], self.headers[i].get("Date")
            return (f.group, math.isnan(f.value), 0.0 if math.isnan(f.value) else f.value,
                    d if isinstance(d, datetime) else datetime.min, f.time or "", f.index)
        return self._take(sorted(range(len(self)), key=key))

    # -- transforms
    def per_second(self) -> SpectraSet:
        """Counts divided by each spectrum's integration time (counts/s)."""
        t = self.integration_time
        if np.any(~np.isfinite(t) | (t <= 0)):
            raise ValueError("integration time missing or non-positive in some headers")
        return replace(self, counts=self.counts / t[:, None])

    def mean_duplicates(self) -> SpectraSet:
        """Average spectra with the same spot name (keeps first occurrence's metadata)."""
        order, groups = [], {}
        for i, n in enumerate(self.names):
            if n not in groups:
                order.append(n)
                groups[n] = []
            groups[n].append(i)
        first = [groups[n][0] for n in order]
        out = self._take(first)
        out.counts = np.stack([self.counts[groups[n]].mean(0) for n in order])
        return out

    def crop(self, wmin: float | None = None, wmax: float | None = None) -> SpectraSet:
        """Keep ``wmin <= wavelength <= wmax``."""
        w = self.wavelength
        m = (w >= (-np.inf if wmin is None else wmin)) & (w <= (np.inf if wmax is None else wmax))
        return replace(self, wavelength=w[m], counts=self.counts[:, m])

    def resample(self, start: float | None = None, stop: float | None = None,
                 step: float | None = None, *, wavelength=None,
                 method: str = "linear") -> SpectraSet:
        """Put all spectra on a new wavelength grid.

        Give either ``start, stop, step`` (``stop`` included, e.g. ``400, 800, 5`` ->
        81 points) or an explicit ``wavelength`` array.

        Parameters
        ----------
        method : {"linear", "mean"}
            ``"linear"``: linear interpolation at each new point (``np.interp``).
            ``"mean"``: average of all pixels within ``±step/2`` of each new point
            (step = local grid spacing). When the new step is much coarser than the
            pixel pitch (5 nm vs ~0.4 nm here) this uses every pixel and lowers noise
            by about ``sqrt(pixels per bin)``; ``"linear"`` keeps the noise of single
            pixels.

        Raises
        ------
        ValueError
            If the new grid reaches outside the measured range, or a ``"mean"`` bin
            holds no pixel.
        """
        new = _grid(start, stop, step, wavelength)
        w = self.wavelength
        if new[0] < w[0] or new[-1] > w[-1]:
            raise ValueError(f"grid {new[0]}..{new[-1]} outside measured {w[0]}..{w[-1]}")
        if method == "linear":
            counts = np.stack([np.interp(new, w, c) for c in self.counts])
        elif method == "mean":
            if new.size < 2:
                raise ValueError("method='mean' needs at least two grid points")
            # Bin i is [edge_i, edge_{i+1}), edges halfway between grid points.
            edges = np.concatenate([[1.5 * new[0] - 0.5 * new[1]], (new[1:] + new[:-1]) / 2,
                                    [1.5 * new[-1] - 0.5 * new[-2]]])
            idx = np.searchsorted(w, edges)
            n = np.diff(idx)
            if np.any(n == 0):
                raise ValueError(f"empty bin at {new[np.flatnonzero(n == 0)[0]]}; "
                                 "step finer than pixel pitch, use method='linear'")
            csum = np.concatenate([np.zeros((len(self), 1)), np.cumsum(self.counts, 1)], 1)
            counts = (csum[:, idx[1:]] - csum[:, idx[:-1]]) / n
        else:
            raise ValueError(f"method must be 'linear' or 'mean', got {method!r}")
        return replace(self, wavelength=new, counts=counts)

    def smooth(self, width: float, method: str = "savgol", order: int = 3) -> SpectraSet:
        """Smooth every spectrum along wavelength.

        Apply on the native pixels, before :meth:`resample`: the filter then sees all
        the data, and ``"mean"`` resampling afterwards adds little further blur.

        Parameters
        ----------
        width : float
            Window in wavelength units (nm). ``"savgol"``: full window length, rounded
            to an odd number of pixels. ``"gaussian"``: FWHM of the kernel.
            Features narrower than about ``width`` get damped, so keep it below the
            narrowest real feature (resonance FWHM); noise with a period longer than
            ``width`` passes through.
        method : {"savgol", "gaussian"}
            ``"savgol"``: Savitzky-Golay, a local polynomial fit of degree ``order``
            that keeps peak height and width much better than a moving average of
            the same length; edges use the fit of the first/last full window.
            ``"gaussian"``: normalized Gaussian convolution; smoother result, but it
            broadens peaks.
        order : int, default 3
            Polynomial degree for ``"savgol"``.
        """
        w = self.wavelength
        dw = float(np.mean(np.diff(w)))
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if method == "savgol":
            n = int(round(width / dw)) // 2 * 2 + 1      # odd pixel count
            n = max(n, order + 1 + order % 2)              # odd and > order
            counts = _savgol(self.counts, n, order)
        elif method == "gaussian":
            sigma = width / dw / (2 * math.sqrt(2 * math.log(2)))
            half = max(int(math.ceil(4 * sigma)), 1)
            k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
            norm = np.convolve(np.ones(w.size), k, "same")
            counts = np.stack([np.convolve(c, k, "same") / norm for c in self.counts])
        else:
            raise ValueError(f"method must be 'savgol' or 'gaussian', got {method!r}")
        if counts.shape[1] != w.size:
            raise ValueError(f"window ({width} nm) longer than the spectrum")
        return replace(self, counts=counts)

    def to_dict(self) -> dict:
        """Plain arrays, e.g. for ``np.savez(path, **s.to_dict())``."""
        return dict(wavelength=self.wavelength, counts=self.counts,
                    names=np.array(self.names), groups=np.array(self.groups),
                    values=self.values, integration_time=self.integration_time)


def _savgol(y: np.ndarray, n: int, order: int) -> np.ndarray:
    """Savitzky-Golay along axis 1 of ``(N, P)``; window ``n`` odd pixels, ``P >= n``."""
    P = y.shape[1]
    if P < n:
        return y[:, :0]  # caller reports the error
    h = n // 2
    x = np.arange(-h, h + 1, dtype=float)
    V = np.vander(x, order + 1, increasing=True)          # (n, order+1)
    fit = np.linalg.pinv(V)                               # rows: polynomial coefficients
    win = np.lib.stride_tricks.sliding_window_view(y, n, axis=1)   # (N, P-n+1, n)
    out = np.empty_like(y, dtype=float)
    out[:, h:P - h] = win @ fit[0]                        # value at window centre
    # Edges: evaluate the polynomial fitted to the first/last full window.
    xe = np.arange(-h, 0, dtype=float)
    Ve = np.vander(xe, order + 1, increasing=True)
    out[:, :h] = (y[:, :n] @ fit.T) @ Ve.T
    out[:, P - h:] = (y[:, P - n:] @ fit.T) @ np.vander(-xe[::-1], order + 1, increasing=True).T
    return out


def _grid(start, stop, step, wavelength) -> np.ndarray:
    if wavelength is not None:
        if any(v is not None for v in (start, stop, step)):
            raise ValueError("give either start/stop/step or wavelength, not both")
        new = np.asarray(wavelength, dtype=float).ravel()
        if new.size == 0 or np.any(np.diff(new) <= 0):
            raise ValueError("wavelength must be non-empty and strictly increasing")
        return new
    if None in (start, stop, step) or step <= 0 or stop < start:
        raise ValueError(f"need start <= stop and step > 0, got {start}, {stop}, {step}")
    n = int(round((stop - start) / step))
    return float(start) + float(step) * np.arange(n + 1)


def read_folder(folder: str | os.PathLike, pattern: str = "*.txt", *, sort: bool = True,
                recursive: bool = False, wavelength_rtol: float = 1e-6,
                smooth: float | None = None,
                grid: tuple[float, float, float] | None = None, method: str = "linear",
                dtype=np.float64) -> SpectraSet:
    """Read every spectrum in ``folder`` into one :class:`SpectraSet`.

    Parameters
    ----------
    folder : path
    pattern : str, default ``"*.txt"``
        Glob for the files to read.
    sort : bool, default True
        Order with :meth:`SpectraSet.sorted`; otherwise file-name order.
    recursive : bool, default False
        Search subfolders too (``rglob``).
    wavelength_rtol : float
        All files must share one wavelength axis to this relative tolerance;
        raises ``ValueError`` naming the first file that differs otherwise.
    smooth : float, optional
        Savitzky-Golay window in nm, applied on the native pixels before ``grid``;
        see :meth:`SpectraSet.smooth` for other filters. ``None`` does not smooth.
    grid : (start, stop, step), optional
        Resample right after reading, e.g. ``(400, 800, 5)``; see
        :meth:`SpectraSet.resample`. ``None`` keeps the native pixels.
    method : {"linear", "mean"}
        Resampling method when ``grid`` is given.
    """
    folder = Path(folder)
    paths = sorted((folder.rglob if recursive else folder.glob)(pattern))
    paths = [p for p in paths if p.is_file() and not p.name.startswith(".")]
    if not paths:
        raise FileNotFoundError(f"no files matching {pattern!r} in {folder}")

    spectra = [read_spectrum(p, dtype=dtype) for p in paths]
    w0 = spectra[0].wavelength
    for s in spectra[1:]:
        if s.wavelength.shape != w0.shape or not np.allclose(s.wavelength, w0, rtol=wavelength_rtol, atol=0):
            raise ValueError(f"wavelength axis of {s.path.name} differs from {spectra[0].path.name}")

    out = SpectraSet(w0, np.stack([s.counts for s in spectra]),
                     [s.file for s in spectra], [s.header for s in spectra],
                     [s.path for s in spectra])
    t = out.integration_time
    if np.unique(t[np.isfinite(t)]).size > 1:
        warnings.warn(f"{folder.name}: integration times differ ({sorted(set(t.tolist()))} s); "
                      "raw counts are not comparable, consider .per_second()", stacklevel=2)
    if smooth is not None:
        out = out.smooth(smooth)
    if grid is not None:
        out = out.resample(*grid, method=method)
    return out.sorted() if sort else out
