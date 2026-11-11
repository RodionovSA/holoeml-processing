"""Interferometer scan folders: dark correction, bad pixels, streaming processing to HDF5.

A scan folder holds one sample measured in wavelength blocks. Every block has its own
black (dark) frame and monochromator check, and one ``.npz`` per wavelength::

    holoeml281026_glass_400_455_black_20261108_043546.npz       dark, (Y, X) mean + std
    holoeml281026_glass_400_455_mono_check_20261108_043528.npz  monochromator spectrum
    holoeml281026_glass_400_455_400nm_20261108_043707.npz       one wavelength

A wavelength file holds the phase-stepped stacks at the sample and reference stage
positions (``stack_{sample,ref}_pos``, uint16, ``(repeat, N, Y, X)``) and the sample arm
alone, with the reference arm closed (``sample_arm_{sample,ref}_pos`` and their
``_std``, float32 ``(Y, X)``).

Processing a file:

1. subtract the block's black frame from every image;
2. replace bad pixels (from the black frame, :func:`bad_pixel_mask`) by the median of
   their good neighbours, :func:`median_fill`;
3. solve the phase with :class:`processing.phase.PhasePipeline` (sample minus
   reference) and put it on the sign branch of the first wavelength, one wavelength
   at a time (:class:`SignTracker`);
4. transmission ``T = I_sample / I_ref`` from the sample-arm images, with ``T_std``
   propagated from their stds, :func:`transmission`.

Optionally, once every wavelength is done (``layout:`` in the phase config, or a GDS
file passed to :func:`process_scan`):

5. align all maps to the GDS layout: one orientation and grid for every
   wavelength, per-wavelength drift removed, a plane fitted on the clean glass
   subtracted from ``phi``, and masks for every structure, the glass, particles and
   extended non-uniformities, :func:`processing.layout.align_scan`.

Only one wavelength file is in memory at a time. Results go to HDF5 files named
from an output prefix, with one ``(Wvl, Y, X)`` dataset per quantity, written row by
row (:class:`ScanStore`):

- ``<prefix>_phi.h5``: ``phi``, ``phi_error``, ``b_sample``, ``b_ref``;
- ``<prefix>_T.h5``: ``T``, ``T_std`` (optional: skip it when re-running only a
  different phase processing).

Both also hold ``wavelength``, ``bad_pixels`` and per-wavelength metadata. An
interrupted run resumes where it stopped.

Usage::

    from processing.interferometer import scan_folder, process_scan
    from processing.phase import PhasePipeline

    scan = scan_folder(".../interferometer/SiO2")
    process_scan(scan, PhasePipeline.from_yaml("configs/phase.yaml"), "out/SiO2_aia")

    import h5py
    with h5py.File("out/SiO2_aia_phi.h5") as f:
        wvl = f["wavelength"][:]
        phi_550 = f["phi"][np.argmin(abs(wvl - 550))]   # reads one map only
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from phase_shift import align_signs

NAME_RE = re.compile(
    r"^(?P<exp>[^_]+)_(?P<sample>.+?)_(?P<start>\d+)_(?P<stop>\d+)_"
    r"(?P<kind>\d+(?:\.\d+)?nm|black|mono_check)_(?P<date>\d{8})_(?P<time>\d{6})$"
)
STACK_KEYS = ("stack_sample_pos", "stack_ref_pos")
ARM_KEYS = ("sample_arm_sample_pos", "sample_arm_ref_pos")
ARM_STD_KEYS = ("sample_arm_std_sample_pos", "sample_arm_std_ref_pos")


# --------------------------------------------------------------------------- folder

@dataclass(frozen=True)
class ScanFile:
    """One file of a scan folder, described by its name.

    Attributes
    ----------
    path : Path
    exp, sample : str
        ``holoeml281026``, ``glass``.
    block : str
        Wavelength block the file belongs to, e.g. ``"400_455"``.
    kind : {"wvl", "black", "mono_check"}
    wavelength : float or None
        nm, for ``kind == "wvl"``.
    time : datetime
    """

    path: Path
    exp: str
    sample: str
    block: str
    kind: str
    wavelength: float | None
    time: datetime


def parse_name(path: str | Path) -> ScanFile:
    """Parse ``<exp>_<sample>_<start>_<stop>_<kind>_<YYYYMMDD>_<HHMMSS>.npz``.

    ``<kind>`` is ``<wavelength>nm``, ``black`` or ``mono_check``; ``<sample>`` may
    contain underscores.

    Raises
    ------
    ValueError
        If the name does not follow the pattern.
    """
    path = Path(path)
    m = NAME_RE.match(path.stem)
    if m is None:
        raise ValueError(f"cannot parse scan file name {path.name!r}")
    kind = m["kind"]
    wavelength = None
    if kind.endswith("nm"):
        wavelength, kind = float(kind[:-2]), "wvl"
    return ScanFile(path=path, exp=m["exp"], sample=m["sample"],
                    block=f"{m['start']}_{m['stop']}", kind=kind, wavelength=wavelength,
                    time=datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S"))


@dataclass
class Block:
    """Wavelength files of one block, sorted by wavelength, and their black frame."""

    name: str
    black: Path
    mono_check: Path | None
    files: list[ScanFile]


@dataclass
class Scan:
    """All blocks of one sample folder, sorted by wavelength."""

    folder: Path
    exp: str
    sample: str
    blocks: list[Block]

    @property
    def name(self) -> str:
        return f"{self.exp}_{self.sample}"

    @property
    def files(self) -> list[ScanFile]:
        return [f for b in self.blocks for f in b.files]

    @property
    def wavelengths(self) -> np.ndarray:
        return np.array([f.wavelength for f in self.files])

    def block_of(self, file: ScanFile) -> Block:
        return next(b for b in self.blocks if b.name == file.block)


def scan_folder(folder: str | Path, pattern: str = "*.npz") -> Scan:
    """Group the files of a scan folder into blocks.

    Files whose names do not parse are skipped with a warning.

    Raises
    ------
    ValueError
        If the folder holds several experiments/samples, a block has no black frame
        or more than one, or a wavelength appears twice.
    """
    folder = Path(folder)
    parsed = []
    for p in sorted(folder.glob(pattern)):
        try:
            parsed.append(parse_name(p))
        except ValueError:
            warnings.warn(f"skipping {p.name}: name does not match the scan pattern",
                          stacklevel=2)
    if not parsed:
        raise ValueError(f"no scan files matching {pattern!r} in {folder}")
    ids = {(f.exp, f.sample) for f in parsed}
    if len(ids) != 1:
        raise ValueError(f"folder mixes several experiments/samples: {sorted(ids)}")
    exp, sample = ids.pop()

    blocks = []
    for name in sorted({f.block for f in parsed}, key=lambda s: tuple(map(int, s.split("_")))):
        fs = [f for f in parsed if f.block == name]
        blacks = [f.path for f in fs if f.kind == "black"]
        if len(blacks) != 1:
            raise ValueError(f"block {name}: expected one black frame, found {len(blacks)}")
        monos = sorted((f for f in fs if f.kind == "mono_check"), key=lambda f: f.time)
        wvls = sorted((f for f in fs if f.kind == "wvl"), key=lambda f: f.wavelength)
        if wvls:
            blocks.append(Block(name, blacks[0], monos[-1].path if monos else None, wvls))
    all_w = [f.wavelength for b in blocks for f in b.files]
    dup = sorted({w for w in all_w if all_w.count(w) > 1})
    if dup:
        raise ValueError(f"wavelengths measured more than once: {dup}")
    blocks.sort(key=lambda b: b.files[0].wavelength)
    return Scan(folder, exp, sample, blocks)


# --------------------------------------------------------------------------- dark, pixels

def bad_pixel_mask(black: np.ndarray, black_std: np.ndarray, k_offset: float = 10.0,
                   k_std: float = 5.0) -> np.ndarray:
    """Bad pixels from a dark frame.

    A pixel is bad if it is

    - hot: ``|black - median(black)| > k_offset * MAD`` (MAD scaled to a Gaussian
      sigma);
    - noisy: ``black_std > k_std * median(black_std)``;
    - stuck: ``black_std == 0``.

    Parameters
    ----------
    black, black_std : np.ndarray, shape (Y, X)
        Mean and std of the dark frames.
    k_offset, k_std : float
        Thresholds above.

    Returns
    -------
    np.ndarray of bool, shape (Y, X)
        True where bad.
    """
    med = np.median(black)
    mad = 1.4826 * np.median(np.abs(black - med))
    hot = np.abs(black - med) > k_offset * max(mad, np.finfo(np.float32).eps)
    noisy = black_std > k_std * np.median(black_std)
    stuck = black_std == 0
    return hot | noisy | stuck


@dataclass
class DarkFrame:
    """Black frame of one block and the bad pixels found in it."""

    black: np.ndarray
    black_std: np.ndarray
    bad: np.ndarray
    exposure_ms: float
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path, k_offset: float = 10.0, k_std: float = 5.0) -> DarkFrame:
        """Read a ``*_black_*.npz`` and build its bad-pixel mask (:func:`bad_pixel_mask`)."""
        with np.load(path) as z:
            black = z["black"].astype(np.float32)
            black_std = z["black_std"].astype(np.float32)
            exposure = float(z["exposure_ms"])
        return cls(black, black_std, bad_pixel_mask(black, black_std, k_offset, k_std),
                   exposure, Path(path))


def median_fill(img: np.ndarray, bad: np.ndarray, max_radius: int = 5) -> np.ndarray:
    """Replace bad pixels, in place, by the median of their good neighbours.

    Each bad pixel takes the median of the good pixels in the ``(2r+1)^2`` window
    around it, starting with ``r = 1`` and growing up to ``max_radius`` for pixels
    inside a bad cluster. Only originally good pixels are used as sources.

    Parameters
    ----------
    img : np.ndarray, shape (..., Y, X), float
        Image or stack; every leading index is filled with the same pixel pattern.
    bad : np.ndarray of bool, shape (Y, X)
    max_radius : int, default 5

    Returns
    -------
    np.ndarray
        ``img``, modified in place.

    Warns
    -----
    UserWarning
        If some bad pixels have no good neighbour within ``max_radius``; those are
        left unchanged.
    """
    if bad.shape != img.shape[-2:]:
        raise ValueError(f"bad mask shape {bad.shape} does not match image {img.shape[-2:]}")
    H, W = bad.shape
    ys, xs = np.nonzero(bad)
    todo = np.arange(ys.size)
    for r in range(1, max_radius + 1):
        if not todo.size:
            break
        dy, dx = (a.ravel() for a in np.mgrid[-r:r + 1, -r:r + 1])
        yy, xx = ys[todo, None] + dy, xs[todo, None] + dx
        inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        yy, xx = np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)
        valid = inside & ~bad[yy, xx]
        has = valid.any(axis=1)
        if has.any():
            vals = np.where(valid[has], img[..., yy[has], xx[has]], np.nan)
            img[..., ys[todo[has]], xs[todo[has]]] = np.nanmedian(vals, axis=-1)
        todo = todo[~has]
    if todo.size:
        warnings.warn(f"{todo.size} bad pixels have no good neighbour within "
                      f"{max_radius} px; left unchanged", stacklevel=2)
    return img


def load_corrected(z, key: str, dark: DarkFrame) -> np.ndarray:
    """Read one image or stack from an open npz, subtract black, fill bad pixels.

    Stacks ``(repeat, N, Y, X)`` are flattened to ``(repeat * N, Y, X)``.

    Returns
    -------
    np.ndarray, float32
    """
    a = z[key]
    out = a.reshape(-1, *a.shape[-2:]) if a.ndim > 3 else a
    out = out.astype(np.float32)          # new array; the raw one is dropped below
    del a
    out -= dark.black
    return median_fill(out, dark.bad)


def load_std(z, key: str, dark: DarkFrame) -> np.ndarray:
    """Read a std image (no black subtraction) and fill bad pixels."""
    return median_fill(z[key].astype(np.float32), dark.bad)


def transmission(i_sample: np.ndarray, i_ref: np.ndarray, std_sample: np.ndarray,
                 std_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``T = I_sample / I_ref`` and its std from independent errors.

    ``T_std = sqrt(std_sample^2 + T^2 std_ref^2) / I_ref``, i.e. first-order
    propagation of uncorrelated ``std_sample`` and ``std_ref``. Pass the std of the
    mean if that is what ``T_std`` should describe. Pixels with ``I_ref <= 0`` are
    NaN.

    Returns
    -------
    T, T_std : np.ndarray, float32
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        i_ref = np.where(i_ref > 0, i_ref, np.nan)
        t = i_sample / i_ref
        t_std = np.sqrt(std_sample ** 2 + t ** 2 * std_ref ** 2) / i_ref
    return t.astype(np.float32), t_std.astype(np.float32)


# --------------------------------------------------------------------------- sign tracking

class SignTracker:
    """Put a stream of phase maps on the sign branch of the first one.

    Same rule as :func:`phase_shift.align_signs`, which needs the whole stack in
    memory: each map is compared with the last map that was resolved confidently,
    so only that one map (and its weight) is kept.
    """

    def __init__(self, ambiguous_ratio: float = 1.5) -> None:
        self.ambiguous_ratio = ambiguous_ratio
        self.last: tuple[int, np.ndarray, np.ndarray | None] | None = None

    def reset_to(self, index: int, phi: np.ndarray, weight: np.ndarray | None) -> None:
        """Use an already aligned map as the comparison map (e.g. on resume)."""
        self.last = (index, phi, weight)

    def align(self, index: int, phi: np.ndarray, weight: np.ndarray | None = None,
              mask: np.ndarray | None = None) -> tuple[np.ndarray, int, bool, int]:
        """Return ``(aligned phi, sign, ambiguous, compared-with index)``.

        The first map defines the branch: sign +1, compared with -1.
        """
        if self.last is None:
            self.last = (index, phi, weight)
            return phi, 1, False, -1
        j, phi_j, w_j = self.last
        w = None if weight is None or w_j is None else np.stack([w_j, weight])
        res = align_signs(np.stack([phi_j, phi]), weight=w, mask=mask,
                          ambiguous_ratio=self.ambiguous_ratio, device="cpu")
        sign, amb = int(res.signs[1]), bool(res.ambiguous[1])
        out = res.phi[1]
        if not amb:
            self.last = (index, out, weight)
        return out, sign, amb, j


# --------------------------------------------------------------------------- HDF5 store

class ScanStore:
    """HDF5 file with one ``(Wvl, ...)`` dataset per quantity, filled one row at a time.

    Datasets are created on first write with the row's shape and dtype. Maps are
    chunked one row (wavelength) at a time in ``(256, 256)`` tiles, so reading one
    wavelength, a crop, or a pixel's spectrum is cheap. The ``done`` row flag is set
    last, so a row interrupted mid-write is redone on resume.

    Parameters
    ----------
    path : Path
    wavelength : np.ndarray, shape (Wvl,)
    attrs : dict, optional
        Stored as file attributes on creation.
    overwrite : bool, default False
        Start a new file even if one exists. Otherwise an existing file is resumed;
        its ``wavelength`` must match.
    compression : str or None, default "gzip"
        HDF5 filter for float maps: ``"gzip"`` (level 1, readable by any HDF5 tool),
        ``"lzf"`` (~10% larger, faster, h5py only) or None. Masks always use gzip.
    """

    def __init__(self, path: str | Path, wavelength: np.ndarray, attrs: dict | None = None,
                 overwrite: bool = False, compression: str | None = "gzip") -> None:
        import h5py

        self.path = Path(path)
        self.compression = compression
        new = overwrite or not self.path.exists()
        self.f = h5py.File(self.path, "w" if new else "a")
        if new:
            self.f["wavelength"] = np.asarray(wavelength, dtype=float)
            self.f["wavelength"].attrs["units"] = "nm"
            self.f.create_dataset("done", data=np.zeros(len(wavelength), dtype=bool))
            for k, v in (attrs or {}).items():
                self.f.attrs[k] = v
        elif not np.array_equal(self.f["wavelength"][:], wavelength):
            self.f.close()
            raise ValueError(f"{self.path}: stored wavelengths differ from this scan; "
                             f"use overwrite=True to start over")

    @property
    def done(self) -> np.ndarray:
        return self.f["done"][:]

    def write(self, k: int, **values) -> None:
        """Write row ``k`` of each named dataset (arrays or scalars)."""
        n = len(self.f["wavelength"])
        for name, v in values.items():
            v = np.asarray(v)
            if name not in self.f:
                kw = {}
                if v.ndim >= 2:
                    kw["chunks"] = (1,) + tuple(min(s, 256) for s in v.shape)
                    kw["compression"] = "gzip" if v.dtype == bool else self.compression
                    if kw["compression"]:
                        kw["shuffle"] = True
                    if kw["compression"] == "gzip":
                        kw["compression_opts"] = 1
                fill = np.nan if v.dtype.kind == "f" else None
                self.f.create_dataset(name, shape=(n,) + v.shape, dtype=v.dtype,
                                      fillvalue=fill, **kw)
            self.f[name][k] = v

    def finish_row(self, k: int) -> None:
        self.f["done"][k] = True
        self.f.flush()

    def close(self) -> None:
        self.f.close()

    def __enter__(self) -> ScanStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- processing

@dataclass
class WavelengthResult:
    """Per-wavelength output of :func:`process_file` (host arrays)."""

    phi: dict[str, np.ndarray]                      # maps for the phi file
    T: dict[str, np.ndarray]                        # maps for the T file
    meta: dict[str, object] = field(default_factory=dict)


def process_file(file: ScanFile | str | Path, dark: DarkFrame, pipeline,
                 transmission_: bool = True) -> WavelengthResult:
    """Dark-correct one wavelength file, solve its phase, optionally compute T.

    ``phi`` is the sample-minus-reference phase from ``pipeline`` (not yet aligned
    across wavelengths); ``phi_error = sqrt(err_sample^2 + err_ref^2)``. With
    ``transmission_=False`` the sample-arm images are not read and ``T`` is empty.

    Raises
    ------
    ValueError
        If the file's exposure differs from the black frame's.
    """
    path = file.path if isinstance(file, ScanFile) else Path(file)
    with np.load(path) as z:
        exposure = float(z["exposure_ms"])
        if not np.isclose(exposure, dark.exposure_ms):
            raise ValueError(f"{path.name}: exposure {exposure} ms differs from black "
                             f"frame {dark.exposure_ms} ms")
        meta = {k: z[k].item() for k in ("wavelength", "exposure_ms", "n", "n_back",
                                          "repeat", "mono_check_peak_nm") if k in z.files}

        sample = load_corrected(z, STACK_KEYS[0], dark)
        reference = load_corrected(z, STACK_KEYS[1], dark)
        res = pipeline.process_stack(sample, reference, output="full", mask=~dark.bad)
        del sample, reference
        gc.collect()

        T = {}
        if transmission_:
            i_s, i_r = (load_corrected(z, k, dark) for k in ARM_KEYS)
            s_s, s_r = (load_std(z, k, dark) for k in ARM_STD_KEYS)
            T["T"], T["T_std"] = transmission(i_s, i_r, s_s, s_r)
            del i_s, i_r, s_s, s_r

    rs, rr = res.sample, res.reference
    if rs.phi_error is None or rr.phi_error is None:
        phi_err = np.full(res.phi.shape, np.nan, dtype=np.float32)
    else:
        phi_err = np.hypot(np.asarray(rs.phi_error), np.asarray(rr.phi_error))
    phi = {
        "phi": np.asarray(res.phi, dtype=np.float32),
        "phi_error": phi_err.astype(np.float32),
        "b_sample": np.asarray(rs.b, dtype=np.float32),
        "b_ref": np.asarray(rr.b, dtype=np.float32),
    }
    d = res.difference
    meta.update(
        ref_sign=d.sign, ref_ambiguous=d.ambiguous,
        ref_spread_same=d.spread_same, ref_spread_flipped=d.spread_flipped,
        reconstruction_error_sample=rs.reconstruction_error,
        reconstruction_error_ref=rr.reconstruction_error,
        delta_sample=np.asarray(rs.delta), delta_ref=np.asarray(rr.delta),
        g_sample=np.asarray(rs.g), g_ref=np.asarray(rr.g),
        alpha_sample=np.asarray(rs.alpha), alpha_ref=np.asarray(rr.alpha),
    )
    del res
    return WavelengthResult(phi=phi, T=T, meta=meta)


def process_scan(scan: Scan, pipeline, out: str | Path, *, transmission: bool = True,
                 overwrite: bool = False, k_offset: float = 10.0, k_std: float = 5.0,
                 compression: str | None = "gzip", gds: str | Path | None = None,
                 progress: bool = True) -> tuple[Path, Path | None]:
    """Process every wavelength of ``scan`` into ``<out>_phi.h5`` and ``<out>_T.h5``.

    Wavelengths already marked done in every output file are skipped, so an
    interrupted run resumes; sign alignment then continues from the last confidently
    aligned stored map. ``phi`` is stored on the sign branch of the first wavelength
    (one global sign ambiguity remains, see :mod:`processing.phase`).

    With ``pipeline.config.layout.enabled`` (or ``gds`` given), this is pass 1 and
    writes ``<out>_phi_unaligned.h5`` / ``<out>_T_unaligned.h5``; pass 2,
    :func:`processing.layout.align_scan`, then writes the final files on the layout
    grid with structure, glass and defect masks, and removes the unaligned files
    unless ``layout.keep_unaligned``. Final files that an earlier run wrote without
    a layout are reused as pass-1 output. Pass 2 is redone as a whole if interrupted.

    Parameters
    ----------
    out : str or Path
        Output prefix, e.g. ``.../HoloEML281026_SiO2_aia``; a trailing ``.h5`` is
        dropped. Its folder must exist (it is not created).
    transmission : bool, default True
        Also compute and write ``<out>_T.h5``. T does not depend on the phase
        settings, so turn it off when re-running only a different phase processing.

    Returns
    -------
    phi_path : Path
    T_path : Path or None
        None when ``transmission`` is False.

    Raises
    ------
    FileNotFoundError
        If the folder of ``out`` does not exist.
    """
    import yaml

    out = Path(out)
    if out.suffix == ".h5":
        out = out.with_suffix("")
    if not out.parent.is_dir():
        raise FileNotFoundError(f"output folder {out.parent} does not exist")
    wvl = scan.wavelengths
    attrs = {"source_folder": str(scan.folder), "exp": scan.exp, "sample": scan.sample,
             "bad_pixel_k_offset": k_offset, "bad_pixel_k_std": k_std,
             "created": datetime.now().isoformat(timespec="seconds")}
    phi_attrs = dict(attrs, pipeline_config=yaml.safe_dump(pipeline.config.to_dict(),
                                                           sort_keys=False),
                     phi_convention="sample minus reference, rad, wrapped, sign branch of "
                                    "wavelength[0]")
    t_attrs = dict(attrs, T_convention="I_sample_arm(sample_pos) / I_sample_arm(ref_pos), "
                                       "dark subtracted; T_std propagated from the stored "
                                       "sample_arm stds")
    phi_path = out.with_name(out.name + "_phi.h5")
    t_path = out.with_name(out.name + "_T.h5") if transmission else None

    layout = pipeline.config.layout
    if gds is not None:
        layout = dataclasses.replace(layout, enabled=True, gds=str(gds))
    if layout.enabled:
        if not layout.gds:
            raise ValueError("layout alignment is enabled but no GDS file is given")
        phi_attrs["pipeline_config"] = yaml.safe_dump(
            dict(pipeline.config.to_dict(), layout=layout.to_dict()), sort_keys=False)
        finals = [p for p in (phi_path, t_path) if p is not None]
        if not overwrite and all(_is_aligned(p, complete=True) for p in finals):
            return phi_path, t_path
        phi_work = out.with_name(out.name + "_phi_unaligned.h5")
        t_work = out.with_name(out.name + "_T_unaligned.h5") if transmission else None
        for fin, work in ((phi_path, phi_work), (t_path, t_work)):
            if (fin is not None and fin.exists() and not work.exists() and not overwrite
                    and not _is_aligned(fin)):
                fin.rename(work)
                warnings.warn(f"reusing unaligned {fin.name} as pass-1 output {work.name}",
                              stacklevel=2)
    else:
        phi_work, t_work = phi_path, t_path

    with contextlib.ExitStack() as stack:
        sp = stack.enter_context(ScanStore(phi_work, wvl, phi_attrs, overwrite, compression))
        st = (stack.enter_context(ScanStore(t_work, wvl, t_attrs, overwrite, compression))
              if transmission else None)
        done = sp.done & st.done if st is not None else sp.done
        tracker = SignTracker(pipeline.config.ambiguous_ratio)
        if done.any():
            ok = done & ~sp.f["align_ambiguous"][:] if "align_ambiguous" in sp.f else done
            j = int(np.flatnonzero(ok)[-1]) if ok.any() else int(np.flatnonzero(done)[-1])
            weight = sp.f["b_ref"][j] if pipeline.config.weight == "b_ref" else None
            tracker.reset_to(j, sp.f["phi"][j], weight)

        todo = [(k, f) for k, f in enumerate(scan.files) if not done[k]]
        it = todo
        if progress:
            from tqdm.auto import tqdm
            it = tqdm(todo, desc=out.name)
        dark = None
        for k, f in it:
            if dark is None or dark.path != scan.block_of(f).black:
                dark = DarkFrame.load(scan.block_of(f).black, k_offset, k_std)
            r = process_file(f, dark, pipeline, transmission)
            weight = r.phi["b_ref"] if pipeline.config.weight == "b_ref" else None
            r.phi["phi"], sign, amb, ref = tracker.align(k, r.phi["phi"], weight, ~dark.bad)
            if amb:
                warnings.warn(f"{f.wavelength} nm: sign alignment to wavelength index {ref} "
                              f"was ambiguous; check that map", stacklevel=2)
            common = dict(bad_pixels=dark.bad, exposure_ms=r.meta["exposure_ms"],
                          block=np.bytes_(f.block))
            sp.write(k, **r.phi, **common, align_sign=sign, align_ambiguous=amb,
                     align_reference=ref,
                     **{m: v for m, v in r.meta.items() if m not in ("wavelength", "exposure_ms")})
            if st is not None:
                st.write(k, **r.T, **common, n_back=r.meta.get("n_back", -1))
                st.finish_row(k)
            sp.finish_row(k)
            del r
            gc.collect()

    if layout.enabled:
        from processing.layout import align_scan

        align_scan(phi_work, layout, phi_path, t_work, t_path, compression=compression,
                   progress=progress)
        if not layout.keep_unaligned:
            for p in (phi_work, t_work):
                if p is not None:
                    p.unlink()
    return phi_path, t_path


def _is_aligned(path: Path, complete: bool = False) -> bool:
    """Whether ``path`` exists and was written by the layout alignment (and, with
    ``complete``, every wavelength is done)."""
    import h5py

    if path is None or not Path(path).exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            ok = bool(f.attrs.get("layout_aligned", False))
            return ok and (not complete or bool(f["done"][:].all()))
    except OSError:
        return False
