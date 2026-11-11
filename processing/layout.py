"""Align a wavelength scan to its GDS layout: common orientation, structure masks, glass.

The sample carries alignment markers (squares of a known size at the corners of the
patterned area). Their positions in the GDS define a coordinate frame in µm. This
module

1. reads the layout: markers, other top-level polygons ("solid") and arrays (cell
   references, e.g. metasurfaces) as named regions, :class:`Layout`;
2. finds the corner markers on a deviation map combined over all wavelengths (a
   structure can vanish at one wavelength when its phase is near 0 mod 2π) and fits
   the µm -> camera-pixel affine transform with the orientation fixed by the config,
   :func:`find_markers`, :func:`fit_frame`;
3. refines that transform per wavelength from the edges of the markers and solid
   structures (sample drift between wavelengths), :func:`register`;
4. resamples every map onto one grid in layout orientation (rows from +y to -y, text
   readable), phase as ``exp(i*phi)`` so wraps are not smeared, :func:`resample`;
5. builds masks on that grid: one integer label per structure, the glass (inside
   the marker frame, away from every structure), particles and extended
   non-uniformities (e.g. polymer) on the glass, :func:`structure_labels`,
   :class:`DefectAccumulator`;
6. optionally subtracts a low-order surface fitted on the clean glass from every
   phase map, :func:`fit_glass_surface`.

:func:`align_scan` runs 2-6 on the ``_phi.h5`` / ``_T.h5`` files written by
:func:`processing.interferometer.process_scan`, which calls it as its second pass
when ``layout.enabled`` is set.

Coordinates: layout points ``(u, v)`` are µm with ``v`` up, as in the GDS. Camera
pixels ``(x, y)`` are column/row with ``y`` down. A transform is a ``(2, 3)`` array
``A`` with ``[x, y] = A @ [u, v, 1]``.
"""

from __future__ import annotations

import itertools
import os
import warnings
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

KINDS = ("marker", "solid", "array")


# --------------------------------------------------------------------------- config

@dataclass
class PlaneConfig:
    """Surface subtracted from every aligned phase map, fitted on the clean glass.

    Attributes
    ----------
    enabled : bool, default True
    degree : int, default 1
        Total polynomial degree in ``(u, v)``: 0 piston only, 1 plane, 2 adds curvature.
    """

    enabled: bool = True
    degree: int = 1

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise ValueError("plane degree must be >= 0")


@dataclass
class DefectConfig:
    """Detection of particles and extended non-uniformities on the glass.

    Both are judged on the glass pixels only and averaged over all wavelengths.

    Attributes
    ----------
    particle_window_um : float, default 5
        A glass pixel is compared with the mean of the glass in this window.
    particle_k : float, default 5
        Particle if its deviation (phase, or relative T) is more than ``particle_k``
        robust sigmas of the glass, averaged over wavelengths.
    extended_window_um : float, default 30
        Smoothing window for extended non-uniformities.
    extended_rad : float, default 0.1
        Extended non-uniformity if the smoothed glass phase deviates from the glass
        level by more than this (rad), averaged over wavelengths.
    grow_um : float, default 1
        Both masks are dilated by this much.
    max_wavelengths : int, default 20
        Statistics use at most this many wavelengths, evenly spread over the scan
        (particles and polymer do not depend on wavelength).
    """

    particle_window_um: float = 5.0
    particle_k: float = 5.0
    extended_window_um: float = 30.0
    extended_rad: float = 0.1
    grow_um: float = 1.0
    max_wavelengths: int = 20


@dataclass
class LayoutConfig:
    """Settings for aligning a scan to its GDS layout (``layout:`` in the phase YAML).

    Attributes
    ----------
    enabled : bool, default False
        Run the alignment as the second pass of
        :func:`processing.interferometer.process_scan`.
    gds : str or None
        Layout file; ``--gds`` of ``scripts/process_interferometer.py`` overrides it.
    cell : str, default "TOP"
        Cell holding the layout.
    marker_size_um : float, default 80
        Top-level polygons of this size (±10 %) are markers; the four outermost
        define the frame and the transform.
    exclude_cells : list of str, default ["TEXT"]
        Cell references that are not structures.
    rotation : {0, 90, 180, 270}, default 0
        Rotation of the layout as seen on the camera image, degrees
        counter-clockwise on screen, applied after ``mirror``.
    mirror : bool, default True
        The camera sees the layout mirrored left-right (e.g. imaged through the
        substrate). See :func:`orientation_matrix`.
    margin_um : float or None, default 50
        The aligned grid covers the marker frame plus this margin; None covers the
        whole camera image.
    pixel_um : float or None
        Aligned grid pixel; None uses the camera pixel from the marker fit.
    guard_um : float, default 6
        Gap around every structure that is not counted as glass: edge diffraction
        halos (at least 2 grid pixels, the reach of the interpolation).
    max_shift_um : float, default 10
        Largest per-wavelength drift searched for, relative to the combined fit.
    min_match : float, default 0.3
        Normalised edge correlation below which a structure is not used for the
        per-wavelength fit.
    plane : PlaneConfig
    defects : DefectConfig
    keep_unaligned : bool, default False
        Keep the unaligned ``*_unaligned.h5`` files after alignment.
    """

    enabled: bool = False
    gds: str | None = None
    cell: str = "TOP"
    marker_size_um: float = 80.0
    exclude_cells: list = field(default_factory=lambda: ["TEXT"])
    rotation: int = 0
    mirror: bool = True
    margin_um: float | None = 50.0
    pixel_um: float | None = None
    guard_um: float = 6.0
    max_shift_um: float = 10.0
    min_match: float = 0.3
    plane: PlaneConfig = field(default_factory=PlaneConfig)
    defects: DefectConfig = field(default_factory=DefectConfig)
    keep_unaligned: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.plane, dict):
            self.plane = PlaneConfig(**self.plane)
        if isinstance(self.defects, dict):
            self.defects = DefectConfig(**self.defects)
        if self.rotation % 90 or not 0 <= self.rotation < 360:
            raise ValueError(f"rotation must be 0, 90, 180 or 270, got {self.rotation}")
        self.exclude_cells = list(self.exclude_cells)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- layout

@dataclass
class Region:
    """One structure of the layout, in µm.

    Attributes
    ----------
    name : str
        ``marker_<i>`` / ``solid_<i>`` (numbered top row first, left to right, in
        layout coordinates) or the referenced cell name for arrays.
    kind : {"marker", "solid", "array"}
    polygon : np.ndarray, shape (P, 2)
        Outline; the bounding box for arrays.
    """

    name: str
    kind: str
    polygon: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.polygon.min(0) + self.polygon.max(0))

    @property
    def size(self) -> np.ndarray:
        return np.ptp(self.polygon, axis=0)


@dataclass
class Layout:
    """Regions of a GDS cell and which markers define the frame."""

    regions: list[Region]
    corners: list[int]          # indices of the four frame markers in ``regions``
    path: str | None = None

    @classmethod
    def from_gds(cls, path: str | os.PathLike, cell: str = "TOP", marker_size_um: float = 80.0,
                 exclude_cells=("TEXT",)) -> Layout:
        """Read a GDS cell: top-level polygons become markers (by size) or solids,
        cell references become arrays (bounding box).

        Raises
        ------
        ValueError
            If the cell is missing or fewer than four markers are found.
        """
        import gdstk

        lib = gdstk.read_gds(str(path))
        cells = {c.name: c for c in lib.cells}
        if cell not in cells:
            raise ValueError(f"{path}: no cell {cell!r}; cells: {sorted(cells)}")
        top = cells[cell]
        to_um = lib.unit / 1e-6
        markers, solids, arrays = [], [], []
        for p in top.polygons:
            pts = np.asarray(p.points, float) * to_um
            size = np.ptp(pts, axis=0)
            if size.min() <= 0:
                continue
            if np.all(np.abs(size / marker_size_um - 1) < 0.1):
                markers.append(pts)
            else:
                solids.append(pts)
        exclude = set(exclude_cells)
        names: dict[str, int] = {}
        for r in top.references:
            name = r.cell if isinstance(r.cell, str) else r.cell.name
            if name in exclude:
                continue
            bb = r.bounding_box()
            if bb is None:
                continue
            (x0, y0), (x1, y1) = np.asarray(bb, float) * to_um
            names[name] = names.get(name, 0) + 1
            label = name if names[name] == 1 else f"{name}#{names[name]}"
            arrays.append((label, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])))
        if len(markers) < 4:
            raise ValueError(f"{path}: found {len(markers)} markers of {marker_size_um} µm, "
                             f"need at least 4; check marker_size_um")

        def reading_order(polys):          # top row first, left to right
            c = np.array([0.5 * (p.min(0) + p.max(0)) for p in polys]).reshape(-1, 2)
            return [polys[i] for i in np.lexsort((c[:, 0], -np.round(c[:, 1], 3)))]

        regions = [Region(f"marker_{i + 1}", "marker", p)
                   for i, p in enumerate(reading_order(markers))]
        regions += [Region(f"solid_{i + 1}", "solid", p)
                    for i, p in enumerate(reading_order(solids))]
        regions += [Region(n, "array", p) for n, p in arrays]
        mc = np.array([r.center for r in regions if r.kind == "marker"])
        mid = mc.mean(0)
        corners = []
        for sx, sy in ((-1, 1), (1, 1), (-1, -1), (1, -1)):     # TL, TR, BL, BR
            score = sx * (mc[:, 0] - mid[0]) + sy * (mc[:, 1] - mid[1])
            corners.append(int(np.argmax(score)))
        if len(set(corners)) != 4:
            raise ValueError(f"{path}: markers do not span a frame")
        return cls(regions, corners, str(path))

    @property
    def frame_um(self) -> np.ndarray:
        """``[[u0, v0], [u1, v1]]``: bounding box of the four frame markers."""
        pts = np.concatenate([self.regions[i].polygon for i in self.corners])
        return np.array([pts.min(0), pts.max(0)])

    def of_kind(self, *kinds: str) -> list[Region]:
        return [r for r in self.regions if r.kind in kinds]


# --------------------------------------------------------------------------- transforms

def orientation_matrix(rotation: int, mirror: bool) -> np.ndarray:
    """2x2 map from layout ``(u, v)`` (v up) to camera ``(x, y)`` (y down), unit scale.

    ``mirror`` flips the layout left-right (``u -> -u``); the result is then rotated
    by ``rotation`` degrees counter-clockwise as seen on screen. ``(0, False)`` shows
    the layout as drawn in the GDS.
    """
    flip = np.array([[1.0, 0.0], [0.0, -1.0]])                 # v up -> y down
    m = np.diag([-1.0, 1.0]) if mirror else np.eye(2)
    t = np.radians(rotation)
    rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])  # ccw on screen
    return np.round(rot @ flip @ m, 12)


def apply(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a ``(2, 3)`` affine to ``(..., 2)`` points."""
    pts = np.asarray(pts, float)
    return pts @ A[:, :2].T + A[:, 2]


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares ``(2, 3)`` affine with ``dst ≈ apply(A, src)``."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    M = np.c_[src, np.ones(len(src))]
    sol, *_ = np.linalg.lstsq(M, dst, rcond=None)
    return sol.T


def invert(A: np.ndarray) -> np.ndarray:
    L = np.linalg.inv(A[:, :2])
    return np.c_[L, -L @ A[:, 2]]


def pixel_size(A: np.ndarray) -> float:
    """Mean µm per camera pixel of a transform."""
    return float(1.0 / np.sqrt(abs(np.linalg.det(A[:, :2]))))


# --------------------------------------------------------------------------- small image tools

def box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over a ``(2r+1)^2`` window, edges padded with zeros."""
    if r <= 0:
        return a.copy()
    # running sums in double precision: float32 sums over a large image lose ~1e-3
    a = a.astype(np.complex128 if np.iscomplexobj(a) else np.float64)
    c = np.pad(a, ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
    return c[2 * r + 1:, 2 * r + 1:] - c[:-2 * r - 1, 2 * r + 1:] - c[2 * r + 1:, :-2 * r - 1] \
        + c[:-2 * r - 1, :-2 * r - 1]


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    return box_sum(m.astype(np.int32), r) > 0 if r > 0 else m.copy()


def masked_mean(a: np.ndarray, m: np.ndarray, r: int, min_frac: float = 0.2) -> np.ndarray:
    """Mean of ``a`` over the pixels of ``m`` in a ``(2r+1)^2`` window; NaN if too few."""
    w = box_sum(m.astype(np.float64), r)
    s = box_sum(np.where(m, a, 0), r)
    out = s / np.maximum(w, 1)
    return np.where(w >= min_frac * (2 * r + 1) ** 2, out, np.nan)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labels (0 = background)."""
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    H, W = mask.shape
    for sy, sx in zip(*np.nonzero(mask)):
        if lab[sy, sx]:
            continue
        n += 1
        lab[sy, sx] = n
        q = deque([(sy, sx)])
        while q:
            y, x = q.popleft()
            for a, b in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= a < H and 0 <= b < W and mask[a, b] and not lab[a, b]:
                    lab[a, b] = n
                    q.append((a, b))
    return lab, n


def rasterize(poly_px: np.ndarray, shape: tuple[int, int], origin=(0, 0)) -> np.ndarray:
    """Pixels of a ``shape`` image whose centres lie inside a polygon.

    Pixel ``[r, c]`` has its centre at ``(x, y) = (c, r) + origin``; ``poly_px`` is in
    the same ``(x, y)`` coordinates.
    """
    from matplotlib.path import Path as MPath

    H, W = shape
    out = np.zeros(shape, bool)
    q = np.asarray(poly_px, float) - np.asarray(origin, float)
    x0 = max(int(np.floor(q[:, 0].min())) - 1, 0)
    x1 = min(int(np.ceil(q[:, 0].max())) + 2, W)
    y0 = max(int(np.floor(q[:, 1].min())) - 1, 0)
    y1 = min(int(np.ceil(q[:, 1].max())) + 2, H)
    if x0 >= x1 or y0 >= y1:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = MPath(q).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(
        y1 - y0, x1 - x0)
    return out


def phase_edges(phi: np.ndarray) -> np.ndarray:
    """Wrap-safe gradient magnitude of a phase map (same shape, last row/col 0)."""
    z = np.exp(1j * phi)
    e = np.zeros(phi.shape, np.float32)
    gx = np.abs(np.angle(z[:, 1:] * np.conj(z[:, :-1])))
    gy = np.abs(np.angle(z[1:] * np.conj(z[:-1])))
    e[:, :-1] += np.nan_to_num(gx)
    e[:-1] += np.nan_to_num(gy)
    return e


def amplitude_edges(b: np.ndarray) -> np.ndarray:
    lb = np.log(np.maximum(np.nan_to_num(b, nan=0.0), 1e-6))
    e = np.zeros(b.shape, np.float32)
    e[:, :-1] += np.abs(np.diff(lb, axis=1))
    e[:-1] += np.abs(np.diff(lb, axis=0))
    return e


# --------------------------------------------------------------------------- markers + frame

def floor_phasor(z: np.ndarray) -> complex:
    """Dominant phase level of a map (histogram peak, refined by circular mean)."""
    p = np.angle(z[np.isfinite(z)]).ravel()
    h, e = np.histogram(p, 180, (-np.pi, np.pi))
    pk = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    near = np.abs(np.angle(np.exp(1j * (p - pk)))) < 0.4
    return np.exp(1j * np.angle(np.exp(1j * p[near]).mean()))


def deviation_map(phi: np.ndarray, detrend: int = 2) -> np.ndarray:
    """``|exp(i*phi) - glass|``: 0 on the dominant (glass) level, up to 2 on structures.

    A smooth carrier of total degree ``detrend`` is removed first
    (:func:`phase_shift.remove_carrier`, wrap-safe), so a tilted or curved glass
    still sits on one level; 0 skips it.
    """
    phi = np.nan_to_num(np.asarray(phi, np.float32))
    if detrend:
        from phase_shift import remove_carrier

        phi = np.asarray(remove_carrier(phi, degree=detrend, subsample=1, device="cpu").phi)
    z = np.exp(1j * phi)
    return np.abs(z - floor_phasor(z)).astype(np.float32)


def find_markers(dev: np.ndarray, threshold: float = 0.5, min_px: int = 30) -> np.ndarray:
    """Square, well-filled regions of a deviation map, largest first.

    Returns
    -------
    np.ndarray, shape (n, 3)
        ``x, y`` centre and side length, in the pixels of ``dev``.
    """
    lab, n = label_components(dev > threshold)
    out = []
    H, W = dev.shape
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size < min_px:
            continue
        h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
        if ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1:
            continue
        if ys.size / (h * w) > 0.8 and 0.8 < w / h < 1.25:
            out.append((xs.mean(), ys.mean(), np.sqrt(ys.size)))
    out.sort(key=lambda t: -t[2])
    return np.array(out).reshape(-1, 3)


def _match_frame(D: np.ndarray, pts: np.ndarray, sizes: np.ndarray, O: np.ndarray
                 ) -> tuple[float, float, np.ndarray] | None:
    """Best assignment of 4 of ``pts`` to the frame-marker centres ``D`` under a
    similarity ``s * O @ D + t`` (orientation fixed). Returns ``(residual, s, P)``."""
    best = None
    for idx in itertools.permutations(range(len(pts)), 4):
        idx = list(idx)
        if sizes[idx].max() > 1.3 * sizes[idx].min():
            continue
        P = pts[idx]
        Dc, Pc = D - D.mean(0), P - P.mean(0)
        OD = Dc @ O.T
        s = float(np.sum(OD * Pc) / np.sum(OD * OD))
        if s <= 0:
            continue
        r = float(np.linalg.norm(s * OD - Pc, axis=1).max())
        if best is None or r < best[0]:
            best = (r, s, P)
    return best


def fit_frame(layout: Layout, candidates: np.ndarray, rotation: int, mirror: bool,
              scale_px: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine µm -> camera px from marker candidates, orientation fixed.

    The four frame markers are matched to the four of the largest candidates (of
    similar size) that fit them best with a similarity of the configured
    orientation; every other marker of the layout found near its predicted position
    is then added to the final affine fit.

    Parameters
    ----------
    candidates : np.ndarray, shape (n, 3)
        From :func:`find_markers`, in pixels of a map subsampled by ``scale_px``
        (``map[i] = image[i * scale_px]``).

    Returns
    -------
    A : np.ndarray, shape (2, 3)
    residual_px : np.ndarray
        Fit residual of every marker used, camera pixels.
    corners_px : np.ndarray, shape (4, 2)
        Camera positions matched to the frame markers.
    """
    if len(candidates) < 4:
        raise ValueError(f"found {len(candidates)} marker candidates, need 4")
    c = candidates[:8].copy()
    c[:, :3] *= scale_px
    O = orientation_matrix(rotation, mirror)
    D = np.array([layout.regions[i].center for i in layout.corners])
    best = _match_frame(D, c[:, :2], c[:, 2], O)
    if best is None:
        raise ValueError("could not match the frame markers")
    r, s, P = best
    if r > 0.1 * s * layout.regions[layout.corners[0]].size.max():
        raise ValueError(f"frame markers do not match the layout (residual {r:.1f} px); "
                         f"check rotation/mirror and marker_size_um")
    A = fit_affine(D, P)
    src, dst = list(D), list(P)
    for i, reg in enumerate(layout.regions):
        if reg.kind != "marker" or i in layout.corners:
            continue
        pred = apply(A, reg.center)
        d = np.linalg.norm(c[:, :2] - pred, axis=1)
        j = int(np.argmin(d))
        if d[j] < 0.2 * s * reg.size.max():
            src.append(reg.center)
            dst.append(c[j, :2])
    A = fit_affine(np.array(src), np.array(dst))
    res = np.linalg.norm(apply(A, np.array(src)) - np.array(dst), axis=1)
    return A, res, P


def orientation_scores(layout: Layout, dev: np.ndarray, corners_px: np.ndarray,
                       scale_px: float = 1.0) -> dict[tuple[int, bool], float]:
    """Mean deviation inside all non-marker structures for each of the 8 orientations.

    The frame markers are symmetric, so every orientation fits them; the other
    structures tell the orientations apart. The configured one should score highest.
    """
    D = np.array([layout.regions[i].center for i in layout.corners])
    regs = layout.of_kind("solid", "array")
    scores: dict[tuple[int, bool], float] = {}
    if not regs:
        return scores
    for rot in (0, 90, 180, 270):
        for mir in (False, True):
            m = _match_frame(D, corners_px, np.ones(4), orientation_matrix(rot, mir))
            if m is None:
                continue
            Ao = fit_affine(D, m[2])
            vals = []
            for reg in regs:
                mask = rasterize(apply(Ao, reg.polygon) / scale_px, dev.shape)
                if mask.any():
                    vals.append(float(dev[mask].mean()))
            scores[(rot, mir)] = float(np.mean(vals)) if vals else np.nan
    return scores


# --------------------------------------------------------------------------- per-wavelength registration

def _ncc_shift(meas: np.ndarray, tmpl: np.ndarray, max_shift: int) -> tuple[float, float, float]:
    """Shift of ``meas`` relative to ``tmpl`` (same shape) and normalised peak."""
    a = meas - meas.mean()
    b = tmpl - tmpl.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0, 0.0, 0.0
    sh = (a.shape[0] + max_shift, a.shape[1] + max_shift)
    c = np.fft.irfft2(np.fft.rfft2(a, sh) * np.conj(np.fft.rfft2(b, sh)), sh) / (na * nb)
    ys = np.r_[0:max_shift + 1, sh[0] - max_shift:sh[0]]
    xs = np.r_[0:max_shift + 1, sh[1] - max_shift:sh[1]]
    sub = c[np.ix_(ys, xs)]
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    peak = float(sub[i, j])

    def para(v0, v1, v2):
        d = v0 - 2 * v1 + v2
        return 0.0 if d >= 0 else 0.5 * (v0 - v2) / d
    dy = ys[i] if ys[i] <= max_shift else ys[i] - sh[0]
    dx = xs[j] if xs[j] <= max_shift else xs[j] - sh[1]
    yi, xj = ys[i], xs[j]
    dy += para(c[(yi - 1) % sh[0], xj], c[yi, xj], c[(yi + 1) % sh[0], xj])
    dx += para(c[yi, (xj - 1) % sh[1]], c[yi, xj], c[yi, (xj + 1) % sh[1]])
    return float(dx), float(dy), peak


def register(phi: np.ndarray, b: np.ndarray | None, layout: Layout, A0: np.ndarray,
             max_shift_px: int, min_match: float = 0.3) -> tuple[np.ndarray | None, np.ndarray]:
    """Refine ``A0`` for one wavelength from the edges of markers and solid structures.

    Each structure's outline, rendered with ``A0``, is correlated with the measured
    edge map (phase, plus fringe amplitude ``b`` when given) in a window around it.
    Structures matching better than ``min_match`` give a displacement each; an
    affine (>= 4 structures spread over the frame), a shift (>= 1), or None (none).

    Returns
    -------
    A : np.ndarray, shape (2, 3), or None
    matches : np.ndarray, shape (R, 3)
        ``dx, dy, peak`` per marker/solid structure (camera px).
    """
    def normalised(e):
        pos = e[e > 0]
        return e / np.median(pos) if pos.size > 0.01 * e.size else np.zeros_like(e)

    E = normalised(phase_edges(phi))
    if b is not None:
        E = E + normalised(amplitude_edges(b))
    E = np.minimum(E, np.percentile(E, 99.5))
    H, W = phi.shape
    regs = layout.of_kind("marker", "solid")
    src, dst, matches = [], [], []
    for reg in regs:
        Q = apply(A0, reg.polygon)
        x0, y0 = np.floor(Q.min(0)).astype(int) - max_shift_px - 3
        x1, y1 = np.ceil(Q.max(0)).astype(int) + max_shift_px + 4
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            matches.append((np.nan, np.nan, 0.0))
            continue
        m = rasterize(Q, (y1 - y0, x1 - x0), origin=(x0, y0)).astype(np.float32)
        tm = box_sum(m, 1) / 9.0
        te = np.zeros_like(tm)
        te[:, :-1] += np.abs(np.diff(tm, axis=1))
        te[:-1] += np.abs(np.diff(tm, axis=0))
        dx, dy, pk = _ncc_shift(E[y0:y1, x0:x1], te, max_shift_px)
        matches.append((dx, dy, pk))
        if pk >= min_match:
            src.append(reg.center)
            dst.append(apply(A0, reg.center) + (dx, dy))
    matches = np.array(matches)
    if not src:
        return None, matches
    src, dst = np.array(src), np.array(dst)
    span = np.ptp(layout.frame_um, axis=0)
    # robust: start from the median shift, keep structures that agree with the model
    # to within ``tol`` px, refit an affine on those when they span the frame
    tol = 1.5
    A = np.c_[A0[:, :2], A0[:, 2] + np.median(dst - apply(A0, src), axis=0)]
    for _ in range(3):
        inl = np.linalg.norm(apply(A, src) - dst, axis=1) < tol
        if inl.sum() < 4 or not np.all(np.ptp(src[inl], axis=0) > 0.5 * span):
            break
        Af = fit_affine(src[inl], dst[inl])
        # an affine straying from A0 more than the drift itself would explain is rejected
        if np.abs(Af[:, :2] - A0[:, :2]).max() * span.max() >= 2 * max_shift_px:
            break
        A = Af
    inl = np.linalg.norm(apply(A, src) - dst, axis=1) < tol
    if inl.sum() < max(2, 0.3 * len(src)):
        return None, matches                     # structures disagree: not trustworthy
    return A, matches


def register_to(phi: np.ndarray, phi_ref: np.ndarray, window: tuple[slice, slice],
                A_ref: np.ndarray, max_shift_px: int, min_peak: float = 0.15
                ) -> tuple[np.ndarray | None, float]:
    """Shift-only registration of one map against an already registered one.

    The phase-edge maps of both are correlated over ``window`` (the marker frame);
    the fallback for wavelengths whose individual structures disagree.

    Returns
    -------
    A : np.ndarray, shape (2, 3), or None
        ``A_ref`` shifted by the measured displacement; None if the correlation peak
        is below ``min_peak``.
    peak : float
    """
    dx, dy, pk = _ncc_shift(phase_edges(phi[window]), phase_edges(phi_ref[window]), max_shift_px)
    if pk < min_peak:
        return None, pk
    return np.c_[A_ref[:, :2], A_ref[:, 2] + (dx, dy)], pk


def fill_transforms(transforms: list, wavelength: np.ndarray, A0: np.ndarray) -> np.ndarray:
    """Replace missing per-wavelength transforms by interpolation in wavelength
    (nearest at the ends), or ``A0`` when none was found."""
    ok = np.array([t is not None for t in transforms])
    out = np.zeros((len(transforms), 2, 3))
    if not ok.any():
        out[:] = A0
        return out
    good = np.array([t for t in transforms if t is not None])
    wg = np.asarray(wavelength, float)[ok]
    for i in range(2):
        for j in range(3):
            out[:, i, j] = np.interp(wavelength, wg, good[:, i, j])
    return out


# --------------------------------------------------------------------------- grid + resampling

@dataclass
class Grid:
    """Aligned image grid in layout µm: pixel centres ``x_um`` (columns, increasing)
    and ``y_um`` (rows, decreasing: layout up is image up)."""

    x_um: np.ndarray
    y_um: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.y_um), len(self.x_um)

    @property
    def pixel_um(self) -> float:
        return float(self.x_um[1] - self.x_um[0])

    @classmethod
    def around(cls, box_um: np.ndarray, pixel_um: float) -> Grid:
        (u0, v0), (u1, v1) = box_um
        nx = int(np.ceil((u1 - u0) / pixel_um))
        ny = int(np.ceil((v1 - v0) / pixel_um))
        x = u0 + (np.arange(nx) + 0.5) * pixel_um
        y = v1 - (np.arange(ny) + 0.5) * pixel_um
        return cls(x, y)

    def to_px(self, pts_um: np.ndarray) -> np.ndarray:
        """Layout µm -> fractional grid pixel ``(col, row)``."""
        pts = np.asarray(pts_um, float)
        return np.stack([(pts[..., 0] - self.x_um[0]) / self.pixel_um,
                         (self.y_um[0] - pts[..., 1]) / self.pixel_um], axis=-1)


def make_grid(layout: Layout, A: np.ndarray, image_shape: tuple[int, int],
              margin_um: float | None, pixel_um: float | None) -> Grid:
    p = pixel_um or round(pixel_size(A), 4)
    if margin_um is None:
        H, W = image_shape
        corners = apply(invert(A), np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]]))
        box = np.array([corners.min(0), corners.max(0)])
    else:
        box = layout.frame_um + np.array([[-margin_um, -margin_um], [margin_um, margin_um]])
    return Grid.around(box, p)


def _sample_coords(A: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(grid.x_um, grid.y_um)
    x = A[0, 0] * u + A[0, 1] * v + A[0, 2]
    y = A[1, 0] * u + A[1, 1] * v + A[1, 2]
    return x, y


def resample(img: np.ndarray, A: np.ndarray, grid: Grid, kind: str = "linear",
             bad: np.ndarray | None = None) -> np.ndarray:
    """Bilinear resampling of a camera image onto ``grid``.

    Parameters
    ----------
    kind : {"linear", "phase", "mask"}
        ``"phase"`` interpolates ``exp(i*img)`` and returns its angle (wraps are not
        smeared); ``"mask"`` returns True where any contributing pixel is True.
    bad : array of bool, optional
        Camera pixels to treat as missing: grid pixels that depend on them are NaN.

    Grid pixels outside the camera image are NaN (False for masks).
    """
    H, W = img.shape
    x, y = _sample_coords(A, grid)
    inside = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)
    x0 = np.clip(np.floor(x).astype(np.int64), 0, W - 2)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, H - 2)
    fx = np.clip(x - x0, 0, 1).astype(np.float32)
    fy = np.clip(y - y0, 0, 1).astype(np.float32)

    def interp(a):
        return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
                + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)

    if kind == "mask":
        return (interp(img.astype(np.float32)) > 0) & inside
    if kind == "phase":
        out = np.angle(interp(np.exp(1j * img.astype(np.float32)).astype(np.complex64)))
    elif kind == "linear":
        out = interp(img.astype(np.float32))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    out = out.astype(np.float32)
    invalid = ~inside | ~np.isfinite(out)
    if bad is not None:
        invalid |= interp(bad.astype(np.float32)) > 0
    out[invalid] = np.nan
    return out


# --------------------------------------------------------------------------- masks

def structure_labels(layout: Layout, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Label image on the grid and the marker-frame mask.

    Returns
    -------
    labels : np.ndarray of int16
        ``i + 1`` on region ``layout.regions[i]``, 0 elsewhere.
    frame : np.ndarray of bool
        Inside the bounding box of the four frame markers.
    """
    labels = np.zeros(grid.shape, np.int16)
    for i, reg in enumerate(layout.regions):
        m = rasterize(grid.to_px(reg.polygon), grid.shape)
        labels[m & (labels == 0)] = i + 1
    (u0, v0), (u1, v1) = layout.frame_um
    u, v = np.meshgrid(grid.x_um, grid.y_um)
    frame = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    return labels, frame


def glass_mask(labels: np.ndarray, frame: np.ndarray, guard_px: int) -> np.ndarray:
    """Inside the frame and at least ``guard_px`` away from every structure."""
    return frame & ~dilate(labels > 0, guard_px)


def _poly_basis(u: np.ndarray, v: np.ndarray, degree: int) -> np.ndarray:
    return np.stack([u ** i * v ** j for i in range(degree + 1) for j in range(degree + 1 - i)], -1)


def fit_glass_surface(phi: np.ndarray, mask: np.ndarray, grid: Grid, degree: int,
                      iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Fit a degree-``degree`` polynomial in ``(u, v)`` to the wrapped phase on ``mask``.

    The fit runs on ``angle(exp(i*(phi - fit)))``, refined ``iters`` times, so the
    glass phase may sit anywhere on the circle (only its variation over the frame
    must stay well below π).

    Returns
    -------
    surface : np.ndarray, shape grid.shape
        Fitted phase, rad (unwrapped polynomial), defined everywhere.
    coef : np.ndarray
        Coefficients for :func:`_poly_basis` on ``u, v`` scaled by the frame half-size.
    """
    m = mask & np.isfinite(phi)
    u, v = np.meshgrid(grid.x_um, grid.y_um)
    cu, cv = grid.x_um.mean(), grid.y_um.mean()
    su = max(np.ptp(grid.x_um), np.ptp(grid.y_um)) / 2
    U, V = (u - cu) / su, (v - cv) / su
    B = _poly_basis(U[m], V[m], degree)
    z = np.exp(1j * phi[m])
    coef = np.zeros(B.shape[1])
    coef[0] = np.angle(z.mean())
    for _ in range(iters):
        q = np.angle(z * np.exp(-1j * (B @ coef)))
        dc, *_ = np.linalg.lstsq(B, q, rcond=None)
        coef = coef + dc
    surface = _poly_basis(U, V, degree) @ coef
    return surface.astype(np.float32), coef


class DefectAccumulator:
    """Average glass deviations over wavelengths into particle / extended masks.

    Feed every aligned wavelength with :meth:`add`, then call :meth:`result`.
    """

    def __init__(self, glass: np.ndarray, grid: Grid, cfg: DefectConfig) -> None:
        self.glass = glass
        self.grid = grid
        self.cfg = cfg
        p = grid.pixel_um
        self.rp = max(1, int(round(cfg.particle_window_um / p / 2)))
        self.re = max(1, int(round(cfg.extended_window_um / p / 2)))
        self.part = np.zeros(glass.shape)
        self.ext = np.zeros(glass.shape)
        self.n = np.zeros(glass.shape)

    def add(self, phi: np.ndarray, T: np.ndarray | None = None) -> None:
        g = self.glass & np.isfinite(phi)
        if T is not None:
            g &= np.isfinite(T)
        if g.sum() < 100:
            return
        surf, _ = fit_glass_surface(phi, g, self.grid, 1)
        z = np.exp(1j * (np.nan_to_num(phi) - surf))
        # particles: deviation from the local glass mean, in robust sigmas
        loc = masked_mean(z, g, self.rp)
        d = np.abs(np.angle(z * np.conj(loc)))
        s = max(1.4826 * np.nanmedian(d[g]), 1e-3)          # rad; floor for noise-free data
        score = d / s
        if T is not None:
            tl = masked_mean(np.nan_to_num(T), g, self.rp)
            dt = np.abs(np.nan_to_num(T) / tl - 1)
            st = max(1.4826 * np.nanmedian(dt[g]), 1e-4)    # relative T
            score = np.maximum(score, dt / st)
        # extended: smoothed glass phase vs the glass level (0 after the plane)
        sm = masked_mean(z, g, self.re)
        de = np.abs(np.angle(sm))
        ok = g & np.isfinite(score) & np.isfinite(de)
        self.part[ok] += score[ok]
        self.ext[ok] += de[ok]
        self.n[ok] += 1

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        """``particles, extended`` boolean masks (glass pixels only)."""
        n = np.maximum(self.n, 1)
        grow = max(0, int(round(self.cfg.grow_um / self.grid.pixel_um)))
        part = self.glass & (self.n > 0) & (self.part / n > self.cfg.particle_k)
        ext = self.glass & (self.n > 0) & (self.ext / n > self.cfg.extended_rad)
        part = dilate(part, grow) & self.glass
        ext = dilate(ext, grow) & self.glass & ~part
        return part, ext


# --------------------------------------------------------------------------- scan alignment

PHASE_MAPS = {"phi"}


def _map_names(f, shape) -> list[str]:
    return [k for k, d in f.items() if getattr(d, "ndim", 0) == 3 and d.shape[1:] == shape
            and k != "bad_pixels"]


def align_scan(phi_in: str | Path, cfg: LayoutConfig, phi_out: str | Path,
               t_in: str | Path | None = None, t_out: str | Path | None = None,
               compression: str | None = "gzip", progress: bool = True) -> dict:
    """Align the maps of a scan to the layout and write them with masks.

    Every ``(Wvl, Y, X)`` dataset of ``phi_in`` (and ``t_in``) is resampled onto the
    aligned grid (``phi`` as a phase); ``bad_pixels`` becomes the missing-data mask
    on the grid; all other datasets and attributes are copied. With ``cfg.plane``
    enabled, the surface fitted on the clean glass is subtracted from ``phi``.
    Both outputs get a ``layout`` group:

    - ``labels`` (int16): ``i`` on structure ``names[i-1]``, 0 elsewhere;
    - ``names``, ``kinds``, ``polygon_um``: the regions;
    - ``frame``, ``glass``, ``particles``, ``extended`` (bool masks);
    - ``x_um``, ``y_um``: grid pixel centres;
    - ``transform`` (Wvl, 2, 3): layout µm -> camera px per wavelength,
      ``registered`` (Wvl,) whether it was measured (else interpolated),
      ``registration`` (Wvl,) how: 1 from the structures, 2 shifted against the
      nearest wavelength registered from the structures, 0 interpolated,
      ``matches`` (Wvl, R, 3): per-structure ``dx, dy, peak``;
    - ``glass_fit`` (Wvl, C): surface coefficients removed from ``phi``.

    Returns
    -------
    dict
        Summary: ``transform0``, ``marker_residual_px``, ``pixel_um``, ``shape``.
    """
    import h5py
    import yaml

    from processing.interferometer import ScanStore

    if not cfg.gds:
        raise ValueError("layout.gds is not set")
    layout = Layout.from_gds(cfg.gds, cfg.cell, cfg.marker_size_um, cfg.exclude_cells)
    fi = h5py.File(phi_in, "r")
    ft = h5py.File(t_in, "r") if t_in is not None else None
    try:
        wvl = fi["wavelength"][:]
        K, H, W = fi["phi"].shape
        it = range(K)
        if progress:
            from tqdm.auto import tqdm

        import time

        timing = {}
        t0 = time.perf_counter()

        def lap(name):
            nonlocal t0
            now = time.perf_counter()
            timing[name] = round(now - t0, 1)
            t0 = now

        # 1. combined deviation map (4x downsampled) -> markers -> base transform
        ds = 4
        dev = np.zeros((len(range(0, H, ds)), len(range(0, W, ds))), np.float32)
        for k in (tqdm(it, desc="markers") if progress else it):
            dev += deviation_map(fi["phi"][k, ::ds, ::ds]) / K
        cand = find_markers(dev)
        A0, res, corners_px = fit_frame(layout, cand, cfg.rotation, cfg.mirror, scale_px=ds)
        if res.max() > 3:
            warnings.warn(f"marker fit residual up to {res.max():.1f} px", stacklevel=2)
        scores = orientation_scores(layout, dev, corners_px, scale_px=ds)
        if scores:
            mine = scores[(cfg.rotation, cfg.mirror)]
            other = max((v, k_) for k_, v in scores.items() if k_ != (cfg.rotation, cfg.mirror))
            if other[0] > 1.15 * mine:
                warnings.warn(f"layout orientation rotation={other[1][0]}, mirror={other[1][1]} "
                              f"matches the structures better than the configured "
                              f"rotation={cfg.rotation}, mirror={cfg.mirror}", stacklevel=2)

        lap("markers")

        # 2. per-wavelength transforms
        p_cam = pixel_size(A0)
        max_shift = int(np.ceil(cfg.max_shift_um / p_cam))
        trans, matches = [], []
        for k in (tqdm(it, desc="register") if progress else it):
            b = fi["b_sample"][k] if "b_sample" in fi else None
            A, mt = register(fi["phi"][k], b, layout, A0, max_shift, cfg.min_match)
            trans.append(A)
            matches.append(mt)
        # 0 interpolated, 1 from the structures, 2 shifted against the nearest such wavelength
        registration = np.array([int(t is not None) for t in trans])
        if not registration.any():
            warnings.warn("no wavelength could be registered; using the combined fit for all",
                          stacklevel=2)
        else:
            (u0, v0), (u1, v1) = layout.frame_um
            frame_px = apply(A0, np.array([[u0, v0], [u1, v0], [u0, v1], [u1, v1]]))
            x0, y0 = np.maximum(np.floor(frame_px.min(0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(frame_px.max(0)).astype(int), (W, H))
            good = np.flatnonzero(registration == 1)
            for k in np.flatnonzero(registration == 0):
                j = int(good[np.argmin(np.abs(wvl[good] - wvl[k]))])
                A, rel = register_to(fi["phi"][k], fi["phi"][j], (slice(y0, y1), slice(x0, x1)),
                                     trans[j], max_shift)
                if A is not None:
                    trans[k], registration[k] = A, 2
        registered = registration > 0
        T_all = fill_transforms(trans, wvl, A0)
        lap("register")

        # 3. grid, structure masks
        grid = make_grid(layout, A0, (H, W), cfg.margin_um, cfg.pixel_um)
        labels, frame = structure_labels(layout, grid)
        guard = max(2, int(round(cfg.guard_um / grid.pixel_um)))       # >= interpolation blur
        glass = glass_mask(labels, frame, guard)

        def bad_k(f, k):
            return f["bad_pixels"][k] if "bad_pixels" in f else None

        # 4. defects (needs every wavelength) — resample phi (and T) once
        acc = DefectAccumulator(glass, grid, cfg.defects)
        n_def = min(K, max(1, cfg.defects.max_wavelengths))
        ks = np.unique(np.round(np.linspace(0, K - 1, n_def)).astype(int))
        for k in (tqdm(ks, desc="defects") if progress else ks):
            ph = resample(fi["phi"][k], T_all[k], grid, "phase", bad_k(fi, k))
            Tk = resample(ft["T"][k], T_all[k], grid, "linear", bad_k(ft, k)) \
                if ft is not None and "T" in ft else None
            acc.add(ph, Tk)
        particles, extended = acc.result()
        clean = glass & ~particles & ~extended
        lap("defects")

        # 5. write aligned files
        layout_ds = dict(
            labels=labels, frame=frame, glass=glass, particles=particles, extended=extended,
            x_um=grid.x_um, y_um=grid.y_um, transform=T_all, registered=registered,
            registration=registration,
            matches=np.array(matches),
            names=np.array([r.name for r in layout.regions], dtype="S"),
            kinds=np.array([r.kind for r in layout.regions], dtype="S"),
            polygon_um=_pad_polys([r.polygon for r in layout.regions]),
        )
        attrs = dict(layout_aligned=True, layout_gds=str(cfg.gds),
                     layout_config=yaml.safe_dump(cfg.to_dict(), sort_keys=False),
                     layout_pixel_um=grid.pixel_um,
                     layout_convention="rows from +y to -y, columns from -x to +x (layout µm)")
        jobs = [(fi, phi_out, True)] + ([(ft, t_out, False)] if ft is not None else [])
        for src, dst, is_phi in jobs:
            maps = _map_names(src, (H, W))
            with ScanStore(dst, wvl, dict(dict(src.attrs), **attrs), overwrite=True,
                           compression=compression) as out:
                for name, d in src.items():
                    if name in maps or name in ("bad_pixels", "wavelength", "done"):
                        continue
                    out.f.create_dataset(name, data=d[()])
                g = out.f.create_group("layout")
                for name, v in layout_ds.items():
                    kw = dict(compression="gzip", compression_opts=1) if np.ndim(v) >= 2 else {}
                    g.create_dataset(name, data=v, **kw)
                fits = []
                for k in (tqdm(it, desc=Path(dst).name) if progress else it):
                    bad = bad_k(src, k)
                    row = {}
                    for name in maps:
                        kind = "phase" if name in PHASE_MAPS else "linear"
                        row[name] = resample(src[name][k], T_all[k], grid, kind, bad)
                    if is_phi and cfg.plane.enabled and "phi" in row:
                        surf, coef = fit_glass_surface(row["phi"], clean, grid, cfg.plane.degree)
                        row["phi"] = np.angle(np.exp(1j * (row["phi"] - surf))).astype(np.float32)
                        fits.append(coef)
                    missing = ~np.isfinite(row[maps[0]]) if maps else np.zeros(grid.shape, bool)
                    out.write(k, **row, bad_pixels=missing)
                    out.finish_row(k)
                lap(f"write {Path(dst).name}")
                if fits:
                    g.create_dataset("glass_fit", data=np.array(fits))
                    g.attrs["glass_fit_basis"] = (f"sum c * U^i V^j for i+j <= {cfg.plane.degree} "
                                                  f"(i outer); U, V = (x_um - mean, y_um - mean) / "
                                                  f"max half-extent of the grid")
        return dict(transform0=A0, marker_residual_px=res, pixel_um=grid.pixel_um,
                    shape=grid.shape, registered=registered, registration=registration,
                    orientation_scores=scores, timing=timing)
    finally:
        fi.close()
        if ft is not None:
            ft.close()


def _pad_polys(polys: list[np.ndarray]) -> np.ndarray:
    n = max(len(p) for p in polys)
    out = np.full((len(polys), n, 2), np.nan)
    for i, p in enumerate(polys):
        out[i, :len(p)] = p
    return out


def region_mask(f, name: str) -> np.ndarray:
    """Boolean mask of one structure from an aligned file's ``layout`` group."""
    names = [n.decode() for n in f["layout/names"][:]]
    return f["layout/labels"][:] == names.index(name) + 1
