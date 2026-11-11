"""Synthetic tests for processing.layout (GDS alignment, masks, glass plane)."""

import gdstk
import h5py
import numpy as np
import pytest
from phase_shift import PhaseConfig

from processing import layout as L
from processing.interferometer import ScanStore, process_scan, scan_folder
from processing.phase import PhasePipeline, PipelineConfig

H, W = 300, 320
SCALE = 0.45                       # camera px per µm
WVL = np.array([500.0, 550.0, 600.0])
DRIFT = np.array([[0.0, 0.0], [2.6, -1.4], [-3.2, 2.1]])   # px, per wavelength
PARTICLES = [(-160.0, 60.0), (40.0, -170.0)]               # µm, on the glass
BLOB = (170.0, -30.0, 30.0)                                # µm centre + radius, "polymer"


def _square(cx, cy, s):
    return gdstk.rectangle((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2))


def write_gds(path):
    lib = gdstk.Library()
    top = lib.new_cell("TOP")
    for x in (-250, 250):
        for y in (-250, 250):
            top.add(_square(x, y, 80))
    top.add(_square(0, 0, 80))                                   # centre marker
    for c in ((-125, 125), (125, 125), (-125, -125), (125, -125), (125, -225)):
        top.add(_square(*c, 50))                                  # asymmetric: no (-125, -225)
    pillar = lib.new_cell("pillars_100")
    pillar.add(_square(0, 0, 0.1))
    top.add(gdstk.Reference(pillar, (-149.85, -24.85), columns=166, rows=166, spacing=(0.3, 0.3)))
    text = lib.new_cell("TEXT")
    text.add(*gdstk.text("AB", 20, (-50, -340)))
    top.add(gdstk.Reference(text))
    lib.write_gds(str(path))
    return path


def true_transform(k, rotation=0, mirror=True):
    A = np.zeros((2, 3))
    t = np.radians(0.25)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    A[:, :2] = SCALE * R @ L.orientation_matrix(rotation, mirror)
    A[:, 2] = (W / 2 + 4, H / 2 - 3) + DRIFT[k]
    return A


def phase_of(kind, wvl):
    return {"marker": 1.8, "solid": 1.8, "array": 0.7}[kind] * 500 / wvl


def render(layout, k):
    """Camera maps of wavelength k: (phi, T) with glass tilt, blob, particles, noise."""
    rng = np.random.default_rng(k)
    A = true_transform(k)
    yy, xx = np.mgrid[:H, :W]
    uv = L.apply(L.invert(A), np.stack([xx, yy], -1))
    u, v = uv[..., 0], uv[..., 1]
    phi = 0.4 + 0.002 * u - 0.001 * v                      # glass tilt (removed by the plane)
    T = np.full((H, W), 0.9)
    phi = phi + 0.35 * ((u - BLOB[0]) ** 2 + (v - BLOB[1]) ** 2 < BLOB[2] ** 2)
    for (pu, pv) in PARTICLES:
        spot = (u - pu) ** 2 + (v - pv) ** 2 < 4.5 ** 2
        phi = phi + 1.0 * spot
        T = np.where(spot, 0.5, T)
    for reg in layout.regions:
        m = L.rasterize(L.apply(A, reg.polygon), (H, W))
        phi = phi + phase_of(reg.kind, WVL[k]) * m
    phi = np.angle(np.exp(1j * (phi + rng.normal(0, 0.02, (H, W)))))
    return phi.astype(np.float32), T.astype(np.float32)


@pytest.fixture(scope="module")
def gds(tmp_path_factory):
    return write_gds(tmp_path_factory.mktemp("gds") / "layout.gds")


@pytest.fixture(scope="module")
def layout(gds):
    return L.Layout.from_gds(gds)


def write_scan(tmp_path, layout):
    phi_p, t_p = tmp_path / "s_phi_unaligned.h5", tmp_path / "s_T_unaligned.h5"
    with ScanStore(phi_p, WVL, {"phi_convention": "test"}) as sp, ScanStore(t_p, WVL) as st:
        for k in range(len(WVL)):
            phi, T = render(layout, k)
            bad = np.zeros((H, W), bool)
            bad[40, 50] = True
            sp.write(k, phi=phi, phi_error=np.full((H, W), 0.02, np.float32),
                     b_sample=np.full((H, W), 100, np.float32), bad_pixels=bad,
                     align_sign=1)
            st.write(k, T=T, T_std=np.full((H, W), 0.01, np.float32), bad_pixels=bad)
            sp.finish_row(k)
            st.finish_row(k)
    return phi_p, t_p


# --------------------------------------------------------------------------- unit pieces

def test_layout_from_gds(layout):
    kinds = [r.kind for r in layout.regions]
    assert kinds.count("marker") == 5 and kinds.count("solid") == 5 and kinds.count("array") == 1
    names = [r.name for r in layout.regions]
    assert "pillars_100" in names and "TEXT" not in names
    corners = [layout.regions[i].center for i in layout.corners]
    np.testing.assert_allclose(corners, [[-250, 250], [250, 250], [-250, -250], [250, -250]])
    np.testing.assert_allclose(layout.frame_um, [[-290, -290], [290, 290]])
    arr = layout.regions[names.index("pillars_100")]
    np.testing.assert_allclose(arr.size, 165 * 0.3 + 0.1, atol=1e-6)


def test_orientation_matrix():
    np.testing.assert_allclose(L.orientation_matrix(0, False), [[1, 0], [0, -1]])   # as drawn
    np.testing.assert_allclose(L.orientation_matrix(0, True), [[-1, 0], [0, -1]])   # mirrored
    # 90 deg counter-clockwise on screen: layout +u (right) appears up (-y)
    np.testing.assert_allclose(L.orientation_matrix(90, False) @ [1, 0], [0, -1], atol=1e-12)


def test_resample_phase_does_not_smear_wraps():
    y, x = np.mgrid[:40, :40]
    phi = np.angle(np.exp(1j * 0.4 * x)).astype(np.float32)      # many wraps
    A = np.array([[1.0, 0, 0.5], [0, -1.0, 39.0]])               # µm == px, half-pixel shift
    grid = L.Grid(np.arange(0, 38.0), np.arange(38.0, 0, -1))
    out = L.resample(phi, A, grid, "phase")
    truth = np.angle(np.exp(1j * 0.4 * (grid.x_um + 0.5)))
    d = np.angle(np.exp(1j * (out - truth[None, :])))
    assert np.nanmax(np.abs(d)) < 1e-3


# --------------------------------------------------------------------------- full alignment

@pytest.fixture(scope="module")
def aligned(tmp_path_factory, gds, layout):
    tmp = tmp_path_factory.mktemp("align")
    phi_in, t_in = write_scan(tmp, layout)
    cfg = L.LayoutConfig(enabled=True, gds=str(gds), margin_um=20, guard_um=2,
                         defects=L.DefectConfig(extended_rad=0.15))
    summary = L.align_scan(phi_in, cfg, tmp / "s_phi.h5", t_in, tmp / "s_T.h5", progress=False)
    return tmp, summary


def test_transforms_recovered(aligned):
    tmp, summary = aligned
    assert summary["marker_residual_px"].max() < 1.0
    with h5py.File(tmp / "s_phi.h5") as f:
        T_all = f["layout/transform"][:]
        assert f["layout/registered"][:].all()
    pts = np.array([[-290, -290], [290, 290], [0, 0], [290, -290]])
    for k in range(len(WVL)):
        err = np.linalg.norm(L.apply(T_all[k], pts) - L.apply(true_transform(k), pts), axis=1)
        assert err.max() < 0.5, (k, err)


def test_aligned_maps_and_masks(aligned, layout):
    tmp, summary = aligned
    with h5py.File(tmp / "s_phi.h5") as f, h5py.File(tmp / "s_T.h5") as g:
        assert f.attrs["layout_aligned"] and f["done"][:].all()
        assert f["phi"].shape[1:] == f["layout/labels"].shape
        assert "align_sign" in f and f.attrs["phi_convention"] == "test"     # copied over
        lab = f["layout/labels"][:]
        names = [n.decode() for n in f["layout/names"][:]]
        glass = f["layout/glass"][:]
        part, ext = f["layout/particles"][:], f["layout/extended"][:]
        x_um, y_um = f["layout/x_um"][:], f["layout/y_um"][:]
        u, v = np.meshgrid(x_um, y_um)
        # text is excluded, every region labelled, layout orientation kept (rows: +y first)
        assert "TEXT" not in names and lab.max() == len(layout.regions)
        assert y_um[0] > y_um[-1] and x_um[0] < x_um[-1]
        sol = L.region_mask(f, "solid_2")                      # solid at (+125, +125)
        assert np.all(u[sol] > 90) and np.all(v[sol] > 90)
        clean = glass & ~part & ~ext
        for k, w in enumerate(WVL):
            phi = f["phi"][k]
            # glass after the plane: ~0
            g_rms = np.sqrt(np.nanmean(phi[clean] ** 2))
            assert g_rms < 0.04, (w, g_rms)
            # every structure: its phase relative to the glass
            for i, reg in enumerate(layout.regions):
                m = L.dilate(~(lab == i + 1), 2)
                core = (lab == i + 1) & ~m
                val = np.angle(np.nanmean(np.exp(1j * phi[core])))
                assert abs(np.angle(np.exp(1j * (val - phase_of(reg.kind, w))))) < 0.05, (w, reg.name)
            np.testing.assert_allclose(np.nanmedian(g["T"][k][clean]), 0.9, atol=1e-3)
        # particles and the blob are found on the glass, few false alarms
        for (pu, pv) in PARTICLES:
            near = (u - pu) ** 2 + (v - pv) ** 2 < 2 ** 2
            assert part[near].all()
        blob = (u - BLOB[0]) ** 2 + (v - BLOB[1]) ** 2 < (BLOB[2] - 8) ** 2
        assert (ext | part)[blob & glass].mean() > 0.9
        far = glass & ((u - BLOB[0]) ** 2 + (v - BLOB[1]) ** 2 > (BLOB[2] + 25) ** 2)
        for (pu, pv) in PARTICLES:
            far &= (u - pu) ** 2 + (v - pv) ** 2 > 10 ** 2
        assert (part | ext)[far].mean() < 0.01
        # the bad camera pixel became missing data on the grid
        assert f["bad_pixels"][0].any() and np.isnan(f["phi"][0][f["bad_pixels"][0]]).all()


def test_register_to_neighbour(layout):
    """Fallback registration: shift between two wavelengths from whole-frame edges."""
    A1 = true_transform(1)
    phi1, _ = render(layout, 1)
    phi2, _ = render(layout, 2)
    win = (slice(20, H - 20), slice(20, W - 20))
    A, pk = L.register_to(phi2, phi1, win, A1, max_shift_px=8)
    assert pk > 0.15
    # synthetic edges are rendered on whole pixels, which limits sub-pixel accuracy
    np.testing.assert_allclose(A[:, 2], true_transform(2)[:, 2], atol=0.5)


def test_wrong_orientation_warns(tmp_path, gds, layout):
    phi_in, _ = write_scan(tmp_path, layout)
    cfg = L.LayoutConfig(enabled=True, gds=str(gds), rotation=180, mirror=True)
    with pytest.warns(UserWarning, match="orientation"):
        L.align_scan(phi_in, cfg, tmp_path / "o_phi.h5", progress=False)


# --------------------------------------------------------------------------- through process_scan

def _write_raw_folder(folder, layout, wavelengths=(500, 550)):
    N = 8
    steps = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.1 * np.sin(np.arange(N))
    yy, xx = np.mgrid[:H, :W]
    carrier = 0.05 * xx + 0.03 * yy
    black = np.full((H, W), 9.0, np.float32)
    meta = dict(n=N, n_back=4, gain=0, repeat=1, mono_check_peak_nm=549.0)
    np.savez(folder / "e1_glass_500_550_black_20260101_000000.npz", black=black,
             black_std=np.full((H, W), 1.3, np.float32) + 0.01 * np.sin(xx + yy).astype(np.float32),
             exposure_ms=100.0, **meta)
    for w in wavelengths:
        k = int(np.argmin(np.abs(WVL - w)))
        phi, T = render(layout, k)

        def stack(ph):
            s = 200 + 100 * np.cos(ph[None] + steps[:, None, None]) + black
            return s[None].astype(np.uint16)
        arm = np.full((H, W), 300.0, np.float32)
        np.savez(folder / f"e1_glass_500_550_{w}nm_20260101_01{w % 100:04d}.npz",
                 wavelength=float(w), exposure_ms=100.0,
                 stack_sample_pos=stack(carrier + phi), stack_ref_pos=stack(carrier),
                 sample_arm_sample_pos=(T * arm + black).astype(np.float32),
                 sample_arm_ref_pos=(arm + black).astype(np.float32),
                 sample_arm_std_sample_pos=np.full((H, W), 2.0, np.float32),
                 sample_arm_std_ref_pos=np.full((H, W), 2.0, np.float32), **meta)


def test_process_scan_with_layout(tmp_path, gds, layout):
    (tmp_path / "in").mkdir()
    _write_raw_folder(tmp_path / "in", layout)
    scan = scan_folder(tmp_path / "in")
    cfg = PipelineConfig(solver=PhaseConfig(), device="cpu",
                         layout=L.LayoutConfig(margin_um=20))
    pipe = PhasePipeline(cfg)
    phi_p, t_p = process_scan(scan, pipe, tmp_path / "run", gds=gds, progress=False)
    assert phi_p == tmp_path / "run_phi.h5" and t_p == tmp_path / "run_T.h5"
    assert not (tmp_path / "run_phi_unaligned.h5").exists()
    assert not (tmp_path / "run_T_unaligned.h5").exists()
    with h5py.File(phi_p) as f, h5py.File(t_p) as g:
        assert f.attrs["layout_aligned"] and g.attrs["layout_aligned"]
        assert "layout:" in f.attrs["pipeline_config"]
        phi = f["phi"][0]
        lab = f["layout/labels"][:]
        sol = L.region_mask(f, "solid_1") & ~L.dilate(~L.region_mask(f, "solid_1"), 2)
        clean = f["layout/glass"][:] & ~f["layout/particles"][:] & ~f["layout/extended"][:]
        assert np.sqrt(np.nanmean(phi[clean] ** 2)) < 0.05
        val = np.angle(np.nanmean(np.exp(1j * phi[sol])))
        # one global sign is free after the solve
        assert min(abs(val - 1.8), abs(val + 1.8)) < 0.05
        assert g["T"].shape[1:] == lab.shape
    # a second call finds the finished aligned files and does nothing
    mtime = phi_p.stat().st_mtime
    process_scan(scan, pipe, tmp_path / "run", gds=gds, progress=False)
    assert phi_p.stat().st_mtime == mtime


def test_existing_unaligned_output_is_reused(tmp_path, gds, layout):
    (tmp_path / "in").mkdir()
    _write_raw_folder(tmp_path / "in", layout, wavelengths=(500,))
    scan = scan_folder(tmp_path / "in")
    pipe = PhasePipeline(PipelineConfig(solver=PhaseConfig(), device="cpu",
                                        layout=L.LayoutConfig(margin_um=20)))
    process_scan(scan, pipe, tmp_path / "run", progress=False)          # no layout
    with h5py.File(tmp_path / "run_phi.h5") as f:
        assert "layout_aligned" not in f.attrs
    with pytest.warns(UserWarning, match="reusing unaligned"):
        process_scan(scan, pipe, tmp_path / "run", gds=gds, progress=False)
    with h5py.File(tmp_path / "run_phi.h5") as f:
        assert f.attrs["layout_aligned"]
