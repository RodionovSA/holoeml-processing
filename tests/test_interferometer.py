"""Synthetic tests for processing.interferometer."""

import h5py
import numpy as np
import pytest
from phase_shift import PhaseConfig
from phase_shift.utils import wrap

from processing.interferometer import (
    DarkFrame,
    SignTracker,
    bad_pixel_mask,
    median_fill,
    parse_name,
    process_scan,
    scan_folder,
    transmission,
)
from processing.phase import PhasePipeline, PipelineConfig

H = W = 48
N = 8
BLACK = 9.0


def test_parse_name():
    f = parse_name("/x/holoeml281026_glass_400_455_405nm_20261108_043827.npz")
    assert (f.exp, f.sample, f.block, f.kind, f.wavelength) == (
        "holoeml281026", "glass", "400_455", "wvl", 405.0)
    assert f.time.hour == 4 and f.time.second == 27
    assert parse_name("e_my_sample_1_2_black_20260101_000000.npz").sample == "my_sample"
    assert parse_name("e_s_1_2_mono_check_20260101_000000.npz").kind == "mono_check"
    with pytest.raises(ValueError):
        parse_name("random.npz")


def test_bad_pixel_mask():
    rng = np.random.default_rng(0)
    black = BLACK + rng.normal(0, 1, (H, W)).astype(np.float32)
    std = np.full((H, W), 1.5, np.float32) + rng.normal(0, 0.05, (H, W)).astype(np.float32)
    black[3, 4] = 500          # hot
    std[10, 11] = 40           # noisy
    std[20, 21] = 0            # stuck
    bad = bad_pixel_mask(black, std)
    assert bad[3, 4] and bad[10, 11] and bad[20, 21]
    assert bad.sum() == 3


def test_median_fill_stack_and_cluster():
    img = np.tile(np.arange(W, dtype=np.float32), (3, H, 1)) + np.arange(3)[:, None, None]
    truth = img.copy()
    bad = np.zeros((H, W), bool)
    bad[5, 5] = True
    bad[20:23, 30:33] = True          # 3x3 cluster: centre needs radius 2
    bad[0, 0] = True                  # corner
    img[:, bad] = 1e4
    median_fill(img, bad)
    np.testing.assert_allclose(img, truth, atol=1.0)
    np.testing.assert_allclose(img[:, 5, 5], truth[:, 5, 5])


def test_transmission():
    t, s = transmission(np.array([50.0, 1.0]), np.array([100.0, 0.0]),
                        np.array([3.0, 1.0]), np.array([4.0, 1.0]))
    assert t[0] == pytest.approx(0.5)
    assert s[0] == pytest.approx(np.sqrt(9 + 0.25 * 16) / 100)
    assert np.isnan(t[1]) and np.isnan(s[1])


def test_sign_tracker_matches_flip():
    y, x = np.mgrid[:H, :W]
    base = 0.5 * np.sin(x / 7.0) + 0.3 * y / H
    tr = SignTracker()
    out0, s0, _, ref0 = tr.align(0, base)
    out1, s1, amb, ref1 = tr.align(1, -1.05 * base)
    assert (s0, ref0) == (1, -1) and (s1, ref1) == (-1, 0) and not amb
    np.testing.assert_allclose(out1, 1.05 * base, atol=1e-6)


# --------------------------------------------------------------------------- folder

def _write_folder(tmp_path, wavelengths=((500, 505), (510,)), bad=((7, 9),)):
    """Two blocks; per wavelength the film adds ``bump * 500 / wvl`` and T = 0.8."""
    y, x = np.mgrid[:H, :W]
    carrier = 0.3 * x + 0.2 * y
    bump = 1.2 * np.exp(-((x - 24) ** 2 + (y - 22) ** 2) / 80.0)
    steps = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.1 * np.sin(np.arange(N))
    black = np.full((H, W), BLACK, np.float32)
    black_std = np.full((H, W), 1.3, np.float32) + 0.01 * np.sin(x + y).astype(np.float32)
    for yy, xx in bad:
        black[yy, xx] = 2000
    meta = dict(n=N, n_back=5, gain=0, repeat=1, mono_check_peak_nm=549.0)
    for bi, block in enumerate(wavelengths):
        name = f"{block[0]}_{block[-1]}"
        exp = 500.0 + 300 * bi
        np.savez(tmp_path / f"e1_glass_{name}_black_20260101_00000{bi}.npz", black=black,
                 black_std=black_std, exposure_ms=exp, wavelengths=np.array(block, float), **meta)
        np.savez(tmp_path / f"e1_glass_{name}_mono_check_20260101_00000{bi}.npz",
                 wavelength=550.0, peak_nm=549.0)
        for w in block:
            def stack(ph, rev):
                s = 200 + 100 * np.cos(ph[None] + steps[:, None, None])
                s = s[::-1] if rev else s
                s = s + black
                s[:, bad[0][0], bad[0][1]] = 4000     # junk on the bad pixel
                return s[None].astype(np.uint16)
            arm_r = 300.0 + 0.5 * x
            np.savez(tmp_path / f"e1_glass_{name}_{w}nm_20260101_01{w % 100:04d}.npz",
                     wavelength=float(w), exposure_ms=exp,
                     stack_sample_pos=stack(carrier + bump * 500 / w, rev=(w == 505)),
                     stack_ref_pos=stack(carrier, rev=False),
                     sample_arm_sample_pos=(0.8 * arm_r + black).astype(np.float32),
                     sample_arm_ref_pos=(arm_r + black).astype(np.float32),
                     sample_arm_std_sample_pos=np.full((H, W), 4.0, np.float32),
                     sample_arm_std_ref_pos=np.full((H, W), 5.0, np.float32), **meta)
    return bump


def test_scan_folder(tmp_path):
    _write_folder(tmp_path)
    (tmp_path / "notes.npz").write_bytes(b"")
    with pytest.warns(UserWarning, match="notes.npz"):
        scan = scan_folder(tmp_path)
    assert scan.name == "e1_glass"
    assert [b.name for b in scan.blocks] == ["500_505", "510_510"]
    np.testing.assert_array_equal(scan.wavelengths, [500, 505, 510])
    assert DarkFrame.load(scan.blocks[1].black).exposure_ms == 800


@pytest.fixture
def pipe():
    return PhasePipeline(PipelineConfig(solver=PhaseConfig(), device="cpu"))


def test_process_scan_end_to_end_and_resume(tmp_path, pipe):
    (tmp_path / "in").mkdir()
    bump = _write_folder(tmp_path / "in")
    scan = scan_folder(tmp_path / "in")
    phi_p, t_p = process_scan(scan, pipe, tmp_path / "SiO2_aia", progress=False)
    assert (phi_p.name, t_p.name) == ("SiO2_aia_phi.h5", "SiO2_aia_T.h5")

    with h5py.File(phi_p) as f, h5py.File(t_p) as g:
        assert f["done"][:].all() and g["done"][:].all()
        assert f["phi"].shape == (3, H, W) and f["b_ref"].shape == (3, H, W)
        assert f["bad_pixels"][0, 7, 9] and f["bad_pixels"][0].sum() == 1
        phi = f["phi"][:]
        # one global sign for the whole scan, despite the reversed stack at 505 nm
        gsign = np.sign(np.sum(phi[0] * bump))
        for k, w in enumerate(scan.wavelengths):
            d = wrap(gsign * phi[k] - bump * 500 / w)
            d = wrap(d - np.angle(np.mean(np.exp(1j * d))))
            assert np.sqrt(np.mean(d ** 2)) < 0.02, w
        assert not f["align_ambiguous"][:].any()
        np.testing.assert_allclose(g["T"][:], 0.8, atol=1e-4)
        expected = np.sqrt(16 + 0.64 * 25) / (300 + 0.5 * np.arange(W))
        np.testing.assert_allclose(g["T_std"][0], np.broadcast_to(expected, (H, W)), rtol=1e-4)
        assert list(g["exposure_ms"][:]) == [500, 500, 800]

    # resume: drop the last row's flag, rerun, only that row is recomputed
    with h5py.File(phi_p, "a") as f:
        f["done"][2] = False
        f["phi"][2] = 0
    process_scan(scan, pipe, tmp_path / "SiO2_aia", progress=False)
    with h5py.File(phi_p) as f:
        assert f["done"][:].all()
        np.testing.assert_allclose(f["phi"][2], phi[2], atol=1e-5)


def test_exposure_mismatch(tmp_path, pipe):
    _write_folder(tmp_path)
    scan = scan_folder(tmp_path)
    blk = scan.blocks[0]
    blk.black = scan.blocks[1].black                 # 800 ms dark for 500 ms data
    with pytest.raises(ValueError, match="exposure"):
        process_scan(scan, pipe, tmp_path / "x", progress=False)


def test_no_transmission_and_missing_folder(tmp_path, pipe):
    _write_folder(tmp_path, wavelengths=((500,),))
    scan = scan_folder(tmp_path)
    phi_p, t_p = process_scan(scan, pipe, tmp_path / "run_b.h5", transmission=False,
                              progress=False)
    assert phi_p == tmp_path / "run_b_phi.h5" and t_p is None
    assert not (tmp_path / "run_b_T.h5").exists()
    with h5py.File(phi_p) as f:
        assert f["done"][:].all() and "phi" in f
    with pytest.raises(FileNotFoundError):
        process_scan(scan, pipe, tmp_path / "missing" / "run", progress=False)
    assert not (tmp_path / "missing").exists()
