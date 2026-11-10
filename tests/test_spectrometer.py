"""Tests for processing.spectrometer on synthetic OceanView exports."""

import math

import numpy as np
import pytest

from processing.spectrometer import parse_name, read_folder, read_spectrum

W = np.round(np.linspace(188.0, 1047.0, 16), 3)


def _write(folder, stem, counts, t_int=0.0338, w=W):
    rows = "\n".join(f"{a}\t{b}" for a, b in zip(w, counts))
    text = (f"Data from {stem}.txt Node\n\nDate: Sun Nov 08 10:34:46 AST 2026\nUser: x\n"
            f"Spectrometer: SR600411\nIntegration Time (sec): {t_int:E}\nScans to average: 10\n"
            f"Nonlinearity correction enabled: false\nXAxis mode: Wavelengths\n"
            f"Number of Pixels in Spectrum: {len(w)}\n>>>>>Begin Spectral Data<<<<<\n{rows}\n")
    (folder / f"{stem}.txt").write_text(text.replace("\n", "\r\n"))


@pytest.mark.parametrize("stem, name, group, value, mode, index", [
    ("HoloEML281026_LPCVD_meta125_Reflection__0__10-34-46-933", "meta125", "meta", 125.0, "Reflection", 0),
    ("HoloEML281026_square1_Transmission__0__11-50-21-086", "square1", "square", 1.0, "Transmission", 0),
    ("HoloEML281026_TiO2_glass_Reflection__1__11-24-19-022", "glass", "glass", math.nan, "Reflection", 1),
    ("plain_name", "name", "name", math.nan, None, -1),
])
def test_parse_name(stem, name, group, value, mode, index):
    f = parse_name(stem + ".txt")
    assert (f.name, f.group, f.mode, f.index) == (name, group, mode, index)
    assert f.value == value or (math.isnan(f.value) and math.isnan(value))


def test_read_spectrum_header_and_data(tmp_path):
    _write(tmp_path, "E_S_meta5_Reflection__0__10-00-00-000", np.arange(16.0))
    s = read_spectrum(tmp_path / "E_S_meta5_Reflection__0__10-00-00-000.txt")
    np.testing.assert_allclose(s.wavelength, W)
    np.testing.assert_allclose(s.counts, np.arange(16.0))
    assert s.integration_time == pytest.approx(0.0338)
    assert s.header["Nonlinearity correction enabled"] is False
    assert s.header["Date"].year == 2026


def test_read_folder_sort_select_duplicates(tmp_path):
    for i, v in enumerate([100, 75, 250]):
        _write(tmp_path, f"E_S_meta{v}_Reflection__0__10-00-0{i}-000", np.full(16, v))
    _write(tmp_path, "E_S_glass_Reflection__0__10-01-00-000", np.full(16, 1.0))
    _write(tmp_path, "E_S_glass_Reflection__0__10-02-00-000", np.full(16, 3.0))

    s = read_folder(tmp_path)
    assert s.counts.shape == (5, 16)
    assert s.names == ["glass", "glass", "meta75", "meta100", "meta250"]
    np.testing.assert_array_equal(s.select(group="meta").values, [75, 100, 250])
    assert s["glass"].shape == (2, 16)
    np.testing.assert_allclose(s.mean_duplicates()["glass"], 2.0)
    assert s[1:3].names == ["glass", "meta75"]
    assert s.crop(200, 500).wavelength.min() >= 200


def test_integration_time_warning_and_per_second(tmp_path):
    _write(tmp_path, "a_x1__0__10-00-00-000", np.full(16, 10.0), t_int=0.1)
    _write(tmp_path, "a_x2__0__10-00-01-000", np.full(16, 10.0), t_int=0.5)
    with pytest.warns(UserWarning, match="integration times differ"):
        s = read_folder(tmp_path)
    np.testing.assert_allclose(s.per_second().counts[:, 0], [100.0, 20.0])


def test_mismatched_wavelengths_raise(tmp_path):
    _write(tmp_path, "a_x1__0__10-00-00-000", np.zeros(16))
    _write(tmp_path, "a_x2__0__10-00-01-000", np.zeros(16), w=W + 1.0)
    with pytest.raises(ValueError, match="wavelength axis"):
        read_folder(tmp_path)


def test_resample_linear_and_mean(tmp_path):
    w = np.round(np.arange(380.0, 820.0, 0.5), 3)
    _write(tmp_path, "a_x1__0__10-00-00-000", 2.0 * w + 1.0, w=w)
    s = read_folder(tmp_path)

    lin = s.resample(400, 800, 5)
    np.testing.assert_allclose(lin.wavelength, np.arange(400, 801, 5))
    np.testing.assert_allclose(lin.counts[0], 2.0 * lin.wavelength + 1.0)

    # Bins [w-2.5, w+2.5) on a 0.5 nm grid are symmetric minus one end pixel.
    mean = read_folder(tmp_path, grid=(400, 800, 5), method="mean")
    np.testing.assert_allclose(mean.counts[0], 2.0 * (mean.wavelength - 0.25) + 1.0)

    custom = s.resample(wavelength=[450.0, 532.0, 633.0])
    np.testing.assert_allclose(custom.counts[0], [901.0, 1065.0, 1267.0])

    with pytest.raises(ValueError, match="outside measured"):
        s.resample(300, 800, 5)


# Gaussian trades lower noise for amplitude loss on real features (~2.5% of the
# 120 nm cosine at 10 nm FWHM), hence the looser bound.
@pytest.mark.parametrize("method, ratio", [("savgol", 0.35), ("gaussian", 0.5)])
def test_smooth_keeps_shape_and_reduces_noise(tmp_path, method, ratio):
    w = np.round(np.arange(400.0, 800.0, 0.4), 3)
    clean = 50 + 20 * np.cos(2 * np.pi * (w - 400) / 120)        # 120 nm period
    noisy = clean + np.random.default_rng(0).normal(0, 1.0, w.size)
    _write(tmp_path, "a_x1__0__10-00-00-000", noisy, w=w)
    s = read_folder(tmp_path)
    out = s.smooth(10, method=method).counts[0]
    inner = slice(30, -30)
    assert np.std((out - clean)[inner]) < ratio * np.std((noisy - clean)[inner])
    assert np.max(np.abs(out - clean)[inner]) < 1.5


def test_savgol_exact_on_cubic_including_edges(tmp_path):
    w = np.round(np.arange(400.0, 500.0, 0.5), 3)
    x = (w - 450) / 50
    y = 3 + x - 2 * x**2 + 0.5 * x**3
    _write(tmp_path, "a_x1__0__10-00-00-000", y, w=w)
    np.testing.assert_allclose(read_folder(tmp_path).smooth(8).counts[0], y, atol=1e-3)


def test_mirror_reference_shipped_and_sane():
    from processing.references import load_mirror, mirror_reflectance

    for aoi in (12, 45):
        t = load_mirror(aoi)
        assert t.shape[1] == 4 and np.all(np.diff(t[:, 0]) > 0)
    R = mirror_reflectance(np.arange(430.0, 901.0, 5.0))
    assert np.all((R > 0.9) & (R < 1.0))
    with pytest.raises(ValueError, match="outside"):
        mirror_reflectance([250.0])
