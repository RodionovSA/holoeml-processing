"""Synthetic tests for processing.phase.PhasePipeline."""

import numpy as np
import pytest
from phase_shift import PhaseConfig, PhaseResult, remove_carrier
from phase_shift.utils import wrap

from processing.phase import (
    CarrierConfig,
    PhasePipeline,
    PipelineConfig,
    StackResult,
    UniformMaskConfig,
    WvlStackResult,
    fit_carrier,
)

H = W = 64
N = 8


def _stack(phase: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Dark-corrected (N, H, W) stack with slightly non-uniform steps.

    Reversing the frame order makes the solver land on the opposite sign branch.
    """
    steps = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.1 * np.sin(np.arange(N))
    stack = 2.0 + np.cos(phase[None] + steps[:, None, None])
    return (stack[::-1] if reverse else stack).astype(np.float32)


def _scene():
    y, x = np.mgrid[:H, :W]
    carrier = 0.3 * x + 0.2 * y
    bump = np.exp(-((x - 32) ** 2 + (y - 30) ** 2) / 150.0)
    return carrier, bump


def _sign_and_rms(phi: np.ndarray, truth: np.ndarray) -> tuple[int, float]:
    """Branch of ``phi`` relative to ``truth`` and the RMS error on it, piston removed."""
    best = None
    for sign in (1, -1):
        d = wrap(sign * phi - truth)
        rms = float(np.sqrt(np.mean(wrap(d - np.angle(np.mean(np.exp(1j * d)))) ** 2)))
        if best is None or rms < best[1]:
            best = (sign, rms)
    return best


@pytest.fixture
def pipe() -> PhasePipeline:
    return PhasePipeline(PipelineConfig(solver=PhaseConfig(), device="cpu"))


def test_stack_phi(pipe):
    carrier, bump = _scene()
    phi = pipe.process(_stack(carrier + bump), _stack(carrier))
    assert isinstance(phi, np.ndarray) and phi.shape == (H, W)
    _, rms = _sign_and_rms(phi, bump)
    assert rms < 1e-2


def test_stack_full(pipe):
    carrier, bump = _scene()
    res = pipe.process_stack(_stack(carrier + bump), _stack(carrier, reverse=True),
                             output="full")
    assert isinstance(res, StackResult)
    assert isinstance(res.sample, PhaseResult) and isinstance(res.reference, PhaseResult)
    assert res.phi is res.difference.phi
    assert res.difference.sign == -1          # reference solved on the other branch
    assert not res.difference.ambiguous
    _, rms = _sign_and_rms(res.phi, bump)
    assert rms < 1e-2


def test_wvl_stack_single_global_sign(pipe):
    carrier, bump = _scene()
    scale = np.array([1.0, 0.9, 0.8, 0.7, 0.6])        # object phase vs wavelength
    flipped = [False, True, True, False, True]         # wavelengths solved on the other branch
    sample = np.stack([_stack(carrier + s * bump, r) for s, r in zip(scale, flipped)])
    ref = np.stack([_stack(carrier, r) for r in flipped])

    res = pipe.process(sample, ref, output="full", progress=False)
    assert isinstance(res, WvlStackResult)
    assert res.phi.shape == (len(scale), H, W)

    raw = [_sign_and_rms(st.phi, s * bump)[0] for st, s in zip(res.stacks, scale)]
    assert len(set(raw)) == 2                          # solves disagree before alignment

    aligned = [_sign_and_rms(p, s * bump) for p, s in zip(res.phi, scale)]
    assert len({sign for sign, _ in aligned}) == 1     # one global sign after
    assert all(rms < 1e-2 for _, rms in aligned)
    np.testing.assert_array_equal(res.alignment.signs * np.array(raw), raw[0])

    phi = pipe.process(sample, ref, progress=False)
    np.testing.assert_allclose(phi, res.phi)


def _plane_residual_rms(phi: np.ndarray, truth: np.ndarray) -> tuple[int, float]:
    """Branch of ``phi`` and its RMS error vs ``truth`` after fitting piston + tilt."""
    y, x = np.mgrid[:H, :W]
    A = np.stack([np.ones(H * W), x.ravel(), y.ravel()], -1)
    best = None
    for sign in (1, -1):
        d = wrap(sign * phi - truth).ravel()
        resid = d - A @ np.linalg.lstsq(A, d, rcond=None)[0]
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if best is None or rms < best[1]:
            best = (sign, rms)
    return best


@pytest.fixture
def carrier_pipe() -> PhasePipeline:
    return PhasePipeline(PipelineConfig(solver=PhaseConfig(), device="cpu",
                                        carrier=CarrierConfig(enabled=True, degree=1)))


def test_stack_carrier_removed(carrier_pipe):
    carrier, bump = _scene()
    y, x = np.mgrid[:H, :W]
    tilt = 0.04 * x - 0.025 * y                        # residual tilt sample vs reference
    res = carrier_pipe.process_stack(_stack(carrier + tilt + bump), _stack(carrier),
                                     output="full")
    assert res.carrier is not None and res.phi is res.carrier.phi
    _, rms = _plane_residual_rms(res.phi, bump)
    assert rms < 1e-2
    # the raw difference still carries the tilt; the output does not
    _, rms_raw = _sign_and_rms(res.difference.phi, bump)
    _, rms_out = _sign_and_rms(res.phi, bump)
    assert rms_raw > 0.5 and rms_out < 0.2
    expected = remove_carrier(res.difference.phi, weight=res.reference.b, degree=1,
                              device="cpu").phi
    np.testing.assert_allclose(res.phi, expected, atol=1e-5)


def test_wvl_stack_carrier_single_global_sign(carrier_pipe):
    carrier, bump = _scene()
    y, x = np.mgrid[:H, :W]
    scale = np.array([1.0, 0.9, 0.8, 0.7])
    flipped = [False, True, False, True]
    tilts = [0.04 * x, 0.03 * y, -0.035 * x, 0.02 * (x + y)]
    sample = np.stack([_stack(carrier + t + s * bump, r)
                       for s, t, r in zip(scale, tilts, flipped)])
    ref = np.stack([_stack(carrier, r) for r in flipped])

    res = carrier_pipe.process(sample, ref, output="full", progress=False)
    for k, st in enumerate(res.stacks):
        np.testing.assert_array_equal(res.phi[k], res.alignment.signs[k] * st.phi)
    aligned = [_plane_residual_rms(p, s * bump) for p, s in zip(res.phi, scale)]
    assert len({sign for sign, _ in aligned}) == 1
    assert all(rms < 1e-2 for _, rms in aligned)
    assert set(res.alignment.signs.tolist()) == {1, -1}


def test_shape_errors(pipe):
    a = np.ones((N, H, W), np.float32)
    with pytest.raises(ValueError):
        pipe.process(a[0], a[0])
    with pytest.raises(ValueError):
        pipe.process_stack(a, a[:, :-1])
    with pytest.raises(ValueError):
        pipe.process_wvl_stack(a[None], np.stack([a, a]))
    with pytest.raises(ValueError):
        pipe.process_stack(a, a, output="bogus")


def test_config_yaml_roundtrip(tmp_path):
    cfg = PipelineConfig(solver=PhaseConfig(method="aia", method_kwargs={"iters": 5}),
                         device="cpu", weight="none",
                         carrier=CarrierConfig(enabled=True, degree=3, subsample=2,
                                               uniform_mask=UniformMaskConfig(enabled=True,
                                                                              grow=3)))
    path = tmp_path / "phase.yaml"
    cfg.to_yaml(path)
    assert PipelineConfig.from_yaml(path) == cfg


# --------------------------------------------------------------------------- uniform mask

def _plateau_scene(n: int = 128, step: float = 1.5):
    """Tilted substrate with an off-centre plateau of height ``step`` and mild noise."""
    rng = np.random.default_rng(1)
    y, x = np.mgrid[:n, :n]
    structure = ((x - 90) ** 2 + (y - 40) ** 2 < 30 ** 2).astype(float)
    tilt = 0.04 * x - 0.03 * y
    phi = wrap(tilt + step * structure + 0.02 * rng.standard_normal((n, n)))
    return phi, structure.astype(bool)


def test_fit_carrier_uniform_mask_removes_plateau_bias():
    phi, structure = _plateau_scene()
    glass = ~structure

    plain, m0 = fit_carrier(phi, CarrierConfig(enabled=True, degree=1), device="cpu")
    assert m0 is None
    masked, m = fit_carrier(
        phi, CarrierConfig(enabled=True, degree=1,
                           uniform_mask=UniformMaskConfig(enabled=True, threshold=0.3, grow=2)),
        device="cpu")

    def glass_rms(p):
        d = p[glass]
        return float(np.sqrt(np.mean(wrap(d - np.angle(np.mean(np.exp(1j * d)))) ** 2)))

    assert glass_rms(plain.phi) > 0.1                 # plateau tilts the full-map fit
    assert glass_rms(masked.phi) < 0.03               # substrate flat to the noise level
    assert not m[structure].any()                     # plateau excluded
    assert m[glass].mean() > 0.9                      # most substrate kept
    level = np.angle(np.mean(np.exp(1j * masked.phi[structure])))
    assert abs(level - 1.5) < 0.05                    # plateau height recovered


def test_fit_carrier_uniform_mask_respects_user_mask_and_min_fraction():
    phi, structure = _plateau_scene()
    user = np.zeros_like(structure)
    user[:, :64] = True                               # left half only
    _, m = fit_carrier(
        phi, CarrierConfig(enabled=True, uniform_mask=UniformMaskConfig(enabled=True)),
        mask=user, device="cpu")
    assert not m[~user].any()

    with pytest.warns(UserWarning, match="min_fraction"):
        fit_carrier(phi, CarrierConfig(enabled=True, uniform_mask=UniformMaskConfig(
            enabled=True, threshold=0.3, min_fraction=0.99)), device="cpu")


def test_pipeline_reports_carrier_mask():
    carrier, bump = _scene()
    cfg = PipelineConfig(solver=PhaseConfig(), device="cpu", carrier=CarrierConfig(
        enabled=True, uniform_mask=UniformMaskConfig(enabled=True, threshold=0.2)))
    res = PhasePipeline(cfg).process_stack(_stack(carrier + bump), _stack(carrier),
                                           output="full")
    assert res.carrier_mask is not None and res.carrier_mask.shape == (H, W)
    assert not res.carrier_mask[bump > 0.5].any()     # bump top excluded from the fit
