"""Fast unit tests for `phase`, on small synthetic stacks (no data/ files needed).

These exist to catch regressions in seconds during the CPU<->GPU backend
work (see phase/backend.py) without loading the ~1.5 GB real acquisitions in
data/ -- run with real data (scripts/test/*.ipynb) remains the authority on
physical correctness; these check the math and the numpy/cupy dispatch.
"""

import numpy as np
import pytest

from phase import (
    PhaseConfig,
    PhaseSolver,
    apply_phase_ripple,
    combine_acquisitions,
    estimate_phase_ripple,
    measure_frame_contrast,
    remove_carrier,
    subtract_reference,
)
from phase.backend import CUPY_AVAILABLE, wrap


def circ_rms_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Wrapped RMS difference between two phase arrays, in degrees."""
    return float(np.degrees(np.sqrt(np.mean(wrap(np.asarray(a) - np.asarray(b)) ** 2))))


def make_stack(H=48, W=64, N=12, seed=0, sign=1.0, dtype=np.float32):
    """Synthetic phase-shifted interferogram stack with known ground truth.

    Phase steps are irregular (not evenly spaced) and per-frame contrast
    varies, so this exercises AIA's blind estimation the way real data does
    rather than the trivial evenly-spaced/unit-gain case.
    """
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    phi_true = sign * np.angle(np.exp(1j * (0.31 * X + 0.19 * Y + 0.4 * np.sin(X / 11.0))))
    b_true = 1.0 + 0.3 * rng.random((H, W))
    a_true = 2.0 + 0.2 * rng.random((H, W))
    delta_true = np.sort(rng.uniform(0, 2 * np.pi, N))
    delta_true -= delta_true[0]
    g_true = 0.6 + 0.8 * rng.random(N)

    stack = np.empty((N, H, W), dtype=np.float64)
    for n in range(N):
        stack[n] = a_true + g_true[n] * b_true * np.cos(sign * phi_true + delta_true[n])
    stack += 0.01 * rng.standard_normal(stack.shape)
    return stack.astype(dtype), dict(phi=phi_true, b=b_true, a=a_true, delta=delta_true, g=g_true)


class TestAIA:
    def test_recovers_known_phase(self):
        stack, truth = make_stack()
        # supply the exact known gain (PhaseConfig(g=...)) to isolate the
        # AIA solve's accuracy from gain-estimation accuracy.
        solver = PhaseSolver(PhaseConfig(g=truth["g"])).fit(stack)
        assert solver.method_param_.converged
        # aia has an exact (phi, delta) -> (-phi, -delta) sign ambiguity
        # (I_n = a + b*cos(phi+delta_n) is invariant under it) -- accept
        # either branch.
        err_same = circ_rms_deg(solver.phi_, truth["phi"])
        err_flip = circ_rms_deg(solver.phi_, -truth["phi"])
        assert min(err_same, err_flip) < 0.5

    def test_dtype_float32_close_to_float64(self):
        stack, truth = make_stack()
        # use_g=False (fixed g=1) isolates dtype effects in the solve itself
        # from any dtype-dependent variation in gain estimation.
        config = PhaseConfig(use_g=False)
        r64 = PhaseSolver(config, dtype=np.float64).fit(stack)
        r32 = PhaseSolver(config, dtype=np.float32).fit(stack)
        assert circ_rms_deg(r32.phi_, r64.phi_) < 1e-2
        assert r32.method_param_.iters_run == r64.method_param_.iters_run
        assert r32.method_param_.converged == r64.method_param_.converged

    def test_gain_auto_matches_supplied_gain_ranking(self):
        stack, truth = make_stack(seed=1)
        g = measure_frame_contrast(stack)
        # recovered gain should correlate strongly with the true per-frame
        # contrast (both normalized to median 1)
        g_true_norm = truth["g"] / np.median(truth["g"])
        assert np.corrcoef(g, g_true_norm)[0, 1] > 0.9

    def test_device_cuda_without_cupy_raises(self):
        if CUPY_AVAILABLE:
            pytest.skip("cupy is installed in this environment")
        stack, _ = make_stack(H=8, W=8, N=6)
        with pytest.raises(RuntimeError):
            PhaseSolver(PhaseConfig(), device="cuda").fit(stack)

    def test_bad_device_raises(self):
        stack, _ = make_stack(H=8, W=8, N=6)
        with pytest.raises(ValueError):
            PhaseSolver(PhaseConfig(), device="tpu").fit(stack)


class TestCarrier:
    def test_recovers_pure_carrier_and_curvature(self):
        H, W = 80, 96
        Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
        kx, ky = 0.05, -0.03
        kxx, kyy, kxy = 2e-4, -1e-4, 5e-5
        piston = 0.6
        phi = np.angle(np.exp(1j * (kx * X + ky * Y + kxx * X**2 + kyy * Y**2
                                     + kxy * X * Y + piston)))
        r = remove_carrier(phi, defocus=True, refine_iters=10, n_blocks=6)
        assert abs(r.kx - kx) < 1e-6
        assert abs(r.ky - ky) < 1e-6
        assert abs(r.kxx - kxx) < 1e-8
        assert abs(r.kyy - kyy) < 1e-8
        assert abs(r.kxy - kxy) < 1e-8
        assert circ_rms_deg(r.phi, np.zeros((H, W))) < 1e-4

    def test_defaults_match_defocus_true_call(self):
        H, W = 64, 64
        rng = np.random.default_rng(2)
        phi = np.angle(np.exp(1j * (0.02 * np.arange(W))[None, :] * np.ones((H, 1))
                              + 1j * 0.05 * rng.standard_normal((H, W))))
        r_default = remove_carrier(phi)
        r_explicit = remove_carrier(phi, defocus=True, refine_iters=10, n_blocks=10)
        assert r_default.kx == r_explicit.kx
        assert r_default.ky == r_explicit.ky


class TestReference:
    def test_resolves_sign_branch(self):
        H, W = 48, 48
        Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
        common = np.angle(np.exp(1j * (0.3 * X + 0.2 * Y)))
        rng = np.random.default_rng(3)
        phi = np.angle(np.exp(1j * (common + 0.02 * rng.standard_normal((H, W)))))
        # reference measured on the opposite sign branch
        phi_ref = np.angle(np.exp(-1j * (common + 0.02 * rng.standard_normal((H, W)))))
        r = subtract_reference(phi, phi_ref)
        assert r.sign == -1
        assert not r.ambiguous
        # phi_ref = -common (opposite branch), so phi + phi_ref cancels the
        # common aberration while phi - phi_ref doubles it -- the chosen
        # (flipped) branch should have much lower spread.
        assert r.spread_flipped < r.spread_same


class TestCombine:
    def test_reduces_scatter_and_resolves_flips(self):
        H, W = 48, 64
        Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
        phi_true = np.angle(np.exp(1j * (0.25 * X + 0.15 * Y)))
        rng = np.random.default_rng(4)

        def acquisition(sign, noise):
            return np.angle(np.exp(1j * sign * (phi_true + noise * rng.standard_normal((H, W)))))

        phis2 = [acquisition(1, 0.05), acquisition(-1, 0.05)]
        phis5 = phis2 + [acquisition(1, 0.05), acquisition(-1, 0.05), acquisition(1, 0.05)]

        r2 = combine_acquisitions(phis2, align_carrier=False)
        r5 = combine_acquisitions(phis5, align_carrier=False)
        assert set(r2.sign_flips) | set(r5.sign_flips)  # some flips detected
        assert circ_rms_deg(r5.phi, phi_true) <= circ_rms_deg(r2.phi, phi_true) + 0.5

    def test_raises_on_single_acquisition(self):
        with pytest.raises(ValueError):
            combine_acquisitions([np.zeros((4, 4))])


class TestRipple:
    def test_roundtrip_recovers_known_ripple(self):
        H, W = 96, 96
        Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
        phi = np.angle(np.exp(1j * (0.1 * X + 0.05 * Y)))
        coeffs_true = {0: 0.01, 1: (0.02, -0.01), 2: (0.005, 0.015)}

        def eps(p):
            e = np.full(p.shape, coeffs_true[0])
            for k, (a, b) in ((1, coeffs_true[1]), (2, coeffs_true[2])):
                e = e + a * np.cos(k * p) + b * np.sin(k * p)
            return e

        phi_w = wrap(phi)
        corrupted = wrap(phi_w + eps(phi_w))
        mask = np.ones((H, W), bool)

        r = estimate_phase_ripple(corrupted, mask, orders=(1, 2), nbins=180)
        assert r.rms_after < r.rms_before

        recovered = apply_phase_ripple(corrupted, r)
        assert circ_rms_deg(recovered, phi_w) < 1.0


class TestBackendWrap:
    def test_wrap_matches_angle_exp(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(-50, 50, 10_000)
        ref = np.angle(np.exp(1j * x))
        assert np.max(np.abs(wrap(x) - ref)) < 1e-12
