# Gauge Conventions

Several steps in this pipeline have exact ambiguities — transformations of
the recovered fields that leave the fitted data unchanged. This is a
reference for which convention pins each one down, and where. It documents
existing choices; it does not introduce new ones.

## Shared model (`interference_model.md` Eq. 8, `aia.md`)

| Freedom | Convention | Where |
|---|---|---|
| `(Φ, δ_n) → (Φ−c, δ_n+c)` for any constant `c` | `δ_n[0] = 0`, re-applied every iteration. `Φ` itself still carries an unresolved additive constant — harmless for relative phase maps. | `aia.py` (`aia`, `aia_step_field`) |
| `(Φ, δ) → (−Φ, −δ)` (cosine is even) | **Not** pinned by `aia()` — two independent solves may land on opposite branches. Resolved downstream (see Reference/Combine below). | `aia.py` docstrings; `reference.py` |
| `α_n` (per-frame source-power scale): `α→λα`, `(a,b)→(a,b)/λ` | `median(alpha) = 1` | `solver.py` (`PhaseSolver._normalize`) |
| `g_n` (fringe-contrast scale): only enters as `g_n·b` | `median(g) = 1` | `aia.py` (`aia`), `utils.py` (`measure_frame_contrast`) |
| `u, v` sign labeling (not a free gauge — a naming choice) | `u = b·cos(Φ)`, `v = −b·sin(Φ)` ⇒ `Φ = atan2(−v, u)`, `b = hypot(u, v)` | `aia.py` (`aia_pixel_step`, `aia`); `interference_model.md` Eq. 18 |

## AIA joint-gain fit (`fit_gain=True`)

| Freedom | Convention | Where |
|---|---|---|
| `a(x) + c_n` invariant under `a→a+k`, `c_n→c_n−k` | `mean(c_n) = 0` | `aia.py` (`aia`, `aia_frame_step`) |
| With `g_n` free, `(u,v)` determined only up to any invertible `M` (`GL(2)`), not just a rotation | Whiten `(u,v)` so `Σu² = Σv²`, `Σu·v = 0` — collapses `GL(2)` down to `O(2)`, the same residual ambiguity plain (`g≡1`) AIA already has | `aia.py` (`_whiten_uv`); derivation in `docs/aia.md` §"A gauge freedom that only appears once g_n is free" |
| `(g_n, δ_n) → (−g_n, δ_n+π)` | `g_n = hypot(P_n, Q_n) ≥ 0` always | `aia.py` (`aia_frame_step`) |

## Step-field refinement (`aia_step_field`)

| Freedom | Convention | Where |
|---|---|---|
| Spatial split of `δ_n(x,y)` into piston `δ_n` + field `Δ_n(x,y)` | Origin at the field centroid, coordinates scaled to ≈`[-1,1]`, each basis term zero-mean over the field, orthonormalized in ascending degree | `step_field.py` (`_poly_basis`); `step_field_residuals.md` §1.2(i), Eq. T3 |
| `c_jn ↔ Φ`: a per-frame-constant basis coefficient is indistinguishable from part of `Φ` | Frame-mean-zero: `c_jn ← c_jn − mean_n(c_jn)`, applied before correcting the data each refine round. **`StepFieldParam.coeffs` is reported un-gauge-fixed** — subtract the frame mean yourself before reading a row as physical per-frame drift. | `step_field.py` (`aia_step_field`); `step_field_residuals.md` §1.2(ii)/§8.4, Eq. T3b/E4 |

## Carrier removal (`remove_carrier`)

| Freedom | Convention | Where |
|---|---|---|
| Spatial origin for the tilt/curvature/piston split (origin-dependent — differs from the step-field's centroid convention above) | Pixel `(0,0)`, unnormalized `x,y` | `carrier.py` |
| Global piston of the output | Weighted circular mean set to zero: `arg(Σ w·e^{iφ}) = 0` | `carrier.py` |
| Carrier frequency `(fx,fy)`, defined only modulo 1 cycle/pixel | FFT-bin peak picks the representative; the refine step tracks the nearest branch to the current estimate | `carrier.py` |

## Reference subtraction (`subtract_reference`)

| Freedom | Convention | Where |
|---|---|---|
| `(Φ,δ)→(−Φ,−δ)` sign branch between `phi` and `phi_ref` | Compute both `wrap(phi∓phi_ref)`; keep whichever has lower weighted circular spread. `phi_ref` is the one flipped; `phi` is kept as-is. Ties go to `phi − phi_ref`. | `reference.py` (`subtract_reference`) |

## Multi-acquisition combination (`combine_acquisitions`)

| Freedom | Convention | Where |
|---|---|---|
| Sign branch across `k` acquisitions | Every map reoriented to agree with `phis[reference]` (default index `0`), using `subtract_reference`'s discriminant — run on the *raw* maps, before carrier removal (a stronger, unambiguous discriminant there) | `combine.py` (`combine_acquisitions`) |
| Averaging convention for wrapped angles | Weighted complex mean, `angle(Σ w·e^{iφ} / Σw)` — never an arithmetic mean of wrapped angles | `combine.py` |

## Phase ripple (`estimate_phase_ripple` / `apply_phase_ripple`)

| Freedom | Convention | Where |
|---|---|---|
| Which phase variable `eps(·)` is binned/expressed in | The *original*, pre-carrier-removal wrapped `phi` — so `RippleResult.coeffs` are only valid for maps sharing that same `phi` (i.e. the same solve's `δ[0]=0` pin and sign branch) | `ripple.py` |
| `c0` term is degenerate with the map's global piston | Not re-zeroed — applying `eps` shifts the corrected map's global offset by `−c0` | `ripple.py` |
