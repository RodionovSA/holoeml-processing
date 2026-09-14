# Carrier removal

This document derives carrier removal — subtracting a smooth, low-order phase field from a
recovered wrapped phase map — as a single optimization problem, posed on the complex exponential
of the phase rather than on the phase itself. Doing so is what makes the problem well posed at
all: the measurement in [`interference_model.md`](interference_model.md) constrains $e^{i\Phi}$,
never $\Phi$, so any estimator built from $\Phi$ or a difference of $\Phi$'s carries a spurious
$2\pi$-wrap ambiguity that one built from $e^{i\Phi}$ does not. It backs `phase.carrier`, and its
basis is the same degree-$M$ polynomial family
[`step_field_residuals.md`](step_field_residuals.md) already built for the spatially-varying
phase-step field — the carrier is that family's static member (§6 makes this precise).

Unlike `interference_model.md`'s and `step_field_residuals.md`'s derivations, the $(x,y)$
dependence is not suppressed here: the whole subject is the spatial shape of $\Phi$, so pixel
coordinates stay explicit throughout.

## 1. Setup

### 1.1 What the carrier is

`interference_model.md` Eq. (15) defines the total static phase a solve returns as

$$\Phi(x,y) = \phi(x,y) + \phi_{\text{inst}}(x,y) + \phi_{\text{carrier}}(x,y) \tag{C0}$$

and Eq. (17) is fit for $\Phi$ directly — never for $\phi$ alone, since a single acquisition
cannot separate $\phi$ from $\phi_{\text{inst}}$ or $\phi_{\text{carrier}}$ by their functional
form. $\phi_{\text{carrier}}$ is smooth and low-order by construction (the tilt/defocus set by
the reference-arm angle and curvature mismatch); $\phi_{\text{inst}}$ is typically low-order too,
from fixed path/aberration mismatch. Carrier removal does not distinguish the two — it fits and
subtracts a low-order model of $\Phi$ itself, on the working assumption that whatever of $\Phi$
is captured by that model is instrumental rather than sample structure. §4.5 returns to what this
assumption costs.

### 1.2 Why exponentials

Eq. (17) depends on $\Phi(x,y)$ only through $\cos(\Phi + \delta_n + \Delta_n)$. At a single
pixel, two candidate phases that differ by any integer multiple of $2\pi$ produce identical data
for every frame $n$ — the model, and therefore any solve of it, cannot tell them apart. So the
solved $\Phi(x,y)$ that carrier removal receives is only ever known up to an independent integer
multiple of $2\pi$ at each pixel: what the data actually determines is the field
$e^{i\Phi(x,y)}$, not $\Phi(x,y)$ itself. An estimator is only correctly posed against this model
if it is exactly invariant under $\Phi(x,y) \to \Phi(x,y) + 2\pi m(x,y)$ for arbitrary integer
$m(x,y)$ — and that invariance is available for free, without unwrapping, only to a criterion
built from $e^{i\Phi}$. §2.4 shows concretely what breaks in a criterion built from $\Phi$ (or a
wrap-safe difference of it) instead.

### 1.3 Basis and its gauge

Reuse `step_field_residuals.md` Eq. (T1)/(T3) directly: on coordinates centred at the field
centroid and scaled to $\approx[-1,1]$, $p_1,\dots,p_J$ are the monomials of total degree $1$
through $M$, each made zero-mean over the field and Gram–Schmidt-orthonormalized in ascending
degree ($J = \tfrac{(M+1)(M+2)}{2}-1$) — exactly what `phase.methods.step_field._poly_basis`
already builds; this document does not redefine it.

Carrier removal additionally needs the constant term that basis deliberately excludes (it is
exactly the step field's piston $\delta_n$ there, Eq. T1). Add it back as $p_0(x,y) \equiv
1/\sqrt{N_p}$ ($N_p = H\times W$, so $p_0$ has unit norm over the field). Because every $p_j$,
$j\ge1$, is field-mean-zero (Eq. T3), $p_0$ is automatically orthogonal to all of them — no
re-orthonormalization needed — so $\{p_0, p_1, \dots, p_J\}$ is an orthonormal basis of
$J+1$ functions. Define the fitted carrier field

$$P(x,y;\mathbf a) = \sum_{j=0}^{J} a_j\,p_j(x,y) \tag{C1}$$

$a_0$ carries the constant (piston) part of $\Phi$; $a_1,\dots,a_J$ the low-order spatial shape.

## 2. The estimation problem

Index the field's $N_p$ pixels by $k$, so $\Phi_k \equiv \Phi(x_k,y_k)$, $P_k(\mathbf a) \equiv
P(x_k,y_k;\mathbf a)$, and similarly for any other per-pixel quantity below.

### 2.1 The weighted complex field

Define the weighted complex field $z_k = w_k\,e^{i\Phi_k}$, with $w_k \ge 0$ a per-pixel
reliability weight. Two instances of this are used in practice, and they coincide:

- **From a wrapped map.** $\Phi_k$ is the solved `phi`; take $w_k = b_k$ (the modulation map,
  `PhaseResult.b`) or a supplied `weight`/`mask`, mirroring `remove_carrier`'s existing arguments.
- **From quadrature.** `interference_model.md` Eq. (18) gives $u = b\cos\Phi$, $v = -b\sin\Phi$,
  so $e^{i\Phi} = \cos\Phi + i\sin\Phi = (u - iv)/b$. Taking $w_k = b_k$ exactly cancels the
  division: $z_k = b_k\cdot(u_k-iv_k)/b_k = u_k - iv_k$ (Eq. C2). The modulation weight is then
  free — no separate array, no division by a $b$ that may be near zero — and a pixel with little
  or no fringe contrast self-down-weights to $z_k \approx 0$ rather than contributing an unstable
  phase.

$$z_k = w_k\,e^{i\Phi_k} = u_k - i v_k \quad\text{(with } w_k = b_k\text{)} \tag{C2}$$

### 2.2 The objective

With the demodulated field $d_k(\mathbf a) = z_k\,e^{-iP_k(\mathbf a)}$ and residual $r_k(\mathbf
a) \equiv \Phi_k - P_k(\mathbf a)$, define

$$F(\mathbf a) \;\equiv\; \sum_k w_k\cos\big(r_k(\mathbf a)\big) \;=\; \operatorname{Re}\Big(\sum_k z_k\,e^{-iP_k(\mathbf a)}\Big) \tag{C3}$$

and pose carrier removal as $\boxed{\max_{\mathbf a} F(\mathbf a)}$. This is equivalent to
minimizing a wrap-invariant squared residual: since $w_k(1-\cos r_k) = \tfrac12 w_k\big|e^{ir_k}
-1\big|^2 = \tfrac12 w_k\big|e^{i\Phi_k}-e^{iP_k}\big|^2$ (from $|e^{i\theta}-1|^2 = 2-2\cos
\theta$), and $\sum_k w_k$ does not depend on $\mathbf a$,

$$\sum_k w_k\big(1-\cos r_k\big) \;=\; \frac12\sum_k w_k\,\big|e^{i\Phi_k}-e^{iP_k}\big|^2 \;=\; \Big(\sum_k w_k\Big) - F(\mathbf a) \tag{C3b}$$

so maximizing $F$ is exactly minimizing $\sum_k w_k|e^{i\Phi_k}-e^{iP_k}|^2$ — the squared
Euclidean distance between two points on the unit circle, the natural distance once $\Phi$ is
replaced by its exponential. And since $1-\cos r \approx r^2/2$ for small $r$, this *is* "minimize
the leftover phase" — stated in the one form that is exactly wrap-invariant rather than only
approximately so, reducing to ordinary weighted least squares on the residual once the fit is
close (§3.3).

### 2.3 Piston-free variant

Split $P = a_0 p_0 + P'(x,y;a_{1:J})$ with $P' = \sum_{j=1}^J a_j p_j$, and write $\psi_0 \equiv
a_0 p_0$ for the (spatially uniform) piston phase. Then $F = \operatorname{Re}\big(e^{-i\psi_0}R
\big)$ with $R \equiv \sum_k w_k e^{i(\Phi_k-P'_k)}$, which is maximized over $\psi_0$ at $\psi_0
= \arg R$, giving $\max_{\psi_0} F = |R|$. So fitting the piston jointly with the rest is
equivalent to profiling it out:

$$\max_{a_1,\dots,a_J}\ \Big|\sum_k w_k\,e^{i(\Phi_k - P'_k(\mathbf a))}\Big|, \qquad \hat\psi_0 = \arg\Big(\sum_k w_k\,e^{i(\Phi_k-P'_k)}\Big) \tag{C4}$$

The resultant modulus $|R|$ is the same weighted-circular-spread quantity `combine.py` and
`reference.py` already use as a discriminant (`gauge_conventions.md`'s reference-subtraction row:
"keep whichever has lower weighted circular spread").

### 2.4 Why not least squares on `wrap_sub(Φ, P)`

`phase.backend.wrap_sub` gives a wrap-safe *difference* $\operatorname{wrap}(\Phi-P)$, but its
square is still a discontinuous function of $\mathbf a$: as $\mathbf a$ varies continuously, any
pixel whose residual crosses $\pm\pi$ makes $\operatorname{wrap}(\Phi-P)^2$ jump discontinuously,
creating a spurious stationary point at the jump and making the objective's value (and hence the
fit) depend on which branch of the wrap each pixel happened to land on. $F(\mathbf a)$ (Eq. C3) is
the smooth surrogate that agrees with it to $O(r^2)$ (§2.2) but has no such jumps at any $\mathbf
a$, for the reason given in §1.2. Credit where due: `phase.carrier`'s present implementation
already works entirely on $e^{i\phi}$ and `xp.angle(...)`, never unwrapping — what §2.2 adds on
top is a single explicit objective that its result can be shown to optimize, not the exponential
framing itself.

## 3. Normal equations

### 3.1 Stationarity

Since $\partial r_k/\partial a_j = -p_j(x_k)$, $\partial F/\partial a_j = \sum_k w_k p_j(x_k)
\sin(r_k)$, so a maximizer of $F$ satisfies

$$\sum_k w_k\,p_j(x_k)\,\sin(r_k) = 0, \qquad j = 0,\dots,J \tag{C5}$$

the wrap-invariant analogue of "residual orthogonal to the column space": it is the *sine* of the
residual, not the residual itself, that must be orthogonal to every basis function. Equivalently,
in compact complex form, $\sum_k w_k\,p_j(x_k)\,e^{ir_k}$ is real for every $j$.

The $j=0$ row alone reduces (since $p_0$ is a positive constant) to $\sum_k w_k\sin(r_k) = 0
\iff \arg\big(\sum_k w_k e^{ir_k}\big) \in \{0,\pi\}$; the maximizer of $F$ (rather than its
minimizer) selects the $\arg = 0$ branch, so

$$\arg\Big(\sum_k w_k\,e^{ir_k}\Big) = 0 \tag{C5b}$$

— the weighted circular mean of the demodulated field set to zero, exactly the piston convention
already recorded for `remove_carrier` in `gauge_conventions.md`, and the direct spatial analogue
of `aia.md`'s phase-origin pin $\delta_1=0$ for the temporal piston.

### 3.2 Curvature and the Newton step

Differentiating Eq. (C5) once more, $\partial^2F/\partial a_j\partial a_l = -\sum_k w_k\cos(r_k)
\,p_j p_l$. Writing $g_j \equiv \sum_k w_k\,p_j(x_k)\sin(r_k)$ for the gradient and

$$H_{jl} \equiv \sum_k w_k\cos(r_k)\,p_j(x_k)\,p_l(x_k) \tag{C6}$$

for (minus) the Hessian, a Newton step toward the maximizer solves

$$\boxed{H\,\Delta\mathbf a = \mathbf g} \tag{C7}$$

$H$ is a $\cos(r)$-**weighted** Gram matrix of the basis — the direct structural analogue of
`step_field_residuals.md` Eq. (E1)'s $w_n^2$-weighted $G^{(n)}$ — and inherits the same warning
given there (§8.3): orthonormal under the plain (unweighted) field inner product, which
$\{p_j\}$ is by construction (§1.3), does **not** imply orthogonal under this weight. Define

$$\kappa_c \equiv \operatorname{cond}(H) \tag{C8}$$

as the per-round diagnostic. $H$ is positive definite only where $w_k\cos(r_k) > 0$ dominates; in
the small-residual limit $H \to \sum_k w_k\,p_j p_l$, so the system is best conditioned exactly
where the current fit is already close, and can be indefinite far from it (§4.4).

### 3.3 The linearized/IRLS reading

Substituting $\sin(r) \to r$, $\cos(r) \to 1$ (small-residual limit) turns Eq. (C7) into
$\sum_k w_k p_j p_l\,\Delta a_l = \sum_k w_k p_j r_k$ — the ordinary weighted least-squares
projection of $\operatorname{wrap}(\Phi-P)$ onto $\{p_j\}$. This is why a "wrap the residual,
least-squares fit the wrapped residual, repeat" loop works at all once it is close to converged:
it is the linearization of Eq. (C7), differing from the exact Newton system only in dropping the
$\sin/\cos$ weighting of Eq. (C6).

## 4. Identifiability and ambiguity

1. **Piston mod $2\pi$.** Pinned by §3.1's $j=0$ row (Eq. C5b) — the global piston is fixed by
   the same weighted-circular-mean convention already in `gauge_conventions.md`.
2. **Continuum uniqueness.** Two coefficient vectors $\mathbf a \ne \mathbf a'$ give identical
   $F$ for every possible $\Phi$ only if $P(\cdot;\mathbf a) - P(\cdot;\mathbf a')$ is an integer
   multiple of $2\pi$ at *every* point of the field — and on a connected domain, a polynomial
   with that property (continuous, taking only values in $2\pi\mathbb Z$) must be a constant. So
   away from the piston (item 1), degrees $\ge 1$ are unique in the continuum.
3. **Grid aliasing.** On the actual integer-pixel grid, item 2's argument only has to hold at
   integer $(x,y)$, not everywhere — a strictly weaker condition, and the one that actually
   matters. A pure tilt already shows this: $e^{i2\pi f_x x}$ is unchanged by $f_x \to f_x+1$ at
   integer $x$, so a linear coefficient is identifiable only modulo one cycle/pixel
   (`gauge_conventions.md`'s existing `carrier.py` row). A **sufficient** condition ruling this
   out at any degree, the pointwise multi-dimensional Nyquist bound already invoked informally in
   `phase.carrier._estimate_curvature`'s docstring (local instantaneous frequency description):
   $$\max_{x,y}\,\big|\nabla P(x,y;\mathbf a)\big| < \pi \ \text{rad/pixel} \tag{C9}$$
   i.e. $P$ changes by less than half a cycle between neighbouring pixels everywhere on the
   field. This is sufficient, not necessary or tight — it is the condition under which no alias
   of *any* degree $\le M$ can reproduce the same sampled data.
4. **Non-convexity.** $F(\mathbf a) \le \sum_k w_k$, with roughly one local maximum per fringe
   swept along each basis direction once Eq. (C9) is violated anywhere over the search range. A
   Newton step from Eq. (C7) only finds the *nearest* maximum; an initializer must land within
   about half a fringe of the truth. This — not the fit itself — is what the FFT peak search in
   the current `remove_carrier` is genuinely for.
5. **Object/carrier degeneracy.** Any component of the true $\phi(x,y)$ that happens to lie in
   $\operatorname{span}\{p_1,\dots,p_J\}$ is removed along with the carrier, irreducibly — Eq.
   (C0) shows $\phi$ and $\phi_{\text{carrier}}$ enter $\Phi$ identically, so nothing in the data
   can attribute a low-order component to one rather than the other. $M$ is therefore a
   *modelling* choice trading carrier removal against sample-phase removal, not a fit-quality
   knob to raise until the residual stops shrinking.
6. **Sign branch.** Under $(\Phi,\delta)\to(-\Phi,-\delta)$ (`aia.md`'s cosine-is-even ambiguity,
   restated in `gauge_conventions.md`), every $a_j$ flips sign along with $\Phi$; nothing in
   §2–3 resolves it, consistent with `combine.py` running its sign-branch discriminant on the raw
   maps *before* carrier removal.

## 5. What this supersedes

Today's `phase.carrier.remove_carrier` runs three sequential stages that optimize no single
objective: an FFT peak search for a coarse tilt; a closed-form but gradient-domain refine
($\arg\sum_k w\,z_{k+1}\overline{z_k}$ over neighbouring-pixel pairs, not the field-wide resultant
of Eq. C4); and, for curvature, a block-wise regression of per-block local frequencies, run
*before* the final tilt pass rather than jointly with it, with accuracy set by the essentially
arbitrary `n_blocks` grid size, and hard-wired to degree $\le 2$ in raw monomials about pixel
$(0,0)$. Sections 2–3 replace all of it with one objective (Eq. C3) at arbitrary degree $M$,
fit jointly over every coefficient by Eq. (C7) — which demotes the FFT peak search to exactly
what item 4 of §4 says it is good for, a starting point for the Newton iteration, not the
estimate itself.

One consequence for the existing gauge table: `gauge_conventions.md`'s `remove_carrier` row
records the origin as pixel $(0,0)$ with unnormalized $x,y$ — different from the centroid,
unit-scaled convention §1.3 inherits from the step field. Adopting §1.3's basis resolves that
clash in favour of the step field's convention, once the code follows this derivation.

### 5.1 Reading `CarrierResult` against this section

Read-only mapping from today's output to this document's notation, not a claim that the numbers
agree: `kx, ky` (rad/pixel) and `fx, fy` (cycles/pixel, `kx = 2π·fx`) are the present code's tilt
estimate; `kxx, kyy, kxy` (rad/pixel²) its curvature estimate; `piston` its constant offset. All
correspond to *some* linear combination of the $a_j$ of Eq. (C1) at $M=2$ — but not numerically,
since the two use different bases, a different coordinate origin, and different scaling. They are
comparable only through the reconstructed field $P(x,y)$ each produces, never
coefficient-by-coefficient.

## 6. Relation to the step field

The carrier and the step field $\Delta_n(x,y)$ occupy the *same* polynomial span
$\{p_1,\dots,p_J\}$: by `step_field_residuals.md` Eq. (T3b), the frame-mean of the step-field
coefficients $\bar c_j \equiv \langle c_{jn}\rangle_n$ is, by construction, indistinguishable
from — and by convention *is* treated as — part of $\phi_{\text{carrier}}$, not the step field.
So a degree-$M$ carrier removal and a degree-$M$ step-field fit are not independent quantities:
`StepFieldParam.coeffs`, reported un-gauge-fixed per `gauge_conventions.md`, carries a carrier
piece that a subsequent carrier-removal fit (this document) would remove a second time from
$\Phi$.

`step_field_residuals.md` §5.4/§11 show an *uncorrected* step field biases the recovered $\Phi$ by
a ramp plus a second harmonic, and already call the ramp harmless "since that is exactly the
functional form of $\phi_{\text{carrier}}$" — removed along with it. Sections 2–3 here make that
claim precise in both directions: at $M\ge1$ the ramp lies in $\operatorname{span}\{p_j\}$ and is
absorbed into $a_j$ exactly by Eq. (C7), so the fitted carrier coefficients are *not* a clean
instrumental diagnostic once an uncorrected step field is present upstream. The $2\Phi$-harmonic
half of that same bias is **not** in the polynomial span and survives carrier removal untouched.

## 7. Summary table

| Quantity | Definition | Eq |
|---|---|---|
| Fitted carrier field | $P(x,y;\mathbf a) = \sum_{j=0}^J a_j\,p_j(x,y)$, basis from `_poly_basis` plus $p_0\equiv1/\sqrt{N_p}$ | C1 |
| Weighted complex field | $z_k = w_k e^{i\Phi_k}$; $= u_k - iv_k$ when $w_k=b_k$ | C2 |
| Objective | $F(\mathbf a) = \sum_k w_k\cos(\Phi_k-P_k) = \operatorname{Re}\sum_k z_k e^{-iP_k}$, maximize | C3 |
| Equivalent minimization | $\sum_k w_k\vert e^{i\Phi_k}-e^{iP_k}\vert^2 / 2 = \sum_k w_k - F(\mathbf a)$ | C3b |
| Piston-profiled objective | $\max\vert\sum_k w_k e^{i(\Phi_k-P'_k)}\vert$, $\hat\psi_0=\arg(\cdot)$ | C4 |
| Stationarity | $\sum_k w_k p_j \sin(r_k)=0$, all $j$ | C5 |
| Piston convention | $\arg\big(\sum_k w_k e^{ir_k}\big)=0$ | C5b |
| Newton system | $H_{jl}=\sum_k w_k\cos(r_k)p_jp_l$, $g_j=\sum_k w_k p_j\sin(r_k)$, $H\Delta\mathbf a=\mathbf g$ | C6/C7 |
| Conditioning | $\kappa_c=\operatorname{cond}(H)$ | C8 |
| Identifiability (sufficient) | $\max\vert\nabla P\vert < \pi$ rad/pixel | C9 |

## 8. Assumptions used

1. **A converged solve.** $\Phi(x,y)$ (and, where used, $b(x,y)$/$u,v$) come from a piston-model
   or step-field-corrected AIA solve already consistent with `interference_model.md` Eq. (17) —
   this document treats $\Phi$ as given data, not as something it re-derives.
2. **The carrier is genuinely low-order.** $\phi_{\text{carrier}}(x,y)$ (and whatever of
   $\phi,\phi_{\text{inst}}$ is fit alongside it) is well approximated by a degree-$\le M$
   polynomial — §4.5's degeneracy is the cost of this assumption being too generous.
3. **Weight tracks reliability.** $w_k \ge 0$, and larger where the fringe is trustworthy (higher
   modulation, inside a valid mask) — required for Eq. (C5)/(C7)'s stationarity to weight the fit
   sensibly, not merely for it to be well defined.
4. **Sub-Nyquist sampling of the fitted field.** Eq. (C9) — needed for §4's identifiability
   argument and for the FFT-based initializer (§4.4) to land in the correct basin.
5. **Sign branch already resolved.** §4.6 — this document does not fix $(\Phi,\delta) \to
   (-\Phi,-\delta)$; that is `combine.py`/`reference.py`'s responsibility, run beforehand.
