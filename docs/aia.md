# Advanced Iterative Algorithm (AIA)

This document derives the Advanced Iterative Algorithm (AIA) for
phase-shifting interferometry (Wang & Han 2004; enhanced per Chen & Kemao,
*Optics Express* 27(26), 37634-37651, 2019), starting from the per-frame
model of `docs/interference_model.md`.

## Starting point

`docs/interference_model.md`'s per-frame model (that document's Eq. 17) is

$$I_n(x, y) = \alpha_n\Big[a(x, y) + g_n\,b(x, y)\,\cos\big(\Phi(x, y) + \delta_n + \Delta_n(x, y)\big)\Big], \qquad n = 1 \dots N$$

AIA works on the **uniform-piston limit** of this model (`interference_model.md`
Eq. 20, $\Delta_n(x, y) \equiv 0$): the phase step from frame to frame must be a
single scalar $\delta_n$, not a field. If $\Delta_n \not\equiv 0$ in the actual
data, AIA's recovered $\Phi$ and $b$ carry a systematic, low-order bias that
does not shrink with more frames — see `docs/step_field_residuals.md` for the
size of that bias and `phase.methods.step_field.aia_step_field`, which refines
an AIA solution against a spatially-varying step instead of assuming it away.

The per-frame source-power factor $\alpha_n$ is not solved for either: since
$\langle a\rangle \gg \langle b\rangle$ once the field spans many fringes, the
frame mean $\langle I_n\rangle \approx \alpha_n\langle a\rangle$, so $\alpha_n$
is estimated directly from each frame's mean intensity and divided out before
the fields below are fit (`PhaseSolver._normalize`), fixed by the convention
$\operatorname{median}(\alpha) = 1$. Whatever residual frame-to-frame level
that normalization leaves behind is small and is absorbed into the per-frame
offset $c_n$ introduced below, rather than treated as a separate unknown.

With $\alpha_n$ divided out and $\Delta_n \equiv 0$, AIA solves

$$I_n(x, y) = a(x, y) + g_n\,b(x, y)\,\cos\big(\Phi(x, y) + \delta_n\big) \tag{1}$$

for the static fields $a(x,y)$, $b(x,y)$, $\Phi(x,y)$ and the per-frame phase
steps $\delta_n$ and gains $g_n$, without requiring $\delta_n$ or $g_n$ to be
known in advance. (The code calls the recovered phase `phi` for brevity; it is
$\Phi$ here, matching `interference_model.md` — the sample-induced phase $\phi$
alone is only obtained afterward, by carrier removal and reference
subtraction.)

## Quadrature form

Expanding the cosine in Eq. (1),

$$\cos\big(\Phi + \delta_n\big) = \cos\Phi\cos\delta_n - \sin\Phi\sin\delta_n$$

and defining the quadrature components

$$u(x,y) = b(x,y)\cos\Phi(x,y), \qquad v(x,y) = -b(x,y)\sin\Phi(x,y) \tag{2}$$

Eq. (1) becomes

$$I_n(x, y) = a(x, y) + g_n\big[u(x, y)\cos\delta_n + v(x, y)\sin\delta_n\big] \tag{3}$$

Writing $P_n = g_n\cos\delta_n$, $Q_n = g_n\sin\delta_n$, this is
$I_n = a + P_n u + Q_n v$: linear in $(a, u, v)$ for fixed $(P_n, Q_n)$, and
linear in $(P_n, Q_n)$ for fixed $(a, u, v)$ — but not linear in both sets of
unknowns at once, since their product appears. That bilinearity rules out
closed-form recovery, and motivates solving by alternating least squares.

## Objective function

AIA minimizes the sum of squared residuals against Eq. (3), extended with a
free per-frame offset $c_n$ that absorbs the background level and whatever
frame-to-frame drift the $\alpha_n$ normalization did not fully remove:

$$\mathcal{L}(a, u, v, \{c_n\}, \{\delta_n\}, \{g_n\}) = \sum_{n=1}^N \sum_{x,y} \Big[I_n^{\text{meas}}(x, y) - a(x, y) - c_n - g_n\big(u(x, y)\cos\delta_n + v(x, y)\sin\delta_n\big)\Big]^2 \tag{4}$$

This is stated in terms of $(a, u, v)$ rather than $(a, b, \Phi)$ since that is
the parameterization each linear sub-problem below actually solves for;
recovering $(b, \Phi)$ from $(u, v)$ is Eq. (9) below.

## Alternating least squares

AIA alternates two linear solves until $\delta_n$ (and $g_n$) stop changing
(or an iteration limit is reached):

### 1. Pixel step

Because $\mathcal{L}$ (Eq. 4) sums over pixels with no cross terms between
them, minimizing it over the fields $(a, u, v)$ decomposes into an
independent minimization at each pixel, each a sum over frames only:

$$\mathcal{L}_{x,y}(a,u,v) = \sum_{n=1}^N \Big[\big(I_n^{\text{meas}}(x,y) - c_n\big) - a - g_n\big(u\cos\delta_n + v\sin\delta_n\big)\Big]^2$$

With $\{c_n, \delta_n, g_n\}$ fixed, minimizing $\mathcal{L}_{x,y}$ over
$(a, u, v)$ at a given pixel is a linear regression of that pixel's $N$
frame values, each with $c_n$ subtracted, onto the design matrix

$$A = \begin{bmatrix} 1 & g_1\cos\delta_1 & g_1\sin\delta_1 \\ \vdots & \vdots & \vdots \\ 1 & g_N\cos\delta_N & g_N\sin\delta_N \end{bmatrix} \tag{5}$$

solved (by pseudoinverse, across every pixel at once) for $(a, u, v)$ —
exactly $\arg\min_{a,u,v}\mathcal{L}$ with $\{c_n, \delta_n, g_n\}$ held fixed,
by the separability above. $A^\top A$ is the pixel-step normal matrix, $A_p$
below.

The resulting $(u, v)$ are only determined up to the reparametrization
described in "Identifiability and gauge" below; before the frame step, they
are whitened (Eq. 14) to remove the part of that freedom the frame step would
otherwise leave unresolved.

### 2. Frame step

With $(a, u, v)$ now fixed and whitened, $\mathcal{L}$ (Eq. 4) also sums over
frames with no cross terms between them, so minimizing it over
$\{c_n, \delta_n, g_n\}$ decomposes into an independent minimization at each
frame, each a sum over pixels only:

$$\mathcal{L}_n(c_n, \delta_n, g_n) = \sum_{x,y} \Big[\big(I_n^{\text{meas}}(x,y) - a(x,y)\big) - c_n - g_n\big(u(x,y)\cos\delta_n + v(x,y)\sin\delta_n\big)\Big]^2$$

— a regression of $I_n - a$ onto $\{1, u, v\}$. What AIA actually solves is
not this: the frame step regresses the *measured* frame directly, with no
$a(x,y)$ term at all —

$$\mathcal{L}'_n(c_n, P_n, Q_n) = \sum_{x,y} \Big[I_n^{\text{meas}}(x, y) - c_n - P_n\,u(x, y) - Q_n\,v(x, y)\Big]^2 \tag{6}$$

replacing the field $a(x,y)$, subtracted explicitly above, with the single
free scalar $c_n$ — a linear regression, at each frame independently, of the $P = H \times W$
pixel values onto the design matrix

$$B = \begin{bmatrix} 1 & u(x_1,y_1) & v(x_1,y_1) \\ \vdots & \vdots & \vdots \\ 1 & u(x_P,y_P) & v(x_P,y_P) \end{bmatrix} \tag{7}$$

— the same matrix for every frame, since it is built directly from $(u, v)$
and does not carry $g_n$.

Solving frame $n$'s regression gives coefficients $(c_n, P_n, Q_n)$ with
$P_n \approx g_n\cos\delta_n$, $Q_n \approx g_n\sin\delta_n$, so that

$$\delta_n = \operatorname{atan2}(Q_n, P_n) \tag{8}$$

recovers $\delta_n$ exactly regardless of $g_n$ (for $g_n > 0$): a common
positive scale factor on both $P_n$ and $Q_n$ cancels in
$\operatorname{atan2}$. $B^\top B$ is the frame-step normal matrix, $A_{ps}$
below. See "Why the background field can be dropped" below for what
dropping $a(x,y)$ from Eq. (6) costs and why it is safe.

### Why the background field can be dropped

Write the pixel inner product $\langle f, g\rangle = \sum_{x,y} f(x,y)\,g(x,y)$.
The design $B$ (Eq. 7) spans $\{1, u, v\}$, a fixed 3-dimensional subspace of
pixel-space that does not depend on $n$. Decompose $a$ against it:

$$a(x,y) = a_0 + a_u\,u(x,y) + a_v\,v(x,y) + a_\perp(x,y), \qquad \langle a_\perp, 1\rangle = \langle a_\perp, u\rangle = \langle a_\perp, v\rangle = 0 \tag{6a}$$

with $(a_0, a_u, a_v)$ the unique least-squares coefficients of $a$ on
$\{1, u, v\}$, i.e. the solution of $B^\top B\,(a_0,a_u,a_v)^\top = B^\top a$.
Nothing here depends on $n$: $a$ is one field shared by every frame, so this
is a single, fixed decomposition, not one per frame.

Substituting Eq. (6a) into the true per-frame model,

$$I_n = a + c_n + P_n u + Q_n v = (c_n + a_0) + (P_n+a_u)\,u + (Q_n+a_v)\,v + a_\perp \tag{6b}$$

and since $a_\perp$ is, by construction, orthogonal to every column of $B$,
it contributes nothing to the normal equations $B^\top I_n$ that Eq. (6)
solves. The regression's exact minimizer is therefore

$$\hat c_n = c_n + a_0, \qquad \hat P_n = P_n + a_u, \qquad \hat Q_n = Q_n + a_v \tag{6c}$$

for every frame, regardless of the size or shape of $a_\perp$. Three
consequences follow.

$a_\perp$ — everything about $a$ that is neither constant nor aligned with
the fringe pattern, i.e. essentially all of its spatial structure — is
invisible to the fit: Eq. (6c) holds no matter how large $a_\perp$ is. It
shows up only in $\mathcal{L}'_n$'s own residual, which the frame step
discards; that residual is not a fit-quality signal, since it is not what
$\sigma$ (Eq. 10) measures — $\sigma$ is the *pixel*-step residual, and the
pixel step's design (Eq. 5) does include $a$ explicitly.

$a_0$ is the same constant for every frame, since it depends only on $a$
and $(u,v)$, neither of which varies with $n$. It is removed exactly by the
gauge fix $\bar c_n = 0$ (Eq. 12).

$a_u, a_v$ displace every frame's $(\hat P_n, \hat Q_n)$ by the same
fixed vector, and this is the only piece of $a$ that reaches
$\delta_n = \operatorname{atan2}(\hat Q_n, \hat P_n)$ (Eq. 8). It is a
displacement along the shift freedom of Eq. (14a), not a new one.

$(a_u, a_v)$ measure the overlap of the background with the fringe
pattern $u = b\cos\Phi$, $v=-b\sin\Phi$ — small whenever $a$ varies slowly
compared to a fringe period, the same regime (many fringes across the field,
$\langle a\rangle \gg \langle b\rangle$) that already underlies the
$\alpha_n$ estimate in "Starting point" above. They grow exactly when that
regime breaks down: too little phase coverage for $u, v$ to oscillate, which
is what $\kappa_{ps}$ (see "Accuracy diagnostics" below) already flags.

The pixel step is the exact minimizer of $\mathcal{L}$ (Eq. 4) over
$(a, u, v)$ with $(c_n, \delta_n, g_n)$ fixed. The frame step is the exact
minimizer of Eq. (6), which by Eq. (6c) equals the minimizer of $\mathcal{L}$
over $(c_n, \delta_n, g_n)$ with $(a, u, v)$ fixed, displaced by the
frame-independent offset $(a_0, a_u, a_v)$ — $a_0$ removed exactly
by Eq. (12), $(a_u, a_v)$ a step along the shift freedom of Eq. (14a).
Whitening between the two steps changes neither one's objective — it
reparametrizes the same 3-D column space $\{1, u, v\}$, and the frame step is
an unconstrained regression against whichever basis of that space it is
handed. Wherever that displacement is negligible, the two steps minimize one
shared objective, the joint residual decreases monotonically every
iteration, and a stall (the iteration limit reached without meeting the
tolerance) is a genuine local optimum, not the two steps chasing different
targets.

### Phase-origin convention

Eq. (3) has a global offset ambiguity: adding a constant to every $\delta_n$
while subtracting it from $\Phi$ reproduces the same data. Each iteration
re-references $\delta_n \to \delta_n - \delta_1$ so $\delta_1 = 0$, fixing the
split.

### Convergence

Iterate steps 1–2 until the largest per-frame change in $\delta_n$ (and, when
$g_n$ is being fit, in $g_n$) between iterations falls below a tolerance, or
an iteration limit is reached. The final phase map and fringe amplitude
follow directly from $u, v$:

$$\Phi(x,y) = \operatorname{atan2}\big(-v(x,y),\, u(x,y)\big), \qquad b(x,y) = \sqrt{u(x,y)^2 + v(x,y)^2} \tag{9}$$

## Accuracy diagnostics

Chen & Kemao (2019) show AIA's accuracy is governed by how well-conditioned
the two normal matrices $A_p = A^\top A$ and $A_{ps} = B^\top B$ are:

- $\kappa_p = \operatorname{cond}(A_p)$ — how well the phase-shift
  distribution $\{\delta_n\}$ conditions the pixel-step solve. Enters the
  accuracy prediction (Eq. 10) directly.
- $\kappa_{ps} = \operatorname{cond}(A_{ps})$, evaluated on the *normalized*
  unit-circle design (columns $\cos\Phi, \sin\Phi$, amplitude divided out) —
  how well the recovered phase pattern covers the unit circle. Bounded below
  by 2, achieved when $\Phi$ is evenly distributed over $2\pi$. Large values
  mean the field spans too little phase (less than roughly one fringe) for
  the frame step to reliably separate $\delta_n$ from noise.

and predict the RMS phase error as

$$\sigma_\Phi \approx 0.42\,\big(\sqrt{\kappa_p} + 2\big)\,\frac{\sigma}{b}\,\frac{1}{\sqrt{N}} \tag{10}$$

where $\sigma$ is the RMS residual of the final pixel-step fit and $b$ is the
median fringe amplitude. A poorly conditioned acquisition ($\kappa_p$ or
$\kappa_{ps}$ large) should not be trusted even if the iteration reports
convergence. Eq. (10) is an empirical fit, accurate near the well-conditioned regime it
was validated on but not an exact result; see "Direct phase-error computation" below for
the exact first-order expression it approximates, its bounds, and the conditions under
which the two agree.

A third diagnostic, $g_{\min\text{-ratio}} = \min(g_n)/\operatorname{median}(g_n)$,
flags a frame whose data is nearly uncorrelated with the recovered fringe
pattern: its $(P_n, Q_n)$ (and hence $\delta_n$) is poorly determined even
though the fit as a whole may look converged, and such a frame is a candidate
for dropping from the acquisition rather than trusting.

## How many frames are needed

Stack the per-frame model, $I_n(x,y) = a(x,y) + c_n + P_n\,u(x,y) + Q_n\,v(x,y)$,
into an $N \times P$ matrix $I$ of all measured pixel values ($P = H \times W$):

$$I = \mathbf{1}_N\,a^\top + c\,\mathbf{1}_P^\top + P\,u^\top + Q\,v^\top \tag{11}$$

where $\mathbf{1}_N, \mathbf{1}_P$ are all-ones vectors and $a, u, v \in \mathbb{R}^P$,
$c, P, Q \in \mathbb{R}^N$ are the fields and per-frame coefficients above —
a sum of four rank-1 (outer-product) terms, so $I$ has rank at most 4.

For $N \le 4$, the four frame-side vectors $\{\mathbf{1}_N, c, P, Q\}$ live in
$\mathbb{R}^N$ with $N \le 4$ dimensions, so they can be chosen to span all of
$\mathbb{R}^N$: *any* $N \times P$ data matrix can then be written in this
form for some choice of the per-frame parameters, and the model imposes no
constraint the data could fail to satisfy. The per-frame unknowns are then
not determined by the data at all. Only once $N \ge 5$ does the model
actually constrain anything — the frame-side vectors are confined to a
4-dimensional subspace of a space with more than 4 dimensions — so **the
joint solve for $(\delta_n, g_n)$ together with the free offset $c_n$ needs
at least 5 frames.**

This is a necessary condition, not a sufficient one: even with $N \ge 5$, a
poorly distributed $\{\delta_n\}$ (large $\kappa_p$) or a field with too
little phase variation (large $\kappa_{ps}$) still leaves the per-frame
parameters weakly determined — see "Accuracy diagnostics" above.

## Identifiability and gauge

With $(a, u, v)$ and $(c_n, \delta_n, g_n)$ both free, Eq. (4)'s minimizer is
not unique: several exact symmetries of the model let one solution be
transformed into another that fits the data identically. Two are already
resolved above — the phase origin ($\delta_1 = 0$) and, implicitly, the sign
branch $(\Phi, \delta) \to (-\Phi, -\delta)$ (cosine is even; not pinned by
the alternation itself, so two independent solves may land on opposite
branches — resolved downstream, see `docs/gauge_conventions.md`). Two more
are specific to fitting $c_n$ and $g_n$ jointly.

The gain and offset are only defined up to a shared scale and a shared shift:

$$g_n = \sqrt{P_n^2 + Q_n^2}, \qquad g_n \leftarrow g_n / \operatorname{median}(g_n), \qquad c_n \leftarrow c_n - \overline{c_n} \tag{12}$$

fixes $\operatorname{median}(g) = 1$ and $\overline{c_n} = 0$ — the pixel step
(Eq. 5) is then re-solved against $I - c$, so both steps minimize the same
$\mathcal{L}$ rather than drifting apart.

### A gauge freedom that only appears once g_n is free

With $g_n \equiv 1$, $(P_n, Q_n)$ is constrained to the unit circle — a single
degree of freedom per frame. Once $g_n$ is free, $(P_n, Q_n)$ is an
*unconstrained* point in the plane, and Eq. (3)/(6)'s model is invariant
under **any** invertible linear reparametrization

$$(u, v) \to (u, v)\,M, \qquad (P, Q) \to (P, Q)\,M^{-\top} \tag{13}$$

for a $2\times 2$ matrix $M$, since $\{1, u, v\}$ and $\{1, uM, vM\}$ span the
same subspace for any invertible $M$, not just a rotation. Left alone, the
alternating solve can converge to *any* basis of that subspace — fitting $I$
exactly as well (often better, since a generic basis has more freedom to
explain noise) — without $(P_n, Q_n)$ tracing
$(g_n\cos\delta_n, g_n\sin\delta_n)$ for any physically meaningful $\delta_n$.

The fix (`phase.methods.aia._whiten_uv`, run on $(u, v)$ after every pixel
step) rescales/shears $(u, v)$ so that

$$\sum_{x,y} u^2 = \sum_{x,y} v^2, \qquad \sum_{x,y} uv = 0 \tag{14}$$

which collapses the residual gauge from all of $GL(2,\mathbb{R})$ down to
just rotations and reflections, $O(2)$ — exactly the ambiguity already
resolved by the phase origin and sign-branch conventions above, rather than
that plus a two-parameter shear/scale family on top. Total pixel-sum energy
(the trace of $(u,v)$'s $2\times 2$ Gram matrix) is preserved, so only
Eq. (12)'s own normalization changes $g$'s overall scale.

### A residual shift freedom

Eq. (13)'s invariance is linear; with $c_n$ free the model is also invariant
under an affine shift that Eq. (14) does not touch:

$$u \to u + s, \qquad a \to a - s\,\overline{P}, \qquad c_n \to c_n - s\,(P_n - \overline{P}) \tag{14a}$$

for any scalar $s$ (and symmetrically for $v$ against $Q$), since this leaves
$a + c_n + P_n u + Q_n v$ unchanged at every pixel and frame — an additive
constant on $u$ (or $v$) is compensated by a matching per-frame term in
$c_n$. This shift does move the recovered $\Phi = \operatorname{atan2}(-v, u)$,
and it is pinned by none of the conventions above: whitening (Eq. 14) fixes
only the linear part of the freedom in Eq. (13), and the phase-origin and
sign-branch conventions constrain $\delta_n$ and the overall sign, not an
additive shift of $(u, v)$ itself. In practice the alternation starts from
$c_n \equiv 0$ and Eq. (14a) is a flat direction of $\mathcal{L}$ rather than
a descent direction, so the iteration does not drift along it — but no step
in the algorithm resolves it explicitly. The frame step's own
$(a_u, a_v)$ displacement (Eq. 6c) is exactly a step along this
direction, injected fresh every iteration; its size is bounded by the
background's overlap with the fringe pattern rather than by anything in the
iteration's dynamics.

## Known gain as a special case

If $g_n$ is measured independently of this solve (e.g. from a calibration
shot), it can be held fixed rather than fit jointly: $c_n$ is no longer
needed (the pixel step's fixed $a(x,y)$ absorbs the background exactly, as
in the textbook scheme), the design matrix $A$ (Eq. 5) is built from the
known $g_n$, and the whitening step (Eq. 14) is unnecessary, since holding
$g_n$ fixed already constrains $(P_n, Q_n)$ to the unit circle and removes
the gauge freedom of Eq. (13). With $c_n$ dropped, Eq. (11) loses one of its
four rank-1 terms, and the frame-count argument above drops one frame in
lockstep — recovering $(a, u, v)$ and $\{\delta_n\}$ from unknown, unit-gain
phase steps needs only $N \ge 4$.

## Direct phase-error computation

Eq. (10) is an empirical fit to simulated data, not a derivation from AIA's own normal
equations: it gives one scalar for the whole field, and its own fit does not separately
account for the extra error AIA carries by fitting $\delta_n$ and $g_n$ from the data
rather than assuming them known. Two things can be said in its defense: for
well-conditioned, evenly-spaced acquisitions it is accurate to within 1.4% of
the exact result, and its $\sqrt{\kappa_p}$ scaling is the right worst-case shape, not an
arbitrary choice. But $\kappa_p$ alone cannot be the *exact* predictor Eq. (10) implies:
rescaling $A$'s constant column (equivalently, solving for $a/3$ instead of $a$) leaves
the recovered $\Phi$ bit-for-bit identical while moving $\kappa_p$ from 2.1 to 18.8 —
nearly the code's own `kappa_p > 20` warning threshold — for the same fit. And a single
scalar cannot express the $2\Phi$-periodic modulation the exact error carries whenever
$\{\delta_n, g_n\}$ is anisotropic (see "Bounds and limiting cases" below).

This section derives the exact, first-order phase- and amplitude-error covariance
directly from AIA's own normal equations, in three stages: (1) $\delta_n$ and $g_n$ known
exactly, only $(a,u,v)$ fit — the case Eq. (9) applies to unmodified; (2) $\delta_n$
fitted jointly with $g_n$ fixed (`fit_gain=False`); (3) $\delta_n$ and $g_n$ both fitted
jointly (`fit_gain=True`). Throughout, $N_p = H\times W$ is the pixel count (kept
distinct from $P_n$, the frame-step coefficient of Eq. 6) and
$\mathbf i(x,y)\in\mathbb R^N$ denotes the per-pixel vector of measured frame
intensities, $i_n(x,y) = I_n^{\text{meas}}(x,y)$.

### Noise model

The camera is modeled as adding zero-mean noise to the intensity, independent from pixel
to pixel and frame to frame — true of read noise, shot noise and dark current alike, since
each is an independent event per pixel per readout — but not necessarily of equal
variance:

$$I_n^{\text{meas}}(x,y) = I_n(x,y) + \varepsilon_n(x,y), \qquad
\operatorname{E}[\varepsilon_n(x,y)] = 0, \qquad
\operatorname{Cov}\big(\varepsilon_n(x,y),\,\varepsilon_m(x',y')\big) = \sigma_n^2(x,y)\,\delta_{nm}\,\delta_{xx'}\delta_{yy'} \tag{15}$$

Only these first two moments are used below — Stages 1 and 2 assume no particular
distribution. What independence does rule out is correlation *within* a frame or across
frames: banded or common-mode readout noise, inter-pixel crosstalk, and source-level
intensity or speckle fluctuation correlated across the field (the frame-common part of
which is what $\alpha_n$ already absorbs, "Starting point" above).

### Stage 1: $\delta_n$, $g_n$ known — pixel-step covariance

With $\delta_n, g_n$ fixed at their true values, Eq. (5)'s pixel step is, for a single
pixel, the explicit linear solve

$$(a, u, v)^\top = A_p^{-1}A^\top\mathbf i, \qquad A_p = A^\top A \tag{16}$$

($A$ full rank here, so $A_p^{-1}$ — not the pseudoinverse `aia_pixel_step` uses in
general.) With Eq. (15)'s noise model, $\mathbf i=\mathbf i^{\text{true}}+\varepsilon$,
$\operatorname{Cov}(\varepsilon)=D\equiv\operatorname{diag}(\sigma_n^2(x,y))$, so the
estimate's error $(e_a,e_u,e_v)^\top = A_p^{-1}A^\top\varepsilon$ has the sandwich
covariance

$$\operatorname{Cov}\big((e_a,e_u,e_v)^\top\big) = A_p^{-1}A^\top D\,A\,A_p^{-1} \tag{17}$$

which collapses to

$$\operatorname{Cov}\big((e_a,e_u,e_v)^\top\big) = \sigma^2 A_p^{-1} \tag{17a}$$

exactly when $D=\sigma^2 I_N$ — variance equal across the $N$ *frames* at that pixel, a
weaker requirement than equal variance across pixels. Write $S$ for the
$\{u,v\}\times\{u,v\}$ block of $A_p^{-1}$ (dropping the row/column for $a$); Eq. (17a)
gives $\operatorname{Cov}((e_u,e_v)^\top) = \sigma^2 S$ directly, with $\sigma$ read as a
per-pixel quantity $\sigma(x,y)$ wherever it appears below.

Differentiating Eq. (9), $\Phi=\operatorname{atan2}(-v,u)$, $b=\sqrt{u^2+v^2}$, to first
order in $(e_u,e_v)$:

$$e_\Phi = -\frac{\sin\Phi\,e_u+\cos\Phi\,e_v}{b}, \qquad
e_b = \cos\Phi\,e_u-\sin\Phi\,e_v \tag{18}$$

so, with $w=(\sin\Phi,\cos\Phi)^\top$ and $\tilde w=(\cos\Phi,-\sin\Phi)^\top$ (an
orthonormal pair),

$$\sigma_\Phi^2(x,y) = \frac{\sigma^2}{b(x,y)^2}\,w^\top S\,w, \qquad
\sigma_b^2(x,y) = \sigma^2\,\tilde w^\top S\,\tilde w \tag{19}$$

### Closed form for $S$

Writing out $A_p$ (Eq. 16) from $A$'s columns (Eq. 5) explicitly,

$$A_p = N\begin{bmatrix} 1 & R_c & R_s \\ R_c & \tfrac12(\langle g^2\rangle_n+R_c^{(2)}) & \tfrac12 R_s^{(2)} \\ R_s & \tfrac12 R_s^{(2)} & \tfrac12(\langle g^2\rangle_n-R_c^{(2)}) \end{bmatrix} \tag{20}$$

with the $g$-weighted first- and second-harmonic circular moments of the phase-step
distribution,

$$R_c=\langle g_n\cos\delta_n\rangle_n, \quad R_s=\langle g_n\sin\delta_n\rangle_n, \qquad
R_c^{(2)}=\langle g_n^2\cos2\delta_n\rangle_n, \quad R_s^{(2)}=\langle g_n^2\sin2\delta_n\rangle_n \tag{20a}$$

Eliminating the $a$-row of $A_p^{-1}$ by Schur complement gives $S$ in closed form:

$$S = \frac1N\,C^{-1}, \qquad
C = \big\langle g_n^2\,d_n d_n^\top\big\rangle_n - \big\langle g_n d_n\big\rangle_n\big\langle g_n d_n\big\rangle_n^\top, \qquad
d_n = \begin{pmatrix}\cos\delta_n\\ \sin\delta_n\end{pmatrix} \tag{21}$$

$C$ is exactly the frame-side covariance matrix of the $g$-weighted unit-circle directions
$\{g_nd_n\}$: solving jointly for the background $a$ costs exactly a mean-centering of
those directions — the same marginalization argument as "Why the background field can be
dropped" above, applied on the frame axis instead of the pixel axis. Its trace,
$\operatorname{tr}C=\langle g^2\rangle_n-|R|^2$ with $|R|=\sqrt{R_c^2+R_s^2}$, shrinks as
$|R|$ approaches its triangle-inequality bound $\langle g\rangle_n$ (little phase
diversity): the same failure mode $\kappa_{ps}$ already flags, seen from the frame-step
side of the same coupling.

Combining Eqs. (19), (21),

$$\boxed{\;\sigma_\Phi^2(x,y) = \frac{\sigma^2}{N\,b(x,y)^2}\,w^\top C^{-1}w\;}, \qquad
\sigma_b^2(x,y) = \frac{\sigma^2}{N}\,\tilde w^\top C^{-1}\tilde w \tag{22}$$

and, since $w,\tilde w$ are orthonormal, $b^2\sigma_\Phi^2+\sigma_b^2$ drops the
$\Phi$-dependence entirely,

$$b(x,y)^2\,\sigma_\Phi^2(x,y) + \sigma_b^2(x,y) = \frac{\sigma^2}{N}\operatorname{tr}\big(C^{-1}\big) \tag{23}$$

— the same total error budget at every pixel, split between phase and amplitude by how
$\Phi$ aligns with $C$'s principal axes. $\sigma_b$ is not currently reported by
`AIAParam`, but Eq. (23) makes it a free byproduct of $S$ if the contrast map's own
uncertainty is ever needed downstream.

### Bounds and limiting cases

$w$ is a unit vector, so Eq. (22) is bounded by $C$'s eigenvalues
$\lambda_\pm=\tfrac12\big(\operatorname{tr}C\pm\sqrt{(\operatorname{tr}C)^2-4\det C}\big)$:

$$\frac{\sigma}{b\sqrt{N\lambda_+}} \;\le\; \sigma_\Phi \;\le\; \frac{\sigma}{b\sqrt{N\lambda_-}} \tag{24}$$

with equality when $\Phi$ aligns with an eigenvector of $C$ — so **whenever $C$ is
anisotropic ($\lambda_+\ne\lambda_-$), $\sigma_\Phi(x,y)$ is not constant across pixels of
equal $b$: it carries a $2\Phi$-periodic modulation**, a prediction no scalar RMS (Eq. 10
included) can express, and one directly checkable against a real acquisition's residual
map.

Averaging Eq. (22) over a field with $\Phi$ uniformly distributed over $2\pi$
($\langle\sin^2\Phi\rangle=\langle\cos^2\Phi\rangle=\tfrac12$,
$\langle\sin\Phi\cos\Phi\rangle=0$) collapses $w^\top C^{-1}w$ to
$\tfrac12\operatorname{tr}(C^{-1})=\operatorname{tr}(C)/(2\det C)$:

$$\big\langle\sigma_\Phi^2\big\rangle_\Phi = \frac{\sigma^2}{2Nb^2}\,\frac{\operatorname{tr}C}{\det C} \tag{25}$$

This holds $\sigma$, $b$ fixed and averages only $\Phi$ — it is not yet a true field
average, since real pixels carry their own $\sigma_0(x,y)$ and $b(x,y)$ together. Under
the same "$\Phi$ decorrelated from local illumination" approximation used elsewhere (many
fringes spread across whatever illumination gradient exists), the actual field average
keeps $\sigma_0^2/b^2$ together under one spatial average rather than factoring it:

$$\big\langle\sigma_\Phi^2\big\rangle_{\text{field}} = \frac{1}{2N}\Big\langle\frac{\sigma_0^2(x,y)}{b(x,y)^2}\Big\rangle_{x,y}\,\frac{\operatorname{tr}C}{\det C} \tag{25a}$$

Factoring this as $\langle\sigma_0^2\rangle\langle1/b^2\rangle$ overstates it whenever
$\sigma_0$ and $b$ are positively correlated across the field — which shot noise makes
the rule rather than the exception, since both track local illumination.

**Ideal case** — evenly-spaced $\delta_n=2\pi n/N$ ($N\ge3$), unit gain $g_n\equiv1$: both
harmonics vanish, $R=R^{(2)}=0$, so $C=\tfrac12 I_2$ exactly and Eq. (22) reduces to

$$\sigma_\Phi = \sqrt{\frac2N}\,\frac{\sigma}{b} \tag{26}$$

independent of $\Phi$ (isotropic $C$). This configuration has $\kappa_p=2$ always, so
Eq. (10) evaluates to $0.42(\sqrt2+2)/\sqrt N = [0.42(1+\sqrt2)]\sqrt{2/N} \approx
1.014\,\sigma_\Phi$ — the 1.4% mentioned above, independent of $N$ since both sides scale
as $1/\sqrt N$, and the cleanest evidence Eq. (10) is a good empirical fit near the
well-conditioned regime it was validated on, not a derived exact result.

**Bound in terms of $\kappa_p$.** By Cauchy interlacing, $S$'s largest eigenvalue is
bounded by $A_p$'s smallest, $\lambda_{\max}(S)\le1/\lambda_{\min}(A_p)=
\kappa_p/\lambda_{\max}(A_p)$; and since $\lambda_{\max}(A_p)\ge\operatorname{tr}(A_p)/3=
N(1+\langle g^2\rangle_n)/3$ (Eq. 20's trace),

$$\sigma_\Phi \;\le\; \frac{\sigma}{b}\sqrt{\frac{3\,\kappa_p}{N(1+\langle g^2\rangle_n)}} \tag{27}$$

which at $g\equiv1$ is $1.22\sqrt{\kappa_p}\,\sigma/(b\sqrt N)$ — confirming Eq. (10)'s
$\sqrt{\kappa_p}$ growth is the right worst-case shape, even though $\kappa_p$ alone
cannot fix the constant in front of it, as the rescaling argument above shows.

### Shot noise

Eq. (15)'s per-pixel variance $\sigma_n^2(x,y)$ is, in general, genuinely different from
frame to frame at a fixed pixel: shot noise depends on that frame's own illumination
$I_n(x,y) = a(x,y) + g_n\,b(x,y)\cos(\Phi(x,y)+\delta_n)$, which is fringe-modulated, not
just the static background. $\sigma_n(x,y)$ is not derived analytically here — it comes
from a calibrated per-frame noise model (e.g. a photon transfer curve), evaluated on the
actual measured $I_n^{\text{meas}}(x,y)$, and is treated as external data the pipeline is
handed, not assumed to follow any particular closed form.

Eq. (17a)'s exact collapse to $\sigma^2A_p^{-1}$ needs $D\propto I_N$ — equal variance
across frames at a pixel — which real, frame-varying $\sigma_n(x,y)$ only satisfies
approximately. The natural single-number stand-in for use there is the quadrature mean
across frames,

$$\sigma_0(x,y)^2 = \frac{1}{N}\sum_{n=1}^N \sigma_n(x,y)^2 \tag{27a}$$

giving, via Eq. (17a), Eq. (22) with $\sigma\to\sigma_0(x,y)$ — a per-pixel noise map
paired pointwise with $b(x,y)$:

$$\sigma_\Phi(x,y) = \frac{\sigma_0(x,y)}{b(x,y)\sqrt N}\sqrt{w^\top C^{-1}w} \tag{27b}$$

How good this approximation is depends on how much $\sigma_n(x,y)^2$ spreads around its
mean $\sigma_0(x,y)^2$ across frames — directly checkable from the calibrated per-frame
noise maps themselves, rather than bounded analytically the way a specific noise model
would allow.

**Idealized special case.** If the noise is instead assumed to follow the textbook linear
shot-plus-read model, $\sigma_n^2(x,y) = \sigma_{\text{read}}^2 + I_n(x,y)/G$ (camera gain
$G$ in electrons per ADU), then $\sigma_0(x,y)^2 = \sigma_{\text{read}}^2 + \langle
I_n(x,y)\rangle_n/G$, which equals $\sigma_{\text{read}}^2 + a(x,y)/G$ exactly only when
the phase-step distribution's first harmonic vanishes ($R=0$, e.g. evenly-spaced steps).
Setting $\sigma_{\text{read}}\to0$ there as well: with $N_e$ the background
photoelectron count per pixel per frame and $V=b/a$ the fringe visibility,
$\sigma_0=\sqrt{N_e}$ in electron units and Eq. (26) becomes

$$\sigma_\Phi = \frac{1}{V}\sqrt{\frac{2}{N\,N_e}} \tag{27c}$$

the standard shot-noise-limited phase uncertainty — inversely proportional to visibility
and to the square root of total photons collected across the stack. This is a
sanity-check limit, not the default: real cameras' noise is better characterized
empirically (above) than assumed to follow this specific model.

### Stage 2: $\delta_n$ fitted jointly, $g_n$ fixed (`fit_gain=False`)

Stage 1 held $\delta_n$ at its true value when solving Eq. (16). Here the pixel step
instead uses the frame step's own estimate $\hat\delta_n=\delta_n+e_{\delta_n}$, so the
design matrix $A(\delta)$ (Eq. 5) is itself perturbed by $e_{\delta_n}$ before the pixel
step ever sees it.

Differentiating the normal equations $A(\hat\delta)^\top A(\hat\delta)\,X=A(\hat\delta)^\top\mathbf i$
in $e_{\delta_n}$ at the true, noiseless data $\mathbf i=A(\delta)X_0$ — where the true
model's own zero residual, $i_n=a_n^\top X_0$, cancels the leading term — gives

$$\frac{\partial X}{\partial\delta_n} = -A_p^{-1}\,a_n\,\frac{\partial I_n}{\partial\delta_n}, \qquad
\frac{\partial I_n}{\partial\delta_n} = -g_n\,b(x,y)\sin\big(\Phi(x,y)+\delta_n\big) \tag{28}$$

— a wrong $\delta_n$ shifts the pixel-step solution exactly as an extra noise term on
frame $n$'s data would, scaled by the true model's own sensitivity to $\delta_n$. Eq. (17)
therefore carries through with an effective noise
$\varepsilon_n^{\text{eff}}=\varepsilon_n-(\partial I_n/\partial\delta_n)\,e_{\delta_n}$,
and since different frames' $e_{\delta_n}$ come from independent data (below) — dropping
the same-order cross-term between $e_{\delta_n}$ and this pixel's own $\varepsilon_n$,
both drawn from frame $n$'s data —

$$\operatorname{Cov}\big((e_u,e_v)^\top\big) = \sigma^2 S \;+\; \sum_{n=1}^N\Big(\frac{\partial I_n}{\partial\delta_n}\Big)^2\operatorname{Var}(e_{\delta_n})\,k_nk_n^\top, \qquad
k_n=\frac1N\,C^{-1}\big(g_nd_n-R\big) \tag{29}$$

reusing $C$, $R=(R_c,R_s)^\top$, $d_n$ from Eqs. (20a)-(21) — $k_n$ is just $A_p^{-1}$'s
$(u,v)$-rows against frame $n$'s column of $A$, the same closed form as $S$. So

$$\sigma_\Phi^2(x,y) = \underbrace{\frac{\sigma^2}{Nb(x,y)^2}\,w^\top C^{-1}w}_{\text{Stage 1, Eq. (22)}} \;+\; \frac{1}{b(x,y)^2}\sum_{n=1}^N\Big(\frac{\partial I_n}{\partial\delta_n}\Big)^2\operatorname{Var}(e_{\delta_n})\,(w^\top k_n)^2 \tag{30}$$

— Stage 1's result plus a correction that still needs $\operatorname{Var}(e_{\delta_n})$.

**Computing $e_{\delta_n}$.** It comes from the frame step (Eqs. 6, 7), the pixel step's
transpose: a regression over $N_p$ pixel samples instead of $N$ frame samples, with
design $B$ in place of $A$. Isolating the frame step's own error the way Stage 1 isolated
the pixel step's, by holding $(u,v)$ at their true, noise-free values $u=b\cos\Phi$,
$v=-b\sin\Phi$ (Eq. 2), the identical algebra as Eqs. (16)-(17) gives the sandwich

$$\operatorname{Cov}\big((e_{c_n},e_{P_n},e_{Q_n})^\top\big) = A_{ps}^{-1}\big(B^\top D B\big)A_{ps}^{-1}, \qquad A_{ps}=B^\top B, \qquad D=\operatorname{diag}\big(\sigma_0^2(x,y)\big) \tag{31}$$

per frame, independently (each frame's regression in Eq. 6 draws on its own noise only —
the independence Eq. (29) needed), with $\sigma_0^2(x,y)$ the cross-frame quadrature mean
of Eq. (27a). $D\propto I_{N_p}$ — the pixel-domain analogue of Eq. (17a)'s condition — would
need spatially uniform illumination, an idealization rather than the default; only in
that special case does Eq. (31) collapse to

$$\operatorname{Cov}\big((e_{c_n},e_{P_n},e_{Q_n})^\top\big) = \sigma^2\,A_{ps}^{-1} \tag{31a}$$

which is where $S_{ps}:=(A_{ps}^{-1})_{\{P,Q\}}$ is defined — a property of the $(u,v)$
geometry alone, the same block `kappa_ps` already uses elsewhere, no $D$ involved.
Differentiating $\delta_n=\operatorname{atan2}(Q_n,P_n)$ exactly as in Eq. (18), with
$w_n=(-\sin\delta_n,\cos\delta_n)^\top$, the general sandwich (31) gives

$$\sigma_{\delta_n}^2 = \frac{1}{g_n^2}\,w_n^\top\Big[A_{ps}^{-1}\big(B^\top DB\big)A_{ps}^{-1}\Big]_{\{P,Q\}}w_n \tag{32}$$

reducing to $\sigma^2/g_n^2\,w_n^\top S_{ps}w_n$ only in the $D\propto I_{N_p}$ special
case (31a), with $\kappa_{ps}$ playing the same conditioning role for this step that
$\kappa_p$ plays for Eq. (22).

With $g_n$ fixed (not jointly fit), there is no `_whiten_uv` call in this branch — the
$GL(2,\mathbb R)$ gauge freedom of Eq. (13) "only appears once $g_n$ is free" — so
$u=b\cos\Phi$, $v=-b\sin\Phi$ pointwise, exactly, is what $B$ (Eq. 7) is built from.
Applying the well-spread-$\Phi$ approximation (many fringes, no strongly preferred
orientation — the same regime already invoked in "Starting point" and "Why the
background field can be dropped" above) directly to the general sandwich (31), rather
than to the idealized (31a):
$\sum u^2\approx\sum v^2\approx N_p\langle b^2\rangle/2$, $\sum uv\approx0$,
$\sum u\approx\sum v\approx0$, so $A_{ps}\approx\operatorname{diag}(N_p,E,E)$ with
$E=N_p\langle b^2\rangle/2$; and, since $\Phi$ is also decorrelated from local
illumination in this regime, $B^\top DB$'s $\{u,v\}$-block averages to
$\approx\operatorname{diag}\big(N_p\langle\sigma_0^2b^2\rangle/2\big)$. Together these
give an isotropic

$$\sigma_{\delta_n}^2 \;\approx\; \frac{2\sigma_{\text{eff}}^2}{N_p\langle b^2\rangle\,g_n^2}, \qquad
\sigma_{\text{eff}}^2 = \frac{\langle\sigma_0^2\,b^2\rangle}{\langle b^2\rangle} \tag{33}$$

a $b^2$-weighted mean rather than the plain field average $\langle\sigma_0^2\rangle$:
bright, high-contrast pixels dominate the frame-step fit, so their noise level is what
sets $\delta_n$'s uncertainty — $\sigma_{\text{eff}}^2\to\sigma^2$ only in the idealized
uniform-illumination case (31a).

**Final formula.** Substituting Eq. (33) into Eq. (30) gives the total Stage-2 phase
variance:

$$\boxed{\;\sigma_\Phi^2(x,y) = \sigma_\Phi^2\big|_{\text{Stage 1}} \;+\; \frac{2\sigma_{\text{eff}}^2}{N_p\langle b^2\rangle\,b(x,y)^2}\sum_{n=1}^N\frac{\big(\partial I_n/\partial\delta_n\big)^2\,(w^\top k_n)^2}{g_n^2}\;} \tag{34}$$

**Ideal case** — evenly-spaced $\delta_n=2\pi n/N$, unit gain, $N\ge5$: $R=0$ so
$k_n=C^{-1}d_n/N=2d_n/N$ (Eq. 26's $C=\tfrac12I_2$), and
$(\partial I_n/\partial\delta_n)^2(w^\top k_n)^2 = 4b^2\sin^4(\Phi+\delta_n)/N^2$. The sum
collapses exactly, pointwise in $\Phi$ — not merely on average — via
$\sum_n\sin^4(\Phi+\delta_n)=3N/8$ (the second and fourth harmonics both vanish for
$N\ge5$, the same way the second alone vanished in Eq. 26), to a clean multiplier on
Eq. (26):

$$\sigma_\Phi^2\big|_{\text{ideal}} = \sigma_\Phi^2\big|_{\text{Eq. (26)}}\Big(1+\frac{3}{2N_p}\Big) \tag{34a}$$

Not knowing $\delta_n$ exactly costs a $3/(2N_p)$ relative inflation of the per-pixel
phase variance — a couple of parts in a thousand for a megapixel-scale field, negligible
next to Stage 1's own term. Frame-step errors are nonetheless **spatially coherent** —
one number per frame, not one per pixel — so this ratio, not the per-pixel RMS, is the
right comparison: $\delta_n$ is fit far tighter than any single pixel's phase, but that
error does not average away across the field the way per-pixel noise does, and it matters
wherever the reconstruction is later differenced against a reference or another
acquisition with correlated $\delta_n$ error. It also grows back whenever $N_p$ is
effectively smaller than the raw pixel count — a small fringed ROI, or spatially
correlated (not per-pixel-independent) noise such as speckle, both violate Eq. (15)'s
pixel-independence assumption and inflate Eq. (34) accordingly.

### Stage 3: $\delta_n$ and $g_n$ fitted jointly (`fit_gain=True`)

Stages 1 and 2 each held one parameter block fixed at its true value while propagating
noise through the other. With both fitted, AIA's converged solution is a stationary point
of the *joint* nonlinear least-squares problem obtained by combining $\mathcal L$'s two
blocks into one $(3N_p+3N)$-parameter fit (this is what the alternation's own convergence
check, "Convergence" above, is checking a fixed point of). Standard nonlinear
least-squares theory gives the asymptotic covariance of that combined fit as
$\sigma^2(J^\top J)^+$, $J$ the model's Jacobian w.r.t. every parameter — a pseudoinverse
because "Identifiability and gauge" above already exhibits directions (Eq. 13's
$GL(2,\mathbb R)$ reparametrization, Eq. 14a's shift) along which the model $I_n(x,y)$ is
*exactly* unchanged to every order, not just to first order: these are exactly the null
directions of $J$, hence of $J^\top J$. The pseudoinverse itself is always well-defined,
but along those null directions it assigns the minimum-norm covariance consistent with
zero information there, rather than a meaningful physical uncertainty — which is why the
gauge fixes already in the algorithm (Eq. 12's $\overline{c_n}=0$, $\operatorname{median}(g)=1$;
Eq. 14's whitening) are not optional bookkeeping: they are what pins those directions to
one specific, reportable answer instead of an arbitrary point on the null space.

Forming $J^\top J$ explicitly ($(3N_p+3N)\times(3N_p+3N)$, block-sparse in the pixel and
frame indices the way $\mathcal L$'s own separability — "Pixel step"/"Frame step" above —
already exploits) is beyond this section's scope. But Stage 2's technique — the
sensitivity of the pixel-step solution to an error in an assumed frame parameter —
extends directly from $\delta_n$ alone to $(\delta_n,g_n)$ jointly, and gives a concrete
leading-order piece of that joint covariance rather than only an order argument for it.

By the same differentiation as Eq. (28), perturbing the assumed $g_n$ instead of
$\delta_n$:

$$\frac{\partial X}{\partial g_n} = -A_p^{-1}\,a_n\,\frac{\partial I_n}{\partial g_n}, \qquad
\frac{\partial I_n}{\partial g_n} = b(x,y)\cos\big(\Phi(x,y)+\delta_n\big) \tag{35}$$

— the same leverage vector $k_n$ as $\delta_n$'s (both trace back to $-A_p^{-1}a_n$), only
the sensitivity scalar differs. Under the well-spread approximation $S_{ps}\approx
E^{-1}I_2$ (isotropic) already used in Eq. (33), $e_{\delta_n}$ and $e_{g_n}$ are
uncorrelated to leading order: $\delta_n$ is an angle and $g_n$ a radius, so their errors
are orthogonal projections ($w_n\perp\tilde w_n$) of one isotropic $(P_n,Q_n)$-space
error, with no preferred direction to correlate them. The two error sources add rather
than mix:

$$\operatorname{Cov}\big((e_u,e_v)^\top\big) = \sigma^2 S \;+\; \sum_{n=1}^N\Big[\Big(\frac{\partial I_n}{\partial\delta_n}\Big)^2\operatorname{Var}(e_{\delta_n}) + \Big(\frac{\partial I_n}{\partial g_n}\Big)^2\operatorname{Var}(e_{g_n})\Big]k_nk_n^\top \tag{36}$$

with $\operatorname{Var}(e_{g_n})$ the same well-spread reduction as Eq. (33), differing
only by having no $1/g_n^2$: $g_n$ is a radius, not an angle, so its precision doesn't
shrink as $g_n$ grows the way an angle's does at fixed arc-length error —

$$\sigma_{g_n}^2 \;\approx\; \frac{2\sigma_{\text{eff}}^2}{N_p\langle b^2\rangle} \tag{37}$$

Combining Eqs. (30), (33), (36), (37) as in Stage 2's "Final formula":

$$\boxed{\;\sigma_\Phi^2(x,y) = \sigma_\Phi^2\big|_{\text{Stage 1}} \;+\; \frac{2\sigma_{\text{eff}}^2}{N_p\langle b^2\rangle\,b(x,y)^2}\sum_{n=1}^N\left[\frac{\big(\partial I_n/\partial\delta_n\big)^2}{g_n^2} + \big(\partial I_n/\partial g_n\big)^2\right](w^\top k_n)^2\;} \tag{38}$$

**Ideal case**, $N\ge5$: the $\delta_n$ term collapses exactly as in Eq. (34a); the $g_n$
term collapses the same way, via $\sum_n\sin^2(\Phi+\delta_n)\cos^2(\Phi+\delta_n) =
\tfrac14\sum_n\sin^2\big(2(\Phi+\delta_n)\big) = N/8$ (pointwise in $\Phi$, the same
second- and fourth-harmonic cancellation as Eq. 34a). Together:

$$\sigma_\Phi^2\big|_{\text{ideal}} = \sigma_\Phi^2\big|_{\text{Eq. (26)}}\Big(1+\frac{2}{N_p}\Big) \tag{38a}$$

— bigger than Stage 2's $(1+3/(2N_p))$ by the added $g_n$ piece, still negligible next to
Stage 1's own term for any realistic $N_p$.

Eqs. (36)-(38) capture the effect of feeding the frame step's $(\hat\delta_n,\hat g_n)$
into the pixel step's design matrix — the same mechanism as Stage 2, now for both
parameters — but not `_whiten_uv`'s own subsequent transform on $(u,v)$, which this
derivation does not reach. That transform acts on a $(u,v)$ that, under the same
well-spread condition behind Eqs. (33)/(37), already nearly satisfies its own constraint
$\sum u^2\approx\sum v^2$, $\sum uv\approx0$ — so it perturbs an already-near-identity
map, a further, smaller correction on top of Eq. (38a)'s, not a first-order one this
section is missing. This is the same separability "Accuracy diagnostics" already assumes
in treating $\kappa_p$ (Stage 1) and $\kappa_{ps}$ (Stage 2) as two independent
diagnostics rather than requiring the full joint solve.

### Validity of the linearized model

Eqs. (18)/(29) are first-order (delta-method) results: they linearize
$\operatorname{atan2}$ and $\sqrt{\cdot}$ around the noise-free solution, valid while
$b/\sigma\gg1$ so the noisy phasor $(u+e_u,v+e_v)$ stays far from the origin — roughly
$\sigma_\Phi\lesssim0.3$ rad. Below that, the exact noisy-phasor distribution
(Rice-distributed amplitude, wrapped phase) takes over, and Eq. (22) understates the true
tail probability of a $2\pi$ phase slip.

Eq. (15)'s independence assumption is not shot noise's to break — it holds for read, shot
and dark noise alike. What does break it is correlation the noise model excludes by
construction: banded or common-mode readout, crosstalk, and speckle or source-intensity
fluctuation correlated across the field within a frame. `PhaseSolver._normalize`'s
per-frame $\alpha_n$ division is the one genuinely frame-dependent effect in the pipeline:
it rescales frame $n$'s variance by $1/\alpha_n^2$ — diagonal, but not $\propto I_N$, so
Eq. (17a)'s collapse does not apply exactly. Since $\alpha_n$ is estimated, not merely
assumed, this is correctable rather than approximate: $A_p\to A^\top WA$ for a diagonal
per-frame weight $W=\operatorname{diag}(\alpha_n^2)$, everything from Eq. (17) onward
carried through unchanged with $A_p$ replaced throughout.

### Reading `AIAParam.predicted_rms` against this section

Two gaps between $\sigma$/$b$ as this section defines them and what `_aia_diagnostics`
(`phase/methods/aia.py`) actually reports — worth knowing before comparing Eq. (22) or
Eq. (27) to the code's number directly. Neither is fixed here, both are read-only
observations about how to interpret the existing diagnostic:

- **$\sigma$.** The pixel-step residual has $N-3$ degrees of freedom per pixel (three
  parameters fit per pixel), but `_chunked_sigma` divides the summed squared residual by
  $N\cdot N_p$, not $(N-3)\cdot N_p$: it reports a maximum-likelihood, not
  minimum-variance-unbiased, estimate of $\sigma$, biased low by $\sqrt{(N-3)/N}$ — 25%
  at $N=7$, 37% at $N=5$. `predicted_rms` reads optimistic by the same factor.
- **$b$.** `_aia_diagnostics` scales by $\operatorname{median}(b)$, a single number,
  while the true field-averaged $\sigma_\Phi^2$ (Eq. 25a) keeps $\sigma_0^2(x,y)/b(x,y)^2$
  together under one spatial average — a field's low-contrast, low-illumination pixels
  dominate the true RMS far more than dividing two separately aggregated scalars
  suggests, and shot noise widens the gap further by correlating $\sigma_0$ with $b$.

## References

Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of
randomly phase-shifted interferograms," *Optics and Lasers in Engineering*
(2004).

Y. Chen and Q. Kemao, "Advanced iterative algorithm for phase extraction:
performance evaluation and enhancement," *Optics Express* 27(26),
37634-37651 (2019).
