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

With $(a, u, v)$ now fixed and whitened, the textbook next step would
minimize $\mathcal{L}$ (Eq. 4) over $\{\delta_n, g_n\}$ with $a$ pinned at its
step-1 value. AIA instead minimizes a per-frame objective that replaces the
fixed $a(x,y)$ with the free per-frame offset $c_n$ already carried by
$\mathcal{L}$:

$$\mathcal{L}'_n(c_n, P_n, Q_n) = \sum_{x,y} \Big[I_n^{\text{meas}}(x, y) - c_n - P_n\,u(x, y) - Q_n\,v(x, y)\Big]^2 \tag{6}$$

— a linear regression, at each frame independently, of the $P = H \times W$
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
below.

Steps 1 and 2 are exact least-squares minimizers of the one joint objective
$\mathcal{L}$ (Eq. 4): the pixel step minimizes it over $(a, u, v)$ with
$(c_n, \delta_n, g_n)$ fixed, and the frame step minimizes it over
$(c_n, \delta_n, g_n)$ with $(a, u, v)$ fixed. Whitening between them changes
neither step's objective — it reparametrizes the same 3-D column space
$\{1, u, v\}$, and the frame step is an unconstrained regression against
whichever basis of that space it is handed — so the joint residual decreases
monotonically every iteration, and a stall (the iteration limit reached
without meeting the tolerance) is a genuine local optimum, not the two steps
chasing different targets.

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
convergence.

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
in the algorithm resolves it explicitly.

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

## References

Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of
randomly phase-shifted interferograms," *Optics and Lasers in Engineering*
(2004).

Y. Chen and Q. Kemao, "Advanced iterative algorithm for phase extraction:
performance evaluation and enhancement," *Optics Express* 27(26),
37634-37651 (2019).
