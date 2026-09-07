# Advanced Iterative Algorithm (AIA)

This document derives the Advanced Iterative Algorithm (AIA) for
phase-shifting interferometry (Wang & Han 2004; enhanced per Chen & Kemao,
*Optics Express* 27(26), 37634-37651, 2019), starting from the per-frame
model of `docs/interference_model.md`.

## Starting point

`docs/interference_model.md`'s extraction pipeline gives the simplified
per-frame model used by every extraction method in this package, once the
per-frame normalization factor $\alpha_n$ has been divided out and the
fringe gain $g_n$ separately estimated (that document's Eq. 9):

$$I_n(x, y) = a(x, y) + b(x, y)\,\cos\big(\Phi(x, y) + \delta_n\big), \qquad n = 1 \dots N$$

AIA recovers $a(x,y)$, $b(x,y)$, $\Phi(x,y)$, and the per-frame steps
$\delta_n$ from a stack of $N$ such frames, without requiring $\delta_n$ to
be precisely known in advance. (The code calls this recovered phase `phi`
for brevity; it is $\Phi$ here, matching `interference_model.md` — the
sample-induced phase $\phi$ alone is only obtained afterward, by carrier
removal and reference subtraction.)

### Note on $g_n$

In practice, per-frame fringe gain $g_n$ (present in Eq. (8) of
`interference_model.md`, dropped in Eq. (9) since normalizing it away like
$\alpha_n$ is not generally valid — see that document's "Consequences") is
reintroduced as a known, separately-measured multiplicative factor rather
than assumed equal to 1:

$$I_n(x, y) = a(x, y) + g_n\,b(x, y)\,\cos\big(\Phi(x, y) + \delta_n\big) \tag{1}$$

This does not change the algorithm derived below — $g_n$ only ever enters
as a known coefficient in one linear system (the pixel step, Eq. 5), not as
a new unknown to solve for. See `phase.utils.measure_frame_contrast` for
how $g_n$ is measured in this package.

## Quadrature form

Expanding the cosine in Eq. (1),

$$\cos\big(\Phi + \delta_n\big) = \cos\Phi\cos\delta_n - \sin\Phi\sin\delta_n$$

and defining the quadrature components

$$u(x,y) = b(x,y)\cos\Phi(x,y), \qquad v(x,y) = -b(x,y)\sin\Phi(x,y) \tag{2}$$

Eq. (1) becomes

$$I_n(x, y) = a(x, y) + g_n\big[u(x, y)\cos\delta_n + v(x, y)\sin\delta_n\big] \tag{3}$$

This is linear in $(a, u, v)$ for fixed $\delta_n$ (and known $g_n$), and
linear in $(\cos\delta_n, \sin\delta_n)$ for fixed $(u, v)$ — but not linear
in both sets of unknowns at once, since their product appears. That
bilinearity rules out closed-form recovery when $\delta_n$ is unknown, and
motivates solving by alternating least squares.

## Objective function

`docs/interference_model.md`'s Eq. (10) poses extraction as minimizing the
sum of squared residuals against that document's Eq. (9). Substituting the
$g_n$-extended, quadrature form derived here (Eq. 3) gives the specific loss
AIA minimizes, in terms of $(a, u, v)$ rather than $(a, b, \Phi)$ — since
that is the parameterization each linear sub-problem below actually solves
for (recovering $(b, \Phi)$ from $(u, v)$ is Eq. 9 below):

$$\mathcal{L}(a, u, v, \{\delta_n\}) = \sum_{n=1}^N \sum_{x,y} \Big[I_n^{\text{meas}}(x, y) - a(x, y) - g_n\big(u(x, y)\cos\delta_n + v(x, y)\sin\delta_n\big)\Big]^2 \tag{4}$$

## Alternating least squares

AIA alternates two linear solves until $\delta_n$ stops changing (or an
iteration limit is reached):

### 1. Pixel step

Because $\mathcal{L}$ (Eq. 4) sums over pixels with no cross terms between
them, minimizing it over the fields $(a, u, v)$ decomposes into an
independent minimization at each pixel, each a sum over frames only:

$$\mathcal{L}_{x,y}(a,u,v) = \sum_{n=1}^N \Big[I_n^{\text{meas}}(x,y) - a - g_n\big(u\cos\delta_n + v\sin\delta_n\big)\Big]^2$$

With $\delta_n$ (and $g_n$) fixed, minimizing $\mathcal{L}_{x,y}$ over
$(a, u, v)$ at a given pixel is a linear regression of that pixel's $N$
frame values onto the design matrix

$$A = \begin{bmatrix} 1 & g_1\cos\delta_1 & g_1\sin\delta_1 \\ \vdots & \vdots & \vdots \\ 1 & g_N\cos\delta_N & g_N\sin\delta_N \end{bmatrix} \tag{5}$$

solved (by pseudoinverse, across every pixel at once) for $(a, u, v)$ —
exactly $\arg\min_{a,u,v}\mathcal{L}$ with $\{\delta_n\}$ held fixed, by the
separability above. $A^\top A$ is the pixel-step normal matrix, $A_p$ below.

### 2. Frame step

With $(a, u, v)$ now fixed from step 1, the textbook next step would
minimize $\mathcal{L}$ (Eq. 4) over $\{\delta_n\}$ with $a$ pinned at its
step-1 value. AIA instead minimizes a **modified** per-frame objective that
replaces the fixed $a(x,y)$ with a free per-frame offset $\alpha_n$:

$$\mathcal{L}'_n(\alpha_n, P_n, Q_n) = \sum_{x,y} \Big[I_n^{\text{meas}}(x, y) - \alpha_n - P_n\,u(x, y) - Q_n\,v(x, y)\Big]^2 \tag{6}$$

— a linear regression, at each frame independently, of the $P = H \times W$
pixel values onto the design matrix

$$B = \begin{bmatrix} 1 & u(x_1,y_1) & v(x_1,y_1) \\ \vdots & \vdots & \vdots \\ 1 & u(x_P,y_P) & v(x_P,y_P) \end{bmatrix} \tag{7}$$

— note $B$ does **not** carry $g_n$: it is built directly from $u, v$, the
same for every frame. This is a deliberate departure from strict
block-coordinate descent on $\mathcal{L}$: subtracting the fixed (and,
mid-iteration, still imperfect) $a$ explicitly was tried and found
empirically less robust, since $a$'s own error then feeds back into the fit
rather than being re-derived from the raw data every iteration. AIA's two
steps therefore alternately minimize two different, closely related
objectives ($\mathcal{L}$ and $\mathcal{L}'_n$), not one shared loss in the
strict sense.

Solving frame $n$'s regression gives coefficients $(\alpha_n, P_n, Q_n)$
with $P_n \approx g_n\cos\delta_n$, $Q_n \approx g_n\sin\delta_n$, so that

$$\delta_n = \operatorname{atan2}(Q_n, P_n) \tag{8}$$

recovers $\delta_n$ exactly regardless of $g_n$ (for $g_n > 0$): a common
positive scale factor on both $P_n$ and $Q_n$ cancels in $\operatorname{atan2}$.
This is the precise sense in which the algorithm itself does not change when
gain varies frame to frame — $g_n$ only ever appears in the pixel-step
design matrix $A$ (Eq. 5); the frame step (Eq. 6–8) has exactly the same
form whether or not it does. $B^\top B$ is the frame-step normal matrix,
$A_{ps}$ below.

The free offset $\alpha_n$ is discarded once $\delta_n$ is extracted —
$a$ itself comes only from step 1.

### Phase-origin convention

Eq. (3) — like Eq. (8) of `interference_model.md` — has a global offset
ambiguity: adding a constant to every $\delta_n$ while subtracting it from
$\Phi$ reproduces the same data. Each iteration re-references
$\delta_n \to \delta_n - \delta_1$ so $\delta_1 = 0$, fixing the split (the
same convention `interference_model.md` notes under "Global offset
ambiguity").

### Convergence

Iterate steps 1–2 until the largest per-frame change in $\delta_n$ between
iterations falls below a tolerance, or an iteration limit is reached. The
final phase map and fringe amplitude follow directly from $u, v$:

$$\Phi(x,y) = \operatorname{atan2}\big(-v(x,y),\, u(x,y)\big), \qquad b(x,y) = \sqrt{u(x,y)^2 + v(x,y)^2} \tag{9}$$

## Accuracy diagnostics

Chen & Kemao (2019) show AIA's accuracy is governed by how well-conditioned
the two normal matrices $A_p = A^\top A$ and $A_{ps} = B^\top B$ are:

- $\kappa_p = \operatorname{cond}(A_p)$ — how well the phase-shift
  distribution $\{\delta_n\}$ conditions the pixel-step solve. Enters the
  accuracy prediction (Eq. 10) directly.
- $\kappa_{ps} = \operatorname{cond}(A_{ps})$, evaluated on the *normalized*
  unit-circle design (columns $\cos\Phi, \sin\Phi$, amplitude divided out)
  — how well the recovered phase pattern covers the unit circle. Bounded
  below by 2, achieved when $\Phi$ is evenly distributed over $2\pi$. Large
  values mean the field spans too little phase (less than roughly one
  fringe) for the frame step to reliably separate $\delta_n$ from noise.

and predict the RMS phase error as

$$\sigma_\Phi \approx 0.42\,\big(\sqrt{\kappa_p} + 2\big)\,\frac{\sigma}{b}\,\frac{1}{\sqrt{N}} \tag{10}$$

where $\sigma$ is the RMS residual of the final pixel-step fit and $b$ is
the median fringe amplitude. A poorly conditioned acquisition ($\kappa_p$ or
$\kappa_{ps}$ large) should not be trusted even if the iteration reports
convergence.

## Jointly estimating gain (`fit_gain`)

The "Note on $g_n$" above assumes $g_n$ is known in advance, e.g. from
`phase.utils.measure_frame_contrast`, which locates a linear spatial
carrier in each frame's 2-D FFT. That assumption fails outright on
circular (or otherwise carrier-free) fringes — there is no carrier peak to
locate — and more generally couples gain accuracy to a spatial-frequency
assumption that has nothing to do with the phase-shifting model itself.
`aia(..., fit_gain=True)` instead recovers $g_n$ from the same alternating
solve, at no extra asymptotic cost.

The frame step (Eq. 6–8) already computes $(P_n, Q_n)$ with
$P_n \approx g_n\cos\delta_n$, $Q_n \approx g_n\sin\delta_n$, and discards
their magnitude after extracting $\delta_n$ (Eq. 8). `fit_gain` keeps it:

$$g_n = \sqrt{P_n^2 + Q_n^2}, \qquad g_n \leftarrow g_n / \operatorname{median}(g_n) \tag{11}$$

and feeds it back into the next pixel step's design matrix $A$ (Eq. 5).
It also keeps the frame step's per-frame offset $\alpha_n$ (Eq. 6, renamed
$c_n$ in code to avoid clashing with `interference_model.md`'s $\alpha_n$),
gauge-fixed by $c_n \leftarrow c_n - \overline{c_n}$, and subtracts it from
the stack *before* the next pixel step:

$$A\,[a\;\, u\;\, v]^\top = I - c \tag{12}$$

Both this and Eq. (11) are needed together: without Eq. (12), the pixel
and frame steps are minimizing different objectives ($\mathcal{L}$ with a
fixed per-pixel $a$, vs. $\mathcal{L}'_n$ with a free per-frame $c_n$ that
$\mathcal{L}$ does not have), so there is no guarantee the joint residual
decreases every iteration. With it, both steps are exact least-squares
minimizers of the one joint objective
$\sum_{n,x,y}\big[I_n - a - c_n - g_n(u\cos\delta_n+v\sin\delta_n)\big]^2$,
so that residual decreases monotonically every iteration (a stall is then
a genuine local optimum, not the two steps chasing different targets) —
this is unrelated to, and does not fix, the identifiability issue below.

### A gauge freedom that only appears once $g_n$ is free

With $g_n \equiv 1$, $(P_n, Q_n) = (\cos\delta_n, \sin\delta_n)$ is
constrained to the unit circle — a single degree of freedom per frame.
Once $g_n$ is free, $(P_n, Q_n)$ is an *unconstrained* point in the plane,
and Eq. (3)/(12)'s model is invariant under **any** invertible linear
reparametrization

$$(u, v) \to (u, v)\,M, \qquad (P, Q) \to (P, Q)\,M^{-\top} \tag{13}$$

for a $2\times 2$ matrix $M$, since $\{1, u, v\}$ and $\{1, u M, v M\}$
span the same subspace for any invertible $M$, not just a rotation. Left
alone, the alternating solve can converge to *any* basis of that
subspace — fitting $I$ exactly as well (often better, since a generic
basis has more freedom to explain noise) — without $(P_n, Q_n)$ tracing
$(g_n\cos\delta_n, g_n\sin\delta_n)$ for any physically meaningful
$\delta_n$. Verified on synthetic data: without correcting this, a fit
that *halves* the residual relative to the correctly-identified
(fixed-$g$) solution can still land tens of degrees off in recovered
phase, despite the iteration reporting `converged`.

The fix (`phase.methods.aia._whiten_uv`, run on $(u, v)$ after every
pixel step, only when `fit_gain` is True) rescales/shears $(u, v)$ so that

$$\sum_{x,y} u^2 = \sum_{x,y} v^2, \qquad \sum_{x,y} uv = 0 \tag{14}$$

which collapses the residual gauge from all of $GL(2,\mathbb{R})$ down to
just rotations and reflections, $O(2)$ — exactly the ambiguity plain
($g \equiv 1$) AIA already has and already resolves (phase origin via
$\delta_1 = 0$; sign via the documented
$(\Phi, \delta) \to (-\Phi, -\delta)$ branch), rather than that plus a
two-parameter shear/scale family on top. Total pixel-sum energy (the
trace of $(u,v)$'s $2\times 2$ Gram matrix) is preserved, so only Eq.
(11)'s own normalization changes $g$'s overall scale. This whitening step
does not change the achievable value of the joint objective either: it
reparametrizes the same 3-D column space $\{1, u, v\}$, and the very next
frame step is an unconstrained regression against whichever basis it is
handed, so it always reaches a cost at least as low as before whitening —
the monotone-descent property above holds with this step included.

## References

Z. Wang and B. Han, "Advanced iterative algorithm for phase extraction of
randomly phase-shifted interferograms," *Optics and Lasers in Engineering*
(2004).

Y. Chen and Q. Kemao, "Advanced iterative algorithm for phase extraction:
performance evaluation and enhancement," *Optics Express* 27(26),
37634-37651 (2019).
