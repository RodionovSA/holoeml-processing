# Frame moments and phase-step leakage

This document derives the statistical moments — mean and variance — of the interferogram
stack $I_n(x,y)$ across frames $n = 1 \dots N$, starting from the per-frame model established in
[`interference_model.md`](interference_model.md). The purpose is to make explicit how a
non-uniform set of phase steps $\{\delta_n\}$, together with a frame-dependent contrast factor
$g_n$, "leaks" residual fringe structure into these moments, since the mean and variance maps
of a frame stack are routinely used as diagnostics and as inputs to phase-extraction algorithms.

Throughout, $(x, y)$ dependence of $a$, $b$, $\Phi$ is suppressed for readability: every
equation below holds pixel-by-pixel.

## 1. Setup

### 1.1 The model

[`interference_model.md`](interference_model.md) derives the per-frame intensity (its Eq. 17,
with the total static phase $\Phi = \phi + \phi_{\text{inst}} + \phi_{\text{carrier}}$ defined in
its Eq. 15) as

$$I_n = \alpha_n\Big[a + g_n\,b\cos(\Phi + \delta_n)\Big], \qquad n = 1 \dots N \tag{F0}$$

where $\alpha_n$ is the per-frame source-power factor, $g_n \ge 0$ is the per-frame contrast
factor, and $\delta_n$ is the commanded (but generally imperfect) phase step of frame $n$. This
document takes Eq. (F0) as given and does not re-derive it; all symbols $a$, $b$, $\alpha_n$,
$g_n$, $\delta_n$, $\Phi$ carry exactly the meaning assigned there.

### 1.2 The frame-averaging operator

Define the average of a per-frame quantity $X_n$ over the realized stack of $N$ frames as

$$\langle X\rangle_n \;=\; \frac{1}{N}\sum_{n=1}^{N} X_n \tag{F1}$$

This is a **finite-sample average over the frames actually acquired**, not an ensemble
expectation. It is what a mean or variance image, computed pixel-wise across a real frame
stack, actually evaluates. Consequently the residual quantities defined below are numbers one
can compute directly from the realized sequences $\{\delta_n\}$ and $\{g_n\}$, not statistical
abstractions.

## 2. $g$-weighted phase-step residuals

$g_n$ multiplies exactly the same $\cos(\Phi+\delta_n)$ term that $\delta_n$ sits inside — it is
a **per-frame weight on the modulation**, not a separate additive effect. It therefore cannot be
pulled out of a frame average of that term in general; it must be carried *inside* the circular
moments alongside $\delta_n$. (Section 3 explains why this is different from how $\alpha_n$ is
treated.)

### 2.1 First harmonic

$$R_c = \big\langle g_n\cos\delta_n\big\rangle_n, \qquad R_s = \big\langle g_n\sin\delta_n\big\rangle_n \tag{F2}$$

$$R = \big\langle g_n\,e^{i\delta_n}\big\rangle_n = R_c + iR_s, \qquad
|R| = \sqrt{R_c^2+R_s^2}, \qquad \psi = \operatorname{atan2}(R_s,R_c) \tag{F3}$$

### 2.2 Second harmonic

The second moment of $I_n$ involves $g_n^2$, not $g_n$ (Section 5 shows this explicitly), so the
second-harmonic residual is weighted by $g_n^2$:

$$R_c^{(2)} = \big\langle g_n^2\cos2\delta_n\big\rangle_n, \qquad R_s^{(2)} = \big\langle g_n^2\sin2\delta_n\big\rangle_n \tag{F4}$$

$$R^{(2)} = \big\langle g_n^2\,e^{i2\delta_n}\big\rangle_n = R_c^{(2)}+iR_s^{(2)}, \qquad
\big|R^{(2)}\big| = \sqrt{\big(R_c^{(2)}\big)^2+\big(R_s^{(2)}\big)^2}, \qquad
\psi_2 = \operatorname{atan2}\big(R_s^{(2)},R_c^{(2)}\big) \tag{F5}$$

### 2.3 Bounds and normalized coverage quality

Since $g_n \ge 0$ and $|e^{ik\delta_n}| = 1$, the triangle inequality applied to Eqs. (F3) and
(F5) gives the exact bounds

$$|R| \le \langle g\rangle, \qquad \big|R^{(2)}\big| \le \big\langle g^2\big\rangle \tag{F6}$$

so the natural, dimensionless measures of step-coverage quality are the normalized ratios

$$r_1 = \frac{|R|}{\langle g\rangle} \in [0,1], \qquad r_2 = \frac{\big|R^{(2)}\big|}{\langle g^2\rangle} \in [0,1] \tag{F7}$$

The unnormalized $R$, $R^{(2)}$ are used directly in the formulas below because they keep the
final expressions compact; $r_1, r_2$ are the right quantities to *report* as a coverage-quality
diagnostic, since they isolate the step-timing contribution from the contrast level.

### 2.4 Master identities

For any fixed (non-random) phase $\Phi$, exactly as for the unweighted case,

$$\big\langle g_n\cos(\Phi+\delta_n)\big\rangle_n = \cos\Phi\langle g_n\cos\delta_n\rangle_n - \sin\Phi\langle g_n\sin\delta_n\rangle_n
= R_c\cos\Phi - R_s\sin\Phi = |R|\cos(\Phi+\psi) \tag{F8}$$

$$\big\langle g_n^2\cos(2\Phi+2\delta_n)\big\rangle_n = R_c^{(2)}\cos2\Phi - R_s^{(2)}\sin2\Phi = \big|R^{(2)}\big|\cos(2\Phi+\psi_2) \tag{F9}$$

using the same angle-addition step as before ($R_c=|R|\cos\psi$, $R_s=|R|\sin\psi$, and
likewise at the second harmonic). These are used repeatedly below.

**Assumption (good $\delta$ coverage) — constant-$g$ baseline.** If $g_n \equiv g_0$ is the same
for every frame, it factors out of Eqs. (F3), (F5) entirely: $R = g_0\langle e^{i\delta_n}\rangle_n$,
$R^{(2)} = g_0^2\langle e^{i2\delta_n}\rangle_n$. For a uniform step set $\delta_n = 2\pi n/N$
(plus an arbitrary offset), $\sum_{n=1}^N e^{ik\delta_n} = 0$ for every integer $k$ not a
multiple of $N$, so $R = 0$ for $N\ge2$ and $R^{(2)}=0$ for $N\ge3$ — recovering the classical
"good coverage kills the residuals" statement. **This requires $g$ to be constant.** When $g_n$
varies from frame to frame, uniform $\delta_n$ alone does not make $R$ or $R^{(2)}$ vanish in
general — the weighted sum $\sum_n g_n e^{ik\delta_n}$ need not cancel just because
$\sum_n e^{ik\delta_n}$ does. Section 7 works out what happens instead.

## 3. Why $g$ stays inside and $\alpha$ stays outside

$g_n$ and $\alpha_n$ enter Eq. (F0) differently, and that difference is exactly what
determines how each is handled:

- $g_n$ multiplies **only** the modulation term, $\cos(\Phi+\delta_n)$. It is a per-frame
  reweighting of the same oscillatory quantity that $\delta_n$ perturbs, so it belongs inside
  the frame average alongside $\delta_n$ — as done in Section 2. It cannot be factored out as a
  constant unless it *is* constant (Section 2.4); in general it changes the size and even the
  phase of the leaked residual, and a purely random or drifting $g_n$ is an independent source
  of leakage in its own right (Section 7).
- $\alpha_n$ multiplies the **whole bracket** — background and modulation alike. It cannot be
  absorbed into $b$ (it does not sit next to $\cos(\Phi+\delta_n)$ alone), so it is kept as an
  explicit multiplier outside the bracket throughout, exactly as in Eq. (F0). This is what
  lets $\langle\alpha\rangle$ and $\langle\alpha^2\rangle$ appear visibly in the formulas below.

**Assumption ($\alpha \perp (g,\delta)$).** The source-power fluctuation $\alpha_n$ (a global
light-source effect) is driven by a mechanism unrelated to both the contrast jitter $g_n$ and
the commanded step $\delta_n$, so the realized sequence $\{\alpha_n\}$ is statistically
independent of the pair $\{(g_n,\delta_n)\}$ across the frame stack. This is what allows a frame
average of a product like $\alpha_n\,h(g_n,\delta_n)$ to factor:

$$\big\langle\alpha_n\,h(g_n,\delta_n)\big\rangle_n \;\approx\; \langle\alpha\rangle_n\,\langle h(g,\delta)\rangle_n \tag{F10}$$

for any function $h$ appearing below ($h = a$, $h=g\cos(\Phi+\cdot)$, or $h=g^2\cos(2\Phi+2\cdot)$).
This factorization is exact in expectation over independent realizations of $\alpha$ and
$(g,\delta)$; for a single realized finite stack it holds to the same good approximation as any
other finite-sample average used in this document, improving as $N$ grows. Note that **no
independence between $g$ and $\delta$ is assumed anywhere** — any correlation between them is
exactly what $R$ and $R^{(2)}$ measure (Section 7).

## 4. Mean

Averaging Eq. (F0) over frames and using $\langle\alpha_n a\rangle_n = a\langle\alpha\rangle_n$
(since $a$ is a static constant),

$$\langle I\rangle_n = a\langle\alpha\rangle_n + b\big\langle\alpha_n\,g_n\cos(\Phi+\delta_n)\big\rangle_n \tag{F11}$$

Applying the factorization Eq. (F10) to the second term (with $h = g\cos(\Phi+\cdot)$) and then
the master identity Eq. (F8):

$$\langle I\rangle_n = \langle\alpha\rangle_n\Big[a + b\big\langle g_n\cos(\Phi+\delta_n)\big\rangle_n\Big]
= \langle\alpha\rangle_n\big[a + b(R_c\cos\Phi - R_s\sin\Phi)\big] \tag{F12}$$

which is the **component form**. Collapsing the bracket with Eq. (F8) gives the equivalent
amplitude–phase form (writing $\langle\alpha\rangle \equiv \langle\alpha\rangle_n$ from here on):

$$\boxed{\;\langle I\rangle = \langle\alpha\rangle\big[a + b\,|R|\cos(\Phi + \psi)\big]\;} \tag{F13}$$

On top of the ideal, phase-independent background $\langle\alpha\rangle a$, a non-uniform step
set — now understood to include a non-constant contrast factor $g_n$ — leaks a
**first-harmonic** residual fringe, oscillating once per cycle of $\Phi$, with amplitude
$\langle\alpha\rangle b\,|R|$ and phase offset $\psi$. It vanishes identically as $|R|\to0$,
leaving $\langle I\rangle \to \langle\alpha\rangle a$.

## 5. Second moment

Square Eq. (F0):

$$I_n^2 = \alpha_n^2\Big[a^2 + 2ab\,g_n\cos(\Phi+\delta_n) + b^2 g_n^2\cos^2(\Phi+\delta_n)\Big] \tag{F14}$$

This is exactly where the $g_n^2$ weighting of the second harmonic (Section 2.2) comes from.
Using $\cos^2(\cdot) = \tfrac12 + \tfrac12\cos2(\cdot)$ on the last term,

$$I_n^2 = \alpha_n^2\Big[a^2 + \tfrac12 b^2 g_n^2 + 2ab\,g_n\cos(\Phi+\delta_n) + \tfrac12 b^2 g_n^2\cos(2\Phi+2\delta_n)\Big] \tag{F15}$$

Averaging over frames and factoring each $\alpha_n^2$-weighted term via Eq. (F10) (valid for
$\alpha^2$ under the same $\alpha\perp(g,\delta)$ assumption), then applying the master
identities Eqs. (F8)–(F9):

$$\boxed{\;\langle I^2\rangle = \langle\alpha^2\rangle\Big[\underbrace{a^2+\tfrac12 b^2\langle g^2\rangle}_{\text{ideal}}
\;+\; \underbrace{2ab\,|R|\cos(\Phi+\psi)}_{\text{1st-harmonic leakage}}
\;+\; \underbrace{\tfrac12 b^2\,\big|R^{(2)}\big|\cos(2\Phi+\psi_2)}_{\text{2nd-harmonic leakage}}\Big]\;} \tag{F16}$$

The three pieces are:
- **Ideal part**, $a^2 + \tfrac12 b^2\langle g^2\rangle$: the phase-independent second moment.
  Note it already carries $\langle g^2\rangle$, not just $\langle g\rangle^2$ — contrast jitter
  raises this floor even before any leakage is considered, since $\langle g^2\rangle =
  \langle g\rangle^2 + \mathrm{Var}(g)$.
- **First-harmonic leakage**, $2ab\,|R|\cos(\Phi+\psi)$: the cross term between background and
  fringe, controlled by the same $|R|$ that leaked into the mean.
- **Second-harmonic leakage**, $\tfrac12 b^2 |R^{(2)}|\cos(2\Phi+\psi_2)$: from the
  self-interference of the fringe term, at twice the frequency of $\Phi$, controlled by the
  second circular moment $|R^{(2)}|$.

## 6. Variance

By definition, $\mathrm{Var}(I) = \langle I^2\rangle - \langle I\rangle^2$. Square Eq. (F13),
expanding $\cos^2(\Phi+\psi)$ with the same half-angle identity used in Eq. (F15), for
consistency:

$$\langle I\rangle^2 = \langle\alpha\rangle^2\Big[a^2 + \tfrac12 b^2|R|^2 + 2ab\,|R|\cos(\Phi+\psi) + \tfrac12 b^2|R|^2\cos(2\Phi+2\psi)\Big] \tag{F17}$$

using $\cos^2(\Phi+\psi) = \tfrac12+\tfrac12\cos(2\Phi+2\psi)$. Split $\langle\alpha^2\rangle$ in
Eq. (F16) using the exact, assumption-free identity $\mathrm{Var}(\alpha) = \langle\alpha^2\rangle
- \langle\alpha\rangle^2$ (the definition of the sample variance of $\alpha_n$ over the stack —
no independence assumption is needed for this step):

$$\langle I^2\rangle = \langle\alpha\rangle^2\Big[a^2+\tfrac12 b^2\langle g^2\rangle+2ab|R|\cos(\Phi+\psi)+\tfrac12 b^2|R^{(2)}|\cos(2\Phi+\psi_2)\Big]
+ \mathrm{Var}(\alpha)\Big[a^2+\tfrac12 b^2\langle g^2\rangle+2ab|R|\cos(\Phi+\psi)+\tfrac12 b^2|R^{(2)}|\cos(2\Phi+\psi_2)\Big] \tag{F18}$$

Subtracting Eq. (F17), the $\langle\alpha\rangle^2$ bracket in Eq. (F18) shares the term
$2ab|R|\cos(\Phi+\psi)$ with $\langle I\rangle^2$ — and it **cancels exactly**:

$$\langle\alpha\rangle^2\Big[\big(a^2+\tfrac12 b^2\langle g^2\rangle+2ab|R|\cos(\Phi+\psi)+\tfrac12 b^2|R^{(2)}|\cos(2\Phi+\psi_2)\big)
- \big(a^2+\tfrac12 b^2|R|^2+2ab|R|\cos(\Phi+\psi)+\tfrac12 b^2|R|^2\cos(2\Phi+2\psi)\big)\Big]$$
$$= \langle\alpha\rangle^2\Big[\tfrac12 b^2\big(\langle g^2\rangle-|R|^2\big) + \tfrac12 b^2\Big(|R^{(2)}|\cos(2\Phi+\psi_2) - |R|^2\cos(2\Phi+2\psi)\Big)\Big] \tag{F19}$$

The two leftover terms in Eq. (F19) are *both* second harmonics in $\Phi$ — $|R^{(2)}|\cos(2\Phi+\psi_2)
= \mathrm{Re}\big[R^{(2)}e^{i2\Phi}\big]$ and $|R|^2\cos(2\Phi+2\psi) = \mathrm{Re}\big[R^2e^{i2\Phi}\big]$
(since $R^2 = |R|^2e^{i2\psi}$) — so they combine into a single fringe governed by one complex
residual,

$$D \;\equiv\; R^{(2)} - R^2 \;=\; \big\langle g_n^2 e^{i2\delta_n}\big\rangle_n - \big\langle g_n e^{i\delta_n}\big\rangle_n^{\,2},
\qquad \psi_D = \arg D, \qquad
|R^{(2)}|\cos(2\Phi+\psi_2) - |R|^2\cos(2\Phi+2\psi) = |D|\cos(2\Phi+\psi_D) \tag{F20}$$

so the exact variance is

$$\boxed{\;\mathrm{Var}(I) = \langle\alpha\rangle^2\Big[\underbrace{\tfrac12 b^2\big(\langle g^2\rangle-|R|^2\big)}_{\text{DC}} + \underbrace{\tfrac12 b^2\,|D|\cos(2\Phi+\psi_D)}_{\text{2nd-harmonic fringe}}\Big]
+ \mathrm{Var}(\alpha)\Big[a^2+\tfrac12 b^2\langle g^2\rangle+2ab|R|\cos(\Phi+\psi)+\tfrac12 b^2\big|R^{(2)}\big|\cos(2\Phi+\psi_2)\Big]\;} \tag{F21}$$

This is the key structural statement of this document: **the mean leaks through the first
harmonic ($|R|$), the variance leaks through the second harmonic ($D = R^{(2)}-R^2$)** — the
first-harmonic residual that dominated $\langle I\rangle$ cancels out of $\mathrm{Var}(I)$
completely, and the only oscillatory term left is the single second-harmonic fringe carried by
$D$. Writing it as $D = R^{(2)}-R^2$ also shows why the earlier, unexpanded form obscured the
structure: the $-b^2|R|^2\cos^2(\Phi+\psi)$ term was doing two jobs at once — lowering the DC
floor from $\langle g^2\rangle$ to $\langle g^2\rangle - |R|^2$, *and* partially cancelling the
$|R^{(2)}|$ fringe — and Eq. (F21) separates them cleanly. The second bracket in Eq. (F21),
multiplying $\mathrm{Var}(\alpha)$, is exactly the $\langle I^2\rangle$ bracket from Eq. (F16);
it makes visible a flicker (source-power fluctuation) contribution to the background variance,
$a^2\,\mathrm{Var}(\alpha)$, that persists even where $|R|=|R^{(2)}|=0$.

**Positivity.** Two consistency checks on Eq. (F21) worth stating explicitly:
- The DC term is non-negative: $|R|^2 \le \langle g\rangle^2 \le \langle g^2\rangle$, the first
  inequality from Eq. (F6) and the second from Cauchy–Schwarz (or directly, $\mathrm{Var}(g)\ge0$).
- Since $\mathrm{Var}(I)\ge0$ must hold for *every* $\Phi$, Eq. (F21) forces
  $|D| \le \langle g^2\rangle - |R|^2$ — the second-harmonic fringe can never exceed the DC floor
  that carries it. The bound is tight exactly when $g_n$ is constant and $\delta_n$ is a single
  repeated value (Section 7.1's constant-step check below), where the variance touches zero at
  the phase where $\cos(\Phi+\delta)=0$ for every frame.

**Small-residual first-order approximation.** Writing $|R| = r_1\langle g\rangle$ with $r_1\ll1$
(Eq. F7, good step/contrast coverage), $|R|^2$ is $O(r_1^2)$ and $D = R^{(2)} - R^2 \to R^{(2)}$
to leading order, recovering the simpler, single-residual form

$$\mathrm{Var}(I) \;\approx\; \langle\alpha\rangle^2\Big[\tfrac12 b^2\langle g^2\rangle + \tfrac12 b^2\big|R^{(2)}\big|\cos(2\Phi+\psi_2)\Big]
+ \mathrm{Var}(\alpha)\Big[a^2+\tfrac12 b^2\langle g^2\rangle+2ab|R|\cos(\Phi+\psi)+\tfrac12 b^2\big|R^{(2)}\big|\cos(2\Phi+\psi_2)\Big] \tag{F22}$$

i.e. $|D|$ and $|R^{(2)}|$ agree to first order in the residuals; Eq. (F21) is needed only when
$|R|$ is not small enough to neglect $R^2$ next to $R^{(2)}$.

**Practical remark.** Since $\Phi$ carries the deliberately-set carrier $\phi_{\text{carrier}}$
(Eq. 15 of `interference_model.md`), Eq. (F13) predicts that a *mean* image computed across a
non-ideally-stepped stack shows residual fringes at the carrier frequency, while Eq. (F21)
predicts that the corresponding *variance* image shows residual fringes at **twice** the
carrier frequency — a direct, checkable diagnostic signature, whether the non-ideality comes
from the steps $\delta_n$, the contrast $g_n$, or their interplay (Section 7).

## 7. Special cases of $g$

The general residuals $R$, $R^{(2)}$ (Eqs. F3, F5) mix the behavior of the steps and of the
contrast factor together. It is instructive to separate out what each contributes.

### 7.1 $g_n \equiv 1$ — pure step leakage

If contrast is perfect and constant, $R \to \langle e^{i\delta_n}\rangle_n \equiv R_\delta$ and
$R^{(2)} \to \langle e^{i2\delta_n}\rangle_n \equiv R_\delta^{(2)}$: this is the classical
phase-stepping leakage problem, with $\langle g^2\rangle=1$ and $D = R_\delta^{(2)}-R_\delta^2$
in Eqs. (F16), (F21). For the further special case of a single repeated step,
$\delta_n\equiv\delta$ for every frame: $R_\delta=\langle e^{i\delta_n}\rangle_n=e^{i\delta}$ and
$R_\delta^{(2)}=\langle e^{i2\delta_n}\rangle_n=e^{i2\delta}$, so $D = e^{i2\delta}-\big(e^{i\delta}\big)^2=0$
and the DC floor $\tfrac12b^2(1-1)=0$ — Eq. (F21) correctly gives $\mathrm{Var}(I)=0$, the tight
case of the positivity bound noted in Section 6.

### 7.2 $g_n \equiv g_0$ constant — leakage rescaled, not reshaped

If contrast is reduced by the same factor every frame, $g_0$ factors out of Eqs. (F3), (F5)
exactly:

$$R = g_0\,R_\delta, \qquad R^{(2)} = g_0^2\,R_\delta^{(2)}, \qquad
D = R^{(2)}-R^2 = g_0^2\big(R_\delta^{(2)} - R_\delta^2\big) \tag{F23}$$

so $|R|=g_0|R_\delta|$, $|R^{(2)}|=g_0^2|R_\delta^{(2)}|$, while the phases $\psi=\psi_\delta$,
$\psi_2=\psi_{2\delta}$ and the normalized coverage ratios $r_1=|R_\delta|$, $r_2=|R^{(2)}_\delta|$
are **unchanged** — a constant contrast loss rescales the size of the leaked fringe (through
$b$'s effective weight $g_0$ or $g_0^2$) but does not change how well the steps cover the circle
or where the leakage phase sits. **This is the special case in which folding $g$ into a single
static amplitude $b\langle g\rangle$ would have been valid** — it is not the general case,
because $g_n$ is not constant in the measurements this model targets (Section 4 of
[`interference_model.md`](interference_model.md)). With uniform steps ($N\ge3$), $D=0$ and the
DC floor from Eq. (F21) reduces to $\tfrac12 b^2 g_0^2$, matching Eq. (F16)'s ideal part with
$\langle g^2\rangle = g_0^2$.

### 7.3 $g_n$ random, uncorrelated with $\delta_n$ — a leakage floor from contrast jitter alone

Let $g_n = \langle g\rangle + \varepsilon_n$ with $\varepsilon_n$ zero-mean, i.i.d. across
frames, variance $\mathrm{Var}(g)$, and statistically independent of $\delta_n$. Take the step
set to be perfectly uniform, so $R_\delta = 0$ ($N\ge2$). Then

$$R = \langle g\rangle R_\delta + \big\langle \varepsilon_n e^{i\delta_n}\big\rangle_n
= \frac{1}{N}\sum_{n=1}^N \varepsilon_n e^{i\delta_n} \tag{F24}$$

$\mathbb{E}[R] = 0$ (the jitter is unbiased), but it is not exactly zero for a given
realization: since the $\varepsilon_n$ are independent of each other and of $\delta_n$,

$$\mathbb{E}\big[|R|^2\big] = \frac{1}{N^2}\sum_{n=1}^N \mathbb{E}[\varepsilon_n^2] = \frac{\mathrm{Var}(g)}{N} \tag{F25}$$

so the RMS residual is $|R|_{\text{rms}} \sim \sqrt{\mathrm{Var}(g)/N}$. **Even with an ideal,
perfectly uniform step set, an uncorrelated random contrast jitter leaves a leakage floor** —
it just shrinks like $1/\sqrt N$ as more frames are averaged, unlike a deterministic
non-uniformity in $\delta_n$, which does not shrink with $N$ at all. The same argument applied
to $g_n^2$ shows $R^{(2)} = O(1/\sqrt N)$ as well, so $D = R^{(2)}-R^2 = O(1/\sqrt N) - O(1/N)
\to R^{(2)}$ to leading order: the fringe residual $D$ inherits this $1/\sqrt N$ floor directly
from $R^{(2)}$.

### 7.4 $g_n$ correlated with $\delta_n$ — a leakage that does not average away

If the contrast jitter is instead driven by the same physical process as the phase step —
e.g. vibration excited by the stepping actuator itself, or a contrast fade that tracks
progress through the sweep — the cross term in Eq. (F24) becomes a genuine, systematic
covariance rather than zero-mean noise, and it need **not** shrink with $N$. A worked example
makes this concrete: take a uniform step set ($N\ge3$, so $R_\delta = R_\delta^{(2)}=0$) with
contrast fully correlated with the cosine of the step, $g_n = \langle g\rangle + \kappa\cos\delta_n$
for a constant $\kappa$. Then, using $\cos\delta_n\,e^{i\delta_n} = \cos^2\delta_n + i\sin\delta_n\cos\delta_n
= \tfrac12(1+\cos2\delta_n) + \tfrac{i}{2}\sin2\delta_n$ and $\langle\cos2\delta_n\rangle_n=\langle\sin2\delta_n\rangle_n=0$,

$$R = \langle g\rangle R_\delta + \kappa\big\langle\cos\delta_n\,e^{i\delta_n}\big\rangle_n = 0 + \frac{\kappa}{2} \tag{F26}$$

a fixed, non-vanishing residual, independent of $N$, produced entirely by the $g$–$\delta$
correlation even though the step set itself is ideal. Working out $R^{(2)}$ for the same model —
$g_n^2 = \langle g\rangle^2+\tfrac12\kappa^2 + 2\langle g\rangle\kappa\cos\delta_n + \tfrac12\kappa^2\cos2\delta_n$,
whose only term surviving $\langle\,\cdot\,e^{i2\delta_n}\rangle_n$ under uniform $\delta_n$ is
the $\cos2\delta_n$ one, giving $\langle\cos2\delta_n\,e^{i2\delta_n}\rangle_n=\tfrac12$ — gives
$R^{(2)} = \kappa^2/4$, exactly equal to $R^2 = (\kappa/2)^2$. So for this particular
correlation pattern $D = R^{(2)}-R^2 = 0$ **exactly** (confirmed numerically), even though
$R=\kappa/2\neq0$: the correlation leaks fully into the *mean* but not at all into the
*variance*. This is not a general cancellation — it is a feature of a contrast fluctuation that
tracks $\cos\delta_n$ exactly; a correlation with a different phase or shape relative to
$\delta_n$ would leave $D\neq0$ and leak into both moments. The general lesson still holds: a
contrast fluctuation correlated with the phase-stepping sequence (rather than independent
vibration) can leak fringe structure — into the mean, the variance, or both depending on its
exact form — that averaging over more frames cannot remove.

## 8. Small-flicker limit

In our measurements $\alpha_n$ is normalized frame-by-frame so that $\langle\alpha\rangle = 1$
by definition, and the measured flicker standard deviation is small. Then

$$\langle\alpha^2\rangle = \langle\alpha\rangle^2 + \mathrm{Var}(\alpha) = 1 + \mathrm{Var}(\alpha) \approx 1$$

and every term proportional to $\mathrm{Var}(\alpha)$ in Eqs. (F21)–(F22) is negligible. Dropping
$\alpha$ entirely, the boxed results of Sections 4 and 6 reduce to

$$\langle I\rangle \;\approx\; a + b\,|R|\cos(\Phi+\psi) \tag{F27}$$

$$\mathrm{Var}(I) \;\approx\; \tfrac12 b^2\big(\langle g^2\rangle - |R|^2\big) + \tfrac12 b^2\,|D|\cos(2\Phi+\psi_D) \tag{F28}$$

(Eq. F28 keeps the exact, non-flicker DC-plus-fringe form of Eq. (F21) rather than the further
first-order approximation of Eq. (F22) — the two small parameters, step/contrast residuals and
flicker, are independent and are not conflated here.) In the ideal limit $|R|\to0$ (so $D\to
R^{(2)}\to0$ as well) these reduce to $\langle I\rangle \to a$ and $\mathrm{Var}(I) \to
\tfrac12 b^2\langle g^2\rangle$, so the fringe amplitude estimate is $b \approx
\sqrt{2\,\mathrm{Var}(I)/\langle g^2\rangle}$. The familiar estimate $b\approx\sqrt{2\,\mathrm{Var}(I)}$
used for a perfect, unit-contrast system is the special case $\langle g^2\rangle = 1$; using it
while the true $\langle g^2\rangle \neq 1$ over- or under-estimates $b$ according to whether
$\langle g^2\rangle$ is below or above 1 (equivalently, according to how much of
$\langle g^2\rangle = \langle g\rangle^2+\mathrm{Var}(g)$ comes from mean contrast loss versus
contrast jitter).

## 9. Summary table

| Quantity | Controlling residual | Harmonic | Vanishes when |
|---|---|---|---|
| $\langle I\rangle$ leakage term, $b\lvert R\rvert\cos(\Phi+\psi)$ | $\lvert R\rvert$ | 1st | $g$ constant and uniform steps ($N\ge2$); or, for varying $g$, only if $\langle g_n e^{i\delta_n}\rangle_n=0$ exactly (Section 7) |
| $\langle I^2\rangle$ cross term, $2ab\lvert R\rvert\cos(\Phi+\psi)$ | $\lvert R\rvert$ | 1st | same condition as above |
| $\langle I^2\rangle$ fringe term, $\tfrac12 b^2\lvert R^{(2)}\rvert\cos(2\Phi+\psi_2)$ | $\lvert R^{(2)}\rvert$ | 2nd | $g$ constant and uniform steps ($N\ge3$); or $\langle g_n^2 e^{i2\delta_n}\rangle_n=0$ exactly |
| $\mathrm{Var}(I)$ DC floor, $\tfrac12 b^2\big(\langle g^2\rangle-\lvert R\rvert^2\big)$ | $\langle g^2\rangle$, $\lvert R\rvert$ | — (DC, not a fringe) | never exactly zero unless $b=0$; reduces to $\tfrac12b^2\langle g^2\rangle$ once $\lvert R\rvert\to0$, and to $\tfrac12b^2g_0^2$ for constant contrast $g_n\equiv g_0$ (Section 7.2) |
| $\mathrm{Var}(I)$ fringe term, $\tfrac12 b^2\lvert D\rvert\cos(2\Phi+\psi_D)$, $D=R^{(2)}-R^2$ | $\lvert D\rvert$ | 2nd | $R^{(2)}=R^2$ exactly — in particular whenever $g$ is constant and uniform steps give $R=R^{(2)}=0$ ($N\ge3$, Section 7.1–7.2), but also for special correlated-$g$ patterns where $D=0$ even with $R\neq0$ (Section 7.4) |
| Random-jitter leakage floor, $|R|_{\text{rms}}\sim\sqrt{\mathrm{Var}(g)/N}$ | $\mathrm{Var}(g)$, uncorrelated with $\delta$ | 1st | $\mathrm{Var}(g)\to0$, or $N\to\infty$ (Section 7.3) |
| Correlated-jitter leakage, e.g. $R=\kappa/2$ | $\mathrm{Cov}(g,e^{i\delta})$ | 1st | $g_n$ and $\delta_n$ uncorrelated (Section 7.4) — **does not average out with $N$** |
| Background flicker term, $a^2\,\mathrm{Var}(\alpha)$ | $\mathrm{Var}(\alpha)$ | — (not step- or contrast-related) | stable light source, $\mathrm{Var}(\alpha)\to0$ (Section 8) |

## 10. Assumptions used

For reference, the assumptions invoked above, each already stated inline at its first use:

1. **Good $\delta$ coverage together with constant $g$** (Section 2.4) — not required for the
   derivations to hold, but is the specific condition under which $|R|$ and $|R^{(2)}|$ vanish
   exactly. For non-constant $g_n$, good $\delta$ coverage alone is not sufficient (Section 7).
2. **$\alpha \perp (g,\delta)$** (Section 3) — the source-power fluctuation is independent of
   both the contrast jitter and the commanded phase step, allowing frame averages of products
   $\alpha_n\,h(g_n,\delta_n)$ to factor (Eq. F10), used throughout Sections 4–6.
3. **No assumption of $g\perp\delta$.** Unlike the treatment of $\alpha$, this document does
   **not** assume the contrast factor is independent of the phase step; Section 7 works out the
   constant, independent, and correlated cases explicitly because the correlated case is
   physically plausible (vibration excited by the stepping actuator) and behaves qualitatively
   differently (its leakage does not average down with $N$).
4. **Static $a$, $b$, $\Phi$ over the acquisition** — inherited unchanged from assumptions 2–4 of
   [`interference_model.md`](interference_model.md); not re-derived here.
5. **Small flicker** (Section 8) — $\langle\alpha\rangle = 1$ by normalization and
   $\mathrm{Var}(\alpha)$ is small enough to neglect, specific to our measurement conditions.

## 11. Summary: working formulas (small-flicker limit)

This section collects the two results actually used in practice — the mean and variance of a
frame stack, in the small-flicker regime of Section 8 ($\langle\alpha\rangle=1$,
$\mathrm{Var}(\alpha)\approx0$) — so they can be read off without tracing back through the
derivation. For the general forms that keep $\alpha_n$ explicit, see the boxed Eqs. (F13) and
(F21).

**Quantities to compute**, directly from the realized per-frame sequences $\{g_n\}$, $\{\delta_n\}$:

$$R = \big\langle g_n\,e^{i\delta_n}\big\rangle_n, \quad \psi=\arg R; \qquad
R^{(2)} = \big\langle g_n^2\,e^{i2\delta_n}\big\rangle_n; \qquad
D = R^{(2)} - R^2, \quad \psi_D = \arg D; \qquad \langle g^2\rangle = \big\langle g_n^2\big\rangle_n \tag{F29}$$

**Mean and variance:**

$$\boxed{\;\langle I\rangle \;\approx\; a + b\,|R|\cos(\Phi+\psi)\;}$$

$$\boxed{\;\mathrm{Var}(I) \;\approx\; \tfrac12 b^2\big(\langle g^2\rangle - |R|^2\big) + \tfrac12 b^2\,|D|\cos(2\Phi+\psi_D)\;} \tag{F30}$$

matching Eqs. (F27) and (F28). Equivalently, in phasor form — more compact to implement, and
the harmonic structure is explicit — using $\mathrm{Re}[Re^{i\Phi}]=|R|\cos(\Phi+\psi)$ and
$\mathrm{Re}[De^{i2\Phi}]=|D|\cos(2\Phi+\psi_D)$:

$$\langle I\rangle \approx a + b\,\mathrm{Re}\big[R\,e^{i\Phi}\big], \qquad
\mathrm{Var}(I) \approx \tfrac12 b^2\big(\langle g^2\rangle-|R|^2\big) + \tfrac12 b^2\,\mathrm{Re}\big[D\,e^{i2\Phi}\big]$$

**Two things to remember:**

- **The mean leaks at the first harmonic of $\Phi$ ($R$), the variance at the second ($D$)** — a
  non-ideally-stepped stack shows residual fringe at the carrier frequency in a mean image, and
  at twice the carrier frequency in a variance image.
- **Ideal limit.** As $|R|\to0$ (good coverage), $D\to0$ as well, and
  $\langle I\rangle\to a$, $\mathrm{Var}(I)\to\tfrac12 b^2\langle g^2\rangle$ — giving the working
  fringe-amplitude estimate $b \approx \sqrt{2\,\mathrm{Var}(I)/\langle g^2\rangle}$.
