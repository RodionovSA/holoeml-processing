# Interference model

This document investigates a theoretical model for interference in a Mach–Zehnder imaging interferometer. The idea is to establish a relation between the measured intensity maps and the phase map of a sample. This model serves as a starting point for phase extraction algorithms.

## Starting equation

In a Mach–Zehnder imaging interferometer, through measurements we acquire intensity maps $I(x, y, t)$ that have both spatial and temporal dependence. To acquire phase information we should establish a model that relates the two beam intensities and their phase difference. A very general form of this relation can be written as

$$I(x, y, t) = a(x, y, t) + b(x, y, t)\,\cos\big(\phi(x, y, t)\big) \tag{1}$$

where

- $a(x, y, t)$ is a background intensity term (the two beam intensities plus background light),
- $b(x, y, t)$ is a fringe contrast term (contains the beam intensities and the coherence envelope),
- $\phi(x, y, t)$ is the phase difference between the two beams.

The fringe visibility is defined as $V(x, y, t) = b(x, y, t) / a(x, y, t)$. 

## Main assumptions

Equation (1) contains one observable $I(x, y, t)$ and three unknown functions $a(x, y, t)$, $b(x, y, t)$, and $\phi(x, y, t)$. So, generally the problem of phase extraction from intensity maps cannot be solved without some assumptions. Let us introduce the main assumptions that are mild and physically justified.

1. We assume that our sample and reference are stable enough and do not move. That means we can split $\phi(x, y, t)$ into a static, sample-induced phase difference map $\phi(x, y)$ — the quantity of interest — and a time-dependent instrumental term:
   $$\phi(x, y, t) = \phi(x, y) + \phi_{\text{inst}}(x, y, t) \tag{2}$$
2. We assume that there are only small variations between the paths and angles of the two arms that affect only the phase term. This gives us two things. First, we can further expand the phase:
   $$\phi(x, y, t) = \phi(x, y) + \phi_{\text{inst}}(x, y) + \delta(x, y, t) \tag{3}$$
   Since $\phi(x, y)$ and $\phi_{\text{inst}}(x, y)$ are both static, they cannot be separated from a single acquisition; a reference measurement is needed to isolate $\phi(x, y)$. Second, this leads to the fact that transmission amplitudes are fixed for sample and reference and only intensities can vary, which leads to a common factor term $\alpha(t)$ shared by $a$ and $b$. This does not, however, account for all of the time dependence of $b$: the coherence envelope also depends on the optical path difference between the arms, which drifts and is deliberately stepped during acquisition, so $b$ carries an additional envelope factor $\gamma(x, y, t)$, normalised so that $\gamma = 1$ at the nominal path difference. Assuming stray/background light in $a(x, y, t)$ is negligible compared to the two beam intensities,
   $$a(x, y, t) = \alpha(t)\,a(x, y) \quad b(x, y, t) = \alpha(t)\,\gamma(x, y, t)\,b(x, y) \tag{4}$$
   The envelope $\gamma(x, y, t)$ varies on the scale of the coherence length $L_c$. Its static spatial structure can be absorbed into $b(x, y)$, and since the path-difference excursion over the acquisition — from phase stepping, drift, and vibration combined — is common to the whole field, the remaining time variation is approximately spatially uniform: $\gamma(x, y, t) \approx g(t)$. Only when that excursion stays much smaller than $L_c$ does $g(t) \approx 1$; vibration-induced breathing of the path difference makes $g(t)$ vary in time.
3. Unlike the passive instrumental phase $\phi_{\text{inst}}(x, y)$, the geometric mismatch between the two wavefronts — tilt, defocus, and other low-order aberrations from the angle and curvature mismatch between the arms — is set deliberately and can differ from one acquisition to the next, so we separate out its contribution as a carrier term $\phi_{\text{carrier}}(x, y)$, a smooth, slowly varying function of $(x, y)$, with $\phi_{\text{inst}}(x, y)$ now understood to exclude it. We assume the carrier is constant within a single acquisition, so the only remaining time dependence is a spatially uniform piston $\delta(t)$, so:
   $$\phi(x, y, t) = \phi(x, y) + \phi_{\text{inst}}(x, y) + \phi_{\text{carrier}}(x, y) + \delta(t) \tag{5}$$
   Between acquisitions the carrier may differ, since the tilt and curvature mismatch can be readjusted, so it is estimated and removed from each measurement before a reference acquisition is subtracted.

## Full model

Substituting Eqs. (4) and (5) into Eq. (1) gives the full model:

$$I(x, y, t) = \alpha(t)\Big[a(x, y) + \gamma(x, y, t)\,b(x, y)\,\cos\big(\phi(x, y) + \phi_{\text{inst}}(x, y) + \phi_{\text{carrier}}(x, y) + \delta(t)\big)\Big] \tag{6}$$

The three static phase terms always appear together, so it is convenient to define the total static phase

$$\Phi(x, y) = \phi(x, y) + \phi_{\text{inst}}(x, y) + \phi_{\text{carrier}}(x, y) \tag{7}$$

In practice the model is used on a sequence of $N$ acquired frames. Using the reduction $\gamma(x, y, t) \approx g(t)$ from assumption 2, and writing $\alpha_n = \alpha(t_n)$, $g_n = g(t_n)$, $\delta_n = \delta(t_n)$ for frame $n$, Eq. (6) reduces to the working per-frame form

$$I_n(x, y) = \alpha_n\Big[a(x, y) + g_n\,b(x, y)\,\cos\big(\Phi(x, y) + \delta_n\big)\Big], \qquad n = 1 \dots N \tag{8}$$

where

- $\alpha(t)$ is the common source-power factor — spatially uniform, scaling $a$ and $b$ together,
- $a(x, y)$ is the static background intensity,
- $b(x, y)$ is the static fringe amplitude at the nominal path difference,
- $\gamma(x, y, t)$ is the normalised coherence envelope,
- $g_n$ is its per-frame reduction — the fringe-amplitude scaling from envelope breathing, e.g. due to vibration — sharing its origin with $\delta_n$ since both come from the same path-difference excursion,
- $\phi(x, y)$ is the sample-induced phase — the quantity of interest,
- $\phi_{\text{inst}}(x, y)$ is the static instrumental phase, carrier excluded, expected to repeat between acquisitions,
- $\phi_{\text{carrier}}(x, y)$ is the smooth low-order wavefront mismatch, which may change between acquisitions,
- $\delta(t)$ is the spatially uniform piston phase shift.

### Consequences

**Only $\Phi$ is observable — hence reference measurements.** The three static terms enter Eq. (7) identically and none varies with $t$, so no amount of phase stepping can separate them: a single acquisition yields $\Phi$, not $\phi$. A reference acquisition without the sample gives $\Phi_{\text{ref}} = \phi_{\text{inst}} + \phi_{\text{carrier}}^{\text{ref}}$, so $\phi = \Phi - \Phi_{\text{ref}}$. This works only if $\phi_{\text{inst}}$ repeats between the two acquisitions; the carrier need not, which is why it is estimated and removed separately first (assumption 3).

**At least three frames are needed.** After normalising out $\alpha_n$, each pixel in Eq. (8) carries three unknowns — $a$, $b$, $\Phi$ — and each frame supplies one equation, so recovering them requires $N \ge 3$ frames with distinct $\delta_n$. If the steps $\delta_n$ are themselves unknown, they become additional unknowns, requiring more frames and an iterative solver. $g_n$ is a further per-frame unknown: closed-form formulas that assume fixed fringe amplitude become biased when $g_n$ varies, but since $\alpha_n$ and $g_n$ are only two scalars per frame shared by every pixel, they can be estimated jointly with the pixel unknowns from the same frame sequence.

**Global offset ambiguity.** $\Phi$ and $\delta_n$ enter Eq. (8) only through the sum $\Phi + \delta_n$, so adding a constant $c$ to every $\delta_n$ while subtracting $c$ from $\Phi$ reproduces the same data exactly. A convention such as $\delta_1 = 0$ fixes the split, but the recovered $\Phi$ — and hence $\phi$ — still carries an unknown global constant, which is harmless for relative phase maps but must be resolved separately for an absolute optical path difference.

**Wrapping and sign.** $\cos$ is an even, $2\pi$-periodic function, so $\Phi$ is recovered only modulo $2\pi$ and must be unwrapped, and $\cos(\Phi) = \cos(-\Phi)$ leaves an overall sign undetermined from a single frame. With $N \ge 3$ frames the sine component of $\Phi$ is also recovered, fixing the sign — provided the ordering and sign of the steps $\delta_n$ are known; if they are not, $\Phi \to -\Phi$ remains unresolvable.

**$\alpha$ and $\gamma$ affect the data differently.** $\alpha_n$ is common to $a$ and $b$, so normalising each frame by its mean or a reference removes $\alpha_n$ entirely. $g_n$ scales only the cosine term, so this same normalisation does not remove it: the per-frame visibility is $V_n = g_n\,b/a$, which is precisely why a visibility measurement is a direct probe of the coherence envelope, and why residual envelope breathing biases the recovered fringe amplitude unless $g_n$ is estimated separately.

## Extraction pipeline

Recovering $\phi(x, y)$ from a measured stack follows the same sequence of stages regardless of which method performs the extraction:

1. **Acquire $N$ frames.** Collect a phase-shifted stack $I_n(x, y)$, $n = 1 \dots N$, per Eq. (8).
2. **Normalize.** Estimate and divide out the per-frame factor $\alpha_n$ from each frame, leaving $\alpha_n \approx 1$.
3. **Estimate $g_n$.** Measure the per-frame fringe gain from the normalized stack.
4. **Extract $a$, $b$, $\Phi$, $\delta_n$.** Solve the normalized, gain-corrected stack for the static background, fringe amplitude, total static phase, and per-frame steps.
5. **Remove the carrier.** Estimate and subtract $\phi_{\text{carrier}}(x, y)$ from $\Phi(x, y)$, leaving $\phi(x, y) + \phi_{\text{inst}}(x, y)$.
6. **Subtract a reference.** Repeat steps 1–5 on a reference acquisition (no sample) to get $\phi_{\text{inst}}(x, y)$, and subtract it to isolate $\phi(x, y)$.

Steps 2–3 are assumed done before phase extraction from here on: once a stack is normalized and $g_n$ is known, $\alpha_n = 1$ and $g_n$ can be supplied rather than re-estimated. Extraction-method documents (e.g. `docs/aia.md`) — and the extraction methods themselves — therefore work with the simplified per-frame model obtained by dropping $\alpha_n$ and $g_n$ from Eq. (8):

$$I_n(x, y) = a(x, y) + b(x, y)\,\cos\big(\Phi(x, y) + \delta_n\big) \tag{9}$$

Step 4 (extraction) is method-specific; see `docs/aia.md` for how AIA solves Eq. (9) for $a$, $b$, $\Phi$, and $\delta_n$.

## Loss function

Step 4 of the extraction pipeline — recovering $a$, $b$, $\Phi$, and $\{\delta_n\}$ from a measured stack $I_n^{\text{meas}}(x,y)$ — is posed, by every extraction method in this package, as minimizing the sum of squared residuals against Eq. (9):

$$\mathcal{L}\big(a, b, \Phi, \{\delta_n\}\big) = \sum_{n=1}^{N} \sum_{x,y} \Big[I_n^{\text{meas}}(x,y) - a(x,y) - b(x,y)\cos\big(\Phi(x,y) + \delta_n\big)\Big]^2 \tag{10}$$

$\mathcal{L}$ is nonlinear in $(\Phi, \delta_n)$ jointly, since they only enter through their sum inside a cosine — the same coupling noted under "At least three frames are needed" above. It becomes a *linear* least-squares problem the moment either block of unknowns is held fixed: fixing $\{\delta_n\}$ leaves a per-pixel linear fit for $(a,b,\Phi)$ — the classical closed-form phase-shifting formulas, valid when the steps are precisely known; fixing $(a,b,\Phi)$ leaves a per-frame linear fit for $\{\delta_n\}$'s quadrature components. Extraction methods differ in how they handle $\{\delta_n\}$ being unknown — e.g. by alternating between the two linear sub-problems; see `docs/aia.md` for how AIA does this, including a deliberate modification it makes to the per-frame sub-problem, and why.
