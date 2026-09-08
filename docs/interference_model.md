# Interference model

This document investigates a theoretical model for interference in Mach–Zehnder and Linnik imaging interferometers. The idea is to establish a relation between the measured intensity maps and the phase map of a sample. This model serves as a starting point for phase extraction algorithms.

## Starting equation

Both Mach–Zehnder and Linnik imaging interferometers are classical two beam interferometers. So, we can generally represent intensity maps $I(x, y, t)$ at the camera sensor as the coherent sum of two electric fields $E_1(x, y, z, t)$ and $E_2(x, y, z, t)$:

$$\begin{aligned} I(x, y, t) = {}&I_1(x,y,t) + I_2(x,y,t) \\ &+ 2\gamma(x, y, \Delta z, t)\sqrt{I_1(x,y,t)\,I_2(x,y,t)} \\ &\qquad\times\cos\big(\theta(x, y, \Delta z, t)\big)\,\cos\big(\phi(x, y, \Delta z, t)\big) \end{aligned}\tag{1}$$

where

- $I_1(x, y, t)$, $I_2(x, y, t)$ are the intensities of the two interfering beams at the sensor,
- $\gamma(x, y, \Delta z, t)$ is the coherence envelope (degree of coherence) at path difference $\Delta z$,
- $\theta(x, y, \Delta z, t)$ is the angle between the polarization vectors of the two fields,
- $\phi(x, y, \Delta z, t)$ is the phase difference between the two beams.

Eq. (1) splits naturally into a term independent of the phase and a term modulated by it. Grouping them this way is common in interferometry, so we define

$$\begin{aligned} a(x, y, t) &= I_1(x,y,t) + I_2(x,y,t) \\ b(x, y, t) &= 2\gamma(x, y, \Delta z, t)\sqrt{I_1(x,y,t)\,I_2(x,y,t)}\,\cos\big(\theta(x, y, \Delta z, t)\big) \end{aligned}\tag{2}$$

which lets us rewrite Eq. (1) in the compact form

$$I(x, y, t) = a(x, y, t) + b(x, y, t)\,\cos\big(\phi(x, y, \Delta z, t)\big) \tag{3}$$

where

- $a(x, y, t)$ is the background (DC) intensity: the sum of the two beam intensities,
- $b(x, y, t)$ is the fringe amplitude (AC term), which carries the coherence envelope $\gamma$ and the polarization mismatch $\cos\theta$.

## Main assumptions

Eq. (3) is fully general and, as it stands, cannot be solved from measured intensity maps alone: it carries more unknown functions than a single acquisition can constrain. To make the problem tractable we introduce a series of assumptions that hold for our setup, each one removing an unknown from the model.

1. **Linear polarization and non-chiral samples.** We assume the input beam is linearly polarized and that the samples under study are not chiral, i.e. they do not rotate the polarization plane. Neither arm of the interferometer then rotates the polarization state, so the two beams reaching the sensor stay co-polarized: $\theta(x, y, \Delta z, t) = 0$ and $\cos\theta = 1$. The polarization mismatch factor therefore drops out of $b$, leaving
   $$b(x, y, t) = 2\gamma(x, y, \Delta z, t)\sqrt{I_1(x,y,t)\,I_2(x,y,t)}. \tag{4}$$

2. **Stability and a common light source.** We assume the sample and the reference are stable and do not drift over an acquisition, so the sample-induced phase does not change in time and separates from a time-dependent instrumental term:
   $$\phi(x, y, \Delta z, t) = \phi(x, y) + \phi_{\text{inst}}(x, y, t) \tag{5}$$
   where $\phi(x, y)$ is the static, sample-induced phase difference — the quantity of interest — and $\phi_{\text{inst}}(x, y, t)$ is the time-dependent instrumental phase (carrying what was the $\Delta z$ dependence). We further assume both arms are fed by the same light source, so a fluctuation in source power scales both beams identically: the time dependence of each beam intensity separates from its spatial profile into one spatially uniform factor $\alpha(t)$,
   $$I_k(x, y, t) = \alpha(t)\,I_k(x, y), \qquad k = 1, 2 \tag{6}$$
   Since $\alpha(t)$ is common to both beams, it factors out of $a$ and $b$ entirely, leaving them static,
   $$a(x, y) = I_1(x,y) + I_2(x,y), \qquad b(x, y) = 2\sqrt{I_1(x,y)\,I_2(x,y)} \tag{7}$$
   and Eq. (3) becomes
   $$\begin{aligned} I(x, y, t) = \alpha(t)\Big[&a(x, y) \\ &+ \gamma(x, y, \Delta z, t)\,b(x, y)\,\cos\big(\phi(x, y) + \phi_{\text{inst}}(x, y, t)\big)\Big] \end{aligned}\tag{8}$$
   $\gamma(x, y, \Delta z, t)$ still carries its own time dependence and stays outside $b$; it is not addressed by this assumption.

3. **Instrumental phase: static aberration, carrier, and the phase step.** The instrumental phase separates into a part that is static over the acquisition and a time-dependent part introduced by the phase stepping. The static part further splits into a passive component $\phi_{\text{inst}}(x, y)$ (fixed path/aberration mismatch between the arms) and a deliberately-set carrier $\phi_{\text{carrier}}(x, y)$ — the tilt, defocus, and other low-order terms from the angle and curvature mismatch of the two wavefronts, smooth and slowly varying in $(x, y)$, set per acquisition and constant within it. The time-dependent part is the phase step, in general a field $\delta(x, y, t)$: a purely axial actuator motion contributes a path difference common to the whole field, but any tilt or bending of that motion, and any deformation of the arms during the sweep, makes the step vary across the field:
   $$\phi_{\text{inst}}(x, y, t) = \phi_{\text{inst}}(x, y) + \phi_{\text{carrier}}(x, y) + \delta(x, y, t) \tag{9}$$
   Split the step into its spatial average over the field and the remainder,
   $$\delta(x, y, t) = \delta(t) + \Delta(x, y, t), \qquad \big\langle\Delta(x, y, t)\big\rangle_{x, y} = 0 \tag{9a}$$
   so $\delta(t)$ is the piston — the uniform part of the step — and $\Delta(x, y, t)$ is its zero-mean spatial residual. The residual is smooth and low-order in $(x, y)$; expand it on a complete polynomial basis $p_j(x, y)$ of total degree 1 through $M$ — $J = \tfrac{(M+1)(M+2)}{2} - 1$ modes in all, e.g. $J = 2$ (just $x, y$) for tilt alone, $J = 5$ once curvature is included — each with zero average over the field:
   $$\Delta(x, y, t) = \sum_{j=1}^{J} c_j(t)\,p_j(x, y), \qquad \big\langle c_j(t)\big\rangle_t = 0 \tag{9b}$$
   Degree 1 ($x$, $y$) is a tilt of the step, degree 2 ($x^2$, $xy$, $y^2$) a curvature change; degree 0 is excluded from the basis, since that is exactly the piston. The two zero-mean conditions make the decomposition unambiguous: the spatial one (Eq. 9a) defines $\delta(t)$ as the field average of the true step, and the temporal one (Eq. 9b) separates the step residual from the carrier — a mode coefficient with a nonzero frame mean adds the same static pattern to every frame, indistinguishable from a static addition to the phase, and belongs in $\phi_{\text{carrier}}$ rather than in $\Delta$; only the part of the step that varies from frame to frame is $\Delta$.

   Substituting into Eq. (5) gives the total phase
   $$\begin{aligned} \phi(x, y, \Delta z, t) = {}&\phi(x, y) + \phi_{\text{inst}}(x, y) + \phi_{\text{carrier}}(x, y) \\ &+ \delta(t) + \Delta(x, y, t) \end{aligned}\tag{10}$$
   and Eq. (8) becomes
   $$\begin{aligned} I(x, y, t) = \alpha(t)\Big[&a(x, y) \\ &+ \gamma(x, y, \Delta z, t)\,b(x, y)\,\cos\big(\phi(x, y) + \phi_{\text{inst}}(x, y) \\ &\qquad\qquad + \phi_{\text{carrier}}(x, y) + \delta(t) + \Delta(x, y, t)\big)\Big] \end{aligned}\tag{11}$$
   Measured frame to frame on this setup, $\Delta$ is not negligible: the step carries a tilt that changes from frame to frame and, more weakly, a curvature change, at a level comparable to the low-order phase structure the model is otherwise meant to resolve, not a small correction to it. Whether this comes from the stepping actuator's own motion or from vibration of the interferometer arms during the sweep is not established; both produce the same functional form in Eq. (9a)–(9b).

4. **Random per-frame contrast factor.** Within an acquisition the coherence envelope $\gamma$ is effectively static — its variation over the sweep is negligible — so its static spatial structure absorbs into $b(x, y)$. Separately, mechanical vibration and index fluctuations during each frame's exposure reduce the fringe contrast by a spatially-uniform, temporally-random factor $g(t)$, with $\langle g \rangle$ set by the typical disturbance and frame-to-frame fluctuations that are non-periodic and vary between sessions. This factor multiplies only the modulation term:
   $$\gamma(x, y, \Delta z, t) = \gamma(x, y)\,g(t) \tag{12}$$
   Absorbing the static envelope $\gamma(x, y)$ into $b$ redefines the $b(x, y)$ of Eq. (7):
   $$b(x, y) = 2\gamma(x, y)\sqrt{I_1(x,y)\,I_2(x,y)} \tag{13}$$
   and Eq. (11) becomes
   $$\begin{aligned} I(x, y, t) = \alpha(t)\Big[&a(x, y) \\ &+ g(t)\,b(x, y)\,\cos\big(\phi(x, y) + \phi_{\text{inst}}(x, y) + \phi_{\text{carrier}}(x, y) \\ &\qquad\qquad + \delta(t) + \Delta(x, y, t)\big)\Big] \end{aligned}\tag{14}$$
   This follows from averaging the modulation term over the exposure: for a zero-mean, symmetric jitter $\varepsilon$ during the frame, $\langle\cos(\cdot + \varepsilon)\rangle = \langle\cos\varepsilon\rangle\cos(\cdot)$ — the sine term averages away — so the disturbance enters only as $g = \langle\cos\varepsilon\rangle \le 1$ on the modulation term and does not bias $\delta$ (an asymmetric $\varepsilon$ would also shift $\delta$). The assumption itself comes from experimental observation: the fringe amplitude is irregularly perturbed from frame to frame, like breathing, not periodic in time — sometimes present, sometimes not — with external vibration on the interferometer arms the likely source.

## Full model

The phase terms $\phi(x, y)$, $\phi_{\text{inst}}(x, y)$, and $\phi_{\text{carrier}}(x, y)$ are all static and always appear together as a sum, so it is convenient to wrap them into a single total static phase

$$\Phi(x, y) = \phi(x, y) + \phi_{\text{inst}}(x, y) + \phi_{\text{carrier}}(x, y) \tag{15}$$

Substituting into Eq. (14) gives the full model

$$I(x, y, t) = \alpha(t)\Big[a(x, y) + g(t)\,b(x, y)\,\cos\big(\Phi(x, y) + \delta(t) + \Delta(x, y, t)\big)\Big] \tag{16}$$

In practice this model is applied to a sequence of $N$ acquired frames. Writing $\alpha_n = \alpha(t_n)$, $g_n = g(t_n)$, $\delta_n = \delta(t_n)$, $\Delta_n(x, y) = \Delta(x, y, t_n) = \sum_j c_{jn}\,p_j(x, y)$ for frame $n$, Eq. (16) becomes the per-frame form

$$\begin{aligned} I_n(x, y) = \alpha_n\Big[&a(x, y) \\ &+ g_n\,b(x, y)\,\cos\big(\Phi(x, y) + \delta_n + \Delta_n(x, y)\big)\Big], \qquad n = 1 \dots N \end{aligned}\tag{17}$$

where

- $\alpha(t)$ is the spatially uniform source-power factor, common to $a$ and $b$,
- $a(x, y)$ is the static background (DC) intensity, $I_1(x,y) + I_2(x,y)$,
- $b(x, y)$ is the static fringe amplitude, $2\gamma(x,y)\sqrt{I_1(x,y)\,I_2(x,y)}$, including the static coherence envelope,
- $g(t)$ is the spatially uniform, temporally random per-frame contrast factor from vibration and index fluctuations during the exposure; it multiplies only the modulation term,
- $\Phi(x, y)$ is the total static phase, Eq. (15), comprising:
  - $\phi(x, y)$ — the sample-induced phase, the quantity of interest,
  - $\phi_{\text{inst}}(x, y)$ — the passive static instrumental phase (fixed path/aberration mismatch between the arms),
  - $\phi_{\text{carrier}}(x, y)$ — the deliberately-set carrier: a smooth, low-order tilt/defocus term from the wavefront mismatch, fixed within an acquisition but which may change between acquisitions,
- $\delta(t)$ is the piston phase step — the field average of the phase step (Assumption 3),
- $\Delta(x, y, t)$ is the zero-mean spatial residual of the phase step: low-order in $(x, y)$, dominated by a tilt that changes from frame to frame and, more weakly, a curvature change; its frame-independent part is not distinguishable from $\phi_{\text{carrier}}$ and is carried there instead (Assumption 3).

Expanding the cosine in Eq. (17), $\cos(\Phi + \delta_n + \Delta_n) = \cos\Phi\cos(\delta_n+\Delta_n) - \sin\Phi\sin(\delta_n+\Delta_n)$, and collecting the static factors gives a second, equivalent form of Eq. (17) that is linear in the per-pixel unknowns. Defining the quadrature components

$$u(x, y) = b(x, y)\cos\Phi(x, y), \qquad v(x, y) = -b(x, y)\sin\Phi(x, y) \tag{18}$$

Eq. (17) becomes

$$\begin{aligned} I_n(x, y) = \alpha_n\Big[&a(x, y) \\ &+ g_n\big(u(x, y)\cos(\delta_n+\Delta_n(x,y)) \\ &\qquad\quad + v(x, y)\sin(\delta_n+\Delta_n(x,y))\big)\Big], \qquad n = 1 \dots N \end{aligned}\tag{19}$$

Eq. (19) is linear in $(a, u, v)$ for a known phase step, which is why extraction methods work in this quadrature form rather than the polar one. With $\Delta_n \equiv 0$ the coefficients $\cos\delta_n$, $\sin\delta_n$ are the same at every pixel and Eq. (19) is one linear system for the whole field; with $\Delta_n \not\equiv 0$ the coefficients $\cos(\delta_n+\Delta_n)$, $\sin(\delta_n+\Delta_n)$ vary from pixel to pixel, and the system must be solved pointwise. Either way the polar quantities are recovered from $\Phi = \operatorname{atan2}(-v, u)$ and $b = \sqrt{u^2 + v^2}$.

## Uniform-piston limit

Setting $\Delta_n \equiv 0$ — equivalently $M = 0$, so $J = 0$ and Eq. (9b) has no modes — reduces Eq. (17) to

$$I_n(x, y) = \alpha_n\Big[a(x, y) + g_n\,b(x, y)\,\cos\big(\Phi(x, y) + \delta_n\big)\Big] \tag{20}$$

This is the model most extraction methods are built on, since it lets the phase step be recovered as a single scalar per frame rather than a field. It is exact only for a purely axial phase-stepping motion with the interferometer arms stable across the acquisition; more generally it is an approximation, and its accuracy is set by how large $\Delta$ actually is relative to $\Phi$.

Because a frame-independent step residual is absorbed into $\phi_{\text{carrier}}$ rather than into $\Delta$ (Eq. 9b), what the uniform-piston limit discards is specifically the part of the phase step that varies from frame to frame. The resulting error in the recovered fields is therefore systematic and smooth, low-order in $(x, y)$ — a tilt or curvature riding on the recovered phase and amplitude — rather than a random per-frame disturbance like the contrast factor $g(t)$ of Assumption 4, and it does not average down by acquiring more frames.
