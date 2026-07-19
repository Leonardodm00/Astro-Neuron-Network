# Theoretical documentation — Virtual-MEA pipeline for the Astro-Neuron-Network campaigns

This document is the complete theoretical account of the pipeline that converts
Astro-Neuron-Network HPC campaign output (neuron positions and spike times) into
synthetic 9-channel microelectrode-array (MEA) recordings, detects spikes on
those recordings, and saves the detected events together with the simulation's
parameter vector. It accompanies (and assumes as already built) the code
`eap_template_library.py`, `mea_probe.py`, `mea_synthesis.py`,
`mea_detection.py`, `process_campaign.py`.

Every claim below is tagged with its provenance:

- **[KB, full text]** — read from a PDF in the project's knowledge base.
- **[general reasoning]** — standard/textbook method or a design choice made
  during our design conversation; not verified against a specific retrieved
  full text in this session (no PubMed/bioRxiv tool was callable from this
  chat interface — see §9).
- **[design decision]** — a choice you made explicitly during the design
  conversation, recorded here for the permanent record.

Notation is introduced once, in full, at first use, and is not abbreviated
afterward; every symbol that depends on other quantities is written with its
full argument list every time it appears (e.g. $\tilde r(n,s)$, never a bare
$\tilde r$, once $n$ and $s$ are in scope). Any place where a simplification
or an abuse of notation is made, it is flagged explicitly in the surrounding
text.

---

## Contents

1. Problem statement and why the pipeline is structured as it is
2. The input data model
3. EAP template generation (the minimal Hodgkin&ndash;Huxley model)
4. The extracellular field: physical motivation and the phenomenological law actually used
5. Probe geometry
6. Signal synthesis
7. Detection theory
8. Ground-truth matching
8b. Diagnostic plots: what each one is evidence for
9. Assumptions, limitations, and full provenance table
10. Symbol glossary

---

## 1. Problem statement and why the pipeline is structured as it is

Each HPC campaign integrates a phenomenological spiking network (neurons,
astrocytes, synapses) and saves, per run, the **spike times** of every neuron
and astrocyte — not the continuous membrane potential $V(t)$ of any cell. This
is a deliberate and necessary storage choice: saving $V(t)$ for every neuron,
at the simulation's native time step, across the hundreds of thousands of runs
in a campaign, is infeasible.

The consequence for this pipeline is structural. An extracellular electrode
does not see spike times directly; it sees a continuous voltage produced by
current flowing through the extracellular medium around active cells. Since
that continuous signal was never simulated or stored, it must be
**synthesised**: at every stored spike time $t_k$ of neuron $n$, a fixed
extracellular waveform is deposited into each electrode's trace, scaled by how
strongly that neuron couples to that electrode. This is the single governing
design decision of the whole pipeline, and it factorises the problem into two
independent pieces that are built and can be reasoned about separately:

1. **What waveform shape does a spike produce extracellularly?**
   (§3 — the EAP template.)
2. **How does that waveform's amplitude fall off with distance from the
   electrode, and how is a physical culture geometry mapped onto electrode
   coordinates?** (§4, §5 — the spatial/scaling model and the probe.)

These two pieces are then combined and corrupted by noise (§6), after which a
detector recovers discrete events from the continuous trace (§7), and those
events are linked back to the ground-truth spikes that generated them (§8) for
validation.

---

## 2. The input data model

For a campaign directory laid out as `campaign_<TAG>/topo_<k>/iter_<n>.npz`,
topology is shared across all iterations of a given `topo_<k>`:

- `topology.npz` contains `N_pos`, an array of shape $(N_n, 2)$ giving the
  planar position $\mathbf r_n = (x_n, y_n) \in \mathbb R^2$, in micrometres,
  of every neuron $n \in \{0, 1, \dots, N_n-1\}$, drawn uniformly in the square
  arena $[0, c_{\max}] \times [0, c_{\max}]$.
- `topology_meta.json` (falling back to `job_args.json`) contains $c_{\max}$,
  the arena side length in micrometres (default $1100\ \mu\text{m}$, fixed per
  campaign).
- `iter_<n>.npz` contains, among other fields, `spk_N_t` (the spike times
  $t_k \in \mathbb R_{>0}$, in seconds, for every spike $k$ in that run),
  `spk_N_i` (the emitting neuron index $i_k \in \{0,\dots,N_n-1\}$ for every
  spike $k$), the 36-component swept parameter vector
  $\boldsymbol\theta_{\text{sweep}} \in \mathbb R^{36}$ (called `params` in the
  file), and the SBI inference coordinate vector (called `theta`).

Everything downstream is built from $\{\mathbf r_n\}_{n=1}^{N_n}$, $c_{\max}$,
and the spike-time list $\{(t_k, i_k)\}_{k=1}^{K}$ of a given iteration.

---

## 3. EAP template generation

### 3.1 Why a biophysical model is used for the waveform shape

The extracellular action potential (EAP) is not an arbitrary pulse; its
biphasic shape — a fast negative deflection (the trough, coincident with the
inward $\text{Na}^+$ current driving the somatic upstroke) followed by a
slower positive lobe (repolarisation) — is a signature of the underlying
transmembrane current dynamics. To reproduce this shape rather than impose an
ad hoc analytic pulse, the waveform is derived from an actual simulated
Hodgkin&ndash;Huxley (HH) neuron.

### 3.2 The minimal HH model used [KB, full text: Pospischil et al. 2008,
*Minimal Hodgkin&ndash;Huxley type models for different classes of cortical and
thalamic neurons*, Biol Cybern 99:427&ndash;441]

The regular-spiking excitatory (RS-exc) class of that paper is used, since it
matches the cortical/pyramidal-like excitatory neurons populating your
cultures. The state of one neuron at time $\tau$ (a local time variable used
only within this template-generation sub-model; not to be confused with the
network-level simulation time $t$) is the vector

$$\mathbf y(\tau) = \big(V(\tau),\, m(\tau),\, h(\tau),\, n(\tau),\, p(\tau)\big) \in \mathbb R^5,$$

where $V(\tau)$ is the membrane potential (mV), $m(\tau)$ and $h(\tau)$ are the
activation and inactivation gating variables of the fast $\text{Na}^+$
current, $n(\tau)$ is the activation gating variable of the delayed-rectifier
$\text{K}^+$ current, and $p(\tau)$ is the activation gating variable of the
slow non-inactivating $\text{M}$-current. Each gating variable takes values in
$[0,1]$ and represents, following the Hodgkin&ndash;Huxley convention, the
fraction of independent gates in the open state **[KB, full text]**.

The membrane equation, for every $\tau$ in the simulated interval, is

$$C_m \frac{dV(\tau)}{d\tau} = I_{\text{app}}(\tau) - I_L(V(\tau)) - I_{Na}\big(V(\tau), m(\tau), h(\tau)\big) - I_{Kd}\big(V(\tau), n(\tau)\big) - I_M\big(V(\tau), p(\tau)\big),$$

with $C_m = 1\ \mu\text{F/cm}^2$ the specific membrane capacitance, and
$I_{\text{app}}(\tau)$ an externally injected step current (used only to
elicit spikes; see §3.4). The four current terms are, for every $\tau$
**[KB, full text, Eqs. 2 and 4 of Pospischil et al. 2008]**:

$$I_L(V) = \bar g_L\,(V - E_L), \qquad I_{Na}(V,m,h) = \bar g_{Na}\, m^3 h\,(V - E_{Na}), \qquad I_{Kd}(V,n) = \bar g_{Kd}\, n^4\,(V - E_K), \qquad I_M(V,p) = \bar g_M\, p\,(V - E_K),$$

where $\bar g_L, \bar g_{Na}, \bar g_{Kd}, \bar g_M$ are maximal conductance
densities ($\text{mS/cm}^2$) and $E_L, E_{Na}, E_K$ are the leak, sodium, and
potassium reversal potentials (mV). Fixed constants used throughout, all
**[KB, full text]** except where noted: $E_{Na} = 50\ \text{mV}$,
$E_K = -90\ \text{mV}$; $E_L$ and $\bar g_L$ are held at a fixed density
(**[design decision]**: Table 1 of the source paper reports $\bar g_L$ as an
absolute conductance in nS per fitted cell, which requires a soma-area
assumption to convert to a density; rather than invent a soma area, a fixed
density $\bar g_L = 0.1\ \text{mS/cm}^2$ is used for every template, since
$\bar g_L$ mainly sets the resting potential and rheobase, not the shape of
the action potential itself).

Each gating variable obeys a first-order kinetic equation of the form, for
every $\tau$ and for each gate $x \in \{m, h, n\}$,

$$\frac{dx(\tau)}{d\tau} = \alpha_x\big(V(\tau)\big)\,\big(1 - x(\tau)\big) - \beta_x\big(V(\tau)\big)\, x(\tau),$$

with voltage-dependent rate functions **[KB, full text, Eq. 4]**, all
functions of $V$ and the threshold-adjustment parameter $V_T$ (mV):

$$\alpha_m(V) = \frac{-0.32\,(V - V_T - 13)}{\exp\!\big[-(V - V_T - 13)/4\big] - 1}, \qquad \beta_m(V) = \frac{0.28\,(V - V_T - 40)}{\exp\!\big[(V - V_T - 40)/5\big] - 1},$$

$$\alpha_h(V) = 0.128\, \exp\!\big[-(V - V_T - 17)/18\big], \qquad \beta_h(V) = \frac{4}{1 + \exp\!\big[-(V - V_T - 40)/5\big]},$$

$$\alpha_n(V) = \frac{-0.032\,(V - V_T - 15)}{\exp\!\big[-(V - V_T - 15)/5\big] - 1}, \qquad \beta_n(V) = 0.5\, \exp\!\big[-(V - V_T - 10)/40\big].$$

The slow $M$-current gate follows a steady-state/time-constant formulation
rather than $\alpha/\beta$ form **[KB, full text, §2.2.2]**:

$$\frac{dp(\tau)}{d\tau} = \frac{p_\infty\big(V(\tau)\big) - p(\tau)}{\tau_p\big(V(\tau)\big)}, \qquad p_\infty(V) = \frac{1}{1 + \exp\!\big[-(V+35)/10\big]}, \qquad \tau_p(V) = \frac{\tau_{\max}}{3.3\, \exp\!\big[(V+35)/20\big] + \exp\!\big[-(V+35)/20\big]}.$$

**Removable singularity.** Each of $\alpha_m, \beta_m, \alpha_n$ has the form
$c\,x / \big(\exp(x/y) - 1\big)$, which is of indeterminate form $0/0$ at
$x=0$. By l'Hôpital's rule, for every such rate function, the limit as
$x \to 0$ is $c\,y$ (finite and well-defined) — for example
$\lim_{x \to 0} \dfrac{x}{\exp(x/y)-1} = y$. The implementation evaluates this
limit directly whenever $|x/y| < 10^{-6}$, rather than the raw ratio, to avoid
floating-point cancellation. This is a numerical-implementation detail, not a
modification of the model — the two expressions agree to machine precision
away from $x=0$ and agree exactly (analytically) at $x=0$.

### 3.3 Parameter heterogeneity — the template *library*
**[KB, full text: Pospischil et al. 2008, Table 1]**

You asked for a library, not a single template. The source paper fitted this
five-parameter model $(\bar g_{Na}, \bar g_{Kd}, V_T, \bar g_M, \tau_{\max})$
independently to $13$ RS-exc cells from rat somatosensory cortex and reports,
for each parameter, the sample mean and standard deviation across those $13$
fits (its Table 1):

| parameter | mean | SD | unit |
|---|---|---|---|
| $\bar g_{Na}$ | $50$ | $10$ | $\text{mS/cm}^2$ |
| $\bar g_{Kd}$ | $4.8$ | $1.4$ | $\text{mS/cm}^2$ |
| $V_T$ | $-61.5$ | $3.2$ | $\text{mV}$ |
| $\bar g_M$ | $0.13$ | $0.05$ | $\text{mS/cm}^2$ |
| $\tau_{\max}$ | $1123.5$ | $500.5$ | $\text{ms}$ |

To build a library of $L$ templates, indexed $\ell \in \{1,\dots,L\}$, one
parameter vector $\boldsymbol\vartheta_\ell = (\bar g_{Na,\ell}, \bar g_{Kd,\ell}, V_{T,\ell}, \bar g_{M,\ell}, \tau_{\max,\ell})$
is drawn, independently for each $\ell$ and independently across the five
components, as

$$\bar g_{Na,\ell} \sim \mathcal N(50,\,10^2)\big|_{\text{clipped}}, \quad \bar g_{Kd,\ell} \sim \mathcal N(4.8,\,1.4^2)\big|_{\text{clipped}}, \quad V_{T,\ell} \sim \mathcal N(-61.5,\,3.2^2)\big|_{\text{clipped}}, \quad \bar g_{M,\ell} \sim \mathcal N(0.13,\,0.05^2)\big|_{\text{clipped}}, \quad \tau_{\max,\ell} \sim \mathcal N(1123.5,\,500.5^2)\big|_{\text{clipped}},$$

where $\mathcal N(\mu,\sigma^2)\big|_{\text{clipped}}$ denotes the Gaussian
truncated to a physiologically plausible range by clipping (not
resampling/rejection) — **[design decision]**, chosen so a rare extreme draw
still yields *a* valid parameter set rather than looping indefinitely.

**Explicitly flagged simplification.** Treating the five components as
*independent* draws is an abuse of the source data: the $13$ cells in Table 1
were fitted as five-tuples, and different parameters of the same cell are very
plausibly correlated (e.g. a cell with high $\bar g_{Na}$ fitted from a
particular preparation may systematically also show a shifted $V_T$). The
paper reports only marginal means/SDs, not the joint covariance or the raw
per-cell table beyond what is reproduced in §3.3's table (which *does* list
each of the $13$ fitted tuples — see §9 for the note on this). Independent
per-component sampling is the simplification actually implemented; drawing
whole rows of Table 1's per-cell tuples (with replacement/jitter) would
preserve the empirical correlation structure and is a natural extension if
richer heterogeneity is later required.

For each $\ell$, the neuron is simulated (§3.4) with parameters
$\boldsymbol\vartheta_\ell$ to obtain one normalised template $w_\ell(\tau)$. A
network neuron $n$ is then assigned one template index
$\text{tpl}(n) \in \{1,\dots,L\}$ (uniformly at random, seeded for
reproducibility), so that heterogeneous neurons in the network produce
heterogeneous — but still individually biophysically self-consistent — EAP
shapes.

### 3.4 Eliciting spikes and extracting the surrogate waveform

For each drawn $\boldsymbol\vartheta_\ell$, the neuron is initialised at rest,
$V(0) = E_L$, with each gating variable at its steady state at that voltage,
$x(0) = \alpha_x(E_L) / \big(\alpha_x(E_L) + \beta_x(E_L)\big)$ for
$x \in \{m,h,n\}$ and $p(0) = p_\infty(E_L)$, and driven by a step current
$I_{\text{app}}(\tau) = I_0$ for $\tau \ge \tau_{\text{on}}$ (and $0$ before),
with default $I_0 = 7\ \mu\text{A/cm}^2$, chosen empirically to elicit several
spikes within a $250\ \text{ms}$ window for every $\boldsymbol\vartheta_\ell$
drawn from the clipped ranges. The system is integrated with an adaptive-step
implicit solver (`scipy.integrate.solve_ivp`, method `LSODA`, since the fast
$\text{Na}^+$/$\text{K}^+$ kinetics make the system numerically stiff near the
upstroke) and resampled onto a uniform high-resolution grid at
$dt_{\text{hr}} = 0.02\ \text{ms}$ (50 kHz).

**The EAP surrogate.** A single, space-clamped compartment cannot itself
produce an extracellular field: by construction of the space-clamp, the total
transmembrane current at every instant equals the injected current
$I_{\text{app}}(\tau)$ exactly (there is no second location for current to
return to — see the multipole argument in §4.1, which formalises why a
*true* single point source is physically degenerate for this purpose).
Consequently the extracellular waveform shape cannot be read directly off any
current in this model; it must be taken from a **surrogate**. Three surrogates
are implemented; the default, used throughout, is the **capacitive current**:

$$w_{\text{raw}}(\tau) = -C_m \frac{dV(\tau)}{d\tau}.$$

**[general reasoning; flagged simplification.]** This is a standard, but not
rigorously derived, single-compartment stand-in for the extracellular
waveform. Its justification is qualitative: $-C_m\,dV/d\tau$ is large and
negative exactly during the fast $\text{Na}^+$-driven upstroke (mirroring the
sharp negative trough seen in real extracellular recordings at the site of
inward current) and swings positive during repolarisation (mirroring the
positive lobe from the return $\text{K}^+$ current), giving the correct
qualitative biphasic shape without requiring a multi-compartment model with an
explicit somato-dendritic current-return path. It is not a first-principles
derivation of what a real, spatially extended neuron would produce; the two
alternative surrogates provided —
$w_{\text{raw}}(\tau) = -\big(I_{Na}(\tau) + I_{Kd}(\tau) + I_M(\tau)\big)$
(the summed *ionic*, as opposed to capacitive, current) and
$w_{\text{raw}}(\tau) = -d^2V(\tau)/d\tau^2$ (a discrete second-derivative
form, motivated by the line-source approximation of Holt & Koch-type
multi-compartment models, where the extracellular potential near a point is
proportional to the local second spatial derivative of $V$, here substituted
by a temporal second derivative purely as a shape heuristic) — are offered as
alternatives precisely because none of the three has a rigorous claim to being
"the" correct single-compartment surrogate; the choice among them changes the
waveform's symmetry (biphasic vs. more triphasic) but not the overall
pipeline.

**Windowing, alignment, and normalisation.** From the elicited spike train,
the last $n_{\text{avg}}$ spikes (default $3$) that have enough surrounding
samples are selected — using late spikes rather than the first avoids any
transient-onset artefact and captures the neuron in its adapted, steady firing
regime. For each selected spike, a window
$\big[\tau^\star - \tau_{\text{pre}},\ \tau^\star + \tau_{\text{post}}\big]$
(default $\tau_{\text{pre}} = 0.8\ \text{ms}$, $\tau_{\text{post}} = 1.6\ \text{ms}$)
is extracted around $\tau^\star = \arg\min_\tau w_{\text{raw}}(\tau)$, the
local negative trough of the surrogate near that spike, and the $n_{\text{avg}}$
windows are averaged sample-by-sample to suppress residual numerical noise.
The resulting mean waveform is normalised so its trough equals exactly $-1$:

$$w_\ell(\tau) = \frac{\overline{w}_{\text{raw}}(\tau)}{-\min_\tau \overline{w}_{\text{raw}}(\tau)}, \qquad \text{so that } \min_\tau w_\ell(\tau) = -1 \text{ for every } \ell.$$

This normalisation is exactly what allows the amplitude scale to be
**decoupled** from the shape: $w_\ell(\cdot)$ is a pure, dimensionless
waveform *shape*; all physical amplitude information is reintroduced later
(§4.3) as a single multiplicative factor per (neuron, electrode) pair.

### 3.5 Resampling to the acquisition rate

Each template $w_\ell(\tau)$, stored on the $50\ \text{kHz}$ high-resolution
grid, is resampled once to the target recording sample rate $f_s$ (default
$f_s = 10110.09\ \text{Hz}$, a **[design decision]**, your specified value) by
linear interpolation:

$$w_\ell^{(f_s)}[j] = w_\ell\big(j / f_s\big), \qquad j = 0, 1, \dots, \Big\lceil f_s\,\big(\tau_{\text{pre}}+\tau_{\text{post}}\big) \Big\rceil,$$

using piecewise-linear interpolation of the high-resolution samples at the
non-integer high-resolution index corresponding to $\tau = j/f_s$.
**[design decision, justified]:** linear interpolation is used, rather than
band-limited (sinc/Fourier) resampling, specifically *because* the template is
a short, sharply-peaked, non-periodic transient — sinc interpolation of such a
signal produces Gibbs-phenomenon ringing on the flanks of the trough, which
would corrupt exactly the feature (a clean, sharp, single trough) that
detection depends on. Linear interpolation is monotone-safe (it cannot
introduce new extrema between samples) at the cost of a small, bounded
underestimate of the true peak/trough amplitude between original samples,
which the smoke test bounds empirically at under $10\%$ at $f_s = 10110.09\ \text{Hz}$
for this template's duration.

---

## 4. The extracellular field: physical motivation and the phenomenological law actually used

### 4.1 Why a dipole is the physically correct leading-order term
**[general reasoning — standard volume-conductor electrostatics; not
verified against a specific retrieved full text in this session]**

This subsection is retained in full because it is the physical justification
for choosing a *dipole-like* decay exponent, even though (§4.2) the law
actually implemented is phenomenological rather than a literal solution of
this equation.

Model the extracellular medium as an infinite, homogeneous, purely resistive
(ohmic) volume conductor of conductivity $\sigma$ (S/m), and model neuron $n$,
at a given instant $t$, as a finite set of point current sources/sinks
$\{I_{n,j}(t)\}_{j=1}^{M}$ located at positions $\{\mathbf r_{n,j}\}_{j=1}^{M}$
(e.g. one per morphological compartment), where $I_{n,j}(t)$ is the
transmembrane current leaving compartment $j$ of neuron $n$ into the
extracellular space at time $t$ (outward positive). In the quasi-static
approximation of Maxwell's equations (valid at neurophysiological
frequencies, where propagation delays across the tissue are negligible
compared with the signal's timescale), the extracellular potential at any
field point $\mathbf r_e$ **not** coincident with a source, produced by
neuron $n$ alone at time $t$, is the superposition of point-source solutions
of Poisson's equation:

$$\varphi_n(\mathbf r_e, t) = \frac{1}{4\pi\sigma} \sum_{j=1}^{M} \frac{I_{n,j}(t)}{\lVert \mathbf r_e - \mathbf r_{n,j} \rVert}.$$

**Charge conservation.** For every neuron $n$ and every instant $t$, Kirchhoff's
current law applied to the whole cell requires that the net current the cell
exchanges with the extracellular medium is zero (whatever current enters at
one compartment must leave at another; the cell does not accumulate net
charge):

$$\sum_{j=1}^{M} I_{n,j}(t) = 0 \qquad \text{for every neuron } n \text{ and every time } t.$$

Now expand $\varphi_n(\mathbf r_e, t)$ in the **far field**, i.e. for field
points $\mathbf r_e$ whose distance from the neuron's centroid
$\mathbf r_{n,c} = \frac{1}{M}\sum_j \mathbf r_{n,j}$ is large compared with the
neuron's own spatial extent, $\lVert \mathbf r_e - \mathbf r_{n,c}\rVert \gg \max_j \lVert \mathbf r_{n,j} - \mathbf r_{n,c}\rVert$.
A first-order Taylor expansion of $1/\lVert\mathbf r_e - \mathbf r_{n,j}\rVert$
about $\mathbf r_{n,c}$ gives, for every $j$,

$$\frac{1}{\lVert \mathbf r_e - \mathbf r_{n,j} \rVert} \approx \frac{1}{\lVert \mathbf r_e - \mathbf r_{n,c} \rVert} + \frac{(\mathbf r_{n,j} - \mathbf r_{n,c}) \cdot (\mathbf r_e - \mathbf r_{n,c})}{\lVert \mathbf r_e - \mathbf r_{n,c} \rVert^{3}} + O\!\left(\frac{1}{\lVert \mathbf r_e - \mathbf r_{n,c} \rVert^{3}}\right).$$

Substituting into $\varphi_n(\mathbf r_e, t)$ and using charge conservation to
eliminate the leading (monopole) term term-by-term:

$$\varphi_n(\mathbf r_e, t) = \underbrace{\frac{1}{4\pi\sigma \lVert \mathbf r_e - \mathbf r_{n,c} \rVert} \sum_{j} I_{n,j}(t)}_{=\ 0 \text{ for every } t \text{, by charge conservation}} \; + \; \frac{(\mathbf r_e - \mathbf r_{n,c})}{4\pi\sigma \lVert \mathbf r_e - \mathbf r_{n,c} \rVert^{3}} \cdot \underbrace{\sum_j I_{n,j}(t)\,(\mathbf r_{n,j} - \mathbf r_{n,c})}_{\displaystyle =: \ \mathbf p_n(t)} \; + \; O\!\left(\frac{1}{\lVert \mathbf r_e - \mathbf r_{n,c}\rVert^3}\right),$$

where $\mathbf p_n(t) \in \mathbb R^2$ (or $\mathbb R^3$, depending on the
embedding) is the neuron's instantaneous **current-dipole moment**. The
monopole term vanishes *identically*, for every $n$ and every $t$, because of
charge conservation — not approximately, and not only in the far field. The
dipole term is therefore the **leading non-vanishing term** of the expansion,
and it decays as

$$\big\lVert \varphi_n(\mathbf r_e, t) \big\rVert \; \sim \; \frac{\lVert \mathbf p_n(t) \rVert}{4\pi\sigma \lVert \mathbf r_e - \mathbf r_{n,c} \rVert^{2}}, \qquad \text{i.e. an inverse-square, dipole-like, decay law.}$$

This is the rigorous sense in which "the dipole is the right physical
approximation": it is not a preference between two equally valid models, it is
the mathematically forced leading-order behaviour once the (always-true)
zero-monopole condition is applied. A pure $1/r$ (monopole) law, as used by
some simplified virtual-electrode codes (including the earlier version of
this pipeline), is only exactly correct for an idealised, physically
unrealisable isolated point source with $\sum_j I_{n,j}(t) \ne 0$, or as a
*near-field* approximation valid only when $\mathbf r_e$ is so close to one
particular compartment $j^\star$ that the field is locally dominated by that
compartment's own current alone (i.e. $\lVert \mathbf r_e - \mathbf r_{n,j^\star}\rVert \ll \lVert \mathbf r_{n,j^\star} - \mathbf r_{n,j}\rVert$
for all other $j \ne j^\star$) — a regime in which the multipole expansion
itself is not yet valid, so neither law is rigorously justified there. This
near-field breakdown is exactly the physical (not merely numerical) reason
the near-field floor $r_{\min}$ is introduced in §4.2: below $r_{\min}$, *no*
far-field power law — dipole or monopole — has a rigorous claim to
correctness, so the model is deliberately flattened rather than trusted.

**Explicitly flagged simplification carried into the implementation.** The
rigorous dipole term above is a *vector* quantity: its magnitude at
$\mathbf r_e$ depends not only on $\lVert \mathbf r_e - \mathbf r_{n,c}\rVert$
but also on the angle between $\mathbf p_n(t)$ and
$(\mathbf r_e - \mathbf r_{n,c})$ (a $\cos\theta$-type angular dependence,
vanishing for field points in the plane perpendicular to $\mathbf p_n(t)$).
Using it correctly would require knowing each neuron's dipole *orientation* —
physically, its somato-dendritic axis — which is not defined for the
point-neuron representation used by the network model, and which you
explicitly declined to assume (an oriented dipole was considered and rejected
in the design conversation: real network somata in a dissociated culture have
no well-defined dendritic axis). What is carried into §4.2 is therefore only
the **radial decay rate** of the dipole far field, $1/r^2$, applied
**isotropically** (i.e. with no angular dependence) — the magnitude of an
idealised dipole averaged over orientation, not the true anisotropic dipole
field. This is a phenomenological choice made explicitly at your request
("dipole-like decay of the intensity"), not a claim that the implemented law
solves the electrostatics problem above.

### 4.2 The scaling law actually implemented **[design decision]**

Given §4.1, and given that an absolute physical calibration is unattainable
without additional invented constants (§4.3), the spatial model used is a
**phenomenological amplitude scaling law** — a decay rate chosen to match the
dipole far field's radial exponent, applied as a pure distance-dependent
multiplier on the (already shape-normalised) template. It is emphatically
*not* a numerical solution of $\varphi_n(\mathbf r_e, t)$ above: it uses
no $\sigma$, no $\mathbf p_n(t)$, and no orientation. For neuron $n$ at
position $\mathbf r_n = (x_n,y_n)$ and any point $\mathbf r_s = (x_s,y_s)$ in
the plane (in particular, an electrode sub-site — see §5), define the in-plane
Euclidean distance

$$d(n,s) := \lVert \mathbf r_s - \mathbf r_n \rVert_2 = \sqrt{(x_s-x_n)^2 + (y_s - y_n)^2},$$

and the **floored** distance

$$\tilde r(n,s) := \max\big(d(n,s),\, r_{\min}\big),$$

where $r_{\min}$ (default $10\ \mu\text{m}$, on the order of a soma radius) is
the near-field floor motivated in §4.1: below this distance neither a
monopole nor a dipole far-field law is rigorously applicable, so the
attenuation is capped rather than allowed to diverge. The amplitude delivered
by neuron $n$ at point $s$ is

$$A(n,s) := A_{\text{ref}} \left( \frac{r_{\text{ref}}}{\tilde r(n,s)} \right)^{n_{\text{dec}}},$$

where $r_{\text{ref}}$ is a reference distance (default $30\ \mu\text{m}$),
$A_{\text{ref}}$ is the amplitude (µV) a neuron would deliver at exactly
$\tilde r(n,s) = r_{\text{ref}}$ (defined in §4.3), and $n_{\text{dec}}$ is the
**decay exponent**, the single free parameter that encodes the choice of
physical analogy:

- $n_{\text{dec}} = 2$ — **the default**, matching the dipole far-field radial
  decay rate derived in §4.1 (your explicit request: "I want a dipole like
  decay of the intensity of the EAP from the source").
- $n_{\text{dec}} = 1$ — inverse-distance, matching the original
  `Virtual_electrodes.py` and a pure (unphysical, but simpler) monopole law;
  retained as a one-flag alternative (`--n_dec 1`).

### 4.3 Amplitude calibration without an absolute physical scale

A single-compartment surrogate (§3.4) cannot supply an absolute microvolt
scale on its own: converting the dimensionless HH surrogate peak into a
physical $A_{\text{ref}}$ under the (rejected) physical-dipole route of §4.1
would require the extracellular conductivity $\sigma$, an effective dipole
separation length $\ell$, and a membrane area $S$ — three constants with no
principled value available either from the campaign data or from the project
knowledge base (checked; not found — see §9). Rather than assert plausible
textbook values for a different physical system (brain tissue) and apply them
to a dissociated culture in conductive medium, where the applicability is
itself questionable, the pipeline sidesteps the absolute scale entirely:

**Only the signal-to-noise ratio matters for detection.** The detector (§7)
thresholds at $k$ multiples of a noise estimate computed *from the same
signal*; multiplying every amplitude in the system (signal and noise) by the
same constant leaves every detection decision unchanged. So instead of fixing
$A_{\text{ref}}$ in absolute µV from unknown physical constants, it is fixed
relative to the target post-filter noise level:

$$A_{\text{ref}} := \mathrm{SNR}_{\text{ref}} \cdot \sigma_{\text{noise}},$$

with $\mathrm{SNR}_{\text{ref}}$ (default $15$) the desired signal-to-noise
ratio of a neuron sitting exactly at $r_{\text{ref}}$ from an electrode, and
$\sigma_{\text{noise}}$ (default $5\ \mu\text{V}$) the target robust
post-band-pass noise level (§6.2). With the defaults,
$A_{\text{ref}} = 15 \times 5 = 75\ \mu\text{V}$.

### 4.4 The reach cutoff $r_{\max}$ **[your design idea, formalised]**

For a culture of $N_n \sim 10^5$&ndash;$10^6$ neurons, evaluating the weight
$A(n,s)$ for every neuron against every electrode sub-site is wasteful: the
overwhelming majority of neurons are far enough from the probe that their
contribution is smaller than the noise floor and therefore has no effect on
any detection decision. Define the reach cutoff $r_{\max}$ as the distance at
which a neuron's amplitude, evaluated at the reference law of §4.2, falls to a
fraction $\gamma$ (default $0.5$) of the target noise level:

$$A_{\text{ref}} \left( \frac{r_{\text{ref}}}{r_{\max}} \right)^{n_{\text{dec}}} \; \stackrel{!}{=} \; \gamma\, \sigma_{\text{noise}} \qquad \Longrightarrow \qquad \boxed{\; r_{\max} = r_{\text{ref}} \left( \frac{A_{\text{ref}}}{\gamma\, \sigma_{\text{noise}}} \right)^{1/n_{\text{dec}}} \;}$$

Neurons with $\tilde r(n,s) > r_{\max}$ for every sub-site $s$ of every
electrode are excluded from the weight computation and from the
spike-rendering loop for that electrode, at negligible cost to the synthesised
signal (their true contribution was, by construction, below $\gamma$ of the
noise floor at every site).

With the defaults $A_{\text{ref}}=75\ \mu\text{V}$, $\gamma=0.5$,
$\sigma_{\text{noise}}=5\ \mu\text{V}$, $r_{\text{ref}}=30\ \mu\text{m}$:

$$n_{\text{dec}}=2: \quad r_{\max} = 30\sqrt{75/2.5} = 30\sqrt{30} \approx 164.3\ \mu\text{m}. \qquad\qquad n_{\text{dec}}=1: \quad r_{\max} = 30\,(75/2.5) = 900\ \mu\text{m}.$$

The steeper $n_{\text{dec}}=2$ decay makes this cutoff genuinely effective:
only neurons within roughly $164\ \mu\text{m}$ of a probe sub-site can
possibly contribute a detectable event, versus a much larger $900\ \mu\text{m}$
reach under the slower $1/r$ law.

---

## 5. Probe geometry

The probe is a $3\times 3$ grid of $E = 9$ square electrodes, each of edge
length $\epsilon = 25\ \mu\text{m}$, with centre-to-centre pitch
$\Delta = 60\ \mu\text{m}$, centred on the culture. For electrode indices
$(a,b) \in \{-1,0,1\}^2$ (so the centre electrode is $(0,0)$), the electrode
centre coordinates are

$$\mathbf c_{a,b} = \left( \frac{c_{\max}}{2} + a\,\Delta,\ \ \frac{c_{\max}}{2} + b\,\Delta \right), \qquad a,b \in \{-1,0,1\},$$

giving $E=9$ centres $\{\mathbf c_e\}_{e=1}^{9}$. Each electrode's finite
square face is not treated as a single point; it is tiled by an
$n_{\text{sub}} \times n_{\text{sub}}$ grid of point recording sub-sites
(default $n_{\text{sub}}=4$, so $16$ sub-sites per electrode), at offsets from
the electrode centre

$$\boldsymbol\delta_{u,v} = \epsilon \left( \frac{u+0.5}{n_{\text{sub}}} - 0.5,\ \ \frac{v+0.5}{n_{\text{sub}}} - 0.5 \right), \qquad u,v \in \{0, 1, \dots, n_{\text{sub}}-1\},$$

so that sub-site centres are evenly spaced within (and not on the boundary of)
the electrode's physical face, giving sub-site set
$S_e = \{\mathbf c_e + \boldsymbol\delta_{u,v}\}_{u,v}$, $|S_e| = n_{\text{sub}}^2$.

The finite-face electrode signal is modelled as the arithmetic mean over its
sub-sites (a first-order approximation to the true surface integral of
$\varphi$ over the electrode face, exact in the limit $n_{\text{sub}} \to \infty$
for a continuous $\varphi$); this mean is linear in the per-sub-site
amplitude, so it is convenient to fold sub-site averaging directly into a
single per-(neuron, electrode) weight (used in §6):

$$W(n,e) := \frac{1}{|S_e|} \sum_{s \in S_e} A(n,s) = \frac{1}{n_{\text{sub}}^2} \sum_{u=0}^{n_{\text{sub}}-1}\sum_{v=0}^{n_{\text{sub}}-1} A\big(n, \mathbf c_e + \boldsymbol\delta_{u,v}\big).$$

$W(n,e)$ is computed once per topology (since $\{\mathbf r_n\}$ and the probe
geometry are shared across all iterations of that topology) and reused for
every iteration.

---

## 6. Signal synthesis

### 6.1 The synthesis equation

For electrode $e \in \{1,\dots,9\}$, at sample time $t$ (seconds, on the
$f_s$-Hz grid), the synthesised raw channel is

$$x_e[t] = \underbrace{\sum_{n=1}^{N_n} W(n,e) \sum_{k:\, i_k = n} w^{(f_s)}_{\text{tpl}(n)}\big[t - t_k\big]}_{\text{signal: superposition of every neuron's EAPs, distance-and-template-weighted}} \; + \; \underbrace{\eta_e[t]}_{\text{noise}}, \qquad \eta_e[t] \overset{\text{i.i.d.}}{\sim} \mathcal N(0,\, \sigma_{\text{pre}}^2)\ \text{independently per channel } e,$$

where the inner sum runs over every spike $k$ emitted by neuron $n$ (i.e.
every $k$ with $i_k = n$), $w^{(f_s)}_{\text{tpl}(n)}$ is the template
assigned to neuron $n$ (§3.3), resampled to $f_s$ (§3.5), and indexed relative
to the spike time $t_k$ (zero outside the template's finite support), and
$\sigma_{\text{pre}}$ is the **pre-filter** white-noise standard deviation
(distinct from the **post-filter** target $\sigma_{\text{noise}}$ used in §4;
the relation between the two is derived in §6.2).

**What the noise term represents.** $\eta_e[t]$ is not purely "instrument"
noise in a narrow sense; because of the reach cutoff (§4.4), the true
contribution of every neuron beyond $r_{\max}$ is discarded rather than summed.
The additive Gaussian term stands in for both genuine electronic/thermal noise
*and* the unresolved aggregate hash of the (very many, individually
sub-threshold) distant neurons that were excluded — the standard
"resolve the near units explicitly, lump the far background into noise"
approximation used throughout the extracellular-recording literature
**[general reasoning]**.

Only neurons that pass the reach cutoff for at least one electrode are
included in the outer sum (§4.4); their contribution is rendered once into a
scratch buffer and added into the appropriate channels, rather than looping
sample-by-sample.

### 6.2 Calibrating $\sigma_{\text{pre}}$ from a target $\sigma_{\text{noise}}$

The detector (§7) operates on the **band-passed** signal, and its noise
threshold is set relative to the band-passed noise level, not the raw
pre-filter level — but §4.3 fixes the target in terms of the *post*-filter
level $\sigma_{\text{noise}}$, since that is what actually competes with the
signal at detection time. The two are related by the filter's power transfer
function. Let $H(\omega)$ be the (single-pass) frequency response of the
digital band-pass filter (Butterworth, order $4$, applied as second-order
sections). The detector uses **zero-phase** filtering (`scipy.signal.sosfiltfilt`,
i.e. forward-then-backward application), which is the standard way to filter
without introducing a time (phase) shift that would displace detected spike
times relative to the true underlying event. For a zero-phase forward-backward
filter, applying $H$ once forward and once backward on the time-reversed
signal is equivalent, at the power-spectral-density level, to applying the
*squared* magnitude response twice — i.e. the effective power transfer is
$|H(\omega)|^4$ (the amplitude transfer of `filtfilt` alone is the familiar
$|H(\omega)|^2$; squaring again converts the amplitude relation into the
*power/variance* relation used here). For pre-filter white noise with variance
$\sigma_{\text{pre}}^2$ (i.e. flat power spectral density $\sigma_{\text{pre}}^2$
at every $\omega$), the post-filter variance is

$$\sigma_{\text{post-filter, raw}}^{2} = \sigma_{\text{pre}}^2 \cdot \frac{1}{2\pi}\int_{-\pi}^{\pi} \big|H(\omega)\big|^{4}\, d\omega.$$

Rather than evaluate this integral analytically for the specific realised
digital filter (which would require tracking the exact bilinear-transform
warping of the requested $300$&ndash;$3000\ \text{Hz}$ band at the given
$f_s$), the pipeline measures the gain **empirically**: a long ($\ge 5\ \text{s}$)
probe of unit-variance white noise is generated with a fixed seed, passed
through the exact same `sosfiltfilt` call the real detector will use, and its
robust post-filter noise level is measured with the identical estimator used
downstream (§7.2), $\hat\sigma(\mathbf 1) = \mathrm{median}(|x_{\text{probe}}|)/0.6745$
(evaluated on unit pre-filter RMS, so this **is** the dimensionless gain).
This ties the calibration to the *robust* estimator actually used for
detection (not the raw variance), which matters because the robust estimator
and the raw standard deviation are only proportional for perfectly Gaussian
noise; measuring the actual downstream statistic directly avoids that
mismatch entirely. Then

$$\sigma_{\text{pre}} = \frac{\sigma_{\text{noise}}}{\hat\sigma(\mathbf 1)}.$$

This calibration is performed once per pipeline configuration (band, order,
$f_s$) and reused across every channel, every iteration, and every topology.

---

## 7. Detection theory

### 7.1 Band-pass filtering

Every channel is band-passed at $[f_{\text{lo}}, f_{\text{hi}}]$ (default
$300$&ndash;$3000\ \text{Hz}$) with a Butterworth filter of order $4$,
implemented in second-order-section form for numerical stability, applied
zero-phase (`sosfiltfilt`) as described in §6.2. The band is chosen to
suppress slow local-field-potential-range fluctuations below $f_{\text{lo}}$
and high-frequency noise above $f_{\text{hi}}$ while passing the fast
biphasic EAP transient (§3.4), whose energy is concentrated in the
hundreds-of-Hz to few-kHz range — consistent with the bands used in your own
project's MEA literature **[KB, full text]**: $100$&ndash;$5000\ \text{Hz}$
(high-pass at $100\ \text{Hz}$, sampled at $10\ \text{kHz}$) in the Kleefstra-
syndrome study, and $200$&ndash;$3000\ \text{Hz}$ in the astrocyte-EV study
(both cited fully in §7.3 and §9).

### 7.2 Robust noise estimation

The per-channel noise level is estimated, **[general reasoning — the
median-absolute-deviation ("Quiroga") estimator is a widely used convention
in the extracellular spike-detection literature, generally attributed to
Quiroga, Nadasdy & Ben-Shaul (2004); this attribution was not verified against
a specific retrieved full text in this session — see §9]**, as

$$\hat\sigma_e = \frac{\mathrm{median}\big(|x_e|\big)}{0.6745},$$

where $x_e$ is the band-passed channel $e$. The constant $0.6745$ is the
median absolute deviation of a standard normal random variable: if
$Z \sim \mathcal N(0,1)$, then $\mathrm{median}(|Z|) = \Phi^{-1}(3/4) \approx 0.6745$,
where $\Phi^{-1}$ is the standard normal quantile function; dividing by this
constant makes $\hat\sigma_e$ an unbiased estimator of $\sigma$ under the null
hypothesis that $x_e$ is Gaussian noise. The reason the **median** is used
rather than the ordinary sample standard deviation is robustness: a channel
containing genuine spikes is a mixture of a large fraction of Gaussian
background samples and a small fraction of much larger-magnitude spike
samples; the sample standard deviation is heavily inflated by that small
contaminating fraction (its breakdown point is $0\%$ — a single arbitrarily
large outlier can make it arbitrarily large), whereas the median has a
breakdown point of $50\%$ and is essentially unaffected by a spike rate well
under half the samples, so $\hat\sigma_e$ estimates the *background* noise
level even on a channel that is actively spiking.

### 7.3 Thresholding

A sample is flagged as supra-threshold if $x_e[t] < -k\,\hat\sigma_e$
(negative-going polarity, the default, matching the sign of the dominant EAP
trough established in §3.4; positive and two-sided variants are also
implemented). The multiplier $k$ is a free, tunable parameter (default $5$).
For context, your own project's MEA literature uses
$k=4.5$ **[KB, full text: the Kleefstra-syndrome study, whose reported
protocol is "the noise threshold was set at $\pm 4.5$ standard deviations",
sampled at $10\ \text{kHz}$ after a $100\ \text{Hz}$ high-pass]** and $k=6$
**[KB, full text: the astrocyte-EV study, whose reported protocol is "spikes
were detected using an adaptive threshold crossing set to six times the
standard deviation of the estimated noise on each electrode", over a
$200$&ndash;$3000\ \text{Hz}$ band]**. Both are within the plausible range for
$k$ in this pipeline; $5$ is a **[design decision]** midpoint default.

### 7.4 Event formation: lockout-merge

A single supra-threshold *sample* is not the same thing as a single detected
*event*: because the detector operates on a zero-phase-filtered signal, a
single true spike typically produces not one isolated threshold crossing but
a short burst of several crossings in immediate succession — the trough
itself, plus filter-ringing side-lobes on either side of it, all of which can
individually exceed $k\hat\sigma_e$ when the underlying SNR is high (this was
observed directly during testing: at $\mathrm{SNR}\sim 15$&ndash;$40$ near an
electrode, a single true spike produced as many as two supra-threshold
crossings under a naive "one run of consecutive crossings = one event" rule,
because the ringing side-lobes were themselves separated from the main trough
by a brief sub-threshold gap).

The detector therefore uses a **lockout-merge** rule, which is the standard
non-paralyzable dead-time model of detection theory applied at the level of
*groups* of crossings rather than individual samples: letting
$\{u_1 < u_2 < \dots\}$ be the sample indices where $x_e[u] < -k\hat\sigma_e$,
consecutive crossings are merged into the same event whenever the gap between
them is no larger than a lockout window $\rho$ (in samples; default
$\rho = 2\ \text{ms} \cdot f_s$),

$$u_i,\, u_{i+1} \text{ belong to the same event} \iff u_{i+1} - u_i \le \rho,$$

and each resulting merged group $\{u_{i}, u_{i+1}, \dots, u_j\}$ contributes
**exactly one** detected event, located at the sample of most extreme
amplitude within the merged span,

$$\hat t = \arg\min_{u \in \{u_i,\dots,u_j\}} x_e[u].$$

This single rule serves two purposes simultaneously, which is why it replaced
an earlier two-stage design (group consecutive crossings into runs, *then*
separately enforce a refractory dead-time between run-picked extrema): it
both (a) collapses the multi-lobe ringing complex of one true spike into one
event, and (b) enforces that no two returned events can be closer than $\rho$,
which is the conventional refractory/dead-time constraint used to prevent a
detector from double-counting a single biological refractory period. The
trade-off inherent to any dead-time model is explicit and unavoidable: two
*genuinely distinct* spikes (from two different neurons, or even the same
neuron firing unusually fast) landing on the same electrode within $\rho$ of
each other are merged into a single reported event. Resolving such collisions
requires amplitude/shape-based spike sorting on the multi-channel waveform,
which is outside the scope of a single-channel threshold detector; $\rho$ is
exposed as a tunable parameter precisely so it can be set according to the
expected burst statistics of a given culture.

---

## 8. Ground-truth matching

Because the true spike list $\{(t_k, i_k)\}$ that generated the synthetic
trace is known exactly (it is the campaign's own `spk_N_t`/`spk_N_i`), every
detected event $\hat t$ on electrode $e$ can be linked back to its most
plausible source. For each detected event, the nearest true spike time (over
all neurons, not just the strongest contributor) within a matching window
(default $\pm 2\ \text{ms}$) is found; ties (multiple true spikes equally
close in time) are broken by preferring the neuron with the larger weight
$W(n,e)$ to that electrode, since that neuron is the more probable physical
source of the event's amplitude. Detected events with no true spike within
the matching window are labelled with source index $-1$ and are the
pipeline's operational definition of a **false positive** for validation
purposes.

**Caveat, stated explicitly.** This matching is a *nearest-neighbour
heuristic*, not a causal proof. In particular, during the merge described in
§7.4, a reported event's matched "source" is simply whichever true spike
happens to be temporally closest to the reported extremum time — if two
different neurons' spikes were merged into one event, the match records only
one of them, and the discarded one is not separately flagged as "detected but
mislabelled." This is an accepted limitation for the pipeline's intended use
(bulk validation of detection quality across many simulated cultures), not a
guarantee suitable for training a spike-sorter that assumes ground-truth
labels are causally exact.

---

## 8b. Diagnostic plots: what each one is evidence *for*

The plotting layer (`mea_plots.py`) is deliberately not decorative. Each plot
is chosen to falsify a specific way the pipeline could be silently wrong;
this section records which claim each one tests, since a plot nobody knows how
to read is worse than no plot.

| plot | the claim it tests | what a failure looks like |
|---|---|---|
| `01_probe_layout` | The probe really is a $3\times3$ grid of $25\ \mu\text{m}$ faces at $60\ \mu\text{m}$ pitch, centred at $(c_{\max}/2, c_{\max}/2)$ (§5), and $r_{\max}$ (§4.4) encloses a sensible neuron population. | Probe off-centre; electrodes the wrong size relative to the arena; the $r_{\max}$ circle enclosing everything (no compute saving) or nothing (no signal). |
| `02_weight_decay` | $W(n,e)$ as *actually computed* by `compute_weights` obeys the analytic law $A(n,s) = A_{\text{ref}}(r_{\text{ref}}/\tilde r(n,s))^{n_{\text{dec}}}$ of §4.2, including the $r_{\min}$ floor and the sub-site average of §5. | Scatter departing from the overlaid analytic curve: a wrong exponent shows as a different slope on the log-log axes; a units error shows as a vertical offset; a broken floor shows as divergence at small $\tilde r(n,s)$. |
| `03_template_library` | The HH integration (§3.2&ndash;3.4) produced biphasic waveforms of physiological duration, normalised to trough $=-1$, with genuine cross-template heterogeneity (§3.3). | A monophasic or ringing template; wrong duration; all templates exactly superimposed (heterogeneity not actually applied). |
| `04_stacked_traces_raw` | Synthesis (§6.1) produced a signal at all, on the right channels, with a plausible noise floor. | Flat channels; a single channel carrying everything; amplitudes orders of magnitude from the target SNR. |
| `05_traces_detections` | The threshold $-k\hat\sigma_e$ (§7.3) sits sensibly relative to both noise and signal, and events land on troughs. | A field of red (unmatched) markers = threshold too low; visible troughs with no marker = threshold too high. |
| `06_raster` | Detected events across the $9$ channels correspond in time to the ground-truth activity of the neurons within reach (§8). | Detected bursts with no corresponding true activity, or dense true bursts that the probe never registers. |
| `07_zoom` | **The lockout-merge of §7.4 works**: one true spike yields exactly one detection despite band-pass ringing. | Two or more markers on the lobes of a single waveform &mdash; precisely the failure this rule was introduced to fix. |
| `08_detected_waveforms` | The detector is triggering on EAPs, not on noise excursions: aligned snippets should average to a clean biphasic mean. | A shapeless cloud whose mean is near zero &mdash; the signature of threshold crossings on noise. |
| `09_amplitude_hist` | There is real separation between the detected population and the threshold, i.e. detection is not marginal. | Amplitudes piling up against the threshold line, meaning the reported spike count is an artefact of where $k$ was set. |
| `10_detection_summary` | Per-channel bookkeeping is consistent: channels nearer the active population detect more; $\hat\sigma_e$ (§7.2) is uniform across channels as expected for i.i.d. noise; matched-event timing error is sub-millisecond. | Wildly unequal $\hat\sigma_e$ (a synthesis bug, since the noise is i.i.d. by construction); a timing-error distribution that is broad rather than concentrated near zero. |

**Selection of the plotted window.** Time-domain plots do not show $t=0$ by
default. At $f_s \approx 10\ \text{kHz}$ and $180\ \text{s}$ of simulated
activity, a full-record plot is both unreadable and slow to render, and
cultures are typically near-silent between network bursts, so a fixed window
at the start of the record would frequently show nothing at all. Instead the
window of width $\Delta t_{\text{plot}}$ (default $2\ \text{s}$) maximising
the number of detected events is selected:

$$[t_0^\star,\, t_0^\star + \Delta t_{\text{plot}}], \qquad t_0^\star = \arg\max_{t_0 \in \{0,\ \Delta t_{\text{plot}},\ 2\Delta t_{\text{plot}},\ \dots\}} \ \Big| \big\{\, m \ :\ \hat t_m \in [t_0,\, t_0+\Delta t_{\text{plot}}) \,\big\} \Big|,$$

where $\hat t_m$ is the time of detected event $m$ (§7.4) and $|\cdot|$ denotes
set cardinality. **[design decision]** Note this is a coarse, bin-aligned
search over non-overlapping windows rather than a continuous sliding maximum:
it is a plotting convenience, and a burst straddling a bin boundary may be
shown split. It deliberately biases the displayed window towards the *most
active* period, which is what one wants for inspection but is emphatically
**not** a representative sample of the recording &mdash; no statistic should be
read off these figures.

**Raster row capping.** The ground-truth raster panel shows at most
$N_{\text{rows}}$ neurons (default $120$), selected as those with the largest
$\max_e W(n,e)$ among neurons passing the reach mask. A culture with $N_n$ in
the $10^5$ range cannot be rendered as one row per neuron, and the neurons
that matter for interpreting an electrode trace are exactly the
strongly-coupled ones. **[design decision]** This is a display cap only; it
never affects synthesis, detection, or any saved numerical output.

**Cost and failure isolation.** Plots are off by default and, when enabled,
geometry plots are drawn once per topology and signal plots only for the first
iteration of each topology &mdash; a campaign of hundreds of thousands of runs
would otherwise generate a comparable number of images. Plotting is wrapped so
that any exception is caught and reported as a warning *after* the numerical
`.npz` has already been written atomically: a plotting failure can never cost
a completed computation.

---

## 9. Assumptions, limitations, and full provenance table

| Design element | Provenance | Note |
|---|---|---|
| Minimal HH model, gating kinetics, fixed constants ($E_{Na}, E_K$) | **KB, full text** | Pospischil et al. 2008, §2.2.1&ndash;2.2.2 |
| RS-exc parameter distribution (Table 1 means/SDs) | **KB, full text** | Pospischil et al. 2008, Table 1 |
| Independent per-parameter sampling for the library | **Design decision / flagged simplification** | Table 1's within-cell parameter correlations are discarded; see §3.3 |
| Capacitive-current EAP surrogate $-C_m\,dV/d\tau$ | **General reasoning, flagged** | Standard qualitative stand-in; not a first-principles derivation |
| Linear-interpolation resampling of the template | **Design decision, justified** | Chosen specifically to avoid Gibbs ringing near the sharp trough |
| Multipole expansion, monopole vanishing, dipole leading order | **General reasoning — standard volume-conductor electrostatics** | Not checked against a specific retrieved full text this session; textbook-level result (e.g. treatments following Nunez & Srinivasan; Pettersen & Einevoll-style forward modelling) |
| $n_{\text{dec}}=2$ phenomenological scaling law, isotropic (no orientation) | **Design decision**, physically motivated by the row above | Explicitly *not* a solution of the electrostatics problem; see §4.1's flag |
| SNR-relative amplitude calibration ($A_{\text{ref}} = \mathrm{SNR}_{\text{ref}}\cdot\sigma_{\text{noise}}$) | **Design decision** | Forced by the absence of a groundable $\sigma,\ell,S$ triple; see §4.3 |
| $r_{\max}$ reach-cutoff formula | **Your design idea, formalised** | §4.4 |
| Probe geometry (3×3, 25 µm edge, 60 µm pitch, sub-site averaging) | **Your specification** | §5 |
| Additive Gaussian noise as background-hash proxy | **General reasoning** | Standard "resolve near units, lump far background" approximation |
| $300$&ndash;$3000\ \text{Hz}$ Butterworth order-4 zero-phase band-pass | **Design decision**, informed by KB conventions below | |
| Detection threshold conventions ($k=4.5$; $100$ Hz high-pass; $10$ kHz sampling) | **KB, full text** | Kleefstra-syndrome MEA study |
| Detection threshold conventions ($k=6$; $200$&ndash;$3000$ Hz band-pass) | **KB, full text** | Astrocyte-EV MEA study |
| MAD-based robust noise estimator ($\mathrm{median}(\lvert x\rvert)/0.6745$) | **General reasoning, flagged** | Standard convention, commonly attributed to Quiroga, Nadasdy & Ben-Shaul 2004; **not** verified against that paper's full text — no PubMed/bioRxiv tool was reachable from this chat interface (checked directly; only browser-automation tools were returned by the deferred-tool search) |
| Lockout-merge event formation, $\rho = 2\ \text{ms}$ default | **Design decision**, standard non-paralyzable dead-time model | §7.4 |
| Ground-truth nearest-neighbour matching | **Design decision** | §8, with explicit caveat on merge collisions |
| Diagnostic plot set and busiest-window selection | **Design decision** | §8b; display conveniences, never affecting saved numerical output |
| Per-topology process parallelism + single-threaded BLAS per worker | **Design decision**, standard HPC hygiene | Not a modeling choice, but recorded here since it affects reproducibility of run *timing* (not results): every worker pins `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` to 1 to prevent oversubscription; see `README.md` section 5 for the full reasoning |

**On PubMed/bioRxiv access.** Your standing instructions ask for the PubMed
and bioRxiv/medRxiv connectors to be used to verify claims. This chat
interface was checked directly for a callable PubMed or bioRxiv tool; none is
available (the only tools returned were browser-automation tools for a
Chrome extension, unrelated to literature search). The PubMed/bioRxiv/Consensus
servers mentioned in this environment's configuration are wired for a
different feature entirely (calls made from *inside* a generated code
artifact to the Anthropic API), not for direct use in this conversation. Every
claim above that would ordinarily be strengthened by such a search is
instead explicitly flagged as **general reasoning, not full-text-verified**,
per your hard rule that unverified sources never carry the same confidence as
verified ones. If PubMed/bioRxiv access becomes available (e.g. via a
connector enabled through Settings), the Quiroga 2004 attribution in
particular is the first item worth verifying.

**Summary of what is *not* claimed.** This pipeline does not claim to be a
physically calibrated forward model of extracellular recording (no volume
conductor equation is numerically solved; no absolute impedance, tissue
conductivity, or dipole orientation is used). It is a phenomenological,
internally consistent signal-and-noise generator whose spatial decay rate is
chosen to match the *qualitative* physical behaviour derived in §4.1, and
whose absolute scale is defined relative to a target signal-to-noise ratio
rather than to physical units.

---

## 10. Symbol glossary

| Symbol | Meaning | First defined |
|---|---|---|
| $N_n$ | number of neurons in the culture | §2 |
| $\mathbf r_n = (x_n,y_n)$ | position of neuron $n$, µm | §2 |
| $c_{\max}$ | arena side length, µm | §2 |
| $t_k, i_k$ | time and emitting-neuron index of spike $k$ | §2 |
| $\boldsymbol\theta_{\text{sweep}} \in \mathbb R^{36}$ | swept parameter vector of the run | §2 |
| $\tau$ | local time variable within the single-neuron HH sub-model | §3.2 |
| $V(\tau), m(\tau), h(\tau), n(\tau), p(\tau)$ | HH state variables | §3.2 |
| $C_m, \bar g_{Na}, \bar g_{Kd}, \bar g_M, \bar g_L, E_{Na}, E_K, E_L, V_T, \tau_{\max}$ | HH biophysical constants/parameters | §3.2 |
| $\ell \in \{1,\dots,L\}$ | template-library index | §3.3 |
| $\boldsymbol\vartheta_\ell$ | drawn HH parameter tuple for template $\ell$ | §3.3 |
| $w_\ell(\tau)$ | normalised EAP template $\ell$, high-resolution grid | §3.4 |
| $w^{(f_s)}_\ell[j]$ | template $\ell$ resampled to $f_s$ | §3.5 |
| $\text{tpl}(n)$ | template index assigned to neuron $n$ | §3.3 |
| $\sigma$ | extracellular conductivity (used only in the §4.1 derivation; **not** used in the implemented law) | §4.1 |
| $\mathbf p_n(t)$ | current-dipole moment of neuron $n$ at time $t$ | §4.1 |
| $d(n,s), \tilde r(n,s)$ | in-plane and floored distance from neuron $n$ to point $s$ | §4.2 |
| $r_{\min}, r_{\text{ref}}, r_{\max}$ | near-field floor, reference, and reach-cutoff distances, µm | §4.2, §4.4 |
| $n_{\text{dec}}$ | decay exponent of the scaling law | §4.2 |
| $A(n,s)$ | amplitude of neuron $n$ at point $s$ | §4.2 |
| $A_{\text{ref}}$ | reference amplitude at $r_{\text{ref}}$, µV | §4.3 |
| $\mathrm{SNR}_{\text{ref}}$ | target SNR at $r_{\text{ref}}$ | §4.3 |
| $\sigma_{\text{noise}}$ | target post-filter robust noise level, µV | §4.3 |
| $\gamma$ | reach-cutoff fraction of noise | §4.4 |
| $\mathbf c_e, S_e$ | electrode $e$ centre and sub-site set | §5 |
| $\epsilon, \Delta, n_{\text{sub}}$ | electrode edge length, pitch, sub-grid resolution | §5 |
| $W(n,e)$ | per-(neuron, electrode) weight, µV | §5 |
| $x_e[t]$ | synthesised raw channel $e$ | §6.1 |
| $\eta_e[t], \sigma_{\text{pre}}$ | additive noise and its pre-filter SD | §6.1 |
| $H(\omega)$ | band-pass filter frequency response | §6.2 |
| $f_s$ | recording sample rate, Hz | §3.5 |
| $\hat\sigma_e$ | robust per-channel noise estimate | §7.2 |
| $k$ | detection threshold multiplier | §7.3 |
| $\rho$ | lockout window (samples) | §7.4 |
| $\hat t$ | detected event time | §7.4 |
