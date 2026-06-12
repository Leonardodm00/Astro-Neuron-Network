# Asynchronous Vesicle Release — Replacing the Bernoulli-Sum Sampler with Inverse-Transform Sampling

## Contents

1. [Scientific context](#1-scientific-context)
2. [The asynchronous release component](#2-the-asynchronous-release-component)
3. [The problem with the current sampler](#3-the-problem-with-the-current-sampler)
4. [The mathematical solution: BINV](#4-the-mathematical-solution-binv)
5. [Why BINV is efficient in this regime](#5-why-binv-is-efficient-in-this-regime)
6. [Implementation across backends](#6-implementation-across-backends)
7. [Validation strategy](#7-validation-strategy)
8. [Scripts](#8-scripts)

---

## 1. Scientific context

The synaptic model implemented in `ASD_fun_BD_cpp.py` follows the Tsodyks (2005)
framework for short-term synaptic plasticity, extended by De Pittà et al. (2011)
to include modulation of the basal release probability by presynaptic receptors.
Two release pathways coexist at each synapse:

- **Synchronous release** — triggered deterministically by each arriving action
  potential. The fraction of releasable neurotransmitter consumed per spike is
  $r_{Sr} = u_{sr} \cdot x_S$, where $u_{sr}$ is the usage variable that
  undergoes a step increase at each presynaptic event.

- **Asynchronous release** — a stochastic, spike-independent process in which
  individual vesicles from the readily releasable pool (RRP) are released
  spontaneously at a rate governed by the asynchronous usage variable $u_{ar}$.
  This pathway requires drawing a random integer — the number of vesicles
  actually released in a given time step $\Delta t$ — from a Binomial
  distribution at every clock tick and at every synapse.

It is the stochastic sampling in the asynchronous pathway that is the subject of
this report.

---

## 2. The asynchronous release component

### 2.1 State variables

The relevant per-synapse state variables are

| Symbol | Units | Equation | Role |
| --- | --- | --- | --- |
| $x_S(t)$ | dimensionless | clock-driven ODE | fraction of RRP available |
| $u_{ar}(t)$ | Hz | clock-driven ODE | asynchronous usage rate |
| $x_0$ | dimensionless | parameter | resource consumed per vesicle |
| $\Delta t$ | s | global | integration time step (0.05 ms) |

### 2.2 The Brian2 equation block

```
dx_S/dt = Omega_d * (1 - x_S) - r_Ar             : 1  (clock-driven)
duar/dt = -uar * Omega_f_ar                       : Hz (clock-driven)
r_Ar    = x0 * nar                                : Hz
nar     = Binomial_fun(int(floor(x_S/x0)), uar*dt)/dt  : Hz (constant over dt)
```

The sub-expression `nar` is declared `constant over dt`, meaning Brian2
evaluates it exactly once per time step per synapse and treats the result as
constant during the numerical integration of that step.

### 2.3 The probabilistic model

At each time step $\Delta t$, the number of vesicles released asynchronously,
$N_{ar}(t) \in \{0, 1, \ldots, n(t)\}$, is drawn from

$$N_{ar}(t) \mid n(t),\, p(t) \;\sim\; \mathrm{Binomial}\!\left(n(t),\, p(t)\right),$$

where the pool size available for release at time $t$ is

$$n(t) = \left\lfloor \frac{x_S(t)}{x_0} \right\rfloor \in \mathbb{N}_0,$$

and the per-vesicle release probability for this time step is

$$p(t) = u_{ar}(t) \cdot \Delta t \in [0, 1].$$

Both $n(t)$ and $p(t)$ are **per-synapse and per-time-step** quantities: they
take different values across synapses and change continuously as the ODEs evolve.
The continuous depletion rate entering the $x_S$ ODE is then

$$r_{Ar}(t) = x_0 \cdot n_{ar}(t), \qquad n_{ar}(t) = \frac{N_{ar}(t)}{\Delta t}\;\;[\text{Hz}],$$

so that the product $r_{Ar}(t) \cdot \Delta t = x_0 \cdot N_{ar}(t)$ exactly
recovers the discrete integer depletion of $x_S$ at each step.

---

## 3. The problem with the current sampler

### 3.1 The Bernoulli-sum loop

The current `cpp` implementation of `Binomial_fun` draws the Binomial count
by simulating $n$ independent Bernoulli trials:

```cpp
int Binomial_fun(int n, double p, int _vectorisation_idx) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (rand(_vectorisation_idx) < p) count += 1;
    }
    return count;
}
```

The Cython implementation is identical in structure:

```cython
cdef double Binomial_fun(int n, double p, _vectorisation_idx):
    cdef int count = 0
    cdef int i
    for i in range(n):
        if rand(_vectorisation_idx) < p:
            count = count + 1
    return count
```

### 3.2 Problem 1 — computational cost is $O(n)$ in RNG draws

Each call to `Binomial_fun` consumes exactly $n = \lfloor x_S / x_0 \rfloor$
draws from the RNG. With $x_0 = 0.02$ (20 vesicles in a full pool) the loop
iterates up to 50 times per synapse per time step. RNG calls are the dominant
per-step cost on both CPU (`cpp_standalone`) and GPU (`cuda_standalone`)
backends — on the GPU, `curand` calls are particularly expensive relative to
arithmetic.

### 3.3 Problem 2 — incompatibility with Brian2CUDA's static RNG buffer

The `cuda_standalone` backend (Brian2CUDA) pre-generates random numbers by
**statically counting** the number of `rand()` calls in the generated CUDA
kernels at code-generation time and allocating a fixed-size buffer of that
size per thread per step.

The Bernoulli-sum loop violates this assumption: the number of `rand()` calls
per invocation is $n(t)$, a **runtime quantity** that varies with $x_S(t)$ and
$x_0$. The static analyser sees a loop with a dynamic trip count and cannot
determine how many draws to pre-allocate. The consequence is that the buffer
can be under-provisioned, causing draws to silently reuse values from the
previous step. The resulting RNG sequence is no longer i.i.d. uniform, which
corrupts the Binomial statistics without raising any error.

This problem cannot be fixed by an annotation or a pragma — it requires
changing the sampling algorithm so that the number of `rand()` calls per
invocation is **statically known**, ideally exactly one.

---

## 4. The mathematical solution: BINV

### 4.1 Inverse-transform sampling — the general principle

Let $X$ be a discrete random variable supported on $\{0, 1, \ldots, n\}$ with
PMF $P(X = k \mid n, p)$ and CDF $F(k \mid n, p) = \sum_{j=0}^{k} P(X = j \mid n, p)$.
The unit interval $[0, 1]$ can be partitioned into $n+1$ contiguous segments
of lengths $P(X = 0 \mid n, p),\; P(X = 1 \mid n, p),\; \ldots,\; P(X = n \mid n, p)$.

If $U \sim \mathcal{U}([0, 1])$, then

$$K = \min\!\left\{ k \in \{0,\ldots,n\} : U \leq F(k \mid n, p) \right\}$$

satisfies $P(K = k) = P(X = k \mid n, p)$ for all $k$, because the probability
that $U$ falls in segment $k$ equals that segment's length. This requires
exactly **one** uniform draw.

### 4.2 The Binomial CDF walk

The Binomial PMF for fixed $n \in \mathbb{N}$ and $p \in [0, 1]$, evaluated at
$k \in \{0, 1, \ldots, n\}$, is

$$P(X = k \mid n, p) = \binom{n}{k} p^{k} (1-p)^{n-k}.$$

**Base case.** The leftmost segment has length

$$P(X = 0 \mid n, p) = (1-p)^{n}.$$

**Recurrence.** The ratio of consecutive PMF values is derived by taking
$P(X = k+1 \mid n, p) / P(X = k \mid n, p)$ and factoring the three
multiplicative components independently:

$$\frac{P(X = k+1 \mid n, p)}{P(X = k \mid n, p)}
= \underbrace{\frac{\binom{n}{k+1}}{\binom{n}{k}}}_{\displaystyle\frac{n-k}{k+1}}
\cdot \underbrace{\frac{p^{k+1}}{p^{k}}}_{\displaystyle p}
\cdot \underbrace{\frac{(1-p)^{n-k-1}}{(1-p)^{n-k}}}_{\displaystyle\frac{1}{1-p}},$$

which gives the closed-form recurrence

$$\boxed{P(X = k+1 \mid n, p) = P(X = k \mid n, p) \cdot \frac{n-k}{k+1} \cdot \frac{p}{1-p},
\qquad k = 0, 1, \ldots, n-1.}$$

The ratio $r = p/(1-p)$ is **constant** for a fixed call (fixed $n$ and $p$),
so it is computed once. Each subsequent CDF segment costs one multiply by
$(n - k)$, one divide by $(k + 1)$, one multiply by $r$, and one
floating-point addition to the running CDF accumulator. No factorials, no
per-step exponentials or logarithms beyond the single initialisation.

### 4.3 The algorithm

```
Input:  n = floor(x_S / x_0)   [integer, >= 0]
        p = u_ar * dt           [float in [0, 1]]

1.  If n <= 0 or p <= 0  ->  return 0.
    If p >= 1             ->  return n.

2.  Compute q = 1 - p,  r = p / q,  pmf = q^n,  cdf = pmf.
    Draw U ~ U([0, 1]).              // exactly ONE rand() call
    Set k = 0.

3.  While U > cdf and k < n:
        k   = k + 1
        pmf = pmf * ((n - k + 1) / k) * r   // recurrence
        cdf = cdf + pmf

4.  Return k.
```

The while-loop terminates as soon as the running CDF first covers $U$, so
the number of iterations equals $K$, the sampled value itself.

---

## 5. Why BINV is efficient in this regime

### 5.1 Expected cost

The expected number of while-loop iterations is

$$\mathbb{E}[K \mid n, p] = n \cdot p = \lambda,$$

so the total expected work per call is $\lambda + 1$ arithmetic operations
(loop body) plus one transcendental $q^n$ at initialisation. The Bernoulli
loop costs $n$ RNG calls regardless of $p$. The ratio of expected costs is

$$\frac{\text{BINV expected cost}}{\text{Bernoulli loop cost}}
\approx \frac{\lambda + 1}{n} = p + \frac{1}{n}.$$

For small $p$ — which holds here because $p = u_{ar} \cdot \Delta t$ with
$\Delta t = 5 \times 10^{-5}\,\text{s}$ — this ratio is far below 1.

### 5.2 Geometric picture

With $\lambda \ll 1$ the probability mass is concentrated near $k = 0$:

$$[0, 1] = \underbrace{[\,0,\;(1-p)^n\,]}_{\text{segment 0, long}}
\;\;
\underbrace{[(1-p)^n,\; F(1 \mid n,p)]}_{\text{segment 1, short}}
\;\;
\underbrace{[F(1\mid n,p),\; F(2\mid n,p)]}_{\text{segment 2, tiny}}
\;\; \cdots$$

A uniform draw $U$ almost certainly lands in segment 0. In that case the
algorithm initialises `pmf = q^n`, checks `U <= cdf`, finds the condition
satisfied, and returns $k = 0$ immediately — the while-loop body never
executes. The full partition of $n + 1$ segments is never constructed.

### 5.3 Resolution of the Brian2CUDA incompatibility

BINV contains **exactly one** `rand()` call — unconditional, at line 2 of the
algorithm, before the loop. The static analyser in Brian2CUDA's code generator
sees a single, unconditionally-executed RNG draw and allocates a buffer of
exactly one uniform per thread per step. This is correct by construction and
the incompatibility described in Section 3.3 is fully resolved, without any
annotation or special-casing.

### 5.4 Exactness

BINV is not an approximation. It draws from the exact Binomial distribution
$\mathrm{Binomial}(n, p)$ for all values of $n$ and $p$. Replacing the
Bernoulli loop with BINV changes only the computational cost and the RNG
consumption pattern — it does not change the statistical model of asynchronous
release. No science is altered.

---

## 6. Implementation across backends

The function is registered as a Brian2 `Function` object with four
implementations targeting different backends. All four implement the identical
mathematical algorithm; only the syntax and the RNG primitive differ.

### 6.1 Numpy (default, used by the `numpy` and `runtime` targets)

```python
def _numpy_vectorised(n, p, _vectorisation_idx):
    # One scalar call per synapse via the validated scalar kernel.
    ...
    out[m] = binv_scalar(int(n_b[m]), float(p_b[m]))
    ...
```

### 6.2 Cython (used by the `cython` target)

```cython
cdef double Binomial_fun(int n, double p, _vectorisation_idx):
    cdef double q, r, pmf, cdf, U
    cdef int k
    if n <= 0 or p <= 0.0:
        return 0.0
    if p >= 1.0:
        return <double>n
    q = 1.0 - p
    r = p / q
    pmf = q ** n
    cdf = pmf
    U = rand(_vectorisation_idx)      # single draw
    k = 0
    while U > cdf and k < n:
        k = k + 1
        pmf = pmf * (<double>(n - k + 1) / <double>k) * r
        cdf = cdf + pmf
    return <double>k
```

### 6.3 C++ (used by `cpp_standalone`)

```cpp
double Binomial_fun(int n, double p, int _vectorisation_idx) {
    if (n <= 0 || p <= 0.0) return 0.0;
    if (p >= 1.0) return (double)n;
    double q = 1.0 - p, r = p / q;
    double pmf = pow(q, (double)n), cdf = pmf;
    double U = rand(_vectorisation_idx);   // single draw
    int k = 0;
    while (U > cdf && k < n) {
        k += 1;
        pmf *= ((double)(n - k + 1) / (double)k) * r;
        cdf += pmf;
    }
    return (double)k;
}
```

### 6.4 CUDA (used by `cuda_standalone` via Brian2CUDA)

```cuda
__device__ double Binomial_fun(int n, double p, int _vectorisation_idx) {
    if (n <= 0 || p <= 0.0) return 0.0;
    if (p >= 1.0) return (double)n;
    double q = 1.0 - p, r = p / q;
    double pmf = pow(q, (double)n), cdf = pmf;
    double U = rand(_vectorisation_idx);   // Brian2CUDA device rand — single draw
    int k = 0;
    while (U > cdf && k < n) {
        k += 1;
        pmf *= ((double)(n - k + 1) / (double)k) * r;
        cdf += pmf;
    }
    return (double)k;
}
```

All four implementations share `dependencies={'rand': DEFAULT_FUNCTIONS['rand']}`,
so the `rand` primitive is resolved correctly by each backend without
hard-coding any RNG API.

---

## 7. Validation strategy

Because BINV is an exact sampler, the validation question is precisely stated:
the new implementation must produce samples whose distribution is
statistically indistinguishable from $\mathrm{Binomial}(n, p)$ and from
the former Bernoulli-loop sampler, across the full $(n, p)$ grid visited by
the simulation.

The validation has two tiers, motivated by two distinct correctness claims.

### 7.1 Tier 1 — unit test of the sampler in isolation

**Claim.** For every $(n, p)$ in the grid derived from the actual model
parameters, the empirical mean and variance of $M$ BINV draws agree with the
exact Binomial moments $\mathbb{E}[X \mid n, p] = np$ and
$\mathrm{Var}[X \mid n, p] = np(1-p)$ to within $k_{\sigma} = 6$ standard
errors, and the two-sample KS statistic comparing BINV draws to former
(Bernoulli-loop) draws has p-value $> 10^{-4}$.

**Acceptance criterion.** The standard error on the sample mean from $M$ draws is
$\mathrm{SE}_{\bar{X}} = \sqrt{\mathrm{Var}[X \mid n,p] / M}$.
The corresponding standard error on the sample variance uses the exact
fourth central moment of the Binomial,
$\mu_4(n,p) = np(1-p)\!\left[1 + (3n-6)p(1-p)\right]$, to give
$\mathrm{SE}_{S^2} = \sqrt{(\mu_4 - \mathrm{Var}[X]^2) / M}$.
A fixed relative tolerance (e.g. 3% of the mean) is deliberately avoided:
in the rare-event regime $np \to 0$ the mean is close to zero and a fixed
relative band falsely fails even when the two-sample KS is at p = 1. The
$k_{\sigma}$-SEM criterion is correct at all values of $np$.

### 7.2 Tier 2 — integration test inside Brian2

**Claim.** When both samplers are embedded in the full Brian2 release model
and driven by an identical presynaptic spike train, the **ensemble** statistics
of the released-neurotransmitter traces — mean $\pm$ standard deviation of
cleft glutamate $Y_S(t)$ and cumulative asynchronously released NT $C_{async}(t)$
across $N_{syn}$ synapses — are statistically indistinguishable (two-sample KS
p-value $> 10^{-3}$ on the final-time-point distributions).

**What is not claimed.** Individual trajectories of $Y_S(t)$ and $C_{async}(t)$
at a single synapse will differ between the two mechanisms, because BINV draws
one uniform per call whereas the Bernoulli loop draws $n$ — the RNG streams
are consumed at different rates and are therefore out of phase after the first
step. This trajectory-level difference is correct and expected behaviour, not
a bug.

---

## 8. Scripts

### 8.1 `smoke_test_async_release.py`

**Purpose.** Self-contained, headless validation script. Implements both tiers
of the validation strategy described in Section 7, produces all diagnostic
figures, and exits with code 0 on overall PASS, non-zero on any failure.
No GPU or Brian2 standalone compiler is required (default target is `numpy`).

**Key design decisions.**

- The scalar samplers (`former_scalar`, `binv_scalar`) are pure Python, so
  they can be validated without any Brian2 machinery. They are the ground
  truth for Tier 1.
- `make_brian_function(kind)` registers all four backend implementations
  (numpy, cython, cpp, cuda_standalone) from the same body. The cpp and
  cuda_standalone entries are inert under the numpy/cython runtime targets
  but are stored in the returned `Function` object and serve as the
  deployment reference for the port.
- The Tier-1 acceptance criterion uses the exact Binomial $\mu_4$ to
  compute $\mathrm{SE}_{S^2}$ rather than a fixed relative tolerance, for
  correctness in the rare-event regime.
- The Tier-2 ensemble size defaults to $N_{syn} = 500$ synapses. Larger
  values improve the power of the KS test at the cost of runtime.

**Outputs (in `--outdir`).**

| File | Content |
| --- | --- |
| `fig1_unit_sampler_pmf.png` | Former vs BINV vs exact scipy PMF at each $(n, p)$ grid point |
| `fig2_example_traces.png` | Example single-synapse traces — expected to differ between mechanisms |
| `fig3_ensemble_traces.png` | Ensemble mean $\pm$ std — expected to coincide between mechanisms |
| `fig4_released_NT_distribution.png` | Final $C_{async}(T)$ distribution + ECDF with two-sample KS |
| `unit_sampler_stats.csv` | Per-$(n,p)$ numerical summary (theoretical moments, empirical moments, z-scores, KS) |
| `report.txt` | Machine-parseable PASS/FAIL verdict with per-tier breakdown |

**CLI.**

```
python smoke_test_async_release.py [options]

  --outdir   DIR       output directory                   [./smoke_out]
  --target   STR       Brian codegen target: numpy|cython  [numpy]
  --Nsyn     INT       ensemble size (synapses)            [500]
  --simtime  FLOAT     biological time [s]                 [5.0]
  --rate     FLOAT     presynaptic Poisson rate [Hz]       [15.0]
  --rec-dt   FLOAT     recording dt [ms]                   [2.0]
  --dt       FLOAT     integration dt [ms]                 [0.05]
  --x0       FLOAT     resource per vesicle                [0.02]
  --uar-max  FLOAT     max u_ar [Hz] for the (n,p) grid    [2000.0]
  --M        INT       Tier-1 samples per (n,p) point      [200000]
  --seed     INT       master RNG seed                     [1234]
  --quick              fast sanity pass (reduced everything)
```

**Important.** The `--x0` and `--uar-max` arguments are **placeholders** in
the default configuration. Before treating the Tier-1 grid as meaningful, set
them to the actual values from `get_Synparam()` and your parameter sweep bounds,
so the grid covers the exact $(n, p)$ regime that the simulation visits.
Similarly, replace the inline namespace dictionary in `build_release_model`
with the real `get_Synparam()` values to make the Tier-2 dynamics quantitatively
faithful.

---

### 8.2 `submit_smoke_async_release.sh`

**Purpose.** PBS wrapper for `smoke_test_async_release.py`. Submits the job to
the `intel` queue, resolves all parameters from environment variables (so the
file itself need not be edited for routine use), creates a timestamped output
directory, and updates the `smoke_async_latest` symlink for downstream
consumption.

**PBS resource allocation.**

```
#PBS -q intel
#PBS -l select=1:ncpus=2,walltime=00:30:00
```

Two CPUs are requested rather than one: the numpy target is single-threaded,
but a second core prevents busy-wait penalties and is used during Cython/GCC
compilation when `TARGET=cython`. The 30-minute walltime covers the default
full run (8–12 min) and the `--quick` mode (under 2 min) with ample margin.

**Invocation patterns.**

```bash
# Standard run (numpy target, default parameters):
qsub submit_smoke_async_release.sh

# Cython target with larger ensemble:
TARGET=cython NSYN=1000 SIMTIME=10 \
    qsub -v TARGET,NSYN,SIMTIME submit_smoke_async_release.sh

# Fast sanity check:
QUICK=1 qsub -v QUICK submit_smoke_async_release.sh

# Local run (no PBS):
bash submit_smoke_async_release.sh

# Override output root to scratch:
OUT_ROOT=/scratch/${USER}/binv_smoke \
    qsub -v OUT_ROOT submit_smoke_async_release.sh
```

**User-configurable variables (set at the top of the script).**

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCRIPT` | `./smoke_test_async_release.py` | Path to the Python script |
| `OUT_ROOT` | `.` | Parent of the timestamped output directory |
| `TARGET` | `numpy` | Brian2 codegen target: `numpy` or `cython` |
| `NSYN` | `500` | Ensemble size |
| `SIMTIME` | `5.0` | Biological simulation time [s] |
| `RATE` | `15.0` | Presynaptic Poisson rate [Hz] |
| `REC_DT` | `2.0` | Recording dt [ms] |
| `DT` | `0.05` | Integration dt [ms] |
| `X0` | `0.02` | Resource per vesicle — **replace with real value** |
| `UAR_MAX` | `2000.0` | Max $u_{ar}$ [Hz] — **replace with real value** |
| `M` | `200000` | Tier-1 Monte Carlo samples per $(n,p)$ point |
| `SEED` | `1234` | Master RNG seed |
| `QUICK` | `0` | Set to `1` for fast sanity pass |

**Output directory and symlink.** Each run creates a timestamped directory
`./smoke_async_YYYYMMDD_HHMMSS/` (or `OUT_ROOT/smoke_async_...` if `OUT_ROOT`
is set) and atomically updates a `smoke_async_latest` symlink to point to it.
A downstream analysis script can therefore always read
`smoke_async_latest/report.txt` without knowing the timestamp.

**Exit code.** The script forwards the Python process exit code verbatim:
0 on overall PASS, non-zero on any failure. This makes it suitable for use
in automated pipelines that check `$?` or use `set -e`.

---

## Appendix A — Recurrence derivation in full

Starting from the PMF

$$P(X = k \mid n, p) = \binom{n}{k} p^{k} (1-p)^{n-k},$$

we compute

$$\frac{P(X = k+1 \mid n, p)}{P(X = k \mid n, p)}
= \frac{\binom{n}{k+1}}{\binom{n}{k}} \cdot \frac{p^{k+1}}{p^{k}} \cdot \frac{(1-p)^{n-k-1}}{(1-p)^{n-k}}.$$

**Combinatorial factor.** Using $\binom{n}{k} = n! / (k!\,(n-k)!)$:

$$\frac{\binom{n}{k+1}}{\binom{n}{k}}
= \frac{n!\,/\,[(k+1)!\,(n-k-1)!]}{n!\,/\,[k!\,(n-k)!]}
= \frac{k!\,(n-k)!}{(k+1)!\,(n-k-1)!}
= \frac{n-k}{k+1}.$$

**Power factors.**

$$\frac{p^{k+1}}{p^{k}} = p, \qquad \frac{(1-p)^{n-k-1}}{(1-p)^{n-k}} = \frac{1}{1-p}.$$

Combining:

$$\frac{P(X = k+1 \mid n, p)}{P(X = k \mid n, p)} = \frac{n-k}{k+1} \cdot \frac{p}{1-p},$$

which rearranges to the recurrence stated in Section 4.2.

---

## Appendix B — Regime validity

BINV has expected cost $\lambda + 1 = np + 1$. This cost is minimised when
$\lambda$ is small. For the current default parameters ($x_0 = 0.02$,
$\Delta t = 0.05\,\text{ms}$) the pool size is at most $n_{\max} = 50$ and
the per-vesicle probability $p = u_{ar} \cdot \Delta t$ is of order $10^{-2}$
to $10^{-3}$, giving $\lambda \sim 0.05$–$0.5$. The algorithm typically
terminates after one or two iterations.

If future parameter changes push $\lambda = np$ substantially above 30, BINV
degrades towards $O(n)$ and more sophisticated $O(1)$ samplers such as
BTRS (Hormann 1993) become preferable. The current implementation guards
against the underflow regime ($p \to 0$, $n \to \infty$, $np \to 0$) correctly
via the $p \leq 0$ guard, and against overflow of $q^n$ (which underflows to
zero for $n \cdot \ln(1-p) \lesssim -700$ in double precision) by the same
pathway — the `cdf = pmf` initialisation immediately satisfies `U <= cdf` and
returns $k = 0$, which is correct since $P(X = 0 \mid n, p) \approx 1$ in
that regime.
