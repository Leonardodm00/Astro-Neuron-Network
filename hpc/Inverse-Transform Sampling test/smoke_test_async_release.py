#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
smoke_test_async_release.py
===========================

Compare the TWO sampling mechanisms for the *asynchronous* vesicle-release count

        n_release ~ Binomial(n, p),   n = floor(x_S / x0),   p = u_ar * dt

inside the synaptic model, with an eye to replacing the current sampler.

    FORMER   Bernoulli-sum loop      : O(n) RNG draws / call   (the current code)
    BINV     inverse-transform (CDF) : 1  RNG draw  / call + O(n*p) cheap arithmetic

Both samplers draw from the *identical* distribution Binomial(n, p). BINV is NOT an
approximation. The scientifically correct expectation is therefore:

  (1) UNIT level  : both match scipy.stats.binom (mean, variance, full PMF) exactly,
                    to within Monte-Carlo error.
  (2) TRACE level : single trajectories DIFFER between mechanisms, because the two
                    consume the RNG stream differently (n draws vs 1 draw per call).
                    The ENSEMBLE statistics (mean +/- std of the released-NT traces,
                    and the distribution of the cumulative released NT) COINCIDE.

This script verifies both, and saves qualitative plots. It runs on a CPU node
(no GPU needed): the comparison is about the sampling algorithm, which is
device-independent. The 'cython'/'cpp'/'cuda_standalone' implementations of the
new sampler are registered here too, so this file doubles as the deployment
reference for the cpp_standalone / cuda_standalone port.

------------------------------------------------------------------------------
USAGE (HPC, headless)
------------------------------------------------------------------------------
    python smoke_test_async_release.py --outdir ./smoke_out
    python smoke_test_async_release.py --outdir ./smoke_out --target cython --Nsyn 1000
    python smoke_test_async_release.py --quick          # fast sanity pass

Outputs (in --outdir):
    fig1_unit_sampler_pmf.png        former vs BINV vs scipy PMF, per (n,p)
    fig2_example_traces.png          example single-synapse traces (differ: expected)
    fig3_ensemble_traces.png         ensemble mean +/- std (coincide: expected)
    fig4_released_NT_distribution.png distribution of final cumulative released NT
    unit_sampler_stats.csv           per-(n,p) numerical comparison
    report.txt                       PASS/FAIL summary with tolerances

------------------------------------------------------------------------------
NOTE ON PARAMETERS
------------------------------------------------------------------------------
The (n, p) grid and the synapse namespace below use PLACEHOLDER values spanning
the plausible regime. Replace them with the real values your sweep visits:
  - n is set by x0  (n_max = floor(1/x0));      pass via --x0
  - p = u_ar * dt is set by u_ar and dt;        pass --uar-max (Hz) and --dt (ms)
The defaults assume the rare-event regime (small p) that BINV targets.
"""

import os
import sys
import csv
import argparse

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")               # headless: no display on HPC
import matplotlib.pyplot as plt


# ==========================================================================
# SECTION 1 — Scalar samplers (the ground-truth logic, transliterated below)
# ==========================================================================
def former_scalar(n, p):
    """Bernoulli-sum loop. O(n) uniform draws. Mirrors the current cpp/cython code."""
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    return int(np.sum(np.random.rand(n) < p))


def binv_scalar(n, p):
    """Inverse-transform (BINV). Exactly ONE uniform draw + O(n*p) CDF walk.

    P(X=0) = (1-p)^n ;  P(X=k+1) = P(X=k) * (n-k)/(k+1) * p/(1-p).
    Returns the smallest k with U <= F(k).
    """
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    q = 1.0 - p
    r = p / q
    pmf = q ** n                    # P(X=0)
    cdf = pmf
    U = np.random.rand()            # the single draw
    k = 0
    while U > cdf and k < n:
        k += 1
        pmf *= ((n - k + 1) / k) * r
        cdf += pmf
    return k


# ==========================================================================
# SECTION 2 — Brian2 Function factory
#   Registers numpy (validated, default), cython (validated), and cpp/cuda
#   (deployment) implementations. cpp/cuda are inert under runtime targets but
#   are the exact code the cpp_standalone / cuda_standalone port will use.
# ==========================================================================
def make_brian_function(kind):
    """kind in {'former', 'binv'} -> a brian2.Function with all implementations."""
    from brian2 import Function, DEFAULT_FUNCTIONS

    scalar = former_scalar if kind == "former" else binv_scalar

    def _numpy_vectorised(n, p, _vectorisation_idx):
        # auto_vectorise passes _vectorisation_idx (array of element indices, or an
        # int count). One independent draw per element, reusing the scalar logic.
        if np.ndim(_vectorisation_idx) == 0:
            length = int(_vectorisation_idx)
        else:
            length = len(_vectorisation_idx)
        n_b = np.broadcast_to(np.atleast_1d(n).astype(float), (length,))
        p_b = np.broadcast_to(np.atleast_1d(p).astype(float), (length,))
        out = np.empty(length, dtype=np.float64)
        for m in range(length):
            out[m] = scalar(int(n_b[m]), float(p_b[m]))
        return out

    fun = Function(_numpy_vectorised, arg_units=[1, 1], return_unit=1,
                   stateless=False, auto_vectorise=True)

    if kind == "former":
        cython_code = r"""
cdef double Binomial_fun(int n, double p, _vectorisation_idx):
    cdef int count = 0
    cdef int i
    for i in range(n):
        if rand(_vectorisation_idx) < p:
            count = count + 1
    return count
"""
        cpp_code = r"""
double Binomial_fun(int n, double p, int _vectorisation_idx) {
    int count = 0;
    for (int i = 0; i < n; ++i)
        if (rand(_vectorisation_idx) < p) count += 1;
    return (double)count;
}
"""
        cuda_code = r"""
__device__ double Binomial_fun(int n, double p, int _vectorisation_idx) {
    int count = 0;
    for (int i = 0; i < n; ++i)
        if (rand(_vectorisation_idx) < p) count += 1;
    return (double)count;
}
"""
    else:  # binv
        cython_code = r"""
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
    U = rand(_vectorisation_idx)
    k = 0
    while U > cdf and k < n:
        k = k + 1
        pmf = pmf * (<double>(n - k + 1) / <double>k) * r
        cdf = cdf + pmf
    return <double>k
"""
        cpp_code = r"""
double Binomial_fun(int n, double p, int _vectorisation_idx) {
    if (n <= 0 || p <= 0.0) return 0.0;
    if (p >= 1.0) return (double)n;
    double q = 1.0 - p, r = p / q, pmf = pow(q, (double)n), cdf = pmf;
    double U = rand(_vectorisation_idx);          // single draw -> static rand count
    int k = 0;
    while (U > cdf && k < n) {
        k += 1;
        pmf *= ((double)(n - k + 1) / (double)k) * r;
        cdf += pmf;
    }
    return (double)k;
}
"""
        cuda_code = r"""
__device__ double Binomial_fun(int n, double p, int _vectorisation_idx) {
    if (n <= 0 || p <= 0.0) return 0.0;
    if (p >= 1.0) return (double)n;
    double q = 1.0 - p, r = p / q, pmf = pow(q, (double)n), cdf = pmf;
    double U = rand(_vectorisation_idx);          // Brian2CUDA device rand
    int k = 0;
    while (U > cdf && k < n) {
        k += 1;
        pmf *= ((double)(n - k + 1) / (double)k) * r;
        cdf += pmf;
    }
    return (double)k;
}
"""

    dep = {"rand": DEFAULT_FUNCTIONS["rand"]}
    fun.implementations.add_implementation("cython", cython_code, dependencies=dep)
    fun.implementations.add_implementation("cpp", cpp_code, dependencies=dep)
    # 'cuda_standalone' key is stored harmlessly even without brian2cuda imported.
    fun.implementations.add_implementation("cuda_standalone", cuda_code, dependencies=dep)
    return fun


# ==========================================================================
# SECTION 3 — TIER 1: unit test of the sampler vs scipy.stats.binom
# ==========================================================================
def tier1_unit_test(grid, M, seed, outdir):
    """Draw M samples from each sampler at each (n,p); compare to exact binomial."""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    rows, n_panels = [], len(grid)
    ncol = min(3, n_panels)
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)

    all_ok = True
    for ax_idx, (n, p) in enumerate(grid):
        f = np.fromiter((former_scalar(n, p) for _ in range(M)), dtype=int, count=M)
        b = np.fromiter((binv_scalar(n, p) for _ in range(M)), dtype=int, count=M)
        q = 1.0 - p
        mean_th, var_th = n * p, n * p * q
        mu4 = n * p * q * (1.0 + (3.0 * n - 6.0) * p * q)   # 4th central moment, binomial
        ks, pval = stats.ks_2samp(f, b)          # former vs binv: same dist => large p

        # Principled acceptance: empirical mean/variance must lie within k*SE of the
        # EXACT binomial moments (SE shrinks as 1/sqrt(M)); a fixed relative tolerance
        # false-fails in the rare-event limit (np->0) where MC noise dominates the mean.
        KSIG = 6.0
        se_mean = np.sqrt(var_th / M)
        se_var = np.sqrt(max(mu4 - var_th ** 2, 0.0) / M)
        z_mean = (b.mean() - mean_th) / se_mean if se_mean > 0 else 0.0
        z_var = (b.var() - var_th) / se_var if se_var > 0 else 0.0
        ok = (abs(z_mean) <= KSIG and abs(z_var) <= KSIG and pval > 1e-4)
        all_ok &= ok
        rows.append(dict(n=n, p=p, mean_th=mean_th, mean_former=f.mean(),
                         mean_binv=b.mean(), var_th=var_th, var_former=f.var(),
                         var_binv=b.var(), z_mean=z_mean, z_var=z_var,
                         ks_former_binv=ks, p_former_binv=pval, passed=ok))

        ax = axes.flat[ax_idx]
        kmax = int(max(f.max(), b.max(), np.ceil(mean_th + 4 * np.sqrt(var_th + 1e-9)))) + 1
        bins = np.arange(-0.5, kmax + 0.5, 1.0)
        ax.hist(f, bins=bins, density=True, alpha=0.45, label="former", color="#1f77b4")
        ax.hist(b, bins=bins, density=True, alpha=0.45, label="BINV", color="#d62728")
        kk = np.arange(0, kmax + 1)
        ax.plot(kk, stats.binom.pmf(kk, n, p), "k.-", lw=1, ms=4, label="scipy")
        ax.set_title(f"n={n}, p={p:g}  (np={mean_th:g})", fontsize=9)
        ax.set_xlabel("count k"); ax.set_ylabel("P(k)")
        ax.legend(fontsize=7)
    for j in range(n_panels, nrow * ncol):
        axes.flat[j].axis("off")
    fig.suptitle("TIER 1 — asynchronous-release count: former vs BINV vs exact binomial",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, "fig1_unit_sampler_pmf.png"), dpi=140)
    plt.close(fig)

    with open(os.path.join(outdir, "unit_sampler_stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    np.random.set_state(rng_state)
    return all_ok, rows


# ==========================================================================
# SECTION 4 — Minimal Brian2 release model (async binomial + cleft dynamics)
# ==========================================================================
def build_release_model(fun, namespace):
    """Single presyn source -> Nsyn independent synapses. Returns the eqs/pre strings.

    SIMPLIFICATION (flagged): the cleft variables Y_S, Y_A and the cumulative
    released NT C_async are kept dimensionless (':1'), with the spillover gain
    rho*Y_T folded into a single dimensionless g_rel, so the smoke test avoids
    mole-unit bookkeeping. This is adequate for a *qualitative* mechanism
    comparison; it does not change the stochastic structure under test.
    """
    eqs = """
    dx_S/dt = Omega_d*(1 - x_S) - r_Ar                  : 1  (clock-driven)
    dusr/dt = -Omega_f_sr*usr                           : 1  (clock-driven)
    r_Ar    = x0*nar                                    : Hz
    nar     = Binomial_fun(int(floor(x_S/x0)), uar*dt)/dt : Hz (constant over dt)
    duar/dt = -uar*Omega_f_ar                           : Hz (clock-driven)
    dY_S/dt = -Omega_c*Y_S + g_rel*r_Ar                 : 1  (clock-driven)
    dC_async/dt = g_rel*r_Ar                            : 1  (clock-driven)  # async-only cumulative
    r_Sr                                                : 1
    """
    pre = """
    usr  += U_0_sr*(1 - usr)
    r_Sr  = usr*x_S
    x_S  -= r_Sr
    uar  += U_0_ar*(Umax - uar)
    Y_S  += g_rel*r_Sr
    """
    return eqs, pre


def run_ensemble(kind, namespace, spike_times, Nsyn, simtime, rec_dt, target, seed):
    """Run one mechanism; return dict of recorded ensemble traces (syn x time)."""
    from brian2 import (NeuronGroup, Synapses, SpikeGeneratorGroup, StateMonitor,
                        Network, defaultclock, prefs, start_scope, second, ms)
    from brian2 import seed as brian_seed

    prefs.codegen.target = target
    start_scope()
    defaultclock.dt = namespace["__dt__"]

    fun = make_brian_function(kind)
    eqs, pre = build_release_model(fun, namespace)

    P = SpikeGeneratorGroup(1, np.zeros(len(spike_times), dtype=int),
                            spike_times * second)
    Q = NeuronGroup(Nsyn, "v : 1")
    ns = {k: v for k, v in namespace.items() if not k.startswith("__")}
    ns["Binomial_fun"] = fun
    S = Synapses(P, Q, model=eqs, on_pre=pre, namespace=ns,
                 method="exponential_euler")
    S.connect(i=0, j=np.arange(Nsyn))
    S.x_S = 1.0

    mon = StateMonitor(S, ["Y_S", "C_async", "nar"], record=True, dt=rec_dt)
    net = Network(P, Q, S, mon)

    brian_seed(seed)                 # reproducible per mechanism (streams still differ)
    net.run(simtime)

    return dict(
        t=np.asarray(mon.t),
        Y_S=np.asarray(mon.Y_S),         # (Nsyn, T)
        C_async=np.asarray(mon.C_async), # (Nsyn, T)
        nar=np.asarray(mon.nar),
    )


# ==========================================================================
# SECTION 5 — TIER 2: trace/ensemble analysis + plots
# ==========================================================================
def _band(ax, t, mat, color, label):
    m = mat.mean(axis=0)
    s = mat.std(axis=0)
    ax.plot(t, m, color=color, lw=1.6, label=f"{label} mean")
    ax.fill_between(t, m - s, m + s, color=color, alpha=0.20, lw=0,
                    label=f"{label} ±1 std")


def tier2_plots_and_stats(resF, resB, outdir, n_example=5):
    t = resF["t"]
    blue, red = "#1f77b4", "#d62728"

    # --- fig2: example single-synapse traces (EXPECTED to differ) ---
    fig, ax = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for s in range(min(n_example, resF["Y_S"].shape[0])):
        ax[0, 0].plot(t, resF["Y_S"][s], lw=0.8, alpha=0.8)
        ax[0, 1].plot(t, resB["Y_S"][s], lw=0.8, alpha=0.8)
        ax[1, 0].plot(t, resF["C_async"][s], lw=0.8, alpha=0.8)
        ax[1, 1].plot(t, resB["C_async"][s], lw=0.8, alpha=0.8)
    ax[0, 0].set_title("FORMER — cleft Y_S (example synapses)")
    ax[0, 1].set_title("BINV — cleft Y_S (example synapses)")
    ax[1, 0].set_title("FORMER — cumulative async released NT")
    ax[1, 1].set_title("BINV — cumulative async released NT")
    ax[1, 0].set_xlabel("t [s]"); ax[1, 1].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("Y_S [a.u.]"); ax[1, 0].set_ylabel("C_async [a.u.]")
    fig.suptitle("TIER 2 — single trajectories DIFFER between mechanisms (expected: "
                 "different RNG consumption)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(outdir, "fig2_example_traces.png"), dpi=140)
    plt.close(fig)

    # --- fig3: ensemble mean +/- std (EXPECTED to coincide) ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    _band(ax[0], t, resF["Y_S"], blue, "former")
    _band(ax[0], t, resB["Y_S"], red, "BINV")
    ax[0].set_title("Cleft Y_S — ensemble mean ±1 std"); ax[0].set_ylabel("Y_S [a.u.]")
    _band(ax[1], t, resF["C_async"], blue, "former")
    _band(ax[1], t, resB["C_async"], red, "BINV")
    ax[1].set_title("Cumulative async released NT — ensemble mean ±1 std")
    ax[1].set_ylabel("C_async [a.u.]")
    for a in ax:
        a.set_xlabel("t [s]"); a.legend(fontsize=8)
    fig.suptitle("TIER 2 — ensemble statistics COINCIDE (same distribution sampled)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "fig3_ensemble_traces.png"), dpi=140)
    plt.close(fig)

    # --- fig4: distribution of final cumulative released NT ---
    cF, cB = resF["C_async"][:, -1], resB["C_async"][:, -1]
    ks_c, p_c = stats.ks_2samp(cF, cB)
    yF, yB = resF["Y_S"][:, -1], resB["Y_S"][:, -1]
    ks_y, p_y = stats.ks_2samp(yF, yB)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    bins = np.histogram_bin_edges(np.concatenate([cF, cB]), bins=30)
    ax[0].hist(cF, bins=bins, density=True, alpha=0.45, color=blue, label="former")
    ax[0].hist(cB, bins=bins, density=True, alpha=0.45, color=red, label="BINV")
    ax[0].set_title(f"Final C_async distribution\nKS={ks_c:.3f}, p={p_c:.3f}")
    ax[0].set_xlabel("C_async(T) [a.u.]"); ax[0].set_ylabel("density"); ax[0].legend()
    xs = np.sort(np.concatenate([cF, cB]))
    ax[1].plot(np.sort(cF), np.linspace(0, 1, len(cF)), color=blue, label="former")
    ax[1].plot(np.sort(cB), np.linspace(0, 1, len(cB)), color=red, label="BINV")
    ax[1].set_title("Final C_async — ECDF"); ax[1].set_xlabel("C_async(T) [a.u.]")
    ax[1].set_ylabel("F(x)"); ax[1].legend()
    fig.suptitle("TIER 2 — pooled released-NT distributions agree (two-sample KS)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, "fig4_released_NT_distribution.png"), dpi=140)
    plt.close(fig)

    return dict(ks_Cfinal=ks_c, p_Cfinal=p_c, ks_Yfinal=ks_y, p_Yfinal=p_y,
                mean_Cfinal_former=float(cF.mean()), mean_Cfinal_binv=float(cB.mean()),
                std_Cfinal_former=float(cF.std()), std_Cfinal_binv=float(cB.std()))


# ==========================================================================
# SECTION 6 — orchestration / report
# ==========================================================================
def realistic_grid(x0, uar_max_hz, dt_ms):
    """Build an (n,p) grid from the actual model knobs (call with your swept ranges)."""
    n_max = int(np.floor(1.0 / x0))
    p_max = uar_max_hz * (dt_ms * 1e-3)
    ns = sorted(set(int(round(v)) for v in
                    [max(1, n_max // 20), max(2, n_max // 5), max(2, n_max // 2), n_max]))
    ps = [p_max / 50.0, p_max / 10.0, p_max / 2.0, p_max]
    return [(n, float(p)) for n in ns for p in ps if 0 < p < 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="./smoke_out")
    ap.add_argument("--target", default="numpy", choices=["numpy", "cython"],
                    help="Brian runtime codegen target for the integration test.")
    ap.add_argument("--Nsyn", type=int, default=500, help="ensemble size (synapses).")
    ap.add_argument("--simtime", type=float, default=5.0, help="biological seconds.")
    ap.add_argument("--rate", type=float, default=15.0, help="presynaptic rate [Hz].")
    ap.add_argument("--rec-dt", type=float, default=2.0, help="recording dt [ms].")
    ap.add_argument("--dt", type=float, default=0.05, help="integration dt [ms].")
    ap.add_argument("--x0", type=float, default=0.02, help="resource per vesicle (sets n).")
    ap.add_argument("--uar-max", type=float, default=2000.0,
                    help="max u_ar [Hz] (sets p=uar*dt) for the unit-test grid.")
    ap.add_argument("--M", type=int, default=200000, help="samples per (n,p) in Tier 1.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--quick", action="store_true", help="fast pass (smaller everything).")
    args = ap.parse_args()

    if args.quick:
        args.Nsyn, args.simtime, args.M = 150, 1.5, 40000

    os.makedirs(args.outdir, exist_ok=True)
    from brian2 import ms, Hz, second, BrianLogger
    BrianLogger.suppress_hierarchy("brian2")

    # ---- model namespace (PLACEHOLDERS — replace with get_Synparam() values) ----
    ns = dict(
        Omega_d=2.0 * Hz, Omega_f_sr=2.0 * Hz, Omega_f_ar=2.0 * Hz,
        U_0_sr=0.5, U_0_ar=0.3, Umax=300.0 * Hz, x0=args.x0,
        g_rel=1.0, Omega_c=200.0 * Hz,
        __dt__=args.dt * ms,
    )

    # ---- TIER 1: unit sampler test --------------------------------------------
    grid = realistic_grid(args.x0, args.uar_max, args.dt)
    print(f"[tier1] unit test on {len(grid)} (n,p) points, M={args.M} ...")
    t1_ok, t1_rows = tier1_unit_test(grid, args.M, args.seed, args.outdir)

    # ---- shared presynaptic drive (identical for both mechanisms) -------------
    rng = np.random.default_rng(args.seed)
    n_spk = max(1, int(args.rate * args.simtime))
    spike_times = np.sort(rng.uniform(0.0, args.simtime, size=n_spk))

    # ---- TIER 2: Brian integration, both mechanisms ---------------------------
    print(f"[tier2] Brian integration (target={args.target}, Nsyn={args.Nsyn}, "
          f"T={args.simtime}s) ...")
    resF = run_ensemble("former", ns, spike_times, args.Nsyn,
                        args.simtime * second, args.rec_dt * ms, args.target, args.seed)
    resB = run_ensemble("binv", ns, spike_times, args.Nsyn,
                        args.simtime * second, args.rec_dt * ms, args.target, args.seed)
    t2 = tier2_plots_and_stats(resF, resB, args.outdir)

    # Tier-2 acceptance: pooled released-NT distributions must be compatible.
    t2_ok = (t2["p_Cfinal"] > 1e-3 and t2["p_Yfinal"] > 1e-3)

    # ---- report ----------------------------------------------------------------
    lines = []
    lines.append("=" * 74)
    lines.append("SMOKE TEST — asynchronous release: FORMER (Bernoulli loop) vs BINV")
    lines.append("=" * 74)
    lines.append("\nTIER 1 (unit: sampler vs exact binomial)")
    lines.append(f"  grid points: {len(grid)}   samples/point: {args.M}")
    hdr = (f"  {'n':>4} {'p':>9} | {'np':>8} {'mn_form':>8} {'mn_binv':>8} | "
           f"{'z_mean':>7} {'z_var':>7} | {'KS(f,b)':>8} {'p':>6}  res")
    lines.append(hdr)
    for r in t1_rows:
        lines.append(f"  {r['n']:>4} {r['p']:>9.2e} | {r['mean_th']:>8.4f} "
                     f"{r['mean_former']:>8.4f} {r['mean_binv']:>8.4f} | "
                     f"{r['z_mean']:>7.2f} {r['z_var']:>7.2f} | "
                     f"{r['ks_former_binv']:>8.4f} {r['p_former_binv']:>6.3f}  "
                     f"{'OK' if r['passed'] else 'FAIL'}")
    lines.append(f"  TIER 1: {'PASS' if t1_ok else 'FAIL'}")

    lines.append("\nTIER 2 (integration: released-NT traces, ensemble)")
    lines.append(f"  final cumulative async NT  mean: former={t2['mean_Cfinal_former']:.4f}"
                 f"  binv={t2['mean_Cfinal_binv']:.4f}")
    lines.append(f"                              std: former={t2['std_Cfinal_former']:.4f}"
                 f"  binv={t2['std_Cfinal_binv']:.4f}")
    lines.append(f"  KS final C_async : stat={t2['ks_Cfinal']:.4f}  p={t2['p_Cfinal']:.3f}")
    lines.append(f"  KS final Y_S     : stat={t2['ks_Yfinal']:.4f}  p={t2['p_Yfinal']:.3f}")
    lines.append(f"  TIER 2: {'PASS' if t2_ok else 'FAIL'}")
    lines.append("\n  Reminder: single traces (fig2) DIFFER by design; only ensemble")
    lines.append("  statistics (fig3) and pooled distributions (fig4) are expected to match.")
    lines.append("\nOVERALL: " + ("PASS" if (t1_ok and t2_ok) else "FAIL"))
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(args.outdir, "report.txt"), "w") as fh:
        fh.write(report + "\n")

    sys.exit(0 if (t1_ok and t2_ok) else 1)


if __name__ == "__main__":
    main()
