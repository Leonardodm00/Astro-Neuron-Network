#!/usr/bin/env python3
# =============================================================================
# indegree_convergence_analyze.py   (STEP 2 of 2)
# -----------------------------------------------------------------------------
# Analyse the in-degree distributions produced by indegree_convergence_generate.py
# and decide, per kernel cell (boundary, d0, beta, p0), whether the in-degree
# distribution P(k) becomes independent of culture size L, and from which
# minimum L* onward.
#
# Inputs
# ------
# A directory of  net_<boundary>_L<Lum>_rep<r>.npz  files, each holding an
# int32 tensor `indegrees` of shape (n_d0, n_beta, n_p0, N) plus axis lists.
#
# Method
# ------
# For each (boundary, d0, beta, p0) cell and each L:
#   * Pool the in-degree samples over all repetitions  ->  empirical P(k; L).
#   * MOMENTS (mean, std, CV) are computed on the FULL pooled sample.
#   * DISTRIBUTIONAL distances are computed after SUBSAMPLING every L to a fixed
#     size M_sub. This is essential: the pooled sample size grows with L
#     (n_rep * rho * L^2), and empirical-divergence bias shrinks with sample
#     size, so without equalising it a metric could fall purely because the
#     estimate got less noisy — a false 'convergence' confounded with L itself.
#
# Two framings (both reported):
#   reference  : distance( P(k;L)  ||  P(k; L_max) )      [L_max as proxy for L->inf]
#   consecutive: distance( P(k;L)  ||  P(k; L_next) )     [detects 'stops changing']
#
# Metrics (pure numpy; no scipy needed):
#   W1  : 1-D Wasserstein-1 (Earth-Mover) distance, in SYNAPSE units, primary.
#   JS  : Jensen-Shannon divergence (nats, bounded by ln2), symmetric, no infinities.
#   KL  : Kullback-Leibler divergence (nats), Laplace-smoothed (asymmetric).
#
# Noise floor & convergence threshold:
#   At sample size M_sub the distance between two independent subsamples of the
#   SAME distribution is not zero. We estimate that floor by repeatedly drawing
#   two subsamples from the reference (L_max) distribution. A cell is declared
#   "converged at L" when its distance is within  c_mult * floor  (default 2x):
#   i.e. statistically indistinguishable from the asymptote at this sample size.
#   Optionally an additional relative gate (W1 / mean_k_ref <= rel_w1_thresh).
#   L* = smallest eligible L such that the criterion holds for that L and ALL
#   larger L. If never satisfied, the cell is "not converged in range" (L*=NaN).
#
# Outputs (in --out_dir)
# ----------------------
#   indeg_convergence_long.csv     one row per (boundary,d0,beta,p0,L)
#   indeg_convergence_summary.csv  one row per (boundary,d0,beta,p0): L*, converged, floors
#   analysis_config.json
#   plots/ (optional, --plots): mean_k-vs-L and W1-vs-L grids per boundary
# =============================================================================

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

import numpy as np


# -----------------------------------------------------------------------------
# Pure-numpy distribution metrics
# -----------------------------------------------------------------------------
def w1_equal(p_sorted_pool, q_sorted_pool):
    """Wasserstein-1 between two EQUAL-SIZE empirical samples = mean abs diff of
    order statistics. Inputs may be unsorted; we sort here."""
    return float(np.mean(np.abs(np.sort(p_sorted_pool) - np.sort(q_sorted_pool))))


def _pmf(samples, kmax, alpha=0.0):
    c = np.bincount(np.asarray(samples, dtype=np.int64), minlength=kmax + 1).astype(np.float64)
    if alpha:
        c += alpha
    s = c.sum()
    return c / s if s > 0 else c


def kl_div(p_samp, q_samp, alpha=0.5):
    """KL(P || Q) in nats, on the common integer support, Laplace-smoothed so Q>0."""
    kmax = int(max(np.max(p_samp), np.max(q_samp)))
    p = _pmf(p_samp, kmax, alpha)
    q = _pmf(q_samp, kmax, alpha)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def js_div(p_samp, q_samp):
    """Jensen-Shannon divergence in nats (bounded by ln2), symmetric, finite."""
    kmax = int(max(np.max(p_samp), np.max(q_samp)))
    p = _pmf(p_samp, kmax, 0.0)
    q = _pmf(q_samp, kmax, 0.0)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def subsample(a, m, rng):
    """Draw m elements; without replacement if possible, else with replacement."""
    if len(a) >= m:
        return rng.choice(a, size=m, replace=False)
    return rng.choice(a, size=m, replace=True)


def boot_pair(p, q, m, n_boot, rng):
    """Bootstrap all three metrics between samples p and q, each subsampled to m."""
    w = np.empty(n_boot); j = np.empty(n_boot); k = np.empty(n_boot)
    for t in range(n_boot):
        ps = subsample(p, m, rng)
        qs = subsample(q, m, rng)
        w[t] = w1_equal(ps, qs)
        j[t] = js_div(ps, qs)
        k[t] = kl_div(ps, qs)
    return (w.mean(), w.std()), (j.mean(), j.std()), (k.mean(), k.std())


def noise_floor(ref, m, n_boot, rng):
    """Self-distance floor: two independent subsamples of the SAME (reference)
    distribution. Returns means for (W1, JS, KL)."""
    (w, _), (j, _), (k, _) = boot_pair(ref, ref, m, n_boot, rng)
    return w, j, k


# -----------------------------------------------------------------------------
# Data loading -> pooled samples per cell per L
# -----------------------------------------------------------------------------
_FNAME = re.compile(r"net_(?P<boundary>finite|periodic)_L(?P<L>\d+)_rep(?P<rep>\d+)\.npz$")


def load_pooled(in_dir, boundary):
    """Return (axes, pooled) where
       axes  = dict(d0_list, beta_list, p0_list, L_list, rho, N_by_L)
       pooled[(a,b,q)][L_um] = 1-D int array of pooled in-degrees over all reps."""
    files = sorted(glob.glob(os.path.join(in_dir, f"net_{boundary}_L*_rep*.npz")))
    if not files:
        raise FileNotFoundError(f"No files for boundary '{boundary}' in {in_dir}")

    by_L = {}
    for f in files:
        m = _FNAME.search(os.path.basename(f))
        if not m:
            continue
        by_L.setdefault(int(m["L"]), []).append(f)

    d0_list = beta_list = p0_list = None
    rho = None
    N_by_L = {}
    pooled = {}

    for L_um in sorted(by_L):
        rep_tensors = []
        for f in sorted(by_L[L_um]):
            d = np.load(f, allow_pickle=True)
            if d0_list is None:
                d0_list = d["d0_list"]; beta_list = d["beta_list"]; p0_list = d["p0_list"]
                rho = float(d["rho_per_mm2"])
            rep_tensors.append(d["indegrees"])     # (n_d0,n_beta,n_p0,N)
            N_by_L[L_um] = int(d["N"])
        stack = np.stack(rep_tensors, axis=0)       # (n_rep,n_d0,n_beta,n_p0,N)
        n_d0, n_beta, n_p0 = stack.shape[1:4]
        for a in range(n_d0):
            for b in range(n_beta):
                for q in range(n_p0):
                    key = (a, b, q)
                    pooled.setdefault(key, {})[L_um] = stack[:, a, b, q, :].ravel().astype(np.int64)
        del stack, rep_tensors

    axes = dict(d0_list=np.asarray(d0_list, float), beta_list=np.asarray(beta_list, float),
                p0_list=np.asarray(p0_list, float), L_list=sorted(by_L), rho=rho, N_by_L=N_by_L)
    return axes, pooled


# -----------------------------------------------------------------------------
# Per-cell convergence analysis
# -----------------------------------------------------------------------------
def analyse_cell(samples_by_L, args, rng):
    """Compute moments + distances + L* for one cell. Returns (long_rows, summary)."""
    Ls = sorted(samples_by_L)
    L_ref = Ls[-1]
    ref = samples_by_L[L_ref]

    eligible = [L for L in Ls if len(samples_by_L[L]) >= args.min_pooled_n]
    if len(eligible) < 2:
        eligible = Ls[:]                            # fall back: use all L
    M_sub = args.subsample if args.subsample > 0 else min(len(samples_by_L[L]) for L in eligible)

    fl_w1, fl_js, fl_kl = noise_floor(ref, M_sub, args.n_boot, rng)
    mean_k_ref = float(np.mean(ref))

    thr_w1 = args.c_mult * fl_w1
    rel_gate = (args.rel_w1_thresh > 0)

    long_rows = {}
    w1_ref_by_L, w1_next_by_L = {}, {}

    for i, L in enumerate(Ls):
        s = samples_by_L[L]
        mean_k = float(np.mean(s)); std_k = float(np.std(s))
        cv = std_k / mean_k if mean_k > 0 else float("nan")

        # reference-framing distances
        if L == L_ref:
            w1_r, js_r, kl_r, w1_r_sd = fl_w1, fl_js, fl_kl, 0.0
        else:
            (w1_r, w1_r_sd), (js_r, _), (kl_r, _) = boot_pair(s, ref, M_sub, args.n_boot, rng)
        w1_ref_by_L[L] = w1_r

        # consecutive-framing distance (to next larger L)
        if i < len(Ls) - 1:
            Lnext = Ls[i + 1]
            (w1_n, _), (js_n, _), (kl_n, _) = boot_pair(s, samples_by_L[Lnext], M_sub, args.n_boot, rng)
        else:
            w1_n = js_n = kl_n = float("nan")
        w1_next_by_L[L] = w1_n

        long_rows[L] = dict(
            L_um=L, N=0, pooled_n=len(s),   # N filled in run() from rho and L
            mean_k=mean_k, std_k=std_k, cv_k=cv,
            w1_ref=w1_r, w1_ref_sd=w1_r_sd, w1_ref_rel=(w1_r / mean_k_ref if mean_k_ref > 0 else float("nan")),
            js_ref=js_r, kl_ref=kl_r, w1_next=w1_n, js_next=js_n, kl_next=kl_n,
            M_sub=M_sub, eligible=(L in eligible),
        )

    # ---- L* (reference framing): smallest eligible L s.t. criterion holds for it and all larger ----
    def crit_ref(L):
        ok = w1_ref_by_L[L] <= thr_w1
        if rel_gate:
            ok = ok and (w1_ref_by_L[L] / mean_k_ref <= args.rel_w1_thresh if mean_k_ref > 0 else False)
        return ok

    Lstar_ref = float("nan"); converged_ref = False
    elig_sorted = [L for L in Ls if L in eligible]
    for idx, L in enumerate(elig_sorted):
        if all(crit_ref(L2) for L2 in elig_sorted[idx:]):
            Lstar_ref = L; converged_ref = True
            break

    # ---- L* (consecutive framing): smallest eligible L s.t. all consecutive gaps from L on are small ----
    Lstar_consec = float("nan"); converged_consec = False
    for idx, L in enumerate(elig_sorted[:-1]):
        if all((not np.isnan(w1_next_by_L[L2])) and (w1_next_by_L[L2] <= thr_w1)
               for L2 in elig_sorted[idx:-1]):
            Lstar_consec = L; converged_consec = True
            break

    summary = dict(
        mean_k_ref=mean_k_ref, std_k_ref=float(np.std(ref)),
        M_sub=M_sub, floor_w1=fl_w1, floor_js=fl_js, floor_kl=fl_kl, thr_w1=thr_w1,
        Lstar_ref=Lstar_ref, converged_ref=converged_ref,
        Lstar_consec=Lstar_consec, converged_consec=converged_consec,
    )
    return long_rows, summary


def _n_rep_guess_removed():
    pass


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def n_neurons(L_um, rho_per_mm2):
    return int(round(rho_per_mm2 * (L_um / 1000.0) ** 2))


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    long_path = os.path.join(args.out_dir, "indeg_convergence_long.csv")
    summ_path = os.path.join(args.out_dir, "indeg_convergence_summary.csv")
    long_fields = ["boundary", "d0_um", "beta", "p0", "L_um", "N", "pooled_n",
                   "mean_k", "std_k", "cv_k", "w1_ref", "w1_ref_sd", "w1_ref_rel",
                   "js_ref", "kl_ref", "w1_next", "js_next", "kl_next", "M_sub", "eligible"]
    summ_fields = ["boundary", "d0_um", "beta", "p0", "mean_k_ref", "std_k_ref",
                   "M_sub", "floor_w1", "thr_w1", "floor_js", "floor_kl",
                   "Lstar_ref_um", "converged_ref", "Lstar_consec_um", "converged_consec"]

    lf = open(long_path, "w", newline=""); lw = csv.DictWriter(lf, long_fields); lw.writeheader()
    sf = open(summ_path, "w", newline=""); sw = csv.DictWriter(sf, summ_fields); sw.writeheader()

    plot_data = {}   # boundary -> dict for optional plotting

    for boundary in args.boundaries:
        axes, pooled = load_pooled(args.in_dir, boundary)
        d0_list, beta_list, p0_list = axes["d0_list"], axes["beta_list"], axes["p0_list"]
        rho = axes["rho"]
        print(f"[ana] {boundary}: {len(pooled)} cells, L={axes['L_list']} um, "
              f"rho={rho}/mm^2", flush=True)

        plot_data[boundary] = dict(axes=axes, mean_k={}, w1_ref={}, floor={}, Lstar={})

        for (a, b, q), samples_by_L in pooled.items():
            long_rows, summary = analyse_cell(samples_by_L, args, rng)
            d0, beta, p0 = float(d0_list[a]), float(beta_list[b]), float(p0_list[q])

            for L, row in sorted(long_rows.items()):
                row["N"] = n_neurons(L, rho)
                lw.writerow(dict(boundary=boundary, d0_um=d0, beta=beta, p0=p0, **row))

            sw.writerow(dict(
                boundary=boundary, d0_um=d0, beta=beta, p0=p0,
                mean_k_ref=summary["mean_k_ref"], std_k_ref=summary["std_k_ref"],
                M_sub=summary["M_sub"], floor_w1=summary["floor_w1"], thr_w1=summary["thr_w1"],
                floor_js=summary["floor_js"], floor_kl=summary["floor_kl"],
                Lstar_ref_um=summary["Lstar_ref"], converged_ref=summary["converged_ref"],
                Lstar_consec_um=summary["Lstar_consec"], converged_consec=summary["converged_consec"]))

            if args.plots:
                Ls = sorted(long_rows)
                plot_data[boundary]["mean_k"][(a, b, q)] = [long_rows[L]["mean_k"] for L in Ls]
                plot_data[boundary]["w1_ref"][(a, b, q)] = [long_rows[L]["w1_ref"] for L in Ls]
                plot_data[boundary]["floor"][(a, b, q)] = summary["floor_w1"]
                plot_data[boundary]["Lstar"][(a, b, q)] = summary["Lstar_ref"]

    lf.close(); sf.close()
    with open(os.path.join(args.out_dir, "analysis_config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"[ana] wrote {long_path}\n[ana] wrote {summ_path}", flush=True)

    if args.plots:
        make_plots(plot_data, args)


def make_plots(plot_data, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = os.path.join(args.out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)

    for boundary, pd in plot_data.items():
        axes = pd["axes"]
        d0_list, beta_list, p0_list = axes["d0_list"], axes["beta_list"], axes["p0_list"]
        Ls = axes["L_list"]
        nB, nP = len(beta_list), len(p0_list)
        colors = plt.cm.viridis(np.linspace(0, 1, len(d0_list)))

        for kind, ylabel, logy in [("mean_k", "mean in-degree <k>", False),
                                    ("w1_ref", "W1 to L_max  [synapses]", True)]:
            fig, axarr = plt.subplots(nB, nP, figsize=(3.2 * nP, 2.6 * nB),
                                      squeeze=False, sharex=True)
            for b in range(nB):
                for q in range(nP):
                    ax = axarr[b][q]
                    for a in range(len(d0_list)):
                        y = pd[kind].get((a, b, q))
                        if y is None:
                            continue
                        ax.plot(Ls, y, "-o", ms=3, color=colors[a],
                                label=f"d0={d0_list[a]:.0f}")
                    if kind == "w1_ref":
                        fl = np.mean([pd["floor"].get((a, b, q), np.nan)
                                      for a in range(len(d0_list))])
                        ax.axhline(args.c_mult * fl, ls="--", c="k", lw=0.8)
                        if logy:
                            ax.set_yscale("log")
                    if b == 0:
                        ax.set_title(f"p0={p0_list[q]:.2g}", fontsize=8)
                    if q == 0:
                        ax.set_ylabel(f"β={beta_list[b]:.1f}\n{ylabel}", fontsize=7)
                    if b == nB - 1:
                        ax.set_xlabel("L [µm]", fontsize=8)
                    ax.tick_params(labelsize=6)
            axarr[0][-1].legend(fontsize=5, ncol=2, loc="upper right")
            fig.suptitle(f"{boundary}: {ylabel} vs culture size", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            out = os.path.join(pdir, f"{boundary}_{kind}_grid.png")
            fig.savefig(out, dpi=args.dpi); plt.close(fig)
            print(f"[ana] wrote {out}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="STEP 2: convergence analysis of in-degree distributions.")
    p.add_argument("--in_dir", required=True, help="directory of net_*.npz from step 1")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--boundaries", nargs="+", default=["finite", "periodic"],
                   choices=["finite", "periodic"])
    p.add_argument("--n_boot", type=int, default=200, help="bootstrap reps per distance")
    p.add_argument("--subsample", type=int, default=0,
                   help="fixed M_sub for distributional metrics; 0 => auto (min eligible pooled_n)")
    p.add_argument("--min_pooled_n", type=int, default=200,
                   help="L points with fewer pooled samples are excluded from L* search")
    p.add_argument("--c_mult", type=float, default=2.0,
                   help="convergence threshold = c_mult * noise floor")
    p.add_argument("--rel_w1_thresh", type=float, default=0.0,
                   help="optional extra gate: require W1/mean_k_ref <= this (0 disables)")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
