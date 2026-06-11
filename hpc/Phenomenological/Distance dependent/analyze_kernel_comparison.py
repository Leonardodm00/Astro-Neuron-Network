#!/usr/bin/env python3
"""
analyze_kernel_comparison.py
============================

Compare the two cultures produced by smoke_kernel_comparison.py (beta=1 exponential
vs beta=2 Gaussian, matched mean in-degree) on the summary statistics that actually
discriminate kernel SHAPE. Fully decoupled from the simulator: operates only on the
saved .npz / meta.json.

Summary statistics
------------------
  0. In-degree control     mean/sd realised <k>  (should be ~equal -> the match held)
  1. Firing rate           population mean rate + across-neuron CV
  2. ISI irregularity      population-mean CV of the inter-spike interval
  3. Correlation length    spike-count corr vs distance, fit C(d)=C0*exp(-d/xi)+b -> xi
  4. Synchrony chi         Var_t(pop) / mean_i Var_t(individual)   (Golomb-Rinzel)
  5. Network bursts        rate / size (fraction recruited) / duration / CV(IBI)
  6. Event spatial extent  periodic-aware spatial footprint of population events (um)

Prediction (heavy exponential tail vs fast Gaussian tail, matched <k>): the beta=1
culture has more long-range edges -> longer correlation length xi, higher synchrony
chi, larger/more-global bursts; the beta=2 culture is more spatially local ->
shorter xi, more graded correlation decay, more spatially-confined events.

Usage
-----
    python analyze_kernel_comparison.py --in_dir ./kernel_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- #
# IO
# ----------------------------------------------------------------------------- #
def load_culture(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- #
# Shared utilities
# ----------------------------------------------------------------------------- #
def binned_counts(spk_i, spk_t, N, T_ms, bin_ms):
    """(N, n_bins) spike-count matrix."""
    n_bins = int(np.ceil(T_ms / bin_ms))
    edges = np.arange(n_bins + 1) * bin_ms
    counts = np.zeros((N, n_bins), dtype=np.float64)
    if spk_t.size:
        bi = np.clip((spk_t / bin_ms).astype(int), 0, n_bins - 1)
        np.add.at(counts, (spk_i.astype(int), bi), 1.0)
    return counts, edges


def min_image_dist(pos, i, j, L, periodic):
    diff = pos[i] - pos[j]
    if periodic:
        diff = diff - L * np.round(diff / L)
    return np.sqrt((diff ** 2).sum(axis=-1))


# ----------------------------------------------------------------------------- #
# 1 / 2  rate & ISI
# ----------------------------------------------------------------------------- #
def rate_stats(spk_i, N, T_ms):
    per = np.bincount(spk_i.astype(int), minlength=N) / (T_ms / 1000.0)
    mean = per.mean()
    cv = per.std() / mean if mean > 0 else np.nan
    return dict(mean_rate_hz=float(mean), rate_cv=float(cv)), per


def isi_cv(spk_i, spk_t, N, min_spikes=5):
    cvs = []
    order = np.argsort(spk_t, kind="stable")
    si, st = spk_i[order].astype(int), spk_t[order]
    for n in range(N):
        t = st[si == n]
        if t.size >= min_spikes:
            isi = np.diff(t)
            if isi.mean() > 0:
                cvs.append(isi.std() / isi.mean())
    return float(np.mean(cvs)) if cvs else np.nan


# ----------------------------------------------------------------------------- #
# 3  spike-count correlation vs distance  ->  correlation length xi
# ----------------------------------------------------------------------------- #
def _exp_decay(d, C0, xi, b):
    return C0 * np.exp(-d / xi) + b


def corr_vs_distance(counts, pos, L, periodic, rng, n_pairs=20000,
                     n_dist_bins=14, min_active=3):
    """Mean pairwise spike-count correlation in distance bins, plus an exp-decay fit."""
    N = counts.shape[0]
    active = np.where((counts > 0).sum(axis=1) >= min_active)[0]
    if active.size < 10:
        return None
    # z-score the active rows once
    z = counts[active]
    z = (z - z.mean(axis=1, keepdims=True))
    norm = np.sqrt((z ** 2).sum(axis=1))
    keep = norm > 0
    active, z, norm = active[keep], z[keep], norm[keep]
    z = z / norm[:, None]
    M = active.size

    # sample unordered pairs of active neurons
    n_pairs = min(n_pairs, M * (M - 1) // 2)
    ia = rng.integers(0, M, size=n_pairs * 2)
    ib = rng.integers(0, M, size=n_pairs * 2)
    ok = ia != ib
    ia, ib = ia[ok][:n_pairs], ib[ok][:n_pairs]

    corr = (z[ia] * z[ib]).sum(axis=1)                       # Pearson r (rows pre-normalised)
    dist = min_image_dist(pos[active][ia], np.arange(0), pos[active][ib], L, periodic) \
        if False else _pair_dist(pos[active], ia, ib, L, periodic)

    bins = np.linspace(0, dist.max(), n_dist_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    mean_c = np.full(n_dist_bins, np.nan)
    for k in range(n_dist_bins):
        m = (dist >= bins[k]) & (dist < bins[k + 1])
        if m.sum() >= 20:
            mean_c[k] = corr[m].mean()

    valid = np.isfinite(mean_c)

    # --- model-free spatial-structure statistics (robust) ------------------- #
    # Short-range vs long-range mean correlation and their ratio. This captures
    # "how spatially graded is the coordination" WITHOUT relying on a fit:
    #   ratio >> 1 -> steep decay (local, Gaussian-like)
    #   ratio ~  1 -> flat / global coordination (heavy-tailed, exponential-like)
    c_short = float(mean_c[valid][0]) if valid.any() else np.nan
    c_long = float(mean_c[valid][-1]) if valid.any() else np.nan
    decay_ratio = float(c_short / c_long) if (np.isfinite(c_long) and abs(c_long) > 1e-4) else np.nan

    # --- exponential-decay fit (convenience; flagged if degenerate) --------- #
    # xi is meaningful only when correlations genuinely decay over the arena.
    # When bursts make correlations nearly flat the fit runs to its bound, so we
    # flag xi as non-identified once it exceeds the arena half-diagonal.
    xi = np.nan
    fit = None
    xi_saturated = False
    half_diag = (L / np.sqrt(2.0)) if periodic else (L * np.sqrt(2.0) / 2.0)
    if valid.sum() >= 4:
        x, y = centres[valid], mean_c[valid]
        try:
            p0 = (max(y[0] - y[-1], 1e-3), max(centres[-1] / 3, 1.0), y[-1])
            popt, _ = curve_fit(_exp_decay, x, y, p0=p0,
                                bounds=([0, 1.0, -1.0], [2.0, 3 * centres[-1], 1.0]),
                                maxfev=20000)
            xi = float(popt[1])
            fit = popt
            if xi > half_diag:        # longer than the arena can resolve -> "flat"
                xi_saturated = True
        except Exception:
            pass
    return dict(centres=centres, mean_corr=mean_c, xi_um=xi, fit=fit,
                xi_saturated=xi_saturated, c_short=c_short, c_long=c_long,
                decay_ratio=decay_ratio)


def _pair_dist(pos_sub, ia, ib, L, periodic):
    diff = pos_sub[ia] - pos_sub[ib]
    if periodic:
        diff = diff - L * np.round(diff / L)
    return np.sqrt((diff ** 2).sum(axis=1))


# ----------------------------------------------------------------------------- #
# 4  synchrony (Golomb-Rinzel chi)
# ----------------------------------------------------------------------------- #
def synchrony_chi(counts):
    pop = counts.mean(axis=0)
    var_pop = pop.var()
    mean_var_ind = counts.var(axis=1).mean()
    return float(np.sqrt(var_pop / mean_var_ind)) if mean_var_ind > 0 else np.nan


# ----------------------------------------------------------------------------- #
# 5 / 6  population bursts + spatial extent
# ----------------------------------------------------------------------------- #
def _smooth(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def population_bursts(spk_i, spk_t, pos, N, T_ms, L, periodic,
                      bin_ms=5.0, smooth_bins=3, thresh_sd=4.0, min_gap_ms=50.0):
    """Detect population events from the smoothed population rate; size = fraction of
    distinct neurons active in the event; footprint = periodic-aware spatial spread."""
    counts, edges = binned_counts(spk_i, spk_t, N, T_ms, bin_ms)
    pop = counts.sum(axis=0)
    pr = _smooth(pop, smooth_bins)
    base, sd = pr.mean(), pr.std()
    thresh = base + thresh_sd * sd

    above = pr > thresh
    if not above.any():
        return dict(n_events=0, rate_per_min=0.0, mean_size_frac=np.nan,
                    mean_duration_ms=np.nan, ibi_cv=np.nan,
                    mean_footprint_um=np.nan, sizes=np.array([]),
                    pop_rate_hz=pop / (N * bin_ms / 1000.0), bin_ms=bin_ms)

    # contiguous suprathreshold runs
    edges_idx = np.diff(above.astype(int))
    starts = np.where(edges_idx == 1)[0] + 1
    ends = np.where(edges_idx == -1)[0] + 1
    if above[0]:
        starts = np.r_[0, starts]
    if above[-1]:
        ends = np.r_[ends, above.size]

    # merge events closer than min_gap
    min_gap_bins = int(round(min_gap_ms / bin_ms))
    merged = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] <= min_gap_bins:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    sizes, durations, footprints, peak_times = [], [], [], []
    order = np.argsort(spk_t, kind="stable")
    st, si = spk_t[order], spk_i[order].astype(int)
    for s, e in merged:
        t0, t1 = edges[s], edges[e]
        m = (st >= t0) & (st < t1)
        if m.sum() == 0:
            continue
        parts = np.unique(si[m])
        sizes.append(parts.size / N)
        durations.append(t1 - t0)
        footprints.append(_spatial_footprint(pos[parts], L, periodic))
        peak_times.append(0.5 * (t0 + t1))

    sizes = np.asarray(sizes)
    peak_times = np.asarray(peak_times)
    ibi = np.diff(np.sort(peak_times))
    ibi_cv = float(ibi.std() / ibi.mean()) if ibi.size and ibi.mean() > 0 else np.nan

    return dict(
        n_events=len(sizes),
        rate_per_min=len(sizes) / (T_ms / 60000.0),
        mean_size_frac=float(np.mean(sizes)) if sizes.size else np.nan,
        mean_duration_ms=float(np.mean(durations)) if durations else np.nan,
        ibi_cv=ibi_cv,
        mean_footprint_um=float(np.mean(footprints)) if footprints else np.nan,
        sizes=sizes,
        pop_rate_hz=pop / (N * bin_ms / 1000.0),
        bin_ms=bin_ms,
    )


def _spatial_footprint(p, L, periodic):
    """Periodic-aware spatial spread (um): per-axis circular std, then RMS over axes."""
    if p.shape[0] < 2:
        return 0.0
    if not periodic:
        return float(np.sqrt((p.std(axis=0) ** 2).sum()))
    spreads = []
    for ax in range(p.shape[1]):
        theta = 2 * np.pi * p[:, ax] / L
        R = np.abs(np.exp(1j * theta).mean())
        R = min(max(R, 1e-9), 1 - 1e-12)
        spreads.append((L / (2 * np.pi)) * np.sqrt(-2.0 * np.log(R)))
    return float(np.sqrt(np.mean(np.square(spreads))))


# ----------------------------------------------------------------------------- #
# Per-culture pipeline
# ----------------------------------------------------------------------------- #
def analyze_one(cc, cfg, rng, corr_bin_ms=20.0, burst_bin_ms=5.0):
    N = int(cc["positions"].shape[0])
    T = float(cfg["simtime_ms"])
    L = float(cfg["L_um"])
    periodic = bool(cfg["periodic"])
    pos = cc["positions"]

    rs, _ = rate_stats(cc["spk_i"], N, T)
    counts_corr, _ = binned_counts(cc["spk_i"], cc["spk_t"], N, T, corr_bin_ms)
    counts_sync, _ = binned_counts(cc["spk_i"], cc["spk_t"], N, T, burst_bin_ms)

    res = dict(
        beta=float(cc["beta"]), d0_um=float(cc["d0_um"]), n_edges=int(cc["n_edges"]),
        realised_mean_indegree=float(cc["in_degree"].mean()),
        realised_sd_indegree=float(cc["in_degree"].std()),
        **rs,
        isi_cv=isi_cv(cc["spk_i"], cc["spk_t"], N),
        chi=synchrony_chi(counts_sync),
    )
    cvd = corr_vs_distance(counts_corr, pos, L, periodic, rng)
    if cvd:
        res["xi_um"] = cvd["xi_um"]
        res["xi_saturated"] = cvd["xi_saturated"]
        res["corr_short"] = cvd["c_short"]
        res["corr_long"] = cvd["c_long"]
        res["decay_ratio"] = cvd["decay_ratio"]
    else:
        res.update(xi_um=np.nan, xi_saturated=False, corr_short=np.nan,
                   corr_long=np.nan, decay_ratio=np.nan)
    bursts = population_bursts(cc["spk_i"], cc["spk_t"], pos, N, T, L, periodic,
                               bin_ms=burst_bin_ms)
    res.update(n_events=bursts["n_events"], burst_rate_per_min=bursts["rate_per_min"],
               burst_size_frac=bursts["mean_size_frac"],
               burst_duration_ms=bursts["mean_duration_ms"],
               burst_ibi_cv=bursts["ibi_cv"],
               event_footprint_um=bursts["mean_footprint_um"])
    return res, cvd, bursts, cc["in_degree"]


# ----------------------------------------------------------------------------- #
# Reporting
# ----------------------------------------------------------------------------- #
ROWS = [
    ("realised <k>           ", "realised_mean_indegree", "{:.2f}"),
    ("  sd(<k>)              ", "realised_sd_indegree", "{:.2f}"),
    ("# edges                ", "n_edges", "{:d}"),
    ("mean rate [Hz]         ", "mean_rate_hz", "{:.2f}"),
    ("rate CV (across cells) ", "rate_cv", "{:.2f}"),
    ("ISI CV (pop mean)      ", "isi_cv", "{:.2f}"),
    ("synchrony chi          ", "chi", "{:.3f}"),
    ("corr short-range r     ", "corr_short", "{:.3f}"),
    ("corr long-range r      ", "corr_long", "{:.3f}"),
    ("corr decay ratio       ", "decay_ratio", "{:.2f}"),
    ("corr length xi [um]    ", "xi_um", "{:.1f}"),
    ("# population events    ", "n_events", "{:d}"),
    ("burst rate [1/min]     ", "burst_rate_per_min", "{:.2f}"),
    ("burst size [frac N]    ", "burst_size_frac", "{:.3f}"),
    ("burst duration [ms]    ", "burst_duration_ms", "{:.1f}"),
    ("burst IBI CV           ", "burst_ibi_cv", "{:.2f}"),
    ("event footprint [um]   ", "event_footprint_um", "{:.1f}"),
]


def _fmt_xi(r):
    """xi string with a flag when the fit is non-identified (flat correlations)."""
    if not np.isfinite(r.get("xi_um", np.nan)):
        return "   n/a"
    s = f"{r['xi_um']:.1f}"
    return s + " (flat)" if r.get("xi_saturated") else s


def print_table(r1, r2):
    def fmt(v, f):
        if isinstance(v, float) and not np.isfinite(v):
            return "   n/a"
        try:
            return f.format(v)
        except Exception:
            return str(v)
    print("\n" + "=" * 64)
    print(f"{'statistic':<24}{'beta=1 (exp)':>18}{'beta=2 (gauss)':>20}")
    print("-" * 64)
    for label, key, f in ROWS:
        if key == "xi_um":
            v1, v2 = _fmt_xi(r1), _fmt_xi(r2)
        else:
            v1, v2 = fmt(r1[key], f), fmt(r2[key], f)
        print(f"{label:<24}{v1:>18}{v2:>20}")
    print("=" * 64)
    print("(beta=1: d0=sigma ; beta=2: d0=sqrt(2)*sigma ; analytic <k> matched.")
    print(" Realised <k> may differ slightly: on a 1mm^2 torus the exponential tail")
    print(" is truncated beyond L/sqrt(2), so beta=1 loses a little in-degree -- the")
    print(" non-convergence-at-1mm^2 signature. Robust spatial stat = decay ratio")
    print(" (>>1 local/Gaussian-like, ~1 global/exponential-like); xi flagged 'flat'")
    print(" when correlations don't decay within the arena.)")


def make_figure(c1, c2, r1, r2, cvd1, cvd2, b1, b2, deg1, deg2, cfg, out_png):
    L, T = float(cfg["L_um"]), float(cfg["simtime_ms"])
    fig, ax = plt.subplots(3, 2, figsize=(13, 12))

    # rasters (first 4 s, subsample neurons for clarity)
    for col, (cc, lab) in enumerate([(c1, "beta=1 exponential"), (c2, "beta=2 Gaussian")]):
        a = ax[0, col]
        tmax = min(4000.0, T)
        m = cc["spk_t"] < tmax
        a.scatter(cc["spk_t"][m] / 1000.0, cc["spk_i"][m], s=0.6, c="k", marker=".",
                  rasterized=True)
        a.set(title=lab, xlabel="time [s]", ylabel="neuron")
        a.set_xlim(0, tmax / 1000.0)

    # population rate
    for col, (bb, lab) in enumerate([(b1, "beta=1"), (b2, "beta=2")]):
        a = ax[1, col]
        t = np.arange(bb["pop_rate_hz"].size) * bb["bin_ms"] / 1000.0
        a.plot(t, bb["pop_rate_hz"], lw=0.5, c="C0")
        a.set(title=f"population rate ({lab})", xlabel="time [s]", ylabel="rate [Hz]")
        a.set_xlim(0, min(10.0, T / 1000.0))

    # correlation vs distance + fit (overlaid)
    a = ax[2, 0]
    for cvd, lab, col in [(cvd1, f"beta=1 (ratio={r1['decay_ratio']:.1f})", "C0"),
                          (cvd2, f"beta=2 (ratio={r2['decay_ratio']:.1f})", "C1")]:
        if cvd is None:
            continue
        a.plot(cvd["centres"], cvd["mean_corr"], "o", ms=4, color=col, label=lab)
        if cvd["fit"] is not None:
            xx = np.linspace(0, cvd["centres"][-1], 200)
            a.plot(xx, _exp_decay(xx, *cvd["fit"]), "-", color=col, lw=1.5)
    a.axhline(0, color="0.7", lw=0.6)
    a.set(title="spike-count correlation vs distance",
          xlabel="inter-soma distance [um]", ylabel="mean Pearson r")
    a.legend(fontsize=8)

    # in-degree distributions + burst-size distributions
    a = ax[2, 1]
    bins = np.linspace(0, max(deg1.max(), deg2.max()) + 1, 30)
    a.hist(deg1, bins=bins, alpha=0.5, label=f"beta=1 (mean {deg1.mean():.1f})", color="C0")
    a.hist(deg2, bins=bins, alpha=0.5, label=f"beta=2 (mean {deg2.mean():.1f})", color="C1")
    a.set(title="in-degree distribution (matched means)",
          xlabel="in-degree", ylabel="# neurons")
    a.legend(fontsize=8)

    fig.suptitle("Kernel-form comparison at matched mean in-degree "
                 f"(sigma={cfg['sigma_um']:.0f} um, L={L:.0f} um, "
                 f"rho={cfg['density_per_mm2']:.0f}/mm^2)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_png, dpi=140)
    print(f"\nFigure -> {out_png}")


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", default="./kernel_smoke")
    ap.add_argument("--out_dir", default=None, help="defaults to --in_dir")
    ap.add_argument("--seed", type=int, default=0, help="seed for pair sampling")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((in_dir / "meta.json").read_text())["config"]
    c1 = load_culture(in_dir / "culture_beta1.npz")
    c2 = load_culture(in_dir / "culture_beta2.npz")
    rng = np.random.default_rng(args.seed)

    r1, cvd1, b1, deg1 = analyze_one(c1, cfg, rng)
    r2, cvd2, b2, deg2 = analyze_one(c2, cfg, rng)

    print_table(r1, r2)
    out = {"beta1_exponential": r1, "beta2_gaussian": r2,
           "config": {k: cfg[k] for k in ("L_um", "density_per_mm2", "sigma_um",
                                          "p0", "periodic", "simtime_ms")}}
    (out_dir / "comparison_stats.json").write_text(json.dumps(out, indent=2, default=float))

    # tidy CSV
    keys = [k for _, k, _ in ROWS]
    lines = ["statistic,beta1_exponential,beta2_gaussian"]
    for label, k, _ in ROWS:
        lines.append(f"{label.strip()},{r1[k]},{r2[k]}")
    (out_dir / "comparison_stats.csv").write_text("\n".join(lines) + "\n")

    make_figure(c1, c2, r1, r2, cvd1, cvd2, b1, b2, deg1, deg2, cfg,
                out_dir / "comparison.png")
    print(f"Stats -> {out_dir}/comparison_stats.json , comparison_stats.csv")


if __name__ == "__main__":
    main()
