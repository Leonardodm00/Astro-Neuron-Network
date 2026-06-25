#!/usr/bin/env python3
# network_burst_detector.py
#
# Network-burst detector for the CAdEx neuron-astrocyte sweep outputs.
#
# Consumes the per-simulation spike data written by HPC_main_sweep.py /
# HPC_single_run.py:
#       <out_dir>/topo_<NN>/iter_<NN>.npz  with keys
#           spk_N_t : neuron spike times  [s]   (float32, 1-D)
#           spk_N_i : neuron spike index        (int32,   1-D)
#       and <out_dir>/job_args.json with Nn and simtime (read for normalisation).
#
# It maps the spike raster onto the same observables the wet-lab MEA pipeline
# reports (Frega/Mossink/Kleefstra: burst rate, burst duration, inter-burst
# interval IBI, CV of IBI) so model output is directly comparable to data.
#
# DETECTION LOGIC (the "synchrony gate" of the adaptation report, section 11.3:
#   min_synchrony ~ 0.20, prominence-frac 0.4, min-amp-ratio 2.5):
#
#   1. Population rate.  With bin width Delta = bin_s and M = ceil(T/Delta) bins,
#      bin edges b_m = m*Delta (m = 0..M). The population spike count in bin m is
#          c_m = #{ k : b_{m-1} <= t_k < b_m }     (summed over ALL neurons),
#      and the per-neuron population rate is
#          r_m = c_m / (Delta * N)                  [Hz],   N = n_neurons.
#      Bin centres are tbar_m = 0.5*(b_{m-1}+b_m).
#
#   2. Participation (synchrony proxy).  The fraction of DISTINCT neurons that
#      fired in bin m,
#          a_m = (1/N) * #{ j : neuron j fired at least once in bin m }  in [0,1].
#      a_m at a burst peak must be >= min_synchrony for the peak to count as a
#      genuine population burst (this rejects rate bumps produced by a small,
#      fast-firing subset -- tested explicitly in the smoke test).
#
#   3. Peak detection.  On a Gaussian-smoothed r (sigma = smooth_sigma_bins),
#      scipy.signal.find_peaks with
#          height    >= min_amp_ratio * r_base     (r_base = median_m r_m),
#          prominence>= prominence_frac * (max_m r_m - r_base),
#          distance  >= round(min_ibi_s / Delta).
#      Surviving peaks are then filtered by the synchrony gate (step 2).
#
#   4. Extent.  scipy.signal.peak_widths at rel_height = edge_rel_height gives
#      left/right intercepts (bin units) -> burst start/end -> duration.
#
#   5. Metrics + regime label (silent | async_irregular | tonic_high | bursting).
#
# Design (separation of concerns -- each layer is independently swappable):
#   IO        : load_iter_npz, read_network_size, iter_spike_arrays
#   signal    : population_rate
#   detection : detect_bursts
#   metrics   : summarize_bursts, analyze_recording
#   plot      : plot_detection                (lazy matplotlib import)
#   test      : smoke_test                    (synthetic ground truth)
#
# Dependencies: numpy, scipy.  matplotlib only if you call plot_detection.
# ASCII-only source (HPC locale safe). Greek spelled out in text.

import os
import json
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d


# ===========================================================================
# Parameters
# ===========================================================================
@dataclass
class BurstParams:
    """All tunables in one place (modularity: change one, nothing else breaks).

    Defaults encode the adaptation-report network-burst gate
    (min_synchrony 0.20, prominence_frac 0.40, min_amp_ratio 2.5) plus standard
    hiPSC-MEA timescales (network bursts ~ 0.1-1 s, IBI ~ 0.3-5 s).
    """
    bin_s: float = 0.025              # population-rate bin width [s] (25 ms)
    smooth_sigma_bins: float = 2.0    # Gaussian smoothing of r_m, in bins
    min_amp_ratio: float = 2.5        # peak rate must exceed this * baseline
    prominence_frac: float = 0.40     # prominence >= this * (max - baseline)
    min_synchrony: float = 0.20       # >= this fraction of neurons at the peak
    min_ibi_s: float = 0.10           # min separation between burst peaks [s]
    edge_rel_height: float = 0.80     # peak_widths rel_height for start/end
    # regime thresholds (per-neuron rates, Hz):
    silent_rate_hz: float = 0.10      # below -> 'silent'
    tonic_rate_hz: float = 5.0        # above this AND < 2 bursts -> 'tonic_high'
    min_bursts_for_bursting: int = 2  # need >= this many gated peaks to call it


# ===========================================================================
# IO layer
# ===========================================================================
def load_iter_npz(path):
    """Load one iter_<NN>.npz. Returns (spk_t [s], spk_i, meta dict)."""
    with np.load(path, allow_pickle=False) as d:
        spk_t = np.asarray(d['spk_N_t'], dtype=np.float64)
        spk_i = np.asarray(d['spk_N_i'], dtype=np.int64)
        meta = {}
        for k in ('conn_prob', 'p0_conn', 'd0_conn', 'beta_conn',
                  'topo_idx', 'seed_run'):
            if k in d:
                meta[k] = d[k].item() if d[k].ndim == 0 else d[k]
    return spk_t, spk_i, meta


def read_network_size(out_dir):
    """Read (Nn, simtime_s) from job_args.json. Looks at <out_dir>/job_args.json
    first; if absent (e.g. out_dir is a campaign ROOT and the file lives in each
    sweep_*/ task subdir), searches recursively and uses the first one found
    (Nn/simtime are campaign-wide constants). Returns (None, None) if none."""
    candidates = []
    top = os.path.join(out_dir, 'job_args.json')
    if os.path.isfile(top):
        candidates.append(top)
    else:
        for root, _dirs, files in os.walk(out_dir):
            if 'job_args.json' in files:
                candidates.append(os.path.join(root, 'job_args.json'))
                break
    for p in candidates:
        with open(p, 'r') as f:
            a = json.load(f)
        if a.get('Nn') is not None and a.get('simtime') is not None:
            return int(a['Nn']), float(a['simtime'])
    return None, None


def iter_spike_arrays(out_dir):
    """Yield (relpath, spk_t, spk_i, meta) for every iter_*.npz under out_dir."""
    for root, _dirs, files in os.walk(out_dir):
        for fn in sorted(files):
            if fn.startswith('iter_') and fn.endswith('.npz'):
                full = os.path.join(root, fn)
                spk_t, spk_i, meta = load_iter_npz(full)
                yield os.path.relpath(full, out_dir), spk_t, spk_i, meta


# ===========================================================================
# Signal layer
# ===========================================================================
def population_rate(spk_t, spk_i, n_neurons, simtime_s, bin_s):
    """Bin the raster into the per-neuron population rate r_m and the
    participation fraction a_m.

    Parameters
    ----------
    spk_t : (S,) array   spike times [s]
    spk_i : (S,) array   neuron index per spike (0 .. n_neurons-1)
    n_neurons : int      N (denominator for both r_m and a_m)
    simtime_s : float    recording length T [s]
    bin_s : float        bin width Delta [s]

    Returns
    -------
    t_centers : (M,) array  bin centres tbar_m [s]
    rate_hz   : (M,) array  per-neuron population rate r_m [Hz]
    active_frac : (M,) array  participation a_m in [0,1]
    """
    N = max(int(n_neurons), 1)
    M = max(int(np.ceil(simtime_s / bin_s)), 1)
    edges = np.arange(M + 1, dtype=np.float64) * bin_s
    t_centers = 0.5 * (edges[:-1] + edges[1:])

    if spk_t.size == 0:
        zeros = np.zeros(M, dtype=np.float64)
        return t_centers, zeros, zeros.copy()

    # population counts c_m (all neurons pooled)
    counts, _ = np.histogram(spk_t, bins=edges)
    rate_hz = counts.astype(np.float64) / (bin_s * N)

    # participation a_m: distinct (bin, neuron) pairs per bin
    bidx = np.minimum(np.floor(spk_t / bin_s).astype(np.int64), M - 1)
    bidx = np.maximum(bidx, 0)
    key = bidx * np.int64(N) + spk_i.astype(np.int64)
    uniq = np.unique(key)
    ubin = uniq // np.int64(N)
    active_counts = np.bincount(ubin, minlength=M).astype(np.float64)
    active_frac = active_counts / N
    return t_centers, rate_hz, active_frac


# ===========================================================================
# Detection layer
# ===========================================================================
@dataclass
class Burst:
    peak_time: float
    start_time: float
    end_time: float
    peak_rate_hz: float
    duration_s: float
    synchrony: float          # a_m at (or near) the peak bin


def detect_bursts(t_centers, rate_hz, active_frac, p: BurstParams):
    """Return (bursts, diagnostics). bursts is a list[Burst] passing all gates.

    diagnostics carries the smoothed rate, baseline and the RAW (pre-synchrony-
    gate) peak indices so callers/tests can inspect what the gate removed.
    """
    M = rate_hz.size
    diag = {'rate_smooth': rate_hz.copy(), 'baseline': 0.0,
            'raw_peaks': np.array([], dtype=int),
            'gated_peaks': np.array([], dtype=int)}
    if M < 3 or not np.any(rate_hz > 0):
        return [], diag

    r = gaussian_filter1d(rate_hz, sigma=max(p.smooth_sigma_bins, 1e-6))
    diag['rate_smooth'] = r
    r_base = float(np.median(r))
    r_span = float(r.max() - r_base)
    diag['baseline'] = r_base
    if r_span <= 0.0:
        return [], diag

    height = max(p.min_amp_ratio * r_base, r_base + 0.05 * r_span)
    prominence = p.prominence_frac * r_span
    distance = max(1, int(round(p.min_ibi_s / (t_centers[1] - t_centers[0]))))

    raw_peaks, _props = find_peaks(
        r, height=height, prominence=prominence, distance=distance)
    diag['raw_peaks'] = raw_peaks
    if raw_peaks.size == 0:
        return [], diag

    # synchrony gate: a_m at the peak bin (max over +-1 bin for robustness)
    def sync_at(idx):
        lo, hi = max(0, idx - 1), min(M, idx + 2)
        return float(active_frac[lo:hi].max())

    keep_mask = np.array([sync_at(pk) >= p.min_synchrony for pk in raw_peaks])
    gated = raw_peaks[keep_mask]
    diag['gated_peaks'] = gated
    if gated.size == 0:
        return [], diag

    widths, _wh, left_ips, right_ips = peak_widths(
        r, gated, rel_height=p.edge_rel_height)
    dt = t_centers[1] - t_centers[0]
    t0 = t_centers[0]

    bursts = []
    for k, pk in enumerate(gated):
        start = t0 + left_ips[k] * dt
        end = t0 + right_ips[k] * dt
        bursts.append(Burst(
            peak_time=float(t_centers[pk]),
            start_time=float(start),
            end_time=float(end),
            peak_rate_hz=float(r[pk]),
            duration_s=float(max(end - start, 0.0)),
            synchrony=float(sync_at(pk)),
        ))
    return bursts, diag


# ===========================================================================
# Metrics layer
# ===========================================================================
def summarize_bursts(bursts, rate_hz, simtime_s, p: BurstParams):
    """Reduce a detection to scalar observables + a regime label."""
    mean_rate_hz = float(np.mean(rate_hz)) if rate_hz.size else 0.0
    n = len(bursts)
    peak_times = np.array([b.peak_time for b in bursts], dtype=np.float64)
    durs = np.array([b.duration_s for b in bursts], dtype=np.float64)

    if n >= 2:
        ibis = np.diff(np.sort(peak_times))
        mean_ibi = float(ibis.mean())
        cv_ibi = float(ibis.std() / ibis.mean()) if ibis.mean() > 0 else 0.0
    else:
        ibis = np.array([])
        mean_ibi = float('nan')
        cv_ibi = float('nan')

    # regime classification (mutually exclusive, checked in order)
    if mean_rate_hz < p.silent_rate_hz and n == 0:
        regime = 'silent'
    elif n >= p.min_bursts_for_bursting:
        regime = 'bursting'
    elif mean_rate_hz >= p.tonic_rate_hz:
        regime = 'tonic_high'
    else:
        regime = 'async_irregular'

    return {
        'regime': regime,
        'is_bursting': regime == 'bursting',
        'n_bursts': int(n),
        'burst_rate_hz': float(n / simtime_s) if simtime_s > 0 else 0.0,
        'mean_rate_hz': mean_rate_hz,
        'mean_ibi_s': mean_ibi,
        'cv_ibi': cv_ibi,
        'mean_burst_dur_s': float(durs.mean()) if durs.size else float('nan'),
        'mean_synchrony': (float(np.mean([b.synchrony for b in bursts]))
                           if n else 0.0),
    }


def analyze_recording(spk_t, spk_i, n_neurons, simtime_s,
                      params: Optional[BurstParams] = None,
                      return_traces=False):
    """Top-level convenience: signal -> detect -> summarize."""
    p = params or BurstParams()
    t_c, rate, afrac = population_rate(spk_t, spk_i, n_neurons, simtime_s, p.bin_s)
    bursts, diag = detect_bursts(t_c, rate, afrac, p)
    summary = summarize_bursts(bursts, rate, simtime_s, p)
    if return_traces:
        return summary, bursts, dict(t_centers=t_c, rate_hz=rate,
                                     active_frac=afrac, **diag)
    return summary, bursts


# ===========================================================================
# Plot layer (lazy import: core + smoke test run headless without matplotlib)
# ===========================================================================
def plot_detection(spk_t, spk_i, n_neurons, simtime_s, out_path,
                   params: Optional[BurstParams] = None, dpi=150, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    p = params or BurstParams()
    summary, bursts, tr = analyze_recording(
        spk_t, spk_i, n_neurons, simtime_s, p, return_traces=True)

    fig, (axr, axp) = plt.subplots(
        2, 1, figsize=(11, 5.5), sharex=True,
        gridspec_kw=dict(height_ratios=[2, 1]))
    if spk_t.size:
        axr.scatter(spk_t, spk_i, s=1.5, c='k', alpha=0.5, rasterized=True)
    axr.set_ylabel('neuron index')
    axr.set_ylim(-1, n_neurons)

    axp.plot(tr['t_centers'], tr['rate_hz'], lw=0.8, color='0.6',
             label='r_m (raw)')
    axp.plot(tr['t_centers'], tr['rate_smooth'], lw=1.4, color='C0',
             label='r_m (smoothed)')
    axp.axhline(tr['baseline'], ls='--', lw=0.8, color='0.4', label='baseline')
    for b in bursts:
        axr.axvspan(b.start_time, b.end_time, color='C1', alpha=0.15)
        axp.axvspan(b.start_time, b.end_time, color='C1', alpha=0.15)
        axp.plot(b.peak_time, b.peak_rate_hz, 'v', color='C3', ms=6)
    axp.set_ylabel('pop. rate (Hz)')
    axp.set_xlabel('time (s)')
    axp.legend(fontsize=8, loc='upper right')

    ttl = title or ''
    axr.set_title(f"{ttl}  regime={summary['regime']}  "
                  f"n_bursts={summary['n_bursts']}  "
                  f"IBI={summary['mean_ibi_s']:.2f}s  "
                  f"CV_IBI={summary['cv_ibi']:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return summary


# ===========================================================================
# Smoke test (synthetic ground truth -- runs with numpy/scipy only)
# ===========================================================================
def _poisson_times(rng, rate_hz, t0, t1):
    """Homogeneous-Poisson spike times in [t0, t1) at rate_hz (thinning-free)."""
    if rate_hz <= 0 or t1 <= t0:
        return np.empty(0)
    expected = rate_hz * (t1 - t0)
    n = rng.poisson(expected)
    return np.sort(rng.uniform(t0, t1, size=int(n)))


def _gen_bursting(rng, N, T, ibi, burst_dur, frac, intra_rate, bg_rate):
    """Synchronous network bursts: a fraction `frac` of neurons co-fire in each
    burst window; low background everywhere. Returns spk_t, spk_i, true_nbursts."""
    onsets = np.arange(0.5, T - burst_dur, ibi)
    n_part = max(int(round(frac * N)), 1)
    ts, isx = [], []
    for on in onsets:
        part = rng.choice(N, size=n_part, replace=False)
        for j in part:
            tj = _poisson_times(rng, intra_rate, on, on + burst_dur)
            ts.append(tj); isx.append(np.full(tj.size, j))
    for j in range(N):                                   # background
        tj = _poisson_times(rng, bg_rate, 0.0, T)
        ts.append(tj); isx.append(np.full(tj.size, j))
    spk_t = np.concatenate(ts) if ts else np.empty(0)
    spk_i = np.concatenate(isx).astype(np.int64) if isx else np.empty(0, np.int64)
    order = np.argsort(spk_t)
    return spk_t[order], spk_i[order], len(onsets)


def _gen_homogeneous(rng, N, T, rate_hz):
    """Every neuron independent Poisson at rate_hz (no synchrony)."""
    ts, isx = [], []
    for j in range(N):
        tj = _poisson_times(rng, rate_hz, 0.0, T)
        ts.append(tj); isx.append(np.full(tj.size, j))
    spk_t = np.concatenate(ts) if ts else np.empty(0)
    spk_i = np.concatenate(isx).astype(np.int64) if isx else np.empty(0, np.int64)
    order = np.argsort(spk_t)
    return spk_t[order], spk_i[order]


def _gen_subset_bump(rng, N, T, n_bumps, bump_dur, frac, intra_rate):
    """Rate bumps produced by a SMALL subset (frac < min_synchrony) firing fast.
    Makes a rate peak but must be REJECTED by the synchrony gate."""
    onsets = np.linspace(1.0, T - bump_dur - 1.0, n_bumps)
    n_part = max(int(round(frac * N)), 1)
    ts, isx = [], []
    for on in onsets:
        part = rng.choice(N, size=n_part, replace=False)
        for j in part:
            tj = _poisson_times(rng, intra_rate, on, on + bump_dur)
            ts.append(tj); isx.append(np.full(tj.size, j))
    spk_t = np.concatenate(ts) if ts else np.empty(0)
    spk_i = np.concatenate(isx).astype(np.int64) if isx else np.empty(0, np.int64)
    order = np.argsort(spk_t)
    return spk_t[order], spk_i[order]


def smoke_test(verbose=True):
    """Six checks against synthetic ground truth. Returns True iff all pass.

    1. clean bursting   -> regime 'bursting', n_bursts ~ true (+-1), IBI ~ true
    2. tonic-high       -> 'tonic_high', is_bursting False
    3. silent           -> 'silent', n_bursts 0
    4. async-irregular  -> is_bursting False (not 'bursting')
    5. synchrony gate   -> subset bump: raw peaks exist BUT gated peaks == 0
    6. empty raster     -> 'silent', no crash
    """
    rng = np.random.default_rng(0)
    N, T = 200, 60.0
    p = BurstParams()
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok = ok and bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

    # 1. clean bursting (true IBI = 2.0 s)
    true_ibi = 2.0
    st, si, n_true = _gen_bursting(rng, N, T, ibi=true_ibi, burst_dur=0.15,
                                   frac=0.7, intra_rate=80.0, bg_rate=0.2)
    s1, b1, tr1 = analyze_recording(st, si, N, T, p, return_traces=True)
    check("bursting: regime", s1['regime'] == 'bursting', f"-> {s1['regime']}")
    check("bursting: n_bursts within +-1 of true",
          abs(s1['n_bursts'] - n_true) <= 1, f"det={s1['n_bursts']} true={n_true}")
    check("bursting: IBI within 15% of true",
          np.isfinite(s1['mean_ibi_s']) and
          abs(s1['mean_ibi_s'] - true_ibi) / true_ibi < 0.15,
          f"det={s1['mean_ibi_s']:.3f}s true={true_ibi}s")

    # 2. tonic-high (30 Hz/neuron asynchronous)
    st, si = _gen_homogeneous(rng, N, T, rate_hz=30.0)
    s2, _ = analyze_recording(st, si, N, T, p)
    check("tonic_high: regime", s2['regime'] == 'tonic_high', f"-> {s2['regime']}")
    check("tonic_high: not bursting", not s2['is_bursting'])

    # 3. silent
    st, si = _gen_homogeneous(rng, N, T, rate_hz=0.01)
    s3, _ = analyze_recording(st, si, N, T, p)
    check("silent: regime", s3['regime'] == 'silent', f"-> {s3['regime']}")
    check("silent: n_bursts == 0", s3['n_bursts'] == 0)

    # 4. async-irregular (2 Hz/neuron, no synchrony)
    st, si = _gen_homogeneous(rng, N, T, rate_hz=2.0)
    s4, _ = analyze_recording(st, si, N, T, p)
    check("async_irregular: not bursting", not s4['is_bursting'],
          f"-> {s4['regime']}")

    # 5. synchrony gate: small subset (5% << 20%) firing fast
    st, si = _gen_subset_bump(rng, N, T, n_bumps=10, bump_dur=0.15,
                              frac=0.05, intra_rate=120.0)
    s5, b5, tr5 = analyze_recording(st, si, N, T, p, return_traces=True)
    raw_n = int(tr5['raw_peaks'].size)
    gated_n = int(tr5['gated_peaks'].size)
    check("gate: raw peaks exist", raw_n >= 1, f"raw={raw_n}")
    check("gate: all rejected by synchrony", gated_n == 0 and not s5['is_bursting'],
          f"gated={gated_n} regime={s5['regime']}")

    # 6. empty raster
    s6, _ = analyze_recording(np.empty(0), np.empty(0, np.int64), N, T, p)
    check("empty: silent, no crash", s6['regime'] == 'silent' and
          s6['n_bursts'] == 0)

    if verbose:
        print(f"\nsmoke_test: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


# ===========================================================================
# CLI
# ===========================================================================
def _cli():
    import argparse
    ap = argparse.ArgumentParser(
        description="Network-burst detector for CAdEx sweep outputs.")
    ap.add_argument('--smoke_test', action='store_true',
                    help="Run the synthetic-ground-truth smoke test and exit.")
    ap.add_argument('--out_dir', default=None,
                    help="Sweep output dir; scans topo_*/iter_*.npz and prints "
                         "a per-iter burst summary (reads Nn/simtime from "
                         "job_args.json).")
    ap.add_argument('--plot_first', type=int, default=0,
                    help="Also write detection PNGs for the first K iters.")
    args = ap.parse_args()

    if args.smoke_test:
        raise SystemExit(0 if smoke_test() else 1)

    if args.out_dir:
        Nn, T = read_network_size(args.out_dir)
        if Nn is None:
            raise SystemExit("ERROR: job_args.json not found in --out_dir")
        print(f"Nn={Nn}  simtime={T}s   scanning {args.out_dir} ...")
        n_plotted = 0
        for rel, st, si, meta in iter_spike_arrays(args.out_dir):
            s, _ = analyze_recording(st, si, Nn, T)
            print(f"  {rel:40s} regime={s['regime']:15s} "
                  f"n_bursts={s['n_bursts']:3d} IBI={s['mean_ibi_s']:.2f} "
                  f"rate={s['mean_rate_hz']:.2f}Hz")
            if n_plotted < args.plot_first:
                png = os.path.join(args.out_dir, rel.replace('.npz', '_burst.png'))
                plot_detection(st, si, Nn, T, png, title=rel)
                n_plotted += 1
    else:
        # no args -> run the smoke test as the default self-check
        raise SystemExit(0 if smoke_test() else 1)


if __name__ == '__main__':
    _cli()
