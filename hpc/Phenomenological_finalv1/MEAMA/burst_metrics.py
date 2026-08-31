#!/usr/bin/env python3
"""
burst_metrics.py -- network-burst detection and scalar metrics from population
spike trains.

PURE COMPUTE LAYER. No file I/O, no plotting, no globals. Everything here takes
numpy arrays in and returns numpy arrays / plain dicts out, so it is trivially
unit-testable (see test_burst_metrics.py) and can be reused by the HPC sweep,
the aggregate index builder, or an interactive notebook without dragging in
matplotlib or a directory layout.

------------------------------------------------------------------------------
WHAT IT COMPUTES
------------------------------------------------------------------------------
For one simulation we are handed the neuronal population spike train as two
row-aligned arrays:

    spk_t[k]  : time (seconds) of the k-th spike, for k = 0 .. K-1
    spk_i[k]  : index (0 .. N_neurons-1) of the neuron that emitted spike k

together with the recording duration T_rec (seconds) and the neuron count
N_neurons. From these we detect *network bursts* with TWO independent
detectors and compute, conditional on the detected burst set B of that run,
the metric panel used by the MEA literature (Mossink et al. 2019, Nat Commun
10:4928 -- the Kleefstra/NMDAR paper in the project library):

    burst_rate_per_min                 (count) / (T_rec / 60)
    burst_dur_mean_s, burst_dur_cv     mean and CV of burst DURATION  D | B
    nibi_mean_s, nibi_cv               mean and CV of the network inter-burst
                                       interval  IBI | B  (silent gap:
                                       onset_{b+1} - offset_b)
    spikes_per_burst_mean              mean spike count inside a burst
    intra_burst_rate_hz                mean within-burst per-neuron rate
    frac_spikes_outside_burst          spikes not in any burst / all spikes
    participation_mean                 mean (active neurons firing in burst) /
                                       (active neurons), i.e. the network-wide
                                       analogue of Mossink's ">80% of active
                                       channels" criterion

Two detector-independent single-cell irregularity descriptors are also
returned (Gorski et al. 2021, CAdEx; Neural Computation 33:41):

    cv_isi_mean      population mean over neurons of  CV_ISI = std(ISI)/mean(ISI)
    adapt_index_mean population mean of the adaptation index A (eq. 6.1):
                       A = (1/(M-1)) * sum_{j=1..M-1}
                             (ISI_{j+1} - ISI_j) / (ISI_{j+1} + ISI_j)
                     for each neuron with M >= 2 ISIs. A in (-1, 1); A > 0 for
                     decelerating (adapting) trains, A < 0 for accelerating.

Plus two cheap whole-train descriptors (these fill the previously-empty
across_cell_rate_cv / frac_active columns in summary.csv):

    across_cell_rate_cv   CV across neurons of the per-neuron mean rate
    frac_active           (neurons with >= 1 spike) / N_neurons

------------------------------------------------------------------------------
THE TWO DETECTORS (the user asked for both, cross-checked)
------------------------------------------------------------------------------
(1) POPULATION-RATE detector  [key: "poprate"]
    Bin the whole population into a per-neuron population rate r_pn(t)
    [Hz/neuron], Gaussian-smooth it, and threshold at a robust level
    theta = median(r_pn) + k * (1.4826 * MAD(r_pn)). Contiguous supra-threshold
    epochs (after merging gaps < merge_gap_s and dropping epochs shorter than
    min_dur_s or with fewer than min_spikes spikes) are network bursts. Fast,
    detector of choice for a fully observed population raster
    (Wagenaar 2006; Chiappalone 2006 style).

(2) PARTICIPATION / log-ISI detector  [key: "logisi"]
    MEA-faithful, i.e. it reproduces Mossink's pipeline as closely as a fully
    observed network allows. Per neuron we find single-cell bursts as maximal
    runs of >= min_spikes_cell spikes whose internal ISIs are all <= ISI_th,
    where ISI_th is the log-ISI void (the minimum of the log10(ISI) histogram
    between the intra-burst peak and the inter-burst peak) capped at
    isi_cutoff_ms (default 100 ms, exactly Mossink's "minimal inter-spike
    interval of 100 ms"; we fall back to the cap when the histogram is not
    cleanly bimodal -- this fallback is an explicit simplification of the full
    Pasquale et al. 2010 logISI routine and is flagged here, not hidden).
    A network burst is then a contiguous epoch in which the fraction of ACTIVE
    neurons currently inside one of their single-cell bursts is >=
    participation_frac (default 0.8 == Mossink's ">80% of active channels").

CROSS-CHECK
    cross_check_jaccard : Jaccard overlap of the two detectors' in-burst time
                          supports on a fine grid (1.0 == identical supports,
                          0.0 == disjoint). Lets you see at a glance whether the
                          modeling-style and MEA-style detectors agree for a run.

------------------------------------------------------------------------------
DEGENERATE CASES
------------------------------------------------------------------------------
A quantity that is undefined for a given run is returned as float('nan'), never
silently as 0. Examples: CV of a single-burst run (need >= 2 bursts), IBI of a
<2-burst run, intra-burst rate of a 0-burst run. Counts that are genuinely zero
(burst_rate_per_min for a run with no bursts) are returned as 0.0. Downstream
ranking code must decide how to treat NaNs (see literature_distance()).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class BurstConfig:
    """All free knobs of the two detectors, in one editable place."""
    # --- shared time grid ---
    grid_dt_s: float = 0.005              # fine grid for rate / participation (5 ms)

    # --- population-rate detector ---
    pr_smooth_sigma_s: float = 0.015      # Gaussian smoothing of r_pn(t) (15 ms)
    pr_thresh_k: float = 3.0              # theta = mean + k * std (network-burst rule)
    pr_merge_gap_s: float = 0.10          # merge supra-threshold epochs closer than this
    pr_min_dur_s: float = 0.02            # drop bursts shorter than this
    pr_min_spikes: int = 5                # drop bursts with fewer spikes (Mossink: >=5)

    # --- participation / log-ISI detector ---
    li_isi_cutoff_ms: float = 100.0       # cap on intra-burst ISI threshold (Mossink)
    li_min_spikes_cell: int = 5           # min spikes for a single-cell burst (Mossink)
    li_participation_frac: float = 0.80   # network-burst threshold (Mossink: >80%)
    li_merge_gap_s: float = 0.10
    li_min_dur_s: float = 0.02
    li_logisi_nbins: int = 60             # bins for the log10(ISI) histogram

    # --- which detector defines the metrics used for the literature distance ---
    # "logisi" matches Mossink's network-burst definition exactly; "poprate" is
    # the robust modeling-style alternative.
    scoring_detector: str = "logisi"


# ===========================================================================
# Small numeric helpers
# ===========================================================================
def _robust_std(x: np.ndarray) -> float:
    """MAD-based standard-deviation estimate (1.4826 * MAD). 0 if degenerate."""
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)


def _cv(x: np.ndarray) -> float:
    """Coefficient of variation; NaN if < 2 samples or zero mean."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    m = x.mean()
    if m == 0.0:
        return float("nan")
    return float(x.std(ddof=1) / m)


def _intervals_from_mask(mask: np.ndarray, dt: float,
                         t0: float = 0.0) -> List[Tuple[float, float]]:
    """Contiguous True runs of a boolean grid mask -> list of (start_s, stop_s).

    The stop time is the right edge of the last True bin, i.e. (last_idx+1)*dt,
    so a single True bin has duration exactly dt.
    """
    if mask.size == 0 or not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    stops = np.flatnonzero(d == -1) + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        stops = np.r_[stops, mask.size]
    return [(t0 + s * dt, t0 + e * dt) for s, e in zip(starts, stops)]


def _merge_intervals(intervals: List[Tuple[float, float]],
                     gap: float) -> List[Tuple[float, float]]:
    """Merge intervals whose inter-interval gap is < gap seconds."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ===========================================================================
# Population rate
# ===========================================================================
def population_rate(spk_t: np.ndarray, n_neurons: int, t_rec: float,
                    dt: float, smooth_sigma_s: float
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-neuron population rate r_pn(t) [Hz/neuron] on a regular grid.

    Returns (t_centers, r_pn). r_pn[k] = (spikes in bin k) / (dt * n_neurons),
    optionally Gaussian-smoothed with std smooth_sigma_s (set 0 to disable).
    """
    n_bins = max(int(np.ceil(t_rec / dt)), 1)
    edges = np.arange(n_bins + 1) * dt
    counts, _ = np.histogram(spk_t, bins=edges)
    r_pn = counts.astype(np.float64) / (dt * max(n_neurons, 1))

    if smooth_sigma_s and smooth_sigma_s > 0:
        sigma_bins = smooth_sigma_s / dt
        half = int(np.ceil(3 * sigma_bins))
        xk = np.arange(-half, half + 1)
        kern = np.exp(-0.5 * (xk / sigma_bins) ** 2)
        kern /= kern.sum()
        r_pn = np.convolve(r_pn, kern, mode="same")

    t_centers = (edges[:-1] + edges[1:]) * 0.5
    return t_centers, r_pn


# ===========================================================================
# Detector 1 -- population-rate threshold
# ===========================================================================
def detect_bursts_poprate(spk_t: np.ndarray, n_neurons: int, t_rec: float,
                          cfg: BurstConfig) -> List[Tuple[float, float]]:
    """Network bursts as robust-threshold crossings of the smoothed pop rate."""
    if spk_t.size == 0 or t_rec <= 0:
        return []
    dt = cfg.grid_dt_s
    _, r_pn = population_rate(spk_t, n_neurons, t_rec, dt, cfg.pr_smooth_sigma_s)

    # Standard scale-adaptive network-burst threshold (Chiappalone-style):
    #   theta = mean(r_pn) + k * std(r_pn).
    # We deliberately do NOT use a median/MAD robust estimator here: the fine
    # grid is strongly zero-inflated (most bins are empty), which drives both
    # median and MAD to 0 and collapses the threshold so that every background
    # spike is flagged. The plain mean/std is dominated by the burst bins and so
    # lands sensibly between baseline and peak across very different regimes.
    mu = float(r_pn.mean())
    sd = float(r_pn.std())
    if sd <= 0.0:
        return []
    theta = mu + cfg.pr_thresh_k * sd
    mask = r_pn > theta
    intervals = _intervals_from_mask(mask, dt, t0=0.0)
    intervals = _merge_intervals(intervals, cfg.pr_merge_gap_s)

    spk_sorted = np.sort(spk_t)
    out: List[Tuple[float, float]] = []
    for s, e in intervals:
        if (e - s) < cfg.pr_min_dur_s:
            continue
        n_in = int(np.searchsorted(spk_sorted, e) - np.searchsorted(spk_sorted, s))
        if n_in < cfg.pr_min_spikes:
            continue
        out.append((s, e))
    return out


# ===========================================================================
# Detector 2 -- per-neuron log-ISI single-cell bursts + participation
# ===========================================================================
def _logisi_threshold_ms(isis_ms: np.ndarray, cfg: BurstConfig) -> float:
    """Intra-burst ISI threshold [ms] = log-ISI void, capped at isi_cutoff_ms.

    Simplification (flagged): the full Pasquale et al. 2010 logISI routine fits
    the intra/inter peaks and picks the void between them with a peak-skipping
    rule. Here we take the histogram minimum that lies below the cutoff and
    after the first non-empty bin; if no interior minimum is found we fall back
    to the cutoff. The cutoff alone already reproduces Mossink's "minimal ISI
    100 ms" rule, so the fallback is conservative, not arbitrary.
    """
    cutoff = cfg.li_isi_cutoff_ms
    pos = isis_ms[isis_ms > 0]
    if pos.size < 3:
        return cutoff
    logs = np.log10(pos)
    hist, edges = np.histogram(logs, bins=cfg.li_logisi_nbins)
    centers_ms = 10.0 ** (0.5 * (edges[:-1] + edges[1:]))
    # candidate threshold bins: interior local minima below the cutoff
    below = centers_ms < cutoff
    if below.sum() < 3 or hist.sum() == 0:
        return cutoff
    # find the global peak among below-cutoff bins (intra-burst peak), then the
    # smallest-count interior bin to its right but still below cutoff (the void)
    idx_below = np.flatnonzero(below)
    peak_rel = idx_below[np.argmax(hist[idx_below])]
    right = idx_below[idx_below > peak_rel]
    if right.size == 0:
        return cutoff
    void_bin = right[np.argmin(hist[right])]
    if hist[void_bin] >= hist[peak_rel]:        # no real void -> not bimodal
        return cutoff
    return float(min(centers_ms[void_bin], cutoff))


def single_cell_bursts(spk_t_neuron: np.ndarray, cfg: BurstConfig
                       ) -> List[Tuple[float, float]]:
    """Single-cell bursts of one neuron -> list of (first_spike_s, last_spike_s).

    A burst is a maximal run of >= li_min_spikes_cell spikes whose consecutive
    ISIs are all <= the (per-neuron) log-ISI threshold.
    """
    t = np.sort(np.asarray(spk_t_neuron, dtype=np.float64))
    if t.size < cfg.li_min_spikes_cell:
        return []
    isis_ms = np.diff(t) * 1000.0
    thr_ms = _logisi_threshold_ms(isis_ms, cfg)
    thr_s = thr_ms / 1000.0

    short = np.diff(t) <= thr_s          # short[j] True if gap t[j]->t[j+1] intra-burst
    bursts: List[Tuple[float, float]] = []
    j = 0
    n_gaps = short.size
    while j < n_gaps:
        if not short[j]:
            j += 1
            continue
        k = j
        while k < n_gaps and short[k]:
            k += 1
        # spikes t[j .. k] inclusive form a run; count = (k - j + 1)
        if (k - j + 1) >= cfg.li_min_spikes_cell:
            bursts.append((float(t[j]), float(t[k])))
        j = k + 1
    return bursts


def detect_bursts_participation(spk_t: np.ndarray, spk_i: np.ndarray,
                                n_neurons: int, t_rec: float,
                                cfg: BurstConfig
                                ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """Network bursts where >= participation_frac of ACTIVE neurons are bursting.

    Returns (intervals, participation_grid) where participation_grid[k] is the
    fraction of active neurons inside a single-cell burst during grid bin k.
    """
    if spk_t.size == 0 or t_rec <= 0:
        return [], np.zeros(0)
    dt = cfg.grid_dt_s
    n_bins = max(int(np.ceil(t_rec / dt)), 1)
    count = np.zeros(n_bins, dtype=np.int32)

    active = np.unique(spk_i)
    n_active = int(active.size)
    if n_active == 0:
        return [], np.zeros(n_bins)

    for nid in active:
        tn = spk_t[spk_i == nid]
        for (b0, b1) in single_cell_bursts(tn, cfg):
            k0 = int(np.floor(b0 / dt))
            k1 = int(np.floor(b1 / dt))
            k0 = max(k0, 0)
            k1 = min(k1, n_bins - 1)
            if k1 >= k0:
                count[k0:k1 + 1] += 1

    participation = count.astype(np.float64) / float(n_active)
    mask = participation >= cfg.li_participation_frac
    intervals = _merge_intervals(_intervals_from_mask(mask, dt), cfg.li_merge_gap_s)
    intervals = [(s, e) for (s, e) in intervals if (e - s) >= cfg.li_min_dur_s]
    return intervals, participation


# ===========================================================================
# Metrics conditional on a burst set
# ===========================================================================
def metrics_from_bursts(intervals: List[Tuple[float, float]],
                        spk_t: np.ndarray, spk_i: np.ndarray,
                        n_neurons: int, t_rec: float,
                        prefix: str) -> Dict[str, float]:
    """Scalar metrics conditional on a given detected burst set B.

    prefix is prepended to every key (e.g. "pr_" / "li_") so the two detectors'
    panels coexist in one flat row.
    """
    spk_t = np.asarray(spk_t, dtype=np.float64)
    spk_i = np.asarray(spk_i)
    K = spk_t.size
    n_bursts = len(intervals)
    out: Dict[str, float] = {}
    out[prefix + "n_bursts"] = float(n_bursts)
    out[prefix + "burst_rate_per_min"] = (
        n_bursts / (t_rec / 60.0) if t_rec > 0 else float("nan"))

    if n_bursts == 0:
        for key in ("burst_dur_mean_s", "burst_dur_cv", "nibi_mean_s",
                    "nibi_cv", "spikes_per_burst_mean", "intra_burst_rate_hz",
                    "participation_mean"):
            out[prefix + key] = float("nan")
        out[prefix + "frac_spikes_outside_burst"] = (1.0 if K > 0 else float("nan"))
        return out

    starts = np.array([s for s, _ in intervals])
    stops = np.array([e for _, e in intervals])
    durs = stops - starts
    out[prefix + "burst_dur_mean_s"] = float(durs.mean())
    out[prefix + "burst_dur_cv"] = _cv(durs)

    # IBI = silent gap between consecutive bursts (onset_{b+1} - offset_b)
    if n_bursts >= 2:
        nibi = starts[1:] - stops[:-1]
        out[prefix + "nibi_mean_s"] = float(nibi.mean())
        out[prefix + "nibi_cv"] = _cv(nibi)
    else:
        out[prefix + "nibi_mean_s"] = float("nan")
        out[prefix + "nibi_cv"] = float("nan")

    # spike-resolved quantities
    order = np.argsort(spk_t)
    ts = spk_t[order]
    isort = spk_i[order]
    spikes_in = 0
    part_fracs = []
    n_active = max(int(np.unique(spk_i).size), 1)
    for s, e in intervals:
        lo = np.searchsorted(ts, s, side="left")
        hi = np.searchsorted(ts, e, side="right")
        spikes_in += (hi - lo)
        if hi > lo:
            part_fracs.append(np.unique(isort[lo:hi]).size / n_active)
        else:
            part_fracs.append(0.0)
    spikes_in = int(spikes_in)
    out[prefix + "spikes_per_burst_mean"] = float(spikes_in / n_bursts)
    total_burst_time = float(durs.sum())
    out[prefix + "intra_burst_rate_hz"] = (
        float(spikes_in / (total_burst_time * n_neurons))
        if total_burst_time > 0 and n_neurons > 0 else float("nan"))
    out[prefix + "frac_spikes_outside_burst"] = (
        float((K - spikes_in) / K) if K > 0 else float("nan"))
    out[prefix + "participation_mean"] = float(np.mean(part_fracs))
    return out


# ===========================================================================
# Detector-independent descriptors
# ===========================================================================
def single_cell_irregularity(spk_t: np.ndarray, spk_i: np.ndarray,
                             n_neurons: int) -> Dict[str, float]:
    """Population means of CV_ISI and the Gorski adaptation index A.

    Only neurons with >= 2 ISIs (>= 3 spikes) contribute. Returns NaN means if
    no neuron qualifies.
    """
    cv_list: List[float] = []
    a_list: List[float] = []
    if spk_t.size:
        for nid in np.unique(spk_i):
            t = np.sort(spk_t[spk_i == nid])
            if t.size < 3:
                continue
            isi = np.diff(t)
            m = isi.mean()
            if m > 0:
                cv_list.append(isi.std(ddof=1) / m if isi.size >= 2 else float("nan"))
            num = isi[1:] - isi[:-1]
            den = isi[1:] + isi[:-1]
            good = den > 0
            if good.any():
                a_list.append(float(np.mean(num[good] / den[good])))
    return {
        "cv_isi_mean": float(np.nanmean(cv_list)) if cv_list else float("nan"),
        "adapt_index_mean": float(np.nanmean(a_list)) if a_list else float("nan"),
    }


def whole_train_descriptors(spk_t: np.ndarray, spk_i: np.ndarray,
                            n_neurons: int, t_rec: float) -> Dict[str, float]:
    """across_cell_rate_cv and frac_active (fill the old empty summary columns)."""
    if n_neurons <= 0:
        return {"across_cell_rate_cv": float("nan"), "frac_active": float("nan")}
    counts = np.bincount(np.asarray(spk_i, dtype=np.int64),
                         minlength=n_neurons)[:n_neurons]
    rates = counts.astype(np.float64) / t_rec if t_rec > 0 else counts.astype(np.float64)
    n_active = int((counts > 0).sum())
    return {
        "across_cell_rate_cv": _cv(rates) if rates.size >= 2 else float("nan"),
        "frac_active": float(n_active / n_neurons),
    }


# ===========================================================================
# Cross-check
# ===========================================================================
def cross_check_jaccard(intervals_a: List[Tuple[float, float]],
                        intervals_b: List[Tuple[float, float]],
                        t_rec: float, dt: float) -> float:
    """Jaccard overlap of two in-burst time supports on a dt grid."""
    if t_rec <= 0:
        return float("nan")
    n_bins = max(int(np.ceil(t_rec / dt)), 1)
    a = np.zeros(n_bins, dtype=bool)
    b = np.zeros(n_bins, dtype=bool)
    for s, e in intervals_a:
        a[int(np.floor(s / dt)):int(np.ceil(e / dt))] = True
    for s, e in intervals_b:
        b[int(np.floor(s / dt)):int(np.ceil(e / dt))] = True
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")          # neither detector fired -> undefined overlap
    return float(np.logical_and(a, b).sum() / union)


# ===========================================================================
# One-call entry point
# ===========================================================================
def compute_all(spk_t: np.ndarray, spk_i: np.ndarray, n_neurons: int,
                t_rec: float, cfg: Optional[BurstConfig] = None
                ) -> Dict[str, float]:
    """Full burst-metric row for one simulation (both detectors + descriptors).

    Keys are prefixed "pr_" (population-rate detector) and "li_" (log-ISI /
    participation detector); detector-independent keys are unprefixed. Also
    returns the detected interval lists under the non-scalar keys
    "_pr_intervals" / "_li_intervals" so a caller (e.g. the plotter) can reuse
    them without re-detecting. Strip keys beginning with "_" before writing CSV.
    """
    cfg = cfg or BurstConfig()
    spk_t = np.asarray(spk_t, dtype=np.float64)
    spk_i = np.asarray(spk_i)

    pr = detect_bursts_poprate(spk_t, n_neurons, t_rec, cfg)
    li, _part = detect_bursts_participation(spk_t, spk_i, n_neurons, t_rec, cfg)

    row: Dict[str, float] = {}
    row.update(metrics_from_bursts(pr, spk_t, spk_i, n_neurons, t_rec, "pr_"))
    row.update(metrics_from_bursts(li, spk_t, spk_i, n_neurons, t_rec, "li_"))
    row.update(single_cell_irregularity(spk_t, spk_i, n_neurons))
    row.update(whole_train_descriptors(spk_t, spk_i, n_neurons, t_rec))
    row["cross_check_jaccard"] = cross_check_jaccard(pr, li, t_rec, cfg.grid_dt_s)

    # non-scalar payload for the plotter (not for CSV)
    row["_pr_intervals"] = pr
    row["_li_intervals"] = li
    return row


# ===========================================================================
# Literature anchor + distance
# ===========================================================================
# Mossink et al. 2019 (Nat Commun 10:4928), Fig. 2 control iNeuron networks at
# DIV 28. VALUES BELOW ARE BEST READS OFF THE PUBLISHED FIGURE AXES, NOT source
# data -- replace each (mean, sd) with the exact Source Data numbers if you have
# them. The literature distance scales by the campaign's own robust spread by
# default (lit_scale="campaign"), so these SDs only matter if you switch to
# lit_scale="literature".
#
# Metric key -> (target_mean, target_sd, summary-column it is compared against).
# The summary column is the SCORING detector's metric (chosen in BurstConfig).
MOSSINK_CONTROL_DIV28: Dict[str, Tuple[float, float]] = {
    "burst_rate_per_min":        (5.0, 2.0),
    "burst_dur_mean_s":          (0.4, 0.2),
    "nibi_mean_s":               (8.0, 4.0),
    "nibi_cv":                   (0.4, 0.15),
    "frac_spikes_outside_burst": (0.3, 0.10),
    "mean_FR_Hz":                (3.0, 1.5),
}


def scoring_keys(cfg: BurstConfig) -> Dict[str, str]:
    """Map literature metric names -> the row key produced by the chosen detector.

    mean_FR_Hz is detector-independent (it comes from the sweep sidecar / whole
    train), so it maps to itself.
    """
    p = "li_" if cfg.scoring_detector == "logisi" else "pr_"
    out = {}
    for k in MOSSINK_CONTROL_DIV28:
        out[k] = k if k == "mean_FR_Hz" else (p + k)
    return out


def literature_distance(rows: List[Dict[str, float]], cfg: BurstConfig,
                        targets: Optional[Dict[str, Tuple[float, float]]] = None,
                        scale: str = "campaign",
                        weights: Optional[Dict[str, float]] = None,
                        nan_penalty_z: float = 4.0) -> np.ndarray:
    """z-scored Euclidean distance of each row to the literature anchor.

    distance_n = sqrt( sum_j w_j * z_{n,j}^2 / sum_j w_j ),
        z_{n,j} = (x_{n,j} - mean_j) / scale_j,
    where for each metric j:
      scale="campaign"   -> scale_j = robust std of column j across all rows,
      scale="literature" -> scale_j = target_sd_j.
    A NaN metric for a run (e.g. no bursts -> no duration) is replaced by
    nan_penalty_z in z-units so non-bursting runs sink in the ranking rather
    than being silently dropped. Returns an array of length len(rows); rows
    with every scored metric NaN get +inf.
    """
    targets = targets or MOSSINK_CONTROL_DIV28
    keymap = scoring_keys(cfg)
    metric_names = list(targets.keys())
    weights = weights or {k: 1.0 for k in metric_names}

    X = np.full((len(rows), len(metric_names)), np.nan)
    for n, r in enumerate(rows):
        for j, mk in enumerate(metric_names):
            X[n, j] = r.get(keymap[mk], np.nan)

    scales = np.zeros(len(metric_names))
    means = np.array([targets[mk][0] for mk in metric_names])
    for j, mk in enumerate(metric_names):
        if scale == "literature":
            scales[j] = targets[mk][1]
        else:
            col = X[:, j]
            col = col[np.isfinite(col)]
            s = _robust_std(col) if col.size else 0.0
            # Floor the campaign scale at the literature SD: a metric that is
            # (near-)constant across the campaign has a tiny/zero robust spread,
            # which would otherwise explode its z-score and let it dominate the
            # distance. Campaign scale may only WIDEN beyond the literature SD,
            # never collapse below it.
            lit_sd = targets[mk][1] or 1.0
            scales[j] = max(s, lit_sd)

    w = np.array([weights.get(mk, 1.0) for mk in metric_names])
    wsum = w.sum() if w.sum() > 0 else 1.0

    dist = np.full(len(rows), np.inf)
    for n in range(len(rows)):
        z = (X[n] - means) / scales
        finite = np.isfinite(z)
        if not finite.any():
            continue
        z = np.where(finite, z, nan_penalty_z)
        dist[n] = float(np.sqrt(np.sum(w * z ** 2) / wsum))
    return dist


# scalar-only view of a compute_all row, for CSV writing
def scalar_row(row: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in row.items() if not k.startswith("_")}
