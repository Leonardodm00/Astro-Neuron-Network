#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
campaign_report.py  (astrocyte campaign, 37-D registry)
===================================================================
Campaign-level diagnostic harvester for the RS-frozen neuron + astrocyte SBI
sweep (sweep_group='synapse_astro': 21 free axes = 12 synaptic + 9 astro incl
O_N; neurons frozen at the Gorski 2021 RS set; EC50_ampa/EC50_nmda and the two
GJ-shape axes I_Theta/omega_I frozen at nominal).

Target a campaign directory; walk every topology and every iter_*.npz; extract
DYNAMICAL (neuron firing statistics + regimes; astrocyte Ca2+ event statistics,
regimes, neuron<->astro coupling, and GJ-wave/timing proxies), TOPOLOGICAL
(neuron degree distribution + astrocyte GJ network), and IDENTIFIABILITY
(Spearman of every SWEPT axis against BOTH log firing rate and log astro event
rate) properties; emit:

    <out>/campaign_report.md      verbose human + LLM status report
    <out>/campaign_summary.json   machine-readable roll-up
    <out>/figures/*.png           neuron + astro + predictivity figures

WHAT CHANGED vs the pre-astrocyte (35-D Doorn) report
-----------------------------------------------------
* Registry is 37-D: Cm (idx 35) and O_N (idx 36) added.  'params'/'theta' are
  now length-37; the loader accepts >=35 but uses 37-aware indices.
* The flagship "does sigma / I_inj predict rate" analysis is REMOVED: those
  neuron axes are FROZEN now, so they carry zero variance.  It is replaced by a
  generic SWEPT-AXIS predictivity table (axes auto-detected from the data) vs
  BOTH firing rate and astrocyte event rate -- the SBI identifiability readout
  for this campaign.
* Membrane-time-constant facts corrected: RS freeze gives tau_m = Cm/gL =
  200 pF / 10 nS = 20 ms (NOT the old ~3.3 ms Doorn membrane).  The old
  "fast membrane suppresses bursts" verdict is inverted: a 20 ms membrane
  summates, so bursting is plausible.
* GAP_MV = V_T - E_L - DeltaT = -50 - (-60) - 2 = 8 mV (was 5.0).
* Heavy astrocyte block added: event-rate regimes, population synchrony,
  neuron<->astro cross-correlation lag, and a GJ-neighbour co-activation
  "wave enrichment" proxy.

ON-DISK CONTRACT (from HPC_main_sweep.py / HPC_single_run.py)
------------------------------------------------------------
campaign_<TAG>/
  sweep_<part>_task<NNNN>/
    topo_<NNNNN>/
      topology.npz            N_pos(Nn,2) S_i S_j(n_syn) A_pos(Na,2)
                              GJ_i GJ_j StoA_i StoA_j  [S_x_syn S_y_syn]
      topology_meta.json      {Nn, Na, simtime_s, mode, topo_idx, conn_rule,
                               p0_conn, d0_conn, beta_conn, c_max, ...}
      iter_<NNNNN>.npz        spk_N_t spk_N_i spk_A_t spk_A_i
                              params(37,) theta(37,)
      iter_<NNNNN>.json       sidecar stats (mean_FR, across_cell_rate_cv,
                               mean_astro_rate_Hz, ...)

Run
---
    python campaign_report.py /path/to/campaign_<TAG>
    python campaign_report.py campaign_<TAG> --out ./report --topo-sample 50
    python campaign_report.py --smoke-test                 # synthetic self-check

Notes
-----
* Pure numpy / scipy / matplotlib; no Brian2 needed (reads results only).
* Streams iter files (never holds all spike arrays at once); safe on 100k+ sims.
* Degenerate / missing files are skipped and COUNTED, never fatal.
===================================================================
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parameter-vector index map -- 37-D registry (must match HPC_single_run.py)
# ----------------------------------------------------------------------
PARAM_NAMES = [
    'Sigma', 'gbarA', 'EC50_ampa', 'EC50_nmda', 'tauA',            # 0-4
    'U_0_ar', 'U_max', 'U_0_sr', 'Omega_f_sr', 'Omega_f_ar', 'Omega_d',  # 5-10
    'alpha_syn', 'DeltaT', 'VT', 'g_ampa', 'g_nmda', 'delta_gA', 'x0',   # 11-17
    'O_G', 'Omega_G', 'O_beta', 'O_3K', 'Omega_5P', 'I_bias', 'F',       # 18-24
    'I_Theta', 'omega_I', 'C_Theta', 'U_A', 'G_T', 'gL', 'VA', 'DeltaA', # 25-32
    'VR', 'I_inj', 'Cm', 'O_N',                                          # 33-36
]
IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
N_DIMS = len(PARAM_NAMES)                       # 37
MIN_PARAMS = 35                                 # accept legacy 35-D vectors too

# Frozen-RS membrane facts (single source of truth for the verdict text).
TAU_M_MS = 200.0 / 10.0                         # Cm/gL = 20 ms (RS freeze)
GAP_MV   = -50.0 - (-60.0) - 2.0                # V_T - E_L - DeltaT = 8 mV

# Reference grouping of the 21 axes the campaign is DESIGNED to sweep. The actual
# swept set is auto-detected from the data (detect_swept_axes); these are only
# used for figure grouping / labelling and to flag config drift.
SYNAPTIC_SWEPT_REF = ['U_0_ar', 'U_max', 'U_0_sr', 'Omega_f_sr', 'Omega_f_ar',
                      'Omega_d', 'alpha_syn', 'g_ampa', 'g_nmda', 'x0',
                      'O_G', 'Omega_G']
ASTRO_SWEPT_REF    = ['O_beta', 'O_3K', 'Omega_5P', 'I_bias', 'F',
                      'C_Theta', 'U_A', 'G_T', 'O_N']
EXPECTED_SWEPT_REF = SYNAPTIC_SWEPT_REF + ASTRO_SWEPT_REF        # 21

# Neuron firing-rate regime thresholds [Hz]; ONE_SPIKE flagged separately.
R_SILENT = 0.005
R_LOW    = 0.1
R_SEIZE  = 5.0

# Astrocyte Ca2+/exocytosis event-rate regime thresholds [Hz per astrocyte].
# Astro events are SLOW (seconds-scale), so per-cell rates run ~2 orders of
# magnitude below neuronal rates. TUNABLE; defaults bracket the De Pitta/Wallach
# operating range (~0.01-0.1 Hz physiological Ca2+ event rate).
A_SILENT = 0.002
A_LOW    = 0.02
A_HIGH   = 0.2

# Active-mask floors used when correlating a driver against an output (so silent
# sims don't dominate a log-rate Spearman with a floor value).
FR_ACTIVE_FLOOR    = 0.01     # Hz
ASTRO_ACTIVE_FLOOR = 0.002    # Hz


# ======================================================================
# Data containers
# ======================================================================
@dataclass
class IterRecord:
    topo_idx: int
    iter_idx: int
    # neuron dynamics
    n_spikes: int
    mean_fr: float
    frac_active: float
    across_cell_rate_cv: float
    mean_isi_cv: float
    peak_pop_rate: float
    peak_over_mean: float
    burst_proxy: int
    one_spike_artifact: bool
    regime: str
    # astrocyte dynamics
    n_astro_events: int
    astro_rate_hz: float            # per-astrocyte event rate
    astro_frac_active: float
    astro_across_cv: float
    astro_pop_peak: float
    astro_sync_peaks: int
    astro_regime: str
    wave_enrichment: float          # GJ-neighbour co-activation / random-pair
    na_lag_s: float                 # astro-minus-neuron pop-rate peak lag [s]
    na_xcorr: float                 # peak neuron<->astro cross-correlation
    # full parameter vector (NATURAL units) -- used for predictivity; never JSON'd
    params: np.ndarray = field(default=None, repr=False)


@dataclass
class TopoRecord:
    topo_idx: int
    task: str
    Nn: int
    Na: int
    T: float
    conn_rule: str
    p0_conn: float
    d0_conn: float
    beta_conn: float
    c_max: float
    n_syn: int
    mean_in_degree: float
    mean_out_degree: float
    median_in_degree: float
    max_in_degree: int
    frac_zero_in: float
    n_components: int
    largest_component_frac: float
    mean_conn_distance: float
    reciprocity: float
    # astrocyte GJ network
    n_gj: int
    mean_gj_degree: float
    gj_largest_component_frac: float
    n_iters: int


# ======================================================================
# Loading helpers (defensive)
# ======================================================================
def _safe_load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    except Exception:
        return None


def _read_json(path: Path) -> Dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def _discover_topologies(campaign: Path) -> List[Path]:
    topos = sorted(campaign.glob('sweep_*_task*/topo_*'))
    if not topos:
        topos = sorted(campaign.glob('**/topo_*'))
    return [t for t in topos if t.is_dir()]


# ======================================================================
# Shared signal helpers
# ======================================================================
def _population_rate(spk_t: np.ndarray, N: int, T: float,
                     bin_s: float, smooth_bins: int = 3) -> np.ndarray:
    if spk_t.size == 0 or T <= 0 or N <= 0:
        return np.zeros(1)
    n_bins = max(1, int(T / bin_s))
    counts, _ = np.histogram(spk_t, bins=n_bins, range=(0.0, T))
    rate = counts / (N * bin_s)
    if smooth_bins > 1:
        k = np.ones(smooth_bins) / smooth_bins
        rate = np.convolve(rate, k, mode='same')
    return rate


def _per_unit_rates(spk_i: np.ndarray, N: int, T: float) -> np.ndarray:
    if spk_i.size == 0 or N <= 0:
        return np.zeros(max(N, 0))
    counts = np.bincount(spk_i, minlength=N)[:N]
    return counts / T


def _sync_peaks(pop_rate: np.ndarray, min_amp_ratio: float = 2.5,
                prom_frac: float = 0.4) -> Tuple[int, float, float]:
    """Population-rate peaks exceeding both an absolute ratio (peak/mean) and a
    relative prominence. Returns (n_peaks, peak_rate, peak_over_mean)."""
    if pop_rate.size < 3:
        return 0, float(pop_rate.max() if pop_rate.size else 0.0), 0.0
    mean_r = pop_rate.mean()
    peak_r = pop_rate.max()
    if mean_r <= 0:
        return 0, float(peak_r), 0.0
    ratio = peak_r / mean_r
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            pop_rate, height=max(min_amp_ratio * mean_r, 1e-12),
            prominence=prom_frac * peak_r,
            distance=max(1, int(len(pop_rate) * 0.01)))
        n = int(peaks.size)
    except Exception:
        n = int(ratio >= min_amp_ratio)
    return n, float(peak_r), float(ratio)


def _isi_cv_mean(spk_t: np.ndarray, spk_i: np.ndarray, N: int,
                 min_spikes: int = 4) -> float:
    if spk_t.size == 0 or N <= 0:
        return 0.0
    order = np.argsort(spk_i, kind='stable')
    si, st = spk_i[order], spk_t[order]
    bounds = np.searchsorted(si, np.arange(N + 1))
    cvs = []
    for k in range(N):
        a, b = bounds[k], bounds[k + 1]
        if b - a >= min_spikes:
            isi = np.diff(np.sort(st[a:b]))
            m = isi.mean()
            if m > 0:
                cvs.append(isi.std() / m)
    return float(np.mean(cvs)) if cvs else 0.0


# ======================================================================
# Neuron dynamical feature extraction
# ======================================================================
def _classify_neuron(mean_fr: float, one_spike: bool) -> str:
    if one_spike:
        return 'one_spike_artifact'
    if mean_fr < R_SILENT:
        return 'silent'
    if mean_fr < R_LOW:
        return 'low'
    if mean_fr < R_SEIZE:
        return 'useful'
    return 'seizing'


def _classify_astro(rate: float) -> str:
    if rate < A_SILENT:
        return 'silent'
    if rate < A_LOW:
        return 'sparse'
    if rate < A_HIGH:
        return 'active'
    return 'hyperactive'


# ======================================================================
# Astrocyte dynamics: coupling + GJ-wave proxies
# ======================================================================
def _min_gap(ta: np.ndarray, tb: np.ndarray) -> float:
    """Smallest |t_a - t_b| over the two sorted event-time arrays."""
    if ta.size == 0 or tb.size == 0:
        return np.inf
    idx = np.searchsorted(tb, ta)
    best = np.inf
    for t, i in zip(ta, idx):
        if i < tb.size:
            best = min(best, tb[i] - t)
        if i > 0:
            best = min(best, t - tb[i - 1])
        if best <= 0:
            return 0.0
    return float(best)


def _wave_enrichment(spk_A_t: np.ndarray, spk_A_i: np.ndarray,
                     gj_i: np.ndarray, gj_j: np.ndarray, Na: int,
                     dt: float = 2.0, max_events_per_cell: int = 400,
                     rng: Optional[np.random.Generator] = None) -> float:
    """GJ-wave proxy: enrichment of co-activation among GJ-connected astrocyte
    pairs relative to random pairs.

    co-active(a, b)  ==  exists an event in a within +/- dt of an event in b.
    enrichment = P(co-active | GJ edge) / P(co-active | random non-edge).
    >1 indicates Ca2+ activity preferentially co-occurs along the GJ graph
    (the signature of gap-junction-mediated propagation); ~1 indicates
    spatially independent activation. NaN if undefined (no events/edges).
    """
    if spk_A_t.size == 0 or gj_i.size == 0 or Na <= 1:
        return float('nan')
    rng = rng or np.random.default_rng(0)
    # per-astro sorted, capped event-time arrays
    order = np.argsort(spk_A_i, kind='stable')
    si, st = spk_A_i[order], spk_A_t[order]
    bounds = np.searchsorted(si, np.arange(Na + 1))
    times = []
    for a in range(Na):
        seg = np.sort(st[bounds[a]:bounds[a + 1]])
        if seg.size > max_events_per_cell:
            seg = seg[:max_events_per_cell]
        times.append(seg)

    def co(a, b):
        return _min_gap(times[a], times[b]) <= dt

    gj_co = np.mean([co(int(a), int(b)) for a, b in zip(gj_i, gj_j)])

    n_null = min(max(gj_i.size * 3, 500), 5000)
    cnt = tot = 0
    while tot < n_null:
        a, b = int(rng.integers(0, Na)), int(rng.integers(0, Na))
        if a == b:
            continue
        tot += 1
        cnt += int(co(a, b))
    rand_co = cnt / max(tot, 1)
    return float(gj_co / max(rand_co, 1e-3))


def _neuron_astro_lag(spk_N_t: np.ndarray, Nn: int,
                      spk_A_t: np.ndarray, Na: int, T: float,
                      bin_s: float = 1.0, max_lag_s: float = 25.0
                      ) -> Tuple[float, float]:
    """Cross-correlate neuron and astro population rates; return (lag_s, xcorr).
    lag_s > 0 means the astro population rate peaks AFTER the neuron rate (the
    expected glutamate -> Ca2+ delay). NaN lag if either signal is flat."""
    if spk_N_t.size == 0 or spk_A_t.size == 0 or T <= 0:
        return float('nan'), 0.0
    n = _population_rate(spk_N_t, Nn, T, bin_s, smooth_bins=1)
    a = _population_rate(spk_A_t, Na, T, bin_s, smooth_bins=1)
    L = min(n.size, a.size)
    if L < 3:
        return float('nan'), 0.0
    n, a = n[:L], a[:L]
    if n.std() < 1e-12 or a.std() < 1e-12:
        return float('nan'), 0.0
    n = (n - n.mean()) / n.std()
    a = (a - a.mean()) / a.std()
    full = np.correlate(a, n, mode='full') / L          # a shifted vs n
    lags = np.arange(-L + 1, L) * bin_s
    sel = np.abs(lags) <= max_lag_s
    full, lags = full[sel], lags[sel]
    k = int(np.argmax(full))
    return float(lags[k]), float(full[k])


def _analyze_astro(npz: Dict[str, np.ndarray], sidecar: Dict, Na: int, T: float,
                   gj_i: np.ndarray, gj_j: np.ndarray,
                   spk_N_t: np.ndarray, Nn: int) -> Dict:
    spk_A_t = np.asarray(npz.get('spk_A_t', np.empty(0)), dtype=np.float64)
    spk_A_i = np.asarray(npz.get('spk_A_i', np.empty(0)), dtype=np.int64)
    n_ev = int(spk_A_t.size)

    if Na <= 0:
        Na = int(sidecar.get('Na', 0) or 0)
    if Na <= 0 and spk_A_i.size:
        Na = int(spk_A_i.max()) + 1

    rate_hz = n_ev / (Na * T) if (Na > 0 and T > 0) else 0.0
    rates = _per_unit_rates(spk_A_i, Na, T) if Na > 0 else np.zeros(0)
    frac_active = float((rates > 0).mean()) if Na > 0 else 0.0
    mu = rates[rates > 0].mean() if np.any(rates > 0) else 0.0
    across_cv = float(rates[rates > 0].std() / mu) if mu > 0 else 0.0

    pr = _population_rate(spk_A_t, Na, T, bin_s=0.5)
    n_pk, peak_r, _ = _sync_peaks(pr)

    wave = _wave_enrichment(spk_A_t, spk_A_i, gj_i, gj_j, Na)
    na_lag, na_xcorr = _neuron_astro_lag(spk_N_t, Nn, spk_A_t, Na, T)

    return dict(
        n_astro_events=n_ev, astro_rate_hz=rate_hz, astro_frac_active=frac_active,
        astro_across_cv=across_cv, astro_pop_peak=peak_r, astro_sync_peaks=n_pk,
        astro_regime=_classify_astro(rate_hz), wave_enrichment=wave,
        na_lag_s=na_lag, na_xcorr=na_xcorr,
    )


# ======================================================================
# Per-iter assembly
# ======================================================================
def _analyze_iter(npz: Dict[str, np.ndarray], sidecar: Dict,
                  Nn: int, Na: int, T: float, topo_idx: int, iter_idx: int,
                  gj_i: np.ndarray, gj_j: np.ndarray) -> Optional[IterRecord]:
    if 'params' not in npz:
        return None
    params = np.asarray(npz['params'], dtype=np.float64)
    if params.shape[0] < MIN_PARAMS:
        return None
    # pad legacy 35-D vectors to 37 with NaN so index math never overflows
    if params.shape[0] < N_DIMS:
        params = np.concatenate([params, np.full(N_DIMS - params.shape[0], np.nan)])

    spk_t = np.asarray(npz.get('spk_N_t', np.empty(0)), dtype=np.float64)
    spk_i = np.asarray(npz.get('spk_N_i', np.empty(0)), dtype=np.int64)
    n_spk = int(spk_t.size)

    if not (Nn > 0):
        Nn = int(sidecar.get('Nn', 0) or 0)
    if not (Nn > 0) and spk_i.size:
        Nn = int(spk_i.max()) + 1
    if not (Nn > 0):
        return None
    if not (T > 0):
        T = float(sidecar.get('simtime_s', sidecar.get('T', 0.0)) or 0.0)
    if not (T > 0) and spk_t.size:
        T = float(spk_t.max())
    if not (T > 0):
        T = 1.0

    mean_fr = n_spk / (Nn * T)
    one_spike = (Nn > 0 and abs(n_spk - Nn) <= max(2, 0.02 * Nn)
                 and (spk_t.max() < 0.5 if spk_t.size else False))

    rates = _per_unit_rates(spk_i, Nn, T)
    frac_active = float((rates > 0).mean()) if Nn else 0.0
    mu_r = rates[rates > 0].mean() if np.any(rates > 0) else 0.0
    across_cv = float(rates[rates > 0].std() / mu_r) if mu_r > 0 else 0.0
    isi_cv = _isi_cv_mean(spk_t, spk_i, Nn)
    pr = _population_rate(spk_t, Nn, T, bin_s=0.005)
    n_pk, peak_r, peak_over_mean = _sync_peaks(pr)

    # sidecar values authoritative where present
    frac_active = float(sidecar.get('frac_active', frac_active))
    across_cv = float(sidecar.get('across_cell_rate_cv', across_cv))
    isi_cv = float(sidecar.get('mean_isi_cv', isi_cv))

    astro = _analyze_astro(npz, sidecar, Na, T, gj_i, gj_j, spk_t, Nn)

    return IterRecord(
        topo_idx=topo_idx, iter_idx=iter_idx, n_spikes=n_spk, mean_fr=mean_fr,
        frac_active=frac_active, across_cell_rate_cv=across_cv,
        mean_isi_cv=isi_cv, peak_pop_rate=peak_r, peak_over_mean=peak_over_mean,
        burst_proxy=n_pk, one_spike_artifact=one_spike,
        regime=_classify_neuron(mean_fr, one_spike),
        params=params.astype(np.float32),
        **astro,
    )


# ======================================================================
# Topology feature extraction
# ======================================================================
def _degree_stats(S_i, S_j, Nn) -> Dict:
    out_deg = np.bincount(S_i, minlength=Nn)[:Nn] if S_i.size else np.zeros(Nn, int)
    in_deg  = np.bincount(S_j, minlength=Nn)[:Nn] if S_j.size else np.zeros(Nn, int)
    return dict(in_deg=in_deg, out_deg=out_deg,
                mean_in=float(in_deg.mean()) if Nn else 0.0,
                mean_out=float(out_deg.mean()) if Nn else 0.0,
                median_in=float(np.median(in_deg)) if Nn else 0.0,
                max_in=int(in_deg.max()) if Nn else 0,
                frac_zero_in=float((in_deg == 0).mean()) if Nn else 1.0)


def _reciprocity(S_i, S_j) -> float:
    if S_i.size == 0:
        return 0.0
    edges = set(zip(S_i.tolist(), S_j.tolist()))
    return sum(1 for (a, b) in edges if (b, a) in edges) / len(edges)


def _weak_components(S_i, S_j, N) -> Tuple[int, float]:
    if N == 0 or S_i.size == 0:
        return (0, 0.0)
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        A = coo_matrix((np.ones(S_i.size, np.int8), (S_i, S_j)), shape=(N, N))
        n_comp, labels = connected_components(A, directed=True, connection='weak')
        _, counts = np.unique(labels, return_counts=True)
        return int(n_comp), float(counts.max() / N)
    except Exception:
        return -1, 0.0


def _mean_conn_distance(N_pos, S_i, S_j, max_edges=200_000) -> float:
    if S_i.size == 0 or N_pos.size == 0:
        return 0.0
    if S_i.size > max_edges:
        sel = np.random.default_rng(0).choice(S_i.size, max_edges, replace=False)
        S_i, S_j = S_i[sel], S_j[sel]
    d = N_pos[S_i] - N_pos[S_j]
    return float(np.sqrt((d * d).sum(axis=1)).mean())


def _analyze_topology(topo_dir: Path, meta: Dict):
    npz = _safe_load_npz(topo_dir / 'topology.npz')
    if npz is None or 'S_i' not in npz:
        return None
    Nn = int(meta.get('Nn', npz['N_pos'].shape[0] if 'N_pos' in npz else 0))
    Na = int(meta.get('Na', npz['A_pos'].shape[0] if 'A_pos' in npz else 0))
    S_i = np.asarray(npz['S_i'], dtype=np.int64)
    S_j = np.asarray(npz['S_j'], dtype=np.int64)
    N_pos = np.asarray(npz.get('N_pos', np.empty((Nn, 2))), dtype=np.float64)
    c_max = float(meta.get('c_max', N_pos.max() if N_pos.size else 0.0))
    gj_i = np.asarray(npz.get('GJ_i', np.empty(0)), dtype=np.int64)
    gj_j = np.asarray(npz.get('GJ_j', np.empty(0)), dtype=np.int64)

    dg = _degree_stats(S_i, S_j, Nn)
    n_comp, big = _weak_components(S_i, S_j, Nn)
    # astrocyte GJ network (undirected: symmetrise degree)
    if Na > 0 and gj_i.size:
        gj_deg = (np.bincount(gj_i, minlength=Na)[:Na]
                  + np.bincount(gj_j, minlength=Na)[:Na])
        mean_gj_deg = float(gj_deg.mean())
        _, gj_big = _weak_components(gj_i, gj_j, Na)
    else:
        mean_gj_deg, gj_big = 0.0, 0.0

    rec = TopoRecord(
        topo_idx=int(meta.get('topo_idx', -1)), task=topo_dir.parent.name,
        Nn=Nn, Na=Na,
        T=float(meta.get('simtime_s', meta.get('T', 0.0)) or 0.0),
        conn_rule=str(meta.get('conn_rule', '?')),
        p0_conn=float(meta.get('p0_conn', np.nan) or np.nan),
        d0_conn=float(meta.get('d0_conn', np.nan) or np.nan),
        beta_conn=float(meta.get('beta_conn', np.nan) or np.nan),
        c_max=c_max, n_syn=int(S_i.size),
        mean_in_degree=dg['mean_in'], mean_out_degree=dg['mean_out'],
        median_in_degree=dg['median_in'], max_in_degree=dg['max_in'],
        frac_zero_in=dg['frac_zero_in'], n_components=n_comp,
        largest_component_frac=big,
        mean_conn_distance=_mean_conn_distance(N_pos, S_i, S_j),
        reciprocity=_reciprocity(S_i, S_j),
        n_gj=int(gj_i.size), mean_gj_degree=mean_gj_deg,
        gj_largest_component_frac=gj_big, n_iters=0,
    )
    extra = dict(in_deg=dg['in_deg'], N_pos=N_pos, S_i=S_i, S_j=S_j,
                 A_pos=np.asarray(npz.get('A_pos', np.empty((Na, 2))), float),
                 gj_i=gj_i, gj_j=gj_j, c_max=c_max)
    return rec, extra, gj_i, gj_j


# ======================================================================
# Swept-axis detection + predictivity (the SBI identifiability readout)
# ======================================================================
def detect_swept_axes(param_matrix: np.ndarray, rtol: float = 1e-6) -> List[int]:
    """An axis is 'swept' if it varies across sims. Robust to the campaign config
    (auto-discovers the active set rather than hardcoding it). Uses range > a
    small fraction of the median magnitude, ignoring NaN-padded legacy columns."""
    swept = []
    for i in range(param_matrix.shape[1]):
        col = param_matrix[:, i]
        col = col[np.isfinite(col)]
        if col.size < 2:
            continue
        scale = max(abs(np.median(col)), 1e-12)
        if (col.max() - col.min()) > rtol * scale:
            swept.append(i)
    return swept


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 5 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float('nan')
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y)[0])
    except Exception:
        # rank-correlation fallback
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])


def compute_predictivity(iters: List[IterRecord]) -> Dict:
    """Spearman of every SWEPT axis against BOTH log firing rate and log astro
    event rate (each over its own active subset). This is the per-axis
    identifiability signal the SBI posterior will or won't have."""
    if not iters:
        return dict(swept_axes=[], table=[])
    P = np.vstack([r.params for r in iters]).astype(np.float64)   # (n, 37)
    fr = np.array([r.mean_fr for r in iters])
    ar = np.array([r.astro_rate_hz for r in iters])
    swept = detect_swept_axes(P)

    fr_act = fr > FR_ACTIVE_FLOOR
    ar_act = ar > ASTRO_ACTIVE_FLOOR
    log_fr = np.log(np.where(fr_act, fr, np.nan))
    log_ar = np.log(np.where(ar_act, ar, np.nan))

    table = []
    for i in swept:
        x = P[:, i]
        m_fr = fr_act & np.isfinite(x)
        m_ar = ar_act & np.isfinite(x)
        rho_fr = _spearman(x[m_fr], log_fr[m_fr]) if m_fr.sum() >= 5 else float('nan')
        rho_ar = _spearman(x[m_ar], log_ar[m_ar]) if m_ar.sum() >= 5 else float('nan')
        grp = ('astro' if PARAM_NAMES[i] in ASTRO_SWEPT_REF else
               'synaptic' if PARAM_NAMES[i] in SYNAPTIC_SWEPT_REF else 'other')
        table.append(dict(idx=i, name=PARAM_NAMES[i], group=grp,
                          rho_fr=rho_fr, rho_astro=rho_ar,
                          n_fr=int(m_fr.sum()), n_astro=int(m_ar.sum())))
    # config-drift check vs the expected 21
    swept_names = {PARAM_NAMES[i] for i in swept}
    unexpected = sorted(swept_names - set(EXPECTED_SWEPT_REF))
    missing    = sorted(set(EXPECTED_SWEPT_REF) - swept_names)
    return dict(swept_axes=[PARAM_NAMES[i] for i in swept], table=table,
                unexpected_swept=unexpected, missing_swept=missing)


# ======================================================================
# Roll-up
# ======================================================================
def _frac(vals, pred) -> float:
    return float(np.mean([pred(v) for v in vals])) if vals else 0.0


def _roll_up(iters: List[IterRecord], topos: List[TopoRecord],
             predict: Dict) -> Dict:
    n = max(1, len(iters))
    fr  = np.array([r.mean_fr for r in iters]) if iters else np.zeros(0)
    ar  = np.array([r.astro_rate_hz for r in iters]) if iters else np.zeros(0)
    nreg = [r.regime for r in iters]
    areg = [r.astro_regime for r in iters]
    waves = np.array([r.wave_enrichment for r in iters if np.isfinite(r.wave_enrichment)])
    lags  = np.array([r.na_lag_s for r in iters if np.isfinite(r.na_lag_s)])

    # neuron<->astro coupling across sims (active in both)
    both = (fr > FR_ACTIVE_FLOOR) & (ar > ASTRO_ACTIVE_FLOOR)
    na_coupling = (_spearman(np.log(fr[both]), np.log(ar[both]))
                   if both.sum() >= 5 else float('nan'))

    roll = dict(
        n_iters=len(iters), n_topologies=len(topos),
        neuron_regime_fractions={r: nreg.count(r) / n for r in
                                 ['one_spike_artifact', 'silent', 'low',
                                  'useful', 'seizing']},
        astro_regime_fractions={r: areg.count(r) / n for r in
                                ['silent', 'sparse', 'active', 'hyperactive']},
        one_spike_artifact_frac=_frac([r.one_spike_artifact for r in iters], bool),
        fr_median=float(np.median(fr)) if fr.size else 0.0,
        fr_mean=float(fr.mean()) if fr.size else 0.0,
        fr_max=float(fr.max()) if fr.size else 0.0,
        astro_rate_median=float(np.median(ar)) if ar.size else 0.0,
        astro_rate_mean=float(ar.mean()) if ar.size else 0.0,
        astro_rate_max=float(ar.max()) if ar.size else 0.0,
        sims_with_burst_proxy=sum(1 for r in iters if r.burst_proxy > 0),
        sims_with_astro_sync=sum(1 for r in iters if r.astro_sync_peaks > 0),
        na_coupling_rho=na_coupling,
        na_lag_median_s=float(np.median(lags)) if lags.size else float('nan'),
        na_lag_frac_positive=float((lags > 0).mean()) if lags.size else float('nan'),
        wave_enrichment_median=float(np.median(waves)) if waves.size else float('nan'),
        wave_frac_above_1=float((waves > 1.0).mean()) if waves.size else float('nan'),
        predictivity=predict,
    )
    if topos:
        roll['topology'] = dict(
            mean_in_degree=float(np.mean([t.mean_in_degree for t in topos])),
            median_in_degree=float(np.mean([t.median_in_degree for t in topos])),
            frac_zero_in=float(np.mean([t.frac_zero_in for t in topos])),
            mean_n_syn=float(np.mean([t.n_syn for t in topos])),
            mean_conn_distance=float(np.mean([t.mean_conn_distance for t in topos])),
            mean_largest_component_frac=float(np.mean([t.largest_component_frac for t in topos])),
            mean_reciprocity=float(np.mean([t.reciprocity for t in topos])),
            mean_gj_degree=float(np.mean([t.mean_gj_degree for t in topos])),
            mean_gj_component_frac=float(np.mean([t.gj_largest_component_frac for t in topos])),
            conn_rules=sorted({t.conn_rule for t in topos}),
        )
    return roll


# ======================================================================
# Figures
# ======================================================================
def _make_figures(iters, topos, topo_extra, predict, fig_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    if not iters:
        return made

    fr = np.array([r.mean_fr for r in iters])
    ar = np.array([r.astro_rate_hz for r in iters])

    # 1. neuron + astro rate distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, bands, lab in [
        (axes[0], fr, [(R_SILENT, 'silent'), (R_LOW, 'low'), (R_SEIZE, 'seize')],
         'neuron mean_FR (Hz)'),
        (axes[1], ar, [(A_SILENT, 'sil'), (A_LOW, 'sparse'), (A_HIGH, 'active')],
         'astro event rate (Hz)')]:
        pos = vals[vals > 0]
        if pos.size:
            ax.hist(np.log10(pos), bins=50, color='#4477aa', alpha=0.85)
        for x, t in bands:
            ax.axvline(np.log10(x), color='k', ls='--', lw=0.8)
            ax.text(np.log10(x), ax.get_ylim()[1] * 0.92, t, fontsize=7,
                    rotation=90, va='top')
        ax.set_xlabel(f'log10 {lab}'); ax.set_ylabel('# sims')
    axes[0].set_title('Neuron regime distribution')
    axes[1].set_title('Astrocyte regime distribution')
    fig.tight_layout(); p = fig_dir / 'rate_distributions.png'
    fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    # 2. predictivity: Spearman of swept axes vs FR and astro rate
    tbl = predict.get('table', [])
    if tbl:
        tbl = sorted(tbl, key=lambda d: (d['group'], d['name']))
        names = [d['name'] for d in tbl]
        rfr = [0.0 if not np.isfinite(d['rho_fr']) else d['rho_fr'] for d in tbl]
        rar = [0.0 if not np.isfinite(d['rho_astro']) else d['rho_astro'] for d in tbl]
        y = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(names))))
        ax.barh(y - 0.2, rfr, height=0.38, color='#cc6677', label='vs log FR')
        ax.barh(y + 0.2, rar, height=0.38, color='#117733', label='vs log astro rate')
        ax.axvline(0, color='k', lw=0.8)
        for xv in (-0.3, 0.3):
            ax.axvline(xv, color='gray', ls=':', lw=0.7)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Spearman rho'); ax.set_xlim(-1, 1)
        ax.set_title('Swept-axis predictivity (identifiability)')
        ax.legend(fontsize=8, loc='lower right')
        fig.tight_layout(); p = fig_dir / 'predictivity.png'
        fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    # 3. neuron<->astro coupling scatter
    both = (fr > FR_ACTIVE_FLOOR) & (ar > ASTRO_ACTIVE_FLOOR)
    fig, ax = plt.subplots(figsize=(6, 5))
    if both.sum():
        ax.scatter(fr[both], ar[both], s=8, alpha=0.45, color='#882255')
        ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('neuron mean_FR (Hz)'); ax.set_ylabel('astro event rate (Hz)')
    ax.set_title('Neuron <-> astrocyte coupling')
    fig.tight_layout(); p = fig_dir / 'neuron_astro_coupling.png'
    fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    # 4. wave enrichment + neuron->astro lag
    waves = np.array([r.wave_enrichment for r in iters if np.isfinite(r.wave_enrichment)])
    lags = np.array([r.na_lag_s for r in iters if np.isfinite(r.na_lag_s)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if waves.size:
        axes[0].hist(np.clip(waves, 0, 5), bins=40, color='#44aa99')
        axes[0].axvline(1.0, color='k', ls='--', label='no enrichment')
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel('GJ-wave enrichment'); axes[0].set_ylabel('# sims')
    axes[0].set_title('GJ-neighbour co-activation')
    if lags.size:
        axes[1].hist(lags, bins=40, color='#ddaa33')
        axes[1].axvline(0, color='k', ls='--')
    axes[1].set_xlabel('astro-minus-neuron peak lag (s)')
    axes[1].set_ylabel('# sims'); axes[1].set_title('Neuron -> astro delay')
    fig.tight_layout(); p = fig_dir / 'astro_wave_and_lag.png'
    fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    # 5. topology: neuron in-degree + astro GJ degree
    if topo_extra is not None and topo_extra.get('in_deg') is not None:
        in_deg = topo_extra['in_deg']
        gj_i, gj_j = topo_extra['gj_i'], topo_extra['gj_j']
        Na = topo_extra['A_pos'].shape[0]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(in_deg, bins=40, color='#44aa99')
        axes[0].axvline(in_deg.mean(), color='k', ls='--',
                        label=f'mean={in_deg.mean():.1f}')
        axes[0].set_xlabel('neuron in-degree'); axes[0].set_ylabel('# neurons')
        axes[0].set_title('Neuron in-degree'); axes[0].legend(fontsize=8)
        if Na > 0 and gj_i.size:
            gj_deg = (np.bincount(gj_i, minlength=Na)[:Na]
                      + np.bincount(gj_j, minlength=Na)[:Na])
            axes[1].hist(gj_deg, bins=30, color='#aa4499')
            axes[1].axvline(gj_deg.mean(), color='k', ls='--',
                            label=f'mean={gj_deg.mean():.1f}')
            axes[1].legend(fontsize=8)
        axes[1].set_xlabel('astrocyte GJ degree'); axes[1].set_ylabel('# astrocytes')
        axes[1].set_title('Astrocyte GJ degree')
        fig.suptitle('Topology (representative topology)')
        fig.tight_layout(); p = fig_dir / 'topology_degrees.png'
        fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    return made


# ======================================================================
# Verdict + markdown
# ======================================================================
def _pct(x: float) -> str:
    return f'{100 * x:.0f}%'


def _verdict_lines(roll: Dict) -> List[str]:
    L = []
    nrf = roll['neuron_regime_fractions']
    arf = roll['astro_regime_fractions']
    art = roll['one_spike_artifact_frac']

    if art > 0.1:
        L.append(f"- **STARTUP ARTIFACT**: {_pct(art)} of sims fire ~1 spike/neuron "
                 f"then go silent. With the RS freeze V(0) is uniform on "
                 f"[E_L, V_T)=[-60,-50) mV, so this should be rare; if it isn't, "
                 f"discard a 1-2 s warm-up before computing stats.")
    if nrf['useful'] < 0.1:
        L.append(f"- **LOW NEURON YIELD**: only {_pct(nrf['useful'])} of sims land in "
                 f"the useful {R_LOW}-{R_SEIZE} Hz band. Most synaptic-prior volume "
                 f"is silent or seizing; consider tightening g_ampa/g_nmda.")
    else:
        L.append(f"- Neuron useful-band yield {_pct(nrf['useful'])} -- workable for SBI.")

    # astrocyte engagement
    astro_engaged = arf['sparse'] + arf['active'] + arf['hyperactive']
    if astro_engaged < 0.1:
        L.append(f"- **ASTROCYTES MOSTLY SILENT**: only {_pct(astro_engaged)} of sims "
                 f"show any Ca2+ activity. The gliotransmission axes (O_N, O_beta, "
                 f"G_T, C_Theta, U_A) may be too weak across the prior, or neuron "
                 f"drive too low to engage them -- check neuron<->astro coupling below.")
    else:
        L.append(f"- Astrocyte engagement {_pct(astro_engaged)} of sims "
                 f"(active+hyperactive {_pct(arf['active'] + arf['hyperactive'])}).")

    # neuron <-> astro coupling
    rho = roll['na_coupling_rho']
    if np.isfinite(rho):
        if rho > 0.3:
            L.append(f"- **COUPLED**: corr(log FR, log astro rate) = {rho:+.2f} -- "
                     f"astrocyte activity tracks network firing, as expected from "
                     f"glutamate-driven Ca2+. The astro observable carries real "
                     f"information about neuronal state (useful as an SBI feature).")
        elif abs(rho) < 0.15:
            L.append(f"- **DECOUPLED**: corr(log FR, log astro rate) = {rho:+.2f} ~ 0 -- "
                     f"astrocyte Ca2+ is NOT following neuronal rate. Either O_N "
                     f"(glutamate sensitivity) is saturating/flooring across the prior "
                     f"or astro dynamics are self-driven (I_bias/F). Inspect the O_N "
                     f"predictivity row.")
        else:
            L.append(f"- Neuron<->astro coupling weak-positive (rho={rho:+.2f}).")
    lagm = roll['na_lag_median_s']
    if np.isfinite(lagm):
        sign = 'astro LAGS neuron' if lagm > 0 else 'astro LEADS neuron (check)'
        L.append(f"- Neuron->astro median peak lag = {lagm:+.1f} s ({sign}); "
                 f"{_pct(roll['na_lag_frac_positive'])} of coupled sims have the "
                 f"physiological positive (astro-after-neuron) delay.")

    # GJ waves
    wem = roll['wave_enrichment_median']
    if np.isfinite(wem):
        if wem > 1.2:
            L.append(f"- **GJ WAVES**: median co-activation enrichment {wem:.2f}x along "
                     f"GJ edges vs random pairs -- Ca2+ activity propagates through the "
                     f"astrocyte gap-junction network. F (GJ permeability) is the axis "
                     f"to inspect; {_pct(roll['wave_frac_above_1'])} of sims show "
                     f"enrichment > 1.")
        else:
            L.append(f"- No clear GJ propagation (median enrichment {wem:.2f}x ~ 1): "
                     f"astro activation looks spatially independent. If F spans a wide "
                     f"prior and waves never appear, the coupling term may be too weak.")

    # bursts (membrane fact now corrected)
    if roll['sims_with_burst_proxy'] == 0:
        L.append(f"- **NO NEURON BURSTS** under the relaxed proxy. Note tau_m = "
                 f"Cm/gL = {TAU_M_MS:.0f} ms now (RS freeze), which is slow enough to "
                 f"summate -- so absence of bursts is a network-gain/recurrence "
                 f"issue, NOT the old fast-membrane explanation.")
    else:
        L.append(f"- Neuron burst proxy fired in {roll['sims_with_burst_proxy']} sims "
                 f"({_pct(roll['sims_with_burst_proxy'] / max(1, roll['n_iters']))}).")

    # identifiability
    p = roll['predictivity']
    if p.get('table'):
        weak = [d['name'] for d in p['table']
                if (not np.isfinite(d['rho_fr']) or abs(d['rho_fr']) < 0.1)
                and (not np.isfinite(d['rho_astro']) or abs(d['rho_astro']) < 0.1)]
        if weak:
            L.append(f"- **POORLY IDENTIFIABLE** (|rho|<0.1 vs BOTH observables): "
                     f"{', '.join(weak)}. These axes barely move either summary "
                     f"statistic, so the SBI posterior over them will be near-prior "
                     f"unless the feature vector adds statistics they DO move.")
        if p.get('unexpected_swept'):
            L.append(f"- CONFIG DRIFT: axes swept but not in the expected 21: "
                     f"{', '.join(p['unexpected_swept'])}.")
        if p.get('missing_swept'):
            L.append(f"- CONFIG DRIFT: expected-swept axes that are CONSTANT in the "
                     f"data: {', '.join(p['missing_swept'])} (frozen, or the run used "
                     f"a different sweep_group).")
    if 'topology' in roll:
        t = roll['topology']
        L.append(f"- Topology: neuron in-degree K={t['mean_in_degree']:.1f}, "
                 f"{_pct(t['frac_zero_in'])} zero-input; astro GJ degree "
                 f"{t['mean_gj_degree']:.1f}, GJ component "
                 f"{_pct(t['mean_gj_component_frac'])}; neuron component "
                 f"{_pct(t['mean_largest_component_frac'])}.")
    return L


def _write_markdown(out: Path, campaign: Path, roll, iters, topos,
                    figs, skipped, elapsed) -> None:
    nrf = roll['neuron_regime_fractions']
    arf = roll['astro_regime_fractions']
    fig_names = [Path(f).name for f in figs]
    L = []
    A = L.append
    A(f"# Campaign report -- `{campaign.name}`  (astrocyte sweep, 37-D)\n")
    A(f"_Generated by campaign_report.py in {elapsed:.1f}s. RS-frozen neurons; "
      f"swept group = synaptic + astro + O_N (21 axes)._\n")

    A("## 1. Status verdict\n")
    for v in _verdict_lines(roll):
        A(v)
    A("")

    A("## 2. What was found on disk\n")
    A(f"- campaign path: `{campaign}`")
    A(f"- topologies analysed: **{roll['n_topologies']}**")
    A(f"- simulations analysed: **{roll['n_iters']}**")
    A(f"- skipped (unreadable/degenerate): {skipped.get('iters',0)} iters, "
      f"{skipped.get('topos',0)} topologies")
    if roll.get('predictivity', {}).get('swept_axes'):
        A(f"- swept axes detected ({len(roll['predictivity']['swept_axes'])}): "
          f"{', '.join(roll['predictivity']['swept_axes'])}")
    A("")

    A("## 3. Neuron dynamics\n")
    A("| regime | fraction | meaning |")
    A("|---|---|---|")
    A(f"| one_spike_artifact | {_pct(nrf['one_spike_artifact'])} | ~1 spike/neuron then silent |")
    A(f"| silent (<{R_SILENT} Hz) | {_pct(nrf['silent'])} | sub-threshold |")
    A(f"| low ({R_SILENT}-{R_LOW} Hz) | {_pct(nrf['low'])} | sparse asynchronous |")
    A(f"| **useful ({R_LOW}-{R_SEIZE} Hz)** | **{_pct(nrf['useful'])}** | workable SBI yield |")
    A(f"| seizing (>{R_SEIZE} Hz) | {_pct(nrf['seizing'])} | supercritical |")
    A("")
    A(f"- mean_FR: median **{roll['fr_median']:.3f}**, mean **{roll['fr_mean']:.3f}**, "
      f"max **{roll['fr_max']:.2f}** Hz")
    A(f"- burst proxy fired in **{roll['sims_with_burst_proxy']}** sims "
      f"(tau_m = {TAU_M_MS:.0f} ms RS; gap V_T-E_L-DeltaT = {GAP_MV:.0f} mV)")
    A("")

    A("## 4. Astrocyte dynamics\n")
    A("| Ca2+ regime | fraction | per-cell event rate |")
    A("|---|---|---|")
    A(f"| silent (<{A_SILENT} Hz) | {_pct(arf['silent'])} | effectively no Ca2+ events |")
    A(f"| sparse ({A_SILENT}-{A_LOW} Hz) | {_pct(arf['sparse'])} | occasional events |")
    A(f"| **active ({A_LOW}-{A_HIGH} Hz)** | **{_pct(arf['active'])}** | physiological band |")
    A(f"| hyperactive (>{A_HIGH} Hz) | {_pct(arf['hyperactive'])} | saturating Ca2+ |")
    A("")
    A(f"- astro event rate: median **{roll['astro_rate_median']:.4f}**, "
      f"mean **{roll['astro_rate_mean']:.4f}**, max **{roll['astro_rate_max']:.3f}** Hz/cell")
    A(f"- neuron<->astro coupling: **rho(log FR, log astro rate) = "
      f"{roll['na_coupling_rho']:+.2f}**")
    A(f"- neuron->astro delay: median lag **{roll['na_lag_median_s']:+.1f} s**, "
      f"{_pct(roll['na_lag_frac_positive'])} positive (astro after neuron)")
    A(f"- GJ-wave enrichment: median **{roll['wave_enrichment_median']:.2f}x**, "
      f"{_pct(roll['wave_frac_above_1'])} of sims > 1x  "
      f"(>1 -> Ca2+ co-activates along GJ edges)")
    A(f"- astro synchrony: **{roll['sims_with_astro_sync']}** sims show population "
      f"Ca2+ peaks")
    A("")

    A("## 5. Swept-axis identifiability (Spearman vs BOTH observables)\n")
    A("Each swept axis vs log neuron FR and log astro event rate, on the active "
      "subset. |rho|>~0.3 = strongly identifiable from that observable; weak on "
      "BOTH = the SBI posterior will be near-prior unless the feature vector adds "
      "statistics that axis moves.\n")
    A("| axis | group | rho vs FR | rho vs astro rate | n_FR | n_astro |")
    A("|---|---|---|---|---|---|")
    for d in sorted(roll['predictivity'].get('table', []),
                    key=lambda r: (r['group'], -abs(r['rho_fr'])
                                   if np.isfinite(r['rho_fr']) else 0)):
        rfr = '  n/a' if not np.isfinite(d['rho_fr']) else f"{d['rho_fr']:+.2f}"
        rar = '  n/a' if not np.isfinite(d['rho_astro']) else f"{d['rho_astro']:+.2f}"
        A(f"| {d['name']} | {d['group']} | {rfr} | {rar} | {d['n_fr']} | {d['n_astro']} |")
    A("")

    if 'topology' in roll:
        t = roll['topology']
        A("## 6. Topology\n")
        A(f"- neuron mean in-degree **K = {t['mean_in_degree']:.1f}** "
          f"(median {t['median_in_degree']:.1f}); zero-input **{_pct(t['frac_zero_in'])}**")
        A(f"- mean synapses/topology: {t['mean_n_syn']:.0f}; "
          f"largest neuron component **{_pct(t['mean_largest_component_frac'])}**")
        A(f"- astrocyte GJ degree **{t['mean_gj_degree']:.1f}**; "
          f"GJ component **{_pct(t['mean_gj_component_frac'])}**")
        A(f"- mean edge length {t['mean_conn_distance']:.0f} um; "
          f"reciprocity {t['mean_reciprocity']:.2f}; rules {t['conn_rules']}")
        A("")

    A("## 7. Recommended next actions\n")
    A("1. If astrocytes are mostly silent but neurons fire: O_N/O_beta lower bounds "
      "or G_T may be too weak -- widen upward, or check neuron->astro StoA wiring.")
    A("2. Feed the SBI inference an x-vector that includes the astro observables "
      "(astro_rate, wave_enrichment, na_lag) so the astro-only axes (O_N, F, G_T) "
      "become identifiable -- the table above shows which need it.")
    A("3. If GJ waves never appear across a wide F prior, raise F's upper bound or "
      "confirm the GJ network is connected (GJ component fraction in section 6).")
    A("4. Re-run with --topo-sample to spot-check before a full-campaign pass.")
    A("")

    if fig_names:
        A("## 8. Figures\n")
        for f in fig_names:
            A(f"- `figures/{f}`")
        A("")

    A("## 9. Per-topology table (first 30)\n")
    A("| topo | task | Nn | Na | n_syn | K_in | %zero_in | comp | n_gj | GJ_deg | n_iter |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for tr in topos[:30]:
        A(f"| {tr.topo_idx} | {tr.task} | {tr.Nn} | {tr.Na} | {tr.n_syn} | "
          f"{tr.mean_in_degree:.1f} | {_pct(tr.frac_zero_in)} | "
          f"{tr.largest_component_frac:.2f} | {tr.n_gj} | {tr.mean_gj_degree:.1f} | "
          f"{tr.n_iters} |")
    A("")
    out.write_text("\n".join(L))


# ======================================================================
# Driver
# ======================================================================
def analyze_campaign(campaign: Path, out_dir: Path,
                     max_iters_per_topo: int = 0, topo_sample: int = 0) -> Dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / 'figures'

    topo_dirs = _discover_topologies(campaign)
    if topo_sample and len(topo_dirs) > topo_sample:
        sel = np.linspace(0, len(topo_dirs) - 1, topo_sample).astype(int)
        topo_dirs = [topo_dirs[i] for i in sel]

    iters: List[IterRecord] = []
    topos: List[TopoRecord] = []
    topo_extra_repr: Optional[Dict] = None
    skipped = dict(iters=0, topos=0)

    for td in topo_dirs:
        meta = _read_json(td / 'topology_meta.json')
        ta = _analyze_topology(td, meta)
        if ta is None:
            skipped['topos'] += 1
            tr = None
            gj_i = gj_j = np.empty(0, dtype=np.int64)
        else:
            tr, extra, gj_i, gj_j = ta
            if topo_extra_repr is None or tr.n_syn > topo_extra_repr.get('_n_syn', -1):
                topo_extra_repr = dict(extra); topo_extra_repr['_n_syn'] = tr.n_syn

        Nn = int(meta.get('Nn') or 0) or (tr.Nn if tr else 0)
        Na = int(meta.get('Na') or 0) or (tr.Na if tr else 0)
        T = float(meta.get('simtime_s', meta.get('T', 0.0)) or 0.0) or (tr.T if tr else 0.0)

        npzs = sorted(td.glob('iter_*.npz'))
        if max_iters_per_topo:
            npzs = npzs[:max_iters_per_topo]
        n_here = 0
        for npz_path in npzs:
            d = _safe_load_npz(npz_path)
            if d is None:
                skipped['iters'] += 1
                continue
            sc = _read_json(npz_path.with_suffix('.json'))
            try:
                iter_idx = int(npz_path.stem.split('_')[-1])
            except Exception:
                iter_idx = -1
            rec = _analyze_iter(d, sc, Nn, Na, T, tr.topo_idx if tr else -1,
                                iter_idx, gj_i, gj_j)
            if rec is None:
                skipped['iters'] += 1
                continue
            iters.append(rec)
            n_here += 1
        if tr is not None:
            tr.n_iters = n_here
            topos.append(tr)

    predict = compute_predictivity(iters)
    roll = _roll_up(iters, topos, predict)
    figs = _make_figures(iters, topos, topo_extra_repr, predict, fig_dir)
    elapsed = time.time() - t0

    _write_markdown(out_dir / 'campaign_report.md', campaign, roll, iters, topos,
                    figs, skipped, elapsed)
    summary = dict(campaign=str(campaign), roll_up=roll, skipped=skipped,
                   elapsed_s=elapsed, figures=[Path(f).name for f in figs],
                   per_topology=[asdict(t) for t in topos])
    (out_dir / 'campaign_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


# ======================================================================
# Smoke test (synthetic campaign; exercises every new code path)
# ======================================================================
def _smoke_test() -> int:
    import tempfile
    print("[smoke] building synthetic astrocyte campaign ...")
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    Nn, Na, T = 192, 192, 60.0
    camp = tmp / 'campaign_smoke'
    td = camp / 'sweep_cpu_task0000' / 'topo_00000'
    td.mkdir(parents=True)

    # neuron graph (mean K ~ 15)
    n_syn = Nn * 15
    S_i = rng.integers(0, Nn, n_syn).astype(np.int32)
    S_j = rng.integers(0, Nn, n_syn).astype(np.int32)
    keep = S_i != S_j
    S_i, S_j = S_i[keep], S_j[keep]
    N_pos = rng.uniform(0, 400, (Nn, 2))

    # astrocyte GJ graph: 1-D chain a-(a+1) so neighbours are well defined
    A_pos = rng.uniform(0, 400, (Na, 2))
    GJ_i = np.arange(Na - 1, dtype=np.int32)
    GJ_j = np.arange(1, Na, dtype=np.int32)
    np.savez_compressed(td / 'topology.npz', N_pos=N_pos, S_i=S_i, S_j=S_j,
                        A_pos=A_pos, GJ_i=GJ_i, GJ_j=GJ_j,
                        StoA_i=np.empty(0, np.int32), StoA_j=np.empty(0, np.int32))
    (td / 'topology_meta.json').write_text(json.dumps(
        dict(Nn=Nn, Na=Na, simtime_s=T, mode='Full', topo_idx=0,
             conn_rule='flat', c_max=400.0)))

    # iters: vary g_ampa (idx14) and O_N (idx36). Activity is TIME-LOCALISED in
    # bursts so the temporal proxies are well-posed: neuron bursts at burst_times,
    # astro events trail by +5 s (lag), placed on GJ-adjacent cell pairs (wave),
    # with both neuron and astro counts scaling with g_ampa (coupling).
    n_iters = 12
    n_bursts = 6
    for k in range(n_iters):
        params = np.zeros(N_DIMS, dtype=np.float64)
        params[IDX['Sigma']] = 4.0
        params[IDX['I_inj']] = 7.5
        params[IDX['Cm']] = 200.0
        params[IDX['gL']] = 10.0
        g = float(rng.uniform(0.1, 5.0)); params[IDX['g_ampa']] = g
        on = float(10 ** rng.uniform(-1.5, 0.5)); params[IDX['O_N']] = on
        params[IDX['g_nmda']] = float(rng.uniform(0.01, 1.0))
        params[IDX['G_T']] = float(rng.uniform(50, 1000))

        burst_times = np.sort(rng.uniform(5.0, T - 10.0, n_bursts))

        # neuron spikes: Gaussian bumps at burst_times, amplitude scales with g
        per_burst = int(40 * (g / 5.0)) + Nn // n_bursts
        nt, ni = [], []
        for bt in burst_times:
            m = max(1, int(rng.poisson(per_burst)))
            nt.append(bt + rng.normal(0.0, 0.5, m))
            ni.append(rng.integers(0, Nn, m))
        spk_t = np.clip(np.concatenate(nt), 0, T).astype(np.float32)
        spk_i = np.concatenate(ni).astype(np.int32)
        o = np.argsort(spk_t); spk_t, spk_i = spk_t[o], spk_i[o]

        # astro events: burst_time + 5 s on GJ-adjacent pairs; count scales w/ g
        astro_per_burst = int(5 * (g / 5.0)) + 2
        at, ai = [], []
        for bt in burst_times:
            for _ in range(astro_per_burst):
                a0 = int(rng.integers(0, Na - 1))
                at.append(bt + 5.0 + rng.normal(0.0, 0.3)); ai.append(a0)
                if rng.random() < 0.85:                       # co-activate GJ neighbour
                    at.append(bt + 5.0 + rng.normal(0.0, 0.3)); ai.append(a0 + 1)
        a_t = np.clip(np.asarray(at), 0, T).astype(np.float32)
        a_i = np.asarray(ai, np.int32)
        o = np.argsort(a_t); a_t, a_i = a_t[o], a_i[o]

        np.savez_compressed(td / f'iter_{k:05d}.npz',
                            spk_N_t=spk_t, spk_N_i=spk_i,
                            spk_A_t=a_t, spk_A_i=a_i,
                            params=params, theta=params)

    summary = analyze_campaign(camp, tmp / 'report')
    r = summary['roll_up']
    p = r['predictivity']
    swept = set(p['swept_axes'])
    g_ampa_rho = next((d['rho_fr'] for d in p['table'] if d['name'] == 'g_ampa'), np.nan)

    print(f"[smoke] iters={r['n_iters']} topos={r['n_topologies']}")
    print(f"[smoke] swept detected: {sorted(swept)}")
    print(f"[smoke] g_ampa rho_vs_FR = {g_ampa_rho:+.2f}  (expect strongly +)")
    print(f"[smoke] na_coupling rho  = {r['na_coupling_rho']:+.2f}  (expect +)")
    print(f"[smoke] na_lag median    = {r['na_lag_median_s']:+.1f} s  (expect ~ +5)")
    print(f"[smoke] wave enrichment  = {r['wave_enrichment_median']:.2f}x  (expect > 1)")

    checks = {
        "n_iters == 12":            r['n_iters'] == n_iters,
        "n_topologies == 1":        r['n_topologies'] == 1,
        "g_ampa & O_N swept":       {'g_ampa', 'O_N', 'g_nmda', 'G_T'} <= swept,
        "frozen Sigma NOT swept":   'Sigma' not in swept,
        "g_ampa predicts FR (+)":   np.isfinite(g_ampa_rho) and g_ampa_rho > 0.5,
        "neuron-astro coupled (+)": np.isfinite(r['na_coupling_rho']) and r['na_coupling_rho'] > 0.3,
        "astro lags neuron (+)":    np.isfinite(r['na_lag_median_s']) and r['na_lag_median_s'] > 2.0,
        "GJ wave enrichment > 1":   np.isfinite(r['wave_enrichment_median']) and r['wave_enrichment_median'] > 1.0,
        "report.md exists":         (tmp / 'report' / 'campaign_report.md').exists(),
        "summary.json exists":      (tmp / 'report' / 'campaign_summary.json').exists(),
        "GJ topo degree > 0":       r['topology']['mean_gj_degree'] > 0,
    }
    ok = all(checks.values())
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("[smoke]", "PASS" if ok else "FAIL")
    print(f"[smoke] report at {tmp/'report'/'campaign_report.md'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('campaign', nargs='?', help='path to campaign_<TAG> directory')
    ap.add_argument('--out', default=None, help='output dir (default <campaign>/report)')
    ap.add_argument('--max-iters-per-topo', type=int, default=0,
                    help='cap iters per topology (0 = all)')
    ap.add_argument('--topo-sample', type=int, default=0,
                    help='analyse only N evenly-spaced topologies (0 = all)')
    ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args()

    if args.smoke_test:
        return _smoke_test()
    if not args.campaign:
        ap.error('campaign path required (or use --smoke-test)')
    campaign = Path(args.campaign).resolve()
    if not campaign.is_dir():
        ap.error(f'not a directory: {campaign}')
    out_dir = Path(args.out) if args.out else campaign / 'report'

    summary = analyze_campaign(campaign, out_dir,
                               max_iters_per_topo=args.max_iters_per_topo,
                               topo_sample=args.topo_sample)
    r = summary['roll_up']
    print(f"\n[campaign_report] {r['n_iters']} sims / {r['n_topologies']} topologies")
    print(f"  neuron useful-band yield: {_pct(r['neuron_regime_fractions']['useful'])}")
    print(f"  astro active+hyper:       "
          f"{_pct(r['astro_regime_fractions']['active'] + r['astro_regime_fractions']['hyperactive'])}")
    print(f"  neuron<->astro rho:       {r['na_coupling_rho']:+.2f}")
    print(f"  GJ-wave enrichment:       {r['wave_enrichment_median']:.2f}x")
    print(f"  report : {out_dir/'campaign_report.md'}")
    print(f"  summary: {out_dir/'campaign_summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
