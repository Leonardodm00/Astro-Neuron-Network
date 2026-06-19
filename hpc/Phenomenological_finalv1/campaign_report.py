#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
campaign_report.py
===================================================================
Campaign-level diagnostic harvester for the CAdEx neuron-astrocyte SBI sweep.

Target a campaign directory; walk every topology and every iter_*.npz; extract
DYNAMICAL (per-sim firing statistics, regime classification, burst proxies) and
TOPOLOGICAL (degree distribution, distance-dependent connectivity, components)
properties; emit:

    <out>/campaign_report.md     human + LLM-oriented status report (VERY verbose)
    <out>/campaign_summary.json  machine-readable roll-up (every number above)
    <out>/figures/*.png          dynamical + topological diagnostic figures

The .md is deliberately written to be maximally informative to an assistant
picking up the campaign cold: it states the on-disk layout actually found, the
parameter ranges actually sampled, the regime mix, the I_inj/sigma predictivity,
the startup-artifact fraction, the connectivity statistics, and an explicit
"WHAT THIS MEANS" verdict block keyed to known failure modes.

ON-DISK CONTRACT (from HPC_main_sweep.py / HPC_single_run.py)
------------------------------------------------------------
campaign_<TAG>/
  sweep_<part>_task<NNNN>/
    topo_<NNNNN>/
      topology.npz            N_pos(Nn,2) S_i S_j(n_syn) A_pos(Na,2)
                              GJ_i GJ_j StoA_i StoA_j  [S_x_syn S_y_syn]
      topology_meta.json      {Nn, Na, T, mode, topo_idx, conn_rule,
                               p0_conn, d0_conn, beta_conn, c_max, ...}
      iter_<NNNNN>.npz        spk_N_t spk_N_i spk_A_t spk_A_i
                              params(35,) theta(35,)
      iter_<NNNNN>.json       sidecar stats (mean_FR, across_cell_rate_cv, ...)

Run on HPC
----------
    python campaign_report.py /path/to/campaign_<TAG>
    python campaign_report.py campaign_<TAG> --out ./report --max-iters-per-topo 0
    python campaign_report.py campaign_<TAG> --topo-sample 50   # subsample topologies
    python campaign_report.py --smoke-test                      # synthetic self-check

Notes
-----
* Pure numpy / scipy / matplotlib; no Brian2 needed (reads results only).
* Streams iter files (never holds all spike arrays at once); safe on 100k+ sims.
* Degenerate / missing files are skipped and COUNTED in the report, never fatal.
===================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parameter-vector index map (35-D registry; Doorn / Bucket-A)
# ----------------------------------------------------------------------
PARAM_NAMES = [
    'Sigma', 'gbarA', 'EC50_ampa', 'EC50_nmda', 'tauA',
    'U_0_ar', 'U_max', 'U_0_sr', 'Omega_f_sr', 'Omega_f_ar', 'Omega_d',
    'alpha_syn', 'DeltaT', 'VT', 'g_ampa', 'g_nmda', 'delta_gA', 'x0',
    'O_G', 'Omega_G', 'O_beta', 'O_3K', 'Omega_5P', 'I_bias', 'F',
    'I_Theta', 'omega_I', 'C_Theta', 'U_A', 'G_T', 'gL', 'VA', 'DeltaA',
    'VR', 'I_inj',
]
IDX = {n: i for i, n in enumerate(PARAM_NAMES)}
GAP_MV = 5.0   # frozen V_T - E_L - DeltaT (for eta = gap / sigma)

# Regime thresholds (Hz); ONE_SPIKE flagged separately below.
R_SILENT = 0.005
R_LOW    = 0.1
R_SEIZE  = 5.0


# ======================================================================
# Data containers
# ======================================================================
@dataclass
class IterRecord:
    topo_idx: int
    iter_idx: int
    n_spikes: int
    mean_fr: float
    sigma: float
    i_inj: float
    g_ampa: float
    g_nmda: float
    gbar_a: float
    delta_ga: float
    vr: float
    # dynamical features
    frac_active: float
    across_cell_rate_cv: float
    mean_isi_cv: float
    peak_pop_rate: float
    peak_over_mean: float
    burst_proxy: int            # # of supra-threshold population-rate peaks
    one_spike_artifact: bool    # exactly ~1 spike / neuron
    regime: str


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
    """All topo_* dirs under any sweep_*_task* (or directly under campaign)."""
    topos = sorted(campaign.glob('sweep_*_task*/topo_*'))
    if not topos:
        topos = sorted(campaign.glob('**/topo_*'))
    return [t for t in topos if t.is_dir()]


# ======================================================================
# Dynamical feature extraction (per iter)
# ======================================================================
def _population_rate(spk_t: np.ndarray, Nn: int, T: float,
                     bin_s: float = 0.005, smooth_bins: int = 3
                     ) -> np.ndarray:
    if spk_t.size == 0 or T <= 0:
        return np.zeros(1)
    n_bins = max(1, int(T / bin_s))
    counts, _ = np.histogram(spk_t, bins=n_bins, range=(0.0, T))
    rate = counts / (Nn * bin_s)
    if smooth_bins > 1:
        k = np.ones(smooth_bins) / smooth_bins
        rate = np.convolve(rate, k, mode='same')
    return rate


def _per_neuron_rates(spk_i: np.ndarray, Nn: int, T: float) -> np.ndarray:
    if spk_i.size == 0:
        return np.zeros(Nn)
    counts = np.bincount(spk_i, minlength=Nn)[:Nn]
    return counts / T


def _isi_cv_mean(spk_t: np.ndarray, spk_i: np.ndarray, Nn: int,
                 min_spikes: int = 4) -> float:
    """Mean within-neuron ISI CV across neurons with >= min_spikes."""
    if spk_t.size == 0:
        return 0.0
    order = np.argsort(spk_i, kind='stable')
    si = spk_i[order]
    st = spk_t[order]
    cvs = []
    # group by neuron index
    bounds = np.searchsorted(si, np.arange(Nn + 1))
    for k in range(Nn):
        a, b = bounds[k], bounds[k + 1]
        if b - a >= min_spikes:
            isi = np.diff(np.sort(st[a:b]))
            m = isi.mean()
            if m > 0:
                cvs.append(isi.std() / m)
    return float(np.mean(cvs)) if cvs else 0.0


def _burst_proxy(pop_rate: np.ndarray, min_amp_ratio: float = 2.5,
                 prom_frac: float = 0.4) -> Tuple[int, float, float]:
    """
    Cheap detector-agnostic burst proxy: count population-rate peaks that exceed
    both an absolute ratio (peak/mean) and a relative prominence. Returns
    (n_peaks, peak_rate, peak_over_mean). Deliberately uses the RELAXED Doorn-
    scale thresholds discussed in the burst diagnosis, not the legacy 5.0/0.8.
    """
    if pop_rate.size < 3:
        return 0, 0.0, 0.0
    mean_r = pop_rate.mean()
    peak_r = pop_rate.max()
    if mean_r <= 0:
        return 0, float(peak_r), 0.0
    ratio = peak_r / mean_r
    try:
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(pop_rate,
                              height=max(min_amp_ratio * mean_r, 1e-9),
                              prominence=prom_frac * peak_r,
                              distance=max(1, int(len(pop_rate) * 0.01)))
        n = int(peaks.size)
    except Exception:
        n = int(ratio >= min_amp_ratio)
    return n, float(peak_r), float(ratio)


def _classify(mean_fr: float, one_spike: bool) -> str:
    if one_spike:
        return 'one_spike_artifact'
    if mean_fr < R_SILENT:
        return 'silent'
    if mean_fr < R_LOW:
        return 'low'
    if mean_fr < R_SEIZE:
        return 'useful'
    return 'seizing'


def _analyze_iter(npz: Dict[str, np.ndarray], sidecar: Dict,
                  Nn: int, T: float, topo_idx: int, iter_idx: int
                  ) -> Optional[IterRecord]:
    if 'params' not in npz:
        return None
    params = np.asarray(npz['params'], dtype=np.float64)
    if params.shape[0] < 35:
        return None
    spk_t = np.asarray(npz.get('spk_N_t', np.empty(0)), dtype=np.float64)
    spk_i = np.asarray(npz.get('spk_N_i', np.empty(0)), dtype=np.int64)
    n_spk = int(spk_t.size)

    mean_fr = n_spk / (Nn * T) if (Nn > 0 and T > 0) else 0.0
    one_spike = (Nn > 0) and (abs(n_spk - Nn) <= max(2, 0.02 * Nn)) \
        and (mean_fr < (1.5 / T))

    rates = _per_neuron_rates(spk_i, Nn, T)
    frac_active = float((rates > 0).mean()) if Nn else 0.0
    mu_r = rates[rates > 0].mean() if np.any(rates > 0) else 0.0
    across_cv = float(rates[rates > 0].std() / mu_r) if mu_r > 0 else 0.0
    isi_cv = _isi_cv_mean(spk_t, spk_i, Nn)

    pr = _population_rate(spk_t, Nn, T)
    n_pk, peak_r, peak_over_mean = _burst_proxy(pr)

    # prefer sidecar values where present (authoritative)
    frac_active = float(sidecar.get('frac_active', frac_active))
    across_cv = float(sidecar.get('across_cell_rate_cv', across_cv))
    isi_cv = float(sidecar.get('mean_isi_cv', isi_cv))

    return IterRecord(
        topo_idx=topo_idx, iter_idx=iter_idx, n_spikes=n_spk,
        mean_fr=mean_fr, sigma=float(params[IDX['Sigma']]),
        i_inj=float(params[IDX['I_inj']]), g_ampa=float(params[IDX['g_ampa']]),
        g_nmda=float(params[IDX['g_nmda']]), gbar_a=float(params[IDX['gbarA']]),
        delta_ga=float(params[IDX['delta_gA']]), vr=float(params[IDX['VR']]),
        frac_active=frac_active, across_cell_rate_cv=across_cv,
        mean_isi_cv=isi_cv, peak_pop_rate=peak_r, peak_over_mean=peak_over_mean,
        burst_proxy=n_pk, one_spike_artifact=one_spike,
        regime=_classify(mean_fr, one_spike),
    )


# ======================================================================
# Topological feature extraction (per topo)
# ======================================================================
def _degree_stats(S_i: np.ndarray, S_j: np.ndarray, Nn: int) -> Dict:
    """in/out degree from directed edges S_i (pre) -> S_j (post)."""
    out_deg = np.bincount(S_i, minlength=Nn)[:Nn] if S_i.size else np.zeros(Nn, int)
    in_deg  = np.bincount(S_j, minlength=Nn)[:Nn] if S_j.size else np.zeros(Nn, int)
    return dict(
        in_deg=in_deg, out_deg=out_deg,
        mean_in=float(in_deg.mean()), mean_out=float(out_deg.mean()),
        median_in=float(np.median(in_deg)), max_in=int(in_deg.max() if Nn else 0),
        frac_zero_in=float((in_deg == 0).mean()) if Nn else 1.0,
    )


def _reciprocity(S_i: np.ndarray, S_j: np.ndarray) -> float:
    """Fraction of edges i->j for which j->i also exists."""
    if S_i.size == 0:
        return 0.0
    edges = set(zip(S_i.tolist(), S_j.tolist()))
    recip = sum(1 for (a, b) in edges if (b, a) in edges)
    return recip / len(edges)


def _components(S_i: np.ndarray, S_j: np.ndarray, Nn: int) -> Tuple[int, float]:
    """Weakly-connected components via scipy sparse; returns (n_comp, largest_frac)."""
    if Nn == 0:
        return 0, 0.0
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        data = np.ones(S_i.size, dtype=np.int8)
        A = coo_matrix((data, (S_i, S_j)), shape=(Nn, Nn))
        n_comp, labels = connected_components(A, directed=True, connection='weak')
        _, counts = np.unique(labels, return_counts=True)
        return int(n_comp), float(counts.max() / Nn)
    except Exception:
        return -1, 0.0


def _mean_conn_distance(N_pos: np.ndarray, S_i: np.ndarray,
                        S_j: np.ndarray, c_max: float,
                        max_edges: int = 200_000) -> float:
    if S_i.size == 0 or N_pos.size == 0:
        return 0.0
    if S_i.size > max_edges:                       # subsample for speed
        sel = np.random.default_rng(0).choice(S_i.size, max_edges, replace=False)
        S_i, S_j = S_i[sel], S_j[sel]
    d = N_pos[S_i] - N_pos[S_j]
    return float(np.sqrt((d * d).sum(axis=1)).mean())


def _analyze_topology(topo_dir: Path, meta: Dict) -> Optional[Tuple[TopoRecord, Dict]]:
    npz = _safe_load_npz(topo_dir / 'topology.npz')
    if npz is None or 'S_i' not in npz:
        return None
    Nn = int(meta.get('Nn', npz['N_pos'].shape[0] if 'N_pos' in npz else 0))
    S_i = np.asarray(npz['S_i'], dtype=np.int64)
    S_j = np.asarray(npz['S_j'], dtype=np.int64)
    N_pos = np.asarray(npz.get('N_pos', np.empty((Nn, 2))), dtype=np.float64)
    c_max = float(meta.get('c_max', N_pos.max() if N_pos.size else 0.0))

    dg = _degree_stats(S_i, S_j, Nn)
    n_comp, big = _components(S_i, S_j, Nn)
    rec = TopoRecord(
        topo_idx=int(meta.get('topo_idx', -1)),
        task=topo_dir.parent.name,
        Nn=Nn, Na=int(meta.get('Na', 0)), T=float(meta.get('T', 0.0)),
        conn_rule=str(meta.get('conn_rule', '?')),
        p0_conn=float(meta.get('p0_conn', np.nan)),
        d0_conn=float(meta.get('d0_conn', np.nan)),
        beta_conn=float(meta.get('beta_conn', np.nan)),
        c_max=c_max, n_syn=int(S_i.size),
        mean_in_degree=dg['mean_in'], mean_out_degree=dg['mean_out'],
        median_in_degree=dg['median_in'], max_in_degree=dg['max_in'],
        frac_zero_in=dg['frac_zero_in'],
        n_components=n_comp, largest_component_frac=big,
        mean_conn_distance=_mean_conn_distance(N_pos, S_i, S_j, c_max),
        reciprocity=_reciprocity(S_i, S_j),
        n_iters=0,
    )
    extra = dict(in_deg=dg['in_deg'], N_pos=N_pos, S_i=S_i, S_j=S_j, c_max=c_max)
    return rec, extra


# ======================================================================
# Figures
# ======================================================================
def _make_figures(iters: List[IterRecord], topos: List[TopoRecord],
                  topo_extra: Optional[Dict], fig_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    if not iters:
        return made

    fr   = np.array([r.mean_fr for r in iters])
    sig  = np.array([r.sigma for r in iters])
    iinj = np.array([r.i_inj for r in iters])
    gA   = np.array([r.g_ampa for r in iters])
    regimes = [r.regime for r in iters]

    # 1. firing-rate distribution (log) with regime bands
    fig, ax = plt.subplots(figsize=(7, 4))
    pos = fr[fr > 0]
    if pos.size:
        ax.hist(np.log10(pos), bins=50, color='#4477aa', alpha=0.85)
    for x, lab in [(np.log10(R_SILENT), 'silent'), (np.log10(R_LOW), 'low'),
                   (np.log10(R_SEIZE), 'seize')]:
        ax.axvline(x, color='k', ls='--', lw=0.8)
        ax.text(x, ax.get_ylim()[1]*0.92, lab, fontsize=7, rotation=90, va='top')
    ax.set_xlabel('log10 mean_FR (Hz)'); ax.set_ylabel('# sims')
    ax.set_title('Dynamical regime distribution')
    fig.tight_layout(); p = fig_dir / 'fr_distribution.png'
    fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    # 2. sigma / I_inj predictivity (the key SBI identifiability check)
    act = fr > 0.01
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, x, lab in [(axes[0], sig, 'sigma (mV)'), (axes[1], iinj, 'I_inj (pA)')]:
        ax.scatter(x[act], fr[act], s=6, alpha=0.4, color='#cc6677')
        ax.set_yscale('log'); ax.set_xlabel(lab); ax.set_ylabel('mean_FR (Hz)')
        if act.sum() > 5:
            from scipy.stats import spearmanr
            rho = spearmanr(x[act], np.log(fr[act]))[0]
            ax.set_title(f'rho(log FR) = {rho:+.2f}')
    fig.suptitle('Does the driver predict network rate?  (≈0 → recurrence-driven)')
    fig.tight_layout(); p = fig_dir / 'driver_predictivity.png'
    fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    # 3. regime pie
    fig, ax = plt.subplots(figsize=(5.5, 5))
    order = ['one_spike_artifact', 'silent', 'low', 'useful', 'seizing']
    counts = [regimes.count(o) for o in order]
    cols = ['#999999', '#332288', '#88ccee', '#117733', '#cc6677']
    ax.pie([c for c in counts if c], labels=[o for o, c in zip(order, counts) if c],
           colors=[c for c, n in zip(cols, counts) if n], autopct='%1.0f%%',
           textprops={'fontsize': 8})
    ax.set_title(f'Regime mix (N={len(iters)})')
    fig.tight_layout(); p = fig_dir / 'regime_mix.png'
    fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    # 4. topology: degree distribution + connection-distance kernel
    if topo_extra is not None and topo_extra.get('in_deg') is not None:
        in_deg = topo_extra['in_deg']
        N_pos, S_i, S_j = topo_extra['N_pos'], topo_extra['S_i'], topo_extra['S_j']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(in_deg, bins=40, color='#44aa99')
        axes[0].axvline(in_deg.mean(), color='k', ls='--',
                        label=f'mean={in_deg.mean():.1f}')
        axes[0].set_xlabel('in-degree K'); axes[0].set_ylabel('# neurons')
        axes[0].set_title('In-degree distribution'); axes[0].legend(fontsize=8)
        if S_i.size and N_pos.size:
            d = N_pos[S_i] - N_pos[S_j]
            dist = np.sqrt((d * d).sum(axis=1))
            axes[1].hist(dist, bins=50, color='#aa4499')
            axes[1].set_xlabel('connection distance (µm)')
            axes[1].set_ylabel('# synapses')
            axes[1].set_title(f'Distance kernel (mean={dist.mean():.0f} µm)')
        fig.suptitle('Topology (representative topology)')
        fig.tight_layout(); p = fig_dir / 'topology_connectivity.png'
        fig.savefig(p, dpi=130); plt.close(fig); made.append(str(p))

    return made


# ======================================================================
# Report writers
# ======================================================================
def _pct(x: float) -> str:
    return f'{100*x:.0f}%'


def _roll_up(iters: List[IterRecord], topos: List[TopoRecord]) -> Dict:
    fr   = np.array([r.mean_fr for r in iters]) if iters else np.zeros(0)
    sig  = np.array([r.sigma for r in iters]) if iters else np.zeros(0)
    iinj = np.array([r.i_inj for r in iters]) if iters else np.zeros(0)
    reg  = [r.regime for r in iters]
    act  = fr > 0.01

    def _rho(x):
        if act.sum() < 5:
            return float('nan')
        from scipy.stats import spearmanr
        return float(spearmanr(x[act], np.log(fr[act]))[0])

    n = max(1, len(iters))
    roll = dict(
        n_iters=len(iters), n_topologies=len(topos),
        regime_fractions={r: reg.count(r) / n for r in
                          ['one_spike_artifact', 'silent', 'low', 'useful', 'seizing']},
        one_spike_artifact_frac=sum(r.one_spike_artifact for r in iters) / n,
        fr_median=float(np.median(fr)) if fr.size else 0.0,
        fr_mean=float(fr.mean()) if fr.size else 0.0,
        fr_max=float(fr.max()) if fr.size else 0.0,
        bimodality_mean_over_median=(float(fr.mean() / max(np.median(fr), 1e-3))
                                     if fr.size else 0.0),
        corr_sigma_logFR=_rho(sig),
        corr_iinj_logFR=_rho(iinj),
        any_bursts=sum(r.burst_proxy for r in iters),
        sims_with_burst_proxy=sum(1 for r in iters if r.burst_proxy > 0),
        sigma_range=[float(sig.min()), float(sig.max())] if sig.size else [0, 0],
        seizing_median_sigma=(float(np.median([r.sigma for r in iters
                              if r.regime == 'seizing']))
                              if any(r.regime == 'seizing' for r in iters) else None),
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
            conn_rules=sorted({t.conn_rule for t in topos}),
        )
    return roll


def _verdict_lines(roll: Dict) -> List[str]:
    """Explicit status verdict keyed to the known failure modes."""
    L = []
    rf = roll['regime_fractions']
    art = roll['one_spike_artifact_frac']
    if art > 0.1:
        L.append(f"- **STARTUP ARTIFACT**: {_pct(art)} of sims fire ~1 spike/neuron then "
                 f"go silent (V(0)=E_L onset volley). Randomise V(0) and discard a 1–2 s "
                 f"warm-up before computing stats; otherwise these labels are degenerate.")
    if rf['useful'] < 0.1:
        L.append(f"- **LOW YIELD**: only {_pct(rf['useful'])} of sims land in the useful "
                 f"0.1–5 Hz band. Most prior volume is wasted on silent/artifact draws.")
    else:
        L.append(f"- Useful-band yield {_pct(rf['useful'])} — workable for SBI.")
    cs, ci = roll['corr_sigma_logFR'], roll['corr_iinj_logFR']
    if abs(cs) < 0.15 and abs(ci) < 0.15:
        L.append(f"- **RECURRENCE-DRIVEN**: corr(σ,logFR)={cs:+.2f}, corr(I_inj,logFR)={ci:+.2f} "
                 f"≈ 0 → network rate is set by recurrent synaptic gain, NOT the single-cell "
                 f"noise geometry. σ/I_inj are weakly identifiable from rate alone; the SBI "
                 f"feature vector MUST carry across_cell_rate_cv + mean_isi_cv to separate them.")
    if roll.get('seizing_median_sigma') is not None and roll['seizing_median_sigma'] < 6:
        L.append(f"- Seizing tail has median σ={roll['seizing_median_sigma']:.1f} mV (not high) "
                 f"→ ignition is recurrence-driven; gain is already supercritical. Adding "
                 f"connectivity pushes toward seizure, not rhythmic bursting.")
    if roll['sims_with_burst_proxy'] == 0:
        L.append("- **NO BURSTS** even under relaxed proxy thresholds (ratio≥2.5, prom≥0.4). "
                 "Likely the fast Doorn membrane (τ_m=C_m/g_L≈3.3 ms) suppressing temporal "
                 "summation; test by raising C_m (ASD line ~665) on one seizing topology.")
    else:
        L.append(f"- Burst proxy fired in {roll['sims_with_burst_proxy']} sims "
                 f"({_pct(roll['sims_with_burst_proxy']/max(1,roll['n_iters']))}).")
    if 'topology' in roll:
        t = roll['topology']
        L.append(f"- Topology: mean in-degree K={t['mean_in_degree']:.1f}, "
                 f"{_pct(t['frac_zero_in'])} neurons with zero inputs, largest weak "
                 f"component {_pct(t['mean_largest_component_frac'])} of network, "
                 f"reciprocity {t['mean_reciprocity']:.2f}, mean edge length "
                 f"{t['mean_conn_distance']:.0f} µm. "
                 + ("Wiring is NOT the bottleneck if K≳10 and component≈1."
                    if t['mean_in_degree'] >= 10 and t['mean_largest_component_frac'] > 0.9
                    else "Sparse/fragmented wiring may contribute — check component fraction."))
    return L


def _write_markdown(out: Path, campaign: Path, roll: Dict,
                    iters: List[IterRecord], topos: List[TopoRecord],
                    figs: List[str], skipped: Dict, elapsed: float) -> None:
    rf = roll['regime_fractions']
    fig_names = [Path(f).name for f in figs]
    lines = []
    A = lines.append
    A(f"# Campaign report — `{campaign.name}`\n")
    A(f"_Generated by campaign_report.py in {elapsed:.1f}s. "
      f"This file is written to be maximally informative to an assistant reading "
      f"the campaign cold._\n")

    A("## 1. Status verdict\n")
    for v in _verdict_lines(roll):
        A(v)
    A("")

    A("## 2. What was found on disk\n")
    A(f"- campaign path: `{campaign}`")
    A(f"- topologies analysed: **{roll['n_topologies']}**")
    A(f"- simulations analysed: **{roll['n_iters']}**")
    A(f"- skipped (unreadable/degenerate): "
      f"{skipped.get('iters',0)} iters, {skipped.get('topos',0)} topologies")
    if 'topology' in roll:
        A(f"- connectivity rules present: {roll['topology']['conn_rules']}")
    A("")

    A("## 3. Dynamical properties\n")
    A("| regime | fraction | meaning |")
    A("|---|---|---|")
    A(f"| one_spike_artifact | {_pct(rf['one_spike_artifact'])} | ~1 spike/neuron then silent (onset volley) |")
    A(f"| silent (<{R_SILENT} Hz) | {_pct(rf['silent'])} | sub-threshold draws |")
    A(f"| low ({R_SILENT}–{R_LOW} Hz) | {_pct(rf['low'])} | sparse asynchronous |")
    A(f"| **useful ({R_LOW}–{R_SEIZE} Hz)** | **{_pct(rf['useful'])}** | workable SBI yield |")
    A(f"| seizing (>{R_SEIZE} Hz) | {_pct(rf['seizing'])} | supercritical gain |")
    A("")
    A(f"- mean_FR: median **{roll['fr_median']:.3f}**, mean **{roll['fr_mean']:.3f}**, "
      f"max **{roll['fr_max']:.2f}** Hz")
    A(f"- bimodality (mean/median): **{roll['bimodality_mean_over_median']:.0f}×** "
      f"(>>1 ⇒ silent-mass + active-tail split)")
    A(f"- **corr(σ, log FR) = {roll['corr_sigma_logFR']:+.2f}**, "
      f"**corr(I_inj, log FR) = {roll['corr_iinj_logFR']:+.2f}**  "
      f"(≈0 ⇒ recurrence-driven, not noise-driven)")
    A(f"- σ sampled range: {roll['sigma_range'][0]:.1f}–{roll['sigma_range'][1]:.1f} mV "
      f"(η = gap/σ = {GAP_MV}/σ)")
    A(f"- burst proxy (relaxed ratio≥2.5, prom≥0.4): fired in "
      f"**{roll['sims_with_burst_proxy']}** sims, {roll['any_bursts']} total peaks")
    if roll.get('seizing_median_sigma') is not None:
        A(f"- seizing tail median σ = **{roll['seizing_median_sigma']:.1f} mV**")
    A("")

    if 'topology' in roll:
        t = roll['topology']
        A("## 4. Topological properties\n")
        A(f"- mean in-degree **K = {t['mean_in_degree']:.1f}** "
          f"(median {t['median_in_degree']:.1f})")
        A(f"- neurons with zero inputs: **{_pct(t['frac_zero_in'])}**")
        A(f"- mean # synapses / topology: {t['mean_n_syn']:.0f}")
        A(f"- largest weakly-connected component: **{_pct(t['mean_largest_component_frac'])}** of network")
        A(f"- edge reciprocity (i→j and j→i): {t['mean_reciprocity']:.2f}")
        A(f"- mean connection distance: {t['mean_conn_distance']:.0f} µm")
        A("")

    A("## 5. Recommended next actions\n")
    A("1. If startup-artifact fraction >10%: randomise V(0) + warm-up discard (data quality).")
    A("2. If no bursts: raise C_m on one seizing topology to test τ_m as the burst-blocker.")
    A("3. If σ/I_inj uncorrelated with rate: ensure the SBI feature vector x includes "
      "across_cell_rate_cv + mean_isi_cv (else those axes are unidentifiable).")
    A("4. Re-run find_network_bursts.py with --prominence-frac 0.4 --min-amp-ratio 2.5.")
    A("")

    if fig_names:
        A("## 6. Figures\n")
        for f in fig_names:
            A(f"- `figures/{f}`")
        A("")

    A("## 7. Per-topology table (first 30)\n")
    A("| topo | task | Nn | n_syn | K_in | %zero_in | comp_frac | recip | d_mean(µm) | n_iter |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for tr in topos[:30]:
        A(f"| {tr.topo_idx} | {tr.task} | {tr.Nn} | {tr.n_syn} | "
          f"{tr.mean_in_degree:.1f} | {_pct(tr.frac_zero_in)} | "
          f"{tr.largest_component_frac:.2f} | {tr.reciprocity:.2f} | "
          f"{tr.mean_conn_distance:.0f} | {tr.n_iters} |")
    A("")

    out.write_text("\n".join(lines))


# ======================================================================
# Driver
# ======================================================================
def analyze_campaign(campaign: Path, out_dir: Path,
                     max_iters_per_topo: int = 0,
                     topo_sample: int = 0) -> Dict:
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
        else:
            tr, extra = ta
            # keep the connectivity arrays of the LARGEST topology for figures
            if topo_extra_repr is None or tr.n_syn > topo_extra_repr.get('_n_syn', -1):
                topo_extra_repr = dict(extra); topo_extra_repr['_n_syn'] = tr.n_syn

        Nn = int(meta.get('Nn', tr.Nn if tr else 0))
        T  = float(meta.get('T', tr.T if tr else 0.0))
        npzs = sorted(td.glob('iter_*.npz'))
        if max_iters_per_topo:
            npzs = npzs[:max_iters_per_topo]
        n_iter_here = 0
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
            rec = _analyze_iter(d, sc, Nn, T,
                                tr.topo_idx if tr else -1, iter_idx)
            if rec is None:
                skipped['iters'] += 1
                continue
            iters.append(rec)
            n_iter_here += 1
        if tr is not None:
            tr.n_iters = n_iter_here
            topos.append(tr)

    roll = _roll_up(iters, topos)
    figs = _make_figures(iters, topos, topo_extra_repr, fig_dir)
    elapsed = time.time() - t0

    _write_markdown(out_dir / 'campaign_report.md', campaign, roll,
                    iters, topos, figs, skipped, elapsed)
    summary = dict(campaign=str(campaign), roll_up=roll, skipped=skipped,
                   elapsed_s=elapsed, figures=[Path(f).name for f in figs],
                   per_topology=[asdict(t) for t in topos])
    (out_dir / 'campaign_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


# ======================================================================
# Smoke test
# ======================================================================
def _smoke_test() -> int:
    import tempfile
    print("[smoke] building synthetic campaign ...")
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    Nn, T = 200, 60.0
    camp = tmp / 'campaign_smoke'
    td = camp / 'sweep_cpu_task0000' / 'topo_00000'
    td.mkdir(parents=True)

    # synthetic topology: weibull-ish, mean K ~ 15
    n_syn = Nn * 15
    S_i = rng.integers(0, Nn, n_syn).astype(np.int32)
    S_j = rng.integers(0, Nn, n_syn).astype(np.int32)
    keep = S_i != S_j
    S_i, S_j = S_i[keep], S_j[keep]
    N_pos = rng.uniform(0, 700, (Nn, 2))
    np.savez_compressed(td / 'topology.npz', N_pos=N_pos, S_i=S_i, S_j=S_j,
                        A_pos=np.empty((0, 2)))
    (td / 'topology_meta.json').write_text(json.dumps(
        dict(Nn=Nn, Na=0, T=T, mode='Neuronal', topo_idx=0,
             conn_rule='weibull', p0_conn=0.6, d0_conn=100.0, beta_conn=1.0,
             c_max=700.0)))

    # synthetic iters spanning regimes
    specs = [('silent', 0.0), ('one_spike', 1.0), ('useful', 1.5), ('seize', 8.0)]
    for k, (name, rate) in enumerate(specs):
        params = np.zeros(35)
        params[IDX['Sigma']] = rng.uniform(1, 12)
        params[IDX['I_inj']] = rng.uniform(1, 8)
        params[IDX['g_ampa']] = rng.uniform(0.05, 1.0)
        if name == 'one_spike':
            spk_t = rng.uniform(0, 0.05, Nn); spk_i = np.arange(Nn)
        elif rate == 0.0:
            spk_t = np.empty(0); spk_i = np.empty(0, int)
        else:
            n = int(rate * Nn * T)
            spk_t = np.sort(rng.uniform(0, T, n))
            spk_i = rng.integers(0, Nn, n)
        np.savez_compressed(td / f'iter_{k:05d}.npz',
                            spk_N_t=spk_t.astype(np.float32),
                            spk_N_i=spk_i.astype(np.int32),
                            spk_A_t=np.empty(0, np.float32),
                            spk_A_i=np.empty(0, np.int32),
                            params=params, theta=params)

    summary = analyze_campaign(camp, tmp / 'report')
    r = summary['roll_up']
    print(f"[smoke] iters={r['n_iters']} topos={r['n_topologies']} "
          f"K_in={r['topology']['mean_in_degree']:.1f} "
          f"artifact={r['one_spike_artifact_frac']:.2f}")
    ok = (r['n_iters'] == 4 and r['n_topologies'] == 1
          and r['topology']['mean_in_degree'] > 10
          and r['one_spike_artifact_frac'] >= 0.2
          and (tmp / 'report' / 'campaign_report.md').exists()
          and (tmp / 'report' / 'campaign_summary.json').exists())
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
    print(f"  useful-band yield: {100*r['regime_fractions']['useful']:.0f}%")
    print(f"  one-spike artifact: {100*r['one_spike_artifact_frac']:.0f}%")
    print(f"  corr(sigma,logFR)={r['corr_sigma_logFR']:+.2f}  "
          f"corr(I_inj,logFR)={r['corr_iinj_logFR']:+.2f}")
    print(f"  report : {out_dir/'campaign_report.md'}")
    print(f"  summary: {out_dir/'campaign_summary.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
