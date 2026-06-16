#!/usr/bin/env python3
"""
plot_topology_rasters.py — raster-plot all iterations within a single topology.

For every iter_*.npz found under `topo_dir` (produced by HPC_main_sweep.py) the
script saves:
    raster_neuron_iter{IDX:05d}.png     neuronal spike raster + mean-FR trace
    raster_astro_iter{IDX:05d}.png      astrocyte Ca²⁺ raster + event-rate trace
                                         (only when MODE=Full and Na > 0)

Plus, when --overview is set (default), a single summary grid:
    raster_overview.png    compact grid of all neuronal rasters, one panel per
                           iteration, sorted by iter_idx.

The axis style (colours, tick helpers, rate-trace overlay) is imported directly
from HPC_single_run.py so all outputs are visually consistent with the
single-run plots.

Within each figure title the four most diagnostic axes for this model are shown:
    σ    (Sigma,  idx  0) — per-neuron noise amplitude [mV]
    I_inj(        idx 34) — per-neuron excitability spread SCALE [pA]
    g_ampa          idx 14) — AMPA synaptic conductance [nS]
    g_nmda          idx 15) — NMDA synaptic conductance [nS]
along with the three SBI discriminating statistics from the matching sidecar
(across_cell_rate_cv, mean_isi_cv, frac_active).

Usage
-----
    # from the campaign directory:
    python plot_topology_rasters.py campaign_cadex_hhgap_v1/sweep_egeos_task0000/topo_00042

    # custom output dir, higher DPI, limit to first 20 iters:
    python plot_topology_rasters.py topo_00042 --out_dir ./figs --dpi 200 --max-iters 20

    # skip the overview grid:
    python plot_topology_rasters.py topo_00042 --no-overview

Smoke test
----------
    python plot_topology_rasters.py --smoke-test
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Import style helpers + PARAM_NAMES from the production single-run module.
# They live at module top-level in HPC_single_run.py and are safe to import
# without triggering any simulation code.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from HPC_single_run import (
        _style_axis, _add_rate_trace,
        _C_NEURON, _C_ASTRO,
        PARAM_NAMES, PARAM_UNITS,
    )
except ImportError as _e:
    raise ImportError(
        "Cannot import style helpers from HPC_single_run.py. "
        "Ensure HPC_single_run.py is in the same directory as this script "
        f"(looked in {_HERE}).\n  Original error: {_e}"
    )


# ---------------------------------------------------------------------------
# Parameter indices that appear in every figure title
# ---------------------------------------------------------------------------
_IDX_SIGMA  = PARAM_NAMES.index('Sigma')    # 0
_IDX_I_INJ  = PARAM_NAMES.index('I_inj')   # 34
_IDX_G_AMPA = PARAM_NAMES.index('g_ampa')  # 14
_IDX_G_NMDA = PARAM_NAMES.index('g_nmda')  # 15
_IDX_GBAR_A = PARAM_NAMES.index('gbarA')   # 1


# =============================================================================
# I/O helpers
# =============================================================================

def _read_topo_meta(topo_dir: Path) -> Dict:
    """
    Load topology_meta.json from `topo_dir`.  Falls back to graceful defaults
    if the file is absent (old data, or a topo_dir from HPC_single_run.py).

    Returns a dict with at least: Nn, Na, simtime_s, mode.
    """
    meta_path = topo_dir / 'topology_meta.json'
    if meta_path.exists():
        with open(meta_path) as fh:
            return json.load(fh)
    # Fallback: infer from the first iter_*.json sidecar
    sidecars = sorted(topo_dir.glob('iter_*.json'))
    if sidecars:
        with open(sidecars[0]) as fh:
            sc = json.load(fh)
        return {
            'Nn':        sc.get('Nn', 100),
            'Na':        sc.get('Na', 0),
            'simtime_s': sc.get('simtime_s', sc.get('simtime', 10.0)),
            'mode':      sc.get('mode', 'Neuronal'),
        }
    return {'Nn': 100, 'Na': 0, 'simtime_s': 10.0, 'mode': 'Neuronal'}


def _load_iter(npz_path: Path) -> Tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
    """
    Load spike arrays and parameter vector from one iter_*.npz.

    Returns
    -------
    spk_N_t : float32 (n_spikes,)  neuronal spike times [s]
    spk_N_i : int32   (n_spikes,)  neuron indices
    spk_A_t : float32 (n_events,)  astrocyte event times [s]
    spk_A_i : int32   (n_events,)  astrocyte indices
    params  : float64 (35,)        natural-unit parameter vector
    theta   : float64 (35,)        inference-coordinate label
    """
    data    = np.load(npz_path)
    spk_N_t = np.asarray(data['spk_N_t'], dtype=np.float32)
    spk_N_i = np.asarray(data['spk_N_i'], dtype=np.int32)
    spk_A_t = np.asarray(data['spk_A_t'], dtype=np.float32) \
              if 'spk_A_t' in data else np.array([], dtype=np.float32)
    spk_A_i = np.asarray(data['spk_A_i'], dtype=np.int32) \
              if 'spk_A_i' in data else np.array([], dtype=np.int32)
    params  = np.asarray(data['params'],  dtype=np.float64)
    theta   = np.asarray(data['theta'],   dtype=np.float64)
    return spk_N_t, spk_N_i, spk_A_t, spk_A_i, params, theta


def _read_sidecar(json_path: Path) -> Dict:
    """Load iter_*.json sidecar; return {} on any read / parse error."""
    try:
        with open(json_path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _iter_pairs(topo_dir: Path,
                max_iters: Optional[int] = None) -> List[Tuple[Path, Path]]:
    """
    Return sorted (npz_path, json_path) pairs for all iter_*.npz files.
    json_path may not exist (older runs); the caller handles missing sidecars.
    """
    npzs = sorted(topo_dir.glob('iter_*.npz'))
    if max_iters is not None:
        npzs = npzs[:max_iters]
    pairs = []
    for npz in npzs:
        pairs.append((npz, npz.with_suffix('.json')))
    return pairs


# =============================================================================
# Title builder
# =============================================================================

def _make_title(params: np.ndarray, sc: Dict,
                topo_idx: int, iter_idx: int,
                Nn: int, simtime_s: float,
                short: bool = False) -> str:
    """
    Build the figure title string.

    Parameters
    ----------
    params    : natural-unit parameter vector (35-D)
    sc        : sidecar dict (may be empty)
    short     : compact one-liner for the overview-grid panels
    """
    sigma  = float(params[_IDX_SIGMA])
    i_inj  = float(params[_IDX_I_INJ])
    g_ampa = float(params[_IDX_G_AMPA])
    g_nmda = float(params[_IDX_G_NMDA])
    gbar_a = float(params[_IDX_GBAR_A])

    mfr    = sc.get('mean_FR_Hz',         None)
    cv_r   = sc.get('across_cell_rate_cv', None)
    isi_cv = sc.get('mean_isi_cv',         None)
    frac   = sc.get('frac_active',         None)

    if short:
        # Single compact line for overview grid panels
        stats = (f'FR={mfr:.2f} Hz' if mfr is not None else '')
        return (f'T{topo_idx:04d}-I{iter_idx:04d}  '
                f'σ={sigma:.1f}  I_inj={i_inj:.1f}  '
                f'g_ampa={g_ampa:.1f}  {stats}')

    # Two-line title for individual figures
    line1 = (f'Topo {topo_idx:05d}  |  Iter {iter_idx:05d}  '
             f'|  {Nn} neurons  |  {simtime_s:.0f} s')
    line2 = (f'σ={sigma:.2f} mV   I_inj={i_inj:.1f} pA   '
             f'g_ampa={g_ampa:.2f} nS   g_nmda={g_nmda:.2f} nS   '
             f'ḡ_A={gbar_a:.1f} nS')
    if any(v is not None for v in (mfr, cv_r, isi_cv, frac)):
        parts = []
        if mfr    is not None: parts.append(f'FR={mfr:.3f} Hz')
        if cv_r   is not None: parts.append(f'CV_rate={cv_r:.3f}')
        if frac   is not None: parts.append(f'frac_act={frac:.2f}')
        if isi_cv is not None: parts.append(f'ISI_CV={isi_cv:.3f}')
        line2 += '\n' + '   '.join(parts)
    return f'{line1}\n{line2}'


# =============================================================================
# Axes-level drawers  (reuse production style helpers)
# =============================================================================

def _draw_neuron_raster(ax: plt.Axes,
                        spk_N_t: np.ndarray, spk_N_i: np.ndarray,
                        simtime_s: float, Nn: int) -> None:
    """Draw neuronal spikes on `ax` using the production scatter style."""
    if len(spk_N_t):
        cmap = plt.get_cmap('Blues')
        norm = plt.Normalize(0, max(Nn - 1, 1))
        ax.scatter(spk_N_t, spk_N_i,
                   c=cmap(norm(spk_N_i)),
                   s=2.5, linewidths=0, alpha=0.7,
                   zorder=3, rasterized=True)
        _add_rate_trace(ax, spk_N_t, simtime_s, Nn,
                        _C_NEURON, bin_s=1.0, label='Mean FR')
    else:
        ax.text(0.5, 0.5, 'No neuronal spikes detected',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='gray')

    n_spk  = len(spk_N_t)
    mfr    = n_spk / (simtime_s * max(Nn, 1))
    ax.text(0.99, 0.97,
            f'{n_spk:,} spikes  |  mean FR = {mfr:.3f} Hz',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=8, color=_C_NEURON,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=2))

    _style_axis(ax, simtime_s, Nn, 'Neuron index', _C_NEURON)


def _draw_astro_raster(ax: plt.Axes,
                       spk_A_t: np.ndarray, spk_A_i: np.ndarray,
                       simtime_s: float, Na: int) -> None:
    """Draw astrocyte Ca²⁺ events on `ax` using the production scatter style."""
    if len(spk_A_t):
        cmap = plt.get_cmap('Oranges')
        norm = plt.Normalize(0, max(Na - 1, 1))
        ax.scatter(spk_A_t, spk_A_i,
                   c=cmap(norm(spk_A_i)),
                   s=22, marker='D', linewidths=0.4,
                   edgecolors='#4a2000', alpha=0.85,
                   zorder=3, rasterized=True)
        _add_rate_trace(ax, spk_A_t, simtime_s, Na,
                        _C_ASTRO, bin_s=5.0, label='Event rate')
    else:
        ax.text(0.5, 0.5, 'No astrocyte Ca²⁺ events detected',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='gray')

    n_ev  = len(spk_A_t)
    rate  = n_ev / (simtime_s * max(Na, 1))
    ax.text(0.99, 0.97,
            f'{n_ev:,} events  |  mean rate = {rate:.4f} Hz',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=8, color=_C_ASTRO,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=2))

    _style_axis(ax, simtime_s, Na, 'Astrocyte index', _C_ASTRO)


# =============================================================================
# Per-iteration figure  (one .png per iter)
# =============================================================================

def _plot_iter_figure(
        spk_N_t: np.ndarray, spk_N_i: np.ndarray,
        spk_A_t: np.ndarray, spk_A_i: np.ndarray,
        params: np.ndarray, sc: Dict,
        Nn: int, Na: int, simtime_s: float, mode: str,
        topo_idx: int, iter_idx: int,
        out_dir: Path, dpi: int,
) -> List[Path]:
    """
    Save raster figure(s) for one iteration.  Returns list of paths written.
    """
    saved = []
    full_mode = (mode == 'Full') and (Na > 0)

    # ── neuronal raster ──────────────────────────────────────────────────────
    n_rows = 2 if full_mode else 1
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(11, 4 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor('white')

    ax_n = axes[0, 0]
    ax_n.set_facecolor('white')
    _draw_neuron_raster(ax_n, spk_N_t, spk_N_i, simtime_s, Nn)
    ax_n.set_title(_make_title(params, sc, topo_idx, iter_idx,
                               Nn, simtime_s, short=False),
                   fontsize=9.5, pad=8, color=_C_NEURON,
                   loc='left')

    if full_mode:
        ax_a = axes[1, 0]
        ax_a.set_facecolor('white')
        _draw_astro_raster(ax_a, spk_A_t, spk_A_i, simtime_s, Na)
        ax_a.set_title(f'Astrocyte Ca²⁺ events — {Na} cells',
                       fontsize=9.5, pad=8, color=_C_ASTRO, loc='left')

    plt.tight_layout()
    out_path = out_dir / f'raster_iter{iter_idx:05d}.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    saved.append(out_path)
    return saved


# =============================================================================
# Overview grid  (one figure for the whole topology)
# =============================================================================

def _plot_overview_grid(
        records: List[Dict],       # list of dicts with keys below
        topo_idx: int,
        out_dir: Path,
        dpi: int,
        n_cols: int = 4,
) -> Path:
    """
    Save a compact grid of neuronal rasters — one panel per iteration.

    Each element of `records` must have:
        spk_N_t, spk_N_i, params, sc, Nn, simtime_s, iter_idx
    """
    n_iters = len(records)
    if n_iters == 0:
        return None

    n_cols  = min(n_cols, n_iters)
    n_rows  = math.ceil(n_iters / n_cols)
    pw, ph  = 3.8, 2.5                  # panel width/height [inches]
    fig     = plt.figure(figsize=(pw * n_cols, ph * n_rows))
    fig.patch.set_facecolor('white')

    for pos, rec in enumerate(records):
        ax = fig.add_subplot(n_rows, n_cols, pos + 1)
        ax.set_facecolor('white')

        spk_N_t  = rec['spk_N_t']
        spk_N_i  = rec['spk_N_i']
        simtime_s = rec['simtime_s']
        Nn        = rec['Nn']
        params    = rec['params']
        sc        = rec['sc']
        iter_idx  = rec['iter_idx']

        if len(spk_N_t):
            cmap = plt.get_cmap('Blues')
            norm = plt.Normalize(0, max(Nn - 1, 1))
            ax.scatter(spk_N_t, spk_N_i,
                       c=cmap(norm(spk_N_i)),
                       s=1.0, linewidths=0, alpha=0.65,
                       zorder=3, rasterized=True)
        else:
            ax.text(0.5, 0.5, 'silent',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='gray')

        ax.set_xlim(0.0, simtime_s)
        ax.set_ylim(-0.5, max(Nn - 0.5, 0.5))
        ax.tick_params(axis='both', labelsize=6.5)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=4))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color(_C_NEURON)
        ax.spines['bottom'].set_color(_C_NEURON)
        ax.tick_params(colors=_C_NEURON)

        # Label axes only on edge panels to avoid clutter
        col_pos = pos % n_cols
        row_pos = pos // n_cols
        if row_pos == n_rows - 1 or pos == n_iters - 1:
            ax.set_xlabel('t (s)', fontsize=7, color=_C_NEURON, labelpad=2)
        if col_pos == 0:
            ax.set_ylabel('Neuron', fontsize=7, color=_C_NEURON, labelpad=2)

        ax.set_title(
            _make_title(params, sc, topo_idx, iter_idx,
                        Nn, simtime_s, short=True),
            fontsize=6.5, pad=3, color=_C_NEURON, loc='left')

    fig.suptitle(
        f'Topology {topo_idx:05d}  —  {n_iters} iterations  '
        f'(σ · I_inj · g_ampa · g_nmda sweep)',
        fontsize=10, y=1.002, color=_C_NEURON)

    plt.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.4)
    out_path = out_dir / 'raster_overview.png'
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path


# =============================================================================
# Public entry-point
# =============================================================================

def plot_topology_rasters(
        topo_dir: Union[str, Path],
        out_dir:   Optional[Union[str, Path]] = None,
        dpi:       int  = 150,
        overview:  bool = True,
        max_iters: Optional[int] = None,
        n_overview_cols: int = 4,
        verbose:   bool = True,
) -> List[Path]:
    """
    Save raster-plot figures for every iteration in `topo_dir`.

    Parameters
    ----------
    topo_dir : path to a topo_NNNNN/ directory produced by HPC_main_sweep.py,
               e.g. ``campaign_cadex_hhgap_v1/sweep_egeos_task0000/topo_00042``
    out_dir  : where to write .png files.  Defaults to ``topo_dir/figures/``.
    dpi      : raster DPI (150 for screen, 300 for print).
    overview : if True (default) save a single compact-grid overview PNG.
    max_iters: cap on the number of iterations processed (None = all).
    n_overview_cols : columns in the overview grid (default 4).
    verbose  : print progress lines.

    Returns
    -------
    list of Path objects for every .png written.
    """
    topo_dir = Path(topo_dir).resolve()
    if not topo_dir.is_dir():
        raise FileNotFoundError(f'topo_dir not found: {topo_dir}')

    out_dir = Path(out_dir).resolve() if out_dir is not None \
              else topo_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── topology metadata ────────────────────────────────────────────────────
    meta      = _read_topo_meta(topo_dir)
    Nn        = int(meta['Nn'])
    Na        = int(meta.get('Na', 0))
    simtime_s = float(meta['simtime_s'])
    mode      = str(meta.get('mode', 'Neuronal'))
    topo_idx  = int(meta.get('topo_idx', 0))

    if verbose:
        print(f'[plot_topology_rasters] topo_dir : {topo_dir}')
        print(f'  Nn={Nn}  Na={Na}  T={simtime_s}s  mode={mode}  '
              f'topo_idx={topo_idx}')
        print(f'  out_dir  : {out_dir}')

    # ── iterate over iter_*.npz / iter_*.json pairs ──────────────────────────
    pairs    = _iter_pairs(topo_dir, max_iters)
    saved    = []
    records  = []            # for the overview grid

    if not pairs:
        print(f'  WARNING: no iter_*.npz found under {topo_dir}', file=sys.stderr)
        return []

    for npz_path, json_path in pairs:
        # Extract iter_idx from filename (iter_NNNNN.npz)
        stem     = npz_path.stem                  # 'iter_00007'
        iter_idx = int(stem.split('_')[1])

        spk_N_t, spk_N_i, spk_A_t, spk_A_i, params, theta = _load_iter(npz_path)
        sc = _read_sidecar(json_path)

        if verbose:
            mfr = sc.get('mean_FR_Hz', len(spk_N_t) / (simtime_s * max(Nn, 1)))
            print(f'  iter {iter_idx:05d}  '
                  f'spikes={len(spk_N_t):>6,}  '
                  f'mean_FR={mfr:.3f} Hz  '
                  f'σ={params[_IDX_SIGMA]:.2f}  '
                  f'I_inj={params[_IDX_I_INJ]:.1f}')

        # Per-iteration figure
        paths = _plot_iter_figure(
            spk_N_t, spk_N_i, spk_A_t, spk_A_i,
            params, sc,
            Nn=Nn, Na=Na, simtime_s=simtime_s, mode=mode,
            topo_idx=topo_idx, iter_idx=iter_idx,
            out_dir=out_dir, dpi=dpi,
        )
        saved.extend(paths)

        # Accumulate for overview
        if overview:
            records.append(dict(
                spk_N_t=spk_N_t, spk_N_i=spk_N_i,
                params=params, sc=sc,
                Nn=Nn, simtime_s=simtime_s,
                iter_idx=iter_idx,
            ))

    # ── overview grid ─────────────────────────────────────────────────────────
    if overview and records:
        ov_path = _plot_overview_grid(
            records, topo_idx, out_dir, dpi, n_cols=n_overview_cols)
        if ov_path is not None:
            saved.append(ov_path)
            if verbose:
                print(f'  overview → {ov_path}')

    if verbose:
        print(f'  {len(saved)} figure(s) saved to {out_dir}')
    return saved


# =============================================================================
# Smoke test  (--smoke-test flag)
# =============================================================================

def _run_smoke_test(tmp_dir: Optional[Path] = None) -> int:
    """
    Build a synthetic topo_dir that mimics HPC_main_sweep.py output, run
    plot_topology_rasters(), and verify the expected files were created.

    Returns 0 on success, 1 on failure.  Cleans up on success.
    """
    import tempfile, shutil

    root = Path(tempfile.mkdtemp(prefix='smoke_raster_')) if tmp_dir is None \
           else tmp_dir
    root.mkdir(parents=True, exist_ok=True)

    print('Smoke test — plot_topology_rasters')
    print(f'  synthetic topo_dir: {root}')

    # ── build synthetic parameter vectors ──────────────────────────────────
    rng     = np.random.default_rng(42)
    Nn      = 80
    Na      = 0
    T       = 5.0          # s
    n_iters = 6
    N_PARAMS = len(PARAM_NAMES)     # 35

    # topology_meta.json
    with open(root / 'topology_meta.json', 'w') as fh:
        json.dump({'Nn': Nn, 'Na': Na, 'simtime_s': T,
                   'mode': 'Neuronal', 'topo_idx': 7}, fh)

    ok = True
    expected_files = [f'raster_iter{i:05d}.png' for i in range(n_iters)]
    expected_files.append('raster_overview.png')

    for i in range(n_iters):
        # synthetic spike train: Poisson at varying rates
        mean_rate = rng.uniform(0.5, 8.0)    # Hz
        n_spikes  = rng.poisson(mean_rate * Nn * T)
        spk_N_t   = np.sort(rng.uniform(0, T, n_spikes)).astype(np.float32)
        spk_N_i   = rng.integers(0, Nn, n_spikes).astype(np.int32)

        # synthetic parameter vector (35-D, natural units)
        params          = rng.uniform(0.5, 2.0, N_PARAMS)
        params[_IDX_SIGMA]  = rng.uniform(1.0, 10.0)
        params[_IDX_I_INJ]  = rng.uniform(0.0, 50.0)
        params[_IDX_G_AMPA] = rng.uniform(0.1, 15.0)
        params[_IDX_G_NMDA] = rng.uniform(0.1, 15.0)

        theta  = params.copy()   # simplified: identity for smoke test

        # iter_NNNNN.npz
        np.savez_compressed(
            root / f'iter_{i:05d}.npz',
            spk_N_t=spk_N_t, spk_N_i=spk_N_i,
            spk_A_t=np.array([], dtype=np.float32),
            spk_A_i=np.array([], dtype=np.int32),
            params=params, theta=theta,
        )

        # iter_NNNNN.json sidecar
        counts   = np.bincount(spk_N_i, minlength=Nn).astype(float)
        rates    = counts / T
        mu_r     = rates.mean()
        cv_rate  = float(rates.std() / mu_r) if mu_r > 0 else 0.0
        with open(root / f'iter_{i:05d}.json', 'w') as fh:
            json.dump({
                'topo_idx': 7, 'iter_idx': i,
                'params': params.tolist(), 'theta': theta.tolist(),
                'mean_FR_Hz': float(mu_r),
                'across_cell_rate_cv': cv_rate,
                'frac_active': float((rates > 0.1).mean()),
                'mean_isi_cv': float(rng.uniform(0.5, 1.2)),
            }, fh)

    # ── run the function under test ─────────────────────────────────────────
    out_dir = root / 'figures'
    try:
        saved = plot_topology_rasters(
            topo_dir=root,
            out_dir=out_dir,
            dpi=72,
            overview=True,
            verbose=True,
        )
    except Exception as exc:
        import traceback
        print(f'  [FAIL] plot_topology_rasters raised: {exc}')
        traceback.print_exc()
        return 1

    # ── check expected files ─────────────────────────────────────────────────
    for fname in expected_files:
        path = out_dir / fname
        exists = path.is_file() and path.stat().st_size > 0
        print(f'  [{"PASS" if exists else "FAIL"}] {fname}  '
              f'({path.stat().st_size // 1024} KiB)' if exists else
              f'  [FAIL] {fname}  MISSING')
        ok = ok and exists

    if ok:
        shutil.rmtree(root)
        print('\nSmoke test: PASS')
        return 0
    else:
        print(f'\nSmoke test: FAIL  (artefacts kept at {root})')
        return 1


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('topo_dir', nargs='?',
                   help='path to a topo_NNNNN/ directory from HPC_main_sweep.py')
    p.add_argument('--out_dir', default=None,
                   help='output directory for .png files (default: topo_dir/figures/)')
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--no-overview', dest='overview', action='store_false',
                   help='skip the overview grid figure')
    p.add_argument('--max-iters', type=int, default=None,
                   help='process only the first N iterations')
    p.add_argument('--n-cols', type=int, default=4,
                   help='columns in the overview grid (default: 4)')
    p.add_argument('--smoke-test', action='store_true',
                   help='run the built-in smoke test and exit')
    return p


def main() -> int:
    args = _build_parser().parse_args()

    if args.smoke_test:
        return _run_smoke_test()

    if args.topo_dir is None:
        _build_parser().print_help()
        return 1

    plot_topology_rasters(
        topo_dir=args.topo_dir,
        out_dir=args.out_dir,
        dpi=args.dpi,
        overview=args.overview,
        max_iters=args.max_iters,
        n_overview_cols=args.n_cols,
        verbose=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
