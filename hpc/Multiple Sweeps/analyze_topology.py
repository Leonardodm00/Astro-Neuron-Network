#!/usr/bin/env python3
# =============================================================================
# analyze_topology.py — per-topology metrics + raster plots
#
# Walks one topo_XXXXX/ directory produced by HPC_main_sweep.py and, for every
# iter_*.npz it contains:
#   1. computes a panel of spike-train metrics for neurons (and, when
#      recorded, astrocytes);
#   2. renders raster plots for that iteration's neuronal and astrocytic
#      activity;
#   3. aggregates the per-iter rows into a flat CSV and a JSON summary;
#   4. produces a distributional overview figure across all iters.
#
# All outputs are written directly into the topology folder (same level as
# the iter_*.npz files), with a clear naming convention so they group
# alphabetically apart from the per-iter data:
#
#     topo_XXXXX/
#     ├── topology.npz                 (existing)
#     ├── topology_meta.json           (existing)
#     ├── spatial_layout.png           (existing)
#     ├── iter_00000.npz / .json       (existing)
#     ├── ...
#     ├── raster_neurons_iter_00000.png        ← new
#     ├── raster_astrocytes_iter_00000.png     ← new (if Na > 0)
#     ├── ...
#     ├── analysis_metrics.csv                 ← new
#     ├── analysis_metrics_summary.json        ← new
#     └── analysis_overview.png                ← new
#
# USAGE
#   python analyze_topology.py --topo_dir /path/to/sweep_<JOBID>/topo_00042
#   python analyze_topology.py --topo_dir <dir> --workers 8 --dpi 150
#   python analyze_topology.py --topo_dir <dir> --no_plots     # metrics only
#
# METRICS  (one row per iter_*.npz in analysis_metrics.csv)
#   Activity
#     n_spikes_N, mean_FR_Hz, FR_std_Hz, FR_max_Hz, FR_median_Hz,
#     n_active_neurons, active_fraction_N
#   Regularity
#     mean_ISI_s, median_ISI_s, mean_CV_ISI   (CV of inter-spike intervals)
#   Synchrony
#     pop_Fano_50ms                          (Fano factor of 50 ms bin counts)
#   Network bursting
#     n_network_bursts, mean_burst_duration_s, burst_rate_per_min, mean_IBI_s
#   Astrocyte (only when Na > 0)
#     n_astro_events, astro_event_rate_Hz,
#     n_active_astrocytes, active_fraction_A
#   Provenance (also stamped in the row)
#     iter_idx, topo_idx, conn_prob, seed_run, noise_mode, + 14 params
# =============================================================================

import argparse
import csv
import glob
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import matplotlib
matplotlib.use('Agg')                           # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# =============================================================================
# Optional reuse of HPC_single_run.py raster plotters (for visual consistency).
# If they're not findable, fall back to the equivalent inline implementations
# at the bottom of this section, so the script is fully self-contained.
# =============================================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
try:
    from HPC_single_run import (
        plot_neuronal_raster   as _ext_plot_neurons,
        plot_astrocyte_raster  as _ext_plot_astros,
    )
    _USE_EXT_PLOTTERS = True
except Exception:
    _USE_EXT_PLOTTERS = False


# Inline plotters — mirror HPC_single_run.py's style ---------------------------
_C_NEURON = '#1f4e79'
_C_ASTRO  = '#7b3f00'
_C_GRID   = '#d0d0d0'


def _style_axis(ax, simtime_s, n_units, ylabel, color):
    ax.set_xlim(0.0, simtime_s)
    ax.set_ylim(-0.5, max(n_units - 0.5, 0.5))
    ax.set_xlabel('Time (s)', fontsize=11, labelpad=4)
    ax.set_ylabel(ylabel,    fontsize=11, labelpad=4)
    ax.tick_params(axis='both', labelsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.grid(axis='x', color=_C_GRID, linewidth=0.6, linestyle='--', zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(color)
    ax.spines['bottom'].set_color(color)
    ax.tick_params(colors=color)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)


def _add_rate_trace(ax, spk_t, simtime_s, n_units, color, bin_s, label):
    bins = np.arange(0.0, simtime_s + bin_s, bin_s)
    counts, _ = np.histogram(spk_t, bins=bins)
    rate = counts / (bin_s * max(n_units, 1))
    t_c  = 0.5 * (bins[:-1] + bins[1:])
    ax_r = ax.twinx()
    ax_r.plot(t_c, rate, color=color, alpha=0.45, linewidth=1.2, label=label)
    ax_r.set_ylabel('Mean FR (Hz)', fontsize=9, color=color, labelpad=4)
    ax_r.tick_params(axis='y', labelsize=8, colors=color)
    ax_r.set_ylim(bottom=0.0)
    for s in ('top', 'left', 'bottom'):
        ax_r.spines[s].set_visible(False)
    ax_r.spines['right'].set_color(color)
    return ax_r


def _inline_plot_neurons(spk_t, spk_i, simtime_s, Nn, params, out_path, dpi=150):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    if len(spk_t):
        cmap = plt.get_cmap('Blues')
        norm = plt.Normalize(0, max(Nn - 1, 1))
        ax.scatter(spk_t, spk_i, c=cmap(norm(spk_i)),
                   s=2.5, linewidths=0, alpha=0.7, zorder=3, rasterized=True)
        _add_rate_trace(ax, spk_t, simtime_s, Nn,
                        _C_NEURON, bin_s=1.0, label='Mean FR')
    else:
        ax.text(0.5, 0.5, 'No neuronal spikes detected',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
    _style_axis(ax, simtime_s, Nn, 'Neuron index', _C_NEURON)
    n_spk   = len(spk_t)
    mean_fr = n_spk / (simtime_s * max(Nn, 1))
    ax.text(0.99, 0.97,
            f'{n_spk:,} spikes — mean FR = {mean_fr:.2f} Hz',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=8.5, color=_C_NEURON,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=2))
    title_p = (f'σ={params[0]:.2f} mV   g_AHP={params[1]:.1f} nS   '
               f'g_na={params[12]:.1f}   g_kd={params[13]:.1f}   '
               f'α_syn={params[11]:.2f}')
    ax.set_title(f'Neuronal raster — {Nn} neurons, {simtime_s:.0f} s\n{title_p}',
                 fontsize=10, pad=8, color=_C_NEURON)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _inline_plot_astros(spk_t, spk_i, simtime_s, Na, params, out_path, dpi=150):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    if len(spk_t):
        cmap = plt.get_cmap('Oranges')
        norm = plt.Normalize(0, max(Na - 1, 1))
        ax.scatter(spk_t, spk_i, c=cmap(norm(spk_i)),
                   s=22, marker='D', linewidths=0.4,
                   edgecolors='#4a2000', alpha=0.85,
                   zorder=3, rasterized=True)
        _add_rate_trace(ax, spk_t, simtime_s, Na,
                        _C_ASTRO, bin_s=5.0, label='Event rate')
    else:
        ax.text(0.5, 0.5, 'No astrocyte Ca²⁺ events detected',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
    _style_axis(ax, simtime_s, Na, 'Astrocyte index', _C_ASTRO)
    n_ev   = len(spk_t)
    mean_r = n_ev / (simtime_s * max(Na, 1))
    ax.text(0.99, 0.97,
            f'{n_ev:,} events — mean rate = {mean_r:.4f} Hz',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=8.5, color=_C_ASTRO,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=2))
    title_p = (f'Tau_Ca={params[4]:.1f} s   '
               f'Xi_ampa={params[2]:.2f}   Xi_nmda={params[3]:.2f}   '
               f'α_syn={params[11]:.2f}')
    ax.set_title(f'Astrocyte Ca²⁺ events — {Na} cells, {simtime_s:.0f} s\n{title_p}',
                 fontsize=10, pad=8, color=_C_ASTRO)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_neurons(*args, **kwargs):
    return (_ext_plot_neurons if _USE_EXT_PLOTTERS else _inline_plot_neurons)(*args, **kwargs)

def plot_astros(*args, **kwargs):
    return (_ext_plot_astros if _USE_EXT_PLOTTERS else _inline_plot_astros)(*args, **kwargs)


# =============================================================================
# Metrics
# =============================================================================

PARAM_NAMES = [
    'Sigma', 'g_AHP', 'Xi_ampa', 'Xi_nmda', 'Tau_Ca',
    'U_0_ar', 'U_max', 'U_0_sr', 'Omega_f_sr', 'Omega_f_ar', 'Omega_d',
    'alpha_syn', 'g_na', 'g_kd',
]


def compute_metrics(spk_N_t, spk_N_i, spk_A_t, spk_A_i, Nn, Na, simtime):
    """
    Compute a panel of spike-train metrics for one iteration.

    All times in seconds, all rates in Hz.  NaNs are returned for metrics that
    are mathematically undefined for the given input (e.g. mean CV when no
    neuron has 2+ spikes, mean IBI when fewer than 2 bursts were detected);
    that is the correct behaviour and downstream analyses should treat NaN as
    "no observation" rather than 0.
    """
    m = {}

    # ---- Basic neuronal rates ----
    m['n_spikes_N']   = int(len(spk_N_t))
    m['mean_FR_Hz']   = float(len(spk_N_t)) / (simtime * max(Nn, 1)) if Nn else 0.0

    if Nn > 0 and len(spk_N_t):
        spk_count_per_n = np.bincount(spk_N_i, minlength=Nn)
        per_n_FR        = spk_count_per_n / simtime
        m['n_active_neurons']  = int((per_n_FR > 0.1).sum())     # >0.1 Hz threshold
        m['active_fraction_N'] = float(m['n_active_neurons']) / Nn
        m['FR_std_Hz']         = float(per_n_FR.std())
        m['FR_max_Hz']         = float(per_n_FR.max())
        m['FR_median_Hz']      = float(np.median(per_n_FR))
    else:
        m['n_active_neurons']  = 0
        m['active_fraction_N'] = 0.0
        m['FR_std_Hz']         = 0.0
        m['FR_max_Hz']         = 0.0
        m['FR_median_Hz']      = 0.0

    # ---- ISI distribution & regularity (CV per neuron, then averaged) ----
    all_isis = []
    per_neuron_cv = []
    if Nn > 0 and len(spk_N_t) >= 2:
        sort_order = np.argsort(spk_N_i, kind='stable')
        spk_i_s    = spk_N_i[sort_order]
        spk_t_s    = spk_N_t[sort_order]
        unique_ns, starts = np.unique(spk_i_s, return_index=True)
        starts = np.append(starts, len(spk_i_s))
        for k in range(len(unique_ns)):
            t_n = np.sort(spk_t_s[starts[k]:starts[k+1]])
            if len(t_n) >= 2:
                isis = np.diff(t_n)
                all_isis.append(isis)
                if len(isis) >= 2 and isis.mean() > 0:
                    per_neuron_cv.append(isis.std() / isis.mean())

    if all_isis:
        all_isis = np.concatenate(all_isis)
        m['mean_ISI_s']   = float(all_isis.mean())
        m['median_ISI_s'] = float(np.median(all_isis))
    else:
        m['mean_ISI_s']   = float('nan')
        m['median_ISI_s'] = float('nan')
    m['mean_CV_ISI'] = float(np.mean(per_neuron_cv)) if per_neuron_cv else float('nan')

    # ---- Population synchrony — Fano factor of bin counts ----
    # Fano = var/mean. For a Poisson population, Fano ≈ 1. Fano >> 1 indicates
    # bursty/synchronized activity; Fano << 1 indicates regular tonic firing.
    bin_s = 0.05            # 50 ms
    if simtime > bin_s and Nn > 0:
        bins = np.arange(0.0, simtime + bin_s, bin_s)
        pop_counts, _ = np.histogram(spk_N_t, bins=bins)
        if pop_counts.mean() > 0:
            m['pop_Fano_50ms'] = float(pop_counts.var() / pop_counts.mean())
        else:
            m['pop_Fano_50ms'] = 0.0
    else:
        m['pop_Fano_50ms'] = float('nan')

    # ---- Network burst detection ----
    # Standard threshold method: bin population activity in 25 ms windows,
    # smooth, then count contiguous regions where the per-neuron rate
    # exceeds mean + 2*std.  This is intentionally simple — for a single
    # paper-quality detection (e.g. Pasquale / Chiappalone style) you'd want
    # something fancier (e.g. ISIN), but as a regime descriptor across
    # thousands of sims this is robust enough.
    if Nn > 0 and len(spk_N_t) > 10 and simtime > 1.0:
        bin_b  = 0.025
        bins_b = np.arange(0.0, simtime + bin_b, bin_b)
        pop_b, _ = np.histogram(spk_N_t, bins=bins_b)
        pop_rate = pop_b / (bin_b * Nn)
        thresh = pop_rate.mean() + 2.0 * pop_rate.std()
        above  = pop_rate > thresh
        if above.any():
            d = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
            burst_starts = np.where(d ==  1)[0]
            burst_ends   = np.where(d == -1)[0]
            n_bursts  = len(burst_starts)
            durations = (burst_ends - burst_starts) * bin_b
            m['n_network_bursts']      = int(n_bursts)
            m['mean_burst_duration_s'] = float(durations.mean()) if n_bursts else 0.0
            m['burst_rate_per_min']    = float(n_bursts / (simtime / 60.0))
            if n_bursts >= 2:
                bs_t = burst_starts * bin_b
                m['mean_IBI_s'] = float(np.diff(bs_t).mean())
            else:
                m['mean_IBI_s'] = float('nan')
        else:
            m['n_network_bursts']      = 0
            m['mean_burst_duration_s'] = 0.0
            m['burst_rate_per_min']    = 0.0
            m['mean_IBI_s']            = float('nan')
    else:
        m['n_network_bursts']      = 0
        m['mean_burst_duration_s'] = 0.0
        m['burst_rate_per_min']    = 0.0
        m['mean_IBI_s']            = float('nan')

    # ---- Astrocyte metrics ----
    if Na > 0:
        m['n_astro_events']      = int(len(spk_A_t))
        m['astro_event_rate_Hz'] = float(len(spk_A_t)) / (simtime * Na)
        if len(spk_A_t):
            count_per_a = np.bincount(spk_A_i, minlength=Na)
            m['n_active_astrocytes'] = int((count_per_a > 0).sum())
            m['active_fraction_A']   = float(m['n_active_astrocytes']) / Na
        else:
            m['n_active_astrocytes'] = 0
            m['active_fraction_A']   = 0.0
    else:
        m['n_astro_events']      = 0
        m['astro_event_rate_Hz'] = 0.0
        m['n_active_astrocytes'] = 0
        m['active_fraction_A']   = 0.0

    return m


# =============================================================================
# Worker — one iter_*.npz at a time
# =============================================================================

def _process_one_iter(work_args):
    """
    Load one iter_*.npz, compute metrics, render rasters.  Errors are
    caught and returned as a dict with an `_error` key so the main loop can
    skip the bad file without taking the whole analysis down.
    """
    (iter_path, topo_dir, Nn, Na, simtime, dpi, do_plots) = work_args
    try:
        with np.load(iter_path) as z:
            params     = np.asarray(z['params'], dtype=np.float64)
            conn_prob  = float(z['conn_prob'])
            topo_idx   = int(z['topo_idx'])
            seed_run   = int(z['seed_run'])
            noise_mode = str(z['noise_mode']) if 'noise_mode' in z.files else 'unknown'
            spk_N_t    = np.asarray(z['spk_N_t'])
            spk_N_i    = np.asarray(z['spk_N_i'])
            spk_A_t    = np.asarray(z['spk_A_t'])
            spk_A_i    = np.asarray(z['spk_A_i'])

        # Recover iter_idx from the filename (iter_00007.npz → 7)
        base = os.path.splitext(os.path.basename(iter_path))[0]
        try:
            iter_idx = int(base.split('_')[-1])
        except ValueError:
            iter_idx = -1

        # ---- Metrics ----
        metrics = compute_metrics(
            spk_N_t, spk_N_i, spk_A_t, spk_A_i, Nn, Na, simtime,
        )

        # ---- Flat row: provenance + 14 params + metrics ----
        row = {
            'iter_idx':   iter_idx,
            'topo_idx':   topo_idx,
            'conn_prob':  conn_prob,
            'seed_run':   seed_run,
            'noise_mode': noise_mode,
        }
        for name, val in zip(PARAM_NAMES, params):
            row[name] = float(val)
        row.update(metrics)

        # ---- Raster plots ----
        if do_plots:
            n_path = os.path.join(
                topo_dir, f'raster_neurons_iter_{iter_idx:05d}.png')
            plot_neurons(spk_N_t, spk_N_i, simtime, Nn, params, n_path, dpi=dpi)
            if Na > 0:
                a_path = os.path.join(
                    topo_dir, f'raster_astrocytes_iter_{iter_idx:05d}.png')
                plot_astros(spk_A_t, spk_A_i, simtime, Na, params, a_path, dpi=dpi)

        return row

    except Exception as e:
        return {
            '_error':       str(e),
            '_iter_path':   iter_path,
            '_traceback':   traceback.format_exc(),
        }


# =============================================================================
# Overview figure
# =============================================================================

def plot_overview(rows, topo_meta, out_path, dpi):
    """
    2×3 grid of histograms summarising the distribution of key metrics
    across all completed iterations in this topology.  Useful for spotting
    bimodality (e.g. quiescent vs bursting regimes coexisting in the
    parameter sweep at this fixed conn_prob).
    """
    Na_topo = int(topo_meta.get('Na') or 0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor('white')

    def _hist(ax, key, color, label):
        vals = np.array([r.get(key) for r in rows if r.get(key) is not None],
                        dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            ax.text(0.5, 0.5, f'no data\n({label})',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
        else:
            ax.hist(vals, bins=min(30, max(5, len(vals) // 2)),
                    color=color, alpha=0.75, edgecolor='white')
            ax.axvline(np.median(vals), color='k', linestyle=':',
                       linewidth=1.0, alpha=0.7,
                       label=f'median = {np.median(vals):.3g}')
            ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
        ax.set_title(label, fontsize=10, color=color)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    _hist(axes[0, 0], 'mean_FR_Hz',        _C_NEURON, 'Mean neuronal FR (Hz)')
    _hist(axes[0, 1], 'pop_Fano_50ms',     _C_NEURON, 'Population Fano factor\n(50 ms bins)')
    _hist(axes[0, 2], 'n_network_bursts',  _C_NEURON, 'Number of network bursts')
    _hist(axes[1, 0], 'mean_CV_ISI',       _C_NEURON, 'Mean CV of ISI')
    _hist(axes[1, 1], 'active_fraction_N', _C_NEURON, 'Fraction of active neurons')

    if Na_topo > 0:
        _hist(axes[1, 2], 'astro_event_rate_Hz', _C_ASTRO,
              'Mean astrocyte event rate (Hz)')
    else:
        axes[1, 2].text(0.5, 0.5, 'no astrocytes recorded',
                        ha='center', va='center',
                        transform=axes[1, 2].transAxes,
                        fontsize=10, color='gray')
        axes[1, 2].set_title('Astrocyte event rate', fontsize=10,
                             color=_C_ASTRO)
        for s in ('top', 'right'):
            axes[1, 2].spines[s].set_visible(False)

    topo_idx  = topo_meta.get('topo_idx', '?')
    conn_prob = topo_meta.get('conn_prob', float('nan'))
    Nn_topo   = topo_meta.get('Nn', '?')
    fig.suptitle(
        f'topo_{topo_idx:05d}  •  conn_prob = {conn_prob:.4f}  •  '
        f'{Nn_topo} N / {Na_topo} A  •  {len(rows)} iterations analysed',
        fontsize=12, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Summary JSON
# =============================================================================

def _make_summary(rows, topo_meta):
    """Aggregate per-iter rows into topology-level distribution statistics."""
    def stats(key):
        vals = np.array([r.get(key) for r in rows if r.get(key) is not None],
                        dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return {'mean': None, 'std': None, 'median': None,
                    'q25': None, 'q75': None, 'min': None, 'max': None, 'n': 0}
        return {
            'mean':   float(vals.mean()),
            'std':    float(vals.std()),
            'median': float(np.median(vals)),
            'q25':    float(np.quantile(vals, 0.25)),
            'q75':    float(np.quantile(vals, 0.75)),
            'min':    float(vals.min()),
            'max':    float(vals.max()),
            'n':      int(len(vals)),
        }

    summary = {
        'topo_idx':              topo_meta.get('topo_idx'),
        'conn_prob':             topo_meta.get('conn_prob'),
        'Nn':                    topo_meta.get('Nn'),
        'Na':                    topo_meta.get('Na'),
        'simtime_s':             topo_meta.get('simtime_s'),
        'n_iterations_analyzed': len(rows),
        'metric_distributions':  {},
    }
    keys = [
        'mean_FR_Hz', 'FR_std_Hz', 'FR_max_Hz', 'active_fraction_N',
        'mean_ISI_s', 'mean_CV_ISI',
        'pop_Fano_50ms',
        'n_network_bursts', 'mean_burst_duration_s',
        'burst_rate_per_min', 'mean_IBI_s',
        'astro_event_rate_Hz', 'active_fraction_A',
    ]
    for k in keys:
        summary['metric_distributions'][k] = stats(k)
    return summary


def _write_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =============================================================================
# Entry point
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Per-topology analysis: compute spike-train metrics and render '
            'raster plots for every iter_*.npz in one topo_XXXXX/ folder.\n\n'
            'All outputs are saved directly into the topology folder.'
        ),
    )
    p.add_argument('--topo_dir', required=True,
                   help='Path to one topo_XXXXX/ subdirectory of a sweep.')
    p.add_argument('--workers',  type=int, default=1,
                   help='Process iterations in parallel using N spawn workers. '
                        '(Default: %(default)s)')
    p.add_argument('--dpi',      type=int, default=150,
                   help='DPI for saved figures. (Default: %(default)s)')
    p.add_argument('--no_plots', action='store_true',
                   help='Compute metrics only; skip raster + overview rendering.')
    return p


def main():
    args = build_parser().parse_args()
    topo_dir = os.path.abspath(args.topo_dir)
    if not os.path.isdir(topo_dir):
        sys.exit(f'[error] not a directory: {topo_dir}')

    # ---- Topology metadata ----
    meta_path = os.path.join(topo_dir, 'topology_meta.json')
    if not os.path.isfile(meta_path):
        sys.exit(f'[error] missing topology_meta.json in {topo_dir}')
    with open(meta_path) as f:
        topo_meta = json.load(f)

    Nn       = int(topo_meta.get('Nn', 100))
    Na       = int(topo_meta.get('Na',   0))
    simtime  = float(topo_meta.get('simtime_s', 180.0))
    do_plots = not args.no_plots

    # ---- Discover iterations ----
    iter_files = sorted(glob.glob(os.path.join(topo_dir, 'iter_*.npz')))
    if not iter_files:
        sys.exit(f'[error] no iter_*.npz files in {topo_dir}')

    print(f'[main] topo_dir        : {topo_dir}')
    print(f'[main] iter files      : {len(iter_files)}')
    print(f'[main] Nn / Na / sim   : {Nn} / {Na} / {simtime:.1f} s')
    print(f'[main] conn_prob       : {topo_meta.get("conn_prob"):.4f}')
    print(f'[main] do_plots        : {do_plots}')
    print(f'[main] workers         : {args.workers}')
    print(f'[main] raster style    : '
          f'{"HPC_single_run import" if _USE_EXT_PLOTTERS else "inline"}',
          flush=True)

    work_items = [
        (p, topo_dir, Nn, Na, simtime, args.dpi, do_plots)
        for p in iter_files
    ]

    rows   = []
    errors = []

    if args.workers <= 1:
        # Serial path — easier to debug, no spawn overhead for small sweeps.
        for w in work_items:
            r = _process_one_iter(w)
            if '_error' in r:
                errors.append(r)
            else:
                rows.append(r)
                print(f'  iter {r["iter_idx"]:05d}: '
                      f'FR={r["mean_FR_Hz"]:6.2f} Hz  '
                      f'bursts={r["n_network_bursts"]:3d}  '
                      f'Fano={r["pop_Fano_50ms"]:6.2f}',
                      flush=True)
    else:
        # Parallel path — spawn context to avoid fork-time matplotlib pitfalls.
        from multiprocessing import get_context
        ctx = get_context('spawn')
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=ctx) as ex:
            futures = {ex.submit(_process_one_iter, w): w for w in work_items}
            for fut in as_completed(futures):
                r = fut.result()
                if '_error' in r:
                    errors.append(r)
                else:
                    rows.append(r)
                    print(f'  iter {r["iter_idx"]:05d}: '
                          f'FR={r["mean_FR_Hz"]:6.2f} Hz  '
                          f'bursts={r["n_network_bursts"]:3d}  '
                          f'Fano={r["pop_Fano_50ms"]:6.2f}',
                          flush=True)

    if errors:
        print(f'\n[!] {len(errors)} iter files could not be processed:')
        for e in errors[:5]:
            print(f'    {e["_iter_path"]}:  {e["_error"]}')
        if len(errors) > 5:
            print(f'    ... and {len(errors) - 5} more')

    if not rows:
        sys.exit('[error] no valid metrics computed; refusing to write empty CSV')

    rows.sort(key=lambda r: r['iter_idx'])

    # ---- Persist ----
    csv_path     = os.path.join(topo_dir, 'analysis_metrics.csv')
    _write_csv(rows, csv_path)
    print(f'\n[wrote] {csv_path}')

    summary      = _make_summary(rows, topo_meta)
    summary_path = os.path.join(topo_dir, 'analysis_metrics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'[wrote] {summary_path}')

    if do_plots:
        overview_path = os.path.join(topo_dir, 'analysis_overview.png')
        plot_overview(rows, topo_meta, overview_path, dpi=args.dpi)
        print(f'[wrote] {overview_path}')

    print(f'\n[done] analysed {len(rows)} iterations in {topo_dir}')


if __name__ == '__main__':
    main()
