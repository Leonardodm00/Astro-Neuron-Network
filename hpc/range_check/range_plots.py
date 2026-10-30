#!/usr/bin/env python3
"""
range_plots.py -- figures of a range_check output folder, from the tables that
range_analysis.py wrote. Plotting only: computes nothing scientific.
Spec: specs/SPEC.md, Block 6.

    oat_curves.png      one panel per free axis: median mean rate against the
                        axis value, one line per conn_prob level, min-max band;
                        log x on the sweep's log axes; dotted line = nominal runs
    prior_rates.png     prior sample: histogram of log10(mean rate) and mean rate
                        against conn_prob, coloured by the burst rate
    prior_spearman.png  Spearman rho of each axis with log10(mean rate) and with
                        the burst rate

Usage:  python3 range_plots.py OUT
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

STYLE = dict(dpi=130, cmap='viridis', rate_floor_hz=0.01)


def log_axes(root):
    """Names of the free axes the sweep samples on a log scale (from design.json)."""
    out = set()
    for p in glob.glob(os.path.join(root, '*', 'design.json')):
        d = json.load(open(p))
        out |= {d['param_names'][k] for k in d['log_params']}
    return out


def plot_oat(root, oat, runs, logx, style=STYLE):
    axes_names = sorted(a for a in oat.axis.unique() if a not in ('nominal', 'unmatched'))
    if not axes_names:
        return None
    ncol = 4
    nrow = int(math.ceil(len(axes_names) / float(ncol)))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), squeeze=False)
    levels = sorted(oat.conn_level.dropna().unique())
    colors = plt.get_cmap(style['cmap'])(np.linspace(0.1, 0.85, max(len(levels), 1)))
    # one y scale for every panel, so an axis with no effect looks flat
    ymax = 1.5 * max(float(np.nanmax(oat.rate_max)), float(np.nanmax(runs.mean_rate_hz)), 1.0)
    for ax, name in zip(axs.flat, axes_names):
        for c, lev in zip(colors, levels):
            s = oat[(oat.axis == name) & (oat.conn_level == lev)].sort_values('value')
            ax.plot(s.value, s.rate_median, '-o', color=c, ms=3, label='conn_prob %.2f' % lev)
            ax.fill_between(s.value, s.rate_min, s.rate_max, color=c, alpha=0.2, lw=0)
            nom = runs[(runs.phase == 'oat') & (runs.axis == 'nominal') & (runs.conn_level == lev)]
            if len(nom):
                ax.axhline(float(np.median(nom.mean_rate_hz)), color=c, ls=':', lw=1)
        ax.set_title(name, fontsize=10)
        if name in logx:
            ax.set_xscale('log')
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())   # no crowded minor labels
        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_ylim(0.0, ymax)
        ax.set_ylabel('mean rate [Hz]', fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axs.flat)[len(axes_names):]:
        ax.axis('off')
    axs.flat[0].legend(fontsize=7)
    fig.suptitle('One-axis sweeps around the nominal vector (20 s runs, window after t_skip)', fontsize=11)
    fig.tight_layout()
    path = os.path.join(root, 'oat_curves.png')
    fig.savefig(path, dpi=style['dpi'])
    plt.close(fig)
    return path


def plot_prior(root, runs, style=STYLE):
    p = runs[runs.phase == 'prior']
    if p.empty:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    y = np.log10(p.mean_rate_hz + style['rate_floor_hz'])
    a1.hist(y, bins=30, color='0.4')
    a1.set_xlabel('log10(mean rate [Hz] + %g)' % style['rate_floor_hz'])
    a1.set_ylabel('runs')
    sc = a2.scatter(p.conn_prob, p.mean_rate_hz + style['rate_floor_hz'], c=p.li_burst_rate_per_min.fillna(0),
                    cmap=style['cmap'], s=10)
    a2.set_yscale('log')
    a2.set_xlabel('conn_prob')
    a2.set_ylabel('mean rate [Hz] + %g' % style['rate_floor_hz'])
    fig.colorbar(sc, ax=a2, label='network bursts / min')
    fig.suptitle('Prior sample (%d runs)' % len(p), fontsize=11)
    fig.tight_layout()
    path = os.path.join(root, 'prior_rates.png')
    fig.savefig(path, dpi=style['dpi'])
    plt.close(fig)
    return path


def plot_spearman(root, sp, style=STYLE):
    if sp is None or sp.empty:
        return None
    sp = sp.sort_values('rho_log_rate')
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(sp) + 1.2))
    yy = np.arange(len(sp))
    ax.barh(yy - 0.2, sp.rho_log_rate, height=0.4, label='log10 rate', color='0.3')
    ax.barh(yy + 0.2, sp.rho_burst, height=0.4, label='burst rate', color='0.7')
    ax.set_yticks(yy)
    ax.set_yticklabels(sp.axis, fontsize=8)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel('Spearman rho (prior sample)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(root, 'prior_spearman.png')
    fig.savefig(path, dpi=style['dpi'])
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description='Figures of a range_check output folder.')
    ap.add_argument('root')
    a = ap.parse_args(argv)
    runs = pd.read_csv(os.path.join(a.root, 'runs.csv'))
    oat_p = os.path.join(a.root, 'oat_summary.csv')
    oat = pd.read_csv(oat_p) if os.path.getsize(oat_p) > 1 else pd.DataFrame(columns=['axis'])
    sp_p = os.path.join(a.root, 'prior_spearman.csv')
    sp = pd.read_csv(sp_p) if os.path.getsize(sp_p) > 1 else None
    made = [plot_oat(a.root, oat, runs, log_axes(a.root)) if len(oat) else None,
            plot_prior(a.root, runs), plot_spearman(a.root, sp)]
    for m in made:
        if m:
            print('wrote', m)
    return 0


if __name__ == '__main__':
    sys.exit(main())
