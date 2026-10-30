#!/usr/bin/env python3
"""
range_plots.py -- figures of a range_check output folder, from the tables that
range_analysis.py wrote. Plotting only: computes nothing scientific.
Spec: specs/SPEC.md, Block 6.

    corner_effects.png      main effect of every design dimension (upper bound
                            minus lower bound, over the corners) on log10 rate,
                            and on the shares of silent and high-rate runs
    interior_bins.png       one panel per dimension: median rate per bin with
                            the interquartile band (one shared log scale), and
                            the shares of silent and high-rate runs per bin
    rate_distributions.png  histograms of log10 rate for the corners, the
                            centre points and the interior cells

Usage:  python3 range_plots.py OUT
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

STYLE = dict(dpi=130, rate_floor_hz=0.01, c_rate='0.25', c_silent='tab:blue', c_high='tab:red')


def read(root, name):
    p = os.path.join(root, name)
    return pd.read_csv(p) if os.path.isfile(p) and os.path.getsize(p) > 1 else pd.DataFrame()


def plot_corner_effects(root, eff, style=STYLE):
    if eff.empty:
        return None
    eff = eff.sort_values('effect_y')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.32 * len(eff) + 1.6), sharey=True)
    yy = np.arange(len(eff))
    a1.barh(yy, eff.effect_y, color=style['c_rate'])
    a1.set_yticks(yy)
    a1.set_yticklabels(eff.dimension, fontsize=8)
    a1.axvline(0, color='k', lw=0.5)
    a1.set_xlabel('effect on log10(mean rate [Hz] + %g)' % style['rate_floor_hz'])
    a2.barh(yy - 0.2, eff.effect_silent, height=0.4, color=style['c_silent'], label='share silent')
    a2.barh(yy + 0.2, eff.effect_high, height=0.4, color=style['c_high'], label='share high rate')
    a2.axvline(0, color='k', lw=0.5)
    a2.set_xlabel('effect on the share of runs')
    a2.legend(fontsize=8)
    fig.suptitle('Corners: upper bound minus lower bound, averaged over all other corners', fontsize=11)
    fig.tight_layout()
    path = os.path.join(root, 'corner_effects.png')
    fig.savefig(path, dpi=style['dpi'])
    plt.close(fig)
    return path


def plot_interior_bins(root, bins_t, style=STYLE):
    if bins_t.empty:
        return None
    dims = list(dict.fromkeys(bins_t.dimension))
    ncol = 4
    nrow = int(math.ceil(len(dims) / float(ncol)))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), squeeze=False)
    ymax = 1.5 * max(float(bins_t.rate_q75.max()), 1.0)       # one scale: a flat axis looks flat
    for ax, j in zip(axs.flat, dims):
        b = bins_t[bins_t.dimension == j].sort_values('bin')
        ax.plot(b.bin, b.rate_median, '-o', ms=3, color=style['c_rate'])
        ax.fill_between(b.bin, b.rate_q25, b.rate_q75, color=style['c_rate'], alpha=0.2, lw=0)
        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_ylim(0.0, ymax)
        ax.set_title(j, fontsize=10)
        ax.set_xlabel('bin', fontsize=8)
        ax.set_ylabel('mean rate [Hz]', fontsize=8)
        ax.tick_params(labelsize=7)
        tw = ax.twinx()
        tw.plot(b.bin, b.share_silent, '--', color=style['c_silent'], lw=1)
        tw.plot(b.bin, b.share_high, ':', color=style['c_high'], lw=1.2)
        tw.set_ylim(-0.02, 1.02)
        tw.tick_params(labelsize=7)
    for ax in list(axs.flat)[len(dims):]:
        ax.axis('off')
    fig.suptitle('Interior cells per bin: median rate and interquartile band (left); share silent (dashed) '
                 'and share high rate (dotted) (right)', fontsize=10)
    fig.tight_layout()
    path = os.path.join(root, 'interior_bins.png')
    fig.savefig(path, dpi=style['dpi'])
    plt.close(fig)
    return path


def plot_rates(root, runs, style=STYLE):
    if runs.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.log10(runs.mean_rate_hz + style['rate_floor_hz'])
    edges = np.linspace(float(y.min()) - 0.05, float(y.max()) + 0.05, 41)
    for kind, color in (('corner', '0.3'), ('centre', 'tab:orange'), ('interior', 'tab:green')):
        sel = runs.kind == kind
        if sel.any():
            ax.hist(y[sel], bins=edges, histtype='step', lw=1.5, color=color, density=True,
                    label='%s (%d runs)' % (kind, int(sel.sum())))
    ax.set_xlabel('log10(mean rate [Hz] + %g)' % style['rate_floor_hz'])
    ax.set_ylabel('density')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(root, 'rate_distributions.png')
    fig.savefig(path, dpi=style['dpi'])
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description='Figures of a range_check output folder.')
    ap.add_argument('root')
    a = ap.parse_args(argv)
    runs = read(a.root, 'runs.csv')
    made = [plot_corner_effects(a.root, read(a.root, 'corner_effects.csv')),
            plot_interior_bins(a.root, read(a.root, 'interior_bins.csv')),
            plot_rates(a.root, runs)]
    for m in made:
        if m:
            print('wrote', m)
    return 0


if __name__ == '__main__':
    sys.exit(main())
