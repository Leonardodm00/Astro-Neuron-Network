#!/usr/bin/env python3
"""
sic_demo_plots.py -- figures from the outputs of sic_demo.py (SPEC Block 12).
Analysis functions (rates, peri-release histograms, SIC event sizes) are kept
apart from the drawing functions; nothing is simulated here.

Figures written into the demo folder:
  fig_clamp_check.png        Brian2 against scipy at -60 mV, the calibration
                             targets, and the scipy reference for G0 = 3, 78,
                             585 uM
  fig_anatomy_<regime>.png   one astrocyte and its partner neurons: the map,
                             the astrocyte's C and G_A, the partners' I_sic and
                             r_SIC, and their V with the SIC on and off
  fig_network_<regime>.png   rasters with the SIC off and on, astrocyte
                             events, the population rate of both twins
  fig_summary.png            all regimes: per-neuron rate on against off,
                             rate change against the number of partner
                             astrocytes, firing around the releases of the
                             partner astrocytes (on and off), and the size of
                             the SIC events (clamp-equivalent at -60 mV)

Run:  python3 sic_demo_plots.py /path/to/sic_demo_out
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- style: the validated reference palette (categorical slots 1-2, ordinal blues)
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
C_ON, C_OFF = '#eb6834', '#2a78d6'           # SIC on: orange; SIC off: blue
ORD = ('#86b6ef', '#2a78d6', '#104281')
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 9.5, 'legend.frameon': False, 'lines.linewidth': 1.2})

B60 = 1.0 / (1.0 + np.exp(0.062 * 60.0) / 3.57)     # Mg block at -60 mV
LABEL = {'nominal': 'registry nominal (O_N 0.3)',
         'oscillating': 'oscillating astrocytes (O_N 1e-4), G0 78 uM',
         'quiet_strong': 'quiet network, G0 585 uM (G_T 1000, U_A 0.9)',
         'repeated': 'repeated releases (O_N 1e-4, C_Theta 0.3 uM), G0 78 uM'}


# =============================================================================
# loading
# =============================================================================
def load(folder):
    meta = json.load(open(os.path.join(folder, 'demo_meta.json')))
    topo = dict(np.load(os.path.join(folder, 'topology.npz')))
    runs = {}
    for reg in meta['regimes']:
        pair = {}
        for tag in ('on', 'off'):
            f = os.path.join(folder, 'run_%s_%s.npz' % (reg, tag))
            if os.path.exists(f):
                pair[tag] = dict(np.load(f))
        if len(pair) == 2:
            runs[reg] = pair
    clamp = None
    f = os.path.join(folder, 'clamp.npz')
    if os.path.exists(f):
        clamp = dict(np.load(f))
    return meta, topo, runs, clamp


def param(run, name):
    names = [str(x) for x in run['param_names']]
    return float(run['params'][names.index(name)])


# =============================================================================
# analysis
# =============================================================================
def partners(topo):
    """astrocyte -> its partner neurons; neuron -> number of partners."""
    a, i = topo['SIC_i'], topo['SIC_j']
    Na, Nn = len(topo['A_pos']), len(topo['N_pos'])
    of_astro = [i[a == k] for k in range(Na)]
    n_per_neuron = np.bincount(i, minlength=Nn)
    return of_astro, n_per_neuron


def rates(run, Nn, t0=0.0):
    T = float(run['simtime'])
    m = run['spk_N_t'] >= t0
    return np.bincount(run['spk_N_i'][m], minlength=Nn) / (T - t0)


def pop_rate(run, Nn, bin_s=0.05):
    T = float(run['simtime'])
    edges = np.arange(0.0, T + bin_s, bin_s)
    h, _ = np.histogram(run['spk_N_t'], bins=edges)
    return edges[:-1] + bin_s / 2, h / (bin_s * Nn)


def releases(run, min_jump_uM=0.01):
    """Gliotransmitter releases (Gliot_release crossing C > C_Theta), found as
    the upward jumps of G_A in the 5-ms record: between releases G_A only
    decays. Returns (times in s, astrocyte index); a release is dated at the
    end of the 5-ms interval that contains it. (The astrocyte 'events' spk_A
    are C > C_osc crossings, 0.2 uM, not releases.)"""
    G, t = run['rec_G_A'].astype(np.float64), run['rec_t_astro']
    a, k = np.nonzero(np.diff(G, axis=1) > min_jump_uM)
    order = np.argsort(t[k + 1], kind='stable')
    return t[k + 1][order], a[order]


def peri_release(run, of_astro, Nn, pre=1.0, post=4.0, bin_s=0.05, partner=True,
                 t_min=0.0):
    """Mean firing rate (Hz) of the partner (or non-partner) neurons of each
    release at t >= t_min, aligned to the release; one sample per (release,
    neuron)."""
    edges = np.arange(-pre, post + bin_s / 2, bin_s)
    T = float(run['simtime'])
    st, si = run['spk_N_t'], run['spk_N_i']
    by_n = [st[si == n] for n in range(Nn)]
    acc, n_samples = np.zeros(len(edges) - 1), 0
    for te, a in zip(*releases(run)):
        if te < t_min or te - pre < 0 or te + post > T:
            continue
        p = of_astro[int(a)]
        group = p if partner else np.setdiff1d(np.arange(Nn), p)
        for n in group:
            h, _ = np.histogram(by_n[int(n)] - te, bins=edges)
            acc += h
            n_samples += 1
    centres = edges[:-1] + bin_s / 2
    return centres, (acc / (max(n_samples, 1) * bin_s)), n_samples


def sic_event_sizes(run, g_sic_nS, min_pA=2.0):
    """Peaks of each neuron's r_SIC (all-neuron record, 5 ms) as the current
    they would give at -60 mV: g_sic r B(-60) 60 mV, in pA."""
    from scipy.signal import find_peaks
    if 'rec_r_sic_all' not in run:
        return np.array([])
    out = []
    for r in run['rec_r_sic_all']:
        I_eq = g_sic_nS * r.astype(np.float64) * B60 * 60.0
        k, _ = find_peaks(I_eq, height=min_pA, prominence=min_pA)
        out.append(I_eq[k])
    return np.concatenate(out) if out else np.array([])


def decay_fit(t_ms, mag):
    """Single exponential a exp(-(t - t_pk)/tau) fitted from the peak of a
    positive trace down to 10 % of the peak (the calibration's definition)."""
    from scipy.optimize import curve_fit
    k = int(np.argmax(mag))
    tail = np.nonzero(mag[k:] < 0.1 * mag[k])[0]
    end = k + (tail[0] if len(tail) else len(mag) - k)
    (a, tau), _ = curve_fit(lambda x, a, b: a * np.exp(-x / b), t_ms[k:end] - t_ms[k],
                            mag[k:end], p0=(mag[k], 700.0))
    return t_ms[k], a, tau


def release_times(run, a):
    t, i = releases(run)
    return np.sort(t[i == a])


# =============================================================================
# drawing
# =============================================================================
def fig_clamp(clamp, path):
    t = clamp['t_ms']
    g, V = float(clamp['g_sic_nS']), float(clamp['V_mV'])
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ref78 = clamp['r_ref_G78']
    ax[0].plot(t / 1e3, ref78, color=INK2, lw=2.4, alpha=0.45, label='scipy (rtol 1e-10)')
    ax[0].plot(t / 1e3, clamp['r_brian'], color=C_ON, lw=1.1, label='Brian2, dt 0.05 ms')
    ax[0].set_title('(a) open fraction r(t), one release, G0 = 78 uM')
    ax[0].set_ylabel('r (open fraction)')
    ax[0].legend(loc='upper right')
    I = clamp['I_pA']
    k = int(np.argmax(-I))
    ax[1].plot(t / 1e3, I, color=C_ON)
    ax[1].axhline(-43.3, color=INK2, ls=':', lw=0.8)
    ax[1].axvline(0.2201, color=INK2, ls=':', lw=0.8)
    t_pk, a_fit, tau_fit = decay_fit(t, -I)
    tt = np.linspace(t_pk, 6000.0, 200)
    ax[1].plot(tt / 1e3, -a_fit * np.exp(-(tt - t_pk) / tau_fit), color=INK, ls='--', lw=0.9,
               label='single-exp. fit, peak to 10 %%: tau %.0f ms (target 706)' % tau_fit)
    ax[1].annotate('peak %.1f pA at %.0f ms\n(target 43.3 pA, 220 ms)' % (I[k], t[k]),
                   (t[k] / 1e3, I[k]), xytext=(1.6, -36), fontsize=8,
                   arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    ax[1].set_title('(b) I_SIC at V = %.0f mV (Kovacs & Pal 2017 targets)' % V)
    ax[1].set_ylabel('I_SIC (pA, inward < 0)')
    ax[1].legend(loc='lower right', fontsize=8)
    for key, c, lab in (('r_ref_G3', ORD[0], 'G0 3 uM'), ('r_ref_G78', ORD[1], 'G0 78 uM'),
                        ('r_ref_G585', ORD[2], 'G0 585 uM')):
        r = clamp[key]
        Imag = g * r * B60 * 60.0
        kk = int(np.argmax(r))
        ax[2].plot(t / 1e3, Imag, color=c, label='%s: peak %.3g pA at %.0f ms' % (lab, Imag[kk], t[kk]))
    ax[2].set_yscale('log')
    ax[2].set_ylim(1e-3, 500)
    ax[2].set_title('(c) the release size changes size and rise (scipy)')
    ax[2].set_ylabel('|I_SIC| at -60 mV (pA, log)')
    ax[2].legend(loc='lower right', fontsize=7.5)
    for a in ax:
        a.set_xlabel('time after the release (s)')
        a.set_xlim(0, 6)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _zoom_window(run, a, T):
    """A 6-s window around a release of astrocyte a: the second release if any
    (the first is the onset transient), else the first."""
    rt = release_times(run, a)
    if len(rt) == 0:
        return 0.0, min(6.0, T)
    te = rt[1] if len(rt) > 1 else rt[0]
    lo = max(0.0, te - 1.5)
    return lo, min(T, lo + 6.0)


def fig_anatomy(reg, pair, topo, meta, path):
    on, off = pair['on'], pair['off']
    of_astro, n_per = partners(topo)
    focus = int(meta['focus_astrocytes'][0])
    part = [int(x) for x in of_astro[focus]]
    T = float(on['simtime'])
    lo, hi = _zoom_window(on, focus, T)
    C_Theta = param(on, 'C_Theta')
    fig = plt.figure(figsize=(12.5, 9.2))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.2, 1, 1, 1.5], hspace=0.55, wspace=0.3)
    # (a) map
    ax = fig.add_subplot(gs[0, 0])
    Np, Ap = topo['N_pos'], topo['A_pos']
    for a_, i_ in zip(topo['SIC_i'], topo['SIC_j']):
        ax.plot([Ap[a_, 0], Np[i_, 0]], [Ap[a_, 1], Np[i_, 1]], color=GRID, lw=0.6, zorder=1)
    for i_ in part:
        ax.plot([Ap[focus, 0], Np[i_, 0]], [Ap[focus, 1], Np[i_, 1]], color=C_ON, lw=1.6, zorder=2)
    ax.scatter(Np[:, 0], Np[:, 1], s=9, color=C_OFF, zorder=3, label='neurons')
    ax.scatter(Ap[:, 0], Ap[:, 1], s=14, marker='^', color=INK2, zorder=3, label='astrocytes')
    ax.scatter(Ap[focus, 0], Ap[focus, 1], s=70, marker='^', color=C_ON, edgecolor=INK, zorder=4,
               label='astrocyte %d' % focus)
    ax.set_aspect('equal')
    ax.set_title('(a) SIC pairs (grey; Voronoi neighbours). Dots: neurons;\n'
                 'triangles: astrocytes; orange: astrocyte %d, %d partners' % (focus, len(part)),
                 fontsize=8)
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    # (b) C and G_A of the focus astrocyte, whole run
    ax = fig.add_subplot(gs[0, 1:])
    ta = on['rec_t_astro']
    ax.plot(ta, on['rec_C'][focus], color=INK, lw=1.0, label='C (uM)')
    ax.axhline(C_Theta, color=INK2, ls=':', lw=0.8)
    ax.text(0.2, C_Theta * 1.04, 'C_Theta %.2f uM: release' % C_Theta, fontsize=7.5, color=INK2)
    ax.set_ylabel('C (uM)')
    ax2 = ax.twinx()
    ax2.plot(ta, on['rec_G_A'][focus], color=C_ON, lw=1.0, label='G_A (uM)')
    ax2.set_ylabel('G_A (uM)', color=C_ON)
    ax2.grid(False)
    ax2.spines['right'].set_visible(True)
    ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0)
    ax.set_title('(b) astrocyte %d: calcium and gliotransmitter, SIC-on run (shaded: window of c-e)' % focus)
    ax.set_xlabel('time (s)')
    ax.set_xlim(0, T)
    # (c) I_sic of the partners, zoom
    idx = [int(x) for x in on['rec_idx_fine']]
    tf = on['rec_t_fine']
    w = (tf >= lo) & (tf <= hi)
    ax = fig.add_subplot(gs[1, :])
    for k, n in enumerate(part):
        if n in idx and 'rec_I_sic_fine' in on:
            ax.plot(tf[w], on['rec_I_sic_fine'][idx.index(n)][w], lw=0.9,
                    color=ORD[k % 3], label='neuron %d (%d partners)' % (n, n_per[n]))
    for te in release_times(on, focus):
        if lo <= te <= hi:
            ax.axvline(te, color=C_ON, ls='--', lw=0.8)
    ax.set_ylabel('I_SIC (pA)')
    ax.set_title('(c) slow inward current of the partner neurons (SIC on); dashed: releases of astrocyte %d' % focus)
    ax.legend(loc='lower right', fontsize=7, ncol=3)
    ax.set_xlim(lo, hi)
    # (d) r_SIC of the partners
    ax = fig.add_subplot(gs[2, :])
    for k, n in enumerate(part):
        if n in idx and 'rec_r_sic_fine' in on:
            ax.plot(tf[w], on['rec_r_sic_fine'][idx.index(n)][w], lw=0.9, color=ORD[k % 3])
    ax.set_ylabel('r_SIC (sum of open fractions)')
    ax.set_title('(d) open fraction summed over each neuron\'s partner astrocytes')
    ax.set_xlim(lo, hi)
    # (e) V on vs off for up to 3 partners and one control
    ax = fig.add_subplot(gs[3, :])
    show = [n for n in part if n in idx][:3]
    ctrl = [int(c) for c in meta['control_neurons'] if int(c) in idx][:1]
    off_idx = [int(x) for x in off['rec_idx_fine']]
    for k, n in enumerate(show + ctrl):
        off_v = off['rec_V'][off_idx.index(n)][w] - 45.0 * k
        on_v = on['rec_V'][idx.index(n)][w] - 45.0 * k
        ax.plot(tf[w], off_v, color=C_OFF, lw=0.7, label='SIC off' if k == 0 else None)
        ax.plot(tf[w], on_v, color=C_ON, lw=0.7, label='SIC on' if k == 0 else None)
        tag = 'control, %d partners' % n_per[n] if n in ctrl else '%d partners' % n_per[n]
        ax.text(lo + 0.02 * (hi - lo), -50.0 - 45.0 * k, 'neuron %d (%s)' % (n, tag), fontsize=7.5,
                color=INK2, va='bottom')
    for te in release_times(on, focus):
        if lo <= te <= hi:
            ax.axvline(te, color=C_ON, ls='--', lw=0.8)
    ax.set_yticks([])
    ax.set_ylabel('V (mV; traces offset by 45 mV)')
    ax.set_xlabel('time (s)')
    ax.set_title('(e) membrane potential, same topology, seeds and noise: SIC on (orange) vs off (blue)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(lo, hi)
    fig.suptitle('SIC anatomy -- %s' % LABEL.get(reg, reg), fontsize=11, y=0.995)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def fig_network(reg, pair, topo, path):
    Nn, Na = len(topo['N_pos']), len(topo['A_pos'])
    T = float(pair['on']['simtime'])
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 2, 1, 1.4]))
    for a, tag, c in ((ax[0], 'off', C_OFF), (ax[1], 'on', C_ON)):
        r = pair[tag]
        a.scatter(r['spk_N_t'], r['spk_N_i'], s=0.4, color=c, lw=0)
        a.set_ylabel('neuron')
        a.set_ylim(-1, Nn)
        a.set_title('(%s) neurons, SIC %s: %d spikes (%.2f Hz per neuron)'
                    % ('a' if tag == 'off' else 'b', tag, r['spk_N_t'].size, r['spk_N_t'].size / (T * Nn)))
    on = pair['on']
    rt, ra = releases(on)
    ax[2].scatter(on['spk_A_t'], on['spk_A_i'], s=3, color=GRID, lw=0, label='C > C_osc events')
    ax[2].scatter(rt, ra, s=6, color=C_ON, lw=0, label='releases (C > C_Theta)')
    ax[2].set_ylabel('astrocyte')
    ax[2].set_ylim(-1, Na)
    ax[2].legend(loc='upper right', fontsize=7, markerscale=2)
    ax[2].set_title('(c) astrocytes, SIC-on run: %d gliotransmitter releases, %d C_osc events'
                    % (rt.size, on['spk_A_t'].size))
    for tag, c in (('off', C_OFF), ('on', C_ON)):
        t, pr = pop_rate(pair[tag], Nn)
        ax[3].plot(t, pr, color=c, lw=0.9, label='SIC %s' % tag)
    ax[3].set_ylabel('rate (Hz per neuron)\n50 ms bins')
    ax[3].set_xlabel('time (s)')
    ax[3].legend(loc='upper right')
    ax[3].set_title('(d) population rate')
    ax[3].set_xlim(0, T)
    fig.suptitle('Network activity -- %s' % LABEL.get(reg, reg), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_summary(runs, topo, path, t0=3.0):
    of_astro, n_per = partners(topo)
    Nn = len(topo['N_pos'])
    regs = list(runs)
    fig, ax = plt.subplots(len(regs), 4, figsize=(15, 3.4 * len(regs)), squeeze=False)
    for row, reg in enumerate(regs):
        on, off = runs[reg]['on'], runs[reg]['off']
        r_on, r_off = rates(on, Nn, t0), rates(off, Nn, t0)
        # (a) rate on vs off
        a = ax[row, 0]
        cols = np.array([ORD[min(int(k), 2)] if k > 0 else GRID for k in np.clip(n_per - 1, -1, 2)])
        pos = np.concatenate([r_on[r_on > 0], r_off[r_off > 0]])
        lo_ = max(0.01, pos.min() / 2.0) if pos.size else 0.01
        a.scatter(np.maximum(r_off, lo_), np.maximum(r_on, lo_), s=10, c=list(cols), edgecolor=INK2, lw=0.2)
        m = max(r_on.max(), r_off.max(), 1.0) * 1.5
        a.plot([lo_, m], [lo_, m], color=INK2, ls=':', lw=0.8)
        a.set_xlim(lo_, m)
        a.set_ylim(lo_, m)
        a.set_xscale('log')
        a.set_yscale('log')
        a.set_xlabel('rate, SIC off (Hz)')
        a.set_ylabel('rate, SIC on (Hz)')
        a.set_title('%s\n(a) per neuron, t >= %.0f s' % (LABEL.get(reg, reg), t0), fontsize=8.5)
        # (b) change vs partner count
        a = ax[row, 1]
        ks = np.unique(n_per)
        data = [r_on[n_per == k] - r_off[n_per == k] for k in ks]
        a.boxplot(data, positions=ks, widths=0.6, showfliers=False,
                  medianprops=dict(color=C_ON), boxprops=dict(color=INK2),
                  whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        for k, d in zip(ks, data):
            a.scatter(np.full(d.size, k) + np.linspace(-0.15, 0.15, d.size), d, s=5, color=C_ON, alpha=0.6, lw=0)
        a.axhline(0, color=INK2, lw=0.8)
        a.set_xlabel('partner astrocytes of the neuron')
        a.set_ylabel('rate on - off (Hz)')
        a.set_title('(b) change against the number of partners', fontsize=8.5)
        # (c) peri-release firing: the releases after 5 s when there are
        # enough of them (the onset releases are synchronous across the
        # astrocytes, so partners and non-partners cannot be told apart there)
        a = ax[row, 2]
        _, _, n_late = peri_release(on, of_astro, Nn, partner=True, t_min=5.0)
        t_min = 5.0 if n_late >= 10 else 0.0
        for run, c, tag in ((off, C_OFF, 'off'), (on, C_ON, 'on')):
            x, y, n = peri_release(run, of_astro, Nn, partner=True, t_min=t_min)
            a.plot(x, y, color=c, label='partners, SIC %s (%d)' % (tag, n))
            x, y, n = peri_release(run, of_astro, Nn, partner=False, t_min=t_min)
            a.plot(x, y, color=c, ls=':', lw=0.9, label='non-partners, SIC %s' % tag)
        a.axvline(0, color=INK2, lw=0.8)
        a.set_xlabel('time from a release of the astrocyte (s)')
        a.set_ylabel('rate (Hz)')
        a.set_title('(c) firing around each release %s' % ('after 5 s' if t_min > 0 else
                    '(onset releases only)'), fontsize=8.5)
        a.legend(fontsize=6.5, loc='upper right')
        # (d) SIC event sizes
        a = ax[row, 3]
        g = 40.6
        sizes = sic_event_sizes(on, g)
        if sizes.size:
            a.hist(sizes, bins=np.linspace(0, max(sizes.max(), 50) * 1.05, 40), color=C_ON, alpha=0.85)
            a.axvline(43.3, color=INK, ls='--', lw=0.9)
            a.text(45, a.get_ylim()[1] * 0.9, '43.3 pA\n(one release, G0 78 uM)', fontsize=7)
        a.set_xlabel('SIC event peak, clamp-equivalent at -60 mV (pA)')
        a.set_ylabel('events (neuron x peak)')
        a.set_title('(d) SIC events: %d peaks > 2 pA' % sizes.size, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def numbers(runs, topo, t0=3.0):
    """Printed summary, quoted in the report."""
    of_astro, n_per = partners(topo)
    Nn = len(topo['N_pos'])
    for reg, pair in runs.items():
        on, off = pair['on'], pair['off']
        r_on, r_off = rates(on, Nn, t0), rates(off, Nn, t0)
        r_on0, r_off0 = rates(on, Nn), rates(off, Nn)
        sizes = sic_event_sizes(on, 40.6)
        I_min = float(on['rec_I_sic_all'].min()) if 'rec_I_sic_all' in on else float('nan')
        _, _, n_late = peri_release(on, of_astro, Nn, t_min=5.0)
        t_min = 5.0 if n_late >= 10 else 0.0
        x, y_on, n_ev = peri_release(on, of_astro, Nn, t_min=t_min)
        _, y_off, _ = peri_release(off, of_astro, Nn, t_min=t_min)
        rel_on = releases(on)[0]
        base_on = y_on[x < 0].mean() if n_ev else float('nan')
        post = (x > 0) & (x < 2.0)
        print('%s: releases on %d / off %d; rate whole run on %.2f / off %.2f Hz; t >= %.0f s on '
              '%.2f / off %.2f Hz; most inward I_sic %.1f pA; SIC peaks > 2 pA: %d (median %.1f pA, '
              'max %.1f pA); releases after 5 s: %d; partners peri-release (t >= %.0f s) 0-2 s: on %.2f / '
              'off %.2f Hz (baseline on %.2f) over %d (release, neuron) samples'
              % (reg, releases(on)[0].size, releases(off)[0].size, r_on0.mean(), r_off0.mean(), t0,
                 r_on.mean(), r_off.mean(), I_min, sizes.size,
                 np.median(sizes) if sizes.size else float('nan'),
                 sizes.max() if sizes.size else float('nan'), int((rel_on >= 5.0).sum()), t_min,
                 y_on[post].mean() if n_ev else float('nan'), y_off[post].mean() if n_ev else float('nan'),
                 base_on, n_ev))


def main():
    folder = sys.argv[1]
    meta, topo, runs, clamp = load(folder)
    if clamp is not None:
        fig_clamp(clamp, os.path.join(folder, 'fig_clamp_check.png'))
    for reg, pair in runs.items():
        fig_anatomy(reg, pair, topo, meta, os.path.join(folder, 'fig_anatomy_%s.png' % reg))
        fig_network(reg, pair, topo, os.path.join(folder, 'fig_network_%s.png' % reg))
    if runs:
        fig_summary(runs, topo, os.path.join(folder, 'fig_summary.png'))
        numbers(runs, topo)
    print('figures written to', folder)


if __name__ == '__main__':
    main()
