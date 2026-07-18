#!/usr/bin/env python3
# mea_plots.py
# =============================================================================
# Diagnostic plots for the virtual-MEA pipeline.
#
# SEPARATION OF CONCERNS
#   Every function here takes PLAIN ARRAYS (and scalars) only. This module has
#   no knowledge of .npz files, of PipelineConfig, or of the campaign layout.
#   It never computes science: it does not detect, filter, or re-weight -- it
#   only draws what it is given. That keeps plotting swappable without
#   touching the pipeline, and keeps the pipeline runnable without matplotlib.
#
# HEADLESS / HPC SAFETY
#   matplotlib is imported LAZILY inside _plt() and forced to the 'Agg'
#   backend, so (a) compute nodes with no display still work, and (b) a node
#   without matplotlib installed can still run the numerical pipeline -- only
#   plotting is skipped (see HAVE_MPL).
#
# THE PLOTS
#   Per-topology (geometry; identical for every iter of that topology):
#     1. plot_probe_layout      - culture, neurons, electrodes to scale, r_max
#     2. plot_weight_decay      - W vs distance + the analytic scaling law
#     3. plot_template_library  - the EAP template bank
#   Per-iteration (signal):
#     4. plot_stacked_traces         - 9 channels stacked (raw and/or filtered)
#     5. plot_traces_with_detections - same + thresholds + detected troughs
#     6. plot_raster                 - detected raster (TP/FP) + true raster
#     7. plot_zoom                   - short window, one channel, waveform-level
#     8. plot_detected_waveforms     - snippet overlay per channel + mean
#     9. plot_amplitude_histogram    - |amplitude| vs threshold per channel
#    10. plot_detection_summary      - per-channel TP/FP counts + sigma
#
# UNITS: positions um, times s, amplitudes uV.
# =============================================================================

import os

import numpy as np

try:                       # probe availability WITHOUT importing pyplot yet
    import matplotlib      # noqa: F401
    HAVE_MPL = True
except Exception:          # noqa: BLE001
    HAVE_MPL = False


# -----------------------------------------------------------------------------
# Backend / style helpers
# -----------------------------------------------------------------------------

def _plt():
    """Lazily import pyplot with a headless backend. Raises if unavailable."""
    if not HAVE_MPL:
        raise RuntimeError('matplotlib is not installed; plotting unavailable')
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    return plt


def _save(fig, out_dir, name, dpi=140):
    """Save `fig` as <out_dir>/<name>.png and close it. Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + '.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    _plt().close(fig)
    return path


def _chan_label(e, n_side=3):
    """Human label for electrode e in a row-major n_side x n_side grid."""
    return 'E%d (r%d,c%d)' % (e, e // n_side, e % n_side)


def busiest_window(det_t, simtime_s, width_s=2.0):
    """
    Pick the [t0, t1] window of length `width_s` containing the most detected
    events. Plotting 180 s at 10 kHz is unreadable and slow, so every
    time-domain plot defaults to the most informative slice instead of t=0.

    Returns (t0, t1). Falls back to the start of the recording if no events.
    """
    width_s = float(min(width_s, simtime_s))
    if det_t is None or len(det_t) == 0 or simtime_s <= width_s:
        return (0.0, width_s)
    edges = np.arange(0.0, simtime_s + width_s, width_s)
    counts, _ = np.histogram(det_t, bins=edges)
    j = int(np.argmax(counts))
    t0 = float(edges[j])
    return (t0, min(t0 + width_s, simtime_s))


def _slice_idx(fs_hz, t0, t1, n_samples):
    """Sample index range [i0, i1) for the time window [t0, t1)."""
    i0 = max(int(np.floor(t0 * fs_hz)), 0)
    i1 = min(int(np.ceil(t1 * fs_hz)), n_samples)
    return i0, max(i1, i0 + 1)


# =============================================================================
# 1-3. Per-topology geometry / model plots
# =============================================================================

def plot_probe_layout(neuron_xy, probe, W=None, r_max=None, c_max=None,
                      out_dir='.', name='01_probe_layout'):
    """
    The culture arena with the probe drawn TO SCALE.

    What it lets you verify at a glance:
      - the 3x3 probe really is centred in the arena,
      - electrode faces are 25 um squares at 60 um pitch,
      - which neurons are close enough to matter (colour = coupling weight),
      - the r_max reach cutoff (dashed circle) sits where the theory says.

    neuron_xy : (Nn, 2) positions [um]
    probe     : dict from mea_probe.make_probe
    W         : (Nn, E) weights [uV] or None (colours the neurons)
    r_max     : reach cutoff [um] or None
    c_max     : arena side [um] or None (inferred from positions if None)
    """
    plt = _plt()
    from matplotlib.patches import Rectangle, Circle

    xy = np.asarray(neuron_xy, dtype=float)
    centers = probe['centers']
    cfg = probe['cfg']
    ctr = probe['center_xy']
    if c_max is None:
        c_max = float(np.max(xy)) if xy.size else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, zoom in zip(axes, [False, True]):
        # --- neurons -------------------------------------------------------
        if W is not None and W.size:
            wmax = W.max(axis=1)                       # best coupling per neuron
            vis = wmax > 0
            # log colour scale: weights span orders of magnitude under 1/r^2
            c = np.log10(np.maximum(wmax, 1e-3))
            sc = ax.scatter(xy[vis, 0], xy[vis, 1], c=c[vis], s=6,
                            cmap='viridis', alpha=0.75, linewidths=0)
            if zoom:
                cb = fig.colorbar(sc, ax=ax, shrink=0.85)
                cb.set_label(r'log$_{10}$ max$_e$ W(n,e)  [$\mu$V]')
        else:
            ax.scatter(xy[:, 0], xy[:, 1], s=5, color='0.7', linewidths=0)

        # --- electrode faces, to scale -------------------------------------
        half = cfg.edge / 2.0
        for e, (ex, ey) in enumerate(centers):
            ax.add_patch(Rectangle((ex - half, ey - half), cfg.edge, cfg.edge,
                                   facecolor='crimson', edgecolor='k',
                                   alpha=0.85, lw=0.8, zorder=5))
            if zoom:
                ax.text(ex, ey - half - 6, str(e), ha='center', va='top',
                        fontsize=8, color='crimson', zorder=6)

        # --- reach cutoff ---------------------------------------------------
        if r_max is not None and np.isfinite(r_max):
            ax.add_patch(Circle((ctr[0], ctr[1]), r_max, fill=False,
                                ls='--', lw=1.2, edgecolor='navy', zorder=4))

        ax.set_aspect('equal')
        ax.set_xlabel(r'x [$\mu$m]')
        ax.set_ylabel(r'y [$\mu$m]')
        if zoom:
            pad = (r_max * 1.25) if (r_max and np.isfinite(r_max)) else 200.0
            ax.set_xlim(ctr[0] - pad, ctr[0] + pad)
            ax.set_ylim(ctr[1] - pad, ctr[1] + pad)
            ax.set_title('zoom on probe (dashed = $r_{max}$ reach)')
        else:
            ax.set_xlim(0, c_max)
            ax.set_ylim(0, c_max)
            ax.set_title('culture %.0f x %.0f $\\mu$m, N = %d neurons'
                         % (c_max, c_max, xy.shape[0]))
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_weight_decay(neuron_xy, probe, W, scaling, r_max=None,
                      noise_floor=None, out_dir='.', name='02_weight_decay'):
    """
    Empirical coupling weight vs distance, against the ANALYTIC scaling law.

    This is the single most useful correctness check in the pipeline: the
    scatter (what compute_weights actually produced, including sub-site
    averaging) must lie on the analytic curve
        A(d) = A_ref * (r_ref / max(d, r_min)) ** n_dec
    apart from a small sub-site-averaging lift at short distance. A wrong
    exponent, a wrong floor, or a unit error shows up here immediately.

    scaling : object with attributes A_ref, r_ref, r_min, n_dec
              (mea_probe.ScalingConfig)
    """
    plt = _plt()
    xy = np.asarray(neuron_xy, dtype=float)
    centers = probe['centers']

    # distance from each neuron to the NEAREST electrode centre
    d = np.sqrt(((xy[:, None, :] - centers[None, :, :]) ** 2).sum(-1)).min(1)
    w = W.max(axis=1)

    keep = w > 0
    d, w = d[keep], w[keep]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(d, w, s=7, alpha=0.45, color='steelblue', linewidths=0,
               label='computed W(n,e), best electrode')

    if d.size:
        dd = np.logspace(np.log10(max(d.min(), 1.0)),
                         np.log10(max(d.max(), 10.0)), 200)
        aa = scaling.A_ref * (scaling.r_ref /
                              np.maximum(dd, scaling.r_min)) ** scaling.n_dec
        ax.plot(dd, aa, 'k-', lw=2,
                label=r'analytic $A_{ref}(r_{ref}/\tilde r)^{n_{dec}}$, '
                      '$n_{dec}$=%g' % scaling.n_dec)

    ax.axvline(scaling.r_ref, color='green', ls=':', lw=1.2,
               label=r'$r_{ref}$ = %g $\mu$m' % scaling.r_ref)
    ax.axvline(scaling.r_min, color='orange', ls=':', lw=1.2,
               label=r'$r_{min}$ = %g $\mu$m' % scaling.r_min)
    if r_max is not None and np.isfinite(r_max):
        ax.axvline(r_max, color='navy', ls='--', lw=1.2,
                   label=r'$r_{max}$ = %.0f $\mu$m' % r_max)
    if noise_floor is not None:
        ax.axhline(noise_floor, color='red', ls='--', lw=1.2,
                   label=r'reach floor $\gamma\sigma$ = %.2f $\mu$V' % noise_floor)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'distance to nearest electrode centre [$\mu$m]')
    ax.set_ylabel(r'coupling weight W [$\mu$V]')
    ax.set_title('scaling law: computed weights vs analytic curve')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_template_library(templates, dt_hr_ms, out_dir='.',
                          name='03_template_library'):
    """
    The EAP template bank: every normalised template (trough = -1) plus mean.
    Confirms the HH model produced biphasic waveforms of sane duration, and
    shows how much shape heterogeneity the library actually carries.
    """
    plt = _plt()
    T = np.atleast_2d(np.asarray(templates, dtype=float))
    t = np.arange(T.shape[1]) * float(dt_hr_ms)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for w in T:
        ax.plot(t, w, color='0.55', alpha=0.35, lw=0.8)
    ax.plot(t, T.mean(axis=0), color='crimson', lw=2.2, label='mean template')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('normalised EAP (trough = -1)')
    ax.set_title('EAP template library (n = %d)' % T.shape[0])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_dir, name)


# =============================================================================
# 4-5. Stacked traces
# =============================================================================

def _stack_offsets(traces, sigma=None, gap_sigmas=10.0):
    """
    Vertical offset per channel for a stacked plot. Offsets are a multiple of
    the noise level so channels are comparably spaced regardless of amplitude.
    """
    E = traces.shape[0]
    if sigma is not None and np.all(np.isfinite(sigma)) and np.max(sigma) > 0:
        step = gap_sigmas * float(np.median(sigma))
    else:
        step = gap_sigmas * float(np.std(traces)) if traces.size else 1.0
    if step <= 0:
        step = 1.0
    return np.arange(E) * step, step


def plot_stacked_traces(traces, fs_hz, t_window=None, sigma=None,
                        title='synthetic MEA traces', out_dir='.',
                        name='04_stacked_traces', color='k'):
    """
    All E channels stacked with a constant vertical offset.

    traces   : (E, T) array [uV]  (raw or filtered -- caller decides)
    t_window : (t0, t1) seconds, or None for the whole record
    sigma    : (E,) noise levels, used only to scale the vertical spacing
    """
    plt = _plt()
    traces = np.atleast_2d(np.asarray(traces, dtype=float))
    E, T = traces.shape
    simtime = T / float(fs_hz)
    if t_window is None:
        t_window = (0.0, simtime)
    i0, i1 = _slice_idx(fs_hz, t_window[0], t_window[1], T)
    tt = np.arange(i0, i1) / float(fs_hz)

    offs, step = _stack_offsets(traces[:, i0:i1], sigma)

    fig, ax = plt.subplots(figsize=(12, 7))
    for e in range(E):
        ax.plot(tt, traces[e, i0:i1] + offs[e], lw=0.6, color=color)
    ax.set_yticks(offs)
    ax.set_yticklabels([_chan_label(e) for e in range(E)], fontsize=8)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('channel (offset %.1f $\\mu$V)' % step)
    ax.set_title('%s  |  window %.3f-%.3f s' % (title, tt[0], tt[-1]))
    ax.grid(alpha=0.2, axis='x')
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_traces_with_detections(filtered, fs_hz, det_t, det_ch, det_amp,
                                sigma, k, t_window=None, src_neuron=None,
                                out_dir='.', name='05_traces_detections'):
    """
    Stacked BAND-PASSED traces with the detection threshold drawn per channel
    and every detected event marked at its trough.

    If `src_neuron` is given, events matched to a true spike are drawn in green
    and unmatched events (false positives) in red -- so detector performance is
    visible directly on the trace.

    filtered : (E, T) band-passed traces [uV]
    det_t/ch/amp : detected event time [s] / channel / amplitude [uV]
    sigma    : (E,) per-channel robust noise [uV]
    k        : threshold multiplier (threshold = -k * sigma[e])
    """
    plt = _plt()
    X = np.atleast_2d(np.asarray(filtered, dtype=float))
    E, T = X.shape
    simtime = T / float(fs_hz)
    if t_window is None:
        t_window = busiest_window(det_t, simtime, 2.0)
    i0, i1 = _slice_idx(fs_hz, t_window[0], t_window[1], T)
    tt = np.arange(i0, i1) / float(fs_hz)

    offs, step = _stack_offsets(X[:, i0:i1], sigma)
    det_t = np.asarray(det_t, dtype=float)
    det_ch = np.asarray(det_ch, dtype=int)
    det_amp = np.asarray(det_amp, dtype=float)
    in_win = (det_t >= tt[0]) & (det_t <= tt[-1])

    fig, ax = plt.subplots(figsize=(12, 7.5))
    for e in range(E):
        ax.plot(tt, X[e, i0:i1] + offs[e], lw=0.6, color='0.25')
        thr = -k * float(sigma[e])
        ax.hlines(thr + offs[e], tt[0], tt[-1], color='steelblue',
                  lw=0.9, ls='--', alpha=0.85)

        m = in_win & (det_ch == e)
        if not np.any(m):
            continue
        if src_neuron is None:
            ax.plot(det_t[m], det_amp[m] + offs[e], 'v', ms=5,
                    color='crimson', mec='k', mew=0.4, ls='none')
        else:
            src = np.asarray(src_neuron, dtype=int)
            tp = m & (src >= 0)
            fp = m & (src < 0)
            ax.plot(det_t[tp], det_amp[tp] + offs[e], 'v', ms=5,
                    color='forestgreen', mec='k', mew=0.4, ls='none')
            ax.plot(det_t[fp], det_amp[fp] + offs[e], 'v', ms=5,
                    color='red', mec='k', mew=0.4, ls='none')

    ax.set_yticks(offs)
    ax.set_yticklabels([_chan_label(e) for e in range(E)], fontsize=8)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('channel (offset %.1f $\\mu$V)' % step)
    lbl = ('green = matched to a true spike, red = unmatched (false positive)'
           if src_neuron is not None else 'detected troughs')
    ax.set_title('band-passed traces + detections  |  dashed = $-k\\sigma$ '
                 '(k = %g)\n%s' % (k, lbl))
    ax.grid(alpha=0.2, axis='x')
    fig.tight_layout()
    return _save(fig, out_dir, name)


# =============================================================================
# 6. Raster
# =============================================================================

def plot_raster(det_t, det_ch, n_el, simtime_s, src_neuron=None,
                spk_t=None, spk_i=None, visible_neurons=None,
                t_window=None, out_dir='.', name='06_raster'):
    """
    Top   : detected-event raster, one row per electrode (TP green / FP red).
    Bottom: ground-truth spike raster of the neurons the probe can actually
            see (`visible_neurons`), so that detected bursts can be compared
            against the network activity that produced them.

    visible_neurons : bool mask or index array of neurons within reach. Passing
                      all Nn neurons of a large culture makes the panel
                      unreadable, so the caller should pass the reach mask.
    """
    plt = _plt()
    det_t = np.asarray(det_t, dtype=float)
    det_ch = np.asarray(det_ch, dtype=int)
    if t_window is None:
        t_window = (0.0, float(simtime_s))
    t0, t1 = t_window

    have_truth = (spk_t is not None) and (spk_i is not None)
    if have_truth:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.0, 1.3]))
        ax_d, ax_t = axes
    else:
        fig, ax_d = plt.subplots(figsize=(12, 4.2))
        ax_t = None

    m = (det_t >= t0) & (det_t <= t1)
    if src_neuron is None:
        ax_d.plot(det_t[m], det_ch[m], '|', ms=7, color='crimson', mew=1.1)
    else:
        src = np.asarray(src_neuron, dtype=int)
        tp, fp = m & (src >= 0), m & (src < 0)
        ax_d.plot(det_t[tp], det_ch[tp], '|', ms=7, color='forestgreen',
                  mew=1.1, label='matched (TP)')
        ax_d.plot(det_t[fp], det_ch[fp], '|', ms=7, color='red',
                  mew=1.1, label='unmatched (FP)')
        if np.any(tp) or np.any(fp):
            ax_d.legend(fontsize=8, loc='upper right', ncol=2)
    ax_d.set_yticks(np.arange(n_el))
    ax_d.set_yticklabels([_chan_label(e) for e in range(n_el)], fontsize=8)
    ax_d.set_ylim(-0.6, n_el - 0.4)
    ax_d.set_ylabel('electrode')
    ax_d.set_title('detected events (%d in window)' % int(m.sum()))
    ax_d.grid(alpha=0.2, axis='x')

    if have_truth:
        st = np.asarray(spk_t, dtype=float)
        si = np.asarray(spk_i, dtype=int)
        if visible_neurons is not None:
            vis = np.asarray(visible_neurons)
            if vis.dtype == bool:
                vis_ids = np.where(vis)[0]
            else:
                vis_ids = vis
            sel = np.isin(si, vis_ids)
            st, si = st[sel], si[sel]
            # compact the neuron ids to consecutive rows for a dense plot
            remap = {int(n): j for j, n in enumerate(np.sort(vis_ids))}
            rows = np.array([remap[int(n)] for n in si]) if si.size else si
            n_rows = len(vis_ids)
        else:
            rows = si
            n_rows = int(si.max()) + 1 if si.size else 1
        mm = (st >= t0) & (st <= t1)
        ax_t.plot(st[mm], rows[mm], '|', ms=4, color='0.25', mew=0.7)
        ax_t.set_ylabel('neuron (within reach)')
        ax_t.set_ylim(-0.6, max(n_rows - 0.4, 0.6))
        ax_t.set_xlabel('time [s]')
        ax_t.set_title('ground-truth spikes of the %d neurons within reach'
                       % n_rows)
        ax_t.grid(alpha=0.2, axis='x')
    else:
        ax_d.set_xlabel('time [s]')

    ax_d.set_xlim(t0, t1)
    fig.tight_layout()
    return _save(fig, out_dir, name)


# =============================================================================
# 7-9. Waveform-level detail
# =============================================================================

def plot_zoom(filtered, fs_hz, det_t, det_ch, det_amp, sigma, k,
              channel=None, width_ms=60.0, out_dir='.', name='07_zoom'):
    """
    A very short window on ONE channel, at waveform resolution: this is where
    you can actually see the biphasic EAP shape, the threshold, and whether
    the lockout merged a ringing complex into a single detection.

    channel : electrode index, or None to auto-pick the busiest channel.
    """
    plt = _plt()
    X = np.atleast_2d(np.asarray(filtered, dtype=float))
    E, T = X.shape
    det_t = np.asarray(det_t, dtype=float)
    det_ch = np.asarray(det_ch, dtype=int)
    det_amp = np.asarray(det_amp, dtype=float)

    if channel is None:
        if det_ch.size:
            counts = np.bincount(det_ch, minlength=E)
            channel = int(np.argmax(counts))
        else:
            channel = 0

    on_ch = det_ch == channel
    width_s = width_ms * 1e-3
    if np.any(on_ch):
        # centre the window on this channel's densest activity
        t0, t1 = busiest_window(det_t[on_ch], T / float(fs_hz), width_s)
    else:
        t0, t1 = 0.0, width_s
    i0, i1 = _slice_idx(fs_hz, t0, t1, T)
    tt = np.arange(i0, i1) / float(fs_hz)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(tt, X[channel, i0:i1], lw=1.0, color='0.2', label='band-passed')
    thr = -k * float(sigma[channel])
    ax.axhline(thr, color='steelblue', ls='--', lw=1.2,
               label=r'threshold $-k\sigma$ = %.1f $\mu$V' % thr)
    ax.axhline(0, color='k', lw=0.5)
    m = on_ch & (det_t >= tt[0]) & (det_t <= tt[-1])
    ax.plot(det_t[m], det_amp[m], 'v', ms=8, color='crimson', mec='k',
            mew=0.5, ls='none', label='detected trough')
    ax.set_xlabel('time [s]')
    ax.set_ylabel(r'amplitude [$\mu$V]')
    ax.set_title('%s, zoom (%.0f ms)  |  %d events shown'
                 % (_chan_label(channel), (tt[-1] - tt[0]) * 1e3, int(m.sum())))
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_detected_waveforms(filtered, fs_hz, det_t, det_ch, n_el,
                            pre_ms=1.0, post_ms=2.0, max_per_ch=300,
                            out_dir='.', name='08_detected_waveforms'):
    """
    Snippet overlay: every detected event's waveform, aligned on its trough,
    one panel per electrode, with the mean in red.

    This is the standard spike-sorting sanity view. If the detector is firing
    on noise or on filter ringing, the overlay looks like a hairball with no
    consistent shape; if it is finding real EAPs, a clean biphasic mean stands
    out of the cloud.
    """
    plt = _plt()
    X = np.atleast_2d(np.asarray(filtered, dtype=float))
    T = X.shape[1]
    det_t = np.asarray(det_t, dtype=float)
    det_ch = np.asarray(det_ch, dtype=int)

    n_pre = int(round(pre_ms * 1e-3 * fs_hz))
    n_post = int(round(post_ms * 1e-3 * fs_hz))
    tt = (np.arange(-n_pre, n_post) / float(fs_hz)) * 1e3      # ms

    n_side = int(np.ceil(np.sqrt(n_el)))
    fig, axes = plt.subplots(n_side, n_side, figsize=(11, 9),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for e in range(n_el):
        ax = axes[e]
        idx = np.where(det_ch == e)[0]
        if idx.size > max_per_ch:                     # cap for legibility/speed
            idx = idx[np.linspace(0, idx.size - 1, max_per_ch).astype(int)]
        snips = []
        for j in idx:
            c = int(round(det_t[j] * fs_hz))
            a, b = c - n_pre, c + n_post
            if a < 0 or b > T:
                continue
            snips.append(X[e, a:b])
        if snips:
            S_ = np.vstack(snips)
            for s in S_:
                ax.plot(tt, s, color='0.6', alpha=0.25, lw=0.6)
            ax.plot(tt, S_.mean(axis=0), color='crimson', lw=1.8)
            ax.set_title('%s  n=%d' % (_chan_label(e), S_.shape[0]),
                         fontsize=8)
        else:
            ax.set_title('%s  n=0' % _chan_label(e), fontsize=8)
        ax.axvline(0, color='k', lw=0.5, ls=':')
        ax.grid(alpha=0.2)
    for e in range(n_el, len(axes)):
        axes[e].axis('off')

    fig.suptitle('detected-event waveforms, aligned on trough (mean in red)',
                 y=0.995)
    fig.supxlabel('time relative to trough [ms]')
    fig.supylabel(r'amplitude [$\mu$V]')
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_amplitude_histogram(det_amp, det_ch, sigma, k, n_el,
                             out_dir='.', name='09_amplitude_hist'):
    """
    Distribution of |detected amplitude| per channel against the threshold.

    Reading it: a healthy channel shows a population sitting clearly ABOVE the
    threshold line. A pile-up hugging the threshold means the detector is
    scraping the noise floor and k should probably be raised.
    """
    plt = _plt()
    det_amp = np.asarray(det_amp, dtype=float)
    det_ch = np.asarray(det_ch, dtype=int)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    data = []
    for e in range(n_el):
        a = np.abs(det_amp[det_ch == e])
        data.append(a if a.size else np.array([np.nan]))
    # NB: the tick labels are set via set_xticklabels rather than boxplot's
    # own keyword, because that keyword was renamed 'labels' -> 'tick_labels'
    # in matplotlib 3.9 and the old name is removed in 3.11. Setting the
    # ticks separately works identically on every version, which matters
    # when the cluster's matplotlib is not the one this was written against.
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(np.arange(1, n_el + 1))
    ax.set_xticklabels(['%d' % e for e in range(n_el)])
    thr = k * np.asarray(sigma, dtype=float)
    ax.plot(np.arange(1, n_el + 1), thr, 'r^--', ms=7,
            label=r'threshold $k\sigma_e$ (k = %g)' % k)
    ax.set_xlabel('electrode')
    ax.set_ylabel(r'|detected amplitude| [$\mu$V]')
    ax.set_title('detected amplitude distribution vs threshold, per channel')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    return _save(fig, out_dir, name)


def plot_detection_summary(det_ch, src_neuron, sigma, n_el, dt_to_truth=None,
                           out_dir='.', name='10_detection_summary'):
    """
    Per-channel bookkeeping: how many events, how many matched to a true spike
    (TP) vs unmatched (FP), the robust noise level of each channel, and the
    timing error distribution of matched events.
    """
    plt = _plt()
    det_ch = np.asarray(det_ch, dtype=int)
    src = np.asarray(src_neuron, dtype=int) if src_neuron is not None else None

    tp = np.zeros(n_el, dtype=int)
    fp = np.zeros(n_el, dtype=int)
    for e in range(n_el):
        m = det_ch == e
        if src is None:
            tp[e] = int(m.sum())
        else:
            tp[e] = int(np.sum(m & (src >= 0)))
            fp[e] = int(np.sum(m & (src < 0)))

    has_dt = dt_to_truth is not None and np.any(np.isfinite(dt_to_truth))
    ncol = 3 if has_dt else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.2))
    axes = np.atleast_1d(axes)

    x = np.arange(n_el)
    axes[0].bar(x, tp, color='forestgreen', label='matched (TP)')
    if src is not None:
        axes[0].bar(x, fp, bottom=tp, color='red', label='unmatched (FP)')
    axes[0].set_xticks(x)
    axes[0].set_xlabel('electrode')
    axes[0].set_ylabel('detected events')
    axes[0].set_title('events per channel (total %d)' % int(tp.sum() + fp.sum()))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25, axis='y')

    axes[1].bar(x, np.asarray(sigma, dtype=float), color='steelblue')
    axes[1].set_xticks(x)
    axes[1].set_xlabel('electrode')
    axes[1].set_ylabel(r'robust $\hat\sigma_e$ [$\mu$V]')
    axes[1].set_title('per-channel noise estimate')
    axes[1].grid(alpha=0.25, axis='y')

    if has_dt:
        dts = np.asarray(dt_to_truth, dtype=float)
        dts = dts[np.isfinite(dts)] * 1e3                     # ms
        axes[2].hist(dts, bins=30, color='0.4')
        axes[2].set_xlabel('|t_detected - t_true| [ms]')
        axes[2].set_ylabel('count')
        axes[2].set_title('timing error of matched events\n(median %.3f ms)'
                          % (np.median(dts) if dts.size else np.nan))
        axes[2].grid(alpha=0.25, axis='y')

    fig.tight_layout()
    return _save(fig, out_dir, name)


# =============================================================================
# Orchestrators
# =============================================================================

def save_topo_diagnostics(out_dir, neuron_xy, probe, W, scaling, r_max=None,
                          noise_floor=None, templates=None, dt_hr_ms=None):
    """
    Geometry/model plots that depend only on the TOPOLOGY (identical for every
    iter of that topology, so drawn once per topo_ directory).

    Returns the list of written paths.
    """
    paths = []
    c_max = None
    paths.append(plot_probe_layout(neuron_xy, probe, W=W, r_max=r_max,
                                   c_max=c_max, out_dir=out_dir))
    paths.append(plot_weight_decay(neuron_xy, probe, W, scaling, r_max=r_max,
                                   noise_floor=noise_floor, out_dir=out_dir))
    if templates is not None and dt_hr_ms is not None:
        paths.append(plot_template_library(templates, dt_hr_ms,
                                           out_dir=out_dir))
    return paths


def save_iter_diagnostics(out_dir, traces, filtered, fs_hz, det, src_neuron,
                          dt_to_truth, k, simtime_s, spk_t=None, spk_i=None,
                          visible_neurons=None, window_s=2.0):
    """
    Signal plots for ONE iteration.

    traces   : (E, T) raw synthetic traces [uV]
    filtered : (E, T) band-passed traces [uV] (from detect()['filtered'])
    det      : dict from mea_detection.detect  (keys ch, t, amp, sigma)
    Returns the list of written paths.
    """
    paths = []
    E = np.atleast_2d(traces).shape[0]
    win = busiest_window(det['t'], simtime_s, window_s)

    paths.append(plot_stacked_traces(traces, fs_hz, t_window=win,
                                     sigma=det['sigma'],
                                     title='raw synthetic traces',
                                     out_dir=out_dir,
                                     name='04_stacked_traces_raw'))
    paths.append(plot_traces_with_detections(
        filtered, fs_hz, det['t'], det['ch'], det['amp'], det['sigma'], k,
        t_window=win, src_neuron=src_neuron, out_dir=out_dir))
    paths.append(plot_raster(det['t'], det['ch'], E, simtime_s,
                             src_neuron=src_neuron, spk_t=spk_t, spk_i=spk_i,
                             visible_neurons=visible_neurons,
                             out_dir=out_dir))
    paths.append(plot_zoom(filtered, fs_hz, det['t'], det['ch'], det['amp'],
                           det['sigma'], k, out_dir=out_dir))
    paths.append(plot_detected_waveforms(filtered, fs_hz, det['t'], det['ch'],
                                         E, out_dir=out_dir))
    paths.append(plot_amplitude_histogram(det['amp'], det['ch'], det['sigma'],
                                          k, E, out_dir=out_dir))
    paths.append(plot_detection_summary(det['ch'], src_neuron, det['sigma'], E,
                                        dt_to_truth=dt_to_truth,
                                        out_dir=out_dir))
    return paths
