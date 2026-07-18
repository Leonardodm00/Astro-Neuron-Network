#!/usr/bin/env python3
# mea_detection.py
# =============================================================================
# Spike detection on virtual-MEA traces + optional ground-truth matching.
#
# SEPARATION OF CONCERNS
#   Consumes raw E-channel traces (from mea_synthesis) and produces detected
#   trough times / amplitudes per channel. Optionally links each detected
#   event to the nearest contributing true spike (validation / sorting label).
#
# METHOD (Quiroga et al. 2004, robust threshold)
#   1. band-pass [lo, hi] Hz (mea_synthesis.bandpass), zero-phase.
#   2. robust noise:  sigma_hat = median(|x|) / 0.6745   (per channel).
#   3. threshold:     Theta = -k * sigma_hat  (negative-going extracellular
#                     spikes); k tunable.
#   4. group threshold crossings into events separated by a refractory dead
#      time; align each event to its local trough (most-negative sample).
#   ALL parameters (band, k, refractory, polarity) are exposed.
# =============================================================================

from dataclasses import dataclass

import numpy as np

from mea_synthesis import bandpass


@dataclass
class DetectConfig:
    lo_hz: float = 300.0
    hi_hz: float = 3000.0
    order: int = 4
    k: float = 5.0             # threshold multiplier (tunable)
    refractory_ms: float = 2.0 # lockout/dead time; also merges band-pass ringing
    polarity: str = 'neg'      # 'neg' | 'pos' | 'both'


def robust_sigma(x):
    """Quiroga robust noise estimate: median(|x|)/0.6745."""
    return np.median(np.abs(x)) / 0.6745


def _detect_channel(x, fs_hz, cfg: DetectConfig):
    """
    Detect events on ONE already-band-passed channel x.
    Returns (trough_samples, trough_amps, sigma_hat).
    """
    sigma = robust_sigma(x)
    # Defensive guard: a silent / noise-free channel has sigma -> 0, which
    # would collapse the threshold and fire on numerical ringing. The MAD
    # threshold is only defined when there is noise, so return no events.
    if sigma <= 1e-12:
        return (np.empty(0, np.int64), np.empty(0, float), sigma)
    ref = int(round(cfg.refractory_ms * 1e-3 * fs_hz))
    ref = max(ref, 1)

    if cfg.polarity == 'neg':
        over = x < (-cfg.k * sigma)
        pick = lambda seg: int(np.argmin(seg))            # noqa: E731
    elif cfg.polarity == 'pos':
        over = x > (cfg.k * sigma)
        pick = lambda seg: int(np.argmax(seg))            # noqa: E731
    else:  # both: detect on |x|
        over = np.abs(x) > (cfg.k * sigma)
        pick = lambda seg: int(np.argmax(np.abs(seg)))    # noqa: E731

    idx = np.where(over)[0]
    if idx.size == 0:
        return (np.empty(0, np.int64), np.empty(0, float), sigma)

    # Lockout-merge: begin a new event only when the gap since the previous
    # supra-threshold sample exceeds the lockout `ref`. A single spike, after
    # zero-phase band-pass, produces a multi-lobe complex (filter ringing) whose
    # lobes sit within a few tenths of a ms of the trough; merging crossings
    # separated by < ref collapses that whole complex into ONE detection taken
    # at the true extremum over the merged span. `ref` therefore doubles as the
    # refractory dead time (no two detections closer than ref).
    breaks = np.where(np.diff(idx) > ref)[0] + 1
    groups = np.split(idx, breaks)

    troughs, amps = [], []
    for g in groups:
        lo, hi = int(g[0]), int(g[-1]) + 1
        loc = lo + pick(x[lo:hi])          # true extremum over the merged span
        troughs.append(loc)
        amps.append(float(x[loc]))

    return (np.asarray(troughs, np.int64), np.asarray(amps, float), sigma)


def detect(traces, fs_hz, cfg: DetectConfig = None, prefiltered=False):
    """
    Detect on all channels of `traces` (E, T).

    Returns dict:
      ch      : (M,) channel index per detected event
      t       : (M,) trough time [s]
      amp     : (M,) trough amplitude [uV] (signed)
      sigma   : (E,) per-channel robust sigma [uV]
      filtered: (E, T) band-passed traces (for inspection / matching)
    """
    if cfg is None:
        cfg = DetectConfig()
    traces = np.atleast_2d(traces)
    E = traces.shape[0]
    filt = traces if prefiltered else bandpass(
        traces, fs_hz, cfg.lo_hz, cfg.hi_hz, order=cfg.order)

    ch_all, t_all, amp_all, sig = [], [], [], np.empty(E)
    for e in range(E):
        tr, amp, s = _detect_channel(filt[e], fs_hz, cfg)
        sig[e] = s
        ch_all.append(np.full(len(tr), e, dtype=np.int64))
        t_all.append(tr / fs_hz)
        amp_all.append(amp)

    return dict(
        ch=np.concatenate(ch_all) if ch_all else np.empty(0, np.int64),
        t=np.concatenate(t_all) if t_all else np.empty(0, float),
        amp=np.concatenate(amp_all) if amp_all else np.empty(0, float),
        sigma=sig,
        filtered=filt,
    )


def match_to_truth(det, W, spk_t, spk_i, fs_hz, window_ms=1.5):
    """
    Link each detected event to the nearest TRUE spike (in time) among neurons
    that actually contribute to that event's channel, weighting the tie-break
    by W (a stronger contributor is the more likely source).

    Returns arrays aligned with det['t']:
      src_neuron : (M,) nearest contributing neuron id, or -1 if none in window
      dt_to_truth: (M,) |t_det - t_true| [s], or nan
    A detected event with src_neuron == -1 is a likely false positive.
    """
    det_t = det['t']
    det_ch = det['ch']
    M = len(det_t)
    src = np.full(M, -1, dtype=np.int64)
    dts = np.full(M, np.nan, dtype=float)
    win = window_ms * 1e-3

    spk_t = np.asarray(spk_t)
    spk_i = np.asarray(spk_i)
    # Sort spikes by time for fast windowing.
    o = np.argsort(spk_t)
    st = spk_t[o]
    si = spk_i[o]

    for m in range(M):
        e = det_ch[m]
        lo = np.searchsorted(st, det_t[m] - win, 'left')
        hi = np.searchsorted(st, det_t[m] + win, 'right')
        if hi <= lo:
            continue
        cand_i = si[lo:hi]
        cand_t = st[lo:hi]
        # Score = temporal closeness, broken by contribution weight W(n,e).
        wcol = W[cand_i, e]
        # ignore candidates that do not contribute to this channel
        good = wcol > 0
        if not np.any(good):
            continue
        cand_i = cand_i[good]
        cand_t = cand_t[good]
        wcol = wcol[good]
        dt = np.abs(cand_t - det_t[m])
        # nearest in time; if ties, largest weight
        best = np.lexsort((-wcol, dt))[0]
        src[m] = int(cand_i[best])
        dts[m] = float(dt[best])
    return src, dts
