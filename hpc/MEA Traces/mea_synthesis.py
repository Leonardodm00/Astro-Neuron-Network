#!/usr/bin/env python3
# mea_synthesis.py
# =============================================================================
# Synthesise multi-channel virtual-MEA traces from spike times.
#
# SEPARATION OF CONCERNS
#   Consumes: spike times/ids, a per-(neuron, electrode) weight matrix W
#   (from mea_probe), an EAP template LIBRARY resampled to the recording rate,
#   and a per-neuron template assignment. Produces the raw (unfiltered)
#   E-channel trace = distance-scaled sum of templates at spike times + noise.
#   No filtering and no detection here (that is mea_detection).
#
# SIGNAL MODEL
#   channel_e(t) = sum_n W(n,e) * sum_{spikes k of n} w_{tmpl(n)}(t - t_k)
#                  + eta_e(t),   eta_e ~ N(0, noise_rms^2)  i.i.d. per sample
#   The template w is normalised (trough = -1); W carries A_ref and sub-site
#   averaging. Near neurons dominate (signal); far neurons form the biological
#   background; unresolved far neurons + instrument noise are the eta term.
#
# EFFICIENCY
#   Each neuron's spike train is rendered ONCE into a scratch buffer (template
#   scatter-add at spike samples), then added to the E channels scaled by
#   W(n,e). Cost ~ (total spikes * template_len) + Nn * E adds. FFTs avoided
#   (templates are short and spikes sparse).
#
# UNITS: times in seconds, amplitudes in uV (or sigma-units, matching A_ref).
# =============================================================================

import numpy as np
from scipy import signal as _sps


def assign_templates(Nn, n_templates, seed=0):
    """Assign each neuron one template index from the library (seeded)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_templates, size=Nn).astype(np.int64)


def prepare_library_at_fs(lib, fs_hz):
    """
    Resample every template in the library to the recording rate fs_hz.

    Returns a dict:
      wf          : list of (len_i,) resampled templates (trough = -1)
      trough_idx  : (n_templates,) trough sample index of each resampled tmpl
    Uses linear interpolation (see eap_template_library.resample_template_to_fs).
    """
    from eap_template_library import resample_template_to_fs
    wf, tr = [], []
    for w in lib['templates']:
        w_rs, tr_rs = resample_template_to_fs(w, lib['dt_hr_ms'], fs_hz)
        wf.append(np.asarray(w_rs, dtype=float))
        tr.append(int(tr_rs))
    return dict(wf=wf, trough_idx=np.asarray(tr, dtype=np.int64))


def calibrate_noise_rms(fs_hz, band, target_post_uv, order=4, seed=12345,
                        n_probe=None):
    """
    Return the PRE-filter white-noise RMS such that, AFTER the band-pass, the
    robust noise estimate sigma_hat = median(|x|)/0.6745 ~= target_post_uv.

    The band-pass attenuates broadband white noise; this measures that gain on
    a noise-only probe and inverts it. Deterministic given `seed`.
    """
    if n_probe is None:
        n_probe = int(max(fs_hz * 5.0, 50000))    # >= 5 s probe
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_probe)              # unit-RMS white noise
    xf = bandpass(x[None, :], fs_hz, band[0], band[1], order=order)[0]
    sigma_hat = np.median(np.abs(xf)) / 0.6745
    gain = sigma_hat / 1.0                         # post/pre RMS ratio (robust)
    if gain <= 0:
        raise RuntimeError('degenerate filter gain in noise calibration')
    return target_post_uv / gain


def synthesize_traces(spk_t, spk_i, W, lib_fs, tmpl_assign, fs_hz, simtime_s,
                      noise_rms, reach=None, seed=0, return_clean=False):
    """
    Build the E-channel raw trace.

    Parameters
    ----------
    spk_t   : (K,) spike times [s]
    spk_i   : (K,) neuron index per spike
    W       : (Nn, E) weight matrix [uV] from mea_probe.compute_weights
    lib_fs  : dict from prepare_library_at_fs (templates already at fs_hz)
    tmpl_assign : (Nn,) template index per neuron
    fs_hz   : recording sampling rate [Hz]
    simtime_s : total duration [s]
    noise_rms : PRE-filter white-noise RMS [uV] (0 for a clean trace)
    reach   : (Nn,) bool mask; neurons with False are skipped (default all True)
    seed    : RNG seed for the additive noise
    return_clean : if True also return the noise-free signal

    Returns
    -------
    traces : (E, T) raw trace (signal + noise)
    t      : (T,) time axis [s]
    (clean): (E, T) noise-free signal, only if return_clean
    """
    Nn, E = W.shape
    T = int(round(simtime_s * fs_hz))
    t = np.arange(T) / fs_hz
    clean = np.zeros((E, T), dtype=float)

    if reach is None:
        reach = np.ones(Nn, dtype=bool)

    wf = lib_fs['wf']
    tr = lib_fs['trough_idx']

    # Group spike sample-indices by neuron.
    spk_i = np.asarray(spk_i)
    spk_samp = np.round(np.asarray(spk_t) * fs_hz).astype(np.int64)
    order = np.argsort(spk_i, kind='stable')
    spk_i_s = spk_i[order]
    spk_samp_s = spk_samp[order]
    # boundaries of each neuron's block
    uniq, starts = np.unique(spk_i_s, return_index=True)
    starts = list(starts) + [len(spk_i_s)]

    scratch = np.zeros(T, dtype=float)
    for bi, n in enumerate(uniq):
        if not reach[n]:
            continue
        s0, s1 = starts[bi], starts[bi + 1]
        samps = spk_samp_s[s0:s1]
        w = wf[tmpl_assign[n]]
        toff = tr[tmpl_assign[n]]
        Lw = len(w)

        scratch[:] = 0.0
        touched_lo = T
        touched_hi = 0
        for smp in samps:
            a = smp - toff              # start sample so trough lands on `smp`
            b = a + Lw
            lo = max(a, 0)
            hi = min(b, T)
            if lo >= hi:
                continue
            scratch[lo:hi] += w[lo - a: hi - a]
            touched_lo = min(touched_lo, lo)
            touched_hi = max(touched_hi, hi)
        if touched_hi <= touched_lo:
            continue
        seg = scratch[touched_lo:touched_hi]
        we = W[n]                       # (E,)
        for e in range(E):
            if we[e] != 0.0:
                clean[e, touched_lo:touched_hi] += we[e] * seg

    if noise_rms and noise_rms > 0:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((E, T)) * noise_rms
        traces = clean + noise
    else:
        traces = clean.copy()

    if return_clean:
        return traces, t, clean
    return traces, t


# -----------------------------------------------------------------------------
# Band-pass lives here as a shared primitive (used by calibration + detection).
# -----------------------------------------------------------------------------

def bandpass(x, fs_hz, lo_hz, hi_hz, order=4):
    """
    Zero-phase Butterworth band-pass. x may be (T,) or (E, T). Returns same
    shape. hi_hz is clipped just below Nyquist for safety.
    """
    nyq = 0.5 * fs_hz
    hi = min(hi_hz, 0.999 * nyq)
    sos = _sps.butter(order, [lo_hz / nyq, hi / nyq], btype='band', output='sos')
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return _sps.sosfiltfilt(sos, x)
    return _sps.sosfiltfilt(sos, x, axis=-1)
