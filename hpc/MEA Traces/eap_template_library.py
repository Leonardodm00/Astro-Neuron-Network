#!/usr/bin/env python3
# eap_template_library.py
# =============================================================================
# Extracellular action-potential (EAP) TEMPLATE LIBRARY generator.
#
# PURPOSE
# -------
# The Astro-Neuron-Network HPC campaigns save spike TIMES only (no membrane
# voltage). To synthesise virtual multi-electrode-array (MEA) traces from
# those spike times we need a canonical extracellular waveform w(tau) to place
# at every spike. This module produces a LIBRARY of such waveforms by
# simulating a minimal single-compartment Hodgkin-Huxley (HH) cortical neuron
# and extracting an EAP surrogate from each simulated action potential.
#
# MODEL (source: project knowledge base, full text)
#   Pospischil et al. (2008) "Minimal Hodgkin-Huxley type models for different
#   classes of cortical and thalamic neurons", Biol Cybern 99:427-441.
#   Fast Na (I_Na) and delayed-rectifier K (I_Kd) from Traub and Miles (1991);
#   slow M-current (I_M) from Yamada et al. (1989). Regular-spiking (RS)
#   excitatory class. The per-cell fitted parameter distributions in Table 1
#   (mean +/- SD) are used to draw a heterogeneous library, so waveform
#   variability across templates reflects real fitted cell-to-cell spread.
#
# WHAT THIS MODULE DOES NOT DO (by design / separation of concerns)
#   - No electrode geometry, no distance kernel, no noise, no detection.
#   - No absolute microvolt calibration. Templates are NORMALISED so that the
#     dominant negative trough equals -1. The absolute scale (sigma, dipole
#     length, membrane area) is applied later, in the trace-synthesis stage.
#
# EAP SURROGATE (single-compartment caveat)
#   A space-clamped single compartment conserves current, so its TOTAL
#   membrane current equals the injected current (no spike shape). The
#   extracellular waveform is therefore taken from a SURROGATE of the local
#   somatic membrane current. Default: the capacitive current
#       w_raw(tau) = -C_m * dV/dtau
#   which is biphasic with a dominant NEGATIVE trough at the Na+ upstroke,
#   matching the polarity of real extracellular APs. Alternatives provided:
#       'ionic_sum' : -(I_Na + I_Kd + I_M)   (equals -C_m dV/dt minus I_app)
#       'neg_d2V'   : -d^2 V / dtau^2         (triphasic, line-source-like)
#
# UNITS
#   V in mV, t in ms, conductances in mS/cm^2, C_m in uF/cm^2,
#   currents in uA/cm^2  =>  dV/dt in mV/ms. Stored template time step in ms.
#
# USAGE
#   from eap_template_library import generate_template_library, save_library
#   lib = generate_template_library(n_templates=30, seed=0)
#   save_library(lib, 'eap_library.npz')
#   # or as a script:
#   python eap_template_library.py --out eap_library.npz --n_templates 30 --plot
#
# See smoke_test_eap_library.py for the correctness self-check.
# =============================================================================

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp


# =============================================================================
# Fixed biophysical constants (Pospischil 2008; Traub and Miles 1991)
# =============================================================================

C_M = 1.0        # uF/cm^2   membrane capacitance
E_NA = 50.0      # mV        Na reversal (Pospischil)
E_K = -90.0      # mV        K  reversal (Pospischil)
E_L = -70.0      # mV        leak reversal (rest); shape-insensitive
GL_DENSITY = 0.1 # mS/cm^2   leak conductance DENSITY. NB: Table 1 lists gleak
                 #           in nS (absolute); converting needs a soma area and
                 #           mainly sets rest/rheobase, NOT the AP shape, so a
                 #           fixed density is used and gleak is NOT sampled.


# RS-excitatory fitted parameter distribution, Pospischil 2008 Table 1
# (MEAN, SD). Used to draw the library. Units: mS/cm^2 except VT (mV),
# taumax (ms).
RS_EXC_DIST = {
    'gNa':    (50.0, 10.0),
    'gKd':    (4.8,  1.4),
    'VT':     (-61.5, 3.2),
    'gM':     (0.13, 0.05),
    'taumax': (1123.5, 500.5),
}

# Physiological clip ranges to keep random draws well-posed (positive
# conductances, plausible threshold). Loose on purpose.
RS_EXC_CLIP = {
    'gNa':    (20.0, 90.0),
    'gKd':    (2.0,  9.0),
    'VT':     (-72.0, -50.0),
    'gM':     (0.02, 0.35),
    'taumax': (300.0, 2600.0),
}


@dataclass
class HHParams:
    """One RS neuron parameter set (densities in mS/cm^2; VT mV; taumax ms)."""
    gNa: float = 50.0
    gKd: float = 5.0
    VT: float = -61.5
    gM: float = 0.13
    taumax: float = 1123.5
    gL: float = GL_DENSITY
    EL: float = E_L
    Cm: float = C_M
    ENa: float = E_NA
    EK: float = E_K


# =============================================================================
# Gating kinetics (guarded against removable 0/0 singularities)
# =============================================================================

def _trap(x, y):
    """
    Safe evaluation of  x / (exp(x / y) - 1)  with the removable-singularity
    limit  y  as x -> 0 (first-order: exp(x/y)-1 ~ x/y => x/(x/y) = y).

    x : array-like numerator argument
    y : scalar scale (the L'Hopital limit at x = 0)
    """
    x = np.asarray(x, dtype=float)
    small = np.abs(x / y) < 1e-6
    # Full formula where safe; limit y where x ~ 0.
    denom = np.expm1(np.where(small, 1.0, x) / y)  # expm1 for accuracy
    out = np.where(small, y, x / denom)
    return out


def _rates(V, VT):
    """Return (m_inf-style) alpha/beta rates [1/ms] for m, h, n at voltage V."""
    # I_Na activation m  (Traub and Miles 1991, shifted by VT)
    am = 0.32 * _trap(-(V - VT - 13.0), 4.0)          # = -0.32*(x)/(exp(-x/4)-1)
    bm = 0.28 * _trap((V - VT - 40.0), 5.0)           # = 0.28*(x)/(exp(x/5)-1)
    # I_Na inactivation h
    ah = 0.128 * np.exp(-(V - VT - 17.0) / 18.0)
    bh = 4.0 / (1.0 + np.exp(-(V - VT - 40.0) / 5.0))
    # I_Kd activation n
    an = 0.032 * _trap(-(V - VT - 15.0), 5.0)         # = -0.032*(x)/(exp(-x/5)-1)
    bn = 0.5 * np.exp(-(V - VT - 10.0) / 40.0)
    return am, bm, ah, bh, an, bn


def _m_current_steady_tau(V, taumax):
    """I_M gating: p_inf and tau_p [ms] (Yamada et al. 1989)."""
    p_inf = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    tau_p = taumax / (3.3 * np.exp((V + 35.0) / 20.0) + np.exp(-(V + 35.0) / 20.0))
    return p_inf, tau_p


# =============================================================================
# Model right-hand side and current decomposition
# =============================================================================

def _ionic_currents(V, m, h, n, p, hp: HHParams):
    """Return (I_Na, I_Kd, I_M, I_L) in uA/cm^2 (outward positive)."""
    I_Na = hp.gNa * (m ** 3) * h * (V - hp.ENa)
    I_Kd = hp.gKd * (n ** 4) * (V - hp.EK)
    I_M = hp.gM * p * (V - hp.EK)
    I_L = hp.gL * (V - hp.EL)
    return I_Na, I_Kd, I_M, I_L


def _rhs(t, y, hp: HHParams, I_app_func):
    """ODE RHS. y = [V, m, h, n, p]. I_app_func(t) -> uA/cm^2."""
    V, m, h, n, p = y
    am, bm, ah, bh, an, bn = _rates(V, hp.VT)
    p_inf, tau_p = _m_current_steady_tau(V, hp.taumax)

    I_Na, I_Kd, I_M, I_L = _ionic_currents(V, m, h, n, p, hp)
    I_app = I_app_func(t)

    dV = (I_app - I_L - I_Na - I_Kd - I_M) / hp.Cm
    dm = am * (1.0 - m) - bm * m
    dh = ah * (1.0 - h) - bh * h
    dn = an * (1.0 - n) - bn * n
    dp = (p_inf - p) / tau_p
    return [dV, dm, dh, dn, dp]


def _steady_gates(V, hp: HHParams):
    """Steady-state gates at voltage V (for initial condition at rest)."""
    am, bm, ah, bh, an, bn = _rates(V, hp.VT)
    m0 = am / (am + bm)
    h0 = ah / (ah + bh)
    n0 = an / (an + bn)
    p0, _ = _m_current_steady_tau(V, hp.taumax)
    return float(m0), float(h0), float(n0), float(p0)


# =============================================================================
# Simulation
# =============================================================================

@dataclass
class SimConfig:
    t_total_ms: float = 250.0    # total simulated time
    t_on_ms: float = 20.0        # step-current onset
    I_app: float = 7.0           # uA/cm^2 drive during the step
    dt_hr_ms: float = 0.02       # high-res output grid (50 kHz) for extraction
    rtol: float = 1e-7
    atol: float = 1e-9


def simulate_hh(hp: HHParams, cfg: SimConfig):
    """
    Integrate the minimal HH model under a step current.

    Returns
    -------
    t   : (T,) time grid [ms] at dt_hr_ms
    V   : (T,) membrane potential [mV]
    dV  : (T,) dV/dt [mV/ms] (recomputed exactly from the RHS, not finite-diff)
    curr: dict of ionic current arrays [uA/cm^2]
    """
    def I_app_func(t):
        return cfg.I_app if t >= cfg.t_on_ms else 0.0

    V0 = hp.EL
    m0, h0, n0, p0 = _steady_gates(V0, hp)
    y0 = [V0, m0, h0, n0, p0]

    t_eval = np.arange(0.0, cfg.t_total_ms + cfg.dt_hr_ms, cfg.dt_hr_ms)
    sol = solve_ivp(
        _rhs, (0.0, cfg.t_total_ms), y0,
        t_eval=t_eval, args=(hp, I_app_func),
        method='LSODA', rtol=cfg.rtol, atol=cfg.atol, max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError('HH integration failed: ' + sol.message)

    V, m, h, n, p = sol.y
    # Exact dV/dt from the RHS at each output sample (vectorised).
    am, bm, ah, bh, an, bn = _rates(V, hp.VT)
    I_Na, I_Kd, I_M, I_L = _ionic_currents(V, m, h, n, p, hp)
    I_app_arr = np.where(sol.t >= cfg.t_on_ms, cfg.I_app, 0.0)
    dV = (I_app_arr - I_L - I_Na - I_Kd - I_M) / hp.Cm

    curr = dict(I_Na=I_Na, I_Kd=I_Kd, I_M=I_M, I_L=I_L, I_app=I_app_arr)
    return sol.t, V, dV, curr


# =============================================================================
# EAP surrogate extraction and template building
# =============================================================================

@dataclass
class TemplateConfig:
    surrogate: str = 'capacitive'   # 'capacitive' | 'ionic_sum' | 'neg_d2V'
    pre_ms: float = 0.8             # window before the trough
    post_ms: float = 1.6            # window after the trough
    n_avg_spikes: int = 3           # average the last k steady-state spikes
    v_thresh_mv: float = 0.0        # spike = upward crossing of this V level


def _surrogate_trace(V, dV, curr, dt_ms, kind):
    """Compute the chosen EAP surrogate on the full high-res grid."""
    if kind == 'capacitive':
        # w_raw = -C_m dV/dt  (biphasic; negative trough at upstroke)
        return -C_M * dV
    if kind == 'ionic_sum':
        return -(curr['I_Na'] + curr['I_Kd'] + curr['I_M'])
    if kind == 'neg_d2V':
        d2V = np.gradient(dV, dt_ms)
        return -d2V
    raise ValueError('unknown surrogate: ' + repr(kind))


def _find_spike_indices(V, level):
    """Indices of upward crossings of `level` (spike onsets)."""
    below = V[:-1] < level
    above = V[1:] >= level
    cross = np.where(below & above)[0] + 1
    return cross


def build_template(hp: HHParams, sim_cfg: SimConfig, tpl_cfg: TemplateConfig):
    """
    Simulate one neuron and return its normalised EAP template.

    Returns
    -------
    dict with:
      w          : (n_hr,) template, normalised so min(w) == -1
      dt_hr_ms   : sample interval [ms]
      trough_idx : index of the negative trough within w
      n_spikes   : number of spikes elicited
    """
    t, V, dV, curr = simulate_hh(hp, sim_cfg)
    w_full = _surrogate_trace(V, dV, curr, sim_cfg.dt_hr_ms, tpl_cfg.surrogate)

    spk = _find_spike_indices(V, tpl_cfg.v_thresh_mv)
    if len(spk) == 0:
        raise RuntimeError('no spikes elicited; increase I_app or check params')

    pre = int(round(tpl_cfg.pre_ms / sim_cfg.dt_hr_ms))
    post = int(round(tpl_cfg.post_ms / sim_cfg.dt_hr_ms))
    n_hr = pre + post

    # Use the LAST k spikes (steady state, after adaptation) that have room.
    usable = [s for s in spk if (s - pre) >= 0 and (s + post) < len(w_full)]
    if len(usable) == 0:
        raise RuntimeError('spikes too close to trace edges for the window')
    chosen = usable[-min(tpl_cfg.n_avg_spikes, len(usable)):]

    stack = []
    for s in chosen:
        # Refine to the negative trough of the surrogate near the V-crossing:
        lo = max(s - pre, 0)
        hi = min(s + post, len(w_full))
        local = w_full[lo:hi]
        tr_local = int(np.argmin(local))       # most-negative sample
        tr = lo + tr_local
        seg = w_full[tr - pre: tr + post]
        if len(seg) == n_hr:
            stack.append(seg)
    if len(stack) == 0:
        raise RuntimeError('trough windowing failed')

    w = np.mean(np.vstack(stack), axis=0)
    # Normalise so the dominant NEGATIVE trough is exactly -1.
    trough_val = np.min(w)
    if trough_val >= 0:
        raise RuntimeError('surrogate has no negative trough; check polarity')
    w = w / (-trough_val)
    trough_idx = int(np.argmin(w))

    return dict(w=w.astype(np.float64), dt_hr_ms=sim_cfg.dt_hr_ms,
                trough_idx=trough_idx, n_spikes=int(len(spk)))


# =============================================================================
# Library generation
# =============================================================================

def _draw_params(rng):
    """Draw one RS-exc HHParams from Table 1 mean+/-SD, clipped to ranges."""
    vals = {}
    for k, (mu, sd) in RS_EXC_DIST.items():
        lo, hi = RS_EXC_CLIP[k]
        vals[k] = float(np.clip(rng.normal(mu, sd), lo, hi))
    return HHParams(gNa=vals['gNa'], gKd=vals['gKd'], VT=vals['VT'],
                    gM=vals['gM'], taumax=vals['taumax'])


def generate_template_library(n_templates=30, seed=0,
                              sim_cfg: SimConfig = None,
                              tpl_cfg: TemplateConfig = None):
    """
    Build a library of EAP templates from random RS-exc HH neurons.

    Returns a dict:
      templates   : (n_templates, n_hr) float64, each normalised (trough = -1)
      dt_hr_ms    : scalar sample interval [ms]
      trough_idx  : (n_templates,) trough index per template
      params      : (n_templates, 5) drawn [gNa, gKd, VT, gM, taumax]
      param_names : list of the 5 names
      meta        : dict of configuration / provenance
    """
    if sim_cfg is None:
        sim_cfg = SimConfig()
    if tpl_cfg is None:
        tpl_cfg = TemplateConfig()

    rng = np.random.default_rng(seed)
    templates, troughs, draws = [], [], []
    for _ in range(n_templates):
        hp = _draw_params(rng)
        res = build_template(hp, sim_cfg, tpl_cfg)
        templates.append(res['w'])
        troughs.append(res['trough_idx'])
        draws.append([hp.gNa, hp.gKd, hp.VT, hp.gM, hp.taumax])

    templates = np.vstack(templates)
    meta = dict(
        surrogate=tpl_cfg.surrogate, seed=int(seed),
        pre_ms=tpl_cfg.pre_ms, post_ms=tpl_cfg.post_ms,
        I_app=sim_cfg.I_app, model='Pospischil2008_RS_exc',
        Cm=C_M, ENa=E_NA, EK=E_K, EL=E_L, gL=GL_DENSITY,
    )
    return dict(
        templates=templates,
        dt_hr_ms=float(sim_cfg.dt_hr_ms),
        trough_idx=np.asarray(troughs, dtype=np.int64),
        params=np.asarray(draws, dtype=np.float64),
        param_names=['gNa', 'gKd', 'VT', 'gM', 'taumax'],
        meta=meta,
    )


# =============================================================================
# Resampling to the recording grid
# =============================================================================

def resample_template_to_fs(w, dt_hr_ms, fs_hz):
    """
    Resample one template from the high-res grid (dt_hr_ms) to a recording
    sampling rate fs_hz [Hz], preserving the trough amplitude by linear
    interpolation. Returns (w_rs, trough_idx_rs).

    Linear interpolation is used deliberately: it is monotone-safe and does
    not introduce Gibbs ringing near the sharp trough (a concern with sinc /
    Fourier resampling on a short, non-periodic transient).
    """
    dt_hr_s = dt_hr_ms * 1e-3
    n = len(w)
    t_hr = np.arange(n) * dt_hr_s
    dt_rs = 1.0 / fs_hz
    t_rs = np.arange(0.0, t_hr[-1] + dt_rs * 0.5, dt_rs)
    w_rs = np.interp(t_rs, t_hr, w)
    return w_rs, int(np.argmin(w_rs))


# =============================================================================
# IO
# =============================================================================

def save_library(lib, path):
    """Save a library dict to .npz (meta stored as a JSON string)."""
    import json
    np.savez_compressed(
        path,
        templates=lib['templates'],
        dt_hr_ms=np.float64(lib['dt_hr_ms']),
        trough_idx=lib['trough_idx'],
        params=lib['params'],
        param_names=np.array(lib['param_names']),
        meta_json=np.array(json.dumps(lib['meta'])),
    )
    return path


def load_library(path):
    """Load a library dict saved by save_library."""
    import json
    d = np.load(path, allow_pickle=False)
    return dict(
        templates=d['templates'],
        dt_hr_ms=float(d['dt_hr_ms']),
        trough_idx=d['trough_idx'],
        params=d['params'],
        param_names=list(d['param_names']),
        meta=json.loads(str(d['meta_json'])),
    )


# =============================================================================
# CLI
# =============================================================================

def _build_parser():
    p = argparse.ArgumentParser(description='Generate an EAP template library.')
    p.add_argument('--out', default='eap_library.npz')
    p.add_argument('--n_templates', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--surrogate', default='capacitive',
                   choices=['capacitive', 'ionic_sum', 'neg_d2V'])
    p.add_argument('--I_app', type=float, default=7.0)
    p.add_argument('--plot', action='store_true',
                   help='save eap_library_preview.png (uses Agg backend)')
    return p


def main():
    args = _build_parser().parse_args()
    sim_cfg = SimConfig(I_app=args.I_app)
    tpl_cfg = TemplateConfig(surrogate=args.surrogate)
    lib = generate_template_library(
        n_templates=args.n_templates, seed=args.seed,
        sim_cfg=sim_cfg, tpl_cfg=tpl_cfg)
    save_library(lib, args.out)
    print('[eap] saved %d templates -> %s' % (lib['templates'].shape[0], args.out))
    print('[eap] template length: %d samples at dt=%.3f ms (%.1f Hz)'
          % (lib['templates'].shape[1], lib['dt_hr_ms'], 1e3 / lib['dt_hr_ms']))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t = np.arange(lib['templates'].shape[1]) * lib['dt_hr_ms']
        fig, ax = plt.subplots(figsize=(6, 4))
        for w in lib['templates']:
            ax.plot(t, w, color='0.5', alpha=0.35, lw=0.8)
        ax.plot(t, lib['templates'].mean(0), color='crimson', lw=2, label='mean')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel('time [ms]')
        ax.set_ylabel('normalised EAP (trough = -1)')
        ax.set_title('EAP template library (n=%d)' % lib['templates'].shape[0])
        ax.legend()
        fig.tight_layout()
        fig.savefig('eap_library_preview.png', dpi=140)
        print('[eap] preview -> eap_library_preview.png')


if __name__ == '__main__':
    main()
