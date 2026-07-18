#!/usr/bin/env python3
# smoke_test_eap_library.py
# =============================================================================
# Correctness self-check for eap_template_library.py.
#
# WHAT IT VERIFIES (each check is independent and prints PASS/FAIL)
#   1  HH sanity: quiescent at rest (no drive), spikes with drive, AP overshoot.
#   2  Surrogate polarity: capacitive surrogate has a dominant NEGATIVE trough.
#   3  dV/dt cross-check: exact RHS dV/dt matches finite-difference of V.
#   4  Normalisation: every template has min == -1 and that is the global min.
#   5  Biphasic structure: a positive repolarisation lobe follows the trough.
#   6  Duration sanity: trough sits inside the window; template ~ 1-3 ms wide.
#   7  Determinism: same seed -> identical library (bitwise-equal arrays).
#   8  Heterogeneity: distinct templates are not identical; measurable spread.
#   9  Resampling: round-trip to fs preserves trough position and amplitude.
#  10  Parameter provenance: drawn params lie within the clip ranges.
#
# HOW TO RUN
#   python smoke_test_eap_library.py
#   # exits 0 if all pass, 1 otherwise. Fast (a few seconds).
#
# QUICK INTERACTIVE SNIPPET (drop into a Python shell)
#   import eap_template_library as E
#   lib = E.generate_template_library(n_templates=5, seed=0)
#   import numpy as np
#   assert np.allclose(lib['templates'].min(axis=1), -1.0)   # trough = -1
#   print(lib['templates'].shape, lib['dt_hr_ms'])
# =============================================================================

import sys

import numpy as np

import eap_template_library as E


class Check:
    def __init__(self):
        self.rows = []

    def run(self, name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001  (smoke test: report any error)
            ok, detail = False, 'EXC: %r' % (exc,)
        self.rows.append((name, ok, detail))
        print('[%s] %-34s %s' % ('PASS' if ok else 'FAIL', name, detail))
        return ok

    def all_ok(self):
        return all(ok for _, ok, _ in self.rows)


# -----------------------------------------------------------------------------
# Individual checks
# -----------------------------------------------------------------------------

def check_hh_sanity():
    hp = E.HHParams()
    # No drive -> should stay near rest, no spikes.
    cfg0 = E.SimConfig(I_app=0.0, t_total_ms=120.0)
    _, V0, _, _ = E.simulate_hh(hp, cfg0)
    quiescent = (V0.max() < -40.0)                      # never approaches spike
    near_rest = abs(np.median(V0) - hp.EL) < 5.0
    # With drive -> spikes and AP overshoot above 0 mV.
    cfg1 = E.SimConfig(I_app=7.0, t_total_ms=250.0)
    _, V1, _, _ = E.simulate_hh(hp, cfg1)
    spk = E._find_spike_indices(V1, 0.0)
    spikes = len(spk) >= 3
    overshoot = V1.max() > 10.0
    ok = quiescent and near_rest and spikes and overshoot
    return ok, ('rest~%.1f mV, Vmax(no drive)=%.1f, n_spk=%d, Vmax=%.1f'
                % (np.median(V0), V0.max(), len(spk), V1.max()))


def check_surrogate_polarity():
    hp = E.HHParams()
    cfg = E.SimConfig()
    t, V, dV, curr = E.simulate_hh(hp, cfg)
    w = E._surrogate_trace(V, dV, curr, cfg.dt_hr_ms, 'capacitive')
    # During an AP the dominant deflection must be negative (trough).
    ok = (w.min() < 0) and (abs(w.min()) > abs(w.max()))
    return ok, 'min=%.2f  max=%.2f (|trough|>|peak| required)' % (w.min(), w.max())


def check_dvdt_consistency():
    hp = E.HHParams()
    cfg = E.SimConfig()
    t, V, dV, curr = E.simulate_hh(hp, cfg)
    # Finite-difference dV/dt vs exact RHS dV/dt on the interior.
    dV_fd = np.gradient(V, cfg.dt_hr_ms)
    # Compare on a robust scale (max |dV| during spikes is large).
    denom = np.max(np.abs(dV)) + 1e-9
    rel = np.max(np.abs(dV_fd - dV)) / denom
    ok = rel < 0.05                                     # < 5% peak-normalised
    return ok, 'max relative dV/dt mismatch = %.3f (< 0.05)' % rel


def check_normalisation():
    lib = E.generate_template_library(n_templates=8, seed=1)
    mins = lib['templates'].min(axis=1)
    ok = np.allclose(mins, -1.0, atol=1e-9)
    # Trough index must coincide with the global min sample.
    idx_ok = all(int(np.argmin(w)) == int(ti)
                 for w, ti in zip(lib['templates'], lib['trough_idx']))
    return (ok and idx_ok), 'min(w) in [%.4f, %.4f]; trough_idx matches=%s' % (
        mins.min(), mins.max(), idx_ok)


def check_biphasic():
    lib = E.generate_template_library(n_templates=8, seed=2)
    ratios = []
    for w, ti in zip(lib['templates'], lib['trough_idx']):
        post_peak = w[ti:].max()                        # repolarisation lobe
        ratios.append(post_peak)
    ratios = np.asarray(ratios)
    ok = np.all(ratios > 0.05)                          # a real positive lobe
    return ok, 'min post-trough positive lobe = %.3f (> 0.05)' % ratios.min()


def check_duration():
    lib = E.generate_template_library(n_templates=8, seed=3)
    dt = lib['dt_hr_ms']
    n = lib['templates'].shape[1]
    total_ms = n * dt
    # Trough must be strictly inside the window (not clipped at an edge).
    inside = np.all((lib['trough_idx'] > 2) & (lib['trough_idx'] < n - 2))
    ok = inside and (0.5 < total_ms < 5.0)
    return ok, 'window=%.2f ms, %d samples, trough inside=%s' % (
        total_ms, n, inside)


def check_determinism():
    a = E.generate_template_library(n_templates=6, seed=7)
    b = E.generate_template_library(n_templates=6, seed=7)
    ok = (np.array_equal(a['templates'], b['templates'])
          and np.array_equal(a['params'], b['params']))
    # And a different seed should differ.
    c = E.generate_template_library(n_templates=6, seed=8)
    differ = not np.array_equal(a['templates'], c['templates'])
    return (ok and differ), 'seed match=%s, seed8 differs=%s' % (ok, differ)


def check_heterogeneity():
    lib = E.generate_template_library(n_templates=12, seed=4)
    T = lib['templates']
    # Pairwise correlation: distinct templates should not be identical.
    Tn = (T - T.mean(axis=1, keepdims=True))
    Tn /= (np.linalg.norm(Tn, axis=1, keepdims=True) + 1e-12)
    corr = Tn @ Tn.T
    off = corr[~np.eye(len(T), dtype=bool)]
    ok = (off.max() < 0.99999) and (off.min() < 0.999)  # not all-identical
    spread = T.min(axis=1)                               # all -1 by norm; use width
    # Width proxy: samples below -0.5.
    widths = (T < -0.5).sum(axis=1)
    ok = ok and (widths.std() > 0)
    return ok, 'max off-diag corr=%.4f; trough-width std=%.2f samp' % (
        off.max(), widths.std())


def check_resampling():
    lib = E.generate_template_library(n_templates=4, seed=5)
    w = lib['templates'][0]
    dt = lib['dt_hr_ms']
    fs = 10110.09
    w_rs, tr_rs = E.resample_template_to_fs(w, dt, fs)
    # Trough amplitude after resampling should stay close to -1 (linear interp
    # slightly undershoots the peak; allow a few percent).
    amp_ok = abs(w_rs.min() + 1.0) < 0.10
    # Trough time should be preserved to within one resampled sample.
    t_tr_hr = np.argmin(w) * dt * 1e-3
    t_tr_rs = tr_rs / fs
    time_ok = abs(t_tr_hr - t_tr_rs) <= 1.5 / fs
    ok = amp_ok and time_ok
    return ok, 'resamp trough amp=%.3f, dt_trough=%.2e s (<=%.2e)' % (
        w_rs.min(), abs(t_tr_hr - t_tr_rs), 1.5 / fs)


def check_param_ranges():
    lib = E.generate_template_library(n_templates=20, seed=6)
    names = lib['param_names']
    ok = True
    detail = []
    for j, nm in enumerate(names):
        lo, hi = E.RS_EXC_CLIP[nm]
        col = lib['params'][:, j]
        in_range = np.all((col >= lo - 1e-9) & (col <= hi + 1e-9))
        ok = ok and in_range
        detail.append('%s:%s' % (nm, 'ok' if in_range else 'OUT'))
    return ok, ' '.join(detail)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print('=' * 68)
    print('EAP template library smoke test')
    print('=' * 68)
    c = Check()
    c.run('1_hh_sanity', check_hh_sanity)
    c.run('2_surrogate_polarity', check_surrogate_polarity)
    c.run('3_dvdt_consistency', check_dvdt_consistency)
    c.run('4_normalisation', check_normalisation)
    c.run('5_biphasic_structure', check_biphasic)
    c.run('6_duration_sanity', check_duration)
    c.run('7_determinism', check_determinism)
    c.run('8_heterogeneity', check_heterogeneity)
    c.run('9_resampling_roundtrip', check_resampling)
    c.run('10_param_provenance', check_param_ranges)
    print('-' * 68)
    if c.all_ok():
        print('ALL CHECKS PASSED')
        return 0
    print('SOME CHECKS FAILED')
    return 1


if __name__ == '__main__':
    sys.exit(main())
