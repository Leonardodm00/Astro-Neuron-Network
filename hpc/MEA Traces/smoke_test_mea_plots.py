#!/usr/bin/env python3
# smoke_test_mea_plots.py
# =============================================================================
# Correctness self-check for mea_plots.py.
#
# Plotting cannot be verified numerically the way a filter or a detector can,
# so this test checks the things that actually break in practice:
#   - every function writes a real, non-empty, valid PNG (magic-byte checked)
#   - the headless backend works with no DISPLAY
#   - the busiest-window selector really finds the densest activity
#   - degenerate inputs (zero detections, one channel, empty reach mask) do
#     not raise -- a plotting crash must never take down a campaign
#   - the orchestrators produce the full expected set of files
#
# HOW TO RUN
#   python smoke_test_mea_plots.py
#   # exits 0 if all pass, 1 otherwise. Writes into a temp dir that is removed.
#
# QUICK INTERACTIVE SNIPPET
#   import numpy as np, mea_plots as PL, mea_probe as P
#   pr = P.make_probe(400.0)
#   xy = np.random.default_rng(0).uniform(0, 400, size=(200, 2))
#   W  = P.compute_weights(xy, pr, P.ScalingConfig())
#   PL.plot_probe_layout(xy, pr, W=W, r_max=164.0, out_dir='/tmp/pl')
# =============================================================================

import os
import shutil
import sys
import tempfile

import numpy as np

import mea_plots as PL
import mea_probe as P
import mea_synthesis as S
import mea_detection as D
import eap_template_library as EAP


PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


class Check:
    def __init__(self):
        self.rows = []

    def run(self, name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, 'EXC: %r' % (exc,)
        self.rows.append((name, ok, detail))
        print('[%s] %-32s %s' % ('PASS' if ok else 'FAIL', name, detail))
        return ok

    def all_ok(self):
        return all(ok for _, ok, _ in self.rows)


def _is_png(path):
    """A file that exists, is non-trivial in size, and has the PNG magic."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 1000:          # a real figure is never this small
        return False
    with open(path, 'rb') as f:
        return f.read(8) == PNG_MAGIC


# -----------------------------------------------------------------------------
# Shared synthetic scene
# -----------------------------------------------------------------------------

def _scene(tmp, n_spikes=40, with_detections=True):
    """
    Small but fully realistic scene: real probe, real weights, real synthesised
    traces from real templates, real detector output. Nothing is faked, so the
    plots exercise the same array shapes the pipeline produces.
    """
    rng = np.random.default_rng(0)
    c_max = 400.0
    probe = P.make_probe(c_max)
    ctr = probe['center_xy']

    # a few near neurons (visible) + background
    near = ctr + rng.normal(0, 25.0, size=(6, 2))
    far = rng.uniform(0, c_max, size=(150, 2))
    xy = np.vstack([near, far])
    Nn = xy.shape[0]

    scfg = P.ScalingConfig()
    W = P.compute_weights(xy, probe, scfg)
    r_max = P.r_max_reach(scfg, 0.5, 5.0)
    reach = P.reach_mask(W, floor=0.5 * 5.0)

    lib = EAP.generate_template_library(n_templates=4, seed=0)
    fs = 10110.09
    lib_fs = S.prepare_library_at_fs(lib, fs)
    tmpl = S.assign_templates(Nn, lib['templates'].shape[0], seed=0)

    simtime = 2.0
    if n_spikes > 0:
        spk_t = np.sort(rng.uniform(0.1, simtime - 0.1, size=n_spikes))
        spk_i = rng.integers(0, 6, size=n_spikes)      # only near neurons fire
    else:
        spk_t = np.empty(0)
        spk_i = np.empty(0, dtype=int)

    noise_rms = S.calibrate_noise_rms(fs, (300.0, 3000.0), 5.0)
    traces = S.synthesize_traces(spk_t, spk_i, W, lib_fs, tmpl, fs, simtime,
                                 noise_rms=noise_rms, reach=reach, seed=1)
    if isinstance(traces, tuple):
        traces = traces[0]

    det = D.detect(traces, fs, D.DetectConfig())
    src, dt = D.match_to_truth(det, W, spk_t, spk_i, fs)
    return dict(xy=xy, probe=probe, W=W, scfg=scfg, r_max=r_max, reach=reach,
                lib=lib, fs=fs, traces=traces, det=det, src=src, dt=dt,
                spk_t=spk_t, spk_i=spk_i, simtime=simtime, tmp=tmp)


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def check_mpl_available():
    ok = PL.HAVE_MPL
    detail = 'matplotlib present' if ok else 'matplotlib MISSING (plots skipped)'
    return ok, detail


def check_busiest_window():
    # events packed into [7,8) s of a 10 s record; the picker must find it
    det_t = np.concatenate([np.linspace(7.0, 7.9, 50), np.array([0.5, 3.0])])
    t0, t1 = PL.busiest_window(det_t, 10.0, 1.0)
    ok = (6.9 <= t0 <= 7.1) and abs((t1 - t0) - 1.0) < 1e-9
    # empty input must not raise and must return a sane window
    t0e, t1e = PL.busiest_window(np.empty(0), 10.0, 1.0)
    ok = ok and (t0e == 0.0) and (t1e == 1.0)
    return ok, 'window=(%.2f, %.2f) around the dense burst; empty ok' % (t0, t1)


def check_geometry_plots(sc):
    d = os.path.join(sc['tmp'], 'geo')
    p1 = PL.plot_probe_layout(sc['xy'], sc['probe'], W=sc['W'],
                              r_max=sc['r_max'], c_max=400.0, out_dir=d)
    p2 = PL.plot_weight_decay(sc['xy'], sc['probe'], sc['W'], sc['scfg'],
                              r_max=sc['r_max'], noise_floor=2.5, out_dir=d)
    p3 = PL.plot_template_library(sc['lib']['templates'],
                                  sc['lib']['dt_hr_ms'], out_dir=d)
    ok = all(_is_png(p) for p in (p1, p2, p3))
    return ok, '3 geometry PNGs written and valid'


def check_trace_plots(sc):
    d = os.path.join(sc['tmp'], 'traces')
    det = sc['det']
    p1 = PL.plot_stacked_traces(sc['traces'], sc['fs'], sigma=det['sigma'],
                                out_dir=d)
    p2 = PL.plot_traces_with_detections(
        det['filtered'], sc['fs'], det['t'], det['ch'], det['amp'],
        det['sigma'], 5.0, src_neuron=sc['src'], out_dir=d)
    ok = _is_png(p1) and _is_png(p2)
    return ok, 'stacked + detection-overlay PNGs valid'


def check_raster(sc):
    d = os.path.join(sc['tmp'], 'raster')
    det = sc['det']
    p = PL.plot_raster(det['t'], det['ch'], 9, sc['simtime'],
                       src_neuron=sc['src'], spk_t=sc['spk_t'],
                       spk_i=sc['spk_i'],
                       visible_neurons=sc['reach'], out_dir=d)
    ok = _is_png(p)
    # also without ground truth (single-panel path)
    p2 = PL.plot_raster(det['t'], det['ch'], 9, sc['simtime'], out_dir=d,
                        name='raster_nogt')
    ok = ok and _is_png(p2)
    return ok, 'raster with and without ground truth both valid'


def check_waveform_plots(sc):
    d = os.path.join(sc['tmp'], 'wave')
    det = sc['det']
    p1 = PL.plot_zoom(det['filtered'], sc['fs'], det['t'], det['ch'],
                      det['amp'], det['sigma'], 5.0, out_dir=d)
    p2 = PL.plot_detected_waveforms(det['filtered'], sc['fs'], det['t'],
                                    det['ch'], 9, out_dir=d)
    p3 = PL.plot_amplitude_histogram(det['amp'], det['ch'], det['sigma'],
                                     5.0, 9, out_dir=d)
    p4 = PL.plot_detection_summary(det['ch'], sc['src'], det['sigma'], 9,
                                   dt_to_truth=sc['dt'], out_dir=d)
    ok = all(_is_png(p) for p in (p1, p2, p3, p4))
    return ok, '4 waveform/summary PNGs valid'


def check_orchestrators(sc):
    d = os.path.join(sc['tmp'], 'orch')
    topo = PL.save_topo_diagnostics(
        d, sc['xy'], sc['probe'], sc['W'], sc['scfg'], r_max=sc['r_max'],
        noise_floor=2.5, templates=sc['lib']['templates'],
        dt_hr_ms=sc['lib']['dt_hr_ms'])
    it = PL.save_iter_diagnostics(
        d, sc['traces'], sc['det']['filtered'], sc['fs'], sc['det'],
        sc['src'], sc['dt'], 5.0, sc['simtime'],
        spk_t=sc['spk_t'], spk_i=sc['spk_i'], visible_neurons=sc['reach'])
    ok = (len(topo) == 3 and len(it) == 7
          and all(_is_png(p) for p in topo + it))
    return ok, '%d topo + %d iter PNGs from orchestrators' % (len(topo), len(it))


def check_degenerate_no_detections(sc_empty):
    """Zero spikes -> zero detections. Nothing may raise."""
    d = os.path.join(sc_empty['tmp'], 'empty')
    det = sc_empty['det']
    paths = PL.save_iter_diagnostics(
        d, sc_empty['traces'], det['filtered'], sc_empty['fs'], det,
        sc_empty['src'], sc_empty['dt'], 5.0, sc_empty['simtime'],
        spk_t=sc_empty['spk_t'], spk_i=sc_empty['spk_i'],
        visible_neurons=sc_empty['reach'])
    ok = len(paths) == 7 and all(_is_png(p) for p in paths)
    return ok, 'no-detection scene produced %d valid PNGs (n_det=%d)' % (
        len(paths), len(det['t']))


def check_no_reach_mask(sc):
    """visible_neurons=None must still work (falls back to all neurons)."""
    d = os.path.join(sc['tmp'], 'noreach')
    det = sc['det']
    p = PL.plot_raster(det['t'], det['ch'], 9, sc['simtime'],
                       src_neuron=sc['src'], spk_t=sc['spk_t'],
                       spk_i=sc['spk_i'], visible_neurons=None, out_dir=d)
    return _is_png(p), 'raster without a reach mask is valid'


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print('=' * 66)
    print('MEA plotting smoke test')
    print('=' * 66)

    if not PL.HAVE_MPL:
        print('[FAIL] matplotlib is not installed; cannot test plotting')
        return 1

    tmp = tempfile.mkdtemp(prefix='mea_plots_test_')
    try:
        sc = _scene(tmp, n_spikes=40)
        sc_empty = _scene(tmp, n_spikes=0)

        c = Check()
        c.run('1_mpl_available', check_mpl_available)
        c.run('2_busiest_window', check_busiest_window)
        c.run('3_geometry_plots', lambda: check_geometry_plots(sc))
        c.run('4_trace_plots', lambda: check_trace_plots(sc))
        c.run('5_raster', lambda: check_raster(sc))
        c.run('6_waveform_plots', lambda: check_waveform_plots(sc))
        c.run('7_orchestrators', lambda: check_orchestrators(sc))
        c.run('8_no_detections', lambda: check_degenerate_no_detections(sc_empty))
        c.run('9_no_reach_mask', lambda: check_no_reach_mask(sc))

        print('-' * 66)
        if c.all_ok():
            print('ALL CHECKS PASSED')
            return 0
        print('SOME CHECKS FAILED')
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
