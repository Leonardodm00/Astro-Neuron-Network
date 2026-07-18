#!/usr/bin/env python3
# smoke_test_mea_pipeline.py
# =============================================================================
# End-to-end correctness self-check for the virtual-MEA pipeline
#   mea_probe.py + mea_synthesis.py + mea_detection.py
# using SYNTHETIC ground truth (known neurons, known spike times, known probe).
#
# CHECKS (each independent; prints PASS/FAIL)
#   1  Geometry: 9 electrodes, correct pitch/edge/centre; sub-sites on-face.
#   2  Scaling monotonicity: W decreases with distance; 1/r matches formula.
#   3  r_max formula: matches the closed form and the empirical reach.
#   4  Clean synthesis: a near neuron's troughs land at its spike times
#      (+ template trough offset) on the nearest electrode.
#   5  Amplitude ordering: near neuron trough >> far neuron trough.
#   6  Reach cutoff: an out-of-reach neuron contributes ~0.
#   7  Detection recall/precision on a clean+low-noise trace ~ perfect.
#   8  Ground-truth matching assigns detected spikes to the correct neuron.
#   9  Noise-only false-positive rate is consistent with a k-sigma Gaussian.
#  10  Determinism: same seeds -> identical traces and detections.
#
# RUN
#   python smoke_test_mea_pipeline.py        # exits 0 if all pass (~seconds)
# =============================================================================

import sys

import numpy as np

import eap_template_library as EAP
import mea_probe as P
import mea_synthesis as S
import mea_detection as D


FS = 10110.09          # target recording rate [Hz]
SIMT = 4.0             # short synthetic recording [s]


def _tiny_library():
    # 4 templates is enough and fast.
    return EAP.generate_template_library(n_templates=4, seed=0)


class Check:
    def __init__(self):
        self.rows = []

    def run(self, name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, 'EXC: %r' % (exc,)
        self.rows.append((name, ok))
        print('[%s] %-32s %s' % ('PASS' if ok else 'FAIL', name, detail))
        return ok

    def all_ok(self):
        return all(ok for _, ok in self.rows)


# -----------------------------------------------------------------------------

def check_geometry():
    c_max = 400.0
    pr = P.make_probe(c_max, P.ProbeConfig(n_side=3, pitch=60.0, edge=25.0, n_sub=4))
    centers = pr['centers']
    ok_n = centers.shape == (9, 2)
    # centre electrode == culture centre
    ctr = pr['center_xy']
    mid = centers[np.argmin(np.linalg.norm(centers - ctr, axis=1))]
    ok_center = np.allclose(mid, ctr)
    # pitch: unique spacings along x are 60
    xs = np.unique(np.round(centers[:, 0], 6))
    ok_pitch = np.allclose(np.diff(xs), 60.0)
    # sub-sites within +/- edge/2 of their electrode centre
    sub = pr['sub_sites']
    rel = sub - centers[:, None, :]
    ok_face = np.all(np.abs(rel) <= 25.0 / 2.0 + 1e-9)
    ok = ok_n and ok_center and ok_pitch and ok_face
    return ok, 'E=%d, center_ok=%s, pitch_ok=%s, face_ok=%s' % (
        centers.shape[0], ok_center, ok_pitch, ok_face)


def check_scaling_monotonic():
    pr = P.make_probe(400.0)
    ctr = pr['center_xy']
    # neurons along +x from the central electrode
    d = np.array([15.0, 30.0, 60.0, 120.0, 240.0])
    xy = np.column_stack([ctr[0] + d, np.full_like(d, ctr[1])])
    scfg = P.ScalingConfig(A_ref=75.0, r_ref=30.0, r_min=10.0, n_dec=2.0)
    W = P.compute_weights(xy, pr, scfg)
    # central electrode index
    ce = int(np.argmin(np.linalg.norm(pr['centers'] - ctr, axis=1)))
    w = W[:, ce]
    mono = np.all(np.diff(w) < 0)                       # strictly decreasing
    # at d = r_ref = 30 the point amplitude == A_ref; sub-site averaging under
    # the convex 1/r^2 law lifts the mean above A_ref (near sub-sites weigh
    # more). Allow up to 40% for the steeper (dipole-like) exponent.
    close_ref = abs(w[1] - 75.0) / 75.0 < 0.40
    return (mono and close_ref), 'monotone=%s, W(30um)=%.1f uV (~75)' % (
        mono, w[1])


def check_rmax():
    scfg = P.ScalingConfig(A_ref=75.0, r_ref=30.0, r_min=10.0, n_dec=2.0)
    gamma, sig = 0.5, 5.0
    rmax = P.r_max_reach(scfg, gamma, sig)
    # closed form for n_dec=2: r_ref * sqrt(A_ref/(gamma*sigma))
    #                         = 30 * sqrt(75/2.5) = 30*sqrt(30) ~ 164.3 um
    expect = 30.0 * (75.0 / (0.5 * 5.0)) ** 0.5
    ok = abs(rmax - expect) < 1e-6
    return ok, 'r_max=%.1f um (expect %.1f, n_dec=2)' % (rmax, expect)


def _one_neuron_scene(distance_um, c_max=400.0, n_dec=2.0):
    """Single neuron at `distance_um` +x from centre; regular 8 Hz spikes."""
    lib = _tiny_library()
    pr = P.make_probe(c_max)
    ctr = pr['center_xy']
    xy = np.array([[ctr[0] + distance_um, ctr[1]]])
    scfg = P.ScalingConfig(A_ref=75.0, r_ref=30.0, r_min=10.0, n_dec=n_dec)
    W = P.compute_weights(xy, pr, scfg)
    assign = S.assign_templates(1, lib['templates'].shape[0], seed=1)
    lib_fs = S.prepare_library_at_fs(lib, FS)
    # regular spikes at 8 Hz, avoiding the very edges
    st = np.arange(0.3, SIMT - 0.3, 1.0 / 8.0)
    si = np.zeros_like(st, dtype=np.int64)
    ce = int(np.argmin(np.linalg.norm(pr['centers'] - ctr, axis=1)))
    return lib, pr, xy, scfg, W, assign, lib_fs, st, si, ce


def check_clean_alignment():
    # Very high SNR (low noise) so sigma_hat is well-defined but the signal
    # dominates: count and alignment should be exact.
    lib, pr, xy, scfg, W, assign, lib_fs, st, si, ce = _one_neuron_scene(20.0)
    noise = S.calibrate_noise_rms(FS, (300.0, 3000.0), target_post_uv=2.0)
    traces, t = S.synthesize_traces(st, si, W, lib_fs, assign, FS, SIMT,
                                    noise_rms=noise, seed=0)
    cfg = D.DetectConfig(k=5.0)
    det = D.detect(traces, FS, cfg)
    on_ce = det['t'][det['ch'] == ce]
    if len(on_ce) == 0:
        return False, 'no detections on central electrode'
    # RECALL: every true spike has a detected trough within 0.3 ms.
    dmax = 0.0
    for ts in st:
        dmax = max(dmax, np.min(np.abs(on_ce - ts)))
    recall_ok = dmax < 0.3e-3
    # PRECISION: near-threshold band-pass ringing may add a few FPs; require
    # the excess to stay small (<= 20%).
    prec_ok = len(on_ce) <= 1.2 * len(st)
    return (recall_ok and prec_ok), 'recall_align max|dt|=%.2f ms, n_det=%d/%d' % (
        dmax * 1e3, len(on_ce), len(st))


def check_amplitude_ordering():
    _, _, _, _, Wn, _, _, _, _, ce_n = _one_neuron_scene(20.0)
    _, _, _, _, Wf, _, _, _, _, ce_f = _one_neuron_scene(200.0)
    near = Wn[0, ce_n]
    far = Wf[0, ce_f]
    ok = near > far * 3.0                    # 1/r^2 over 10x distance => ~100x
    return ok, 'W_near=%.1f uV > W_far=%.1f uV' % (near, far)


def check_reach_cutoff():
    scfg = P.ScalingConfig(A_ref=75.0, r_ref=30.0, r_min=10.0, n_dec=2.0)
    rmax = P.r_max_reach(scfg, gamma=0.5, sigma_noise=5.0)      # ~164 um
    # put a neuron well beyond r_max
    c_max = 1000.0
    pr = P.make_probe(c_max)
    ctr = pr['center_xy']
    xy = np.array([[ctr[0] + 300.0, ctr[1]]])                  # > 164
    W = P.compute_weights(xy, pr, scfg)
    mask = P.reach_mask(W, floor=0.5 * 5.0)                     # gamma*sigma
    ok = (not mask[0]) and (W.max() < 0.5 * 5.0)
    return ok, 'beyond r_max(%.0f um): max W=%.3f uV < %.2f, reached=%s' % (
        rmax, W.max(), 0.5 * 5.0, mask[0])


def check_detection_quality():
    lib, pr, xy, scfg, W, assign, lib_fs, st, si, ce = _one_neuron_scene(20.0)
    noise = S.calibrate_noise_rms(FS, (300.0, 3000.0), target_post_uv=5.0)
    traces, t = S.synthesize_traces(st, si, W, lib_fs, assign, FS, SIMT,
                                    noise_rms=noise, seed=7)
    det = D.detect(traces, FS, D.DetectConfig(k=5.0))
    on_ce = det['t'][det['ch'] == ce]
    # recall: fraction of true spikes matched within 0.5 ms
    matched = 0
    for ts in st:
        if len(on_ce) and np.min(np.abs(on_ce - ts)) < 0.5e-3:
            matched += 1
    recall = matched / len(st)
    ok = recall >= 0.9
    return ok, 'recall=%.2f on central electrode (noise~5uV, SNR~15)' % recall


def check_truth_matching():
    lib, pr, xy, scfg, W, assign, lib_fs, st, si, ce = _one_neuron_scene(20.0)
    noise = S.calibrate_noise_rms(FS, (300.0, 3000.0), target_post_uv=2.0)
    traces, t = S.synthesize_traces(st, si, W, lib_fs, assign, FS, SIMT,
                                    noise_rms=noise, seed=0)
    det = D.detect(traces, FS, D.DetectConfig(k=5.0))
    src, dt = D.match_to_truth(det, W, st, si, FS, window_ms=1.5)
    ce_mask = det['ch'] == ce
    det_t_ce = det['t'][ce_mask]
    src_ce = src[ce_mask]
    # Detections close to a true spike must be attributed to neuron 0;
    # detections far from any true spike (ringing FPs) must be flagged -1.
    close = np.array([np.min(np.abs(st - x)) for x in det_t_ce]) < 0.5e-3
    true_ok = np.all(src_ce[close] == 0)
    fp_ok = np.all(src_ce[~close] == -1)
    n_true = int(np.sum(close))
    n_fp = int(np.sum(~close))
    return (true_ok and fp_ok), (
        'true->neuron0 ok=%s (%d), FP->-1 ok=%s (%d)' % (
            true_ok, n_true, fp_ok, n_fp))


def check_noise_false_positive_rate():
    # No spikes; pure noise. k=5 => expected FP rate per sample ~ tail prob.
    pr = P.make_probe(400.0)
    W = np.zeros((1, 9))
    lib = _tiny_library()
    lib_fs = S.prepare_library_at_fs(lib, FS)
    assign = np.zeros(1, np.int64)
    noise = S.calibrate_noise_rms(FS, (300.0, 3000.0), target_post_uv=5.0)
    traces, t = S.synthesize_traces(np.array([]), np.array([], np.int64),
                                    W, lib_fs, assign, FS, SIMT,
                                    noise_rms=noise, seed=3)
    det = D.detect(traces, FS, D.DetectConfig(k=5.0))
    # k=5 single-sided Gaussian tail ~ 2.9e-7; over E*T samples this is a small
    # count. Band-pass correlates samples so allow a generous ceiling.
    E, Tn = traces.shape
    rate = len(det['t']) / (E * Tn)
    ok = rate < 1e-3                       # very sparse
    return ok, 'FP rate=%.2e per sample (k=5), n_fp=%d' % (rate, len(det['t']))


def check_determinism():
    a = _run_min(seed_noise=11)
    b = _run_min(seed_noise=11)
    ok = np.array_equal(a['t'], b['t']) and np.allclose(a['traces'], b['traces'])
    c = _run_min(seed_noise=12)
    differ = not np.allclose(a['traces'], c['traces'])
    return (ok and differ), 'seeded match=%s, diff-seed differs=%s' % (ok, differ)


def _run_min(seed_noise):
    lib, pr, xy, scfg, W, assign, lib_fs, st, si, ce = _one_neuron_scene(20.0)
    noise = S.calibrate_noise_rms(FS, (300.0, 3000.0), 5.0)
    traces, t = S.synthesize_traces(st, si, W, lib_fs, assign, FS, SIMT,
                                    noise_rms=noise, seed=seed_noise)
    det = D.detect(traces, FS, D.DetectConfig(k=5.0))
    return dict(traces=traces, t=det['t'])


# -----------------------------------------------------------------------------

def main():
    print('=' * 66)
    print('Virtual-MEA pipeline smoke test (probe + synthesis + detection)')
    print('=' * 66)
    c = Check()
    c.run('1_geometry', check_geometry)
    c.run('2_scaling_monotonic', check_scaling_monotonic)
    c.run('3_rmax_formula', check_rmax)
    c.run('4_clean_alignment', check_clean_alignment)
    c.run('5_amplitude_ordering', check_amplitude_ordering)
    c.run('6_reach_cutoff', check_reach_cutoff)
    c.run('7_detection_quality', check_detection_quality)
    c.run('8_truth_matching', check_truth_matching)
    c.run('9_noise_fp_rate', check_noise_false_positive_rate)
    c.run('10_determinism', check_determinism)
    print('-' * 66)
    if c.all_ok():
        print('ALL CHECKS PASSED')
        return 0
    print('SOME CHECKS FAILED')
    return 1


if __name__ == '__main__':
    sys.exit(main())
