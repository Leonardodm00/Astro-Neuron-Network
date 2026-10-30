#!/usr/bin/env python3
"""
calibrate_refractory_scan.py -- standalone check (not pipeline code).

Claim tested
    The spike-train rules of refractory_scan.py (classify_run) flag the
    neurons that really went NaN and do not flag healthy neurons.

Truth chosen (fixed before running)
    Runs from check_refractory_blowup.py, where Brian2's post-run state check
    recorded which neurons ended each run with V NaN or |V| > 1e50
    (result_<label>.json, field nan_idx). Positives: those neurons. Negatives:
    every other neuron, including every neuron of the clamped runs.

Reported
    per rule (early, early_strict, late, early|late): per-neuron sensitivity
    TP/(TP+FN) and false-positive rate FP/(FP+TN); per run: runs with a NaN
    neuron that are flagged (>= 1 neuron flagged) or missed, and flagged runs
    without any NaN neuron.

Usage
    python3 calibrate_refractory_scan.py --scan_dir DIR_WITH_refractory_scan.py \
        --results WORK/result_a.json WORK/result_b.json ...
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan_dir', required=True)
    ap.add_argument('--results', nargs='+', required=True)
    ap.add_argument('--t-early', type=float, default=1.0, dest='t_early')
    a = ap.parse_args()
    sys.path.insert(0, a.scan_dir)
    from refractory_scan import classify_run, DEFAULT_CFG
    cfg = dict(DEFAULT_CFG, t_early=a.t_early)
    print('rule settings: %s' % json.dumps(cfg, sort_keys=True))
    rules = ('early', 'early_strict', 'late', 'union')
    for path in a.results:
        d = json.load(open(path))
        work = os.path.dirname(os.path.abspath(path))
        T, Nn = float(d['simtime_s']), int(d['Nn'])
        conf = {r: [0, 0, 0, 0] for r in rules}          # TP FN FP TN
        runs = {r: [0, 0, 0, 0] for r in rules}          # flagged&nan, missed&nan, flagged&clean, clean
        n_runs = 0
        for rec in d['records']:
            if rec.get('status') != 'ok':
                continue
            n_runs += 1
            z = np.load(os.path.join(work, 'out_%s' % d['label'], 'topo_%03d' % rec['topo'],
                                     'iter_%05d.npz' % rec['iter']))
            f = classify_run(z['spk_N_t'], z['spk_N_i'], Nn, T, cfg)
            truth = np.zeros(Nn, dtype=bool)
            truth[list(rec['nan_idx'])] = True
            flags = dict(early=f['early'], early_strict=f['early_strict'], late=f['late'],
                         union=f['early'] | f['late'])
            for r in rules:
                fl = flags[r]
                c = conf[r]
                c[0] += int((fl & truth).sum()); c[1] += int((~fl & truth).sum())
                c[2] += int((fl & ~truth).sum()); c[3] += int((~fl & ~truth).sum())
                rf, rt = bool(fl.any()), bool(truth.any())
                u = runs[r]
                u[0] += rf and rt; u[1] += (not rf) and rt; u[2] += rf and not rt; u[3] += (not rf) and not rt
        print('\n== %s  (%s, T = %g s, Nn = %d, %d runs, clamp=%s)' % (
            d['label'], d.get('mode', 'Neuronal'), T, Nn, n_runs, d['clamp']))
        for r in rules:
            TP, FN, FP, TN = conf[r]
            a1, a2, a3, a4 = runs[r]
            print('  %-13s neurons: sensitivity %s  false-positive rate %.4f (%d / %d) | runs: NaN-run flagged %d, '
                  'missed %d; clean run flagged %d of %d' % (
                      r, ('%.3f (%d / %d)' % (TP / float(TP + FN), TP, TP + FN)) if TP + FN else '  n/a       ',
                      FP / float(max(FP + TN, 1)), FP, FP + TN, a1, a2, a3, a3 + a4))


if __name__ == '__main__':
    main()
