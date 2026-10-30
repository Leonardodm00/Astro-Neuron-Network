#!/usr/bin/env python3
"""
check_nan_last_spike_timing.py -- standalone check (not pipeline code).

Claim tested
    Most neurons that end a run with V NaN (refractory runaway) fired their
    last spike at the very start of the run. Reported per run set: the
    fraction of NaN neurons whose last spike time is <= 10 ms, <= 100 ms and
    <= 1 s after the start, and the number of NaN neurons that never spiked
    (expected 0: a neuron can only diverge during t_ref, i.e. after a spike).

Truth chosen (fixed before running)
    The runs of check_refractory_blowup.py: Brian2's post-run state check
    recorded the neurons that ended each run with V NaN or |V| > 1e50
    (result_<label>.json, field nan_idx). Spike times spk_N_t [s] and indices
    spk_N_i from out_<label>/topo_XXX/iter_XXXXX.npz of the same run.
    Last spike per neuron with np.maximum.at (well defined with repeated
    indices, unlike fancy assignment).

Usage
    python3 check_nan_last_spike_timing.py WORK/result_a.json WORK/result_b.json ...
"""
import json
import os
import sys

import numpy as np

CUTS_S = (0.010, 0.100, 1.0)   # s


def last_spike_times(spk_t, spk_i, n_neurons):
    """Last spike time [s] per neuron; -inf for a neuron that never spiked."""
    last = np.full(n_neurons, -np.inf)
    if spk_t.size:
        np.maximum.at(last, spk_i.astype(np.int64), spk_t.astype(np.float64))
    return last


def main(paths):
    for path in paths:
        d = json.load(open(path))
        work = os.path.dirname(os.path.abspath(path))
        n_neurons = int(d['Nn'])
        lasts, n_runs = [], 0
        for rec in d['records']:
            if rec.get('status') != 'ok':
                continue
            n_runs += 1
            z = np.load(os.path.join(work, 'out_%s' % d['label'], 'topo_%03d' % rec['topo'],
                                     'iter_%05d.npz' % rec['iter']))
            last = last_spike_times(z['spk_N_t'], z['spk_N_i'], n_neurons)
            idx = np.asarray(list(rec['nan_idx']), dtype=np.int64)
            lasts.append(last[idx])
        x = np.concatenate(lasts) if lasts else np.zeros(0)
        never = int(np.isneginf(x).sum())
        n = x.size
        parts = ['%s: %d runs, clamp=%s, T = %g s, NaN neurons %d, never spiked %d'
                 % (d['label'], n_runs, d['clamp'], float(d['simtime_s']), n, never)]
        for c in CUTS_S:
            k = int(((x <= c) & ~np.isneginf(x)).sum())
            parts.append('last spike <= %g ms: %d (%.1f %%)' % (c * 1e3, k, 100.0 * k / n if n else float('nan')))
        print(' | '.join(parts))


if __name__ == '__main__':
    main(sys.argv[1:])
