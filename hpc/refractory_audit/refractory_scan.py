#!/usr/bin/env python3
"""
refractory_scan.py -- screen saved sweep simulations for the refractory runaway.

What it looks for
    In the sweep's neuron model (ASD_fun_BD_cpp.py, Neuronal_Network) V keeps
    integrating during the refractory period while Brian2 ignores the
    threshold there. A neuron pushed a few DeltaT above VT inside that window
    diverges, its V becomes NaN, and it never spikes again. V is not saved, so
    this screen reads the spike trains: a NaN neuron's last spike is followed
    by silence until the end of the run. A neuron can only die right after
    one of its own spikes, so neurons that never spiked are never flagged.

    Rules, per neuron that spiked at least once (T = simulated time):
      early         its last spike is at or before --t-early (default 1.0 s)
                    and nothing follows. Also counted with --t-early-strict
                    (default 0.1 s). Evaluated only when T >= 5 * t_early.
      late          its last spike is after --t-early, then at least
                    tau = max(--late-min-s, --late-frac * T) s of silence to
                    the end, while at least --late-active-frac of the other
                    neurons still spike more than --late-gap-s after it: a
                    neuron that falls silent inside a network that keeps
                    firing.
    A run whose spiking neurons ALL fall silent early is reported apart
    ("network silent early"): numerically it is the mass-death pattern, but
    a spike train alone cannot exclude a network silenced by its own
    adaptation; check those runs' parameters before discarding them.

Calibration (2026-10-29/30, sandbox, runs whose NaN neurons were recorded by
Brian2's own post-run check; calibrate_refractory_scan.py): rule "early"
(1.0 s) flagged no healthy neuron (0 of 3,816 in clamped runs, 0 of 2,715
non-NaN neurons in unclamped runs) and caught 91-96 % of the NaN neurons and
58 of 60 affected runs (5 s runs, 159 or 98+98 neurons). The headline
numbers use "early" only; "late" is reported apart, see the summary.

Outputs, per campaign, in --out:
    <campaign>_runs.csv      one row per scanned run (counts, flags, parameters)
    <campaign>_summary.txt   the incidence summary, also printed
    <campaign>_summary.json  the headline numbers (read by check_refractory_nan.sh)

Usage (campaign environment active; needs numpy, not brian2):
    python refractory_scan.py CAMPAIGN_DIR [CAMPAIGN_DIR ...] --out DIR
           [--sample 1000] [--seed 1] [--workers 4]
    --sample 0 scans every run (use submit_refractory_scan.sh for that).
Exit status: 0 when every campaign was scanned, 1 when one could not be read.
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

DEFAULT_CFG = dict(t_early=1.0, t_early_strict=0.1, late_frac=0.5, late_min_s=20.0,
                   late_active_frac=0.10, late_gap_s=1.0)


# ---------------------------------------------------------------------------
# the rules (one function, used by the scan, the smoke test and the calibration)
# ---------------------------------------------------------------------------
def classify_run(spk_t, spk_i, Nn, T, cfg=None):
    """Per-neuron flags for one run.

    spk_t : spike times [s], any float dtype; spk_i : neuron index per spike;
    Nn    : number of neurons; T : simulated time [s].
    Returns a dict of length-Nn arrays (count, last, spiked, early,
    early_strict, late) and the scalars tau_late, early_evaluated.
    """
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    t = np.asarray(spk_t, dtype=np.float64).ravel()
    i = np.asarray(spk_i, dtype=np.int64).ravel()
    if t.shape != i.shape:
        raise ValueError('spike times and indices differ in length')
    keep = (i >= 0) & (i < Nn) & np.isfinite(t)
    t, i = t[keep], i[keep]
    count = np.bincount(i, minlength=Nn)[:Nn] if i.size else np.zeros(Nn, dtype=np.int64)
    last = np.full(Nn, -np.inf)
    if i.size:
        np.maximum.at(last, i, t)
    spiked = count > 0
    early_evaluated = T >= 5.0 * cfg['t_early']
    if early_evaluated:
        early = spiked & (last <= cfg['t_early'])
        early_strict = spiked & (last <= cfg['t_early_strict'])
    else:
        early = np.zeros(Nn, dtype=bool)
        early_strict = np.zeros(Nn, dtype=bool)
    tau = max(cfg['late_min_s'], cfg['late_frac'] * T)
    late = np.zeros(Nn, dtype=bool)
    cand = spiked & (last > cfg['t_early']) & (last <= T - tau)
    if cand.any():
        lasts = np.sort(last[spiked])
        need = max(1, int(math.ceil(cfg['late_active_frac'] * (Nn - 1))))
        for j in np.flatnonzero(cand):
            # neurons whose last spike comes later than last_j + gap have spiked
            # inside the silent window of j (j itself is not counted)
            n_after = lasts.size - np.searchsorted(lasts, last[j] + cfg['late_gap_s'], side='right')
            late[j] = n_after >= need
    return dict(count=count, last=last, spiked=spiked, early=early,
                early_strict=early_strict, late=late, tau_late=tau,
                early_evaluated=early_evaluated)


# ---------------------------------------------------------------------------
# campaign discovery
# ---------------------------------------------------------------------------
def read_task_meta(task_dir):
    """simtime, Nn, lib_dir and parameter names of one sweep_* task folder."""
    meta = dict(task=os.path.basename(task_dir.rstrip('/')), T=None, Nn=None,
                lib_dir=None, param_names=None, error=None)
    try:
        with open(os.path.join(task_dir, 'job_args.json')) as fh:
            ja = json.load(fh)
        meta['T'] = float(ja['simtime'])
        if ja.get('Nn') is not None:
            meta['Nn'] = int(ja['Nn'])
        meta['lib_dir'] = ja.get('lib_dir')
    except Exception as e:
        meta['error'] = 'job_args.json: %s: %s' % (type(e).__name__, e)
        return meta
    for js in sorted(glob.glob(os.path.join(task_dir, 'topo_*', 'iter_*.json')))[:5]:
        try:
            with open(js) as fh:
                names = json.load(fh).get('param_names')
            if names:
                meta['param_names'] = list(names)
                break
        except Exception:
            continue
    return meta


def list_runs(campaign_dir):
    """[(task_dir, npz_path)] for every iter_*.npz of the campaign, and task metas."""
    tasks = sorted(d for d in glob.glob(os.path.join(campaign_dir, 'sweep_*')) if os.path.isdir(d))
    runs, metas = [], {}
    for td in tasks:
        metas[td] = read_task_meta(td)
        for topo in sorted(glob.glob(os.path.join(td, 'topo_*'))):
            try:
                names = os.listdir(topo)
            except OSError:
                continue
            for n in sorted(names):
                if n.startswith('iter_') and n.endswith('.npz'):
                    runs.append((td, os.path.join(topo, n)))
    return runs, metas


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
_CFG = dict(DEFAULT_CFG)


def _init_worker(cfg):
    _CFG.clear()
    _CFG.update(cfg)


def scan_one(job):
    task_dir, npz_path, T, Nn = job
    row = dict(task=os.path.basename(task_dir), topo=os.path.basename(os.path.dirname(npz_path)),
               iter=os.path.basename(npz_path)[:-4], status='ok')
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            spk_t = np.asarray(z['spk_N_t'])
            spk_i = np.asarray(z['spk_N_i'])
            params = np.asarray(z['params'], dtype=np.float64) if 'params' in z.files else None
            conn_prob = float(z['conn_prob']) if 'conn_prob' in z.files else float('nan')
    except Exception as e:
        row['status'] = 'unreadable: %s' % type(e).__name__
        return row, None
    nn_source = 'job_args'
    if Nn is None:
        Nn = int(spk_i.max()) + 1 if spk_i.size else 0
        nn_source = 'max_index+1'
    if Nn <= 0:
        row.update(T=T, Nn=0, nn_source=nn_source, n_spikes=0, n_spiking=0, n_early=0,
                   n_early_strict=0, n_late=0, n_suspect=0, frac_suspect=0.0,
                   net_silent_early=False, t_last_any=float('nan'), conn_prob=conn_prob)
        return row, params
    f = classify_run(spk_t, spk_i, Nn, T, _CFG)
    n_spiking = int(f['spiked'].sum())
    n_early, n_late = int(f['early'].sum()), int(f['late'].sum())
    finite_last = f['last'][np.isfinite(f['last'])]
    row.update(T=T, Nn=Nn, nn_source=nn_source, n_spikes=int(f['count'].sum()),
               n_spiking=n_spiking, n_early=n_early, n_early_strict=int(f['early_strict'].sum()),
               n_late=n_late, n_suspect=n_early + n_late,
               frac_suspect=(n_early + n_late) / float(Nn),
               net_silent_early=bool(n_spiking > 0 and n_early == n_spiking),
               t_last_any=float(finite_last.max()) if finite_last.size else float('nan'),
               conn_prob=conn_prob)
    if not f['early_evaluated']:
        row['status'] = 'ok (T < 5 t_early: early rule not evaluated)'
    return row, params


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan')
    p = k / float(n)
    d = 1.0 + z * z / n
    c = p + z * z / (2.0 * n)
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (c - h) / d, (c + h) / d


def pct(k, n):
    return 100.0 * k / n if n else float('nan')


def summarize(name, campaign_dir, rows, n_found, sampled, cfg, seed, tasks_meta):
    ok = [r for r in rows if r['status'].startswith('ok')]
    bad = [r for r in rows if not r['status'].startswith('ok')]
    n = len(ok)
    L = []
    L.append('== %s  (%s)' % (name, campaign_dir))
    Ts = sorted({r['T'] for r in ok})
    Ns = sorted({r['Nn'] for r in ok})
    skipped = [m['task'] for m in tasks_meta.values() if m['error']]
    L.append('runs found %d | scanned %d (%s) | unreadable %d | tasks without job_args %d | T = %s s | Nn = %s'
             % (n_found, len(rows), ('random sample, seed %d' % seed) if sampled else 'all',
                len(bad), len(skipped), ','.join('%g' % t for t in Ts) or '-',
                ','.join(str(x) for x in Ns) or '-'))
    if not n:
        L.append('no readable run')
        return '\n'.join(L)

    def line(label, k):
        lo, hi = wilson(k, n)
        L.append('  %-52s %7d / %d = %5.1f %%  [95%% CI %.1f, %.1f]'
                 % (label, k, n, pct(k, n), 100 * lo, 100 * hi))

    line('runs with >= 1 neuron silent after its last spike <= %g s' % cfg['t_early'],
         sum(r['n_early'] > 0 for r in ok))
    line('  same, last spike <= %g s' % cfg['t_early_strict'], sum(r['n_early_strict'] > 0 for r in ok))
    line('  runs with >= 50 % of neurons so silenced', sum(r['n_early'] >= 0.5 * r['Nn'] for r in ok))
    line('  runs whose spiking neurons ALL are so silenced', sum(r['net_silent_early'] for r in ok))
    line('runs with a late silence in an active network (*)', sum(r['n_late'] > 0 for r in ok))
    line('runs with either', sum(r['n_suspect'] > 0 for r in ok))
    line('runs without any spike', sum(r['n_spikes'] == 0 for r in ok))
    tot_nn = sum(r['Nn'] for r in ok)
    L.append('  %-52s %5.1f %% of %d neurons (early rule)' % (
        'neurons silenced, pooled', pct(sum(r['n_early'] for r in ok), tot_nn), tot_nn))
    L.append('  (*) late rule: silent >= max(%g s, %g T) while >= %g of the other neurons still fire;'
             % (cfg['late_min_s'], cfg['late_frac'], cfg['late_active_frac']))
    L.append('      a heuristic for deaths after the first second, not calibrated on runs of this length')
    if any(r['nn_source'] != 'job_args' for r in ok):
        L.append('  NOTE: Nn missing from job_args.json for some runs; taken as max index + 1 (a lower bound)')
    if any('not evaluated' in r['status'] for r in ok):
        L.append('  NOTE: some runs are shorter than 5 x t_early; the early rule was not evaluated there')
    # breakdown by parameter terciles (only axes that vary in the scanned runs)
    cols = sorted({k for r in ok for k in r if k.startswith('par_')}) + ['conn_prob']
    spreads = []
    for k in cols:
        x = np.array([r.get(k, np.nan) for r in ok], dtype=float)
        y = np.array([r['n_early'] > 0 for r in ok], dtype=float)
        m = np.isfinite(x)
        if m.sum() < 30 or np.nanstd(x[m]) <= 0:
            continue
        q1, q2 = np.quantile(x[m], [1 / 3.0, 2 / 3.0])
        if not q1 < q2:
            continue
        g = [y[m][x[m] <= q1], y[m][(x[m] > q1) & (x[m] <= q2)], y[m][x[m] > q2]]
        if min(len(v) for v in g) == 0:
            continue
        inc = [100.0 * v.mean() for v in g]
        spreads.append((max(inc) - min(inc), k, inc, q1, q2))
    if spreads:
        L.append('  runs with >= 1 early-silenced neuron, by tercile of each varying axis (largest spread first):')
        for s, k, inc, q1, q2 in sorted(spreads, reverse=True)[:10]:
            L.append('    %-14s low %5.1f %%  mid %5.1f %%  high %5.1f %%   (tercile edges %.4g, %.4g)'
                     % (k.replace('par_', ''), inc[0], inc[1], inc[2], q1, q2))
    if bad:
        kinds = {}
        for r in bad:
            kinds[r['status']] = kinds.get(r['status'], 0) + 1
        L.append('  unreadable runs by error: %s' % ', '.join('%s x%d' % kv for kv in sorted(kinds.items())))
    return '\n'.join(L)


# ---------------------------------------------------------------------------
def scan_campaign(campaign_dir, out_dir, sample, seed, workers, cfg):
    campaign_dir = os.path.abspath(campaign_dir.rstrip('/'))
    name = os.path.basename(campaign_dir)
    t0 = time.time()
    runs, metas = list_runs(campaign_dir)
    n_found = len(runs)
    usable = [(td, p) for td, p in runs if metas[td]['error'] is None]
    sampled = 0 < sample < len(usable)
    if sampled:
        usable = random.Random(seed).sample(usable, sample)
    jobs = [(td, p, metas[td]['T'], metas[td]['Nn']) for td, p in usable]
    print('[scan] %s: %d runs found, %d to scan (%d tasks; listing took %.0f s)'
          % (name, n_found, len(jobs), len(metas), time.time() - t0), flush=True)
    results = []
    if jobs:
        if workers > 1:
            with Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                for k, res in enumerate(pool.imap_unordered(scan_one, jobs, chunksize=16), 1):
                    results.append(res)
                    if k % 5000 == 0:
                        print('[scan]   %d / %d' % (k, len(jobs)), flush=True)
        else:
            _init_worker(cfg)
            results = [scan_one(j) for j in jobs]
    # parameter columns, named from the sidecar's param_names
    rows = []
    names_any = next((m['param_names'] for m in metas.values() if m['param_names']), None)
    for row, params in results:
        if params is not None:
            names = names_any if names_any and len(names_any) == params.size else \
                ['p%02d' % k for k in range(params.size)]
            for nm, v in zip(names, params):
                row['par_' + nm] = float(v)
        rows.append(row)
    rows.sort(key=lambda r: (r['task'], r['topo'], r['iter']))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, '%s_runs.csv' % name)
    fields = ['task', 'topo', 'iter', 'status', 'T', 'Nn', 'nn_source', 'n_spikes', 'n_spiking',
              'n_early', 'n_early_strict', 'n_late', 'n_suspect', 'frac_suspect',
              'net_silent_early', 't_last_any', 'conn_prob']
    extra = sorted({k for r in rows for k in r if k.startswith('par_')})
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields + extra, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    text = summarize(name, campaign_dir, rows, n_found, sampled, cfg, seed, metas)
    text += '\n  rule settings: %s' % json.dumps(cfg, sort_keys=True)
    text += '\n  per-run table: %s  (%.0f s)' % (csv_path, time.time() - t0)
    with open(os.path.join(out_dir, '%s_summary.txt' % name), 'w') as fh:
        fh.write(text + '\n')
    ok = [r for r in rows if r['status'].startswith('ok')]
    k_early = sum(r['n_early'] > 0 for r in ok)
    lo, hi = wilson(k_early, len(ok))
    with open(os.path.join(out_dir, '%s_summary.json' % name), 'w') as fh:
        json.dump(dict(campaign=name, campaign_dir=campaign_dir, n_found=n_found, n_scanned=len(rows),
                       n_ok=len(ok), sampled=bool(sampled), seed=seed, cfg=cfg,
                       runs_early=k_early,
                       pct_runs_early=pct(k_early, len(ok)),
                       ci95_early=[100 * lo, 100 * hi],
                       runs_any_suspect=sum(r['n_suspect'] > 0 for r in ok),
                       runs_late=sum(r['n_late'] > 0 for r in ok),
                       pct_runs_net_silent_early=pct(sum(r['net_silent_early'] for r in ok), len(ok)),
                       pct_neurons_early=pct(sum(r['n_early'] for r in ok),
                                             sum(r['Nn'] for r in ok)),
                       pct_neurons_suspect=pct(sum(r['n_suspect'] for r in ok),
                                               sum(r['Nn'] for r in ok))),
                  fh, indent=1)
    print(text, flush=True)
    return n_found > 0


def main(argv=None):
    ap = argparse.ArgumentParser(description='Screen sweep campaigns for neurons silenced by the refractory runaway.')
    ap.add_argument('campaigns', nargs='+', help='campaign_<TAG> folders (holding sweep_* task folders)')
    ap.add_argument('--out', required=True, help='output folder for the CSV and summary files')
    ap.add_argument('--sample', type=int, default=1000, help='random runs per campaign; 0 = all (default 1000)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--workers', type=int, default=4)
    for k, v in DEFAULT_CFG.items():
        ap.add_argument('--' + k.replace('_', '-'), type=float, default=v, dest=k)
    a = ap.parse_args(argv)
    cfg = {k: getattr(a, k) for k in DEFAULT_CFG}
    ok = True
    for c in a.campaigns:
        if not os.path.isdir(c):
            print('[scan] SKIPPED %s: not a folder' % c, flush=True)
            ok = False
            continue
        ok = scan_campaign(c, a.out, a.sample, a.seed, a.workers, cfg) and ok
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
