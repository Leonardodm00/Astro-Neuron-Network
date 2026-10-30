#!/usr/bin/env python3
"""
smoke_test_refractory_scan.py -- checks refractory_scan.py and
check_refractory_nan.sh on synthetic campaigns whose answers are known.

Run next to the two scripts, with numpy available (no brian2, no simulation):
    python smoke_test_refractory_scan.py          # ~10 s
Expected last line:  RESULT: <n> passed, 0 failed
Exit status 0 iff every check passes.

Checks
  T1  classify_run on hand-made spike trains: early / strict / late / never
      spiking / short run / whole network stopping / out-of-range indices
  T2  scan_campaign on a synthetic campaign: per-run counts in the CSV, the
      summary JSON, an empty npz counted unreadable, a task without
      job_args.json skipped, sampling
  T3  check_refractory_nan.sh end to end with a fake HOME: CODE section
      (clamped / not clamped / missing), LOGS table, DATA, VERDICT; an
      unknown option refused
  T4  every file of the package is pure ASCII with LF line endings
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refractory_scan as R  # noqa: E402

_n = {'pass': 0, 'fail': 0}


def check(cond, msg):
    _n['pass' if cond else 'fail'] += 1
    print('  %s %s' % ('ok  ' if cond else 'FAIL', msg))


def train(times, idx):
    times = np.asarray(times, dtype=np.float32)
    return times, np.full(times.size, idx, dtype=np.int32)


def cat(*pairs):
    t = np.concatenate([p[0] for p in pairs]) if pairs else np.zeros(0, np.float32)
    i = np.concatenate([p[1] for p in pairs]) if pairs else np.zeros(0, np.int32)
    o = np.argsort(t, kind='stable')
    return t[o], i[o]


def t1():
    print('\n[T1] classify_run')
    T = 200.0
    t, i = cat(train(np.arange(0.05, 199.95, 0.1), 0),      # fires throughout
               train([0.005], 1),                             # dies at the first discharge
               train([0.02, 0.5], 2),                         # last spike 0.5 s
               train(np.arange(1.5, 50.0, 0.5), 3),           # stops at ~50 s, network active after
               train(np.arange(1.5, 150.0, 0.5), 5))          # stops at ~150 s: silence 50 s < 100 s
    f = R.classify_run(t, i, 6, T)                            # neuron 4 never spikes
    check(list(np.flatnonzero(f['early'])) == [1, 2], 'early (1.0 s) flags neurons 1 and 2')
    check(list(np.flatnonzero(f['early_strict'])) == [1], 'early strict (0.1 s) flags neuron 1 only')
    check(list(np.flatnonzero(f['late'])) == [3], 'late flags neuron 3 (silent 150 s, others firing)')
    check(not f['early'][4] and not f['late'][4] and f['count'][4] == 0, 'never-spiking neuron 4 not flagged')
    check(abs(f['tau_late'] - 100.0) < 1e-12, 'tau_late = max(20, 0.5 T) = 100 s')
    # whole network stops at the same time: nobody is "late" (no active others)
    t2, i2 = cat(*[train(np.arange(0.5, 50.0, 0.5) + 0.001 * k, k) for k in range(5)])
    f2 = R.classify_run(t2, i2, 5, T)
    check(not f2['late'].any() and not f2['early'].any(), 'network stopping together at 50 s: no neuron flagged')
    # whole network dies at the first discharge
    t3, i3 = cat(*[train([0.003 + 0.0005 * k], k) for k in range(5)])
    f3 = R.classify_run(t3, i3, 5, T)
    check(f3['early'].all() and f3['early_strict'].all(), 'network silent after 5 ms: all early')
    # short run: early rule not evaluated
    f4 = R.classify_run(np.array([0.01]), np.array([0]), 2, 3.0)
    check(not f4['early_evaluated'] and not f4['early'].any(), 'T = 3 s < 5 t_early: early not evaluated')
    # junk entries ignored, empty input
    f5 = R.classify_run(np.array([0.01, np.nan, 0.5]), np.array([0, 1, 7]), 2, T)
    check(f5['count'].tolist() == [1, 0], 'NaN times and out-of-range indices ignored')
    f6 = R.classify_run(np.zeros(0), np.zeros(0, int), 3, T)
    check(f6['count'].sum() == 0 and not f6['early'].any(), 'empty spike arrays handled')
    try:
        R.classify_run(np.zeros(2), np.zeros(3, int), 3, T)
        check(False, 'length mismatch refused')
    except ValueError:
        check(True, 'length mismatch refused')


PARAM_NAMES = ['Sigma', 'g_ampa', 'U_0_sr']


def write_run(topo_dir, it, t, i, params, conn_prob):
    np.savez_compressed(os.path.join(topo_dir, 'iter_%05d.npz' % it), spk_N_t=t, spk_N_i=i,
                        params=np.asarray(params, float), conn_prob=np.float64(conn_prob),
                        spk_A_t=np.zeros(0, np.float32), spk_A_i=np.zeros(0, np.int32))
    with open(os.path.join(topo_dir, 'iter_%05d.json' % it), 'w') as fh:
        json.dump({'param_names': PARAM_NAMES, 'params': list(params), 'conn_prob': conn_prob}, fh)


def make_campaign(root, name, lib_dir, Nn=10, T=200.0):
    camp = os.path.join(root, name)
    td = os.path.join(camp, 'sweep_intel_task0000')
    topo = os.path.join(td, 'topo_00000')
    os.makedirs(topo)
    with open(os.path.join(td, 'job_args.json'), 'w') as fh:
        json.dump({'simtime': T, 'Nn': Nn, 'lib_dir': lib_dir}, fh)
    alive = [train(np.arange(0.05 + 0.01 * k, T - 0.05, 0.25), k) for k in range(Nn)]
    # run 0: healthy
    write_run(topo, 0, *cat(*alive), [3.0, 0.1, 0.2], 0.2)
    # run 1: 3 neurons die at the first discharge, the rest keep firing
    write_run(topo, 1, *cat(*([train([0.004], k) for k in range(3)] + alive[3:])), [9.0, 1.0, 0.5], 0.4)
    # run 2: 2 neurons stop at 40 s inside an active network
    write_run(topo, 2, *cat(*([train(np.arange(0.3, 40.0, 0.5), k) for k in range(2)] + alive[2:])),
              [6.0, 0.5, 0.3], 0.3)
    # run 3: the whole network silent after 10 ms
    write_run(topo, 3, *cat(*[train([0.002 + 0.0007 * k], k) for k in range(Nn)]), [14.0, 4.0, 0.9], 0.6)
    # run 4: no spike at all
    write_run(topo, 4, np.zeros(0, np.float32), np.zeros(0, np.int32), [3.5, 0.02, 0.1], 0.15)
    # run 5: empty file (a node that lost its last writes)
    open(os.path.join(topo, 'iter_00005.npz'), 'wb').close()
    # a second task without job_args.json: skipped
    td2 = os.path.join(camp, 'sweep_cpu_task0001', 'topo_00000')
    os.makedirs(td2)
    write_run(td2, 0, *cat(*alive), [3.0, 0.1, 0.2], 0.2)
    return camp


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def t2(tmp):
    print('\n[T2] scan_campaign on a synthetic campaign')
    lib = os.path.join(tmp, 'libA')
    os.makedirs(lib)
    camp = make_campaign(tmp, 'campaign_fake', lib)
    out = os.path.join(tmp, 'out_t2')
    rc = R.main([camp, '--out', out, '--sample', '0', '--workers', '2'])
    check(rc == 0, 'main returns 0')
    rows = {r['iter']: r for r in read_csv(os.path.join(out, 'campaign_fake_runs.csv'))}
    check(len(rows) == 6, '6 runs of the task with job_args.json scanned (got %d)' % len(rows))
    g = lambda it, k: rows['iter_%05d' % it][k]
    check(g(0, 'n_suspect') == '0' and g(0, 'status') == 'ok', 'run 0 healthy: 0 suspects')
    check(g(1, 'n_early') == '3' and g(1, 'n_early_strict') == '3' and g(1, 'n_late') == '0',
          'run 1: 3 early, 3 strict, 0 late')
    check(g(2, 'n_late') == '2' and g(2, 'n_early') == '0', 'run 2: 2 late')
    check(g(3, 'net_silent_early') == 'True' and g(3, 'n_early') == '10', 'run 3: network silent early, 10 early')
    check(g(4, 'n_spikes') == '0' and g(4, 'n_suspect') == '0', 'run 4: no spikes, no suspects')
    check(g(5, 'status').startswith('unreadable'), 'run 5: empty npz counted unreadable')
    check(g(1, 'par_Sigma') == '9.0' and g(1, 'conn_prob') == '0.4', 'parameter columns named from param_names')
    s = json.load(open(os.path.join(out, 'campaign_fake_summary.json')))
    check(s['n_ok'] == 5 and s['runs_any_suspect'] == 3 and s['runs_early'] == 2 and s['runs_late'] == 1,
          'summary: 5 readable; 2 runs early, 1 late, 3 either')
    check(abs(s['pct_neurons_early'] - 100.0 * 13 / 50) < 1e-9
          and abs(s['pct_neurons_suspect'] - 100.0 * 15 / 50) < 1e-9,
          'summary: 13 of 50 neurons early-silenced, 15 of 50 either')
    txt = open(os.path.join(out, 'campaign_fake_summary.txt')).read()
    check('tasks without job_args 1' in txt and 'unreadable 1' in txt, 'summary text reports the skip and the unreadable run')
    out2 = os.path.join(tmp, 'out_t2s')
    R.main([camp, '--out', out2, '--sample', '3', '--seed', '7', '--workers', '1'])
    check(len(read_csv(os.path.join(out2, 'campaign_fake_runs.csv'))) == 3, '--sample 3 scans 3 runs')
    rc_missing = R.main([os.path.join(tmp, 'nope'), '--out', out2])
    check(rc_missing == 1, 'missing campaign folder: exit status 1')


ASD_LINE = ('    dV/dt  = noise + (gl*(El-V) + gl*DeltaT*exp((V-VT)/DeltaT) + gA*(EA-V)'
            ' + I_inj*bias_unit - I_syn)/Cm : volt%s\n')


def write_asd(lib, clamped):
    with open(os.path.join(lib, 'ASD_fun_BD_cpp.py'), 'w') as fh:
        fh.write("params = {'t_ref':   3*ms,  # refractory period\n}\n")
        fh.write("eqs = '''\n" + ASD_LINE % (' (unless refractory)' if clamped else '') + "'''\n")


def t3(tmp):
    print('\n[T3] check_refractory_nan.sh end to end (fake HOME)')
    libA, libB = os.path.join(tmp, 'libA'), os.path.join(tmp, 'libB')
    os.makedirs(libB, exist_ok=True)
    write_asd(libA, clamped=False)
    write_asd(libB, clamped=True)
    campB = make_campaign(os.path.join(tmp, 'B'), 'campaign_clamped', libB)
    home = os.path.join(tmp, 'home')
    os.makedirs(home)
    with open(os.path.join(home, 'meama_campaign.e1544754.0'), 'w') as fh:
        fh.write("WARNING    Neuron's variable 'V' has NaN, very large values, or encountered an error\n")
    with open(os.path.join(home, 'meama_campaign.e1544754.1'), 'w') as fh:
        fh.write('[env] fine\n')
    with open(os.path.join(home, 'asn_campaign.e99.3'), 'w') as fh:
        fh.write("WARNING    Astrocyte's variable 'C' has NaN, very large values\n")
    out = os.path.join(tmp, 'out_t3')
    env = dict(os.environ, HOME=home)
    cmd = ['bash', os.path.join(HERE, 'check_refractory_nan.sh'), '--sample', '0', '--workers', '1',
           '--out', out, os.path.join(tmp, 'campaign_fake'), campB, os.path.join(tmp, 'missing_campaign')]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log = open(os.path.join(out, 'audit.log')).read() if os.path.exists(os.path.join(out, 'audit.log')) else ''
    check(r.returncode == 0, 'driver exits 0 (got %d)' % r.returncode)
    check('campaign_fake: NOT CLAMPED (exposed)' in log, 'CODE: unclamped campaign reported exposed')
    check('campaign_clamped: CLAMPED' in log, 'CODE: clamped campaign reported clamped')
    check('missing_campaign: FOLDER NOT FOUND' in log, 'CODE: missing folder reported')
    lines = [l.split() for l in log.splitlines()]
    check(['meama_campaign', '2', '1', '0'] in lines, 'LOGS: meama_campaign 2 logs, 1 with V NaN, 0 astro')
    check(['asn_campaign', '1', '0', '1'] in lines, 'LOGS: asn_campaign 1 log, 0 V NaN, 1 astro')
    check('=== VERDICT ===' in log and '40.0 % of 5 scanned runs' in log, 'VERDICT: 40.0 % of 5 runs for campaign_fake')
    check(os.path.exists(os.path.join(out, 'campaign_clamped_runs.csv')), 'per-run CSV written for every campaign')
    check(not any(n.startswith('.') for n in os.listdir(out)), 'temporary files removed from the output folder')
    r2 = subprocess.run(['bash', os.path.join(HERE, 'check_refractory_nan.sh'), '--bogus'],
                        capture_output=True, text=True, env=env)
    check(r2.returncode == 2 and 'unknown option' in r2.stderr, 'unknown option refused (exit 2)')


def t4():
    print('\n[T4] bytes of the package files')
    for n in sorted(os.listdir(HERE)):
        p = os.path.join(HERE, n)
        if not os.path.isfile(p) or n.endswith('.pyc'):
            continue
        b = open(p, 'rb').read()
        check(b.count(b'\r') == 0 and all(x < 128 for x in b), '%s: 0 CR, pure ASCII' % n)


def main():
    tmp = tempfile.mkdtemp(prefix='refr_smoke_')
    try:
        t1()
        t2(tmp)
        t3(tmp)
        t4()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nRESULT: %d passed, %d failed' % (_n['pass'], _n['fail']))
    return 1 if _n['fail'] else 0


if __name__ == '__main__':
    sys.exit(main())
