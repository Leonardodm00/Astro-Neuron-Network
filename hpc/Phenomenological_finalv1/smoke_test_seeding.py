#!/usr/bin/env python3
"""
smoke_test_seeding.py -- smoke test of SPEC Block 11 (random seeds).

Every check goes through the production code: HPC_main_sweep._worker_entry
called in-process (as the sweep's pool workers call it), HPC_single_run.py
from its command line, and the helpers of HPC_single_run.py.

WHAT IT CHECKS
  S0  derive_network_seeds / validate_run_seed: deterministic, in range,
      distinct per purpose and per network, changed by seed_device; invalid
      input refused
  S1  fresh run seeds: one parameter vector, two run seeds, one binary ->
      different spike trains
  S2  reproducible: the same (vector, run seed) twice in one chunk -> identical
  S3  order independence: the same (vector, seed) pairs in the other order, in
      a new build -> identical pair by pair
  S4  frozen within a network: bias_unit (read from the binary) and the
      astrocytes' C after 0.2 ms identical for two run seeds
  S5  different networks: another topo_seed_base -> different bias_unit and C
  S6  replay from the record in a NEW process: topology arrays + topo_seed_base
      + the run's npz (params, seed_run) -> the same spike train, bit for bit
  S7  guard: with the reseed code removed from main(), the worker stops the
      chunk with seed_failure and writes no npz for the run
  S8  records: npz and json carry seed_run (= requested), seed_verified,
      noise_mode 'fresh', topo_seed_base, the four network seeds, rng_scheme;
      the network seeds equal derive_network_seeds(seed_device, topo_seed_base)
  S9  Neuronal mode: S1 and S2 without astrocytes
  S10 single run: --seed_run 11 / 12 differ, 11 / 11 identical, seed_run
      recorded as applied
  S11 refusals: a pack without topo_seed_base (KeyError), a run seed outside
      [0, 2**31 - 1) (ValueError), both before anything is built
  S12 helpers on a minimal network: a rand() pattern set before the reseed is
      the same for every run seed; seeds 5 / 5 identical, 5 / 6 different; the
      applied seed is read back; a run without its seed fails (exit 97)

HOW TO RUN (login node or sandbox; needs brian2 and a C++ compiler)
    cd <repo>/hpc/Phenomenological_finalv1
    python3 smoke_test_seeding.py            # all checks, about 10 min
    python3 smoke_test_seeding.py --quick    # S0, S11, S12 only, about 1 min
Expected last line:  RESULT: <n> passed, 0 failed
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import zlib

import numpy as np

import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import HPC_single_run as R          # noqa: E402
import HPC_main_sweep as S          # noqa: E402

NN, NA, CMAX, CONN = 24, 24, 120.0, 0.3
RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print('%-6s %-5s %s' % (name, 'PASS' if ok else 'FAIL', detail), flush=True)


# ----------------------------------------------------------------- fixtures
def topology(topo_seed_base, mode='Full'):
    import pandas as pd
    df = pd.read_csv(os.path.join(HERE, 'synapse_pdist.csv'))
    m = 2**31 - 1
    return R.build_topology(
        Nn=NN, Na=(NA if mode == 'Full' else 0), c_max=CMAX, conn_prob=CONN,
        syn_prob_csv=df, displ_bias=20.0, gj_dist=200.0, stoa_cutoff=70.0,
        stoa_sigma=200.0, seed_neuron=(topo_seed_base + 39) % m,
        seed_synapse=(topo_seed_base + 35) % m, seed_astro=(topo_seed_base + 60) % m,
        mode=mode, topology_mode='wallach', gj_max_dist=150.0, conn_rule='flat',
        stoa_gate_p=1.0, seed_stoa_gate=(topo_seed_base + 71) % m)


def vector(sigma=4.0):
    p = np.array(R.NOMINAL_PARAMS, dtype=np.float64)
    p[S.PARAM_NAMES.index('Sigma')] = sigma
    return p


def make_pack(topo, out_dir, runs, topo_seed_base, simtime, mode='Full', topo_idx=0):
    topo_dir = os.path.join(out_dir, 'topo_%05d' % topo_idx)
    os.makedirs(topo_dir, exist_ok=True)
    cli = dict(lib_dir=HERE, out_dir=out_dir, simtime=float(simtime), mode=mode,
               Nn=NN, Na=(NA if mode == 'Full' else 0), seed_device=50,
               seed_neuron=39, seed_synapse=35, seed_astro=60)
    return dict(worker_id=0, topo=topo, topo_idx=topo_idx, topo_dir=topo_dir,
                conn_prob=CONN, conn_rule='flat', conn_periodic=False,
                p0_conn=float('nan'), d0_conn=float('nan'), beta_conn=float('nan'),
                stoa_gate_p=1.0, params_list=[p for p, _ in runs],
                theta_list=[p for p, _ in runs], iter_indices=list(range(len(runs))),
                seed_runs=[s for _, s in runs], cli=cli,
                scratch_root=os.path.join(out_dir, 'scratch'),
                topo_seed_base=topo_seed_base)


def spikes(topo_dir, it):
    z = np.load(os.path.join(topo_dir, 'iter_%05d.npz' % it))
    return z['spk_N_t'], z['spk_N_i']


def same(a, b):
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def result_array(name, n):
    """An array of the last standalone run, read from its results folder."""
    from brian2 import get_device
    fname = os.path.join(get_device().results_dir,
                         '%s_%d' % (name, zlib.crc32(name.encode('utf-8'))))
    raw = np.fromfile(fname, dtype=np.uint8)
    for dt in (np.float64, np.float32, np.int32):
        if raw.size == n * np.dtype(dt).itemsize:
            return raw.view(dt).copy()
    raise ValueError('%s: %d bytes for %d values' % (name, raw.size, n))


# ------------------------------------------------------------------- checks
def s0_helpers():
    a = R.derive_network_seeds(50, 123456)
    b = R.derive_network_seeds(50, 123456)
    c = R.derive_network_seeds(50, 123457)
    d = R.derive_network_seeds(51, 123456)
    vals = list(a.values())
    check('S0a', a == b, 'deterministic: %s' % a)
    check('S0b', all(1 <= v < R.SEED_MAX for v in vals) and len(set(vals)) == 4,
          'in [1, 2**31-1), four distinct values')
    check('S0c', all(a[k] != c[k] for k in a) and all(a[k] != d[k] for k in a),
          'every seed changes with network_key and with seed_device')
    ok_valid = [R.validate_run_seed(v) for v in (0, 1, 2**31 - 2, np.int64(7))]
    refused = 0
    for bad in (-1, 2**31 - 1, 2**31, 3.5, 'x'):
        try:
            R.validate_run_seed(bad)
        except (ValueError, TypeError):
            refused += 1
    check('S0d', ok_valid == [0, 1, 2**31 - 2, 7] and refused == 5,
          'run seeds 0 .. 2**31-2 accepted, 5 invalid values refused')


def s11_refusals(tmp):
    topo = {'S_i': np.zeros(0, int)}            # never reached: refused first
    p = make_pack(topo, os.path.join(tmp, 's11'), [(vector(), 5)], 1, 0.1)
    p.pop('topo_seed_base')
    try:
        S._worker_entry(p)
        check('S11a', False, 'a pack without topo_seed_base was accepted')
    except KeyError as e:
        check('S11a', 'topo_seed_base' in str(e), 'KeyError: %s' % e)
    p = make_pack(topo, os.path.join(tmp, 's11'), [(vector(), -1)], 1, 0.1)
    try:
        S._worker_entry(p)
        check('S11b', False, 'run seed -1 accepted')
    except ValueError as e:
        check('S11b', True, 'ValueError: %s' % e)


def s12_minimal(tmp):
    from brian2 import (set_device, get_device, start_scope, NeuronGroup, StateMonitor,
                        Network, ms, defaultclock)
    d = os.path.join(tmp, 's12')
    set_device('cpp_standalone', build_on_run=False, directory=d)
    dev = get_device()
    dev.reinit()
    dev.activate(build_on_run=False, directory=d)
    start_scope()
    dev.seed(R.derive_network_seeds(50, 0)['seed_build'])
    defaultclock.dt = 0.1 * ms
    G = NeuronGroup(8, '''dv/dt = -v/(10*ms) + 0.5*xi/sqrt(10*ms) : 1
                          frozen : 1''', method='euler', name='G')
    G.frozen = 'rand()'
    M = StateMonitor(G, ['v', 'frozen'], record=True, name='M')
    H = R.install_run_seed(dev, NeuronGroup)
    net = Network(G, M, H)
    net.run(5 * ms)
    dev.build(run=False, directory=None)
    out = {}
    for tag, s in (('a', 5), ('b', 5), ('c', 6)):
        dev.run(run_args={H.run_seed: s})
        out[tag] = (np.array(M.v), np.array(M.frozen), R.check_run_seed(net, s))
    check('S12a', np.array_equal(out['a'][1], out['c'][1]),
          'pattern set before the reseed: identical for run seeds 5 and 6')
    check('S12b', np.array_equal(out['a'][0], out['b'][0]) and
          not np.array_equal(out['a'][0], out['c'][0]),
          'noise: seeds 5 / 5 identical, 5 / 6 different')
    check('S12c', (out['a'][2], out['c'][2]) == (5, 6), 'applied seeds read back: 5, 6')
    try:
        dev.run(run_args={})
        check('S12d', False, 'a run without its seed completed')
    except Exception as e:
        check('S12d', '97' in str(e) or 'failed' in str(e).lower(),
              'run without its seed refused: %s' % str(e).splitlines()[0][:80])


def s1_to_s8(tmp):
    tsb_a, tsb_b = 1234567, 7654321
    topo_a = topology(tsb_a)
    v1, v2 = vector(4.0), vector(6.0)
    s1, s2, s3 = 1111, 2222, 3333
    out = os.path.join(tmp, 'chunkA')
    pack_a = make_pack(topo_a, out, [(v1, s1), (v1, s2), (v1, s1), (v2, s3)], tsb_a, 0.5)
    res = S._worker_entry(pack_a)
    td = pack_a['topo_dir']
    ok = sum(1 for r in res if r.get('ok'))
    r0, r1, r2, r3 = (spikes(td, k) for k in range(4))
    check('S1', ok == 4 and len(r0[0]) > 20 and not same(r0, r1),
          'v1 seed %d vs %d: %d vs %d spikes, trains differ' % (s1, s2, len(r0[0]),
                                                                 len(r1[0])))
    check('S2', same(r0, r2), 'v1 seed %d twice in one chunk: identical (%d spikes)'
          % (s1, len(r0[0])))
    # S8 records
    z = np.load(os.path.join(td, 'iter_00001.npz'))
    j = json.load(open(os.path.join(td, 'iter_00001.json')))
    ns = R.derive_network_seeds(50, tsb_a)
    keys_ok = all(int(z[k]) == ns[k] and j[k] == ns[k] for k in ns)
    check('S8', int(z['seed_run']) == s2 and j['seed_run'] == s2 and
          bool(z['seed_verified']) and j['seed_verified'] is True and
          str(z['noise_mode']) == 'fresh' and j['noise_mode'] == 'fresh' and
          int(z['topo_seed_base']) == tsb_a and keys_ok and
          str(z['rng_scheme']) == R.RNG_SCHEME == j['rng_scheme'],
          'npz and json: seed_run %d, verified, fresh, network seeds = derived' % s2)
    # S3 order independence (new build, other order)
    out_b = os.path.join(tmp, 'chunkB')
    pack_b = make_pack(topo_a, out_b, [(v2, s3), (v1, s1)], tsb_a, 0.5)
    S._worker_entry(pack_b)
    b0, b1 = spikes(pack_b['topo_dir'], 0), spikes(pack_b['topo_dir'], 1)
    check('S3', same(b0, r3) and same(b1, r0),
          'pairs (v2, %d) and (v1, %d) in the other order, new build: identical'
          % (s3, s1))
    # S6 replay in a new process
    topo_file = os.path.join(tmp, 'topo_a.npz')
    np.savez(topo_file, **{k: np.asarray(v) for k, v in topo_a.items()})
    rec = np.load(os.path.join(td, 'iter_00003.npz'))
    job = dict(topo_file=topo_file, out=os.path.join(tmp, 'replay'), here=HERE,
               params=[float(x) for x in rec['params']], seed_run=int(rec['seed_run']),
               topo_seed_base=int(rec['topo_seed_base']), NN=NN, NA=NA, CONN=CONN)
    job_file = os.path.join(tmp, 'replay_job.json')
    json.dump(job, open(job_file, 'w'))
    script = textwrap.dedent('''
        import json, os, sys
        import numpy as np
        import matplotlib; matplotlib.use('Agg')
        job = json.load(open(sys.argv[1]))
        sys.path.insert(0, job['here'])
        import HPC_main_sweep as S
        topo = dict(np.load(job['topo_file'], allow_pickle=True))
        td = os.path.join(job['out'], 'topo_00000'); os.makedirs(td, exist_ok=True)
        cli = dict(lib_dir=job['here'], out_dir=job['out'], simtime=0.5, mode='Full',
                   Nn=job['NN'], Na=job['NA'], seed_device=50, seed_neuron=39,
                   seed_synapse=35, seed_astro=60)
        p = np.array(job['params'])
        S._worker_entry(dict(worker_id=0, topo=topo, topo_idx=0, topo_dir=td,
            conn_prob=job['CONN'], conn_rule='flat', conn_periodic=False,
            p0_conn=float('nan'), d0_conn=float('nan'), beta_conn=float('nan'),
            stoa_gate_p=1.0, params_list=[p], theta_list=[p], iter_indices=[0],
            seed_runs=[job['seed_run']], cli=cli,
            scratch_root=os.path.join(job['out'], 'scratch'),
            topo_seed_base=job['topo_seed_base']))
    ''')
    sfile = os.path.join(tmp, 'replay.py')
    open(sfile, 'w').write(script)
    pr = subprocess.run([sys.executable, sfile, job_file], capture_output=True, text=True)
    try:
        rp = spikes(os.path.join(job['out'], 'topo_00000'), 0)
        check('S6', pr.returncode == 0 and same(rp, r3),
              'new process, from the record: identical (%d spikes)' % len(rp[0]))
    except Exception as e:
        check('S6', False, 'replay failed: %s %s' % (e, pr.stderr[-300:]))
    # S4 / S5 frozen draws, from the binary
    vals = {}
    for tag, tsb, s in (('a1', tsb_a, s1), ('a2', tsb_a, s2), ('b', tsb_b, s1)):
        topo = topo_a if tsb == tsb_a else topology(tsb_b)
        pk = make_pack(topo, os.path.join(tmp, 'frozen_' + tag), [(v1, s)], tsb, 0.0002)
        S._worker_entry(pk)
        vals[tag] = (result_array('_array_Neuron_bias_unit', NN),
                     result_array('_array_Astrocyte_C', NA))
    check('S4', np.array_equal(vals['a1'][0], vals['a2'][0]) and
          np.array_equal(vals['a1'][1], vals['a2'][1]),
          'one network, run seeds %d / %d: bias_unit and C(0.2 ms) identical' % (s1, s2))
    check('S5', not np.array_equal(vals['a1'][0], vals['b'][0]) and
          not np.array_equal(vals['a1'][1], vals['b'][1]),
          'topo_seed_base %d vs %d: bias_unit and C(0.2 ms) differ' % (tsb_a, tsb_b))
    # S7 guard
    from brian2.devices.cpp_standalone.device import CPPStandaloneDevice
    orig = CPPStandaloneDevice.insert_code

    def sabotaged(self, slot, code):
        if 'SPEC Block 11' in code:
            return None
        return orig(self, slot, code)
    CPPStandaloneDevice.insert_code = sabotaged
    try:
        pk = make_pack(topo_a, os.path.join(tmp, 'guard'), [(v1, s1), (v1, s2)], tsb_a, 0.1)
        res = S._worker_entry(pk)
    finally:
        CPPStandaloneDevice.insert_code = orig
    flag = any(r.get('seed_failure') for r in res)
    no_npz = not any(f.endswith('.npz') for f in os.listdir(pk['topo_dir']))
    fl = os.path.join(pk['topo_dir'], '_failures.jsonl')
    logged = os.path.exists(fl) and 'SeedNotAppliedError' in open(fl).read()
    check('S7', flag and no_npz and logged,
          'reseed removed: seed_failure %s, npz written %s, logged %s'
          % (flag, not no_npz, logged))


def s9_neuronal(tmp):
    tsb = 2468
    topo = topology(tsb, mode='Neuronal')
    v = vector(5.0)
    pk = make_pack(topo, os.path.join(tmp, 'neuronal'), [(v, 10), (v, 20), (v, 10)], tsb,
                   0.5, mode='Neuronal')
    S._worker_entry(pk)
    a, b, c = (spikes(pk['topo_dir'], k) for k in range(3))
    check('S9', len(a[0]) > 20 and not same(a, b) and same(a, c),
          'Neuronal: seeds 10 / 20 differ, 10 / 10 identical (%d spikes)' % len(a[0]))


def s10_single_run(tmp):
    def run(seed, tag):
        out = os.path.join(tmp, 'single_' + tag)
        cmd = [sys.executable, os.path.join(HERE, 'HPC_single_run.py'), '--out_dir', out,
               '--lib_dir', HERE, '--simtime', '0.3', '--mode', 'Full', '--Nn', str(NN),
               '--Na', str(NA), '--c_max', str(CMAX), '--conn_prob', str(CONN),
               '--conn_rule', 'flat', '--topology_mode', 'wallach', '--seed_run', str(seed),
               '--dpi', '40']
        pr = subprocess.run(cmd, capture_output=True, text=True)
        if pr.returncode != 0:
            raise RuntimeError(pr.stderr[-500:])
        z = np.load(os.path.join(out, 'iter_0000000.npz'))
        return (z['spk_N_t'], z['spk_N_i']), int(z['seed_run'])
    try:
        a, sa = run(11, 'a')
        b, sb = run(12, 'b')
        c, sc = run(11, 'c')
        check('S10', len(a[0]) > 5 and not same(a, b) and same(a, c) and
              (sa, sb, sc) == (11, 12, 11),
              'single run: 11 / 12 differ, 11 / 11 identical; recorded 11, 12, 11')
    except Exception as e:
        check('S10', False, 'single run failed: %s' % e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='S0, S11, S12 only')
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix='smoke_seeding_')
    print('work folder:', tmp, flush=True)
    s0_helpers()
    s11_refusals(tmp)
    s12_minimal(tmp)
    if not a.quick:
        s1_to_s8(tmp)
        s9_neuronal(tmp)
        s10_single_run(tmp)
    n_ok = sum(ok for _, ok, _ in RESULTS)
    n_bad = len(RESULTS) - n_ok
    print('RESULT: %d passed, %d failed' % (n_ok, n_bad))
    return 0 if n_bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
