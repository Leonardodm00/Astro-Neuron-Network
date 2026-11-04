#!/usr/bin/env python3
"""
smoke_test_astro_drive.py -- checks of Block 10 (specs/SPEC.md): an
astrocyte's glutamate drive Y_extra is the MEDIAN of Y_A over its links
(numpy.median convention: the mean of the two middle values for an even
count; 0 for an astrocyte without a link), recomputed at the start of every
time step by the Brian2 Function astro_median_Y_A (C++ for cpp_standalone,
NumPy for the runtime device).

  A1 the model code: Synapse_to_astro builds the links with no model, adds the
     run_regularly "Y_extra = astro_median_Y_A(i, t)", and no summed Y_extra
     line (sum or mean) is left
  A2 the NumPy implementation, through the library's own Synapse_to_astro on a
     stand-in network (runtime device, numpy target): a Synapses source with
     known Y_A, links given as explicit (i, j) pairs as build_topology gives
     them. After two steps the six astrocytes hold the medians
     2, 10, 0, 5, 5, 2.5 mM: odd counts, a skewed set (4, 10, 20: the mean
     would be 11.3), no link, one link, four equal links (the contact gate
     removes links, it does not rescale the drive), and an even count with an
     outlier (1, 2, 3, 100: 2.5; the mean would be 26.5). Relative error
     < 1e-12 (float64 roundoff)
  A3 the C++ implementation on the same stand-in, built with cpp_standalone
     (about 20 s): the same six values (the production path)
  A4 end to end (cpp_standalone, about 40 s): HPC_single_run.py, Full mode,
     nominal parameters, 1200 + 1200 cells per mm^2 on a 200 um well
     (48 + 48), conn_prob 0.225, gate 0.6, 3 s, seed 7: exit 0, no Brian2 NaN
     warning, the record says the astrocytes started from the driven state
     (astro_ic, SPEC Block 13). Then (A4b, about 40 s) the same network from
     the quiescent start of before Block 13 (build_and_run(...,
     astro_ic='quiescent')), where a healthy astrocyte crosses C_osc once
     after the start: an event from at least half of the 48 astrocytes. With
     the summed drive of before 2026-10-31 that run gave four NaN warnings
     (Gamma_A, I, C, h) and one astrocyte event in all. (From the driven
     start the astrocytes begin above C_osc and refractory, so an event count
     says nothing about their health there.)

Run (from this folder):
    python3 smoke_test_astro_drive.py | tail -1            (about 1 min)
    python3 smoke_test_astro_drive.py --no-sim | tail -1   (A1, A2 only; seconds)
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, 'ASD_fun_BD_cpp.py')
WANT = [2.0, 10.0, 0.0, 5.0, 5.0, 2.5]                    # mM, astrocytes 0..5
RESULTS = {'passed': 0, 'failed': 0, 'skipped': 0}

STAND_IN = r'''
import json, sys
import numpy as np
sys.path.insert(0, %(here)r)
from brian2 import (NeuronGroup, Synapses, Network, defaultclock, prefs, mmole, set_device,
                    get_device)
standalone = %(standalone)r
if standalone:
    set_device('cpp_standalone', directory=%(build)r, build_on_run=False)
else:
    prefs.codegen.target = 'numpy'
import ASD_fun_BD_cpp as M
y = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 100.0])   # mM
pre_n = NeuronGroup(len(y), 'v : 1', name='pre_stand_in')
post_n = NeuronGroup(1, 'v : 1', name='post_stand_in')
syn = Synapses(pre_n, post_n, model='Y_A : mole', name='Synapse_stand_in')
syn.connect(i=np.arange(len(y)), j=np.zeros(len(y), dtype=int))
syn.Y_A = y * mmole
ast = NeuronGroup(6, 'Y_extra : mole', name='Astrocyte_stand_in')
# astro 0 <- 0,1,2; 1 <- 3,4,5; 2 <- none; 3 <- 6; 4 <- 7..10 (equal); 5 <- 11..14 (even, outlier)
i = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
j = [0, 0, 0, 1, 1, 1, 3, 4, 4, 4, 4, 5, 5, 5, 5]
links, back = M.Synapse_to_astro(syn, ast, connections=[i, j])
net = Network(pre_n, post_n, syn, ast, links)
net.run(2 * defaultclock.dt)
if standalone:
    get_device().build(directory=%(build)r, compile=True, run=True)
got = np.asarray(ast.Y_extra / mmole, dtype=np.float64).tolist()
print(json.dumps(dict(got=got, back=[list(map(int, back[0])), list(map(int, back[1]))])))
'''


def check(cond, label):
    RESULTS['passed' if cond else 'failed'] += 1
    print('  %s %s' % ('ok  ' if cond else 'FAIL', label))
    return cond


def skip(label):
    RESULTS['skipped'] += 1
    print('  SKIP %s' % label)


def a1():
    print('\n[A1] the model code')
    text = open(LIB, encoding='utf-8').read()
    fn = text[text.index('def Synapse_to_astro('):text.index('def astro_median_function(')]
    check("model=''" in fn and "'Y_extra = astro_median_Y_A(i, t)'" in fn
          and not re.search(r'Y_extra_post\s*=', text),
          'links without a model, the median run_regularly, no summed Y_extra line left')


def stand_in(tmp, standalone):
    code = STAND_IN % dict(here=HERE, standalone=standalone, build=os.path.join(tmp, 'build'))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=tmp)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-3000:])
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def values_ok(d, label):
    if not check(d is not None, '%s: stand-in network built and run' % label):
        return
    got = d['got']
    rel = max(abs(g - w) / max(abs(w), 1.0) for g, w in zip(got, WANT))
    check(rel < 1e-12, '%s: Y_extra = %s mM (want %s; max rel. error %.1e)'
          % (label, ['%.6g' % g for g in got], ['%.6g' % w for w in WANT], rel))
    check(d['back'] == [[0, 0, 0, 1, 1, 1, 3, 4, 4, 4, 4, 5, 5, 5, 5], list(range(15))],
          '%s: the returned pairs are (astrocyte, synapse), as Astro_to_Syn expects' % label)


def a2(tmp):
    print('\n[A2] the NumPy implementation (runtime device)')
    values_ok(stand_in(tmp, False), 'numpy')


def a3(tmp):
    print('\n[A3] the C++ implementation (cpp_standalone)')
    values_ok(stand_in(tmp, True), 'C++')


def a4(tmp):
    print('\n[A4] end to end: HPC_single_run.py, Full mode, nominal parameters, 3 s')
    out = os.path.join(tmp, 'out')
    cmd = [sys.executable, os.path.join(HERE, 'HPC_single_run.py'), '--out_dir', out, '--lib_dir', HERE,
           '--simtime', '3', '--mode', 'Full', '--density', '1200', '--density_astro', '1200',
           '--c_max', '200', '--conn_prob', '0.225', '--conn_rule', 'flat', '--topology_mode', 'wallach',
           '--stoa_gate_p', '0.6', '--seed_run', '7', '--dpi', '60']
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)
    log = r.stdout + r.stderr
    n_nan = len(re.findall(r"has NaN", log))
    npz = sorted(glob.glob(os.path.join(out, 'iter_*.npz')))
    if not check(r.returncode == 0 and len(npz) == 1, 'exit 0, one spike file written'):
        print(log[-3000:])
        return
    import numpy as np
    z = np.load(npz[0])
    n_a = int(z['spk_A_t'].size)
    n_n = int(z['spk_N_t'].size)
    n_astro = int(round(1200 * 0.2 ** 2))                 # density x well area: 48 astrocytes
    active = int(np.unique(z['spk_A_i']).size)
    check(n_nan == 0, 'no Brian2 NaN warning (%d)' % n_nan)
    ic = str(z['astro_ic']) if 'astro_ic' in z.files else None
    check(ic == 'driven-v1-2026-11-04' and n_n > 0,
          'driven start recorded (astro_ic %s); neuron spikes %d; astrocyte events %d of %d astrocytes'
          % (ic, n_n, active, n_astro))
    a4b(tmp, cmd, n_astro)


A4B = r'''
import json, sys
import numpy as np
sys.path.insert(0, %(here)r)
import HPC_single_run as R
args = R.build_parser().parse_args(%(argv)r)
args.Nn, args.Na = R.resolve_population_from_density(args.Nn, args.Na, args.c_max, density=args.density,
                                                     density_astro=args.density_astro)
topo = R.build_topology(
    Nn=args.Nn, Na=args.Na, c_max=args.c_max, conn_prob=args.conn_prob,
    syn_prob_csv=R._resolve_syn_pdist_csv(args), displ_bias=args.displ_bias, gj_dist=args.gj_dist,
    stoa_cutoff=args.stoa_cutoff, stoa_sigma=args.stoa_sigma, seed_neuron=args.seed_neuron,
    seed_synapse=args.seed_synapse, seed_astro=args.seed_astro, mode='Full',
    topology_mode=args.topology_mode, gj_max_dist=args.gj_max_dist, conn_rule=args.conn_rule,
    p0_conn=args.p0_conn, d0_conn=args.d0_conn, beta_conn=args.beta_conn,
    conn_periodic=args.conn_periodic, stoa_gate_p=args.stoa_gate_p, seed_stoa_gate=args.seed_stoa_gate)
res = R.build_and_run(topo, args, R.resolve_params(args).astype(np.float64), astro_ic='quiescent')
print('A4B ' + json.dumps(dict(active=int(np.unique(res['spk_A_i']).size), n_a=int(res['spk_A_t'].size),
                              n_n=int(res['spk_N_t'].size), Na=int(args.Na), ic=res['astro_ic'])))
'''


def a4b(tmp, cmd, n_astro):
    print('\n[A4b] the same network from the quiescent start (before Block 13)')
    argv = cmd[2:]
    out = os.path.join(tmp, 'out_a4b')
    argv[argv.index('--out_dir') + 1] = out
    r = subprocess.run([sys.executable, '-c', A4B % dict(here=HERE, argv=argv)], capture_output=True,
                       text=True, cwd=tmp)
    log = r.stdout + r.stderr
    line = [ln for ln in r.stdout.splitlines() if ln.startswith('A4B ')]
    if not check(r.returncode == 0 and len(line) == 1, 'build_and_run(astro_ic=quiescent): exit 0'):
        print(log[-3000:])
        return
    d = json.loads(line[0][4:])
    check(len(re.findall(r"has NaN", log)) == 0, 'no Brian2 NaN warning')
    # with the summed drive one astrocyte still fired before the NaN (planted fault, 2026-10-31),
    # so the check asks for most astrocytes, not for any event
    check(d['ic'] == 'quiescent' and d['active'] >= n_astro // 2 and d['n_n'] > 0,
          'astrocytes with an event %d of %d (at least half), astrocyte events %d, neuron spikes %d'
          % (d['active'], n_astro, d['n_a'], d['n_n']))


def main():
    import importlib.util
    tmp = tempfile.mkdtemp(prefix='astro_drive_')
    try:
        a1()
        a2(tmp)
        if '--no-sim' in sys.argv:
            skip('A3 (--no-sim)')
            skip('A4 (--no-sim)')
        elif importlib.util.find_spec('brian2') is None:
            skip('A3, A4: brian2 not importable')
        else:
            a3(tmp)
            a4(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nRESULT: %d passed, %d failed%s' % (RESULTS['passed'], RESULTS['failed'],
          (', %d skipped' % RESULTS['skipped']) if RESULTS['skipped'] else ''))
    return 0 if RESULTS['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
