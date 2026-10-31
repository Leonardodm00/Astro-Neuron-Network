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
     warning, an event from at least half of the 48 astrocytes. With the
     summed drive of before 2026-10-31 the same run gave four NaN warnings
     (Gamma_A, I, C, h) and one astrocyte event in all

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
    # with the summed drive one astrocyte still fired before the NaN (planted fault, 2026-10-31),
    # so the check asks for most astrocytes, not for any event
    check(active >= n_astro // 2 and n_n > 0,
          'astrocytes with an event %d of %d (at least half), astrocyte events %d, neuron spikes %d'
          % (active, n_astro, n_a, n_n))


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
