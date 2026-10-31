#!/usr/bin/env python3
"""
smoke_test_astro_drive.py -- checks of Block 10 (specs/SPEC.md): an
astrocyte's glutamate drive Y_extra is the mean of Y_A over its links
(Y_A_pre / N_incoming, summed), no longer the sum.

  A1 the model line: ASD_fun_BD_cpp.py declares
     "Y_extra_post = Y_A_pre / N_incoming : mole (summed)" exactly once, and
     the old summed line is gone
  A2 the semantics, through the library's own Synapse_to_astro (Brian2
     runtime device, numpy target, no compiler): a stand-in source group with
     known Y_A, links given as explicit (i, j) pairs as build_topology gives
     them. After one step each astrocyte's Y_extra equals the mean of its
     sources' Y_A (relative error < 1e-12, float64 roundoff); an astrocyte
     with no link stays at 0; the mean does not depend on how many links an
     astrocyte has when its sources share one value (the contact gate removes
     links, it does not rescale the drive)
  A3 end to end (cpp_standalone, about 1 min): HPC_single_run.py, Full mode,
     nominal parameters, 1200 + 1200 cells per mm^2 on a 200 um well
     (48 + 48), conn_prob 0.225, gate 0.6, 3 s, seed 7: exit 0, no Brian2
     NaN warning, and an event from at least half of the 48 astrocytes. With
     the summed drive restored (planted fault, 2026-10-31) the same run gives
     four NaN warnings (Gamma_A, I, C, h) and one astrocyte event in all;
     A1 and A2 fail as well (specs/SPEC.md section 8)

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
NEW_LINE = 'Y_extra_post = Y_A_pre / N_incoming : mole (summed)'
OLD_LINE = re.compile(r'Y_extra_post\s*=\s*Y_A_pre\s*:\s*mole\s*\(summed\)')
RESULTS = {'passed': 0, 'failed': 0, 'skipped': 0}

A2_CODE = r'''
import json, sys
import numpy as np
sys.path.insert(0, %r)
from brian2 import NeuronGroup, Network, defaultclock, prefs, mmole
prefs.codegen.target = 'numpy'
import ASD_fun_BD_cpp as M
y = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 5.0, 5.0, 5.0, 5.0, 5.0])     # mM, sources 0..10
src = NeuronGroup(len(y), 'Y_A : mole', name='src_stand_in')
src.Y_A = y * mmole
ast = NeuronGroup(5, 'Y_extra : mole', name='astro_stand_in')
# astro 0 <- 0,1,2; astro 1 <- 3,4,5; astro 2 <- none; astro 3 <- 6 (one link); astro 4 <- 7..10 (four links)
i = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
j = [0, 0, 0, 1, 1, 1, 3, 4, 4, 4, 4]
syn, back = M.Synapse_to_astro(src, ast, connections=[i, j])
net = Network(src, ast, syn)
net.run(defaultclock.dt)
got = np.asarray(ast.Y_extra / mmole, dtype=np.float64).tolist()
print(json.dumps(dict(got=got, back=[list(map(int, back[0])), list(map(int, back[1]))])))
''' % HERE


def check(cond, label):
    RESULTS['passed' if cond else 'failed'] += 1
    print('  %s %s' % ('ok  ' if cond else 'FAIL', label))
    return cond


def skip(label):
    RESULTS['skipped'] += 1
    print('  SKIP %s' % label)


def a1():
    print('\n[A1] the model line')
    text = open(LIB, encoding='utf-8').read()
    check(text.count(NEW_LINE) == 1 and not OLD_LINE.search(text),
          'Synapse_to_astro declares the mean over links once; the summed line is gone')


def a2():
    print('\n[A2] summed-variable semantics through Synapse_to_astro (runtime, numpy)')
    r = subprocess.run([sys.executable, '-c', A2_CODE], capture_output=True, text=True, cwd=HERE)
    if not check(r.returncode == 0, 'stand-in network built and run one step'):
        print(r.stdout[-1500:], r.stderr[-2500:])
        return
    d = json.loads(r.stdout.strip().splitlines()[-1])
    got = d['got']
    want = [2.0, 34.0 / 3.0, 0.0, 5.0, 5.0]
    rel = max(abs(g - w) / max(abs(w), 1.0) for g, w in zip(got, want))
    check(rel < 1e-12, 'Y_extra = mean of the sources: %s (want %s, max rel. error %.1e)'
          % (['%.6g' % g for g in got], ['%.6g' % w for w in want], rel))
    check(got[2] == 0.0, 'an astrocyte with no link stays at 0')
    check(abs(got[3] - got[4]) < 1e-12, 'one link or four links of equal Y_A give the same drive (%.6g, %.6g)'
          % (got[3], got[4]))
    check(d['back'] == [[0, 0, 0, 1, 1, 1, 3, 4, 4, 4, 4], list(range(11))],
          'the returned pairs are (astrocyte, synapse), as Astro_to_Syn expects')


def a3():
    print('\n[A3] end to end: HPC_single_run.py, Full mode, nominal parameters, 3 s')
    import importlib.util
    if importlib.util.find_spec('brian2') is None:
        return skip('brian2 not importable')
    tmp = tempfile.mkdtemp(prefix='astro_drive_')
    try:
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    a1()
    a2()
    if '--no-sim' in sys.argv:
        skip('A3 (--no-sim)')
    else:
        a3()
    print('\nRESULT: %d passed, %d failed%s' % (RESULTS['passed'], RESULTS['failed'],
          (', %d skipped' % RESULTS['skipped']) if RESULTS['skipped'] else ''))
    return 0 if RESULTS['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
