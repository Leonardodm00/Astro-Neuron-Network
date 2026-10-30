#!/usr/bin/env python3
"""
smoke_test_range_check.py -- checks of hpc/range_check/ (specs/SPEC.md Block 6).

  R1  oat design on the repository's 36-D root pipeline: every free axis with
      lo < hi gets K points, each point differs from the nominal theta on exactly
      one axis, the end points lie on the theta box, padding to n_workers x k
  R2  the same pipeline with the LindaColloDUP freeze patch and freeze list:
      13 free axes, the frozen ones absent, VA/VR held at the freeze values
  R3  --rc-like: parsing the rebuilt argument list reproduces every job_args value
  R4  37-D Giulia copy: the design builds for its synapse_astro group (22 axes)
  R5  end to end (needs Brian2 and a C++ compiler): oat phase on a tiny
      Neuronal network (Nn 12, 100 um, 0.3 s, 2 workers, K = 2, R = 1), every
      design point found once in the sidecars; prior phase 2 runs; the analysis
      labels every oat run; refuses an unclamped model and an occupied folder
  R6  analysis on synthetic runs: Poisson trains at 5 Hz give a mean rate within
      5 % (3 Poisson SD); a silent run gives active fraction 0 and is 'silent'
  R7  bytes: pure ASCII, LF only, in every file of this folder
  R8  submit_range_check.sh with stub module/conda/python: for SETUP=lcdup it
      calls range_check.py 3 times (oat at 0.1 0.35 0.6) and once (prior), then
      the analysis and the plots, with the LindaColloDUP settings; SETUP=giulia
      finds LIKE in the hhgap_v5 campaign and passes EXTRA to every phase and
      OAT_EXTRA to the oat phases only; an unknown SETUP exits 2

Run:  python3 smoke_test_range_check.py | tail -1        (about 2 min with R5)
      python3 smoke_test_range_check.py --no-sim          (skip R5)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, '..', 'Phenomenological_finalv1')
GIULIA = os.path.join(PIPE, 'Giulia')
CSV = os.path.join(PIPE, 'MEAMA', 'synapse_pdist.csv')
LCDUP_FREEZE = 'Cm,I_inj,VA=-50,DeltaA,VR=-55,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G'
RESULTS = {'passed': 0, 'failed': 0, 'skipped': 0}


def check(cond, label):
    RESULTS['passed' if cond else 'failed'] += 1
    print('  %s %s' % ('ok  ' if cond else 'FAIL', label))


def skip(label):
    RESULTS['skipped'] += 1
    print('  SKIP %s' % label)


def make_lib(dst, src=PIPE, freeze_patch=False):
    os.makedirs(dst, exist_ok=True)
    for f in ('HPC_main_sweep.py', 'HPC_single_run.py', 'ASD_fun_BD_cpp.py'):
        shutil.copy2(os.path.join(src, f), dst)
    shutil.copy2(CSV, dst)
    if freeze_patch:
        r = subprocess.run([sys.executable, os.path.join(PIPE, 'LindaColloDUP', 'apply_freeze_patch.py'),
                            '--file', os.path.join(dst, 'HPC_main_sweep.py')], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
    return dst


def in_subprocess(lib, body):
    """Run `body` with range_check imported as RC and the setup module as S, in a
    fresh interpreter (each pipeline copy defines its own HPC_main_sweep)."""
    code = ('import sys, json\nsys.path.insert(0, %r)\nimport numpy as np\nimport range_check as RC\n'
            'S = RC.load_sweep_module(%r)\n' % (HERE, lib)) + body
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


DESIGN_BODY = '''
args = S.build_parser().parse_args(%r)
free = RC.free_axes(S, args)
nominal, pts, labels = RC.oat_design(S, free, 7)
k, padded = RC.pad_design(nominal, pts, 48)
B = S.PARAM_BOUNDS_THETA
one_axis = all(set(np.flatnonzero(p != nominal).tolist()) <= {l['index']} for p, l in zip(pts, labels))
n_at_nominal = sum(1 for p in pts if not np.any(p != nominal))
ends = all(abs(p[l['index']] - B[l['index'], 0]) < 1e-12 or abs(p[l['index']] - B[l['index'], 1]) < 1e-12
           for i, (p, l) in enumerate(zip(pts, labels)) if i %% 7 in (0, 6))
lohi = [int(j) for j in free if B[j, 1] > B[j, 0]]
nat = S.theta_to_natural(nominal)
print(json.dumps(dict(free=[S.PARAM_NAMES[j] for j in free], n_lohi=len(lohi), n_pts=len(pts), k=k,
                      n_at_nominal=n_at_nominal,
                      n_pad=len(padded) - len(pts), one_axis=one_axis, ends=ends,
                      va=float(nat[S.PARAM_NAMES.index('VA')]), vr=float(nat[S.PARAM_NAMES.index('VR')]))))
'''


def r1(tmp):
    print('\n[R1] oat design, 36-D root pipeline')
    d = in_subprocess(make_lib(os.path.join(tmp, 'lib36')), DESIGN_BODY % (
        ['--out_dir', 'x', '--mode', 'Neuronal', '--sweep_group', 'neuron_synapse'],))
    check(d is not None, 'design built')
    if d:
        check(d['n_pts'] == 7 * d['n_lohi'] and d['n_lohi'] == len(d['free']) == 23,
              '23 free axes x 7 points = %d' % d['n_pts'])
        check(d['one_axis'] and d['ends'], 'each point moves only its own axis (%d coincide with nominal); '
              'end points on the theta box' % d['n_at_nominal'])
        check(d['k'] == 4 and d['n_pad'] == 48 * 4 - 161, 'padding to 48 x k (k=%d, %d nominal)' % (d['k'], d['n_pad']))


def r2(tmp):
    print('\n[R2] oat design with the LindaColloDUP freeze')
    d = in_subprocess(make_lib(os.path.join(tmp, 'lib36f'), freeze_patch=True), DESIGN_BODY % (
        ['--out_dir', 'x', '--mode', 'Neuronal', '--sweep_group', 'neuron_synapse', '--freeze', LCDUP_FREEZE],))
    check(d is not None, 'design built')
    if d:
        frozen = {x.split('=')[0] for x in LCDUP_FREEZE.split(',')}
        check(len(d['free']) == 13 and not frozen & set(d['free']), '13 free axes, none frozen: %s' % d['free'])
        check(abs(d['va'] + 50) < 1e-12 and abs(d['vr'] + 55) < 1e-12, 'VA, VR held at -50, -55 mV')
        check(d['n_pts'] == 91 and d['k'] == 2 and d['n_pad'] == 5, '91 points, k=2, 5 nominal')


def r3(tmp):
    print('\n[R3] --rc-like rebuilds the arguments')
    body = '''
p = S.build_parser()
a = p.parse_args(%r)
job = vars(a)
argv = RC.args_from_job_args(p, job, skip=())
b = p.parse_args(argv)
same = all(getattr(b, k) == v for k, v in job.items())
print(json.dumps(dict(same=same, n=len(job))))
''' % (['--out_dir', '/x/y', '--mode', 'Full', '--sweep_group', 'neuron', '--conn_prob_lo', '0.15',
        '--conn_prob_hi', '0.45', '--simtime', '12.5', '--c_max', '400', '--density', '1300'],)
    d = in_subprocess(os.path.join(tmp, 'lib36'), body)
    check(d is not None and d['same'], 'all %s values reproduced' % (d or {}).get('n'))


def r4(tmp):
    print('\n[R4] 37-D Giulia copy')
    if not os.path.isfile(os.path.join(GIULIA, 'HPC_main_sweep.py')):
        return skip('no Giulia copy')
    d = in_subprocess(make_lib(os.path.join(tmp, 'lib37g'), src=GIULIA), DESIGN_BODY % (
        ['--out_dir', 'x', '--mode', 'Full', '--sweep_group', 'synapse_astro'],))
    check(d is not None and len(d['free']) == 22 and d['one_axis'],
          'synapse_astro: 22 free axes, one-axis points (%s)' % (d or {}).get('n_pts'))


def r5(tmp):
    print('\n[R5] end to end on a tiny network')
    import importlib.util
    if importlib.util.find_spec('brian2') is None:
        return skip('brian2 not importable')
    lib = os.path.join(tmp, 'lib36')
    out = os.path.join(tmp, 'out')
    common = ['--lib_dir', lib, '--mode', 'Neuronal', '--sweep_group', 'neuron_synapse', '--Nn', '12',
              '--c_max', '100', '--n_workers', '2', '--rc-out', out, '--rc-simtime', '0.3']
    tool = [sys.executable, os.path.join(HERE, 'range_check.py')]
    r = subprocess.run(tool + ['--rc-phase', 'oat', '--rc-conn', '0.3', '--rc-points', '2',
                               '--rc-topologies', '1'] + common, capture_output=True, text=True)
    check(r.returncode == 0, 'oat phase exit 0')
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        return
    pdir = os.path.join(out, 'oat_cp0.300')
    design = json.load(open(os.path.join(pdir, 'design.json')))
    sides = [json.load(open(os.path.join(dp, f))) for dp, _, fs in os.walk(pdir) for f in fs
             if f.startswith('iter_') and f.endswith('.json')]
    n_design = design['n_design']
    pos_ok = all(np.array_equal(np.asarray(s['theta']), np.asarray(
        design['points'][s['iter_idx']]['theta_vector'] if s['iter_idx'] < n_design else design['nominal_theta']))
        for s in sides)
    covered = sorted(s['iter_idx'] for s in sides if s['iter_idx'] < n_design)
    check(len(sides) == design['k'] * 2 and pos_ok and covered == list(range(n_design)),
          '%d runs; run i carries design vector i; all %d design points once' % (len(sides), n_design))
    r = subprocess.run(tool + ['--rc-phase', 'prior', '--rc-topologies', '1'] + common, capture_output=True, text=True)
    n_prior = sum(1 for dp, _, fs in os.walk(os.path.join(out, 'prior')) for f in fs
                  if f.startswith('iter_') and f.endswith('.json'))
    check(r.returncode == 0 and n_prior == 2, 'prior phase: %d runs' % n_prior)
    r = subprocess.run(tool + ['--rc-phase', 'prior', '--rc-topologies', '1'] + common, capture_output=True, text=True)
    check(r.returncode != 0 and 'already holds runs' in (r.stdout + r.stderr), 'refuses an occupied folder')
    a = subprocess.run([sys.executable, os.path.join(HERE, 'range_analysis.py'), out, '--t-skip', '0.05',
                        '--burst-lib', PIPE], capture_output=True, text=True)
    import pandas as pd
    ok = a.returncode == 0 and os.path.isfile(os.path.join(out, 'runs.csv'))
    df = pd.read_csv(os.path.join(out, 'runs.csv')) if ok else None
    check(ok and len(df) == len(sides) + 2 and not (df.axis == 'unmatched').any(),
          'analysis: %s runs, none unmatched' % (None if df is None else len(df)))
    if a.returncode:
        print(a.stdout[-2000:], a.stderr[-2000:])
    p = subprocess.run([sys.executable, os.path.join(HERE, 'range_plots.py'), out], capture_output=True, text=True)
    check(p.returncode == 0 and os.path.isfile(os.path.join(out, 'oat_curves.png')), 'plots written')
    unc = make_lib(os.path.join(tmp, 'lib_unclamped'))
    path = os.path.join(unc, 'ASD_fun_BD_cpp.py')
    open(path, 'w').write(open(path).read().replace(': volt (unless refractory)', ': volt'))
    r = subprocess.run(tool + ['--rc-phase', 'prior', '--lib_dir', unc, '--rc-out', os.path.join(tmp, 'o2'),
                               '--rc-design-only'], capture_output=True, text=True)
    check(r.returncode != 0 and 'REFUSED' in (r.stdout + r.stderr), 'refuses an unclamped model')


def r6(tmp):
    print('\n[R6] analysis on synthetic runs')
    sys.path.insert(0, HERE)
    import range_analysis as RA
    root = os.path.join(tmp, 'synth')
    pdir = os.path.join(root, 'oat_cp0.350')
    os.makedirs(os.path.join(pdir, 'topo_00000'))
    names = ['g_ampa', 'g_nmda']
    nominal = [0.0, 0.0]
    design = dict(phase='oat', conn_prob=0.35, free_axes=names, param_names=names, log_params=[0, 1],
                  lib_dir=PIPE, nominal_theta=nominal, n_design=1, n_padding=1,
                  points=[dict(axis='g_ampa', index=0, theta=1.0, natural=float(np.exp(1.0)),
                               theta_vector=[1.0, 0.0])])
    json.dump(design, open(os.path.join(pdir, 'design.json'), 'w'))
    json.dump(dict(Nn=50, simtime=20.0), open(os.path.join(pdir, 'job_args.json'), 'w'))
    rng = np.random.default_rng(3)
    trains = {0: (1.0, 5.0), 1: (0.0, 0.0)}           # iter -> (theta of g_ampa, rate Hz)
    for it, (th, rate) in trains.items():
        n = rng.poisson(rate * 20.0 * 50)
        t = np.sort(rng.uniform(0, 20.0, n)).astype(np.float32)
        i = rng.integers(0, 50, n).astype(np.int32)
        np.savez(os.path.join(pdir, 'topo_00000', 'iter_%05d.npz' % it), spk_N_t=t, spk_N_i=i)
        json.dump(dict(topo_idx=0, iter_idx=it, conn_prob=0.35, theta=[th, 0.0], params=[np.exp(th), 1.0],
                       param_names=names, npz_relpath='topo_00000/iter_%05d.npz' % it),
                  open(os.path.join(pdir, 'topo_00000', 'iter_%05d.json' % it), 'w'))
    df, _, _ = RA.build_table(root, workers=1, burst_lib=PIPE)
    live, dead = df[df.iter_idx == 0].iloc[0], df[df.iter_idx == 1].iloc[0]
    check(abs(live.mean_rate_hz - 5.0) < 0.25, 'Poisson 5 Hz -> %.3f Hz (tolerance 0.25 = 3.7 SD)' % live.mean_rate_hz)
    check(live.axis == 'g_ampa' and abs(live.value - np.exp(1.0)) < 1e-12 and dead.axis == 'nominal',
          'labels from design.json (g_ampa, nominal)')
    check(dead.frac_active == 0.0 and dead.mean_rate_hz == 0.0, 'silent run: rate 0, active fraction 0')


def r7():
    print('\n[R7] bytes')
    for f in sorted(os.listdir(HERE)):
        p = os.path.join(HERE, f)
        if os.path.isfile(p) and not f.endswith('.pyc'):
            b = open(p, 'rb').read()
            check(b.count(b'\r') == 0 and all(x < 128 for x in b), '%s: LF only, pure ASCII' % f)


STUB_PY = '''#!/bin/bash
printf '%s\\n' "$*" >> "$STUB_LOG"
exit 0
'''
STUB_CONDA = '''#!/bin/bash
if [ "$1" = "info" ]; then echo "$STUB_BASE"; fi
exit 0
'''


def r8(tmp):
    print('\n[R8] the PBS job script with stub module, conda and python')
    base = os.path.join(tmp, 'stubbase')
    os.makedirs(os.path.join(base, 'etc', 'profile.d'))
    open(os.path.join(base, 'etc', 'profile.d', 'conda.sh'), 'w').write(
        'conda() { return 0; }\nmodule() { return 0; }\n')
    stub = os.path.join(tmp, 'stubbin')
    os.makedirs(stub)
    for name, text in (('python', STUB_PY), ('conda', STUB_CONDA), ('module', '#!/bin/bash\nexit 0\n')):
        open(os.path.join(stub, name), 'w').write(text)
        os.chmod(os.path.join(stub, name), 0o755)
    lib = os.path.join(tmp, 'fake_setup')
    os.makedirs(os.path.join(lib, 'campaign_cadex_hhgap_v5', 'sweep_intel_task0000'))
    for f in ('HPC_main_sweep.py', 'ASD_fun_BD_cpp.py'):
        open(os.path.join(lib, f), 'w').write('# stub\n')
    open(os.path.join(lib, 'campaign_cadex_hhgap_v5', 'sweep_intel_task0000', 'job_args.json'), 'w').write('{}')

    def job(**env):
        log = os.path.join(tmp, 'stub_%d.log' % len(os.listdir(tmp)))
        e = dict(os.environ, PATH=stub + os.pathsep + os.environ['PATH'], STUB_LOG=log, STUB_BASE=base,
                 PBS_O_WORKDIR=HERE, PBS_NCPUS='48', LIB_DIR=lib, OUT=os.path.join(tmp, 'job_out'), **env)
        r = subprocess.run(['bash', os.path.join(HERE, 'submit_range_check.sh')], env=e,
                           capture_output=True, text=True)
        calls = open(log).read().splitlines() if os.path.exists(log) else []
        return r, [c for c in calls if not c.startswith('-c ')]

    r, calls = job(SETUP='lcdup')
    oat = [c for c in calls if '--rc-phase oat' in c]
    check(r.returncode == 0 and len(oat) == 3 and sum('--rc-phase prior' in c for c in calls) == 1
          and calls[-2].startswith('range_analysis.py') and calls[-1].startswith('range_plots.py'),
          'lcdup: 3 oat + 1 prior + analysis + plots (%d calls)' % len(calls))
    check(all('--freeze Cm,I_inj,VA=-50,DeltaA,VR=-55' in c and '--mode Neuronal' in c and '--n_workers 48' in c
              for c in oat) and ['--rc-conn 0.1 ' in oat[0], '--rc-conn 0.35 ' in oat[1],
                                 '--rc-conn 0.6 ' in oat[2]] == [True] * 3,
          'lcdup: settings, 48 workers, conn levels in order')
    r, calls = job(SETUP='giulia', EXTRA='--stoa_gate_lo 0.2 --stoa_gate_hi 1.0',
                   OAT_EXTRA='--stoa_gate_lo 0.6 --stoa_gate_hi 0.6')
    oat = [c for c in calls if '--rc-phase oat' in c]
    prior = [c for c in calls if '--rc-phase prior' in c]
    check(r.returncode == 0 and len(oat) == 3 and len(prior) == 1
          and all('campaign_cadex_hhgap_v5/sweep_intel_task0000/job_args.json' in c for c in oat + prior),
          'giulia: LIKE found in the hhgap_v5 campaign')
    check(all('--stoa_gate_lo 0.2 --stoa_gate_hi 1.0' in c for c in oat + prior)
          and all(c.endswith('--stoa_gate_lo 0.6 --stoa_gate_hi 0.6') for c in oat)
          and '0.6 --stoa_gate_hi 0.6' not in prior[0], 'giulia: EXTRA everywhere, OAT_EXTRA last in oat only')
    r, calls = job(SETUP='nonsense')
    check(r.returncode == 2 and not calls, 'unknown SETUP: exit 2, nothing run')


def main():
    tmp = tempfile.mkdtemp(prefix='range_smoke_')
    try:
        r1(tmp)
        r2(tmp)
        r3(tmp)
        r4(tmp)
        if '--no-sim' in sys.argv:
            skip('R5 (--no-sim)')
        else:
            r5(tmp)
        r6(tmp)
        r7()
        r8(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nRESULT: %d passed, %d failed%s' % (RESULTS['passed'], RESULTS['failed'],
          (', %d skipped' % RESULTS['skipped']) if RESULTS['skipped'] else ''))
    return 0 if RESULTS['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
