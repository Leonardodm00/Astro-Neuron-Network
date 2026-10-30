#!/usr/bin/env python3
"""
smoke_test_range_check.py -- checks of hpc/range_check/ (specs/SPEC.md Block 6).

  R1  oat design on the repository's 36-D root pipeline: every free axis with
      lo < hi gets K points, each point differs from the nominal theta on exactly
      one axis, the end points lie on the theta box, padding to n_workers x k
  R2  the same pipeline with the LindaColloDUP freeze patch and freeze list:
      13 free axes, the frozen ones absent, VA/VR held at the freeze values;
      a list with values off the nominal shows that NAME=VALUE is applied
  R3  --rc-like: parsing the rebuilt argument list reproduces every job_args value
  R4  37-D Giulia copy: the design builds for its synapse_astro group (22 axes)
  R5  end to end (needs Brian2 and a C++ compiler): oat phase on a tiny
      Neuronal network (Nn 12, 100 um, 0.3 s, 2 workers, K = 2, R = 1), every
      design point found once in the sidecars; prior phase 2 runs; the analysis
      labels every oat run; refuses an unclamped model and an occupied folder
  R6  analysis on synthetic runs: Poisson trains at 5 Hz give a mean rate within
      5 % (3 Poisson SD); a silent run gives active fraction 0 and is 'silent'
  R7  bytes: pure ASCII, LF only, in every file of this folder
  R8  submit_range_check.sh with module, conda and python replaced by stub
      shell functions, in an environment built from scratch (only PATH kept;
      see the comment above STUB_FUNCS); a probe first confirms the stubs
      are the functions the job's shell sees, else R8 stops. For SETUP=lcdup
      it calls range_check.py 3 times (oat at 0.01 0.255 0.5) and once (prior),
      then the analysis and the plots, with the LindaColloDUP settings;
      SETUP=giulia finds LIKE in the hhgap_v5 campaign, uses conn_prob 0.1
      0.35 0.6, passes EXTRA to every phase and OAT_EXTRA to the oat phases
      only; FREEZE_ENC is decoded; PHASES=oat / prior / none and ANALYZE
      select what runs; CHECK_ONLY=1 runs nothing; an lcdup folder without
      the g box patch, a bad PHASES and an unknown SETUP exit 2
  R9  launch_range_check.sh with a stub qsub, in the same sealed environment
      (probed first like R8, so the real qsub is never reached and no job is
      ever submitted): three jobs (oat on intel, prior x16 on cpu, the
      analysis on cpu with 8 cores held by depend=afterany on the first two),
      one OUT and the pass-through variables in all three; DRYRUN=1, a failed
      precheck and a comma in a passed variable submit nothing; a failed
      second submission stops there and names the job to qdel

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
LCDUP_FREEZE = 'Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G'
OFF_NOMINAL_FREEZE = 'Cm,I_inj,VA=-50,DeltaA,VR=-55,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G'
BOX_MARK = '[LindaColloDUP box 2026-10-30]'
RESULTS = {'passed': 0, 'failed': 0, 'skipped': 0}


def check(cond, label):
    RESULTS['passed' if cond else 'failed'] += 1
    print('  %s %s' % ('ok  ' if cond else 'FAIL', label))
    return cond


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
        check(abs(d['va'] + 48) < 1e-12 and abs(d['vr'] + 58.2) < 1e-12, 'VA, VR held at -48, -58.2 mV')
        check(d['n_pts'] == 91 and d['k'] == 2 and d['n_pad'] == 5, '91 points, k=2, 5 nominal')
    # -48 / -58.2 are also the registry nominals: a list off the nominal shows
    # that NAME=VALUE really reaches the design
    d2 = in_subprocess(os.path.join(tmp, 'lib36f'), DESIGN_BODY % (
        ['--out_dir', 'x', '--mode', 'Neuronal', '--sweep_group', 'neuron_synapse', '--freeze', OFF_NOMINAL_FREEZE],))
    check(d2 is not None and abs(d2['va'] + 50) < 1e-12 and abs(d2['vr'] + 55) < 1e-12,
          'off-nominal list: VA, VR held at -50, -55 mV')


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


# R8 runs the job script with module, conda and python replaced by shell
# functions, exported by a wrapper shell, in an environment built from scratch:
# of the caller's environment only PATH is kept. Inherited, a login shell's own
# module function (exported, or defined again in every child shell through
# BASH_ENV) replaced the stubs, so the real module, conda and python ran and
# the stubs logged no call (davinci, 2026-10-30). Functions also need no exec
# permission, so a noexec /tmp cannot stop them either.
STUB_FUNCS = r'''python() { printf '%s\n' "$*" >> "$STUB_LOG"; return 0; }
conda() { if [ "${1:-}" = info ]; then printf '%s\n' "$STUB_BASE"; fi; return 0; }
module() { return 0; }
export -f python conda module
exec "$BASH" "$1"
'''
BASH = shutil.which('bash') or '/bin/bash'


def stubs_loaded(funcs, names, env):
    """Whether every command in `names` is a shell function in the shell that
    will run the script under test (same wrapper, same environment). R8 and R9
    stop when it is not: a stub that failed to load would leave the real
    python running range_check.py, or the real qsub submitting jobs."""
    probe = os.path.join(env['HOME'], 'stub_probe.sh')
    with open(probe, 'w') as f:
        f.write('for n in %s; do printf "%%s=%%s\\n" "$n" "$(type -t "$n")"; done\n' % ' '.join(names))
    r = subprocess.run([BASH, '-c', funcs, 'probe', probe], env=env, capture_output=True, text=True)
    got = dict(line.split('=', 1) for line in r.stdout.split() if '=' in line)
    ok = check(all(got.get(n) == 'function' for n in names),
               '%s: the stub shell function%s, in the tested shell'
               % (', '.join(names), 's' if len(names) > 1 else ''))
    if not ok:
        print('    resolved as %s; the remaining checks of this section were not run' % (got or r.stderr.strip()))
    return ok


def make_fake_setup(path, box=True):
    """A stand-in setup folder for the job script: stub pipeline files, the
    LindaColloDUP box markers when box=True, and a Giulia-style campaign task."""
    os.makedirs(os.path.join(path, 'campaign_cadex_hhgap_v5', 'sweep_intel_task0000'))
    marks = ('    (0.01,   10.0),     # 14 g_ampa  %s\n    (0.01,   10.0),     # 15 g_nmda  %s\n'
             % (BOX_MARK, BOX_MARK)) if box else ''
    open(os.path.join(path, 'HPC_main_sweep.py'), 'w').write('# stub\n' + marks)
    open(os.path.join(path, 'ASD_fun_BD_cpp.py'), 'w').write('# stub\n')
    open(os.path.join(path, 'campaign_cadex_hhgap_v5', 'sweep_intel_task0000', 'job_args.json'), 'w').write('{}')
    return path


def r8(tmp):
    print('\n[R8] the PBS job script with stub module, conda and python')
    base = os.path.join(tmp, 'stubbase')
    os.makedirs(os.path.join(base, 'etc', 'profile.d'))
    open(os.path.join(base, 'etc', 'profile.d', 'conda.sh'), 'w').write(
        'conda() { return 0; }\nmodule() { return 0; }\n')
    home = os.path.join(tmp, 'r8_home')
    os.makedirs(home)
    lib = make_fake_setup(os.path.join(tmp, 'fake_setup'))
    nobox = make_fake_setup(os.path.join(tmp, 'fake_setup_nobox'), box=False)
    n_jobs = [0]
    if not stubs_loaded(STUB_FUNCS, ['python', 'conda', 'module'],
                        dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=home, LC_ALL='C',
                             STUB_LOG=os.path.join(tmp, 'r8_probe.log'), STUB_BASE=base)):
        return

    def job(**env):
        n_jobs[0] += 1
        log = os.path.join(tmp, 'r8_calls_%d.log' % n_jobs[0])
        e = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=home, LC_ALL='C',
                 STUB_LOG=log, STUB_BASE=base, PBS_O_WORKDIR=HERE, PBS_NCPUS='48',
                 LIB_DIR=lib, OUT=os.path.join(tmp, 'job_out_%d' % n_jobs[0]))
        e.update(env)
        r = subprocess.run([BASH, '-c', STUB_FUNCS, 'r8', os.path.join(HERE, 'submit_range_check.sh')],
                           env=e, capture_output=True, text=True)
        raw = open(log).read().splitlines() if os.path.exists(log) else []
        return r, [c for c in raw if not c.startswith('-c ')], raw

    def explain(ok, r, calls):
        """After a failed check, print what the job did, so a pasted run shows the cause."""
        if ok:
            return
        print('    job exit %d, %d stub calls; its last output lines:' % (r.returncode, len(calls)))
        for tag, text in (('out', r.stdout), ('err', r.stderr)):
            for line in text.strip().splitlines()[-8:]:
                print('    %s| %s' % (tag, line))

    def phase_calls(calls):
        return ([c for c in calls if '--rc-phase oat' in c], [c for c in calls if '--rc-phase prior' in c])

    r, calls, _ = job(SETUP='lcdup')
    oat, prior = phase_calls(calls)
    ok = check(r.returncode == 0 and len(oat) == 3 and len(prior) == 1
               and len(calls) >= 2 and calls[-2].startswith('range_analysis.py')
               and calls[-1].startswith('range_plots.py'),
               'lcdup: 3 oat + 1 prior + analysis + plots (%d calls)' % len(calls))
    explain(ok, r, calls)
    check(len(oat) == 3 and len(prior) == 1
          and all('--freeze Cm,I_inj,VA=-48,DeltaA,VR=-58.2' in c and '--mode Neuronal' in c
                  and '--n_workers 48' in c and '--conn_prob_lo 0.01 --conn_prob_hi 0.5' in c for c in oat + prior)
          and ['--rc-conn 0.01 ' in oat[0], '--rc-conn 0.255 ' in oat[1], '--rc-conn 0.5 ' in oat[2]] == [True] * 3,
          'lcdup: freeze VA=-48 VR=-58.2, conn_prob box 0.01-0.5, 48 workers, oat at 0.01 0.255 0.5')
    r, calls, _ = job(SETUP='giulia', EXTRA='--stoa_gate_lo 0.2 --stoa_gate_hi 1.0',
                      OAT_EXTRA='--stoa_gate_lo 0.6 --stoa_gate_hi 0.6')
    oat, prior = phase_calls(calls)
    ok = check(r.returncode == 0 and len(oat) == 3 and len(prior) == 1
               and all('campaign_cadex_hhgap_v5/sweep_intel_task0000/job_args.json' in c for c in oat + prior),
               'giulia: LIKE found in the hhgap_v5 campaign')
    explain(ok, r, calls)
    check(len(oat) == 3
          and ['--rc-conn 0.1 ' in oat[0], '--rc-conn 0.35 ' in oat[1], '--rc-conn 0.6 ' in oat[2]] == [True] * 3,
          'giulia: oat at conn_prob 0.1 0.35 0.6')
    check(len(oat) == 3 and len(prior) == 1
          and all('--stoa_gate_lo 0.2 --stoa_gate_hi 1.0' in c for c in oat + prior)
          and all(c.endswith('--stoa_gate_lo 0.6 --stoa_gate_hi 0.6') for c in oat)
          and '0.6 --stoa_gate_hi 0.6' not in prior[0], 'giulia: EXTRA everywhere, OAT_EXTRA last in oat only')
    r, calls, _ = job(SETUP='lcdup', FREEZE_ENC='Cm:I_inj:VA@-50:DeltaA:VR@-55:EC50_ampa:EC50_nmda:alpha_syn:O_G:Omega_G')
    phases = [c for c in calls if '--rc-phase' in c]
    ok = check(r.returncode == 0 and len(phases) == 4
               and all('--freeze Cm,I_inj,VA=-50,DeltaA,VR=-55,EC50_ampa' in c for c in phases),
               'lcdup: FREEZE_ENC decoded (a list that differs from the default)')
    explain(ok, r, calls)
    r, calls, _ = job(SETUP='lcdup', PHASES='oat', ANALYZE='0')
    oat, prior = phase_calls(calls)
    ok = check(r.returncode == 0 and len(oat) == 3 and len(calls) == 3, 'PHASES=oat ANALYZE=0: the 3 oat calls only')
    explain(ok, r, calls)
    r, calls, _ = job(SETUP='lcdup', PHASES='prior', ANALYZE='0', PRIOR_TOPOS='16')
    ok = check(r.returncode == 0 and len(calls) == 1 and '--rc-phase prior --rc-topologies 16 ' in calls[0],
               'PHASES=prior ANALYZE=0 PRIOR_TOPOS=16: one prior call with 16 topologies')
    explain(ok, r, calls)
    out_a = os.path.join(tmp, 'analysis_out')
    os.makedirs(os.path.join(out_a, 'prior'))
    open(os.path.join(out_a, 'prior', 'design.json'), 'w').write('{}')
    r, calls, _ = job(SETUP='lcdup', PHASES='none', ANALYZE='1', OUT=out_a, PBS_NCPUS='8')
    ok = check(r.returncode == 0 and len(calls) == 2 and calls[0].startswith('range_analysis.py %s --workers 8' % out_a)
               and calls[1].startswith('range_plots.py'), 'PHASES=none ANALYZE=1: analysis and plots only')
    explain(ok, r, calls)
    r, calls, raw = job(SETUP='lcdup', PHASES='none', ANALYZE='1', OUT=os.path.join(tmp, 'empty_out'))
    ok = check(r.returncode == 2 and not raw, 'PHASES=none on an OUT without results: exit 2, nothing run')
    explain(ok, r, calls)
    r, calls, raw = job(SETUP='lcdup', CHECK_ONLY='1')
    ok = check(r.returncode == 0 and not raw and 'precheck OK' in r.stdout, 'CHECK_ONLY=1: exit 0, no python at all')
    explain(ok, r, calls)
    r, calls, raw = job(SETUP='lcdup', LIB_DIR=nobox)
    ok = check(r.returncode == 2 and not raw and 'box patch' in r.stderr,
               'lcdup folder without the g box patch: exit 2, nothing run')
    explain(ok, r, calls)
    r, calls, raw = job(SETUP='lcdup', PHASES='everything')
    ok = check(r.returncode == 2 and not raw, 'bad PHASES: exit 2, nothing run')
    explain(ok, r, calls)
    r, calls, raw = job(SETUP='nonsense')
    ok = check(r.returncode == 2 and not raw, 'unknown SETUP: exit 2, nothing run')
    explain(ok, r, calls)


# qsub stand-in for R9, a shell function like the stubs of R8: it logs each
# call on its own line and answers "10<n>.pbsserver"; call number QSUB_FAIL_AT
# fails without output, as a refused submission does.
QSUB_FUNCS = r'''qsub() {
  local n=$(( $(wc -l < "$QSUB_LOG") + 1 ))
  printf '%s\n' "$*" >> "$QSUB_LOG"
  if [ "$n" = "${QSUB_FAIL_AT:-0}" ]; then echo "qsub: refused" >&2; return 1; fi
  echo "10${n}.pbsserver"
}
export -f qsub
exec "$BASH" "$1"
'''


def r9(tmp):
    print('\n[R9] launch_range_check.sh with a stub qsub')
    home = os.path.join(tmp, 'r9_home')
    os.makedirs(home)
    lib = make_fake_setup(os.path.join(tmp, 'r9_setup'))
    nobox = make_fake_setup(os.path.join(tmp, 'r9_setup_nobox'), box=False)
    out = os.path.join(tmp, 'r9_out')
    n_runs = [0]
    probe_log = os.path.join(tmp, 'r9_probe.log')
    open(probe_log, 'w').close()
    if not stubs_loaded(QSUB_FUNCS, ['qsub'], dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'),
                                                  HOME=home, LC_ALL='C', QSUB_LOG=probe_log)):
        return

    def launch(**env):
        n_runs[0] += 1
        log = os.path.join(tmp, 'r9_qsub_%d.log' % n_runs[0])
        open(log, 'w').close()
        e = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=home, LC_ALL='C',
                 QSUB_LOG=log, SETUP='lcdup', LIB_DIR=lib, OUT=out)
        e.update(env)
        r = subprocess.run([BASH, '-c', QSUB_FUNCS, 'r9', os.path.join(HERE, 'launch_range_check.sh')],
                           env=e, capture_output=True, text=True)
        return r, open(log).read().splitlines()

    def explain(ok, r, calls):
        if ok:
            return
        print('    launcher exit %d, %d qsub calls; its last output lines:' % (r.returncode, len(calls)))
        for tag, text in (('out', r.stdout), ('err', r.stderr)):
            for line in text.strip().splitlines()[-8:]:
                print('    %s| %s' % (tag, line))

    r, calls = launch()
    common = 'SETUP=lcdup,OUT=%s,LIB_DIR=%s' % (out, lib)
    ok = check(r.returncode == 0 and len(calls) == 3
               and '-q intel -N rc_oat -l select=1:ncpus=48 -l walltime=12:00:00' in calls[0]
               and (common + ',PHASES=oat,ANALYZE=0') in calls[0]
               and '-q cpu -N rc_prior -l select=1:ncpus=48' in calls[1]
               and (common + ',PHASES=prior,ANALYZE=0,PRIOR_TOPOS=16') in calls[1]
               and calls[0].endswith('submit_range_check.sh') and calls[1].endswith('submit_range_check.sh'),
               'oat job on intel and prior job (16 topologies) on cpu, 48 cores each, one OUT')
    explain(ok, r, calls)
    ok = check(len(calls) == 3 and '-q cpu -N rc_analysis -l select=1:ncpus=8 -l walltime=02:00:00' in calls[2]
               and '-W depend=afterany:101.pbsserver:102.pbsserver' in calls[2]
               and (common + ',PHASES=none,ANALYZE=1') in calls[2],
               'analysis job on cpu, 8 cores, held by depend=afterany on both, same OUT')
    explain(ok, r, calls)
    r, calls = launch(DRYRUN='1')
    ok = check(r.returncode == 0 and not calls and r.stdout.count('[dry run] qsub') == 3,
               'DRYRUN=1: three qsub lines printed, nothing submitted')
    explain(ok, r, calls)
    r, calls = launch(LIB_DIR=nobox)
    r2, calls2 = launch(EXTRA='--a 1,--b 2')
    ok = check(r.returncode == 2 and not calls and r2.returncode == 2 and not calls2,
               'failed precheck, or a comma in a passed variable: exit 2, nothing submitted')
    explain(ok, r, calls)
    r, calls = launch(QSUB_FAIL_AT='2')
    ok = check(r.returncode == 3 and len(calls) == 2 and 'qdel 101.pbsserver' in r.stderr,
               'second submission refused: exit 3, no third job, the first job named for qdel')
    explain(ok, r, calls)


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
        r9(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nRESULT: %d passed, %d failed%s' % (RESULTS['passed'], RESULTS['failed'],
          (', %d skipped' % RESULTS['skipped']) if RESULTS['skipped'] else ''))
    return 0 if RESULTS['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
