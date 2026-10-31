#!/usr/bin/env python3
"""
smoke_test_gg.py -- checks of the GiuliaGate package (specs/SPEC.md Block 9) on
a fresh copy of this folder, with the pipeline copied in from the clone's root
folder and patched as APPLY_GG.txt steps 0-4 do it. Touches nothing in this
folder; no cluster, no qsub; no simulation unless --sim.

  G1 prior_config.tsv: apply_prior_patch.py --self-test PASS; the TSV lists the
     37 axes of the registry plus conn_prob and stoa_gate_p; 15 free axes, each
     with 0 < low < high; its scale column follows the rule of
     HPC_main_sweep.py (log iff low > 0 and high/low >= 10)
  G2 patches on the copy: apply_freeze_patch.py PATCHED; apply_prior_patch.py
     --check CHECK OK, then PATCHED with Sigma, g_ampa and g_nmda (the boxes
     that differ from the registry; O_N's [0.0001, 1] is the registry box
     since 2026-10-31); a second run ALREADY APPLIED; --check-applied BOXES
     OK; the patched file compiles, 3 rows marked
  G3 the patched module: N_DIMS 37; tripartite minus the freeze list = the 15
     free axes of the TSV; their PARAM_BOUNDS = the TSV boxes; LOG_PARAMS
     follows the rule (Sigma, I_bias, U_A linear, the other 12 log)
  G4 refusals: a TSV edited after patching (Sigma [3, 9]) fails
     --check-applied and the patcher refuses the file; a fixed DeltaT off the
     registry is refused
  G5 launch_GG.sh dry run on the copy (SEED_SCAN_TREE a temporary folder): the
     banner shows Nn=108, Na=108, conn_prob U[0.05, 0.4], the gate U[0.2, 1],
     the freeze list and "V clamp (t_ref): yes"; two "[dry run] qsub" lines
     (intel, cpu) carry FREEZE_ENC, the boxes, the densities, SIMTIME=180 and
     N_TOPOLOGIES=48; refuses without SIZING_CONFIRMED, with
     REFRACTORY_CLAMP=no, and on an unpatched copy
  G6 submit_GG.sh under GG_NO_ENV=1 with a stub python first on PATH (it logs
     its arguments and hands apply_prior_patch.py to python3): --check-applied
     runs first, then HPC_main_sweep.py with --mode Full --sweep_group
     tripartite, the decoded freeze list, the injected boxes and densities and
     --seed_master SEED_BASE + index; an unpatched copy exits 2 before
     HPC_main_sweep.py; a missing NODETAG or FREEZE_ENC exits 2
  G7 bytes: every file of the package pure ASCII, LF only
  G8 (--sim; Brian2 and a C++ compiler, 2-4 min) the real worker on a tiny
     Full network (APPLY_GG.txt step 7: 2 runs of 2 s on a 120 um well):
     "topo_00000 done: 2 ok", the gate p in its box in every sidecar, and no
     NaN warning for the neurons or the astrocytes (before Block 10, the
     summed astrocyte drive gave 4 astrocyte warnings here)

Run:  python3 smoke_test_gg.py | tail -1          (about 30 s)
      python3 smoke_test_gg.py --sim | tail -1    (also G8)
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
ROOT = os.path.abspath(os.path.join(HERE, '..'))
PIPELINE = ('HPC_main_sweep.py', 'HPC_single_run.py', 'ASD_fun_BD_cpp.py', 'aggregate_sweep.py',
            'burst_metrics.py', 'burst_plots.py', 'synapse_pdist.csv')
PACKAGE = ('launch_GG.sh', 'submit_GG.sh', 'apply_prior_patch.py', 'apply_freeze_patch.py', 'seed_alloc.py',
           'prior_config.tsv', 'smoke_test_gg.py', 'APPLY_GG.txt', '.gitignore')
# O_N [0.0001, 1] equals the registry box since 2026-10-31 (specs/SPEC.md Block 10), so it is not patched
CHANGED = {'Sigma': (2.0, 10.0), 'g_ampa': (0.05, 10.0), 'g_nmda': (0.05, 10.0)}
LINEAR = {'Sigma', 'I_bias', 'U_A'}
RESULTS = {'passed': 0, 'failed': 0, 'skipped': 0}


def check(cond, label):
    RESULTS['passed' if cond else 'failed'] += 1
    print('  %s %s' % ('ok  ' if cond else 'FAIL', label))
    return cond


def skip(label):
    RESULTS['skipped'] += 1
    print('  SKIP %s' % label)


def run(cmd, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)


def make_copy(dst, patch=True):
    """The package plus the pipeline copied from the root folder (step 0), patched (steps 3-4)."""
    os.makedirs(dst)
    for f in PACKAGE:
        if os.path.exists(os.path.join(HERE, f)):
            shutil.copy2(os.path.join(HERE, f), dst)
    for f in PIPELINE:
        shutil.copy2(os.path.join(ROOT, f), dst)
    if patch:
        for tool in ('apply_freeze_patch.py', 'apply_prior_patch.py'):
            r = run([sys.executable, tool], cwd=dst)
            assert r.returncode == 0, r.stdout + r.stderr
    return dst


def read_tsv(path):
    rows = [ln.split('\t') for ln in open(path).read().splitlines() if ln and not ln.startswith('#')]
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:]]


def module_facts(lib, freeze):
    code = ('import sys, json\nsys.path.insert(0, %r)\nimport HPC_main_sweep as S\n'
            'args = S.build_parser().parse_args(["--out_dir", "x", "--mode", "Full", "--sweep_group", "tripartite",'
            ' "--freeze", %r])\n'
            'act = [int(i) for i in S._apply_freeze(args.freeze, [int(i) for i in S.resolve_sweep_group("tripartite")])]\n'
            'print(json.dumps(dict(n=int(S.N_DIMS), active=[S.PARAM_NAMES[j] for j in act],'
            ' bounds={S.PARAM_NAMES[j]: [float(x) for x in S.PARAM_BOUNDS[j]] for j in act},'
            ' log=[S.PARAM_NAMES[j] for j in sorted(S.LOG_PARAMS)])))\n' % (lib, freeze))
    r = run([sys.executable, '-c', code], cwd=lib)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-2500:])
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def g1(tmp):
    print('\n[G1] prior_config.tsv')
    r = run([sys.executable, os.path.join(HERE, 'apply_prior_patch.py'), '--self-test'])
    check(r.returncode == 0 and 'SELF-TEST: PASS' in r.stdout, 'apply_prior_patch.py --self-test: PASS')
    rows = read_tsv(os.path.join(HERE, 'prior_config.tsv'))
    axes = [x['axis'] for x in rows]
    code = ('import sys, json\nsys.path.insert(0, %r)\nimport HPC_main_sweep as S\n'
            'print(json.dumps(list(S.PARAM_NAMES)))\n' % ROOT)
    names = json.loads(run([sys.executable, '-c', code], cwd=ROOT).stdout.strip().splitlines()[-1])
    check(sorted(axes) == sorted(list(names) + ['conn_prob', 'stoa_gate_p']) and len(axes) == 39,
          'the 37 registry axes plus conn_prob and stoa_gate_p, once each')
    free = [x for x in rows if x['status'] == 'free' and x['group'] != 'topology']
    check(len(free) == 15 and all(0 < float(x['low']) < float(x['high']) for x in free),
          '15 free axes, each with 0 < low < high')
    rule = all(x['scale'] == ('log' if float(x['high']) / float(x['low']) >= 10 else 'lin') for x in free)
    check(rule and {x['axis'] for x in free if x['scale'] == 'lin'} == LINEAR,
          'scale column follows the rule: Sigma, I_bias, U_A linear, the other 12 log')
    return {x['axis']: (float(x['low']), float(x['high'])) for x in free}


def g2(tmp):
    print('\n[G2] patches on a fresh copy')
    lib = make_copy(os.path.join(tmp, 'g2'), patch=False)
    r1 = run([sys.executable, 'apply_freeze_patch.py'], cwd=lib)
    r2 = run([sys.executable, 'apply_prior_patch.py', '--check'], cwd=lib)
    r3 = run([sys.executable, 'apply_prior_patch.py'], cwd=lib)
    r4 = run([sys.executable, 'apply_prior_patch.py'], cwd=lib)
    r5 = run([sys.executable, 'apply_prior_patch.py', '--check-applied'], cwd=lib)
    check(r1.returncode == 0 and 'PATCHED' in r1.stdout, 'apply_freeze_patch.py: PATCHED')
    check(r2.returncode == 0 and 'CHECK OK' in r2.stdout, 'apply_prior_patch.py --check: CHECK OK')
    check(r3.returncode == 0 and 'PATCHED' in r3.stdout and all(n in r3.stdout for n in CHANGED),
          'apply_prior_patch.py: PATCHED, with %s' % ', '.join(CHANGED))
    check(r4.returncode == 0 and 'ALREADY APPLIED' in r4.stdout, 'a second run: ALREADY APPLIED, nothing written')
    check(r5.returncode == 0 and 'BOXES OK: 15 free axes' in r5.stdout, '--check-applied: BOXES OK: 15 free axes')
    r6 = run([sys.executable, '-m', 'py_compile', 'HPC_main_sweep.py'], cwd=lib)
    n_mark = open(os.path.join(lib, 'HPC_main_sweep.py')).read().count('[GiuliaGate box 2026-10-31]')
    check(r6.returncode == 0 and n_mark == len(CHANGED),
          'the patched file compiles; %d rows carry the GiuliaGate mark' % len(CHANGED))
    return lib


def g3(lib, free_tsv):
    print('\n[G3] the patched module')
    freeze = run([sys.executable, 'apply_prior_patch.py', '--freeze-list'], cwd=lib).stdout.strip()
    d = module_facts(lib, freeze)
    if not check(d is not None, 'module imported, freeze list parsed by _apply_freeze'):
        return
    check(d['n'] == 37 and sorted(d['active']) == sorted(free_tsv),
          'N_DIMS 37; tripartite minus the freeze list = the 15 free axes of the TSV')
    check(all(abs(d['bounds'][n][0] - lo) < 1e-12 and abs(d['bounds'][n][1] - hi) < 1e-12
              for n, (lo, hi) in free_tsv.items()), 'PARAM_BOUNDS of the free axes = the TSV boxes')
    act_log = set(d['active']) & set(d['log'])
    check(act_log == set(free_tsv) - LINEAR, 'LOG_PARAMS: Sigma, I_bias, U_A linear, the other 12 log')


def g4(tmp):
    print('\n[G4] refusals')
    lib = make_copy(os.path.join(tmp, 'g4'))
    tsv = os.path.join(lib, 'prior_config.tsv')
    text = open(tsv).read()
    edited = re.sub(r'^(Sigma\tneuron\t[^\t]*\tfree\t)2\t10\t', r'\g<1>3\t9\t', text, flags=re.M)
    open(tsv, 'w').write(edited)
    a = run([sys.executable, 'apply_prior_patch.py', '--check-applied'], cwd=lib)
    b = run([sys.executable, 'apply_prior_patch.py'], cwd=lib)
    check(edited != text and a.returncode == 1 and 'NOT APPLIED' in a.stdout and b.returncode == 1
          and 'REFUSED' in b.stdout, 'TSV edited after patching (Sigma [3, 9]): NOT APPLIED; the patcher refuses')
    open(tsv, 'w').write(re.sub(r'^(DeltaT\t[^\t]*\t[^\t]*\tfixed\t-\t-\t)2\t', r'\g<1>2.5\t', text, flags=re.M))
    c = run([sys.executable, 'apply_prior_patch.py', '--freeze-list'], cwd=lib)
    check(c.returncode == 1 and 'REFUSED' in c.stdout, 'DeltaT fixed off the registry (2.5): refused')


def stub_dir(tmp):
    d = os.path.join(tmp, 'stubbin')
    os.makedirs(d)
    py = os.path.join(d, 'python')
    open(py, 'w').write('#!/bin/sh\nprintf \'%s\\n\' "$*" >> "$STUB_LOG"\n'
                        'case "$1" in */apply_prior_patch.py) exec python3 "$@" ;; esac\nexit 0\n')
    os.chmod(py, 0o755)
    return d


def g5(tmp):
    print('\n[G5] launch_GG.sh dry run')
    lib = make_copy(os.path.join(tmp, 'g5', 'GiuliaGate'))
    raw = make_copy(os.path.join(tmp, 'g5', 'GiuliaGate_unpatched'), patch=False)
    run([sys.executable, 'apply_freeze_patch.py'], cwd=raw)
    scan = os.path.join(tmp, 'g5')
    base = dict(os.environ, SEED_SCAN_TREE=scan, DRYRUN='1')
    r = run(['bash', os.path.join(lib, 'launch_GG.sh')], env=dict(base, SIZING_CONFIRMED='1', REFRACTORY_CLAMP='yes'))
    out = r.stdout
    q = [ln for ln in out.splitlines() if '[dry run] qsub' in ln]
    v = re.findall(r'-v "([^"]+)"', out)
    freeze = run([sys.executable, 'apply_prior_patch.py', '--freeze-list'], cwd=lib).stdout.strip()
    enc = freeze.replace(',', ':').replace('=', '@')
    ok = check(r.returncode == 0 and 'Nn=108, Na=108' in out and 'conn_prob      : U[0.05, 0.4]' in out
               and 'contact gate   : U[0.2, 1]' in out and ('freeze         : ' + freeze) in out
               and 'V clamp (t_ref): yes' in out and 'N_TOPOLOGIES=48' in out,
               'banner: Nn=108 Na=108, conn_prob U[0.05, 0.4], gate U[0.2, 1], the freeze list, clamp yes, 48 topologies')
    if not ok:
        print(out[-2000:], r.stderr[-1500:])
    need = ['FREEZE_ENC=' + enc, 'CONN_PROB_LO=0.05', 'CONN_PROB_HI=0.4', 'STOA_GATE_LO=0.2', 'STOA_GATE_HI=1',
            'DENSITY=1200', 'DENSITY_ASTRO=1200', 'SIMTIME=180', 'N_TOPOLOGIES=48', 'MODE=Full',
            'SWEEP_GROUP=tripartite']
    check(len(q) == 2 and '-q intel' in q[0] and '-q cpu' in q[1] and len(v) == 2
          and all(all(n in x.split(',') for n in need) for x in v),
          'two qsub lines (intel, cpu) carrying the freeze list, boxes, densities, SIMTIME=180, N_TOPOLOGIES=48')
    r1 = run(['bash', os.path.join(lib, 'launch_GG.sh')], env=dict(base, REFRACTORY_CLAMP='yes'))
    r2 = run(['bash', os.path.join(lib, 'launch_GG.sh')], env=dict(base, SIZING_CONFIRMED='1', REFRACTORY_CLAMP='no'))
    r3 = run(['bash', os.path.join(raw, 'launch_GG.sh')], env=dict(base, SIZING_CONFIRMED='1', REFRACTORY_CLAMP='yes'))
    check(r1.returncode != 0 and r2.returncode != 0 and r3.returncode == 2 and 'boxes of prior_config.tsv' in r3.stderr,
          'refuses without SIZING_CONFIRMED, with REFRACTORY_CLAMP=no, and on an unpatched copy')


def g6(tmp):
    print('\n[G6] submit_GG.sh with a stub python')
    lib = make_copy(os.path.join(tmp, 'g6'))
    raw = make_copy(os.path.join(tmp, 'g6_unpatched'), patch=False)
    run([sys.executable, 'apply_freeze_patch.py'], cwd=raw)
    sb = stub_dir(tmp)
    freeze = run([sys.executable, 'apply_prior_patch.py', '--freeze-list'], cwd=lib).stdout.strip()
    enc = freeze.replace(',', ':').replace('=', '@')

    def job(folder, n, **kw):
        log = os.path.join(tmp, 'g6_%d.log' % n)
        open(log, 'w').close()
        env = dict(os.environ, PATH=sb + os.pathsep + os.environ.get('PATH', ''), STUB_LOG=log, GG_NO_ENV='1',
                   REPO_DIR=folder, NODETAG='intel', FREEZE_ENC=enc, SEED_BASE='5000', PBS_ARRAY_INDEX='3',
                   PBS_NCPUS='2', CONN_PROB_LO='0.05', CONN_PROB_HI='0.4', STOA_GATE_LO='0.2', STOA_GATE_HI='1',
                   CAMPAIGN_TAG='smoke')
        env.update(kw)
        env = {k: v for k, v in env.items() if v is not None}
        r = run(['bash', os.path.join(folder, 'submit_GG.sh')], env=env)
        return r, open(log).read().splitlines()

    r, calls = job(lib, 1)
    main = [c for c in calls if c.split(' ', 1)[0].endswith('HPC_main_sweep.py')
            and '--rebuild_manifest_only' not in c]
    ok = check(r.returncode == 0 and len(calls) >= 3 and '--check-applied' in calls[0] and len(main) == 1,
               '--check-applied first, then one HPC_main_sweep.py call (%d calls)' % len(calls))
    if not ok:
        print(r.stdout[-1500:], r.stderr[-1500:], calls)
    m = main[0] if main else ''
    check(('--mode Full' in m and '--sweep_group tripartite' in m and ('--freeze ' + freeze + ' ') in m
           and '--conn_prob_lo 0.05 --conn_prob_hi 0.4' in m and '--stoa_gate_lo 0.2 --stoa_gate_hi 1' in m
           and '--density 1200 --density_astro 1200' in m and '--simtime 180' in m and '--n_topologies 48' in m
           and '--seed_master 5003' in m and '--seed_stoa_gate 71' in m),
          'Full, tripartite, the decoded freeze list, the boxes, 1200/1200, 180 s, 48 topologies, seed 5000 + 3')
    r2, calls2 = job(raw, 2)
    check(r2.returncode == 2 and not any(c.split(' ', 1)[0].endswith('HPC_main_sweep.py') for c in calls2),
          'an unpatched copy exits 2 before HPC_main_sweep.py')
    r3, _ = job(lib, 3, NODETAG=None)
    r4, _ = job(lib, 4, FREEZE_ENC=None)
    check(r3.returncode == 2 and r4.returncode == 2, 'no NODETAG, no FREEZE_ENC: exit 2')


def g7():
    print('\n[G7] bytes')
    for f in PACKAGE:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            b = open(p, 'rb').read()
            check(b.count(b'\r') == 0 and all(x < 128 for x in b), '%s: LF only, pure ASCII' % f)
        else:
            check(False, '%s present' % f)


def g8(tmp):
    print('\n[G8] the real worker on a tiny Full network')
    import importlib.util
    if importlib.util.find_spec('brian2') is None:
        return skip('brian2 not importable')
    lib = make_copy(os.path.join(tmp, 'g8'))
    freeze = run([sys.executable, 'apply_prior_patch.py', '--freeze-list'], cwd=lib).stdout.strip()
    env = dict(os.environ, GG_NO_ENV='1', NODETAG='probe', CAMPAIGN_TAG='probe_check', SEED_BASE='1000',
               N_TOPOLOGIES='1', C_MAX='120', SIMTIME='2', PBS_NCPUS='2', REPO_DIR=lib,
               FREEZE_ENC=freeze.replace(',', ':').replace('=', '@'))
    r = run(['bash', os.path.join(lib, 'submit_GG.sh')], env=env)
    out = r.stdout + r.stderr
    check(r.returncode == 0 and 'topo_00000 done: 2 ok' in out, 'exit 0; "topo_00000 done: 2 ok"')
    sides = [json.load(open(p)) for p in glob.glob(os.path.join(lib, 'campaign_probe_check', '*', 'topo_*',
                                                                'iter_*.json'))]
    check(len(sides) == 2 and all(0.2 <= s.get('stoa_gate_p', -1) <= 1.0 for s in sides),
          'two sidecars, the gate p in [0.2, 1] in each')
    n_v = len(re.findall(r"variable 'V' has NaN", out))
    n_a = len(re.findall(r"Astrocyte's variable '\w+' has NaN", out))
    print('    NaN warnings: neurons %d, astrocytes %d' % (n_v, n_a))
    # before Block 10 (2026-10-31) the summed astrocyte drive gave 4 astrocyte warnings here
    check(n_v == 0 and n_a == 0, 'no NaN warning for the neurons or the astrocytes (Block 10)')


def main():
    tmp = tempfile.mkdtemp(prefix='gg_smoke_')
    try:
        free = g1(tmp)
        lib = g2(tmp)
        g3(lib, free)
        g4(tmp)
        g5(tmp)
        g6(tmp)
        g7()
        if '--sim' in sys.argv:
            g8(tmp)
        else:
            skip('G8 (no --sim)')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nRESULT: %d passed, %d failed%s' % (RESULTS['passed'], RESULTS['failed'],
          (', %d skipped' % RESULTS['skipped']) if RESULTS['skipped'] else ''))
    return 0 if RESULTS['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
