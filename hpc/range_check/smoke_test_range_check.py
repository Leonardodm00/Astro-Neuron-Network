#!/usr/bin/env python3
"""
smoke_test_range_check.py -- checks of hpc/range_check/ (specs/SPEC.md Block 6),
the standard prior range calibration: the corners of a setup's prior box and
random cells of its 7-bin grid.

  R1  corner design on the repository's 36-D root pipeline (23 free axes +
      conn_prob = 24 dimensions): a two-level fraction of 16,384 runs of
      resolution >= 5, no main effect or two-axis interaction aliased with
      another (Gram check), every column balanced; per conn level 8,192
      corners in 8 topologies x 48 workers x k=22 slots with 256 centre
      points; every design vector at the bounds its codes name, the inactive
      axes at nominal; the two levels together are the design
  R2  the same pipeline with the LindaColloDUP freeze (13 axes + conn_prob =
      14): the full factorial, each level all 2^13 corners exactly once; VA,
      VR held at -48, -58.2 mV in every vector, and at -50, -55 mV with a
      list off the nominal
  R3  --rc-like: parsing the rebuilt argument list reproduces every job_args value
  R4  37-D Giulia copy: the corner and interior designs build for synapse_astro
  R5  interior design with the LindaColloDUP freeze: 2,352 cells per level (7
      topologies x 48 workers x k=7), every value a bin midpoint, the bins of
      every axis uniform (chi-square), conn_prob at the 7 bin midpoints of
      [0.01, 0.5], the same seed giving the same cells
  R6  end to end (needs Brian2 and a C++ compiler): corners level 0 and
      interior level 0 on a tiny Neuronal network (--rc-subset 4, 2 workers);
      every run labelled from its theta (the 4 corners with the design's
      codes, the 4 cells with its bins); the analysis tables and the three
      figures are written; refuses an occupied folder and an unclamped model
  R7  analysis on synthetic runs with a known effect: Poisson trains at 2 Hz
      with axis A at its lower bound and 8 Hz at its upper bound give
      E_A = log10(8.01 / 2.01) within 0.05 and |E| < 0.05 for the inert B and
      conn_prob; interior medians follow 0.5 x 2^bin for A and stay flat for
      B; a run off the grid is 'unmatched' and the analysis exits 1
  R8  bytes: pure ASCII, LF only, in every file of this folder
  R9  submit_range_check.sh with module, conda and python replaced by stub
      shell functions, in an environment built from scratch (only PATH kept;
      see the comment above STUB_FUNCS); a probe first confirms the stubs are
      the functions the job's shell sees, else R9 stops. For SETUP=lcdup:
      corners at levels 0 1, interior at levels 0..6, the analysis and the
      plots, with the LindaColloDUP settings; giulia finds LIKE in the
      hhgap_v5 campaign; EXTRA reaches every call; FREEZE_ENC is decoded;
      PHASES, ANALYZE and BINS select what runs; CHECK_ONLY=1 runs nothing; an
      lcdup folder without the g box patch, a bad PHASES and an unknown SETUP
      exit 2
  R10 launch_range_check.sh with a stub qsub, in the same sealed environment
      (probed first like R9, so the real qsub is never reached): the corners
      job on intel and the interior job on cpu (48 cores, 24 h each), the
      analysis on cpu with 8 cores held by depend=afterany on both, one OUT
      and the pass-through variables in all three; DRYRUN=1, a failed
      precheck and a comma in a passed variable submit nothing; a failed
      second submission stops there and names the job to qdel

Run:  python3 smoke_test_range_check.py | tail -1        (about 2 min with R6)
      python3 smoke_test_range_check.py --no-sim          (skip R6)
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
    """Run `body` with range_check imported as RC, range_design as RD and the
    setup module as S, in a fresh interpreter (each pipeline copy defines its
    own HPC_main_sweep). The body prints one JSON line last."""
    code = ('import sys, json\nsys.path.insert(0, %r)\nimport numpy as np\nimport range_check as RC\n'
            'import range_design as RD\nS = RC.load_sweep_module(%r)\n' % (HERE, lib)) + body
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


CORNER_BODY = '''
args = S.build_parser().parse_args(%r)
space = RC.design_space(S, args)
free = space['free']
d = len(free) + (1 if space['has_conn'] else 0)
X, info = RD.two_level_design(d, RC.CFG['corner_budget'], np.random.default_rng(RC.CFG['seed'] + 20000),
                              RC.CFG['max_resolution'])
res = dict(d=d, n_free=len(free), kind=info['kind'], n_runs=info['n_runs'], resolution=info['resolution'],
           balanced=bool((X.sum(0) == 0).all()), distinct=len({tuple(r) for r in X}),
           conn_box=list(space['conn_box']), levels=[])
if %r:
    res['aliased'] = len(RD.aliased_low_order_pairs(X))
iva, ivr = S.PARAM_NAMES.index('VA'), S.PARAM_NAMES.index('VR')
rebuilt = set()
inactive = [j for j in range(S.N_DIMS) if j not in free]
for L in (0, 1):
    rc = RC.rc_parser().parse_args(['--rc-phase', 'corners', '--rc-level', str(L), '--rc-out', 'x'])
    design, padded, conn = RC.build_level(S, args, rc, 48)
    P = np.array(padded)
    codes = np.array(design['codes'])
    nd = design['n_design']
    Pd = P[:nd][:, free]
    at_bounds = bool(np.all(np.where(codes > 0, Pd == space['hi'], Pd == space['lo'])))
    centre = bool(np.all(P[nd:][:, free] == 0.5 * (space['lo'] + space['hi'])))
    nat = np.array([S.theta_to_natural(p) for p in P])
    rebuilt |= {tuple(int(x) for x in c) + ((-1 if L == 0 else 1),) for c in codes}
    res['levels'].append(dict(conn=conn, n_design=nd, n_padding=design['n_padding'], k=design['k'],
                              n_topologies=design['n_topologies'], n_slots=design['n_slots'],
                              at_bounds=at_bounds, centre=centre, codes_pm=bool(set(np.unique(codes)) <= {-1, 1}),
                              distinct=len({tuple(c) for c in codes}),
                              inactive_nominal=bool(np.all(P[:, inactive] == space['nominal'][inactive])),
                              va=sorted(set(np.round(nat[:, iva], 9).tolist())),
                              vr=sorted(set(np.round(nat[:, ivr], 9).tolist()))))
res['union_is_design'] = rebuilt == {tuple(int(x) for x in r) for r in X}
print(json.dumps(res))
'''


def r1(tmp):
    print('\n[R1] corner design, 36-D root pipeline (24 dimensions: a fraction)')
    d = in_subprocess(make_lib(os.path.join(tmp, 'lib36')), CORNER_BODY % (
        ['--out_dir', 'x', '--mode', 'Neuronal', '--sweep_group', 'neuron_synapse'], True))
    check(d is not None, 'design built')
    if not d:
        return
    check(d['n_free'] == 23 and d['d'] == 24 and d['kind'] == 'fraction' and d['n_runs'] == 16384
          and d['resolution'] >= 5, '23 free axes + conn_prob: fraction of 16384 runs, resolution %s' % d['resolution'])
    # quality floor: the seeded greedy search reaches 8 here; without its forbidden-region
    # update it falls to 6 (still valid, as min_word_length verifies, but weaker)
    check(d['resolution'] >= 7, 'the generator search keeps its quality (resolution %s >= 7)' % d['resolution'])
    check(d['aliased'] == 0 and d['balanced'] and d['distinct'] == 16384,
          'no main effect or two-axis interaction aliased (Gram check); columns balanced; 16384 distinct corners')
    lv = d['levels']
    check(all(x['n_design'] == 8192 and x['n_padding'] == 256 and x['k'] == 22 and x['n_topologies'] == 8
              and x['n_slots'] == 8448 for x in lv),
          'per conn level: 8192 corners + 256 centre points = 8 topologies x 48 workers x k=22')
    check(all(x['at_bounds'] and x['centre'] and x['codes_pm'] and x['inactive_nominal'] for x in lv),
          'every vector at the bounds its codes name; centre points at the box centre; inactive axes at nominal')
    check(lv[0]['conn'] == d['conn_box'][0] and lv[1]['conn'] == d['conn_box'][1] and d['union_is_design'],
          'conn_prob at its lower (level 0) and upper (level 1) bound; the two levels together are the design')


def r2(tmp):
    print('\n[R2] corner design with the LindaColloDUP freeze (14 dimensions: the full factorial)')
    lib = make_lib(os.path.join(tmp, 'lib36f'), freeze_patch=True)
    argv = ['--out_dir', 'x', '--mode', 'Neuronal', '--sweep_group', 'neuron_synapse', '--freeze', LCDUP_FREEZE,
            '--conn_prob_lo', '0.01', '--conn_prob_hi', '0.5']
    d = in_subprocess(lib, CORNER_BODY % (argv, False))
    check(d is not None, 'design built')
    if not d:
        return
    lv = d['levels']
    check(d['n_free'] == 13 and d['kind'] == 'full' and d['n_runs'] == 16384 and d['union_is_design'],
          '13 free axes + conn_prob: the full factorial of 16384 corners')
    check(all(x['distinct'] == 8192 for x in lv) and lv[0]['conn'] == 0.01 and lv[1]['conn'] == 0.5,
          'each level holds all 2^13 corners once; conn_prob 0.01 and 0.5')
    check(all(x['va'] == [-48.0] and x['vr'] == [-58.2] for x in lv), 'VA, VR held at -48, -58.2 mV in every vector')
    # -48 / -58.2 are also the registry nominals: a list off the nominal shows
    # that NAME=VALUE really reaches the design
    argv2 = [a if a != LCDUP_FREEZE else OFF_NOMINAL_FREEZE for a in argv]
    d2 = in_subprocess(lib, CORNER_BODY % (argv2, False))
    check(d2 is not None and all(x['va'] == [-50.0] and x['vr'] == [-55.0] for x in d2['levels']),
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
    body = '''
args = S.build_parser().parse_args(['--out_dir', 'x', '--mode', 'Full', '--sweep_group', 'synapse_astro'])
out = {}
for phase in ('corners', 'interior'):
    rc = RC.rc_parser().parse_args(['--rc-phase', phase, '--rc-level', '0', '--rc-out', 'x'])
    design, padded, conn = RC.build_level(S, args, rc, 48)
    out[phase] = dict(n_free=len(design['free_axes']), n_design=design['n_design'], n_slots=design['n_slots'],
                      info=design.get('corner_design'))
print(json.dumps(out))
'''
    d = in_subprocess(make_lib(os.path.join(tmp, 'lib37g'), src=GIULIA), body)
    ok = d is not None
    check(ok and d['corners']['n_free'] == 22 and d['corners']['info']['kind'] == 'fraction'
          and d['corners']['info']['resolution'] >= 5 and d['corners']['n_design'] == 8192,
          'synapse_astro: 22 free axes; corners a fraction of resolution %s, 8192 per level'
          % (d['corners']['info']['resolution'] if ok else '?'))
    check(ok and d['interior']['n_design'] == d['interior']['n_slots'] == 2352,
          'interior: 2352 cells per level')


def r5(tmp):
    print('\n[R5] interior design with the LindaColloDUP freeze')
    body = '''
from scipy import stats
args = S.build_parser().parse_args(%r)
space = RC.design_space(S, args)
free = space['free']
res = dict(levels=[])
for L in range(7):
    rc = RC.rc_parser().parse_args(['--rc-phase', 'interior', '--rc-level', str(L), '--rc-out', 'x'])
    design, padded, conn = RC.build_level(S, args, rc, 48)
    P = np.array(padded)[:, free]
    cells = np.array(design['cells'])
    on_mid = all(RD.bin_of(P[r, a], space['lo'][a], space['hi'][a], 7) == cells[r, a]
                 for r in range(len(P)) for a in range(len(free)))
    pmin = min(stats.chisquare(np.bincount(cells[:, a], minlength=7)).pvalue for a in range(len(free)))
    res['levels'].append(dict(conn=conn, n=design['n_design'], k=design['k'], n_topologies=design['n_topologies'],
                              on_mid=on_mid, pmin=pmin, first=cells[0].tolist()))
rc = RC.rc_parser().parse_args(['--rc-phase', 'interior', '--rc-level', '0', '--rc-out', 'x'])
again = RC.build_level(S, args, rc, 48)[0]['cells']
rc1 = RC.rc_parser().parse_args(['--rc-phase', 'interior', '--rc-level', '1', '--rc-out', 'x'])
res['same_seed_same'] = again == RC.build_level(S, args, rc, 48)[0]['cells']
res['levels_differ'] = again != RC.build_level(S, args, rc1, 48)[0]['cells']
res['mids'] = RD.bin_midpoints(0.01, 0.5, 7).tolist()
print(json.dumps(res))
''' % (['--out_dir', 'x', '--mode', 'Neuronal', '--sweep_group', 'neuron_synapse', '--freeze', LCDUP_FREEZE,
        '--conn_prob_lo', '0.01', '--conn_prob_hi', '0.5'],)
    d = in_subprocess(os.path.join(tmp, 'lib36f'), body)
    check(d is not None, 'design built')
    if not d:
        return
    lv = d['levels']
    check(all(x['n'] == 2352 and x['k'] == 7 and x['n_topologies'] == 7 for x in lv),
          '7 levels x 2352 cells (7 topologies x 48 workers x k=7)')
    check(all(abs(x['conn'] - m) < 1e-12 for x, m in zip(lv, d['mids'])),
          'conn_prob at the bin midpoints 0.045 ... 0.465')
    check(all(x['on_mid'] for x in lv), 'every free-axis value is the midpoint of its drawn bin')
    # 13 axes x 7 levels = 91 chi-square tests; 1e-4 keeps the chance of a false alarm below 1 %
    check(min(x['pmin'] for x in lv) > 1e-4,
          'bins uniform on every axis (smallest chi-square p over 91 tests: %.3g)' % min(x['pmin'] for x in lv))
    check(d['same_seed_same'] and d['levels_differ'], 'the same seed gives the same cells; levels differ')


def r6(tmp):
    print('\n[R6] end to end on a tiny network')
    import importlib.util
    if importlib.util.find_spec('brian2') is None:
        return skip('brian2 not importable')
    import pandas as pd
    lib = os.path.join(tmp, 'lib36f')
    out = os.path.join(tmp, 'out')
    common = ['--lib_dir', lib, '--mode', 'Neuronal', '--sweep_group', 'neuron_synapse', '--freeze', LCDUP_FREEZE,
              '--Nn', '12', '--c_max', '100', '--conn_prob_lo', '0.2', '--conn_prob_hi', '0.4',
              '--n_workers', '2', '--rc-out', out, '--rc-simtime', '0.3', '--rc-subset', '4']
    tool = [sys.executable, os.path.join(HERE, 'range_check.py')]
    rcs = [subprocess.run(tool + ['--rc-phase', ph, '--rc-level', '0'] + common, capture_output=True, text=True)
           for ph in ('corners', 'interior')]
    check(all(r.returncode == 0 for r in rcs), 'corners level 0 and interior level 0: exit 0')
    if any(r.returncode for r in rcs):
        for r in rcs:
            print(r.stdout[-2500:], r.stderr[-2500:])
        return
    a = subprocess.run([sys.executable, os.path.join(HERE, 'range_analysis.py'), out, '--t-skip', '0.05',
                        '--burst-lib', PIPE], capture_output=True, text=True)
    ok = a.returncode == 0 and os.path.isfile(os.path.join(out, 'runs.csv'))
    if not ok:
        print(a.stdout[-2500:], a.stderr[-2500:])
    df = pd.read_csv(os.path.join(out, 'runs.csv')) if ok else pd.DataFrame(columns=['kind'])
    dc = json.load(open(os.path.join(out, 'corners_c0', 'design.json')))
    di = json.load(open(os.path.join(out, 'interior_b0', 'design.json')))
    c, it = df[df.kind == 'corner'], df[df.kind == 'interior']
    names = dc['free_axes']
    got_codes = {tuple(int(r['code_' + n]) for n in names) for _, r in c.iterrows()}
    got_bins = {tuple(int(r['bin_' + n]) for n in names) for _, r in it.iterrows()}
    check(ok and len(df) == 8 and len(c) == 4 and len(it) == 4 and not (df.kind == 'unmatched').any(),
          'analysis: 8 runs labelled from their theta (4 corners, 4 interior), none unmatched')
    check(got_codes == {tuple(x) for x in dc['codes']} and got_bins == {tuple(x) for x in di['cells']}
          and set(c.code_conn_prob) == {-1} and set(it.bin_conn_prob) == {0},
          "labels equal the design's codes and bins (conn_prob: code -1, bin 0)")
    tables = ['corner_effects.csv', 'corner_interactions.csv', 'interior_bins.csv', 'interior_pairs.csv',
              'interior_spearman.csv', 'calibration_summary.txt']
    eff = pd.read_csv(os.path.join(out, 'corner_effects.csv')) if ok else pd.DataFrame()
    check(ok and all(os.path.isfile(os.path.join(out, t)) for t in tables) and len(eff) == 14,
          'tables written; corner_effects lists the 14 dimensions')
    p = subprocess.run([sys.executable, os.path.join(HERE, 'range_plots.py'), out], capture_output=True, text=True)
    check(p.returncode == 0 and all(os.path.isfile(os.path.join(out, f)) for f in
                                    ('corner_effects.png', 'interior_bins.png', 'rate_distributions.png')),
          'three figures written')
    r = subprocess.run(tool + ['--rc-phase', 'corners', '--rc-level', '0'] + common, capture_output=True, text=True)
    check(r.returncode != 0 and 'already holds runs' in (r.stdout + r.stderr), 'refuses an occupied folder')
    unc = make_lib(os.path.join(tmp, 'lib_unclamped'))
    path = os.path.join(unc, 'ASD_fun_BD_cpp.py')
    open(path, 'w').write(open(path).read().replace(': volt (unless refractory)', ': volt'))
    r = subprocess.run(tool + ['--rc-phase', 'corners', '--rc-level', '0', '--lib_dir', unc,
                               '--rc-out', os.path.join(tmp, 'o2'), '--rc-design-only'], capture_output=True, text=True)
    check(r.returncode != 0 and 'REFUSED' in (r.stdout + r.stderr), 'refuses an unclamped model')


def _synthetic_run(pdir, topo, it, theta, names, rate, conn, rng, n_neurons=50, t_total=20.0):
    tdir = os.path.join(pdir, 'topo_%05d' % topo)
    os.makedirs(tdir, exist_ok=True)
    n = rng.poisson(rate * t_total * n_neurons)
    np.savez(os.path.join(tdir, 'iter_%05d.npz' % it),
             spk_N_t=np.sort(rng.uniform(0, t_total, n)).astype(np.float32),
             spk_N_i=rng.integers(0, n_neurons, n).astype(np.int32))
    json.dump(dict(topo_idx=topo, iter_idx=it, conn_prob=conn, theta=list(theta), params=list(theta),
                   param_names=names, npz_relpath='topo_%05d/iter_%05d.npz' % (topo, it)),
              open(os.path.join(tdir, 'iter_%05d.json' % it), 'w'))


def r7(tmp):
    print('\n[R7] analysis on synthetic runs with a known effect')
    sys.path.insert(0, HERE)
    import pandas as pd
    import range_analysis as RA
    import range_design as RD
    root = os.path.join(tmp, 'synth')
    names, lo, hi = ['A', 'B'], [0.0, -1.0], [1.0, 1.0]
    rng = np.random.default_rng(7)
    common = dict(free_axes=names, theta_lo=lo, theta_hi=hi, conn_box=[0.1, 0.5], lib_dir=PIPE,
                  param_names=names, log_params=[])
    for L, conn in ((0, 0.1), (1, 0.5)):
        pdir = os.path.join(root, 'corners_c%d' % L)
        os.makedirs(pdir)
        json.dump(dict(common, phase='corners', level=L, has_conn=True,
                       corner_design=dict(kind='full', n_runs=8, d=3, resolution=None)),
                  open(os.path.join(pdir, 'design.json'), 'w'))
        json.dump(dict(Nn=50, simtime=20.0), open(os.path.join(pdir, 'job_args.json'), 'w'))
        it = 0
        for rep in range(6):
            for ca in (-1, 1):
                for cb in (-1, 1):
                    theta = [hi[0] if ca > 0 else lo[0], hi[1] if cb > 0 else lo[1]]
                    _synthetic_run(pdir, rep, it, theta, names, 8.0 if ca > 0 else 2.0, conn, rng)
                    it += 1
    pdir = os.path.join(root, 'interior_b0')
    os.makedirs(pdir)
    json.dump(dict(common, phase='interior', level=0, has_conn=False, bins=7), open(os.path.join(pdir, 'design.json'), 'w'))
    json.dump(dict(Nn=50, simtime=20.0), open(os.path.join(pdir, 'job_args.json'), 'w'))
    mids = [RD.bin_midpoints(a, b, 7) for a, b in zip(lo, hi)]
    it = 0
    for rep in range(2):
        for ia in range(7):
            for ib in range(7):
                _synthetic_run(pdir, rep, it, [mids[0][ia], mids[1][ib]], names, 0.5 * 2 ** ia, 0.3, rng)
                it += 1
    _synthetic_run(pdir, 9, 0, [0.33, mids[1][3]], names, 1.0, 0.3, rng)        # off the grid
    rc = RA.main([root, '--burst-lib', PIPE, '--workers', '1'])
    eff = pd.read_csv(os.path.join(root, 'corner_effects.csv')).set_index('dimension')
    want = np.log10(8.01) - np.log10(2.01)
    # SE of E_A is about 0.0023 (Poisson counts of 1,800 and 7,200 spikes, 24 runs per side)
    check(abs(eff.loc['A', 'effect_y'] - want) < 0.05,
          'E_A = %.4f (planted %.4f, tolerance 0.05)' % (eff.loc['A', 'effect_y'], want))
    check(abs(eff.loc['B', 'effect_y']) < 0.05 and abs(eff.loc['conn_prob', 'effect_y']) < 0.05,
          'inert B and conn_prob: |E| = %.4f, %.4f' % (abs(eff.loc['B', 'effect_y']),
                                                      abs(eff.loc['conn_prob', 'effect_y'])))
    bins_t = pd.read_csv(os.path.join(root, 'interior_bins.csv'))
    a = bins_t[bins_t.dimension == 'A'].sort_values('bin')
    b = bins_t[bins_t.dimension == 'B']
    # 14 runs per bin; at 0.5 Hz a run's count is Poisson(450), so 15 % is > 3 SD of a median
    a_ok = list(a.bin) == list(range(7)) and bool(
        np.all(np.abs(a.rate_median.values / (0.5 * 2.0 ** np.arange(7)) - 1) < 0.15))
    check(a_ok, 'interior: bins 0..6 of A only, medians following 0.5 x 2^bin within 15 %% (bins %s)' % list(a.bin))
    check(list(b.bin) == list(range(7)) and b.rate_median.max() / b.rate_median.min() < 1.3,
          'interior: medians of B flat (max/min %.2f)' % (b.rate_median.max() / b.rate_median.min()))
    df = pd.read_csv(os.path.join(root, 'runs.csv'))
    check(rc == 1 and int((df.kind == 'unmatched').sum()) == 1 and int((df.kind == 'corner').sum()) == 48,
          'the off-grid run is unmatched and the analysis exits 1; 48 corners, 98 interior cells labelled')


def r8():
    print('\n[R8] bytes')
    for f in sorted(os.listdir(HERE)):
        p = os.path.join(HERE, f)
        if os.path.isfile(p) and not f.endswith('.pyc'):
            b = open(p, 'rb').read()
            check(b.count(b'\r') == 0 and all(x < 128 for x in b), '%s: LF only, pure ASCII' % f)


# R9 runs the job script with module, conda and python replaced by shell
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
    will run the script under test (same wrapper, same environment). R9 and R10
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


def explain(ok, r, calls, what='job'):
    """After a failed check, print what the script did, so a pasted run shows the cause."""
    if ok:
        return
    print('    %s exit %d, %d stub calls; its last output lines:' % (what, r.returncode, len(calls)))
    for tag, text in (('out', r.stdout), ('err', r.stderr)):
        for line in text.strip().splitlines()[-8:]:
            print('    %s| %s' % (tag, line))


def r9(tmp):
    print('\n[R9] the PBS job script with stub module, conda and python')
    base = os.path.join(tmp, 'stubbase')
    os.makedirs(os.path.join(base, 'etc', 'profile.d'))
    open(os.path.join(base, 'etc', 'profile.d', 'conda.sh'), 'w').write(
        'conda() { return 0; }\nmodule() { return 0; }\n')
    home = os.path.join(tmp, 'r9_home')
    os.makedirs(home)
    lib = make_fake_setup(os.path.join(tmp, 'fake_setup'))
    nobox = make_fake_setup(os.path.join(tmp, 'fake_setup_nobox'), box=False)
    n_jobs = [0]
    if not stubs_loaded(STUB_FUNCS, ['python', 'conda', 'module'],
                        dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=home, LC_ALL='C',
                             STUB_LOG=os.path.join(tmp, 'r9_probe.log'), STUB_BASE=base)):
        return

    def job(**env):
        n_jobs[0] += 1
        log = os.path.join(tmp, 'r9_calls_%d.log' % n_jobs[0])
        e = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=home, LC_ALL='C',
                 STUB_LOG=log, STUB_BASE=base, PBS_O_WORKDIR=HERE, PBS_NCPUS='48',
                 LIB_DIR=lib, OUT=os.path.join(tmp, 'job_out_%d' % n_jobs[0]))
        e.update(env)
        r = subprocess.run([BASH, '-c', STUB_FUNCS, 'r9', os.path.join(HERE, 'submit_range_check.sh')],
                           env=e, capture_output=True, text=True)
        raw = open(log).read().splitlines() if os.path.exists(log) else []
        return r, [c for c in raw if not c.startswith('-c ')], raw

    def split(calls):
        return ([c for c in calls if '--rc-phase corners' in c], [c for c in calls if '--rc-phase interior' in c])

    r, calls, _ = job(SETUP='lcdup')
    cor, inte = split(calls)
    ok = check(r.returncode == 0 and len(cor) == 2 and len(inte) == 7 and len(calls) == 11
               and calls[-2].startswith('range_analysis.py') and calls[-1].startswith('range_plots.py'),
               'lcdup: 2 corner levels + 7 interior levels + analysis + plots (%d calls)' % len(calls))
    explain(ok, r, calls)
    check(len(cor) == 2 and len(inte) == 7
          and all('--rc-phase corners --rc-level %d --rc-corner-topologies 8 --rc-corner-budget 16384 ' % L in c
                  for L, c in enumerate(cor))
          and all('--rc-phase interior --rc-level %d --rc-bins 7 --rc-interior-runs 16384 '
                  '--rc-interior-topologies 7 ' % L in c for L, c in enumerate(inte)),
          'levels in order with their design arguments (corners 0 1; interior 0..6)')
    check(len(cor + inte) == 9
          and all('--freeze Cm,I_inj,VA=-48,DeltaA,VR=-58.2' in c and '--mode Neuronal' in c
                  and '--n_workers 48' in c and '--conn_prob_lo 0.01 --conn_prob_hi 0.5' in c for c in cor + inte),
          'lcdup: freeze VA=-48 VR=-58.2, conn_prob box 0.01-0.5, 48 workers in every call')
    r, calls, _ = job(SETUP='giulia', EXTRA='--stoa_gate_lo 0.2 --stoa_gate_hi 1.0')
    cor, inte = split(calls)
    ok = check(r.returncode == 0 and len(cor + inte) == 9
               and all('campaign_cadex_hhgap_v5/sweep_intel_task0000/job_args.json' in c
                       and c.endswith('--stoa_gate_lo 0.2 --stoa_gate_hi 1.0') for c in cor + inte),
               'giulia: LIKE found in the hhgap_v5 campaign; EXTRA last in every call')
    explain(ok, r, calls)
    r, calls, _ = job(SETUP='lcdup', FREEZE_ENC='Cm:I_inj:VA@-50:DeltaA:VR@-55:EC50_ampa:EC50_nmda:alpha_syn:O_G:Omega_G')
    cor, inte = split(calls)
    ok = check(r.returncode == 0 and len(cor + inte) == 9
               and all('--freeze Cm,I_inj,VA=-50,DeltaA,VR=-55,EC50_ampa' in c for c in cor + inte),
               'lcdup: FREEZE_ENC decoded (a list that differs from the default)')
    explain(ok, r, calls)
    r, calls, _ = job(SETUP='lcdup', PHASES='corners', ANALYZE='0')
    ok = check(r.returncode == 0 and len(calls) == 2 and all('--rc-phase corners' in c for c in calls),
               'PHASES=corners ANALYZE=0: the 2 corner levels only')
    explain(ok, r, calls)
    r, calls, _ = job(SETUP='lcdup', PHASES='interior', ANALYZE='0', BINS='5')
    ok = check(r.returncode == 0 and len(calls) == 5
               and all('--rc-phase interior --rc-level %d --rc-bins 5 ' % L in c for L, c in enumerate(calls)),
               'PHASES=interior ANALYZE=0 BINS=5: interior levels 0..4 with 5 bins')
    explain(ok, r, calls)
    out_a = os.path.join(tmp, 'analysis_out')
    os.makedirs(os.path.join(out_a, 'corners_c0'))
    open(os.path.join(out_a, 'corners_c0', 'design.json'), 'w').write('{}')
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
    r, calls, raw = job(SETUP='lcdup', PHASES='oat')
    ok = check(r.returncode == 2 and not raw, 'bad PHASES (the old oat): exit 2, nothing run')
    explain(ok, r, calls)
    r, calls, raw = job(SETUP='nonsense')
    ok = check(r.returncode == 2 and not raw, 'unknown SETUP: exit 2, nothing run')
    explain(ok, r, calls)


# qsub stand-in for R10, a shell function like the stubs of R9: it logs each
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


def r10(tmp):
    print('\n[R10] launch_range_check.sh with a stub qsub')
    home = os.path.join(tmp, 'r10_home')
    os.makedirs(home)
    lib = make_fake_setup(os.path.join(tmp, 'r10_setup'))
    nobox = make_fake_setup(os.path.join(tmp, 'r10_setup_nobox'), box=False)
    out = os.path.join(tmp, 'r10_out')
    n_runs = [0]
    probe_log = os.path.join(tmp, 'r10_probe.log')
    open(probe_log, 'w').close()
    if not stubs_loaded(QSUB_FUNCS, ['qsub'], dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'),
                                                  HOME=home, LC_ALL='C', QSUB_LOG=probe_log)):
        return

    def launch(**env):
        n_runs[0] += 1
        log = os.path.join(tmp, 'r10_qsub_%d.log' % n_runs[0])
        open(log, 'w').close()
        e = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), HOME=home, LC_ALL='C',
                 QSUB_LOG=log, SETUP='lcdup', LIB_DIR=lib, OUT=out)
        e.update(env)
        r = subprocess.run([BASH, '-c', QSUB_FUNCS, 'r10', os.path.join(HERE, 'launch_range_check.sh')],
                           env=e, capture_output=True, text=True)
        return r, open(log).read().splitlines()

    r, calls = launch()
    common = 'SETUP=lcdup,OUT=%s,LIB_DIR=%s' % (out, lib)
    ok = check(r.returncode == 0 and len(calls) == 3
               and '-q intel -N rc_corners -l select=1:ncpus=48 -l walltime=24:00:00' in calls[0]
               and (common + ',PHASES=corners,ANALYZE=0') in calls[0]
               and '-q cpu -N rc_interior -l select=1:ncpus=48 -l walltime=24:00:00' in calls[1]
               and (common + ',PHASES=interior,ANALYZE=0') in calls[1]
               and calls[0].endswith('submit_range_check.sh') and calls[1].endswith('submit_range_check.sh'),
               'corners job on intel, interior job on cpu, 48 cores and 24 h each, one OUT')
    explain(ok, r, calls, 'launcher')
    ok = check(len(calls) == 3 and '-q cpu -N rc_analysis -l select=1:ncpus=8 -l walltime=02:00:00' in calls[2]
               and '-W depend=afterany:101.pbsserver:102.pbsserver' in calls[2]
               and (common + ',PHASES=none,ANALYZE=1') in calls[2],
               'analysis job on cpu, 8 cores, held by depend=afterany on both, same OUT')
    explain(ok, r, calls, 'launcher')
    r, calls = launch(BINS='5', INTERIOR_RUNS='8000')
    ok = check(r.returncode == 0 and len(calls) == 3
               and all(',BINS=5' in c and ',INTERIOR_RUNS=8000' in c for c in calls),
               'pass-through variables (BINS, INTERIOR_RUNS) reach all three jobs')
    explain(ok, r, calls, 'launcher')
    r, calls = launch(DRYRUN='1')
    ok = check(r.returncode == 0 and not calls and r.stdout.count('[dry run] qsub') == 3,
               'DRYRUN=1: three qsub lines printed, nothing submitted')
    explain(ok, r, calls, 'launcher')
    r, calls = launch(LIB_DIR=nobox)
    r2, calls2 = launch(EXTRA='--a 1,--b 2')
    ok = check(r.returncode == 2 and not calls and r2.returncode == 2 and not calls2,
               'failed precheck, or a comma in a passed variable: exit 2, nothing submitted')
    explain(ok, r, calls, 'launcher')
    r, calls = launch(QSUB_FAIL_AT='2')
    ok = check(r.returncode == 3 and len(calls) == 2 and 'qdel 101.pbsserver' in r.stderr,
               'second submission refused: exit 3, no third job, the first job named for qdel')
    explain(ok, r, calls, 'launcher')


def main():
    tmp = tempfile.mkdtemp(prefix='range_smoke_')
    try:
        r1(tmp)
        r2(tmp)
        r3(tmp)
        r4(tmp)
        r5(tmp)
        if '--no-sim' in sys.argv:
            skip('R6 (--no-sim)')
        else:
            r6(tmp)
        r7(tmp)
        r8()
        r9(tmp)
        r10(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nRESULT: %d passed, %d failed%s' % (RESULTS['passed'], RESULTS['failed'],
          (', %d skipped' % RESULTS['skipped']) if RESULTS['skipped'] else ''))
    return 0 if RESULTS['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
