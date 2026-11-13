"""Block 4 -- LindaColloDUP launch package (SPEC.md Block 4 with its corrections).

Oracles: the sizing formula computed by hand; the dry run's lines; every guard's
exit code; apply_box_patch.py's rule (exactly the two g rows, the marker, the
derived log flag and theta box), its refusals (file byte-identical, no
backup) and idempotence; the .gitignore contract. Run:
    python -m pytest tests/spec/test_spec_block04_lcdup.py -q
"""
import difflib
import math
import os
import re
import shutil
import stat
import subprocess
import sys

import numpy as np
import pytest

from conftest import GG, LCDUP, PIPE, REPO, copy_tree, package_with_pipeline, run

BASE_ENV = dict(SIZING_CONFIRMED='1', REFRACTORY_CLAMP='yes', DRYRUN='1')
MARK = '[LindaColloDUP box 2026-10-30]'


def ntasks(target, headroom_pct, n_top, k, queues):
    w = {q: n * c for q, n, c in queues}
    tot = sum(w.values())
    return {q: math.ceil((target * w[q] / tot) * headroom_pct / 100.0 / (n_top * n * k)) for q, n, c in queues}


def test_sizing_formula_by_hand():
    assert ntasks(10 ** 6, 200, 32, 1, [('cfd', 192, 1), ('cpu', 48, 4)]) == {'cfd': 163, 'cpu': 652}
    assert ntasks(10 ** 6, 200, 48, 1, [('intel', 48, 5), ('cpu', 48, 4)]) == {'intel': 483, 'cpu': 386}
    assert round(1300 * 0.35 ** 2) == 159 and round(1200 * 0.30 ** 2) == 108


def launch(folder, tmp_path, **env):
    tree = tmp_path / 'ANN_scan'
    tree.mkdir(exist_ok=True)
    e = dict(BASE_ENV, SEED_SCAN_TREE=str(tree))
    e.update(env)
    e = {k: v for k, v in e.items() if v is not None}
    return run(['bash', 'launch_LCDUP.sh'], cwd=folder, env=e, timeout=300)


@pytest.fixture
def lfolder(lcdup_folder, tmp_path):
    return copy_tree(lcdup_folder, str(tmp_path / 'LindaColloDUP'))


def test_dry_run_lines(lfolder, tmp_path):
    r = launch(lfolder, tmp_path)
    out = r.stdout
    assert r.returncode == 0, out + r.stderr
    assert re.search(r'Nn=159\b', out)
    assert 'conn_prob      : U[0.01, 0.4] per topology' in out
    assert 'g boxes        : g_ampa [0.01, 10.0] nS; g_nmda [0.01, 10.0] nS' in out
    assert 'V clamp (t_ref): yes' in out
    assert 'freeze         : Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G' in out
    q = re.findall(r'qsub -q (\S+) -l select=1:ncpus=(\d+),walltime=(\S+) \\\s+-J (\S+)', out)
    assert q == [('cfd', '192', '24:00:00', '0-162%1'), ('cpu', '48', '24:00:00', '0-651%4')]
    assert '[dry run] nothing submitted' in out
    for v in ('MODE=Neuronal', 'SWEEP_GROUP=neuron_synapse', 'CONN_RULE=flat', 'CONN_PERIODIC=0',
              'CONN_PROB_LO=0.01', 'CONN_PROB_HI=0.4', 'STOA_GATE_LO=1.0', 'STOA_GATE_HI=1.0',
              'SIMTIME=180', 'DENSITY=1300', 'C_MAX=350', 'N_TOPOLOGIES=32', 'N_PARAMS_PER_WORKER=1',
              'CAMPAIGN_TAG=lcdup15_rho1300_neur_n01', 'CONDA_ENV=brian_final'):
        assert v in out, v
    # a dry run still reserves its span in the folder's own ledger
    led = open(os.path.join(lfolder, 'artifacts', 'seed_ledger.tsv')).read()
    assert 'DRYRUN-not-submitted' in led


@pytest.mark.parametrize('env,code', [
    (dict(SIZING_CONFIRMED=None), 3),
    (dict(REFRACTORY_CLAMP=None), 3),
    (dict(REFRACTORY_CLAMP='maybe'), 3),
    (dict(REFRACTORY_CLAMP='no'), 4),                      # disagrees with the clamped file
    (dict(MODE='Full'), 4),
    (dict(ASTRO_FRAC='0.5'), 4),
    (dict(STOA_GATE_LO='0.5'), 4),
    (dict(CONN_PROB_LO='0.6', CONN_PROB_HI='0.4'), 4),
    (dict(FREEZE='Cm,VA=minus48'), 4),
])
def test_guards(lfolder, tmp_path, env, code):
    r = launch(lfolder, tmp_path, **env)
    assert r.returncode == code, (env, r.stdout[-800:], r.stderr[-800:])
    assert 'qsub -q' not in r.stdout


def test_guard_missing_files_and_patches(lcdup_folder, tmp_path):
    for case in ('missing_model', 'no_freeze_patch', 'no_box_patch'):
        d = copy_tree(lcdup_folder, str(tmp_path / case))
        if case == 'missing_model':
            os.remove(os.path.join(d, 'ASD_fun_BD_cpp.py'))
        elif case == 'no_freeze_patch':
            shutil.copy2(os.path.join(PIPE, 'HPC_main_sweep.py'), os.path.join(d, 'HPC_main_sweep.py'))
        else:
            txt = open(os.path.join(d, 'HPC_main_sweep.py')).read().replace(MARK, '', 1)
            open(os.path.join(d, 'HPC_main_sweep.py'), 'w').write(txt)
        r = launch(d, tmp_path)
        assert r.returncode == 2, (case, r.stderr[-500:])


def test_guard_seed_scan_tree_and_old_allocator(lfolder, tmp_path):
    r = launch(lfolder, tmp_path, SEED_SCAN_TREE=str(tmp_path / 'no_such_tree'))
    assert r.returncode == 5 and 'qsub -q' not in r.stdout
    txt = open(os.path.join(lfolder, 'seed_alloc.py')).read().replace('--scan-tree', '--scan-trees')
    open(os.path.join(lfolder, 'seed_alloc.py'), 'w').write(txt)
    r = launch(lfolder, tmp_path)
    assert r.returncode == 5


def test_guard_existing_tag_before_any_qsub(lfolder, tmp_path):
    os.makedirs(os.path.join(lfolder, 'campaign_lcdup15_rho1300_neur_n01'))
    stub = tmp_path / 'bin'
    stub.mkdir()
    (stub / 'qsub').write_text('#!/bin/sh\necho QSUB_CALLED >> "%s"\necho 1.x\n' % (tmp_path / 'qsub.log'))
    os.chmod(str(stub / 'qsub'), 0o755)
    r = launch(lfolder, tmp_path, DRYRUN='0', PATH='%s:%s' % (stub, os.environ['PATH']))
    assert r.returncode == 4
    assert not (tmp_path / 'qsub.log').exists()


# ---------------------------------------------------------------------------
# apply_box_patch.py
# ---------------------------------------------------------------------------
@pytest.fixture
def boxdir(tmp_path):
    d = str(tmp_path / 'pkg')
    os.makedirs(d)
    shutil.copy2(os.path.join(LCDUP, 'apply_box_patch.py'), d)
    shutil.copy2(os.path.join(PIPE, 'HPC_main_sweep.py'), d)
    shutil.copy2(os.path.join(PIPE, 'HPC_single_run.py'), d)
    return d


def box(d, *args):
    return run([sys.executable, 'apply_box_patch.py'] + list(args), cwd=d)


def g_rows(text):
    return [ln for ln in text.splitlines() if re.search(r'#\s*1[45]\s+g_(ampa|nmda)\b', ln)]


def test_box_patch_rule(boxdir):
    f = os.path.join(boxdir, 'HPC_main_sweep.py')
    os.chmod(f, 0o640)
    before = open(f).read()
    r = box(boxdir)
    assert r.returncode == 0, r.stdout + r.stderr
    after = open(f).read()
    diff = [d for d in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm='', n=0)
            if d[:1] in '+-' and not d.startswith(('+++', '---'))]
    assert len(diff) == 4, diff                                # two rows replaced, nothing else
    rows = g_rows(after)
    assert len(rows) == 2
    for ln in rows:
        assert re.match(r'^\s*\(0\.01,\s*10\.0\),', ln) and MARK in ln, ln
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o640
    baks = [x for x in os.listdir(boxdir) if x.startswith('HPC_main_sweep.py.bak-box-')]
    assert len(baks) == 1 and re.match(r'HPC_main_sweep\.py\.bak-box-\d{8}_\d{6}$', baks[0])
    assert open(os.path.join(boxdir, baks[0])).read() == before
    code = ("import sys, json; sys.path.insert(0, '.'); import HPC_main_sweep as M; "
            "i, j = M.PARAM_NAMES.index('g_ampa'), M.PARAM_NAMES.index('g_nmda'); "
            "print(json.dumps([i in M.LOG_PARAMS, j in M.LOG_PARAMS, list(M.PARAM_BOUNDS_THETA[i]), "
            "list(M.PARAM_BOUNDS_THETA[j])]))")
    out = run([sys.executable, '-c', code], cwd=boxdir).stdout.strip().splitlines()[-1]
    import json
    li, lj, ti, tj = json.loads(out)
    assert li and lj
    np.testing.assert_allclose([ti, tj], [[np.log(0.01), np.log(10.0)]] * 2, rtol=0, atol=1e-15)
    r2 = box(boxdir)                                           # already patched: exit 0, no change
    assert r2.returncode == 0 and open(f).read() == after
    assert len([x for x in os.listdir(boxdir) if x.startswith('HPC_main_sweep.py.bak-box-')]) == 1


def _mutate(kind, text):
    rows = g_rows(text)
    if kind == 'crlf':
        return text.replace('\n', '\r\n')
    if kind == 'no_block':
        return text.replace('PARAM_BOUNDS = np.array([', 'PARAM_BOUNDZ = np.array([', 1)
    if kind == 'row_missing':
        return text.replace(rows[0] + '\n', '', 1)
    if kind == 'row_twice':
        return text.replace(rows[0] + '\n', rows[0] + '\n' + rows[0] + '\n', 1)
    if kind == 'other_box':
        return text.replace(rows[0], rows[0].replace('(0.01,    5.0)', '(0.02,    5.0)', 1), 1)
    if kind == 'new_box_no_marker':
        return text.replace(rows[0], rows[0].replace('(0.01,    5.0)', '(0.01,   10.0)', 1), 1)
    raise ValueError(kind)


@pytest.mark.parametrize('kind', ['crlf', 'no_block', 'row_missing', 'row_twice', 'other_box',
                                  'new_box_no_marker', 'not_utf8', 'partial'])
def test_box_patch_refusals(boxdir, kind):
    f = os.path.join(boxdir, 'HPC_main_sweep.py')
    text = open(f).read()
    if kind == 'not_utf8':
        open(f, 'wb').write(text.encode('utf-8') + b'\n# \xff\n')
    elif kind == 'partial':
        assert box(boxdir).returncode == 0
        patched = open(f).read()
        first = g_rows(patched)[0]
        orig = g_rows(text)[0]
        open(f, 'w').write(patched.replace(first, orig, 1))     # one row back to the registry box
    else:
        mutated = _mutate(kind, text)
        assert mutated != text, kind
        open(f, 'w', newline='').write(mutated)
    for x in os.listdir(boxdir):
        if '.bak-box-' in x:
            os.remove(os.path.join(boxdir, x))
    before, listing = open(f, 'rb').read(), sorted(os.listdir(boxdir))
    r = box(boxdir)
    assert r.returncode == 1, (kind, r.stdout, r.stderr)
    assert open(f, 'rb').read() == before and sorted(os.listdir(boxdir)) == listing


# ---------------------------------------------------------------------------
# .gitignore of the two launch packages
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('pkg', ['LindaColloDUP', 'GiuliaGate'])
def test_gitignore_contract(pkg):
    rel = os.path.join('hpc', 'Phenomenological_finalv1', pkg)
    ignored = ['HPC_main_sweep.py', 'HPC_single_run.py', 'ASD_fun_BD_cpp.py', 'aggregate_sweep.py',
               'burst_metrics.py', 'burst_plots.py', 'synapse_pdist.csv',
               'HPC_main_sweep.py.bak-box-20261101_000000', 'ASD_fun_BD_cpp.py.crlf_bak',
               'campaign_x/sweep_cpu_task0000/job_args.json', 'artifacts/seed_ledger.tsv']
    tracked = subprocess.run(['git', 'ls-files', rel], cwd=REPO, stdout=subprocess.PIPE,
                             universal_newlines=True).stdout.split()
    for p in ignored:
        r = subprocess.run(['git', 'check-ignore', '-q', '--no-index', os.path.join(rel, p)], cwd=REPO)
        assert r.returncode == 0, p
    assert tracked
    for p in tracked:
        r = subprocess.run(['git', 'check-ignore', '-q', '--no-index', p], cwd=REPO)
        assert r.returncode == 1, p
