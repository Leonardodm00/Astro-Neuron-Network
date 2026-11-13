"""Block 1 -- the refractory hold in the CAdEx neuron (SPEC.md section 2, Block 1).

Static oracles here (both branches, the GitHub-wide check of the spec); the
behavioural oracle (V = V_R exactly through (t_s, t_s + t_ref] in a compiled
run) is in test_spec_e2e_pipeline.py, and the repository invariant through
clamp_cadex_copies.py in test_spec_block05_clamp_cadex.py. Run:
    python -m pytest tests/spec/test_spec_block01_refractory.py -q
"""
import os
import re
import subprocess
import sys

import pytest

from conftest import PIPE, REPO

FLAG = '(unless refractory)'
OTHER = 'origin/Inverse-Transform-Sampling'
OTHER_FILE = 'hpc/Phenomenological main/ASD_fun_BD_cpp.py'
CADEX = re.compile(r'exp\(\(V-VT\)/DeltaT\)')


def git(*args):
    return subprocess.run(['git'] + list(args), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)


def other_branch():
    if git('rev-parse', '--verify', '-q', OTHER).returncode != 0:
        git('fetch', '-q', 'origin', 'Inverse-Transform-Sampling:refs/remotes/' + OTHER)
    if git('rev-parse', '--verify', '-q', OTHER).returncode != 0:
        pytest.skip('branch Inverse-Transform-Sampling not reachable from this clone')
    return OTHER


def cadex_dv_lines(rev):
    """(path, line) of every dV/dt line with the CAdEx exponential, in every tracked .py file of rev."""
    out = []
    for hit in git('grep', '-n', '-E', r'^\s*dV/dt\s*=', rev, '--', '*.py').stdout.splitlines():
        _, path, _, line = hit.split(':', 3)
        if CADEX.search(line.replace(' ', '')):
            out.append((path, line))
    return out


@pytest.mark.parametrize('which', ['main', 'other'])
def test_every_cadex_model_file_holds_V(which):
    rev = 'HEAD' if which == 'main' else other_branch()
    hits = cadex_dv_lines(rev)
    models = [(p, ln) for p, ln in hits if os.path.basename(p) == 'ASD_fun_BD_cpp.py']
    assert models, 'no CAdEx model file found on %s' % rev
    for p, ln in models:
        assert ln.rstrip().endswith(FLAG), (rev, p, ln)
    if which == 'other':
        assert [p for p, _ in models] == [OTHER_FILE]
    else:
        assert sorted(p for p, _ in models) == ['hpc/Phenomenological_finalv1/ASD_fun_BD_cpp.py',
                                                'hpc/Phenomenological_finalv1/Giulia/ASD_fun_BD_cpp.py']


def test_hh_model_unflagged():
    text = open(os.path.join(REPO, 'hpc', 'ASD_fun_BD_cpp.py'), encoding='utf-8').read()
    dv = [ln for ln in text.splitlines() if re.match(r'^\s*dV/dt\s*=', ln)]
    assert dv and not any(FLAG in ln for ln in dv)


def eqs_block(text):
    """The Equations('''...''') string holding the CAdEx dV/dt line."""
    k = text.index('dV/dt  = noise + (gl*(El-V) + gl*DeltaT*exp((V-VT)/DeltaT)')
    a = text.rindex("Equations('''", 0, k) + len("Equations('''")
    b = text.index("''')", k)
    return text[a:b]


@pytest.mark.parametrize('which', ['root', 'giulia', 'other'])
def test_brian2_parses_the_flag(which):
    from brian2 import Equations
    if which == 'other':
        text = git('show', '%s:%s' % (other_branch(), OTHER_FILE)).stdout
    else:
        sub = '' if which == 'root' else 'Giulia'
        text = open(os.path.join(PIPE, sub, 'ASD_fun_BD_cpp.py'), encoding='utf-8').read()
    eqs = Equations(eqs_block(text))
    assert 'unless refractory' in eqs['V'].flags
    assert 'unless refractory' not in eqs['gA'].flags          # g_A keeps evolving during t_ref


def test_group_arguments_and_t_ref():
    for sub, t_ref in (('', '3*ms'), ('Giulia', '5*ms')):
        text = open(os.path.join(PIPE, sub, 'ASD_fun_BD_cpp.py'), encoding='utf-8').read()
        m = re.search(r"'t_ref':\s*([0-9.]+\s*\*\s*ms)", text)
        assert m and m.group(1).replace(' ', '') == t_ref, (sub, m and m.group(0))
        assert re.search(r"threshold='V > VD', reset='V = VR; gA \+= delta_gA',\s*"
                         r"refractory=params_NN\['t_ref'\]", text), sub
    sys.path.insert(0, PIPE)
    import ASD_fun_BD_cpp as L
    from brian2 import ms, mV
    p = L.get_Neuronparam()
    assert float(p['t_ref'] / ms) == pytest.approx(3.0) and float(p['VD'] / mV) == pytest.approx(-40.0)


# ---------------------------------------------------------------------------
# spec oracles (1) and (2): paired runs through the production worker
# ---------------------------------------------------------------------------
def test_paired_runs_clamped_source_equals_clamp_flag(lcdup_folder, tmp_path):
    """check_refractory_blowup.py (Block 1's oracle): the repository's clamped file gives 0 NaN
    neurons, and the same spike counts as an unclamped copy with the flag added by --clamp;
    the unclamped copy itself loses neurons (sensitivity of the check)."""
    import json
    import shutil
    unclamped = tmp_path / 'unclamped'
    unclamped.mkdir()
    for f in ('HPC_main_sweep.py', 'HPC_single_run.py', 'ASD_fun_BD_cpp.py', 'synapse_pdist.csv'):
        shutil.copy2(os.path.join(lcdup_folder, f), str(unclamped))
    p = unclamped / 'ASD_fun_BD_cpp.py'
    p.write_text(p.read_text(encoding='utf-8').replace(' : volt (unless refractory)\n', ' : volt\n'),
                 encoding='utf-8')
    tool = os.path.join(REPO, 'hpc', 'refractory_audit', 'checks', 'check_refractory_blowup.py')
    common = ['--n_topo', '1', '--k', '3', '--simtime', '3', '--seed', '2026', '--work_dir', str(tmp_path / 'w')]
    from conftest import run
    res = {}
    for label, d, extra in (('src', lcdup_folder, ['--allow_clamped_source']), ('flag', str(unclamped), ['--clamp']),
                            ('asis', str(unclamped), [])):
        r = run([sys.executable, tool, 'run', '--pipeline_dir', d, '--label', label] + extra + common,
                cwd=str(tmp_path), timeout=3000)
        assert r.returncode == 0, (label, r.stdout[-1500:], r.stderr[-1500:])
        res[label] = json.load(open(str(tmp_path / 'w' / ('result_%s.json' % label))))['records']
    assert [x['n_nan'] for x in res['src']] == [0, 0, 0]
    assert [x['n_nan'] for x in res['flag']] == [0, 0, 0]
    assert [x['n_spikes'] for x in res['src']] == [x['n_spikes'] for x in res['flag']]   # exact equality
    assert sum(x['n_nan'] for x in res['asis']) > 0
