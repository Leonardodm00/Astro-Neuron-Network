"""Block 9 -- GiuliaGate launch package: prior_config.tsv and apply_prior_patch.py
(SPEC.md Block 9 with its corrections of 2026-10-31, 2026-11-03, 2026-11-04).

Oracles: the decided values written out below from the spec; the scale rule;
the registry nominals and boxes; the patch touches exactly the three rows the
spec names; every refusal leaves the target byte-identical. Run:
    python -m pytest tests/spec/test_spec_block09_giuliagate.py -q
"""
import difflib
import os
import shutil
import sys
from fractions import Fraction

import pytest

from conftest import GG, PIPE, package_with_pipeline, run

sys.path.insert(0, PIPE)
import HPC_main_sweep as S      # noqa: E402

TSV = os.path.join(GG, 'prior_config.tsv')
FREE = {'Sigma': (2, 10, 'lin'), 'U_0_sr': (0.01, 0.7, 'log'), 'Omega_f_sr': (0.1, 4.5, 'log'),
        'Omega_d': (0.1, 4.5, 'log'), 'alpha_syn': (0.1, 1, 'log'), 'g_ampa': (0.05, 10, 'log'),
        'g_nmda': (0.05, 10, 'log'), 'x0': (0.05, 0.5, 'log'), 'O_G': (0.1, 10, 'log'),
        'Omega_G': (0.001, 0.1, 'log'), 'I_bias': (0.3, 1, 'lin'), 'F': (0.1, 10, 'log'),
        'U_A': (0.1, 0.9, 'lin'), 'G_T': (50, 1000, 'log'), 'O_N': (0.0001, 1, 'log'),
        'g_sic': (4, 400, 'log')}
FIXED = {'gbarA': 0.9, 'delta_gA': 0.135, 'tauA': 200, 'VA': -48, 'DeltaA': 5, 'VR': -58.2, 'I_inj': 4,
         'Cm': 17.25, 'DeltaT': 2, 'VT': -51, 'gL': 1.15, 'EC50_ampa': 8.6, 'EC50_nmda': 3,
         'U_0_ar': 0.003, 'U_max': 0.5, 'Omega_f_ar': 1.42857, 'O_beta': 1, 'O_3K': 4.5,
         'Omega_5P': 0.1, 'I_Theta': 0.3, 'omega_I': 0.05, 'C_Theta': 0.5}
TOPOLOGY = {'conn_prob': (0.05, 0.4), 'stoa_gate_p': (0.2, 1)}
CHANGED = {'Sigma': ((3, 15), (2, 10)), 'g_ampa': ((0.01, 5), (0.05, 10)), 'g_nmda': ((0.01, 5), (0.05, 10))}


def read_tsv(path=TSV):
    raw = open(path, 'rb').read()
    lines = [ln for ln in raw.decode('ascii').split('\n') if ln and not ln.startswith('#')]
    header = lines[0].split('\t')
    return raw, header, [dict(zip(header, ln.split('\t'))) for ln in lines[1:]]


def scale_rule(lo, hi):
    lo, hi = Fraction(str(lo)), Fraction(str(hi))
    return 'log' if lo > 0 and hi / lo >= 10 else 'lin'


def test_tsv_bytes_header_rows():
    raw, header, rows = read_tsv()
    assert b'\r' not in raw and max(raw) < 128
    assert header == ['axis', 'group', 'unit', 'status', 'low', 'high', 'value', 'scale', 'registry',
                      'giulia_copy', 'meaning']
    axes = [r['axis'] for r in rows]
    assert len(rows) == 40 and len(set(axes)) == 40
    assert set(axes) == set(S.PARAM_NAMES) | {'conn_prob', 'stoa_gate_p'}


def test_tsv_decided_values():
    _, _, rows = read_tsv()
    by = {r['axis']: r for r in rows}
    assert {a for a, r in by.items() if r['status'] == 'free' and r['group'] != 'topology'} == set(FREE)
    assert {a for a, r in by.items() if r['status'] == 'fixed'} == set(FIXED)
    for a, (lo, hi, sc) in FREE.items():
        assert (float(by[a]['low']), float(by[a]['high'])) == (lo, hi), a
        assert scale_rule(by[a]['low'], by[a]['high']) == sc == by[a]['scale'], a
    for a, v in FIXED.items():
        assert float(by[a]['value']) == v, a
    for a, (lo, hi) in TOPOLOGY.items():
        assert by[a]['group'] == 'topology' and (float(by[a]['low']), float(by[a]['high'])) == (lo, hi)


def test_tsv_against_registry():
    """Fixed axes at the registry nominals; free boxes = registry boxes except Sigma, g_ampa, g_nmda."""
    for a, v in FIXED.items():
        assert S.NOMINAL_PARAMS[S.PARAM_NAMES.index(a)] == pytest.approx(v, rel=0, abs=0), a
    for a, (lo, hi, _) in FREE.items():
        reg = tuple(float(x) for x in S.PARAM_BOUNDS[S.PARAM_NAMES.index(a)])
        if a in CHANGED:
            assert reg == tuple(float(x) for x in CHANGED[a][0]), a
        else:
            assert reg == (lo, hi), a


@pytest.fixture
def fresh(tmp_path):
    return package_with_pipeline(GG, tmp_path)


def patch(folder, *args):
    return run([sys.executable, 'apply_prior_patch.py'] + list(args), cwd=folder)


def test_patch_changes_exactly_three_rows(fresh):
    before = open(os.path.join(fresh, 'HPC_main_sweep.py')).read()
    r = patch(fresh)
    assert r.returncode == 0 and 'PATCHED' in r.stdout, r.stdout + r.stderr
    after = open(os.path.join(fresh, 'HPC_main_sweep.py')).read()
    diff = [d for d in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm='', n=0)
            if d[:1] in '+-' and not d.startswith(('+++', '---'))]
    plus = [d for d in diff if d.startswith('+')]
    assert len(plus) == 3 and len(diff) == 6, diff
    for name, (old, new) in CHANGED.items():
        line = [d for d in plus if '# %d ' % S.PARAM_NAMES.index(name) in d or name in d]
        assert len(line) == 1, (name, plus)
        assert '[GiuliaGate box 2026-10-31] registry' in line[0]
    assert patch(fresh, '--check-applied').returncode == 0
    ca = patch(fresh, '--check-applied').stdout
    assert 'BOXES OK: 16 free axes' in ca
    r2 = patch(fresh)                                          # idempotent
    assert r2.returncode == 0 and 'ALREADY APPLIED' in r2.stdout
    assert open(os.path.join(fresh, 'HPC_main_sweep.py')).read() == after
    # the patched module: the three new boxes, tripartite minus the freeze list = the 16 free axes
    code = ("import sys; sys.path.insert(0, '.'); import HPC_main_sweep as M; import json; "
            "print(json.dumps([[float(x) for x in M.PARAM_BOUNDS[M.PARAM_NAMES.index(n)]] "
            "for n in ('Sigma','g_ampa','g_nmda')]))")
    out = run([sys.executable, '-c', code], cwd=fresh).stdout.strip().splitlines()[-1]
    assert out == '[[2.0, 10.0], [0.05, 10.0], [0.05, 10.0]]'


def test_freeze_list_and_topology(fresh):
    fl = patch(fresh, '--freeze-list').stdout.strip()
    items = dict(x.split('=') for x in fl.split(','))
    trip = {S.PARAM_NAMES[i] for i in S.resolve_sweep_group('tripartite')}
    assert set(items) == {a for a in FIXED if a in trip}
    assert len(items) == 17
    for a, v in items.items():
        assert float(v) == FIXED[a]
    assert patch(fresh, '--topology', 'conn_prob').stdout.split() in (['0.05', '0.4'],)
    lo, hi = (float(x) for x in patch(fresh, '--topology', 'stoa_gate_p').stdout.split())
    assert (lo, hi) == (0.2, 1.0)


def edited_tsv(tmp_path, fn, name):
    _, header, rows = read_tsv()
    rows = fn([dict(r) for r in rows])
    p = str(tmp_path / ('%s.tsv' % name))
    with open(p, 'w', newline='') as fh:
        fh.write('\t'.join(header) + '\n')
        for r in rows:
            fh.write('\t'.join(r[h] for h in header) + '\n')
    return p


def _set(axis, **kv):
    def f(rows):
        for r in rows:
            if r['axis'] == axis:
                r.update(kv)
        return rows
    return f


REFUSALS = {
    'missing_row': lambda rows: [r for r in rows if r['axis'] != 'x0'],
    'duplicate_row': lambda rows: rows + [dict(rows[0])],
    'bad_status': _set('x0', status='maybe'),
    'free_empty_box': _set('Sigma', low='10', high='2'),
    'fixed_not_number': _set('tauA', value='abc'),
    'frozen_off_registry': _set('DeltaT', value='2.5'),
    'I_Theta_free': _set('I_Theta', status='free', low='0.1', high='1', value='-'),
    'topology_box_outside': _set('stoa_gate_p', low='0.5', high='1.2'),
}


@pytest.mark.parametrize('case', sorted(REFUSALS))
def test_refusals_write_nothing(fresh, tmp_path, case):
    tsv = edited_tsv(tmp_path, REFUSALS[case], case)
    target = os.path.join(fresh, 'HPC_main_sweep.py')
    before, listing = open(target, 'rb').read(), sorted(os.listdir(fresh))
    r = patch(fresh, '--tsv', tsv, '--file', target)
    assert r.returncode == 1, (case, r.stdout, r.stderr)
    assert open(target, 'rb').read() == before and sorted(os.listdir(fresh)) == listing


def test_cr_bytes_in_tsv_refused(fresh, tmp_path):
    """SPEC Block 9 lists 'a file with CR bytes' among the TSV refusals (read here as the TSV;
    see the report's 'Problems in the spec': the sentence may also mean the patched file)."""
    p = str(tmp_path / 'crlf.tsv')
    open(p, 'wb').write(open(TSV, 'rb').read().replace(b'\n', b'\r\n'))
    target = os.path.join(fresh, 'HPC_main_sweep.py')
    before = open(target, 'rb').read()
    assert patch(fresh, '--tsv', p, '--file', target).returncode == 1
    assert open(target, 'rb').read() == before


def test_cr_bytes_in_patched_file_refused(fresh):
    target = os.path.join(fresh, 'HPC_main_sweep.py')
    open(target, 'wb').write(open(target, 'rb').read().replace(b'\n', b'\r\n'))
    before, listing = open(target, 'rb').read(), sorted(os.listdir(fresh))
    assert patch(fresh, '--file', target).returncode == 1
    assert open(target, 'rb').read() == before and sorted(os.listdir(fresh)) == listing


def test_patched_to_other_boxes_refused(fresh, tmp_path):
    assert patch(fresh).returncode == 0
    target = os.path.join(fresh, 'HPC_main_sweep.py')
    before = open(target, 'rb').read()
    tsv = edited_tsv(tmp_path, _set('Sigma', low='2', high='12'), 'other')
    r = patch(fresh, '--tsv', tsv, '--file', target)
    assert r.returncode == 1 and open(target, 'rb').read() == before


def test_self_test_writes_no_file(fresh, tmp_path):
    tmpdir = tmp_path / 'tmpdir'
    tmpdir.mkdir()
    listing = sorted(os.listdir(fresh))
    r = run([sys.executable, 'apply_prior_patch.py', '--self-test'], cwd=fresh, env={'TMPDIR': str(tmpdir)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert os.listdir(str(tmpdir)) == [] and sorted(os.listdir(fresh)) == listing
