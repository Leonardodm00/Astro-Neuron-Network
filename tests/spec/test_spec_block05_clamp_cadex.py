"""Block 5 -- clamp_cadex_copies, the cluster-wide clamp (SPEC.md Block 5), and
the repository invariant of Block 1 that it checks.

Oracles: the classification table of the spec on hand-made fixtures; the patch
rule (one comment line with the marker plus the flag, nothing else); the exit
codes; discovery depth and pruning; GIT CLONE; HELD; the qstat rules. Run:
    python -m pytest tests/spec/test_spec_block05_clamp_cadex.py -q
"""
import difflib
import os
import re
import stat
import subprocess
import sys
import time

import pytest

from conftest import AUDIT, REPO, run

TOOL = os.path.join(AUDIT, 'clamp_cadex_copies.py')
FLAG = '(unless refractory)'

CADEX_EQ = ("eqs_NN = Equations('''\n"
            "    dV/dt  = noise + (gl*(El-V) + gl*DeltaT*exp((V-VT)/DeltaT) + gA*(EA-V) - I_syn)/Cm : volt\n"
            "    dgA/dt = (gbarA/(1 + exp((VA-V)/DeltaA)) - gA)/tauA : siemens\n"
            "    ''')\n")
HH_EQ = ("eqs = '''\n    dV/dt = noise + (-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)+I-I_syn)/Cm  : volt\n'''\n")


def write(path, text, mode='w'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as fh:
        fh.write(text)
    return path


def tool(*args, cwd=None, env=None):
    e = {'PATH': '/usr/bin:/bin'}      # no qstat on PATH unless a test adds a stub
    if env:
        e.update(env)
    return run([sys.executable, TOOL] + list(args), cwd=cwd, env=e)


def rows_of(stdout):
    """{path: status} from the report table."""
    out = {}
    for line in stdout.splitlines():
        m = re.match(r'^(SKIP|CLAMPED|NOT CLAMPED|REFUSED|GIT CLONE|HELD|PATCHED|FAILED)\s+'
                     r'(?:-|[0-9.]+ h)\s+(/\S.*)$', line)      # the 'output' column: '-' or '<age> h'
        if m:
            out[m.group(2).strip()] = m.group(1)
    return out


@pytest.fixture
def tree(tmp_path):
    t = tmp_path / 'ANN'
    t.mkdir()
    return t


def test_repository_invariant_on_this_commit():
    """Block 1: exit 0; the root and Giulia/ models CLAMPED; hpc/ASD_fun_BD_cpp.py SKIP; nothing else."""
    r = tool(REPO)
    rows = rows_of(r.stdout)
    want = {os.path.join(REPO, 'hpc', 'Phenomenological_finalv1', 'ASD_fun_BD_cpp.py'): 'CLAMPED',
            os.path.join(REPO, 'hpc', 'Phenomenological_finalv1', 'Giulia', 'ASD_fun_BD_cpp.py'): 'CLAMPED',
            os.path.join(REPO, 'hpc', 'ASD_fun_BD_cpp.py'): 'SKIP'}
    assert r.returncode == 0, r.stdout
    assert rows == want


def test_independent_grep_of_the_invariant():
    """Block 1 without the tool: every tracked ASD_fun_BD_cpp.py, the CAdEx dV/dt line ends with the flag."""
    files = subprocess.run(['git', 'ls-files', '*ASD_fun_BD_cpp.py'], cwd=REPO, stdout=subprocess.PIPE,
                           universal_newlines=True).stdout.split()
    assert files, 'no model file tracked'
    n_cadex = 0
    for f in files:
        text = open(os.path.join(REPO, f), encoding='utf-8').read()
        dv = [ln for ln in text.splitlines() if re.match(r'^\s*dV/dt\s*=', ln)]
        cadex = [ln for ln in dv if 'exp((V-VT)/DeltaT)' in ln.replace(' ', '')]
        if cadex:
            n_cadex += 1
            assert len(cadex) == 1, f
            assert cadex[0].rstrip().endswith(FLAG), f
        else:
            assert not any(FLAG in ln for ln in dv), 'HH model carries the flag: %s' % f
    assert n_cadex == 2


@pytest.mark.parametrize('name,text,status', [
    ('skip_hh', HH_EQ, 'SKIP'),
    ('clamped', CADEX_EQ.replace(': volt\n', ': volt ' + FLAG + '\n', 1), 'CLAMPED'),
    ('not_clamped', CADEX_EQ, 'NOT CLAMPED'),
    ('two_lines', CADEX_EQ + CADEX_EQ, 'REFUSED'),
    ('other_flag', CADEX_EQ.replace(': volt\n', ': volt (constant over dt)\n', 1), 'REFUSED'),
    ('marker_no_flag', '# [refractory clamp] held\n' + CADEX_EQ, 'REFUSED'),
    ('term_no_line', "x = 'gl*DeltaT*exp((V-VT)/DeltaT)'\n", 'REFUSED'),
    ('spaces_in_term', CADEX_EQ.replace('exp((V-VT)/DeltaT)', 'exp(( V - VT ) / DeltaT)'), 'NOT CLAMPED'),
])
def test_classification(tree, name, text, status):
    path = write(str(tree / name / 'ASD_fun_BD_cpp.py'), text)
    r = tool(str(tree / name))
    assert rows_of(r.stdout).get(path) == status, r.stdout


def test_refused_bytes(tree):
    p1 = write(str(tree / 'crlf' / 'ASD_fun_BD_cpp.py'), CADEX_EQ.replace('\n', '\r\n'), mode='w')
    p2 = str(tree / 'latin' / 'ASD_fun_BD_cpp.py')
    os.makedirs(os.path.dirname(p2))
    with open(p2, 'wb') as fh:
        fh.write(CADEX_EQ.encode('utf-8') + b'# \xe9\n')
    r = tool(str(tree))
    rows = rows_of(r.stdout)
    assert rows.get(p1) == 'REFUSED' and rows.get(p2) == 'REFUSED'
    assert r.returncode == 1


def test_exit_codes(tree, tmp_path):
    write(str(tree / 'a' / 'ASD_fun_BD_cpp.py'), CADEX_EQ.replace(': volt\n', ': volt ' + FLAG + '\n', 1))
    assert tool(str(tree)).returncode == 0
    write(str(tree / 'b' / 'ASD_fun_BD_cpp.py'), CADEX_EQ)
    assert tool(str(tree)).returncode == 3                     # report mode, one NOT CLAMPED
    assert tool(str(tmp_path / 'missing')).returncode == 1     # missing root
    only_hh = tmp_path / 'hh'
    write(str(only_hh / 'ASD_fun_BD_cpp.py'), HH_EQ)
    assert tool(str(only_hh)).returncode == 1                  # no CAdEx copy at all
    # a missing root next to a good one still fails
    assert tool(str(tree / 'a'), str(tmp_path / 'missing')).returncode == 1


def test_discovery_depth_and_pruning(tree):
    good = CADEX_EQ.replace(': volt\n', ': volt ' + FLAG + '\n', 1)
    found = [write(str(tree / 'p1' / 'ASD_fun_BD_cpp.py'), good),
             write(str(tree / 'a/b/c/d/e/f' / 'ASD_fun_BD_cpp.py'), good)]           # depth 6
    hidden = [write(str(tree / 'a/b/c/d/e/f/g' / 'ASD_fun_BD_cpp.py'), CADEX_EQ),  # depth 7
              write(str(tree / 'campaign_x' / 'ASD_fun_BD_cpp.py'), CADEX_EQ),
              write(str(tree / 'scratch1' / 'ASD_fun_BD_cpp.py'), CADEX_EQ),
              write(str(tree / 'mea_out2' / 'ASD_fun_BD_cpp.py'), CADEX_EQ),
              write(str(tree / '.hidden' / 'ASD_fun_BD_cpp.py'), CADEX_EQ),
              write(str(tree / '__pycache__' / 'ASD_fun_BD_cpp.py'), CADEX_EQ)]
    r = tool(str(tree))
    rows = rows_of(r.stdout)
    assert sorted(rows) == sorted(found), r.stdout
    assert r.returncode == 0
    rows7 = rows_of(tool(str(tree), '--max-depth', '7').stdout)
    assert hidden[0] in rows7


def test_apply_patch_rule(tree):
    path = write(str(tree / 'p' / 'ASD_fun_BD_cpp.py'), '# header\n' + CADEX_EQ + 'x = 1\n')
    os.chmod(path, 0o640)
    before = open(path).read()
    r = tool(str(tree), '--apply', '--force-running')
    assert r.returncode == 0, r.stdout
    assert rows_of(r.stdout).get(path) == 'PATCHED'
    after = open(path).read()
    diff = [d for d in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm='', n=0)
            if d[:1] in '+-' and not d.startswith(('+++', '---'))]
    added = [d for d in diff if d.startswith('+')]
    removed = [d for d in diff if d.startswith('-')]
    assert len(removed) == 1 and len(added) == 2, diff
    comment = [a for a in added if a[1:].lstrip().startswith('#')]
    assert len(comment) == 1 and '[refractory clamp]' in comment[0]
    flagged = [a for a in added if not a[1:].lstrip().startswith('#')][0]
    assert flagged[1:] == removed[0][1:] + ' ' + FLAG
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640
    backups = [f for f in os.listdir(os.path.dirname(path)) if f.startswith('ASD_fun_BD_cpp.py.bak-clamp-')]
    assert len(backups) == 1 and re.match(r'ASD_fun_BD_cpp\.py\.bak-clamp-\d{8}_\d{6}$', backups[0])
    assert open(os.path.join(os.path.dirname(path), backups[0])).read() == before
    r2 = tool(str(tree), '--apply', '--force-running')                 # idempotent
    assert r2.returncode == 0 and open(path).read() == after


def test_held_by_recent_campaign_output(tree):
    path = write(str(tree / 'p' / 'ASD_fun_BD_cpp.py'), CADEX_EQ)
    os.makedirs(str(tree / 'p' / 'campaign_running' / 'sweep_x'))
    r = tool(str(tree), '--apply')
    assert rows_of(r.stdout).get(path) == 'HELD' and r.returncode == 1
    assert open(path).read() == CADEX_EQ
    old = time.time() - 7 * 3600                                      # older than H = 6 h
    for d in (tree / 'p' / 'campaign_running', tree / 'p' / 'campaign_running' / 'sweep_x'):
        os.utime(str(d), (old, old))
    r = tool(str(tree), '--apply')
    assert rows_of(r.stdout).get(path) == 'PATCHED', r.stdout


def test_git_clone_never_edited(tmp_path):
    clone = tmp_path / 'clone'
    clone.mkdir()
    path = write(str(clone / 'm' / 'ASD_fun_BD_cpp.py'), CADEX_EQ)
    for cmd in (['git', 'init', '-q'], ['git', 'add', '.'],
                ['git', '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'x']):
        subprocess.run(cmd, cwd=str(clone), check=True)
    for extra in ([], ['--apply', '--force-running']):
        r = tool(str(clone), *extra)
        assert rows_of(r.stdout).get(path) == 'GIT CLONE' and r.returncode == 1, r.stdout
    assert open(path).read() == CADEX_EQ


QSTAT_STUB = r'''#!/bin/sh
if [ "$1" = "-u" ]; then
  echo "Job id            Name             User              Time Use S Queue"
  echo "----------------  ---------------- ----------------  -------- - -----"
  echo "1234.davinci      lcdup_campaign   me                00:00:00 Q cpu"
  exit 0
fi
printf 'Job Id: 1234.davinci\n    Job_Name = lcdup_campaign\n    job_state = Q\n    Variable_List = PBS_O_HOME=/home/me,PBS_O_LANG=C,\n\tPBS_O_WORKDIR=%s,REPO_DIR=/nowhere\n' "$QDIR"
'''


def test_queued_job_holds_its_folder_wrapped_variable_list(tree, tmp_path):
    path = write(str(tree / 'p' / 'ASD_fun_BD_cpp.py'), CADEX_EQ)
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    write(str(bindir / 'qstat'), QSTAT_STUB)
    os.chmod(str(bindir / 'qstat'), 0o755)
    env = {'PATH': '%s:/usr/bin:/bin' % bindir, 'QDIR': str(tree / 'p'), 'USER': 'me'}
    r = tool(str(tree), '--apply', env=env)
    assert rows_of(r.stdout).get(path) == 'HELD', r.stdout
    assert open(path).read() == CADEX_EQ
    env['QDIR'] = str(tmp_path / 'elsewhere')
    r = tool(str(tree), '--apply', env=env)
    assert rows_of(r.stdout).get(path) == 'PATCHED', r.stdout


def test_no_qstat_warns_and_fails_open(tree):
    path = write(str(tree / 'p' / 'ASD_fun_BD_cpp.py'), CADEX_EQ)
    r = tool(str(tree), '--apply')
    assert 'WARNING' in r.stdout and 'qstat' in r.stdout
    assert 'WARNING' in [ln for ln in r.stdout.splitlines() if ln.startswith('verdict')][0]
    assert rows_of(r.stdout).get(path) == 'PATCHED'
