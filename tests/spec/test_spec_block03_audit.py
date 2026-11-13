"""Block 3 -- the audit driver and its PBS job (SPEC.md Block 3).

Oracles: audit.log's four sections; CODE CLAMPED / NOT CLAMPED / missing for
the model named by each campaign's job_args.json lib_dir; the LOGS count of
Brian2's NaN warning in $HOME; DATA = Block 2's files per campaign; the
output folder next to the script and nothing written elsewhere; an unknown
option refused. The PBS job's roadmap marker: only inside a PBS job, pushed
to the hpc-status branch of origin, and never changing the job's exit status.
Run:
    python -m pytest tests/spec/test_spec_block03_audit.py -q
"""
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import AUDIT, REPO, run

CLAMPED = ("    dV/dt  = noise + (gl*(El-V) + gl*DeltaT*exp((V-VT)/DeltaT) - I_syn)/Cm : volt (unless refractory)\n"
           "    't_ref':   3*ms,\n")
UNCLAMPED = CLAMPED.replace(' (unless refractory)', '')


def campaign(root, name, lib_dir, n_runs=3, T=10.0, Nn=5):
    c = os.path.join(root, name)
    td = os.path.join(c, 'sweep_cpu_task0000')
    os.makedirs(os.path.join(td, 'topo_00000'))
    json.dump(dict(simtime=T, Nn=Nn, lib_dir=lib_dir), open(os.path.join(td, 'job_args.json'), 'w'))
    rng = np.random.default_rng(len(name))
    for k in range(n_runs):
        t = rng.uniform(0, T, 40).astype(np.float32)
        i = rng.integers(0, Nn, 40).astype(np.int32)
        np.savez(os.path.join(td, 'topo_00000', 'iter_%05d.npz' % k), spk_N_t=t, spk_N_i=i)
    return c


@pytest.fixture
def setup(tmp_path):
    audit = tmp_path / 'audit'
    audit.mkdir()
    for f in ('check_refractory_nan.sh', 'refractory_scan.py'):
        shutil.copy2(os.path.join(AUDIT, f), str(audit))
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'lcdup_campaign.e101').write_text("WARNING Neuron's variable 'V' has NaN values\n")
    (home / 'lcdup_campaign.o101').write_text('ok\n')
    (home / 'gg_campaign.e7').write_text("Astrocyte's variable 'C' has NaN\n")
    libs = tmp_path / 'libs'
    for name, text in (('clamped', CLAMPED), ('unclamped', UNCLAMPED)):
        (libs / name).mkdir(parents=True)
        (libs / name / 'ASD_fun_BD_cpp.py').write_text(text)
    camps = tmp_path / 'camps'
    camps.mkdir()
    cs = [campaign(str(camps), 'campaign_a', str(libs / 'clamped')),
          campaign(str(camps), 'campaign_b', str(libs / 'unclamped')),
          campaign(str(camps), 'campaign_c', str(tmp_path / 'gone'))]
    return tmp_path, audit, home, cs


def listing(root):
    return sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs)


def test_audit_log_sections_and_verdict(setup):
    tmp, audit, home, cs = setup
    before = listing(str(tmp))
    r = run(['bash', str(audit / 'check_refractory_nan.sh'), '--sample', '0', '--workers', '1'] + cs,
            cwd=str(tmp), env={'HOME': str(home)})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    outs = glob.glob(str(audit / 'refractory_audit_*'))
    assert len(outs) == 1                                      # next to the script
    log = open(os.path.join(outs[0], 'audit.log')).read()
    for sec in ('=== 1. CODE', '=== 2. LOGS', '=== 3. DATA', '=== VERDICT'):
        assert sec in log, sec
    code = log[log.index('=== 1. CODE'):log.index('=== 2. LOGS')]
    assert 'campaign_a: CLAMPED' in code and 'campaign_b: NOT CLAMPED' in code
    assert 'campaign_c: ASD_fun_BD_cpp.py NOT FOUND' in code
    logs = log[log.index('=== 2. LOGS'):log.index('=== 3. DATA')]
    row = [ln.split() for ln in logs.splitlines() if ln.strip().startswith('lcdup_campaign')]
    assert row and row[0][1:] == ['2', '1', '0']                # 2 logs, 1 with V NaN, 0 astrocyte NaN
    for name in ('campaign_a', 'campaign_b', 'campaign_c'):
        for suf in ('_runs.csv', '_summary.json', '_summary.txt'):
            assert os.path.isfile(os.path.join(outs[0], name + suf)), name + suf
    after = [p for p in listing(str(tmp)) if not p.startswith(os.path.relpath(outs[0], str(tmp)))]
    assert after == before                                     # nothing written outside the output folder


@pytest.mark.parametrize('args', [['--bogus'], ['--sample', 'x'], ['--workers', '0']])
def test_refusals(setup, args):
    tmp, audit, home, cs = setup
    r = run(['bash', str(audit / 'check_refractory_nan.sh')] + args + cs, cwd=str(tmp), env={'HOME': str(home)})
    assert r.returncode == 2


# ---------------------------------------------------------------------------
# submit_refractory_scan.sh and the roadmap marker
# ---------------------------------------------------------------------------
CONDA_SH = '''conda() {
  case "$1" in
    activate) export CONDA_PREFIX="%(base)s/envs/$2"; export PATH="$CONDA_PREFIX/bin:$PATH" ;;
    deactivate) : ;;
    info) echo "%(base)s" ;;
  esac
}
'''


@pytest.fixture
def jobrepo(tmp_path):
    origin = tmp_path / 'origin.git'
    subprocess.run(['git', 'init', '-q', '--bare', str(origin)], check=True)
    clone = tmp_path / 'clone'
    (clone / 'hpc' / 'refractory_audit').mkdir(parents=True)
    (clone / 'specs' / 'roadmap').mkdir(parents=True)
    shutil.copy2(os.path.join(AUDIT, 'submit_refractory_scan.sh'), str(clone / 'hpc' / 'refractory_audit'))
    shutil.copy2(os.path.join(REPO, 'specs', 'roadmap', 'hpc_status.sh'), str(clone / 'specs' / 'roadmap'))
    shutil.copy2(os.path.join(REPO, '.gitignore'), str(clone))
    g = ['git', '-c', 'user.name=t', '-c', 'user.email=t@t']
    for cmd in (['git', 'init', '-q', '-b', 'main'], ['git', 'add', '.'], g + ['commit', '-qm', 'x'],
                ['git', 'remote', 'add', 'origin', str(origin)], ['git', 'push', '-q', 'origin', 'main']):
        subprocess.run(cmd, cwd=str(clone), check=True)
    base = tmp_path / 'conda'
    (base / 'etc' / 'profile.d').mkdir(parents=True)
    (base / 'etc' / 'profile.d' / 'conda.sh').write_text(CONDA_SH % dict(base=base))
    (base / 'envs' / 'brian_final' / 'bin').mkdir(parents=True)
    os.symlink(sys.executable, str(base / 'envs' / 'brian_final' / 'bin' / 'python'))
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    (bindir / 'conda').write_text('#!/bin/sh\necho "%s"\n' % base)
    os.chmod(str(bindir / 'conda'), 0o755)
    return tmp_path, origin, clone, bindir


def job(jobrepo, scan_exit, **env):
    tmp, origin, clone, bindir = jobrepo
    stub = clone / 'hpc' / 'refractory_audit' / 'refractory_scan.py'
    stub.write_text('import sys\nsys.exit(%d)\n' % scan_exit)
    e = dict(PATH='%s:/usr/bin:/bin' % bindir, REPO_DIR=str(clone / 'hpc' / 'refractory_audit'),
             CAMPAIGNS='/nowhere', HOME=str(tmp))
    e.update(env)
    r = run(['bash', str(clone / 'hpc' / 'refractory_audit' / 'submit_refractory_scan.sh')],
            cwd=str(clone / 'hpc' / 'refractory_audit'), env=e)
    show = subprocess.run(['git', '--git-dir', str(origin), 'ls-tree', '-r', '--name-only', 'hpc-status'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    markers = [p for p in show.stdout.split() if p.startswith('runs/I3/')]
    content = {}
    for p in markers:
        content[p] = json.loads(subprocess.run(['git', '--git-dir', str(origin), 'show', 'hpc-status:' + p],
                                               stdout=subprocess.PIPE, universal_newlines=True).stdout)
    return r, content


def test_marker_failed_keeps_exit_status(jobrepo):
    r, markers = job(jobrepo, 3, PBS_JOBID='555.server')
    assert r.returncode == 3, r.stdout[-1500:] + r.stderr[-1500:]
    assert list(markers) == ['runs/I3/555.server.json']
    m = markers['runs/I3/555.server.json']
    assert m['state'] == 'failed' and m['last_exit'] == 3


def test_marker_done(jobrepo):
    r, markers = job(jobrepo, 0, PBS_JOBID='556.server')
    assert r.returncode == 0
    assert markers['runs/I3/556.server.json']['state'] == 'done'


@pytest.mark.parametrize('env', [dict(PBS_JOBID='557.server', ROADMAP_DISABLE='1'), dict()])
def test_no_marker_outside_pbs_or_disabled(jobrepo, env):
    r, markers = job(jobrepo, 3, **env)
    assert r.returncode == 3 and markers == {}
