"""Block boundaries (procedure step 7), the directive audit's swappability checks
(step 8), Block 6's launcher, and reproducers of the silent failures found in
step 9.

Boundaries checked on real producer output:
- sweep (Blocks 8, 11) -> refractory_scan (Block 2), in the campaign layout of section 3;
- range_check.py -> the sweep -> range_analysis (Block 6): labels equal the design;
- LindaColloDUP worker probe (Block 4) -> job_args.json -> the seed guard (Block 7).
Run (a few minutes: four small compiled sweeps):
    python -m pytest tests/spec/test_spec_boundaries_audit.py -q
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import pytest

from conftest import AUDIT, PIPE, RANGE, copy_tree, import_from, run

RS = import_from(AUDIT, 'refractory_scan')
SA = import_from(PIPE, 'seed_alloc')
RA = import_from(RANGE, 'range_analysis')


# ---------------------------------------------------------------------------
# Block 6 launcher (dry runs and refusals)
# ---------------------------------------------------------------------------
def rc_launch(lib, out, **env):
    e = dict(SETUP='lcdup', LIB_DIR=lib, OUT=out, DRYRUN='1')
    e.update(env)
    e = {k: v for k, v in e.items() if v is not None}
    return run(['bash', os.path.join(RANGE, 'launch_range_check.sh')], cwd=RANGE, env=e, timeout=300)


def qsub_lines(stdout):
    return [ln for ln in stdout.splitlines() if ln.startswith('[dry run] qsub')]


def test_launcher_two_nodes(lcdup_folder, tmp_path):
    r = rc_launch(lcdup_folder, str(tmp_path / 'out'))
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    q = qsub_lines(r.stdout)
    assert len(q) == 3
    assert '-q intel -N rc_corners ' in q[0] and 'PHASES=corners' in q[0] and 'ANALYZE=0' in q[0]
    assert '-q cpu -N rc_interior ' in q[1] and 'PHASES=interior' in q[1] and 'ANALYZE=0' in q[1]
    assert '-N rc_analysis' in q[2] and 'depend=afterany' in q[2] and 'PHASES=none,ANALYZE=1' in q[2]
    assert all('select=1:ncpus=48' in x and 'walltime=24:00:00' in x for x in q[:2])
    assert 'ncpus=8' in q[2] and re.search(r'walltime=0?2:00:00', q[2])     # ANALYSIS_NCPUS 8, 2 h


@pytest.mark.parametrize('bins,lo,hi', [(None, '0:1:2:3', '4:5:6'), ('5', '0:1:2', '3:4')])
def test_launcher_four_nodes(lcdup_folder, tmp_path, bins, lo, hi):
    r = rc_launch(lcdup_folder, str(tmp_path / 'out'), NODES='4', BINS=bins)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    q = qsub_lines(r.stdout)
    assert len(q) == 5
    assert '-N rc_corners_a' in q[0] and 'CORNER_LEVELS=0' in q[0] and '-q intel' in q[0]
    assert '-N rc_corners_b' in q[1] and 'CORNER_LEVELS=1' in q[1] and '-q intel' in q[1]
    assert '-N rc_interior_a' in q[2] and 'INTERIOR_LEVELS=%s' % lo in q[2] and '-q cpu' in q[2]
    assert '-N rc_interior_b' in q[3] and 'INTERIOR_LEVELS=%s' % hi in q[3] and '-q cpu' in q[3]
    assert 'depend=afterany:<the 4 jobs above>' in q[4]


@pytest.mark.parametrize('env', [dict(NODES='3'), dict(SETUP='other'), dict(NODES='4', CORNER_LEVELS='0'),
                                 dict(NODES='4', BINS='1'), dict(EXTRA='--a,b')])
def test_launcher_refusals(lcdup_folder, tmp_path, env):
    r = rc_launch(lcdup_folder, str(tmp_path / 'out'), **env)
    assert r.returncode == 2 and not qsub_lines(r.stdout), env


def test_launcher_comma_in_out(lcdup_folder, tmp_path):
    assert rc_launch(lcdup_folder, str(tmp_path / 'a,b')).returncode == 2


def test_launcher_refused_second_job_prints_qdel(lcdup_folder, tmp_path):
    stub = tmp_path / 'bin'
    stub.mkdir()
    (stub / 'qsub').write_text('#!/bin/sh\nn=$(cat %s/n 2>/dev/null || echo 0); n=$((n+1)); echo $n > %s/n\n'
                               '[ "$n" = "1" ] && { echo 101.server; exit 0; }\nexit 1\n' % (tmp_path, tmp_path))
    os.chmod(str(stub / 'qsub'), 0o755)
    r = rc_launch(lcdup_folder, str(tmp_path / 'out'), DRYRUN='0', PATH='%s:%s' % (stub, os.environ['PATH']))
    assert r.returncode == 3
    assert 'qdel 101.server' in (r.stdout + r.stderr)


# ---------------------------------------------------------------------------
# sweep -> refractory_scan
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp('camp')
    task = os.path.join(str(root), 'campaign_t', 'sweep_cpu_task0000')
    r = run([sys.executable, os.path.join(PIPE, 'HPC_main_sweep.py'), '--out_dir', task, '--mode', 'Neuronal',
             '--sweep_group', 'neuron_synapse', '--Nn', '12', '--c_max', '100', '--conn_prob_lo', '0.3',
             '--conn_prob_hi', '0.3', '--simtime', '0.5', '--n_topologies', '1', '--n_params_per_worker', '3',
             '--n_workers', '1', '--seed_master', '99', '--lib_dir', PIPE,
             '--scratch_root', os.path.join(str(root), 'scratch')], cwd=PIPE, timeout=3000)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return os.path.join(str(root), 'campaign_t')


def test_refractory_scan_reads_real_sweep_output(campaign, tmp_path):
    out = str(tmp_path / 'scan')
    RS.scan_campaign(campaign, out, sample=0, seed=1, workers=1, cfg=dict(RS.DEFAULT_CFG))
    rows = list(csv.DictReader(open(os.path.join(out, 'campaign_t_runs.csv'))))
    assert len(rows) == 3
    ja = json.load(open(os.path.join(campaign, 'sweep_cpu_task0000', 'job_args.json')))
    for row in rows:
        npz = os.path.join(campaign, row['task'], row['topo'], row['iter'] + '.npz')
        z = np.load(npz)
        side = json.load(open(npz[:-4] + '.json'))
        assert row['status'].startswith('ok') and row['nn_source'] == 'job_args'
        assert int(row['Nn']) == ja['Nn'] == 12 and float(row['T']) == ja['simtime'] == 0.5
        assert int(row['n_spikes']) == len(z['spk_N_t'])
        assert float(row['conn_prob']) == pytest.approx(0.3)
        for name, v in zip(side['param_names'], z['params']):
            assert float(row['par_' + name]) == pytest.approx(float(v), rel=1e-12), name


# ---------------------------------------------------------------------------
# range_check.py -> sweep -> range_analysis
# ---------------------------------------------------------------------------
RC_SETTINGS = ['--mode', 'Neuronal', '--sweep_group', 'neuron_synapse', '--Nn', '10', '--c_max', '100',
               '--conn_rule', 'flat', '--conn_prob_lo', '0.1', '--conn_prob_hi', '0.5', '--n_workers', '1']


@pytest.fixture(scope='module')
def rc_out(tmp_path_factory):
    out = str(tmp_path_factory.mktemp('rc') / 'out')
    for phase, n in (('corners', 3), ('interior', 2)):
        r = run([sys.executable, os.path.join(RANGE, 'range_check.py'), '--rc-phase', phase, '--rc-level', '1',
                 '--rc-out', out, '--rc-subset', str(n), '--rc-simtime', '2.5', '--lib_dir', PIPE] + RC_SETTINGS,
                cwd=RANGE, timeout=3000)
        assert r.returncode == 0, (phase, r.stdout[-2000:], r.stderr[-2000:])
    r = run([sys.executable, os.path.join(RANGE, 'range_analysis.py'), out, '--workers', '1'], cwd=RANGE)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return out


def test_range_analysis_labels_equal_the_design(rc_out):
    runs = list(csv.DictReader(open(os.path.join(rc_out, 'runs.csv'))))
    corners = [x for x in runs if x['phase'] == 'corners']
    interior = [x for x in runs if x['phase'] == 'interior']
    assert len(corners) == 3 and len(interior) == 2
    d = json.load(open(os.path.join(rc_out, 'corners_c1', 'design.json')))
    assert {x['kind'] for x in corners} == {'corner'} and {x['kind'] for x in interior} == {'interior'}
    got = sorted(tuple(int(float(x['code_' + n])) for n in d['free_axes']) for x in corners)
    assert got == sorted(tuple(c) for c in d['codes'][:3])
    assert all(int(float(x['code_conn_prob'])) == 1 for x in corners)
    di = json.load(open(os.path.join(rc_out, 'interior_b1', 'design.json')))
    got_i = sorted(tuple(int(float(x['bin_' + n])) for n in di['free_axes']) for x in interior)
    assert got_i == sorted(tuple(c) for c in di['cells'][:2])
    for x in corners:
        assert float(x['conn_prob']) == pytest.approx(0.5)                # level 1 = c_hi
    for x in interior:
        assert float(x['conn_prob']) == pytest.approx(0.1 + 1.5 * 0.4 / 7)  # bin 1 midpoint


def test_swappable_t_skip(rc_out, tmp_path):
    """Directive audit: one parameter changed through the CLI, the rest of the analysis still runs."""
    r = run([sys.executable, os.path.join(RANGE, 'range_analysis.py'), rc_out, '--workers', '1',
             '--t-skip', '1.0'], cwd=RANGE)
    assert r.returncode == 0 and 'window [1 s, T]' in r.stdout


# ---------------------------------------------------------------------------
# LindaColloDUP worker probe -> job_args.json -> seed guard
# ---------------------------------------------------------------------------
def test_lcdup_probe_records_and_seed_guard(lcdup_folder, tmp_path):
    d = copy_tree(lcdup_folder, str(tmp_path / 'LindaColloDUP'))
    env = dict(LCDUP_NO_ENV='1', NODETAG='probe', CAMPAIGN_TAG='probe_check', SEED_BASE='1000', N_TOPOLOGIES='1',
               C_MAX='120', SIMTIME='1', PBS_NCPUS='1',
               FREEZE_ENC='Cm:I_inj:VA@-48:DeltaA:VR@-58.2:EC50_ampa:EC50_nmda:alpha_syn:O_G:Omega_G')
    r = run(['bash', 'submit_LCDUP.sh'], cwd=d, env=env, timeout=3000)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "has NaN" not in r.stdout + r.stderr
    ja = json.load(open(glob.glob(os.path.join(d, 'campaign_probe_check', 'sweep_*', 'job_args.json'))[0]))
    assert len(ja['_active_param_names']) == 13 and ja['mode'] == 'Neuronal'
    ax = ja['_axis_declaration']
    for g in ('g_ampa', 'g_nmda'):
        assert ax['swept_axes'][g] == {'level': 'run_args', 'low': 0.01, 'high': 10.0, 'log': True}
    assert (ax['swept_axes']['conn_prob']['low'], ax['swept_axes']['conn_prob']['high']) == (0.01, 0.4)
    assert ax['fixed_axes']['VA'] == -48.0 and ax['fixed_axes']['VR'] == -58.2 and ax['fixed_axes']['I_inj'] == 4.0
    seeds, n_ok, n_bad = SA.used_seeds_from_job_args(d)
    assert seeds == {ja['_resolved_seed_master']} and n_bad == 0       # what the guard reads is what ran


def test_refractory_scan_cli_swappable(campaign, tmp_path):
    r = run([sys.executable, os.path.join(AUDIT, 'refractory_scan.py'), campaign, '--out', str(tmp_path / 'o'),
             '--sample', '0', '--workers', '1', '--t-early', '0.05'], cwd=str(tmp_path))
    assert r.returncode == 0
    s = json.load(open(str(tmp_path / 'o' / 'campaign_t_summary.json')))
    assert s['cfg']['t_early'] == 0.05


# ---------------------------------------------------------------------------
# step 9: reproducers of silent failures (expected to FAIL until fixed; see the report)
# ---------------------------------------------------------------------------
def test_ledger_with_a_bad_line_keeps_later_reservations(tmp_path):
    """Block 7 invariant: a span reserved in the ledger is never reused. One undecodable line
    must not hide the reservations written after it."""
    p = tmp_path / 'seed_ledger.tsv'
    rows = ['2026-01-01T00:00:00Z\tL1\tt1\t1000\t100\t1000\t1100\t-\n',
            '2026-01-01T00:00:00Z\tL2\tt\xe9\t2000\t100\t2000\t2100\t-\n',
            '2026-01-01T00:00:00Z\tL3\tt3\t3000\t100\t3000\t3100\t-\n']
    with open(str(p), 'wb') as fh:
        fh.write(SA.LEDGER_HEADER.encode('ascii'))
        fh.write(rows[0].encode('ascii'))
        fh.write(rows[1].encode('latin-1'))            # one non-UTF-8 byte in a tag
        fh.write(rows[2].encode('ascii'))
    spans = sorted((lo, hi) for lo, hi, _ in SA.reserved_spans_from_ledger(str(p)))
    assert (3000, 3100) in spans, spans


def test_run_shorter_than_t_skip_is_not_silently_silent():
    """Block 6 analysis: a run with T <= t_skip has no window; it must not be scored as a silent run."""
    bm = RA.import_burst_metrics(RA.find_burst_metrics())
    t = np.sort(np.random.default_rng(1).uniform(0, 1.5, 200)).astype(np.float32)
    i = np.random.default_rng(2).integers(0, 10, 200).astype(np.int32)
    try:
        m = RA.window_metrics(t, i, 10, 1.5, bm)
    except (ValueError, ZeroDivisionError):
        return                                          # refusing is acceptable
    assert not (m['frac_active'] == 0.0 and m['mean_rate_hz'] <= 0.0), m
