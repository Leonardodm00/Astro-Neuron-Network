"""Block 7 -- the seed blocks of launch_campaign.sh and MEAMA/launch_MEAMA.sh
(SPEC.md Block 7; LindaColloDUP's launcher is in test_spec_block04_lcdup.py).

Oracles: refuse (exit 5, nothing submitted) an allocator without --scan-tree
and a missing tree; with a tree, print the guard header and allocate a span
clear of the seeds recorded under it; a dry run still reserves its span. Run:
    python -m pytest tests/spec/test_spec_block07_launchers.py -q
"""
import json
import os
import shutil

import pytest

from conftest import PIPE, PIPELINE_FILES, copy_tree, run


def stub_qsub(tmp_path):
    b = tmp_path / 'bin'
    b.mkdir(exist_ok=True)
    (b / 'qsub').write_text('#!/bin/sh\necho "$@" >> "%s"\necho 4242.server\n' % (tmp_path / 'qsub.log'))
    os.chmod(str(b / 'qsub'), 0o755)
    return '%s:%s' % (b, os.environ['PATH'])


def ann_tree(tmp_path, seeds=(1500000000,)):
    t = tmp_path / 'ANN'
    d = t / 'old' / 'campaign_old' / 'sweep_cpu_task0000'
    d.mkdir(parents=True, exist_ok=True)
    json.dump({'_resolved_seed_master': seeds[0]}, open(str(d / 'job_args.json'), 'w'))
    return t


@pytest.fixture
def meama(tmp_path):
    d = copy_tree(os.path.join(PIPE, 'MEAMA'), str(tmp_path / 'MEAMA'))
    for f in PIPELINE_FILES:
        shutil.copy2(os.path.join(PIPE, f), d)
    return d


@pytest.fixture
def root(tmp_path):
    d = str(tmp_path / 'root')
    os.makedirs(d)
    for f in PIPELINE_FILES + ('launch_campaign.sh', 'submit_sweep_mixed.sh', 'seed_alloc.py'):
        shutil.copy2(os.path.join(PIPE, f), d)
    return d


def meama_launch(d, tmp_path, **env):
    e = dict(SIZING_CONFIRMED='1', DRYRUN='1', SEED_SCAN_TREE=str(ann_tree(tmp_path)), PATH=stub_qsub(tmp_path))
    e.update(env)
    return run(['bash', os.path.join(d, 'launch_MEAMA.sh')], cwd=d, env=e, timeout=300)


def test_meama_guard(meama, tmp_path):
    r = meama_launch(meama, tmp_path)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert 'Seed guard     : this folder and every launch folder under %s' % (tmp_path / 'ANN') in r.stdout
    assert not (tmp_path / 'qsub.log').exists()                         # dry run
    led = open(os.path.join(meama, 'artifacts', 'seed_ledger.tsv')).read().splitlines()[-1].split('\t')
    lo, hi = int(led[5]), int(led[6])
    assert not lo <= 1500000000 < hi and 'DRYRUN-not-submitted' in led[7]
    assert meama_launch(meama, tmp_path, SEED_SCAN_TREE=str(tmp_path / 'none')).returncode == 5
    txt = open(os.path.join(meama, 'seed_alloc.py')).read().replace('--scan-tree', '--scan-trees')
    open(os.path.join(meama, 'seed_alloc.py'), 'w').write(txt)
    assert meama_launch(meama, tmp_path).returncode == 5


def test_campaign_launcher_guard(root, tmp_path):
    env = dict(SEED_SCAN_TREE=str(tmp_path / 'none'), PATH=stub_qsub(tmp_path))
    r = run(['bash', os.path.join(root, 'launch_campaign.sh')], cwd=root, env=env, timeout=300)
    assert r.returncode == 5 and not (tmp_path / 'qsub.log').exists(), r.stdout[-800:] + r.stderr[-800:]
    txt = open(os.path.join(root, 'seed_alloc.py')).read().replace('--scan-tree', '--scan-trees')
    open(os.path.join(root, 'seed_alloc.py'), 'w').write(txt)
    env['SEED_SCAN_TREE'] = str(ann_tree(tmp_path))
    r = run(['bash', os.path.join(root, 'launch_campaign.sh')], cwd=root, env=env, timeout=300)
    assert r.returncode == 5 and not (tmp_path / 'qsub.log').exists()
    shutil.copy2(os.path.join(PIPE, 'seed_alloc.py'), root)
    r = run(['bash', os.path.join(root, 'launch_campaign.sh')], cwd=root, env=env, timeout=300)
    assert 'Seed guard  : this folder and every launch folder under %s' % (tmp_path / 'ANN') in r.stdout, \
        r.stdout[-1500:] + r.stderr[-1500:]
