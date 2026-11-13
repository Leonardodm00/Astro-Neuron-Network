"""Block 7 -- the seed guard across ~/ANN (SPEC.md Block 7).

Oracles: byte-identical copies; what scan_tree reads and skips (depth, names,
symbolic links); the invariant of an allocation (no recorded seed inside the
span, no overlap with a reserved span), checked by brute force over many
allocations in a small seed space; only the own ledger written; the CLI's
refusal of a missing tree. Run:
    python -m pytest tests/spec/test_spec_block07_seed_alloc.py -q
"""
import filecmp
import json
import os
import sys

import pytest

from conftest import GG, LCDUP, PIPE, import_from, run

SA = import_from(PIPE, 'seed_alloc')


def test_copies_byte_identical():
    root = os.path.join(PIPE, 'seed_alloc.py')
    for d in (os.path.join(PIPE, 'MEAMA'), LCDUP, GG):
        assert filecmp.cmp(root, os.path.join(d, 'seed_alloc.py'), shallow=False), d


def campaign(folder, seeds, name='campaign_x'):
    for k, s in enumerate(seeds):
        d = os.path.join(folder, name, 'sweep_cpu_task%04d' % k)
        os.makedirs(d, exist_ok=True)
        json.dump({'_resolved_seed_master': s}, open(os.path.join(d, 'job_args.json'), 'w'))


def ledger(folder, spans):
    p = os.path.join(folder, 'artifacts', 'seed_ledger.tsv')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as fh:
        fh.write(SA.LEDGER_HEADER)
        for lo, hi in spans:
            fh.write('2026-01-01T00:00:00Z\tL\ttag\t%d\t%d\t%d\t%d\t-\n' % (lo, hi - lo, lo, hi))
    return p


@pytest.fixture
def ann(tmp_path):
    t = tmp_path / 'ANN'
    campaign(str(t / 'f1'), [101, 102])                       # depth 1 launch folder
    ledger(str(t / 'f1'), [(1000, 1100)])
    campaign(str(t / 'a' / 'f2'), [201])                      # depth 2
    ledger(str(t / 'a' / 'f2'), [(2000, 2100)])
    campaign(str(t / '.hid'), [301])                          # hidden: skipped
    campaign(str(t / 'scratch9'), [302])                      # scratch*: skipped
    campaign(str(t / 'topo_00001'), [303])                    # topo_*: skipped
    campaign(str(t / 'mea_out'), [304])                       # mea_out*: skipped
    campaign(str(t / 'f1' / 'campaign_x' / 'deep'), [305])    # inside a campaign: not entered
    deep = t
    for k in range(7):
        deep = deep / ('d%d' % k)
    campaign(str(deep), [306])                                 # depth 7: beyond the default 6
    os.makedirs(str(t / 'outside'))
    campaign(str(t / 'outside'), [307])
    os.symlink(str(t / 'outside'), str(t / 'link'))            # a link to a folder: not followed
    return t


def test_scan_tree_reads_and_skips(ann):
    s = SA.scan_tree(str(ann))
    assert s['seeds'] == {101, 102, 201, 307}                  # 'outside' is a real folder, read once
    assert sorted((lo, hi) for lo, hi, _ in s['spans']) == [(1000, 1100), (2000, 2100)]
    s7 = SA.scan_tree(str(ann), max_depth=7)
    assert 306 in s7['seeds']
    os.remove(str(ann / 'link'))
    os.rename(str(ann / 'outside'), str(ann.parent / 'outside_real'))
    os.symlink(str(ann.parent / 'outside_real'), str(ann / 'link'))
    assert 307 not in SA.scan_tree(str(ann))['seeds']           # symbolic link not followed


def test_scan_tree_not_a_directory(tmp_path):
    with pytest.raises(ValueError):
        SA.scan_tree(str(tmp_path / 'nope'))


def test_allocation_invariant_property(tmp_path):
    """Small seed space: every allocated span avoids every recorded seed and reserved span."""
    t = tmp_path / 'ANN'
    own = t / 'clone' / 'LindaColloDUP'
    neighbour = t / 'MEAMA'
    used = list(range(500, 5000, 97))
    campaign(str(neighbour), used)
    reserved = [(6000, 6400), (8000, 8050)]
    ledger(str(neighbour), reserved)
    own_ledger = os.path.join(str(own), 'artifacts', 'seed_ledger.tsv')
    os.makedirs(str(own))
    for k in range(60):
        base = SA.allocate(100, seed_max=10000, root=str(own), ledger_path=own_ledger, verbose=False,
                           scan_trees=[str(t)])
        assert 1 <= base and base + 100 <= 10000 - 1
        assert not any(base <= s < base + 100 for s in used)
        assert not any(base < hi and lo < base + 100 for lo, hi in reserved)
    # only the own ledger is ever written (allocate itself writes nothing)
    assert not os.path.exists(own_ledger)
    assert open(os.path.join(str(neighbour), 'artifacts', 'seed_ledger.tsv')).read().count('\n') == \
        SA.LEDGER_HEADER.count('\n') + len(reserved)


def test_crowded_space_raises(tmp_path):
    t = tmp_path / 'ANN'
    ledger(str(t / 'MEAMA'), [(0, 10000)])
    os.makedirs(str(t / 'own'))
    with pytest.raises(RuntimeError):
        SA.allocate(100, seed_max=10000, root=str(t / 'own'), verbose=False, scan_trees=[str(t)])


def test_cli_missing_tree_and_ledger(tmp_path):
    own = tmp_path / 'own'
    own.mkdir()
    led = str(own / 'artifacts' / 'seed_ledger.tsv')
    r = run([sys.executable, os.path.join(PIPE, 'seed_alloc.py'), '--root', str(own), '--ledger', led,
             '--span', '1000', '--scan-tree', str(tmp_path / 'missing')])
    assert r.returncode == 2 and r.stdout.strip() == ''         # no base on stdout
    assert not os.path.exists(led)
    tree = tmp_path / 'ANN'
    campaign(str(tree / 'other'), [5])
    r = run([sys.executable, os.path.join(PIPE, 'seed_alloc.py'), '--root', str(own), '--ledger', led,
             '--span', '1000', '--scan-tree', str(tree)])
    assert r.returncode == 0
    base = int(r.stdout.strip())
    spans = SA.reserved_spans_from_ledger(led)
    assert [(lo, hi) for lo, hi, _ in spans] == [(base, base + 1000)]
    assert not os.path.exists(str(tree / 'other' / 'artifacts'))   # the scanned folder untouched
