#!/usr/bin/env python3
"""
apply_prior_patch.py -- apply this folder's prior_config.tsv to its copy of HPC_main_sweep.py.

prior_config.tsv holds the prior of the GiuliaGate campaign, one row per axis
(decided 2026-10-31; specs/SPEC.md Block 9). This script is the only reader:

  (no option)        patch PARAM_BOUNDS: every free axis whose box in the TSV
                     differs from the row in ./HPC_main_sweep.py gets the TSV
                     box, and the row is marked with MARK and the registry box
                     it replaced. Everything derived from PARAM_BOUNDS follows at
                     import (LOG_PARAMS, PARAM_BOUNDS_THETA, the axis declaration
                     of job_args.json, the range check's design).
  --check            validate the TSV against the file, print what would change,
                     write nothing.
  --check-applied    exit 0 when every free axis's row equals its TSV box; prints
                     the boxes that differ from the registry (for banners and
                     provenance). The launchers and the job scripts call it.
  --freeze-list      print the freeze list for HPC_main_sweep.py --freeze: every
                     fixed axis of the sweep group 'tripartite', NAME=VALUE with
                     the TSV value as written.
  --topology NAME    print "LOW HIGH" of a topology row (conn_prob, stoa_gate_p).
  --self-test        embedded fixture; touches no file.

Rules checked on every call: the TSV lists exactly the axes of PARAM_BOUNDS plus
conn_prob and stoa_gate_p; status is free or fixed; a free row has
0 < high - low, a fixed row a number; DeltaT, VT, gL, I_Theta and omega_I lie
outside the group, so they must be fixed at the registry value (the freeze list
cannot move them); topology boxes satisfy 0 <= LOW <= HIGH <= 1.

Writing: timestamped backup, write through a temporary file, re-read, check,
compile; the backup is restored if anything fails. Running it again on a
patched file changes nothing. A file patched to other boxes (the TSV was edited
afterwards) is refused: copy the file in again (APPLY_GG.txt step 0) and patch.

Usage (login node, any python3; standard library only):
    python3 apply_prior_patch.py --check | (none) | --check-applied
    python3 apply_prior_patch.py --freeze-list
    python3 apply_prior_patch.py --topology conn_prob
    python3 apply_prior_patch.py --self-test
Options: --tsv PATH (default prior_config.tsv beside this script), --file PATH
(default ./HPC_main_sweep.py), --nominal PATH (default HPC_single_run.py beside
--file). Exit status: 0 = done / OK; 1 = refused or not applied (nothing written).
"""
import argparse
import os
import py_compile
import re
import shutil
import sys
import tempfile
import time

MARK = '[GiuliaGate box 2026-10-31]'
OUTSIDE_GROUP = ('DeltaT', 'VT', 'gL', 'I_Theta', 'omega_I')   # not in 'tripartite'
TOPOLOGY = ('conn_prob', 'stoa_gate_p')
NUM = re.compile(r'^-?(\d+(\.\d*)?|\.\d+)([eE]-?\d+)?$')       # as the launchers' FREEZE check
BLOCK = re.compile(r"^PARAM_BOUNDS\s*=\s*np\.array\(\[[ \t]*\n(?P<body>.*?)^\][ \t]*,[ \t]*dtype", re.M | re.S)
ROW = re.compile(r"^(?P<indent>[ \t]*)\((?P<lo>[ \t]*[-+0-9.eE]+[ \t]*),(?P<hi>[ \t]*[-+0-9.eE]+[ \t]*)\)"
                 r"(?P<sep>[ \t]*,[ \t]*)(?P<comment>#[ \t]*(?P<idx>\d+)[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)[^\n]*)$",
                 re.M)
NOM_BLOCK = re.compile(r"^NOMINAL_PARAMS\s*=\s*np\.array\(\[[ \t]*\n(?P<body>.*?)^\]\)", re.M | re.S)
NOM_ROW = re.compile(r"^[ \t]*(?P<v>[-+0-9.eE]+)[ \t]*,[ \t]*#[ \t]*(?P<idx>\d+)[ \t]+(?P<name>[A-Za-z_][A-Za-z0-9_]*)", re.M)


def _close(a, b):
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def read_tsv(text):
    """Rows of the TSV as dicts (comment lines start with '#'); problems as a list."""
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith('#')]
    if not lines:
        return [], ['the TSV has no header']
    head = lines[0].split('\t')
    need = ('axis', 'status', 'low', 'high', 'value')
    missing = [c for c in need if c not in head]
    if missing:
        return [], ['the TSV lacks the column(s) %s' % ', '.join(missing)]
    rows, problems = [], []
    for k, l in enumerate(lines[1:], start=2):
        cells = l.split('\t')
        if len(cells) != len(head):
            problems.append('data line %d has %d cells, the header %d' % (k, len(cells), len(head)))
            continue
        rows.append(dict(zip(head, cells)))
    return rows, problems


def file_rows(text):
    """{name: match} of the PARAM_BOUNDS rows (offsets relative to the block body), block start, problems."""
    blocks = list(BLOCK.finditer(text))
    if len(blocks) != 1:
        return {}, None, ['PARAM_BOUNDS = np.array([...]) block found %d times (need exactly 1)' % len(blocks)]
    rows, problems = {}, []
    for m in ROW.finditer(blocks[0].group('body')):
        if m.group('name') in rows:
            problems.append('%s: two rows inside PARAM_BOUNDS' % m.group('name'))
        rows[m.group('name')] = m
    if not rows:
        problems.append('no row of the form "(lo, hi),  # k name" inside PARAM_BOUNDS')
    return rows, blocks[0].start('body'), problems


def nominal_values(text):
    """{name: nominal} parsed from NOMINAL_PARAMS of HPC_single_run.py."""
    b = list(NOM_BLOCK.finditer(text))
    if len(b) != 1:
        return {}
    return {m.group('name'): float(m.group('v')) for m in NOM_ROW.finditer(b[0].group('body'))}


def box(m):
    return float(m.group('lo')), float(m.group('hi'))


def validate(tsv_rows, rows, nominal):
    """The TSV against the file: (free {name: (lo, hi)}, fixed {name: value str}, topology, problems)."""
    problems, free, fixed, topo = [], {}, {}, {}
    seen = set()
    for r in tsv_rows:
        n, st = r['axis'], r['status']
        if n in seen:
            problems.append('%s listed twice in the TSV' % n)
        seen.add(n)
        if n not in rows and n not in TOPOLOGY:
            problems.append('%s is not an axis of PARAM_BOUNDS' % n)
            continue
        if st == 'free':
            try:
                lo, hi = float(r['low']), float(r['high'])
            except ValueError:
                problems.append('%s: free, but low/high are not numbers (%r, %r)' % (n, r['low'], r['high']))
                continue
            if not hi > lo:
                problems.append('%s: free needs low < high (got %g, %g)' % (n, lo, hi))
                continue
            if n in TOPOLOGY:
                if not 0.0 <= lo <= hi <= 1.0:
                    problems.append('%s: needs 0 <= low <= high <= 1 (got %g, %g)' % (n, lo, hi))
                topo[n] = (r['low'], r['high'])
            elif n in OUTSIDE_GROUP:
                problems.append('%s lies outside the sweep group: it cannot be free' % n)
            else:
                free[n] = (lo, hi)
        elif st == 'fixed':
            if n in TOPOLOGY:
                if not NUM.match(r['value']):
                    problems.append('%s: fixed, but value %r is not a number' % (n, r['value']))
                else:
                    topo[n] = (r['value'], r['value'])
                continue
            if not NUM.match(r['value']):
                problems.append('%s: fixed, but value %r is not a plain number (no "+", e.g. 1e3)'
                                % (n, r['value']))
                continue
            v = float(r['value'])
            if n in OUTSIDE_GROUP:
                reg = box(rows[n])[0] if _close(*box(rows[n])) else nominal.get(n)
                if reg is None:
                    problems.append('%s: registry value not found (NOMINAL_PARAMS)' % n)
                elif not _close(v, reg):
                    problems.append('%s lies outside the sweep group: its value must be the registry %g (got %g)'
                                    % (n, reg, v))
            else:
                fixed[n] = r['value']
        else:
            problems.append('%s: status must be free or fixed (got %r)' % (n, st))
    for n in list(rows) + list(TOPOLOGY):
        if n not in seen:
            problems.append('%s is missing from the TSV' % n)
    return free, fixed, topo, problems


def load(tsv_path, file_path, nominal_path):
    tsv_rows, p1 = read_tsv(open(tsv_path, encoding='ascii').read())
    text = open(file_path, encoding='utf-8').read()
    rows, start, p2 = file_rows(text)
    nominal = nominal_values(open(nominal_path, encoding='utf-8').read()) if os.path.isfile(nominal_path) else {}
    if p1 or p2:
        return text, rows, start, {}, {}, {}, p1 + p2
    free, fixed, topo, p3 = validate(tsv_rows, rows, nominal)
    return text, rows, start, free, fixed, topo, p3


# ---------------------------------------------------------------------------
# patching
# ---------------------------------------------------------------------------
def changes(rows, free):
    """Free axes whose row differs from the TSV box: [(name, old box, new box, marked)]."""
    out = []
    for n, (lo, hi) in free.items():
        got = box(rows[n])
        if not (_close(got[0], lo) and _close(got[1], hi)):
            out.append((n, got, (lo, hi), MARK in rows[n].group('comment')))
    return out


def patch_text(text, rows, start, free):
    """Return (new_text, status, problems); status in {'patched', 'already', 'refused'}."""
    todo = changes(rows, free)
    if not todo:
        return text, 'already', []
    marked = [n for n, _, _, mk in todo if mk]
    if marked:
        return text, 'refused', ['%s already patched to another box (the TSV changed afterwards): copy '
                                 'HPC_main_sweep.py in again (APPLY_GG.txt step 0) and patch' % ', '.join(marked)]
    for n, old, new, _ in sorted(todo, key=lambda t: rows[t[0]].start(), reverse=True):
        m = rows[n]
        line = ('%s(%s, %s)%s%s  %s registry (%s, %s)'
                % (m.group('indent'), repr(float(new[0])), repr(float(new[1])), m.group('sep'),
                   m.group('comment').rstrip(), MARK, repr(old[0]), repr(old[1])))
        a, b = start + m.start(), start + m.end()
        text = text[:a] + line + text[b:]
    return text, 'patched', []


def describe(rows, free):
    """Boxes of the free axes that differ from the registry, as 'name [lo, hi]' (patched rows)."""
    return '; '.join('%s [%g, %g]' % (n, free[n][0], free[n][1]) for n in sorted(free)
                     if MARK in rows[n].group('comment')) or '(none)'


def apply_to_file(path, free, check_only=False, tsv_path='', nominal_path=''):
    raw = open(path, 'rb').read()
    if b'\r' in raw:
        print("REFUSED: %s contains CR bytes; fix with  sed -i 's/\\r$//' %s  and re-run" % (path, path))
        return 1
    text = raw.decode('utf-8')
    rows, start, problems = file_rows(text)
    if problems:
        print('REFUSED: nothing written to %s' % path)
        for p in problems:
            print('  - ' + p)
        return 1
    new, status, problems = patch_text(text, rows, start, free)
    if status == 'already':
        print('ALREADY APPLIED: %s holds every box of the TSV; nothing written' % path)
        return 0
    if status == 'refused':
        print('REFUSED: nothing written to %s' % path)
        for p in problems:
            print('  - ' + p)
        return 1
    todo = changes(rows, free)
    if check_only:
        print('CHECK OK: %s; would set %s (nothing written, --check)'
              % (path, '; '.join('%s [%g, %g] -> [%g, %g]' % (n, o[0], o[1], w[0], w[1]) for n, o, w, _ in todo)))
        return 0
    mode = os.stat(path).st_mode & 0o7777
    backup = '%s.bak-box-%s' % (path, time.strftime('%Y%m%d_%H%M%S'))
    shutil.copy2(path, backup)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as f:
            f.write(new)
        rows2, _, p2 = file_rows(open(tmp, encoding='utf-8').read())
        if p2 or changes(rows2, free):
            raise RuntimeError('re-read check failed: %s' % (p2 or changes(rows2, free)))
        py_compile.compile(tmp, doraise=True)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception as e:                                   # restore, then report
        if os.path.exists(tmp):
            os.remove(tmp)
        shutil.copy2(backup, path)
        print('REFUSED: %s (file restored from %s)' % (e, backup))
        return 1
    print('PATCHED: %s  (%s; backup %s)' % (path, '; '.join('%s [%g, %g]' % (n, w[0], w[1]) for n, _, w, _ in todo),
                                           backup))
    return 0


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
_FIX_PY = '''import numpy as np
PARAM_BOUNDS = np.array([
    (3.0,    15.0),     # 0  Sigma      [mV]   noise
    (0.01,    5.0),     # 1  g_ampa     [nS]
    (0.1,     1.0),     # 2  U_0_sr     [1]
    (2.0,     2.0),     # 3  DeltaT     [mV]   FROZEN
    (0.1,     1.0),     # 4  I_Theta    [uM]
    (0.003,  0.05),     # 5  U_0_ar     [1]
], dtype=np.float64)
'''
_FIX_NOM = '''NOMINAL_PARAMS = np.array([
    4.0,        # 0  Sigma
    0.3,        # 1  g_ampa
    0.15,       # 2  U_0_sr
    2.0,        # 3  DeltaT
    0.3,        # 4  I_Theta
    0.003,      # 5  U_0_ar
])
'''
_FIX_TSV = '''# fixture
axis\tgroup\tstatus\tlow\thigh\tvalue
Sigma\tneuron\tfree\t2\t10\t-
g_ampa\tsynapse\tfree\t0.05\t10\t-
U_0_sr\tsynapse\tfree\t0.1\t1\t-
DeltaT\tneuron_frozen\tfixed\t-\t-\t2
I_Theta\tastro\tfixed\t-\t-\t0.3
U_0_ar\tsynapse\tfixed\t-\t-\t0.003
conn_prob\ttopology\tfree\t0.05\t0.4\t-
stoa_gate_p\ttopology\tfree\t0.2\t1\t-
'''


def self_test():
    ok = [True]

    def expect(label, cond):
        print('  %s  %s' % ('ok  ' if cond else 'FAIL', label))
        ok[0] = ok[0] and bool(cond)

    tsv_rows, p = read_tsv(_FIX_TSV)
    rows, start, p2 = file_rows(_FIX_PY)
    nom = nominal_values(_FIX_NOM)
    free, fixed, topo, p3 = validate(tsv_rows, rows, nom)
    expect('fixture parses: 6 axes, 8 TSV rows, no problem', not (p or p2 or p3) and len(rows) == 6 and len(tsv_rows) == 8)
    expect('free / fixed / topology split', sorted(free) == ['Sigma', 'U_0_sr', 'g_ampa'] and fixed == {'U_0_ar': '0.003'}
           and topo == {'conn_prob': ('0.05', '0.4'), 'stoa_gate_p': ('0.2', '1')})
    new, st, _ = patch_text(_FIX_PY, rows, start, free)
    rows_n, _, _ = file_rows(new)
    expect('patch: Sigma and g_ampa rewritten and marked, U_0_sr untouched',
           st == 'patched' and box(rows_n['Sigma']) == (2.0, 10.0) and box(rows_n['g_ampa']) == (0.05, 10.0)
           and MARK in rows_n['Sigma'].group('comment') and MARK not in rows_n['U_0_sr'].group('comment')
           and 'registry (3.0, 15.0)' in rows_n['Sigma'].group('comment'))
    expect('patched text still a valid python module', _compiles(new))
    _, st2, _ = patch_text(new, rows_n, file_rows(new)[1], free)
    expect('second run: already applied', st2 == 'already')
    free_b = dict(free, Sigma=(1.0, 10.0))
    _, st3, pr3 = patch_text(new, rows_n, file_rows(new)[1], free_b)
    expect('a TSV edited after patching is refused', st3 == 'refused' and 'Sigma' in pr3[0])
    bad = _FIX_TSV.replace('DeltaT\tneuron_frozen\tfixed\t-\t-\t2', 'DeltaT\tneuron_frozen\tfixed\t-\t-\t3')
    expect('an axis outside the group at a non-registry value is refused',
           any('DeltaT' in q for q in validate(read_tsv(bad)[0], rows, nom)[3]))
    bad = _FIX_TSV.replace('I_Theta\tastro\tfixed\t-\t-\t0.3', 'I_Theta\tastro\tfree\t0.1\t1\t-')
    expect('an axis outside the group set free is refused',
           any('I_Theta' in q for q in validate(read_tsv(bad)[0], rows, nom)[3]))
    bad = _FIX_TSV.replace('U_0_ar\tsynapse\tfixed\t-\t-\t0.003\n', '')
    expect('a missing axis is refused', any('U_0_ar is missing' in q for q in validate(read_tsv(bad)[0], rows, nom)[3]))
    bad = _FIX_TSV.replace('stoa_gate_p\ttopology\tfree\t0.2\t1', 'stoa_gate_p\ttopology\tfree\t0.2\t1.5')
    expect('a topology box above 1 is refused', any('stoa_gate_p' in q for q in validate(read_tsv(bad)[0], rows, nom)[3]))
    bad = _FIX_TSV.replace('U_0_ar\tsynapse\tfixed\t-\t-\t0.003', 'U_0_ar\tsynapse\tfixed\t-\t-\t+3e-3')
    expect('a value with "+" is refused (the FREEZE list allows none)',
           any('U_0_ar' in q for q in validate(read_tsv(bad)[0], rows, nom)[3]))
    bad = _FIX_TSV.replace('g_ampa\tsynapse\tfree\t0.05\t10', 'g_ampa\tsynapse\tfree\t10\t0.05')
    expect('a free box with low >= high is refused', any('g_ampa' in q for q in validate(read_tsv(bad)[0], rows, nom)[3]))
    print('SELF-TEST: %s' % ('PASS' if ok[0] else 'FAIL'))
    return 0 if ok[0] else 1


def _compiles(text):
    fd, tmp = tempfile.mkstemp(suffix='.py')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        py_compile.compile(tmp, doraise=True)
        return True
    except py_compile.PyCompileError:
        return False
    finally:
        os.remove(tmp)


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description='Apply prior_config.tsv to HPC_main_sweep.py (GiuliaGate).')
    ap.add_argument('--tsv', default=os.path.join(here, 'prior_config.tsv'))
    ap.add_argument('--file', default='HPC_main_sweep.py')
    ap.add_argument('--nominal', default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--check', action='store_true')
    g.add_argument('--check-applied', action='store_true')
    g.add_argument('--freeze-list', action='store_true')
    g.add_argument('--topology', choices=TOPOLOGY)
    g.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    nominal = a.nominal or os.path.join(os.path.dirname(os.path.abspath(a.file)), 'HPC_single_run.py')
    for p in (a.tsv, a.file):
        if not os.path.isfile(p):
            print('REFUSED: %s not found' % p)
            return 1
    text, rows, start, free, fixed, topo, problems = load(a.tsv, a.file, nominal)
    if problems:
        print('REFUSED: %s does not match %s:' % (a.tsv, a.file))
        for p in problems:
            print('  - ' + p)
        return 1
    if a.freeze_list:
        print(','.join('%s=%s' % (n, v) for n, v in fixed.items()))
        return 0
    if a.topology:
        print('%s %s' % topo[a.topology])
        return 0
    if a.check_applied:
        todo = changes(rows, free)
        if todo:
            print('NOT APPLIED: %s' % '; '.join('%s holds [%g, %g], TSV [%g, %g]' % (n, o[0], o[1], w[0], w[1])
                                                for n, o, w, _ in todo))
            return 1
        print('BOXES OK: %d free axes as in %s; changed from the registry: %s'
              % (len(free), os.path.basename(a.tsv), describe(rows, free)))
        return 0
    return apply_to_file(a.file, free, check_only=a.check)


if __name__ == '__main__':
    sys.exit(main())
