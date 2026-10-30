#!/usr/bin/env python3
"""
apply_box_patch.py -- widen the g_ampa and g_nmda prior boxes in HPC_main_sweep.py.

Run ONCE, inside the LindaColloDUP folder, on its own copy of the pipeline
(copied from the cluster's MEAMA/ folder, 37-D). Never run it on MEAMA/.

Decided 2026-10-30: in LindaColloDUP, g_ampa and g_nmda are drawn on
[0.01, 10] nS instead of the registry's [0.01, 5] nS. The patch changes the
upper bound of those two rows of PARAM_BOUNDS and nothing else. Everything
derived from PARAM_BOUNDS follows when the module is imported: LOG_PARAMS
(both rows still span >= 1 decade, so both stay log-uniform),
PARAM_BOUNDS_THETA, the axis declaration written into job_args.json, and the
range check's design.

Each row is found inside the block "PARAM_BOUNDS = np.array([ ... ])" by its
comment "# <index> <name>"; it must appear there exactly once and hold the
registry box. Both rows are verified before anything is written. Then the
script writes a timestamped backup, writes through a temporary file, re-reads
the result, checks the two rows and compiles it, and restores the backup if
anything fails. Running it again on a patched file changes nothing.

Usage (login node, any python3; no numpy needed):
    python3 apply_box_patch.py --check        # rows only, writes nothing
    python3 apply_box_patch.py                # patch ./HPC_main_sweep.py
    python3 apply_box_patch.py --file PATH    # patch another copy
    python3 apply_box_patch.py --self-test    # embedded fixture, no file touched
Exit status: 0 = patched or already patched; 1 = refused (nothing written).
"""
import argparse
import os
import py_compile
import re
import shutil
import sys
import tempfile
import time

MARK = '[LindaColloDUP box 2026-10-30]'
# axis: (lower bound, registry upper bound, new upper bound), natural units (nS)
BOXES = {'g_ampa': (0.01, 5.0, 10.0),
         'g_nmda': (0.01, 5.0, 10.0)}

BLOCK = re.compile(r"^PARAM_BOUNDS\s*=\s*np\.array\(\[[ \t]*\n(?P<body>.*?)^\][ \t]*,[ \t]*dtype", re.M | re.S)
ROW_TMPL = (r"^(?P<indent>[ \t]*)\((?P<lo>[ \t]*[-+0-9.eE]+[ \t]*),(?P<hi>[ \t]*[-+0-9.eE]+[ \t]*)\)"
            r"(?P<sep>[ \t]*,[ \t]*)(?P<comment>#[ \t]*\d+[ \t]+%s(?![A-Za-z0-9_])[^\n]*)$")


def _close(a, b):
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def find_rows(text):
    """Return (block_start, {name: match}, problems); matches are relative to the block body."""
    blocks = list(BLOCK.finditer(text))
    if len(blocks) != 1:
        return None, {}, ['PARAM_BOUNDS = np.array([...]) block found %d times (need exactly 1)'
                          % len(blocks)]
    body = blocks[0].group('body')
    rows, problems = {}, []
    for name in BOXES:
        hits = list(re.finditer(ROW_TMPL % re.escape(name), body, re.M))
        if len(hits) != 1:
            problems.append('%s: row found %d times inside PARAM_BOUNDS (need exactly 1)'
                            % (name, len(hits)))
        else:
            rows[name] = hits[0]
    return blocks[0].start('body'), rows, problems


def row_box(m):
    return float(m.group('lo')), float(m.group('hi'))


def patch_text(text):
    """Return (new_text, status, problems). status in {'patched', 'already', 'refused'}."""
    n_mark = text.count(MARK)
    start, rows, problems = find_rows(text)
    if problems:
        return text, 'refused', problems
    boxes = {name: row_box(m) for name, m in rows.items()}
    if n_mark == len(BOXES):
        if all(MARK in rows[n].group('comment') and _close(boxes[n][0], BOXES[n][0])
               and _close(boxes[n][1], BOXES[n][2]) for n in BOXES):
            return text, 'already', []
        return text, 'refused', ['%d markers present but the rows do not hold the target boxes; '
                                 'restore the pre-patch copy before re-running' % n_mark]
    if n_mark:
        return text, 'refused', ['partially patched file (%d of %d markers); restore the '
                                 'pre-patch copy before re-running' % (n_mark, len(BOXES))]
    for name, (lo, old_hi, new_hi) in BOXES.items():
        got = boxes[name]
        if _close(got[0], lo) and _close(got[1], new_hi):
            problems.append('%s already holds [%g, %g] without this patch\'s marker; refusing to '
                            'guess (copy the file from MEAMA/ again)' % (name, got[0], got[1]))
        elif not (_close(got[0], lo) and _close(got[1], old_hi)):
            problems.append('%s holds [%g, %g], expected the registry box [%g, %g]'
                            % (name, got[0], got[1], lo, old_hi))
    if problems:
        return text, 'refused', problems
    # rewrite from the last row to the first so that earlier offsets stay valid
    for name in sorted(rows, key=lambda n: rows[n].start(), reverse=True):
        m = rows[name]
        hi_raw = m.group('hi')
        new_hi = '%.1f' % BOXES[name][2]
        width = len(hi_raw.rstrip())
        hi_new = new_hi.rjust(width) if width > len(new_hi) else ' ' + new_hi
        hi_new += hi_raw[len(hi_raw.rstrip()):]
        line = (m.group('indent') + '(' + m.group('lo') + ',' + hi_new + ')' + m.group('sep')
                + m.group('comment').rstrip() + '  ' + MARK)
        a, b = start + m.start(), start + m.end()
        text = text[:a] + line + text[b:]
    return text, 'patched', []


def describe(text):
    _, rows, _ = find_rows(text)
    return ', '.join('%s [%g, %g]' % ((n,) + row_box(rows[n])) for n in sorted(rows))


def apply_to_file(path, check_only=False):
    raw = open(path, 'rb').read()
    if b'\r' in raw:
        print('REFUSED: %s contains CR bytes (CRLF line endings); fix with  '
              "sed -i 's/\\r$//' %s  and re-run" % (path, path))
        return 1
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as e:
        print('REFUSED: %s is not valid UTF-8 (%s) -- transfer corruption?' % (path, e))
        return 1
    new, status, problems = patch_text(text)
    if status == 'already':
        print('ALREADY PATCHED: %s (%s); nothing written' % (path, describe(text)))
        return 0
    if status == 'refused':
        print('REFUSED: nothing written to %s' % path)
        for p in problems:
            print('  - ' + p)
        return 1
    if check_only:
        print('CHECK OK: both rows found once with the registry box in %s (%s); nothing written '
              '(--check)' % (path, describe(text)))
        return 0
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = '%s.bak-box-%s' % (path, stamp)
    shutil.copy2(path, backup)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix='.box_patch_', dir=d)
    with os.fdopen(fd, 'wb') as fh:
        fh.write(new.encode('utf-8'))
    shutil.copymode(path, tmp)          # keep the original permissions (mkstemp gives 0600)
    os.replace(tmp, path)
    back = open(path, 'rb').read().decode('utf-8')
    ok = back == new and patch_text(back)[1] == 'already'
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as e:
        ok = False
        print('compile failed: %s' % e)
    if not ok:
        shutil.copy2(backup, path)
        print('FAILED: verification failed; original restored from %s' % backup)
        return 1
    print('PATCHED: %s  (%s; backup %s)' % (path, describe(back), backup))
    return 0


_FIXTURE = (
    "import numpy as np\n"
    "# g_ampa outside the block must never be touched: (0.01, 5.0), # 14 g_ampa\n"
    "PARAM_BOUNDS = np.array([\n"
    "    (3.0,    15.0),     # 0  Sigma      [mV]   noise amplitude\n"
    "    (0.01,   10.0),     # 1  gbarA      [nS]\n"
    "    (0.01,    5.0),     # 14 g_ampa     [nS]   restored to the v1-v9 box\n"
    "    (0.01,    5.0),     # 15 g_nmda     [nS]   restored to the v1-v9 box\n"
    "    (0.005,   10.0),    # 16 delta_gA   [nS]\n"
    "], dtype=np.float64)\n"
    "LOG_PARAMS = {1, 2, 3}\n"
)


def self_test():
    ok = True

    def expect(label, cond):
        nonlocal ok
        print('  %-62s %s' % (label, 'ok' if cond else 'FAIL'))
        ok = ok and bool(cond)

    print('SELF-TEST on an embedded fixture')
    new, st, pr = patch_text(_FIXTURE)
    expect('fixture patches', st == 'patched' and not pr)
    expect('two markers inserted', new.count(MARK) == 2)
    _, rows, _ = find_rows(new)
    expect('g_ampa and g_nmda now [0.01, 10]',
           all(_close(row_box(rows[n])[0], 0.01) and _close(row_box(rows[n])[1], 10.0) for n in BOXES))
    changed = [(a, b) for a, b in zip(_FIXTURE.splitlines(), new.splitlines()) if a != b]
    expect('exactly the two rows changed', len(changed) == 2 and all('g_' in a for a, _ in changed))
    expect('line outside the block untouched', new.splitlines()[1] == _FIXTURE.splitlines()[1])
    expect('delta_gA (name contains "g") untouched', '(0.005,   10.0),    # 16 delta_gA' in new)
    compile(new, '<patched fixture>', 'exec')
    expect('patched fixture compiles', True)
    again, st2, _ = patch_text(new)
    expect('second run is a no-op (already)', st2 == 'already' and again == new)
    _, st3, pr3 = patch_text(_FIXTURE.replace('# 15 g_nmda', '# 15 g_nmdaX'))
    expect('missing row -> refused', st3 == 'refused' and pr3)
    dup = _FIXTURE.replace('    (0.005,', '    (0.01,    5.0),     # 15 g_nmda  dup\n    (0.005,')
    _, st4, _ = patch_text(dup)
    expect('duplicate row -> refused', st4 == 'refused')
    _, st5, pr5 = patch_text(_FIXTURE.replace('(0.01,    5.0),     # 14', '(0.05,    5.0),     # 14'))
    expect('unexpected registry box -> refused', st5 == 'refused' and 'expected' in pr5[0])
    half = new.replace('  ' + MARK, '', 1)
    _, st6, _ = patch_text(half)
    expect('partially patched -> refused', st6 == 'refused')
    hand = _FIXTURE.replace('(0.01,    5.0),     # 14', '(0.01,   10.0),     # 14')
    _, st7, pr7 = patch_text(hand)
    expect('target box without marker -> refused', st7 == 'refused' and 'marker' in pr7[0])
    _, st8, _ = patch_text(_FIXTURE.replace('PARAM_BOUNDS = np.array([', 'BOUNDS = np.array(['))
    expect('no PARAM_BOUNDS block -> refused', st8 == 'refused')
    expect('inserted text is pure ASCII', all(ord(c) < 128 for c in new))
    tmpd = tempfile.mkdtemp(prefix='box_selftest_')
    try:
        f = os.path.join(tmpd, 'HPC_main_sweep.py')
        open(f, 'w').write(_FIXTURE)
        os.chmod(f, 0o640)
        expect('file: --check writes nothing', apply_to_file(f, check_only=True) == 0
               and open(f).read() == _FIXTURE)
        expect('file: patch returns 0', apply_to_file(f) == 0 and open(f).read() == new)
        expect('file: permissions kept, one backup',
               (os.stat(f).st_mode & 0o777) == 0o640
               and len([n for n in os.listdir(tmpd) if n.startswith('HPC_main_sweep.py.bak-box-')]) == 1)
        expect('file: second run is a no-op', apply_to_file(f) == 0 and open(f).read() == new)
        open(f, 'wb').write(_FIXTURE.replace('\n', '\r\n').encode())
        expect('file: CR bytes -> refused', apply_to_file(f) == 1)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    print('SELF-TEST: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description='Widen the g_ampa/g_nmda boxes in HPC_main_sweep.py '
                                             '(LindaColloDUP).')
    ap.add_argument('--file', default='HPC_main_sweep.py')
    ap.add_argument('--check', action='store_true', help='verify the two rows only; write nothing')
    ap.add_argument('--self-test', action='store_true', help='run the embedded fixture test')
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not os.path.isfile(a.file):
        print('REFUSED: %s not found (run inside the LindaColloDUP folder)' % a.file)
        return 1
    return apply_to_file(a.file, check_only=a.check)


if __name__ == '__main__':
    sys.exit(main())
