#!/usr/bin/env python3
"""
apply_freeze_patch.py -- add a --freeze option to HPC_main_sweep.py.

Run ONCE, inside the LindaColloDUP folder, on its own copy of the pipeline
(copied from the cluster's MEAMA/ folder, 37-D). Never run it on MEAMA/.

What --freeze does (after this patch):
    --freeze "NAME,NAME=VALUE,..."   removes the named axes from --sweep_group
    and holds them fixed: NAME at its nominal value, NAME=VALUE at VALUE
    (natural units, replacing the nominal for this run). Every NAME must be in
    the active sweep group. The fixed values reach the parameter matrix, the
    stored theta, job_args.json ('freeze', '_active_indices',
    '_active_param_names', '_axis_declaration') and manifest.json.
    Without --freeze the pipeline behaves exactly as before.

The patch makes three insertions, each located by an anchor line that must
match exactly once:
    A1  before   p.add_argument('--sweep_group', ...    (the CLI option)
    A2  before   def main():                            (the helper)
    A3  after    active_idx = resolve_sweep_group(args.sweep_group)
It verifies every anchor before writing anything, writes a timestamped backup,
writes through a temporary file, re-reads and compiles the result, and
restores the backup if the compile fails. Running it again on a patched file
changes nothing.

Usage (login node, any python3; no numpy needed):
    python3 apply_freeze_patch.py --check        # anchors only, writes nothing
    python3 apply_freeze_patch.py                # patch ./HPC_main_sweep.py
    python3 apply_freeze_patch.py --file PATH    # patch another copy
    python3 apply_freeze_patch.py --self-test    # embedded fixture, no file touched
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

MARK = '[LindaColloDUP patch 2026-10-29]'

A1 = re.compile(r"^    p\.add_argument\('--sweep_group',", re.M)
A2 = re.compile(r"^def main\(\):", re.M)
A3 = re.compile(r"^    active_idx\s*=\s*resolve_sweep_group\(args\.sweep_group\)[ \t]*$", re.M)

INS_A1 = (
    "    # " + MARK + " freeze axes of the chosen sweep group\n"
    "    p.add_argument('--freeze', default='',\n"
    "                   help='Comma-separated axes taken out of --sweep_group and '\n"
    "                        'held fixed: NAME (at its nominal value) or NAME=VALUE '\n"
    "                        '(natural units, replaces the nominal for this run). '\n"
    "                        'Every NAME must belong to --sweep_group. Default: none.')\n"
)

INS_A2 = (
    "# " + MARK + " --freeze: remove axes from the active sweep group\n"
    "def _apply_freeze(spec, active_idx):\n"
    "    \"\"\"Return active_idx without the axes named in `spec`.\n"
    "\n"
    "    spec: comma-separated NAME or NAME=VALUE items (VALUE in the natural\n"
    "    units of PARAM_UNITS). NAME=VALUE overwrites NOMINAL_PARAMS[NAME] in\n"
    "    place, so the frozen value reaches the parameter matrix, the stored\n"
    "    theta and build_axis_declaration() alike. Raises ValueError on an\n"
    "    unknown name, a name outside the active group, a repeated name, or a\n"
    "    non-finite value (non-positive on a log axis). An empty spec returns\n"
    "    active_idx unchanged.\"\"\"\n"
    "    active = sorted(int(i) for i in active_idx)\n"
    "    spec = (spec or '').strip()\n"
    "    if not spec:\n"
    "        return active\n"
    "    if not isinstance(NOMINAL_PARAMS, np.ndarray):\n"
    "        raise TypeError('--freeze needs NOMINAL_PARAMS as a numpy array')\n"
    "    frozen = {}\n"
    "    for item in spec.split(','):\n"
    "        item = item.strip()\n"
    "        if not item:\n"
    "            continue\n"
    "        name, _, val = item.partition('=')\n"
    "        name = name.strip()\n"
    "        if name not in PARAM_NAMES:\n"
    "            raise ValueError('--freeze: unknown axis %r' % name)\n"
    "        k = PARAM_NAMES.index(name)\n"
    "        if k in frozen:\n"
    "            raise ValueError('--freeze: %r given twice' % name)\n"
    "        if k not in active:\n"
    "            raise ValueError('--freeze: %r is not in the active sweep group' % name)\n"
    "        if val.strip():\n"
    "            v = float(val)\n"
    "            if not np.isfinite(v) or (k in LOG_PARAMS and v <= 0.0):\n"
    "                raise ValueError('--freeze: bad value for %r: %r' % (name, val))\n"
    "            NOMINAL_PARAMS[k] = v\n"
    "        frozen[k] = float(NOMINAL_PARAMS[k])\n"
    "    for k in sorted(frozen):\n"
    "        lo, hi = PARAM_BOUNDS[k]\n"
    "        note = '' if lo <= frozen[k] <= hi else '  (outside the sweep box [%g, %g])' % (lo, hi)\n"
    "        print('[freeze] %-11s held at %g %s%s' % (PARAM_NAMES[k], frozen[k], PARAM_UNITS[k], note))\n"
    "    kept = [i for i in active if i not in frozen]\n"
    "    print('[freeze] %d axes frozen, %d free: %s' % (len(frozen), len(kept),\n"
    "                                                   [PARAM_NAMES[i] for i in kept]))\n"
    "    return kept\n"
    "\n"
    "\n"
)

INS_A3 = (
    "    active_idx    = _apply_freeze(args.freeze, active_idx)   # " + MARK + "\n"
)


def _find_once(rx, text, label):
    hits = [m for m in rx.finditer(text)]
    if len(hits) != 1:
        return None, '%s: anchor found %d times (need exactly 1): %s' % (label, len(hits), rx.pattern)
    return hits[0], None


def patch_text(text):
    """Return (new_text, status, problems). status in {'patched', 'already', 'refused'}."""
    n_mark = text.count(MARK)
    if n_mark == 3 and 'def _apply_freeze(' in text and "'--freeze'" in text:
        return text, 'already', []
    if n_mark:
        return text, 'refused', ['partially patched file (%d of 3 markers); restore the '
                                 'pre-patch copy before re-running' % n_mark]
    if 'def _apply_freeze(' in text or "'--freeze'" in text:
        return text, 'refused', ['file already defines --freeze or _apply_freeze without '
                                 'this patch marker; refusing to guess']
    problems = []
    m1, e1 = _find_once(A1, text, 'A1 (--sweep_group argument)')
    m2, e2 = _find_once(A2, text, 'A2 (def main)')
    m3, e3 = _find_once(A3, text, 'A3 (active_idx = resolve_sweep_group(...))')
    for e in (e1, e2, e3):
        if e:
            problems.append(e)
    if problems:
        return text, 'refused', problems
    if not (m1.start() < m2.start() < m3.start()):
        return text, 'refused', ['anchors out of the expected order A1 < A2 < A3']
    # insert from the end so earlier offsets stay valid
    end3 = text.index('\n', m3.end()) + 1 if '\n' in text[m3.end():] else len(text)
    text = text[:end3] + INS_A3 + text[end3:]
    text = text[:m2.start()] + INS_A2 + text[m2.start():]
    text = text[:m1.start()] + INS_A1 + text[m1.start():]
    return text, 'patched', []


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
        print('ALREADY PATCHED: %s (3 markers, --freeze present); nothing written' % path)
        return 0
    if status == 'refused':
        print('REFUSED: nothing written to %s' % path)
        for p in problems:
            print('  - ' + p)
        return 1
    if check_only:
        print('CHECK OK: all 3 anchors found exactly once in %s; nothing written (--check)' % path)
        return 0
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = '%s.bak-freeze-%s' % (path, stamp)
    shutil.copy2(path, backup)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix='.freeze_patch_', dir=d)
    with os.fdopen(fd, 'wb') as fh:
        fh.write(new.encode('utf-8'))
    shutil.copymode(path, tmp)          # keep the original permissions (mkstemp gives 0600)
    os.replace(tmp, path)
    back = open(path, 'rb').read().decode('utf-8')
    ok = back.count(MARK) == 3 and back == new
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as e:
        ok = False
        print('compile failed: %s' % e)
    if not ok:
        shutil.copy2(backup, path)
        print('FAILED: verification failed; original restored from %s' % backup)
        return 1
    print('PATCHED: %s  (3 insertions; backup %s)' % (path, backup))
    return 0


_FIXTURE = (
    "import numpy as np\n"
    "PARAM_NAMES = ['a', 'b']\n"
    "def build_parser():\n"
    "    p = argparse.ArgumentParser()\n"
    "    p.add_argument('--mode', default='Full', choices=['Full', 'Neuronal'])\n"
    "    p.add_argument('--sweep_group', default='all',\n"
    "                   choices=['all'])\n"
    "    return p\n"
    "\n"
    "\n"
    "def main():\n"
    "    args = build_parser().parse_args()\n"
    "    active_idx    = resolve_sweep_group(args.sweep_group)\n"
    "    inactive_idx  = []\n"
)


def self_test():
    ok = True

    def expect(label, cond):
        nonlocal ok
        print('  %-58s %s' % (label, 'ok' if cond else 'FAIL'))
        ok = ok and bool(cond)

    print('SELF-TEST on an embedded fixture')
    new, st, pr = patch_text(_FIXTURE)
    expect('fixture patches', st == 'patched' and not pr)
    expect('three markers inserted', new.count(MARK) == 3)
    expect('option sits before --sweep_group',
           new.index("'--freeze'") < new.index("'--sweep_group'"))
    expect('helper sits before def main', new.index('def _apply_freeze(') < new.index('def main():'))
    expect('call sits right after resolve_sweep_group',
           new.index('_apply_freeze(args.freeze') > new.index('resolve_sweep_group(args.sweep_group)')
           and new.index('_apply_freeze(args.freeze') < new.index('inactive_idx'))
    compile(new, '<patched fixture>', 'exec')
    expect('patched fixture compiles', True)
    again, st2, _ = patch_text(new)
    expect('second run is a no-op (already)', st2 == 'already' and again == new)
    _, st3, pr3 = patch_text(_FIXTURE.replace('def main():', 'def main_renamed():'))
    expect('missing anchor -> refused', st3 == 'refused' and pr3)
    _, st4, _ = patch_text(_FIXTURE + "\ndef main():\n    pass\n")
    expect('duplicate anchor -> refused', st4 == 'refused')
    half = new.replace(INS_A3, '')
    _, st5, _ = patch_text(half)
    expect('partially patched -> refused', st5 == 'refused')
    expect('inserted text is pure ASCII',
           all(ord(c) < 128 for c in INS_A1 + INS_A2 + INS_A3))
    print('SELF-TEST: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description='Add --freeze to HPC_main_sweep.py (LindaColloDUP).')
    ap.add_argument('--file', default='HPC_main_sweep.py')
    ap.add_argument('--check', action='store_true', help='verify anchors only; write nothing')
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
