#!/usr/bin/env python3
"""
clamp_cadex_copies.py -- find every copy of the CAdEx neuron model under the
project folders and hold V at VR during the refractory period in each
(Brian2 equation flag "(unless refractory)").

Why
    In ASD_fun_BD_cpp.py the CAdEx dV/dt line had no "(unless refractory)":
    during t_ref V kept integrating while Brian2 skipped the threshold test,
    so a neuron pushed past the unstable fixed point ran away to inf/NaN and
    never spiked again. Gorski et al. 2021 (Sec. 2) hold V at VR during t_ref.
    GitHub main carries the fix since commit 0963b7b; each project folder on
    the cluster (Main/, Main/Giulia_Astro/, MEAMA/, LindaColloDUP/, ...) runs
    its own copy, which this tool finds, reports and, with --apply, patches.

What counts as a CAdEx copy
    A file named ASD_fun_BD_cpp.py whose single CAdEx dV/dt line contains the
    exponential spike term exp((V-VT)/DeltaT). Files whose dV/dt lines lack
    that term (the Hodgkin-Huxley models) are listed as SKIP and never
    touched: holding V during refractoriness would break an HH neuron. A file
    that contains the term but no single-line "dV/dt = ... : volt" carrying it
    is REFUSED, never SKIP, so an unrecognised CAdEx copy cannot pass silently.

Patch (only with --apply, only on NOT CLAMPED files)
    Appends " (unless refractory)" to that dV/dt line and inserts one comment
    line above it; nothing else changes. A timestamped backup
    <file>.bak-clamp-<stamp> is written first; the result is re-read, checked
    and compiled, and the backup is restored if anything fails.

Safety
    A folder whose campaign_* output was written in the last --recent-hours
    hours (default 6) may still be running a campaign, and patching would
    change the model for its remaining tasks. Such files are reported HELD and
    left untouched unless --force-running is given.

Usage
    python3 clamp_cadex_copies.py                    # report; ROOT = /davinci-1/home/ldellamea/ANN
    python3 clamp_cadex_copies.py ROOT [ROOT ...]    # report other roots
    python3 clamp_cadex_copies.py --apply            # patch every NOT CLAMPED copy
    python3 clamp_cadex_copies.py --self-test        # embedded fixture test, touches nothing else
Exit status
    0  every CAdEx copy found is clamped (after patching, with --apply)
    3  report mode: some copies are NOT CLAMPED (run again with --apply)
    1  something was refused, held or failed (see the table), or no CAdEx copy was found
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
import time
import warnings

DEFAULT_ROOT = '/davinci-1/home/ldellamea/ANN'
TARGET = 'ASD_fun_BD_cpp.py'
FLAG = '(unless refractory)'
CADEX_TERM = 'exp((V-VT)/DeltaT)'
MARK = '[refractory clamp]'
DV = re.compile(r'^([ \t]*)(dV/dt[ \t]*=.*?:[ \t]*volt)([ \t]*(\([^()\n]*\))?)[ \t]*$', re.M)
PRUNE_PREFIXES = ('campaign_', 'scratch', 'mea_out', '.', '__pycache__')
CFG = dict(recent_hours=6.0, max_depth=6)


# ---------------------------------------------------------------------------
# classification and patch of one file's text (pure functions)
# ---------------------------------------------------------------------------
def cadex_matches(text):
    """dV/dt lines of the text that carry the CAdEx exponential term."""
    return [m for m in DV.finditer(text) if CADEX_TERM in m.group(2).replace(' ', '')]


def classify_text(text):
    """Return (status, detail, match) with status in CLAMPED, NOT CLAMPED, SKIP, REFUSED."""
    hits = cadex_matches(text)
    if not hits:
        n_dv = len(DV.findall(text))
        if CADEX_TERM in text.replace(' ', ''):
            return ('REFUSED', 'CAdEx term present, but no single-line "dV/dt = ... : volt" '
                    'carries it; check the file by hand', None)
        return 'SKIP', 'no CAdEx dV/dt line (%d other dV/dt line(s))' % n_dv, None
    if len(hits) > 1:
        return 'REFUSED', '%d CAdEx dV/dt lines; refusing to guess' % len(hits), None
    m = hits[0]
    flags = (m.group(4) or '').strip()
    if flags == FLAG:
        return 'CLAMPED', m.group(0).strip(), m
    if flags:
        return 'REFUSED', 'the dV/dt line carries other flags %r' % flags, m
    if MARK in text:
        return 'REFUSED', 'partial patch: marker present, flag missing; restore the backup', m
    return 'NOT CLAMPED', m.group(0).strip(), m


def patch_text(text):
    """Text with the flag appended and one comment line inserted, or None if not patchable."""
    status, _, m = classify_text(text)
    if status != 'NOT CLAMPED':
        return None
    comment = '%s# %s V held at VR during t_ref (Gorski et al. 2021, Sec. 2)\n' % (m.group(1), MARK)
    return text[:m.start()] + comment + m.group(1) + m.group(2) + ' ' + FLAG + text[m.end():]


# ---------------------------------------------------------------------------
# file system
# ---------------------------------------------------------------------------
def find_copies(roots, max_depth):
    """Absolute paths of every TARGET file under the roots (campaign output pruned)."""
    out = []
    for root in roots:
        root = os.path.abspath(root)
        base = root.rstrip(os.sep).count(os.sep)
        for d, dirs, files in os.walk(root):
            depth = d.count(os.sep) - base
            dirs[:] = sorted(x for x in dirs if not x.startswith(PRUNE_PREFIXES)) if depth < max_depth else []
            if TARGET in files:
                out.append(os.path.join(d, TARGET))
    return sorted(set(out))


def newest_campaign_age_h(folder, now):
    """Hours since the newest campaign_*/ or campaign_*/<child>/ in the folder was written, or None."""
    newest = None
    try:
        for e in os.scandir(folder):
            if not (e.name.startswith('campaign_') and e.is_dir(follow_symlinks=False)):
                continue
            ts = [e.stat(follow_symlinks=False).st_mtime]
            try:
                ts += [c.stat(follow_symlinks=False).st_mtime for c in os.scandir(e.path)]
            except OSError:
                pass
            newest = max(ts) if newest is None else max([newest] + ts)
    except OSError:
        return None
    return None if newest is None else (now - newest) / 3600.0


def read_text(path):
    raw = open(path, 'rb').read()
    if b'\r' in raw:
        return None, "CR bytes (CRLF line endings): fix with  sed -i 's/\\r$//' FILE"
    try:
        return raw.decode('utf-8'), None
    except UnicodeDecodeError as e:
        return None, 'not valid UTF-8 (%s)' % e


def apply_patch(path, text):
    """Patch one NOT CLAMPED file in place; return (status, detail)."""
    new = patch_text(text)
    if new is None:
        return 'FAILED', 'nothing to patch'
    backup = '%s.bak-clamp-%s' % (path, time.strftime('%Y%m%d_%H%M%S'))
    shutil.copy2(path, backup)
    fd, tmp = tempfile.mkstemp(prefix='.clamp_', dir=os.path.dirname(path))
    with os.fdopen(fd, 'wb') as fh:
        fh.write(new.encode('utf-8'))
    shutil.copymode(path, tmp)
    os.replace(tmp, path)
    back = open(path, 'rb').read().decode('utf-8')
    ok = back == new and classify_text(back)[0] == 'CLAMPED'
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)   # the file's own '$\mu m$' plot labels
            compile(back, path, 'exec')                      # syntax check, writes nothing
    except (SyntaxError, ValueError) as e:
        ok = False
        print('  compile failed for %s: %s' % (path, e))
    if not ok:
        shutil.copy2(backup, path)
        return 'FAILED', 'verification failed; original restored from %s' % backup
    return 'PATCHED', 'backup %s' % backup


def scan(roots, apply=False, force_running=False, cfg=None):
    """Classify (and with apply, patch) every copy; return the list of row dicts."""
    cfg = dict(CFG, **(cfg or {}))
    now = time.time()
    rows = []
    for path in find_copies(roots, cfg['max_depth']):
        age = newest_campaign_age_h(os.path.dirname(path), now)
        text, err = read_text(path)
        if err:
            rows.append(dict(path=path, status='REFUSED', detail=err, age=age))
            continue
        status, detail, _ = classify_text(text)
        if apply and status == 'NOT CLAMPED':
            if age is not None and age < cfg['recent_hours'] and not force_running:
                status, detail = 'HELD', ('campaign output written %.1f h ago: a campaign may still be '
                                          'running here; stop it or use --force-running' % age)
            else:
                status, detail = apply_patch(path, text)
        rows.append(dict(path=path, status=status, detail=detail, age=age))
    return rows


def exit_code(rows, apply):
    statuses = [r['status'] for r in rows]
    cadex = [s for s in statuses if s != 'SKIP']
    if not cadex or any(s in ('REFUSED', 'HELD', 'FAILED') for s in statuses):
        return 1
    if any(s == 'NOT CLAMPED' for s in statuses):
        return 3
    return 0


def report(rows, apply):
    print('%-12s %-9s %s' % ('status', 'output', 'file'))
    for r in rows:
        age = '-' if r['age'] is None else '%.1f h' % r['age']
        print('%-12s %-9s %s' % (r['status'], age, r['path']))
        print('%-12s %-9s   %s' % ('', '', r['detail'][-150:]))
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print('\nfiles found: %d  %s' % (len(rows), '  '.join('%s %d' % kv for kv in sorted(counts.items()))))
    print("('output' = age of the newest campaign_* output in the file's folder)")
    code = exit_code(rows, apply)
    verdict = {0: 'ALL CAdEx COPIES CLAMPED', 3: 'SOME COPIES NOT CLAMPED: run again with --apply',
               1: 'ATTENTION: refused, held or failed copies (or no CAdEx copy found)'}[code]
    print('verdict: %s (exit %d)' % (verdict, code))
    return code


# ---------------------------------------------------------------------------
# self-test on embedded fixtures
# ---------------------------------------------------------------------------
_CADEX = ("eqs_NN = Equations('''\n"
          "    dV/dt  = noise + (gl*(El-V) + gl*DeltaT*exp((V-VT)/DeltaT) + gA*(EA-V) + I_inj*bias_unit - I_syn)/Cm : volt\n"
          "    dgA/dt = (gbarA/(1 + exp((VA-V)/DeltaA)) - gA)/tauA : siemens\n"
          "    noise = sigma*(2*gl_ref/Cm)**.5*randn()/sqrt(dt) : volt/second (constant over dt)\n"
          "    I_syn : amp\n    sigma : volt\n    gl : siemens\n    gbarA : siemens\n    tauA : second\n"
          "    DeltaT : volt\n    VT : volt\n    VA : volt\n    DeltaA : volt\n    I_inj : amp\n"
          "    bias_unit : 1\n    Cm : farad\n    ''')\n")
_HH = ("eqs = '''\n    dV/dt = noise + (-gl*(V-El)-g_na*(m*m*m)*h*(V-ENa)-g_kd*(n*n*n*n)*(V-EK)+I-I_syn)/Cm  : volt\n'''\n")


def self_test():
    ok = True

    def expect(label, cond):
        nonlocal ok
        print('  %-66s %s' % (label, 'ok' if cond else 'FAIL'))
        ok = ok and bool(cond)

    print('SELF-TEST (embedded fixtures, temporary folder)')
    clamped = patch_text(_CADEX)
    expect('unclamped CAdEx -> NOT CLAMPED', classify_text(_CADEX)[0] == 'NOT CLAMPED')
    expect('patched text -> CLAMPED', classify_text(clamped)[0] == 'CLAMPED')
    expect('patch adds only the comment line and the flag',
           clamped.replace('    # %s V held at VR during t_ref (Gorski et al. 2021, Sec. 2)\n' % MARK, '')
           .replace(' ' + FLAG, '') == _CADEX)
    expect('dgA/dt line untouched', '- gA)/tauA : siemens\n' in clamped)
    expect('Hodgkin-Huxley model -> SKIP', classify_text(_HH)[0] == 'SKIP')
    expect('CAdEx term, dV/dt line not recognised -> REFUSED (not SKIP)',
           classify_text(_CADEX.replace(': volt\n', ': V\n', 1))[0] == 'REFUSED')
    expect('two CAdEx lines -> REFUSED', classify_text(_CADEX + _CADEX)[0] == 'REFUSED')
    expect('other flags -> REFUSED',
           classify_text(_CADEX.replace(': volt\n', ': volt (constant over dt)\n', 1))[0] == 'REFUSED')
    expect('marker without flag -> REFUSED', classify_text(clamped.replace(' ' + FLAG, ''))[0] == 'REFUSED')
    expect('flag without marker -> CLAMPED',
           classify_text(_CADEX.replace(': volt\n', ': volt (unless refractory)\n', 1))[0] == 'CLAMPED')
    expect('inserted text is pure ASCII', all(ord(c) < 128 for c in clamped))
    tmp = tempfile.mkdtemp(prefix='clamp_selftest_')
    try:
        tree = dict(a=_CADEX, b=clamped, hh=_HH, live=_CADEX, crlf=_CADEX.replace('\n', '\r\n'))
        for k, v in tree.items():
            os.makedirs(os.path.join(tmp, k))
            open(os.path.join(tmp, k, TARGET), 'wb').write(v.encode('utf-8'))
        os.makedirs(os.path.join(tmp, 'live', 'campaign_x', 'sweep_intel_task0000'))
        os.makedirs(os.path.join(tmp, 'a', 'campaign_old', 'deep'))
        open(os.path.join(tmp, 'a', 'campaign_old', TARGET), 'w').write(_CADEX)   # inside output: pruned
        old = time.time() - 30 * 3600
        for p in ('campaign_old/deep', 'campaign_old/' + TARGET, 'campaign_old'):   # children first
            os.utime(os.path.join(tmp, 'a', p), (old, old))
        st = {os.path.basename(os.path.dirname(r['path'])): r['status'] for r in scan([tmp])}
        expect('report: 5 copies found, campaign output pruned', len(st) == 5)
        expect('report statuses', st == dict(a='NOT CLAMPED', b='CLAMPED', hh='SKIP', live='NOT CLAMPED',
                                              crlf='REFUSED'))
        rows = scan([tmp], apply=True)
        st = {os.path.basename(os.path.dirname(r['path'])): r['status'] for r in rows}
        expect('apply: a patched, live held (recent output), others unchanged',
               st == dict(a='PATCHED', b='CLAMPED', hh='SKIP', live='HELD', crlf='REFUSED'))
        expect('apply: exit 1 while a copy is held or refused', exit_code(rows, True) == 1)
        expect('apply: backup written for a', any(n.startswith(TARGET + '.bak-clamp-')
                                                for n in os.listdir(os.path.join(tmp, 'a'))))
        expect('apply: hh file byte-identical', open(os.path.join(tmp, 'hh', TARGET)).read() == _HH)
        rows = scan([tmp], apply=True, force_running=True)
        st = {os.path.basename(os.path.dirname(r['path'])): r['status'] for r in rows}
        expect('--force-running patches the live folder', st['live'] == 'PATCHED' and st['a'] == 'CLAMPED')
        os.remove(os.path.join(tmp, 'crlf', TARGET))
        rows = scan([tmp], apply=True)
        expect('second apply is a no-op, exit 0', exit_code(rows, True) == 0 and
               all(r['status'] in ('CLAMPED', 'SKIP') for r in rows))
        expect('report mode exit 3 when a copy is not clamped',
               exit_code([dict(status='NOT CLAMPED'), dict(status='CLAMPED')], False) == 3)
        expect('no CAdEx copy at all -> exit 1', exit_code([dict(status='SKIP')], False) == 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        import brian2
    except Exception:
        brian2 = None
    if brian2 is None:
        print('  %-66s %s' % ('Brian2 parses the patched equations', 'SKIPPED (no brian2)'))
    else:
        eqs = brian2.Equations(clamped.split("'''")[1])
        expect('Brian2 parses the patched equations; only V has the flag',
               'unless refractory' in eqs['V'].flags and not eqs['gA'].flags)
    print('SELF-TEST: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description='Find and clamp every CAdEx copy of ASD_fun_BD_cpp.py.')
    ap.add_argument('roots', nargs='*', default=[DEFAULT_ROOT])
    ap.add_argument('--apply', action='store_true', help='patch every NOT CLAMPED copy')
    ap.add_argument('--force-running', action='store_true',
                    help='also patch folders whose campaign output is recent')
    ap.add_argument('--recent-hours', type=float, default=CFG['recent_hours'])
    ap.add_argument('--max-depth', type=int, default=CFG['max_depth'])
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    missing = [r for r in a.roots if not os.path.isdir(r)]
    if missing:
        print('ERROR: not a folder: %s' % ', '.join(missing))
        return 1
    print('clamp_cadex_copies  %s  roots: %s  mode: %s' % (
        time.strftime('%Y-%m-%d %H:%M'), ' '.join(a.roots), 'APPLY' if a.apply else 'report'))
    rows = scan(a.roots, apply=a.apply, force_running=a.force_running,
                cfg=dict(recent_hours=a.recent_hours, max_depth=a.max_depth))
    return report(rows, a.apply)


if __name__ == '__main__':
    sys.exit(main())
