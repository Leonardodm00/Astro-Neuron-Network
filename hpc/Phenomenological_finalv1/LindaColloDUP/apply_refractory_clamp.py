#!/usr/bin/env python3
"""
apply_refractory_clamp.py -- hold V at VR during the refractory period.

OPTIONAL. Run it only if you decide to clamp (see APPLY_LCDUP.txt, step 5),
inside the LindaColloDUP folder, on its own copy of ASD_fun_BD_cpp.py. Never
run it on MEAMA/.

Why
    In the shipped neuron model (ASD_fun_BD_cpp.py, Neuronal_Network) dV/dt
    keeps integrating during the refractory period t_ref, while Brian2 ignores
    the threshold there (its threshold code is "(V > VD) and not_refractory").
    A neuron pushed above VT during t_ref -- by the noise term or by synaptic
    input -- runs away exponentially, V becomes inf/NaN, and the neuron never
    spikes again. The sweep keeps such runs: it discards a run only when spike
    TIMES are non-finite. Gorski et al. 2021 (Sec. 2) define the model with V
    clamped: after reaching VD "the voltage is reset to the reset potential VR,
    and it remains at this value during a refractory period t_ref".

What it changes (nothing else)
    1. appends ' (unless refractory)' to the single dV/dt line of the neuron
       equations, so V stays at VR during t_ref (gA keeps evolving, as in
       Gorski 2021);
    2. adds one comment line, inside the equation string, above that line.
    The dV/dt line must match exactly once; the patch is verified before and
    after writing, a timestamped backup is kept, and the backup is restored if
    the result does not compile. Running it again changes nothing.

Usage (campaign environment active)
    python apply_refractory_clamp.py --check       # anchor only, writes nothing
    python apply_refractory_clamp.py               # patch ./ASD_fun_BD_cpp.py
    python apply_refractory_clamp.py --file PATH   # patch another copy
    python apply_refractory_clamp.py --self-test   # fixture test (+ Brian2 parse if available)
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
import warnings

MARK = '[LindaColloDUP clamp 2026-10-29]'
FLAG = ' (unless refractory)'
DV = re.compile(r'^([ \t]*)(dV/dt[ \t]*=.*?:[ \t]*volt)([ \t]*(\([^()\n]*\))?)[ \t]*$', re.M)


def _comment(indent):
    return ('%s# %s V held at VR during t_ref (Gorski et al. 2021, Sec. 2)\n'
            % (indent, MARK))


def patch_text(text):
    """Return (new_text, status, problems); status in {'patched', 'already', 'refused'}."""
    hits = list(DV.finditer(text))
    n_mark = text.count(MARK)
    if len(hits) != 1:
        return text, 'refused', ['dV/dt line found %d times (need exactly 1)' % len(hits)]
    m = hits[0]
    flags = (m.group(4) or '').strip()
    if flags == FLAG.strip():
        if n_mark <= 1:
            return text, 'already', ([] if n_mark == 1 else
                                     ['V is already clamped, without this patch marker'])
        return text, 'refused', ['patch marker repeated %d times' % n_mark]
    if n_mark:
        return text, 'refused', ['partially patched file (marker present, dV/dt flags %r); '
                                 'restore the pre-patch copy' % (flags or 'none')]
    if flags:
        return text, 'refused', ['the dV/dt line already carries flags %r; refusing to '
                                 'guess' % flags]
    new_line = m.group(1) + m.group(2) + FLAG
    text = text[:m.start()] + _comment(m.group(1)) + new_line + text[m.end():]
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
        print('ALREADY CLAMPED: %s (dV/dt ends with%s); nothing written' % (path, FLAG))
        for p in problems:
            print('  note: ' + p)
        return 0
    if status == 'refused':
        print('REFUSED: nothing written to %s' % path)
        for p in problems:
            print('  - ' + p)
        return 1
    if check_only:
        line = DV.search(text).group(0).strip()
        print('CHECK OK: one dV/dt line in %s; nothing written (--check)' % path)
        print('  current : %s' % line)
        print('  would be: %s' % (line + FLAG))
        return 0
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = '%s.bak-clamp-%s' % (path, stamp)
    shutil.copy2(path, backup)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix='.clamp_patch_', dir=d)
    with os.fdopen(fd, 'wb') as fh:
        fh.write(new.encode('utf-8'))
    shutil.copymode(path, tmp)          # keep the original permissions (mkstemp gives 0600)
    os.replace(tmp, path)
    back = open(path, 'rb').read().decode('utf-8')
    ok = back == new and patch_text(back)[1] == 'already'
    try:
        with warnings.catch_warnings():
            # the file's own '$\mu m$' plot labels raise SyntaxWarning on any
            # compile; they predate this patch and are not touched by it
            warnings.simplefilter('ignore', SyntaxWarning)
            py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as e:
        ok = False
        print('compile failed: %s' % e)
    if not ok:
        shutil.copy2(backup, path)
        print('FAILED: verification failed; original restored from %s' % backup)
        return 1
    print('CLAMPED: %s  (dV/dt now ends with%s; backup %s)' % (path, FLAG, backup))
    print('  %s' % DV.search(back).group(0).strip())
    return 0


_FIXTURE = (
    "eqs_NN = Equations('''\n"
    "\n"
    "    # CAdEx membrane dynamics\n"
    "    dV/dt  = noise + (gl*(El-V) + gl*DeltaT*exp((V-VT)/DeltaT) + gA*(EA-V) + I_inj*bias_unit - I_syn)/Cm : volt\n"
    "    dgA/dt = (gbarA/(1 + exp((VA-V)/DeltaA)) - gA)/tauA                                     : siemens\n"
    "    noise = sigma*(2*gl_ref/Cm)**.5*randn()/sqrt(dt) : volt/second (constant over dt)\n"
    "    I_syn : amp\n"
    "    sigma : volt\n"
    "    gl : siemens\n"
    "    gbarA : siemens\n"
    "    tauA : second\n"
    "    DeltaT : volt\n"
    "    VT : volt\n"
    "    VA : volt\n"
    "    DeltaA : volt\n"
    "    I_inj : amp\n"
    "    bias_unit : 1\n"
    "    Cm : farad\n"
    "    ''')\n"
)


def self_test():
    ok = True

    def expect(label, cond):
        nonlocal ok
        print('  %-60s %s' % (label, 'ok' if cond else 'FAIL'))
        ok = ok and bool(cond)

    print('SELF-TEST on an embedded fixture')
    new, st, pr = patch_text(_FIXTURE)
    expect('fixture patches', st == 'patched' and not pr)
    expect('one marker inserted', new.count(MARK) == 1)
    expect('dV/dt line ends with (unless refractory)',
           DV.search(new).group(0).rstrip().endswith('(unless refractory)'))
    expect('dgA/dt line unchanged', new.count('- gA)/tauA') == 1
           and '(unless refractory)' not in new.split('dgA/dt')[1].split('\n')[0])
    expect('only the comment line and the flag were added',
           new.replace(_comment('    '), '').replace(FLAG, '') == _FIXTURE)
    again, st2, _ = patch_text(new)
    expect('second run is a no-op (already)', st2 == 'already' and again == new)
    _, st3, _ = patch_text(_FIXTURE.replace('dV/dt', 'dU/dt'))
    expect('missing dV/dt line -> refused', st3 == 'refused')
    _, st4, _ = patch_text(_FIXTURE + _FIXTURE)
    expect('two dV/dt lines -> refused', st4 == 'refused')
    _, st5, _ = patch_text(new.replace(FLAG, ''))
    expect('marker without flag (partial) -> refused', st5 == 'refused')
    _, st6, pr6 = patch_text(_FIXTURE.replace(': volt\n', ': volt (unless refractory)\n', 1))
    expect('flag without marker -> already, with a note', st6 == 'already' and pr6)
    _, st7, _ = patch_text(_FIXTURE.replace(': volt\n', ': volt (constant over dt)\n', 1))
    expect('other flags on the dV/dt line -> refused', st7 == 'refused')
    expect('inserted text is pure ASCII', all(ord(c) < 128 for c in _comment('    ') + FLAG))
    try:
        import brian2
    except Exception:
        brian2 = None
    if brian2 is None:
        print('  %-60s %s' % ('Brian2 parses the patched equations', 'SKIPPED (no brian2)'))
    else:
        body = new.split("'''")[1]
        eqs = brian2.Equations(body)
        expect('Brian2 parses the patched equations; V has the flag',
               'unless refractory' in eqs['V'].flags and not eqs['gA'].flags)
    print('SELF-TEST: %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description='Clamp V during t_ref in ASD_fun_BD_cpp.py (LindaColloDUP).')
    ap.add_argument('--file', default='ASD_fun_BD_cpp.py')
    ap.add_argument('--check', action='store_true', help='verify the anchor only; write nothing')
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
