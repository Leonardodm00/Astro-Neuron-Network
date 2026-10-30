#!/usr/bin/env python3
"""
smoke_test_clamp_cadex_copies.py -- checks clamp_cadex_copies.py.

  S1  the tool's own --self-test passes (embedded fixtures, Brian2 parse if available)
  S2  a mock of the cluster's project folders built from the REAL model files
      of this repository (Main/, Main/Giulia_Astro/, MEAMA/ unclamped;
      LindaColloDUP/ clamped; an HH model): report exit 3 with the right
      statuses; --apply exit 0; every patched file differs from its source
      only by the comment line and the flag; the HH file and the clamped file
      are untouched; one backup per patched file; a second --apply is a no-op
  S3  pure ASCII and LF-only bytes of the tool and of this test

Run (any Python 3 with numpy; brian2 optional):
    python3 smoke_test_clamp_cadex_copies.py | tail -1
S2 needs ../Phenomenological_finalv1/ASD_fun_BD_cpp.py (a repository clone);
outside a clone it is skipped and the summary says so.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, 'clamp_cadex_copies.py')
REPO_CADEX = os.path.join(HERE, '..', 'Phenomenological_finalv1', 'ASD_fun_BD_cpp.py')
REPO_GIULIA = os.path.join(HERE, '..', 'Phenomenological_finalv1', 'Giulia', 'ASD_fun_BD_cpp.py')
REPO_HH = os.path.join(HERE, '..', 'ASD_fun_BD_cpp.py')
RESULTS = {'passed': 0, 'failed': 0, 'skipped': 0}


def check(cond, label):
    RESULTS['passed' if cond else 'failed'] += 1
    print('  %s %s' % ('ok  ' if cond else 'FAIL', label))


def run(*args):
    return subprocess.run([sys.executable, TOOL] + list(args), capture_output=True, text=True)


def unclamp(text):
    """The pre-fix form of a clamped repository file: comment lines and flag removed."""
    out = []
    for line in text.split('\n'):
        if '# V is held at VR during t_ref, as in Gorski' in line or \
           "'(unless refractory)' V keeps integrating" in line or \
           'test, so a neuron pushed past the unstable point' in line or \
           'never spikes again (refractory audit, 2026-10-30).' in line:
            continue
        out.append(line.replace(' : volt (unless refractory)', ' : volt'))
    return '\n'.join(out)


def s1():
    print('\n[S1] the tool\'s self-test')
    r = run('--self-test')
    check(r.returncode == 0 and 'SELF-TEST: PASS' in r.stdout, 'self-test passes (exit %d)' % r.returncode)


def s2():
    print('\n[S2] mock cluster tree from the repository\'s real model files')
    if not (os.path.isfile(REPO_CADEX) and os.path.isfile(REPO_GIULIA) and os.path.isfile(REPO_HH)):
        RESULTS['skipped'] += 1
        print('  SKIP repository model files not found next to this folder (not a clone)')
        return
    main_txt = open(REPO_CADEX, encoding='utf-8').read()
    giulia_txt = open(REPO_GIULIA, encoding='utf-8').read()
    hh_txt = open(REPO_HH, encoding='utf-8').read()
    src = {'Main': unclamp(main_txt), 'Main/Giulia_Astro': unclamp(giulia_txt),
           'MEAMA': unclamp(main_txt), 'LindaColloDUP': main_txt, 'hhmodel': hh_txt}
    check(' : volt (unless refractory)' not in src['Main'] and src['Main'] != main_txt,
          'unclamped sources rebuilt from the clamped repository files')
    root = tempfile.mkdtemp(prefix='clamp_smoke_')
    try:
        for d, txt in src.items():
            os.makedirs(os.path.join(root, d), exist_ok=True)
            open(os.path.join(root, d, 'ASD_fun_BD_cpp.py'), 'w', encoding='utf-8').write(txt)
        r = run(root)
        check(r.returncode == 3, 'report exit 3 (some copies not clamped): %d' % r.returncode)
        lines = r.stdout.splitlines()
        stat = {}
        for ln in lines:
            for d in src:
                if ln.rstrip().endswith(os.path.join(root, d, 'ASD_fun_BD_cpp.py')):
                    stat[d] = ln.split()[0] + (' ' + ln.split()[1] if ln.startswith('NOT') else '')
        check(stat == {'Main': 'NOT CLAMPED', 'Main/Giulia_Astro': 'NOT CLAMPED', 'MEAMA': 'NOT CLAMPED',
                       'LindaColloDUP': 'CLAMPED', 'hhmodel': 'SKIP'}, 'report statuses %s' % stat)
        r = run(root, '--apply')
        check(r.returncode == 0 and 'ALL CAdEx COPIES CLAMPED' in r.stdout, 'apply exit 0, verdict all clamped')
        for d in ('Main', 'Main/Giulia_Astro', 'MEAMA'):
            new = open(os.path.join(root, d, 'ASD_fun_BD_cpp.py'), encoding='utf-8').read()
            gone = [l for l in new.split('\n') if '[refractory clamp]' not in l]
            check('\n'.join(gone).replace(' : volt (unless refractory)', ' : volt') == src[d] and
                  new.count('(unless refractory)') == 1, '%s: only the comment line and the flag added' % d)
            baks = [n for n in os.listdir(os.path.join(root, d)) if n.startswith('ASD_fun_BD_cpp.py.bak-clamp-')]
            check(len(baks) == 1 and open(os.path.join(root, d, baks[0]), encoding='utf-8').read() == src[d],
                  '%s: one backup, identical to the original' % d)
        check(open(os.path.join(root, 'hhmodel', 'ASD_fun_BD_cpp.py'), encoding='utf-8').read() == hh_txt,
              'HH model untouched')
        check(open(os.path.join(root, 'LindaColloDUP', 'ASD_fun_BD_cpp.py'), encoding='utf-8').read() == main_txt,
              'already clamped copy untouched')
        r = run(root, '--apply')
        check(r.returncode == 0 and 'PATCHED' not in r.stdout, 'second apply is a no-op (exit 0)')
    finally:
        shutil.rmtree(root, ignore_errors=True)


def s3():
    print('\n[S3] bytes')
    for p in (TOOL, os.path.abspath(__file__)):
        b = open(p, 'rb').read()
        check(b.count(b'\r') == 0 and all(x < 128 for x in b), '%s: 0 CR, pure ASCII' % os.path.basename(p))


def main():
    s1()
    s2()
    s3()
    print('\nRESULT: %d passed, %d failed%s' % (RESULTS['passed'], RESULTS['failed'],
          (', %d skipped' % RESULTS['skipped']) if RESULTS['skipped'] else ''))
    return 0 if RESULTS['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
