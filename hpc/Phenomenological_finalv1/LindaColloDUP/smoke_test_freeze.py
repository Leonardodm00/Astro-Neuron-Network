#!/usr/bin/env python3
"""
smoke_test_freeze.py -- checks the --freeze patch, the g box patch and the
LindaColloDUP axis set.

Run inside the LindaColloDUP folder, after apply_freeze_patch.py and
apply_box_patch.py, with the campaign environment active (numpy + matplotlib;
no brian2, no compilation):

    conda activate brian_final
    python smoke_test_freeze.py                 # ~10 s
    python smoke_test_freeze.py --freeze "..."  # test another freeze list

Expected last line:  RESULT: <n> passed, 0 failed
Exit status 0 iff every check passes.

Checks
  T1  the patch is present: _apply_freeze exists, --freeze parses
  T2  empty --freeze changes nothing
  T3  the campaign freeze list leaves exactly the 13 agreed free axes
  T4  NAME=VALUE replaces the nominal value; NAME alone keeps it
  T5  bad lists are refused (unknown, outside the group, repeated, bad value)
  T6  build_axis_declaration: 13 run_args axes + conn_prob swept, frozen axes
      fixed at the right values, no gliotransmission axis swept
  T7  main() end to end with --n_topologies 0: job_args.json records the
      13 free axes, the freeze list, the frozen values and the g boxes
      (no simulation)
  T8  the g box patch (decided 2026-10-30): g_ampa and g_nmda span
      [0.01, 10] nS, both log-uniform, theta box [ln 0.01, ln 10]; the file
      carries both markers
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

# The campaign's freeze list (keep identical to FREEZE in launch_LCDUP.sh).
CAMPAIGN_FREEZE = ('Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,'
                   'alpha_syn,O_G,Omega_G')
# The g boxes after apply_box_patch.py (keep identical to BOXES there).
CAMPAIGN_G_BOX = (0.01, 10.0)
BOX_MARK = '[LindaColloDUP box 2026-10-30]'
EXPECTED_FREE = {'Sigma', 'gbarA', 'delta_gA', 'tauA',
                 'U_0_ar', 'U_max', 'U_0_sr', 'Omega_f_sr', 'Omega_f_ar', 'Omega_d',
                 'g_ampa', 'g_nmda', 'x0'}
GLIO_COUPLING = {'alpha_syn', 'O_G', 'Omega_G'}   # inert under MODE=Neuronal

_n_pass = 0
_n_fail = 0


def check(cond, msg):
    global _n_pass, _n_fail
    if cond:
        _n_pass += 1
        print('  ok   ' + msg)
    else:
        _n_fail += 1
        print('  FAIL ' + msg)


def parse_values(spec):
    out = {}
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        name, _, val = item.partition('=')
        out[name.strip()] = float(val) if val.strip() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', default=CAMPAIGN_FREEZE)
    ap.add_argument('--group', default='neuron_synapse')
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import HPC_main_sweep as S

    print('=' * 72)
    print('smoke_test_freeze -- %s (N_DIMS=%d)' % (os.path.join(here, 'HPC_main_sweep.py'), S.N_DIMS))
    print('freeze list: ' + a.freeze)
    print('=' * 72)
    nominal0 = np.array(S.NOMINAL_PARAMS, dtype=np.float64).copy()
    # The 37-D pipeline has the astrocyte contact gate; pass it switched off
    # exactly as submit_LCDUP.sh does (the 36-D pipeline has no such flags).
    opts = {o for act in S.build_parser()._actions for o in act.option_strings}
    gate_off = (['--stoa_gate_lo', '1.0', '--stoa_gate_hi', '1.0']
                if '--stoa_gate_lo' in opts else [])
    print('contact-gate flags: %s' % ('present, passed as 1.0/1.0' if gate_off else 'absent (36-D)'))

    def reset():
        S.NOMINAL_PARAMS[:] = nominal0

    print('\n[T1] patch present')
    check(hasattr(S, '_apply_freeze'), '_apply_freeze defined')
    try:
        ns = S.build_parser().parse_args(['--out_dir', 'x', '--freeze', 'Cm'])
        check(ns.freeze == 'Cm', '--freeze parses into args.freeze')
    except SystemExit:
        check(False, '--freeze parses into args.freeze')
    ns0 = S.build_parser().parse_args(['--out_dir', 'x'])
    check(ns0.freeze == '', '--freeze defaults to empty')

    group = S.resolve_sweep_group(a.group)
    print('\n[T2] empty freeze is a no-op')
    check(S._apply_freeze('', group) == sorted(group), "'' returns the group unchanged")
    check(np.array_equal(S.NOMINAL_PARAMS, nominal0), "'' leaves NOMINAL_PARAMS unchanged")

    print('\n[T3] campaign axis set')
    reset()
    act = S._apply_freeze(a.freeze, group)
    names = [S.PARAM_NAMES[i] for i in act]
    check(set(names) == EXPECTED_FREE,
          'free axes == the agreed 13 (got %d: %s)' % (len(names), names))
    check(act == sorted(act), 'returned indices sorted')
    check(not set(names) & GLIO_COUPLING, 'alpha_syn, O_G, Omega_G not free')
    missing = [n for n in EXPECTED_FREE if n not in S.PARAM_NAMES]
    check(not missing, 'every expected axis exists in this registry %s' % (missing or ''))

    print('\n[T4] frozen values')
    vals = parse_values(a.freeze)
    for name, v in sorted(vals.items()):
        k = S.PARAM_NAMES.index(name)
        want = nominal0[k] if v is None else v
        check(abs(S.NOMINAL_PARAMS[k] - want) < 1e-12,
              '%-10s frozen at %g (%s)' % (name, want, 'nominal' if v is None else 'given'))
    untouched = [i for i in range(S.N_DIMS) if S.PARAM_NAMES[i] not in vals]
    check(np.array_equal(S.NOMINAL_PARAMS[untouched], nominal0[untouched]),
          'no other nominal value changed')
    # The campaign's VA/VR values equal the registry nominals, so the checks
    # above cannot see a broken NAME=VALUE path; test it with a value that is not.
    reset()
    kva = S.PARAM_NAMES.index('VA')
    probe = float(nominal0[kva]) - 2.0
    S._apply_freeze('VA=%g' % probe, group)
    check(abs(S.NOMINAL_PARAMS[kva] - probe) < 1e-12 and abs(probe - nominal0[kva]) > 1.0,
          'NAME=VALUE overrides a nominal (VA=%g, nominal %g)' % (probe, nominal0[kva]))
    reset()
    S._apply_freeze(a.freeze, group)

    print('\n[T5] bad freeze lists are refused')
    for bad, why in (('NoSuchAxis', 'unknown name'),
                     ('O_beta', 'axis outside the group'),
                     ('Cm,Cm', 'repeated name'),
                     ('gbarA=0', 'non-positive value on a log axis'),
                     ('VA=nan', 'non-finite value')):
        reset()
        try:
            S._apply_freeze(bad, group)
            check(False, '%-14r refused (%s)' % (bad, why))
        except ValueError:
            check(True, '%-14r refused (%s)' % (bad, why))
    reset()

    print('\n[T6] axis declaration')
    act = S._apply_freeze(a.freeze, group)
    ns = S.build_parser().parse_args(['--out_dir', 'x', '--mode', 'Neuronal',
                                      '--sweep_group', a.group, '--freeze', a.freeze]
                                     + gate_off)
    d = S.build_axis_declaration(ns, act)
    run_level = {k for k, v in d['swept_axes'].items() if v.get('level') == 'run_args'}
    check(run_level == EXPECTED_FREE, 'swept run_args axes == the 13')
    check('conn_prob' in d['swept_axes'], 'conn_prob swept at topology level')
    for name, v in vals.items():
        if v is not None:
            check(abs(d['fixed_axes'].get(name, float('nan')) - v) < 1e-12,
                  'declaration fixes %s at %g' % (name, v))
    check(not run_level & GLIO_COUPLING, 'no gliotransmission-coupling axis swept')
    reset()

    print('\n[T7] main() end to end, --n_topologies 0 (no simulation)')
    tmp = tempfile.mkdtemp(prefix='freeze_smoke_')
    try:
        argv = ['HPC_main_sweep.py', '--out_dir', tmp, '--lib_dir', here,
                '--n_topologies', '0', '--n_workers', '1', '--mode', 'Neuronal',
                '--sweep_group', a.group, '--freeze', a.freeze,
                '--density', '1300', '--c_max', '350', '--Na', '0',
                '--seed_master', '12345'] + gate_off
        old = sys.argv
        sys.argv = argv
        try:
            S.main()
        finally:
            sys.argv = old
        ja = json.load(open(os.path.join(tmp, 'job_args.json')))
        check(set(ja.get('_active_param_names', [])) == EXPECTED_FREE,
              'job_args.json _active_param_names == the 13')
        check(ja.get('freeze') == a.freeze, 'job_args.json records the freeze list')
        check(ja.get('Nn') == 159 and ja.get('Na') == 0,
              'population Nn=159, Na=0 (got Nn=%s, Na=%s)' % (ja.get('Nn'), ja.get('Na')))
        fx = ja.get('_axis_declaration', {}).get('fixed_axes', {})
        for name, v in vals.items():
            if v is not None:
                check(abs(fx.get(name, float('nan')) - v) < 1e-12,
                      'job_args fixed_axes %s == %g' % (name, v))
        sw = ja.get('_axis_declaration', {}).get('swept_axes', {})
        check(all(abs(sw.get(g, {}).get('low', -1) - CAMPAIGN_G_BOX[0]) < 1e-12
                  and abs(sw.get(g, {}).get('high', -1) - CAMPAIGN_G_BOX[1]) < 1e-12
                  and sw.get(g, {}).get('log') is True for g in ('g_ampa', 'g_nmda')),
              'job_args swept_axes g_ampa, g_nmda == [%g, %g] nS, log' % CAMPAIGN_G_BOX)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        reset()

    print('\n[T8] g box patch')
    src = open(os.path.join(here, 'HPC_main_sweep.py'), encoding='utf-8').read()
    check(src.count(BOX_MARK) == 2, 'HPC_main_sweep.py carries both box markers (got %d)' % src.count(BOX_MARK))
    for g in ('g_ampa', 'g_nmda'):
        k = S.PARAM_NAMES.index(g)
        lo, hi = (float(x) for x in S.PARAM_BOUNDS[k])
        tlo, thi = (float(x) for x in S.PARAM_BOUNDS_THETA[k])
        check(abs(lo - CAMPAIGN_G_BOX[0]) < 1e-12 and abs(hi - CAMPAIGN_G_BOX[1]) < 1e-12
              and k in S.LOG_PARAMS
              and abs(tlo - np.log(CAMPAIGN_G_BOX[0])) < 1e-12 and abs(thi - np.log(CAMPAIGN_G_BOX[1])) < 1e-12,
              '%s box [%g, %g] nS, log, theta [%.4f, %.4f]' % (g, lo, hi, tlo, thi))

    print('\n' + '=' * 72)
    print('RESULT: %d passed, %d failed' % (_n_pass, _n_fail))
    print('=' * 72)
    return 1 if _n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
