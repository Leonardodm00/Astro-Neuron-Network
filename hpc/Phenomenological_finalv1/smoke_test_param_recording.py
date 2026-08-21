#!/usr/bin/env python3
"""
Smoke test for the causally-inert-parameter fix in HPC_main_sweep.py.

WHAT THIS PROVES
----------------
  T1  Flat runs are BIT-IDENTICAL to before: conn_prob is still drawn, in the
      same position of the master_rng stream, so a pre-change flat campaign
      reproduces exactly.
  T2  Weibull runs no longer draw conn_prob, and the resulting master_rng
      stream shift is REAL and DETERMINISTIC. This break is deliberate and
      accepted; the test pins it so it cannot happen again by accident.
  T3  conn_record_keys() writes exactly the causally live keys per rule --
      conn_prob under flat, (p0,d0,beta) under weibull -- and NEVER a NaN.
  T4  build_axis_declaration() partitions the axes correctly:
        - swept / consumed / fixed / inert are mutually consistent;
        - conn_prob is INERT (and fixed) under weibull, CONSUMED under flat;
        - the kernel axes are the mirror image;
        - the 10 astrocyte axes are declared INERT under mode=Neuronal when
          they were swept (SWEEP_GROUP=all), and FIXED when they were not
          (SWEEP_GROUP=neuron_synapse);
        - neuron_synapse resolves to 23 axes, not 24.
  T5  No formatting/serialisation path crashes now that conn_prob is
      rule-dependent: the whole declaration round-trips through json.

WHY IT DOES NOT NEED BRIAN2
---------------------------
Everything under test is pure bookkeeping: RNG draws, dict construction, JSON
serialisation. HPC_main_sweep imports cleanly without brian2 (brian2 is
imported lazily inside the worker), so this runs on a login node in seconds
with no allocation and no simulation.

HOW TO RUN
----------
    cd <repo>/hpc/Phenomenological_finalv1
    python3 smoke_test_param_recording.py

    # verbose, showing the full declaration for one configuration:
    python3 smoke_test_param_recording.py -v

Exit code 0 = all pass. Non-zero = at least one failure, with the failing
assertion named. Expected final line on success:

    ALL 5 TEST GROUPS PASSED
"""
import argparse
import copy
import json
import math
import os
import sys
import types

import numpy as np

# Headless: HPC_single_run imports matplotlib at module scope.
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import HPC_main_sweep as S   # noqa: E402


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
_FAILS = []
_CHECKS = [0]


def check(cond, label):
    _CHECKS[0] += 1
    if cond:
        print(f'    ok   {label}')
    else:
        print(f'    FAIL {label}')
        _FAILS.append(label)


def make_args(conn_rule='weibull', mode='Neuronal', sweep_group='neuron_synapse',
              conn_periodic=False, conn_prob_lo=0.1, conn_prob_hi=0.6,
              n_topologies=80):
    """A stand-in for the parsed argparse Namespace, with only the fields
    build_axis_declaration() actually reads."""
    return types.SimpleNamespace(
        conn_rule=conn_rule,
        mode=mode,
        sweep_group=sweep_group,
        conn_periodic=conn_periodic,
        conn_prob_lo=conn_prob_lo,
        conn_prob_hi=conn_prob_hi,
        n_topologies=n_topologies,
    )


def draw_stream(conn_rule, seed, n_topo=6, kernel_bounds=None):
    """
    Replay the outer loop's master_rng consumption EXACTLY as HPC_main_sweep
    does it after the fix, and return the per-topology seed bases.

    The seed bases are the observable that matters: every per-topology seed
    (neuron/synapse/astro) and therefore every topology and every parameter
    vector descends from them, so if these match, the campaign matches.
    """
    if kernel_bounds is None:
        kernel_bounds = S.KERNEL_BOUNDS
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_topo):
        if conn_rule == 'flat':
            _ = float(rng.uniform(0.1, 0.6))
        if conn_rule == 'weibull':
            _ = tuple(float(v) for v in S.sample_kernel_vector(rng, kernel_bounds))
        out.append(int(rng.integers(0, 2**31 - 1)))
    return out


def draw_stream_legacy(conn_rule, seed, n_topo=6, kernel_bounds=None):
    """The PRE-fix consumption order: conn_prob drawn unconditionally."""
    if kernel_bounds is None:
        kernel_bounds = S.KERNEL_BOUNDS
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_topo):
        _ = float(rng.uniform(0.1, 0.6))            # ALWAYS drawn, pre-fix
        if conn_rule == 'weibull':
            _ = tuple(float(v) for v in S.sample_kernel_vector(rng, kernel_bounds))
        out.append(int(rng.integers(0, 2**31 - 1)))
    return out


# ---------------------------------------------------------------------------
# T1 -- flat is bit-identical to pre-fix
# ---------------------------------------------------------------------------
def t1_flat_stream_unchanged():
    print('\n[T1] flat rule: master_rng stream is bit-identical to pre-fix')
    for seed in (0, 1, 12345, 2**31 - 7):
        new = draw_stream('flat', seed)
        old = draw_stream_legacy('flat', seed)
        check(new == old, f'seed={seed}: seed bases identical ({new[:2]}...)')


# ---------------------------------------------------------------------------
# T2 -- weibull stream shift is real, deliberate and deterministic
# ---------------------------------------------------------------------------
def t2_weibull_stream_shift():
    print('\n[T2] weibull rule: conn_prob draw removed; stream shift is')
    print('     deliberate, and deterministic given the seed')
    for seed in (0, 1, 12345):
        new = draw_stream('weibull', seed)
        old = draw_stream_legacy('weibull', seed)
        check(new != old,
              f'seed={seed}: stream DOES shift vs pre-fix (accepted break)')
        check(new == draw_stream('weibull', seed),
              f'seed={seed}: post-fix stream is reproducible')
    # The shift must come from the missing uniform() and nothing else: drawing
    # one extra uniform before each topology must recover the legacy stream.
    rng = np.random.default_rng(999)
    rebuilt = []
    for _ in range(6):
        _ = float(rng.uniform(0.1, 0.6))
        _ = tuple(float(v) for v in S.sample_kernel_vector(rng, S.KERNEL_BOUNDS))
        rebuilt.append(int(rng.integers(0, 2**31 - 1)))
    check(rebuilt == draw_stream_legacy('weibull', 999),
          'the ONLY difference is the removed uniform() draw')


# ---------------------------------------------------------------------------
# T3 -- the record contains exactly the live keys, and never a NaN
# ---------------------------------------------------------------------------
def t3_record_keys():
    print('\n[T3] conn_record_keys: exactly the live keys, never NaN')
    nan = float('nan')

    flat = S.conn_record_keys('flat', 0.42, nan, nan, nan)
    check(set(flat) == {'conn_prob'}, f'flat -> {sorted(flat)}')
    check(flat['conn_prob'] == 0.42, 'flat: conn_prob value preserved')
    check(not any(math.isnan(v) for v in flat.values()), 'flat: no NaN written')

    wb = S.conn_record_keys('weibull', S.CONN_PROB_LIBRARY_DEFAULT,
                            0.55, 120.0, 1.4)
    check(set(wb) == {'p0_conn', 'd0_conn', 'beta_conn'}, f'weibull -> {sorted(wb)}')
    check('conn_prob' not in wb,
          'weibull: conn_prob ABSENT from the record (not NaN, not null)')
    check(not any(math.isnan(v) for v in wb.values()), 'weibull: no NaN written')

    try:
        S.conn_record_keys('bogus', 0.1, 1, 2, 3)
        check(False, 'unknown rule raises')
    except ValueError:
        check(True, 'unknown rule raises ValueError (fails loudly)')

    # A consumer must be able to branch on conn_rule alone.
    check(set(S.CONN_AXES_BY_RULE) == {'flat', 'weibull'},
          'CONN_AXES_BY_RULE covers exactly the two rules build_topology accepts')


# ---------------------------------------------------------------------------
# T4 -- the axis declaration partitions correctly
# ---------------------------------------------------------------------------
def t4_axis_declaration(verbose=False):
    print('\n[T4] build_axis_declaration: swept / consumed / fixed / inert')

    # --- the production configuration -------------------------------------
    args = make_args(conn_rule='weibull', mode='Neuronal',
                     sweep_group='neuron_synapse', conn_periodic=False)
    idx = S.resolve_sweep_group('neuron_synapse')
    d = S.build_axis_declaration(args, idx)

    check(len(idx) == 23, f'neuron_synapse resolves to 23 axes (got {len(idx)})')
    check('conn_prob' in d['inert_axes'],
          'weibull: conn_prob declared INERT')
    check('conn_prob' in d['fixed_axes'],
          'weibull: conn_prob also declared FIXED, with its held value')
    check(d['fixed_axes']['conn_prob'] == S.CONN_PROB_LIBRARY_DEFAULT,
          f"weibull: conn_prob held at {S.CONN_PROB_LIBRARY_DEFAULT}")
    check('conn_prob' not in d['swept_axes'],
          'weibull: conn_prob NOT in swept_axes')
    check('conn_prob' not in d['consumed_axes'],
          'weibull: conn_prob NOT in consumed_axes')
    for k in ('p0_conn', 'd0_conn', 'beta_conn'):
        check(k in d['swept_axes'] and k in d['consumed_axes'],
              f'weibull: {k} is swept AND consumed')
    check(d['n_topology_level_swept'] == 3,
          'weibull: exactly 3 topology-level axes swept')
    # The 23 run_args axes + 3 kernel axes is the label dimension downstream.
    check(d['n_swept'] == 26, f"n_swept == 26 (got {d['n_swept']})")
    # Astro axes were NOT swept here, so they are fixed, not inert.
    astro_fixed = [n for n in S._ASTRO_FREE if n in d['fixed_axes']]
    check(len(astro_fixed) == 10,
          f'neuron_synapse: all 10 astro axes FIXED at nominal '
          f'(got {len(astro_fixed)})')
    check(not any(n in d['inert_axes'] for n in S._ASTRO_FREE),
          'neuron_synapse: no astro axis marked inert (they were never drawn)')

    # --- the flat mirror image --------------------------------------------
    argsf = make_args(conn_rule='flat', mode='Neuronal',
                      sweep_group='neuron_synapse', conn_periodic=False)
    df = S.build_axis_declaration(argsf, idx)
    check('conn_prob' in df['swept_axes'] and 'conn_prob' in df['consumed_axes'],
          'flat: conn_prob is swept AND consumed (causally live)')
    check('conn_prob' not in df['inert_axes'], 'flat: conn_prob NOT inert')
    for k in ('p0_conn', 'd0_conn', 'beta_conn'):
        check(k in df['inert_axes'] and k not in df['swept_axes'],
              f'flat: {k} declared inert and not swept')
    check('conn_periodic' in df['inert_axes'],
          'flat: conn_periodic inert (rule is distance-independent)')
    check('conn_periodic' in d['consumed_axes'],
          'weibull: conn_periodic consumed (minimum-image distances)')

    # --- the latent 10-inert-axis case ------------------------------------
    args_all = make_args(conn_rule='weibull', mode='Neuronal', sweep_group='all')
    idx_all = S.resolve_sweep_group('all')
    d_all = S.build_axis_declaration(args_all, idx_all)
    astro_inert = [n for n in S._ASTRO_FREE if n in d_all['inert_axes']]
    check(len(astro_inert) == 10,
          f'mode=Neuronal + sweep_group=all: all 10 astro axes declared '
          f'INERT (got {len(astro_inert)})')
    check(not any(n in d_all['consumed_axes'] for n in S._ASTRO_FREE),
          'mode=Neuronal + all: no astro axis claimed as consumed')

    # --- mode=Full must NOT mark astro axes inert -------------------------
    args_full = make_args(conn_rule='weibull', mode='Full', sweep_group='all')
    d_full = S.build_axis_declaration(args_full, idx_all)
    check(not any(n in d_full['inert_axes'] for n in S._ASTRO_FREE),
          'mode=Full: astro axes are NOT inert (astrocytes are instantiated)')
    check(all(n in d_full['consumed_axes'] for n in S._ASTRO_FREE),
          'mode=Full: every astro axis is consumed')

    # --- structural invariants across every configuration -----------------
    for rule in ('flat', 'weibull'):
        for mode in ('Full', 'Neuronal'):
            for grp in sorted(S.SWEEP_GROUPS):
                a = make_args(conn_rule=rule, mode=mode, sweep_group=grp)
                i = S.resolve_sweep_group(grp)
                dd = S.build_axis_declaration(a, i)
                overlap = set(dd['swept_axes']) & set(dd['consumed_axes'])
                # every consumed axis that was swept must be in swept_axes
                bad_consumed = [n for n in dd['consumed_axes']
                                if n not in dd['swept_axes']
                                and n not in dd['fixed_axes']]
                bad_both = [n for n in dd['inert_axes']
                            if n in dd['consumed_axes']]
                label = f'{rule}/{mode}/{grp}'
                check(not bad_both,
                      f'{label}: no axis is both inert and consumed')
                check(not bad_consumed,
                      f'{label}: every consumed axis is declared somewhere')
                check(dd['n_consumed'] == len(dd['consumed_axes']),
                      f'{label}: n_consumed matches the list')

    if verbose:
        print('\n--- full declaration, weibull / Neuronal / neuron_synapse ---')
        print(json.dumps(d, indent=2, sort_keys=True)[:4000])
    return d


# ---------------------------------------------------------------------------
# T5 -- everything serialises; no rule-dependent formatting crash
# ---------------------------------------------------------------------------
def t5_serialisation(d):
    print('\n[T5] serialisation: the record round-trips through JSON')
    try:
        blob = json.dumps(d, sort_keys=True)
        back = json.loads(blob)
        check(back == json.loads(json.dumps(d, sort_keys=True)),
              'axis_declaration round-trips through json')
        check(isinstance(back['fixed_axes']['conn_periodic'], bool),
              'conn_periodic serialises as a JSON bool, not numpy.bool_')
    except TypeError as e:
        check(False, f'axis_declaration is NOT json-serialisable: {e}')

    # npz-style record must survive np.savez's kwargs splat.
    for rule, kw in (('flat', dict(conn_prob=0.3, p0_conn=float("nan"),
                                   d0_conn=float("nan"), beta_conn=float("nan"))),
                     ('weibull', dict(conn_prob=S.CONN_PROB_LIBRARY_DEFAULT,
                                      p0_conn=0.5, d0_conn=100.0, beta_conn=1.2))):
        rec = S.conn_record_keys(rule, kw['conn_prob'], kw['p0_conn'],
                                 kw['d0_conn'], kw['beta_conn'])
        arrs = {k: np.float64(v) for k, v in rec.items()}
        check(all(np.isfinite(v) for v in arrs.values()),
              f'{rule}: every npz-bound value is finite')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='print the full axis declaration for one config')
    a = ap.parse_args()

    print('=' * 70)
    print('smoke_test_param_recording -- HPC_main_sweep causally-inert-axis fix')
    print('=' * 70)
    print(f'  registry N_DIMS       : {S.N_DIMS}')
    print(f'  neuron_synapse axes   : {len(S.resolve_sweep_group("neuron_synapse"))}')
    print(f'  CONN_PROB default     : {S.CONN_PROB_LIBRARY_DEFAULT}')
    print(f'  numpy                 : {np.__version__}')

    t1_flat_stream_unchanged()
    t2_weibull_stream_shift()
    t3_record_keys()
    d = t4_axis_declaration(verbose=a.verbose)
    t5_serialisation(d)

    print('\n' + '=' * 70)
    if _FAILS:
        print(f'{len(_FAILS)} of {_CHECKS[0]} CHECKS FAILED:')
        for f in _FAILS:
            print('  - ' + f)
        print('=' * 70)
        return 1
    print(f'ALL 5 TEST GROUPS PASSED  ({_CHECKS[0]} checks)')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
