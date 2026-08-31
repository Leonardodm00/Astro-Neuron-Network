#!/usr/bin/env python3
# =============================================================================
# smoke_test_stoa_gate.py
#
# Self-contained correctness checks for two coupled 2026-08 changes:
#
#   (1) O_N registry port (Giulia table): the run_args registry is 37-D, with
#       O_N (astrocyte mGluR glutamate-binding rate) at index 36, bounds
#       [0.03, 3.0] (log), nominal 0.3; I_bias capped at [0.3, 1.0]; and a new
#       'tripartite' sweep group = neuron(9) + synapse(14) + the Giulia astro
#       table (I_Theta / omega_I frozen at nominal) = 32 run_args axes.
#
#   (2) Astrocyte contact gate: per topology, every VIABLE synapse->astrocyte
#       link is retained i.i.d. with probability stoa_gate_p in [0, 1], drawn
#       on a DEDICATED RNG. Both spillover directions are built downstream
#       from the same gated list. stoa_gate_p >= 1.0 makes NO draw, so the
#       gate-off path is bit-identical to pre-gate builds.
#
# Usage (login node, brian_env -- no scheduler, no compilation, ~30 s):
#     python smoke_test_stoa_gate.py
#     python smoke_test_stoa_gate.py -v      # verbose
#
# Exit status: 0 iff every check passes.
# =============================================================================

import sys
import types

import numpy as np

_VERBOSE = '-v' in sys.argv

_n_pass = 0
_n_fail = 0


def check(cond, msg):
    global _n_pass, _n_fail
    if cond:
        _n_pass += 1
        if _VERBOSE:
            print(f'  ok   {msg}')
    else:
        _n_fail += 1
        print(f'  FAIL {msg}')


def _fake_syn_pdist():
    """Small chi2(4)-like Sholl PDF standing in for synapse_pdist.csv."""
    import pandas as pd
    r = np.linspace(1.0, 120.0, 60)
    pdf = (r / 20.0) * np.exp(-r / 20.0)
    pdf = pdf / pdf.sum()
    return pd.DataFrame({'Syn_prob': pdf, 'Radius_val': r})


def _args(mode='Full', group='tripartite', rule='flat',
          gate_lo=1.0, gate_hi=1.0):
    return types.SimpleNamespace(
        mode=mode, sweep_group=group, conn_rule=rule,
        conn_prob_lo=0.1, conn_prob_hi=0.6, conn_periodic=False,
        n_topologies=80, stoa_gate_lo=gate_lo, stoa_gate_hi=gate_hi,
    )


def t1_registry(S):
    print('\n[T1] 37-D registry: O_N port + Giulia astro table')
    check(S.N_DIMS == 37, f'N_DIMS == 37 (got {S.N_DIMS})')
    check(S.PARAM_NAMES[36] == 'O_N', 'O_N is registry index 36')
    check(S.PARAM_UNITS[36] == '1/(uM*s)', 'O_N units 1/(uM*s)')
    check(abs(S.NOMINAL_PARAMS[36] - 0.3) < 1e-12, 'O_N nominal 0.3')
    check(tuple(S.PARAM_BOUNDS[36]) == (0.03, 3.0), 'O_N bounds [0.03, 3.0]')
    check(36 in S.LOG_PARAMS, 'O_N is a log axis (2-decade span)')
    k_ib = S.PARAM_NAMES.index('I_bias')
    check(tuple(S.PARAM_BOUNDS[k_ib]) == (0.3, 1.0),
          'I_bias capped at [0.3, 1.0] (Giulia table)')
    idx = S.resolve_sweep_group('tripartite')
    names = {S.PARAM_NAMES[i] for i in idx}
    check(len(idx) == 32, f'tripartite has 32 free axes (got {len(idx)})')
    check('O_N' in names, 'tripartite sweeps O_N')
    check('I_Theta' not in names and 'omega_I' not in names,
          'tripartite freezes I_Theta and omega_I')
    check(not names & {'DeltaT', 'VT', 'gL'},
          'tripartite keeps the Bucket-A freeze')
    all_idx = S.resolve_sweep_group('all')
    check(len(all_idx) == 34, f"'all' now has 34 free axes (got {len(all_idx)})")
    # theta <-> natural round trip must hold on the widened registry
    th = S.natural_to_theta(S.NOMINAL_PARAMS.copy())
    back = S.theta_to_natural(th)
    check(np.allclose(back, S.NOMINAL_PARAMS),
          'theta_to_natural(natural_to_theta(nominal)) round-trips')


def t2_gate_mechanics(HSR):
    print('\n[T2] contact gate on the REAL build_topology (wallach, flat)')
    csv = _fake_syn_pdist()
    common = dict(Nn=120, Na=60, c_max=300.0, conn_prob=0.25,
                  syn_prob_csv=csv, displ_bias=15.0,
                  gj_dist=200.0, stoa_cutoff=70.0, stoa_sigma=200.0,
                  seed_neuron=39, seed_synapse=35, seed_astro=60,
                  mode='Full', topology_mode='wallach', gj_max_dist=150.0,
                  conn_rule='flat')

    t_default = HSR.build_topology(**common)                       # kwargs absent
    t_off     = HSR.build_topology(**common, stoa_gate_p=1.0,
                                   seed_stoa_gate=12345)
    check(np.array_equal(t_default['StoA_i'], t_off['StoA_i']) and
          np.array_equal(t_default['StoA_j'], t_off['StoA_j']),
          'gate off (p=1.0): StoA identical to a no-kwargs build (backward compat)')
    check(np.array_equal(t_off['StoA_i'], t_off['StoA_viable_i']),
          'gate off: kept == viable')
    n_viable = len(t_off['StoA_viable_i'])
    check(n_viable > 200, f'test topology has enough viable links (n={n_viable})')

    p = 0.5
    t_half  = HSR.build_topology(**common, stoa_gate_p=p, seed_stoa_gate=777)
    t_half2 = HSR.build_topology(**common, stoa_gate_p=p, seed_stoa_gate=777)
    t_half3 = HSR.build_topology(**common, stoa_gate_p=p, seed_stoa_gate=778)
    check(np.array_equal(t_half['StoA_i'], t_half2['StoA_i']),
          'same seed_stoa_gate => identical gated topology (deterministic)')
    check(not np.array_equal(t_half['StoA_i'], t_half3['StoA_i']),
          'different seed_stoa_gate => different mask')
    check(np.array_equal(t_half['StoA_viable_i'], t_off['StoA_viable_i']),
          'viable list is gate-independent (geometry untouched)')
    kept = set(zip(t_half['StoA_i'].tolist(), t_half['StoA_j'].tolist()))
    viab = set(zip(t_half['StoA_viable_i'].tolist(),
                   t_half['StoA_viable_j'].tolist()))
    check(kept <= viab, 'kept links are a subset of viable links')
    frac = len(t_half['StoA_i']) / n_viable
    tol = 4.0 * np.sqrt(p * (1 - p) / n_viable)
    check(abs(frac - p) < tol,
          f'realised keep fraction {frac:.3f} within 4*SE of p={p} (tol {tol:.3f})')

    t_zero = HSR.build_topology(**common, stoa_gate_p=0.0, seed_stoa_gate=1)
    check(len(t_zero['StoA_i']) == 0 and
          len(t_zero['StoA_viable_i']) == n_viable,
          'p=0: all links dropped, viable list intact (decoupled limit)')

    for bad in (-0.1, 1.5):
        try:
            HSR.build_topology(**common, stoa_gate_p=bad)
            check(False, f'stoa_gate_p={bad} raises ValueError')
        except ValueError:
            check(True, f'stoa_gate_p={bad} raises ValueError')

    t_neu = HSR.build_topology(**{**common, 'Na': 0, 'mode': 'Neuronal'},
                               stoa_gate_p=0.5, seed_stoa_gate=1)
    check(len(t_neu['StoA_i']) == 0 and len(t_neu['StoA_viable_i']) == 0,
          'Neuronal mode: empty StoA and StoA_viable arrays (uniform schema)')


def t3_axis_declaration(S):
    print('\n[T3] build_axis_declaration: swept / fixed / inert gate states')
    d = S.build_axis_declaration(_args(gate_lo=0.2, gate_hi=1.0),
                                 S.resolve_sweep_group('tripartite'))
    g = d['swept_axes'].get('stoa_gate_p')
    check(g is not None and g['level'] == 'topology'
          and g['low'] == 0.2 and g['high'] == 1.0 and g['log'] is False,
          'Full + lo<hi: gate declared SWEPT at topology level, U[0.2, 1.0]')
    check('stoa_gate_p' in d['consumed_axes'], 'Full: gate declared CONSUMED')
    check(d['n_topology_level_swept'] == 2,
          'flat + gate: exactly 2 topology-level swept axes (conn_prob, gate)')
    check(d['n_swept'] == 32 + 2,
          f"tripartite/flat/gated: n_swept == 34 (got {d['n_swept']})")

    d = S.build_axis_declaration(_args(gate_lo=0.7, gate_hi=0.7),
                                 S.resolve_sweep_group('tripartite'))
    check('stoa_gate_p' not in d['swept_axes']
          and d['fixed_axes'].get('stoa_gate_p') == 0.7
          and 'stoa_gate_p' in d['consumed_axes'],
          'Full + lo==hi: gate FIXED (0.7) and consumed, not swept')

    d = S.build_axis_declaration(_args(mode='Neuronal', group='neuron_synapse',
                                       gate_lo=1.0, gate_hi=1.0),
                                 S.resolve_sweep_group('neuron_synapse'))
    check('stoa_gate_p' in d['inert_axes']
          and d['fixed_axes'].get('stoa_gate_p') == 1.0
          and 'stoa_gate_p' not in d['consumed_axes'],
          'Neuronal: gate declared INERT (not drawn), pinned at 1.0')

    try:
        S.build_axis_declaration(_args(gate_lo=0.9, gate_hi=0.2),
                                 S.resolve_sweep_group('tripartite'))
        check(False, 'lo > hi raises ValueError')
    except ValueError:
        check(True, 'lo > hi raises ValueError')


def t4_record_keys(S):
    print('\n[T4] stoa_gate_record_keys: present under Full, ABSENT under Neuronal')
    r = S.stoa_gate_record_keys('Full', 0.31)
    check(r == {'stoa_gate_p': 0.31}, "Full -> {'stoa_gate_p': 0.31}")
    r = S.stoa_gate_record_keys('Neuronal', 0.31)
    check(r == {}, 'Neuronal -> {} (absent, not NaN, not null)')


def t5_stream_guard(S):
    print('\n[T5] master_rng consumption: gate draw is GUARDED (off => no draw)')
    # Emulate the outer-loop consumption order of main() for one topology
    # under conn_rule=flat.  Gate OFF must reproduce the PRE-gate stream
    # (uniform conn_prob, then integers topo_seed_base); gate SWEPT inserts
    # exactly one extra uniform before topo_seed_base.
    def stream(gate_swept):
        rng = np.random.default_rng(424242)
        conn_prob = float(rng.uniform(0.1, 0.6))
        if gate_swept:
            _gate = float(rng.uniform(0.2, 1.0))
        topo_seed_base = int(rng.integers(0, 2**31 - 1))
        return conn_prob, topo_seed_base

    pre_gate = stream(gate_swept=False)     # the historical order
    off      = stream(gate_swept=False)
    swept    = stream(gate_swept=True)
    check(pre_gate == off,
          'gate off: (conn_prob, topo_seed_base) identical to the pre-gate order')
    check(pre_gate[0] == swept[0] and pre_gate[1] != swept[1],
          'gate swept: conn_prob unchanged, downstream seeds shift (documented)')


def main():
    print('=' * 72)
    print('smoke_test_stoa_gate -- O_N registry port + astrocyte contact gate')
    print('=' * 72)
    import HPC_main_sweep as S          # noqa: E402  (heavy: pulls the registry)
    import HPC_single_run as HSR        # noqa: E402

    t1_registry(S)
    t2_gate_mechanics(HSR)
    t3_axis_declaration(S)
    t4_record_keys(S)
    t5_stream_guard(S)

    print('\n' + '=' * 72)
    print(f'RESULT: {_n_pass} passed, {_n_fail} failed')
    print('=' * 72)
    return 1 if _n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
