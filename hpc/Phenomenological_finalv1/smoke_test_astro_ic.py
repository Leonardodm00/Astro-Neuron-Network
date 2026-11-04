#!/usr/bin/env python3
"""
smoke_test_astro_ic.py -- checks of SPEC Block 13 (astrocyte initial
conditions): every astrocyte of a Full-mode run starts from the driven state
of its own equations for the run's parameters -- the fixed point itself, or
its own phase on the driven limit cycle -- and starts refractory, so that no
astrocyte releases at the onset of a run only because the exogenous IP3
drive and the glutamate drive switch on at t = 0.

  A1 astro_driven_state returns a fixed point of the LIVE equations: the
     equations of the library's Astrocyte_Group (its user_equations, in
     float64), run uncoupled on Brian2's runtime device with a constant
     Y_extra, stay at the returned state for 12 s (registry nominal, linked
     and unlinked drive; O_N 1e-4; O_N 1): every variable moves by < 1e-6 of
     its value. Control: a state computed with I_bias 1 % off drifts by more
     than 1e-4 uM, so the tolerance can fail.
  A2 a limit cycle (I_bias 0.3 uM, Y_ref): oscillating is True, a period of
     10.6 s (+-0.2 s); started at the trough (phase 0) the live run never goes
     below C(0) by more than 1e-3 uM and reaches the cycle's C_max within
     1e-2 uM in 12 s; started at phase 0.5 it follows the phase-0 run shifted
     by half a period (C within 5e-3 uM over 6 s): the sampled cycle is the
     live model's cycle.
  A3 the onset artefact: from the quiescent state (ics='steady', the start
     before Block 13) at the registry nominal point with Y_ref, C crosses
     C_Theta (0.5 uM) within 5 s; from the driven state no cell crosses it.
  A4 refractory start, with the library's own groups (float32, dt 0.05 ms):
     right after construction not_refractory is False for every astrocyte
     and every release unit; a cell with C(0) = 0.9 uM (above C_osc and
     C_Theta) has no astrocyte event and no release in the first 1 ms, the
     same cell with not_refractory = True (the code before Block 13) has
     both at t = 0.
  A5 astro_ic_jitter(N, seed) equals the draws of the code before Block 13
     (916bf61: one default_rng(seed_astro), three arrays I, C, h, in that
     order), and Astrocyte_Group(ics='steady') still applies them (the
     quiescent start; the driven start has no jitter).
  A6 astro_driven_ics: linked cells get the Y_ref state, unlinked ones the
     Y = 0 state; a fixed point exactly (every cell of its kind the same), a
     cycle at sample floor(phase * n_cycle) of each cell's phase (phase 0:
     the trough); astro_ic_phases is in [0, 1), the same for the same seed,
     and not the jitter's stream; it refuses a phase count mismatch, a phase
     outside [0, 1), an unknown parameter and a negative drive.
  A7 (skipped with --no-sim; a C++ compiler, about 3 min) the compiled
     pipeline: build_and_run on a 150 um well (27 + 27 cells, conn_prob 0.10,
     6 s) with astro_ic='driven' and 'quiescent': the recorded C of every
     astrocyte at t = 0 equals the driven initial state (the run_args arrays
     reach the binary); no release in the first 5 s when driven, releases
     from most astrocytes when quiescent; the results say which start.

Run (from this folder; any machine with Brian2):
    python3 smoke_test_astro_ic.py --no-sim | tail -1     (A1-A6, about 1 min)
    python3 smoke_test_astro_ic.py | tail -1              (also A7)
Expected: "RESULT: N passed, 0 failed" (with --no-sim also ", 1 skipped").
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RESULTS = {'passed': 0, 'failed': 0, 'skipped': 0}
Y_REF = 1700.0          # uM, ASTRO_IC_Y_REF_UM of the library (checked in A6)
C_THETA = 0.5           # uM, registry nominal C_Theta
UNITS = None            # filled by _units(): registry name -> Brian2 unit


def check(cond, label):
    RESULTS['passed' if cond else 'failed'] += 1
    print('  %s %s' % ('ok  ' if cond else 'FAIL', label))
    return bool(cond)


def skip(label):
    RESULTS['skipped'] += 1
    print('  SKIP %s' % label)


def _units():
    from brian2 import umole, second
    return {'O_beta': umole / second, 'O_3K': umole / second, 'Omega_5P': 1 / second,
            'I_bias': umole, 'F': umole / second, 'I_Theta': umole, 'omega_I': umole,
            'O_N': 1 / (umole * second)}


# ---------------------------------------------------------------------------
# the live equations, uncoupled, constant drive (float64, runtime device)
# ---------------------------------------------------------------------------
def live_run(cells, T_s, dt_ms=0.5, rec_dt_ms=5.0):
    """Run one astrocyte per entry of `cells` with the live equations of
    Astrocyte_Group (user_equations), float64, no gap junctions (I_coupling_tot
    stays 0), Y_extra constant; plus a release unit per cell with the live
    threshold 'C > C_Theta'. Each cell: dict(params={registry name: value},
    Y=uM, ics=dict(Gamma_A, I [uM], C [uM], h)). The step is 0.5 ms (the
    pipeline uses 0.05 ms): rk4 keeps a fixed point exactly at any step, and
    the fastest rate of these cells (the Gamma_A binding at O_N 1, about
    1300 /s) is resolved to better than 1e-3 relative at 0.5 ms.
    Returns dict t [s], C, Gamma_A, I, h (cells x samples; uM for I and C),
    rel = list of release-time arrays per cell [s]."""
    import ASD_fun_BD_cpp as M
    import brian2 as b2
    from brian2 import umole, second, ms
    b2.start_scope()
    b2.set_device('runtime')
    b2.prefs.codegen.target = 'numpy'
    b2.defaultclock.dt = dt_ms * ms
    lib_astro, _gj = M.Astrocyte_Group(1, 'Full', seed_astro=11, ics=None, connections=None)
    ns = M.get_Astroparam()
    N = len(cells)
    A = b2.NeuronGroup(N, lib_astro.user_equations, method='rk4', namespace=ns,
                       dtype=np.float64, name='live_astro')
    for k, u in UNITS.items():
        setattr(A, k, np.array([float(c['params'].get(k, ns[k] / u)) for c in cells]) * u)
    A.Y_bias = 0 * umole
    A.Y_extra = np.array([float(c['Y']) for c in cells]) * umole
    A.I_coupling_tot = 0 * umole / second
    A.Gamma_A = np.array([c['ics']['Gamma_A'] for c in cells])
    A.I = np.array([c['ics']['I'] for c in cells]) * umole
    A.C = np.array([c['ics']['C'] for c in cells]) * umole
    A.h = np.array([c['ics']['h'] for c in cells])
    # the release threshold as in Gliotransmission: an upward crossing of C_Theta
    R = b2.NeuronGroup(N, 'C : mole (linked)\nC_Theta : mole', threshold='C > C_Theta',
                       refractory='C > C_Theta', dtype=np.float64, name='live_release')
    R.C = b2.linked_var(A, 'C')
    R.C_Theta = C_THETA * umole
    R.not_refractory = False
    mon = b2.StateMonitor(A, ['C', 'Gamma_A', 'I', 'h'], record=True, dt=rec_dt_ms * ms)
    sp = b2.SpikeMonitor(R)
    b2.Network(A, R, mon, sp).run(T_s * second)
    trains = sp.spike_trains()
    return dict(t=np.asarray(mon.t / second), C=np.asarray(mon.C / umole), I=np.asarray(mon.I / umole),
                Gamma_A=np.asarray(mon.Gamma_A), h=np.asarray(mon.h),
                rel=[np.asarray(trains[k] / second) for k in range(N)])


def quiescent_state():
    """The start before Block 13: the isolated-cell state of ics='steady'."""
    import ASD_fun_BD_cpp as M
    from brian2 import umole
    ss = M._astrocyte_steady_state(M.get_Astroparam())
    return dict(Gamma_A=float(ss['Gamma_A']), I=float(ss['I'] / umole), C=float(ss['C'] / umole),
                h=float(ss['h']))


def a1_to_a3():
    import ASD_fun_BD_cpp as M
    print('\n[A1-A3] the live equations, uncoupled, constant drive (one 12 s run)')
    st = {
        'nominal, linked': ({}, Y_REF),
        'nominal, unlinked': ({}, 0.0),
        'O_N 1e-4, linked': ({'O_N': 1e-4}, Y_REF),
        'O_N 1, linked': ({'O_N': 1.0}, Y_REF),
        'I_bias 0.3, linked': ({'I_bias': 0.3}, Y_REF),
    }
    states = {k: M.astro_driven_state(p, Y) for k, (p, Y) in st.items()}
    for k, s in states.items():
        print('    %-20s oscillating %-5s Gamma_A %.4f I %.4f C %.4f h %.4f (C range %.4f-%.4f)'
              % (k, s['oscillating'], s['Gamma_A'], s['I'], s['C'], s['h'], s['C_min'], s['C_max']))
    planted = M.astro_driven_state({'I_bias': 0.808}, Y_REF)    # 1 % off, run at 0.8
    q = quiescent_state()
    keys = list(st)
    cyc = states['I_bias 0.3, linked']['cycle']
    half = dict(zip(('Gamma_A', 'I', 'C', 'h'), (float(v) for v in cyc[len(cyc) // 2])))
    cells = [dict(params=st[k][0], Y=st[k][1], ics=states[k]) for k in keys]
    cells.append(dict(params={}, Y=Y_REF, ics=planted))               # control of A1
    cells.append(dict(params={}, Y=Y_REF, ics=q))                     # A3, quiescent start
    cells.append(dict(params={}, Y=0.0, ics=q))                       # A3, unlinked, quiescent
    cells.append(dict(params={'I_bias': 0.3}, Y=Y_REF, ics=half))     # A2, phase 0.5
    out = live_run(cells, 12.0)
    t = out['t']

    print('  [A1] fixed points of the live equations')
    for j, k in enumerate(keys):
        if states[k]['oscillating']:
            continue
        dev = max(float(np.max(np.abs(out[v][j] - out[v][j][0])) / max(abs(out[v][j][0]), 1e-12))
                  for v in ('Gamma_A', 'I', 'C', 'h'))
        check(dev < 1e-6, '%s: the driven state holds for 12 s (largest relative change %.1e)' % (k, dev))
    n = len(keys)
    drift = float(np.max(np.abs(out['C'][n] - out['C'][n][0])))
    check(drift > 1e-4, 'control: a state computed with I_bias 0.808 run at 0.8 drifts by %.1e uM (> 1e-4)' % drift)

    print('  [A2] a limit cycle')
    j = keys.index('I_bias 0.3, linked')
    s = states['I_bias 0.3, linked']
    C = out['C'][j]
    check(s['oscillating'] and s['period'] is not None and abs(s['period'] - 10.6) < 0.2
          and s['cycle'].shape == (1000, 4),
          'I_bias 0.3 uM, Y_ref: oscillating, period %s s, cycle of 1000 samples'
          % (('%.2f' % s['period']) if s['period'] else None))
    check(float(C.min()) >= C[0] - 1e-3,
          'the run never goes below C(0) = %.4f by more than 1e-3 uM (min %.4f)' % (C[0], C.min()))
    check(abs(float(C.max()) - s['C_max']) < 1e-2,
          'the run reaches the cycle maximum %.4f within 1e-2 uM (max %.4f)' % (s['C_max'], C.max()))
    check(len(out['rel'][j]) == 0, 'no release from the trough (C_max %.3f < C_Theta %.1f)' % (C.max(), C_THETA))
    jh = n + 3
    lag = int(round(0.5 * s['period'] / (t[1] - t[0])))
    w = t <= 6.0
    dev = float(np.max(np.abs(out['C'][jh][w] - C[lag:lag + int(w.sum())])))
    check(dev < 5e-3, 'phase 0.5 follows phase 0 shifted by half a period: largest difference %.1e uM over 6 s' % dev)
    j = keys.index('nominal, unlinked')
    check(float(out['C'][j].min()) >= out['C'][j][0] - 1e-3,
          'nominal, unlinked (Y = 0; oscillating flag %s): never below C(0) by more than 1e-3 uM'
          % states['nominal, unlinked']['oscillating'])

    print('  [A3] the onset artefact')
    jq, jqu = n + 1, n + 2
    tq = out['rel'][jq]
    check(len(tq) >= 1 and tq[0] < 5.0,
          'quiescent start, nominal, Y_ref: C peaks at %.3f uM and crosses C_Theta at %s s'
          % (out['C'][jq].max(), ('%.2f' % tq[0]) if len(tq) else 'never'))
    tqu = out['rel'][jqu]
    print('    (quiescent start, nominal, Y = 0: C peaks at %.3f uM, %s)'
          % (out['C'][jqu].max(), ('release at %.2f s' % tqu[0]) if len(tqu) else 'no release'))
    driven = [keys.index(k) for k in ('nominal, linked', 'nominal, unlinked', 'O_N 1e-4, linked', 'O_N 1, linked')]
    early = [len(out['rel'][j][out['rel'][j] < 5.0]) for j in driven]
    check(sum(early) == 0, 'driven start: no release in the first 5 s (C max %s uM)'
          % ', '.join('%.3f' % out['C'][j][t < 5.0].max() for j in driven))


def a4_refractory_start():
    import ASD_fun_BD_cpp as M
    import brian2 as b2
    from brian2 import umole, ms
    print('\n[A4] refractory start, the library groups (float32, dt 0.05 ms)')
    b2.start_scope()
    b2.set_device('runtime')
    b2.prefs.codegen.target = 'numpy'
    b2.defaultclock.dt = 0.05 * ms
    Astro, _gj = M.Astrocyte_Group(2, 'Full', seed_astro=11, ics='steady', connections=None)
    GT = M.Gliotransmission(2, Astro, ics='jitter', seed_astro=11)
    check(not np.any(Astro.not_refractory[:]) and not np.any(GT.not_refractory[:]),
          'right after construction: not_refractory False for every astrocyte and release unit')
    Astro.Y_extra = Y_REF * umole
    Astro.C = 0.9 * umole
    Astro.not_refractory[1] = True          # the code before Block 13
    GT.not_refractory[1] = True
    sa, sg = b2.SpikeMonitor(Astro), b2.SpikeMonitor(GT)
    b2.Network(Astro, GT, sa, sg).run(1 * ms)
    ia, ig = np.asarray(sa.i), np.asarray(sg.i)
    check(0 not in ia and 0 not in ig, 'C(0) = 0.9 uM, starting refractory: no astrocyte event, no release in 1 ms')
    check(1 in ia and 1 in ig and float(sa.t[list(ia).index(1)] / ms) == 0.0,
          'the same cell with not_refractory = True: an astrocyte event and a release at t = 0')


def a5_jitter():
    import ASD_fun_BD_cpp as M
    import brian2 as b2
    from brian2 import umole
    print('\n[A5] the jitter of the start')
    N, seed = 37, 12345
    rng = np.random.default_rng(seed)                       # the code of 916bf61, verbatim
    jitter = lambda: 1.0 + 0.05 * (2.0 * rng.random(N) - 1.0)
    old = (jitter(), jitter(), jitter())
    new = M.astro_ic_jitter(N, seed)
    check(all(np.array_equal(a, b) for a, b in zip(old, new)), 'astro_ic_jitter(37, 12345) == the draws of 916bf61')
    check(all(np.all(np.abs(a - 1.0) <= 0.05) for a in new), 'every factor within 1 +- 0.05')
    b2.start_scope()
    b2.set_device('runtime')
    Astro, _gj = M.Astrocyte_Group(N, 'Full', seed_astro=seed, ics='steady', connections=None)
    q = quiescent_state()
    ok = (np.allclose(np.asarray(Astro.I[:] / umole), np.float32(q['I'] * new[0]), rtol=1e-6)
          and np.allclose(np.asarray(Astro.C[:] / umole), np.float32(q['C'] * new[1]), rtol=1e-6)
          and np.allclose(np.asarray(Astro.h[:]), np.float32(q['h'] * new[2]), rtol=1e-6))
    check(ok, "Astrocyte_Group(ics='steady') applies them (I, C, h)")


def a6_driven_ics():
    import ASD_fun_BD_cpp as M
    print('\n[A6] astro_driven_ics')
    check(M.ASTRO_IC_Y_REF_UM == Y_REF and M.ASTRO_IC_SCHEME == 'driven-v1-2026-11-04',
          'Y_ref 1700 uM; scheme driven-v1-2026-11-04')
    ph = M.astro_ic_phases(5, 7)
    check(np.all((ph >= 0) & (ph < 1)) and np.array_equal(ph, M.astro_ic_phases(5, 7))
          and not np.array_equal(ph, M.astro_ic_phases(5, 8))
          and not any(np.allclose(ph, (j - 1.0) / 0.1 + 0.5) for j in M.astro_ic_jitter(5, 7)),
          'astro_ic_phases: in [0, 1), the same for the same seed, another for another seed, not the jitter stream')
    linked = np.array([True, False, True, True, False])
    p = {'I_bias': 1.0, 'F': 10.0}                        # fixed points for both kinds
    sl, su = M.astro_driven_state(p, Y_REF), M.astro_driven_state(p, 0.0)
    ics, states = M.astro_driven_ics(p, linked, ph)
    ok = not sl['oscillating'] and not su['oscillating']
    for k in ('Gamma_A', 'I', 'C', 'h'):
        ok = ok and np.array_equal(ics[k], np.where(linked, sl[k], su[k]))
    check(ok and set(states) == {True, False},
          'fixed points: linked -> the Y_ref state, unlinked -> the Y = 0 state, exactly (no jitter)')
    pc = {'I_bias': 0.3}                                  # a cycle for the linked kind
    phc = np.array([0.0, 0.5, 0.25, 0.999, 0.5])
    icc, stc = M.astro_driven_ics(pc, np.ones(5, bool), phc)
    cyc = stc[True]['cycle']
    exp = cyc[np.minimum((phc * len(cyc)).astype(int), len(cyc) - 1), 2]
    check(set(stc) == {True} and np.array_equal(icc['C'], exp) and icc['C'][0] == stc[True]['C']
          and icc['C'][1] == icc['C'][4],
          'a cycle: each cell at sample floor(phase * 1000); phase 0 the trough; all linked -> one state computed')

    def raises(exc, f, *a):
        try:
            f(*a)
        except exc:
            return True
        except Exception:
            return False
        return False
    check(raises(ValueError, M.astro_driven_ics, p, linked[:4], ph), 'phase count mismatch: ValueError')
    check(raises(ValueError, M.astro_driven_ics, p, linked, np.array([0.1, 0.2, 1.0, 0.3, 0.4])),
          'a phase outside [0, 1): ValueError')
    check(raises(KeyError, M.astro_driven_state, {'C_Theta': 0.3}, Y_REF), 'a parameter it does not use: KeyError')
    check(raises(ValueError, M.astro_driven_state, {}, -1.0), 'a negative drive: ValueError')
    check(set(M.ASTRO_IC_PARAMS) == set(UNITS), 'ASTRO_IC_PARAMS = the astrocyte axes of the ODE (%s)'
          % ', '.join(M.ASTRO_IC_PARAMS))


# ---------------------------------------------------------------------------
# A7: the compiled pipeline (one process per build: Brian2's device is global)
# ---------------------------------------------------------------------------
def a7_worker(mode, out_dir):
    """Child process: one build_and_run with astro_ic=mode; writes a7_<mode>.npz."""
    import HPC_single_run as R
    import ASD_fun_BD_cpp as M
    argv = ['--out_dir', out_dir, '--lib_dir', HERE, '--mode', 'Full', '--simtime', '6',
            '--density', '1200', '--density_astro', '1200', '--c_max', '150', '--conn_prob', '0.10',
            '--conn_rule', 'flat', '--topology_mode', 'wallach', '--stoa_gate_p', '1.0',
            '--seed_run', '7', '--seed_neuron', '101', '--seed_synapse', '102', '--seed_astro', '103']
    args = R.build_parser().parse_args(argv)
    args.Nn, args.Na = R.resolve_population_from_density(
        args.Nn, args.Na, args.c_max, density=args.density, density_astro=args.density_astro)
    topo = R.build_topology(
        Nn=args.Nn, Na=args.Na, c_max=args.c_max, conn_prob=args.conn_prob,
        syn_prob_csv=R._resolve_syn_pdist_csv(args), displ_bias=args.displ_bias, gj_dist=args.gj_dist,
        stoa_cutoff=args.stoa_cutoff, stoa_sigma=args.stoa_sigma, seed_neuron=args.seed_neuron,
        seed_synapse=args.seed_synapse, seed_astro=args.seed_astro, mode='Full',
        topology_mode=args.topology_mode, gj_max_dist=args.gj_max_dist, conn_rule=args.conn_rule,
        p0_conn=args.p0_conn, d0_conn=args.d0_conn, beta_conn=args.beta_conn,
        conn_periodic=args.conn_periodic, stoa_gate_p=args.stoa_gate_p, seed_stoa_gate=args.seed_stoa_gate)
    params = R.resolve_params(args).astype(np.float64)
    rec = dict(neurons=np.array([0], dtype=np.int32), dt_ms=5.0, all_dt_ms=5.0)
    res = R.build_and_run(topo, args, params, record=rec, astro_ic=mode)
    r = res['record']
    linked = np.zeros(args.Na, bool)
    linked[np.unique(np.asarray(topo['StoA_j'], dtype=np.int64))] = True
    seed_ic = R.derive_network_seeds(args.seed_device, 0)['seed_ic_astro']
    a_par = {k: float(params[R.PARAM_NAMES.index(k)]) for k in M.ASTRO_IC_PARAMS}
    exp, _ = M.astro_driven_ics(a_par, linked, M.astro_ic_phases(args.Na, seed_ic))
    np.savez(os.path.join(out_dir, 'a7_%s.npz' % mode), t_astro=r['t_astro'], C=r['C'], rel_t=r['rel_t'],
             rel_i=r['rel_i'], exp_C=exp['C'], linked=linked, astro_ic=np.array(res['astro_ic']),
             Na=np.int64(args.Na), n_spk=np.int64(res['spk_N_t'].size))


def a7():
    print('\n[A7] the compiled pipeline (build_and_run, 27 + 27 cells, 6 s)')
    tmp = tempfile.mkdtemp(prefix='astro_ic_a7_')
    got = {}
    for mode in ('driven', 'quiescent'):
        d = os.path.join(tmp, mode)
        os.makedirs(d)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), '--a7-worker', mode, d],
                           cwd=HERE, capture_output=True, text=True)
        f = os.path.join(d, 'a7_%s.npz' % mode)
        if r.returncode != 0 or not os.path.exists(f):
            print(r.stdout[-2000:], r.stderr[-3000:])
            check(False, '%s: build_and_run ran' % mode)
            return
        got[mode] = dict(np.load(f))
    dv, qu = got['driven'], got['quiescent']
    check(str(dv['astro_ic']) == 'driven-v1-2026-11-04' and str(qu['astro_ic']) == 'quiescent',
          "results: astro_ic 'driven-v1-2026-11-04' and 'quiescent'")
    C0 = dv['C'][:, 0]
    err = float(np.max(np.abs(C0 - dv['exp_C']) / dv['exp_C']))
    check(dv['t_astro'][0] == 0.0 and err < 1e-6,
          'driven: C of every astrocyte at t = 0 = the driven state (largest relative error %.1e; float32)' % err)
    check(float(np.max(np.abs(qu['C'][:, 0] - 0.0524))) < 0.004,
          'quiescent: C at t = 0 = the isolated state 0.052 uM (+-5 %)')
    Na = int(dv['Na'])
    n_dv = len(set(dv['rel_i'][dv['rel_t'] < 5.0].tolist()))
    n_qu = len(set(qu['rel_i'][qu['rel_t'] < 5.0].tolist()))
    print('    astrocytes releasing in the first 5 s: driven %d of %d, quiescent %d of %d; neuron spikes %d, %d'
          % (n_dv, Na, n_qu, Na, int(dv['n_spk']), int(qu['n_spk'])))
    check(n_dv == 0, 'driven: no release in the first 5 s')
    check(n_qu >= Na // 2, 'quiescent: at least half of the astrocytes release in the first 5 s (the artefact)')


def main():
    global UNITS
    if len(sys.argv) >= 4 and sys.argv[1] == '--a7-worker':
        return a7_worker(sys.argv[2], sys.argv[3])
    import importlib.util
    if importlib.util.find_spec('brian2') is None:
        print('brian2 not importable: nothing checked')
        return 1
    UNITS = _units()
    from brian2 import BrianLogger
    for name in ('resolution_conflict', 'unused_brian_object'):   # namespace shadowing; unused GJ groups
        BrianLogger.suppress_name(name)
    a1_to_a3()
    a4_refractory_start()
    a5_jitter()
    a6_driven_ics()
    if '--no-sim' in sys.argv:
        skip('A7 (--no-sim)')
    else:
        a7()
    print('\nRESULT: %d passed, %d failed%s' % (RESULTS['passed'], RESULTS['failed'],
          (', %d skipped' % RESULTS['skipped']) if RESULTS['skipped'] else ''))
    return 0 if RESULTS['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
