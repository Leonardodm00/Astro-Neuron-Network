#!/usr/bin/env python3
"""
smoke_test_sic.py -- checks of Block 12 (specs/SPEC.md): the slow inward
current (SIC) from astrocytes to neurons through extrasynaptic NMDA receptors,
driven by the astrocyte's gliotransmitter G_A at every step:
    dx/dt = k_on H(G_A(t)) (1 - x) - x/tau_d,  dr/dt = (x - r)/tau_r,
    I_sic,i = g_sic (sum of r over the neuron's pairs) B(V) (V - E_nmda),
on the pairs (astrocyte, neuron) that are Voronoi neighbours in the joint
tessellation (capped at gj_max_dist), always on in Full mode.

  S1  constants: get_SICparam() holds the SPEC values; SIC_SCHEME
  S2  sic_pairs against independent oracles: a 5-point layout solved by hand
      (one astrocyte-neuron pair separated by a third cell is NOT a pair; the
      cap removes the far pairs), and a random layout against
      scipy.spatial.Voronoi ridge_points (the cells that share a ridge); order,
      dtype, cap 0 and inf, two cells, no astrocyte, degenerate sets raise
  S3  the switch in Neuronal_Network (runtime device): Full has r_sic_tot,
      I_sic and the per-neuron g_sic (40.6 nS, registry axis 37) by default,
      sic=False and Neuronal do not, sic=True in Neuronal
      raises; with the SIC off the neuron equations equal those of the
      library before Block 12 (git show PRE_SIC_COMMIT; skipped without git)
  S4  the SIC kinetics in Brian2 (runtime device, numpy, dt 0.05 ms) against
      scipy solve_ivp (rtol 1e-10) for one release G_A(t) = 78 uM
      exp(-5 t/s): max |r - r_ref| / max r_ref < 2e-3, and the error halves
      when dt halves (first-order exponential Euler; ratio in [1.6, 2.4])
  S5  voltage clamp at -60 mV on Brian2's own I_sic: peak -43.31 pA
      (0.5 %), 0-100 % rise 220.1 ms (1 ms), decay tau 706.0 ms (0.5 %),
      the calibration of Block 12 (Kovacs & Pal 2017: 43.3 pA, 220.1, 706 ms)
  S6  summation and independence: a neuron with two partner astrocytes gets
      r_sic_tot = r(a0) + r(a1) (rel. 1e-12; at lag 0 or one step: Brian2
      updates the summed variable before the synapses' state, so the neuron
      sees the sum of the previous step); a neuron with no pair has exactly
      0; a release 1 s later gives the same r(t) shifted
  S7  build_topology: Full mode returns SIC_i, SIC_j = sic_pairs(N_pos, A_pos,
      inf) (no cap, the author's decision of 2026-11-04), the same in both
      topology modes, for any contact gate and any gj_max_dist,
      deterministic; Neuronal mode returns them empty
  S8  sic_record_keys: Full -> sic_scheme, n_sic_pairs; Neuronal -> absent
  S9  Astro_to_Neuron_SIC refuses a neuron group without r_sic_tot, index
      arrays of different length, and an index out of range
  S10 cpp_standalone (about 30 s): the stand-in of S4-S6 compiled gives the
      numpy values (rel. 1e-8); HPC_single_run.main() (its figures switched
      off: the 3-D figure fails at random, SPEC section 8), Full mode, 2 s: exit 0,
      sic_scheme and n_sic_pairs (= len(topology.npz SIC_i) > 0) recorded;
      Neuronal mode: the keys absent, SIC_i empty
  S11 SIC off = before Block 12 (cpp_standalone, about 2 min; needs git):
      HPC_single_run.py of PRE_SIC_COMMIT and the current one with the SIC
      switched off (build_and_run sic=False) and the astrocytes' initial
      state of before (astro_ic='quiescent', Block 13) give the same spikes,
      Full mode 2 s; so does the current one with the SIC terms in the
      neuron equations but no SIC drive (r_sic_tot = 0); and Neuronal mode
      (where the SIC never exists)

Run (from this folder):
    python3 smoke_test_sic.py | tail -1            (about 2-3 min, two cores)
    python3 smoke_test_sic.py --no-sim | tail -1   (S1-S9; about 40 s)
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRE_SIC_COMMIT = '6c50aed'          # last commit before Block 12
RESULTS = {'passed': 0, 'failed': 0, 'skipped': 0}

# Block 12 values (specs/SPEC.md)
SPEC = dict(k_on_sic=3.596, tau_d_sic=651.9, tau_r_sic=2.0, EC50_sic=3.0,
            n_sic=1.9, g_sic=40.6)                    # /ms, ms, ms, mM, 1, nS
G0_UM, OMEGA_E = 78.0, 5e-3                           # uM, 1/ms
V_CLAMP, E_NMDA = -60.0, 0.0                          # mV
TARGET = dict(I=-43.31, rise=220.1, tau=706.0)        # pA, ms, ms


def check(cond, label):
    RESULTS['passed' if cond else 'failed'] += 1
    print('  %s %s' % ('ok  ' if cond else 'FAIL', label))
    return cond


def skip(label):
    RESULTS['skipped'] += 1
    print('  SKIP %s' % label)


def lib():
    sys.path.insert(0, HERE)
    import ASD_fun_BD_cpp as M
    return M


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------
def hill(G, EC50, n):
    q = (np.asarray(G, dtype=np.float64) / EC50) ** n
    return q / (1.0 + q)


def r_reference(t_ms, G0=G0_UM, t_rel=0.0):
    """scipy solution of the SPEC equations for one release at t_rel (ms)."""
    from scipy.integrate import solve_ivp
    k, td, tr = SPEC['k_on_sic'], SPEC['tau_d_sic'], SPEC['tau_r_sic']
    ec, n = SPEC['EC50_sic'] * 1e3, SPEC['n_sic']          # uM

    def rhs(t, y):
        G = G0 * np.exp(-OMEGA_E * t)
        return [k * hill(G, ec, n) * (1.0 - y[0]) - y[0] / td, (y[0] - y[1]) / tr]
    tt = np.asarray(t_ms, dtype=np.float64) - t_rel
    out = np.zeros_like(tt)
    m = tt >= 0
    if m.any():
        sol = solve_ivp(rhs, (0.0, tt[m].max()), [0.0, 0.0], t_eval=tt[m],
                        method='LSODA', rtol=1e-10, atol=1e-14, max_step=1.0)
        assert sol.success, sol.message
        out[m] = sol.y[1]
    return out


def measure(t, s):
    """0-100 % rise (peak time, parabolic refinement), single-exponential
    decay tau fitted from the peak down to 10 % of it, peak value -- the
    calibration's definitions (Block 12). s is positive (an open fraction or
    the magnitude of a current)."""
    from scipy.optimize import curve_fit
    k = int(np.argmax(s))
    if 0 < k < len(s) - 1:
        y0, y1, y2 = s[k - 1], s[k], s[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        t_pk = t[k] + off * (t[1] - t[0])
        s_pk = y1 - 0.25 * (y0 - y2) * off
    else:
        t_pk, s_pk = t[k], s[k]
    tail = np.arange(k, len(s))
    stop = tail[s[tail] < 0.1 * s_pk]
    end = stop[0] if len(stop) else len(s)
    tt, yy = t[k:end] - t[k], s[k:end]
    (a, tau), _ = curve_fit(lambda x, a, b: a * np.exp(-x / b), tt, yy, p0=(s_pk, 700.0))
    return t_pk, tau, s_pk


# ---------------------------------------------------------------------------
# S1, S2: constants and pairs (numpy only)
# ---------------------------------------------------------------------------
def s1(M):
    print('\n[S1] constants')
    from brian2 import ms, mmole, nS
    p = M.get_SICparam()
    got = dict(k_on_sic=float(p['k_on_sic'] * ms), tau_d_sic=float(p['tau_d_sic'] / ms),
               tau_r_sic=float(p['tau_r_sic'] / ms), EC50_sic=float(p['EC50_sic'] / mmole),
               n_sic=float(p['n_sic']), g_sic=float(p['g_sic'] / nS))
    check(all(abs(got[k] - SPEC[k]) <= 1e-12 * abs(SPEC[k]) for k in SPEC),
          'get_SICparam = SPEC values %s' % got)
    check(M.SIC_SCHEME == 'sic-v1-2026-11-04', 'SIC_SCHEME = %r' % M.SIC_SCHEME)


def voronoi_oracle(N_pos, A_pos, cap):
    """Pairs (a, i) whose cells share a ridge (scipy Voronoi), within cap."""
    from scipy.spatial import Voronoi
    Nn = len(N_pos)
    vor = Voronoi(np.vstack([N_pos, A_pos]))
    out = set()
    for u, v in vor.ridge_points:
        u, v = int(min(u, v)), int(max(u, v))
        if u < Nn <= v and np.hypot(*(A_pos[v - Nn] - N_pos[u])) <= cap:
            out.add((v - Nn, u))
    return out


def s2(M):
    print('\n[S2] sic_pairs')
    # hand-solved layout: N0 (0,0), N1 (10,10), N2 (10,-10); A0 (10,1) inside
    # the hull N0-N2-A1-N1, A1 (20,0). The only triangulation joins A0 to the
    # four hull points; N0-A1 is not an edge (every circle through them holds
    # A0 or N2). Distances: A0-N0 10.05, A0-N1 9, A0-N2 11, A1-N1 = A1-N2 14.14.
    N_pos = np.array([[0.0, 0.0], [10.0, 10.0], [10.0, -10.0]])
    A_pos = np.array([[10.0, 1.0], [20.0, 0.0]])
    a, i = M.sic_pairs(N_pos, A_pos, max_dist=np.inf)
    check(list(zip(a.tolist(), i.tolist())) == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)],
          'hand layout, no cap: %s (want A0 with N0, N1, N2; A1 with N1, N2; not A1-N0)'
          % list(zip(a.tolist(), i.tolist())))
    a, i = M.sic_pairs(N_pos, A_pos, max_dist=12.0)
    check(list(zip(a.tolist(), i.tolist())) == [(0, 0), (0, 1), (0, 2)],
          'hand layout, cap 12 um: %s (the 14.14 um pairs removed)' % list(zip(a.tolist(), i.tolist())))
    check(a.dtype == np.int32 and i.dtype == np.int32, 'int32 output')

    rng = np.random.default_rng(3)
    N_pos = rng.uniform(0, 300, (108, 2))
    A_pos = rng.uniform(0, 300, (108, 2))
    for cap in (150.0, 40.0, np.inf):
        a, i = M.sic_pairs(N_pos, A_pos, max_dist=cap)
        got = set(zip(a.tolist(), i.tolist()))
        want = voronoi_oracle(N_pos, A_pos, cap)
        check(got == want and len(got) == len(a),
              'random 108 + 108 layout, cap %s um: %d pairs = Voronoi ridge oracle (%d)'
              % (cap, len(got), len(want)))
    a_def, i_def = M.sic_pairs(N_pos, A_pos)
    a_inf, i_inf = M.sic_pairs(N_pos, A_pos, max_dist=np.inf)
    check(np.array_equal(a_def, a_inf) and np.array_equal(i_def, i_inf),
          'the default is no cap (max_dist = inf): %d pairs' % len(a_def))
    a, i = M.sic_pairs(N_pos, A_pos, max_dist=150.0)
    d = np.hypot(*(A_pos[a] - N_pos[i]).T)
    order_ok = bool(np.all(np.diff(a.astype(np.int64) * 1000 + i) > 0))
    check(order_ok and d.max() <= 150.0, 'sorted by (astrocyte, neuron), unique; max distance %.1f um' % d.max())
    a0, _ = M.sic_pairs(N_pos, A_pos, max_dist=0.0)
    e1, e2 = M.sic_pairs(N_pos, np.empty((0, 2)), max_dist=150.0)
    t1, t2 = M.sic_pairs([[0.0, 0.0]], [[3.0, 4.0]], max_dist=5.0)
    t3, _ = M.sic_pairs([[0.0, 0.0]], [[3.0, 4.0]], max_dist=4.9)
    check(len(a0) == 0 and len(e1) == 0 and len(e2) == 0 and t1.tolist() == [0]
          and t2.tolist() == [0] and len(t3) == 0,
          'cap 0 -> none; no astrocyte -> none; two cells 5 um apart -> a pair at cap 5, none at 4.9')
    bad = 0
    for N_b, A_b in (([[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [3.0, 3.0]]),   # collinear
                     ([[0.0, 0.0], [5.0, 1.0]], [[0.0, 0.0], [2.0, 7.0]])):  # coincident
        try:
            M.sic_pairs(N_b, A_b)
        except ValueError:
            bad += 1
    check(bad == 2, 'collinear and coincident somata raise ValueError (%d of 2)' % bad)


# ---------------------------------------------------------------------------
# S3: the switch (runtime device)
# ---------------------------------------------------------------------------
def _binomial_stub():
    from brian2 import Function
    return Function(lambda n, p: 0.0 * n, arg_units=[1, 1], return_unit=1)


def _neuron_eqs(M, mode, sic):
    from brian2 import start_scope
    start_scope()
    kw = {} if sic == 'default' else {'sic': sic}
    N, S = M.Neuronal_Network(4, Simulated_network=mode, synapse_type='facilitating',
                              Binomial_fun=_binomial_stub(), **kw)
    return N


def s3(M, tmp):
    print('\n[S3] the switch in Neuronal_Network')
    from brian2 import prefs
    prefs.codegen.target = 'numpy'
    Nf = _neuron_eqs(M, 'Full', 'default')
    from brian2 import nS
    g0 = np.asarray(Nf.g_sic[:] / nS) if 'g_sic' in Nf.variables else np.array([])
    check('r_sic_tot' in Nf.variables and 'I_sic' in Nf.variables and g0.size == 4
          and np.all(np.abs(g0 - SPEC['g_sic']) < 1e-9) and 'g_sic' not in Nf.namespace,
          'Full mode, default: r_sic_tot, I_sic, and g_sic a per-neuron parameter = %s nS '
          '(registry axis 37; not a namespace constant)' % SPEC['g_sic'])
    Noff = _neuron_eqs(M, 'Full', False)
    Nn = _neuron_eqs(M, 'Neuronal', 'default')
    check('r_sic_tot' not in Noff.variables and 'r_sic_tot' not in Nn.variables
          and 'g_sic' not in Noff.variables and 'g_sic' not in Nn.variables,
          'Full sic=False and Neuronal: no SIC terms')
    try:
        _neuron_eqs(M, 'Neuronal', True)
        raised = False
    except ValueError:
        raised = True
    check(raised, 'sic=True in Neuronal mode raises ValueError')
    # the equations with the SIC off against the library before Block 12
    old_path = os.path.join(tmp, 'old_lib', 'ASD_fun_BD_cpp.py')
    os.makedirs(os.path.dirname(old_path), exist_ok=True)
    r = subprocess.run(['git', '-C', HERE, 'show',
                        '%s:hpc/Phenomenological_finalv1/ASD_fun_BD_cpp.py' % PRE_SIC_COMMIT],
                       capture_output=True)
    if r.returncode != 0:
        skip('equations against %s: git show failed (%s)' % (PRE_SIC_COMMIT,
             r.stderr.decode(errors='replace').strip()[:120]))
        return
    open(old_path, 'wb').write(r.stdout)
    import importlib.util
    spec = importlib.util.spec_from_file_location('ASD_pre_sic', old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for mode, label in (('Full', 'Full, sic=False'), ('Neuronal', 'Neuronal')):
        new_N = _neuron_eqs(M, mode, False if mode == 'Full' else 'default')
        old_N = _neuron_eqs(old, mode, 'default')
        same = (str(new_N.equations) == str(old_N.equations)
                and set(new_N.namespace) == set(old_N.namespace))
        check(same, '%s: neuron equations and namespace equal those of %s'
              % (label, PRE_SIC_COMMIT))


# ---------------------------------------------------------------------------
# S4-S6, S10a: the stand-in network (subprocess; numpy or cpp_standalone)
# ---------------------------------------------------------------------------
STAND_IN = r'''
import json, sys
import numpy as np
sys.path.insert(0, %(here)r)
from brian2 import (NeuronGroup, Network, StateMonitor, defaultclock, prefs, set_device,
                    get_device, ms, mV, pA, umole, second, mole)
standalone = %(standalone)r
if standalone:
    set_device('cpp_standalone', directory=%(build)r, build_on_run=False)
else:
    prefs.codegen.target = 'numpy'
import ASD_fun_BD_cpp as M
defaultclock.dt = %(dt_ms)r * ms
# astrocyte stand-ins: a0 releases G0 at t = 0 (initial value), a1 at t_rel
astro = NeuronGroup(2, """dG_A/dt = -Omega_e*G_A : mole
                          t_rel : second (constant)
                          released : 1""",
                    threshold='t >= t_rel and released < 0.5',
                    reset='G_A += G0; released = 1.0', method='exact',
                    namespace=dict(Omega_e=5.0/second, G0=78.0*umole), name='Gliot_stand_in')
astro.G_A = [78.0, 0.0] * umole
astro.released = [1.0, 0.0]
astro.t_rel = [0.0, 1000.0] * ms
# neuron stand-ins clamped at -60 mV: n0 <- a0, a1; n1 <- a1; n2 none; n3 <- a0
p = M.get_SICparam()
neu = NeuronGroup(4, """r_sic_tot : 1
                        V : volt (constant)
                        I_sic = g_sic*(V-E_nmda)*r_sic_tot/(1+exp(-0.062*V/mV)/3.57) : amp""",
                  namespace=dict(g_sic=p['g_sic'], E_nmda=0.0*mV), name='Neuron_stand_in')
neu.V = -60.0 * mV
sic = M.Astro_to_Neuron_SIC(astro, neu, connections=[[0, 0, 1, 1], [0, 3, 0, 1]])
rec_dt = None if %(rec_dt_ms)r is None else %(rec_dt_ms)r * ms     # None: every step
mon_n = StateMonitor(neu, ['r_sic_tot', 'I_sic'], record=True, dt=rec_dt, name='mon_n')
mon_s = StateMonitor(sic, 'r_sic', record=True, dt=rec_dt, name='mon_s')
net = Network(astro, neu, sic, mon_n, mon_s)
net.run(%(t_end_ms)r * ms)
if standalone:
    get_device().build(directory=%(build)r, compile=True, run=True)
print(json.dumps(dict(t=np.asarray(mon_n.t / ms).tolist(),
                      r_tot=np.asarray(mon_n.r_sic_tot).tolist(),
                      I=np.asarray(mon_n.I_sic / pA).tolist(),
                      r_pair=np.asarray(mon_s.r_sic).tolist(),
                      pair_i=np.asarray(sic.i[:]).tolist(), pair_j=np.asarray(sic.j[:]).tolist())))
'''


def stand_in(tmp, standalone, dt_ms=0.05, t_end_ms=6000.0, rec_dt_ms=0.5):
    code = STAND_IN % dict(here=HERE, standalone=standalone, dt_ms=dt_ms, t_end_ms=t_end_ms,
                           rec_dt_ms=rec_dt_ms, build=os.path.join(tmp, 'build_stand_in'))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=tmp)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-3000:])
        return None
    d = json.loads(r.stdout.strip().splitlines()[-1])
    return {k: np.asarray(v) for k, v in d.items()}


def _pair_row(d, a, n):
    for k, (i, j) in enumerate(zip(d['pair_i'], d['pair_j'])):
        if i == a and j == n:
            return d['r_pair'][k]
    raise KeyError((a, n))


def s4_s6(tmp):
    print('\n[S4] SIC kinetics, Brian2 (numpy, dt 0.05 ms) against scipy')
    d = stand_in(tmp, False, 0.05)
    d2 = stand_in(tmp, False, 0.1)
    if not check(d is not None and d2 is not None, 'stand-in network built and run (dt 0.05 and 0.1 ms)'):
        return None
    t = d['t']
    ref = r_reference(t)
    r03 = _pair_row(d, 0, 3)
    err = np.max(np.abs(r03 - ref)) / ref.max()
    err2 = np.max(np.abs(_pair_row(d2, 0, 3) - r_reference(d2['t']))) / ref.max()
    check(err < 2e-3, 'max |r - r_ref| / max r_ref = %.2e at dt 0.05 ms (< 2e-3: first-order '
          'exponential Euler, about dt/tau_r of the rise)' % err)
    check(1.6 <= err2 / err <= 2.4, 'error at dt 0.1 / at dt 0.05 = %.2f (first order: 2)' % (err2 / err))

    print('\n[S5] voltage clamp at -60 mV on Brian2\'s I_sic')
    I3 = -d['I'][3]                         # pA, magnitude of the inward current of n3
    t_pk, tau, I_pk = measure(t, I3)
    check(abs(-I_pk - TARGET['I']) <= 0.005 * abs(TARGET['I']),
          'peak %.2f pA (want %.2f, 0.5 %%)' % (-I_pk, TARGET['I']))
    check(abs(t_pk - TARGET['rise']) <= 1.0, '0-100 %% rise %.1f ms (want %.1f, 1 ms)' % (t_pk, TARGET['rise']))
    check(abs(tau - TARGET['tau']) <= 0.005 * TARGET['tau'],
          'decay tau %.1f ms (want %.1f, 0.5 %%)' % (tau, TARGET['tau']))

    print('\n[S6] summation and independence of pairs')
    # every step recorded over 1.2 s (both releases): Brian2 may update the
    # summed r_sic_tot before or after the synapses' state in a step, so the
    # sum is compared at lag 0 and at lag one step; one of the two must hold
    # to float64 roundoff
    df = stand_in(tmp, False, 0.05, t_end_ms=1200.0, rec_dt_ms=None)
    if not check(df is not None, 'stand-in recorded at every step (1.2 s)'):
        return d
    tot0, tot1 = df['r_tot'][0], df['r_tot'][1]
    summ = _pair_row(df, 0, 0) + _pair_row(df, 1, 0)
    r11 = _pair_row(df, 1, 1)
    scale = max(np.max(np.abs(summ)), 1e-300)
    rel0 = np.max(np.abs(tot0 - summ)) / scale
    rel1 = np.max(np.abs(tot0[1:] - summ[:-1])) / scale
    lag = 0 if rel0 <= rel1 else 1
    check(min(rel0, rel1) < 1e-12 and tot0[0] == 0.0,
          'neuron 0: r_sic_tot = r(a0) + r(a1) at lag %d step (max rel. diff %.1e; at the other '
          'lag %.1e)' % (lag, min(rel0, rel1), max(rel0, rel1)))
    same1 = np.array_equal(tot1, r11) if lag == 0 else np.array_equal(tot1[1:], r11[:-1])
    check(same1 and np.all(df['r_tot'][2] == 0.0),
          'neuron 1: r_sic_tot = r(a1) exactly (same lag); neuron 2 (no pair): exactly 0')
    # a1 releases at the reset of the step starting at t = 1000 ms; its SIC sees it one step later
    shift = 1000.0 + 0.05
    late = _pair_row(d, 1, 1)
    want = np.interp(t - shift, t, r03, left=0.0)
    e_shift = np.max(np.abs(late - want)) / r03.max()
    check(e_shift < 2e-3, 'a release 1 s later: the same r(t) shifted (max diff %.1e of the peak)' % e_shift)
    return d


# ---------------------------------------------------------------------------
# S7-S9 (numpy)
# ---------------------------------------------------------------------------
def _topo(R, **kw):
    import pandas as pd
    csv = pd.read_csv(os.path.join(HERE, 'synapse_pdist.csv'))
    base = dict(Nn=40, Na=40, c_max=200.0, conn_prob=0.1, syn_prob_csv=csv, displ_bias=20.0,
                gj_dist=60.0, stoa_cutoff=70.0, stoa_sigma=30.0, seed_neuron=11,
                seed_synapse=12, seed_astro=13, mode='Full', topology_mode='wallach',
                gj_max_dist=150.0)
    base.update(kw)
    return R.build_topology(**base)


def s7(M):
    print('\n[S7] build_topology')
    import io
    import contextlib
    sys.path.insert(0, HERE)
    import HPC_single_run as R
    with contextlib.redirect_stdout(io.StringIO()):
        t1 = _topo(R)
        t2 = _topo(R)
        t3 = _topo(R, stoa_gate_p=0.3, seed_stoa_gate=5)
        t4 = _topo(R, topology_mode='distance')
        t5 = _topo(R, gj_max_dist=30.0)
        tn = _topo(R, mode='Neuronal', Na=0)
    a, i = M.sic_pairs(t1['N_pos'], t1['A_pos'], max_dist=np.inf)
    a150, _ = M.sic_pairs(t1['N_pos'], t1['A_pos'], max_dist=150.0)
    check(np.array_equal(t1['SIC_i'], a) and np.array_equal(t1['SIC_j'], i) and len(a) > 0,
          'Full, wallach: SIC_i, SIC_j = sic_pairs(N_pos, A_pos, inf), no cap (%d pairs; %d with a '
          '150 um cap)' % (len(a), len(a150)))
    same = all(np.array_equal(t1[k], t[k]) for t in (t2, t3, t4, t5) for k in ('SIC_i', 'SIC_j'))
    check(same, 'the same pairs on a second build, with the contact gate 0.3, in distance mode, '
          'and with gj_max_dist 30 um (the cap no longer follows it)')
    check(len(tn['SIC_i']) == 0 and len(tn['SIC_j']) == 0 and tn['SIC_i'].dtype == np.int32,
          'Neuronal: SIC_i, SIC_j empty (int32)')
    return R


def s8(R, M):
    print('\n[S8] sic_record_keys')
    check(R.sic_record_keys('Full', 17, M.SIC_SCHEME) == {'sic_scheme': M.SIC_SCHEME, 'n_sic_pairs': 17}
          and R.sic_record_keys('Neuronal', 0, M.SIC_SCHEME) == {},
          'Full -> sic_scheme, n_sic_pairs; Neuronal -> absent')


def s9(M):
    print('\n[S9] Astro_to_Neuron_SIC refuses bad input')
    from brian2 import NeuronGroup, start_scope, prefs
    prefs.codegen.target = 'numpy'
    start_scope()
    astro = NeuronGroup(2, 'G_A : mole', name='G_stub')
    plain = NeuronGroup(3, 'v : 1', name='N_plain')
    good = NeuronGroup(3, 'r_sic_tot : 1', name='N_good')
    n_raise = 0
    for args in ((astro, plain, [[0], [0]]), (astro, good, [[0, 1], [0]]),
                 (astro, good, [[2], [0]]), (astro, good, [[0], [3]])):
        try:
            M.Astro_to_Neuron_SIC(*args)
        except ValueError:
            n_raise += 1
    check(n_raise == 4, 'no r_sic_tot, length mismatch, astrocyte and neuron index out of range: '
          '%d of 4 raise ValueError' % n_raise)


# ---------------------------------------------------------------------------
# S10, S11: compiled runs
# ---------------------------------------------------------------------------
RUN_ARGS = ['--simtime', '2', '--density', '1200', '--density_astro', '1200', '--c_max', '150',
            '--conn_prob', '0.2', '--conn_rule', 'flat', '--topology_mode', 'wallach',
            '--seed_run', '7', '--dpi', '50']

# Every compiled run goes through HPC_single_run.main() of the code under test,
# with its figures switched off: the 3-D topology figure fails at random
# (SPEC section 8) and would make these checks flaky. variant (current code):
#   'on'   the pipeline as it is (main() passes no sic: on in Full mode)
#   'off'  build_and_run(..., sic=False, astro_ic='quiescent'): the neurons without
#          r_sic_tot, no SIC group, the astrocytes' initial state of before (Block 13)
#   'zero' the SIC terms in the neuron equations, but an inert group in place of
#          the SIC synapses, so r_sic_tot stays 0; astro_ic='quiescent'
DRIVER = r'''
import sys, functools
sys.path.insert(0, %(code_dir)r)
import ASD_fun_BD_cpp as M
import HPC_single_run as R
if %(variant)r == 'off':
    R.build_and_run = functools.partial(R.build_and_run, sic=False, astro_ic='quiescent')
elif %(variant)r == 'zero':
    R.build_and_run = functools.partial(R.build_and_run, astro_ic='quiescent')
    from brian2 import Synapses
    def _no_sic(GT, N, connections, params=None):
        return Synapses(GT, N, model='', name='ecs_astro_to_neuron_sic')
    M.Astro_to_Neuron_SIC = _no_sic
for _f in ('plot_spatial_layout', 'plot_3d_layered_topology', 'plot_neuronal_raster',
           'plot_astrocyte_raster'):
    setattr(R, _f, lambda *a, **k: None)
sys.argv = ['HPC_single_run.py'] + %(argv)r
R.main()
'''


def _drive(code_dir, mode, out, cwd, variant='on'):
    argv = ['--out_dir', out, '--lib_dir', code_dir, '--mode', mode] + RUN_ARGS
    return _run([sys.executable, '-c', DRIVER % dict(code_dir=code_dir, variant=variant, argv=argv)], cwd)


def _run(cmd, cwd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout + r.stderr


def _spikes(out):
    f = sorted(glob.glob(os.path.join(out, 'iter_*.npz')))
    return np.load(f[0]) if len(f) == 1 else None


def s10(tmp, d_numpy, M):
    print('\n[S10] cpp_standalone')
    dc = stand_in(tmp, True, 0.05)
    if check(dc is not None, 'stand-in compiled and run'):
        if d_numpy is None:
            skip('C++ against numpy: the numpy stand-in failed')
        else:
            rel = np.max(np.abs(dc['r_pair'] - d_numpy['r_pair'])) / np.max(np.abs(d_numpy['r_pair']))
            check(rel < 1e-8, 'C++ r_sic = numpy r_sic, max rel. diff %.1e (< 1e-8)' % rel)
    res = {}
    for mode in ('Full', 'Neuronal'):
        out = os.path.join(tmp, 'new_' + mode)
        code, log = _drive(HERE, mode, out, tmp)
        z = _spikes(out)
        if not check(code == 0 and z is not None, 'HPC_single_run.py, %s mode, 2 s: exit 0, one npz' % mode):
            print(log[-3000:])
            continue
        topo = np.load(os.path.join(out, 'topology.npz'))
        res[mode] = z
        if mode == 'Full':
            check(str(z['sic_scheme']) == M.SIC_SCHEME and int(z['n_sic_pairs']) == len(topo['SIC_i']) > 0
                  and str(z['astro_ic']) == M.ASTRO_IC_SCHEME,
                  'Full: sic_scheme %s, n_sic_pairs %d = len(SIC_i) %d > 0, astro_ic %s'
                  % (z['sic_scheme'], int(z['n_sic_pairs']), len(topo['SIC_i']), z['astro_ic']))
        else:
            check('sic_scheme' not in z.files and 'n_sic_pairs' not in z.files and 'astro_ic' not in z.files
                  and len(topo['SIC_i']) == 0, 'Neuronal: no SIC or astro_ic keys, SIC_i empty')
    return res


def s11(tmp, new_runs):
    print('\n[S11] SIC off = before Block 12 (%s)' % PRE_SIC_COMMIT)
    old_dir = os.path.join(tmp, 'pre_sic')
    os.makedirs(old_dir, exist_ok=True)
    for f in ('ASD_fun_BD_cpp.py', 'HPC_single_run.py', 'synapse_pdist.csv'):
        r = subprocess.run(['git', '-C', HERE, 'show',
                            '%s:hpc/Phenomenological_finalv1/%s' % (PRE_SIC_COMMIT, f)], capture_output=True)
        if r.returncode != 0:
            skip('S11: git show %s failed' % f)
            return
        open(os.path.join(old_dir, f), 'wb').write(r.stdout)
    keys = ('spk_N_t', 'spk_N_i', 'spk_A_t', 'spk_A_i')
    for mode in ('Full', 'Neuronal'):
        out_old = os.path.join(tmp, 'old_' + mode)
        code, log = _drive(old_dir, mode, out_old, old_dir)
        z_old = _spikes(out_old)
        if not check(code == 0 and z_old is not None, '%s, %s mode: exit 0' % (PRE_SIC_COMMIT, mode)):
            print(log[-3000:])
            continue
        if mode == 'Full':
            pairs = []
            for variant, label in (('off', 'Full mode: current code with the SIC off (sic=False)'),
                                   ('zero', 'Full mode: current code with the SIC terms but r_sic_tot = 0')):
                out_new = os.path.join(tmp, 'new_Full_sic_' + variant)
                code, log = _drive(HERE, 'Full', out_new, tmp, variant=variant)
                z_new = _spikes(out_new)
                if not check(code == 0 and z_new is not None, 'current code, Full mode, SIC %s: exit 0' % variant):
                    print(log[-3000:])
                    continue
                pairs.append((label, z_new))
        else:
            z_new = new_runs.get('Neuronal')
            if z_new is None:
                skip('Neuronal comparison: the current Neuronal run failed (S10)')
                continue
            pairs = [('Neuronal mode: current code', z_new)]
        for label, z_new in pairs:
            same = all(np.array_equal(z_old[k], z_new[k]) for k in keys)
            check(same, '%s = %s, spike for spike (%d neuron spikes, %d astrocyte events)'
                  % (label, PRE_SIC_COMMIT, len(z_old['spk_N_t']), len(z_old['spk_A_t'])))
        if mode == 'Full' and 'Full' in new_runs:
            z_on = new_runs['Full']
            same_on = all(np.array_equal(z_old[k], z_on[k]) for k in keys)
            print('  info: with the SIC on, the same 2 s run %s the pre-Block-12 spikes '
                  '(%d vs %d neuron spikes, %d astrocyte events)'
                  % ('reproduces' if same_on else 'differs from', len(z_on['spk_N_t']),
                     len(z_old['spk_N_t']), len(z_on['spk_A_t'])))


def main():
    import importlib.util
    tmp = tempfile.mkdtemp(prefix='sic_smoke_')
    try:
        M = lib()
        s1(M)
        s2(M)
        s3(M, tmp)
        d = s4_s6(tmp)
        R = s7(M)
        s8(R, M)
        s9(M)
        if '--no-sim' in sys.argv:
            skip('S10 (--no-sim)')
            skip('S11 (--no-sim)')
        elif shutil.which('g++') is None and shutil.which('c++') is None:
            skip('S10, S11: no C++ compiler')
        else:
            new_runs = s10(tmp, d, M)
            s11(tmp, new_runs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('\nRESULT: %d passed, %d failed%s' % (RESULTS['passed'], RESULTS['failed'],
          (', %d skipped' % RESULTS['skipped']) if RESULTS['skipped'] else ''))
    return 0 if RESULTS['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
