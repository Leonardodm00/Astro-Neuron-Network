"""Block 12 -- the slow inward current (SPEC.md Block 12).

Oracles: the constants; the reach rule against scipy's Delaunay and Voronoi
computed here; the calibration target re-derived with scipy.integrate from the
spec's equations; the library's Brian2 SIC synapse against that scipy
solution; the switch and the refusals; the record keys. The compiled
spike-for-spike check (g_sic = 0 against the SIC-off twin) is in
test_spec_e2e_sweep.py. Run:
    python -m pytest tests/spec/test_spec_block12_sic.py -q
"""
import itertools
import re
import sys

import numpy as np
import numpy.testing as npt
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay, Voronoi

from conftest import PIPE, brian2_runtime

sys.path.insert(0, PIPE)
import ASD_fun_BD_cpp as L      # noqa: E402
import HPC_single_run as R      # noqa: E402

K_ON, TAU_D, TAU_R, EC50, N_H, G_SIC = 3.596e3, 0.6519, 0.002, 3.0e-3, 1.9, 40.6e-9   # SI: 1/s, s, s, M, -, S


def test_constants():
    from brian2 import ms, mmole, nS
    p = L.get_SICparam()
    assert float(p['k_on_sic'] * ms) == pytest.approx(3.596, rel=1e-12)
    assert float(p['tau_d_sic'] / ms) == pytest.approx(651.9, rel=1e-12)
    assert float(p['tau_r_sic'] / ms) == pytest.approx(2.0, rel=1e-12)
    assert float(p['EC50_sic'] / mmole) == pytest.approx(3.0, rel=1e-12)
    assert p['n_sic'] == 1.9
    assert float(p['g_sic'] / nS) == pytest.approx(40.6, rel=1e-12)
    assert L.SIC_SCHEME == 'sic-v1-2026-11-04'


# ---------------------------------------------------------------------------
# reach rule
# ---------------------------------------------------------------------------
def ref_pairs(N_pos, A_pos, max_dist=np.inf, use_voronoi=False):
    Nn = len(N_pos)
    pts = np.vstack([N_pos, A_pos])
    if use_voronoi:
        edges = {tuple(sorted(map(int, rp))) for rp in Voronoi(pts).ridge_points}
    else:
        edges = set()
        for s in Delaunay(pts).simplices:
            for a, b in itertools.combinations(sorted(map(int, s)), 2):
                edges.add((a, b))
    out = []
    for a, b in edges:
        if a < Nn <= b:
            n, ast = a, b - Nn
            if np.hypot(*(A_pos[ast] - N_pos[n])) <= max_dist:
                out.append((ast, n))
    return sorted(out)


@pytest.mark.parametrize('seed', range(6))
def test_sic_pairs_equal_joint_tessellation(seed):
    rng = np.random.default_rng(seed)
    N_pos, A_pos = rng.uniform(0, 300, (108, 2)), rng.uniform(0, 300, (108, 2))
    a, n = L.sic_pairs(N_pos, A_pos, max_dist=np.inf)
    got = list(zip(a.tolist(), n.tolist()))
    assert a.dtype == np.int32 and n.dtype == np.int32
    assert got == sorted(got)                                       # sorted by (astrocyte, neuron)
    assert got == ref_pairs(N_pos, A_pos) == ref_pairs(N_pos, A_pos, use_voronoi=True)
    capped = list(zip(*[x.tolist() for x in L.sic_pairs(N_pos, A_pos, max_dist=40.0)]))
    assert capped == ref_pairs(N_pos, A_pos, 40.0)


def test_sic_pairs_degenerate_and_empty():
    with pytest.raises(ValueError):
        L.sic_pairs(np.array([[0, 0], [1, 1], [2, 2]], float), np.array([[3, 3], [4, 4]], float))   # collinear
    with pytest.raises(ValueError):
        L.sic_pairs(np.array([[0, 0], [5, 1], [2, 7]], float), np.array([[5, 1], [9, 9]], float))   # coincident
    a, n = L.sic_pairs(np.zeros((0, 2)), np.array([[1.0, 2.0]]))
    assert len(a) == len(n) == 0


def test_build_topology_uses_uncapped_pairs():
    import pandas as pd
    import os
    pd_csv = pd.read_csv(os.path.join(PIPE, 'synapse_pdist.csv'))
    t = R.build_topology(Nn=60, Na=60, c_max=300.0, conn_prob=0.1, syn_prob_csv=pd_csv, displ_bias=20.0,
                         gj_dist=200.0, stoa_cutoff=70.0, stoa_sigma=200.0, seed_neuron=1, seed_synapse=2,
                         seed_astro=3, mode='Full', topology_mode='wallach', gj_max_dist=150.0,
                         stoa_gate_p=0.3, seed_stoa_gate=9)
    got = list(zip(t['SIC_i'].tolist(), t['SIC_j'].tolist()))
    assert got == ref_pairs(t['N_pos'], t['A_pos'])               # no cap, not thinned by the gate


# ---------------------------------------------------------------------------
# calibration target (spec): G0 = 78 uM, Omega_e = 5/s, V = -60 mV
# ---------------------------------------------------------------------------
def scipy_sic(G0=78e-6, omega_e=5.0, T=6.0, dt_out=1e-4):
    def rhs(t, y):
        G = G0 * np.exp(-omega_e * t)
        h = (G / EC50) ** N_H / (1 + (G / EC50) ** N_H)
        x, r = y
        return [K_ON * h * (1 - x) - x / TAU_D, (x - r) / TAU_R]
    t = np.arange(0.0, T, dt_out)
    sol = solve_ivp(rhs, (0, T), [0.0, 0.0], t_eval=t, method='LSODA', rtol=1e-10, atol=1e-13)
    return t, sol.y[0], sol.y[1]


def mg_block(v_mV):
    return 1.0 / (1.0 + np.exp(-0.062 * v_mV) / 3.57)


def test_calibration_target_from_the_spec_equations():
    t, x, r = scipy_sic()
    k = int(np.argmax(r))
    I_peak = G_SIC * r[k] * (-60e-3 - 0.0) * mg_block(-60.0) * 1e12              # pA
    assert r[k] == pytest.approx(0.2233, abs=5e-5)
    assert I_peak == pytest.approx(-43.31, abs=0.01)
    assert t[k] * 1e3 == pytest.approx(220.1, abs=0.2)                         # 0-100 % rise, ms
    tail = slice(k, k + int(np.argmax(r[k:] < 0.1 * r[k])))
    (A, tau), _ = curve_fit(lambda s, A, tau: A * np.exp(-s / tau), t[tail] - t[k], r[tail], p0=(r[k], 0.7))
    assert tau * 1e3 == pytest.approx(706.0, abs=1.5)                           # decay tau, ms


def standin_groups(b2, n_astro, n_neu, G0_uM, omega_e):
    G = b2.NeuronGroup(n_astro, 'dG_A/dt = -omega_e*G_A : mole', name='Gliot_release',
                       namespace=dict(omega_e=omega_e / b2.second))
    G.G_A = np.asarray(G0_uM) * b2.umole
    Nn = b2.NeuronGroup(n_neu, 'r_sic_tot : 1', name='Neuron')
    return G, Nn


def test_brian2_sic_synapse_matches_scipy():
    b2 = brian2_runtime()
    b2.defaultclock.dt = 0.05 * b2.ms
    G, N = standin_groups(b2, 3, 3, [78.0, 78.0, 300.0], 5.0)
    # neuron 0: astrocyte 0; neuron 1: astrocytes 0 and 2 (sum); neuron 2: no pair
    syn = L.Astro_to_Neuron_SIC(G, N, [np.array([0, 0, 2]), np.array([0, 1, 1])])
    mon = b2.StateMonitor(N, 'r_sic_tot', record=True, dt=1 * b2.ms)
    net = b2.Network(G, N, syn, mon)
    net.run(1.0 * b2.second)
    tt = np.asarray(mon.t / b2.second)
    _, _, r78 = scipy_sic(78e-6, 5.0, T=1.05, dt_out=1e-5)
    _, _, r300 = scipy_sic(300e-6, 5.0, T=1.05, dt_out=1e-5)
    # the spec's one-step lag of the summed r_sic_tot, plus the monitor reading at the step's start:
    # each sample is compared with scipy at a delay s in [0, 2 steps] = [0, 0.1 ms] (best s per point)
    got = np.asarray(mon.r_sic_tot)

    def best_err(y, ref):
        errs = [np.abs(y - ref[np.maximum(np.round((tt - s) / 1e-5).astype(int), 0)])
                for s in np.arange(0, 11) * 1e-5]
        return np.min(errs, axis=0)
    # tolerance: first-order integration in dt = 0.05 ms gives <= ~1e-4 here once the lag is
    # allowed for; 1.5e-4 per pair is 0.07 % of the 78 uM peak 0.2233
    assert best_err(got[0], r78).max() < 1.5e-4
    assert best_err(got[1], r78 + r300).max() < 3e-4
    assert np.all(got[2] == 0.0)
    assert got[0].max() == pytest.approx(r78.max(), rel=5e-4)          # peak 0.2233 (spec)
    assert abs(tt[np.argmax(got[0])] - 0.2201) <= 0.002                  # rise 220.1 ms, 1 ms sampling


def test_sic_group_refusals():
    b2 = brian2_runtime()
    G, N = standin_groups(b2, 2, 2, [1.0, 1.0], 5.0)
    plain = b2.NeuronGroup(2, 'v : 1', name='Plain')
    with pytest.raises(ValueError):
        L.Astro_to_Neuron_SIC(G, plain, [np.array([0]), np.array([0])])        # no r_sic_tot
    for conn in ([np.array([0, 1]), np.array([0])], [np.array([2]), np.array([0])],
                 [np.array([0]), np.array([-1])], [np.array([0]), np.array([2])]):
        with pytest.raises(ValueError):
            L.Astro_to_Neuron_SIC(G, N, conn)
    empty = L.Astro_to_Neuron_SIC(G, N, [np.array([], int), np.array([], int)])
    net = b2.Network(G, N, empty)
    net.run(1 * b2.ms)
    assert len(empty) == 0 and np.all(np.asarray(N.r_sic_tot) == 0)


def test_neuron_equation_and_switch():
    src = open(L.__file__, encoding='utf-8').read()
    assert re.search(r'I_sic\s*=\s*g_sic\*\(V-E_nmda\)\*\(r_sic_tot\)/\(1\+exp\(-0\.062\*V/mV\)/3\.57\)\s*:\s*amp',
                     src)
    assert re.search(r'I_syn\s*=\s*I_ampa \+ I_nmda \+ I_sic\s*:\s*amp', src)
    dummy = object()
    with pytest.raises(ValueError):
        L.Neuronal_Network(5, Simulated_network='Neuronal', Binomial_fun=dummy, sic=True)
    with pytest.raises(ValueError):
        L.Neuronal_Network(5, Simulated_network='Full', Binomial_fun=dummy, sic='yes')


def test_record_keys_and_build_and_run_switch():
    assert R.sic_record_keys('Full', 12, L.SIC_SCHEME) == {'sic_scheme': 'sic-v1-2026-11-04', 'n_sic_pairs': 12}
    assert R.sic_record_keys('Neuronal', 0, L.SIC_SCHEME) == {}
    for bad in (True, 'on', 1):
        with pytest.raises(ValueError):
            R.build_and_run({}, None, None, sic=bad)
