"""Block 13 -- astrocyte initial conditions at the driven state (SPEC.md Block 13).

Oracles: the driven fixed point is stationary under the library's own Brian2
equations (Astrocyte_Group, runtime device); the cycle's sample at phase 1/2
is where the live equations take the trough after half a period; the phase
and jitter streams as the spec defines them; the mapping of astro_driven_ics;
both groups start refractory; the record keys. Run:
    python -m pytest tests/spec/test_spec_block13_astro_ic.py -q
"""
import sys

import numpy as np
import numpy.testing as npt
import pytest

from conftest import PIPE, brian2_runtime

sys.path.insert(0, PIPE)
import ASD_fun_BD_cpp as L      # noqa: E402
import HPC_single_run as R      # noqa: E402

Y_REF_UM = 1700.0


def test_constants():
    assert L.ASTRO_IC_SCHEME == 'driven-v1-2026-11-04'
    assert L.ASTRO_IC_Y_REF_UM == Y_REF_UM                      # Y_ref = 1.7 mM
    assert set(L.ASTRO_IC_PARAMS) == {'O_beta', 'O_3K', 'Omega_5P', 'I_bias', 'F', 'I_Theta', 'omega_I', 'O_N'}


def test_phase_stream():
    for seed in (1, 12345, 2 ** 31 - 2):
        u = L.astro_ic_phases(50, seed)
        want = np.random.default_rng(np.random.SeedSequence([seed, 13])).random(50)
        npt.assert_array_equal(u, want)
        assert np.all((u >= 0) & (u < 1))
        npt.assert_array_equal(L.astro_ic_phases(50, seed), u)


def test_jitter_stream_unchanged():
    seed = 4242
    rng = np.random.default_rng(seed)
    want = [1.0 + 0.05 * (2.0 * rng.random(30) - 1.0) for _ in range(3)]       # I, C, h, in that order
    got = L.astro_ic_jitter(30, seed)
    for g, w in zip(got, want):
        npt.assert_array_equal(g, w)
    assert not np.array_equal(L.astro_ic_phases(30, seed), rng.random(30))


def brian_astro(state, y_extra_uM, params=None, T=10.0, dt_ms=1.0, record_dt_ms=None):
    """One astrocyte of the library's Astrocyte_Group (no gap junction), Y_extra held fixed;
    started at `state`; returns the final (Gamma_A, I, C, h) and, if asked, C(t)."""
    b2 = brian2_runtime()
    b2.defaultclock.dt = dt_ms * b2.ms
    # one astrocyte with a gap junction onto itself: delta_I = 0, so sign(0) = 0 and no flux
    Astro, GJ = L.Astrocyte_Group(1, 'Full', seed_astro=1, ics='steady',
                                  connections=[np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)])
    for k, v in (params or {}).items():
        setattr(Astro, k, v)
    Astro.Y_extra = y_extra_uM * b2.umole
    Astro.Gamma_A = state['Gamma_A']
    Astro.I = state['I'] * b2.umole
    Astro.C = state['C'] * b2.umole
    Astro.h = state['h']
    objs = [Astro, GJ]
    mon = None
    if record_dt_ms:
        mon = b2.StateMonitor(Astro, ['C'], record=True, dt=record_dt_ms * b2.ms)
        objs.append(mon)
    net = b2.Network(*objs)
    net.run(T * b2.second)
    final = dict(Gamma_A=float(Astro.Gamma_A[0]), I=float(Astro.I[0] / b2.umole),
                 C=float(Astro.C[0] / b2.umole), h=float(Astro.h[0]))
    return final, (None if mon is None else (np.asarray(mon.t / b2.second), np.asarray(mon.C[0] / b2.umole)))


@pytest.mark.parametrize('Y', [Y_REF_UM])
def test_driven_fixed_point_is_stationary_under_live_equations(Y):
    # registry nominal point (get_Astroparam's values): a linked astrocyte (Y_ref) is at a fixed point
    st = L.astro_driven_state(None, Y)
    assert not st['oscillating'] and st['cycle'] is None
    final, _ = brian_astro(st, Y, T=10.0)
    # spec: a fixed point when C varies by at most 1e-4 uM; the group is float32 (C ~ 0.3 uM has
    # ~3e-8 uM resolution), so 1e-4 uM over 10 s is the spec's own criterion
    assert abs(final['C'] - st['C']) <= 1e-4
    assert abs(final['I'] - st['I']) <= 1e-3 and abs(final['h'] - st['h']) <= 1e-3
    assert abs(final['Gamma_A'] - st['Gamma_A']) <= 1e-3


def test_quiescent_start_is_not_stationary_under_drive():
    """Sensitivity of the oracle above: the old quiescent state moves (the artefact of Block 13)."""
    b2 = brian2_runtime()
    ss = L._astrocyte_steady_state(L.get_Astroparam())
    q = dict(Gamma_A=ss['Gamma_A'], I=float(ss['I'] / b2.umole), C=float(ss['C'] / b2.umole), h=ss['h'])
    final, _ = brian_astro(q, Y_REF_UM, T=10.0)
    assert abs(final['C'] - q['C']) > 1e-2


@pytest.mark.parametrize('p,Y', [({'I_bias': 0.3}, Y_REF_UM),   # the spec's cycle example (I_bias 0.3)
                                 ({}, 0.0)])                      # an unlinked astrocyte at the nominal point
def test_limit_cycle_samples(p, Y):
    st = L.astro_driven_state(p, Y)
    assert st['oscillating'] and st['period'] and st['period'] > 0
    cyc = np.asarray(st['cycle'])
    assert cyc.shape == (1000, 4) and np.all(np.isfinite(cyc))
    C = cyc[:, 2]
    assert C[0] <= C.min() + 1e-3 * (C.max() - C.min())           # starts at a trough of C
    assert (st['Gamma_A'], st['I'], st['C'], st['h']) == tuple(cyc[0])
    # live equations: from the trough, half a period later the state is the cycle's sample 500
    from brian2 import umole
    params = {k: v * umole for k, v in p.items()}
    start = dict(Gamma_A=cyc[0, 0], I=cyc[0, 1], C=cyc[0, 2], h=cyc[0, 3])
    final, _ = brian_astro(start, Y, params=params, T=st['period'] / 2.0, dt_ms=0.5)
    amp = C.max() - C.min()
    assert abs(final['C'] - cyc[500, 2]) < 0.05 * amp, (final['C'], cyc[500, 2], amp)


def test_driven_ics_mapping():
    linked = np.array([True, False, True, True, False])
    phases = np.array([0.0, 0.5, 0.999, 0.25, 0.75])
    ics, states = L.astro_driven_ics(None, linked, phases)
    on, off = L.astro_driven_state(None, Y_REF_UM), L.astro_driven_state(None, 0.0)
    assert off['cycle'] is not None                              # nominal point: unlinked on a cycle
    for j, k in enumerate(('Gamma_A', 'I', 'C', 'h')):
        npt.assert_array_equal(ics[k][linked], on[k])            # fixed point: no jitter
        npt.assert_array_equal(ics[k][~linked], np.asarray(off['cycle'])[np.floor(phases[~linked] * 1000).astype(int), j])
    p = {'I_bias': 0.3}
    ics2, st2 = L.astro_driven_ics(p, linked, phases)
    cyc = np.asarray(st2[True]['cycle'])
    for j, k in enumerate(('Gamma_A', 'I', 'C', 'h')):
        npt.assert_array_equal(ics2[k][linked], cyc[np.floor(phases[linked] * 1000).astype(int), j])
    with pytest.raises(ValueError):
        L.astro_driven_ics(None, linked, phases[:4])
    with pytest.raises(ValueError):
        L.astro_driven_ics(None, linked, np.array([0.0, 0.5, 1.0, 0.25, 0.75]))


def test_both_groups_start_refractory():
    b2 = brian2_runtime()
    Astro, GJ = L.Astrocyte_Group(4, 'Full', seed_astro=3, ics='steady',
                                  connections=[np.arange(4, dtype=np.int32), np.arange(4, dtype=np.int32)])
    GT = L.Gliotransmission(4, Astro, ics='jitter', seed_astro=3)
    assert not np.any(np.asarray(Astro.not_refractory)) and not np.any(np.asarray(GT.not_refractory))


def test_record_keys():
    assert R.astro_ic_record_keys('Full', L.ASTRO_IC_SCHEME) == {'astro_ic': 'driven-v1-2026-11-04'}
    assert R.astro_ic_record_keys('Neuronal', L.ASTRO_IC_SCHEME) == {}
