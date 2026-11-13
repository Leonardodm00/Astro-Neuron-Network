"""Block 8 -- the 38-D registry, the sweep groups and the astrocyte contact gate
(SPEC.md Block 8, with the g_sic axis of Block 12 and U_0_sr's box of section 4).

Oracles: the values the spec states; the log rule (lo > 0 and hi/lo >= 10)
evaluated in exact decimal arithmetic; group sizes; the gate's contract
(subset of the viable list, Bernoulli(p) per viable pair, no effect on the
position and wiring streams, p >= 1 draws nothing, ValueError outside [0, 1]).
Run:
    python -m pytest tests/spec/test_spec_block08_registry_gate.py -q
"""
import os
import sys
from fractions import Fraction

import numpy as np
import numpy.testing as npt
import pytest
from scipy import stats

from conftest import PIPE, run

sys.path.insert(0, PIPE)
import HPC_main_sweep as S      # noqa: E402
import HPC_single_run as R      # noqa: E402


def test_registry_shape_and_new_rows():
    assert S.N_DIMS == 38 and S.PARAM_BOUNDS.shape == (38, 2) and len(S.PARAM_NAMES) == 38
    n = S.PARAM_NAMES
    assert n[36] == 'O_N' and n[37] == 'g_sic' and n[7] == 'U_0_sr'
    assert S.NOMINAL_PARAMS[36] == 0.3 and S.NOMINAL_PARAMS[37] == 40.6
    assert tuple(S.PARAM_BOUNDS[36]) == (0.0001, 1.0)
    assert tuple(S.PARAM_BOUNDS[37]) == (4.0, 400.0)
    assert tuple(S.PARAM_BOUNDS[7]) == (0.01, 0.7)
    assert tuple(S.PARAM_BOUNDS[n.index('I_bias')]) == (0.3, 1.0)
    assert S.PARAM_UNITS[37] == 'nS'
    assert 'O_N' in R.ASTRO_PARAMS and 'g_sic' in R.ASTRO_PARAMS
    npt.assert_allclose(S.PARAM_BOUNDS_THETA[36], [np.log(1e-4), 0.0], rtol=0, atol=1e-15)


def test_log_rule_exact_decimal():
    """log iff lo > 0 and hi/lo >= 10, with the ratio taken in exact decimal arithmetic."""
    for k in range(S.N_DIMS):
        lo, hi = (Fraction(repr(float(x))) for x in S.PARAM_BOUNDS[k])
        want = lo > 0 and hi > 0 and hi / lo >= 10
        assert (k in S.LOG_PARAMS) == want, (S.PARAM_NAMES[k], float(lo), float(hi))
        if want:
            npt.assert_allclose(S.PARAM_BOUNDS_THETA[k], np.log(S.PARAM_BOUNDS[k]), rtol=0, atol=0)
        else:
            npt.assert_array_equal(S.PARAM_BOUNDS_THETA[k], S.PARAM_BOUNDS[k])
    for name in ('O_N', 'g_sic', 'U_0_sr', 'x0', 'alpha_syn'):
        assert S.PARAM_NAMES.index(name) in S.LOG_PARAMS, name
    for name in ('Sigma', 'I_bias', 'U_A', 'VA', 'VR'):
        assert S.PARAM_NAMES.index(name) not in S.LOG_PARAMS, name


def test_theta_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(20):
        th = S.sample_theta(rng)
        assert np.all(th >= S.PARAM_BOUNDS_THETA[:, 0]) and np.all(th <= S.PARAM_BOUNDS_THETA[:, 1])
        npt.assert_allclose(S.natural_to_theta(S.theta_to_natural(th)), th, rtol=1e-12, atol=1e-12)


def test_sweep_groups():
    names = S.PARAM_NAMES
    g = {k: [names[i] for i in S.resolve_sweep_group(k)] for k in ('neuron_synapse', 'tripartite', 'all')}
    assert len(g['neuron_synapse']) == 23
    assert set(g['all']) - set(g['tripartite']) == {'I_Theta', 'omega_I'}
    assert set(g['tripartite']) <= set(g['all'])
    for frozen in ('DeltaT', 'VT', 'gL'):
        assert all(frozen not in v for v in g.values())
    assert 'O_N' in g['tripartite'] and 'g_sic' in g['tripartite'] and 'g_sic' not in g['neuron_synapse']
    # with g_sic in ASTRO_PARAMS (Block 12) the 37-D counts of Block 8 (32, 34) grow by one
    assert len(g['tripartite']) == 33 and len(g['all']) == 35


# ---------------------------------------------------------------------------
# contact gate in build_topology
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def pdist():
    import pandas as pd
    return pd.read_csv(os.path.join(PIPE, 'synapse_pdist.csv'))


def topo(pdist, p, seed_gate=71, mode='Full', seeds=(39, 35, 60)):
    return R.build_topology(Nn=40, Na=40, c_max=200.0, conn_prob=0.2, syn_prob_csv=pdist, displ_bias=20.0,
                            gj_dist=200.0, stoa_cutoff=70.0, stoa_sigma=200.0, seed_neuron=seeds[0],
                            seed_synapse=seeds[1], seed_astro=seeds[2], mode=mode, topology_mode='wallach',
                            gj_max_dist=150.0, stoa_gate_p=p, seed_stoa_gate=seed_gate)


def pairs(t, a='StoA_i', b='StoA_j'):
    return list(zip(np.asarray(t[a]).tolist(), np.asarray(t[b]).tolist()))


def test_gate_off_draws_nothing_and_keeps_all(pdist):
    t1 = topo(pdist, 1.0, seed_gate=71)
    t2 = topo(pdist, 1.0, seed_gate=None)
    for k in t1:
        npt.assert_array_equal(t1[k], t2[k], err_msg=k)
    assert pairs(t1) == pairs(t1, 'StoA_viable_i', 'StoA_viable_j') and len(t1['StoA_i']) > 50


def test_gate_zero_and_half(pdist):
    full = topo(pdist, 1.0)
    viable = pairs(full)
    t0 = topo(pdist, 0.0)
    assert len(t0['StoA_i']) == 0 and pairs(t0, 'StoA_viable_i', 'StoA_viable_j') == viable
    th = topo(pdist, 0.5)
    kept = pairs(th)
    assert set(kept) <= set(viable) and len(set(kept)) == len(kept)
    assert pairs(th, 'StoA_viable_i', 'StoA_viable_j') == viable
    # positions, wiring, gap junctions and SIC pairs do not depend on the gate stream
    for k in ('N_pos', 'A_pos', 'S_i', 'S_j', 'S_x_syn', 'S_y_syn', 'GJ_i', 'GJ_j', 'SIC_i', 'SIC_j'):
        npt.assert_array_equal(th[k], full[k], err_msg=k)
    assert pairs(topo(pdist, 0.5)) == kept                     # same seeds, same draw


def test_gate_is_bernoulli_p_per_viable_pair(pdist):
    viable = pairs(topo(pdist, 1.0))
    n = len(viable)
    p = 0.3
    reps = 300
    hits = np.zeros(n)
    for s in range(reps):
        kept = set(pairs(topo(pdist, p, seed_gate=1000 + s)))
        hits += [v in kept for v in viable]
    # overall share: Binomial(n * reps, p); 5 sigma tolerance
    tot = hits.sum()
    sd = np.sqrt(n * reps * p * (1 - p))
    assert abs(tot - n * reps * p) < 5 * sd
    # no pair favoured: per-pair counts ~ Binomial(reps, p), chi-square over pairs
    chi2 = np.sum((hits - reps * p) ** 2 / (reps * p * (1 - p)))
    assert stats.chi2.sf(chi2, n) > 1e-4


@pytest.mark.parametrize('p', [-0.01, 1.01, float('nan')])
def test_gate_out_of_range_raises(pdist, p):
    with pytest.raises(ValueError):
        topo(pdist, p)


def test_neuronal_mode_has_no_astrocyte_lists(pdist):
    t = topo(pdist, 1.0, mode='Neuronal')
    for k in ('StoA_i', 'StoA_viable_i', 'SIC_i', 'SIC_j'):
        assert len(t[k]) == 0


@pytest.mark.parametrize('lo,hi', [(0.8, 0.2), (-0.1, 0.5), (0.2, 1.5)])
def test_gate_prior_validated_before_any_run(tmp_path, lo, hi):
    r = run([sys.executable, os.path.join(PIPE, 'HPC_main_sweep.py'), '--out_dir', str(tmp_path / 'o'),
             '--mode', 'Full', '--stoa_gate_lo', str(lo), '--stoa_gate_hi', str(hi), '--n_workers', '1',
             '--n_topologies', '1', '--simtime', '0.1', '--lib_dir', PIPE], cwd=str(tmp_path), timeout=300)
    assert r.returncode != 0 and 'ValueError' in r.stderr
    assert not any(n.startswith('topo_') for n in os.listdir(str(tmp_path / 'o')))
