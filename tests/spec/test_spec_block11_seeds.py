"""Block 11 -- random seeds (SPEC.md Block 11): the derivation formula, the
validity range, the refusals before anything is built. The behavioural
oracles (same seed -> identical runs, network-level draws per network) are in
test_spec_e2e_sweep.py, through the real entry point. Run:
    python -m pytest tests/spec/test_spec_block11_seeds.py -q
"""
import sys

import numpy as np
import pytest

from conftest import PIPE

sys.path.insert(0, PIPE)
import HPC_single_run as R      # noqa: E402
import HPC_main_sweep as S      # noqa: E402

TAGS = (('seed_build', 1), ('seed_ic_neuron', 2), ('seed_ic_synapse', 3), ('seed_ic_astro', 4))


def spec_seeds(seed_device, network_key):
    """1 + w mod (2^31 - 2), w = first uint32 word of SeedSequence([seed_device, key, tag])."""
    out = {}
    for name, tag in TAGS:
        w = int(np.random.SeedSequence([seed_device, network_key, tag]).generate_state(1, dtype=np.uint32)[0])
        out[name] = 1 + w % (2 ** 31 - 2)
    return out


@pytest.mark.parametrize('dev,key', [(50, 0), (50, 1), (50, 1821970821), (7, 0), (0, 2 ** 31 - 2)])
def test_derive_network_seeds_formula(dev, key):
    got = R.derive_network_seeds(dev, key)
    assert got == spec_seeds(dev, key)
    assert all(1 <= v <= 2 ** 31 - 2 for v in got.values())
    assert len(set(got.values())) == 4


def test_network_seeds_differ_between_networks():
    a, b = R.derive_network_seeds(50, 100), R.derive_network_seeds(50, 101)
    assert all(a[k] != b[k] for k in a)
    assert R.derive_network_seeds(51, 100) != a


@pytest.mark.parametrize('s', [0, 1, 12345, 2 ** 31 - 2, np.int64(77), np.int32(5)])
def test_validate_run_seed_accepts(s):
    assert R.validate_run_seed(s) == int(s)


@pytest.mark.parametrize('s', [-1, 2 ** 31 - 1, 2 ** 40, 1.5, '12', 'x', None, float('nan'), float('inf')])
def test_validate_run_seed_rejects_with_valueerror(s):
    with pytest.raises(ValueError):
        R.validate_run_seed(s)


def fake_pack(tmp_path, **over):
    pack = dict(worker_id=0, topo={}, topo_idx=0, topo_dir=str(tmp_path / 't'), conn_prob=0.1,
                conn_rule='flat', conn_periodic=False, p0_conn=np.nan, d0_conn=np.nan, beta_conn=np.nan,
                stoa_gate_p=1.0, params_list=np.zeros((1, 38)), theta_list=np.zeros((1, 38)),
                iter_indices=[0], seed_runs=[5], topo_seed_base=10,
                cli=dict(seed_device=50, lib_dir=PIPE), scratch_root=str(tmp_path / 'scratch'))
    pack.update(over)
    return pack


def test_pack_refusals_before_build(tmp_path):
    p = fake_pack(tmp_path)
    del p['topo_seed_base']
    with pytest.raises(KeyError):
        S._worker_entry(p)
    with pytest.raises(ValueError):
        S._worker_entry(fake_pack(tmp_path, seed_runs=[2 ** 31 - 1]))
    with pytest.raises(ValueError):
        S._worker_entry(fake_pack(tmp_path, seed_runs=[1, 2]))       # two seeds for one vector
    assert not (tmp_path / 'scratch').exists()                       # nothing built


def test_parent_draws_run_seeds_in_range():
    """seed_run = master_rng.integers(0, 2^31 - 1): the spec's range, checked on the sweep's draw."""
    src = open(S.__file__).read()
    assert 'integers(0, 2**31 - 1' in src.replace(' ', '').replace('integers(0,2**31-1', 'integers(0, 2**31 - 1') \
        or 'integers(0, SEED_MAX' in src
