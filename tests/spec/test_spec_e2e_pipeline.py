"""End to end through the real entry points (procedure step 6) and the compiled
oracles of Blocks 1, 11, 12 and 13.

- HPC_main_sweep.py (the pipeline's entry point) on a minimal Full and a
  minimal Neuronal case: it finishes; every output exists, is finite, and has
  the dtypes and keys the spec states; two runs with the same seed_master give
  identical results (Block 11); the network seeds follow the spec's formula.
- HPC_single_run.build_and_run (compiled), small Full networks:
  Block 1: V = V_R at every recorded step of (t_s, t_s + 3 ms] after each spike;
  Block 12: with g_sic = 0 the run is spike-for-spike the SIC-off twin;
  Block 11: same seed_run -> identical spikes across builds, another -> different;
  Block 13: the recorded C at t = 0 is the driven state of the construction.
Run (about 10-15 min on two cores):
    python -m pytest tests/spec/test_spec_e2e_pipeline.py -q
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.testing as npt
import pytest

from conftest import PIPE, run

sys.path.insert(0, PIPE)
import ASD_fun_BD_cpp as L      # noqa: E402
import HPC_main_sweep as S      # noqa: E402

RUNNER = os.path.join(os.path.dirname(__file__), '_pipeline_runner.py')


def spec_seeds(seed_device, network_key):
    out = {}
    for name, tag in (('seed_build', 1), ('seed_ic_neuron', 2), ('seed_ic_synapse', 3), ('seed_ic_astro', 4)):
        w = int(np.random.SeedSequence([seed_device, network_key, tag]).generate_state(1, dtype=np.uint32)[0])
        out[name] = 1 + w % (2 ** 31 - 2)
    return out


# ---------------------------------------------------------------------------
# the sweep, through its command line
# ---------------------------------------------------------------------------
FULL = ['--mode', 'Full', '--sweep_group', 'tripartite', '--Nn', '14', '--Na', '8', '--c_max', '120',
        '--conn_prob_lo', '0.2', '--conn_prob_hi', '0.4', '--stoa_gate_lo', '0.2', '--stoa_gate_hi', '1.0']
NEUR = ['--mode', 'Neuronal', '--sweep_group', 'neuron_synapse', '--Nn', '14', '--Na', '0', '--c_max', '120',
        '--conn_prob_lo', '0.2', '--conn_prob_hi', '0.4']
COMMON = ['--simtime', '0.3', '--n_topologies', '2', '--n_params_per_worker', '2', '--n_workers', '1',
          '--lib_dir', PIPE]


def sweep(out_dir, extra, seed):
    r = run([sys.executable, os.path.join(PIPE, 'HPC_main_sweep.py'), '--out_dir', out_dir,
             '--scratch_root', out_dir + '_scratch', '--seed_master', str(seed)] + extra + COMMON,
            cwd=PIPE, timeout=3000)
    return r


@pytest.fixture(scope='module')
def sweeps(tmp_path_factory):
    base = tmp_path_factory.mktemp('sweeps')
    jobs = {'full_a': (FULL, 4242), 'full_b': (FULL, 4242), 'neur': (NEUR, 777)}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {k: ex.submit(sweep, str(base / k), a, s) for k, (a, s) in jobs.items()}
        res = {k: f.result() for k, f in futs.items()}
    return {k: (str(base / k), r) for k, r in res.items()}


def runs_of(out_dir):
    return sorted(glob.glob(os.path.join(out_dir, 'topo_*', 'iter_*.npz')))


def test_sweep_finishes_and_writes_outputs(sweeps):
    for k, (d, r) in sweeps.items():
        assert r.returncode == 0, (k, r.stdout[-3000:], r.stderr[-3000:])
        assert os.path.isfile(os.path.join(d, 'manifest.json')) and os.path.isfile(os.path.join(d, 'job_args.json'))
        assert len(runs_of(d)) == 4, k                               # 2 topologies x 1 worker x 2 vectors
        assert not glob.glob(os.path.join(d, 'topo_*', '_failures.jsonl'))


def test_run_contract(sweeps):
    for k, (d, _) in sweeps.items():
        full = k.startswith('full')
        Nn, Na, T = 14, 8, 0.3
        for npz in runs_of(d):
            z = np.load(npz)
            side = json.load(open(npz[:-4] + '.json'))
            assert z['spk_N_t'].dtype == np.float32 and z['spk_N_i'].dtype == np.int32
            assert z['spk_N_t'].shape == z['spk_N_i'].shape
            assert np.all(np.isfinite(z['spk_N_t'])) and np.all((z['spk_N_t'] >= 0) & (z['spk_N_t'] <= T))
            assert np.all((z['spk_N_i'] >= 0) & (z['spk_N_i'] < Nn))
            th, par = z['theta'], z['params']
            assert th.shape == (38,) and par.shape == (38,) and np.all(np.isfinite(th))
            nat = th.copy()
            for j in S.LOG_PARAMS:
                nat[j] = np.exp(th[j])
            npt.assert_allclose(par, nat, rtol=1e-12)                    # params = theta_to_natural(theta)
            lo, hi = S.PARAM_BOUNDS_THETA[:, 0], S.PARAM_BOUNDS_THETA[:, 1]
            assert np.all(th >= lo - 1e-12) and np.all(th <= hi + 1e-12)
            # Block 11 records
            assert side['seed_verified'] is True and side['rng_scheme'] == 'v2-2026-11-04'
            assert side['noise_mode'] == 'fresh' and int(z['seed_run']) == side['seed_run']
            want = spec_seeds(50, side['topo_seed_base'])
            for key, v in want.items():
                assert side[key] == v and int(z[key]) == v, key
            if full:
                assert 0.2 <= side['stoa_gate_p'] <= 1.0 and float(z['stoa_gate_p']) == side['stoa_gate_p']
                assert side['sic_scheme'] == 'sic-v1-2026-11-04' and side['astro_ic'] == 'driven-v1-2026-11-04'
                topo = np.load(os.path.join(os.path.dirname(npz), 'topology.npz'))
                assert side['n_sic_pairs'] == len(topo['SIC_i']) > 0
                assert z['spk_A_t'].dtype == np.float32 and np.all((z['spk_A_i'] >= 0) & (z['spk_A_i'] < Na))
            else:
                for key in ('stoa_gate_p', 'sic_scheme', 'n_sic_pairs', 'astro_ic'):
                    assert key not in side and key not in z.files, key


def test_topology_meta(sweeps):
    for k, (d, _) in sweeps.items():
        for tdir in sorted(glob.glob(os.path.join(d, 'topo_*'))):
            meta = json.load(open(os.path.join(tdir, 'topology_meta.json')))
            topo = np.load(os.path.join(tdir, 'topology.npz'))
            assert meta['rng_scheme'] == 'v2-2026-11-04'
            tsb = meta['topo_seed_base']
            for key, v in spec_seeds(50, tsb).items():
                assert meta[key] == v, key
            if k.startswith('full'):
                assert meta['seed_stoa_gate'] == (tsb + 71) % (2 ** 31 - 1)
                assert meta['n_stoa_viable'] == len(topo['StoA_viable_i'])
                assert meta['n_sic_pairs'] == len(topo['SIC_i'])
            else:
                assert len(topo['SIC_i']) == 0 and len(topo['SIC_j']) == 0
                assert 'seed_stoa_gate' not in meta and 'n_stoa_viable' not in meta


def test_manifest_version(sweeps):
    m = json.load(open(os.path.join(sweeps['full_a'][0], 'manifest.json')))
    assert m['n_dims'] == 38
    assert m['manifest_version'] == 5                                 # SPEC Block 8


def test_same_seed_identical_runs(sweeps):
    a, b = runs_of(sweeps['full_a'][0]), runs_of(sweeps['full_b'][0])
    assert [os.path.relpath(x, sweeps['full_a'][0]) for x in a] == [os.path.relpath(x, sweeps['full_b'][0]) for x in b]
    for x, y in zip(a, b):
        za, zb = np.load(x), np.load(y)
        for key in ('spk_N_t', 'spk_N_i', 'spk_A_t', 'spk_A_i', 'theta', 'params', 'seed_run'):
            npt.assert_array_equal(za[key], zb[key], err_msg=key)
    seeds = [int(np.load(x)['seed_run']) for x in a]
    assert len(set(seeds)) == len(seeds)                              # a fresh run seed per run


def test_network_seeds_shared_within_and_differ_between_topologies(sweeps):
    d = sweeps['full_a'][0]
    builds = {}
    for npz in runs_of(d):
        side = json.load(open(npz[:-4] + '.json'))
        builds.setdefault(os.path.dirname(npz), set()).add(side['seed_build'])
    assert all(len(v) == 1 for v in builds.values()) and len({min(v) for v in builds.values()}) == len(builds)


# ---------------------------------------------------------------------------
# compiled single runs (build_and_run)
# ---------------------------------------------------------------------------
BASE = dict(pipe=PIPE, mode='Full', Nn=20, Na=12, c_max=150.0, conn_prob=0.3, simtime=0.6, seed_device=50,
            seeds=[39, 35, 60], stoa_gate_p=1.0,
            params=dict(Sigma=12.0, g_ampa=1.5, g_nmda=1.5))


def single(tmp, name, **over):
    cfg = json.loads(json.dumps(BASE))
    for k, v in over.items():
        if k == 'params':
            cfg['params'].update(v)
        else:
            cfg[k] = v
    path = os.path.join(tmp, name + '.json')
    json.dump(cfg, open(path, 'w'))
    out = os.path.join(tmp, name + '.npz')
    r = run([sys.executable, RUNNER, path, out], cwd=tmp, timeout=3000)
    return name, r, out


@pytest.fixture(scope='module')
def singles(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp('singles'))
    configs = {
        'rec': dict(seed_run=7, record=dict(neurons=list(range(20)), dt_ms=0.05, all_dt_ms=1.0)),
        'gsic0': dict(seed_run=7, params=dict(g_sic=0.0)),
        'sicoff': dict(seed_run=7, sic=False),
        'seed7': dict(seed_run=7),
        'seed7b': dict(seed_run=7),
        'seed8': dict(seed_run=8),
    }
    with ThreadPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(lambda kv: single(tmp, kv[0], **kv[1]), configs.items()))
    out = {}
    for name, r, path in res:
        assert r.returncode == 0, (name, r.stdout[-3000:], r.stderr[-3000:])
        out[name] = np.load(path)
    return out


def test_block1_V_held_at_VR_during_refractory(singles):
    z = singles['rec']
    V, t = z['rec_V'], z['rec_t_fine']                    # (neurons, steps) in mV, s
    idx = z['rec_idx_fine']
    VR = float(z['params'][S.PARAM_NAMES.index('VR')])
    dt = 5e-5
    n_steps_ref = int(round(3e-3 / dt))                    # t_ref = 3 ms = 60 steps
    assert np.all(np.isfinite(V))
    checked = 0
    for row, i in enumerate(idx):
        for ts in z['spk_N_t'][z['spk_N_i'] == i]:
            k = int(round(float(ts) / dt))                 # spike step; monitor reads at step starts
            seg = V[row, k + 1:k + 1 + n_steps_ref]
            if len(seg) < n_steps_ref:
                continue
            # exactly V_R for t in (t_s, t_s + t_ref]: the record is cast to float32 by
            # build_and_run's harvest, so compare with V_R rounded the same way (bitwise)
            npt.assert_array_equal(seg, np.float32(VR))
            checked += 1
    assert checked >= 20, checked                          # enough spikes for the check to bite


def test_block12_gsic0_equals_sic_off(singles):
    a, b = singles['gsic0'], singles['sicoff']
    assert len(a['spk_N_t']) > 0
    npt.assert_array_equal(a['spk_N_t'], b['spk_N_t'])
    npt.assert_array_equal(a['spk_N_i'], b['spk_N_i'])


def test_block11_run_seed(singles):
    a, b, c = singles['seed7'], singles['seed7b'], singles['seed8']
    npt.assert_array_equal(a['spk_N_t'], b['spk_N_t'])
    npt.assert_array_equal(a['spk_N_i'], b['spk_N_i'])
    assert not (len(a['spk_N_t']) == len(c['spk_N_t']) and np.array_equal(a['spk_N_t'], c['spk_N_t']))
    assert int(a['seed_run_applied']) == 7 and int(c['seed_run_applied']) == 8


def test_block13_compiled_start_is_driven_state(singles):
    z = singles['rec']
    C0 = z['rec_C'][:, 0]                                 # uM, first sample (t = 0)
    Na = 12
    linked = np.zeros(Na, dtype=bool)
    linked[np.unique(z['topo_StoA_j'])] = True
    seed_ic_astro = spec_seeds(50, 0)['seed_ic_astro']      # network key 0 in a single run (spec)
    phases = np.random.default_rng(np.random.SeedSequence([seed_ic_astro, 13])).random(Na)
    p = {n: float(z['params'][S.PARAM_NAMES.index(n)]) for n in L.ASTRO_IC_PARAMS}
    ics, _ = L.astro_driven_ics(p, linked, phases)
    # the astrocyte group is float32: 1e-6 relative is its round-off
    npt.assert_allclose(C0, ics['C'], rtol=1e-6)
