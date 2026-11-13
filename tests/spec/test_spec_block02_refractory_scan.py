"""Block 2 -- refractory_scan, the spike-train screen (SPEC.md section 2, 3, Block 2).

Oracles: a reference implementation of section 2's formulas written here from
the spec; the output contract of Block 2; section 3's CSV columns and summary
keys; the Wilson 95 % interval from statsmodels. Run:
    python -m pytest tests/spec/test_spec_block02_refractory_scan.py -q
"""
import csv
import json
import math
import os

import numpy as np
import numpy.testing as npt
import pytest

from conftest import AUDIT, import_from

RS = import_from(AUDIT, 'refractory_scan')

SPEC_DEFAULT = dict(t_early=1.0, t_early_strict=0.1, late_frac=0.5, late_min_s=20.0,
                    late_active_frac=0.10, late_gap_s=1.0)


def ref_classify(spk_t, spk_i, Nn, T, cfg=None):
    """Section 2 of SPEC.md, written out neuron by neuron (no vectorisation)."""
    c = dict(SPEC_DEFAULT, **(cfg or {}))
    count = np.zeros(Nn, dtype=np.int64)
    last = np.full(Nn, -np.inf)
    for tk, ik in zip(np.asarray(spk_t, dtype=np.float64), np.asarray(spk_i)):
        ik = int(ik)
        if not np.isfinite(tk) or not (0 <= ik < Nn):
            continue
        count[ik] += 1
        last[ik] = max(last[ik], tk)
    spiked = count > 0
    evaluated = T >= 5 * c['t_early']
    early = np.array([evaluated and spiked[i] and last[i] <= c['t_early'] for i in range(Nn)], dtype=bool)
    early_strict = np.array([evaluated and spiked[i] and last[i] <= c['t_early_strict']
                             for i in range(Nn)], dtype=bool)
    tau = max(c['late_min_s'], c['late_frac'] * T)
    need = max(1, math.ceil(c['late_active_frac'] * (Nn - 1)))
    late = np.zeros(Nn, dtype=bool)
    for i in range(Nn):
        if spiked[i] and c['t_early'] < last[i] <= T - tau:
            n_after = sum(1 for k in range(Nn) if spiked[k] and last[k] > last[i] + c['late_gap_s'])
            late[i] = n_after >= need
    return dict(count=count, last=last, spiked=spiked, early=early, early_strict=early_strict,
                late=late, tau_late=tau, early_evaluated=evaluated)


def compare(got, want):
    for k in ('count', 'spiked', 'early', 'early_strict', 'late'):
        npt.assert_array_equal(np.asarray(got[k]), want[k], err_msg=k)
    npt.assert_array_equal(np.asarray(got['last']), want['last'])   # exact: max of the same floats
    assert got['tau_late'] == pytest.approx(want['tau_late'], rel=0, abs=0)
    assert bool(got['early_evaluated']) == want['early_evaluated']


def test_default_cfg_matches_spec():
    assert RS.DEFAULT_CFG == SPEC_DEFAULT


@pytest.mark.parametrize('seed', range(40))
def test_classify_run_equals_reference_on_random_runs(seed):
    rng = np.random.default_rng(1000 + seed)
    Nn = int(rng.integers(1, 40))
    T = float(rng.choice([2.0, 4.999, 5.0, 10.0, 45.0, 60.0, 180.0]))
    K = int(rng.integers(0, 400))
    t = rng.uniform(0, T, K)
    # last spikes clustered near the rule boundaries, plus non-finite times
    if K:
        t[rng.random(K) < 0.05] = 1.0
        t[rng.random(K) < 0.05] = 0.1
        t[rng.random(K) < 0.03] = np.nan
        t[rng.random(K) < 0.02] = np.inf
    i = rng.integers(-2, Nn + 2, K)                      # out-of-range indices included
    cfg = None if seed % 3 else dict(late_min_s=1.0, late_frac=0.1)   # make the late rule bite
    got = RS.classify_run(t.copy(), i.copy(), Nn, T, cfg)
    compare(got, ref_classify(t, i, Nn, T, cfg))


def test_output_contract_and_dtypes():
    t = np.array([0.5, 2.0, 30.0], dtype=np.float32)
    i = np.array([0, 1, 1], dtype=np.int32)
    out = RS.classify_run(t, i, 4, 60.0)
    for k in ('count', 'last', 'spiked', 'early', 'early_strict', 'late'):
        assert np.asarray(out[k]).shape == (4,), k
    assert np.issubdtype(np.asarray(out['count']).dtype, np.integer)
    assert np.issubdtype(np.asarray(out['last']).dtype, np.floating)
    for k in ('spiked', 'early', 'early_strict', 'late'):
        assert np.asarray(out[k]).dtype == bool, k
    assert out['last'][2] == -np.inf and out['last'][3] == -np.inf   # silent neurons
    assert out['tau_late'] == 30.0            # max(20 s, 0.5 x 60 s)
    assert bool(out['early_evaluated'])       # 60 s >= 5 x 1 s


def test_boundaries_closed_and_half_open():
    # early: last <= t_early (closed); T = 5 t_early evaluated, just below not
    t = np.array([1.0, 1.0 + 1e-9, 0.1], dtype=np.float64)
    i = np.array([0, 1, 2])
    out = RS.classify_run(t, i, 3, 5.0)
    assert list(out['early']) == [True, False, True]
    assert list(out['early_strict']) == [False, False, True]
    out2 = RS.classify_run(t, i, 3, 5.0 - 1e-9)
    assert not out2['early_evaluated'] and not out2['early'].any() and not out2['early_strict'].any()
    # late: t_early < last <= T - tau, tau = max(20, 0.5 T); T = 60 -> tau 30, edge 30.0 included
    t = np.array([30.0, 30.0 + 1e-9, 59.0, 59.0, 59.0])
    i = np.array([0, 1, 2, 3, 4])
    out = RS.classify_run(t, i, 5, 60.0)
    assert out['tau_late'] == 30.0
    assert list(out['late']) == [True, False, False, False, False]


def test_late_needs_enough_later_neurons():
    # Nn = 21: need = max(1, ceil(0.1 * 20)) = 2 neurons spiking > last_i + 1 s
    Nn, T = 21, 100.0
    t = np.array([10.0, 11.5, 50.0])           # neuron 0 dies at 10 s; 1 at 11.5 s, 2 at 50 s
    i = np.array([0, 1, 2])
    out = RS.classify_run(t, i, Nn, T)
    want = ref_classify(t, i, Nn, T)
    compare(out, want)
    assert out['late'][0] and not out['late'][2]        # 0: two later neurons; 2: last > T - tau
    t2 = np.array([10.0, 10.5, 50.0])          # neuron 1 only 0.5 s later: not counted (gap 1 s)
    out2 = RS.classify_run(t2, i, Nn, T)
    assert not out2['late'][0]


def test_inputs_not_mutated():
    t = np.array([0.5, np.nan, 3.0], dtype=np.float32)
    i = np.array([0, 1, 7], dtype=np.int32)
    t0, i0 = t.copy(), i.copy()
    RS.classify_run(t, i, 3, 10.0)
    npt.assert_array_equal(t, t0)
    npt.assert_array_equal(i, i0)


# ---------------------------------------------------------------------------
# scan_campaign on a synthetic campaign (section 3 layout)
# ---------------------------------------------------------------------------
PARAM_NAMES = ['a', 'b', 'c']


def make_campaign(root, runs, T=10.0, Nn=6, lib_dir='/x'):
    """runs: list of (task, topo, iter, spk_t, spk_i, params, conn_prob)."""
    camp = os.path.join(root, 'campaign_synthetic')
    for task, topo, it, t, i, par, cp in runs:
        td = os.path.join(camp, 'sweep_cpu_task%04d' % task)
        os.makedirs(os.path.join(td, 'topo_%05d' % topo), exist_ok=True)
        with open(os.path.join(td, 'job_args.json'), 'w') as fh:
            json.dump(dict(simtime=T, Nn=Nn, lib_dir=lib_dir), fh)
        base = os.path.join(td, 'topo_%05d' % topo, 'iter_%05d' % it)
        np.savez(base + '.npz', spk_N_t=np.asarray(t, dtype=np.float32),
                 spk_N_i=np.asarray(i, dtype=np.int32), params=np.asarray(par, dtype=np.float64),
                 conn_prob=np.float64(cp))
        with open(base + '.json', 'w') as fh:
            json.dump(dict(param_names=PARAM_NAMES), fh)
    return camp


def synthetic_runs(rng, n):
    runs = []
    for r in range(n):
        Nn = 6
        K = int(rng.integers(0, 60))
        t = rng.uniform(0, 10.0, K)
        i = rng.integers(0, Nn, K)
        if r % 3 == 0 and K:                 # make neuron 0 die early
            keep = ~((i == 0) & (t > 0.5))
            t, i = t[keep], i[keep]
        runs.append((r // 4, r % 4, 0, t, i, rng.uniform(0, 1, 3), float(rng.uniform(0.01, 0.5))))
    return runs


def wilson_statsmodels(k, n):
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=0.05, method='wilson')
    return 100 * lo, 100 * hi


def test_scan_campaign_tables(tmp_path):
    rng = np.random.default_rng(7)
    runs = synthetic_runs(rng, 24)
    camp = make_campaign(str(tmp_path), runs)
    out = str(tmp_path / 'out')
    RS.scan_campaign(camp, out, sample=0, seed=1, workers=1, cfg=dict(SPEC_DEFAULT))
    name = os.path.basename(camp)
    with open(os.path.join(out, name + '_runs.csv')) as fh:
        rows = list(csv.DictReader(fh))
        header = rows and list(rows[0].keys())
    spec_cols = ['task', 'topo', 'iter', 'status', 'T', 'Nn', 'nn_source', 'n_spikes', 'n_spiking',
                 'n_early', 'n_early_strict', 'n_late', 'n_suspect', 'frac_suspect',
                 'net_silent_early', 't_last_any', 'conn_prob']
    assert header[:len(spec_cols)] == spec_cols
    assert sorted(header[len(spec_cols):]) == sorted('par_' + p for p in PARAM_NAMES)
    assert len(rows) == len(runs)
    by_key = {('sweep_cpu_task%04d' % r[0], 'topo_%05d' % r[1], 'iter_%05d' % r[2]): r for r in runs}
    n_early_runs = 0
    for row in rows:
        r = by_key[(row['task'], row['topo'], row['iter'])]
        want = ref_classify(np.asarray(r[3], dtype=np.float32), r[4], 6, 10.0)
        assert int(row['n_early']) == int(want['early'].sum())
        assert int(row['n_early_strict']) == int(want['early_strict'].sum())
        assert int(row['n_late']) == int(want['late'].sum())
        assert int(row['n_spiking']) == int(want['spiked'].sum())
        assert int(row['n_spikes']) == int(want['count'].sum())
        assert float(row['T']) == 10.0 and int(row['Nn']) == 6
        assert float(row['conn_prob']) == pytest.approx(r[6], rel=1e-12)
        for p, v in zip(PARAM_NAMES, r[5]):
            assert float(row['par_' + p]) == pytest.approx(v, rel=1e-12)
        n_spiking = int(want['spiked'].sum())
        silent_early = n_spiking > 0 and int(want['early'].sum()) == n_spiking   # "every spiking neuron is early"
        assert (row['net_silent_early'] == 'True') == silent_early
        n_early_runs += int(want['early'].sum()) > 0
    with open(os.path.join(out, name + '_summary.json')) as fh:
        summ = json.load(fh)
    for k in ('runs_early', 'pct_runs_early', 'ci95_early', 'runs_any_suspect', 'runs_late',
              'pct_runs_net_silent_early', 'pct_neurons_early'):
        assert k in summ, k
    assert summ['runs_early'] == n_early_runs
    assert summ['pct_runs_early'] == pytest.approx(100.0 * n_early_runs / len(runs), rel=1e-12)
    lo, hi = wilson_statsmodels(n_early_runs, len(runs))
    # tolerance: the code may round z to 1.96 (vs 1.959964); that moves the bounds by < 1e-3 %-points here
    npt.assert_allclose(summ['ci95_early'], [lo, hi], atol=1e-3)


def test_sample_reproducible_under_seed(tmp_path):
    rng = np.random.default_rng(11)
    camp = make_campaign(str(tmp_path), synthetic_runs(rng, 40))
    picks = []
    for k, seed in enumerate((1, 1, 2)):
        out = str(tmp_path / ('o%d' % k))
        RS.scan_campaign(camp, out, sample=10, seed=seed, workers=1, cfg=dict(SPEC_DEFAULT))
        with open(os.path.join(out, 'campaign_synthetic_runs.csv')) as fh:
            picks.append(sorted((r['task'], r['topo']) for r in csv.DictReader(fh)))
    assert len(picks[0]) == 10
    assert picks[0] == picks[1]          # same seed, same sample
    assert picks[0] != picks[2]          # another seed, another sample (fails with prob. ~1e-9)
