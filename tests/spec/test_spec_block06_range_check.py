"""Block 6 -- range_check, the standard prior range calibration (SPEC.md Block 6).

Oracles: the full factorial's bit rule; the defining relation and a Gram
matrix for the fraction (resolution >= 5: main effects and two-axis
interactions mutually orthogonal); the sizes the spec states for
LindaColloDUP and GiuliaGate; every corner vector at the bounds its codes
name and every interior value a bin midpoint; the analysis' formulas on
planted spike trains, with SciPy's Spearman. Run:
    python -m pytest tests/spec/test_spec_block06_range_check.py -q
"""
import csv
import itertools
import json
import math
import os
import sys

import numpy as np
import numpy.testing as npt
import pytest
from scipy import stats

from conftest import RANGE, import_from, run

RD = import_from(RANGE, 'range_design')
PROBE = os.path.join(os.path.dirname(__file__), '_rc_design_probe.py')
SEED = 20261030                                  # --rc-seed default (spec)


def popcount(x):
    return bin(int(x)).count('1')


# ---------------------------------------------------------------------------
# designs (pure functions)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('d', [1, 2, 3, 5, 8])
def test_full_factorial_bit_rule(d):
    X = np.asarray(RD.full_factorial(d))
    assert X.shape == (2 ** d, d)
    for r in range(2 ** d):
        for k in range(d):
            assert X[r, k] == (1 if ((r >> k) & 1) == 0 else -1)


def low_order_gram(X):
    """Gram matrix / n of the main-effect and two-axis-interaction columns."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    cols = [X[:, i] for i in range(d)] + [X[:, i] * X[:, j] for i, j in itertools.combinations(range(d), 2)]
    M = np.array(cols)
    return M @ M.T / n


@pytest.mark.parametrize('d', [15, 17, 24])
def test_fraction_resolution_at_least_5(d):
    X, info = RD.two_level_design(d, 16384, np.random.default_rng(SEED + 20000), 8)
    X = np.asarray(X)
    b = int(math.floor(math.log2(16384)))
    assert info['kind'] == 'fraction' and X.shape == (2 ** b, d) and info['n_runs'] == 2 ** b
    assert info['resolution'] >= 5
    # base columns = full factorial of b; each generated column = product of its base columns
    npt.assert_array_equal(X[:, :b], np.asarray(RD.full_factorial(b)))
    masks = []
    for i, gen in enumerate(info['generators']):
        npt.assert_array_equal(X[:, b + i], np.prod(X[:, gen], axis=1))
        masks.append(sum(1 << k for k in gen))
    # defining relation, recomputed here: every word |G| + popcount(xor of masks) >= resolution
    shortest = min(s + popcount(np.bitwise_xor.reduce(G))
                   for s in range(1, len(masks) + 1) for G in itertools.combinations(masks, s))
    assert shortest >= info['resolution'] >= 5
    # Gram: balanced columns, main effects and two-axis interactions mutually orthogonal
    assert np.all(X.sum(axis=0) == 0)
    G = low_order_gram(X)
    npt.assert_allclose(G, np.eye(len(G)), atol=0)   # exact: sums of +-1 over 2^b rows


def test_interior_helpers():
    mids = RD.bin_midpoints(-2.0, 5.0, 7)
    npt.assert_allclose(mids, -2.0 + (np.arange(7) + 0.5) * 1.0, rtol=0, atol=1e-15)
    assert RD.bin_of(mids[3], -2.0, 5.0, 7) == 3
    assert RD.bin_of(mids[3] + 0.5e-9 * 7, -2.0, 5.0, 7) == 3         # within 1e-9 of the width
    assert RD.bin_of(mids[3] + 2e-9 * 7, -2.0, 5.0, 7) == -1
    assert RD.corner_code(-2.0, -2.0, 5.0) == -1 and RD.corner_code(5.0, -2.0, 5.0) == 1
    assert RD.corner_code(1.5, -2.0, 5.0) == 0 and RD.corner_code(1.0, -2.0, 5.0) is None


# ---------------------------------------------------------------------------
# the two setups' designs, as range_check.py builds them (no simulation)
# ---------------------------------------------------------------------------
LCDUP_SETTINGS = ['--mode', 'Neuronal', '--sweep_group', 'neuron_synapse',
                  '--freeze', 'Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G',
                  '--density', '1300', '--density_astro', '0', '--c_max', '350', '--Nn', '115', '--Na', '0',
                  '--conn_rule', 'flat', '--conn_prob_lo', '0.01', '--conn_prob_hi', '0.5',
                  '--n_workers', '48']


def gg_settings(gg_folder):
    fl = run([sys.executable, 'apply_prior_patch.py', '--freeze-list'], cwd=gg_folder).stdout.strip()
    return ['--mode', 'Full', '--sweep_group', 'tripartite', '--freeze', fl,
            '--density', '1200', '--density_astro', '1200', '--c_max', '300', '--Nn', '115', '--Na', '115',
            '--conn_rule', 'flat', '--conn_prob_lo', '0.05', '--conn_prob_hi', '0.4',
            '--stoa_gate_lo', '0.2', '--stoa_gate_hi', '1.0', '--seed_stoa_gate', '71', '--n_workers', '48']


def probe(lib_dir, phase, level, settings, tmp_path, tag=''):
    out = str(tmp_path / ('%s_%s%d%s.json' % (os.path.basename(lib_dir), phase, level, tag)))
    r = run([sys.executable, PROBE, RANGE, lib_dir, phase, str(level), out, '--'] + settings, cwd=lib_dir)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.load(open(out))


def check_corner_level(p, level, n_level_corners):
    d = p['design']
    free = [p['param_names'].index(n) for n in d['free_axes']]
    lo, hi = np.array(d['theta_lo']), np.array(d['theta_hi'])
    codes = np.array(d['codes'])
    padded = np.array(p['padded'])
    nominal = np.array(d['nominal_theta'])
    assert d['n_design'] == n_level_corners == len(codes)
    k = math.ceil(n_level_corners / (8 * 48))
    assert d['k'] == k and d['n_topologies'] == 8 and len(padded) == 8 * 48 * k
    assert d['n_padding'] == 8 * 48 * k - n_level_corners
    others = [j for j in range(len(nominal)) if j not in free]
    for r in range(len(padded)):
        th = padded[r]
        want = np.where(codes[r] > 0, hi, lo) if r < len(codes) else 0.5 * (lo + hi)
        npt.assert_array_equal(th[free], want)
        npt.assert_array_equal(th[others], nominal[others])
    assert p['conn'] == d['conn_box'][level]
    return free, codes


def test_lcdup_corners_full_factorial(lcdup_folder, tmp_path):
    rows = []
    for level in (0, 1):
        p = probe(lcdup_folder, 'corners', level, LCDUP_SETTINGS, tmp_path)
        d = p['design']
        assert len(d['free_axes']) == 13 and d['has_conn']
        assert d['corner_design']['kind'] == 'full' and d['corner_design']['d'] == 14
        free, codes = check_corner_level(p, level, 8192)
        assert d['n_padding'] == 256 and d['k'] == 22
        names = p['param_names']
        nominal_nat = np.array(p['nominal'])
        # the frozen NAME=VALUE axes are held at the written values in every vector
        th = np.array(p['padded'])
        assert np.all(th[:, names.index('VA')] == -48.0) and np.all(th[:, names.index('VR')] == -58.2)
        rows += [tuple(c) + (2 * level - 1,) for c in codes]
    assert len(rows) == 2 ** 14 and len(set(rows)) == 2 ** 14      # every corner exactly once


def test_lcdup_interior_levels(lcdup_folder, tmp_path):
    for level in (0, 3, 6):
        p = probe(lcdup_folder, 'interior', level, LCDUP_SETTINGS, tmp_path)
        d = p['design']
        assert d['n_design'] == 2352 == 7 * 48 * 7 and d['k'] == 7 and d['n_topologies'] == 7
        assert p['conn'] == pytest.approx(0.045 + 0.07 * level, abs=1e-12)
        free = [p['param_names'].index(n) for n in d['free_axes']]
        lo, hi = np.array(d['theta_lo']), np.array(d['theta_hi'])
        th = np.array(p['padded'])[:, free]
        mids = lo[None, :] + (np.arange(7)[:, None] + 0.5) * (hi - lo)[None, :] / 7   # (bins, axes)
        cells = np.array(d['cells'])
        npt.assert_allclose(th, mids[cells, np.arange(len(free))[None, :]], rtol=0, atol=1e-12)
        for j in range(len(free)):                  # uniform bins, chi-square (deterministic seeds)
            counts = np.bincount(cells[:, j], minlength=7)
            assert stats.chisquare(counts).pvalue > 1e-4, (d['free_axes'][j], counts)


def test_interior_reproducible(lcdup_folder, tmp_path):
    a = probe(lcdup_folder, 'interior', 2, LCDUP_SETTINGS, tmp_path, tag='a')
    b = probe(lcdup_folder, 'interior', 2, LCDUP_SETTINGS, tmp_path, tag='b')
    assert a['design']['cells'] == b['design']['cells'] and a['padded'] == b['padded']


def test_gg_corners_fraction(gg_folder, tmp_path):
    settings = gg_settings(gg_folder)
    allrows = []
    for level in (0, 1):
        p = probe(gg_folder, 'corners', level, settings, tmp_path)
        d = p['design']
        names = d['free_axes']
        assert len(names) == 16 and d['corner_design']['d'] == 17
        assert d['corner_design']['kind'] == 'fraction' and d['corner_design']['n_runs'] == 16384
        assert d['corner_design']['resolution'] >= 5
        free, codes = check_corner_level(p, level, 8192)
        assert d['n_padding'] == 256
        allrows.append(np.column_stack([codes, np.full(len(codes), 2 * level - 1)]))
        lp = set(p['log_params'])
        idx = {n: p['param_names'].index(n) for n in names}
        assert idx['Sigma'] not in lp and idx['g_ampa'] in lp and idx['g_nmda'] in lp and idx['g_sic'] in lp
        bt = p['bounds']
        assert bt[idx['Sigma']] == [2.0, 10.0] and bt[idx['g_ampa']] == [0.05, 10.0]
        assert bt[idx['g_nmda']] == [0.05, 10.0] and bt[idx['g_sic']] == [4.0, 400.0]
    X = np.vstack(allrows)
    assert len({tuple(r) for r in X}) == 16384
    npt.assert_allclose(low_order_gram(X), np.eye(17 + 136), atol=0)


def test_gg_interior_conn_levels(gg_folder, tmp_path):
    settings = gg_settings(gg_folder)
    for level in (0, 6):
        p = probe(gg_folder, 'interior', level, settings, tmp_path)
        assert p['design']['n_design'] == 2352
        assert p['conn'] == pytest.approx(0.075 + 0.05 * level, abs=1e-12)


# ---------------------------------------------------------------------------
# analysis on planted runs
# ---------------------------------------------------------------------------
def regular_train(rate, T, t_skip, n_neurons, extra_before=True):
    """Spike times with exactly rate * (T - t_skip) spikes per neuron in [t_skip, T)."""
    ts, ids = [], []
    n = int(round(rate * (T - t_skip)))
    for i in range(n_neurons):
        t = t_skip + (np.arange(n) + 0.5 + 0.01 * i) / rate
        ts.append(t)
        ids.append(np.full(n, i))
        if extra_before:
            ts.append(np.array([0.5, 1.0]))          # outside the window: must not count
            ids.append(np.array([i, i]))
    return np.concatenate(ts).astype(np.float32), np.concatenate(ids).astype(np.int32)


def make_level(root, level, lo, hi, free_names, param_names, rate_fn, full=False, Na=4,
               astro_rate_fn=None, offgrid=False):
    pdir = os.path.join(root, 'corners_c%d' % level)
    os.makedirs(pdir)
    T, t_skip, Nn = 20.0, 2.0, 10
    n_dims = len(free_names) + 1
    points_all = [c for c in itertools.product((-1, 1), repeat=len(free_names))]
    # design.json fields as SPEC.md Block 6 lists them for a corners level folder
    design = dict(phase='corners', level=level, free_axes=free_names, theta_lo=lo, theta_hi=hi,
                  has_conn=True, conn_box=[0.1, 0.3], conn_prob=[0.1, 0.3][level], bins=7,
                  nominal_theta=[0.0] * len(param_names), n_workers=1, n_topologies=len(points_all) + 1,
                  k=1, n_slots=len(points_all) + 1, n_design=len(points_all), n_padding=1,
                  corner_design=dict(kind='full', n_runs=2 ** n_dims, resolution=None, base=n_dims,
                                     generators=[], d=n_dims, n_levels=2, n_level_corners=len(points_all)),
                  codes=[list(c) for c in points_all], centre_theta=[0.5 * (a + b) for a, b in zip(lo, hi)],
                  lib_dir=os.path.join(RANGE, '..', 'Phenomenological_finalv1'))
    json.dump(design, open(os.path.join(pdir, 'design.json'), 'w'))
    json.dump(dict(Nn=Nn, Na=Na if full else 0, mode='Full' if full else 'Neuronal', simtime=T),
              open(os.path.join(pdir, 'job_args.json'), 'w'))
    points = [c for c in itertools.product((-1, 1), repeat=len(free_names))] + [(0,) * len(free_names)]
    for k, codes in enumerate(points):
        theta = np.zeros(len(param_names))
        for name, c, a, b in zip(free_names, codes, lo, hi):
            theta[param_names.index(name)] = {-1: a, 1: b, 0: 0.5 * (a + b)}[c]
        if offgrid and k == 0:
            theta[param_names.index(free_names[0])] += 0.3
        topo = os.path.join(pdir, 'topo_%05d' % k)
        os.makedirs(topo)
        t, i = regular_train(rate_fn(codes, level), T, t_skip, Nn)
        arrays = dict(spk_N_t=t, spk_N_i=i)
        if full:
            at, ai = regular_train(astro_rate_fn(codes, level), T, t_skip, Na)
            arrays.update(spk_A_t=at, spk_A_i=ai)
        np.savez(os.path.join(topo, 'iter_00000.npz'), **arrays)
        json.dump(dict(topo_idx=k, iter_idx=0, conn_prob=design['conn_prob'], theta=theta.tolist(),
                       params=theta.tolist(), param_names=param_names,
                       npz_relpath=os.path.join('topo_%05d' % k, 'iter_00000.npz'),
                       stoa_gate_p=0.5 + 0.1 * k if full else None),
                  open(os.path.join(topo, 'iter_00000.json'), 'w'))
    return pdir


def run_analysis(root):
    return run([sys.executable, os.path.join(RANGE, 'range_analysis.py'), root, '--workers', '1'], cwd=root)


def test_analysis_recovers_planted_effects(tmp_path):
    root = str(tmp_path / 'rc')
    names = ['A', 'B', 'Z']
    rate = lambda codes, level: 1.0 if codes[0] < 0 else (4.0 if codes[0] > 0 else 2.0)  # A only
    for level in (0, 1):
        make_level(root, level, [0.0, -1.0], [1.0, 1.0], ['A', 'B'], names, rate)
    r = run_analysis(root)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    runs = list(csv.DictReader(open(os.path.join(root, 'runs.csv'))))
    assert sorted({x['kind'] for x in runs}) == ['centre', 'corner']
    for x in runs:
        want = 1.0 if x['code_A'] in ('-1', '-1.0') else (4.0 if x['code_A'] in ('1', '1.0') else 2.0)
        assert float(x['mean_rate_hz']) == pytest.approx(want, rel=1e-9)   # spikes / (Nn (T - t_skip))
        assert float(x['frac_active']) == 1.0
    eff = {e['dimension']: e for e in csv.DictReader(open(os.path.join(root, 'corner_effects.csv')))}
    want_A = math.log10(4.0 + 0.01) - math.log10(1.0 + 0.01)     # y = log10(rate + 0.01 Hz)
    assert float(eff['A']['effect_y']) == pytest.approx(want_A, abs=1e-9)
    assert abs(float(eff['B']['effect_y'])) < 1e-9
    assert abs(float(eff['conn_prob']['effect_y'])) < 1e-9
    assert int(eff['A']['n_plus']) == 4 and int(eff['A']['n_minus']) == 4   # 8 corner runs, centres excluded
    inter = list(csv.DictReader(open(os.path.join(root, 'corner_interactions.csv'))))
    assert all(abs(float(x['effect_y'])) < 1e-9 for x in inter)


def test_analysis_full_mode_astro_outcomes(tmp_path):
    root = str(tmp_path / 'rc')
    names = ['A', 'B']
    for level in (0, 1):
        make_level(root, level, [0.0, 0.0], [1.0, 1.0], names, names, lambda c, l: 2.0, full=True, Na=4,
                   astro_rate_fn=lambda c, l: 0.05 if c[1] < 0 else (0.4 if c[1] > 0 else 0.2))
    r = run_analysis(root)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    eff = {e['dimension']: e for e in csv.DictReader(open(os.path.join(root, 'corner_effects.csv')))}
    # planted astro rates: 0.05 Hz at B = -1 (0.9 events per astrocyte in 18 s -> rounded to 1 event,
    # i.e. 1/18 Hz), 0.4 Hz at B = +1 (7.2 -> 7 events, 7/18 Hz); y_astro = log10(rate + 0.001 Hz)
    lo_rate, hi_rate = round(0.05 * 18) / 18.0, round(0.4 * 18) / 18.0
    want_B = math.log10(hi_rate + 0.001) - math.log10(lo_rate + 0.001)
    assert float(eff['B']['effect_y_astro']) == pytest.approx(want_B, abs=1e-9)
    assert abs(float(eff['A']['effect_y_astro'])) < 1e-9
    assert os.path.isfile(os.path.join(root, 'gate_networks.csv'))


def test_analysis_offgrid_run_is_unmatched_exit_1(tmp_path):
    root = str(tmp_path / 'rc')
    for level in (0, 1):
        make_level(root, level, [0.0, 0.0], [1.0, 1.0], ['A', 'B'], ['A', 'B'], lambda c, l: 2.0,
                   offgrid=(level == 0))
    r = run_analysis(root)
    assert r.returncode == 1
    runs = list(csv.DictReader(open(os.path.join(root, 'runs.csv'))))
    assert sum(x['kind'] == 'unmatched' for x in runs) == 1
