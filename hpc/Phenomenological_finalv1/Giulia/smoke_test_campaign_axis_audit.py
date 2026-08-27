#!/usr/bin/env python3
# =============================================================================
# smoke_test_campaign_axis_audit.py
#
# Self-contained correctness harness for campaign_axis_audit.py.
#
# It builds a SYNTHETIC campaign tree whose ground truth is known by
# construction, runs the auditor against it, and asserts every claim the
# auditor makes about that tree. Nothing outside a temporary directory is
# touched, and no cluster data is needed -- run it anywhere, including on the
# login node before pointing the auditor at the real campaigns.
#
# The fixture deliberately contains every awkward case the real cohort has:
#
#   campaign_test_v1  36-D registry (NO O_N), mode=Neuronal, sweep_group=all,
#                     conn_rule=flat, NO axis_declaration (pre-schema-v4).
#                     -> astrocyte axes are drawn but read by nothing, and the
#                        auditor must INFER that from mode alone.
#                     -> one declared-swept axis is written CONSTANT (must be
#                        reported as 'swept-but-const').
#                     -> one undeclared axis is written VARYING (must be
#                        reported as 'varies-undeclared').
#   campaign_test_v2  37-D registry (WITH O_N), mode=Full,
#                     sweep_group=synapse_astro, conn_rule=weibull, WITH a
#                     full axis_declaration. Astro axes swept AND consumed.
#   campaign_test_v3  same seed set as v1 (tests prefix containment) and a
#                     seed_master reused across two of its own tasks.
#
#   Seed ground truth:  v1 = {1000, 1001, 1002}
#                       v2 = {1002, 2000}          -> |v1 & v2| = 1
#                       v3 = {1000, 1001, 1002}    -> v3 == v1 (containment)
#                       v3 additionally repeats 1002 in two task dirs.
#
# One manifest is written with a large 'topologies' array to prove the
# head-only parse never reads it.
#
# USAGE
#   python3 smoke_test_campaign_axis_audit.py            # run everything
#   python3 smoke_test_campaign_axis_audit.py --keep     # keep the fixture
#   python3 smoke_test_campaign_axis_audit.py --fixture-only /tmp/fix
#
# Expected final line:  ALL 8 TEST GROUPS PASSED (<n> checks)
# =============================================================================

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import campaign_axis_audit as A                                  # noqa: E402


# =============================================================================
# Tiny assertion harness
# =============================================================================

class Checker(object):
    def __init__(self):
        self.n = 0
        self.failures = []
        self.group = None

    def start(self, name):
        self.group = name
        print('\n--- %s ---' % name)

    def check(self, cond, msg, detail=''):
        self.n += 1
        if cond:
            print('  [PASS] %s' % msg)
        else:
            print('  [FAIL] %s %s' % (msg, detail))
            self.failures.append('%s :: %s %s' % (self.group, msg, detail))
        return bool(cond)

    def eq(self, got, want, msg):
        return self.check(got == want, msg, '(got %r, want %r)' % (got, want))


# =============================================================================
# Registry definitions used by the fixture
# =============================================================================

NAMES_36 = [
    'Sigma', 'gbarA', 'EC50_ampa', 'EC50_nmda', 'tauA', 'U_0_ar', 'U_max',
    'U_0_sr', 'Omega_f_sr', 'Omega_f_ar', 'Omega_d', 'alpha_syn', 'DeltaT',
    'VT', 'g_ampa', 'g_nmda', 'delta_gA', 'x0', 'O_G', 'Omega_G', 'O_beta',
    'O_3K', 'Omega_5P', 'I_bias', 'F', 'I_Theta', 'omega_I', 'C_Theta',
    'U_A', 'G_T', 'gL', 'VA', 'DeltaA', 'VR', 'I_inj', 'Cm',
]
NAMES_37 = NAMES_36 + ['O_N']

ASTRO = ['O_beta', 'O_3K', 'Omega_5P', 'I_bias', 'F', 'I_Theta', 'omega_I',
         'C_Theta', 'U_A', 'G_T', 'O_N']

# v1: sweep_group 'all' under mode Neuronal -> neuron + synapse + astro drawn.
V1_SWEPT = [n for n in NAMES_36 if n not in ('DeltaT', 'VT', 'gL')]
# One of them is written constant on purpose:
V1_CONST_BUT_DECLARED = 'x0'
# One axis is NOT declared but is written varying on purpose:
V1_UNDECLARED_VARYING = 'VT'

# v2: synapse_astro-like group.
V2_SWEPT = ['Sigma', 'U_0_ar', 'U_max', 'U_0_sr', 'Omega_f_sr', 'Omega_f_ar',
            'Omega_d', 'alpha_syn', 'g_ampa', 'g_nmda', 'x0', 'O_G', 'Omega_G',
            'O_beta', 'O_3K', 'Omega_5P', 'I_bias', 'F', 'C_Theta', 'U_A',
            'G_T', 'O_N']

BOUNDS_36 = {n: (0.1, 1.0) for n in NAMES_36}
BOUNDS_37 = {n: (0.1, 1.0) for n in NAMES_37}
BOUNDS_37['Sigma'] = (2.0, 10.0)
BOUNDS_36['Sigma'] = (3.0, 15.0)          # deliberately different from v2


# =============================================================================
# Fixture construction
# =============================================================================

def _write_json(path, obj):
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=2)


def _make_manifest(names, bounds, sweep_group, swept, axis_decl,
                   n_fake_topologies=0):
    """Build a manifest dict with 'topologies' LAST, as rebuild_manifest does."""
    pb = [[bounds[n][0], bounds[n][1]] for n in names]
    m = {
        'manifest_version': 4 if axis_decl else 3,
        'n_dims': len(names),
        'updated_at': '2026-01-01T00:00:00',
        'job_id': '12345.pbsserver',
        'host': 'dvnode001',
        'n_topologies_completed_or_partial': 2,
        'n_total_runs': 8,
        'n_total_failures_discarded': 0,
        'param_names': list(names),
        'param_units': ['1'] * len(names),
        'param_bounds': pb,
        'sweep_group': sweep_group,
        'active_indices': sorted(names.index(n) for n in swept),
        'active_param_names': sorted(swept),
        'axis_declaration': axis_decl,
        'param_bounds_theta': pb,
        'log_transform': 'natural',
        'log_params': [],
        'log_param_indices': [],
        'theta_storage_keys': {'npz': 'theta', 'json': 'theta'},
    }
    # 'topologies' must be written LAST and can be enormous in reality.
    m['topologies'] = [{'topo_dir': 'topo_%05d' % k,
                        'iterations': [{'iter_idx': j, 'pad': 'x' * 64}
                                       for j in range(20)]}
                       for k in range(n_fake_topologies)]
    return m


def _axis_declaration(names, swept, bounds, mode, sweep_group, conn_rule,
                      inert_astro):
    swept_axes = {n: {'level': 'run_args', 'low': bounds[n][0],
                      'high': bounds[n][1], 'log': False} for n in swept}
    if conn_rule == 'flat':
        swept_axes['conn_prob'] = {'level': 'topology', 'low': 0.05,
                                   'high': 0.4, 'log': False}
        consumed_conn = ['conn_prob']
    else:
        for nm, (lo, hi) in (('p0_conn', (0.1, 1.0)), ('d0_conn', (60.0, 300.0)),
                             ('beta_conn', (1.0, 2.0))):
            swept_axes[nm] = {'level': 'topology', 'low': lo, 'high': hi,
                              'log': False}
        consumed_conn = ['p0_conn', 'd0_conn', 'beta_conn', 'conn_periodic']
    inert = {}
    if inert_astro:
        for n in swept:
            if n in ASTRO:
                inert[n] = 'swept but never read: mode=%s' % mode
    consumed = [n for n in swept if n not in inert] + consumed_conn
    fixed = {n: 0.5 for n in names if n not in swept}
    return {
        'schema_version': 1,
        'generated_by': 'fixture',
        'mode': mode,
        'sweep_group': sweep_group,
        'conn_rule': conn_rule,
        'n_swept': len(swept_axes),
        'n_consumed': len(consumed),
        'swept_axes': swept_axes,
        'consumed_axes': sorted(consumed),
        'fixed_axes': fixed,
        'inert_axes': inert,
    }


def _write_task(task_dir, names, bounds, swept, sweep_group, mode, conn_rule,
                conn_periodic, seed_master, c_max, Nn, Na, rng,
                axis_decl=None, n_topos=2, n_iters=4,
                const_axes=(), varying_undeclared=(), n_fake_topologies=0):
    os.makedirs(task_dir, exist_ok=True)

    job_args = {
        'out_dir': task_dir, 'lib_dir': '.', 'mode': mode,
        'sweep_group': sweep_group, 'conn_rule': conn_rule,
        'conn_periodic': bool(conn_periodic),
        'c_max': float(c_max), 'Nn': int(Nn), 'Na': int(Na),
        'density': None, 'density_astro': None,
        'simtime': 180.0, 'n_topologies': n_topos,
        'n_params_per_worker': 1, 'topology_mode': 'wallach',
        'displ_bias': 20.0, 'gj_dist': 200.0, 'gj_max_dist': 150.0,
        'stoa_cutoff': 70.0, 'stoa_sigma': 200.0,
        'conn_prob_lo': 0.05, 'conn_prob_hi': 0.4,
        '_resolved_seed_master': int(seed_master),
        '_resolved_n_workers': 48,
        '_sweep_group': sweep_group,
        '_active_indices': sorted(names.index(n) for n in swept),
        '_active_param_names': sorted(swept),
    }
    if axis_decl is not None:
        job_args['_axis_declaration'] = axis_decl
    _write_json(os.path.join(task_dir, 'job_args.json'), job_args)

    _write_json(os.path.join(task_dir, 'manifest.json'),
                _make_manifest(names, bounds, sweep_group, swept, axis_decl,
                               n_fake_topologies=n_fake_topologies))

    swept_set = set(swept) - set(const_axes)
    for t in range(n_topos):
        td = os.path.join(task_dir, 'topo_%05d' % t)
        os.makedirs(td, exist_ok=True)
        meta = {'topo_idx': t, 'conn_rule': conn_rule,
                'conn_periodic': bool(conn_periodic),
                'Nn': int(Nn), 'Na': int(Na), 'c_max': float(c_max),
                'topology_mode': 'wallach', 'n_synapses': 500}
        if conn_rule == 'flat':
            meta['conn_prob'] = float(rng.uniform(0.05, 0.4))
        else:
            meta['p0_conn'] = float(rng.uniform(0.1, 1.0))
            meta['d0_conn'] = float(rng.uniform(60.0, 300.0))
            meta['beta_conn'] = float(rng.uniform(1.0, 2.0))
        _write_json(os.path.join(td, 'topology_meta.json'), meta)

        for it in range(n_iters):
            params = np.full(len(names), 0.5, dtype=np.float64)
            for j, nm in enumerate(names):
                if nm in swept_set or nm in varying_undeclared:
                    lo, hi = bounds[nm]
                    params[j] = float(rng.uniform(lo, hi))
            extra = {}
            if conn_rule == 'flat':
                extra['conn_prob'] = np.float64(meta['conn_prob'])
            else:
                extra['p0_conn'] = np.float64(meta['p0_conn'])
                extra['d0_conn'] = np.float64(meta['d0_conn'])
                extra['beta_conn'] = np.float64(meta['beta_conn'])
            np.savez_compressed(
                os.path.join(td, 'iter_%05d.npz' % it),
                params=params, theta=params.copy(),
                conn_rule=np.array(conn_rule),
                conn_periodic=np.bool_(conn_periodic),
                topo_idx=np.int32(t), seed_run=np.int64(1234 + it),
                noise_mode=np.array('fresh'),
                spk_N_t=np.array([0.1, 0.2], dtype=np.float32),
                spk_N_i=np.array([0, 1], dtype=np.int32),
                spk_A_t=np.array([], dtype=np.float32),
                spk_A_i=np.array([], dtype=np.int32),
                **extra)
            _write_json(os.path.join(td, 'iter_%05d.json' % it),
                        {'topo_idx': t, 'iter_idx': it,
                         'param_names': list(names),
                         'params': params.tolist(),
                         'theta': params.tolist(),
                         'conn_rule': conn_rule})


def build_fixture(root, seed=7):
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)

    # ---- campaign_test_v1: 36-D, Neuronal, no axis_declaration -------------
    c1 = os.path.join(root, 'campaign_test_v1')
    for k, sm in enumerate([1000, 1001, 1002]):
        _write_task(os.path.join(c1, 'sweep_intel_task%04d' % k),
                    NAMES_36, BOUNDS_36, V1_SWEPT, 'all', 'Neuronal',
                    'flat', 0, sm, 240.0, 115, 115, rng,
                    axis_decl=None,
                    const_axes=(V1_CONST_BUT_DECLARED,),
                    varying_undeclared=(V1_UNDECLARED_VARYING,),
                    n_fake_topologies=(30 if k == 0 else 0))

    # ---- campaign_test_v2: 37-D, Full, with axis_declaration ---------------
    c2 = os.path.join(root, 'campaign_test_v2')
    decl2 = _axis_declaration(NAMES_37, V2_SWEPT, BOUNDS_37, 'Full',
                              'synapse_astro', 'weibull', inert_astro=False)
    for k, sm in enumerate([1002, 2000]):
        _write_task(os.path.join(c2, 'sweep_cpu_task%04d' % k),
                    NAMES_37, BOUNDS_37, V2_SWEPT, 'synapse_astro', 'Full',
                    'weibull', 0, sm, 300.0, 108, 108, rng, axis_decl=decl2)

    # ---- campaign_test_v3: seed set identical to v1, plus internal reuse ---
    c3 = os.path.join(root, 'campaign_test_v3')
    for k, sm in enumerate([1000, 1001, 1002, 1002]):
        _write_task(os.path.join(c3, 'sweep_intel_task%04d' % k),
                    NAMES_36, BOUNDS_36, V1_SWEPT, 'all', 'Neuronal',
                    'flat', 0, sm, 240.0, 115, 115, rng, axis_decl=None,
                    n_topos=1, n_iters=2)
    return root


def tree_digest(root):
    """SHA-256 over (relative path, size, bytes) of every file under root."""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            h.update(os.path.relpath(p, root).encode())
            with open(p, 'rb') as fh:
                h.update(fh.read())
    return h.hexdigest()


# =============================================================================
# Tests
# =============================================================================

def run_tests(root, out_dir, ck):
    argv = [root, '--glob', 'campaign_test_*', '--out', out_dir,
            '--iters-per-campaign', '200', '--topos-per-task', '5',
            '--iters-per-topo', '10', '--seed', '3']

    ck.start('GROUP 1 -- auditor runs end to end and is READ-ONLY')
    before = tree_digest(root)
    rc = A.main(argv)
    after = tree_digest(root)
    ck.eq(rc, 0, 'main() returns 0')
    ck.eq(after, before, 'campaign tree byte-identical after the audit')
    ck.check(os.path.exists(os.path.join(out_dir, 'campaign_axis_audit.md')),
             'markdown report written')
    ck.check(os.path.exists(os.path.join(out_dir, 'campaign_axis_audit.json')),
             'json report written')

    with open(os.path.join(out_dir, 'campaign_axis_audit.json')) as fh:
        payload = json.load(fh)
    recs = {r['campaign']: r for r in payload['campaigns']}

    ck.start('GROUP 2 -- discovery and registry width')
    ck.eq(sorted(recs), ['campaign_test_v1', 'campaign_test_v2',
                         'campaign_test_v3'], 'all three campaigns found')
    ck.eq(recs['campaign_test_v1']['n_task_dirs'], 3, 'v1 has 3 task dirs')
    ck.eq(recs['campaign_test_v2']['n_task_dirs'], 2, 'v2 has 2 task dirs')
    ck.eq(recs['campaign_test_v1']['registry']['n_dims'], 36, 'v1 is 36-D')
    ck.eq(recs['campaign_test_v2']['registry']['n_dims'], 37, 'v2 is 37-D')
    ck.eq(recs['campaign_test_v2']['empirical']['param_vector_widths'], [37],
          'v2 npz params vectors are 37 wide')
    ck.check('O_N' not in (recs['campaign_test_v1']['registry']['param_names']),
             'O_N absent from the v1 registry')
    ck.check('O_N' in (recs['campaign_test_v2']['registry']['param_names']),
             'O_N present in the v2 registry')

    ck.start('GROUP 3 -- swept-axis recovery, by name')
    st1 = recs['campaign_test_v1']['axis_status']
    st2 = recs['campaign_test_v2']['axis_status']
    got1 = sorted(n for n, s in st1.items() if s['status'].startswith('swept'))
    ck.eq(got1, sorted(V1_SWEPT), 'v1 swept set recovered exactly')
    got2 = sorted(n for n, s in st2.items() if s['status'].startswith('swept'))
    ck.eq(got2, sorted(V2_SWEPT), 'v2 swept set recovered exactly')
    ck.eq(st2['O_N']['status'], 'swept+consumed', 'O_N swept and consumed in v2')
    ck.eq(st2['EC50_ampa']['status'], 'fixed', 'EC50_ampa frozen in v2')
    ck.eq(st2['EC50_ampa']['fixed_value'], 0.5,
          'v2 frozen value read from axis_declaration')
    ck.eq(st2['Sigma']['declared_low'], 2.0, 'v2 Sigma lower bound')
    ck.eq(st2['Sigma']['declared_high'], 10.0, 'v2 Sigma upper bound')
    b1 = recs['campaign_test_v1']['registry']['param_bounds']
    n1 = recs['campaign_test_v1']['registry']['param_names']
    ck.eq(b1[n1.index('Sigma')], [3.0, 15.0],
          'v1 Sigma bounds differ from v2 (bounds change is visible)')

    ck.start('GROUP 4 -- astrocyte axes: swept vs consumed')
    astro_sw1 = sorted(n for n in ASTRO
                       if n in st1 and st1[n]['status'].startswith('swept'))
    ck.check(len(astro_sw1) == 10,
             'v1 draws the 10 astro axes of the 36-D registry',
             '(got %d: %s)' % (len(astro_sw1), astro_sw1))
    ck.check(all(st1[n]['status'] == 'swept+inert' for n in astro_sw1),
             'v1 astro axes are INERT (mode=Neuronal), inferred without an '
             'axis_declaration')
    astro_sw2 = sorted(n for n in ASTRO
                       if n in st2 and st2[n]['status'].startswith('swept'))
    ck.check(all(st2[n]['status'] == 'swept+consumed' for n in astro_sw2),
             'v2 astro axes are CONSUMED (mode=Full)')
    ck.check(all(st1[n]['is_astro_axis'] for n in ASTRO if n in st1),
             'astro membership flag set on every astro axis')

    ck.start('GROUP 5 -- empirical scan catches declaration/data mismatch')
    ck.eq(st1[V1_CONST_BUT_DECLARED]['status'], 'swept-but-const',
          'declared-swept but constant column detected (%s)'
          % V1_CONST_BUT_DECLARED)
    ck.eq(st1[V1_CONST_BUT_DECLARED]['n_distinct'], 1,
          'constant column has exactly 1 distinct value')
    ck.eq(st1[V1_UNDECLARED_VARYING]['status'], 'varies-undeclared',
          'undeclared but varying column detected (%s)'
          % V1_UNDECLARED_VARYING)
    ck.check(st1['Sigma']['n_distinct'] > 5,
             'a genuinely swept axis shows many distinct values')
    ck.check(recs['campaign_test_v1']['empirical']['n_failed'] == 0,
             'no npz read failures in v1')

    ck.start('GROUP 6 -- seed footprint and cross-campaign collisions')
    s1 = recs['campaign_test_v1']['seeds']
    s3 = recs['campaign_test_v3']['seeds']
    ck.eq(s1['values'], [1000, 1001, 1002], 'v1 seed set')
    ck.eq(recs['campaign_test_v2']['seeds']['values'], [1002, 2000],
          'v2 seed set')
    ck.eq(s3['n_tasks_with_seed'], 4, 'v3 has 4 tasks')
    ck.eq(s3['n_distinct'], 3, 'v3 reuses one seed internally')
    ck.check(any('REUSED within this campaign' in w
                 for w in recs['campaign_test_v3']['warnings']),
             'internal seed reuse raises a warning')
    pairs = {(p['a'], p['b']): p for p in payload['seed_overlap']['pairs']}
    p12 = pairs[('campaign_test_v1', 'campaign_test_v2')]
    ck.eq(p12['n_overlap'], 1, 'v1 vs v2 overlap is exactly one seed')
    ck.eq(p12['overlap_examples'], [1002], 'the colliding seed is 1002')
    p13 = pairs[('campaign_test_v1', 'campaign_test_v3')]
    ck.eq(p13['n_overlap'], 3, 'v1 vs v3 overlap is the whole set')
    ck.check(p13['prefix_containment'], 'v1 vs v3 flagged as containment')

    ck.start('GROUP 7 -- geometry and connectivity provenance')
    g1 = recs['campaign_test_v1']['geometry']
    g2 = recs['campaign_test_v2']['geometry']
    ck.eq(g1['side_um'], 240.0, 'v1 arena side')
    ck.check(abs(g1['area_mm2'] - 0.0576) < 1e-12, 'v1 arena area in mm^2')
    ck.check(abs(g1['rho_neuron_per_mm2'] - 115 / 0.0576) < 1e-6,
             'v1 neuron density back-computed from Nn and c_max')
    ck.eq(g2['side_um'], 300.0, 'v2 arena side')
    ck.check(abs(g2['rho_neuron_per_mm2'] - 108 / 0.09) < 1e-6,
             'v2 neuron density = 1200 /mm^2')
    conn1 = recs['campaign_test_v1']['empirical']['conn']
    conn2 = recs['campaign_test_v2']['empirical']['conn']
    ck.check('conn_prob' in conn1 and 'p0_conn' not in conn1,
             'flat campaign records conn_prob only')
    ck.check(('p0_conn' in conn2 and 'd0_conn' in conn2
              and 'beta_conn' in conn2 and 'conn_prob' not in conn2),
             'weibull campaign records the kernel triple only')
    ck.check(conn2['d0_conn']['min'] >= 60.0 and conn2['d0_conn']['max'] <= 300.0,
             'observed d0_conn inside the fixture kernel prior')

    ck.start('GROUP 8 -- helper units: manifest head parse and ast registry')
    big = os.path.join(root, 'campaign_test_v1', 'sweep_intel_task0000',
                       'manifest.json')
    size_mb = os.path.getsize(big) / 1e6
    obj, err = A.read_manifest_head(big, head_bytes=1 << 20, max_full_mb=1)
    ck.check(err is None, 'large manifest parsed head-only', str(err))
    ck.check(size_mb > 0.05, 'the test manifest is genuinely large (%.2f MB)'
             % size_mb)
    ck.check(obj is not None and obj.get('topologies') is None,
             'topologies array not loaded')
    ck.eq(obj['n_dims'], 36, 'head parse still recovers n_dims')
    ck.eq(len(obj['param_names']), 36, 'head parse still recovers param_names')

    src = os.path.join(root, '_fake_registry.py')
    with open(src, 'w') as fh:
        fh.write('import numpy as np\n'
                 'PARAM_NAMES = list(PARAM_NAMES)\n'
                 "PARAM_NAMES = ['a', 'b', 'c']\n"
                 'NOMINAL_PARAMS = np.array([1.0, 2.0, 3.0], dtype=float)\n'
                 'raise SystemExit("this module must never be executed")\n')
    ck.eq(A.literal_from_source(src, 'PARAM_NAMES'), ['a', 'b', 'c'],
          'ast picks the last real literal, not the shadowing assignment')
    ck.eq(A.literal_from_source(src, 'NOMINAL_PARAMS'), [1.0, 2.0, 3.0],
          'ast unwraps np.array([...])')
    ck.eq(A.literal_from_source(src, 'NOT_THERE'), None,
          'missing name returns None')
    os.remove(src)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--keep', action='store_true',
                    help='keep the temporary fixture tree and print its path')
    ap.add_argument('--fixture-only', default=None, metavar='DIR',
                    help='only build the fixture at DIR, run no tests')
    args = ap.parse_args()

    if args.fixture_only:
        build_fixture(args.fixture_only)
        print('fixture built at %s' % os.path.abspath(args.fixture_only))
        return 0

    tmp = tempfile.mkdtemp(prefix='axis_audit_smoke_')
    root = os.path.join(tmp, 'Giulia_Astro')
    out = os.path.join(tmp, 'audit_out')
    ck = Checker()
    try:
        build_fixture(root)
        run_tests(root, out, ck)
    finally:
        if args.keep:
            print('\nfixture kept at: %s' % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print('')
    if ck.failures:
        print('FAILURES (%d of %d checks):' % (len(ck.failures), ck.n))
        for f in ck.failures:
            print('  - %s' % f)
        print('SMOKE TEST FAILED')
        return 1
    print('ALL 8 TEST GROUPS PASSED (%d checks)' % ck.n)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
