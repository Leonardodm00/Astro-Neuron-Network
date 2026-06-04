#!/usr/bin/env python3
# =============================================================================
# HPC_main_sweep.py
#
# Nested random-search driver:  (random topology)  ×  (random parameters).
#
# ARCHITECTURE
# ------------
# Outer loop (in the parent process, pure numpy):
#   For each topology iteration k:
#     • draw conn_prob_k ~ U(--conn_prob_lo, --conn_prob_hi)
#     • derive topology seeds from the master RNG
#     • build the topology with HPC_single_run.build_topology   (~1–5 s)
#     • save topology.npz + spatial_layout.png + topology_meta.json
#     • sample (n_workers × n_params_per_worker) parameter vectors
#     • split them across n_workers chunks
#     • spawn a multiprocessing.Pool (spawn context, NOT fork) and dispatch
#       one chunk per worker
#     • after the pool joins, rebuild manifest.json from per-iter JSONs
#
# Inner (each worker, separate Python process via spawn):
#   • set_device('cpp_standalone', build_on_run=False, directory=worker_dir)
#     device.reinit(); device.activate(build_on_run=False)
#   • build the network from the *shared* topology arrays
#   • compile ONCE
#   • for each parameter vector in the worker's chunk:
#         device.run(run_args={...params...}, seed=fresh_seed_run)
#         harvest spk_N_t / spk_N_i / spk_A_t / spk_A_i
#         atomic write of iter_<NN>.npz   (spike data + params + conn_prob)
#         atomic write of iter_<NN>.json  (metadata, used for manifest rebuild)
#
# WALLTIME ROBUSTNESS
# -------------------
# Each completed (topology, param) pair is fully on disk *before* the next
# one starts.  If the PBS scheduler kills the job mid-sweep, every iter file
# that exists in <out_dir>/topo_*/iter_*.npz is a valid, replayable result.
# A consolidating manifest rebuild can be triggered post-hoc with
#     python HPC_main_sweep.py --rebuild_manifest_only --out_dir <DIR>
#
# DEPENDENCIES
# ------------
# • HPC_single_run.py     (build_topology, save_topology, plot_spatial_layout,
#                          _resolve_syn_pdist_csv) — one source of truth for
#                          the geometry, so this script never duplicates the
#                          topology code.
# • ASD_fun_BD_cpp.py     (Neuronal_Network, Astrocyte_Group, Gliotransmission,
#                          Synapse_to_astro, Astro_to_Syn)
# • synapse_pdist.csv     (Sholl chi²(4) PDF — auto-discovered in --lib_dir)
#
# USAGE
# -----
#   python HPC_main_sweep.py --out_dir /scratch/<user>/sweep_$JOBID \
#                            --lib_dir . \
#                            --n_workers $PBS_NCPUS \
#                            --n_topologies 100 \
#                            --n_params_per_worker 1 \
#                            --conn_prob_lo 0.1 --conn_prob_hi 0.6 \
#                            --simtime 180 --mode Full
#
# See submit_main_sweep.sh for the PBS wrapper.
# =============================================================================

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                   # non-interactive backend
import numpy as np


# =============================================================================
# Reuse the pass-1 topology builder from HPC_single_run.py
# (kept as one source of truth; never copy-pasted)
# =============================================================================
# We add the script's own directory to sys.path so the import works even when
# the job's working directory has been changed by the PBS prologue.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from HPC_single_run import (                                         # noqa: E402
    build_topology,
    save_topology,
    plot_spatial_layout,
    _resolve_syn_pdist_csv,
    PARAM_NAMES, PARAM_UNITS,
)


# =============================================================================
# Parameter sampling — 30-D box (expanded from the original 14-D)
# =============================================================================

# (low, high) for each of the 30 swept parameters, in the same order as
# PARAM_NAMES.
PARAM_BOUNDS = np.array([
    (1.0,   10.0),      # Sigma         [mV]   (widened 2-6 -> 1-10: 1 mV = near-quiescent
                        #                       control spontaneous rate; 10 mV = high-noise
                        #                       up-state-prone pathology. NB now spans 1 decade
                        #                       -> flips to LOG under the >=1-decade rule.)
    (0.5,   20.0),      # g_AHP         [nS]   (floor 0.5 reaches severely-reduced-adaptation /
                        #                        seizure-prone regime, pair w/ low alpha_Ca,
                        #                        short Tau_Ca; ceiling raised 15->20 to reach
                        #                        the strong-adaptation / regular-low-rate end)
    (5.0,   70.0),      # EC50_ampa     [mmole]   (brackets B_tot 0.2-2.5 mM, reconciled)
    (1.0,   25.0),      # EC50_nmda     [mmole]   (brackets B_tot 0.2-2.5 mM, reconciled)
    (0.5,   20.0),      # Tau_Ca        [s]    (widened 1-11 -> 0.5-20: short tau + low g_AHP =
                        #                       seizure-prone corner; long tau + high g_AHP =
                        #                       strong long-lasting adaptation. nominal 8 s.)
    (1e-4,  0.05),      # U_0_ar        [dimensionless]  (floor 1e-4 lifted off 0: a draw at
                        #                                  exactly 0 abolishes async release
                        #                                  and is a log(0) hazard; ceiling
                        #                                  raised 5e-3->0.05 to reach the
                        #                                  pathologically-elevated async-release
                        #                                  regime, ~16x over nominal 3e-3)
    (0.1,   1.0),       # U_max         [1/ms]
    (0.1,   1.0),       # U_0_sr        [dimensionless]
    (0.1,   4.5),       # Omega_f_sr    [1/s]
    (0.1,   4.5),       # Omega_f_ar    [1/s]
    (0.1,   4.5),       # Omega_d       [1/s]
    (0.1,   1.0),       # alpha_syn     [dimensionless]
    (0.3 * 50.0,  4.0 * 50.0),   # g_na coeff  -> 15 .. 200   (widened x0.5-3 -> x0.3-4 to
                        #                       cover Na channelopathy: loss-of-function (hypo-
                        #                       excitable) to gain-of-function (hyperexcitable).
                        #                       NB now spans >1 decade -> flips to LOG.)
    (0.3 * 5.0,   4.0 * 5.0),    # g_kd coeff  -> 1.5 .. 20    (widened x0.5-3 -> x0.3-4 to
                        #                       cover K channelopathy; reduced g_kd -> broadened
                        #                       spikes / hyperexcitability. Flips to LOG.)
    (0.1,   15.0),      # g_ampa        [nS]
    (0.1,   15.0),      # g_nmda        [nS]
    (1e-4,  1e-3),      # alpha_Ca      [dimensionless]  SFA amplitude
    (0.05,  0.5),       # x0            [dimensionless]  quantal vesicle size
    (0.1,   10.0),      # O_G           [1/(uM·s)]  mGluR binding rate
    (1e-3,  1e-1),      # Omega_G       [1/s]   mGluR inactivation
    (0.1,   5.0),       # O_beta        [uM/s]  PLCbeta gain
    (1.0,   15.0),      # O_3K          [uM/s]  IP3-3K rate
    (0.01,  1.0),       # Omega_5P      [1/s]   IP3-5P degradation
    (0.3,   1.5),       # I_bias        [uM]    IP3 exogenous set-point
    (0.1,   10.0),      # F             [uM/s]  GJ + exogenous permeability
    (0.1,   1.0),       # I_Theta       [uM]    tanh threshold
    (0.01,  0.5),       # omega_I       [uM]    tanh steepness
    (0.1,   2.0),       # C_Theta       [uM]    exocytosis Ca2+ threshold
    (0.1,   0.9),       # U_A           [dimensionless]  gliotransmitter release prob
    (50.0,  1000.0),    # G_T           [mM]    total gliotransmitter (ceiling 500->1000 to
                        #                       cover reactive-astrocyte elevated stores)
], dtype=np.float64)

N_DIMS = PARAM_BOUNDS.shape[0]
assert PARAM_BOUNDS.shape == (30, 2)

# =============================================================================
# Log parametrisation:  sampling coordinate  +  storage / inference coordinate
# =============================================================================
# Two DISTINCT operations, made mutually consistent here:
#
#   (1) Log SAMPLING — *where draws land*. Under LINEAR uniform sampling on a
#       multi-decade range (e.g. [1e-3, 1e-1]) ~90% of draws land in the top
#       decade, systematically under-covering the LOW end — which for several
#       axes is the pathological regime (slow mGluR inactivation -> sustained
#       astrocyte activation; low IP3 degradation -> IP3 accumulation; weak SFA;
#       near-abolished async release; synaptic hypofunction; weak GJ drive).
#       Drawing log-uniformly balances coverage per decade.
#
#   (2) Log TRANSFORMATION (parametrisation) — *the coordinate downstream SBI
#       reasons in*. We carry log-axes as their NATURAL LOG, theta = ln(value),
#       so inference operates on a well-conditioned, ~O(1)-scaled space rather
#       than one mixing 1e-4 with 5e2. The simulator is ALWAYS fed NATURAL units;
#       the inversion value = exp(theta) happens in theta_to_natural() in the
#       parent, right before each worker pack is built. run_args is therefore
#       byte-for-byte unaffected.
#
# These are the SAME act viewed from either side of exp(): a uniform draw in
# theta IS the log-uniform draw in natural units. PARAM_BOUNDS stays in NATURAL
# units throughout; PARAM_BOUNDS_THETA (below) is its image in theta-space and
# is what the SBI prior box (e.g. sbi.utils.BoxUniform) must be defined over.
#
# WHICH AXES ARE LOG — every axis spanning >= 1 decade (log10 hi/lo >= 1.0).
# This applies the stated coverage criterion mechanically rather than by hand.
_DECADES = np.log10(PARAM_BOUNDS[:, 1] / PARAM_BOUNDS[:, 0])
LOG_PARAMS = {int(k) for k in range(N_DIMS) if _DECADES[k] >= 1.0}

# ── Optional overrides (uncomment to apply) ──────────────────────────────────
# • If your pathology is strictly hyperexcitable (E/I imbalance toward
#   excitation), you may prefer LINEAR g_ampa/g_nmda so the sweep weights toward
#   strong conductances rather than splitting coverage with the hypofunction
#   tail (these two span ~2.2 decades and are log under the rule above):
# LOG_PARAMS -= {PARAM_NAMES.index('g_ampa'), PARAM_NAMES.index('g_nmda')}
#
# • Bounded probabilities / fractions: flat-in-VALUE (linear) is at least as
#   defensible a prior as flat-in-log. Under the >=1.0 rule, U_0_sr / U_max /
#   alpha_syn (all 0.1–1.0, exactly one decade) are LOG while U_A (0.1–0.9,
#   0.95 decade) is LINEAR — an asymmetry you may want to remove. Uncomment to
#   force all bounded fractions linear for consistency:
# LOG_PARAMS -= {PARAM_NAMES.index(_n) for _n in ('U_0_sr', 'U_max', 'alpha_syn')}

# Safety: the log transform requires a strictly positive lower bound.
for _k in LOG_PARAMS:
    assert PARAM_BOUNDS[_k, 0] > 0.0, (
        f"LOG_PARAMS axis {_k} ({PARAM_NAMES[_k]}) has non-positive lower bound "
        f"{PARAM_BOUNDS[_k, 0]}; cannot log-transform.")

# The base of the log is a free choice (it only rescales the theta axis); we use
# NATURAL LOG, matching the sbi-package convention. Recorded in the manifest as
# 'log_transform': 'natural_log' so inference-time inversion is unambiguous.
LOG_BASE = 'natural'


def _compute_param_bounds_theta() -> np.ndarray:
    """PARAM_BOUNDS mapped into the theta (inference) coordinate: natural-log
    bounds on log-axes, natural-unit bounds on linear axes. Computed once at
    import (cheap, deterministic, re-derived identically in spawned workers)."""
    b = PARAM_BOUNDS.astype(np.float64).copy()
    for k in LOG_PARAMS:
        b[k, 0] = np.log(PARAM_BOUNDS[k, 0])
        b[k, 1] = np.log(PARAM_BOUNDS[k, 1])
    return b


# theta-space bounds: this is the box the SBI prior is defined over.
PARAM_BOUNDS_THETA = _compute_param_bounds_theta()


def theta_to_natural(theta: np.ndarray) -> np.ndarray:
    """Invert the theta parametrisation to NATURAL units for the simulator:
    value = exp(theta) on log-axes, identity on linear axes. run_args is built
    from the returned vector, so the simulator never sees log coordinates."""
    nat = np.asarray(theta, dtype=np.float64).copy()
    for k in LOG_PARAMS:
        nat[k] = np.exp(nat[k])
    return nat


def sample_theta(rng: np.random.Generator) -> np.ndarray:
    """Draw a single 30-D vector in theta (inference) coordinates: uniform in
    NATURAL LOG on log-axes (ln(lo)..ln(hi)), uniform linear on the rest. A
    uniform draw here is exactly the intended log-uniform prior on the natural
    parameter. STORE THIS VECTOR as the SBI training label; feed
    theta_to_natural(theta) to the simulator."""
    return rng.uniform(PARAM_BOUNDS_THETA[:, 0], PARAM_BOUNDS_THETA[:, 1])


def sample_param_vector(rng: np.random.Generator) -> np.ndarray:
    """DEPRECATED backward-compat shim. Returns a NATURAL-unit draw, equal in
    distribution to the legacy sampler (log-uniform is base-independent). New
    code should use sample_theta() + theta_to_natural() so the theta label can
    be stored alongside the natural-unit vector."""
    return theta_to_natural(sample_theta(rng))


# Import-time correctness guard: the theta<->natural inversion must reproduce
# PARAM_BOUNDS exactly at both bound edges (catches any base/exp mismatch).
assert np.allclose(theta_to_natural(PARAM_BOUNDS_THETA[:, 0]), PARAM_BOUNDS[:, 0]) \
   and np.allclose(theta_to_natural(PARAM_BOUNDS_THETA[:, 1]), PARAM_BOUNDS[:, 1]), \
    "theta<->natural inversion is inconsistent with PARAM_BOUNDS"


# =============================================================================
# Atomic JSON write (used for manifest + per-iter sidecars)
# =============================================================================

def _atomic_write_json(path, data) -> None:
    """
    Write JSON atomically: data goes to <path>.tmp, then os.replace makes the
    swap atomic so a SIGKILL mid-write can never leave a partial file.
    """
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, str(path))


def _append_failure_jsonl(topo_dir, record: dict) -> None:
    """
    Atomic-append a single failure record to <topo_dir>/_failures.jsonl.

    JSONL is used (not a per-iter JSON) so concurrent appends from workers
    within the same topology can interleave safely: POSIX guarantees that
    write() calls smaller than PIPE_BUF (≥4096 bytes) on a file opened in
    O_APPEND mode are atomic with respect to each other.  A failure record
    stays well under that limit.

    No iter_<NN>.npz / iter_<NN>.json is ever written for a discarded run,
    so the main dataset only ever contains clean simulations.
    """
    path = os.path.join(topo_dir, '_failures.jsonl')
    line = json.dumps(record, default=str) + '\n'
    if len(line.encode('utf-8')) >= 4000:
        # Truncate fields that are likely the culprits (error_msg, traceback)
        # to keep the line atomically writable.
        for k in ('traceback', 'error_msg'):
            if k in record and isinstance(record[k], str):
                record[k] = record[k][:1500] + '...[truncated]'
        line = json.dumps(record, default=str) + '\n'
    with open(path, 'a') as f:
        f.write(line)


# =============================================================================
# Manifest (re)builder — walks the output tree and aggregates per-iter JSONs
# =============================================================================

def rebuild_manifest(out_dir) -> dict:
    """
    Walk <out_dir>/topo_*/iter_*.json and write a fresh manifest.json that
    bijectively indexes every completed simulation in the job.

    Safe to call at any time: it's a pure function of what's on disk, so a
    walltime kill never leaves the manifest stale beyond the most-recent
    topology that didn't complete.
    """
    root = Path(out_dir)
    topo_dirs = sorted([p for p in root.glob('topo_*') if p.is_dir()])

    topologies = []
    n_total_runs = 0
    n_total_failures = 0
    for td in topo_dirs:
        topo_meta_path = td / 'topology_meta.json'
        topo_meta = {}
        if topo_meta_path.exists():
            try:
                topo_meta = json.loads(topo_meta_path.read_text())
            except Exception as e:
                topo_meta = {'_topology_meta_error': str(e)}

        iter_jsons = sorted(td.glob('iter_*.json'))
        iters = []
        for j in iter_jsons:
            try:
                iters.append(json.loads(j.read_text()))
            except Exception as e:
                iters.append({'_error': str(e), '_path': str(j)})

        # Aggregate discarded-simulation records (one JSON per line) ---------
        fail_path = td / '_failures.jsonl'
        failures = []
        if fail_path.exists():
            try:
                with open(fail_path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            failures.append(json.loads(line))
                        except Exception as e:
                            failures.append({'_parse_error': str(e),
                                             '_raw': line[:200]})
            except Exception as e:
                failures = [{'_failures_jsonl_error': str(e)}]

        topologies.append({
            **topo_meta,
            'topo_dir':                 str(td.relative_to(root)),
            'n_iterations_completed':   len(iters),
            'n_iterations_discarded':   len(failures),
            'iterations':               iters,
            'discarded_iterations':     failures,
        })
        n_total_runs     += len(iters)
        n_total_failures += len(failures)

    manifest = {
        'manifest_version':                 2,        # v1 = legacy 14-D; v2 = 30-D expansion
        'n_dims':                           int(N_DIMS),
        'updated_at':                       time.strftime('%Y-%m-%dT%H:%M:%S'),
        'job_id':                           os.environ.get('PBS_JOBID', ''),
        'host':                             os.environ.get('HOSTNAME', ''),
        'n_topologies_completed_or_partial': len(topo_dirs),
        'n_total_runs':                     n_total_runs,
        'n_total_failures_discarded':       n_total_failures,
        'param_names':                      PARAM_NAMES,
        'param_units':                      PARAM_UNITS,
        'param_bounds':                     PARAM_BOUNDS.tolist(),
        # ── theta (inference) coordinate: SBI prior box + provenance ──────────
        # Build the prior over THESE bounds (BoxUniform(low=col0, high=col1)),
        # train on the per-iter 'theta' arrays, and map posterior samples back
        # with value = exp(theta) on log_param_indices. 'params' stays natural.
        'param_bounds_theta':               PARAM_BOUNDS_THETA.tolist(),
        'log_transform':                    LOG_BASE,   # 'natural' -> ln / exp
        'log_params':                       sorted(PARAM_NAMES[k] for k in LOG_PARAMS),
        'log_param_indices':                sorted(int(k) for k in LOG_PARAMS),
        'theta_storage_keys':               {'npz': 'theta', 'json': 'theta'},
        'topologies':                       topologies,
    }
    _atomic_write_json(root / 'manifest.json', manifest)
    return manifest


# =============================================================================
# Worker entry point
# (Imports Brian2 inside the function so the parent process never touches it,
#  and so multiprocessing-spawn imports are clean.)
# =============================================================================

def _worker_entry(pack: dict) -> list:
    """
    Run one worker's chunk of parameter vectors against a single topology.

    Pack keys
    ---------
    worker_id        : int   — used for the unique cpp_standalone scratch dir
    topo             : dict  — output of HPC_single_run.build_topology
    topo_idx         : int
    topo_dir         : str   — output directory for this topology
    conn_prob        : float — the outer-loop's conn_prob_k
    params_list      : (k, 30) ndarray — NATURAL units; built into run_args
    theta_list       : (k, 30) ndarray — inference coords (ln on log-axes);
                       stored verbatim as the SBI label, never used in run_args
    iter_indices     : list[int]   — local iter indices for the saved filenames
    seed_runs        : list[int]   — one fresh seed_run per parameter vector
    cli              : dict — flattened CLI args needed by the worker
    scratch_root     : str  — root for per-worker cpp_standalone build dirs

    Returns
    -------
    list of dicts (one per parameter vector) with summary stats.
    """
    worker_id     = pack['worker_id']
    topo          = pack['topo']
    topo_idx      = pack['topo_idx']
    topo_dir      = pack['topo_dir']
    conn_prob     = pack['conn_prob']
    params_list   = pack['params_list']
    theta_list    = pack['theta_list']
    iter_indices  = pack['iter_indices']
    seed_runs     = pack['seed_runs']
    cli           = pack['cli']
    scratch_root  = pack['scratch_root']

    # ── Brian2 device init (per the exact pattern requested) ─────────────────
    worker_scratch = os.path.join(scratch_root, f'worker_{worker_id:03d}')
    os.makedirs(worker_scratch, exist_ok=True)

    from brian2 import (set_device, get_device, devices, start_scope,
                        defaultclock, Network, SpikeMonitor,
                        second, ms, mV, nS, mmole, umole, msiemens, cm, um,
                        BrianLogger, Function, DEFAULT_FUNCTIONS)

    set_device('cpp_standalone', build_on_run=False, directory=worker_scratch)
    device = get_device()
    device.reinit()
    device.activate(build_on_run=False)

    BrianLogger.suppress_hierarchy('brian2.devices')
    BrianLogger.suppress_hierarchy('brian2.parsing')

    start_scope()
    devices.device.seed(cli['seed_device'])
    defaultclock.dt = 0.05 * ms

    # ── Library import (inside worker so spawn re-imports cleanly) ──────────
    lib_dir = cli['lib_dir'] or os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, lib_dir)
    from ASD_fun_BD_cpp import (Neuronal_Network, Astrocyte_Group,
                                Gliotransmission, Synapse_to_astro,
                                Astro_to_Syn)

    # ── Binomial_fun (mirrors HPC_single_run.py exactly) ────────────────────
    def _binom_py(n, p, _vectorisation_idx):
        return sum(np.random.rand(n) < p)

    Binomial_fun = Function(
        _binom_py, arg_units=[1, 1], return_unit=1,
        stateless=False, auto_vectorise=True,
    )
    Binomial_fun.implementations.add_implementation(
        'cython',
        '''
cdef double Binomial_fun(int n, double p, _vectorisation_idx):
    cdef int count = 0
    cdef int i
    for i in range(n):
        if rand(_vectorisation_idx) < p:
            count = count + 1
    return count;
''',
        dependencies={'rand': DEFAULT_FUNCTIONS['rand']},
    )
    Binomial_fun.implementations.add_implementation(
        'cpp',
        '''
int Binomial_fun(int n, double p, int _vectorisation_idx) {
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (rand(_vectorisation_idx) < p) count += 1;
    }
    return count;
}
''',
        dependencies={'rand': DEFAULT_FUNCTIONS['rand']},
    )

    simtime = cli['simtime'] * second
    syn_positions = np.column_stack([topo['S_x_syn'], topo['S_y_syn']])

    # ── Build groups ────────────────────────────────────────────────────────
    N, S = Neuronal_Network(
        cli['Nn'],
        Syn_pdist=None,
        ics=False,
        Simulated_network=cli['mode'],
        Decay_type='Double_exp',
        synapse_type='facilitating',
        conn_prob_=conn_prob,                       # this topology's conn_prob
        seed_neu=cli['seed_neuron'],
        seed_syn=cli['seed_synapse'],
        connections=[topo['S_i'], topo['S_j']],
        Binomial_fun=Binomial_fun,
        syn_positions=syn_positions,
    )
    N.I = '(rand() - 0.5) * I_inj'
    N.x = topo['N_pos'][:, 0] * um
    N.y = topo['N_pos'][:, 1] * um

    SpikesN = SpikeMonitor(N, name='Spike_monitor_N')
    net = Network()

    if cli['mode'] == 'Full':
        Astro, GJ = Astrocyte_Group(
            cli['Na'], 'Full',
            seed_astro=cli['seed_astro'],
            ics='steady',
            connections=[topo['GJ_i'], topo['GJ_j']],
        )
        Astro.x_astro = topo['A_pos'][:, 0] * um
        Astro.y_astro = topo['A_pos'][:, 1] * um

        GT = Gliotransmission(cli['Na'], Astro,
                              ics='jitter', seed_astro=cli['seed_astro'])

        StoA, Connections_list = Synapse_to_astro(
            S, Astro,
            connections=[topo['StoA_i'], topo['StoA_j']],
        )
        AtoS = Astro_to_Syn(GT, S, connections=Connections_list)

        SpikesA = SpikeMonitor(Astro, name='Spike_monitor_A')
        net.add([N, S, Astro, GJ, GT, StoA, AtoS, SpikesN, SpikesA])
    else:
        SpikesA = None
        net.add([N, S, SpikesN])

    # ── Compile ONCE ────────────────────────────────────────────────────────
    t0_compile = time.time()
    net.run(simtime)
    device.build(run=False, directory=None)
    t_compile = time.time() - t0_compile

    # ── Detect whether this Brian2 supports device.run(seed=...) ─────────────
    # The kwarg was added in Brian2 ≥2.5.  Older installs (which is what
    # davinci-1 ships by default) raise TypeError on first use.  Detect once,
    # before the inner loop, so we don't spam _failures.jsonl with the same
    # 48 spurious TypeErrors per topology.
    import inspect
    try:
        _run_sig = inspect.signature(device.run)
        _supports_seed = 'seed' in _run_sig.parameters
    except (TypeError, ValueError):
        _supports_seed = False   # can't introspect → assume old API
    noise_mode = 'fresh' if _supports_seed else 'paired'

    if not _supports_seed:
        print(
            f'[worker {worker_id:03d}] NOTE: This Brian2 install lacks '
            f"device.run(seed=...).  Falling back to PAIRED-COMPARISON mode "
            f"(the cpp_standalone binary replays the build-time RNG sequence; "
            f"all parameter vectors within topo_{topo_idx:05d} see the same "
            f"(rand()-0.5)*I_inj initialisation and the same xi noise stream). "
            f"To enable true fresh-noise-per-run, upgrade Brian2 ≥2.5 in the "
            f"conda env:  pip install --upgrade brian2",
            flush=True,
        )

    # ── Inner loop: device.run() once per parameter vector ──────────────────
    # Each iteration is wrapped in try/except.  A run that crashes the
    # cpp_standalone binary, raises a Brian2 exception, or yields non-finite
    # spike times is DISCARDED: no iter_<NN>.npz and no iter_<NN>.json are
    # written.  Only a one-line record goes to <topo_dir>/_failures.jsonl
    # so the corner of parameter space that triggered the pathology is still
    # recoverable post-hoc, without polluting the main dataset.
    area = net['Neuron'].namespace['area']

    summaries = []
    n_consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5    # bail out if the binary appears broken

    for params, theta, iter_idx, seed_run in zip(
            params_list, theta_list, iter_indices, seed_runs):
        params = np.asarray(params, dtype=np.float64)   # NATURAL units (run_args)
        theta  = np.asarray(theta,  dtype=np.float64)   # inference label (stored)

        run_args = {
            # ---- Synapse group (present in both Neuronal and Full) ----
            net['Synapse'].U_0_ar:     params[5],
            net['Synapse'].Umax:       params[6] / ms,
            net['Synapse'].U_0_sr:     params[7],
            net['Synapse'].Omega_f_sr: params[8] / second,
            net['Synapse'].Omega_f_ar: params[9] / second,
            net['Synapse'].Omega_d:    params[10] / second,
            net['Synapse'].alpha_syn:  params[11],
            net['Synapse'].EC50_ampa:  params[2] * mmole,   # was Xi_ampa: params/mmole
            net['Synapse'].EC50_nmda:  params[3] * mmole,   # was Xi_nmda: params/mmole
            net['Synapse'].x0:         params[17],
            net['Synapse'].O_G:        params[18] / umole / second,
            net['Synapse'].Omega_G:    params[19] / second,
            # ---- Neuron group (present in both modes) ----
            net['Neuron'].sigma:       params[0] * mV,
            net['Neuron'].g_AHP:       params[1] * nS,
            net['Neuron'].tau_Ca:      params[4] * second,
            net['Neuron'].g_na:        params[12] * msiemens * cm**-2 * area,
            net['Neuron'].g_kd:        params[13] * msiemens * cm**-2 * area,
            net['Neuron'].g_ampa:      params[14] * nS,
            net['Neuron'].g_nmda:      params[15] * nS,
            net['Neuron'].alpha_Ca:    params[16],
        }

        # Astrocyte + gliotransmission axes: only in Full mode (groups absent in
        # Neuronal mode; targeting them would KeyError).
        if cli['mode'] == 'Full':
            run_args.update({
                net['Astrocyte'].O_beta:      params[20] * umole / second,
                net['Astrocyte'].O_3K:        params[21] * umole / second,
                net['Astrocyte'].Omega_5P:    params[22] / second,
                net['Astrocyte'].I_bias:      params[23] * umole,
                net['Astrocyte'].F:           params[24] * umole / second,
                net['Astrocyte'].I_Theta:     params[25] * umole,
                net['Astrocyte'].omega_I:     params[26] * umole,
                net['Gliot_release'].C_Theta: params[27] * umole,
                net['Gliot_release'].U_A:     params[28],
                net['Gliot_release'].G_T:     params[29] * mmole,
            })

        # ──────────────────────────────────────────────────────────────────
        # Per-parameter try/except: a crash here MUST NOT abort the rest of
        # the worker's chunk, and MUST NOT write any iter_*.npz/iter_*.json.
        # ──────────────────────────────────────────────────────────────────
        try:
            t0_run = time.time()
            if _supports_seed:
                device.run(run_args=run_args, seed=int(seed_run))
            else:
                device.run(run_args=run_args)
            t_run = time.time() - t0_run

            # Harvest --------------------------------------------------------
            spk_N_t = np.asarray(net['Spike_monitor_N'].t / second, dtype=np.float32)
            spk_N_i = np.asarray(net['Spike_monitor_N'].i,          dtype=np.int32)
            if SpikesA is not None:
                spk_A_t = np.asarray(net['Spike_monitor_A'].t / second, dtype=np.float32)
                spk_A_i = np.asarray(net['Spike_monitor_A'].i,          dtype=np.int32)
            else:
                spk_A_t = np.array([], dtype=np.float32)
                spk_A_i = np.array([], dtype=np.int32)

            # Numerical-pathology check: NaN or +/-Inf in spike TIMES is the
            # signature of overflow/underflow in the integrator.  A finite
            # but huge spike count is NOT discarded — that's a legitimate
            # (if extreme) physiological regime; only non-finite is.
            if len(spk_N_t) and not np.isfinite(spk_N_t).all():
                bad = int((~np.isfinite(spk_N_t)).sum())
                raise FloatingPointError(
                    f'spk_N_t has {bad} non-finite entries (overflow/underflow)'
                )
            if len(spk_A_t) and not np.isfinite(spk_A_t).all():
                bad = int((~np.isfinite(spk_A_t)).sum())
                raise FloatingPointError(
                    f'spk_A_t has {bad} non-finite entries (overflow/underflow)'
                )

        except Exception as e:
            # ───── Discard: do NOT write iter_*.npz or iter_*.json ─────────
            n_consecutive_failures += 1
            fail_record = {
                'topo_idx':    int(topo_idx),
                'iter_idx':    int(iter_idx),
                'worker_id':   int(worker_id),
                'conn_prob':   float(conn_prob),
                'params':      params.tolist(),
                'theta':       theta.tolist(),
                'param_names': PARAM_NAMES,
                'seed_run':    int(seed_run),
                'error_type':  type(e).__name__,
                'error_msg':   str(e)[:500],
                'traceback':   traceback.format_exc(),
                'timestamp':   time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            try:
                _append_failure_jsonl(topo_dir, fail_record)
            except Exception:
                pass    # never let the failure-log itself become a failure

            summaries.append({
                'ok':         False,
                'discarded':  True,
                'worker_id':  worker_id,
                'topo_idx':   topo_idx,
                'iter_idx':   iter_idx,
                'error_type': type(e).__name__,
                'error_msg':  str(e)[:200],
            })

            # If the cpp_standalone binary itself is broken (e.g. SIGSEGV
            # left the device in a corrupt state), every subsequent
            # device.run() call in this worker is doomed.  Bail out of the
            # chunk rather than spinning through every parameter vector.
            if n_consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                summaries.append({
                    'ok':       False,
                    'worker_id': worker_id,
                    'topo_idx':  topo_idx,
                    'note':     (f'aborting worker chunk after '
                                 f'{n_consecutive_failures} consecutive failures '
                                 f'— binary likely corrupt'),
                })
                return [{'t_compile_s': float(t_compile),
                         'worker_id': worker_id,
                         'topo_idx':  topo_idx,
                         'n_runs_completed': sum(
                             1 for s in summaries if s.get('ok')),
                         'n_runs_discarded': sum(
                             1 for s in summaries if s.get('discarded')),
                         'aborted_early': True}] + summaries
            continue

        # ───── Success path: save .npz + .json ─────────────────────────────
        n_consecutive_failures = 0

        # In paired mode, store seed_run as -1: that draw was not actually
        # applied to the binary, so keeping its sampled value would be
        # misleading (would imply reproducibility we don't have).
        seed_run_saved = int(seed_run) if _supports_seed else -1

        npz_name = f'iter_{iter_idx:05d}.npz'
        npz_path = os.path.join(topo_dir, npz_name)
        np.savez_compressed(
            npz_path,
            params=params,                       # NATURAL units (human/biophysical)
            theta=theta,                         # inference coords (SBI training label)
            conn_prob=np.float64(conn_prob),
            topo_idx=np.int32(topo_idx),
            seed_run=np.int64(seed_run_saved),
            noise_mode=np.array(noise_mode),     # 'fresh' or 'paired'
            spk_N_t=spk_N_t, spk_N_i=spk_N_i,
            spk_A_t=spk_A_t, spk_A_i=spk_A_i,
        )

        # Save the matching JSON sidecar (used for manifest rebuild) --------
        Nn = int(cli['Nn'])
        Na = int(cli['Na']) if cli['mode'] == 'Full' else 0
        sidecar = {
            'topo_idx':           int(topo_idx),
            'iter_idx':           int(iter_idx),
            'worker_id':          int(worker_id),
            'conn_prob':          float(conn_prob),
            'params':             params.tolist(),
            'theta':              theta.tolist(),
            'param_names':        PARAM_NAMES,
            'seed_run':           seed_run_saved,
            'noise_mode':         noise_mode,
            'n_neuronal_spikes':  int(len(spk_N_t)),
            'n_astrocyte_events': int(len(spk_A_t)),
            'mean_FR_Hz':         float(len(spk_N_t) / (cli['simtime'] * max(Nn, 1))),
            'mean_astro_rate_Hz': (
                float(len(spk_A_t) / (cli['simtime'] * max(Na, 1)))
                if cli['mode'] == 'Full' and Na > 0 else None
            ),
            't_compile_s':        float(t_compile),
            't_run_s':            float(t_run),
            'npz_relpath':        os.path.relpath(npz_path, start=cli['out_dir']),
        }
        json_path = os.path.join(topo_dir, f'iter_{iter_idx:05d}.json')
        _atomic_write_json(json_path, sidecar)

        summaries.append({
            'ok':           True,
            'worker_id':    worker_id,
            'topo_idx':     topo_idx,
            'iter_idx':     iter_idx,
            'n_spk_N':      int(len(spk_N_t)),
            'n_spk_A':      int(len(spk_A_t)),
            't_run_s':      float(t_run),
        })

    n_ok        = sum(1 for s in summaries if s.get('ok'))
    n_discarded = sum(1 for s in summaries if s.get('discarded'))
    return [{'t_compile_s':       float(t_compile),
             'worker_id':         worker_id,
             'topo_idx':          topo_idx,
             'n_runs_completed':  n_ok,
             'n_runs_discarded':  n_discarded}] + summaries


def _worker_entry_wrapped(pack: dict) -> list:
    """
    Exception-safe shell around _worker_entry: a worker crash returns a
    failure record instead of taking down the whole pool.
    """
    try:
        return _worker_entry(pack)
    except Exception as e:
        return [{
            'ok':         False,
            'worker_id':  pack.get('worker_id', -1),
            'topo_idx':   pack.get('topo_idx',  -1),
            'iter_indices': pack.get('iter_indices', []),
            'error':      str(e),
            'traceback':  traceback.format_exc(),
        }]


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'HPC nested random sweep:\n'
            '  outer iter -> draw conn_prob and a fresh topology\n'
            '  inner sweep -> n_workers parallel sims with random 30-D '
            'biophysical parameters.\n'
            'Each completed simulation is saved before the next starts, '
            'so walltime kills are safe.'
        ),
    )

    # Paths -------------------------------------------------------------------
    p.add_argument('--out_dir', required=True)
    p.add_argument('--lib_dir', default=None,
                   help='Directory holding ASD_fun_BD_cpp.py and synapse_pdist.csv.')
    p.add_argument('--syn_pdist_csv', default=None,
                   help='Explicit path to synapse_pdist.csv (overrides auto-discovery).')

    # Sweep size --------------------------------------------------------------
    p.add_argument('--n_workers', type=int, default=None,
                   help='Number of parallel workers per topology. '
                        'Default: $PBS_NCPUS, then $OMP_NUM_THREADS, then os.cpu_count().')
    p.add_argument('--n_topologies', type=int, default=10_000,
                   help='Upper bound on outer iterations. The loop normally '
                        'exits when PBS walltime kills it; this is a safety cap. '
                        '(Default: %(default)s)')
    p.add_argument('--n_params_per_worker', type=int, default=1,
                   help='Number of parameter vectors each worker runs serially '
                        'after compiling once. (Default: %(default)s — one '
                        'parameter vector per worker per topology, matching '
                        'the "max parallelism" design.)')

    # conn_prob sampling ------------------------------------------------------
    p.add_argument('--conn_prob_lo', type=float, default=0.1)
    p.add_argument('--conn_prob_hi', type=float, default=0.6)

    # Topology hyperparameters (fixed across the outer loop) ------------------
    p.add_argument('--Nn',          type=int,   default=100)
    p.add_argument('--Na',          type=int,   default=43)
    p.add_argument('--c_max',       type=float, default=1100.0, metavar='UM')
    p.add_argument('--displ_bias',  type=float, default=15.0,   metavar='UM')
    p.add_argument('--topology_mode', default='wallach',
                   choices=['wallach', 'distance'],
                   help='Astrocyte connectivity rule (Wallach 2014 joint '
                        'Voronoi or legacy distance-based). '
                        'Default: %(default)s.')
    p.add_argument('--gj_dist',     type=float, default=200.0,  metavar='UM',
                   help='[topology_mode=distance only] KDTree radius for '
                        'GJC. Default: %(default)s.')
    p.add_argument('--gj_max_dist', type=float, default=150.0,  metavar='UM',
                   help='[topology_mode=wallach only] Soft distance cap on '
                        'top of the Voronoi rule. Default: %(default)s.')
    p.add_argument('--stoa_cutoff', type=float, default=70.0,   metavar='UM')
    p.add_argument('--stoa_sigma',  type=float, default=200.0,  metavar='UM',
                   help='[topology_mode=distance only] σ of the Gaussian '
                        'StoA acceptance. Default: %(default)s.')

    # Simulation --------------------------------------------------------------
    p.add_argument('--simtime', type=float, default=180.0, metavar='SECONDS')
    p.add_argument('--mode', default='Full', choices=['Full', 'Neuronal'])

    # Seeds -------------------------------------------------------------------
    p.add_argument('--seed_master',  type=int, default=None,
                   help='Master seed for all outer-loop randomness (conn_prob, '
                        'topology seeds, parameter vectors, seed_runs). '
                        'Default: derived from PBS_JOBID for reproducibility.')
    p.add_argument('--seed_device',  type=int, default=50)
    p.add_argument('--seed_neuron',  type=int, default=39)
    p.add_argument('--seed_synapse', type=int, default=35)
    p.add_argument('--seed_astro',   type=int, default=60)

    # Output options ----------------------------------------------------------
    p.add_argument('--dpi', type=int, default=150,
                   help='DPI for spatial-layout PNGs. (Default: %(default)s)')
    p.add_argument('--scratch_root', default=None,
                   help='Root for per-worker cpp_standalone build dirs. '
                        'Default: $TMPDIR/brian2_sweep_$JOBID, fallback '
                        '<out_dir>/_scratch.')

    # Maintenance mode --------------------------------------------------------
    p.add_argument('--rebuild_manifest_only', action='store_true',
                   help='Skip the sweep; just walk <out_dir> and rebuild '
                        'manifest.json from per-iter JSONs.')

    return p


def _resolve_n_workers(args) -> int:
    if args.n_workers is not None and args.n_workers > 0:
        return int(args.n_workers)
    for var in ('PBS_NCPUS', 'PBS_NP', 'OMP_NUM_THREADS'):
        if var in os.environ:
            try:
                v = int(os.environ[var])
                if v > 0:
                    return v
            except ValueError:
                pass
    return int(os.cpu_count() or 1)


def _resolve_seed_master(args) -> int:
    if args.seed_master is not None:
        return int(args.seed_master)
    job_id = os.environ.get('PBS_JOBID', '')
    head = job_id.split('.')[0]
    if head.isdigit():
        return (int(head) * 1_000_003) % (2**31 - 1)
    # No PBS_JOBID — use OS entropy.  Print it so the run is replayable.
    return int.from_bytes(os.urandom(4), 'little') % (2**31 - 1)


def _print_banner(args, n_workers, seed_master) -> None:
    print('=' * 72)
    print('HPC_main_sweep — nested topology × parameter random search')
    print('=' * 72)
    print(f'  out_dir              : {args.out_dir}')
    print(f'  lib_dir              : {args.lib_dir}')
    print(f'  mode                 : {args.mode}   Nn={args.Nn}  Na={args.Na}')
    print(f'  c_max                : {args.c_max} µm')
    print(f'  simtime              : {args.simtime} s')
    print(f'  conn_prob range      : [{args.conn_prob_lo}, {args.conn_prob_hi}]')
    print(f'  topology_mode        : {args.topology_mode}')
    print(f'  topology fixed args  : displ_bias={args.displ_bias} gj_dist={args.gj_dist} '
          f'gj_max_dist={args.gj_max_dist} '
          f'stoa_cutoff={args.stoa_cutoff} stoa_sigma={args.stoa_sigma}')
    print(f'  n_workers            : {n_workers}')
    print(f'  n_params_per_worker  : {args.n_params_per_worker}')
    print(f'  sims per topology    : {n_workers * args.n_params_per_worker}')
    print(f'  max n_topologies     : {args.n_topologies}')
    print(f'  seed_master          : {seed_master}')
    print(f'  PBS_JOBID            : {os.environ.get("PBS_JOBID", "(none)")}')
    print('=' * 72, flush=True)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Rebuild-only mode ────────────────────────────────────────────────────
    if args.rebuild_manifest_only:
        m = rebuild_manifest(args.out_dir)
        print(f'[rebuild] manifest.json updated  '
              f'({m["n_topologies_completed_or_partial"]} topologies, '
              f'{m["n_total_runs"]} total runs)')
        return

    n_workers   = _resolve_n_workers(args)
    seed_master = _resolve_seed_master(args)
    _print_banner(args, n_workers, seed_master)

    # ── Save the resolved CLI args at the job root (for replay) ──────────────
    args_dict = vars(args).copy()
    args_dict['_resolved_n_workers']   = n_workers
    args_dict['_resolved_seed_master'] = seed_master
    args_dict['_pbs_jobid']            = os.environ.get('PBS_JOBID', '')
    args_dict['_hostname']             = os.environ.get('HOSTNAME', '')
    args_dict['_started_at']           = time.strftime('%Y-%m-%dT%H:%M:%S')
    _atomic_write_json(os.path.join(args.out_dir, 'job_args.json'), args_dict)

    # ── Scratch root for cpp_standalone build dirs ───────────────────────────
    if args.scratch_root is not None:
        scratch_root = args.scratch_root
    elif 'TMPDIR' in os.environ:
        scratch_root = os.path.join(
            os.environ['TMPDIR'],
            f'brian2_sweep_{os.environ.get("PBS_JOBID","local").split(".")[0]}',
        )
    else:
        scratch_root = os.path.join(args.out_dir, '_scratch')
    os.makedirs(scratch_root, exist_ok=True)
    print(f'[main] cpp_standalone build dirs under: {scratch_root}')

    # ── Resolve synapse_pdist.csv once (passed to every topology build) ─────
    syn_prob_csv = _resolve_syn_pdist_csv(args)

    # ── Master RNG drives ALL stochasticity in the outer loop ────────────────
    master_rng = np.random.default_rng(seed_master)

    # ── multiprocessing context: spawn, NOT fork.  Critical for Brian2:
    #    fork would inherit half-initialised Cython / matplotlib state into
    #    the children and cause subtle hangs.  Spawn re-imports cleanly.
    ctx = mp.get_context('spawn')

    sims_per_topo = n_workers * args.n_params_per_worker

    # ─────────────────────────────────────────────────────────────────────────
    # OUTER LOOP — one iteration per (random topology, random conn_prob)
    # ─────────────────────────────────────────────────────────────────────────
    for topo_idx in range(args.n_topologies):

        topo_dir = os.path.join(args.out_dir, f'topo_{topo_idx:05d}')
        os.makedirs(topo_dir, exist_ok=True)

        # 1) Outer-loop random draws -----------------------------------------
        conn_prob       = float(master_rng.uniform(args.conn_prob_lo, args.conn_prob_hi))
        topo_seed_base  = int(master_rng.integers(0, 2**31 - 1))
        seed_neuron_k   = (topo_seed_base + args.seed_neuron)  % (2**31 - 1)
        seed_synapse_k  = (topo_seed_base + args.seed_synapse) % (2**31 - 1)
        seed_astro_k    = (topo_seed_base + args.seed_astro)   % (2**31 - 1)

        # 2) Pass-1 numpy topology -------------------------------------------
        t0 = time.time()
        try:
            topo = build_topology(
                Nn=args.Nn,
                Na=(args.Na if args.mode == 'Full' else 0),
                c_max=args.c_max,
                conn_prob=conn_prob,
                syn_prob_csv=syn_prob_csv,
                displ_bias=args.displ_bias,
                gj_dist=args.gj_dist,
                stoa_cutoff=args.stoa_cutoff,
                stoa_sigma=args.stoa_sigma,
                seed_neuron=seed_neuron_k,
                seed_synapse=seed_synapse_k,
                seed_astro=seed_astro_k,
                mode=args.mode,
                topology_mode=args.topology_mode,
                gj_max_dist=args.gj_max_dist,
            )
        except Exception as e:
            print(f'[main] topo_{topo_idx:05d}: FAILED to build topology '
                  f'(conn_prob={conn_prob:.4f}): {e}', flush=True)
            traceback.print_exc()
            continue
        t_topo = time.time() - t0

        save_topology(topo, topo_dir)
        plot_spatial_layout(
            topo, c_max=args.c_max, mode=args.mode,
            out_path=os.path.join(topo_dir, 'spatial_layout.png'),
            dpi=args.dpi,
            topology_mode=args.topology_mode,
            also_connectivity=True,
        )

        # 3) Topology metadata (sidecar) -------------------------------------
        Na_eff = int(args.Na) if args.mode == 'Full' else 0
        topo_meta = {
            'topo_idx':          int(topo_idx),
            'conn_prob':         float(conn_prob),
            'topo_seed_base':    int(topo_seed_base),
            'seed_neuron':       int(seed_neuron_k),
            'seed_synapse':      int(seed_synapse_k),
            'seed_astro':        int(seed_astro_k),
            'Nn':                int(args.Nn),
            'Na':                Na_eff,
            'c_max':             float(args.c_max),
            'displ_bias':        float(args.displ_bias),
            'topology_mode':     args.topology_mode,
            'gj_dist':           float(args.gj_dist),
            'gj_max_dist':       float(args.gj_max_dist),
            'stoa_cutoff':       float(args.stoa_cutoff),
            'stoa_sigma':        float(args.stoa_sigma),
            'n_synapses':        int(len(topo['S_i'])),
            'n_gj_links':        int(len(topo['GJ_i'])),
            'n_stoa_links':      int(len(topo['StoA_i'])),
            'p_eff_actual':      float(len(topo['S_i']) / max(args.Nn * (args.Nn - 1), 1)),
            't_topo_build_s':    float(t_topo),
            'mode':              args.mode,
            'simtime_s':         float(args.simtime),
        }
        _atomic_write_json(os.path.join(topo_dir, 'topology_meta.json'), topo_meta)

        # 4) Sample sims_per_topo vectors in theta (inference) space, then
        #    invert to NATURAL units for the simulator. theta is the SBI label;
        #    param (natural) is what run_args is built from. Deriving natural
        #    solely via theta_to_natural() keeps the two coordinates consistent
        #    by construction (no chance of desync in the worker pack).
        theta_matrix = np.array(
            [sample_theta(master_rng) for _ in range(sims_per_topo)]
        )
        param_matrix = np.array(
            [theta_to_natural(th) for th in theta_matrix]
        )
        seed_runs = master_rng.integers(0, 2**31 - 1, size=sims_per_topo)

        # 5) Split into n_workers chunks (round-robin) -----------------------
        tasks = []
        cli_dict = {
            'lib_dir':       args.lib_dir,
            'out_dir':       args.out_dir,
            'simtime':       float(args.simtime),
            'mode':          args.mode,
            'Nn':            int(args.Nn),
            'Na':            int(args.Na),
            'seed_device':   int(args.seed_device),
            'seed_neuron':   int(args.seed_neuron),
            'seed_synapse':  int(args.seed_synapse),
            'seed_astro':    int(args.seed_astro),
        }
        for w in range(n_workers):
            chunk_idxs = list(range(w, sims_per_topo, n_workers))
            if not chunk_idxs:
                continue
            tasks.append({
                'worker_id':    w,
                'topo':         topo,
                'topo_idx':     topo_idx,
                'topo_dir':     topo_dir,
                'conn_prob':    conn_prob,
                'params_list':  [param_matrix[i] for i in chunk_idxs],
                'theta_list':   [theta_matrix[i] for i in chunk_idxs],
                'iter_indices': chunk_idxs,
                'seed_runs':    [int(seed_runs[i]) for i in chunk_idxs],
                'cli':          cli_dict,
                'scratch_root': scratch_root,
            })

        # 6) Dispatch ---------------------------------------------------------
        print(f'\n[main] topo_{topo_idx:05d} | conn_prob = {conn_prob:.4f} | '
              f'n_syn={len(topo["S_i"])}, n_gj={len(topo["GJ_i"])}, '
              f'n_stoa={len(topo["StoA_i"])}', flush=True)
        print(f'[main]   launching {len(tasks)} workers × '
              f'{args.n_params_per_worker} params each '
              f'= {sims_per_topo} simulations  (topo built in {t_topo:.2f} s)',
              flush=True)

        t0_pool = time.time()
        try:
            with ctx.Pool(n_workers) as pool:
                all_results = pool.map(_worker_entry_wrapped, tasks)
        except KeyboardInterrupt:
            print('[main] KeyboardInterrupt — rebuilding manifest and exiting.')
            rebuild_manifest(args.out_dir)
            return
        t_pool = time.time() - t0_pool

        # 7) Report this topology's batch ------------------------------------
        n_ok = 0
        n_fail = 0          # worker-level (whole chunk) failures
        n_discarded = 0     # per-parameter discards (overflow / underflow / etc.)
        compile_times = []
        run_times = []
        for worker_results in all_results:
            for r in worker_results:
                if not isinstance(r, dict):
                    continue
                if 't_compile_s' in r:
                    compile_times.append(r['t_compile_s'])
                    continue
                if r.get('discarded'):
                    n_discarded += 1
                    continue
                if r.get('ok', False):
                    n_ok += 1
                    run_times.append(r.get('t_run_s', 0.0))
                else:
                    n_fail += 1
                    print(f'[main]   WORKER FAILURE iter={r.get("iter_indices", "?")} '
                          f'worker={r.get("worker_id", -1)}: {r.get("error", "?")}',
                          flush=True)
                    if 'traceback' in r:
                        print(r['traceback'], flush=True)

        if compile_times:
            print(f'[main]   compile times (s): '
                  f'min={min(compile_times):.1f} mean={np.mean(compile_times):.1f} '
                  f'max={max(compile_times):.1f}', flush=True)
        if run_times:
            print(f'[main]   per-run times (s): '
                  f'min={min(run_times):.1f} mean={np.mean(run_times):.1f} '
                  f'max={max(run_times):.1f}', flush=True)
        print(f'[main]   topo_{topo_idx:05d} done: '
              f'{n_ok} ok / {n_discarded} discarded / {n_fail} worker-failed '
              f'in {t_pool:.1f} s wall', flush=True)

        # 8) Rebuild manifest after each topology ----------------------------
        try:
            rebuild_manifest(args.out_dir)
        except Exception as e:
            print(f'[main]   manifest rebuild failed (non-fatal): {e}', flush=True)

    print('\n[done] outer loop exhausted args.n_topologies '
          f'({args.n_topologies}) without walltime kill.')
    rebuild_manifest(args.out_dir)


if __name__ == '__main__':
    main()
