#!/usr/bin/env python3
# =============================================================================
# HPC_single_run.py
#
# Self-contained, zero-precomputed-files simulation of the neuron+astrocyte
# network for a single parameter set on an HPC node.
#
# ARCHITECTURE — two-pass approach
# ─────────────────────────────────
# Pass 1  (pure numpy, no Brian2)
#   Computes all topology from scratch:
#     • neuron positions    (uniform random in [0, c_max]²)
#     • synapse (i,j) pairs (Bernoulli with conn_prob, no autapses)
#     • synapse bouton positions (Gaussian around post-soma, σ = bouton_sigma)
#     • astrocyte positions (uniform random)
#     • gap-junction (i,j) pairs (KDTree radius = gj_dist)
#     • synapse→astrocyte (i,j) pairs (Gaussian p, cutoff = stoa_cutoff)
#   No Brian2 device is ever touched in this pass.
#
# Pass 2  (Brian2 cpp_standalone)
#   set_device → device.reinit → start_scope.
#   All groups are built with explicit (i=..., j=...) lists from pass 1 and
#   numpy assignment of positions.  No state variable is ever read back before
#   the C++ binary runs, so the cpp_standalone limitation does not apply.
#
# OUTPUTS (all written to --out_dir)
#   iter_0000000.npz       spike times and indices for neurons and astrocytes
#   topology.npz           full topology from pass 1 (positions, (i,j) arrays)
#   spatial_layout.png     top-down view of neurons / astrocytes / synapse
#                            boutons, with GJ and StoA links overlaid
#   raster_neurons.png     neuronal raster + population firing-rate trace
#   raster_astrocytes.png  astrocyte Ca²⁺ event raster + event-rate trace
#   run_summary.txt        parameter vector, spike statistics, wall-clock times
#
# USAGE
#   python HPC_single_run.py --out_dir /path/to/results [options]
#   python HPC_single_run.py --help
#
# See submit_single_run.sh for the SLURM wrapper.
# =============================================================================

import argparse
import os
import sys
import textwrap
import time as _wall

# Non-interactive backend — mandatory on HPC nodes (no display).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# =============================================================================
# CLI
# =============================================================================

PARAM_NAMES = [
    'Sigma', 'g_AHP',
    'Xi_ampa', 'Xi_nmda', 'Tau_Ca',
    'U_0_ar', 'U_max', 'U_0_sr',
    'Omega_f_sr', 'Omega_f_ar', 'Omega_d',
    'alpha_syn',
    'g_na', 'g_kd',
]

PARAM_UNITS = [
    'mV', 'nS',
    '1/mmole', '1/mmole', 's',
    '(dimensionless)', '1/ms', '(dimensionless)',
    '1/s', '1/s', '1/s',
    '(dimensionless)',
    'mS cm⁻² coeff', 'mS cm⁻² coeff',
]

# Nominal parameter vector.  Each entry is used as the default when the
# corresponding CLI flag is omitted.
NOMINAL_PARAMS = np.array([
    4.0,        # Sigma          mV
    5.0,        # g_AHP          nS
    0.5,        # Xi_ampa        1/mmole
    0.3,        # Xi_nmda        1/mmole
    8.0,        # Tau_Ca         s
    0.003,      # U_0_ar         (dimensionless)
    0.5,        # U_max          1/ms
    0.15,       # U_0_sr         (dimensionless)
    2.0,        # Omega_f_sr     1/s
    1.42857,    # Omega_f_ar     1/s   (= 1/0.7)
    2.0,        # Omega_d        1/s
    1.0,        # alpha_syn      (dimensionless)
    80.0,       # g_na  coeff    (= 1.6 × 50)
    6.5,        # g_kd  coeff    (= 1.3 × 5)
])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Two-pass, self-contained neuron+astrocyte simulation (cpp_standalone).

            Pass 1 — pure-numpy topology (positions, (i,j) arrays, bouton positions).
            Pass 2 — cpp_standalone build, compile, single run.

            All connectivity parameters are exposed on the CLI; no pre-computed
            connection directory is required.
        """),
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    p.add_argument('--out_dir', required=True, metavar='PATH',
                   help='Output directory (created if absent).')
    p.add_argument('--lib_dir', default=None, metavar='PATH',
                   help='Directory containing ASD_fun_BD_cpp.py.  '
                        'Defaults to the directory of this script.')

    # ── Simulation ───────────────────────────────────────────────────────────
    p.add_argument('--simtime', type=float, default=180.0, metavar='SECONDS',
                   help='Simulated duration in seconds (default: %(default)s).')
    p.add_argument('--mode', default='Full',
                   choices=['Full', 'Neuronal'],
                   help='Network mode (default: %(default)s).')

    # ── Population sizes ─────────────────────────────────────────────────────
    p.add_argument('--Nn', type=int, default=100,
                   help='Number of neurons (default: %(default)s).')
    p.add_argument('--Na', type=int, default=43,
                   help='Number of astrocytes (default: %(default)s).  '
                        'Ignored when --mode=Neuronal.')

    # ── Arena ────────────────────────────────────────────────────────────────
    p.add_argument('--c_max', type=float, default=1100.0, metavar='UM',
                   help='Side length of the square arena in µm (default: %(default)s).')

    # ── Synapse connectivity ──────────────────────────────────────────────────
    p.add_argument('--conn_prob', type=float, default=0.107, metavar='PROB',
                   help='Bernoulli connection probability for neuron→neuron '
                        'synapses (autapses excluded, default: %(default)s).')
    p.add_argument('--syn_pdist_csv', default=None, metavar='PATH',
                   help='Path to the synapse_pdist.csv file (columns: '
                        'Syn_prob, Radius_val), produced by '
                        'generate_dendritic_arbour() in the library.  '
                        'If omitted, the script looks for synapse_pdist.csv '
                        'inside --lib_dir.')
    p.add_argument('--displ_bias', type=float, default=15.0, metavar='UM',
                   help='Constant µm offset added to every sampled distance '
                        'from the post-soma so the bouton never coincides '
                        'with the soma centre (default: %(default)s).')

    # ── Astrocyte connectivity ────────────────────────────────────────────────
    p.add_argument('--gj_dist', type=float, default=200.0, metavar='UM',
                   help='KDTree radius for gap-junction coupling between '
                        'astrocytes (µm, default: %(default)s).')
    p.add_argument('--stoa_cutoff', type=float, default=70.0, metavar='UM',
                   help='Hard distance cutoff for synapse→astrocyte links '
                        '(µm, default: %(default)s).')
    p.add_argument('--stoa_sigma', type=float, default=200.0, metavar='UM',
                   help='σ of the Gaussian connection probability for '
                        'synapse→astrocyte links (µm, default: %(default)s).')

    # ── Seeds ────────────────────────────────────────────────────────────────
    p.add_argument('--seed_device',  type=int, default=50)
    p.add_argument('--seed_neuron',  type=int, default=39)
    p.add_argument('--seed_synapse', type=int, default=35)
    p.add_argument('--seed_astro',   type=int, default=60)
    p.add_argument('--seed_run',     type=int, default=None, metavar='INT',
                   help='If given, forwarded to device.run(seed=...) so that '
                        'each run sees its own stochastic draw.  '
                        'Omit for paired-comparison mode.')

    # ── Swept parameters ─────────────────────────────────────────────────────
    g = p.add_argument_group('Swept parameters (units as noted; '
                             'defaults = nominal operating point)')
    g.add_argument('--Sigma',      type=float, default=None, metavar='mV')
    g.add_argument('--g_AHP',      type=float, default=None, metavar='nS')
    g.add_argument('--Xi_ampa',    type=float, default=None, metavar='1/mmole')
    g.add_argument('--Xi_nmda',    type=float, default=None, metavar='1/mmole')
    g.add_argument('--Tau_Ca',     type=float, default=None, metavar='s')
    g.add_argument('--U_0_ar',     type=float, default=None)
    g.add_argument('--U_max',      type=float, default=None, metavar='1/ms')
    g.add_argument('--U_0_sr',     type=float, default=None)
    g.add_argument('--Omega_f_sr', type=float, default=None, metavar='1/s')
    g.add_argument('--Omega_f_ar', type=float, default=None, metavar='1/s')
    g.add_argument('--Omega_d',    type=float, default=None, metavar='1/s')
    g.add_argument('--alpha_syn',  type=float, default=None)
    g.add_argument('--g_na',       type=float, default=None, metavar='COEFF',
                   help='g_na coefficient  (SI value = coeff × mS cm⁻² × area)')
    g.add_argument('--g_kd',       type=float, default=None, metavar='COEFF',
                   help='g_kd coefficient  (SI value = coeff × mS cm⁻² × area)')

    # ── Figure options ────────────────────────────────────────────────────────
    p.add_argument('--dpi', type=int, default=200,
                   help='DPI for saved figures (default: %(default)s).')

    return p


def resolve_params(args) -> np.ndarray:
    """Merge CLI-supplied values onto the nominal parameter vector."""
    p = NOMINAL_PARAMS.copy()
    cli_values = [
        args.Sigma, args.g_AHP,
        args.Xi_ampa, args.Xi_nmda, args.Tau_Ca,
        args.U_0_ar, args.U_max, args.U_0_sr,
        args.Omega_f_sr, args.Omega_f_ar, args.Omega_d,
        args.alpha_syn, args.g_na, args.g_kd,
    ]
    for k, val in enumerate(cli_values):
        if val is not None:
            p[k] = val
    return p


# =============================================================================
# Pass 1 — pure-numpy topology builder
# =============================================================================

def build_topology(Nn, Na, c_max,
                   conn_prob,
                   syn_prob_csv, displ_bias,
                   gj_dist, stoa_cutoff, stoa_sigma,
                   seed_neuron, seed_synapse, seed_astro,
                   mode='Full') -> dict:
    """
    Build the full network topology using plain NumPy / SciPy.
    No Brian2 device is touched.

    Bouton placement follows the morphometric chi²(4) Sholl distribution
    loaded from `syn_prob_csv` (columns: Syn_prob, Radius_val).  For each
    synapse the distance from the post-soma is drawn from that PDF (plus
    `displ_bias` µm) and the bouton is placed along the line connecting
    the pre and post somata, moving toward the pre-soma — exactly
    mirroring `get_synapse_coordinates` in the library.

    Returns
    -------
    dict with keys
        N_pos       (Nn, 2) float64  — neuron (x, y) positions in µm
        S_i, S_j    (n_syn,) int32   — synapse pre/post indices
        S_x_syn     (n_syn,) float64 — bouton x position in µm
        S_y_syn     (n_syn,) float64 — bouton y position in µm
        A_pos       (Na, 2) float64  — astrocyte (x, y) positions in µm
        GJ_i, GJ_j  (n_gj,) int32    — gap-junction (bidirectional) pairs
        StoA_i      (n_sta,) int32   — synapse index  (→ astrocyte)
        StoA_j      (n_sta,) int32   — astrocyte index
    """
    from scipy.spatial import KDTree

    print('\n[pass 1] Building topology (pure numpy) ...')
    t0 = _wall.time()

    rng_n = np.random.default_rng(seed_neuron)
    rng_s = np.random.default_rng(seed_synapse)
    rng_a = np.random.default_rng(seed_astro)

    # ── Neuron positions ──────────────────────────────────────────────────────
    N_pos = rng_n.uniform(0.0, c_max, (Nn, 2))

    # ── Random synapse connectivity (Bernoulli, no autapses) ──────────────────
    # Vectorised: draw a full Nn×Nn matrix, force the diagonal to fail.
    draw = rng_s.random((Nn, Nn))
    draw[np.arange(Nn), np.arange(Nn)] = 1.0
    S_i_arr, S_j_arr = np.where(draw < conn_prob)
    S_i = S_i_arr.astype(np.int32)
    S_j = S_j_arr.astype(np.int32)
    n_syn = len(S_i)

    # ── Synapse bouton positions (chi²(4) Sholl PDF from CSV) ─────────────────
    # Distance from the post-soma is drawn from the loaded PDF + displ_bias.
    # The bouton is then placed along the pre→post axis, on the post side,
    # at that distance from the post-soma.
    syn_prob   = np.asarray(syn_prob_csv['Syn_prob'].values,   dtype=np.float64)
    radius_val = np.asarray(syn_prob_csv['Radius_val'].values, dtype=np.float64)

    # Precompute normalised PDF and its non-zero support (used in every draw).
    prob_sum = syn_prob.sum()
    if prob_sum <= 0.0:
        raise ValueError("synapse_pdist.csv: all Syn_prob values are zero.")
    normalised = syn_prob / prob_sum
    valid_mask = normalised > 0.0
    valid_radii = radius_val[valid_mask]
    valid_probs = normalised[valid_mask]

    def _sample_distance():
        return rng_s.choice(valid_radii, p=valid_probs)

    S_x_syn = np.empty(n_syn, dtype=np.float64)
    S_y_syn = np.empty(n_syn, dtype=np.float64)

    MAX_TRIES = 50
    n_fallback = 0
    for k in range(n_syn):
        pre_pos  = N_pos[S_i[k]]
        post_pos = N_pos[S_j[k]]

        vec        = post_pos - pre_pos
        vec_length = np.linalg.norm(vec)

        # Degenerate case (two neurons at identical position): skip rejection
        # sampling and fall back to midpoint placement.
        if vec_length <= displ_bias:
            S_x_syn[k] = 0.5 * (pre_pos[0] + post_pos[0])
            S_y_syn[k] = 0.5 * (pre_pos[1] + post_pos[1])
            n_fallback += 1
            continue

        unit_vec = vec / vec_length

        # Rejection sample distance so that the bouton stays on the (pre, post)
        # segment.  Hard cap on tries to avoid pathological infinite loops.
        distance = _sample_distance() + displ_bias
        n_try = 1
        while distance > vec_length and n_try < MAX_TRIES:
            distance = _sample_distance() + displ_bias
            n_try += 1
        if distance > vec_length:
            distance = 0.5 * vec_length      # safe fallback
            n_fallback += 1

        # Move 'distance' along the unit vector from the post-syn neuron.
        new_pos = post_pos - unit_vec * distance
        S_x_syn[k] = new_pos[0]
        S_y_syn[k] = new_pos[1]

    if n_fallback:
        print(f'[pass 1] WARNING: {n_fallback}/{n_syn} synapses used the '
              f'midpoint fallback (pre/post somata closer than displ_bias '
              f'or rejection sampling exhausted).')

    if mode == 'Neuronal':
        # Astrocyte groups are not needed — return empty arrays.
        topo = dict(
            N_pos=N_pos,
            S_i=S_i, S_j=S_j, S_x_syn=S_x_syn, S_y_syn=S_y_syn,
            A_pos=np.empty((0, 2)),
            GJ_i=np.array([], dtype=np.int32),
            GJ_j=np.array([], dtype=np.int32),
            StoA_i=np.array([], dtype=np.int32),
            StoA_j=np.array([], dtype=np.int32),
        )
        _log_topology(topo, _wall.time() - t0)
        return topo

    # ── Astrocyte positions ───────────────────────────────────────────────────
    A_pos = rng_a.uniform(0.0, c_max, (Na, 2))

    # ── Gap-junction connectivity (KDTree, bidirectional) ─────────────────────
    tree_a = KDTree(A_pos)
    gj_pairs_set = tree_a.query_pairs(r=gj_dist)        # set of (i, j) with i < j
    if gj_pairs_set:
        gj_arr = np.array(sorted(gj_pairs_set), dtype=np.int32)
        # Make bidirectional.
        GJ_i = np.concatenate([gj_arr[:, 0], gj_arr[:, 1]])
        GJ_j = np.concatenate([gj_arr[:, 1], gj_arr[:, 0]])
    else:
        GJ_i = np.array([], dtype=np.int32)
        GJ_j = np.array([], dtype=np.int32)

    # ── Synapse → Astrocyte connectivity ──────────────────────────────────────
    # For each astrocyte, find all boutons within stoa_cutoff µm.
    # Accept each candidate with Gaussian probability p ~ exp(-d²/2σ²).
    bouton_pos = np.column_stack([S_x_syn, S_y_syn])    # (n_syn, 2)
    tree_s     = KDTree(bouton_pos)

    candidates = tree_s.query_ball_point(A_pos, r=stoa_cutoff, workers=1)

    StoA_i_list, StoA_j_list = [], []
    for aj, syn_indices in enumerate(candidates):
        if not syn_indices:
            continue
        syn_idx = np.array(syn_indices, dtype=np.int32)
        d = np.linalg.norm(bouton_pos[syn_idx] - A_pos[aj], axis=1)
        p = np.exp(-(d ** 2) / (2.0 * stoa_sigma ** 2))
        accept = rng_s.random(len(syn_idx)) < p
        for si in syn_idx[accept]:
            StoA_i_list.append(si)
            StoA_j_list.append(aj)

    StoA_i = np.array(StoA_i_list, dtype=np.int32)
    StoA_j = np.array(StoA_j_list, dtype=np.int32)

    topo = dict(
        N_pos=N_pos,
        S_i=S_i, S_j=S_j, S_x_syn=S_x_syn, S_y_syn=S_y_syn,
        A_pos=A_pos,
        GJ_i=GJ_i, GJ_j=GJ_j,
        StoA_i=StoA_i, StoA_j=StoA_j,
    )
    _log_topology(topo, _wall.time() - t0)
    return topo


def _log_topology(topo: dict, elapsed: float) -> None:
    """Print a compact topology summary."""
    n_syn  = len(topo['S_i'])
    n_gj   = len(topo['GJ_i'])
    n_stoa = len(topo['StoA_i'])
    Na     = len(topo['A_pos'])
    Nn     = len(topo['N_pos'])
    print(f'[pass 1] Done in {elapsed:.2f} s')
    print(f'         {Nn} neurons     {n_syn} synapses '
          f'(p_actual = {n_syn / max(Nn*(Nn-1), 1):.4f})')
    if Na > 0:
        print(f'         {Na} astrocytes  {n_gj} GJ links  '
              f'{n_stoa} StoA links')


def save_topology(topo: dict, out_dir: str) -> str:
    """Save topology arrays so the run can be reproduced or inspected."""
    path = os.path.join(out_dir, 'topology.npz')
    np.savez_compressed(path, **topo)
    print(f'[pass 1] Topology saved → {path}')
    return path


# =============================================================================
# Pass 2 — cpp_standalone build + single run
# =============================================================================

def build_and_run(topo: dict, args, params: np.ndarray) -> dict:
    """
    Build the cpp_standalone network from pre-computed topology arrays,
    compile, run once, and return the harvested spike data.

    Parameters
    ----------
    topo : dict
        Output of `build_topology`.
    args : argparse.Namespace
    params : np.ndarray  shape (14,)

    Returns
    -------
    dict with spk_N_t, spk_N_i, spk_A_t, spk_A_i (numpy arrays),
         plus t_compile_s and t_run_s (floats).
    """
    # ── Brian2 device setup (must precede start_scope and group creation) ────
    from brian2 import (
        set_device, get_device, devices, start_scope,
        defaultclock, Network,
        NeuronGroup, Synapses, SpikeMonitor,
        second, ms, mV, nS, mmole, msiemens, cm, um,
        BrianLogger, Function, DEFAULT_FUNCTIONS, Equations,
        linked_var, float32,
    )

    set_device('cpp_standalone', build_on_run=False)
    device = get_device()
    device.reinit()
    device.activate()

    BrianLogger.suppress_hierarchy('brian2.devices')
    BrianLogger.suppress_hierarchy('brian2.parsing')

    start_scope()
    devices.device.seed(args.seed_device)
    defaultclock.dt = 0.05 * ms

    print('\n[pass 2] Brian2 cpp_standalone: building groups ...')

    # ── Library import ────────────────────────────────────────────────────────
    lib_dir = args.lib_dir or os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, lib_dir)
    from ASD_fun_BD_cpp import (
        Neuronal_Network, Astrocyte_Group, Gliotransmission,
        Synapse_to_astro, Astro_to_Syn,
    )

    # ── Binomial_fun ──────────────────────────────────────────────────────────
    def _binom_py(n, p, _vectorisation_idx):
        return sum(np.random.rand(n) < p)

    Binomial_fun = Function(
        _binom_py,
        arg_units=[1, 1], return_unit=1,
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

    simtime = args.simtime * second

    # ── Synapse positions as (N, 2) array in µm ───────────────────────────────
    syn_positions = np.column_stack([topo['S_x_syn'], topo['S_y_syn']])

    # ── Build neuronal network (explicit (i,j) lists from pass 1) ─────────────
    N, S = Neuronal_Network(
        args.Nn,
        Syn_pdist=None,
        ics=False,
        Simulated_network=args.mode,
        Decay_type='Double_exp',
        synapse_type='facilitating',
        conn_prob_=args.conn_prob,
        seed_neu=args.seed_neuron,
        seed_syn=args.seed_synapse,
        connections=[topo['S_i'], topo['S_j']],
        Binomial_fun=Binomial_fun,
        syn_positions=syn_positions,
    )
    N.I = '(rand() - 0.5) * I_inj'

    # Override neuron positions with pass-1 values (µm → metre).
    # This is a set-only operation; we never read them back before run().
    N.x = topo['N_pos'][:, 0] * um
    N.y = topo['N_pos'][:, 1] * um

    SpikesN = SpikeMonitor(N, name='Spike_monitor_N')

    net = Network()

    if args.mode == 'Full':
        # ── Astrocyte group ───────────────────────────────────────────────────
        Astro, GJ = Astrocyte_Group(
            args.Na, 'Full',
            seed_astro=args.seed_astro,
            ics='steady',
            connections=[topo['GJ_i'], topo['GJ_j']],
        )
        # Override astrocyte positions with pass-1 values.
        Astro.x_astro = topo['A_pos'][:, 0] * um
        Astro.y_astro = topo['A_pos'][:, 1] * um

        GT = Gliotransmission(args.Na, Astro,
                              ics='jitter', seed_astro=args.seed_astro)

        StoA, Connections_list = Synapse_to_astro(
            S, Astro,
            connections=[topo['StoA_i'], topo['StoA_j']],
        )
        AtoS = Astro_to_Syn(GT, S, connections=Connections_list)

        SpikesA = SpikeMonitor(Astro, name='Spike_monitor_A')

        net.add([N, S, Astro, GJ, GT, StoA, AtoS, SpikesN, SpikesA])

    else:   # Neuronal
        SpikesA = None
        net.add([N, S, SpikesN])

    # ── Compile ───────────────────────────────────────────────────────────────
    print('[pass 2] Compiling cpp_standalone binary ...')
    t0 = _wall.time()
    net.run(simtime)
    device.build(run=False, directory=None)
    t_compile = _wall.time() - t0
    print(f'[pass 2] Compilation finished in {t_compile:.1f} s.')

    # ── Unpack parameter vector ───────────────────────────────────────────────
    area = net['Neuron'].namespace['area']

    run_args = {
        net['Synapse'].U_0_ar:     params[5],
        net['Synapse'].Umax:       params[6] / ms,
        net['Synapse'].U_0_sr:     params[7],
        net['Synapse'].Omega_f_sr: params[8] / second,
        net['Synapse'].Omega_f_ar: params[9] / second,
        net['Synapse'].Omega_d:    params[10] / second,
        net['Synapse'].alpha_syn:  params[11],
        net['Synapse'].Xi_ampa:    params[2] / mmole,
        net['Synapse'].Xi_nmda:    params[3] / mmole,
        net['Neuron'].sigma:       params[0] * mV,
        net['Neuron'].g_AHP:       params[1] * nS,
        net['Neuron'].tau_Ca:      params[4] * second,
        net['Neuron'].g_na:        params[12] * msiemens * cm**-2 * area,
        net['Neuron'].g_kd:        params[13] * msiemens * cm**-2 * area,
    }

    # ── Execute ───────────────────────────────────────────────────────────────
    print(f'[pass 2] Executing binary '
          f'(simtime={args.simtime:.1f} s, seed_run={args.seed_run}) ...')
    t0 = _wall.time()

    if args.seed_run is None:
        device.run(run_args=run_args)
    else:
        device.run(run_args=run_args, seed=args.seed_run)

    t_run = _wall.time() - t0
    ratio = t_run / args.simtime
    print(f'[pass 2] Done in {t_run:.1f} s  '
          f'(wall-clock/simulated ratio = {ratio:.2f}).')

    # ── Harvest spikes ────────────────────────────────────────────────────────
    spk_N_t = np.asarray(net['Spike_monitor_N'].t / second, dtype=np.float32)
    spk_N_i = np.asarray(net['Spike_monitor_N'].i,          dtype=np.int32)

    if SpikesA is not None:
        spk_A_t = np.asarray(net['Spike_monitor_A'].t / second, dtype=np.float32)
        spk_A_i = np.asarray(net['Spike_monitor_A'].i,          dtype=np.int32)
    else:
        spk_A_t = np.array([], dtype=np.float32)
        spk_A_i = np.array([], dtype=np.int32)

    return dict(
        spk_N_t=spk_N_t, spk_N_i=spk_N_i,
        spk_A_t=spk_A_t, spk_A_i=spk_A_i,
        t_compile_s=t_compile, t_run_s=t_run,
    )


# =============================================================================
# Plots
# =============================================================================

_C_NEURON = '#1f4e79'
_C_ASTRO  = '#7b3f00'
_C_GRID   = '#d0d0d0'


def _style_axis(ax, simtime_s, n_units, ylabel, color):
    ax.set_xlim(0.0, simtime_s)
    ax.set_ylim(-0.5, max(n_units - 0.5, 0.5))
    ax.set_xlabel('Time (s)', fontsize=11, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=4)
    ax.tick_params(axis='both', labelsize=9)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.grid(axis='x', color=_C_GRID, linewidth=0.6, linestyle='--', zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(color)
    ax.spines['bottom'].set_color(color)
    ax.tick_params(colors=color)
    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)


def _add_rate_trace(ax, spk_t, simtime_s, n_units, color, bin_s, label):
    """Overlay a population mean firing-rate trace on a twin right axis."""
    bins   = np.arange(0.0, simtime_s + bin_s, bin_s)
    counts, _ = np.histogram(spk_t, bins=bins)
    rate   = counts / (bin_s * max(n_units, 1))
    t_c    = 0.5 * (bins[:-1] + bins[1:])

    ax_r = ax.twinx()
    ax_r.plot(t_c, rate, color=color, alpha=0.45, linewidth=1.2, label=label)
    ax_r.set_ylabel('Mean FR (Hz)', fontsize=9, color=color, labelpad=4)
    ax_r.tick_params(axis='y', labelsize=8, colors=color)
    ax_r.set_ylim(bottom=0.0)
    for spine in ('top', 'left', 'bottom'):
        ax_r.spines[spine].set_visible(False)
    ax_r.spines['right'].set_color(color)
    return ax_r


def plot_neuronal_raster(spk_N_t, spk_N_i, simtime_s, Nn, params,
                         out_path, dpi=200):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if len(spk_N_t):
        cmap  = plt.get_cmap('Blues')
        norm  = plt.Normalize(0, max(Nn - 1, 1))
        ax.scatter(spk_N_t, spk_N_i, c=cmap(norm(spk_N_i)),
                   s=2.5, linewidths=0, alpha=0.7, zorder=3, rasterized=True)
        _add_rate_trace(ax, spk_N_t, simtime_s, Nn,
                        _C_NEURON, bin_s=1.0, label='Mean FR')
    else:
        ax.text(0.5, 0.5, 'No neuronal spikes detected',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')

    _style_axis(ax, simtime_s, Nn, 'Neuron index', _C_NEURON)

    n_spk   = len(spk_N_t)
    mean_fr = n_spk / (simtime_s * max(Nn, 1))
    ax.text(0.99, 0.97,
            f'{n_spk:,} spikes — mean FR = {mean_fr:.2f} Hz',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=8.5, color=_C_NEURON,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=2))

    title_p = (f'σ={params[0]:.2f} mV   g_AHP={params[1]:.1f} nS   '
               f'g_na={params[12]:.1f}   g_kd={params[13]:.1f}   '
               f'α_syn={params[11]:.2f}')
    ax.set_title(f'Neuronal raster — {Nn} neurons, {simtime_s:.0f} s\n{title_p}',
                 fontsize=10, pad=8, color=_C_NEURON)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] Neuronal raster → {out_path}')


def plot_astrocyte_raster(spk_A_t, spk_A_i, simtime_s, Na, params,
                          out_path, dpi=200):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    if len(spk_A_t):
        cmap = plt.get_cmap('Oranges')
        norm = plt.Normalize(0, max(Na - 1, 1))
        ax.scatter(spk_A_t, spk_A_i, c=cmap(norm(spk_A_i)),
                   s=22, marker='D', linewidths=0.4,
                   edgecolors='#4a2000', alpha=0.85,
                   zorder=3, rasterized=True)
        _add_rate_trace(ax, spk_A_t, simtime_s, Na,
                        _C_ASTRO, bin_s=5.0, label='Event rate')
    else:
        ax.text(0.5, 0.5, 'No astrocyte Ca²⁺ events detected',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')

    _style_axis(ax, simtime_s, Na, 'Astrocyte index', _C_ASTRO)

    n_ev   = len(spk_A_t)
    mean_r = n_ev / (simtime_s * max(Na, 1))
    ax.text(0.99, 0.97,
            f'{n_ev:,} events — mean rate = {mean_r:.4f} Hz',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=8.5, color=_C_ASTRO,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=2))

    title_p = (f'Tau_Ca={params[4]:.1f} s   '
               f'Xi_ampa={params[2]:.2f}   Xi_nmda={params[3]:.2f}   '
               f'α_syn={params[11]:.2f}')
    ax.set_title(f'Astrocyte Ca²⁺ events — {Na} cells, {simtime_s:.0f} s\n{title_p}',
                 fontsize=10, pad=8, color=_C_ASTRO)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] Astrocyte raster → {out_path}')


def plot_spatial_layout(topo: dict, c_max: float, mode: str,
                        out_path: str, dpi: int = 200):
    """
    Single-panel top-down view of the network topology.

    Drawn elements (back to front):
        • arena boundary
        • gap-junction links between astrocytes      (faint amber lines)
        • synapse → astrocyte links                  (faint olive lines)
        • synapse boutons                            (small red dots)
        • astrocytes                                 (amber diamonds)
        • neurons                                    (navy circles)

    All inputs are plain numpy arrays from `build_topology`; no Brian2.
    """
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ── Arena boundary ───────────────────────────────────────────────────────
    ax.add_patch(Rectangle((0.0, 0.0), c_max, c_max,
                           fill=False, edgecolor='#888888',
                           linewidth=1.0, linestyle='--', alpha=0.6, zorder=0))

    # ── Connectivity (drawn first so markers sit on top) ─────────────────────
    # Gap-junction links: deduplicate (i, j) since they were stored bidirectionally.
    if mode == 'Full' and len(topo['GJ_i']):
        seen = set()
        for i, j in zip(topo['GJ_i'], topo['GJ_j']):
            key = (int(min(i, j)), int(max(i, j)))
            if key in seen:
                continue
            seen.add(key)
            ax.plot([topo['A_pos'][i, 0], topo['A_pos'][j, 0]],
                    [topo['A_pos'][i, 1], topo['A_pos'][j, 1]],
                    color=_C_ASTRO, alpha=0.25, linewidth=0.7, zorder=1)
        n_gj_unique = len(seen)
    else:
        n_gj_unique = 0

    # Synapse → astrocyte links.
    if mode == 'Full' and len(topo['StoA_i']):
        for si, aj in zip(topo['StoA_i'], topo['StoA_j']):
            ax.plot([topo['S_x_syn'][si], topo['A_pos'][aj, 0]],
                    [topo['S_y_syn'][si], topo['A_pos'][aj, 1]],
                    color='#8a6a00', alpha=0.18, linewidth=0.4, zorder=1)

    # ── Markers ──────────────────────────────────────────────────────────────
    n_syn = len(topo['S_i'])
    if n_syn:
        ax.scatter(topo['S_x_syn'], topo['S_y_syn'],
                   s=6, c='#c62828', alpha=0.55, marker='.',
                   linewidths=0, zorder=2,
                   label=f'Synapses ({n_syn})',
                   rasterized=True)

    if mode == 'Full' and len(topo['A_pos']):
        ax.scatter(topo['A_pos'][:, 0], topo['A_pos'][:, 1],
                   s=80, c=_C_ASTRO, alpha=0.85, marker='D',
                   edgecolors='#3a1900', linewidths=0.7, zorder=3,
                   label=f'Astrocytes ({len(topo["A_pos"])})')

    ax.scatter(topo['N_pos'][:, 0], topo['N_pos'][:, 1],
               s=110, c=_C_NEURON, alpha=0.88, marker='o',
               edgecolors='#0a2a40', linewidths=0.7, zorder=4,
               label=f'Neurons ({len(topo["N_pos"])})')

    # ── Frame ────────────────────────────────────────────────────────────────
    pad = 0.05 * c_max
    ax.set_xlim(-pad, c_max + pad)
    ax.set_ylim(-pad, c_max + pad)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('x (µm)', fontsize=11)
    ax.set_ylabel('y (µm)', fontsize=11)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, color='#e8e8e8', linewidth=0.5, linestyle=':', zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')

    # Title with topology summary
    if mode == 'Full':
        title = (f'Network spatial layout — '
                 f'{len(topo["N_pos"])} N · {len(topo["A_pos"])} A · '
                 f'{n_syn} synapses · {n_gj_unique} GJ pairs · '
                 f'{len(topo["StoA_i"])} StoA links')
    else:
        title = (f'Network spatial layout (Neuronal mode) — '
                 f'{len(topo["N_pos"])} neurons · {n_syn} synapses')
    ax.set_title(title, fontsize=11, pad=10, color='#222222')

    leg = ax.legend(loc='upper right', fontsize=9, framealpha=0.92,
                    facecolor='white', edgecolor='#cccccc')
    leg.set_zorder(5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] Spatial layout → {out_path}')


# =============================================================================
# Summary
# =============================================================================

def write_summary(path, args, params, topo, results):
    n_syn  = len(topo['S_i'])
    n_gj   = len(topo['GJ_i'])
    n_stoa = len(topo['StoA_i'])

    lines = [
        '=' * 62,
        'HPC_single_run — run summary',
        '=' * 62,
        '',
        '  Network',
        f'    Mode              : {args.mode}',
        f'    Nn / Na           : {args.Nn} / {args.Na}',
        f'    Synapses          : {n_syn}  '
        f'(p_eff = {n_syn / max(args.Nn*(args.Nn-1), 1):.4f}, '
        f'target = {args.conn_prob:.4f})',
        f'    Gap junctions     : {n_gj}',
        f'    StoA links        : {n_stoa}',
        '',
        '  Connectivity geometry',
        f'    Arena             : [{args.c_max:.0f} µm]²',
        f'    conn_prob         : {args.conn_prob}',
        f'    bouton placement  : chi²(4) Sholl PDF from synapse_pdist.csv',
        f'    displ_bias        : {args.displ_bias} µm',
        f'    gj_dist           : {args.gj_dist} µm',
        f'    stoa_cutoff       : {args.stoa_cutoff} µm',
        f'    stoa_sigma        : {args.stoa_sigma} µm',
        '',
        '  Simulation',
        f'    Simtime           : {args.simtime:.1f} s',
        f'    seed_device       : {args.seed_device}',
        f'    seed_neuron       : {args.seed_neuron}',
        f'    seed_synapse      : {args.seed_synapse}',
        f'    seed_astro        : {args.seed_astro}',
        f'    seed_run          : {args.seed_run}',
        '',
        '  Parameter vector',
    ]

    for name, unit, val in zip(PARAM_NAMES, PARAM_UNITS, params):
        lines.append(f'    {name:<14s} = {val:>12.6g}  [{unit}]')

    Nn, Na = args.Nn, args.Na
    spk_N_t = results['spk_N_t']
    spk_A_t = results['spk_A_t']

    lines += [
        '',
        '  Spike statistics',
        f'    Neuronal spikes   : {len(spk_N_t):,}',
        f'    Mean neuron FR    : {len(spk_N_t) / (args.simtime * max(Nn, 1)):.4f} Hz',
        f'    Astro events      : {len(spk_A_t):,}',
        f'    Mean astro rate   : '
        f'{len(spk_A_t) / (args.simtime * max(Na, 1)):.6f} Hz',
        '',
        '  Wall-clock times',
        f'    Pass 1 (topology) : {results.get("t_topo_s", 0.0):.2f} s',
        f'    Pass 2 compile    : {results["t_compile_s"]:.1f} s',
        f'    Pass 2 run        : {results["t_run_s"]:.1f} s',
        f'    wc/sim ratio      : {results["t_run_s"] / args.simtime:.3f}',
        '',
        '  Output directory',
        f'    {args.out_dir}',
        '',
        '=' * 62,
    ]

    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'[summary] Written → {path}')


# =============================================================================
# Entry point
# =============================================================================

def _resolve_syn_pdist_csv(args) -> 'pandas.DataFrame':
    """
    Locate and load synapse_pdist.csv.

    Resolution order:
      1. explicit --syn_pdist_csv path
      2. <lib_dir>/synapse_pdist.csv
      3. <script_dir>/synapse_pdist.csv
    """
    import pandas as pd

    candidates = []
    if args.syn_pdist_csv is not None:
        candidates.append(args.syn_pdist_csv)
    lib_dir    = args.lib_dir or os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(lib_dir,    'synapse_pdist.csv'))
    candidates.append(os.path.join(script_dir, 'synapse_pdist.csv'))

    for path in candidates:
        if path and os.path.isfile(path):
            print(f'[pass 1] Loading synapse_pdist.csv from: {path}')
            df = pd.read_csv(path)
            for required in ('Syn_prob', 'Radius_val'):
                if required not in df.columns:
                    raise ValueError(
                        f"{path}: missing required column '{required}'. "
                        f"Got columns: {list(df.columns)}")
            return df

    raise FileNotFoundError(
        'synapse_pdist.csv not found. Searched: '
        + '; '.join(c for c in candidates if c)
        + '. Provide it via --syn_pdist_csv PATH or place it next to '
          'ASD_fun_BD_cpp.py.'
    )


def main():
    parser = build_parser()
    args   = parser.parse_args()
    params = resolve_params(args)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load Syn_pdist CSV (required for chi²(4) Sholl bouton placement) ─────
    syn_prob_csv = _resolve_syn_pdist_csv(args)

    # ── Pass 1: topology ─────────────────────────────────────────────────────
    t0_topo = _wall.time()
    topo = build_topology(
        Nn=args.Nn, Na=(args.Na if args.mode == 'Full' else 0),
        c_max=args.c_max,
        conn_prob=args.conn_prob,
        syn_prob_csv=syn_prob_csv,
        displ_bias=args.displ_bias,
        gj_dist=args.gj_dist,
        stoa_cutoff=args.stoa_cutoff,
        stoa_sigma=args.stoa_sigma,
        seed_neuron=args.seed_neuron,
        seed_synapse=args.seed_synapse,
        seed_astro=args.seed_astro,
        mode=args.mode,
    )
    t_topo = _wall.time() - t0_topo
    save_topology(topo, args.out_dir)

    # ── Spatial layout plot (independent of pass 2 — done up-front so the
    #    user has a visual sanity check even if the simulation later fails) ──
    plot_spatial_layout(
        topo, c_max=args.c_max, mode=args.mode,
        out_path=os.path.join(args.out_dir, 'spatial_layout.png'),
        dpi=args.dpi,
    )

    # ── Pass 2: build, compile, run ───────────────────────────────────────────
    results = build_and_run(topo, args, params)
    results['t_topo_s'] = t_topo

    # ── Save spike data ───────────────────────────────────────────────────────
    npz_path = os.path.join(args.out_dir, 'iter_0000000.npz')
    np.savez_compressed(
        npz_path,
        params=params,
        spk_N_t=results['spk_N_t'],
        spk_N_i=results['spk_N_i'],
        spk_A_t=results['spk_A_t'],
        spk_A_i=results['spk_A_i'],
        seed_run=np.int64(-1 if args.seed_run is None else args.seed_run),
    )
    print(f'[data]  Spike data saved → {npz_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_neuronal_raster(
        results['spk_N_t'], results['spk_N_i'],
        args.simtime, args.Nn, params,
        out_path=os.path.join(args.out_dir, 'raster_neurons.png'),
        dpi=args.dpi,
    )
    if args.mode == 'Full':
        plot_astrocyte_raster(
            results['spk_A_t'], results['spk_A_i'],
            args.simtime, args.Na, params,
            out_path=os.path.join(args.out_dir, 'raster_astrocytes.png'),
            dpi=args.dpi,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(
        os.path.join(args.out_dir, 'run_summary.txt'),
        args, params, topo, results,
    )

    print('\n[done] All outputs written to', args.out_dir)


if __name__ == '__main__':
    main()
