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
    'Sigma', 'gbarA',                            # 0-1   Neuron  (was Sigma, g_AHP)
    'EC50_ampa', 'EC50_nmda', 'tauA',            # 2-4   Syn, Syn, Neuron (was Tau_Ca)
    'U_0_ar', 'U_max', 'U_0_sr',                 # 5-7   Synapse
    'Omega_f_sr', 'Omega_f_ar', 'Omega_d',       # 8-10  Synapse
    'alpha_syn',                                 # 11    Synapse
    'DeltaT', 'VT',                              # 12-13 Neuron  (was g_na, g_kd)
    'g_ampa', 'g_nmda',                          # 14-15 Synapse (set on neuron)
    'delta_gA',                                  # 16    Neuron  (was alpha_Ca)
    'x0',                                        # 17    Synapse
    'O_G', 'Omega_G',                            # 18-19 Synapse
    'O_beta', 'O_3K', 'Omega_5P', 'I_bias',      # 20-23 Astrocyte
    'F', 'I_Theta', 'omega_I',                   # 24-26 Astrocyte
    'C_Theta', 'U_A', 'G_T',                     # 27-29 Gliot_release
    'gL', 'VA', 'DeltaA', 'VR',                  # 30-33 Neuron  (NEW: CAdEx)
]

PARAM_UNITS = [
    'mV', 'nS',
    'mmole', 'mmole', 'ms',
    '(dimensionless)', '1/ms', '(dimensionless)',
    '1/s', '1/s', '1/s',
    '(dimensionless)',
    'mV', 'mV',
    'nS', 'nS',
    'nS',
    '(dimensionless)',
    '1/(uM·s)', '1/s',
    'uM/s', 'uM/s', '1/s', 'uM',
    'uM/s', 'uM', 'uM',
    'uM', '(dimensionless)', 'mM',
    'nS', 'mV', 'mV', 'mV',
]

# Nominal parameter vector.  Each entry is used as the default when the
# corresponding CLI flag is omitted.
NOMINAL_PARAMS = np.array([
    4.0,        # 0  Sigma          mV
    10.0,       # 1  gbarA          nS    (ḡ_A, max subthreshold adaptation)
    8.6,        # 2  EC50_ampa      mmole  (kappa/mu-reconciled; raw P&K 27.2)
    3.0,        # 3  EC50_nmda      mmole  (kappa/mu-reconciled; raw P&K  9.5)
    200.0,      # 4  tauA           ms    (τ_A, adaptation time constant)
    0.003,      # 5  U_0_ar         (dimensionless)
    0.5,        # 6  U_max          1/ms
    0.15,       # 7  U_0_sr         (dimensionless)
    2.0,        # 8  Omega_f_sr     1/s
    1.42857,    # 9  Omega_f_ar     1/s   (= 1/0.7)
    2.0,        # 10 Omega_d        1/s
    1.0,        # 11 alpha_syn      (dimensionless)
    2.0,        # 12 DeltaT         mV    (Δ_T, spike-initiation slope)
    -50.0,      # 13 VT             mV    (spike threshold)
    1.6,        # 14 g_ampa         nS
    0.4,        # 15 g_nmda         nS
    1.0,        # 16 delta_gA       nS    (δg_A, post-spike adaptation increment)
    0.2,        # 17 x0             (dimensionless)  quantal vesicle size
    1.5,        # 18 O_G            1/(uM·s)  mGluR binding rate
    0.00833,    # 19 Omega_G        1/s   mGluR inactivation (= 0.5/60)
    1.0,        # 20 O_beta         uM/s  PLCbeta gain
    4.5,        # 21 O_3K           uM/s  IP3-3K rate
    0.1,        # 22 Omega_5P       1/s   IP3-5P degradation
    0.8,        # 23 I_bias         uM    IP3 exogenous set-point
    2.0,        # 24 F              uM/s  GJ + exogenous permeability
    0.3,        # 25 I_Theta        uM    tanh threshold
    0.05,       # 26 omega_I        uM    tanh steepness
    0.5,        # 27 C_Theta        uM    exocytosis Ca2+ threshold
    0.6,        # 28 U_A            (dimensionless)  gliotransmitter release prob
    200.0,      # 29 G_T            mM    total gliotransmitter
    10.0,       # 30 gL             nS    (leak conductance g_L)
    -45.0,      # 31 VA             mV    (subthreshold adaptation activation)
    5.0,        # 32 DeltaA         mV    (Δ_A, subthreshold adaptation slope, > 0)
    -55.0,      # 33 VR             mV    (reset potential)
])

# =============================================================================
# Sweep groups — which axes are FREE for a given campaign
# =============================================================================
# Defined by NAME and resolved to indices via PARAM_NAMES, so they auto-track
# any future re-ordering of the parameter vector. A campaign run with
# --sweep_group <g> draws only SWEEP_GROUPS[<g>] from the prior; every other
# axis is frozen at its NOMINAL_PARAMS value (run_args still injects all of them).
NEURON_PARAMS  = ['Sigma', 'gbarA', 'delta_gA', 'tauA', 'DeltaT', 'VT',
                  'gL', 'VA', 'DeltaA', 'VR']                 # 10 CAdEx intrinsic axes
SYNAPSE_PARAMS = ['EC50_ampa', 'EC50_nmda', 'U_0_ar', 'U_max', 'U_0_sr',
                  'Omega_f_sr', 'Omega_f_ar', 'Omega_d', 'alpha_syn',
                  'g_ampa', 'g_nmda', 'x0', 'O_G', 'Omega_G']
ASTRO_PARAMS   = ['O_beta', 'O_3K', 'Omega_5P', 'I_bias', 'F', 'I_Theta',
                  'omega_I', 'C_Theta', 'U_A', 'G_T']


def _grp_idx(names):
    return sorted(PARAM_NAMES.index(n) for n in names)


SWEEP_GROUPS = {
    'all':            list(range(len(PARAM_NAMES))),
    'neuron':         _grp_idx(NEURON_PARAMS),
    'synapse':        _grp_idx(SYNAPSE_PARAMS),
    'astro':          _grp_idx(ASTRO_PARAMS),
    'neuron_synapse': _grp_idx(NEURON_PARAMS + SYNAPSE_PARAMS),
}

# Partition sanity: neuron + synapse + astro must tile {0..N-1} exactly once.
assert sorted(_grp_idx(NEURON_PARAMS + SYNAPSE_PARAMS + ASTRO_PARAMS)) \
       == list(range(len(PARAM_NAMES))), "sweep-group partition does not tile PARAM_NAMES"


def resolve_sweep_group(name):
    """Return the sorted list of FREE (active) axis indices for a group name."""
    if name not in SWEEP_GROUPS:
        raise ValueError(
            f"unknown --sweep_group {name!r}; choices: {sorted(SWEEP_GROUPS)}")
    return sorted(SWEEP_GROUPS[name])


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
    p.add_argument('--topology_mode', default='wallach',
                   choices=['wallach', 'distance'],
                   help='Rule for building astrocyte connectivity from cell '
                        'positions. "wallach" (Wallach et al. 2014, default) '
                        'uses the joint (neurons ∪ astrocytes) Voronoi '
                        'tessellation: GJC = Delaunay-adjacent astrocyte '
                        'pairs capped at --gj_max_dist; StoA = each bouton '
                        'is assigned to its nearest astrocyte within '
                        '--stoa_cutoff (strictly 1-to-1). "distance" is the '
                        'legacy rule (KDTree radius for GJC, per-astrocyte '
                        'Gaussian acceptance for StoA, may yield 1-syn → '
                        'many-astro). Default: %(default)s.')
    p.add_argument('--gj_dist', type=float, default=200.0, metavar='UM',
                   help='[topology_mode=distance only] KDTree radius for '
                        'gap-junction coupling between astrocytes '
                        '(µm, default: %(default)s).')
    p.add_argument('--gj_max_dist', type=float, default=150.0, metavar='UM',
                   help='[topology_mode=wallach only] Soft distance cap '
                        'applied on top of the Voronoi-adjacency rule; '
                        'astrocyte pairs farther than this are NOT GJ-coupled '
                        'even if their Voronoi cells are contiguous '
                        '(µm, default: %(default)s).')
    p.add_argument('--stoa_cutoff', type=float, default=70.0, metavar='UM',
                   help='Hard distance cutoff for synapse→astrocyte links '
                        '(µm, default: %(default)s).')
    p.add_argument('--stoa_sigma', type=float, default=200.0, metavar='UM',
                   help='[topology_mode=distance only] σ of the Gaussian '
                        'connection probability for synapse→astrocyte links '
                        '(µm, default: %(default)s).')

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
    g.add_argument('--EC50_ampa',  type=float, default=None, metavar='mmole')
    g.add_argument('--EC50_nmda',  type=float, default=None, metavar='mmole')
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
    # NEW axes (idx 14-29)
    g.add_argument('--g_ampa',     type=float, default=None, metavar='nS')
    g.add_argument('--g_nmda',     type=float, default=None, metavar='nS')
    g.add_argument('--alpha_Ca',   type=float, default=None)
    g.add_argument('--x0',         type=float, default=None)
    g.add_argument('--O_G',        type=float, default=None, metavar='1/(uM·s)')
    g.add_argument('--Omega_G',    type=float, default=None, metavar='1/s')
    g.add_argument('--O_beta',     type=float, default=None, metavar='uM/s')
    g.add_argument('--O_3K',       type=float, default=None, metavar='uM/s')
    g.add_argument('--Omega_5P',   type=float, default=None, metavar='1/s')
    g.add_argument('--I_bias',     type=float, default=None, metavar='uM')
    # NB: --F_gj (not --F) to avoid collision with argparse abbreviation / builtins;
    # maps internally to params[24] and net['Astrocyte'].F.
    g.add_argument('--F_gj',       type=float, default=None, metavar='uM/s',
                   dest='F_gj', help='Astrocyte GJ+exogenous permeability F (uM/s).')
    g.add_argument('--I_Theta',    type=float, default=None, metavar='uM')
    g.add_argument('--omega_I',    type=float, default=None, metavar='uM')
    g.add_argument('--C_Theta',    type=float, default=None, metavar='uM')
    g.add_argument('--U_A',        type=float, default=None)
    g.add_argument('--G_T',        type=float, default=None, metavar='mM')

    # ── Figure options ────────────────────────────────────────────────────────
    p.add_argument('--dpi', type=int, default=200,
                   help='DPI for saved figures (default: %(default)s).')

    return p


def resolve_params(args) -> np.ndarray:
    """Merge CLI-supplied values onto the nominal parameter vector."""
    p = NOMINAL_PARAMS.copy()
    cli_values = [
        args.Sigma, args.g_AHP,
        args.EC50_ampa, args.EC50_nmda, args.Tau_Ca,
        args.U_0_ar, args.U_max, args.U_0_sr,
        args.Omega_f_sr, args.Omega_f_ar, args.Omega_d,
        args.alpha_syn, args.g_na, args.g_kd,
        args.g_ampa, args.g_nmda,
        args.alpha_Ca,
        args.x0,
        args.O_G, args.Omega_G,
        args.O_beta, args.O_3K, args.Omega_5P, args.I_bias,
        args.F_gj, args.I_Theta, args.omega_I,
        args.C_Theta, args.U_A, args.G_T,
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
                   mode='Full',
                   topology_mode='wallach',
                   gj_max_dist=150.0) -> dict:
    """
    Build the full network topology using plain NumPy / SciPy.
    No Brian2 device is touched.

    Bouton placement follows the morphometric chi²(4) Sholl distribution
    loaded from `syn_prob_csv` (columns: Syn_prob, Radius_val).  For each
    synapse the distance from the post-soma is drawn from that PDF (plus
    `displ_bias` µm) and the bouton is placed along the line connecting
    the pre and post somata, moving toward the pre-soma — exactly
    mirroring `get_synapse_coordinates` in the library.

    Two topology rules for astrocyte connectivity are supported, selected
    by `topology_mode`:

      * 'wallach' (default, Wallach et al. 2014):
        GJC: Delaunay-adjacency in the JOINT (neurons ∪ astrocytes) Voronoi
             tessellation, capped at `gj_max_dist` µm.
        StoA: each bouton is assigned to its NEAREST astrocyte within
              `stoa_cutoff` µm — strictly 1 astrocyte per synapse.
        The `gj_dist` and `stoa_sigma` parameters are ignored.

      * 'distance' (legacy):
        GJC: every astrocyte pair within `gj_dist` µm is connected (KDTree
             radius query).
        StoA: every (synapse, astrocyte) pair within `stoa_cutoff` µm is
              independently accepted with probability
              p = exp(-d² / (2·stoa_sigma²)). A synapse may be linked to
              multiple astrocytes under this rule.
        The `gj_max_dist` parameter is ignored.

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
        _log_topology(topo, _wall.time() - t0, topology_mode=topology_mode)
        return topo

    # ── Astrocyte positions ───────────────────────────────────────────────────
    A_pos = rng_a.uniform(0.0, c_max, (Na, 2))

    # Validate topology_mode early — fail fast in pass 1 rather than partway
    # through pass 2.
    if topology_mode not in ('wallach', 'distance'):
        raise ValueError(
            f"build_topology: unknown topology_mode={topology_mode!r}. "
            "Expected 'wallach' or 'distance'."
        )

    # ── Gap-junction connectivity ─────────────────────────────────────────────
    if topology_mode == 'wallach':
        # Joint Voronoi (≡ Delaunay) over neurons ∪ astrocytes; keep only
        # astrocyte–astrocyte edges; soft cap at `gj_max_dist` µm.
        # Wallach et al. 2014, Fig. 2; De Pittà & Berry 2019, ch. 7, eq. (7.9).
        from ASD_fun_BD_cpp import voronoi_astro_gj_pairs
        GJ_i, GJ_j = voronoi_astro_gj_pairs(N_pos, A_pos,
                                            gj_max_dist=gj_max_dist)
    else:
        # Legacy KDTree radius rule — every pair within `gj_dist` µm.
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
    bouton_pos = np.column_stack([S_x_syn, S_y_syn])    # (n_syn, 2)

    if topology_mode == 'wallach':
        # Each bouton → nearest astrocyte within `stoa_cutoff`. By
        # construction every synapse appears at most once in StoA_i.
        from ASD_fun_BD_cpp import nearest_astro_for_synapse
        StoA_i, StoA_j = nearest_astro_for_synapse(
            bouton_pos, A_pos, stoa_cutoff=stoa_cutoff
        )
    else:
        # Legacy per-astrocyte Gaussian acceptance — a synapse may end up
        # linked to multiple astrocytes.
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
    _log_topology(topo, _wall.time() - t0, topology_mode=topology_mode)
    return topo


def _log_topology(topo: dict, elapsed: float, topology_mode: str = '') -> None:
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
        mode_tag = f' ({topology_mode})' if topology_mode else ''
        print(f'         {Na} astrocytes  {n_gj} GJ links  '
              f'{n_stoa} StoA links{mode_tag}')


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
    params : np.ndarray  shape (30,)

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
        second, ms, mV, nS, mmole, umole, msiemens, cm, um,
        BrianLogger, Function, DEFAULT_FUNCTIONS, Equations,
        linked_var, float32,
    )

    set_device('cpp_standalone', build_on_run=False)
    device = get_device()
    device.reinit()
    device.activate(build_on_run=False)

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
        # ---- Neuron group: CAdEx intrinsic axes (present in both modes) ----
        net['Neuron'].sigma:       params[0]  * mV,
        net['Neuron'].gbarA:       params[1]  * nS,
        net['Neuron'].tauA:        params[4]  * ms,
        net['Neuron'].DeltaT:      params[12] * mV,
        net['Neuron'].VT:          params[13] * mV,
        net['Neuron'].delta_gA:    params[16] * nS,
        net['Neuron'].gl:          params[30] * nS,
        net['Neuron'].VA:          params[31] * mV,
        net['Neuron'].DeltaA:      params[32] * mV,
        net['Neuron'].VR:          params[33] * mV,
        net['Neuron'].g_ampa:      params[14] * nS,
        net['Neuron'].g_nmda:      params[15] * nS,
    }

    # ---- Astrocyte + gliotransmission axes: only in Full mode (groups absent
    #      in Neuronal mode; targeting them would KeyError). ----
    if args.mode == 'Full':
        run_args.update({
            net['Astrocyte'].O_beta:     params[20] * umole / second,
            net['Astrocyte'].O_3K:       params[21] * umole / second,
            net['Astrocyte'].Omega_5P:   params[22] / second,
            net['Astrocyte'].I_bias:     params[23] * umole,
            net['Astrocyte'].F:          params[24] * umole / second,
            net['Astrocyte'].I_Theta:    params[25] * umole,
            net['Astrocyte'].omega_I:    params[26] * umole,
            net['Gliot_release'].C_Theta: params[27] * umole,
            net['Gliot_release'].U_A:     params[28],
            net['Gliot_release'].G_T:     params[29] * mmole,
        })

    # ── Execute ───────────────────────────────────────────────────────────────
    print(f'[pass 2] Executing binary '
          f'(simtime={args.simtime:.1f} s, seed_run={args.seed_run}) ...')
    t0 = _wall.time()

    # device.run() has never accepted a `seed=` kwarg (it silently fails on
    # modern Brian2). The documented cpp_standalone idiom for reproducible
    # per-run noise is device.seed(value) called immediately BEFORE device.run()
    # — mirrors HPC_main_sweep._worker_entry. device.seed() is core API in every
    # modern Brian2 (>=2.x); we probe for it defensively and, lacking it, fall
    # back to replaying the build-time RNG sequence.
    if args.seed_run is not None and callable(getattr(device, 'seed', None)):
        device.seed(int(args.seed_run))
    device.run(run_args=run_args)

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
               f'EC50_ampa={params[2]:.2f}   EC50_nmda={params[3]:.2f}   '
               f'α_syn={params[11]:.2f}')
    ax.set_title(f'Astrocyte Ca²⁺ events — {Na} cells, {simtime_s:.0f} s\n{title_p}',
                 fontsize=10, pad=8, color=_C_ASTRO)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] Astrocyte raster → {out_path}')


# =============================================================================
# Topology visualisations
# =============================================================================
# Three complementary views are produced for every topology:
#   1) plot_spatial_layout  — 2D anatomy:    cells + boutons + Voronoi overlay
#   2) plot_spatial_connectivity — 2D wiring: cells + Voronoi + GJ + StoA links
#   3) plot_3d_layered_topology  — 3D layered: neurons / astrocytes z-stacked
#                                  with all link classes (single_run only)
# -----------------------------------------------------------------------------

# Visual palette — colour-blind safe (Wong, 2011), shared across all 3 plots.
_VIZ_NEURON_FACE  = '#1f4e79'   # deep blue
_VIZ_NEURON_EDGE  = '#0a2a40'
_VIZ_ASTRO_FACE   = '#2e7d32'   # green (Wallach Fig. 2 convention)
_VIZ_ASTRO_EDGE   = '#0e3010'
_VIZ_ASTRO_FILL   = '#a5d6a7'   # pale green fill for astrocyte Voronoi cells
_VIZ_NEURON_FILL  = '#fff7eb'   # warm cream fill for neuron Voronoi cells
_VIZ_BOUTON       = '#c62828'   # red dots
_VIZ_VORONOI      = '#9aa0a6'   # gray Voronoi ridges
_VIZ_GJ           = '#f9a825'   # amber GJ links
_VIZ_STOA         = '#6a1b9a'   # purple StoA links


def _voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram into finite polygons,
    by clipping rays to a large bounding circle of radius `radius`.

    Adapted from a widely-used SciPy cookbook recipe
    (https://gist.github.com/pv/8036995). Returns (regions, vertices) where
    `regions` is a list of vertex-index lists (one polygon per input point),
    and `vertices` is the corresponding (n_vertices, 2) coordinate array.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("voronoi_finite_polygons_2d requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    if radius is None:
        radius = float(vor.points.ptp().max()) * 2.0

    # Build ridge map: input point index → list of (other point, v1, v2)
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue                   # already finite

            # Compute the missing endpoint by extending the bisector.
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Sort polygon vertices counter-clockwise around their centroid.
        vs = np.asarray([new_vertices[v] for v in new_region])
        c  = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [new_region[i] for i in np.argsort(angles)]

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def _draw_voronoi_overlay(ax, N_pos, A_pos, c_max,
                          fill_astro=True, fill_neuron=False,
                          line_alpha=0.35, line_width=0.4):
    """
    Draw the JOINT Voronoi tessellation of neurons + astrocytes on `ax`,
    clipped to the [0, c_max]² arena. Astrocyte Voronoi cells are lightly
    tinted to make the astrocyte "anatomical domains" visible at a glance,
    in the spirit of Wallach et al. 2014, Fig. 2B/C.

    Cell positions must be supplied as (N, 2) numpy arrays in µm.
    """
    from scipy.spatial import Voronoi
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    Nn = int(len(N_pos))
    Na = int(len(A_pos))
    if Nn + Na < 3:
        return                                 # cannot build a Voronoi diagram

    points = np.vstack([N_pos, A_pos])
    try:
        vor = Voronoi(points)
    except Exception:
        return                                 # degenerate (all collinear etc.)

    regions, vertices = _voronoi_finite_polygons_2d(vor, radius=4.0 * c_max)

    # Clip polygons to the arena rectangle [0, c_max]² (Sutherland–Hodgman).
    arena = np.array([
        [0.0, 0.0], [c_max, 0.0], [c_max, c_max], [0.0, c_max]
    ])

    astro_patches  = []
    neuron_patches = []
    for pt_idx, region in enumerate(regions):
        if not region:
            continue
        poly = vertices[region]
        poly = _clip_polygon_to_rect(poly, 0.0, 0.0, c_max, c_max)
        if poly is None or len(poly) < 3:
            continue
        if pt_idx >= Nn:
            astro_patches.append(MplPolygon(poly, closed=True))
        else:
            neuron_patches.append(MplPolygon(poly, closed=True))

    if fill_neuron and neuron_patches:
        pc = PatchCollection(neuron_patches, facecolor=_VIZ_NEURON_FILL,
                             edgecolor=_VIZ_VORONOI,
                             linewidths=line_width, alpha=0.55, zorder=0.5)
        ax.add_collection(pc)
    if fill_astro and astro_patches:
        pc = PatchCollection(astro_patches, facecolor=_VIZ_ASTRO_FILL,
                             edgecolor=_VIZ_VORONOI,
                             linewidths=line_width, alpha=0.55, zorder=0.5)
        ax.add_collection(pc)

    if not (fill_astro or fill_neuron):
        # Pure outline mode — draw all polygons' edges in pale gray.
        pc = PatchCollection(astro_patches + neuron_patches,
                             facecolor='none',
                             edgecolor=_VIZ_VORONOI,
                             linewidths=line_width, alpha=line_alpha,
                             zorder=0.5)
        ax.add_collection(pc)


def _clip_polygon_to_rect(poly, xmin, ymin, xmax, ymax):
    """
    Sutherland–Hodgman polygon clipping against an axis-aligned rectangle.
    Returns the clipped polygon as an (n, 2) array, or None if the polygon
    lies entirely outside the rectangle.
    """
    def clip(subject, edge):
        if len(subject) == 0:
            return subject
        output = []
        s = subject[-1]
        for e in subject:
            if _inside(e, edge, xmin, ymin, xmax, ymax):
                if not _inside(s, edge, xmin, ymin, xmax, ymax):
                    output.append(_intersect(s, e, edge, xmin, ymin, xmax, ymax))
                output.append(e)
            elif _inside(s, edge, xmin, ymin, xmax, ymax):
                output.append(_intersect(s, e, edge, xmin, ymin, xmax, ymax))
            s = e
        return output

    subject = list(map(tuple, poly))
    for edge in ('left', 'right', 'bottom', 'top'):
        subject = clip(subject, edge)
        if not subject:
            return None
    return np.asarray(subject)


def _inside(p, edge, xmin, ymin, xmax, ymax):
    if edge == 'left':   return p[0] >= xmin
    if edge == 'right':  return p[0] <= xmax
    if edge == 'bottom': return p[1] >= ymin
    if edge == 'top':    return p[1] <= ymax
    raise ValueError(edge)


def _intersect(p1, p2, edge, xmin, ymin, xmax, ymax):
    # Parametric intersection of the segment p1→p2 with the clip edge.
    x1, y1 = p1
    x2, y2 = p2
    if   edge == 'left':   x, y = xmin, y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
    elif edge == 'right':  x, y = xmax, y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
    elif edge == 'bottom': x, y = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1), ymin
    elif edge == 'top':    x, y = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1), ymax
    return (x, y)


def _topology_summary_str(topo, mode, topology_mode=''):
    n_syn  = len(topo['S_i'])
    Nn     = len(topo['N_pos'])
    Na     = len(topo['A_pos']) if mode == 'Full' else 0
    n_gj_u = _count_undirected_pairs(topo.get('GJ_i', []), topo.get('GJ_j', []))
    n_stoa = len(topo.get('StoA_i', []))
    if mode == 'Full':
        tag = f' [{topology_mode}]' if topology_mode else ''
        return (f'{Nn} neurons · {Na} astrocytes · {n_syn} synapses · '
                f'{n_gj_u} GJ pairs · {n_stoa} StoA links{tag}')
    return f'{Nn} neurons · {n_syn} synapses (Neuronal mode)'


def _count_undirected_pairs(I, J):
    seen = set()
    for i, j in zip(I, J):
        seen.add((int(min(i, j)), int(max(i, j))))
    return len(seen)


def plot_spatial_layout(topo: dict, c_max: float, mode: str,
                        out_path: str, dpi: int = 200,
                        topology_mode: str = '',
                        also_connectivity: bool = True):
    """
    Anatomy view (2D top-down): joint Voronoi tessellation of neurons +
    astrocytes (astrocyte cells tinted pale green), with all somata and
    boutons drawn on top. This is the "what does the tissue look like?"
    plot — connectivity is drawn on a separate companion figure (see
    `plot_spatial_connectivity` below), unless `also_connectivity=False`.

    Output written to `out_path`. If `also_connectivity=True` (default), the
    companion connectivity plot is written to the same directory with the
    filename derived from `out_path` (e.g. spatial_layout.png →
    spatial_connectivity.png).
    """
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    # ── Voronoi tessellation (neurons ∪ astrocytes) ─────────────────────────
    if mode == 'Full' and len(topo['A_pos']):
        _draw_voronoi_overlay(
            ax, topo['N_pos'], topo['A_pos'], c_max,
            fill_astro=True, fill_neuron=False,
            line_alpha=0.5, line_width=0.4,
        )

    # ── Arena boundary ───────────────────────────────────────────────────────
    ax.add_patch(Rectangle((0.0, 0.0), c_max, c_max,
                           fill=False, edgecolor='#666666',
                           linewidth=1.0, linestyle='--', alpha=0.7, zorder=1))

    # ── Boutons (small red dots) ─────────────────────────────────────────────
    n_syn = len(topo['S_i'])
    if n_syn:
        ax.scatter(topo['S_x_syn'], topo['S_y_syn'],
                   s=5, c=_VIZ_BOUTON, alpha=0.55, marker='.',
                   linewidths=0, zorder=2.5,
                   label=f'Boutons ({n_syn})',
                   rasterized=True)

    # ── Somata ───────────────────────────────────────────────────────────────
    if mode == 'Full' and len(topo['A_pos']):
        ax.scatter(topo['A_pos'][:, 0], topo['A_pos'][:, 1],
                   s=85, c=_VIZ_ASTRO_FACE, alpha=0.9, marker='D',
                   edgecolors=_VIZ_ASTRO_EDGE, linewidths=0.8, zorder=4,
                   label=f'Astrocytes ({len(topo["A_pos"])})')

    ax.scatter(topo['N_pos'][:, 0], topo['N_pos'][:, 1],
               s=70, c=_VIZ_NEURON_FACE, alpha=0.9, marker='o',
               edgecolors=_VIZ_NEURON_EDGE, linewidths=0.7, zorder=4,
               label=f'Neurons ({len(topo["N_pos"])})')

    # ── Frame & cosmetics ────────────────────────────────────────────────────
    pad = 0.04 * c_max
    ax.set_xlim(-pad, c_max + pad)
    ax.set_ylim(-pad, c_max + pad)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('x (µm)', fontsize=11)
    ax.set_ylabel('y (µm)', fontsize=11)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, color='#ececec', linewidth=0.5, linestyle=':', zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')

    ax.set_title('Network anatomy — joint Voronoi tessellation\n'
                 + _topology_summary_str(topo, mode, topology_mode),
                 fontsize=11, pad=10, color='#222222')

    leg = ax.legend(loc='upper right', fontsize=9, framealpha=0.92,
                    facecolor='white', edgecolor='#cccccc')
    leg.set_zorder(5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] Spatial anatomy → {out_path}')

    # Companion connectivity figure
    if also_connectivity:
        base, ext = os.path.splitext(out_path)
        if base.endswith('_layout'):
            base = base[: -len('_layout')]
        conn_path = base + '_connectivity' + ext
        plot_spatial_connectivity(topo, c_max=c_max, mode=mode,
                                  out_path=conn_path, dpi=dpi,
                                  topology_mode=topology_mode)


def plot_spatial_connectivity(topo: dict, c_max: float, mode: str,
                              out_path: str, dpi: int = 200,
                              topology_mode: str = ''):
    """
    Connectivity view (2D top-down). Same arena + same Voronoi tessellation
    as `plot_spatial_layout`, but cells and boutons are desaturated and the
    GJ links (astrocyte ↔ astrocyte, amber) plus the StoA links (bouton →
    astrocyte, purple) are drawn on top to make the wiring legible.
    """
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(9.0, 9.0))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    # Faint Voronoi backdrop (outline only, no fill).
    if mode == 'Full' and len(topo['A_pos']):
        _draw_voronoi_overlay(
            ax, topo['N_pos'], topo['A_pos'], c_max,
            fill_astro=False, fill_neuron=False,
            line_alpha=0.28, line_width=0.35,
        )

    ax.add_patch(Rectangle((0.0, 0.0), c_max, c_max,
                           fill=False, edgecolor='#666666',
                           linewidth=1.0, linestyle='--', alpha=0.7, zorder=1))

    # ── StoA links (bouton → astrocyte, drawn first, lightest) ──────────────
    if mode == 'Full' and len(topo.get('StoA_i', [])):
        for si, aj in zip(topo['StoA_i'], topo['StoA_j']):
            ax.plot([topo['S_x_syn'][si], topo['A_pos'][aj, 0]],
                    [topo['S_y_syn'][si], topo['A_pos'][aj, 1]],
                    color=_VIZ_STOA, alpha=0.35, linewidth=0.45, zorder=2)

    # ── GJ links (deduplicate bidirectional) ────────────────────────────────
    n_gj_unique = 0
    if mode == 'Full' and len(topo.get('GJ_i', [])):
        seen = set()
        for i, j in zip(topo['GJ_i'], topo['GJ_j']):
            key = (int(min(i, j)), int(max(i, j)))
            if key in seen:
                continue
            seen.add(key)
            ax.plot([topo['A_pos'][i, 0], topo['A_pos'][j, 0]],
                    [topo['A_pos'][i, 1], topo['A_pos'][j, 1]],
                    color=_VIZ_GJ, alpha=0.85, linewidth=1.6, zorder=3)
        n_gj_unique = len(seen)

    # ── Desaturated somata + boutons (background context) ───────────────────
    n_syn = len(topo['S_i'])
    if n_syn:
        ax.scatter(topo['S_x_syn'], topo['S_y_syn'],
                   s=3, c=_VIZ_BOUTON, alpha=0.25, marker='.',
                   linewidths=0, zorder=2.5, rasterized=True)

    if mode == 'Full' and len(topo['A_pos']):
        ax.scatter(topo['A_pos'][:, 0], topo['A_pos'][:, 1],
                   s=85, c=_VIZ_ASTRO_FACE, alpha=0.85, marker='D',
                   edgecolors=_VIZ_ASTRO_EDGE, linewidths=0.8, zorder=4)

    ax.scatter(topo['N_pos'][:, 0], topo['N_pos'][:, 1],
               s=55, c=_VIZ_NEURON_FACE, alpha=0.55, marker='o',
               edgecolors=_VIZ_NEURON_EDGE, linewidths=0.5, zorder=3.5)

    # ── Legend (proxy artists for the links) ────────────────────────────────
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([], [], color=_VIZ_GJ, linewidth=2.0,
               label=f'Astrocyte–astrocyte GJ ({n_gj_unique})'),
        Line2D([], [], color=_VIZ_STOA, linewidth=1.0, alpha=0.7,
               label=f'Bouton→astrocyte ({len(topo.get("StoA_i", []))})'),
        Line2D([], [], color=_VIZ_NEURON_FACE, marker='o', linewidth=0,
               markeredgecolor=_VIZ_NEURON_EDGE,
               label=f'Neurons ({len(topo["N_pos"])})'),
    ]
    if mode == 'Full' and len(topo['A_pos']):
        legend_items.insert(
            2,
            Line2D([], [], color=_VIZ_ASTRO_FACE, marker='D', linewidth=0,
                   markeredgecolor=_VIZ_ASTRO_EDGE,
                   label=f'Astrocytes ({len(topo["A_pos"])})'),
        )

    pad = 0.04 * c_max
    ax.set_xlim(-pad, c_max + pad)
    ax.set_ylim(-pad, c_max + pad)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('x (µm)', fontsize=11)
    ax.set_ylabel('y (µm)', fontsize=11)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, color='#ececec', linewidth=0.5, linestyle=':', zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')

    ax.set_title('Network connectivity — GJ + synapse→astrocyte links\n'
                 + _topology_summary_str(topo, mode, topology_mode),
                 fontsize=11, pad=10, color='#222222')

    leg = ax.legend(handles=legend_items, loc='upper right', fontsize=9,
                    framealpha=0.92, facecolor='white', edgecolor='#cccccc')
    leg.set_zorder(5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] Spatial connectivity → {out_path}')


def plot_3d_layered_topology(topo: dict, c_max: float, mode: str,
                             out_path: str, dpi: int = 200,
                             topology_mode: str = ''):
    """
    3D layered view of the topology. Neurons live in the z = 0 plane,
    astrocytes in the z = +z_astro plane. All four link classes are drawn:

        • neuron ↔ neuron synapses   — green dashed, very faint
        • astrocyte ↔ astrocyte GJC  — amber, thick
        • bouton → astrocyte         — purple, thin, crossing planes
        • bouton dots                — red, in the neuron plane

    Cells in each plane are also outlined with their joint Voronoi
    tessellation, projected onto the corresponding z-plane.

    Pure numpy — no Brian2 dependency.
    """
    from mpl_toolkits.mplot3d import Axes3D                       # noqa: F401
    from matplotlib.patches import Rectangle
    import mpl_toolkits.mplot3d.art3d as art3d

    Z_NEU   = 0.0
    Z_ASTRO = 0.45 * c_max     # vertical separation set to ~half the arena;
                               # large enough to read, small enough that
                               # bouton→astrocyte links aren't near-vertical.

    fig = plt.figure(figsize=(11.5, 9.0))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    try:
        ax.set_facecolor('white')
    except Exception:
        pass

    # Reduce panel/grid prominence so the data dominates.
    try:
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0))
        ax.xaxis.pane.set_edgecolor('#dddddd')
        ax.yaxis.pane.set_edgecolor('#dddddd')
        ax.zaxis.pane.set_edgecolor('#dddddd')
    except Exception:
        pass
    ax.grid(True, linestyle=':', linewidth=0.4, color='#cccccc', alpha=0.6)

    # ── Voronoi tessellation, projected onto each plane ─────────────────────
    if mode == 'Full' and len(topo['A_pos']):
        _draw_voronoi_3d_at_z(ax, topo['N_pos'], topo['A_pos'], c_max,
                              z=Z_NEU,   which='neuron')
        _draw_voronoi_3d_at_z(ax, topo['N_pos'], topo['A_pos'], c_max,
                              z=Z_ASTRO, which='astro')

    # Plane outlines (thin gray rectangles flush with each z layer).
    for z, color in ((Z_NEU, '#999999'), (Z_ASTRO, '#999999')):
        rect = Rectangle((0.0, 0.0), c_max, c_max,
                         fill=False, edgecolor=color, linewidth=0.8,
                         linestyle='--', alpha=0.5)
        ax.add_patch(rect)
        art3d.pathpatch_2d_to_3d(rect, z=z, zdir='z')

    # ── Neuron ↔ neuron synapses (very faint, green dashed) ────────────────
    if len(topo['S_i']):
        for si, sj in zip(topo['S_i'], topo['S_j']):
            ax.plot([topo['N_pos'][si, 0], topo['N_pos'][sj, 0]],
                    [topo['N_pos'][si, 1], topo['N_pos'][sj, 1]],
                    [Z_NEU, Z_NEU],
                    color='#2e7d32', alpha=0.06,
                    linewidth=0.4, linestyle='--', zorder=1)

    # ── Bouton → astrocyte links (purple, crossing planes) ─────────────────
    if mode == 'Full' and len(topo.get('StoA_i', [])):
        for si, aj in zip(topo['StoA_i'], topo['StoA_j']):
            ax.plot([topo['S_x_syn'][si], topo['A_pos'][aj, 0]],
                    [topo['S_y_syn'][si], topo['A_pos'][aj, 1]],
                    [Z_NEU, Z_ASTRO],
                    color=_VIZ_STOA, alpha=0.45, linewidth=0.6, zorder=2)

    # ── Astrocyte ↔ astrocyte GJC (amber, thick, in the astro plane) ───────
    n_gj_unique = 0
    if mode == 'Full' and len(topo.get('GJ_i', [])):
        seen = set()
        for i, j in zip(topo['GJ_i'], topo['GJ_j']):
            key = (int(min(i, j)), int(max(i, j)))
            if key in seen:
                continue
            seen.add(key)
            ax.plot([topo['A_pos'][i, 0], topo['A_pos'][j, 0]],
                    [topo['A_pos'][i, 1], topo['A_pos'][j, 1]],
                    [Z_ASTRO, Z_ASTRO],
                    color=_VIZ_GJ, alpha=0.95, linewidth=2.0, zorder=3)
        n_gj_unique = len(seen)

    # ── Boutons in the neuron plane (red dots) ─────────────────────────────
    if len(topo['S_i']):
        ax.scatter(topo['S_x_syn'], topo['S_y_syn'],
                   np.full(len(topo['S_x_syn']), Z_NEU),
                   c=_VIZ_BOUTON, s=4, alpha=0.55,
                   marker='.', linewidths=0, zorder=4)

    # ── Neurons & astrocytes ───────────────────────────────────────────────
    ax.scatter(topo['N_pos'][:, 0], topo['N_pos'][:, 1],
               np.full(len(topo['N_pos']), Z_NEU),
               c=_VIZ_NEURON_FACE, s=55, alpha=0.95, marker='o',
               edgecolors=_VIZ_NEURON_EDGE, linewidths=0.7, zorder=5,
               label=f'Neurons ({len(topo["N_pos"])})')

    if mode == 'Full' and len(topo['A_pos']):
        ax.scatter(topo['A_pos'][:, 0], topo['A_pos'][:, 1],
                   np.full(len(topo['A_pos']), Z_ASTRO),
                   c=_VIZ_ASTRO_FACE, s=85, alpha=0.95, marker='D',
                   edgecolors=_VIZ_ASTRO_EDGE, linewidths=0.8, zorder=6,
                   label=f'Astrocytes ({len(topo["A_pos"])})')

    # ── Legend (proxy artists with link colours) ───────────────────────────
    from matplotlib.lines import Line2D
    proxies = [
        Line2D([], [], color=_VIZ_NEURON_FACE, marker='o', linewidth=0,
               markeredgecolor=_VIZ_NEURON_EDGE,
               label=f'Neurons ({len(topo["N_pos"])})'),
    ]
    if mode == 'Full' and len(topo['A_pos']):
        proxies += [
            Line2D([], [], color=_VIZ_ASTRO_FACE, marker='D', linewidth=0,
                   markeredgecolor=_VIZ_ASTRO_EDGE,
                   label=f'Astrocytes ({len(topo["A_pos"])})'),
            Line2D([], [], color=_VIZ_GJ, linewidth=2.0,
                   label=f'GJ links ({n_gj_unique})'),
            Line2D([], [], color=_VIZ_STOA, linewidth=1.0, alpha=0.7,
                   label=f'Bouton→astrocyte ({len(topo.get("StoA_i", []))})'),
        ]
    proxies.append(
        Line2D([], [], color='#2e7d32', linewidth=0.9, linestyle='--', alpha=0.55,
               label=f'Synapses ({len(topo["S_i"])})')
    )
    ax.legend(handles=proxies, loc='upper left', fontsize=9,
              framealpha=0.92, facecolor='white', edgecolor='#cccccc')

    # ── Axes ───────────────────────────────────────────────────────────────
    pad = 0.04 * c_max
    ax.set_xlim(-pad, c_max + pad)
    ax.set_ylim(-pad, c_max + pad)
    ax.set_zlim(-0.1 * c_max, Z_ASTRO + 0.1 * c_max)

    ax.set_xlabel('x (µm)', fontsize=10, labelpad=8)
    ax.set_ylabel('y (µm)', fontsize=10, labelpad=8)
    ax.set_zlabel('layer (a.u.)', fontsize=10, labelpad=4)

    ax.set_title('Layered network topology — neurons / astrocytes\n'
                 + _topology_summary_str(topo, mode, topology_mode),
                 fontsize=11, pad=14, color='#222222')

    # Slightly elevated view angle for readability.
    ax.view_init(elev=24, azim=-55)

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] 3D layered topology → {out_path}')


def _draw_voronoi_3d_at_z(ax, N_pos, A_pos, c_max, z, which):
    """
    Project the joint (neurons ∪ astrocytes) Voronoi tessellation onto a
    horizontal plane at height z, drawing only the polygons of one cell
    class (`which='astro'` or `'neuron'`) as tinted polygons.
    """
    from scipy.spatial import Voronoi
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    Nn = int(len(N_pos))
    Na = int(len(A_pos))
    if Nn + Na < 3:
        return

    points = np.vstack([N_pos, A_pos])
    try:
        vor = Voronoi(points)
    except Exception:
        return

    regions, vertices = _voronoi_finite_polygons_2d(vor, radius=4.0 * c_max)

    polys_3d = []
    for pt_idx, region in enumerate(regions):
        if not region:
            continue
        if which == 'astro'  and pt_idx <  Nn:
            continue
        if which == 'neuron' and pt_idx >= Nn:
            continue
        poly = vertices[region]
        poly = _clip_polygon_to_rect(poly, 0.0, 0.0, c_max, c_max)
        if poly is None or len(poly) < 3:
            continue
        polys_3d.append([(p[0], p[1], z) for p in poly])

    if not polys_3d:
        return

    face = _VIZ_ASTRO_FILL if which == 'astro' else _VIZ_NEURON_FILL
    pc3 = Poly3DCollection(polys_3d, facecolors=face, edgecolors=_VIZ_VORONOI,
                           linewidths=0.3, alpha=0.32)
    ax.add_collection3d(pc3)


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

def _syn_pdist_candidates(args) -> list:
    """Ordered candidate paths for synapse_pdist.csv:
      1. explicit --syn_pdist_csv path
      2. <lib_dir>/synapse_pdist.csv
      3. <script_dir>/synapse_pdist.csv
    """
    candidates = []
    if getattr(args, 'syn_pdist_csv', None) is not None:
        candidates.append(args.syn_pdist_csv)
    lib_dir    = args.lib_dir or os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(lib_dir,    'synapse_pdist.csv'))
    candidates.append(os.path.join(script_dir, 'synapse_pdist.csv'))
    return candidates


def _find_syn_pdist_csv(args):
    """Return the first existing synapse_pdist.csv candidate path, else None.
    Pure path resolution — does NOT load the file (used for run provenance so
    job_args.json can record the file actually used instead of a bare null)."""
    for path in _syn_pdist_candidates(args):
        if path and os.path.isfile(path):
            return path
    return None


def _resolve_syn_pdist_csv(args) -> 'pandas.DataFrame':
    """
    Locate and load synapse_pdist.csv (see _syn_pdist_candidates for the order).
    The resolved path is stashed in df.attrs['source_path'] for provenance.
    """
    import pandas as pd

    path = _find_syn_pdist_csv(args)
    if path is None:
        raise FileNotFoundError(
            'synapse_pdist.csv not found. Searched: '
            + '; '.join(c for c in _syn_pdist_candidates(args) if c)
            + '. Provide it via --syn_pdist_csv PATH or place it next to '
              'ASD_fun_BD_cpp.py.'
        )
    print(f'[pass 1] Loading synapse_pdist.csv from: {path}')
    df = pd.read_csv(path)
    for required in ('Syn_prob', 'Radius_val'):
        if required not in df.columns:
            raise ValueError(
                f"{path}: missing required column '{required}'. "
                f"Got columns: {list(df.columns)}")
    df.attrs['source_path'] = path
    return df


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
        topology_mode=args.topology_mode,
        gj_max_dist=args.gj_max_dist,
    )
    t_topo = _wall.time() - t0_topo
    save_topology(topo, args.out_dir)

    # ── Spatial layout plots (independent of pass 2 — done up-front so the
    #    user has a visual sanity check even if the simulation later fails) ──
    plot_spatial_layout(
        topo, c_max=args.c_max, mode=args.mode,
        out_path=os.path.join(args.out_dir, 'spatial_layout.png'),
        dpi=args.dpi,
        topology_mode=args.topology_mode,
        also_connectivity=True,
    )
    plot_3d_layered_topology(
        topo, c_max=args.c_max, mode=args.mode,
        out_path=os.path.join(args.out_dir, 'spatial_layout_3d.png'),
        dpi=args.dpi,
        topology_mode=args.topology_mode,
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
