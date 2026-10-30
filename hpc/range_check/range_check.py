#!/usr/bin/env python3
"""
range_check.py -- the standard prior range calibration (decided by the author on
2026-10-30): 20 s runs at the corners of a setup's prior box and at random cells
of its 7-bin grid, through the setup's own sweep code. Spec: specs/SPEC.md,
Block 6; designs: range_design.py.

Why
    A campaign's prior box has to produce plausible activity across its whole
    range. The corners show what every combination of extreme values does; the
    interior cells show where inside the box the activity changes.

What it runs
    The setup's own HPC_main_sweep.main(), imported from the --lib_dir among the
    arguments (the project folder: 36-D or 37-D, with or without the --freeze
    patch). Every other argument is passed through; these are appended, and
    later arguments win: --out_dir, --simtime, --n_topologies,
    --n_params_per_worker, --seed_master, --conn_prob_lo P --conn_prob_hi P.
    The only change to the production path: in this (parent) process the
    module's sample_theta returns the design, in order, chunk after chunk for
    the successive topologies.

    Design dimensions: the free axes (the sweep group minus --freeze) in the
    sweep's theta coordinate (ln on log axes), plus conn_prob when the setup
    draws it per topology (flat rule, lo < hi). Each call runs one level:
    --rc-phase corners --rc-level L   conn_prob at its lower (L = 0) or upper
                      (L = 1) bound; the corners of that half of the two-level
                      design (full factorial if 2^d <= --rc-corner-budget,
                      else a resolution >= V fraction), shuffled, split over
                      --rc-corner-topologies topologies, the last slots filled
                      with the box centre (centre points).
    --rc-phase interior --rc-level L  conn_prob at the midpoint of its bin L of
                      --rc-bins; --rc-interior-runs / bins random cells (at
                      least), each free axis at the midpoint of a uniformly
                      drawn bin, over --rc-interior-topologies topologies.
    --rc-like FILE    start from a campaign's job_args.json (rebuilt through the
                      module's own parser); explicit arguments override it.

Usage (one level per call; submit_range_check.sh runs them all)
    python3 range_check.py --rc-phase corners --rc-level 0 --rc-out OUT \\
        --lib_dir DIR --mode Neuronal --sweep_group neuron_synapse ...
    add --rc-design-only to print the design of that level and run nothing;
    --rc-subset N runs only the first N design points on one topology (tests).
Output: OUT/corners_c<L>/ or OUT/interior_b<L>/ in the standard task layout,
plus design.json. Analysis: range_analysis.py OUT; figures: range_plots.py OUT.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import range_design as RD   # noqa: E402

CFG = dict(bins=7, corner_budget=RD.CORNER_BUDGET, corner_topologies=8, interior_runs=16384,
           interior_topologies=7, simtime=20.0, seed=20261030, max_resolution=8)
# job_args keys never taken from --rc-like: paths and sizes of the old task,
# and the values this driver sets itself.
LIKE_SKIP = ('out_dir', 'scratch_root', 'lib_dir', 'seed_master', 'n_workers', 'n_topologies',
             'n_params_per_worker', 'simtime', 'rebuild_manifest_only')
DV = re.compile(r'^[ \t]*dV/dt[ \t]*=.*$', re.M)


# ---------------------------------------------------------------------------
# argument handling
# ---------------------------------------------------------------------------
def rc_parser():
    p = argparse.ArgumentParser(description='Prior range calibration runs (specs/SPEC.md Block 6). '
                                'Arguments not listed here go to HPC_main_sweep.py.',
                                allow_abbrev=False)
    p.add_argument('--rc-phase', choices=('corners', 'interior'), required=True)
    p.add_argument('--rc-level', type=int, required=True,
                   help='corners: 0 = conn_prob at its lower bound, 1 = upper; interior: the conn_prob bin')
    p.add_argument('--rc-out', required=True, help='output root; each level gets a subfolder')
    p.add_argument('--rc-like', default=None, help="a campaign's job_args.json to start from")
    p.add_argument('--rc-bins', type=int, default=CFG['bins'], help='bins per dimension (interior)')
    p.add_argument('--rc-corner-budget', type=int, default=CFG['corner_budget'],
                   help='largest corner design; above it a two-level fraction is used')
    p.add_argument('--rc-corner-topologies', type=int, default=CFG['corner_topologies'])
    p.add_argument('--rc-interior-runs', type=int, default=CFG['interior_runs'],
                   help='interior cells over all levels (at least)')
    p.add_argument('--rc-interior-topologies', type=int, default=CFG['interior_topologies'])
    p.add_argument('--rc-simtime', type=float, default=CFG['simtime'], help='seconds per run')
    p.add_argument('--rc-seed', type=int, default=CFG['seed'])
    p.add_argument('--rc-subset', type=int, default=None,
                   help='run only the first N design points of this level, on one topology (tests)')
    p.add_argument('--rc-allow-unclamped', action='store_true',
                   help='run even if the model does not hold V during t_ref')
    p.add_argument('--rc-design-only', action='store_true', help='print the design, run nothing')
    return p


def lib_dir_from(rest):
    """The --lib_dir among the pass-through arguments (required, made absolute)."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument('--lib_dir', default=None)
    known, _ = p.parse_known_args(rest)
    if not known.lib_dir:
        raise SystemExit('ERROR: give --lib_dir <the setup folder holding HPC_main_sweep.py>')
    return os.path.abspath(known.lib_dir)


def load_sweep_module(lib_dir):
    if not os.path.isfile(os.path.join(lib_dir, 'HPC_main_sweep.py')):
        raise SystemExit('ERROR: %s has no HPC_main_sweep.py' % lib_dir)
    sys.path.insert(0, lib_dir)
    import HPC_main_sweep
    return HPC_main_sweep


def long_option(action):
    longs = [o for o in action.option_strings if o.startswith('--')]
    return max(longs, key=len) if longs else action.option_strings[0]


def args_from_job_args(parser, job_args, skip=LIKE_SKIP):
    """Argument list that reproduces a campaign's job_args.json through the parser."""
    out = []
    for action in parser._actions:
        if not action.option_strings or action.dest in skip or action.dest not in job_args:
            continue
        v, opt = job_args[action.dest], long_option(action)
        if isinstance(action, argparse._StoreTrueAction):
            out += [opt] if v else []
        elif isinstance(action, argparse._StoreFalseAction):
            out += [] if v else [opt]
        elif v is None or isinstance(action, (argparse._HelpAction, argparse._VersionAction)):
            continue
        elif isinstance(v, (list, tuple)):
            out += [opt] + [str(x) for x in v]
        else:
            out += [opt, str(v)]
    return out


# ---------------------------------------------------------------------------
# the setup's design space
# ---------------------------------------------------------------------------
def free_axes(S, args):
    """Indices the sweep draws: the sweep group minus the --freeze axes."""
    active = [int(i) for i in S.resolve_sweep_group(args.sweep_group)]
    spec = getattr(args, 'freeze', '') or ''
    if spec.strip():
        if not hasattr(S, '_apply_freeze'):
            raise SystemExit('ERROR: --freeze given, but this HPC_main_sweep.py has no _apply_freeze')
        active = [int(i) for i in S._apply_freeze(spec, active)]   # idempotent; main() repeats it
    return [j for j in active if float(S.PARAM_BOUNDS_THETA[j][1]) > float(S.PARAM_BOUNDS_THETA[j][0])]


def design_space(S, args):
    """The free axes, their theta boxes, the nominal theta (after the freeze) and
    the conn_prob box when it is a design dimension (flat rule, lo < hi)."""
    free = free_axes(S, args)
    lo = np.array([float(S.PARAM_BOUNDS_THETA[j][0]) for j in free])
    hi = np.array([float(S.PARAM_BOUNDS_THETA[j][1]) for j in free])
    nominal = S.natural_to_theta(np.asarray(S.NOMINAL_PARAMS, dtype=np.float64))
    c_lo, c_hi = float(args.conn_prob_lo), float(args.conn_prob_hi)
    has_conn = getattr(args, 'conn_rule', 'flat') == 'flat' and c_hi > c_lo
    return dict(free=free, lo=lo, hi=hi, nominal=nominal, has_conn=has_conn, conn_box=(c_lo, c_hi))


def theta_vectors(space, values):
    """Nominal theta with the free axes set to the rows of `values` (n x d_r)."""
    out = np.repeat(space['nominal'][None, :], len(values), axis=0)
    out[:, space['free']] = values
    return out


def corner_level(space, rc):
    """(conn value or None, codes n x d_r of +-1 for this level, design info)."""
    d_r = len(space['free'])
    d = d_r + (1 if space['has_conn'] else 0)
    X, info = RD.two_level_design(d, rc.rc_corner_budget, np.random.default_rng(rc.rc_seed + 20000),
                                  CFG['max_resolution'])
    n_levels = 2 if space['has_conn'] else 1
    if not 0 <= rc.rc_level < n_levels:
        raise SystemExit('ERROR: --rc-level %d: corners have levels 0..%d here' % (rc.rc_level, n_levels - 1))
    if space['has_conn']:
        rows = X[X[:, -1] == (-1 if rc.rc_level == 0 else 1)][:, :d_r]
        conn = space['conn_box'][rc.rc_level]
    else:
        rows, conn = X, None
    order = np.random.default_rng(rc.rc_seed + 20100 + rc.rc_level).permutation(len(rows))
    info = dict(info, d=d, n_levels=n_levels, n_level_corners=len(rows))
    return conn, rows[order].astype(int), info


def interior_level(space, rc, n_workers, n_topologies, per_level=None):
    """(conn value or None, bin indices n x d_r, midpoints per free axis, n_levels).
    per_level: cells wanted at this level (default --rc-interior-runs / levels);
    n is rounded up to fill n_topologies x n_workers x k slots."""
    B = rc.rc_bins
    n_levels = B if space['has_conn'] else 1
    if not 0 <= rc.rc_level < n_levels:
        raise SystemExit('ERROR: --rc-level %d: interior has levels 0..%d here' % (rc.rc_level, n_levels - 1))
    conn = (float(RD.bin_midpoints(space['conn_box'][0], space['conn_box'][1], B)[rc.rc_level])
            if space['has_conn'] else None)
    if per_level is None:
        per_level = int(np.ceil(rc.rc_interior_runs / float(n_levels)))
    k, _ = RD.chunk_plan(per_level, n_topologies, n_workers)
    n = n_topologies * n_workers * k
    bins = RD.interior_cells(n, len(space['free']), B, np.random.default_rng(rc.rc_seed + 30000 + rc.rc_level))
    mids = [RD.bin_midpoints(a, b, B) for a, b in zip(space['lo'], space['hi'])]
    return conn, bins, mids, n_levels


class ThetaList(object):
    """Stand-in for sample_theta(rng): returns the padded design in order. main()
    calls it n_workers x k times per topology, so successive topologies take
    successive chunks."""

    def __init__(self, padded):
        self.padded = [np.array(p, dtype=np.float64) for p in padded]
        self.calls = 0

    def __call__(self, rng):
        th = self.padded[self.calls % len(self.padded)].copy()
        self.calls += 1
        return th


def resolve_n_workers(S, args):
    if hasattr(S, '_resolve_n_workers'):
        return int(S._resolve_n_workers(args))
    return int(args.n_workers or os.environ.get('PBS_NCPUS') or os.cpu_count())


def model_is_clamped(lib_dir):
    path = os.path.join(lib_dir, 'ASD_fun_BD_cpp.py')
    try:
        lines = DV.findall(open(path, encoding='utf-8').read())
    except (OSError, UnicodeDecodeError):
        return None, 'cannot read %s' % path
    cadex = [ln for ln in lines if 'exp((V-VT)/DeltaT)' in ln.replace(' ', '')]
    if len(cadex) != 1:
        return None, '%d CAdEx dV/dt lines in %s' % (len(cadex), path)
    return cadex[0].rstrip().endswith('(unless refractory)'), cadex[0].strip()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_level(S, args, rc, n_workers):
    """(design dict without run metadata, padded theta list, conn value)."""
    space = design_space(S, args)
    names = [S.PARAM_NAMES[j] for j in space['free']]
    design = dict(phase=rc.rc_phase, level=rc.rc_level, free_axes=names,
                  theta_lo=space['lo'].tolist(), theta_hi=space['hi'].tolist(),
                  has_conn=space['has_conn'], conn_box=list(space['conn_box']),
                  nominal_theta=space['nominal'].tolist(), n_workers=n_workers)
    if rc.rc_phase == 'corners':
        conn, codes, info = corner_level(space, rc)
        n_topo = rc.rc_corner_topologies
        if rc.rc_subset:
            codes, n_topo = codes[:rc.rc_subset], 1
        k, n_pad = RD.chunk_plan(len(codes), n_topo, n_workers)
        values = np.where(codes > 0, space['hi'][None, :], space['lo'][None, :])
        centre = 0.5 * (space['lo'] + space['hi'])
        padded = list(theta_vectors(space, values)) + [theta_vectors(space, centre[None, :])[0]] * n_pad
        design.update(corner_design=info, codes=codes.tolist(), n_design=len(codes), n_padding=n_pad,
                      centre_theta=centre.tolist())
    else:
        n_topo = 1 if rc.rc_subset else rc.rc_interior_topologies
        conn, bins, mids, n_levels = interior_level(space, rc, n_workers, n_topo,
                                                    per_level=rc.rc_subset or None)
        k = len(bins) // (n_topo * n_workers)
        values = np.column_stack([m[b] for m, b in zip(mids, bins.T)]) if len(mids) else np.zeros((len(bins), 0))
        padded = list(theta_vectors(space, values))
        design.update(bins=rc.rc_bins, n_levels=n_levels, midpoints=[m.tolist() for m in mids],
                      cells=bins.tolist(), n_design=len(bins), n_padding=0)
    design.update(conn_prob=conn, n_topologies=n_topo, k=k, n_slots=len(padded))
    return design, padded, conn


def main(argv=None):
    rc, rest = rc_parser().parse_known_args(sys.argv[1:] if argv is None else argv)
    lib_dir = lib_dir_from(rest)
    clamped, line = model_is_clamped(lib_dir)
    print('[range_check] model: %s' % line)
    if clamped is not True and not rc.rc_allow_unclamped:
        raise SystemExit('REFUSED: the model in %s does not hold V during t_ref (%s); run '
                         'clamp_cadex_copies.py --apply first, or pass --rc-allow-unclamped' % (lib_dir, line))
    S = load_sweep_module(lib_dir)
    parser = S.build_parser()
    base = []
    if rc.rc_like:
        base = args_from_job_args(parser, json.load(open(rc.rc_like)))
        print('[range_check] from %s: %s' % (rc.rc_like, ' '.join(base)))
    args = parser.parse_args(base + rest + ['--out_dir', 'x', '--n_params_per_worker', '1'])
    n_workers = resolve_n_workers(S, args)
    design, padded, conn = build_level(S, args, rc, n_workers)
    sub = '%s_%s%d' % (rc.rc_phase, 'c' if rc.rc_phase == 'corners' else 'b', rc.rc_level)
    seed = rc.rc_seed + (2000 if rc.rc_phase == 'corners' else 3000) + rc.rc_level
    out_dir = os.path.abspath(os.path.join(rc.rc_out, sub))
    tail = ['--out_dir', out_dir, '--simtime', repr(rc.rc_simtime), '--n_topologies',
            str(design['n_topologies']), '--seed_master', str(seed)]
    if conn is not None:
        tail += ['--conn_prob_lo', repr(conn), '--conn_prob_hi', repr(conn)]
    argv_main = base + rest + tail
    parser.parse_args(argv_main + ['--n_params_per_worker', str(design['k'])])   # validates everything
    design.update(simtime=rc.rc_simtime, seed_master=seed, lib_dir=lib_dir, module=S.__file__,
                  model_line=line, param_names=list(S.PARAM_NAMES),
                  log_params=sorted(int(k) for k in S.LOG_PARAMS), argv=argv_main)
    cd = design.get('corner_design', {})
    print('[range_check] %s level %d: %d free axes %s, conn_prob %s' % (
        rc.rc_phase, rc.rc_level, len(design['free_axes']), design['free_axes'],
        'not a design dimension' if conn is None else repr(conn)))
    if rc.rc_phase == 'corners':
        print('[range_check] corner design: %s of %d dimensions, %d runs%s; this level: %d corners + %d centre '
              'points = %d topologies x %d workers x k=%d' % (
                  cd['kind'], cd['d'], cd['n_runs'],
                  (', resolution %d' % cd['resolution']) if cd['resolution'] else '',
                  design['n_design'], design['n_padding'], design['n_topologies'], n_workers, design['k']))
    else:
        print('[range_check] interior: %d bins; this level: %d cells = %d topologies x %d workers x k=%d' % (
            design['bins'], design['n_design'], design['n_topologies'], n_workers, design['k']))
    if rc.rc_design_only:
        return 0

    if os.path.isdir(out_dir) and any(n.startswith('topo_') for n in os.listdir(out_dir)):
        raise SystemExit('ERROR: %s already holds runs; give a new --rc-out' % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'design.json'), 'w') as fh:
        json.dump(design, fh)
    S.sample_theta = ThetaList(padded)
    sys.argv = ['HPC_main_sweep.py'] + argv_main + ['--n_params_per_worker', str(design['k'])]
    os.chdir(lib_dir)                          # as the production workers do
    S.main()
    drawn = S.sample_theta.calls
    print('[range_check] %s level %d: %d of %d design slots drawn' % (rc.rc_phase, rc.rc_level, drawn,
                                                                    len(padded)))
    if drawn != len(padded):
        print('[range_check] WARNING: not every topology drew its chunk (a topology failed to build?); '
              'range_analysis.py labels runs from their theta, so the runs made are still usable')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
