#!/usr/bin/env python3
"""
range_check.py -- 20 s runs over a setup's prior boxes, through the setup's own
sweep code. Standalone analysis tool; spec: specs/SPEC.md, Block 6.

Why
    The campaigns' prior boxes were chosen while the CAdEx neuron had no
    refractory hold, and most of those networks lost their neurons to NaN after
    the first burst. With the hold (main 0963b7b) the same parameters give very
    different activity, so before a campaign is relaunched we measure what its
    prior produces.

What it runs
    The setup's own HPC_main_sweep.main(), imported from the --lib_dir among the
    arguments (the project folder: 36-D or 37-D, with or without the --freeze
    patch). Every other argument is passed through; these are appended, and
    later arguments win: --out_dir, --simtime, --n_topologies,
    --n_params_per_worker, --seed_master, and for the oat phase
    --conn_prob_lo P --conn_prob_hi P.

    --rc-phase oat    one-axis sweeps: each free axis at K values across its
                      box, in the sweep's theta coordinate (ln on log axes),
                      every other axis at the nominal vector; R topologies at
                      conn_prob P. The only change to the production path: in
                      this (parent) process the module's sample_theta returns
                      that list, in order, for every topology.
    --rc-phase prior  the production sampling, unchanged: T topologies x
                      n_workers x k draws from the setup's own prior.

    --rc-like FILE    start from a campaign's job_args.json (rebuilt through the
                      module's own parser); explicit arguments override it.

Usage (one phase per call; submit_range_check.sh runs them all)
    python3 range_check.py --rc-phase oat --rc-conn 0.35 --rc-out OUT \\
        --lib_dir DIR --mode Neuronal --sweep_group neuron_synapse ...
    python3 range_check.py --rc-phase prior --rc-out OUT --rc-like JOB_ARGS --lib_dir DIR
    add --rc-design-only to print the oat design and run nothing.
Output: OUT/oat_cp<P>/ or OUT/prior/ in the standard task layout, plus
design.json. Analysis: range_analysis.py OUT; figures: range_plots.py OUT.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

CFG = dict(points=7, oat_topologies=2, prior_topologies=8, k_prior=1, simtime=20.0,
           seed=20261030)
# job_args keys never taken from --rc-like: paths and sizes of the old task,
# and the values this driver sets itself.
LIKE_SKIP = ('out_dir', 'scratch_root', 'lib_dir', 'seed_master', 'n_workers', 'n_topologies',
             'n_params_per_worker', 'simtime', 'rebuild_manifest_only')
DV = re.compile(r'^[ \t]*dV/dt[ \t]*=.*$', re.M)


# ---------------------------------------------------------------------------
# argument handling
# ---------------------------------------------------------------------------
def rc_parser():
    p = argparse.ArgumentParser(description='20 s runs over a setup prior (specs/SPEC.md Block 6). '
                                'Arguments not listed here go to HPC_main_sweep.py.',
                                allow_abbrev=False)
    p.add_argument('--rc-phase', choices=('oat', 'prior'), required=True)
    p.add_argument('--rc-out', required=True, help='output root; each phase gets a subfolder')
    p.add_argument('--rc-like', default=None, help="a campaign's job_args.json to start from")
    p.add_argument('--rc-conn', type=float, default=None, help='conn_prob of the oat phase')
    p.add_argument('--rc-points', type=int, default=CFG['points'], help='values per axis (oat)')
    p.add_argument('--rc-topologies', type=int, default=None,
                   help='topologies (default %d oat, %d prior)' % (CFG['oat_topologies'],
                                                                 CFG['prior_topologies']))
    p.add_argument('--rc-k-prior', type=int, default=CFG['k_prior'],
                   help='draws per worker per topology (prior)')
    p.add_argument('--rc-simtime', type=float, default=CFG['simtime'], help='seconds per run')
    p.add_argument('--rc-seed', type=int, default=CFG['seed'])
    p.add_argument('--rc-allow-unclamped', action='store_true',
                   help='run even if the model does not hold V during t_ref')
    p.add_argument('--rc-design-only', action='store_true', help='print the oat design, run nothing')
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
# design (pure functions of the module's registry)
# ---------------------------------------------------------------------------
def free_axes(S, args):
    """Indices the sweep draws: the sweep group minus the --freeze axes."""
    active = [int(i) for i in S.resolve_sweep_group(args.sweep_group)]
    spec = getattr(args, 'freeze', '') or ''
    if spec.strip():
        if not hasattr(S, '_apply_freeze'):
            raise SystemExit('ERROR: --freeze given, but this HPC_main_sweep.py has no _apply_freeze')
        active = [int(i) for i in S._apply_freeze(spec, active)]   # idempotent; main() repeats it
    return active


def oat_design(S, free, n_points):
    """(nominal theta, points, labels): each point is the nominal theta with one
    free axis moved to one of n_points values spanning its theta box."""
    nominal = S.natural_to_theta(np.asarray(S.NOMINAL_PARAMS, dtype=np.float64))
    points, labels = [], []
    for j in free:
        lo, hi = (float(x) for x in S.PARAM_BOUNDS_THETA[j])
        if not hi > lo:
            continue
        for v in np.linspace(lo, hi, n_points):
            th = nominal.copy()
            th[j] = v
            points.append(th)
            labels.append(dict(axis=S.PARAM_NAMES[j], index=int(j), theta=float(v),
                               natural=float(S.theta_to_natural(th)[j])))
    return nominal, points, labels


def pad_design(nominal, points, n_workers):
    """Pad with nominal vectors to n_workers x k, k = ceil(len / n_workers)."""
    k = max(1, int(math.ceil(len(points) / float(n_workers))))
    padded = list(points) + [nominal.copy() for _ in range(n_workers * k - len(points))]
    return k, padded


class ThetaList(object):
    """Stand-in for sample_theta(rng): returns the padded design in order, again
    for every topology (main() calls it exactly n_workers x k times per topology)."""

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
def main(argv=None):
    rc, rest = rc_parser().parse_known_args(sys.argv[1:] if argv is None else argv)
    if rc.rc_phase == 'oat' and rc.rc_conn is None:
        raise SystemExit('ERROR: --rc-phase oat needs --rc-conn P')
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

    if rc.rc_phase == 'oat':
        n_topo = rc.rc_topologies or CFG['oat_topologies']
        sub = 'oat_cp%.3f' % rc.rc_conn
        seed = rc.rc_seed + 1000 + int(round(rc.rc_conn * 1000))
    else:
        n_topo = rc.rc_topologies or CFG['prior_topologies']
        sub = 'prior'
        seed = rc.rc_seed + 1
    out_dir = os.path.abspath(os.path.join(rc.rc_out, sub))
    tail = ['--out_dir', out_dir, '--simtime', repr(rc.rc_simtime), '--n_topologies', str(n_topo),
            '--seed_master', str(seed)]
    if rc.rc_phase == 'oat':
        tail += ['--conn_prob_lo', repr(rc.rc_conn), '--conn_prob_hi', repr(rc.rc_conn)]
    argv_main = base + rest + tail
    args = parser.parse_args(argv_main + ['--n_params_per_worker', '1'])   # validates everything
    n_workers = resolve_n_workers(S, args)
    free = free_axes(S, args)
    print('[range_check] %d free axes: %s' % (len(free), [S.PARAM_NAMES[i] for i in free]))

    design = dict(phase=rc.rc_phase, conn_prob=rc.rc_conn, simtime=rc.rc_simtime, n_topologies=n_topo,
                  n_workers=n_workers, seed_master=seed, lib_dir=lib_dir, module=S.__file__,
                  model_line=line, param_names=list(S.PARAM_NAMES),
                  log_params=sorted(int(k) for k in S.LOG_PARAMS),
                  free_axes=[S.PARAM_NAMES[i] for i in free], argv=argv_main)
    if rc.rc_phase == 'oat':
        nominal, points, labels = oat_design(S, free, rc.rc_points)
        k, padded = pad_design(nominal, points, n_workers)
        design.update(n_points=rc.rc_points, k=k, n_design=len(points),
                      n_padding=len(padded) - len(points), nominal_theta=nominal.tolist(),
                      nominal_natural=S.theta_to_natural(nominal).tolist(),
                      points=[dict(lab, theta_vector=p.tolist()) for lab, p in zip(labels, points)])
        print('[range_check] oat: %d points + %d nominal = %d per topology (k=%d, %d workers), '
              '%d topologies at conn_prob %g' % (len(points), len(padded) - len(points),
                                                 len(padded), k, n_workers, n_topo, rc.rc_conn))
    else:
        k = rc.rc_k_prior
        design.update(k=k)
        print('[range_check] prior: %d topologies x %d workers x k=%d = %d runs' % (
            n_topo, n_workers, k, n_topo * n_workers * k))
    if rc.rc_design_only:
        for lab in design.get('points', []):
            print('  %-11s %12.5g' % (lab['axis'], lab['natural']))
        return 0

    if os.path.isdir(out_dir) and any(n.startswith('topo_') for n in os.listdir(out_dir)):
        raise SystemExit('ERROR: %s already holds runs; give a new --rc-out' % out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'design.json'), 'w') as fh:
        json.dump(design, fh, indent=1)
    if rc.rc_phase == 'oat':
        S.sample_theta = ThetaList(padded)
    sys.argv = ['HPC_main_sweep.py'] + argv_main + ['--n_params_per_worker', str(k)]
    os.chdir(lib_dir)                          # as the production workers do
    S.main()
    if rc.rc_phase == 'oat':
        n_full = S.sample_theta.calls // len(padded)
        print('[range_check] oat: design drawn for %d of %d topologies (%d calls)' % (
            n_full, n_topo, S.sample_theta.calls))
        if S.sample_theta.calls % len(padded):
            print('[range_check] WARNING: a topology drew a partial design')
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
