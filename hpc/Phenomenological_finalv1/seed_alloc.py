#!/usr/bin/env python3
"""
seed_alloc.py -- allocate a collision-free seed range for one campaign launch.

WHY THIS EXISTS
---------------
HPC_main_sweep.py builds one generator per task,

    master_rng = np.random.default_rng(seed_master)

and derives EVERYTHING from it: the per-topology connectivity kernel draw, the
per-topology seed base (hence every neuron/synapse/astro seed), every parameter
vector, and every per-run seed. A task's entire scientific content is therefore
a deterministic function of that single integer.

submit_sweep_mixed.sh computes

    SEED_MASTER = SEED_BASE + PBS_ARRAY_INDEX

and launch_campaign.sh used to hardcode SEED_BASE per queue. Every relaunch
restarted the array index at 0 from the same base, so a later campaign
reproduced every seed of an earlier one and merely appended a few at the end --
"prefix containment". Measured across six exported campaigns: 251,614 rows but
only 86,750 distinct parameter vectors.

Note the defect is seed REUSE, not seed QUALITY. numpy's default_rng puts an
integer seed through SeedSequence, which does proper entropy mixing, so
consecutive seeds already give well-separated streams. Nothing was wrong with
`base + index`; the problem was that `base` never changed.

WHAT THIS DOES
--------------
Draws a uniformly random base from OS entropy, then REJECTS it if the span
[base, base + span) would touch anything already used, and retries. Two
independent sources define "already used":

  1. every `_resolved_seed_master` recorded in campaign_*/sweep_*/job_args.json
     -- ground truth about what was actually consumed;
  2. every span previously reserved in the ledger (artifacts/seed_ledger.tsv)
     -- covers ranges that were allocated but whose jobs have not written
     job_args.json yet, and ranges whose campaign output has been archived.

Because the check is against recorded facts rather than a hash, allocation is
collision-free by construction, not merely improbable.

REPLAYABILITY
-------------
A launch is reproducible AFTER the fact, which is what actually matters:
every task writes its own `_resolved_seed_master` into job_args.json, and this
script appends the allocated base to the ledger. To reproduce a past campaign
exactly, pass --override <base> (launch_campaign.sh exposes this as the
SEED_BASE_OVERRIDE environment variable), which skips allocation entirely.

DEPENDENCIES
------------
Standard library only -- no numpy, no pandas. It must run on a login node
outside any conda environment.

USAGE
-----
    # allocate (prints the base to stdout, diagnostics to stderr)
    python3 seed_alloc.py --span 400000

    # allocate, record in the ledger, and label the entry
    python3 seed_alloc.py --span 400000 --ledger artifacts/seed_ledger.tsv \\
        --launch-id 20260821T101500-ab12cd34 --tag cadex_rho1300v10

    # deliberate exact replay of a past campaign
    python3 seed_alloc.py --span 400000 --override 1234567

    # inspect what is already used, allocate nothing
    python3 seed_alloc.py --report

EXIT CODES
----------
    0  success; the allocated base is on stdout, alone, with no trailing text
    2  bad arguments
    3  could not find a free range after --attempts tries
"""
import argparse
import glob
import json
import os
import sys
import time

# 2**31 - 1. The ceiling _resolve_seed_master() already respects on both of its
# fallback paths; staying under it keeps every seed representable the same way
# regardless of which path produced it.
SEED_MAX_DEFAULT = 2147483647

LEDGER_HEADER = (
    '# seed_ledger.tsv -- one row per campaign launch. Append-only audit trail.\n'
    '# The authoritative per-task record is _resolved_seed_master in each\n'
    '# campaign_*/sweep_*/job_args.json; this file additionally covers ranges\n'
    '# that were reserved before any job wrote its job_args.json.\n'
    '# utc_iso\tlaunch_id\ttag\tseed_base\tspan\tseed_lo\tseed_hi\tnote\n'
)


# ---------------------------------------------------------------------------
# Reading what is already used
# ---------------------------------------------------------------------------
def used_seeds_from_job_args(root='.'):
    """
    Every seed_master actually consumed, read from campaign job_args.json.

    Returns (seeds, n_files_read, n_files_unreadable). Unreadable files are
    counted rather than raised on: a job killed mid-write leaves a truncated
    JSON, and that must not block a launch.
    """
    seeds, n_ok, n_bad = set(), 0, 0
    pattern = os.path.join(root, 'campaign_*', 'sweep_*', 'job_args.json')
    for p in glob.glob(pattern):
        try:
            with open(p, 'r') as fh:
                v = json.load(fh).get('_resolved_seed_master')
            n_ok += 1
        except Exception:
            n_bad += 1
            continue
        if v is not None:
            try:
                seeds.add(int(v))
            except (TypeError, ValueError):
                n_bad += 1
    return seeds, n_ok, n_bad


def reserved_spans_from_ledger(ledger_path):
    """
    Every [lo, hi) span previously reserved. Returns a list of (lo, hi, tag).

    A missing ledger is not an error -- it is the expected state on the first
    run after this change, and on any tree where campaigns were archived.
    """
    spans = []
    if not ledger_path or not os.path.exists(ledger_path):
        return spans
    try:
        with open(ledger_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 7:
                    continue
                try:
                    spans.append((int(parts[5]), int(parts[6]), parts[2]))
                except ValueError:
                    continue
    except Exception:
        pass
    return spans


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------
def range_is_free(lo, span, used_seeds, reserved):
    """
    True iff [lo, lo+span) contains no recorded seed and overlaps no reserved
    span. Overlap, not containment: two spans clash if either end is inside
    the other, which `lo < hi_r and lo_r < hi` captures exactly.
    """
    hi = lo + span
    for s in used_seeds:
        if lo <= s < hi:
            return False
    for lo_r, hi_r, _tag in reserved:
        if lo < hi_r and lo_r < hi:
            return False
    return True


def allocate(span, seed_max=SEED_MAX_DEFAULT, root='.', ledger_path=None,
             attempts=256, verbose=True):
    """
    Draw a uniformly random base in [1, seed_max - span) whose whole span is
    free. Raises RuntimeError if no free range is found in `attempts` tries.

    The draw uses os.urandom, so two launches a second apart -- or two launches
    on different login nodes -- get unrelated bases. This is the property the
    old hardcoded base lacked.
    """
    if span <= 0:
        raise ValueError('span must be positive')
    if span >= seed_max - 2:
        raise ValueError('span %d does not fit below seed_max %d'
                         % (span, seed_max))

    used, n_ok, n_bad = used_seeds_from_job_args(root)
    reserved = reserved_spans_from_ledger(ledger_path)
    if verbose:
        print('[seed_alloc] scanned %d job_args.json (%d unreadable, skipped)'
              % (n_ok, n_bad), file=sys.stderr)
        print('[seed_alloc] %d distinct seed_master values already recorded'
              % len(used), file=sys.stderr)
        print('[seed_alloc] %d span(s) reserved in the ledger' % len(reserved),
              file=sys.stderr)

    hi_bound = seed_max - span - 1
    for attempt in range(1, attempts + 1):
        lo = int.from_bytes(os.urandom(6), 'little') % hi_bound + 1
        if range_is_free(lo, span, used, reserved):
            if verbose:
                print('[seed_alloc] allocated [%d, %d) on attempt %d'
                      % (lo, lo + span, attempt), file=sys.stderr)
            return lo
    raise RuntimeError(
        'no free seed range of width %d found in %d attempts; %d seeds and '
        '%d reserved spans are in the way. The seed space may be genuinely '
        'crowded -- inspect with --report.'
        % (span, attempts, len(used), len(reserved)))


def append_ledger(ledger_path, launch_id, tag, base, span, note=''):
    """Append one row. Creates the parent directory and header if absent."""
    if not ledger_path:
        return
    d = os.path.dirname(ledger_path)
    if d:
        os.makedirs(d, exist_ok=True)
    new = not os.path.exists(ledger_path)
    with open(ledger_path, 'a') as fh:
        if new:
            fh.write(LEDGER_HEADER)
        fh.write('%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\n' % (
            time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            launch_id or '-', tag or '-', base, span, base, base + span,
            note or '-'))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description='Allocate a collision-free seed range for a campaign launch.')
    p.add_argument('--span', type=int, default=400000,
                   help='total width to reserve: n_queues * per-queue spacing '
                        '(default: 400000)')
    p.add_argument('--seed-max', type=int, default=SEED_MAX_DEFAULT,
                   help='exclusive upper bound on any seed (default: 2**31-1)')
    p.add_argument('--root', default='.',
                   help='directory holding campaign_*/ (default: .)')
    p.add_argument('--ledger', default=None,
                   help='append the allocation here, and treat its spans as '
                        'reserved')
    p.add_argument('--launch-id', default=None, help='label for the ledger row')
    p.add_argument('--tag', default=None, help='campaign tag for the ledger row')
    p.add_argument('--note', default=None, help='free-text ledger note')
    p.add_argument('--attempts', type=int, default=256,
                   help='how many random bases to try (default: 256)')
    p.add_argument('--override', type=int, default=None,
                   help='skip allocation and use this base verbatim. For '
                        'DELIBERATE exact replay of a past campaign; it will '
                        'reproduce that campaign\'s draws.')
    p.add_argument('--report', action='store_true',
                   help='print what is already used and exit without allocating')
    p.add_argument('--quiet', action='store_true',
                   help='suppress the stderr diagnostics')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    if args.report:
        used, n_ok, n_bad = used_seeds_from_job_args(args.root)
        reserved = reserved_spans_from_ledger(args.ledger)
        print('job_args.json read      : %d (%d unreadable)' % (n_ok, n_bad))
        print('distinct seed_master    : %d' % len(used))
        if used:
            print('seed_master range       : %d .. %d' % (min(used), max(used)))
        print('ledger spans reserved   : %d' % len(reserved))
        for lo, hi, tag in sorted(reserved):
            print('    [%d, %d)  %s' % (lo, hi, tag))
        return 0

    if args.override is not None:
        if args.override < 0 or args.override + args.span > args.seed_max:
            print('[seed_alloc] ERROR: override %d + span %d exceeds seed_max %d'
                  % (args.override, args.span, args.seed_max), file=sys.stderr)
            return 2
        if verbose:
            print('[seed_alloc] OVERRIDE: using base %d verbatim. This will '
                  'REPRODUCE the draws of whichever launch used it.'
                  % args.override, file=sys.stderr)
        append_ledger(args.ledger, args.launch_id, args.tag, args.override,
                      args.span, note=(args.note or 'override/replay'))
        print(args.override)
        return 0

    try:
        base = allocate(args.span, seed_max=args.seed_max, root=args.root,
                        ledger_path=args.ledger, attempts=args.attempts,
                        verbose=verbose)
    except RuntimeError as e:
        print('[seed_alloc] ERROR: %s' % e, file=sys.stderr)
        return 3
    except ValueError as e:
        print('[seed_alloc] ERROR: %s' % e, file=sys.stderr)
        return 2

    append_ledger(args.ledger, args.launch_id, args.tag, base, args.span,
                  note=(args.note or 'allocated'))
    print(base)
    return 0


if __name__ == '__main__':
    sys.exit(main())
