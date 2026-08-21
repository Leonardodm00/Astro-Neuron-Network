#!/usr/bin/env python3
"""
Smoke test for the per-launch seed allocation fix.

WHAT THIS PROVES
----------------
  T1  THE BUG, REPRODUCED FROM FIRST PRINCIPLES. With a hardcoded SEED_BASE,
      relaunching with a different task count produces PREFIX CONTAINMENT: the
      shorter campaign's parameter vectors are a strict subset of the longer
      one's. Reproduces the pattern measured over v1..v9 (duplication factor
      2.90; v4/v7/v8 wholly inside v3) using the real sample_theta and the real
      outer-loop draw order -- not a mock.

  T2  THE DEFECT IS SEED REUSE, NOT SEED QUALITY. Consecutive integer seeds
      (base+0, base+1, ...) already give well-separated streams, because numpy
      puts an integer seed through SeedSequence before it reaches the bit
      generator. Verified here by showing consecutive seeds share no draws and
      are uncorrelated. This matters: it means the fix only has to guarantee
      DISTINCTNESS, and the existing `SEED_BASE + IDX` arithmetic can stay.

  T3  THE FIX. Simulated repeated launches through the real seed_alloc.py
      produce pairwise-disjoint seed ranges, zero repeated seed_master values,
      and zero repeated parameter vectors across campaigns.

  T4  THE GUARD FIRES (negative paths). seed_alloc.py refuses a range that
      would touch a seed already recorded in a campaign's job_args.json,
      refuses a range overlapping a reserved ledger span, survives a truncated
      job_args.json, and raises rather than colliding when the space is
      saturated. A guard that never fires is a guard that was never tested.

  T5  REPLAY. --override reproduces a past allocation exactly, and the ledger
      round-trips, so a campaign remains reproducible after the fact.

  T6  BOUNDS. Every allocated seed stays inside [1, 2**31-1), the ceiling
      _resolve_seed_master respects on both of its fallback paths.

HOW TO RUN
----------
    cd <repo>/hpc/Phenomenological_finalv1
    python3 smoke_test_seed_allocation.py

    python3 smoke_test_seed_allocation.py -v      # show per-campaign detail

Needs numpy (for the real sample_theta) but NOT brian2: everything under test
is seeding and bookkeeping. Runs on a login node in a few seconds.

Expected final line on success:

    ALL 6 TEST GROUPS PASSED
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

import matplotlib
matplotlib.use('Agg')

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import HPC_main_sweep as S        # noqa: E402
import seed_alloc                 # noqa: E402

SEED_ALLOC_PY = os.path.join(_HERE, 'seed_alloc.py')

_FAILS = []
_CHECKS = [0]


def check(cond, label):
    _CHECKS[0] += 1
    print(('    ok   ' if cond else '    FAIL ') + label)
    if not cond:
        _FAILS.append(label)


# ---------------------------------------------------------------------------
# A faithful replay of one task's outer-loop draw sequence.
# ---------------------------------------------------------------------------
def task_theta_set(seed_master, n_topologies=2, sims_per_topo=8,
                   conn_rule='weibull'):
    """
    Replay the master_rng consumption of ONE task exactly as HPC_main_sweep's
    outer loop does it (post-conn_prob-guard), and return the set of parameter
    vectors it would produce.

    Order matters: any deviation would make the duplication measurement wrong.
        [flat only] conn_prob   <- uniform
        kernel vector           <- sample_kernel_vector      (weibull only)
        topo_seed_base          <- integers
        theta_matrix            <- sims_per_topo x sample_theta
        seed_runs               <- integers(size=sims_per_topo)
    """
    rng = np.random.default_rng(seed_master)
    thetas = set()
    for _ in range(n_topologies):
        if conn_rule == 'flat':
            rng.uniform(0.1, 0.6)
        else:
            S.sample_kernel_vector(rng, S.KERNEL_BOUNDS)
        rng.integers(0, 2**31 - 1)                       # topo_seed_base
        for _ in range(sims_per_topo):
            thetas.add(tuple(S.sample_theta(rng).tolist()))
        rng.integers(0, 2**31 - 1, size=sims_per_topo)   # seed_runs
    return thetas


def campaign_thetas(seed_base, n_tasks, **kw):
    """Union of every task's parameter vectors, for SEED_MASTER = base + idx."""
    out = set()
    for idx in range(n_tasks):
        out |= task_theta_set(seed_base + idx, **kw)
    return out


# ---------------------------------------------------------------------------
# T1 -- reproduce the bug
# ---------------------------------------------------------------------------
def t1_reproduce_bug(verbose=False):
    print('\n[T1] THE BUG: hardcoded base + restarting array index')
    print('     -> prefix containment across campaigns')
    BASE = 3000000                       # the value hardcoded for the intel queue
    # Task counts follow the measured campaigns (v1=32, v3=54, v4=12, v7=27),
    # scaled down by 4 so the test runs in seconds. The PATTERN is what matters.
    campaigns = {'v1': 8, 'v3': 14, 'v4': 3, 'v7': 6}
    sets = {k: campaign_thetas(BASE, n) for k, n in campaigns.items()}

    total_rows = sum(len(s) for s in sets.values())
    pooled = set().union(*sets.values())
    dup_factor = total_rows / len(pooled)
    print(f'     rows={total_rows}  distinct={len(pooled)}  D={dup_factor:.2f}')

    check(dup_factor > 1.5,
          f'duplication factor D={dup_factor:.2f} > 1.5 (bug reproduced)')
    longest = max(campaigns, key=lambda k: campaigns[k])
    for k in campaigns:
        if k == longest:
            continue
        contained = sets[k] <= sets[longest]
        pct = 100.0 * len(sets[k] & sets[longest]) / max(len(sets[k]), 1)
        check(contained,
              f'{k} is 100% contained in {longest} ({pct:.1f}%) '
              f'-- matches the measured v4/v7/v8-in-v3 pattern')
    if verbose:
        for k in campaigns:
            print(f'       {k}: {campaigns[k]} tasks, {len(sets[k])} distinct theta')


# ---------------------------------------------------------------------------
# T2 -- seed quality is fine; reuse was the problem
# ---------------------------------------------------------------------------
def t2_seed_quality():
    print('\n[T2] consecutive seeds already give independent streams')
    print('     (numpy SeedSequence mixes an integer seed before use)')
    BASE = 3000000
    a = task_theta_set(BASE + 0)
    b = task_theta_set(BASE + 1)
    c = task_theta_set(BASE + 2)
    check(not (a & b), 'seeds base+0 and base+1 share NO parameter vectors')
    check(not (a & c), 'seeds base+0 and base+2 share NO parameter vectors')
    check(a == task_theta_set(BASE + 0), 'same seed reproduces exactly')

    xs = np.random.default_rng(BASE).random(20000)
    ys = np.random.default_rng(BASE + 1).random(20000)
    r = float(np.corrcoef(xs, ys)[0, 1])
    check(abs(r) < 0.05,
          f'adjacent-seed streams are uncorrelated (r={r:+.4f}, |r|<0.05)')
    print('     => the arithmetic SEED_BASE + IDX is sound; only the constant')
    print('        base had to change.')


# ---------------------------------------------------------------------------
# T3 -- the fix
# ---------------------------------------------------------------------------
def alloc_via_cli(root, ledger, span=400000, override=None, tag='t'):
    """Call the real seed_alloc.py as a subprocess, exactly as the launcher does."""
    cmd = [sys.executable, SEED_ALLOC_PY, '--span', str(span),
           '--root', root, '--ledger', ledger, '--tag', tag, '--quiet']
    if override is not None:
        cmd += ['--override', str(override)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def t3_fix_gives_disjoint_launches(verbose=False):
    print('\n[T3] THE FIX: repeated launches get disjoint, non-repeating seeds')
    SPAN, SPACING, N_LAUNCH, N_TASKS = 400000, 100000, 6, 5
    with tempfile.TemporaryDirectory() as td:
        ledger = os.path.join(td, 'artifacts', 'seed_ledger.tsv')
        bases, all_seeds, all_thetas = [], [], set()
        n_theta_rows = 0
        for i in range(N_LAUNCH):
            rc, out, _ = alloc_via_cli(td, ledger, SPAN, tag=f'camp_v{i}')
            check(rc == 0, f'launch {i}: seed_alloc.py exited 0')
            base = int(out)
            bases.append(base)
            # Simulate the launcher's per-queue offsets and the worker's +IDX.
            seeds = [base + q * SPACING + idx
                     for q in range(SPAN // SPACING) for idx in range(N_TASKS)]
            all_seeds += seeds
            # Write job_args.json so the NEXT allocation sees these as used.
            for s in seeds:
                d = os.path.join(td, f'campaign_camp_v{i}', f'sweep_x_task{s}')
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, 'job_args.json'), 'w') as fh:
                    json.dump({'_resolved_seed_master': s}, fh)
            th = set()
            for s in seeds[:4]:            # 4 tasks is enough to detect reuse
                th |= task_theta_set(s)
            n_theta_rows += len(th)
            all_thetas |= th
            if verbose:
                print(f'       launch {i}: base={base:>11d}  '
                      f'seeds {min(seeds)}..{max(seeds)}')

        check(len(set(bases)) == N_LAUNCH,
              f'all {N_LAUNCH} launch bases distinct')
        check(len(set(all_seeds)) == len(all_seeds),
              f'all {len(all_seeds)} seed_master values distinct (0 repeats)')
        spans = sorted((b, b + SPAN) for b in bases)
        overlaps = [(spans[i], spans[i + 1]) for i in range(len(spans) - 1)
                    if spans[i][1] > spans[i + 1][0]]
        check(not overlaps, 'no two launch spans overlap')
        check(len(all_thetas) == n_theta_rows,
              f'zero duplicated parameter vectors across launches '
              f'({len(all_thetas)} distinct of {n_theta_rows}); D=1.00')


# ---------------------------------------------------------------------------
# T4 -- the guard actually fires
# ---------------------------------------------------------------------------
def t4_guard_fires():
    print('\n[T4] the disjointness guard fires (negative paths)')
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, 'campaign_x', 'sweep_x_task0')
        os.makedirs(d)
        with open(os.path.join(d, 'job_args.json'), 'w') as fh:
            json.dump({'_resolved_seed_master': 500000}, fh)
        used, n_ok, n_bad = seed_alloc.used_seeds_from_job_args(td)
        check(used == {500000}, 'recorded seed_master is read back')
        check((n_ok, n_bad) == (1, 0), 'file counted as readable')
        check(not seed_alloc.range_is_free(490000, 100000, used, []),
              'a range containing a recorded seed is REJECTED')
        check(seed_alloc.range_is_free(600001, 100000, used, []),
              'a clear range is accepted')

        reserved = [(1000, 2000, 'prev')]
        check(not seed_alloc.range_is_free(1500, 100, set(), reserved),
              'a range overlapping a reserved ledger span is REJECTED')
        check(not seed_alloc.range_is_free(900, 200, set(), reserved),
              'partial overlap at the low edge is REJECTED')
        check(seed_alloc.range_is_free(2000, 100, set(), reserved),
              'a range starting exactly at the reserved end is accepted')

        # A truncated job_args.json (walltime-killed job) is skipped, not fatal.
        d2 = os.path.join(td, 'campaign_x', 'sweep_x_task1')
        os.makedirs(d2)
        with open(os.path.join(d2, 'job_args.json'), 'w') as fh:
            fh.write('{"_resolved_seed_master": 12')
        used2, n_ok2, n_bad2 = seed_alloc.used_seeds_from_job_args(td)
        check(n_bad2 == 1 and used2 == {500000},
              'a truncated job_args.json is skipped, not fatal')

        # A SATURATED space must fail loudly rather than hand back a clash.
        # Seeds every 50 across [0, 300] leave no free window of width 100,
        # since any window [lo, lo+100) with lo in [1, 199] must contain one.
        full = tempfile.mkdtemp()
        for k, s in enumerate(range(0, 301, 50)):
            d3 = os.path.join(full, 'campaign_f', f'sweep_f_task{k}')
            os.makedirs(d3)
            with open(os.path.join(d3, 'job_args.json'), 'w') as fh:
                json.dump({'_resolved_seed_master': s}, fh)
        try:
            seed_alloc.allocate(span=100, seed_max=300, root=full,
                                attempts=64, verbose=False)
            raised = False
        except RuntimeError:
            raised = True
        check(raised,
              'a saturated seed space raises RuntimeError rather than '
              'returning a colliding base')

        # CLI surface: a span that cannot fit must exit non-zero and print
        # NOTHING to stdout, so the launcher's -z check aborts the launch.
        r = subprocess.run(
            [sys.executable, SEED_ALLOC_PY, '--span', str(2**31), '--root', td,
             '--quiet'], capture_output=True, text=True)
        check(r.returncode != 0, 'CLI exits non-zero when the span cannot fit')
        check(r.stdout.strip() == '',
              'CLI prints NO base on failure (launcher aborts on empty output)')


# ---------------------------------------------------------------------------
# T5 -- replay
# ---------------------------------------------------------------------------
def t5_replay():
    print('\n[T5] replay: --override reproduces a past launch exactly')
    with tempfile.TemporaryDirectory() as td:
        ledger = os.path.join(td, 'artifacts', 'seed_ledger.tsv')
        rc, out, _ = alloc_via_cli(td, ledger, 400000, tag='orig')
        base = int(out)
        rc2, out2, _ = alloc_via_cli(td, ledger, 400000, override=base,
                                     tag='replay')
        check(rc2 == 0 and int(out2) == base,
              f'--override returns the same base ({base})')
        check(task_theta_set(base) == task_theta_set(int(out2)),
              'the replayed base reproduces identical parameter vectors')
        spans = seed_alloc.reserved_spans_from_ledger(ledger)
        check(len(spans) == 2, f'ledger has both rows ({len(spans)})')
        check(all(lo == base for lo, hi, tag in spans),
              'both ledger rows record the same span')
        tags = sorted(t for _, _, t in spans)
        check(tags == ['orig', 'replay'], f'ledger tags round-trip ({tags})')


# ---------------------------------------------------------------------------
# T6 -- bounds
# ---------------------------------------------------------------------------
def t6_bounds():
    print('\n[T6] every allocated seed stays inside [1, 2**31-1)')
    SPAN = 400000
    with tempfile.TemporaryDirectory() as td:
        ledger = os.path.join(td, 'artifacts', 'seed_ledger.tsv')
        lo_ok = hi_ok = True
        for _ in range(50):
            b = seed_alloc.allocate(SPAN, root=td, ledger_path=ledger,
                                    verbose=False)
            lo_ok &= (b >= 1)
            hi_ok &= (b + SPAN < seed_alloc.SEED_MAX_DEFAULT)
        check(lo_ok, '50 allocations all >= 1')
        check(hi_ok, '50 allocations all end below 2**31-1')
        b = seed_alloc.allocate(SPAN, root=td, ledger_path=ledger, verbose=False)
        check(np.random.default_rng(b) is not None,
              'an allocated base is a valid numpy seed')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()

    print('=' * 70)
    print('smoke_test_seed_allocation -- per-launch seed range allocation')
    print('=' * 70)
    print(f'  seed_alloc.py         : {SEED_ALLOC_PY}')
    print(f'  SEED_MAX              : {seed_alloc.SEED_MAX_DEFAULT}')
    print(f'  registry N_DIMS       : {S.N_DIMS}')
    print(f'  numpy                 : {np.__version__}')

    t1_reproduce_bug(a.verbose)
    t2_seed_quality()
    t3_fix_gives_disjoint_launches(a.verbose)
    t4_guard_fires()
    t5_replay()
    t6_bounds()

    print('\n' + '=' * 70)
    if _FAILS:
        print(f'{len(_FAILS)} of {_CHECKS[0]} CHECKS FAILED:')
        for f in _FAILS:
            print('  - ' + f)
        print('=' * 70)
        return 1
    print(f'ALL 6 TEST GROUPS PASSED  ({_CHECKS[0]} checks)')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
