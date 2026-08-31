#!/usr/bin/env python3
# =============================================================================
# bench_report.py -- turn the MEAMA sizing benchmark into campaign parameters.
#
# Reads the per-simulation timing that HPC_main_sweep.py already persists
# (t_compile_s and t_run_s in every topo_*/iter_*.json; t_topo_build_s in every
# topo_*/topology_meta.json) and answers three questions:
#
#   Q1  What does one Full-mode tripartite simulation cost at each of the six
#       (c_max, simtime) configurations, at rho_total = 1600 cells/mm^2?
#
#   Q2  What does k = n_params_per_worker cost?  A worker compiles ONCE per
#       topology and then runs k parameter vectors through the same binary, so
#       the compile is amortised over k simulations.  Lowering k buys more
#       distinct topologies (finer resolution on the topology-level axes
#       conn_prob and stoa_gate_p) and pays for it in compile overhead.  The
#       trade is computed here analytically from the SAME measurement -- it
#       does not need its own benchmark run.
#
#   Q3  Given a target number of simulations and a per-task walltime budget,
#       how many topologies per task, how many tasks, and how many distinct
#       topologies does the campaign end up with?
#
# MODEL (stated explicitly; see the accompanying document, section 5).
# For one topology block with W workers each running k parameter vectors:
#
#     T_block(k)  =  t_topo  +  t_compile  +  k * t_run                    (1)
#     sims/block  =  W * k                                                 (2)
#     node-seconds per sim  =  T_block(k) / (W * k)                        (3)
#
# t_topo is the SERIAL per-topology overhead seen by the main process. It is
# NOT read from t_topo_build_s, which times the geometry build ONLY and omits
# the two spatial PNGs, the topology.npz write and the pool spawn/join -- on
# measured data those omitted steps dominate. It is instead RECOVERED from the
# per-cell wall clock recorded in bench_cells.jsonl:
#
#     t_topo = [T_cell_wall - n_topo*(t_compile + k*t_run_mean)] / n_topo   (0)
#
# which also absorbs one-off job startup (imports, device init, manifest
# rebuild) amortised over n_topo, and is therefore an OVER-estimate -- the
# conservative direction for walltime sizing. t_compile is the
# per-worker C++ build, paid once per worker per topology and overlapped
# across workers; t_run is one simulation.  Equation (1) assumes the workers
# are balanced -- for WALLTIME sizing the script therefore uses a high
# percentile of t_run (the block ends when the SLOWEST worker ends), while for
# THROUGHPUT it uses the median.  Both are reported.
#
# Usage:
#     python bench_report.py <BENCH_ROOT> [--target 1000000] [--hours 20]
#                            [--workers 192] [--k 1 2 3 5] [--csv out.csv]
#     python bench_report.py --selftest      # synthetic-data smoke test
# =============================================================================

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Queue table: (name, ncpus, max_concurrent_tasks). Mirrors launch_MEAMA.sh.
# Only used for the calendar-time projection at the end.
# ---------------------------------------------------------------------------
DEFAULT_QUEUES = [('cfd', 192, 2), ('intel', 48, 5), ('cpu', 48, 5)]


# ---------------------------------------------------------------------------
# Small statistics helpers (no scipy dependency on the login node)
# ---------------------------------------------------------------------------
def _percentile(xs, q):
    """Linear-interpolated percentile of a non-empty list, q in [0, 100]."""
    if not xs:
        return float('nan')
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    pos = (len(ys) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ys[lo])
    return float(ys[lo] + (ys[hi] - ys[lo]) * (pos - lo))


def _median(xs):
    return _percentile(xs, 50.0)


def _mean(xs):
    return float(sum(xs)) / len(xs) if xs else float('nan')


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_cell(cell_dir):
    """
    Parse one benchmark cell directory (c<CMAX>_t<SIMTIME>/).

    Returns a dict of measurements, or None if the cell produced no
    simulations at all (crashed / walltime-killed before any iter completed).
    """
    cell_dir = Path(cell_dir)
    t_run, t_compile_by_worker, topo_rows = [], {}, []

    for meta_path in sorted(cell_dir.glob('topo_*/topology_meta.json')):
        try:
            m = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        topo_rows.append({
            'topo_idx':     m.get('topo_idx'),
            't_topo_build': float(m.get('t_topo_build_s', float('nan'))),
            'Nn':           m.get('Nn'),
            'Na':           m.get('Na'),
            'conn_prob':    m.get('conn_prob'),
            'stoa_gate_p':  m.get('stoa_gate_p'),
            'n_synapses':   m.get('n_synapses'),
            'n_gj_links':   m.get('n_gj_links'),
            'n_stoa_links': m.get('n_stoa_links'),
            'n_stoa_viable': m.get('n_stoa_viable'),
        })

    for it_path in sorted(cell_dir.glob('topo_*/iter_*.json')):
        try:
            d = json.loads(it_path.read_text())
        except (OSError, ValueError):
            continue
        if 't_run_s' in d:
            t_run.append(float(d['t_run_s']))
        # One compile per (topology, worker): de-duplicate on that key so the
        # compile mean is not weighted by k.
        if 't_compile_s' in d:
            key = (d.get('topo_idx'), d.get('worker_id'))
            t_compile_by_worker[key] = float(d['t_compile_s'])

    if not t_run:
        return None

    t_compile = list(t_compile_by_worker.values())
    t_topo = [r['t_topo_build'] for r in topo_rows
              if not math.isnan(r['t_topo_build'])]

    name = cell_dir.name                       # c350_t200
    c_max = simtime = None
    try:
        c_part, t_part = name.split('_')
        c_max = int(c_part.lstrip('c'))
        simtime = int(t_part.lstrip('t'))
    except (ValueError, AttributeError):
        pass

    return {
        'cell':        name,
        'c_max':       c_max,
        'simtime':     simtime,
        'n_sims':      len(t_run),
        'n_compiles':  len(t_compile),
        'n_topos':     len(topo_rows),
        't_run_med':   _median(t_run),
        't_run_mean':  _mean(t_run),
        't_run_p90':   _percentile(t_run, 90.0),
        't_run_max':   max(t_run),
        't_run_min':   min(t_run),
        't_compile':   _mean(t_compile) if t_compile else float('nan'),
        't_topo_build': _mean(t_topo) if t_topo else float('nan'),
        't_topo':      _mean(t_topo) if t_topo else float('nan'),  # refined below
        'Nn':          topo_rows[0]['Nn'] if topo_rows else None,
        'Na':          topo_rows[0]['Na'] if topo_rows else None,
        'n_syn_med':   _median([r['n_synapses'] for r in topo_rows
                                if r['n_synapses'] is not None]),
        'n_stoa_med':  _median([r['n_stoa_links'] for r in topo_rows
                                if r['n_stoa_links'] is not None]),
        'topo_rows':   topo_rows,
        'real_time_factor': (_median(t_run) / simtime) if simtime else float('nan'),
    }


def parse_bench_root(root):
    """Parse every c*_t* cell under a benchmark root. Returns (cfg, [cells])."""
    root = Path(root)
    if not root.is_dir():
        sys.exit(f'not a directory: {root}')
    cfg_path = root / 'bench_config.json'
    if not cfg_path.is_file():
        sys.exit(f'missing {cfg_path}. Is {root} really a bench_sizing.sh output '
                 f'root? (Expected MEAMA/bench_out/bench_<timestamp>/.)')
    cfg = json.loads(cfg_path.read_text())

    # Per-cell wall clock, written by bench_sizing.sh after each config.
    walls = {}
    jl = root / 'bench_cells.jsonl'
    if jl.is_file():
        for line in jl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                walls[r['cell']] = float(r['wall_s'])
            except (ValueError, KeyError):
                continue

    k_bench = float(cfg.get('k_bench', 1) or 1)

    cells = []
    for d in sorted(root.glob('c*_t*')):
        if not d.is_dir():
            continue
        c = parse_cell(d)
        if c is None:
            print(f'  [warn] {d.name}: no completed simulations, skipped',
                  file=sys.stderr)
            continue
        c['wall_s'] = walls.get(c['cell'])
        c['t_topo'] = _serial_overhead(c, k_bench)
        cells.append(c)
    if not cells:
        sys.exit('no benchmark cell produced any completed simulation.')
    return cfg, cells


def _serial_overhead(cell, k_bench):
    """
    Serial per-topology overhead, equation (0): everything in the cell's wall
    clock that is NOT compile and NOT simulation, divided by the number of
    topologies. Falls back to the (under-estimating) t_topo_build_s when the
    wall clock is unavailable or the arithmetic goes negative -- which happens
    when workers are badly imbalanced, since equation (1) assumes balance.
    """
    fallback = cell.get('t_topo_build', float('nan'))
    wall, n_topo = cell.get('wall_s'), cell.get('n_topos', 0)
    if wall is None or not n_topo:
        return fallback
    busy = n_topo * (cell['t_compile'] + k_bench * cell['t_run_mean'])
    est = (wall - busy) / n_topo
    if not math.isfinite(est) or est < 0:
        return fallback
    return max(est, fallback if math.isfinite(fallback) else 0.0)


# ---------------------------------------------------------------------------
# The sizing model
# ---------------------------------------------------------------------------
def block_seconds(t_topo, t_compile, t_run, k):
    """T_block(k), equation (1). Serial topo overhead + compile + k runs."""
    return t_topo + t_compile + k * t_run


def size_campaign(cell, k, workers, hours, target):
    """
    Campaign sizing for one benchmark cell at a given k and node width.

    Throughput uses the MEDIAN run time; walltime safety uses the p90 (a
    topology block ends when the slowest of `workers` workers ends).
    """
    t_topo, t_comp = cell['t_topo'], cell['t_compile']
    blk_med = block_seconds(t_topo, t_comp, cell['t_run_med'], k)
    blk_p90 = block_seconds(t_topo, t_comp, cell['t_run_p90'], k)

    sims_per_block = workers * k
    node_s_per_sim = blk_med / sims_per_block
    overhead_frac = (t_topo + t_comp) / blk_med if blk_med > 0 else float('nan')

    budget_s = hours * 3600.0
    n_topo_task = max(1, int(budget_s // blk_p90))       # p90: do not overrun
    sims_per_task = n_topo_task * sims_per_block
    n_tasks = math.ceil(target / sims_per_task) if sims_per_task else float('inf')
    n_distinct_topologies = math.ceil(target / sims_per_block)

    return {
        'k': k, 'workers': workers,
        't_block_med': blk_med, 't_block_p90': blk_p90,
        'sims_per_block': sims_per_block,
        'node_s_per_sim': node_s_per_sim,
        'overhead_frac': overhead_frac,
        'n_topologies_per_task': n_topo_task,
        'sims_per_task': sims_per_task,
        'n_tasks': n_tasks,
        'n_distinct_topologies': n_distinct_topologies,
        'node_hours_total': target * node_s_per_sim / 3600.0,
    }


def calendar_days(node_hours_total, queues):
    """
    Wall-clock days if all queues run at full declared concurrency.
    node_hours here are per-NODE hours at that node's width, so convert
    through the width used for the sizing.
    """
    total_nodes = sum(conc for _, _, conc in queues)
    if total_nodes <= 0:
        return float('nan')
    return node_hours_total / total_nodes / 24.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_measurements(cfg, cells):
    print('=' * 100)
    print('MEASURED  (rho_total = %s cells/mm^2, astro fraction %s '
          '-> %s neurons + %s astro per mm^2)'
          % (cfg.get('rho_total_per_mm2'), cfg.get('astro_fraction'),
             cfg.get('density_neuron_per_mm2'), cfg.get('density_astro_per_mm2')))
    print('mode=%s  group=%s  rule=%s  gate=U[%s, %s]  seed_master=%s'
          % (cfg.get('mode'), cfg.get('sweep_group'), cfg.get('conn_rule'),
             cfg.get('stoa_gate_lo'), cfg.get('stoa_gate_hi'),
             cfg.get('seed_master')))
    print('=' * 100)
    hdr = (f"{'cell':<11}{'Nn':>5}{'Na':>5}{'syn':>7}{'stoa':>7}{'sims':>6}"
           f"{'t_run med':>11}{'p90':>9}{'max':>9}{'t_comp':>9}{'t_serial':>10}"
           f"{'(build)':>9}{'x real':>8}")
    print(hdr)
    print('-' * len(hdr))
    for c in cells:
        print(f"{c['cell']:<11}{str(c['Nn']):>5}{str(c['Na']):>5}"
              f"{c['n_syn_med']:>7.0f}{c['n_stoa_med']:>7.0f}{c['n_sims']:>6}"
              f"{c['t_run_med']:>11.1f}{c['t_run_p90']:>9.1f}{c['t_run_max']:>9.1f}"
              f"{c['t_compile']:>9.1f}{c['t_topo']:>10.1f}"
              f"{c['t_topo_build']:>9.1f}"
              f"{c['real_time_factor']:>8.2f}")
    print('\n  t_* in seconds. "x real" = t_run / simtime (>1 = slower than real time).')
    print('  t_comp is per worker per topology (de-duplicated). t_serial is the')
    print('  FULL serial per-topology overhead recovered from the cell wall clock,')
    print('  eq. (0): geometry + both PNGs + topology.npz + pool spawn + amortised')
    print('  job startup. "(build)" is t_topo_build_s alone, shown only to make')
    print('  the gap visible -- the model uses t_serial, never (build).')


def print_k_tradeoff(cells, ks, workers, hours, target):
    print()
    print('=' * 100)
    print(f'k TRADE-OFF   (W = {workers} workers/node, budget {hours} h/task, '
          f'target {target:,} sims)')
    print('=' * 100)
    for c in cells:
        print(f"\n  {c['cell']}  (c_max={c['c_max']} um, simtime={c['simtime']} s)")
        hdr = (f"    {'k':>3}{'sims/topo':>11}{'T_block med':>13}{'overhead':>10}"
               f"{'node-s/sim':>12}{'topo/task':>11}{'tasks':>8}"
               f"{'distinct topo':>15}{'node-h':>10}")
        print(hdr)
        print('    ' + '-' * (len(hdr) - 4))
        base = None
        for k in ks:
            s = size_campaign(c, k, workers, hours, target)
            if base is None:
                base = s['node_s_per_sim']
            rel = s['node_s_per_sim'] / base
            print(f"    {k:>3}{s['sims_per_block']:>11}"
                  f"{s['t_block_med']:>13.0f}{100 * s['overhead_frac']:>9.1f}%"
                  f"{s['node_s_per_sim']:>12.1f}"
                  f"{s['n_topologies_per_task']:>11}{s['n_tasks']:>8}"
                  f"{s['n_distinct_topologies']:>15,}"
                  f"{s['node_hours_total']:>10.0f}"
                  + ('' if k == ks[0] else f'   ({rel:.2f}x cost of k={ks[0]})'))


def print_recommendation(cells, workers, hours, target, queues):
    print()
    print('=' * 100)
    print('CAMPAIGN SIZING at k = 1  (one parameter vector per worker per topology)')
    print('=' * 100)
    hdr = (f"{'cell':<11}{'N_TOPOLOGIES':>14}{'sims/task':>11}{'tasks':>8}"
           f"{'distinct topo':>15}{'node-hours':>12}{'~days':>8}")
    print(hdr)
    print('-' * len(hdr))
    for c in cells:
        s = size_campaign(c, 1, workers, hours, target)
        print(f"{c['cell']:<11}{s['n_topologies_per_task']:>14}"
              f"{s['sims_per_task']:>11,}{s['n_tasks']:>8}"
              f"{s['n_distinct_topologies']:>15,}"
              f"{s['node_hours_total']:>12,.0f}"
              f"{calendar_days(s['node_hours_total'], queues):>8.1f}")
    print('\n  N_TOPOLOGIES is sized on the p90 run time so a task finishes inside')
    print('  the budget; "~days" assumes every queue runs at full concurrency and')
    print('  ignores queue wait, which on a busy cluster usually dominates.')
    print('  Set N_TOPOLOGIES and K_PARAMS in MEAMA/launch_MEAMA.sh from the row')
    print('  you choose, then flip SIZING_CONFIRMED=1.')


def write_csv(path, cells, ks, workers, hours, target):
    cols = ['cell', 'c_max', 'simtime', 'Nn', 'Na', 'n_sims',
            't_run_med', 't_run_p90', 't_run_max', 't_compile', 't_topo',
            'real_time_factor', 'k', 't_block_med', 'node_s_per_sim',
            'overhead_frac', 'n_topologies_per_task', 'sims_per_task',
            'n_tasks', 'n_distinct_topologies', 'node_hours_total']
    with open(path, 'w', encoding='ascii') as fh:
        fh.write(','.join(cols) + '\n')
        for c in cells:
            for k in ks:
                s = size_campaign(c, k, workers, hours, target)
                row = {**c, **s}
                fh.write(','.join(
                    ('' if row.get(col) is None else f'{row.get(col)}')
                    for col in cols) + '\n')
    print(f'\n[csv] wrote {path}')


# ---------------------------------------------------------------------------
# Self-test: synthetic tree, known answers
# ---------------------------------------------------------------------------
def _selftest():
    import shutil
    import tempfile
    n_pass = n_fail = 0

    def check(cond, msg):
        nonlocal n_pass, n_fail
        if cond:
            n_pass += 1
            print(f'  ok   {msg}')
        else:
            n_fail += 1
            print(f'  FAIL {msg}')

    tmp = Path(tempfile.mkdtemp(prefix='benchtest_'))
    try:
        root = tmp / 'bench_TEST'
        root.mkdir(parents=True)
        (root / 'bench_config.json').write_text(json.dumps({
            'rho_total_per_mm2': 1600, 'astro_fraction': 0.5,
            'density_neuron_per_mm2': 800, 'density_astro_per_mm2': 800,
            'mode': 'Full', 'sweep_group': 'tripartite', 'conn_rule': 'flat',
            'stoa_gate_lo': 0.2, 'stoa_gate_hi': 1.0, 'seed_master': 1,
            'k_bench': 3, 'n_bench_workers': 2, 'n_topologies_bench': 2}))

        # One cell, 2 topologies x 2 workers x 3 params.
        # t_run = 100 s exactly; t_compile = 60 s; t_topo = 10 s.
        cell = root / 'c400_t200'
        for ti in range(2):
            td = cell / f'topo_{ti:05d}'
            td.mkdir(parents=True)
            (td / 'topology_meta.json').write_text(json.dumps({
                'topo_idx': ti, 't_topo_build_s': 10.0, 'Nn': 128, 'Na': 128,
                'conn_prob': 0.3, 'stoa_gate_p': 0.5, 'n_synapses': 4000,
                'n_gj_links': 500, 'n_stoa_links': 2000, 'n_stoa_viable': 4000}))
            for wid in range(2):
                for it in range(3):
                    (td / f'iter_{wid:02d}{it:02d}.json').write_text(json.dumps({
                        'topo_idx': ti, 'worker_id': wid,
                        't_compile_s': 60.0, 't_run_s': 100.0}))

        # Cell wall clock: 2 topologies x (t_serial 25 + compile 60 + 3*100)
        # = 2 * 385 = 770 s. The recovered t_serial must be 25, NOT the 10 s
        # t_topo_build_s, which omits the PNGs.
        (root / 'bench_cells.jsonl').write_text(json.dumps(
            {'cell': 'c400_t200', 'c_max': 400, 'simtime': 200,
             'rc': 0, 'wall_s': 770.0}) + '\n')

        cfg, cells = parse_bench_root(root)
        c = cells[0]
        check(len(cells) == 1, 'one cell parsed')
        check(c['n_sims'] == 12, f"12 sims counted (got {c['n_sims']})")
        check(c['n_compiles'] == 4,
              f"compiles de-duplicated per (topo, worker): 4 (got {c['n_compiles']})")
        check(abs(c['t_run_med'] - 100.0) < 1e-9, 't_run median = 100 s')
        check(abs(c['t_compile'] - 60.0) < 1e-9, 't_compile mean = 60 s')
        check(abs(c['t_topo_build'] - 10.0) < 1e-9,
              't_topo_build_s (build only) = 10 s')
        check(abs(c['t_topo'] - 25.0) < 1e-9,
              f"serial overhead RECOVERED from wall clock = 25 s, not the 10 s "
              f"build time (got {c['t_topo']:.3f})")
        check(abs(c['real_time_factor'] - 0.5) < 1e-9,
              'real-time factor = 100/200 = 0.5')
        check(c['c_max'] == 400 and c['simtime'] == 200,
              'c_max/simtime recovered from the directory name')

        # T_block(k) = 25 + 60 + 100k
        s1 = size_campaign(c, 1, workers=100, hours=20, target=1_000_000)
        s5 = size_campaign(c, 5, workers=100, hours=20, target=1_000_000)
        check(abs(s1['t_block_med'] - 185.0) < 1e-9, 'T_block(k=1) = 185 s')
        check(abs(s5['t_block_med'] - 585.0) < 1e-9, 'T_block(k=5) = 585 s')
        check(abs(s1['node_s_per_sim'] - 1.85) < 1e-9,
              'node-s/sim at k=1 = 185/(100*1) = 1.85')
        check(abs(s5['node_s_per_sim'] - 1.17) < 1e-9,
              'node-s/sim at k=5 = 585/(100*5) = 1.17')
        check(abs(s1['node_s_per_sim'] / s5['node_s_per_sim'] - 185.0 / 117.0) < 1e-9,
              'k=1 / k=5 cost ratio = (t_serial+t_comp+t_run)/(t_serial/5+t_comp/5+t_run)')
        check(s1['n_distinct_topologies'] == 10_000,
              f"k=1 gives 1e6/100 = 10,000 distinct topologies "
              f"(got {s1['n_distinct_topologies']:,})")
        check(s5['n_distinct_topologies'] == 2_000,
              f"k=5 gives 1e6/500 = 2,000 distinct topologies "
              f"(got {s5['n_distinct_topologies']:,})")
        check(s1['n_topologies_per_task'] == int(20 * 3600 // 185),
              'N_TOPOLOGIES per task uses the p90 block time and floors')
        check(s1['sims_per_task'] == s1['n_topologies_per_task'] * 100,
              'sims/task = topologies * W * k')
        check(s1['n_tasks'] == math.ceil(1_000_000 / s1['sims_per_task']),
              'task count covers the target')

        # p90 == median here (all runs identical), so both block times agree.
        check(abs(s1['t_block_p90'] - s1['t_block_med']) < 1e-9,
              'degenerate timing: p90 block == median block')

        # A cell with no completed sims must be skipped, not crash.
        (root / 'c300_t200').mkdir()
        cfg2, cells2 = parse_bench_root(root)
        check(len(cells2) == 1, 'empty cell skipped with a warning, no crash')

        # Missing wall clock must fall back to the build time, not crash.
        (root / 'bench_cells.jsonl').unlink()
        _, cells3 = parse_bench_root(root)
        check(abs(cells3[0]['t_topo'] - 10.0) < 1e-9,
              'no bench_cells.jsonl -> falls back to t_topo_build_s')

        # A wall clock too small to be consistent must not go negative.
        (root / 'bench_cells.jsonl').write_text(json.dumps(
            {'cell': 'c400_t200', 'rc': 0, 'wall_s': 1.0}) + '\n')
        _, cells4 = parse_bench_root(root)
        check(cells4[0]['t_topo'] >= 0.0 and math.isfinite(cells4[0]['t_topo']),
              'inconsistent wall clock -> non-negative fallback, no crash')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f'\nRESULT: {n_pass} passed, {n_fail} failed')
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description='Turn the MEAMA sizing benchmark into campaign parameters.')
    ap.add_argument('bench_root', nargs='?',
                    help='MEAMA/bench_out/bench_<timestamp>/')
    ap.add_argument('--target', type=int, default=1_000_000,
                    help='target simulations for the campaign (default: %(default)s)')
    ap.add_argument('--hours', type=float, default=20.0,
                    help='compute budget per task, hours (default: %(default)s)')
    ap.add_argument('--workers', type=int, default=192,
                    help='workers per node = ncpus of the node you are sizing '
                         'for (default: %(default)s)')
    ap.add_argument('--k', type=int, nargs='+', default=[1, 2, 3, 5],
                    help='n_params_per_worker values to compare '
                         '(default: %(default)s)')
    ap.add_argument('--csv', default=None, help='also write a CSV here')
    ap.add_argument('--selftest', action='store_true',
                    help='run the synthetic-data smoke test and exit')
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.bench_root:
        ap.error('bench_root is required (or use --selftest)')

    cfg, cells = parse_bench_root(args.bench_root)
    cells.sort(key=lambda c: (c['c_max'] or 0, c['simtime'] or 0))

    print_measurements(cfg, cells)
    print_k_tradeoff(cells, args.k, args.workers, args.hours, args.target)
    print_recommendation(cells, args.workers, args.hours, args.target,
                         DEFAULT_QUEUES)
    if args.csv:
        write_csv(args.csv, cells, args.k, args.workers, args.hours, args.target)
    return 0


if __name__ == '__main__':
    sys.exit(main())
