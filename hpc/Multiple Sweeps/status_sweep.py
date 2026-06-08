#!/usr/bin/env python3
"""
status_sweep.py — comprehensive real-time status for a 300k sweep campaign.

Reads per-task manifest.json files (fast, one per task) and randomly samples
per-iteration JSON sidecars for quality statistics. No npz files are opened.

Usage:
    python status_sweep.py campaign_300k_v1
    python status_sweep.py campaign_300k_v1 --target 300000 --sample 8000
    python status_sweep.py campaign_300k_v1 --json          # machine-readable
    watch -n 300 python status_sweep.py campaign_300k_v1   # refresh every 5 min
"""

import argparse, json, math, os, random, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import numpy as np
    _USE_NP = True
except ImportError:
    _USE_NP = False


# ── stats helpers ──────────────────────────────────────────────────────────────

def _pct(vals: list, qs=(0, 5, 25, 50, 75, 95, 100)) -> dict:
    nan = float('nan')
    if not vals:
        return {q: nan for q in qs}
    if _USE_NP:
        a = __import__('numpy').array(vals, dtype=float)
        return {q: float(__import__('numpy').percentile(a, q)) for q in qs}
    sv = sorted(float(v) for v in vals)
    n = len(sv)
    result = {}
    for q in qs:
        idx = q / 100 * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        result[q] = sv[lo] + (idx - lo) * (sv[hi] - sv[lo])
    return result


def _mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else float('nan')


# ── formatting helpers ─────────────────────────────────────────────────────────

def _bar(pct: float, width: int = 44) -> str:
    filled = int(round(max(0.0, min(100.0, pct)) / 100 * width))
    return '█' * filled + '░' * (width - filled)


def _hist(vals: list, n_bins: int = 8, bar_width: int = 22,
          lo: float = None, hi: float = None,
          label_fn=None) -> List[str]:
    """ASCII histogram with optional edge-label transform (e.g. 10** for log-scale)."""
    if not vals:
        return ['    (no data)']
    lo = lo if lo is not None else min(vals)
    hi = hi if hi is not None else max(vals)
    if lo >= hi:
        hi = lo + 1e-9
    bw = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in vals:
        b = int((float(v) - lo) / (hi - lo) * n_bins)
        counts[min(b, n_bins - 1)] += 1
    total = len(vals)
    max_c = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        e0 = lo + i * bw
        e1 = lo + (i + 1) * bw
        lbl = (f'{label_fn(e0):>11}–{label_fn(e1):<11}'
               if label_fn else f'{e0:>9.3f}–{e1:<9.3f}')
        bar = '█' * int(c / max_c * bar_width)
        lines.append(f'    {lbl}  {bar:<{bar_width}}  {c:>7,}  ({100*c/total:5.1f}%)')
    return lines


def _fmt_bytes(b: int) -> str:
    for unit, f in [('TiB', 2**40), ('GiB', 2**30), ('MiB', 2**20)]:
        if b >= f:
            return f'{b/f:.2f} {unit}'
    return f'{b} B'


def _fmt_eta(h: float) -> str:
    if math.isnan(h) or h < 0:
        return '—'
    return f'~{h:.0f} h  (~{h/24:.1f} d)'


def _sec(label: str, width: int = 68) -> str:
    pad = max(0, (width - len(label) - 4) // 2)
    return f"\n  {'─'*pad}  {label}  {'─'*(width - pad - len(label) - 4)}"


# ── data loading ───────────────────────────────────────────────────────────────

def _disk_bytes(path: Path) -> int:
    """Fast disk usage via du -sb; falls back to os.walk."""
    try:
        raw = subprocess.check_output(
            ['du', '-sb', str(path)], stderr=subprocess.DEVNULL).decode()
        return int(raw.split()[0])
    except Exception:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += (Path(root) / f).stat().st_size
                except OSError:
                    pass
        return total


def _count_sidecars(task_dir: Path) -> Tuple[int, int]:
    """Count (completed, discarded) by file listing only — no content read."""
    n_done = sum(1 for _ in task_dir.glob('topo_*/iter_*.json'))
    n_fail = 0
    for fp in task_dir.glob('topo_*/_failures.jsonl'):
        try:
            with open(fp) as fh:
                n_fail += sum(1 for ln in fh if ln.strip())
        except OSError:
            pass
    return n_done, n_fail


def _sample_sidecars(task_dir: Path, n: int, rng: random.Random) -> List[dict]:
    """Read up to n randomly chosen sidecar dicts."""
    fps = list(task_dir.glob('topo_*/iter_*.json'))
    chosen = rng.sample(fps, min(n, len(fps)))
    out = []
    for fp in chosen:
        try:
            with open(fp) as fh:
                out.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            pass
    return out


def _mtime_range(task_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    """Return (oldest, newest) mtime using only the first+last files by name.
    O(1) stat calls per task regardless of how many sims are in it."""
    fps = sorted(task_dir.glob('topo_*/iter_*.json'))
    if not fps:
        return None, None
    try:
        return fps[0].stat().st_mtime, fps[-1].stat().st_mtime
    except OSError:
        return None, None


def scan_task(task_dir: Path, max_sample: int, rng: random.Random) -> dict:
    """Collect all stats for one sweep_* task directory."""
    name = task_dir.name
    queue_tag = ('c192' if 'c192' in name else
                 'c48'  if 'c48'  in name else '?')

    # ── try manifest first (fast path) ──────────────────────────────────────
    mp = task_dir / 'manifest.json'
    records: List[dict] = []
    n_done = n_fail = 0
    has_manifest = False

    if mp.exists():
        try:
            mf = json.load(open(mp))
            has_manifest = True
            n_done = mf.get('n_total_runs', 0)
            n_fail = mf.get('n_total_failures_discarded', 0)
            for topo in mf.get('topologies', []):
                records.extend(topo.get('iterations', []))
        except (OSError, json.JSONDecodeError):
            has_manifest = False

    # ── fall back to filesystem scan + sidecar sampling ─────────────────────
    if not has_manifest:
        n_done, n_fail = _count_sidecars(task_dir)
    if not records and n_done > 0:
        records = _sample_sidecars(task_dir, max_sample, rng)

    # ── extract per-sim metrics ──────────────────────────────────────────────
    def _pull(key):
        return [r[key] for r in records if key in r]

    fr    = _pull('mean_FR_Hz')
    trun  = _pull('t_run_s')
    spk   = _pull('n_neuronal_spikes')
    astro = _pull('n_astrocyte_events')
    cp    = _pull('conn_prob')

    t_old, t_new = _mtime_range(task_dir)
    n_topos = sum(1 for d in task_dir.glob('topo_*') if d.is_dir())

    return dict(
        name=name, queue_tag=queue_tag,
        has_manifest=has_manifest,
        n_done=n_done, n_fail=n_fail, n_records=len(records),
        fr=fr, trun=trun, spk=spk, astro=astro, cp=cp,
        n_with_astro=sum(1 for v in astro if v > 0),
        t_old=t_old, t_new=t_new, n_topos=n_topos,
    )


# ── report ─────────────────────────────────────────────────────────────────────

def print_report(root: Path, tasks: List[dict],
                 target: int, as_json: bool) -> None:
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    W = 68

    # ── aggregate ────────────────────────────────────────────────────────────
    total_done = sum(t['n_done'] for t in tasks)
    total_fail = sum(t['n_fail'] for t in tasks)
    total_att  = total_done + total_fail

    all_fr    = [v for t in tasks for v in t['fr']]
    all_trun  = [v for t in tasks for v in t['trun']]
    all_astro = [v for t in tasks for v in t['astro']]
    all_cp    = [v for t in tasks for v in t['cp']]
    n_topos   = sum(t['n_topos'] for t in tasks)

    # ── throughput ───────────────────────────────────────────────────────────
    olds = [t['t_old'] for t in tasks if t['t_old']]
    news = [t['t_new'] for t in tasks if t['t_new']]
    rate = eta = float('nan')
    if olds and news and total_done > 0:
        elapsed_h = (max(news) - min(olds)) / 3600.0
        if elapsed_h > 0.01:
            rate = total_done / elapsed_h
            rem = target - total_done
            if rem > 0 and rate > 0:
                eta = rem / rate

    # ── storage ──────────────────────────────────────────────────────────────
    disk = _disk_bytes(root)
    proj = int(disk / total_done * target) if total_done > 0 else 0

    # ── JSON output ──────────────────────────────────────────────────────────
    if as_json:
        print(json.dumps({
            'timestamp':    now,
            'campaign':     root.name,
            'target':       target,
            'total_done':   total_done,
            'total_fail':   total_fail,
            'pct_done':     round(100.0 * total_done / target, 2) if target else 0,
            'rate_per_hour': None if math.isnan(rate) else round(rate, 1),
            'eta_hours':     None if math.isnan(eta)  else round(eta,  1),
            'disk_bytes':   disk,
            'disk_proj':    proj,
            'n_topology_dirs': n_topos,
            'conn_prob_range': ([min(all_cp), max(all_cp)] if all_cp else None),
            'fr_pct':   _pct(all_fr)   if all_fr   else None,
            'trun_pct': _pct(all_trun) if all_trun else None,
            'tasks': [{'name': t['name'], 'done': t['n_done'],
                       'fail': t['n_fail']} for t in tasks],
        }, indent=2))
        return

    # ── terminal output ───────────────────────────────────────────────────────
    pct      = 100.0 * total_done / target if target else 0.0
    disc_pct = 100.0 * total_fail / total_att if total_att else 0.0

    print()
    print('  ' + '═' * W)
    print(f"  {'SWEEP  STATUS':^{W}}")
    print(f"  {root.name}   ·   {now}   ·   target {target:,}")
    print('  ' + '═' * W)

    # progress
    print()
    print(f"  Progress   {total_done:>10,} / {target:,}   ({pct:.1f} %)")
    print(f"             {_bar(pct)}  {pct:.0f}%")
    print(f"  Discarded  {total_fail:>10,} ({disc_pct:.2f} %)    "
          f"Disk: {_fmt_bytes(disk)}   projected: {_fmt_bytes(proj)}")

    # throughput
    print(_sec('THROUGHPUT'))
    if not math.isnan(rate):
        print(f"  Rate     :  {rate:>9,.0f}  sims / h")
        print(f"  ETA      :  {_fmt_eta(eta)}")
    else:
        print("  Rate     :  not yet available  (need ≥ 2 completed iterations)")

    # per-task table
    print(_sec('PER-TASK BREAKDOWN'))
    print(f"  {'Task':<30}  {'Done':>8}  {'Fail':>5}  "
          f"{'FR̄ (Hz)':>8}  {'t̄ run (s)':>9}  Source")
    print('  ' + '·' * (W - 2))
    for tag in ['c192', 'c48', '?']:
        grp = sorted([t for t in tasks if t['queue_tag'] == tag],
                     key=lambda x: x['name'])
        if not grp:
            continue
        for t in grp:
            mfr  = f"{_mean(t['fr']):.1f}"   if t['fr']   else '—'
            mrun = f"{_mean(t['trun']):.0f}" if t['trun'] else '—'
            src  = ('manifest' if t['has_manifest'] and t['n_records'] > 0
                    else f"~{t['n_records']} spl"  if t['n_records'] > 0
                    else 'file count')
            print(f"  {t['name']:<30}  {t['n_done']:>8,}  {t['n_fail']:>5,}  "
                  f"{mfr:>8}  {mrun:>9}  {src}")
        print('  ' + '·' * (W - 2))

    # simulation quality — firing rate
    print(_sec('SIMULATION QUALITY'))
    if all_fr:
        p = _pct(all_fr)
        print(f"\n  Firing Rate (Hz)   [{len(all_fr):,} sims in sample]")
        print(f"  {'min':>7}  {'p5':>7}  {'p25':>7}  {'p50':>7}  "
              f"{'p75':>7}  {'p95':>7}  {'max':>7}")
        print(f"  {p[0]:>7.2f}  {p[5]:>7.2f}  {p[25]:>7.2f}  {p[50]:>7.2f}  "
              f"{p[75]:>7.2f}  {p[95]:>7.2f}  {p[100]:>7.2f}")
        print()
        # log-scale histogram: bin in log10 space, label in Hz
        log_fr = [math.log10(max(v, 0.01)) for v in all_fr]
        lo_l = math.log10(max(min(all_fr), 0.01) * 0.9)
        hi_l = math.log10(max(all_fr) * 1.1 + 1e-9)
        for ln in _hist(log_fr, n_bins=8, lo=lo_l, hi=hi_l,
                        label_fn=lambda x: f'{10**x:.2f} Hz'):
            print(ln)
        # flag silent (FR=0) and hyperactive sims
        n_silent = sum(1 for v in all_fr if v < 0.1)
        n_hyper  = sum(1 for v in all_fr if v > 100)
        flags = []
        if n_silent: flags.append(f'{n_silent:,} silent (<0.1 Hz)')
        if n_hyper:  flags.append(f'{n_hyper:,} hyperactive (>100 Hz)')
        if flags:
            print(f"  ⚠  {',  '.join(flags)}")
    else:
        print("\n  Firing rate: no data yet")

    # run time + straggler
    if all_trun:
        p = _pct(all_trun)
        strag = p[95] / p[50] if p[50] > 0 else float('nan')
        print(f"\n  Run time (s)   [{len(all_trun):,} sims in sample]")
        print(f"  {'min':>7}  {'p5':>7}  {'p25':>7}  {'p50':>7}  "
              f"{'p75':>7}  {'p95':>7}  {'max':>7}")
        print(f"  {p[0]:>7.1f}  {p[5]:>7.1f}  {p[25]:>7.1f}  {p[50]:>7.1f}  "
              f"{p[75]:>7.1f}  {p[95]:>7.1f}  {p[100]:>7.1f}")
        if not math.isnan(strag):
            flag = ('  ⚠  high — consider raising N_PARAMS_PER_WORKER'
                    if strag > 2.0 else '  ✓  within expected range')
            print(f"  Straggler ratio  p95/p50 = {strag:.2f}×{flag}")

    # astrocyte events
    if all_astro:
        n_act    = sum(1 for v in all_astro if v > 0)
        pct_a    = 100.0 * n_act / len(all_astro)
        mean_act = _mean([v for v in all_astro if v > 0]) if n_act else 0.0
        print(f"\n  Astrocyte events: {n_act:,} / {len(all_astro):,} sims active "
              f"({pct_a:.1f} %)    mean when active: {mean_act:.1f}")

    # parameter space
    print(_sec('PARAMETER SPACE'))
    print(f"  Topology dirs  :  {n_topos:,}")
    if all_cp:
        print(f"  conn_prob      :  [{min(all_cp):.3f} … {max(all_cp):.3f}]  "
              f"mean = {_mean(all_cp):.3f}")
        for ln in _hist(all_cp, n_bins=6):
            print(ln)
    else:
        print("  conn_prob      :  no data yet")

    # storage
    print(_sec('STORAGE'))
    print(f"  Used           :  {_fmt_bytes(disk)}")
    if proj:
        bytes_per = disk / total_done
        print(f"  Per sim (mean) :  {_fmt_bytes(int(bytes_per))}")
        print(f"  Projected @{target//1000}k :  {_fmt_bytes(proj)}")
    print()


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('campaign_root')
    ap.add_argument('--target', type=int, default=300_000,
                    help='total sim target (default 300000)')
    ap.add_argument('--sample', type=int, default=5_000,
                    help='max sidecars to sample per task for quality stats '
                         '(default 5000; raise for more precise distributions)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--json', action='store_true',
                    help='machine-readable JSON output (for scripting / logging)')
    args = ap.parse_args()

    root = Path(args.campaign_root).resolve()
    if not root.is_dir():
        sys.exit(f'campaign root not found: {root}')

    task_dirs = sorted(
        d for d in root.glob('sweep_*')
        if d.is_dir() and d.name != 'campaign_index'
    )
    if not task_dirs:
        sys.exit(f'no sweep_* task dirs under {root}')

    rng = random.Random(args.seed)
    tasks = [scan_task(d, args.sample, rng) for d in task_dirs]
    print_report(root, tasks, args.target, as_json=args.json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
