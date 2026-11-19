#!/usr/bin/env python3
"""
campaign_inspect.py -- a brief behaviour report of a finished or stopped
campaign: the distribution of the per-run mean rate and the shares of four
activity classes, read from the per-run sidecars that
`aggregate_sweep.py --write-index` copies into
<campaign>/campaign_index/summary.csv. No spike train is opened.
specs/SPEC.md Block 14; decision D-004 (2026-11-19).

Run (from the folder that holds this file):
    python3 campaign_inspect.py <campaign folder> [--out DIR] [--no-plots]
    python3 smoke_test_campaign_inspect.py
On davinci, as a job that first refreshes a stale campaign_index/:
    qsub -q cpu -v CAMPAIGN=<campaign folder> submit_campaign_inspect.sh

Per run n (its sidecar, written by the sweep worker): rbar_n = mean_FR_Hz =
K_n / (N_n T_n), with K_n the neuron spikes, N_n the neurons and T_n the
simulated seconds; a_n = frac_active, the share of neurons whose own rate
exceeds 0.1 Hz. Classes, in this order of precedence (InspectConfig):
    silent     a_n <  a_min                         a_min  = 0.1
    saturated  a_n >= a_min and rbar_n >= r_sat     r_sat  = 300 Hz = 0.9/t_ref
    high       a_n >= a_min and r_high < rbar_n < r_sat
    active     a_n >= a_min and rbar_n <= r_high    r_high = 20 Hz
Runs of partial topologies (fewer sidecars + failures than the sweep's
simulations per topology) are left out; a topology without
topology_meta.json holds no run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

CLASS_NAMES = ("silent", "active", "high", "saturated")   # index = class code
SILENT, ACTIVE, HIGH, SATURATED = 0, 1, 2, 3
INDEX_COLUMNS = ("task", "topo_idx", "iter_idx", "mean_FR_Hz", "frac_active",
                 "n_neuronal_spikes")
KEPT_STATES = ("complete", "unknown")


@dataclass(frozen=True)
class InspectConfig:
    """Every threshold and binning of the report (D-004 for the classes)."""
    a_min: float = 0.1            # silent below this share of neurons above 0.1 Hz
    r_high_hz: float = 20.0       # active at or below, high above [Hz]
    r_sat_hz: float = 300.0       # saturated at or above [Hz] (0.9 / t_ref)
    t_ref_s: float = 0.003        # refractory period [s]; 1/t_ref is the rate ceiling
    quantiles: Tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    log_rate_lo: float = -4.5     # histogram of log10(rbar / Hz): range [decades]
    log_rate_hi: float = 2.7
    log_rate_bin: float = 0.05    # bin width [decades]
    frac_bins: int = 50           # histogram of frac_active: equal bins on [0, 1]


# ===========================================================================
# Loading
# ===========================================================================
def _count_lines(path: Path) -> int:
    """Non-empty lines of a text file (one failure record per line)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def scan_tree(campaign: Path) -> Dict:
    """State of every topology of every task, from the folder tree.

    Returns {task folder name: {"k": int or None, "topos": {topo_idx:
    {"state", "n_json", "n_fail"}}}}. k = _resolved_n_workers x
    n_params_per_worker from the task's job_args.json (None when absent);
    state is "complete", "partial", "unstarted" or "unknown" (SPEC Block 14).
    """
    tasks = {}
    for td in sorted(p for p in Path(campaign).glob("sweep_*") if p.is_dir()):
        k = None
        try:
            with open(td / "job_args.json", "r", encoding="utf-8") as fh:
                ja = json.load(fh)
            k = int(ja["_resolved_n_workers"]) * int(ja["n_params_per_worker"])
        except (OSError, ValueError, KeyError, TypeError):
            k = None
        topos = {}
        for tp in sorted(p for p in td.glob("topo_*") if p.is_dir()):
            try:
                ti = int(tp.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            names = os.listdir(tp)
            n_json = sum(1 for x in names if x.startswith("iter_") and x.endswith(".json"))
            n_fail = _count_lines(tp / "_failures.jsonl") if "_failures.jsonl" in names else 0
            if "topology_meta.json" not in names:
                state = "unstarted"
            elif k is None:
                state = "unknown"
            elif n_json + n_fail >= k:
                state = "complete"
            else:
                state = "partial"
            topos[ti] = {"state": state, "n_json": n_json, "n_fail": n_fail}
        tasks[td.name] = {"k": k, "topos": topos}
    return tasks


def load_index(campaign: Path) -> Dict[str, np.ndarray]:
    """The summary.csv columns this block uses; numeric columns as float64
    (an empty cell becomes NaN), task as str."""
    import pandas as pd
    path = Path(campaign) / "campaign_index" / "summary.csv"
    df = pd.read_csv(path, usecols=list(INDEX_COLUMNS), dtype={"task": str},
                     keep_default_na=False, na_values=[""])
    out = {"task": df["task"].astype(str).to_numpy()}
    for col in INDEX_COLUMNS[1:]:
        out[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    return out


# ===========================================================================
# Analysis
# ===========================================================================
def classify(rate_hz: np.ndarray, frac_active: np.ndarray,
             cfg: InspectConfig) -> np.ndarray:
    """Class code of each run (SILENT, ACTIVE, HIGH, SATURATED), int8.

    rate_hz, frac_active: 1-D float arrays of one length, all finite (drop
    missing values first). Precedence silent > saturated > high > active.
    """
    r = np.asarray(rate_hz, dtype=np.float64)
    a = np.asarray(frac_active, dtype=np.float64)
    if r.ndim != 1 or r.shape != a.shape:
        raise ValueError("rate_hz and frac_active must be 1-D arrays of one length")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(a))):
        raise ValueError("classify needs finite values; drop missing values first")
    codes = np.full(r.shape, ACTIVE, dtype=np.int8)
    codes[r > cfg.r_high_hz] = HIGH
    codes[r >= cfg.r_sat_hz] = SATURATED
    codes[a < cfg.a_min] = SILENT
    return codes


def _quantiles(x: np.ndarray, ps) -> Dict[str, float]:
    if x.size == 0:
        return {}
    return {f"{p:g}": float(v) for p, v in zip(ps, np.quantile(x, ps))}


def summarize(index: Dict[str, np.ndarray], tree: Dict, cfg: InspectConfig) -> Dict:
    """Every number of the report (SPEC Block 14, "Statistics")."""
    n_rows = int(index["task"].shape[0])
    state_of, n_json_of = {}, {}
    topo_states = {s: 0 for s in ("complete", "partial", "unstarted", "unknown")}
    n_fail_kept = 0
    for task, rec in tree.items():
        for ti, t in rec["topos"].items():
            state_of[(task, ti)] = t["state"]
            n_json_of[(task, ti)] = t["n_json"]
            topo_states[t["state"]] += 1
            if t["state"] in KEPT_STATES:
                n_fail_kept += t["n_fail"]

    kept = np.zeros(n_rows, dtype=bool)
    rows_by_state: Dict[str, int] = {}
    rows_of: Dict[Tuple[str, int], int] = {}
    topo_col = index["topo_idx"]
    for i in range(n_rows):
        if not np.isfinite(topo_col[i]):
            st = "no topo_idx"
        else:
            key = (index["task"][i], int(topo_col[i]))
            st = state_of.get(key, "not in tree")
            rows_of[key] = rows_of.get(key, 0) + 1
        rows_by_state[st] = rows_by_state.get(st, 0) + 1
        kept[i] = st in KEPT_STATES

    mismatched = [f"{t}/topo_{ti:05d}: {rows_of.get((t, ti), 0)} rows, {n} sidecars"
                  for (t, ti), n in sorted(n_json_of.items())
                  if state_of[(t, ti)] in KEPT_STATES and rows_of.get((t, ti), 0) != n]
    stale = bool(mismatched) or rows_by_state.get("not in tree", 0) > 0

    r_all = index["mean_FR_Hz"][kept]
    a_all = index["frac_active"][kept]
    k_all = index["n_neuronal_spikes"][kept]
    fin = np.isfinite(r_all) & np.isfinite(a_all)
    r, a, k = r_all[fin], a_all[fin], k_all[fin]
    codes = classify(r, a, cfg)
    n_cls = int(codes.size)
    classes = {}
    for code, name in enumerate(CLASS_NAMES):
        n = int(np.sum(codes == code))
        classes[name] = {"n": n, "share": (n / n_cls) if n_cls else None}

    nb = int(round((cfg.log_rate_hi - cfg.log_rate_lo) / cfg.log_rate_bin))
    edges = np.linspace(cfg.log_rate_lo, cfg.log_rate_hi, nb + 1)
    lr = np.log10(r[r > 0])
    lr_counts, _ = np.histogram(lr, bins=edges)
    a_edges = np.linspace(0.0, 1.0, cfg.frac_bins + 1)
    a_counts, _ = np.histogram(a, bins=a_edges)
    ceiling = 1.0 / cfg.t_ref_s

    return {
        "thresholds": asdict(cfg),
        "rate_ceiling_hz": ceiling,
        "tasks": len(tree),
        "tasks_without_k": sorted(t for t, rec in tree.items() if rec["k"] is None),
        "topologies": topo_states,
        "index_rows": n_rows,
        "index_rows_by_topology_state": rows_by_state,
        "runs_kept": int(kept.sum()),
        "runs_missing_values": int((~fin).sum()),
        "runs_classified": n_cls,
        "failures_in_kept_topologies": n_fail_kept,
        "index_stale": stale,
        "index_mismatched_topologies": len(mismatched),
        "index_mismatch_examples": mismatched[:5],
        "classes": classes,
        "no_spike": int(np.sum(k == 0)),
        "no_spike_not_silent": int(np.sum((k == 0) & (codes != SILENT))),
        "silent_above_r_high": int(np.sum((codes == SILENT) & (r > cfg.r_high_hz))),
        "max_rate_hz": float(r.max()) if n_cls else None,
        "runs_above_ceiling": int(np.sum(r > ceiling * (1.0 + 1e-9))),
        "quantiles": {"mean_FR_Hz": _quantiles(r, cfg.quantiles),
                      "frac_active": _quantiles(a, cfg.quantiles)},
        "histograms": {
            "log10_rate": {"edges": edges.tolist(), "counts": lr_counts.tolist(),
                           "n_zero_rate": int(np.sum(r == 0)),
                           "n_below_range": int(np.sum(lr < edges[0])),
                           "n_above_range": int(np.sum(lr > edges[-1]))},
            "frac_active": {"edges": a_edges.tolist(), "counts": a_counts.tolist()},
        },
    }


def format_report(s: Dict) -> str:
    """The report as text (the same numbers as the JSON)."""
    th = s["thresholds"]
    pct = lambda v: "   n/a" if v is None else f"{100.0 * v:6.2f} %"
    out = [
        f"campaign  : {s.get('inputs', {}).get('campaign', '?')}",
        f"index     : {s.get('inputs', {}).get('summary_csv', '?')} "
        f"(modified {s.get('inputs', {}).get('summary_csv_mtime_utc', '?')})",
        f"tasks {s['tasks']} | topologies: " + ", ".join(
            f"{k} {v}" for k, v in s["topologies"].items()),
        f"index rows {s['index_rows']} by topology state: " + ", ".join(
            f"{k} {v}" for k, v in sorted(s["index_rows_by_topology_state"].items())),
        f"runs kept {s['runs_kept']} | with missing values {s['runs_missing_values']} "
        f"| classified {s['runs_classified']} | failures in kept topologies "
        f"{s['failures_in_kept_topologies']}",
    ]
    if s["tasks_without_k"]:
        out.append(f"WARNING: completeness not judged for {len(s['tasks_without_k'])} "
                   f"task(s) without k in job_args.json: {s['tasks_without_k'][:3]}")
    if s["index_stale"]:
        out.append(f"WARNING: stale index: {s['index_mismatched_topologies']} kept "
                   f"topologies differ from their sidecars; rebuild it (REINDEX=1): "
                   f"{s['index_mismatch_examples'][:2]}")
    out.append("")
    out.append(f"classes (a = frac_active, r = mean rate; a_min {th['a_min']:g}, "
               f"r_high {th['r_high_hz']:g} Hz, r_sat {th['r_sat_hz']:g} Hz)")
    rule = {"silent": "a < a_min", "active": "r <= r_high",
            "high": "r_high < r < r_sat", "saturated": "r >= r_sat"}
    for name in CLASS_NAMES:
        c = s["classes"][name]
        out.append(f"  {name:<10} {c['n']:>9}  {pct(c['share'])}   {rule[name]}")
    out.append(f"  of the silent: no spike at all {s['no_spike']}, mean rate above "
               f"r_high {s['silent_above_r_high']}")
    if s["no_spike_not_silent"]:
        out.append(f"WARNING: {s['no_spike_not_silent']} runs with no spike are not "
                   f"silent (inconsistent sidecars)")
    out.append(f"max mean rate {s['max_rate_hz']} Hz | runs above 1/t_ref "
               f"({s['rate_ceiling_hz']:.1f} Hz): {s['runs_above_ceiling']}")
    out.append("")
    for col in ("mean_FR_Hz", "frac_active"):
        q = s["quantiles"][col]
        out.append(f"{col} quantiles: " + ("none" if not q else
                   "  ".join(f"p{float(p) * 100:g} {v:.4g}" for p, v in q.items())))
    return "\n".join(out)


# ===========================================================================
# Entry point: loading -> analysis -> report and figures
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaign", help="campaign folder (holds sweep_* and campaign_index/)")
    ap.add_argument("--out", default=None,
                    help="output folder (default <campaign>/campaign_inspect_<UTC>)")
    ap.add_argument("--no-plots", action="store_true", help="write no figures")
    args = ap.parse_args(argv)

    campaign = Path(args.campaign).resolve()
    csv_path = campaign / "campaign_index" / "summary.csv"
    if not campaign.is_dir():
        print(f"ERROR: campaign folder not found: {campaign}", file=sys.stderr)
        return 2
    if not csv_path.is_file():
        print(f"ERROR: {csv_path} not found; run "
              f"'python aggregate_sweep.py {campaign} --write-index' first", file=sys.stderr)
        return 2
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(args.out) if args.out else campaign / f"campaign_inspect_{stamp}"

    cfg = InspectConfig()
    tree = scan_tree(campaign)
    index = load_index(campaign)
    summary = summarize(index, tree, cfg)
    summary["inputs"] = {
        "campaign": str(campaign),
        "summary_csv": str(csv_path),
        "summary_csv_mtime_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(csv_path.stat().st_mtime)),
        "generated_utc": stamp,
    }
    report = format_report(summary)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "inspect_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(out / "inspect_report.txt", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)
    if not args.no_plots:
        import campaign_inspect_plots as plots
        for p in plots.write_all(summary, out):
            print(f"[figure] {p}")
    print(f"[inspect] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
