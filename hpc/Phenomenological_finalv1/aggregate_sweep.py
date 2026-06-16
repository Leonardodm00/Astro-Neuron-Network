#!/usr/bin/env python3
"""
aggregate_sweep.py — monitor a job-array campaign and build an SBI-ready index.

Walks  campaign_<TAG>/sweep_task*/  and:
  * reports per-task and cumulative completed / discarded sim counts + disk use,
  * emits a single SBI-ready index under  campaign_<TAG>/campaign_index/ :
        theta_all.npy   (N, 35)  inference-coordinate labels (the SBI prior space).
                                 NB: DeltaT/VT/gL (idx 12/13/30) are FROZEN
                                 constant columns; train the density estimator on
                                 the manifest's _active_indices (22 free axes under
                                 neuron_synapse), NOT on the full 35.
        npz_paths.txt   (N,)     absolute path to each iter_*.npz, row-aligned
        summary.csv     (N,)     per-sim scalars (rates, spikes, run time, path)

Run it any time mid-campaign to track progress toward the 300k target; run it
once at the end to produce the index your misspecification screen + NPE load.

Usage:
    python aggregate_sweep.py campaign_300k_v1 --target 300000 [--write-index]
"""
import argparse, csv, json, os, sys
from pathlib import Path

import numpy as np


def _dir_bytes(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def _count_failures(task_dir: Path) -> int:
    n = 0
    for fp in task_dir.glob("topo_*/_failures.jsonl"):
        try:
            with open(fp) as fh:
                n += sum(1 for line in fh if line.strip())
        except OSError:
            pass
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign_root")
    ap.add_argument("--target", type=int, default=300_000)
    ap.add_argument("--write-index", action="store_true",
                    help="emit theta_all.npy / npz_paths.txt / summary.csv")
    args = ap.parse_args()

    root = Path(args.campaign_root).resolve()
    if not root.is_dir():
        sys.exit(f"campaign root not found: {root}")

    # matches sweep_task* (single-queue) and sweep_c48_task* / sweep_c192_task*
    # (mixed-queue), but not the campaign_index/ output dir
    task_dirs = sorted(d for d in root.glob("sweep_*") if d.is_dir())
    if not task_dirs:
        sys.exit(f"no sweep_* task dirs under {root}")

    thetas, npz_paths, rows = [], [], []
    tot_done = tot_fail = tot_bytes = 0

    print(f"{'task':>18} {'done':>8} {'discarded':>10} {'disk(GiB)':>10}")
    print("-" * 50)
    for td in task_dirs:
        sidecars = sorted(td.glob("topo_*/iter_*.json"))
        done = len(sidecars)
        fail = _count_failures(td)
        nbytes = _dir_bytes(td)
        tot_done += done; tot_fail += fail; tot_bytes += nbytes
        print(f"{td.name:>18} {done:>8} {fail:>10} {nbytes/2**30:>10.2f}")

        if args.write_index:
            for sc in sidecars:
                try:
                    with open(sc) as fh:
                        d = json.load(fh)
                    npz = (td / d["npz_relpath"]) if "npz_relpath" in d \
                          else sc.with_suffix(".npz")
                    thetas.append(d["theta"])
                    npz_paths.append(str(npz.resolve()))
                    rows.append({
                        "task": td.name,
                        "topo_idx": d.get("topo_idx"),
                        "iter_idx": d.get("iter_idx"),
                        "conn_prob": d.get("conn_prob"),
                        "mean_FR_Hz": d.get("mean_FR_Hz"),
                        "across_cell_rate_cv": d.get("across_cell_rate_cv"),
                        "frac_active": d.get("frac_active"),
                        "mean_isi_cv": d.get("mean_isi_cv"),
                        "n_neuronal_spikes": d.get("n_neuronal_spikes"),
                        "n_astrocyte_events": d.get("n_astrocyte_events"),
                        "t_run_s": d.get("t_run_s"),
                        "seed_run": d.get("seed_run"),
                        "npz_path": npz_paths[-1],
                    })
                except (OSError, KeyError, json.JSONDecodeError) as e:
                    print(f"  ! skipped {sc.name}: {e}", file=sys.stderr)

    print("-" * 50)
    pct = 100.0 * tot_done / args.target if args.target else 0.0
    drate = 100.0 * tot_fail / (tot_done + tot_fail) if (tot_done + tot_fail) else 0.0
    print(f"{'TOTAL':>18} {tot_done:>8} {tot_fail:>10} {tot_bytes/2**30:>10.2f}")
    print(f"\nprogress: {tot_done:,} / {args.target:,}  ({pct:.1f}%)   "
          f"discard rate: {drate:.2f}%   total disk: {tot_bytes/2**40:.3f} TiB")

    if args.write_index and thetas:
        idx = root / "campaign_index"
        idx.mkdir(exist_ok=True)
        np.save(idx / "theta_all.npy", np.asarray(thetas, dtype=np.float64))
        (idx / "npz_paths.txt").write_text("\n".join(npz_paths) + "\n")
        with open(idx / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n[index] wrote {len(thetas):,} rows -> {idx}/"
              f"  (theta_all.npy {np.asarray(thetas).shape}, npz_paths.txt, summary.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
