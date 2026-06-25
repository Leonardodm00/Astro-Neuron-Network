#!/usr/bin/env python3
"""
aggregate_sweep.py -- monitor a job-array campaign and build an SBI-ready index.

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

BURST METRICS (opt-in, --burst-metrics)
---------------------------------------
With --burst-metrics the index build additionally loads each iter_*.npz spike
train and computes a panel of network-burst metrics (two detectors,
cross-checked) via burst_metrics.py, appends them as extra summary.csv columns,
scores every run against the Mossink et al. 2019 control-DIV28 anchor, and
renders the top-K best-matching runs (full raster + population rate + a
representative-burst zoom) into campaign_index/burst_gallery/ via burst_plots.py.
Without the flag the script behaves exactly as before (no npz spike I/O, no
matplotlib import), so the lightweight progress-monitoring mode is unchanged.

Usage:
    # progress only (unchanged):
    python aggregate_sweep.py campaign_300k_v1 --target 300000

    # build the plain index (unchanged):
    python aggregate_sweep.py campaign_300k_v1 --write-index

    # build the index AND burst metrics + gallery:
    python aggregate_sweep.py campaign_300k_v1 --burst-metrics \
        --score-detector logisi --n-gallery 10 --zoom-window 6.0
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


# ---------------------------------------------------------------------------
# Burst-metrics support (only exercised under --burst-metrics)
# ---------------------------------------------------------------------------
def _load_topo_meta(topo_dir: Path, cache: dict):
    """Return (simtime_s, Nn, Na) for a topo_* dir, cached. (None,None,None) if absent."""
    key = str(topo_dir)
    if key in cache:
        return cache[key]
    meta_fp = topo_dir / "topology_meta.json"
    simtime = nn = na = None
    try:
        with open(meta_fp) as fh:
            m = json.load(fh)
        simtime = float(m.get("simtime_s")) if m.get("simtime_s") is not None else None
        nn = int(m.get("Nn")) if m.get("Nn") is not None else None
        na = int(m.get("Na")) if m.get("Na") is not None else None
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    cache[key] = (simtime, nn, na)
    return cache[key]


def _burst_row_from_npz(npz_path: Path, simtime, nn, cfg):
    """Load one iter_*.npz and return (scalar_metrics_dict, full_row_with_intervals).

    full_row carries the detected interval lists under _pr_intervals/_li_intervals
    so the gallery renderer can reuse them without re-detecting. Returns
    (None, None) on any failure (caller leaves burst columns blank).
    """
    import burst_metrics as bm  # local import: keep numpy-only dep lazy
    try:
        with np.load(npz_path) as z:
            spk_t = np.asarray(z["spk_N_t"], dtype=np.float64)
            spk_i = np.asarray(z["spk_N_i"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as e:
        print(f"  ! burst: cannot read {npz_path.name}: {e}", file=sys.stderr)
        return None, None

    # Fallbacks if topology_meta.json was missing.
    t_rec = simtime if (simtime and simtime > 0) else (
        float(spk_t.max()) if spk_t.size else 0.0)
    n_neurons = nn if (nn and nn > 0) else (
        int(spk_i.max()) + 1 if spk_i.size else 0)
    if t_rec <= 0 or n_neurons <= 0:
        return None, None

    full = bm.compute_all(spk_t, spk_i, n_neurons, t_rec, cfg)
    return bm.scalar_row(full), full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign_root")
    ap.add_argument("--target", type=int, default=300_000)
    ap.add_argument("--write-index", action="store_true",
                    help="emit theta_all.npy / npz_paths.txt / summary.csv")
    # --- burst-metrics options (opt-in) ---
    ap.add_argument("--burst-metrics", action="store_true",
                    help="compute network-burst metrics from each npz spike train, "
                         "append to summary.csv, and render the top-K gallery "
                         "(implies --write-index)")
    ap.add_argument("--score-detector", choices=("logisi", "poprate"),
                    default="logisi",
                    help="detector whose metrics are scored against the literature "
                         "anchor (logisi == Mossink >80%% participation; default)")
    ap.add_argument("--lit-scale", choices=("campaign", "literature"),
                    default="campaign",
                    help="z-score scale for the literature distance: campaign "
                         "robust spread (default) or the literature SDs")
    ap.add_argument("--n-gallery", type=int, default=10,
                    help="number of best-matching runs to render (default 10)")
    ap.add_argument("--zoom-window", type=float, default=6.0,
                    help="seconds for the representative-burst zoom panel")
    args = ap.parse_args()

    if args.burst_metrics:
        args.write_index = True   # burst columns live in summary.csv

    root = Path(args.campaign_root).resolve()
    if not root.is_dir():
        sys.exit(f"campaign root not found: {root}")

    # matches sweep_task* (single-queue) and sweep_c48_task* / sweep_c192_task*
    # (mixed-queue), but not the campaign_index/ output dir
    task_dirs = sorted(d for d in root.glob("sweep_*") if d.is_dir())
    if not task_dirs:
        sys.exit(f"no sweep_* task dirs under {root}")

    cfg = None
    if args.burst_metrics:
        import burst_metrics as bm
        cfg = bm.BurstConfig(scoring_detector=args.score_detector)

    thetas, npz_paths, rows = [], [], []
    meta_cache: dict = {}
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
                    row = {
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
                    }

                    if args.burst_metrics:
                        simtime, nn, _na = _load_topo_meta(sc.parent, meta_cache)
                        bmetrics, _full = _burst_row_from_npz(npz, simtime, nn, cfg)
                        if bmetrics is not None:
                            # fill the previously-empty whole-train columns from
                            # the spike-derived values (sidecar had None)
                            row["across_cell_rate_cv"] = bmetrics.get(
                                "across_cell_rate_cv", row["across_cell_rate_cv"])
                            row["frac_active"] = bmetrics.get(
                                "frac_active", row["frac_active"])
                            row["mean_isi_cv"] = bmetrics.get(
                                "cv_isi_mean", row["mean_isi_cv"])
                            # append the full burst panel
                            for k, v in bmetrics.items():
                                if k in ("across_cell_rate_cv", "frac_active"):
                                    continue
                                row[k] = v

                    rows.append(row)
                except (OSError, KeyError, json.JSONDecodeError) as e:
                    print(f"  ! skipped {sc.name}: {e}", file=sys.stderr)

    print("-" * 50)
    pct = 100.0 * tot_done / args.target if args.target else 0.0
    drate = 100.0 * tot_fail / (tot_done + tot_fail) if (tot_done + tot_fail) else 0.0
    print(f"{'TOTAL':>18} {tot_done:>8} {tot_fail:>10} {tot_bytes/2**30:>10.2f}")
    print(f"\nprogress: {tot_done:,} / {args.target:,}  ({pct:.1f}%)   "
          f"discard rate: {drate:.2f}%   total disk: {tot_bytes/2**40:.3f} TiB")

    # ---- literature scoring + gallery (only with --burst-metrics) --------
    if args.burst_metrics and rows:
        import burst_metrics as bm
        dist = bm.literature_distance(rows, cfg, scale=args.lit_scale)
        for r, dv in zip(rows, dist):
            r["lit_distance"] = float(dv)
        order = np.argsort(dist)
        finite = [int(i) for i in order if np.isfinite(dist[i])]
        top = finite[:max(args.n_gallery, 0)]
        print(f"\n[burst] scored {len(rows):,} runs against Mossink control DIV28 "
              f"(detector={args.score_detector}, scale={args.lit_scale}); "
              f"{len(finite):,} have finite distance.")

        if top:
            import burst_plots as bp
            gal = root / "campaign_index" / "burst_gallery"
            gal.mkdir(parents=True, exist_ok=True)
            n_written = 0
            for rank, i in enumerate(top, start=1):
                r = rows[i]
                npz_path = Path(r["npz_path"])
                topo_dir = npz_path.parent
                simtime, nn, _na = _load_topo_meta(topo_dir, meta_cache)
                bmetrics, full = _burst_row_from_npz(npz_path, simtime, nn, cfg)
                if full is None:
                    continue
                with np.load(npz_path) as z:
                    spk_t = np.asarray(z["spk_N_t"], dtype=np.float64)
                    spk_i = np.asarray(z["spk_N_i"], dtype=np.int64)
                t_rec = simtime if (simtime and simtime > 0) else float(spk_t.max())
                n_neurons = nn if (nn and nn > 0) else int(spk_i.max()) + 1
                title = (f"rank {rank:02d} | {r['task']} "
                         f"topo {r['topo_idx']} iter {r['iter_idx']} | "
                         f"dist={r['lit_distance']:.3f}")
                out_fp = gal / (f"rank{rank:02d}_{r['task']}_"
                                f"topo{r['topo_idx']}_iter{r['iter_idx']}.png")
                bp.plot_run(
                    spk_t, spk_i, n_neurons, t_rec,
                    full["_pr_intervals"], full["_li_intervals"],
                    {**r, **bmetrics}, str(out_fp), cfg=cfg,
                    zoom_window_s=args.zoom_window, title=title)
                n_written += 1
            print(f"[burst] wrote {n_written} gallery figures -> {gal}/")

    # ---- write the index -------------------------------------------------
    if args.write_index and thetas:
        idx = root / "campaign_index"
        idx.mkdir(exist_ok=True)
        np.save(idx / "theta_all.npy", np.asarray(thetas, dtype=np.float64))
        (idx / "npz_paths.txt").write_text("\n".join(npz_paths) + "\n")
        # union of keys (in first-seen order) so burst columns appear uniformly
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(idx / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[index] wrote {len(thetas):,} rows -> {idx}/"
              f"  (theta_all.npy {np.asarray(thetas).shape}, npz_paths.txt, "
              f"summary.csv [{len(fieldnames)} cols])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
