#!/usr/bin/env python3
"""
run_burst_analysis.py -- sophisticated network-burst analysis for one campaign.

PIPELINE (hybrid: cheap screen over everything, rich metrics on the best few)
----------------------------------------------------------------------------
  1. RESOLVE   the campaign directory (an explicit path, else the lone
               campaign_* directory sitting beside this script / in $CWD).
  2. SCREEN    every simulation with the scalable AutoMIND-style walker
               (find_network_bursts.run_campaign) to produce / reuse
               <campaign>/burst_analysis/burst_features.csv. One cheap pass
               flags which sims burst at all.
  3. SELECT    the bursting sims as the candidate pool; if that pool is larger
               than --candidate-pool, coarse-rank by a Mossink-mapped distance
               on the screen's own features and keep the closest.
  4. FINE      run the rich two-detector metric suite
               (burst_metrics.compute_all: population-rate AND log-ISI /
               participation) on each candidate, in parallel.
  5. RANK      by burst_metrics.literature_distance to the Mossink 2019 DIV28
               control anchor (campaign-scaled z-distance); keep the top-K.
  6. RENDER    poster figures (burst_poster_plots) for the top-K: the full
               8-figure matrix for the --full-top best exemplars, the hero
               composition for the rest, in every requested palette, plus a
               top-K contact sheet. Vector PDF + high-dpi PNG at A0 panel size.
  7. SUMMARISE write <campaign>/burst_poster/burst_poster_summary.csv and a
               short README.md.

The three stages are decoupled: detection lives in burst_metrics /
find_network_bursts, sizing+color in burst_palettes, drawing in
burst_poster_plots. This script is only orchestration + I/O.

USAGE
  python run_burst_analysis.py [CAMPAIGN_DIR] [options]
  python run_burst_analysis.py --smoke-test            # self-check, no real data

Run `python run_burst_analysis.py -h` for all options.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import burst_metrics as bm
import find_network_bursts as fnb
import burst_palettes as bp
import burst_poster_plots as bpp


# ===========================================================================
# Options (decoupled from argparse so the smoke test can drive the pipeline)
# ===========================================================================
@dataclass
class Options:
    campaign: Path
    out_dir: Path
    topk: int = 6
    full_top: int = 1
    palettes: Sequence[str] = field(
        default_factory=lambda: list(bp.DEFAULT_PALETTE_ORDER))
    workers: int = 1
    rescreen: bool = False
    candidate_pool: int = 300
    ifr_sigma_s: float = 0.05
    panel_mm: float = 380.0
    dpi: int = 300
    max_sims: Optional[int] = None
    formats: Sequence[str] = field(default_factory=lambda: ["pdf", "png"])
    fallback_Nn: int = 100
    fallback_T: float = 100.0
    verbose: bool = True


# ===========================================================================
# 1. campaign resolution
# ===========================================================================
def resolve_campaign(arg: Optional[str], script_dir: Path) -> Path:
    """Explicit path wins; otherwise find the single campaign_* directory in the
    script's directory or the current working directory."""
    if arg:
        p = Path(arg).expanduser().resolve()
        if not p.is_dir():
            raise SystemExit(f"campaign path is not a directory: {p}")
        return p
    seen: Dict[str, Path] = {}
    for base in (Path.cwd(), script_dir):
        for c in sorted(base.glob("campaign_*")):
            if c.is_dir():
                seen[str(c.resolve())] = c.resolve()
    cands = sorted(seen.values())
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise SystemExit(
            "no campaign_* directory found beside this script or in the current "
            "directory; pass the campaign path explicitly")
    raise SystemExit("multiple campaign_* directories found "
                     f"({[c.name for c in cands]}); pass one explicitly")


# ===========================================================================
# 2. screen (reuse existing CSV or run the AutoMIND walker)
# ===========================================================================
def ensure_screen(opt: Options) -> Path:
    csv_path = opt.campaign / "burst_analysis" / "burst_features.csv"
    if csv_path.is_file() and not opt.rescreen:
        if opt.verbose:
            print(f"[screen] reusing existing {csv_path}")
        return csv_path
    if opt.verbose:
        print(f"[screen] scanning campaign with AutoMIND walker "
              f"(workers={opt.workers}) ...")
    cfg = fnb.BurstConfig()
    fnb.run_campaign(opt.campaign, cfg, workers=opt.workers,
                     max_sims=opt.max_sims, write_index=False,
                     fallback_Nn=opt.fallback_Nn, fallback_T=opt.fallback_T,
                     verbose=opt.verbose)
    if not csv_path.is_file():
        raise SystemExit(
            f"screen produced no {csv_path} (no iter_*.npz under the campaign?)")
    return csv_path


# ===========================================================================
# 3. read screen + select candidate pool
# ===========================================================================
def _to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _is_true(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "1.0", "yes", "y")


def read_features(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as fh:
        return list(csv.DictReader(fh))


# Mossink metric key -> (AutoMIND CSV column, transform) for the COARSE screen
# distance only. frac_spikes_outside_burst has no cheap screen analogue, so it
# is simply omitted from the coarse distance (the fine literature_distance uses
# the proper value). Documented abuse: AutoMIND's IBI is peak-to-peak whereas
# Mossink's nibi is the silent gap; close enough to PRE-rank a candidate pool.
_COARSE_MAP = {
    "burst_rate_per_min": ("burst_rate_hz", lambda v: v * 60.0),
    "burst_dur_mean_s":   ("burst_width_mean_s", lambda v: v),
    "nibi_mean_s":        ("ibi_mean_s", lambda v: v),
    "nibi_cv":            ("ibi_cv", lambda v: v),
    "mean_FR_Hz":         ("mean_FR_Hz", lambda v: v),
}


def coarse_distance(row: Dict[str, str]) -> float:
    """Cheap RMS z-distance to the Mossink anchor from the screen's features."""
    zs = []
    for mk, (col, fn) in _COARSE_MAP.items():
        m, s = bm.MOSSINK_CONTROL_DIV28[mk]
        v = _to_float(row.get(col))
        if math.isnan(v):
            continue
        s = s or 1.0
        zs.append(((fn(v) - m) / s) ** 2)
    return math.sqrt(sum(zs) / len(zs)) if zs else float("inf")


def select_candidates(rows: List[Dict[str, str]], opt: Options
                      ) -> List[Dict[str, str]]:
    bursting = [r for r in rows if _is_true(r.get("is_bursting"))]
    if opt.verbose:
        print(f"[select] {len(bursting)} bursting / {len(rows)} sims")
    if not bursting:
        return []
    if len(bursting) > opt.candidate_pool:
        bursting.sort(key=coarse_distance)
        bursting = bursting[:opt.candidate_pool]
        if opt.verbose:
            print(f"[select] coarse-ranked down to {len(bursting)} candidates "
                  f"(pool cap {opt.candidate_pool})")
    return bursting


# ===========================================================================
# 4. fine two-detector metrics (parallel-friendly, module-level worker)
# ===========================================================================
def _load_spikes(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path) as d:
        spk_t = np.asarray(d["spk_N_t"], dtype=np.float64)
        spk_i = np.asarray(d["spk_N_i"], dtype=np.int64)
    return spk_t, spk_i


def _load_astro(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Astrocyte Ca2+ events from one npz (empty arrays if absent)."""
    with np.load(npz_path) as d:
        if "spk_A_t" in d and "spk_A_i" in d:
            at = np.asarray(d["spk_A_t"], dtype=np.float64)
            ai = np.asarray(d["spk_A_i"], dtype=np.int64)
        else:
            at, ai = np.zeros(0), np.zeros(0, dtype=np.int64)
    return at, ai


def _resolve_n_astro(npz_path: str, spk_A_i: np.ndarray) -> int:
    """Na from the topo's topology_meta.json (authoritative), else max index+1."""
    meta = Path(npz_path).parent / "topology_meta.json"
    if meta.is_file():
        try:
            na = json.loads(meta.read_text()).get("Na")
            if na is not None:
                return int(na)
        except Exception:
            pass
    return (int(spk_A_i.max()) + 1) if spk_A_i.size else 0


def _fine_worker(item: Tuple[str, int, float, float]) -> Dict[str, float]:
    """Compute the full scalar metric row for one candidate sim.

    item = (npz_path, Nn, simtime_s, mean_FR_Hz_from_screen). mean_FR_Hz is
    detector-independent (it comes from the screen sidecar); inject it so the
    literature distance and the metrics box are complete. Returns a scalar dict
    (intervals are recomputed later only for the top-K, to keep this picklable
    and memory-light)."""
    npz_path, Nn, T, mean_fr = item
    try:
        spk_t, spk_i = _load_spikes(npz_path)
        cfg = bm.BurstConfig()
        row = bm.compute_all(spk_t, spk_i, int(Nn), float(T), cfg)
        sc = bm.scalar_row(row)
        if math.isnan(_to_float(mean_fr)):
            denom = max(int(Nn), 1) * float(T)
            sc["mean_FR_Hz"] = (float(spk_t.size) / denom) if denom > 0 else float("nan")
        else:
            sc["mean_FR_Hz"] = float(mean_fr)
        sc["npz_path"] = str(npz_path)
        sc["Nn"] = int(Nn)
        sc["simtime_s"] = float(T)
        return sc
    except Exception as e:                         # never let one bad file abort
        return {"_error": f"{type(e).__name__}: {e}", "npz_path": str(npz_path)}


def fine_metrics(candidates: List[Dict[str, str]], opt: Options
                 ) -> List[Dict[str, float]]:
    items = [(r["npz_path"], int(_to_float(r.get("Nn")) or opt.fallback_Nn),
              _to_float(r.get("simtime_s")) or opt.fallback_T,
              _to_float(r.get("mean_FR_Hz"))) for r in candidates]
    out: List[Dict[str, float]] = []
    if opt.workers and opt.workers > 1 and len(items) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=opt.workers) as ex:
            for sc in ex.map(_fine_worker, items, chunksize=1):
                out.append(sc)
    else:
        out = [_fine_worker(it) for it in items]
    errs = [r for r in out if "_error" in r]
    if errs and opt.verbose:
        print(f"[fine] {len(errs)} candidate(s) failed to load; skipping")
        for r in errs[:5]:
            print(f"  ! {r['_error']}  ({r['npz_path']})")
    return [r for r in out if "_error" not in r]


# ===========================================================================
# 5. rank by literature distance
# ===========================================================================
def rank_by_literature(fine_rows: List[Dict[str, float]], cfg: bm.BurstConfig
                       ) -> Tuple[List[Dict[str, float]], np.ndarray]:
    if not fine_rows:
        return [], np.zeros(0)
    dist = bm.literature_distance(fine_rows, cfg)          # campaign-scaled z
    order = np.argsort(dist, kind="stable")
    ranked = [fine_rows[i] for i in order]
    for rank, i in enumerate(order):
        ranked[rank]["literature_distance"] = float(dist[i])
    return ranked, dist[order]


# ===========================================================================
# 6. render
# ===========================================================================
def build_rundata(fine_row: Dict[str, float], opt: Options, cfg: bm.BurstConfig,
                  title: str) -> bpp.RunData:
    """Reload spikes for one exemplar and re-detect (with intervals) for plots."""
    npz_path = fine_row["npz_path"]
    Nn = int(fine_row["Nn"])
    T = float(fine_row["simtime_s"])
    spk_t, spk_i = _load_spikes(npz_path)
    spk_A_t, spk_A_i = _load_astro(npz_path)
    n_astro = _resolve_n_astro(npz_path, spk_A_i)
    row = bm.compute_all(spk_t, spk_i, Nn, T, cfg)
    metrics = bm.scalar_row(row)
    metrics["mean_FR_Hz"] = fine_row.get("mean_FR_Hz", float("nan"))
    return bpp.RunData(
        spk_t=spk_t, spk_i=spk_i, n_neurons=Nn, t_rec=T,
        pr_intervals=row["_pr_intervals"], li_intervals=row["_li_intervals"],
        metrics=metrics, cfg=cfg, ifr_display_sigma_s=opt.ifr_sigma_s,
        title=title, spk_A_t=spk_A_t, spk_A_i=spk_A_i, n_astro=n_astro)


def render_exemplars(ranked: List[Dict[str, float]], opt: Options,
                     cfg: bm.BurstConfig) -> Tuple[List[str], List[bpp.RunData]]:
    os.makedirs(opt.out_dir, exist_ok=True)
    written: List[str] = []
    top = ranked[:opt.topk]
    rundata: List[bpp.RunData] = []

    for rank, fr in enumerate(top, start=1):
        d = fr.get("literature_distance", float("nan"))
        title = "Exemplar #%d  (d=%.2f to Mossink DIV28)" % (rank, d)
        D = build_rundata(fr, opt, cfg, title)
        rundata.append(D)
        which = (bpp.ALL_FIGURES if rank <= opt.full_top else bpp.HERO_ONLY)
        # add the dedicated astrocyte figure whenever the exemplar has astro data
        if D.has_astro:
            which = tuple(which) + bpp.ASTRO_FIGURES
        for pname in opt.palettes:
            style = bp.make_style(pname, panel_mm=opt.panel_mm, dpi=opt.dpi)
            paths = bpp.render_run(D, style, str(opt.out_dir),
                                   "exemplar%02d" % rank, which=which,
                                   formats=opt.formats)
            written += paths
        if opt.verbose:
            print(f"[render] exemplar #{rank}: {Path(fr['npz_path']).name} "
                  f"d={d:.2f} ({'full set' if rank <= opt.full_top else 'hero'})")

    # contact sheet over the top-K, in each palette
    if rundata:
        for i, D in enumerate(rundata, start=1):
            D.title = "#%d  d=%.2f" % (i, ranked[i - 1].get(
                "literature_distance", float("nan")))
        for pname in opt.palettes:
            style = bp.make_style(pname, panel_mm=opt.panel_mm, dpi=opt.dpi)
            written += bpp.render_contact_sheet(rundata, style, str(opt.out_dir),
                                                formats=opt.formats)
    return written, rundata


# ===========================================================================
# 7. summary
# ===========================================================================
def write_summary(ranked: List[Dict[str, float]], opt: Options,
                  cfg: bm.BurstConfig, csv_path: Path) -> Path:
    os.makedirs(opt.out_dir, exist_ok=True)
    keymap = bm.scoring_keys(cfg)
    mossink_keys = list(bm.MOSSINK_CONTROL_DIV28.keys())
    cols = (["rank", "literature_distance", "npz_path", "Nn", "simtime_s"]
            + mossink_keys
            + ["li_n_bursts", "pr_n_bursts", "cross_check_jaccard"])
    out_csv = opt.out_dir / "burst_poster_summary.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rank, fr in enumerate(ranked[:opt.topk], start=1):
            rec = {"rank": rank,
                   "literature_distance": fr.get("literature_distance"),
                   "npz_path": fr.get("npz_path"),
                   "Nn": fr.get("Nn"), "simtime_s": fr.get("simtime_s")}
            for mk in mossink_keys:
                rec[mk] = fr.get(keymap[mk])
            rec["li_n_bursts"] = fr.get("li_n_bursts")
            rec["pr_n_bursts"] = fr.get("pr_n_bursts")
            rec["cross_check_jaccard"] = fr.get("cross_check_jaccard")
            w.writerow(rec)

    readme = opt.out_dir / "README.md"
    pal_list = ", ".join(opt.palettes)
    fig_list = ", ".join(bpp.ALL_FIGURES)
    readme.write_text(
        "# Burst poster figures\n\n"
        "Generated by `run_burst_analysis.py` from\n"
        f"`{opt.campaign}`\n"
        f"(screen: `{csv_path}`).\n\n"
        f"- Exemplars: top {opt.topk} sims closest to the Mossink 2019 DIV28 "
        "control by campaign-scaled literature distance "
        f"(scoring detector: `{cfg.scoring_detector}`).\n"
        f"- Full 8-figure matrix for the top {opt.full_top}; hero composition "
        "for the rest.\n"
        f"- Palettes: {pal_list}.\n"
        f"- Figure types: {fig_list}.\n"
        f"- Panel width {opt.panel_mm:.0f} mm, {opt.dpi} dpi, "
        f"formats: {', '.join(opt.formats)}.\n\n"
        "File naming: `exemplar<NN>__<palette>__<figtype>.<fmt>`, plus "
        "`contact_sheet__<palette>.<fmt>` and `burst_poster_summary.csv`.\n",
        encoding="ascii")
    return out_csv


# ===========================================================================
# pipeline
# ===========================================================================
def run_pipeline(opt: Options) -> Dict[str, object]:
    t0 = time.time()
    cfg = bm.BurstConfig()
    csv_path = ensure_screen(opt)
    rows = read_features(csv_path)
    candidates = select_candidates(rows, opt)
    if not candidates:
        print("[done] no bursting sims found; nothing to render.")
        return {"n_candidates": 0, "n_rendered": 0, "summary_csv": None,
                "ranked": []}
    fine_rows = fine_metrics(candidates, opt)
    ranked, dists = rank_by_literature(fine_rows, cfg)
    written, rundata = render_exemplars(ranked, opt, cfg)
    summary = write_summary(ranked, opt, cfg, csv_path)
    if opt.verbose:
        print(f"[done] {len(ranked)} ranked, "
              f"top-{min(opt.topk, len(ranked))} rendered "
              f"({len(written)} files) in {time.time() - t0:.1f}s")
        print(f"[done] outputs -> {opt.out_dir}")
    return {"n_candidates": len(candidates), "n_rendered": len(written),
            "summary_csv": summary, "ranked": ranked, "files": written,
            "out_dir": opt.out_dir}


# ===========================================================================
# CLI
# ===========================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sophisticated network-burst analysis + poster figures "
                    "for one campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("campaign", nargs="?", default=None,
                   help="campaign directory (default: the single campaign_* "
                        "beside this script or in the current directory)")
    p.add_argument("--out", default=None,
                   help="output directory (default: <campaign>/burst_poster)")
    p.add_argument("--topk", type=int, default=6,
                   help="number of exemplars to render")
    p.add_argument("--full-top", type=int, default=1,
                   help="render the full figure matrix for this many top "
                        "exemplars (hero only for the rest)")
    p.add_argument("--palettes", default=",".join(bp.DEFAULT_PALETTE_ORDER),
                   help="comma-separated palette names "
                        f"(choices: {','.join(bp.PALETTES)})")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel workers for screen + fine metrics")
    p.add_argument("--rescreen", action="store_true",
                   help="re-run the AutoMIND screen even if burst_features.csv "
                        "already exists")
    p.add_argument("--candidate-pool", type=int, default=300,
                   help="max bursting sims to run the fine metrics on "
                        "(coarse-ranked if exceeded)")
    p.add_argument("--ifr-sigma", type=float, default=0.05,
                   help="Gaussian sigma [s] for the DISPLAY IFR envelope")
    p.add_argument("--panel-mm", type=float, default=380.0,
                   help="figure panel width on the poster [mm]")
    p.add_argument("--dpi", type=int, default=300, help="PNG raster dpi")
    p.add_argument("--max-sims", type=int, default=None,
                   help="cap sims scanned in the screen (quick look)")
    p.add_argument("--formats", default="pdf,png",
                   help="comma-separated output formats")
    p.add_argument("--quiet", action="store_true", help="less logging")
    p.add_argument("--smoke-test", action="store_true",
                   help="run a self-contained synthetic-campaign self-check")
    return p


def options_from_args(args: argparse.Namespace, script_dir: Path) -> Options:
    campaign = resolve_campaign(args.campaign, script_dir)
    out_dir = Path(args.out).expanduser().resolve() if args.out \
        else (campaign / "burst_poster")
    palettes = [s.strip() for s in args.palettes.split(",") if s.strip()]
    for pn in palettes:
        if pn not in bp.PALETTES:
            raise SystemExit(f"unknown palette {pn!r}; choose from "
                             f"{sorted(bp.PALETTES)}")
    formats = [s.strip() for s in args.formats.split(",") if s.strip()]
    return Options(
        campaign=campaign, out_dir=out_dir, topk=args.topk,
        full_top=args.full_top, palettes=palettes, workers=args.workers,
        rescreen=args.rescreen, candidate_pool=args.candidate_pool,
        ifr_sigma_s=args.ifr_sigma, panel_mm=args.panel_mm, dpi=args.dpi,
        max_sims=args.max_sims, formats=formats, verbose=not args.quiet)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    if args.smoke_test:
        return _smoke_test()
    opt = options_from_args(args, script_dir)
    if opt.verbose:
        print(f"[campaign] {opt.campaign}")
    run_pipeline(opt)
    return 0


# ===========================================================================
# Smoke test  (python run_burst_analysis.py --smoke-test)
# ===========================================================================
def _make_synthetic_campaign(root: Path) -> Path:
    """Write a tiny but realistic campaign tree under root and return its path.

    Crucially the bursting sims emit per-neuron INTRA-CELLULAR bursts so BOTH
    the screen and the strict log-ISI scoring detector fire (single-spike
    volleys would screen as bursting yet leave the scoring detector -- hence the
    poster figures -- empty)."""
    rng = np.random.default_rng(11)
    campaign = root / "campaign_SMOKE"
    topo = campaign / "sweep_node0_task0000" / "topo_00000"
    topo.mkdir(parents=True)
    Nn, T = 80, 60.0
    Na = 40
    (topo / "topology_meta.json").write_text(json.dumps(
        {"Nn": Nn, "Na": Na, "simtime_s": T, "mode": "Full", "topo_idx": 0}))

    def astro_events(seed, burst_times):
        r = np.random.default_rng(seed + 500)
        at, ai = [], []
        for bt in burst_times:                       # astro lag the volleys
            resp = r.choice(Na, size=int(0.80 * Na), replace=False)
            for aid in resp:
                at.append(bt + r.uniform(0.30, 0.70)); ai.append(int(aid))
        nb = int(0.3 * Na)
        at.extend(r.uniform(0, T, nb)); ai.extend(r.integers(0, Na, nb))
        return np.asarray(at, np.float32), np.asarray(ai, np.int32)

    def mini_burst_sim(seed, burst_times, recruit=0.90, spc=11, isi=0.006):
        r = np.random.default_rng(seed)
        t, ix = [], []
        for bt in burst_times:
            part = r.choice(Nn, size=int(recruit * Nn), replace=False)
            for nid in part:
                t0 = bt + r.normal(0, 0.010)
                offs = np.arange(spc) * isi + r.normal(0, 0.0006, spc)
                t.extend(t0 + offs); ix.extend([int(nid)] * spc)
        bg = int(0.4 * Nn * T)
        t.extend(r.uniform(0, T, bg)); ix.extend(r.integers(0, Nn, bg))
        return np.clip(np.asarray(t, float), 0, T), np.asarray(ix, int)

    def save(i, spk_t, spk_i, a_seed=0, burst_times=None):
        o = np.argsort(spk_t)
        if burst_times is not None:
            at, ai = astro_events(a_seed, burst_times)
        else:
            at, ai = np.array([], np.float32), np.array([], np.int32)
        np.savez_compressed(
            topo / f"iter_{i:05d}.npz",
            spk_N_t=np.asarray(spk_t[o], np.float32),
            spk_N_i=np.asarray(spk_i[o], np.int32),
            spk_A_t=at, spk_A_i=ai,
            params=np.zeros(37), theta=np.zeros(37))

    # sim 0: clean bursting near the Mossink rate (~8 bursts / 60 s ~ 8/min)
    bt0 = np.array([6, 12.5, 19, 26, 33, 40.5, 47, 54.0])
    save(0, *mini_burst_sim(1, bt0), a_seed=1, burst_times=bt0)
    # sim 1: asynchronous Poisson (should screen as NOT bursting)
    n = rng.poisson(3.0 * Nn * T)
    save(1, np.sort(rng.uniform(0, T, n)), rng.integers(0, Nn, n))
    # sim 2: near-silent (should screen as NOT bursting)
    save(2, np.sort(rng.uniform(0, T, 10)), rng.integers(0, 2, 10))
    # sim 3: faster bursting (further from the Mossink rate -> should rank below 0)
    bt3 = np.linspace(3.0, 58.0, 16)
    save(3, *mini_burst_sim(2, bt3), a_seed=2, burst_times=bt3)
    return campaign


def _smoke_test() -> int:
    import tempfile
    print("Smoke test -- run_burst_analysis")
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              f"{('  ' + detail) if detail else ''}")
        ok = ok and bool(cond)

    root = Path(tempfile.mkdtemp(prefix="smoke_runburst_"))
    campaign = _make_synthetic_campaign(root)

    opt = Options(campaign=campaign, out_dir=campaign / "burst_poster",
                  topk=3, full_top=1, palettes=list(bp.DEFAULT_PALETTE_ORDER),
                  workers=1, candidate_pool=300, panel_mm=380, dpi=110,
                  fallback_Nn=80, fallback_T=60.0, verbose=False)

    res = run_pipeline(opt)

    # screen CSV exists and flagged the bursting sims
    csv_path = campaign / "burst_analysis" / "burst_features.csv"
    check("screen wrote burst_features.csv", csv_path.is_file())
    rows = read_features(csv_path) if csv_path.is_file() else []
    n_burst = sum(_is_true(r.get("is_bursting")) for r in rows)
    check("screen flagged >= 2 bursting sims", n_burst >= 2,
          f"bursting={n_burst}/{len(rows)}")

    # ranking + outputs
    ranked = res.get("ranked", [])
    check("at least 2 exemplars ranked", len(ranked) >= 2,
          f"ranked={len(ranked)}")
    check("literature distances sorted ascending",
          all(ranked[i]["literature_distance"] <= ranked[i + 1]["literature_distance"] + 1e-9
              for i in range(len(ranked) - 1)) if len(ranked) >= 2 else True)
    # the clean ~8/min sim (iter_00000) should be the closest to Mossink (5/min)
    if ranked:
        top_name = Path(ranked[0]["npz_path"]).name
        check("top exemplar is the clean-bursting sim (iter_00000)",
              top_name == "iter_00000.npz", f"top={top_name}")

    # summary + figures on disk
    summ = campaign / "burst_poster" / "burst_poster_summary.csv"
    check("summary CSV written", summ.is_file())
    out_dir = campaign / "burst_poster"
    pngs = list(out_dir.glob("exemplar01__*__hero.png"))
    pdfs = list(out_dir.glob("exemplar01__*__composite.pdf"))
    contact = list(out_dir.glob("contact_sheet__*.png"))
    check("exemplar01 hero PNG rendered in every palette",
          len(pngs) == len(opt.palettes), f"{len(pngs)} png")
    check("exemplar01 full-set composite PDF rendered", len(pdfs) >= 1,
          f"{len(pdfs)} pdf")
    check("contact sheet rendered", len(contact) >= 1, f"{len(contact)} png")
    nonempty = all(p.stat().st_size > 1200
                   for p in list(out_dir.glob('*.png')) + list(out_dir.glob('*.pdf')))
    check("all rendered figures are non-empty", nonempty)

    # astrocytes loaded from the npz and carried into RunData for rendering
    if ranked:
        D_top = build_rundata(ranked[0], opt, bm.BurstConfig(), "top")
        check("astrocytes loaded into top exemplar RunData",
              D_top.has_astro and D_top.n_astro == 40,
              f"n_astro={D_top.n_astro} "
              f"events={0 if D_top.spk_A_t is None else len(D_top.spk_A_t)}")

    print("\nSmoke test:", "PASS" if ok else "FAIL")
    if not ok:
        print("  (artefacts kept at", out_dir, ")")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
