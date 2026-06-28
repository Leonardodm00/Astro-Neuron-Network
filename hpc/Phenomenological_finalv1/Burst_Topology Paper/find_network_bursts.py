#!/usr/bin/env python3
"""
find_network_bursts.py  -  detect network bursts across an entire sweep campaign.

Walks  campaign_<TAG>/ ... /topo_*/iter_*.npz  (the authoritative spike data
written by HPC_main_sweep.py) and, for every simulation, computes population
network-burst statistics, then flags which simulations exhibit network bursting.

METHOD (faithful to Gao et al. 2024, AutoMIND, Methods "Network bursts")
------------------------------------------------------------------------
  1. Bin neuronal spikes at `bin_ms` (default 1 ms) and average across the Nn
     recorded neurons -> population firing-rate trace r(t).
  2. Smooth r(t) with a Gaussian window of std `smooth_ms` (default 5 ms).
  3. Detect burst peaks with scipy.signal.find_peaks using
        prominence = prominence_frac * max(r_smooth)   (default 0.8)
        distance   = min_ibi_s                          (default 0.5 s)
        wlen       = wlen_s                             (default 20 s)
  4. Inter-burst interval (IBI) = diff of peak times; report IBI mean & CV.
  5. Burst widths via scipy.signal.peak_widths at rel_height=0.95.

ADDED SYNCHRONY GATE (network-burst specificity)
------------------------------------------------
AutoMIND removes near-silent / pathological sims with separate QA exclusions
*before* burst detection. Here, sims are not pre-filtered, so a noise peak in a
near-silent trace could be mislabeled a "burst". We therefore additionally
compute, per detected peak, the fraction of distinct neurons firing within
+/- `sync_window_ms`, and average it over peaks (`burst_synchrony`). A genuine
NETWORK burst recruits a large fraction of the population; a few-cell blip does
not. The `is_bursting` flag requires BOTH:
        n_bursts >= min_bursts   AND   burst_synchrony >= min_synchrony.

OUTPUTS
-------
  <root>/burst_analysis/burst_features.csv     one row per simulation (all sims)
  if --write-index:
  <root>/burst_analysis/bursting_index/
        theta_all.npy   (M, n_dims)  inference-coord labels for the M bursting sims
        npz_paths.txt   (M,)         row-aligned absolute npz paths
        bursting_summary.csv (M,)    burst features for the bursting subset only

USAGE
-----
    python find_network_bursts.py campaign_cadex_500k_rv1 --workers 32 --write-index
    python find_network_bursts.py <root> --max-sims 2000           # quick look
    python find_network_bursts.py <root> --prominence-frac 0.5 --min-bursts 2
    python find_network_bursts.py --smoke-test                     # self-check

NOTES
-----
  * Reads Nn / simtime_s from each topo's topology_meta.json (one read per topo).
  * Only neuronal spikes (spk_N_t / spk_N_i) are used; astrocyte events ignored.
  * Embarrassingly parallel over simulations (ProcessPoolExecutor).
"""

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


# =============================================================================
# Config container (passed to each worker; keeps the signal pipeline decoupled
# from the I/O and the campaign-walk layers)
# =============================================================================
class BurstConfig:
    __slots__ = ('bin_ms', 'smooth_ms', 'prominence_frac', 'min_ibi_s',
                 'wlen_s', 'sync_window_ms', 'min_bursts', 'min_synchrony',
                 'min_amp_ratio', 'active_thresh_hz')

    def __init__(self, bin_ms=1.0, smooth_ms=5.0, prominence_frac=0.8,
                 min_ibi_s=0.5, wlen_s=20.0, sync_window_ms=25.0,
                 min_bursts=3, min_synchrony=0.20, min_amp_ratio=5.0,
                 active_thresh_hz=0.1):
        self.bin_ms          = bin_ms
        self.smooth_ms       = smooth_ms
        self.prominence_frac = prominence_frac
        self.min_ibi_s       = min_ibi_s
        self.wlen_s          = wlen_s
        self.sync_window_ms  = sync_window_ms
        self.min_bursts      = min_bursts
        self.min_synchrony   = min_synchrony
        self.min_amp_ratio   = min_amp_ratio
        self.active_thresh_hz = active_thresh_hz

    def as_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


# =============================================================================
# Signal pipeline (pure functions; no I/O)
# =============================================================================
def population_rate(spk_t: np.ndarray, Nn: int, simtime_s: float,
                    bin_s: float) -> np.ndarray:
    """Per-neuron-averaged population firing rate [Hz], binned at `bin_s`."""
    n_bins = max(int(round(simtime_s / bin_s)), 1)
    edges  = np.arange(n_bins + 1, dtype=np.float64) * bin_s
    counts, _ = np.histogram(spk_t, bins=edges)
    return counts.astype(np.float64) / (max(Nn, 1) * bin_s)


def smooth_rate(rate: np.ndarray, smooth_ms: float, bin_ms: float) -> np.ndarray:
    """Gaussian smoothing (std in samples = smooth_ms / bin_ms)."""
    sigma = max(smooth_ms / bin_ms, 1e-6)
    return gaussian_filter1d(rate, sigma=sigma, mode='constant')


def detect_burst_peaks(rsm: np.ndarray, bin_s: float, cfg: BurstConfig):
    """Return (peak_indices, peak_times_s, widths_s, peak_rate_hz)."""
    mx = float(rsm.max()) if rsm.size else 0.0
    if mx <= 0.0:
        return (np.array([], dtype=int), np.array([]), np.array([]), 0.0)

    # tiny jitter to break exactly-equal plateaus (AutoMIND adds ~1e-7)
    sig = rsm + np.random.default_rng(0).normal(0.0, 1e-7 * mx, size=rsm.shape)

    distance = max(1, int(round(cfg.min_ibi_s / bin_s)))
    wlen     = max(1, min(int(round(cfg.wlen_s / bin_s)), len(sig)))
    peaks, _ = find_peaks(sig,
                          prominence=cfg.prominence_frac * mx,
                          distance=distance,
                          wlen=wlen)
    if peaks.size == 0:
        return (peaks, np.array([]), np.array([]), mx)

    widths_samp, *_ = peak_widths(rsm, peaks, rel_height=0.95)
    return (peaks, peaks * bin_s, widths_samp * bin_s, mx)


def burst_synchrony(spk_t: np.ndarray, spk_i: np.ndarray, peak_times_s: np.ndarray,
                    Nn: int, sync_window_ms: float) -> float:
    """Mean over bursts of (distinct neurons firing within +/- window) / Nn."""
    if peak_times_s.size == 0 or spk_t.size == 0:
        return 0.0
    w = sync_window_ms / 1000.0
    order = np.argsort(spk_t, kind='stable')
    st, si = spk_t[order], spk_i[order]
    fracs = np.empty(peak_times_s.size, dtype=np.float64)
    for k, tp in enumerate(peak_times_s):
        lo = np.searchsorted(st, tp - w, side='left')
        hi = np.searchsorted(st, tp + w, side='right')
        fracs[k] = (np.unique(si[lo:hi]).size / max(Nn, 1)) if hi > lo else 0.0
    return float(fracs.mean())


def compute_burst_features(spk_t: np.ndarray, spk_i: np.ndarray,
                           Nn: int, simtime_s: float,
                           cfg: BurstConfig) -> Dict:
    """Full per-simulation burst-feature dict (orchestrates the pipeline)."""
    bin_s = cfg.bin_ms / 1000.0
    n_spk = int(spk_t.size)

    # population-level mean stats (cheap; also used as guards)
    mean_fr = n_spk / (simtime_s * max(Nn, 1)) if simtime_s > 0 else 0.0
    counts  = np.bincount(spk_i, minlength=Nn).astype(np.float64) if n_spk else np.zeros(Nn)
    rates   = counts / simtime_s if simtime_s > 0 else counts
    frac_active = float((rates > cfg.active_thresh_hz).mean()) if Nn else 0.0

    rate = population_rate(spk_t, Nn, simtime_s, bin_s)
    rsm  = smooth_rate(rate, cfg.smooth_ms, cfg.bin_ms)
    peaks, pt, widths_s, peak_rate = detect_burst_peaks(rsm, bin_s, cfg)

    n_bursts = int(peaks.size)
    burst_rate_hz = n_bursts / simtime_s if simtime_s > 0 else 0.0

    # IBI statistics (need >=2 bursts for an interval; >=3 for a CV)
    if n_bursts >= 2:
        ibi = np.diff(pt)
        ibi_mean = float(ibi.mean())
        ibi_cv   = float(ibi.std() / ibi.mean()) if (n_bursts >= 3 and ibi.mean() > 0) else float('nan')
    else:
        ibi_mean = float('nan')
        ibi_cv   = float('nan')

    if widths_s.size:
        bw_mean = float(widths_s.mean())
        bw_cv   = float(widths_s.std() / widths_s.mean()) if (widths_s.size >= 2 and widths_s.mean() > 0) else float('nan')
    else:
        bw_mean = float('nan')
        bw_cv   = float('nan')

    sync = burst_synchrony(spk_t, spk_i, pt, Nn, cfg.sync_window_ms)

    # Peak-to-mean amplitude ratio: a NETWORK burst spikes the population rate
    # far above its own mean. Rate- and Nn-normalised, so it cleanly separates
    # transient synchronous volleys from asynchronous (Poisson-like) activity
    # whose smoothed rate hovers near its mean.
    amp_ratio = float(peak_rate / mean_fr) if mean_fr > 0 else 0.0

    # A network burst requires (i) enough bursts for an IBI statistic,
    # (ii) substantial population recruitment, AND (iii) a true rate transient.
    is_bursting = (n_bursts >= cfg.min_bursts) and \
                  (sync >= cfg.min_synchrony) and \
                  (amp_ratio >= cfg.min_amp_ratio)

    return dict(
        n_spikes=n_spk,
        mean_FR_Hz=float(mean_fr),
        frac_active=frac_active,
        peak_rate_hz=float(peak_rate),
        burst_amp_ratio=amp_ratio,
        n_bursts=n_bursts,
        burst_rate_hz=float(burst_rate_hz),
        ibi_mean_s=ibi_mean,
        ibi_cv=ibi_cv,
        burst_width_mean_s=bw_mean,
        burst_width_cv=bw_cv,
        burst_synchrony=sync,
        is_bursting=bool(is_bursting),
    )


# =============================================================================
# Worker (I/O + feature computation for one npz)
# =============================================================================
_CFG: Optional[BurstConfig] = None        # set per-process via initializer


def _init_worker(cfg_dict: dict):
    global _CFG
    _CFG = BurstConfig(**cfg_dict)


def _process_one(work_item: Tuple[str, int, float]) -> Optional[dict]:
    npz_path, Nn, simtime_s = work_item
    try:
        with np.load(npz_path) as d:
            spk_t = np.asarray(d['spk_N_t'], dtype=np.float64)
            spk_i = np.asarray(d['spk_N_i'], dtype=np.int64)
            theta = np.asarray(d['theta'], dtype=np.float64) if 'theta' in d else None
        feats = compute_burst_features(spk_t, spk_i, Nn, simtime_s, _CFG)
    except Exception as e:           # never let one bad file kill the campaign walk
        return {'_error': f'{type(e).__name__}: {e}', 'npz_path': npz_path}

    p = Path(npz_path)
    feats.update(dict(
        npz_path=str(p.resolve()),
        topo_dir=p.parent.name,
        task=p.parent.parent.name,
        iter_idx=_iter_idx_from_name(p.stem),
        Nn=int(Nn),
        simtime_s=float(simtime_s),
    ))
    # carry theta only for bursting sims (keeps the returned payload small)
    if feats['is_bursting'] and theta is not None:
        feats['_theta'] = theta.tolist()
    return feats


def _iter_idx_from_name(stem: str) -> int:
    try:
        return int(stem.split('_')[1])
    except (IndexError, ValueError):
        return -1


# =============================================================================
# Campaign walk (resolve Nn / simtime_s once per topo)
# =============================================================================
def _read_topo_meta(topo_dir: Path) -> Tuple[Optional[int], Optional[float]]:
    mp = topo_dir / 'topology_meta.json'
    if mp.exists():
        try:
            m = json.loads(mp.read_text())
            return m.get('Nn'), m.get('simtime_s')
        except Exception:
            pass
    return None, None


def build_work_list(root: Path, max_sims: Optional[int],
                    fallback_Nn: int, fallback_T: float) -> List[Tuple[str, int, float]]:
    topo_dirs = sorted({p.parent for p in root.rglob('iter_*.npz')})
    if not topo_dirs:
        return []
    # campaign-root fallback Nn/T from job_args.json if present
    ja = root / 'job_args.json'
    if ja.exists():
        try:
            jd = json.loads(ja.read_text())
            fallback_Nn = int(jd.get('Nn', fallback_Nn))
            fallback_T  = float(jd.get('simtime', fallback_T))
        except Exception:
            pass

    work = []
    for td in topo_dirs:
        Nn, T = _read_topo_meta(td)
        Nn = int(Nn) if Nn else fallback_Nn
        T  = float(T) if T else fallback_T
        for npz in sorted(td.glob('iter_*.npz')):
            work.append((str(npz), Nn, T))
            if max_sims and len(work) >= max_sims:
                return work
    return work


# =============================================================================
# Output schema
# =============================================================================
CSV_FIELDS = [
    'task', 'topo_dir', 'iter_idx', 'Nn', 'simtime_s',
    'n_spikes', 'mean_FR_Hz', 'frac_active', 'peak_rate_hz', 'burst_amp_ratio',
    'n_bursts', 'burst_rate_hz', 'ibi_mean_s', 'ibi_cv',
    'burst_width_mean_s', 'burst_width_cv', 'burst_synchrony',
    'is_bursting', 'npz_path',
]


def _csv_row(feats: dict) -> dict:
    return {k: feats.get(k) for k in CSV_FIELDS}


# =============================================================================
# Driver
# =============================================================================
def run_campaign(root: Path, cfg: BurstConfig, workers: int,
                 max_sims: Optional[int], write_index: bool,
                 fallback_Nn: int, fallback_T: float,
                 verbose: bool = True) -> dict:
    work = build_work_list(root, max_sims, fallback_Nn, fallback_T)
    if not work:
        print(f"No iter_*.npz found under {root}", file=sys.stderr)
        return {}

    out_dir = root / 'burst_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'burst_features.csv'

    if verbose:
        print(f"[bursts] root      : {root}")
        print(f"[bursts] sims      : {len(work):,}")
        print(f"[bursts] workers   : {workers}")
        print(f"[bursts] config    : {cfg.as_dict()}")
        print(f"[bursts] writing   : {csv_path}")

    n_done = n_burst = n_err = 0
    theta_rows: List[List[float]] = []
    npz_rows:   List[str] = []
    burst_summ: List[dict] = []
    nb_hist: List[int] = []

    fh = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    writer.writeheader()

    chunk = max(1, min(256, len(work) // (workers * 4) or 1))
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init_worker,
                             initargs=(cfg.as_dict(),)) as ex:
        for feats in ex.map(_process_one, work, chunksize=chunk):
            n_done += 1
            if feats is None or '_error' in feats:
                n_err += 1
                if verbose and feats and n_err <= 10:
                    print(f"  ! {feats.get('_error')}  ({feats.get('npz_path')})",
                          file=sys.stderr)
            else:
                writer.writerow(_csv_row(feats))
                nb_hist.append(feats['n_bursts'])
                if feats['is_bursting']:
                    n_burst += 1
                    if write_index and '_theta' in feats:
                        theta_rows.append(feats['_theta'])
                        npz_rows.append(feats['npz_path'])
                        burst_summ.append(_csv_row(feats))
            if verbose and n_done % 5000 == 0:
                print(f"  ... {n_done:,}/{len(work):,}  "
                      f"({n_burst:,} bursting, {n_err} errors)", flush=True)
    fh.close()

    # ---- summary -----------------------------------------------------------
    nb = np.asarray(nb_hist) if nb_hist else np.array([0])
    pct = 100.0 * n_burst / max(n_done - n_err, 1)
    if verbose:
        print("-" * 60)
        print(f"[bursts] processed : {n_done:,}  ({n_err} read errors)")
        print(f"[bursts] BURSTING  : {n_burst:,}  ({pct:.2f}% of valid sims)")
        print(f"[bursts] n_bursts/sim  p50={np.percentile(nb,50):.0f}  "
              f"p90={np.percentile(nb,90):.0f}  max={nb.max():.0f}")
        print(f"[bursts] features  -> {csv_path}")

    # ---- SBI-ready index of bursting sims ----------------------------------
    if write_index and theta_rows:
        idx = out_dir / 'bursting_index'
        idx.mkdir(exist_ok=True)
        np.save(idx / 'theta_all.npy', np.asarray(theta_rows, dtype=np.float64))
        (idx / 'npz_paths.txt').write_text("\n".join(npz_rows) + "\n")
        with open(idx / 'bursting_summary.csv', 'w', newline='') as f2:
            w2 = csv.DictWriter(f2, fieldnames=CSV_FIELDS)
            w2.writeheader(); w2.writerows(burst_summ)
        if verbose:
            print(f"[bursts] index     -> {idx}/  "
                  f"(theta_all.npy {np.asarray(theta_rows).shape}, "
                  f"npz_paths.txt, bursting_summary.csv)")

    return dict(n_processed=n_done, n_bursting=n_burst, n_errors=n_err,
                csv=str(csv_path))


# =============================================================================
# Smoke test  (--smoke-test): synthesize a tiny campaign and verify detection
# =============================================================================
def _run_smoke_test() -> int:
    import tempfile, shutil
    rng = np.random.default_rng(7)
    root = Path(tempfile.mkdtemp(prefix='smoke_bursts_'))
    topo = root / 'sweep_test_task0000' / 'topo_00000'
    topo.mkdir(parents=True)

    Nn, T = 100, 10.0
    (topo / 'topology_meta.json').write_text(json.dumps(
        {'Nn': Nn, 'simtime_s': T, 'mode': 'Neuronal', 'topo_idx': 0}))

    def save_npz(i, spk_t, spk_i, theta=None):
        np.savez_compressed(
            topo / f'iter_{i:05d}.npz',
            spk_N_t=np.asarray(spk_t, np.float32),
            spk_N_i=np.asarray(spk_i, np.int32),
            spk_A_t=np.array([], np.float32), spk_A_i=np.array([], np.int32),
            params=np.zeros(35), theta=(theta if theta is not None else np.zeros(35)))

    # --- sim 0: clean network bursting (8 synchronous volleys) -------------
    t, ix = [], []
    burst_times = np.array([1.0, 2.2, 3.3, 4.5, 5.6, 6.8, 7.9, 9.1])
    for bt in burst_times:
        part = rng.choice(Nn, size=85, replace=False)          # 85% recruited
        t.extend(bt + rng.normal(0, 0.006, part.size))         # ~6 ms jitter
        ix.extend(part)
    bg = int(0.3 * Nn * T)                                      # sparse background
    t.extend(rng.uniform(0, T, bg)); ix.extend(rng.integers(0, Nn, bg))
    save_npz(0, t, ix)

    # --- sim 1: asynchronous (Poisson 3 Hz/neuron, no structure) -----------
    n = rng.poisson(3.0 * Nn * T)
    save_npz(1, np.sort(rng.uniform(0, T, n)), rng.integers(0, Nn, n))

    # --- sim 2: near-silent (a handful of spikes from 2 cells) -------------
    save_npz(2, np.sort(rng.uniform(0, T, 8)), rng.integers(0, 2, 8))

    # --- sim 3: regular bursting (12 volleys -> low IBI CV) ----------------
    t, ix = [], []
    for bt in np.linspace(0.7, 9.5, 12):
        part = rng.choice(Nn, size=70, replace=False)
        t.extend(bt + rng.normal(0, 0.005, part.size)); ix.extend(part)
    save_npz(3, t, ix)

    print("Smoke test  -  find_network_bursts")
    cfg = BurstConfig()
    res = run_campaign(root, cfg, workers=2, max_sims=None, write_index=True,
                       fallback_Nn=Nn, fallback_T=T, verbose=False)

    # read back the per-sim CSV and assert expected classifications
    import csv as _csv
    rows = {}
    with open(root / 'burst_analysis' / 'burst_features.csv') as fh:
        for r in _csv.DictReader(fh):
            rows[int(r['iter_idx'])] = r

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}{('  '+detail) if detail else ''}")
        ok = ok and cond

    check("sim0 (synchronous volleys) -> bursting",
          rows[0]['is_bursting'] == 'True',
          f"n_bursts={rows[0]['n_bursts']} sync={float(rows[0]['burst_synchrony']):.2f}")
    check("sim1 (asynchronous) -> NOT bursting",
          rows[1]['is_bursting'] == 'False',
          f"n_bursts={rows[1]['n_bursts']} sync={float(rows[1]['burst_synchrony']):.2f}")
    check("sim2 (near-silent) -> NOT bursting",
          rows[2]['is_bursting'] == 'False',
          f"n_bursts={rows[2]['n_bursts']}")
    check("sim3 (regular bursting) -> bursting",
          rows[3]['is_bursting'] == 'True',
          f"n_bursts={rows[3]['n_bursts']} ibi_mean={rows[3]['ibi_mean_s']} "
          f"ibi_cv={rows[3]['ibi_cv']}")
    check("sim3 detects ~12 bursts",
          abs(int(rows[3]['n_bursts']) - 12) <= 2,
          f"got {rows[3]['n_bursts']}")
    check("sim3 regular -> low IBI CV (<0.25)",
          float(rows[3]['ibi_cv']) < 0.25,
          f"ibi_cv={rows[3]['ibi_cv']}")
    idx_ok = (root / 'burst_analysis' / 'bursting_index' / 'theta_all.npy').exists()
    check("bursting_index written", idx_ok)

    if ok:
        shutil.rmtree(root)
        print("\nSmoke test: PASS")
        return 0
    print(f"\nSmoke test: FAIL  (artefacts kept at {root})")
    return 1


# =============================================================================
# CLI
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('campaign_root', nargs='?',
                   help='campaign_<TAG>/ directory to scan')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--max-sims', type=int, default=None,
                   help='process only the first N sims (quick look)')
    p.add_argument('--write-index', action='store_true',
                   help='emit SBI-ready theta_all.npy / npz_paths.txt for bursting sims')
    # detection params (defaults follow AutoMIND)
    p.add_argument('--bin-ms', type=float, default=1.0)
    p.add_argument('--smooth-ms', type=float, default=5.0)
    p.add_argument('--prominence-frac', type=float, default=0.8)
    p.add_argument('--min-ibi-s', type=float, default=0.5)
    p.add_argument('--wlen-s', type=float, default=20.0)
    p.add_argument('--sync-window-ms', type=float, default=25.0)
    p.add_argument('--min-bursts', type=int, default=3)
    p.add_argument('--min-synchrony', type=float, default=0.20,
                   help='min fraction of neurons recruited per burst')
    p.add_argument('--min-amp-ratio', type=float, default=5.0,
                   help='min peak-rate / mean-rate ratio for a network burst')
    p.add_argument('--active-thresh-hz', type=float, default=0.1)
    # fallbacks if topology_meta.json is missing
    p.add_argument('--fallback-Nn', type=int, default=100)
    p.add_argument('--fallback-simtime', type=float, default=180.0)
    p.add_argument('--smoke-test', action='store_true')
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.smoke_test:
        return _run_smoke_test()
    if not args.campaign_root:
        _build_parser().print_help()
        return 1

    root = Path(args.campaign_root).resolve()
    if not root.is_dir():
        print(f"campaign root not found: {root}", file=sys.stderr)
        return 1

    cfg = BurstConfig(
        bin_ms=args.bin_ms, smooth_ms=args.smooth_ms,
        prominence_frac=args.prominence_frac, min_ibi_s=args.min_ibi_s,
        wlen_s=args.wlen_s, sync_window_ms=args.sync_window_ms,
        min_bursts=args.min_bursts, min_synchrony=args.min_synchrony,
        min_amp_ratio=args.min_amp_ratio, active_thresh_hz=args.active_thresh_hz)

    run_campaign(root, cfg, workers=args.workers, max_sims=args.max_sims,
                 write_index=args.write_index,
                 fallback_Nn=args.fallback_Nn, fallback_T=args.fallback_simtime)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
