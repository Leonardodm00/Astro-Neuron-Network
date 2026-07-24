#!/usr/bin/env python3
# process_campaign.py
# =============================================================================
# Drive the virtual-MEA pipeline over an Astro-Neuron-Network campaign.
#
# INPUT LAYOUT (produced by HPC_main_sweep.py)
#   <campaign>/
#     job_args.json                     (c_max fallback)
#     topo_<k>/
#       topology.npz                    N_pos (Nn,2) [um], ...
#       topology_meta.json              c_max, seeds, ...
#       iter_<n>.npz                    params, theta, spk_N_t [s], spk_N_i, ...
#
# WHAT THIS DOES
#   For each topo_<k>:  build the probe + weight matrix W + reach mask + a
#   stable per-neuron template assignment ONCE (topology is shared by all its
#   iters). For each iter_<n>:  synthesise the E-channel trace from spike
#   times, band-pass + detect (Quiroga), match detections to true spikes, and
#   write mea_iter_<n>.npz (detections + parameter vector + provenance).
#
# OUTPUT (per iter)  <out>/topo_<k>/mea_iter_<n>.npz
#   det_ch, det_t, det_amp            detected events (channel, time[s], uV)
#   det_src_neuron, det_dt_to_truth   ground-truth link (-1 / nan = false pos)
#   sigma                             (E,) per-channel robust noise [uV]
#   electrode_centers                 (E,2) [um]
#   params, theta                     the SWEPT parameter vector + SBI coords
#   conn_prob, topo_idx, iter_idx, seed_run
#   meta_json                         full pipeline config (reproducibility)
#   traces (optional)                 (E,T) raw trace, only for a saved subset
#
# SEPARATION OF CONCERNS: this file is orchestration + IO only. Physics lives
# in mea_probe / mea_synthesis / mea_detection; the EAP library in
# eap_template_library.
#
# USAGE
#   python process_campaign.py --campaign <dir> --out <dir> --library eap_library.npz
#   python process_campaign.py --campaign <dir> --out <dir> --workers 32
#   python process_campaign.py --self_test        # 1 synthetic topo end-to-end
#   python process_campaign.py --help
# =============================================================================

# =============================================================================
#
# THREADING (read this before changing --workers)
#   This pipeline's parallelism is entirely at the PROCESS level: walk_campaign
#   below spawns one worker process per topology (multiprocessing.Pool). Each
#   worker's own numpy/scipy calls must therefore run SINGLE-threaded, or a
#   run with --workers N can silently oversubscribe the node: numpy here is
#   commonly built against OpenBLAS, whose default thread count is the number
#   of VISIBLE CPUs (e.g. 48 on a `-l select=1:ncpus=48` PBS allocation) --
#   unless told otherwise, EVERY one of the N worker processes would each try
#   to run its own BLAS calls across up to 48 threads, contending for the same
#   48 physical cores. The fix is the four os.environ lines immediately below,
#   which MUST run before numpy (or anything that imports numpy, including
#   every mea_*.py module a few lines down) is imported anywhere in the
#   process -- OpenBLAS/MKL/OMP read their thread count at library-LOAD time,
#   not at call time, so this only works if it is the first thing this file
#   does. Under multiprocessing's 'spawn' context (used below), each worker
#   re-executes this module from the top in a fresh interpreter, so this same
#   guard applies automatically to every worker, not just the main process.
#   setdefault (not a hard assignment) is used deliberately: if you have
#   already set these yourself -- e.g. to deliberately leave a couple of BLAS
#   threads per worker because --workers is set below the core count -- your
#   value is respected rather than overwritten.
# =============================================================================
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')   # macOS Accelerate; harmless elsewhere

import argparse
import glob
import json
import multiprocessing as mp
import tempfile
import time
import traceback
from dataclasses import asdict

import numpy as np

import mea_probe as P
import mea_synthesis as S
import mea_detection as D
import mea_plots as PL
import eap_template_library as EAP


# =============================================================================
# Config
# =============================================================================

class PipelineConfig:
    """All tunable knobs in one place (kept plain for easy JSON round-trip)."""
    def __init__(self, **kw):
        # probe
        self.n_side = kw.get('n_side', 3)
        self.pitch = kw.get('pitch', 60.0)
        self.edge = kw.get('edge', 25.0)
        self.n_sub = kw.get('n_sub', 4)
        # scaling law
        self.n_dec = kw.get('n_dec', 2.0)
        self.r_ref = kw.get('r_ref', 30.0)
        self.r_min = kw.get('r_min', 10.0)
        self.snr_ref = kw.get('snr_ref', 15.0)
        self.gamma = kw.get('gamma', 0.5)
        # noise / recording
        self.fs = kw.get('fs', 10110.09)
        self.target_noise_uv = kw.get('target_noise_uv', 5.0)
        # detection
        self.band_lo = kw.get('band_lo', 300.0)
        self.band_hi = kw.get('band_hi', 3000.0)
        self.filt_order = kw.get('filt_order', 4)
        self.k = kw.get('k', 5.0)
        self.refractory_ms = kw.get('refractory_ms', 2.0)
        self.polarity = kw.get('polarity', 'neg')
        # template assignment
        self.tmpl_seed_base = kw.get('tmpl_seed_base', 20260)
        self.noise_seed_base = kw.get('noise_seed_base', 70000)
        # match window
        self.match_window_ms = kw.get('match_window_ms', 1.5)
        # diagnostics plots
        self.plots = kw.get('plots', False)          # master switch
        self.plot_window_s = kw.get('plot_window_s', 2.0)
        self.plot_max_true_rows = kw.get('plot_max_true_rows', 120)

    def A_ref(self):
        """Reference amplitude [uV] = SNR_ref * target post-filter noise."""
        return self.snr_ref * self.target_noise_uv

    def scaling(self):
        return P.ScalingConfig(A_ref=self.A_ref(), r_ref=self.r_ref,
                               r_min=self.r_min, n_dec=self.n_dec)

    def probe_cfg(self):
        return P.ProbeConfig(n_side=self.n_side, pitch=self.pitch,
                             edge=self.edge, n_sub=self.n_sub)

    def detect_cfg(self):
        return D.DetectConfig(lo_hz=self.band_lo, hi_hz=self.band_hi,
                              order=self.filt_order, k=self.k,
                              refractory_ms=self.refractory_ms,
                              polarity=self.polarity)

    def to_dict(self):
        return dict(self.__dict__)


# =============================================================================
# IO helpers
# =============================================================================

def _atomic_savez(path, **arrays):
    d = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(suffix='.npz', dir=d)
    os.close(fd)
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp + '.npz' if os.path.exists(tmp + '.npz') else tmp, path)


def read_c_max(topo_dir, campaign_dir, N_pos):
    """c_max from topology_meta.json, else job_args.json, else inferred."""
    meta = os.path.join(topo_dir, 'topology_meta.json')
    if os.path.exists(meta):
        try:
            j = json.loads(open(meta).read())
            if 'c_max' in j:
                return float(j['c_max'])
        except Exception:
            pass
    ja = os.path.join(campaign_dir, 'job_args.json')
    if os.path.exists(ja):
        try:
            j = json.loads(open(ja).read())
            if 'c_max' in j:
                return float(j['c_max'])
        except Exception:
            pass
    # Fallback: infer a square arena from the position extent.
    return float(np.ceil(N_pos.max()))


# =============================================================================
# Core per-iter and per-topo processing
# =============================================================================

def _visible_rows(W, reach, max_rows):
    """
    Neuron ids to show in the ground-truth raster: those within reach, capped
    at `max_rows` by taking the most strongly coupled ones. A full culture has
    far too many neurons for a legible raster.
    """
    ids = np.where(reach)[0] if reach is not None else np.arange(W.shape[0])
    if ids.size <= max_rows:
        return ids
    strength = W[ids].max(axis=1)
    keep = ids[np.argsort(strength)[::-1][:max_rows]]
    return np.sort(keep)


def process_iter(iter_npz, probe, W, reach, lib_fs, tmpl_assign, cfg,
                 noise_rms, out_path, topo_idx, save_traces=False,
                 plot_dir=None):
    """Synthesize + detect + match for a single iter; write output npz."""
    d = np.load(iter_npz, allow_pickle=False)
    spk_t = np.asarray(d['spk_N_t'], dtype=float)
    spk_i = np.asarray(d['spk_N_i'], dtype=np.int64)
    params = d['params'] if 'params' in d else np.array([])
    theta = d['theta'] if 'theta' in d else np.array([])
    conn_prob = float(d['conn_prob']) if 'conn_prob' in d else float('nan')
    seed_run = int(d['seed_run']) if 'seed_run' in d else -1
    iter_idx = _iter_index_from_name(iter_npz)

    # simtime: infer from the last spike (rounded up) if not otherwise known.
    simtime = float(np.ceil(spk_t.max())) if spk_t.size else 1.0

    noise_seed = cfg.noise_seed_base + 1000 * topo_idx + iter_idx
    traces, t = S.synthesize_traces(
        spk_t, spk_i, W, lib_fs, tmpl_assign, cfg.fs, simtime,
        noise_rms=noise_rms, reach=reach, seed=noise_seed)

    det = D.detect(traces, cfg.fs, cfg.detect_cfg())
    src, dt = D.match_to_truth(det, W, spk_t, spk_i, cfg.fs,
                               window_ms=cfg.match_window_ms)

    out = dict(
        det_ch=det['ch'].astype(np.int32),
        det_t=det['t'].astype(np.float32),
        det_amp=det['amp'].astype(np.float32),
        det_src_neuron=src.astype(np.int32),
        det_dt_to_truth=dt.astype(np.float32),
        sigma=det['sigma'].astype(np.float32),
        electrode_centers=probe['centers'].astype(np.float32),
        params=params, theta=theta,
        conn_prob=np.float64(conn_prob),
        topo_idx=np.int32(topo_idx),
        iter_idx=np.int32(iter_idx),
        seed_run=np.int64(seed_run),
        fs=np.float64(cfg.fs),
        simtime=np.float64(simtime),
        meta_json=np.array(json.dumps(cfg.to_dict())),
    )
    if save_traces:
        out['traces'] = traces.astype(np.float32)
    _atomic_savez(out_path, **out)

    # ---- optional diagnostics (never fatal: a plotting failure must not
    #      lose the numerical result that was just written above) ----------
    if plot_dir is not None and PL.HAVE_MPL:
        try:
            PL.save_iter_diagnostics(
                plot_dir, traces, det['filtered'], cfg.fs, det, src, dt,
                cfg.k, simtime, spk_t=spk_t, spk_i=spk_i,
                visible_neurons=_visible_rows(W, reach,
                                              cfg.plot_max_true_rows),
                window_s=cfg.plot_window_s)
        except Exception as exc:  # noqa: BLE001
            print('[mea] WARNING: iter plots failed for %s: %r'
                  % (iter_npz, exc))

    return len(det['t'])


def _iter_index_from_name(path):
    base = os.path.basename(path)
    digits = ''.join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else 0


def process_topo(topo_dir, campaign_dir, out_topo_dir, lib, cfg,
                 save_traces_first=False, limit=None):
    """Build geometry once, then process every iter in this topo dir."""
    os.makedirs(out_topo_dir, exist_ok=True)
    topo_idx = _iter_index_from_name(topo_dir)

    topo = np.load(os.path.join(topo_dir, 'topology.npz'), allow_pickle=False)
    N_pos = np.asarray(topo['N_pos'], dtype=float)
    Nn = N_pos.shape[0]
    c_max = read_c_max(topo_dir, campaign_dir, N_pos)

    # ---- geometry + weights + reach + template assignment (ONCE) ----
    probe = P.make_probe(c_max, cfg.probe_cfg())
    W = P.compute_weights(N_pos, probe, cfg.scaling())
    reach = P.reach_mask(W, floor=cfg.gamma * cfg.target_noise_uv)
    n_templates = lib['templates'].shape[0]
    tmpl_assign = S.assign_templates(
        Nn, n_templates, seed=cfg.tmpl_seed_base + topo_idx)
    lib_fs = S.prepare_library_at_fs(lib, cfg.fs)

    # pre-filter noise RMS to hit the target post-filter sigma
    noise_rms = S.calibrate_noise_rms(
        cfg.fs, (cfg.band_lo, cfg.band_hi), cfg.target_noise_uv,
        order=cfg.filt_order)

    iters = sorted(glob.glob(os.path.join(topo_dir, 'iter_*.npz')))
    if limit is not None:
        iters = iters[:limit]

    # ---- topology-level diagnostics: geometry + scaling law, drawn ONCE ---
    plot_root = os.path.join(out_topo_dir, 'plots') if cfg.plots else None
    if plot_root is not None and PL.HAVE_MPL:
        try:
            PL.save_topo_diagnostics(
                plot_root, N_pos, probe, W, cfg.scaling(),
                r_max=P.r_max_reach(cfg.scaling(), cfg.gamma,
                                    cfg.target_noise_uv),
                noise_floor=cfg.gamma * cfg.target_noise_uv,
                templates=lib['templates'], dt_hr_ms=lib['dt_hr_ms'])
        except Exception as exc:  # noqa: BLE001
            print('[mea] WARNING: topo plots failed for %s: %r'
                  % (topo_dir, exc))
    elif plot_root is not None and not PL.HAVE_MPL:
        print('[mea] WARNING: --plots requested but matplotlib is missing; '
              'skipping all plots')

    n_done = 0
    for k, ip in enumerate(iters):
        out_path = os.path.join(
            out_topo_dir, 'mea_' + os.path.basename(ip))
        save_tr = bool(save_traces_first and k == 0)
        # signal plots only for the FIRST iter of each topo (storage/time)
        pdir = (os.path.join(plot_root, 'iter_%d' % _iter_index_from_name(ip))
                if (plot_root is not None and k == 0) else None)
        try:
            process_iter(ip, probe, W, reach, lib_fs, tmpl_assign, cfg,
                         noise_rms, out_path, topo_idx, save_traces=save_tr,
                         plot_dir=pdir)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            with open(os.path.join(out_topo_dir, '_failures.log'), 'a') as f:
                f.write('%s: %r\n%s\n' % (ip, exc, traceback.format_exc()))
    return topo_idx, len(iters), n_done


# =============================================================================
# Campaign walk (+ multiprocessing over topos)
# =============================================================================

def _worker(pack):
    (topo_dir, campaign_dir, out_topo_dir, lib_path, cfg_dict,
     save_traces_first, limit) = pack
    lib = EAP.load_library(lib_path)
    cfg = PipelineConfig(**cfg_dict)
    return process_topo(topo_dir, campaign_dir, out_topo_dir, lib, cfg,
                        save_traces_first=save_traces_first, limit=limit)


def walk_campaign(campaign_dir, out_dir, lib_path, cfg, workers=1,
                  save_traces_first=False, limit_iters=None, limit_topos=None):
    os.makedirs(out_dir, exist_ok=True)

    # Diagnose the input path BEFORE globbing. glob() returns [] both for a
    # path that does not exist and for one that exists but holds no topo_*,
    # so globbing first would collapse a simple typo (by far the most common
    # cause) into a misleading "no topo_* directories" message.
    if not os.path.exists(campaign_dir):
        parent = os.path.dirname(campaign_dir.rstrip(os.sep))
        msg = ['campaign path does not exist: ' + campaign_dir]
        if parent and os.path.isdir(parent):
            siblings = sorted(os.listdir(parent))[:20]
            msg.append('parent %s contains: %s'
                       % (parent, ', '.join(siblings) if siblings else '(empty)'))
        else:
            msg.append('parent directory %s does not exist either -- check the '
                       'path from the top (username, campaign name, task number)'
                       % parent)
        raise SystemExit('\n'.join(msg))
    if not os.path.isdir(campaign_dir):
        raise SystemExit('campaign path is not a directory: ' + campaign_dir)

    topo_dirs = sorted(glob.glob(os.path.join(campaign_dir, 'topo_*')))
    if not topo_dirs:
        entries = sorted(os.listdir(campaign_dir))[:20]
        raise SystemExit(
            'no topo_* directories under ' + campaign_dir + '\n'
            'this directory contains: '
            + (', '.join(entries) if entries else '(empty)') + '\n'
            'NB: --campaign must point at the directory that DIRECTLY contains '
            'topo_*/ (a sweep_<TAG>_task<IDX> dir), not the campaign_<TAG> root '
            'above it.')
    if limit_topos is not None:
        topo_dirs = topo_dirs[:limit_topos]

    # persist the library once so workers load an identical copy
    lib_path_abs = os.path.abspath(lib_path)
    packs = []
    for td in topo_dirs:
        out_td = os.path.join(out_dir, os.path.basename(td))
        packs.append((td, campaign_dir, out_td, lib_path_abs, cfg.to_dict(),
                      save_traces_first, limit_iters))

    print('[mea] %d topo(s) found, dispatching over %d worker(s) '
          '(1 topo/worker; BLAS pinned: OMP=%s OPENBLAS=%s MKL=%s)'
          % (len(topo_dirs), max(workers, 1), os.environ.get('OMP_NUM_THREADS'),
             os.environ.get('OPENBLAS_NUM_THREADS'), os.environ.get('MKL_NUM_THREADS')))
    if workers > len(topo_dirs):
        print('[mea] NOTE: --workers %d exceeds the %d topologies found; '
              '%d worker(s) will sit idle this run' % (
                  workers, len(topo_dirs), workers - len(topo_dirs)))

    t0 = time.time()
    if workers <= 1:
        results = [_worker(p) for p in packs]
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers) as pool:
            results = pool.map(_worker, packs)
    dt = time.time() - t0

    total_iters = sum(r[1] for r in results)
    total_done = sum(r[2] for r in results)
    print('[mea] %d topos, %d/%d iters processed in %.1f s'
          % (len(results), total_done, total_iters, dt))
    _write_manifest(out_dir, results, cfg)
    return results


def _write_manifest(out_dir, results, cfg):
    man = dict(
        n_topos=len(results),
        total_iters=int(sum(r[1] for r in results)),
        total_done=int(sum(r[2] for r in results)),
        config=cfg.to_dict(),
        created=time.strftime('%Y-%m-%dT%H:%M:%S'),
    )
    with open(os.path.join(out_dir, 'mea_manifest.json'), 'w') as f:
        json.dump(man, f, indent=2)


# =============================================================================
# Library resolution
# =============================================================================

def resolve_library(path):
    """Load a library from `path`, or generate a default one if missing."""
    if path and os.path.exists(path):
        return EAP.load_library(path), path
    gen_path = path or 'eap_library.npz'
    print('[mea] library not found; generating default -> ' + gen_path)
    lib = EAP.generate_template_library(n_templates=30, seed=0)
    EAP.save_library(lib, gen_path)
    return lib, gen_path


# =============================================================================
# Self-test: one synthetic topo end-to-end (no campaign needed)
# =============================================================================

def self_test(tmpdir=None):
    import shutil
    root = tmpdir or tempfile.mkdtemp(prefix='mea_selftest_')
    camp = os.path.join(root, 'campaign')
    td = os.path.join(camp, 'topo_00000')
    os.makedirs(td, exist_ok=True)

    rng = np.random.default_rng(0)
    c_max = 600.0
    Nn = 40
    N_pos = rng.uniform(0, c_max, (Nn, 2))
    # place neuron 0 dead-centre so it is strongly recorded
    N_pos[0] = [c_max / 2, c_max / 2]
    np.savez_compressed(os.path.join(td, 'topology.npz'), N_pos=N_pos)
    json.dump({'c_max': c_max}, open(os.path.join(td, 'topology_meta.json'), 'w'))

    # two iters with Poisson-ish spikes
    for n in range(2):
        spk_t, spk_i = [], []
        for i in range(Nn):
            times = np.sort(rng.uniform(0.2, 5.8, rng.integers(5, 25)))
            spk_t.append(times)
            spk_i.append(np.full(len(times), i))
        spk_t = np.concatenate(spk_t).astype(np.float32)
        spk_i = np.concatenate(spk_i).astype(np.int32)
        np.savez_compressed(
            os.path.join(td, 'iter_%05d.npz' % n),
            params=np.arange(36, dtype=float), theta=np.arange(10, dtype=float),
            conn_prob=np.float64(0.3),
            spk_N_t=spk_t, spk_N_i=spk_i,
            spk_A_t=np.array([]), spk_A_i=np.array([]),
            seed_run=np.int64(123 + n))

    out = os.path.join(root, 'out')
    lib, lib_path = resolve_library(os.path.join(root, 'eap_library.npz'))
    cfg = PipelineConfig(plots=True)
    walk_campaign(camp, out, lib_path, cfg, workers=1, save_traces_first=True)

    # validate output
    o0 = np.load(os.path.join(out, 'topo_00000', 'mea_iter_00000.npz'),
                 allow_pickle=False)
    ok = True
    msgs = []
    checks = [
        ('has detections', len(o0['det_t']) > 0),
        ('9 channels sigma', o0['sigma'].shape == (9,)),
        ('centers 9x2', o0['electrode_centers'].shape == (9, 2)),
        ('params length 36', len(o0['params']) == 36),
        ('some true-matched', np.any(o0['det_src_neuron'] >= 0)),
        ('centre neuron seen',
         np.any(o0['det_src_neuron'] == 0)),
        ('traces saved on iter0', 'traces' in o0.files),
    ]
    for name, c in checks:
        ok = ok and c
        msgs.append('[%s] %s' % ('PASS' if c else 'FAIL', name))
    print('\n'.join(msgs))
    print('SELF-TEST', 'PASSED' if ok else 'FAILED')
    if tmpdir is None:
        shutil.rmtree(root, ignore_errors=True)
    return 0 if ok else 1


# =============================================================================
# CLI
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--campaign', help='campaign root (contains topo_*/)')
    p.add_argument('--out', help='output root')
    p.add_argument('--library', default='eap_library.npz',
                   help='EAP template library .npz (generated if missing)')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--limit_iters', type=int, default=None)
    p.add_argument('--limit_topos', type=int, default=None)
    p.add_argument('--save_traces_first', action='store_true',
                   help='dump the raw (E,T) trace for iter 0 of each topo')
    p.add_argument('--self_test', action='store_true')
    # knob overrides
    p.add_argument('--fs', type=float, default=10110.09)
    p.add_argument('--n_dec', type=float, default=2.0)
    p.add_argument('--snr_ref', type=float, default=15.0)
    p.add_argument('--r_ref', type=float, default=30.0)
    p.add_argument('--r_min', type=float, default=10.0)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--target_noise_uv', type=float, default=5.0)
    p.add_argument('--k', type=float, default=5.0)
    p.add_argument('--band_lo', type=float, default=300.0)
    p.add_argument('--band_hi', type=float, default=3000.0)
    p.add_argument('--refractory_ms', type=float, default=2.0)
    p.add_argument('--plots', action='store_true',
                   help='save diagnostic PNGs (geometry per topo; signal '
                        'plots for the first iter of each topo)')
    p.add_argument('--plot_window_s', type=float, default=2.0,
                   help='width of the plotted time window [s]')
    p.add_argument('--plot_max_true_rows', type=int, default=120,
                   help='max neurons shown in the ground-truth raster')
    p.add_argument('--n_sub', type=int, default=4)
    p.add_argument('--pitch', type=float, default=60.0)
    p.add_argument('--edge', type=float, default=25.0)
    p.add_argument('--n_side', type=int, default=3)
    return p


def main():
    args = build_parser().parse_args()
    if args.self_test:
        raise SystemExit(self_test())
    if not args.campaign or not args.out:
        raise SystemExit('--campaign and --out are required (or use --self_test)')

    cfg = PipelineConfig(
        fs=args.fs, n_dec=args.n_dec, snr_ref=args.snr_ref, r_ref=args.r_ref,
        r_min=args.r_min, gamma=args.gamma, target_noise_uv=args.target_noise_uv,
        k=args.k, band_lo=args.band_lo, band_hi=args.band_hi,
        refractory_ms=args.refractory_ms, n_sub=args.n_sub, pitch=args.pitch,
        edge=args.edge, n_side=args.n_side,
        plots=args.plots, plot_window_s=args.plot_window_s,
        plot_max_true_rows=args.plot_max_true_rows)
    lib, lib_path = resolve_library(args.library)
    walk_campaign(args.campaign, args.out, lib_path, cfg,
                  workers=args.workers,
                  save_traces_first=args.save_traces_first,
                  limit_iters=args.limit_iters, limit_topos=args.limit_topos)


if __name__ == '__main__':
    main()
