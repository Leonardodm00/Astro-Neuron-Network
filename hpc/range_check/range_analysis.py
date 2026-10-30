#!/usr/bin/env python3
"""
range_analysis.py -- per-run metrics and summaries of a range_check.py output
folder. Standalone analysis tool; spec: specs/SPEC.md, Block 6. Computes
nothing it can take from the repository: the network-burst panel comes from
burst_metrics.compute_all, the rank correlations from scipy.stats.spearmanr.

Per run (topo_*/iter_*.npz + the .json sidecar of each phase folder):
    window       neuron spikes with t >= t_skip, shifted to start at 0
    mean_rate_hz spikes in the window / (Nn * (T - t_skip))
    frac_active  neurons with a window rate > active_rate_hz
    burst_rate   burst_metrics' log-ISI network bursts per minute
                 (li_burst_rate_per_min; also pr_burst_rate_per_min)
    cv_isi_mean, adapt_index_mean, li_burst_dur_mean_s,
    li_frac_spikes_outside_burst  from the same panel
    astro_rate_hz from the sidecar (Full mode)
Each oat run is labelled by its position in the design: range_check.py feeds
the padded design to every topology in order, so iter_idx < n_design is design
point iter_idx and the rest are the nominal padding; the stored theta must equal
that vector exactly, or the run is 'unmatched'. (Labelling by position keeps a
design point that coincides with the nominal vector -- an axis whose nominal sits
on a grid value -- apart from the nominal runs.)

Writes into the output root: runs.csv, oat_summary.csv, prior_summary.txt,
prior_spearman.csv. The cut-offs below are reporting conventions, not targets.

Usage:  python3 range_analysis.py OUT [--t-skip 2] [--workers N] [--burst-lib DIR]
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

CFG = dict(t_skip=2.0, active_rate_hz=0.1, silent_frac_active=0.1, high_rate_hz=20.0,
           rate_floor_hz=0.01)
HERE = os.path.dirname(os.path.abspath(__file__))
_BM = {}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def find_burst_metrics(burst_lib=None, phase_lib_dirs=()):
    """burst_metrics.py: --burst-lib, else the repository copy next to this
    folder, else a setup folder the runs came from. Returns its path."""
    for d in ([burst_lib] if burst_lib else []) + [os.path.join(HERE, '..', 'Phenomenological_finalv1')] \
            + list(phase_lib_dirs):
        p = os.path.join(d, 'burst_metrics.py')
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise SystemExit('ERROR: burst_metrics.py not found (give --burst-lib DIR)')


def import_burst_metrics(path):
    if 'mod' not in _BM:
        sys.path.insert(0, os.path.dirname(path))
        import burst_metrics
        _BM['mod'] = burst_metrics
    return _BM['mod']


def phase_dirs(root):
    """Subfolders of root written by range_check.py (they hold design.json)."""
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, '*', 'design.json')))


def run_records(pdir):
    """One dict per finished run of a phase folder (sidecar fields + paths)."""
    design = json.load(open(os.path.join(pdir, 'design.json')))
    job = json.load(open(os.path.join(pdir, 'job_args.json')))
    recs = []
    for js in sorted(glob.glob(os.path.join(pdir, 'topo_*', 'iter_*.json'))):
        side = json.load(open(js))
        npz = os.path.join(pdir, side['npz_relpath'])
        recs.append(dict(phase_dir=pdir, phase=design['phase'], conn_level=design.get('conn_prob'),
                         topo_idx=side['topo_idx'], iter_idx=side['iter_idx'],
                         conn_prob=side.get('conn_prob', np.nan), theta=side['theta'],
                         params=side['params'], param_names=side['param_names'], npz=npz,
                         Nn=int(job['Nn']), simtime=float(job['simtime']),
                         astro_rate_hz=side.get('mean_astro_rate_Hz')))
    return design, recs


def label_oat(theta, iter_idx, design):
    """(axis, natural value) of oat run iter_idx: design point iter_idx if
    iter_idx < n_design, ('nominal', nan) for the padding, provided the stored
    theta equals that vector exactly; ('unmatched', nan) otherwise."""
    th = np.asarray(theta, dtype=np.float64)
    n_design = int(design['n_design'])
    if iter_idx < n_design:
        p = design['points'][iter_idx]
        if np.array_equal(th, np.asarray(p['theta_vector'], dtype=np.float64)):
            return p['axis'], p['natural']
    elif iter_idx < n_design + int(design['n_padding']) and \
            np.array_equal(th, np.asarray(design['nominal_theta'], dtype=np.float64)):
        return 'nominal', np.nan
    return 'unmatched', np.nan


# ---------------------------------------------------------------------------
# per-run metrics
# ---------------------------------------------------------------------------
def window_metrics(spk_t, spk_i, n_neurons, t_total, bm, cfg=CFG):
    """Metrics of one run on the window [t_skip, t_total] (times shifted to 0)."""
    t_rec = float(t_total) - cfg['t_skip']
    keep = np.asarray(spk_t, dtype=np.float64) >= cfg['t_skip']
    t = np.asarray(spk_t, dtype=np.float64)[keep] - cfg['t_skip']
    i = np.asarray(spk_i)[keep]
    rates = np.bincount(i.astype(np.int64), minlength=n_neurons)[:n_neurons] / t_rec
    row = bm.scalar_row(bm.compute_all(t, i, n_neurons, t_rec))
    out = dict(mean_rate_hz=float(len(t)) / (n_neurons * t_rec),
               frac_active=float(np.mean(rates > cfg['active_rate_hz'])),
               n_spikes_window=int(len(t)))
    for k in ('li_burst_rate_per_min', 'pr_burst_rate_per_min', 'li_burst_dur_mean_s',
              'li_frac_spikes_outside_burst', 'cv_isi_mean', 'adapt_index_mean'):
        out[k] = float(row.get(k, np.nan))
    return out


def _metrics_job(args):
    rec, bm_path, cfg = args
    bm = import_burst_metrics(bm_path)
    z = np.load(rec['npz'])
    m = window_metrics(z['spk_N_t'], z['spk_N_i'], rec['Nn'], rec['simtime'], bm, cfg)
    return m


def build_table(root, workers=1, burst_lib=None, cfg=CFG):
    designs, recs = {}, []
    for pdir in phase_dirs(root):
        d, r = run_records(pdir)
        designs[pdir] = d
        recs += r
    if not recs:
        raise SystemExit('ERROR: no runs under %s' % root)
    bm_path = find_burst_metrics(burst_lib, sorted({d['lib_dir'] for d in designs.values()}))
    jobs = [(r, bm_path, cfg) for r in recs]
    if workers > 1:
        with Pool(workers) as pool:
            mets = pool.map(_metrics_job, jobs, chunksize=4)
    else:
        mets = [_metrics_job(j) for j in jobs]
    rows = []
    for r, m in zip(recs, mets):
        d = designs[r['phase_dir']]
        axis, value = label_oat(r['theta'], r['iter_idx'], d) if r['phase'] == 'oat' else ('prior', np.nan)
        row = dict(phase_name=os.path.basename(r['phase_dir']), phase=r['phase'],
                   conn_level=r['conn_level'], conn_prob=r['conn_prob'],
                   topo_idx=r['topo_idx'], iter_idx=r['iter_idx'], axis=axis, value=value,
                   astro_rate_hz=r['astro_rate_hz'], Nn=r['Nn'], simtime=r['simtime'])
        row.update(m)
        for name in d['free_axes']:
            row['theta_' + name] = r['theta'][r['param_names'].index(name)]
            row['param_' + name] = r['params'][r['param_names'].index(name)]
        rows.append(row)
    sha = hashlib.sha256(open(bm_path, 'rb').read()).hexdigest()[:12]
    return pd.DataFrame(rows), designs, '%s (sha256 %s)' % (bm_path, sha)


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def oat_summary(df):
    o = df[df.phase == 'oat']
    if o.empty:
        return pd.DataFrame()
    g = o.groupby(['conn_level', 'axis', 'value'], dropna=False)
    out = g.agg(n=('mean_rate_hz', 'size'),
                rate_median=('mean_rate_hz', 'median'), rate_min=('mean_rate_hz', 'min'),
                rate_max=('mean_rate_hz', 'max'),
                burst_median=('li_burst_rate_per_min', 'median'),
                active_median=('frac_active', 'median')).reset_index()
    return out.sort_values(['axis', 'conn_level', 'value'])


def prior_summary(df, free, cfg=CFG):
    p = df[df.phase == 'prior']
    if p.empty:
        return 'no prior runs\n', pd.DataFrame()
    r = p.mean_rate_hz.values
    q = np.percentile(r, [5, 25, 50, 75, 95])
    lines = ['prior sample: %d runs, %d topologies' % (len(p), p.topo_idx.nunique()),
             'mean rate [Hz], percentiles 5/25/50/75/95: %s' % ' / '.join('%.3g' % x for x in q),
             'silent (active fraction < %g): %.1f %%' % (cfg['silent_frac_active'],
                                                       100 * np.mean(p.frac_active < cfg['silent_frac_active'])),
             'bursting (log-ISI burst rate > 0): %.1f %%' % (100 * np.mean(p.li_burst_rate_per_min > 0)),
             'mean rate > %g Hz: %.1f %%' % (cfg['high_rate_hz'], 100 * np.mean(r > cfg['high_rate_hz'])),
             '(cut-offs are reporting conventions, not targets)']
    y_rate = np.log10(r + cfg['rate_floor_hz'])
    y_burst = p.li_burst_rate_per_min.fillna(0.0).values
    rows = []
    for name in list(free) + ['conn_prob']:
        x = p['theta_' + name].values if name != 'conn_prob' else p.conn_prob.values
        if np.ptp(x) == 0:
            continue
        rho_r, pv_r = stats.spearmanr(x, y_rate)
        rho_b, pv_b = stats.spearmanr(x, y_burst)
        rows.append(dict(axis=name, rho_log_rate=rho_r, p_log_rate=pv_r, rho_burst=rho_b, p_burst=pv_b))
    sp = pd.DataFrame(rows)
    if not sp.empty:
        sp = sp.reindex(sp.rho_log_rate.abs().sort_values(ascending=False).index)
    return '\n'.join(lines) + '\n', sp


def main(argv=None):
    ap = argparse.ArgumentParser(description='Summaries of a range_check.py output folder.')
    ap.add_argument('root')
    ap.add_argument('--t-skip', type=float, default=CFG['t_skip'])
    ap.add_argument('--workers', type=int, default=int(os.environ.get('PBS_NCPUS', '1')))
    ap.add_argument('--burst-lib', default=None)
    a = ap.parse_args(argv)
    cfg = dict(CFG, t_skip=a.t_skip)
    df, designs, bm = build_table(a.root, a.workers, a.burst_lib, cfg)
    free = sorted({n for d in designs.values() for n in d['free_axes']})
    df.to_csv(os.path.join(a.root, 'runs.csv'), index=False)
    oat = oat_summary(df)
    oat.to_csv(os.path.join(a.root, 'oat_summary.csv'), index=False)
    text, sp = prior_summary(df, free, cfg)
    sp.to_csv(os.path.join(a.root, 'prior_spearman.csv'), index=False)
    counts = df.groupby('phase_name').size()
    head = ('range_analysis  root %s\nwindow [%g s, T]; burst metrics from %s\nphases: %s\n'
            % (a.root, cfg['t_skip'], bm, ', '.join('%s (%d runs)' % kv for kv in counts.items())))
    with open(os.path.join(a.root, 'prior_summary.txt'), 'w') as fh:
        fh.write(head + text + ('\nSpearman, by |rho| with log10 rate:\n' + sp.to_string(index=False)
                                if not sp.empty else ''))
    unmatched = int((df.axis == 'unmatched').sum())
    print(head + text)
    if not sp.empty:
        print(sp.head(12).to_string(index=False))
    print('wrote runs.csv (%d runs), oat_summary.csv, prior_summary.txt, prior_spearman.csv in %s'
          % (len(df), a.root))
    if unmatched:
        print('WARNING: %d oat runs matched no design point' % unmatched)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
