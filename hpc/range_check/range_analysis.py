#!/usr/bin/env python3
"""
range_analysis.py -- per-run metrics and the calibration summaries of a
range_check.py output folder (specs/SPEC.md Block 6). Computes nothing it can
take from the repository: the network-burst panel comes from
burst_metrics.compute_all, the rank correlations from scipy.stats.spearmanr.

Per run (topo_*/iter_*.npz + the .json sidecar of each level folder):
    window       neuron spikes with t >= t_skip, shifted to start at 0
    mean_rate_hz spikes in the window / (Nn * (T - t_skip))
    frac_active  neurons with a window rate > active_rate_hz
    burst_rate   burst_metrics' log-ISI network bursts per minute
                 (li_burst_rate_per_min; also pr_burst_rate_per_min)
    cv_isi_mean, adapt_index_mean, li_burst_dur_mean_s,
    li_frac_spikes_outside_burst  from the same panel
    astro_rate_hz from the sidecar (Full mode)
Labels come from each run's stored theta and its folder's design.json, so they
do not depend on the order in which the runs were drawn:
    corners_c<L>   every free axis at its lower (code -1) or upper (+1) bound:
                   'corner'; every free axis at the box centre: 'centre';
                   conn_prob code -1 for L = 0, +1 for L = 1
    interior_b<L>  every free axis at a bin midpoint: 'interior', with its bin;
                   conn_prob bin = L
    else           'unmatched' (exit status 1)

Summaries, with y = log10(mean_rate_hz + rate_floor_hz) and the indicators
silent = frac_active < silent_frac_active, high = mean_rate_hz > high_rate_hz,
bursting = li_burst_rate_per_min > 0 (reporting conventions, not targets):
    corner_effects.csv       per design dimension j: main effect
                             E_j(f) = mean(f | code_j = +1) - mean(f | code_j = -1)
                             over the corner runs, for f = y, silent, high, bursting
    corner_interactions.csv  per pair (j, k): E_jk(y) = mean(y | code_j code_k = +1)
                             - mean(y | code_j code_k = -1)
    interior_bins.csv        per dimension and bin: n, rate quartiles, shares
    interior_pairs.csv       per pair of dimensions and pair of bins: n, shares,
                             median y
    interior_spearman.csv    Spearman rho of each dimension (theta; conn_prob)
                             with y and with the burst rate
    calibration_summary.txt  the above in short, plus the centre-point curvature
                             (mean y at the centre minus mean y over the corners)
and runs.csv, one row per run.

Usage:  python3 range_analysis.py OUT [--t-skip 2] [--workers N] [--burst-lib DIR]
"""
import argparse
import glob
import hashlib
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import range_design as RD   # noqa: E402

CFG = dict(t_skip=2.0, active_rate_hz=0.1, silent_frac_active=0.1, high_rate_hz=20.0,
           rate_floor_hz=0.01, majority=0.5, n_top=20)
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
    """One dict per finished run of a level folder (sidecar fields + paths)."""
    design = json.load(open(os.path.join(pdir, 'design.json')))
    job = json.load(open(os.path.join(pdir, 'job_args.json')))
    recs = []
    for js in sorted(glob.glob(os.path.join(pdir, 'topo_*', 'iter_*.json'))):
        side = json.load(open(js))
        recs.append(dict(phase_dir=pdir, topo_idx=side['topo_idx'], iter_idx=side['iter_idx'],
                         conn_prob=side.get('conn_prob', np.nan), theta=side['theta'],
                         params=side['params'], param_names=side['param_names'],
                         npz=os.path.join(pdir, side['npz_relpath']),
                         Nn=int(job['Nn']), simtime=float(job['simtime']),
                         astro_rate_hz=side.get('mean_astro_rate_Hz')))
    return design, recs


def label_run(theta, param_names, design):
    """(kind, per-free-axis codes or bins): see the module docstring."""
    th = np.asarray(theta, dtype=np.float64)
    vals = [th[param_names.index(n)] for n in design['free_axes']]
    lo, hi = design['theta_lo'], design['theta_hi']
    if design['phase'] == 'corners':
        codes = [RD.corner_code(v, a, b) for v, a, b in zip(vals, lo, hi)]
        if all(c in (-1, 1) for c in codes):
            return 'corner', codes
        if all(c == 0 for c in codes):
            return 'centre', codes
        return 'unmatched', codes
    bins = [RD.bin_of(v, a, b, int(design['bins'])) for v, a, b in zip(vals, lo, hi)]
    return ('interior' if all(i >= 0 for i in bins) else 'unmatched'), bins


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
    return window_metrics(z['spk_N_t'], z['spk_N_i'], rec['Nn'], rec['simtime'], bm, cfg)


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
        kind, lab = label_run(r['theta'], r['param_names'], d)
        row = dict(phase_name=os.path.basename(r['phase_dir']), phase=d['phase'], level=d['level'],
                   kind=kind, conn_prob=r['conn_prob'], topo_idx=r['topo_idx'], iter_idx=r['iter_idx'],
                   astro_rate_hz=r['astro_rate_hz'], Nn=r['Nn'], simtime=r['simtime'])
        if d['has_conn']:
            row['code_conn_prob' if d['phase'] == 'corners' else 'bin_conn_prob'] = (
                (-1 if d['level'] == 0 else 1) if d['phase'] == 'corners' else d['level'])
        row.update(m)
        for name, v in zip(d['free_axes'], lab):
            row[('code_' if d['phase'] == 'corners' else 'bin_') + name] = v
            row['theta_' + name] = r['theta'][r['param_names'].index(name)]
            row['param_' + name] = r['params'][r['param_names'].index(name)]
        rows.append(row)
    sha = hashlib.sha256(open(bm_path, 'rb').read()).hexdigest()[:12]
    return pd.DataFrame(rows), designs, '%s (sha256 %s)' % (bm_path, sha)


def add_outcomes(df, cfg=CFG):
    df = df.copy()
    df['y'] = np.log10(df.mean_rate_hz + cfg['rate_floor_hz'])
    df['silent'] = (df.frac_active < cfg['silent_frac_active']).astype(float)
    df['high'] = (df.mean_rate_hz > cfg['high_rate_hz']).astype(float)
    df['bursting'] = (df.li_burst_rate_per_min.fillna(0.0) > 0).astype(float)
    return df


def dimensions(df, prefix):
    return [c[len(prefix):] for c in df.columns if c.startswith(prefix)]


# ---------------------------------------------------------------------------
# corners
# ---------------------------------------------------------------------------
def corner_effects(c, dims):
    rows = []
    for j in dims:
        x = c['code_' + j]
        plus, minus = c[x == 1], c[x == -1]
        row = dict(dimension=j, n_plus=len(plus), n_minus=len(minus))
        for f in ('y', 'silent', 'high', 'bursting'):
            row['effect_' + f] = (plus[f].mean() - minus[f].mean()) if len(plus) and len(minus) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.reindex(out.effect_y.abs().sort_values(ascending=False).index) if len(out) else out


def corner_interactions(c, dims):
    rows = []
    for j, k in itertools.combinations(dims, 2):
        x = c['code_' + j] * c['code_' + k]
        plus, minus = c[x == 1], c[x == -1]
        rows.append(dict(dimension_1=j, dimension_2=k, n_plus=len(plus), n_minus=len(minus),
                         effect_y=(plus.y.mean() - minus.y.mean()) if len(plus) and len(minus) else np.nan))
    out = pd.DataFrame(rows)
    return out.reindex(out.effect_y.abs().sort_values(ascending=False).index) if len(out) else out


def corner_text(df, designs, cfg=CFG):
    c, ctr = df[df.kind == 'corner'], df[df.kind == 'centre']
    cdes = [d for d in designs.values() if d['phase'] == 'corners']
    if c.empty:
        return 'corners: no corner runs\n', pd.DataFrame(), pd.DataFrame()
    info = cdes[0]['corner_design']
    planned = info['n_runs']
    dims = dimensions(c, 'code_')
    eff, inter = corner_effects(c, dims), corner_interactions(c, dims)
    kind = ('full factorial of %d dimensions' % info['d'] if info['kind'] == 'full' else
            'two-level fraction of %d dimensions, resolution %d' % (info['d'], info['resolution']))
    distinct = len({tuple(r) for r in c[['code_' + j for j in dims]].astype(int).values})
    lines = ['corners: %d corner runs covering %d of the %d corners of the design (%s), %d centre points'
             % (len(c), distinct, planned, kind, len(ctr)),
             '  shares over the corners: silent %.1f %%, bursting %.1f %%, > %g Hz %.1f %%'
             % (100 * c.silent.mean(), 100 * c.bursting.mean(), cfg['high_rate_hz'], 100 * c.high.mean())]
    if len(ctr):
        lines.append('  curvature: mean log10 rate at the centre %.3f, over the corners %.3f, difference %.3f'
                     % (ctr.y.mean(), c.y.mean(), ctr.y.mean() - c.y.mean()))
    lines.append('  main effects (upper bound minus lower bound), by |effect| on log10 rate:')
    lines.append('    %-12s %9s %9s %9s %9s %11s' % ('dimension', 'log10rate', 'silent', '>%gHz' % cfg['high_rate_hz'],
                                                   'bursting', 'n(+)/n(-)'))
    for _, r in eff.iterrows():
        lines.append('    %-12s %9.3f %9.3f %9.3f %9.3f %5d/%-5d' % (r.dimension, r.effect_y, r.effect_silent,
                                                                     r.effect_high, r.effect_bursting,
                                                                     r.n_plus, r.n_minus))
    lines.append('  largest two-dimension interactions on log10 rate:')
    for _, r in inter.head(cfg['n_top']).iterrows():
        lines.append('    %-12s x %-12s %8.3f' % (r.dimension_1, r.dimension_2, r.effect_y))
    for label, part in (('lowest', c.nsmallest(5, 'mean_rate_hz')), ('highest', c.nlargest(5, 'mean_rate_hz'))):
        lines.append('  %s-rate corners (rate [Hz]: dimensions at their upper bound):' % label)
        for _, r in part.iterrows():
            ups = [j for j in dims if r['code_' + j] == 1]
            lines.append('    %10.3g: %s' % (r.mean_rate_hz, ', '.join(ups) if ups else '(all at the lower bound)'))
    return '\n'.join(lines) + '\n', eff, inter


# ---------------------------------------------------------------------------
# interior
# ---------------------------------------------------------------------------
def interior_bins(it, dims):
    rows = []
    for j in dims:
        for i, g in it.groupby('bin_' + j):
            q = np.percentile(g.mean_rate_hz, [25, 50, 75])
            rows.append(dict(dimension=j, bin=int(i), n=len(g), rate_q25=q[0], rate_median=q[1],
                             rate_q75=q[2], share_silent=g.silent.mean(), share_high=g.high.mean(),
                             share_bursting=g.bursting.mean(),
                             burst_rate_median=g.li_burst_rate_per_min.fillna(0.0).median()))
    return pd.DataFrame(rows)


def interior_pairs(it, dims):
    rows = []
    for j, k in itertools.combinations(dims, 2):
        for (a, b), g in it.groupby(['bin_' + j, 'bin_' + k]):
            rows.append(dict(dimension_1=j, dimension_2=k, bin_1=int(a), bin_2=int(b), n=len(g),
                             share_silent=g.silent.mean(), share_high=g.high.mean(), y_median=g.y.median()))
    return pd.DataFrame(rows)


def interior_spearman(it, dims):
    rows = []
    yb = it.li_burst_rate_per_min.fillna(0.0).values
    for j in dims:
        x = it.conn_prob.values if j == 'conn_prob' else it['theta_' + j].values
        if np.ptp(x) == 0:
            continue
        rho_r, pv_r = stats.spearmanr(x, it.y.values)
        rho_b, pv_b = stats.spearmanr(x, yb)
        rows.append(dict(dimension=j, rho_log_rate=rho_r, p_log_rate=pv_r, rho_burst=rho_b, p_burst=pv_b))
    sp = pd.DataFrame(rows)
    return sp.reindex(sp.rho_log_rate.abs().sort_values(ascending=False).index) if len(sp) else sp


def interior_text(df, designs, cfg=CFG):
    it = df[df.kind == 'interior']
    if it.empty:
        return 'interior: no interior runs\n', pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ides = [d for d in designs.values() if d['phase'] == 'interior']
    dims = dimensions(it, 'bin_')
    bins_t, pairs_t, sp = interior_bins(it, dims), interior_pairs(it, dims), interior_spearman(it, dims)
    lines = ['interior: %d cells in %d folders, %d bins per dimension' % (len(it), len(ides), ides[0]['bins']),
             '  shares: silent %.1f %%, bursting %.1f %%, > %g Hz %.1f %%'
             % (100 * it.silent.mean(), 100 * it.bursting.mean(), cfg['high_rate_hz'], 100 * it.high.mean()),
             '  bins where more than %d %% of the runs are silent (S) or above %g Hz (H):'
             % (100 * cfg['majority'], cfg['high_rate_hz'])]
    for j in dims:
        b = bins_t[bins_t.dimension == j]
        flags = ['%d%s' % (r.bin, ('S' if r.share_silent > cfg['majority'] else '')
                           + ('H' if r.share_high > cfg['majority'] else ''))
                 for _, r in b.iterrows() if r.share_silent > cfg['majority'] or r.share_high > cfg['majority']]
        lines.append('    %-12s %s' % (j, ' '.join(flags) if flags else '-'))
    lines.append('  Spearman with log10 rate, by |rho|:')
    for _, r in sp.iterrows():
        lines.append('    %-12s %7.3f (burst rate %7.3f)' % (r.dimension, r.rho_log_rate, r.rho_burst))
    return '\n'.join(lines) + '\n', bins_t, pairs_t, sp


def main(argv=None):
    ap = argparse.ArgumentParser(description='Calibration summaries of a range_check.py output folder.')
    ap.add_argument('root')
    ap.add_argument('--t-skip', type=float, default=CFG['t_skip'])
    ap.add_argument('--workers', type=int, default=int(os.environ.get('PBS_NCPUS', '1')))
    ap.add_argument('--burst-lib', default=None)
    a = ap.parse_args(argv)
    cfg = dict(CFG, t_skip=a.t_skip)
    df, designs, bm = build_table(a.root, a.workers, a.burst_lib, cfg)
    df = add_outcomes(df, cfg)
    df.to_csv(os.path.join(a.root, 'runs.csv'), index=False)
    c_text, eff, inter = corner_text(df, designs, cfg)
    i_text, bins_t, pairs_t, sp = interior_text(df, designs, cfg)
    for name, t in (('corner_effects.csv', eff), ('corner_interactions.csv', inter),
                    ('interior_bins.csv', bins_t), ('interior_pairs.csv', pairs_t),
                    ('interior_spearman.csv', sp)):
        t.to_csv(os.path.join(a.root, name), index=False)
    counts = df.groupby('phase_name').size()
    head = ('range_analysis  root %s\nwindow [%g s, T]; burst metrics from %s\nfolders: %s\n'
            '(silent: active fraction < %g; high: mean rate > %g Hz; reporting conventions, not targets)\n'
            % (a.root, cfg['t_skip'], bm, ', '.join('%s (%d runs)' % kv for kv in counts.items()),
               cfg['silent_frac_active'], cfg['high_rate_hz']))
    text = head + c_text + i_text
    with open(os.path.join(a.root, 'calibration_summary.txt'), 'w') as fh:
        fh.write(text)
    print(text)
    print('wrote runs.csv (%d runs), corner_effects.csv, corner_interactions.csv, interior_bins.csv, '
          'interior_pairs.csv, interior_spearman.csv, calibration_summary.txt in %s' % (len(df), a.root))
    unmatched = int((df.kind == 'unmatched').sum())
    if unmatched:
        print('WARNING: %d runs matched no design value' % unmatched)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
