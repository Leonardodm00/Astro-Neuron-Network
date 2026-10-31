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
    astro_rate_hz from the sidecar (Full mode; whole run, kept for reference)
  Full mode (job_args mode Full, Na > 0), from the astrocyte events spk_A_t /
  spk_A_i (threshold crossings C > C_osc of the astrocyte group) on the same
  window, Na from job_args (density-resolved):
    astro_rate_win_hz  events in the window / (Na * (T - t_skip))
    astro_frac_active  astrocytes with at least one event in the window
    stoa_gate_p        the network's contact gate, from the sidecar
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
bursting = li_burst_rate_per_min > 0 (reporting conventions, not targets);
in Full mode also y_astro = log10(astro_rate_win_hz + astro_rate_floor_hz) and
astro_silent = astro_frac_active < silent_frac_active:
    corner_effects.csv       per design dimension j: main effect
                             E_j(f) = mean(f | code_j = +1) - mean(f | code_j = -1)
                             over the corner runs, for f = y, silent, high, bursting
                             (and y_astro, astro_silent in Full mode)
    corner_interactions.csv  per pair (j, k): E_jk(y) = mean(y | code_j code_k = +1)
                             - mean(y | code_j code_k = -1)
    interior_bins.csv        per dimension and bin: n, rate quartiles, shares
    interior_pairs.csv       per pair of dimensions and pair of bins: n, shares,
                             median y
    interior_spearman.csv    Spearman rho of each dimension (theta; conn_prob)
                             with y and with the burst rate (and y_astro)
    gate_networks.csv        Full mode with a gate: one row per network (level
                             folder, topology): its gate p, conn_prob, runs, and
                             the median y (y_astro) of its runs minus the median
                             over the networks of its folder (dy, dy_astro).
                             The gate is drawn per topology, not designed, so its
                             effect is read across networks: Spearman rho of p
                             with dy and dy_astro (the folder median removes the
                             conn_prob level shared within a folder)
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
           rate_floor_hz=0.01, astro_rate_floor_hz=0.001, majority=0.5, n_top=20)
ASTRO = ('y_astro', 'astro_silent')          # outcomes added in Full mode
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
    na = int(job.get('Na') or 0) if job.get('mode') == 'Full' else 0
    recs = []
    for js in sorted(glob.glob(os.path.join(pdir, 'topo_*', 'iter_*.json'))):
        side = json.load(open(js))
        recs.append(dict(phase_dir=pdir, topo_idx=side['topo_idx'], iter_idx=side['iter_idx'],
                         conn_prob=side.get('conn_prob', np.nan), theta=side['theta'],
                         params=side['params'], param_names=side['param_names'],
                         npz=os.path.join(pdir, side['npz_relpath']),
                         Nn=int(job['Nn']), Na=na, simtime=float(job['simtime']),
                         astro_rate_hz=side.get('mean_astro_rate_Hz'),
                         stoa_gate_p=side.get('stoa_gate_p', np.nan)))
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


def astro_window_metrics(spk_t, spk_i, n_astro, t_total, cfg=CFG):
    """Astrocyte events of one run on the window [t_skip, t_total]."""
    t_rec = float(t_total) - cfg['t_skip']
    keep = np.asarray(spk_t, dtype=np.float64) >= cfg['t_skip']
    i = np.asarray(spk_i)[keep].astype(np.int64)
    counts = np.bincount(i, minlength=n_astro)[:n_astro]
    return dict(astro_rate_win_hz=float(keep.sum()) / (n_astro * t_rec),
                astro_frac_active=float(np.mean(counts > 0)),
                n_astro_events_window=int(keep.sum()))


def _metrics_job(args):
    rec, bm_path, cfg = args
    bm = import_burst_metrics(bm_path)
    z = np.load(rec['npz'])
    out = window_metrics(z['spk_N_t'], z['spk_N_i'], rec['Nn'], rec['simtime'], bm, cfg)
    if rec['Na'] > 0 and 'spk_A_t' in z.files:
        out.update(astro_window_metrics(z['spk_A_t'], z['spk_A_i'], rec['Na'], rec['simtime'], cfg))
    return out


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
                   kind=kind, conn_prob=r['conn_prob'], stoa_gate_p=r['stoa_gate_p'], topo_idx=r['topo_idx'],
                   iter_idx=r['iter_idx'], astro_rate_hz=r['astro_rate_hz'], Nn=r['Nn'], Na=r['Na'],
                   simtime=r['simtime'])
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
    if has_astro(df):
        df['y_astro'] = np.log10(df.astro_rate_win_hz + cfg['astro_rate_floor_hz'])
        df['astro_silent'] = (df.astro_frac_active < cfg['silent_frac_active']).astype(float)
        df.loc[df.astro_rate_win_hz.isna(), ['y_astro', 'astro_silent']] = np.nan
    return df


def has_astro(df):
    """Whether any run of the table carries astrocyte metrics (Full mode)."""
    return 'astro_rate_win_hz' in df.columns and bool(df.astro_rate_win_hz.notna().any())


def dimensions(df, prefix):
    return [c[len(prefix):] for c in df.columns if c.startswith(prefix)]


# ---------------------------------------------------------------------------
# corners
# ---------------------------------------------------------------------------
def corner_effects(c, dims):
    rows = []
    outcomes = ('y', 'silent', 'high', 'bursting') + (ASTRO if has_astro(c) else ())
    for j in dims:
        x = c['code_' + j]
        plus, minus = c[x == 1], c[x == -1]
        row = dict(dimension=j, n_plus=len(plus), n_minus=len(minus))
        for f in outcomes:
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
    if has_astro(c):
        q = np.nanpercentile(c.astro_rate_win_hz, [25, 50, 75])
        lines.append('  astrocytes over the corners: astro-silent (active fraction < %g) %.1f %%; event rate per '
                     'astrocyte [Hz] quartiles %.3g / %.3g / %.3g'
                     % (cfg['silent_frac_active'], 100 * c.astro_silent.mean(), q[0], q[1], q[2]))
        if len(ctr):
            lines.append('  astro curvature: mean log10 astro rate at the centre %.3f, over the corners %.3f'
                         % (ctr.y_astro.mean(), c.y_astro.mean()))
        lines.append('  main effects on the astrocytes, by |effect| on log10 astro rate:')
        lines.append('    %-12s %12s %13s' % ('dimension', 'log10astro', 'astro-silent'))
        for _, r in eff.reindex(eff.effect_y_astro.abs().sort_values(ascending=False).index).iterrows():
            lines.append('    %-12s %12.3f %13.3f' % (r.dimension, r.effect_y_astro, r.effect_astro_silent))
    return '\n'.join(lines) + '\n', eff, inter


# ---------------------------------------------------------------------------
# interior
# ---------------------------------------------------------------------------
def interior_bins(it, dims):
    rows = []
    astro = has_astro(it)
    for j in dims:
        for i, g in it.groupby('bin_' + j):
            q = np.percentile(g.mean_rate_hz, [25, 50, 75])
            row = dict(dimension=j, bin=int(i), n=len(g), rate_q25=q[0], rate_median=q[1],
                       rate_q75=q[2], share_silent=g.silent.mean(), share_high=g.high.mean(),
                       share_bursting=g.bursting.mean(),
                       burst_rate_median=g.li_burst_rate_per_min.fillna(0.0).median())
            if astro:
                qa = np.nanpercentile(g.astro_rate_win_hz, [25, 50, 75])
                row.update(astro_q25=qa[0], astro_median=qa[1], astro_q75=qa[2],
                           share_astro_silent=g.astro_silent.mean())
            rows.append(row)
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
    astro = has_astro(it)
    for j in dims:
        x = it.conn_prob.values if j == 'conn_prob' else it['theta_' + j].values
        if np.ptp(x) == 0:
            continue
        rho_r, pv_r = stats.spearmanr(x, it.y.values)
        rho_b, pv_b = stats.spearmanr(x, yb)
        row = dict(dimension=j, rho_log_rate=rho_r, p_log_rate=pv_r, rho_burst=rho_b, p_burst=pv_b)
        if astro:
            ok = np.isfinite(it.y_astro.values)
            ya = it.y_astro.values[ok]
            if ok.sum() > 2 and np.ptp(ya) > 0 and np.ptp(x[ok]) > 0:
                row['rho_log_astro'], row['p_log_astro'] = stats.spearmanr(x[ok], ya)
            else:
                row['rho_log_astro'], row['p_log_astro'] = np.nan, np.nan
        rows.append(row)
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
    if has_astro(it):
        q = np.nanpercentile(it.astro_rate_win_hz, [25, 50, 75])
        lines.append('  astrocytes: astro-silent %.1f %%; event rate per astrocyte [Hz] quartiles %.3g / %.3g / %.3g'
                     % (100 * it.astro_silent.mean(), q[0], q[1], q[2]))
        lines.append('  bins where more than %d %% of the runs are astro-silent (A):' % (100 * cfg['majority']))
        for j in dims:
            b = bins_t[bins_t.dimension == j]
            flags = ['%dA' % r.bin for _, r in b.iterrows() if r.share_astro_silent > cfg['majority']]
            lines.append('    %-12s %s' % (j, ' '.join(flags) if flags else '-'))
        lines.append('  Spearman with log10 astro rate, by |rho|:')
        for _, r in sp.reindex(sp.rho_log_astro.abs().sort_values(ascending=False).index).iterrows():
            lines.append('    %-12s %7.3f' % (r.dimension, r.rho_log_astro))
    return '\n'.join(lines) + '\n', bins_t, pairs_t, sp


# ---------------------------------------------------------------------------
# the contact gate, per network
# ---------------------------------------------------------------------------
def gate_networks(df):
    """One row per network (level folder, topology) of the runs that carry a
    gate; dy (dy_astro): the network's median y (y_astro) minus the median of
    those over the networks of its folder."""
    g = df[np.isfinite(df.stoa_gate_p.astype(float))] if 'stoa_gate_p' in df.columns else df.iloc[:0]
    if g.empty:
        return pd.DataFrame()
    agg = dict(stoa_gate_p=('stoa_gate_p', 'first'), conn_prob=('conn_prob', 'first'), n_runs=('y', 'size'),
               y_median=('y', 'median'))
    if has_astro(g):
        agg.update(y_astro_median=('y_astro', 'median'), share_astro_silent=('astro_silent', 'mean'))
    net = g.groupby(['phase_name', 'topo_idx']).agg(**agg).reset_index()
    net['dy'] = net.y_median - net.groupby('phase_name').y_median.transform('median')
    if 'y_astro_median' in net.columns:
        net['dy_astro'] = net.y_astro_median - net.groupby('phase_name').y_astro_median.transform('median')
    return net


def gate_text(net):
    if net.empty:
        return ''
    lines = ['contact gate: %d networks with a gate, p from %.3f to %.3f (drawn per topology, not designed)'
             % (len(net), net.stoa_gate_p.min(), net.stoa_gate_p.max())]
    for col, label in (('dy', 'log10 rate'), ('dy_astro', 'log10 astro rate')):
        if col in net.columns:
            ok = np.isfinite(net[col].values)
            if ok.sum() > 2 and np.ptp(net[col].values[ok]) > 0 and np.ptp(net.stoa_gate_p.values[ok]) > 0:
                rho, pv = stats.spearmanr(net.stoa_gate_p.values[ok], net[col].values[ok])
                lines.append('  Spearman of p with the network median %s (folder median removed): rho %.3f, '
                             'p-value %.3g, %d networks' % (label, rho, pv, int(ok.sum())))
            else:
                lines.append('  Spearman of p with the network median %s: too few networks' % label)
    return '\n'.join(lines) + '\n'


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
    net = gate_networks(df)
    tables = [('corner_effects.csv', eff), ('corner_interactions.csv', inter),
              ('interior_bins.csv', bins_t), ('interior_pairs.csv', pairs_t),
              ('interior_spearman.csv', sp)] + ([('gate_networks.csv', net)] if not net.empty else [])
    for name, t in tables:
        t.to_csv(os.path.join(a.root, name), index=False)
    counts = df.groupby('phase_name').size()
    head = ('range_analysis  root %s\nwindow [%g s, T]; burst metrics from %s\nfolders: %s\n'
            '(silent: active fraction < %g; high: mean rate > %g Hz; reporting conventions, not targets)\n'
            % (a.root, cfg['t_skip'], bm, ', '.join('%s (%d runs)' % kv for kv in counts.items()),
               cfg['silent_frac_active'], cfg['high_rate_hz']))
    if has_astro(df):
        head += ('astrocytes: events = threshold crossings C > C_osc in the window; astro-silent: fewer than %g of '
                 'the astrocytes with an event; log10(astro rate + %g)\n'
                 % (cfg['silent_frac_active'], cfg['astro_rate_floor_hz']))
    text = head + c_text + i_text + gate_text(net)
    with open(os.path.join(a.root, 'calibration_summary.txt'), 'w') as fh:
        fh.write(text)
    print(text)
    print('wrote runs.csv (%d runs), %s, calibration_summary.txt in %s'
          % (len(df), ', '.join(n for n, _ in tables), a.root))
    unmatched = int((df.kind == 'unmatched').sum())
    if unmatched:
        print('WARNING: %d runs matched no design value' % unmatched)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
