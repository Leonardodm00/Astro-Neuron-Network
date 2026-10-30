#!/usr/bin/env python3
"""
check_refractory_blowup.py -- standalone check (not pipeline code).

Claim tested
    In the production neuron model (ASD_fun_BD_cpp.py: CAdEx, Brian2
    cpp_standalone, refractory = t_ref) dV/dt keeps integrating during the
    refractory period, while Brian2 ignores the threshold there. A neuron
    pushed above VT during refractoriness (by the noise term or by synaptic
    input) runs away exponentially, V becomes inf/NaN, and the neuron never
    spikes again. The sweep saves such runs as ok: it discards a run only when
    spike TIMES are non-finite. Clamping V during refractoriness, as Gorski et
    al. 2021 (Sec. 2) define the model -- V "remains at this value [VR] during
    a refractory period" -- removes the NaNs.

Truth chosen (fixed before running)
    The production sweep worker HPC_main_sweep._worker_entry, called
    in-process on topologies built by HPC_single_run.build_topology, with the
    LindaColloDUP settings: MODE=Neuronal, rho 1300 /mm^2 and c_max 350 um
    (Nn=159), the campaign freeze list, the 13 free axes drawn by
    sample_theta (the sweep's own prior), conn_prob ~ U[0.1, 0.6] per
    topology. Brian2's own post-run state check
    (Group._check_for_invalid_values, called by device.run after every run)
    is intercepted to record, per run, which neurons end the run with V NaN
    or |V| > 1e50.

Variants (identical seeds and draws, so the comparison is paired)
    as-is : pipeline as shipped
    clamp : ASD_fun_BD_cpp.py with ' (unless refractory)' appended to the
            dV/dt line (the only change)
    --va/--vr replace the VA/VR freeze values (e.g. the registry -48/-58.2)

Comparison (fixed before running)
    as-is : fraction of runs with >= 1 NaN neuron and of NaN neurons per run,
            reported (no pass threshold: this measures the size of the effect)
    clamp : no run ends with a NaN neuron                         -> PASS

Usage (the four runs of 2026-10-29, then the table)
    W=work; D=<patched LindaColloDUP pipeline folder>
    python3 check_refractory_blowup.py run --pipeline_dir $D --work_dir $W --label asis_vr55 \\
        --n_topo 6 --k 4 --simtime 5 --seed 2026
    python3 check_refractory_blowup.py run --pipeline_dir $D --work_dir $W --label clamp_vr55 --clamp \\
        --n_topo 6 --k 4 --simtime 5 --seed 2026
    python3 check_refractory_blowup.py run --pipeline_dir $D --work_dir $W --label asis_registry \\
        --va -48 --vr -58.2 --n_topo 6 --k 4 --simtime 5 --seed 2026
    python3 check_refractory_blowup.py run --pipeline_dir $D --work_dir $W --label meama_proxy_full \\
        --mode Full --rho 1600 --astro_frac 0.5 --group all --no_freeze --n_topo 4 --k 4 --simtime 5 --seed 2026
    python3 check_refractory_blowup.py compare $W/result_*.json
DIR must hold the patched HPC_main_sweep.py (with --freeze), HPC_single_run.py,
ASD_fun_BD_cpp.py and synapse_pdist.csv, with the unclamped dV/dt line. Needs
brian2 + a C++ compiler; each topology compiles once (~20 s), each 5 s run of
159 neurons takes ~10-50 s on one core.

Results 2026-10-29 (sandbox, main d0ced88): runs with >= 1 NaN neuron /
runs: asis_vr55 23/24, asis_registry 23/24, clamp_vr55 0/24 (PASS),
meama_proxy_full 14/16.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np

FREEZE_FIXED = 'Cm,I_inj,DeltaA,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G'
PIPE_FILES = ('HPC_main_sweep.py', 'HPC_single_run.py', 'ASD_fun_BD_cpp.py',
              'synapse_pdist.csv')
DV_LINE = re.compile(r'^([ \t]*dV/dt[ \t]*=.*?:[ \t]*volt)[ \t]*$', re.M)
DV_ANY = re.compile(r'^[ \t]*dV/dt[ \t]*=.*$', re.M)
SEEDS = dict(seed_device=50, seed_neuron=39, seed_synapse=35, seed_astro=60)


def make_copy(src, dst, clamp, allow_clamped=False):
    """Private copy of the pipeline; with clamp, V is held during t_ref.
    allow_clamped: the source must already clamp (e.g. the patched repository
    file); it is copied unchanged and run as is."""
    os.makedirs(dst, exist_ok=True)
    for f in PIPE_FILES:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    path = os.path.join(dst, 'ASD_fun_BD_cpp.py')
    text = open(path, encoding='utf-8').read()
    if allow_clamped:
        lines = DV_ANY.findall(text)
        if clamp or len(lines) != 1 or not lines[0].rstrip().endswith('(unless refractory)'):
            raise SystemExit('--allow_clamped_source needs one dV/dt line ending in '
                             '(unless refractory) and no --clamp; got %r' % lines)
        return lines[0].strip()
    hits = DV_LINE.findall(text)
    if len(hits) != 1:
        raise SystemExit('dV/dt line found %d times in %s (need 1)' % (len(hits), path))
    if 'unless refractory' in hits[0]:
        raise SystemExit('the pipeline already clamps V during t_ref: %r' % hits[0])
    if clamp:
        text = DV_LINE.sub(lambda m: m.group(1) + ' (unless refractory)', text)
        open(path, 'w', encoding='utf-8').write(text)
    lines = DV_ANY.findall(open(path, encoding='utf-8').read())
    if len(lines) != 1 or lines[0].rstrip().endswith('(unless refractory)') != bool(clamp):
        raise SystemExit('copy check failed: dV/dt lines %r (clamp=%s)' % (lines, clamp))
    return lines[0].strip()


def run_variant(a):
    pipe = os.path.join(a.work_dir, 'pipe_' + a.label)
    dv_line = make_copy(a.pipeline_dir, pipe, a.clamp, a.allow_clamped_source)
    sys.path.insert(0, pipe)
    import brian2.groups.group as bgroup
    import brian2.devices.cpp_standalone.device as bdev
    import HPC_main_sweep as S
    import HPC_single_run as R

    # --- interception: one record per device.run, bad V indices of 'Neuron'
    state = {'run': -1, 'bad': {}}
    orig_run = bdev.CPPStandaloneDevice.run
    orig_chk = bgroup.Group._check_for_invalid_values

    def run_hook(self, *args, **kw):
        state['run'] += 1
        return orig_run(self, *args, **kw)

    def chk_hook(self, k, v):
        if self.name == 'Neuron' and k == 'V':
            with np.errstate(all='ignore'):
                x = np.asarray(v, dtype=np.float64)
                bad = ~np.isfinite(x) | (np.abs(x) > 1e50)
            state['bad'][state['run']] = [int(i) for i in np.flatnonzero(bad)]
        return orig_chk(self, k, v)

    bdev.CPPStandaloneDevice.run = run_hook
    bgroup.Group._check_for_invalid_values = chk_hook

    freeze = '' if a.no_freeze else '%s,VA=%g,VR=%g' % (FREEZE_FIXED, a.va, a.vr)
    group = S.resolve_sweep_group(a.group)
    active = S._apply_freeze(freeze, group)
    inactive = sorted(set(range(S.N_DIMS)) - set(active))
    nominal_theta = S.natural_to_theta(np.asarray(S.NOMINAL_PARAMS, dtype=np.float64))
    full = a.mode == 'Full'
    Nn, Na = R.resolve_population_from_density(
        115, 115 if full else 0, 350.0, density=a.rho * (1.0 - a.astro_frac),
        density_astro=a.rho * a.astro_frac)
    Na = int(Na) if full else 0
    ns = argparse.Namespace(syn_pdist_csv=None, lib_dir=pipe)
    try:
        syn_df = R._resolve_syn_pdist_csv(ns)
    except Exception:
        import pandas as pd
        syn_df = pd.read_csv(os.path.join(pipe, 'synapse_pdist.csv'))

    rng = np.random.default_rng(a.seed)
    records = []
    t_start = time.time()
    for t in range(a.n_topo):
        conn_prob = float(rng.uniform(0.1, 0.6))
        base = int(rng.integers(0, 2**31 - 1))
        topo = R.build_topology(
            Nn=Nn, Na=Na, c_max=350.0, conn_prob=conn_prob, syn_prob_csv=syn_df,
            displ_bias=20.0, gj_dist=200.0, stoa_cutoff=70.0, stoa_sigma=200.0,
            seed_neuron=(base + SEEDS['seed_neuron']) % (2**31 - 1),
            seed_synapse=(base + SEEDS['seed_synapse']) % (2**31 - 1),
            seed_astro=(base + SEEDS['seed_astro']) % (2**31 - 1),
            mode=a.mode, topology_mode='wallach', gj_max_dist=150.0,
            conn_rule='flat', p0_conn=float('nan'), d0_conn=float('nan'),
            beta_conn=float('nan'), conn_periodic=False)
        theta = np.array([S.sample_theta(rng) for _ in range(a.k)])
        params = np.array([S.theta_to_natural(th) for th in theta])
        params[:, inactive] = S.NOMINAL_PARAMS[inactive]
        theta[:, inactive] = nominal_theta[inactive]
        seed_runs = [int(s) for s in rng.integers(0, 2**31 - 1, size=a.k)]
        topo_dir = os.path.join(a.work_dir, 'out_' + a.label, 'topo_%03d' % t)
        os.makedirs(topo_dir, exist_ok=True)
        cli = dict(lib_dir=pipe, out_dir=os.path.dirname(topo_dir),
                   simtime=float(a.simtime), mode=a.mode, Nn=int(Nn), Na=int(Na),
                   **SEEDS)
        pack = dict(worker_id=0, topo=topo, topo_idx=t, topo_dir=topo_dir,
                    conn_prob=conn_prob, conn_rule='flat', conn_periodic=False,
                    p0_conn=float('nan'), d0_conn=float('nan'),
                    beta_conn=float('nan'), params_list=list(params),
                    theta_list=list(theta), iter_indices=list(range(a.k)),
                    seed_runs=seed_runs, cli=cli,
                    scratch_root=os.path.join(a.work_dir, 'scratch_' + a.label))
        run0 = state['run'] + 1
        S._worker_entry(pack)
        for j in range(a.k):
            npz = os.path.join(topo_dir, 'iter_%05d.npz' % j)
            rec = {'topo': t, 'iter': j, 'conn_prob': conn_prob,
                   'free': {S.PARAM_NAMES[i]: float(params[j, i]) for i in active}}
            bad = state['bad'].get(run0 + j)
            if bad is None or not os.path.exists(npz):
                rec['status'] = 'no record (run discarded or crashed)'
                records.append(rec)
                continue
            z = np.load(npz)
            ti, ii = z['spk_N_t'].astype(np.float64), z['spk_N_i']
            last = np.full(Nn, -1.0)
            for n in np.unique(ii):
                last[n] = ti[ii == n].max()
            late = a.simtime * 0.9
            rec.update({
                'status': 'ok',
                'n_nan': len(bad),
                'nan_idx': bad,
                'nan_last_spike_s': [float(last[n]) for n in bad],
                'n_spikes': int(len(ti)),
                'n_active_last10pct': int(np.sum(last >= late)),
            })
            records.append(rec)
        print('[check] %s topo %d/%d done (%.0f s elapsed): n_nan per run %s' % (
            a.label, t + 1, a.n_topo, time.time() - t_start,
            [r.get('n_nan') for r in records[-a.k:]]), flush=True)

    out = {'label': a.label, 'clamp': bool(a.clamp), 'va': a.va, 'vr': a.vr,
           'mode': a.mode, 'group': a.group, 'rho': a.rho, 'astro_frac': a.astro_frac,
           'dv_line': dv_line, 'Nn': int(Nn), 'Na': int(Na), 'simtime_s': a.simtime,
           'n_topo': a.n_topo, 'k': a.k, 'seed': a.seed, 'freeze': freeze,
           'free_axes': [S.PARAM_NAMES[i] for i in active], 'records': records,
           'wall_s': time.time() - t_start}
    path = os.path.join(a.work_dir, 'result_%s.json' % a.label)
    json.dump(out, open(path, 'w'), indent=1)
    print('[check] wrote %s' % path)


def compare(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        ok = [r for r in d['records'] if r.get('status') == 'ok']
        n_bad_runs = sum(1 for r in ok if r['n_nan'] > 0)
        frac = [r['n_nan'] / d['Nn'] for r in ok]
        rows.append((d['label'], d['clamp'], d['vr'], len(d['records']), len(ok),
                     n_bad_runs, float(np.mean(frac)) if frac else float('nan'),
                     float(np.max(frac)) if frac else float('nan'), d))
    print('%-14s %-6s %-7s %-6s %-5s %-16s %-14s %-12s' % (
        'variant', 'clamp', 'VR', 'runs', 'ok', 'runs with NaN', 'mean NaN frac', 'max NaN frac'))
    for lab, cl, vr, n, nok, nb, mf, xf, _ in rows:
        print('%-14s %-6s %-7g %-6d %-5d %3d (%5.1f %%)     %-14.3f %-12.3f' % (
            lab, cl, vr, n, nok, nb, 100.0 * nb / max(nok, 1), mf, xf))
    verdict = []
    for lab, cl, vr, n, nok, nb, mf, xf, d in rows:
        if cl:
            verdict.append('%s: %s (%d runs with a NaN neuron)' % (
                lab, 'PASS' if nb == 0 and nok > 0 else 'FAIL', nb))
    for v in verdict:
        print('clamp check  ' + v)
    # which free axes separate NaN runs from clean runs (as-is variants only)
    for lab, cl, vr, n, nok, nb, mf, xf, d in rows:
        if cl or nb == 0 or nb == nok:
            continue
        ok = [r for r in d['records'] if r.get('status') == 'ok']
        y = np.array([r['n_nan'] > 0 for r in ok])
        print('\n%s: median of each free axis, runs WITH vs WITHOUT NaN neurons' % lab)
        for name in list(ok[0]['free']) + ['conn_prob']:
            x = np.array([r['free'][name] if name != 'conn_prob' else r['conn_prob']
                          for r in ok])
            print('  %-11s %10.4g  vs %10.4g' % (name, np.median(x[y]), np.median(x[~y])))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--pipeline_dir', required=True)
    r.add_argument('--work_dir', required=True)
    r.add_argument('--label', required=True)
    r.add_argument('--clamp', action='store_true')
    r.add_argument('--allow_clamped_source', action='store_true',
                   help='run a source that already clamps V (e.g. the patched repository file) as is')
    r.add_argument('--va', type=float, default=-50.0)
    r.add_argument('--vr', type=float, default=-55.0)
    r.add_argument('--no_freeze', action='store_true',
                   help='draw every axis of --group (no LindaColloDUP freeze list)')
    r.add_argument('--group', default='neuron_synapse')
    r.add_argument('--mode', default='Neuronal', choices=['Neuronal', 'Full'])
    r.add_argument('--rho', type=float, default=1300.0, help='total cells per mm^2')
    r.add_argument('--astro_frac', type=float, default=0.0)
    r.add_argument('--n_topo', type=int, default=6)
    r.add_argument('--k', type=int, default=4)
    r.add_argument('--simtime', type=float, default=20.0)
    r.add_argument('--seed', type=int, default=2026)
    c = sub.add_parser('compare')
    c.add_argument('results', nargs='+')
    a = ap.parse_args()
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if a.cmd == 'run':
        os.makedirs(a.work_dir, exist_ok=True)
        run_variant(a)
    else:
        compare(sorted(set(sum((glob.glob(p) for p in a.results), []))))


if __name__ == '__main__':
    main()
