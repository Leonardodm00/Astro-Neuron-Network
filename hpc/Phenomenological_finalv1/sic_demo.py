#!/usr/bin/env python3
"""
sic_demo.py -- SIC on/off twins (SPEC Block 12): the simulations behind the
figures of sic_demo_plots.py. Simulation only; no figure is drawn here.

One topology (GiuliaGate geometry by default: 1200 neurons and 1200
astrocytes per mm^2 on a 300 um well, i.e. 108 + 108 cells; wallach rule;
flat conn_prob 0.10; contact gate 1.0) and, for each regime, two runs that
differ only in the SIC: on (the pipeline) and off (build_and_run(sic=False),
the ablation). Same topology, same network seeds, same run seed, so the
random streams are the same until the dynamics differ.

Regimes (registry nominal values except where named):
  nominal      the registry nominal point: O_N 0.3 1/(uM s); the astrocytes
               sit in a calcium plateau below C_Theta (SPEC section 8), so few
               or no releases are expected and the twins should barely differ
  oscillating  O_N 1e-4: the astrocytes oscillate and release; G0 = rho_e G_T
               U_A = 78 uM per release (the calibration point, about 43 pA
               per pair at -60 mV)
  quiet_strong the quiet corner of SPEC section 8 (Sigma 3 mV, U_0_sr 0.1,
               O_N 1e-4, O_G 10, Omega_G 0.001, G_T 1000 mM, U_A 0.9): a
               quiet network and G0 = 585 uM per release (about 190 pA per
               pair at -60 mV)
  repeated     O_N 1e-4 and C_Theta 0.3 uM (registry box [0.1, 2]): the slow
               calcium oscillation crosses C_Theta again after the onset, so
               the astrocytes release about every 10 s, not only at the
               onset of the run

Recorded: spikes of every neuron and astrocyte; V (and I_sic, r_sic_tot when
the SIC is on) of a few neurons every 0.5 ms (the partner neurons of the two
astrocytes with most partners, and neurons with no partner); I_sic and
r_sic_tot of every neuron, C and G_A of every astrocyte every 5 ms. Also a
voltage-clamp check (the stand-in of smoke_test_sic.py, Brian2 numpy device)
and the scipy reference for G0 = 3, 78 and 585 uM.

Outputs in --out: topology.npz, run_<regime>_<on|off>.npz, clamp.npz,
demo_meta.json.

Run (from this folder; about 12-15 min on two cores for the four regimes):
    python3 sic_demo.py --out /path/to/sic_demo_out [--simtime 40] [--jobs 2]
    python3 sic_demo_plots.py /path/to/sic_demo_out
"""
import argparse
import json
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REGIMES = {
    'nominal': {},
    'oscillating': {'O_N': 1e-4},
    'quiet_strong': {'Sigma': 3.0, 'U_0_sr': 0.1, 'O_N': 1e-4, 'O_G': 10.0,
                     'Omega_G': 0.001, 'G_T': 1000.0, 'U_A': 0.9},
    'repeated': {'O_N': 1e-4, 'C_Theta': 0.3},
}


def build_args(opts, out_dir):
    """The single-run argparse namespace for the demo geometry."""
    import HPC_single_run as R
    argv = ['--out_dir', out_dir, '--lib_dir', HERE, '--mode', 'Full',
            '--simtime', str(opts.simtime), '--density', str(opts.density),
            '--density_astro', str(opts.density), '--c_max', str(opts.c_max),
            '--conn_prob', str(opts.conn_prob), '--conn_rule', 'flat',
            '--topology_mode', 'wallach', '--stoa_gate_p', str(opts.gate),
            '--seed_run', str(opts.seed_run), '--seed_neuron', str(opts.seed_topo),
            '--seed_synapse', str(opts.seed_topo + 1), '--seed_astro', str(opts.seed_topo + 2)]
    args = R.build_parser().parse_args(argv)
    args.Nn, args.Na = R.resolve_population_from_density(
        args.Nn, args.Na, args.c_max, density=args.density, density_astro=args.density_astro)
    return args


def regime_params(args, name):
    import HPC_single_run as R
    p = R.resolve_params(args).astype(np.float64)
    for k, v in REGIMES[name].items():
        p[R.PARAM_NAMES.index(k)] = v
    return p


def make_topology(args):
    import HPC_single_run as R
    csv = R._resolve_syn_pdist_csv(args)
    return R.build_topology(
        Nn=args.Nn, Na=args.Na, c_max=args.c_max, conn_prob=args.conn_prob,
        syn_prob_csv=csv, displ_bias=args.displ_bias, gj_dist=args.gj_dist,
        stoa_cutoff=args.stoa_cutoff, stoa_sigma=args.stoa_sigma,
        seed_neuron=args.seed_neuron, seed_synapse=args.seed_synapse,
        seed_astro=args.seed_astro, mode='Full', topology_mode=args.topology_mode,
        gj_max_dist=args.gj_max_dist, conn_rule=args.conn_rule, p0_conn=args.p0_conn,
        d0_conn=args.d0_conn, beta_conn=args.beta_conn, conn_periodic=args.conn_periodic,
        stoa_gate_p=args.stoa_gate_p, seed_stoa_gate=args.seed_stoa_gate)


def pick_neurons(topo, n_ctrl=3):
    """Neurons whose V is recorded finely: the partners of the two astrocytes
    with most partners, and up to n_ctrl neurons with the fewest partners."""
    a, i = topo['SIC_i'], topo['SIC_j']
    Nn = len(topo['N_pos'])
    per_a = np.bincount(a, minlength=len(topo['A_pos']))
    per_n = np.bincount(i, minlength=Nn)
    focus = [int(x) for x in np.argsort(-per_a, kind='stable')[:2]]
    chosen = []
    for f in focus:
        chosen += [int(x) for x in i[a == f] if int(x) not in chosen]
    rest = [int(x) for x in np.argsort(per_n, kind='stable') if int(x) not in chosen]
    ctrl = rest[:n_ctrl]
    return focus, chosen, ctrl


def run_one(job):
    """One simulation in its own process (Brian2's device is per process)."""
    opts, out_dir, name, sic_on, record = job
    import HPC_single_run as R
    args = build_args(opts, out_dir)
    topo = dict(np.load(os.path.join(out_dir, 'topology.npz')))
    params = regime_params(args, name)
    t0 = time.time()
    res = R.build_and_run(topo, args, params, sic=(None if sic_on else False), record=record)
    rec = res.pop('record')
    path = os.path.join(out_dir, 'run_%s_%s.npz' % (name, 'on' if sic_on else 'off'))
    np.savez_compressed(path, params=params, param_names=np.array(R.PARAM_NAMES),
                        regime=np.array(name), sic_on=np.bool_(sic_on),
                        simtime=np.float64(args.simtime),
                        n_sic_pairs=np.int64(res['n_sic_pairs']),
                        sic_scheme=np.array(res['sic_scheme']),
                        **{k: v for k, v in res.items() if isinstance(v, np.ndarray)},
                        **{'rec_' + k: v for k, v in rec.items()})
    return dict(path=path, regime=name, sic_on=sic_on, wall_s=time.time() - t0,
                n_spikes=int(res['spk_N_t'].size), n_astro_events=int(res['spk_A_t'].size),
                t_run_s=float(res['t_run_s']), t_compile_s=float(res['t_compile_s']))


def clamp_check(out_dir):
    """Brian2 voltage-clamp traces (smoke_test_sic stand-in, numpy device) and
    the scipy reference for three release sizes."""
    import smoke_test_sic as T
    with tempfile.TemporaryDirectory(prefix='sic_demo_clamp_') as tmp:
        d = T.stand_in(tmp, False, 0.05, t_end_ms=6000.0)
    if d is None:
        raise RuntimeError('clamp stand-in failed')
    t = d['t']
    refs = {('r_ref_G%d' % int(g)): T.r_reference(t, G0=g) for g in (3.0, 78.0, 585.0)}
    np.savez_compressed(os.path.join(out_dir, 'clamp.npz'), t_ms=t, I_pA=d['I'][3],
                        r_brian=T._pair_row(d, 0, 3), **refs,
                        g_sic_nS=np.float64(T.SPEC['g_sic']), V_mV=np.float64(T.V_CLAMP))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', required=True)
    ap.add_argument('--simtime', type=float, default=40.0, help='s per run')
    ap.add_argument('--density', type=float, default=1200.0, help='cells/mm^2 (each type)')
    ap.add_argument('--c_max', type=float, default=300.0, help='um')
    ap.add_argument('--conn_prob', type=float, default=0.10)
    ap.add_argument('--gate', type=float, default=1.0, help='contact gate stoa_gate_p')
    ap.add_argument('--seed_topo', type=int, default=101)
    ap.add_argument('--seed_run', type=int, default=7)
    ap.add_argument('--regimes', nargs='+', default=list(REGIMES), choices=list(REGIMES))
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--no_clamp', action='store_true')
    opts = ap.parse_args()
    out_dir = os.path.abspath(opts.out)
    os.makedirs(out_dir, exist_ok=True)

    import HPC_single_run as R
    args = build_args(opts, out_dir)
    topo = make_topology(args)
    R.save_topology(topo, out_dir)
    focus, chosen, ctrl = pick_neurons(topo)
    record = dict(neurons=np.array(chosen + ctrl, dtype=np.int32), dt_ms=0.5, all_dt_ms=5.0)
    regimes = {k: REGIMES[k] for k in opts.regimes}
    meta_path = os.path.join(out_dir, 'demo_meta.json')
    if os.path.exists(meta_path):                      # a later call adds regimes
        old = json.load(open(meta_path))
        regimes = dict(old.get('regimes', {}), **regimes)
    meta = dict(regimes=regimes, Nn=int(args.Nn), Na=int(args.Na),
                c_max=float(args.c_max), conn_prob=float(args.conn_prob), gate=float(args.stoa_gate_p),
                simtime=float(args.simtime), seed_run=int(opts.seed_run), seed_topo=int(opts.seed_topo),
                focus_astrocytes=focus, recorded_neurons=chosen + ctrl, control_neurons=ctrl,
                n_sic_pairs=int(len(topo['SIC_i'])), n_synapses=int(len(topo['S_i'])))
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print('[demo] %d + %d cells, %d synapses, %d SIC pairs; focus astrocytes %s; recorded '
          'neurons %s (controls %s)' % (args.Nn, args.Na, len(topo['S_i']), len(topo['SIC_i']),
                                        focus, chosen, ctrl), flush=True)
    if not opts.no_clamp and not os.path.exists(os.path.join(out_dir, 'clamp.npz')):
        clamp_check(out_dir)
        print('[demo] clamp check written', flush=True)
    jobs = [(opts, out_dir, name, on, record) for name in opts.regimes for on in (True, False)]
    with ProcessPoolExecutor(max_workers=max(1, opts.jobs),
                             mp_context=multiprocessing.get_context('spawn')) as ex:
        for r in ex.map(run_one, jobs):
            print('[demo] %-12s SIC %-3s: %6d neuron spikes, %4d astrocyte events, run %.0f s, '
                  'compile %.0f s -> %s' % (r['regime'], 'on' if r['sic_on'] else 'off', r['n_spikes'],
                                           r['n_astro_events'], r['t_run_s'], r['t_compile_s'],
                                           os.path.basename(r['path'])), flush=True)
    print('[demo] done:', out_dir)


if __name__ == '__main__':
    main()
