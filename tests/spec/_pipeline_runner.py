"""Helper for the spec tests (not a test): one HPC_single_run.build_and_run in a
fresh process (cpp_standalone builds touch Brian2's global device state).

usage: python _pipeline_runner.py CONFIG_JSON OUT_NPZ
CONFIG keys: pipe (folder of the pipeline), mode, Nn, Na, c_max, conn_prob,
simtime, seed_device, seed_run (or null), seeds (neuron, synapse, astro),
stoa_gate_p, params (name -> value overrides of NOMINAL_PARAMS), sic (null or
false), record (null or dict), astro_ic ('driven' or 'quiescent').
The topology is saved too (topo_<key> arrays) so a test can read it.
"""
import json
import os
import sys

sys.dont_write_bytecode = True


def main():
    cfg = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    pipe = cfg['pipe']
    sys.path.insert(0, pipe)
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import pandas as pd
    import HPC_single_run as R
    args = R.build_parser().parse_args(['--out_dir', os.path.dirname(os.path.abspath(out)),
                                        '--lib_dir', pipe])
    args.mode = cfg.get('mode', 'Full')
    args.Nn, args.Na = int(cfg['Nn']), int(cfg.get('Na', 0))
    args.c_max = float(cfg.get('c_max', 200.0))
    args.conn_prob = float(cfg.get('conn_prob', 0.2))
    args.simtime = float(cfg['simtime'])
    args.seed_device = int(cfg.get('seed_device', 50))
    args.seed_run = cfg.get('seed_run')
    sn, ss, sa = cfg.get('seeds', (39, 35, 60))
    topo = R.build_topology(Nn=args.Nn, Na=(args.Na if args.mode == 'Full' else 0), c_max=args.c_max,
                            conn_prob=args.conn_prob,
                            syn_prob_csv=pd.read_csv(os.path.join(pipe, 'synapse_pdist.csv')),
                            displ_bias=20.0, gj_dist=200.0, stoa_cutoff=70.0, stoa_sigma=200.0,
                            seed_neuron=sn, seed_synapse=ss, seed_astro=sa, mode=args.mode,
                            topology_mode='wallach', gj_max_dist=150.0,
                            stoa_gate_p=float(cfg.get('stoa_gate_p', 1.0)), seed_stoa_gate=71)
    params = np.asarray(R.NOMINAL_PARAMS, dtype=np.float64).copy()
    for name, v in (cfg.get('params') or {}).items():
        params[R.PARAM_NAMES.index(name)] = float(v)
    kw = dict(record=cfg.get('record'), astro_ic=cfg.get('astro_ic', 'driven'))
    if cfg.get('sic', None) is False:
        kw['sic'] = False
    res = R.build_and_run(topo, args, params, **kw)
    arrays = {k: np.asarray(v) for k, v in res.items() if k != 'record' and np.ndim(v) >= 0
              and not isinstance(v, (dict, str))}
    for k, v in (res.get('record') or {}).items():
        arrays['rec_' + k] = np.asarray(v)
    for k, v in topo.items():
        arrays['topo_' + k] = np.asarray(v)
    arrays['params'] = params
    np.savez(out, **arrays)
    json.dump({k: v for k, v in res.items() if isinstance(v, (str, bool, int, float))},
              open(out + '.json', 'w'))


if __name__ == '__main__':
    main()
