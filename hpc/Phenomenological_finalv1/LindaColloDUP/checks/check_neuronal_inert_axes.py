#!/usr/bin/env python3
"""
check_neuronal_inert_axes.py -- standalone check (not pipeline code).

Claim tested
    Under MODE=Neuronal, the registry's synapse axes alpha_syn, O_G and Omega_G
    are causally inert: running the production single-simulation path with
    them changed leaves the spike output bit-identical at fixed seeds, while
    changing a live synapse axis (U_0_sr) changes it.

Why it matters
    --sweep_group neuron_synapse draws all three, and build_axis_declaration()
    in HPC_main_sweep.py records them as consumed under MODE=Neuronal.

Truth chosen (fixed before running)
    The repository's own production path, HPC_single_run.py + ASD_fun_BD_cpp.py
    (Astro-Neuron-Network main @ d0ced88), Brian2 cpp_standalone, every axis at
    its registry NOMINAL value unless overridden below, one set of seeds.

Comparison and tolerance (fixed before running)
    replicate     (same flags, run again) : spike arrays EXACTLY equal  (determinism)
    inert variant (3 axes at registry extremes) : EXACTLY equal         -> PASS
    live control  (U_0_sr at its upper bound)  : any element differs   -> PASS

Usage
    python3 -I check_neuronal_inert_axes.py --pipeline_dir <dir> --work_dir <dir>
    <dir> must hold HPC_single_run.py, ASD_fun_BD_cpp.py and synapse_pdist.csv.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

CFG = dict(
    mode='Neuronal', Nn=40, c_max=300, conn_prob=0.3, simtime=2.0,
    seed_device=50, seed_neuron=39, seed_synapse=35, seed_run=777,
)
VARIANTS = {
    'base':          [],
    'replicate':     [],
    'inert_variant': ['--alpha_syn', '0.1', '--O_G', '10.0', '--Omega_G', '0.1'],
    'live_control':  ['--U_0_sr', '0.9'],
}


def run(pipeline_dir, work_dir, tag, extra):
    out = os.path.join(work_dir, f'out_{tag}')
    cmd = [sys.executable, '-I', os.path.join(pipeline_dir, 'HPC_single_run.py'),
           '--out_dir', out, '--lib_dir', pipeline_dir,
           '--mode', CFG['mode'], '--Nn', str(CFG['Nn']),
           '--c_max', str(CFG['c_max']), '--conn_prob', str(CFG['conn_prob']),
           '--simtime', str(CFG['simtime']),
           '--seed_device', str(CFG['seed_device']),
           '--seed_neuron', str(CFG['seed_neuron']),
           '--seed_synapse', str(CFG['seed_synapse']),
           '--seed_run', str(CFG['seed_run'])] + extra
    env = dict(os.environ, MPLBACKEND='Agg')
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise RuntimeError(f'{tag}: HPC_single_run.py exited {r.returncode}')
    d = np.load(os.path.join(out, 'iter_0000000.npz'))
    return {'t': np.asarray(d['spk_N_t']), 'i': np.asarray(d['spk_N_i'])}


def identical(a, b):
    return np.array_equal(a['i'], b['i']) and np.array_equal(a['t'], b['t'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline_dir', required=True)
    ap.add_argument('--work_dir', required=True)
    a = ap.parse_args()
    os.makedirs(a.work_dir, exist_ok=True)
    res = {k: run(a.pipeline_dir, a.work_dir, k, v) for k, v in VARIANTS.items()}
    b = res['base']
    print(f'config: {CFG}')
    print(f"base: {len(b['t'])} spikes, {len(np.unique(b['i']))}/{CFG['Nn']} neurons "
          f"active, mean rate {len(b['t']) / CFG['Nn'] / CFG['simtime']:.2f} Hz")
    want = {'replicate': True, 'inert_variant': True, 'live_control': False}
    ok = True
    for k, expect_equal in want.items():
        eq = identical(res[k], b)
        verdict = (eq == expect_equal)
        ok &= verdict
        print(f"{k:14s} flags={VARIANTS[k]}  spikes={len(res[k]['t'])}  "
              f"identical_to_base={eq}  -> {'PASS' if verdict else 'FAIL'}")
    print('RESULT:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
