#!/usr/bin/env python3
"""
prepare_kernel_comparison.py
============================

Thin adapter between HPC_single_run.py and analyze_kernel_comparison.py.

HPC_single_run.py runs ONE culture (one beta) per invocation and writes, into its
--out_dir:
    topology.npz        (keys: N_pos, S_i, S_j, ...)         <- positions + edges
    iter_0000000.npz    (keys: spk_N_t [seconds], spk_N_i)   <- neuronal spikes

The kernel-form smoke test runs it twice (beta=1 exponential, beta=2 Gaussian) into
two such directories. This adapter reads both and repackages them into the schema
that analyze_kernel_comparison.py consumes:

    <out_dir>/culture_beta1.npz, culture_beta2.npz
        positions (Nn,2) [um], exc_mask (Nn,), spk_t [ms], spk_i,
        in_degree (Nn,), beta, d0_um, n_edges
    <out_dir>/meta.json
        config = {L_um, density_per_mm2, sigma_um, p0, periodic, simtime_ms}

The d0<->sigma mapping is the integrated-count match used by the smoke test:
    beta = 1 (exponential): d0_conn = sigma
    beta = 2 (Gaussian)   : d0_conn = sqrt(2) * sigma
so the analytic mean in-degree  <k>_inf = 2*pi*rho*p0*sigma^2  is identical across
the two cultures.

No simulation or Brian2 here -- pure numpy repackaging, safe to run on a login node.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np


def _find_iter_npz(culture_dir: Path) -> Path:
    """Return the (single) iter_*.npz spike file in a HPC_single_run output dir."""
    hits = sorted(glob.glob(str(culture_dir / "iter_*.npz")))
    if not hits:
        raise FileNotFoundError(f"No iter_*.npz spike file found in {culture_dir}")
    return Path(hits[0])


def _load_one(culture_dir: Path):
    """Load positions + edges (topology.npz) and neuronal spikes (iter_*.npz)."""
    topo_path = culture_dir / "topology.npz"
    if not topo_path.exists():
        raise FileNotFoundError(f"Missing {topo_path}")
    topo = np.load(topo_path)
    N_pos = np.asarray(topo["N_pos"], dtype=np.float64)
    S_i = np.asarray(topo["S_i"], dtype=np.int64)
    S_j = np.asarray(topo["S_j"], dtype=np.int64)

    spk = np.load(_find_iter_npz(culture_dir))
    spk_t_s = np.asarray(spk["spk_N_t"], dtype=np.float64)   # HPC_single_run saves seconds
    spk_i = np.asarray(spk["spk_N_i"], dtype=np.int32)

    Nn = N_pos.shape[0]
    in_degree = np.bincount(S_j, minlength=Nn).astype(np.int32)
    return dict(
        positions=N_pos, spk_t_ms=spk_t_s * 1000.0, spk_i=spk_i,
        in_degree=in_degree, n_edges=int(S_i.size),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beta1_dir", required=True,
                    help="HPC_single_run output dir for the beta=1 (exponential) culture")
    ap.add_argument("--beta2_dir", required=True,
                    help="HPC_single_run output dir for the beta=2 (Gaussian) culture")
    ap.add_argument("--out_dir", required=True,
                    help="where to write culture_beta1.npz, culture_beta2.npz, meta.json")
    ap.add_argument("--sigma_um", type=float, required=True)
    ap.add_argument("--p0", type=float, required=True)
    ap.add_argument("--density_per_mm2", type=float, required=True)
    ap.add_argument("--c_max", type=float, required=True, help="arena side [um] (= L_um)")
    ap.add_argument("--simtime_s", type=float, required=True)
    ap.add_argument("--periodic", action="store_true",
                    help="set if the runs used --conn_periodic (recorded in meta for the "
                         "analyzer's periodic-aware spatial statistics)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sigma = args.sigma_um
    d0_for_beta = {1.0: sigma, 2.0: float(np.sqrt(2.0) * sigma)}
    rho_per_um2 = args.density_per_mm2 / 1e6
    analytic_k = 2.0 * np.pi * rho_per_um2 * args.p0 * sigma ** 2

    cultures = {}
    for beta, src, tag in ((1.0, args.beta1_dir, "beta1"),
                           (2.0, args.beta2_dir, "beta2")):
        cc = _load_one(Path(src))
        Nn = cc["positions"].shape[0]
        np.savez_compressed(
            out / f"culture_{tag}.npz",
            positions=cc["positions"],
            exc_mask=np.ones(Nn, dtype=bool),
            spk_t=cc["spk_t_ms"], spk_i=cc["spk_i"],
            in_degree=cc["in_degree"],
            beta=float(beta), d0_um=float(d0_for_beta[beta]),
            n_edges=cc["n_edges"],
        )
        realised_k = float(cc["in_degree"].mean())
        cultures[f"beta{int(beta)}"] = dict(
            beta=float(beta), d0_um=float(d0_for_beta[beta]),
            n_edges=cc["n_edges"], realised_mean_indegree=realised_k,
            n_spikes=int(cc["spk_i"].size),
            mean_rate_hz=float(cc["spk_i"].size / (Nn * args.simtime_s)) if args.simtime_s > 0 else 0.0,
        )
        print(f"[adapt] {tag}: Nn={Nn}  edges={cc['n_edges']}  "
              f"<k>={realised_k:.2f} (analytic {analytic_k:.2f})  "
              f"spikes={cc['spk_i'].size}")

    Nn_total = int(np.load(Path(args.beta1_dir) / "topology.npz")["N_pos"].shape[0])
    meta = dict(
        config=dict(
            L_um=args.c_max, density_per_mm2=args.density_per_mm2,
            sigma_um=sigma, p0=args.p0, periodic=bool(args.periodic),
            simtime_ms=args.simtime_s * 1000.0,
            model="CAdEx + Tsodyks-Markram/AMPA/NMDA (HPC_single_run, Neuronal mode)",
        ),
        N=Nn_total,
        analytic_mean_indegree=float(analytic_k),
        cultures=cultures,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[adapt] wrote {out}/culture_beta1.npz, culture_beta2.npz, meta.json")
    print(f"[adapt] analyse with:  python analyze_kernel_comparison.py --in_dir {out}")


if __name__ == "__main__":
    main()
