#!/usr/bin/env python3
"""
smoke_test_mea_workers.py -- the detections do not depend on how many worker
processes process_campaign.py uses, nor on the BLAS / OpenMP thread settings.

    python3 smoke_test_mea_workers.py                      # the tools next to this file
    python3 smoke_test_mea_workers.py --library LIB.npz    # skip the ~30 s library build
    python3 smoke_test_mea_workers.py --tools-dir DIR      # another copy of the tools

Expect: ALL 5 CHECKS PASSED. Needs numpy + scipy; no PBS, no conda, no
campaign data: it writes a throwaway synthetic campaign (4 topologies x 3
iterations, 40 neurons, ~2 s of spikes each) and runs the REAL
process_campaign.py on it several times, each into its own output root.

Why it holds, from the code: one topology is one unit of work
(walk_campaign -> _worker -> process_topo); inside it, the template draw is
seeded by the topology index (tmpl_seed_base + topo_idx) and the noise of an
iteration by its noise_entropy -- the topology and iteration indices read from
the file names and, under the default 'sim' scheme (D-072), the simulation's
own seed_run read from the file -- so no random stream is shared between
workers and the order in which they finish changes nothing; pool.map returns
the results in topology order, so mea_manifest.json is the same too. This
test checks that by running it, not by reading it.

CHECKS
------
W0  the comparison is not vacuous: two runs that differ in one knob
    (--target_noise_uv 5.0 vs 5.5) are reported as different
W1  DUP15HD geometry (--n_side 3 --pitch 60 --edge 25 --fs 10110.09):
    --workers 1 and --workers 3 give the same files, arrays and manifest
W2  Giulia geometry (--n_side 1 --pitch 200 --edge 26.59 --fs 10000): the same
W3  --workers 3 with OMP_NUM_THREADS=1 and with OMP_NUM_THREADS=2 (PBS sets
    OMP_NUM_THREADS to the job's ncpus; process_campaign.py only setdefault()s
    it): the same
W4  every comparison above covered every iteration file of every topology
    (12 mea_iter files per run, none skipped)

Compared per run: the set of files under the output root; every array of
every mea_iter_*.npz, bit for bit (dtype, shape and bytes; meta_json as its
string; np.array_equal would call two NaNs unequal, and det_dt_to_truth holds
NaN where a detection has no true spike nearby); mea_manifest.json minus its
'created' timestamp; the absence of any _failures.log.

HPC note (hpc-python-compat): pure ASCII, LF only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

_PASS, _FAIL = [], []
N_TOPOS, N_ITERS, NN = 4, 3, 40
GEOM_DUP15HD = ["--n_side", "3", "--pitch", "60.0", "--edge", "25.0", "--fs", "10110.09"]
GEOM_GIULIA = ["--n_side", "1", "--pitch", "200", "--edge", "26.59", "--fs", "10000"]
TOOLS = ("process_campaign.py", "mea_probe.py", "mea_detection.py", "mea_synthesis.py",
         "mea_plots.py", "eap_template_library.py")


def check(name, fn):
    try:
        d = fn()
    except Exception as exc:                                  # noqa: BLE001
        _FAIL.append((name, "%s: %s" % (type(exc).__name__, exc)))
        print("  [FAIL] %-3s %s: %s" % (name, type(exc).__name__, exc))
        traceback.print_exc()
        return
    _PASS.append((name, d))
    print("  [PASS] %-3s %s" % (name, d))


def make_campaign(root, rng):
    """A sweep task as the ANN sweep writes it: topo_*/topology.npz +
    topology_meta.json + iter_*.npz (the fields process_campaign.py reads)."""
    camp = os.path.join(root, "campaign_test", "sweep_cpu_task0000")
    c_max = 300.0
    for k in range(N_TOPOS):
        td = os.path.join(camp, "topo_%05d" % k)
        os.makedirs(td)
        N_pos = rng.uniform(0, c_max, (NN, 2))
        N_pos[0] = [c_max / 2, c_max / 2]          # one neuron under the centre electrode
        np.savez_compressed(os.path.join(td, "topology.npz"), N_pos=N_pos)
        with open(os.path.join(td, "topology_meta.json"), "w") as fh:
            json.dump({"c_max": c_max}, fh)
        for n in range(N_ITERS):
            spk_t, spk_i = [], []
            for i in range(NN):
                t = np.sort(rng.uniform(0.2, 1.8, rng.integers(3, 12)))
                spk_t.append(t)
                spk_i.append(np.full(len(t), i))
            np.savez_compressed(
                os.path.join(td, "iter_%05d.npz" % n),
                params=np.arange(36, dtype=float), theta=np.arange(26, dtype=float),
                conn_prob=np.float64(0.3),
                spk_N_t=np.concatenate(spk_t).astype(np.float32),
                spk_N_i=np.concatenate(spk_i).astype(np.int32),
                spk_A_t=np.array([]), spk_A_i=np.array([]), seed_run=np.int64(123 + n))
    return camp


def run(tools, camp, out, lib, workers, geom, omp="1", extra=()):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_"))}
    env["OMP_NUM_THREADS"] = omp
    cmd = [sys.executable, os.path.join(tools, "process_campaign.py"), "--campaign", camp,
           "--out", out, "--library", lib, "--workers", str(workers)] + list(geom) + list(extra)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900, cwd=tools)
    if p.returncode != 0:
        raise AssertionError("process_campaign.py exited %d:\n%s" % (p.returncode, (p.stdout + p.stderr)[-1500:]))
    want = "%d topos, %d/%d iters processed" % (N_TOPOS, N_TOPOS * N_ITERS, N_TOPOS * N_ITERS)
    if want not in p.stdout:
        raise AssertionError("not every iteration was processed:\n%s" % p.stdout[-800:])
    return p.stdout


def compare(a, b):
    """Differences between two output roots, as a list of strings; also the
    number of mea_iter files compared."""
    diffs = []
    fa = sorted(os.path.relpath(p, a) for p in glob.glob(os.path.join(a, "**", "*"), recursive=True)
                if os.path.isfile(p))
    fb = sorted(os.path.relpath(p, b) for p in glob.glob(os.path.join(b, "**", "*"), recursive=True)
                if os.path.isfile(p))
    if fa != fb:
        diffs.append("file sets differ: only in A %s, only in B %s"
                     % (sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))))
    if any(os.path.basename(f) == "_failures.log" for f in fa + fb):
        diffs.append("a _failures.log exists")
    n = 0
    for rel in fa:
        if not rel.endswith(".npz") or rel not in fb:
            continue
        n += 1
        za, zb = np.load(os.path.join(a, rel), allow_pickle=False), np.load(os.path.join(b, rel), allow_pickle=False)
        if sorted(za.files) != sorted(zb.files):
            diffs.append("%s: keys %s vs %s" % (rel, sorted(za.files), sorted(zb.files)))
            continue
        for k in za.files:
            x, y = za[k], zb[k]
            # bitwise: np.array_equal calls two NaNs unequal, and the
            # detections hold NaN where a detection has no true spike nearby
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                diffs.append("%s[%s] differs (dtype %s/%s, shape %s/%s)" % (rel, k, x.dtype, y.dtype, x.shape, y.shape))
    ma, mb = (json.load(open(os.path.join(r, "mea_manifest.json"))) for r in (a, b))
    ma.pop("created", None)
    mb.pop("created", None)
    if ma != mb:
        diffs.append("mea_manifest.json differs: %r vs %r" % (ma, mb))
    return diffs, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--library", default=None, help="a pre-built eap_library.npz")
    a = ap.parse_args(argv)
    root = tempfile.mkdtemp(prefix="mea_workers_")
    tools = os.path.join(root, "tools")
    os.makedirs(tools)
    for n in TOOLS:
        shutil.copyfile(os.path.join(a.tools_dir, n), os.path.join(tools, n))
    lib = os.path.join(root, "eap_library.npz")
    if a.library:
        shutil.copyfile(a.library, lib)
    else:
        subprocess.run([sys.executable, os.path.join(tools, "eap_template_library.py"), "--out", lib],
                       check=True, capture_output=True, text=True, cwd=tools)
    camp = make_campaign(os.path.join(root, "raw"), np.random.default_rng(0))
    print("smoke_test_mea_workers -- tools from %s" % os.path.abspath(a.tools_dir))
    print("-" * 70)
    out = lambda tag: os.path.join(root, "out_" + tag)      # noqa: E731
    counts = []

    def w0():
        run(tools, camp, out("n50"), lib, 1, GEOM_DUP15HD, extra=["--target_noise_uv", "5.0"])
        run(tools, camp, out("n55"), lib, 1, GEOM_DUP15HD, extra=["--target_noise_uv", "5.5"])
        d, n = compare(out("n50"), out("n55"))
        if not d:
            raise AssertionError("two runs with different noise targets compared equal")
        return "a one-knob change is caught (%d difference(s) over %d files)" % (len(d), n)

    def _pair(tag, geom, w_b, omp_a="1", omp_b="1"):
        run(tools, camp, out(tag + "_a"), lib, 1 if tag != "omp" else w_b, geom, omp=omp_a)
        s2 = run(tools, camp, out(tag + "_b"), lib, w_b, geom, omp=omp_b)
        if "dispatching over %d worker(s)" % w_b not in s2:
            raise AssertionError("run B did not use %d workers:\n%s" % (w_b, s2[-500:]))
        d, n = compare(out(tag + "_a"), out(tag + "_b"))
        if d:
            raise AssertionError("; ".join(d[:5]))
        counts.append(n)
        return n

    def w1():
        n = _pair("dup", GEOM_DUP15HD, 3)
        return "n_side 3: --workers 1 == --workers 3 over %d mea_iter files and the manifest" % n

    def w2():
        n = _pair("giu", GEOM_GIULIA, 3)
        return "n_side 1, pitch 200, edge 26.59, fs 10000: --workers 1 == --workers 3 over %d files" % n

    def w3():
        n = _pair("omp", GEOM_DUP15HD, 3, omp_a="1", omp_b="2")
        return "--workers 3: OMP_NUM_THREADS 1 == 2 over %d files" % n

    def w4():
        want = N_TOPOS * N_ITERS
        if counts != [want] * 3:
            raise AssertionError("files compared per check %r, want %d each" % (counts, want))
        return "each comparison covered all %d mea_iter files" % want

    for nm, fn in (("W0", w0), ("W1", w1), ("W2", w2), ("W3", w3), ("W4", w4)):
        check(nm, fn)
    shutil.rmtree(root, ignore_errors=True)
    print("-" * 70)
    if _FAIL:
        print("FAILED %d of %d" % (len(_FAIL), len(_PASS) + len(_FAIL)))
        return 1
    print("ALL %d CHECKS PASSED" % len(_PASS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
