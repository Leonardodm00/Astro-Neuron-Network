#!/usr/bin/env python3
"""
smoke_test_mea_noise.py -- how process_campaign.py seeds each iteration's
additive recording noise (--noise_seed_scheme; 2026-10-06, D-072).

    python3 smoke_test_mea_noise.py                       # the tools next to this file
    python3 smoke_test_mea_noise.py --library LIB.npz     # skip the ~30 s library build
    python3 smoke_test_mea_noise.py --old-tools DIR       # N1 against that copy instead of git

Expect: ALL 7 CHECKS PASSED. Needs numpy + scipy, and for N1 the previous
process_campaign.py (by default read from git, commit 04977e0, the C8 state;
or --old-tools DIR holding it with its mea_*.py modules). No PBS, no conda: it
writes throwaway campaign folders whose topology and spike files are
byte-identical across tasks, so that the noise is the only thing that can make
two outputs differ, and runs the REAL process_campaign.py on them.

The two schemes (process_campaign.noise_entropy):
  topo_iter  noise_seed_base + 1000 * topo_idx + iter_idx: the scheme before
             2026-10-06 (C8's Outputs_v2, August's mea_out_1electrode).
  sim        SeedSequence(noise_seed_base, seed_run, topo_idx, iter_idx), the
             default: seed_run is the simulation's own draw in the ANN sweep;
             a missing or negative seed_run is replaced by 2**32 + crc32 of
             'campaign/sweep'.

CHECKS
------
N0  the comparison is not vacuous: one output with a changed noise target is
    reported as different
N1  topo_iter reproduces the previous process_campaign.py bit for bit (every
    array; meta_json equal once the new noise_seed_scheme key is set aside)
N2  two tasks, same topology and spikes, different seed_run: identical outputs
    under topo_iter (the shared noise), different outputs under sim
N3  a replay -- the same files, seed_run included, in another campaign folder:
    identical outputs under sim
N4  seed_run = -1 in two tasks: different outputs under sim; the recorded key
    is 2**32 + crc32('campaign/sweep')
N5  noise_entropy in every file is the documented tuple, under both schemes
N6  within one task, the noise of iteration 1000 of topology 0 and of
    iteration 0 of topology 1, drawn as process_iter draws it: equal under
    topo_iter (the collision past 1000 iterations), different under sim; and
    such a task runs through

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
import zlib

import numpy as np

_PASS, _FAIL = [], []
TOOLS = ("process_campaign.py", "mea_probe.py", "mea_detection.py", "mea_synthesis.py",
         "mea_plots.py", "eap_template_library.py")
GEOM = ["--n_side", "1", "--pitch", "200", "--edge", "26.59", "--fs", "10000"]
NN = 40
BASE = 70000


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


def make_inputs(seed=0):
    """One topology (positions) and three iterations' spikes, as arrays."""
    rng = np.random.default_rng(seed)
    c_max = 300.0
    pos = rng.uniform(0, c_max, (NN, 2))
    pos[0] = [c_max / 2, c_max / 2]            # a neuron under the centre electrode
    spikes = []
    for _ in range(3):
        t, i = [], []
        for n in range(NN):
            tt = np.sort(rng.uniform(0.2, 1.8, rng.integers(3, 12)))
            t.append(tt)
            i.append(np.full(len(tt), n))
        spikes.append((np.concatenate(t).astype(np.float32), np.concatenate(i).astype(np.int32)))
    return c_max, pos, spikes


def write_task(root, campaign, sweep, c_max, pos, layout):
    """layout: {topo_idx: [(iter_idx, (spk_t, spk_i), seed_run), ...]}"""
    d = os.path.join(root, campaign, sweep)
    for k, iters in layout.items():
        td = os.path.join(d, "topo_%05d" % k)
        os.makedirs(td, exist_ok=True)
        np.savez_compressed(os.path.join(td, "topology.npz"), N_pos=pos)
        with open(os.path.join(td, "topology_meta.json"), "w") as fh:
            json.dump({"c_max": c_max}, fh)
        for n, (st, si), sr in iters:
            np.savez_compressed(os.path.join(td, "iter_%05d.npz" % n),
                                params=np.arange(37, dtype=float), theta=np.arange(23, dtype=float),
                                conn_prob=np.float64(0.2), spk_N_t=st, spk_N_i=si,
                                spk_A_t=np.array([]), spk_A_i=np.array([]), seed_run=np.int64(sr))
    return d


def run(tools, camp, out, lib, scheme=None, extra=()):
    cmd = [sys.executable, os.path.join(tools, "process_campaign.py"), "--campaign", camp,
           "--out", out, "--library", lib] + GEOM + list(extra)
    if scheme:
        cmd += ["--noise_seed_scheme", scheme]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900, cwd=tools)
    if p.returncode != 0:
        raise AssertionError("process_campaign.py exited %d:\n%s" % (p.returncode, (p.stdout + p.stderr)[-1500:]))
    if glob.glob(os.path.join(out, "topo_*", "_failures.log")):
        raise AssertionError("an iteration raised: %s" % glob.glob(os.path.join(out, "topo_*", "_failures.log")))
    return out


def outputs(out):
    return sorted(os.path.relpath(p, out) for p in glob.glob(os.path.join(out, "topo_*", "mea_iter_*.npz")))


def diff_file(pa, pb, skip=("noise_entropy", "meta_json")):
    """Keys whose arrays differ bit for bit, apart from `skip`."""
    za, zb = np.load(pa, allow_pickle=False), np.load(pb, allow_pickle=False)
    keys = (set(za.files) | set(zb.files)) - set(skip)
    bad = []
    for k in sorted(keys):
        if k not in za.files or k not in zb.files:
            bad.append(k + " (missing)")
            continue
        x, y = za[k], zb[k]
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            bad.append(k)
    return bad


def same_outputs(a, b, rel_a=None, rel_b=None, also_skip=()):
    """(identical?, details) over the paired files of two output roots;
    noise_entropy, meta_json and also_skip are set aside."""
    fa, fb = (rel_a or outputs(a)), (rel_b or outputs(b))
    if len(fa) != len(fb) or not fa:
        return False, "file lists differ: %r vs %r" % (fa, fb)
    det = {}
    for x, y in zip(fa, fb):
        d = diff_file(os.path.join(a, x), os.path.join(b, y),
                      skip=("noise_entropy", "meta_json") + tuple(also_skip))
        if d:
            det[x] = d
    return not det, det


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--library", default=None)
    ap.add_argument("--old-tools", default=None,
                    help="a folder with the previous process_campaign.py and its modules (default: git 04977e0)")
    a = ap.parse_args(argv)
    root = tempfile.mkdtemp(prefix="mea_noise_")
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
    raw = os.path.join(root, "raw")
    c_max, pos, sp = make_inputs(0)
    three = lambda sr0: {0: [(n, sp[n], sr0 + n) for n in range(3)]}      # noqa: E731
    A = write_task(raw, "campaign_x_v1", "sweep_cpu_task0000", c_max, pos, three(111))
    B = write_task(raw, "campaign_x_v2", "sweep_cpu_task0000", c_max, pos, three(555))
    C = write_task(raw, "campaign_x_v3", "sweep_cpu_task0007", c_max, pos, three(111))
    Dd = write_task(raw, "campaign_x_v1", "sweep_cpu_task0001", c_max, pos, {0: [(n, sp[n], -1) for n in range(3)]})
    E = write_task(raw, "campaign_x_v2", "sweep_cpu_task0001", c_max, pos, {0: [(n, sp[n], -1) for n in range(3)]})
    F = write_task(raw, "campaign_x_v1", "sweep_cfd_task0000", c_max, pos,
                   {0: [(1000, sp[0], 801)], 1: [(0, sp[0], 802)]})
    o = lambda tag: os.path.join(root, "out", tag)     # noqa: E731
    print("smoke_test_mea_noise -- tools from %s" % os.path.abspath(a.tools_dir))
    print("-" * 70)

    def n0():
        run(tools, A, o("n0a"), lib, "sim")
        run(tools, A, o("n0b"), lib, "sim", extra=["--target_noise_uv", "5.5"])
        same, det = same_outputs(o("n0a"), o("n0b"))
        if same:
            raise AssertionError("a changed noise target compared equal")
        return "a changed noise target is caught (%d of %d files differ)" % (len(det), len(outputs(o("n0a"))))

    def n1():
        old = os.path.abspath(a.old_tools) if a.old_tools else None
        if not old:
            old = os.path.join(root, "old_tools")
            os.makedirs(old)
            repo = subprocess.run(["git", "-C", a.tools_dir, "rev-parse", "--show-toplevel"],
                                  capture_output=True, text=True)
            if repo.returncode != 0:
                raise AssertionError("not a git checkout: give --old-tools DIR")
            for n in TOOLS:
                blob = subprocess.run(["git", "-C", repo.stdout.strip(), "show", "04977e0:hpc/MEA Traces/%s" % n],
                                      capture_output=True)
                if blob.returncode != 0:
                    raise AssertionError("git show 04977e0:%s failed: %s" % (n, blob.stderr[:200]))
                with open(os.path.join(old, n), "wb") as fh:
                    fh.write(blob.stdout)
        if "noise_seed_scheme" in open(os.path.join(old, "process_campaign.py")).read():
            raise AssertionError("%s is not the previous process_campaign.py" % old)
        run(old, A, o("n1_old"), lib)
        run(tools, A, o("n1_new"), lib, "topo_iter")
        same, det = same_outputs(o("n1_old"), o("n1_new"))
        if not same:
            raise AssertionError("topo_iter differs from the previous code: %r" % det)
        for rel in outputs(o("n1_old")):
            mo = json.loads(str(np.load(os.path.join(o("n1_old"), rel))["meta_json"]))
            mn = json.loads(str(np.load(os.path.join(o("n1_new"), rel))["meta_json"]))
            if mn.pop("noise_seed_scheme", None) != "topo_iter" or mo != mn:
                raise AssertionError("%s: meta_json differs beyond noise_seed_scheme" % rel)
        return "topo_iter == the previous process_campaign.py over %d files, bit for bit" % len(outputs(o("n1_old")))

    def n2():
        for sch in ("topo_iter", "sim"):
            run(tools, A, o("n2a_" + sch), lib, sch)
            run(tools, B, o("n2b_" + sch), lib, sch)
        # seed_run is passed through to the output and differs by construction
        same_old, det_old = same_outputs(o("n2a_topo_iter"), o("n2b_topo_iter"), also_skip=("seed_run",))
        same_new, det = same_outputs(o("n2a_sim"), o("n2b_sim"), also_skip=("seed_run",))
        if not same_old:
            raise AssertionError("topo_iter: the two tasks differ in %r, so the noise was not shared "
                                 "(test broken?)" % det_old)
        if same_new:
            raise AssertionError("sim: two simulations with different seed_run gave identical outputs")
        return "different seed_run: shared noise under topo_iter, own noise under sim (%d of 3 files differ)" % len(det)

    def n3():
        run(tools, A, o("n3a"), lib, "sim")
        run(tools, C, o("n3c"), lib, "sim")
        same, det = same_outputs(o("n3a"), o("n3c"))
        if not same:
            raise AssertionError("a replay with the same seed_run differs: %r" % det)
        return "a replay in another campaign folder keeps its copy's noise (3 files identical)"

    def n4():
        run(tools, Dd, o("n4d"), lib, "sim")
        run(tools, E, o("n4e"), lib, "sim")
        same, _ = same_outputs(o("n4d"), o("n4e"))
        if same:
            raise AssertionError("seed_run -1 in two tasks gave identical outputs")
        for tag, name in (("n4d", "campaign_x_v1/sweep_cpu_task0001"), ("n4e", "campaign_x_v2/sweep_cpu_task0001")):
            for rel in outputs(o(tag)):
                ent = np.load(os.path.join(o(tag), rel))["noise_entropy"].tolist()
                if ent[1] != 2 ** 32 + zlib.crc32(name.encode("utf-8")):
                    raise AssertionError("%s %s: key %r, want 2**32 + crc32(%r)" % (tag, rel, ent[1], name))
        return "seed_run -1: each task its own noise, keyed by 2**32 + crc32('campaign/sweep')"

    def n5():
        n = 0
        for tag, sch in (("n2a_sim", "sim"), ("n2a_topo_iter", "topo_iter"), ("n2b_sim", "sim")):
            for rel in outputs(o(tag)):
                z = np.load(os.path.join(o(tag), rel))
                ent = z["noise_entropy"].tolist()
                t, i, sr = int(z["topo_idx"]), int(z["iter_idx"]), int(z["seed_run"])
                want = [BASE, sr, t, i] if sch == "sim" else [BASE + 1000 * t + i]
                if ent != want or z["noise_entropy"].dtype != np.int64:
                    raise AssertionError("%s %s: noise_entropy %r, want %r" % (tag, rel, ent, want))
                if json.loads(str(z["meta_json"])).get("noise_seed_scheme") != sch:
                    raise AssertionError("%s %s: meta_json noise_seed_scheme is not %s" % (tag, rel, sch))
                n += 1
        if n != 9:                       # 3 runs x 3 iterations, written by N2
            raise AssertionError("%d files checked, want 9 (did N2's runs write their outputs?)" % n)
        return "noise_entropy and meta_json's noise_seed_scheme as documented in %d files" % n

    def n6():
        # the topology's template draw (tmpl_seed_base + topo_idx) differs between topologies,
        # so whole outputs cannot match here: the noise itself is compared, drawn exactly as
        # process_iter draws it (default_rng(noise_seed(entropy)), E x T standard normals)
        sys.path.insert(0, tools)
        import process_campaign as PC                       # noqa: E402
        res = {}
        for sch in ("topo_iter", "sim"):
            cfg = PC.PipelineConfig(noise_seed_scheme=sch)
            draws = []
            for t, i, sr in ((0, 1000, 801), (1, 0, 802)):
                ent = PC.noise_entropy(cfg, sr, t, i, "campaign_x_v1/sweep_cfd_task0000")
                draws.append(np.random.default_rng(PC.noise_seed(ent)).standard_normal((1, 2000)))
            res[sch] = bool(np.array_equal(draws[0], draws[1]))
        if res != {"topo_iter": True, "sim": False}:
            raise AssertionError("noise of (topology 0, iteration 1000) vs (topology 1, iteration 0) "
                                 "equal under each scheme: %r, want topo_iter True, sim False" % res)
        run(tools, F, o("n6"), lib, "sim")                  # and the task runs through, both files written
        if len(outputs(o("n6"))) != 2:
            raise AssertionError("not both iterations were written: %r" % outputs(o("n6")))
        return "past 1000 iterations: topo_iter repeats the next topology's noise, sim does not"

    for nm, fn in (("N0", n0), ("N1", n1), ("N2", n2), ("N3", n3), ("N4", n4), ("N5", n5), ("N6", n6)):
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
