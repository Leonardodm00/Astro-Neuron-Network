"""Smoke test for campaign_inspect -- Block 14 in specs/SPEC.md.

A synthetic campaign in the sweep's layout (two tasks, k = 4 and k = 2
simulations per topology; complete, partial and unstarted topologies; one
failure record; runs on every class boundary) is indexed by the real
aggregate_sweep.py --write-index, then inspected.

Checks, from the strongest oracle to the weakest
    test_known_answer   class counts, kept runs and topology states known
                        by construction
    test_reference      classify() against a brute-force rule, row by row
    test_convergence    not a discretisation (skipped)
    test_invariants     shares sum to 1; histogram counts add up
    test_contract       main() writes the JSON, the report and 3 figures;
                        quantiles equal numpy.quantile of the kept values
    test_determinism    two runs of main() give the same numbers
    test_edge_cases     empty and NaN input, all-partial campaign, stale
                        index, missing summary.csv
    test_job_script     submit_campaign_inspect.sh: bash -n, dry run, real
                        run (SKIP_CONDA=1), stale-index rule, bad REINDEX
    test_job_env_clobber  the job's real activation block, through a stand-in
                        conda whose activation exports BUILD and HOST and
                        overwrites the job's former variable names: the index
                        is still rebuilt and the report written (job 2100916)

Run (from the folder that holds this file; numpy, pandas, matplotlib, bash):
    python3 smoke_test_campaign_inspect.py
Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table, then
the full message of every failure; exit code 1 if any check failed.
Pure ASCII, so it survives transfer to the cluster.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign_inspect as ci  # noqa: E402

SEED = 20261119
REPORT_PACKAGES = ("numpy", "pandas", "matplotlib")
AGG = HERE / "aggregate_sweep.py"
JOB = HERE / "submit_campaign_inspect.sh"
NN, T_SIM = 159, 180.0

# (task, k, [(topo_idx, state, [(rate_hz, frac_active), ...], n_failures)])
DESIGN = [
    ("sweep_cfd_task0000", 4, [
        (0, "complete", [(0.0, 0.0), (20.0, 0.1), (300.0, 1.0), (299.999, 1.0)], 0),
        (1, "complete", [(25.0, 0.05), (20.0001, 0.5), (0.01, 0.0999), (1.0, 0.6)], 0),
        (2, "partial", [(310.0, 1.0), (310.0, 1.0)], 0),
        (3, "unstarted", [], 0),
        # cut by a kill: 2 sidecars + 2 failure records reach k = 4 by count,
        # yet only 2 runs finished, so the topology is partial (lcdup15 n01:
        # 3 such topologies, 2026-11-20)
        (4, "cut", [(310.0, 1.0), (310.0, 1.0)], 2),
    ]),
    ("sweep_cpu_task0000", 2, [
        (0, "complete", [(5.0, 0.9), (0.5, 0.3)], 0),
    ]),
]
# Expected for the kept runs (complete topologies), by construction:
KEPT_RATES = [0.0, 20.0, 300.0, 299.999, 25.0, 20.0001, 0.01, 1.0, 5.0, 0.5]
KEPT_FRACS = [0.0, 0.1, 1.0, 1.0, 0.05, 0.5, 0.0999, 0.6, 0.9, 0.3]
EXPECTED_CLASSES = {"silent": 3, "active": 4, "high": 2, "saturated": 1}


def build_campaign(root: Path, design=DESIGN, manifest_age_s: float = 1000.0) -> Path:
    """Write a campaign tree in the sweep's layout; returns the campaign folder."""
    camp = root / "campaign_synthetic"
    old = time.time() - manifest_age_s
    for task, k, topos in design:
        td = camp / task
        td.mkdir(parents=True)
        (td / "job_args.json").write_text(json.dumps(
            {"_resolved_n_workers": k, "n_params_per_worker": 1}))
        for ti, state, runs, n_fail in topos:
            tp = td / ("topo_%05d" % ti)
            tp.mkdir()
            (tp / "topology.npz").write_bytes(b"")
            if state != "unstarted":
                (tp / "topology_meta.json").write_text(json.dumps({"Nn": NN, "simtime_s": T_SIM}))
            for j, (rate, frac) in enumerate(runs):
                side = {"theta": [0.0] * 37, "topo_idx": ti, "iter_idx": j, "conn_prob": 0.2,
                        "mean_FR_Hz": rate, "across_cell_rate_cv": 0.0, "frac_active": frac,
                        "mean_isi_cv": 0.0, "n_neuronal_spikes": int(round(rate * NN * T_SIM)),
                        "n_astrocyte_events": 0, "t_run_s": 1.0, "seed_run": 1000 + j}
                (tp / ("iter_%05d.json" % j)).write_text(json.dumps(side))
                (tp / ("iter_%05d.npz" % j)).write_bytes(b"")
            if n_fail:
                (tp / "_failures.jsonl").write_text(
                    "".join(json.dumps({"iter_idx": 90 + f}) + "\n" for f in range(n_fail)))
        man = td / "manifest.json"
        man.write_text("{}")
        os.utime(man, (old, old))
    return camp


def index_campaign(camp: Path) -> None:
    r = subprocess.run([sys.executable, str(AGG), str(camp), "--write-index"],
                       cwd=str(HERE), capture_output=True, text=True)
    assert r.returncode == 0, "aggregate_sweep.py failed:\n" + r.stdout + r.stderr


def inspect(camp: Path):
    cfg = ci.InspectConfig()
    return ci.summarize(ci.load_index(camp), ci.scan_tree(camp), cfg)


def run_job(env_extra, camp: Path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PBS_")}
    env.update({"REPO_DIR": str(HERE), "CAMPAIGN": str(camp), "SKIP_CONDA": "1",
                "PATH": os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")})
    env.update(env_extra)
    return subprocess.run(["bash", str(JOB)], cwd=str(HERE), env=env,
                          capture_output=True, text=True)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    with tempfile.TemporaryDirectory() as tmp:
        camp = build_campaign(Path(tmp))
        index_campaign(camp)
        s = inspect(camp)
        cut = ci.scan_tree(camp)["sweep_cfd_task0000"]["topos"][4]
    # the topology cut by a kill reaches k only with its failure records
    assert cut == {"state": "partial", "n_json": 2, "n_fail": 2}, cut
    got = {c: s["classes"][c]["n"] for c in ci.CLASS_NAMES}
    assert got == EXPECTED_CLASSES, got
    assert s["index_rows"] == 14, s["index_rows"]          # 10 kept + 2 + 2 of the partial ones
    assert s["runs_kept"] == 10 and s["runs_classified"] == 10, (s["runs_kept"], s["runs_classified"])
    assert s["index_rows_by_topology_state"] == {"complete": 10, "partial": 4}, \
        s["index_rows_by_topology_state"]
    assert s["topologies"] == {"complete": 3, "partial": 2, "unstarted": 1, "unknown": 0}, \
        s["topologies"]
    assert s["failures_in_kept_topologies"] == 0, s["failures_in_kept_topologies"]
    assert s["failure_records_by_topology_state"]["partial"] == 2, s["failure_records_by_topology_state"]
    assert s["topologies_with_failures_by_state"]["partial"] == 1, s["topologies_with_failures_by_state"]
    assert s["no_spike"] == 1 and s["silent_above_r_high"] == 1, (s["no_spike"], s["silent_above_r_high"])
    assert s["no_spike_not_silent"] == 0 and s["runs_above_ceiling"] == 0
    assert s["index_stale"] is False, s["index_mismatch_examples"]
    assert s["max_rate_hz"] == 300.0, s["max_rate_hz"]


def _reference_class(r, a, cfg):
    if a < cfg.a_min:
        return ci.SILENT
    if r >= cfg.r_sat_hz:
        return ci.SATURATED
    if r > cfg.r_high_hz:
        return ci.HIGH
    return ci.ACTIVE


def test_reference():
    cfg = ci.InspectConfig()
    rng = np.random.default_rng(SEED)
    r = np.concatenate([10.0 ** rng.uniform(-5, 2.6, 5000),
                        [0.0, 20.0, np.nextafter(20.0, 21.0), 300.0, np.nextafter(300.0, 0.0), 333.3]])
    a = np.concatenate([rng.uniform(0, 1, 5000), [0.1, np.nextafter(0.1, 0.0), 0.1, 1.0, 1.0, 0.0]])
    got = ci.classify(r, a, cfg)
    want = np.array([_reference_class(ri, ai, cfg) for ri, ai in zip(r, a)], dtype=np.int8)
    assert np.array_equal(got, want), int(np.sum(got != want))
    assert got.dtype == np.int8


def test_convergence():
    raise unittest.SkipTest("not a discretisation")


def test_invariants():
    with tempfile.TemporaryDirectory() as tmp:
        camp = build_campaign(Path(tmp))
        index_campaign(camp)
        s = inspect(camp)
    shares = [s["classes"][c]["share"] for c in ci.CLASS_NAMES]
    # float64 sum of four ratios n/9: exact to a few ulp
    assert abs(sum(shares) - 1.0) < 1e-12, sum(shares)
    assert sum(s["classes"][c]["n"] for c in ci.CLASS_NAMES) == s["runs_classified"]
    h = s["histograms"]["log10_rate"]
    assert sum(h["counts"]) + h["n_zero_rate"] + h["n_below_range"] + h["n_above_range"] \
        == s["runs_classified"], h
    assert sum(s["histograms"]["frac_active"]["counts"]) == s["runs_classified"]
    assert len(h["edges"]) == len(h["counts"]) + 1 == 145, len(h["edges"])


def test_contract():
    with tempfile.TemporaryDirectory() as tmp:
        camp = build_campaign(Path(tmp))
        index_campaign(camp)
        out = Path(tmp) / "out"
        rc = ci.main([str(camp), "--out", str(out)])
        assert rc == 0, rc
        names = sorted(p.name for p in out.iterdir())
        assert names == ["fig_class_shares.png", "fig_frac_active_hist.png", "fig_rate_hist.png",
                         "inspect_report.txt", "inspect_summary.json"], names
        s = json.loads((out / "inspect_summary.json").read_text())
        report = (out / "inspect_report.txt").read_text()
    ps = ci.InspectConfig().quantiles
    want_r = np.quantile(np.array(KEPT_RATES), ps)
    want_a = np.quantile(np.array(KEPT_FRACS), ps)
    got_r = np.array([s["quantiles"]["mean_FR_Hz"][f"{p:g}"] for p in ps])
    got_a = np.array([s["quantiles"]["frac_active"][f"{p:g}"] for p in ps])
    # same function on the same float64 values, through a JSON round trip
    # (repr of a float is exact): equality, no tolerance
    assert np.array_equal(got_r, want_r), (got_r, want_r)
    assert np.array_equal(got_a, want_a), (got_a, want_a)
    for c in ci.CLASS_NAMES:
        assert f"  {c} " in report, c
    assert s["inputs"]["summary_csv"].endswith("campaign_index/summary.csv")


def test_determinism():
    with tempfile.TemporaryDirectory() as tmp:
        camp = build_campaign(Path(tmp))
        index_campaign(camp)
        runs = []
        for name in ("a", "b"):
            out = Path(tmp) / name
            assert ci.main([str(camp), "--out", str(out), "--no-plots"]) == 0
            s = json.loads((out / "inspect_summary.json").read_text())
            s["inputs"].pop("generated_utc")
            runs.append(s)
    assert runs[0] == runs[1]


def test_edge_cases():
    cfg = ci.InspectConfig()
    e = ci.classify(np.array([]), np.array([]), cfg)
    assert e.shape == (0,) and e.dtype == np.int8
    for r, a in ((np.array([np.nan]), np.array([0.5])), (np.array([1.0, 2.0]), np.array([0.5]))):
        try:
            ci.classify(r, a, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("classify accepted %r, %r" % (r, a))
    all_partial = [("sweep_cfd_task0000", 4, [(0, "partial", [(5.0, 0.9)], 0)])]
    with tempfile.TemporaryDirectory() as tmp:
        camp = build_campaign(Path(tmp), all_partial)
        index_campaign(camp)
        s = inspect(camp)
        assert s["runs_classified"] == 0 and s["quantiles"]["mean_FR_Hz"] == {}, s
        assert all(s["classes"][c]["n"] == 0 and s["classes"][c]["share"] is None
                   for c in ci.CLASS_NAMES)
        ci.format_report(s)
    with tempfile.TemporaryDirectory() as tmp:
        camp = build_campaign(Path(tmp))
        index_campaign(camp)
        extra = camp / "sweep_cpu_task0000" / "topo_00000" / "iter_00007.json"
        extra.write_text(json.dumps({"theta": [0.0] * 37, "topo_idx": 0, "iter_idx": 7,
                                     "mean_FR_Hz": 1.0, "frac_active": 0.5,
                                     "n_neuronal_spikes": 28620}))
        s = inspect(camp)
        assert s["index_stale"] is True and s["index_mismatched_topologies"] == 1, s
        (camp / "campaign_index" / "summary.csv").unlink()
        assert ci.main([str(camp), "--no-plots"]) == 2


def test_job_script():
    if shutil.which("bash") is None:
        raise unittest.SkipTest("bash not available")
    r = subprocess.run(["bash", "-n", str(JOB)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with tempfile.TemporaryDirectory() as tmp:
        camp = build_campaign(Path(tmp))
        dry = run_job({"DRYRUN": "1"}, camp)
        assert dry.returncode == 0, dry.stdout + dry.stderr
        assert "rebuild  : 1 (no summary.csv yet)" in dry.stdout, dry.stdout
        assert ("campaign_inspect.py %s --out" % camp.resolve()) in dry.stdout, dry.stdout
        assert not (camp / "campaign_index").exists()
        out = Path(tmp) / "job_out"
        real = run_job({"OUT": str(out)}, camp)
        assert real.returncode == 0, real.stdout + real.stderr
        assert (camp / "campaign_index" / "summary.csv").is_file()
        for name in ("inspect_report.txt", "inspect_summary.json", "aggregate_index.log",
                     "fig_rate_hist.png"):
            assert (out / name).is_file(), name
        assert "saturated" in real.stdout
        fresh = run_job({"DRYRUN": "1"}, camp)
        assert "rebuild  : 0 (summary.csv newer than all 2 manifest.json)" in fresh.stdout, fresh.stdout
        man = camp / "sweep_cfd_task0000" / "manifest.json"
        later = (camp / "campaign_index" / "summary.csv").stat().st_mtime + 60
        os.utime(man, (later, later))
        stale = run_job({"DRYRUN": "1"}, camp)
        assert "rebuild  : 1 (1 of 2 manifest.json newer than summary.csv)" in stale.stdout, stale.stdout
        bad = run_job({"DRYRUN": "1", "REINDEX": "yes"}, camp)
        assert bad.returncode == 2 and "REINDEX must be" in bad.stderr, bad.stderr


# A stand-in for `module` and conda, so that the job's real activation block
# runs. Its `conda activate` exports BUILD and HOST, as conda's compiler
# packages do, and overwrites every name the job reads or once read after the
# activation (job 2100916 on 2026-11-19 lost its rebuild flag BUILD this way).
FAKE_CONDA_SH = r"""
conda() {
    case "$1" in
        activate)
            export CONDA_PREFIX="@ENV@"
            export BUILD=x86_64-conda-linux-gnu HOST=x86_64-conda-linux-gnu
            OUT=/nonexistent/out; IDX=/nonexistent/summary.csv; CAMPAIGN=/nonexistent
            REINDEX=0; DRYRUN=1; WHY=clobbered; PY=/bin/false; RC=99; EXIT_CODE=99
            N_MAN=0; N_NEW=0; REPO_DIR=/nonexistent
            ;;
        deactivate) : ;;
    esac
}
"""


def test_job_env_clobber():
    if shutil.which("bash") is None:
        raise unittest.SkipTest("bash not available")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base, env_dir = tmp / "fakeconda", tmp / "fakeenv"
        (base / "bin").mkdir(parents=True)
        (base / "etc" / "profile.d").mkdir(parents=True)
        (env_dir / "bin").mkdir(parents=True)
        os.symlink(sys.executable, env_dir / "bin" / "python")
        (base / "etc" / "profile.d" / "conda.sh").write_text(
            FAKE_CONDA_SH.replace("@ENV@", str(env_dir)))
        conda_bin = base / "bin" / "conda"
        conda_bin.write_text('#!/bin/sh\n[ "$1" = info ] && [ "$2" = --base ] && echo "%s"\n' % base)
        module_bin = base / "bin" / "module"
        module_bin.write_text("#!/bin/sh\nexit 0\n")
        for p in (conda_bin, module_bin):
            p.chmod(0o755)
        camp = build_campaign(tmp)
        out = tmp / "job_out"
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("PBS_", "BASH_FUNC_", "CONDA", "BUILD", "HOST"))}
        env.update({"REPO_DIR": str(HERE), "CAMPAIGN": str(camp), "OUT": str(out),
                    "PATH": str(base / "bin") + os.pathsep + "/usr/bin" + os.pathsep + "/bin"})
        r = subprocess.run(["bash", str(JOB)], cwd=str(HERE), env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "[index] python" in r.stdout, r.stdout
        assert "[env] %s" % (env_dir / "bin" / "python") in r.stdout, r.stdout
        assert (camp / "campaign_index" / "summary.csv").is_file()
        assert (out / "inspect_report.txt").is_file() and (out / "aggregate_index.log").is_file()
        s = json.loads((out / "inspect_summary.json").read_text())
        assert {c: s["classes"][c]["n"] for c in ci.CLASS_NAMES} == EXPECTED_CLASSES, s["classes"]


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())
    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")
    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
