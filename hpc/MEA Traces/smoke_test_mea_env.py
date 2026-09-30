#!/usr/bin/env python3
"""
smoke_test_mea_env.py -- how the MEA array launchers choose, forward and
activate the python environment of their jobs.

    python3 smoke_test_mea_env.py                    # tests the scripts next to this file
    python3 smoke_test_mea_env.py --scripts-dir DIR  # tests another copy (negative control)

Expect: ALL 12 CHECKS PASSED. Needs bash and a python3 with numpy + scipy
(the job's own preflight imports both). No conda, no PBS, no campaign data:
it writes a throwaway fixture and a FAKE `conda` that reproduces the
davinci failure mode -- the hook and `conda activate` read variables that
are unset in a batch shell, which kills a `set -u` shell inside the eval,
silently.

CHECKS
------
submit_mea_array.sh, run end to end (a stub process_campaign.py prints what
ran it):
E1  CONDA_ENV=<name>, conda present   -> activated, job completes
    (before 2026-09-28: the job died inside the hook, silently)
E2  ENV_PREFIX=<path>, conda present  -> activated through conda (so the
    env's activate.d hooks run), not a bare PATH prepend
E3  ENV_PREFIX=<path>, no conda       -> PATH-prepend fallback, WARNING
E4  no env at all, no conda           -> DEFAULT_ENV_NAME not found, WARNING,
    the job still reaches its preflight (unchanged behaviour)
E5  activate_env_by_name lifts -u and RESTORES it: -u on before -> on after
E6  ... and off before -> off after
launch_mea_array.sh --dry-run (prints the qsub line, submits nothing):
L1  an env active in the shell AND --conda-env NAME -> CONDA_ENV=NAME
    forwarded, no ENV_PREFIX (before: the ambient prefix won)
L2  an env active, no flag            -> ENV_PREFIX=<ambient>, announced
L3  --conda-prefix AND --conda-env    -> refused, exit 2
L4  no env active, no flag            -> neither forwarded
L5  an env active AND --conda-prefix P -> ENV_PREFIX=P
launch_mea_per_campaign.sh --dry-run:
P1  an env active AND --conda-env NAME, scripts at mode 644 as a git
    checkout leaves them -> the inner launch forwards CONDA_ENV=NAME (before:
    the ambient prefix won, and the inner script was exec'd directly)

HPC note (hpc-python-compat): pure ASCII, LF only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback

_PASS, _FAIL = [], []

FAKE_CONDA = r'''#!/bin/bash
# Fake conda for smoke_test_mea_env.py. `conda shell.bash hook` prints a shell
# function `conda`. Both the hook text and `conda activate` read a variable
# that is never set, as geotiff's deactivate.d script does on davinci:
# harmless without -u, fatal (and silent, output being discarded) with it.
if [ "${1:-}" = "shell.bash" ] && [ "${2:-}" = "hook" ]; then
cat <<'HOOK'
: "${_FAKE_DEACTIVATE_D_UNSET}"
conda() {
    if [ "$1" = "activate" ]; then
        : "${_FAKE_ACTIVATE_D_UNSET}"
        local target="$2" pfx
        case "$target" in
            /*) pfx="$target" ;;
            *)  pfx="${FAKE_ENVS_ROOT}/$target" ;;
        esac
        [ -x "$pfx/bin/python3" ] || return 1
        export PATH="$pfx/bin:$PATH"
        export CONDA_PREFIX="$pfx"
        export CONDA_DEFAULT_ENV="$(basename "$pfx")"
        export FAKE_ACTIVATED="$pfx"
        return 0
    fi
    return 1
}
HOOK
exit 0
fi
exit 1
'''

STUB_PROCESS_CAMPAIGN = '''import os, shutil, sys
print("STUB ran python3=%s FAKE_ACTIVATED=%s argv=%s" % (
    shutil.which("python3"), os.environ.get("FAKE_ACTIVATED", ""), sys.argv[1:]))
'''


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


class Fixture:
    def __init__(self, scripts_dir):
        self.src = os.path.abspath(scripts_dir)
        self.root = tempfile.mkdtemp(prefix="mea_env_")
        r = self.root
        # scripts copied at mode 644, as a git checkout of this repo leaves them
        self.scripts = os.path.join(r, "scripts")
        os.makedirs(self.scripts)
        for n in ("submit_mea_array.sh", "launch_mea_array.sh",
                  "launch_mea_per_campaign.sh", "build_mea_manifest.py"):
            shutil.copyfile(os.path.join(self.src, n), os.path.join(self.scripts, n))
            os.chmod(os.path.join(self.scripts, n), 0o644)
        # fake conda
        self.conda_bin = os.path.join(r, "conda_bin")
        os.makedirs(self.conda_bin)
        self._exe(os.path.join(self.conda_bin, "conda"), FAKE_CONDA)
        # a fake env whose python3 is a wrapper around this interpreter
        self.envs = os.path.join(r, "envs")
        self.env = os.path.join(self.envs, "fakeenv")
        os.makedirs(os.path.join(self.env, "bin"))
        self._exe(os.path.join(self.env, "bin", "python3"),
                  '#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
        # this interpreter, reached through a wrapper in a directory of its own:
        # putting dirname(sys.executable) on PATH would also expose a `conda`
        # that sits next to it (a conda base python), and the job would find it
        self.py_bin = os.path.join(r, "py_bin")
        os.makedirs(self.py_bin)
        self._exe(os.path.join(self.py_bin, "python3"),
                  '#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
        # PBS_O_WORKDIR with the stub program, a one-line manifest, a library
        self.work = os.path.join(r, "work")
        os.makedirs(self.work)
        with open(os.path.join(self.work, "process_campaign.py"), "w") as fh:
            fh.write(STUB_PROCESS_CAMPAIGN)
        self.lib = os.path.join(r, "eap_library.npz")
        open(self.lib, "wb").close()
        camp = os.path.join(r, "campaigns", "campaign_x")
        self.unit = os.path.join(camp, "sweep_cpu_task0000")
        os.makedirs(os.path.join(self.unit, "topo_00000"))
        self.camp = camp
        self.manifest = os.path.join(r, "one.tsv")
        with open(self.manifest, "w") as fh:
            fh.write("%s\t%s\n" % (self.unit, os.path.join(r, "out", "x")))
        # a PATH with bash, sed, cut, ... but no conda and no python3 of ours
        self.base_path = os.pathsep.join(
            d for d in os.environ.get("PATH", "").split(os.pathsep)
            if d and not os.path.isfile(os.path.join(d, "conda")))

    @staticmethod
    def _exe(path, text):
        with open(path, "w") as fh:
            fh.write(text)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def env_base(self, with_conda):
        # Hermetic on a cluster too: besides conda's own variables, drop the
        # shell functions a module system exports (BASH_FUNC_module%% ...) and
        # BASH_ENV / ENV, through which it defines `module` in every
        # non-interactive bash. Left in, the job's `module load anaconda3`
        # route could reach a REAL conda and activate a REAL env (E4). Not
        # neutralised: a conda installed at one of the ABSOLUTE paths the job
        # also searches (/opt/conda, /usr/local/{anaconda3,miniconda3}).
        e = {k: v for k, v in os.environ.items()
             if not k.startswith(("CONDA", "_CONDA", "FAKE_", "ENV_PREFIX",
                                  "PBS_", "EXTRA_ARGS", "_FAKE", "BASH_FUNC_",
                                  "MODULE", "LOADEDMODULES", "_LMFILES_",
                                  "LMOD", "__LMOD", "_ModuleTable"))
             and k not in ("BASH_ENV", "ENV")}
        e["PATH"] = (self.conda_bin + os.pathsep if with_conda else "") + \
            self.py_bin + os.pathsep + self.base_path
        e["HOME"] = self.root            # step 3/4 of the activation search find nothing here
        e["FAKE_ENVS_ROOT"] = self.envs
        return e

    def run_submit(self, with_conda, **extra):
        e = self.env_base(with_conda)
        e.update(MANIFEST=self.manifest, LIB=self.lib, PBS_O_WORKDIR=self.work,
                 PBS_ARRAY_INDEX="0")
        e.update(extra)
        p = subprocess.run(["bash", os.path.join(self.scripts, "submit_mea_array.sh")],
                           env=e, capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout + p.stderr

    def run_launch(self, script, args, ambient=None):
        e = self.env_base(False)
        if ambient:
            e["CONDA_PREFIX"] = ambient
            e["CONDA_DEFAULT_ENV"] = os.path.basename(ambient)
        cwd = tempfile.mkdtemp(prefix="launch_", dir=self.root)
        p = subprocess.run(["bash", os.path.join(self.scripts, script)] + args,
                           env=e, capture_output=True, text=True, timeout=120, cwd=cwd)
        return p.returncode, p.stdout + p.stderr


def _qsub_line(out):
    lines = [ln for ln in out.splitlines() if ln.startswith("[launch] qsub ")]
    if len(lines) != 1:
        raise AssertionError("expected one '[launch] qsub' line, got %d:\n%s"
                             % (len(lines), out[-800:]))
    return lines[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts-dir", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args(argv)
    F = Fixture(a.scripts_dir)
    print("smoke_test_mea_env -- scripts from %s" % F.src)
    print("-" * 70)

    def e1():
        rc, out = F.run_submit(True, CONDA_ENV="fakeenv")
        if rc != 0 or "index 0 done." not in out:
            raise AssertionError("job did not complete (rc %d):\n%s" % (rc, out[-600:]))
        if "FAKE_ACTIVATED=%s" % F.env not in out:
            raise AssertionError("env not activated:\n%s" % out[-600:])
        return "by name: activated through the -u trap, job completed"

    def e2():
        rc, out = F.run_submit(True, ENV_PREFIX=F.env)
        if rc != 0 or "FAKE_ACTIVATED=%s" % F.env not in out:
            raise AssertionError("prefix not activated through conda (rc %d):\n%s"
                                 % (rc, out[-600:]))
        return "by prefix: activated through conda, activate.d would run"

    def e3():
        rc, out = F.run_submit(False, ENV_PREFIX=F.env)
        want = "python3=%s" % os.path.join(F.env, "bin", "python3")
        if rc != 0 or want not in out or "conda could not activate" not in out:
            raise AssertionError("no PATH fallback with a warning (rc %d):\n%s"
                                 % (rc, out[-600:]))
        if "FAKE_ACTIVATED=%s" % F.env in out:
            raise AssertionError("claims activation without conda")
        return "no conda: PATH-prepend fallback, warned"

    def e4():
        rc, out = F.run_submit(False)
        if "could not activate 'brian_env'" not in out or "[mea-array] python" not in out:
            raise AssertionError("default-env fallback changed (rc %d):\n%s" % (rc, out[-600:]))
        return "no env: DEFAULT_ENV_NAME missing, warned, preflight reached"

    def _funcs():
        src = open(os.path.join(F.scripts, "submit_mea_array.sh")).read()
        i0 = src.find("_activate_env_by_name_unguarded() {")
        i1 = src.find('if [ -n "${ENV_PREFIX:-}" ]; then')
        if i0 < 0 or i1 < 0 or "activate_env_by_name() {" not in src[i0:i1]:
            raise AssertionError("the guarded activate_env_by_name is not in this script")
        return src[i0:i1]

    def _flags_after(opts):
        body = "set %s\n%s\nactivate_env_by_name fakeenv\necho \"FLAGS=$-\"\n" % (opts, _funcs())
        p = subprocess.run(["bash", "-c", body], env=F.env_base(True),
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            raise AssertionError("rc %d: %s" % (p.returncode, (p.stdout + p.stderr)[-300:]))
        return [ln for ln in p.stdout.splitlines() if ln.startswith("FLAGS=")][-1][6:]

    def e5():
        f = _flags_after("-euo pipefail")
        if "u" not in f:
            raise AssertionError("-u not restored: $- = %s" % f)
        return "-u on before, on after ($- = %s)" % f

    def e6():
        f = _flags_after("-eo pipefail")
        if "u" in f:
            raise AssertionError("-u switched on: $- = %s" % f)
        return "-u off before, off after ($- = %s)" % f

    base = ["--out-root", os.path.join(F.root, "out"), "--campaign-root", F.camp,
            "--lib", F.lib, "--dry-run"]
    ambient = "/nonexistent/envs/ambientenv"

    def l1():
        rc, out = F.run_launch("launch_mea_array.sh", base + ["--conda-env", "fakeenv"], ambient)
        q = _qsub_line(out)
        if rc != 0 or "CONDA_ENV=fakeenv" not in q or "ENV_PREFIX=" in q:
            raise AssertionError("explicit --conda-env lost (rc %d): %s" % (rc, q))
        return "explicit --conda-env beats the ambient env"

    def l2():
        rc, out = F.run_launch("launch_mea_array.sh", base, ambient)
        q = _qsub_line(out)
        if rc != 0 or "ENV_PREFIX=%s" % ambient not in q:
            raise AssertionError("ambient env not forwarded (rc %d): %s" % (rc, q))
        if "forwarding the env active" not in out:
            raise AssertionError("ambient forwarding not announced")
        return "ambient env forwarded and announced"

    def l3():
        rc, out = F.run_launch("launch_mea_array.sh",
                               base + ["--conda-prefix", F.env, "--conda-env", "fakeenv"])
        if rc != 2 or "not both" not in out:
            raise AssertionError("both flags not refused (rc %d):\n%s" % (rc, out[-300:]))
        return "--conda-prefix with --conda-env refused, exit 2"

    def l4():
        rc, out = F.run_launch("launch_mea_array.sh", base)
        q = _qsub_line(out)
        if rc != 0 or "ENV_PREFIX=" in q or "CONDA_ENV=" in q:
            raise AssertionError("something forwarded (rc %d): %s" % (rc, q))
        return "nothing active, nothing given: nothing forwarded"

    def l5():
        rc, out = F.run_launch("launch_mea_array.sh", base + ["--conda-prefix", F.env], ambient)
        q = _qsub_line(out)
        if rc != 0 or "ENV_PREFIX=%s" % F.env not in q or ambient in q:
            raise AssertionError("explicit --conda-prefix lost (rc %d): %s" % (rc, q))
        return "explicit --conda-prefix beats the ambient env"

    def p1():
        args = ["--out-root", os.path.join(F.root, "out"), "--campaign-root", F.camp,
                "--lib", F.lib, "--dry-run", "--conda-env", "fakeenv"]
        rc, out = F.run_launch("launch_mea_per_campaign.sh", args, ambient)
        q = _qsub_line(out)
        if rc != 0 or "CONDA_ENV=fakeenv" not in q or "ENV_PREFIX=" in q:
            raise AssertionError("per-campaign launch lost --conda-env (rc %d): %s" % (rc, q))
        return "per-campaign: --conda-env reaches the inner launch; 644 scripts run"

    for nm, fn in (("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4), ("E5", e5),
                   ("E6", e6), ("L1", l1), ("L2", l2), ("L3", l3), ("L4", l4),
                   ("L5", l5), ("P1", p1)):
        check(nm, fn)
    shutil.rmtree(F.root, ignore_errors=True)
    print("-" * 70)
    if _FAIL:
        print("FAILED %d of %d" % (len(_FAIL), len(_PASS) + len(_FAIL)))
        return 1
    print("ALL %d CHECKS PASSED" % len(_PASS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
