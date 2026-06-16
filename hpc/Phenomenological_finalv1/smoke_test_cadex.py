#!/usr/bin/env python3
"""
smoke_test_cadex.py — single nominal-value smoke test for the CAdEx "HH-gap"
refit + swept `I_inj` excitability-spread axis.

WHAT IT DOES (two independent stages):

  Stage 1  REGISTRY ASSERTS  (no brian2 needed)
      Imports HPC_main_sweep (which imports HPC_single_run). The act of importing
      fires every module-level invariant added by this change:
          * PARAM_BOUNDS.shape == (35, 2)
          * groups + frozen tile {0..34} exactly  (tiling assert)
          * theta <-> natural round-trip at both bound edges
      then prints the sweep-group accounting and confirms:
          * len(PARAM_NAMES/PARAM_UNITS/NOMINAL_PARAMS) == 35
          * NOMINAL[13]==VT==-48.0, NOMINAL[34]==I_inj==15.0
          * I_inj is FREE under neuron_synapse; DeltaT/VT/gL are FROZEN
          * I_inj is linear (not in LOG_PARAMS)

  Stage 2  SINGLE NOMINAL RUN  (needs brian2 / the `brian_env` on davinci-1)
      Runs ONE short simulation at nominal parameters through the production
      single-run driver (HPC_single_run.py), MODE=Neuronal, flat connectivity.
      PASS if:
          * the driver exits 0,
          * >= 1 iter_*.npz is produced,
          * the matching iter_*.json sidecar carries the three new dispersion
            monitors: across_cell_rate_cv, frac_active, mean_isi_cv,
          * mean_FR_Hz is finite (and, as an ignition sanity print, ideally > 0).

USAGE (swift):
      # on davinci-1, inside brian_env, from the dir holding the .py library files:
      conda activate brian_env
      python smoke_test_cadex.py                       # full test (both stages)
      python smoke_test_cadex.py --simtime 3 --Nn 80   # quicker
      python smoke_test_cadex.py --registry-only       # Stage 1 only (no brian2)
      python smoke_test_cadex.py --keep                # keep the scratch out_dir

EXIT CODE: 0 = all stages PASS; 1 = any failure.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# --------------------------------------------------------------------------- #
# Stage 1 — registry / bounds invariants (pure-python; no brian2)
# --------------------------------------------------------------------------- #
def stage1_registry(lib_dir: Path) -> bool:
    sys.path.insert(0, str(lib_dir))
    try:
        import matplotlib            # noqa: F401  (HPC_* import it at module top)
        matplotlib.use("Agg")
    except Exception:
        pass

    # Importing fires the module-level asserts (tiling, shape, round-trip).
    import HPC_single_run as s
    import HPC_main_sweep as m

    ok = True

    def check(label, cond, got=None, exp=None):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        extra = "" if got is None else f"   got={got!r} exp={exp!r}"
        print(f"  [{status}] {label}{extra}")
        ok = ok and bool(cond)

    print("Stage 1 — registry / bounds invariants")
    check("len(PARAM_NAMES)  == 35", len(s.PARAM_NAMES) == 35, len(s.PARAM_NAMES), 35)
    check("len(PARAM_UNITS)  == 35", len(s.PARAM_UNITS) == 35, len(s.PARAM_UNITS), 35)
    check("len(NOMINAL)      == 35", len(s.NOMINAL_PARAMS) == 35, len(s.NOMINAL_PARAMS), 35)
    check("PARAM_NAMES[34] == 'I_inj'", s.PARAM_NAMES[34] == "I_inj", s.PARAM_NAMES[34], "I_inj")
    check("PARAM_UNITS[34] == 'pA'", s.PARAM_UNITS[34] == "pA", s.PARAM_UNITS[34], "pA")
    check("NOMINAL[13] (VT)   == -48.0", float(s.NOMINAL_PARAMS[13]) == -48.0,
          float(s.NOMINAL_PARAMS[13]), -48.0)
    check("NOMINAL[34] (I_inj)== 15.0", float(s.NOMINAL_PARAMS[34]) == 15.0,
          float(s.NOMINAL_PARAMS[34]), 15.0)
    check("FROZEN_PARAMS == [DeltaT, VT, gL]", s.FROZEN_PARAMS == ["DeltaT", "VT", "gL"],
          s.FROZEN_PARAMS, ["DeltaT", "VT", "gL"])
    check("N_DIMS == 35", m.N_DIMS == 35, m.N_DIMS, 35)
    check("PARAM_BOUNDS.shape == (35,2)", tuple(m.PARAM_BOUNDS.shape) == (35, 2),
          tuple(m.PARAM_BOUNDS.shape), (35, 2))
    check("I_inj bounds == (0.0, 50.0)", tuple(map(float, m.PARAM_BOUNDS[34])) == (0.0, 50.0),
          tuple(map(float, m.PARAM_BOUNDS[34])), (0.0, 50.0))
    check("I_inj linear (34 not in LOG_PARAMS)", 34 not in m.LOG_PARAMS)

    allg = s.SWEEP_GROUPS["all"]
    check("|SWEEP_GROUPS['all']| == 32 (frozen excluded)", len(allg) == 32, len(allg), 32)
    check("12/13/30 absent from 'all'", all(k not in allg for k in (12, 13, 30)))

    ns = s.resolve_sweep_group("neuron_synapse")
    names = [s.PARAM_NAMES[i] for i in ns]
    check("|neuron_synapse free| == 22", len(ns) == 22, len(ns), 22)
    check("I_inj IS free under neuron_synapse", "I_inj" in names)
    check("DeltaT/VT/gL NOT free under neuron_synapse",
          not any(n in names for n in ("DeltaT", "VT", "gL")))

    print(f"  -> Stage 1 {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Stage 2 — single nominal run through the production driver (needs brian2)
# --------------------------------------------------------------------------- #
def stage2_single_run(lib_dir: Path, simtime: float, Nn: int,
                      conn_prob: float, keep: bool) -> bool:
    driver = lib_dir / "HPC_single_run.py"
    if not driver.is_file():
        print(f"  [FAIL] driver not found: {driver}")
        return False

    out_dir = Path(tempfile.mkdtemp(prefix="smoke_cadex_"))
    print("Stage 2 — single nominal run (MODE=Neuronal, flat connectivity)")
    print(f"  driver   : {driver}")
    print(f"  out_dir  : {out_dir}")
    print(f"  simtime  : {simtime} s   Nn : {Nn}   conn_prob : {conn_prob}")
    print("  (all CAdEx/synapse params left at NOMINAL via the driver defaults)")

    cmd = [
        sys.executable, str(driver),
        "--out_dir", str(out_dir),
        "--lib_dir", str(lib_dir),
        "--mode", "Neuronal",
        "--simtime", str(simtime),
        "--Nn", str(Nn),
        "--conn_prob", str(conn_prob),
        "--dpi", "80",
    ]
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  [FAIL] driver exited {proc.returncode}")
        tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
        print("  --- driver stderr (tail) ---\n" + tail)
        if "No module named 'brian2'" in proc.stderr:
            print("  NOTE: Stage 2 requires brian2 (run inside `brian_env` on davinci-1).")
        return False

    npzs = sorted(out_dir.rglob("iter_*.npz"))
    jsons = sorted(out_dir.rglob("iter_*.json"))
    ok = True

    def check(label, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        ok = ok and bool(cond)

    check(f">= 1 iter_*.npz produced (found {len(npzs)})", len(npzs) >= 1)
    check(f">= 1 iter_*.json sidecar produced (found {len(jsons)})", len(jsons) >= 1)

    if jsons:
        with open(jsons[0]) as fh:
            d = json.load(fh)
        for key in ("across_cell_rate_cv", "frac_active", "mean_isi_cv"):
            check(f"sidecar carries '{key}'", key in d)
        mfr = d.get("mean_FR_Hz")
        import math
        check("mean_FR_Hz is finite", isinstance(mfr, (int, float)) and math.isfinite(mfr))
        print(f"  ignition: mean_FR_Hz = {mfr:.4f} Hz   "
              f"across_cell_rate_cv = {d.get('across_cell_rate_cv')}   "
              f"mean_isi_cv = {d.get('mean_isi_cv')}")
        if mfr == 0.0:
            print("  WARNING: network silent at nominal (mean_FR_Hz == 0). The HH-gap "
                  "refit should ignite at nominal; investigate before launching.")

    if keep:
        print(f"  (kept scratch: {out_dir})")
    else:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)

    print(f"  -> Stage 2 {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib_dir", default=None,
                    help="dir holding ASD_fun_BD_cpp.py / HPC_single_run.py "
                         "(default: this script's dir)")
    ap.add_argument("--simtime", type=float, default=3.0, help="seconds (short smoke)")
    ap.add_argument("--Nn", type=int, default=100, help="number of neurons")
    ap.add_argument("--conn_prob", type=float, default=0.2, help="flat Bernoulli conn prob")
    ap.add_argument("--registry-only", action="store_true",
                    help="run Stage 1 only (no brian2 required)")
    ap.add_argument("--keep", action="store_true", help="keep the scratch out_dir")
    args = ap.parse_args()

    lib_dir = Path(args.lib_dir).resolve() if args.lib_dir \
        else Path(__file__).resolve().parent

    print("=" * 64)
    print("smoke_test_cadex.py   lib_dir =", lib_dir)
    print("=" * 64)

    s1 = stage1_registry(lib_dir)
    if args.registry_only:
        print("\nRESULT:", "PASS" if s1 else "FAIL", "(Stage 1 only)")
        return 0 if s1 else 1

    print()
    s2 = stage2_single_run(lib_dir, args.simtime, args.Nn, args.conn_prob, args.keep)

    allp = s1 and s2
    print("\n" + "=" * 64)
    print("RESULT:", "ALL PASS" if allp else "FAILURE(S) — see above")
    print("=" * 64)
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
