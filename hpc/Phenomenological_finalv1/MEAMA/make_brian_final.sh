#!/bin/bash
# =============================================================================
# make_brian_final.sh -- build a clean `brian_final` conda environment for the
# MEAMA pipeline, replacing the damaged `brian_env` (whose python3.11 stdlib was
# found incomplete: lib/python3.11/urllib/ absent entirely).
#
#     bash make_brian_final.sh              # build it
#     DRYRUN=1 bash make_brian_final.sh     # print what would run, do nothing
#     ENV_NAME=brian_test bash make_brian_final.sh
#
# This script does NOT delete brian_env. Keep the broken environment until
# brian_final passes env_check.py -- its conda-meta/ directory is the only
# record of which package versions produced the existing campaigns, and
# `conda list` still reads it even though its interpreter is broken.
#
# NETWORK. Package downloads need the proxy on this cluster:  module load proxy
# The script loads it if the module command exists, and warns if it cannot.
# =============================================================================
set -u

ENV_NAME="${ENV_NAME:-brian_final}"
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# PROFILE -- brian2 and python versions are COUPLED, pick one coherent pair.
#
#   brian2 >= 2.10 requires python >= 3.12   (confirmed from pip's own resolver
#                                             2026-08-31: "2.10.0/2.10.1
#                                             Requires-Python >=3.12")
#   brian2 <= 2.9  breaks against numpy 2.x  (calls the ndarray.ptp method,
#                                             removed in numpy 2.0)
#
# So there is no valid python-3.11 + numpy-2.x combination. The two coherent
# choices are:
#
#   PROFILE=modern (default)  python 3.12, numpy 2.x, brian2 >= 2.10
#       The stack verified end to end for this pipeline: python 3.12.3,
#       numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, sklearn 1.8.0,
#       matplotlib 3.10.8, seaborn 0.13.2, brian2 2.10.1 -- env_check.py
#       48/48 including a real cpp_standalone compile-and-run.
#
#   PROFILE=legacy            python 3.11, numpy < 2, brian2 == 2.9.0
#       Closer to the original brian_env (python 3.11). Use this if you need
#       MEAMA to sit on the same numerical stack as the earlier campaigns.
#       Check what the old environment actually had first:
#           grep -E '^(brian2|numpy|scipy) ' brian_env_versions_*.txt
#
# The numpy constraint is applied at CONDA CREATE time, never with a later
# `pip install numpy<2`: conda-forge builds scipy/pandas/scikit-learn against
# whichever numpy is resolved here, and downgrading numpy underneath them
# afterwards is an ABI mismatch that fails in far more confusing ways than a
# clean version error.
# ---------------------------------------------------------------------------
PROFILE="${PROFILE:-modern}"
case "$PROFILE" in
    modern)
        PY_VER="${PY_VER:-3.12}"
        NUMPY_SPEC="numpy"
        BRIAN2_SPEC="${BRIAN2_SPEC:-brian2>=2.10}"
        ;;
    legacy)
        PY_VER="${PY_VER:-3.11}"
        NUMPY_SPEC="numpy<2"
        BRIAN2_SPEC="${BRIAN2_SPEC:-brian2==2.9.0}"
        ;;
    *)
        echo "ERROR: PROFILE must be 'modern' or 'legacy', got '${PROFILE}'." >&2
        exit 2
        ;;
esac
echo "[env] PROFILE=${PROFILE}: python ${PY_VER}, ${NUMPY_SPEC}, ${BRIAN2_SPEC}"

echo "============================================================"
echo "Building conda environment : ${ENV_NAME}   (python ${PY_VER})"
echo "============================================================"

# --- proxy (needed for any download on this cluster) ----------------------
if command -v module >/dev/null 2>&1; then
    module load proxy 2>/dev/null && echo "[env] module load proxy: ok" \
        || echo "[env] WARNING: 'module load proxy' failed; downloads may hang."
else
    echo "[env] WARNING: no 'module' command; assuming network is reachable."
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not on PATH. Load it first, e.g.: module load python" >&2
    exit 2
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- refuse to clobber an existing environment ----------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "ERROR: an environment named '${ENV_NAME}' already exists." >&2
    echo "       Remove it deliberately first:  conda env remove -n ${ENV_NAME}" >&2
    echo "       (or set ENV_NAME=<other> to build under a different name)" >&2
    exit 3
fi

# --- record what the OLD environment had, for provenance ------------------
# `conda list` reads conda-meta/*.json, not the interpreter, so this still
# works against a broken env. Do this BEFORE removing anything.
OLD_ENV="${OLD_ENV:-brian_env}"
SPEC_OUT="${_HERE}/${OLD_ENV}_versions_$(date -u +%Y%m%dT%H%M%SZ).txt"
if conda env list | awk '{print $1}' | grep -qx "${OLD_ENV}"; then
    if conda list -n "${OLD_ENV}" > "${SPEC_OUT}" 2>/dev/null; then
        echo "[env] recorded ${OLD_ENV} package versions -> ${SPEC_OUT}"
        echo "[env] key versions in the old environment:"
        grep -E '^(python|numpy|scipy|pandas|matplotlib|seaborn|brian2|scikit-learn|cython)\s' \
            "${SPEC_OUT}" | sed 's/^/        /' || echo "        (none matched)"
    else
        echo "[env] WARNING: could not read ${OLD_ENV} package list."
    fi
else
    echo "[env] note: no environment named ${OLD_ENV}; skipping provenance capture."
fi

echo
echo "IMPORTANT -- version reproducibility:"
echo "  The campaigns already on disk were produced by ${OLD_ENV}. If you need"
echo "  the new environment to reproduce them numerically, pin brian2/numpy to"
echo "  the versions listed above instead of taking whatever is current."
echo "  For MEAMA (a NEW campaign family) current versions are fine, but the"
echo "  version set becomes part of that campaign's provenance either way."
echo

PKGS=(
    "python=${PY_VER}"
    "${NUMPY_SPEC}" scipy pandas matplotlib seaborn scikit-learn
    cython            # brian2 code generation
    gxx_linux-64      # C++ compiler inside the env, so cpp_standalone does not
                      # depend on whichever gcc a compute node happens to expose
    pip
)

echo "conda create -n ${ENV_NAME} -c conda-forge ${PKGS[*]}"
echo "then: pip install brian2"
echo

if [ "${DRYRUN:-0}" = "1" ]; then
    echo "[dry run] nothing executed."
    exit 0
fi

conda create -y -n "${ENV_NAME}" -c conda-forge "${PKGS[@]}" || {
    echo "ERROR: conda create failed." >&2; exit 4; }

# `set -u` is on for this whole script, but `conda activate` sources
# THIRD-PARTY hook scripts (etc/conda/activate.d/*.sh) that were not written
# to be nounset-safe. Observed on this cluster 2026-08-31:
#   .../deactivate.d/geotiff-deactivate.sh: line 5:
#   _CONDA_SET_GEOTIFF_CSV: unbound variable
# That belongs to a GDAL/GEOTIFF package in the base miniconda install,
# unrelated to this pipeline; it fires because activating ${ENV_NAME}
# deactivates whatever was already active in the invoking shell first.
set +u
conda activate "${ENV_NAME}"
_activate_rc=$?
set -u
if [ $_activate_rc -ne 0 ]; then
    echo "ERROR: could not activate ${ENV_NAME}." >&2; exit 5
fi

# DO NOT trust bare `python`/`pip` on PATH from here on. Confirmed 2026-08-31
# on this cluster: after a successful `conda activate`, CONDA_PREFIX resolves
# correctly to the new env, but bare `python` can still resolve to a
# module-loaded system interpreter instead (observed: /archive/apps/Python/
# 3.12.13/bin/python winning over the env's own 3.11). CONDA_PREFIX itself IS
# reliable -- conda sets it directly, independent of PATH search order -- so
# resolve the interpreter from it explicitly and use ONLY that path.
ENV_PY="${CONDA_PREFIX}/bin/python"
if [ ! -x "$ENV_PY" ]; then
    echo "ERROR: ${ENV_PY} does not exist. conda create may have failed" >&2
    echo "       silently, or CONDA_PREFIX (${CONDA_PREFIX}) is wrong." >&2
    exit 5
fi
echo "[env] resolved interpreter: ${ENV_PY}"

# brian2 from pip: conda-forge's build sometimes lags, and pip resolves against
# the numpy already installed above. Using ENV_PY -m pip (not bare `pip`) for
# the same PATH-reliability reason as above.
#
# VERSION FLOOR IS LOAD-BEARING, do not drop it. NumPy 2.0 removed the
# `ndarray.ptp` METHOD (the `np.ptp` function still exists). brian2 2.9.0
# calls `np.ndarray.ptp` at import time, so brian2 <= 2.9 dies with
#   AttributeError: type object 'numpy.ndarray' has no attribute 'ptp'
# against the numpy 2.x that conda-forge installs above. Observed here
# 2026-08-31: an unpinned `pip install brian2` resolved to 2.9.0 and broke
# exactly this way. brian2 2.10.x is numpy-2 clean (verified: 2.10.1 with
# numpy 2.4.4, full cpp_standalone compile-and-run).
echo "[env] installing ${BRIAN2_SPEC}"
"$ENV_PY" -m pip install --no-cache-dir "$BRIAN2_SPEC" || {
    echo "ERROR: pip install ${BRIAN2_SPEC} failed." >&2
    echo "       brian2 >= 2.10 needs python >= 3.12; this env is ${PY_VER}." >&2
    echo "       Either rebuild with PROFILE=modern (python 3.12), or use" >&2
    echo "       PROFILE=legacy (python 3.11 + numpy<2 + brian2 2.9.0)." >&2
    echo "       See what pip can actually see:" >&2
    echo "         ${ENV_PY} -m pip index versions brian2" >&2
    exit 6; }

# Fail here, not three checks later, if the pair is still incompatible.
"$ENV_PY" - <<'PYVER' || { echo "ERROR: brian2/numpy pair is not usable." >&2; exit 6; }
import numpy, brian2
print(f"[env] brian2 {brian2.__version__} imported cleanly against numpy {numpy.__version__}")
PYVER

echo
echo "============================================================"
echo "Installed versions in ${ENV_NAME}:"
"$ENV_PY" - <<'PY'
import importlib
for m in ('numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'brian2'):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:<14} IMPORT FAILED: {e}")
import sys
print(f"  python         {sys.version.split()[0]}  ({sys.executable})")
PY
echo "============================================================"
echo
echo "NEXT -- verify before trusting it. Use the SAME absolute path as above,"
echo "not bare 'python' -- bare python can still resolve to a module-loaded"
echo "system interpreter even after 'conda activate' succeeds (see the"
echo "comment above ENV_PY in this script for what that looked like)."
echo "    cd ${_HERE}"
echo "    ${ENV_PY} env_check.py          # expect 47 passed, 0 failed"
echo
echo "Nothing else needs editing: bench_sizing.sh and submit_MEAMA.sh read"
echo "CONDA_ENV (default brian_final) rather than a hardcoded name, and both"
echo "resolve their own interpreter the same way env_check.py was just run --"
echo "via \\\${CONDA_PREFIX}/bin/python, never bare 'python'. To use a"
echo "different environment name for one run:  CONDA_ENV=other bash launch_bench.sh"
