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
PY_VER="${PY_VER:-3.11}"
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    numpy scipy pandas matplotlib seaborn scikit-learn
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

conda activate "${ENV_NAME}" || {
    echo "ERROR: could not activate ${ENV_NAME}." >&2; exit 5; }

# brian2 from pip: conda-forge's build sometimes lags, and pip resolves against
# the numpy already installed above.
pip install --no-cache-dir brian2 || {
    echo "ERROR: pip install brian2 failed." >&2; exit 6; }

echo
echo "============================================================"
echo "Installed versions in ${ENV_NAME}:"
python - <<'PY'
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
echo "NEXT -- verify before trusting it:"
echo "    conda activate ${ENV_NAME}"
echo "    cd ${_HERE}"
echo "    python env_check.py          # expect 47 passed, 0 failed"
echo
echo "Then point the job scripts at it:"
echo "    sed -i 's/conda activate brian_env/conda activate ${ENV_NAME}/' \\"
echo "        ${_HERE}/bench_sizing.sh ${_HERE}/submit_MEAMA.sh"
echo "    grep -n 'conda activate' ${_HERE}/*.sh"
