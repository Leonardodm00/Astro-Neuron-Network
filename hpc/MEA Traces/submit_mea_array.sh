#!/bin/bash
#PBS -S /bin/bash
#PBS -N mea_array
#PBS -k eo
#PBS -l walltime=06:00:00
##########################################################################
# Array WORKER for MEA post-processing across many campaigns/sweeps.
#
# DO NOT submit this directly -- submit it via launch_mea_array.sh, which
# builds the manifest, computes the array range, and calls qsub -J for you.
#
# Each array member reads exactly ONE line from MANIFEST (its own index + 1,
# manifests are 1-indexed by `sed`, PBS_ARRAY_INDEX is 0-indexed) and runs
# process_campaign.py on just that campaign/sweep directory. This is
# POST-processing (numpy + scipy only, no Brian2, no compilation), so it
# needs far less walltime than the original simulation sweep.
#
# REQUIRED -v variables (set by launch_mea_array.sh):
#     MANIFEST   path to the manifest built by build_mea_manifest.py
#                (tab-separated: campaign_dir <TAB> out_dir, one per line)
#     LIB        path to a pre-built eap_library.npz (build this ONCE before
#                submitting the array -- see launch_mea_array.sh -- so that
#                dozens/hundreds of array members don't each pay the ~30s
#                HH-integration cost separately)
# OPTIONAL -v variables:
#     ENV_PREFIX full path to the python environment to use (preferred; see
#                the environment block below). Set automatically by
#                launch_mea_array.sh from --conda-prefix / $CONDA_PREFIX.
#     CONDA_ENV  conda environment NAME, as a fallback to ENV_PREFIX.
#     EXTRA_ARGS extra flags passed through to process_campaign.py verbatim,
#                e.g. "--plots --limit_iters 5" for a partial/inspectable run
##########################################################################

set -euo pipefail

if [ -z "${MANIFEST:-}" ] || [ -z "${LIB:-}" ]; then
    echo "ERROR: submit via launch_mea_array.sh (MANIFEST and LIB must be" >&2
    echo "       passed with qsub -v). Refusing to run un-configured." >&2
    exit 2
fi
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2
    exit 2
fi

IDX="${PBS_ARRAY_INDEX:-0}"           # 0 if run as a non-array single task
LINE="$((IDX + 1))"
ROW="$(sed -n "${LINE}p" "$MANIFEST")"
if [ -z "$ROW" ]; then
    echo "ERROR: manifest has no line $LINE (index $IDX) -- array range and" >&2
    echo "       manifest length must match. Manifest: $MANIFEST" >&2
    exit 2
fi

CAMPAIGN="$(printf '%s' "$ROW" | cut -f1)"
OUT="$(printf '%s' "$ROW" | cut -f2)"
if [ -z "$CAMPAIGN" ] || [ -z "$OUT" ]; then
    echo "ERROR: malformed manifest line $LINE: '$ROW'" >&2
    exit 2
fi

cd "${PBS_O_WORKDIR:-.}"

# --- environment ---------------------------------------------------------
# A compute node does NOT inherit the conda environment from your login
# shell: it starts with the system PATH, whose /usr/bin/python3 on a
# RHEL/CentOS 7 cluster is typically Python 3.6 -- too old for this pipeline
# (dataclasses landed in the stdlib in 3.7) and usually without numpy/scipy.
# So the environment is activated HERE, inside the job.
#
#   DEFAULT_ENV_NAME  <-- EDIT THIS if your environment is ever renamed.
# It can also be overridden per-submission without editing this file, via
# qsub -v: ENV_PREFIX=/full/path/to/env (preferred, no conda init needed) or
# CONDA_ENV=name. The launchers pass those automatically when given.
DEFAULT_ENV_NAME="brian_env"

# Try every reasonable way to put an environment's python on the PATH.
# Returns 0 on success. Kept deliberately exhaustive because which of these
# works depends on how conda was installed on the node, and a job that picks
# the wrong interpreter fails in a confusing way much later.
activate_env_by_name() {
    envname="$1"

    # 1. conda already resolvable, with a working shell hook
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook 2>/dev/null)" >/dev/null 2>&1 || true
        conda activate "$envname" >/dev/null 2>&1 && return 0
        source activate "$envname" >/dev/null 2>&1 && return 0
    fi

    # 2. environment-modules (note: `module` is often a shell FUNCTION, so
    #    `command -v` can miss it -- `type` catches both)
    if type module >/dev/null 2>&1; then
        module load anaconda3 >/dev/null 2>&1 \
            || module load conda >/dev/null 2>&1 \
            || module load python >/dev/null 2>&1 || true
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook 2>/dev/null)" >/dev/null 2>&1 || true
            conda activate "$envname" >/dev/null 2>&1 && return 0
            source activate "$envname" >/dev/null 2>&1 && return 0
        fi
    fi

    # 3. source conda.sh directly from the usual install locations
    for csh in "${HOME}/miniconda3/etc/profile.d/conda.sh" \
               "${HOME}/anaconda3/etc/profile.d/conda.sh" \
               "${HOME}/.conda/etc/profile.d/conda.sh" \
               "/opt/conda/etc/profile.d/conda.sh" \
               "/usr/local/anaconda3/etc/profile.d/conda.sh" \
               "/usr/local/miniconda3/etc/profile.d/conda.sh"; do
        if [ -f "$csh" ]; then
            # shellcheck disable=SC1090
            source "$csh" >/dev/null 2>&1 || continue
            conda activate "$envname" >/dev/null 2>&1 && return 0
        fi
    done

    # 4. last resort: find the env directory itself and prepend its bin/.
    #    This needs no conda machinery at all -- if the interpreter is there,
    #    putting it first on PATH is sufficient.
    for pfx in "${HOME}/.conda/envs/${envname}" \
               "${HOME}/miniconda3/envs/${envname}" \
               "${HOME}/anaconda3/envs/${envname}" \
               "/opt/conda/envs/${envname}"; do
        if [ -x "${pfx}/bin/python3" ]; then
            export PATH="${pfx}/bin:${PATH}"
            return 0
        fi
    done

    return 1
}

if [ -n "${ENV_PREFIX:-}" ]; then
    echo "[mea-array] activating env by prefix: ${ENV_PREFIX}"
    export PATH="${ENV_PREFIX}/bin:${PATH}"
else
    ENV_NAME="${CONDA_ENV:-$DEFAULT_ENV_NAME}"
    echo "[mea-array] activating env by name: ${ENV_NAME}"
    if activate_env_by_name "${ENV_NAME}"; then
        echo "[mea-array] env activated: $(command -v python3)"
    else
        echo "[mea-array] WARNING: could not activate '${ENV_NAME}' by any known" >&2
        echo "[mea-array]          method; falling through to the default python3." >&2
        echo "[mea-array]          If this fails below, pass the full path with" >&2
        echo "[mea-array]          launch_mea_array.sh --conda-prefix /path/to/env" >&2
    fi
fi

# --- preflight: fail FAST and CLEARLY, before any real work --------------
# Without this, a wrong interpreter surfaces as a bare ModuleNotFoundError
# from deep inside an import chain, which is a slow and confusing way to
# learn that the environment was never activated.
python3 - <<'PYEOF' || exit 3
import sys
ok = True
v = sys.version_info
print('[mea-array] python   : %d.%d.%d (%s)' % (v[0], v[1], v[2], sys.executable))
if v < (3, 7):
    sys.stderr.write(
        'ERROR: Python %d.%d is too old for this pipeline (needs >= 3.7;\n'
        '       the dataclasses module entered the stdlib in 3.7).\n'
        '       The job did not pick up the intended environment. Check\n'
        '       DEFAULT_ENV_NAME at the top of submit_mea_array.sh, or pass\n'
        '       launch_mea_array.sh --conda-prefix /full/path/to/env.\n'
        % (v[0], v[1]))
    ok = False
for mod in ('numpy', 'scipy'):
    try:
        m = __import__(mod)
        print('[mea-array] %-8s : %s' % (mod, getattr(m, '__version__', '?')))
    except ImportError:
        sys.stderr.write('ERROR: %s is not importable in this environment.\n' % mod)
        ok = False
sys.exit(0 if ok else 1)
PYEOF

NCPUS="${PBS_NCPUS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[mea-array] index    : $IDX (manifest line $LINE)"
echo "[mea-array] campaign : $CAMPAIGN"
echo "[mea-array] out      : $OUT"
echo "[mea-array] library  : $LIB"
echo "[mea-array] workers  : $NCPUS"
echo "[mea-array] extra    : ${EXTRA_ARGS:-<none>}"

python3 process_campaign.py \
    --campaign "$CAMPAIGN" \
    --out "$OUT" \
    --library "$LIB" \
    --workers "$NCPUS" \
    $EXTRA_ARGS

echo "[mea-array] index $IDX done."
