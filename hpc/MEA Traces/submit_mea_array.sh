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
#     ENV_PREFIX full path to the python environment to use (see the
#                environment block below). Set by launch_mea_array.sh from
#                --conda-prefix, or from the launching shell's $CONDA_PREFIX
#                when neither --conda-prefix nor --conda-env was given.
#     CONDA_ENV  conda environment NAME (launch_mea_array.sh --conda-env).
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
# qsub -v: ENV_PREFIX=/full/path/to/env or CONDA_ENV=name. The launchers pass
# those automatically when given. Both are ACTIVATED through conda when conda
# is available, so the env's activate.d hooks run -- on davinci they set the
# LD_LIBRARY_PATH scipy needs; a bare PATH prepend is only the fallback.
DEFAULT_ENV_NAME="brian_env"
# Where the cluster's own conda lives when it is neither on PATH nor a
# module. davinci-1: the "base" row of `conda env list` (2026-10-01). Edit
# for another cluster; a path that does not exist is skipped.
KNOWN_CONDA_BASE="${KNOWN_CONDA_BASE:-/archive/apps/miniconda/miniconda3/py312_2}"

# Try every reasonable way to put an environment's python on the PATH.
# Returns 0 on success. Kept deliberately exhaustive because which of these
# works depends on how conda was installed on the node, and a job that picks
# the wrong interpreter fails in a confusing way much later. $1 is an env
# NAME or a full env PATH (conda activate accepts both).
_activate_env_by_name_unguarded() {
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
        module load miniconda3 >/dev/null 2>&1 \
            || module load anaconda3 >/dev/null 2>&1 \
            || module load conda >/dev/null 2>&1 \
            || module load python >/dev/null 2>&1 || true
        if command -v conda >/dev/null 2>&1; then
            eval "$(conda shell.bash hook 2>/dev/null)" >/dev/null 2>&1 || true
            conda activate "$envname" >/dev/null 2>&1 && return 0
            source activate "$envname" >/dev/null 2>&1 && return 0
        fi
    fi

    # 3. source conda.sh directly from the usual install locations. First
    #    the cluster's own base (davinci-1: the "base" row of `conda env
    #    list`; neither a module nor on a compute node's PATH there, which
    #    is why every job of 2026-08 fell through to the PATH prepend), then
    #    the generic ones.
    for csh in "${KNOWN_CONDA_BASE}/etc/profile.d/conda.sh" \
               "${HOME}/miniconda3/etc/profile.d/conda.sh" \
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
               "${KNOWN_CONDA_BASE}/envs/${envname}" \
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

# conda's hook, `conda activate` and the env's activate.d / deactivate.d
# scripts read variables that are unset in a batch shell. Under this
# script's `set -u` the shell dies INSIDE the eval -- silently, since every
# line above discards its output, and `|| true` cannot catch an unbound-
# variable abort. So -u is lifted around the whole attempt and restored after
# (2026-09-28; the same guard as the conda block of every job script in
# Simulation-Based-Inference/hpc).
activate_env_by_name() {
    local _had_u=0 _rc=0
    case "$-" in *u*) _had_u=1 ;; esac
    set +u
    if _activate_env_by_name_unguarded "$1"; then _rc=0; else _rc=$?; fi
    if [ "$_had_u" -eq 1 ]; then set -u; fi
    return "$_rc"
}

if [ -n "${ENV_PREFIX:-}" ]; then
    echo "[mea-array] activating env by prefix: ${ENV_PREFIX}"
    if activate_env_by_name "${ENV_PREFIX}"; then
        echo "[mea-array] env activated: $(command -v python3)"
    else
        echo "[mea-array] WARNING: conda could not activate ${ENV_PREFIX}; putting" >&2
        echo "[mea-array]          its bin/ first on PATH instead. Its activate.d" >&2
        echo "[mea-array]          hooks do NOT run; the preflight below checks scipy." >&2
        export PATH="${ENV_PREFIX}/bin:${PATH}"
    fi
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

# --- provenance: one mea_env.json per work unit (2026-10-01) ---------------
# process_campaign.py records its own knobs in every mea_iter_*.npz
# (meta_json) and nothing else: not the interpreter, not numpy / scipy, not
# the template library, not which copy of the code ran. This file records
# them, next to the mea_manifest.json the run will write, so that a root of
# detections can say what produced it. Written BEFORE the run (a killed task
# still leaves it); the run's completion is mea_manifest.json.
mkdir -p "$OUT"
MEA_ENV_JSON="${OUT}/mea_env.json" MEA_CAMPAIGN="$CAMPAIGN" MEA_LIB="$LIB" \
MEA_EXTRA="$EXTRA_ARGS" MEA_INDEX="$IDX" MEA_WORKERS="$NCPUS" \
MEA_ENV_NAME="${CONDA_DEFAULT_ENV:-}" MEA_ENV_PREFIX="${CONDA_PREFIX:-${ENV_PREFIX:-}}" \
python3 - <<'PYEOF' || echo "[mea-array] WARNING: mea_env.json not written" >&2
import hashlib, json, os, platform, socket, sys, time
def sha(p):
    h = hashlib.sha256()
    try:
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None
import numpy, scipy
rec = {
    'record': 'mea_env', 'version': 1,
    'written': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'host': socket.gethostname(),
    'pbs_jobid': os.environ.get('PBS_JOBID'),
    'array_index': int(os.environ['MEA_INDEX']),
    'campaign': os.environ['MEA_CAMPAIGN'],
    'out': os.path.dirname(os.environ['MEA_ENV_JSON']),
    'workers': int(os.environ['MEA_WORKERS']),
    'extra_args': os.environ['MEA_EXTRA'],
    'env_name': os.environ['MEA_ENV_NAME'] or None,
    'env_prefix': os.environ['MEA_ENV_PREFIX'] or None,
    'python': platform.python_version(),
    'executable': sys.executable,
    'numpy': numpy.__version__,
    'scipy': scipy.__version__,
    'library': os.path.abspath(os.environ['MEA_LIB']),
    'library_sha256': sha(os.environ['MEA_LIB']),
    'tools_dir': os.getcwd(),
    'tools_sha256': {n: sha(n) for n in (
        'process_campaign.py', 'mea_probe.py', 'mea_detection.py',
        'mea_synthesis.py', 'mea_plots.py', 'eap_template_library.py',
        'submit_mea_array.sh')},
}
tmp = os.environ['MEA_ENV_JSON'] + '.tmp'
with open(tmp, 'w') as f:
    json.dump(rec, f, indent=2, sort_keys=True)
    f.write('\n')
os.replace(tmp, os.environ['MEA_ENV_JSON'])
print('[mea-array] env record: ' + os.environ['MEA_ENV_JSON'])
PYEOF

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
