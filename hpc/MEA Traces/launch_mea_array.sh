#!/bin/bash
##########################################################################
# launch_mea_array.sh -- one-command, non-interactive launch of the MEA
# pipeline across MANY campaigns/sweeps.
#
# WHAT IT DOES
#   1. Builds a manifest of every sweep_*_task* work unit found under the
#      given campaign root(s)/glob(s), via build_mea_manifest.py.
#   2. Submits ONE PBS array job (qsub -J) that processes them all, one work
#      unit per array member, via submit_mea_array.sh.
#   3. Falls back to a single non-array qsub if there is only ONE work unit
#      -- PBS Pro rejects a single-element array (same convention as this
#      repo's own launch_campaign.sh).
#
# USAGE
#   ./launch_mea_array.sh \
#       --out-root  /path/to/mea_out \
#       --campaign-root /path/campaign_cadex_rho1300v3 \
#       --campaign-root /path/campaign_cadex_rho2000v1
#
#   # or sweep every campaign_* directory under Main/ in one call:
#   ./launch_mea_array.sh \
#       --out-root /path/to/mea_out \
#       --glob '/davinci-1/home/ldellamea/ANN/Phenomenological/Main/campaign_*'
#
# OPTIONS
#   --out-root DIR        required; see build_mea_manifest.py for the layout
#   --campaign-root DIR    repeatable, forwarded to build_mea_manifest.py
#   --glob PATTERN         repeatable, forwarded to build_mea_manifest.py
#                          (quote it -- do not let your shell expand it)
#   --lib PATH             pre-built eap_library.npz (default: ./eap_library.npz;
#                          built once, synchronously, before submitting, so
#                          array members never race to generate it)
#   --queue NAME            PBS queue (default: cpu)
#   --ncpus N               cores per array member (default: 48)
#   --walltime HH:MM:SS     per array member (default: 06:00:00)
#   --conda-prefix PATH     full path to the python env the JOBS should use.
#                          OPTIONAL: submit_mea_array.sh already activates a
#                          built-in default environment internally (see
#                          DEFAULT_ENV_NAME there). Use this only to override
#                          that for one submission. Auto-filled from
#                          $CONDA_PREFIX if you launch with an env active.
#   --conda-env NAME        env NAME instead of a full path (same purpose)
#   --concurrency N         max array members running at once (default: 20;
#                          this is the %N throttle in -J "0-M%N", keep it
#                          within your fairshare/queue limits)
#   --skip-done             forwarded to build_mea_manifest.py
#   --extra-args "..."      forwarded verbatim to process_campaign.py on
#                          every array member, e.g. "--plots" or
#                          "--limit_iters 5" for a partial/inspectable pass
#   --dry-run                build the manifest and print the qsub command,
#                          but do not actually submit
##########################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_ROOT=""
LIB="./eap_library.npz"
QUEUE="cpu"
NCPUS=48
WALLTIME="06:00:00"
CONCURRENCY=20
# If an env is active in THIS shell, forward it to the jobs as an override.
# If not, that is fine: submit_mea_array.sh activates its own built-in default
# (DEFAULT_ENV_NAME) internally, so jobs never rely on inheriting anything.
CONDA_PREFIX_ARG="${CONDA_PREFIX:-}"
CONDA_ENV_ARG="${CONDA_DEFAULT_ENV:-}"
MANIFEST="./mea_manifest_$(date +%Y%m%d_%H%M%S).tsv"
EXTRA_ARGS=""
SKIP_DONE=""
DRY_RUN=0
CAMPAIGN_ROOTS=()
GLOBS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --campaign-root) CAMPAIGN_ROOTS+=("$2"); shift 2 ;;
        --glob) GLOBS+=("$2"); shift 2 ;;
        --lib) LIB="$2"; shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --ncpus) NCPUS="$2"; shift 2 ;;
        --walltime) WALLTIME="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --manifest) MANIFEST="$2"; shift 2 ;;
        --conda-prefix) CONDA_PREFIX_ARG="$2"; shift 2 ;;
        --conda-env) CONDA_ENV_ARG="$2"; shift 2 ;;
        --skip-done) SKIP_DONE="--skip-done"; shift ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$OUT_ROOT" ]; then
    echo "ERROR: --out-root is required" >&2
    exit 2
fi
if [ ${#CAMPAIGN_ROOTS[@]} -eq 0 ] && [ ${#GLOBS[@]} -eq 0 ]; then
    echo "ERROR: give at least one --campaign-root or --glob" >&2
    exit 2
fi

# --- 1. build the eap library ONCE, synchronously, before submitting -----
#     so no array member has to (or races to) generate it under load.
if [ ! -f "$LIB" ]; then
    echo "[launch] library not found, building once -> $LIB"
    python3 "${SCRIPT_DIR}/eap_template_library.py" --out "$LIB"
else
    echo "[launch] using existing library: $LIB"
fi

# --- 2. build the manifest ------------------------------------------------
MANIFEST_ARGS=(--out-root "$OUT_ROOT" --manifest "$MANIFEST" $SKIP_DONE)
for r in "${CAMPAIGN_ROOTS[@]-}"; do
    [ -n "$r" ] && MANIFEST_ARGS+=(--campaign-root "$r")
done
for g in "${GLOBS[@]-}"; do
    [ -n "$g" ] && MANIFEST_ARGS+=(--glob "$g")
done

python3 "${SCRIPT_DIR}/build_mea_manifest.py" "${MANIFEST_ARGS[@]}"

N_UNITS=$(wc -l < "$MANIFEST")
if [ "$N_UNITS" -eq 0 ]; then
    echo "ERROR: manifest is empty, nothing to submit" >&2
    exit 2
fi

# --- 3. submit ------------------------------------------------------------
QSUB_V="MANIFEST=$(realpath "$MANIFEST"),LIB=$(realpath "$LIB")"
if [ -n "$CONDA_PREFIX_ARG" ]; then
    QSUB_V="${QSUB_V},ENV_PREFIX=${CONDA_PREFIX_ARG}"
    echo "[launch] jobs will use env prefix: ${CONDA_PREFIX_ARG}"
elif [ -n "$CONDA_ENV_ARG" ]; then
    QSUB_V="${QSUB_V},CONDA_ENV=${CONDA_ENV_ARG}"
    echo "[launch] jobs will use env name: ${CONDA_ENV_ARG}"
else
    echo "[launch] no env override given; jobs will activate their built-in"
    echo "[launch] default internally (DEFAULT_ENV_NAME in submit_mea_array.sh)"
fi
if [ -n "$EXTRA_ARGS" ]; then
    QSUB_V="${QSUB_V},EXTRA_ARGS=${EXTRA_ARGS}"
fi

if [ "$N_UNITS" -gt 1 ]; then
    CMD=(qsub -q "$QUEUE" -l "select=1:ncpus=${NCPUS},walltime=${WALLTIME}"
         -J "0-$((N_UNITS - 1))%${CONCURRENCY}"
         -v "$QSUB_V" "${SCRIPT_DIR}/submit_mea_array.sh")
else
    # PBS Pro rejects a single-element array; submit a plain job instead
    # (PBS_ARRAY_INDEX is then unset, and submit_mea_array.sh defaults it to 0,
    # matching the manifest's only line).
    echo "[launch] only 1 work unit found -- submitting as a plain (non-array) job"
    CMD=(qsub -q "$QUEUE" -l "select=1:ncpus=${NCPUS},walltime=${WALLTIME}"
         -v "$QSUB_V" "${SCRIPT_DIR}/submit_mea_array.sh")
fi

echo "[launch] ${N_UNITS} work unit(s), concurrency ${CONCURRENCY}, queue ${QUEUE}"
echo "[launch] ${CMD[*]}"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "[launch] --dry-run: not submitting"
    exit 0
fi

JID=$("${CMD[@]}")
echo "[launch] submitted: $JID"
echo "[launch] monitor with: qstat -t \"$JID\"    (or: qstat -u \$USER)"
