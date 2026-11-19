#!/bin/bash
#PBS -S /bin/bash
#PBS -N camp_inspect
#PBS -k eo
#PBS -l select=1:ncpus=48
#PBS -l walltime=01:00:00
##########################################################################
# Behaviour inspection of a finished or stopped campaign (specs/SPEC.md
# Block 14, decision D-004). Refreshes <campaign>/campaign_index/ with
# aggregate_sweep.py --write-index when it is stale, then writes the
# activity-class report and its figures to <campaign>/campaign_inspect_<UTC>/.
# Writes nothing else; the campaign folder is ignored by git.
#
# Rehearse on the login node (prints the plan, runs nothing), from the folder
# that holds this script (hpc/Phenomenological_finalv1 of the clone):
#     DRYRUN=1 CAMPAIGN=LindaColloDUP/campaign_lcdup15_rho1300_neur_n01 bash submit_campaign_inspect.sh
# Submit:
#     qsub -q cpu -v CAMPAIGN=LindaColloDUP/campaign_lcdup15_rho1300_neur_n01 submit_campaign_inspect.sh
# Variables (qsub -v, or the environment for bash):
#     CAMPAIGN      campaign folder, absolute or relative to this folder (required)
#     REINDEX       auto (default): rebuild the index when summary.csv is
#                   missing, older than a sweep_*/manifest.json, or when the
#                   campaign has no manifest; 1: always; 0: never
#     OUT           output folder (default <CAMPAIGN>/campaign_inspect_<UTC>)
#     CONDA_ENV     default brian_final
#     DRYRUN=1      print the plan and exit
#     SKIP_CONDA=1  use python3 from PATH (tests outside davinci)
#     ROADMAP_DISABLE=1  no roadmap marker
##########################################################################
set -u

REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}"
CONDA_ENV="${CONDA_ENV:-brian_final}"
REINDEX="${REINDEX:-auto}"
DRYRUN="${DRYRUN:-0}"

if [ ! -f "${REPO_DIR}/campaign_inspect.py" ] || [ ! -f "${REPO_DIR}/aggregate_sweep.py" ]; then
    echo "ERROR: ${REPO_DIR} lacks campaign_inspect.py or aggregate_sweep.py (submit from hpc/Phenomenological_finalv1)." >&2
    exit 2
fi
cd "$REPO_DIR" || exit 2
if [ -z "${CAMPAIGN:-}" ]; then
    echo "ERROR: set CAMPAIGN, e.g. qsub -v CAMPAIGN=<campaign folder> submit_campaign_inspect.sh" >&2
    exit 2
fi
if [ ! -d "$CAMPAIGN" ]; then
    echo "ERROR: campaign folder ${CAMPAIGN} not found (a relative path starts at ${REPO_DIR})." >&2
    exit 2
fi
case "$REINDEX" in
    auto|0|1) ;;
    *) echo "ERROR: REINDEX must be auto, 0 or 1 (got ${REINDEX})." >&2; exit 2 ;;
esac
CAMPAIGN="$(cd "$CAMPAIGN" && pwd)"
OUT="${OUT:-${CAMPAIGN}/campaign_inspect_$(date -u +%Y%m%dT%H%M%SZ)}"
IDX="${CAMPAIGN}/campaign_index/summary.csv"

# Does campaign_index/ need a rebuild? (SPEC Block 14, "Job")
N_MAN=$(find "$CAMPAIGN" -mindepth 2 -maxdepth 2 -name manifest.json 2>/dev/null | wc -l)
if [ "$REINDEX" = "1" ]; then
    BUILD=1; WHY="REINDEX=1"
elif [ ! -f "$IDX" ]; then
    if [ "$REINDEX" = "0" ]; then
        echo "ERROR: ${IDX} is missing and REINDEX=0." >&2
        exit 2
    fi
    BUILD=1; WHY="no summary.csv yet"
elif [ "$REINDEX" = "0" ]; then
    BUILD=0; WHY="REINDEX=0"
elif [ "$N_MAN" -eq 0 ]; then
    BUILD=1; WHY="no manifest.json to date the index against"
else
    N_NEW=$(find "$CAMPAIGN" -mindepth 2 -maxdepth 2 -name manifest.json -newer "$IDX" 2>/dev/null | wc -l)
    if [ "$N_NEW" -gt 0 ]; then
        BUILD=1; WHY="${N_NEW} of ${N_MAN} manifest.json newer than summary.csv"
    else
        BUILD=0; WHY="summary.csv newer than all ${N_MAN} manifest.json"
    fi
fi

echo "campaign inspection | host $(hostname) | job ${PBS_JOBID:-none}"
echo "campaign : ${CAMPAIGN}"
echo "index    : ${IDX}"
echo "rebuild  : ${BUILD} (${WHY})"
echo "output   : ${OUT}"
if [ "$DRYRUN" = "1" ]; then
    if [ "$BUILD" = "1" ]; then
        echo "[dry run] python aggregate_sweep.py ${CAMPAIGN} --write-index > ${OUT}/aggregate_index.log"
    fi
    echo "[dry run] python campaign_inspect.py ${CAMPAIGN} --out ${OUT}"
    exit 0
fi

if [ "${SKIP_CONDA:-0}" = "1" ]; then
    PY="$(command -v python3 || true)"
else
    set +u                               # module/conda hooks may read unset variables
    module load python
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || { echo "ERROR: conda activate ${CONDA_ENV} failed." >&2; exit 2; }
    set -u
    # The interpreter comes from CONDA_PREFIX, never from a bare python on PATH
    # (a module-loaded system python can win on PATH; Block 6's job does the same).
    PY="${CONDA_PREFIX:-}/bin/python"
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "ERROR: no usable python (${PY:-none})." >&2
    exit 2
fi
"$PY" -c "import sys, numpy, pandas, matplotlib; print('[env]', sys.executable, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| matplotlib', matplotlib.__version__)" \
    || { echo "ERROR: ${PY} lacks numpy, pandas or matplotlib." >&2; exit 2; }
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

# Roadmap marker (specs/roadmap/hpc_status.sh, stage I14 of specs/roadmap/ANN.yaml):
# inside a PBS job only, it pushes "running" at the start and "done" or
# "failed" at the end to the hpc-status branch of origin. It never fails the
# job; ROADMAP_DISABLE=1 turns it off; runs outside PBS skip it.
ROADMAP_HELPER="${REPO_DIR}/../../specs/roadmap/hpc_status.sh"
if [ -n "${PBS_JOBID:-}" ] && [ "${ROADMAP_DISABLE:-0}" != "1" ] && [ -f "$ROADMAP_HELPER" ]; then
    export ROADMAP_STAGE=I14
    export ROADMAP_N_TASKS="1"
    export ROADMAP_SCRIPT="hpc/Phenomenological_finalv1/submit_campaign_inspect.sh"
    . "$ROADMAP_HELPER" || true
    roadmap_job_start
    trap 'roadmap_job_end $?' EXIT
fi

mkdir -p "$OUT" || { echo "ERROR: cannot create ${OUT}." >&2; exit 2; }
if [ "$BUILD" = "1" ]; then
    echo "[index] python aggregate_sweep.py ${CAMPAIGN} --write-index (log: ${OUT}/aggregate_index.log)"
    "$PY" aggregate_sweep.py "$CAMPAIGN" --write-index > "${OUT}/aggregate_index.log" 2>&1
    RC=$?
    tail -n 4 "${OUT}/aggregate_index.log"
    if [ "$RC" -ne 0 ] || [ ! -f "$IDX" ]; then
        echo "ERROR: the index rebuild failed (exit ${RC}); see ${OUT}/aggregate_index.log" >&2
        exit 3
    fi
fi
"$PY" campaign_inspect.py "$CAMPAIGN" --out "$OUT"
EXIT_CODE=$?
if [ "${SKIP_CONDA:-0}" != "1" ]; then
    set +u; conda deactivate; set -u
fi
exit $EXIT_CODE
