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
# Submit (the intel or the cpu queue):
#     qsub -q intel -v CAMPAIGN=LindaColloDUP/campaign_lcdup15_rho1300_neur_n01 submit_campaign_inspect.sh
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
#
# Every value the job needs is copied into a CI_* variable BEFORE the
# environment is activated, and only CI_* variables are read after it: conda's
# activation scripts export variables of their own (its compiler packages
# export BUILD and HOST), and job 2100916 (2026-11-19) skipped the index
# rebuild because the activation overwrote this script's former BUILD flag.
##########################################################################
set -u

CI_REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}"
CI_CONDA_ENV="${CONDA_ENV:-brian_final}"
CI_REINDEX="${REINDEX:-auto}"
CI_DRYRUN="${DRYRUN:-0}"
CI_SKIP_CONDA="${SKIP_CONDA:-0}"
CI_ROADMAP_OFF="${ROADMAP_DISABLE:-0}"
CI_CAMPAIGN_ARG="${CAMPAIGN:-}"
CI_OUT_ARG="${OUT:-}"

if [ ! -f "${CI_REPO_DIR}/campaign_inspect.py" ] || [ ! -f "${CI_REPO_DIR}/aggregate_sweep.py" ]; then
    echo "ERROR: ${CI_REPO_DIR} lacks campaign_inspect.py or aggregate_sweep.py (submit from hpc/Phenomenological_finalv1)." >&2
    exit 2
fi
cd "$CI_REPO_DIR" || exit 2
if [ -z "$CI_CAMPAIGN_ARG" ]; then
    echo "ERROR: set CAMPAIGN, e.g. qsub -v CAMPAIGN=<campaign folder> submit_campaign_inspect.sh" >&2
    exit 2
fi
if [ ! -d "$CI_CAMPAIGN_ARG" ]; then
    echo "ERROR: campaign folder ${CI_CAMPAIGN_ARG} not found (a relative path starts at ${CI_REPO_DIR})." >&2
    exit 2
fi
case "$CI_REINDEX" in
    auto|0|1) ;;
    *) echo "ERROR: REINDEX must be auto, 0 or 1 (got ${CI_REINDEX})." >&2; exit 2 ;;
esac
CI_CAMPAIGN="$(cd "$CI_CAMPAIGN_ARG" && pwd)"
CI_OUT="${CI_OUT_ARG:-${CI_CAMPAIGN}/campaign_inspect_$(date -u +%Y%m%dT%H%M%SZ)}"
CI_IDX="${CI_CAMPAIGN}/campaign_index/summary.csv"
CI_AGGREGATE="${CI_REPO_DIR}/aggregate_sweep.py"
CI_INSPECT="${CI_REPO_DIR}/campaign_inspect.py"

# Does campaign_index/ need a rebuild? (SPEC Block 14, "Job")
CI_N_MAN=$(find "$CI_CAMPAIGN" -mindepth 2 -maxdepth 2 -name manifest.json 2>/dev/null | wc -l)
if [ "$CI_REINDEX" = "1" ]; then
    CI_BUILD=1; CI_WHY="REINDEX=1"
elif [ ! -f "$CI_IDX" ]; then
    if [ "$CI_REINDEX" = "0" ]; then
        echo "ERROR: ${CI_IDX} is missing and REINDEX=0." >&2
        exit 2
    fi
    CI_BUILD=1; CI_WHY="no summary.csv yet"
elif [ "$CI_REINDEX" = "0" ]; then
    CI_BUILD=0; CI_WHY="REINDEX=0"
elif [ "$CI_N_MAN" -eq 0 ]; then
    CI_BUILD=1; CI_WHY="no manifest.json to date the index against"
else
    CI_N_NEW=$(find "$CI_CAMPAIGN" -mindepth 2 -maxdepth 2 -name manifest.json -newer "$CI_IDX" 2>/dev/null | wc -l)
    if [ "$CI_N_NEW" -gt 0 ]; then
        CI_BUILD=1; CI_WHY="${CI_N_NEW} of ${CI_N_MAN} manifest.json newer than summary.csv"
    else
        CI_BUILD=0; CI_WHY="summary.csv newer than all ${CI_N_MAN} manifest.json"
    fi
fi

echo "campaign inspection | host $(hostname) | job ${PBS_JOBID:-none}"
echo "campaign : ${CI_CAMPAIGN}"
echo "index    : ${CI_IDX}"
echo "rebuild  : ${CI_BUILD} (${CI_WHY})"
echo "output   : ${CI_OUT}"
if [ "$CI_DRYRUN" = "1" ]; then
    if [ "$CI_BUILD" = "1" ]; then
        echo "[dry run] python ${CI_AGGREGATE} ${CI_CAMPAIGN} --write-index > ${CI_OUT}/aggregate_index.log"
    fi
    echo "[dry run] python ${CI_INSPECT} ${CI_CAMPAIGN} --out ${CI_OUT}"
    exit 0
fi

# ---- environment: from here on only CI_* variables are read ---------------
if [ "$CI_SKIP_CONDA" = "1" ]; then
    CI_PY="$(command -v python3 || true)"
else
    set +u                               # module/conda hooks may read unset variables
    module load python
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CI_CONDA_ENV" || { echo "ERROR: conda activate ${CI_CONDA_ENV} failed." >&2; exit 2; }
    set -u
    # The interpreter comes from CONDA_PREFIX, never from a bare python on PATH
    # (a module-loaded system python can win on PATH; Block 6's job does the same).
    CI_PY="${CONDA_PREFIX:-}/bin/python"
fi
if [ -z "$CI_PY" ] || [ ! -x "$CI_PY" ]; then
    echo "ERROR: no usable python (${CI_PY:-none})." >&2
    exit 2
fi
"$CI_PY" -c "import sys, numpy, pandas, matplotlib; print('[env]', sys.executable, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| matplotlib', matplotlib.__version__)" \
    || { echo "ERROR: ${CI_PY} lacks numpy, pandas or matplotlib." >&2; exit 2; }
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg

# Roadmap marker (specs/roadmap/hpc_status.sh, stage I14 of specs/roadmap/ANN.yaml):
# inside a PBS job only, it pushes "running" at the start and "done" or
# "failed" at the end to the hpc-status branch of origin. It never fails the
# job; ROADMAP_DISABLE=1 turns it off; runs outside PBS skip it.
ROADMAP_HELPER="${CI_REPO_DIR}/../../specs/roadmap/hpc_status.sh"
if [ -n "${PBS_JOBID:-}" ] && [ "$CI_ROADMAP_OFF" != "1" ] && [ -f "$ROADMAP_HELPER" ]; then
    export ROADMAP_STAGE=I14
    export ROADMAP_N_TASKS="1"
    export ROADMAP_SCRIPT="hpc/Phenomenological_finalv1/submit_campaign_inspect.sh"
    . "$ROADMAP_HELPER" || true
    roadmap_job_start
    trap 'roadmap_job_end $?' EXIT
fi

mkdir -p "$CI_OUT" || { echo "ERROR: cannot create ${CI_OUT}." >&2; exit 2; }
if [ "$CI_BUILD" = "1" ]; then
    echo "[index] python ${CI_AGGREGATE} ${CI_CAMPAIGN} --write-index (log: ${CI_OUT}/aggregate_index.log)"
    "$CI_PY" "$CI_AGGREGATE" "$CI_CAMPAIGN" --write-index > "${CI_OUT}/aggregate_index.log" 2>&1
    CI_RC=$?
    tail -n 4 "${CI_OUT}/aggregate_index.log"
    if [ "$CI_RC" -ne 0 ]; then
        echo "ERROR: the index rebuild failed (exit ${CI_RC}); see ${CI_OUT}/aggregate_index.log" >&2
        exit 3
    fi
fi
if [ ! -f "$CI_IDX" ]; then
    echo "ERROR: ${CI_IDX} is missing (rebuild ${CI_BUILD}: ${CI_WHY}); see ${CI_OUT}/aggregate_index.log" >&2
    exit 3
fi
"$CI_PY" "$CI_INSPECT" "$CI_CAMPAIGN" --out "$CI_OUT"
CI_EXIT=$?
if [ "$CI_SKIP_CONDA" != "1" ]; then
    set +u; conda deactivate; set -u
fi
exit $CI_EXIT
