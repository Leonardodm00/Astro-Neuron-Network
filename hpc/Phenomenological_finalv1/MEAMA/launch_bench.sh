#!/bin/bash
# =============================================================================
# launch_bench.sh -- submit the MEAMA sizing benchmark (6 configs, one job).
#
# Run this from ANYWHERE; it resolves the repository directory from its own
# location and injects it into the PBS job, because a PBS job script is spooled
# to a scratch path and cannot find the repository by itself.
#
#     bash MEAMA/launch_bench.sh                 # submit to the default queue
#     QUEUE=cpu NCPUS=48 bash MEAMA/launch_bench.sh
#     DRYRUN=1 bash MEAMA/launch_bench.sh        # print the qsub, submit nothing
#
# Interactive alternative (no scheduler, for a quick single-config probe):
#     REPO_DIR=$(pwd)/MEAMA C_MAX_LIST=300 SIMTIME_LIST=200 N_TOPO_BENCH=1 \
#         K_BENCH=2 N_BENCH_WORKERS=4 bash MEAMA/bench_sizing.sh
# =============================================================================
set -u

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# MEAMA/ is self-contained: the pipeline files (HPC_main_sweep.py etc.) live
# IN this folder, not in its parent. REPO_DIR is therefore this folder itself.
REPO_DIR="$_HERE"
WORKER="${_HERE}/bench_sizing.sh"

if [ ! -f "${REPO_DIR}/HPC_main_sweep.py" ]; then
    echo "ERROR: ${REPO_DIR} does not contain HPC_main_sweep.py." >&2
    echo "       MEAMA/ must be self-contained: HPC_main_sweep.py," >&2
    echo "       HPC_single_run.py, ASD_fun_BD_cpp.py, aggregate_sweep.py," >&2
    echo "       burst_metrics.py, burst_plots.py, seed_alloc.py, and" >&2
    echo "       synapse_pdist.csv all belong in THIS folder." >&2
    exit 2
fi
if [ ! -f "$WORKER" ]; then
    echo "ERROR: worker not found: $WORKER" >&2
    exit 2
fi

QUEUE="${QUEUE:-intel}"
NCPUS="${NCPUS:-16}"
WALLTIME="${WALLTIME:-12:00:00}"

# Benchmark grid / sizing -- forwarded to the worker via -v. Override from the
# environment to probe a single cell first.
C_MAX_LIST="${C_MAX_LIST:-300 350 400}"
SIMTIME_LIST="${SIMTIME_LIST:-200 300}"
RHO_TOTAL="${RHO_TOTAL:-1600}"
ASTRO_FRAC="${ASTRO_FRAC:-0.5}"
N_TOPO_BENCH="${N_TOPO_BENCH:-2}"
K_BENCH="${K_BENCH:-3}"
N_BENCH_WORKERS="${N_BENCH_WORKERS:-12}"
SEED_MASTER_BENCH="${SEED_MASTER_BENCH:-20260831}"

if [ "$(awk -v f="$ASTRO_FRAC" 'BEGIN{print (f>=0 && f<1) ? 1 : 0}')" != "1" ]; then
    echo "ERROR: ASTRO_FRAC=${ASTRO_FRAC} must satisfy 0 <= f < 1." >&2
    exit 4
fi

RHO_N=$(awk -v r="$RHO_TOTAL" -v f="$ASTRO_FRAC" 'BEGIN{printf "%.6g", r*(1-f)}')
RHO_A=$(awk -v r="$RHO_TOTAL" -v f="$ASTRO_FRAC" 'BEGIN{printf "%.6g", r*f}')

echo "============================================================"
echo "MEAMA sizing benchmark"
echo "REPO_DIR   : ${REPO_DIR}"
echo "worker     : ${WORKER}"
echo "queue      : ${QUEUE}   ncpus=${NCPUS}   walltime=${WALLTIME}"
echo "conda env  : ${CONDA_ENV:-brian_final}"
echo "grid       : C_MAX {${C_MAX_LIST}}  x  SIMTIME {${SIMTIME_LIST}}"
echo "density    : rho_total=${RHO_TOTAL}/mm^2  ->  neurons ${RHO_N}, astro ${RHO_A}"
echo "per config : ${N_TOPO_BENCH} topo x ${N_BENCH_WORKERS} workers x ${K_BENCH} params"
echo
echo "expected populations (Nn = round(rho_neuron * (c_max/1000)^2)):"
for c in $C_MAX_LIST; do
    awk -v c="$c" -v rn="$RHO_N" -v ra="$RHO_A" \
        'BEGIN{a=(c/1000)^2; printf "    c_max=%4d um  area=%.4f mm^2  Nn=%d  Na=%d  total=%d\n",
               c, a, int(rn*a+0.5), int(ra*a+0.5), int(rn*a+0.5)+int(ra*a+0.5)}'
done
echo "============================================================"

CONDA_ENV="${CONDA_ENV:-brian_final}"

VARS="REPO_DIR=${REPO_DIR},CONDA_ENV=${CONDA_ENV}"
VARS="${VARS},C_MAX_LIST=${C_MAX_LIST}"
VARS="${VARS},SIMTIME_LIST=${SIMTIME_LIST}"
VARS="${VARS},RHO_TOTAL=${RHO_TOTAL}"
VARS="${VARS},ASTRO_FRAC=${ASTRO_FRAC}"
VARS="${VARS},N_TOPO_BENCH=${N_TOPO_BENCH}"
VARS="${VARS},K_BENCH=${K_BENCH}"
VARS="${VARS},N_BENCH_WORKERS=${N_BENCH_WORKERS}"
VARS="${VARS},SEED_MASTER_BENCH=${SEED_MASTER_BENCH}"

# C_MAX_LIST / SIMTIME_LIST contain spaces, so -v must be a single quoted arg.
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "[dry run] would submit:"
    echo "  qsub -q ${QUEUE} -l select=1:ncpus=${NCPUS},walltime=${WALLTIME} \\"
    echo "       -v \"${VARS}\" ${WORKER}"
    exit 0
fi

jid=$(qsub -q "$QUEUE" \
      -l "select=1:ncpus=${NCPUS},walltime=${WALLTIME}" \
      -v "$VARS" \
      "$WORKER")
rc=$?
if [ $rc -ne 0 ] || [ -z "$jid" ]; then
    echo "ERROR: qsub failed (rc=${rc})." >&2
    exit 5
fi

echo "submitted: ${jid}"
echo
echo "watch    : qstat -u \$USER"
echo "report   : python ${_HERE}/bench_report.py \$(ls -td ${REPO_DIR}/bench_out/bench_* | head -1)"
