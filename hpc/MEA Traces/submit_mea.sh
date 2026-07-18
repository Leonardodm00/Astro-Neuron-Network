#!/bin/bash
#PBS -S /bin/bash
#PBS -N mea_postproc
#PBS -k eo
#PBS -l walltime=06:00:00
##########################################################################
# Virtual-MEA post-processing wrapper.
#
# This is POST-processing: it reads an existing campaign (spike times +
# positions) and writes detected electrode spikes. It needs only numpy +
# scipy (NO Brian2, NO C++ compilation), so it is far lighter than the
# simulation sweep and a short walltime suffices.
#
# SUBMIT (single campaign directory that directly contains topo_*/):
#     qsub -l select=1:ncpus=48 \
#          -v CAMPAIGN=/scratch/USER/campaign_TAG/sweep_c48_task0000,\
#OUT=/scratch/USER/mea_out/sweep_c48_task0000,\
#LIB=/scratch/USER/eap_library.npz \
#          submit_mea.sh
#
# NOTE: --campaign must point at the directory that DIRECTLY contains the
#       topo_* folders (i.e. a sweep_<NODETAG>_task<IDX> dir, not the
#       campaign_<TAG> root). Submit one job per sweep dir, or loop them.
#
# REQUIRED -v variables:
#     CAMPAIGN   path containing topo_*/  (input)
#     OUT        output path              (mea_iter_*.npz written here)
# OPTIONAL -v variables:
#     LIB        EAP library .npz (default: <script dir>/eap_library.npz;
#                auto-generated if missing)
#     SAVE_TRACES  if set to 1, dump the raw (E,T) trace for iter 0 per topo
##########################################################################

set -euo pipefail

if [ -z "${CAMPAIGN:-}" ] || [ -z "${OUT:-}" ]; then
    echo "ERROR: pass -v CAMPAIGN=<dir with topo_*> and -v OUT=<dir>" >&2
    exit 2
fi

# --- environment: activate the env that has numpy + scipy ---------------
# Edit to match your cluster (same env used for the sweep is fine).
# module load anaconda3 2>/dev/null || true
# source activate brian_env 2>/dev/null || conda activate brian_env

cd "${PBS_O_WORKDIR:-.}"

NCPUS="${PBS_NCPUS:-1}"
LIB="${LIB:-./eap_library.npz}"
EXTRA=""
if [ "${SAVE_TRACES:-0}" = "1" ]; then
    EXTRA="--save_traces_first"
fi

echo "[mea] campaign : ${CAMPAIGN}"
echo "[mea] out      : ${OUT}"
echo "[mea] library  : ${LIB}"
echo "[mea] workers  : ${NCPUS}"

python process_campaign.py \
    --campaign "${CAMPAIGN}" \
    --out "${OUT}" \
    --library "${LIB}" \
    --workers "${NCPUS}" \
    ${EXTRA}

echo "[mea] done."
