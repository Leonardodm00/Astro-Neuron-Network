#!/bin/bash

#PBS -S /bin/bash
#PBS -N "asn_analysis"
#PBS -q intel
#PBS -l select=1:ncpus=48,walltime=04:00:00
#PBS -k eo

##########################################################################
# submit_analysis.sh — run analyze_topology.py on every topo_* directory
# inside a completed sweep folder.
#
# Each topo_* is processed sequentially by this single job.  The per-iter
# work (metric computation + raster rendering) is parallelised internally
# by analyze_topology.py using --workers $N_WORKERS (spawn pool).
#
# USAGE
#   Edit SWEEP_DIR below, then:
#       qsub submit_analysis.sh
#
# To analyse multiple sweep directories from separate jobs, copy this
# script and change SWEEP_DIR for each one — they are fully independent.
##########################################################################

# ─── USER CONFIG ────────────────────────────────────────────────────────
SWEEP_DIR="./sweep_JOBID"    # <-- set to the sweep directory to analyse
SCRIPT_PATH="./analyze_topology.py"

DPI=150
NO_PLOTS=""          # set to "--no_plots" to skip raster rendering
# ────────────────────────────────────────────────────────────────────────

cd "$PBS_O_WORKDIR"

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate brian_env

# ─── Worker count ───────────────────────────────────────────────────────
if   [ -n "$PBS_NP" ];    then N_WORKERS="$PBS_NP"
elif [ -n "$PBS_NCPUS" ]; then N_WORKERS="$PBS_NCPUS"
else                            N_WORKERS=$(nproc 2>/dev/null || echo 1)
fi

# ─── Sanity prints ──────────────────────────────────────────────────────
echo "============================================================"
echo "Running on node:       $(hostname)"
echo "Job ID:                $PBS_JOBID"
echo "Conda env:             $CONDA_DEFAULT_ENV"
echo "Sweep dir:             $SWEEP_DIR"
echo "N_WORKERS (resolved)   : $N_WORKERS"
echo "DPI                    : $DPI"
echo "============================================================"

# ─── Discover topology directories ──────────────────────────────────────
TOPO_DIRS=( $(ls -d "${SWEEP_DIR}"/topo_*/ 2>/dev/null | sort) )

if [ ${#TOPO_DIRS[@]} -eq 0 ]; then
    echo "[error] no topo_* directories found in ${SWEEP_DIR}"
    exit 1
fi

echo "Found ${#TOPO_DIRS[@]} topology directories to analyse."
echo

# ─── Process each topology ──────────────────────────────────────────────
N_OK=0
N_FAIL=0

for TOPO_DIR in "${TOPO_DIRS[@]}"; do
    echo "------------------------------------------------------------"
    echo "Analysing: $TOPO_DIR"

    python "$SCRIPT_PATH" \
        --topo_dir  "$TOPO_DIR" \
        --workers   "$N_WORKERS" \
        --dpi       "$DPI" \
        $NO_PLOTS

    if [ $? -eq 0 ]; then
        N_OK=$((N_OK + 1))
    else
        echo "[!] analyze_topology.py returned non-zero for $TOPO_DIR"
        N_FAIL=$((N_FAIL + 1))
    fi
done

echo
echo "============================================================"
echo "Analysis complete: ${N_OK} ok / ${N_FAIL} failed"
echo "============================================================"

conda deactivate
sleep 5s
