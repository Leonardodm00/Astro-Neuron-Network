#!/bin/bash

#PBS -S /bin/bash
#PBS -N "indeg_analyze"
#PBS -q cpu
#PBS -l select=1:ncpus=4,walltime=02:00:00
#PBS -k eo

##########################################################################
# STEP 2 — in-degree convergence ANALYSIS (HPC, pure numpy)
#
# Runs indegree_convergence_analyze.py on the net_*.npz produced by STEP 1.
# For each (boundary, d0, beta, p0) cell it pools the in-degrees over reps,
# computes mean/std/CV vs L, and the Wasserstein-1 / Jensen-Shannon / KL
# distances of P(k;L) to (a) the largest L [reference] and (b) the next L
# [consecutive]. It then reports the minimum culture size L* beyond which the
# distribution is statistically indistinguishable from convergence.
#
# Distributional metrics are computed on a FIXED subsample size per L (so the
# sample-size bias cannot fake convergence), with the threshold set relative to
# a bootstrapped noise floor. Moments use the full pooled sample.
#
# This step is light and SERIAL (numpy/BLAS may use the few requested CPUs).
#
# EDIT THE VARIABLES IN THE "USER CONFIG" BLOCK BEFORE SUBMITTING.
##########################################################################

# ─── USER CONFIG ────────────────────────────────────────────────────────
# Absolute paths recommended (jobs do not always inherit your $PWD nicely).

SCRIPT_PATH="./indegree_convergence_analyze.py"      # path to the analyser
INPUT_DIR="./indeg_sweep"                             # dir of net_*.npz (STEP 1 output)
OUTPUT_DIR="./indeg_analysis_${PBS_JOBID%%.*}"        # analysis results dir

# ─── Analysis settings ──────────────────────────────────────────────────
BOUNDARIES="finite periodic"   # which boundaries to analyse
N_BOOT=200                     # bootstrap reps per distance estimate
SUBSAMPLE=0                    # fixed M_sub; 0 => auto (min eligible pooled_n)
MIN_POOLED_N=200               # drop L points with fewer pooled samples from L* search
C_MULT=2.0                     # convergence threshold = C_MULT * noise floor
REL_W1_THRESH=0.0              # extra gate: require W1/<k>_ref <= this (0 disables)
SEED=0                         # RNG seed for bootstrap reproducibility
MAKE_PLOTS=1                   # 1 => emit mean_k / W1 grid PNGs; 0 => CSV only
DPI=150
# ────────────────────────────────────────────────────────────────────────

# Move to the directory from which the job was submitted
cd "$PBS_O_WORKDIR"

# Load modules
module load python

# Activate the conda env (must source conda.sh in non-interactive shells)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate brian_env

# Make output dir if it does not exist yet
mkdir -p "$OUTPUT_DIR"

# This step is serial Python; let numpy/BLAS use the requested CPUs for array ops.
NCPUS_AVAILABLE=$(nproc 2>/dev/null || echo 1)
export OMP_NUM_THREADS=$NCPUS_AVAILABLE
export OPENBLAS_NUM_THREADS=$NCPUS_AVAILABLE
export MKL_NUM_THREADS=$NCPUS_AVAILABLE

# Headless plotting
export MPLBACKEND=Agg

# Sanity prints
echo "Running on node:       $(hostname)"
echo "Job ID:                $PBS_JOBID"
echo "Working dir:           $PBS_O_WORKDIR"
echo "Python:                $(which python3)"
echo "Conda env:             $CONDA_DEFAULT_ENV"
echo "Script:                $SCRIPT_PATH"
echo "Input dir:             $INPUT_DIR"
echo "Output dir:            $OUTPUT_DIR"
echo "Boundaries:            $BOUNDARIES"
echo "Bootstrap reps:        $N_BOOT   subsample=$SUBSAMPLE (0=auto)   min_pooled_n=$MIN_POOLED_N"
echo "Threshold:             c_mult=$C_MULT   rel_w1_thresh=$REL_W1_THRESH   seed=$SEED"
echo "Plots:                 $MAKE_PLOTS   (dpi=$DPI)"
echo "-----------------------------------------"

# ─── Build the argument list ────────────────────────────────────────────
ARGS=(
    --in_dir         "$INPUT_DIR"
    --out_dir        "$OUTPUT_DIR"
    --boundaries     $BOUNDARIES
    --n_boot         "$N_BOOT"
    --subsample      "$SUBSAMPLE"
    --min_pooled_n   "$MIN_POOLED_N"
    --c_mult         "$C_MULT"
    --rel_w1_thresh  "$REL_W1_THRESH"
    --seed           "$SEED"
    --dpi            "$DPI"
)

# Optional: emit diagnostic plots
if [ "$MAKE_PLOTS" -eq 1 ]; then
    ARGS+=(--plots)
fi

# ─── Run ────────────────────────────────────────────────────────────────
python "$SCRIPT_PATH" "${ARGS[@]}"

# Clean up
conda deactivate

sleep 5s
