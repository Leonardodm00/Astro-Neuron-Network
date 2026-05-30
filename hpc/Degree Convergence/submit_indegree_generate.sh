#!/bin/bash

#PBS -S /bin/bash
#PBS -N "indeg_generate"
#PBS -q cpu
#PBS -l select=1:ncpus=32,walltime=12:00:00
#PBS -k eo

##########################################################################
# STEP 1 — in-degree convergence DATA GENERATION (HPC, pure numpy)
#
# Runs indegree_convergence_generate.py: places point neurons at fixed
# density, wires them with the Weibull/stretched-exponential distance kernel
# under BOTH finite and periodic (minimum-image torus) boundaries, and writes
# one atomic net_<boundary>_L<Lum>_rep<r>.npz per (boundary, L, repetition).
# No dynamics are simulated — connectivity only.
#
# This step is EMBARRASSINGLY PARALLEL: it multiprocesses across all CPUs the
# job is pinned to (--n_workers 0 => auto). Request a fat single node here.
# For multi-node scaling, see the PBS-array notes at the bottom.
#
# EDIT THE VARIABLES IN THE "USER CONFIG" BLOCK BEFORE SUBMITTING.
##########################################################################

# ─── USER CONFIG ────────────────────────────────────────────────────────
# Absolute paths recommended (jobs do not always inherit your $PWD nicely).

SCRIPT_PATH="./indegree_convergence_generate.py"     # path to the generator
OUTPUT_DIR="./indeg_sweep_${PBS_JOBID%%.*}"          # results dir (per-job by default)

# ─── Density & grid ─────────────────────────────────────────────────────
RHO=2000                                  # neuronal density [neurons / mm^2]
L_LIST="100 200 300 500 750 1000 1500 2000"   # culture edge sizes [µm]
D0_LIST="20 40 60 100 150 200"            # kernel length scale d0 [µm]
BETA_LIST="0.2 0.4 0.6 0.8 1.0"           # kernel stretch exponent beta
P0_LIST="0.05 0.1 0.2 0.5 1.0"            # at-zero connection probability p0
BOUNDARIES="finite periodic"              # both boundary conditions
N_REP=20                                  # repetitions per (boundary, L)

# ─── Numerics & reproducibility ─────────────────────────────────────────
CHUNK=512                                 # post-neuron block size (mem/speed)
MASTER_SEED=20240517                      # master seed (placement + wiring)

# ─── Parallelism / sharding ─────────────────────────────────────────────
N_WORKERS=0                               # 0 => use all CPUs the job is pinned to
SHARD=0                                   # this shard index (0-based)
N_SHARDS=1                                # total shards (1 => single-node, no array)
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

# ─── Thread pinning ─────────────────────────────────────────────────────
# CRITICAL: we parallelise with PROCESSES (one per CPU). Each process must run
# numpy/BLAS single-threaded, otherwise n_workers processes x BLAS-threads each
# oversubscribe the node and thrash. Pin every BLAS backend to 1 thread.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# If launched as a PBS array (#PBS -J), use the array index as the shard.
if [ -n "$PBS_ARRAY_INDEX" ]; then
    SHARD=$PBS_ARRAY_INDEX
fi

# Sanity prints
echo "Running on node:       $(hostname)"
echo "Job ID:                $PBS_JOBID"
echo "Working dir:           $PBS_O_WORKDIR"
echo "Python:                $(which python3)"
echo "Conda env:             $CONDA_DEFAULT_ENV"
echo "Script:                $SCRIPT_PATH"
echo "Output dir:            $OUTPUT_DIR"
echo "Density:               rho=$RHO neurons/mm^2"
echo "L list [µm]:           $L_LIST"
echo "d0 list [µm]:          $D0_LIST"
echo "beta list:             $BETA_LIST"
echo "p0 list:               $P0_LIST"
echo "Boundaries:            $BOUNDARIES"
echo "Repetitions:           $N_REP   chunk=$CHUNK   master_seed=$MASTER_SEED"
echo "CPUs pinned:           $(nproc 2>/dev/null || echo '?')   n_workers=$N_WORKERS (0=auto)"
echo "Shard:                 $SHARD / $N_SHARDS"
echo "-----------------------------------------"

# ─── Build the argument list ────────────────────────────────────────────
ARGS=(
    --out_dir      "$OUTPUT_DIR"
    --rho_per_mm2  "$RHO"
    --L_list_um    $L_LIST
    --d0_list      $D0_LIST
    --beta_list    $BETA_LIST
    --p0_list      $P0_LIST
    --boundaries   $BOUNDARIES
    --n_rep        "$N_REP"
    --chunk        "$CHUNK"
    --master_seed  "$MASTER_SEED"
    --n_workers    "$N_WORKERS"
    --shard        "$SHARD"
    --n_shards     "$N_SHARDS"
)

# ─── Run ────────────────────────────────────────────────────────────────
python "$SCRIPT_PATH" "${ARGS[@]}"

# Clean up
conda deactivate

sleep 5s

##########################################################################
# MULTI-NODE SCALING (optional)
# -----------------------------------------------------------------------
# The generator deals out tasks via tasks[shard::n_shards], so you can fan the
# work over N independent array elements, each multiprocessing on its own node:
#
#   1) set  N_SHARDS=<N>  above,
#   2) add the array directive at submit time, e.g. for 8 shards:
#        qsub -J 0-7 submit_indegree_generate.sh
#      (the PBS_ARRAY_INDEX is picked up automatically as SHARD above),
#   3) point every element at the SAME OUTPUT_DIR — the skip-if-exists logic
#      makes overlap harmless and the whole sweep restartable.
##########################################################################
