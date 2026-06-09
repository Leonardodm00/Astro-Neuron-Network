#!/bin/bash

#PBS -S /bin/bash
#PBS -N "asn_sweep"
#PBS -q intel
#PBS -l select=1:ncpus=48,walltime=24:00:00
#PBS -k eo

##########################################################################
# davinci-1 submit script for HPC_main_sweep.py
#
# Nested random sweep:
#   outer iter -> random conn_prob in [CONN_PROB_LO, CONN_PROB_HI]
#                 + fresh topology from numpy
#   inner sweep -> n_workers parallel cpp_standalone sims, each with a
#                  fresh 34-D parameter vector (CAdEx neuron + synapse +
#                  astro) and a fresh noise seed
#
# Each completed simulation is saved (npz + json) BEFORE the next starts,
# so the manifest reflects whatever was on disk when walltime hit.
#
# QUEUE / RESOURCE CHEAT SHEET (davinci-1, OPEN queues)
# -----------------------------------------------------
#   #PBS -q intel  -l select=1:ncpus=48          # Intel Sapphire Rapids, 1 TB
#   #PBS -q cpu    -l select=1:ncpus=48          # Intel Cascade Lake, 768 GB
#   #PBS -q cfd    -l select=1:ncpus=192         # AMD EPYC Turin, 1.5 TB - HUGE
#   #PBS -q egeos  -l select=1:ncpus=192         # AMD EPYC Turin, 1.5 TB - HUGE
# A 100-N / 43-A network plus its cpp_standalone build dir is <1 GB/worker,
# so 192 workers comfortably fit in 1.5 TB.  Pick `cfd` / `egeos` for the
# fattest sweeps; the script automatically picks up $PBS_NCPUS.
##########################################################################

# --- USER CONFIG --------------------------------------------------------
SCRIPT_PATH="./HPC_main_sweep.py"           # path to the python script
LIB_DIR="."                                 # dir with ASD_fun_BD_cpp.py, HPC_single_run.py
OUTPUT_DIR="./sweep_${PBS_JOBID%%.*}"       # one dir per PBS job
SYN_PDIST_CSV=""                            # leave empty -> auto-discover in LIB_DIR

# --- Sweep size ---------------------------------------------------------
N_TOPOLOGIES=2          # upper bound; loop exits on walltime kill
N_PARAMS_PER_WORKER=1       # 1 = max-parallelism design (one param vector
                            #     per worker per topology, fresh compile each)
                            # >1 = each worker compiles once then runs k
                            #     params serially.  Use 2-4 if you want
                            #     fewer topologies but cheaper amortised compile.

# --- Outer-loop random conn_prob distribution ---------------------------
CONN_PROB_LO=0.1
CONN_PROB_HI=0.6

# --- Simulation settings ------------------------------------------------
SIMTIME=50             # simulated duration [seconds]
MODE="Full"             # Full | Neuronal
SWEEP_GROUP="all"       # all | neuron | synapse | astro | neuron_synapse
                        #   which parameter group is FREE (drawn from the prior);
                        #   all other axes are frozen at their nominal value.
                        #   'neuron' => vary only the 10 CAdEx intrinsic axes
                        #   (Sigma gbarA delta_gA tauA DeltaT VT gL VA DeltaA VR).
                        #   For an ISOLATED-neuron search also set MODE="Neuronal"
                        #   and CONN_PROB_LO=0 ; CONN_PROB_HI=0 below.

# --- Population sizes ---------------------------------------------------
NN=100                  # number of neurons
NA=43                   # number of astrocytes (ignored when MODE=Neuronal)

# --- Arena & topology hyperparameters (fixed across outer iterations) --
C_MAX=1100
DISPL_BIAS=15

# Astrocyte topology rule - Wallach et al. 2014 joint-Voronoi (default)
# or legacy distance-based. See Report5_Topology_Wallach.md.
TOPOLOGY_MODE="wallach" # wallach | distance
GJ_DIST=200             # [distance only] KDTree radius for GJC [um]
GJ_MAX_DIST=150         # [wallach  only] soft cap on Voronoi-adjacent GJC [um]
STOA_CUTOFF=70          # hard distance cutoff for synapse->astrocyte links [um]
STOA_SIGMA=200          # [distance only] sigma of Gaussian StoA acceptance [um]

# --- Seeds (base values; the master RNG re-derives per-topology seeds) --
SEED_DEVICE=50
SEED_NEURON=39
SEED_SYNAPSE=35
SEED_ASTRO=60
SEED_MASTER=""          # empty -> derived from PBS_JOBID for reproducibility

# --- Figure options -----------------------------------------------------
DPI=150
# ------------------------------------------------------------------------

# Move to the directory from which the job was submitted
cd "$PBS_O_WORKDIR"

# Load modules
module load python

# Activate the conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate brian_env

# Make output dir
mkdir -p "$OUTPUT_DIR"

# --- Number of parallel workers = number of CPUs PBS gave us ------------
NCPUS_AVAILABLE=$(nproc 2>/dev/null || echo 1)
if [ -n "$PBS_NCPUS" ]; then
    N_WORKERS="$PBS_NCPUS"
else
    N_WORKERS="$NCPUS_AVAILABLE"
fi

# --- Compiler parallelism (CRITICAL!) -----------------------------------
# Each worker process invokes `make` on its own cpp_standalone build dir.
# With N_WORKERS workers all running `make -j$NCPUS` we would oversubscribe
# the node by a factor of N_WORKERS - the node would page-thrash.
# Force each worker to use a SINGLE compile thread:
export OMP_NUM_THREADS=1
export MAKEFLAGS="-j1"

# Per-worker cpp_standalone build dirs go on the node's local fast scratch
# (avoids thousands of small-file metadata ops on the shared filesystem).
if [ -n "$TMPDIR" ]; then
    SCRATCH_ROOT="${TMPDIR}/brian2_sweep_${PBS_JOBID%%.*}"
else
    SCRATCH_ROOT="${OUTPUT_DIR}/_scratch"
fi
mkdir -p "$SCRATCH_ROOT"

# --- Sanity prints ------------------------------------------------------
echo "============================================================"
echo "Running on node:       $(hostname)"
echo "Job ID:                $PBS_JOBID"
echo "Queue:                 ${PBS_O_QUEUE:-unknown}"
echo "Working dir:           $PBS_O_WORKDIR"
echo "Python:                $(which python3)"
echo "Conda env:             $CONDA_DEFAULT_ENV"
echo "Output dir:            $OUTPUT_DIR"
echo "Scratch root:          $SCRATCH_ROOT"
echo "------------------------------------------------------------"
echo "N_WORKERS              : $N_WORKERS"
echo "N_PARAMS_PER_WORKER    : $N_PARAMS_PER_WORKER"
echo "sims per topology      : $((N_WORKERS * N_PARAMS_PER_WORKER))"
echo "N_TOPOLOGIES (cap)     : $N_TOPOLOGIES"
echo "conn_prob              : U[$CONN_PROB_LO, $CONN_PROB_HI]"
echo "MODE                   : $MODE   (Nn=$NN, Na=$NA, c_max=$C_MAX um)"
echo "SWEEP_GROUP            : $SWEEP_GROUP   (free axes; others frozen at nominal)"
echo "Simtime                : $SIMTIME s"
echo "Topology rule          : $TOPOLOGY_MODE   (gj_dist=$GJ_DIST [distance]   gj_max_dist=$GJ_MAX_DIST [wallach]   stoa_cutoff=$STOA_CUTOFF um)"
echo "MAKEFLAGS              : $MAKEFLAGS   (per-worker single-threaded compile)"
echo "============================================================"

# --- Build argument list ------------------------------------------------
ARGS=(
    --out_dir              "$OUTPUT_DIR"
    --lib_dir              "$LIB_DIR"
    --scratch_root         "$SCRATCH_ROOT"
    --n_workers            "$N_WORKERS"
    --n_topologies         "$N_TOPOLOGIES"
    --n_params_per_worker  "$N_PARAMS_PER_WORKER"
    --conn_prob_lo         "$CONN_PROB_LO"
    --conn_prob_hi         "$CONN_PROB_HI"
    --simtime              "$SIMTIME"
    --mode                 "$MODE"
    --sweep_group          "$SWEEP_GROUP"
    --Nn                   "$NN"
    --Na                   "$NA"
    --c_max                "$C_MAX"
    --displ_bias           "$DISPL_BIAS"
    --topology_mode        "$TOPOLOGY_MODE"
    --gj_dist              "$GJ_DIST"
    --gj_max_dist          "$GJ_MAX_DIST"
    --stoa_cutoff          "$STOA_CUTOFF"
    --stoa_sigma           "$STOA_SIGMA"
    --seed_device          "$SEED_DEVICE"
    --seed_neuron          "$SEED_NEURON"
    --seed_synapse         "$SEED_SYNAPSE"
    --seed_astro           "$SEED_ASTRO"
    --dpi                  "$DPI"
)

if [ -n "$SYN_PDIST_CSV" ]; then
    ARGS+=(--syn_pdist_csv "$SYN_PDIST_CSV")
fi
if [ -n "$SEED_MASTER" ]; then
    ARGS+=(--seed_master "$SEED_MASTER")
fi

# --- Run ----------------------------------------------------------------
python "$SCRIPT_PATH" "${ARGS[@]}"
EXIT_CODE=$?

# --- Post-mortem manifest rebuild (in case the run was killed) ----------
# This is a no-op if main() reached its final rebuild_manifest() call,
# and a rescue if walltime killed us mid-topology.
echo
echo "[post] rebuilding manifest from per-iter JSON sidecars (rescue) ..."
python "$SCRIPT_PATH" --rebuild_manifest_only --out_dir "$OUTPUT_DIR" || true

# --- Cleanup ------------------------------------------------------------
# The cpp_standalone build dirs on TMPDIR are auto-purged when the job
# ends, so nothing more to do there.

conda deactivate
sleep 5s

exit $EXIT_CODE
