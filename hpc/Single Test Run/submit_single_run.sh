#!/bin/bash

#PBS -S /bin/bash
#PBS -N "asn_single_run"
#PBS -q cpu
#PBS -l select=1:ncpus=1,walltime=01:00:00
#PBS -k eo

##########################################################################
# Single-parameter-set neuron+astrocyte simulation (HPC, cpp_standalone)
#
# Runs HPC_single_run.py: builds topology from scratch (pass 1, numpy),
# compiles cpp_standalone (pass 2), runs once, and emits spike .npz,
# topology .npz, spatial layout PNG, two raster PNGs, and a summary.
#
# EDIT THE VARIABLES IN THE "USER CONFIG" BLOCK BEFORE SUBMITTING.
##########################################################################

# ─── USER CONFIG ────────────────────────────────────────────────────────
# Absolute paths recommended (jobs do not always inherit your $PWD nicely).

SCRIPT_PATH="./HPC_single_run.py"          # path to the python script
LIB_DIR="."                                # dir with ASD_fun_BD_cpp.py
OUTPUT_DIR="./single_run_${PBS_JOBID%%.*}" # results dir (per-job by default)
SYN_PDIST_CSV="."                           # leave empty → auto-discover in LIB_DIR

# ─── Simulation settings ────────────────────────────────────────────────
SIMTIME=300             # simulated duration [seconds]
MODE="Neuronal"             # Full | Neuronal

# ─── Population sizes ───────────────────────────────────────────────────
NN=100                  # number of neurons
NA=43                   # number of astrocytes (ignored when MODE=Neuronal)

# ─── Arena ──────────────────────────────────────────────────────────────
C_MAX=1100              # side length of the square arena [µm]

# ─── Connectivity parameters ────────────────────────────────────────────
CONN_PROB=0.107         # neuron→neuron Bernoulli connection probability
DISPL_BIAS=15           # constant offset added to every sampled distance [µm]

# Astrocyte topology rule — Wallach et al. 2014 joint-Voronoi (default)
# or legacy distance-based. See Report5_Topology_Wallach.md.
TOPOLOGY_MODE="wallach" # wallach | distance
GJ_DIST=200             # [distance only] KDTree radius for GJC [µm]
GJ_MAX_DIST=150         # [wallach  only] soft cap on Voronoi-adjacent GJC [µm]
STOA_CUTOFF=70          # hard distance cutoff for synapse→astrocyte links [µm]
STOA_SIGMA=200          # [distance only] σ of Gaussian StoA acceptance [µm]

# ─── Seeds ──────────────────────────────────────────────────────────────
SEED_DEVICE=50
SEED_NEURON=39
SEED_SYNAPSE=35
SEED_ASTRO=60
SEED_RUN=""             # empty → replay device seed; integer → re-seed stochastic terms

# ─── 14 swept parameters (units shown next to each) ─────────────────────
SIGMA=4.0               # [mV]
G_AHP=5.0               # [nS]
XI_AMPA=0.5             # [1/mmole]
XI_NMDA=0.3             # [1/mmole]
TAU_CA=8.0              # [s]
U_0_AR=0.003            # [dimensionless]
U_MAX=0.5               # [1/ms]
U_0_SR=0.15             # [dimensionless]
OMEGA_F_SR=2.0          # [1/s]
OMEGA_F_AR=1.42857      # [1/s]
OMEGA_D=2.0             # [1/s]
ALPHA_SYN=1.0           # [dimensionless]
G_NA=80.0               # g_na coefficient (SI = coeff × mS cm⁻² × area)
G_KD=6.5                # g_kd coefficient (SI = coeff × mS cm⁻² × area)

# ─── Figure options ─────────────────────────────────────────────────────
DPI=300
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

# ─── Compiler parallelism for cpp_standalone build step ─────────────────
# Brian2 uses 'make' to build the standalone binary; let it use all CPUs.
NCPUS_AVAILABLE=$(nproc 2>/dev/null || echo 1)
export OMP_NUM_THREADS=$NCPUS_AVAILABLE
export MAKEFLAGS="-j${NCPUS_AVAILABLE}"

# Build the C++ binary on local scratch when available (avoids thousands
# of small-file metadata operations on shared filesystems).
if [ -n "$TMPDIR" ]; then
    export BRIAN2_STANDALONE_DIR="${TMPDIR}/brian2_${PBS_JOBID%%.*}"
fi

# Sanity prints
echo "Running on node:       $(hostname)"
echo "Job ID:                $PBS_JOBID"
echo "Working dir:           $PBS_O_WORKDIR"
echo "Python:                $(which python3)"
echo "Conda env:             $CONDA_DEFAULT_ENV"
echo "Script:                $SCRIPT_PATH"
echo "Library dir:           $LIB_DIR"
echo "Output dir:            $OUTPUT_DIR"
echo "Mode:                  $MODE   (Nn=$NN, Na=$NA, c_max=$C_MAX µm)"
echo "Simtime:               $SIMTIME s"
echo "Topology rule:         $TOPOLOGY_MODE   (gj_dist=$GJ_DIST µm [distance]   gj_max_dist=$GJ_MAX_DIST µm [wallach])"
echo "Connectivity:          p=$CONN_PROB   stoa_cutoff=$STOA_CUTOFF µm   stoa_sigma=$STOA_SIGMA µm [distance]"
echo "Bouton placement:      Sholl PDF + displ_bias=$DISPL_BIAS µm"
echo "Seeds:                 device=$SEED_DEVICE  neuron=$SEED_NEURON  synapse=$SEED_SYNAPSE  astro=$SEED_ASTRO  run=${SEED_RUN:-replay}"
echo "Compilation:           OMP_NUM_THREADS=$OMP_NUM_THREADS   MAKEFLAGS='$MAKEFLAGS'"
echo "-----------------------------------------"

# ─── Build the argument list ────────────────────────────────────────────
ARGS=(
    --out_dir       "$OUTPUT_DIR"
    --lib_dir       "$LIB_DIR"
    --simtime       "$SIMTIME"
    --mode          "$MODE"
    --Nn            "$NN"
    --Na            "$NA"
    --c_max         "$C_MAX"
    --conn_prob     "$CONN_PROB"
    --displ_bias    "$DISPL_BIAS"
    --topology_mode "$TOPOLOGY_MODE"
    --gj_dist       "$GJ_DIST"
    --gj_max_dist   "$GJ_MAX_DIST"
    --stoa_cutoff   "$STOA_CUTOFF"
    --stoa_sigma    "$STOA_SIGMA"
    --seed_device   "$SEED_DEVICE"
    --seed_neuron   "$SEED_NEURON"
    --seed_synapse  "$SEED_SYNAPSE"
    --seed_astro    "$SEED_ASTRO"
    --Sigma         "$SIGMA"
    --g_AHP         "$G_AHP"
    --Xi_ampa       "$XI_AMPA"
    --Xi_nmda       "$XI_NMDA"
    --Tau_Ca        "$TAU_CA"
    --U_0_ar        "$U_0_AR"
    --U_max         "$U_MAX"
    --U_0_sr        "$U_0_SR"
    --Omega_f_sr    "$OMEGA_F_SR"
    --Omega_f_ar    "$OMEGA_F_AR"
    --Omega_d       "$OMEGA_D"
    --alpha_syn     "$ALPHA_SYN"
    --g_na          "$G_NA"
    --g_kd          "$G_KD"
    --dpi           "$DPI"
)

# Optional: explicit path to synapse_pdist.csv (otherwise auto-discovered)
if [ -n "$SYN_PDIST_CSV" ]; then
    ARGS+=(--syn_pdist_csv "$SYN_PDIST_CSV")
fi

# Optional: independently re-seed stochastic terms
if [ -n "$SEED_RUN" ]; then
    ARGS+=(--seed_run "$SEED_RUN")
fi

# ─── Run ────────────────────────────────────────────────────────────────
python "$SCRIPT_PATH" "${ARGS[@]}"

# Clean up
conda deactivate

sleep 5s
