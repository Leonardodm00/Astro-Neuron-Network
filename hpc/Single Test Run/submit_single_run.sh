#!/bin/bash
# =============================================================================
# submit_single_run.sh
#
# SLURM submission script for HPC_single_run.py  (two-pass, self-contained).
# No pre-computed connectivity directory is needed.
#
# Tested layout: CINECA Leonardo / Galileo100.
# Adjust the #SBATCH directives and module names for your cluster.
#
# Quickstart:
#   1. Edit the four paths in the "Paths" section below.
#   2. Edit account and partition to match your allocation.
#   3. sbatch submit_single_run.sh
# =============================================================================

# ── Resource allocation ─────────────────────────────────────────────────────
#SBATCH --job-name=asn_single_run
#SBATCH --partition=g100_usr_prod           # adjust for your cluster
#SBATCH --account=YOUR_ACCOUNT_HERE         # mandatory on CINECA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                   # used during C++ compilation (make -j)
#SBATCH --mem=8G                            # 8 GB is comfortable for Nn≤200, Na≤80
#SBATCH --time=02:00:00                     # 2 h ceiling: ~10 min compile + run time
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL               # remove if you don't want email alerts
#SBATCH --mail-user=your.email@domain.it

# ── Environment banner ───────────────────────────────────────────────────────
echo "============================================================"
echo "  Job        : $SLURM_JOB_NAME  (#$SLURM_JOB_ID)"
echo "  Node       : $SLURMD_NODENAME"
echo "  Partition  : $SLURM_JOB_PARTITION"
echo "  Submit dir : $SLURM_SUBMIT_DIR"
echo "  Start      : $(date)"
echo "============================================================"

# ── Modules ──────────────────────────────────────────────────────────────────
# Module names below are valid on CINECA Galileo100 and Leonardo.
# Run 'module avail python' and 'module avail gsl' to check names on your cluster.
module purge
module load python/3.10.8--gcc--11.3.0
module load gsl/2.7--gcc--11.3.0           # required for Brian2 GSL integrators

# If GSL is not available as a module, install it locally and point the
# compiler to it:
#   export CPATH=$HOME/.local/include:$CPATH
#   export LIBRARY_PATH=$HOME/.local/lib:$LIBRARY_PATH
#   export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

# ── Virtual environment ───────────────────────────────────────────────────────
# Edit the path to your Python virtual environment.
VENV_PATH="$HOME/venvs/brian2_env"
source "${VENV_PATH}/bin/activate"

python -c "import brian2;     print('Brian2      :', brian2.__version__)"
python -c "import numpy;      print('NumPy       :', numpy.__version__)"
python -c "import scipy;      print('SciPy       :', scipy.__version__)"
python -c "import matplotlib; print('Matplotlib  :', matplotlib.__version__)"

# ── Paths ─────────────────────────────────────────────────────────────────────
# Directory containing HPC_single_run.py and ASD_fun_BD_cpp.py.
SCRIPT_DIR="$HOME/ASN"

# Where all output files will be written for this job.
# Embedding $SLURM_JOB_ID keeps different submissions from colliding.
OUT_DIR="$SCRATCH/ASN_results/single_run_${SLURM_JOB_ID}"

# ── Compiler parallelism ─────────────────────────────────────────────────────
# Let make exploit all allocated CPUs during the cpp_standalone build step.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MAKEFLAGS="-j${SLURM_CPUS_PER_TASK}"

# Build the C++ binary in $TMPDIR (compute-node local SSD) to avoid thousands
# of small-file metadata operations on the shared Lustre filesystem.
export BRIAN2_STANDALONE_DIR="${TMPDIR}/brian2_${SLURM_JOB_ID}"

# ── Simulation settings ───────────────────────────────────────────────────────
SIMTIME=180        # simulated duration  [seconds]
MODE=Full          # Full | Neuronal

# ── Population sizes ──────────────────────────────────────────────────────────
NN=100             # number of neurons
NA=43              # number of astrocytes (ignored when MODE=Neuronal)

# ── Arena ─────────────────────────────────────────────────────────────────────
C_MAX=1100         # side length of the square arena  [µm]

# ── Connectivity parameters ───────────────────────────────────────────────────
CONN_PROB=0.107    # neuron→neuron Bernoulli connection probability
DISPL_BIAS=15      # constant offset added to every sampled distance  [µm]
GJ_DIST=200        # KDTree radius for gap-junction coupling  [µm]
STOA_CUTOFF=70     # hard distance cutoff for synapse→astrocyte links  [µm]
STOA_SIGMA=200     # σ of Gaussian StoA connection probability  [µm]

# Path to synapse_pdist.csv (columns: Syn_prob, Radius_val).
# If left empty, the script auto-discovers it next to ASD_fun_BD_cpp.py.
SYN_PDIST_CSV=""

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEED_DEVICE=50
SEED_NEURON=39
SEED_SYNAPSE=35
SEED_ASTRO=60
# SEED_RUN=42      # uncomment to re-seed stochastic terms independently

# ── Swept parameters (units as noted; omitting a flag uses the nominal value) ─
#
#   Sigma       [mV]          g_AHP       [nS]
#   Xi_ampa     [1/mmole]     Xi_nmda     [1/mmole]
#   Tau_Ca      [s]           U_0_ar      [dimensionless]
#   U_max       [1/ms]        U_0_sr      [dimensionless]
#   Omega_f_sr  [1/s]         Omega_f_ar  [1/s]
#   Omega_d     [1/s]         alpha_syn   [dimensionless]
#   g_na  (coefficient, SI = coeff × mS cm⁻² × area;  nominal: 1.6 × 50 = 80)
#   g_kd  (coefficient, SI = coeff × mS cm⁻² × area;  nominal: 1.3 ×  5 = 6.5)

SIGMA=4.0
G_AHP=5.0
XI_AMPA=0.5
XI_NMDA=0.3
TAU_CA=8.0
U_0_AR=0.003
U_MAX=0.5
U_0_SR=0.15
OMEGA_F_SR=2.0
OMEGA_F_AR=1.42857
OMEGA_D=2.0
ALPHA_SYN=1.0
G_NA=80.0
G_KD=6.5

DPI=200

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "--- Launching HPC_single_run.py ---"
echo "  Script dir : $SCRIPT_DIR"
echo "  Output dir : $OUT_DIR"
echo "  Mode       : $MODE   Nn=$NN   Na=$NA"
echo "  Simtime    : $SIMTIME s"
echo "  conn_prob  : $CONN_PROB"
echo ""

python "${SCRIPT_DIR}/HPC_single_run.py" \
    --out_dir      "$OUT_DIR"            \
    --lib_dir      "$SCRIPT_DIR"         \
    --simtime      "$SIMTIME"            \
    --mode         "$MODE"               \
    --Nn           "$NN"                 \
    --Na           "$NA"                 \
    --c_max        "$C_MAX"              \
    --conn_prob    "$CONN_PROB"          \
    --displ_bias   "$DISPL_BIAS"         \
    ${SYN_PDIST_CSV:+--syn_pdist_csv "$SYN_PDIST_CSV"} \
    --gj_dist      "$GJ_DIST"           \
    --stoa_cutoff  "$STOA_CUTOFF"        \
    --stoa_sigma   "$STOA_SIGMA"         \
    --seed_device  "$SEED_DEVICE"        \
    --seed_neuron  "$SEED_NEURON"        \
    --seed_synapse "$SEED_SYNAPSE"       \
    --seed_astro   "$SEED_ASTRO"         \
    --Sigma        "$SIGMA"              \
    --g_AHP        "$G_AHP"             \
    --Xi_ampa      "$XI_AMPA"           \
    --Xi_nmda      "$XI_NMDA"           \
    --Tau_Ca       "$TAU_CA"            \
    --U_0_ar       "$U_0_AR"            \
    --U_max        "$U_MAX"             \
    --U_0_sr       "$U_0_SR"            \
    --Omega_f_sr   "$OMEGA_F_SR"        \
    --Omega_f_ar   "$OMEGA_F_AR"        \
    --Omega_d      "$OMEGA_D"           \
    --alpha_syn    "$ALPHA_SYN"         \
    --g_na         "$G_NA"              \
    --g_kd         "$G_KD"              \
    --dpi          "$DPI"
    # Uncomment to re-seed stochastic terms:
    # --seed_run   "$SEED_RUN"

EXITCODE=$?

echo ""
echo "============================================================"
echo "  End  : $(date)"
echo "  Exit : $EXITCODE"
echo "============================================================"

exit $EXITCODE
