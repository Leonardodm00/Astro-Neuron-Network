#!/bin/bash
#PBS -S /bin/bash
#PBS -N asn_kernel_smoke
#PBS -q intel
#PBS -l select=1:ncpus=4,walltime=01:00:00
#PBS -k eo
##########################################################################
# Kernel-form smoke test, run THROUGH HPC_single_run.py.
#
# Runs the SAME single-culture driver you use in production, twice, on a
# 1 mm^2 culture in NEURON+SYNAPSE-ONLY mode (no astrocytes), differing only
# in the distance-kernel shape:
#
#   culture_beta1 : --conn_rule weibull --beta_conn 1  (exponential)
#                   --d0_conn = sigma
#   culture_beta2 : --conn_rule weibull --beta_conn 2  (Gaussian)
#                   --d0_conn = sqrt(2)*sigma
#
# d0 is set by the INTEGRATED-COUNT MATCH so the analytic mean in-degree
#   <k>_inf = 2*pi*rho*p0*sigma^2
# is identical for the two cultures; the only difference is the kernel tail.
# Both cultures share neuron positions and the common-random-number edge
# field (same seed_neuron / seed_synapse), and seed_run is OMITTED (paired
# comparison), exactly as HPC_single_run documents.
#
# After the two runs, prepare_kernel_comparison.py repackages the two output
# directories (topology.npz + iter_*.npz) into the analyzer schema
# (culture_beta1.npz, culture_beta2.npz, meta.json) in SMOKE_ROOT, ready for
# submit_analyze_kernel.sh.
#
# PASS per culture:  HPC_single_run exits 0  AND  >=1 iter_*.npz produced.
#
# Run either way:
#     qsub submit_smoke_kernel.sh        # on the cluster
#     bash submit_smoke_kernel.sh        # locally
##########################################################################

# --- USER CONFIG --------------------------------------------------------
SCRIPT_PATH="./HPC_single_run.py"      # the production single-run driver
LIB_DIR="."                            # holds ASD_fun_BD_cpp.py + synapse_pdist.csv
ADAPTER="./prepare_kernel_comparison.py"
SYN_PDIST_CSV=""                       # empty -> auto-discover in LIB_DIR

JOBTAG="${PBS_JOBID%%.*}"; [ -z "$JOBTAG" ] && JOBTAG="local"
SMOKE_ROOT="./smoke_kernel_${JOBTAG}"

# --- arena / population / kernel ---------------------------------------
DENSITY=2000           # neurons / mm^2  (Nn derived from c_max -> 1 mm^2 = 2000 neurons)
C_MAX=1000             # arena side [um]  (1 mm^2)
SIGMA_UM=130           # Gaussian lateral spread (Campagnola-anchored)
P0=0.10                # contact probability at d=0 (shared across beta)
CONN_PERIODIC=1        # 1 -> pass --conn_periodic (minimum-image torus); 0 -> bounded

SIMTIME=2.0            # seconds (short smoke; lengthen for stable burst statistics)

# --- operating point ----------------------------------------------------
# NOTE: at the nominal centre the network is essentially silent (sub-rheobase
# bias, noise-driven), so the comparison would be trivial. These three swept
# axes push it into an ACTIVE, recurrence-coordinated regime where the kernel
# form is measurable. Set to your own active sweep point, or blank them out
# (and drop the flags below) to run at the nominal centre.
SIGMA_MV=12            # --Sigma  : CAdEx noise amplitude (mV)  [params idx 0]
G_AMPA=9.0             # --g_ampa : AMPA conductance (nS)        [idx 14]
G_NMDA=2.5             # --g_nmda : NMDA conductance (nS)        [idx 15]

# --- seeds (SAME for both cultures -> paired comparison) ----------------
SEED_DEVICE=50; SEED_NEURON=39; SEED_SYNAPSE=35
DPI=80
# ------------------------------------------------------------------------

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Environment (no-ops locally if modules / conda are absent)
module load python 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate brian_env 2>/dev/null || true
fi

if [ -n "$PBS_NCPUS" ]; then NC="$PBS_NCPUS"; else NC=$(nproc 2>/dev/null || echo 1); fi
export OMP_NUM_THREADS=1               # single-threaded run (Brian2 cpp_standalone)
export MAKEFLAGS="-j${NC}"             # parallelise the g++ compile only

mkdir -p "$SMOKE_ROOT"

# d0 for the two kernels (integrated-count match): exp -> sigma, gauss -> sqrt(2)*sigma
D0_EXP="$SIGMA_UM"
D0_GAUSS=$(python3 -c "import math;print(math.sqrt(2.0)*${SIGMA_UM})")

PERIODIC_FLAG=""; [ "$CONN_PERIODIC" = "1" ] && PERIODIC_FLAG="--conn_periodic"

echo "============================================================"
echo "Kernel-form smoke (via HPC_single_run)   node=$(hostname)  job=${JOBTAG}"
echo "mode=Neuronal  density=${DENSITY}/mm^2  c_max=${C_MAX}um  sigma=${SIGMA_UM}um  p0=${P0}"
echo "d0(exp)=${D0_EXP}um  d0(gauss)=${D0_GAUSS}um  simtime=${SIMTIME}s  periodic=${CONN_PERIODIC}"
echo "operating point: Sigma=${SIGMA_MV}mV g_ampa=${G_AMPA}nS g_nmda=${G_NMDA}nS"
echo "output root: $SMOKE_ROOT"
echo "============================================================"

# --- one culture --------------------------------------------------------
# $1 = beta   $2 = d0_conn   $3 = output subdir
run_culture () {
    local beta="$1"; local d0="$2"; local out="$3"
    echo
    echo "------------------------------------------------------------"
    echo "CULTURE beta=${beta}  d0_conn=${d0}um  ->  ${out}"
    echo "------------------------------------------------------------"
    local ARGS=(
        --out_dir       "$out"
        --lib_dir       "$LIB_DIR"
        --mode          Neuronal           # neurons + synapses only (NO astrocytes)
        --conn_rule     weibull
        $PERIODIC_FLAG
        --density       "$DENSITY"
        --c_max         "$C_MAX"
        --p0_conn       "$P0"
        --d0_conn       "$d0"
        --beta_conn     "$beta"
        --simtime       "$SIMTIME"
        --Sigma         "$SIGMA_MV"
        --g_ampa        "$G_AMPA"
        --g_nmda        "$G_NMDA"
        --seed_device   "$SEED_DEVICE"
        --seed_neuron   "$SEED_NEURON"
        --seed_synapse  "$SEED_SYNAPSE"
        --dpi           "$DPI"
        # seed_run intentionally OMITTED -> paired comparison
    )
    [ -n "$SYN_PDIST_CSV" ] && ARGS+=(--syn_pdist_csv "$SYN_PDIST_CSV")

    python "$SCRIPT_PATH" "${ARGS[@]}"
    local ec=$?
    local n_npz; n_npz=$(find "$out" -name 'iter_*.npz' 2>/dev/null | wc -l)
    echo "[beta=${beta}] exit=${ec}   iter_*.npz produced=${n_npz}"
    [ "$ec" -eq 0 ] && [ "$n_npz" -ge 1 ] && { echo "[beta=${beta}] PASS"; return 0; }
    echo "[beta=${beta}] FAIL"; return 1
}

FAILS=0
run_culture 1.0 "$D0_EXP"   "${SMOKE_ROOT}/culture_beta1" || FAILS=$((FAILS+1))
run_culture 2.0 "$D0_GAUSS" "${SMOKE_ROOT}/culture_beta2" || FAILS=$((FAILS+1))

# --- repackage into the analyzer schema (pure numpy) --------------------
if [ "$FAILS" -eq 0 ]; then
    echo
    echo "------------------------------------------------------------"
    echo "Repackaging both cultures for the analyzer ..."
    echo "------------------------------------------------------------"
    PERIODIC_ADAPT=""; [ "$CONN_PERIODIC" = "1" ] && PERIODIC_ADAPT="--periodic"
    python "$ADAPTER" \
        --beta1_dir "${SMOKE_ROOT}/culture_beta1" \
        --beta2_dir "${SMOKE_ROOT}/culture_beta2" \
        --out_dir   "${SMOKE_ROOT}" \
        --sigma_um "$SIGMA_UM" --p0 "$P0" \
        --density_per_mm2 "$DENSITY" --c_max "$C_MAX" \
        --simtime_s "$SIMTIME" $PERIODIC_ADAPT \
        || FAILS=$((FAILS+1))
    # stable handle for submit_analyze_kernel.sh
    ln -sfn "$(basename "$SMOKE_ROOT")" "./smoke_kernel_latest"
fi

echo
echo "============================================================"
if [ "$FAILS" -eq 0 ]; then
    echo "KERNEL SMOKE: PASSED  ->  ${SMOKE_ROOT}"
    echo "Next:  qsub submit_analyze_kernel.sh   (or: IN_DIR=${SMOKE_ROOT} bash submit_analyze_kernel.sh)"
else
    echo "KERNEL SMOKE: ${FAILS} FAILURE(S) - inspect logs and ${SMOKE_ROOT}"
fi
echo "============================================================"

if command -v conda >/dev/null 2>&1; then conda deactivate 2>/dev/null || true; fi
exit "$FAILS"
