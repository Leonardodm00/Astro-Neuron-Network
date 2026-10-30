#!/bin/bash
#PBS -S /bin/bash
#PBS -N lcdup_campaign
#PBS -k eo
##########################################################################
# LindaColloDUP campaign worker -- neurons only (MODE=Neuronal), all
# excitatory, 13 free axes (the rest frozen with --freeze).
#
# Derived from MEAMA/submit_MEAMA.sh (2026-10-29). DO NOT qsub this
# directly: submit via launch_LCDUP.sh, which resolves REPO_DIR on the login
# node and injects every campaign setting with qsub -v.
#
# WHAT IS INJECTED (qsub -v):
#   REPO_DIR            absolute path of the LindaColloDUP folder   [REQUIRED]
#   NODETAG             queue tag, prevents output-dir collisions   [REQUIRED]
#   FREEZE_ENC          freeze list, ',' sent as ':' and '=' as '@' [REQUIRED]
#   SEED_BASE           SEED_MASTER = SEED_BASE + PBS_ARRAY_INDEX
#   CAMPAIGN_TAG, N_TOPOLOGIES, N_PARAMS_PER_WORKER, C_MAX, SIMTIME,
#   DENSITY, DENSITY_ASTRO, MODE, SWEEP_GROUP, CONN_RULE, CONN_PERIODIC,
#   CONN_PROB_LO, CONN_PROB_HI, STOA_GATE_LO, STOA_GATE_HI, CONDA_ENV
# The defaults below equal the launcher's and are only used if this file
# is submitted by hand.
##########################################################################

set -u

if [ -z "${NODETAG:-}" ]; then
    echo "ERROR: submit via launch_LCDUP.sh (NODETAG must be passed with qsub -v)." >&2
    exit 2
fi
if [ -z "${FREEZE_ENC:-}" ]; then
    echo "ERROR: FREEZE_ENC not set: refusing to run the full neuron_synapse group." >&2
    exit 2
fi
REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}"
if [ ! -f "${REPO_DIR}/HPC_main_sweep.py" ]; then
    echo "ERROR: REPO_DIR=${REPO_DIR} does not contain HPC_main_sweep.py." >&2
    exit 2
fi
if ! grep -q "def _apply_freeze(" "${REPO_DIR}/HPC_main_sweep.py"; then
    echo "ERROR: ${REPO_DIR}/HPC_main_sweep.py lacks the --freeze patch." >&2
    exit 2
fi
if [ -n "${SEED_BASE:-}" ] && ! [ "$SEED_BASE" -ge 0 ] 2>/dev/null; then
    echo "ERROR: SEED_BASE=${SEED_BASE} is not a non-negative integer." >&2
    exit 2
fi
IDX="${PBS_ARRAY_INDEX:-0}"

# --- injected settings (defaults = launch_LCDUP.sh defaults) ---------------
CAMPAIGN_TAG="${CAMPAIGN_TAG:-lcdup15_rho1300_neur_n01}"
N_TOPOLOGIES="${N_TOPOLOGIES:-32}"
N_PARAMS_PER_WORKER="${N_PARAMS_PER_WORKER:-1}"
C_MAX="${C_MAX:-350}"
SIMTIME="${SIMTIME:-180}"
DENSITY="${DENSITY:-1300}"
DENSITY_ASTRO="${DENSITY_ASTRO:-0}"
MODE="${MODE:-Neuronal}"
SWEEP_GROUP="${SWEEP_GROUP:-neuron_synapse}"
CONN_RULE="${CONN_RULE:-flat}"
CONN_PERIODIC="${CONN_PERIODIC:-0}"
CONN_PROB_LO="${CONN_PROB_LO:-0.1}"
CONN_PROB_HI="${CONN_PROB_HI:-0.6}"
STOA_GATE_LO="${STOA_GATE_LO:-1.0}"
STOA_GATE_HI="${STOA_GATE_HI:-1.0}"
FREEZE="$(printf '%s' "$FREEZE_ENC" | tr ':@' ',=')"

# --- fixed simulator configuration (as MEAMA, except NA) -------------------
NN=115                               # ignored: DENSITY is always set here
NA=0                                 # no astrocytes; keeps job_args.json Na = 0
DISPL_BIAS=20
TOPOLOGY_MODE="wallach"              # astrocyte wiring: not built in Neuronal mode
GJ_DIST=200
GJ_MAX_DIST=150
STOA_CUTOFF=70
STOA_SIGMA=200
SEED_STOA_GATE=71
SEED_DEVICE=50
SEED_NEURON=39
SEED_SYNAPSE=35
SEED_ASTRO=60
DPI=150
# --------------------------------------------------------------------------

CAMPAIGN_ROOT="${REPO_DIR}/campaign_${CAMPAIGN_TAG}"
OUTPUT_DIR="${CAMPAIGN_ROOT}/sweep_${NODETAG}_task$(printf '%04d' ${IDX})"
SCRIPT_PATH="${REPO_DIR}/HPC_main_sweep.py"
LIB_DIR="${REPO_DIR}"

cd "$REPO_DIR" || exit 2
# Environment. LCDUP_NO_ENV=1 skips module/conda activation, for checking
# this script inside an already-active environment; the launcher never sets it.
CONDA_ENV="${CONDA_ENV:-brian_final}"
if [ -z "${LCDUP_NO_ENV:-}" ]; then
    set +u                           # module/conda hooks may read unset variables
    module load python
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || {
        echo "ERROR: conda activate ${CONDA_ENV} failed." >&2; exit 2; }
    set -u
    python - <<'PYCHK' || { echo "ERROR: ${CONDA_ENV} is not usable." >&2; exit 2; }
import os, sys, urllib, pathlib
p = os.environ.get('CONDA_PREFIX', '')
assert p and sys.executable.startswith(p), \
    f"python is {sys.executable}, not from CONDA_PREFIX={p}"
import brian2, numpy
print(f"[env] {sys.executable} | brian2 {brian2.__version__} | numpy {numpy.__version__}")
PYCHK
else
    echo "[env] LCDUP_NO_ENV=1 -- using the ambient python, NOT ${CONDA_ENV}."
fi
mkdir -p "$OUTPUT_DIR"

if [ -n "${PBS_NCPUS:-}" ]; then N_WORKERS="$PBS_NCPUS"; else N_WORKERS=$(nproc); fi
export OMP_NUM_THREADS=1
export MAKEFLAGS="-j1"

if [ -n "${TMPDIR:-}" ]; then
    SCRATCH_ROOT="${TMPDIR}/brian2_${CAMPAIGN_TAG}_${NODETAG}_${IDX}"
else
    SCRATCH_ROOT="${OUTPUT_DIR}/_scratch"
fi
mkdir -p "$SCRATCH_ROOT"

if [ -n "${SEED_BASE:-}" ]; then
    SEED_MASTER=$(( SEED_BASE + IDX ))
    SEED_SOURCE="launcher base ${SEED_BASE} + idx ${IDX}"
else
    SEED_MASTER=""
    SEED_SOURCE="NOT SET -- worker derives from PBS_JOBID/OS entropy"
fi

echo "============================================================"
echo "LindaColloDUP campaign / node / task : ${CAMPAIGN_TAG} / ${NODETAG} / ${IDX}"
echo "Node / JobID / Queue   : $(hostname) / ${PBS_JOBID:-?} / ${PBS_O_QUEUE:-?}"
echo "REPO_DIR               : ${REPO_DIR}"
echo "N_WORKERS (= ncpus)    : $N_WORKERS"
echo "N_TOPOLOGIES           : $N_TOPOLOGIES   k=$N_PARAMS_PER_WORKER"
echo "sims this task         : $((N_TOPOLOGIES * N_WORKERS * N_PARAMS_PER_WORKER))"
echo "C_MAX / SIMTIME        : ${C_MAX} um / ${SIMTIME} s"
echo "density n / a          : ${DENSITY} / ${DENSITY_ASTRO} per mm^2"
echo "MODE / SWEEP_GROUP     : $MODE / $SWEEP_GROUP"
echo "FREEZE                 : $FREEZE"
echo "dV/dt (model variant)  : $(grep -m1 -E '^[[:space:]]*dV/dt[[:space:]]*=' "${REPO_DIR}/ASD_fun_BD_cpp.py" | sed 's/^[[:space:]]*//')"
echo "CONN_RULE / conn_prob  : $CONN_RULE (periodic=$CONN_PERIODIC) / U[${CONN_PROB_LO}, ${CONN_PROB_HI}]"
echo "LAUNCH_ID              : ${LAUNCH_ID:-<not set>}"
echo "SEED_MASTER            : ${SEED_MASTER:-<deferred>}   ($SEED_SOURCE)"
echo "Output dir             : $OUTPUT_DIR"
echo "============================================================"

ARGS=(
    --out_dir              "$OUTPUT_DIR"
    --lib_dir              "$LIB_DIR"
    --scratch_root         "$SCRATCH_ROOT"
    --n_workers            "$N_WORKERS"
    --n_topologies         "$N_TOPOLOGIES"
    --n_params_per_worker  "$N_PARAMS_PER_WORKER"
    --conn_prob_lo         "$CONN_PROB_LO"
    --conn_prob_hi         "$CONN_PROB_HI"
    --conn_rule            "$CONN_RULE"
    --simtime              "$SIMTIME"
    --mode                 "$MODE"
    --sweep_group          "$SWEEP_GROUP"
    --freeze               "$FREEZE"
    --Nn                   "$NN"
    --Na                   "$NA"
    --density              "$DENSITY"
    --density_astro        "$DENSITY_ASTRO"
    --c_max                "$C_MAX"
    --displ_bias           "$DISPL_BIAS"
    --topology_mode        "$TOPOLOGY_MODE"
    --gj_dist              "$GJ_DIST"
    --gj_max_dist          "$GJ_MAX_DIST"
    --stoa_cutoff          "$STOA_CUTOFF"
    --stoa_gate_lo         "$STOA_GATE_LO"
    --stoa_gate_hi         "$STOA_GATE_HI"
    --seed_stoa_gate       "$SEED_STOA_GATE"
    --stoa_sigma           "$STOA_SIGMA"
    --seed_device          "$SEED_DEVICE"
    --seed_neuron          "$SEED_NEURON"
    --seed_synapse         "$SEED_SYNAPSE"
    --seed_astro           "$SEED_ASTRO"
    --dpi                  "$DPI"
)
[ -n "$SEED_MASTER" ] && ARGS+=(--seed_master "$SEED_MASTER")
[ "$CONN_PERIODIC" -eq 1 ] && ARGS+=(--conn_periodic)

python "$SCRIPT_PATH" "${ARGS[@]}"
EXIT_CODE=$?

echo "[post] rebuilding manifest for ${NODETAG}/task${IDX} ..."
python "$SCRIPT_PATH" --rebuild_manifest_only --out_dir "$OUTPUT_DIR" || true

if [ -z "${LCDUP_NO_ENV:-}" ]; then set +u; conda deactivate; set -u; fi
sleep 5s
exit $EXIT_CODE
