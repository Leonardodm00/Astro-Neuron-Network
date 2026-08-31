#!/bin/bash
#PBS -S /bin/bash
#PBS -N meama_campaign
#PBS -k eo
##########################################################################
# MEAMA campaign worker -- MEA Manifold Analysis.
#
# Full-mode tripartite sweep at MEA well scale, sized from the benchmark
# (MEAMA/bench_sizing.sh + MEAMA/bench_report.py).  Output feeds two
# downstream stages: SBI posterior fitting against control and pathological
# MEA recordings, and manifold analysis of the resulting summary-statistic
# space.
#
# DO NOT qsub this directly -- submit via MEAMA/launch_MEAMA.sh, which
# resolves REPO_DIR on the login node and injects every sizing variable with
# qsub -v.  A PBS job script is spooled to a scratch path, so this file cannot
# locate the repository by itself.
#
# WHAT IS INJECTED (all with -v, all with a safe default here):
#   REPO_DIR            absolute path of Phenomenological_finalv1   [REQUIRED]
#   NODETAG             queue tag, prevents output-dir collisions   [REQUIRED]
#   SEED_BASE           SEED_MASTER = SEED_BASE + PBS_ARRAY_INDEX
#   N_TOPOLOGIES        topologies per task      (from bench_report)
#   N_PARAMS_PER_WORKER k, params per worker     (from bench_report; 1 here)
#   C_MAX               culture side [um]        (from bench_report)
#   SIMTIME             biological time [s]      (from bench_report)
#   DENSITY             neurons/mm^2             (derived in the launcher)
#   DENSITY_ASTRO       astrocytes/mm^2          (derived in the launcher)
#   MODE, SWEEP_GROUP, CONN_RULE, CONN_PERIODIC, STOA_GATE_LO/HI
#
# There is deliberately NO "MUST MATCH the launcher" constant left in this
# file: every quantity that appears in both places is injected from one source
# of truth in launch_MEAMA.sh.  The historical N_TOPOLOGIES/K_PARAMS drift
# between the two scripts is structurally impossible here.
##########################################################################

set -u

if [ -z "${NODETAG:-}" ]; then
    echo "ERROR: submit via MEAMA/launch_MEAMA.sh (NODETAG must be passed" >&2
    echo "       with qsub -v). Refusing to run un-tagged." >&2
    exit 2
fi
REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}"
if [ ! -f "${REPO_DIR}/HPC_main_sweep.py" ]; then
    echo "ERROR: REPO_DIR=${REPO_DIR} does not contain HPC_main_sweep.py." >&2
    exit 2
fi
if [ -n "${SEED_BASE:-}" ] && ! [ "$SEED_BASE" -ge 0 ] 2>/dev/null; then
    echo "ERROR: SEED_BASE=${SEED_BASE} is not a non-negative integer." >&2
    exit 2
fi
IDX="${PBS_ARRAY_INDEX:-0}"

# --- injected sizing (defaults are conservative, not authoritative) --------
CAMPAIGN_TAG="${CAMPAIGN_TAG:-meama_rho1600_full_v1}"
N_TOPOLOGIES="${N_TOPOLOGIES:-40}"
N_PARAMS_PER_WORKER="${N_PARAMS_PER_WORKER:-1}"
C_MAX="${C_MAX:-350}"
SIMTIME="${SIMTIME:-200}"
DENSITY="${DENSITY:-800}"
DENSITY_ASTRO="${DENSITY_ASTRO:-800}"

MODE="${MODE:-Full}"
SWEEP_GROUP="${SWEEP_GROUP:-tripartite}"
CONN_RULE="${CONN_RULE:-flat}"
CONN_PERIODIC="${CONN_PERIODIC:-0}"
CONN_PROB_LO="${CONN_PROB_LO:-0.1}"
CONN_PROB_HI="${CONN_PROB_HI:-0.6}"
STOA_GATE_LO="${STOA_GATE_LO:-0.2}"
STOA_GATE_HI="${STOA_GATE_HI:-1.0}"
SEED_STOA_GATE=71

# --- fixed simulator configuration (identical to bench_sizing.sh) ---------
NN=115                               # ignored: DENSITY is always set here
NA=115
DISPL_BIAS=20
TOPOLOGY_MODE="wallach"
GJ_DIST=200
GJ_MAX_DIST=150
STOA_CUTOFF=70
STOA_SIGMA=200
SEED_DEVICE=50
SEED_NEURON=39
SEED_SYNAPSE=35
SEED_ASTRO=60
DPI=150
# --------------------------------------------------------------------------

# Campaign output lives at the REPOSITORY level, NOT inside MEAMA/, so that
# seed_alloc.py's campaign_*/sweep_*/job_args.json scan, aggregate_sweep.py,
# status_sweep.py and the MEA pipeline all find it where they expect. Only the
# launch CODE lives in MEAMA/.
CAMPAIGN_ROOT="${REPO_DIR}/campaign_${CAMPAIGN_TAG}"
OUTPUT_DIR="${CAMPAIGN_ROOT}/sweep_${NODETAG}_task$(printf '%04d' ${IDX})"

SCRIPT_PATH="${REPO_DIR}/HPC_main_sweep.py"
LIB_DIR="${REPO_DIR}"

cd "$REPO_DIR" || exit 2
# Environment. MEAMA_NO_ENV=1 skips module/conda activation -- for verifying
# this script inside an ALREADY-ACTIVE environment (an interactive login-node
# session, or a local check). It is never set by the launchers, so a scheduled
# job always takes the normal path.
CONDA_ENV="${CONDA_ENV:-brian_final}"     # injected via qsub -v; see launchers
if [ -z "${MEAMA_NO_ENV:-}" ]; then
    module load python
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV" || {
        echo "ERROR: conda activate ${CONDA_ENV} failed." >&2; exit 2; }
    # Guard against the module-loaded system python winning on PATH, and
    # against a damaged environment (brian_env was found with an incomplete
    # stdlib: lib/python3.11/urllib/ missing). Both fail loudly here rather
    # than deep inside a simulation hours later.
    python - <<'PYCHK' || { echo "ERROR: ${CONDA_ENV} is not usable." >&2; exit 2; }
import os, sys, urllib, pathlib          # urllib: the exact brian_env failure
p = os.environ.get('CONDA_PREFIX', '')
assert p and sys.executable.startswith(p), \
    f"python is {sys.executable}, not from CONDA_PREFIX={p}"
import brian2, numpy
print(f"[env] {sys.executable} | brian2 {brian2.__version__} | numpy {numpy.__version__}")
PYCHK
else
    echo "[env] MEAMA_NO_ENV=1 -- using the ambient python, NOT ${CONDA_ENV}."
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
echo "MEAMA campaign / node / task : ${CAMPAIGN_TAG} / ${NODETAG} / ${IDX}"
echo "Node / JobID / Queue   : $(hostname) / ${PBS_JOBID:-?} / ${PBS_O_QUEUE:-?}"
echo "REPO_DIR               : ${REPO_DIR}"
echo "N_WORKERS (= ncpus)    : $N_WORKERS"
echo "N_TOPOLOGIES           : $N_TOPOLOGIES   k=$N_PARAMS_PER_WORKER"
echo "sims this task         : $((N_TOPOLOGIES * N_WORKERS * N_PARAMS_PER_WORKER))"
echo "distinct topologies    : $N_TOPOLOGIES  (k=1 => one topology per sim per worker)"
echo "C_MAX / SIMTIME        : ${C_MAX} um / ${SIMTIME} s"
echo "density n / a          : ${DENSITY} / ${DENSITY_ASTRO} per mm^2"
echo "LAUNCH_ID              : ${LAUNCH_ID:-<not set>}"
echo "SEED_MASTER            : ${SEED_MASTER:-<deferred>}   ($SEED_SOURCE)"
echo "Output dir             : $OUTPUT_DIR"
echo "MODE / SWEEP_GROUP     : $MODE / $SWEEP_GROUP"
echo "CONN_RULE              : $CONN_RULE   (periodic=$CONN_PERIODIC)"
echo "STOA_GATE              : U[${STOA_GATE_LO}, ${STOA_GATE_HI}]"
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

[ -z "${MEAMA_NO_ENV:-}" ] && conda deactivate
sleep 5s
exit $EXIT_CODE
