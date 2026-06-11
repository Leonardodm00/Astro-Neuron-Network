#!/bin/bash
#PBS -S /bin/bash
#PBS -N asn_campaign
#PBS -k eo
#PBS -l walltime=24:00:00
##########################################################################
# Resource-AGNOSTIC worker for the 300k campaign.
#
# DO NOT submit this directly - submit it via launch_campaign.sh, which calls
# qsub TWICE (once per queue) and passes the queue, ncpus, array range, and
# the two -v variables this script requires:
#
#     NODETAG    e.g. c192 or c48   -> tags the output dir, prevents collisions
#     SEED_BASE  e.g. 1000 or 100000-> SEED_MASTER = SEED_BASE + array index
#                                       (the two queues MUST use disjoint
#                                        [SEED_BASE, SEED_BASE+N_tasks) ranges)
#
# The worker auto-adapts to the node it lands on:
#     N_WORKERS  = $PBS_NCPUS  (48 on intel/cpu, 192 on cfd/egeos)
#     sims/task  = N_TOPOLOGIES * N_WORKERS * N_PARAMS_PER_WORKER
# Note N_TOPOLOGIES is the SAME for both node types - per-topology wall is set
# by single-sim time, not core count, so a 48- and a 192-core node clear the
# same number of topologies in TARGET_HOURS; the 192 just yields 4x the sims.
##########################################################################

# --- REQUIRED from -v (guarded) -----------------------------------------
if [ -z "$NODETAG" ] || [ -z "$SEED_BASE" ]; then
    echo "ERROR: submit via launch_campaign.sh (NODETAG and SEED_BASE must be"
    echo "       passed with qsub -v). Refusing to run un-tagged." >&2
    exit 2
fi
IDX="${PBS_ARRAY_INDEX:-0}"          # 0 if run as a non-array single task

# --- CAMPAIGN-WIDE CONFIG (edit once; identical for both queues) ---------
# Pre-set for a PESSIMISTIC 800 s/sim assumption - NO pilot required.
#   80 topologies * 800 s = 64,000 s = 17.8 h of compute per task, which leaves
#   margin under the 24 h walltime cap: even if the real per-topology wall runs
#   up to ~1080 s (35% over 800 s, e.g. mild straggler spread), 80 topologies
#   still complete cleanly (80*1080 = 86,400 s = 24 h). If stragglers are worse,
#   the task is walltime-killed safely (each sim saved before the next).
CAMPAIGN_TAG="300k_v1"
TARGET_HOURS=20                      # ~compute target per task (buffer under 24 h)
N_TOPOLOGIES=80                      # pre-sized for 800 s/sim (see note above)
N_PARAMS_PER_WORKER=1                # 3-4 to tame straggler tax; fewer topologies

SCRIPT_PATH="./HPC_main_sweep.py"
LIB_DIR="."
SYN_PDIST_CSV=""
CAMPAIGN_ROOT="./campaign_${CAMPAIGN_TAG}"
OUTPUT_DIR="${CAMPAIGN_ROOT}/sweep_${NODETAG}_task$(printf '%04d' ${IDX})"

CONN_PROB_LO=0.1
CONN_PROB_HI=0.6

# Connectivity rule, injectable via qsub -v (launch_campaign.sh passes these).
#   CONN_RULE     : flat | weibull   (the SBI campaign sets weibull in
#                   launch_campaign.sh; 'flat' here is only the bare-run fallback).
#                   Under weibull the kernel (p0,d0,beta) is DRAWN per topology
#                   from KERNEL_BOUNDS in HPC_main_sweep.py.
#   CONN_PERIODIC : 1 => minimum-image (flat-torus) distances (weibull only).
#   DENSITY       : neurons/mm^2. If set, Nn is DERIVED from C_MAX
#                   (Nn=round(DENSITY*(C_MAX/1000)^2)) and held fixed across
#                   sizes; NN below is then ignored. Empty => use NN as-is.
CONN_RULE="${CONN_RULE:-flat}"
CONN_PERIODIC="${CONN_PERIODIC:-0}"
DENSITY="${DENSITY:-}"

# --- NETWORK / SIM CONFIG (MUST MATCH YOUR BENCHMARK) -------------------
SIMTIME=300
MODE="${MODE:-Full}"                 # Full | Neuronal (optionally injected via qsub -v)
NN=115                               # ignored if DENSITY is set (Nn derived from C_MAX)
NA=115
C_MAX=240                            # <-- value that gave [240 um]2 in your run
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
# Parameter group to sweep (optionally injected by launch_campaign.sh via qsub
# -v SWEEP_GROUP=...). all|neuron|synapse|astro|neuron_synapse. NOTE: in
# MODE=Neuronal the synaptic axes are ALWAYS swept and astro axes are inert, so
# 'all'/'neuron'/'neuron_synapse' all sweep neuron+synapse there.
SWEEP_GROUP="${SWEEP_GROUP:-all}"
# ------------------------------------------------------------------------

cd "$PBS_O_WORKDIR"
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate brian_env
mkdir -p "$OUTPUT_DIR"

if [ -n "$PBS_NCPUS" ]; then N_WORKERS="$PBS_NCPUS"; else N_WORKERS=$(nproc); fi
export OMP_NUM_THREADS=1
export MAKEFLAGS="-j1"

if [ -n "$TMPDIR" ]; then
    SCRATCH_ROOT="${TMPDIR}/brian2_${CAMPAIGN_TAG}_${NODETAG}_${IDX}"
else
    SCRATCH_ROOT="${OUTPUT_DIR}/_scratch"
fi
mkdir -p "$SCRATCH_ROOT"

SEED_MASTER=$(( SEED_BASE + IDX ))

echo "============================================================"
echo "Campaign / node / task : ${CAMPAIGN_TAG} / ${NODETAG} / ${IDX}"
echo "Node / JobID / Queue   : $(hostname) / $PBS_JOBID / ${PBS_O_QUEUE:-?}"
echo "N_WORKERS (= ncpus)    : $N_WORKERS"
echo "N_TOPOLOGIES           : $N_TOPOLOGIES   k=$N_PARAMS_PER_WORKER"
echo "sims this task         : $((N_TOPOLOGIES * N_WORKERS * N_PARAMS_PER_WORKER))"
echo "SEED_MASTER            : $SEED_MASTER   (base $SEED_BASE + idx $IDX)"
echo "Output dir             : $OUTPUT_DIR"
echo "SWEEP_GROUP            : $SWEEP_GROUP"
echo "CONN_RULE              : $CONN_RULE   (weibull: kernel drawn per-topology; periodic=$CONN_PERIODIC)"
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
    --c_max                "$C_MAX"
    --displ_bias           "$DISPL_BIAS"
    --topology_mode        "$TOPOLOGY_MODE"
    --gj_dist              "$GJ_DIST"
    --gj_max_dist          "$GJ_MAX_DIST"
    --stoa_cutoff          "$STOA_CUTOFF"
    --stoa_sigma           "$STOA_SIGMA"
    --seed_master          "$SEED_MASTER"
    --seed_device          "$SEED_DEVICE"
    --seed_neuron          "$SEED_NEURON"
    --seed_synapse         "$SEED_SYNAPSE"
    --seed_astro           "$SEED_ASTRO"
    --dpi                  "$DPI"
)
[ -n "$SYN_PDIST_CSV" ] && ARGS+=(--syn_pdist_csv "$SYN_PDIST_CSV")
[ "$CONN_PERIODIC" -eq 1 ] && ARGS+=(--conn_periodic)   # weibull-only; no-op under flat
[ -n "$DENSITY" ] && ARGS+=(--density "$DENSITY")        # derive Nn from C_MAX if set

python "$SCRIPT_PATH" "${ARGS[@]}"
EXIT_CODE=$?

echo "[post] rebuilding manifest for ${NODETAG}/task${IDX} ..."
python "$SCRIPT_PATH" --rebuild_manifest_only --out_dir "$OUTPUT_DIR" || true

conda deactivate
sleep 5s
exit $EXIT_CODE
