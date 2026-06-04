#!/bin/bash
#PBS -S /bin/bash
#PBS -N asn_campaign
#PBS -q cfd
#PBS -l select=1:ncpus=192,walltime=24:00:00
#PBS -k eo
#PBS -J 0-19
##########################################################################
# davinci-1 JOB-ARRAY driver for HPC_main_sweep.py  (300k-sim campaign)
#
# Each array sub-job (PBS_ARRAY_INDEX = 0,1,2,...) is an INDEPENDENT sweep:
#   * its own output dir          -> campaign_<TAG>/sweep_task<IDX>
#   * its own master seed         -> SEED_MASTER = SEED_BASE + IDX
#                                    (=> distinct, NON-overlapping draws,
#                                     and fully reproducible if you rerun)
#   * sized to finish CLEANLY in ~TARGET_HOURS, well under the 24 h cap,
#     so no walltime kill wastes a partial topology.
#
# Total sims = N_ARRAY_TASKS * N_TOPOLOGIES * PBS_NCPUS * N_PARAMS_PER_WORKER
#   -> edit  #PBS -J 0-(N_ARRAY_TASKS-1)  and N_TOPOLOGIES below to hit 300k.
#
# SIZING (do the PILOT first, then fill these in):
#   1) run 1 task with N_TOPOLOGIES_PILOT=5 (set -J 0-0), read the manifest:
#        - T_topo_real = median per-topology wall  (incl. straggler & compile)
#   2) N_TOPOLOGIES = floor( TARGET_HOURS*3600 / T_topo_real )
#   3) sims_per_task = N_TOPOLOGIES * PBS_NCPUS * N_PARAMS_PER_WORKER
#   4) N_ARRAY_TASKS = ceil( 300000 / sims_per_task )
#
# WORKED EXAMPLE (mean times, k=1, 192 cores, target 21 h):
#   T_topo ~ 720 s -> N_TOPOLOGIES = 75600/720 = 105
#   sims/task = 105 * 192 * 1 = 20,160 -> N_ARRAY_TASKS = ceil(300000/20160)=15
# WORKED EXAMPLE (straggler-tamed, k=4, target 21 h):
#   T_topo ~ 3700 s -> N_TOPOLOGIES = 75600/3700 = 20
#   sims/task = 20 * 192 * 4 = 15,360 -> N_ARRAY_TASKS = ceil(300000/15360)=20
##########################################################################

# ─── CAMPAIGN IDENTITY ──────────────────────────────────────────────────
CAMPAIGN_TAG="300k_v1"      # bump this for every fresh campaign
SEED_BASE=1000              # SEED_MASTER = SEED_BASE + PBS_ARRAY_INDEX

# ─── SIZING (fill from pilot) ───────────────────────────────────────────
TARGET_HOURS=21             # aim to finish each task in this many h (<24 cap)
N_TOPOLOGIES=105            # per task; sized so task finishes in ~TARGET_HOURS
N_PARAMS_PER_WORKER=1       # 1 = max topology diversity; 3-4 = tame straggler tax
                            # (raise to 3-4 if the pilot shows max/median per-
                            #  topology run-time ratio > ~1.5)

# ─── PATHS ──────────────────────────────────────────────────────────────
SCRIPT_PATH="./HPC_main_sweep.py"
LIB_DIR="."
SYN_PDIST_CSV=""            # empty -> auto-discover in LIB_DIR
CAMPAIGN_ROOT="./campaign_${CAMPAIGN_TAG}"
OUTPUT_DIR="${CAMPAIGN_ROOT}/sweep_task$(printf '%04d' ${PBS_ARRAY_INDEX})"

# ─── Outer-loop random conn_prob ────────────────────────────────────────
CONN_PROB_LO=0.1
CONN_PROB_HI=0.6

# ─── NETWORK / SIM CONFIG  (MUST MATCH YOUR BENCHMARK) ──────────────────
# These reproduce the single_run you timed (115/115, 300 s, [240 µm]² arena).
# If any of these differ from your benchmark, the timings/sizing won't hold.
SIMTIME=300
MODE="Full"
NN=115
NA=115
C_MAX=240                   # <-- set to the value that gave [240 µm]² in your run
DISPL_BIAS=20

# ─── Topology rule ──────────────────────────────────────────────────────
TOPOLOGY_MODE="wallach"
GJ_DIST=200
GJ_MAX_DIST=150
STOA_CUTOFF=70
STOA_SIGMA=200

# ─── Fixed sub-seeds (master RNG re-derives per-topology seeds from these) ─
SEED_DEVICE=50
SEED_NEURON=39
SEED_SYNAPSE=35
SEED_ASTRO=60

DPI=150
# ────────────────────────────────────────────────────────────────────────

cd "$PBS_O_WORKDIR"
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate brian_env

mkdir -p "$OUTPUT_DIR"

# Workers = all CPUs PBS gave us
if [ -n "$PBS_NCPUS" ]; then N_WORKERS="$PBS_NCPUS"; else N_WORKERS=$(nproc); fi

# CRITICAL: single-threaded compile per worker (no make-level oversubscription)
export OMP_NUM_THREADS=1
export MAKEFLAGS="-j1"

# Per-worker cpp_standalone build dirs on node-local fast scratch
if [ -n "$TMPDIR" ]; then
    SCRATCH_ROOT="${TMPDIR}/brian2_${CAMPAIGN_TAG}_${PBS_ARRAY_INDEX}"
else
    SCRATCH_ROOT="${OUTPUT_DIR}/_scratch"
fi
mkdir -p "$SCRATCH_ROOT"

# Distinct, reproducible master seed per array task
SEED_MASTER=$(( SEED_BASE + PBS_ARRAY_INDEX ))

echo "============================================================"
echo "Campaign / task        : ${CAMPAIGN_TAG} / ${PBS_ARRAY_INDEX}"
echo "Node / JobID           : $(hostname) / $PBS_JOBID"
echo "Queue / NCPUS          : ${PBS_O_QUEUE:-?} / $N_WORKERS"
echo "Output dir             : $OUTPUT_DIR"
echo "Scratch root           : $SCRATCH_ROOT"
echo "N_TOPOLOGIES           : $N_TOPOLOGIES"
echo "N_PARAMS_PER_WORKER    : $N_PARAMS_PER_WORKER"
echo "sims this task         : $((N_TOPOLOGIES * N_WORKERS * N_PARAMS_PER_WORKER))"
echo "SEED_MASTER            : $SEED_MASTER"
echo "MODE / Nn / Na         : $MODE / $NN / $NA"
echo "Simtime                : $SIMTIME s   c_max=$C_MAX  displ_bias=$DISPL_BIAS"
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
    --simtime              "$SIMTIME"
    --mode                 "$MODE"
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

python "$SCRIPT_PATH" "${ARGS[@]}"
EXIT_CODE=$?

# Safety-net manifest rebuild (no-op if main() finished cleanly)
echo "[post] rebuilding manifest for task ${PBS_ARRAY_INDEX} ..."
python "$SCRIPT_PATH" --rebuild_manifest_only --out_dir "$OUTPUT_DIR" || true

conda deactivate
sleep 5s
exit $EXIT_CODE
