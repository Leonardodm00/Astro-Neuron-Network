#!/bin/bash
#PBS -S /bin/bash
#PBS -N meama_bench
#PBS -k eo
##########################################################################
# MEAMA sizing benchmark -- measures what a Full-mode tripartite simulation
# actually costs at MEA well scale, so N_TOPOLOGIES / K_PARAMS / walltime for
# the real campaign are chosen from data instead of from the stale 800 s/sim
# Neuronal-mode assumption.
#
# DO NOT qsub this directly -- submit it via launch_bench.sh, which resolves
# REPO_DIR (this script's parent directory) on the login node and passes it
# with qsub -v.  PBS spools the job script to a scratch path, so ${BASH_SOURCE}
# inside a running PBS job does NOT point at the repository copy; REPO_DIR
# cannot be recovered here and must be injected.
#
# GRID (6 cells): C_MAX in {300, 350, 400} um  x  SIMTIME in {200, 300} s
#   at a FIXED TOTAL cell density RHO_TOTAL = 1600 cells/mm^2, split by
#   ASTRO_FRAC.  Neuron and astrocyte densities are derived, never typed in
#   twice:  rho_neuron = RHO_TOTAL*(1-ASTRO_FRAC),
#           rho_astro  = RHO_TOTAL*ASTRO_FRAC.
#
# CONTROLLED COMPARISON.  Every config runs with the SAME --seed_master, so
# the master RNG hands out the same conn_prob draws, the same contact-gate
# draws and the same parameter vectors in all six cells.  Differences in
# measured time are therefore attributable to C_MAX and SIMTIME only, not to
# having drawn a denser network in one cell than another.  This is the whole
# reason the seed is hardcoded rather than allocated.
#
# The benchmark output is a TIMING ARTIFACT, not campaign data: it is written
# to MEAMA/bench_out/ (deliberately NOT named campaign_*), so no aggregation
# glob, no MEA pipeline and no seed_alloc scan will ever pick it up, and its
# simulations can never leak into an SBI training set.
##########################################################################

set -u

REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}"
if [ ! -f "${REPO_DIR}/HPC_main_sweep.py" ]; then
    echo "ERROR: REPO_DIR=${REPO_DIR} does not contain HPC_main_sweep.py." >&2
    echo "       Submit via launch_bench.sh, which sets REPO_DIR." >&2
    exit 2
fi
cd "$REPO_DIR" || exit 2

# --- benchmark grid -------------------------------------------------------
C_MAX_LIST="${C_MAX_LIST:-300 350 400}"      # um, culture side
SIMTIME_LIST="${SIMTIME_LIST:-200 300}"      # s, biological simulated time
RHO_TOTAL="${RHO_TOTAL:-1600}"               # cells/mm^2, TOTAL (neurons+astro)
ASTRO_FRAC="${ASTRO_FRAC:-0.5}"              # astrocyte share of TOTAL cells

# --- benchmark sizing -----------------------------------------------------
# 2 topologies x N_BENCH_WORKERS x K_BENCH sims per config.  Two topologies
# give two DIFFERENT conn_prob draws (identical across configs, see above), so
# the report can separate "cost of a bigger well" from "cost of a denser
# graph".  K_BENCH > 1 is required to separate compile cost from run cost.
N_TOPO_BENCH="${N_TOPO_BENCH:-2}"
K_BENCH="${K_BENCH:-3}"
SEED_MASTER_BENCH="${SEED_MASTER_BENCH:-20260831}"

# --- simulator configuration: MUST MATCH THE REAL CAMPAIGN ----------------
# If any of these drifts from submit_MEAMA.sh the measured times do not
# transfer and the whole benchmark is void.
MODE="Full"
SWEEP_GROUP="tripartite"
CONN_RULE="flat"
CONN_PROB_LO=0.1
CONN_PROB_HI=0.6
STOA_GATE_LO=0.2
STOA_GATE_HI=1.0
SEED_STOA_GATE=71
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
DPI=110
# --------------------------------------------------------------------------

BENCH_ROOT="${REPO_DIR}/MEAMA/bench_out/bench_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BENCH_ROOT"

# Environment. MEAMA_NO_ENV=1 skips module/conda activation -- for verifying
# this script inside an ALREADY-ACTIVE environment (an interactive login-node
# session, or a local check). It is never set by the launchers, so a scheduled
# job always takes the normal path.
if [ -z "${MEAMA_NO_ENV:-}" ]; then
    module load python
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate brian_env
else
    echo "[env] MEAMA_NO_ENV=1 -- using the ambient python, NOT brian_env."
fi

if [ -n "${PBS_NCPUS:-}" ]; then NCPUS="$PBS_NCPUS"; else NCPUS=$(nproc); fi
N_BENCH_WORKERS="${N_BENCH_WORKERS:-12}"
if [ "$N_BENCH_WORKERS" -gt "$NCPUS" ]; then N_BENCH_WORKERS="$NCPUS"; fi

export OMP_NUM_THREADS=1
export MAKEFLAGS="-j1"

if [ -n "${TMPDIR:-}" ]; then
    SCRATCH_ROOT="${TMPDIR}/brian2_meama_bench_$$"
else
    SCRATCH_ROOT="${BENCH_ROOT}/_scratch"
fi
mkdir -p "$SCRATCH_ROOT"

# Derived densities (single source of truth: RHO_TOTAL and ASTRO_FRAC).
DENSITY=$(awk -v r="$RHO_TOTAL" -v f="$ASTRO_FRAC" 'BEGIN{printf "%.6g", r*(1-f)}')
DENSITY_ASTRO=$(awk -v r="$RHO_TOTAL" -v f="$ASTRO_FRAC" 'BEGIN{printf "%.6g", r*f}')

echo "============================================================"
echo "MEAMA sizing benchmark"
echo "Node / JobID / Queue : $(hostname) / ${PBS_JOBID:-<interactive>} / ${PBS_O_QUEUE:-?}"
echo "REPO_DIR             : $REPO_DIR"
echo "BENCH_ROOT           : $BENCH_ROOT"
echo "ncpus / bench workers: $NCPUS / $N_BENCH_WORKERS"
echo "rho_total            : ${RHO_TOTAL} cells/mm^2  (astro fraction ${ASTRO_FRAC})"
echo "  -> rho_neuron      : ${DENSITY} /mm^2"
echo "  -> rho_astro       : ${DENSITY_ASTRO} /mm^2"
echo "grid                 : C_MAX {${C_MAX_LIST}} x SIMTIME {${SIMTIME_LIST}}"
echo "per config           : ${N_TOPO_BENCH} topologies x ${N_BENCH_WORKERS} workers x ${K_BENCH} params"
echo "seed_master (fixed)  : ${SEED_MASTER_BENCH}   [same draws in every cell]"
echo "mode / group / rule  : ${MODE} / ${SWEEP_GROUP} / ${CONN_RULE}"
echo "stoa gate            : U[${STOA_GATE_LO}, ${STOA_GATE_HI}]"
echo "============================================================"

# Provenance sidecar: the report refuses to run without it.
cat > "${BENCH_ROOT}/bench_config.json" <<JSON
{
  "rho_total_per_mm2": ${RHO_TOTAL},
  "astro_fraction": ${ASTRO_FRAC},
  "density_neuron_per_mm2": ${DENSITY},
  "density_astro_per_mm2": ${DENSITY_ASTRO},
  "c_max_list": "${C_MAX_LIST}",
  "simtime_list": "${SIMTIME_LIST}",
  "n_topologies_bench": ${N_TOPO_BENCH},
  "n_bench_workers": ${N_BENCH_WORKERS},
  "k_bench": ${K_BENCH},
  "seed_master": ${SEED_MASTER_BENCH},
  "ncpus": ${NCPUS},
  "mode": "${MODE}",
  "sweep_group": "${SWEEP_GROUP}",
  "conn_rule": "${CONN_RULE}",
  "conn_prob_lo": ${CONN_PROB_LO},
  "conn_prob_hi": ${CONN_PROB_HI},
  "stoa_gate_lo": ${STOA_GATE_LO},
  "stoa_gate_hi": ${STOA_GATE_HI},
  "host": "$(hostname)",
  "job_id": "${PBS_JOBID:-interactive}"
}
JSON

RC_ALL=0
for CMAX in $C_MAX_LIST; do
    for ST in $SIMTIME_LIST; do
        CELL="c${CMAX}_t${ST}"
        OUT="${BENCH_ROOT}/${CELL}"
        mkdir -p "$OUT"
        echo
        echo "------------------------------------------------------------"
        echo "[bench] ${CELL}: c_max=${CMAX} um, simtime=${ST} s"
        echo "------------------------------------------------------------"
        T0=$(date +%s)
        python "${REPO_DIR}/HPC_main_sweep.py" \
            --out_dir              "$OUT" \
            --lib_dir              "$REPO_DIR" \
            --scratch_root         "${SCRATCH_ROOT}/${CELL}" \
            --n_workers            "$N_BENCH_WORKERS" \
            --n_topologies         "$N_TOPO_BENCH" \
            --n_params_per_worker  "$K_BENCH" \
            --seed_master          "$SEED_MASTER_BENCH" \
            --conn_prob_lo         "$CONN_PROB_LO" \
            --conn_prob_hi         "$CONN_PROB_HI" \
            --conn_rule            "$CONN_RULE" \
            --simtime              "$ST" \
            --mode                 "$MODE" \
            --sweep_group          "$SWEEP_GROUP" \
            --density              "$DENSITY" \
            --density_astro        "$DENSITY_ASTRO" \
            --c_max                "$CMAX" \
            --displ_bias           "$DISPL_BIAS" \
            --topology_mode        "$TOPOLOGY_MODE" \
            --gj_dist              "$GJ_DIST" \
            --gj_max_dist          "$GJ_MAX_DIST" \
            --stoa_cutoff          "$STOA_CUTOFF" \
            --stoa_gate_lo         "$STOA_GATE_LO" \
            --stoa_gate_hi         "$STOA_GATE_HI" \
            --seed_stoa_gate       "$SEED_STOA_GATE" \
            --stoa_sigma           "$STOA_SIGMA" \
            --seed_device          "$SEED_DEVICE" \
            --seed_neuron          "$SEED_NEURON" \
            --seed_synapse         "$SEED_SYNAPSE" \
            --seed_astro           "$SEED_ASTRO" \
            --dpi                  "$DPI"
        RC=$?
        T1=$(date +%s)
        echo "[bench] ${CELL} finished rc=${RC} in $((T1 - T0)) s"
        echo "{\"cell\": \"${CELL}\", \"c_max\": ${CMAX}, \"simtime\": ${ST}, \"rc\": ${RC}, \"wall_s\": $((T1 - T0))}" \
            >> "${BENCH_ROOT}/bench_cells.jsonl"
        [ "$RC" -ne 0 ] && RC_ALL=$RC
    done
done

echo
echo "============================================================"
echo "[bench] all cells done. Root: ${BENCH_ROOT}"
echo "Now run, on the login node:"
echo "    python MEAMA/bench_report.py ${BENCH_ROOT}"
echo "============================================================"

[ -z "${MEAMA_NO_ENV:-}" ] && conda deactivate
exit $RC_ALL
