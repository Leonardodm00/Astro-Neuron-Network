#!/bin/bash
#PBS -S /bin/bash
#PBS -N asn_smoke
#PBS -q intel
#PBS -l select=1:ncpus=4,walltime=01:00:00
#PBS -k eo
##########################################################################
# Smoke test for the CAdEx 34-D pipeline.
#
# Runs TWO tiny end-to-end sweeps on a small network so a regression
# surfaces in minutes, not after a 24 h campaign:
#
#   (A) FULL    : --mode Full --sweep_group all
#                 exercises the cpp_standalone compile of the WHOLE
#                 Neuronal_Network - CAdEx neuron + synapse (incl. the
#                 stochastic-release Binomial_fun C++ path) + astrocyte.
#
#   (B) NEURON  : --mode Neuronal --sweep_group neuron
#                 exercises the seamless neuron-only search: only the 10
#                 CAdEx intrinsic axes vary; synapse/astro frozen at nominal.
#
# PASS criterion per leg:  python exits 0  AND  >= 1 iter_*.npz produced.
#
# Runnable either way:
#     qsub submit_smoke_test.sh      # on the cluster
#     bash submit_smoke_test.sh      # locally (uses nproc + cwd; modules/conda optional)
##########################################################################

# --- USER CONFIG (deliberately tiny) ------------------------------------
SCRIPT_PATH="./HPC_main_sweep.py"
LIB_DIR="."
SYN_PDIST_CSV=""                          # empty -> auto-discover in LIB_DIR
JOBTAG="${PBS_JOBID%%.*}"; [ -z "$JOBTAG" ] && JOBTAG="local"
SMOKE_ROOT="./smoke_${JOBTAG}"

SIMTIME=5                  # seconds - short, just enough to produce spikes
NN=20                      # small network
NA=10
N_TOPOLOGIES=1
N_PARAMS_PER_WORKER=1
N_WORKERS_CAP=2            # cap workers so the smoke test stays light/fast
CONN_PROB=0.3             # FIXED (lo==hi) so a synapse population exists

C_MAX=240
DISPL_BIAS=20
TOPOLOGY_MODE="wallach"
GJ_DIST=200; GJ_MAX_DIST=150; STOA_CUTOFF=70; STOA_SIGMA=200
SEED_MASTER=12345          # fixed -> reproducible smoke test
SEED_DEVICE=50; SEED_NEURON=39; SEED_SYNAPSE=35; SEED_ASTRO=60
DPI=80
# ------------------------------------------------------------------------

cd "${PBS_O_WORKDIR:-$(pwd)}"

# Environment (no-ops locally if modules / conda are absent)
module load python 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate brian_env 2>/dev/null || true
fi

# Workers = min(available CPUs, cap)
if [ -n "$PBS_NCPUS" ]; then NC="$PBS_NCPUS"; else NC=$(nproc 2>/dev/null || echo 1); fi
N_WORKERS=$(( NC < N_WORKERS_CAP ? NC : N_WORKERS_CAP ))
[ "$N_WORKERS" -lt 1 ] && N_WORKERS=1
export OMP_NUM_THREADS=1
export MAKEFLAGS="-j1"

if [ -n "$TMPDIR" ]; then SCRATCH_ROOT="${TMPDIR}/brian2_smoke_${JOBTAG}"
else SCRATCH_ROOT="${SMOKE_ROOT}/_scratch"; fi
mkdir -p "$SMOKE_ROOT" "$SCRATCH_ROOT"

echo "============================================================"
echo "CAdEx smoke test   node=$(hostname)   job=${JOBTAG}"
echo "N_WORKERS=$N_WORKERS  SIMTIME=${SIMTIME}s  Nn=$NN Na=$NA  conn_prob=$CONN_PROB"
echo "output root: $SMOKE_ROOT"
echo "============================================================"

# --- One leg ------------------------------------------------------------
# $1 = human label   $2 = output subdir   $3.. = extra args (mode, sweep_group, ...)
run_leg () {
    local label="$1"; local out="$2"; shift 2
    echo
    echo "------------------------------------------------------------"
    echo "LEG: ${label}"
    echo "     out=${out}   extra: $*"
    echo "------------------------------------------------------------"
    local ARGS=(
        --out_dir              "$out"
        --lib_dir              "$LIB_DIR"
        --scratch_root         "$SCRATCH_ROOT"
        --n_workers            "$N_WORKERS"
        --n_topologies         "$N_TOPOLOGIES"
        --n_params_per_worker  "$N_PARAMS_PER_WORKER"
        --conn_prob_lo         "$CONN_PROB"
        --conn_prob_hi         "$CONN_PROB"
        --simtime              "$SIMTIME"
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
        "$@"
    )
    [ -n "$SYN_PDIST_CSV" ] && ARGS+=(--syn_pdist_csv "$SYN_PDIST_CSV")

    python "$SCRIPT_PATH" "${ARGS[@]}"
    local ec=$?
    local n_npz
    n_npz=$(find "$out" -name 'iter_*.npz' 2>/dev/null | wc -l)
    echo "[${label}] exit=${ec}   iter_*.npz produced=${n_npz}"
    if [ "$ec" -eq 0 ] && [ "$n_npz" -ge 1 ]; then
        echo "[${label}] PASS"; return 0
    fi
    echo "[${label}] FAIL"; return 1
}

FAILS=0

# (A) FULL pipeline: neuron + synapse + astrocyte
run_leg "FULL (neuron+synapse+astro)" "${SMOKE_ROOT}/full" \
        --mode Full --sweep_group all || FAILS=$((FAILS+1))

# (B) NEURON-ONLY: vary only the 10 CAdEx axes; synapses present but frozen
run_leg "NEURON-ONLY (frozen synapses)" "${SMOKE_ROOT}/neuron_only" \
        --mode Neuronal --sweep_group neuron || FAILS=$((FAILS+1))

# (C) OPTIONAL - truly ISOLATED neurons (conn_prob=0, zero synapses). Enabled by
#     the zero-synapse guard in Neuronal_Network. Uncomment to also test it:
# run_leg "NEURON-ONLY (isolated, conn_prob=0)" "${SMOKE_ROOT}/neuron_isolated" \
#         --mode Neuronal --sweep_group neuron --conn_prob_lo 0 --conn_prob_hi 0 \
#         || FAILS=$((FAILS+1))

echo
echo "============================================================"
if [ "$FAILS" -eq 0 ]; then
    echo "SMOKE TEST: ALL LEGS PASSED"
else
    echo "SMOKE TEST: ${FAILS} LEG(S) FAILED - inspect the logs above and ${SMOKE_ROOT}"
fi
echo "============================================================"

if command -v conda >/dev/null 2>&1; then conda deactivate 2>/dev/null || true; fi
exit "$FAILS"
