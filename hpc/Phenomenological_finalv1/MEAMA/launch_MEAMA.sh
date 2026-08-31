#!/bin/bash
# =============================================================================
# launch_MEAMA.sh -- MEA Manifold Analysis campaign launcher.
#
# Spreads a Full-mode tripartite campaign across the compute queues, sized from
# MEAMA/bench_report.py.  Run from anywhere:
#
#     bash MEAMA/launch_MEAMA.sh
#     DRYRUN=1 bash MEAMA/launch_MEAMA.sh        # print every qsub, submit none
#     SEED_BASE_OVERRIDE=<n> bash MEAMA/launch_MEAMA.sh    # replay a launch
#
# This is the SINGLE SOURCE OF TRUTH for N_TOPOLOGIES, K_PARAMS, C_MAX,
# SIMTIME and the two densities; the worker receives all of them via qsub -v
# and has no competing constants to drift from.
# =============================================================================
set -u

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${_HERE}/.." && pwd)"
WORKER="${_HERE}/submit_MEAMA.sh"

for f in "${REPO_DIR}/HPC_main_sweep.py" "$WORKER"; do
    [ -f "$f" ] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done

# ===========================================================================
# SIZING -- fill in from `python MEAMA/bench_report.py <BENCH_ROOT>`
# ===========================================================================
# The launcher REFUSES to submit while SIZING_CONFIRMED=0.  This is deliberate:
# the previous campaign's 80-topologies-per-task figure was derived from an
# 800 s/sim Neuronal-mode assumption that does not describe a Full tripartite
# network, and submitting a million simulations on an unmeasured assumption is
# exactly the failure this gate exists to prevent.
SIZING_CONFIRMED="${SIZING_CONFIRMED:-0}"

C_MAX="${C_MAX:-350}"                # um -- benchmark cell you selected
SIMTIME="${SIMTIME:-200}"            # s  -- benchmark cell you selected
N_TOPOLOGIES_WORKER="${N_TOPOLOGIES_WORKER:-40}"   # from bench_report
K_PARAMS="${K_PARAMS:-1}"            # n_params_per_worker; 1 = max topologies
WALLTIME="${WALLTIME:-200:00:00}"
TARGET="${TARGET:-1000000}"
HEADROOM_PCT="${HEADROOM_PCT:-200}"

# --- population: fixed TOTAL cell density, split by astrocyte fraction -----
# rho_neuron and rho_astro are DERIVED, so the 1:1 split cannot drift.
#   ASTRO_FRAC = 0.5  ->  astrocytes are 50% of all cells (Na = Nn).
# If you instead meant "Na:Nn = 0.5" (half as many astrocytes as neurons at
# the same total), set ASTRO_FRAC=0.3333333 -- that gives Na/Nn = 0.5 with the
# same 1600 cells/mm^2 total, and a 1.78x larger synapse count because the
# neuron count rises.  The two readings are NOT interchangeable.
RHO_TOTAL="${RHO_TOTAL:-1600}"       # cells/mm^2, neurons + astrocytes
ASTRO_FRAC="${ASTRO_FRAC:-0.5}"      # astrocyte share of TOTAL cells

# --- campaign configuration -----------------------------------------------
CAMPAIGN_TAG="${CAMPAIGN_TAG:-meama_rho1600_full_v1}"
MODE="${MODE:-Full}"
SWEEP_GROUP="${SWEEP_GROUP:-tripartite}"
CONN_RULE="${CONN_RULE:-flat}"
CONN_PERIODIC="${CONN_PERIODIC:-0}"
STOA_GATE_LO="${STOA_GATE_LO:-0.2}"
STOA_GATE_HI="${STOA_GATE_HI:-1.0}"

# --- Queues:  "name:ncpus:concurrency" ------------------------------------
QUEUES=(
    "cfd:192:2"
    "intel:48:5"
    "cpu:48:5"
  # "egeos:192:8"
)
# ===========================================================================

# --- guards ---------------------------------------------------------------
if [ "$SIZING_CONFIRMED" != "1" ]; then
    echo "REFUSING TO LAUNCH: SIZING_CONFIRMED=0." >&2
    echo "" >&2
    echo "  Run the sizing benchmark first:" >&2
    echo "      bash MEAMA/launch_bench.sh" >&2
    echo "      python MEAMA/bench_report.py \\" >&2
    echo "          \$(ls -td ${REPO_DIR}/MEAMA/bench_out/bench_* | head -1)" >&2
    echo "" >&2
    echo "  Then set C_MAX, SIMTIME, N_TOPOLOGIES_WORKER and K_PARAMS in this" >&2
    echo "  file from the report and flip SIZING_CONFIRMED=1 (or export it for" >&2
    echo "  a one-off launch)." >&2
    exit 3
fi
if [ "$MODE" != "Full" ]; then
    if [ "$(awk -v a="$STOA_GATE_LO" -v b="$STOA_GATE_HI" \
            'BEGIN{print (a==1.0 && b==1.0) ? 1 : 0}')" != "1" ]; then
        echo "ERROR: gated launch with MODE=${MODE}: the contact gate is only" >&2
        echo "       read in Full mode; sweeping it here draws an inert axis." >&2
        exit 4
    fi
fi
if [ "$(awk -v a="$STOA_GATE_LO" -v b="$STOA_GATE_HI" \
        'BEGIN{print (a>=0 && a<=b && b<=1) ? 1 : 0}')" != "1" ]; then
    echo "ERROR: need 0 <= STOA_GATE_LO <= STOA_GATE_HI <= 1." >&2
    exit 4
fi
if [ "$(awk -v f="$ASTRO_FRAC" 'BEGIN{print (f>=0 && f<1) ? 1 : 0}')" != "1" ]; then
    echo "ERROR: ASTRO_FRAC=${ASTRO_FRAC} must satisfy 0 <= f < 1." >&2
    exit 4
fi

DENSITY=$(awk -v r="$RHO_TOTAL" -v f="$ASTRO_FRAC" 'BEGIN{printf "%.6g", r*(1-f)}')
DENSITY_ASTRO=$(awk -v r="$RHO_TOTAL" -v f="$ASTRO_FRAC" 'BEGIN{printf "%.6g", r*f}')
read -r NN_EXP NA_EXP <<EOF
$(awk -v c="$C_MAX" -v rn="$DENSITY" -v ra="$DENSITY_ASTRO" \
      'BEGIN{a=(c/1000)^2; printf "%d %d", int(rn*a+0.5), int(ra*a+0.5)}')
EOF

total_weight=0
for q in "${QUEUES[@]}"; do
    IFS=':' read -r name nc conc <<< "$q"
    total_weight=$(( total_weight + nc * conc ))
done

# ===========================================================================
# SEED ALLOCATION -- one verified-disjoint range per launch.
# --root and --ledger are made ABSOLUTE against REPO_DIR: this launcher lives
# one directory down, and a relative ledger would silently create a SECOND,
# empty ledger under MEAMA/ that shares no history with the campaigns at the
# repository root -- defeating the entire disjointness check.
# ===========================================================================
SEED_MAX=2147483647
SEED_SPACING=100000
SEED_LEDGER="${REPO_DIR}/artifacts/seed_ledger.tsv"
SEED_ALLOC="${REPO_DIR}/seed_alloc.py"
LAUNCH_SPAN=$(( ${#QUEUES[@]} * SEED_SPACING ))
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"

if [ -n "${SEED_BASE_OVERRIDE:-}" ]; then
    echo "!! SEED_BASE_OVERRIDE=${SEED_BASE_OVERRIDE} -- REPLAY MODE."
fi

if [ -f "$SEED_ALLOC" ] && command -v python3 >/dev/null 2>&1; then
    LAUNCH_BASE=$(python3 "$SEED_ALLOC" \
        --span "$LAUNCH_SPAN" --seed-max "$SEED_MAX" \
        --root "$REPO_DIR" --ledger "$SEED_LEDGER" \
        --launch-id "$LAUNCH_ID" --tag "$CAMPAIGN_TAG" \
        ${DRYRUN:+--note "DRYRUN-not-submitted"} \
        ${SEED_BASE_OVERRIDE:+--override "$SEED_BASE_OVERRIDE"})
    if [ -z "$LAUNCH_BASE" ]; then
        echo "ERROR: seed_alloc.py failed to allocate a seed range." >&2
        echo "       Inspect with: python3 $SEED_ALLOC --root $REPO_DIR --report" >&2
        exit 5
    fi
else
    echo "WARNING: seed_alloc.py unavailable -- raw OS entropy, NO disjointness" >&2
    echo "         check against previously recorded seeds." >&2
    LAUNCH_BASE=$(( (16#$(head -c6 /dev/urandom | od -An -tx1 | tr -d ' \n') \
                     % (SEED_MAX - LAUNCH_SPAN - 1)) + 1 ))
    [ -n "${SEED_BASE_OVERRIDE:-}" ] && LAUNCH_BASE="$SEED_BASE_OVERRIDE"
fi

echo "============================================================"
echo "MEAMA campaign : ${CAMPAIGN_TAG}"
echo "REPO_DIR       : ${REPO_DIR}"
echo "Launch ID      : ${LAUNCH_ID}"
echo "Seed range     : [${LAUNCH_BASE}, $((LAUNCH_BASE + LAUNCH_SPAN)))"
echo "Ledger         : ${SEED_LEDGER}"
echo "------------------------------------------------------------"
echo "mode/group/rule: ${MODE} / ${SWEEP_GROUP} / ${CONN_RULE} (periodic=${CONN_PERIODIC})"
echo "stoa gate      : p ~ U[${STOA_GATE_LO}, ${STOA_GATE_HI}] per topology"
echo "well / simtime : ${C_MAX} um / ${SIMTIME} s"
echo "density        : rho_total=${RHO_TOTAL}/mm^2, astro frac ${ASTRO_FRAC}"
echo "                 -> rho_n=${DENSITY}, rho_a=${DENSITY_ASTRO}"
echo "                 -> Nn=${NN_EXP}, Na=${NA_EXP}, total=$((NN_EXP + NA_EXP)) cells"
echo "sizing         : N_TOPOLOGIES=${N_TOPOLOGIES_WORKER}  k=${K_PARAMS}"
echo "target         : ${TARGET} sims  (headroom ${HEADROOM_PCT}%)"
echo "============================================================"
echo

QDEL_IDS=""
GRAND_MAX=0
_q_index=0
for q in "${QUEUES[@]}"; do
    IFS=':' read -r name nc conc <<< "$q"
    sb=$(( LAUNCH_BASE + _q_index * SEED_SPACING ))
    _q_index=$(( _q_index + 1 ))

    weight=$(( nc * conc ))
    sims_q=$(( TARGET * weight / total_weight ))
    yield_q=$(( N_TOPOLOGIES_WORKER * nc * K_PARAMS ))
    ntasks=$(( (sims_q * HEADROOM_PCT / 100 + yield_q - 1) / yield_q ))
    [ "$ntasks" -lt 1 ] && ntasks=1
    hi=$(( sb + ntasks - 1 ))
    GRAND_MAX=$(( GRAND_MAX + ntasks * yield_q ))

    printf "  %-6s : %4d tasks  (<=%2d at once, %3dc)  seeds %d-%d  ~%d target sims\n" \
        "$name" "$ntasks" "$conc" "$nc" "$sb" "$hi" "$sims_q"

    VARS="REPO_DIR=${REPO_DIR},NODETAG=${name},SEED_BASE=${sb}"
    VARS="${VARS},LAUNCH_ID=${LAUNCH_ID},CAMPAIGN_TAG=${CAMPAIGN_TAG}"
    VARS="${VARS},N_TOPOLOGIES=${N_TOPOLOGIES_WORKER},N_PARAMS_PER_WORKER=${K_PARAMS}"
    VARS="${VARS},C_MAX=${C_MAX},SIMTIME=${SIMTIME}"
    VARS="${VARS},DENSITY=${DENSITY},DENSITY_ASTRO=${DENSITY_ASTRO}"
    VARS="${VARS},MODE=${MODE},SWEEP_GROUP=${SWEEP_GROUP}"
    VARS="${VARS},CONN_RULE=${CONN_RULE},CONN_PERIODIC=${CONN_PERIODIC}"
    VARS="${VARS},STOA_GATE_LO=${STOA_GATE_LO},STOA_GATE_HI=${STOA_GATE_HI}"

    if [ "${DRYRUN:-0}" = "1" ]; then
        if [ "$ntasks" -gt 1 ]; then
            echo "    [dry run] qsub -q ${name} -l select=1:ncpus=${nc},walltime=${WALLTIME} \\"
            echo "              -J 0-$((ntasks-1))%${conc} -v \"${VARS}\" ${WORKER}"
        else
            echo "    [dry run] qsub -q ${name} -l select=1:ncpus=${nc},walltime=${WALLTIME} \\"
            echo "              -v \"${VARS}\" ${WORKER}"
        fi
        continue
    fi

    if [ "$ntasks" -gt 1 ]; then
        jid=$(qsub -q "$name" \
            -l "select=1:ncpus=${nc},walltime=${WALLTIME}" \
            -J "0-$((ntasks-1))%${conc}" \
            -v "$VARS" "$WORKER")
    else
        jid=$(qsub -q "$name" \
            -l "select=1:ncpus=${nc},walltime=${WALLTIME}" \
            -v "$VARS" "$WORKER")
    fi
    echo "            -> ${jid}"
    QDEL_IDS="${QDEL_IDS} ${jid}"
done

echo
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "[dry run] nothing submitted. Re-run without DRYRUN=1 to launch."
    exit 0
fi
echo "maximum yield if every task runs to completion: ${GRAND_MAX} sims"
echo "monitor : python ${REPO_DIR}/status_sweep.py ${REPO_DIR}/campaign_${CAMPAIGN_TAG} --target ${TARGET}"
echo "stop all: qdel${QDEL_IDS}"
