#!/bin/bash
# =============================================================================
# launch_LCDUP.sh -- LindaColloDUP campaign launcher (neurons only, all
# excitatory: MODE=Neuronal, no astrocytes).
#
# Derived from MEAMA/launch_MEAMA.sh (2026-10-29). Run from anywhere:
#
#     SIZING_CONFIRMED=1 REFRACTORY_CLAMP=yes DRYRUN=1 bash launch_LCDUP.sh   # print every qsub
#     SIZING_CONFIRMED=1 REFRACTORY_CLAMP=yes bash launch_LCDUP.sh            # launch
#     SEED_BASE_OVERRIDE=<n> ...                         # replay a launch's seeds
#
# REFRACTORY_CLAMP has no default: say yes if ASD_fun_BD_cpp.py was patched
# with apply_refractory_clamp.py (V held at VR during t_ref), no if it was
# not. The launcher reads the file and refuses when the answer disagrees.
#
# This file is the SINGLE SOURCE OF TRUTH for every campaign setting: the
# worker (submit_LCDUP.sh) receives all of them with qsub -v and has no
# competing constants. Unlike MEAMA, CONN_PROB_LO/HI and the freeze list are
# forwarded too.
#
# The folder is self-contained: the pipeline (HPC_main_sweep.py etc., the
# 37-D copy from the cluster's MEAMA/ folder, patched with
# apply_freeze_patch.py), the seed ledger and the campaign output all live in
# the folder that holds this script.
# =============================================================================
set -u

DRYRUN="${DRYRUN:-0}"
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$_HERE"
WORKER="${_HERE}/submit_LCDUP.sh"

for f in "${REPO_DIR}/HPC_main_sweep.py" "${REPO_DIR}/HPC_single_run.py" \
         "${REPO_DIR}/ASD_fun_BD_cpp.py" "${REPO_DIR}/synapse_pdist.csv" "$WORKER"; do
    [ -f "$f" ] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done
if ! grep -q "def _apply_freeze(" "${REPO_DIR}/HPC_main_sweep.py"; then
    echo "ERROR: ${REPO_DIR}/HPC_main_sweep.py has no --freeze support." >&2
    echo "       Run  python3 apply_freeze_patch.py  in this folder first." >&2
    exit 2
fi

# ===========================================================================
# SIZING -- per-simulation cost differs from MEAMA (Neuronal, Nn=159); see
# claude/LindaColloDUP_Campaign_Operations.md section 5 for the estimate
# behind 32 (about 1,600-2,000 s per topology, from MEAMA campaign 1's
# measured 1,350 s per topology scaled by a sandbox cost ratio). Re-size from
# the first tasks' measured time per topology (APPLY_LCDUP.txt step 10).
# ===========================================================================
SIZING_CONFIRMED="${SIZING_CONFIRMED:-0}"

C_MAX="${C_MAX:-350}"                # um, culture side (as MEAMA)
SIMTIME="${SIMTIME:-180}"            # s, biological time per simulation
N_TOPOLOGIES_WORKER="${N_TOPOLOGIES_WORKER:-32}"   # topologies per task (~14-18 h, estimate)
K_PARAMS="${K_PARAMS:-1}"            # parameter vectors per worker per topology (as MEAMA)
WALLTIME="${WALLTIME:-24:00:00}"     # as MEAMA
TARGET="${TARGET:-1000000}"          # as MEAMA
HEADROOM_PCT="${HEADROOM_PCT:-200}"  # as MEAMA

# --- population: neurons only -------------------------------------------------
# rho_neuron = RHO_TOTAL * (1 - ASTRO_FRAC); ASTRO_FRAC must stay 0 here.
RHO_TOTAL="${RHO_TOTAL:-1300}"       # cells/mm^2 -> 1300 neurons/mm^2
ASTRO_FRAC="${ASTRO_FRAC:-0}"

# --- campaign configuration ---------------------------------------------------
CAMPAIGN_TAG="${CAMPAIGN_TAG:-lcdup15_rho1300_neur_n01}"
CONDA_ENV="${CONDA_ENV:-brian_final}"
MODE="${MODE:-Neuronal}"
SWEEP_GROUP="${SWEEP_GROUP:-neuron_synapse}"
# Axes taken out of SWEEP_GROUP: NAME = held at its nominal value,
# NAME=VALUE = held at VALUE (natural units). Leaves 13 free axes.
FREEZE="${FREEZE:-Cm,I_inj,VA=-50,DeltaA,VR=-55,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G}"
CONN_RULE="${CONN_RULE:-flat}"
CONN_PERIODIC="${CONN_PERIODIC:-0}"
CONN_PROB_LO="${CONN_PROB_LO:-0.1}"  # conn_prob ~ U[LO, HI], one draw per topology
CONN_PROB_HI="${CONN_PROB_HI:-0.6}"
STOA_GATE_LO="${STOA_GATE_LO:-1.0}"  # astrocyte contact gate: off (no astrocytes)
STOA_GATE_HI="${STOA_GATE_HI:-1.0}"

# --- Queues: "name:ncpus:concurrency" (as MEAMA on davinci) --------------------
QUEUES=(
    "intel:48:2"
    "cpu:48:2"
)
# ===========================================================================

# --- guards -----------------------------------------------------------------
if [ "$SIZING_CONFIRMED" != "1" ]; then
    echo "REFUSING TO LAUNCH: SIZING_CONFIRMED=0." >&2
    echo "  N_TOPOLOGIES_WORKER=${N_TOPOLOGIES_WORKER} is an estimate (see the" >&2
    echo "  operations doc). Re-run with SIZING_CONFIRMED=1 once you accept it." >&2
    exit 3
fi
# Neuron-model variant: is V held at VR during t_ref? (see apply_refractory_clamp.py)
REFRACTORY_CLAMP="${REFRACTORY_CLAMP:-}"
N_DV=$(grep -Ec '^[[:space:]]*dV/dt[[:space:]]*=' "${REPO_DIR}/ASD_fun_BD_cpp.py")
if [ "$N_DV" != "1" ]; then
    echo "ERROR: ${N_DV} dV/dt lines in ASD_fun_BD_cpp.py (expected 1): cannot tell the model variant." >&2
    exit 4
fi
DV_LINE="$(grep -E '^[[:space:]]*dV/dt[[:space:]]*=' "${REPO_DIR}/ASD_fun_BD_cpp.py" | sed 's/^[[:space:]]*//')"
if printf '%s' "$DV_LINE" | grep -Eq ':[[:space:]]*volt[[:space:]]*\(unless refractory\)[[:space:]]*$'; then
    CLAMP_STATE="yes"
else
    CLAMP_STATE="no"
fi
if [ "$REFRACTORY_CLAMP" != "yes" ] && [ "$REFRACTORY_CLAMP" != "no" ]; then
    echo "REFUSING TO LAUNCH: REFRACTORY_CLAMP must be set to yes or no." >&2
    echo "  ASD_fun_BD_cpp.py currently: V held at VR during t_ref = ${CLAMP_STATE}" >&2
    echo "  (without the clamp, neurons driven above VT during t_ref go to NaN and fall" >&2
    echo "  silent; see the operations doc). Patch with apply_refractory_clamp.py, or not," >&2
    echo "  then re-run with REFRACTORY_CLAMP=yes|no matching the file." >&2
    exit 3
fi
if [ "$REFRACTORY_CLAMP" != "$CLAMP_STATE" ]; then
    echo "ERROR: REFRACTORY_CLAMP=${REFRACTORY_CLAMP}, but in ASD_fun_BD_cpp.py V held during t_ref = ${CLAMP_STATE}:" >&2
    echo "       ${DV_LINE}" >&2
    exit 4
fi
if [ "$MODE" != "Neuronal" ]; then
    echo "ERROR: this launcher is for MODE=Neuronal (got ${MODE})." >&2
    exit 4
fi
if [ "$(awk -v f="$ASTRO_FRAC" 'BEGIN{print (f==0) ? 1 : 0}')" != "1" ]; then
    echo "ERROR: ASTRO_FRAC=${ASTRO_FRAC}: no astrocytes are built in Neuronal mode; keep 0." >&2
    exit 4
fi
if [ "$(awk -v a="$STOA_GATE_LO" -v b="$STOA_GATE_HI" \
        'BEGIN{print (a==1.0 && b==1.0) ? 1 : 0}')" != "1" ]; then
    echo "ERROR: STOA_GATE_LO/HI must both be 1.0 in Neuronal mode (the gate is never read)." >&2
    exit 4
fi
if [ "$(awk -v a="$CONN_PROB_LO" -v b="$CONN_PROB_HI" \
        'BEGIN{print (a>=0 && a<=b && b<=1) ? 1 : 0}')" != "1" ]; then
    echo "ERROR: need 0 <= CONN_PROB_LO <= CONN_PROB_HI <= 1." >&2
    exit 4
fi
# FREEZE travels through qsub -v, whose list separator is ','. It is sent with
# ',' -> ':' and '=' -> '@' and decoded by the worker; so the list may contain
# only names, '=', '-', '.', digits, letters and '_' (write 1e3, not 1e+3).
if ! printf '%s' "$FREEZE" | grep -Eq '^[A-Za-z0-9_]+(=-?[0-9.]+([eE]-?[0-9]+)?)?(,[A-Za-z0-9_]+(=-?[0-9.]+([eE]-?[0-9]+)?)?)*$'; then
    echo "ERROR: FREEZE='${FREEZE}' is not a list of NAME or NAME=NUMBER items." >&2
    exit 4
fi
FREEZE_ENC="$(printf '%s' "$FREEZE" | tr ',=' ':@')"
if [ "$DRYRUN" != "1" ] && [ -e "${REPO_DIR}/campaign_${CAMPAIGN_TAG}" ]; then
    echo "ERROR: ${REPO_DIR}/campaign_${CAMPAIGN_TAG} already exists; choose a new CAMPAIGN_TAG." >&2
    exit 4
fi

DENSITY=$(awk -v r="$RHO_TOTAL" -v f="$ASTRO_FRAC" 'BEGIN{printf "%.6g", r*(1-f)}')
DENSITY_ASTRO=0
NN_EXP=$(awk -v c="$C_MAX" -v rn="$DENSITY" 'BEGIN{a=(c/1000)^2; printf "%d", int(rn*a+0.5)}')

total_weight=0
for q in "${QUEUES[@]}"; do
    IFS=':' read -r name nc conc <<< "$q"
    total_weight=$(( total_weight + nc * conc ))
done

# ===========================================================================
# SEED ALLOCATION -- one verified-disjoint range per launch, against this
# folder's campaigns and its own ledger (artifacts/seed_ledger.tsv).
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
DRYNOTE=""
[ "$DRYRUN" = "1" ] && DRYNOTE="DRYRUN-not-submitted"
if [ -f "$SEED_ALLOC" ] && command -v python3 >/dev/null 2>&1; then
    LAUNCH_BASE=$(python3 "$SEED_ALLOC" \
        --span "$LAUNCH_SPAN" --seed-max "$SEED_MAX" \
        --root "$REPO_DIR" --ledger "$SEED_LEDGER" \
        --launch-id "$LAUNCH_ID" --tag "$CAMPAIGN_TAG" \
        ${DRYNOTE:+--note "$DRYNOTE"} \
        ${SEED_BASE_OVERRIDE:+--override "$SEED_BASE_OVERRIDE"})
    if [ -z "$LAUNCH_BASE" ]; then
        echo "ERROR: seed_alloc.py failed to allocate a seed range." >&2
        echo "       Inspect with: python3 $SEED_ALLOC --root $REPO_DIR --ledger $SEED_LEDGER --report" >&2
        exit 5
    fi
else
    echo "ERROR: seed_alloc.py or python3 unavailable -- refusing to launch without" >&2
    echo "       a disjointness check (copy seed_alloc.py into ${REPO_DIR})." >&2
    exit 5
fi

echo "============================================================"
echo "LindaColloDUP campaign : ${CAMPAIGN_TAG}"
echo "REPO_DIR       : ${REPO_DIR}"
echo "Launch ID      : ${LAUNCH_ID}"
echo "Seed range     : [${LAUNCH_BASE}, $((LAUNCH_BASE + LAUNCH_SPAN)))"
echo "Ledger         : ${SEED_LEDGER}"
echo "------------------------------------------------------------"
echo "mode/group/rule: ${MODE} / ${SWEEP_GROUP} / ${CONN_RULE} (periodic=${CONN_PERIODIC})"
echo "freeze         : ${FREEZE}"
echo "V clamp (t_ref): ${CLAMP_STATE}   [${DV_LINE}]"
echo "conn_prob      : U[${CONN_PROB_LO}, ${CONN_PROB_HI}] per topology"
echo "well / simtime : ${C_MAX} um / ${SIMTIME} s"
echo "density        : ${DENSITY} neurons/mm^2, no astrocytes -> Nn=${NN_EXP}"
echo "sizing         : N_TOPOLOGIES=${N_TOPOLOGIES_WORKER}  k=${K_PARAMS}  walltime ${WALLTIME}"
echo "conda env      : ${CONDA_ENV}"
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

    VARS="REPO_DIR=${REPO_DIR},NODETAG=${name},SEED_BASE=${sb},CONDA_ENV=${CONDA_ENV}"
    VARS="${VARS},LAUNCH_ID=${LAUNCH_ID},CAMPAIGN_TAG=${CAMPAIGN_TAG}"
    VARS="${VARS},N_TOPOLOGIES=${N_TOPOLOGIES_WORKER},N_PARAMS_PER_WORKER=${K_PARAMS}"
    VARS="${VARS},C_MAX=${C_MAX},SIMTIME=${SIMTIME}"
    VARS="${VARS},DENSITY=${DENSITY},DENSITY_ASTRO=${DENSITY_ASTRO}"
    VARS="${VARS},MODE=${MODE},SWEEP_GROUP=${SWEEP_GROUP},FREEZE_ENC=${FREEZE_ENC}"
    VARS="${VARS},CONN_RULE=${CONN_RULE},CONN_PERIODIC=${CONN_PERIODIC}"
    VARS="${VARS},CONN_PROB_LO=${CONN_PROB_LO},CONN_PROB_HI=${CONN_PROB_HI}"
    VARS="${VARS},STOA_GATE_LO=${STOA_GATE_LO},STOA_GATE_HI=${STOA_GATE_HI}"

    if [ "$DRYRUN" = "1" ]; then
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
    QDEL_IDS="${QDEL_IDS} \"${jid}\""
done

echo
if [ "$DRYRUN" = "1" ]; then
    echo "[dry run] nothing submitted. Re-run without DRYRUN=1 to launch."
    exit 0
fi

# Provenance: which model variant and which file versions this launch used.
PROV="${REPO_DIR}/campaign_${CAMPAIGN_TAG}/PROVENANCE_${LAUNCH_ID}.txt"
mkdir -p "${REPO_DIR}/campaign_${CAMPAIGN_TAG}"
{
    echo "LindaColloDUP campaign provenance"
    echo "campaign_tag   : ${CAMPAIGN_TAG}"
    echo "launch_id      : ${LAUNCH_ID}"
    echo "repo_dir       : ${REPO_DIR}"
    echo "seed_range     : [${LAUNCH_BASE}, $((LAUNCH_BASE + LAUNCH_SPAN)))"
    echo "mode / group   : ${MODE} / ${SWEEP_GROUP}"
    echo "freeze         : ${FREEZE}"
    echo "v_clamp_tref   : ${CLAMP_STATE}"
    echo "dV/dt line     : ${DV_LINE}"
    echo "density / well : ${DENSITY} neurons/mm^2, Nn=${NN_EXP} / ${C_MAX} um"
    echo "simtime        : ${SIMTIME} s"
    echo "conn_prob      : U[${CONN_PROB_LO}, ${CONN_PROB_HI}] (${CONN_RULE})"
    echo "sizing         : N_TOPOLOGIES=${N_TOPOLOGIES_WORKER} k=${K_PARAMS} walltime ${WALLTIME}"
    echo "jobs           :${QDEL_IDS}"
    echo "sha256         :"
    ( cd "$REPO_DIR" && sha256sum HPC_main_sweep.py HPC_single_run.py ASD_fun_BD_cpp.py \
        synapse_pdist.csv launch_LCDUP.sh submit_LCDUP.sh )
} > "$PROV" 2>&1 || echo "WARNING: could not write ${PROV}" >&2
echo "provenance: ${PROV}"
echo "maximum yield if every task runs to completion: ${GRAND_MAX} sims"
echo "monitor : cd ${REPO_DIR} && python3 aggregate_sweep.py campaign_${CAMPAIGN_TAG} --target ${TARGET} | tail -3"
echo "stop all: qdel${QDEL_IDS}"
