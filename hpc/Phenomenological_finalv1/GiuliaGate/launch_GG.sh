#!/bin/bash
##########################################################################
# launch_GG.sh -- GiuliaGate campaign launcher (login node).
#
# Giulia's design on the 37-D default pipeline (decided 2026-10-31): MODE
# Full, neurons fixed at the default cell, 15 free axes (prior_config.tsv),
# the astrocyte contact gate drawn per network, the refractory hold.
# Derived from LindaColloDUP/launch_LCDUP.sh. Run from anywhere:
#
#     SIZING_CONFIRMED=1 REFRACTORY_CLAMP=yes DRYRUN=1 bash launch_GG.sh   # print every qsub
#     SIZING_CONFIRMED=1 REFRACTORY_CLAMP=yes bash launch_GG.sh            # launch
#
# REFRACTORY_CLAMP has no default: say yes if ASD_fun_BD_cpp.py holds V at VR
# during t_ref (the root model does). The launcher refuses a mismatch.
#
# Two sources of truth: this file for every campaign setting (the worker
# submit_GG.sh receives them with qsub -v and has no competing constants),
# and prior_config.tsv for the prior: the freeze list (fixed axes, NAME=VALUE),
# the boxes of the free axes (patched into HPC_main_sweep.py by
# apply_prior_patch.py) and the conn_prob and contact-gate boxes. The launcher
# reads the TSV through apply_prior_patch.py and refuses a copy of
# HPC_main_sweep.py whose boxes differ from it.
#
# The folder is self-contained: the pipeline (copies of the clone's root
# files, APPLY_GG.txt step 0, patched in place), the seed ledger and the
# campaign output all live in the folder that holds this script;
# .gitignore keeps the copies and the output out of git (never run
# "git clean -x" there). seed_alloc.py comes with git: its seed guard also
# covers every launch folder under SEED_SCAN_TREE (default the cluster's ANN).
# Spec: specs/SPEC.md Block 9.
# =============================================================================
set -u

DRYRUN="${DRYRUN:-0}"
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$_HERE"
WORKER="${_HERE}/submit_GG.sh"
PRIOR="${REPO_DIR}/apply_prior_patch.py"
TSV="${REPO_DIR}/prior_config.tsv"

for f in "${REPO_DIR}/HPC_main_sweep.py" "${REPO_DIR}/HPC_single_run.py" \
         "${REPO_DIR}/ASD_fun_BD_cpp.py" "${REPO_DIR}/synapse_pdist.csv" "$WORKER" "$PRIOR" "$TSV"; do
    [ -f "$f" ] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found: it reads prior_config.tsv and allocates the seeds." >&2
    exit 2
fi
if ! grep -q "def _apply_freeze(" "${REPO_DIR}/HPC_main_sweep.py"; then
    echo "ERROR: ${REPO_DIR}/HPC_main_sweep.py has no --freeze support." >&2
    echo "       Run  python3 apply_freeze_patch.py  in this folder first." >&2
    exit 2
fi
if ! BOXES_LINE="$(cd "$REPO_DIR" && python3 "$PRIOR" --check-applied)"; then
    echo "ERROR: ${REPO_DIR}/HPC_main_sweep.py does not hold the boxes of prior_config.tsv:" >&2
    echo "       ${BOXES_LINE}" >&2
    echo "       Run  python3 apply_prior_patch.py  in this folder (after step 0 if the TSV changed)." >&2
    exit 2
fi
BOXES="${BOXES_LINE#*changed from the registry: }"
FREEZE="$(cd "$REPO_DIR" && python3 "$PRIOR" --freeze-list)" || { echo "ERROR: freeze list: ${FREEZE}" >&2; exit 2; }
read -r CONN_PROB_LO CONN_PROB_HI <<< "$(cd "$REPO_DIR" && python3 "$PRIOR" --topology conn_prob)"
read -r STOA_GATE_LO STOA_GATE_HI <<< "$(cd "$REPO_DIR" && python3 "$PRIOR" --topology stoa_gate_p)"
for v in "${CONN_PROB_LO:-}" "${CONN_PROB_HI:-}" "${STOA_GATE_LO:-}" "${STOA_GATE_HI:-}"; do
    if ! printf '%s' "$v" | grep -Eq '^[0-9]*\.?[0-9]+([eE]-?[0-9]+)?$'; then
        echo "ERROR: the conn_prob and gate boxes were not read from prior_config.tsv (got '${v}')." >&2
        exit 2
    fi
done

# ===========================================================================
# SIZING -- Full mode, 108 neurons + 108 astrocytes, 180 s. N_TOPOLOGIES_WORKER
# is an estimate, not a measurement (claude/GiuliaGate_Campaign_Operations.md):
# MEAMA's campaign 1 on davinci took about 22.5 min per topology (200 s, ~3,300
# synapses on average, k = 1, the 48-worker barrier included); scaled by the
# synapse count (conn_prob U[0.05, 0.4]: ~2,600 on average) and by 180/200 s,
# about 16 min per topology, so 48 topologies is about 13 h of the 24 h
# walltime, a 2x margin. Re-size from the range check's measured time per
# topology or the first tasks' (APPLY_GG.txt step 10).
# ===========================================================================
SIZING_CONFIRMED="${SIZING_CONFIRMED:-0}"

C_MAX="${C_MAX:-300}"                # um, culture side (Giulia's v5)
SIMTIME="${SIMTIME:-180}"            # s, biological time per simulation (decided 2026-10-31)
N_TOPOLOGIES_WORKER="${N_TOPOLOGIES_WORKER:-48}"   # topologies per task (estimate, ~13 h)
K_PARAMS="${K_PARAMS:-1}"            # parameter vectors per worker per topology (as MEAMA)
WALLTIME="${WALLTIME:-24:00:00}"     # as MEAMA
TARGET="${TARGET:-1000000}"          # as MEAMA
HEADROOM_PCT="${HEADROOM_PCT:-200}"  # as MEAMA

# --- population: neurons and astrocytes, one each per site --------------------
DENSITY="${DENSITY:-1200}"           # neurons/mm^2
DENSITY_ASTRO="${DENSITY_ASTRO:-1200}"   # astrocytes/mm^2 (Na = Nn, decided June 2026)

# --- campaign configuration ---------------------------------------------------
CAMPAIGN_TAG="${CAMPAIGN_TAG:-cadex_hhgap_clamp_v1}"
CONDA_ENV="${CONDA_ENV:-brian_final}"
MODE="Full"                          # the gate and the astrocytes are read only in Full mode
SWEEP_GROUP="tripartite"             # minus the freeze list of prior_config.tsv: 15 free axes
CONN_RULE="${CONN_RULE:-flat}"
CONN_PERIODIC="${CONN_PERIODIC:-0}"

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
    exit 3
fi
if [ "$REFRACTORY_CLAMP" != "$CLAMP_STATE" ]; then
    echo "ERROR: REFRACTORY_CLAMP=${REFRACTORY_CLAMP}, but in ASD_fun_BD_cpp.py V held during t_ref = ${CLAMP_STATE}:" >&2
    echo "       ${DV_LINE}" >&2
    exit 4
fi
if [ "$(awk -v n="$DENSITY" -v a="$DENSITY_ASTRO" 'BEGIN{print (n>0 && a>0) ? 1 : 0}')" != "1" ]; then
    echo "ERROR: DENSITY and DENSITY_ASTRO must both be > 0 (got ${DENSITY}, ${DENSITY_ASTRO})." >&2
    exit 4
fi
if [ "$(awk -v a="$CONN_PROB_LO" -v b="$CONN_PROB_HI" 'BEGIN{print (a>=0 && a<=b && b<=1) ? 1 : 0}')" != "1" ] ||
   [ "$(awk -v a="$STOA_GATE_LO" -v b="$STOA_GATE_HI" 'BEGIN{print (a>=0 && a<=b && b<=1) ? 1 : 0}')" != "1" ]; then
    echo "ERROR: need 0 <= LOW <= HIGH <= 1 for conn_prob and the gate in prior_config.tsv." >&2
    exit 4
fi
# FREEZE travels through qsub -v, whose list separator is ','. It is sent with
# ',' -> ':' and '=' -> '@' and decoded by the worker.
if ! printf '%s' "$FREEZE" | grep -Eq '^[A-Za-z0-9_]+(=-?[0-9.]+([eE]-?[0-9]+)?)?(,[A-Za-z0-9_]+(=-?[0-9.]+([eE]-?[0-9]+)?)?)*$'; then
    echo "ERROR: FREEZE='${FREEZE}' (from prior_config.tsv) is not a list of NAME or NAME=NUMBER items." >&2
    exit 4
fi
FREEZE_ENC="$(printf '%s' "$FREEZE" | tr ',=' ':@')"
if [ "$DRYRUN" != "1" ] && [ -e "${REPO_DIR}/campaign_${CAMPAIGN_TAG}" ]; then
    echo "ERROR: ${REPO_DIR}/campaign_${CAMPAIGN_TAG} already exists; choose a new CAMPAIGN_TAG." >&2
    exit 4
fi

NN_EXP=$(awk -v c="$C_MAX" -v rn="$DENSITY" 'BEGIN{a=(c/1000)^2; printf "%d", int(rn*a+0.5)}')
NA_EXP=$(awk -v c="$C_MAX" -v ra="$DENSITY_ASTRO" 'BEGIN{a=(c/1000)^2; printf "%d", int(ra*a+0.5)}')

total_weight=0
for q in "${QUEUES[@]}"; do
    IFS=':' read -r name nc conc <<< "$q"
    total_weight=$(( total_weight + nc * conc ))
done

# ===========================================================================
# SEED ALLOCATION -- one verified-disjoint range per launch, against this
# folder's campaigns and ledger (artifacts/seed_ledger.tsv) and every campaign
# and ledger of every launch folder under SEED_SCAN_TREE. A scan tree that does
# not exist refuses the launch (exit 5).
# ===========================================================================
SEED_SCAN_TREE="${SEED_SCAN_TREE:-/davinci-1/home/ldellamea/ANN}"
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
if [ -f "$SEED_ALLOC" ] && ! grep -q -- "--scan-tree" "$SEED_ALLOC"; then
    echo "ERROR: ${SEED_ALLOC} cannot check other launch folders (no --scan-tree):" >&2
    echo "       restore the package's own with  git checkout -- seed_alloc.py   (in ${REPO_DIR})" >&2
    exit 5
fi
if [ -f "$SEED_ALLOC" ]; then
    LAUNCH_BASE=$(python3 "$SEED_ALLOC" \
        --span "$LAUNCH_SPAN" --seed-max "$SEED_MAX" \
        --root "$REPO_DIR" --ledger "$SEED_LEDGER" \
        --scan-tree "$SEED_SCAN_TREE" \
        --launch-id "$LAUNCH_ID" --tag "$CAMPAIGN_TAG" \
        ${DRYNOTE:+--note "$DRYNOTE"} \
        ${SEED_BASE_OVERRIDE:+--override "$SEED_BASE_OVERRIDE"})
    if [ -z "$LAUNCH_BASE" ]; then
        echo "ERROR: seed_alloc.py failed to allocate a seed range." >&2
        echo "       Inspect with: python3 $SEED_ALLOC --root $REPO_DIR --ledger $SEED_LEDGER --scan-tree $SEED_SCAN_TREE --report" >&2
        exit 5
    fi
else
    echo "ERROR: seed_alloc.py unavailable -- refusing to launch without a disjointness" >&2
    echo "       check (git checkout -- seed_alloc.py in ${REPO_DIR})." >&2
    exit 5
fi

echo "============================================================"
echo "GiuliaGate campaign : ${CAMPAIGN_TAG}"
echo "REPO_DIR       : ${REPO_DIR}"
echo "Launch ID      : ${LAUNCH_ID}"
echo "Seed range     : [${LAUNCH_BASE}, $((LAUNCH_BASE + LAUNCH_SPAN)))"
echo "Ledger         : ${SEED_LEDGER}"
echo "Seed guard     : this folder and every launch folder under ${SEED_SCAN_TREE}"
echo "------------------------------------------------------------"
echo "mode/group/rule: ${MODE} / ${SWEEP_GROUP} / ${CONN_RULE} (periodic=${CONN_PERIODIC})"
echo "freeze         : ${FREEZE}"
echo "boxes          : ${BOXES} (changed from the registry; the rest as the registry)"
echo "V clamp (t_ref): ${CLAMP_STATE}   [${DV_LINE}]"
echo "conn_prob      : U[${CONN_PROB_LO}, ${CONN_PROB_HI}] per topology"
echo "contact gate   : U[${STOA_GATE_LO}, ${STOA_GATE_HI}] per topology"
echo "well / simtime : ${C_MAX} um / ${SIMTIME} s"
echo "density        : ${DENSITY} neurons + ${DENSITY_ASTRO} astrocytes per mm^2 -> Nn=${NN_EXP}, Na=${NA_EXP}"
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

# Provenance: which model variant, prior and file versions this launch used.
PROV="${REPO_DIR}/campaign_${CAMPAIGN_TAG}/PROVENANCE_${LAUNCH_ID}.txt"
mkdir -p "${REPO_DIR}/campaign_${CAMPAIGN_TAG}"
{
    echo "GiuliaGate campaign provenance"
    echo "campaign_tag   : ${CAMPAIGN_TAG}"
    echo "launch_id      : ${LAUNCH_ID}"
    echo "repo_dir       : ${REPO_DIR}"
    echo "seed_range     : [${LAUNCH_BASE}, $((LAUNCH_BASE + LAUNCH_SPAN)))"
    echo "seed guard     : this folder and every launch folder under ${SEED_SCAN_TREE}"
    echo "mode / group   : ${MODE} / ${SWEEP_GROUP}"
    echo "freeze         : ${FREEZE}"
    echo "boxes          : ${BOXES}"
    echo "v_clamp_tref   : ${CLAMP_STATE}"
    echo "dV/dt line     : ${DV_LINE}"
    echo "density / well : ${DENSITY} neurons + ${DENSITY_ASTRO} astrocytes per mm^2, Nn=${NN_EXP}, Na=${NA_EXP} / ${C_MAX} um"
    echo "simtime        : ${SIMTIME} s"
    echo "conn_prob      : U[${CONN_PROB_LO}, ${CONN_PROB_HI}] (${CONN_RULE})"
    echo "contact gate   : U[${STOA_GATE_LO}, ${STOA_GATE_HI}]"
    echo "sizing         : N_TOPOLOGIES=${N_TOPOLOGIES_WORKER} k=${K_PARAMS} walltime ${WALLTIME}"
    echo "jobs           :${QDEL_IDS}"
    echo "sha256         :"
    ( cd "$REPO_DIR" && sha256sum HPC_main_sweep.py HPC_single_run.py ASD_fun_BD_cpp.py \
        synapse_pdist.csv prior_config.tsv apply_prior_patch.py launch_GG.sh submit_GG.sh seed_alloc.py )
} > "$PROV" 2>&1 || echo "WARNING: could not write ${PROV}" >&2
echo "provenance: ${PROV}"
echo "maximum yield if every task runs to completion: ${GRAND_MAX} sims"
echo "monitor : cd ${REPO_DIR} && python3 aggregate_sweep.py campaign_${CAMPAIGN_TAG} --target ${TARGET} | tail -3"
echo "stop all: qdel${QDEL_IDS}"
