#!/bin/bash
##########################################################################
# launch_campaign.sh - spread a sweep campaign across MULTIPLE CPU queues.
#
# Submits submit_sweep_mixed.sh as one independent job array per queue. Work is
# auto-distributed proportional to each queue's throughput (ncpus x concurrency),
# so all queues finish at roughly the same wall time. Every array writes into
# campaign_<TAG>/ with a queue-name output tag (no dir collisions) and a disjoint
# seed range (no duplicate parameter draws). status_sweep.py / aggregate_sweep.py
# see every queue and track combined progress to the target.
#
# Procedure:  bash launch_campaign.sh  ->  MONITOR  ->  qdel the tail at target.
##########################################################################

WORKER="./submit_sweep_mixed.sh"

# Campaign-wide simulation mode + parameter group, passed to every worker via -v
# (overriding the worker's own defaults).
#   MODE        : Full | Neuronal
#   SWEEP_GROUP : all | neuron | synapse | astro | neuron_synapse
#   In MODE=Full     'neuron' sweeps only the 10 CAdEx axes.
#   In MODE=Neuronal astro axes are inert and synaptic axes are ALWAYS swept, so
#                    'all'/'neuron'/'neuron_synapse' all sweep neuron+synapse (23
#                    = 9 NEURON_PARAMS + 14 SYNAPSE_PARAMS; DeltaT/VT/gL frozen).
#                    This comment read "24" until 2026-08. The registry has
#                    always resolved to 23 and manifest.json's active_indices
#                    was the correct number; only the comment was wrong.
#   WARNING: under MODE=Neuronal, SWEEP_GROUP=all DRAWS the 10 astrocyte axes
#            while instantiating no astrocyte, gliotransmission or
#            synapse-to-astrocyte group -- 10 causally inert axes at once.
#            Use neuron_synapse. HPC_main_sweep now declares any such axis in
#            manifest.json 'axis_declaration'.inert_axes and prints it at launch.
MODE="Neuronal"
SWEEP_GROUP="neuron_synapse"

# Burst-validation design (leave EMPTY for the normal SBI campaign).
#   A : sweep ONLY adaptation (delta_gA, tauA, gbarA), synapses frozen at a
#       burst-favourable ignition point. Decisive adaptation test.
#   B : joint burst-permissive neuron+synapse sweep (13 narrowed axes).
# When set, it OVERRIDES SWEEP_GROUP in the driver and outputs go to a SEPARATE
# campaign dir (campaign_cadex_burstval_<A|B>/), never colliding with production.
# Override at the CLI:  BURST_DESIGN=A bash launch_campaign.sh
BURST_DESIGN="${BURST_DESIGN:-}"

# Single-sourced campaign tag (passed to every worker via -v so the worker does
# not have to recompute it). Auto-separates burst-validation outputs.
CAMPAIGN_TAG="cadex_hhgap_v1"
[ -n "$BURST_DESIGN" ] && CAMPAIGN_TAG="cadex_burstval_${BURST_DESIGN}"

# Campaign-wide connectivity rule, also passed to every worker via -v.
#   CONN_RULE     : flat | weibull   (this SBI campaign uses weibull: every
#                   worker draws the (p0,d0,beta) kernel per topology from the
#                   TRIMMED KERNEL_BOUNDS in HPC_main_sweep.py). Set 'flat' to
#                   reproduce the legacy Bernoulli campaign.
#   CONN_PERIODIC : 1 => minimum-image (flat-torus) distances; 0 => finite
#                   boundaries. SET TO 0, matching what the eight
#                   campaign_cadex_rho1300v* campaigns actually ran: their
#                   job_args.json records conn_periodic=false for all eight.
#                   This comment previously asserted periodic boundaries were
#                   the intended design while the launcher that produced those
#                   campaigns used 0. Finite boundaries were the intent; the
#                   comment was wrong and the committed value (1) would have
#                   silently produced a NON-COMPARABLE campaign set on the next
#                   launch.
#                   DO NOT flip this to 1 without treating the output as a
#                   separate campaign set: it changes the distance metric the
#                   weibull kernel is evaluated on, which is precisely the
#                   simulator-versus-intent gap a misspecification gate exists
#                   to detect.
#   DENSITY       : neurons/mm^2. If set, Nn is DERIVED from C_MAX in the worker
#                   (Nn=round(DENSITY*(C_MAX/1000)^2)) and held fixed across
#                   sizes -- the scale-invariance prerequisite. Empty => the
#                   worker's NN is used as-is. SET THIS (and C_MAX in
#                   submit_sweep_mixed.sh) to pin density before launching.
CONN_RULE="weibull"
CONN_PERIODIC=0
DENSITY=""                  # e.g. 2000  (neurons/mm^2); empty keeps worker's NN

# --- Campaign-wide config -------------------------------------------------
TARGET=300000
# Burst validation does not need the full SBI budget: A is decisive with a few
# thousand sims (each topology already samples 48-192 adaptation draws on fixed
# wiring); B needs more to map a 13-D box. Stop early via qdel the moment the
# detector shows bursts.
if [ -n "$BURST_DESIGN" ]; then
    case "$BURST_DESIGN" in
        A) TARGET=10000 ;;
        B) TARGET=60000 ;;
    esac
fi
N_TOPOLOGIES_WORKER=80      # MUST match N_TOPOLOGIES in submit_sweep_mixed.sh
                            # NOTE: weibull sweeps a 3-D kernel prior on top of
                            # the 34-D run_args, so the topology budget should be
                            # raised substantially vs the flat (1-D conn_prob)
                            # campaign. Scale this with your compute decision.
K_PARAMS=1                  # MUST match N_PARAMS_PER_WORKER in submit_sweep_mixed.sh
HEADROOM_PCT=200            # 2x oversize -- covers straggler spread and walltime
                            # kills (with real ~4-6k s/topology an 80-topo task
                            # is killed partway through a 24h slot). Tasks are
                            # concurrency-capped and qdel'd at target, so
                            # over-provisioning costs nothing.
WALLTIME="24:00:00"

# --- Queues:  "name:ncpus:concurrency:seedbase" ---------------------------
# Verified CPU compute queues on davinci-1 (ncpus from the queue cheat-sheet).
#   * Work auto-distributes proportional to (ncpus * concurrency).
#   * Edit freely: delete a line to drop a queue; append one to add a queue.
#     If you add 'amd', VERIFY its ncpus first -- it was not in the cheat-sheet.
#   * The 4th field (seedbase) is now IGNORED and kept only so that older queue
#     tables still parse. Seed bases are ALLOCATED PER LAUNCH by seed_alloc.py
#     -- see the SEED ALLOCATION block below. Do NOT hand-edit seeds here; that
#     is the mechanism that produced 2.9x duplicated draws across v1..v9.
QUEUES=(
    "cfd:192:4:1000000"      # AMD EPYC Turin 192c   (lightly loaded: 3 running)
    "egeos:192:8:2000000"    # AMD EPYC Turin 192c   (huge: 811 running, 1 queued)
    "intel:48:10:3000000"    # Intel Sapphire  48c   (busy: 19 running, 20 queued)
    "cpu:48:10:4000000"      # Intel Cascade   48c   (25 running, 6 queued)
  # "amd:96:8:5000000"       # AMD EPYC Genoa 96c (dvnode057-062) -- RESTRICTED
                              #   enable only if your project has entitlement
)
# --------------------------------------------------------------------------

# Total throughput weight = sum of (ncpus * concurrency)
total_weight=0
for q in "${QUEUES[@]}"; do
    IFS=':' read -r name nc conc sb <<< "$q"
    total_weight=$(( total_weight + nc * conc ))
done

# ===========================================================================
# SEED ALLOCATION -- one fresh, verified-disjoint range per LAUNCH
# ===========================================================================
# WHY. submit_sweep_mixed.sh computes SEED_MASTER = SEED_BASE + PBS_ARRAY_INDEX
# and HPC_main_sweep.py does master_rng = default_rng(SEED_MASTER), from which
# every topology, every connectivity-kernel draw, every parameter vector and
# every per-run seed descends. A task's whole content is therefore a
# deterministic function of that one integer.
#
# Previously SEED_BASE was hardcoded in the queue table, so every relaunch
# restarted the array index at 0 from the same base: a later campaign replayed
# every seed of an earlier one and appended a handful. Measured over six
# exported campaigns, 251,614 rows held only 86,750 distinct parameter vectors
# (duplication factor 2.90), with v4/v7/v8 wholly contained in v3.
#
# The defect was seed REUSE, not seed QUALITY: numpy runs an integer seed
# through SeedSequence, so consecutive bases already give well-separated
# streams. The only thing that had to change is that the base must change.
#
# NOW. seed_alloc.py draws a base from OS entropy and REJECTS it unless the
# whole span is disjoint from (a) every _resolved_seed_master recorded in any
# campaign_*/sweep_*/job_args.json and (b) every span in artifacts/
# seed_ledger.tsv -- retrying until free. Collision-free by construction, and
# it cannot be defeated by forgetting to edit a tag or a constant.
#
# REPLAY. Every task writes its own _resolved_seed_master, and the allocation
# is appended to the ledger, so any launch is reproducible after the fact:
#     SEED_BASE_OVERRIDE=<base> bash launch_campaign.sh
# ---------------------------------------------------------------------------
SEED_MAX=2147483647            # 2**31-1: the ceiling _resolve_seed_master uses
SEED_SPACING=100000            # per-queue block; >> any plausible task count
SEED_LEDGER="artifacts/seed_ledger.tsv"
# The root launcher does not define CAMPAIGN_TAG (the worker does), so read
# it back for the ledger row. Never overrides a tag already set here.
CAMPAIGN_TAG="${CAMPAIGN_TAG:-$(grep -m1 '^CAMPAIGN_TAG=' "$WORKER" 2>/dev/null \
                 | cut -d'"' -f2)}"

LAUNCH_SPAN=$(( ${#QUEUES[@]} * SEED_SPACING ))
LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"

# Locate seed_alloc.py next to this script, then one level up (the Burst Tests
# and Giulia launchers share the canonical copy in the parent directory).
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED_ALLOC=""
for _cand in "$_HERE/seed_alloc.py" "$_HERE/../seed_alloc.py"; do
    [ -f "$_cand" ] && { SEED_ALLOC="$_cand"; break; }
done

if [ -n "${SEED_BASE_OVERRIDE:-}" ]; then
    echo "!! SEED_BASE_OVERRIDE=${SEED_BASE_OVERRIDE} -- REPLAY MODE."
    echo "!! This reproduces the draws of whichever launch used that base."
    echo "!! Unset it for a normal campaign."
fi

if [ -n "$SEED_ALLOC" ] && command -v python3 >/dev/null 2>&1; then
    LAUNCH_BASE=$(python3 "$SEED_ALLOC" \
        --span "$LAUNCH_SPAN" --seed-max "$SEED_MAX" \
        --ledger "$SEED_LEDGER" --launch-id "$LAUNCH_ID" \
        --tag "${CAMPAIGN_TAG:-unknown}" \
        ${SEED_BASE_OVERRIDE:+--override "$SEED_BASE_OVERRIDE"})
    if [ -z "$LAUNCH_BASE" ]; then
        echo "ERROR: seed_alloc.py failed to allocate a seed range. Refusing" >&2
        echo "       to launch: an unverified base is how v1..v9 collided." >&2
        echo "       Inspect with: python3 $SEED_ALLOC --report" >&2
        exit 5
    fi
else
    # Degraded path: no python3, or seed_alloc.py missing. Fall back to raw OS
    # entropy WITHOUT the disjointness check. Still vastly better than a
    # hardcoded constant, but say so loudly rather than pretending otherwise.
    echo "WARNING: seed_alloc.py or python3 unavailable -- allocating from raw" >&2
    echo "         OS entropy WITHOUT checking previously recorded seeds." >&2
    LAUNCH_BASE=$(( (16#$(head -c6 /dev/urandom | od -An -tx1 | tr -d ' \n') \
                     % (SEED_MAX - LAUNCH_SPAN - 1)) + 1 ))
    [ -n "${SEED_BASE_OVERRIDE:-}" ] && LAUNCH_BASE="$SEED_BASE_OVERRIDE"
fi

echo
echo "Launch ID   : ${LAUNCH_ID}"
echo "Seed range  : [${LAUNCH_BASE}, $((LAUNCH_BASE + LAUNCH_SPAN))) "\
     "(${#QUEUES[@]} queues x ${SEED_SPACING})"
echo "Ledger      : ${SEED_LEDGER}"
echo

echo "Spreading ${TARGET} sims across ${#QUEUES[@]} queues"
echo "(work proportional to ncpus*concurrency; up to ${total_weight} cores at once)"
echo "mode = ${MODE}   sweep_group = ${SWEEP_GROUP}"
echo "burst_design = ${BURST_DESIGN:-<none>}   campaign_tag = ${CAMPAIGN_TAG}"
echo "conn_rule = ${CONN_RULE}   conn_periodic = ${CONN_PERIODIC}"
echo

QDEL_IDS=""
GRAND_MAX=0
_q_index=0                      # position in QUEUES, drives the per-queue base
for q in "${QUEUES[@]}"; do
    IFS=':' read -r name nc conc sb <<< "$q"
    # Per-queue base: derived from this launch's allocation, NOT from the
    # queue table's 4th field (which is ignored, see the SEED ALLOCATION block).
    sb=$(( LAUNCH_BASE + _q_index * SEED_SPACING ))
    _q_index=$(( _q_index + 1 ))

    weight=$(( nc * conc ))
    sims_q=$(( TARGET * weight / total_weight ))
    yield_q=$(( N_TOPOLOGIES_WORKER * nc * K_PARAMS ))
    ntasks=$(( (sims_q * HEADROOM_PCT / 100 + yield_q - 1) / yield_q ))
    [ "$ntasks" -lt 1 ] && ntasks=1
    hi=$(( sb + ntasks - 1 ))
    GRAND_MAX=$(( GRAND_MAX + ntasks * yield_q ))

    printf "  %-6s : %3d tasks  (<=%2d at once, %3dc)   seeds %d-%d   ~%d target sims\n" \
        "$name" "$ntasks" "$conc" "$nc" "$sb" "$hi" "$sims_q"

    if [ "$ntasks" -gt 1 ]; then
        jid=$(qsub -q "$name" \
            -l "select=1:ncpus=${nc},walltime=${WALLTIME}" \
            -J "0-$((ntasks-1))%${conc}" \
            -v "NODETAG=${name},SEED_BASE=${sb},LAUNCH_ID=${LAUNCH_ID},SWEEP_GROUP=${SWEEP_GROUP},MODE=${MODE},CONN_RULE=${CONN_RULE},CONN_PERIODIC=${CONN_PERIODIC},DENSITY=${DENSITY},BURST_DESIGN=${BURST_DESIGN},CAMPAIGN_TAG=${CAMPAIGN_TAG}" \
            "$WORKER")
    else
        # PBS rejects a single-element array; submit a plain job (IDX defaults to 0)
        jid=$(qsub -q "$name" \
            -l "select=1:ncpus=${nc},walltime=${WALLTIME}" \
            -v "NODETAG=${name},SEED_BASE=${sb},LAUNCH_ID=${LAUNCH_ID},SWEEP_GROUP=${SWEEP_GROUP},MODE=${MODE},CONN_RULE=${CONN_RULE},CONN_PERIODIC=${CONN_PERIODIC},DENSITY=${DENSITY},BURST_DESIGN=${BURST_DESIGN},CAMPAIGN_TAG=${CAMPAIGN_TAG}" \
            "$WORKER")
    fi
    echo "           submitted: ${jid:-<FAILED>}"
    [ -n "$jid" ] && QDEL_IDS="$QDEL_IDS ${jid%%.*}"
done

echo
echo "  max yield across all queues: ${GRAND_MAX} sims  (target ${TARGET})"
echo
echo "Monitor combined progress:"
echo "    python status_sweep.py campaign_${CAMPAIGN_TAG} --target ${TARGET}"
echo "Score bursts in the outputs (once some iters exist):"
echo "    python network_burst_detector.py --out_dir campaign_${CAMPAIGN_TAG} --plot_first 8"
echo "Stop ALL remaining subjobs once you reach ${TARGET}:"
echo "    qdel${QDEL_IDS}"
