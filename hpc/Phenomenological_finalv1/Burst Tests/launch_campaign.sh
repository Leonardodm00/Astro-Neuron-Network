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
#                    'all'/'neuron'/'neuron_synapse' all sweep neuron+synapse (24).
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
#   CONN_PERIODIC : 1 => minimum-image (flat-torus) distances. ON, per the
#                   decision to train on periodic boundaries (cleaner plateaus,
#                   larger size-invariant region than finite -- see the study).
#   DENSITY       : neurons/mm^2. If set, Nn is DERIVED from C_MAX in the worker
#                   (Nn=round(DENSITY*(C_MAX/1000)^2)) and held fixed across
#                   sizes -- the scale-invariance prerequisite. Empty => the
#                   worker's NN is used as-is. SET THIS (and C_MAX in
#                   submit_sweep_mixed.sh) to pin density before launching.
CONN_RULE="weibull"
CONN_PERIODIC=1
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
#   * SEEDBASEs start at 1,000,000 and are spaced 1,000,000 apart -- disjoint
#     from your existing campaign (which used 1000 / 100000), so there are NO
#     duplicate draws even if the older jobs are still running.
#   * Edit freely: delete a line to drop a queue; append one to add a queue.
#     If you add 'amd', VERIFY its ncpus first -- it was not in the cheat-sheet.
#   * To RERUN later without seed collisions, add a constant to every seedbase.
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

echo "Spreading ${TARGET} sims across ${#QUEUES[@]} queues"
echo "(work proportional to ncpus*concurrency; up to ${total_weight} cores at once)"
echo "mode = ${MODE}   sweep_group = ${SWEEP_GROUP}"
echo "burst_design = ${BURST_DESIGN:-<none>}   campaign_tag = ${CAMPAIGN_TAG}"
echo "conn_rule = ${CONN_RULE}   conn_periodic = ${CONN_PERIODIC}"
echo

QDEL_IDS=""
GRAND_MAX=0
for q in "${QUEUES[@]}"; do
    IFS=':' read -r name nc conc sb <<< "$q"
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
            -v "NODETAG=${name},SEED_BASE=${sb},SWEEP_GROUP=${SWEEP_GROUP},MODE=${MODE},CONN_RULE=${CONN_RULE},CONN_PERIODIC=${CONN_PERIODIC},DENSITY=${DENSITY},BURST_DESIGN=${BURST_DESIGN},CAMPAIGN_TAG=${CAMPAIGN_TAG}" \
            "$WORKER")
    else
        # PBS rejects a single-element array; submit a plain job (IDX defaults to 0)
        jid=$(qsub -q "$name" \
            -l "select=1:ncpus=${nc},walltime=${WALLTIME}" \
            -v "NODETAG=${name},SEED_BASE=${sb},SWEEP_GROUP=${SWEEP_GROUP},MODE=${MODE},CONN_RULE=${CONN_RULE},CONN_PERIODIC=${CONN_PERIODIC},DENSITY=${DENSITY},BURST_DESIGN=${BURST_DESIGN},CAMPAIGN_TAG=${CAMPAIGN_TAG}" \
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
