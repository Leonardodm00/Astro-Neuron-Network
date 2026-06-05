#!/bin/bash
##########################################################################
# launch_campaign.sh — submit the 300k campaign as a MIX of node types.
#
# Fires submit_sweep_mixed.sh as TWO independent job arrays:
#   * cfd/egeos  192-core nodes   (fast; the workhorse)
#   * intel/cpu   48-core nodes   (extra throughput from idle capacity)
# Both write into campaign_<TAG>/, with disjoint seed ranges (no duplicate
# draws) and distinct output tags (no dir collisions). aggregate_sweep.py
# sees both and tracks the combined progress to 300k.
#
# Per-task yield in ~18 h (N_TOPOLOGIES=80, k=1) under the 800 s/sim assumption:
#     192-core task -> 80 * 192 * 1 = 15,360 sims
#      48-core task -> 80 *  48 * 1 =  3,840 sims
# (recompute if you change N_TOPOLOGIES or k in submit_sweep_mixed.sh)
#
# These defaults are PRE-SIZED for 300k with NO pilot. The arrays are oversized
# on purpose (max clean yield ~491k, ~1.6x the target) so you still reach 300k
# even if a severe straggler spread (2x) walltime-kills tasks at ~54 topologies:
#     worst-case yield = 24*54*192 + 32*54*48 = 248,832 + 82,944 = 331,776 (>=300k)
# Procedure: launch, MONITOR with aggregate_sweep.py, and `qdel` the tail the
# moment the combined count crosses 300k (the extra tasks cost nothing — they
# are concurrency-capped and cancelled). If you somehow still fall short, just
# bump SEEDBASE_192/SEEDBASE_48 to fresh disjoint ranges and rerun this script.
#
# WALL TIME (default concurrency 3x192 + 4x48, 800 s/sim):
#     sims_per_wave = 3*15,360 + 4*3,840 = 46,080 + 15,360 = 61,440
#     waves to 300k = ceil(300000/61440) = 5     (~5 * ~20h ~ 4-5 days, + queue)
#     -> raise CONC_* if your allocation allows more nodes (fewer waves, faster);
#        lower them if it allows fewer (more waves, slower). Nothing else changes.
##########################################################################

WORKER="./submit_sweep_mixed.sh"

# ─── LOAD SPLIT — the only knob you need to touch ───────────────────────
# What fraction of the 300k sims should the 48-core (intel/cpu) queue do?
# The rest goes to the 192-core (cfd/egeos) queue.
#   LOAD_PCT_48=80  -> 48-core does 80%, 192-core does 20%
#   LOAD_PCT_48=50  -> equal split
#   LOAD_PCT_48=0   -> everything on 192-core (48-core array skipped)
#   LOAD_PCT_48=100 -> everything on 48-core  (192-core array skipped)
LOAD_PCT_48=80

TARGET=300000               # total sim target
N_TOPOLOGIES_WORKER=80      # MUST match N_TOPOLOGIES in submit_sweep_mixed.sh
HEADROOM_PCT=150            # 150% = 1.5x oversizing; guarantees target even if
                            # straggler spread reduces per-task yield by ~33%

# ─── Per-queue fixed knobs (queue name, cores, concurrency, seeds) ───────
Q_192="cfd"     ; NC_192=192 ; CONC_192=3 ; SEEDBASE_192=1000
Q_48="intel"    ; NC_48=48   ; CONC_48=4  ; SEEDBASE_48=100000
# ────────────────────────────────────────────────────────────────────────

# ─── Derive N_TASKS from the load split (do not edit below this line) ───
YIELD_192=$(( N_TOPOLOGIES_WORKER * NC_192 ))   # sims per 192-core task
YIELD_48=$(( N_TOPOLOGIES_WORKER * NC_48 ))     # sims per  48-core task

SIMS_192=$(( TARGET * (100 - LOAD_PCT_48) / 100 ))
SIMS_48=$(( TARGET  *        LOAD_PCT_48  / 100 ))

# Apply headroom, then ceiling-divide by per-task yield
N_TASKS_192=$(( (SIMS_192 * HEADROOM_PCT / 100 + YIELD_192 - 1) / YIELD_192 ))
N_TASKS_48=$((  (SIMS_48  * HEADROOM_PCT / 100 + YIELD_48  - 1) / YIELD_48  ))

# Wall-time estimate (mean, ignoring queue gaps and straggler spread):
#   wall = max( ceil(tasks/concurrency) ) * ~20h per wave
WAVES_192=$(( N_TASKS_192 > 0 ? (N_TASKS_192 + CONC_192 - 1) / CONC_192 : 0 ))
WAVES_48=$(( N_TASKS_48 > 0 ? (N_TASKS_48 + CONC_48 - 1) / CONC_48 : 0 ))
WALL_WAVES=$(( WAVES_192 > WAVES_48 ? WAVES_192 : WAVES_48 ))
WALL_H=$(( WALL_WAVES * 20 ))
MAX_YIELD=$(( N_TASKS_192 * YIELD_192 + N_TASKS_48 * YIELD_48 ))

# Safety: refuse to launch if the two seed ranges could overlap.
HI_192=$(( SEEDBASE_192 + N_TASKS_192 - 1 ))
HI_48=$((  SEEDBASE_48  + N_TASKS_48  - 1 ))
if [ "$SEEDBASE_48" -le "$HI_192" ] && [ "$SEEDBASE_192" -le "$HI_48" ]; then
    echo "ERROR: seed ranges overlap (192: $SEEDBASE_192-$HI_192, 48: $SEEDBASE_48-$HI_48)." >&2
    echo "       Increase SEEDBASE_48 so the ranges are disjoint." >&2
    exit 1
fi

echo "Launching mixed campaign (${LOAD_PCT_48}% on 48-core / $((100-LOAD_PCT_48))% on 192-core):"
echo "  192-core: ${N_TASKS_192} tasks (<=${CONC_192} at once) on '${Q_192}', seeds ${SEEDBASE_192}-${HI_192}  [${SIMS_192} target sims]"
echo "   48-core: ${N_TASKS_48} tasks (<=${CONC_48} at once) on '${Q_48}', seeds ${SEEDBASE_48}-${HI_48}  [${SIMS_48} target sims]"
echo "  max yield: ${MAX_YIELD} sims  |  est. wall: ~${WALL_H}h (~$((WALL_H/24))d, mean, +queue gaps)"

# 192-core array (skipped if LOAD_PCT_48=100)
if [ "$N_TASKS_192" -gt 0 ]; then
    JID_192=$(qsub \
        -q "$Q_192" \
        -l "select=1:ncpus=${NC_192},walltime=24:00:00" \
        -J "0-$((N_TASKS_192-1))%${CONC_192}" \
        -v "NODETAG=c192,SEED_BASE=${SEEDBASE_192}" \
        "$WORKER")
    echo "  submitted 192-core array: $JID_192"
else
    echo "  192-core: skipped (LOAD_PCT_48=100)"
    JID_192=""
fi

# 48-core array (skipped if LOAD_PCT_48=0)
if [ "$N_TASKS_48" -gt 0 ]; then
    JID_48=$(qsub \
        -q "$Q_48" \
        -l "select=1:ncpus=${NC_48},walltime=24:00:00" \
        -J "0-$((N_TASKS_48-1))%${CONC_48}" \
        -v "NODETAG=c48,SEED_BASE=${SEEDBASE_48}" \
        "$WORKER")
    echo "  submitted  48-core array: $JID_48"
else
    echo "   48-core: skipped (LOAD_PCT_48=0)"
    JID_48=""
fi

echo
echo "Monitor combined progress with:"
echo "    python aggregate_sweep.py campaign_300k_v1 --target 300000"
echo "Once it reports >=300000, stop the tail with:"
QDEL_IDS=""
[ -n "$JID_192" ] && QDEL_IDS="${JID_192%%\[*}[]"
[ -n "$JID_48"  ] && QDEL_IDS="$QDEL_IDS ${JID_48%%\[*}[]"
echo "    qdel $QDEL_IDS"
