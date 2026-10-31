#!/bin/bash
##########################################################################
# launch_range_check.sh -- the standard prior range calibration of one setup
# on two 48-core nodes (decided 2026-10-30) or four (NODES=4, asked for
# Giulia's setup on 2026-10-31), submitted from the login node as PBS jobs.
# NODES=2 (default), three jobs:
#   job 1 (Q_CORNERS, default intel): the corners of the prior box
#         (PHASES=corners; 16,384 corners + 512 centre points for LindaColloDUP)
#   job 2 (Q_INTERIOR, default cpu): random cells of the 7-bin grid
#         (PHASES=interior; 16,464 cells for LindaColloDUP)
#   job 3 (Q_ANALYSIS, default cpu, 8 cores): range_analysis.py and
#         range_plots.py, held until jobs 1 and 2 have ended. It uses
#         -W depend=afterany, so it also runs if one of them failed; the
#         summaries then show which parts are there.
# NODES=4, five jobs: the corners split by conn_prob level (rc_corners_a:
#   CORNER_LEVELS=0, rc_corners_b: CORNER_LEVELS=1, both on Q_CORNERS), the
#   interior split by conn_prob bin (rc_interior_a: bins 0..ceil(BINS/2)-1,
#   rc_interior_b: the rest, both on Q_INTERIOR), and the analysis held by
#   depend=afterany on all four. The level lists travel with ':' between the
#   levels (qsub -v splits on commas). CORNER_LEVELS and INTERIOR_LEVELS
#   cannot be set with NODES=4.
# All jobs write into one OUT. Run from anywhere:
#
#   SETUP=lcdup bash launch_range_check.sh             # submit the jobs
#   DRYRUN=1 SETUP=lcdup bash launch_range_check.sh    # print them, submit nothing
#   NODES=4 SETUP=giulia bash launch_range_check.sh    # Giulia's setup on four nodes
#
# First, submit_range_check.sh runs its own guards once here (CHECK_ONLY=1):
# a setup folder that is not ready stops the launch, with nothing submitted.
# Variables: OUT (default ~/ANN/range_check/<setup>_<UTC>), NODES (2 or 4), Q_CORNERS,
# Q_INTERIOR, Q_ANALYSIS, NCPUS (48), WALLTIME (24:00:00), ANALYSIS_NCPUS (8),
# ANALYSIS_WALLTIME (02:00:00). LIB_DIR, LIKE, BINS, CORNER_LEVELS,
# CORNER_TOPOS, CORNER_BUDGET, INTERIOR_LEVELS, INTERIOR_TOPOS, INTERIOR_RUNS,
# RC_SIMTIME, EXTRA, FREEZE_ENC and CONDA_ENV are passed on to the jobs when set
# (no commas: qsub -v splits on them). Spec: specs/SPEC.md, Block 6.
##########################################################################
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT="${HERE}/submit_range_check.sh"
ANN=/davinci-1/home/ldellamea/ANN
SETUP="${SETUP:-}"
DRYRUN="${DRYRUN:-0}"
NODES="${NODES:-2}"
Q_CORNERS="${Q_CORNERS:-intel}"
Q_INTERIOR="${Q_INTERIOR:-cpu}"
Q_ANALYSIS="${Q_ANALYSIS:-cpu}"
NCPUS="${NCPUS:-48}"
WALLTIME="${WALLTIME:-24:00:00}"
ANALYSIS_NCPUS="${ANALYSIS_NCPUS:-8}"
ANALYSIS_WALLTIME="${ANALYSIS_WALLTIME:-02:00:00}"
OUT="${OUT:-$ANN/range_check/${SETUP}_$(date -u +%Y%m%dT%H%M%SZ)}"

if [ ! -f "$SUBMIT" ]; then
    echo "ERROR: ${SUBMIT} not found." >&2
    exit 2
fi
case "$SETUP" in
    lcdup|giulia) ;;
    *) echo "ERROR: SETUP must be lcdup or giulia." >&2; exit 2 ;;
esac
case "$OUT" in
    *,*) echo "ERROR: OUT contains a comma; qsub -v splits on commas." >&2; exit 2 ;;
esac
case "$NODES" in
    2|4) ;;
    *) echo "ERROR: NODES must be 2 or 4 (got ${NODES})." >&2; exit 2 ;;
esac
if [ "$NODES" = "4" ]; then
    if [ -n "${CORNER_LEVELS:-}" ] || [ -n "${INTERIOR_LEVELS:-}" ]; then
        echo "ERROR: NODES=4 sets the levels of each job itself; unset CORNER_LEVELS and INTERIOR_LEVELS" >&2
        echo "       (or use NODES=2 to choose levels)." >&2
        exit 2
    fi
    B="${BINS:-7}"
    case "$B" in
        ''|*[!0-9]*) echo "ERROR: BINS must be an integer (got ${B})." >&2; exit 2 ;;
    esac
    if [ "$B" -lt 2 ]; then
        echo "ERROR: NODES=4 splits the interior bins in two: BINS must be at least 2." >&2
        exit 2
    fi
    HALF=$(( (B + 1) / 2 ))
    IL_A="$(seq -s ':' 0 $(( HALF - 1 )))"
    IL_B="$(seq -s ':' "$HALF" $(( B - 1 )))"
fi
PASS=""
for v in LIB_DIR LIKE BINS CORNER_LEVELS CORNER_TOPOS CORNER_BUDGET INTERIOR_LEVELS INTERIOR_TOPOS INTERIOR_RUNS RC_SIMTIME EXTRA FREEZE_ENC CONDA_ENV; do
    val="${!v:-}"
    [ -z "$val" ] && continue
    case "$val" in
        *,*) echo "ERROR: ${v} contains a comma; qsub -v splits on commas." >&2; exit 2 ;;
    esac
    PASS="${PASS},${v}=${val}"
done

# the job script's own guards, once, on this node; nothing is submitted if they fail
if ! ( cd "$HERE" && unset PBS_O_WORKDIR && CHECK_ONLY=1 SETUP="$SETUP" bash "$SUBMIT" ); then
    echo "ERROR: precheck failed (see above); nothing submitted." >&2
    exit 2
fi

V="SETUP=${SETUP},OUT=${OUT}${PASS}"
# the compute jobs: name, queue, extra qsub -v settings
if [ "$NODES" = "2" ]; then
    NAMES=(rc_corners rc_interior)
    QUEUES=("$Q_CORNERS" "$Q_INTERIOR")
    VARS=("PHASES=corners,ANALYZE=0" "PHASES=interior,ANALYZE=0")
else
    NAMES=(rc_corners_a rc_corners_b rc_interior_a rc_interior_b)
    QUEUES=("$Q_CORNERS" "$Q_CORNERS" "$Q_INTERIOR" "$Q_INTERIOR")
    VARS=("PHASES=corners,ANALYZE=0,CORNER_LEVELS=0" "PHASES=corners,ANALYZE=0,CORNER_LEVELS=1"
          "PHASES=interior,ANALYZE=0,INTERIOR_LEVELS=${IL_A}" "PHASES=interior,ANALYZE=0,INTERIOR_LEVELS=${IL_B}")
fi
N=${#NAMES[@]}

echo "range check | setup ${SETUP} | ${NODES} nodes | OUT ${OUT}"
if [ "$DRYRUN" = "1" ]; then
    for (( k = 0; k < N; k++ )); do
        echo "[dry run] qsub -q ${QUEUES[$k]} -N ${NAMES[$k]} -l select=1:ncpus=${NCPUS} -l walltime=${WALLTIME} -v ${V},${VARS[$k]} ${SUBMIT}"
    done
    echo "[dry run] qsub -q ${Q_ANALYSIS} -N rc_analysis -l select=1:ncpus=${ANALYSIS_NCPUS} -l walltime=${ANALYSIS_WALLTIME} -W depend=afterany:<the ${N} jobs above> -v ${V},PHASES=none,ANALYZE=1 ${SUBMIT}"
    echo "[dry run] nothing submitted."
    exit 0
fi

cd "$HERE" || exit 2
JOBS=()
for (( k = 0; k < N; k++ )); do
    J=$(qsub -q "${QUEUES[$k]}" -N "${NAMES[$k]}" -l "select=1:ncpus=${NCPUS}" -l "walltime=${WALLTIME}" \
            -v "${V},${VARS[$k]}" "$SUBMIT") || J=""
    if [ -z "$J" ]; then
        if [ "$k" -eq 0 ]; then
            echo "ERROR: the ${NAMES[$k]} job was not submitted; nothing else was submitted." >&2
        else
            echo "ERROR: the ${NAMES[$k]} job was not submitted; stop the jobs already submitted with  qdel ${JOBS[*]}" >&2
        fi
        exit 3
    fi
    JOBS+=("$J")
    printf '%-14s: %s  (queue %s, %s cores)\n' "${NAMES[$k]}" "$J" "${QUEUES[$k]}" "$NCPUS"
done
DEP="$(IFS=:; echo "${JOBS[*]}")"
JA=$(qsub -q "$Q_ANALYSIS" -N rc_analysis -l "select=1:ncpus=${ANALYSIS_NCPUS}" -l "walltime=${ANALYSIS_WALLTIME}" \
        -W "depend=afterany:${DEP}" -v "${V},PHASES=none,ANALYZE=1" "$SUBMIT") || JA=""
if [ -z "$JA" ]; then
    echo "WARNING: the analysis job was not submitted. Once the ${N} jobs have ended, run from ${HERE}:" >&2
    echo "  qsub -q ${Q_ANALYSIS} -l select=1:ncpus=${ANALYSIS_NCPUS} -v \"${V},PHASES=none,ANALYZE=1\" submit_range_check.sh" >&2
    exit 3
fi
printf '%-14s: %s  (queue %s, %s cores; starts when the %d jobs have ended)\n' rc_analysis "$JA" "$Q_ANALYSIS" \
    "$ANALYSIS_NCPUS" "$N"
echo "follow        : qstat -u \$USER   (logs ~/rc_*.o<id> and .e<id>)"
echo "stop all      : qdel ${JOBS[*]} ${JA}"
echo "results       : ${OUT}/calibration_summary.txt and the png files, once the analysis job has ended"
