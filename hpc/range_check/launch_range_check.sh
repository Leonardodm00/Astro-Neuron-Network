#!/bin/bash
##########################################################################
# launch_range_check.sh -- the standard prior range calibration of one setup
# on two 48-core nodes (decided 2026-10-30), submitted from the login node as
# three PBS jobs:
#   job 1 (Q_CORNERS, default intel): the corners of the prior box
#         (PHASES=corners; 16,384 corners + 512 centre points for LindaColloDUP)
#   job 2 (Q_INTERIOR, default cpu): random cells of the 7-bin grid
#         (PHASES=interior; 16,464 cells for LindaColloDUP)
#   job 3 (Q_ANALYSIS, default cpu, 8 cores): range_analysis.py and
#         range_plots.py, held until jobs 1 and 2 have ended. It uses
#         -W depend=afterany, so it also runs if one of them failed; the
#         summaries then show which parts are there.
# All three write into one OUT. Run from anywhere:
#
#   SETUP=lcdup bash launch_range_check.sh             # submit the three jobs
#   DRYRUN=1 SETUP=lcdup bash launch_range_check.sh    # print them, submit nothing
#
# First, submit_range_check.sh runs its own guards once here (CHECK_ONLY=1):
# a setup folder that is not ready stops the launch, with nothing submitted.
# Variables: OUT (default ~/ANN/range_check/<setup>_<UTC>), Q_CORNERS,
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
SUB_CORNERS=(qsub -q "$Q_CORNERS" -N rc_corners -l "select=1:ncpus=${NCPUS}" -l "walltime=${WALLTIME}"
             -v "${V},PHASES=corners,ANALYZE=0" "$SUBMIT")
SUB_INTERIOR=(qsub -q "$Q_INTERIOR" -N rc_interior -l "select=1:ncpus=${NCPUS}" -l "walltime=${WALLTIME}"
              -v "${V},PHASES=interior,ANALYZE=0" "$SUBMIT")

echo "range check | setup ${SETUP} | OUT ${OUT}"
if [ "$DRYRUN" = "1" ]; then
    echo "[dry run] ${SUB_CORNERS[*]}"
    echo "[dry run] ${SUB_INTERIOR[*]}"
    echo "[dry run] qsub -q ${Q_ANALYSIS} -N rc_analysis -l select=1:ncpus=${ANALYSIS_NCPUS} -l walltime=${ANALYSIS_WALLTIME} -W depend=afterany:<corners job>:<interior job> -v ${V},PHASES=none,ANALYZE=1 ${SUBMIT}"
    echo "[dry run] nothing submitted."
    exit 0
fi

cd "$HERE" || exit 2
J1=$("${SUB_CORNERS[@]}") || J1=""
if [ -z "$J1" ]; then
    echo "ERROR: the corners job was not submitted; nothing else was submitted." >&2
    exit 3
fi
echo "corners job  : ${J1}  (queue ${Q_CORNERS}, ${NCPUS} cores)"
J2=$("${SUB_INTERIOR[@]}") || J2=""
if [ -z "$J2" ]; then
    echo "ERROR: the interior job was not submitted; stop the corners job with  qdel ${J1}" >&2
    exit 3
fi
echo "interior job : ${J2}  (queue ${Q_INTERIOR}, ${NCPUS} cores)"
J3=$(qsub -q "$Q_ANALYSIS" -N rc_analysis -l "select=1:ncpus=${ANALYSIS_NCPUS}" -l "walltime=${ANALYSIS_WALLTIME}" \
        -W "depend=afterany:${J1}:${J2}" -v "${V},PHASES=none,ANALYZE=1" "$SUBMIT") || J3=""
if [ -z "$J3" ]; then
    echo "WARNING: the analysis job was not submitted. Once both jobs have ended, run from ${HERE}:" >&2
    echo "  qsub -q ${Q_ANALYSIS} -l select=1:ncpus=${ANALYSIS_NCPUS} -v \"${V},PHASES=none,ANALYZE=1\" submit_range_check.sh" >&2
    exit 3
fi
echo "analysis job : ${J3}  (queue ${Q_ANALYSIS}, ${ANALYSIS_NCPUS} cores; starts when both have ended)"
echo "follow       : qstat -u \$USER   (logs ~/rc_corners.o*, ~/rc_interior.o*, ~/rc_analysis.o*)"
echo "stop all     : qdel ${J1} ${J2} ${J3}"
echo "results      : ${OUT}/calibration_summary.txt and the three png files, once the analysis job has ended"
