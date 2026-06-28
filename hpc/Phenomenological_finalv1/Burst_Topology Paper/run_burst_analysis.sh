#!/bin/bash
#PBS -S /bin/bash
#PBS -N burst_poster
#PBS -k eo
#PBS -l select=1:ncpus=48
#PBS -l walltime=04:00:00
##########################################################################
# Single-node burst analysis + poster figures over ONE campaign.
#
# This runs run_burst_analysis.py: screen (reuse/build burst_features.csv) ->
# pick the top-K exemplars closest to the Mossink DIV28 control -> render the
# poster figure matrix. It is a SINGLE job (not an array): one node, many cores.
#
# SUBMIT (cluster):
#     qsub -v CAMPAIGN=campaign_<TAG> run_burst_analysis.sh
#   override resources/knobs at submit time, e.g.:
#     qsub -l select=1:ncpus=192 -l walltime=08:00:00 \
#          -v CAMPAIGN=campaign_cadex_hhgap_v1,TOPK=8,FULL_TOP=2 \
#          run_burst_analysis.sh
#
# RUN DIRECTLY (login node / quick look) -- campaign as the first argument:
#     bash run_burst_analysis.sh campaign_<TAG>
#
# CAMPAIGN may come from `-v CAMPAIGN=...`, the first positional argument, or be
# omitted entirely -- then the Python driver auto-discovers the single
# campaign_* directory in the working directory. The campaign folder is expected
# to sit beside this script (i.e. in $PBS_O_WORKDIR), per the campaign_<TAG>
# layout written by the sweep.
#
# Tunable knobs (env / qsub -v): TOPK FULL_TOP PALETTES PANEL_MM DPI IFR_SIGMA
#   CANDIDATE_POOL EXTRA_ARGS. EXTRA_ARGS is passed verbatim, e.g.
#   EXTRA_ARGS="--rescreen --max-sims 20000".
##########################################################################

set -u

# --- knobs (override via qsub -v VAR=... or the environment) -------------
TOPK="${TOPK:-6}"
FULL_TOP="${FULL_TOP:-1}"
PALETTES="${PALETTES:-teal_analogous,teal_amber_comp,teal_triadic}"
PANEL_MM="${PANEL_MM:-380}"
DPI="${DPI:-300}"
IFR_SIGMA="${IFR_SIGMA:-0.05}"
CANDIDATE_POOL="${CANDIDATE_POOL:-300}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- working directory ---------------------------------------------------
# Under PBS, cd to the submission directory (PBS starts jobs in $HOME), which is
# where the campaign_<TAG> folder lives. For a DIRECT run, stay in the caller's
# current directory so a relative campaign path (or auto-discovery there) works.
if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "$PBS_O_WORKDIR"
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- campaign: -v CAMPAIGN, else $1, else let the driver auto-discover ----
CAMPAIGN="${CAMPAIGN:-${1:-}}"

# --- environment (mirror submit_sweep_mixed.sh) --------------------------
module load python 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate brian_env
fi

# N_WORKERS = ncpus on a node; nproc when run directly
if [ -n "${PBS_NCPUS:-}" ]; then WORKERS="$PBS_NCPUS"; else WORKERS="$(nproc)"; fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MAKEFLAGS="-j1"

echo "============================================================"
echo "Burst analysis + poster figures"
echo "Host / JobID     : $(hostname) / ${PBS_JOBID:-<direct-run>}"
echo "Workdir          : $(pwd)"
echo "Script dir       : $SCRIPT_DIR"
echo "Campaign         : ${CAMPAIGN:-<auto-discover campaign_*>}"
echo "Workers (=ncpus) : $WORKERS"
echo "TopK / full-top  : $TOPK / $FULL_TOP"
echo "Palettes         : $PALETTES"
echo "Panel / dpi      : ${PANEL_MM} mm / ${DPI}"
echo "IFR sigma        : ${IFR_SIGMA} s   candidate-pool: ${CANDIDATE_POOL}"
[ -n "$EXTRA_ARGS" ] && echo "Extra args       : $EXTRA_ARGS"
echo "============================================================"

# --- assemble args (campaign positional FIRST when provided) -------------
ARGS=( --workers "$WORKERS" --topk "$TOPK" --full-top "$FULL_TOP" \
       --palettes "$PALETTES" --panel-mm "$PANEL_MM" --dpi "$DPI" \
       --ifr-sigma "$IFR_SIGMA" --candidate-pool "$CANDIDATE_POOL" )
[ -n "$CAMPAIGN" ] && ARGS=( "$CAMPAIGN" "${ARGS[@]}" )
# EXTRA_ARGS intentionally word-split into separate flags
[ -n "$EXTRA_ARGS" ] && ARGS+=( $EXTRA_ARGS )

python "$SCRIPT_DIR/run_burst_analysis.py" "${ARGS[@]}"
EXIT_CODE=$?

if command -v conda >/dev/null 2>&1; then conda deactivate; fi
echo "[exit] run_burst_analysis.py returned $EXIT_CODE"
exit $EXIT_CODE
