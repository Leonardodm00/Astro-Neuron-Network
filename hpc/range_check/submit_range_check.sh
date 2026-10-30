#!/bin/bash
#PBS -S /bin/bash
#PBS -N range_check
#PBS -k eo
#PBS -l select=1:ncpus=48
#PBS -l walltime=24:00:00
##########################################################################
# The standard prior range calibration of one setup (specs/SPEC.md Block 6,
# decided 2026-10-30): 20 s runs at the corners of the prior box and at random
# cells of its 7-bin grid, then the analysis and the figures. Normally
# submitted by launch_range_check.sh, which runs the corners on one node, the
# interior on another, and the analysis as a short third job once both have
# ended. Submitted alone, it runs everything on one node:
#
#   qsub -q cpu -v SETUP=lcdup submit_range_check.sh
#
# PHASES     = all (default) | corners | interior | none   what this job runs
#              corners : range_check.py --rc-phase corners, levels CORNER_LEVELS
#                        (0 = conn_prob at its lower bound, 1 = upper)
#              interior: range_check.py --rc-phase interior, levels 0..BINS-1
#                        (the conn_prob bins)
# ANALYZE    = 1 (default) | 0      range_analysis.py and range_plots.py at the end
# CHECK_ONLY = 1                    run the guards below and exit, before conda
#                                   and python (launch_range_check.sh's precheck)
#
# lcdup : LIB_DIR=~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP,
#         the folder in the git clone (decided 2026-10-30), set up with
#         APPLY_LCDUP.txt steps 0-5a (pipeline copied, freeze and box patches,
#         clamp); the LindaColloDUP settings of launch_LCDUP.sh, written out
#         below (13 free axes + conn_prob: a full factorial of 16,384 corners).
# giulia: LIB_DIR=~/ANN/Phenomenological/Main/Giulia_Astro; settings from
#         LIKE = the job_args.json of its campaign_cadex_hhgap_v5, plus EXTRA
#         (e.g. the gate box of the new setup, if its code has
#         --stoa_gate_lo/hi; the gate is then drawn per topology, not designed).
# Any variable below can be overridden with qsub -v NAME=value (EXTRA is a
# space-separated list of HPC_main_sweep.py arguments). qsub -v splits on
# commas, so a different freeze list goes in as FREEZE_ENC with ':' for ','
# and '@' for '=' (as submit_LCDUP.sh), e.g.
# FREEZE_ENC=Cm:I_inj:VA@-48:DeltaA:VR@-58.2:EC50_ampa:EC50_nmda:alpha_syn:O_G:Omega_G
# Output: OUT=~/ANN/range_check/<setup>_<UTC> (launch_range_check.sh passes
# one OUT to its three jobs), outside every project folder. Reads the setup
# folder, writes only under OUT and node scratch. Logs: ~/<job name>.o<id> / .e<id>.
##########################################################################
set -u

HERE="${PBS_O_WORKDIR:-$PWD}"
ANN=/davinci-1/home/ldellamea/ANN
SETUP="${SETUP:-}"
PHASES="${PHASES:-all}"
ANALYZE="${ANALYZE:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
CONDA_ENV="${CONDA_ENV:-brian_final}"
NW="${PBS_NCPUS:-$(nproc)}"
BINS="${BINS:-7}"
CORNER_LEVELS="${CORNER_LEVELS:-0 1}"
INTERIOR_LEVELS="${INTERIOR_LEVELS:-$(seq -s ' ' 0 $(( BINS - 1 )))}"
CORNER_TOPOS="${CORNER_TOPOS:-8}"
CORNER_BUDGET="${CORNER_BUDGET:-16384}"
INTERIOR_TOPOS="${INTERIOR_TOPOS:-7}"
INTERIOR_RUNS="${INTERIOR_RUNS:-16384}"
RC_SIMTIME="${RC_SIMTIME:-20}"
EXTRA="${EXTRA:-}"
FREEZE_ENC="${FREEZE_ENC:-}"
FREEZE="Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G"
[ -n "$FREEZE_ENC" ] && FREEZE="$(printf '%s' "$FREEZE_ENC" | tr ':@' ',=')"

case "$SETUP" in
    lcdup)
        LIB_DIR="${LIB_DIR:-$ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP}"
        LIKE="${LIKE:-}"
        SETTINGS=(--mode Neuronal --sweep_group neuron_synapse
                  --freeze "$FREEZE"
                  --density 1300 --density_astro 0 --c_max 350 --Nn 115 --Na 0
                  --conn_rule flat --conn_prob_lo 0.01 --conn_prob_hi 0.5
                  --displ_bias 20 --topology_mode wallach --gj_dist 200 --gj_max_dist 150
                  --stoa_cutoff 70 --stoa_sigma 200
                  --seed_device 50 --seed_neuron 39 --seed_synapse 35 --seed_astro 60 --dpi 100)
        ;;
    giulia)
        LIB_DIR="${LIB_DIR:-$ANN/Phenomenological/Main/Giulia_Astro}"
        LIKE="${LIKE:-$(ls -d "$LIB_DIR"/campaign_cadex_hhgap_v5/sweep_*/job_args.json 2>/dev/null | head -1)}"
        SETTINGS=(--dpi 100)
        if [ -z "$LIKE" ]; then
            echo "ERROR: no job_args.json found for LIKE (set LIKE=<campaign task>/job_args.json)." >&2
            exit 2
        fi
        ;;
    *)
        echo "ERROR: SETUP must be lcdup or giulia (qsub -v SETUP=...)." >&2
        exit 2
        ;;
esac
OUT="${OUT:-$ANN/range_check/${SETUP}_$(date -u +%Y%m%dT%H%M%SZ)}"

case "$PHASES" in
    all|corners|interior|none) ;;
    *) echo "ERROR: PHASES must be all, corners, interior or none (got ${PHASES})." >&2; exit 2 ;;
esac
if [ "$ANALYZE" != "0" ] && [ "$ANALYZE" != "1" ]; then
    echo "ERROR: ANALYZE must be 0 or 1 (got ${ANALYZE})." >&2
    exit 2
fi
if [ "$PHASES" = "none" ] && [ "$ANALYZE" = "0" ]; then
    echo "ERROR: PHASES=none with ANALYZE=0 does nothing." >&2
    exit 2
fi
if [ ! -f "${HERE}/range_check.py" ] || [ ! -f "${HERE}/range_design.py" ]; then
    echo "ERROR: range_check.py or range_design.py not found in ${HERE} (submit from their folder)." >&2
    exit 2
fi
if [ ! -f "${LIB_DIR}/HPC_main_sweep.py" ]; then
    echo "ERROR: ${LIB_DIR} has no HPC_main_sweep.py." >&2
    exit 2
fi
if [ "$SETUP" = "lcdup" ]; then
    n_box=$(grep -cF "[LindaColloDUP box 2026-10-30]" "${LIB_DIR}/HPC_main_sweep.py")
    if [ "$n_box" != "2" ]; then
        echo "ERROR: ${LIB_DIR}/HPC_main_sweep.py lacks the g_ampa/g_nmda box patch (${n_box} of 2 markers):" >&2
        echo "       run APPLY_LCDUP.txt steps 0-5a in that folder first." >&2
        exit 2
    fi
fi
if [ "$CHECK_ONLY" = "1" ]; then
    echo "[range_check] precheck OK: setup ${SETUP}, LIB_DIR ${LIB_DIR}"
    exit 0
fi
if [ "$PHASES" = "none" ] && ! ls "$OUT"/*/design.json >/dev/null 2>&1; then
    echo "ERROR: nothing to analyse: no level folder with design.json in ${OUT}." >&2
    exit 2
fi
cd "$HERE" || exit 2

set +u                               # module/conda hooks may read unset variables
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "ERROR: conda activate ${CONDA_ENV} failed." >&2; exit 2; }
set -u
python -c "import os, sys, brian2, numpy, scipy, pandas; p = os.environ.get('CONDA_PREFIX', ''); assert p and sys.executable.startswith(p), sys.executable; print('[env]', sys.executable, '| brian2', brian2.__version__, '| numpy', numpy.__version__)" \
    || { echo "ERROR: ${CONDA_ENV} is not usable." >&2; exit 2; }
export OMP_NUM_THREADS=1
export MAKEFLAGS="-j1"

LIKE_ARG=()
[ -n "$LIKE" ] && LIKE_ARG=(--rc-like "$LIKE")
read -r -a EXTRA_A <<< "$EXTRA"
COMMON=(--lib_dir "$LIB_DIR" --n_workers "$NW" --rc-out "$OUT" --rc-simtime "$RC_SIMTIME"
        ${LIKE_ARG[@]+"${LIKE_ARG[@]}"} "${SETTINGS[@]}" ${EXTRA_A[@]+"${EXTRA_A[@]}"})

echo "============================================================"
echo "range_check | setup ${SETUP} | host $(hostname) | job ${PBS_JOBID:-?} | workers ${NW}"
echo "phases   : ${PHASES} | analysis ${ANALYZE}"
echo "LIB_DIR  : ${LIB_DIR}"
echo "dV/dt    : $(grep -m1 -E '^[[:space:]]*dV/dt[[:space:]]*=' "${LIB_DIR}/ASD_fun_BD_cpp.py" | sed 's/^[[:space:]]*//')"
echo "LIKE     : ${LIKE:-<none>}"
echo "corners  : levels ${CORNER_LEVELS} | ${CORNER_TOPOS} topologies per level | budget ${CORNER_BUDGET}"
echo "interior : levels ${INTERIOR_LEVELS} | ${BINS} bins | ${INTERIOR_RUNS} cells | ${INTERIOR_TOPOS} topologies per level"
echo "runs     : ${RC_SIMTIME} s each | EXTRA ${EXTRA:-<none>}"
echo "OUT      : ${OUT}"
echo "============================================================"
mkdir -p "$OUT"
status=0
if [ "$PHASES" = "all" ] || [ "$PHASES" = "corners" ]; then
    for L in $CORNER_LEVELS; do
        python range_check.py --rc-phase corners --rc-level "$L" --rc-corner-topologies "$CORNER_TOPOS" \
            --rc-corner-budget "$CORNER_BUDGET" "${COMMON[@]}" || status=1
    done
fi
if [ "$PHASES" = "all" ] || [ "$PHASES" = "interior" ]; then
    for L in $INTERIOR_LEVELS; do
        python range_check.py --rc-phase interior --rc-level "$L" --rc-bins "$BINS" \
            --rc-interior-runs "$INTERIOR_RUNS" --rc-interior-topologies "$INTERIOR_TOPOS" "${COMMON[@]}" || status=1
    done
fi
if [ "$ANALYZE" = "1" ]; then
    python range_analysis.py "$OUT" --workers "$NW" || status=1
    python range_plots.py "$OUT" || status=1
fi
set +u; conda deactivate; set -u
if [ "$ANALYZE" = "1" ]; then
    echo "[range_check] finished with status ${status}; send ${OUT}/calibration_summary.txt and the three png files"
else
    echo "[range_check] phases ${PHASES} finished with status ${status}; the analysis job reads ${OUT}"
fi
exit $status
