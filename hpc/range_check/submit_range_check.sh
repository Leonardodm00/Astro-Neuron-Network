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
# giulia: LIB_DIR=~/ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/GiuliaGate,
#         Giulia's new campaign in the git clone (decided 2026-10-31), set up
#         with APPLY_GG.txt steps 0-4. Its prior_config.tsv is the only source of
#         the prior: once python is available this script reads from it, through
#         apply_prior_patch.py, the freeze list (unless FREEZE_ENC is given), the
#         conn_prob box and the gate box, and checks that HPC_main_sweep.py holds
#         its boxes. The rest are launch_GG.sh's settings, written out below:
#         Full, tripartite, 1200 neurons and 1200 astrocytes per mm^2 on 300 um
#         (15 free axes + conn_prob: a resolution-8 fraction of 16,384 corners).
#         The gate is drawn per topology from its box, as in the campaign; it is
#         not a design dimension.
# Any variable below can be overridden with qsub -v NAME=value (EXTRA is a
# space-separated list of HPC_main_sweep.py arguments). qsub -v splits on
# commas, so a different freeze list goes in as FREEZE_ENC with ':' for ','
# and '@' for '=' (as submit_LCDUP.sh), e.g.
# FREEZE_ENC=Cm:I_inj:VA@-48:DeltaA:VR@-58.2:EC50_ampa:EC50_nmda:alpha_syn:O_G:Omega_G
# and level lists may separate their levels with ':' instead of spaces
# (CORNER_LEVELS=1, INTERIOR_LEVELS=4:5:6; launch_range_check.sh NODES=4).
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
CORNER_LEVELS="$(printf '%s' "$CORNER_LEVELS" | tr ':' ' ')"
INTERIOR_LEVELS="$(printf '%s' "$INTERIOR_LEVELS" | tr ':' ' ')"
CORNER_TOPOS="${CORNER_TOPOS:-8}"
CORNER_BUDGET="${CORNER_BUDGET:-16384}"
INTERIOR_TOPOS="${INTERIOR_TOPOS:-7}"
INTERIOR_RUNS="${INTERIOR_RUNS:-16384}"
RC_SIMTIME="${RC_SIMTIME:-20}"
EXTRA="${EXTRA:-}"
FREEZE_ENC="${FREEZE_ENC:-}"
FREEZE=""
[ -n "$FREEZE_ENC" ] && FREEZE="$(printf '%s' "$FREEZE_ENC" | tr ':@' ',=')"
for L in $CORNER_LEVELS $INTERIOR_LEVELS; do
    case "$L" in
        ''|*[!0-9]*) echo "ERROR: CORNER_LEVELS / INTERIOR_LEVELS must list integers (got ${L})." >&2; exit 2 ;;
    esac
done

case "$SETUP" in
    lcdup)
        LIB_DIR="${LIB_DIR:-$ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/LindaColloDUP}"
        LIKE="${LIKE:-}"
        [ -z "$FREEZE" ] && FREEZE="Cm,I_inj,VA=-48,DeltaA,VR=-58.2,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G"
        SETTINGS=(--mode Neuronal --sweep_group neuron_synapse
                  --freeze "$FREEZE"
                  --density 1300 --density_astro 0 --c_max 350 --Nn 115 --Na 0
                  --conn_rule flat --conn_prob_lo 0.01 --conn_prob_hi 0.5
                  --displ_bias 20 --topology_mode wallach --gj_dist 200 --gj_max_dist 150
                  --stoa_cutoff 70 --stoa_sigma 200
                  --seed_device 50 --seed_neuron 39 --seed_synapse 35 --seed_astro 60 --dpi 100)
        ;;
    giulia)
        LIB_DIR="${LIB_DIR:-$ANN/Astro-Neuron-Network/hpc/Phenomenological_finalv1/GiuliaGate}"
        LIKE="${LIKE:-}"
        SETTINGS=()          # from the folder's prior_config.tsv, once python is available (below)
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
if [ "$SETUP" = "giulia" ]; then
    for f in apply_prior_patch.py prior_config.tsv; do
        if [ ! -f "${LIB_DIR}/${f}" ]; then
            echo "ERROR: ${LIB_DIR} has no ${f}: run APPLY_GG.txt steps 0-4 there first." >&2
            exit 2
        fi
    done
    if ! grep -q "def _apply_freeze(" "${LIB_DIR}/HPC_main_sweep.py" \
            || ! grep -q -- "--stoa_gate_lo" "${LIB_DIR}/HPC_main_sweep.py"; then
        echo "ERROR: ${LIB_DIR}/HPC_main_sweep.py lacks the --freeze patch or the gate options:" >&2
        echo "       run APPLY_GG.txt steps 0-4 there first." >&2
        exit 2
    fi
    # the boxes, on the login node (any python3, standard library only); the job
    # repeats the check with the environment's python before it runs anything
    if [ "$CHECK_ONLY" = "1" ] && command -v python3 >/dev/null 2>&1; then
        if ! python3 "${LIB_DIR}/apply_prior_patch.py" --check-applied --tsv "${LIB_DIR}/prior_config.tsv" \
                --file "${LIB_DIR}/HPC_main_sweep.py"; then
            echo "ERROR: ${LIB_DIR}/HPC_main_sweep.py does not hold the boxes of prior_config.tsv:" >&2
            echo "       run APPLY_GG.txt step 4 there first." >&2
            exit 2
        fi
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
# The interpreter comes from CONDA_PREFIX, never from a bare `python` on PATH:
# on davinci a module-loaded system python can win on PATH even after a
# successful `conda activate` (seen by MEAMA's job script, 2026-08-31, which
# resolves it the same way).
PY="${CONDA_PREFIX:-}/bin/python"
if [ ! -x "$PY" ]; then
    echo "ERROR: ${PY} is not an executable python (CONDA_PREFIX=${CONDA_PREFIX:-unset})." >&2
    exit 2
fi
"$PY" -c "import os, sys, brian2, numpy, scipy, pandas; p = os.environ.get('CONDA_PREFIX', ''); assert p and sys.executable.startswith(p), sys.executable; print('[env]', sys.executable, '| brian2', brian2.__version__, '| numpy', numpy.__version__)" \
    || { echo "ERROR: ${CONDA_ENV} is not usable (${PY})." >&2; exit 2; }
export OMP_NUM_THREADS=1
export MAKEFLAGS="-j1"

if [ "$SETUP" = "giulia" ]; then
    GG=("$PY" "${LIB_DIR}/apply_prior_patch.py" --tsv "${LIB_DIR}/prior_config.tsv" --file "${LIB_DIR}/HPC_main_sweep.py")
    if ! "${GG[@]}" --check-applied; then
        echo "ERROR: ${LIB_DIR}/HPC_main_sweep.py does not hold the boxes of prior_config.tsv (APPLY_GG.txt step 4)." >&2
        exit 2
    fi
    [ -z "$FREEZE" ] && FREEZE="$("${GG[@]}" --freeze-list)"
    read -r C_LO C_HI <<< "$("${GG[@]}" --topology conn_prob)"
    read -r G_LO G_HI <<< "$("${GG[@]}" --topology stoa_gate_p)"
    for v in "${C_LO:-}" "${C_HI:-}" "${G_LO:-}" "${G_HI:-}"; do
        if ! [[ "$v" =~ ^[0-9]*\.?[0-9]+([eE]-?[0-9]+)?$ ]]; then
            echo "ERROR: could not read the conn_prob and gate boxes from prior_config.tsv (got '${v}')." >&2
            exit 2
        fi
    done
    if [ -z "$FREEZE" ]; then
        echo "ERROR: empty freeze list from prior_config.tsv: refusing to run the full tripartite group." >&2
        exit 2
    fi
    SETTINGS=(--mode Full --sweep_group tripartite
              --freeze "$FREEZE"
              --density 1200 --density_astro 1200 --c_max 300 --Nn 115 --Na 115
              --conn_rule flat --conn_prob_lo "$C_LO" --conn_prob_hi "$C_HI"
              --stoa_gate_lo "$G_LO" --stoa_gate_hi "$G_HI" --seed_stoa_gate 71
              --displ_bias 20 --topology_mode wallach --gj_dist 200 --gj_max_dist 150
              --stoa_cutoff 70 --stoa_sigma 200
              --seed_device 50 --seed_neuron 39 --seed_synapse 35 --seed_astro 60 --dpi 100)
fi

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
echo "freeze   : ${FREEZE}"
echo "corners  : levels ${CORNER_LEVELS} | ${CORNER_TOPOS} topologies per level | budget ${CORNER_BUDGET}"
echo "interior : levels ${INTERIOR_LEVELS} | ${BINS} bins | ${INTERIOR_RUNS} cells | ${INTERIOR_TOPOS} topologies per level"
echo "runs     : ${RC_SIMTIME} s each | EXTRA ${EXTRA:-<none>}"
echo "OUT      : ${OUT}"
echo "============================================================"
mkdir -p "$OUT"
status=0
if [ "$PHASES" = "all" ] || [ "$PHASES" = "corners" ]; then
    for L in $CORNER_LEVELS; do
        "$PY" range_check.py --rc-phase corners --rc-level "$L" --rc-corner-topologies "$CORNER_TOPOS" \
            --rc-corner-budget "$CORNER_BUDGET" "${COMMON[@]}" || status=1
    done
fi
if [ "$PHASES" = "all" ] || [ "$PHASES" = "interior" ]; then
    for L in $INTERIOR_LEVELS; do
        "$PY" range_check.py --rc-phase interior --rc-level "$L" --rc-bins "$BINS" \
            --rc-interior-runs "$INTERIOR_RUNS" --rc-interior-topologies "$INTERIOR_TOPOS" "${COMMON[@]}" || status=1
    done
fi
if [ "$ANALYZE" = "1" ]; then
    "$PY" range_analysis.py "$OUT" --workers "$NW" || status=1
    "$PY" range_plots.py "$OUT" || status=1
fi
set +u; conda deactivate; set -u
if [ "$ANALYZE" = "1" ]; then
    echo "[range_check] finished with status ${status}; send ${OUT}/calibration_summary.txt and the three png files"
else
    echo "[range_check] phases ${PHASES} finished with status ${status}; the analysis job reads ${OUT}"
fi
exit $status
