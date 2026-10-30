#!/bin/bash
#PBS -S /bin/bash
#PBS -N range_check
#PBS -k eo
#PBS -l select=1:ncpus=48
#PBS -l walltime=08:00:00
##########################################################################
# 20 s runs over one setup's prior boxes (specs/SPEC.md Block 6): the oat
# phase at three conn_prob levels, the prior phase, then the analysis and the
# figures. One setup per job; submit from the folder holding this script:
#
#   qsub -q cpu -v SETUP=lcdup  submit_range_check.sh
#   qsub -q cpu -v SETUP=giulia,EXTRA="--stoa_gate_lo 0.2 --stoa_gate_hi 1.0",OAT_EXTRA="--stoa_gate_lo 0.6 --stoa_gate_hi 0.6" submit_range_check.sh
#
# lcdup : LIB_DIR=~/ANN/Phenomenological/LindaColloDUP (created and patched
#         with APPLY_LCDUP.txt steps 0-5a); the LindaColloDUP settings of
#         launch_LCDUP.sh and submit_LCDUP.sh, written out below.
# giulia: LIB_DIR=~/ANN/Phenomenological/Main/Giulia_Astro; settings from
#         LIKE = the job_args.json of its campaign_cadex_hhgap_v5, plus EXTRA
#         (the gate box of the new setup, if its code has --stoa_gate_lo/hi).
# Any variable below can be overridden with qsub -v NAME=value (EXTRA and
# OAT_EXTRA are space-separated HPC_main_sweep.py arguments; OAT_EXTRA only
# for the oat phase). qsub -v splits on commas, so a different freeze list
# goes in as FREEZE_ENC with ':' for ',' and '@' for '=' (as submit_LCDUP.sh),
# e.g. FREEZE_ENC=Cm:I_inj:VA@-48:DeltaA:VR@-58.2:EC50_ampa:EC50_nmda:alpha_syn:O_G:Omega_G
# Output: OUT=~/ANN/range_check/<setup>_<UTC>, outside
# every project folder. Reads the setup folder, writes only under OUT and
# node scratch. Logs: ~/range_check.o<id> / .e<id>.
##########################################################################
set -u

HERE="${PBS_O_WORKDIR:-$PWD}"
ANN=/davinci-1/home/ldellamea/ANN
SETUP="${SETUP:-}"
CONDA_ENV="${CONDA_ENV:-brian_final}"
NW="${PBS_NCPUS:-$(nproc)}"
CONN_LEVELS="${CONN_LEVELS:-0.1 0.35 0.6}"
POINTS="${POINTS:-7}"
OAT_TOPOS="${OAT_TOPOS:-2}"
PRIOR_TOPOS="${PRIOR_TOPOS:-8}"
RC_SIMTIME="${RC_SIMTIME:-20}"
EXTRA="${EXTRA:-}"
OAT_EXTRA="${OAT_EXTRA:-}"
FREEZE_ENC="${FREEZE_ENC:-}"
FREEZE="Cm,I_inj,VA=-50,DeltaA,VR=-55,EC50_ampa,EC50_nmda,alpha_syn,O_G,Omega_G"
[ -n "$FREEZE_ENC" ] && FREEZE="$(printf '%s' "$FREEZE_ENC" | tr ':@' ',=')"

case "$SETUP" in
    lcdup)
        LIB_DIR="${LIB_DIR:-$ANN/Phenomenological/LindaColloDUP}"
        LIKE="${LIKE:-}"
        SETTINGS=(--mode Neuronal --sweep_group neuron_synapse
                  --freeze "$FREEZE"
                  --density 1300 --density_astro 0 --c_max 350 --Nn 115 --Na 0
                  --conn_rule flat --conn_prob_lo 0.1 --conn_prob_hi 0.6
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

if [ ! -f "${HERE}/range_check.py" ]; then
    echo "ERROR: ${HERE}/range_check.py not found (submit from its folder)." >&2
    exit 2
fi
if [ ! -f "${LIB_DIR}/HPC_main_sweep.py" ]; then
    echo "ERROR: ${LIB_DIR} has no HPC_main_sweep.py." >&2
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
read -r -a OAT_EXTRA_A <<< "$OAT_EXTRA"
COMMON=(--lib_dir "$LIB_DIR" --n_workers "$NW" --rc-out "$OUT" --rc-simtime "$RC_SIMTIME"
        ${LIKE_ARG[@]+"${LIKE_ARG[@]}"} "${SETTINGS[@]}" ${EXTRA_A[@]+"${EXTRA_A[@]}"})

echo "============================================================"
echo "range_check | setup ${SETUP} | host $(hostname) | job ${PBS_JOBID:-?} | workers ${NW}"
echo "LIB_DIR  : ${LIB_DIR}"
echo "dV/dt    : $(grep -m1 -E '^[[:space:]]*dV/dt[[:space:]]*=' "${LIB_DIR}/ASD_fun_BD_cpp.py" | sed 's/^[[:space:]]*//')"
echo "LIKE     : ${LIKE:-<none>}"
echo "oat      : conn_prob ${CONN_LEVELS} | ${POINTS} values per axis | ${OAT_TOPOS} topologies | ${OAT_EXTRA}"
echo "prior    : ${PRIOR_TOPOS} topologies x ${NW} | ${RC_SIMTIME} s per run | ${EXTRA}"
echo "OUT      : ${OUT}"
echo "============================================================"
mkdir -p "$OUT"
status=0
for P in $CONN_LEVELS; do
    python range_check.py --rc-phase oat --rc-conn "$P" --rc-points "$POINTS" --rc-topologies "$OAT_TOPOS" \
        "${COMMON[@]}" ${OAT_EXTRA_A[@]+"${OAT_EXTRA_A[@]}"} || status=1
done
python range_check.py --rc-phase prior --rc-topologies "$PRIOR_TOPOS" "${COMMON[@]}" || status=1
python range_analysis.py "$OUT" --workers "$NW" || status=1
python range_plots.py "$OUT" || status=1
set +u; conda deactivate; set -u
echo "[range_check] finished with status ${status}; send ${OUT}/prior_summary.txt and the three png files"
exit $status
