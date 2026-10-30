#!/bin/bash
#PBS -S /bin/bash
#PBS -N refr_scan
#PBS -k eo
#PBS -l select=1:ncpus=16
#PBS -l walltime=12:00:00
##########################################################################
# Full data scan (every run) of refractory_scan.py, as a PBS job.
# Submit from the folder that holds this script and refractory_scan.py:
#
#     qsub -q cpu submit_refractory_scan.sh
#     qsub -q cpu -v CAMPAIGNS=/path/campaign_a:/path/campaign_b submit_refractory_scan.sh
#
# CAMPAIGNS is a ':'-separated list (qsub -v splits on ','). Default: MEAMA
# n01 and Giulia's v1 v2 v4 v5. Output: refractory_audit_full_<UTC>/ here,
# with the per-run tables <campaign>_runs.csv used to filter training sets.
# Reads the campaigns only.
##########################################################################
set -u

REPO_DIR="${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}"
DEFAULT_CAMPAIGNS="/davinci-1/home/ldellamea/ANN/Phenomenological/MEAMA/campaign_meama_rho1600_n01"
G=/davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro
DEFAULT_CAMPAIGNS="${DEFAULT_CAMPAIGNS}:${G}/campaign_cadex_hhgap_v1:${G}/campaign_cadex_hhgap_v2"
DEFAULT_CAMPAIGNS="${DEFAULT_CAMPAIGNS}:${G}/campaign_cadex_hhgap_v4:${G}/campaign_cadex_hhgap_v5"
CAMPAIGNS="${CAMPAIGNS:-$DEFAULT_CAMPAIGNS}"
OUT="${OUT:-${REPO_DIR}/refractory_audit_full_$(date -u +%Y%m%dT%H%M%SZ)}"
CONDA_ENV="${CONDA_ENV:-brian_final}"
NW="${PBS_NCPUS:-$(nproc)}"

if [ ! -f "${REPO_DIR}/refractory_scan.py" ]; then
    echo "ERROR: ${REPO_DIR}/refractory_scan.py not found (submit from its folder)." >&2
    exit 2
fi
cd "$REPO_DIR" || exit 2

set +u                               # module/conda hooks may read unset variables
module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "ERROR: conda activate ${CONDA_ENV} failed." >&2; exit 2; }
set -u
python -c "import os, sys, numpy; p = os.environ.get('CONDA_PREFIX', ''); assert p and sys.executable.startswith(p), sys.executable; print('[env]', sys.executable, '| numpy', numpy.__version__)" \
    || { echo "ERROR: ${CONDA_ENV} is not usable." >&2; exit 2; }

IFS=':' read -r -a CAMPS <<< "$CAMPAIGNS"
echo "refractory full scan | host $(hostname) | job ${PBS_JOBID:-?} | workers ${NW}"
echo "campaigns: ${CAMPS[*]}"
echo "output   : ${OUT}"
python refractory_scan.py "${CAMPS[@]}" --out "$OUT" --sample 0 --workers "$NW"
EXIT_CODE=$?
set +u; conda deactivate; set -u
exit $EXIT_CODE
