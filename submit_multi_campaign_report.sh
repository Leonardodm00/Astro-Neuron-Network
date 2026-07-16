#!/bin/bash
# =============================================================================
# submit_multi_campaign_report.sh
#
# PBS wrapper to run multi_campaign_report.py on a davinci-1 compute node
# (non-interactive). Produces the cohort-layer report + summary + PNGs.
#
# Reproduces, on a batch node:
#
#   python multi_campaign_report.py . \
#          --glob 'campaign_cadex_rho1300v*' \
#          --jobs 3 \
#          --out ./cohort_rho1300
#
# ---------------------------------------------------------------------------
# USAGE
#   # 1) from the directory that CONTAINS the campaign_cadex_rho1300v* folders
#   #    (i.e. the dir where you would have typed the interactive command):
#   qsub -v RUN_DIR="$PWD" /path/to/submit_multi_campaign_report.sh
#
#   # 2) or hard-code RUN_DIR below and just: qsub submit_multi_campaign_report.sh
#
#   # smoke test (synthetic self-check, no data needed):
#   qsub -v SMOKE=1 /path/to/submit_multi_campaign_report.sh
#
# NOTE ON CPUs: the script parallelises over campaigns with --jobs. With
# --jobs 3 it uses at most 3 worker processes AND only when there is more than
# one campaign. Requesting 3 ncpus is therefore the right size; more cores sit
# idle. ncpus is exported to the python call via PBS_NCPUS so the two stay in
# sync if you edit the select line.
# =============================================================================

#PBS -N multicamp_rho1300
#PBS -l select=1:ncpus=3:mem=16gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o multicamp_rho1300.log

set -euo pipefail

# ---- 0. configuration (edit these two if you prefer hard-coding) -----------
# Directory that CONTAINS the campaign_cadex_rho1300v* subdirectories.
# If you submitted with `qsub -v RUN_DIR="$PWD" ...` this is picked up
# automatically; otherwise it falls back to PBS_O_WORKDIR (the dir you ran
# qsub from), which is usually what you want.
RUN_DIR="${RUN_DIR:-${PBS_O_WORKDIR:-$PWD}}"

# Path to the script itself. Adjust if it does not live under RUN_DIR.
SCRIPT="${SCRIPT:-${RUN_DIR}/multi_campaign_report.py}"

GLOB_PAT="${GLOB_PAT:-campaign_cadex_rho1300v*}"
OUT_DIR="${OUT_DIR:-./cohort_rho1300}"
CONDA_ENV="${CONDA_ENV:-brian_env}"   # reuse the project env; script only needs numpy/scipy/matplotlib

# JOBS defaults to the ncpus PBS actually granted, so the two never drift.
JOBS="${JOBS:-${PBS_NCPUS:-3}}"

# ---- 1. locale hygiene (avoid C-locale unicode surprises in child procs) ---
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
# Keep numpy/scipy from oversubscribing: each of the JOBS worker processes
# should stay single-threaded so total threads ~= JOBS, not JOBS * ncpus.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg   # belt-and-braces; the script already forces Agg

# ---- 2. activate the conda environment -------------------------------------
# davinci-1: source the base conda, then activate the project env.
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
else
    echo "WARNING: conda not found on PATH; relying on system python" >&2
fi

# ---- 3. go to the data directory (PBS starts you in \$HOME) -----------------
cd "${RUN_DIR}"

echo "=============================================================="
echo " job id      : ${PBS_JOBID:-<interactive>}"
echo " host        : $(hostname)"
echo " run dir     : ${RUN_DIR}"
echo " script      : ${SCRIPT}"
echo " python      : $(command -v python)  ($(python --version 2>&1))"
echo " ncpus/jobs  : PBS_NCPUS=${PBS_NCPUS:-?}  --jobs=${JOBS}"
echo " glob        : ${GLOB_PAT}"
echo " out dir     : ${OUT_DIR}"
echo " started     : $(date -Is)"
echo "=============================================================="

# ---- 4. run ----------------------------------------------------------------
if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "[SMOKE TEST MODE] running synthetic self-check"
    python "${SCRIPT}" --smoke-test
    rc=$?
else
    # Fail early with a clear message if no campaigns match, rather than
    # letting the python script raise deep in discover_campaigns().
    shopt -s nullglob
    matches=( ${GLOB_PAT}/ )
    shopt -u nullglob
    if (( ${#matches[@]} == 0 )); then
        echo "ERROR: no directories match '${GLOB_PAT}' under ${RUN_DIR}" >&2
        echo "       (did you submit from / point RUN_DIR at the parent dir?)" >&2
        exit 2
    fi
    echo "matched ${#matches[@]} campaign dir(s):"
    printf '  %s\n' "${matches[@]}"

    python "${SCRIPT}" . \
        --glob "${GLOB_PAT}" \
        --jobs "${JOBS}" \
        --out "${OUT_DIR}"
    rc=$?
fi

echo "=============================================================="
echo " finished    : $(date -Is)"
echo " exit code   : ${rc}"
if [[ "${SMOKE:-0}" != "1" ]]; then
    echo " report      : ${OUT_DIR}/multi_campaign_report.md"
    echo " summary     : ${OUT_DIR}/multi_campaign_summary.json"
fi
echo "=============================================================="
exit ${rc}
