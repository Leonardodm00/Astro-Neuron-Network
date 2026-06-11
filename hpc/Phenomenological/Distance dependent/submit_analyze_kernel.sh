#!/bin/bash
#PBS -S /bin/bash
#PBS -N asn_kernel_analyze
#PBS -q intel
#PBS -l select=1:ncpus=2,walltime=00:20:00
#PBS -k eo
##########################################################################
# Analyzer for the kernel-form smoke test.
#
# Consumes the analyzer-ready output produced by submit_smoke_kernel.sh
# (culture_beta1.npz, culture_beta2.npz, meta.json in SMOKE_ROOT) and writes
# the comparison:
#
#   comparison.png              rasters, population rate, correlation-vs-distance,
#                               in-degree distributions
#   comparison_stats.json/.csv  scalar summary statistics, beta=1 vs beta=2
#
# Pure numpy / scipy / matplotlib -- no Brian2, no compile. Light + fast.
#
# Input directory resolution (first hit wins):
#   1. $IN_DIR environment variable
#   2. first positional arg ($1)
#   3. ./smoke_kernel_latest  (symlink updated by submit_smoke_kernel.sh)
#
# Run either way:
#     qsub submit_analyze_kernel.sh
#     IN_DIR=./smoke_kernel_123 qsub -v IN_DIR submit_analyze_kernel.sh
#     bash submit_analyze_kernel.sh ./smoke_kernel_123
##########################################################################

# --- USER CONFIG --------------------------------------------------------
ANALYZER="./analyze_kernel_comparison.py"
IN_DIR="${IN_DIR:-${1:-./smoke_kernel_latest}}"
# ------------------------------------------------------------------------

cd "${PBS_O_WORKDIR:-$(pwd)}"

module load python 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate brian_env 2>/dev/null || true
fi

echo "============================================================"
echo "Kernel-form analyzer   node=$(hostname)"
echo "input dir: ${IN_DIR}"
echo "============================================================"

if [ ! -f "${IN_DIR}/culture_beta1.npz" ] || [ ! -f "${IN_DIR}/culture_beta2.npz" ]; then
    echo "ERROR: ${IN_DIR} does not contain culture_beta1.npz / culture_beta2.npz."
    echo "       Run submit_smoke_kernel.sh first, or pass the correct dir:"
    echo "         IN_DIR=/path/to/smoke_kernel_XXXX bash submit_analyze_kernel.sh"
    exit 2
fi

python "$ANALYZER" --in_dir "${IN_DIR}"
ec=$?

echo
echo "============================================================"
if [ "$ec" -eq 0 ]; then
    echo "ANALYSIS DONE  ->  ${IN_DIR}/comparison.png , comparison_stats.{json,csv}"
else
    echo "ANALYSIS FAILED (exit ${ec}) - inspect the log above."
fi
echo "============================================================"

if command -v conda >/dev/null 2>&1; then conda deactivate 2>/dev/null || true; fi
exit "$ec"
