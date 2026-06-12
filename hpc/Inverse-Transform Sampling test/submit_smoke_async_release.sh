#!/bin/bash
#PBS -S /bin/bash
#PBS -N asn_smoke_binv
#PBS -q intel
#PBS -l select=1:ncpus=2,walltime=00:30:00
#PBS -k eo
##########################################################################
# submit_smoke_async_release.sh
#
# PBS wrapper for smoke_test_async_release.py.
# Compares the two asynchronous-release samplers:
#
#   FORMER   Bernoulli-sum loop  (current mechanism)
#   BINV     Inverse-transform   (proposed replacement)
#
# Both draw from the identical Binomial(n,p).  The test has two tiers:
#
#   Tier 1  unit: each sampler vs exact scipy.stats.binom over the (n,p)
#           grid derived from --x0 and --uar_max.
#
#   Tier 2  integration: Brian2 released-NT ensemble traces; verifies
#           ensemble statistics coincide and single traces differ.
#
# Outputs written to OUT_DIR (timestamped sub-directory of OUT_ROOT):
#   fig1_unit_sampler_pmf.png
#   fig2_example_traces.png
#   fig3_ensemble_traces.png
#   fig4_released_NT_distribution.png
#   unit_sampler_stats.csv
#   report.txt                  ← machine-parseable PASS/FAIL
#
# A symlink  smoke_async_latest -> OUT_DIR  is updated after every run.
# Downstream analysis scripts can hardcode that symlink path.
#
# No GPU required. No Brian2 compilation (TARGET=numpy default).
# Set TARGET=cython to exercise the near-deployment code path.
#
# ── Resource reasoning ──────────────────────────────────────────────────
#   ncpus=2 : numpy target is single-threaded, but the extra core avoids
#             busy-wait penalties and assists cython/gcc compilation.
#   walltime : full default run (Nsyn=500, T=5 s, M=200000, numpy)
#             typically finishes in 8-12 min; 30 min is a comfortable
#             margin. Raise WALLTIME below or pass -l walltime=HH:MM:SS
#             on the qsub command line for heavier TARGET=cython runs.
#
# ── Invocation ──────────────────────────────────────────────────────────
#   qsub submit_smoke_async_release.sh
#
#   # override individual knobs without editing this file:
#   TARGET=cython NSYN=1000 SIMTIME=10 qsub -v TARGET,NSYN,SIMTIME \
#       submit_smoke_async_release.sh
#
#   # fast sanity check:
#   QUICK=1 qsub -v QUICK submit_smoke_async_release.sh
#
#   # local (non-PBS) run:
#   bash submit_smoke_async_release.sh
##########################################################################

# ==========================================================================
# USER CONFIG  — edit here; everything else is derived automatically
# ==========================================================================

# Path to the smoke-test script (resolved relative to PBS_O_WORKDIR).
SCRIPT="${SCRIPT:-./smoke_test_async_release.py}"

# Root under which the timestamped output directory is created.
OUT_ROOT="${OUT_ROOT:-.}"

# Brian2 codegen target.  'numpy' is portable and needs no compilation.
# 'cython' exercises the near-deployment path and is faster but needs gcc.
TARGET="${TARGET:-numpy}"

# Ensemble size (number of independent synapses in the integration test).
# Larger values give tighter distribution comparisons.
NSYN="${NSYN:-500}"

# Biological simulation time [seconds].
SIMTIME="${SIMTIME:-5.0}"

# Presynaptic Poisson rate for the integration test [Hz].
RATE="${RATE:-15.0}"

# Recording dt for the StateMonitor [ms].
REC_DT="${REC_DT:-2.0}"

# Integration dt [ms].  Must match defaultclock.dt in the production run.
DT="${DT:-0.05}"

# Resource per vesicle x0 (dimensionless). Sets n_max = floor(1/x0).
# Replace with the actual x0 from get_Synparam() before trusting the grid.
X0="${X0:-0.02}"

# Maximum u_ar [Hz].  Sets p_max = uar_max * dt for the Tier-1 grid.
# Replace with the actual swept range from your parameter bounds.
UAR_MAX="${UAR_MAX:-2000.0}"

# Number of Monte-Carlo samples per (n,p) point in Tier 1.
# 200000 gives tight convergence across the full grid; use 40000 for --quick.
M="${M:-200000}"

# Master RNG seed (passed to both mechanisms for reproducibility).
SEED="${SEED:-1234}"

# Set QUICK=1 for a fast sanity pass (reduced Nsyn, simtime, M).
QUICK="${QUICK:-0}"

# ==========================================================================
# END USER CONFIG — do not edit below this line unless you know what you do
# ==========================================================================

cd "${PBS_O_WORKDIR:-$(pwd)}"

module load python 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate brian_env 2>/dev/null || true
fi

# ── Timestamped output directory + symlink ─────────────────────────────────
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_ROOT}/smoke_async_${STAMP}"
mkdir -p "${OUT_DIR}"

# Update the 'latest' symlink atomically (ln -sfn handles existing symlinks).
ln -sfn "$(realpath "${OUT_DIR}")" "${OUT_ROOT}/smoke_async_latest"

echo "============================================================"
echo "smoke_test_async_release  node=$(hostname)"
echo "PBS_JOBID : ${PBS_JOBID:-(local)}"
echo "target    : ${TARGET}"
echo "out dir   : ${OUT_DIR}"
echo "Nsyn=${NSYN}  T=${SIMTIME}s  rate=${RATE}Hz  dt=${DT}ms"
echo "x0=${X0}  uar_max=${UAR_MAX}Hz  M=${M}  seed=${SEED}"
echo "QUICK     : ${QUICK}"
echo "============================================================"

# ── Build argument list ────────────────────────────────────────────────────
ARGS=(
    --outdir  "${OUT_DIR}"
    --target  "${TARGET}"
    --Nsyn    "${NSYN}"
    --simtime "${SIMTIME}"
    --rate    "${RATE}"
    --rec-dt  "${REC_DT}"
    --dt      "${DT}"
    --x0      "${X0}"
    --uar-max "${UAR_MAX}"
    --M       "${M}"
    --seed    "${SEED}"
)

[ "${QUICK}" = "1" ] && ARGS+=(--quick)

# ── Run ────────────────────────────────────────────────────────────────────
python "${SCRIPT}" "${ARGS[@]}"
ec=$?

# ── Summary ────────────────────────────────────────────────────────────────
echo
echo "============================================================"
if [ "${ec}" -eq 0 ]; then
    echo "SMOKE TEST PASSED  ->  ${OUT_DIR}"
    echo "  figures : fig1_unit_sampler_pmf.png"
    echo "            fig2_example_traces.png"
    echo "            fig3_ensemble_traces.png"
    echo "            fig4_released_NT_distribution.png"
    echo "  data    : unit_sampler_stats.csv"
    echo "  verdict : report.txt  (OVERALL: PASS)"
else
    echo "SMOKE TEST FAILED (exit ${ec}) — inspect the log above."
    echo "  partial outputs may exist in ${OUT_DIR}"
    echo "  report.txt will contain the first failing tier."
fi
echo "============================================================"

if command -v conda >/dev/null 2>&1; then conda deactivate 2>/dev/null || true; fi
exit "${ec}"
