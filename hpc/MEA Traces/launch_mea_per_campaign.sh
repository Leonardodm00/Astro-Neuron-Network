#!/bin/bash
##########################################################################
# launch_mea_per_campaign.sh -- submit ONE SEPARATE PBS job per campaign,
# as opposed to launch_mea_array.sh, which flattens every campaign's work
# units into a SINGLE combined array job.
#
# This is a THIN LOOP around launch_mea_array.sh: for each campaign it calls
# that script once, with --campaign-root pointing at just that one campaign.
# It does not reimplement manifest-building, array-vs-plain-job fallback, or
# BLAS pinning -- all of that stays in the already-tested launch_mea_array.sh
# / submit_mea_array.sh / build_mea_manifest.py. Each per-campaign job is
# itself a PBS array if that campaign has more than one sweep_*_task*
# directory (or a plain job if it has exactly one), exactly as
# launch_mea_array.sh already handles.
#
# WHEN TO USE THIS INSTEAD OF launch_mea_array.sh
#   - You want an independent job ID per campaign, so you can monitor,
#     cancel, or re-run one campaign without touching the others.
#   - You want a per-campaign concurrency throttle (each campaign gets its
#     own --concurrency budget, rather than sharing one across everything).
#   launch_mea_array.sh remains the right tool when you want ONE combined
#   job across everything instead.
#
# USAGE
#   # auto-discover every campaign_* directory directly under --parent:
#   ./launch_mea_per_campaign.sh \
#       --parent /davinci-1/home/USER/ANN/Phenomenological/Main \
#       --out-root /path/to/mea_out
#
#   # or name specific campaigns explicitly (repeatable):
#   ./launch_mea_per_campaign.sh \
#       --campaign-root /path/campaign_cadex_rho1300v1 \
#       --campaign-root /path/campaign_cadex_rho1300v2 \
#       --out-root /path/to/mea_out
#
# All other options match launch_mea_array.sh exactly (they are forwarded to
# every per-campaign call): --lib --queue --ncpus --walltime --concurrency
# --skip-done --extra-args --dry-run. Run with --help for the full list, or
# see launch_mea_array.sh --help -- the semantics are identical, just applied
# once per campaign instead of once overall.
##########################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT=""
CAMPAIGN_ROOTS=()
OUT_ROOT=""
LIB="./eap_library.npz"
QUEUE="cpu"
NCPUS=48
WALLTIME="06:00:00"
CONCURRENCY=20
SKIP_DONE=""
EXTRA_ARGS=""
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --parent) PARENT="$2"; shift 2 ;;
        --campaign-root) CAMPAIGN_ROOTS+=("$2"); shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        --lib) LIB="$2"; shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --ncpus) NCPUS="$2"; shift 2 ;;
        --walltime) WALLTIME="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --skip-done) SKIP_DONE="--skip-done"; shift ;;
        --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$OUT_ROOT" ]; then
    echo "ERROR: --out-root is required" >&2
    exit 2
fi

# --- resolve the campaign list --------------------------------------------
if [ -n "$PARENT" ]; then
    if [ ! -d "$PARENT" ]; then
        echo "ERROR: --parent is not a directory: $PARENT" >&2
        exit 2
    fi
    for d in "$PARENT"/campaign_*/; do
        [ -d "$d" ] && CAMPAIGN_ROOTS+=("${d%/}")
    done
fi
if [ ${#CAMPAIGN_ROOTS[@]} -eq 0 ]; then
    echo "ERROR: no campaigns found -- give --parent DIR (containing " >&2
    echo "       campaign_*/ subdirectories) or one or more --campaign-root" >&2
    exit 2
fi

echo "[per-campaign] ${#CAMPAIGN_ROOTS[@]} campaign(s) to submit:"
for c in "${CAMPAIGN_ROOTS[@]}"; do echo "    $(basename "$c")"; done
echo ""

# --- build the eap library ONCE up front, shared by every campaign's job -
# (launch_mea_array.sh would otherwise build it independently on its first
# invocation below anyway, but doing it once here up front avoids the first
# campaign's job submission being slower than the rest for no good reason.)
if [ ! -f "$LIB" ]; then
    echo "[per-campaign] library not found, building once -> $LIB"
    python3 "${SCRIPT_DIR}/eap_template_library.py" --out "$LIB"
fi

# --- one launch_mea_array.sh call per campaign ----------------------------
SUMMARY=()
for CROOT in "${CAMPAIGN_ROOTS[@]}"; do
    NAME="$(basename "$CROOT")"
    # a per-campaign manifest filename, so concurrent/rapid-fire submissions
    # in this loop can never collide on the same default timestamped name.
    MANIFEST="./mea_manifest_${NAME}_$(date +%Y%m%d_%H%M%S)_$$.tsv"

    ARGS=(--out-root "$OUT_ROOT" --campaign-root "$CROOT" --lib "$LIB"
          --queue "$QUEUE" --ncpus "$NCPUS" --walltime "$WALLTIME"
          --concurrency "$CONCURRENCY" --manifest "$MANIFEST")
    [ -n "$SKIP_DONE" ] && ARGS+=(--skip-done)
    [ -n "$EXTRA_ARGS" ] && ARGS+=(--extra-args "$EXTRA_ARGS")
    [ "$DRY_RUN" -eq 1 ] && ARGS+=(--dry-run)

    echo "=== $NAME ==="
    LOG="$(mktemp)"
    "${SCRIPT_DIR}/launch_mea_array.sh" "${ARGS[@]}" | tee "$LOG"
    JID="$(grep '^\[launch\] submitted:' "$LOG" | sed 's/.*submitted: //' || true)"
    SUMMARY+=("$NAME	${JID:-<dry-run, not submitted>}")
    rm -f "$LOG"
    echo ""
done

echo "=== summary ==="
for row in "${SUMMARY[@]}"; do
    printf '%-32s %s\n' "${row%%$'\t'*}" "${row#*$'\t'}"
done
if [ "$DRY_RUN" -eq 0 ]; then
    echo ""
    echo "monitor all of them with: qstat -u \$USER"
fi
