#!/bin/bash
#PBS -S /bin/bash
#PBS -N mea_array
#PBS -k eo
#PBS -l walltime=06:00:00
##########################################################################
# Array WORKER for MEA post-processing across many campaigns/sweeps.
#
# DO NOT submit this directly -- submit it via launch_mea_array.sh, which
# builds the manifest, computes the array range, and calls qsub -J for you.
#
# Each array member reads exactly ONE line from MANIFEST (its own index + 1,
# manifests are 1-indexed by `sed`, PBS_ARRAY_INDEX is 0-indexed) and runs
# process_campaign.py on just that campaign/sweep directory. This is
# POST-processing (numpy + scipy only, no Brian2, no compilation), so it
# needs far less walltime than the original simulation sweep.
#
# REQUIRED -v variables (set by launch_mea_array.sh):
#     MANIFEST   path to the manifest built by build_mea_manifest.py
#                (tab-separated: campaign_dir <TAB> out_dir, one per line)
#     LIB        path to a pre-built eap_library.npz (build this ONCE before
#                submitting the array -- see launch_mea_array.sh -- so that
#                dozens/hundreds of array members don't each pay the ~30s
#                HH-integration cost separately)
# OPTIONAL -v variables:
#     EXTRA_ARGS extra flags passed through to process_campaign.py verbatim,
#                e.g. "--plots --limit_iters 5" for a partial/inspectable run
##########################################################################

set -euo pipefail

if [ -z "${MANIFEST:-}" ] || [ -z "${LIB:-}" ]; then
    echo "ERROR: submit via launch_mea_array.sh (MANIFEST and LIB must be" >&2
    echo "       passed with qsub -v). Refusing to run un-configured." >&2
    exit 2
fi
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: manifest not found: $MANIFEST" >&2
    exit 2
fi

IDX="${PBS_ARRAY_INDEX:-0}"           # 0 if run as a non-array single task
LINE="$((IDX + 1))"
ROW="$(sed -n "${LINE}p" "$MANIFEST")"
if [ -z "$ROW" ]; then
    echo "ERROR: manifest has no line $LINE (index $IDX) -- array range and" >&2
    echo "       manifest length must match. Manifest: $MANIFEST" >&2
    exit 2
fi

CAMPAIGN="$(printf '%s' "$ROW" | cut -f1)"
OUT="$(printf '%s' "$ROW" | cut -f2)"
if [ -z "$CAMPAIGN" ] || [ -z "$OUT" ]; then
    echo "ERROR: malformed manifest line $LINE: '$ROW'" >&2
    exit 2
fi

cd "${PBS_O_WORKDIR:-.}"

NCPUS="${PBS_NCPUS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "[mea-array] index    : $IDX (manifest line $LINE)"
echo "[mea-array] campaign : $CAMPAIGN"
echo "[mea-array] out      : $OUT"
echo "[mea-array] library  : $LIB"
echo "[mea-array] workers  : $NCPUS"
echo "[mea-array] extra    : ${EXTRA_ARGS:-<none>}"

python3 process_campaign.py \
    --campaign "$CAMPAIGN" \
    --out "$OUT" \
    --library "$LIB" \
    --workers "$NCPUS" \
    $EXTRA_ARGS

echo "[mea-array] index $IDX done."
