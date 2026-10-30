#!/bin/bash
# =============================================================================
# check_refractory_nan.sh -- do MEAMA and Giulia's simulations carry the
# refractory runaway (neurons whose V went NaN during t_ref, silent after)?
#
# Three checks per campaign:
#   1. CODE  the dV/dt line of the ASD_fun_BD_cpp.py the campaign ran with
#            (lib_dir of its job_args.json): "(unless refractory)" or not.
#   2. LOGS  PBS job logs in $HOME and in the folder above each campaign that
#            carry Brian2's warning "Neuron's variable 'V' has NaN". Brian2
#            prints it once per worker process: the counts are lower bounds.
#   3. DATA  refractory_scan.py on the saved spike trains (a random sample by
#            default): runs with neurons that fall silent for good right
#            after an early spike. Calibration: claude/Refractory_NaN_Audit.md.
#
# Usage (login node; needs numpy, so: conda activate brian_final first):
#   bash check_refractory_nan.sh                   # MEAMA n01 + Giulia's v1 v2 v4 v5, 1000 runs each
#   bash check_refractory_nan.sh --sample 300      # quicker estimate
#   bash check_refractory_nan.sh --sample 1000 CAMPAIGN_DIR [CAMPAIGN_DIR ...]
#   every run, per-run tables for filtering: qsub -q cpu submit_refractory_scan.sh
# Output: refractory_audit_<UTC>/ next to this script (audit.log, and per
#         campaign <campaign>_summary.txt, _summary.json, _runs.csv)
# Writes nothing outside that folder; reads campaigns and logs only.
# =============================================================================
set -u

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE=1000
WORKERS="${WORKERS:-4}"
OUT=""
CAMPS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --sample)  SAMPLE="${2:-}"; shift 2 ;;
        --workers) WORKERS="${2:-}"; shift 2 ;;
        --out)     OUT="${2:-}"; shift 2 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        -*)        echo "ERROR: unknown option $1" >&2; exit 2 ;;
        *)         CAMPS+=("$1"); shift ;;
    esac
done
if ! [ "$SAMPLE" -ge 0 ] 2>/dev/null || ! [ "$WORKERS" -ge 1 ] 2>/dev/null; then
    echo "ERROR: --sample must be an integer >= 0 and --workers an integer >= 1" >&2
    exit 2
fi
if [ ${#CAMPS[@]} -eq 0 ]; then
    CAMPS=(
        /davinci-1/home/ldellamea/ANN/Phenomenological/MEAMA/campaign_meama_rho1600_n01
        /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/campaign_cadex_hhgap_v1
        /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/campaign_cadex_hhgap_v2
        /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/campaign_cadex_hhgap_v4
        /davinci-1/home/ldellamea/ANN/Phenomenological/Main/Giulia_Astro/campaign_cadex_hhgap_v5
    )
fi
[ -n "$OUT" ] || OUT="${_HERE}/refractory_audit_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT" || { echo "ERROR: cannot create $OUT" >&2; exit 2; }

PY="$(command -v python || command -v python3 || true)"
if [ -z "$PY" ] || ! "$PY" -c "import numpy" >/dev/null 2>&1; then
    echo "ERROR: no python with numpy on PATH; run  conda activate brian_final  first." >&2
    exit 2
fi
if [ ! -f "${_HERE}/refractory_scan.py" ]; then
    echo "ERROR: refractory_scan.py must sit next to this script (${_HERE})." >&2
    exit 2
fi
WARN_V="Neuron's variable 'V' has NaN"
WARN_A="Astrocyte's variable"
CODE_FILE="$OUT/.code_status"      # both defined here: main() runs in the pipe's subshell
LOGLIST="$OUT/.logfiles"
: > "$CODE_FILE"

main() {
echo "refractory audit  $(date -u +%Y-%m-%dT%H:%M:%SZ)  host $(hostname)"
echo "python : $PY ($("$PY" -c 'import sys, numpy; print(sys.version.split()[0], "| numpy", numpy.__version__)'))"
echo "sample : ${SAMPLE} runs per campaign (0 = all)   workers: ${WORKERS}"
echo "output : ${OUT}"
echo

echo "=== 1. CODE: the dV/dt line each campaign ran with ==="
for c in "${CAMPS[@]}"; do
    c="${c%/}"
    name="$(basename "$c")"
    if [ ! -d "$c" ]; then
        echo "  ${name}: FOLDER NOT FOUND (${c})"
        printf '%s\t%s\n' "$name" "folder not found" >> "$CODE_FILE"
        continue
    fi
    ja="$(ls "$c"/sweep_*/job_args.json 2>/dev/null | head -1)"
    lib=""
    if [ -n "$ja" ]; then
        lib="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("lib_dir") or "")' "$ja" 2>/dev/null)"
    fi
    if [ -z "$lib" ] || [ ! -d "$lib" ]; then
        [ -n "$lib" ] && echo "  ${name}: lib_dir ${lib} from job_args.json is gone; looking next to the campaign"
        lib="$(dirname "$c")"
    fi
    asd="${lib}/ASD_fun_BD_cpp.py"
    if [ ! -f "$asd" ]; then
        echo "  ${name}: ASD_fun_BD_cpp.py NOT FOUND in ${lib}"
        printf '%s\t%s\n' "$name" "ASD file not found" >> "$CODE_FILE"
        continue
    fi
    n_dv="$(grep -Ec '^[[:space:]]*dV/dt[[:space:]]*=' "$asd")"
    dv="$(grep -E '^[[:space:]]*dV/dt[[:space:]]*=' "$asd" | head -1 | sed 's/^[[:space:]]*//')"
    if [ "$n_dv" = "0" ]; then
        st="NO dV/dt LINE (another neuron model? check by hand)"
    elif printf '%s' "$dv" | grep -q 'unless refractory'; then
        st="CLAMPED (V held during t_ref: not exposed)"
    else
        st="NOT CLAMPED (exposed)"
    fi
    echo "  ${name}: ${st}"
    echo "      file  : ${asd}  (${n_dv} dV/dt line(s))"
    echo "      dV/dt : ${dv}"
    echo "      t_ref : $(grep -E "'t_ref'[[:space:]]*:" "$asd" | head -1 | sed 's/^[[:space:]]*//')"
    if [ -n "$ja" ] && [ "$asd" -nt "$ja" ]; then
        echo "      WARNING: the file was modified ($(date -r "$asd" '+%Y-%m-%d %H:%M')) after the campaign's first"
        echo "               task started; the code that ran may differ from what is shown"
    fi
    printf '%s\t%s\n' "$name" "$st" >> "$CODE_FILE"
done
echo

echo "=== 2. LOGS: PBS job logs carrying Brian2's NaN warnings (lower bounds) ==="
: > "$LOGLIST"
seen=" "
for d in "$HOME" $(for c in "${CAMPS[@]}"; do dirname "${c%/}"; done); do
    [ -d "$d" ] || continue
    case "$seen" in *" $d "*) continue ;; esac
    seen="${seen}${d} "
    find "$d" -maxdepth 1 -type f -name '*.[eo][0-9]*' 2>/dev/null >> "$LOGLIST"
done
echo "  searched: ${seen}"
if [ ! -s "$LOGLIST" ]; then
    echo "  no PBS job logs (*.e<N>, *.o<N>) found there"
else
    printf '  %-28s %8s %16s %20s\n' "job name" "logs" "with V NaN" "with astrocyte NaN"
    for p in $(sed 's|.*/||; s|\.[eo][0-9].*||' "$LOGLIST" | sort -u); do
        awk -v p="$p" '{ n = $0; sub(/.*\//, "", n); sub(/\.[eo][0-9].*/, "", n); if (n == p) print }' \
            "$LOGLIST" > "$OUT/.logs_one"
        n="$(grep -c . "$OUT/.logs_one")"
        nv="$(tr '\n' '\0' < "$OUT/.logs_one" | xargs -0 -r grep -lF "$WARN_V" 2>/dev/null | wc -l)"
        na="$(tr '\n' '\0' < "$OUT/.logs_one" | xargs -0 -r grep -lF "$WARN_A" 2>/dev/null | wc -l)"
        printf '  %-28s %8d %16d %20d\n' "$p" "$n" "$nv" "$na"
    done
    echo "  (MEAMA jobs are named meama_campaign; the rho1300 / Giulia worker asn_campaign; LindaColloDUP lcdup_campaign)"
fi
rm -f "$OUT/.logs_one"
echo

echo "=== 3. DATA: spike-train screen (refractory_scan.py) ==="
"$PY" "${_HERE}/refractory_scan.py" "${CAMPS[@]}" --out "$OUT" --sample "$SAMPLE" --workers "$WORKERS"
echo

echo "=== VERDICT ==="
"$PY" - "$OUT" "$CODE_FILE" <<'PYEOF'
import json, os, sys
out, code_file = sys.argv[1], sys.argv[2]
code = dict(l.rstrip('\n').split('\t', 1) for l in open(code_file) if '\t' in l)
for name, st in code.items():
    p = os.path.join(out, '%s_summary.json' % name)
    if os.path.exists(p):
        s = json.load(open(p))
        data = ('%.1f %% of %d scanned runs with >= 1 neuron silenced early [95%% CI %.1f, %.1f]; '
                '%.1f %% of runs fully silenced early; %.1f %% of neurons silenced early'
                % (s['pct_runs_early'], s['n_ok'], s['ci95_early'][0], s['ci95_early'][1],
                   s['pct_runs_net_silent_early'], s['pct_neurons_early']))
    else:
        data = 'no data scan'
    print('  %s\n      code: %s\n      data: %s' % (name, st, data))
print('  logs: section 2 (counts per job name, lower bounds)')
PYEOF
}

main 2>&1 | tee "$OUT/audit.log"
rm -f "$CODE_FILE" "$LOGLIST"
echo "saved: $OUT/audit.log"
