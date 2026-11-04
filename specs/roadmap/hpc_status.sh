#!/bin/bash
# hpc_status.sh -- mark the roadmap stage of a PBS/SLURM job on the
# hpc-status branch, so that specs/roadmap/roadmap.py can colour the stage
# "running", "done" or "failed" without reaching the cluster.
#
# Source it from a job script AFTER the environment block, then:
#
#   export ROADMAP_STAGE=I14                  # implementation stage id (roadmap yaml)
#   export ROADMAP_N_TASKS="${DIAM_N_TASKS:-1}"   # array size; 1 for a plain job
#   . "$REPO/specs/roadmap/hpc_status.sh"
#   roadmap_job_start
#   trap 'roadmap_job_end $?' EXIT
#
# The first task of the array to start writes runs/<STAGE>/<jobid>.json with
# state "running" and pushes it; the last task to end rewrites it as "done",
# or "failed" when any task exited non-zero, and pushes again. Two pushes
# per array whatever its size. The counters live in <repo>/.roadmap/state/
# (gitignored; on the shared filesystem) under flock. A task killed by the
# scheduler (walltime, memory) never runs its EXIT trap, so the "done"
# marker never comes: roadmap.py reports a running marker older than
# stale_after_hours (48 by default) as "failing".
#
# roadmap_mark done|failed|running   writes a marker by hand (interactive
#   session, a merge step done at the prompt); needs ROADMAP_STAGE.
#
# Nothing here can fail the job: every function returns 0 and reports to
# stderr with a [roadmap] prefix. ROADMAP_DISABLE=1 turns everything off.
# Needs on the cluster: git with push rights to origin, flock (util-linux),
# date. The branch is created on first use if origin does not have it.
#
# Knobs (all optional): ROADMAP_BRANCH (default hpc-status), ROADMAP_REPO
# (default: the repository holding this file), ROADMAP_SCRIPT (the job
# script's path, recorded in the marker; default: PBS/SLURM job name).

ROADMAP_BRANCH="${ROADMAP_BRANCH:-hpc-status}"
if [ -z "${ROADMAP_REPO:-}" ]; then
    ROADMAP_REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"
fi
ROADMAP_GIT="git -c user.name=roadmap -c user.email=roadmap@hpc"

roadmap__log() { echo "[roadmap] $*" >&2; }

roadmap__job_id() {
    local id
    if [ -n "${PBS_JOBID:-}" ]; then
        id="$(printf '%s' "$PBS_JOBID" | sed 's/\[[^]]*\]//')"
    elif [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
        id="$SLURM_ARRAY_JOB_ID"
    elif [ -n "${SLURM_JOB_ID:-}" ]; then
        id="$SLURM_JOB_ID"
    else
        id="local-$(hostname -s 2>/dev/null || echo host)-$$"
    fi
    printf '%s' "$id" | tr -c 'A-Za-z0-9._-' '_'
}

roadmap__job_name() {
    printf '%s' "${PBS_JOBNAME:-${SLURM_JOB_NAME:-${ROADMAP_SCRIPT:-unknown}}}"
}

roadmap__task_index() {
    printf '%s' "${PBS_ARRAY_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
}

roadmap__now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

roadmap__json_str() {
    # escape backslashes and double quotes; the values are plain tokens
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

roadmap__ensure_worktree() {
    # Set up <repo>/.roadmap/wt as a worktree of the hpc-status branch.
    local wt="$ROADMAP_REPO/.roadmap/wt" br="$ROADMAP_BRANCH" empty commit
    if [ -e "$wt/.git" ]; then
        return 0
    fi
    mkdir -p "$ROADMAP_REPO/.roadmap"
    git -C "$ROADMAP_REPO" fetch -q origin "+refs/heads/$br:refs/remotes/origin/$br" 2>/dev/null
    if git -C "$ROADMAP_REPO" rev-parse -q --verify "refs/remotes/origin/$br" >/dev/null 2>&1; then
        if ! git -C "$ROADMAP_REPO" rev-parse -q --verify "refs/heads/$br" >/dev/null 2>&1; then
            git -C "$ROADMAP_REPO" branch -q --track "$br" "origin/$br" || return 1
        fi
        git -C "$ROADMAP_REPO" worktree add -q "$wt" "$br" || return 1
        git -C "$wt" reset -q --hard "origin/$br" || return 1
    else
        # no branch on origin yet: an orphan commit with an empty tree
        empty="$(git -C "$ROADMAP_REPO" mktree </dev/null)" || return 1
        commit="$($ROADMAP_GIT -C "$ROADMAP_REPO" commit-tree "$empty" -m "roadmap: hpc-status branch")" || return 1
        git -C "$ROADMAP_REPO" branch -q "$br" "$commit" || return 1
        git -C "$ROADMAP_REPO" worktree add -q "$wt" "$br" || return 1
        git -C "$wt" push -q -u origin "$br" || return 1
    fi
    git -C "$wt" branch -q --set-upstream-to="origin/$br" "$br" 2>/dev/null
    return 0
}

roadmap__write_marker() {
    # roadmap__write_marker STATE JOBID JOBNAME NTASKS NDONE NFAILED LASTRC STARTED
    local state="$1" jobid="$2" jobname="$3" ntasks="$4" ndone="$5" nfailed="$6" lastrc="$7" started="$8"
    local wt="$ROADMAP_REPO/.roadmap/wt" br="$ROADMAP_BRANCH" ended="" commit branch i
    local dir="$wt/runs/$ROADMAP_STAGE" file
    file="$dir/$jobid.json"
    commit="$(git -C "$ROADMAP_REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    branch="$(git -C "$ROADMAP_REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    if [ "$state" != "running" ]; then ended="$(roadmap__now)"; fi
    mkdir -p "$dir"
    {
        printf '{\n'
        printf '  "stage": "%s",\n' "$(roadmap__json_str "$ROADMAP_STAGE")"
        printf '  "job_id": "%s",\n' "$(roadmap__json_str "$jobid")"
        printf '  "job_name": "%s",\n' "$(roadmap__json_str "$jobname")"
        printf '  "script": "%s",\n' "$(roadmap__json_str "${ROADMAP_SCRIPT:-}")"
        printf '  "state": "%s",\n' "$state"
        printf '  "started": "%s",\n' "$started"
        printf '  "ended": "%s",\n' "$ended"
        printf '  "n_tasks": %s,\n' "$ntasks"
        printf '  "n_done": %s,\n' "$ndone"
        printf '  "n_failed": %s,\n' "$nfailed"
        printf '  "last_exit": %s,\n' "$lastrc"
        printf '  "commit": "%s",\n' "$commit"
        printf '  "branch": "%s",\n' "$(roadmap__json_str "$branch")"
        printf '  "host": "%s",\n' "$(hostname -s 2>/dev/null || echo unknown)"
        printf '  "user": "%s"\n' "$(roadmap__json_str "${USER:-unknown}")"
        printf '}\n'
    } > "$file"
    git -C "$wt" add -A runs || return 1
    if git -C "$wt" diff --cached --quiet; then
        return 0
    fi
    $ROADMAP_GIT -C "$wt" commit -q -m "roadmap: $ROADMAP_STAGE $jobid $state" || return 1
    for i in 1 2 3 4; do
        if $ROADMAP_GIT -C "$wt" pull -q --rebase origin "$br" >/dev/null 2>&1 \
           && git -C "$wt" push -q origin "$br" >/dev/null 2>&1; then
            roadmap__log "$ROADMAP_STAGE $jobid -> $state (pushed to $br)"
            return 0
        fi
        sleep $((i * 5))
    done
    roadmap__log "could not push $ROADMAP_STAGE $jobid -> $state after 4 attempts; the commit stays in $wt"
    return 1
}

roadmap__counter_add() {
    # roadmap__counter_add FILE INCREMENT -> prints the new value
    local n
    n="$(cat "$1" 2>/dev/null || echo 0)"
    case "$n" in ''|*[!0-9]*) n=0 ;; esac
    n=$((n + $2))
    echo "$n" > "$1"
    echo "$n"
}

roadmap__check() {
    if [ "${ROADMAP_DISABLE:-0}" = "1" ]; then return 1; fi
    if [ -z "${ROADMAP_STAGE:-}" ]; then roadmap__log "ROADMAP_STAGE not set: no marker written"; return 1; fi
    case "$ROADMAP_STAGE" in *[!A-Za-z0-9._-]*) roadmap__log "ROADMAP_STAGE '$ROADMAP_STAGE' has characters outside A-Za-z0-9._-"; return 1 ;; esac
    if [ -z "$ROADMAP_REPO" ] || ! git -C "$ROADMAP_REPO" rev-parse --show-toplevel >/dev/null 2>&1; then
        roadmap__log "repository not found (ROADMAP_REPO='$ROADMAP_REPO'): no marker written"; return 1
    fi
    if ! command -v flock >/dev/null 2>&1; then roadmap__log "flock not found: no marker written"; return 1; fi
    return 0
}

roadmap_job_start() {
    roadmap__check || return 0
    (
        set +e
        local jobid ntasks state lock n started
        jobid="$(roadmap__job_id)"
        ntasks="${ROADMAP_N_TASKS:-1}"
        case "$ntasks" in ''|*[!0-9]*) ntasks=1 ;; esac
        state="$ROADMAP_REPO/.roadmap/state/$jobid"
        lock="$ROADMAP_REPO/.roadmap/lock"
        mkdir -p "$state"
        exec 9>"$lock"
        if ! flock -w 120 9; then roadmap__log "lock timeout at start"; exit 0; fi
        n="$(roadmap__counter_add "$state/n_started" 1)"
        if [ "$n" -eq 1 ]; then
            started="$(roadmap__now)"
            echo "$started" > "$state/started"
            echo "$ntasks" > "$state/n_tasks"
            roadmap__ensure_worktree && roadmap__write_marker running "$jobid" "$(roadmap__job_name)" "$ntasks" 0 0 0 "$started"
        fi
        exit 0
    ) || true
    return 0
}

roadmap_job_end() {
    local rc="${1:-0}"
    roadmap__check || return 0
    (
        set +e
        local jobid ntasks state lock d f started final
        jobid="$(roadmap__job_id)"
        state="$ROADMAP_REPO/.roadmap/state/$jobid"
        lock="$ROADMAP_REPO/.roadmap/lock"
        mkdir -p "$state"
        exec 9>"$lock"
        if ! flock -w 120 9; then roadmap__log "lock timeout at end"; exit 0; fi
        ntasks="$(cat "$state/n_tasks" 2>/dev/null || echo "${ROADMAP_N_TASKS:-1}")"
        case "$ntasks" in ''|*[!0-9]*) ntasks=1 ;; esac
        d="$(roadmap__counter_add "$state/n_done" 1)"
        f="$(cat "$state/n_failed" 2>/dev/null || echo 0)"
        case "$f" in ''|*[!0-9]*) f=0 ;; esac
        if [ "$rc" != "0" ]; then f="$(roadmap__counter_add "$state/n_failed" 1)"; fi
        echo "$rc" > "$state/last_exit"
        if [ "$d" -ge "$ntasks" ]; then
            started="$(cat "$state/started" 2>/dev/null || roadmap__now)"
            final=done
            if [ "$f" -gt 0 ]; then final=failed; fi
            roadmap__ensure_worktree && roadmap__write_marker "$final" "$jobid" "$(roadmap__job_name)" "$ntasks" "$d" "$f" "$rc" "$started"
        fi
        exit 0
    ) || true
    return 0
}

roadmap_mark() {
    # roadmap_mark running|done|failed  (by hand; one marker, pushed at once)
    local st="${1:-}" jobid
    case "$st" in running|done|failed) ;; *) roadmap__log "usage: roadmap_mark running|done|failed"; return 0 ;; esac
    roadmap__check || return 0
    (
        set +e
        jobid="manual-$(date -u +%Y%m%dT%H%M%S)"
        exec 9>"$ROADMAP_REPO/.roadmap/lock"
        mkdir -p "$ROADMAP_REPO/.roadmap"
        flock -w 120 9 || exit 0
        roadmap__ensure_worktree && roadmap__write_marker "$st" "$jobid" "${ROADMAP_SCRIPT:-manual}" 1 1 "$([ "$st" = failed ] && echo 1 || echo 0)" 0 "$(roadmap__now)"
        exit 0
    ) || true
    return 0
}
