#!/bin/bash
# macOS / Bash 3.2. Run from any directory; do not source this file.
# -i prevents system idle sleep; -m prevents disk idle sleep (not display sleep).
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
SCRIPT="$SCRIPT_DIR/$(basename "$0")"
cd "$REPO_ROOT"
LOCK="$REPO_ROOT/results/logs/.joint_overnight.lock"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

check_running() {
    local matches status
    if matches=$(/usr/bin/pgrep -fl '[s]rc[.]training[.](finetune_hifigan|joint_finetune)'); then
        die "Fine-tuning is already running; refusing a second job: $matches"
    else
        status=$?
        [ "$status" -eq 1 ] || die "Cannot inspect processes (pgrep=$status)."
    fi
}

if [ "${1:-}" = "--exec-training" ]; then
    # On macOS caffeinate execs its utility in the original PID and creates
    # a separate assertion-holding child. Record that child before exec Python.
    LOG=$2
    ASSERTION_PID=$(/usr/bin/pgrep -P "$$" -x caffeinate) ||
        die "Cannot identify this job's caffeinate assertion process."
    printf '%s\n' "$ASSERTION_PID" > "$LOG.caffeinate.pid"
    printf '%s\n' "$$" > "$LOG.pid"
    export MPLCONFIGDIR=/tmp/thai-dsr-mpl PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
    exec .venv/bin/python -m src.training.joint_finetune \
        --device mps \
        --content-device cpu \
        --max-steps "${JOINT_MAX_STEPS:-2000}" \
        --vocoder-init "${JOINT_VOCODER_INIT:-thai}" \
        --content-weight "${JOINT_CONTENT_WEIGHT:-2}" \
        --manifest data/manifest_w5.csv \
        --splits data/splits_w5.json \
        --checkpoint-interval 500 \
        --validation-interval 250 \
        --val-items 3
fi

if [ "${1:-}" = "--worker" ]; then
    LOG=$2
    trap 'rmdir "$LOCK"' EXIT
    check_running
    printf 'STARTED=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    /usr/bin/caffeinate -i -m /bin/bash "$SCRIPT" --exec-training "$LOG" &
    TRAINING_PID=$!
    # Poll the real process, then reap it to obtain its actual exit status.
    while kill -0 "$TRAINING_PID" 2>/dev/null; do sleep 2; done
    EXIT_CODE=0
    wait "$TRAINING_PID" || EXIT_CODE=$?
    printf 'TRAINING_FINISHED=%s EXIT_CODE=%s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$EXIT_CODE"
    # Also leave a standalone exit code as the final log line, even if sleep fails.
    trap 'printf "EXIT_CODE=%s\n" "$EXIT_CODE"; rmdir "$LOCK"' EXIT
    [ -s "$LOG.caffeinate.pid" ] && [ -s "$LOG.pid" ] ||
        die "Training startup could not be confirmed; automatic sleep cancelled."
    ASSERTION_PID=$(< "$LOG.caffeinate.pid")
    while kill -0 "$ASSERTION_PID" 2>/dev/null; do sleep 1; done
    # Python has exited, including synchronous checkpoint writes, and this
    # job's assertion process has exited. Flush filesystem buffers before sleep.
    printf 'EXIT_CODE=%s\n' "$EXIT_CODE"
    /bin/sync
    check_running
    printf 'Requesting system sleep after training exit (EXIT_CODE=%s).\n' "$EXIT_CODE"
    /usr/bin/pmset sleepnow || die "pmset sleepnow failed; training has already finished."
    exit "$EXIT_CODE"
fi

[ "$#" -eq 0 ] || die "Usage: $0"
[ "$(uname -s)" = Darwin ] || die "This script requires macOS."
check_running
[ -x .venv/bin/python ] || die "Missing executable .venv/bin/python"
for path in data/manifest_w5.csv data/splits_w5.json \
    results/audio_compare/w6_vocoder_ft/provenance.json \
    results/checkpoints/hifigan_thai/config.json \
    results/checkpoints/hifigan_thai/g_00010000 \
    results/checkpoints/hifigan_thai/do_latest \
    src/training/joint_finetune.py; do
    [ -r "$path" ] || die "Missing or unreadable input: $path"
done
# Complete joint snapshots are about 1.2 GiB each. Keep a safety margin.
.venv/bin/python -c 'import os,shutil; n=int(os.environ.get("JOINT_MAX_STEPS", "2000")); assert n>0; needed=((n+499)//500)*1400*1024**2+2*1024**3; free=shutil.disk_usage(".").free; assert free>needed, f"Need {needed/1024**3:.1f} GiB free; have {free/1024**3:.1f} GiB"' || die "Disk-space preflight failed."
mkdir -p results/logs results/checkpoints/joint_finetune /tmp/thai-dsr-mpl
# Atomic lock closes the race between two simultaneous invocations of this script.
mkdir "$LOCK" 2>/dev/null ||
    die "Overnight launcher lock exists: $LOCK (if stale, verify no job is running before removing it)."
trap 'rmdir "$LOCK"' EXIT
check_running
# mktemp adds a unique suffix even if two launches happen within the same second.
LOG=$(mktemp "$REPO_ROOT/results/logs/joint_overnight_$(date '+%Y%m%d_%H%M%S')_XXXXXX")
mv "$LOG" "$LOG.log"
LOG="$LOG.log"
nohup /bin/bash "$SCRIPT" --worker "$LOG" </dev/null >>"$LOG" 2>&1 &
WORKER_PID=$!
disown "$WORKER_PID"
# The detached worker now owns the lock and survives terminal closure.
trap - EXIT
while [ ! -s "$LOG.pid" ]; do
    kill -0 "$WORKER_PID" 2>/dev/null || die "Startup failed; inspect $LOG"
    sleep 0.1
done
TRAINING_PID=$(< "$LOG.pid")
printf 'STARTED: training PID=%s | log=%s\n' "$TRAINING_PID" "$LOG"
