#!/bin/bash
# Detached local smoke. Never requests system sleep or touches previous run folders.
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd -P)
cd "$REPO_ROOT"
if [ "${1:-}" = '--worker' ]; then
    OUTPUT=$2
    LOG=$3
    printf 'WORKER_STARTED PID=%s\n' "$$"
    printf '%s\n' "$$" > "$LOG.started"
    export MPLCONFIGDIR=/tmp/thai-dsr-mpl HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
    set +e
    if [ "$(uname -s)" = Darwin ]; then
        /usr/bin/caffeinate -i -m .venv/bin/python -u -m src.training.train_postnet \
          --device cpu --max-steps 500 --validation-interval 100 --checkpoint-interval 100 \
          --val-items 3 --vocoder universal --output-dir "$OUTPUT"
    else
        .venv/bin/python -u -m src.training.train_postnet \
          --device cpu --max-steps 500 --validation-interval 100 --checkpoint-interval 100 \
          --val-items 3 --vocoder universal --output-dir "$OUTPUT"
    fi
    RESULT=$?
    printf '%s\n' "$RESULT" > "$LOG.exit"
    printf 'EXIT_CODE=%s\n' "$RESULT"
    exit "$RESULT"
fi
[ "$#" -eq 0 ] || { printf 'Usage: bash %s\n' "$0"; exit 1; }
mkdir -p results/logs results/checkpoints/postnet
RUN_NAME="smoke500_cpu_$(date '+%Y%m%d_%H%M%S')_$$"
OUTPUT="results/checkpoints/postnet/$RUN_NAME"
LOG="$REPO_ROOT/results/logs/postnet_$RUN_NAME.log"
# Create a new session before exec so the terminal/CLI process group can exit safely.
nohup .venv/bin/python -c 'import os,sys; os.setsid(); os.execv("/bin/bash", ["/bin/bash"]+sys.argv[1:])' \
  "$SCRIPT_DIR/run_postnet_smoke_detached.sh" --worker "$OUTPUT" "$LOG" </dev/null >"$LOG" 2>&1 &
WORKER=$!
disown "$WORKER"
printf '%s\n' "$WORKER" > "$LOG.pid"
printf 'PID=%s\nOUTPUT=%s\nLOG=%s\n' "$WORKER" "$OUTPUT" "$LOG"
# Confirm the detached worker actually started before returning to the caller.
for attempt in {1..100}; do
    [ -s "$LOG.started" ] && exit 0
    kill -0 "$WORKER" 2>/dev/null || { cat "$LOG"; exit 1; }
    sleep 0.1
done
printf 'Worker startup unconfirmed; inspect %s\n' "$LOG" >&2
exit 1
