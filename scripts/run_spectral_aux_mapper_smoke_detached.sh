#!/bin/bash
# Fixed 300-step smoke; a new output/log on every launch.
set -eu
cd "$(dirname "$0")/.."
if [ "${1:-}" = --worker ]; then
    OUT=$2
    LOG=$3
    export MPLCONFIGDIR=/tmp/thai-dsr-mpl HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
    printf '%s\n' "$$" > "$LOG.started"
    set +e
    /usr/bin/caffeinate -i .venv/bin/python -u -m src.training.train_mapper_spectral_aux \
      --device cpu --encoder-device cpu --threads 4 --max-steps 300 \
      --validation-interval 100 --val-items 8 --output-dir "$OUT"
    RESULT=$?
    printf '%s\n' "$RESULT" > "$LOG.exit"
    exit "$RESULT"
fi
RUN="smoke300_$(date '+%Y%m%d_%H%M%S')_$$"
OUT="results/checkpoints/spectral_aux_mapper/$RUN"
LOG="$PWD/results/logs/spectral_aux_$RUN.log"
mkdir -p results/logs
nohup .venv/bin/python -c 'import os,sys; os.setsid(); os.execv("/bin/bash", ["/bin/bash"]+sys.argv[1:])' \
  "$PWD/scripts/run_spectral_aux_mapper_smoke_detached.sh" --worker "$OUT" "$LOG" </dev/null >"$LOG" 2>&1 &
WORKER=$!
disown "$WORKER"
printf '%s\n' "$WORKER" > "$LOG.pid"
printf 'PID=%s\nOUTPUT=%s\nLOG=%s\n' "$WORKER" "$OUT" "$LOG"
# Wait for the worker to survive setsid/exec before the launching shell exits.
for ((i=0; i<100; i++)); do
    if [ -f "$LOG.started" ]; then
        printf 'DETACHED_WORKER_STARTED=%s\n' "$(cat "$LOG.started")"
        exit 0
    fi
    sleep 0.1
done
printf 'Worker failed to acknowledge startup; inspect %s\n' "$LOG" >&2
exit 1
