#!/bin/bash
# Steps 1-2 of the expanded-dataset experiment, detached:
#   1. stream CV Thai clips 200..TARGET-1, distort (severe), cache layer-9
#      embeddings, build manifest + W5-preserving split
#   2. train a fresh L1-only layer-9 mapper to early stop, then compare it with
#      mapper_layer9.pt under the W5 smoke-validation protocol.
# Every stage is skipped when its output already exists, so relaunching resumes.
set -eu
cd "$(dirname "$0")/.."
TARGET=${TARGET:-2000}
PATIENCE=${PATIENCE:-6}

free_gib() { df -k . | awk 'NR==2 {printf "%.2f", $4/1048576}'; }
wait_for_disk() {  # $1 = GiB needed; polls every 5 min for up to 12 h
    for ((i=0; i<144; i++)); do
        if awk -v f="$(free_gib)" -v n="$1" 'BEGIN{exit !(f>=n)}'; then
            echo "DISK_OK free=$(free_gib)GiB need=$1GiB"; return 0
        fi
        echo "WAITING_FOR_DISK free=$(free_gib)GiB need=$1GiB $(date '+%F %T')"
        sleep 300
    done
    echo "DISK_TIMEOUT need=$1GiB"; return 1
}
stage() { echo; echo "===== $1: $(date '+%F %T') free=$(free_gib)GiB ====="; }

if [ "${1:-}" = --worker ]; then
    OUT=$2
    LOG=$3
    export MPLCONFIGDIR=/tmp/thai-dsr-mpl PYTHONUNBUFFERED=1
    printf '%s\n' "$$" > "$LOG.started"
    set +e
    /usr/bin/caffeinate -i -w $$ &
    (
    set -eo pipefail
    PY=.venv/bin/python
    NEW=$((TARGET - 200))

    stage "1a prepare Common Voice clips 200..$((TARGET - 1))"
    if [ ! -f data/common_voice_ext/transcripts.csv ]; then
        rm -rf data/common_voice_ext
        $PY scripts/prepare_common_voice.py --streaming --start-index 200 --limit "$TARGET" \
            --prefix-transcripts data/common_voice/transcripts.csv --output-dir data/common_voice_ext
    fi
    export HF_HUB_OFFLINE=1

    stage "1b distort (severe, seed 42)"
    if [ "$(ls data/distorted_cv_ext 2>/dev/null | wc -l)" -ne "$NEW" ]; then
        rm -rf data/distorted_cv_ext
        $PY -m src.preprocessing.distortion --input_dir data/common_voice_ext \
            --output_dir data/distorted_cv_ext --severities severe --seed 42 | grep -v '^  \['
        [ "$(ls data/distorted_cv_ext | wc -l)" -eq "$NEW" ]
    fi

    stage "1c layer-9 embeddings"
    wait_for_disk 2.2
    $PY -m src.preprocessing.extract_embedding --input_dir data/distorted_cv_ext \
        --output_dir data/embeddings/distorted --layer 9 | grep -v '^Layer 09 shape'
    [ "$(ls data/embeddings/distorted | grep -c '^cv_.*_severe_layer09.npy$')" -eq "$TARGET" ]
    du -sh data/common_voice_ext data/distorted_cv_ext data/embeddings/distorted

    stage "1d manifest + split"
    $PY -m src.preprocessing.build_manifest \
        --clean_dirs data/raw data/common_voice data/common_voice_ext \
        --distorted_dirs data/distorted data/distorted_cv data/distorted_cv_ext \
        --severities severe --output data/manifest_expanded.csv
    $PY scripts/build_expanded_splits.py --manifest data/manifest_expanded.csv \
        --output data/splits_expanded.json

    stage "2a train fresh L1 layer-9 mapper (patience $PATIENCE)"
    wait_for_disk 1.0
    mkdir -p results/checkpoints/mapper_expanded
    $PY -u -m src.training.train --train-config configs/train_expanded.yaml \
        --best-checkpoint results/checkpoints/mapper_expanded/mapper_layer9_expanded.pt \
        --loss-curve results/logs/loss_curve_expanded.png \
        --early-stop-patience "$PATIENCE"

    stage "2b compare with W5 baseline"
    $PY scripts/evaluate_expanded_mapper.py \
        --checkpoint baseline=results/checkpoints/mapper_layer9.pt \
        --checkpoint expanded=results/checkpoints/mapper_expanded/mapper_layer9_expanded.pt \
        --output-dir "$OUT"
    stage DONE
    )
    RESULT=$?
    printf '%s\n' "$RESULT" > "$LOG.exit"
    exit "$RESULT"
fi
RUN="expanded$(date '+%Y%m%d_%H%M%S')_$$"
OUT="results/checkpoints/mapper_expanded/eval_$RUN"
LOG="$PWD/results/logs/expanded_baseline_$RUN.log"
mkdir -p results/logs
nohup .venv/bin/python -c 'import os,sys; os.setsid(); os.execv("/bin/bash", ["/bin/bash"]+sys.argv[1:])' \
  "$PWD/scripts/run_expanded_baseline_detached.sh" --worker "$OUT" "$LOG" </dev/null >"$LOG" 2>&1 &
WORKER=$!
disown "$WORKER"
printf '%s\n' "$WORKER" > "$LOG.pid"
printf 'PID=%s\nOUTPUT=%s\nLOG=%s\n' "$WORKER" "$OUT" "$LOG"
for ((i=0; i<100; i++)); do
    if [ -f "$LOG.started" ]; then
        printf 'DETACHED_WORKER_STARTED=%s\n' "$(cat "$LOG.started")"
        exit 0
    fi
    sleep 0.1
done
printf 'Worker failed to acknowledge startup; inspect %s\n' "$LOG" >&2
exit 1
