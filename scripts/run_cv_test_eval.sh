#!/usr/bin/env bash
# Full CV 27.0 test evaluation: floor, control, both ceilings, (a) and (b),
# then the summary tables. One system at a time.
#
# Expects the Colab checkpoints at (copy run_info.json along with best.pt so
# the evaluator can confirm each model's alignment):
#   <data root>/checkpoints/cv27_layer9_linear/{best.pt,run_info.json}
#   <data root>/checkpoints/cv27_layer9_tempo/{best.pt,run_info.json}
# Results: <data root>/eval/test_full/<system>/ and .../summary/.
# Extra arguments (e.g. --limit 5 --out-dir /tmp/x) go to every evaluator run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
export PYTHONWARNINGS=ignore
ROOT=$("$PY" -c "from src.utils.config import cv_data_root; print(cv_data_root())")
CKPT="$ROOT/checkpoints"

for alignment in linear tempo; do
  [[ -f "$CKPT/cv27_layer9_$alignment/best.pt" ]] || { echo "missing $CKPT/cv27_layer9_$alignment/best.pt"; exit 1; }
done

for system in distorted pv_roundtrip copysynth_clean copysynth_tempo; do
  "$PY" scripts/evaluate_cv_aligned.py --system "$system" "$@"
done
for alignment in linear tempo; do
  "$PY" scripts/evaluate_cv_aligned.py --system mapper --alignment "$alignment" \
    --checkpoint "$CKPT/cv27_layer9_$alignment/best.pt" "$@"
done
