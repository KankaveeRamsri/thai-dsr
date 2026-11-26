#!/usr/bin/env bash
# Evaluate the six fixed-layer ablation runs (docs/ablation_plan.md) on the full CV 27.0
# test set with the standard protocol (scripts/evaluate_cv_aligned.py), then run the
# pre-registered H1/H2 tests (scripts/ablation_tests.py).
#
# Expects <data root>/checkpoints/cv27_layer<NN>_tempo/{best.pt,run_info.json,input_norm.npz}
# and test embeddings for every layer in <data root>/embeddings_test_layers/ (made on this
# Mac with: python -m src.preprocessing.extract_embedding --layers 3 6 9 12 18 24).
# Results: <data root>/eval/test_full/mapper_tempo_norm_layer<NN>/ and .../summary_ablation/.
# Extra arguments go to every evaluator run (e.g. --limit 2 --out-dir /tmp/x).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
export PYTHONWARNINGS=ignore
ROOT=$("$PY" -c "from src.utils.config import cv_data_root; print(cv_data_root())")
LAYERS=${LAYERS:-"3 6 9 12 18 24"}

for layer in $LAYERS; do
  nn=$(printf "%02d" "$layer")
  "$PY" scripts/evaluate_cv_aligned.py --system mapper --alignment tempo --layer "$layer" \
    --checkpoint "$ROOT/checkpoints/cv27_layer${nn}_tempo/best.pt" \
    --embeddings-dir "$ROOT/embeddings_test_layers" "$@"
done
