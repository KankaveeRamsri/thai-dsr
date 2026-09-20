#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LAYERS=(6 8 9 10 12)
EPOCHS=50

mkdir -p results/checkpoints results/logs results/audio_samples

for layer in "${LAYERS[@]}"; do
  checkpoint="results/checkpoints/mapper_layer${layer}.pt"
  final_periodic="results/checkpoints/mapper_layer${layer}_epoch_050.pt"
  loss_curve="results/logs/loss_curve_layer${layer}.png"
  metrics="results/metrics_layer${layer}.json"
  audio_dir="results/audio_samples/layer${layer}"
  log="results/logs/layer${layer}_training.log"

  if [[ -f "$metrics" && -f "$checkpoint" && -f "$final_periodic" ]]; then
    echo "Layer ${layer}: complete; skipping."
    continue
  fi

  {
    echo "===== Layer ${layer}: $(date) ====="

    python -m src.preprocessing.extract_embedding \
      --input_dir data/distorted \
      --output_dir data/embeddings/distorted \
      --layer "$layer"
    python -m src.preprocessing.extract_embedding \
      --input_dir data/distorted_cv \
      --output_dir data/embeddings/distorted \
      --layer "$layer"

    if [[ ! -f "$final_periodic" ]]; then
      shopt -s nullglob
      candidates=(results/checkpoints/mapper_layer${layer}_epoch_*.pt)
      shopt -u nullglob
      if [[ -f "$checkpoint" ]]; then
        candidates+=("$checkpoint")
      fi
      resume_args=()
      if (( ${#candidates[@]} > 0 )); then
        latest="$(ls -t "${candidates[@]}" | head -n 1)"
        resume_args=(--resume-checkpoint "$latest")
        echo "Resuming layer ${layer} from ${latest}"
      fi

      python -m src.training.train \
        --layer "$layer" \
        --num-epochs "$EPOCHS" \
        --best-checkpoint "$checkpoint" \
        --loss-curve "$loss_curve" \
        "${resume_args[@]+"${resume_args[@]}"}"
    else
      echo "Training for layer ${layer} already reached epoch ${EPOCHS}."
    fi

    python -m src.inference.run --batch \
      --manifest data/manifest_w5.csv \
      --splits data/splits_w5.json \
      --split test \
      --layer "$layer" \
      --checkpoint "$checkpoint" \
      --output_dir "$audio_dir"

    python -m src.evaluation.metrics \
      --manifest data/manifest_w5.csv \
      --splits data/splits_w5.json \
      --split test \
      --reconstructed_dir "$audio_dir" \
      --checkpoint "$checkpoint" \
      --output_json "$metrics"

    echo "===== Layer ${layer} complete: $(date) ====="
  } 2>&1 | tee -a "$log"
done

echo "All layer runs are complete. Run: python scripts/compare_layers.py"
