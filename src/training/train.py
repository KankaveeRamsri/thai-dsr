# train.py
#
# Purpose:
#   Main training entry point that ties together dataset.py, the models in
#   src/models/, and the hyperparameters in configs/train.yaml to fit the
#   dysarthric-to-clean reconstruction model.
#
# Expected responsibilities:
#   - Parse configs/{data,model,train}.yaml
#   - Build DataLoaders (dataset.py), model (encoder/mapper/vocoder)
#   - Run the training loop: forward pass, loss, backward, optimizer step
#   - Log metrics/checkpoints to results/logs/ and results/checkpoints/
#   - Support resuming from a checkpoint
