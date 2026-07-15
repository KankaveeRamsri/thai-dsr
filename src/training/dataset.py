# dataset.py
#
# Purpose:
#   PyTorch Dataset/DataLoader definitions that read the manifest produced
#   by build_manifest.py and yield (distorted, clean, speaker_embedding)
#   tuples for training.
#
# Expected responsibilities:
#   - Load manifest (CSV/JSON) and split by train/val/test
#   - Load audio/embeddings referenced in each manifest row
#   - Apply padding/collation for variable-length audio in batches
#   - Read dataset-related settings from configs/data.yaml
