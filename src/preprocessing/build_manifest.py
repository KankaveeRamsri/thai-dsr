# build_manifest.py
#
# Purpose:
#   Scan data/raw/, data/clean/, data/distorted/, and data/embeddings/ to
#   build a manifest file (CSV/JSON) that maps each utterance to its
#   audio paths, embedding paths, speaker id, transcript, and split
#   (train/val/test).
#
# Expected responsibilities:
#   - Walk dataset directories and match clean/distorted pairs by file id
#   - Attach metadata (speaker, duration, transcript if available)
#   - Assign train/val/test splits
#   - Write the manifest consumed by src/training/dataset.py
