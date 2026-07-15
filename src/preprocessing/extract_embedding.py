# extract_embedding.py
#
# Purpose:
#   Extract speaker/content embeddings (e.g. ECAPA-TDNN speaker embeddings
#   via speechbrain, or content/phonetic embeddings via a pretrained
#   HuggingFace model) from clean and/or distorted audio.
#
# Expected responsibilities:
#   - Load audio from data/clean/ or data/distorted/
#   - Run embedding model(s) defined in src/models/encoder.py
#   - Save resulting embeddings (.npy / .pt) to data/embeddings/
#   - Support batch processing driven by the manifest produced by
#     build_manifest.py
