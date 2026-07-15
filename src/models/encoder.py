# encoder.py
#
# Purpose:
#   Define the encoder(s) that turn raw/distorted audio into feature
#   representations used downstream: an ECAPA-TDNN speaker encoder
#   (via speechbrain) and/or a content encoder (e.g. HuBERT/Wav2Vec2 via
#   transformers).
#
# Expected responsibilities:
#   - Wrap pretrained speechbrain/transformers models behind a simple
#     encode(audio) -> embedding interface
#   - Handle freezing/fine-tuning options
#   - Be importable by extract_embedding.py and mapper.py
