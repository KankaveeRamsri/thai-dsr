# wer.py
#
# Purpose:
#   Compute Word Error Rate (WER) for reconstructed Thai speech by running
#   ASR (e.g. a HuggingFace transformers Thai ASR model) on reconstructed
#   audio and comparing transcripts against ground truth.
#
# Expected responsibilities:
#   - Run ASR inference on reconstructed audio in results/audio_samples/
#   - Normalize Thai text (tokenization/word segmentation) before comparison
#   - Compute and report WER/CER against reference transcripts
