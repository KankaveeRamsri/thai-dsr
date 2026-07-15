# metrics.py
#
# Purpose:
#   Compute objective speech quality/intelligibility metrics comparing
#   reconstructed audio against ground-truth clean audio.
#
# Expected responsibilities:
#   - Compute STOI via pystoi
#   - Compute PESQ via pesq
#   - Aggregate metrics across a results/audio_samples/ folder or manifest
#     and report summary statistics
