# distortion.py
#
# Purpose:
#   Generate synthetic "dysarthric-like" distortions from clean Thai speech
#   audio, so the model has (distorted, clean) pairs to learn reconstruction
#   from when real dysarthric recordings are scarce.
#
# Expected responsibilities:
#   - Load clean audio from data/clean/
#   - Apply distortion transforms (e.g. time-stretch, pitch jitter, slurring
#     simulation, articulation dropout, noise injection) using librosa /
#     parselmouth
#   - Save distorted audio to data/distorted/
#   - Expose a CLI or function entry point that can process a whole folder
#     or a single file, with configurable distortion severity/type
