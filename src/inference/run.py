# run.py
#
# Purpose:
#   CLI/script entry point to run the full inference pipeline on a single
#   audio file or a folder: encode -> map -> vocode -> save reconstructed
#   audio.
#
# Expected responsibilities:
#   - Load a trained checkpoint from results/checkpoints/
#   - Run encoder.py -> mapper.py -> vocoder.py on input audio
#   - Save reconstructed output to results/audio_samples/
#   - Accept CLI args for input path, checkpoint path, output path
