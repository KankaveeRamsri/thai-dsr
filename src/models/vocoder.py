# vocoder.py
#
# Purpose:
#   Convert reconstructed acoustic features (output of mapper.py) back
#   into a listenable waveform.
#
# Expected responsibilities:
#   - Wrap a pretrained or fine-tuned vocoder (e.g. HiFi-GAN) to synthesize
#     audio from mel-spectrogram / feature sequences
#   - Provide a synthesize(features) -> waveform interface
#   - Be used by both training (for audio-domain losses/preview) and
#     src/inference/run.py for final output generation
