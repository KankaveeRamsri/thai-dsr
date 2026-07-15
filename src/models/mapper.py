# mapper.py
#
# Purpose:
#   Define the core reconstruction network that maps dysarthric
#   (distorted) speech representations to clean/target speech
#   representations, conditioned on speaker embedding.
#
# Expected responsibilities:
#   - Define the mapping model architecture (e.g. seq2seq / transformer /
#     diffusion-based feature mapper)
#   - Take encoder outputs (from encoder.py) as input
#   - Produce features consumable by vocoder.py to synthesize audio
#   - Expose forward() and loss computation used by src/training/train.py
