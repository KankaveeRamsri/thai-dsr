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

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

INPUT_DIM = 1024  # wav2vec2 hidden size
LSTM_HIDDEN_SIZE = 512
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.1
PROJECTION_HIDDEN_DIM = 256
MEL_DIM = 80


class MapperModel(nn.Module):
    """Bi-LSTM mapper: distorted wav2vec2 embeddings -> clean mel-spectrogram.

    Input:  (batch, T, 1024) wav2vec2 hidden states, already interpolated to
             the mel frame rate (see src/training/dataset.py).
    Output: (batch, T, 80) predicted log-mel spectrogram.
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_num_layers=LSTM_NUM_LAYERS,
        lstm_dropout=LSTM_DROPOUT,
        projection_hidden_dim=PROJECTION_HIDDEN_DIM,
        mel_dim=MEL_DIM,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        lstm_output_dim = lstm_hidden_size * 2  # bidirectional concat
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, projection_hidden_dim),
            nn.ReLU(),
            nn.Linear(projection_hidden_dim, mel_dim),
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, T, input_dim) wav2vec2 embeddings.
            lengths: optional (batch,) tensor of true sequence lengths.
                When given, the sequence is packed before the LSTM so padded
                timesteps don't leak into the recurrent state, then unpacked
                back to (batch, T, ...) for the projection head.

        Returns:
            (batch, T, mel_dim) predicted mel-spectrogram.
        """
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.size(1)
            )
        else:
            lstm_out, _ = self.lstm(x)

        normed = self.layer_norm(lstm_out)
        return self.projection(normed)

    def count_parameters(self):
        """Print and return the total number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total:,}")
        return total


def build_mapper_from_config(model_config):
    """Construct the configured mapper and reject unsupported architectures."""
    mapper_config = model_config["mapper"]
    architecture = str(mapper_config["architecture"]).lower()
    if architecture != "bilstm":
        raise ValueError(
            f"Unsupported mapper architecture '{architecture}'; expected 'bilstm'"
        )
    return MapperModel(
        input_dim=int(mapper_config["input_dim"]),
        lstm_hidden_size=int(mapper_config["lstm_hidden_size"]),
        lstm_num_layers=int(mapper_config["lstm_num_layers"]),
        lstm_dropout=float(mapper_config["dropout"]),
        projection_hidden_dim=int(mapper_config["projection_hidden_dim"]),
        mel_dim=int(mapper_config["mel_dim"]),
    )


if __name__ == "__main__":
    model = MapperModel()
    print(model)
    model.count_parameters()

    batch_size, seq_len = 4, 431
    dummy_input = torch.randn(batch_size, seq_len, INPUT_DIM)
    dummy_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)

    output = model(dummy_input, dummy_lengths)
    print(f"\nInput shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    assert output.shape == (batch_size, seq_len, MEL_DIM), "Unexpected output shape!"
    print("OK: output shape matches (batch, T, 80).")

    # Sanity check the variable-length path: padded timesteps should not
    # change the prediction at valid timesteps for a shorter sequence.
    var_lengths = torch.tensor([431, 300, 250, 431], dtype=torch.long)
    var_output = model(dummy_input, var_lengths)
    print(f"Variable-length batch output shape: {tuple(var_output.shape)}")
    assert var_output.shape == (batch_size, seq_len, MEL_DIM)
    print("OK: packed-sequence forward pass handles variable lengths.")
