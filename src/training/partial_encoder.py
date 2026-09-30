"""wav2vec2 layer features with the top N blocks below the tap point trainable.

The mapper reads hidden_states[layer]. With do_stable_layer_norm, that is the
output of encoder.layers[layer - 1], so blocks above the tap never affect it.
"Top N" therefore means the N transformer blocks directly beneath the tap
(encoder.layers[layer - N:layer]); the CNN feature extractor, feature
projection, positional convolution and every lower block stay frozen.

The frozen prefix hidden_states[layer - N] is deterministic, so it is computed
once per utterance and cached in memory; only the top N blocks run per step.
The encoder always stays in eval mode (no dropout/layerdrop), matching how the
cached frozen features were produced, so with unchanged weights the features
equal the cached layer embeddings.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.models.encoder import TARGET_SAMPLE_RATE, Wav2Vec2ContentEncoder
from src.utils.config import REPO_ROOT


class PartialWav2Vec2(nn.Module):
    def __init__(self, layer, unfreeze_top_layers, device):
        super().__init__()
        if not 1 <= unfreeze_top_layers <= layer:
            raise ValueError(f"--unfreeze-top-layers must be in 1..{layer}, got {unfreeze_top_layers}")
        wrapper = Wav2Vec2ContentEncoder(device=device, layer=layer)
        self.feature_extractor = wrapper.feature_extractor
        self.load_audio = wrapper._load_audio
        self.model = wrapper.model
        self.model.encoder.layers = self.model.encoder.layers[:layer]
        self.top = self.model.encoder.layers[layer - unfreeze_top_layers:layer]
        self.layer, self.unfreeze_top_layers, self.device = layer, unfreeze_top_layers, device
        self.model.requires_grad_(False)
        self.top.requires_grad_(True)
        self._prefix = {}
        self.train(False)

    def train(self, mode=True):
        return super().train(False)

    def trainable_parameters(self):
        return list(self.top.parameters())

    def top_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.top.state_dict().items()}

    def load_top_state_dict(self, state_dict):
        self.top.load_state_dict(state_dict)

    @torch.no_grad()
    def prefix(self, utterance_id, wav_path):
        if utterance_id not in self._prefix:
            audio = self.load_audio(wav_path)
            inputs = self.feature_extractor(audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
            hidden = self.model(inputs.input_values.to(self.device), output_hidden_states=True).hidden_states
            self._prefix[utterance_id] = hidden[self.layer - self.unfreeze_top_layers][0].cpu()
        return self._prefix[utterance_id]

    def forward(self, utterance_id, wav_path):
        """Return (T_emb, 1024) layer features; differentiable w.r.t. the top blocks."""
        hidden = self.prefix(utterance_id, wav_path).to(self.device)[None]
        for block in self.top:
            hidden = block(hidden)[0]
        return hidden[0]

    def batch_features(self, keys, lengths):
        """Features for (utterance_id, distorted_path) keys, interpolated to mel lengths and padded."""
        features = [
            F.interpolate(self(uid, REPO_ROOT / path).T[None], size=int(n),
                          mode="linear", align_corners=False)[0].T
            for (uid, path), n in zip(keys, lengths)
        ]
        return pad_sequence(features, batch_first=True)


class EncoderInputDataset(Dataset):
    """Wrap DysarthricDataset to yield ((utterance_id, distorted_path), clean mel)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.base.manifest.iloc[index]
        mel = torch.from_numpy(self.base._extract_clean_mel(row["clean_path"])).float()
        return (row["utterance_id"], row["distorted_path"]), mel


def collate_encoder_inputs(batch):
    keys, mels = zip(*batch)
    lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    return list(keys), pad_sequence(list(mels), batch_first=True), lengths
