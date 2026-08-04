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

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Pretrained Thai wav2vec2 XLSR-53 checkpoint on HuggingFace -- same model
# used by src/preprocessing/extract_embedding.py to build data/embeddings/.
MODEL_NAME = "airesearch/wav2vec2-large-xlsr-53-th"
TARGET_SAMPLE_RATE = 16000
DEFAULT_LAYER = 9  # wav2vec2 layer selected via scripts/select_layer.py


class Wav2Vec2ContentEncoder:
    """Wraps the pretrained Thai wav2vec2 model to extract one layer's hidden
    state as a content embedding.

    Reuses the model-loading/inference logic from
    src/preprocessing/extract_embedding.py, packaged as a class so
    src/models/mapper.py and src/inference/run.py can encode audio directly
    instead of shelling out to the standalone extraction script.
    """

    def __init__(self, layer=DEFAULT_LAYER, device=None, model_name=MODEL_NAME, freeze=True):
        self.layer = layer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(self.device)

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def _load_audio(self, wav_path):
        """Load a .wav file and resample it to TARGET_SAMPLE_RATE if needed.

        Uses soundfile rather than torchaudio.load, since torchaudio>=2.9
        needs the optional torchcodec backend for file I/O that isn't
        installed in this project's environment.
        """
        audio_data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio_data.T)  # (channels, time)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        return waveform.squeeze(0).numpy()

    def encode(self, wav_path, layer=None):
        """Run wav2vec2 inference on a .wav file and return one layer's hidden state.

        Returns:
            np.ndarray of shape (T, 1024).
        """
        layer = self.layer if layer is None else layer
        audio_array = self._load_audio(wav_path)

        inputs = self.feature_extractor(
            audio_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        hidden_state = outputs.hidden_states[layer]  # (1, T, 1024)
        return hidden_state.squeeze(0).cpu().numpy()

    def encode_all_layers(self, wav_path):
        """Run wav2vec2 inference once and return every layer's hidden state.

        Returns:
            tuple of 25 np.ndarray, each shaped (T, 1024) -- layer 0 is the
            CNN feature-encoder output, layers 1-24 are the transformer blocks.
        """
        audio_array = self._load_audio(wav_path)

        inputs = self.feature_extractor(
            audio_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)

        return tuple(h.squeeze(0).cpu().numpy() for h in outputs.hidden_states)
