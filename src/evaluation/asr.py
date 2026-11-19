"""Thai ASR for the intelligibility metric (CER / WER), with a pinned model.

The ASR model must not be the wav2vec2 model inside the pipeline
(airesearch/wav2vec2-large-xlsr-53-th), so the metric does not reward
representations it shares with our encoder. It also must not be fine-tuned on
Common Voice Thai, because the test sentences may be in its training data.
The model id, revision and decoding settings below are fixed for every system
and every run; the evaluator records them with the results.

Runs on the CPU in float32 (pinned). On Apple MPS (torch 2.8), float16 and
bfloat16 both decode garbage ("!!!!" or empty), and float32 on MPS needs a
second copy of the 6 GB weights that does not fit in 16 GB. CPU float32 gives
correct transcripts at about 3.7 s per clip.

Decoding: language "th", task "transcribe", greedy search (num_beams=1,
do_sample=False), no timestamps, at most 128 new tokens. Clips are 16 kHz mono
and shorter than 30 s, so each one is a single Whisper window.
"""

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ASR_MODEL_ID = "openai/whisper-large-v3"
ASR_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
GENERATION = {"language": "th", "task": "transcribe", "num_beams": 1, "do_sample": False,
              "max_new_tokens": 128, "return_timestamps": False}
SAMPLE_RATE = 16000


class ThaiASR:
    def __init__(self, device=torch.device("cpu"), dtype=torch.float32, model_id=ASR_MODEL_ID,
                 revision=ASR_REVISION):
        self.model_id, self.revision, self.device, self.dtype = model_id, revision, device, dtype
        self.processor = WhisperProcessor.from_pretrained(model_id, revision=revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()

    @torch.no_grad()
    def transcribe(self, waveforms):
        """List of 16 kHz float waveforms -> list of transcripts."""
        for wav in waveforms:
            if len(wav) > 30 * SAMPLE_RATE:
                raise ValueError(f"clip of {len(wav) / SAMPLE_RATE:.1f} s is longer than one Whisper window")
        inputs = self.processor([np.asarray(w, dtype=np.float32) for w in waveforms], sampling_rate=SAMPLE_RATE,
                                return_tensors="pt", return_attention_mask=True)
        ids = self.model.generate(inputs.input_features.to(self.device, self.dtype),
                                  attention_mask=inputs.attention_mask.to(self.device), **GENERATION)
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]

    def settings(self):
        return {"model_id": self.model_id, "revision": self.revision, "dtype": str(self.dtype).replace("torch.", ""),
                "device": str(self.device), "generation": GENERATION}
