# Thai-DSR: Dysarthric Speech Reconstruction for Thai

Speech Re-Synthesis system for dysarthric speakers, using a frozen self-supervised
speech model (wav2vec2) to extract content representations, a lightweight trainable
mapper network to predict mel-spectrograms, and a pretrained neural vocoder to
synthesize intelligible speech. Developed as part of an AI Engineering senior
project (PSU, Faculty of Engineering), advised by Dr. Anant Choksuriwong.

> **Status:** Prototype pipeline complete and working end-to-end. 8 post-midterm
> experiments aimed at improving intelligibility (STOI/PESQ) did not yield a
> reproducible improvement over baseline — see [Results](#results-so-far) below.
> Next step: an isolate-variable test on wav2vec2 layer selection (see
> [Next Steps](#next-steps)).

---

## Problem

Dysarthria (e.g. from stroke) degrades speech intelligibility while largely
preserving speaker identity and semantic intent. This project reconstructs more
intelligible speech from dysarthric input, aiming to help affected speakers
communicate more effectively — a form of assistive technology (AAC).

## Pipeline Overview

```
Dysarthric audio
      │
      ▼
wav2vec2-XLSR-53-TH (frozen, 25 transformer blocks)
      │  tap → layer 9 hidden state (1,024-dim)
      ▼
Mapper (bi-LSTM, ~12.9M trainable params)
      │  predicts 80-dim mel-spectrogram
      ▼
HiFi-GAN vocoder (frozen, UNIVERSAL_V1 pretrained)
      │
      ▼
Synthesized (re-synthesized) audio
```

- **wav2vec2-XLSR-53-TH** — frozen throughout; only the layer-9 hidden state is used
  as input to the Mapper. Layer 9 was selected via a 25-layer cosine-similarity
  analysis between clean/distorted embeddings (layers 6–12 form a stable
  ~0.91–0.93 similarity band; layer 24's higher similarity is a LayerNorm
  magnitude-compression artifact, not a better representation).
- **Mapper** — the only trained component. bi-LSTM (hidden_size=512,
  bidirectional), LayerNorm, linear projection 1024→256→80. Trained with L1 loss
  against ground-truth mel-spectrograms from parallel (distorted, clean) pairs.
- **HiFi-GAN** — frozen pretrained vocoder (UNIVERSAL_V1), never fine-tuned in any
  experiment to date.

### Training data

Parallel data is synthesized from clean Thai speech using three distortion
techniques to simulate dysarthric-like degradation:

| Technique | Tool | Notes |
|---|---|---|
| Formant / F0 perturbation | `parselmouth` | severity levels, e.g. "severe" |
| Segment-level tempo perturbation | `librosa` | 0.6×–1.4× speed |
| Additive noise & reverb | `numpy` | — |

Distortion realism was validated acoustically against literature thresholds
(Hernandez et al., 2022; Kent & Kim, 2003):

| Metric | Measured | Threshold | Result |
|---|---|---|---|
| Jitter | 4.12 ± 0.91% | > 3% | ✅ |
| Shimmer | 18.89 ± 1.38% | > 6% | ✅ |
| HNR | 4.78 ± 1.43 dB | < 15 dB | ✅ |

## Evaluation

Fixed benchmark used identically across all experiments for fair comparison:

- **STOI / PESQ** — 8 fixed held-out clips, vocoder output vs. ground-truth normal
  speech. *(Small sample size — treat STOI/PESQ deltas between approaches as
  noisy; do not over-interpret small differences.)*
- **mel-L1** — 32 held-out validation utterances, predicted mel vs. target mel
  (no vocoder involved).

## Results So Far

### Cross-lingual probe (pre-dates the 8 approaches below)

Thai (n=10) vs. English (n=5) at "severe" distortion: SNR improved for both
languages after vocoding, but STOI diverged — Thai got *worse* (0.165→0.139)
while English improved (0.113→0.124). This indicated HiFi-GAN alone isn't the
bottleneck, motivating the dataset-expansion and representation experiments
below.

### 8 post-midterm improvement attempts

| # | Approach | Result | Outcome |
|---|---|---|---|
| 1 | Joint adversarial fine-tuning, end-to-end (waveform discriminator) | Training unstable | ❌ Abandoned |
| 2 | Postnet residual mel correction | Stable; no STOI/PESQ change | ❌ |
| 3 | Weighted sum over all 25 layers (learned weights) | STOI peaked 0.36 @ step 200, then fell below baseline; weights converge near-uniform | ❌ |
| 4 | Raw spectral feature added to Mapper input | Made results worse | ❌ (disproves "SSL lacks detail" hypothesis) |
| 5 | Mel-domain-only adversarial loss | Stable; best mel-L1 (1.276); STOI/PESQ unchanged | ❌ (for intelligibility) |
| 6 | Dataset expansion, 147 → 1,407 utterances | Better STOI on new clips (0.2743→0.3169); worse on fixed benchmark (0.2615→0.2501) | ⚠️ Mixed |
| 7A | Continue from #6 checkpoint w/ mel-adversarial loss, 3,000 steps | STOI declined monotonically | ❌ Discriminator dominated |
| 7B | Unfreeze wav2vec2 blocks 8–9 (just below the tap), plain L1 loss | STOI oscillates 0.24–0.28, no sustained trend | ❌ Looks like noise |

**Net conclusion:** none of the 8 approaches produced a statistically trustworthy,
reproducible STOI/PESQ improvement over baseline. Collectively they rule out
several hypotheses (end-to-end adversarial training, light residual correction,
missing acoustic detail in the SSL features, adversarial training "paying off"
with more time/data) and point toward the layer-9 choice *not* being the primary
bottleneck — approaches #3 and #7B both probed near the tap point without
finding a sustained gain.

## Next Steps

- **Isolate-variable layer test** — swap in fixed alternative wav2vec2 layers in
  place of layer 9, keeping the known-stable plain-L1 training recipe unchanged,
  to directly test whether layer choice significantly affects STOI/PESQ (cheaper
  and cleaner than approach #3's learned-weighting setup).
- Fine-tune HiFi-GAN on Thai speech + ablation study, then full evaluation
  (WER / MOS / Speaker Similarity via ECAPA-TDNN) — planned for next semester.

## Repository Structure

```
.
├── configs/
│   ├── data.yaml              # data paths, split ratios/seed
│   ├── model.yaml             # wav2vec2 model id, Mapper (bi-LSTM) and vocoder settings
│   ├── train.yaml             # baseline recipe: layer 9, W5 manifest/splits, L1, Adam 1e-3
│   ├── train_expanded.yaml    # same recipe on the expanded (2,010-utterance) dataset
│   └── joint_smoke_hybrid.json
├── data/                      # audio + embeddings are git-ignored; manifests/splits are tracked
│   ├── manifest_w5.csv / splits_w5.json              # 210 utterances (fixed benchmark split)
│   ├── manifest_expanded.csv / splits_expanded.json  # 2,010 utterances, W5 assignments kept
│   └── manifest.csv / splits.json                    # original 10-utterance set
├── src/
│   ├── preprocessing/         # distortion.py, extract_embedding.py, build_manifest.py
│   ├── models/                # encoder (wav2vec2), mapper (bi-LSTM), vocoder, postnet,
│   │                          # weighted_layer_mapper, spectral_aux_mapper, mel_discriminator
│   ├── training/              # train.py (L1 baseline, optional wav2vec2 unfreezing),
│   │                          # train_mapper_{weighted,spectral_aux,mel_adversarial}.py,
│   │                          # train_postnet.py, joint_finetune.py, finetune_hifigan.py,
│   │                          # dataset/splits/partial_encoder/grad_clip helpers
│   ├── inference/run.py       # end-to-end: wav2vec2 → Mapper → HiFi-GAN
│   ├── evaluation/            # metrics.py (STOI/PESQ/SNR), mapper_eval.py (fixed benchmark),
│   │                          # metrics_en.py (cross-lingual probe)
│   └── utils/                 # config loading, mel-spectrogram computation
├── scripts/                   # data download, layer analysis, evaluation, detached-run
│                              # launchers (run_*_detached.sh) and Colab notebooks
├── tests/                     # unittest suite (python -m unittest discover tests)
├── notebooks/                 # 01_explore … 04_eval exploratory notebooks
├── docs/                      # pipeline.md, related_work.md, per-experiment notes
├── reports/                   # weekly progress reports (W1–W7, PDF)
├── results/                   # tracked metrics (metrics_*.json, layer analysis);
│                              # checkpoints/, logs/, audio_samples/ are git-ignored
├── vendor/hifi-gan/           # HiFi-GAN source + UNIVERSAL_V1 (git-ignored; see Setup)
└── requirements.txt
```

## Setup

Tested with Python 3.10. There is no `setup.py`/`pyproject.toml`; run everything
from the repository root with `python -m ...`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # pesq builds a C extension: needs Xcode CLT / build tools
```

HiFi-GAN is not tracked in git. Clone it into `vendor/hifi-gan` and place the
UNIVERSAL_V1 pretrained generator (`config.json` and `g_02500000`, linked from the
HiFi-GAN README) in `vendor/hifi-gan/checkpoints/UNIVERSAL_V1/`:

```bash
git clone https://github.com/jik876/hifi-gan vendor/hifi-gan
git -C vendor/hifi-gan checkout 4769534d45265d52a904b850da5a622601885777
```

The wav2vec2 model (`airesearch/wav2vec2-large-xlsr-53-th`, ~1.2 GB) downloads
automatically from Hugging Face on first use.

### Run

```bash
# 1. Data: 200 validated Common Voice Thai clips (own recordings go in data/raw/)
python scripts/prepare_common_voice.py --limit 200 --output-dir data/common_voice

# 2. Distort (severe) and cache layer-9 embeddings
python -m src.preprocessing.distortion --input_dir data/raw --output_dir data/distorted --severities severe --seed 42
python -m src.preprocessing.distortion --input_dir data/common_voice --output_dir data/distorted_cv --severities severe --seed 42
python -m src.preprocessing.extract_embedding --input_dir data/distorted --output_dir data/embeddings/distorted --layer 9
python -m src.preprocessing.extract_embedding --input_dir data/distorted_cv --output_dir data/embeddings/distorted --layer 9

# 3. Manifest of (clean, distorted) pairs
python -m src.preprocessing.build_manifest --clean_dirs data/raw data/common_voice \
    --distorted_dirs data/distorted data/distorted_cv --severities severe --output data/manifest_w5.csv

# 4. Train the baseline Mapper (configs/train.yaml: layer 9, plain L1); reuses the
#    utterance-level split in data/splits_w5.json (created if missing)
python -m src.training.train --train-config configs/train.yaml \
    --best-checkpoint results/checkpoints/mapper_layer9.pt

# 5. End-to-end synthesis (wav2vec2 → Mapper → HiFi-GAN); writes *_reconstructed.wav
python -m src.inference.run --input path/to/distorted.wav \
    --checkpoint results/checkpoints/mapper_layer9.pt --output_dir results/audio_samples

# 6. Fixed benchmark (32-utterance mel-L1 + 8-clip STOI/PESQ)
python scripts/evaluate_expanded_mapper.py \
    --checkpoint baseline=results/checkpoints/mapper_layer9.pt --output-dir results/eval_baseline
```

Step 6 also scores the expanded validation split, so it needs
`data/manifest_expanded.csv` and layer-9 embeddings for those clips. For the
W5-only STOI/PESQ/SNR table, batch-synthesize the validation split and score it:

```bash
python -m src.inference.run --batch --manifest data/manifest_w5.csv --splits data/splits_w5.json \
    --split val --checkpoint results/checkpoints/mapper_layer9.pt --output_dir results/audio_samples/v2
python -m src.evaluation.metrics --manifest data/manifest_w5.csv --splits data/splits_w5.json --split val
```

Experiment trainers (`src/training/train_mapper_*.py`, `train_postnet.py`,
`joint_finetune.py`) are run with `python -m src.training.<name> --help`; detached
launchers live in `scripts/run_*_detached.sh`. Optional wav2vec2 unfreezing in the
baseline trainer: `python -m src.training.train --unfreeze-top-layers 2 ...`.

Tests:

```bash
python -m unittest discover tests
```

## Author

กันต์กวี รามศรี (Bas) — AI Engineering, Prince of Songkla University
Advisor: Dr. Anant Choksuriwong

## References

- Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0. NeurIPS.
- Kong, J., Kim, J., & Bae, J. (2020). HiFi-GAN. NeurIPS.
- Wang, Y., et al. (2024). Unit-DSR. ICASSP.
- El Hajal, K., et al. (2025). RnV: Unsupervised rhythm and voice conversion of
  dysarthric to healthy speech for ASR. ICASSP Workshop.
- Chen, X., et al. (2025). Diff-DSR. Interspeech.
- Hernandez, A., et al. (2022). Cross-lingual dysarthria severity classification.
  APSIPA.
- Kent, R. D., & Kim, Y. (2003). Acoustic analysis of speech.
