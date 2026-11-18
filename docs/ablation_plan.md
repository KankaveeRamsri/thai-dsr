# Fixed-layer ablation: which wav2vec2 layer feeds the mapper (pre-registered)

Status: **pre-registration.** This file is committed before any ablation model is
trained. Anything changed after that commit goes in "Deviations" at the end,
with the date and the reason. The commit that adds this file is the
registration record (`git log --follow docs/ablation_plan.md`).

## Background

- Tempo alignment (option A, `--alignment tempo`) is the standard for this and
  every future run. On the full CV 27.0 test set, the layer-9 tempo mapper beat
  the linear-stretch mapper on every main metric for all 48 test speakers.
  - stoi_melwarp: 0.612 for tempo vs 0.341 (linear, oracle length) and 0.385
    (linear, deployable length).
  - Results are in `<data root>/eval/test_full/summary/`.
- Listening and analysis of the layer-9 tempo mapper show it is intelligible and
  denoised, but harmonics and pitch are smeared. Symptoms: low HNR, F0 errors,
  and low speaker similarity (spk_sim_other 0.203, about the 0.200 level of
  clean clips from *different* speakers).
- This ablation asks how much of that comes from the choice of layer 9. It does
  so by training the identical mapper on other frozen layers.

## Layers and the index convention

Layers: **3, 6, 9, 12, 18, 24.**

A layer number is the index into `outputs.hidden_states` of
`transformers.Wav2Vec2Model(...)(input_values, output_hidden_states=True)` for
`airesearch/wav2vec2-large-xlsr-53-th`. That model has 24 transformer blocks and
`do_stable_layer_norm = true` (pre-LayerNorm blocks). `hidden_states` has 25
entries:

| index | what it is |
|---|---|
| 0 | CNN feature encoder → feature projection, plus the convolutional positional embedding. This is the input to block 1. |
| k = 1…23 | output of transformer block k (residual stream, no final LayerNorm) |
| 24 | output of block 24 **after** the encoder's final LayerNorm (= `last_hidden_state`) |

- This is the convention already used for "layer 9":
  - `src/preprocessing/extract_embedding.py` and `Wav2Vec2ContentEncoder` both
    use `hidden_states[layer]`.
  - Files are named `<clip>_<severity>_layer<NN>.npy`.
- Layer 24 is the only one that passes through a LayerNorm, so its scale differs
  from the others (see Confounds).

## What is held fixed

Everything except the layer is identical to the layer-9 tempo run
(`cv27_layer9_tempo`; commit ef6b427, best epoch 18, best val loss 1.1004).

| setting | value |
|---|---|
| alignment | `tempo` (target = clean mel warped onto the distorted timeline with the logged tempo segments; embedding interpolated to the distorted mel frame count) |
| encoder | `airesearch/wav2vec2-large-xlsr-53-th`, frozen, fp32 |
| mapper | BiLSTM, `configs/model.yaml` unchanged (2 × 512, LayerNorm, 256 → 80 projection, dropout 0.1) |
| vocoder | HiFi-GAN UNIVERSAL_V1 (evaluation only) |
| data | CV 27.0 Thai speaker split (`data/manifests/cv27_th_*`): train = 2,383 severe clips from 379 speakers; val = 305 severe clips from 48 speakers; test = 308 clips × mild/moderate/severe, 48 speakers |
| optimisation | Adam, lr 1e-3, weight decay 0, batch size 4, ReduceLROnPlateau (patience 5), **50 epochs**, seed 42 |
| model selection | `best.pt` = lowest validation loss over the 50 epochs |
| hardware | Colab GPU. Recorded per session in `run_info.json`; T4 where possible, as for layer 9. |

- The layer-9 run is reused rather than retrained.
- Training code is unchanged since ef6b427: `git diff ef6b427 -- src/training
  src/models configs data/manifests` is empty.
- The only change in `src/utils/time_warp.py` is new inverse functions.
  `warp_frames` gives bit-identical output (max abs diff 0 on 200 clips).
- Tempo targets come from the same cached `mel_cache/tempo/` archive.

### Embeddings

- Extracted on the Colab GPU from the frozen distorted wavs (`derived/distorted/`, never
  regenerated): 3,612 files, i.e. 2,996 severe plus 308 each of mild and moderate.
- One forward pass per file gives all requested layers. The code path is the
  same one that produced layer 9: `src/preprocessing/extract_embedding.py`, with
  `Wav2Vec2FeatureExtractor` normalisation, fp32, no attention mask.
- Before training, the notebook recomputes layer 9 for a sample of files on the
  GPU and compares it with the existing layer-9 archive. This confirms the index
  convention and preprocessing are the same.

## Metrics

All metrics come from `scripts/evaluate_cv_aligned.py --system mapper --alignment tempo --layer L`.
They are computed on the full test set (924 rows) against the **untouched clean
clip** on the clean timeline. Outputs are moved onto that timeline with the
mel-domain inverse warp (melwarp). Summaries use the speaker-clustered bootstrap
in `scripts/summarize_cv_eval.py`.

| group | metric | direction |
|---|---|---|
| intelligibility (main) | `stoi_melwarp` | higher |
| quality | `pesq_melwarp` | higher |
| pitch / tone | `f0_corr`: Pearson correlation of F0 (semitones) on frames voiced in both | higher |
| | `f0_rmse_st`: RMS F0 error in semitones on frames voiced in both | lower |
| | `vuv_agree`: share of frames with the same voiced/unvoiced decision | higher |
| | `hnr_db`: mean harmonics-to-noise ratio of the output (reported with the clean clip's `hnr_clean_db`) | closer to clean |
| speaker identity | `spk_sim_other`: ECAPA cosine to the speaker's *other* clean clips | higher |
| | `spk_sim_same`: ECAPA cosine to the same clean clip | higher |
| spectral | `mel_l1_clean`: mel L1 on the clean grid | lower |
| secondary | `stoi` / `pesq` with the waveform inverse warp | higher |

- F0 and HNR come from Praat via parselmouth:
  - autocorrelation pitch: 75–500 Hz, 10 ms step
  - cross-correlation harmonicity: 10 ms step, 75 Hz floor, silence threshold 0.1, 1 period per window
- A clip with fewer than 10 frames voiced in both gets NaN for `f0_corr` and
  `f0_rmse_st`; the count of such clips is reported.

## Pre-registered hypotheses

- **H1.** Lower layers (3, 6) preserve F0 and speaker identity better than layer 9:
  - higher `f0_corr`
  - lower `f0_rmse_st`
  - higher `spk_sim_other`

  They may be less intelligible (lower `stoi_melwarp`). This is the expected
  trade-off; it is reported but not tested as a directional claim.
- **H2.** Deeper layers (18, 24) lose more speaker identity than layer 9: lower
  `spk_sim_other`.

Layer 12 has no hypothesis and is reported descriptively. Every other metric
and layer comparison is exploratory.

### Test procedure

- **Unit of analysis.** Paired per-clip differences of layer L minus layer 9,
  pooled over the three severities (924 pairs). Per-severity results are
  descriptive.
- **Test.** Two-sided Wilcoxon signed-rank on the 48 per-speaker mean
  differences (`p_wilcoxon_speaker`), with a 95% speaker-clustered bootstrap CI
  (10,000 resamples, seed 42).
- **Multiple comparisons.** Holm-Bonferroni within each family at α = 0.05:
  - H1: layers {3, 6} × {`f0_corr`, `f0_rmse_st`, `spk_sim_other`}, 6 tests
  - H2: layers {18, 24} × {`spk_sim_other`}, 2 tests
- **Decision rule.** A (layer, metric) claim is *supported* if the
  Holm-adjusted p < 0.05 and the mean difference is in the predicted direction.
  - The same result in the opposite direction is reported as *contradicted*.
  - Anything else is *not supported*.
  - H1 is supported for a layer only if `spk_sim_other` and at least one F0
    metric are supported for it.

## Confounds and limits

- **One seed per layer.** Seed-to-seed variance is unknown, so a small
  difference may be seed noise even if it is statistically significant across
  speakers. If a pre-registered result hinges on a difference smaller than
  0.01 STOI or 0.02 cosine, layer 9 and that layer are re-run with seed 43
  before any claim is made.
- **Input scale.** The mapper has no input normalisation, and the residual-stream
  scale grows with depth (layer 24 is re-normalised). No per-layer tuning is
  done. Differences may therefore partly reflect how well one fixed recipe suits
  each layer's scale.
  - The per-layer mean frame norm on the training set is reported alongside the
    results.
- **Proxy metrics.** STOI is a proxy for intelligibility (no ASR or listening
  test here), and ECAPA similarity is a proxy for speaker identity.
- **Different commits.** The layer-9 run was trained at ef6b427 and the new
  runs at a later commit. This has no effect on training (see above) but is
  recorded in each `run_info.json`.

## Artefacts

- Drive embedding archives: `cv27_embeddings_layer<NN>.tar` (same layout as
  `cv27_embeddings_layer09.tar`), each with a `.sha256` file.
- Drive runs: `runs/cv27_layer<NN>_tempo/` (`best.pt`, `latest.pt`,
  `run_info.json`, `train.log`). The existing layer-9 run stays at
  `runs/cv27_layer9_tempo/`.
- Mac checkpoints: `<data root>/checkpoints/cv27_layer<NN>_tempo/`.
- Mac evaluation: `<data root>/eval/test_full/mapper_tempo_layer<NN>/`.

## Deviations

None yet.
