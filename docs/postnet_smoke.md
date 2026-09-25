# Frozen-mapper residual postnet experiment

Entry point: `python -m src.training.train_postnet`. This is independent of `joint_finetune.py` and does not change MapperModel or any input checkpoint. No commit or push is part of this experiment.

The module in `src/models/postnet.py` returns a correction shaped `(B,80,T)`. Its `refine()` method computes `predicted_mel + postnet(predicted_mel)`. The five Conv1D layers use kernel 5, same-length padding, channels 80 -> 512 -> 512 -> 512 -> 512 -> 80, BatchNorm after every convolution, and tanh after the first four. There is no final activation, dropout, discriminator or adversarial loss. There are **4,346,016 trainable parameters**. This follows the convolution/normalization layout in [NVIDIA's Tacotron2 postnet](https://github.com/NVIDIA/tacotron2/blob/master/model.py); the final BatchNorm affine scale and bias start at zero so the fresh residual is exactly zero in both train and eval modes. Earlier convolutions start randomly; only final affine parameters have gradients on the first update, and the earlier layers become trainable as the final scale opens.

Frozen models: Thai XLSR layer 9, `results/checkpoints/mapper_layer9.pt`, and UNIVERSAL_V1 `g_02500000`. The universal vocoder is a conservative fixed reference; no new subjective listening comparison established a superior checkpoint. The original raw mapper output is cached unchanged and passed to the postnet. The existing clamp is applied only to the final vocoder input, so zero residual reproduces the original reconstruction pipeline exactly.

The first objective is **L1 mel only**, Adam LR 1e-4, global postnet gradient norm clip 1.0. A waveform/multi-resolution STFT loss is deferred so that L1 and intelligibility can be examined without another loss tradeoff. Ground-truth mel comes from the existing `compute_mel()` (80 channels, HiFi-GAN defaults). The mapper always runs at distorted/inference duration. Because clean and distorted duration differ, only the refined-mel loss view is linearly resized to the clean target length. The clean target and inference mel are unchanged. This handles global duration only, not segment-level tempo/phonetic alignment, and should not be mistaken for a solution to that remaining issue.

Only the 147 TRAIN utterances and the first three sorted VAL utterances are encoded. TEST audio is never read. Frozen encoder and mapper outputs are computed once under no-grad, cached inside the new run, and those two models are then freed. Training uses one full utterance per update without padding. This keeps train and validation mapper input lengths consistent. The frozen vocoder is used only for validation. Every evaluation scores actual saved PCM_16 reconstructions against saved clean PCM references via the existing `load_audio_16k`, `compute_stoi` and `compute_pesq` utilities. The zero-residual step-0 scores are the relevant baseline, not the separate TEST-set 0.180 result.

Local CPU was chosen because preprocessing is a single pass and optimizer updates involve only the postnet. The 500-step smoke validates every 100 steps on the same three VAL IDs and also reports L1 on a fixed 16-utterance TRAIN monitoring subset. A random per-step training loss is not directly comparable across steps because utterances differ. Checkpoints are written every 100 steps to a new `results/checkpoints/postnet/smoke500_cpu_...` folder, with postnet/optimizer/RNG state and frozen-input provenance. Frozen encoder/mapper states are hashed before and after caching; vocoder state is checked during training, and source/checkpoint file hashes are verified at completion.

Launch independently of this CLI session:

```sh
bash scripts/run_postnet_smoke_detached.sh
```

The launcher uses nohup + disown, creates a separate OS session, redirects stdin/stdout/stderr, and verifies a worker startup marker before returning. On macOS, caffeinate prevents idle system sleep while Python runs. The script does not request automatic sleep. It prints the PID, output folder and log path; `<log>.exit` contains the final process exit code. For a direct Colab run with a fresh output folder:

```sh
python -m src.training.train_postnet --device cuda --max-steps 500 \
  --validation-interval 100 --checkpoint-interval 100 --val-items 3 \
  --output-dir results/checkpoints/postnet/NEW_smoke500_cuda
```

Implementation contracts: `MPLCONFIGDIR=/tmp/thai-dsr-mpl .venv/bin/python -m unittest tests.test_postnet -v`. The four unit tests check exact identity, residual shape, gradient flow into earlier convolution layers after opening the final scale, frozen-model immutability, and differing-length L1 without input mutation. They are code checks, not speech-quality evidence. Actual smoke results are recorded separately below when the detached run finishes.

## Completed real smoke results

Run: `results/checkpoints/postnet/smoke500_cpu_20260925_165833_35665`, 500 updates, exit code **0**. Log: `results/logs/postnet_smoke500_cpu_20260925_165833_35665.log`. The first launcher attempt did not persist; the final launcher adds a separate OS session and startup acknowledgement. The completed run is the one named here.

| Step | Fixed TRAIN L1 (16 items) | VAL L1 (3 items) | VAL STOI | VAL PESQ |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.938763 | 1.244862 | 0.264994 | 1.120857 |
| 100 | 0.938324 | 1.244453 | 0.265288 | 1.120644 |
| 200 | 0.937912 | 1.243963 | 0.265034 | 1.119371 |
| 300 | 0.936071 | 1.243050 | 0.264789 | 1.117736 |
| 400 | 0.936188 | 1.243250 | 0.264843 | 1.124024 |
| 500 | 0.934768 | 1.243544 | 0.265539 | 1.121324 |

Fixed TRAIN L1 fell **0.426%**. Randomly sampled per-step L1 averages were 0.94608 (first 50) and 0.96437 (last 50); these windows contain different utterances and should not be used to claim that comparable training loss fell. The fixed-subset comparison above is the controlled loss trend.

All 500 loss/gradient records were finite. Raw postnet gradient norms ranged **0.02936–0.08724**, below the clip threshold throughout. Average optimizer-update duration was **0.02637 s**, about 37.9 steps/sec on CPU; this excludes initial cache preparation, validation and checkpoint writes. Final mean absolute mel correction was 0.02017. Encoder/mapper state hashes matched before/after cache preparation, vocoder state stayed identical through validation, and original mapper/vocoder checkpoints plus MapperModel/joint trainer source hashes remained identical at completion. Only postnet parameters were passed to Adam. Eighteen real generated validation WAVs are PCM_16.

The mean final STOI delta is **+0.000546** and PESQ delta **+0.000468**. Per-utterance STOI deltas are -0.000926 (`002_ขอน้ำหนึ่งแก้วได้ไหม`), -0.002147 (`cv_0001`), and +0.004710 (`cv_0007`): **two of three regressed**, with one improvement raising the mean. Some intermediate mean scores also fell below baseline. This is a tiny mixed signal, not convincing evidence that postnet has solved intelligibility or oversmoothing.

Decision: the implementation is numerically stable and fast enough for further controlled experiments, but **a long full training run is not justified yet**. A bounded follow-up on more validation utterances/seeds would be more informative than blindly extending training. No further run was launched and no files were committed or pushed. These are VAL results with the universal vocoder, so their 0.264994 baseline must not be compared directly to the old 0.180 TEST result or the Approach-A validation baseline.

Artifacts in the run folder: `postnet_00000500.pt` (and steps 100/200/300/400), `losses.jsonl`, `train_monitor.jsonl`, `validation.jsonl`, `summary.json`, `frozen_audit.json`, `learning_curves.png`, cached raw mapper predictions/clean target mels, clean references, and actual generated validation audio.

## Resume with a new learning rate (prepared, not executed)

`train_postnet.py` now accepts `--resume PATH`. It restores postnet weights/BatchNorm buffers, Adam moments and counters, and CPU/sampling RNG state. The CLI `--lr` is applied **after** restoring Adam, so `--lr 3e-4` really replaces the saved 1e-4. The existing defaults remain max-steps 500, LR 1e-4, and validation interval 100. `--max-steps 3000` resumes a step-500 snapshot for 2500 additional updates. Output must be a new run subdirectory; prior checkpoints are not overwritten.

Before training resumes, the new run rebuilds its frozen-output cache and remeasures both the zero-residual frozen baseline (evaluation step 0) and restored model (evaluation step 500). Training then starts at step 501. All subsequent validation deltas continue to reference the frozen baseline, not the step-500 model. Manifest/split hashes, training IDs, postnet architecture/source and frozen mapper/vocoder checkpoint hashes must match the saved experiment. Encoder/mapper/vocoder stay frozen. Resume support was reviewed in source only at the user's request; no test or training was executed to verify this change.

Recommended bounded next attempt: total 3000 steps, LR 3e-4, validation every 100, checkpoint every 500. This increases LR threefold while avoiding the larger initial jump to 5e-4. Monitor `losses.jsonl` (every update), `train_monitor.jsonl` (fixed TRAIN subset), and `validation.jsonl` (real STOI/PESQ); a lower L1 alone is not a success criterion. No run has been launched for this follow-up.
