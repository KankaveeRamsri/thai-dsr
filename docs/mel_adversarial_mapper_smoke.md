# Mel-only adversarial mapper smoke

Implementation: `src/models/mel_discriminator.py`, `src/training/train_mapper_mel_adversarial.py`, `scripts/run_mel_adversarial_mapper_smoke_detached.sh`, `tests/test_mel_discriminator.py`.

The mapper is the unchanged 1024-input layer-9 MapperModel. It loads every tensor from `results/checkpoints/mapper_layer9.pt`; exact state equality is asserted before training. Adam states are fresh. There is no spectral auxiliary input, weighted-layer sum, postnet, waveform loss, or waveform discriminator. Frozen layer-9 embeddings are reused from the content-hashed FP16 cache, with a frozen encoder fallback. No test audio is accessed.

The discriminator is four 2D convolutions: 1→16→32→64→1, kernel 3; the first three use stride 2 and LeakyReLU(0.2), the last produces local patch scores. Padding is 1. No sigmoid, batch normalization, dropout, or vocoder components. Real and generated inputs share the fixed affine scale `(log_mel + 5)/5`, without clipping. True-length utterances are scored separately so batch padding never enters the discriminator or adversarial objective. Patch losses are averaged within each utterance, then across the batch.

Mapper objective: `10 * masked_mel_L1 + lambda(step) * mean((D(predicted_mel)-1)^2)`.
Discriminator objective: `0.5 * mean((D(real_mel)-1)^2 + D(predicted_mel.detach())^2)`.
The discriminator updates first. Its parameters are frozen during the mapper update, while its input derivative remains connected to the mapper. Only mapper parameters enter the mapper optimizer; only discriminator parameters enter its optimizer.

Schedule: lambda is zero through step 50, increases linearly for 100 updates, first positive at step 51 (0.001), reaches 0.1 at step 150 and stays there through step 300. D learns during warmup; mapper receives only the L1 objective during warmup. Adam LR: mapper 1e-5, D 1e-4; betas mapper (0.9,0.999), D (0.5,0.999), weight decay 0. Fixed LR, batch size 4, seed 42. Both global gradient norms are clipped to 1.0; raw and post-clip norms are logged on every update; nonfinite gradients fail explicitly, while small float32 post-norm overshoots are diagnostic only. Clipping activation is not itself evidence of divergence.

The universal HiFi-GAN vocoder remains frozen and is used only inside no-grad validation for waveform synthesis needed by STOI/PESQ. It never participates in any training loss, graph, or optimizer. Source checkpoint/config/split hashes and frozen vocoder state versions are checked at completion.

Validation uses the same eight evenly selected utterances and metric utilities as weighted-layer and spectral-aux experiments, plus the same 32-utterance mel-L1 aggregation. Steps 0/100/200/300 log audio metrics, full-val D loss and generator adversarial loss, lambda and cumulative gradient diagnostics. Epoch boundaries log mel and adversarial validation losses without audio metrics. Saved PCM16 audio is scored through the existing utilities; PESQ failures are explicit. Training/full-val mel uses clean-target frame count and audio inference uses distorted frame count, preserving previous alignment conventions. Local timing mismatches remain.

Six tests cover schedule boundaries, zero adversarial gradient in warmup, padding isolation, discriminator size, gradient routing, rescaling large gradients, diagnostic reduction roundoff, and rejection of nonfinite gradients. Test command: `MPLCONFIGDIR=/tmp/thai-dsr-mpl .venv/bin/python -m unittest tests.test_mel_discriminator -v`.

The launcher runs exactly 300 updates with nohup, os.setsid(), disown, redirected standard streams, startup acknowledgement, and caffeinate. Trainer refuses more than 500 updates. No longer run is launched automatically.

Run: `results/checkpoints/mel_adversarial_mapper/smoke300_20260925_223941_78539`.
Log: `results/logs/mel_adversarial_smoke300_20260925_223941_78539.log`.
Training PID 78546; detached worker/session leader/process group 78543. Independently verified after launcher exit. The log's PROCESS record also confirms the trainer belongs to that separate session. `<log>.exit` records eventual exit status.

## Clipping failure and fresh rerun

The original run completed three updates and failed in the mapper clipping check at step 4, during zero-adversarial-weight warmup. The function already called `torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)`, then incorrectly treated a float32 post-clipping norm above 1.00001 as fatal. The fresh seed-42 rerun reproduced steps 1–3 exactly and measured step 4 mapper raw norm **26.155202865600586**, post-clipping norm **1.000014305114746**. D raw norm was **1.3750332593917847**, post-clipping **0.9999993443489075**. This was a numerical-tolerance assertion failure, not evidence of divergence. The mapper objective at this step was exactly `10 * masked_mel_L1`.

The fix retains standard PyTorch rescaling and its nonfinite hard failure, removes the finite post-norm assertion, and records both components' norms, step, and adversarial weight in `gradients.jsonl` before optimizer updates. A diagnostic flag records whether the old assertion would have failed. Nonfinite clipping errors are logged before re-raising.

Fresh run: `results/checkpoints/mel_adversarial_mapper/smoke300_20260925_224308_80333`.
Log: `results/logs/mel_adversarial_smoke300_20260925_224308_80333.log`.
Worker 80337; trainer 80340. Started from the original layer-9 checkpoint with fresh optimizers, identical settings and seed, using nohup + os.setsid() + disown. Limited to the original 300 updates.
