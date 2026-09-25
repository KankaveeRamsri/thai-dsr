# Spectral auxiliary input smoke

Implementation: `src/models/spectral_aux_mapper.py`, `src/training/train_mapper_spectral_aux.py`, and `scripts/run_spectral_aux_mapper_smoke_detached.sh`.

Fresh MapperModel, seed 42. Input changes from 1024 to 1104; the original two-layer bidirectional LSTM (512 hidden units, dropout 0.1), LayerNorm and 256-unit projection to 80 bins are unchanged. There is no weighted layer sum, postnet, learned input scaling, or extra loss. The original mapper checkpoint supplies configuration and a separately evaluated baseline only; its weights never initialize the experimental model.

The source log-mel is extracted from the exact distorted file used by the frozen encoder cache. Source and clean target use the same helper/configuration: mono, resampled to 22050 Hz, FFT/window 1024, hop 256, 80 bins, fmin 0/fmax 8000, magnitude spectrum, reflect padding, non-centered Hann STFT, natural log with floor 1e-5. Features are cached without gradients. The existing content-hashed FP16 encoder cache is reused, but only hidden_states[9] enters this experiment; the other 24 representations do not participate in training. The encoder and universal vocoder stay frozen.

The original mapper operates on the mel grid (~86.13 Hz), not the native wav2vec2 grid (~50 Hz). Each unpadded layer-9 sequence and distorted mel sequence is linearly interpolated with align_corners=False to a common frame count before concatenation. Training/full validation L1 retain the existing clean-target frame count. Audio inference uses the distorted frame count and requires no clean target. This preserves the earlier global-duration convention; it does not fix local phonetic timing mismatch.

Training reuses Adam LR 0.001, batch size 4, seed 42 and epoch-level ReduceLROnPlateau configuration from the baseline checkpoint. As requested, global gradient clipping is 1.0, with raw norm and clipping activation logged at every step. This is an optimization difference versus the unclipped weighted-layer run and must be acknowledged when interpreting causal evidence.

The exact weighted-layer experiment's eight fixed validation utterances are reused for STOI/PESQ, through the same saved PCM16 audio, frozen UNIVERSAL_V1 vocoder and metric functions. Full validation mel L1 uses the same 32 rows and batch aggregation. Baseline plus steps 0, 100, 200, 300 receive audio metrics. Epoch boundaries additionally log mel L1. PESQ failures are explicit with valid counts. The earlier Postnet smoke used three validation utterances, so its absolute metrics are not directly comparable. No test audio is read.

Two targeted tests verify architecture shapes, padding isolation, gradients through spectral inputs and their mapper weights, and exact source/target extraction equivalence. Command: `MPLCONFIGDIR=/tmp/thai-dsr-mpl .venv/bin/python -m unittest tests.test_spectral_aux_mapper -v`.

The launcher is fixed to 300 updates, uses nohup + Python os.setsid() + bash disown, redirects all standard streams and waits for a worker startup marker. On macOS caffeinate inhibits idle sleep. The trainer rejects runs exceeding 500 updates. No longer run is launched.

Completed run: `results/checkpoints/spectral_aux_mapper/smoke300_20260925_220350_64148`.
Worker PID: 64176; independently checked session ID = process group ID = 64176 after the launcher returned.
Log: `results/logs/spectral_aux_smoke300_20260925_220350_64148.log`.
The earlier launch PID 64001 exited before worker startup and performed no training; it is not the reported run.

## Completed smoke results

Completed 300 updates, exit code 0. No subsequent run was launched. Detachment was verified while running (worker PID/session/process-group 64176); that worker has now completed.

Training L1 mean: first 50 updates **2.071230**, last 50 **1.327769**. These windows use different batches and the first includes initialization. All losses/gradients were finite; clipping activated on 295/300 updates (maximum raw norm 6.260087). Mean update duration 4.528 seconds, excluding validation/preparation. Original checkpoint/config/split hashes remained unchanged; vocoder frozen audit passed.

Every logged validation point follows. L1 covers 32 utterances; STOI/PESQ cover the fixed eight. A dash means audio metrics were not scheduled at that epoch boundary. All audio points had 8/8 valid PESQ results.

| Step | VAL mel-L1 | STOI | PESQ |
| ---: | ---: | ---: | ---: |
| Trained layer-9 baseline | 1.288527 | 0.261529 | 1.094354 |
| 0 | 6.812374 | 0.339782 | 1.264579 |
| 37 | 1.403217 | — | — |
| 74 | 1.371040 | — | — |
| 100 | 1.422799 | 0.348162 | 1.084902 |
| 111 | 1.385404 | — | — |
| 148 | 1.323146 | — | — |
| 185 | 1.393842 | — | — |
| 200 | 1.331352 | 0.309145 | 1.079212 |
| 222 | 1.306476 | — | — |
| 259 | 1.332974 | — | — |
| 296 | 1.407030 | — | — |
| 300 | 1.594168 | 0.277464 | 1.074935 |

Weighted-layer comparison on identical eight IDs and full validation L1:

| Step | Weighted STOI | Weighted PESQ | Weighted mel-L1 |
| ---: | ---: | ---: | ---: |
| 0 | 0.335700 | 1.125846 | 6.803890 |
| 100 | 0.360574 | 1.086571 | 1.435603 |
| 200 | 0.361683 | 1.082542 | 1.335541 |
| 300 | 0.304264 | 1.075623 | 1.450147 |

Final per-utterance changes from the trained layer-9 baseline:

| Utterance | Δ STOI | Δ PESQ |
| --- | ---: | ---: |
| 002_ขอน้ำหนึ่งแก้วได้ไหม | -0.025582 | -0.013143 |
| cv_0014 | -0.009592 | -0.014760 |
| cv_0039 | +0.027597 | -0.002698 |
| cv_0076 | +0.098028 | -0.013191 |
| cv_0082 | -0.023261 | -0.039035 |
| cv_0127 | -0.032216 | -0.008386 |
| cv_0165 | +0.153785 | -0.002581 |
| cv_0190 | -0.061284 | -0.061552 |

Assessment: no convincing positive signal. STOI peaks at 100 then declines at 200 and 300, finishing below the untrained model's score. PESQ declines at every audio evaluation. Final L1 is worse than the original layer-9 baseline and than this experiment's earlier values. Only 3/8 final utterances improve STOI over baseline; all eight regress PESQ. The trained layer-9 baseline itself scores below random-initialization STOI here, underscoring that these metrics on eight timing-mismatched samples cannot establish intelligible speech alone. No listening judgment is claimed.

The weighted sum similarly showed a temporary STOI peak (0.360574/0.361683 at 100/200) before falling to 0.304264 at 300. Spectral input is below that run at all three trained audio checkpoints, including final PESQ and L1. The earlier frozen-mapper Postnet smoke produced only +0.000546 STOI and +0.000468 PESQ on three validation utterances, with two of three STOI values regressing. The stabilized joint smoke on three utterances and the Approach-A vocoder reduced STOI from 0.281058 to 0.260298 while increasing PESQ from 1.107004 to 1.123442. User-reported earlier joint TEST scores also degraded (0.180 to 0.146/0.139); those are different subsets and must not be compared numerically to this eight-item VAL result.

This smoke shows successful optimization of a fresh model, not a solution to oversmoothing/unintelligibility. The validation sample is small, one seed is insufficient, local clean/distorted timing mismatch remains, and requested clipping differs from the weighted run (it activated on 295 updates). There is also no matched fresh layer-9-only 300-step clipped control. Consequently it cannot establish the causal effect of spectral input. Nothing here justifies automatically launching a longer run. No longer run was started.

Artifacts: `summary.json`, `validation.jsonl` (per-utterance metrics), `epochs.jsonl`, `losses.jsonl`, `baseline.jsonl`, `run.json`, `cache_audit.json`, `best.pt`, `final.pt`, and 40 saved validation WAVs in the run folder. The `best.pt` epoch checkpoint is step 222; `final.pt` is step 300. The final checkpoint reloads through `SpectralAuxMapper(checkpoint['model_config'])` and `load_state_dict(checkpoint['model'])`.
